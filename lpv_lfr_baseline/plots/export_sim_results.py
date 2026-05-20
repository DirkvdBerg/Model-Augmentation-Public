"""
export_sim_results.py
---------------------
Load a .pt checkpoint, re-simulate all train/val/test trajectories with
best_log_params, and write a .mat file for MATLAB plotting.

Run as:
    conda run -n GraduationProject python -m lpv_lfr_baseline.plots.export_sim_results <path_to_pt>
"""

import os
import sys

import numpy as np
import torch
from scipy.io import savemat

from lpv_lfr_baseline.blocks.lfr_param_block import (
    ParameterizedLFRBlock, _build_matrices,
)
from lpv_lfr_baseline.core.lfr_matrices import build_G_matrix
from lpv_lfr_baseline.core.lfr_simulate import simulate
from lpv_lfr_baseline.core.physics import P as _P, ts as _ts, build_poly_constants
from lpv_lfr_baseline.scripts.precompute import load_eval_trajs

# ── Dataset config (mirrors train_param_recovery.py) ─────────────────────────
_BASE = os.path.join(os.path.dirname(__file__), '..', '..')

_ALL_SPECS = {
    'T1': 'T1_Y_sweep_conservative.mat',
    'T2': 'T2_X_sym_Y030.mat',
    'T3': 'T3_X_sym_Y000.mat',
    'T4': 'T4_X_antisym_Y020.mat',
    'T5': 'T5_X_sym_Y_sweep.mat',
    'T6': 'T6_Y_sweep_aggressive.mat',
    'T7': 'T7_X_antisym_Y_sweep.mat',
    'T8': 'T8_X_sym_anti_Y_sweep.mat',
    'V1': 'V1_X_sym_Y_mid_sweep.mat',
    'E1': 'E1_X_sym_anti_Y_low_offset_sweep.mat',
}

_MULTISINE_VAL_TEST = os.path.join(_BASE, 'Matlab-output', 'parameter-recovery-multisine-val-test')
_IDENTIFICATION_DIR = os.path.join(_BASE, 'Matlab-output', 'identification-trajectories')

_DATASETS = {
    'base': dict(
        traj_dir     = os.path.join(_BASE, 'Matlab-output', 'parameter-recovery'),
        val_test_dir = _MULTISINE_VAL_TEST,
    ),
    'multisine': dict(
        traj_dir     = os.path.join(_BASE, 'Matlab-output', 'parameter-recovery-multisine'),
        val_test_dir = _MULTISINE_VAL_TEST,
    ),
    'ref_injection': dict(
        traj_dir     = os.path.join(_BASE, 'Matlab-output', 'parameter-recovery-ref-injection'),
        val_test_dir = _MULTISINE_VAL_TEST,
    ),
    'identification': dict(
        traj_dir     = _IDENTIFICATION_DIR,
        val_test_dir = _IDENTIFICATION_DIR,
    ),
    'base_extended': dict(
        traj_dir     = os.path.join(_BASE, 'Matlab-output', 'identification-trajectories-no-multisine'),
        val_test_dir = os.path.join(_BASE, 'Matlab-output', 'identification-trajectories-no-multisine'),
    ),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _infer_dataset(pt_path):
    """Extract dataset name from .pt parent directory name.
    e.g. '.../simulations/param_recovery_base_extended/...' → 'base_extended'
    """
    dirname = os.path.basename(os.path.dirname(os.path.abspath(pt_path)))
    return dirname.split('param_recovery_', 1)[1]


def _build_sim_params(block):
    params = block._recover_params()
    kb1, kb2, cg1, cg2, cy, cb1, cb2, mh, m1, m2, mb, Jb, Jh, d = params
    params_10 = torch.stack([kb1 + kb2, cg1, cg2, cy, cb1 + cb2, mh, m1, m2, mb, Jb + Jh])
    _, M1, M2, K, C = _build_matrices(params_10, block._Lb, d)
    alpha, beta, gamma, N0, N1, N2 = build_poly_constants(m1, m2, mb, mh, Jb, Jh, block._Lb, d)
    d0 = mh * (alpha * gamma - beta ** 2)
    G  = build_G_matrix(N0, d0, M1, M2, K, C)
    return G, K, C, mh, alpha, beta, gamma, N0, N1, N2


def _simulate(block, x0, u, ts_tensor):
    """Full-trajectory simulation, no gradient."""
    with torch.no_grad():
        G, K, C, mh, alpha, beta, gamma, N0, N1, N2 = _build_sim_params(block)
        result = simulate(
            x0, u, G, K, C, mh, alpha, beta, gamma, N0, N1, N2,
            block._P, ts_tensor, bptt_mode='full', return_latents=False,
        )
    return result.Y[0].cpu().numpy()   # (T, 3)


def _specs_from_ids(ids):
    return [{'id': i, 'file': _ALL_SPECS[i]} for i in ids]


# ── Main ──────────────────────────────────────────────────────────────────────

def main(pt_path):
    pt_path  = os.path.abspath(pt_path)
    save_dir = os.path.dirname(pt_path)

    # Load checkpoint
    pt      = torch.load(pt_path, weights_only=False)
    dtype   = torch.float64
    run_id  = str(pt['run_id'])
    dataset = _infer_dataset(pt_path)
    ds      = _DATASETS[dataset]
    print(f'Dataset : {dataset}')
    print(f'Run ID  : {run_id}')

    # Load precompute cache (guaranteed to exist after training) for decimated
    # trajectory data, ts_eff, and D — avoids re-running any computation.
    traj_tag   = '_'.join(pt['active_traj_ids'])
    cache_path = os.path.join(save_dir, f'precomputed_{traj_tag}_float64.pt')
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f'Precompute cache not found: {cache_path}')
    cache  = torch.load(cache_path, weights_only=False)
    trajs  = cache['trajs']           # list of dicts with decimated u, q1, state_traj
    ts_eff = float(cache['metadata']['ts_eff'])
    D      = int(cache['metadata']['D'])
    print(f'ts_eff={ts_eff:.6f} s  D={D}')

    # Build model with best parameters
    block = ParameterizedLFRBlock(RMSE_baseline=pt['rmse_baseline_normalized']).to(dtype=dtype)
    with torch.no_grad():
        block.log_params.copy_(pt['best_log_params'].to(dtype=dtype))

    ts_tensor = torch.tensor(ts_eff, dtype=dtype)

    # Val / test trajectory IDs (from eval entries stored in .pt)
    val_ids  = [e['id'] for e in pt['eval_val_entries']]
    test_ids = [e['id'] for e in pt['eval_test_entries']]
    val_trajs  = load_eval_trajs(_specs_from_ids(val_ids),  ds['val_test_dir'], D, dtype) if val_ids  else []
    test_trajs = load_eval_trajs(_specs_from_ids(test_ids), ds['val_test_dir'], D, dtype) if test_ids else []

    # Simulate all trajectories
    all_results = []
    for split, traj_list in [('train', trajs), ('val', val_trajs), ('test', test_trajs)]:
        for traj in traj_list:
            x0 = traj['state_traj'][:1].to(dtype=dtype)   # (1, 6)
            u  = traj['u'].to(dtype=dtype)                 # (1, T, 3)
            q1 = traj['q1'].to(dtype=dtype).cpu().numpy()  # (T, 3)
            T  = q1.shape[0]

            q1_model = _simulate(block, x0, u, ts_tensor)  # (T, 3)
            t        = np.arange(T, dtype=np.float64) * ts_eff

            print(f'  {traj["id"]:6s} [{split}]  RMSE = '
                  f'{np.sqrt(((q1_model - q1)**2).mean()):.4e} m')

            all_results.append({
                'id':           traj['id'],
                'split':        split,
                't':            t.reshape(-1, 1),
                'q1_measured':  q1,
                'q1_model':     q1_model,
                'u':            u[0].cpu().numpy(),   # (T, 3)
            })

    # Pack as numpy structured array → MATLAB struct array: trajs(i).id etc.
    dt  = np.dtype([('id', object), ('split', object), ('t', object),
                    ('q1_measured', object), ('q1_model', object), ('u', object)])
    arr = np.empty((1, len(all_results)), dtype=dt)
    for i, r in enumerate(all_results):
        arr[0, i]['id']          = r['id']
        arr[0, i]['split']       = r['split']
        arr[0, i]['t']           = r['t']
        arr[0, i]['q1_measured'] = r['q1_measured']
        arr[0, i]['q1_model']    = r['q1_model']
        arr[0, i]['u']           = r['u']

    out_path = os.path.join(save_dir, f'sim_results_{run_id}.mat')
    savemat(out_path, {'trajs': arr, 'run_id': run_id, 'dataset': dataset})
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python -m lpv_lfr_baseline.plots.export_sim_results <path_to_pt>')
        sys.exit(1)
    main(sys.argv[1])
