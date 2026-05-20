"""
plot_sim_results.py
-------------------
Simulate all train/val/test trajectories with best_log_params from a .pt
checkpoint and save comparison plots (model vs measured, error, forces).

Each figure — 3 rows x 3 cols:
  Row 1  model (blue) vs measured (gray --)   X1 | X2 | Y   [m]
  Row 2  error = model - measured              X1 | X2 | Y   [m]
  Row 3  plant force input                    FX1 | FX2 | FY [N]

Run as:
    conda run -n GraduationProject python -m lpv_lfr_baseline.plots.plot_sim_results <path_to_pt>
"""

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch

from lpv_lfr_baseline.blocks.lfr_param_block import ParameterizedLFRBlock, _build_matrices
from lpv_lfr_baseline.core.lfr_matrices import build_G_matrix
from lpv_lfr_baseline.core.lfr_simulate import simulate
from lpv_lfr_baseline.core.physics import build_poly_constants
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

# ── Style (MATLAB-matched) ────────────────────────────────────────────────────
plt.rcParams.update({
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.linewidth':    0.4,
    'grid.alpha':        0.5,
    'lines.linewidth':   0.8,
    'font.size':         9,
})

_C_MEAS  = '#999999'   # measured  — gray dashed
_C_MODEL = '#0072bd'   # model     — MATLAB blue
_C_ERR   = '#000000'   # error     — black
_C_FORCE = '#d62728'   # force     — red


# ── Simulation helpers ────────────────────────────────────────────────────────

def _infer_dataset(pt_path):
    dirname = os.path.basename(os.path.dirname(os.path.abspath(pt_path)))
    return dirname.split('param_recovery_', 1)[1]


def _build_sim_params(block):
    params = block._recover_params()
    kb1, kb2, cg1, cg2, cy, cb1, cb2, mh, m1, m2, mb, Jb, Jh, d = params
    params_10 = torch.stack([kb1+kb2, cg1, cg2, cy, cb1+cb2, mh, m1, m2, mb, Jb+Jh])
    _, M1, M2, K, C = _build_matrices(params_10, block._Lb, d)
    alpha, beta, gamma, N0, N1, N2 = build_poly_constants(m1, m2, mb, mh, Jb, Jh, block._Lb, d)
    G = build_G_matrix(N0, mh * (alpha * gamma - beta**2), M1, M2, K, C)
    return G, K, C, mh, alpha, beta, gamma, N0, N1, N2


def _simulate(block, x0, u, ts_tensor):
    with torch.no_grad():
        G, K, C, mh, alpha, beta, gamma, N0, N1, N2 = _build_sim_params(block)
        result = simulate(x0, u, G, K, C, mh, alpha, beta, gamma, N0, N1, N2,
                          block._P, ts_tensor, bptt_mode='full', return_latents=False)
    return result.Y[0].cpu().numpy()   # (T, 3)


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _human_title(traj_id, traj_file, split):
    suffix = traj_file.replace('.mat', '')[len(traj_id) + 1:].replace('_', ' ')
    return f'{traj_id} \u2014 {suffix}  [{split}]'


def _fix_ticks(ax):
    fmt = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    ax.yaxis.set_major_formatter(fmt)


def _plot_trajectory(traj_id, traj_file, split, t, q1_meas, q1_model, u, fig_dir):
    err = q1_model - q1_meas

    fig, axes = plt.subplots(
        3, 3, figsize=(14, 9), sharex=True,
        gridspec_kw={'hspace': 0.10, 'wspace': 0.38},
    )
    fig.suptitle(_human_title(traj_id, traj_file, split),
                 fontsize=11, fontweight='bold', y=0.995)

    pos_labels   = ['X1  [m]',       'X2  [m]',       'Y  [m]']
    err_labels   = ['\u0394X1  [m]', '\u0394X2  [m]', '\u0394Y  [m]']
    force_labels = ['FX1  [N]',      'FX2  [N]',      'FY  [N]']

    for col in range(3):
        # Row 0 — position: model vs measured
        ax = axes[0, col]
        ax.plot(t, q1_meas[:, col],  color=_C_MEAS,  ls='--', label='Measured')
        ax.plot(t, q1_model[:, col], color=_C_MODEL,           label='Model')
        ax.set_ylabel(pos_labels[col])
        _fix_ticks(ax)
        if col == 0:
            ax.legend(fontsize=8, framealpha=0.7, loc='best')

        # Row 1 — error
        ax = axes[1, col]
        ax.plot(t, err[:, col], color=_C_ERR)
        ax.axhline(0, color='#bbbbbb', lw=0.6, ls='--', zorder=0)
        ax.set_ylabel(err_labels[col])
        _fix_ticks(ax)

        # Row 2 — forces
        ax = axes[2, col]
        ax.plot(t, u[:, col], color=_C_FORCE)
        ax.set_ylabel(force_labels[col])
        ax.set_xlabel('Time  [s]')
        _fix_ticks(ax)

    out_dir = os.path.join(fig_dir, split)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{traj_id}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'  Saved: {path}')
    plt.close(fig)


# ── Set this to your .pt checkpoint ──────────────────────────────────────────
PT_FILE = r'C:\Users\20203253\OneDrive - TU Eindhoven\Graduation Project\Baseline FP model\Baseline-LPV-Augmentation\simulations\param_recovery_base_extended\lfr_param_recovery_base_extended_T1_T2_T3_T4_T5_T6_T7_T8_e1500_63535.pt'


# ── Main ──────────────────────────────────────────────────────────────────────

def main(pt_path):
    pt_path  = os.path.abspath(pt_path)
    save_dir = os.path.dirname(pt_path)

    pt      = torch.load(pt_path, weights_only=False)
    dtype   = torch.float64
    run_id  = str(pt['run_id'])
    dataset = _infer_dataset(pt_path)
    ds      = _DATASETS[dataset]
    print(f'Dataset : {dataset}  |  Run : {run_id}\n')

    # Precompute cache — guaranteed to exist after training
    traj_tag   = '_'.join(pt['active_traj_ids'])
    cache_path = os.path.join(save_dir, f'precomputed_{traj_tag}_float64.pt')
    cache  = torch.load(cache_path, weights_only=False)
    trajs  = cache['trajs']
    ts_eff = float(cache['metadata']['ts_eff'])
    D      = int(cache['metadata']['D'])

    # Build model with best parameters
    block = ParameterizedLFRBlock(RMSE_baseline=pt['rmse_baseline_normalized']).to(dtype=dtype)
    with torch.no_grad():
        block.log_params.copy_(pt['best_log_params'].to(dtype=dtype))
    ts_tensor = torch.tensor(ts_eff, dtype=dtype)

    # Val / test trajectories
    val_ids  = [e['id'] for e in pt['eval_val_entries']]
    test_ids = [e['id'] for e in pt['eval_test_entries']]
    val_trajs  = load_eval_trajs([{'id': i, 'file': _ALL_SPECS[i]} for i in val_ids],
                                  ds['val_test_dir'], D, dtype) if val_ids  else []
    test_trajs = load_eval_trajs([{'id': i, 'file': _ALL_SPECS[i]} for i in test_ids],
                                  ds['val_test_dir'], D, dtype) if test_ids else []

    fig_dir = os.path.join(os.path.dirname(__file__), 'figures', run_id)

    for split, traj_list in [('train', trajs), ('val', val_trajs), ('test', test_trajs)]:
        for traj in traj_list:
            x0       = traj['state_traj'][:1].to(dtype=dtype)
            u_torch  = traj['u'].to(dtype=dtype)
            q1_meas  = traj['q1'].cpu().numpy()
            q1_model = _simulate(block, x0, u_torch, ts_tensor)
            t        = np.arange(q1_meas.shape[0]) * ts_eff
            u_np     = u_torch[0].cpu().numpy()

            rmse = np.sqrt(((q1_model - q1_meas) ** 2).mean())
            print(f'  {traj["id"]:6s} [{split}]  RMSE = {rmse:.4e} m')

            _plot_trajectory(traj['id'], traj['file'], split,
                             t, q1_meas, q1_model, u_np, fig_dir)

    print('\nDone.')


if __name__ == '__main__':
    pt = sys.argv[1] if len(sys.argv) == 2 else PT_FILE
    main(pt)
