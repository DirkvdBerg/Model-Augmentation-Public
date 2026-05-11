"""
evaluate_checkpoint.py
----------------------
Load a checkpoint or final-save .pt, print parameter tables, and plot
predicted vs measured output on all training trajectories.

Configure CHECKPOINT_PATH and DATASET at the top, then run:
    conda run -n GraduationProject python -m lpv_lfr_baseline.scripts.evaluate_checkpoint
"""

import os
import torch
import matplotlib.pyplot as plt

from lpv_lfr_baseline.blocks.lfr_param_block import ParameterizedLFRBlock, _build_matrices
from lpv_lfr_baseline.core.lfr_matrices import build_G_matrix
from lpv_lfr_baseline.core.lfr_simulate import simulate
from lpv_lfr_baseline.core.physics import P as _P, ts as _ts, build_poly_constants
from lpv_lfr_baseline.scripts.precompute import load_eval_trajs

# ── Configure ─────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = r'C:\Users\20203253\OneDrive - TU Eindhoven\Graduation Project\Baseline FP model\Baseline-LPV-Augmentation\simulations\param_recovery\checkpoint_e400_62737.pt'
DATASET         = 'base'   # 'base' or 'identification'
DTYPE           = torch.float64

# ── Dataset registry (mirrors train_param_recovery.py) ───────────────────────
_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
_TRAJ_BASE = (
    {'id': 'T1', 'file': 'T1_Y_sweep_conservative.mat'},
    {'id': 'T2', 'file': 'T2_X_sym_Y030.mat'},
    {'id': 'T3', 'file': 'T3_X_sym_Y000.mat'},
    {'id': 'T4', 'file': 'T4_X_antisym_Y020.mat'},
    {'id': 'T5', 'file': 'T5_X_sym_Y_sweep.mat'},
    {'id': 'T6', 'file': 'T6_Y_sweep_aggressive.mat'},
)
_DATASETS = {
    'base': dict(
        traj_dir   = os.path.join(_ROOT, 'Matlab-output', 'parameter-recovery'),
        traj_specs = _TRAJ_BASE,
    ),
    'identification': dict(
        traj_dir   = os.path.join(_ROOT, 'Matlab-output', 'identification-trajectories'),
        traj_specs = _TRAJ_BASE + (
            {'id': 'T7', 'file': 'T7_X_antisym_Y_sweep.mat'},
            {'id': 'T8', 'file': 'T8_X_sym_anti_Y_sweep.mat'},
        ),
    ),
}

# ── Physics helper (mirrors _build_sim_params in train_param_recovery.py) ────

def _build_sim_params(block):
    params = block._recover_params()
    kb1, kb2, cg1, cg2, cy, cb1, cb2, mh, m1, m2, mb, Jb, Jh, d = params
    params_10 = torch.stack([kb1+kb2, cg1, cg2, cy, cb1+cb2, mh, m1, m2, mb, Jb+Jh])
    _, M1, M2, K, C = _build_matrices(params_10, block._Lb, d)
    alpha, beta, gamma, N0, N1, N2 = build_poly_constants(m1, m2, mb, mh, Jb, Jh, block._Lb, d)
    d0 = mh * (alpha * gamma - beta ** 2)
    G  = build_G_matrix(N0, d0, M1, M2, K, C)
    return G, K, C, mh, alpha, beta, gamma, N0, N1, N2


if __name__ == '__main__':
    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt  = torch.load(CHECKPOINT_PATH, weights_only=False)
    epoch = ckpt.get('epoch', '?')

    # Use best eval params from history if available, else fall back to current
    eval_entries = [h for h in ckpt.get('history', []) if 'full_traj_rmse_m' in h]
    if eval_entries:
        best = min(eval_entries, key=lambda h: h['full_traj_rmse_m'])
        log_params = best['log_params_snapshot']
        print(f'Checkpoint: epoch={epoch}, using best eval epoch={best["epoch"]} '
              f'(RMSE={best["full_traj_rmse_m"]:.4e})\n  {CHECKPOINT_PATH}\n')
    else:
        log_params = ckpt['log_params']
        print(f'Checkpoint: epoch={epoch} (no eval history, using current params)\n'
              f'  {CHECKPOINT_PATH}\n')

    # ── Reconstruct block ─────────────────────────────────────────────────────
    block = ParameterizedLFRBlock(RMSE_baseline=1.0).to(dtype=DTYPE)
    with torch.no_grad():
        block.log_params.copy_(log_params.to(dtype=DTYPE))

    # ── Parameter tables ──────────────────────────────────────────────────────
    print(block.param_table())

    # ── Load trajectories and build simulation matrices (once) ────────────────
    cfg   = _DATASETS[DATASET]
    trajs = load_eval_trajs(cfg['traj_specs'], cfg['traj_dir'], D=1, dtype=DTYPE)
    ts_t  = _ts.to(DTYPE)
    P_t   = _P.to(DTYPE)

    with torch.no_grad():
        G, K, C, mh, alpha, beta, gamma, N0, N1, N2 = _build_sim_params(block)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print(f'\n  {"Traj":<6}  {"RMSE [m]":>12}  {"X1 [m]":>10}  {"X2 [m]":>10}  {"Y [m]":>10}')
    print(f'  {"-"*6}  {"-"*12}  {"-"*10}  {"-"*10}  {"-"*10}')

    n_trajs   = len(trajs)
    ch_names  = ['X1 [m]', 'X2 [m]', 'Y [m]']
    fig, axes = plt.subplots(n_trajs, 3, figsize=(15, 3 * n_trajs), sharex='row')
    if n_trajs == 1:
        axes = axes[None, :]

    rmse_all = []
    for row, traj in enumerate(trajs):
        with torch.no_grad():
            result = simulate(
                traj['state_traj'].to(DTYPE),  # (1, 6)  x0
                traj['u'].to(DTYPE),            # (1, T, 3)
                G, K, C, mh, alpha, beta, gamma, N0, N1, N2,
                P_t, ts_t, bptt_mode='full', return_latents=False,
            )

        pred    = result.Y[0]                         # (T, 3)
        q1      = traj['q1'].to(DTYPE)                # (T, 3)
        diff    = pred - q1
        rmse_ch = diff.pow(2).mean(dim=0).sqrt()      # (3,)
        rmse    = diff.pow(2).mean().sqrt().item()
        rmse_all.append(rmse)

        print(
            f'  {traj["id"]:<6}  {rmse:>12.4e}'
            f'  {rmse_ch[0]:>10.4e}  {rmse_ch[1]:>10.4e}  {rmse_ch[2]:>10.4e}'
        )

        t = torch.arange(q1.shape[0]).float() * float(ts_t)
        for ch in range(3):
            ax = axes[row, ch]
            ax.plot(t, q1[:, ch],   lw=0.6, color='steelblue',  label='measured')
            ax.plot(t, pred[:, ch], lw=0.6, color='darkorange', label='predicted', alpha=0.8)
            ax.set_title(f'{traj["id"]}  {ch_names[ch]}', fontsize=8)
            ax.set_xlabel('Time [s]', fontsize=7)
            ax.grid(True, alpha=0.3)
            if ch == 0:
                ax.set_ylabel('Position [m]', fontsize=7)
            if row == 0 and ch == 2:
                ax.legend(fontsize=7)

    mean_rmse = sum(rmse_all) / len(rmse_all)
    print(f'\n  Mean RMSE: {mean_rmse:.4e} m')

    save_path = os.path.join(
        os.path.dirname(CHECKPOINT_PATH), f'eval_e{epoch}_{DATASET}.png'
    )
    fig.suptitle(
        f'Epoch {epoch} — predicted vs measured ({DATASET}, mean RMSE={mean_rmse:.4e} m)'
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f'  Saved: {save_path}')
    plt.show()
