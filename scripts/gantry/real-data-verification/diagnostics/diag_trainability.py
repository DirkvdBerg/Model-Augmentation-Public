"""
diag_trainability.py
--------------------
Trainability diagnostic for Telica data.

Answers the question: can the FP model parameters learn from real Telica ILC data?

Steps
-----
1. Evaluate RMSE / NRMSE before any training (initial = TRUE Kamtin params).
2. Train for EPOCHS epochs with SPLIT_REG_WEIGHT=0 (no Kamtin-biased regularization).
3. Evaluate RMSE / NRMSE after training.
4. Print before/after table.
5. Save trajectory comparison figure (initial vs trained vs measured).

Key design choices vs run_telica_param_recovery.py
---------------------------------------------------
- SPLIT_REG_WEIGHT = 0.0  : regularization toward Kamtin detuned params would resist
                            the optimizer finding Telica-specific parameters.
- init = _TRUE_PARAMS     : best physical prior for a similar gantry (no artificial
                            +-10% offset that only makes sense for recovery testing).
- EPOCHS = 30             : enough to see a learning trend without a full run.

Run as:
    conda run -n GraduationProject python scripts/gantry/real-data-verification/diag_trainability.py
"""

__project_origin__ = "added"

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from telica_loader import load_telica_log

import lpv_lfr_baseline.blocks.lfr_param_block as _lfr_pb
import lpv_lfr_baseline.scripts.precompute as _precompute
import lpv_lfr_baseline.scripts.train_param_recovery as tr
from lpv_lfr_baseline.blocks.lfr_param_block import ParameterizedLFRBlock
from lpv_lfr_baseline.scripts.precompute import _build_state_traj_logical
from lpv_lfr_baseline.scripts.train_param_recovery import _run_no_grad
from lpv_lfr_baseline.core.physics import P as _P, ts as _ts

# ── Config ────────────────────────────────────────────────────────────────────
OP_FOLDER = 'xpos_-60_ypos-40'
_LOG_FILE  = 'iterETEL.log'
EPOCHS     = 10

_DATA_ROOT = os.path.join(
    _ROOT, 'kamtin-data', 'Data Telica', '06 40 mm XL 80 mm YL', 'train', OP_FOLDER
)
_SAVE_DIR = os.path.join(_ROOT, 'simulations', f'diag_trainability_telica_{OP_FOLDER}')

# ── Patches ───────────────────────────────────────────────────────────────────

# Use TRUE Kamtin params as init (not detuned +-10% which is only meaningful for
# simulation recovery tests). ParameterizedLFRBlock.__init__ reads _DETUNED_PARAMS
# from the module namespace at construction time, so this patch takes effect for
# all subsequent instantiations, including the one inside tr.train().
_lfr_pb._DETUNED_PARAMS = _lfr_pb._TRUE_PARAMS

# Replace .mat loader with Telica .log loader (same (u, q1, fs) return contract)
_precompute._load_trajectory = load_telica_log


def _dummy_rmse_baseline(mat_path, x0_logical, verbose=False):
    """Kamtin detuned-model baseline is not meaningful for Telica hardware."""
    return {'mse_total': 1.0, 'rmse_total': 1.0, 'rmse_ch': [1.0, 1.0, 1.0]}


_precompute.compute_rmse_baseline_metrics = _dummy_rmse_baseline

# ── Training config overrides ─────────────────────────────────────────────────
tr.TRAJ_DIR            = _DATA_ROOT
tr.TRAJ_SPECS          = ({'id': 'ETEL', 'file': _LOG_FILE},)
tr.SAVE_DIR            = _SAVE_DIR
tr._VAL_TEST_DIR       = _DATA_ROOT
tr.VAL_SPECS           = ()
tr.TEST_SPECS          = ()
tr.DATASET             = f'telica_{OP_FOLDER}'
tr.FS_NEW              = 20_000
tr.SEGMENT_LEN         = 650
tr.EPOCHS              = EPOCHS
tr.LR                  = 1e-3
tr.VALIDATION_INTERVAL = None   # no held-out val -> LR scheduler disabled
tr.NORM_MODE           = 'global'
tr.FULL_COVERAGE       = False  # 1 random segment per epoch -- enough to see learning trend
tr.OVERLAP_FRACTION    = 0.0
tr.SPLIT_REG_WEIGHT    = 0.0    # disable regularization toward Kamtin params
tr.W                   = None

# ── Evaluation helper ─────────────────────────────────────────────────────────

def _simulate(block, log_path, dtype=torch.float64):
    """Load log file, run open-loop simulation, return (y_sim, q1, fs)."""
    u, q1, fs = load_telica_log(log_path, dtype=dtype)
    x0 = _build_state_traj_logical(
        q1[:2], _P.to(dtype), float(_ts), dtype
    )[:1]                                              # (1, 6)
    ts_tensor = torch.tensor(float(_ts), dtype=dtype)
    result    = _run_no_grad(block, x0, u, ts_tensor)
    y_sim     = result.Y[0].detach()                   # (T, 3)
    T         = min(y_sim.shape[0], q1.shape[0])
    return y_sim[:T], q1[:T], fs


def _print_eval(label, y_sim, q1):
    """Print per-channel RMSE [m] and NRMSE [%] with GOOD/AMBIGUOUS/POOR verdict."""
    diff     = y_sim - q1
    rmse_ch  = diff.pow(2).mean(dim=0).sqrt()           # (3,) [m]
    sigma_ch = q1.std(dim=0).clamp(min=1e-9)            # (3,) [m]
    nrmse_ch = (rmse_ch / sigma_ch * 100.0).tolist()    # [%]
    rmse_tot = float(diff.pow(2).mean().sqrt())

    axes = ['X1', 'X2', 'Y']
    print(f'\n  {label}')
    print(f'  {"Ch":<4}  {"RMSE [m]":>12}  {"NRMSE [%]":>10}  Verdict')
    print(f'  {"-"*4}  {"-"*12}  {"-"*10}  {"-"*12}')
    for i, ax in enumerate(axes):
        n = nrmse_ch[i]
        verdict = 'GOOD (<15%)' if n < 15 else ('AMBIGUOUS' if n < 30 else 'POOR (>30%)')
        print(f'  {ax:<4}  {rmse_ch[i].item():>12.4e}  {n:>10.2f}  {verdict}')
    print(f'\n  Overall RMSE: {rmse_tot:.4e} m')

    return rmse_ch, nrmse_ch


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(_SAVE_DIR, exist_ok=True)
    dtype    = tr.DTYPE
    log_path = os.path.join(_DATA_ROOT, _LOG_FILE)

    # ── Step 1: Initial evaluation (TRUE Kamtin params, no training) ──────────
    print('=' * 60)
    print('Step 1: Initial evaluation  (init = TRUE Kamtin params)')
    print('=' * 60)

    initial_block = ParameterizedLFRBlock(RMSE_baseline=1.0).to(dtype=dtype)
    y_init, q1_ref, fs = _simulate(initial_block, log_path, dtype=dtype)
    rmse_init, nrmse_init = _print_eval('Initial (TRUE Kamtin params)', y_init, q1_ref)

    # ── Step 2: Train ─────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print(f'Step 2: Training  ({EPOCHS} epochs, SPLIT_REG_WEIGHT=0.0)')
    print('=' * 60)

    trained_block = tr.train(
        epochs=EPOCHS,
        lr=1e-1,
        save_dir=_SAVE_DIR,
        split_reg_weight=0.0,
        norm_mode='global',
    )

    # ── Step 3: Final evaluation ──────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('Step 3: Final evaluation  (after training)')
    print('=' * 60)

    y_final, _, _ = _simulate(trained_block, log_path, dtype=dtype)
    rmse_final, nrmse_final = _print_eval('After training', y_final, q1_ref)

    # ── Step 4: Before / after summary ───────────────────────────────────────
    print('\n' + '=' * 60)
    print('Summary')
    print('=' * 60)
    axes = ['X1', 'X2', 'Y']
    print(
        f'\n  {"Ch":<4}  {"RMSE_init":>12}  {"RMSE_final":>12}'
        f'  {"NRMSE_init":>11}  {"NRMSE_final":>12}  {"Delta":>8}'
    )
    print(f'  {"-"*4}  {"-"*12}  {"-"*12}  {"-"*11}  {"-"*12}  {"-"*8}')
    for i, ax in enumerate(axes):
        ni = nrmse_init[i]
        nf = nrmse_final[i]
        print(
            f'  {ax:<4}  {rmse_init[i].item():>12.4e}  {rmse_final[i].item():>12.4e}'
            f'  {ni:>10.2f}%  {nf:>11.2f}%  {nf-ni:>+7.2f}%'
        )

    # ── Step 5: Comparison figure ─────────────────────────────────────────────
    T   = q1_ref.shape[0]
    t_s = torch.arange(T).float() / fs

    fig, axs = plt.subplots(3, 2, figsize=(14, 9), sharex=True)
    for i, ax in enumerate(axes):
        q_mm    = q1_ref[:, i].numpy() * 1e3
        yi_mm   = y_init[:, i].numpy() * 1e3
        yf_mm   = y_final[:, i].numpy() * 1e3
        t_np    = t_s.numpy()

        # Left column: initial
        axs[i, 0].plot(t_np, q_mm,  label='Measured',    color='tab:blue',   lw=0.8)
        axs[i, 0].plot(t_np, yi_mm, label='Initial sim', color='tab:orange', lw=0.8, ls='--')
        axs[i, 0].set_ylabel(f'{ax} [mm]')
        axs[i, 0].set_title(
            f'{ax} initial: NRMSE={nrmse_init[i]:.1f}%', fontsize=9
        )
        axs[i, 0].legend(loc='upper right', fontsize=8)
        axs[i, 0].grid(True, alpha=0.3)

        # Right column: trained
        axs[i, 1].plot(t_np, q_mm,  label='Measured',              color='tab:blue',  lw=0.8)
        axs[i, 1].plot(t_np, yf_mm, label=f'Trained ({EPOCHS} ep)', color='tab:green', lw=0.8, ls='--')
        axs[i, 1].set_title(
            f'{ax} trained: NRMSE={nrmse_final[i]:.1f}%', fontsize=9
        )
        axs[i, 1].legend(loc='upper right', fontsize=8)
        axs[i, 1].grid(True, alpha=0.3)

    axs[-1, 0].set_xlabel('Time [s]')
    axs[-1, 1].set_xlabel('Time [s]')
    fig.suptitle(
        f'Trainability diagnostic -- Telica {OP_FOLDER} -- {_LOG_FILE}\n'
        f'{EPOCHS} epochs, SPLIT_REG_WEIGHT=0, init=TRUE Kamtin params'
    )
    fig.tight_layout()

    fig_path = os.path.join(_SAVE_DIR, f'diag_trainability_{OP_FOLDER}.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f'\nFigure saved: {fig_path}')
    plt.close(fig)
