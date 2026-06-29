"""
run_telica_param_recovery.py
----------------------------
Run train_param_recovery on real Telica gantry data at a single operating point.

Strategy
--------
1. Patch precompute._load_trajectory to call telica_loader.load_telica_log,
   so .log files are read directly — no intermediate .mat conversion.
2. Patch precompute.compute_rmse_baseline_metrics with a dummy (the detuned-model
   baseline is for the Kamtin gantry, not Telica — not meaningful here).
3. Overwrite train_param_recovery module-level config globals before calling train().
4. After training: compute NRMSE per channel and save a trajectory comparison plot.

NOTE on param_table() output
-----------------------------
The printed % error columns compare recovered parameters against _TRUE_PARAMS,
which are the Kamtin gantry simulation values. These percentages are NOT meaningful
for Telica data. Ignore them. Only RMSE [m] and NRMSE [%] matter.

Run as:
    conda run -n GraduationProject python scripts/gantry/real-data-verification/run_telica_param_recovery.py
"""

__project_origin__ = "added"

import os
import sys

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)   # for telica_loader import (hyphen in folder name)

import matplotlib
matplotlib.use('Agg')   # non-interactive backend — safe for headless runs
import matplotlib.pyplot as plt
import torch

from telica_loader import load_telica_log

import lpv_lfr_baseline.scripts.precompute as _precompute
import lpv_lfr_baseline.scripts.train_param_recovery as tr
from lpv_lfr_baseline.scripts.precompute import _build_state_traj_logical
from lpv_lfr_baseline.scripts.train_param_recovery import _run_no_grad
from lpv_lfr_baseline.core.physics import P as _P, ts as _ts

# ── Operating point ───────────────────────────────────────────────────────────
# Start with a single operating point and a single iteration file.
# Change OP_FOLDER to switch operating point.
OP_FOLDER = 'xpos_-60_ypos-40'

_DATA_ROOT = os.path.join(
    _ROOT, 'kamtin-data', 'Data Telica', '06 40 mm XL 80 mm YL', 'train', OP_FOLDER
)
_SAVE_DIR = os.path.join(_ROOT, 'simulations', f'param_recovery_telica_{OP_FOLDER}')

# Trajectory files to use for training.
# Start with iterETEL (ETEL default feedforward, nonzero excitation).
# Add more iter files (e.g. iter5, iter8) once the single-file fit is assessed.
TRAJ_SPECS = (
    {'id': 'ETEL', 'file': 'iterETEL.log'},
)

# ── Patches ───────────────────────────────────────────────────────────────────

# 0. Use TRUE Kamtin params as init (not detuned +-10% which only makes sense for
#    simulation recovery tests). Patching the module-level name takes effect for all
#    subsequent ParameterizedLFRBlock instantiations, including the one inside tr.train().
import lpv_lfr_baseline.blocks.lfr_param_block as _lfr_pb
_lfr_pb._DETUNED_PARAMS = _lfr_pb._TRUE_PARAMS

# 1. Replace .mat loader with .log loader — same (u, q1, fs) return contract.
_precompute._load_trajectory = load_telica_log


def _dummy_rmse_baseline(mat_path, x0_logical, verbose=False):
    """
    Placeholder for compute_rmse_baseline_metrics.
    The detuned-model baseline (Kamtin params) is not meaningful for Telica.
    Returns unit values so precompute does not crash and rmse_baseline_normalized
    is set to a harmless dummy. The training loss and NRMSE post-eval are the
    only meaningful metrics here.
    """
    return {'mse_total': 1.0, 'rmse_total': 1.0, 'rmse_ch': [1.0, 1.0, 1.0]}


# 2. Replace RMSE baseline computation with dummy.
_precompute.compute_rmse_baseline_metrics = _dummy_rmse_baseline

# ── Dataset config overrides ──────────────────────────────────────────────────
tr.TRAJ_DIR      = _DATA_ROOT
tr.TRAJ_SPECS    = TRAJ_SPECS
tr.SAVE_DIR      = _SAVE_DIR
tr._VAL_TEST_DIR = _DATA_ROOT
tr.VAL_SPECS     = ()    # no held-out validation for single-trajectory fit
tr.TEST_SPECS    = ()
tr.DATASET       = f'telica_{OP_FOLDER}'

# ── Hyperparameter overrides ──────────────────────────────────────────────────
# THEORY: FS_NEW = 20000 → D=1 (loader already resamples to 20 kHz);
#         ts_eff = _ts × 1 = 1/20000 s, matching the simulation pipeline.
tr.FS_NEW            = 20_000
tr.SEGMENT_LEN       = 650     # same as simulation default; reduce if trajectory too short
tr.EPOCHS            = 80
tr.LR                = 1e-1
tr.VALIDATION_INTERVAL = None  # no val set → LR scheduler disabled
tr.NORM_MODE         = 'global'
tr.FULL_COVERAGE     = True
tr.OVERLAP_FRACTION  = 0.0
tr.SPLIT_REG_WEIGHT  = 0.0   # no regularization toward Kamtin params on real hardware
tr.W                 = None    # full segment per BPTT step

# ── Training ──────────────────────────────────────────────────────────────────

def _post_eval(block, traj_spec):
    """
    Compute per-channel NRMSE and save a trajectory comparison plot.
    Called after tr.train() returns.

    NRMSE thresholds (Schoukens & Ljung, Mech. Sys. Signal Process. 25(7), 2011;
                      Paduart et al., arXiv:1804.10758, 2018):
        THEORY: NRMSE < 15%  → model structure compatible with data
        THEORY: NRMSE > 30%  → structural mismatch (or force-signal problem)
        15–30%: ambiguous — inspect trajectory plot and residual spectrum
    """
    dtype = tr.DTYPE
    log_path = os.path.join(_DATA_ROOT, traj_spec['file'])
    traj_id  = traj_spec['id']

    print(f'\n{"=" * 60}')
    print(f'Post-training structural validation  [{traj_id}]')
    print(f'{"=" * 60}')

    # Load full trajectory (fresh, undecomposed into segments)
    u_eval, q1_eval, fs = load_telica_log(log_path, dtype=dtype)
    # u_eval: (1, T, 3)    q1_eval: (T, 3)

    # Build initial state from first samples of q1
    x0 = _build_state_traj_logical(
        q1_eval[:2], _P.to(dtype), float(_ts), dtype
    )[:1]   # (1, 6)

    ts_tensor = torch.tensor(float(_ts), dtype=dtype)

    # Simulate full trajectory open-loop with trained parameters
    result = _run_no_grad(block, x0, u_eval, ts_tensor)
    y_sim  = result.Y[0].detach()   # (T, 3)  [m]

    # Trim to minimum length (interpolation may shift end by 1 sample)
    T = min(y_sim.shape[0], q1_eval.shape[0])
    y_sim   = y_sim[:T]
    q1_eval = q1_eval[:T]

    # Per-channel RMSE and NRMSE
    diff     = y_sim - q1_eval                                   # (T, 3)
    rmse_ch  = diff.pow(2).mean(dim=0).sqrt()                    # (3,)  [m]
    sigma_ch = q1_eval.std(dim=0).clamp(min=1e-9)               # (3,)  [m]
    # THEORY: NRMSE = RMSE / std(y_measured) × 100%
    #         (Schoukens & Ljung 2011 — scale-independent structural metric)
    nrmse_ch = (rmse_ch / sigma_ch * 100.0).tolist()            # [%]
    rmse_tot = float(diff.pow(2).mean().sqrt().item())

    axes_lbl = ['X1', 'X2', 'Y']
    print(f'\n  {"Ch":<4}  {"RMSE [m]":>12}  {"NRMSE [%]":>10}  Verdict')
    print(f'  {"-"*4}  {"-"*12}  {"-"*10}  {"-"*12}')
    for i, lbl in enumerate(axes_lbl):
        n = nrmse_ch[i]
        verdict = 'GOOD (<15%)' if n < 15 else ('AMBIGUOUS' if n < 30 else 'POOR (>30%)')
        print(f'  {lbl:<4}  {rmse_ch[i].item():>12.4e}  {n:>10.2f}  {verdict}')
    print(f'\n  Overall RMSE : {rmse_tot:.4e} m')
    print(f'  Thresholds   : NRMSE < 15% compatible | > 30% structural mismatch')

    # ── Trajectory comparison plot ────────────────────────────────────────────
    t_s = torch.arange(T).float() / fs

    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for i, lbl in enumerate(axes_lbl):
        axs[i].plot(t_s.numpy(), q1_eval[:, i].numpy() * 1e3,
                    label='Measured', color='tab:blue', linewidth=0.8)
        axs[i].plot(t_s.numpy(), y_sim[:, i].numpy() * 1e3,
                    label='FP model (sim)', color='tab:orange',
                    linewidth=0.8, linestyle='--')
        axs[i].set_ylabel(f'{lbl} [mm]')
        axs[i].set_title(f'{lbl}: NRMSE = {nrmse_ch[i]:.1f}%', fontsize=9)
        axs[i].legend(loc='upper right', fontsize=8)
        axs[i].grid(True, alpha=0.3)
    axs[-1].set_xlabel('Time [s]')
    fig.suptitle(
        f'FP model vs Telica — {OP_FOLDER} — {traj_id}\n'
        f'Overall RMSE = {rmse_tot:.4e} m  '
        f'(X1: {nrmse_ch[0]:.1f}%  X2: {nrmse_ch[1]:.1f}%  Y: {nrmse_ch[2]:.1f}%)'
    )
    fig.tight_layout()

    os.makedirs(_SAVE_DIR, exist_ok=True)
    fig_path = os.path.join(_SAVE_DIR, f'trajectory_comparison_{traj_id}.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f'\n  Trajectory plot saved: {fig_path}')
    plt.close(fig)


if __name__ == '__main__':
    block = tr.train(
        epochs=tr.EPOCHS,
        lr=tr.LR,
        save_dir=_SAVE_DIR,
        split_reg_weight=0.0,
        norm_mode=tr.NORM_MODE,
    )

    # Evaluate the first (and typically only) training trajectory
    _post_eval(block, TRAJ_SPECS[0])
