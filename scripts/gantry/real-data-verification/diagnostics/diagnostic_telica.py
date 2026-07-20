"""
diagnostic_telica.py
--------------------
Four-stage diagnostic for the Telica real-data verification pipeline.
Run this before run_telica_param_recovery.py to catch problems early.

Each stage is independent. Later stages build on earlier outputs but will
report clearly if an earlier stage must run first.

Stages
------
1. Loader only    — load_telica_log(): shapes, value ranges, motion detection
2. Precompute     — patches + precompute(): sigma, segment_len, pool sizes
3. Detuned sim    — full open-loop simulation with initial (Kamtin) parameters
4. One grad step  — verify loss and grad_norm are finite before 500-epoch run

Outputs: printed PASS/FAIL per check + plots in simulations/param_recovery_telica_diagnostic/

Run as:
    conda run -n GraduationProject python scripts/gantry/real-data-verification/diagnostic_telica.py
"""

__project_origin__ = "added"

import os
import sys

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..', '..'))
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from telica_loader import load_telica_log, _PRE_MOTION_MS, _CI_TO_AMP, _DPI_TO_M

import lpv_lfr_baseline.scripts.precompute as _precompute
import lpv_lfr_baseline.scripts.train_param_recovery as tr
from lpv_lfr_baseline.scripts.precompute import (
    precompute as _run_precompute,
    _build_state_traj_logical,
    _build_segment_pools,
)
from lpv_lfr_baseline.scripts.train_param_recovery import (
    _run_no_grad, _build_sim_params, _sample_batch, DTYPE,
)
from lpv_lfr_baseline.blocks.lfr_param_block import ParameterizedLFRBlock
from lpv_lfr_baseline.core.lfr_simulate import simulate
from lpv_lfr_baseline.core.physics import P as _P, ts as _ts

# ── Config — mirrors run_telica_param_recovery.py ─────────────────────────────
OP_FOLDER = 'xpos_-60_ypos-40'
_DATA_ROOT = os.path.join(
    _ROOT, 'kamtin-data', 'Data Telica', '06 40 mm XL 80 mm YL', 'train', OP_FOLDER
)
_LOG_FILE  = 'iterETEL.log'
_LOG_PATH  = os.path.join(_DATA_ROOT, _LOG_FILE)

_DIAG_DIR  = os.path.join(_ROOT, 'simulations', 'param_recovery_telica_diagnostic')
_TRAJ_SPECS = ({'id': 'ETEL', 'file': _LOG_FILE},)

# THEORY: NRMSE thresholds from Schoukens & Ljung (2011) Mech. Syst. Signal Process.
_NRMSE_GOOD = 15.0   # %
_NRMSE_POOR = 30.0   # %

# HEURISTIC: plausible physical range for stage position [m] at any Telica op point
_Q1_MAX_ABS = 0.5    # m  (any axis position unlikely to exceed 500 mm)
# HEURISTIC: plausible total current command range [A]
_U_MAX_ABS  = 20.0   # A


# ── Utilities ─────────────────────────────────────────────────────────────────

def _check(name, cond, detail=''):
    status = 'PASS' if cond else 'FAIL'
    suffix = f'  ({detail})' if detail else ''
    marker = '  [' + status + '] '
    print(marker + name + suffix)
    return bool(cond)


def _save_fig(fig, name):
    os.makedirs(_DIAG_DIR, exist_ok=True)
    path = os.path.join(_DIAG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Plot saved → {path}')


def _nrmse(y_sim, y_meas):
    """Per-channel NRMSE [%]. y_sim, y_meas: (T, 3) tensors."""
    diff     = y_sim - y_meas
    rmse_ch  = diff.pow(2).mean(dim=0).sqrt()
    sigma_ch = y_meas.std(dim=0).clamp(min=1e-9)
    return (rmse_ch / sigma_ch * 100.0)   # (3,) [%]


# ── Stage 1 — Loader ──────────────────────────────────────────────────────────

def run_stage1():
    """
    Load one .log file and verify shapes, value ranges, and motion detection.
    Saves a 6-panel signal plot.
    """
    print(f'  File: {_LOG_PATH}')
    results = []

    # ── 1a. File exists ───────────────────────────────────────────────────────
    results.append(_check('log file exists', os.path.isfile(_LOG_PATH)))
    if not results[-1]:
        return False

    # ── 1b. Loader returns without error ─────────────────────────────────────
    try:
        u, q1, fs = load_telica_log(_LOG_PATH, dtype=DTYPE)
        loaded_ok = True
    except Exception as exc:
        print(f'  [FAIL] loader raised: {exc}')
        return False
    results.append(_check('load_telica_log completed without error', loaded_ok))

    T = q1.shape[0]

    # ── 1c. Shapes ───────────────────────────────────────────────────────────
    results.append(_check('u shape is (1, T, 3)',
                          u.shape == (1, T, 3),
                          f'got {tuple(u.shape)}'))
    results.append(_check('q1 shape is (T, 3)',
                          q1.shape == (T, 3),
                          f'got {tuple(q1.shape)}'))
    results.append(_check('fs == 20000.0',
                          abs(fs - 20_000.0) < 1.0,
                          f'got {fs}'))

    # ── 1d. Data is finite ────────────────────────────────────────────────────
    results.append(_check('q1 has no NaN/Inf', q1.isfinite().all().item()))
    results.append(_check('u  has no NaN/Inf', u.isfinite().all().item()))

    # ── 1e. Value ranges ─────────────────────────────────────────────────────
    q1_max = float(q1.abs().max().item())
    u_max  = float(u.abs().max().item())
    results.append(_check(
        f'q1 range plausible (|max| < {_Q1_MAX_ABS} m)',
        q1_max < _Q1_MAX_ABS,
        f'|max| = {q1_max:.4f} m'
    ))
    results.append(_check(
        f'u  range plausible (|max| < {_U_MAX_ABS} A)',
        u_max < _U_MAX_ABS,
        f'|max| = {u_max:.4f} A'
    ))

    # ── 1f. Motion is non-trivial ─────────────────────────────────────────────
    q1_range = float((q1.max(dim=0).values - q1.min(dim=0).values).max().item())
    results.append(_check(
        'q1 shows motion (peak-to-peak > 1 mm on at least one channel)',
        q1_range > 1e-3,
        f'max p-p = {q1_range*1e3:.2f} mm'
    ))

    # ── 1g. Trajectory length ─────────────────────────────────────────────────
    duration_s = T / fs
    results.append(_check(
        'trajectory length > 0.1 s',
        duration_s > 0.1,
        f'T = {T} samples  ({duration_s:.3f} s at {fs:.0f} Hz)'
    ))
    print(f'\n  Summary: T={T} samples, {duration_s:.3f} s, '
          f'q1 p-p={q1_range*1e3:.1f} mm, u_max={u_max:.3f} A')

    # ── 1h. Signal plot ───────────────────────────────────────────────────────
    t_s = np.arange(T) / fs
    axes_lbl = ['X1', 'X2', 'Y']
    fig, axs = plt.subplots(6, 1, figsize=(12, 10), sharex=True)
    for i, lbl in enumerate(axes_lbl):
        axs[i].plot(t_s, q1[:, i].numpy() * 1e3, linewidth=0.6)
        axs[i].set_ylabel(f'q1 {lbl} [mm]')
        axs[i].grid(True, alpha=0.3)
    for i, lbl in enumerate(axes_lbl):
        axs[3 + i].plot(t_s, u[0, :, i].numpy(), linewidth=0.6, color='tab:orange')
        axs[3 + i].set_ylabel(f'u {lbl} [A]')
        axs[3 + i].grid(True, alpha=0.3)
    axs[-1].set_xlabel('Time [s]')
    fig.suptitle(f'Stage 1 — Raw signals: {OP_FOLDER} / {_LOG_FILE}')
    fig.tight_layout()
    _save_fig(fig, 'stage1_raw_signals.png')

    return all(results)


# ── Stage 2 — Precompute ──────────────────────────────────────────────────────

def run_stage2():
    """
    Apply both patches and call precompute.precompute(). Verify sigma, segment_len,
    and pool sizes. Confirms the monkey-patch binds correctly end-to-end.
    """
    # Apply patches (same as run_telica_param_recovery.py)
    _precompute._load_trajectory = load_telica_log

    def _dummy_rmse_baseline(mat_path, x0_logical, verbose=False):
        return {'mse_total': 1.0, 'rmse_total': 1.0, 'rmse_ch': [1.0, 1.0, 1.0]}

    _precompute.compute_rmse_baseline_metrics = _dummy_rmse_baseline

    results = []

    try:
        pre = _run_precompute(
            traj_specs       = _TRAJ_SPECS,
            traj_dir         = _DATA_ROOT,
            save_dir         = _DIAG_DIR,
            dtype            = DTYPE,
            norm_mode        = 'global',
            overlap_fraction = 0.0,
            fs_new           = 20_000,
            segment_len      = 650,
        )
        pc_ok = True
    except Exception as exc:
        print(f'  [FAIL] precompute raised: {exc}')
        return False

    results.append(_check('precompute() completed without error', pc_ok))

    # Check keys
    for key in ('trajs', 'sigma', 'segment_len', 'pools', 'D', 'fs_new', 'ts_eff'):
        results.append(_check(f'output key "{key}" present', key in pre,
                              f'keys={list(pre.keys())}'))

    D        = pre['D']
    fs_new   = pre['fs_new']
    seg_len  = pre['segment_len']
    sigma    = pre['sigma']

    results.append(_check('D == 1 (no decimation needed)',
                          D == 1,
                          f'D={D}, fs_new={fs_new}'))
    results.append(_check('segment_len == 650',
                          seg_len == 650,
                          f'segment_len={seg_len}'))

    # Sigma: should be physically plausible (order of trajectory std)
    global_sigma = sigma[_TRAJ_SPECS[0]['id']]
    sigma_ok = (global_sigma > 1e-6).all().item() and (global_sigma < 1.0).all().item()
    results.append(_check('sigma per-channel in (1e-6, 1.0) m',
                          sigma_ok,
                          f'sigma={global_sigma.tolist()}'))

    # Pool sizes
    pools = pre['pools']
    for spec in _TRAJ_SPECS:
        tid   = spec['id']
        n_seg = len(pools[tid])
        results.append(_check(f'pool[{tid}] has >= 1 segment',
                              n_seg >= 1,
                              f'{n_seg} segments'))
        print(f'  Pool[{tid}]: {n_seg} segments of length {seg_len}')

    return all(results)


# ── Stage 3 — Detuned simulation ──────────────────────────────────────────────

def run_stage3():
    """
    Build ParameterizedLFRBlock with initial (Kamtin) parameters and run a
    full open-loop simulation on the Telica trajectory. Reports NRMSE as the
    detuned starting point — tells us how far from Telica the initial params are.
    """
    results = []

    # Load trajectory (direct, no precompute needed)
    try:
        u_eval, q1_eval, fs = load_telica_log(_LOG_PATH, dtype=DTYPE)
    except Exception as exc:
        print(f'  [FAIL] load_telica_log raised: {exc}')
        return False

    T = q1_eval.shape[0]
    ts_tensor = torch.tensor(float(_ts), dtype=DTYPE)

    # Build initial state (from first 2 samples of q1)
    x0 = _build_state_traj_logical(
        q1_eval[:2], _P.to(DTYPE), float(_ts), DTYPE
    )[:1]   # (1, 6)

    results.append(_check('x0 shape is (1, 6)',
                          x0.shape == (1, 6),
                          f'got {tuple(x0.shape)}'))
    results.append(_check('x0 is finite', x0.isfinite().all().item()))

    # Build block with dummy RMSE_baseline (not meaningful for Telica)
    block = ParameterizedLFRBlock(RMSE_baseline=1.0).to(dtype=DTYPE)
    results.append(_check('ParameterizedLFRBlock constructed', True))

    # Run detuned simulation
    try:
        result  = _run_no_grad(block, x0, u_eval, ts_tensor)
        sim_ok  = True
    except Exception as exc:
        print(f'  [FAIL] _run_no_grad raised: {exc}')
        return False
    results.append(_check('_run_no_grad completed without error', sim_ok))

    y_sim = result.Y[0].detach()   # (T, 3)

    results.append(_check('y_sim shape is (T, 3)',
                          y_sim.shape == (T, 3),
                          f'got {tuple(y_sim.shape)}'))
    results.append(_check('y_sim is finite', y_sim.isfinite().all().item()))

    if not y_sim.isfinite().all().item():
        print('  ODE blew up with detuned parameters — '
              'expected for large parameter mismatch. '
              'Training will use shorter segments; this may still converge.')

    # NRMSE with detuned parameters
    T_min     = min(y_sim.shape[0], q1_eval.shape[0])
    nrmse_ch  = _nrmse(y_sim[:T_min], q1_eval[:T_min])
    nrmse_tot = float(nrmse_ch.mean().item())
    rmse_tot  = float((y_sim[:T_min] - q1_eval[:T_min]).pow(2).mean().sqrt().item())

    axes_lbl = ['X1', 'X2', 'Y']
    print(f'\n  Detuned model NRMSE (starting point for training):')
    print(f'  {"Ch":<4}  {"NRMSE [%]":>10}')
    for i, lbl in enumerate(axes_lbl):
        n = float(nrmse_ch[i].item())
        note = '' if y_sim.isfinite().all() else '  (NaN — ODE diverged)'
        print(f'  {lbl:<4}  {n:>10.2f}{note}')
    print(f'  Mean NRMSE: {nrmse_tot:.2f}%  |  RMSE: {rmse_tot:.4e} m')
    if nrmse_tot < 50:
        print('  Initial params are in the right ballpark — training has a good start.')
    else:
        print('  Large initial mismatch — training will need more epochs to converge.')

    # ── Trajectory comparison plot ─────────────────────────────────────────────
    t_s = np.arange(T_min) / fs
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for i, lbl in enumerate(axes_lbl):
        axs[i].plot(t_s, q1_eval[:T_min, i].numpy() * 1e3,
                    label='Measured', color='tab:blue', linewidth=0.8)
        axs[i].plot(t_s, y_sim[:T_min, i].numpy() * 1e3,
                    label='FP model (detuned init)', color='tab:red',
                    linewidth=0.8, linestyle='--')
        axs[i].set_ylabel(f'{lbl} [mm]')
        axs[i].set_title(f'{lbl}: NRMSE = {float(nrmse_ch[i].item()):.1f}%  (detuned)',
                         fontsize=9)
        axs[i].legend(loc='upper right', fontsize=8)
        axs[i].grid(True, alpha=0.3)
    axs[-1].set_xlabel('Time [s]')
    fig.suptitle(
        f'Stage 3 — Detuned model vs Telica data ({OP_FOLDER})\n'
        f'Overall NRMSE = {nrmse_tot:.1f}%  RMSE = {rmse_tot:.4e} m'
    )
    fig.tight_layout()
    _save_fig(fig, 'stage3_detuned_simulation.png')

    return all(results)


# ── Stage 4 — Single gradient step ────────────────────────────────────────────

def run_stage4():
    """
    Run one gradient step to confirm the loss and gradients are finite before
    committing to a full training run. Uses the first segment of the trajectory.
    """
    results = []

    # Load trajectory
    try:
        u_raw, q1_raw, fs = load_telica_log(_LOG_PATH, dtype=DTYPE)
    except Exception as exc:
        print(f'  [FAIL] load_telica_log raised: {exc}')
        return False

    # Build state trajectory (needed for x0 at each segment start)
    state_traj = _build_state_traj_logical(q1_raw, _P.to(DTYPE), float(_ts), DTYPE)

    seg_len  = 650
    T        = q1_raw.shape[0]
    ts_tensor = torch.tensor(float(_ts), dtype=DTYPE)

    if T < seg_len:
        print(f'  [FAIL] trajectory too short: T={T} < segment_len={seg_len}')
        return False

    # Build block
    block     = ParameterizedLFRBlock(RMSE_baseline=1.0).to(dtype=DTYPE)
    optimizer = torch.optim.Adam(block.parameters(), lr=1e-3)

    # Take first segment (index 0 → index seg_len)
    x0_seg  = state_traj[0:1]                      # (1, 6)
    u_seg   = u_raw[:, 0:seg_len, :]               # (1, seg_len, 3)
    q1_seg  = q1_raw[0:seg_len, :].unsqueeze(0)    # (1, seg_len, 3)

    # Sigma: per-channel std of q1 for normalizing loss
    sigma = q1_raw.std(dim=0).clamp(min=1e-6)      # (3,)

    # Forward pass
    try:
        G, K, C, mh, alpha, beta, gamma, N0, N1, N2 = _build_sim_params(block)
        result = simulate(
            x0_seg, u_seg,
            G, K, C, mh, alpha, beta, gamma, N0, N1, N2,
            block._P, ts_tensor,
            bptt_mode='full', return_latents=False,
        )
        fwd_ok = True
    except Exception as exc:
        print(f'  [FAIL] forward pass raised: {exc}')
        return False
    results.append(_check('forward pass completed', fwd_ok))

    # Loss
    err      = (result.Y - q1_seg) / sigma.unsqueeze(0).unsqueeze(0)  # (1, seg_len, 3)
    mse_loss = err.pow(2).mean()
    loss_ok  = float(mse_loss.item()) > 0 and torch.isfinite(mse_loss).item()
    results.append(_check('loss is finite and > 0',
                          loss_ok,
                          f'loss={mse_loss.item():.4e}'))

    # Backward
    try:
        mse_loss.backward()
        bwd_ok = True
    except Exception as exc:
        print(f'  [FAIL] backward raised: {exc}')
        return False
    results.append(_check('backward completed', bwd_ok))

    # Gradient norm
    grad = block.log_params.grad
    grad_ok    = grad is not None and torch.isfinite(grad).all().item()
    grad_norm  = float(grad.norm().item()) if grad is not None else float('nan')
    results.append(_check('gradients are finite',
                          grad_ok,
                          f'grad_norm={grad_norm:.4e}'))

    # Gradient norm should be non-zero and not exploding
    results.append(_check('grad_norm in (1e-8, 1e4)',
                          1e-8 < grad_norm < 1e4,
                          f'grad_norm={grad_norm:.4e}'))

    optimizer.zero_grad()
    print(f'\n  loss={mse_loss.item():.4e}  grad_norm={grad_norm:.4e}')
    if grad_norm > 1e3:
        print('  WARNING: large grad_norm — consider reducing LR or adding '
              'gradient clipping before the full run.')

    return all(results)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(_DIAG_DIR, exist_ok=True)

    stage_results = {}

    for stage_num, stage_fn, label in [
        (1, run_stage1, 'Loader'),
        (2, run_stage2, 'Precompute'),
        (3, run_stage3, 'Detuned simulation'),
        (4, run_stage4, 'One gradient step'),
    ]:
        print(f'\n{"=" * 60}')
        print(f'Stage {stage_num}: {label}')
        print('=' * 60)
        try:
            stage_results[stage_num] = stage_fn()
        except Exception as exc:
            print(f'  [FAIL] Unhandled exception: {exc}')
            stage_results[stage_num] = False

    print(f'\n{"=" * 60}')
    print('Diagnostic summary')
    print('=' * 60)
    all_pass = True
    for num, label in [(1, 'Loader'), (2, 'Precompute'),
                       (3, 'Detuned sim'), (4, 'Gradient step')]:
        passed = stage_results.get(num, False)
        all_pass = all_pass and passed
        print(f'  Stage {num} ({label}): {"PASS" if passed else "FAIL"}')

    print()
    if all_pass:
        print('  All stages passed — safe to run run_telica_param_recovery.py')
    else:
        print('  Fix failing stages before running the full training.')
    print(f'  Plots saved to: {_DIAG_DIR}')
