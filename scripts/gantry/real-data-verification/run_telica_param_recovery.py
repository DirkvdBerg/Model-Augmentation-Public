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
from matplotlib.ticker import FuncFormatter as _FuncFormatter
from datetime import datetime
import torch

from telica_loader import load_telica_log, load_telica_log_cl, _A_TO_N
from telica_controller import TelicaFeedbackController

import lpv_lfr_baseline.scripts.precompute as _precompute
import lpv_lfr_baseline.scripts.train_param_recovery as tr
from lpv_lfr_baseline.scripts.precompute import _build_state_traj_logical
from lpv_lfr_baseline.scripts.train_param_recovery import _run_no_grad
from lpv_lfr_baseline.core.physics import P as _P, ts as _ts

# ── Data selection ────────────────────────────────────────────────────────────
# Supervisor's train/validation/test split (D-075): split is BY OPERATING
# POINT (folders under the dataset root). Per OP we use iter0 (feedforward
# off, feedback-dominated input) and iter8 (converged ILC, feedforward-
# dominated input): the two extreme input spectra. iter6_1.log (redo
# artifact) is excluded. See docs/kamtin-telica-schema.md.
_DATASET_ROOT = os.path.join(_ROOT, 'kamtin-data', 'Data Telica',
                             '06 40 mm XL 80 mm YL')
_SAVE_DIR = os.path.join(_ROOT, 'simulations', 'pr_telica_split')

_TRAIN_OPS = (
    'xpos_-60_ypos-40',   'xpos_-60_ypos120',  'xpos_-60_ypos-120',
    'xpos_-60_ypos-200',  'xpos_-135_ypos40',  'xpos_-135_ypos-40',
    'xpos_-135_ypos-200', 'xpos_-210_ypos40',  'xpos_-210_ypos120',
    'xpos_-210_ypos-120', 'xpos_-210_ypos-200',
)
_VAL_OPS  = ('xpos_-135_ypos120', 'xpos_-210_ypos-40')
_TEST_OPS = ('xpos_-135_ypos-120', 'xpos_-60_ypos40')

_ITERS_TRAINVAL = (('a', 'iter0.log'), ('b', 'iter8.log'))
_ITERS_TEST     = (('a', 'iter0.log'), ('b', 'iter8.log'), ('T', 'iterTEST.log'))


def _mk_specs(prefix, split, ops, iters):
    """IDs like T1a..T11b (a=iter0, b=iter8, T=iterTEST); 'file' is relative
    to _DATASET_ROOT; 'label' is the human-readable OP + iteration."""
    specs = []
    for k, op in enumerate(ops, start=1):
        for suffix, fname in iters:
            specs.append({'id': f'{prefix}{k}{suffix}',
                          'file': f'{split}/{op}/{fname}',
                          'label': f'{op} {fname[:-4]}'})
    return tuple(specs)


TRAJ_SPECS = _mk_specs('T', 'train',      _TRAIN_OPS, _ITERS_TRAINVAL)  # 22
VAL_SPECS  = _mk_specs('V', 'validation', _VAL_OPS,   _ITERS_TRAINVAL)  # 4
TEST_SPECS = _mk_specs('E', 'test',       _TEST_OPS,  _ITERS_TEST)      # 6

# Final-evaluation set: one train sample (both iterations) + validation + test
EVAL_SPECS = TRAJ_SPECS[:2] + VAL_SPECS + TEST_SPECS

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
tr.TRAJ_DIR      = _DATASET_ROOT
tr.TRAJ_SPECS    = TRAJ_SPECS
tr.SAVE_DIR      = _SAVE_DIR
tr._VAL_TEST_DIR = _DATASET_ROOT
tr.VAL_SPECS     = VAL_SPECS    # drives LR scheduler + best-checkpoint via windowed loss (D-076)
tr.TEST_SPECS    = TEST_SPECS   # evaluated only in tr.train()'s final step
tr.DATASET       = 'telica_split'
# 22 concatenated trajectory IDs would blow the Windows 260-char path limit
tr._traj_set_tag = lambda specs: f'{len(specs)}traj'


# ── Windowed validation selector (D-076) ──────────────────────────────────────
# Replaces tr._full_traj_eval (full-trajectory OPEN-LOOP RMSE) with the SAME
# measure training optimizes: sigma-normalized MSE on teacher-forced windows of
# the held-out validation trajectories. The full-trajectory OL metric is
# dominated by friction/force-scale drift on the K=0 axes (double integrator),
# which no parameter can reduce, so it degraded monotonically and starved the
# LR scheduler (run 68775). Same call signature (block, trajs, ts_tensor) ->
# (scalar, entries), so both the sync and 2-GPU async eval paths and the
# scheduler/best-selection logic are untouched.
def _windowed_val_eval(block, trajs, ts_tensor):
    from lpv_lfr_baseline.core.lfr_simulate import simulate
    seg = int(tr.SEGMENT_LEN)
    dev = block._P.device
    # global sigma from the validation q1 (fixed across epochs -> monotone metric)
    q1_all = torch.cat([t['q1'] for t in trajs], dim=0).to(dev)
    sigma  = torch.stack([q1_all[:, c].std().clamp(min=1e-4) for c in range(3)])

    G, K, C, mh, alpha, beta, gamma, N0, N1, N2 = tr._build_sim_params(block)
    entries, sq_sum, n_win = [], 0.0, 0
    with torch.no_grad():
        for t in trajs:
            u  = t['u'].to(dev)                     # (T,3)
            q1 = t['q1'].to(dev)                    # (T,3)
            st = t['state_traj'].to(dev)            # (T,6)
            T  = q1.shape[0]
            starts = list(range(0, max(1, T - seg + 1), seg)) or [0]
            t_sq, t_n = 0.0, 0
            for s0 in starts:
                s1 = min(s0 + seg, T)
                x0   = st[s0:s0 + 1]                 # (1,6) teacher-forced
                u_w  = u[s0:s1].unsqueeze(0)         # (1,w,3)
                q1_w = q1[s0:s1].unsqueeze(0)        # (1,w,3)
                res  = simulate(x0, u_w, G, K, C, mh, alpha, beta, gamma,
                                N0, N1, N2, block._P, ts_tensor,
                                bptt_mode='full', return_latents=False)
                err  = (res.Y - q1_w) / sigma
                nmse = float(err.pow(2).mean())
                t_sq += nmse; t_n += 1
                sq_sum += nmse; n_win += 1
            rmse_norm = (t_sq / t_n) ** 0.5
            entries.append({'id': t['id'], 'mse_total': t_sq / t_n,
                            'rmse_total': rmse_norm, 'rmse_ch': None})
    overall = (sq_sum / max(1, n_win)) ** 0.5
    return overall, entries


tr._full_traj_eval = _windowed_val_eval
print('[run_telica] validation selector = windowed normalized-RMS on held-out '
      'windows (D-076); scheduler/best-checkpoint use this, not full-traj OL.')

# ── Hyperparameter overrides ──────────────────────────────────────────────────
# THEORY: FS_NEW = 20000 → D=1 (loader already resamples to 20 kHz);
#         ts_eff = _ts × 1 = 1/20000 s, matching the simulation pipeline.
tr.FS_NEW            = 20_000
tr.SEGMENT_LEN       = 2600    # window/LR diagnostic verdict: 130 ms, params identifiable
tr.EPOCHS            = 40      # multi-trajectory epoch is ~11x the old single-traj epoch
tr.LR                = 1e-2    # window/LR diagnostic verdict: largest stable lr (job 68776)
tr.VALIDATION_INTERVAL = 5     # windowed val loss on VAL_SPECS -> LR scheduler + best checkpoint (D-076)
tr.NORM_MODE         = 'global'
tr.FULL_COVERAGE     = True
tr.OVERLAP_FRACTION  = 0.0
tr.SPLIT_REG_WEIGHT  = 0.0   # no regularization toward Kamtin params on real hardware
tr.W                 = None    # full segment per BPTT step

# ── Training ──────────────────────────────────────────────────────────────────

# ── Plot helpers ──────────────────────────────────────────────────────────────

def _apply_sci_fmt(ax):
    """Format y-axis ticks as e-notation (e.g. 1.23e-04) in metres."""
    ax.yaxis.set_major_formatter(_FuncFormatter(lambda x, _: f'{x:.2e}'))


def _plot_loss_curve(save_dir, traj_id):
    """Load history from the latest .pt in save_dir and plot training loss."""
    import glob as _glob
    pt_files = _glob.glob(os.path.join(save_dir, 'lfr_param_recovery_*.pt'))
    if not pt_files:
        print('  [loss curve] No .pt file found — skipping.')
        return
    latest_pt  = max(pt_files, key=os.path.getmtime)
    data       = torch.load(latest_pt, map_location='cpu', weights_only=False)
    history    = data.get('history', [])
    run_id_pt  = data.get('run_id', 'unknown')
    if not history:
        print('  [loss curve] history key missing in .pt — skipping.')
        return

    epochs = [h['epoch']    for h in history]
    losses = [h['mse_loss'] for h in history]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, losses, color='tab:blue', linewidth=1.0)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Normalised MSE loss')
    ax.set_title(f'Training loss — {traj_id}')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig_path = os.path.join(save_dir, f'loss_curve_{traj_id}_{run_id_pt}.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f'  Loss curve saved: {fig_path}')
    plt.close(fig)


def _plot_traj_comparison(t_s, q1_eval, y_sim, rmse_ch, nrmse_ch, rmse_tot,
                           axes_lbl, traj_id, save_dir, metric, label, run_id,
                           op_label=''):
    """Trajectory comparison (Measured vs FP model).  metric: 'RMS' | 'NRMS'."""
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for i, lbl in enumerate(axes_lbl):
        axs[i].plot(t_s.numpy(), q1_eval[:, i].numpy(),
                    label='Measured', color='tab:blue', linewidth=0.8)
        axs[i].plot(t_s.numpy(), y_sim[:, i].numpy(),
                    label='FP model (sim)', color='tab:orange',
                    linewidth=0.8, linestyle='--')
        axs[i].set_ylabel(f'{lbl} [m]')
        if metric == 'RMS':
            axs[i].set_title(f'{lbl}: RMS = {rmse_ch[i].item():.2e} m', fontsize=9)
        else:
            axs[i].set_title(f'{lbl}: NRMSE = {nrmse_ch[i]:.1f}%', fontsize=9)
        axs[i].legend(loc='upper right', fontsize=8)
        axs[i].grid(True, alpha=0.3)
        _apply_sci_fmt(axs[i])
    axs[-1].set_xlabel('Time [s]')
    fig.suptitle(
        f'Telica gantry \u2014 {op_label} \u2014 {traj_id}\n'
        f'{label.capitalize()}  |  Overall RMS = {rmse_tot:.2e} m'
    )
    fig.tight_layout()
    fig_path = os.path.join(save_dir, f'trajectory_{metric}_{label}_{traj_id}_{run_id}.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f'  Trajectory {metric} ({label}) saved: {fig_path}')
    plt.close(fig)


def _plot_residual(t_s, diff, rmse_ch, nrmse_ch, rmse_tot,
                   axes_lbl, traj_id, save_dir, metric, label, run_id,
                   op_label=''):
    """Residual (y_sim − measured) per channel.  metric: 'RMS' | 'NRMS'."""
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for i, lbl in enumerate(axes_lbl):
        axs[i].plot(t_s.numpy(), diff[:, i].numpy(),
                    color='tab:red', linewidth=0.8, label='Residual')
        axs[i].axhline(0, color='k', linewidth=0.5, linestyle='--')
        axs[i].set_ylabel(f'{lbl} [m]')
        if metric == 'RMS':
            axs[i].set_title(f'{lbl}: RMS = {rmse_ch[i].item():.2e} m', fontsize=9)
        else:
            axs[i].set_title(f'{lbl}: NRMSE = {nrmse_ch[i]:.1f}%', fontsize=9)
        axs[i].grid(True, alpha=0.3)
        _apply_sci_fmt(axs[i])
    axs[-1].set_xlabel('Time [s]')
    fig.suptitle(
        f'Residual \u2014 Telica gantry \u2014 {op_label} \u2014 {traj_id}\n'
        f'{label.capitalize()}  |  Overall RMS = {rmse_tot:.2e} m'
    )
    fig.tight_layout()
    fig_path = os.path.join(save_dir, f'residual_{metric}_{label}_{traj_id}_{run_id}.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f'  Residual {metric} ({label}) saved: {fig_path}')
    plt.close(fig)


def _post_eval(block, traj_spec, label='best', run_id='unknown'):
    """
    Compute per-channel RMS/NRMSE and save trajectory + residual plots.
    label: 'baseline' (initial params) | 'best' (trained params).

    NRMSE thresholds (Schoukens & Ljung, Mech. Sys. Signal Process. 25(7), 2011;
                      Paduart et al., arXiv:1804.10758, 2018):
        THEORY: NRMSE < 15%  → model structure compatible with data
        THEORY: NRMSE > 30%  → structural mismatch (or force-signal problem)
        15–30%: ambiguous — inspect trajectory plot and residual spectrum
    """
    dtype = tr.DTYPE
    log_path = os.path.join(_DATASET_ROOT, traj_spec['file'])
    traj_id  = traj_spec['id']
    op_label = traj_spec.get('label', '')

    print(f'\n{"=" * 60}')
    print(f'Structural validation  [{traj_id}]  [{label}]')
    print(f'{"=" * 60}')

    # Load full trajectory (fresh, undecomposed into segments)
    u_eval, q1_eval, fs = load_telica_log(log_path, dtype=dtype)
    # u_eval: (1, T, 3)    q1_eval: (T, 3)

    # Build initial state from first samples of q1
    x0 = _build_state_traj_logical(
        q1_eval[:2], _P.to(dtype), float(_ts), dtype
    )[:1]   # (1, 6)

    ts_tensor = torch.tensor(float(_ts), dtype=dtype)

    # Simulate full trajectory open-loop
    result = _run_no_grad(block, x0, u_eval, ts_tensor)
    y_sim  = result.Y[0].detach()   # (T, 3)  [m]

    # Trim to minimum length (interpolation may shift end by 1 sample)
    T = min(y_sim.shape[0], q1_eval.shape[0])
    y_sim   = y_sim[:T]
    q1_eval = q1_eval[:T]

    # Per-channel RMS and NRMSE
    diff     = y_sim - q1_eval                                   # (T, 3)  [m]
    rmse_ch  = diff.pow(2).mean(dim=0).sqrt()                    # (3,)    [m]
    sigma_ch = q1_eval.std(dim=0).clamp(min=1e-9)               # (3,)    [m]
    # THEORY: NRMSE = RMSE / std(y_measured) × 100%
    #         (Schoukens & Ljung 2011 — scale-independent structural metric)
    nrmse_ch = (rmse_ch / sigma_ch * 100.0).tolist()            # [%]
    rmse_tot = float(diff.pow(2).mean().sqrt().item())           # [m]

    axes_lbl = ['X1', 'X2', 'Y']
    print(f'\n  {"Ch":<4}  {"RMS [m]":>14}  {"NRMSE [%]":>10}  Verdict')
    print(f'  {"-"*4}  {"-"*14}  {"-"*10}  {"-"*12}')
    for i, lbl in enumerate(axes_lbl):
        n = nrmse_ch[i]
        verdict = 'GOOD (<15%)' if n < 15 else ('AMBIGUOUS' if n < 30 else 'POOR (>30%)')
        print(f'  {lbl:<4}  {rmse_ch[i].item():>14.4e}  {n:>10.2f}  {verdict}')
    print(f'\n  Overall RMS : {rmse_tot:.4e} m')
    print(f'  Thresholds  : NRMSE < 15% compatible | > 30% structural mismatch')

    # ── Save eval data for replotting without rerunning ───────────────────────
    t_s = torch.arange(T).float() / fs
    os.makedirs(_SAVE_DIR, exist_ok=True)
    eval_data_path = os.path.join(_SAVE_DIR, f'eval_data_{label}_{traj_id}_{run_id}.pt')
    torch.save({
        't_s':      t_s,
        'q1_eval':  q1_eval,
        'y_sim':    y_sim,
        'diff':     diff,
        'rmse_ch':  rmse_ch,
        'nrmse_ch': nrmse_ch,
        'rmse_tot': rmse_tot,
        'fs':       fs,
        'label':    label,
        'traj_id':  traj_id,
        'run_id':   run_id,
        'axes_lbl': axes_lbl,
    }, eval_data_path)
    print(f'  Eval data saved: {eval_data_path}')

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_traj_comparison(t_s, q1_eval, y_sim, rmse_ch, nrmse_ch, rmse_tot,
                          axes_lbl, traj_id, _SAVE_DIR, 'RMS',  label, run_id, op_label=op_label)
    _plot_traj_comparison(t_s, q1_eval, y_sim, rmse_ch, nrmse_ch, rmse_tot,
                          axes_lbl, traj_id, _SAVE_DIR, 'NRMS', label, run_id, op_label=op_label)
    _plot_residual(t_s, diff, rmse_ch, nrmse_ch, rmse_tot,
                   axes_lbl, traj_id, _SAVE_DIR, 'RMS',  label, run_id, op_label=op_label)
    _plot_residual(t_s, diff, rmse_ch, nrmse_ch, rmse_tot,
                   axes_lbl, traj_id, _SAVE_DIR, 'NRMS', label, run_id, op_label=op_label)


@torch._dynamo.disable
def _run_closed_loop(block, x0, r, u_ff, ctrl, ts_tensor, dtype):
    """
    Full-trajectory closed-loop simulation (D-074): at every 20 kHz step
        y_k    = position output from the current model state
        e_k    = r_k - y_k                      tracking error [m]
        i_fb_k = ctrl.step(e_k)                 feedback current [A]
        u_k    = u_ff_k + i_fb_k * Kt           total stage force [N]
        x_{k+1} = rk4_step(x_k, u_k)
    No gradients. Timing convention: controller acts on the same-sample error
    (no computational delay); output y has no feedthrough (y = Cy x), so the
    loop is well-posed.

    Returns (y_sim (T,3) [m] stage, i_fb_sim (T,3) [A]).
    """
    from lpv_lfr_baseline.core.lfr_simulate import rk4_step as _rk4

    with torch.no_grad():
        G, K, C, mh, alpha, beta, gamma, N0, N1, N2 = tr._build_sim_params(block)
        P = block._P.to(dtype)               # (3,3) stage <-> logical
        T = r.shape[0]
        y_sim    = torch.empty(T, 3, dtype=dtype)
        i_fb_sim = torch.empty(T, 3, dtype=dtype)
        a_to_n   = torch.tensor(_A_TO_N, dtype=dtype)

        ctrl.reset()
        x = x0                                # (1, 6) logical
        for k in range(T):
            y_k = x[:, :3] @ P                # logical positions -> stage (1,3)
            y_sim[k] = y_k[0]
            e_k = (r[k] - y_k[0]).numpy()     # (3,) [m]
            i_k = ctrl.step(e_k)              # (3,) [A]
            i_fb_sim[k] = torch.from_numpy(i_k)
            u_stage   = u_ff[k] + i_fb_sim[k] * a_to_n        # (3,) [N]
            u_logical = (u_stage.unsqueeze(0) @ P.T)          # (1,3)
            x, _, _, _ = _rk4(x, u_logical, G, K, C, mh,
                              alpha, beta, gamma, N0, N1, N2, ts_tensor)
        return y_sim, i_fb_sim


def _plot_feedback_current(t_s, i_fb_sim, i_fb_log, i_fb_replay,
                           traj_id, save_dir, label, run_id, op_label=''):
    """Feedback current: logged vs closed-loop sim vs replay-of-measured-error."""
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for i, lbl in enumerate(['X1', 'X2', 'Y']):
        axs[i].plot(t_s.numpy(), i_fb_log[:, i].numpy(),
                    'k', linewidth=0.8, label='logged MF230 [A]')
        axs[i].plot(t_s.numpy(), i_fb_replay[:, i].numpy(),
                    color='tab:green', linewidth=0.8, alpha=0.7,
                    label='controller(measured error) [A]')
        axs[i].plot(t_s.numpy(), i_fb_sim[:, i].numpy(),
                    color='tab:orange', linewidth=0.8, alpha=0.8,
                    linestyle='--', label='controller(model error), closed loop [A]')
        axs[i].set_ylabel(f'{lbl} [A]')
        axs[i].grid(True, alpha=0.3)
        if i == 0:
            axs[i].legend(loc='upper right', fontsize=8)
    axs[-1].set_xlabel('Time [s]')
    fig.suptitle(
        f'Feedback current | Telica gantry | {op_label} | {traj_id} ({label})\n'
        f'green vs black: controller+wiring check | orange vs black: model quality in the loop'
    )
    fig.tight_layout()
    fig_path = os.path.join(save_dir, f'feedback_current_cl_{label}_{traj_id}_{run_id}.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f'  Feedback current plot saved: {fig_path}')
    plt.close(fig)


def _post_eval_cl(block, traj_spec, label='best', run_id='unknown'):
    """
    Closed-loop structural validation (D-074): initial condition only, then
    full trajectory with u = u_ff + K(r - y_model). Complements the open-loop
    _post_eval: open loop is the harshest test, closed loop is the
    control-relevant one (the controller suppresses model error inside the
    loop bandwidth, so CL numbers are always better -- report both).
    """
    dtype = tr.DTYPE
    log_path = os.path.join(_DATASET_ROOT, traj_spec['file'])
    traj_id  = traj_spec['id']
    op_label = traj_spec.get('label', '')

    print(f'\n{"=" * 60}')
    print(f'Closed-loop validation  [{traj_id}]  [{label}]')
    print(f'{"=" * 60}')

    d = load_telica_log_cl(log_path, dtype=dtype)
    r, q1_eval, u_ff, i_fb_log, fs = d['r'], d['q1'], d['u_ff'], d['i_fb'], d['fs']

    # Initial state from first measured samples (same as open-loop eval)
    x0 = _build_state_traj_logical(
        q1_eval[:2], _P.to(dtype), float(_ts), dtype
    )[:1]   # (1, 6)
    ts_tensor = torch.tensor(float(_ts), dtype=dtype)

    ctrl = TelicaFeedbackController()
    y_sim, i_fb_sim = _run_closed_loop(block, x0, r, u_ff, ctrl, ts_tensor, dtype)

    # Wiring check independent of the model: replay measured error
    ctrl.reset()
    i_fb_replay = torch.tensor(ctrl.filt((r - q1_eval).numpy()), dtype=dtype)

    T = min(y_sim.shape[0], q1_eval.shape[0])
    y_sim, q1_c = y_sim[:T], q1_eval[:T]

    diff     = y_sim - q1_c
    rmse_ch  = diff.pow(2).mean(dim=0).sqrt()
    sigma_ch = q1_c.std(dim=0).clamp(min=1e-9)
    nrmse_ch = (rmse_ch / sigma_ch * 100.0).tolist()
    rmse_tot = float(diff.pow(2).mean().sqrt().item())

    axes_lbl = ['X1', 'X2', 'Y']
    print(f'\n  {"Ch":<4}  {"RMS [m]":>14}  {"NRMSE [%]":>10}')
    print(f'  {"-"*4}  {"-"*14}  {"-"*10}')
    for i, lbl in enumerate(axes_lbl):
        print(f'  {lbl:<4}  {rmse_ch[i].item():>14.4e}  {nrmse_ch[i]:>10.2f}')
    print(f'\n  Overall RMS : {rmse_tot:.4e} m')
    print('  NOTE: closed-loop errors are suppressed inside the loop bandwidth;')
    print('        always read next to the open-loop numbers above.')

    t_s = torch.arange(T).float() / fs
    os.makedirs(_SAVE_DIR, exist_ok=True)
    eval_data_path = os.path.join(_SAVE_DIR, f'eval_data_cl_{label}_{traj_id}_{run_id}.pt')
    torch.save({
        't_s': t_s, 'q1_eval': q1_c, 'y_sim': y_sim, 'diff': diff,
        'rmse_ch': rmse_ch, 'nrmse_ch': nrmse_ch, 'rmse_tot': rmse_tot,
        'i_fb_sim': i_fb_sim[:T], 'i_fb_log': i_fb_log[:T],
        'i_fb_replay': i_fb_replay[:T],
        'fs': fs, 'label': f'cl_{label}', 'traj_id': traj_id,
        'run_id': run_id, 'axes_lbl': axes_lbl,
    }, eval_data_path)
    print(f'  Eval data saved: {eval_data_path}')

    _plot_traj_comparison(t_s, q1_c, y_sim, rmse_ch, nrmse_ch, rmse_tot,
                          axes_lbl, traj_id, _SAVE_DIR, 'RMS',  f'cl_{label}', run_id, op_label=op_label)
    _plot_residual(t_s, diff, rmse_ch, nrmse_ch, rmse_tot,
                   axes_lbl, traj_id, _SAVE_DIR, 'RMS',  f'cl_{label}', run_id, op_label=op_label)
    _plot_feedback_current(t_s, i_fb_sim[:T], i_fb_log[:T], i_fb_replay[:T],
                           traj_id, _SAVE_DIR, label, run_id, op_label=op_label)


if __name__ == '__main__':
    run_id = os.environ.get('SLURM_JOB_ID') or datetime.now().strftime('%Y%m%d_%H%M%S')

    # Baseline evaluation — initial parameters (nominal Kamtin FP values) over
    # the SAME eval set as the trained model, so every metric exists in four
    # variants: {baseline, best} x {open loop, closed loop}. Expected: baseline
    # diverges in closed loop (nominal model + real controller, see D-074).
    _init_block = _lfr_pb.ParameterizedLFRBlock(RMSE_baseline=1.0).to(dtype=tr.DTYPE)
    for _spec in EVAL_SPECS:
        _post_eval(_init_block, _spec, label='baseline', run_id=run_id)
        _post_eval_cl(_init_block, _spec, label='baseline', run_id=run_id)
    del _init_block

    block = tr.train(
        epochs=tr.EPOCHS,
        lr=tr.LR,
        save_dir=_SAVE_DIR,
        split_reg_weight=0.0,
        norm_mode=tr.NORM_MODE,
    )

    _plot_loss_curve(_SAVE_DIR, TRAJ_SPECS[0]['id'])
    # Final evaluation of the best model: open loop (harshest) and closed
    # loop (control-relevant) over train sample + validation + test (D-075).
    for _spec in EVAL_SPECS:
        _post_eval(block, _spec, label='best', run_id=run_id)
        _post_eval_cl(block, _spec, label='best', run_id=run_id)
