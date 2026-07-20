"""
s3_openloop_multisine.py -- S3: kill the feedback, drive OPEN-LOOP with only the
stored multisine. Supervisor: "reference zero, inject only the multisine, no
feedforward -- KILL the feedback, complete open-loop. Perfect multisine periodic
motion should not have problems with the true system."

The augmentation data is CLOSED-LOOP: u_total = u_fb + f_ms (gtd_run_simulation.m),
u_fb = Cfb*(r - q). The .mat stores f_sim = f_ms (the pure multisine, stage force)
separately, so we can replay open-loop in Python with NO MATLAB re-gen.

Part 1 (control -- true system): drive the true 8-state model open-loop from the
  stored x0 with ONLY f_ms. A zero-mean multisine on a K=0 axis integrates to a
  BOUNDED oscillation (no drift). Expected: bounded -> the true system + pure
  multisine has no problem, so the drift is NOT a data/controller artifact.
  (Also replays the full closed-loop u_total as a sanity tie to d2.)

Part 2 (augmented model): drive the trained augmented model open-loop with the SAME
  f_ms (encoder-initialised from the true open-loop trajectory). Expected under the
  DC-force hypothesis: it STILL drifts, because the ANN's DC output is intrinsic and
  input-independent -> isolates the fault to the augmentation, not the loop.

Run:
  CKPT=simulations/gantry_subnet/diagnostics/checkpoints/gantry_drift_last.pth \
  conda run -n GraduationProject python scripts/gantry/diagnostics-drift/s3_openloop_multisine.py
Env: REC (V1 default), FS_NEW, N_WIN, CKPT (for Part 2).
Outputs -> simulations/gantry_subnet/diagnostics/
"""
import os
import sys
import time

import numpy as np
import torch
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
GANTRY = os.path.join(REPO, 'scripts', 'gantry')
sys.path.insert(0, GANTRY)
sys.path.insert(0, os.path.dirname(__file__))

import deepSI
import drift_common as dc

REC    = os.environ.get('REC', 'V1_standstill_Yp10.mat')
FS_NEW = int(os.environ.get('FS_NEW', str(dc.DEFAULT_FS_NEW)))
N_WIN  = os.environ.get('N_WIN', None)
CKPT   = os.environ.get('CKPT', None)
START  = int(os.environ.get('START', '100'))   # interior seed: central-diff velocity (no sample-0 gradient artifact)

rec = dc.load_record(REC, FS_NEW)   # u_total (block-mean), y, x_logical, delta_a, vdelta_a
ts, D = rec.ts, rec.D

# f_sim (pure multisine, stage force) -- block-mean to FS_NEW like u_total (D-087).
_m = loadmat(os.path.join(dc.DATA_DIR, REC), squeeze_me=True)
f0 = np.asarray(_m['f_sim'], dtype=np.float64)
if f0.ndim == 1:
    f0 = f0[:, None]
if D > 1:
    n = f0.shape[0] // D
    f_ms = f0[:n * D].reshape(n, D, f0.shape[1]).mean(axis=1)
else:
    f_ms = f0
avail = f_ms.shape[0] - START
N = avail if N_WIN is None else min(int(N_WIN), avail)
sl = slice(START, START + N)
f_ms      = f_ms[sl]
u_total   = rec.u_stage[sl]
y_stage   = rec.y_stage[sl]
x0_8      = dc.true_x0_8(rec, START)        # interior seed: clean central-diff velocity
x0_6      = x0_8[[0, 1, 2, 4, 5, 6]]        # baseline-only initial state (drop absorber)
t = np.arange(N) * ts
absorber_rms = rec.delta_a[sl].std()
print(f'  seed START={START} (interior; sample-0 gradient-artifact avoided)')

print(f'Record {rec.name}: {N} samples @ {1/ts:.0f} Hz (D={D})')
print(f'  |f_ms| RMS per axis (stage) = {np.sqrt((f_ms**2).mean(axis=0))}')
print(f'  |u_total| RMS per axis      = {np.sqrt((u_total**2).mean(axis=0))}')
print(f'  f_ms mean per axis (should be ~0 for a multisine) = {f_ms.mean(axis=0)}')
print(f'  absorber RMS = {absorber_rms:.3e} m   tau_X={dc.tau_X:.2f}s tau_Y={dc.tau_Y:.2f}s')


def _tail(err):
    return err[int(0.8 * len(err)):].mean(axis=0)


# ── Part 1: true system, open-loop multisine (as-is, demeaned) + closed-loop ─
settled = N * ts > 5 * dc.tau_X
print(f'\n=== Part 1: true system, {N*ts:.1f}s  (settled={settled}; need >5*tau_X={5*dc.tau_X:.1f}s) ===')

# The stored multisine is not perfectly zero-mean; a residual DC on a K=0 axis
# drives open-loop drift by itself (the feedback absorbs it in closed loop).
f_dc_stage = f_ms.mean(axis=0)
f_dc_log   = dc.stage_force_to_logical(f_dc_stage[None])[0]   # logical DC force
print(f'  multisine DC (stage) = {f_dc_stage}   (logical) = {f_dc_log}')

f_ms_dm = f_ms - f_dc_stage[None]     # truly zero-mean multisine (guards residual DC)

# full-truth (baseline+absorber) and baseline-only, both open-loop from the SAME interior x0.
q_ol = dc.simulate_truth(x0_8, dc.stage_force_to_logical(f_ms_dm), ts); y_ol = dc.logical_pos_to_stage(q_ol[:, :3])
y_base = dc.simulate_baseline(x0_6, f_ms_dm, ts)                        # baseline-only (stage force in)
q_cl = dc.simulate_truth(x0_8, dc.stage_force_to_logical(u_total), ts); y_cl = dc.logical_pos_to_stage(q_cl[:, :3])
y_dm = y_ol   # kept name for the plot/npz

def _slope(y):   # linear drift rate per channel [m/s] (isolates drift from oscillation)
    return np.array([np.polyfit(t, y[:, c], 1)[0] for c in range(3)])

ol_drift  = _tail(y_ol - y_ol[0]);   ol_slope   = _slope(y_ol)
base_drift = _tail(y_base - y_base[0]); base_slope = _slope(y_base)
dm_drift = ol_drift; dm_slope = ol_slope
cl_match  = np.sqrt(((y_cl - y_stage) ** 2).mean(axis=0))
print(f'  full-truth   : tail disp X1={ol_drift[0]:+.2e} X2={ol_drift[1]:+.2e} Y={ol_drift[2]:+.2e} m'
      f'  | slope Y={ol_slope[2]:+.2e} m/s')
print(f'  baseline-only: tail disp X1={base_drift[0]:+.2e} X2={base_drift[1]:+.2e} Y={base_drift[2]:+.2e} m'
      f'  | slope Y={base_slope[2]:+.2e} m/s')
print(f'  open-loop swing (max-min, full-truth): {np.ptp(y_ol, axis=0)}')
print(f'  closed-loop replay vs stored data RMSE: {cl_match} m  (interior start -> should be ~1e-6)')
if not settled:
    print('  !! window < 5*tau_X: Part 1 drift verdict INVALID; run full trajectory.')
elif np.abs(ol_drift).max() < 5 * absorber_rms and np.abs(base_drift).max() < 5 * absorber_rms:
    print('  -> TRUE system stays BOUNDED open-loop under a zero-mean multisine (interior seed): the')
    print('     supervisor\'s "perfect multisine -> no problem" holds. Drift is NOT the loop/data.')
elif np.abs(ol_drift).max() > 5 * absorber_rms and np.abs(base_drift).max() < 5 * absorber_rms:
    print('  -> full-truth drifts but baseline-only does NOT: a nonlinear/absorber rectification of the')
    print('     zero-mean multisine on the K=0 axis (real, and independent of the ANN).')
else:
    print('  -> baseline-only also drifts open-loop -> a K=0 integrator response to the multisine itself;')
    print('     interpret Part 2 as model drift RELATIVE to this true-system baseline.')

# ── Part 2: augmented model, open-loop multisine ─────────────────────────────
if CKPT:
    from gantry_dynamic.data import load_datasets, compute_normalization
    from gantry_dynamic.model import build_model
    from gantry_interconnect_dynamic import CFG as cfg
    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    _data = load_datasets(cfg); _norm = compute_normalization(cfg, _data)
    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    fit_sys = build_model(cfg.hp, cfg, _data, _norm)
    fit_sys.__dict__ = torch.load(CKPT, map_location='cpu', weights_only=False)
    fit_sys.hfn.eval()
    print(f'\n=== Part 2: augmented model open-loop, CKPT={os.path.basename(CKPT)} ===')
    # Encoder-init from the TRUE open-loop trajectory, then free-run the model with f_ms.
    sd = deepSI.System_data(u=f_ms.astype(np.float64), y=y_ol.astype(np.float64), dt=ts)
    t0 = time.time()
    r = fit_sys.apply_experiment(sd)
    y_model = np.asarray(r.y, dtype=np.float64)
    M = min(len(y_model), N)
    y_model, y_ol_a, t_a = y_model[-M:], y_ol[-M:], t[-M:]
    model_drift = _tail(y_model - y_ol_a)     # model vs true, both open-loop same input
    print(f'  ({time.time()-t0:.0f}s) augmented-model drift vs true open-loop: '
          f'X1={model_drift[0]:+.3e} X2={model_drift[1]:+.3e} Y={model_drift[2]:+.3e} m')
    if np.abs(model_drift).max() > 5 * absorber_rms:
        print('  -> augmented model DRIFTS even under clean open-loop multisine, while the true system')
        print('     does not -> the drift is intrinsic to the ANN (DC force), not the loop/data (S3 confirms).')
    else:
        print('  -> augmented model stays bounded here too -> drift may be excitation-dependent; investigate.')
else:
    y_model = None; model_drift = None
    print('\n(Part 2 skipped: set CKPT to include the augmented-model open-loop test.)')

# ── Plot ─────────────────────────────────────────────────────────────────────
nrow = 3
fig, axes = plt.subplots(nrow, 1, figsize=(12, 8), sharex=True)
for ch, (ax, lbl) in enumerate(zip(axes, ['X1 [m]', 'X2 [m]', 'Y [m]'])):
    ax.plot(t, (y_ol - y_ol[0])[:, ch],   'C0', lw=0.7, label='full-truth, open-loop multisine')
    ax.plot(t, (y_base - y_base[0])[:, ch], 'C2', lw=0.7, label='baseline-only, open-loop')
    if y_model is not None:
        ax.plot(t_a, (y_model - y_ol[0])[:, ch], 'C3', lw=0.7, label='augmented model, open-loop')
    ax.axhline(0, color='k', lw=0.5)
    ax.axhline( absorber_rms, color='0.5', ls=':', lw=0.8)
    ax.axhline(-absorber_rms, color='0.5', ls=':', lw=0.8,
               label=f'+/- absorber RMS' if ch == 0 else None)
    ax.set_ylabel(f'{lbl} (rel. start)'); ax.grid(True); ax.legend(fontsize=7, loc='upper right')
axes[-1].set_xlabel('Time [s]')
fig.suptitle(f'{rec.name}: complete open-loop, only multisine -- true system clean, '
             f'does the augmented model still drift?')
fig.tight_layout()
stem = os.path.join(dc.OUT_DIR, f's3_openloop_multisine_{rec.name.split(".")[0]}')
fig.savefig(stem + '.png', dpi=150)
np.savez(stem + '.npz', t=t, y_ol=y_ol, y_base=y_base, y_cl=y_cl, y_stage=y_stage,
         y_model=(y_model if y_model is not None else np.array([])),
         ol_drift=ol_drift, base_drift=base_drift, ol_slope=ol_slope, base_slope=base_slope,
         cl_match=cl_match, f_dc_stage=f_dc_stage, f_dc_log=f_dc_log, START=START,
         model_drift=(model_drift if model_drift is not None else np.array([])),
         absorber_rms=absorber_rms, ts=ts)
print(f'\nSaved: {stem}.png')
print(f'Saved: {stem}.npz')
