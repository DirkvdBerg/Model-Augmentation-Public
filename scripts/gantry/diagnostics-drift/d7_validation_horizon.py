"""
d7_validation_horizon.py -- S1: judge the trained model at the 0.1 s TRAINING horizon,
not only the 12 s free-run. Supervisor: "training is 0.1 s so validation should also be
good for 0.1 s; already divergence within 0.1 s in the image" + "increase nf_seconds to
see the dominant dynamics -- find the sweet spot."

Two parts on the drifted checkpoint (gantry_drift_last.pth):

S1a -- WITHIN-WINDOW error growth (mirrors training's per-window re-init).
  For each non-overlapping 0.1 s window, the encoder re-initialises the state from the
  window's own I/O history (exactly as the truncated training loss does), then the model
  free-runs T_win samples. We record |error(t)| vs time-into-window, averaged over windows.
  Distinguishes:
    * fast OSCILLATORY error (absorber mismatch): |error(t)| ~ flat, order ~ val nf-RMS,
      visible already inside 0.1 s  -> "diverges in 0.1 s" = in-horizon model error, NOT drift.
    * slow INTEGRATOR drift: negligible over 0.1 s (needs seconds); only shows in S1b.

S1b -- RMS vs HORIZON (the nf sweet-spot picture).
  One long free-run from a single encoder init; cumulative free-run RMS at horizons
  T = 0.05..12 s. Shows where the drift enters (RMS flat until ~seconds, then rises) and
  hence the runtime-vs-dynamics tradeoff for choosing nf_seconds.

Run:
  CKPT=simulations/gantry_subnet/diagnostics/checkpoints/gantry_drift_last.pth \
  conda run -n GraduationProject python scripts/gantry/diagnostics-drift/d7_validation_horizon.py
Env: CKPT (required for a meaningful result), N_WINDOWS (cap S1a windows), REC (V1 default).
Outputs -> simulations/gantry_subnet/diagnostics/
"""
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
GANTRY = os.path.join(REPO, 'scripts', 'gantry')
sys.path.insert(0, GANTRY)
sys.path.insert(0, os.path.dirname(__file__))

import deepSI
import drift_common as dc
from gantry_dynamic.data import load_datasets, compute_normalization
from gantry_dynamic.model import build_model, get_encoder_dims
from gantry_interconnect_dynamic import CFG as cfg

CKPT      = os.environ.get('CKPT', None)
N_WINDOWS = os.environ.get('N_WINDOWS', None)

# ── Build pipeline + load the drifted checkpoint ─────────────────────────────
np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
data = load_datasets(cfg)
norm = compute_normalization(cfg, data)
np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
fit_sys = build_model(cfg.hp, cfg, data, norm)
if CKPT:
    fit_sys.__dict__ = torch.load(CKPT, map_location='cpu', weights_only=False)
    print(f'Loaded CKPT {os.path.basename(CKPT)}; bestfit={getattr(fit_sys,"bestfit",float("nan")):.3e}')
else:
    print('No CKPT: untrained model (zero ANN) -- machinery check only.')
fit_sys.hfn.eval()

na, nb, _, _ = get_encoder_dims(cfg.hp, cfg)
warm = max(na, nb)
ts   = cfg.ts_new
T_win = int(round(cfg.nf_seconds / ts))    # 0.1 s -> 400 samples @ 4 kHz
absorber_rms = data.val_x_aug[:, 0].std()

v1 = data.val_data
u  = np.asarray(v1.u, dtype=np.float64)
y  = np.asarray(v1.y, dtype=np.float64)
Ntot = len(u)
print(f'V1: {Ntot} samples @ {1/ts:.0f} Hz  warm={warm}  T_win={T_win} ({cfg.nf_seconds}s)  '
      f'absorber RMS={absorber_rms:.3e} m')


def _freerun(u_seg, y_seg):
    """apply_experiment on a slice; returns yhat aligned to the last len-warm samples."""
    sd = deepSI.System_data(u=u_seg, y=y_seg, dt=ts)
    r  = fit_sys.apply_experiment(sd)
    return np.asarray(r.y, dtype=np.float64)


# ── S1a: within-window error, encoder re-init per window ─────────────────────
starts = list(range(warm, Ntot - T_win, T_win))
if N_WINDOWS is not None:
    step = max(1, len(starts) // int(N_WINDOWS))
    starts = starts[::step][:int(N_WINDOWS)]
print(f'\nS1a: {len(starts)} non-overlapping 0.1 s windows, encoder re-init each ...')
t0 = time.time()
err_stack = []     # (nwin, T_win, 3)
for s in starts:
    sl = slice(s - warm, s + T_win)
    yhat = _freerun(u[sl], y[sl])
    yh = yhat[-T_win:]
    err_stack.append(np.abs(yh - y[s:s + T_win]))
err_stack = np.asarray(err_stack)
err_t = err_stack.mean(axis=0)          # (T_win, 3) mean |err| vs time-into-window
t_win = np.arange(T_win) * ts * 1e3     # ms
print(f'  done ({time.time()-t0:.0f}s)')
print(f'  within-window mean |err| at t=0 / 0.05s / 0.1s  (X1,X2,Y) [m]:')
for lbl, k in [('t=0', 0), ('0.05s', T_win // 2), ('0.10s', T_win - 1)]:
    print(f'    {lbl:6s}: {err_t[k]}')
grow = err_t[-1] / (err_t[0] + 1e-30)
print(f'  growth factor over window (end/start): X1={grow[0]:.2f} X2={grow[1]:.2f} Y={grow[2]:.2f}')
print(f'  reference: absorber RMS = {absorber_rms:.3e} m')
print('  -> flat & ~absorber level => in-horizon oscillatory error (not drift);')
print('     monotone growth within 0.1 s => a fast error too; strong ramp only over seconds => S1b.')

# ── S1b: RMS vs horizon from one long free-run ───────────────────────────────
print('\nS1b: single long free-run, cumulative RMS vs horizon ...')
t0 = time.time()
yhat_full = _freerun(u, y)
M = len(yhat_full)
err_full = yhat_full - y[-M:]
print(f'  full free-run done ({time.time()-t0:.0f}s, {M} samples = {M*ts:.1f}s)')
horizons_s = [h for h in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 12.0) if h < M * ts] + [M * ts]
print(f"  {'horizon':>9s}  {'RMS X1':>10s} {'RMS X2':>10s} {'RMS Y':>10s}")
rms_rows = []
for h in horizons_s:
    k = min(M, int(round(h / ts)))
    rms = np.sqrt((err_full[:k] ** 2).mean(axis=0))
    rms_rows.append((h, rms))
    print(f'  {h:8.2f}s  {rms[0]:>10.3e} {rms[1]:>10.3e} {rms[2]:>10.3e}')
print(f'  reference: absorber RMS = {absorber_rms:.3e} m  (nf_seconds now = {cfg.nf_seconds}s)')

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ch, lbl in enumerate(['X1', 'X2', 'Y']):
    axes[0].plot(t_win, err_t[:, ch] * 1e6, label=lbl)
axes[0].axhline(absorber_rms * 1e6, color='0.5', ls=':', label=f'absorber RMS ({absorber_rms*1e6:.1f} um)')
axes[0].set_xlabel('time into 0.1 s window [ms]'); axes[0].set_ylabel('mean |error| [um]')
axes[0].set_title('S1a: within-window error (encoder re-init per window)')
axes[0].grid(True); axes[0].legend(fontsize=8)
hs = [r[0] for r in rms_rows]
for ch, lbl in enumerate(['X1', 'X2', 'Y']):
    axes[1].loglog(hs, [r[1][ch] for r in rms_rows], 'o-', label=lbl)
axes[1].axhline(absorber_rms, color='0.5', ls=':', label='absorber RMS')
axes[1].axvline(cfg.nf_seconds, color='C3', ls='--', label=f'nf={cfg.nf_seconds}s')
axes[1].set_xlabel('free-run horizon [s]'); axes[1].set_ylabel('cumulative RMS [m]')
axes[1].set_title('S1b: RMS vs horizon (nf sweet-spot)')
axes[1].grid(True, which='both'); axes[1].legend(fontsize=8)
fig.suptitle(f'S1: {os.path.basename(CKPT) if CKPT else "untrained"} -- '
             f'model quality at the 0.1 s training horizon vs the long free-run')
fig.tight_layout()
stem = os.path.join(dc.OUT_DIR, 'd7_validation_horizon_V1')
fig.savefig(stem + '.png', dpi=150)
np.savez(stem + '.npz', t_win_ms=t_win, err_t=err_t, err_stack=err_stack,
         horizons_s=np.array([r[0] for r in rms_rows]),
         rms=np.array([r[1] for r in rms_rows]), absorber_rms=absorber_rms,
         T_win=T_win, ts=ts, nf_seconds=cfg.nf_seconds)
print(f'\nSaved: {stem}.png')
print(f'Saved: {stem}.npz')
