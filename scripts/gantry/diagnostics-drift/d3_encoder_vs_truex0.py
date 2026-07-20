"""
d3_encoder_vs_truex0.py -- is the augmentation's ENCODER x0 the drift seed? (A2)

The augmentation free-runs the whole trajectory from a single x0 produced by the
SUBNET reconstructability encoder (a linear map from the I/O history window that
DIFFERENTIATES the position window to recover velocities). Param recovery instead
seeds from the true stored state. This diagnostic isolates the encoder:

  1. Build the SAME pipeline as the real run (imports CFG) and call the UNTRAINED
     encoder to get its x0 estimate at the first full-window sample K0.
  2. Compare it component-by-component to the TRUE state at K0 -- especially the
     velocities on the K=0 axes (X idx0, Y idx2), where a reconstruction error
     integrates.
  3. Free-run the baseline from the encoder x0 vs from the true x0 over the whole
     trajectory and compare the K=0 drift.

Readout:
  encoder velocity error on X/Y is nonzero and its free-run drifts >> the true-x0
  free-run -> the encoder x0 seeds the K=0 drift (hypothesis A2); the settled
  offset should track tau*dv of the encoder velocity error. If both free-runs
  drift equally, the encoder x0 is NOT the differentiator and the drift is
  structural / ANN-side (hypothesis B).

Run (builds the model; a few minutes):
  conda run -n GraduationProject python scripts/gantry/diagnostics-drift/d3_encoder_vs_truex0.py
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
sys.path.insert(0, GANTRY)                       # gantry_dynamic + entry file
sys.path.insert(0, os.path.dirname(__file__))    # drift_common

import drift_common as dc
from gantry_dynamic.data import load_datasets, compute_normalization
from gantry_dynamic.model import build_model, get_encoder_dims
from gantry_dynamic.diagnostics import encoder_init_state
from gantry_interconnect_dynamic import CFG as cfg   # exact run config (no training on import)

N_WIN = os.environ.get('N_WIN', None)   # cap the sim length for a quick check; None = full

# ── Build the pipeline exactly as the real run does (seed, data, norm, model) ─
np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
data = load_datasets(cfg)
norm = compute_normalization(cfg, data)
np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
fit_sys = build_model(cfg.hp, cfg, data, norm)

_na, _nb, _na_right, _nb_right = get_encoder_dims(cfg.hp, cfg)
K0 = max(_na, _nb)
ts = cfg.ts_new

# ── Encoder x0 (normalized -> physical) at K0 for V1 ─────────────────────────
x0_enc_norm = encoder_init_state(fit_sys, data.val_data, K0, _na, _nb, _na_right, _nb_right, cfg)
x0_enc = x0_enc_norm * norm.std_x.flatten() + norm.x_mean.flatten()   # (6,) physical logical
x0_true = data.val_x_logical[K0].astype(np.float64)                    # (6,) true logical

names = ['X', 'Theta', 'Y', 'dX', 'dTheta', 'dY']
print(f'\n=== Encoder x0 vs true x0 at K0={K0} ({K0*ts*1e3:.1f} ms), V1 ===')
print(f"  {'state':8s} {'encoder':>13s} {'true':>13s} {'error':>13s}")
for i, nm in enumerate(names):
    print(f'  {nm:8s} {x0_enc[i]:>13.5e} {x0_true[i]:>13.5e} {x0_enc[i]-x0_true[i]:>13.3e}')
dv_enc = x0_enc[3:] - x0_true[3:]                      # velocity reconstruction error
pred_X = dc.tau_X * dv_enc[0]                          # THEORY: settled K=0 offset = tau*dv
pred_Y = dc.tau_Y * dv_enc[2]
print(f'  velocity error dv (X,Th,Y) = {dv_enc}')
print(f'  -> predicted settled offset  tau_X*dvX={pred_X:+.3e} m   tau_Y*dvY={pred_Y:+.3e} m')

# ── Free-run baseline from encoder x0 vs true x0 over the trajectory ─────────
u_stage = np.asarray(data.val_data.u, dtype=np.float64)
y_stage = np.asarray(data.val_data.y, dtype=np.float64)
Ntot = u_stage.shape[0] - K0 if N_WIN is None else min(int(N_WIN), u_stage.shape[0] - K0)
sl = slice(K0, K0 + Ntot)
u_seg, y_seg = u_stage[sl], y_stage[sl]
t = np.arange(Ntot) * ts
absorber_rms = data.val_x_aug[:, 0].std()

t0 = time.time()
y_enc  = dc.simulate_baseline(x0_enc,  u_seg, ts)
y_true = dc.simulate_baseline(x0_true, u_seg, ts)
print(f'\n  two baseline free-runs ({time.time()-t0:.0f}s, {Ntot*ts:.1f} s each)')
err_enc, err_true = y_enc - y_seg, y_true - y_seg
tail = slice(int(0.8 * Ntot), Ntot)
te, tt = err_enc[tail].mean(axis=0), err_true[tail].mean(axis=0)
print('\n=== K=0 tail-mean offset: seeded from encoder x0 vs true x0 ===')
print(f"  {'chan':4s} {'from encoder':>14s} {'from true':>14s}")
for c, lbl in enumerate(['X1', 'X2', 'Y']):
    print(f'  {lbl:4s} {te[c]:>14.3e} {tt[c]:>14.3e}')
print(f'  absorber RMS = {absorber_rms:.3e} m')

settle = 5.0 * dc.tau_X
if Ntot * ts < settle:
    print(f'  !! window {Ntot*ts:.1f}s < 5*tau_X={settle:.1f}s: not settled, verdict INVALID (full run needed).')
elif np.abs(te).max() > 3 * max(np.abs(tt).max(), absorber_rms):
    print('  -> encoder-seeded free-run drifts far beyond the true-x0 run: the encoder x0 seeds the')
    print('     K=0 drift (A2). A better x0 (or re-estimation / multiple shooting) removes it.')
else:
    print('  -> encoder-seeded and true-x0 free-runs drift alike: encoder x0 is NOT the differentiator;')
    print('     the drift is structural / ANN-side (B).')

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
for ch, (ax, lab) in enumerate(zip(axes, ['X1 [m]', 'X2 [m]', 'Y [m]'])):
    ax.plot(t, err_enc[:, ch],  'C3', lw=0.7, label='from encoder x0')
    ax.plot(t, err_true[:, ch], 'C0', lw=0.7, label='from true x0')
    ax.axhline(0, color='k', lw=0.5)
    ax.axhline( absorber_rms, color='0.5', ls=':', lw=0.8)
    ax.axhline(-absorber_rms, color='0.5', ls=':', lw=0.8,
               label='+/- absorber RMS' if ch == 0 else None)
    ax.set_ylabel(f'{lab} error'); ax.grid(True); ax.legend(fontsize=7, loc='upper right')
axes[-1].set_xlabel('Time [s]')
fig.suptitle(f'V1: baseline free-run from ENCODER x0 vs TRUE x0 -- does the encoder seed the K=0 drift?')
fig.tight_layout()
stem = os.path.join(dc.OUT_DIR, 'd3_encoder_vs_truex0_V1')
fig.savefig(stem + '.png', dpi=150)
np.savez(stem + '.npz', x0_enc=x0_enc, x0_true=x0_true, dv_enc=dv_enc,
         pred_X=pred_X, pred_Y=pred_Y, t=t, err_enc=err_enc, err_true=err_true,
         te=te, tt=tt, absorber_rms=absorber_rms, K0=K0, ts=ts)
print(f'\nSaved: {stem}.png')
print(f'Saved: {stem}.npz')
