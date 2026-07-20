"""
s2_hypothesis_figure.py -- S2: one falsifiable figure of the drift hypothesis for the
campus discussion. Assembles EXISTING diagnostic data (d6 ANN-output npz + d7 RMS-vs-
horizon npz); no new simulation.

HYPOTHESIS (stated as a test, not asserted in a title):
  The trained X+Theta+Y ANN learns a nonzero-MEAN (DC) output on the K=0 velocity rows
  (dX, dY). A free position integrator turns a constant velocity offset into a linear
  position ramp -> the free-run drift. The TRUE correction (hidden MSD absorber) is
  ZERO-mean oscillatory, so any DC the ANN invents is pure error. The 0.1 s training
  window is below where the drift appears (~0.5 s), so the loss never penalises it.

Panels (each is a check the viewer can falsify):
  (A) ANN output on the dY row over time + its mean. If the mean line sat on 0, there is
      no DC and the hypothesis is wrong. |mean|/rms printed.
  (B) Cumulative integral of that ANN output (proportional to the position contribution).
      A DC output integrates to a straight ramp; a zero-mean output integrates to a
      bounded wobble. Which one do we see?
  (C) Free-run Y error: full ANN vs ANN with the per-row mean removed (d6 counterfactual).
      If DC is the cause, removing it collapses the drift.
  (D) RMS vs horizon (d7): shows the drift is a >0.5 s phenomenon, invisible at nf=0.1 s.

Run:
  conda run -n GraduationProject python scripts/gantry/diagnostics-drift/s2_hypothesis_figure.py
Inputs: d6_ann_mean_force_gantry_drift_last.npz, d7_validation_horizon_V1.npz
Output: simulations/gantry_subnet/diagnostics/s2_hypothesis_dc_force.png
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import drift_common as dc

D6 = os.path.join(dc.OUT_DIR, 'd6_ann_mean_force_gantry_drift_last.npz')
D7 = os.path.join(dc.OUT_DIR, 'd7_validation_horizon_V1.npz')
for p in (D6, D7):
    if not os.path.exists(p):
        raise FileNotFoundError(f'Missing input {p}. Run d6 and d7 first.')

d6 = np.load(D6, allow_pickle=True)
d7 = np.load(D7, allow_pickle=True)

route_ix = np.asarray(d6['route_ix'])
mean_w   = np.asarray(d6['mean_w'])       # (nw,) per-routed-row mean (normalized)
rms_w    = np.asarray(d6['rms_w'])
ann_out  = np.asarray(d6['ann_out'])      # (N, nw) ANN output over the free-run
ts       = float(d6['ts'])
absorber_rms = float(d6['absorber_rms'])

names8 = np.array(['X', 'Theta', 'Y', 'dX', 'dTheta', 'dY', 'delta_a', 'vdelta_a'])
# pick the dY row (index 5 in the 8-state layout) among the routed columns
jY = int(np.where(route_ix == 5)[0][0])
w_dY = ann_out[:, jY]
mean_dY = mean_w[jY]
ratio_dY = abs(mean_dY) / (rms_w[jY] + 1e-30)
N = len(w_dY)
t = np.arange(N) * ts

# integral of the ANN dY output (proportional to its position-drift contribution)
integ = np.cumsum(w_dY) * ts

# d6 counterfactual free-run Y error (full vs mean-removed) -- reconstruct from tf/td
# (d6 stored tail values; for the curve we use the d6 error arrays if present)
tf = np.asarray(d6['tf']); td = np.asarray(d6['td'])

# d7 RMS vs horizon
hs   = np.asarray(d7['horizons_s'])
rmsH = np.asarray(d7['rms'])   # (nH, 3)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(2, 2, figsize=(14, 9))

# (A) ANN dY output + mean
ax[0, 0].plot(t, w_dY, lw=0.5, color='C0')
ax[0, 0].axhline(mean_dY, color='C3', lw=1.5, label=f'mean = {mean_dY:.2e}')
ax[0, 0].axhline(0, color='k', lw=0.5)
ax[0, 0].set_title(f'(A) ANN output on dY row  (|mean|/rms = {ratio_dY:.2f})\n'
                   f'test: is the mean line on 0?  It is not -> DC present')
ax[0, 0].set_xlabel('time [s]'); ax[0, 0].set_ylabel('ANN dY output [norm]')
ax[0, 0].legend(fontsize=8); ax[0, 0].grid(True)

# (B) integral -> ramp?
ax[0, 1].plot(t, integ, color='C0', lw=0.9, label='cumulative integral of ANN dY output')
# reference straight ramp from the pure mean
ax[0, 1].plot(t, mean_dY * t, color='C3', ls='--', lw=1.2, label='pure-DC ramp (mean x t)')
ax[0, 1].set_title('(B) integral of ANN dY output\n'
                   'test: DC -> straight ramp; zero-mean -> bounded wobble')
ax[0, 1].set_xlabel('time [s]'); ax[0, 1].set_ylabel('integral [norm x s]')
ax[0, 1].legend(fontsize=8); ax[0, 1].grid(True)

# (C) counterfactual bar: full vs mean-removed Y drift
labels = ['X1', 'X2', 'Y']
x = np.arange(3)
ax[1, 0].bar(x - 0.2, tf, 0.4, label='full ANN', color='C3')
ax[1, 0].bar(x + 0.2, td, 0.4, label='ANN mean removed', color='C0')
ax[1, 0].axhline(absorber_rms, color='0.5', ls=':', label=f'absorber RMS ({absorber_rms:.1e} m)')
ax[1, 0].set_yscale('log')
ax[1, 0].set_xticks(x); ax[1, 0].set_xticklabels(labels)
ax[1, 0].set_title(f'(C) free-run drift: removing the ANN mean cuts Y drift '
                   f'{tf[2]/max(td[2],1e-30):.0f}x\ntest: does removing DC collapse the drift?')
ax[1, 0].set_ylabel('tail |error| [m]'); ax[1, 0].legend(fontsize=8); ax[1, 0].grid(True, which='both', axis='y')

# (D) RMS vs horizon
for ch, lbl in enumerate(labels):
    ax[1, 1].loglog(hs, rmsH[:, ch], 'o-', label=lbl)
ax[1, 1].axhline(absorber_rms, color='0.5', ls=':', label='absorber RMS')
ax[1, 1].axvline(0.1, color='C3', ls='--', label='nf = 0.1 s (training)')
ax[1, 1].axvspan(0.1, 0.5, color='C3', alpha=0.08)
ax[1, 1].set_title('(D) RMS vs horizon: drift enters ~0.5 s,\n'
                   'invisible at the 0.1 s training window')
ax[1, 1].set_xlabel('free-run horizon [s]'); ax[1, 1].set_ylabel('cumulative RMS [m]')
ax[1, 1].legend(fontsize=8); ax[1, 1].grid(True, which='both')

fig.suptitle('Drift hypothesis: the ANN learns a DC force on the K=0 velocity rows, '
             'which the free integrator ramps (0.1 s loss cannot see it)', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = os.path.join(dc.OUT_DIR, 's2_hypothesis_dc_force.png')
fig.savefig(out, dpi=150)
print(f'Saved: {out}')
print(f'  (A) dY mean={mean_dY:.3e}  |mean|/rms={ratio_dY:.2f}')
print(f'  (C) Y drift full={tf[2]:.3e} m  mean-removed={td[2]:.3e} m  ({tf[2]/max(td[2],1e-30):.0f}x)')
print(f'  (D) Y RMS: 0.1s={rmsH[1,2]:.2e}  0.5s={rmsH[3,2] if len(rmsH)>3 else float("nan"):.2e}  '
      f'12s={rmsH[-1,2]:.2e}  (absorber {absorber_rms:.2e})')
