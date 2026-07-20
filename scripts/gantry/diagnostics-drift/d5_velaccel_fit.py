"""
d5_velaccel_fit.py -- confirm the K=0 integrator is the drift MECHANISM by taking it
out of the error path: fit velocity (drop one integrator) or acceleration (drop two).

The baseline-only free-run drifts in POSITION on the K=0 axes (d2/d4). If the drift
is an integrated force/velocity bias, then the SAME free-run's error in velocity
(d/dt of position) and acceleration (d2/dt2) must be bounded -- differentiating
removes the ramp. Supervisor: "should not go up -- can go up but not for the same
reason." A velocity- or acceleration-fit loss is the corresponding admissible fix
(keeps X/Y in the routing, D-103; takes the integrator out of the loss).

Compares the baseline-only free-run states against the true x_logical, in LOGICAL
coords, per K=0 axis (X idx0, Y idx2) and the spring axis (Theta idx1, control):
  position error e_q = q_sim - q_true       (expected: ramp/offset -> "goes up")
  velocity error e_v = v_sim - v_true       (expected: bounded)
  accel    error e_a = d/dt e_v             (expected: bounded, smaller)

Run:
  conda run -n GraduationProject python scripts/gantry/diagnostics-drift/d5_velaccel_fit.py
Env overrides: REC (mat file), FS_NEW (4000), N_WIN (full).
Outputs -> simulations/gantry_subnet/diagnostics/
"""
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import drift_common as dc

REC    = os.environ.get('REC', 'V1_standstill_Yp10.mat')
FS_NEW = int(os.environ.get('FS_NEW', str(dc.DEFAULT_FS_NEW)))
N_WIN  = os.environ.get('N_WIN', None)

rec = dc.load_record(REC, FS_NEW)
ts  = rec.ts
N   = rec.u_stage.shape[0] if N_WIN is None else min(int(N_WIN), rec.u_stage.shape[0])
u_stage = rec.u_stage[:N]
x_true  = rec.x_logical[:N]           # (N,6) logical [q, qd]
t = np.arange(N) * ts
print(f'Record {rec.name}: {N} samples @ {1/ts:.0f} Hz')

# ── Baseline-only free-run, keep the full logical state history ──────────────
x0_6 = dc.true_x0_8(rec, 0)[[0, 1, 2, 4, 5, 6]]
t0 = time.time()
_, x_sim = dc.simulate_baseline(x0_6, u_stage, ts, return_state=True)   # (N,6)
print(f'  baseline free-run ({time.time()-t0:.0f}s)')

q_sim, v_sim   = x_sim[:, :3], x_sim[:, 3:]
q_true, v_true = x_true[:, :3], x_true[:, 3:]
e_q = q_sim - q_true                                   # position error (N,3)
e_v = v_sim - v_true                                   # velocity error (N,3)
e_a = np.vstack([np.zeros((1, 3)), np.diff(e_v, axis=0) * (1.0 / ts)])   # accel error (N,3)

# ── Growth metric: tail |value| for each derivative order ────────────────────
tail = slice(int(0.8 * N), N)
axes_lbl = ['X (K=0)', 'Theta (spring)', 'Y (K=0)']
print('\n=== Tail |error| by derivative order (does differentiating remove the growth?) ===')
print(f"  {'axis':16s} {'|e_q| [m]':>12s} {'|e_v| [m/s]':>13s} {'|e_a| [m/s2]':>13s}")
for c, lbl in enumerate(axes_lbl):
    print(f'  {lbl:16s} {np.abs(e_q[tail,c]).mean():>12.3e} '
          f'{np.abs(e_v[tail,c]).mean():>13.3e} {np.abs(e_a[tail,c]).mean():>13.3e}')

# Ramp test on the K=0 axes: linear-fit slope of e_q vs the flat e_v.
for c in (0, 2):
    slope = np.polyfit(t, e_q[:, c], 1)[0]
    print(f'  {axes_lbl[c]}: position-error linear slope = {slope:+.3e} m/s '
          f'(nonzero -> integrated drift); velocity-error tail |mean| = {np.abs(e_v[tail,c]).mean():.3e} m/s')
print('  -> if e_q ramps but e_v / e_a stay bounded, the K=0 integrator is the drift mechanism and a')
print('     velocity-/acceleration-fit loss removes the "going up" while keeping X/Y in the routing.')

# ── Plot: the three derivative orders on the K=0 Y axis (worst case) ─────────
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
ch = 2   # Y
axes[0].plot(t, e_q[:, ch], 'C3', lw=0.7); axes[0].set_ylabel('e_q  Y [m]')
axes[1].plot(t, e_v[:, ch], 'C0', lw=0.7); axes[1].set_ylabel('e_v  Y [m/s]')
axes[2].plot(t, e_a[:, ch], 'C2', lw=0.7); axes[2].set_ylabel('e_a  Y [m/s2]')
for ax in axes:
    ax.axhline(0, color='k', lw=0.5); ax.grid(True)
axes[-1].set_xlabel('Time [s]')
fig.suptitle(f'{rec.name}: baseline free-run error vs derivative order on the K=0 Y axis -- '
             f'does the drift disappear when differentiated?')
fig.tight_layout()
stem = os.path.join(dc.OUT_DIR, f'd5_velaccel_fit_{rec.name.split(".")[0]}')
fig.savefig(stem + '.png', dpi=150)
np.savez(stem + '.npz', t=t, e_q=e_q, e_v=e_v, e_a=e_a, ts=ts)
print(f'\nSaved: {stem}.png')
print(f'Saved: {stem}.npz')
