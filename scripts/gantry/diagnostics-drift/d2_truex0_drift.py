"""
d2_truex0_drift.py -- does an open-loop free-run drift with TRUE parameters and the
TRUE state, no encoder and no ANN? (the core (A) vs (B) test)

Two parts, both over the FULL trajectory (drift needs t >> 5*tau; tau_X~1.5s):

Part 1 -- "is the drift off with true params + true (clean) state?"
  Seed from an INTERIOR sample (stored velocity = accurate central difference) and
  free-run the whole trajectory:
    full-truth (baseline + MSD)  -> control; must match the data to solver tol.
    baseline-only (no absorber)  -> residual should be the absorber effect only.
  Readout on the K=0 axes (X1, X2, Y) tail-mean offset:
    baseline-only tail offset ~ absorber RMS and bounded -> a correct-everything
      free-run does NOT drift. The augmentation drift then lives in the ENCODER x0
      or the ANN, not the physics/input (points to d3). (hypothesis A2/ANN)
    baseline-only grows unbounded, >> absorber RMS -> the long free-run itself
      accumulates drift even with everything correct -> structural. (hypothesis B)

Part 2 -- "what does the sample-0 velocity artifact do?" (ties d1 -> d3)
  Seed at sample 0, where the stored x_logical velocity is the one-sided gradient()
  FD artifact, and compare baseline-only with that v0 vs with v0 zeroed. The offset
  difference should match tau*dv (viscous decay of an initial-velocity error on a
  K=0 axis). This is the SAME failure mode the encoder produces in d3, from a
  different source: an initial-VELOCITY error on a free integrator -> settled offset.

Run (full V1 trajectory, pipeline rate ~ a few minutes):
  conda run -n GraduationProject python scripts/gantry/diagnostics-drift/d2_truex0_drift.py
Env overrides: REC (mat file), FS_NEW (4000), START_INT (100), N_WIN (full).
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

REC       = os.environ.get('REC', 'V1_standstill_Yp10.mat')
FS_NEW    = int(os.environ.get('FS_NEW', str(dc.DEFAULT_FS_NEW)))
START_INT = int(os.environ.get('START_INT', '100'))   # interior clean-velocity seed
N_WIN     = os.environ.get('N_WIN', None)

rec = dc.load_record(REC, FS_NEW)
ts  = rec.ts
absorber_rms = rec.delta_a.std()
print(f'Record {rec.name}: {rec.u_stage.shape[0]} samples @ {1/ts:.0f} Hz (D={rec.D}), '
      f'up_sample={dc.UP_SAMPLE}')
print(f'absorber RMS = {absorber_rms:.3e} m   tau_X={dc.tau_X:.3f}s  tau_Y={dc.tau_Y:.3f}s')


def _run(start):
    N = rec.u_stage.shape[0] - start if N_WIN is None else min(int(N_WIN), rec.u_stage.shape[0] - start)
    sl = slice(start, start + N)
    return (N, sl, rec.u_stage[sl], rec.y_stage[sl], rec.delta_a[sl],
            dc.stage_force_to_logical(rec.u_stage[sl]))


def _tail(err):
    return err[int(0.8 * len(err)):].mean(axis=0)


# ── Part 1: interior clean x0, full trajectory ───────────────────────────────
N, sl, u_stage, y_stage, delta_dat, u_log = _run(START_INT)
t1 = np.arange(N) * ts
x0_8 = dc.true_x0_8(rec, START_INT)
x0_6 = x0_8[[0, 1, 2, 4, 5, 6]]     # drop absorber states for baseline-only

print(f'\n=== Part 1: full-truth vs baseline-only from CLEAN true x0 '
      f'(START={START_INT}, {N*ts:.1f} s) ===')
t0 = time.time()
q_truth = dc.simulate_truth(x0_8, u_log, ts)
y_truth = dc.logical_pos_to_stage(q_truth[:, :3])
err_truth = y_truth - y_stage
print(f'  full-truth      ({time.time()-t0:.0f}s)  RMSE X1={np.sqrt((err_truth[:,0]**2).mean()):.2e} '
      f'X2={np.sqrt((err_truth[:,1]**2).mean()):.2e} Y={np.sqrt((err_truth[:,2]**2).mean()):.2e} m  '
      f'tail={_tail(err_truth)}')
t0 = time.time()
y_base = dc.simulate_baseline(x0_6, u_stage, ts)
err_base = y_base - y_stage
tb = _tail(err_base)
print(f'  baseline-only   ({time.time()-t0:.0f}s)  RMSE X1={np.sqrt((err_base[:,0]**2).mean()):.2e} '
      f'X2={np.sqrt((err_base[:,1]**2).mean()):.2e} Y={np.sqrt((err_base[:,2]**2).mean()):.2e} m  '
      f'tail={tb}')

drift_ratio = np.abs(tb).max() / (absorber_rms + 1e-30)
print(f'\n  baseline-only max |K=0 tail offset| = {np.abs(tb).max():.3e} m  '
      f'= {drift_ratio:.1f} x absorber RMS')
settle = 5.0 * dc.tau_X
if N * ts < settle:
    print(f'  !! window {N*ts:.1f}s < 5*tau_X={settle:.1f}s: K=0 axes NOT settled, drift verdict INVALID.')
    print('     Re-run over the full trajectory (unset N_WIN) before concluding.')
elif drift_ratio < 3.0:
    print('  -> bounded near the absorber signal: a correct-everything free-run does NOT drift.')
    print('     The augmentation drift lives in the ENCODER x0 or the ANN (go to d3), not physics.')
else:
    print('  -> grows well beyond the absorber signal: the long free-run accumulates drift even')
    print('     with true params + true x0 -> structural (hypothesis B).')

# ── Part 2: sample-0 velocity artifact (baseline-only) ───────────────────────
N0, sl0, u0_stage, y0_stage, _, _ = _run(0)
t2 = np.arange(N0) * ts
x0_asis  = dc.true_x0_8(rec, 0)[[0, 1, 2, 4, 5, 6]]   # stored (gradient) v0
x0_vzero = x0_asis.copy(); x0_vzero[3:] = 0.0          # v0 zeroed
dv_log   = x0_asis[3:].copy()                          # logical velocity error at sample 0
pred_X   = dc.tau_X * dv_log[0]                         # THEORY: settled offset = tau*dv (X axis)
pred_Y   = dc.tau_Y * dv_log[2]                         # (Y axis); theta has a spring -> no offset

print(f'\n=== Part 2: sample-0 velocity artifact, baseline-only (START=0, {N0*ts:.1f} s) ===')
print(f'  stored v0 (logical) = {dv_log}   tau_X*dvX={pred_X:+.3e} m  tau_Y*dvY={pred_Y:+.3e} m')
y_asis  = dc.simulate_baseline(x0_asis,  u0_stage, ts)
y_vzero = dc.simulate_baseline(x0_vzero, u0_stage, ts)
err_asis, err_vzero = y_asis - y0_stage, y_vzero - y0_stage
ta, tz = _tail(err_asis), _tail(err_vzero)
print(f'  v0 as-is  tail offset  X1={ta[0]:+.3e} X2={ta[1]:+.3e} Y={ta[2]:+.3e} m')
print(f'  v0 zeroed tail offset  X1={tz[0]:+.3e} X2={tz[1]:+.3e} Y={tz[2]:+.3e} m')
print(f'  difference (artifact)  X1={ta[0]-tz[0]:+.3e} X2={ta[1]-tz[1]:+.3e} Y={ta[2]-tz[2]:+.3e} m')
print(f'  predicted tau*dv        X~{pred_X:+.3e}          Y={pred_Y:+.3e} m')
print('  -> if the artifact difference tracks tau*dv, a sample-0 (or encoder) VELOCITY error on a')
print('     K=0 axis is a real drift mechanism -- the same failure the encoder can produce (d3).')

# ── Plots ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
for ch, (ax, lab) in enumerate(zip(axes, ['X1 [m]', 'X2 [m]', 'Y [m]'])):
    ax.plot(t1, err_truth[:, ch], 'C0', lw=0.7, label='full-truth (control)')
    ax.plot(t1, err_base[:, ch],  'C1', lw=0.7, label='baseline-only (no absorber)')
    ax.axhline(0, color='k', lw=0.5)
    ax.axhline( absorber_rms, color='0.5', ls=':', lw=0.8)
    ax.axhline(-absorber_rms, color='0.5', ls=':', lw=0.8,
               label=f'+/- absorber RMS ({absorber_rms:.1e} m)' if ch == 0 else None)
    ax.set_ylabel(f'{lab} error'); ax.grid(True); ax.legend(fontsize=7, loc='upper right')
axes[-1].set_xlabel('Time [s]')
fig.suptitle(f'{rec.name} Part 1: does a free-run from CLEAN true x0 drift? '
             f'(true params, no encoder, no ANN)')
fig.tight_layout()
stem = os.path.join(dc.OUT_DIR, f'd2_truex0_drift_{rec.name.split(".")[0]}')
fig.savefig(stem + '_part1.png', dpi=150)

fig2, axes2 = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
for ch, (ax, lab) in enumerate(zip(axes2, ['X1 [m]', 'X2 [m]', 'Y [m]'])):
    ax.plot(t2, err_asis[:, ch],  'C3', lw=0.7, label='baseline, v0 as-is (gradient artifact)')
    ax.plot(t2, err_vzero[:, ch], 'C2', lw=0.7, label='baseline, v0 zeroed')
    pred = pred_X if ch in (0, 1) else pred_Y
    ax.axhline(pred, color='k', ls=':', lw=1.0, label=f'tau*dv = {pred:+.1e} m')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_ylabel(f'{lab} error'); ax.grid(True); ax.legend(fontsize=7, loc='upper right')
axes2[-1].set_xlabel('Time [s]')
fig2.suptitle(f'{rec.name} Part 2: a sample-0 VELOCITY error on the K=0 axes settles to tau*dv')
fig2.tight_layout()
fig2.savefig(stem + '_part2.png', dpi=150)

np.savez(stem + '.npz',
         t1=t1, err_truth=err_truth, err_base=err_base, tail_base=tb,
         t2=t2, err_asis=err_asis, err_vzero=err_vzero, dv_log=dv_log,
         pred_X=pred_X, pred_Y=pred_Y, absorber_rms=absorber_rms,
         tau_X=dc.tau_X, tau_Y=dc.tau_Y, START_INT=START_INT, ts=ts)
print(f'\nSaved: {stem}_part1.png')
print(f'Saved: {stem}_part2.png')
print(f'Saved: {stem}.npz')
