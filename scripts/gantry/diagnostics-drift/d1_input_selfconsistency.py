"""
d1_input_selfconsistency.py -- is the stored INPUT (u) and storage self-consistent?

Question (hypothesis A1, "the input/storage is wrong"): seed the full 8-state
truth (baseline + MSD absorber) from the TRUE stored 8-state x0 and drive it with
the stored u over the whole trajectory. In a noiseless sim this MUST reproduce the
stored y / delta_a to integration tolerance IF u and storage are correct. Using
the true x0 (not the encoder) takes the reconstruction error out of the equation,
so any mismatch here is attributable to the input/storage, not the encoder.

Variants (the last three are controls -- what a bug WOULD look like):
  correct     u_log = P @ u_stage, no shift        -> should match to solver tol
  u_shift +1  u leads the states by one sample      -> off-by-one probe
  u_shift -1  u lags the states by one sample       -> off-by-one probe
  no-P frame  u_log = u_stage (P-transform omitted)  -> stage/logical frame probe

Falsifiable readout:
  (A1 rejected) correct-variant stage RMSE << absorber RMS (~1e-4 m) AND far below
                every control variant -> input + storage clean, current alignment
                and frame are the right ones. Move to d2/d3.
  (A1 found)    correct variant is large, OR a shifted/no-P variant is SMALLER than
                the correct one -> localised off-by-one / frame bug.

START defaults to an INTERIOR sample: the stored x_logical velocity there is a
central difference (accurate), so x0 is the true state and any residual is the
input, not the one-sided gradient() velocity artifact at sample 0 (that sample-0
artifact and the true-x0 drift are d2's subject, not d1's).

Run (full V1 trajectory, pipeline rate):
  conda run -n GraduationProject python scripts/gantry/diagnostics-drift/d1_input_selfconsistency.py
Env overrides: REC (mat file), FS_NEW (4000), START (100), N_WIN (full).
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
START  = int(os.environ.get('START', '100'))   # interior: stored v0 is a central diff (accurate)
N_WIN  = os.environ.get('N_WIN', None)   # None -> full trajectory from START

rec = dc.load_record(REC, FS_NEW)
N   = rec.u_stage.shape[0] if N_WIN is None else min(int(N_WIN), rec.u_stage.shape[0] - START)
sl  = slice(START, START + N)

u_stage   = rec.u_stage[sl]
y_stage   = rec.y_stage[sl]
delta_dat = rec.delta_a[sl]
x0_8      = dc.true_x0_8(rec, START)
ts        = rec.ts
t         = np.arange(N) * ts
absorber_rms = delta_dat.std()

print(f'Record {rec.name}: {N} samples @ {1/ts:.0f} Hz (D={rec.D}, START={START}), '
      f'up_sample={dc.UP_SAMPLE}')
print(f'absorber (delta_a) RMS = {absorber_rms:.3e} m   tau_X={dc.tau_X:.3f}s  tau_Y={dc.tau_Y:.3f}s')

# ── Build the four input variants (all seeded from the SAME true x0) ──────────
u_log_correct = dc.stage_force_to_logical(u_stage)
variants = {
    'correct (P@u, no shift)': u_log_correct,
    'u_shift +1':              np.roll(u_log_correct,  -1, axis=0),  # u[k] <- u[k+1]
    'u_shift -1':              np.roll(u_log_correct,  +1, axis=0),  # u[k] <- u[k-1]
    'no-P frame (u_stage)':    u_stage.copy(),
}

tail = slice(int(0.8 * N), N)   # last 20% -- well past 5*tau on the K=0 axes
results = {}
print(f'\n=== Full-truth open-loop from TRUE x0 ({N*ts:.1f} s) ===')
for label, u_log in variants.items():
    t0 = time.time()
    q_log = dc.simulate_truth(x0_8, u_log, ts)          # (N,4) [X,Th,Y,da]
    y_hat = dc.logical_pos_to_stage(q_log[:, :3])       # (N,3) stage
    da_hat = q_log[:, 3]
    err = y_hat - y_stage
    rmse_y  = np.sqrt((err ** 2).mean(axis=0))
    rmse_da = np.sqrt(((da_hat - delta_dat) ** 2).mean())
    tail_off = err[tail].mean(axis=0)                   # settled K=0 offset
    results[label] = dict(err=err, rmse_y=rmse_y, rmse_da=rmse_da, tail_off=tail_off)
    print(f'  [{label}]  ({time.time()-t0:.0f}s)')
    print(f'    stage RMSE  X1={rmse_y[0]:.3e}  X2={rmse_y[1]:.3e}  Y={rmse_y[2]:.3e} m   '
          f'delta_a RMSE={rmse_da:.3e} m')
    print(f'    tail-mean offset  X1={tail_off[0]:+.3e}  X2={tail_off[1]:+.3e}  Y={tail_off[2]:+.3e} m')

# ── Verdict ──────────────────────────────────────────────────────────────────
c = results['correct (P@u, no shift)']['rmse_y'].max()
controls = {k: results[k]['rmse_y'].max() for k in variants if not k.startswith('correct')}
print('\n=== Verdict ===')
print(f'  correct-variant max stage RMSE = {c:.3e} m   (absorber RMS = {absorber_rms:.3e} m)')
# Defensible criterion: the true-x0 + stored-u truth reproduces the data to better
# than the absorber signal we are trying to learn, AND no input perturbation helps.
if c < absorber_rms and c <= min(controls.values()):
    print('  -> input + storage self-consistent: correct-variant RMSE is below the absorber')
    print('     signal and no shift/frame perturbation reduces it. A1 rejected for this record;')
    print('     the drift is NOT the stored input. Proceed to d2/d3.')
else:
    worst_beat = [k for k, v in controls.items() if v < c]
    print(f'  -> NOT self-consistent: correct RMSE >= absorber signal or a control beat it '
          f'({worst_beat}). Inspect which variant minimises RMSE (off-by-one / frame bug),')
    print('     then fix the loader before d2/d3.')

# ── Plot: per-channel error, prediction-vs-measurement ───────────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
colors = ['C0', 'C1', 'C2', 'C3']
for ch, (ax, lab) in enumerate(zip(axes, ['X1 [m]', 'X2 [m]', 'Y [m]'])):
    for (label, r), col in zip(results.items(), colors):
        ax.plot(t, r['err'][:, ch], col, lw=0.7,
                label=f'{label} (RMSE={r["rmse_y"][ch]:.1e})')
    ax.axhline(0, color='k', lw=0.5)
    ax.axhline( absorber_rms, color='0.5', ls=':', lw=0.8)
    ax.axhline(-absorber_rms, color='0.5', ls=':', lw=0.8,
               label=f'+/- absorber RMS ({absorber_rms:.1e} m)' if ch == 0 else None)
    ax.set_ylabel(f'{lab} error'); ax.grid(True); ax.legend(fontsize=7, loc='upper right')
axes[-1].set_xlabel('Time [s]')
fig.suptitle(f'{rec.name}: full-truth from TRUE x0 -- does the stored input reproduce the data? '
             f'(A1 test)')
fig.tight_layout()
stem = os.path.join(dc.OUT_DIR, f'd1_input_selfconsistency_{rec.name.split(".")[0]}')
fig.savefig(stem + '.png', dpi=150)
np.savez(stem + '.npz', t=t, y_stage=y_stage, delta_dat=delta_dat,
         absorber_rms=absorber_rms, ts=ts, START=START,
         labels=np.array(list(results.keys())),
         **{f'err_{i}': r['err'] for i, r in enumerate(results.values())},
         **{f'rmse_y_{i}': r['rmse_y'] for i, r in enumerate(results.values())})
print(f'\nSaved: {stem}.png')
print(f'Saved: {stem}.npz')
