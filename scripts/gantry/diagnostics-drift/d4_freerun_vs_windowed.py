"""
d4_freerun_vs_windowed.py -- why param recovery did NOT drift but the augmentation
free-run does: multiple-shooting (windowed re-seed) vs one long free-run.

Param recovery (train_param_recovery.py) seeds each 650-sample segment from the
TRUE stored state and never free-runs longer than one segment; the augmentation
free-runs the whole trajectory from a single x0. On a K=0 integrator a small
persistent force error (here: the missing MSD absorber in the baseline model)
integrates over the FULL rollout into large drift, but is reset every segment
under windowed re-seeding.

This diagnostic uses the baseline-only model (a realistic imperfect model: it
lacks the absorber, exactly the residual the ANN must learn) and compares, on the
SAME trajectory:
  full free-run   : seed once from the true x0, roll the whole trajectory.
  windowed re-seed: chop into W-sample segments, re-seed each from the true state
                    at its start (multiple shooting), roll W samples per segment.

Readout:
  windowed per-segment K=0 error stays bounded (~absorber level) while the full
  free-run tail grows >> that -> the drift is free-run ACCUMULATION, not a bad
  model per se; multiple shooting (or a windowed/DC-guarded loss) removes it.

Run:
  conda run -n GraduationProject python scripts/gantry/diagnostics-drift/d4_freerun_vs_windowed.py
Env overrides: REC (mat file), FS_NEW (4000), W (650 samples, param-recovery-like), N_WIN (full).
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
W      = int(os.environ.get('W', '650'))     # param-recovery segment length
N_WIN  = os.environ.get('N_WIN', None)

rec = dc.load_record(REC, FS_NEW)
ts  = rec.ts
N   = rec.u_stage.shape[0] if N_WIN is None else min(int(N_WIN), rec.u_stage.shape[0])
u_stage = rec.u_stage[:N]
y_stage = rec.y_stage[:N]
absorber_rms = rec.delta_a[:N].std()
t = np.arange(N) * ts
print(f'Record {rec.name}: {N} samples @ {1/ts:.0f} Hz, W={W} ({W*ts*1e3:.0f} ms/segment), '
      f'absorber RMS={absorber_rms:.3e} m')

# ── Full free-run: seed once from the true x0 ────────────────────────────────
x0_6 = dc.true_x0_8(rec, 0)[[0, 1, 2, 4, 5, 6]]
t0 = time.time()
y_full = dc.simulate_baseline(x0_6, u_stage, ts)
err_full = y_full - y_stage
print(f'  full free-run     ({time.time()-t0:.0f}s)')

# ── Windowed re-seed: re-init each segment from the TRUE state (multiple shooting)
t0 = time.time()
err_win = np.empty_like(err_full)
for s in range(0, N, W):
    e = min(s + W, N)
    x0_seg = dc.true_x0_8(rec, s)[[0, 1, 2, 4, 5, 6]]   # re-seed from truth
    y_seg = dc.simulate_baseline(x0_seg, u_stage[s:e], ts)
    err_win[s:e] = y_seg - y_stage[s:e]
print(f'  windowed re-seed  ({time.time()-t0:.0f}s, {int(np.ceil(N/W))} segments)')

# ── Metric: does the error GROW across the trajectory (full) but not per segment
# (windowed)? Compare the mean |error| in the FIRST vs LAST segment for each. A
# free-run that accumulates has last >> first; windowed re-seeding resets it, so
# its per-segment level is flat. Comparing within-segment maxima is invalid (a
# re-seeded segment restarts its own tau*dv transient), so we compare the
# per-segment settled level (mean over each segment) and its growth. ──────────
seg_starts = list(range(0, N, W))
def _seg_levels(err):                       # mean |err| per segment, per channel
    return np.array([np.abs(err[s:min(s + W, N)]).mean(axis=0) for s in seg_starts])
lv_full, lv_win = _seg_levels(err_full), _seg_levels(err_win)
first_full, last_full = lv_full[0], lv_full[-1]
first_win,  last_win  = lv_win[0],  lv_win[-1]
print('\n=== Per-segment |error| level: first vs last segment (does it GROW?) ===')
print(f"  {'chan':4s} {'full first':>12s} {'full last':>12s} {'grow x':>8s} | "
      f"{'win first':>11s} {'win last':>11s} {'grow x':>8s}")
for c, lbl in enumerate(['X1', 'X2', 'Y']):
    gf = last_full[c] / (first_full[c] + 1e-30)
    gw = last_win[c]  / (first_win[c]  + 1e-30)
    print(f'  {lbl:4s} {first_full[c]:>12.3e} {last_full[c]:>12.3e} {gf:>8.1f} | '
          f'{first_win[c]:>11.3e} {last_win[c]:>11.3e} {gw:>8.1f}')
print(f'\n  absorber RMS = {absorber_rms:.3e} m')
grow_full = (last_full / (first_full + 1e-30)).max()
grow_win  = (last_win  / (first_win  + 1e-30)).max()
if grow_full > 3.0 and grow_win < 3.0:
    print(f'  -> full free-run error GROWS across the trajectory ({grow_full:.0f}x) while windowed stays')
    print('     flat: the drift is free-run ACCUMULATION, reset by re-seeding. This is why param')
    print('     recovery (segments re-seeded from the true state) did not drift and the free-run does.')
else:
    print(f'  -> full grow={grow_full:.1f}x, windowed grow={grow_win:.1f}x: no clear accumulation signature')
    print('     on this record (may be excitation-limited; try an APRBS record e.g. T9/T10/T11).')

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
for ch, (ax, lab) in enumerate(zip(axes, ['X1 [m]', 'X2 [m]', 'Y [m]'])):
    ax.plot(t, err_full[:, ch], 'C1', lw=0.7, label='full free-run (single x0)')
    ax.plot(t, err_win[:, ch],  'C0', lw=0.7, label=f'windowed re-seed (W={W})')
    for s in range(0, N, W):
        ax.axvline(s * ts, color='0.85', lw=0.4, zorder=0)
    ax.axhline(0, color='k', lw=0.5)
    ax.axhline( absorber_rms, color='0.5', ls=':', lw=0.8)
    ax.axhline(-absorber_rms, color='0.5', ls=':', lw=0.8,
               label=f'+/- absorber RMS' if ch == 0 else None)
    ax.set_ylabel(f'{lab} error'); ax.grid(True); ax.legend(fontsize=7, loc='upper right')
axes[-1].set_xlabel('Time [s]')
fig.suptitle(f'{rec.name}: full free-run vs windowed re-seed (baseline-only) -- '
             f'does re-estimation kill the drift?')
fig.tight_layout()
stem = os.path.join(dc.OUT_DIR, f'd4_freerun_vs_windowed_{rec.name.split(".")[0]}')
fig.savefig(stem + '.png', dpi=150)
np.savez(stem + '.npz', t=t, err_full=err_full, err_win=err_win, W=W,
         lv_full=lv_full, lv_win=lv_win, absorber_rms=absorber_rms, ts=ts)
print(f'\nSaved: {stem}.png')
print(f'Saved: {stem}.npz')
