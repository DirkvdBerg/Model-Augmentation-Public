"""
d15_pin_target_stationarity.py -- pre-implementation Layer-2 diagnostic: is the
pin target (the ANN's DC output) STATIONARY along the drifting free-run?

Layer 2 pins the DC direction measured on TRAINING windows (near-truth states).
R4 failure mode "distribution shift": if the trained ANN's slow output CHANGES
as the state drifts off-distribution, a fixed training-distribution pin loses
authority along the free-run (the drift accelerates away from the pinned value,
C2 moving-target limit). This probe measures that state-dependence directly on
the SAVED d6 free-run capture (12 s, drifting states; Y reaches ~2.6e-2 m by the
end) -- pure analysis, no simulation, no training, no pipeline change.

Method: split the free-run ANN output into 1 s blocks; per-block mean of each
K=0 routed row; compare first block (near-truth states; should match the
windowed-pass value, d8/d12) to last block (fully drifted states); linear trend
across blocks judged against the between-block scatter.

PRE-DECLARED reading (dY row = the drift driver):
  |last-block minus first-block| <~ 20% of first-block AND trend within ~2 SE
    -> the pin target is approximately stationary under the worst observed
       drift: a FIXED joint-direction pin holds to first order off-distribution.
  monotone growth >> 20% -> the DC is strongly state-dependent along the drift:
       plan for iterative re-aiming (measure -> pin -> re-measure) in the build.

Run:
  conda run -n GraduationProject python scripts/gantry/diagnostics-drift/d15_pin_target_stationarity.py
Env: NPZ (default the trained d6 capture), BLOCK_SEC (default 1.0).
Outputs -> simulations/gantry_subnet/diagnostics/ (npz + png)
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import drift_common as dc

NPZ = os.environ.get(
    'NPZ', os.path.join(dc.OUT_DIR, 'd6_ann_mean_force_gantry_drift_last.npz'))
BLOCK_SEC = float(os.environ.get('BLOCK_SEC', '1.0'))

z = np.load(NPZ, allow_pickle=True)
ann_out  = z['ann_out']            # (N, nw) free-run capture, drifting states
route_ix = z['route_ix']
ts       = float(z['ts'])
names8 = np.array(['X', 'Theta', 'Y', 'dX', 'dTheta', 'dY', 'delta_a', 'vdelta_a'])
N, nw = ann_out.shape
nb = int(BLOCK_SEC / ts)
nblocks = N // nb
print(f'{os.path.basename(NPZ)}: {N} samples ({N*ts:.1f}s), {nblocks} blocks of {BLOCK_SEC}s')
print(f'global mean_w (d6): {np.array2string(z["mean_w"], precision=3)}')

blocks = ann_out[:nblocks * nb].reshape(nblocks, nb, nw)
bmean = blocks.mean(axis=1)                    # (nblocks, nw)
t_blk = (np.arange(nblocks) + 0.5) * BLOCK_SEC

k0_rows = [0, 2, 3, 5]                         # X, Y, dX, dY
print(f"\nper-block mean, K=0 rows [norm]:")
hdr = '  block(t)  ' + ' '.join(f'{names8[r]:>11s}' for r in k0_rows)
print(hdr)
jmap = {int(r): int(np.where(route_ix == r)[0][0]) for r in route_ix}
for b in range(nblocks):
    print(f'  {t_blk[b]:7.1f}s  ' + ' '.join(f'{bmean[b, jmap[r]]:>11.3e}' for r in k0_rows))

print(f"\n{'row':8s} {'first blk':>11s} {'last blk':>11s} {'change %':>9s} "
      f"{'trend/SE':>9s} {'verdict':>12s}")
res = {}
for r in k0_rows:
    jj = jmap[r]
    y = bmean[:, jj]
    A = np.vstack([np.ones(nblocks), t_blk]).T
    (a0, b1), res_ss, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ [a0, b1]
    se_b1 = np.sqrt((resid ** 2).sum() / (nblocks - 2) / ((t_blk - t_blk.mean()) ** 2).sum())
    chg = (y[-1] - y[0]) / (abs(y[0]) + 1e-30) * 100
    tse = b1 / se_b1 if se_b1 > 0 else np.nan
    verdict = 'STATIONARY' if (abs(chg) < 20 and abs(tse) < 2) else 'DRIFTING'
    res[r] = (y[0], y[-1], chg, tse, verdict)
    print(f'{names8[r]:8s} {y[0]:>11.3e} {y[-1]:>11.3e} {chg:>8.1f}% {tse:>9.2f} {verdict:>12s}')

r5 = res[5]
print('\ndY row (the drift driver / pin target):')
if r5[4] == 'STATIONARY':
    print('  -> pin target ~STATIONARY under the worst observed drift: a fixed')
    print('     joint-direction pin holds to first order off-distribution.')
else:
    print('  -> pin target CHANGES along the drift: state-dependent DC; plan the')
    print('     build with iterative re-aiming (measure -> pin -> re-measure, C2).')

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
for ax, r in zip(axes.flat, k0_rows):
    jj = jmap[r]
    ax.plot(t_blk, bmean[:, jj], 'o-', ms=4)
    ax.axhline(z['mean_w'][jj], color='C3', ls=':', label='global mean (d6)')
    ax.set_title(f'{names8[r]} row: per-block mean along the free-run')
    ax.set_ylabel('[norm]'); ax.grid(True); ax.legend(fontsize=7)
for ax in axes[-1]:
    ax.set_xlabel('free-run time [s] (drift grows to ~2.6e-2 m)')
fig.suptitle('d15: is the Layer-2 pin target stationary as the state drifts off-distribution?')
fig.tight_layout()
stem = os.path.join(dc.OUT_DIR, 'd15_pin_target_stationarity')
fig.savefig(stem + '.png', dpi=150)
np.savez(stem + '.npz', t_blk=t_blk, bmean=bmean, route_ix=route_ix,
         mean_w=z['mean_w'], block_sec=BLOCK_SEC, src=str(NPZ),
         **{f'res_{r}': np.array(res[r][:4], dtype=float) for r in k0_rows})
print(f'\nSaved: {stem}.png')
print(f'Saved: {stem}.npz')
