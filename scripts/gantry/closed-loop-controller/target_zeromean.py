"""Is the ANN's learning target zero-mean on the open-loop record, at the window scale?

The target the augmentation must learn is the discrepancy between the truth and the baseline,
i.e. what the baseline gets wrong. Two questions decide whether it is trainable:

  1. over the whole record, is its mean small compared with its AC content?
  2. over one training window (nf = 100 ms), is the per-window mean small compared with the
     per-window AC? A target that is zero-mean globally can still be strongly offset inside
     each window, and the loss sees windows, not records.

Compared against the same statistics on a closed-loop-generated record, where the offset is
known to dominate.
"""
__project_origin__ = "added"

import os
import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import closed_loop as CL
import plant as PL

TS = CL.TS
NF_S = 0.100                      # training window length [s], cfg.nf_seconds
CH = ['X1', 'X2', 'Y']
TRAJ_OL = os.path.join(CL.REPO, 'data', 'gantry', 'matlab', 'trajectory', 'openloop')
TRAJ_CL = os.path.join(CL.REPO, 'data', 'gantry', 'matlab', 'trajectory', 'augmentation')


def rollout_stage(fn, x0, u_grid, u_half, ts, n_out=3):
    N = len(u_grid)
    out = np.empty((N, n_out))
    x = np.asarray(x0, float).copy()
    for k in range(N):
        out[k] = x[:n_out]
        u2 = u_half[k] if u_half is not None else u_grid[k]
        u4 = u_grid[k + 1] if k + 1 < N else u_grid[k]
        k1 = fn(x, u_grid[k])
        k2 = fn(x + .5 * ts * k1, u2)
        k3 = fn(x + .5 * ts * k2, u2)
        k4 = fn(x + ts * k3, u4)
        x = x + (ts / 6.) * (k1 + 2 * k2 + 2 * k3 + k4)
    return out


def target_for(path, label):
    dm = loadmat(path, squeeze_me=True)
    ts = float(dm['dt'])
    u = np.asarray(dm['u_total'], float)
    y = np.asarray(dm['y'], float)
    Yop = float(y[0, 2])
    ul = (PL.P_np @ u.T).T
    # u_half exists only on the stage-sampled open-loop record; closed-loop records are ZOH
    uh = (PL.P_np @ np.asarray(dm['u_half'], float).T).T if 'u_half' in dm else None
    print('  simulating truth and baseline on %s ...' % label, flush=True)
    q8 = rollout_stage(PL.deriv8, np.array([0., 0., Yop, 0., 0., 0., 0., 0.]), ul, uh, ts)
    q6 = rollout_stage(PL.deriv6, np.array([0., 0., Yop, 0., 0., 0.]), ul, uh, ts)
    # the ANN's target: what the baseline must be corrected BY
    tgt = PL.to_stage(q8) - PL.to_stage(q6)
    return tgt, ts


def stats(tgt, ts, label, skip_s=6.0):
    t = np.arange(len(tgt)) * ts
    m = t >= skip_s
    nw = int(round(NF_S / ts))
    n = (m.sum() // nw) * nw
    W = tgt[m][:n].reshape(-1, nw, 3)
    wm = W.mean(axis=1)                        # per-window mean
    wa = W.std(axis=1)                         # per-window AC
    print('\n%s   (after %.0f s, %d windows of %.0f ms)' % (label, skip_s, W.shape[0], NF_S * 1e3))
    print('  global mean   [%+.3e %+.3e %+.3e] m' % tuple(tgt[m].mean(axis=0)))
    print('  global AC rms [%.3e %.3e %.3e] m' % tuple(tgt[m].std(axis=0)))
    print('  |global mean| / AC   [%.4f %.4f %.4f]'
          % tuple(np.abs(tgt[m].mean(axis=0)) / tgt[m].std(axis=0)))
    print('  per-window |mean|/AC : median [%.4f %.4f %.4f]  p90 [%.4f %.4f %.4f]'
          % (*np.median(np.abs(wm) / wa, axis=0), *np.percentile(np.abs(wm) / wa, 90, axis=0)))
    return wm, wa


print('ANN learning target = truth - baseline, on the same recorded input\n')
out = {}
for path, label in ((os.path.join(TRAJ_OL, 'OL1_multisine_Yp10.mat'), 'OPEN-LOOP  OL1'),
                    (os.path.join(TRAJ_CL, 'V1_standstill_Yp10.mat'), 'CLOSED-LOOP V1')):
    if not os.path.exists(path):
        print('MISSING %s' % path)
        continue
    tgt, ts = target_for(path, label)
    out[label] = (tgt, ts) + stats(tgt, ts, label)

fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.6))
for r, (label, (tgt, ts, wm, wa)) in enumerate(out.items()):
    t = np.arange(len(tgt)) * ts
    for c in range(3):
        ax = axes[r, c]
        ax.plot(t, tgt[:, c] * 1e6, color='#D55E00', lw=0.5)
        ax.axhline(0, color='#333333', lw=0.8, ls=':')
        ax.grid(alpha=0.25, lw=0.5)
        if c == 0:
            ax.set_ylabel('%s\ntarget [$\\mu$m]' % label)
        if r == 1:
            ax.set_xlabel('time [s]')
        if r == 0:
            ax.set_title(CH[c], fontsize=11)
        ratio = abs(tgt[len(tgt) // 2:, c].mean()) / tgt[len(tgt) // 2:, c].std()
        ax.text(0.97, 0.08, '|mean|/AC %.3f' % ratio, transform=ax.transAxes,
                ha='right', fontsize=8)
fig.suptitle('What the ANN must learn: truth minus baseline, same input.\n'
             'Open loop should be zero-mean; closed-loop-generated data carries the offset.',
             fontsize=11.5, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.93])
o = os.path.join(CL.HERE, 'figures')
os.makedirs(o, exist_ok=True)
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(o, 'target_zeromean.%s' % ext), dpi=160, bbox_inches='tight')
print('\nwrote %s' % os.path.join(o, 'target_zeromean.png'))
