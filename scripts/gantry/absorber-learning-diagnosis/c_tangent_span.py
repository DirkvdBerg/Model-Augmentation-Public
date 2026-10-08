"""Can the ten physical parameters imitate the absorber? (joint-estimation substitution test)

In the pipeline's residual closed loop (the batched oracle of b_error_budget.py), on the multisine
training records of the thesis noise-free set and the old set:
  target  s = y_nofr - y_bt          the absorber's effect on a friction-free truth
  tangent t_i = (y_bt(theta + d_i) - y_bt) / d_i   for ten physical perturbations spanning the ten
          identifiable combinations (kb1, cg1, cg2, cy, cb1, mh, mb, m1 - m2 antisymmetric, Jb, d)
Least squares of s on the ten tangents per record, channels weighted by 1/ystd (the training
objective's weighting); reports the fraction of s explained, overall and per band.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from b_error_budget import load_set, variant, oracle_cl_batch, band_rms, BANDS   # noqa: E402

# HEURISTIC: 1e-3 relative step, small against the 10 % detuning, large against float64 roundoff
REL = 1e-3
PERTS = [('kb1', ('kb1',), 1), ('cg1', ('cg1',), 1), ('cg2', ('cg2',), 1), ('cy', ('cy',), 1),
         ('cb1', ('cb1',), 1), ('mh', ('mh_rigid',), 1), ('mb', ('mb',), 1),
         ('m_diff', ('m1', 'm2'), -1), ('Jb', ('Jb',), 1), ('d', ('d',), 1)]


def perturbed(tp, keys, sign):
    tp = dict(tp)
    step = REL * tp[keys[0]]
    tp[keys[0]] += step
    if len(keys) == 2:
        tp[keys[1]] += sign * step
    return tp, step


def run(key):
    cfg, recs, ystd = load_set(key)
    recs = [r for r in recs if r['split'] == 'train' and r['cls'] not in ('telica',)]
    N = min(len(r['u']) for r in recs)
    R = len(recs)
    tps, steps = [], []
    for r in recs:
        bt = variant(r['tp'], 'bt')
        row = [variant(r['tp'], 'nofr'), bt]
        for _, keys, sign in PERTS:
            tp, st = perturbed(bt, keys, sign)
            row.append(tp); steps.append(st)
        tps.append(row)
    nv = len(tps[0])
    U = np.repeat(np.stack([r['u'][:N] for r in recs]), nv, 0)
    Y = np.repeat(np.stack([r['y'][:N] for r in recs]), nv, 0)
    X0 = np.repeat(np.stack([r['x0'] for r in recs]), nv, 0)
    C = [r['ctrl'] for r in recs for _ in range(nv)]
    ys = oracle_cl_batch(U, Y, X0, [t for row in tps for t in row], C, cfg.ts_new).reshape(R, nv, N, 3)
    steps = np.array(steps).reshape(R, len(PERTS))
    w = 1.0 / ystd
    print('\n== %s: %d multisine training records, ystd %s ==' % (key, R, ystd))
    fr_all, band_s, band_res = [], [], []
    for i, r in enumerate(recs):
        s = ys[i, 0] - ys[i, 1]                                            # (N, 3)
        T = (ys[i, 2:] - ys[i, 1][None]) / steps[i][:, None, None]         # (10, N, 3)
        A = (T * w).reshape(len(PERTS), -1).T
        b = (s * w).ravel()
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        res = s - np.tensordot(coef, T, 1)
        frac = 1 - np.sum((res * w) ** 2) / np.sum((s * w) ** 2)
        fr_all.append(frac)
        band_s.append(band_rms(s, cfg.fs_new_hz)); band_res.append(band_rms(res, cfg.fs_new_hz))
        rel = coef * steps[i] / REL / np.array([recs[i]['tp'][k[0]] if k[0] != 'mh_rigid' else variant(recs[i]['tp'], 'bt')['mh_rigid'] for _, k, _ in PERTS])
        print('  %-26s explained %.1f %%  | Y rms s %.2e res %.2e | fitted rel. change %s' % (
            os.path.basename(r['file'])[:26], 100 * frac, np.sqrt(np.mean(s[:, 2] ** 2)),
            np.sqrt(np.mean(res[:, 2] ** 2)), ' '.join('%s %+.1f%%' % (p[0], 100 * c) for p, c in zip(PERTS, rel))))
    bs, br = np.mean(band_s, 0), np.mean(band_res, 0)
    print('  mean explained %.1f %%' % (100 * np.mean(fr_all)))
    print('  Y per band, s -> residual after the parameter fit:')
    for j, (lo, hi) in enumerate(BANDS):
        print('    %6.0f-%-6.0f Hz  %.2e -> %.2e m  (explained %.0f %%)' % (
            lo, hi, bs[j, 2], br[j, 2], 100 * (1 - (br[j, 2] / bs[j, 2]) ** 2) if bs[j, 2] > 0 else 0))
    np.savez(os.path.join(HERE, 'outputs', 'c_tangent_%s.npz' % key), frac=np.array(fr_all),
             band_s=np.array(band_s), band_res=np.array(band_res), files=np.array([r['file'] for r in recs]))


if __name__ == '__main__':
    for k in (sys.argv[1] if len(sys.argv) > 1 else 'noisefree,old').split(','):
        run(k)
