"""G3 R1-LS: the closed-form best linear structure against the exact truth (ET-007).

x = sum_{k=0..6} phi_k(Y) W_k w_pure, fitted by ridge least squares (normal equations in chunks,
eigen-solve), lambda chosen on held-out training records T6, T11, refit on all T1-T14, scored on V1-V4
(stride 20) and T1-T5 (stride 5) against the exact state, next to the G2 planted-source init.
No rollout, no model gradient. Output: outputs/g3/r1ls.json
"""
__project_origin__ = "added"

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import et_paths                                                  # noqa: E402
sys.path.insert(0, os.path.join(et_paths.ET, 'encoder'))
import core                                                      # noqa: E402
import enc_common as ec                                          # noqa: E402
import truth                                                     # noqa: E402
from g1_state import windows as state_windows                    # noqa: E402

DEG = 6
R_GRID = [1e-14, 1e-12, 1e-10, 1e-8, 1e-6]
HOLD = ['T6_ysweep_slow', 'T11_aprbs_100']
CH8 = ec.CHANNELS + ['da', 'vda']
OUT = os.path.join(et_paths.OUT, 'g3')


def features(yw, uw, su, sy):
    N = len(yw)
    wp = np.concatenate([(yw / sy).reshape(N, -1), (uw / su).reshape(N, -1)], 1)
    phi = np.concatenate([np.ones((N, 1)), core.cheb_basis(yw[:, -1, 2] / core.S_SCALE, DEG)], 1)
    return (phi[:, :, None] * wp[:, None, :]).reshape(N, -1)


def accumulate(names, cfg, na, stride, su, sy, sx8, chunk=3000):
    G, H, n = None, None, 0
    for nm in names:
        yw, uw, x8, _, _ = state_windows(nm, cfg, na, stride=stride)
        for i in range(0, len(yw), chunk):
            F = features(yw[i:i + chunk], uw[i:i + chunk], su, sy)
            T = x8[i:i + chunk] / sx8
            G = F.T @ F if G is None else G + F.T @ F
            H = F.T @ T if H is None else H + F.T @ T
            n += len(F)
    return G, H, n


def solve(G, H, r, eig=None):
    e, V = eig if eig is not None else np.linalg.eigh(G)
    lam = r * np.trace(G) / len(G)
    return V @ ((V.T @ H) / (e + lam)[:, None])


def predict(theta, names, cfg, na, stride, su, sy, sx8):
    out, tgt = [], []
    for nm in names:
        yw, uw, x8, _, _ = state_windows(nm, cfg, na, stride=stride)
        out.append((features(yw, uw, su, sy) @ theta) * sx8)
        tgt.append(x8)
    return np.concatenate(out), np.concatenate(tgt)


def main():
    t0 = time.time()
    et_paths.check()
    cfg, data, norm, fit_sys, hp, dims = ec.build()
    na = dims[0]
    from gantry_dynamic.data import TRAIN_FILES, VAL_FILES
    tr = [f[:-4] for f in TRAIN_FILES]
    va = [f[:-4] for f in VAL_FILES]
    sx6 = np.asarray(norm.x_all, float).std(0)
    su = np.asarray(norm.u_all, float).std(0)
    sy = np.asarray(norm.y_all, float).std(0)
    xa = np.concatenate([truth.exact_truth(f, cfg.mode)['x8'][:, [3, 7]] for f in tr])
    sa = xa.std(0)
    sx8 = np.concatenate([sx6, sa])
    g2 = torch.load(os.path.join(et_paths.OUT, 'g2', 'g2_attempt1_encoders.pt'), weights_only=False)
    Cp, W0p = g2['Cp'], g2['W0p']

    def g2pred(names, stride):
        out, tgt = [], []
        for nm in names:
            yw, uw, x8, _, _ = state_windows(nm, cfg, na, stride=stride)
            out.append(core.numpy_sched(Cp, W0p, yw, uw, sx8, su, sy)); tgt.append(x8)
        return np.concatenate(out), np.concatenate(tgt)

    # lambda on held-out training records
    fit_names = [n for n in tr if n not in HOLD]
    G, H, n = accumulate(fit_names, cfg, na, 4, su, sy, sx8)
    eig = np.linalg.eigh(G)
    print(f'  normal equations: {n} windows, {G.shape[0]} features, cond ~ '
          f'{eig[0].max() / max(eig[0].min(), 1e-300):.1e}  [{time.time() - t0:.0f} s]')
    xg, tg = g2pred(HOLD, 4)
    ms_g2h = ((xg - tg) ** 2).mean(0)
    scores = {}
    for r in R_GRID:
        th = solve(G, H, r, eig)
        xp, tp_ = predict(th, HOLD, cfg, na, 4, su, sy, sx8)
        scores[r] = float((((xp - tp_) ** 2).mean(0) / ms_g2h).mean())
        print(f'  r {r:.0e}: held-out MSE / G2 MSE (mean over 8 channels) {scores[r]:.4f}')
    r_best = min(scores, key=scores.get)
    G, H, n = accumulate(tr, cfg, na, 4, su, sy, sx8)
    th = solve(G, H, r_best)
    res = dict(r_grid=R_GRID, holdout_scores={str(k): v for k, v in scores.items()}, r_best=r_best)
    for lab, names, stride in (('V1-V4', va, 20), ('T1-T5', tr[:5], 5)):
        xp, tp_ = predict(th, names, cfg, na, stride, su, sy, sx8)
        xg, tg = g2pred(names, stride)
        e_ls = np.sqrt(((xp - tp_) ** 2).mean(0))
        e_g2 = np.sqrt(((xg - tg) ** 2).mean(0))
        res[lab] = dict(ls=e_ls.tolist(), g2=e_g2.tolist(), headroom=(e_g2 / e_ls).tolist())
        print(f'\n  {lab}: {"ch":<7}{"G2 init":>12}{"LS best":>12}{"headroom":>10}')
        for i, c in enumerate(CH8):
            print(f'  {"":<7}{c:<7}{e_g2[i]:>12.3e}{e_ls[i]:>12.3e}{e_g2[i] / e_ls[i]:>10.2f}')
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'r1ls.json'), 'w') as f:
        json.dump(res, f, indent=1)
    print(f'  saved outputs/g3/r1ls.json  [{time.time() - t0:.0f} s]')


if __name__ == '__main__':
    main()
