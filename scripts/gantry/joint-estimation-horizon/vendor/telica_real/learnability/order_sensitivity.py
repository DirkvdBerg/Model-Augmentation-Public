"""G5 descriptive sensitivity (NOT the pre-registered rule): model order and horizon of the residual
from a better-conditioned model-error model.

The TR-017 rule read the order off the reference-acceleration block of a 12-channel FIR whose
regressors are collinear (every record runs the same profile), so that block's coefficients are
noise-dominated and the Hankel test returned order 0. Here: FIR on the 3 reference accelerations
ONLY, 200 lags (40 ms) at 5 kHz, same ridge, same Hankel noise rule, same energy rule, plus the
held-out R^2 of this smaller model. Reported next to the pre-registered settings; it does not
replace them.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tr_env                                                          # noqa: E402

import numpy as np                                                     # noqa: E402

from learnability.g5_learnability import load, regressors, signals, lagged, FS   # noqa: E402
tr_env.check_no_leak()

L = 200
OUT = tr_env.out_dir('g5_learnability')


def main():
    TR, VA, TE = load('train'), load('validation'), load('test')
    def seg(D, i):
        km = D['k_motion'][i]
        return regressors(D['r'][i], D['uff'][i])[km:, :3], signals(D['res'][i].astype(float))[km:, :3]
    Xs = [seg(TR, i)[0] for i in range(len(TR['names']))]
    sd = np.concatenate(Xs).std(0)
    p = 3 * L + 1
    XtX = np.zeros((p, p)); XtY = np.zeros((p, 3)); n = 0
    for i in range(len(TR['names'])):
        X, Y = seg(TR, i)
        A = np.hstack([lagged(X / sd, L), np.ones((len(X), 1))])
        XtX += A.T @ A; XtY += A.T @ Y; n += len(A)
    lam = 1e-6 * np.trace(XtX) / p
    Ainv = np.linalg.inv(XtX + lam * np.eye(p))
    B = Ainv @ XtY
    rss = np.zeros(3)
    for i in range(len(TR['names'])):
        X, Y = seg(TR, i)
        A = np.hstack([lagged(X / sd, L), np.ones((len(X), 1))])
        rss += ((Y - A @ B) ** 2).sum(0)
    s2 = rss / (n - p)
    r2 = []
    for D in (VA, TE):
        for i in range(len(D['names'])):
            X, Y = seg(D, i)
            A = np.hstack([lagged(X / sd, L), np.ones((len(X), 1))])
            r2.append(1 - ((Y - A @ B) ** 2).mean(0) / (Y ** 2).mean(0))
    h = np.zeros((L, 3, 3))
    for l in range(L):
        for j in range(3):
            h[l, :, j] = B[l * 3 + j, :3] / sd[j]
    nb = L // 2
    hank = lambda hh: np.block([[hh[i + j] for j in range(nb)] for i in range(nb)])   # noqa: E731
    sv = np.linalg.svd(hank(h), compute_uv=False)
    Lc = np.linalg.cholesky(Ainv[:3 * L, :3 * L] + 1e-30 * np.eye(3 * L))
    rng = np.random.default_rng(0)
    smax = []
    for _ in range(200):
        pert = np.zeros((p, 3))
        for c in range(3):
            pert[:3 * L, c] = Lc @ rng.standard_normal(3 * L) * np.sqrt(s2[c] * 40.0)
        hp = np.zeros((L, 3, 3))
        for l in range(L):
            for j in range(3):
                hp[l, :, j] = pert[l * 3 + j, :3] / sd[j]
        smax.append(np.linalg.svd(hank(hp), compute_uv=False)[0])
    tau = float(np.percentile(smax, 95))
    order = int((sv > tau).sum())
    en = (h ** 2).sum((1, 2)); cum = np.cumsum(en) / en.sum()
    l99 = int(np.searchsorted(cum, 0.99)) + 1
    out = dict(lags=L, heldout_R2_median=np.median(r2, 0).tolist(), hankel_sv_rel=(sv[:16] / sv[0]).tolist(),
               tau_rel=tau / sv[0], order=order, energy99_ms=l99 / FS * 1e3,
               note='descriptive sensitivity; autocorrelation inflation fixed at 40 (G4 value)')
    json.dump(out, open(os.path.join(OUT, 'order_sensitivity.json'), 'w'), indent=1)
    print(f'[sens] r_dd-only FIR, {L} lags: held-out R2 median {np.round(np.median(r2, 0), 3).tolist()}; '
          f'Hankel SV rel {np.round(sv[:12] / sv[0], 4).tolist()}; tau rel {tau / sv[0]:.4f} -> order '
          f'{order}; 99 % energy at {l99 / FS * 1e3:.1f} ms')


if __name__ == '__main__':
    main()
