"""OA-021: follow the exact solution of the quadratic conditions (OA-016) from small size upward.

F(z) = Lt z - SUM_st z_s z_t Q[:, s, t]   (c0 = Phi_0^T (Delta*_null - Phi_0 b_null) = 0 exactly,
because b_null = Phi_0^+ Delta*_null). At small size the solution is a member of null(Lt); the
branch is followed in size with least squares (trf) started from the previous solution, scaled.
The branch ends where |F| can no longer be driven to round-off: that size is the largest at which
this library admits an addition orthogonal on its own closed-loop data (under the OA-014 model).
Coefficients are built once (23 or 17 shifted-input Jacobians) and cached in outputs/.
Env: OA_MODE (null control), OA_NULL_JSON, OA_LIB (K | KM), OA_SIZES (comma list, increasing),
     OA_VERIFY (comma list of sizes to check with a fresh Jacobian and to save), OBC_REF_STRIDE.
"""
__project_origin__ = "added"

import gc
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.linalg import eigh
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from oa_core import Problem, COMBO_NAMES               # noqa: E402
import columns as colib                                # noqa: E402
import addition_io as aio                              # noqa: E402
from design_quadratic import jac, shifts               # noqa: E402


def coefficients(pb, lib, b, Dnull, cache):
    if os.path.exists(cache):
        C = np.load(cache)
        print('[cont] coefficients from cache %s' % cache)
        return {k: C[k] for k in C.files}
    n = len(lib)
    phi = lambda c: pb.stateless_force(*lib[c])                    # noqa: E731
    nrm = np.array([np.linalg.norm(pb.delta_of(phi(c))) for c in range(n)])
    Fs = np.stack([(phi(c) @ pb.Pinv.T).reshape(-1).astype(np.float32) for c in range(n)], axis=1)
    Fg = (Fs.T @ Fs).astype(np.float64) / np.outer(nrm, nrm); del Fs; gc.collect()
    Ds = np.stack([(pb.delta_of(phi(c)) / nrm[c]).astype(np.float32) for c in range(n)], axis=1)
    Gn = (Ds.T @ Ds).astype(np.float64); del Ds; gc.collect()
    dhat = lambda c: pb.delta_of(phi(c)) / nrm[c]                  # noqa: E731
    J0 = jac(pb, None)
    J0b = J0 @ b
    c0 = J0.T @ Dnull - J0.T @ J0b
    L = np.stack([J0.T @ dhat(t) for t in range(n)], axis=1)
    Lsn = np.zeros((10, n)); Tb = np.zeros((10, n)); Q = np.zeros((10, n, n))
    t1 = time.perf_counter()
    for s in range(n):
        Js = jac(pb, phi(s) / nrm[s])
        np.subtract(J0, Js, out=Js)
        Lsn[:, s] = Js.T @ Dnull
        Tb[:, s] = Js.T @ J0b + J0.T @ (Js @ b)
        for t in range(n):
            Q[:, s, t] = Js.T @ dhat(t)
        del Js; gc.collect()
        print('  [cont] slot %2d/%d done, %.0f s' % (s + 1, n, time.perf_counter() - t1), flush=True)
    del J0; gc.collect()
    C = dict(nrm=nrm, Fg=Fg, Gn=Gn, c0=c0, L=L, Lt=L - Lsn + Tb, Q=Q)
    np.savez(cache, **C)
    return C


def main():
    t00 = time.perf_counter()
    pb = Problem()
    pb.basis = None; gc.collect()
    js = json.load(open(os.environ['OA_NULL_JSON']))
    p = next(p for p in js['predictions'] if p['tag'] == 'J @ theta*, Delta* (stencil)')
    b = np.asarray(p['dtheta'], float)
    Dnull = pb.measured_delta()
    libname = os.environ.get('OA_LIB', 'K')
    lib = colib.stateless_library(max_ypow=2)
    if libname == 'K':
        lib = [s for s in lib if s[0] == 'K']
    n = len(lib)
    cache = os.path.join(HERE, '..', 'outputs', 'quadcoef_%s_%s_s%d.npz' % (libname, pb.mode, pb.stride))
    C = coefficients(pb, lib, b, Dnull, cache)
    nrm, Fg, Gn, c0, Lt, Q = C['nrm'], C['Fg'], C['Gn'], C['c0'], C['Lt'], C['Q']
    rs = np.maximum(np.abs(C['L']).max(axis=1), 1e-300)
    print('[cont] library %s, %d slots; |c0| %.3e (0 up to round-off: b_null = Phi_0^+ Delta_null)'
          % (libname, n, np.linalg.norm(c0 / rs)))

    def F(z):
        return (c0 + Lt @ z - np.einsum('jst,s,t->j', Q, z, z)) / rs

    def dF(z):
        return (Lt - np.einsum('jkt,t->jk', Q, z) - np.einsum('jsk,s->jk', Q, z)) / rs[:, None]

    # small-size start: the member of null(Lt) with the most Delta per unit force (OA-009 metric)
    _u, _s, Vt = np.linalg.svd(Lt / rs[:, None])
    N0 = Vt[10:].T
    w, V = eigh(N0.T @ Gn @ N0, N0.T @ Fg @ N0)
    z = N0 @ V[:, -1]
    sizes = [float(s) for s in os.environ.get('OA_SIZES', '1,2,3,4,5,6,7,8,8.49,9,10,11,12,12.735').split(',')]
    verify = [float(s) for s in os.environ.get('OA_VERIFY', '8.49,9').split(',') if s]
    z = z * sizes[0] / np.sqrt(z @ Gn @ z)
    wsz = 10.0
    hist = []
    s_prev = sizes[0]
    for s in sizes:
        z0 = z * s / s_prev

        def R(zz):
            return np.concatenate([F(zz), [wsz * (np.sqrt(zz @ Gn @ zz) / s - 1.0)]])

        def dR(zz):
            g = Gn @ zz / (np.sqrt(zz @ Gn @ zz) * s)
            return np.vstack([dF(zz), wsz * g[None, :]])
        res = least_squares(R, z0, jac=dR, method='trf', xtol=1e-15, ftol=1e-15, gtol=1e-15,
                            max_nfev=2000)
        z = res.x
        fs = sum((z / nrm)[c] * pb.stateless_force(*lib[c]) for c in range(n)) @ pb.Pinv.T
        print('[cont] size %7.3f: |F| %.3e, achieved size %.4f, force rms %s N, peak %s N'
              % (s, np.linalg.norm(F(z)), np.sqrt(z @ Gn @ z),
                 np.array2string(np.sqrt((fs ** 2).mean(0)), precision=2),
                 np.array2string(np.abs(fs).max(0), precision=1)), flush=True)
        rec = dict(size=s, F=float(np.linalg.norm(F(z))), achieved=float(np.sqrt(z @ Gn @ z)),
                   force_rms=np.sqrt((fs ** 2).mean(0)).tolist(), force_peak=np.abs(fs).max(0).tolist())
        if any(abs(s - v) < 1e-9 for v in verify):
            ff = sum((z / nrm)[c] * pb.stateless_force(*lib[c]) for c in range(n))
            Jv = jac(pb, ff)
            mx, own, sh = shifts(pb, Jv, Dnull + pb.delta_of(ff), b)
            del Jv; gc.collect()
            print('[cont] VERIFY size %.3f: fresh-Jacobian predicted max shift nine %.4f %%, m_diff own '
                  '%+.4f %%, ||Delta_add|| %.4f' % (s, mx, own, np.linalg.norm(pb.delta_of(ff))), flush=True)
            print('       per parameter: ' + '  '.join('%s %+.4f' % (nm, v) for nm, v in zip(COMBO_NAMES, sh)))
            rec.update(verify_max_shift=mx, verify_mdiff_own=own)
            oa = aio.empty()
            for c, sl in enumerate(lib):
                aio.add_slot(oa, sl, (z / nrm)[c])
            outp = os.path.join(HERE, 'addition_cont_%s_s%s.mat' % (libname, ('%.3f' % s).replace('.', 'p')))
            aio.save_mat(outp, oa, 'OA-021 continuation, library %s, size %.3f' % (libname, s))
            print('       saved %s' % outp)
        hist.append(rec)
        s_prev = s
    json.dump(hist, open(os.path.join(HERE, '..', 'outputs', 'continuation_%s.json' % libname), 'w'), indent=2)
    print('[out] %.0f s' % (time.perf_counter() - t00))


if __name__ == '__main__':
    main()
