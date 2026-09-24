"""OA-016: the ten G4-target conditions as an EXACT quadratic system in the addition coefficients.

Regenerated closed-loop data keep x and record u_new = u - phi (loop cancels the addition's force,
OA-014). The step is affine in u, so with normalised coefficients z (z_s = kappa_s ||Delta_s||):
    Phi(z)    = Phi_0 - SUM_s z_s Phi_s ,    Phi_s = Phi(x, u) - Phi(x, u - phi_s / ||Delta_s||)
    Delta(z)  = Delta*_null + SUM_t z_t Dhat_t ,  Dhat_t = G phi_t / ||Delta_t||
    target(z) = Phi(z)^T Phi(z) b_null  (first order in z kept; z^2 |b_null| dropped)
    F(z)      = Phi(z)^T Delta(z) - target(z)
              = c0 + (L - Lsn + Tb) z - SUM_st z_s z_t Q[:, s, t]
Solved for F(z) = 0 and ||SUM z_t Dhat_t|| = OA_SIZE with least stage force (SLSQP from the
starting addition), then VERIFIED with a fresh Jacobian at u - phi(z*).
Env: OA_MODE (null control), OA_ADDITION (start, stateless), OA_OUT, OA_NULL_JSON, OA_SIZE,
     OBC_REF_STRIDE.
"""
__project_origin__ = "added"

import gc
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from oa_core import Problem, COMBO_NAMES, ROWS          # noqa: E402
import columns as colib                                # noqa: E402
import addition_io as aio                              # noqa: E402
from vendor.obc import OBCBasis, build_stacked_sensitivity   # noqa: E402


def jac(pb, f_log):
    Z = pb.ref.Z_step.clone().to(torch.float64)
    if f_log is not None:
        Z[:, 6:9, 0] -= torch.from_numpy((f_log @ pb.Pinv.T) / pb.std_u)
    J = build_stacked_sensitivity(pb.step, pb.vbar.to(pb.device), Z, ROWS,
                                  chunk=pb.cfg.obc_ref_chunk, device=pb.device).cpu().numpy()
    del Z
    if pb.device.type == 'cuda':
        torch.cuda.empty_cache()
    return J


def shifts(pb, J, D, b_null):
    """Predicted per-parameter shift of J^+ D against b_null (combo %, m_diff own %)."""
    bas = OBCBasis.from_matrix(torch.from_numpy(J), pb.vbar.cpu())
    dth = bas.coefficient(torch.from_numpy(D)).numpy()
    del bas
    tc = pb.true_combo.numpy()
    c1 = pb.blk.combinations_from_free((pb.vbar + torch.from_numpy(dth)).to(pb.device)).detach().cpu().numpy()
    c0 = pb.blk.combinations_from_free((pb.vbar + torch.from_numpy(b_null)).to(pb.device)).detach().cpu().numpy()
    scale = np.abs(tc).copy(); scale[7] = 10.45
    sh = 100 * (c1 - c0) / scale
    return float(np.abs(np.delete(sh, 7)).max()), float(100 * (c1[7] - c0[7]) / abs(tc[7])), sh


def main():
    t00 = time.perf_counter()
    pb = Problem()
    pb.basis = None; gc.collect()
    size = float(os.environ.get('OA_SIZE', '12.735'))
    global DELTA_MODE
    DELTA_MODE = os.environ.get('OA_DELTA_MODE') == '1'
    js = json.load(open(os.environ['OA_NULL_JSON']))
    p = next(p for p in js['predictions'] if p['tag'] == 'J @ theta*, Delta* (stencil)')
    b = np.asarray(p['dtheta'], float)
    Dnull = pb.measured_delta()
    lib = colib.stateless_library(max_ypow=2)
    if os.environ.get('OA_LIB', 'KM') == 'K':
        lib = [s for s in lib if s[0] == 'K']                    # OA-017: stiffness slots only
    n = len(lib)
    print('[quad] library %s: %d slots' % (os.environ.get('OA_LIB', 'KM'), n))
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
        np.subtract(J0, Js, out=Js)                              # exact: the step is affine in u
        Lsn[:, s] = Js.T @ Dnull
        Tb[:, s] = Js.T @ J0b + J0.T @ (Js @ b)
        for t in range(n):
            Q[:, s, t] = Js.T @ dhat(t)
        del Js; gc.collect()
        print('  [quad] slot %2d/%d %-6s done, %.0f s' % (s + 1, n, colib.slot_name(lib[s]),
                                                         time.perf_counter() - t1), flush=True)
    Lt = L - Lsn + Tb
    rs = np.maximum(np.abs(L).max(axis=1), 1e-300)          # row scale for the solver

    def F(z):
        return (c0 + Lt @ z - np.einsum('jst,s,t->j', Q, z, z)) / rs

    def dF(z):
        return (Lt - np.einsum('jkt,t->jk', Q, z) - np.einsum('jsk,s->jk', Q, z)) / rs[:, None]

    # start: the given addition in normalised library coordinates
    oa0 = aio.load_mat(os.environ['OA_ADDITION'])
    z0 = np.zeros(n)
    for c, s in enumerate(lib):
        key = {('M', 0): 'oa_Ma0', ('M', 1): 'oa_Ma1', ('M', 2): 'oa_Ma2', ('K', 0): 'oa_Ka0', ('K', 1): 'oa_Ka1',
               ('K', 2): 'oa_Ka2'}[(s[0], s[3])]
        z0[c] = oa0[key][s[1], s[2]] * nrm[c]
    if DELTA_MODE:
        # OA-017: OA_MODE was generated WITH the addition; Dnull above is its MEASURED Delta*, and
        # the unknown is the correction delta (start 0). No size constraint: the addition already
        # has its size, the correction is small. Output = OA_ADDITION + delta.
        z_add = z0.copy()
        z0 = np.zeros(n)
        print('[quad] correction mode on %s: |F(0)| %.3e' % (pb.mode, np.linalg.norm(F(z0))))
        cons = [dict(type='eq', fun=F, jac=dF)]
        fs0 = 1.0
    else:
        print('[quad] start: |F| %.3e (linear part alone %.3e), size %.4f'
              % (np.linalg.norm(F(z0)), np.linalg.norm((c0 + L @ z0) / rs), np.sqrt(z0 @ Gn @ z0)))
        cons = [dict(type='eq', fun=F, jac=dF),
                dict(type='eq', fun=lambda z: np.array([(z @ Gn @ z) / size ** 2 - 1.0]),
                     jac=lambda z: (2 * Gn @ z / size ** 2)[None, :])]
        fs0 = z0 @ Fg @ z0
    res = minimize(lambda z: (z @ Fg @ z) / fs0, z0, jac=lambda z: 2 * Fg @ z / fs0,
                   constraints=cons, method='SLSQP', options=dict(maxiter=500, ftol=1e-14))
    z = res.x
    print('[quad] SLSQP: %s, %d it; |F| %.3e, size %.4f, force energy x%.3f of the start'
          % (res.message, res.nit, np.linalg.norm(F(z)), np.sqrt(z @ Gn @ z), (z @ Fg @ z) / fs0))

    # verification with a fresh Jacobian at the shifted input
    kappa = z / nrm
    f = sum(kappa[c] * phi(c) for c in range(n))
    D = Dnull + pb.delta_of(f)
    for tag, zz in (('start', z0), ('solution', z)):
        ff = sum((zz / nrm)[c] * phi(c) for c in range(n))
        Jv = jac(pb, ff)
        mx, own, sh = shifts(pb, Jv, Dnull + pb.delta_of(ff), b)
        del Jv; gc.collect()
        fs = ff @ pb.Pinv.T
        print('[quad] VERIFY %-8s: predicted shift vs b_null max nine %.4f %%, m_diff own %+.4f %%, '
              '||Delta_add|| %.4f, force rms %s N, peak %s N'
              % (tag, mx, own, np.linalg.norm(pb.delta_of(ff)),
                 np.array2string(np.sqrt((fs ** 2).mean(0)), precision=2),
                 np.array2string(np.abs(fs).max(0), precision=1)), flush=True)
    print('  per parameter (solution): ' + '  '.join('%s %+.4f' % (nm, v) for nm, v in zip(COMBO_NAMES, sh)))
    if DELTA_MODE:
        kappa = (z_add + z) / nrm            # the addition already in the data + the correction
    oa = aio.empty()
    for c, s in enumerate(lib):
        aio.add_slot(oa, s, kappa[c])
    aio.save_mat(os.environ['OA_OUT'], oa, 'OA-016 quadratic solve from %s on %s'
                 % (os.path.basename(os.environ['OA_ADDITION']), pb.mode))
    json.dump(dict(kappa={colib.slot_name(s): float(k) for s, k in zip(lib, kappa)},
                   F_norm=float(np.linalg.norm(F(z))), size=float(np.sqrt(z @ Gn @ z)),
                   verify_shift_max_nine=mx, verify_mdiff_own=own, seconds=time.perf_counter() - t00),
              open(os.environ['OA_OUT'].replace('.mat', '.json'), 'w'), indent=2)
    print('[out] %s (%.0f s)' % (os.environ['OA_OUT'], time.perf_counter() - t00))


if __name__ == '__main__':
    main()
