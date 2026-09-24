"""Numeric machine checks for DERIVATION.md with the PRODUCTION Jacobian. Prints `[Nn] ... OK`.

N1  Eq. (4): for every column of the addition, the discrete coefficient Phi_j^T G phi agrees with
    the continuous formula -Ts^2 (d theta_j/d v_j) SUM r_j^T W phi, per condition row within 5 %
    of that row's largest entry (OA-010).
N2  Eq. (5)/(17), ONE LINE PER CONDITION, on the DESIGN dataset, in the form the design solved:
      OA_TARGET=model     SUM_c A_jc a_c = 0 (every column from the addition's definition)
      OA_TARGET=measured  Phi_j^T Delta*_measured + SUM_s A_js a_s - t_j = 0, where the dataset
                          already contains the stateful part (absorber) and t = Phi^T Phi b_null
                          keeps the floor's own bias (OA-011)
    |row sum| <= 1e-8 SUM |row terms| (OA-010). The row terms are printed as a table.
N4  the fast operator is the production step: Delta through G equals Delta through a fresh
    block evaluation with the addition added to the input, relative 1e-6.
Env: OA_MODE (design dataset), OA_ADDITION (.mat), OA_TARGET, OA_NULL_JSON, OBC_REF_STRIDE.
"""
__project_origin__ = "added"

import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'design'))
from oa_core import Problem, COMBO_NAMES, ROWS           # noqa: E402
import addition_io as aio                                # noqa: E402
from vendor.obc import evaluate_step_rows                # noqa: E402
from model_augmentation.systems import gantry_ss as gss  # noqa: E402

FAIL = []


def ok(tag, cond, msg):
    print('[%s] %s %s' % (tag, msg, 'OK' if cond else 'FAIL'), flush=True)
    if not cond:
        FAIL.append(tag)


def decompose(oa):
    """The addition as named columns: one per stateless structural slot, one per absorber."""
    cols = []
    keys = {'oa_Ma0': 'M', 'oa_Ma1': 'MY', 'oa_Ma2': 'MY2', 'oa_Ca0': 'C', 'oa_Ka0': 'K', 'oa_Ka1': 'KY',
            'oa_Ka2': 'KY2'}
    for key, lab in keys.items():
        Mx = oa[key]
        for i in range(3):
            for j in range(i, 3):
                if Mx[i, j] != 0.0 or Mx[j, i] != 0.0:
                    assert Mx[i, j] == Mx[j, i], 'non-symmetric %s' % key
                    one = aio.empty(); one[key][i, j] = Mx[i, j]; one[key][j, i] = Mx[j, i]
                    cols.append(('%s%d%d%s' % (lab[0], i + 1, j + 1, lab[1:]), one, False))
    for s in range(aio.NS):
        if oa['oa_m'][0, s] != 0.0:
            one = aio.empty()
            for k in ('oa_m', 'oa_w', 'oa_b0', 'oa_b1'):
                one[k][..., s] = oa[k][..., s]
            cols.append(('absorber%d' % s, one, True))
    return cols


def main():
    pb = Problem()
    oa = aio.load_mat(os.environ['OA_ADDITION'])
    target = os.environ.get('OA_TARGET', 'model')
    Ts = pb.cfg.ts_new
    cols = decompose(oa)
    print('[setup] %s stride %d, addition %s, target %s: %d columns'
          % (pb.mode, pb.stride, os.path.basename(os.environ['OA_ADDITION']), target, len(cols)))
    phis = [aio.force_field(pb, one) for _, one, _ in cols]
    A = np.stack([pb.stats(pb.delta_of(f))[0] for f in phis], axis=1)          # 10 x ncol

    # ==== N1: continuous form of each coefficient
    mh, d, Lb = float(gss.mh), float(gss.d), float(gss.Lb)
    tc = pb.true_combo.numpy()
    m_tot, m_dif, J_eff = tc[6], tc[7], tc[8]
    Y = pb.Y
    Mk = np.zeros((len(Y), 3, 3))
    Mk[:, 0, 0] = m_tot + mh
    Mk[:, 0, 1] = Mk[:, 1, 0] = m_dif * Lb / 2 - mh * Y
    Mk[:, 1, 1] = J_eff + mh * d ** 2 + mh * Y ** 2
    Mk[:, 1, 2] = Mk[:, 2, 1] = -mh * d
    Mk[:, 2, 2] = mh
    Minv = np.linalg.inv(Mk)
    sx = np.asarray(pb.norm.std_x, float).reshape(6)
    sv, sq = sx[3:], sx[:3]
    # velocity rows carry Ts, position rows Ts^2/2 (RK4, sect. 8.1 of the prior write-up)
    wdiag = 1.0 / sv ** 2 + (Ts ** 2 / 4.0) / sq ** 2
    Wk = np.einsum('kji,j,kjl->kil', Minv, wdiag, Minv)
    q, qd, qdd = pb.q, pb.qd, pb.qdd
    X, Th, Yq = q.T; Xd, Thd, Yd = qd.T; Xdd, Thdd, Ydd = qdd.T
    z = np.zeros_like(X)
    r = {'kb_sum': [z, Th, z],
         'cg1': [Xd + Lb * Thd / 2, Lb * Xd / 2 + Lb ** 2 * Thd / 4, z],
         'cg2': [Xd - Lb * Thd / 2, -Lb * Xd / 2 + Lb ** 2 * Thd / 4, z],
         'cy': [z, z, Yd], 'cb_sum': [z, Thd, z],
         'mh': [Xdd - Y * Thdd, -Y * Xdd + (d ** 2 + Y ** 2) * Thdd - d * Ydd, Ydd - d * Thdd],
         'm_total': [Xdd, z, z], 'm_diff': [Lb * Thdd / 2, Lb * Xdd / 2, z],
         'J_eff': [z, Thdd, z], 'd': [z, 2 * mh * d * Thdd - mh * Ydd, -mh * Thdd]}
    v = pb.vbar.clone().requires_grad_(True)
    th = pb.blk.combinations_from_free(v.to(pb.device)).cpu()
    dth = np.array([float(torch.autograd.grad(th[j], v, retain_graph=True)[0][j])
                    for j in range(10)])
    Ac = np.zeros_like(A)
    for jn, name in enumerate(COMBO_NAMES):
        rj = np.stack(r[name], axis=1)
        for c, f in enumerate(phis):
            Ac[jn, c] = -Ts ** 2 * dth[jn] * np.einsum('ki,kij,kj->', rj, Wk, f)
    for jn, name in enumerate(COMBO_NAMES):
        rel = np.abs(A[jn] - Ac[jn]).max() / np.abs(A[jn]).max()
        ok('N1', rel < 5e-2, 'condition %-8s discrete vs continuous coefficient, max rel. diff %.2e:'
           % (name, rel))

    # ==== N1b (OA-022): the EXACT frozen-Y RK4 form. For x' = A x + B u with u held, RK4 is the
    #      4th-order Taylor polynomial: x+ = SUM_{n<=4} (Ts A)^n/n! x + SUM_{n<=3} Ts^(n+1) A^n/(n+1)! B u.
    #      Phi_j = S^-1 d x+ / d theta_j (times d theta_j / d v_j), Delta_phi = S^-1 Gamma_d B_f phi.
    import math
    ulog = pb.ref.u_stage.numpy() @ np.linalg.inv(pb.Pinv).T          # stage -> logical force
    Cm = np.array([[tc[1] + tc[2], (tc[1] - tc[2]) * Lb / 2, 0.0],
                   [(tc[1] - tc[2]) * Lb / 2, tc[4] + (tc[1] + tc[2]) * Lb ** 2 / 4, 0.0],
                   [0.0, 0.0, tc[3]]])
    Km = np.zeros((3, 3)); Km[1, 1] = tc[0]
    # dM_j, dC_j, dK_j in COMBO_NAMES order (D2)
    def dmats(name, Yc):
        n_ = len(Yc); Z = np.zeros((n_, 3, 3)); dMj = Z.copy(); dCj = Z.copy(); dKj = Z.copy()
        if name == 'kb_sum': dKj[:, 1, 1] = 1
        elif name == 'cg1':
            dCj[:, 0, 0] = 1; dCj[:, 0, 1] = dCj[:, 1, 0] = Lb / 2; dCj[:, 1, 1] = Lb ** 2 / 4
        elif name == 'cg2':
            dCj[:, 0, 0] = 1; dCj[:, 0, 1] = dCj[:, 1, 0] = -Lb / 2; dCj[:, 1, 1] = Lb ** 2 / 4
        elif name == 'cy': dCj[:, 2, 2] = 1
        elif name == 'cb_sum': dCj[:, 1, 1] = 1
        elif name == 'mh':
            dMj[:, 0, 0] = 1; dMj[:, 0, 1] = dMj[:, 1, 0] = -Yc; dMj[:, 1, 1] = d ** 2 + Yc ** 2
            dMj[:, 1, 2] = dMj[:, 2, 1] = -d; dMj[:, 2, 2] = 1
        elif name == 'm_total': dMj[:, 0, 0] = 1
        elif name == 'm_diff': dMj[:, 0, 1] = dMj[:, 1, 0] = Lb / 2
        elif name == 'J_eff': dMj[:, 1, 1] = 1
        elif name == 'd':
            dMj[:, 1, 1] = 2 * mh * d; dMj[:, 1, 2] = dMj[:, 2, 1] = -mh
        return dMj, dCj, dKj
    mv = lambda Mat, v: np.einsum('kij,kj->ki', Mat, v)            # noqa: E731
    sx2 = sx ** 2
    Aex = np.zeros_like(A)
    nch = 65536
    for k0 in range(0, len(Y), nch):
        sl = slice(k0, min(k0 + nch, len(Y)))
        Mi = Minv[sl]; Yc = Y[sl]; nk = len(Yc)
        Am = np.zeros((nk, 6, 6)); Am[:, :3, 3:] = np.eye(3)
        Am[:, 3:, :3] = -Mi @ Km; Am[:, 3:, 3:] = -Mi @ Cm
        xk = np.hstack([q[sl], qd[sl]])
        wk = np.hstack([np.zeros((nk, 3)), mv(Mi, ulog[sl])])       # B u
        a_ = [xk]; c_ = [wk]
        for _ in range(3):
            a_.append(mv(Am, a_[-1])); c_.append(mv(Am, c_[-1]))
        def gamma(v):                                               # Gamma_d v
            out = np.zeros_like(v); p_ = v
            for n_ in range(4):
                out += Ts ** (n_ + 1) / math.factorial(n_ + 1) * p_
                p_ = mv(Am, p_)
            return out
        g_c = [gamma(np.hstack([np.zeros((nk, 3)), mv(Mi, f[sl])])) for f in phis]
        for jn, name in enumerate(COMBO_NAMES):
            dMj, dCj, dKj = dmats(name, Yc)
            dA = np.zeros((nk, 6, 6))
            dA[:, 3:, :3] = Mi @ dMj @ Mi @ Km - Mi @ dKj
            dA[:, 3:, 3:] = Mi @ dMj @ Mi @ Cm - Mi @ dCj
            s_j = np.zeros((nk, 6))
            # d(A^n x): SUM_i A^i dA A^(n-1-i) x, and the same for the input term
            for n_ in range(1, 5):
                acc = np.zeros((nk, 6))
                for i in range(n_):
                    t_ = mv(dA, a_[n_ - 1 - i])
                    for _ in range(i):
                        t_ = mv(Am, t_)
                    acc += t_
                s_j += Ts ** n_ / math.factorial(n_) * acc
            for n_ in range(1, 4):
                acc = np.zeros((nk, 6))
                for i in range(n_):
                    t_ = mv(dA, c_[n_ - 1 - i])
                    for _ in range(i):
                        t_ = mv(Am, t_)
                    acc += t_
                s_j += Ts ** (n_ + 1) / math.factorial(n_ + 1) * acc
            dB = np.hstack([np.zeros((nk, 3)), -mv(Mi @ dMj @ Mi, ulog[sl])])
            s_j += gamma(dB)
            for c in range(len(phis)):
                Aex[jn, c] += dth[jn] * np.sum(s_j * g_c[c] / sx2[None, :])
    for jn, name in enumerate(COMBO_NAMES):
        rel = np.abs(A[jn] - Aex[jn]).max() / np.abs(A[jn]).max()
        ok('N1b', rel < 1e-3, 'condition %-8s exact frozen-Y RK4 form vs production, max rel. diff %.2e:'
           % (name, rel))

    # ==== N1c (OA-023): an independent numpy implementation of the LPV RK4 map (Eq. 4''), M
    #      re-evaluated at every stage's Y, differentiated by central differences in each combination
    def rk4(th, xk, uk):
        kbs, c1, c2, cyy, cbs, mhh, mt, md, Je, dd = th
        def F(xx):
            Yy = xx[:, 2]
            M_ = np.zeros((len(Yy), 3, 3))
            M_[:, 0, 0] = mt + mhh
            M_[:, 0, 1] = M_[:, 1, 0] = md * Lb / 2 - mhh * Yy
            M_[:, 1, 1] = Je + mhh * dd ** 2 + mhh * Yy ** 2
            M_[:, 1, 2] = M_[:, 2, 1] = -mhh * dd
            M_[:, 2, 2] = mhh
            C_ = np.array([[c1 + c2, (c1 - c2) * Lb / 2, 0], [(c1 - c2) * Lb / 2, cbs + (c1 + c2) * Lb ** 2 / 4, 0],
                           [0, 0, cyy]])
            rhs = uk - xx[:, 3:] @ C_.T - np.stack([np.zeros(len(Yy)), kbs * xx[:, 1], np.zeros(len(Yy))], 1)
            return np.hstack([xx[:, 3:], np.linalg.solve(M_, rhs[:, :, None])[:, :, 0]])
        k1 = F(xk); k2 = F(xk + Ts / 2 * k1); k3 = F(xk + Ts / 2 * k2); k4 = F(xk + Ts * k3)
        return xk + Ts / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    # RAM (R-033c was killed at 2.33 GB holding all columns): one sensitivity row and one column
    # field at a time; the columns are recomputed with the fast operator (cheap).
    del Wk, Minv
    import gc; gc.collect()
    Aind = np.zeros_like(A)
    for jn in range(10):
        h = 1e-4 * abs(tc[jn])
        tp = tc.copy(); tp[jn] += h
        tm = tc.copy(); tm[jn] -= h
        s_j = np.zeros((len(Y), 6))
        for k0 in range(0, len(Y), nch):
            sl = slice(k0, min(k0 + nch, len(Y)))
            xk = np.hstack([q[sl], qd[sl]]); uk = ulog[sl]
            s_j[sl] = (rk4(tp, xk, uk) - rk4(tm, xk, uk)) / (2 * h) / sx[None, :]
        for c in range(len(phis)):
            Aind[jn, c] = dth[jn] * np.sum(s_j * pb.delta_of(phis[c]).reshape(-1, 6))
        del s_j
    for jn, name in enumerate(COMBO_NAMES):
        rel = np.abs(A[jn] - Aind[jn]).max() / np.abs(A[jn]).max()
        ok('N1c', rel < 1e-5, 'condition %-8s independent LPV RK4 (Eq. 4\'\') vs production, max rel. diff %.2e:'
           % (name, rel))

    # ==== the rows as solved by the design
    names = [n_ for n_, _, _ in cols]
    if target == 'measured':
        stateless = [c for c, (_, _, st) in enumerate(cols) if not st]
        Dm = pb.measured_delta()
        js = json.load(open(os.environ['OA_NULL_JSON']))
        p = next(p for p in js['predictions'] if p['tag'] == 'J @ theta*, Delta* (stencil)')
        bt = torch.tensor(p['dtheta'], dtype=torch.float64)
        t = pb.basis.jt(pb.basis.apply(bt)).numpy()
        T = np.column_stack([pb.stats(Dm)[0], A[:, stateless], -t])
        tnames = ['Phi^T Delta*_meas'] + [names[c] for c in stateless] + ['-t (floor)']
    else:
        T = A
        tnames = names
    print('\n| condition | ' + ' | '.join(tnames) + ' | row sum |')
    print('|-|' + '-|' * (len(tnames) + 1))
    for jn, name in enumerate(COMBO_NAMES):
        print('| %s | ' % name + ' | '.join('%+.3e' % a_ for a_ in T[jn]) + ' | %+.1e |' % T[jn].sum())
    print()
    for jn, name in enumerate(COMBO_NAMES):
        rel = abs(T[jn].sum()) / max(np.abs(T[jn]).sum(), 1e-300)
        ok('N2', rel <= 1e-8, 'condition %-8s row sum %+.3e, relative %.2e:' % (name, T[jn].sum(), rel))

    # ==== N6: Eq. (13)'s constant term c0 = Phi_0^T (Delta*_null - Phi_0 b_null) vanishes, because
    #      b_null = Phi_0^+ Delta*_null (needs OA_NULL_JSON and the null control as OA_MODE)
    if os.environ.get('OA_NULL_JSON'):
        js = json.load(open(os.environ['OA_NULL_JSON']))
        p = next(p for p in js['predictions'] if p['tag'] == 'J @ theta*, Delta* (stencil)')
        bt = torch.tensor(p['dtheta'], dtype=torch.float64)
        Dn = torch.from_numpy(pb.measured_delta())
        c0 = (pb.basis.jt(Dn) - pb.basis.jt(pb.basis.apply(bt))).numpy()
        rel = np.linalg.norm(c0) / np.linalg.norm(pb.basis.jt(Dn).numpy())
        ok('N6', rel < 1e-8, 'c0 = Phi_0^T (Delta*_null - Phi_0 b_null), relative to Phi_0^T Delta*_null: %.2e:'
           % rel)

    # ==== N4: fast operator vs fresh block evaluation, on the whole addition's force
    fa = sum(phis)
    D_fast = pb.delta_of(fa)
    du = torch.from_numpy((fa @ pb.Pinv.T) / pb.std_u).to(torch.float64)
    Z = pb.ref.Z_step.clone().to(torch.float64)
    Z[:, 6:9, 0] += du
    D_blk = (evaluate_step_rows(pb.step, pb.vbar.to(pb.device), Z, ROWS,
                                chunk=pb.cfg.obc_ref_chunk, device=pb.device).cpu()
             - pb.base).numpy()
    rel = np.linalg.norm(D_fast - D_blk) / np.linalg.norm(D_blk)
    ok('N4', rel < 1e-6, 'Delta via G[k] vs fresh production-step evaluation, relative %.2e:' % rel)
    del D_blk, Z

    # ==== N5: Eq. (12), the sensitivity is affine in the input: Phi(u - 2 phi) - Phi(u) equals
    #      2 (Phi(u - phi) - Phi(u)) to round-off (a curvature in u would break the exactness of (13))
    from design_quadratic import jac
    pb.basis = None
    J0 = jac(pb, None)
    d1 = jac(pb, fa); np.subtract(d1, J0, out=d1)
    d2 = jac(pb, 2.0 * fa); np.subtract(d2, J0, out=d2)
    del J0
    rel = np.linalg.norm(d2 - 2.0 * d1) / np.linalg.norm(d2)
    ok('N5', rel < 1e-6, 'Phi(x, u - 2 phi) - Phi(x, u) = 2 (Phi(x, u - phi) - Phi(x, u)), relative %.2e:' % rel)
    print('\nD_NUMERIC %s (%d failed)' % ('ALL OK' if not FAIL else 'FAIL', len(FAIL)))
    sys.exit(1 if FAIL else 0)


if __name__ == '__main__':
    main()
