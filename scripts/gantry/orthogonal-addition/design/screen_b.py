"""G2 screen, route B: a STATELESS addition on the baseline's structural zeros (OA-006).

For every candidate column c (a unit generalised force field phi_c on the reference tuples) the
condition coefficient is A[:, c] = J^T Delta_c with Delta_c = G phi_c (discrete production
Jacobian). The ten orthogonality conditions on an addition sum_c v_c phi_c are A v = 0, so the
admissible class is null(A). This script:
  1. builds the columns: the four PHYSICAL mount terms (k_x, k_y, k_xp, gamma) and every
     legitimate stateless slot up to Y^2 (columns.py);
  2. physical-only: the best the four physical terms can do (min rho*, generalised eigenproblem);
  3. full library: rank of A, dimension of null(A), and the member of null(A) whose energy is
     most concentrated on the physical terms;
  4. rho*, ||Delta||, and the predicted bias J^+ Delta of that member, scaled to the OA-004
     minimum ||Delta*|| >= 8.49 (stride 1; at stride s the minimum scales by 1/sqrt(s)).
Writes outputs/screen_b_<tag>.json and design/route_b_member_<tag>.npz.
Run through tools/wd.sh. Env: OA_MODE (dataset), OBC_REF_STRIDE.
"""
__project_origin__ = "added"

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from oa_core import Problem, COMBO_NAMES, OA           # noqa: E402
import columns as colib                                # noqa: E402

DELTA_MIN_STRIDE1 = 8.49          # OA-004
LINE = '=' * 92


def banner(t):
    print('\n' + LINE + '\n' + t + '\n' + LINE, flush=True)


def build_columns(pb, names_forces):
    """A (10 x n), UtD (10 x n), norms, Gram (n x n) for a list of (name, force field)."""
    n = len(names_forces)
    A = np.zeros((10, n)); UtD = np.zeros((10, n)); nrm = np.zeros(n)
    D32 = []
    for c, (nm, f) in enumerate(names_forces):
        D = pb.delta_of(f)
        A[:, c], UtD[:, c], nrm[c] = pb.stats(D)
        D32.append(D.astype(np.float32))
    Dm = np.stack(D32, axis=1)
    del D32
    Gram = (Dm.T @ Dm).astype(np.float64)       # float32 product: used for scaling/selection only
    del Dm
    return A, UtD, nrm, Gram


def min_rho(UtD, Gram, idx):
    """min over v on columns idx of ||P D||^2/||D||^2 = v'Qv / v'Gv (generalised eigenproblem)."""
    from scipy.linalg import eigh
    Q = UtD[:, idx].T @ UtD[:, idx]
    G = Gram[np.ix_(idx, idx)]
    w, V = eigh(Q, G)
    return float(np.sqrt(max(w[0], 0.0))), V[:, 0]


def main():
    t0 = time.perf_counter()
    pb = Problem()
    tag = os.environ.get('OA_TAG', '%s_s%d' % (pb.mode, pb.stride))
    dmin = DELTA_MIN_STRIDE1 / np.sqrt(pb.stride)

    banner('COLUMNS  (%d tuples, stride %d)' % (pb.ref.n, pb.stride))
    phys = colib.physical_mount(pb)
    lib = colib.stateless_library(max_ypow=2)
    if os.environ.get('OA_LIB') == 'M':
        lib = [s for s in lib if s[0] == 'M']            # OA-020: inertial slots only
    cols = [(k, v) for k, v in phys.items()]
    cols += [(colib.slot_name(s), pb.stateless_force(s[0], s[1], s[2], s[3])) for s in lib]
    names = [c[0] for c in cols]
    n_phys = len(phys)
    A, UtD, nrm, Gram = build_columns(pb, cols)
    rho_alone = np.sqrt(np.sum(UtD ** 2, axis=0)) / nrm
    print('  %-8s %12s %10s' % ('column', '||Delta_c||', 'rho alone'))
    for c, nm in enumerate(names):
        print('  %-8s %12.4e %10.4f' % (nm, nrm[c], rho_alone[c]))

    banner('ROUTE B, PHYSICAL TERMS ONLY  (k_x, k_y, k_xp, gamma: 4 unknowns, 10 conditions)')
    idx = list(range(n_phys))
    As = A[:, idx] / nrm[idx]
    sv = np.linalg.svd(As, compute_uv=False)
    print('  singular values of the column-normalised 10 x 4 condition matrix: %s'
          % np.array2string(sv, precision=3))
    r_phys, v_phys = min_rho(UtD, Gram, idx)
    print('  best combination: rho* = %.4e  (needs < 1e-3)  v = %s'
          % (r_phys, np.array2string(v_phys / np.abs(v_phys).max(), precision=4)))

    banner('ROUTE B, FULL LEGITIMATE LIBRARY  (%d columns)' % len(names))
    An = A / nrm[None, :]                    # columns normalised to unit ||Delta_c||
    U_, S_, Vt_ = np.linalg.svd(An)
    rank = int((S_ > S_[0] * 1e-10).sum())
    Nsp = Vt_[rank:].T                       # null space in normalised coordinates
    print('  rank %d of %d columns: admissible class has dimension %d' % (rank, len(names),
                                                                           Nsp.shape[1]))
    print('  singular values: %s' % np.array2string(S_, precision=3))
    # member with the largest share of its (normalised) coefficient energy on the physical terms
    Np = Nsp[:n_phys, :]
    w, V = np.linalg.eigh(Np.T @ Np)
    z = V[:, -1]
    vn = Nsp @ z
    share = float(w[-1])
    v = vn / nrm                              # back to unit-force coefficients
    D = pb.delta_of(sum(v[c] * cols[c][1] for c in range(len(cols))))
    rho = pb.rho(D)
    dn = float(np.linalg.norm(D))
    s = dmin / dn * 1.5                       # 1.5x the minimum, margin for the closed loop
    pct, own = pb.bias_pct(s * D)
    print('  most-physical member: physical share of coefficient energy %.3f' % share)
    print('  rho* = %.3e   ||Delta|| (scaled to 1.5 x minimum) = %.4e   (minimum %.4e)'
          % (rho, s * dn, dmin))
    print('  predicted bias J^+ Delta (combo %%): %s   m_diff own scale %+.3e %%'
          % ('  '.join('%s %+.2e' % (n_, p) for n_, p in zip(COMBO_NAMES, pct)), own))
    print('  coefficients (unit-force units, scaled):')
    for c, nm in enumerate(names):
        if abs(vn[c]) > 1e-3 * np.abs(vn).max():
            print('    %-8s %+.6e   (normalised %+.4f)' % (nm, s * v[c], vn[c]))

    # ==== OA-009: least actuator force per unit of learnable discrepancy ====
    banner('ROUTE B, OA-009 MEMBER  (max ||Delta||^2 / stage-force energy on null(A))')
    # The four physical terms are exact combinations of library slots (k_x = K11, k_y = K33,
    # k_xp = K11 - 2 K12Y + K22Y2, gamma = M13 + M22Y), so they are left out here: with them the
    # null space holds zero-force, zero-Delta directions and the force Gram is singular.
    il = list(range(n_phys, len(names)))
    Fs = np.stack([(cols[c][1] @ pb.Pinv.T).reshape(-1).astype(np.float32) for c in il], axis=1)
    Fg = (Fs.T @ Fs).astype(np.float64)                              # stage-force Gram
    del Fs
    Gl = Gram[np.ix_(il, il)] / np.outer(nrm[il], nrm[il])           # normalised coordinates
    Fl = Fg / np.outer(nrm[il], nrm[il])
    Al = An[:, il]
    _u, Sl, Vtl = np.linalg.svd(Al)
    rl = int((Sl > Sl[0] * 1e-10).sum())
    Nl = Vtl[rl:].T
    print('  library only: %d columns, rank %d, null dim %d' % (len(il), rl, Nl.shape[1]))
    from scipy.linalg import eigh
    w2, V2 = eigh(Nl.T @ Gl @ Nl, Nl.T @ Fl @ Nl)
    vn2 = np.zeros(len(names)); vn2[il] = Nl @ V2[:, -1]
    v2 = np.zeros(len(names)); v2[il] = vn2[il] / nrm[il]
    fa = sum(v2[c] * cols[c][1] for c in range(len(cols)))
    D2 = pb.delta_of(fa)
    rho2 = pb.rho(D2)
    dn2 = float(np.linalg.norm(D2))
    s2 = 1.5 * dmin / dn2
    fst = s2 * (fa @ pb.Pinv.T)
    pct2, own2 = pb.bias_pct(s2 * D2)
    print('  rho* = %.3e   ||Delta|| scaled %.4e   ||Delta||/||f|| gain %.4e' % (rho2, s2 * dn2,
                                                                                np.sqrt(w2[-1])))
    print('  addition stage force on the data: rms %s N, peak %s N'
          % (np.array2string(np.sqrt((fst ** 2).mean(0)), precision=2),
             np.array2string(np.abs(fst).max(0), precision=1)))
    print('  predicted bias (combo %%) max |.| %.2e, m_diff own %+.2e %%' % (np.abs(pct2).max(), own2))
    for c, nm in enumerate(names):
        if abs(vn2[c]) > 1e-3 * np.abs(vn2).max():
            print('    %-8s %+.6e   (normalised %+.4f)' % (nm, s2 * v2[c], vn2[c]))
    import addition_io as aio
    from model_augmentation.systems import gantry_ss as gss
    oa = aio.empty()
    lib_slots = {colib.slot_name(s_): s_ for s_ in lib}
    for c, nm in enumerate(names):
        if nm in lib_slots:
            aio.add_slot(oa, lib_slots[nm], s2 * v2[c])
        else:
            aio.add_physical(oa, nm, s2 * v2[c], float(gss.mh), float(gss.d))
    Dchk = pb.delta_of(aio.force_field(pb, oa))
    print('  round trip through the oa_* arrays: rho* %.3e, ||Delta|| %.4e (must match)'
          % (pb.rho(Dchk), float(np.linalg.norm(Dchk))))
    aio.save_mat(os.path.join(HERE, 'addition_route_b_%s.mat' % tag), oa,
                 'route B, OA-009 member, screened on %s stride %d' % (pb.mode, pb.stride))

    out = dict(mode=pb.mode, stride=pb.stride, n=pb.ref.n, names=names, n_phys=n_phys,
               oa009=dict(rho=rho2, delta_norm_scaled=s2 * dn2, coef=(s2 * v2).tolist(),
                          force_rms=np.sqrt((fst ** 2).mean(0)).tolist(),
                          force_peak=np.abs(fst).max(0).tolist(), bias_pct=pct2.tolist(),
                          m_diff_own=own2),
               col_norm=nrm.tolist(), rho_alone=rho_alone.tolist(), A=A.tolist(),
               phys_only=dict(sv=sv.tolist(), rho=r_phys, v=v_phys.tolist()),
               full=dict(rank=rank, null_dim=int(Nsp.shape[1]), sv=S_.tolist(),
                         phys_share=share, rho=rho, delta_norm_scaled=s * dn, delta_min=dmin,
                         coef=(s * v).tolist(), bias_pct=pct.tolist(), m_diff_own=own),
               seconds=time.perf_counter() - t0)
    os.makedirs(os.path.join(OA, 'outputs'), exist_ok=True)
    with open(os.path.join(OA, 'outputs', 'screen_b_%s.json' % tag), 'w') as fh:
        json.dump(out, fh, indent=2)
    np.savez(os.path.join(HERE, 'route_b_member_%s.npz' % tag), names=np.array(names),
             coef=s * v)
    print('\n[out] outputs/screen_b_%s.json, design/route_b_member_%s.npz  (%.0f s)'
          % (tag, tag, time.perf_counter() - t0))


if __name__ == '__main__':
    main()
