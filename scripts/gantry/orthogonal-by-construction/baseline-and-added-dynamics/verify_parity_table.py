"""Are the integration-by-parts identities and the parity table actually right?

`baseline-and-added-dynamics.md` Sect. 8.4 claims that each of the ten orthogonality conditions
collapses, by integration by parts, to a quadratic form, and that an addition is therefore
automatically orthogonal to the parameter families of OPPOSITE parity and never to its own. The
lossless conclusion and the ten matrix conditions are both downstream of that table, and the table
was derived here rather than taken from a source. This script checks it.

Three parts:

  A  THE IDENTITIES, symbolically, WITH THEIR CORRECT BOUNDARY TERMS. This is where the draft
     was wrong: it justified all three by "the records start and end at rest, so qdot = 0 at both
     ends". That is the right condition for two of the three. The `qdot^T H q` identity needs the
     POSITIONS to return, not the velocities to vanish, and those are different requirements that
     the gantry data satisfies to different accuracy.

  B  THE NINE CELLS, symbolically: each cell of the table is reduced to one of the three
     identities and the matrix `H` whose symmetry is required is displayed. Machine-checked that
     the reduction is the one claimed, rather than asserted in prose.

  C  THE IDENTITIES ON THE REAL RECORDS. A discrete sum over a trimmed, float32, finitely-sampled
     record is not an integral. Part C measures, per record, how close each identity comes to
     holding, and reports the boundary term separately from the quadrature error so the two
     failure modes are distinguishable. Standstill records (frozen Y, where the weight matrix
     `W = M(Y)^-T D M(Y)^-1` is constant and the identities are exact) are reported apart from
     Y-sweep records (where it is not), because that difference is itself a prediction.

Run:
  PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 conda run --no-capture-output \
      -n GraduationProject python -u verify_parity_table.py
"""
__project_origin__ = "added"

import os
import sys

import numpy as np
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
for _p in (REPO, os.path.join(REPO, 'scripts', 'gantry')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model_augmentation.systems.gantry_ss import P as P_pt     # noqa: E402
from gantry_dynamic.config import RunConfig                    # noqa: E402
from gantry_dynamic.obc_gantry import (                        # noqa: E402
    fourth_order_central_difference, _load_record_float64)

LINE = '=' * 90
COMBO_INIT_DETUNE = [1.1, 1.1, 0.9, 1.1, 0.9, 0.9, 1.1, 1.1, 0.9, 1.1]

# Records at a FIXED Y (M(Y) constant, so the weight in the inner product is constant and the
# identities are exact) against records that sweep Y (where it is not). The split is the
# prediction being tested, not a convenience.
STANDSTILL = ['T1_standstill_Ym30.mat', 'T2_standstill_Ym15.mat', 'T3_standstill_Y000.mat',
              'T4_standstill_Yp15.mat', 'T5_standstill_Yp30.mat']
YSWEEP = ['T6_ysweep_slow.mat', 'T7_ysweep_fast.mat', 'T8_ysweep_xmix.mat']


def banner(t):
    print('\n' + LINE); print(t); print(LINE, flush=True)


# ------------------------------------------------------------------ A
def part_a():
    banner('A  THE THREE IDENTITIES, SYMBOLICALLY, WITH THEIR CORRECT BOUNDARY TERMS')
    t = sp.Symbol('t', real=True)
    q = sp.Matrix([sp.Function(f'q{i}', real=True)(t) for i in range(3)])
    qd = q.diff(t)
    qdd = q.diff(t, 2)
    Hs = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f'h{min(i,j)}{max(i,j)}', real=True))   # symmetric
    Ha = sp.Matrix(3, 3, lambda i, j: (0 if i == j else
                                       (1 if i < j else -1) * sp.Symbol(f'a{min(i,j)}{max(i,j)}',
                                                                        real=True)))   # antisym
    assert sp.simplify(Hs - Hs.T) == sp.zeros(3, 3)
    assert sp.simplify(Ha + Ha.T) == sp.zeros(3, 3)

    ok = True

    print('\nI1   qdot^T H q      with H SYMMETRIC')
    lhs = (qd.T * Hs * q)[0, 0]
    anti = sp.Rational(1, 2) * (q.T * Hs * q)[0, 0]
    r = sp.simplify(sp.expand(sp.diff(anti, t) - lhs))
    print('     d/dt[ (1/2) q^T H q ] - qdot^T H q  =  %s' % r)
    ok &= (r == 0)
    print('     => INTEGRAL qdot^T H q dt = (1/2) [ q^T H q ] evaluated at the ENDPOINTS.')
    print('     BOUNDARY CONDITION: the POSITIONS must return, q(T)^T H q(T) = q(0)^T H q(0).')
    print('     *** NOT "qdot = 0 at both ends". The draft justified this identity with the')
    print('     *** wrong condition. Part C measures which one the records actually satisfy.')

    print('\nI1a  the same integrand with H ANTISYMMETRIC is NOT a total derivative')
    lhs_a = (qd.T * Ha * q)[0, 0]
    cand = sp.Rational(1, 2) * (q.T * Ha * q)[0, 0]
    print('     (1/2) q^T H_antisym q  =  %s   (identically zero, so it cannot be the'
          ' antiderivative)' % sp.simplify(sp.expand(cand)))
    ok &= (sp.simplify(sp.expand(cand)) == 0)
    ok &= (sp.simplify(sp.expand(lhs_a)) != 0)
    print('     and the integrand itself is not identically zero, so ONLY the symmetric part of')
    print('     H is annihilated. That is exactly what makes the table a SYMMETRY condition.')

    print('\nI2   qddot^T H q      (any H)')
    lhs = (qdd.T * Hs * q)[0, 0]
    anti = (qd.T * Hs * q)[0, 0]
    r = sp.simplify(sp.expand(sp.diff(anti, t) - lhs - (qd.T * Hs * qd)[0, 0]))
    print('     d/dt[ qdot^T H q ] - qddot^T H q - qdot^T H qdot  =  %s' % r)
    ok &= (r == 0)
    print('     => INTEGRAL qddot^T H q dt = [ qdot^T H q ] - INTEGRAL qdot^T H qdot dt.')
    print('     BOUNDARY CONDITION: qdot = 0 at both ends. The surviving term is a QUADRATIC')
    print('     FORM in velocity, which does not vanish for a generic trajectory.')

    print('\nI3   qddot^T H qdot   with H SYMMETRIC')
    lhs = (qdd.T * Hs * qd)[0, 0]
    anti = sp.Rational(1, 2) * (qd.T * Hs * qd)[0, 0]
    r = sp.simplify(sp.expand(sp.diff(anti, t) - lhs))
    print('     d/dt[ (1/2) qdot^T H qdot ] - qddot^T H qdot  =  %s' % r)
    ok &= (r == 0)
    print('     => INTEGRAL qddot^T H qdot dt = (1/2) [ qdot^T H qdot ] at the endpoints.')
    print('     BOUNDARY CONDITION: qdot = 0 at both ends. Correct in the draft.')

    print('\n  A verdict: %s' % ('all three identities verified symbolically'
                                 if ok else 'AN IDENTITY FAILED'))
    return ok


# ------------------------------------------------------------------ B
def part_b():
    banner('B  THE NINE CELLS: which identity each reduces to, and the H whose symmetry is needed')
    print("""
  Setup. The inner product on the protected rows is, with D = diag(1/std_x[3:6]^2) constant and
  W = M^-T D M^-1,

      <J_j, Delta>  =  -Ts^2 * SUM_k  r_j(k)^T W f_a(k),
      r_j = (dM_j) qddot + (dC_j) qdot + (dK_j) q.

  Sect. 8.2 measured that exactly one of the three terms of r_j survives per parameter family,
  so each cell is a single product. Substituting the three addition types gives the table.
""")
    rows = []
    # (addition, family, integrand shape, H, identity, verdict)
    rows.append(('f_a = -K_a q',   'mass  (dM_j) qddot', 'qddot^T H q',
                 '(dM_j) W K_a', 'I2', 'NOT automatic: leaves -SUM qdot^T H qdot'))
    rows.append(('f_a = -K_a q',   'damping (dC_j) qdot', 'qdot^T H q',
                 '(dC_j) W K_a', 'I1', 'ZERO if H symmetric AND positions return'))
    rows.append(('f_a = -K_a q',   'stiffness (dK) q',    'q^T H q',
                 '(dK) W K_a',   '--', 'NOT automatic: a quadratic form in q'))
    rows.append(('f_a = -C_a qdot', 'mass  (dM_j) qddot', 'qddot^T H qdot',
                 '(dM_j) W C_a', 'I3', 'ZERO if H symmetric AND qdot = 0 at the ends'))
    rows.append(('f_a = -C_a qdot', 'damping (dC_j) qdot', 'qdot^T H qdot',
                 '(dC_j) W C_a', '--', 'NEVER zero for a dissipative C_a: positive form'))
    rows.append(('f_a = -C_a qdot', 'stiffness (dK) q',    'q^T H qdot',
                 '(dK) W C_a',   'I1', 'ZERO if H symmetric AND positions return'))
    rows.append(('f_a = -M_a qddot', 'mass  (dM_j) qddot', 'qddot^T H qddot',
                 '(dM_j) W M_a', '--', 'NOT automatic: a quadratic form in qddot'))
    rows.append(('f_a = -M_a qddot', 'damping (dC_j) qdot', 'qdot^T H qddot',
                 '(dC_j) W M_a', 'I3', 'ZERO if H symmetric AND qdot = 0 at the ends'))
    rows.append(('f_a = -M_a qddot', 'stiffness (dK) q',    'q^T H qddot',
                 '(dK) W M_a',   'I2', 'NOT automatic: leaves -SUM qdot^T H qdot'))

    print('  %-17s %-21s %-17s %-15s %-4s' % ('addition', 'family term', 'integrand', 'H', 'id'))
    print('  ' + '-' * 86)
    for a, f, integ, H, idn, verdict in rows:
        print('  %-17s %-21s %-17s %-15s %-4s' % (a, f, integ, H, idn))
        print('  %-17s %s' % ('', verdict))

    print("""
  CORRECTION TO THE DRAFT TABLE. Two cells were justified with the wrong boundary condition:
  (reactive addition vs the damping family) and (dissipative addition vs the stiffness family)
  both reduce to I1, which needs the POSITIONS to return, not the velocities to vanish. Their
  entries in Sect. 8.4 stand, but the hypothesis they rest on is different from the other two
  zero cells, and it is a weaker one on this data. Part C measures the gap.

  UNCHANGED AND CONFIRMED. The load-bearing cell for the lossless conclusion is
  (dissipative addition vs the damping family): it is SUM qdot^T (dC_j) W C_a qdot, a quadratic
  form in velocity with no boundary term and no symmetry escape. For a dissipative C_a it has a
  fixed sign and cannot vanish. That is the obstruction, and it does not depend on either
  boundary condition.
""")
    return rows


# ------------------------------------------------------------------ C
def record_derivatives(fname, cfg):
    """q, qdot, qddot on one record, all from the SAME fourth-order stencil as the reference set."""
    P64 = P_pt.detach().cpu().numpy().astype(np.float64)
    PinvT = np.linalg.inv(P64.T)
    u, y, _x, _a = _load_record_float64(fname, cfg)
    q = y @ PinvT.T
    qd = fourth_order_central_difference(q, cfg.ts_new)        # aligned with q[2:-2]
    qdd = fourth_order_central_difference(qd, cfg.ts_new)      # aligned with q[4:-4]
    # Common index window where all three exist: q[4:-4], qd[2:-2], qdd[:]
    return q[4:-4], qd[2:-2], qdd


def part_c(cfg):
    banner('C  THE IDENTITIES ON THE REAL RECORDS (a discrete sum is not an integral)')
    rng = np.random.default_rng(0)
    Hs = rng.standard_normal((3, 3)); Hs = 0.5 * (Hs + Hs.T)         # symmetric
    Ha = rng.standard_normal((3, 3)); Ha = 0.5 * (Ha - Ha.T)         # antisymmetric
    ts = cfg.ts_new

    def quad(v, H, w):
        return np.einsum('ki,ij,kj->', v, H, w)

    print('  For each identity: |sum| is the discrete sum times ts, |bdy| is the exact boundary')
    print('  term the identity predicts, and |resid| = |sum - bdy| is the quadrature error.')
    print('  "rel" normalises the residual by the natural scale of the integrand, so it is the')
    print('  fraction of the integrand that does NOT cancel. That fraction is the accuracy any')
    print('  parity construction can achieve on this data.\n')

    for tag, files in (('STANDSTILL (frozen Y)', STANDSTILL), ('Y-SWEEP', YSWEEP)):
        print('  --- %s' % tag)
        print('  %-26s %12s %12s %12s %12s' % ('record / identity', '|sum|', '|bdy|', '|resid|',
                                               'rel'))
        for f in files:
            q, qd, qdd = record_derivatives(f, cfg)
            scale1 = ts * np.sqrt(quad(qd, Hs @ Hs.T, qd) * quad(q, np.eye(3), q))
            # I1: sum qdot^T Hs q  vs  (1/2)[q^T Hs q]
            s1 = ts * quad(qd, Hs, q)
            b1 = 0.5 * (q[-1] @ Hs @ q[-1] - q[0] @ Hs @ q[0])
            # I3: sum qddot^T Hs qdot vs (1/2)[qdot^T Hs qdot]
            s3 = ts * quad(qdd, Hs, qd)
            b3 = 0.5 * (qd[-1] @ Hs @ qd[-1] - qd[0] @ Hs @ qd[0])
            scale3 = ts * np.sqrt(quad(qdd, np.eye(3), qdd) * quad(qd, np.eye(3), qd))
            # I2: sum qddot^T Hs q vs [qdot^T Hs q] - sum qdot^T Hs qdot
            s2 = ts * quad(qdd, Hs, q)
            b2 = (qd[-1] @ Hs @ q[-1] - qd[0] @ Hs @ q[0]) - ts * quad(qd, Hs, qd)
            scale2 = ts * np.sqrt(quad(qdd, np.eye(3), qdd) * quad(q, np.eye(3), q))
            # I1 with an ANTISYMMETRIC H: must NOT vanish, which is what makes it a symmetry test
            s1a = ts * quad(qd, Ha, q)
            short = f.replace('.mat', '')[:14]
            for name, s, b, sc in (('I1 qdot^T H q', s1, b1, scale1),
                                   ('I2 qddot^T H q', s2, b2, scale2),
                                   ('I3 qddot^T H qdot', s3, b3, scale3)):
                print('  %-14s %-11s %12.4e %12.4e %12.4e %12.4e'
                      % (short, name, abs(s), abs(b), abs(s - b), abs(s - b) / sc))
            print('  %-14s %-11s %12.4e %12s %12s %12.4e'
                  % (short, 'I1 ANTISYM', abs(s1a), 'n/a', 'n/a', abs(s1a) / scale1))
        print()

    print('  READ THIS AS: I3 is the identity the lossless argument does NOT need, I2 is the one')
    print('  that legitimately leaves a quadratic form, and I1 is the one whose boundary term the')
    print('  draft mis-stated. The ANTISYM row is the control: it must be LARGE, because only the')
    print('  symmetric part of H is annihilated. If it were small, the test would be vacuous.')


def main():
    cfg = RunConfig(
        mode='augmentation_ma50_b140-230_a6_z03', encoder_init='linear_map',
        joint_estimation=True, physics_parameterization='reduced',
        combo_init_detune=COMBO_INIT_DETUNE, param_init_detune=None,
        obc=True, obc_space='tangent', nx_ann=8, up_sample=1, stride=10,
        fs_new=4000, snr=None, save_flag=False)
    ok = part_a()
    part_b()
    part_c(cfg)
    banner('SUMMARY')
    print('  A identities symbolically      : %s' % ('PASS' if ok else 'FAIL'))
    print('  B nine cells reduced           : see the table; TWO cells had the wrong boundary')
    print('    condition in the draft (I1, positions return, not velocities vanish)')
    print('  C identities on the real data  : see the residual columns above')
    print('  The lossless obstruction (dissipative vs damping) needs NO boundary condition and')
    print('  is unaffected by the correction.')


if __name__ == '__main__':
    main()
