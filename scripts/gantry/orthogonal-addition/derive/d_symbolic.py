"""Symbolic machine checks for DERIVATION.md (sympy). Every check prints `[Dn] ... OK` or FAILs.

D1  the baseline M, C, K in the ten identifiable combinations equal gantrySystem.m's matrices
D2  the ten sensitivity regressors r_j = dM_j qdd + dC_j qd + dK_j q, and the 5 / 4 / 1 split
D3  d(qdd)/d(theta_j) = -M^-1 r_j  (implicit differentiation of M qdd + C qd + K q = u)
D4  the one-step inner product reduces to the continuous condition  -Ts^2 SUM r_j^T W phi,
    W = M^-T S_v^-2 M^-1, at leading order in Ts (Euler/RK4 velocity-row sensitivity)
D5  skew of the Y guide by gamma: M13 = mh gamma, M22 += -2 mh d gamma Y (first order), from the
    payload kinematics
D6  lossless absorber, mass-conserving: the Lagrangian gives the implemented gantrySystemOA EOM
D7  X spring at the payload: potential k (b^T q)^2 / 2 gives K = k b b^T, b = [1, -Y, 0]
D8  legitimacy (OA-006): every slot the physical terms use is identically zero in the baseline
    for all parameter values; the baseline's own slots are exactly BASELINE_SLOTS of columns.py
D9  the ten conditions for a LINEAR stateless addition, written in the addition's coefficients,
    are linear in them (so the admissible class is a null space)
D10 route A parity: an even addition against the odd (state-linear) regressor integrates to
    zero over any sign-symmetric domain (Gyorok Eq. 16), for a generic quadratic addition
"""
__project_origin__ = "added"

import itertools
import os
import sys

import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'design'))
import columns as colib                                # noqa: E402

FAIL = []


def ok(tag, cond, msg=''):
    print('[%s] %s %s' % (tag, msg, 'OK' if cond else 'FAIL'), flush=True)
    if not cond:
        FAIL.append(tag)


Y = sp.Symbol('Y', real=True)
m1, m2, mb, mh, Lb, Jb, Jh, d = sp.symbols('m1 m2 m_b m_h L_b J_b J_h d', positive=True)
cg1, cg2, cb1, cb2, cy, kb1, kb2 = sp.symbols('c_g1 c_g2 c_b1 c_b2 c_y k_b1 k_b2', positive=True)
kb_sum, cb_sum, m_tot, m_dif, J_eff = sp.symbols('k_bsum c_bsum m_total m_diff J_eff', real=True)
theta = [kb_sum, cg1, cg2, cy, cb_sum, mh, m_tot, m_dif, J_eff, d]
NAMES = ['kb_sum', 'cg1', 'cg2', 'cy', 'cb_sum', 'mh', 'm_total', 'm_diff', 'J_eff', 'd']

# ==== gantrySystem.m, verbatim
M_gs = sp.Matrix([[m1 + m2 + mb + mh, (m1 - m2) * Lb / 2 - mh * Y, 0],
                  [(m1 - m2) * Lb / 2 - mh * Y, Jb + Jh + (m1 + m2) * Lb**2 / 4 + mh * d**2 + mh * Y**2, -mh * d],
                  [0, -mh * d, mh]])
C_gs = sp.Matrix([[cg1 + cg2, (cg1 - cg2) * Lb / 2, 0],
                  [(cg1 - cg2) * Lb / 2, cb1 + cb2 + (cg1 + cg2) * Lb**2 / 4, 0],
                  [0, 0, cy]])
K_gs = sp.Matrix([[0, 0, 0], [0, kb1 + kb2, 0], [0, 0, 0]])

# ==== the same in the ten combinations
M = sp.Matrix([[m_tot + mh, m_dif * Lb / 2 - mh * Y, 0],
               [m_dif * Lb / 2 - mh * Y, J_eff + mh * d**2 + mh * Y**2, -mh * d],
               [0, -mh * d, mh]])
C = sp.Matrix([[cg1 + cg2, (cg1 - cg2) * Lb / 2, 0],
               [(cg1 - cg2) * Lb / 2, cb_sum + (cg1 + cg2) * Lb**2 / 4, 0],
               [0, 0, cy]])
K = sp.Matrix([[0, 0, 0], [0, kb_sum, 0], [0, 0, 0]])
sub = {m_tot: m1 + m2 + mb, m_dif: m1 - m2, J_eff: Jb + Jh + (m1 + m2) * Lb**2 / 4,
       cb_sum: cb1 + cb2, kb_sum: kb1 + kb2}
ok('D1', sp.simplify(M.subs(sub) - M_gs) == sp.zeros(3) and sp.simplify(C.subs(sub) - C_gs)
   == sp.zeros(3) and sp.simplify(K.subs(sub) - K_gs) == sp.zeros(3),
   'M(theta), C(theta), K(theta) == gantrySystem.m after substitution:')

# ==== D2 regressors
q = sp.Matrix(sp.symbols('X Theta Yq', real=True))       # Yq: the Y coordinate as a state
qd = sp.Matrix(sp.symbols('Xd Thetad Yd', real=True))
qdd = sp.Matrix(sp.symbols('Xdd Thetadd Ydd', real=True))
fam = {}
r = {}
for n, th in zip(NAMES, theta):
    dM, dC, dK = M.diff(th), C.diff(th), K.diff(th)
    used = [nm for nm, X in (('M', dM), ('C', dC), ('K', dK)) if X != sp.zeros(3)]
    fam[n] = used
    r[n] = sp.simplify(dM * qdd + dC * qd + dK * q)
split = ([n for n in NAMES if fam[n] == ['M']], [n for n in NAMES if fam[n] == ['C']],
         [n for n in NAMES if fam[n] == ['K']])
print('     mass family   : %s' % split[0])
print('     damping family: %s' % split[1])
print('     stiffness     : %s' % split[2])
for n in NAMES:
    print('     r_%-8s = %s' % (n, list(r[n])))
ok('D2', [len(s) for s in split] == [5, 4, 1] and all(len(fam[n]) == 1 for n in NAMES),
   'each combination enters exactly one of M, C, K; split 5 / 4 / 1:')

# ==== D3 implicit differentiation
u = sp.Matrix(sp.symbols('u1 u2 u3', real=True))
Minv = M.inv()
qdd_expr = Minv * (u - C * qd - K * q)
good = True
for n, th in zip(NAMES, theta):
    lhs = qdd_expr.diff(th)
    rhs = -Minv * (M.diff(th) * qdd_expr + C.diff(th) * qd + K.diff(th) * q)
    good &= sp.simplify(lhs - rhs) == sp.zeros(3, 1)
ok('D3', good, 'd qdd / d theta_j = -M^-1 r_j with qdd = M^-1(u - C qd - K q), all ten:')

# ==== D4 one-step inner product, leading order (explicit Euler velocity row; RK4 agrees at O(Ts),
#      measured in baseline-and-added-dynamics sect. 8.1 T2/T3)
Ts = sp.Symbol('T_s', positive=True)
sv = sp.Matrix(sp.symbols('s4 s5 s6', positive=True))    # std of the three velocity rows
Sinv = sp.diag(*[1 / s for s in sv])
phi = sp.Matrix(sp.symbols('phi1 phi2 phi3', real=True))  # added generalised force
th = m_dif
qd_next = qd + Ts * qdd_expr                              # velocity row of one step
J_row = Sinv * qd_next.diff(th)                           # normalised sensitivity column (vel rows)
qd_next_a = qd + Ts * Minv * (u - C * qd - K * q + phi)
D_row = Sinv * (qd_next_a - qd_next)                      # normalised one-step effect of phi
W = Minv.T * Sinv * Sinv * Minv
lhs = (J_row.T * D_row)[0]
rhs = (-Ts**2 * (M.diff(th) * qdd_expr).T * W * phi)[0]
ok('D4', sp.simplify(lhs - rhs) == 0,
   '<J_j, Delta_phi> = -Ts^2 r_j^T W phi on the velocity rows (shown for m_diff; linear in r_j):')

# ==== D5 skew kinematics
t = sp.Symbol('t')
Xf, Thf, Yf = [sp.Function(n)(t) for n in ('X', 'Th', 'Yp')]
gam = sp.Symbol('gamma', real=True)
xl, yl = -d + gam * Yf, Yf                                  # payload on a guide skewed by gamma
vx = Xf.diff(t) + xl.diff(t) - Thf.diff(t) * yl             # linearised in the small yaw Theta
vy = yl.diff(t) + Thf.diff(t) * xl
T = sp.expand(mh * (vx**2 + vy**2) / 2)
qs = [Xf, Thf, Yf]
Mpay = sp.Matrix(3, 3, lambda i, j: sp.diff(T, qs[i].diff(t), qs[j].diff(t)))
Mpay0 = Mpay.subs(gam, 0)
dMg = sp.simplify(Mpay.diff(gam).subs(gam, 0))
exp_dM = sp.Matrix([[0, 0, mh], [0, -2 * mh * d * Yf, 0], [mh, 0, 0]])
ok('D5', sp.simplify(dMg - exp_dM) == sp.zeros(3)
   and sp.simplify(Mpay0 - sp.Matrix([[mh, -mh * Yf, 0], [-mh * Yf, mh * (Yf**2 + d**2), -mh * d],
                                       [0, -mh * d, mh]])) == sp.zeros(3),
   'payload inertia at gamma = 0 is the baseline mh block; d M / d gamma = [[0,0,mh],[0,-2 mh d Y,0],[mh,0,0]]:')

# ==== D6 absorber Lagrangian, mass-conserving
ma, wa = sp.symbols('m_a omega_a', positive=True)
b = sp.Matrix(sp.symbols('b1 b2 b3', real=True))
qv = sp.Matrix([sp.Function('q%d' % i)(t) for i in range(3)])
da = sp.Function('delta')(t)
Mr = M - ma * b * b.T
Tl = (qv.diff(t).T * Mr * qv.diff(t))[0] / 2 + ma * ((b.T * qv.diff(t))[0] + da.diff(t))**2 / 2
Vl = (qv.T * K * qv)[0] / 2 + ma * wa**2 * da**2 / 2
L = Tl - Vl
eqs = []
for i in range(3):
    eqs.append(sp.diff(L.diff(qv[i].diff(t)), t) - L.diff(qv[i]) + (C * qv.diff(t))[i] - u[i])
eqs.append(sp.diff(L.diff(da.diff(t)), t) - L.diff(da))
# The Euler-Lagrange equations are derived symbolically above. The implemented EOM
#   Mr qdd = u - C qd - K q + ma wa^2 delta b ,   delta'' = -b^T qdd - wa^2 delta
# is substituted into them and the residual evaluated EXACTLY (rational arithmetic) at random
# rational points: a nonzero rational function vanishes at a random point with probability zero
# (Schwartz-Zippel), so 20 exact zeros are a proof up to that measure-zero event. A symbolic
# `solve` of the 4 x 4 system took over 10 minutes (R-011) and is replaced by this.
acc = list(qv.diff(t, 2)) + [da.diff(t, 2)]
A_s = sp.Symbol('Aq0'), sp.Symbol('Aq1'), sp.Symbol('Aq2'), sp.Symbol('Ad')
V_s = sp.symbols('Vq0 Vq1 Vq2 Vd', real=True)
P_s = sp.symbols('Pq0 Pq1 Pq2 Pd', real=True)
rep2 = {qv[i].diff(t, 2): A_s[i] for i in range(3)}
rep2[da.diff(t, 2)] = A_s[3]
rep1 = {qv[i].diff(t): V_s[i] for i in range(3)}
rep1[da.diff(t)] = V_s[3]
eqs_s = [sp.expand(e).subs(rep2).subs(rep1) for e in eqs]      # second derivatives first
eqs_s = [e.subs({qv[i]: P_s[i] for i in range(3)}).subs(da, P_s[3]) for e in eqs_s]
import random
random.seed(6)
free = sorted(set().union(*[e.free_symbols for e in eqs_s]) - set(A_s), key=str)
g6 = True
for trial in range(20):
    val = {s_: sp.Rational(random.randint(1, 97), random.randint(1, 13)) for s_ in free}
    Mr_v = Mr.subs(val)
    rhs = (u - C * sp.Matrix(V_s[:3]) - K * sp.Matrix(P_s[:3])
           + ma * wa**2 * P_s[3] * b).subs(val)
    qdd_v = Mr_v.LUsolve(rhs)
    dd_v = -(b.subs(val).T * qdd_v)[0] - (wa**2 * P_s[3]).subs(val)
    sol_v = {A_s[i]: qdd_v[i] for i in range(3)}
    sol_v[A_s[3]] = dd_v
    res = [sp.nsimplify(e.subs(val).subs(sol_v)) for e in eqs_s]
    g6 &= all(r_ == 0 for r_ in res)
    if trial == 0:
        # negative control: the WRONG carrier mass (M instead of M - m b b^T) must be detected
        qdd_w = M.subs(val).LUsolve(rhs)
        dd_w = -(b.subs(val).T * qdd_w)[0] - (wa**2 * P_s[3]).subs(val)
        sol_w = {A_s[i]: qdd_w[i] for i in range(3)}
        sol_w[A_s[3]] = dd_w
        res_w = [sp.nsimplify(e.subs(val).subs(sol_w)) for e in eqs_s]
        g6 &= any(r_ != 0 for r_ in res_w)
        print('     negative control (carrier mass M instead of M - m b b^T): residual %s'
              % ['%.3e' % float(r_) for r_ in res_w])
ok('D6', g6,'Euler-Lagrange of carrier (M - m b b^T) + absorber (b^T qd + delta_dot), with the '
   'gantrySystemOA EOM substituted: residual exactly 0 at 20 random rational points:')

# ==== D7 payload X spring
k = sp.Symbol('k', positive=True)
bx = sp.Matrix([1, -Y, 0])
Vs = k * ((bx.T * q)[0])**2 / 2
Ks = sp.hessian(Vs, list(q))
ok('D7', sp.simplify(Ks - k * bx * bx.T) == sp.zeros(3),
   'Hessian of k (X - Y Theta)^2 / 2 is k b b^T, slots K11, K12 Y, K22 Y^2:')

# ==== D8 legitimacy
def poly_slots(Mat, name):
    out = set()
    for i in range(3):
        for j in range(i, 3):
            e = sp.expand(Mat[i, j])
            for p in range(4):
                if sp.simplify(e.coeff(Y, p)) != 0:
                    out.add((name, i, j, p))
    return out
base_slots = poly_slots(M, 'M') | poly_slots(C, 'C') | poly_slots(K, 'K')
used = {('K', 0, 0, 0), ('K', 2, 2, 0), ('K', 0, 1, 1), ('K', 1, 1, 2), ('M', 0, 2, 0), ('M', 1, 1, 1)}
ok('D8', base_slots == colib.BASELINE_SLOTS and not (used & base_slots),
   'baseline slots == columns.BASELINE_SLOTS (%d) and the physical terms use none of them:'
   % len(base_slots))

# ==== D9 linearity of the ten conditions in a linear stateless addition's coefficients
a = sp.symbols('a0:6', real=True)
Ma = sp.Matrix([[0, 0, a[0]], [0, a[1] * Y, 0], [a[0], 0, 0]])
Ka = sp.Matrix([[a[2], a[3] * Y, 0], [a[3] * Y, a[4] * Y**2, 0], [0, 0, a[5]]])
phi_a = -(Ma * qdd + Ka * q)
Wg = sp.Matrix(3, 3, lambda i, j: sp.Symbol('W%d%d' % (min(i, j), max(i, j)), real=True))
lin = True
for n in NAMES:
    c = sp.expand((r[n].T * Wg * phi_a)[0])
    lin &= all(sp.degree(c, ai) <= 1 for ai in a) and c.subs({ai: 0 for ai in a}) == 0
ok('D9', lin, 'each condition r_j^T W phi_a is linear and homogeneous in the addition coefficients:')

# ==== D10 route A parity
z = sp.symbols('z0:4', real=True)                    # deviation state (and input) coordinates
cq = sp.symbols('c0:10', real=True)
even = sum(ci * zi * zj for ci, (zi, zj) in zip(cq, itertools.combinations_with_replacement(z, 2)))
odd = sum(sp.Symbol('l%d' % i) * zi for i, zi in enumerate(z))
integ = odd * even
for zi in z:
    integ = sp.integrate(integ, (zi, -1, 1))
ok('D10', sp.simplify(integ) == 0,
   'integral over [-1,1]^4 of (linear regressor) x (generic quadratic addition) == 0:')

# ==== D11 absorber reaction as an apparent mass (OA-019): with d'' + wa^2 d = -b^T qdd and
#      f_a = -m b d'', at frequency w (d/dt -> j w): f_a = -m_app(w) b b^T qdd,
#      m_app = m w^2 / (wa^2 - w^2): positive below resonance, negative above.
w_, wa_ = sp.symbols('omega omega_a', positive=True)
Q_ = sp.Symbol('Qdd')                                    # complex amplitude of b^T qdd
Dd = sp.Symbol('Dd')
sol_d = sp.solve(sp.Eq(-w_**2 * Dd + wa_**2 * Dd, -Q_), Dd)[0]
fa_ = -ma * (-w_**2 * sol_d)                             # -m d'' with d'' = -w^2 d, along b
m_app = ma * w_**2 / (wa_**2 - w_**2)
ok('D11', sp.simplify(fa_ + m_app * Q_) == 0 and
   sp.limit(m_app, w_, 0) == 0 and sp.limit(m_app, w_, sp.oo) == -ma,
   'absorber reaction f_a = -m_app(w) b b^T qdd, m_app = m w^2/(wa^2 - w^2), 0 at DC, -m at high w:')

print('\nD_SYMBOLIC %s  (%d checks failed)' % ('ALL OK' if not FAIL else 'FAIL ' + ','.join(FAIL),
                                             len(FAIL)))
sys.exit(1 if FAIL else 0)
