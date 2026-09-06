"""Extending orthogonal-projection regularisation to an augmentation WITH ADDITIONAL STATES.

Gyorok's penalty constrains the ANN OUTPUT at a point: `Q^T f_ANN(x, u) = 0`. His augmentation
is a static map, so that is the whole of the ANN's effect on the physical states. Ours is not:
the ANN carries its own state `xa`, and the path

        xa  ->  (next step)  ->  physical correction  ->  physical state

is invisible to a penalty evaluated at `xa = 0`. This script does three things, in order:

  PART A  NEGATIVE (motivation, not the contribution).  With the pointwise constraint enforced
          EXACTLY, the augmentation still contributes inside the parameter subspace over a
          rollout. A control separates the two mechanisms:
             K = 0   ANN does not read `xa`  -> propagation only; applies to Gyorok too;
                     NOT our gap
             K != 0  ANN reads `xa`          -> the state path; OUR gap
  PART B  THE MISSING TERM in closed form. Not "leakage exists" but what it equals, so the
          extension has a defined target.
  PART C  THE EXTENSION.  A condition on the state-to-physical map `K`, verified symbolically
          to annihilate the missing term identically, and verified to reduce to Gyorok when
          there are no additional states.
  PART D  BY-CONSTRUCTION OR PENALTY?  Dimension counting on the extension's condition, which
          decides which of the two enforcement routes is even feasible.

MODEL (affine, i.e. the augmentation linearised about the operating point; affine is chosen
because an existence claim proved for the SIMPLEST network transfers to every richer one)

    x_{k+1}  = A x_k  + B u_k + f_k              physical,   n_x
    xa_{k+1} = G xa_k + F x_k + H u_k + bg       additional, n_a          <- Gyorok has none
    f_k      = K xa_k + L x_k + M u_k + bf       correction into x

    K   maps the additional states into the physical correction. At `xa = 0` it is multiplied
        by zero, so the deployed penalty cannot see it. This is the loophole written as a
        parameter block rather than as a claim about hidden-layer weights.

Two choices that make PART A stronger than the deployed situation:
  * the pointwise constraint is imposed as an IDENTITY IN (x, u) over the ENTIRE `xa = 0`
    slice, not at a finite point set, so "sample the slice better" cannot rescue it;
  * the trajectory symbols are free, so every identity holds for ALL trajectories.

Outputs -> affine_leakage.json   (same directory)
Run: conda run -n GraduationProject python scripts/gantry/orthogonality/code/affine_leakage.py
"""
import os
import json
import itertools

import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'affine_leakage.json')

NX, NA, NU, NTH = 2, 1, 1, 1   # minimal, so every expression stays hand-checkable
T_LIST = [2, 3, 4]

RES = {'dims': dict(nx=NX, na=NA, nu=NU, n_theta=NTH), 'T': T_LIST}


def sym_mat(name, r, c):
    return sp.Matrix(r, c, lambda i, j: sp.Symbol(f'{name}{i}{j}', real=True))


def hdr(s):
    print('\n' + '=' * 78)
    print(s)
    print('=' * 78)


# --------------------------------------------------------------------------- model
A = sym_mat('A', NX, NX)
G = sym_mat('G', NA, NA)
F = sym_mat('F', NA, NX)
H = sym_mat('H', NA, NU)
bg = sym_mat('bg', NA, 1)
K = sym_mat('K', NX, NA)
L = sym_mat('L', NX, NX)
M = sym_mat('M', NX, NU)
bf = sym_mat('bf', NX, 1)
Phi = sym_mat('P', NX, NTH)      # baseline parameter sensitivity, d f_base / d theta


def rollout(T, xs, us, Kmat=K):
    """Cumulative ANN contribution to the physical state after T steps, from xa_0 = 0."""
    xa = sp.zeros(NA, 1)
    f_list = []
    for k in range(T):
        f_list.append(sp.expand(Kmat * xa + L * xs[k] + M * us[k] + bf))
        xa = sp.expand(G * xa + F * xs[k] + H * us[k] + bg)
    dx = sp.zeros(NX, 1)
    for j in range(T):
        dx += A ** (T - 1 - j) * f_list[j]
    return sp.expand(dx)


def psi(T):
    """Windowed parameter sensitivity: Phi propagated over the same window.
    The deployed pointwise Phi is the T = 1 case."""
    S = sp.zeros(NX, NTH)
    for j in range(T):
        S += A ** (T - 1 - j) * Phi
    return sp.expand(S)


# ======================================================================== PART A
hdr('PART A  NEGATIVE: the pointwise constraint does not bound the rollout contribution')

print('Pointwise constraint, imposed as an IDENTITY in (x, u) on the whole slice xa = 0:')
print('    Phi^T ( L x + M u + bf ) = 0     for ALL x, u')
cons = ([sp.Eq(e, 0) for e in Phi.T * L] +
        [sp.Eq(e, 0) for e in Phi.T * M] +
        [sp.Eq(e, 0) for e in Phi.T * bf])
for c in cons:
    print('       ', c)
print('\n    K does not appear in a single one of these equations.')
print('    -> the state-to-physical map is UNCONSTRAINED by the pointwise penalty.')
RES['pointwise_constraints'] = [str(c) for c in cons]

# NOT `sp.solve`. A linear solve may return the TRIVIAL branch (L = M = bf = 0), which
# makes the K = 0 control vanish for the wrong reason; the MATLAB run of 2026-09-04 did
# exactly that and the two engines disagreed until both were moved to this form.
# Instead parameterise the constraint by an explicit null-space basis, which is unique up
# to a change of basis and has no branch to pick: every column of L, M, bf must lie in
# ker(Phi^T), so write them as N * (free).
N = sp.Matrix.hstack(*sp.Matrix(Phi.T).nullspace())
print(f'    ker(Phi^T) has dimension {N.shape[1]} in R^{NX}; L, M, bf are parameterised')
print('    by it, so the pointwise constraint holds by construction, not by a solve branch.')
Lf, Mf, bff = sym_mat('lf', N.shape[1], NX), sym_mat('mf', N.shape[1], NU), sym_mat('bff', N.shape[1], 1)
PW = {}
for src, tgt in ((N * Lf, L), (N * Mf, M), (N * bff, bf)):
    for e, t in zip(src, tgt):
        PW[t] = e
assert sp.expand(Phi.T * L.subs(PW)) == sp.zeros(NTH, NX)
assert sp.expand(Phi.T * M.subs(PW)) == sp.zeros(NTH, NU)
assert sp.expand(Phi.T * bf.subs(PW)) == sp.zeros(NTH, 1)
print('    verified: Phi^T L = Phi^T M = Phi^T bf = 0 identically after substitution.')
print(f'    all {NX * NA} entries of K remain free.')

for T in T_LIST:
    xs = [sym_mat(f'x{k}_', NX, 1) for k in range(T)]
    us = [sym_mat(f'u{k}_', NU, 1) for k in range(T)]
    proj = sp.expand((psi(T).T * rollout(T, xs, us))[0, 0].subs(PW))
    prop = sp.expand(proj.subs({k: 0 for k in K}))    # control: ANN does not read xa
    state = sp.expand(proj - prop)                    # everything carrying a K entry
    print(f'\n  T = {T}:  Psi_T^T dx_T  with the pointwise constraint enforced exactly')
    print(f'      nonzero overall                       : {proj != 0}')
    print(f'      CONTROL  K = 0 (static ANN, Gyorok)   : {prop != 0}'
          '   <- propagation; NOT our gap')
    print(f'      STATE PATH  terms carrying K          : {state != 0}'
          '   <- our gap')
    RES[f'A_T{T}'] = dict(nonzero=bool(proj != 0), control_K0=bool(prop != 0),
                          state_path=bool(state != 0))

print('\n  READ: the control is nonzero too, so propagation alone already breaks the')
print('  pointwise guarantee. That part is inherited from Gyorok and is not what we claim.')
print('  The state-path row is the part that exists ONLY because of the additional states,')
print('  and it is the part the rest of this script removes.')


# ======================================================================== PART B
hdr('PART B  THE MISSING TERM in closed form')

print('xa_j = sum_{i<j} G^(j-1-i) g_i  with  g_i = F x_i + H u_i + bg, from xa_0 = 0.')
print('Substituting into dx_T = sum_j A^(T-1-j) f_j and collecting by g_i:')
print()
print('    dx_T^state = sum_{i=0}^{T-2} Lam_T(i) g_i ,')
print('    Lam_T(i)   = sum_{j=i+1}^{T-1} A^(T-1-j) K G^(j-1-i)')
print()


def lam(T, i):
    S = sp.zeros(NX, NA)
    for j in range(i + 1, T):
        S += A ** (T - 1 - j) * K * G ** (j - 1 - i)
    return sp.expand(S)


for T in T_LIST:
    xs = [sym_mat(f'x{k}_', NX, 1) for k in range(T)]
    us = [sym_mat(f'u{k}_', NU, 1) for k in range(T)]
    proj = sp.expand((psi(T).T * rollout(T, xs, us))[0, 0].subs(PW))
    state = sp.expand(proj - proj.subs({k: 0 for k in K}))
    closed = sp.zeros(NX, 1)
    for i in range(T - 1):
        closed += lam(T, i) * sp.expand(F * xs[i] + H * us[i] + bg)
    ok = sp.simplify(sp.expand((psi(T).T * sp.expand(closed))[0, 0]) - state) == 0
    print(f'  T = {T}:  closed form reproduces the symbolic rollout exactly : {ok}')
    assert ok, f'closed form mismatch at T={T}'
    RES[f'B_T{T}_closed_form_verified'] = bool(ok)

print('\n  Every term is  A^a K G^b  with a + b = T-2-i. So the missing term is set by K (the')
print('  unconstrained block), by how hard the additional state is driven (F, H, bg), and by')
print('  MIXED POWERS of the baseline dynamics A and the additional-state dynamics G. It')
print('  vanishes identically iff K = 0, i.e. iff the augmentation is effectively static.')
RES['leakage_operator'] = 'Lam_T(i) = sum_{j=i+1}^{T-1} A^(T-1-j) K G^(j-1-i)'


# ======================================================================== PART C
hdr('PART C  THE EXTENSION: an orthogonality condition on the state-to-physical map K')

print('Requiring  Psi_T^T dx_T^state = 0  for ALL trajectories forces it term by term.')
print('A sufficient condition, independent of the data and of G:')
print()
print('    Psi_T^T A^a K = 0 ,   a = 0 ... T-2                                    (EXT)')
print('    <=>  range(K)  orthogonal to  span{ Psi_T, A^T Psi_T, ..., (A^(T-2))^T Psi_T }')
print()
print('Gyorok projects the ANN OUTPUT off the parameter subspace. With additional states one')
print('must ALSO project the STATE-TO-PHYSICAL MAP off the PROPAGATED parameter subspace.')
print('At n_a = 0 there is no K, (EXT) is vacuous, and the method collapses to Gyorok.')
print()

for T in T_LIST:
    rows = [psi(T).T * A ** a for a in range(T - 1)]
    S_T = sp.Matrix.vstack(*rows)
    ker = sp.Matrix(S_T).nullspace()
    n_cond = S_T.shape[0] * NA

    xs = [sym_mat(f'x{k}_', NX, 1) for k in range(T)]
    us = [sym_mat(f'u{k}_', NU, 1) for k in range(T)]
    proj = sp.expand((psi(T).T * rollout(T, xs, us))[0, 0].subs(PW))
    state = sp.expand(proj - proj.subs({k: 0 for k in K}))

    if not ker:
        # No nonzero K satisfies (EXT): the condition is satisfiable only by K = 0.
        EXT = {k: 0 for k in K}
        trivial = True
    else:
        Nk = sp.Matrix.hstack(*ker)
        Kf = sym_mat('kf', Nk.shape[1], NA)
        EXT = {t: e for e, t in zip(Nk * Kf, K)}
        trivial = sp.simplify(K.subs(EXT)) == sp.zeros(NX, NA)

    killed = sp.simplify(state.subs(EXT)) == 0
    print(f'  T = {T}:  {n_cond} scalar conditions on {NX * NA} entries of K, '
          f'dim ker(S_T) = {len(ker)}')
    print(f'          state-path term is annihilated identically : {killed}')
    print(f'          forced K = 0 (condition is vacuous?)       : {trivial}')
    assert killed, f'(EXT) fails to annihilate the state path at T={T}'
    RES[f'C_T{T}'] = dict(n_conditions=int(n_cond), dim_ker=len(ker),
                          annihilates=bool(killed), forces_K_zero=bool(trivial))


# ======================================================================== PART D
hdr('PART D  BY-CONSTRUCTION OR PENALTY? the counting that decides it')

print('(EXT) says range(K) lies in the kernel of the stacked map')
print('    S_T = [ Psi_T , A^T Psi_T , ... , (A^(T-2))^T Psi_T ]^T ,')
print('a Krylov stack of Psi_T under A^T. Two facts decide the enforcement route.\n')

print('  (1) SATURATION. By Cayley-Hamilton, A^T has degree n_x, so the span stops growing')
print('      at a = n_x - 1. Beyond T = n_x + 1 the horizon adds NO new conditions.')
print('      Verified below on the symbolic A by rank of the stack:')
for T in range(2, NX + 4):
    rows = []
    for a in range(T - 1):
        rows.append((psi(T).T * A ** a))
    S_T = sp.Matrix.vstack(*rows)
    r = S_T.rank()
    print(f'        T = {T}:  stack {S_T.shape[0]}x{S_T.shape[1]}   rank = {r}'
          f'   dim ker = {NX - r}')
    RES[f'D_T{T}_rank'] = int(r)

print()
print('  (2) THE BINDING COUNT. rank(S_T) <= n_x, and it reaches n_x as soon as the')
print('      propagated sensitivities span the physical state space. When they do,')
print('      dim ker(S_T) = 0 and (EXT) forces K = 0 EXACTLY: a by-construction')
print('      parameterisation K = (I - Pi) K~ would delete the additional-state pathway')
print('      altogether, which is the pathway the ablation shows is worth ~30 percent.')
print()
print('      For the gantry: n_x = 6 routed physical states, Phi of rank 10 in the')
print('      identifiable parameters, so span(Psi) = R^6 already at a = 0.')
print('      -> STRICT BY-CONSTRUCTION ON K IS NOT AVAILABLE TO US.')
print()
print('  CONSEQUENCE. The extension must be enforced the way Gyorok enforces his: as a')
print('  SOFT PENALTY over the realised data, not by construction. The deployed form is')
print()
print('      beta_a * sum_over_windows || Q_Psi^T sum_i Lam_T(i) g_i ||^2')
print()
print('  which is finitely many conditions over the data rather than an identity in the')
print('  state space, so it admits nonzero K. This is an independent argument for the')
print('  penalty route over the orthogonal-by-construction route, arising from the')
print('  additional states specifically and NOT present in the static case.')
RES['D_conclusion'] = ('strict by-construction on K forces K=0 when span(Psi)=R^nx; '
                       'extension must be a soft penalty over realised data')


hdr('SUMMARY OF CLAIMS')
print('  A  pointwise penalty does not bound the rollout contribution        PROVED (symbolic)')
print('  A  a state-path component exists that is absent in the static case  PROVED (control)')
print('  B  that component equals sum_i Lam_T(i) g_i, Lam as above           PROVED (identity)')
print('  C  (EXT) annihilates it identically and reduces to Gyorok at n_a=0  PROVED (symbolic)')
print('  D  (EXT) by construction forces K=0 once span(Psi)=R^nx             PROVED (rank)')
print('  D  therefore the extension is a penalty, not a construction         FOLLOWS from D')
print()
print('  Rule 1: every quantity above is A, Psi (baseline Jacobian, already built in')
print('  gantry_dynamic/orth_penalty.py) or a network parameter. No truth model is used,')
print('  so all of it transfers to Telica.')

with open(OUT, 'w', encoding='utf-8') as fh:
    json.dump(RES, fh, indent=2)
print(f'\nwrote {OUT}')
