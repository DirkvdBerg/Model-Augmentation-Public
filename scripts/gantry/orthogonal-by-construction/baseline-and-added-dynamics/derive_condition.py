"""When does orthogonal-by-construction recover the TRUE parameters? A sympy derivation.

Maarten Schoukens, 2026-09-18: orthogonal projection makes the split between baseline and
augmentation UNIQUE, but it recovers the TRUE physical parameters only if the true missing
dynamics are themselves orthogonal to the baseline's parameter sensitivities. This script
establishes that claim symbolically, in the order the numerical companion then follows.

Notation, fixed here and used by `measure_delta_orthogonality.py` and by
`baseline-and-added-dynamics.md`:

    f_base(theta)   the discrete one-step baseline map, stacked over the reference set,
                    as a vector in R^n  (n = N_samples * n_protected_rows)
    theta*          the true physical parameter vector, R^p
    Delta*          the TRUE missing dynamics on the reference set, R^n,
                    defined by  f_truth = f_base(theta*) + Delta*
    J               d f_base / d theta evaluated at the expansion point, R^{n x p}
    P = J J^+       the orthogonal projector onto range(J)
    f_ann           the learning component evaluated on the same reference set, R^n
    dtheta          theta_hat - theta*, the parameter bias

The OBC constraint is  J^T f_ann = 0,  i.e. f_ann in range(J)^perp.

Five results are produced:

  R1  affine baseline, exact interpolation: dtheta = J^+ Delta*, and dtheta = 0 iff J^T Delta* = 0.
  R2  affine baseline, LEAST SQUARES with a CAPACITY-LIMITED f_ann: the same dtheta = J^+ Delta*,
      exactly, independent of what the network can represent. This is stronger than R1 and it is
      what makes the predicted bias a prediction rather than a bound.
  R3  baseline nonlinear in theta: the identity holds to first order, with an explicit
      second-order remainder -1/2 J^+ H[dtheta, dtheta].
  R4  basis built at a WRONG expansion point theta_0: the projection becomes oblique and
      dtheta = (J_0^T J)^{-1} J_0^T Delta*. This is why the numerical companion reports the
      answer at both the true and the detuned parameter point.
  R5  rank deficiency: with rank(J) = r < p the bias is only determined modulo ker(J), and the
      minimum-norm representative is J^+ Delta*.

Run:  conda run -n GraduationProject python -u derive_condition.py
"""
__project_origin__ = "added"

import sympy as sp


LINE = '=' * 78


def banner(title):
    print('\n' + LINE)
    print(title)
    print(LINE)


def symbolic_matrix(name, rows, cols):
    return sp.Matrix(rows, cols, lambda i, j: sp.Symbol(f'{name}_{i}{j}', real=True))


def symbolic_vector(name, rows):
    return sp.Matrix(rows, 1, lambda i, _: sp.Symbol(f'{name}_{i}', real=True))


# ----------------------------------------------------------------------------------- R1
def result_1_affine_exact():
    """Affine baseline, exact interpolation. The claim in its cleanest form."""
    banner('R1  AFFINE BASELINE, EXACT INTERPOLATION')

    n, p = 4, 2                      # small enough to print, general enough to be generic
    J = symbolic_matrix('J', n, p)
    D = symbolic_vector('D', n)      # Delta*

    print('Truth       : f_truth = f_base(theta*) + Delta*')
    print('Model       : f_base(theta*) + J dtheta + f_ann          (f_base affine in theta)')
    print('Interpolate : J dtheta + f_ann = Delta*                  ... (1)')
    print('OBC         : J^T f_ann = 0                              ... (2)')
    print()
    print('Left-multiply (1) by J^T and use (2):  (J^T J) dtheta = J^T Delta*.')
    print('With J of full column rank, J^T J is invertible, hence')
    print()
    print('    dtheta = (J^T J)^-1 J^T Delta* = J^+ Delta*.         ... (3)')
    print()

    # EXISTENCE: substitute the claimed pair into (1) and (2) and check both residuals vanish
    # identically in the symbols. Dense symbolic entries, so nothing is special-cased away.
    pinv = (J.T * J).inv() * J.T
    P = J * pinv
    dth_claim = pinv * D
    fa_claim = (sp.eye(n) - P) * D

    res1 = sp.simplify(J * dth_claim + fa_claim - D)
    res2 = sp.simplify(J.T * fa_claim)
    print('CHECK (existence)  residual of (1) at dtheta = J^+ Delta*, f_ann = (I - P) Delta*:')
    print('      ', res1.T)
    assert res1 == sp.zeros(n, 1), 'R1 failed: the claimed pair does not satisfy (1)'
    print('CHECK (existence)  residual of (2), i.e. J^T (I - P) Delta*:')
    print('      ', res2.T)
    assert res2 == sp.zeros(p, 1), 'R1 failed: the claimed f_ann is not orthogonal to range(J)'
    print('       -> both zero.')
    print()
    print('UNIQUENESS.  The map (dtheta, f_ann) -> (J dtheta + f_ann, J^T f_ann) is injective when')
    print('J has full column rank: if J dtheta + f_ann = 0 and J^T f_ann = 0, then multiplying the')
    print('first by J^T gives (J^T J) dtheta = 0, hence dtheta = 0, hence f_ann = 0. So the pair')
    print('above is THE solution, and the split is UNIQUE. That uniqueness is the property')
    print('projection was introduced for. Uniqueness is not correctness.')
    print()

    # The projector identities the rest of the derivation leans on, verified in the same symbols.
    for label, expr, shape in (
            ('P^T - P', P.T - P, (n, n)),
            ('P P - P', P * P - P, (n, n)),
            ('P J - J', P * J - J, (n, p)),
            ('J^T (I - P)', J.T * (sp.eye(n) - P), (p, n))):
        value = sp.simplify(expr)
        assert value == sp.zeros(*shape), f'projector identity {label} failed'
    print('CHECK  P^T = P, P^2 = P, P J = J, J^T (I - P) = 0: all verified symbolically.')
    print()

    print('THE CONDITION.  J has full column rank, so J^+ Delta* = 0 iff J^T Delta* = 0:')
    print('  (=>) J^+ Delta* = 0  =>  J J^+ Delta* = P Delta* = 0  =>  J^T Delta* = J^T P Delta* = 0')
    print('  (<=) J^T Delta* = 0  =>  (J^T J)^-1 J^T Delta* = 0.')
    print()
    print('    theta_hat = theta*   <==>   J^T Delta* = 0   <==>   Delta* PERPENDICULAR range(J).')
    print()
    print('Equivalently, with rho* = ||P Delta*|| / ||Delta*|| in [0, 1]:')
    print('    rho* = 0  <==>  no parameter bias.   rho* > 0  ==>  a bias of size')
    print('    ||dtheta|| = ||J^+ Delta*|| >= rho* ||Delta*|| / sigma_max(J).')
    print()

    # The bias bound, verified on the singular values of a random-but-symbolic instance is
    # overkill; state it and show the two-sided form that the numerics report.
    print('    rho* ||Delta*|| / sigma_max(J)  <=  ||dtheta||  <=  rho* ||Delta*|| / sigma_min(J)')
    print('    ... so rho* alone does NOT fix the bias magnitude; the conditioning of J does too.')
    print('    This is why the numerical companion reports J^+ Delta* itself, not only rho*.')


# ----------------------------------------------------------------------------------- R2
def result_2_least_squares_capacity():
    """Least squares with a capacity-limited f_ann. The separation argument."""
    banner('R2  LEAST SQUARES WITH A CAPACITY-LIMITED AUGMENTATION')

    print('R1 assumed the model can interpolate Delta* exactly. A three-layer, 24-node network')
    print('cannot. Let A be ANY subspace of reachable augmentation fields, and let OBC enforce')
    print('A subset of range(J)^perp. The training problem is')
    print()
    print('    min over dtheta in R^p, f_ann in A   || J dtheta + f_ann - Delta* ||^2.')
    print()
    print('Split Delta* = P Delta* + (I - P) Delta*. Then J dtheta lies in range(J) and both')
    print('f_ann and (I - P) Delta* lie in range(J)^perp, so the two parts are orthogonal and')
    print('Pythagoras applies term by term:')
    print()
    print('    || J dtheta - P Delta* ||^2  +  || f_ann - (I - P) Delta* ||^2.     ... (4)')
    print()

    # The split (4) holds iff the cross term 2 (J dtheta - P D)^T (f_ann - (I - P) D) vanishes.
    # It expands into four inner products; each is zero for a separate structural reason, and
    # each of those reasons is a matrix identity that can be checked in dense symbols.
    n, p = 4, 2
    J = symbolic_matrix('J', n, p)
    pinv = (J.T * J).inv() * J.T
    P = J * pinv

    checks = [
        ('(J dtheta)^T f_ann = dtheta^T (J^T f_ann) = 0',
         'by the OBC constraint itself', None),
        ('(J dtheta)^T (I - P) D = dtheta^T J^T (I - P) D = 0',
         'because J^T (I - P) = 0', sp.simplify(J.T * (sp.eye(n) - P))),
        ('(P D)^T f_ann = D^T P f_ann = D^T J (J^T J)^-1 (J^T f_ann) = 0',
         'again by the constraint, since P factors through J^T', None),
        ('(P D)^T (I - P) D = D^T P (I - P) D = 0',
         'because P (I - P) = 0', sp.simplify(P * (sp.eye(n) - P))),
    ]
    for statement, reason, verified in checks:
        mark = 'symbolically verified' if verified is not None else 'algebraic'
        if verified is not None:
            assert verified == sp.zeros(*verified.shape), f'R2 failed: {reason}'
        print(f'  {statement}')
        print(f'      {reason}  [{mark}]')
    print()
    print('       -> the cross term is identically zero, so (4) holds.')
    print()
    print('CONSEQUENCE.  The first term of (4) does not contain f_ann and the second does not')
    print('contain dtheta, so the minimisation SEPARATES. The parameter minimiser is')
    print()
    print('    dtheta = argmin || J dtheta - P Delta* ||^2 = J^+ P Delta* = J^+ Delta*,')
    print()
    print('which is EXACTLY R1, whatever A is. Limited network capacity degrades the FIT, by')
    print('|| f_ann - (I - P) Delta* ||, and leaves the parameter bias untouched.')
    print()
    print('So J^+ Delta* is a PREDICTION of the parameter error under OBC, not a lower bound.')
    print('Caveats that break the equality in the real training problem, in order of size:')
    print('  (a) the objective is a multi-step CLOSED-LOOP rollout, not the one-step least')
    print('      squares on the reference set that defines J; the two weight samples and rows')
    print('      differently, so the effective inner product is not the one OBC orthogonalises in;')
    print('  (b) the constraint holds at the CURRENT expansion point, which moves every epoch;')
    print('  (c) the baseline is not affine in theta (R3);')
    print('  (d) optimisation need not reach the minimiser in the budget.')


# ----------------------------------------------------------------------------------- R3
def result_3_nonlinear_remainder():
    """Baseline nonlinear in theta: first-order validity and the second-order remainder."""
    banner('R3  BASELINE NONLINEAR IN THETA: THE SECOND-ORDER REMAINDER')

    print('Write the exact Taylor expansion of the baseline about theta*, with H the (n x p x p)')
    print('second derivative and H[a, b] the vector with entries sum_jk H_ijk a_j b_k:')
    print()
    print('    f_base(theta* + dtheta) = f_base(theta*) + J dtheta + 1/2 H[dtheta, dtheta] + O(|dtheta|^3)')
    print()
    print('Interpolation and the OBC constraint now read')
    print('    J dtheta + 1/2 H[dtheta, dtheta] + f_ann = Delta*,     J^T f_ann = 0,')
    print('so left-multiplying by J^T,')
    print()
    print('    (J^T J) dtheta = J^T Delta* - 1/2 J^T H[dtheta, dtheta]')
    print('    dtheta = J^+ Delta* - 1/2 J^+ H[dtheta, dtheta].                 ... (5)')
    print()
    print('Substituting dtheta = J^+ Delta* + O(||Delta*||^2) into the remainder of (5):')
    print()
    print('    dtheta = J^+ Delta*  -  1/2 J^+ H[J^+ Delta*, J^+ Delta*]  +  O(||Delta*||^3).  ... (6)')
    print()

    # Verify (6) as a series in a bookkeeping parameter eps scaling Delta*, in a case small
    # enough to expand exactly: n = 3, p = 1, f_base genuinely nonlinear in the parameter.
    eps = sp.Symbol('eps', positive=True)
    t = sp.Symbol('t', real=True)                     # dtheta, scalar
    th_star = sp.Symbol('thstar', real=True)
    z1, z2, z3 = sp.symbols('z1 z2 z3', real=True)    # reference-set dependence

    # A baseline nonlinear in the parameter, one parameter, three stacked rows.
    f_base = sp.Matrix([sp.exp(th_star + t) * z1,
                        (th_star + t) ** 2 * z2,
                        sp.sin(th_star + t) * z3])
    J_sym = sp.Matrix([sp.diff(f_base[i], t) for i in range(3)]).subs(t, 0)
    H_sym = sp.Matrix([sp.diff(f_base[i], t, 2) for i in range(3)]).subs(t, 0)

    D = eps * symbolic_vector('D', 3)                 # Delta* scaled by the bookkeeping eps
    f0 = f_base.subs(t, 0)

    # The one scalar equation that survives after projecting the interpolation onto J:
    #   J^T ( f_base(theta* + t) - f_base(theta*) ) = J^T Delta*      (J^T f_ann = 0)
    equation = (J_sym.T * (f_base - f0))[0, 0] - (J_sym.T * D)[0, 0]

    # Solve as a power series in eps: t = c1 eps + c2 eps^2 + ...
    c1, c2 = sp.symbols('c1 c2', real=True)
    series = sp.series(equation.subs(t, c1 * eps + c2 * eps ** 2), eps, 0, 3).removeO()
    poly = sp.Poly(sp.expand(series), eps)
    sol = sp.solve([poly.coeff_monomial(eps), poly.coeff_monomial(eps ** 2)], [c1, c2], dict=True)
    assert sol, 'the series solve returned nothing'
    sol = sol[0]

    JtJ = (J_sym.T * J_sym)[0, 0]
    c1_claim = (J_sym.T * symbolic_vector('D', 3))[0, 0] / JtJ           # J^+ Delta* / eps
    c2_claim = -sp.Rational(1, 2) * (J_sym.T * H_sym)[0, 0] * c1_claim ** 2 / JtJ

    print('CHECK  order eps  (first-order term) against J^+ Delta*:')
    r1 = sp.simplify(sol[c1] - c1_claim)
    print('      ', r1)
    assert sp.simplify(r1) == 0, 'R3 failed at first order'
    print('       -> zero.')
    print('CHECK  order eps^2 (remainder) against -1/2 J^+ H[J^+ Delta*, J^+ Delta*]:')
    r2 = sp.simplify(sol[c2] - c2_claim)
    print('      ', r2)
    assert sp.simplify(r2) == 0, 'R3 failed at second order'
    print('       -> zero. Equation (6) is confirmed.')
    print()
    print('SIZE OF THE REMAINDER, relative to the leading term:')
    print('    || 1/2 J^+ H[dtheta, dtheta] || / || J^+ Delta* ||  ~  1/2 kappa ||dtheta||,')
    print('with kappa = ||J^+|| ||H|| a curvature-to-sensitivity ratio with units of 1/theta.')
    print('For the gantry the nine log coordinates give H = J elementwise in the scalar')
    print('directions (d^2/dv^2 exp(v) = d/dv exp(v)), so kappa ~ 1 in the free coordinate and')
    print('the relative remainder is about ||dtheta||/2. At the run\'s 9.5 percent initial')
    print('detuning that is roughly 5 percent OF the bias, not 5 percentage points: it cannot')
    print('turn a 2 percent prediction into a 7 percent observation.')


# ----------------------------------------------------------------------------------- R4
def result_4_wrong_expansion_point():
    """Basis built at theta_0, sensitivity evaluated at theta*: an oblique projection."""
    banner('R4  A BASIS BUILT AT THE WRONG EXPANSION POINT')

    print('In training the basis is refreshed at the CURRENT parameter point theta_0, not at the')
    print('unknown theta*. The constraint is then J_0^T f_ann = 0 while the parameter direction')
    print('the model actually moves along is J = d f_base / d theta at theta*. Interpolation:')
    print()
    print('    J dtheta + f_ann = Delta*,     J_0^T f_ann = 0')
    print('    => (J_0^T J) dtheta = J_0^T Delta*')
    print('    => dtheta = (J_0^T J)^-1 J_0^T Delta*.                            ... (7)')
    print()

    n, p = 4, 2
    J = symbolic_matrix('J', n, p)
    J0 = symbolic_matrix('K', n, p)
    D = symbolic_vector('D', n)

    dth_claim = (J0.T * J).inv() * J0.T * D
    fa_claim = D - J * dth_claim
    res1 = sp.simplify(J * dth_claim + fa_claim - D)
    res2 = sp.simplify(J0.T * fa_claim)
    print('CHECK (existence)  residual of the interpolation at (7):')
    print('      ', res1.T)
    assert res1 == sp.zeros(n, 1), 'R4 failed: interpolation residual'
    print('CHECK (existence)  residual of the constraint J_0^T f_ann = 0 at (7):')
    print('      ', res2.T)
    assert res2 == sp.zeros(p, 1), 'R4 failed: constraint residual'
    print('       -> both zero; uniqueness follows as in R1 with J_0^T J invertible.')
    print()

    Pobl = J * (J0.T * J).inv() * J0.T
    idem = sp.simplify(Pobl * Pobl - Pobl)
    assert idem == sp.zeros(n, n), 'the oblique projector is not idempotent'
    print('CHECK  J (J_0^T J)^-1 J_0^T is idempotent: verified symbolically.')
    print('       It is NOT symmetric unless range(J_0) = range(J).')
    print()
    print('(7) is an OBLIQUE projection: J (J_0^T J)^-1 J_0^T is idempotent but not symmetric.')
    print('It reduces to J^+ when J_0 = J. Two consequences:')
    print('  (i)  the vanishing condition becomes J_0^T Delta* = 0, a condition on the basis')
    print('       actually used, not on the truth basis;')
    print('  (ii) the bias can be LARGER than the orthogonal one, because (J_0^T J)^-1 is not')
    print('       bounded by 1/sigma_min(J) when the two column spaces are misaligned.')
    print('This is why the numerical companion builds J at BOTH theta* and the detuned theta_0')
    print('and reports the gap between the two predictions as the expansion-point sensitivity.')


# ----------------------------------------------------------------------------------- R5
def result_5_rank_deficiency():
    """What the statement degrades to when J loses column rank."""
    banner('R5  RANK DEFICIENCY')

    print('If rank(J) = r < p, then J^T J is singular and (3) has no unique solution: any')
    print('dtheta + v with v in ker(J) interpolates equally well, and the fit cannot distinguish')
    print('them. Projection does not help, because ker(J) directions leave f_base unchanged to')
    print('first order and therefore leave the orthogonality constraint unchanged too.')
    print()
    print('    dtheta in J^+ Delta* + ker(J),   minimum-norm representative J^+ Delta*.')
    print()
    print('The gantry basis is FULL rank 10 of 10 at condition 8.9e+02, so this branch is not')
    print('active here. It is stated because it is the second way OBC can fail to recover theta*,')
    print('and it is the one the ten-combination reduction (D-190) was introduced to remove: the')
    print('fourteen raw scalars give rank 10 and nullity 4, where every statement above is void')
    print('on four directions.')


def main():
    print(__doc__.split('Run:')[0].strip())
    result_1_affine_exact()
    result_2_least_squares_capacity()
    result_3_nonlinear_remainder()
    result_4_wrong_expansion_point()
    result_5_rank_deficiency()

    banner('SUMMARY')
    print('R1  dtheta = J^+ Delta*,  and theta_hat = theta* iff J^T Delta* = 0.   [exact, affine]')
    print('R2  the same dtheta holds under least squares with ANY capacity-limited f_ann,')
    print('    because the objective separates into a range(J) part and a range(J)^perp part.')
    print('R3  dtheta = J^+ Delta* - 1/2 J^+ H[J^+ Delta*, J^+ Delta*] + O(||Delta*||^3).')
    print('R4  with the basis at theta_0: dtheta = (J_0^T J)^-1 J_0^T Delta*, oblique.')
    print('R5  with rank(J) < p the bias is fixed only modulo ker(J).')
    print()
    print('The two numbers to measure:')
    print('    rho*    = ||P Delta*|| / ||Delta*||     how non-orthogonal the truth actually is')
    print('    J^+ Delta*                              the predicted parameter bias, per parameter')
    print('Both are computed by measure_delta_orthogonality.py in this folder.')


if __name__ == '__main__':
    main()
