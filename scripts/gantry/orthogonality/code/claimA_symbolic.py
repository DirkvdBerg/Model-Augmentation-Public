"""Claim A, symbolically: the baseline factors exactly through kappa(theta).

This PROVES what `nullspace_check.py` only corroborated. That script evaluated
the parameter Jacobian numerically at theta_0 and at 60 sampled Y values; the
claim it tests must hold for ALL theta and ALL Y, because theta moves during
training. A symbolic identity gives that; a numerical residual does not.

WHAT IS PROVED
--------------
Every parameter-dependent matrix of the baseline is annihilated by the four
directions v1..v4, identically in all 14 parameters and in Y:

    M0, M1, M2   (inertia, from gantry_ss.py:70-84)
    K, C         (stiffness and damping, gantry_ss.py:90-100)
    M(Y) = M0 + M1 Y + M2 Y^2
    det(M0), adj(M0)                      (the rational inverse's ingredients)

Everything downstream is a function of these alone: `build_G_matrix_entries`
forms M0inv = adj(M0)/det(M0) and products with K, C, M1, M2, and the RK4
discretisation is a fixed composition of the resulting vector field. So by the
chain rule, annihilating the list above annihilates the discrete transition
f_base and hence every column combination of Phi. Differentiating through RK4
symbolically is therefore unnecessary, not skipped.

STRUCTURAL READING (the proof in one line, which the symbolics then confirm)
---------------------------------------------------------------------------
`build_poly_constants` defines alpha = m1+m2+mb+mh, beta = (m1-m2) Lb/2,
gamma = Jb+Jh+(m1+m2) Lb^2/4, and M0/M1/M2 depend on the raw parameters ONLY
through alpha, beta, gamma, mh, d. Against the identifiable combinations:
    alpha = m_total + mh,   beta = m_diff * Lb / 2,   gamma = J_eff exactly.
K uses only kb1+kb2 = kb_sum; C uses cg1, cg2, cb1+cb2 = cb_sum, cy. Every one
of these is a function of kappa, so the factorization is structural.

NEGATIVE CONTROLS
-----------------
A test that cannot fail proves nothing. Directions deliberately OUTSIDE the
predicted kernel must produce a nonzero derivative, and are asserted to.

Outputs -> scripts/gantry/orthogonality/code/claimA_symbolic.json
Run: conda run -n GraduationProject python scripts/gantry/orthogonality/code/claimA_symbolic.py
"""
import os
import json

import sympy as sp

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'claimA_symbolic.json')

PNAMES = ["kb1", "kb2", "cg1", "cg2", "cy", "cb1", "cb2",
          "mh", "m1", "m2", "mb", "Jb", "Jh", "d"]

# Symbols. Lb and Y are free symbols too, so the identity is proved for every
# cross-arm length and every scheduling value, not for the deployed numbers.
S = {n: sp.Symbol(n, real=True, positive=True) for n in PNAMES}
Lb = sp.Symbol('Lb', real=True, positive=True)
Y = sp.Symbol('Y', real=True)

kb1, kb2 = S['kb1'], S['kb2']
cg1, cg2, cy = S['cg1'], S['cg2'], S['cy']
cb1, cb2 = S['cb1'], S['cb2']
mh, m1, m2, mb = S['mh'], S['m1'], S['m2'], S['mb']
Jb, Jh, d = S['Jb'], S['Jh'], S['d']


def baseline_matrices():
    """Transcribed from model_augmentation/systems/gantry_ss.py lines 70-100."""
    M0 = sp.zeros(3, 3)
    M0[0, 0] = m1 + m2 + mb + mh
    M0[0, 1] = (m1 - m2) * Lb / 2
    M0[1, 0] = (m1 - m2) * Lb / 2
    M0[1, 1] = Jb + Jh + (m1 + m2) * Lb**2 / 4 + mh * d**2
    M0[1, 2] = -mh * d
    M0[2, 1] = -mh * d
    M0[2, 2] = mh

    M1 = sp.zeros(3, 3)
    M1[0, 1] = -mh
    M1[1, 0] = -mh

    M2 = sp.zeros(3, 3)
    M2[1, 1] = mh

    C = sp.zeros(3, 3)
    C[0, 0] = cg1 + cg2
    C[0, 1] = (cg1 - cg2) * Lb / 2
    C[1, 0] = (cg1 - cg2) * Lb / 2
    C[1, 1] = cb1 + cb2 + (cg1 + cg2) * Lb**2 / 4
    C[2, 2] = cy

    K = sp.zeros(3, 3)
    K[1, 1] = kb1 + kb2

    MY = M0 + M1 * Y + M2 * Y**2
    return {"M0": M0, "M1": M1, "M2": M2, "C": C, "K": K, "M_of_Y": MY,
            "det_M0": sp.Matrix([[M0.det()]]),
            "adj_M0": M0.adjugate()}


def combinations():
    """The 10 identifiable combinations, blocks.py::_combos_from_raw."""
    return {
        "kb_sum": kb1 + kb2,
        "cg1": cg1, "cg2": cg2, "cy": cy,
        "cb_sum": cb1 + cb2,
        "mh": mh,
        "m_total": m1 + m2 + mb,
        "m_diff": m1 - m2,
        "J_eff": Jb + Jh + (m1 + m2) * Lb**2 / 4,
        "d": d,
    }


def predicted_directions():
    """v1..v4 as dicts {param: coefficient}; zero coefficients omitted."""
    return {
        "v1 kb1-kb2": {"kb1": 1, "kb2": -1},
        "v2 cb1-cb2": {"cb1": 1, "cb2": -1},
        "v3 Jb-Jh":   {"Jb": 1, "Jh": -1},
        "v4 mass-inertia": {"m1": 1, "m2": 1, "mb": -2, "Jh": -Lb**2 / 2},
    }


def negative_controls():
    """Directions OUTSIDE the kernel. Each MUST give a nonzero derivative."""
    return {
        "kb1+kb2 (moves kb_sum)": {"kb1": 1, "kb2": 1},
        "m1 alone (moves m_total and m_diff)": {"m1": 1},
        "Jb+Jh (moves J_eff)": {"Jb": 1, "Jh": 1},
        "mb alone (moves m_total)": {"mb": 1},
        "v4 with a wrong Jh coefficient": {"m1": 1, "m2": 1, "mb": -2,
                                           "Jh": -Lb**2 / 3},
    }


def dir_derivative(expr_mat, direction):
    """Directional derivative sum_i v_i d/d theta_i, entrywise, simplified."""
    out = sp.zeros(*expr_mat.shape)
    for name, coeff in direction.items():
        out += coeff * expr_mat.diff(S[name])
    return sp.simplify(out)


def main():
    mats = baseline_matrices()
    combos = combinations()
    dirs = predicted_directions()
    negs = negative_controls()

    print("Symbols free: all 14 parameters, Lb, Y. Identities hold for every "
          "value of each.\n")

    res = {"matrices_tested": sorted(mats.keys()),
           "positive": {}, "negative": {}, "combination_invariance": {},
           "kernel_dimension": None}

    # --- positive: every matrix annihilated by every predicted direction ----
    print("[A] PREDICTED DIRECTIONS: every matrix entry must be identically 0")
    all_zero = True
    for dname, dvec in dirs.items():
        per_mat = {}
        for mname, M in mats.items():
            D = dir_derivative(M, dvec)
            is_zero = bool(D.is_zero_matrix) or D == sp.zeros(*D.shape)
            per_mat[mname] = bool(is_zero)
            all_zero &= bool(is_zero)
        ok = all(per_mat.values())
        print(f"  {dname:22s} {'ALL ZERO' if ok else 'NONZERO: ' + str(per_mat)}")
        res["positive"][dname] = per_mat

    # --- the combinations are constant along those directions ---------------
    print("\n[B] The 10 combinations are invariant along v1..v4 "
          "(definition of ker(d kappa))")
    for dname, dvec in dirs.items():
        nz = []
        for cname, cexpr in combos.items():
            dc = sp.simplify(sum(c * sp.diff(cexpr, S[n])
                                 for n, c in dvec.items()))
            if dc != 0:
                nz.append((cname, str(dc)))
        print(f"  {dname:22s} {'all invariant' if not nz else 'MOVES ' + str(nz)}")
        res["combination_invariance"][dname] = nz

    # --- negative controls --------------------------------------------------
    print("\n[C] NEGATIVE CONTROLS: each must move at least one matrix")
    neg_ok = True
    for dname, dvec in negs.items():
        moved = []
        for mname, M in mats.items():
            D = dir_derivative(M, dvec)
            if not (bool(D.is_zero_matrix) or D == sp.zeros(*D.shape)):
                moved.append(mname)
        ok = len(moved) > 0
        neg_ok &= ok
        print(f"  {dname:38s} {'moves ' + ','.join(moved) if ok else 'FAILED: annihilated'}")
        res["negative"][dname] = moved

    # --- kernel dimension of d kappa is exactly 4 ---------------------------
    Jk = sp.Matrix([[sp.diff(c, S[n]) for n in PNAMES] for c in combos.values()])
    rank_k = Jk.rank()
    ker_dim = len(PNAMES) - rank_k
    print(f"\n[D] rank(d kappa) = {rank_k}, so dim ker(d kappa) = {ker_dim} "
          f"(predicted 4, and we exhibit 4 independent directions)")
    res["kernel_dimension"] = {"rank_dkappa": int(rank_k),
                               "ker_dim": int(ker_dim)}

    verdict = bool(all_zero and neg_ok and ker_dim == 4)
    print(f"\nCLAIM A: {'PROVED' if verdict else 'NOT PROVED'}")
    res["claim_A_proved"] = verdict

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
