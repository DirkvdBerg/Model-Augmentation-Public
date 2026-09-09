"""CAS benchmark, sympy side: is MATLAB actually better for our algebra?

The question is not which CAS is stronger in general. It is whether the
difference binds on THIS workload, which is small symbolic matrices with
polynomial entries in 14 parameters plus the scheduling variable Y.

Tasks are graded from what we have already done to the heaviest thing coming,
and are chosen to be useful rather than synthetic: T1 to T3 are the ingredients
of the LFR rational rewrite of M(Y)^-1, and T5 reports the degree of det(M(Y))
in Y, which is what decides whether the rewrite closes at second order.

The MATLAB companion `cas_benchmark.m` runs the identical task list, from the
ground truth main.m rather than from the Python transcription. Timings exclude
engine startup on both sides.

Outputs -> scripts/gantry/orthogonality/code/cas_benchmark_sympy.json
Run: conda run -n GraduationProject python scripts/gantry/orthogonality/code/cas_benchmark.py
"""
import os
import json
import time

import sympy as sp

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'cas_benchmark_sympy.json')

kb1, kb2 = sp.symbols('kb1 kb2', real=True, positive=True)
cg1, cg2, cy = sp.symbols('cg1 cg2 cy', real=True, positive=True)
cb1, cb2 = sp.symbols('cb1 cb2', real=True, positive=True)
mh, m1, m2, mb = sp.symbols('mh m1 m2 mb', real=True, positive=True)
Jb, Jh, d = sp.symbols('Jb Jh d', real=True, positive=True)
Lb = sp.Symbol('Lb', real=True, positive=True)
Y = sp.Symbol('Y', real=True)


def matrices():
    M = sp.Matrix([
        [m1 + m2 + mb + mh, (m1 - m2) * Lb / 2 - mh * Y, 0],
        [(m1 - m2) * Lb / 2 - mh * Y,
         Jb + Jh + (m1 + m2) * Lb**2 / 4 + mh * d**2 + mh * Y**2, -mh * d],
        [0, -mh * d, mh]])
    C = sp.Matrix([
        [cg1 + cg2, (cg1 - cg2) * Lb / 2, 0],
        [(cg1 - cg2) * Lb / 2, cb1 + cb2 + (cg1 + cg2) * Lb**2 / 4, 0],
        [0, 0, cy]])
    K = sp.Matrix([[0, 0, 0], [0, kb1 + kb2, 0], [0, 0, 0]])
    M1 = sp.Matrix([[0, -mh, 0], [-mh, 0, 0], [0, 0, 0]])
    M2 = sp.Matrix([[0, 0, 0], [0, mh, 0], [0, 0, 0]])
    return M, C, K, M1, M2


def timed(label, fn, results):
    t0 = time.perf_counter()
    val = fn()
    dt = time.perf_counter() - t0
    results[label] = dt
    print(f"  {label:34s} {dt:8.3f} s")
    return val


def main():
    M, C, K, M1, M2 = matrices()
    r = {}
    print("sympy", sp.__version__)
    print("\ntimings (engine startup excluded):")

    detM = timed("T1 det(M(Y)) expanded",
                 lambda: sp.expand(M.det()), r)
    adjM = timed("T2 adj(M(Y)) expanded",
                 lambda: sp.expand(M.adjugate()), r)
    Minv = timed("T3 simplify(M(Y)^-1)",
                 lambda: sp.simplify(M.inv()), r)

    def products():
        Mi = adjM / detM
        return [sp.simplify(Mi * X) for X in (K, C, M1, M2)]
    timed("T4 simplify(M^-1 X), X in K,C,M1,M2", products, r)

    def degree_in_Y():
        p = sp.Poly(sp.expand(detM), Y)
        return p.degree(), [sp.simplify(c) for c in p.all_coeffs()]
    deg, coeffs = timed("T5 degree of det(M(Y)) in Y", degree_in_Y, r)

    total = sum(r.values())
    print(f"  {'TOTAL':34s} {total:8.3f} s")

    # Useful side result, not just a timing: does the LFR rewrite close at
    # second order in Y, i.e. is det(M(Y)) quadratic and is adj(M(Y)) at most
    # quadratic entrywise? N0/N1/N2 in gantry_ss.py assume exactly that shape.
    adj_degs = sorted({sp.Poly(sp.expand(e), Y).degree() for e in adjM})
    print(f"\ndet(M(Y)) degree in Y      : {deg}")
    print(f"adj(M(Y)) entry degrees in Y: {adj_degs}")

    r_out = {"engine": f"sympy {sp.__version__}",
             "timings_s": r, "total_s": total,
             "det_degree_in_Y": int(deg),
             "adj_entry_degrees_in_Y": [int(x) for x in adj_degs]}
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(r_out, fh, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
