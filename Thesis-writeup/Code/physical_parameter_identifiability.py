"""Structural identifiability of the gantry baseline: raw parameters -> model coefficients."""
import sympy as sp

m1, m2, mb, mh, Jb, Jh, cg1, cg2, cy, cb1, cb2, kb1, kb2, d = sp.symbols(
    "m1 m2 mb mh Jb Jh cg1 cg2 cy cb1 cb2 kb1 kb2 d", positive=True)
Lb, Y = sp.symbols("Lb Y", positive=True)
raw = [m1, m2, mb, mh, Jb, Jh, cg1, cg2, cy, cb1, cb2, kb1, kb2, d]

M = sp.Matrix([[m1 + m2 + mb + mh, (m1 - m2) * Lb / 2 - mh * Y, 0],
               [(m1 - m2) * Lb / 2 - mh * Y, Jb + Jh + (m1 + m2) * Lb**2 / 4 + mh * d**2 + mh * Y**2, -mh * d],
               [0, -mh * d, mh]])
C = sp.Matrix([[cg1 + cg2, (cg1 - cg2) * Lb / 2, 0],
               [(cg1 - cg2) * Lb / 2, cb1 + cb2 + (cg1 + cg2) * Lb**2 / 4, 0],
               [0, 0, cy]])
K = sp.zeros(3, 3); K[1, 1] = kb1 + kb2

# Every coefficient the dynamics can depend on: entries of M0, M1, M2, C, K (upper triangle).
coeffs = []
for Mi in [M.applyfunc(lambda e: sp.expand(e).coeff(Y, k)) for k in range(3)] + [C, K]:
    for i in range(3):
        for j in range(i, 3):
            e = sp.simplify(Mi[i, j])
            if e != 0:
                coeffs.append(e)
print("nonzero coefficients:", len(coeffs))
for c in coeffs:
    print("  ", c)

J = sp.Matrix(coeffs).jacobian(raw)
r = J.rank(simplify=True)
print("rank of coefficient Jacobian:", r, " nullity:", len(raw) - r)
print("null space (gauge directions), raw order", raw)
for v in J.nullspace(simplify=True):
    print("  ", list(sp.simplify(v.T)))

combos = [kb1 + kb2, cg1, cg2, cy, cb1 + cb2, mh, m1 + m2 + mb, m1 - m2,
          Jb + Jh + (m1 + m2) * Lb**2 / 4, d]
Jc = sp.Matrix(combos).jacobian(raw)
print("rank of combination Jacobian:", Jc.rank(simplify=True))
# Coefficients are functions of the combinations only: stacked Jacobian keeps rank 10.
print("rank of [J; Jc]:", J.col_join(Jc).rank(simplify=True))
