# M(Y) Invertibility Check

Proves that the gantry inertia matrix M(Y) is positive definite (and therefore invertible)
for all Y in the operational range. This is required for the CT ODE formulation
`dxdt = [0, I; -M(Y)^{-1}K, -M(Y)^{-1}C] x + [0; M(Y)^{-1}] u` to be well-defined.

Source: `kamtin-fp-model/03 Simulink gantry/functions/gantrySystem.m`, `docs/fp-model-structure.md`

---

## 1. M(Y) definition

The 3x3 inertia matrix in logical coordinates [X, Theta, Y]:

```
M(Y) = [ a,    b(Y),   0   ]
       [ b(Y), c(Y),  -e   ]
       [ 0,   -e,      f   ]
```

where:
```
a    = m1 + m2 + mb + mh                              = 53.8       (constant)
b(Y) = (m1 - m2)*Lb/2 - mh*Y                         = -0.18125 - 10.1*Y
c(Y) = Jb + Jh + (m1+m2)*Lb^2/4 + mh*d^2 + mh*Y^2   = 3.89739 + 10.1*Y^2
e    = mh*d                                            = 1.01       (constant)
f    = mh                                              = 10.1       (constant)
```

Numerical parameter values (from `main.m`):
```
mb = 22.8 kg,   mh = 10.1 kg,   m1 = 10.2 kg,   m2 = 10.7 kg
Jb = 1.0 kg*m^2,  Jh = 0.05 kg*m^2
Lb = 0.725 m,   d = 0.1 m
```

Intermediate values:
```
(m1 - m2)*Lb/2 = (10.2 - 10.7) * 0.725/2 = -0.18125
(m1 + m2)*Lb^2/4 = 20.9 * 0.525625/4 = 2.74639
mh*d^2 = 10.1 * 0.01 = 0.101
Jb + Jh = 1.05
```

---

## 2. Sylvester's criterion for positive definiteness

A symmetric matrix is positive definite if and only if all leading principal minors are
strictly positive. M(Y) is symmetric by construction (inertia matrix from Lagrangian).

### Minor 1: M[0,0]

```
D1 = a = 53.8 > 0   CHECK
```

### Minor 2: det of top-left 2x2

```
D2 = a*c(Y) - b(Y)^2
   = 53.8 * (3.89739 + 10.1*Y^2) - (-0.18125 - 10.1*Y)^2
```

Expanding:
```
a*c(Y) = 53.8 * 3.89739 + 53.8 * 10.1 * Y^2
       = 209.680 + 543.38 * Y^2

b(Y)^2 = (-0.18125)^2 + 2*0.18125*10.1*Y + (10.1)^2 * Y^2
       = 0.03285 + 3.66125*Y + 102.01*Y^2

D2 = 209.680 + 543.38*Y^2 - 0.03285 - 3.66125*Y - 102.01*Y^2
   = 441.37*Y^2 - 3.66125*Y + 209.647
```

This is a quadratic in Y (opens upward, coefficient 441.37 > 0).

Discriminant: `3.66125^2 - 4 * 441.37 * 209.647 = 13.40 - 370,022 = -370,009`

**Discriminant < 0**: no real roots. Combined with positive leading coefficient,
D2 > 0 for all Y in R.

```
D2(Y) > 0  for all Y in R   CHECK
```

### Minor 3: det(M(Y))

Using cofactor expansion along the first column (M[2,0] = 0):

```
det(M) = a * (c(Y)*f - e^2) - b(Y) * (b(Y)*f - 0)
       = a * (c(Y)*f - e^2) - f * b(Y)^2
       = f * [a*(c(Y) - e^2/f) - b(Y)^2]
```

Wait, let me redo this more carefully. Expanding along row 0:

```
det(M) = a * det([c(Y), -e; -e, f]) - b(Y) * det([b(Y), -e; 0, f]) + 0
       = a * (c(Y)*f - e^2) - b(Y) * (b(Y)*f)
       = a*f*c(Y) - a*e^2 - f*b(Y)^2
```

Factor out f from two terms:

```
det(M) = f * [a*c(Y) - b(Y)^2] - a*e^2
       = f * D2 - a*e^2
```

Substituting:
```
f * D2 = 10.1 * (441.37*Y^2 - 3.66125*Y + 209.647)
       = 4457.8*Y^2 - 36.979*Y + 2117.4

a*e^2  = 53.8 * 1.01^2 = 53.8 * 1.0201 = 54.88

det(M) = 4457.8*Y^2 - 36.979*Y + 2117.4 - 54.88
       = 4457.8*Y^2 - 36.979*Y + 2062.5
```

This is a quadratic in Y (opens upward, coefficient 4457.8 > 0).

Discriminant: `36.979^2 - 4 * 4457.8 * 2062.5 = 1367.4 - 36,756,450 = -36,755,083`

**Discriminant < 0**: no real roots. Combined with positive leading coefficient,
det(M) > 0 for all Y in R.

```
det(M(Y)) > 0  for all Y in R   CHECK
```

---

## 3. Conclusion

All three leading principal minors of M(Y) are strictly positive for all Y in R:

| Minor | Expression | Sign |
|-------|-----------|------|
| D1 = a | 53.8 | > 0 (constant) |
| D2 = a*c(Y) - b(Y)^2 | 441.37*Y^2 - 3.66125*Y + 209.647 | > 0 (no real roots, opens up) |
| D3 = det(M) | 4457.8*Y^2 - 36.979*Y + 2062.5 | > 0 (no real roots, opens up) |

By Sylvester's criterion, **M(Y) is positive definite for all Y in R**.

Since positive definiteness implies invertibility, **M(Y)^{-1} exists for all Y in R**.
This is a stronger result than needed: invertibility holds not just in the operational
range [-0.35, 0.35] m but everywhere.

**Physical interpretation**: This is expected. The kinetic energy of the gantry is
T = 0.5 * q_dot^T * M(Y) * q_dot. For T > 0 for any nonzero velocity (as required by
physics), M(Y) must be positive definite. Any physical Euler-Lagrange system with bounded
mass has a positive definite inertia matrix.

**Textbook reference**: See e.g. Murray, Li & Sastry (1994), "A Mathematical Introduction
to Robotic Manipulation", Proposition 4.2: the inertia matrix of a mechanical system
derived from the Lagrangian is symmetric positive definite.

---

## 4. Condition number (numerical health)

While M(Y) is invertible everywhere, a high condition number could cause numerical issues
when computing M(Y)^{-1} inside the CT ODE.

The condition number is not a polynomial in Y, so an analytical check is not possible.
A numerical sweep should be performed.

- [ ] Compute cond(M(Y)) for Y in linspace(-0.35, 0.35, 200)
- [ ] Report the maximum condition number
- [ ] Pass criterion: cond(M) < 1e4 (reasonable for double precision)

**Preliminary estimate at Y=0**: det(M(0)) = 2062.5, max entry ~ 53.8,
so cond is expected to be moderate (order 10-100). Not a concern.

---

## 5. Trainable parameters warning

The analysis above uses the nominal parameter values from `main.m`. If any inertia
parameters (mb, mh, m1, m2, Jb, Jh, Lb, d) are made trainable during augmentation,
the polynomial coefficients change and the roots shift.

Specifically, the discriminant of det(M(Y)) could become positive if parameters deviate
enough, creating real roots where M(Y) becomes singular.

**Recommendation**: Keep inertia parameters fixed during augmentation. Only allow
damping (cg1, cg2, cy, cb1, cb2) and stiffness (kb1, kb2) to be trainable.
These enter C and K, not M(Y), so invertibility is unaffected.

If inertia parameters must be trainable, add a regularization term or eigenvalue
constraint during training. See `tasks/todo.md` Task 3.7 KEEP IN MIND block.
