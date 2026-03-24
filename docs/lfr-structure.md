# LFR Structure for the Gantry LPV Model

Outlines the Linear Fractional Representation (LFR) as it applies to the gantry
baseline model and the augmentation framework. Based on Drenth (2025) thesis Ch. 2
and Ch. 5, and the companion IFAC paper.

---

## 1. What is an LFR?

An LFR represents an LPV system as a pair {G, Delta(p)}, where G is a constant
interconnection matrix and Delta(p) is a structured matrix that depends on the
scheduling variable p. The key idea: all parameter dependence is isolated in Delta(p),
while G captures the system dynamics in a fixed, linear structure.

**Reference**: Drenth thesis eq. 2.1, IFAC paper eq. 6, Zhou et al. (1996) Ch. 10.

### Continuous-time LFR (Drenth eq. 2.1)

```
G: { [x_dot(t)]   [A_x    B_w    B_u ] [x(t)]
     [z(t)    ] = [C_z    D_zw   D_zu] [w(t)]
     [y(t)    ]   [C_y    D_yw   D_yu] [u(t)]

     w(t) = Delta(p(t)) * z(t)
```

| Symbol | Meaning | Dimension |
|--------|---------|-----------|
| x(t) | State vector | n_x |
| u(t) | System input | n_u |
| y(t) | System output | n_y |
| z(t) | Latent output (to Delta block) | n_w |
| w(t) | Latent input (from Delta block) | n_w |
| p(t) | Scheduling variable | n_p |
| G | Constant interconnection matrix | (n_x + n_w + n_y) x (n_x + n_w + n_u) |
| Delta(p) | Scheduling block | n_w x n_w |

### The Delta block (Drenth eq. 2.2)

```
Delta(p) = diag(p_1 * I_{eta_1}, p_2 * I_{eta_2}, ..., p_{n_p} * I_{eta_{n_p}})
```

Each scheduling variable p_i is repeated eta_i times on the diagonal. The repetition
count eta controls how rich the rational dependence on p can be:
- eta = 0: LTI model (no parameter dependence)
- eta = 1 per variable: affine dependence on p (equivalent to LPV-SS with D_zw = 0)
- eta > 1: rational dependence on p (captures rational functions like M(Y)^{-1})

Total dimension of Delta: n_w = sum(eta_i).

### Recovering LPV-SS from LFR (Drenth eq. 2.3)

By eliminating the latent variables w(t) and z(t):

```
z(t) = (I - D_zw * Delta(p))^{-1} * (C_z * x(t) + D_zu * u(t))
```

Substituting back:

```
A(p) = A_x + B_w * Delta(p) * (I - D_zw * Delta(p))^{-1} * C_z
B(p) = B_u + B_w * Delta(p) * (I - D_zw * Delta(p))^{-1} * D_zu
C(p) = C_y + D_yw * Delta(p) * (I - D_zw * Delta(p))^{-1} * C_z
D(p) = D_yu + D_yw * Delta(p) * (I - D_zw * Delta(p))^{-1} * D_zu
```

This gives the parameter-dependent state-space matrices as rational functions of p,
which is exactly what the gantry model has (rational entries via M(Y)^{-1}).

---

## 2. Well-posedness

The LFR is well-posed if the matrix inversion `(I - D_zw * Delta(p))^{-1}` exists
for all valid scheduling values.

**Definition** (Drenth thesis Definition 2.1, IFAC Definition 1):
The LFR {G, Delta(p)} is well-posed if `det(I - D_zw * Delta(p)) != 0` for all p in P.

**Sufficient condition** (Drenth Theorem 2.5, IFAC Theorem 6):
If Delta(p) is diagonal (Assumption 2.1), p is bounded in the unit ball ||p||_inf <= 1
(Assumption 2.2), and the spectral radius rho(D_zw) < 1 (Condition 2.4), then
the LFR is well-posed.

### Direct parameterization for guaranteed well-posedness

Drenth proposes (eq. 2.12):

```
D_zw = exp(-N),    N > 0  (positive definite)
```

Since exp maps positive definite matrices to matrices with spectral radius < 1,
this guarantees rho(D_zw) < 1 by construction. The parameterization of N is
(Drenth eq. 2.13):

```
N = Psi(D_A^T * D_A + D_B - D_B^T + epsilon * I)
```

where Psi = diag(exp(D_d)) is a positive scaling, D_A, D_B, D_d are free parameters,
and epsilon is a small positive constant. This makes N strictly positive definite
by construction, enabling unconstrained optimization.

**For the gantry**: The augmentation LFR uses this parameterization. The baseline LFR
has a fixed D_zw determined by the physics (see Section 4).

### Scheduling variable range

Assumption 2.2 requires ||p||_inf <= 1. For the gantry, p = Y with Y in [-0.35, 0.35] m.
Since |Y| <= 0.35 < 1, this assumption is satisfied without scaling.

If desired, a scaling T_p can be applied (Drenth eq. 5.7):
```
p_scaled = sat(T_p * psi(x, u, d))
```
where sat clips to [-1, 1]. This is not strictly needed for Y but provides headroom.

---

## 3. Application to the gantry: what we have

### The CT ODE

From `docs/fp-model-structure.md` and `kamtin-fp-model/functions/gantrySystem.m`:

```
x_dot = A_c(Y) * x + B_c(Y) * u

A_c(Y) = [ 0_{3x3}      I_{3x3}    ]
          [-M(Y)^{-1}*K  -M(Y)^{-1}*C]

B_c(Y) = [ 0_{3x3}  ]
          [ M(Y)^{-1}]
```

where M(Y) is the 3x3 inertia matrix (see `docs/m-matrix-invertibility.md` for proof
that M(Y)^{-1} exists for all Y).

### The parameter dependence structure

M(Y) is a 3x3 symmetric matrix polynomial in Y:

```
M(Y) = M_0 + Y * M_1 + Y^2 * M_2
```

where M_0, M_1, M_2 are constant matrices determined by physical parameters.
Therefore M(Y)^{-1} has entries that are **rational functions** of Y.

This means A_c(Y) and B_c(Y) have rational entries in Y, which is exactly
the type of dependence the LFR is designed to capture.

### Output equation

```
y = C * x + D * u
```

C and D are constant (no Y-dependence). In stage coordinates:
```
C = P^T * [I_3, 0_3]    (3x6 matrix)
D = 0                    (3x3 zero matrix)
```

---

## 4. Baseline LFR: converting physics to {G^b, Delta^b(Y)}

The baseline model must be expressed in LFR form (Drenth eq. 5.1):

```
G^b: { [x^b_{k+1}]   [A^b_x    B^b_w    B^b_u ] [x^b_k]
       [z^b_k     ] = [C^b_z    D^b_zw   D^b_zu] [w^b_k]
       [y_k       ]   [C^b_y    D^b_yw   D^b_yu] [u_k  ]

       w^b_k = Delta^b_x(p^b_k) * z^b_k
```

### The conversion problem (OPEN)

Converting the known physics A_c(Y) with rational Y-entries into an explicit
{G^b, Delta^b(Y)} form requires an **LFT realization** procedure. This is the
standard problem of expressing a rational matrix function as a linear fractional
transformation.

**What needs to happen**:
1. Factor out the Y-dependence from A_c(Y) into a structured Delta^b(Y) block
2. The remaining constant matrices form G^b
3. The repetition count eta^b is determined by the degree of the rational dependence

**Key references** (not yet obtained):
- Zhou, Doyle & Glover (1996), "Robust and Optimal Control", Chapter 10: LFT realization
- MATLAB Robust Control Toolbox: `lftdata` function
- Drenth does NOT cover this conversion (his baseline is assumed to already be in LFR form)

### Special structure: affine LFR case

If D_zw = 0 (no direct feedthrough in the latent path), the LFR reduces to an
affine LPV-SS model:

```
A(p) = A_x + B_w * Delta(p) * C_z        (affine in p via Delta)
```

For the gantry, the M(Y)^{-1} entries are truly rational (not polynomial/affine) in Y,
so D_zw != 0 in general. The rational LFR with eta > 1 is needed.

### Illustrative example: NL-MSD from Drenth Section 2.1.1

Drenth provides a worked example converting a nonlinear mass-spring-damper to LFR form.

The MSD ODE: `x_ddot = (1/m) * (u - k1*x - k2*x^3 - d1*x_dot - d2*x_dot*x)`

**Affine LFR** (2 scheduling variables p1=x, p2=x^2, eta=[1,1]):
```
G_aff = [ 0      1     0    0    0    0 ]    Delta_aff = [p1  0 ]
         [-k1/m  -d1/m  1    1    1/m  0 ]                [0   p2]
         [ 0     -d2/m  0    0    0    0 ]
         [-k2/m   0     0    0    0    0 ]
         [ 1      0     0    0    0    0 ]
```

**Rational LFR** (1 scheduling variable p1=x, eta=3):
```
G_rat = [ 0      1     0    0    0    0 ]    Delta_rat = [p1  0   0 ]
         [-k1/m  -d1/m -1    1    1/m  0 ]                [0   p1  0 ]
         [-k2/m   0     0    0    0    0 ]                [0   0   p1]
         [-k1/m   0     1    0    0    0 ]
         [ 0     -d2/m  0    0    0    0 ]
         [ 1      0     0    0    0    0 ]
```

The rational embedding uses only 1 scheduling variable (vs 2 for affine) but needs
eta=3 repetitions. This is the kind of trade-off we face for the gantry.

### Determining eta for the gantry (TODO)

The gantry's rational dependence comes from M(Y)^{-1}. Since M(Y) is degree-2 in Y,
the entries of M(Y)^{-1} = adj(M(Y)) / det(M(Y)) are ratios of degree-2 polynomials
over a degree-2 polynomial, giving rational functions of degree up to 2.

The required eta depends on the rational degree. From Drenth's MSD example:
- Degree-3 nonlinearity (k2*x^3) needed eta=3 with one scheduling variable.
- For degree-2 rational dependence, eta=2 or eta=3 is expected.

- [ ] Determine exact eta by performing the LFT realization on A_c(Y)
- [ ] Log as a design decision (D-019 or D-020)

---

## 5. Augmentation LFR: learned from data

The augmentation adds states, latent variables, and scheduling variables on top of
the baseline. Drenth thesis eq. 5.2 gives the combined structure:

```
     { [x^b_{k+1}]   [A^bb  A^ba  B^bb_w  B^ba_w  B^b_u ] [x^b_k ]
     { [x^a_{k+1}]   [A^ab  A^aa  D^ab_zw B^aa_w  B^a_u ] [x^a_k ]
G^a: { [z^b_k     ] = [C^bb  C^ba  D^bb_zw D^ba_zw D^b_zu] [w^b_k ]
     { [z^a_k     ]   [C^ab  C^aa  D^ab_zw D^aa_zw D^a_zu] [w^a_k ]
     { [y_k       ]   [C^b_y C^a_y D^b_yw  D^a_yw  D^b_yu] [u_k   ]

     [w^b_k]   [Delta^b_x(p^b_k)    0              ] [z^b_k]
     [w^a_k] = [0                    Delta^a_x(p^a_k)] [z^a_k]
```

Key points:
- The baseline and augmentation each have their own Delta block (block-diagonal)
- Cross terms (A^ab, A^ba, C^ab, etc.) couple the baseline and augmentation
- By setting subsets of the cross terms to zero, different augmentation structures
  are recovered (parallel, series, etc.)
- **Parallel augmentation** (our case, Drenth Section 5.2): the augmentation adds
  to x_{k+1} and y_k additively, matching the existing `SSE_Interconnect` architecture

### Augmentation well-posedness

The combined D_zw matrix:

```
D_zw = [D^bb_zw   D^ba_zw]
       [D^ab_zw   D^aa_zw]
```

For the augmentation LFR, D_zw is parameterized using the direct parameterization
(D_zw = exp(-N), N > 0) to guarantee rho(D_zw) < 1 by construction. This is applied
to the full combined D_zw, not to the baseline and augmentation separately.

### Augmentation scheduling map

The scheduling variable for the augmentation, p^a, is computed by a learned scheduling
map (Drenth eq. 2.24):

```
p^a_k = psi(x^a_k, u_k, d_k)
```

This is a ResNet with tanh activations and a saturation output layer to enforce
||p^a||_inf <= 1 (Assumption 2.2).

The baseline scheduling map psi^b is known from physics: for the gantry, p^b = Y = x[2].

---

## 6. Normalization (Drenth eq. 5.5)

The LFR matrices must be expressed in normalized coordinates matching training data
statistics. The normalization transforms for the baseline:

```
A_x_bar   = T_x * A^b_x * T_x^{-1}
B_w_bar   = T_x * B^b_w
B_u_bar   = T_x * B^b_u * T_u^{-1}
C_z_bar   = C^b_z * T_x^{-1}
D_zw_bar  = D^b_zw              (unchanged)
D_zu_bar  = D^b_zu * T_u^{-1}
C_y_bar   = T_y^{-1} * C^b_y * T_x^{-1}    (note: T_y^{-1}, not T_y)
D_yw_bar  = T_y^{-1} * D^b_yw
D_yu_bar  = T_y^{-1} * D^b_yu * T_u^{-1}
```

where T_x, T_u, T_y are diagonal scaling matrices derived from training data statistics
(inverse standard deviations).

**Note**: D_zw is unchanged by normalization. This means well-posedness conditions
(based on spectral radius of D_zw) are invariant under normalization.

The scheduling map is also scaled (Drenth eq. 5.6-5.7):
```
psi_bar^b(u_bar, x_bar, d_bar) = psi^b(T_u^{-1} * u_bar, T_x^{-1} * x_bar, T_d^{-1} * d_bar)
p_bar^b = sat(T_p * psi_bar^b)
```

---

## 7. Integration with CT + RK4 (D-018)

The LFR structure is defined in both CT and DT (Drenth gives CT in eq. 2.1, DT in
IFAC paper eq. 6). For the gantry:

- The **baseline** is a CT model integrated with RK4 at each timestep (D-018)
- The RK4 integration happens **inside** the state block
- From the interconnect's perspective, the block still maps (x_k, u_k) to x_{k+1}
- The LFR structure applies to the CT ODE: the scheduling block Delta^b(Y) enters
  the CT equations, and RK4 integrates the result

**Open question** (Blocker A in Task 3.3): Does the LFR structure need special treatment
during RK4 integration? Specifically, should Delta^b(Y) be re-evaluated at the
intermediate RK4 stages (k1, k2, k3, k4), or only at the beginning of the step?
Since Y changes within the RK4 sub-steps, Delta^b should be re-evaluated. This is
analogous to how the current implementation re-evaluates M(Y) at each sub-step.

---

## 8. Notation collision warning

Two different "M" matrices appear throughout this project:

| Symbol | Meaning | Context |
|--------|---------|---------|
| M(Y) | 3x3 inertia matrix from gantry Lagrangian | Physics, CT ODE |
| G (or M_lfr) | Constant interconnection matrix in the LFR | LFR structure, Drenth |

In Drenth's notation, the interconnection matrix is called G (not M). In the IFAC paper
eq. 6 it is called M. To avoid confusion with the inertia matrix, we use:
- **M(Y)** for the inertia matrix (always with the Y-argument)
- **G^b, G^a** for the baseline and augmented LFR interconnection matrices (Drenth notation)

---

## 9. Summary: what is known vs what is open

| Item | Status | Reference |
|------|--------|-----------|
| LFR definition and notation | Known | Drenth Ch. 2 |
| Well-posedness conditions | Known | Drenth Theorem 2.5, eq. 2.12 |
| Direct parameterization (D_zw = exp(-N)) | Known | Drenth eq. 2.12-2.13 |
| Baseline assumed in LFR form | Known (eq. 5.1) | Drenth Ch. 5 |
| Augmented LFR structure | Known (eq. 5.2) | Drenth Ch. 5 |
| Normalization formulas | Known (eq. 5.5) | Drenth Ch. 5 |
| **Baseline LFR realization** (physics to LFR) | **OPEN** | Need Zhou et al. (1996) Ch. 10 |
| **eta choice** for baseline Delta^b | **OPEN** | Depends on LFT realization |
| **RK4 + LFR interaction** | **OPEN** | No paper found yet (Blocker A) |
| M(Y) invertibility | **PROVEN** | `docs/m-matrix-invertibility.md` |
| Scheduling range ||Y|| <= 1 | Satisfied | Y in [-0.35, 0.35] |
