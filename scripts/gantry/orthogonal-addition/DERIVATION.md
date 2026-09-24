# Derivation: additions the gantry baseline's parameters cannot represent

Every numbered equation carries its machine check in brackets. Symbolic checks: `derive/d_symbolic.py`
(`D1` to `D11`, R-011d, all OK). Numeric checks with the production Jacobian: `derive/d_numeric.py`
(`N1c`, `N2`, `N4`, `N5` in R-033c, `N6` in R-033d, all OK; the approximation checks `N1`, `N1b`
missed their pre-registered tolerances on two rows each and are reported as measured errors,
OA-022/023). Logs: `outputs/R-011d.log`, `outputs/R-033c.log`, `outputs/R-033d.log`. Notation as in `docs/writeup/` (`x_tilde` physical states, `x_bar`
augmentation states, `x` normalised, `f_base`) and the 21-09 uniqueness figures (`Phi` the stacked
parameter sensitivity, `v` the free parameter coordinate, `Delta` the added dynamics,
`rho = ||Phi Phi^+ Delta|| / ||Delta||`).

## 1. Baseline

Logical coordinates `q = [X, Theta, Y]`, frozen-`M` convention of `gantrySystem.m`:

```
(1)  M(theta, Y) qdd + C(theta) qd + K(theta) q = u                                        [D1]
     M = [ m_total + mh          m_diff Lb/2 - mh Y            0     ]
         [ m_diff Lb/2 - mh Y    J_eff + mh d^2 + mh Y^2       -mh d ]
         [ 0                     -mh d                         mh    ]
     C = [ cg1 + cg2             (cg1 - cg2) Lb/2              0     ]
         [ (cg1 - cg2) Lb/2      cb_sum + (cg1 + cg2) Lb^2/4   0     ]
         [ 0                     0                             cy    ]
     K = kb_sum e2 e2^T
```

`theta = (kb_sum, cg1, cg2, cy, cb_sum, mh, m_total, m_diff, J_eff, d)`, the ten identifiable
combinations; `v` are their free coordinates (log for nine, relative-linear for m_diff).

## 2. Sensitivity, per family

```
(2)  r_j = (dM/dtheta_j) qdd + (dC/dtheta_j) qd + (dK/dtheta_j) q                          [D2]
(3)  d qdd / d theta_j = -M^-1 r_j                                                        [D3]
```

Each combination enters exactly one matrix [D2]: mass `mh, m_total, m_diff, J_eff, d`, damping
`cg1, cg2, cy, cb_sum`, stiffness `kb_sum`. Written out [D2 prints them]:

```
     r_kb_sum  = [0, Theta, 0]                 r_cy      = [0, 0, Yd]
     r_cg1     = (Xd + Lb Thetad/2) [1,  Lb/2, 0]
     r_cg2     = (Xd - Lb Thetad/2) [1, -Lb/2, 0]
     r_cb_sum  = [0, Thetad, 0]                r_m_total = [Xdd, 0, 0]
     r_mh      = [Xdd - Y Thetadd,  -Y Xdd + (d^2 + Y^2) Thetadd - d Ydd,  Ydd - d Thetadd]
     r_m_diff  = (Lb/2) [Thetadd, Xdd, 0]      r_J_eff   = [0, Thetadd, 0]
     r_d       = mh [0, 2 d Thetadd - Ydd, -Thetadd]
```

## 3. The one-step sensitivity and the condition

The production model steps (1) with RK4 over `Ts = 2.5e-4 s`, input held, in normalised
coordinates. Its sensitivity is the derivative of this map, written out:

```
(4'') x+ = x + Ts/6 (k1 + 2 k2 + 2 k3 + k4),   k_i = F(x_i, u),   x_1 = x, x_2 = x + Ts k1/2,
      x_3 = x + Ts k2/2, x_4 = x + Ts k3,     F(x, u) = [qd ; M(Y)^-1 (u - C qd - K q)]
      with M re-evaluated at each stage's Y.   Phi_j = S^-1 d x+ / d theta_j (d theta_j / d v_j)   [N1c]
```

N1c is an independent numpy implementation of (4''), differentiated by central differences and
contracted with the addition columns: it reproduces the production coefficients on all ten
conditions to <= 1.5e-08 (R-033c). An addition `phi` (a generalised force, truth
`M qdd + C qd + K q = u + phi`) enters through the same map, `Delta_phi = G phi` with `G[k]` the
exact response to a held unit force (the map is affine in `u`) [N4, 4.5e-07].

Leading order, the explicit continuous form of the inner product is

```
(4)  Phi_j^T Delta_phi = -Ts^2 (d theta_j/d v_j) SUM_k r_j[k]^T W[k] phi[k],
     W = M^-T S_v^-2 M^-1   (Euler step, velocity rows)                                   [D4]
```

REMARK, measured accuracy of (4) against (4''): within 3.6 % on eight conditions, 5.6 % on mh and
43 % on cb_sum, whose leading term is small (yaw velocity) while the O(Ts^2) RK4 term is not (N1,
pre-registered 5 %, FAIL on those two rows, OA-022). The frozen-Y RK4 polynomial
`SUM (Ts A)^n / n!` is within 7.7e-04 on eight rows and misses 1e-3 on cy (5.2e-03) and cb_sum
(1.5e-03), because Y moves inside the step (N1b, OA-023). The design therefore always uses (4'').

Györök et al. (2026) Eq. (16) and Condition 4, Eq. (17), for the gantry:

```
(16) INTEGRAL over the domain of  r_j(x)^T W(Y) phi(x) = 0,     j = 1..10                   [D10 for route A]
(17) SUM_k Phi_j[k]^T Delta_phi[k] = 0,                          j = 1..10                   [N2]
```

Under (17) the OBC estimate is exact (Theorem 7): `v_hat - v* = Phi^+ Delta* = 0`
(`derive_condition.py` R1, R2 in the prior folder: `dtheta = Phi^+ Delta*` under least squares
whatever the network can represent).

## 4. The addition class is a null space

With `phi = SUM_c a_c phi_c` (known unit force fields, unknown coefficients):

```
(5)  SUM_c A_jc a_c = 0,   j = 1..10,     A_jc = Phi_j^T (G phi_c)                         [D9, N2]
```

linear and homogeneous in `a` [D9], so the admissible class is `null(A)`; a nonzero member exists
once the number of columns exceeds `rank A <= 10`. For an absorber, `phi_c` depends on its
frequency and direction, so `A` is nonlinear in those and linear in the mass.

**Legitimacy (OA-006).** A stateless column may only occupy a structural zero of (1): an (entry,
Y-power) slot zero for every `theta`. The baseline's own slots are exactly `M11:1, M12:1,Y,
M22:1,Y^2, M23:1, M33:1, C11, C12, C22, C33, K22:1` [D8]; anything else would be a parameter
change relabelled as an addition.

**Physical terms** (on structural zeros [D8]):

```
(6)  k_x   : phi = -k_x X e1                                   X spring at the beam centre  [D8]
(7)  k_y   : phi = -k_y Y e3                                   Y spring, payload neutral point [D8]
(8)  k_xp  : phi = -k_xp b (b^T q),  b = [1, -Y, 0]            X spring at the payload       [D7]
(9)  gamma : M13 = mh gamma,  M22 += -2 mh d gamma Y           skew of the Y guide           [D5]
(10) absorber: phi = -m b d'',  d'' + w_a^2 d = -b^T qdd,  b = [cos a, l - Y cos a, sin a],
     mass-conserving (only the motion relative to the carrier is added)                     [D6]
(14) at frequency w the absorber is an apparent mass along b:
     phi = -m_app(w) b b^T qdd,  m_app = m w^2 / (w_a^2 - w^2)                               [D11]
```

## 5. Routes, counting, and the Györök level each reaches

Coefficients are the discrete ones (4''), on the null control `augmentation_oa_null_b140-230_a6`
(baseline truth; the production dataset's records, references, multisine and controller).

**Route A, parity (Eq. 16 level).** The regressors (2) are odd in `(q - q_op, qd, qdd, u)` at frozen
`Y`; an even addition integrates to zero against every `r_j` over any sign-symmetric domain [D10].
The data realise it when every tuple has its mirror at the same `Y`, which only standstill records
allow. Counting: nothing to solve. Fails on size: on standstill records every even term is second
order in the 5.4 um vibration (100 % force-constant gradient: 5.0e-04 of the size minimum; Coriolis
5.4e-04; R-008).

**Route B, stateless on structural zeros (Eq. 17 level).** 27 legitimate slots up to `Y^2`
(23 once the inertial slots are capped at the EOM's `Y^1`, R-009b). `rank A = 10`, so
`dim null(A) = 13` for 23 columns [N2]. The four physical terms alone are 10 x 4 of full column
rank (best rho* 4.5e-02). The chosen member (OA-009: most learnable discrepancy per newton of stage
force) is in sect. 6.

**Route C, lossless absorbers (Eq. 17 level).** One absorber: its mass scales all ten coefficients
together, three knobs against ten equations, not orthogonal in general. Four absorbers plus the
four physical mount terms: 8 linear unknowns, so the geometry must drop `rank A` to 7 (three scalar
conditions on 12 numbers); the search reached rho* 6.2e-04 but with a negative mass and a
negative mount spring (R-010). One physical absorber (the production absorber made lossless and
centred) plus an exactly solved compensator: exact on its own data (residual 1.4e-12, R-015), but
the compensator must cancel an inertia-like projection of 1.86 and structural zeros are far from
`range(Phi)`, so it dominates (91 to 99.6 % of `||Delta||`, 250 to 3000 N rms, R-015, R-026).

## 6. The ten conditions for the route B member, explicitly

The member (R-009b, `design/addition_route_b_R-009b_null_s1.mat`), `phi = -(M_a(Y) qdd + K_a(Y) q)`,
units kg, kg/m, N/m, N/m^2, N/m^3 by Y-power:

| slot | coefficient | slot | coefficient | slot | coefficient |
|-|-|-|-|-|-|
| M13 | +1.584e-02 | K11 | +2.418e+01 | K11 Y | -1.164e+02 |
| M11 Y | -4.205e-01 | K12 | +2.671e+01 | K12 Y | -2.689e+01 |
| M13 Y | +1.289e-01 | K13 | -3.002e+00 | K13 Y | +2.518e+01 |
| M22 Y | -1.971e-01 | K23 | -4.842e+01 | K22 Y | -3.188e+04 |
| M23 Y | +2.711e-01 | K33 | -1.640e+01 | K23 Y | -1.288e+02 |
| M33 Y | -2.718e-02 | K11 Y^2 | -3.607e+02 | K33 Y | -4.849e+01 |
| | | K12 Y^2 | -1.928e+03 | K13 Y^2 | -1.342e+02 |
| | | K22 Y^2 | +9.111e+03 | K23 Y^2 | +8.781e+02 |
| | | K33 Y^2 | +3.258e+02 | | |

(`M_a`, `K_a` symmetric; an off-diagonal entry appears in both places.) Stage force on the data
(11.5, 13.0, 3.5) N rms, peak 80 N; `||Delta||` 12.735; rho* 2.2e-15 (R-009b).

The ten conditions, in the addition's coefficients `a_s` with `phi_s = -E_s v_s Y^p_s`
(`v = qdd` for M slots, `q` for K slots, `E_s` the symmetric unit matrix of the slot), explicit at
leading order by (4) and exact by (4''):

```
kb_sum : SUM_s a_s SUM_k  Theta [W E_s v_s]_2 Y^p                                        = 0
cg1    : SUM_s a_s SUM_k (Xd + Lb Thetad/2) ([W E_s v_s]_1 + (Lb/2) [W E_s v_s]_2) Y^p   = 0
cg2    : SUM_s a_s SUM_k (Xd - Lb Thetad/2) ([W E_s v_s]_1 - (Lb/2) [W E_s v_s]_2) Y^p   = 0
cy     : SUM_s a_s SUM_k  Yd [W E_s v_s]_3 Y^p                                           = 0
cb_sum : SUM_s a_s SUM_k  Thetad [W E_s v_s]_2 Y^p                                       = 0
mh     : SUM_s a_s SUM_k  r_mh^T W E_s v_s Y^p                                           = 0
m_total: SUM_s a_s SUM_k  Xdd [W E_s v_s]_1 Y^p                                          = 0
m_diff : SUM_s a_s SUM_k (Lb/2)(Thetadd [W E_s v_s]_1 + Xdd [W E_s v_s]_2) Y^p           = 0
J_eff  : SUM_s a_s SUM_k  Thetadd [W E_s v_s]_2 Y^p                                      = 0
d      : SUM_s a_s SUM_k  mh ((2 d Thetadd - Ydd) [W E_s v_s]_2 - Thetadd [W E_s v_s]_3) Y^p = 0
```

Each holds on the design data to round-off with the exact coefficients `A_js = Phi_j^T G phi_s`
(N2, one check per condition, R-033c; relative residual 7.3e-16 to 6.3e-14). The per-term values
`A_js a_s` and their row sums are in `outputs/R-033c_table.md`.

## 7. On closed-loop data the conditions are quadratic

Everything above evaluates (17) on data WITHOUT the addition. On data generated WITH it, the loop
(100 Hz bandwidth) cancels the low-frequency part of `phi`: the record keeps (almost) the same `x`
and carries `u_new = u - phi`. Because (4'') is affine in `u`:

```
(11) Delta*_new = x_next - f_base(x, u - phi) = Delta*_0 + G phi                   [N4; R-019]
(12) Phi_new = Phi(x, u - phi) = Phi_0 - SUM_s z_s Phi_s,
     Phi_s = Phi(x, u) - Phi(x, u - phi_s/||Delta_s||)                              [N5, 2.8e-08]
(13) F(z) = Phi_new^T Delta*_new - Phi_new^T Phi_new b_null
          = c0 + L_t z - SUM_st z_s z_t (Phi_s^T Dhat_t) = 0,   c0 = 0 exactly      [N5, N6]
```

(`c0 = Phi_0^T (Delta*_0 - Phi_0 b_null) = 0` because `b_null = Phi_0^+ Delta*_0` [N6]. The solves of (13) are R-021 (`|F|` 2.4e-15) and R-031 (continuation).)
The quadratic term is the addition's force seen twice: in `Delta`, and in the acceleration
`M^-1 (u - C qd - K q)` at which the mass-family sensitivity is formed. A design of the linear
part alone leaves it: predicted from the null tuples 11.71 % max shift, m_diff -12.75 % (R-020),
measured on the regenerated data 10.93 % and -12.79 % (R-019). Its size dependence is exactly
quadratic: 0.069 % at `||Delta||` 1, 1.11 % at 4, 5.07 % at 8.49 (R-029).

Solving (13) exactly: at `||Delta||` 12.735 with the full library `|F|` 2.4e-15, but the solver used
kg-level inertial slots whose in-band force the loop does not cancel, the model of (11)-(12) does
not hold for them, and the regenerated data miss by 16 % (R-021, R-023). With stiffness slots only,
where the model holds, the solution branch exists at `||Delta||` 1 (`|F|` 1.8e-14) and ends
between 1 and 2 (R-031). Regenerated at `||Delta||` 1.09, the linear member meets the band
(max shift 0.066 %, predicted 0.076 %; m_diff -0.070 %, predicted -0.088 %; R-032).

## 8. What is established, and at which level

| claim | level | evidence |
|-|-|-|
| the admissible class is `null(A)`, `A` explicit | exact, per dataset (Eq. 17) | D9, N2, N1c |
| route A additions are orthogonal on sign-symmetric domains | exact (Eq. 16) | D10 |
| route B members exist with rho* at round-off on data without them | exact (Eq. 17) | R-009b, N2 |
| on data WITH them the condition is (13), quadratic | measured | R-019 vs R-020, N5 |
| low-frequency stateless additions are orthogonal on their own closed-loop data only up to `||Delta*|| ~ 1` to 2 | measured (model validated to 7 % of the effect) | R-029, R-031, R-032 |
| no stiffness-only member at `||Delta||` >= 8.49 solves (13) | found, not proved | R-024, R-031 |
