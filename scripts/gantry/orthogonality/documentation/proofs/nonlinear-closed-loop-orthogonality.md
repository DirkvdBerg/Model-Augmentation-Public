# Verification report: trajectory-level orthogonality on a nonlinear closed-loop model with an internal learned state

Date: 2026-09-07
Scripts: [`nonlinear_closed_loop_orthogonality.m`](../../code/nonlinear_closed_loop_orthogonality.m),
[`nonlinear_closed_loop_orthogonality.py`](../../code/nonlinear_closed_loop_orthogonality.py),
[`nonlinear_closed_loop_orthogonality_compare.py`](../../code/nonlinear_closed_loop_orthogonality_compare.py)
Results: [`matlab.json`](../../code/nonlinear_closed_loop_orthogonality/nonlinear_closed_loop_orthogonality_matlab.json),
[`sympy.json`](../../code/nonlinear_closed_loop_orthogonality/nonlinear_closed_loop_orthogonality_sympy.json),
[`compare.json`](../../code/nonlinear_closed_loop_orthogonality/nonlinear_closed_loop_orthogonality_compare.json)
Generated derivative functions: [`generated/`](../../code/nonlinear_closed_loop_orthogonality/generated/)

This report verifies the algebra of the extension proposed in
[the implementation plan](../plan/fixed-reference-rollout-orthogonality-implementation.md),
namely projecting the **complete augmentation-on/off rollout effect** against the
physical-parameter sensitivities of the **full predictor**, on a small nonlinear model that
actually has an internal learned state, a controller in the loop and a shared encoder.
It owns no theory. [gaps.md](../gaps.md) remains the live theory and evidence record.

## 0. Scope and claim boundary

**RULES.md Rule 1 category: Structural throughout.** Every quantity here needs the model
only, no data and no simulation truth, so all of it extends to Telica trivially, and none
of it is evidence about Telica or about the gantry. This is a mathematical reference
implementation, not an experiment.

Four levels are kept apart deliberately, and no result is quoted across the boundary:

| Level | What it means here | Where |
|-|-|-|
| 1. Exact symbolic identity | Proved to hold for all parameter values, over the rationals, under the stated real assumptions | Sections 3, 5, 6a, 7 |
| 2. Numerical derivative / implementation check | Analytic derivatives against central finite differences of the same simulator, and generated code against the symbolic reference | Section 8 |
| 3. Numerical feasibility of a hard-constrained example | One solver run on one 4-parameter problem | Section 9 |
| 4. Training or parameter-recovery guarantee | **Nothing here establishes any.** | Not claimed |

Explicitly **not** claimed, matching the plan's own boundary:

- A finite `beta` does **not** guarantee exact orthogonality. What is proved is the value
  bound in Section 5, which is a statement about the penalty value, not about the optimizer.
- Orthogonality of the **total** contribution does **not** imply that the latent path is
  separately orthogonal, nor that the learned tangent directions are separated. The total
  penalty can be zero while the projected direct and latent routes cancel each other; that
  is why both routes are logged separately (Section 4).
- The production method routes penalty gradients to the augmentation only. Section 6b proves
  this is **not** the gradient of any single scalar objective in this example. The
  hard-constrained `fmincon` experiment of Section 9 is a **different method**, reported
  separately, and no result from it transfers to the penalty.
- Two engines agreeing checks a finite model and a finite set of identities transcribed twice
  from the same written equations. Per [algebra-tooling.md](../algebra-tooling.md) rule 1 that
  is **not** verification of the gantry first-principles equations, and it is not claimed to be.

## 1. The model

Closed loop, exact rational coefficients, output computed **before** the state transition to
match production timing. Physical state `x`, learned state `a`, controller state `xc`,
so `chi = [x; a; xc]`.

```
y_k    = sig*x_k
e_k    = ydata_k - y_k
u_k    = udata_k + Cc*xc_k + Dc*e_k
x_k+1  = kap*x_k + nu*x_k^2 + u_k + ( Kk*a_k + Pp*u_k )
a_k+1  = Ff*x_k + Gg*a_k
xc_k+1 = Ac*xc_k + Bc*e_k
```

| Group | Symbols | Role |
|-|-|-|
| Physical `kappa` | `kap`, `nu`, `sig` | linear term, the nonlinearity, output gain (gives a direct output-parameter derivative) |
| Learned `eta` | `Kk`, `Ff`, `Gg`, `Pp` | latent readout, latent write, latent persistence, direct route |
| Encoder `xi` | `cc` | shared across windows: `x_0 = cc*hx_w`, `a_0 = cc*ha_w` |

Constants: `Ac = 4/5`, `Bc = 1/4`, `Cc = 3/10`, `Dc = 1/5`; reference
`kappa_0 = (1/2, 1/10, 6/5)`, `eta_0 = (1/5, 1/4, 2/5, 1/10)`, `xi_0 = 1`.
Two windows, four scored outputs each, so the scored stack has `m = 8` rows ordered by
(window, time). Loss metric `W = L'L` with `L` a nontrivial positive diagonal.

**Horizon.** Four scored outputs means three state transitions. That is the shortest horizon
that makes the latent state re-enter itself twice (`Gg^2` appears in `a_2`) and the controller
error feed back three times, so it is strictly beyond the two-step V1 toy in the plan.

**Interventions.** All three replay the same recorded `ydata`/`udata` and start from the same
encoder-derived physical state, and **each evolves its own controller state**, which is checked
rather than assumed (Section 4).

| Run | Physical correction | Latent state |
|-|-|-|
| full | `Kk*a_k + Pp*u_k` | evolves |
| clamped | `Pp*u_k` | clamped to the null value 0 at every step |
| off | none | absent |

**Assumptions on record.** All parameters real; `cs` (the latent rescale) nonzero; `br`
(`beta`) positive; slack and norm variables nonnegative. `IgnoreAnalyticConstraints` is left
`false`. At the frozen reference the encoder block `D_0 = L E_0` has rank 1, the unprofiled
`L S_0` has rank 3, and the profiled `Sbar` has **column rank 3**, so `Sbar` has full column
rank and every pseudoinverse below is used under exactly that rank condition.

**Numerical regime, stated narrowly.** The linearisation at the origin has spectral radius
`0.6583`, and the realised 200-step closed-loop trajectory stays bounded with
`max|state| = 0.60`. That is a finite-horizon regime check. It is **not** a stability proof
for the nonlinear closed loop, and no such claim is made from the coefficient values.

## 2. How zero and nonzero are decided

`isAlways(..., 'Unknown', 'error')` is used for every zero decision, so an undecidable
comparison can never be silently reported as "false" or as "proved": it is caught, entered in
an undecided ledger and surfaces as a **failed** check. **Both runs recorded zero undecided
identities.**

"Not identically zero" is a different obligation and needs a different instrument. A symbolic
sign test on a free symbol is undecidable, so non-vanishing is proved by **exhibiting an
explicit integer witness point** at which the expression evaluates to a nonzero rational. That
is a proof of non-vanishing as a function; three independent witness sets are tried.

For homogeneous linear constraints the null space is used, never `solve`, per
[algebra-tooling.md](../algebra-tooling.md) clause 6.

## 3. Sensitivities derived two independent ways (exact identity)

Local derivative blocks are formed once from the one-step map:
`Fstep`, `Fchi = d Fstep/d chi`, `Fth = d Fstep/d theta`, `Hchi`, `Hth`. They are then
propagated by the recurrence `T_{k+1} = Fchi_k T_k + Fth_k`, `dy_k/dtheta = Hchi_k T_k + Hth_k`,
initialised by differentiating the encoder. The alternative is direct differentiation of the
fully expanded stacked prediction. **Their difference simplifies identically to zero.**

| Assertion | Result |
|-|-|
| `Fchi` contains the latent readout, latent write, latent persistence, and both directions of controller feedback | proved nonzero |
| `Hth` is nonzero, so the direct output-parameter derivative is present | proved nonzero |
| Physical forcing enters at every step, not only the first | proved nonzero |
| Expanded rollout value equals the recurrence value (full and off) | identity |
| `d p_full/d theta` from direct differentiation equals the recurrence | **identity** |
| Same for the off rollout | **identity** |

## 4. Interventions: the decomposition, and why it proves nothing on its own

`d_total = p_full - p_off`, `d_lat = p_full - p_clamped`, `d_direct = p_clamped - p_off`, and
`d_lat + d_direct = d_total` is an identity.

**This identity is algebraically trivial.** It holds for any three returned vectors and
validates no simulator semantics. gaps.md Section 10 already records an earlier version of
this claim as FALSE for exactly that reason. The following independent interventions are what
actually check the simulator, and all are exact identities or proved non-vanishing:

| Assertion | Result |
|-|-|
| `p_off` does not depend on any learned parameter | identity (`d p_off/d eta = 0`) |
| `Kk = Pp = 0` makes the full run equal the off run | identity |
| `Kk = 0` makes the full run equal the clamped run | identity |
| `Pp = 0` makes the clamped run equal the off run | identity |
| The latent route `d_lat` is genuinely nonzero | witness |
| The direct route `d_direct` is genuinely nonzero | witness |
| All three interventions share the initial physical state (first scored row of each window agrees) | identity |
| The three interventions **evolve different controller states** | witness |
| The clamped run's latent state is null at every step | identity |

## 5. Projection geometry (exact identities)

With `D_0 = L E_0`, `M_E = I - Q_E Q_E'`, `Sbar = M_E L S_0`, `P_S = Sbar Sbar^+`, and
`Q_0` built by exact modified Gram-Schmidt (no generic symbolic SVD):

| Assertion | Result |
|-|-|
| `M_E` symmetric and idempotent | identity |
| `P_S` symmetric and idempotent | identity |
| Encoder and physical projectors orthogonal: `P_S P_E = 0` and `Q_E' Q_0 = 0` | identity |
| `Q_E' Q_E = I`, `Q_0' Q_0 = I` | identity |
| `Q_0 Q_0' = P_S` and `M_E Sbar = Sbar` | identity |
| `Q_0' M_E = Q_0'` (profiling is embedded in the basis, not omitted) | identity |
| `Sbar' M_E = Sbar'` | identity |
| **Zero projected residual iff contribution orthogonality:** `Sbar' = (Q_0' Sbar)' Q_0'` with `rank(Q_0' Sbar) = 3`, so `Q_0' L d = 0` and `Sbar' M_E L d = 0` have the same solution set | identity |

**Penalty value bound.** For `beta > 0`, writing `eps = beta t^2 + slack` with `slack >= 0`,
`sqrt(eps/beta) - t >= 0` was decided by `isAlways` with `Unknown = 'error'` in MATLAB, and in
SymPy by the exact identity `eps/beta - t^2 = slack/beta >= 0` plus monotonicity of `sqrt`.
So `beta ||Q' M_E L d||^2 <= eps` implies `||Q' M_E L d|| <= sqrt(eps/beta)`. This bounds the
residual **given** that the penalty value is small. It says nothing about whether training
attains a small value.

## 6. Penalty and gradients

### 6a. Universal linear-algebra argument (exact identities, generic `d`)

Proved as identities for a **generic** contribution vector `dg` and a **generic** smooth
parameter dependence `dgen(eg, kg)` that is nonlinear in `eg`. These hold for any smooth `d`,
so in particular for this example's `d_total`:

| Assertion | Result |
|-|-|
| `beta ||Q_0' L d||^2 = beta (L d)' P_S (L d)` | identity |
| `Q_0' L d = A_1 d_1 + A_2 d_2` over row chunks | identity |
| The basis-free rational residual `rho = P_S L d` carries the same penalty value | identity |
| `rho` splits over the same chunks | identity |
| `grad_eta V = 2 beta (d rho/d eta)' rho` | identity |
| The chunked gradient equals the stacked gradient | identity |
| The joint route has a **nonzero** physical gradient, so ANN-only routing is a different update rule | witness |

### 6b. This nonlinear example

The composite quantities are degree-16 polynomials in eight symbols; expanding them proves
nothing the universal identity does not already give, and the expansion cost is not evidence.
So the example is checked by **exact rational arithmetic at three independent integer witness
points**, which is a point check and is labelled as such, plus finite differences:

| Assertion | Level | Result |
|-|-|-|
| `beta ||rho||^2 = beta (Ld)' P_S (Ld)` | exact at 3 witness points | pass |
| `grad_eta V = 2 beta (d rho/d eta)' rho` | exact at 3 witness points | pass |
| `rho = rho_1 + rho_2` | exact at 3 witness points | pass |
| chunked gradient equals stacked gradient | exact at 3 witness points | pass |
| `grad_kappa V` and `grad_xi V` are **not** identically zero | witness | pass |
| `grad_eta V` against central finite differences of the same simulator | numerical | max abs `2.98e-11`, max rel `1.32e-10` |

**Gradient routing, interpretation corrected 2026-09-08.** The default routes only
`grad_eta V`. The nonzero omitted physical/encoder derivatives establish that this is not
the gradient of `J_base + V`. They do not establish failure of mixed-partial symmetry:
that stronger claim needs a nonzero mixed derivative involving augmentation parameters.
For example, `V = kappa^2 + eta^2` has a nonzero physical derivative but the selectively
routed field `(0, 2 eta)` is conservative. The recorded checks were not rerun or changed;
their stronger interpretation is withdrawn pending the additional check.

**Intermediate states must not be detached.** A negative control zeroes the
controller-state coupling inside the recurrence Jacobian while leaving everything else intact.
It changes the ANN gradient (proved nonzero difference) and, at the 200-step horizon, produces
a relative error of `0.27` against finite differences, so the omission is detected rather than
silently absorbed. The off rollout carries no learned-parameter path either way, which is what
licenses computing it without a graph.

**Cross-chunk cancellation.** A contribution is constructed with `r_1 = -r_2 != 0` using the
null space of the chunk operator. Then the stacked penalty `beta||sum_b r_b||^2` is exactly
zero while the per-chunk sum `beta sum_b ||r_b||^2` is nonzero with nonzero gradient. Chunking
is a memory device only; a sum of per-chunk penalties is a different objective.

## 7. Blind spot of the zero-latent pointwise penalty (exact)

The deployed pointwise penalty evaluates the correction on a frozen point set at zero latent
state, `c_eta(x_j, 0, u_j) = Pp*u_j`. Its derivative with respect to `Kk` is **identically
zero**, and with `Pp = 0` the residual is **identically zero for every `Kk`**.

On the same model with `Pp = 0`, the rollout residual `||Q_0' L d_total||` at the reference
`Kk = 1/5` is `0.14962150656204692` with derivative `0.7701936761201384` in `Kk`. The two
engines returned bit-identical doubles for both (relative difference exactly `0`); they differ
only in how many digits each JSON writer prints.

That is the explicit case asked for: the pointwise penalty is blind to the latent readout
while the rollout effect has a nonzero physical-sensitivity projection with nonzero gradient.

## 8. Latent-coordinate invariance (exact identities)

Apply `a' = c a` with `c != 0`, transforming the write, recurrence, readout and initial latent
state together: `Kk' = Kk/c`, `Ff' = c Ff`, `Gg' = Gg`, `a_0' = c a_0`.

| Assertion | Result |
|-|-|
| Predictions unchanged | identity |
| Off and clamped runs unchanged | identity |
| `d_total` unchanged | identity |
| The penalty unchanged | exact at witness points |
| The latent state scales exactly by `c` | identity |
| The latent state norm is **not** invariant | witness |
| `||Kk||` is **not** invariant | witness |

So neither the latent state norm nor the readout norm is a coordinate-invariant diagnostic,
consistent with the standing rule that the augmentation is judged at the **output**. Note the
scope: the zero clamp value survives **scaling**, but it would **not** survive an affine
translation of the latent coordinate, which would require transforming the null intervention too.

## 9. Code generation and longer closed-loop runs (numerical)

`matlabFunction` with explicit `Vars` and `File` generated ten step and local-derivative
functions into `generated/`. Longer trajectories use the sensitivity **recurrence** on these
functions, never an expanded symbolic trajectory.

| Quantity | Value |
|-|-|
| Generated derivatives vs the short symbolic reference | max abs `5.55e-17`, max rel `7.71e-17` |
| SymPy `lambdify` vs its own symbolic reference | max abs `1.11e-16`, max rel `1.54e-16` |
| 200-step, 3-window recurrence vs central finite differences (MATLAB) | max abs `8.00e-11`, max rel (column-scaled) `6.25e-10` |
| Same in SymPy/NumPy, different random signals | max abs `5.65e-11`, max rel `5.08e-10` |
| Negative control, controller coupling dropped | rel error `0.27` (MATLAB) / `0.29` (SymPy), detected |
| `cond(L S_0)` | `7.148383816457189` (both engines) |
| `cond(Sbar)` | `6.413265223666588` (both engines) |
| `cond` of the long-horizon physical block | `4.64` (MATLAB) / `4.71` (SymPy), different signals |

**`Optimize` benchmark, and the decision it produced.** `Optimize = true` cost `1.64 s` and
produced `4678` bytes; `Optimize = false` cost `0.78 s` and produced `4126` bytes. At this
size optimisation is **2.1x slower and yields larger files**, so it buys nothing here. Sparse
output was not requested either: `Fchi` is 3x3 with 6 of 9 entries structurally nonzero.

**Parallelism.** Benchmarked before adopting, as required, and rejected: the entire serial
multi-start solve costs `0.65 s`, while opening a parallel pool costs tens of seconds. The
Parallel Computing Toolbox is licensed and available; it is simply not warranted.

## 10. Separate hard-constraint experiment (numerical feasibility only)

**A different method from the penalty.** Minimize the weighted prediction loss over `eta` with
physical parameters and encoder frozen, subject to the hard equality `Q_0' M_E L d_total(eta) = 0`
(3 constraints, 4 unknowns). Analytic objective and constraint gradients supplied; the
constraint Jacobian is passed as parameters-by-constraints, the transpose of `jacobian(ceq, eta)`.
Both were validated against central differences: relative error `1.95e-11` (objective) and
`6.07e-11` (constraint Jacobian).

**The trap, and why the first attempt failed honestly.** `eta = 0` is feasible with zero
correction, and the constraint Jacobian there has **rank 2 of 3**: with `Kk = Pp = 0` the
latent state cannot reach the output at all, so the whole plane `{Kk = 0, Pp = 0}` is feasible
with `d = 0` and the null-space seed taken at `eta = 0` moves only along the inert `Ff`, `Gg`
directions. The first run therefore returned `eta* = 0`, which is exactly the outcome that must
**not** be accepted as evidence. It was recorded as a failed check, not written up as success.

The fix was to find genuinely nonzero feasible points first, by root-solving the homogeneous
constraint on fixed-`Kk` slices (three equations, three remaining unknowns), and to seed the
solver from those. Seven such points were found, all with constraint violation below `1e-10`
and correction norms from `0.033` to `0.494`.

| Result | MATLAB `fmincon` (sqp) | SciPy `SLSQP` |
|-|-|-|
| `eta*` | `[0.1395822, 0.1363778, 0.9933067, -0.2882060]` | `[0.1395822, 0.1363778, 0.9933067, -0.2882060]` |
| Objective | `0.17185718506003581` | `0.17185718506003572` |
| Objective at `eta = 0` (ANN off) | `0.19235971181250497` | same problem |
| Retained correction `||L d_total||` | `0.14353845055644182` | `0.14353845055643907` |
| Max abs constraint violation | `4.86e-17` | `7.03e-17` |
| Exit status | `1` (converged) | `0` (success) |
| Best seed | index 9, a root-solved feasible point | index 8, likewise |

So a **feasible nonzero correction exists and is retained at the constrained optimum**, which
beats the ANN-off objective. Two different solvers in two different libraries reached the same
point to eleven significant figures.

**What this is not.** One local solve on one 4-parameter problem. No global convergence, no
statement about the penalty method, and nothing about parameter recovery. The final output was
never projected after the rollout to manufacture feasibility; the constraint was imposed during
the solve and its residual is reported.

## 11. Independent-engine comparison

The SymPy file transcribes the model equations from scratch. It does not read, import or parse
anything the MATLAB script produces.

| | Value |
|-|-|
| MATLAB | `25.1.0.2973910 (R2025a) Update 1`, Symbolic Math Toolbox 25.1, Optimization Toolbox 25.1, Parallel Computing Toolbox 25.1 (all licensed and checked out) |
| SymPy | `sympy 1.14.0`, NumPy, SciPy |
| Checks: MATLAB / SymPy / shared | 64 / 62 / **60** |
| Shared checks in agreement | **60 of 60, all true** |
| Disagreements | **none** |
| Undecided identities | **none in either engine** |
| Shared numeric quantities out of tolerance | **none**; largest relative difference `3.33e-16` (spectral radius), the rest exactly `0` |

The four MATLAB-only and two SymPy-only checks are the toolchain-specific ones
(`fmincon` vs `SLSQP`, `matlabFunction` vs `lambdify`, toolbox availability).

**Timings, with engine startup excluded.**

| Phase | MATLAB | SymPy |
|-|-|-|
| Symbolic algebra | `30.75 s` | `194.38 s` |
| Code generation | `2.63 s` | included in numeric |
| Numerical | `1.16 s` | `68.06 s` |
| Optimization | `2.36 s` | `8.25 s` |
| **Total in-engine** | **`39.16 s`** | **`270.7 s`** |

MATLAB is roughly 7x faster on this workload. That is consistent with, and considerably
stronger than, the 2.5x recorded in [algebra-tooling.md](../algebra-tooling.md) Section 2, and
it was measured on substitution-and-expansion-heavy work rather than on `simplify`. MATLAB
startup under `-batch` (roughly 30 to 40 s) is excluded from both columns and is not part of
the comparison.

## 12. Engine failures encountered, for algebra-tooling.md clause 6

Three real failures were hit and resolved. All three are engine-behaviour issues, not algebra
disagreements; the two engines never disagreed on a result.

| # | Engine | Failure | Resolution |
|-|-|-|-|
| 1 | MATLAB R2025a | `syms` raised "Attempt to add ... to a static workspace": a function containing **nested functions** has a static workspace, so `syms` cannot inject variables | Declare every symbol explicitly with `sym('name','real')` |
| 2 | Both | Expression swell. The sensitivity recurrence built deeply nested unexpanded products of the growing state, and downstream `expand`/`simplify` on the composite ran for over 15 minutes without finishing | `expand` the state and the sensitivity matrix at **every** step of the recurrence, plus a syntactic-zero fast path in the zero test |
| 3 | **SymPy 1.14.0** | `OverflowError: 'mpz' too large to convert to float`, raised inside `sympy.ntheory.factor_._perfect_power` while sympy tried to factor a huge integer under a radical. Triggered twice: expanding a degree-16 expression containing `Q_0`'s surds, and calling `pinv()` on a surd matrix. MATLAB handled both without complaint | Work in the **basis-free rational** residual `rho = P_S L d` instead of `Q_0' L d` for everything downstream (proved equivalent in Section 6a), build the chunk-cancellation example in the rational chunk operators, and use small **integer** witness points rather than fractions, whose powers produce astronomically large numerators |

A fourth, purely performance, observation: SymPy's `xreplace` is dramatically faster than
`subs` for a fully numeric witness substitution. Replacing one with the other took the three
slowest example checks from `132 s`, `326 s` and `482 s` down to `75 s`, `9 s` and `43 s`.

**The general lesson, and it generalises past this report.** Surds are a liability in an
otherwise rational computation. An orthonormal basis is convenient for stating projection
geometry and expensive for computing with it, because normalisation introduces square roots
that block exact rational arithmetic and, in SymPy, reach code paths that overflow. Where a
projector suffices, use the projector: `P_S = Sbar Sbar^+` stays rational, carries the same
penalty value, and splits over chunks the same way. Keep `Q_0` for the statements that are
genuinely about an orthonormal basis.

## 13. What is proved, what is checked, what is open

**Proved as exact symbolic identities** (Level 1): the two independent derivations of the
sensitivities agree; the intervention identities; every projection-geometry identity including
`Q_0' M_E = Q_0'` and the zero-residual-iff-orthogonality equivalence; the penalty value bound;
the universal projector, chain-rule and exact-batching identities for a generic contribution
vector; the pointwise penalty's blindness to the latent readout; latent-coordinate invariance
of predictions, interventions and the penalty.

**Verified numerically** (Level 2): analytic gradients against central finite differences at
short and 200-step horizons; generated code against the symbolic reference to `1e-16`; the
detached-state and dropped-scheduling negative controls both fire.

**Shown feasible in one example** (Level 3): a hard equality constraint admits a nonzero
correction that is retained at the constrained optimum and beats the ANN-off objective, found
identically by two solvers.

**Open, and not addressed here.**

1. Everything above is a generic model. Nothing transfers to the gantry predictor until `S`,
   `R`, `E` and the penalty Jacobian are assembled for the actual predictor and its actual
   closed-loop loss, and their directional derivatives are checked against gantry rollouts.
2. Whether a finite `beta` selects usefully. `beta` remains a chosen constant with no
   justification until an L-curve is run (RULES.md, 2026-09-07 note).
3. Whether the constraint preserves correction that improves prediction. The Section 10 result
   is feasibility in a 4-parameter toy, not efficacy.
4. Rank and conditioning at gantry scale. Here `cond(Sbar) = 6.4` and the profiled rank is
   full; the gantry's raw physical rank deficiency is a different regime entirely.
5. The selective-gradient update is not the gradient of a scalar objective, so no convergence
   theory covers it. Section 6b proves the asymmetry; it does not repair it.

## 14. Reproduce

From the repository root:

```text
"C:\Program Files\MATLAB\R2025a\bin\matlab.exe" -batch "addpath('scripts/gantry/orthogonality/code'); nonlinear_closed_loop_orthogonality"
conda run -n GraduationProject python scripts/gantry/orthogonality/code/nonlinear_closed_loop_orthogonality.py
conda run -n GraduationProject python scripts/gantry/orthogonality/code/nonlinear_closed_loop_orthogonality_compare.py
```

The first two write their result JSONs and the generated MATLAB functions into
`scripts/gantry/orthogonality/code/nonlinear_closed_loop_orthogonality/`. The third reads only
those two JSONs and adds no verification of its own. Both runs are deterministic: the long
trajectories use fixed seeds (`rng(20260907,'twister')` and
`np.random.default_rng(20260907)`), which differ between engines by design, so the two
finite-difference errors are independent measurements rather than a repeat of one.
