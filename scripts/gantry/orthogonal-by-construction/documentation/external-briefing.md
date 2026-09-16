# Briefing: extending orthogonal-by-construction augmentation to a state equation with additional states

Self-contained. Paste whole. You have no access to the repository, and nothing here requires it.
Every equation is written in plain text. Where a statement is proven, it says so and names the
check. Where it is open, it says so. Please do not re-derive the settled items; the value you can
add is in Section 8.

---

## 1. What this is about

Master's thesis, TU Eindhoven, control systems. A physics-based model of a dual-gantry
high-precision motion system is augmented with a learned component. The goal is accuracy from the
learned part WITHOUT losing the physical meaning of the baseline's parameters.

The problem being solved is a specific degeneracy. In a plain additive augmentation, the physics
model and the network are fitted jointly, and the split between them is not identifiable: many
pairs of (physical parameters, network parameters) give the same predictor. The fit is identical,
the physical parameters are arbitrary, and they drift to unphysical values. Gyorok's Example 2 is
the minimal case: a linear baseline plus a single linear layer, where every pair whose sum equals
the truth is a global minimum.

The published fix is orthogonality between the two components. This briefing is about extending
that fix to our setting, which differs from the published one in three ways that all matter.

---

## 2. The published method (Gyorok, Schoukens, Peni, Toth, IFAC J. Systems and Control 35:100376, 2026; arXiv 2511.01321)

Model class: discrete-time input-output (NARX). The regressor argument `x_k` is a vector of
lagged MEASURED inputs and outputs. The learning component is static.

```
baseline        y_hat_k = phi(x_k) theta_b                        (linear in theta_b)
additive        y_hat_k = phi(x_k) theta_b + f_ANN(x_k; theta_a)
stacked         Y_hat   = Phi theta_b + F_ANN                     Phi = col_k phi(x_k)
```

The construction (their Eqs. 8, 10, 11; Lemma 3):

```
theta_aux = (Phi' Phi)^-1 Phi' F_ANN
F_tilde   = F_ANN - Phi theta_aux = [I - Phi (Phi'Phi)^-1 Phi'] F_ANN
Phi' F_tilde = 0                    exactly, on the estimation set
```

So the learned component is REPARAMETERISED, not penalised. There is no trade-off weight. Their
stated motivation is exactly that: with regularisation "there exists a trade-off between model
accuracy and the desired complementarity. Finding the appropriate trade-off parameter may not be
intuitive."

At prediction time the coefficient is FROZEN and the correction applied at the current point:
`y_hat = phi(x) theta_b_hat + f_ANN(x) - phi(x) theta_aux_hat` (their Eq. 13). Their released
code does exactly this, and uses a pseudo-inverse throughout.

### Five properties of the source that decide everything downstream

1. The inner product is Euclidean on the STACKED object over the evaluation set. It is `n_theta`
   scalar conditions, each a sum over samples and channels. There is no per-sample condition
   anywhere in the paper.
2. `theta_aux` is a least-squares coefficient, so the subtracted correction lies in `range(Phi)`.
3. The projector does not depend on the network parameters. `Phi` is built from data before
   optimisation.
4. The paper explicitly permits an AUXILIARY evaluation set: immediately after Lemma 3 it states
   `theta_aux` may be built from any auxiliary evaluation of the regressor, on a synthetic set or
   a subset of the estimation data, and that using the whole training set is a convenience.
5. Their Theorem 18 (zero covariance between baseline and network parameter estimates) is the
   practical payoff. Its proof forms `psi_k = [phi(x_k), J_f(x_k)]` with
   `J_f = d f_tilde / d theta_a`, and needs only property 3. It does NOT need exactness against a
   regressor recomputed at the current iterate.

### Two different orthogonalities in the paper, which must not be conflated

```
Phi' F_tilde = 0     a property of the PARAMETERISATION. You enforce it. Always available.
Phi' Delta   = 0     their Condition 4. A property of the PLANT and the EXPERIMENT, where
                     Delta is the true missing dynamics. You cannot enforce it, and outside
                     simulation you cannot even verify it, because Delta is unknown.
```

Only the second yields true-parameter recovery (their Theorem 7). Without it you get their Eq. 21:
the baseline estimate is UNIQUE and its error is the projection of the missing dynamics, plus a
comparative statement that this error is smaller than under standard additive augmentation. So the
honest claim from this method is uniqueness and interpretability, not correctness.

### What the paper's own conclusion says is open

Extension to state-space baselines, where "the general assumption of no available full-state
measurement complicates the orthogonal projection of the learning component, thus requiring
careful investigation." That is precisely our setting. A companion preprint by the same group
defines augmentation states and then sets their number to zero in both case studies, so the group
has never demonstrated this machinery with augmentation states switched on.

---

## 3. Our setting, and the three differences

Plant: dual-gantry motion system. Continuous-time LPV model in linear-fractional representation,
discretised. Equations of motion

```
M(Y) qddot + C qdot + K q = P u,      M(Y) = M0 + M1 Y + M2 Y^2,   Y = q3
```

so the scheduling variable IS one of the states (self-scheduling). `q` is a 3-vector of logical
coordinates, the state is `x_p = [q; qdot]` in R^6, input `u` in R^3, output `y` in R^3.
14 physical parameters: two bearing stiffnesses, two gantry dampings, one Y damping, two bearing
dampings, payload mass, two gantry masses, cross-arm mass, two inertias, one offset distance.

The augmentation is a feed-forward network reading the FULL state and the input, writing into
routed rows. It carries `n_a = 2` additional (latent) states of its own. The initial state comes
from a SUBNET-style encoder over past inputs and outputs, because the states are not measured.
Training is free-run simulation with multiple shooting, not one-step prediction.

Three differences from the source, each with consequences:

- **State-space, not input-output.** Could we convert? In principle yes, but eliminating the
  states mixes all 14 physical parameters nonlinearly into each coefficient of the resulting
  regressor. The coefficients would no longer be masses, stiffnesses and dampings, so we would be
  protecting the interpretability of quantities that are no longer physical. That defeats the
  purpose. Hence: impose the condition on the state equation instead. This is also the
  supervisors' explicit instruction.
- **Baseline nonlinear in the parameters.** They enter through `M(Y)^-1`, so no exact `phi`
  exists. See Section 5.1.
- **The learning component carries state.** The source's does not. This is the substance of the
  extension.

---

## 4. The structure

`R_p` and `R_a` are the physical and latent row blocks of the routing matrix, `E_p` selects the
physical rows, and `E_p' R = R_p`.

```
x_p(k+1) = f_theta(x_p(k), u(k))  +  R_p g_eta(x_p(k), x_a(k), u(k))
x_a(k+1) =                           R_a g_eta(x_p(k), x_a(k), u(k))
y(k)     = C x_p(k) + D u(k)
x(0)     = e_rho(past y, past u)
```

Four observations, each used later:

1. The second row has NO baseline term. The baseline defines no equation for the additional
   states.
2. `g_eta` appears in both rows with the SAME parameters. It is one network split by routing, not
   two independent functions. This coupling matters in Section 6.
3. The output reads `x_p` only. All influence of `x_a` on anything observable is routed through
   `R_p g_eta` at a later step.
4. `rho` appears in no row the condition will touch.

---

## 5. The condition

### 5.1 The regressor, with an offset column

Expand at a fixed `theta_bar` with the evaluation pair held fixed (the device the predecessor
2025 L4DC paper uses for nonlinear parameter dependence):

```
Phi_p(x_p,u) = d f_theta(x_p,u) / d theta   at theta_bar        (6 x 14)
c(x_p,u)     = f_theta_bar(x_p,u) - Phi_p(x_p,u) theta_bar      (6 x 1)
Phi_tilde_p  = [ Phi_p   c ]                                    (6 x 15)
```

The offset column is NOT cosmetic. The supervisors' condition is that the learned output be
orthogonal to the OUTPUT OF THE PHYSICS FUNCTION, not to its parameter variations. For a baseline
linear in its parameters those coincide, which is why the source can use the regressor directly.
Ours is rational in the parameters, so they differ, and the faithful object is the AFFINE set
including the nominal transition. A contribution that cancels the baseline's nominal response is
charged under `Phi_tilde_p` and is NOT charged under `Phi_p` alone.

VERIFIED symbolically on the real equations: the offset column is nonzero and lies OUTSIDE the
span of the 14 parameter columns, so appending it strictly enlarges the protected set.

### 5.2 The condition reduces to the physical rows by itself

Stack the regressor of the augmented map and the learned write over an evaluation set:

```
Psi   = col_i E_p Phi_tilde_p(i)           W_eta = col_i R g_eta(i)
Psi' W_eta = sum_i Phi_tilde_p(i)' E_p' R g_eta(i) = sum_i Phi_tilde_p(i)' R_p g_eta(i)
```

So the inner product does not involve `R_a g_eta` at all, and any correction `Psi lambda` writes
into the physical rows only. This is NOT a modelling choice to restrict the condition; the
structure does it, because the baseline block is identically zero on the latent rows. Orthogonality
against a zero basis is vacuous: anything is orthogonal to zero.

VERIFIED symbolically on the real equations: the baseline parameter Jacobian of the augmented map
is exactly zero on the additional-state rows, as an identity in the state and the input.

Accordingly, with `Phi_Z = col_i Phi_tilde_p(i)` and `G_eta = col_i R_p g_eta(i)`:

```
CONDITION:   Phi_Z' ( G_eta - Phi_Z eta_aux ) = 0,   eta_aux solving Phi_Z' Phi_Z eta_aux = Phi_Z' G_eta
```

### 5.3 A per-step condition is unusable

At one step, the latent block is unconstrained (zero basis) and the physical block is over-
determined: `Phi_tilde_p` has 15 columns against 6 rows, so it generically spans R^6, its
orthogonal complement is {0}, and a per-step projection would force `R_p g_eta = 0` identically.
The condition can only be a condition over a SET of points. This is why the source stacks.

### 5.4 Rank deficiency costs uniqueness, not orthogonality

Any solution of the normal equations gives `Phi' (G - Phi lambda) = 0`, at any rank, because
`Phi' G - Phi' Phi lambda = 0` is the normal equation itself. Solutions differ by null-space
vectors, which `Phi` maps to zero, so the subtracted correction is identical. The pseudo-inverse
picks the minimum-norm one; it does not repair anything.

What IS lost with rank deficiency is the source's Assumption 1 (full column rank), which its
Theorem 7 and Eq. 21 need for a unique baseline estimate.

VERIFIED symbolically on the real equations: rank 10 of 14 parameter columns, 11 of 15 with the
offset column appended. The null space has four directions:

```
kb1 - kb2                                      (bearing stiffness split)
cb1 - cb2                                      (bearing damping split)
two directions spanning the mass/inertia block, whose difference gives Jb - Jh and whose
shared combination is  Jb + Jh + (Lb^2/4)(m1 + m2),  i.e. the mass-matrix entry they occupy
together
```

This INDEPENDENTLY CONFIRMS the ten identifiable combinations already used in the project's
baseline parameter recovery, which reports the effective inertia rather than the inertia sum. It
is not a new finding.

---

## 6. The construction, and the two failures we proved

### 6.1 Why the coefficient cannot be fitted along the trajectory

If the evaluation set is the training rollout, the points depend on `eta`; worse, a coefficient
fitted over a window depends on the learned writes at LATER steps in that window, which depend on
the coefficient. The correction at step k becomes a function of the future. There is no exact fix
inside one forward pass. The source never meets this because its learning component is static.

### 6.2 The construction

Fix a point set `Z_ref` BEFORE the pass, build `Phi_ref` and a fixed left inverse `K_ref`, both
constant in `eta` and `theta`. Then:

```
eta_aux(eta) = K_ref col_i R_p g_eta( x_ref(i), u_ref(i) )            once per pass

x_p(k+1) = f_theta(x_p(k),u(k)) + R_p g_eta(x_p(k),x_a(k),u(k))
                                - Phi_tilde_p(x_p(k),u(k)) eta_aux(eta)
x_a(k+1) =                        R_a g_eta(x_p(k),x_a(k),u(k))
y(k)     = C x_p(k) + D u(k)
```

Causal, differentiable in `eta`, no trade-off weight. Two properties, both exact on `Z_ref`:

```
Phi_ref' ( G_eta - Phi_ref eta_aux ) = 0                        (orthogonality)
d/deta [ G_eta - Phi_ref eta_aux ] = (I - P_ref) dG_eta/deta    (so Phi_ref' J = 0)
```

The second is the requirement of Remark 5 in Section 2, i.e. what the block-Jacobian result needs.
Note: differentiating THROUGH `eta_aux` is mandatory. Stopping the gradient there destroys the
property the construction exists to provide.

Along the actual training trajectory neither holds. The residual there is an empirical quantity of
the run, not a bound. The honest claim is: orthogonal by construction with respect to the frozen
reference set, approximately orthogonal elsewhere by an amount that must be measured.

### 6.3 FAILURE ONE: the obvious reference set makes the construction blind to the latent path

The natural choice of `Z_ref` is the closed-loop rollout of the BASELINE ALONE, network gated off.
In that rollout nothing writes the latent states, so `x_a = 0` at every point.

PROVEN: if `x_a(i) = 0` for all i in `Z_ref`, then for every parameter of `g_eta` acting only on
the `x_a` argument or only on the `R_a` output block,

```
d eta_aux / d eta_latent = 0    and    d G_eta / d eta_latent = 0
```

because those parameters are multiplied by zero or annihilated by `R_p`. Yet the same parameters
DO change the learned physical write at any trajectory point with `x_a(k) != 0`.

VERIFIED symbolically: all six latent-path parameters of a minimal surrogate give an exactly zero
derivative of the fitted coefficient, while the physical write has derivative `x_a(k)` with respect
to the latent-to-physical weight.

Consequence: the projection charges nothing on exactly the feature this extension introduces. This
is a failure the input-output case cannot exhibit, and it is the reason the reference set is a
decision rather than an implementation detail.

Candidate repairs:
- (R1) Draw the latent coordinates over a designed region. Causal, independent of `eta`. Cost: the
  points leave the data manifold.
- (R2) Take `Z_ref` from the AUGMENTED rollout of the PREVIOUS epoch, refreshed between epochs.
  Within a pass the set is fixed, so both properties of 6.2 hold exactly, and the latent states are
  live so the blindness does not arise. Cost: the set depends on `eta` across epochs, making the
  scheme a fixed-point iteration whose convergence is NOT established. The predecessor 2025 paper's
  Remark 2 contemplates exactly this per-epoch refresh for its projection matrix.
- (R3) Accept and declare, claiming only non-overlap at zero latent state.

Current choice: (R2), with (R1) as an independent check.

### 6.4 FAILURE TWO: one step does not control the horizon

This one SURVIVES the repair of 6.3. Along a rollout, with `A(k)` the state Jacobian of the closed
augmented map:

```
S(k+1) = A(k) S(k) + E_p Phi_p(k)            dy(k)/dtheta = C E_p' S(k)
T(k+1) = A(k) T(k) + R dg_tilde/deta         dy(k)/deta   = C E_p' T(k)
```

The condition holds in AGGREGATE over the evaluation set: `sum_k Phi_k' d_k = 0`. Individual steps
can carry large aligned components that cancel against each other in that sum. The rollout then
weights each step differently, by the products of state Jacobians from k to the end, not once each.
Aggregate orthogonality under one weighting is not aggregate orthogonality under another.
Additionally `C` retains 3 of the 6 physical rows, and orthogonality in R^6 is not preserved by a
projection.

VERIFIED symbolically by counterexample: on a minimal surrogate with a live latent path, the
learned write is identically zero on the reference set, so the one-step condition holds EXACTLY and
trivially; yet the six-step output contribution is nonzero and its inner product with the baseline
output parameter sensitivity is a nonzero rational function of the weights.

Consequence for claim wording: the guarantee is ONE-STEP non-overlap. It removes the Example 2 flat
direction at the level of the one-step augmented map. It does NOT prevent the augmentation from
reproducing or negating baseline dynamics over a horizon, and no claim may say otherwise.

---

## 7. Reduction to the published result, and status

Set `n_a = 0`; route all physical rows so `R_p = I`; make the baseline linear in `theta` so `c = 0`
and `Phi_tilde_p = Phi_p`; take `C = I`, `D = 0` with a one-step predictor and measured states in
place of the encoder; take `Z_ref` to be the estimation set. Then the model becomes the source's
stacked Eq. 5 and the construction becomes its Eqs. 8, 10, 11, 13 term for term, with the inverse
replaced by any normal-equations solution. VERIFIED symbolically.

The THIRD reduction (one-step predictor with `C = I`) is the level gap and is named rather than
hidden: without it the source's stacked object is the measured output and ours is the state update,
and those are not the same object.

Assumptions of the source that our setting breaks, each with a verdict:

| Assumption | Verdict |
|-|-|
| Baseline linear in parameters | WEAKENS to orthogonality against the affine tangent set at `theta_bar`. Theorem 7 does not transfer. |
| Regressor argument is measured | WEAKENS to a condition relative to a chosen reference set; introduces dependence on `theta_bar`. |
| No encoder | INVALIDATES coverage of a contribution injected purely through `x(0)`. `rho` is never charged. |
| Static learning component | Guarantee on the latent block is ABSENT (correctly, Sect. 5.2). With a baseline-only reference set, additionally INVALIDATES control of the latent path (6.3). |
| No recursion | WEAKENS to exactness on the frozen set. Block-Jacobian argument survives. |
| Assumption 1, full column rank | Fails structurally (rank 10 of 14). WEAKENS to uniqueness on the identifiable subspace. Orthogonality unaffected. |
| Condition 4 | Unverifiable on real data. WEAKENS to the Eq. 21 error expression. |
| `Sigma_e` diagonal | WEAKENS to block diagonality of `Psi' Psi` without the covariance conclusion. |
| Stable predictor, persistency of excitation | Our rollout is marginally stable on two integrator axes (both are pure integrators, no restoring stiffness). INVALIDATES the consistency results. |
| Frozen basis transfers | Regressor depends on the scheduling variable; held-out scheduling positions are a validation axis. WEAKENS to exactness over the range the reference set visits. |
| One-step prediction cost | Ours is free-run with multiple shooting. Removes comparability with the source's numerical results. |

Verification: exact symbolic arithmetic (MATLAB Symbolic Toolbox), two legs. A minimal surrogate
carrying every structural feature (two physical states, one latent state, parameters entering as a
sum and rationally, self-scheduling, routing, lossy output map). And the REAL gantry equations at
explicit-Euler fidelity with the full rational `M(Y)^-1` and all 14 parameters. Explicit Euler
replaces the actual integrator deliberately: it changes no structural property checked and keeps
expressions tractable.

---

## 8. What we want from you

Settled above; please do not relitigate. These are open, in descending order of value to us.

1. **The strongest one.** Failure 6.4 says aggregate orthogonality under the flat weighting is not
   aggregate orthogonality under the rollout's propagation weighting. That suggests replacing the
   Euclidean inner product on the stacked state updates with one WEIGHTED by the propagation
   operator, something Gramian-like, so that the condition accounts for how each step's write
   reaches the output. Question: can a weighted inner product be chosen so that the one-step
   condition implies (or usefully bounds) horizon-level non-overlap, while keeping the weight
   independent of the network parameters, which is what the block-Jacobian argument needs? If yes,
   this would close the gap without going to the input-output level. If no, what is the obstruction?
2. **Convergence of the fixed point.** (R2) refreshes the reference set from the previous epoch's
   augmented rollout. Does this iteration converge, and under what conditions? Note a specific
   concern: the network is zero-initialised, so at the first epoch the learned write is zero and the
   latent states are zero, which is exactly the degenerate slice (R2) exists to escape. Can the
   iteration leave that slice, and what initialisation guarantees it?
3. **Does the block-Jacobian result survive a per-epoch refresh of the LINEARISATION point?** We
   intend to refresh `theta_bar` per epoch, held constant within the epoch so no gradient flows
   through it. Within an epoch the regressor is constant, so the argument seems to go through. Is
   there a subtlety across epochs, for the asymptotic covariance statement specifically?
4. **The metric.** Orthogonality is not invariant to row scaling, and our pipeline runs in
   normalised coordinates while the physics is in SI units. Which metric should the condition live
   in, and on what principle? This changes which contributions count as overlapping.
5. **Boundedness.** The construction injects `- Phi_tilde_p(x_p(k), u(k)) eta_aux` into the physical
   rows. This term grows with the state, and two of our axes are pure integrators. Is there a
   boundedness or stability statement available for the augmented rollout under this construction,
   or a monitored quantity that would give early warning?

A general note on what we value: verdicts over caveats. If something invalidates a claim, say so;
if it weakens it, say what it weakens to. We would rather hear that an approach is wrong than have
it hedged.
