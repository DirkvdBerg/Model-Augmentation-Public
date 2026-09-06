# The gap, after the 2026-09-04 literature sweep

**What changed on 2026-09-04.** A three-agent sweep over adaptive control, fault detection,
iterative learning control and closure modelling closed one of the two candidate gaps and
sharpened the other. References and verification levels:
`docs/references.md`, subsection "2026-09-04 sweep". This file states what is left.

## 1. Closed: trajectory-level orthogonality is not our gap

The pointwise condition really is insufficient over a rollout, because a parameter change
propagates (`sum_j A^(l-1-j) Phi_j`, `A = df/dx`) and termwise orthogonality does not survive
that mixing. **But the windowed construction already exists, twice.**

**In our own lineage.** Kon et al., IFAC World Congress 2023 (`kon2023xray`), same first author
as the ACC 2022 origin paper, stacks the whole trajectory as `f_M = X_{zeta_n}(theta_d) zeta_l`
with `X` the parameter-sensitivity Jacobian over the window, takes its SVD, and penalises the
network on `U_1`, the physical model's trajectory subspace. Measured effect: feedforward RMS
1.410 to 0.325 V at equal tracking accuracy.

**As a general construction, since 1984.** Parity space chooses `Z_s` from the left null space
of the extended observability matrix, `Z_s H_{0,s} = 0`, over a window of `s+1` samples.
Replacing `H_{0,s}` by the windowed parameter sensitivity gives our object, and **Gyorok's
pointwise penalty is the `s = 0` case of it**. Srinivasarengan et al. have already made exactly
that substitution for parameter identifiability in LPV, with a parameter-dependent `O(theta)`
and its left null space.

**Consequence for the thesis:** do not claim trajectory-level orthogonality as a contribution.
Cite it. The honest framing is that the construction is known and we are asking what it costs
when one assumption of it fails.

## 2. What survives, and it is now the only thing

**Every construction found assumes the learned term is memoryless.** Kon 2023 and van Haren
2024 stack a static map sample-wise. Parity space annihilates an unknown initial state, not a
term with its own dynamics. Mori-Zwanzig requires a projector fixed a priori and independent of
the estimated parameter.

Ours is not memoryless. One network emits `w = phi([x_phys; x_aug; u])`; its first rows correct
the physical states and its last rows **become** `x_aug` at the next step, which the network
then reads back. So the correction reaching the physical rows at step `k` depends on values the
network itself wrote earlier.

**The negative is strong on three independent mechanisms**: whole-of-arXiv
`abs:"parity relation" AND abs:"neural network"` = 0, dblp title search returns nothing usable,
and one full-text Google Scholar sentence returns 0 of 15.

That is the gap. Not "pointwise versus trajectory", which is taken, but **the stateful case of
a construction that assumes statelessness.**

## 2b. The lineage says both halves of this itself, in print

Two clauses, verified against the PDFs on 2026-09-04, one per paper. They matter because they
are the authors' own words about their own methods, which is the strongest kind of support for
a gap statement and the hardest kind for a committee to wave away.

**The dynamic-augmentation case shows negation, in their own results.** Hoekstra, Gyorok, Toth,
Schoukens, LFR augmentation of first-principle models, arXiv 2602.17297, p11
(`literature/closed-loop-id/hoekstra2026_lfr-augmentation-fp-models.pdf`):

> "The estimated physical parameters **remain very close to the initialisation values** for both
> the ideal and the approximate case. This is not desired behaviour for the approximate
> initialisations, **indicating that the learning components are learning parts of the system
> dynamics that could be represented by the baseline model.**"

Read the surrounding paragraph: the same page reports that "the dynamic augmentations are able
to capture the dynamics accurately". So the negation symptom is observed **with a dynamic
augmentation**, by this group, and no orthogonality regularization is applied to it in that
paper (it uses only the nominal-value parameter penalty). That is our gap, evidenced from
inside the lineage rather than argued from outside it. It is also the same shape as this
project's own frozen-`theta` symptom.

**The coefficient is the successor's stated motivation for abandoning the penalty.** Gyorok,
Schoukens, Peni, Toth, IFAC J. Systems and Control 35:100376, 2026, p2
(`literature/Orthogonality/gyorok2026_orthogonal-by-construction-augmentation_IFAC-JSC_arXiv2511.01321.pdf`):

> "since these approaches promote orthogonality via regularization, inherently, there exists a
> **trade-off between model accuracy and the desired complementarity. Finding the appropriate
> trade-off parameter (i.e., regularization coefficient) may not be intuitive.** To overcome
> this, we propose a direct parametrization with guaranteed orthogonality ... without requiring
> any trade-off parameter."

So `beta` selection is not a detail we happen to find awkward; removing it is the reason the
successor paper exists. Their p9 sharpens it: with only test or validation error available,
selecting the coefficient is nontrivial, because validation error cannot reveal parameter
non-uniqueness. **Two of this project's supervisors are authors.** Any defence of the penalty
route has to answer that sentence, which is what gap P1c exists for.

Note the two clauses are about different things and should not be merged: the first is about
**parameter initialisation** and negation, the second about the **regularization coefficient**.
A third 2026 paper from the same group treats **encoder** initialisation (arXiv 2602.13108),
which is a further separate question.

## 3. The mechanism, stated precisely

Two distinct problems, and only the second is exotic.

**Coverage.** The penalty is evaluated on a fixed list of about 6,700 input vectors, each 11
numbers, with the two augmented-state slots **set to zero in every one**. So the network is
constrained on a 9-dimensional slice of an 11-dimensional input space and is free off it.
Sharply: the first-layer weight columns multiplying the `x_aug` inputs are multiplied by zero at
every penalty point, so their gradient from the penalty is exactly zero. Measured
consequence (T4): a 21-dimensional space of final layers at exactly zero penalty that still
places 4% of the network output inside `span(Q)` once `||x_aug|| ~ 1`.

**Composition.** With internal state the network partly chooses where it is evaluated, because
`x_aug` along a trajectory is whatever the network put there. Checking a fixed list of points
constrains the function but not a system that selects its own operating points. Coverage is
fixable by giving the evaluation measure support on the `x_aug` axes; composition is not, and it
is why this is not merely a sampling oversight.

**Correction to an earlier version of this argument.** If orthogonality held on the *whole*
domain, memory alone would not break it, since a continuous function orthogonal everywhere is
orthogonal along any trajectory. The problem is that the condition is enforced on a chosen
finite set, and the set cannot be fixed in advance when part of it is chosen by the thing being
constrained.

## 4. The added states are load-bearing: measured

**Zeroing `x_aug` on a trained model roughly DOUBLES the simulation error.** Ablation run
2026-09-04 on the stored checkpoints of `scripts/gantry/meeting/meeting-07-09-2026`
(same three logs and checkpoints as `meeting-07-02-2026`). Condition A is the trained model,
condition B is the same model with `x_aug` forced to zero, condition C is the initial
validation. Identity checks matched the stored bestfit on both runs.

| Run | `nx_ann` | A (trained) | B (`x_aug = 0`) | C (initial) | `x_a` share | B/A |
|-|-|-|-|-|-|-|
| 81262 | 8 | 5.774e-06 | 1.159e-05 | 2.062e-05 | +39.20% | **2.008** |
| 81265 | 2 | 5.956e-06 | 1.119e-05 | 2.071e-05 | +35.49% | **1.879** |

**This supersedes an earlier claim in this file that `x_aug` is "barely used".** That claim
rested on two checkpoints which were both effectively untrained: one had been rolled back to
initialisation by deepSI, the other had an output layer at absmax `2.1e-5`. Neither was
evidence about a trained model, and the ablation above is a causal test rather than an
observation.

**Two caveats to carry with the numbers.**

`joint_estimation=False` and therefore `orth=False` in all three runs. That is the
configuration, not a caveat about the penalty: with `theta` fixed there is nothing for the ANN
to absorb, so the penalty has no purpose and is correctly off. It does mean these numbers
measure **unconstrained** use of the augmented states, which is the right baseline for "how much
room is there" and says nothing about behaviour under an active penalty.

Run 81263 is **not** a third data point: it diverged at update 3900 and ended at `4.8e-03`.
`lr = 3e-5` is unstable on this dataset once `zeta_a` dropped to 0.03.

**What this does to the gap.** The states carry roughly half the model's accuracy, so under a
penalised run the network has a strong incentive to use a channel the penalty never evaluates.
That is an argument about the future penalised arm, not about these checkpoints. It converts
Sect. 2 from a structural possibility into a live risk.

**Still unmeasured: the magnitude `||x_aug||` itself**, which is what places us on the T4
leakage curve (alignment 0.0011 / 0.0107 / 0.0256 / 0.0406 at `||x_aug||` = 0.01 / 0.1 / 0.3 /
1.0, with the 4% figure measured at 1.0). The ablation shows the states matter; it does not say
whether their realised norm reaches the scale where hiding is possible. Instrumented
2026-09-04 in `training.py`: the per-epoch probe now prints `||x_aug||` mean, std and max at the
end of the validation rollout, in the normalised frame, reusing the final state that `simulate`
already returned and previously discarded.

## 5. The ceiling, which applies whatever we build

Cho et al., CDC 2023: once the flexible term is a universal approximator with a nonzero
learning residual, exponential parameter convergence degrades to **uniform ultimate
boundedness**, and a robustifying modification becomes necessary. So the strongest defensible
claim for a neural augmentation is bounded parameter error, not exact recovery. Any stronger
claim, ours included, should be read against that sentence.

Three unrelated fields independently freeze the projector and demote the constraint to a
penalty (Kon 2023 freezes `U_1`; Mori-Zwanzig needs `P` fixed; adaptive control uses leakage).
Evidence that the hard, moving-projector version does not survive rather than that nobody tried.

## 6. Assumptions of the existing construction that we also violate

Ranked by how much work each would be to defend. All from `kon2023xray` unless noted.

| Assumption | Our situation | Status |
|-|-|-|
| Learned term is memoryless | it carries `x_aug` and reads it back | **the gap, Sect. 2** |
| Physical model linear in the estimated parameters | LPV-LFR, nonlinear in the parameters | Gyorok's Sect. 4 Taylor variant covers this; exact only to first order at `theta_bar` |
| `U_1` frozen at the physics-only fit | ours is a deliberate 10% detune, 11 degrees of measured rotation | open; their justification is empirical, on a static map |
| Full-state measurement for the projection | positions only, states reconstructed (D-111) | named as open by the by-construction paper's own conclusion |
| Full column rank | rank 10 of 14 | closed: the reparameterisation to `kappa` is exact, proved in `nullspace-check.md` |

## 7. What would close the gap

Not settled, and listed so the options are visible rather than as a recommendation.

**Give the evaluation measure support on the `x_aug` axes**, drawn from the model's own rollout
and **detached**. Detachment is not optional: the penalty's gradient into `theta` and the
encoder is currently exactly zero (`STATUS.md` Sect. 0), and an undetached rollout point set
destroys that, letting the optimiser reduce the penalty by rotating the subspace instead of
fixing the network. Closes coverage, not composition.

**Windowed regressor.** Replace the per-sample `Phi` with the stacked `Psi_s` over the rollout,
per `vanharen2024ilc` Remark 2, which is one thin SVD per re-anchoring. Note a dimension check
that runs the opposite way to intuition: `Psi_s` has `n_theta` columns regardless of `s` while
the ambient dimension `p(s+1)` grows, so the annihilated subspace is a shrinking fraction and
**longer windows leave the network more room, not less**. Short windows are the dangerous case.

**Constrain the operator, not the field.** The object that competes with a parameter change is
the augmentation's input-output behaviour over the window, including the `x_aug` path. No
published formulation covers this, which is why it is the contribution and also why it is the
hard option.

## 7b. The derivation, done: an orthogonality condition for the additional states

Section 7's third option is no longer an option, it is derived. `../code/affine_leakage.py`
(sympy, free trajectory symbols so every identity holds for all trajectories) establishes four
things. The augmentation is taken affine, which is the strongest form for the negative half: if
the simplest possible network exhibits the failure, every richer one can.

```
x_{k+1}  = A x_k  + B u_k + f_k            f_k      = K xa_k + L x_k + M u_k + bf
xa_{k+1} = G xa_k + F x_k + H u_k + bg     Gyorok:  no xa, hence no K
```

**A. The negative (motivation, not the contribution).** The pointwise constraint is imposed as
an *identity in `(x, u)` over the entire `xa = 0` slice*, which is strictly stronger than the
deployed finite point set, so "sample the slice better" cannot rescue it. It reduces to
`Phi^T L = 0`, `Phi^T M = 0`, `Phi^T bf = 0`. **`K` appears in none of them.** At `T = 2, 3, 4`
the windowed projection `Psi_T^T dx_T` is nonzero. A control separates two mechanisms that are
easy to conflate: at `K = 0` it is *still* nonzero, so propagation alone already breaks the
guarantee, and that half is inherited from the static method and is not ours. The terms carrying
`K` are the half that exists only because of the additional states.

**B. The missing term, in closed form** (verified against the symbolic rollout at every `T`):

```
dx_T^state = sum_{i=0}^{T-2} Lam_T(i) g_i ,   g_i = F x_i + H u_i + bg
Lam_T(i)   = sum_{j=i+1}^{T-1} A^(T-1-j) K G^(j-1-i)
```

Every term is `A^a K G^b` with `a + b = T-2-i`. The size is governed by `K`, by how hard the
additional state is driven (`F, H, bg`), and by **mixed powers of the baseline dynamics `A` and
the additional-state dynamics `G`**. It vanishes identically iff `K = 0`.

**C. The extension.** Requiring the term to vanish for all trajectories gives a condition that
is independent of the data and of `G`:

```
Psi_T^T A^a K = 0 ,  a = 0 ... T-2
  <=>  range(K)  orthogonal to  span{ Psi_T, A^T Psi_T, ..., (A^(T-2))^T Psi_T }      (EXT)
```

Verified symbolically to annihilate `dx_T^state` identically. **Stated in one line: Gyorok
projects the ANN *output* off the parameter subspace; with additional states one must also
project the *state-to-physical map* off the *propagated* parameter subspace.** At `n_a = 0`
there is no `K`, (EXT) is vacuous, and the method collapses to Gyorok exactly, which is the
consistency check the extension has to pass.

**D. Which enforcement route (EXT) permits, and this is the result that bites.** (EXT) says
`range(K) ⊆ ker(S_T)` with `S_T` the Krylov stack of `Psi_T` under `A^T`. By Cayley-Hamilton the
stack saturates at `a = n_x - 1`, so beyond `T = n_x + 1` the horizon adds no new conditions
(verified by rank: at `n_x = 2` the rank is 1, 2, 2, 2 for `T = 2, 3, 4, 5`). But saturation is
at `rank = n_x`, giving `dim ker(S_T) = 0`, and then **(EXT) forces `K = 0` exactly.** For the
gantry, `Phi` has rank 10 in the identifiable parameters over `n_x = 6` routed rows, so
`span(Psi) = R^6` already at `a = 0`.

> A strict orthogonal-by-construction parameterisation `K = (I - Pi) K~` would therefore delete
> the additional-state pathway altogether, the pathway Section 4 measures at roughly 30% of the
> improvement. By-construction is not available to us **for this term**.

The deployed form must instead be a soft penalty over the realised data, which imposes finitely
many conditions rather than an identity on the state space and so admits nonzero `K`:

```
beta_a * sum_over_windows || Q_Psi^T sum_i Lam_T(i) g_i ||^2
```

This is an argument for the penalty route over the by-construction route that arises **from the
additional states specifically** and has no counterpart in the static case. It is independent of
the `beta`-selection argument in Section 2b, and it runs the other way, so the two should be
presented together rather than one being used to settle the choice.

**Dual engine.** Run in sympy (`../code/affine_leakage.py`) and MATLAB R2025a
(`../code/affine_leakage.m`), agreeing on every row of A, B, C and D.

They did not agree at first, and the disagreement is worth recording because it would have
produced two wrong entries in the table below. Both scripts originally imposed the constraints
with `solve`. For the 4-equation, 8-unknown pointwise system MATLAB returned the **trivial**
branch `L = M = b_f = 0` (reporting "solved 8 of 8"), and the same for `K` in Part C. That makes
the `K = 0` control vanish for the wrong reason and makes `T = 2` look as though it forces
`K = 0`. Both engines now parameterise each constraint by an explicit null-space basis, which
has no branch to pick, and both assert the constraint holds identically after substitution.
Logged under Rule 2 clause 6 in [algebra-tooling.md](algebra-tooling.md).

A scope caveat on the Part A strengthening. Imposing the pointwise constraint as an identity in
`(x, u)` requires `ker(Phi^T)` to be nontrivial, i.e. `n_theta < n_x` pointwise. The deployed
penalty is not this: it is a finite stack over ~6,718 points with a rank truncation, which is
weaker. So Part A proves the failure against a constraint **stronger** than the one we run,
which is the direction that makes the negative result safe. It does not describe the deployed
constraint's geometry.

Rule 1: `A` and `Psi` are the baseline Jacobian already built in
`gantry_dynamic/orth_penalty.py`; everything else is a network parameter. No truth model enters,
so all of it transfers to Telica.

## 7c. Measured: rank of Psi, and the leakage ratio on a trained checkpoint

`../code/leakage_measure.py`, run 81262 checkpoint (`p2LPDA`, `nx_ann = 8`, best val
5.774e-06), 48 windows of 100 steps across the 4 validation records, open-loop rollout from
data-derived `x0` with `x_aug = 0`. Stable across `T = 25` and `T = 100`.

### Test 1 corrects Section 7b Part D

**Per window, `rank(Psi_w) = 6` out of 6, in every window.** So `span(Psi_w) = R^6`, a
per-window projection is the *identity*, and the condition `Psi_T^T A^a K = 0` is not merely
restrictive, it is **vacuous**: it says `K = 0` because there is nothing to be orthogonal to.
Part D read that as "by-construction is unavailable to us". The real conclusion is stronger and
different:

> **Part D posed the condition in the wrong space.** Gyorok's `Q` is not a subspace of `R^nx`.
> It is an orthonormal basis of the column span of the STACKED regressor, ambient dimension
> `n_points * n_rows`, with `span(Q)` of dimension at most `n_theta + 1`. Any per-step or
> per-window statement in `R^nx` degenerates the moment the sensitivities span the state space,
> which they always do here.

In the stacked space the object behaves as it should: `Psi` is `288 x 14` with **numerical rank
10**, so `span(Q)` is 10 of 288 dimensions. Rank 10 is the identifiable-combination count, which
is an independent consistency check on the whole construction. The `(EXT)` condition of Part C
stays correct as *sufficient*; what fails is only the claim that it can be imposed
by construction, and it fails because it is far too strong, not because the geometry is tight.

### Test 2: the leakage ratio, with a null

| Quantity | Value | Meaning |
|-|-|-|
| `r_state = \|\|dx_state\|\| / \|\|dx\|\|` | **0.905** | fraction of the ANN's cumulative effect that flows through `x_aug` |
| `r_leak = \|\|Q^T dx_state\|\| / \|\|dx\|\|` | **0.745** | the unconstrained in-span part |
| `r_total = \|\|Q^T dx\|\| / \|\|dx\|\|` | 0.728 | total in-span fraction, for scale |
| chance level `sqrt(10/288)` | 0.186 | what an unrelated vector scores |
| randomisation null, per-window norms preserved | 0.193 (p95 0.283) | ditto, controlling for concentration |
| **observed / null** | **3.9x**, above the 99th percentile | |

**Read.** Roughly 90% of what the augmentation contributes to the physical state over a window
arrives through the additional states, i.e. through the path the pointwise penalty cannot see.
That contribution sits inside the parameter-sensitivity subspace about four times more than an
unrelated signal of the same shape and the same per-window energy would. The mechanism derived
in 7b is not merely available to the network; **this network uses it.**

### What these numbers may not be used for

- Run 81262 has `orth = OFF` and `joint = False`. This is the **unregularised** network, which
  is the right baseline for "how much leakage appears when nothing opposes it", and is **not** a
  test of whether the penalty would suppress it. That test needs a `joint = True` pair with
  `orth` off and on, which does not exist yet.
- With `joint = False` no parameters are being estimated, so no parameter drift is being
  measured here. The quantity is geometric alignment, which is the precondition for drift, not
  drift itself. Do not report it as the cause of the drift failure.
- The aggregate is norm-weighted and the energy is concentrated (top 5 of 48 windows carry 48%
  of `||dx||^2`). The null test is computed against the same concentration, so the 3.9x survives
  it, but a per-window median should not be quoted as if it were the same statistic.
- Category (Rule 1): data-computable throughout. `A`, `Phi` are baseline Jacobians, the rest is
  network output and measured data. No truth model. This runs on Telica unchanged.

## 8. Status of each claim in this file

| Claim | Level |
|-|-|
| Trajectory-level orthogonality already exists | **Cited**, read in full (`kon2023xray`, `zhao2023parity`) |
| No construction covers a stateful learned term | **Graded negative**, three mechanisms, strong |
| The pointwise penalty is the `s = 0` parity case | **Derived** here from a verified construction, not stated in any source |
| 21-dimensional invisible subspace, 4% leakage | **Measured** (T4), on a constructed exploit, not a trained network |
| Added states are load-bearing (`B/A` ~ 2) | **Measured**, causal ablation on two trained checkpoints, Sect. 4 |
| Realised `||x_aug||` magnitude | **Unmeasured.** Instrumented, never run |
| UUB ceiling for universal approximators | **Cited**, read in full (`cho2023uub`) |
| Negation observed with a dynamic augmentation in the lineage's own results | **Quoted**, verified against the PDF 2026-09-04 (`hoekstra2026lfraug` p11) |
| `beta` selection is the successor's motivation for dropping the penalty | **Quoted**, verified against the PDF 2026-09-04 (`gyorok2026obc` p2, p9) |
| Longer windows leave more room | **Derived** here, dimension count only; not verified numerically |
| Pointwise constraint leaves `K` entirely free | **Proved** symbolically (`affine_leakage.py` A), constraint imposed as an identity on the slice |
| A state-path leakage term exists that the static case does not have | **Proved**, with a `K = 0` control separating it from propagation |
| `Lam_T(i) = sum_j A^(T-1-j) K G^(j-1-i)` is that term | **Proved**, closed form verified against the rollout at `T = 2, 3, 4` |
| (EXT) `Psi_T^T A^a K = 0` annihilates it and reduces to Gyorok at `n_a = 0` | **Proved** symbolically |
| (EXT) enforced by construction forces `K = 0` once `span(Psi) = R^nx` | **Proved** by rank/Cayley-Hamilton, but **posed in the wrong space** (Sect. 7c). Superseded. |
| The extension must therefore be a penalty, not a construction | **Follows** from the row above; not yet implemented or run |
| `rank(Psi_w) = 6` per window, so a per-window projection is vacuous | **Measured** (48 windows, T=25 and 100) |
| Stacked `Psi` has rank 10 of 288, matching the identifiable-combination count | **Measured** |
| ~90% of the augmentation's windowed contribution flows through `x_aug` | **Measured**, unregularised checkpoint 81262 |
| That contribution is in-span 3.9x more than chance (above the 99th pct of the null) | **Measured**, randomisation null preserving per-window norms |
| The penalty would suppress it | **Untested.** Needs a joint=True orth on/off pair |
