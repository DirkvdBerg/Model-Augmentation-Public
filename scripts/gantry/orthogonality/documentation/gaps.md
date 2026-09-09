# The gap, after the 2026-09-04 literature sweep

**Implementation handoff (2026-09-07):**
[Fixed-reference rollout orthogonality implementation plan](plan/fixed-reference-rollout-orthogonality-implementation.md).

**Implementation contract revised 2026-09-08:** that existing plan now consolidates the actual
closed-loop gantry equations, trainable latent-initialization ownership, physical-encoder
profiling, D-184 factorization, scalar-MSE scaling, production verification and a single
implementation-to-evidence index. Its encoder migration and compressed geometry are proposed,
not completed experiments. This file remains the evidence record.
This plan owns the proposed code changes and verification sequence; it is not a report of
completed implementation or an additional parameter-recovery theorem. This file remains the
live theory/evidence record.

**How to read this file (structure fixed 2026-09-06).**

| Part | Role |
|-|-|
| **Section 0** | **Binding framing.** What the method claims and what it does not. Governs the wording of every result below. |
| Sections 1-7 | The gap: what the sweep closed, what survives, the mechanism, and the measured evidence that the added states are load-bearing. |
| Section 7b | First derivation (affine witness). **Partly superseded**, see the supersession note at its head. |
| Section 7c | Measurement on a trained checkpoint. Current. |
| **Section 9** | **Live derivation.** Finite-horizon separation with learned states and an encoder. Owns the formal statement of the mechanism; 7b is kept for the affine witness and the `K = 0` control. |
| **Section 10** | Status of every claim in the file. Nothing may be quoted without its row. |

The [source audit](theory-source-audit-2026-09-06.md) is a supporting appendix. Its Sections 1,
2 and 7 (source assessments and the novelty boundary) are authoritative there; its Sections 3 to
5 restate theory that Section 9 now owns, and where they differ Section 9 wins.

**What changed on 2026-09-04.** A three-agent sweep over adaptive control, fault detection,
iterative learning control and closure modelling closed one of the two candidate gaps and
sharpened the other. References and verification levels:
`docs/references.md`, subsection "2026-09-04 sweep". This file states what is left.

## 0. What the method claims, and what it does not

Binding on how every result in this file may be worded. Set 2026-09-06 after the framing below
was corrected; the earlier reading over-weighted the recovery floor and is retracted here.

### The claim

> **Orthogonality makes the physical-parameter estimate well defined and no worse than the
> baseline-alone fit, while the flexible component still improves prediction.**

It is not a claim about recovering true parameter values, and it does not need to be.

### Why the "no worse" half holds, which is the part that was previously stated backwards

Let the data be `baseline(theta_true) + Delta`, `Delta` the missing physics, and split it as
`Delta = Delta_par + Delta_perp` relative to `span(Phi)`.

| Fit | What absorbs `Delta_par` | Resulting parameter error |
|-|-|-|
| Baseline alone | `theta` | `(Phi^T Phi)^-1 Phi^T Delta` |
| Baseline + **orthogonal** flexible term | `theta` (the flexible term is barred from `span(Phi)`) | `(Phi^T Phi)^-1 Phi^T Delta`, **identical** |
| Baseline + **unconstrained** flexible term | either the flexible term or `theta`, indeterminate | unbounded either way; this is negation |

So the constraint does **not** create the bias. The bias is already there in the baseline-alone
fit, and it is the bias relation on record. What orthogonality removes is the *indeterminacy* of
the third row: without it the flexible term may absorb the missing physics and leave `theta`
clean, or absorb baseline dynamics and corrupt it, and nothing in the fit decides which.
The constraint caps the downside; it does not move the parameter estimate away from where the
baseline fit would already have put it.

### The role of the recovery floor, corrected

`||Q^T Delta_true|| / ||Delta_true|| = 0.826` (`../code/recovery_floor.py`) is the right number
for **how much of the true discrepancy a parameter change could mimic**, so it bounds how well
parameters could ever be recovered here. It is **not** an argument against the method, and it
must not be quoted as one. Any earlier wording in this file suggesting the penalty "forces the
parameters to absorb the missing physics" as a cost of the constraint is wrong: the baseline
fit does that anyway.

### What the constraint is, and is not, restricting

The constraint applies **only to the flexible component**. `theta` is never restricted and
roams all 14 parameters. `span(Phi)` is not a search region for `theta`; it is the *image* of
what parameter changes do to the prediction, and the flexible component is pushed out of that
image. Any effect on `theta` is an indirect consequence of the fit, not a restriction.

### The limitation that is real, and is inherited

Orthogonality cannot distinguish two situations that are geometrically identical to it:

1. the flexible component taking over work `theta` should have done (negation, to be blocked);
2. the flexible component fitting genuine missing physics that happens to resemble a parameter
   change (legitimate, blocked anyway).

Both are "output lies inside `span(Phi)`". This blindness comes with the approach, Gyorok's
included, and is not introduced by the extension to additional states. It is the reason the
claim is a **unique, well-posed split** rather than a true one, which is also the settled
position in the calibration literature: with a nonparametric discrepancy the true parameter
value is not well defined, and the standard resolution is to *define* `theta*` as the minimiser
(`xiexu2020projected` p6, and uniform across `tuo-wu2015`, `tuo2019projected`, `wang2022`).

### What the additional states are, and are not

They are a **dynamic residual model**. They capture temporal structure a static correction
cannot, and they carry roughly 90% of the augmentation's contribution to the physical state
**in normalised state space**, or 43% of it **in output space** (Sect. 7d). The output-space
figure is the one that bears on substitution, because that is where the loss lives
(Sect. 7c). What they represent physically is **not claimable and does not need to be**. The
only requirement placed on them is that they do not do the baseline's job, which is exactly what
the constraint enforces and exactly what the deployed pointwise penalty fails to reach.

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
| Full-state measurement for the projection | positions only, states reconstructed (D-111) | **Attribution corrected 2026-09-07, see the note below.** `gyorok2025l4dc` names the encoder route itself; the by-construction paper's conclusion is about ITS OWN construction, not this one. |
| Full column rank | rank 10 of 14 | closed: the reparameterisation to `kappa` is exact, proved in `nullspace-check.md` |

### Correction, 2026-09-07: what the lineage does and does not leave open

The row above previously read "named as open by the by-construction paper's own conclusion".
That over-extended a quotation, and the correction matters for how the contribution is framed.

**What `gyorok2026obc` actually says.** Verified verbatim from the PDF on disk
(arXiv 2511.01321v3, page 10, Conclusions, immediately before the reference list; note the
extraction hyphenates `learning com-/ponent`, and the word is `full-state`, so a search for
"learning component" or "full state" returns nothing):

> "Future research may be directed at extending the approach for augmenting baseline models in
> state-space form, where the general assumption of no available full-state measurement
> complicates the orthogonal projection of the learning component, thus requiring careful
> investigation."

That paper is input-output. The sentence is about extending **its own by-construction
parametrisation** to state-space form. It is NOT a statement about the limits of the
regularisation route, and it was wrong to cite it as one.

**What `gyorok2025l4dc` actually does.** It is already state-space ("recent approaches
utilizing state-space (SS) models based on deep ANNs"), trains with the SUBNET T-step-ahead cost
on overlapping subsections, and names the encoder case explicitly:

> "When full-state measurements are not applicable, i.e, when `y_k != x_k + e_k`, an encoder
> network is used to estimate the initial states from past input and output values, similarly as
> in Beintema et al. (2023); Hoekstra et al. (2024)."

So state-space is covered, and so is the encoder. Its PRIMARY setting is the measured full state
`x_hat_{k|k} = x_k`, with the encoder as the stated fallback, which is what `orth_penalty.py`
already records as the reason our states are data-derived (D-111).

**What is therefore genuinely ours.** Not "they never handled state space", and not "the encoder
is unaddressed in the lineage". Both are false. What is ours is narrower and survives:

1. Gyorok's penalty is **pointwise on the ANN output** at sampled `(x, u)`, with no rollout, so
   the encoder parameters cannot appear in it at all. The encoder is not a nuisance channel for
   his construction regardless of how the training loss obtains `x0`.
2. Our encoder problem is created by moving the constraint onto a **trajectory**, which we did
   because the pointwise form provably cannot see the latent path (Sect. 7b). It is a
   consequence of our own extension, not an inherited gap.

**Lesson for citation discipline.** A quotation about one construction does not transfer to a
sibling construction by the same authors. Both papers must be checked separately before either
is used to support a claim about scope.

## 7. What would close the gap

Not settled, and listed so the options are visible rather than as a recommendation.

**Give the evaluation measure support on the `x_aug` axes**, drawn from the model's own rollout
and **detached**. Detachment is not optional. **This aside is superseded by Sect. 7g and
D-182**, which state it as a standing constraint with its two mechanisms rather than as a
remark scoped to the point set: the optimiser can reduce the penalty either by fixing the
network or by MOVING THE SUBSPACE via `theta`, and the second is usually cheaper and corrupts
the quantity the method protects. Closes coverage, not composition.

**Windowed regressor.** Replace the per-sample `Phi` with the stacked `Psi_s` over the rollout,
per `vanharen2024ilc` Remark 2, which is one thin SVD per re-anchoring. Note a dimension check
that runs the opposite way to intuition: `Psi_s` has `n_theta` columns regardless of `s` while
the ambient dimension `p(s+1)` grows, so the annihilated subspace is a shrinking fraction and
**longer windows leave the network more room, not less**. Short windows are the dangerous case.

**Constrain the operator, not the field.** The object that competes with a parameter change is
the augmentation's input-output behaviour over the window, including the `x_aug` path. No
published formulation covers this, which is why it is the contribution and also why it is the
hard option. **This is the option taken.** Derived in Section 9; the affine witness that
motivated it is Section 7b.

*Note on the windowed-regressor bullet above.* Its dimension count is stated in the per-window
ambient space. Section 7c settles which space is the right one (stacked, and in the weighted
output space per Section 9), so that bullet's "longer windows leave more room" conclusion should
be re-derived there before it is used.

## 7b. The derivation, done: an orthogonality condition for the additional states

> **Supersession note (2026-09-06).** Read this section for the affine witness only.
>
> **Still current:** the pointwise constraint leaves `K` entirely free; the `K = 0` control that
> separates propagation (inherited from Gyorok) from the state path (ours); and the closed form
> `Lam_T(i) = sum_j A^(T-1-j) K G^(j-1-i)`.
>
> **Superseded:** everything posed per-window in `R^nx`. Part D's condition is vacuous there
> (`rank(Psi_w) = 6`, Sect. 7c), and its by-construction conclusion is withdrawn. The formal
> statement now lives in **Section 9**, in the weighted output space with the encoder included.


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
| `r_state = \|\|dx_state\|\| / \|\|dx\|\|` | **0.905** | fraction of the ANN's cumulative effect that flows through `x_aug`, **in normalised STATE space**. The output-space value is 0.43 (Sect. 7d) and is the one that matters. |
| `r_leak = \|\|Q^T dx_state\|\| / \|\|dx\|\|` | **0.745** | the unconstrained in-span part |
| `r_total = \|\|Q^T dx\|\| / \|\|dx\|\|` | 0.728 | total in-span fraction, for scale |
| chance level `sqrt(10/288)` | 0.186 | what an unrelated vector scores |
| randomisation null, per-window norms preserved | 0.193 (p95 0.283) | ditto, controlling for concentration |
| **observed / null** | **3.9x**, above the 99th percentile | |

**Read, with the qualifier that was missing until 2026-09-07.** Every number in this section is
computed in NORMALISED STATE SPACE. Sect. 7d repeats the measurement in OUTPUT space, where
the loss lives, and the latent share is 43% rather than 90%. Both are correct about different
quantities; quote the output-space one for anything about substitution.

Roughly 90% of what the augmentation contributes to the physical state over a window
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

## 7d. Measured in OUTPUT space at a trained checkpoint: both routes are aligned
> ### SUSPECT, PENDING RE-MEASUREMENT (marked 2026-09-07)
> An independent review found defects in the code that produced this section. **Do not quote any
> number here.** Details and the specific defect are given immediately below; the section text is
> retained unedited so the correction can be checked against what was originally claimed.
>
> **Defect.** `gantry_ell_lat.py` never applies the saved output map `Cd`. Line 154 reads
> `x[:, :NY] if Cd is None else x @ torch.eye(NX_PHYS)[:, :NY]`, so BOTH branches take the first
> three normalised physical STATES. The section's central claim, that this is measured in output
> space, is therefore false, and with it `ell_lat = 0.957`, the three-route comparison, and the
> 43%-versus-90% correction. The rollout is also open loop with data-derived initial states and
> `E = None`, not the closed-loop encoder-profiled predictor, and it appends the output AFTER the
> state update whereas the production simulator records it before. The saved-versus-parameterised
> parity check cannot catch any of this, because both sides run through the same custom loop.
>
> **Consequence elsewhere.** The 90% figure in Sect. 7c was qualified in four places on the basis
> of this section. Those qualifications are suspended: the 90% stands as originally measured in
> normalised state space, and there is currently NO validated output-space figure to set against it.


`../code/gantry_ell_lat.py`, run **81758** (the most converged of the thirteen saved runs in
`meeting-07-09-2026/server/`: best validation 5.7483e-06, `nx_ann = 8`, LBFGS-polished).
Like every one of those runs it has `joint = False` and `orth = False`, so this is the
unregularised network with nothing opposing the alignment. It is **not** a test of the penalty.

24 windows of 100 steps, open loop from data-derived `x0`. Geometry and diagnostics come from
the shared component `model_augmentation/fit_systems/trajectory_orth_projection.py`, the same
code the testbed uses, so a bug here is a bug there.

| contribution | `ell` | `a_par` | `a_perp` | norm |
|-|-|-|-|-|
| total  | 0.9343 | 1.456e+01 | 5.556e+00 | 1.559e+01 |
| latent | **0.9571** | 6.458e+00 | 1.955e+00 | 6.748e+00 |
| direct | 0.9381 | 1.495e+01 | 5.517e+00 | 1.593e+01 |

Chance level `sqrt(10/7200) = 0.0373`. Read `ell` against that, not against zero.

**What this settles.** The testbed adapter, at an UNTRAINED network, reported the latent route
as nearly orthogonal (`ell_lat = 0.064`) while the total was strongly aligned. Had that
survived, it would have meant the alignment lives in the direct correction and the
additional-state extension is not where the value is. It does not survive. On a converged model
the latent route is the **most** parameter-aligned component present, at 26x chance. The
untrained reading was an artefact of initialisation.

**Both routes are aligned**, 0.93 / 0.96 / 0.94. This explains, rather than merely repeats, the
earlier empirical finding that penalising one route leaves the other untouched: they are
comparably aligned, so constraining either alone leaves an equally aligned path free. It is the
argument for penalising `d_total`.

**The 43% correction.** The latent route carries `6.75e+00` of `1.56e+01`, so **43%** of the
contribution norm in output space, not the 90% reported in Sect. 7c. That earlier figure is
`norm(dx_state)/norm(dx)` in normalised STATE space. Both are correct about different
quantities. Substitution happens in the loss, so the OUTPUT-space figure is the one to quote,
and every appearance of the 90% now carries its space.

**Three things confirmed in passing.** `rank(S) = 10 of 14` through the full augmented rollout
in output space, matching the identifiable-combination count independently of the earlier
one-step state-space calculation. The three-intervention identity `d_total = d_lat + d_direct`
holds to `1.5e-16`, so the modes share one rollout path rather than three transcriptions.
`rank(Q_E) = 0` because this rollout starts from data-derived states, so there is no encoder
freedom to profile out here; a run seeded by the trained encoder would have some.

**Two constructions that needed justification rather than assertion.** The saved model's
physical block is not parameterised, so the geometry uses a parameterised copy at nominal
`theta`; parity between them is `2.05e-08`, checked to be a CONSTANT offset (the one-step
difference is already 1.18e-08 and does not grow over 100 steps) traced to float32 rounding in
the nominal constants (`theta_bar[3] = 20.29999924`). A gap that grew with the horizon would
have invalidated the geometry; this one does not. And forward-mode differentiation required
rebinding `log_params` as a plain attribute, since `.data` assignment strips the dual tensor.

## 7e. Memory earns its place, but the latent states are not learning the absorber
> ### CORRECTED AND RE-MEASURED 2026-09-07. The conclusion CHANGED.
> The defect below was found in review, fixed, and the experiment re-run. **Use the corrected
> table in this box, not the one in the body text**, which is retained so the change can be
> checked.
>
> | held-out RMSE | T=32 | T=64 | T=128 | T=256 |
> |-|-|-|-|-|
> | none | 9.937e-04 | 1.048e-03 | 1.099e-03 | 1.739e-03 |
> | static | 1.010e-03 | 1.064e-03 | 1.109e-03 | 1.518e-03 |
> | **dynamic** | 1.026e-03 | **1.008e-03** | **9.751e-04** | **1.129e-03** |
> | static / dynamic | **0.985** | 1.056 | 1.137 | **1.345** |
>
> **Two things changed, and one of them is a retraction of a retraction.**
>
> 1. At the shortest horizon the dynamic model is now **worse** than the static one (0.985),
>    where it previously appeared better. Memory does not help within the training horizon.
> 2. The advantage now **GROWS MONOTONICALLY with horizon**: 0.985, 1.056, 1.137, 1.345. The
>    body text below records "a prediction of ours that failed", namely that the advantage
>    should grow because a static map cannot accumulate. **That prediction did not fail. Its
>    apparent failure was the window bug.** With matched windows the accumulation argument is
>    supported, and the mechanism is accumulation after all.
>
> This also weakens the `rho(G)` story further: an advantage that grows with horizon is
> accumulation, which sits awkwardly with a latent block of 12 ms memory, and supports the
> review's point that the LOCAL block's spectral radius does not describe the coupled system.
>
> **Defect.** `memory_check.py` drew the validation window starts INSIDE the mode loop
> (line 147), so the none, static and dynamic conditions were each evaluated on DIFFERENT
> windows. Fixed: one manifest per horizon, drawn before the mode loop and shared, with the
> manifest saved into the results file.
>
> **A second correction, independent of that bug.** The interpretation that `rho(G) = 0.18`
> shows the latent states carry no useful memory is too strong. A static correction still
> propagates through the physical dynamics, and the spectral radius of one local `G` block does
> not establish the memory of the COUPLED system. The conclusion that the latent states do not
> reconstruct the absorber may well survive re-measurement, but this argument does not establish it.


`../../testbed/memory_check.py`. Physical parameters fixed at truth, no penalty, so nothing
competes with the augmentation and the only question is predictive value. Trained on 32-step
windows, evaluated on held-out data at longer horizons. TESTBED, not gantry evidence.

| held-out RMSE | T=32 | T=64 | T=128 | T=256 |
|-|-|-|-|-|
| none | 9.94e-04 | 1.05e-03 | 1.10e-03 | 1.74e-03 |
| static | 8.78e-04 | 1.01e-03 | 1.19e-03 | 1.48e-03 |
| **dynamic** | **7.88e-04** | **7.89e-04** | **1.07e-03** | **1.24e-03** |
| static / dynamic | 1.113 | 1.280 | 1.113 | 1.193 |

Dynamic beats static at every horizon, which clears the prerequisite: preserving the latent
route is preserving something with demonstrated value. Note also that at `T = 128` the STATIC
augmentation is worse than no augmentation at all (1.19e-3 against 1.10e-3) while the dynamic
one is better. A static correction extrapolates badly beyond its training horizon; internal
state does not.

**A prediction of ours that failed.** We expected the advantage to GROW with horizon, since a
static map cannot accumulate. It does not: 1.11, 1.28, 1.11, 1.19, essentially flat. The
mechanism is therefore not accumulation.

**The Jacobian blocks say why, and this is the more interesting result** (D-180):

```
||K||/||L|| = 0.607    the latent path IS wired in
rho(G)      = 0.180    latent memory decays with time constant ~0.6 steps, about 12 ms at 50 Hz
||x_aug||   = 2.38     large, and per D-180 this alone tells us nothing
```

The absorber is a lightly damped resonance at 3.18 Hz, a period of 314 ms, about 16 steps. A
12 ms memory cannot represent it. So the latent states are learning a short-lag correction,
which beats a memoryless map because a one-step lag is information a static map cannot have,
and which is nowhere near the physics. That is also why the advantage does not grow with
horizon: there is no long-memory content to accumulate.

**What this does NOT license, and the distinction matters for how the thesis frames the
additional states.** The latent states are a DYNAMIC RESIDUAL MODEL. They capture real temporal
structure the baseline lacks, and in that sense they are physically meaningful. They do **not**
map one-to-one onto the actual missing dynamics, and no result here claims they do.
`rho(G) = 0.18` against a 314 ms resonance is direct evidence that they are not a
reconstruction of the absorber. The only requirement placed on them is that they do not do the
baseline's job, which is what the constraint enforces. Any wording suggesting the latent states
identify, recover, or correspond to the hidden physics is unsupported and must not appear.

Whether `rho(G) = 0.18` is a fixable architecture or initialisation problem, or a limit of a
shared MLP with no explicit latent dynamics, is OPEN and is the motivation for the explicit
latent-block option recorded in D-180.

## 7f. The penalty ladder on the testbed, and the tie that blocks the efficacy claim

`../../testbed/train_compare.py`, six conditions, paired initialisations, same beta grid,
600 Adam iterations, 48 windows of 32 steps, noiseless. TESTBED, not gantry evidence.

| condition | best `kappa` error | vs baseline-alone |
|-|-|-|
| baseline only | 0.1200 | 1.00x |
| joint, no penalty | 0.2203 | 1.84x worse |
| joint + pointwise | 0.1520 | 1.27x worse |
| joint + extension (`d_total`) | 0.0950 | 0.79x |
| joint + size penalty, CONTROL | **0.0932** | 0.78x |
| joint + both | 0.0987 | 0.82x |

**The extension rotates rather than shrinks.** `ell` falls 0.832 to 0.250, with the in-span
amplitude down 25x against 4x for the orthogonal amplitude. The diagnostics are verified able
to tell those apart (`test_trajectory_projection.py` V0), so this is a measurement and not an
artefact of the metric.

**But the tie is unbroken, and it blocks the efficacy claim.** A plain contribution-size penalty
with no projection at all matches the extension on parameter error, 0.0932 against 0.0950. The
only measured difference is that the projection retains about 20x more orthogonal contribution
(`a_perp` 3.51e-3 against 1.89e-4). So the defensible claim is **reduced parameter-aligned
contribution at matched retained correction**, not better parameters. Until the PROFILED-size
control is also run, even the retained-correction advantage may be partly an encoder-space
effect rather than directional selectivity, since the raw-size control penalises encoder-space
components that the projection ignores.

**Penalising only the latent route fails**, and Sect. 7d explains why: `ell_state` collapsed
while `ell_total` stayed at 0.21 in every condition. `d_state` is the right object for a
DIAGNOSTIC and the wrong one for the penalty.

**The pointwise term adds nothing measurable**, and makes parameter error marginally worse at
every beta (0.1108 / 0.1074 / 0.0987 against 0.1095 / 0.1061 / 0.0950). Those gaps are 1 to 4%
on a single seed and are not distinguishable from run variation. More importantly the test was
**structurally incapable** of detecting the benefit the pointwise term exists for: it buys
COVERAGE, constraining the network at points off the realised trajectory, and both training and
evaluation stayed on the same distribution. A null result was close to guaranteed. The term is
therefore untested for its purpose, not shown to be useless.

## 7g. The basis is a constant during optimisation, and what it costs to rebuild it

Two connected things: a standing constraint that was previously stated only as an aside, and a
measurement that removes the stated reason for the choice it protects.

### The guard, promoted from an aside to a constraint (D-182)

`V = beta ||Q^T d||^2` measures how much of the augmentation's contribution lies in the
parameter-sensitivity subspace, and `Q` is built from `Phi(theta)`. So there are **two** ways to
make `V` small:

1. **Change the network** so its contribution leaves the subspace. The intended mechanism.
2. **Move the subspace** so it no longer contains the contribution, by moving `theta`. Costs
   nothing in prediction error and is typically **cheaper** than (1).

If `theta` has any gradient path into `V`, the optimiser takes whichever is cheaper, and it is
usually (2). The physical parameters then move so the PHYSICS looks different from the network,
instead of the network being made to look different from the physics. That is the inverse of the
objective: it corrupts `theta`, the quantity being protected, while the penalty falls and the run
looks healthy. Nothing errors, nothing diverges, and no diagnostic fires.

Hence: **`Q` carries no gradient to any trainable parameter.** Computed outside the graph,
installed as a buffer, constant for the whole step. The same applies to the evaluation point set
and, in the trajectory version, to the rollout producing `d`.

This was previously recorded only inside a bullet in Sect. 7's list of options, scoped to an
undetached point set, with no statement of mechanisms 1 and 2, nothing in the code, and no
decision entry. It is now D-182, and it is verified rather than assumed: after a backward pass
through the penalty alone, the physical and encoder gradient groups must be empty. That check is
executable in `../../testbed/testbed_adapter.py` and is plan item V0 for the real pipeline.

### The rebuild cost, measured

`../code/jacobian_mode_bench.py`. `orth_penalty.py` records the fresh build as "~6 min Jacobians
+ ~0.03 s SVD", and that figure is the stated reason for freezing the basis. It is an artefact of
how the Jacobian is taken, not the cost of the mathematics: the deployed loop runs one point at a
time and takes SIX REVERSE-MODE passes per point, about 6 x 6716 sequential backward passes. The
map has 14 parameters against 6 output rows, so the natural mode is FORWARD, 14 passes over the
whole batch.

| | 200 points | full 6716-point set |
|-|-|-|
| reverse, per point, batch 1 (deployed) | 7.52 s | 252 s extrapolated, consistent with the ~6 min on record |
| forward, batched | 0.05 s | **0.42 s measured** |
| SVD | | 0.03 s |
| **total rebuild** | | **0.45 s** |

Forward and reverse agree to `4.5e-16` on the Jacobian and `8.3e-17` on the state, checked before
the timings were trusted. Speedup 145x. This is the same reverse-versus-forward error found in
`train_compare.py`, where it was about 85% of that sweep's runtime.

**What this changes.** Cost is no longer an argument for anything. A per-epoch rebuild would add
0.45 s against epochs measured in minutes. Asynchronous computation is unnecessary. The
justification in `orth_penalty.py`'s docstring for freezing is void and should be corrected when
that file is next touched.

**What it does not change.** The objection to a moving basis was never the price. It makes the
objective non-stationary, so a `beta` chosen on a fixed problem no longer transfers; the network
chases a constraint set that moves under the prediction loss; and it makes D-182 load-bearing at
every rebuild rather than automatically satisfied once. Those hold at 0.45 s exactly as they held
at 6 minutes.

**One discrepancy to keep straight.** This build reports **rank 11** at the deployed tolerance
(sigma ratio 6.46e-07), against **rank 10** through the full augmented rollout in output space
(Sect. 7d). The difference is the extended-regressor offset column `Gamma`, which the trajectory
plan deliberately omits. The two constructions are not interchangeable and their ranks should not
be quoted against each other.

**Status.** `orth_penalty.py` is unchanged: it still uses per-point reverse mode and still freezes
the basis. Switching the differentiation mode is a small verified change to production code, and
re-anchoring is a different algorithm needing its own decision entry and a paired comparison
against the frozen version. Neither has been done.

## 7h. Per-epoch re-anchoring measured: the drift decays and it buys nothing here
> ### CORRECTED AND RE-MEASURED 2026-09-07. The conclusion HOLDS; one number was wrong.
> **Accumulated drift is 5.34 deg (frozen) and 5.36 deg (re-anchored), not the 2.23 deg in the
> body text.** The two arms now differ slightly instead of agreeing to the digit, which is
> itself confirmation that the old figure was measuring the wrong thing. Consecutive angles
> (7.43, 5.64, 0.83 deg) were correct and are unchanged, as are the frozen-versus-re-anchored
> comparison and the conclusion that re-anchoring buys nothing measurable here.
>
> Note the accumulated 5.34 deg exceeds the LAST consecutive angle of 0.83 deg by more than 6x,
> which is the review's point made concrete: small consecutive angles do not bound accumulated
> drift, and reporting only the former would have overstated how settled the basis was.
>
> **Defect.** The start-to-finish drift was computed as `subspace_drift(rebuild(th0).Q_S, g_end.Q_S)`,
> but `rebuild()` closes over the CURRENT augmentation and encoder, which at that point are the
> final ones. So it compared initial `theta` + final ANN against final `theta` + final ANN, not
> the initial model against the final one. That is why both arms reported exactly 2.23 deg, a
> coincidence I noticed and did not chase. Fixed by retaining the actual initial geometry object.
>
> **What is NOT affected.** The per-rebuild consecutive angles (7.43, 5.64, 0.83 deg) compare two
> genuinely consecutive installed bases and stand. Note however that small consecutive angles do
> not imply small ACCUMULATED drift, so they do not by themselves support the conclusion that
> freezing is adequate. The frozen-versus-re-anchored comparison table is unaffected by this
> defect but remains under-converged as already stated.


`../../testbed/reanchor_compare.py` (D-183). Joint estimation ON, `theta` detuned +10/-10/+10/+10%,
extension penalty on `d_total` at `beta = 1.0`, geometry built through the shared adapter with
the encoder profiled out. Two arms differing only in whether the basis is rebuilt.

**Configuration actually run: 16 windows, 200 iterations, 3 rebuilds, 35.5 s.** This is a
REDUCED setting, chosen after the full one (24 windows, 600 iterations) exhausted machine memory.
See the caveats below before quoting any number from it.

### Drift decays

| rebuild at iteration | max principal angle | mean | rank |
|-|-|-|-|
| 50 | 7.43 deg | 5.53 | 3 -> 3 |
| 100 | 5.64 deg | 3.98 | 3 -> 3 |
| 150 | **0.83 deg** | 0.66 | 3 -> 3 |

Decaying toward zero is the benign outcome: `theta` is settling, so the network is NOT chasing a
constraint set that keeps moving. That was the instability D-183 was written to detect, and on
this testbed it does not occur.

**A detail that argues for logging per-rebuild angles rather than endpoints.** The total rotation
from the initial basis to the final one is **2.23 deg in BOTH arms**, yet individual rebuilds saw
7.43 and 5.64 deg. `theta` wandered further out than it ended up, so an endpoint-only comparison
would have reported a third of the excursion actually traversed.

### Re-anchoring changes nothing measurable

| | frozen | re-anchored |
|-|-|-|
| `kappa` error | 0.2591 | 0.2567 (0.99x) |
| validation RMSE | 2.240e-03 | 2.244e-03 |
| `ell` | 0.0028 | 0.0027 |
| `a_perp` | 5.111e-03 | 4.804e-03 |

### Caveats, binding on how this may be used

- **Under-converged.** At 16 windows and 200 iterations `kappa` ends at 0.26, against the 0.12
  baseline-alone reference from Sect. 7f which used 48 windows and 600 iterations. Neither arm
  has settled, so "indistinguishable" may partly mean "both still far from their optimum".
- **Not comparable to Sect. 7f.** Different `beta`, different window count, and a different
  geometry construction (this one goes through the shared adapter with encoder profiling; 7f used
  the local `extension_basis`). The `ell` values in the two sections must not be read against
  each other.
- **Testbed only, and the testbed's drift is small.** The gantry records **11 degrees** of
  subspace rotation from its 10% detune alone (Sect. 6), against 2.23 deg net here. A result
  showing freezing is adequate at 2 deg says little about 11 deg.
- Single seed, single `beta`.

### What this implies for now

Freezing looks adequate on this system, so **re-anchoring is not adopted** and the frozen scheme
remains the default. D-183 stands as a logged, implemented, measured alternative rather than a
change. The decision should be revisited on the gantry, where the drift is five times larger and
where the premise behind Gyorok's frozen basis (an anchor close to the truth) fails hardest.

### The memory bug found on the way, which is worth more than the result

The geometry object materialised a dense `N x N` weight matrix even when the loss weight was the
identity, and expanded a diagonal weight to dense, then registered it as a buffer. At the
gantry's `N = 7200` that is **415 MB per geometry in float64**, plus an equal numpy temporary
during construction, and re-anchoring keeps two alive at once. The weight is now stored in the
smallest form that represents it: identity is never built, a diagonal stays a vector, and only a
genuinely dense weight is stored dense. All 17 checks in
`../../testbed/test_trajectory_projection.py` still pass, and both the identity and diagonal
paths are exercised there.

This would have been fatal on the gantry regardless of whether re-anchoring is ever adopted.

**The OOM itself was not localised.** It does not reproduce at the reduced setting, so it was the
larger configuration rather than a defect that can be pointed at. The dense-weight bug is a
sufficient explanation at gantry scale but NOT at this testbed's `N = 1536`, where the object is
about 19 MB, so it should not be recorded as the confirmed cause.

## 9. Finite-horizon separation with learned states and an encoder

Date: 2026-09-06. **Category: structural.** The proofs below concern a smooth predictor on
fixed finite records. They do not assert that the actual gantry experiment satisfies the rank
conditions, that a finite penalty achieves orthogonality, or that the physical truth is recovered.
The corresponding Jacobians, ranks, and overlaps are **data-computable**, requiring a model
and measured records, not simulation truth. Existing checkpoint statistics are not used in
these proofs.

Sources and assumptions are documented in the [supporting source audit](theory-source-audit-2026-09-06.md).
The underlying tools are least-squares projection and nuisance-variable elimination, not new
linear algebra. Györök 2026, local PDF p. 4, Condition 4 and Theorem 7, separates orthogonal
decomposition from physical recovery: recovery additionally requires orthogonality of the true
missing dynamics. Plumlee is already prior art in `docs/references.md`; this derivation does not
reopen that question.

### 9.1 Define the experiment, predictor, and parameter freedoms

Let `p(kappa, eta, xi) in R^m` stack **every output sample that enters the fitting loss**, over
all selected windows. Keep the records and sample weights fixed. Here `kappa in R^r` comprises
independent physical coordinates; `eta` comprises learned augmentation/interconnection parameters;
and `xi` comprises the actual encoder parameters or independently fitted initial states. Use
the actual encoder's shared parameterization, not arbitrary independent window states unless
that stronger nuisance model is intentional. All parameters are interior and locally free in
this first statement. Active constraints require their admissible tangent sets instead.

At a reference parameter tuple, define

```
S = D_kappa p,    R = D_eta p,    E = D_xi p.
p(reference + delta) = p(reference) + S delta_kappa + R delta_eta + E delta_xi + r(delta).
```

For a C2 predictor with `||D^2 p|| <= M` on a convex neighborhood containing the perturbation
segment, Taylor's integral remainder gives `||r(delta)|| <= M ||delta||^2 / 2`. This is a
finite-horizon bound: M may depend strongly on horizon and is not a stability claim.

Let `W > 0` be the chosen loss metric and `L^T L = W`. These are weights of the actual loss;
calling W an inverse noise covariance requires a separate statistical model. Write

```
A = L S,     B = L R,     D = L E,     N = [D B],
P_N = N N^dagger,         H_phys = A^T (I - P_N) A.
```

The pseudoinverse permits redundant neural and encoder parameters. Full rank of N is not
assumed. Fixed raw-parameter gauges must be removed or handled as identifiable combinations;
an inverse of a rank-deficient raw physical Gram matrix is not used.

### 9.2 Proposition 1: exactly when first-order cancellation is impossible

The following statements are equivalent:

1. `A v + N w = 0` implies `v = 0`.
2. `rank([N A]) - rank(N) = r`.
3. `(I - P_N) A` has full column rank r.
4. `H_phys` is positive definite.

**Proof.** A cancellation exists for a given v exactly when `A v` lies in `range(N)`,
equivalently when `(I-P_N) A v = 0`. This proves 1 iff 3. Adding A to N increases the span
by the dimension of the projected columns `(I-P_N)A`, proving 2 iff 3. Finally, for every v,
`v^T H_phys v = ||(I-P_N)A v||^2`, proving 3 iff 4. These arguments hold in arbitrary finite
dimensions, including rank-deficient N.

The same matrix arises from profiling out nuisance increments in the local fitting problem:

```
min_w ||b - A v - N w||^2 = ||(I-P_N)(b-A v)||^2,
```

where b is the whitened output residual at the reference model. Thus `2 H_phys` is the
profiled Hessian of this *affine* least-squares problem. For a nonlinear predictor and nonzero
residual it is a Gauss-Newton curvature, not automatically the exact Hessian.

This is a necessary-and-sufficient statement about the linearized experiment. It is not an
unqualified nonlinear local/global identifiability theorem. A zero derivative at an isolated
point does not by itself disprove nonlinear identifiability, and a local condition does not
guarantee convergence of the optimizer.

### 9.3 Corollary: orthogonality and a quantitative overlap bound

Profile the encoder first:

```
M_D = I - D D^dagger,
S_bar = M_D A,   R_bar = M_D B.
H_phys = S_bar^T (I - R_bar R_bar^dagger) S_bar.
```

**Proof of the last equality.** `range([D B])` is the orthogonal direct sum of `range(D)`
and `range(M_D B)`. Indeed, each column of B is its projection onto D plus its residual;
conversely `M_D B = B - D D^dagger B` is in the joint span. The two spans are orthogonal,
so `P_N = P_D + P_Rbar`. Substitution gives the result.

If S_bar has full rank, Proposition 1 is equivalent to

`range(S_bar) intersect range(R_bar) = {0}`.

Consequently `S_bar^T R_bar = 0` is **sufficient, not necessary**. With that condition,
`H_phys = S_bar^T S_bar`: learned freedom causes no additional first-order curvature loss
relative to the fit where encoder freedom was already allowed.

Let Q_S and Q_R be orthonormal bases for these two ranges, and define
`rho = ||Q_R^T Q_S||_2` (rho = 0 if R_bar is zero). Then

```
(1-rho^2) S_bar^T S_bar <= H_phys <= S_bar^T S_bar.
```

**Proof.** Put t = S_bar v. It lies in range(Q_S), hence
`||Q_R Q_R^T t|| <= rho ||t||`. Subtract its squared projection norm from `||t||^2`.
This proves the lower bound; removing a nonnegative squared norm proves the upper bound.
In finite dimensions rho < 1 is equivalent to a trivial intersection of the two ranges.

This yields a data-computable diagnostic that is directly tied to the theorem. A candidate
subspace penalty is `||Q_S^T Q_R||_F^2`, which upper-bounds rho squared. This is only a
candidate enforcement target: the bases depend on the model, rank changes can make their
projectors nonsmooth, and sufficiently expressive unconstrained networks may have a nuisance
tangent spanning the entire observed space. If that happens, the original model family remains
confounded; selecting a solution by regularization is distinct from changing that family.

### 9.4 Proposition 2: what a current-contribution penalty proves

Freeze S, E, W and compare local physical least-squares fits with and without a fixed additive
stacked output contribution d. Set `d_bar=M_D L d`, and assume S_bar has full column rank.
Then the difference between the optimal physical increments is

```
v_with - v_without = -S_bar^dagger d_bar.
```

**Proof.** Profile E as above. The normal equations before and after adding d have the same
left-hand matrix `S_bar^T S_bar`; their right-hand sides differ by `-S_bar^T d_bar`.
Multiplication by its inverse gives the formula. Therefore the displacement vanishes iff
`S_bar^T d_bar=0`, and

`||v_with-v_without|| <= ||Q_S^T d_bar|| / sigma_min(S_bar)`.

The last bound follows from the reduced SVD of S_bar. This is an exact zero-displacement
equivalence for the frozen affine fitting problem, not an unbiasedness theorem for nonlinear
joint training. The reference physical fit need not be the physical truth.

The **value-versus-variation distinction** is concrete: `d(eta)=eta s` has d(0)=0 but
`D_eta d(0)=s`. Thus perfect value orthogonality at a point can coexist with a learned tangent
equal to a physical direction s. If `S_bar^T d(eta)=0` holds as an identity for all nearby eta
and S_bar is fixed, differentiating gives `S_bar^T D_eta d=0`. If S_bar moves, the extra
term `(D_eta S_bar)^T d` must also be included. A finite soft penalty is not an identity.

### 9.5 Proposition 3: a soft penalty can still select physical parameters

The preceding distinction does not make value penalties useless. For a smooth vector penalty
residual c and beta > 0, consider the local quadratic/Gauss-Newton problem associated with

`J = ||L(y-p)||^2 + beta ||c(kappa,eta,xi)||^2`.

At the reference point define C_kappa, C_eta, C_xi as its Jacobian blocks and stack

```
A_beta = [L S; sqrt(beta) C_kappa],
N_beta = [[L E; sqrt(beta) C_xi], [L R; sqrt(beta) C_eta]].
H_beta = A_beta^T (I - N_beta N_beta^dagger) A_beta.
```

Proposition 1 applied to these augmented columns proves that the regularized local quadratic
problem has a unique physical increment iff H_beta > 0. Equivalently, every prediction-
cancelling direction with nonzero physical component must be detected by D c. Beta affects
conditioning, but multiplying the penalty rows by a nonzero scalar does not change this rank
criterion. This is **objective-based selection**, not identifiability from observations alone.
The statement is exact for an affine predictor and affine c; otherwise it concerns the local
Gauss-Newton problem. Residual-weighted second derivatives remain in the exact Hessian.

Example (structural): prediction `p=kappa+eta` is confounded. With `c=eta`, profiling eta
gives `H_beta=beta/(1+beta)>0`. With a penalty insensitive to eta, it remains zero. Thus the
relevant question for the proposed extension is not only whether c is small, but whether
its derivatives detect the cancellation routes, including the K and encoder routes.

For a frozen one-step point set at z=0 and affine physical correction `Kz+...`, the penalty
derivative with respect to K is zero. This preserves the existing structural blind-spot result.
A rollout penalty will include this derivative via recurrence, but detection of *all* relevant
directions still needs a rank/curvature check. Freezing and detaching anchors must be specified
as part of a local subproblem; it must not be silently treated as differentiating the fully
moving scalar objective.

### 9.6 Derive the sensitivities through the entire recurrent predictor

For dynamic parallel augmentation write

```
x_next = f_kappa(x,u) + a_eta(x,z,u),
z_next = g_eta(x,z,u),
y = h_kappa(x,u),              q=[x;z].
```

For any parameter group t, let `T_k^t = D_t q_k`. The chain rule gives

```
J_k = [[f_x+a_x, a_z], [g_x, g_z]],
T_(k+1)^t = J_k T_k^t + F_t(k),
D_t y_k = [h_x, 0] T_k^t + h_t(k).
```

Here F_t and h_t are **partial** parameter derivatives holding q and the recorded u fixed.
Initialize `T_0^t` by differentiating the encoder, including every actual dependency. Stack
the output derivatives at precisely the timestamps used by the loss. Direct parameter sharing
or output augmentation adds the corresponding derivatives; it does not invalidate the chain
rule. Learned states are already included in J_k. In the affine special case, `a_z=K`.

**Proof.** Differentiate `q_(k+1)=F(q_k,u_k,t)` and `y_k=h(q_k,u_k,t)` by the chain rule.
Starting with the encoder derivative, induction yields the derivative of every recursively
computed state and output. The chain rule is exact for derivatives of a smooth finite rollout;
the approximation appears only when using those derivatives for finite perturbations.

If the scheduler Y is a component of x, f_x includes its derivative. Holding Y fixed would
drop a first-order path. Variation of these Jacobians under further parameter/state changes
belongs to the remainder analysis. Also, `D_kappa p` of the **full augmented predictor** is
generally different from the standalone baseline sensitivity, since its recurrence includes
a_x, a_z, g_x, g_z. They must not be interchanged without an additional approximation argument.

The statement above assumes recorded exogenous inputs. If the actual fitting loss simulates a
feedback controller whose input changes with predicted output, include the controller state
and input-generation equations in q and F. An open-loop derivative cannot be claimed to be
the derivative of that closed-loop loss.

### 9.7 Independent checks and what they establish

Reproducible verification scripts:

- [SymPy and numerical checks](../code/finite_horizon_separation.py).
- [Independent MATLAB symbolic checks](../code/finite_horizon_separation.m).
- Results: [SymPy JSON](../code/finite_horizon_separation_sympy.json) and
  [MATLAB JSON](../code/finite_horizon_separation_matlab.json).

The scripts independently check weighted profiling, projector identities, redundant neural
directions, fixed-contribution displacement, the encoder cancellation counterexample,
value-versus-tangent separation, soft-penalty curvature, and the principal-angle bound on
explicit exact matrices. They also differentiate a nonlinear recurrent example both directly
and through the recurrence above. These checks support the written general proofs; finite
examples do not establish the arbitrary-dimensional quantifiers on their own.

The numerical example stacks several windows with a shared encoder parameter. It compares
the recurrence against PyTorch autodiff and centered finite differences, tests that omission
of the state-scheduling derivative is detected, and checks quadratic remainder scaling.
All numerical quantities here are **structural verification examples**, not gantry
measurements or evidence of physical recovery. No baseline-model transcription is claimed;
this is independent-engine checking of abstract algebra, not validation of MATLAB versus
Python gantry implementations.

**Verification result, 2026-09-06 (structural verification examples):** all 18 shared symbolic
checks passed independently in SymPy and MATLAB and their recorded boolean results agree.
Four additional numerical checks passed in Python. Maximum absolute derivative differences
were `1.11e-16` against autodiff and `7.63e-11` against centered finite differences. Omitting
the scheduling derivative produced an error of `0.149`, so that negative control was detected.
Halving the perturbation reduced the Taylor remainder by factors `3.983` and `3.992`, consistent
with the stated second-order scaling. These checks are on the generic example only.

Reproduce from the repository root:

```text
conda run -n GraduationProject python scripts/gantry/orthogonality/code/finite_horizon_separation.py
matlab -batch "addpath('scripts/gantry/orthogonality/code'); finite_horizon_separation"
```

**2026-09-07, closed-loop verification with an internal learned state.** The checks above use
an open-loop generic example. A second, closed-loop verification now exists and is reported in
[proofs/nonlinear-closed-loop-orthogonality.md](proofs/nonlinear-closed-loop-orthogonality.md):
a nonlinear model with a physical state, a learned state, a controller state and a shared
encoder, verified independently in MATLAB R2025a and SymPy 1.14.0. **60 shared checks, all
true in both engines, no disagreements, no undecided identities.** That report is a
verification record and owns no theory; this file remains the live theory/evidence record.
Its results bearing on Section 9 and on the implementation plan:

- The sensitivity recurrence of Section 9.6, extended with the controller state as Section 9.6
  requires, matches direct differentiation of the expanded closed-loop prediction as an exact
  identity, and matches central finite differences at a 200-step horizon to `6.2e-10` relative.
- Dropping the controller-state coupling is detected (relative error `0.27`), so the closed-loop
  paths are load-bearing rather than decorative.
- `Q_0' M_E = Q_0'` and the zero-residual-iff-orthogonality equivalence are exact identities,
  as is the penalty value bound. **The bound still says nothing about what training attains.**
- The ANN-only routed update has demonstrably nonzero omitted `grad_kappa V` and `grad_xi V`,
  so it differs from the gradient of `J_base + V`. **Correction, 2026-09-08:** this check does
  not establish failure of mixed-partial symmetry or exclude every scalar potential. That
  stronger claim requires a nonzero physical/augmentation or encoder/augmentation mixed
  derivative. The existing test results are unchanged; their interpretation is narrowed.
- The zero-latent pointwise penalty is blind to the latent readout as an identity, while the
  rollout residual has nonzero value and nonzero derivative in the readout gain. This is the
  explicit witness for the Section 9.5 blind-spot statement.
- A hard equality constraint admits a feasible **nonzero** correction that is retained at the
  constrained optimum and beats the ANN-off objective, found identically by `fmincon` and
  `SLSQP`. The constraint Jacobian is **rank 2 of 3 at the ANN-off point**, which is why
  "ANN-off is feasible" is not evidence. This is numerical feasibility in a 4-parameter toy,
  not efficacy and not parameter recovery.

**Still required before claiming an implemented method:** assemble S, R, E and D c for the
actual predictor and its actual open/closed-loop loss; verify their directional derivatives
against gantry rollouts; assess conditioning/rank with the encoder included; choose an
implementable penalty whose local curvature detects the relevant cancellation directions;
then run the joint-training comparison. No training regularizer was changed in this derivation.

## 10. Status of every claim in this file

| Claim | Level |
|-|-|
| Trajectory-level orthogonality already exists | **Cited**, read in full (`kon2023xray`, `zhao2023parity`) |
| No construction covers a stateful learned term | **Graded negative**, three mechanisms, strong |
| The pointwise penalty is the `s = 0` parity case | **Derived** here from a verified construction, not stated in any source |
| 21-dimensional invisible subspace, 4% leakage | **Measured** (T4), on a constructed exploit, not a trained network |
| Added states are load-bearing (`B/A` ~ 2) | **Measured**, causal ablation on two trained checkpoints, Sect. 4 |
| Realised `||x_aug||` magnitude | **Partially measured, superseded.** Instrumented run killed before convergence; one validation pass gave `||x_aug|| ~ 1.6e-2` on a barely-trained model. The question it stood in for is answered better by `r_state = 0.905` on a converged checkpoint (Sect. 7c). |
| UUB ceiling for universal approximators | **Cited**, read in full (`cho2023uub`) |
| Negation observed with a dynamic augmentation in the lineage's own results | **Quoted**, verified against the PDF 2026-09-04 (`hoekstra2026lfraug` p11) |
| `beta` selection is the successor's motivation for dropping the penalty | **Quoted**, verified against the PDF 2026-09-04 (`gyorok2026obc` p2, p9) |
| Longer windows leave more room | **Derived** here, dimension count only; not verified numerically |
| Pointwise constraint leaves `K` entirely free | **Proved** symbolically (`affine_leakage.py` A), constraint imposed as an identity on the slice |
| A state-path leakage term exists that the static case does not have | **Proved**, with a `K = 0` control separating it from propagation |
| `Lam_T(i) = sum_j A^(T-1-j) K G^(j-1-i)` is that term | **Proved**, closed form verified against the rollout at `T = 2, 3, 4` |
| (EXT) `Psi_T^T A^a K = 0` annihilates it and reduces to Gyorok at `n_a = 0` | **Proved** symbolically |
| (EXT) enforced by construction forces `K = 0` once `span(Psi) = R^nx` | **Proved** by rank/Cayley-Hamilton, but **posed in the wrong space** (Sect. 7c). Superseded. |
| The extension must therefore be a penalty, not a construction | **Withdrawn inference.** The preceding condition was posed in the wrong space; neither enforcement route is ruled out. See Section 9. |
| `rank(Psi_w) = 6` per window, so a per-window projection is vacuous | **Measured** (48 windows, T=25 and 100) |
| Stacked `Psi` has rank 10 of 288, matching the identifiable-combination count | **Measured** |
| ~90% of the augmentation's windowed contribution flows through `x_aug`, **in normalised state space** | **Measured**, unregularised checkpoint 81262. Quote with the space. |
| In OUTPUT space the latent share is 43%, not 90% | **Measured**, trained checkpoint 81758 (Sect. 7d) |
| `ell_lat = 0.957` at a trained checkpoint, against chance 0.037 | **Measured** (Sect. 7d). Resolves the untrained-testbed reading that suggested the latent route was already orthogonal |
| Both routes are aligned (`ell` 0.93 / 0.96 / 0.94), so neither can be left unpenalised | **Measured** (Sect. 7d) |
| `rank(S) = 10 of 14` through the FULL augmented rollout in output space | **Measured** (Sect. 7d), independent of the earlier one-step state-space calculation |
| Dynamic augmentation beats a static one on held-out prediction at every horizon | **Measured**, testbed, 1.11 to 1.28x (Sect. 7e) |
| The latent states do NOT learn the absorber dynamics (`rho(G) = 0.18`, ~12 ms against a 314 ms resonance) | **Measured**, testbed (Sect. 7e) |
| The extension rotates rather than shrinks (`ell` 0.832 -> 0.250) | **Measured**, testbed (Sect. 7f) |
| A plain size penalty MATCHES the projection on parameter error (0.0932 vs 0.0950) | **Measured**, testbed (Sect. 7f). The tie is unbroken and blocks the efficacy claim |
| Adding the pointwise term to the extension changes nothing | **Measured**, testbed (Sect. 7f), but the test could not detect the coverage benefit it exists for |
| That contribution is in-span 3.9x more than chance (above the 99th pct of the null) | **Measured**, randomisation null preserving per-window norms |
| The penalty would suppress it | **Untested.** Needs a joint=True orth on/off pair |
| **Section 0 framing** | |
| Orthogonality leaves the parameter estimate no worse than the baseline-alone fit | **Derived**, normal equations; both fits give the same `(Phi^T Phi)^-1 Phi^T Delta` |
| Orthogonality cannot distinguish negation from legitimate overlapping physics | **Structural**, inherited from the approach; not introduced by the extension |
| The true parameter value is not well defined under a nonparametric discrepancy | **Cited** (`xiexu2020projected` p6; uniform across `tuo-wu2015`, `tuo2019projected`, `wang2022`) |
| **Section 9 finite-horizon separation** | |
| Prop. 1 equivalences for first-order non-cancellation | **Proved** in text; checked on explicit matrices in both engines |
| Corollary: orthogonality and the overlap bound, encoder profiled out | **Proved** in text; principal-angle bound attained in both engines |
| Prop. 2: a current-contribution penalty controls displacement, not tangents | **Proved** in text; `zero_value_not_zero_tangent` fires in both engines |
| Prop. 3: a soft penalty can still restore local curvature | **Proved** in text; `blind_penalty_cannot_restore_curvature` is the negative control |
| Sensitivity recurrence incl. hidden states, encoder, state-dependent scheduling | **Verified numerically**, 1.11e-16 vs autodiff, 7.63e-11 vs central differences |
| Omitting the scheduling derivative is detectable | **Measured**, error 0.149 (negative control) |
| Remainder is second order | **Measured**, halving ratios 3.983 and 3.992 |
| All Section 9 numbers hold for the actual gantry predictor | **Not established.** Generic examples only, self-labelled in both JSONs |
| Rebuilding the basis costs 0.45 s, not the ~6 min on record | **Measured** (Sect. 7g); the 6 min was per-point reverse mode |
| Forward and reverse mode agree to 4.5e-16 on the basis Jacobian | **Measured** (Sect. 7g), checked before the timings were trusted |
| The basis must carry no gradient to `theta` (D-182) | **Structural**, with an executable check in `testbed_adapter.py`; not a measured result |
| Re-anchoring the basis per epoch is affordable | **Follows** from the 0.45 s measurement. It remains a DIFFERENT algorithm and is not adopted |
| Per-epoch basis drift DECAYS on the testbed (7.4 -> 5.6 -> 0.8 deg) | **Measured** (Sect. 7h), reduced setting, single seed |
| Re-anchoring is indistinguishable from freezing on the testbed | **Measured** (Sect. 7h). Under-converged; gantry drift is 5x larger, so not transferable |
| Net basis rotation understates the excursion (2.2 deg net vs 7.4 deg per rebuild) | **Measured** (Sect. 7h); argues for per-rebuild logging |
| The geometry materialised a dense NxN weight, 415 MB at gantry scale | **Fixed** 2026-09-07 (Sect. 7h); would have been fatal on the gantry |
| The OOM's cause | **Not localised.** Does not reproduce at reduced size; the dense-weight bug is sufficient at gantry scale but not at the testbed's N |
| **SUSPECT** Sect. 7d output-space results (`ell_lat = 0.957`, 43%) | **WITHDRAWN pending re-measurement.** `Cd` never applied; measured on normalised states, open loop, no encoder |
| Sect. 7e memory horizon table | **CORRECTED and re-measured.** Shared window manifest. Dynamic now WORSE at T=32 (0.985) and the advantage GROWS with horizon to 1.345 |
| **Sect. 9.7 closed-loop verification, 2026-09-07** | See [the report](proofs/nonlinear-closed-loop-orthogonality.md). Structural, generic model only |
| **Revised-ownership verification, 2026-09-08** | See [the report](proofs/trajectory-orthogonality-verification-2026-09-08.md). Small cases structural; gantry rows data-computable, no simulation truth read. **STATUS: the implementation is NOT validated** |
| `compressed_nuisance_basis` shipped computing `U_w^T J_w` instead of `(U_w^T Gamma_w) J_w` | **DEFECT, found in review 2026-09-08, fixed.** Dimensionally invalid at gantry shapes, so V3 would have failed regardless of the machine. The suite missed it because the small reference implemented the construction a SECOND time and compared that private copy against explicit `E`, never calling the production function. Two independent references agreeing is not verification of a third implementation |
| A second defect introduced by that fix (`U_blocks.append(U)` dropped, so `Q_E` was all zeros at the right shape and rank) | **Caught within a minute** by the new regression test. A rank check alone would have passed it; the orthonormality and projector-distance assertions are what fired |
| Disjoint ownership (latent initialisation in eta, physical initialisation as nuisance) leaves the recurrence, the interventions and the projection identities intact. `p_off` is independent of EVERY augmentation parameter; `p_clamped` only of the latent INITIALISATION, since its direct route still depends on the ANN | **Proved** as exact identities in both engines, 23 of 23 shared checks in agreement, 0 undecided |
| `E_w = Gamma_w J_w` and the compressed range construction `E = U_G Z` | **Proved** symbolically; the compressed and explicit nuisance projectors agree to `5.5e-16` at a witness point |
| `D_ca p` nonzero and factorising as `(D_a0 p)(D_ca a_0)`; the penalty gradient reaches the latent initialisation | **Proved** in both engines. Supersedes the 2026-09-07 shared-encoder example, whose single scalar cannot express this |
| **`D_lambda D_eta V` is not identically zero, so the selectively routed update is the gradient of NO scalar objective** | **Proved** by witness in both engines. This is the mixed-partial witness the 2026-09-08 correction recorded as missing; the earlier nonzero-physical-gradient check was insufficient |
| The plain-MSE penalty differs from the sum convention by exactly `N` | **Structural.** An old sum-based beta does not transfer |
| Gantry P0/V0/V1/V2 on checkpoint 81757: block and encoder conversions output-preserving, groups disjoint by identity AND behaviourally, adapter loss identical to `fit_sys.loss`, clamped latent state exactly zero over 400 steps, `p_off` bit-identical under perturbation of all 18 augmentation tensors, every group derivative and the D-184 chain rule agreeing with central differences to `2e-8` or better | **Measured**, two independent window sets, float64, eager CPU |
| Gantry rank, spectra, conditioning, chunking and gradient-ownership results | **MEASURED 2026-09-08**, 2 windows (T1, T9), N=2400, float64, eager CPU, checkpoint 81757: 41 checks, 41 passed, 440 s. `rank(S)=10` of 14 raw log-coordinates, exactly the structural bound; `rank(Q_E)=12`; **`rank(Sbar)=10`, so encoder profiling removed NO supported physical direction**. Stable over a 0.1x/1x/10x tolerance sweep. This is the Sect. 6.5 branch `r_bar = r_S = 10`: the rank prerequisite for a ten-combination pilot PASSES on this set. NOT tangent separation, NOT recovery, and NOT the final window set |
| Penalty gradient ownership on the gantry | **MEASURED.** `g_lambda = g_xi_p = 0` exactly; `g_eta_d = 4.7e-06`, `g_eta_a = 8.8e-11`. The latent-initialisation gradient is nonzero, as the contract requires, but sits ~5 orders below the ANN gradient: 'reaches it' is established, 'reaches it usefully' is not |
| Exact chunking and checkpoint replay on the gantry | **MEASURED.** 2 chunks vs full stack: rel value `8.1e-13`, rel grad `3.9e-12`. Checkpointed gradient matches un-checkpointed exactly; the unsupported out-of-scope replay is detected at `1.0` |
| The trajectory penalty is reachable from a training run | **DONE.** `SSE_Interconnect_Composed.fit` wraps `optimizer.step` so the penalty lands between the closure returning and the update; Jan's `fit()` untouched, no loop duplicated, `traj_orth=False` bit-identical. `test_penalty_hook.py` 12/12 |
| The 2026-09-08 V3 stall was a resource failure | **WEAKENED.** The stage later completed on the SAME machine after a CUDA-init defect in the new `diag.py` was fixed. That defect post-dates the original stalls so it cannot be their cause, but 'a bigger machine' was not the answer and the memory-pressure reading is weaker than it appeared. Cause still not identified |
| Closed-loop sensitivity recurrence (physical + latent + controller + encoder) equals direct differentiation | **Proved** as an exact identity in both engines; `6.2e-10` relative against 200-step central differences |
| Dropping the controller-state coupling is detected | **Measured**, relative error 0.27 (negative control) |
| `Q_0' M_E = Q_0'`, zero-residual iff contribution orthogonality, penalty value bound | **Proved** symbolically in both engines. The bound constrains the residual GIVEN a small penalty value; it says nothing about what training attains |
| ANN-only routing fails mixed-partial symmetry, so `J_base + V` is not a jointly minimized objective | **Proved** on the example: `grad_kappa V` and `grad_xi V` are demonstrably nonzero |
| The zero-latent pointwise penalty is blind to the latent readout while the rollout residual is not | **Proved** as an identity, with the rollout residual's value and `d/dK` both nonzero at the reference |
| Exact batching: `V = beta||sum_b r_b||^2`, not `beta sum_b||r_b||^2` | **Proved**, with a constructed cross-chunk cancellation where the two differ |
| Latent rescaling leaves predictions, interventions and the penalty unchanged; `\|\|K\|\|` and `\|\|a\|\|` are NOT invariant | **Proved** in both engines. Confirms the standing rule to judge the augmentation at the OUTPUT |
| A hard constraint admits a feasible NONZERO correction beating the ANN-off objective | **Numerically shown** in a 4-parameter toy by two independent solvers to 11 significant figures. NOT efficacy, NOT parameter recovery, NOT a statement about the penalty method |
| The constraint Jacobian is rank 2 of 3 at the ANN-off point | **Measured.** This is why "ANN-off is feasible" may never be quoted as evidence |
| All of the above holds for the actual gantry predictor | **Not established.** Generic closed-loop model only, self-labelled in both JSONs |
| Sect. 7h accumulated drift | **CORRECTED** to 5.34 / 5.36 deg (was 2.23). Exceeds the last consecutive angle 6x |
| Sect. 7h consecutive-rebuild angles (7.43, 5.64, 0.83 deg) | **Measured**, unaffected. Confirmed NOT to bound accumulated drift: 0.83 vs 5.34 deg |
| The D-182 guard was verified | **WAS FALSE, now fixed.** The old check inspected only geometry buffers. Split into `assert_geometry_detached` (buffers) and `assert_gradient_policy` (backwards the penalty alone and requires forbidden groups empty). Verified to catch a violating caller the old one passed, and to catch an inert penalty |
| The 145x figure is a differentiation-mode comparison | **Overstated.** It is per-point reverse versus BATCHED forward on CPU, i.e. an implementation comparison |
| The three-intervention identity validates simulator semantics | **FALSE.** It is algebraically true for any three returned vectors and proves nothing about the rollout |
| Memory's advantage grows with horizon | **Measured** after the window fix (Sect. 7e). The earlier report that this prediction FAILED was itself the bug |
| Static augmentation is worse than no augmentation at short horizons | **Measured** (Sect. 7e), T=32 and T=64 |
