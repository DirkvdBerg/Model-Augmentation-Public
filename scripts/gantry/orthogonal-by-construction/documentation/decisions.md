# Decisions: state-level orthogonal-by-construction augmentation

Local log for this subfolder. Two entries, both required before any code is written against
`state-level-obc-derivation.tex`.

Numbering is local (`OBC-n`) on purpose. `docs/decisions.md` currently holds two entries numbered
`D-186`, so no new global number is claimed here. When these move into the project log they take
the next free `D-` numbers after `D-187` and keep their titles.

---

### [OBC-1] The orthogonality condition is imposed on the state equation, and its implication for the accumulated output is necessary but not sufficient
**Date**: 2026-09-11

**What**: The condition is between the two state-update contributions of the additive structure,
stacked over an evaluation set, as in Sect. 4 of `state-level-obc-derivation.tex`. It is NOT
imposed on the measured output, and no claim derived from it may be worded as an output-level
guarantee over a horizon.

**Why**: Three reasons, heaviest first.
1. The supervisors set the level. Gyorok's model class is discrete-time input-output, a regressor
   on lagged measured signals; ours is state-space LPV-LFR. Recasting our model into that class is
   the expensive path, and the paper's own conclusion names the state-space extension as open,
   citing the absence of full-state measurement as the obstacle.
2. The reduction is clean at this level. With the additional states removed, the baseline linear
   in its parameters, a one-step predictor and `C = I`, the condition becomes Gyorok Eqs. (8),
   (10) and (11) term for term (Prop. 9.1 of the derivation, verified symbolically).
3. The alternative was excluded by the supervisors as a starting point, not by an argument here.

**The cost, stated rather than discovered later**: step-wise orthogonality does not transfer to
the accumulated measured output, for three independent reasons: the sensitivity recursion mixes
time steps through the closed augmented Jacobian; the output matrix retains 3 of the 6 physical
rows and a projection does not preserve orthogonality; and the additional-state write is
unconstrained yet re-enters the physical rows one step later. Verified by symbolic counterexample
on the surrogate, where the one-step condition holds EXACTLY while the six-step output
contribution has a nonzero inner product with the baseline parameter sensitivity
(`code/logs/surrogate_obc_symbolic.log`, check 4).

**Ruled out**:
1. Any wording asserting that the construction prevents the augmentation from reproducing or
   negating baseline dynamics over a horizon. The claim is one-step non-overlap.
2. Starting at the input-output level.
3. Treating the reduction as complete without naming the third reduction (one-step predictor with
   `C = I`). Without it the paper's stacked object is the output and ours is the state update,
   and those are not the same object.

**Constrains**: every claim wording for this method in the thesis; the acceptance criterion for
any implementation, which must report the trajectory residual and not only the reference-set
residual.

---

### [OBC-2] The auxiliary coefficient is frozen on an evaluation set taken from the previous epoch's augmented rollout, not from the baseline-only rollout
**Date**: 2026-09-11

**What**: The projection coefficient `eta_aux(eta)` is fitted once per pass on a point set fixed
before the pass. That set is the augmented closed-loop rollout of the PREVIOUS epoch, refreshed
between epochs. It is NOT the baseline-only rollout, and it is NOT recomputed along the current
trajectory.

**Why**: Four reasons.
1. Fitting along the current trajectory is non-causal. The coefficient would depend on the learned
   writes at later steps in the window, which depend on the coefficient. There is no exact fix
   inside one forward pass; Gyorok never meets this because his learning component is static.
2. A frozen coefficient is sufficient for the payoff. The block-Jacobian argument behind the
   paper's Theorem 18 needs only that the projector be independent of the learned parameters, not
   exactness against a regressor recomputed at the current iterate (Remark 2.2 and Prop. 5.2 of
   the derivation). Freezing is therefore admissible rather than a concession.
3. The paper licenses an auxiliary set explicitly. Immediately after its Lemma 3 it states that
   the coefficient may be built from any auxiliary evaluation of the regressor, on a synthetic set
   or a subset of the estimation data, and that using the whole training set is a convenience.
4. The baseline-only rollout is disqualified by a proof, not a preference. Nothing writes the
   additional states there, so they are zero at every point, and the fitted coefficient is then
   EXACTLY independent of every parameter on the latent path: all six such parameters give a zero
   derivative on the surrogate (`code/logs/surrogate_obc_symbolic.log`, check 3; Prop. 7.1 of the
   derivation). The projection would charge nothing on exactly the feature this extension
   introduces. This is the failure the input-output case cannot exhibit, and it is the reason the
   reference set is a decision at all.

**The cost**: the set depends on the augmentation parameters ACROSS epochs, so the scheme is a
fixed-point iteration whose convergence is not established. Within a pass the set is fixed, so
exactness on the set and the block structure both hold. Gyorok's predecessor paper contemplates
the same per-epoch refresh for its projection matrix, so the pattern has precedent but not a
proof.

**Ruled out**:
1. The baseline-only rollout, for reason 4.
2. Recomputing the projection every forward pass and differentiating through it, for reason 1. It
   also buys nothing reason 2 does not already give.
3. Treating `eta_aux` as a constant in the backward pass. Differentiating THROUGH it is what makes
   Prop. 5.2 hold; stopping the gradient there destroys the property the construction exists to
   provide.
4. Falling back to a penalty at the state level. That method exists, is verified, and answers a
   different question.

**The independent check, not an alternative**: a designed auxiliary set with the latent
coordinates drawn over a justified region, per candidate (R1) of the derivation. It is causal and
independent of the parameters, and it exists to show that any result does not depend on the
refresh. Its own cost is that the points leave the data manifold, so the protected set describes a
designed region that would have to be justified.

**Constrains**: the reference set must span the scheduling range that any held-out validation
position falls in, since the regressor depends on the scheduling variable and exactness holds only
over the range the set visits. Any implementation must report the residual on the CURRENT
trajectory next to the residual on the reference set, and both next to the norm of the unprojected
contribution, so a reader can see a cancellation rather than a small input.
