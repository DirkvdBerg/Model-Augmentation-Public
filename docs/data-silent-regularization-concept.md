# Concept: Data-Silent Regularization (the drift constraint as a re-aimed orthogonal projection)

**Date**: 2026-07-11. **Status**: concept draft, unbuilt. **Context**: `docs/drift-diagnosis-status.md` §5m
(identifiability reframe, D-105); the expressivity requirement (user, 2026-07-11: no mechanism may restrict
the class of dynamics the ANN can learn, because the real system is unknown); `tasks/lessons.md` (rule:
hard-guarantee XOR full-expressivity). This note specifies the ONE kind of "constraint" that survives that
requirement.

## 1. The requirement this must satisfy
- **Full expressivity (non-negotiable).** The ANN must remain able to represent ANY residual dynamics; the
  true nonlinear system is unknown, so any forbidden class might be the true residual (unverifiable). This
  rules out every hard, for-all-weights, class-restricting constraint (passivity, dissipativity,
  net-impulse, NI, contraction).
- **No-drift.** The free-integrator (X/Y) free-run must stay bounded.
- **Consequence (a theorem, not a preference):** a for-all-weights no-drift guarantee and full expressivity
  are logically incompatible. So the no-drift must come from the ESTIMATOR, not the model class. It is
  training-CONDITIONAL, and that is the correct price of full expressivity.

## 2. Principle
The drift is a spurious force component (near-DC on the K=0 X/Y rows) that the short-horizon loss cannot see:
it lives in a LOW-INFORMATION / UNEXCITED direction of the estimation problem (§5m). The mechanism: pin
THAT direction toward minimum-norm; leave every data-informed direction untouched. Where the data informs
the force (any real dynamics, including dissipative friction if it is excited), the penalty is ~0 and the
ANN learns it freely. "Low-information" is a property of the DATA (loss sensitivity), not an assumption about
the dynamics, so the mechanism is knowledge-free and does NOT restrict the model class.

## 3. Mechanism (re-aim of the Gyorok orthogonal projection)
Gyorok's orthogonal projection (our C5 interpretability layer, `literature/Orthogonality/...`, catalog §A3)
penalizes the ANN output in the FP-MODEL subspace: `Pi_FP = Q Q^T` from `SVD(Phi)`, `Phi = d f_theta/d theta`,
cost adds `beta || Pi_FP f_ANN(X,U) ||^2`. It is a SOFT, targeted steering in a KNOWN subspace, and it
preserves expressivity outside that subspace.

Re-aim the TARGET subspace from "FP-spanned" to "data-silent":
1. Form the empirical sensitivity/Gram of the loss with respect to the ANN output over the trajectory (the
   empirical Fisher / regressor Gram of the routed X/Y force).
2. Its LOW-singular-value directions define the unexcited subspace `Pi_low` (the directions the data cannot
   determine).
3. Add `beta || Pi_low . (accumulated ANN force) ||^2` to the cost: pull the underdetermined,
   drift-relevant component toward minimum-norm, leave the informed component free.

Bayesian equivalent: a Tikhonov / GP prior weighted by the INVERSE data-information, which bites only where
information is ~0. Rogers-Friis (below) is this exact idea with a GP; we do it with the ANN plus a
data-adaptive ridge.

## 4. Pair with horizon-conditioning (also class-preserving)
Multiple shooting + a rollout-length / free-run-consistency term SHRINKS the unexcited set itself: it makes
the loss SEE the drift that the 0.1 s window misses (drift enters ~0.5 s, §3 of the diagnosis). The
regularizer of §3 then handles whatever residual low-information direction remains. Neither mechanism
restricts the model class; both act on the estimator.

## 5. Why it satisfies the requirement
- **Expressivity-preserving:** the penalty vanishes where the data is informative, so no representable
  dynamics is forbidden. Contrast the net-impulse / passivity blocks, which forbid a class for ALL weights.
- **Knowledge-free:** the target subspace `Pi_low` is computed from the data (loss sensitivity), not from an
  assumed property of the true dynamics. The prior it pins toward (minimum-norm) is the standard
  minimum-commitment choice for an unidentifiable direction, defensible, not a dynamics assumption.
- **Friction-permitting (the test the net-impulse block fails):** dissipative friction that the data
  EXCITES sits in an informed direction, so the penalty is ~0 there and the ANN learns it. Only genuinely
  unexcited components are pinned, and those are unidentifiable from this data regardless.
- **Marginal-preserving:** it adds no stiffness/damping to the position row; it only reshapes where the
  estimate is free.

## 6. What it does and does NOT guarantee
- **Does:** training-conditional no-drift (holds for the trained weights given the data); full model class
  retained; coexists with (is the same layer as) the interpretability orthogonal projection.
- **Does NOT:** a for-all-weights structural guarantee (impossible alongside full expressivity, §1); it does
  not manufacture information that the data lacks. If the true dynamics carry content in a genuinely
  unexcited direction, no method can identify it from this data; pinning to minimum-norm is the honest best.

## 7. Open design choices (resolve in a build, all data-derived, no magic constants)
- Definition of "accumulated ANN force" for the projection target (time-integral / low-frequency band of the
  routed X/Y force).
- The singular-value cutoff separating `Pi_low` from the informed subspace: derive it from the data (e.g.
  relative to the measured noise floor / information level), never a fixed constant.
- Optional frequency weighting (emphasize the near-DC band where drift lives) combined with the
  information weighting.
- **The `Pi_low` ESTIMATOR itself (added 2026-07-12, sweep Direction 12): Fisher-SVD vs ENSEMBLE
  DISAGREEMENT.** The limits doc (C1/C4) flags the Fisher-SVD construction as the unsolved step for a
  nonlinear residual. Alternative found in offline model-based RL [search-level, verify]: train k small ANN
  heads and use their DISAGREEMENT as the epistemic-uncertainty / data-silence signal, penalizing the output
  where disagreement is high (MOPO, Yu et al. NeurIPS 2020, arXiv:2005.13239 — with a pessimistic
  lower-bound guarantee shape; same quantity as the GP predictive variance, Schuet §II.G, and the Fisher
  null space — three formalisms of one object). Scale-proven, ANN-native, data-derived; candidate
  resolution of limits-C4. Decide at build time; primary-read MOPO first.
- **MEASURED UPDATE (2026-07-12, d16 `d16_pilow_spectrum.py`): the Fisher/Gram spectrum constructor of §3
  is REFUTED as the PRIMARY target on this system — C1 misalignment confirmed by structure.** On the K=0
  axes the integrator gain makes constant ANN-output offsets the MOST loss-informed directions per unit
  amplitude (Gram: DC probes 0.44–290 vs band probes 2.6e-3–1e-9; the drift direction ranks above 83% of
  the spectrum, ALL absorber-band directions rank below it). A low-information cutoff therefore pins the
  band before the drift — the opposite of the intent. What separates drift from the learnable dynamics is
  FREQUENCY (near-DC) and free-run consequence (the marginal integrator), not information content. **Build
  target: the DIRECT measured joint-DC direction (d14: joint, not per-row) with near-DC frequency
  selectivity — the "optional frequency weighting" above is the REQUIRED core, not optional.** R1 status
  unchanged: the pinned direction is measured from data, and the near-DC band selection follows from the
  KNOWN baseline's K=0 structure (physics of the model class, not an assumption on the unknown residual).
  The §2 phrasing "the drift lives in a LOW-INFORMATION direction" is corrected by measurement: it lives in
  a HIGH-gain, SHALLOW-optimum direction (d14: ≤2% at operational scale) whose damage is a free-run
  property the windowed loss cannot price. Ensemble disagreement remains the candidate for the RESIDUAL
  (post-pin) silent directions; it measures a different quantity than this Gram and is not refuted by d16.

## 8. Provenance
- **Rogers, Friis, "A Latent Restoring Force Approach to Nonlinear System Identification", arXiv:2109.10681
  (MSSP 2022) [PRIMARY-READ 2026-07-11].** Grey-box: known linear part + unknown residual force
  `m z_ddot + c z_dot + k z + f_hat(z,z_dot) = U(t)` (their Eq 5), with the unknown force modeled as a
  ZERO-MEAN GP with NO assumed functional form, `f_hat ~ GP(0, k(t,t'))` (their Eq 6), inferred by
  Kalman/RTS smoothing. This is the Bayesian sibling of the mechanism here: a PRIOR on the residual force,
  dominated by the likelihood where the data informs, regularizing only the data-silent part, full
  expressivity retained. Confirms the principle at primary source.
- **Pillonetto, Dinuzzo, Chen, De Nicolao, Ljung, "Kernel methods in system identification...", Automatica
  50(3):657-682, 2014 [NOT primary-read: Automatica only, not on arXiv].** The regularized-ID framework:
  Tikhonov / RKHS priors stabilize the ill-posed inverse WITHOUT restricting the model class; the kernel
  embeds smoothness/stability prior knowledge. Verify at source before thesis citation.
- **Gyorok, Hoekstra, Kon, Peni, Schoukens, Toth, orthogonal projection, L4DC 2025 (arXiv:2501.05842)
  [on disk].** The in-framework null-space steering this note re-aims. Same regularization layer, different
  target subspace.

## 9. Verification plan (sim first, expressivity is the key check)
1. **No-drift:** the regularizer drives the X/Y free-run drift to ~0 (12 s probe, envelope ratio ~1).
2. **Expressivity (critical):** the 130-180 Hz absorber band RMS is UNTOUCHED (the penalty must not suppress
   the informed residual).
3. **Friction-permitting (the discriminator):** inject a known dissipative friction in an EXCITED band and
   confirm the ANN LEARNS it (penalty ~0 there), where the net-impulse block structurally cannot.
4. **Data-derived target:** recompute `Pi_low` under a different excitation and confirm the target subspace
   moves with the data (it is not a fixed, assumed direction).
