# Validation Design: expressivity-preserving augmentation (methods + how we prove they work)

**Date**: 2026-07-11. **Status**: design note, unbuilt. **Relation to existing docs (no duplication):**
the METHOD concept is `docs/data-silent-regularization-concept.md`; the diagnosis and solution families are
`docs/drift-diagnosis-status.md` (§5m families, §5g phases, §5e.2 D-D1/D-D2); evaluation/reporting plumbing
(metrics, folders) is `docs/eval-restructure-plan.md`. THIS doc is the scientific VALIDATION design: what
would actually prove the methods work, with the injected-dynamics library, the subspace-correctness metric,
and the excitation ablation. Paper quotes are transcribed from on-disk PDF text layers (primary-read where
tagged); re-verify character-exact before thesis use.

## 1. What we are validating (the claims)
The deliverable is a FULLY-EXPRESSIVE learned augmentation on the free-integrator X/Y axes that does not
drift, without restricting the class of dynamics it can represent (user directive 2026-07-11: no
class-restricting constraint, because the real system is unknown). Claims:
- **C1 No-drift:** free-run position bounded on X/Y.
- **C2 Expressivity preserved:** the mechanism does NOT suppress genuine dynamics the data informs.
- **C3 Learns dissipative friction:** with excited friction, the ANN captures it.
- **C4 Correct subspace:** the regularization acts on the UNEXCITED direction and leaves EXCITED directions
  free (no bias in identifiable directions).
- **C5 Knowledge-free / general:** works across qualitatively different injected dynamics with no per-case
  tuning; the acting subspace is data-derived.

## 2. Epistemic status: this is EMPIRICAL proof, not a theorem
Because we chose full expressivity, a for-all-weights no-drift guarantee is logically impossible
(a universal approximator can represent a drifting force). So no-drift is TRAINING-CONDITIONAL and validation
is necessarily empirical: a controlled demonstration on data with KNOWN ground truth (injected dynamics) plus
held-out real data. The structural constraints we rejected would have given a theorem at the cost of
expressivity; we deliberately trade the theorem for expressivity and pay for it with empirical validation.

## 3. Foundation, with quotes (why "excited vs unexcited subspace" is well-posed)
The whole design rests on: a fit has IDENTIFIABLE directions (the data informs them) and UNIDENTIFIABLE /
unexcited directions (it does not), determined by the information matrix. This is classical, and its origin
is econometrics/statistics (Rothenberg 1971), ADOPTED by control ID (Ljung; Soderstrom-Stoica), not
originally a control result.
- **Identifiability <=> information-matrix rank [PRIMARY-READ: Little, Heidenreich, Li, "Parameter
  Identifiability and Redundancy: Theoretical Considerations", PLoS ONE 5(1):e8915, 2010, p.4]:**
  > "Rothenberg proved that if the Fisher information matrix, I = I(theta), in a neighborhood of theta is of
  > constant rank and satisfies various other more minor regularity conditions, then theta is locally
  > identifiable if and only if I(theta) is non-singular."
  The UNIDENTIFIABLE / unexcited directions are exactly the null space (rank deficiency) of the information
  matrix. That is the `Pi_low` subspace the method regularizes.
- **Persistency of excitation (control setting) [De Persis, Tesi, "Formulas for Data-driven Control",
  arXiv:1903.06842; abstract-level read]:** persistently exciting data can represent the system's
  input-output behavior (Willems' fundamental lemma). PE is equivalent to positive-definiteness of the
  information matrix, i.e. an empty unexcited subspace.
- **Prior on the residual force, dominated by data [PRIMARY-READ: Rogers, Friis, "A Latent Restoring Force
  Approach to Nonlinear System Identification", arXiv:2109.10681, MSSP 2022, Eq 5-6]:** the unknown force is
  `m z_ddot + c z_dot + k z + f_hat(z, z_dot) = U(t)`, with
  > "f_hat(z, z_dot) ~ GP(0, k(t, t'))"
  a zero-mean GP with NO assumed functional form: "The GP provides a prior over a function ... the function
  is updated in a Bayesian [manner]." This is the expressivity-preserving regularizer (prior where data is
  silent, likelihood where it informs) that our data-silent regularizer is the parametric analog of.

## 4. The core comparison: 3 models x 3 claims
Train all three on the SAME injected-dynamics data; each model isolates what it proves.

| Model | C1 No-drift | C2 Keeps informed dynamics | C3 Learns dissipative friction |
|---|---|---|---|
| Unconstrained ANN | drifts | yes | yes, but drifts |
| Net-impulse block (ours, validated D-B/D-C) | yes | yes | **NO (structurally forbids DC)** |
| **Data-silent regularized ANN** | yes | yes | **yes (the claim)** |

Decisive cell: bottom-right vs the net-impulse block. **Why the net-impulse block CANNOT [see
`docs/dissipative-block-spec.md` and 5f]:** its output is an exact difference `g_k = psi(z_k) - psi(z_{k-1})`,
so the accumulated impulse `Sum g = psi(z_N) - psi(z_0)` is bounded for all weights and any DC-carrying
force (Coulomb friction over asymmetric motion) is forbidden. It is impulse-based, blind to the sign of
power, so it forbids dissipative friction along with injecting drift.

## 5. The injected-dynamics library (each member falsifies a specific failure mode)
Small and principled, NOT open-ended. Each member is chosen because passing it falsifies a specific wrong
method or exposes a specific weakness.

| Injected dynamics | Tests | Falsifies (if passed) |
|---|---|---|
| Zero-DC oscillatory (150 Hz absorber) | keeps informed oscillatory residual | an over-aggressive regularizer (suppresses real dynamics) |
| Dissipative DC friction (Coulomb/Stribeck, EXCITED) | learns a DC-carrying dissipative force | **the net-impulse block** |
| Position-dependent (cogging ~ periodic in q) | learns LPV/position-dependence | input-restriction fixes that drop position |
| Hysteretic / memory (LuGre bristle, backlash) | learns state-dependent memory | memoryless/static assumptions |
| Active-LOOKING residual (e.g. baseline over-subtraction in these coordinates) | learns dynamics a passivity constraint would REFUSE | **the passivity/dissipativity constraints** (proves we did not re-restrict the class) |

The last row is the sharpest for the expressivity requirement: it demonstrates the method learns dynamics
the rejected structural constraints forbid. **Why passivity would refuse it [PRIMARY-READ: van der Schaft,
"Cyclo-dissipativity revisited", arXiv:2003.10143, Remark 3.4, p.8]:** an indefinite-storage (marginal)
relaxation gives
> "the Lyapunov function ... is no longer nonnegative. Hence in principle only instability results can be
> inferred"
i.e. passivity/dissipativity structurally exclude an energy-injecting (active-looking) residual, whereas the
data-silent method learns it if the data informs it. **Why an ISS/attractor method would also fail
[PRIMARY-READ: DiLaR-PINN, arXiv:2604.18277, Prop 3, p.4]:** its guarantee requires an ISS baseline
(`grad V^T f_phys <= -alpha3 + sigma`), which the free integrator is not, so it excludes our case by
assumption.

## 6. Per-case protocol (how each library member is run)
1. **Identifiability precondition (from the DATA, per case):** confirm the injected dynamics is EXCITED,
   i.e. its direction has non-negligible information (Fisher/sensitivity of the loss w.r.t. the routed X/Y
   force). If it is NOT excited it is unidentifiable, the method SHOULD pin it, and "did not learn it" is
   correct, not a failure. Never claim "learns X" on data that does not excite X.
2. **Decompose the KNOWN true injected force** (available in sim) into the data-derived EXCITED (learnable)
   component and the NULL (unidentifiable) component.
3. **Check the method:** (a) recovers the excited component (low error there), (b) pins the null component
   toward minimum-norm (no drift), (c) does NOT bias the excited component (the Rothenberg/Tikhonov no-bias
   property, C4). This triple is "guides in the correct subspace", measured.

## 7. The decisive test: excitation ablation
Run the SAME dynamics under two excitations so its direction moves from unexcited to excited:
- **narrowband** (dynamics unexcited) -> method PINS it (cannot learn it, honestly) -> no drift, not
  recovered;
- **broadband/low-freq** (dynamics excited) -> method SWITCHES to LEARNING it.
If behavior tracks the excitation (pin when unexcited, learn when excited) for every library member, that
proves the method acts on the DATA-DERIVED subspace, not a fixed assumption or a hidden dynamics prior. This
is the strongest evidence for C4+C5 and directly connects to the excitation discussion (`drift-diagnosis-
status.md` §5m excitation note): a friction/dynamics test on dynamics-blind data is vacuous.

## 8. Metrics and thresholds (data-derived, defensible)
- **No-drift (C1):** free-run position-ENVELOPE growth over 12 s (RMS|q| late/early window ratio ~1 bounded,
  >~1.2 drift). NOT a slope/velocity proxy (a bounded oscillation trips those; see lessons).
- **Keeps informed dynamics (C2):** the informed band RMS (e.g. 130-180 Hz absorber) is UNCHANGED under the
  regularizer.
- **Friction/dynamics capture (C3):** held-out free-run RMS/BFR; AND directly, the learned-vs-true residual
  force error (sim ground truth) in the EXCITED subspace.
- **Subspace correctness (C4):** the three-part check of 6.3, reported as excited-recovery error,
  null-pinning magnitude, and excited-subspace bias.
- **Threshold rule:** acceptance judged against the MEASURED NOISE FLOOR, never against the sim-only true
  magnitudes (no oracle threshold). The true force is used only to DECOMPOSE subspaces, not to set the bar.

## 9. Honest limits
- **Coverage, not universal proof.** The library spans PLAUSIBLE dynamics; the real machine may carry a type
  we did not inject. Sim validates the MECHANISM; held-out real-data free-run BFR is the DELIVERABLE.
- **Include adversarial members** (DC in a nearly-unexcited direction; the active-looking residual) designed
  to BREAK the method, not only easy wins. A validation that only shows successes is weak.
- **The nonlinear-subspace computation is itself under test.** Defining the excited/null decomposition for a
  NONLINEAR residual (function-space sensitivity, not a fixed matrix) is the open design task from the
  concept note; these experiments validate that construction too.
- **Training-conditional.** Results certify the trained model on representative data, not all weights.

## 10. Build order (when authorized; still unbuilt)
1. Injected-dynamics sim harness (known force, selectable dynamics type + excitation). Enables everything.
2. The data-derived subspace decomposition (excited vs `Pi_low`) for a nonlinear residual; unit-verify on a
   known linear case first.
3. Run the 3-model x library grid + the excitation ablation; report the §8 metrics.
4. Only then, real Telica data (held-out free-run BFR).
