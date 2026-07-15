# Limits of Data-Silent Regularization (parallel to dissipativity-limits.md)

**Date**: 2026-07-11. **Purpose**: hold the proposed expressivity-preserving method
(`docs/data-silent-regularization-concept.md`) to the SAME scrutiny as the dissipativity family
(`docs/dissipativity-limits.md`). This is a critical self-audit, not a sales sheet: the method is NOT
limit-free, on one axis it is strictly WEAKER than dissipativity, and it might not work. Companion validation
(which tests exactly the risks below): `docs/augmentation-validation-design.md`.

**Framing.** Dissipativity restricts by ASSUMED dynamics class (a fixed, unverifiable wall). This method
restricts by DATA-INFORMATION (a moving wall that only bites where the data is silent). The method moves the
restriction to a more defensible place for an unknown system, but pays with: no hard guarantee, a tunable
soft boundary, new estimation failure modes, an analogy-not-theorem foundation, and excitation-limited
coverage that is unverifiable on real data.

---

## A. Guarantee limits (what it cannot guarantee)
- **A1. No for-all-weights guarantee at all.** It is a SOFT regularizer: it discourages drift, it does not
  forbid it. A trained model can still carry residual DC -> residual drift. This is WEAKER than passivity,
  which at least bounds velocity for every weight (`dissipativity-limits.md` A1). Training-conditional by
  construction; this is the explicit price of full expressivity (hard-guarantee XOR full-expressivity).
- **A2. Efficacy is conditional on the drift being confined to the UNEXCITED subspace.** If spurious drift
  appears in a WEAKLY-excited direction (not a clean null direction), `Pi_low` only partially catches it ->
  residual drift. Degrades gracefully with excitation, but there is no threshold guarantee.
- **A3. No structural backstop.** It relies on the ESTIMATOR not producing the drift. If the optimizer
  converges to a spurious DC anyway (bad local minimum, or the horizon/multiple-shooting term is mis-set),
  nothing structural arrests it.

## B. Expressivity limits (what it suppresses)
- **B1. It suppresses learning in low-information directions.** "Does not bias identifiable directions"
  (Rothenberg/Tikhonov) holds only in the IDEALIZED limit of a sharp null space. With a soft penalty and a
  finite cutoff it is a bias-variance tradeoff: a WEAKLY-excited REAL dynamic near the cutoff is suppressed
  if the penalty is too aggressive. So it DOES limit learning, data-adaptively rather than structurally.
- **B2. It introduces hyperparameters** (`beta`, the singular-value cutoff, any frequency weighting) that
  dissipativity's parameter-free hard constraint does not have. Mis-tuning -> suppress real weak dynamics
  (too hard) or allow drift (too soft). The cutoff MUST be data-derived (noise floor) to stay defensible;
  if it is not, it is a heuristic (flag per CLAUDE.md).

## C. Construction / estimation limits (failure modes dissipativity does not have)
- **C1. `Pi_low` misalignment.** The unexcited subspace is a LOCAL, Gauss-Newton, functional-space estimate.
  For a nonlinear residual, coupling can make the COMPUTED low-information direction differ from the TRUE
  drift direction -> it pins the wrong thing (suppresses real dynamics, or misses the drift). Dissipativity's
  constraint is global and estimate-free; this is a new dependency at the exact place the method claims to
  work.
- **C2. Locality and a moving target.** `Pi_low` is operating-point / trajectory-local and must be
  recomputed as training moves. The regularizer then chases a SHIFTING subspace, which costs compute and
  risks training instability / oscillation.
- **C3. Weight-space back-mapping.** Penalizing a FUNCTIONAL (output-space) direction requires mapping it
  back to weights through the Jacobian; the ANN's huge weight-space null space can interfere (many weight
  directions give the same output-space penalty). A real implementation complication.
- **C4. The nonlinear excited/null decomposition is itself unsolved.** Defining it cleanly and cheaply for a
  nonlinear residual is the open design task (concept note §7). If it cannot be, the method is not
  implementable as specified: the concept is ahead of the construction.

## D. Foundation & epistemic limits
- **D1. The Rothenberg/Fisher foundation transfers only as a PRINCIPLE, not a theorem** (weight-space
  over-parametrization -> use function space; static i.i.d. -> Ljung's dynamic PE version; global -> local).
  Rigorous only in the linearized mass-damper anchor (the DC force is a `~T^5`-tiny-information direction
  over a short window); the nonlinear-ANN case is synthesis by analogy, not proof. See the identifiability
  provenance discussion.
- **D2. The prior it pins toward (minimum-norm / zero) is a CHOICE**, not assumption-free. Defensible
  (minimum-commitment), but if the truth has genuine unexcited content, minimum-norm is wrong (though that
  content is unlearnable from this data regardless).
- **D3. Unproven either way.** Dissipativity's limits are PROVEN (Cauchy-Schwarz, KYP). This method's key
  risks (does `Pi_low` align with the drift? does it suppress weak dynamics?) are EMPIRICAL and TBD -- the
  validation might say no. We are replacing a proven-limited method with an unproven one.
- **D4. It is a regularizer, not a solution.** Fundamentally a bias-variance tradeoff. The HORIZON-
  conditioning (multiple shooting) is arguably the real workhorse and is NOT novel; the novel
  re-aimed-projection part is unverified.

## E. Data-dependence limits (the flip side of knowledge-free)
- **E1. Coverage is only as good as the excitation.** If the data does not inform a direction, the method
  pins it -> on FIXED real hardware data with poor excitation it cannot learn what the data does not show,
  and we cannot re-excite the machine. Dissipativity is knowledge-free-but-class-restricting; this is
  class-free-but-excitation-limited. A different limit, not no limit.
- **E2. The core mechanism is unverifiable on the real deliverable.** The subspace-correctness check needs
  known ground truth, which exists only in sim. On real Telica data we are back to held-out free-run BFR
  (indirect); we cannot directly confirm it guided into the correct subspace where it matters most.

---

## Which limits the validation is designed to expose (not hide)
`augmentation-validation-design.md` targets the falsifiable ones directly:
- **B1/B2 (suppresses real dynamics):** the "keeps the 130-180 Hz absorber" and "learns injected friction"
  checks; the injected-dynamics library with an over-aggressive-regularizer falsifier.
- **C1 (`Pi_low` misalignment):** the subspace-correctness metric (decompose known true dynamics into
  excited/null; check recover-excited + pin-null + no-bias) and the excitation ablation (subspace must move
  with the data).
- **A1/A2 (residual drift):** the 12 s position-envelope probe.
- **E1/E2 (excitation-limited, unverifiable on real data):** acknowledged as scope, not resolved: sim proves
  the mechanism, held-out real-data BFR is the deliverable.
Limits NOT resolvable by validation (own them as honest scope): D1 (analogy-not-theorem), D2 (minimum-norm
is a choice), D4 (it is a regularizer), E2 (unverifiable on real data).

## Honest comparison to dissipativity (symmetry)
| Axis | Dissipativity (`dissipativity-limits.md`) | Data-silent regularization (this doc) |
|---|---|---|
| Guarantee | proven, for-all-weights, but bounds velocity not position / restricts class | none (soft, training-conditional) -> WEAKER |
| Restricts | assumed dynamics CLASS (fixed, unverifiable) | data-unidentifiable DIRECTIONS (moving, defensible) |
| Expressivity | forbids classes (storage/friction/marginal) | suppresses low-info directions (soft, tunable) |
| New failure modes | none of the estimation kind | `Pi_low` misalignment, moving target, undefined for nonlinear |
| Foundation | proven (KYP, Cauchy-Schwarz) | analogy + linear anchor; unproven for the ANN |
| Knowledge | knowledge-free but class-restricting | class-free but excitation-limited |

**Bottom line.** It moves the restriction to a better place for an unknown system, but it is WEAKER as a
guarantee and it MIGHT NOT WORK. That is the honest status; the validation exists to find out, not to
confirm.
