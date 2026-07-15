# Decision: stay OPEN-LOOP; the drift must be SOLVED, not hidden

**Date**: 2026-07-11. **Status**: direction decision (user, this thread). This doc captures WHY we stay
open-loop and what the sole admissible open-loop path is. It is the conclusion of the passivity/augmentation
literature thread. Main working doc: `docs/drift-diagnosis-status.md`. Verdict + requirement table:
`docs/augmentation-literature-verdict.md`.

## The decision
1. **Closed-loop is REJECTED as the solution.** It HIDES a bad model: the servo bounds position for ANY
   model, so a spurious model DC and a correct model both stay at setpoint. Closed-loop no-drift certifies
   the LOOP, not the MODEL. We do not want a method whose "success" is masking.
2. **The OPEN-LOOP free-run metric is KEPT, as a feature.** It EXPOSES drift and bad fits instead of masking
   them. That is exactly why it is the deliverable, and why both the velocity-domain loss and closed-loop
   (which change/remove the metric so the drift stops showing) are demoted.
3. **The drift must be SOLVED (remove the spurious DC from the MODEL), not hidden** (bound position while the
   model stays wrong). The open-loop metric is the honest judge of solve-vs-hide.

## Supporting constraints (user, 2026-07-11)
- Closed-loop "will just hide a bad model" (user).
- Jan's augmentation framework is currently OPEN-LOOP; staying open-loop aligns with it.
- The supervisors named the velocity-domain loss as a LAST RESORT; their stance on closed-loop is unknown,
  so closed-loop is not assumed acceptable (it is at least as much a metric-change as velocity-domain).

## Consequence (honest)
Under OPEN-LOOP + position-domain + full expressivity (keep friction AND the marginal mode):
- **No structural for-all-weights no-drift guarantee exists** (the expressivity-XOR-guarantee impossibility;
  see `augmentation-literature-verdict.md` and the requirement table). A structural guarantee would require
  sacrificing friction (net-impulse) or the marginal mode (contraction) -- both rejected.
- Therefore **requirement 4 (non-drift) can only be EMPIRICAL open-loop** -- which is precisely what the
  open-loop free-run metric tests. This is not a cop-out; it is the honest ceiling under these constraints.

## The sole admissible open-loop path: the ESTIMATION route (solves, does not hide)
These fix the MODEL (remove the spurious DC from the estimate) and are verified honestly by the open-loop
metric. All are in-framework, position-domain, expressivity-preserving:
1. **Data-silent regularization** = Gyorok/Jan orthogonal projection RE-AIMED at the data-unexcited subspace.
   Same regularization machinery as the interpretability contribution (C5), different target subspace -> it
   lives INSIDE Jan's framework and IS the thesis contribution, not a bolt-on. See
   `docs/data-silent-regularization-concept.md` (+ its `-limits.md`).
2. **Horizon conditioning** = multiple shooting + continuity penalty (the actual matching term, not just
   longer nf); makes the loss SEE the drift so the optimizer never inserts a DC. (Distinct from just
   increasing nf.)
3. **Re-excitation** where we control the sim input (make the DC/near-DC direction identifiable). Not
   available for fixed real hardware logs.
4. **Grey-box friction in `f_base`** = explain the legitimate DC physically, leaving the ANN residual
   zero-DC (D-A supports: dominant residual is zero-DC). Real-data step, not for the current sim.

## What is NOT the path (and why)
- **Closed-loop** -- hides (this decision).
- **Velocity/acceleration-domain loss** -- supervisor LAST RESORT; changes the metric so drift stops
  showing; not adopted without explicit go-ahead (standing constraint, `drift-diagnosis-status.md` top).
- **Structural constraints** (dissipativity, net-impulse, contraction, NI-by-construction) -- restrict the
  model class; reject friction or the marginal mode, or need a closed-loop partner. See
  `docs/dissipativity-limits.md`.

## The honest complication and the FIRST real step
The strongest "conditioning fails" evidence (Optuna 69399) is CONFOUNDED: every trial ran at lr=1e-3 (the
D-101 bug, "configured lr silently ignored"), so its nf/lr sweep never applied the intended values. So we
have NOT cleanly tested whether the estimation route controls the drift.
- **First step (defensible, cheap):** a CLEAN position-domain re-run at the correct post-D-101 lr, with
  horizon conditioning (multiple shooting + continuity) and proper excitation, judged by the open-loop
  free-run metric. If that alone controls the drift, no exotic add-on is needed.
- **If it does not:** add the data-silent projection (the re-aimed orthogonal projection) as the principled,
  in-framework regularizer.
- Do NOT conclude "conditioning fails" from 69399; re-run it correctly first.

## Related documents
- `docs/drift-diagnosis-status.md` -- main working doc (diagnosis, solution space §5, standing constraints).
- `docs/augmentation-literature-verdict.md` -- exhaustive requirement table + verdict (no single method has
  all four; Tustin-Net best real partial; closed-loop = the one setting all four hold, but it hides).
- `docs/dissipativity-limits.md` -- every dissipativity/passivity/NI restriction (why structural fails).
- `docs/data-silent-regularization-concept.md` / `-limits.md` -- the estimation-route method + its limits.
- `docs/augmentation-validation-design.md` -- how to validate (injected-dynamics library, subspace metric,
  excitation ablation).
- `docs/passivity-augmentation-literature.md` -- the primary-read literature catalog (§G verification, §H
  marginal-dissipativity).
- `docs/decisions.md` -- D-104/105/106/107.
