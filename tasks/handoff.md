# Session Handoff — open blockers only

**START HERE for the drift / non-zero-mean investigation**:
`scripts/gantry/gantry-zero-mean/README.md` (2026-07-15) is the self-contained picture: context,
run-71167 checkpoint provenance + the `_best` trap, key measured numbers, glossary, what is
established vs demoted, the Jan meeting notes, and the V1-V6 verification plan.

**Trimmed**: 2026-07-15 (full 2026-07-13 content archived to
`archive/sessions/2026-07-15-handoff-layer2-prebuild.md`; prior sessions in `archive/sessions/`).

## Recently closed
- **G-A (does the physics carry a DC the baseline lacks?) CLOSED on the physics side, 2026-07-17**,
  via `v1f_dc_excitation_openloop.m` (open-loop, same input to both plants, sustained offset + 150 Hz
  tone; see `scripts/gantry/gantry-zero-mean/README.md` §V1f). Both DC mechanisms measured:
  static-gain DC = 0 (M drops out at qddot=0); the delta_a^2 rectification DC = 3.1e-10 rad
  (confirmed by amplitude^2 scaling: `<delta_a^2>=(1.67e-5)^2=2.8e-10`); largest DC anywhere ~1e-7
  (the L0/mass-split static asymmetry). All 5+ orders below the ANN's DC. Verdict: the ANN's DC is
  NOT physics-justified; source is the estimator/training. **Live gates are now G-B and G-C**
  (README §7-8: V2 normalization/init audit, then V3 birth-of-the-DC probe — Jan's core ask).
  Lessons added: `verify-nonlinear-mechanism-fully`, `test-zero-mean-properly` clause (5).

## Open blockers

1. **Supervisor gate (THE decision): is empirical R4 acceptable as the deliverable?** For-all-weights
   no-drift is proven incompatible with full expressivity (`all-five-construction-spec.md` §4); Route B =
   demonstrated, not guaranteed. The drift-visual deck (`scripts/gantry/drift-visual/`, regenerated
   2026-07-15 from run 71167's rescued `_last` checkpoint, D-114/D-115) is the meeting material answering
   Jan's mail (not energy: f08; not zero-mean: f04; amplifier: K=0 axes).
2. **Build Layer 2**: the DIRECT measured joint-DC pin with near-DC frequency selectivity
   (`docs/data-silent-regularization-concept.md` §7 as corrected by d16; safety d12/d14v2, stationarity
   d15, Fisher-SVD refuted d16). Validate per concept §9 (12 s envelope ~1, absorber band untouched,
   injected-friction discriminator — needs a NEW friction-injected MATLAB dataset).
   **New caveat (2026-07-15)**: run 71167's DC has a state-dependent component (drift-visual f05: DC
   removal collapses only 2.6x on Y, not 133x as the old checkpoint) → re-verify d15-style pin
   stationarity on `gantry_drift_71167_last` before the build; iterative re-aiming is the documented
   contingency.
3. **Open question**: what was cluster run 71168 (finished Jul 14 23:37, ~4.6 h fit)? If it was a
   zero-mean-pin run, its `_last` checkpoint feeds `demo7_g9_intervention` as the intervention figure.
4. **f02 unit question**: measured V1/V3 Y trajectory sits ~0.10 m while captions say "Y = +10 mm";
   check `Matlab-scripts/Augmentation/data/` generator (`gtd_build_records`) to settle Yp10's unit.

## Standing constraints (unchanged, enforced)
- Theta-only routing never the deliverable (D-103); fixes act on loss/estimator, not routing.
- Velocity/accel-domain loss = LAST RESORT (supervisor-gated).
- No compute-cost adjectives without a measured basis; ask the user what is runnable.
- Read any run log's printed Configuration block FIRST (deployed copies lag local edits).
- Every gantry training script prints per-epoch `[nf-probe] train/val nf-RMS` (D-102).
- `conda run -n GraduationProject`, `PYTHONIOENCODING=utf-8` every PowerShell call.

## Where everything is documented (do NOT re-derive)
- Diagnosis d1–d16: `docs/drift-diagnosis-status.md` (§3b chain, §10 index).
- Layer-2 concept + limits: `docs/data-silent-regularization-concept.md` + `-limits.md`.
- Deck provenance + figure specs: `scripts/gantry/drift-visual/README.md`, D-114/D-115 in
  `docs/decisions.md`; run table `docs/gantry-augmentation-problem-log.md` §12.
