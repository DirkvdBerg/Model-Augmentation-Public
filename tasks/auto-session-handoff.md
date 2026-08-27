# AUTONOMOUS SESSION HANDOFF — the ANN free-run drift fix (2026-07-24)

You are running AUTONOMOUS while the user is away. Mission: find a drift fix that satisfies the FIVE
REQUIREMENTS, verified against evidence, WITHOUT re-deriving the already-converged diagnosis and WITHOUT
tripping any guardrail below. Work the task queue, self-reflect on each run's output, score against R1-R5,
adjust scope, and document. When you hit an ASK-condition, STOP and leave a clear note for the user rather
than guess.

Operating loop each cycle: pick the top runnable task -> log a run-table row (D-090) with its hypothesis
BEFORE launch -> run it (background + unbuffered, live-output convention) -> read the .output -> score the
result against the requirement(s) it targets -> update the R1-R5 SCORECARD below -> branch (prune a failed
candidate, or design the next test) -> write a one-paragraph finding into this file's PROGRESS LOG. Repeat.

## HOW TO RUN THIS AUTONOMOUSLY (do this at session start)
1. **Make the task list explicit.** Immediately call TaskCreate to mirror the TASK QUEUE below as tracked
   tasks (TASK 1..4), then TaskUpdate each to in_progress/completed as you go. Keep it current so the user
   can see progress. Add newly-designed sub-tasks to the list as your self-reflection spawns them.
2. **Chain without waiting for the user.** Launch each run in the BACKGROUND (run_in_background); when it
   completes you are AUTOMATICALLY re-invoked with the result -> reflect (protocol below) -> launch the NEXT
   task's run in the same way. Do NOT stop to ask between tasks; only stop at a STOP/ASK condition.
3. **If nothing is pending and the queue is not empty**, start the next runnable task rather than ending the
   turn. The user is away; keep the queue moving.
4. **Launch mechanism** (the user starts this): `/loop work tasks/auto-session-handoff.md autonomously`
   with NO interval (self-paced) is the intended trigger; it re-fires so the queue continues unattended.
   Absent /loop, the background-run-completion re-invocations still let you chain tasks within the session.
5. **Reflect + readjust IS expected:** if a run falsifies a candidate, prune it from the scorecard and
   re-order the queue; if a knob was inconclusive, redesign it (verify it moves the target first) rather
   than re-running the same thing. Record every scope change in the PROGRESS LOG.

---

## HARD GUARDRAILS (read first; you are unsupervised)
1. **kamtin-fp-model/ is READ-ONLY.** Never modify it. Data Telica is `.claudeignore`-blocked; never read it.
2. **No class-restriction as the DELIVERABLE.** Telescoping/net-impulse/contraction/passivity restrict the
   residual class and can delete real dynamics (Coulomb carries net impulse) — they are COMPARISON/REFERENCE
   arms only, never the deliverable ([[unknown-system-no-class-restriction]]).
3. **Velocity/accel LOSS is the LAST RESORT** (supervisor). Do NOT adopt it or velocity-only INPUT. A
   telescoping FORCE on velocity rows that keeps the position loss + full-state input is allowed (it is the
   reference), but is still a class restriction (guardrail 2). Classify any velocity-adjacent idea before
   coding ([[velocity-loss-is-last-resort]], drift-diagnosis-status.md:1341-1342).
4. **X/Y stay in the ANN routing** (D-103); Theta-only is a diagnostic baseline only, never the deliverable.
5. **No cost adjectives without a measured number.** BPTT is expensive at every scale. Do NOT launch REAL
   training (with-MSD data, encoder + ANN, production epochs) unsupervised — that is an ASK-condition.
6. **Use the EXACT written method.** If a method is named in a spec, grep + read the spec and implement THAT
   structure; a plausible same-name interpretation can invert the result ([[use-users-exact-term]]).
7. **Run discipline:** every run with a new hypothesis gets a run-table row in
   `docs/gantry-augmentation-problem-log.md` §12 BEFORE launch; add the outcome after.
8. **Per-axis + data-derived thresholds.** Break free-run drift out per axis (X and Y separately); judge
   against the ANN-off floor / noise floor, never an oracle/model threshold.
9. **Do NOT re-derive the converged diagnosis.** It is settled (below). New runs must REMOVE an option or
   fill a scorecard gap, not re-confirm what is known.

---

## THE PROBLEM (converged — do NOT re-diagnose; pointers only)
The augmented model (physics + parallel ANN) DRIFTS in long free-run on the marginal (pole~=1) X/Y position
modes; this gates the deliverable. Diagnosis is CONVERGED across three independent investigations:
- **gantry-zero-mean/**: the drift's DC component is Adam's implicit-bias walk in a weakly-constrained
  direction (SGD builds ~2000x less at matched lr) — X-axis dominant.
- **ARTBP/** (`feedback_instrument.py`, `test_efolding.py`): a state-dependent anti-damping GAIN on the Y
  mode that short-horizon training under-penalizes — Y "destabilization".
- **baseline-null/** (`curvature_sensitivity.py`, `gain_vs_dc.py`, `lr_sweep.py`): on a PERFECT-MATCH null
  (nothing to learn), training STILL drifts. It is ~80% Adam DISPLACEMENT of a STIFF DC direction
  (curvature ~7e4, verified) + ~20% anti-damping feedback (destabilizing sign on dX/dY). The DC MINIMIZER is
  ~1e-11 but Adam parks it ~lr away (curvature-blind) and the integrator amplifies -> drift ~ lr.
Full record: `docs/gantry-augmentation-problem-log.md` §12 (rows BASELINE-NULL, CURVATURE+SENSITIVITY,
GAIN-vs-DC, VELOCITY-ROUTING, ROLAND'S TELESCOPING). Cures landscape: `scripts/gantry/baseline-null/
diagnostics-literature.md`. Requirements: `docs/all-five-construction-spec.md` + `docs/dissipative-block-spec.md`.

## THE FIVE REQUIREMENTS (the accountability rubric — score every candidate against ALL five)
| | Requirement | Pass test |
|---|---|---|
| **R1** | knowledge-free | constraint uses DATA properties (unexcited direction) / known FP subspace, not the unknown residual |
| **R2** | full expressivity / friction (**THE most important** per user) | ANN can still LEARN real nonlinear dynamics (absorber, friction); test: injected-friction sim / with-MSD absorber retained |
| **R3** | marginal-preserving | linearize trained model -> X/Y pole \|lambda\|=1 (not <1, not >1) |
| **R4** | non-drifting | 12 s free-run position-ENVELOPE ratio ~1.0, per axis |
| **R5** | scheduling-integrity (Y) | held-out Y / M(Y) retained; schedule off de-drifted/exogenous Y |

## ESTABLISHED POSITION ON THE CANDIDATES (do not relitigate; TEST the open cells)
- **Orthogonal projection (Route B):** an INTERPRETABILITY tool (keeps physical params identifiable), NOT a
  drift fix (the DC is orthogonal to the param subspace, so projection leaves it). It PRESERVES R2 by
  least-squares orthogonality (the learnable residual is orthogonal to the param subspace; projection only
  removes the redundant param-overlap) — PROVIDED joint estimation is on. Risk: genuinely AMBIGUOUS effects
  get assigned to parameters by fiat.
- **Telescoping bounded-impulse (Route A, "Roland's"):** PROVEN no-drift by construction (R4 pass: 1.0x
  floor, `gain_vs_dc_35_tele.npz`). SACRIFICES R2 (forbids net-impulse = Coulomb friction). => VERIFICATION
  REFERENCE, not deliverable.
- **Estimator route (optimizer + conditioning):** curvature-aware optimizer (SGD) kills the DC displacement
  with FULL expressivity (no class restriction) => the R2-preserving drift-fix candidate; pair with
  multiple-shooting/long-horizon for the feedback part. Regularize ONLY the data-null/unexcited direction
  (nothing learnable there => zero expressivity cost). This is the honest R2-first path.

## CURRENT R1-R5 SCORECARD (fill the blanks as you run)
| candidate | R1 | R2 (expressivity) | R3 (pole) | R4 (no-drift) | R5 |
|---|---|---|---|---|---|
| Telescoping (Route A, REFERENCE) | pass (all-weights) | **FAIL (deletes friction)** | **PASS (null): |lambda|=1, shift 3.2e-7, NOT damped** | **PASS (1.0x floor)** | ? |
| Orth projection (Route B) | pass (known subspace) | pass-by-construction* (needs joint est.) | ? | **? (doesn't touch DC)** | ? |
| Estimator: SGD/curvature-aware | pass | **FAIL at drift-taming lr (TASK 5: +0% learned vs Adam +18%)** | **PASS (null): |lambda|=1, shift 1.4e-11** | **null 1.0x = INACTION; with a real residual DRIFTS 83x** | ? |
| Estimator: + null-direction reg | pass | pass (data-null only) | ? | ? | ? |
*R2 caveat: ambiguous effects routed to params.
**Estimator-SGD R4 note (TASK 1, 2026-07-24): 1.0x floor in the PERFECT-MATCH NULL (both Adam DC displacement AND the
tiny null anti-damping feedback removed; Adam was 8.08x). This is R4 pass IN THE NULL ONLY -- the real-data Y
destabilization is under-excited by the null and is NOT tested here; the real-data R2+R4 test needs training (ASK-gate).

## TASK 4 SYNTHESIS + ONE RECOMMENDATION (2026-07-24, autonomous session end)
**Winner with R2 (the most important requirement) intact = the ESTIMATOR: SGD / curvature-aware route.** In the
perfect-match null it now passes R1 (knowledge-free -- the fix is the OPTIMIZER, no constraint on the residual),
R2 (full expressivity -- plain SGD, full-state input, route 0..5, NO class restriction), R3 (|lambda|=1, pole
shift 1.4e-11), and R4 (1.0x floor vs Adam 8.08x). The telescoping REFERENCE also passes R1/R3/R4 but FAILS R2
(deletes net-impulse = Coulomb friction) -> it stays a verification benchmark, not the deliverable. Orthogonal
projection remains the INTERPRETABILITY tool (R2-safe by least-squares orthogonality) and is complementary, not a
standalone drift fix.

**CRITICAL SCOPE (do not over-claim): every demonstrated pass above is in the PERFECT-MATCH NULL** (baseline-on-
baseline, residual ~0, trained ANN output ~0). The null does NOT exercise R2 (no real dynamics to learn) and
under-excites the real-data Y destabilization (the ARTBP feedback_instrument / test_efolding evidence lives at
longer horizons on with-MSD data). So the null establishes that SGD does not INJECT spurious drift that Adam does
-- necessary, not yet sufficient. R5 (Y-scheduling integrity) is untested by any of these runs.

**THE ONE NEXT ACTION (ASK-condition -- prepared, NOT launched): the real-data SGD-vs-Adam R2+R4 test.** Train the
augmentation on the with-MSD (injected absorber/friction) data, encoder+ANN, SGD vs the Adam control at matched lr,
and measure per-axis free-run drift (R4) AND whether the injected friction/absorber is retained (R2). This is the
decisive test of whether the estimator route holds when there is genuine dynamics to learn. It is an ASK-condition
on THREE counts: (1) it is REAL training (with-MSD, encoder+ANN, production epochs) = venue/budget is the user's
call; (2) it forces the R4-empirical acceptance decision (is demonstrated no-drift the deliverable?) = supervisor
gate; (3) R5 Y-scheduling (exogenous vs self-scheduled) = supervisor keystone. PREREQUISITE (small, to prepare on
go-ahead): the production trainer (`gantry_dynamic/model.py:185`) exposes only lr, not optimizer TYPE (deepSI
defaults to Adam) -- add a one-line `optimizer=torch.optim.SGD` knob to `init_model` (scripts pipeline, NOT
model_augmentation/), mirroring gain_vs_dc.py's GV_OPT. Caveat: a curvature-aware step on the FULL encoder+ANN
loss is not identical to SGD on the tiny null ANN -- SGD's convergence on the real, non-convex, higher-curvature
landscape is an empirical question the test answers, and lr will likely need retuning per optimizer
([[classify-not-learning-from-loss-shape]]).

AUTONOMOUS QUEUE COMPLETE (TASK 1 pass, TASK 2 pruned, TASK 3 pass, TASK 4 synthesis). Halting at the ASK-gate.

---

## AUTONOMOUS TASK QUEUE (top = do first; all CHEAP, existing scripts, null/perfect-match regime)
All commands: prepend `PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1` and run via
`conda run --no-capture-output -n GraduationProject python -u <script>` in BACKGROUND; read the job .output.

### TASK 1 — Estimator route: does SGD kill the DC displacement? (scores R4 + R2 for the estimator)
Hypothesis: a curvature-aware optimizer (SGD) parks the stiff DC at its ~1e-11 minimizer instead of ~lr, so
the trained full free-run drift drops toward the floor WITH full expressivity (no class restriction). Predict
SGD full-drift << Adam's 8-18x, ideally ~1x floor.
RUN (already runnable — `GV_OPT` knob added 2026-07-24):
  `GV_OPT=sgd GV_ROUTE=0,1,2,3,4,5 GV_STRIDE=100 ... python -u scripts/gantry/baseline-null/gain_vs_dc.py`
  Compare vs the Adam control `gain_vs_dc_012345.npz` (Adam gave 8x floor, DC-displacement dominant).
DECISION RULE: if SGD full-drift ~1x floor -> estimator route PASSES R4 with R2 intact -> promote it (the
R2-first deliverable candidate). If SGD still drifts markedly -> the feedback part is not optimizer-fixable
-> escalate to conditioning (TASK 2). Either way score R4 for "Estimator: SGD".

### TASK 2 — Conditioning: does a longer free-run-aware window help the FEEDBACK part? (scores R4)
Only if TASK 1 leaves residual drift. Hypothesis: the ~20% anti-damping feedback needs the loss to SEE
seconds-scale growth (Ribeiro L_h): sweep the training window nf upward and/or add a cross-window continuity
term. CHEAP first check: re-run TASK 1's best optimizer at nf in {400, 800, 1600} (env `GV_NF`), read the
per-axis drift trend. DECISION RULE: drift falls monotonically with nf and R2 intact -> conditioning is the
feedback fix; if flat or divergent -> log it (matches the "can't out-window a marginal mode" prior, D-090
row nf-sweep 71013) and mark the feedback part as needing the structural REFERENCE only.

### TASK 3 — R3 eigen-check on the telescoping REFERENCE and the estimator winner (scores R3)
Hypothesis: does each candidate keep the X/Y pole at |lambda|=1? BUILD a small diagnostic (compile + smoke
first) that linearizes the trained augmented one-step map (autograd Jacobian of `fit_sys.hfn` wrt state at
sampled free-run points) and reports the X/Y eigenvalues. Apply to `gain_vs_dc_35_tele.npz` (telescoping)
and TASK 1's checkpoint. DECISION RULE: |lambda|=1 -> R3 pass; <1 -> artificial damping (bad); >1 -> the
anti-damping feedback survived. This is a NEW small script -> allowed (compile + smoke), but if it balloons
past ~120 lines or needs training, STOP and flag for the user.

### TASK 4 — Synthesis + scorecard + one recommendation (no run)
When TASKS 1-3 have filled enough cells: write the completed R1-R5 SCORECARD, name the candidate that
satisfies the most requirements WITH R2 intact, and state the ONE next action for the user (which is almost
certainly a REAL-DATA / injected-friction R2 test = an ASK-condition, since it needs training). Do NOT
launch it; prepare it.

(Gaps you may design cheap tests for, in priority order, if the queue empties: the DATA-NULL-direction
regularizer prototype on the null; an excited-record R2 probe of the telescoping block using the with-MSD
absorber data — but the with-MSD run is heavier, treat as ASK if it needs production epochs.)

---

## SELF-REFLECTION PROTOCOL (do this after EVERY run)
1. Read the .output; extract the per-axis drift, the impulse/curvature/optimizer numbers, and any error.
2. Score against the requirement(s) the task targets; update the SCORECARD table above IN THIS FILE.
3. Log the run-table row outcome in `docs/gantry-augmentation-problem-log.md` §12.
4. Branch: did it PASS (advance), FAIL (prune candidate / adjust), or is it INCONCLUSIVE (redesign the knob
   — verify the knob moves the target before spending another run, [[verify-knob-moves-the-target-before-running]])?
5. Append a dated one-paragraph finding to the PROGRESS LOG below. Keep claims to what the run shows
   (measure-on-target; no recollection-based "matches X" claims).

## STOP / ASK CONDITIONS (leave a note for the user, do not proceed)
- Any REAL training (with-MSD data, encoder+ANN, production epochs) — venue/budget is the user's call.
- The R4-EMPIRICAL acceptance decision (is demonstrated no-drift acceptable as the deliverable?) — supervisor gate.
- The Y-scheduling decision (exogenous/measured vs self-scheduled) — supervisor keystone for R5.
- Building the actual deliverable (orth projection joint-estimation run) — needs the above + venue.
- Any new script >~120 lines, or one that needs to modify `model_augmentation/` — show the skeleton, flag it.

## ASSETS TO LEVERAGE (existing hypotheses/docs — do not duplicate)
- `scripts/gantry/baseline-null/`: `gain_vs_dc.py` (intervention + telescoping + GV_OPT), `curvature_sensitivity.py`,
  `lr_sweep.py`, `run_baseline_null.py`, `README.md`, `curvature-sensitivity-finding.md`, `diagnostics-literature.md`.
- `scripts/gantry/ARTBP/`: `feedback_instrument.py`, `test_efolding.py`, `test_self_scheduling.py`, `README.md`.
- `scripts/gantry/gantry-zero-mean/`: v3x0/v7 optimizer studies, `README.md`.
- Docs: `all-five-construction-spec.md`, `dissipative-block-spec.md`, `drift-diagnosis-status.md` (§5 the five
  requirements, top standing constraint), `gantry-augmentation-problem-log.md` §12 (the run table), `decisions.md`.
- Lessons (active constraints): `tasks/lessons.md`. Read it first (Step 0).

---

## PROGRESS LOG (append dated findings; newest last)
- 2026-07-24 (setup): handoff created. State at handoff: diagnosis converged; telescoping REFERENCE
  established (R4 pass by construction, R2 fail); projection re-scoped to interpretability (R2-safe by
  least-squares orthogonality, doesn't fix drift); estimator route (SGD) is the R2-preserving drift-fix
  candidate and is TASK 1 (now runnable via GV_OPT). Next: run TASK 1.
- 2026-07-24 (TASK 1 smoke, 1 epoch, VERIFIED RUNNABLE): `GV_OPT=sgd GV_STRIDE=100 GV_EPOCHS=1` ->
  full free-run sim-RMS 3.788e-7 = **1.0x floor** at ep1 (Adam gave 29.7x at ep1); intervention DC~0, fb~0
  on all axes. PRELIMINARY: SGD (curvature-aware) does NOT displace the stiff DC -> no drift, full
  expressivity. This is only 1 epoch -- RUN THE FULL TASK 1 (6 ep) + compare vs Adam `gain_vs_dc_012345.npz`,
  then score R4 for the estimator and log the §12 row. If it holds at 6 ep, the estimator route is the
  R2-first drift-fix candidate to promote (still leaves R3 pole-check = TASK 3, and the ~20% feedback part
  may need TASK 2 conditioning).
- 2026-07-24 (TASK 1 FULL, 6 ep, DONE -> PASS): SGD full free-run sim-RMS = 3.782e-07 = **1.0x floor at every
  epoch** (flat ep1..6), vs Adam 8.08x (`gain_vs_dc.npz`). Intervention on the trained ANN: frozen/full sim-RMS
  = 1.000; DC=frozen-off ~0 all axes; feedback=full-frozen ~0 (fb/DC 0.00-0.05); off==full (the residual
  Y=1.12e-6/dY=1.22e-6 is the ANN-OFF FLOOR, not ANN-induced). Jacobian self-feedback ~1e-11-1e-12 (Adam had
  +5.74e-9/+1.43e-8) -> SGD builds NO anti-damping feedback either. VERDICT: in the NULL, curvature-aware SGD
  removes BOTH the Adam DC displacement AND the tiny null anti-damping feedback, FULL expressivity (no class
  restriction) -> **estimator route PASSES R4 in the null with R2 intact -> PROMOTED as the R2-first drift-fix
  candidate.** CAVEAT (measure-on-target): perfect-match null (residual~0, minimizer~0, SGD parks at ~0); this
  does NOT test the real-data Y destabilization (ARTBP feedback_instrument/test_efolding evidence, longer
  horizons) which the null under-excites -- real-data R2+R4 is an ASK-gate. **TASK 2 PRUNED: it targets residual
  feedback drift, but SGD left NONE in the null, so the nf-sweep-on-null cannot move the target
  (inconclusive-by-construction). The feedback part's genuine test is real-data (ASK), not a null nf-sweep.**
  Re-order: skip TASK 2, proceed to TASK 3 (R3 eigen-check on the telescoping REFERENCE + the SGD winner).
  Data `gain_vs_dc_012345_sgd.npz`.
- 2026-07-24 (TASK 3, R3 eigen-check, DONE -> PASS): NEW training-free diagnostic `pole_check.py` (119 lines,
  compile+smoke gated). Baseline one-step Jacobian (ANN off) eigenvalues at 9 points across the full T5 Y range:
  **max|lambda| = 1.000000 everywhere, NO pole >1**; two eigenvalues exactly at 1.0 (X-dominant rigid integrator
  marginal poles), X marginal |lambda|=0.99994, Y |lambda|=0.99975 (both within 0.5% of 1 = R3-consistent);
  Theta/velocity poles lightly damped <1. Trained-ANN pole-shift bounds from the SAVED Jdiag (gain_vs_dc.py has
  no checkpoint; the ANN's linearized contribution is O(Jdiag), baseline J is O(1)): SGD 1.4e-11, Adam 4.2e-8,
  Telescoping 3.2e-7 -- ALL << the ~2.5e-4 gap-to-1. **R3 PASS (null) for BOTH the SGD estimator winner AND the
  telescoping reference; baseline physics is marginal-stable.** NOTABLE: telescoping does NOT push |lambda|<1
  (it bounds the cumulative FORCE, not the pole) -> marginal-preserving, not artificial damping. CAVEAT (same
  scope as R4): null regime (trained ANN ~0 -> trained poles = baseline + O(Jdiag)); real-data R3 with a genuine
  non-zero residual needs a checkpoint = training = ASK-gate. Next: TASK 4 (synthesis + one recommendation).
- 2026-07-24 (TASK 5, R2 injected-friction probe -> SGD route R2-FALSIFIED + TASK 1 CORRECTED): user pushed
  the autonomous re-eval past the ASK-headline; the highest-value open cell was the estimator route's R2 (argued,
  not shown). `r2_fit_probe.py` injects a known, exactly-ANN-representable friction-like residual and trains a fresh
  ANN SGD vs Adam at matched lr=1e-7 to recover it. TWO metric corrections en route (both caught by scrutiny, not
  luck): (a) the SGD/Adam RATIO verdict rewarded SGD for NOT MOVING; (b) FREE-RUN sim-RMS conflates R2 (fit) with
  R4 (drift) -- switched to WINDOWED nf-RMS = identification quality. Clean result: **SGD +0% learned (windowed
  flat) yet still drifts 83x floor; Adam +18% learned but free-run diverges 1034x.** CORRECTION to TASK 1: SGD's
  null 1.0x-floor was INACTION (perfect match -> ~0 gradient -> SGD's lr*g step ~0 -> never moves -> no drift AND
  no learning), NOT curvature-taming; Adam moves ~lr*sign(g) even at ~0 gradient (its null drift). SYNTHESIS:
  learning (R2) and no-drift (R4) are a SINGLE-KNOB optimizer TRADEOFF -- neither wins both -> MANDATES the
  STRUCTURAL split (regularize ONLY the non-identifiable/unexcited direction, leave the excited/friction direction
  free = orthogonal projection). This RECONVERGES on the settled diagnosis (the cure is structural, not optimizer).
  Follow-up TASK 6 (SGD lr-SWEEP {1e-7..1e-4}) launched to remove the matched-lr confound the user flagged and
  confirm NO SGD lr wins R2+R4. The TASK 4 synthesis "winner = SGD estimator route" is hereby SUPERSEDED.
- 2026-07-24 (TASK 6, SGD lr-sweep -> CONFOUNDED, conclusion stands on TASK1+TASK5+theory): swept SGD lr
  {1e-7..1e-4}. SGD +0% learned at 1e-7/1e-6/1e-5 then DIVERGES at 1e-4 (82376x drift); Adam@1e-7 -13%. NOT
  decisive: NON-ROBUST (Adam @1e-7 flipped +18%[TASK5] -> -13%[here] on a MAXLEN change), UNDERPOWERED (windowed
  floor 2.79e-6 = ~7x true floor vs TASK5 ~160x), and the scaled random-ANN injection is anti-damping ->
  ill-conditioned BPTT (0%-then-explode = the signature). Logged the lesson `inject-a-well-conditioned-residual`
  (use dissipative friction F=-c*tanh(v/eps), verify the rig, fix the eval-window magnitude). **HONEST CONCLUSION:
  the OPTIMIZER-ONLY slice of the estimator route (SGD-as-the-fix) is deprioritized on (1) settled theory (Adam =
  SGD + a different lr; converged diagnosis = the cure is STRUCTURAL), (2) TASK 1 (SGD null no-drift = inaction),
  (3) TASK 5 (the better-powered run: SGD +0% vs Adam +18% at the drift-taming lr) -- NOT on this confounded sweep.
  The FULL estimator route per the converged diagnosis is NOT just optimizer choice: it is conditioning (multiple
  shooting + continuity) + regularizing ONLY the non-identifiable/unexcited direction (orthogonal projection).
  That = the STRUCTURAL CURE and remains THE open candidate.** STOPPING autonomous synthetic-probing here: the
  probe hit 3 design confounds (diminishing returns, rising confound risk), and the next real step -- building the
  null-direction orthogonal-projection cure and testing it on a well-conditioned dissipative-friction rig -- is the
  thesis's CORE CONTRIBUTION (D7/Gyorok extension) = an ASK-gate on scope/venue. Prepared, not launched.
- 2026-07-24 (REORIENTATION to the documented plan + two overclaims corrected): user pointed out (1) the plan
  already exists -- orthogonal projection + friction-in-simulation to verify before nonlinear real data -- and
  (2) "orthogonal projection is the only route" is an OVERCLAIM (ARTBP already removes the DC). Both correct.
  Read the full plan: `docs/all-five-construction-spec.md` (Route B = Layer1 conditioning + Layer2 re-aimed
  projection + Layer3 Y-scheduling), `docs/augmentation-validation-design.md` (the 3-model comparison +
  injected-dynamics library + excitation ablation), `docs/data-silent-regularization-concept.md` (Layer2 =
  Gyorok projection re-aimed; §7 d16 update: target = the measured near-DC/joint-DC direction d14, NOT
  low-information). CORRECTION: the cure is MULTI-LAYER -- ARTBP = Anticipated Reweighted Truncated BPTT =
  Layer-1 CONDITIONING that removes the truncation-bias DC (mostly X); projection is Layer 2 for the residual
  unexcited direction; Y-scheduling is Layer 3. What THIS session established is only that the OPTIMIZER-ONLY
  route (SGD swap) is a single-knob tradeoff = eliminated; the SGD-vs-Adam probes were an OFF-PLAN detour
  (lasting value = reusable injection/eval machinery in `r2_fit_probe.py`, `pole_check.py`). Lesson
  `free-run-drift-has-separate-sources` strengthened (multi-layer cure). **NEXT = a CLEAN session** (user
  inclined; this session's context is long + off-plan): opening instruction written to
  `tasks/build-brief-datasilent-friction-sim.md` (light-harness sim-first; dissipative Coulomb injection;
  3-model comparison; projection target = d14; build order = validation-design §10). Autonomous work here is
  DONE -- the build is the core contribution and belongs in the clean session.
