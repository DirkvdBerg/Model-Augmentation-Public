# Handoff: after D-150 (the states train now), attack the window TRANSIENT at its source before deciding on burn-in
**From**: session of 2026-08-20 | **Branch**: Augmentation | **Effort suggested**: high

## 1. Task

Continue the main augmentation thread from the D-150 result. The parameterisation fix works
mechanically (everything trains, including the previously dead `W^a`) but the free run barely
moved, and the identified constraint is the objective: the loss ranks a correct model only 1.13x
above the plateau model because a fixed startup TRANSIENT dominates every 400-sample window
(88 % of a correct model's window loss). The user dislikes burn-in as the fix because it starves
the encoder's physical-row gradient (measured 178x at K = 100). Your job: investigate the
transient at its SOURCE with the oracle-free levers already identified (section 4.9), quantify
each one's effect on the transient share and on discrimination with no-training diagnostics, and
only then, with the user, decide whether burn-in (weighted, `w_burn = 0.1`) is still needed on
top. Two literature sessions run in parallel (section 2); do not duplicate them.

## 2. Out of scope

* The two literature searches, handed off separately and possibly running in parallel sessions:
  `tasks/handoffs/2026-08-20-closed-loop-literature.md` (CLOE, dual-Youla, closed-loop objective
  design, informativity) and `tasks/handoffs/2026-08-20-encoder-augmentation-literature.md`
  (SUBNET encoders for augmentation, washout vs encoder training). If their reports exist when
  you start, read them; do not re-run them.
* The commit. Blocked on the user (another session's P1/P1-e work sits in
  `gantry_dynamic/{config,evaluation,orth_penalty}.py`; do not modify those three files).
* Weakening D-072 (epsilon readout init): user's decision, do not implement.
* The clean (non-env-var) reimplementation of D-150: deliberately deferred
  (`docs/aug-lru-implementation.md` section 11); do not start it.
* The noisy RETRAINING arm: deferred until some arm shows a material gain to qualify.
* `kamtin-fp-model/` read only; `kamtin-data/` blocked.

## 3. Where things stand

Branch `Augmentation`, last commit `4cdb7c1`; tree dirty with the D-150 implementation
(`gantry_dynamic/model.py`), script additions (`cl_train.py`, `cl_residual_spectrum.py`,
`cl_aug_spectrum.py`), docs and handoffs; nothing committed. Nothing in flight. Arm F's best
checkpoint is `C:\Users\20203253\AppData\Local\deepSI\checkpoints\
SSE_Interconnect_MultipleShooting_FFaboQ_best.pth` (loads ONLY into an `AUG_LRU=1` build).

## 4. Established and verified

Full detail with artefacts: `ANN-learning-issue/RESULTS.md` sections 1-10 and
`docs/aug-lru-implementation.md`. The load-bearing facts:

* **4.1 The fix that made everything train (D-150).** `AugLRUBypass` in
  `gantry_dynamic/model.py`, env gate `AUG_LRU=1`: ANN rows 0-5 exactly zero at init (D-072
  verified BIT-identical, `2.1866011034177349e-06 m` gate on and off), rows 6-7 get
  `x_a,k+1 = A_aa x_a + gamma*NL`, `A_aa` trainable in the LRU stable exponential
  parameterisation, ring-initialised from the data-derived band f `[149.90, 164.06] Hz`, rho
  `[0.9794, 0.9956]` (read from `runs/cl_residual_spectrum.json` at build; drawn r 0.9920 at
  154.52 Hz, seed 0+150). Exact env contract and checkpoint rules:
  `docs/aug-lru-implementation.md`.
* **4.2 Its training result (Arm F, run-table row).** lr 1e-5, eps 1e-16, killed flat after
  epoch 5, best epoch 3: free run `1.3841e-06 m` vs plateau `1.3934e-06` (`-0.665 %`), acceptance
  (`<1.215e-06`) FAILED; secondaries PASSED: `rho(A_aa)` 0.9920 after training, `Wa_psi_y` 108/108
  moved (was 0/108 always), ANN 602/602, encoder 3002/3130. Readout grew onto the planted rows
  (dTheta at planted scale; dY 15x short when the window loss flattened).
* **4.3 The constraint is the objective (H2 promoted).** In its own normalised units the training
  loss ranks planted over trained by only **1.131x** (MS scale, `runs/effort_discrimination.json`).
  88 % of the planted model's window loss is its own latent startup transient, paid at 1666
  windows/epoch (D-148 finding 9): the loss taxes state usage.
* **4.4 Burn-in numbers.** Hard K = 100: discrimination 10.14x MS (3.19x RMS). Weighted
  `w_burn = 0.1`: 2.995x RMS, early window stays penalised. Encoder cost: physical-row gradient
  falls 178x at K = 100 (measured on the DEAD-states model, D-148 gate B6; transfer to the live
  model unverified). With live `A_aa` (tau 125 samples) the encoder's AUGMENTED rows keep their
  gradient through burn-in: `x_a(0)` influences samples 100-400 directly, the loop neither
  measures nor drives `x_a`, and 154 Hz is above the ~100 Hz crossover (sensitivity ~1).
* **4.5 The controller-EFFORT loss term, measured (user asked "how much effort").**
  `u_fb` inside the rollout is the loop's own mismatch estimate, target zero, oracle-free.
  `MS(u_fb)`: planted `7.540e-03`, trained `1.569e-02` (the loop works 2.1x harder for the wrong
  model) vs `MS(e)` 1.617e-10 / 1.829e-10. Discrimination: effort-only 2.081x; balanced mix
  (lambda ~ 2e-8, since `MS(u_fb)/MS(e)` ~ 5e7) 1.589x; monotone in lambda, no interior optimum.
  On top of burn-in: 10.14x -> 11.27x. Mechanism: `u_fb = C e`, a `|C|^2`-weighted error that
  undoes loop suppression BELOW crossover; hence secondary here (our mode is above crossover) and
  primarily a Telica-phase term (low-frequency friction/drift visibility). Caveat: amplifies
  measurement noise above crossover. `RESULTS.md` section 10.
* **4.6 Kessels 2025 ch. 5 (on disk, printed pp. 151-168, PDF offset +27) does closed-loop
  augmentation training and avoids our problems structurally.** Split NNs for f_aug (`l_w1`) and
  g_aug (`l_w2`); extended states carry a hard-coded position-velocity integrator
  (`x2p_{k+1} = x2p + dt x2v`) so their `A_aa` has an eigenvalue at 1 by construction (never our
  rho = 0 saddle); encoder inputs include REFERENCE and TRACKING ERROR (footnote 5.4); Remark 5.3
  initialises measured states directly via the inverted output map (encoder handles fewer
  states); **Remark 5.4 initialises the controller state per window by REPLAYING the known
  controller over the recorded error from the record start**, where our harness sets `xc = 0`
  (documented HEURISTIC, `model_augmentation/fit_systems/closed_loop.py` ~line 231). No burn-in,
  no transient handling in the loss.
* **4.7 Noise gate.** The initialisation band is unchanged under Telica-level noise (sigma
  6.5-8.5e-9 m derived from the Telica error-PSD floor; artefact
  `runs/cl_residual_spectrum_noisy.json`). Retraining arm deferred (nothing to qualify).
* **4.8 Guards added.** `cl_train.py` REFUSES to start if `CL_BURNIN`/`CL_CONS_FRAC` are set
  (framework support was reverted; re-apply `patches/2026-08-19-interconnect-burnin-consistency.
  patch` AND lift the guard together). `CL_NOISE_SIGMA` (cl_train), `CL_RS_NOISE_SIGMA`
  (residual spectrum, writes `_noisy` artefact), `CL_SPEC_OUT`/`CL_SPEC_SKIP_PLANTED`
  (aug spectrum) exist; never overwrite the clean artefacts.
* **4.9 The transient levers to investigate (all oracle-free), the core of your task:**
  (a) `xc` replay per Kessels Remark 5.4 instead of `xc = 0`: removes the controller's own
  per-window transient that the loss currently scores; (b) direct initialisation of the measured
  position states via the output map (Kessels Remark 5.3; our three positions are measured),
  shrinking the encoder's job and its initial-state error; (c) encoder inputs extended with
  reference/tracking error; (d) weighted burn-in `w_burn = 0.1` as the residual fix if (a)-(c)
  leave the transient dominant. (a) and (b) reduce the transient at the source, which is what
  the user prefers over not scoring it.

## 5. Assumed but not verified

* That the 88 % transient share still holds on the LIVE-dynamics model (it was measured on the
  planted model in the old parameterisation). Settled by re-running the decomposition on the
  Arm F checkpoint, which is part of section 9.
* That `xc` replay reduces the scored transient materially. Plausible (the controller state is
  currently wrong at every window start by construction); the section 9 diagnostic settles it.
* That the 178x encoder collapse transfers to the live model (measured on dead states).
* That more epochs would not have helped Arm F (window loss was flat; inference, not measurement).
* H3 (Zucchet/Orvieto sensitivity) and H5 (`nx_aug = 2` too small; Kessels' industrial case
  needed `n_ext = 14`): untested fallbacks if the objective work also nulls.

## 6. Tried and failed

* Initialisation as the binding constraint -> Arm F: mechanism fixed and verified, free run
  `-0.665 %` -> the objective cannot rank the right answer (1.13x), so training reconverges to
  the static basin -> `HYPOTHESES-AND-SOLUTIONS.md` section 6.
* Hard burn-in trained on the DEAD-states model -> worse than untrained on the full window, free
  run -0.8 % -> nothing penalised `[0,K)` AND the model could not exploit the fixed objective;
  encoder starved -> D-148 finding 9 / handoff 2026-08-19 section 6. Do not cite as evidence
  against burn-in-on-live-dynamics; do carry both failure mechanisms into any new arm.
* Optimisers, lr, eps, capacity, `burn_in` + consistency as implemented, `ANN_INIT_SCALE`,
  solving `W_out` for a target Jacobian: all closed with evidence, list in
  `ANN-learning-issue/HYPOTHESES-AND-SOLUTIONS.md` section 5 and the 2026-08-19 handoff section 6.

## 7. Achieved

Implemented and validated: the D-150 bypass (4.1, artefacts in `RESULTS.md` 9b/9d); Arm F run
(4.2); effort-term measurement (4.5, `runs/effort_discrimination.json`); noise-gate step 2 (4.7).
Documented: D-150 in `docs/decisions.md`; `docs/aug-lru-implementation.md` (exact code reference,
env contract, known debt, clean-implementation plan); `ANN-learning-issue/` updated through
section 10; run table has Arm E corrected and Arm F closed.

## 8. The open question

Can the window transient be reduced at the SOURCE far enough that the objective discriminates
without (much) burn-in? Candidates: 4.9 (a) xc replay, (b) direct position init, (c) encoder
inputs, alone and combined; the residual option is (d) `w_burn = 0.1`. Evidence that chooses: the
transient share and the planted-vs-trained discrimination under each lever, measured by the
no-training re-reduction (rollouts of both models over the same window grid, same method as
`cl_burnin_sweep.py` and the effort measurement). The user explicitly prefers source reduction
over not scoring samples, and explicitly worries about encoder training under burn-in.

## 9. Next action

Build one no-training diagnostic (scratchpad first; promote to a script only if kept) that rolls
out planted and trained (and optionally Arm F's checkpoint with `AUG_LRU=1`) over the standard
validation window grid under three rollout variants: (i) today's (`xc = 0`, encoder init),
(ii) xc REPLAYED per Kessels Remark 5.4 (run the known controller over the recorded error from
record start to each window start; the controller bank and records are in
`cl_pipeline.build_closed_loop` / `loss_variants.controller_ss`), (iii) variant ii plus the
measured positions written directly into the initial state (Remark 5.3; positions are outputs, so
`h` inversion is trivial here). For each variant report: the transient decomposition (window MS
vs settled MS, as D-148 finding 9 did), and the discrimination at K = 0 and with `w_burn = 0.1`.
Rationale: this tells us in minutes, before any training, whether the transient can be removed at
the source and how much burn-in remains necessary; it directly answers the user's objection.
Write the run-table row only if a training arm follows.

## 10. Acceptance criterion

For the diagnostic: the transient share (percent of window MS) and discrimination for each
variant, planted vs trained, on the same 476-window grid, reported next to today's baseline
(1.131x MS at K = 0). Decision threshold, agreed direction with the user: if any source-reduction
variant reaches a discrimination comparable to weighted burn-in's (2.995x RMS = 8.97x MS) without
de-weighting samples, it becomes the training-arm design; otherwise the arm is `AUG_LRU=1` +
`w_burn = 0.1` + the best source-reduction levers combined. Any training arm keeps the standing
acceptance: free run below `1.215e-06 m` against untrained `2.1866011e-06`, floor `2.81e-08`,
`rho(A_aa)` above 0.5 after training.

## 11. Read these first

1. `ANN-learning-issue/HYPOTHESES-AND-SOLUTIONS.md` sections 2 and 6: H2, the outcome, and the
   pre-registered reasoning.
2. `ANN-learning-issue/RESULTS.md` sections 9-10: every current number with its artefact.
3. `docs/aug-lru-implementation.md`: the exact D-150 implementation, env contract, checkpoint
   compatibility (needed to load the Arm F checkpoint).
4. `literature/augmentation/kessels2025_ai-control.pdf` printed pp. 156-158 (PDF +27): Eqs.
   5.12-5.13d and Remarks 5.3/5.4, the two source-reduction levers.
5. `model_augmentation/fit_systems/closed_loop.py` docstring at the `xc` HEURISTIC: what variant
   (ii) replaces and why it was originally chosen.

## 12. Do not

* Do not run any training arm before the section 9 diagnostic and the user's sign-off on the
  arm design (they killed the last arm to protect their machine; ask before multi-hour runs).
* Do not re-apply the burn-in patch without lifting the `cl_train.py` guard in the same change.
* Do not overwrite `runs/cl_residual_spectrum.json`, `runs/cl_aug_spectrum.json` (2026-08-19
  records) or any clean artefact; new outputs get new names.
* Do not use `x_aug`, the planted weights as an init, or any oracle constant (159 Hz, 0.9856).
* Do not modify `gantry_dynamic/{config,evaluation,orth_penalty}.py`, `kamtin-fp-model/`;
  do not read `kamtin-data/Data Telica/`.
* Do not commit (user's open decision), and do not weaken D-072.

## 13. Operational

Env `GraduationProject`; long runs background with the live-output convention (CLAUDE.md). The
no-training diagnostic runs in tens of seconds (the effort measurement took 16 s for two
rollouts). Any `AUG_LRU=1` build needs `runs/cl_residual_spectrum.json` present (it is). Training
arm, if and when agreed: section 13 of `tasks/handoffs/2026-08-19-ann-learning-issue-fable.md`
launch pattern plus `AUG_LRU=1`, and note the burn-in patch + guard coupling (4.8). Arm F cost
~5.5 min/epoch wall on this machine; budget validations at ~2 min each.

## 14. Delegation

None for the diagnostic (targeted, one context). If the two literature sessions have not run,
they are separate sessions by the user's explicit design; do not fold their work in here.
