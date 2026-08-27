# Handoff: everything now trains except the thing we care about. Decide the loss balance, then run.
**From**: sessions of 2026-08-20 (D-150 + transient investigation) | **Branch**: Augmentation | **Effort suggested**: high

## 1. Task

The user wants to DISCUSS and decide, with you, how to actually make the augmentation, the encoder
and the ANN train together. Three separate blockers have each been identified and individually
broken (section 4): the augmented states had no dynamics (fixed), the augmented encoder block
`W^a` got exactly zero gradient (fixed), and the training objective was 88 % startup transient
(fixed by burn-in). All three fixes are measured and reproducible. **The combined run still made
the model worse**: validation free-run RMS `6.03e-06 m` after one epoch against `2.187e-06`
untrained, i.e. 2.8x worse. The diagnosed cause is loss-term imbalance, and the proposed
correction has never been run and does not survive arithmetic (section 8). Your job is to bring
the user to a decision on the loss balance and the arm design, with one recommendation, then
execute it. Start with the cheap calibration measurement in section 9, not with a training run:
the machine is the user's laptop and they killed the last multi-hour arm to protect it.

## 2. Out of scope

* Two literature searches are handed off separately and may run in parallel sessions; do not
  duplicate or fold them in: `tasks/handoffs/2026-08-20-closed-loop-literature.md` (CLOE,
  dual-Youla, closed-loop objective design, informativity) and
  `tasks/handoffs/2026-08-20-encoder-augmentation-literature.md` (SUBNET encoders for
  augmentation, washout vs encoder training). Read their reports if they exist.
* The commit: blocked on the user (another session's P1/P1-e work sits in
  `gantry_dynamic/{config,evaluation,orth_penalty}.py`). Do not modify those three files, do not
  commit.
* Weakening D-072 (epsilon readout init): the user's decision, not yours. Do not implement.
* The clean (non-env-var) reimplementation of D-150: deferred by design,
  `docs/aug-lru-implementation.md` section 11.
* `kamtin-fp-model/` read only; never read `kamtin-data/Data Telica/`.

## 3. Where things stand

Branch `Augmentation`, last commit `4cdb7c1`, nothing committed. Nothing in flight.

Work of 2026-08-20 lives in two places, both untracked/dirty:
* `scripts/gantry/gantry_dynamic/model.py`: the D-150 `AugLRUBypass` (env gate `AUG_LRU=1`).
* `scripts/gantry/closed-loop-controller/transient-investigation/`: `diag_transient_source.py`,
  `test_encoder_gradients.py`, `train_combined_arm.py`, two result JSONs and `RESULTS.md`. That
  `RESULTS.md` is the primary record of the second session; every number in section 4 below is
  from it or from the earlier `ANN-learning-issue/RESULTS.md`.
* Checkpoint of the last clean arm (Arm F, D-150 only): `SSE_Interconnect_MultipleShooting_
  FFaboQ_best.pth` in the deepSI checkpoints dir. It loads ONLY into an `AUG_LRU=1` build
  (`docs/aug-lru-implementation.md` section 8).

## 4. Established and verified

**4.1 Link 1, the augmented states had no dynamics. FIXED (D-150).** One shared zero-initialised
ANN output layer produced both the physical correction and the augmented-state update, so zeroing
it for baseline equality killed the augmented dynamics: `rho(A_aa) = 0` exactly. `AugLRUBypass`
splits them: rows 0-5 stay exactly zero at init, rows 6-7 get `x_a,k+1 = A_aa x_a + gamma*NL`
with `A_aa` trainable in the LRU stable exponential parameterisation, ring-initialised from a
DATA-derived band (f `[149.90, 164.06] Hz`, rho `[0.9794, 0.9956]`, from
`runs/cl_residual_spectrum.json`; drawn r 0.9920 at 154.5 Hz). D-072 verified BIT-identical
(`2.1866011034177349e-06 m` gate on and off). Survives Telica-level noise. Full reference:
`docs/aug-lru-implementation.md`.

**4.2 Link 2, `W^a` was dead. FIXED (multiple-shooting state defect).** Under pure simulation
loss `dL/dW^a` is EXACTLY zero at init, because the zero readout makes `dy/dx_a(0) = 0`; measured
0 of 108 entries moved in every run for three sessions. The inter-segment defect
`d_j = x_node,j - x_sim(s_j)` compares the encoder's estimate against the state the live `A_aa`
propagated, which is an unsupervised, ORACLE-FREE signal for `W^a`. Measured
`|grad W^a| = 1.83e-01` (from exactly 0), and in the one epoch run 3016 of 3130 encoder
parameters moved.

**4.3 Link 3, the objective was 88 % transient. FIXED (burn-in).** At `nf = 400`, 87.9 % of a
correct model's window loss is its own latent startup transient against 15.2 % for the plateau
model, because the correct model USES its latent states and the encoder cannot initialise them.
Discrimination (planted vs plateau, MS scale): `1.560x` today, `8.968x` at `K = 100, w_burn = 0.1`,
`11.562x` at `K = 100` hard. The hard-burn-in value matches the 12 s free-run discrimination.

**4.4 Burn-in starves the PHYSICAL encoder, and Strategy B fixes that.** Under hard burn-in the
loop has suppressed `x_b(0)`'s influence by sample 100, so `|grad W^b|` collapses 493x. Detaching
the ANN dynamics on `[0, K)` and scoring that region at `w_burn = 0.1` recovers `|grad W^b|` 200x
(`3.28e-07`) while keeping the ANN gradient bit-identical to hard burn-in (`4.00e-05`). Note the
asymmetry established earlier: with live `A_aa` (tau 125 samples) the encoder's AUGMENTED rows do
NOT suffer this, because the loop neither measures nor drives `x_a` and 154 Hz sits above the
~100 Hz crossover.

**4.5 THE COMBINED RUN MADE IT WORSE, and this is the open problem.** One epoch, 416 updates,
`n_seg = 4`, `lambda_defect = 0.1`, `lr_enc = 1e-4`, `lr_ann = 1e-5`, `K = 100`, `w_burn = 0.1`:
free-run validation `6.032757e-06 m` against untrained `2.186601e-06` (2.8x WORSE), while
`rho(A_aa)` held at 0.9920 / 154.9 Hz and `W^a` finally moved (delta `4.22e-02`). Measured cause:
`|grad L_defect| ~ 150` against `|grad L_settled| ~ 4.0e-05`, a ratio of `3.75e6 : 1`, with the
encoder additionally running at 10x the ANN's learning rate. The encoder was optimised into a
self-consistent state space that is miscalibrated for the 12 s free run.

**4.6 Two source-reduction levers are now CLOSED by measurement.** (a) Kessels Remark 5.4
controller-state replay is structurally WRONG for our residual formulation: `u_plant = u_data +
C(y_data - y_model)` already contains the machine's controller integrator state, so replaying
`xc` double-counts it and produces a 17.0 m offset. `xc = 0` is the unique consistent choice
(confirms D-142). (b) Kessels Remark 5.3 direct position init raises the transient share to
99.9 %, because writing `x0[0:3] = y[0]` without consistent velocities creates a derivative
mismatch; the learned reconstructability map `W^b` is better. Both were my earlier
recommendations; both are dead, and the reasons are mechanical, not tuning.

**4.7 The controller-effort term, measured.** Scoring `u_fb` inside the rollout (target zero,
oracle-free): discrimination `2.081x` alone, `11.27x` on top of burn-in against `10.14x` for
burn-in alone, i.e. it adds ~11 %. Mechanism `u_fb = C e`, a `|C|^2`-weighted error that undoes
loop suppression BELOW crossover; therefore secondary in simulation (our mode is above crossover)
and primarily a Telica-phase term (friction, drift). Caveat: amplifies measurement noise above
crossover. `ANN-learning-issue/RESULTS.md` section 10.

## 5. Assumed but not verified

* **That `lambda_defect = 1e-5` is the right correction. It is not, by the source document's own
  numbers, and this is the first thing to settle** (section 8). See also the D-148 precedent: a
  consistency weight calibrated against quantities differing by 1.7e9 was inert, for exactly this
  class of reason.
* That the defect term's optimum is compatible with free-run accuracy at all. The defect rewards
  encoder-and-dynamics AGREEMENT, which is satisfiable by a pair that agrees with each other and
  drifts from reality. Nothing measured yet excludes this; the 4.5 outcome is consistent with it.
* That the ANN should be driven by the defect at all. In the combined strategy `|grad ANN| = 1.5e2`
  is the DEFECT's gradient, 3.75e6x the output-tracking gradient: the dynamics parameters are
  currently trained mostly to make the encoder's job easy, not to fit the output. Detaching the
  defect from the dynamics parameters (let it train the encoder only) is an untested option.
* That one epoch is diagnostic. The run was stopped at one epoch; whether it recovers is unknown.
* That burn-in's discrimination gain survives training (it is a ranking proxy; only a run settles
  it). D-148's burn-in arm ranked well and trained badly, on a model that could not exploit it.
* H3 (Zucchet/Orvieto: sensitivity grows exactly when a recurrent model starts using its state)
  and H5 (`nx_aug = 2` too small; Kessels' industrial case needed `n_ext = 14`) remain untested
  fallbacks if the objective work nulls.

## 6. Tried and failed

* Initialisation as the binding constraint -> D-150 fixed the mechanism, free run moved `-0.665 %`
  (`1.3841e-06` vs `1.3934e-06`) -> the objective could not rank the better model (1.13x-1.56x MS)
  -> `ANN-learning-issue/HYPOTHESES-AND-SOLUTIONS.md` section 6.
* Combined burn-in + Strategy B + defect at `lambda = 0.1`, `lr_enc = 1e-4` -> free run 2.8x WORSE
  after one epoch -> defect gradient 3.75e6x the tracking gradient, encoder at 10x the ANN's rate
  -> section 4.5.
* Kessels `xc` replay -> 17.0 m offset -> double-counts the controller state already inside
  `u_data` -> 4.6a.
* Kessels position init -> transient share 99.9 % -> velocity mismatch -> 4.6b.
* Hard burn-in on the DEAD-states model (D-148) -> worse than untrained -> nothing penalised
  `[0,K)` and the model could not use the fixed objective.
* Optimisers, lr, eps, capacity, `ANN_INIT_SCALE`, solving `W_out` for a target Jacobian: closed
  with evidence, `ANN-learning-issue/HYPOTHESES-AND-SOLUTIONS.md` section 5.

## 7. Achieved

Implemented and validated: D-150 bypass (`docs/aug-lru-implementation.md`, artefacts in
`ANN-learning-issue/RESULTS.md` 9b/9d); the transient decomposition and burn-in diagnostic across
476 windows (`transient_diagnostic_results.json`); the 5-strategy gradient verification
(`gradient_verification_results.json`); the effort-term discrimination
(`runs/effort_discrimination.json`); noise gate step 2 (`runs/cl_residual_spectrum_noisy.json`).
Implemented but NOT validated: `train_combined_arm.py` (ran one epoch, result was worse).
Documented: D-150 in `docs/decisions.md`; `docs/aug-lru-implementation.md`;
`ANN-learning-issue/` through section 10; `transient-investigation/RESULTS.md`; run table has
Arm E corrected and Arm F closed.

## 8. The open question

**How should the three loss terms be balanced so that the free-run POSITION error is what
actually gets minimised?** The user's stated priority is an accurate model, position output
first; the selection metric already is the 12 s closed-loop free-run position RMS.

`L = L_settled + w_burn * L_early(encoder only, dynamics detached) + lambda_defect * L_defect`

Candidate answers, with the evidence that chooses between them:

1. **`lambda_defect = 1e-5` as proposed.** By the source document's own gradient numbers this
   leaves the defect at `1.5e-3` against `L_settled`'s `4.0e-5`, i.e. still **37x dominant**. If
   the intent is parity, the value implied by those numbers is `lambda ~ 2.7e-7`. This
   discrepancy is unresolved and is the single most consequential open number in the file. Do not
   adopt `1e-5` because it is written down; measure the ratio at the proposed settings (section 9)
   and choose deliberately.
2. **Parity or output-dominant (`lambda ~ 1e-7` to `3e-7`).** Consistent with the failure
   mechanism in 4.5 (defect overpowering tracking) and with the user's priority. Risk: `W^a`'s
   gradient shrinks proportionally and may return to being negligible; the check is whether
   `|lambda * grad W^a|` still exceeds what the tracking loss gives it (which is exactly zero at
   init, so any positive value is progress).
3. **Defect on the ENCODER only (detach dynamics parameters from `L_defect`).** Removes the
   pathology that the ANN is currently trained mostly by the defect. Untested, cheap to test with
   the same gradient harness.
4. **Schedule instead of a constant**: defect strong for the first N updates to wake `W^a`, then
   decayed. More knobs, harder to defend in a thesis; only if 2 and 3 fail.

Secondary decisions the user may want to fold in: whether to include the effort term (4.7;
recommendation: not yet in simulation, keep for Telica), and whether `lr_enc` should equal
`lr_ann` (4.5 says the 10x mismatch contributed; equalising at 1e-5 is the conservative choice).

## 9. Next action

**Do not launch a training arm first.** Re-run the gradient harness
(`transient-investigation/test_encoder_gradients.py`, which already measures per-tensor gradient
norms on a live batch) under the exact production candidate settings, and report, for
lambda_defect in `{1e-5, 1e-6, 3e-7, 1e-7}` and for the encoder-only-defect variant:
`|grad|` contributed by `L_settled`, by `w_burn * L_early` and by `lambda * L_defect`, split by
parameter group (`W^b`, `W^a`, encoder net, ANN, `nu_log`/`theta_log`). That is minutes of compute
and it converts section 8 from an argument into a table. Then present the user with ONE
recommendation for the arm (lambda, lr pair, whether the defect touches the dynamics parameters)
and get their go-ahead before any multi-hour run. Write the run-table row before launching, per
the run-discipline rule.

## 10. Acceptance criterion

For the calibration measurement: a per-parameter-group gradient table at each candidate lambda,
with the ratio `|grad from defect| / |grad from settled output|` stated per group, and the lambda
that makes output tracking dominant (ratio below 1) while keeping `|grad W^a|` non-zero.

For the training arm that follows: free-run sim-RMS on V1-V4 below **`1.215e-06 m`** (45 % of
headroom; untrained `2.1866011e-06`, data-derived floor `2.81e-08`, previous plateau
`1.3933793e-06`), all from `closed_loop_free_run_rms`. Secondary, and these say whether the
mechanism worked: `rho(A_aa)` above 0.5 after training, `W^a` still moving, and the free run
IMPROVING monotonically rather than the 4.5 pattern. A first epoch worse than untrained is a stop
condition, not a phase to sit through.

## 11. Read these first

1. `scripts/gantry/closed-loop-controller/transient-investigation/RESULTS.md`: the three links,
   both gradient tables, the Epoch 1 analysis, and the proposed calibration (whose arithmetic is
   the section 8 question).
2. `scripts/gantry/closed-loop-controller/ANN-learning-issue/HYPOTHESES-AND-SOLUTIONS.md`
   sections 2 and 6: H2, the D-150 outcome, and what a null falsifies.
3. `docs/aug-lru-implementation.md`: exact D-150 code reference, env contract (`AUG_LRU`,
   `CL_NOISE_SIGMA`, `CL_SPEC_*`, the `CL_BURNIN`/`CL_CONS_FRAC` refuse-to-start guard),
   checkpoint compatibility.
4. `transient-investigation/train_combined_arm.py`: the runner that produced 4.5; the thing to
   recalibrate rather than rewrite.
5. `docs/decisions.md` D-148 and D-150: the two sessions of evidence, including the earlier
   mis-calibrated consistency weight that this handoff's section 8 rhymes with.

## 12. Do not

* Do not launch a multi-hour training run before section 9's table and the user's sign-off; they
  killed the last arm to protect their machine.
* Do not adopt `lambda_defect = 1e-5` without resolving section 8 item 1.
* Do not re-apply the burn-in patch without lifting the `cl_train.py` refuse-to-start guard in
  the same change (the guard exists so a reverted framework cannot silently ignore the flag).
* Do not retry `xc` replay or direct position init (4.6, closed with mechanisms).
* Do not use `x_aug`, the planted weights as an initialisation, or any oracle constant (159 Hz,
  0.9856, `rho = 0.976`).
* Do not overwrite `runs/cl_residual_spectrum.json`, `runs/cl_aug_spectrum.json` or any clean
  artefact; new outputs get new names.
* Do not modify `gantry_dynamic/{config,evaluation,orth_penalty}.py` or `kamtin-fp-model/`; do not
  read `kamtin-data/Data Telica/`; do not commit; do not weaken D-072.

## 13. Operational

Env `GraduationProject`. Long runs go background with the live-output convention in `CLAUDE.md`
(`PYTHONUNBUFFERED=1 conda run --no-capture-output ... python -u`), and the user is told the
`.output` path. The gradient harness and the transient diagnostic run in tens of seconds; the
combined trainer ran 416 updates per epoch. Any `AUG_LRU=1` build needs
`runs/cl_residual_spectrum.json` present (it is). Arm F cost ~5.5 min/epoch wall on this machine,
validation ~2 min. Checkpoints from gated runs load only into gated builds.

## 14. Delegation

None. The calibration measurement and the arm are targeted work for one context. The two
literature questions are separate sessions by the user's explicit design.
