# Handoff: closed-loop training closes 37 % of the headroom and then stops. Find out why.
**From**: session of 2026-08-18 | **Branch**: Augmentation | **Effort suggested**: high

## 1. Task

Diagnose why closed-loop augmentation training improves the validation free run by 36.3 % within
its first two validations and then stops for the remaining ten, leaving the oracle floor 49.6x
away, WHILE the training loss keeps falling the whole time. Start from the supervisor's diagnostic, which has never been run: measure the error on
the TRAINING WINDOW itself, at the horizon the loss is computed over, against what a perfect model
achieves on that same window. If the training-window error does not approach that floor, the model
is not fitting its own objective and the free-run number cannot tell you anything about the
augmentation; if it does approach it while the free run stays at 1.4e-06 m, the problem is horizon
and generalisation, not fitting. Then, and only guided by that split, investigate what to vary
(optimiser, learning rate, horizon, architecture) with reference to the literature on training
recurrent models through BPTT. Separately and independently, evaluate downsampling the pipeline
from 4 kHz to 1 kHz, and quantify what that rate does to the controller.

## 2. Out of scope

- **The migration itself.** The closed loop now lives in `model_augmentation/fit_systems/closed_loop.py`
  behind four seams in `interconnect.py`. It is complete and gated (section 7). Do not redesign it,
  do not move it, and do not re-run the equivalence, gradient-precision or validation-precision
  experiments; they are settled (D-144, D-145, and section 4).
- **Runtime optimisation.** Measured and closed. The regime is dispatch-bound, the controller step
  is 2.6 % of a training step at batch 256, and `torch.compile` traces but cannot be benchmarked on
  the Windows box. Plan 3.8 carries the numbers. Do not spend time here.
- **The thirteen dangling historical scripts** in `scripts/gantry/closed-loop-controller/`
  (`cl_precision_*`, `cl_gate_*`, `cl_direct_vs_residual`, `cl_lr_probe`, `cl_plot_step6`,
  `cl_sanity`, `cl_smoke`, `cl_diag_step3`, `cl_step5_reset_cost`, `cl_step6_run`). The user decided
  to leave them; their results are recorded. Do not repoint or delete them.
- **`multiple_shooting.py` at `n_seg > 1` with a simulator.** It raises on purpose; the semantics of
  the controller state at a segment boundary are undecided and nothing needs them. User's decision.
- **Do not modify** `kamtin-fp-model/`, `references/step1_reference.old_impl.{json,npz}` (the frozen
  pre-migration recording, which cannot be regenerated), or the `documentation/*.tex` files.

## 3. Where things stand

Branch `Augmentation`, last commit `d4582cf`. **Nothing from this session or the previous one is
committed.** The tree is dirty in `model_augmentation/fit_systems/`, `scripts/gantry/gantry_dynamic/`,
`scripts/gantry/closed-loop-controller/`, `docs/decisions.md`, and ten single-line repair sites
(`augmentation-error/`, `diagnostics-drift/`, `drift-diagnostics/`, `gantry-zero-mean/`,
`msd_transfer_diagnostics/`, `orth-projection/`, `gantry_optuna.py`). Four of the touched files in
`model_augmentation/` and `gantry_dynamic/` ALSO carried uncommitted work from before that session,
which is not this work and has not been reviewed. No run is in flight.

## 4. Established and verified

**The phenomenon, from `server-results/step6_result_76573.json`.** Validation series is THREE
points, not twelve: `2.186551e-06`, `1.396616e-06`, `1.393372e-06` m. So `-36.1 %` in the first
interval and `-0.23 %` in the second. Untrained `2.186551e-06`, trained `1.393372e-06`, and with the
ANN forced to zero `2.186748e-06`, i.e. **the ANN contributes +36.28 % and the trained encoder
contributes -0.009 %, nothing**. Against the oracle floor of `2.81e-08` m (`cl_headroom.py`, the FP
model without the absorber against the true 8-state model on the same harness) the headroom was
77.8x at the start and is still **49.6x** after training; **36.7 % of the available headroom was
closed**.

**The run COMPLETED, and the JSON's three-point series is an artefact.** `step6_76573.out` shows
3120 iterations and 13 validations. `fit()` ends with `checkpoint_load_system('_best')`, which does
`self.__dict__ = torch.load(file)`, so `Loss_val` reverts to the snapshot taken when `_best` was
written, at It 520. Anything written onto the fit system during training is lost the same way. Read
the curve from the `.out`, never from `fs.Loss_val` after `fit()`:

| iteration | val sim-RMS [m] | training sqrt loss [normalised] |
|-|-|-|
| 260 | 1.397e-06  (new best) | 1.383e-05 |
| 520 | 1.393e-06  (new best, and the LAST one) | 1.334e-05 |
| 780 | 1.394e-06 | 1.326e-05 |
| 1560 | 1.395e-06 | 1.317e-05 |
| 2340 | 1.396e-06 | 1.313e-05 |
| 3120 | 1.395e-06 | 1.310e-05 |

**This is the central fact of the handoff.** Over the last ten validations the free-run metric is
flat to within 0.3 % and never beats It 520, while the TRAINING loss falls monotonically the whole
time, 1.383e-05 to 1.310e-05. The optimiser is still making progress on the objective it was given
and that progress stops translating into free-run accuracy almost immediately.

**The closed loop is verified against the implementation it replaced and against MATLAB.** On a
fixed 32-window batch: loss rel `1.4e-05`, gradient `1 - cos 5.3e-09`, trajectory `2.8e-07`,
selection scalar within `1.56e-12` m of the frozen recording, 66626/66626 windows carrying the right
controller. MATLAB gate A passes at eight levels including the new L5 at the TRAINING rate
(`6.85e-14` against exact rational arithmetic). See `references/` and D-144.

**The closed-loop gradient is not a usable quantity at large ANN perturbation** (D-145). At
`sigma = 1e-2` the same implementation gives loss `4.81e-02` in float32 and `2.29e-03` in float64, a
factor 20, and two implementations agreeing to `1 - cos = 8.9e-16` at `sigma = 1e-4` disagree at
`1 - cos = 1.87` IN FLOAT64. The rollout is chaotic at that amplitude. This is directly relevant to
any BPTT diagnosis: gradient pathologies here may be dynamical, not numerical.

**What downsampling does to the controller**, computed this session, `Y_op = 0`:

| fs [Hz] | 2/ts [rad/s] | Cnorm(2/ts) | diag(Dc) [N/m] | samples/period at 150 Hz |
|-|-|-|-|-|
| 20000 | 40000 | 0.1307 | 2.844e6  2.914e6  1.509e6 | 133.3 |
| 4000 | 8000 | 0.3701 | 8.055e6  8.253e6  4.275e6 | 26.7 |
| 2000 | 4000 | 0.4485 | 9.761e6  1.000e7  5.180e6 | 13.3 |
| **1000** | **2000** | **0.4540** | **9.881e6  1.012e7  5.244e6** | **6.7** |
| 500 | 1000 | 0.3995 | 8.696e6  8.909e6  4.615e6 | 3.3 |

`Dc` is 1.23x larger at 1 kHz than at 4 kHz, and the dependence is NON-monotone because `2/ts`
sweeps across `Cnorm`'s roll-off. **The controller's roll-off pole `10*w_b` sits at 1000 Hz, which
is ABOVE the 500 Hz Nyquist of a 1 kHz pipeline**, so at 1 kHz the design's own roll-off is not
representable and Tustin folds it. The absorber mode the ANN must learn is at ~150 Hz, i.e. 6.7
samples per period at 1 kHz against 26.7 at 4 kHz.

**Rate change already costs accuracy at 4 kHz** (`p2_rate_compare.py`, D-141): re-discretising at
4 kHz moves `sigma_max(So)` at 150 Hz by **+15.3 %** and phase margin by 3.6 degrees against the
20 kHz loop that generated the data. That is an accepted, stated bias, and 1 kHz will be worse.

**scipy's `cont2discrete` degrades with the rate** (D-144, `test_controller_exact.py` L5py): numerator
coefficients land `5.0e-12` from exact at 20 kHz and `2.05e-10` at 4 kHz, because `2/ts` approaches
the design frequencies and the alternating-sign sums cancel. At 1 kHz this gets worse again; if the
1 kHz path is pursued, measure it rather than assume, using the exact-rational reference already in
that file.

**`concurrent_val = True` is the default and scores the closed loop** (D-146), verified with the
subprocess returning the untrained closed-loop scalar to rel `7.13e-07`. Saves about a third of wall
clock, not a half. Verified on Windows/spawn only; the cluster forks.

## 5. Assumed but not verified

- ~~That the plateau is real.~~ ESTABLISHED, section 4: flat across ten validations to 3120
  iterations, with no new best after It 520. Do not re-run 12 epochs to confirm this.
- **What the training-window error actually is.** Never measured for this run. This is the whole
  point of section 9. The D-095 nf-probe (`gantry_dynamic/training._NfProbe`, now a
  `validation_probes` entry) records `Loss_train_nf`/`Loss_val_nf`, but **`cl_train.py` does not
  attach it**.
- **What "perfect fit on a training window" is numerically.** The supervisor's ~1e-8 is the free-run
  oracle scale; the equivalent number for an `nf = 400` window has not been computed. Compute it, do
  not assume it equals `2.81e-08`.
- **That 1 kHz is viable at all.** The Nyquist argument above is a real objection, not a detail.
  **And that the model rate and the controller rate have to move together, which they do not.**
  The pipeline downsamples the DATA and steps the MODEL at `cfg.ts_new`; `ControllerBank` is built
  at whatever `ts` it is handed (`cl_pipeline.build_closed_loop` passes `cfg.ts_new` today, which is
  a choice, not a constraint). Nothing in `closed_loop.py` requires the two to be equal, so
  "1 kHz model, 4 kHz controller" is expressible without new machinery, at the cost of the rollout
  needing a sub-step for the controller or an explicit statement that the loop is now sampled
  differently from the plant. Which of the two rates actually costs accuracy is the question to
  answer BEFORE picking a pipeline rate: the model rate governs how well the 150 Hz absorber mode
  is represented (6.7 samples per period at 1 kHz), while the controller rate governs `Dc` and the
  sensitivity peak (`+15.3 %` already at 4 kHz). Settled by re-running `p2_rate_compare.py` at 2 kHz
  and 1 kHz and reading `sigma_max(So)` and the phase margin off it, which is minutes and needs no
  training.
- **That the encoder is not the limitation, and this is the cheapest open question in the file.**
  `annoff = 2.186748e-06` against `base = 2.186551e-06` says the trained encoder contributes
  **-0.009 %**, i.e. nothing measurable. Three readings, and the run does not separate them:
  (a) the encoder was already optimal at init, which is plausible because it is the `linear_map`
  reconstructability init (D-055), not a random one, and because the closed loop SUPPRESSES initial
  state error hard (D-142 measured a 393x to 371,000x reduction in the spread between encoder-init
  and true-`x0` free runs, so a mediocre `x0` costs little once the loop is closed);
  (b) the encoder is training but its contribution is masked by the ANN absorbing the same error;
  (c) **the encoder is not training at all**, which would be a defect, not a property.
  Settled cheaply and without a new run: compare encoder parameters before and after training
  (`max |dW|` per tensor, and whether ANY entry moved), and record the encoder's gradient norm
  alongside the ANN's during the retrain. If the encoder parameters are bit-identical after 12
  epochs, it is (c) and that is the finding; if they moved and the score did not, it is (a) or (b)
  and the split matters much less.

## 6. Tried and failed

- Training at `lr = 1e-3` with `ann_route_ix = 0..7` -> NaN in the training loss at iteration 81 ->
  routing to the K=0 rows (X/Y: 0,2,3,5) needs ~1e-7 per D-101/D-102, and `fit()`'s
  `optimizer_kwargs` are IGNORED once `init_model_done` is True, so the lr must go through
  `cfg.lr` into `build_model` -> `cl_train.py` asserts this at startup.
- Raising `nf` to fight long-horizon drift -> refuted on this rig (SLURM 71013) and divergent at
  `nf = 900`, exactly as Ribeiro et al.'s `O(N^3)` beta-smoothness in the within-segment length
  predicts -> see `multiple_shooting.py` docstring.
- Judging the closed loop by a single end-of-rollout number over 20000 steps -> agreed to 4.6e-09 m
  with the ANN off but 7e-04 m with it on -> a single number cannot separate "the algebra is wrong"
  from "two floating-point programs diverge exponentially" -> replaced by a one-step test plus a
  growth curve against an eps-perturbation control.
- Scoring a closed-loop objective with an open-loop selector (variant B) -> the run was invalid, not
  a negative result -> the model was never asked to be good at what it was measured on -> this is
  now structurally impossible; training, validation and selection share one rollout.

## 7. Achieved

**The migration is done and gated.** `model_augmentation/fit_systems/closed_loop.py` holds
`ControllerBank`, `closed_loop_rollout`, `ClosedLoopSimulator`, `WindowControllerIndex`; the seams
(`simulate`, `make_training_data`, `cal_validation_error`, `validation_probes`) are in
`interconnect.py`; `Interconnect.output_only` evaluates the output's dependency cone. No monkey
patches remain in the owned path, exactly one `closed_loop_rollout` exists, and `model_augmentation/`
imports nothing from `scripts/gantry/`. Gates, all passing: `cl_step1_reference.py --check`
(bit-identical), `cl_test_closed_loop.py` C1-C7, `cl_test_seams.py` S1-S7, `cl_test_output_only.py`,
and the four MATLAB gate A scripts captured to `references/gateA_*.txt`.

**The runner is `cl_train.py`**, smoke-tested end to end (`CL_SMOKE=1`, ~90 s), including that the
simulator survives `fit()`'s closing `checkpoint_load_system('_best')`.

**Not achieved: the 12-epoch retrain.** Existing checkpoints do not load by decision (D-144 section
8); the run has to happen and has not.

## 8. The open question

**Does the model fit its own training window?** Two candidate answers and they lead to opposite work:

- **It does** (training-window RMS approaches the per-window oracle floor) while the free-run metric
  sticks at 1.4e-06 m. Then the objective is being solved and the gap is horizon and generalisation:
  the 400-sample training window versus the 48000-sample selection run, which is the 120x
  train/select horizon gap already documented. Work goes to the horizon, the defect/continuity term,
  or the selection metric.
- **It does not.** Then the optimiser is not solving the problem it was given, and horizon work is
  premature. Work goes to the optimiser, the learning rate schedule, the ANN capacity, and the
  gradient path, informed by the BPTT literature.

The evidence that chooses: `Loss_train_nf` at `nf = 400` against a per-window oracle floor computed
on the same windows. Both numbers are cheap once the probe is attached.

**Section 4's curve already leans, and it should be read carefully rather than as settling this.**
The training loss falls monotonically (1.383e-05 to 1.310e-05) across exactly the span where the
free run is flat. That rules out "the optimiser has stalled" as a description of the WHOLE problem:
something is still being minimised. What it does NOT tell you is whether the training-window error
is anywhere near the floor, because a loss can fall steadily toward a value that is still orders
above it. So the branch to test first is the first one, and the number that decides it is the ratio
in section 10, not the shape of the training curve.

## 9. Next action

**Attach the D-095 nf-probe to `cl_train.py` and launch the 12-epoch retrain**, so the
training-window error is recorded per validation alongside the free-run selector. The retrain has to
happen regardless (section 7), and this makes it answer section 8 at no extra cost. The probe is now
a declared extension point, so attaching it is one line:

```python
from gantry_dynamic.training import _NfProbe
fs.validation_probes = (_NfProbe(fs, cfg.nf, train_list[0], val_data, do_print=True),)
```

Compute the per-window oracle floor in the same script: the FP model without the absorber against
the true 8-state model over the same `nf = 400` windows, which is what `cl_headroom.py` already does
for the free run and what makes the comparison in section 10 defensible.

Record two more things in the same run, because they are free once it is running and they settle
section 5's encoder question without a second job:

- **the encoder's gradient norm next to the ANN's**, per validation. `_NfProbe._joint_probe`
  already walks the blocks for its orth meters, so this is a few lines in the same place;
- **the encoder parameter delta against the initial state**, `max |W - W_init|` per tensor, at the
  end. Snapshot the encoder `state_dict` before `fit()` and diff it afterwards.

If the encoder parameters are unchanged after 12 epochs the diagnosis is a defect in the gradient
path to the encoder, and that outranks everything else in this file.

## 10. Acceptance criterion

The diagnostic is done when `Loss_train_nf` at `nf = 400` is on the record next to the per-window
oracle floor computed from the same windows, and the ratio between them is stated. The floor is
data-derived: it is the error the absorber-free FP model makes on those windows against the true
model, so it is the best any augmentation of this class could do there. A ratio near 1 selects the
first branch of section 8; a ratio of order 10 or more selects the second.

For the retrain itself the sanity band is unchanged: a result within a few percent of 36.3 %
improvement over the untrained `2.1866e-06` m confirms the migration did not change the training
behaviour. It is a band, not a parity assert. `cl_train.py` prints a `concurrent-path check` line
comparing `Loss_val[0]` against the recorded untrained scalar; if that line does not say "closed
loop confirmed", stop and investigate before reading any other number.

## 11. Read these first

1. `server-results/step6_result_76573.json` and `step6_76573.out` — the run whose behaviour is the
   subject, and how it actually ended.
2. `docs/decisions.md` D-144, D-145, D-146 — what the closed loop is now, why the gradient is
   unusable at large perturbation, and the concurrent-validation contract.
3. `scripts/gantry/closed-loop-controller/cl_train.py` — the runner to modify, and its assertions.
4. `model_augmentation/fit_systems/closed_loop.py` — the rollout, the `xc = 0` reasoning, and the
   Kessels attribution.
5. `scripts/gantry/closed-loop-controller/PLAN-move-to-model-augmentation.md` section 3.8 — the
   measured cost table, so no time is spent re-deciding runtime questions.

## 12. Do not

- Do not re-run the equivalence, gradient-precision or validation-precision experiments (section 2).
- Do not use an ANN perturbation of `1e-2` for any gradient comparison; it is chaotic (D-145).
- Do not raise `nf` as the first response to the free-run gap; refuted, section 6.
- Do not change the controller design (`ruleOfThumb`, `Cnorm`, `kappa`) while diagnosing training.
  If the 1 kHz question leads there, treat it as a separate, stated change with its own gate.
- Do not commit without deciding what to do about the pre-existing uncommitted work in the four
  shared files (section 3).

## 13. Operational

Env `GraduationProject`. Launch, per the live-output convention:

```
cd scripts/gantry/closed-loop-controller
CL_EPOCHS=12 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 conda run --no-capture-output \
  -n GraduationProject python -u cl_train.py
```

Result JSON lands in `runs/cl_train_<TAG>.json` (`CL_TAG` or `SLURM_JOB_ID`, else `local`).

**TRAP, and it invalidated a number in the previous run's JSON.** `fit()` ends with
`checkpoint_load_system('_best')`, which does `self.__dict__ = torch.load(file)`. Every attribute
written onto the fit system during training, INCLUDING `Loss_val`, `Loss_train`, `Loss_val_nf` and
anything a probe appends to `fit_sys`, is replaced by the snapshot taken when `_best` was last
written. That is why `step6_result_76573.json` records a three-point validation series for a run
that did 13 validations. Keep probe history on the PROBE OBJECT, which the runner holds, and read
the selector curve from the `.out` file or from a copy taken before `fit()` returns. Any conclusion
drawn from `fs.Loss_val` after `fit()` is a conclusion about the best checkpoint's past, not about
the run.
`concurrent_val` is ON by default; `CL_CONCURRENT=0` disables. **The 12-epoch run is a cluster job**:
extrapolating the step-0 profile it is on the order of a day of wall clock locally (about 18 min of
training plus 8.5 min of validation per epoch, the latter overlapped when concurrent). `CL_SMOKE=1`
runs the whole path on truncated data in ~90 s and is the right first check after editing the
runner. The literature part of the task triggers the `deep-research` skill per D-121; the standard
NN-training and BPTT questions are exactly what it is for.

## 14. Delegation

None for section 9; it is a targeted edit plus a launch, and the context-holding session is better
placed. For the literature sweep (BPTT pathologies, recurrent training practice), use the
`deep-research` skill, one subagent per independent seed question, per D-121. Do not use a subagent
to check the numeric results of the retrain.
