# Handoff: verify the controller against the recorded closed-loop signals, and settle its placement using Kessels Chapter 5
**From**: session of 2026-08-16 | **Branch**: Augmentation | **Effort suggested**: high

## 1. Task

Establish that the project's reconstruction of the feedback controller `Cfb` is numerically exact
against the signals the data generator actually recorded, and then, reading Kessels (2025)
Chapter 5 properly rather than through this repo's second-hand comments, state which of two
placements the controller should take in `model_augmentation/`. The verification comes first and
is purely a measurement: every record in `data/gantry/matlab/trajectory/augmentation/` stores
`r_sim`, `u_fb`, `f_sim`, `u_total` and `y`, so `Cfb` can be checked directly instead of trusted.
Produce, per record and per channel, the residual between the recorded `u_fb` and the `u_fb`
recomputed from `Cfb`, `r_sim` and `y`, plus the residual of the identity `u_total = u_fb + f_sim`.
Then write a short position, with section and remark numbers from Kessels, on whether the
controller should become a block whose states join the interconnect state vector, or stay a
separate subsystem stepped alongside the model in the rollout. Do not implement either yet.

## 2. Out of scope

- **Any edit to `model_augmentation/`.** The user's instruction is explicit: verify the controller
  before that folder is touched. The implementation is the session after this one.
- **Changing `ann_route_ix`.** The user has decided: all states, `(0,1,2,3,4,5,6,7)`. It is not
  currently that, see section 5. Do not re-litigate it and do not "improve" it to a subset.
- **Retraining anything.** Variants A and B are done and recorded (section 7). No training run
  should be launched in this session.
- **Anything to do with `ClosedLoopLossMixin` as a destination.** The user's direction is that the
  controller belongs in the MODEL, not in the loss. The loss-based implementation is what is being
  replaced, so do not improve it, extend it, or argue about it. The one thing to carry across is
  its algebra, section 4.
- **The replay/artefact work.** Closed as D-139. `gen_annoff_data.py` is fixed.

## 3. Where things stand

Branch `Augmentation`, last commit `931cb75`, tree dirty across `scripts/gantry/`, `docs/`,
`tasks/`, `Matlab-scripts/`. Nothing running; both training processes have exited.

New and uncommitted this session: `scripts/gantry/dc-accumulation/compare_annoff_replay.py`, the
`x0` fix in `scripts/gantry/dc-accumulation/gen_annoff_data.py`, baseline and pickling fixes in
`scripts/gantry/closed-loop-controller/{train_variants.py,loss_variants.py}`, run logs under
`scripts/gantry/closed-loop-controller/runs/`, D-139 in `docs/decisions.md`, and two rows in
`docs/gantry-augmentation-problem-log.md` section 12.

## 4. Established and verified

**The records contain everything needed to check the controller.** `whosmat` on
`data/gantry/matlab/trajectory/augmentation/V1_standstill_Yp10.mat` returns, at 20 kHz and
`(240000, 3)` unless noted: `u_total`, `u_fb`, `f_sim`, `y`, `r_sim`, `x_logical` `(240000,6)`,
`delta_a` `(240000,1)`, `vdelta_a` `(240000,1)`, `x_aug` `(240000,2)`, `Y_trajectory`, `t_sim`,
plus `fs`, `dt`, `split`, `amp_rms`, `seed`, `track`. **The loader reads only `u_total` and `y`**
(`gantry_dynamic/data.py:73-75`, `_load_u` returns `d['u_total']`), so `r_sim`, `u_fb` and `f_sim`
have never been loaded or checked by any Python in this project.

**The generating controller is per-record and frozen, not gain-scheduled.**
`Matlab-scripts/Augmentation/data/generate_gantry_lti_augmented.m:90-91` builds
`Cfb(j,j) = ruleOfThumb(fbw, sys(j,j), ts)` from the plant linearised at that record's `Y_op`, and
`gtd_run_simulation.m:33` applies it as `lsim(plant.Cfb, r_sim - q_with)`. `lsim` is LTI, so there
is no scheduling along `Y`, not even on T6-T14 where `Y` sweeps `[-0.30, 0.30]`.

**Training operating points**, from `gtd_build_records.m`: T1-T5 at `-0.30, -0.15, 0.00, +0.15,
+0.30`; **T6-T14 all at `0.00`**. No training record is at `0.10`. Validation: V1 `+0.10`,
V2 `-0.22`, V3 `+0.10`, V4 `-0.10`.

**The closed-loop model has ONE exogenous input.** Lecture 11 slides 5-7 give the structure to
build: plant `G0` with controller `K` in feedback, excitation entering at two points, `r1` before
the controller and `r2` at the plant input, lumped as `r = K(q)r1 + r2` so that `u = r - K y` and
`y = G_cl r + S0 v`. With `snr=None` the noise term `v` is zero, so `y = G_cl r` exactly. With
`r1 = r_sim`, `r2 = f_sim`, `K = Cfb`, and using `u_total = u_fb + f_sim` and
`u_fb = Cfb(r_sim - y_data)`, the lumped reference is `r = u_total + Cfb*y_data`, and substituting
into `u_model = r - Cfb*y_model` gives `u_model = u_data + Cfb*(y_data - y_model)`. Consistent with
the previously measured equivalence of the two forms to `1e-07 m` against a `0.2 m` gap.
**This algebra is what the controller block implements, wherever it ends up living: compute
`r = u_total + Cfb*y_data` once per record in the loader, then drive the plant with
`u_plant = r - Cfb*y_model`. Only `r` is needed as an exogenous input; `y_data` does not enter the
rollout.** Note the two forms differ in what the controller filters, and therefore in its initial
state: driven by `y_data - y_model` it starts at ~0 at a window start, driven by `y_model` it does
not and needs Remark 5.4 reconstruction. That is the cost item in section 8.

**Kessels keeps the controller out of the model's state vector.** In (5.13) the constraints are
separate: (5.13b) EA model, (5.13c) control input, (5.13d) FB controller dynamics. The encoder
(5.13a) returns only EA model states. Figures 5.2 and 5.4 draw FB/FF in one box and the EA model
in another. Remark 5.4: use the machine's true `x^FB` if available, otherwise reconstruct it from
`y`, `r` and the known controller assuming `x^FB = 0` at `k = 1` of the record, and initialise each
truncated window from that reconstruction. Remark 5.3: if `h` is accurate, initialise position
states from the measured outputs via `h^-1` so "the encoder needs to initialize fewer states,
making the training procedure easier/faster to converge". **Kessels therefore does not put
controller states into the encoder.**

**Kessels zero-inits the networks deliberately** (p159): weights close to zero rather than the
default `U(-sqrt(1/nz), sqrt(1/nz))`, so the NNs start with negligible contribution, and because
"in case of closed-loop systems, due to this initialization, the EA model is less likely to be
unstable at the start of training". Our `zero_init_feed_forward_nn` matches this. Not a defect.

**Remark 5.2 states this thesis's contribution as an open problem**: enforcing orthogonality of
the extension and augmentation terms to the FP terms "need[s] to be explored... developed for
linear-in-the-parameter cases [112], but are still missing for more complex problems such as the
problem discussed here".

**Closed-loop and open-loop replay error of the baseline**, measured, sample 0, block-mean `u`,
4 kHz float32 (D-139):

```
                    open loop rms [m]              closed loop rms [m]
V1 standstill  2.992e-08 2.995e-08 2.260e-06   4.149e-07 3.768e-07 3.742e-06
V2 motion      2.154e-04 2.416e-04 4.597e-05   3.356e-07 4.094e-07 3.769e-06
```

Closing the loop drives the open-loop mean of `+8.2e-05 m` to `~1e-12 m` on every record, and the
closed/open rms ratio flips with the band: `13.9/12.6/1.66` on V1 (130-180 Hz, `sigma_max(So)=2.07`
at 150 Hz) against `1.56e-03/1.69e-03/8.20e-02` on V2 (near 10 Hz, `So = 0.021`).

## 5. Assumed but not verified

- **That `Cfb` as rebuilt in Python equals the `Cfb` that generated the data.** This is the whole
  point of section 1 and nothing has ever checked it. `loss_variants.controller_ss` calls
  `p2_rate_compare.build_cfb_at`, and the MATLAB side is `ruleOfThumb`. Settled by the residuals
  in section 10.
- **That `u_total = u_fb + f_sim` holds in the stored records.** Inferred from the generator's
  structure, never measured. One line to check.
- **That `x[6]` is identically zero in the current wiring.** Reasoned from
  `gantry_dynamic/model.py:133-136` (ANN additive on `route_ix`, `phy_block` additive on
  `PHY_IX = 0..5`, nothing on row 6) plus `interconnect.py:77-89` (outputs start at zeros and
  accumulate). Consistent with problem-log section 13 and with the observed
  `x[6] enc=0.0000e+00`, but that observation was taken at an untrained checkpoint where it is
  trivially zero, so it is not independent evidence. Settled by one forward pass with a nonzero
  `x[6]`.
- **`ann_route_ix` is currently `(3,4,5,7)`**, verified at `gantry_interconnect_dynamic.py:69` and
  `train_variants.py:75` (config default is `(1,4,6,7)`, `config.py:63`). The user has decided it
  must be all eight. **Whether all-states routing reintroduces the D-066/D-067 position-row
  instability is not known** and is not this session's question.
- **What Remark 5.5 and the rest of 5.3.2 say.** Only pp147-160 of Chapter 5 were read this
  session. The industrial use case (5.3.2, thesis pp167-180) is the closed-loop FB+FF wire bonder,
  the closest published setting to this project, and it has not been read.

## 6. Tried and failed

- **Variant B, closed-loop training loss, 4 epochs at `lr=1e-7`** -> val sim-RMS rose monotonically
  `1.66e-04 -> 5.48e-02 -> 1.33e-01 -> 2.32e-01 -> 3.43e-01 m` while train loss fell 2.1 % ->
  **the run is invalid, not a negative result**: it optimises a closed-loop objective and selects
  and scores on an open-loop free run, so the model was never asked to be good at what it was
  measured on -> `runs/variantB_lr1e-07.log`, problem-log section 12.
- **Variant A, production open-loop loss, same settings** -> val sim-RMS `5.91e-03, 5.75e-03,
  1.00e-02, 8.26e-03 m`, train loss down 1.5 %, `VERDICT: ANN inactive`, `augmentation vs baseline
  (same init) +0.0%` on all channels -> **the best checkpoint was the untrained model**, because
  selection runs on a 48000-step free run dominated by `z=1` accumulation while training sees only
  400-sample windows; the available prize is a factor 2.35 on Y (encoder-init baseline
  `2.121e-04 m` against FP+MSD oracle `9.023e-05 m`) and drift moves the metric by 36x -> reproduces
  D-067's "best checkpoint = epoch 0 on all 20 epochs" -> `runs/variantA_lr1e-07.log`.
- **Attributing the ANN's failure to §13's `W^a` dead zone** -> wrong: §13 measures the ANN's final
  layer gradient at `6.4e-03`, and val sim-RMS moved 36x in one epoch, so the ANN is learning ->
  the `x[6] = x[7] = 0` in the verdict was read at the untrained best checkpoint and is circular
  evidence -> this session, corrected in the section 12 row.
- **Evaluating variant B** -> `PicklingError: Can't pickle <class 'loss_variants.FitSys_B'>` after
  3.2 h of training, losing its verdict block -> `type()` sets `__module__` but does not bind the
  name in the module -> fixed at `loss_variants.py:154-176`.

## 7. Achieved

**Implemented and validated.** `scripts/gantry/dc-accumulation/compare_annoff_replay.py`: rolls the
same record through `Gantry_State_Block` standalone and through `fs.hfn` with the ANN forced to
zero, with three initialisations, and diffs them. The two implementations agree to
`1.5e-11/1.5e-11/6.5e-09 m` on V1 and `5.0e-09/6.9e-09/2.0e-08 m` on V2, under the `3.9e-08 m`
numerics floor. Figures `figures/annoff_replay_compare_<record>.png`.

**Implemented, validated by its own runs.** `train_variants.py` now computes and passes the
encoder-init baseline (`baseline_encinit_nrms`, captured pre-training per D-089) and starts the
true-x0 baseline at `K0` rather than sample 0; `lr`, `Y_op` and the output paths are
env-parameterised so a sweep cannot overwrite an earlier run. `loss_variants.attach` no longer
breaks checkpoint saving.

**Fixed.** `gen_annoff_data.py` initialised velocities to normalised zero, which is physical
`x_mean` and injected `+1.30e-04 m` on X and `+1.48e-04 m` on Y through the `K=0` axes. Any `.mat`
that script wrote before 2026-08-15 carries that artefact and must be regenerated.

## 8. The open question

**Should `Cfb` become a block whose states join the interconnect state vector, or stay a separate
subsystem stepped alongside the model in the rollout?**

- **In the interconnect (user's initial proposal).** The closed loop becomes a property of the
  model, so training, evaluation, diagnostics and checkpoint selection cannot silently disagree,
  which is the exact failure that invalidated variant B. Orthogonality is unaffected, because the
  FP block is untouched and the projection basis is still `d(FP)/dtheta`. Costs: `nxd` grows by the
  controller order; controller states must be excluded from the encoder (Remarks 5.3, 5.4); and in
  the lumped-`r` form the controller filters `y_model`, a large signal, so `xc` at a window start
  is not zero and needs Remark 5.4 reconstruction.
- **Separate subsystem in the rollout (what Kessels does).** Matches (5.13c)/(5.13d), keeps
  `nxd = 8`, keeps the encoder untouched, and preserves the cancellation that makes `xc = 0` exact
  per window in the residual form. Cost: the loop must be applied in training, validation and
  selection by discipline rather than by construction.

What would choose: a careful reading of 5.2.3 and 5.3.2 on how Kessels actually simulates the
industrial closed-loop case per truncated window, plus whether the controller order makes `nxd`
growth material. `Cfb` is 3 diagonal SISO channels via `_tf_to_ss_batch`
(`loss_variants.py:44-60`); its per-channel order has not been printed.

Merging `Cfb` into the FP block is already excluded: parameter sensitivities would come out
premultiplied by `So`, making the orthogonality claim controller-dependent, and it breaks the
standalone-baseline negation test that runs the FP block alone with learned `theta_hat`.

## 9. Next action

Write `scripts/gantry/closed-loop-controller/verify_cfb_against_records.py`, reading only, which
for each of the 18 records in `data/gantry/matlab/trajectory/augmentation/`:

1. loads `r_sim`, `y`, `u_fb`, `f_sim`, `u_total` at 20 kHz,
2. checks the identity `u_total - (u_fb + f_sim)` and reports max abs and rms per channel,
3. rebuilds `Cfb` at that record's `Y_op` via `loss_variants.controller_ss(Y_op, ts)` with
   `ts = 1/20000`, simulates it on `r_sim - y` from zero initial state, and reports
   `u_fb_recomputed - u_fb` as max abs and rms per channel, and the same after discarding the
   first 0.5 s to separate a transient from a structural mismatch,
4. prints the controller order per channel and the total `n_FB`.

Rationale: this is the one measurement that decides whether any closed-loop work in this project
rests on a correct controller, it needs no training, and it also returns the `n_FB` that the
section 8 decision depends on. Report every channel and every record; do not summarise to a
pass/fail.

## 10. Acceptance criterion

Per record and per channel, after discarding the first 0.5 s:

- `max |u_total - (u_fb + f_sim)|` at the float32 storage floor of the records, i.e. `<= 1e-3 N`
  relative to the rms of `u_total`. This is an exactness identity, not a fit; anything larger means
  the stored signals are not what the generator's block diagram says.
- `rms(u_fb_recomputed - u_fb) / rms(u_fb) <= 1e-3`. The threshold is data-derived: the records are
  stored as `single`, so a relative agreement of `1e-3` is roughly six digits above the storage
  quantisation and cannot be reached by an incorrect controller. A larger residual means `Cfb` as
  rebuilt in Python is not the controller that generated the data, and every closed-loop result in
  this project, including D-139's closed-loop numbers, needs revisiting.

If the residual is large only in the first 0.5 s, that is Remark 5.4's initial-state question and
is a finding, not a failure.

## 11. Read these first

1. `literature/augmentation/kessels2025_ai-control.pdf`, Chapter 5, thesis pp147-183. Sections
   5.2.1-5.2.3 and Remarks 5.1-5.5 are the framework; 5.3.2 is the closed-loop FB+FF wire bonder,
   the closest published setting to this project and unread.
2. `scripts/gantry/closed-loop-controller/loss_variants.py:44-66,116-176`, the controller state
   space construction and the closed-loop loss whose correctness section 4 establishes.
3. `Matlab-scripts/Augmentation/data/generate_gantry_lti_augmented.m:90-91` and
   `gtd_run_simulation.m:17-33`, the controller that actually generated the data.
4. `literature/experiment-design/System-identification/Lecture 11.pdf`, slides 5-7, the lumping
   `r = K(q)r1 + r2` and the unified closed-loop signal relations.
5. `docs/gantry-augmentation-problem-log.md` section 12, the two rows for variants A and B, and
   section 13 for the `W^a` dead zone that section 6 partially retracts.

## 12. Do not

- Do not modify `model_augmentation/`. That is the next session, after this verification.
- Do not change `ann_route_ix` to anything other than all eight states; the user has decided.
- Do not launch a training run.
- Do not propose putting the controller in the loss. The user's direction is that it belongs in the
  model, and `ClosedLoopLossMixin` is the implementation being replaced.
- Do not merge `Cfb` into `Gantry_State_Block`; excluded in section 8 with reasons.
- Do not use `ramp_fraction` as evidence about an offset; it is variance-based and scores `0.00 %`
  on a pure DC shift.
- Do not quote the variant B run as evidence that closed-loop training fails.

## 13. Operational

```
cd "scripts/gantry/closed-loop-controller"
PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 \
  conda run --no-capture-output -n GraduationProject python -u verify_cfb_against_records.py
```

Runs at 20 kHz on 240000-sample records; 18 records, expect a few minutes, no GPU. Records are in
`data/gantry/matlab/trajectory/augmentation/`. `Y_op` per record is in
`baseline_drift_replay.py:73-78` for the four validation records and in
`Matlab-scripts/Augmentation/data/gtd_build_records.m:36-56` for all eighteen.

Note for any long run in this session: harness-managed background jobs have been killed mid-run.
Launch detached with PowerShell `Start-Process ... -RedirectStandardOutput <log> -NoNewWindow`, and
tail the log.

## 14. Delegation

None. The verification is two files and a known data directory, and the Kessels reading is a single
document that the context-holding session should read itself rather than receive summarised.
