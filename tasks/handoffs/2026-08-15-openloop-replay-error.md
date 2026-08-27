# Handoff: what is the true baseline error when closed-loop data is replayed open loop
**From**: session of 2026-08-15 | **Branch**: Augmentation | **Effort suggested**: high

## 1. Task

Establish the correct open-loop replay error of the baseline model against the closed-loop
records, per channel, as a mean and an rms, on at least one standstill record and one
motion-profile record. "Correct" means free of the four initialisation and resampling artefacts
listed in section 4, each of which is between 100x and 100000x the quantity being measured on a
standstill record. Two independent implementations currently disagree by five orders of magnitude
on the V1 Y channel (`4.4e-07 m` against a reported `1e-1 m`); reconcile them and produce one
defensible number per channel per record. The deliverable is the number plus a figure that
annotates the mean, not the ramp fraction.

## 2. Out of scope

- **Variant B training.** The oracle learnability gate passed this session (section 4) but no
  training run should be launched until the replay error is settled, because it is the reference
  the run would be judged against.
- **The open-loop generated records** (`OE*`, `OT*`, `OV*`, `OL1`) and everything in
  `scripts/gantry/open-loop/`. That is the sibling handoff's territory,
  `scripts/gantry/open-loop/HANDOFF-open-loop-data-and-training.md`. Its drift is a different
  mechanism (physical rectification, common mode) and is already characterised.
- **Fixing `model_augmentation/` or `scripts/gantry/gantry_dynamic/`.** Graft at instance level
  as `loss_variants.py` does.
- **The MATLAB three-curve comparison.** `export_baseline_closedloop.m` is written and unrun;
  `baseline_matlab_compare.py` is stale (predates the sample-0 fix). Leave both.

## 3. Where things stand

Branch `Augmentation`, last commit `931cb75`, tree dirty across `scripts/gantry/`,
`Matlab-scripts/`, `docs/`, `tasks/`. Nothing running.

New this session, all in `scripts/gantry/closed-loop-controller/`:
`baseline_drift_replay.py` (working, current), `baseline_matlab_compare.py` (stale),
`export_baseline_closedloop.m` (written, never run in MATLAB), and figures under
`figures/baseline_*` with per-record suffixes for V2 and no suffix for V1.

## 4. Established and verified

**The four artefacts, each measured, each larger than the signal on a standstill record.**

- **Mid-record start.** Starting a replay at `K0 = max(na, nb) = 17` instead of sample 0 inherits
  absorber momentum the baseline cannot represent. On V1 this produced `7.84e-04 m` of Y offset
  against a true `4.4e-07 m`, a factor of 1800. Mechanism: `plant.py:44`,
  `dY(inf) = (MA/cy) * vda(t0) = 0.101 * vda(t0)`. `K0` is an ENCODER constraint
  (`gantry_interconnect_dynamic.py:136`, "first sample with a full encoder window") and does not
  belong in a replay.
- **Normalised zero is not physical rest.** `norm.x_mean` on the velocity states is
  `dX +8.403e-05`, `dTh -3.167e-06`, `dY +1.455e-04 m/s`. Writing `np.zeros(nx)` into a
  NORMALISED state vector therefore starts the model moving, and on the `K = 0` axes that
  integrates to `v0 * tau` with `tau_X = 1.546 s`, `tau_Y = 1.010 s`: **`+1.30e-04 m` on X and
  `+1.47e-04 m` on Y**. This is a live bug in
  `scripts/gantry/dc-accumulation/gen_annoff_data.py:106-107`, which sets
  `x0 = np.zeros((1, nx))` and then corrects only `[:3]` via an identified affine map. Its
  self-check cannot catch it: the check verifies the first OUTPUT, which depends on positions
  only. Evidence: scratchpad `check_xmean.py`, this session.
- **Point-sampled `u`.** D-087: `u[::D]` instead of the per-hold-interval block mean leaves a
  nonzero-mean force error integrating to `-3.5e-4 m` on Y and `+6e-5 m` on X. The pipeline
  loader handles this (`data.py:78-95`); ad-hoc scripts often do not. Hit and fixed this session.
- **Absorber-momentum seeding**, open-loop overlays only. `overlay_records.py:44-52`: copying the
  six shared states drops `ma*vda(0)`, giving a Y offset of exactly `-(ma/cy)*vda(0)`, ratio 0.92
  on all eight records, reaching `4e-03 m` against a `~2e-06 m` real discrepancy.

**The measured replay error, corrected.** `baseline_drift_replay.py`, sample 0, prescribed IC
`x0 = [0, 0, Y_op, 0, 0, 0]` (`closed_loop.x0_for`), block-mean `u`, `Gantry_State_Block`
standalone, 4 kHz float32:

```
open-loop error       rms [m]                     mean [m]
V1 standstill   2.992e-08  2.995e-08  2.260e-06   -1.819e-08  -1.819e-08  -4.441e-07
V2 motion prof  2.154e-04  2.416e-04  4.597e-05   +8.204e-05  +8.214e-05  -6.767e-08
```

The open-loop error is NOT zero-mean: on V1 X the mean is 61 % of the rms. Closed loop the mean
is `1e-12` to `1e-10 m`, zero to precision, because the controller's `z = 1` pole forces it there
(`RESULT-loop.md` P3: `sigma_max(So)` at DC is `3.7e-10`).

**Closed/open rms ratio flips with the excitation**, reproducing `RESULT-loop.md` section A:
V1 `13.9 / 12.6 / 1.66` (loop worse, `sigma_max(So) = 1.80` at 150 Hz), V2
`1.56e-03 / 1.69e-03 / 8.20e-02` (loop 640x better, `sigma_max(So) = 0.021` at 10 Hz).

**Numerics are not a factor.** Pipeline block against `deriv6` float64: `3.9e-08 m` closed loop.
Controller float32 against float64: `5.2e-08 m`, bounded over 48000 samples. Rate 4 kHz against
20 kHz is the only large effect, `1.6e-06 m` on Y.

**`ramp_fraction` is blind to DC.** It returns variance explained and `np.var` subtracts the mean
first, so a pure offset scores `0.00 %`. `summarise` in `baseline_drift_replay.py` now prints
mean and slope alongside.

**The two formulations of the closed-loop input are equivalent.** `u = u_data + Cfb(y_data -
y_model)` with `xc = 0` equals driving from `r` with `x_FB` reconstructed (Kessels Remark 5.4),
to `1e-07 m` against a `0.2 m` gap, at both rates. Scratchpad `b_form_equivalence.py`.

**Oracle learnability passes** in the training configuration: absorber detuning sweep has its
minimum at exactly 150 Hz, monotone both sides, on both records at both rates. Contrast `1.09e+01`
open / `5.63e+01` closed at 4 kHz, `6.75e+02` / `1.42e+03` at 20 kHz.

## 5. Assumed but not verified

- **That `Gantry_State_Block` standalone equals `fs.hfn` with the ANN output zeroed.** My numbers
  use the former (6 states, no ANN object). `gen_annoff_data.py` uses the latter,
  `nx = 6 + nx_ann = 8`, with `ann.forward` monkeypatched to return zeros. The two ANN states
  still exist and evolve; only the output contribution is zeroed. Whether the routing
  (`ann_route_ix=(3,4,5,7)`) makes these identical is untested and is the leading suspect for the
  five-order disagreement. Settled by running both on V1 and diffing `y`.
- **That the reported `1e-1 m` V1 Y offset is an offset at all.** `Y_op = 0.10 m = 1e-1 m`, so a
  panel plotting absolute output rather than error sits there by construction. Settled by
  checking which quantity that panel plots.
- **What `dm.load_T` and `dm.build_pipeline` return.** Not read. `K0` is unpacked at
  `gen_annoff_data.py:62` and never used again, so there is no `K0` slicing in that script, but
  whether `load_T` returns a full or pre-sliced record is unchecked.
- **That V2's `8.2e-05 m` X offset is the centre-of-mass shift rather than the absorber.** The
  open-loop handoff line 55 argues the X discrepancy comes from `mh = mhr + ma` shifting the
  centre of mass, not from the absorber, which acts along Y. Consistent with V2's Y mean being
  `1000x` smaller than its X mean, but not demonstrated here.

## 6. Tried and failed

- **Starting the replay at `K0 = 17`** -> V1 Y offset `7.84e-04 m`, ramp 42 %, apparently showing
  the loop removing a large drift -> the offset was inherited absorber momentum, `0.101 * vda(17)`
  with `vda(17)` about 0.36 sigma of a `2.16e-02 m/s` distribution; from rest it is `4.47e-08 m`
  -> `baseline_drift_replay.py` git history, both runs this session.
- **Concluding from V1 that the offset was entirely a `K0` artefact** -> V2 then showed `8.2e-05 m`
  of genuine offset and `1.37e-05 m/s` of drift from rest -> V1 is the one record where the real
  effect is small, so it is the worst record to generalise from -> the two runs above.
- **Point-sampling `u[::5]` in an ad-hoc oracle script** -> the absorber detuning sweep went flat,
  contrast `1.00`, reading as "the absorber is invisible at 4 kHz" -> D-087's nonzero-mean force
  error, `~3.5e-04 m`, swamped the `1e-06 m` discrepancy -> scratchpad `oracle_windowed.py`, first
  run against second.
- **Differencing the 20 kHz and 4 kHz replays index-by-index** -> `max |dy| = 2.2e-05 m`, 50x the
  rms of either -> both skip `K0 = 17` SAMPLES, which is `0.85 ms` at 20 kHz and `4.25 ms` at
  4 kHz, and `3.4 ms` is half a period at 150 Hz, so it differenced two nearly antiphase signals
  -> `baseline_matlab_compare.py` git history.
- **Attributing the V1 offset to absorber-momentum seeding** (`overlay_records.py`) -> that
  mechanism predicts exactly zero from rest and its documented maximum is `4e-03 m`, 25x below the
  reported `1e-1 m` -> `overlay_records.py:44-52`.

## 7. Achieved

**Implemented and validated.** `baseline_drift_replay.py`: four figures per record with a
per-record suffix, `summarise` reporting rms, mean, slope and ramp. Runs on any of the four val
records via `BDR_RECORD`. Numbers in section 4. Figures for V1 (unsuffixed) and V2
(`_V2_aprbs_Ylow`) are current and free of the `K0` bug.

**Implemented, not validated.** `export_baseline_closedloop.m`, never run.
`baseline_matlab_compare.py`, runs but predates the sample-0 fix, so its figure is stale.

## 8. The open question

**Why do two implementations of the same experiment disagree by five orders of magnitude on the
V1 Y channel?**

Same experiment: baseline, ANN off, open-loop replay of the closed-loop V1 record, from rest.
`baseline_drift_replay.py` gives `-4.44e-07 m`; `gen_annoff_data.py` reportedly gives `~1e-1 m`.

Candidates, with what would choose:

- **The reported Y number is the absolute output, not the error.** `Y_op = 0.10 m`. Check which
  quantity the panel plots. Cheapest, check first.
- **`x_mean` on the velocities**, section 4. Predicts `+1.47e-04 m` on Y and `+1.30e-04 m` on X.
  The X prediction matches the reported `1e-4` on X exactly, so this mechanism is almost certainly
  present; it does not reach `1e-1` on Y. Check by building `x0` physically and renormalising.
- **`fs.hfn` with ANN states is not `Gantry_State_Block`.** Diff the two rollouts on V1.

The X channel is already explained: `1.30e-04 m` predicted against `1e-4` reported, and the true
X error is `1.8e-08 m`, so that figure's X offset is about 4000x the signal and is an artefact.

## 9. Next action

Run both implementations on V1 and diff the outputs sample by sample. Build the comparison in
`scripts/gantry/dc-accumulation/`, importing `pipeline_rollout` and `build_phy_block` from
`scripts/gantry/closed-loop-controller/baseline_drift_replay.py` for one side and
`gen_annoff_data.py`'s own model construction for the other, with both given a physically-at-rest
`x0` built as `x0_norm = (x0_phys - x_mean) / std_x`. Print rms, mean and slope per channel for
each, and the max absolute difference between them.

This settles the open question in one run: if the two agree once `x0` is built physically, the
`x_mean` bug was the whole story and the corrected number is section 4's; if they still diverge,
it is the ANN-state path and the diff will localise it.

## 10. Acceptance criterion

One number per channel per record for the open-loop replay error, as a mean and an rms, produced
by two independent implementations agreeing to within the numerics floor established this session,
`3.9e-08 m` (pipeline block against `deriv6` float64, closed loop).

For reference, the values to reproduce or refute:

```
V1 standstill   mean  -1.819e-08  -1.819e-08  -4.441e-07 m
V2 motion prof  mean  +8.204e-05  +8.214e-05  -6.767e-08 m
```

A disagreement larger than `3.9e-08 m` means one implementation still carries an artefact, and the
diff localises which. Threshold is data-derived: it is the measured agreement between two plants
already gated against MATLAB, not a tolerance chosen for convenience.

## 11. Read these first

1. `scripts/gantry/closed-loop-controller/baseline_drift_replay.py`, the long comment in `main()`
   on why `K0` does not belong in a replay, and `summarise` on why ramp fraction is blind to DC.
2. `scripts/gantry/dc-accumulation/gen_annoff_data.py:85-110`, the `x0` construction and its
   self-check, which is where the `x_mean` bug lives.
3. `scripts/gantry/open-loop/overlay_records.py:44-56`, the momentum-seeding artefact written out
   with its exact formula and the 0.92 ratio; the clearest statement in the repo of how an init
   artefact buries a real discrepancy.
4. `scripts/gantry/closed-loop-controller/RESULT-loop.md` section A, the closed/open ratios per
   record that section 4's numbers now reproduce.
5. `docs/gantry-augmentation-problem-log.md` D-087, the block-mean resampling requirement.

## 12. Do not

- Do not start a replay at `K0`, or at any mid-record sample, when the question is a full-record
  replay. Use sample 0 with `closed_loop.x0_for`.
- Do not write zeros into a normalised state vector to mean "at rest". Build physical, then
  normalise.
- Do not quote ramp fraction as evidence about an offset; it is variance-based and scores `0.00 %`
  on a pure DC shift.
- Do not draw a general conclusion from V1 alone. It is the record where the real effect is
  smallest and every artefact dominates.
- Do not modify `model_augmentation/` or `scripts/gantry/gantry_dynamic/`.
- Do not rerun the MATLAB three-curve comparison or `export_baseline_closedloop.m`; out of scope
  this session.

## 13. Operational

```
cd scripts/gantry/closed-loop-controller
BDR_RECORD=0 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 \
  conda run --no-capture-output -n GraduationProject python -u baseline_drift_replay.py
```

`BDR_RECORD` selects the val record, `0` = V1_standstill_Yp10 (`Y_op +0.10`),
`1` = V2_aprbs_Ylow (`-0.22`), `2` = V3_ysweep_Yp10 (`+0.10`), `3` = V4_lissajous_Ym10 (`-0.10`).
The operating point is per record and the controller is rebuilt at it; `kappa` varies about 40 %
across `Y in [-0.30, +0.30]`, so the wrong `Y_op` is a real error.

Runtime about 2 minutes at 4 kHz, about 8 minutes if `fs_new=None` (20 kHz, 240000 steps).
Figures land in `figures/baseline_*<suffix>.{png,pdf}`. Do not pipe through `tail`; it buffers
until exit and hides progress.

`gen_annoff_data.py` usage is in its docstring:
`python -u scripts/gantry/dc-accumulation/gen_annoff_data.py --records V1_standstill_Yp10.mat`.

## 14. Delegation

None. The next action is a targeted two-implementation diff in two known files. An Explore
subagent would not help and the context-holding session is better informed.
