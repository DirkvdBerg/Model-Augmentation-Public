# Handoff: make the standalone full-ANN black box learn the gantry
**From**: session of 2026-07-30 | **Branch**: Augmentation | **Effort suggested**: high

## 1. Task

Get the standalone full ANN (`scripts/gantry/full-blackbox/blackbox_standalone.py`, deepSI
`SS_encoder_general_hf`, no interconnect and no baseline) to a good validation sim-RMS on the
gantry data, and if it cannot be done, determine what specifically prevents it. "Good" means at
minimum beating the model-free floor of `4.741e-02 m` (hold the last measured sample for the whole
12 s record), with the FP baseline's `1.6e-04 m` as the aspiration. Several candidate causes have
been eliminated by measurement (section 6); none of the eliminations amounts to a proof that the
task is impossible, and the previous session's write-ups lean further toward "impossible" than the
evidence supports. Treat the question as open.

## 2. Out of scope

- **The augmented pipeline** (`scripts/gantry/gantry_interconnect_dynamic.py`, `gantry_dynamic/`).
  Separate problem with its own programme; do not modify.
- **`scripts/gantry/drift-isolation/`** T1 to T6. Complete or paused, and its conclusions are
  evidence for you, not work for you.
- **Stage 2 of the campaign** (the same model routed through `Interconnect` with the physical block
  removed). Not started, deliberately, until stage 1 works.
- **`kamtin-fp-model/`**, read only.
- The **orthogonal-projection** thread and **joint estimation**. Unrelated to this failure.

## 3. Where things stand

Branch `Augmentation`, tree dirty across `docs/`, `scripts/gantry/full-blackbox/`, and others.
Nothing from today is committed.

One run may still be in flight on the cluster: a two-arm objective-reweighting comparison
(`current` vs `per_window`, 40k updates, `runners/run_objective_diag.sh`). It is superseded by the
findings below and is not worth waiting for. Kill it if it is still queued.

New files this session, all under `scripts/gantry/full-blackbox/`: `ref_subnet_v2_example.py`,
`msd_stability_contrast.py`, `objective_rescale_diag.py`, `objective_train_diag.py`,
`blackbox_ct_arm.py`, `runners/run_objective_diag.sh`. `blackbox_standalone.py` was edited (G0 gate,
model-free reference prints, nf-probe now uses all 14 train records).

## 4. Established and verified

Measured or read this session, with evidence.

**The implementation is correct.** The same code path (`SS_encoder_general_hf` on deepSI 0.3.29
plus the local `contiguous_*` net subclasses) fits Beintema's own example plant from
`deepSI-master/examples/docs/basic-example.py` to test NRMS **0.024** (SISO) and a 3x3 coupled
version to **0.090** (`ref_subnet_v2_example.py`, results in `results/ref_subnet/`). The training
path and the simulation path produce **bit-identical** output on real gantry data, max difference
`0.000e+00` over 200 steps and 3 channels. A full audit of `blackbox_standalone.py` against Jan's
`msd_ndof_deepSI_encoder.py` and deepSI v2 found no defect that changes the model class or the
numerics (`docs/blackbox-standalone-audit-2026-07-30.md` Part 1).

**Model-free floors on the val set** (V1-V4, deepSI's own sim-RMS aggregation, computed in
`blackbox_standalone.py` and printed at every run):

| predictor | val sim-RMS |
|---|---|
| predict the global `y0` (what an untrained net does) | `9.549e-02 m` |
| **hold `y[k0]` for the whole 12 s** | **`4.741e-02 m`** |
| oracle per-record DC | `4.699e-02 m` |
| FP baseline, untrained (reported in the campaign brief, not recomputed here) | `1.6e-04 m` |

**Both completed production runs are worse than doing nothing.** 73940 (nf=400, width 16, 130k
updates) best `7.236e-02`, last-quarter median `9.76e-02`. 74045 (nf=800, width 64, killed at 76%)
best `6.199e-02`. Neither beats `4.741e-02`.

**Learned state-map spectrum, `|eig(df/dx)|` at six operating points on V1 and V2.** The plant's
own poles all sit within about `1e-3` of 1 (X and Y are `K=0`, so the position poles are at exactly
1; the velocity poles are damped, T5 measures the Y time constant at about 1 s matching `mh/cy`).

| model | max | count > 0.99 of 48 |
|---|---|---|
| DT SUBNET at initialisation | 0.655 | 0 |
| DT SUBNET, 73940 after 130k updates | 1.0003 | 12 |
| CT SUBNET at init, `tau=2.5e-2` | 1.0056 | 48 |
| CT SUBNET at init, `tau=1.0` | 1.0001 | 48 |

`1.0003^48000 = e^14.4`, which accounts for 73940's excursions to `3.87e14`.

**Error against free-run horizon**, 73940's saved `_best` on V1-V4: `7.80e-03 m` at the FIRST
predicted sample (with `y[16]` inside its own encoder window), flat near `4e-3` out to 0.5 s, then
rising to `7.24e-02` by 3 s. It never beats hold-last-sample at any horizon.

**Per-mode NRMS in the logical frame is about 1 on every mode of every record** for 73940, including
on T12 and T14 which are inside its own training set (X/Theta/Y = 1.63/1.23/1.03 and 1.29/1.05/1.30).
The model is at "predict the mean" in free run.

**The training objective gives five of fourteen records zero weight.** Under the global `ystd`
normaliser, T1 to T5 (all standstill) contribute **0.00%** of the loss each, while T13, T11, T14 and
T10 carry 64.6% (`objective_rescale_diag.py`). The per-record `y std` spans `4.4e-06 m` (T1) to
`0.19 m` (T6).

**From `scripts/gantry/drift-isolation/CONCLUSIONS.md`, on the augmented model, for context:** T5,
the baseline plus encoder with the ANN at zero does NOT drift, it settles to a constant offset. T4,
adding stiffness to X and Y in both the truth and the model buys a constant **6.07x** and does not
change the character of the failure, so marginal poles are a contributor and not the cause. T1, val
sim-RMS scales as `nf^-1.17` and parity with not augmenting extrapolates to `nf` around 27,000.
T1 also found a 2 to 4.5 percent improvement in the windowed objective coinciding with a 12 to 138x
free-run degradation.

## 5. Assumed but not verified

- **That T1's `nf^-1.17` horizon law transfers to the standalone black box.** It was measured on the
  augmented model, where the baseline supplies the dynamics. Nobody has run a horizon sweep on the
  standalone model. This is the single largest untested assumption.
- **That the encoder produces a usable initial state for the black box.** Never measured. T5 measured
  the analogous quantity for the baseline and found the encoder x0 BEATS the true x0 by 3.5x, so
  intuition here is unreliable.
- **That `nx=8` is a reasonable order for a black box.** It is the true order of the augmented plant.
  The campaign's own README notes that fixing it is wrong for a fair black-box baseline.
- **`tau = 2.5e-2`** for the CT model was chosen because it is the smallest value at which all 48
  measured init eigenvalues clear 0.99. Only the init spectrum was measured; no sweep against
  trained performance.
- **That the previous session's normalisation of `u` and `y` is right for a black box.** It is
  inherited from the augmented run for comparability, not chosen for this model.

## 6. Tried and failed

Each with the mechanism, so it is not repeated.

- **Longer horizon plus wider net** (74045, nf 400 to 800, width 16 to 64, 21.5k updates) -> never
  reached the trivial baseline on its own objective (sqrt-loss floor 0.707 against a hold-last-sample
  0.588) and rose monotonically from iteration 11700 -> BPTT through 800 tanh steps degrades the
  optimisation faster than the doubled horizon helps, and doubling cannot close a 120x
  horizon-to-metric gap anyway -> `results/74045/stage1_74045.out`.
- **Learning rate screens**, {1e-4, 1e-3, 1e-2} at two configurations, 6 arms -> in **all six**, val
  sim-RMS on V1 rises within one epoch while the training loss falls -> the objective and the metric
  move in opposite directions from the start, so no step size fixes it -> `results/lr_probe*/`.
- **Re-weighting the loss** so quiet records are not invisible, 3 arms at 900 updates, weights
  renormalised to mean 1 so the effective step size is matched -> `current` 0.0743, `per_window`
  **0.0644**, `per_record` 0.1338 (worse than its own epoch-0); none near the 0.047 floor. A 5000-update
  continuation had `per_window` bouncing above its own init with a flat training loss -> the
  reweighting moves which records get gradient (T1-T5 from 0.00% to 7.14% each) but does not change
  the outcome -> `objective_train_diag.py`, `results/objective_train_diag/u900_*.json`.
- **CT SUBNET** (`SS_encoder_deriv_general`, `integrator_euler`, the state map is an integrator so
  eigenvalues start at 1), 300 updates, otherwise matched to the DT control -> **fits the training
  objective 2.7x better** (sqrt-loss 0.29 against DT's 0.79) and **free-runs far worse**: at
  `tau=2.5e-2` every validation is NaN (`max|eig| = 1.0046`, and `1.0046^48000 = e^220` overflows
  float32; the `simple_res_net` linear bypass is unbounded in `x` so tanh saturation does not cap
  it), at `tau=1.0` it is finite but sim-RMS is **2.23 to 3.92 m** with V1 per-mode NRMS around `1e5`
  to `1e6` -> a model that remembers accumulates its own error, a model that forgets reverts to
  something bland and scores better; **eigenvalues at 1 are not sufficient and made things worse** ->
  `blackbox_ct_arm.py`, `results/ct_arm/`.
- **Identity-initialising the DT state map** was proposed and withdrawn before running. Neither Jan's
  script nor deepSI v2's discrete `SUBNET` does it (v2's `MLP_res_net` linear branch is randomly
  initialised), so it is a deviation from both references.
- **The Theta / logical-frame thread** was pursued for three rounds and is a dead end for this
  problem: Theta is about `1e-9` of the sim-RMS metric, and free-run NRMS is about 1 on every mode,
  so Theta weighting is not what blocks the model.

## 7. Achieved

- **Implementation cleared**, with the reference numbers in section 4. Validated, not just claimed.
- **`blackbox_standalone.py` now prints the model-free floors and grades a G0 gate** against
  hold-last-sample, so a run can no longer look like a success by beating only its own random
  initialisation. Implemented and validated; the floors reproduce the independently computed values.
- **nf-probe now measures all 14 train records** rather than T1 alone. Implemented and validated
  (cost measured at +0.3 s per probe against about 45 min between probes).
- Diagnostic scripts built and validated: `ref_subnet_v2_example.py`, `objective_rescale_diag.py`,
  `objective_train_diag.py`, `blackbox_ct_arm.py`, `msd_stability_contrast.py`.
- **Not achieved: any configuration that beats `4.741e-02 m`.**

## 8. The open question

**Where does the free-run error actually come from: the encoder's initial state, or the state map's
accumulated error?** Nothing measured so far separates these, and they imply opposite fixes.

- If the **encoder** is the problem, the model's `x0` is wrong and the rollout is doomed from step
  one. Consistent with the `7.80e-03 m` error on the FIRST predicted sample, when `y[16]` is inside
  the encoder's own input window. Fix: encoder capacity, `na`/`nb`, or a different initialisation.
- If the **state map** is the problem, `x0` is fine and error accumulates. Consistent with the flat
  error to 0.5 s then divergence. Fix: horizon, multiple shooting, or a structural constraint.
- The CT result argues it is not simply a matter of the spectrum, since fixing the spectrum made the
  free run 30x worse.

**What would decide it:** free-run V1 from the encoder's `x0`, and from an `x0` optimised to
minimise the first 400 samples (or set from the true state where available). If the optimised `x0`
gives a good free run, it is the encoder. If both are bad, it is `f`. T5 did exactly this for the
baseline and found the encoder x0 beat the true x0, so the answer is not predictable.

A second thing worth one sentence, and it is the user's call, not the successor's: **the metric
itself may be the wrong target.** The 12 s open-loop free run is `1e-9` sensitive to the 130-180 Hz
dynamics the thesis is about, and T6 measured the closed loop hiding a 596x open-loop degradation as
1.6x. Raise it, do not act on it.

## 9. Next action

**Split the free-run error into encoder error and state-map error, on V1, using 73940's saved
model.** No training. Roll out from (a) the encoder's `x0`, (b) an `x0` fitted by gradient descent
to minimise the first 400 samples of the rollout, and (c) if obtainable, the true logical state from
`gantry_dynamic.data.load_mat_aug`. Report sim-RMS against horizon for each.

Rationale: it is the cheapest measurement that discriminates between the two families of fix in
section 8, every other candidate has been eliminated by a measurement that cost hours, and no
existing artefact answers it. Everything needed already exists: the model is saved, the rollout code
is in `blackbox_ct_arm.py:spectrum` and `objective_train_diag.py:_Probe`.

## 10. Acceptance criterion

For the overall task: **val sim-RMS below `4.741e-02 m`** on V1-V4 with deepSI's aggregation, the
hold-last-sample floor. Data-derived, computed from the validation records alone with no model
involved, printed by `blackbox_standalone.py` at every run. `9.549e-02` (predict the global mean) is
not an acceptable bar and is roughly where an untrained network already sits.

For the section 9 measurement: it is done when the three rollouts are plotted against horizon and
one of the two families in section 8 is excluded.

## 11. Read these first

1. `docs/blackbox-standalone-audit-2026-07-30.md` Parts 1 and 3. Full measurement record. **Part 3's
   causal claim about poles is overstated and is contradicted by the CT result in section 6 above;
   read the numbers, not the conclusions.**
2. `scripts/gantry/drift-isolation/CONCLUSIONS.md`. T4, T5 and T1. The best-evidenced work on this
   failure anywhere in the repo, on the augmented side.
3. `scripts/gantry/full-blackbox/README.md`. Campaign design, the stage structure, and the
   reading A versus reading B distinction that matters for what "working" means.
4. `docs/decisions.md` D-134. **Written this session and overstated**; it asserts a pole mechanism
   without citing T4, which tested it and found a 6x contributor. Correct or delete it.
5. `docs/gantry-augmentation-problem-log.md` section 12, the top three rows.

## 12. Do not

- Do not re-run the learning-rate screens, the objective reweighting, or nf=800 at width 64. Section 6.
- Do not identity-initialise `default_state_net`; it deviates from both references and the CT result
  suggests eigenvalues at 1 are not the missing piece.
- Do not pursue Theta or logical-frame reweighting for this problem.
- Do not modify `gantry_dynamic/`, `model_augmentation/`, or `kamtin-fp-model/`.
- Do not treat "the black box cannot work" as established. It is not.

## 13. Operational

```
conda run -n GraduationProject python scripts/gantry/full-blackbox/blackbox_ct_arm.py \
    --updates 300 --val-every 100 --tau 1.0
```
about 5.5 min, 1.07 s/update. Use as the template for the section 9 script.

73940's saved model: `scripts/gantry/full-blackbox/results/73940/blackbox_standalone_73940`. It is a
torch pickle referencing `__main__.contiguous_encoder_net`, `__main__.contiguous_state_net` and
`__main__._NfProbe`, so a loader script must define those three names at module level and be run as
a script. Working example in `blackbox_ct_arm.py`.

Long runs: launch with `run_in_background`, and **do not pipe through `grep` or `tail`**, which
block-buffer and leave the log empty until exit. Cost reference at nf=400, width 16, batch 256:
about 0.7 s/update training, about 25 s per validation on two records.

`scripts/gantry/full-blackbox/runners/run_objective_diag.sh` is a working SLURM array template if a
cluster run is needed.

## 14. Delegation

**None.** The next action is a single targeted measurement on one saved model with the code already
written. An Explore subagent is not warranted. If a later step needs a wide search of the repo for
prior work on a specific mechanism, one Explore subagent, not more.
