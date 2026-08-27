# Handoff: implement the residual BLA, initialise the augmented block from it, and train overnight
**From**: session of 2026-08-24 | **Branch**: Augmentation | **Effort suggested**: high

## 1. Task

Implement the residual Best Linear Approximation specified in `scripts/gantry/augmented-states/BLA/PLAN.md`
phases 2 and 3, use its identified poles to initialise the augmented block, and launch a training
arm so results exist by 09:00 tomorrow. Work continuously: implement, check against the gates in
section 10, then launch and monitor. The estimator machinery already exists in `BLA/bla_frf.py` and
is correct, but it is **aimed at the wrong signal**: it estimates the FRF of the plant output
`G_{f -> y}` when PLAN phase 2a specifies the FRF of the **residual** `G_{f -> (y - y_baseline)}`.
Repointing it is the first job. Then fit a parametric model to the residual FRF, extract
`(rho, theta)` pairs, and hand them to `run_augmented.py`, which already accepts exactly those two
arrays. The deliverable is a trained arm plus a written record of which poles the BLA produced and
how they differ from the ARX poles arm iii used.

## 2. Out of scope

- **Do not regenerate data.** PLAN phase 1 Route A (new multisines, `M` realisations, `P` periods)
  is not needed: the existing records already carry an exact-bin random-phase multisine, measured
  in section 4. `Matlab-scripts/Augmentation/data/generate_trajectory_data.m` is dirty from another
  session; do not touch it.
- **Do not compute the nonlinear-distortion split** (PLAN phase 2b). It needs `M` realisations,
  the simulation is noiseless, and the user has explicitly deprioritised it. Say in the write-up
  that it was not measured, per PLAN Route B.
- **Do not rename `probe_residual_bla.py`** or sweep the misnomer documented in
  `DISCUSSION-POINTS.md` M5. It touches another folder's imports and is a separate change.
- **Do not run arm iii, J1, GRU or Kessels experiments.** Different threads.
- `kamtin-fp-model/` read-only. Do not modify `gantry_dynamic/config.py` (frozen, another
  session's uncommitted work).

## 3. Where things stand

Branch `Augmentation`, last commit `a0e3f76`, tree dirty across `scripts/gantry/`, `docs/`,
`tasks/`, `Matlab-scripts/`. No run in flight.

Written this session: `scripts/gantry/augmented-states/BLA/bla_frf.py` with
`BLA/runs/bla_frf.json` and `bla_frf_V1.json`; `DISCUSSION-POINTS.md` section M (lines 2067+).

## 4. Established and verified

**The excitation supports a BLA from a single record.** `gtd_make_multisine.m:10` records
`period = full record`, and `select_multisine` places lines on exact DFT bins. Measured on all six
standstill records: input power on the excited lines is `1.000000000000` of total on every channel,
worst off-bin leakage `4.1e-09` to `6.8e-09`. `601` lines, `130.0000` to `180.0000 Hz`,
`df = 0.0833 Hz`, `fs = 20000`, `N = 240000`.

**The instrument is `f_sim`, not the reference.** PLAN phase 2a says "use the reference". That is
wrong on standstill records: `AUDIT.md` section 2 measured `std(r_sim) = 0` on T1-T5 and V1, so
`G_ref->u` is singular. `f_sim` is the injected stage force, generated from a seed with no
dependence on `y`, therefore exogenous. Topology verified on T3: `u_total = u_fb + f_sim` to
`3.5e-08` relative, so `G_{f->y} = P*S` and `G_{f->u} = S`.

**LPM is required, not optional.** Three inputs are excited simultaneously with independent phases,
so one record gives one input direction per line: 3 equations for 9 unknowns. Local polynomial
fitting over neighbouring lines makes the MIMO FRF identifiable and removes the transient in the
same step, which is PLAN phase 2c. Implemented in `bla_frf.py:lpm_frf`, order 2, median design
matrix condition `1.2e+06`.

**The plant's own modes**, for checking the estimator against something known
(`ann-blackbox/BLA-LITERATURE.md` section 0): two rigid-body integrators at `z = 1`; real poles at
`-0.6469` and `-0.9901`; a pair at `5.122 Hz, zeta 0.0921`; **a pair at `157.894 Hz, zeta 0.0528`**.
The absorber's own natural frequency is `cfg.fa = 150 Hz` (`gtd_config.m:50`).

**The plant output FRF, measured** (this is NOT the deliverable, see section 6): anti-resonance at
`149.0833 Hz` on T1-T5 and held-out V1, depth `2.32x`, spread `0.0000 Hz` across
`Y in [-0.30, +0.30] m` bounded by the one-bin resolution. That anti-resonance is the plant's zero
at the absorber frequency, not a pole.

**`Y`-dependence is settled.** PLAN phase 0 records `r2 = 0.0245` from
`cl_residual_spectrum.json`, and the FRF measurement above independently agrees. **Use a single
bank.** Do not re-open this.

**The epsilon defect.** `run_augmented.py` reaches `train_model` in `gantry_dynamic/model.py`,
which passes only `lr` through `optimizer_kwargs`, so every arm i/ii/iii trained at the torch
default `eps = 1e-8`. At the gradient magnitudes involved Adam is epsilon-dominated and the update
collapses. `cl_train.py` exposes `CL_ADAM_EPS` and the closed-loop runners set `1e-16`. **Any arm
launched without `eps = 1e-16` is confounded and wastes the night.**

## 5. Assumed but not verified

- That the residual FRF has a clear resonance near the plant's `157.894 Hz` pair. Expected, since
  the residual is what the absorber adds, but unmeasured. If it does not appear, stop and report
  rather than installing poles from a feature you cannot explain.
- That `run_augmented.py`'s `(rho, theta)` path accepts BLA-derived pairs unchanged. It takes both
  as arrays and `build_cfg(nx_ann, seed, epochs, stride, rho, theta)` is the entry point, but no BLA
  pair has ever been passed through it.
- That 4 epochs completes before 09:00. See section 8.
- That LPM order 2 is adequate for the residual FRF. It was adequate for the plant FRF.

## 6. Tried and failed

- **Direct ratio `Y/F` presented as the BLA** -> anti-resonance `149.08 Hz`, peak `171.9 Hz` ->
  `f_sim` enters at the plant input, so `Y/F = P*S` and its poles are **closed-loop** poles; the
  plant pair is at `157.894 Hz`. Zeros are feedback-invariant, poles are not, which is exactly the
  observed pattern -> `BLA/runs/bla_frf.json`.
- **Indirect estimate `P = G_fy * inv(G_fu)` with two independent LPM fits** -> moved the peak to
  `162.0 Hz` (better) but the notch to `147.0833 Hz` (worse; a feedback-invariant zero must not
  move) -> estimator error, most likely compounding of two separately-fitted `3x3` matrices, or the
  window clamping at band edges in `lpm_frf` distorting the local abscissa. **Unresolved. Do not
  use the indirect numbers until the zero stops moving.**
- **`augmentation/baseline/` used as a matched baseline** -> its records carry a `1.0-7.0 Hz`
  excitation against this campaign's `130-180 Hz`, and the two `f_sim` are exactly orthogonal
  because they occupy disjoint bins -> the difference of two unrelated experiments, which evaluates
  to `3.9/4.2/7.3 e-06 m` and looks plausible -> `DISCUSSION-POINTS.md` M2. **Form the residual by
  simulating the baseline in Python**, as PLAN phase 2a and `run_augmented.identify()` already do.
- **Estimating the plant FRF instead of the residual FRF** -> produced a correct but wrong-object
  answer, and cost this session its evening. PLAN phase 2a is unambiguous about the object.

## 7. Achieved

`BLA/bla_frf.py`: exact-bin periodicity gate recomputed per run, MIMO LPM with transient term,
anti-resonance extraction, per-record and cross-record reporting. Implemented and validated on six
records including a held-out one. The machinery is right; only the signal it is pointed at is wrong.

## 8. The open question

**Does 4 epochs fit before 09:00?** 4 epochs at stride 10 is roughly 1040 updates, which is the
arm i/ii/iii budget and took 11 to 12 hours plus scoring, hitting a 14 h wall (`DISCUSSION-POINTS.md`
D3, E2). Starting after the BLA work, that lands after 10:00.

Candidate answers:
- **2 epochs, 520 updates.** This is PLAN phase 5's stated comparator budget and the budget at which
  historical arm 2 reached `3.795974e-07` and wave-1 F5 reached `3.790189e-07`. Finishes comfortably.
- **4 epochs as instructed**, accepting arrival after 10:00.

**Resolve it by launching with per-epoch checkpointing and scoring**, so whatever has completed by
09:00 is reportable and the run continues afterwards. Report the epoch count reached, not a
promise. Drop one of the two ablation surfaces per E1 (they agree to three decimals in all seven
completed runs) to buy back a full validation pass of roughly 600-700 s.

## 9. Next action

Repoint `bla_frf.py` at the residual. Concretely: build `y_baseline` with the Python closed-loop
baseline rollout using the D-161 `x0` (measured positions plus central-difference velocities),
exactly as `run_augmented.identify()` does via `cl_residual_spectrum`; form `r = y - y_baseline` on
the full record with no samples dropped, because dropping samples destroys the exact-bin property
that makes the single-record DFT valid; then LPM `G_{f->r}` and `G_{f->u}` and form the phase-2a
ratio. Then PLAN phase 3 for order selection and the modal realisation, then `run_augmented.py`.

## 10. Acceptance criterion

Gates on the estimator, all data-internal:

1. Input power on the excited lines `> 1 - 1e-6`, off-bin leakage `< 1e-6`. Already passing.
2. The residual is formed at the native rate with **no leading samples discarded**; if any are
   dropped the exact-bin property is void and the whole method fails silently.
3. The zero of the plant FRF must not move between the direct and indirect estimates by more than
   one bin (`0.0833 Hz`). This is what failed tonight and it is the check that catches it.
4. The baseline rollout reproduces the recorded untrained closed-loop pooled RMS
   `2.1866011e-06 m`, which is the D-072 reference and is not oracle-derived.

Correctness check on the estimator, reported but **not** used as a selection threshold, per the
Control Engineering Stance: the residual FRF should show a resonance near the plant's known
`157.894 Hz, zeta 0.0528` pair. Use it to confirm the estimator is sane; never tune to it and never
select an order by distance to it, which is the oracle-aided mistake `AUDIT.md` section 1 records
in `probe_residual_bla.py`.

Acceptance for the training arm: **pooled closed-loop free-run RMS on V1-V4, in metres, against the
untrained `2.1866011e-06`**, with the augmented-state ablation fraction `F` reported alongside.
No absolute target: the comparators on record are historical arm 2 `3.795974e-07`, wave-1 F5
`3.790189e-07`, arm iii seed 43 `5.782492e-07`.

## 11. Read these first

1. `scripts/gantry/augmented-states/BLA/PLAN.md` phases 2 and 3. This is the specification. Phase 2a
   is the estimator, 2c the transient handling, 3a the order selection, 3d the realisation.
2. `scripts/gantry/augmented-states/BLA/AUDIT.md` sections 1, 2 and 5. Section 2 corrects PLAN's
   reference-instrument error; section 5 lists which PLAN choices are theory and which are heuristic.
3. `scripts/gantry/augmented-states/BLA/bla_frf.py`. The machinery to repoint.
4. `scripts/gantry/augmented-states/DISCUSSION-POINTS.md` section M. Tonight's measurements.
5. `scripts/gantry/ann-blackbox/BLA-LITERATURE.md` sections 0 and 2. The plant's modes, and why
   poles at `z = 1` are admissible on a frequency grid but break a black-box BLA arm.

## 12. Do not

- Do not use `data/gantry/matlab/trajectory/augmentation/baseline/` for anything.
- Do not use `r_sim` as the instrument on standstill records; it is identically zero.
- Do not discard leading samples before the DFT.
- Do not select model order by distance to `157.894 Hz`, or to any known plant value.
- Do not launch without `eps = 1e-16`.
- Do not re-open `Y`-dependence; PLAN phase 0 and tonight's FRF both say single bank.
- Do not trust `probe_residual_bla.py` or its `residual_bla.json`; it is an ARX probe, not a BLA.

## 13. Operational

`conda run -n GraduationProject python ...`. Long runs go in the background with live output:
`PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 conda run --no-capture-output -n GraduationProject python -u <script>`,
then read the `.output` file.

BLA estimator, existing entry point:

```
cd scripts/gantry/augmented-states/BLA
conda run -n GraduationProject python -u bla_frf.py                      # T1-T5
conda run -n GraduationProject python -u bla_frf.py --records V1_standstill_Yp10
```

Training arm: `scripts/gantry/augmented-states/run_augmented.py`. `build_cfg(nx_ann, seed, epochs,
stride, rho, theta)` takes the pole arrays directly, so BLA poles substitute for
`identify()`'s ARX output with no other change. Per the run-discipline rule, add a row to
`docs/gantry-augmentation-problem-log.md` section 12 stating the hypothesis **before** launching.

Data: `data/gantry/matlab/trajectory/augmentation/T{1..5}_standstill_*.mat` and
`V1_standstill_Yp10.mat`. Fields: `f_sim`, `y`, `u_total`, `u_fb`, `r_sim`, `x_logical`, `fs`,
`Y_trajectory`, `seed`. The augmented records also carry `delta_a`, `vdelta_a`, `x_aug`, the true
absorber states; those are oracle quantities, usable for a post-hoc sanity plot and never in the
estimator.

Weekly usage was near its limit at handoff time and resets at 02:00. Background jobs cost nothing
while running; conversation turns are the expense. Prefer one launch and silence.

## 14. Delegation

None. This is targeted implementation against a written specification in files whose locations are
all given above. Do not spawn Explore subagents, and do not invoke `deep-research`: the literature
is already catalogued in `BLA-LITERATURE.md` and cited by key in `PLAN.md`.
