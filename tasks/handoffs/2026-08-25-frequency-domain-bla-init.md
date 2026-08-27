# Handoff: a frequency-domain BLA initialisation that contains the absorber, and whether it beats N4SID at matched updates
**From**: session of 2026-08-25 | **Branch**: Augmentation | **Effort suggested**: high

## 1. Task
Run the frequency-domain BLA initialisation arm (`--bla frf`) to the same update budget as the
2026-08-11 paired arms, about 85,000 updates, and decide whether an initial linear model that
contains both of the plant's resonances trains to a better free-run sim-RMS than the existing
N4SID initialisation, which contains neither. The initialisation itself is built, validated
against the truth poles, wired into `ann_blackbox.py`, and smoke-tested; what is missing is a run
long enough to answer the question. The pre-registered hypothesis and both falsifiers are already
written in the run table (`docs/gantry-augmentation-problem-log.md`, section 12, the row titled
"FREQUENCY-DOMAIN BLA INIT, third arm of the paired comparison"); read that row before launching
and fill in its OUTCOME afterwards.

## 2. Out of scope
- **Do not modify `data.py`.** Its `u` block-mean / `y` zero-phase-FIR asymmetry was investigated
  and is NOT the cause of anything (see section 6). D-087 justifies the block mean and stands.
- **Do not generate the `joint_lowf` dataset.** The track exists in `gtd_config.m` but its
  rationale was falsified; D-149 records why. Generating it buys precision on two damping
  parameters and nothing else.
- **Do not rewrite the pole figure** (`figures/blackbox-init-figures-v1.py`). That was this
  session's original request and was superseded; a rewrite is worth doing but is a separate task
  and needs the trained checkpoint from section 9 first.
- **Do not touch the encoder.** `epoch0_compare.py` shows it dominates epoch 0, and improving it
  is the obvious follow-on, but changing two things at once destroys the comparison this run
  exists to make.

## 3. Where things stand
Branch `Augmentation`, tree dirty across `scripts/gantry/ann-blackbox/`, `docs/`, and
`Matlab-scripts/Augmentation/data/`. Nothing committed this session.
**No run in flight.** A local run was launched and killed at update 3150 of a planned 13k; it
wrote no artefacts because `fit()` saves only after the loop returns. Its log survives at
`.../tasks/bwfpx11lt.output` and its numbers are in section 4.

## 4. Established and verified
- **The initialisation contains all eight poles.** Read back from the live training process,
  `|eig(A_d)| = [1, 1, 0.999889, 0.999889, 0.99569, 0.99569, 0.937483, 0.937483]` against the
  frozen truth `[1, 1, 0.999192, 0.998763, 0.996286, 0.996286, 0.936582, 0.936582]`. Worst
  z-plane distance over all eight: **7.92e-03**. `max|eig| = 1.000000` exactly, no unstable pole.
  Evidence: `results/frf_init/summary.json`, and the `[BLA-frf]` lines in any run log.
- **The N4SID initialisation contains neither resonance.** Absorber 9.19e-01 away in the z-plane,
  5.12 Hz pair 4.02e-02. Evidence: `results/bla_decimation/summary.json`.
- **The cause is the estimator plus the identification experiment, not the data.** N4SID minimises
  a time-domain error dominated by low-frequency output energy; the absorber carries ~1e-8 of the
  record's output energy while being well excited (input PSD only -17 to -23 dB down at 158 Hz).
  Evidence: `results/frf_diagnostic/summary.json`.
- **Trajectory records admit no LTI model at any order.** Fitted to orders 8, 12, 18 and 26, the
  best median relative FRF error is 0.496; the same fit on a standstill record reaches 0.0020.
  Those records sweep Y over +-0.29 m so the plant is genuinely time-varying during them. This is
  why the BLA is estimated from a standstill multisine record instead.
- **Epoch-0 free-run sim-RMS on V2**: random 1.610e-01, N4SID 3.174e-01, frequency-domain
  1.567e-01. The first two reproduce `bla_init.py`'s docstring values (1.61e-01, 0.31768) exactly,
  so the harness is sound. Evidence: `results/epoch0/summary.json`.
- **At 90 epochs the two BLA arms are indistinguishable.** Best 0.0492 (frequency-domain, update
  2555) against 0.0476 (N4SID, update 3080); random 0.1118. Scatter between adjacent validations
  exceeds that 3% gap.
- **The killed run was still descending**, last-half log-log slope **-0.953**. For contrast the
  random arm's genuine plateau at 85k updates had slope +0.049, R2 = 0.007.
- **The Cramer-Rao bound says every physical parameter is identifiable** from the existing
  excitation: `cg_sum` 1.2%, `cy` 0.87% at 1.5% FRF error. Evidence:
  `results/crb_excitation/summary.json`.

## 5. Assumed but not verified
- **That the pole advantage converts into a training advantage at 85k updates.** This is the whole
  open question and nothing measured so far bears on it, because the run stopped at 3.7% of budget.
- **That the frozen-Y mismatch does not dominate.** The initialisation is fitted at Y = 0 on a
  standstill record while training and validation sweep Y. On the linear-only metric it is 1.3x
  WORSE than N4SID (2.695e-02 against 2.064e-02) despite correct poles. Whether that costs more
  than the correct poles buy is unknown. Settled by the run in section 9.
- **That the poles survive training.** The 2026-08-11 N4SID arm consolidated its near-unit poles
  over 85k updates rather than losing them, but that was a different initialisation. Settled by
  computing the Jacobian spectrum of the trained checkpoint, as `figures/blackbox-init-figures-v1.py:65`
  already does.
- **That one seed is enough.** Everything here is seed 0.

## 6. Tried and failed
- **Fixing the `u`/`y` decimation asymmetry** -> absorber distance unchanged at 8.87e-01 to
  9.19e-01 across four decimation conventions, and two conventions produced no stable BLA at all
  -> the mismatch is real (28 degrees of u-vs-y phase at 158 Hz) but is not what removes the mode
  -> `bla_decimation_test.py`, `results/bla_decimation/summary.json`.
- **Lowering the multisine `f_low` to identify the two real damping poles** -> the CRB then showed
  those poles were already identifiable from the existing band, and N4SID had in fact recovered
  them to 4-5 significant figures all along -> the premise was wrong; a per-line sensitivity
  comparison had been used where an aggregated one was needed -> D-149, `crb_excitation.py`.
- **Monomial (SK) parametric fitting** -> accuracy non-monotone in line count, and it got WORSE on
  the real poles when given MORE exact data -> the basis spans ~14 decades over 1-200 Hz and is
  catastrophically ill-conditioned -> `ident_check_lowf.py`; replaced by `vector_fit.py`.
- **Estimating the BLA on the training records in the frequency domain** -> worst pole error
  1.056, no absorber, FRF fit 0.70 -> those records admit no LTI model at any order (section 4)
  -> `bla_frf.py`, `results/bla_frf/summary.json`.
- **Four silent bugs inside the realisation**, each recorded in `frf_init.py`'s comments with its
  cost: a discrete-time residue basis (4.16, poles crowd near z=1 at 800 Hz); unweighted least
  squares on G (2.68, |G| spans 3e6); an assumed Jordan block at s=0 when `plant._K4` gives two
  SIMPLE poles there; indexing the residue array by `kinds` position when a conjugate pair
  occupies two columns (35x); and per-mode rank-1 truncation where the basis columns are nearly
  collinear so the residues nearly cancel (136x).

## 7. Achieved
Implemented and validated against the truth poles:
- `vector_fit.py` - MIMO common-pole vector fitting (Gustavsen and Semlyen 1999).
- `frf_init.py` - self-contained, one command, ~1 min: measures the FRF of a standstill multisine
  record, fits, realises, discretises, writes `results/frf_init/frf_init_ss.npz`.
  `PIN_REAL_POLES` is a **scope switch, not a tuning knob**: False keeps the arm a black box,
  True takes `cg1+cg2` and `cy` from the baseline and makes it grey-box. Currently False.
- `bla_init.py:apply_frf_bla_init` - same contract as `apply_bla_init`, including the
  normalisation and the unit-std state rescaling from Schoukens ECC 2021.
- `ann_blackbox.py` - `--bla frf` added alongside `off|dyn|full`.
- `runners/run_paired_blafrf.sh` - differs from `run_paired_bla.sh` only in that flag.

Implemented but NOT validated: nothing. Every artefact above has a number in section 4.

Supporting analysis, all with results on disk: `lpm_frf.py` (LPM FRF, 0.5% fit on standstill),
`frf_diagnostic.py`, `crb_excitation.py`, `epoch0_compare.py`, `bla_vs_frozen.py`,
`ident_check_lowf.py`, `vf_check.py`. Literature in `BLA-LITERATURE.md`, plan and results in
`PLAN-BLA.md`, decisions D-148 and D-149.

## 8. The open question
**Does an initial linear model containing both resonances train to a better free-run sim-RMS than
one containing neither, at matched updates?** Candidates: (a) yes, and the gap widens after ~5k
updates where the N4SID arm made almost all of its progress from 0.048 to 0.00951; (b) no,
because the frozen-Y mismatch costs more than the poles buy, which the linear-only metric already
hints at (1.3x worse); (c) neither matters because the random encoder dominates, which epoch 0
supports since the linear part alone reaches 2.695e-02 m with a good x0 but 1.567e-01 m through
the encoder. The run in section 9 chooses between them.

One sentence on a better task, left to the user: if (c) is true, the encoder is worth more than
the bypass, and Hoekstra et al. arXiv:2602.13108 gives a data-based encoder initialisation that
`bla_init.py:encoder_map_ridge` already approximates.

## 9. Next action
Submit the cluster job. Local running was attempted and is impractical: 85k updates at the
measured ~1.9 updates/s is about 12.5 hours competing with the user's machine.

```
sbatch scripts/gantry/ann-blackbox/runners/run_paired_blafrf.sh
```

The runner rebuilds `frf_init_ss.npz` first so the job is self-contained. Output streams to
`/home/dirk_van_den_berg/logs/augmentation/ann-blackbox/paired_blafrf<jobid>.out`; artefacts land
in `scripts/gantry/ann-blackbox/results/paired_blafrf/`.

## 10. Acceptance criterion
Compare on **update count** via `full_batch_id`, never wall clock, since the reference arms ran
12 h on different hardware. The frequency-domain arm beats the N4SID arm if its best free-run
sim-RMS is below **0.00951** and its last-quintile median below **0.03287**, both being the
N4SID arm's measured values at ~85k updates (run 76176). Report the honest range rather than the
best alone: that arm's best-to-median spread was 0.00951 to 0.03287, a factor 3.5, so a single
best value is not a result on its own.

Both numbers are data-derived, from a prior run on the same validation record. The truth-pole
distances in section 4 are diagnostics only and must not become the acceptance criterion.

## 11. Read these first
1. `docs/gantry-augmentation-problem-log.md` section 12, the "FREQUENCY-DOMAIN BLA INIT" row -
   the pre-registered hypothesis and both falsifiers for the run you are about to launch.
2. `scripts/gantry/ann-blackbox/PLAN-BLA.md` - the plan, what each step delivered, and the
   measured results, including the errors found and corrected.
3. `scripts/gantry/ann-blackbox/frf_init.py` - the pipeline, with every structural decision and
   its measured cost in the comments.
4. `docs/decisions.md` D-148 - why the estimator moved to the frequency domain, and the naming
   decision about what is and is not a BLA.
5. `scripts/gantry/ann-blackbox/results/epoch0/summary.json` - the four-arm starting-point
   comparison in one file.

## 12. Do not
- Do not re-run the decimation variants, the `f_low` sensitivity study, or the monomial fit
  (section 6).
- Do not use the truth model to set any band, weight, pole location or threshold. It is a
  diagnostic only. `PIN_REAL_POLES = True` uses the BASELINE, which is a different and legitimate
  thing, but it makes the arm grey-box.
- Do not compare arms on wall clock.
- Do not call the frequency-domain model "the BLA of the training data": it is estimated from a
  standstill multisine experiment. D-148's addendum has the wording.

## 13. Operational
Env `GraduationProject`; on the cluster the runner activates
`/dataB1/dirk_van_den_berg/conda-envs/GraduationProject`. Expected runtime 12 h wall clock
(`--timeout 43200`), 14 h SBATCH limit. `--epochs 4000` is ignored whenever `--timeout` is set
(`fit_system.py:374` swaps the loop for `itertools.count()`), so the JSON's `"epochs": 4000` is
intent, not what ran; use `full_batch_id`.
Consumed artefact: `results/frf_init/frf_init_ss.npz`, rebuilt by the runner. To rebuild by hand:
`conda run -n GraduationProject python scripts/gantry/ann-blackbox/frf_init.py`.
A killed run writes nothing, so never terminate it early.

## 14. Delegation
None. The next action is one `sbatch` and one comparison against two stored JSON files. No
Explore subagent is warranted.
