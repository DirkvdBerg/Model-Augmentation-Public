# Handoff: can the full black box learn this plant from an EXACT linear start, and only then, how close can a data-driven init get

**From**: session of 2026-09-17 | **Branch**: Augmentation | **Effort suggested**: high

## 1. Task

Two phases, in this order, both inside `scripts/gantry/ann-blackbox/`, on data that already exists.

**Phase 1, the gate.** Give the standalone full black box (`ann_blackbox.py`, no interconnect, no
baseline) the **exact truth poles** as its linear initialisation, and find out whether it trains.
This is an oracle-initialised diagnostic arm, deliberately not a defensible thesis result. It exists
to answer one question that is logically upstream of all the identification work: can this model
class, at this horizon, on these records, learn this plant at all when the linear start is correct by
construction. Run it paired against the random-init control at a shared seed and matched update
count, score against `results/bars.json`, and iterate on the training configuration (horizon, record
selection, batch size, update budget, whether the nonlinear branches start at zero) until either the
arm clears the bar in section 10 or you can name the mechanism that stops it.

**Phase 1 runs on the `augmentation` dataset at 800 Hz.** Decided 2026-09-17 after the first
successor session found the gap described in fact 11. On `augmentation`, `plant.py`'s defaults ARE
the plant, `results/bars.json` is the matching bar, and `truth_ct(0.0)` needs no parameterisation, so
the gate costs nothing beyond the run itself. The contrast phase 1 measures is oracle against
**random**, and random carries no correct poles on any dataset, so the contrast is full size here.
The larger absorber signature of the `ma50_b140-230_a6_z03` set matters for phase 2 and for the
augmentation pipeline, not for this gate: the failure in fact 6 is a rigid-body and scheduling
failure (`rms_Y = 0.1908`, roughly 4x the X channels), not an absorber failure. **Known cost of this
choice:** if phase 1 passes, the black-box baseline that goes in the thesis must move to whatever
dataset the augmentation arm uses, and the `--track` plumbing and a rerun of `bars.py` get paid then.
Paying them now risks spending them on a gate that fails.

**Phase 2, only after phase 1 has an answer.** Ask how much of phase 1's behaviour is recoverable
from an initialisation estimated from data alone. Best achievable today is a frozen-point linear
model on one dataset, with the rigid-body block and the two real coast-down poles pinned from the
baseline model and only the absorber and the ~5 Hz pair identified. Section 5 explains why an
eight-pole identified model is not available from data on disk. Report phase 2 against phase 1, not
against the truth.

You own the iteration loop in both phases: change one thing, relaunch, score, repeat.

## 2. Out of scope

- **Generating new MATLAB trajectory data. The user's decision, taken 2026-09-17.** No new
  `OUT_DIR_NAME`, no regeneration, no band redesign. Everything below is scoped to the records
  already on disk. If you conclude that a result is unreachable without new excitation, say so in
  one sentence and stop; do not generate it.
- The **augmentation** arm: `BLA-Augmentation/`, `Augmentation-with-BLA/`,
  `orthogonal-by-construction/`, `gantry_interconnect_dynamic.py`, `gantry_dynamic/`. Different
  pipeline, different question. Do not modify, and do not change `gantry_dynamic/config.py`'s `mode`.
- Orthogonal projection, parameter recovery, physical interpretability. The black box carries no
  physical claim by construction.
- LPV scheduling of the initialisation. `PLAN-BLA.md` step 5 (per-Y pole locus) is a thesis figure,
  not a prerequisite. The black box must learn the Y dependence itself.
- `kamtin-fp-model/`, `kamtin-data/Data Telica/`.
- `scripts/gantry/full-blackbox/`. Abandoned, and the user does not trust it.

## 3. Where things stand

Branch `Augmentation`, last commit `a52dd55`. The tree is dirty across many directories. Inside
`ann-blackbox/` the tracked modified files are `DIAGNOSIS-PROMPT.md`, `ann_blackbox.py`,
`bla_init.py`; most of the FRF chain (`PLAN-BLA.md`, `bla_frf.py`, `bla_vs_frozen.py`,
`crb_excitation.py`, `epoch0_compare.py`, `frf_init.py`, `lpm_frf.py`, `vector_fit.py`,
`frf_to_ss.py`, `frf_peak_check.py`) is untracked. No run is in flight.

## 4. Established and verified

1. **The oracle arm needs almost no new code.** `truthmodel.truth_ct(Y)` returns the exact
   continuous-time `A, B, C` at frozen Y, `truthmodel.discretise` does exact ZOH by block matrix
   exponential, and `bla_init.apply_frf_bla_init(fit_sys, train_data, ss_path=...)` already loads a
   discrete `A, B, C` from an `.npz`, converts it into deepSI's normalised coordinates
   (`B_n = B diag(ustd)`, `C_n = diag(1/ystd) C`, zero bias), rescales the states to unit standard
   deviation, and writes it into the bypass. Truth order is 8 and `fit_sys.nx = 8`, so there is no
   padding and no gradient-dead spare state. **The arm is: write a truth `.npz` in the same key
   format as `results/frf_init/frf_init_ss.npz`, expose an `--ss-path` flag, run.**
2. **`plant.py` does NOT describe the current dataset.** Verified quote,
   `scripts/gantry/msd-offset/plant.py:33-34`, hash `8c40a7cb8557fb9e`:
   ```
   MA_FRAC = 0.10
   MA = MA_FRAC * mh                  # 1.01 kg
   ```
   and `ZETA_A` is 0.05 in the same file. Those values match the **`augmentation`** dataset and
   nothing else. On any `ma50` set, and on `ma50_b140-230_a6_z03` in particular
   (`ZETA_A_OVERRIDE = 0.03`), `truthmodel.truth_ct` returns the truth of a **different plant**
   (absorber 158.11 Hz at zeta 0.0528 instead of 211.60 Hz damped and sharper). **This is why
   phase 1 stays on `augmentation`.** When a later phase leaves it, the fix is parameterisation,
   never editing these constants in place: `bars.py`, `frf_init.py` and others depend on them.
3. **The absorber frequency follows `ma_frac`.** `generate_trajectory_data.m` records
   `fn = fa*sqrt(1 + ma/mh_rigid)`, giving 158.11 Hz at `ma_frac = 0.10` and 212.13 Hz at 0.50, with
   211.60 Hz damped, verified there three ways agreeing to 0.01 Hz.
4. **The `n_val = 5 or 6` mystery is SOLVED and was never truncation.** `fit()` reloads the `_best`
   checkpoint before returning, so `fit_sys.Loss_val` afterwards stops at the best epoch. Fixed
   2026-08-11: `ann_blackbox.py` now reads the `_last` checkpoint back and writes the full curve into
   the metrics JSON under `full_*` keys. It had hidden a 13 percent free-run turnaround on the nf400
   arm. **Every metrics file written before 2026-08-11 is structurally blind after the best epoch.**
   `DIAGNOSIS-PROMPT.md` Part 1 is stale on this point.
5. **The bars exist and are data-derived.** `results/bars.json`, free run with oracle initial state
   at 4 kHz. On V2: floor 4.652e-05 m, FP baseline 1.883e-04 m, frozen LTI 4.781e-04 m, mean
   predictor 1.470e-01 m. Aggregate over the four validation records: floor 4.391e-05, baseline
   8.229e-05, frozen 2.433e-04, mean 9.549e-02. `y_scheduling_gain` 2.957.
6. **The best black-box run so far is barely better than a constant predictor**: 626x above the FP
   baseline on V2 and only 20 percent better than `mean_pred`, with `rms_Y = 0.1908` roughly 4x worse
   than the X channels, and `loss_val` decreasing monotonically without flattening.
7. **A frozen-point LTI model is itself a poor predictor of the validation set**, 2.433e-04 aggregate
   against the baseline's 8.229e-05, i.e. 3x worse. Expected: it is correct at one Y while the
   validation records sweep Y. **Consequence: judge the oracle arm on where training goes, not on
   its epoch-0 score.**
8. **Pole accuracy did not buy prediction accuracy in the one paired measurement that exists.**
   `epoch0_compare.py`: on the linear-only metric with an LS-optimal initial state, the
   frequency-domain init (worst pole error 7.92e-03) scored 2.695e-02 against the N4SID BLA's
   2.064e-02, i.e. 1.3x **worse**, despite the BLA having neither resonance. The BLA is fitted on
   T10, which sweeps Y exactly as the validation record does. Input class and operating point beat
   pole accuracy for prediction. This is the single most important prior for phase 1.
9. **The training records are a genuinely time-varying plant.** No model of any order fits the
   trajectory-record FRF better than 50 percent, because those records sweep Y over +-0.29 m
   (`bla_init.py`, `apply_frf_bla_init` docstring). The BLA of that input class loses the 5 Hz pair
   entirely and returns an unstable pole at `|z| = 1.286` (`bla_vs_frozen.py`).
10. **The trainer has NO dataset selector, and this constrains everything.** `plant.py:19` binds
    `TRAJ` to the `augmentation` folder, and only `lpm_frf.py`, `frf_init.py`, `frf_peak_check.py`,
    `bla_frf.py` and `bla_vs_frozen.py` rebind it. `data.py` imports `load_record` and never does;
    `ann_blackbox.py` goes through `data.load`; `bars.py` calls `plant.load_record`, `plant.M8` and
    `plant.deriv6` at the module defaults. **Consequence: every black-box run in `results/`,
    including the fact 6 numbers, trained on `augmentation` at `ma_frac = 0.10`, and `bars.json` is
    the bar for that same plant.** They are mutually consistent, and they are consistent with
    `plant.py`'s defaults. Nothing can be run on another dataset without adding a `--track` flag
    that rebinds `plant.TRAJ` before `data.py` imports from it.
11. **Every prior black-box run is at 800 Hz, not 4 kHz.** `results/metrics_fs800_*.json`: all four
    at `fs = 800`, `batch_size = 256`, train `T10_aprbs_60`, val `V2_aprbs_Ylow`. The trainer's
    `--fs 4000` default has never been used. The 626x figure of fact 6 is
    `0.117865 / 1.883e-04 = 625.9`, i.e. an 800 Hz run scored against the 4 kHz bar.
    `frf_init.py`'s `FS_TRAIN = 800.0` matches the runs, not the bars.
12. **Horizon: `nf = 3700` beats `nf = 400` at 800 Hz**, best free-run sim-RMS 0.1179 against 0.1289
    (`metrics_fs800_nf3700_s0.json`, `metrics_fs800_nf400_s0.json`). The best arm of all four is
    `--bla dyn --bla-zero-nl` at 0.0972, from an epoch-0 of 0.3177.
13. **`bars.py` hardcodes `FS = 4000` and writes a fixed `results/bars.json`.** Rerunning it at
    another rate overwrites that file, which the never-overwrite rule forbids; it needs an
    output name keyed on the rate.
14. **`ann_blackbox.py` is a verified reconstruction of Jan's reference** (`CORRESPONDENCE.md`), with
    `--bla {off,dyn,full,frf}`, `--bla-zero-nl`, `--seed`, `--n-its` for update-matched pairing, and
    `--timeout` so a wall-clock kill does not lose the artefacts. Defaults: `--train T10_aprbs_60`,
    `--val V2_aprbs_Ylow`, `nf 400`, batch 256, width 8, encoder width 16, `nx = 8`, `na = nb = 17`.

## 5. Assumed but not verified

- **That the black box can learn this plant at all.** This is exactly what phase 1 measures. Do not
  assume either answer.
- **That `worst_dz = 0.2964` on `results/frf_init_joint_lowf_ma50_a5/summary.json` is a band problem.**
  Two candidate causes and they are separable. (a) `joint_lowf_ma50_a5` has band [0.0833, 200] Hz,
  which stops 11.6 Hz below the 211.6 Hz absorber. (b) The comparison and the pinned poles inside
  `frf_init.py` come from `plant.py` at `ma_frac = 0.10`, i.e. the wrong plant, and
  `start_poles_custom` starts at 0.6, 5.0 and 158.0 Hz. Fact 2 makes (b) at least as likely as (a).
  Settled by rerunning that comparison against a correctly parameterised truth.
- **That `joint_lowf_ma50_a5` and `augmentation_ma50_b140-230_a6_z03` are different plants.** Inferred,
  not read: the folder naming convention encodes every knob that moves the excitation, the
  `z03` tag is absent from the former, and it was generated 2026-09-01 while the `ZETA_A_OVERRIDE`
  entered on 2026-09-03. The `.mat` files store no cfg struct, so this cannot be read off directly;
  it would be settled by comparing the absorber half-power bandwidth in the two FRFs.
  **If it holds, line sets from the two datasets must not be merged: that would fit one model to two
  systems.** This is why phase 2 is scoped to one dataset.
- **That the ~5 Hz pair is recoverable from the motion records of the `b140-230` set.** At
  `ma_frac = 0.10` the equivalent route located it to -4.0 percent with an FRF magnitude that was not
  quantitatively trustworthy, and 8 of 9 channels ~100 percent in error at 5 Hz (`PLAN-BLA.md`
  step 1, carried limitation). Phase 2 depends on it.
- **That the training configuration, not the model class, is what limits fact 6.** Unmeasured. The
  full-curve fix (fact 4) means this can now be measured properly for the first time.

## 6. Tried and failed

- Time-domain N4SID BLA -> no complex pole at either resonance, 2 of 8 states spent at 341 and
  400 Hz where the input is at -89 to -115 dB -> an energy-ranked 8-state truncation cannot see a
  mode carrying ~1e-8 of record energy and moving one entry of a 3x3 FRF -> `bla_decimation_test.py`,
  `frf_diagnostic.py`.
- Blaming decimation for the missing absorber -> the u/y decimation mismatch is real (28 degrees at
  158 Hz) but four variants left the pole missing -> not the cause -> `bla_decimation_test.py`.
- Blaming excitation for the missing absorber at `ma_frac = 0.10` -> input PSD only -17 to -23 dB at
  158 Hz relative to 5 Hz, so the mode is excited -> the estimator was at fault -> `frf_diagnostic.py`.
- Stacking six standstill records across Y -> low-band median error 0.0149 to 0.2814 -> `M(Y)` carries
  `mh*Y^2`, so the rigid-body response is Y dependent; the LPM sizing rule was already satisfied by
  one record (13 > 12) -> `PLAN-BLA.md` step 1 correction 2.
- A closed-loop indirect estimator using the injected force as the exogenous signal -> no improvement
  at any frequency -> feedback bias was never the problem -> `PLAN-BLA.md` step 1.
- Fitting residues on a discrete partial-fraction basis at 800 Hz -> median relative FRF error 4.16
  -> every slow pole crowds near z = 1 (0.9962, 0.9951, 1.0000), so the basis columns go nearly
  dependent -> `frf_init.py` docstring.
- Alternating least squares for B and C -> median 6.6 percent, p90 79 percent, worse than a zero
  predictor on its own fitting record -> a poor local optimum -> `frf_init.py` docstring.
- A single LPM window for both modes -> smears the 5 Hz pair -> half-power bandwidths 16.7 Hz and
  0.94 Hz cannot share one 2.08 Hz window -> `PLAN-BLA.md` step 1 correction 3.
- Uniform line weighting in the parametric fit -> the 5 Hz mode is 11 of 2389 lines against ~200 for
  the absorber, so it gets 0.5 percent of the cost -> log-spaced thinning to 60 lines per decade fixes
  it, but the result is **not monotone in line count** (120 and 240 fail again), i.e. the monomial
  basis is ill conditioned; the published cure (van Herpen et al., IEEE TAC 2015) is not implemented
  -> treat any parametric fit as provisional.
- Validating on `V1_standstill_Yp10` a model trained on T10 -> V1's output std is 3e-6 m against
  T10's 5.7e-2 m, six orders down, so it cannot score that model -> `ann_blackbox.py` argument notes.
- Reading `fit_sys.Loss_val` after `fit()` as the training curve -> silently stops at the best epoch
  -> see fact 4.

## 7. Achieved

Implemented and validated: the frequency-domain chain `lpm_frf.py` -> `vector_fit.py` ->
`frf_init.py` -> `bla_init.apply_frf_bla_init`, on the `joint` track at `ma_frac = 0.10`. Worst pole
`dz` 7.92e-03 against 9.19e-01 for N4SID, FRF fit 0.83 percent median and 3.8 percent p90,
`max|eig(A_d)| = 1.000000`. Artefacts under `results/{frf_init,vf_check,frf_to_ss,epoch0}/`.

Implemented, not validated: the same chain on `joint_lowf_ma50_a5`, `worst_dz = 0.2964`, cause
unresolved and now with two candidate explanations (section 5).

Implemented and validated: `ann_blackbox.py` as a line-for-line reconstruction of Jan's reference,
trains without error, full training curve now persisted, paired cluster arms under `server-results/`.
Not validated: that any gantry run has converged.

Not started: the oracle-init arm. Phase 1 is new work.

## 8. The open question

**Does a correct linear start change where training goes, or does the input class dominate?**

Fact 8 is the reason this is a real question and not a formality: the one paired measurement that
exists shows a model with essentially correct poles predicting *worse* than a BLA with no resonances
at all, because the BLA was fitted on the same Y-sweeping input class as the validation record. Two
candidate outcomes and they lead to different projects.

- The oracle arm trains materially better than random. Then the initialisation matters, phase 2 is
  worth doing, and the thesis claim is about convergence speed and spread.
- The oracle arm is indistinguishable from random. Then initialisation is not the lever on this
  plant, phase 2 is not worth its cost, and the open question becomes the horizon and the input
  class instead. `PLAN-BLA.md`'s own step 6 note anticipates this: SUBNET argues initialisation
  matters less, the init papers argue it matters for speed and spread, and neither position was
  formed on a plant with poles on the unit circle.

What chooses between them: the paired run in section 9, scored on the full curve (fact 4), at
matched update count, over enough seeds to see the spread.

## 9. Next action

Build and run the oracle arm on `augmentation` at 800 Hz, in three steps that are one piece of work.
Nothing here needs a `--track` flag, a `plant.py` parameterisation, or a new `bars.py` run, because
on this dataset the truth, the bars and every prior run already agree (facts 2, 10, 11).

1. Write the oracle model. Build `A, B, C` from `truthmodel.truth_ct(0.0)` at `plant.py`'s existing
   defaults, discretise with `truthmodel.discretise` at `ts = 1/800`, and save to
   `results/truth_init/truth_ss_fs800.npz` with the same keys `results/frf_init/frf_init_ss.npz`
   uses. Sanity, all three cheap and all three catch a real error: the absorber lands at 158.11 Hz
   undamped and 157.89 Hz damped, the slow pair near 5.12 Hz, and `max|eig(A_d)| = 1.000000` to six
   digits with nothing outside the unit circle.
2. Add `--ss-path` to `ann_blackbox.py` so `--bla frf` can point at that file, defaulting to the
   current `results/frf_init/frf_init_ss.npz` so every existing invocation reproduces. A `# DEV:`
   comment in the file's existing style is the right marker here; the
   `model_augmentation/` tracking rule applies only if the change reaches that package.
3. Run the paired set: `--bla off` (control) against
   `--bla frf --ss-path results/truth_init/truth_ss_fs800.npz --bla-zero-nl` (oracle), at
   `--fs 800 --nf 3700`, shared `--seed`, matched `--n-its`, at least 5 seeds per arm. Score the
   **full curve** (the `full_*` keys, fact 4) against `results/bars.json`.

Rationale for the rate: 800 Hz with `nf 3700` is the configuration of the best prior arm and of all
four existing runs (facts 11 and 12), so the oracle arm is directly comparable to them. `bars.json`
is at 4 kHz; both quantities are a free-run position RMS in metres over the same record, so the
comparison is the same one every prior run reported, and the rate must be stated with every number.
If that residual inconsistency needs closing, rerun `bars.py` at `FS = 800` into a **separate**
`results/bars_fs800.json`, never over `bars.json` (fact 13).

## 10. Acceptance criterion

**Phase 1 gate.** The oracle arm's best free-run validation sim-RMS on V2 reaches the FP baseline
bar, `1.883e-04 m` from `results/bars.json`. That bar is data-derived: it is what an untrained
physics model achieves on the same record, and a black box that cannot approach it has not learned
the system. Beating `mean_pred = 1.470e-01 m` is the floor of usefulness, not success. The numbers
to beat on the way there, all on V2 and all at `fs = 800`: the random-init control's 0.1179 and the
best arm so far, `--bla dyn --bla-zero-nl` at 0.0972 (fact 12). Report the number for every seed and
both arms, including the arms that do worse, and report the full curve, not the best-epoch value.

**Phase 1 comparison.** Over at least 5 seeds per arm at matched update count, state whether the
oracle arm reaches any given accuracy level in fewer updates than random and whether its across-seed
spread is smaller. That is the form of claim the literature supports (Schoukens ECC 2021; Hoekstra
et al. arXiv:2602.13108), not a better best case.

**Phase 2.** The data-driven init reproduces the **measured** FRF to a median relative error at or
below 1 percent and p90 at or below 5 percent over the excited lines, with no eigenvalue outside the
unit circle, and its training behaviour is reported against phase 1's oracle arm. Comparisons against
`truthmodel.truth_ct` stay diagnostics and never become the gate, per the Control Engineering Stance.

## 11. Read these first

1. `scripts/gantry/ann-blackbox/ann_blackbox.py`. The whole training arm, with every deviation from
   Jan's reference annotated inline, including the `_best`-reload fix of fact 4.
2. `scripts/gantry/ann-blackbox/bla_init.py`, `apply_frf_bla_init` docstring. The exact normalisation
   contract the oracle `.npz` has to satisfy.
3. `scripts/gantry/ann-blackbox/truthmodel.py` and `scripts/gantry/msd-offset/plant.py`. What the
   truth currently is, and the `ma_frac`/`zeta_a` mismatch that step 1 fixes.
4. `scripts/gantry/ann-blackbox/PLAN-BLA.md`, steps 1 and 6 plus the `bla_vs_frozen.py` section. Why
   phase 2 is a frozen-point model rather than a BLA, and what step 6 is allowed to claim.
5. `scripts/gantry/ann-blackbox/bars.py` with `results/bars.json`, and `data.py`. The four bars per
   record and aggregate, and the `plant.TRAJ` binding that ties the trainer, the bars and the truth
   to one dataset (fact 10).

## 12. Do not

- Do not generate new MATLAB data (section 2, first item).
- Do not move phase 1 off the `augmentation` dataset. On any other set the truth model, the bars and
  every prior run disagree with each other (facts 2 and 10), and closing that costs a `--track`
  flag plus a `bars.py` rerun for a gate that may fail.
- Do not score an 800 Hz run against a bar you have silently recomputed at another rate, and do not
  overwrite `results/bars.json` (fact 13).
- Do not edit `plant.py`'s `MA_FRAC` or `ZETA_A` in place; parameterise around them.
- Do not treat the oracle arm as a thesis result or as an acceptance criterion. It is a diagnostic.
- Do not read `fit_sys.Loss_val` after `fit()` as the training curve (fact 4).
- Do not merge line sets across the two `ma50` datasets until the damping question in section 5 is
  settled.
- Do not fit a time-domain N4SID BLA and expect the resonances; do not stack standstill records
  across Y for the low band (section 6).
- Do not revive `scripts/gantry/full-blackbox/`.

## 13. Operational

Python: `conda run -n GraduationProject python ...`. Training runs stream live per the run convention
in `CLAUDE.md`, launched in the background with `--no-capture-output` and `python -u`, output read
from the job's `.output` file, and the path given to the user.

```
PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 conda run --no-capture-output -n GraduationProject python -u \
  scripts/gantry/ann-blackbox/ann_blackbox.py --fs 800 --nf 3700 --bla frf \
  --ss-path scripts/gantry/ann-blackbox/results/truth_init/truth_ss_fs800.npz --bla-zero-nl \
  --seed <s> --n-its <N> --timeout <sec>
```

The control arm is the same line with `--bla off` and no `--ss-path`. Note `--fs 800 --nf 3700`
explicitly: the trainer defaults to `--fs 4000 --nf 400`, which no prior run used (facts 11, 12).

`--timeout` matters: `fit()` writes its artefacts only after the loop, so a wall-clock kill loses the
run. Metrics land in `results/metrics_<tag>.json`, the system in `results/ann_blackbox_<tag>`, and
the full curve under the `full_*` keys of the metrics file. `lpm_frf.py` and `frf_init.py` each carry
a `TRACK` constant selecting the dataset folder, and `frf_init.py` re-keys its output directory on
it; both must be set deliberately (D-149). `ann_blackbox.py`, `data.py` and `bars.py` have no such
selector and always read the `augmentation` folder (fact 10). `FS_TRAIN = 800.0` in `frf_init.py`,
while `bars.json` was computed at 4 kHz, so state the rate with every number. Cluster arms live under
`server-results/`.
Per the run-discipline rule, every new training config gets a row in
`docs/gantry-augmentation-problem-log.md` section 12 before launch, stating the hypothesis it tests.

## 14. Delegation

None. Both phases are targeted work in one folder whose contents are enumerated above. One Explore
subagent is warranted only if the 5SMB0 system identification lecture material has to be located
across `literature/` and `docs/references.md`. Do not exceed one.
