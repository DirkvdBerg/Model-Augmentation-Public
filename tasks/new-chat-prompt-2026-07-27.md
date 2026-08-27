# Prompt for a new chat: build the current status overview

Paste everything below the line into the new session.

---

## Your task

Build an overview of where the gantry augmentation problem currently stands. Do not run
training. Read, verify against artifacts, and write.

## Read in this order

1. `scripts/gantry/dc-accumulation/README.md` - the failure in one paragraph, the acceptance
   gates (G1/G2), and the hard constraints on any candidate fix.
2. `docs/diagnostic-overview.md` - the authoritative status document. Every claim carries an
   artifact path and an evidence grade. **Read section 10 last and carefully: it is a dated
   addendum from 2026-07-26/27 that changes several conclusions above it.**
3. `docs/results-log-2026-07-26.md` sections 11, 12, 13 - the new measurements, numbers only.
4. `docs/gantry-augmentation-problem-log.md` section 12 - the run table. The last six rows are
   the recent campaign, each with its hypothesis and control pre-declared before launch.

## The rules that matter here

**Artifacts beat documents.** This repository contains confident, well-written claims that
later measurement falsified, several of them written the day before. Where a document and a
stored number disagree, the number wins. This has already caught: a "ruled out" verdict on
the only intervention ever measured against the deliverable (ARTBP); two void framings
(`narrowband-objective-problem-2026-07-26.md` section 5,
`flat-direction-problem-2026-07-26.md` sections 2-3); and, on 2026-07-27, a summary of the
MSD transfer diagnostic whose two headline numbers are not in its own artifact.

**State the horizon with every error number.** The same ANN-off model measures `7.86e-05` at
2 s and `1.66e-04` at 12 s. At least two wrong conclusions here came from comparing across
horizons.

**Report voids and refutations, do not quietly drop them.** A run whose control failed has
zero readable rows.

**Seeds.** The project floor is 3. Almost everything in the recent campaign is 1 seed and is
below it. Say so wherever it applies.

## What is settled (do not re-litigate)

* **The failure.** A learned block trained on 0.1 s windows acquires a constant output
  offset. On plants with poles at `z = 1` it integrates. The 12 s free-run error goes from
  `1.661e-04 m` at epoch 0 to `2.109e-02 m` after one epoch and never returns below epoch 0
  in 20 epochs, while the validation windowed error falls 14 percent over the same run.
* **The DC is the failure, not a correlate.** Gate answered 2026-07-26 night
  (`diagnostic-overview.md` C-19, `results-log` section 13). On three ARTBP checkpoints the
  per-row mean alone reproduces **98.7 to 99.5 percent** of the 12 s error, and removing it
  collapses the model to 0.87 to 1.26x its own ANN-off floor.
* **ARTBP's `dc` metric watches one row of eight.** The dY row carries only **0.77 to 1.76
  percent** of the DC norm; the DC lives on `aug0` and `aug1`. So C-6's "the DC was removed
  and the drift was not" was never supported: only the dY component was ever measured. The
  `+-3e-7` band (`ARTBP/README.md:136`) is a dY-only band.
* **ARTBP is NOT ruled out.** It is the benchmark: at `H_max = 1600` it cuts the first-epoch
  collapse about 10x and the drift 4 to 6x at about 2 percent windowed cost, and still fails
  G1 everywhere. Three documents called it ruled out on a theoretical argument while six
  converged runs sat unread on disk. The anti-scope in
  `docs/dc-accumulation-research-brief-2026-07-26.md` section 3 is now corrected in place.
* **Six mechanisms for WHY the DC is acquired are measured FALSE**: the loss is blind to it;
  the loss rewards it; it is paired with a compensator; it is exposure bias; it is entangled
  with the useful fit; it is a transient more steps would remove. See C-8, C-12, C-13, C-14,
  C-2. Why it is acquired at all is currently unexplained.

## What was withdrawn in the last 24 hours

* **C-20 is WITHDRAWN.** It briefly looked like the first model ever to beat its own
  initialisation on the deployment metric (`0.868x`). Step 0b showed that constant had been
  fitted on the evaluation record. Recomputed from **training** records only, the same arms
  give **3.22x, 3.97x and (third arm) epoch-0 ratios above 1** on the production selector.
  There is still no model here that beats its init. Artifact:
  `scripts/gantry/dc-accumulation/results/step0b_train_constant.json`.
* Useful by-product: train and free-run constants agree to about **7 percent** in norm, and
  that 7 percent costs **5x** in the 12 s free run. Any "estimate and subtract the constant"
  fix needs the constant far better than 7 percent.

## What is newly known about the tooling

* **The minimal testbed (C-15) is not a valid screen at 3 seeds.** It ports faithfully
  (`1.713x` reproduced exactly at seed 0), but across seeds the harm ratio is
  `1.713 / 3.403 / 1.214`, and the sibling arms spread 5.9x and 8.8x with one seed at
  `0.627`. **C-15's published `1.713x` is one seed of a quantity with 2.8x spread.** Seed
  scatter exceeds the effects it would be used to screen.
  Artifact: `scripts/gantry/dc-accumulation/results/step1_testbed.json`.
  Caveat recorded there: the `C2` coupled-absorber arm is confounded (the absorber raised the
  ANN-off error about 3000x, so residual magnitude changed along with state-dependence) and
  must be re-run scaled before it is read.
* **ARTBP training barely moves the encoder.** Across all three checkpoints the total encoder
  delta is `5.002781e-06`, identical, with most tensors bit-identical to init and the rest at
  about `1e-7` relative. Corroborates C-1b's zero-gradient finding on a different rig.

## Open, and worth the overview saying so plainly

1. **Nothing has ever beaten the epoch-0 initialisation on the 12 s deliverable.** The best
   is ARTBP `geom` at 9.5x worse. This is the actual deliverable question.
2. **Why the DC is acquired is unexplained** after six candidate mechanisms were falsified.
3. **The orthogonal-projection route, which is the thesis's stated scientific contribution,
   has three failed gates nobody followed up**: `orth_projection/step7b_result.json`
   `pass_all: false` at a maximum principal angle of **56.66 degrees against a 5 degree
   tolerance**, `step8b_result.json` at 38.19 degrees, `harness_check.json` false. If the
   deliverable is the projection, this is the first thing needing an answer, and it appears
   in no framing document.
4. **Is the MSD-augmented system observable?** `diagnostics/system_dynamics.json` records
   `msd.obs_rank = 3` of 8 against `baseline.obs_rank = 6` of 6. Unverified, cheap to check.
5. `scripts/gantry/msd_transfer_diagnostics/` needs an independent look before anything is
   built on it. See the caution below.

## A caution about one recent artifact

`scripts/gantry/msd_transfer_diagnostics/data/diag_msd_summary.json`: the `adam_1e-05` arm
has exactly one entry in every array, equal to the value all five arms share, i.e. the
pre-training probe. That arm completed **zero epochs**. A summary circulated on 2026-07-27
attributed `train nf-RMS 3.80e-05 -> 3.43e-05` and `sim-NRMS 16.27 -> 284.60` to it; neither
number is in the file (`284` appears nowhere in the folder). Every arm that did complete an
epoch moved the free-run metric the *other* way (`-> 15.58`, `-> 12.07`, `-> 7.24`).
The script also uses `dataclasses.replace(RunConfig(), ...)`, so it runs at `up_sample = 2`
(`config.py:66`) against the entry file's `1`
(`gantry_interconnect_dynamic.py:72`); it trims training to 3 of 14 records; and line 98
swallows training exceptions, so a crashed arm still writes a row that looks complete.
Its companion `diag_proximal_summary.json` has `prox_lr` `0.0 / 1e-4 / 1e-3` producing
bit-identical values in every field, i.e. the penalty did nothing at any strength.

## Config traps that have already invalidated runs

* `RunConfig` defaults `up_sample = 2`; the entry file and every checkpoint use `1`.
* Trimming `TRAIN_FILES` changes `compute_normalization`, hence the encoder, hence every
  downstream number. A 4-record trim moved epoch-0 from `1.66e-04` to `1.13e-01`. Validation
  trims are safe; training trims are not.
* `gantry_ckpt_*.pt` is the **best** checkpoint, and since best = epoch 0, that file IS the
  initialisation. Use `*_last.pth` for a trained model.
* `.pth` files pickle `gantry_dynamic` as a top-level module, so put `scripts/gantry` on
  `sys.path` before `torch.load`. They carry their own `norm`; take weights and `norm`
  together.
* The ARTBP checkpoints carry **no** `norm`; rebuild it with
  `drift-demo/demo_common.build_pipeline(dataclasses.replace(CFG, seed=0))`, which is the rig
  that trained them. `CFG.seed` is 42 but `train_artbp.py` overrides it to the arm's seed.
* Filtering non-finite values out of a metric series makes divergence look like a flat pass.
* Y in the record names is in units of **10 mm**: `V1_standstill_Yp10` sits at `+100.000 mm`
  and the training Y envelope is `[-300.018, +300.017] mm`.
* Background jobs get OOM-killed on this machine; make every run resumable and checkpoint per
  epoch. Piping a run through `grep`/`tail` block-buffers stdout, so the log stays empty
  until exit.

## Where the figures are

* Per-record data figures (T1-T14, V1-V4, E1-E4):
  `data/gantry/matlab/trajectory/augmentation/figures/`, mirrored under `joint/figures/` and
  `augmentation/baseline/figures/`.
* The `e0` to `e8` meeting deck: `scripts/gantry/gantry-zero-mean/meeting-2026-07-21/`,
  lowercase names, plus `meeting.html`. Note `e3` and `e7` are not in that folder;
  `e7_residual_spectrum.png` is in `scripts/gantry/gantry-zero-mean/figures/`.
* New diagnostics go under `simulations/gantry_subnet/diagnostics/`, not `scripts/`
  (`tasks/lessons.md` rule `diagnostic-figures-are-falsifiable`).

## Files another session is actively writing

Do not edit these; propose instead: `scripts/gantry/dc-accumulation/*`,
`docs/diagnostic-overview.md` section 10, `docs/results-log-2026-07-26.md` sections 11-13,
and the last six rows of `docs/gantry-augmentation-problem-log.md` section 12.

## Deliverable

A status overview that a new person could act on: what is established and at what grade, what
is open, what is void, and what the single next action should be with its rationale. Prefer
one recommendation over an option menu. No em-dashes anywhere in the output.
