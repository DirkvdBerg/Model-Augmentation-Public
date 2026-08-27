# Prompt for a fresh session: run the dc-accumulation campaign

Copy everything below the line into a new chat.

---

## Start here

Read `scripts/gantry/dc-accumulation/README.md` first, then
`docs/diagnostic-overview.md`. The README states the failure, the acceptance gates and the
hard constraints; the overview is the authoritative status document, with every claim
carrying its artifact path and an evidence grade.

Then work `scripts/gantry/dc-accumulation/step0_dc_sufficiency.md` through
`step5_promote.md`, in order.

**Step 0 is a gate, not a warm-up.** It needs no new runs and it decides whether the whole
campaign is aimed at the right quantity. Do not start step 4 before it is answered.

## The rules that matter here

**Artifacts beat documents.** This repository contains confident, well-written claims that
later measurement falsified, including several written the day before this prompt. Where a
document and a stored number disagree, the number wins. Two specific cases you will hit:

* `narrowband-objective-problem-2026-07-26.md` §5 and
  `flat-direction-problem-2026-07-26.md` §2-3 are **void framings**. Do not adopt them.
* ARTBP is listed as "ruled out" in three documents. **It is not.** Five converged 20-epoch
  runs pre-date every one of those documents and show a 4-6x drift reduction on the
  production path. Step 3 covers this.

**Pre-declare the reading before every run.** Write down what each outcome would mean, and
what the control must show for the run to be readable at all, BEFORE launching. On
2026-07-26 four runs were voided by their own controls and three hypotheses were killed by
their own tests. That is the mechanism working, and it only works if the reading is written
first.

**State the horizon with every error number.** The same ANN-off model measures `7.86e-05` at
2 s and `1.66e-04` at 12 s. At least two wrong conclusions in this repo came from comparing
across horizons, and one voided run came from a diagnostic that sat inside the same horizon
blind spot it was built to measure.

**Report voids and refutations, do not quietly drop them.** A run whose control failed has
zero readable rows however interesting the treatment rows look. Keeping them listed is what
stops them being re-run.

**Seeds.** The project floor is 3. Everything measured on 2026-07-26 is 1 seed and is below
it. Do not let a new result inherit that weakness silently; say so if it does.

## Config traps that have already invalidated runs

* `RunConfig` defaults `up_sample = 2`; the entry file and every checkpoint use `1`.
* Trimming `TRAIN_FILES` changes `compute_normalization`, which changes the encoder built
  from `norm.x_all`, which changes every number downstream. A 4-record trim moved epoch-0
  from `1.66e-04` to `1.13e-01`. Validation trims are safe; training trims are not.
* Filtering non-finite values out of a metric series makes divergence look like a flat pass.
* `gantry_ckpt_*.pt` is the **best** checkpoint. Since the failure is that best = epoch 0,
  that file IS the initialisation. Use `*_last.pth` for a trained model.
* The `.pth` files pickle `gantry_dynamic` as a top-level module, so put `scripts/gantry` on
  `sys.path` before `torch.load`. They carry their own `norm`; take weights and `norm`
  together, never mixed with locally computed constants.
* Those files also hold full per-epoch histories (`Loss_val`, `Loss_train`, `Loss_val_nf`,
  `Loss_train_nf`). Several questions are answerable by reading rather than running.
* Long runs: background jobs get killed unpredictably on this machine. Checkpoint every
  epoch and make runs resumable. Piping a run through `grep` block-buffers stdout, so the
  log stays empty until exit.

## Environment

`conda run -n GraduationProject python -u <script>`. For anything over a few seconds use
`PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 conda run --no-capture-output -n
GraduationProject python -u <script>` so output streams.

## Deliverables

1. Answers to steps 0-3, each with its numbers, artifact paths and grade.
2. A run-table row in `docs/gantry-augmentation-problem-log.md` §12 for every run,
   hypothesis stated before launch and outcome after.
3. New measurements appended to `docs/results-log-2026-07-26.md` (numbers only, no
   interpretation) and to `docs/diagnostic-overview.md` as dated addenda -- do not rewrite
   that document, it was produced by an independent pass and its separability is the point.
4. Step 3 additionally requires fixing the ARTBP anti-scope in
   `docs/dc-accumulation-research-brief-2026-07-26.md` §3.

## One thing to check early

`scripts/gantry/pysynth-data/b0_continue_training.py` was left running. Check
`scripts/gantry/pysynth-data/results/B0_continue.json` before assuming its question is open.
Its readings are pre-declared in the script, and the session that launched it expected it to
refute its own hypothesis.
