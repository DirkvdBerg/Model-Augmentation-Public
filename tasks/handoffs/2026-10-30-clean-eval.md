# Handoff: finish D-284 step 4b on `Clean/` (ruled keep-audit items) and add the run evaluation
**From**: session of 2026-10-30, revised after Dirk's review | **Branch**: Augmentation | **Effort suggested**: high
(well specified, but it touches training code under bitwise gates and adds a new module)

## 0. What Dirk values in `Clean/` (read first; it decides every open choice)
`Clean/` is code Dirk hands to ASMPT.
- **Production code only**: model, training and the run's own evaluation. Everything that verifies the cleanup
  (comparisons with mainline or thesis CSVs, reproduction checks, smoke or test modes, harness hooks, comments that
  name the harness) lives outside it, in `clean-up-verification/` at the repo root (Dirk: "i want to be able to test
  it but it shouldnt be in the clean code").
- **No invented thresholds.** A check is exact (bitwise, or equal on the same computation) or derived from the data
  (a CSV's printed precision). Never a picked tolerance. If you think one is needed, name the source of the
  difference and its size, and ask Dirk.
- **Minimal** = written in a smart efficient way, not fewest lines; useful options and readable helpers stay
  (amendment 9 point 2). **Optimal** = no repeated computation.
- Settings are parameters at the top of the entry script; names follow the thesis ("generalised", not "logical").
- Implement Dirk's rulings and nothing else. A new idea goes to Dirk as a proposal, never into the code.

## 1. Task
Finish step 4b of the second optimisation round of `Clean/` by D-284 amendment 9 (`docs/decisions.md`), with the run
evaluation and the E items of amendment 9 points 6 and 7, and prove everything once at the end. In this order:
1. **Gate the rename alone** ("logical" to "generalised", amendment 9 point 5): L1-2 on 221-lowpass and 227-lowpass,
   MM16 reverted, so every later gate starts from a passing tree.
2. **Implement the ruled keep-audit items** of `tasks/clean-code/keep-audit-merged.md`, one change group at a time,
   each with its gate as `tasks/handoffs/2026-10-26-clean-code-opt/06-audit-claude.md` item 4 describes (layer per
   row, MM16 reverted during every gate and restored after, harness-edit rules, markers on Jan's code). The items:
   K3 (only the single controller lookup per record; `use_gain`, the gain argument and `controller_ss`'s `gain` stay),
   K8, K13, K14, K15, K23, K24, K26, K31, K41, K42. Not implemented: K2 and K20 (the evaluation needs the test
   records and `out_gate`), and K1 only for what the evaluation in step 4 does not use. Every other K stays as it is.
3. **Codex diff review of the K items** (`06-audit-claude.md` item 5; Dirk gives part B of `06a-audit-codex.md`).
4. **The E items** (amendment 9 points 3, 6, 7), one group at a time with its gate:
   - E1 `evaluate(run_dir)` as specified below, without any check line (gate: training output bitwise unchanged,
     L1-2; plus a smoke run through `clean-up-verification/smoke.py`);
   - E2 smoke mode out of `Clean/` (`--smoke` in `Clean/gantry_interconnect_dynamic.py:142`, `:170-172`; the SMOKE
     print `:89`; `ps2_smoke` in `config.py:81`, `:325` and the branch in `training.py:245`). The `config.json`
     reader must still accept `PS2_SMOKE` from existing run folders (check `_CLEAN_KEYS`, `config.py:47`, and the
     rerun path), so reruns of the thesis runs keep working (gate: L3c and the smoke script);
   - E3 the two comments that name the harness (`config.py:71`, `:132`) say what the code does without it;
   - E4 training writes `loss.npz` beside `val_hist.npz` (gate: L1-2 bitwise for every existing output; the new file
     is the only addition).
   E items carry their E ID wherever a K item would carry its K ID (harness edits, `DROPPED_IN_CLEAN`, markers).
5. **`clean-up-verification/`** (new folder, repo root; scripts only, outside `Clean/`), in this order:
   1. `loader_check.py`: on I1, Clean's loaded servo error against `y_Y - r_Y` of
      `Thesis-writeup/Results/24-10-2026/CSV/res_time_trace.csv` (which starts at k0 = 30, t = 7.5 ms). Report where
      in time the difference sits (one-sample alignment, record edges, noise version). It must be explained before
      the R1 check means anything (section 5).
   2. `smoke.py`: builds the short configuration from outside (2 s records, 4 updates, 2-iteration polish through the
      `RunConfig` fields; the short PS2 settings of today's `training.py:245` branch by patching the
      `PS2Spec` that `training.py` builds, `ps2_opening.py:21`, a frozen dataclass) and runs training
      and `evaluate()`. Replaces the `--smoke` flag.
   3. `val_check.py`: on the smoke run folder (CPU), the val mean of the augmented model from `evaluate()` must equal
      the score the training stored for the same checkpoint exactly. First establish which checkpoint the stored best
      score belongs to (best Adam checkpoint or the polished final one) and compare like with like.
   4. `r1_check.py` (prepared here, run in the step-5 server job, which already loads the thesis checkpoints):
      `evaluate()` on the R1 final run folder (job 89792), MM16 reverted, reproduces `res_errors.csv` R1 `model`,
      `physics` and `base_true` NRMS_e per record and axis to the CSV's printed precision (`%.6e`, 7 significant
      digits). The MM16 effect is reported as a measured number, not used as a tolerance.
6. **Codex review of the E items and the new folder** (the same part B, with amendment 9 points 6 and 7).
7. **Prove, once**: `06-audit-claude.md` item 6 (scoreboard `audit-after`, goal check, full suite, new `final`
   snapshot, hash updates, report section "Keep audit (D-284)"). In the goal check, the kept rows of amendment 9
   point 2 carry the reason "useful option or readable name, amendment 9"; each function of `evaluate()` needs a
   reason of `06c-audit-criteria.md`; the evaluation needs a coverage case: add the smoke run with `evaluate()` to
   the scoreboard's cases. The full suite runs here only, not after step 2.

### The evaluation, exactly as approved
- **Entry**: one `evaluate(run_dir)` (suggested home: a new `Clean/gantry_dynamic/evaluation.py`, into which the two
  helpers of `closed_loop_report.py` move and that file is deleted; that is the remainder of K1). Called at the end
  of `main` in `Clean/gantry_interconnect_dynamic.py`, and on its own. Parameters at the top of that script:
  ```python
  # Evaluation
  EVALUATE = True        # evaluate the run after training
  EVAL_RUN_DIR = None    # a run folder: evaluate only that run, no training
  ```
  `--evaluate RUN_DIR` and `--no-evaluate` may override them; they must not be the only way (Dirk). With
  `EVAL_RUN_DIR` set: read its `config.json`, build the model, load the final checkpoint, evaluate, stop.
- **Models scored**, per record of train (18), val (6), test (11): augmented; baseline with the trained parameters
  (`_without_learned(fit_sys)`); baseline with the true parameters (`_without_learned(fit_sys, _true_free(phy))`).
  Free-run closed loop with the training's own scorer (`closed_loop_free_run_rms_batch`, `return_pred=True`,
  `Clean/model_augmentation/fit_systems/closed_loop.py:259`), from each model's own k0 as the thesis evaluation does
  (`Thesis-writeup/Code/Evaluation/eval_common.py:6`).
- **Measure**: NRMS_e = RMS(e_hat - e) / sigma_e per record and axis from k0, sigma_e as `servo_sigma`
  (`Clean/gantry_dynamic/data.py:149`). Two frames, never mixed in one figure or table: stage (X1, X2, Y, as
  stored) and generalised (X, Theta, Y) with q = P^-T y, i.e. X = (X1 + X2)/2, Theta = (X1 - X2)/L_b
  (`Clean/model_augmentation/systems/gantry_ss.py:56-61`, y = P^T q), each generalised channel normalised by its own
  sigma_e. Noise floor per record and axis: RMS(e_noisy - e_noisefree) / sigma_e, the noise-free twin from
  `MANIFEST.csv` column `pair_file`.
- **Terminal output** (and so the `.out` file): `mocks/terminal.txt` is the format, without its val check line
  (amendment 9 point 6). Header; per split one table with a row per record, a mean row, columns augmented, baseline
  (trained), baseline (true), noise floor, each X1 X2 Y, then the mean over records and axes per model; the same
  tables in the generalised frame; the physical-parameter table (true, start, learned, relative error) from
  `estimation_parameters()` and the true combinations as `_true_free` computes them; the list of saved files.
- **Saved under `<run>/evaluation/`**:
  ```
  loss.png  scores.csv  predictions.npz  params.csv  evaluation.json
  train/  val/  test/
    nrms_stage.png        bars X1, X2, Y
    nrms_generalised.png  bars X, Theta, Y
    records/
      <record>_stage.png
      <record>_generalised.png
  ```
  - `nrms_*.png`: three panels (one per axis), linear scale, a bar group per record plus a set-apart mean group,
    bars augmented, baseline (trained), baseline (true) (`mocks/*_nrms_stage.png`).
  - `records/*.png`: per axis the residual (e_hat - e)/sigma_e over the whole record for baseline (trained) and
    augmented, linear, record NRMS_e of baseline, augmented and floor beside each panel, reference X and Y in a thin
    bottom panel (`mocks/I1_stage.png`, `mocks/I1_generalised.png`). No zoom, no moving RMS, no overlay of e.
  - `loss.png`: top training loss (windowed MSE) against the update, bottom validation NRMS_e with the selected
    checkpoint and the L-BFGS result marked (`mocks/loss.png`), drawn from the run's `loss.npz` (E4: written by
    training beside `val_hist.npz`: update, epoch, time, training loss, validation score, selected update, the polish
    rows).
  - `scores.csv`: split, record, frame, model, rms and NRMS_e per axis, sigma_e, floor. `predictions.npz`: per record
    t, r, e, e of the noise-free twin, e_hat per model, k0. `params.csv`: name, true, start, learned. Floats in the
    CSV files with full precision (`%.17g`), so they read back exactly. `evaluation.json`: checkpoint file, run id,
    git commit.
  - Every figure is redrawable from these files alone.
- **Done-criterion, minimal**: every new function passes the `06c-audit-criteria.md` test; nothing reimplements the
  existing scorer, loader, `servo_sigma`, `P` or checkpoint loading.
- **Done-criterion, optimal**: one closed-loop free run per record per model; both frames, every table, every figure
  and every saved file come from those predictions, never from a second simulation.

## 2. Out of scope
- The server run (step 5); prepare nothing for it beyond what `06-audit-claude.md` item 6 updates and
  `clean-up-verification/r1_check.py`.
- Moving the existing harness out of `Thesis-writeup/Code/Verification/clean-code/` (after step 5, Dirk decides).
- `Thesis-writeup/Code/Evaluation/` and `Figures/` (todo item 8, a later session); they keep importing mainline.
- `Clean/blackbox/` (D-285, Dirk: a separate session) and `Clean/data_generation/` (D-286). The six `.m` files
  already renamed there are covered by no harness gate: list them in the final message for the D-286 session to
  check; do not run MATLAB here.
- The two names Dirk has not ruled on: the `.mat` field `x_logical` and the vendor function `gtd_logical_to_stage`.
- Suggestions Claude made that Dirk did not rule on: record type in the table names, a D-204 note under the
  parameter table. Do not implement.
- Rounding items, rejected items of amendment 2, every K not listed in section 1 step 2.

## 3. Where things stand
- Last commit `6981b5a`; tree dirty (Dirk commits). `Clean/` equals snapshot `final` (manifest `98d23ae2...1f99`)
  except the rename "logical" to "generalised" of 2026-10-30 (amendment 9 point 5): `Clean/gantry_dynamic/data.py`
  (locals at lines 219 to 228; SHA256 now `f42d4118...a3b8`, was `c3c1275e...30b5`) and `controller.py` (comment at
  line 28; now `0c563103...7611`, was `e2ba1146...6887`), plus six files of `Clean/data_generation/` (untracked
  folder, D-286). Names and comments only; the MM16 line is still `data.py:218` and its `sed` still applies, but the
  reverted-`data.py` hash `360155fc...a6bb` cited in the implementation report ("Server commands (D-284 step 5)") no
  longer holds.
- `clean-up-verification/` does not exist yet.
- No job running. The coverage job of item 0 finished (`simulations/clean-golden/perf/audit-before/`).

## 4. Established and verified
- Merged audit: 273 sites, both audits list the same sites; all 47 K items bitwise
  (`tasks/clean-code/keep-audit-merged.md` sections 1 to 3). Baselines: `perf/audit-before/static/static.json`
  (80 ruff, 15 vulture, radon E 1, D 1, C 11), `coverage/combined.json` (87.6 % statements, 62.5 % branches);
  harness baseline `simulations/clean-golden/src-snapshots/audit-harness-before/` (29 files).
- `_without_learned` zeroes `ann.out_gate` and `theta_frozen` on a deep copy; `_true_free` maps the true
  combinations through `free_from_combinations` (`Clean/gantry_dynamic/closed_loop_report.py:11-46`).
- `closed_loop_free_run_rms_batch` already returns physical predictions with `return_pred=True` (`closed_loop.py:259`,
  `:298-324`).
- Records per split (K1 controller, `MANIFEST.csv`): train 18, val 6, test 11 (I1 to I6, E1 to E4, E6).
- The thesis evaluation scored every model on the same measured data: `make_results.py` (`time_trace`) asserts the
  measured y of the R1, R5 and black-box caches agree to 1e-12 m.
- Reference numbers for R1 (job 89792), mainline evaluation: `Results/24-10-2026/CSV/res_errors.csv` rows run R1,
  models `model`, `physics`, `base_true`, and `floor`, written with `%.6e`; `res_obc_combinations.csv` rows R1 (the
  mock tables use them).

## 5. Assumed but not verified
- **Clean's loader against the thesis trace**: on I1, Y, `_load_record` e against `y - r` of `res_time_trace.csv`
  differs by up to 1.0e-7 m, about 0.5 % of the servo error (about 2e-5 m) and four orders above the CSV's printed
  precision. Not a match until explained; likely candidates are a one-sample offset against k0, the filter's edge
  transient, or a different noise version. `clean-up-verification/loader_check.py` settles it first.
- **Training-loss history after `fit`**: `checkpoint_load_system` restores the `_best` dump's `__dict__`
  (`interconnect.py:629-636`), so `Loss_train`, `batch_id`, `epoch_id`, `time` after `fit` are probably the history
  up to the best checkpoint, not the whole run (mainline fixed this, D-172). E4 must write the whole history; read
  where it is complete before writing `loss.npz`.
- **Test-record controllers**: `_load_record` registers each record's controller, but the simulator's bank and rows
  are built for train and val (`controller.py:200`). How the scorer gets rows for test records is unread.
- **Harness tree checks**: H2 makes comparators refuse outputs of another tree; the rename already changed two
  grey-box hashes, so the first gate may refuse until the expected manifest is updated. Read `compare.py` before the
  first gate.
- **The CLI-smoke layer** of `run_clean_suite.sh` calls `--smoke`; after E2 it calls `clean-up-verification/smoke.py`
  (a harness edit, listed under E2).
- **Runtime**: unmeasured; validation alone is 47 s on case 221 (6 records, one model). Three models on 35 records
  is a guess of several minutes on CPU; measure on the smoke run first.

## 6. Tried and failed
- Moving NRMS_e figure per record (thesis `res_time_trace.py` style, 50 ms, log %) -> Dirk not happy -> replaced
  by the residual figure.
- Overlay of e, baseline and augmented over a whole record -> at 12 s and 4 kHz the traces cover each other; NRMS_e
  1.0 and 0.06 look the same -> rejected (the thesis needs a zoom for that view).
- A noise-floor band in the residual figure -> 0.0004 of sigma_e, invisible on a linear axis -> number only.
- Step-4b keep test applied strictly -> Dirk: too harsh -> amendment 9 softer line.
- A val check inside `evaluate()` with "OK below 1e-3", and an R1 check at "relative 1e-3" -> invented tolerances and
  verification inside the deliverable -> Dirk: out of `Clean/`, exact checks in `clean-up-verification/`.

## 7. Achieved
- Step 4b items 0 to 3 (coverage and baselines, both audits, merge, ruling recorded as amendment 9).
- Rename "logical" to "generalised": implemented; `py_compile` OK; not yet gated.
- Approved mocks: `tasks/handoffs/2026-10-30-clean-eval/mocks/` (terminal.txt, bar charts, I1 figures, loss.png,
  and the two scripts that drew them, `eval_mock2.py` and `rec_fig.py`; they import `Clean/` and read the final CSVs).
  The val check line in `terminal.txt` is superseded (amendment 9 point 6).

## 8. The open question
Nothing blocked.

## 9. Next action
Read `06-audit-claude.md` item 4 and the harness's tree check (`compare.py`, H2), then gate the rename alone (L1-2
on 221-lowpass and 227-lowpass, MM16 reverted).

## 10. Acceptance criterion
- Training unchanged: every per-group gate and the full suite PASS bitwise with MM16 reverted; E4's `loss.npz` is
  the only new training output.
- `Clean/` holds no verification: no smoke mode, no check line, no tolerance, no comment naming the harness
  (`grep -rn -i "smoke\|harness" Clean/gantry_dynamic Clean/model_augmentation Clean/gantry_interconnect_dynamic.py`
  finds only the accepted `PS2_SMOKE` read of old `config.json` files).
- Evaluation: `loader_check.py` explains the 1e-7 m; `val_check.py` gives exact equality on the smoke run;
  `smoke.py` produces every file of section 1; each figure matches its mock in layout; both done-criteria
  (minimal, optimal) hold; `r1_check.py` is ready for the step-5 job.
- Step 4b: the acceptance criterion of `06-audit-claude.md` section 10, with amendment 9's keep reason and a
  `06c` reason for each new function.

## 11. Read these first
1. Section 0 of this file; `docs/decisions.md` D-284 amendments 8 and 9.
2. `tasks/handoffs/2026-10-26-clean-code-opt/06-audit-claude.md` (items 4 to 6) and `06c-audit-criteria.md`.
3. `tasks/clean-code/keep-audit-merged.md` sections 2 and 3 (the K rows to implement).
4. `tasks/handoffs/2026-10-30-clean-eval/mocks/terminal.txt` and the figures beside it.
5. `tasks/clean-code/implementation-report.md` sections "Gates" and "Server commands (D-284 step 5)".

## 12. Do not
Run `--side mainline` of any harness script; submit server jobs; edit mainline, `Model-Augmentation-Public-main/`,
the thesis or the mainline goldens; implement K items Dirk kept; add a threshold, check or option Dirk did not rule
on; put verification code into `Clean/`; record a ruling Dirk did not give; read `kamtin-data/Data Telica/` or
`kamtin-data/Telica.mat`.

## 13. Operational
`conda run -n GraduationProject`; gates and the suite as detached jobs with a log, path to Dirk (live-output
convention); one numerical job at a time. Until E2, the smoke run is
`python -u Clean/gantry_interconnect_dynamic.py R6 lowpass --cpu --smoke` (output under `simulations/gantry_clean/`);
after E2, `python -u clean-up-verification/smoke.py`. Final message: items implemented per group with gate results,
the evaluation output of the smoke run (its terminal tables and figure paths), the results of the four
verification scripts, suite result, new `final` manifest, the six `.m` files for D-286.

## 14. Delegation
None for the implementation. At most one Explore subagent for caller searches outside `Clean/`.
