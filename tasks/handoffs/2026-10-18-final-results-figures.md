# Handoff: produce the thesis results tables and figures from the D-271 final campaign (train, validation, test)
**From**: session of 2026-10-18 | **Branch**: Augmentation | **Effort suggested**: high (evaluation code changes plus a
server job that must reproduce the training logs exactly)

## 1. Task
Make the evaluation pipeline in `Thesis-writeup/Code/Evaluation/` and `Thesis-writeup/Code/Figures/` produce every
results table and figure of the thesis (the design in `Thesis-writeup/Documentation/THESIS-RESULTS.md` section 5 and
`Thesis-writeup/Documentation/RESULTS-DESIGN.md`) for the final campaign, on the training, validation and test records
(plus the multisine twins, D-269), and prepare the server evaluation so Dirk can launch it. In order: (a) fix the six
known evaluation gaps of section 7.B; (b) let the evaluation load the best-so-far checkpoint of a run that has not
finished (R1, R2) and label every number from it "preliminary, update <n> of 33400"; (c) give Dirk the exact files to
copy, the line-ending conversion and the `final-eval` command (section 13); (d) once he returns the outputs, check them
against section 10 and report the results tables in metres (scientific notation) and NRMS_e. Rerunning the same scripts
after R1 and R2 finish must replace the preliminary numbers with no further code change.

## 2. Out of scope
- Any training run, relaunch or change to the training pipeline (`scripts/gantry/gantry_interconnect_dynamic.py`,
  `scripts/gantry/gantry_dynamic/`, `model_augmentation/`): the campaign is running and final (D-271).
- The polish-only rerun of R1 and R2 if their polish aborts for GPU memory (see section 5): a separate step Dirk decides
  after the runs finish.
- The agreement criterion for table C (predicted against observed parameter bias, `05_setup.tex` around the "stated
  before the results" line): Dirk's decision; compute and tabulate both quantities, do not judge agreement.
- Thesis text: no edits to Dirk's sentences. If a result contradicts the text, say so in the report.
- The 2026-10-16 Adam-start runs (`..._data_fit`, logs thesis-results-3) and the 2026-10-11 runs: superseded, not
  evaluated.
- `scripts/gantry/thesis-results/ablate_ps2_fallback.py`: retired (D-271).

## 3. Where things stand
- Branch `Augmentation`, last commit `4207001`; the tree is dirty (about 630 paths, Dirk commits himself). The D-271
  code is local and on the server.
- Runs (server `/dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation`, run folders
  `simulations/gantry_subnet/thesis_taf_lowpass_noise_data_lsq/<folder id>/`, logs
  `~/logs/augmentation/thesis-results-4/`, local copies of the logs in `Thesis-writeup/Results/18-10-2026/`):

| Run | Design (arm, start, n_a, PS) | Folder id | Status 2026-10-18 | Best validation NRMS_e (log) |
|-|-|-|-|-|
| R1 | OBC, detuned, 2, PS | 89792 | running, update ~16,900 of 33,400, about 14 h + polish left | 0.0807 (so far) |
| R2 | OBC, correct, 2, PS | 89796 | running, same | 0.0809 (so far) |
| R3 | U, correct, 2, plain | 89797 | done, polish aborted (GPU memory) | 0.2521 |
| R4 | U, detuned, 2, PS | 89793 | done, polish rolled back | 0.0735 |
| R5 | U, correct, 2, PS | 89798 | done, polish aborted (GPU memory) | 0.0739 |
| R6 | U, correct, 0, plain | 89799 | done, polish aborted (GPU memory) | 0.2868 |
| R7 | U, correct, 8, PS | 89800 | done, polish rolled back | 0.0714 |
| R13 | OBC, detuned, 0, plain | 89794 | done, polish accepted (0.29267 to 0.29259) | 0.2926 |
| R14 | U, detuned, 0, plain | 89795 | done, polish aborted (GPU memory) | 0.2870 |

- R8 black-box grid, job 89756, all 11 tasks COMPLETED. Selected configuration: task 9, `nx16_w16_lr1e-3`, best
  validation NRMS_e 0.4819 (next best 0.5246, task 3, so the "within 2 %, fewer parameters" rule of
  `Thesis-writeup/Code/Server/blackbox/thesis_r8_grid.conf` changes nothing). Its output folder follows
  `run_bb_search.sh:69`: `scripts/gantry/blackbox-cl-trainable/outputs/search/lowpass/<TAG>9_nx16_w16_lr1e-3_n80000_89756`
  (find the exact name with `ls` on the server); that folder is `THESIS_BB_DIR`.

## 4. Established and verified
- Run discovery: `eval_common.discover_runs` maps the designs above to folder ids from each `config.json`; with
  `THESIS_EVAL=final` the default folder is `..._data_lsq` and the prediction cache is
  `Results/thesis-eval/final-thesis_taf_lowpass_noise_data_lsq` (D-271, `eval_common.py:64-66, 87`).
- The grey-box evaluation loads `gantry_ckpt_<id>` (`eval_greybox.py:101`), which training writes only at the end
  (after the polish). Training also writes deepSI's `<...>_best.pth` at every validation (`training.py:74-75, 353`).
  R1 and R2 have no `gantry_ckpt_<id>` yet.
- Every run starts from the least-squares encoder start (D-271); the logs show all physical states reconstructed with
  at most 8.2e-9 unexplained variance and identical starts per baseline. The PS2 screen passed in R1, R2, R4, R5, R7
  (0.62 to 0.68 at update 150).
- Splits and twins: `eval_common.SPLITS = ('train', 'val', 'test', 'twins')`; validation decides design choices, test
  is reported for final results only (D-269). Records: 18 training, 6 validation, 11 test in the text (`05_setup.tex` about :207-216); the test manifest holds 12,
  and E5 (controller K2) is skipped in evaluation (`eval_greybox.py` about :110-115).
- `final-eval` (`Thesis-writeup/Code/Server/submit.sh` tier `final-eval`) submits `evaluation/run_eval.sbatch` as an
  array 0-10 (tasks 0-8 grey box R1 R2 R3 R4 R5 R6 R7 R13 R14, 9 black box, 10 overlaps/bias/negation) and then
  `make_results.py` after all tasks.

## 5. Assumed but not verified
- That `_best.pth` of a running grey-box run loads into the evaluation's model (it is deepSI's `torch.save` of the
  system dict, as the black-box evaluation already loads at `eval_blackbox.py:64`). Settle by loading R1's file in the
  eval code path on the server before the full array.
- That a model loaded from `_best.pth` reproduces the log's best validation NRMS_e (section 10 tests exactly this).
- The polish of R1 and R2 may abort on their 24 GB cards (R13's OBC polish peaked at 27.9 GB). Not this session's
  action (section 2), but the evaluation must read whatever final checkpoint exists when it is rerun.
- Line numbers of the audit findings below are from 2026-10-17; files may have moved since. Re-locate before editing.

## 6. Tried and failed
- Adam encoder pre-fit (20 epochs from a random start) -> state MSE 0.155, PS2 fell back in every run, 10x worse
  validation -> did not reach the minimum of the encoder fit and the default `torch.linalg.lstsq` call was
  rank-unstable -> D-270, D-271. Irrelevant to this task except: never evaluate the `..._data_fit` folder.
- Polish on 11 GB RTX 2080 Ti cards -> CUDA out of memory, aborted, Adam weights kept (R3, R5, R6, R14) -> the eager
  polish rollout needs 12 to 28 GB -> `03_augmentation.tex` revnote after "We only keep the polish". The evaluation
  itself runs on CPU (`run_eval.sbatch`), so it is not affected.

## 7. Achieved
A. Working: the final-campaign evaluation scaffold (discovery, per-campaign cache, identity check, all splits,
   black-box loader by result.json checkpoint name) from the 2026-10-16 implementation
   (`tasks/final-run-code-changes-2026-10-16.md`). Implemented; not yet run on the final campaign.
B. Known gaps (audit of 2026-10-17, read-only session "Audit code vs thesis Sections IV, V, appendix"); fix all six:
   1. `make_results.py` `res_structure` (about :100-123) averages all records together and defaults to the validation
      split; the design (`06_results.tex` about :18-19, :70, :88) reports with and without multisine separately and the
      table as the mean over I1 to I6, never one mean over all records. Produce per split.
   2. `res_generalisation` "baseline" (about `make_results.py:176-178`) is R1 at its detuned start; `05_setup.tex`
      (about :271-272, :285) says M_LPV uses the true parameters while `06_results.tex` (about :150) says "detuned
      baseline". The text conflicts with itself: produce both rows, labelled, and flag the conflict in the report.
   3. `Figures/res_generalisation.py` (about :31) and `Figures/res_obc_combinations.py` (about :39) show RMS in metres
      averaged over axes; the primary metric is NRMS_e per axis (`05_setup.tex` about :340-344). Plot NRMS_e per axis;
      keep metres where a figure is explicitly in metres.
   4. Table C (`Figures/res_obc_combinations.py` about :25-62) does not read `res_obc_bias.csv` (written by
      `eval_overlap.py` about :163-167) and lacks the NRMS_e and the physical-part-alone rows (`05_setup.tex` about
      :388-392, THESIS-RESULTS about :160-162). Add them, for training, validation and test stacks (D-268).
   5. `eval_negation.py` (about :83-92) prints c_neg against the full next state; the definition (`04_obc.tex` about
      :391-392, :407) is the cosine with the baseline one-step increment. Compute the increment version as c_neg, write
      it to a CSV per split.
   6. The black-box selection rule (lowest validation NRMS_e; within 2 %, fewer parameters) is not coded; code it in
      the evaluation so `THESIS_BB_DIR` is checked against the grid's `val_hist.json` or `result.json` files, and log
      the selected configuration.

## 8. The open question
Nothing blocked. One conflict to report, not decide: the "baseline" of the generalisation table (true parameters vs
the detuned start), item 7.B.2; produce both.

## 9. Next action
Fix item 7.B (all six) and add the preliminary-checkpoint fallback (task (b)), with one local test per change on the
2026-10-11 cached predictions or synthetic inputs where checkpoints are not local, then give Dirk the copy list, the
`sed -i 's/\r$//'` line for exactly those files, and the `final-eval` command of section 13.

## 10. Acceptance criterion
- For every grey-box run and the black box, the evaluation's validation NRMS_e (same measure, same records, D-262)
  equals the best validation sim-NRMSe in its training log (section 3 table; R1, R2 at the checkpointed update) to a
  relative 1e-3. A larger gap means the wrong checkpoint or a different measure.
- Every table and figure exists for train, validation and test (and twins where designed), each number labelled with
  its split; R1 and R2 numbers carry "preliminary, update <n> of 33400".
- Errors in metres use scientific notation (x10^-n); NRMS_e per axis is the primary metric.

## 11. Read these first
1. `Thesis-writeup/Documentation/THESIS-RESULTS.md` section 5: the list of results tables and figures.
2. `Thesis-writeup/Code/Evaluation/eval_common.py`: run discovery, splits, cache, what `THESIS_EVAL=final` changes.
3. `Thesis-writeup/Code/Evaluation/make_results.py` and `Thesis-writeup/Code/Figures/README.md`: what each producer writes.
4. `Thesis-writeup/Writing/sections/05_setup.tex` and `06_results.tex`: the stated metrics and table layouts.
5. `docs/decisions.md` D-262, D-264, D-268, D-269, D-271: measure, black-box rule, overlap stacks, splits, encoder start.

## 12. Do not
- Do not evaluate or mix in the `..._data_fit` or 2026-10-11 runs (section 6).
- Do not use test numbers to choose anything (D-269).
- Do not run the evaluation locally on full data (no checkpoints locally; CPU hours); the server runs it.
- Do not start or modify any training job; do not run git write operations.

## 13. Operational
- Server repo `/dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation`; Dirk copies files by hand, then
  converts line endings: `sed -i 's/\r$//' <every copied file>`.
- Launch (Dirk): `THESIS_BB_DIR=<the folder of section 3> bash Thesis-writeup/Code/Server/submit.sh final-eval`.
  Logs go to `~/logs/augmentation/thesis-results-4/` (`submit.sh` LOG_DIR default); `make_results.py` writes the
  CSVs to `Thesis-writeup/Results/` (`res_*.csv`) and the figure scripts read them.
- Runtime: not measured for the final campaign. The CPU evaluation of one grey-box run over all splits and twins is a
  free run per record; give Dirk an estimate only from a measured number (for example time one record in the first
  task's log), not a guess.
- Environment: `conda run -n GraduationProject` locally; server env `/dataB1/dirk_van_den_berg/conda-envs/GraduationProject`.
- After R1 and R2 finish (and any polish rerun), the same `final-eval` command replaces the preliminary numbers.

## 14. Delegation
None: the files are known and the work is targeted.
