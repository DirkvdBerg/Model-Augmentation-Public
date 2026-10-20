# Handoff: make the final thesis results complete and consistent with the method (L-BFGS polish in every run, final R1/R2, a justified black box)
**From**: session of 2026-10-20 | **Branch**: Augmentation | **Effort suggested**: high (server state must be
established from logs before any job, and the reruns must reproduce the final campaign exactly)

## 1. Task
Bring the final campaign (D-271, runs R1 to R7, R13, R14, black box R8) to a state where every reported number comes
from the method as written in the thesis (`Thesis-writeup/Writing/sections/03_augmentation.tex`, Identification
subsection): Adam, then an L-BFGS polish that is kept only if it lowers the validation error (D-255; with OBC after the
basis rebuild, D-253). In order: (a) establish per run, from the server logs, whether the polish was accepted, rolled
back, or aborted, and whether R1 and R2 have finished; (b) for every run whose polish aborted (GPU memory), prepare a
polish-only rerun from that run's saved best Adam checkpoint, with the same code and settings as the campaign, list the
exact jobs and wait for Dirk's OK before submitting; (c) after the reruns, record per run the validation NRMS_e before
and after the polish, the outcome, and for accepted polishes the change of each of the ten combinations of beta in %;
(d) document the black-box comparison so Section VI-C and Appendix D can state and justify it (search grid, selection,
budget, convergence); (e) give Dirk the `final-eval` command that regenerates all results tables and figures from the
final checkpoints. Report findings to Dirk; the thesis text is not edited in this session.

## 2. Out of scope
- Thesis text (`Thesis-writeup/Writing/`): no edits. Where a result changes a statement in the text, list it in the
  report with the file and the sentence.
- Rerunning Adam training for any run: only the polish is rerun. Adam results stand (D-271).
- Changing training or evaluation code that produced the campaign. If the polish-only rerun needs a new entry point,
  it must call the campaign's own functions unchanged (see section 5).
- The noise-floor definition and the redesign of the generalisation figure (`06_results.tex`,
  `fig:res_generalisation`): open points for a later writing session (revnote in `05_setup.tex`, Metrics).
- The source and reference check of the thesis: a separate session.
- The 2026-10-16 Adam-start runs (`..._data_fit`, logs thesis-results-3) and the 2026-10-11 runs: superseded.

## 3. Where things stand
- Local branch `Augmentation`, last commit `e588e63` (2026-10-20); the tree is dirty (about 640 paths; Dirk commits
  himself). **The local training code has changed since the campaign ran.** The campaign ran with the code of commit
  `4207001` and its server copy (`tasks/handoffs/2026-10-18-final-results-figures.md` section 3). Use the server's code
  for the reruns; do not copy newer local training code to the server.
- Server repo `/dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation`; run folders
  `simulations/gantry_subnet/thesis_taf_lowpass_noise_data_lsq/<folder id>/`; logs
  `~/logs/augmentation/thesis-results-4/`; local log copies `Thesis-writeup/Results/18-10-2026/`.
- Status as last recorded (2026-10-18, before R1 and R2 finished):

| Run | Design (arm, start, n_a, PS) | Folder id | Polish status 2026-10-18 | Best val NRMS_e (log) |
|-|-|-|-|-|
| R1 | OBC, detuned, 2, PS | 89792 | running (update ~16,900 of 33,400) | 0.0807 (so far) |
| R2 | OBC, correct, 2, PS | 89796 | running | 0.0809 (so far) |
| R3 | U, correct, 2, plain | 89797 | aborted (GPU memory) | 0.2521 |
| R4 | U, detuned, 2, PS | 89793 | rolled back | 0.0735 |
| R5 | U, correct, 2, PS | 89798 | aborted (GPU memory) | 0.0739 |
| R6 | U, correct, 0, plain | 89799 | aborted (GPU memory) | 0.2868 |
| R7 | U, correct, 8, PS | 89800 | rolled back | 0.0714 |
| R13 | OBC, detuned, 0, plain | 89794 | accepted (0.29267 to 0.29259) | 0.2926 |
| R14 | U, detuned, 0, plain | 89795 | aborted (GPU memory) | 0.2870 |

- Black box R8: job 89756, 11 configurations, all COMPLETED; selected task 9, `nx16_w16_lr1e-3`, best validation
  NRMS_e 0.4819 (next best 0.5246, task 3). Output folder per `run_bb_search.sh:69`:
  `scripts/gantry/blackbox-cl-trainable/outputs/search/lowpass/<TAG>9_nx16_w16_lr1e-3_n80000_89756` (exact name via `ls`).
- The thesis text still labels R1 numbers "preliminary, update 19500 of 33400" (`06_results.tex`, open-points todos).
- Nothing of this task is running.

## 4. Established and verified
- Method as written: Adam with checkpoint selection on validation NRMS_e, then L-BFGS polish on a fixed set of windows,
  kept only if it lowers the validation error (`03_augmentation.tex`, Identification, last paragraph; D-255). With OBC,
  the polish holds `Phi_Gamma` fixed at the best checkpoint's beta and is compared with the error after the rebuild
  (D-253, `04_obc.tex` Identification and recovery). Polish settings: 100 iterations, history 50
  (`Thesis-writeup/Code/Server/greybox/run_thesis.sbatch:19`).
- Dirk's decision (2026-10-20): the polish stays in the method. A polish that does not lower the validation error is
  rolled back and reported as such in the results; an aborted polish is not consistent with the method and is rerun.
- The polish updates all trained parameters, beta included, so an accepted polish changes the reported theta_base.
  Table III (interpretability) compares combinations across runs, so every compared run must have had the same polish
  rule applied.
- Polish memory: the eager polish rollout needs 12 to 28 GB; R13's OBC polish peaked at 27.9 GB; 11 GB cards abort
  (`2026-10-18-final-results-figures.md` section 6). Grey-box runs are on partitions `hawaii,oahu`
  (`run_thesis.sbatch:3`).
- Training writes `gantry_ckpt_<id>` only at the end (after the polish) and deepSI's `<...>_best.pth` at every
  validation (`training.py:74-75, 353` per the 2026-10-18 handoff).
- Black-box grid: stated before launch 2026-10-08 (`Thesis-writeup/Code/Server/blackbox/thesis_r8_grid.conf`): fixed as
  the grey box (same 18 training records, 4 kHz, window 0.1 s, encoder lag 29, seed 1), searched nx, width, lr;
  selection lowest best-validation error, within 2 % the fewer parameters; plateau criterion and expansion rule in the
  same file. D-264: all 11 configurations run to 80000 updates with the matched loss and NRMS_e selection; the selected
  run is R8, no separate retrain; a best validation at the limit is reported as unresolved convergence.
- Grey-box budget: 33400 Adam updates (`06_results.tex` todos), batch 512, lr 1e-5, plus the polish
  (`run_thesis.sbatch:18-19`). Black box: 80000 updates, batch 256 (`thesis_r8_grid.conf`).
- `final-eval`: `THESIS_BB_DIR=<R8 folder> bash Thesis-writeup/Code/Server/submit.sh final-eval` runs
  `evaluation/run_eval.sbatch` as array 0-10 and then `make_results.py`; the same command replaces preliminary numbers.

## 5. Assumed but not verified
- That R1 and R2 have finished and how their polish ended (they ran on 24 GB cards; R13 needed 27.9 GB). Settle from
  their logs in `thesis-results-4` and `squeue`, both from Dirk.
- That a polish-only rerun can start from the saved best Adam checkpoint with the campaign code (load the checkpoint,
  rebuild `Phi_Gamma` for OBC runs, run the polish with the same fixed window set, apply the D-255 acceptance, write
  `gantry_ckpt_<id>`). Settle by reading the server's training entry point and `model_augmentation/fit_systems/lbfgs_polish.py`
  as they are on the server. Whether the polish's fixed window set is reproducible (same seed, same selection) is part
  of this.
- That a large-memory GPU partition is available for the reruns (at least 32 GB); the partition name is Dirk's input.
- Black box, expansion rule: the selected lr 1e-3 is an edge of the searched lr range {1e-3, 3e-3, 1e-2}. The rule in
  `thesis_r8_grid.conf` adds lr 3e-4 when the best sits on an edge. Whether that expansion was run, or replaced by D-264,
  is not recorded. Settle from the grid's outputs and the decision log; report it either way.
- Black box convergence: whether R8's best validation is at the 80000-update limit (unresolved convergence per D-264)
  or plateaued. The Discussion currently says it "had not converged within the available budget"
  (`06_results.tex`, Discussion). Settle from R8's `val_hist.json` or `result.json`.

## 6. Tried and failed
- Polish on 11 GB RTX 2080 Ti cards -> CUDA out of memory, aborted, Adam weights kept (R3, R5, R6, R14) -> the eager
  polish rollout needs 12 to 28 GB -> `2026-10-18-final-results-figures.md` section 6.
- Black-box search truncated at 20000 updates (2026-10-11) -> only the winner had its best validation at the cap, and 3
  of 11 configurations broke on NaN but were reported as converged -> selection at a truncated budget favours fast
  learners -> D-264 (full grid at 80000).

## 7. Achieved
- Adam training of R3 to R7, R13, R14 complete; R8 grid complete (section 3). Evaluation pipeline implemented for the
  final campaign (`2026-10-18-final-results-figures.md`); preliminary results produced on 2026-10-18
  (`Thesis-writeup/Results/18-10-2026/CSV`).
- Not achieved: the polish rule applied in R3, R5, R6, R14 (and possibly R1, R2); final R1, R2 numbers.

## 8. The open question
Nothing blocked for step (a). After (a): for each run without an applied polish, rerun it (the default, Dirk's
decision) on a GPU with enough memory. If (a) shows the polish-only rerun cannot reuse the campaign's own code path,
report the gap to Dirk with the smallest wrapper that calls the existing functions unchanged, and wait for his OK.

## 9. Next action
Ask Dirk for the current `squeue` output and the latest logs of all nine grey-box runs in
`~/logs/augmentation/thesis-results-4/`, then build the per-run table: finished or not, best validation NRMS_e and its
update, polish outcome (accepted / rolled back / aborted, with the log line), and which checkpoint files exist in each
run folder. That table decides the exact list of polish reruns, which you then present to Dirk as job commands for his
OK.

## 10. Acceptance criterion
- Every grey-box run in the campaign has a final checkpoint produced under the D-255 rule (polish accepted or rolled
  back), with its validation NRMS_e before and after the polish recorded; for accepted polishes the per-combination
  change of theta_base in % is recorded.
- The evaluation's validation NRMS_e of each final checkpoint equals the training log's final validation value to a
  relative 1e-3 (as in the 2026-10-18 criterion).
- No "preliminary" label remains in the regenerated results CSVs.
- A black-box note for Dirk states: the grid and when it was fixed, the selection rule and its outcome, the expansion
  rule status, the budget against the grey box (updates, batch, CPU-h), and R8's convergence status from its
  validation history.

## 11. Read these first
1. `tasks/handoffs/2026-10-18-final-results-figures.md`: run table, folders, checkpoint behaviour, `final-eval` details.
2. `tasks/final-run-code-changes-2026-10-16.md`: the campaign's polish rules (D-253, D-255, D-266, D-267).
3. `Thesis-writeup/Code/Server/blackbox/thesis_r8_grid.conf` and `docs/decisions.md` D-264: the black-box design.
4. `model_augmentation/fit_systems/lbfgs_polish.py` (as on the server): the polish itself.
5. `Thesis-writeup/Writing/sections/03_augmentation.tex` (Identification) and `05_setup.tex` (Compared models): what
   the text states about the polish and the black box.

## 12. Do not
- Do not submit any job before Dirk has seen the exact command list and said OK.
- Do not rerun Adam, change seeds, or change any training setting for the reruns.
- Do not copy local training code to the server (section 3).
- Do not use test numbers to decide anything (D-269).
- Do not run the evaluation locally on full data; the server runs it.

## 13. Operational
- Dirk copies files to the server by hand and converts line endings: `sed -i 's/\r$//' <each copied file>`.
- Grey-box jobs: `Thesis-writeup/Code/Server/greybox/run_thesis.sbatch` (partitions `hawaii,oahu`, 64 GB RAM, 24 h);
  the polish reruns need a GPU with at least 32 GB (partition from Dirk).
- Evaluation: `THESIS_BB_DIR=<R8 folder> bash Thesis-writeup/Code/Server/submit.sh final-eval`; logs to
  `~/logs/augmentation/thesis-results-4/`; `make_results.py` writes `res_*.csv` to `Thesis-writeup/Results/`.
- Runtime of a polish-only rerun: not measured; estimate it from the R13 polish duration in its log, not by guess.
- Environment: local `conda run -n GraduationProject`; server env `/dataB1/dirk_van_den_berg/conda-envs/GraduationProject`.

## 14. Delegation
None: the files and runs are known and the work is targeted.
