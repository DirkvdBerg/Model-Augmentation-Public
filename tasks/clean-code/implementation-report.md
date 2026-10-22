# Implementation report: clean minimal code (step 4)

2026-10-20. Handoff `tasks/handoffs/2026-10-20-clean-code-implementation.md`; design `tasks/clean-code/design.md`
(revised) with D-274, D-275 and its amendments 1 to 4. All local layers pass bitwise on CPU; layers 4,
4b and the 6.4 GPU smokes run on the server (commands at the end). The judge fixes of 2026-10-21 (step
4c) changed `Clean/` and the harness; every local layer was rerun on the new tree, see "Judge fixes
(step 4c)", which supersedes the step-4 rows below for the current code.

## Result per layer

| Layer | Cases | Result | Runtime (golden / clean, per case) |
|-|-|-|-|
| static: pyflakes | 24 files | PASS: 0 undefined names (pyflakes 3.0.1, base env; nothing installed) | seconds |
| static: imports | 24 modules, only `Clean/` on the path | PASS: all under `Clean/` | seconds |
| 1 construction + forward | 16 (8 models x lowpass, noisefree) | PASS 16/16, bitwise: 259 to 302 items per case (data, normalisation, sigmas, windows, bank, OBC reference set, state after build and after the pre-fit with key/shape lists, LS rank and singular values, first OBC basis, encoder x0, pass_mats, OBC coefficient and tangent pair, 400-step closed-loop prediction, validation score) | 32 to 112 s / 35 to 177 s |
| 2 loss + gradients | 16 | PASS 16/16, bitwise: weighted loss and the gradient of every parameter in `parameters_with_names` order | in the layer-1 process |
| 3 short training | 5: 221-lowpass, 229-noisefree, 224-noisefree, 226-lowpass, 227-lowpass | PASS 5/5, bitwise, no fallback: both checkpoint pairs (hfn, encoder, Adam `step`/`exp_avg`/`exp_avg_sq`), `.npz` (incl. `bestfit`, `ps2_summary`, `lbfgs_outcome` with `reason`), `val_hist.npz`, config.json on the shared keys. 18 Adam updates, 4 validations, OBC epoch refreshes, PS2 S1/S2 on the PS2 rows; all five polishes accepted. Also the with/without check of the periodic `_last.pth` (D-275 amendment 2) | 935, 1202, 207, 216, 285 s / 741, 967, 317, 230, 343 s |
| 3b polish rejection | 2: 221-lowpass, 226-lowpass | PASS 2/2: the golden `_pre_lbfgs` pair in a freshly built model, `lbfgs_polish(..., val_before=0.0)`; both sides ROLLED BACK with the same reason, parameters equal to the start state (asserted in each run), outcome fields and the full state after restore and `_resync` bitwise equal | 290, 74 s / 243, 79 s |
| 3c reruns | 5: 226-lowpass (i) `_best`, (ii) polish-only, (iii) `_last`; 221-lowpass (i), (ii) | PASS 5/5, each code from its own layer-3 golden: both pairs' state_dicts and Adam entries, `.npz` (except `elapsed`, `orig_run_id`; `resumed_from` by file name), `lbfgs_outcome`, val_hist. Both codes restore the Adam moments from the deepSI dumps (amendment 3; log: "Adam moment estimates restored", resume at 2 done/2 remaining from `_best`, 3 done/1 remaining from `_last`). Clean-only asserts hold in all five: previous files byte-identical in `previous/<rerun_id>/` (dumps included for i, iii), config.json changed only by one `RESUMES` entry | 193, 73, 132, 612, 247 s / 199, 71, 116, 546, 234 s |
| acceptance `python Clean/gantry_interconnect_dynamic.py R6 lowpass --cpu --smoke` | 1 | PASS, runs to the end (rc 0): initial validation sim-NRMSe 0.3949, 4 Adam updates, polish ACCEPTED, best validation sim-NRMSe 0.359563; config.json, both checkpoint pairs, val_hist.npz and the deepSI `_best`/`_last` dumps written (out root and LOCALAPPDATA redirected to `simulations/clean-golden/smoke/`) | 249 s |
| 4, 4b | server | pending; scripts ready, see "Server commands" | |
| loader script test (not the layer-4 result) | R1, R4, R6 staged from the layer-3 goldens, 0.6 s records (`--truncate-s`) | PASS after deviation 7(e). Run A (default driver): every l4 array bitwise equal on all three splits, val score equal to the recorded `bestfit` (rel 0) on both sides, l4b arrays equal; l4b overlap, bias and negation rows of R1 and R4 differed at 1e-13 (cause in 7(e)). Run B (gelsd pin): all l4b rows and arrays bitwise equal. Run C, `--eval-check` on R4 (mainline): the loader equals the unmodified `eval_greybox.evaluate`, `eval_overlap.evaluate`, `eval_negation.evaluate` and `inspect_checkpoint.added_poles` in all 27 compared items | A 965 / 1036 s, B 129 / 124 s, C 335 s |
| 6.4 (a), (b) | server | pending; scripts ready, see "Server commands" | |

## The folder

`Clean/`: the 24 files of design section 1, each a copy-and-trim of the mainline file with the same
package path. Measured: 4,447 lines (design estimate 3,500 to 4,500). Largest: interconnect 644,
blocks 520, training 415, lbfgs_polish 359, config 337. Method: a scratch tool removed the dropped
definitions by syntax tree and the non-marker comments by tokenizer, so every kept statement is the
mainline statement byte for byte; the row trims were then applied as exact, counted replacements (an
edit that did not match exactly once stopped the script). The D-4 marker `# CHANGED (clean): trimmed
to the spec path` sits under the header of every trimmed Jan definition (25: `Interconnect` and nine of
its methods, `Signal_Connection`, `SSE_Interconnect` and eight of its methods and nested functions,
`Block`, `Static_ANN_Block.__init__`/`forward`, `Discrete_Nonlinear_Function_Block` and its `forward`);
the one new statement in a Jan definition, the periodic `_last.pth` save in `fit.validation`, carries
`# CHANGED (clean): D-275, ...`.

Launch: `python -u Clean/gantry_interconnect_dynamic.py [preset] [noise] [--cpu] [--smoke]
[--resume RUN_DIR [--from {best,last}] | --polish-only RUN_DIR] [--data-root P] [--out-root P]`.

## Harness (`Thesis-writeup/Code/Verification/clean-code/`)

| File | Role |
|-|-|
| `hcommon.py` | one code root per process, thread pinning, row configs, fingerprints (meta.json), array capture |
| `capture_l12.py`, `run_l12_all.sh` | layers 1 and 2 for one case (mainline writes the golden; clean loads the golden post-pre-fit state) |
| `run_l3.py` | layer 3 (mainline: `EP.main()` with `EP.CFG` = budget; clean: `main(cfg, run_id)`) |
| `run_l3bc.py` | layers 3b and 3c |
| `compare.py`, `compare_l3.py`, `compare_l3bc.py` | fingerprint refresh check, then bitwise comparison; writes `compare-<layer>.json` |
| `check_imports.py` | step 3: every module imported with only `Clean/` on the path |
| `run_seq.sh` | runs harness jobs one after another, output streamed |
| `l4_stage.py`, `l4_loader.py`, `compare_l4.py`, `server_l4.sbatch` | layers 4 and 4b (server); `l4_loader.py --truncate-s` is the local script test only |
| `gpu_smoke.py`, `compare_gpu.py`, `server_gpu_smoke.sbatch` | 6.4 (a) and (b) (server) |

Goldens: `simulations/clean-golden/L<n>/<case>/` (mainline), clean outputs in
`simulations/clean-golden/_clean/L<n>/<case>/`, each with `meta.json` (SHA256 of the 19 copied mainline
files and of the clean files, both manifests, every record read; data root, libraries, threads,
git provenance, budget). `compare.py` recomputes the mainline and data hashes and refuses a stale golden.

## Deviations from design.md, each with its cause

None changes a number on the CPU path; layers 1 to 3c are the evidence.

1. **Statements left dead by a listed removal are removed too** (rule 1 applied to locals; pyflakes
   "assigned but never used" and "imported but unused"): `closed_loop_rollout` `nf = u_data.shape[1]`
   (only the collapsed header and the removed checkpoint loop read it); `build_closed_loop`
   `train_weights = None` (overwritten by the collapsed servo branch); `attach_obc` `ann = ...`
   (read only by the removed block-type guard); `SSE_Interconnect_Composed.loss` `blocks = ...` (read
   only by the removed param_loss sweep and orth penalty); `Interconnect.forward` `x_size` (read only by
   the removed output-shape assert) and `u_size`, `input_has_correct_dimension` (never read, also in
   mainline); `init_connection_matrices` `n_out` (read only by the removed wiring checks);
   `SSE_Interconnect.fit` `batch_id_start` (read only by the removed batch-rate string);
   `_sync_prediction_state_for_validation` `built` (read only by the removed diagnostic dict);
   `linear_encoder_init_aug.forward` `state_has_correct_dimension` (never read, also in mainline); the
   fourteen raw-parameter names of the `gantry_ss` import in `_Trainable_Gantry_State_Block.__init__`
   (read only by the removed `params_init is None` branch). Kept as the design says although now inert:
   the `terms`/`sterms` parameters of `_deriv_with`/`_deriv_both_with`, the `if self.simulator is not
   None` headers of `simulate` and `cal_validation_error` (the design removes only their other branch).
2. **Imports follow the kept names beyond the "Imports" bullets** (rule 1): `lbfgs_polish`
   `contextmanager` and `field` (unused after `_frozen`/`_quiet_probes`; `field` was unused in mainline);
   `pre_encoder` the deepSI encoder names, `System_data`, `fixed_System_data_norm`; `utils` matplotlib and
   `system_data`; `torch_nets` `Tensor`, numpy; `deepSI_corrections` `hist_future_dataset`;
   `obc_gantry` `replace`. The three star imports of `GD/model.py` are resolved to the names the kept
   code reads (`expansion_matrix, selection_matrix, Interconnect, Reduced_Gantry_State_Block,
   Linear_Output_Block, Static_ANN_Block`), so pyflakes can check every name; the same objects are bound.
3. **`SSE_Interconnect.fit` signature**: the parameters of the removed options go with them
   (`concurrent_val, timeout, num_workers_data_loader, print_full_time_profile, scheduler_kwargs,
   list_val_measures`); `optimizer_kwargs` and `auto_fit_norm` stay (the merged `train_model` call passes
   them).
4. **Comments** (handoff section 2): non-marker comments of kept lines are dropped, every `# CHANGED`
   comment is shortened to its first sentence, docstrings to their first paragraph except those that
   carry a `# THEORY:`/`# HEURISTIC:` label (`_resample_u`, `fourth_order_central_difference`,
   `ControllerBank`, `ControllerBank.step`, `calibration_split`), which keep the label paragraph. A
   script check confirms every label of a kept definition is present. Comments kept although they are
   not markers: the two D-274 comments of `PS2Opening._fit_map` (change 1 and the required `T.var(0)`
   note), the D-274 comment of `run_lbfgs_polish`, the D-274 K1 comment of `record_files`, the D-271
   encoder-start comments of `main` and `build_model`. A double hyphen used as a dash in copied comments is
   replaced by a colon or semicolon (CLAUDE.md).
5. **Polish print**: the peak-memory suffix that the design turns into `''` is removed with its `%s`
   (identical text).
6. **Static check**: `pyflakes` is not in `GraduationProject`; the base env's pyflakes 3.0.1 (analysis
   only, no import of the code) was used, so nothing was installed and Dirk did not need to be asked.
7. **Harness**: (a) the mainline layer-3 and 3c runs point `EP.save_dir` at the golden folder instead of
   writing into the mainline family folder and moving the run (6.2), and skip EP:456-471 (the post-run
   report and diagnostics, after the final checkpoint, H2); (b) layer 4 redirects `eval_greybox.CACHE`
   into the harness output instead of `Thesis-writeup/Results/thesis-eval/` (handoff section 12);
   (c) layer-1 arrays above 8 MB (OBC basis U, training windows uf and yf) are stored as a SHA256 of
   their bytes plus sum and max, compared by hash; `capture_l12.py --full` stores them whole when a
   mismatch must be localised; (d) the mainline side is given `THESIS_SMOKE=1` for the PS2 smoke spec
   only, its `EP.CFG` replaced by the budget (6.2); (e) the layer-4 loader pins the default CPU
   least-squares driver to `gelsd` in its process (both sides, and the unmodified evaluation its
   `--eval-check` calls). Cause, found by the local loader test: the overlap, bias and negation rows
   of R1 and R4 differed between the codes by up to 3e-13 relative while the lstsq inputs (J, F_b,
   Z_step) were bitwise equal; torch's default driver `gelsy` returns different last bits from call
   to call on the same tensors at one thread (also under `MKL_CBWR=COMPATIBLE` or `AUTO,STRICT`),
   while `gelsd` and `gels` are repeatable. Training already passes `driver='gelsd'`
   (`gantry_dynamic/model.py`), so layers 1 to 3c are unaffected. Consequence outside this task: the
   thesis overlap, bias and negation numbers (`eval_overlap.py`, `eval_negation.py`, default
   driver) are repeatable to about 1e-13 relative only, far below their printed precision; the
   evaluation scripts were not edited.

## Judge fixes (step 4c)

2026-10-21. Handoff `tasks/handoffs/2026-10-20-clean-code-judge-fixes.md`; findings `judge-codex.md`;
implementation choices D-279. Every accepted finding is fixed as stated, none reverted. `Clean/` is now
4,504 lines (was 4,447): the fixes add slightly more than the trims remove. pyflakes 0 warnings; all 24
modules import with only `Clean/` on the path.

### Per finding

| Finding | What changed, where | Tested by |
|-|-|-|
| D-278 F3 | `run_lbfgs_polish`: `val_before = bestfit` whenever finite, polish-only included; a U job that ran no Adam update first runs one eager validation (mainline does this for polish-only; Clean also for the spent-budget resume, same reason, D-279) | 3c 221-lowpass-ii (OBC) and 226-lowpass-ii, both codes: `val_before` = the source's `bestfit` exactly |
| D-278 F2 | `train`: a resume with no epochs left skips Adam; `fit` gets `epochs_remaining`, no `or hp['epochs']` | new 3c 226-lowpass-iv, both codes: resumes at 3 of 3, no `adam` history row, polish runs |
| F1 | `main`: a fresh run refuses an existing run folder before reading data (`FileExistsError` naming `--resume`/`--polish-only`); the folder is created when config.json is written | new 3c 226-lowpass-v (clean): refused, folder byte-identical |
| F6 | `main`: `_prepare_rerun` split into `_rerun_source` (nothing moves) and `_archive_rerun`, called only after data, model (CUDA check) and the full `load_checkpoint` succeeded; `resolve_checkpoint` reads the pair's `.npz` into a dict and closes it (Windows cannot move an open file) | new 3c 224-noisefree-vi (clean): truncated `_best.pth` refused, run folder and dumps byte-identical, no `previous/` |
| F16 | `SSE_Interconnect_Composed.checkpoint_save_system`: deepSI's save into `<file>.tmp`, then `os.replace` (retried on a Windows sharing violation); every dump, same save points, `--from last` unchanged | all layer-3/3c clean runs write and resume from such dumps |
| F5 | `lbfgs_polish`: one `try` from the start validation to the final scoring (FixedBatch, probes, steps, final loss, final validation); an uncaught exception is an aborted polish: restore, resync, CUDA cache released after the failed frames are freed; a failed probe returns its exception without traceback before the eager retry | layers 3, 3b, 3c unchanged bitwise; the abort path itself is not reached on CPU |
| F4 | dumps carry `epochs_before` and `ps2_opening` (phase, updates; set in the fit loop after each opening step); a dump's count is round(epochs_before + epoch_counter); `load_checkpoint` refuses a dump inside a started, unfinished opening; a finished opening is never restarted; comments no longer claim an uninterrupted continuation. Mainline rounding of fractional epochs kept (an exact budget changes the tested 3c numbers) | new 3c 226-lowpass-vii (clean): the second continuation resumes at 4 (not 2), runs 1 epoch, ends at 5; 224-noisefree-vi: a dump marked 3 updates into S1 refused before anything moves |
| F7 | `compute_normalization`: the `# THEORY:` label beside the first difference (mainline `data.py:350`) | static |
| F8 | removed `detect_algebraic_loop` with its assertion and import, the `transition_from_free` shape assertion, the `_true_free` round-trip assertion; forward batch-size assertion and external-data guards kept | layers 1 to 3c |
| F9 | `OBCLifecycle`: `n_refresh`, `n_rank_loss`, `timing`, the entry dict, `t0`, `build_s` removed; rank policy, column scaling and its Gram diagonal kept; the builder still returns `(J, seconds)` (evaluation interface) | layers 1 to 3c (OBC rows) |
| F10 | `WindowControllerIndex.last_counts` and `ClosedLoopSimulator.last_per_record` removed | layers 1 to 3c |
| F11 | removed `_make_closure.params`, `Dtype_DataLoader.seed` (fixed RNG seed 0 kept), `terms`/`sterms`, `train.norm`, `init_model`/`fit` `auto_fit_norm`, `fit` `optimizer_kwargs`, with their call-site arguments | layers 1 to 3c |
| F12 | add-to argument, field and parser branch removed from `connect_signals`, `Signal_Connection`, `parse_connection_function_method` (now 26 trim markers) | layers 1 to 3c |
| F13 | `GantryOBCCorrection`: constant affine branches (`theta_frozen` p+1, direction `theta[:-1]`, `jv + alpha fbar`); `space` attribute and the basis builder's `space` argument kept | layers 1 to 3c (OBC rows) |
| F14 | `prediction_coefficient` = `coefficient` under `torch.no_grad()` | layers 1 to 3c (OBC rows) |
| F15 | S2 restores its snapshot unconditionally; `fs_new_hz`, `dtype_np`, `dtype_pt` return the fixed values | layers 1 to 3c (PS2 rows) |
| H1 | `compare_l4.py` asserts the expected set on both sides (9 runs, 3 splits, every l4b array and row per run, a val_score check with finite positive `ref_score` per run, a mainline eval check over every run and part); less is PARTIAL; prints PASS, PARTIAL, FAIL or REFUSED, exit 0 only on PASS | local-test outputs: REFUSED (no manifest); synthetic complete, partial, perturbed sets: PASS, PARTIAL, FAIL |
| H2 | `hcommon.clean_tree_hashes` (every file under `Clean/`, `__pycache__` excluded), captured at process start (a tree changed during a run records none); `compare.py`, `compare_l3bc.py`, `compare_l4.py`, `compare_gpu.py` refuse a clean output from another tree; `l4_loader.py`, `gpu_smoke.py` record it | active in every comparison below |

### Goldens

D-278 reaches only a polish-only start and a resume with the budget spent. Recaptured from mainline: L3c
226-lowpass-ii and 221-lowpass-ii (old ones in `simulations/clean-golden/L3c-preD278/`); new 226-lowpass-iv.
For 221-lowpass-ii the recorded `val_before` moves from the rebuilt score 0.379254204730 to the selected
0.379254207528; the polished model is identical. All other goldens are kept: `compare.py` accepts their
pre-D-278 `training.py` hash for exactly that file and hash pair (D-279) and never for a ii or iv case.

### Result per layer (current tree, all clean outputs regenerated)

| Layer | Cases | Result | Runtime (golden / clean, per case) |
|-|-|-|-|
| 1, 2 | 16 | PASS 16/16 bitwise, 259 to 302 items per case in layer 1 | 32 to 112 s / 34 to 269 s |
| 3 | 5 | PASS 5/5 bitwise (221-lowpass, 229-noisefree, 224-noisefree, 226-lowpass, 227-lowpass) | 935, 1202, 207, 216, 285 s / 1102, 1242, 208, 223, 230 s |
| 3b | 2 | PASS 2/2 (forced reject, parameters restored on both sides) | 290, 74 s / 225, 70 s |
| 3c both codes | 6 | PASS 6/6 bitwise: 226-lowpass i, ii, iii, iv; 221-lowpass i, ii; D-278 checks hold on both sides | 193, 101, 132, 100, 612, 352 s / 211, 101, 164, 86, 563, 361 s |
| 3c clean only | 3 | PASS 3/3: v (F1), vi (F6, F4), vii (F4) | 9, 38, 133 s |
| acceptance `R6 lowpass --cpu --smoke` | 1 | PASS, rc 0: initial validation sim-NRMSe 0.3949, 4 Adam updates, polish ACCEPTED, best 0.359563 (as in step 4); config.json, both pairs, val_hist.npz, `_best`/`_last` dumps written, no `.tmp` left | 260 s |

### Deviations added in step 4c

- 7(f) harness: a clean 3c rerun archives its run folder and at once rewrites `val_hist.npz` under the
  same name; in the OneDrive-synced tree that write failed with Errno 22 in 3 of 11 reruns. `run_l3bc.py`
  now runs clean 3c cases in `%TEMP%\clean-harness\L3c\<case>` and moves the folder into `_clean/L3c/` at
  the end. In this proof only 221-lowpass-i ran that way (the other eight ran in place before the change).
- The first 3c attempt also exposed a Windows-only defect of the F6 reorder (the loaded pair's `.npz`
  stayed open, so the archive could not move it); fixed in `resolve_checkpoint`, after which every clean
  layer was rerun (H2).
- Not changed in mainline (do not edit mainline): mainline's D-278 spent-budget resume reaches a U polish
  without the eager validation; on CUDA its first compiled probe then does the lazy initialisation.
  Clean runs the validation there too.

## Server commands for Dirk (layers 4, 4b and the 6.4 GPU smokes)

All files are in `Thesis-writeup/Code/Verification/clean-code/`; `Clean/` and that folder must be on the
server checkout (`/dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation`). Logs go to
`/home/dirk_van_den_berg/logs/augmentation/clean-code/` (create it once). Nothing below writes into
`Thesis-writeup/Results/`, the server run folders or `~/.deepSI/checkpoints`. Every command runs
from the harness folder (step 2 changes into it; the batch files change into it themselves).

0. (step 4c) Commit and push `Clean/`, the harness folder and mainline `training.py` (D-278), then pull
   on the server checkout. Leave `Clean/` unchanged on the server between the loader and smoke runs and
   their comparisons: the comparators hash the current `Clean/` tree and refuse clean outputs made from
   another one (H2). Run each comparator in the checkout that produced its outputs.
1. Read the `[lbfgs]` lines of 221 (job 89792) and 222 (job 89796) in thesis-results-4 and decide the
   staging case of 6.5: ACCEPTED with val_after below the checkpoint's bestfit, ROLLED BACK or ABORTED
   means a plain copy; ACCEPTED with val_after at or above bestfit (but below the rebuilt score) means
   the row goes in `--middle`.
2. Staging folder (once, CPU, a few minutes per middle row):
   ```
   cd /dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation/Thesis-writeup/Code/Verification/clean-code
   conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject
   python -u l4_stage.py --src ../../../../simulations/gantry_subnet/thesis_taf_lowpass_noise_data_lsq --dst ../../../../simulations/clean-golden/L4-staging
   ```
   (append `--middle 221` or `--middle 221,222` per step 1).
3. Layers 4 and 4b (CPU array: task 0 mainline with the evaluation check, task 1 clean):
   ```
   mkdir -p /home/dirk_van_den_berg/logs/augmentation/clean-code
   sbatch --array=0-1 server_l4.sbatch
   ```
   then, when both tasks are done:
   ```
   python -u compare_l4.py ../../../../simulations/clean-golden/L4/mainline ../../../../simulations/clean-golden/L4/clean
   ```
4. GPU smokes 6.4 (a) R1 and R6, then (b) R6 (GPU array, both codes in one job per row):
   ```
   sbatch --array=0-1 server_gpu_smoke.sbatch
   ```
   then:
   ```
   python -u compare_gpu.py ../../../../simulations/clean-golden/gpu-smoke
   ```
Pass criteria: `COMPARE_L4 PASS` (bitwise per record, axis and split, the val-score acceptance on both
sides and the mainline loader equal to the unmodified evaluation) and `COMPARE_GPU PASS` (compiled branch
ran, no "skipping cudagraphs", initial validation and the four Adam losses within 1e-9 relative, (b)
same deterministic and used_compiled, loss_first within 1e-9 relative). Since step 4c, `COMPARE_L4
PASS` also requires the complete set on both sides (all nine runs, three splits, every layer-4b array
and row, a val-score check per run with a finite positive reference score, the evaluation check over
every run); a subset prints `PARTIAL`, and clean outputs without the current `Clean/` manifest print
`REFUSED` (`COMPARE_GPU REFUSED` likewise). Only PASS counts. Send back the two compare-*.json files
and the loader `summary.json` files.

## Local leftovers for Dirk

- `C:\Users\20203253\AppData\Local\anaconda3\Library\c\` is a stray folder from my first smoke launch
  (a wrong path conversion; it holds only that failed run's config.json and val_hist.npz). Deleting it
  was refused by the permission check; please delete it by hand.
- `simulations/clean-golden/smoke/` (the acceptance run) and `simulations/clean-golden/L4-localtest/`
  (the loader script test on 0.6 s records built from the layer-3 goldens, not a layer-4 result; its
  comparison files are `compare-l4-localtest-{A,B}.json` there; since H2 they are REFUSED) can be
  deleted; the goldens `L1` to `L3c` and `_clean` are the evidence and stay.
- Step 4c: `simulations/clean-golden/_clean-pre4c/` (the step-4 clean outputs and smoke, superseded) and
  `simulations/clean-golden/L3c-preD278/` (the two pre-D-278 polish-only goldens; git shows them as
  deleted under `L3c/`) can be deleted or kept as history; `%TEMP%\clean-harness\L3c\` is the empty
  working folder of deviation 7(f).

## Optimisation (D-281)

2026-10-21. Handoff `tasks/handoffs/2026-10-21-clean-code-optimise-implement.md`; approved list
`optimise-merged.md` (item texts in `optimise-claude.md` and `optimise-codex.md`); rulings D-281 and its
amendment 1 (the implementation choices). Every accepted item is implemented, none reverted, two applied
in part or skipped by a ruling (table below). `Clean/` is now **3,911 lines (was 4,506, -595, -13 %)**.
pyflakes: 0 warnings; all 24 modules import with only `Clean/` on the path. Layers 1, 2, 3, 3b, 3c and the
CLI smoke pass bitwise against the recaptured mainline goldens, with the H2 check active (every clean output
made from the final tree; `Clean/` unchanged since).

**Compiled path unproven until 6.4:** B1 (no `_cur` cache in the compiled forward), B2 (`_propagate` shared by
`forward` and `output_only`) and B26 (no `self.nb` write in the forward) change the graph `torch.compile`
traces. The CPU layers prove them eagerly only; the 6.4 GPU smokes decide, and these three are the first to
revert if a smoke fails (O10). B4, B21 and B27 also change what the compiled forward contains (a module
function instead of an instance attribute, `forward(z, None)`, no unused zero tensors); the same smokes
cover them.

### Mainline goldens recaptured (D-280)

The old goldens moved to `simulations/clean-golden/pre-D280/` (L1, L2, L3, L3b, L3c). All 29 were recaptured
from mainline with the unchanged harness before any edit reached `Clean/`; the edits were staged in a
scratch copy meanwhile. Each new golden was compared with its pre-D-280 one through the comparison functions
of the harness: **all identical**, including 226-lowpass-iv, the spent-budget path D-280 changes (its added
eager validation changes no number on the CPU). The recapture times are the golden column of the result
table below (from each golden's meta.json).

### Items

Every accepted item of `optimise-merged.md` is implemented as stated, except:

| Item | Status | Reason |
|-|-|-|
| C19 | implemented as C19b | O8 takes C19b with its harness edit: `RankReport` is gone, `OBCBasis` keeps `rank`, `tol`, `n_cols` |
| C32 | in part | `n_rows` goes; `n_cols` stays, since after C19b and C25 it is the rank test's reference |
| C35 to C44: `deepSI_corrections.py:13` (drop the `np.copy`) | skipped | inside `fixed_System_data`, which O6 keeps as it is |
| A44 (= Q03) | implemented, `space` constant | `nu` goes; `space` is the constant `'affine'` (the class implements only that space since F13), amendment 1(b) |

Implemented, by file (call-site edits listed with the file they touch):
- `gantry_interconnect_dynamic.py`: A1, A22, A23.
- `gantry_dynamic/config.py`: A2 (= P04, with its reader in data.py), A40.
- `gantry_dynamic/data.py`: A20, A35, A42, Q06.
- `gantry_dynamic/controller.py`: A4, A9, A10, A18, A33, A34, X3, X5, B51.
- `gantry_dynamic/model.py`: A7, A18, A19 (`on_device`, amendment 1(e)), A21 (= P16), A26, A27 (= P26), A36, A37, X4, B14, B40, C23.
- `gantry_dynamic/obc_gantry.py`: A11, A16, A29, A30, A31 (= P28), A32, A44, A45, X4, P29, B1, C7, C8, C9, C24, C25.
- `gantry_dynamic/training.py`: A1, A3, A8, A12, A13, A14, A15 (= P17, P27), A17, A19, A24, A25, A28, A38, A39, A43, C3, C16. `_json_safe` kept (O3).
- `gantry_dynamic/closed_loop_report.py`: A41.
- `fit_systems/interconnect.py`: B2 (= P02), B7 (= P05), B9, B10 and B45 (= P15), B13 (= P06), P07, B14 (= P21, Q07), B15, B16, B17 (= P09), X2, B18, B19, B20, B21, B22, B23, B24, B26, B27, B28, B29, B30, B31, B32, B34, B35 (= P14), B43, B44, B46, B47, B48, B49, B62, C15, P22, Q01, Q02, and its B52 to B71 clarity items (`Interconnect.__init__` without `*args`, `self.obc_correction`, `Signal_Connection` and `Dtype_DataLoader` without defaults, required `interconnect`, `epoch_id[-1]`, one `_epoch_index`, `pop('loss_w')`). Kept: the `optimizer` property (O1), the KeyboardInterrupt handler (O5).
- `fit_systems/blocks.py`: B1 (= P01), B4, B7, B12, B23, B25, B36, B37 (= P13), B38, B39, B40 (= P12), B50 (= P23), and its clarity items (`Block.__init__` without `*args`, `Linear_Output_Block` without defaults, `h` hoisted in `_rk4`).
- `fit_systems/closed_loop.py`: B3 (= P08), B5, B8, B11, B41, B42, B51, X1, X3, X5, and its clarity items (`ctrl_ix` required, `validation_error` without `validation_measure`, no `k0` argument).
- `fit_systems/lbfgs_polish.py`: C1 (= Q04), C2, C3, C4 (= P11), C5, C6, C12, C17, C18, C29, C30, C31, C36 (O7), X1, X5, and its clarity items (`base_stride`, `want`, `_is_oom`, `_validate` before the `try`).
- `fit_systems/obc.py`: C7, C8, C9 (= P10), C14, C15, C19b (= P03), C24, C25, C32 (in part), C33, and its clarity items (`vbar` without the second clone, `evaluate_step_rows`).
- `fit_systems/pre_encoder.py`: C11 (= P19), C26, and its clarity items (`n_in`, `n_out` local, unused defaults).
- `fit_systems/ps2_opening.py`: C16, C20, C21 (= P20), X1, Q05. The `.grad` save and restore kept (O4).
- `utils/torch_nets.py`: C11 (the `@added` helper `mlp_stack`), C23, C27 (= P18).
- `utils/utils.py`: C13 (= P24).
- `utils/deepSI_corrections.py`: C10 (= P25) and its clarity item (import at the top, one `isinstance`). `fixed_System_data` unchanged (O6).
- Not implemented, by ruling: O1 to O6 and O9.

### Line count per file

| File | Before | After | Change |
|-|-|-|-|
| `gantry_interconnect_dynamic.py` | 179 | 175 | -4 |
| `gantry_dynamic/__init__.py` | 15 | 15 | 0 |
| `gantry_dynamic/config.py` | 337 | 317 | -20 |
| `gantry_dynamic/data.py` | 208 | 203 | -5 |
| `gantry_dynamic/controller.py` | 211 | 178 | -33 |
| `gantry_dynamic/model.py` | 244 | 225 | -19 |
| `gantry_dynamic/obc_gantry.py` | 227 | 204 | -23 |
| `gantry_dynamic/training.py` | 448 | 351 | -97 |
| `gantry_dynamic/closed_loop_report.py` | 32 | 31 | -1 |
| `fit_systems/interconnect.py` | 662 | 559 | -103 |
| `fit_systems/blocks.py` | 518 | 440 | -78 |
| `fit_systems/closed_loop.py` | 290 | 236 | -54 |
| `fit_systems/lbfgs_polish.py` | 384 | 315 | -69 |
| `fit_systems/obc.py` | 193 | 137 | -56 |
| `fit_systems/pre_encoder.py` | 84 | 75 | -9 |
| `fit_systems/ps2_opening.py` | 231 | 221 | -10 |
| `systems/gantry_ss.py` | 119 | 119 | 0 |
| `utils/deepSI_corrections.py` | 59 | 48 | -11 |
| `utils/torch_nets.py` | 34 | 37 | +3 |
| `utils/utils.py` | 31 | 25 | -6 |
| four empty `__init__.py` | 0 | 0 | 0 |
| **total** | **4,506** | **3,911** | **-595** |

### Method and gates

The edits were staged in six file groups, each with the call sites it needs (G1 the gantry_dynamic
orchestration, G2 closed_loop, G3 obc, G4 blocks, nets, encoder, utils and norm, G5 interconnect, G6 the
polish and PS2), and every staged diff was read against the item texts. Each group was then applied to
`Clean/`, followed by pyflakes, the import check and layers 1+2 on 221-lowpass (OBC, PS2) and 227-lowpass
(U, n_a = 8): **six gates, all PASS, no item reverted.** Two gates needed a harness-side variant, no `Clean/`
change: before G2 and G3 the clean simulator still had the fields that the O8 edit of `capture_l12.py` no
longer reads, so those gates ran scratch copies of `capture_l12.py` that keep the old reads. The first G3
gate was invalid and is not counted: the capture failed with Errno 22 (OneDrive held the previous gate's
`L1.npz`) after `meta.json` had already been rewritten, so `compare.py` printed PASS on the stale G2 arrays
under the G3 manifest. Rerun with the case folders removed first, it passed.

### Result per layer (final tree)

| Layer | Cases | Result | Runtime (golden / clean, per case) |
|-|-|-|-|
| static | 24 files | pyflakes 0 warnings; all 24 modules import with only `Clean/` on the path | seconds |
| 1, 2 | 16 | PASS 16/16 bitwise, 256 to 299 items per case in layer 1 (3 fewer than before: the O8 fields below) | 36 to 134 s / 30 to 123 s |
| 3 | 5 | PASS 5/5 bitwise | 794, 736, 233, 207, 209 s / 638, 693, 223, 196, 222 s |
| 3b | 2 | PASS 2/2 (forced reject, parameters restored on both sides) | 216, 71 s / 222, 72 s |
| 3c both codes | 6 | PASS 6/6 bitwise; D-278 checks hold on both sides (ii, iv) | 180, 96, 149, 124, 543, 242 s / 181, 93, 130, 83, 596, 296 s |
| 3c clean only | 3 | PASS 3/3: v (F1), vi (F6, F4: both refusals, nothing moved), vii (F4: resumes at 4, ends at 5) | 11, 39, 132 s |
| acceptance `R6 lowpass --cpu --smoke` | 1 | PASS, rc 0: initial validation sim-NRMSe 0.3949, 4 Adam updates, polish ACCEPTED, best 0.359563 (as in steps 4 and 4c); config.json, both pairs, val_hist.npz, `_best`/`_last` dumps written, no `.tmp` left | 272 s |

Not run: layers 4 and 4b and the GPU smokes (server, last); the local loader script test. The evaluation
interface was checked by reading `l4_loader.py`, `l4_stage.py`, `gpu_smoke.py` and the evaluation scripts
against every changed signature (`build_closed_loop(verbose=)`, `build_reference_set(verbose=)`,
`make_basis_builder(space=)`, `closed_loop_free_run_rms_batch(return_pred=)`, `corr.space`, `pass_mats`,
`N_op`/`d_op`/`Y_op`): unchanged.

### Harness edits (O8) and harness findings

- `capture_l12.py` (B5, B41, B51, C19b): on the clean side it reads `sim.record_weights`, `sim.val_rows` and
  `B.tol`, and no longer reads `bank.ystd`, `bank.stdu` or the validation record names; the mainline capture
  is unchanged. `compare.py`: `DROPPED_IN_CLEAN` skips exactly those three golden fields when the clean output
  lacks them and lists them as `dropped` (their content stays covered by `M_state`/`M_error` and the
  validation rows). `run_l3bc.py` (C3): `verbose=True` moves into the mainline-only arguments.
- Found, not fixed (the handoff allows harness edits for O8 only), proposed for Dirk:
  (a) `run_l3bc.py` keeps `z = np.load(...)` open in the D-278 block (kinds ii, iv) and in the vii chain check,
  so a clean case folder cannot leave `%TEMP%` (WinError 5): every clean ii, iv and vii failed its final move
  after the run and its asserts had passed. In step 4c these cases still ran in place, so it never showed.
  These four cases (226-lowpass ii, iv, vii; 221-lowpass-ii) were rerun with a scratch copy that adds
  `z.close()` after the last read in both blocks (two lines, nothing else); the fix belongs in the harness.
  (b) `run_l3bc.py` does not create `_clean/L3c/` before the final move; with a fresh `_clean` the first case
  failed its move (rerun once the folder existed). (c) `capture_l12.py` writes `meta.json` before the arrays,
  so a failed array save leaves old arrays under a new manifest that `compare.py` accepts (the invalid G3
  gate above); writing the arrays first closes that hole.
- Fixed afterwards (harness only, 2026-10-21; py_compile clean, synthetic refusal test 8/8 in a scratch
  folder, all 93 existing golden and `_clean` cases pass the new check): (c) `capture_l12.py:35` removes
  the case's old `meta.json` first and `capture_l12.py:207-211` saves the arrays before `write_meta`;
  `write_meta` (`hcommon.py:230`, `:255`) now records `outputs`, the SHA256 of every file in the case folder,
  and `hcommon.py:204` `output_problems` refuses a listed file that is missing or differs (a `meta.json`
  without that list, the current goldens: L1/L2 arrays older than `meta.json` are refused). Applied in
  `compare.py:150` (REFUSED (outputs)), `compare_l3.py:112`, `compare_l3bc.py:39`, `:76`, `:93`;
  `run_l3.py` and `run_l3bc.py` already wrote `meta.json` last and gain the list through `write_meta`;
  `l4_loader.py:386`, `:407` removes `summary.json` first and lists each run's output hashes, checked by
  `compare_l4.py:107`; `gpu_smoke.py` needs nothing (fresh folder, `result.json` last). (a) `run_l3bc.py:246`
  (vii) and `:263`, `:271`, `:279` (ii, iv) read through `with np.load(...)`, so every handle is closed
  before the move. (b) `run_l3bc.py:297` creates `_clean/L3c/` before the move.

### Server commands (updated)

As in "Server commands for Dirk" above, with step 0 now: commit and push `Clean/`, the harness folder,
mainline `training.py` (D-280) and `docs/decisions.md` (D-281 amendment 1), then pull on the server checkout.
The server runs come last, after the ultrareview and the comment pass; any `Clean/` change before them means
rerunning the local layers first (H2 refuses clean outputs of another tree). If a 6.4 smoke fails, revert B1,
B2 and B26 first (O10).

### Leftovers for Dirk

- The three harness findings above, with their fixes.
- Archives that can be deleted or kept as history: `simulations/clean-golden/pre-D280/` (the pre-D-280
  goldens, identical to the recaptured ones), `_clean-step4c/` (step-4c clean outputs), `smoke-step4c/`.
- For the comment pass, besides the stale-text list of `optimise-claude.md`: the `init_model` docstring
  (no device move or logging arrays any more), the `fit` docstring ("parallel validation"), the
  `linear_encoder_init_aug` dtype comment (names the removed reconstructability map) and the
  `gtd_build_records.m` THEORY comment of `controller.py` (names `TRAIN_FILES`).

## Comments and naming (D-282)

2026-10-22. Handoff `tasks/handoffs/2026-10-21-clean-code-comments-naming.md`; plan `comments-naming-plan.md`;
review `comments-naming-review-codex.md` (wins over the plan, Dirk 2026-10-22); decision D-282; formula check
`comment-check-codex.md`, every finding applied. Only names, comments, docstrings, markers and message texts changed;
the harness proves no number moved. `Clean/` is now 3,905 lines (was 3,911).

### What changed

- Renames (D-282 lists all): `SSE_Interconnect_Composed` -> `AugmentedModel`, `OBCLifecycle` -> `OBCProjection`,
  `linear_encoder_init_aug` -> `AugmentedEncoder`, `live_g_feed_forward_nn` -> `AugmentationNetwork`,
  `ValHistory` -> `ValidationHistory`, the gantry blocks `GantryStateBlock`, `_TrainableGantryStateBlock`,
  `IdentifiableGantryStateBlock`, `WindowedSystemData`, `SystemDataNormalizer`, `FixedWindowBatch`, `DtypeDataLoader`,
  `build_mlp`, the controller and PS2 helpers and the local names of the review. Kept: Jan's names, every name the
  evaluation imports or reads, the harness interfaces, `state_dict` keys, `config.json` keys, CLI flags and every
  string that lands in an output file.
- Markers: five file headers under the module docstring of Jan's files (the 26 trim lines gone), 25 `# CHANGED:`
  in Jan's definitions and import groups, 19 `@added` (seven new, on the methods we added to Jan's classes), 20
  `# THEORY:` and 17 `# HEURISTIC:` labels. The `gantry_ss.py` docstring moved above `__project_origin__`.
- Comments to the standard of the handoff: one-line docstrings on what is called from outside, formulas in the
  thesis symbols with normalised quantities marked `^n`, no D-numbers, job ids, history, section or equation
  references, Kessels named once (the known-controller rollout), every error message says what to do.
- Harness, forced by the class rename (deepSI names its dumps after the class, so clean dumps are now
  `AugmentedModel_<id>_{best,last}.pth`): `run_l3bc.py` (dump names per side), `gpu_smoke.py` (the patched class per
  side), `compare_l3bc.py` (`resumed_from` maps the clean class prefix). Clean dumps written before the rename no
  longer unpickle; every clean output below was regenerated, layer 3 before 3c.

### The Codex comment check: where its text was not used verbatim

Each of F01 to F15, W01 to W10 and S01 to S07 is applied with Codex's text, except these nearest compliant
versions (the standard's one line per public docstring at the code's 120-character width, or a label rule):

| Finding | Applied text or change | Why |
|-|-|-|
| F02 | Codex's two sentences joined into one comment, in the body of `_resample_u` | S02 moves the label out of the docstring |
| F03 | `Dimensionless sim-NRMSe: mean over records and axes of the closed-loop RMS divided by sigma_e.` | one line |
| F05 (dtype) | the message's format arguments drop from four to the two it now uses | message text only |
| F07, S02 `main` | the `resume` and `rerun` argument notes moved to comments beside their use | one-line docstrings |
| F08 | `...; optionally physical y_hat.` | one line |
| F13 | `Build max(1, n_hidden_layers) hidden layers and a final Linear with zero bias; return (Sequential, final layer).` | one line |
| W03 | the heuristic on its own line above `screen_at` | line width |
| W04 | kept on two lines | private helper (Codex S02: allowed) |
| W06 | the first clause on excitation kept, the suffix replaced; the same suffix in the second rank error | consistency |
| S02 `_without_learned` | `..., optionally at another training coordinate.` | one line |
| S02 `PS2Opening.__init__` | `... decoder specification (n_xb physical, n_a added states).` | one line |
| S02, ledger | `ControllerBank.step`, `fourth_order_central_difference`, `calibration_split`, `_resample_u`: one-line docstring, the `# THEORY:`/`# HEURISTIC:` label moved to a comment in the body | one-line docstrings |
| S06 | the two hyphens removed as proposed; the dash check covers em and en dashes and double or triple hyphens used as dashes, not hyphenated words | as Codex notes |

### Static checks (final tree)

| Check | Result |
|-|-|
| syntax tree against the tree before the pass (renames applied; docstrings, comments, `@added`, raise, assert and print messages set aside) | 24/24 equal. Changed non-message strings: the `_dumps` class prefix and the OBC rank-error prefix (read only by its two raises) |
| comment grep, 441 comments and docstrings (D-numbers and dashes also in the code lines) | D-numbers 0 (139 before), job ids 0 (4), section or equation references 0 (18), em or en dashes 0, double or triple hyphens as dashes 0 (1), "Note:", "we", filler, history words 0 (8) |
| pyflakes 3.0.1 (base env), `Clean/` and the harness folder | 0 warnings |
| imports, only `Clean/` on the path | 24 modules, none from outside `Clean/` |

### Result per layer (final tree, every clean output regenerated, H2 active)

| Layer | Cases | Result | Runtime (golden / clean, per case) |
|-|-|-|-|
| 1, 2 | 16 | PASS 16/16 bitwise, 256 to 299 items per case in layer 1 (the three O8 fields dropped as before) | 36 to 134 s / 33 to 147 s |
| 3 | 5 | PASS 5/5 bitwise (221-lowpass, 229-noisefree, 224-noisefree, 226-lowpass, 227-lowpass) | 794, 736, 233, 207, 209 s / 738, 628, 219, 236, 225 s |
| 3b | 2 | PASS 2/2 (forced reject, parameters restored on both sides) | 216, 71 s / 232, 68 s |
| 3c both codes | 6 | PASS 6/6 bitwise; D-278 checks hold on both sides (ii, iv); `resumed_from` compared through the class-prefix map | 180, 96, 149, 124, 543, 242 s / 181, 94, 139, 84, 593, 309 s |
| 3c clean only | 3 | PASS 3/3: v (fresh run refused, folder unchanged), vi (a truncated `_best.pth` and a dump 3 updates into S1 refused, nothing moved; the dump is `AugmentedModel_L3-224-noisefree_best.pth`), vii (resumes at 4, ends at 5) | 9, 30, 122 s |
| acceptance `R6 lowpass --cpu --smoke` | 1 | PASS, rc 0: initial validation sim-NRMSe 0.3949, 4 Adam updates, polish ACCEPTED, best 0.359563 (as in steps 4, 4c and D-281); config.json, both pairs, val_hist.npz and the `AugmentedModel_<id>_best/_last` dumps written, no `.tmp` left | 290 s |

The goldens are the D-280 mainline goldens, unchanged; mainline was not rerun. All clean outputs were made in one
detached sequence from the final tree, which has not changed since (`compare.py` checked the H2 manifest of every
case). Not run: layers 4 and 4b and the GPU smokes (server, last).

### Server commands (updated)

Every command runs from the harness folder on the server checkout
(`/dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation`); logs go to
`/home/dirk_van_den_berg/logs/augmentation/clean-code/`. Nothing writes into `Thesis-writeup/Results/`, the server
run folders or `~/.deepSI/checkpoints`.
0. Commit and push `Clean/`, the harness folder (with the three D-282 edits) and `docs/decisions.md` (D-282), then
   pull on the server checkout. Leave `Clean/` unchanged on the server until the comparisons below have run: they
   hash the current tree and refuse clean outputs of another one (H2).
1. Read the `[lbfgs]` lines of 221 (job 89792) and 222 (job 89796) in thesis-results-4: ACCEPTED with val_after
   below the checkpoint's bestfit, ROLLED BACK or ABORTED means a plain copy; ACCEPTED with val_after at or above
   bestfit means that row goes in `--middle`.
2. Staging folder (once, CPU):
   ```
   cd /dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation/Thesis-writeup/Code/Verification/clean-code
   conda activate /dataB1/dirk_van_den_berg/conda-envs/GraduationProject
   python -u l4_stage.py --src ../../../../simulations/gantry_subnet/thesis_taf_lowpass_noise_data_lsq --dst ../../../../simulations/clean-golden/L4-staging
   ```
   (append `--middle 221` or `--middle 221,222` per step 1).
3. Layers 4 and 4b (CPU array: task 0 mainline with the evaluation check, task 1 clean):
   ```
   mkdir -p /home/dirk_van_den_berg/logs/augmentation/clean-code
   sbatch --array=0-1 server_l4.sbatch
   ```
   then, when both tasks are done:
   ```
   python -u compare_l4.py ../../../../simulations/clean-golden/L4/mainline ../../../../simulations/clean-golden/L4/clean
   ```
4. GPU smokes 6.4 (a) R1 and R6, then (b) R6:
   ```
   sbatch --array=0-1 server_gpu_smoke.sbatch
   ```
   then:
   ```
   python -u compare_gpu.py ../../../../simulations/clean-golden/gpu-smoke
   ```
Only `COMPARE_L4 PASS` and `COMPARE_GPU PASS` count (PARTIAL and REFUSED do not). Send back the two compare-*.json
files and the loader `summary.json` files. The class rename changes none of these steps: layers 4 and 4b stage and
load component pairs (state_dicts), and `gpu_smoke.py` patches the model class of its own side. If a 6.4 smoke
fails, revert B1, B2 and B26 first (O10).

### Leftovers for Dirk

- Archives that can be deleted or kept as history: `simulations/clean-golden/_clean-D281/` and `smoke-D281/` (the
  clean outputs and smoke of the optimisation step, superseded; their dumps no longer unpickle under the new class
  names).
- Codex F01: the same exactness claim about block averaging sits in `99_appendix.tex` after `eq:setup_decim`; the
  thesis was not edited.
- The static checkers of this pass (the syntax-tree comparison against the pre-pass tree and the comment grep) and
  the detached phase-3 batch are in the session scratchpad only; they can go into the harness folder if wanted.
