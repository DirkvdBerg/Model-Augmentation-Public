# Implementation report: clean minimal code (step 4)

2026-10-20. Handoff `tasks/handoffs/2026-10-20-clean-code-implementation.md`; design `tasks/clean-code/design.md`
(revised) with D-274, D-275 and its amendments 1 to 4. All local layers pass bitwise on CPU; layers 4,
4b and the 6.4 GPU smokes run on the server (commands at the end).

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
| 4, 4b | server | pending; scripts ready, see "Server commands". Local script test of the loader: LOADER_RESULT | |
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
| `l4_stage.py`, `l4_loader.py`, `compare_l4.py`, `server_l4.sbatch` | layers 4 and 4b (server) |
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

## Server commands for Dirk (layers 4, 4b and the 6.4 GPU smokes)

All files are in `Thesis-writeup/Code/Verification/clean-code/`; `Clean/` and that folder must be on the
server checkout (`/dataB1/dirk_van_den_berg/repos/LPV-LFR-Baseline-Augmentation`). Logs go to
`/home/dirk_van_den_berg/logs/augmentation/clean-code/` (create it once). Nothing below writes into
`Thesis-writeup/Results/`, the server run folders or `~/.deepSI/checkpoints`.

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
   sbatch --array=0-1 Thesis-writeup/Code/Verification/clean-code/server_l4.sbatch
   ```
   then, when both tasks are done:
   ```
   python -u compare_l4.py ../../../../simulations/clean-golden/L4/mainline ../../../../simulations/clean-golden/L4/clean
   ```
4. GPU smokes 6.4 (a) R1 and R6, then (b) R6 (GPU array, both codes in one job per row):
   ```
   sbatch --array=0-1 Thesis-writeup/Code/Verification/clean-code/server_gpu_smoke.sbatch
   ```
   then:
   ```
   python -u compare_gpu.py ../../../../simulations/clean-golden/gpu-smoke
   ```
Pass criteria: `COMPARE_L4 PASS` (bitwise per record, axis and split, the val-score acceptance on both
sides and the mainline loader equal to the unmodified evaluation) and `COMPARE_GPU PASS` (compiled branch
ran, no "skipping cudagraphs", initial validation and the four Adam losses within 1e-9 relative, (b)
same deterministic and used_compiled, loss_first within 1e-9 relative). Send back the two
compare-*.json files and the loader `summary.json` files.

## Local leftovers for Dirk

- `C:\Users\20203253\AppData\Local\anaconda3\Library\c\` is a stray folder from my first smoke launch
  (a wrong path conversion; it holds only that failed run's config.json and val_hist.npz). Deleting it
  was refused by the permission check; please delete it by hand.
- `simulations/clean-golden/smoke/` (the acceptance run) and `simulations/clean-golden/L4-localtest/`
  (the loader script test on 0.6 s records built from the layer-3 goldens, not a layer-4 result) can be
  deleted; the goldens `L1` to `L3c` and `_clean` are the evidence and stay.
