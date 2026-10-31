# Handoff: implement D-284 amendment 10 (residue of `Clean/`), then prepare the one server proof
**From**: session of 2026-10-31 | **Branch**: Augmentation | **Effort suggested**: high (wide but specified edits
across `Clean/`, the harness and `clean-up-verification/`; every change is bitwise by ruling)

## 1. Task
Dirk: "i see so many trash variables", "if the code needs these trash variables that points to a bigger issue".
Implement every ruled item of D-284 amendment 10 (`docs/decisions.md`, points 1 to 4) on the 24 grey-box files of
`Clean/`, in the five groups of section 9, with the harness readers and `clean-up-verification/` adapted; prove each
group with the quick checks of section 10 only; then update what the step-5 server job needs (section 9, last step)
so that one server run proves the final tree. The yardstick is Dirk's definition of clean code (amendment 10 point
1): production code for ASMPT only; every file, function, setting and key has a reason a user accepts; an option
stays only if it selects a working alternative a user of our method could choose; honest names, comments and
labelled numbers; no repeated computation.

## 2. Out of scope
- Long local runs: no layer-3, layer-3c, full suite or coverage run locally (Dirk 2026-10-31: "i dont want these
  fucking long local runs anymore"; memory `feedback_no_long_local_gates`). They go into the step-5 server job.
- Submitting server jobs (Dirk submits). Git write operations.
- Items ruled "no" or "keep": R18 (the MM16 line `P_inv_T = ...` in `compute_normalization` stays exactly as the
  step-5 `sed` expects it), R21 (Jan's code 1:1), R22 (`__project_origin__` markers stay), K2, K20, and every K or R
  not listed in section 9. Working alternatives stay (amendment 10 point 1): `use_gain`, `up_sample`, `append_value`,
  `baseline_form`, the float32 branches, Jan's `its_per_val='epoch'`.
- New ideas: a proposal to Dirk, never code.
- `Clean/blackbox/` (D-285), `Clean/data_generation/` (D-286), `Thesis-writeup/Code/Evaluation/` and `Figures/`.

## 3. Where things stand
Tree dirty, nothing committed (Dirk commits). `Clean/` = snapshot `final` (manifest `98d23ae2...1f99`) + the rename
"logical" to "generalised" + K3, K8, K13, K14, K15, K23, K24, K26, K31, K41, K42 + E1 to E4 of amendment 9 (new
`gantry_dynamic/evaluation.py`, `closed_loop_report.py` removed) + the 7 Codex fixes and the K14 equation-test fix
(`tasks/clean-code/keep-audit-merged.md` section 6). Snapshots: `simulations/clean-golden/src-snapshots/after-K/`
(manifest `408f1273...7ebd`) and `after-K-harness/`. `data.py` SHA256 with MM16 in final form starts `2b3109f2`.
`clean-up-verification/` holds `loader_check.py`, `smoke.py`, `val_check.py`, `r1_check.py`. Nothing running.
Step 7 of `tasks/handoffs/2026-10-30-clean-eval.md` (scoreboard after, goal check, full suite, new `final`
snapshot, hash updates, report section) is not done; its long parts move to the server job.

## 4. Established and verified
- Every amendment-9 group passed L1-2 (221, 227 lowpass) bitwise with MM16 reverted; groups touching training also
  layer 3 and 3c on 221 (logs `simulations/clean-golden/suite-logs/gate-4b-*.log`). After the Codex fixes: ruff F
  clean, `check_imports.py` 24 modules from `Clean/` only, `equation_tests.py` 72 of 72, `loader_check.py` PASS.
  The Codex fixes themselves ran in no gate (they touch L4, perf_diag and comments only).
- `smoke.py` R6 and R1 lowpass: training then `evaluate()`; `val_check.py` exact on both (0.3846019211713487,
  0.38493066908518037, the accepted polish candidate). `evaluate()` 250 s (R6) and 508 s (R1) on 2 s records.
- deepSI stores the optimizer as a plain `self.optimizer` (deepSI `fit_systems/fit_system.py:156`); the
  `AugmentedModel.optimizer` property's comment claiming `_optimizer` is false (R15).
- Thesis run folders were written by mainline; their state_dict keys equal Clean's today (static reading in
  `residue-audit-claude.md` "Rulings and handoff points"; not confirmed by a load).
- Harness readers of config.json (from the Claude residue audit): `l4_loader.discover` (OBC, NX_ANN,
  COMBO_INIT_DETUNE, PS2, run_id of staged thesis folders), `compare_l3` (common keys except GIT, run_id, data_dir,
  NF_PROBE_PRINT), `run_l3bc` (RESUMES), `from_config_json` via `val_check`, `gpu_smoke`, `l4_loader`, `run_l3bc`.

## 5. Assumed but not verified
- That a thesis final checkpoint loads into Clean after the converter drops the R3 keys; the converter's own test
  settles it (section 10).
- That the step-5 harness (`l4_stage.py`, `l4_loader.py`, `server_l4.sbatch`, `gpu_smoke.py`, `timing_driver.py`)
  needs only the converter call and the dropped-key lists; read each before editing.
- That `compare_l3`/`compare_l3bc` refuse rather than skip a key missing on the clean side; read `compare_l3.py`
  before deciding whether R3 and R4 keys go into a dropped set there.

## 6. Tried and failed
- Per-group local gates with layer 3 and 3c (about 25 min each, seven times) -> days of local compute for mostly
  obviously bitwise edits -> Dirk stopped it -> section 2.
- Editing a gate shell script while bash ran it -> bash resumed at a shifted offset and ran steps twice -> never edit
  a running script; copy it first.
- Overwriting a 28 MB `.npz` in `simulations/clean-golden/_clean/` right after it was written -> `OSError [Errno 22]`
  (OneDrive holds the file while syncing) -> delete the old output first.
- `conda run -n GraduationProject python -c "<multi-line>"` -> conda refuses newlines -> write a script file.
- Keep audit with the fixed interface rules -> both audits missed all config/checkpoint residue -> amendment 10.

## 7. Achieved
Rulings complete (amendment 10). Audits: `tasks/clean-code/residue-audit-claude.md`, `residue-audit-codex.md`,
merged `residue-audit-merged.md` (R1 to R22). Nothing of amendment 10 is implemented.

## 8. The open question
Nothing blocked.

## 9. Next action
Implement in this order, one group at a time, each with the checks of section 10, items carry their R ID in
harness edits and `DROPPED_IN_CLEAN`:
1. **Converter and config.json** (R1, R9, R15 part): `clean-up-verification/convert_run.py` turns a thesis (mainline)
   run folder into a Clean folder: config.json in Clean's schema, the R3 state_dict keys dropped from the checkpoint
   pair (and the whole dumps, if a step needs them). `config_json_dict` writes only Clean's own settings (merged R1);
   `from_config_json` keeps the Clean branch, checks only real settings, refuses `CODE != 'clean'` with "convert it
   with clean-up-verification/convert_run.py"; `_CLEAN_KEYS`, the skip list and the PS2_SMOKE note go; `main` stops
   writing `hp` into config.json. Harness: L4 staging and loader through the converter.
2. **Settings and parameters** (R2, R5, R6, R13, R16 part, R17): the ClassVars of R2 (with `mode`, the run family
   folder named after `noise`, `route_ix` K32, `joint_estimation` K7), `validation_measure`, `val_data`, the invalid
   defaults, the duplicate sources of R17.
3. **Checkpoint state** (R3): the 13 nominal buffers of `GantryStateBlock` (keep `P_mat`), the parameter-prior chain
   (`param_rmse_baseline`, `RMSE_baseline`, `Lambda_combo`), then `gantry_ss` M1, M2, C, K; capture and comparators
   get the dropped keys.
4. **Run outputs** (R4, R10, R11, R12, R15, R16, R19): pair keys of R4 and R10 go, R11 stay; `val_hist_sim_rms` and
   `last_sim_rms` go; `loss.npz` keeps only training loss with update, epoch, time and `evaluation._loss_figure` reads
   the validation rows, selected update and polish candidate from `val_hist.npz`; the `optimizer` property and
   `_optimizer` go (resume reads `optimizer`); `max_iter` leaves `lbfgs_outcome`; the fourth return of
   `estimation_parameters` goes; `allow_pickle=False` for the pair reader.
5. **Text and habits** (R7, R8, R14, R20): comments as in merged R7 (including the tuple range of
   `build_reference_set`), labels of R8 with one labelled constant for the chunk size 16384, the `**kw` comment, the
   small habits of R20.
6. **Prepare the server proof**: new `final` snapshot (method in `06-audit-claude.md` item 6; old one archived as
   `final-step4`), hashes updated where cited (`timing_driver.py`, implementation report "Server commands (D-284 step
   5)": manifest, MM16 line and reverted `data.py` hash, problem log section 12 step-5 row), and the step-5 commands
   extended with the local layers (`run_clean_suite.sh`, which now ends with `clean-up-verification/smoke.py`), the
   scoreboard `coverage` with its new case `smoke-R1-lowpass`, and `r1_check.py` (reverted, then restored). Write
   the report section "Keep audit (D-284)" and the residue section with the static scoreboard (`scoreboard.py
   static` and `report`, minutes) and the goal check of `06-audit-claude.md` item 6 under amendment 10's yardstick.

## 10. Acceptance criterion
Per group: `py_compile`, `ruff check --select F` on `Clean/` and the touched harness files, `check_imports.py`,
`equation_tests.py` 72 of 72 (or each removed test names its R ID). After group 1 and 3: `convert_run.py` on one
thesis folder (the staged R1, job 89792, or a copy) and `load_checkpoint` of its final pair into a Clean model
without error. At the end: one `smoke.py` run (R1 lowpass) and `val_check.py` exact; `grep -rn -i "smoke\|harness"`
on the grey box finds nothing but the accepted `CODE` refusal message if it names the converter; the merged residue
audit's R1 to R20 (except R18, R21) implemented as ruled, each with its R ID in the report.

## 11. Read these first
1. `docs/decisions.md` D-284 amendment 10 (and amendment 9 for the evaluation it revises).
2. `tasks/clean-code/residue-audit-merged.md`: the R items with both audits' text.
3. `tasks/clean-code/residue-audit-claude.md` rows of the files you edit (exact `file:line` and change per item).
4. `tasks/clean-code/implementation-report.md` section "Server commands (D-284 step 5)".
5. `Thesis-writeup/Code/Verification/clean-code/hcommon.py`, `compare.py`, `l4_loader.py`: how the harness reads
   clean outputs.

## 12. Do not
Run long local jobs (section 2). Change the MM16 line. Edit Jan's code beyond what a ruled R item names. Rename or
move Jan's classes, functions or files. Edit a running shell script. Record a ruling Dirk did not give.

## 13. Operational
`conda run -n GraduationProject python ...` for quick checks; multi-line snippets as script files. Harness folder
`Thesis-writeup/Code/Verification/clean-code/`. Smoke: `python -u clean-up-verification/smoke.py R1 lowpass`
(about 15 min, output under `simulations/gantry_clean/`), then `python -u clean-up-verification/val_check.py
<SMOKE_RUN_DIR>`. Final message to Dirk: groups done with check results, the new `final` manifest, the server
commands to submit, and the six renamed `.m` files of `Clean/data_generation/` for the D-286 session to check
(`gantrySystemExtendedTanh.m`, `gtd_make_multisine.m`, `gtd_make_reference.m`, `gtd_save_record.m`,
`gtd_validate_ref.m`, `tdg_config.m`).

## 14. Delegation
None for the implementation; at most one Explore subagent for reader searches outside `Clean/`.
