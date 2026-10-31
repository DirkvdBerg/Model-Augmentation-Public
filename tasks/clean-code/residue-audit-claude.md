# Residue audit of `Clean/` (Claude)

2026-10-31, by `tasks/handoffs/2026-10-31-residue-audit.md` sections 1 to 14, on the dirty tree of that day (line numbers
are of that tree). Scope: the 24 grey-box files; the three empty `__init__.py` (`fit_systems/`, `systems/`, `utils/`)
hold no item. Method: my own AST pass over the 24 files for module constants, class attributes, parameters and
defaults, `register_buffer` calls and `self.<attr>` assignments; `config_json_dict`, the checkpoint writers and the
branches by reading; readers by search in `Clean/`, `clean-up-verification/` and
`Thesis-writeup/Code/Verification/clean-code/`, plus deepSI and mainline where a row says so; ruff (ARG) and vulture.
Nothing was run beyond these static tools. "bitwise" in a row is reasoned from the readers, not tested.
Row format: `file:line | item [kind] | categories | reader | verdict | change; label; what else it touches`.

## Summary

### Counts
- **Rows**: 1150 (A config 176, B checkpoints and logs 85, C hp 9, D other items 324, E parameters 556); 129 in
  Jan's code, of which 10 are non-keep.
- **Per category** (a row can carry several): compute 912, write-only 115, mainline-compat 89, unlabelled 74,
  fixed-option 59, misleading 30, verification 18.
- **Per verdict**: keep 939, remove 120, label 54, comment 15, literal 11, rename 6, move 5 (211 non-keep).
- **config.json**: `config_json_dict` writes 81 keys (not about 80); 33 stay, 48 go (44 removed, 4 moved to the
  converter); of `main`'s 4 extra keys `hp` goes, `run_id`, `data_dir`, `RESUMES` stay.
- **Labels**: every proposed change is `bitwise`; none is a `rounding` item. The one rounding candidate found
  (float64 nominal constants in `gantry_ss.py`, row D16) is a keep.

### Dirk's question: does the trash point to a bigger issue
Yes. Almost every non-keep row traces to one of two requirements that are not ASMPT's: `Clean/` must read the thesis
run folders written by mainline, and it must write files the harness can compare key by key with mainline's goldens
(the interface rules of overview section 6 made config.json, the checkpoint keys and the CLI fixed). The first pulls
mainline's schema into production (the mainline branch of `from_config_json`, MODE, OBC, NX_ANN, CODE, the
`_optimizer` property, Lambda_combo and the nominal buffers that thesis checkpoints carry). The second keeps
literals of options `Clean/` does not have, so a check of literal against literal passes. The structural fix is one
converter in `clean-up-verification/` that turns a thesis folder into a Clean folder (config.json in Clean's schema,
the unread state_dict keys dropped); the harness and `r1_check.py` load through it, and `Clean/` owns its formats.

### Proposals, ranked by how much they clean
1. **P1, config.json owned by Clean/** (A1, A2, A4): `config_json_dict` writes the 33 keys of rules (a), (b), (f)
   in A4; `main` stops writing `hp`; `from_config_json` keeps the Clean branch and compares only rule-(b) keys (a
   real guard against a folder written with other settings); the mainline branch and its check move to
   `clean-up-verification/thesis_folder.py`; `_CLEAN_KEYS`, the skip extras, CODE and the PS2_SMOKE note go. About
   60 lines and 49 keys. Cost: thesis folders load only through the converter (r1_check, L4 clean side,
   gpu_smoke on mainline folders); existing Clean folders still load; compare_l3 compares common keys only.
2. **P3, unread checkpoint keys** (B1, D16): the 13 nominal buffers of `GantryStateBlock` and their build
   (blocks.py:150-175 except P_mat), `Lambda_combo`, `RMSE_baseline`, `_RMSE_baseline`, `param_rmse_baseline`,
   and then `gantry_ss` M1, M2, C, K and the unused returns of `build_G_matrix_entries`. 14 state_dict keys and
   about 30 lines. Cost: every existing checkpoint (thesis and Clean) needs the converter to drop the 14 keys;
   compare_l3/compare_l3bc need a dropped set. Decide P3 and P4 before the step-5 server runs, or those
   checkpoints need converting too.
3. **P2, write-only ClassVars** (A3, with K7, K29, K32): param_prior, use_f64, joint_estimation and the
   `joint_estimation` parameter of `load_checkpoint` (K7), lbfgs, encoder_init with the `mode` property and the
   folder name `<noise>/<run_id>`, controller_gain, obc_rank_rtol, obc_ref_successor; obc_space and obc_refresh
   as literals in the one print; ann_route_ix with the `route_ix` parameters (K32). 12 ClassVars and properties,
   3 parameters. Cost: harness callers drop two keywords; new runs land in a renamed folder.
4. **P4, residue in the pair, the dumps and the logs** (B2, B3, D8, D10, D11, D12): .npz keys hp, orig_run_id,
   bestfit_selected, elapsed, optimizer_state_dropped, start_phase, resumed_from; lbfgs_outcome max_iter;
   val_hist_sim_rms with `last_sim_rms`; the `optimizer` property and `_optimizer` (mainline's backing field) with
   `d.get('optimizer') or d.get('_optimizer')` and the hasattr guard of `resolve_checkpoint`; the false print
   clause about whole dumps; the fourth return value of `estimation_parameters`. 18 rows.
5. **P5, defaults that are wrong or invalid** (E, D9): `pass_mats=None` (three), `mats=None`,
   `params_init=None`, `combo_init=None`, `dtype=torch.float32` (two), `validation_measure='sim-NRMS'` (two), the
   duplicate `chunk=16384`, the float32 branch of `np_dtype`. 12 rows.
6. **P6, history in comments and messages** (A, B2, D): the THEORY line on sigma_v in data.py (a mainline SNR
   remnant), the tuple range of the successor in `build_reference_set`, the six CHANGED headers that point to "the
   repository root", "run this folder with the mainline code", "the mainline implementation", the `_optimizer`
   comment, "kept for callers", the `g_aug=mlp/live` print, the "mainline one" of gantry_dynamic/__init__.py,
   four config.py comments. 18 rows.
7. **P8, duplicate sources and names** (C, D): hp key NX_ANN to n_a, `nx_ann` to `n_a`, `P_inv_T` in data.py,
   `fs` from dt, raw_init[10..12], the S2 plateau window 5, the dtype print slice. 7 rows.
8. **P7, labels** (54 rows): 17 RunConfig numbers, the 15 gantry_ss constants (one THEORY line), the tolerances
   (1e-8 three times, 1e-9, 1e-12, 1e-6 twice, 1e-300), the two retry numbers, seven PS2Spec fields, the class
   size 3 and the two seed offsets, `max_consecutive_rank_loss`, `%.17g`. Comments only, no code change.

### Patterns, with the items behind each
- **Mainline compatibility in production code**: from_config_json mainline branch; MODE, OBC, NX_ANN,
  COMBO_INIT_DETUNE, CODE; `_CLEAN_KEYS` and the NF_PROBE_PRINT skip; `mode`; the `<mode>_<encoder_init>` folder;
  the `_optimizer` property and `d.get('optimizer')`; Lambda_combo and the nominal buffers; `coordinate_version`
  (kept as a guard); hp key NX_ANN and `nx_ann`; `last_sim_rms`; the fourth value of `estimation_parameters`;
  RESUMES `code`.
- **config.json as a copy of mainline's schema, with literals not tied to the code**: the 39 keys of rule (d), for example ANN_ACTIVATION = 'tanh' while model.py:71 sets Tanh on its own, so the record cannot follow a
  change.
- **Write-only state**: 12 RunConfig items, 14 buffers, 7 .npz keys, `last_sim_rms`, `output_signal_sizes`
  (Jan's), lbfgs_outcome max_iter, `estimation_parameters`' label.
- **Options frozen at one value**: routing (`ann_route_ix`, `route_ix`, K32), `controller_gain` not wired to
  `use_gain`, obc_space, obc_refresh, the float32 `np_dtype` branch, `append_value=False` (K18, no caller passes
  it), Jan's `its_per_val == 'epoch'` and `train_loss >= 0` branches.
- **Defaults that disagree with the only used value, or cannot work**: P5.
- **History in comments and messages**: P6.
- **Unlabelled numbers**: P7.
- **Duplicate sources of truth**: hp versus the config.json keys, NX_ANN / N_A / n_a, OBC / ARM, MODE / NOISE,
  DATA_ROOT / data_dir, two float64 P^-T, fs from dt versus fs_new, obc_ref_chunk versus the chunk default,
  orig_run_id versus run_id, resumed_from versus RESUMES, bestfit_selected versus the val_hist 'selected' row.

### Rulings and handoff points
- **Section 6 settled**: (1) yes, `--resume` reads Clean's own deepSI dumps (`_rerun_source` -> `_dumps` ->
  `resolve_checkpoint` 'whole'), so the whole format stays; only its `optimizer` key, the hasattr guard and one
  print clause are residue. (2) Statically yes: mainline registers the same hfn buffers (mainline blocks.py:739-777,
  `Y_op=None` in mainline model.py:88 so no N_op/d_op, out_gate not persistent in both), so the converter needs only
  config.json today and the 14-key drop after P3; not confirmed by a load here. (3) Harness readers of config.json:
  l4_loader `discover` (OBC, NX_ANN, COMBO_INIT_DETUNE, PS2, run_id of staged thesis folders), compare_l3 (every
  common key except GIT, run_id, data_dir, NF_PROBE_PRINT), run_l3bc (RESUMES), from_config_json through
  val_check, gpu_smoke, l4_loader, run_l3bc; hcommon.write_meta stores config_json_dict for information only.
- **Unruled K items re-raised**: K7 remove (E); K11 keep, settled by the evaluation's twin loading (E); K29
  subsumed by P2; K32 remove (E).
- **Ruled items with new evidence**: K6 `val_data` remove (it is no option; "kept for callers" is the excluded
  interface reason); K18 and K38 keep, with a note (no caller passes `append_value=False`; the `nf` default 25 is no
  Clean setting).
- **Jan's code** (handoff section 8, my reading, to confirm): residue is listed, marked `Jan's code`, with a verdict;
  10 rows are non-keep (six CHANGED headers, `output_signal_sizes`, two `None` defaults, the fit default name).
  Branches of Jan's that are dead in Clean/ carry keep, "Dirk decides".

## A. `gantry_dynamic/config.py`: constants, fields, keys

### A1. Module constants and the key lists
- Clean/gantry_dynamic/config.py:14 | ROOT [module constant] | compute | reader: config.py:15 | keep | names the Clean folder that REPO_ROOT is derived from; no change
- Clean/gantry_dynamic/config.py:15 | REPO_ROOT [module constant] | compute | reader: config.py:16, :17, :222 (git cwd) | keep | read by git_provenance and the two default roots; no change
- Clean/gantry_dynamic/config.py:16 | DATA_ROOT = Thesis-writeup/Data [module constant] | compute | reader: gantry_interconnect_dynamic.py:160, evaluation.py:218 | keep | a user sets it (--data-root); the default names this repository's layout, which a standalone copy of Clean/ lacks: say so in Clean/README.md; bitwise; nothing else
- Clean/gantry_dynamic/config.py:17 | OUT_ROOT = simulations/gantry_clean [module constant] | compute | reader: gantry_interconnect_dynamic.py:161 | keep | a user sets it (--out-root); same README note as DATA_ROOT
- Clean/gantry_dynamic/config.py:22 | AA_ORDER = 8, AA_FC_HZ = 1600.0 [module constants] | compute | reader: data.py:27 (SOS), config.py:54 | keep | labelled (HEURISTIC, THEORY), read by the filter design; no change
- Clean/gantry_dynamic/config.py:25 | AA_TRIM_S = 0.005 [module constant] | compute | reader: config.py:60 | keep | labelled HEURISTIC, read by record_trim; no change
- Clean/gantry_dynamic/config.py:29 | PRESETS (R1 to R7, R13, R14) [module constant] | compute | reader: gantry_interconnect_dynamic.py:146, :172 | keep | a user selects a run with it; the values are the run design (comment :27), not tuned numbers; no change
- Clean/gantry_dynamic/config.py:42 | _ENV_KEYS [module constant] | compute | reader: config.py:155, :157 | keep | maps the config.json keys read back into RunConfig; survives P1 unchanged
- Clean/gantry_dynamic/config.py:47 | _CLEAN_KEYS [module constant] | mainline-compat | reader: config.py:157 (skip set only) | remove | exists only to skip Clean-only keys in the equality check against config_json_dict, a check that matters only for mainline folders (P1); bitwise; config.json unchanged by this item alone; part of P1
- Clean/gantry_dynamic/config.py:46 | comment "PS2_SMOKE, written by earlier versions, is read and ignored" [history comment] | mainline-compat, verification | reader: none | remove | describes smoke folders of a removed feature (E2); goes with _CLEAN_KEYS (P1); bitwise
- Clean/gantry_dynamic/config.py:47 | 'PS2_SMOKE' [_CLEAN_KEYS entry] | verification, mainline-compat | reader: config.py:157 (skipped) | remove | only smoke folders written before E2 carry it; under P1 unknown keys are ignored anyway; bitwise; those smoke folders still load
- Clean/gantry_dynamic/config.py:47 | 'ARM', 'START', 'N_A', 'NOISE' [_CLEAN_KEYS entries] | compute | reader: config.py:147 (Clean branch reads them) | keep | read back as the axes of a Clean folder; the skip entry itself goes with P1
- Clean/gantry_dynamic/config.py:47 | 'BASELINE_FORM' [_CLEAN_KEYS entry] | compute | reader: config.py:155 (via _ENV_KEYS) | remove | duplicate: the key is already skipped through _ENV_KEYS; bitwise
- Clean/gantry_dynamic/config.py:47 | 'DATA_ROOT', 'OUT_ROOT', 'LAST_SAVE', 'CODE' [_CLEAN_KEYS entries] | write-only, mainline-compat | reader: config.py:157 (skipped) | remove | the keys themselves go under P1 (rows A4); bitwise
- Clean/gantry_dynamic/config.py:157 | 'GIT' [equality skip] | compute | reader: config.py:157 | keep | provenance differs per run by design; survives P1 as "not compared"
- Clean/gantry_dynamic/config.py:157 | 'NF_PROBE_PRINT' [equality skip] | mainline-compat | reader: config.py:157 | remove | skipped because thesis folders hold True and Clean writes False; the key is a mainline print switch (P1); bitwise
- Clean/gantry_dynamic/config.py:157 | 'START_PHASE' [equality skip] | mainline-compat | reader: config.py:157 | remove | the key goes (row A4, always 'adam' when written); bitwise

### A2. `from_config_json` branches and the check
- Clean/gantry_dynamic/config.py:146 | `meta.get('CODE') == 'clean'` branch [branch] | compute | reader: config.py:147 | keep | the Clean reading of a folder: ARM, START, N_A, PS2, NOISE; under P1 it is the only branch, and a folder without ARM raises "not a Clean run folder; convert it with clean-up-verification/thesis_folder.py"; bitwise
- Clean/gantry_dynamic/config.py:149 | mainline branch (OBC, COMBO_INIT_DETUNE, NX_ANN, MODE mapping) [branch] | mainline-compat, verification | reader: evaluate on thesis folders (clean-up-verification/r1_check.py:36), l4_loader.py:99 clean side, gpu_smoke.py:111 | move | to clean-up-verification/thesis_folder.py: reads a thesis folder, checks every mainline key against the thesis values (the current check), writes a Clean config.json into a staging copy; bitwise; r1_check and the L4 clean side load the staged copy
- Clean/gantry_dynamic/config.py:156 | equality check of every non-skipped key against config_json_dict [check] | mainline-compat, verification | reader: config.py:158 | rename | keep it as the guard of Clean folders, but only over the keys of rule (b) in A4 (settings Clean computes with), so it refuses a folder written with other settings and no longer compares literals with literals; bitwise; its message drops "run this folder with the mainline code" (next row)
- Clean/gantry_dynamic/config.py:161 | message "...outside the thesis runs ...; run this folder with the mainline code" [message] | misleading, mainline-compat | reader: user | comment | "config.json was written with other settings (key: (this code, file)): %s; use the code of its GIT entry"; bitwise
- Clean/gantry_dynamic/config.py:163 | `replace(cfg, **overrides)` (roots, start_phase, device from callers) [option] | compute | reader: evaluation.py:236, gantry_interconnect_dynamic.py:180 | keep | callers set roots and device; no change

### A3. RunConfig fields, ClassVars and properties
- Clean/gantry_dynamic/config.py:66 | arm [field] | compute | reader: config.py:168 (obc) | keep | an axis a user selects; no change
- Clean/gantry_dynamic/config.py:67 | start [field] | compute | reader: config.py:190 | keep | an axis; no change
- Clean/gantry_dynamic/config.py:68 | n_a [field] | compute | reader: config.py:173, model.py:175, training.py:244 | keep | an axis; no change
- Clean/gantry_dynamic/config.py:69 | ps2 [field] | compute | reader: training.py:241 | keep | an axis; no change
- Clean/gantry_dynamic/config.py:70 | noise [field] | compute | reader: data.py:82, :120, evaluation.py:92 | keep | an axis; no change
- Clean/gantry_dynamic/config.py:72 | device = 'cuda' [field] | compute | reader: model.py:107, :128, training.py:202, :264 | keep | a user sets it (--cpu); no change
- Clean/gantry_dynamic/config.py:73 | compile_mode = 'reduce-overhead' [field] | compute | reader: controller.py:227 | keep | a user sets it (--cpu sets None); no change
- Clean/gantry_dynamic/config.py:74 | truncate_s = None [field] | verification | reader: data.py:135 (harness hcommon, smoke.py set it) | keep | cheap option a user can use for a quick run on record prefixes (amendment 9 point 2); not on the CLI; no change
- Clean/gantry_dynamic/config.py:75 | epochs = 200 [field] | compute, unlabelled | reader: config.py:208 (hp), training.py:296 | label | a user sets the Adam budget; add "# HEURISTIC: Adam budget of the thesis runs" (or its source); bitwise
- Clean/gantry_dynamic/config.py:76 | n_its = None [field] | verification | reader: training.py:261, :262 | keep | a user can cap the run in updates; the harness and smoke.py use it for short runs; no change
- Clean/gantry_dynamic/config.py:77 | its_per_val = 1300 [field] | compute, unlabelled | reader: training.py:262 | label | validation cadence a user may set; label its source (updates between validations of the thesis runs); bitwise
- Clean/gantry_dynamic/config.py:78 | lbfgs_max_iter = 100 [field] | compute, unlabelled | reader: training.py:197 | label | polish budget; label; bitwise
- Clean/gantry_dynamic/config.py:79 | lbfgs_inner_iter = 20 [field] | compute, unlabelled | reader: training.py:198 | label | label; bitwise
- Clean/gantry_dynamic/config.py:80 | lbfgs_windows = 16384 [field] | compute, unlabelled | reader: training.py:197, message lbfgs_polish.py:285 | label | memory knob the OOM message names; label (windows per polish batch); bitwise
- Clean/gantry_dynamic/config.py:81 | start_phase = 'adam' [field] | compute | reader: training.py:280, :343, gantry_interconnect_dynamic.py:129 | keep | set by main for a polish-only rerun; no change
- Clean/gantry_dynamic/config.py:83 | baseline_form = 'lfr' [field] | compute | reader: model.py:59 | keep | a user sets it (--closed-form), D-284 amendment 3; no change
- Clean/gantry_dynamic/config.py:85 | data_root [field] | compute | reader: data.py:82, evaluation.py:267, gantry_interconnect_dynamic.py:138 | keep | a user sets it; no change
- Clean/gantry_dynamic/config.py:86 | out_root [field] | compute | reader: config.py:214 | keep | a user sets it; no change
- Clean/gantry_dynamic/config.py:88 | comment "fixed settings of the thesis runs under the mainline names; config_json_dict writes them" [comment] | misleading, mainline-compat | reader: none | comment | after P1/P2: "fixed settings of the method"; bitwise
- Clean/gantry_dynamic/config.py:90 | seed = 1 [ClassVar] | compute | reader: gantry_interconnect_dynamic.py:112, config.py:193, training.py:200, :243 | keep | read by the seeding and the detune draw; no change
- Clean/gantry_dynamic/config.py:91 | nx_phys = 6 [ClassVar] | compute | reader: model.py:40, data.py:233, obc_gantry.py:249, training.py:244 | keep | plant dimension, not a tuning number; no change
- Clean/gantry_dynamic/config.py:92 | nu = 3 [ClassVar] | compute | reader: model.py:47, :66, data.py:235 | keep | plant dimension; no change
- Clean/gantry_dynamic/config.py:93 | ny = 3 [ClassVar] | compute | reader: model.py:47, :98 | keep | plant dimension; no change
- Clean/gantry_dynamic/config.py:94 | fs_orig = 20000 [ClassVar] | compute, unlabelled | reader: config.py:124, data.py:27 | label | "# THEORY: simulation rate of the data sets (generate_trajectory_data)"; bitwise
- Clean/gantry_dynamic/config.py:95 | fs_new = 4000 [ClassVar] | compute, unlabelled | reader: config.py:60, :124, data.py:136 | label | label the training rate choice; bitwise
- Clean/gantry_dynamic/config.py:96 | stride = 10 [ClassVar] | compute, unlabelled | reader: model.py:153, training.py:221, :263 | label | HEURISTIC window stride; bitwise
- Clean/gantry_dynamic/config.py:97 | nf_seconds = 0.100 [ClassVar] | compute, unlabelled | reader: config.py:126 | label | HEURISTIC window length; bitwise
- Clean/gantry_dynamic/config.py:98 | na_nb = 30 [ClassVar] | compute, unlabelled | reader: config.py:205 (hp), model.py:44 | label | encoder window n + 1; label; bitwise
- Clean/gantry_dynamic/config.py:99 | n_nodes_per_layer = 16 [ClassVar] | compute, unlabelled | reader: config.py:201 (hp), model.py:67, :100 | label | HEURISTIC width; bitwise
- Clean/gantry_dynamic/config.py:100 | n_hidden_layers = 2 [ClassVar] | compute | reader: config.py:202 (hp), model.py:68 | label | HEURISTIC depth; bitwise
- Clean/gantry_dynamic/config.py:101 | up_sample = 1 [ClassVar] | fixed-option | reader: config.py:203 (hp), model.py:58 | keep | ruled K34 (amendment 9: useful option); no new evidence
- Clean/gantry_dynamic/config.py:102 | batch_size = 512 [ClassVar] | compute, unlabelled | reader: config.py:206 (hp), training.py:260 | label | HEURISTIC; bitwise
- Clean/gantry_dynamic/config.py:103 | lr = 1e-5 [ClassVar] | compute, unlabelled | reader: config.py:207 (hp), model.py:111 | label | HEURISTIC; bitwise
- Clean/gantry_dynamic/config.py:104 | adam_eps = 1e-16 [ClassVar] | compute, unlabelled | reader: model.py:111, :113 | label | HEURISTIC (float64 eps choice); bitwise
- Clean/gantry_dynamic/config.py:105 | param_prior = False [ClassVar] | write-only, mainline-compat | reader: none (config_json_dict:299 only) | remove | mainline's parameter-prior switch; Clean/ has no prior; with PARAM_PRIOR (A4); bitwise; config.json loses the key
- Clean/gantry_dynamic/config.py:107 | param_rmse_baseline = 0.01 [ClassVar] | write-only, mainline-compat | reader: model.py:59 (only into Lambda_combo, which no loss reads) | remove | with Lambda_combo, RMSE_baseline and PARAM_RMSE_BASELINE (P3); bitwise; hfn state_dict loses Lambda_combo (P3 cost)
- Clean/gantry_dynamic/config.py:108 | use_f64 = True [ClassVar] | write-only, fixed-option | reader: none (config_json_dict:272 only); dtype_pt is float64 regardless | remove | bitwise; config.json loses USE_F64
- Clean/gantry_dynamic/config.py:109 | joint_estimation = True [ClassVar] | write-only, fixed-option | reader: gantry_interconnect_dynamic.py:129, evaluation.py:241 (both into the unused load_checkpoint parameter, K7) | remove | with K7; bitwise; harness callers of load_checkpoint (l4_loader.py:432, :468, run_l3bc.py:94) drop the argument
- Clean/gantry_dynamic/config.py:110 | lbfgs = True [ClassVar] | write-only, fixed-option | reader: none (config_json_dict:287) | remove | the polish always runs; bitwise
- Clean/gantry_dynamic/config.py:111 | encoder_init = 'data_lsq' [ClassVar] | write-only, mainline-compat | reader: config.py:214 (folder name only) | remove | with the folder rename of row save_dir; bitwise; new runs land in <out_root>/<noise>/<run_id>
- Clean/gantry_dynamic/config.py:112 | validation_measure = 'sim-NRMSe' [ClassVar] | compute (name only) | reader: training.py:211, :213, :224, :265, gantry_interconnect_dynamic.py:134 (prints and Jan's fit prints) | keep | the one display name of the score; the measure itself is always ClosedLoopSimulator.validation_error; the misleading 'sim-NRMS' defaults are rows in E
- Clean/gantry_dynamic/config.py:113 | controller_gain = False [ClassVar] | write-only, fixed-option, mainline-compat | reader: none in Clean/ (config_json_dict:307); harness l4_loader.py:263, :385 passes it as use_gain | remove | not wired to use_gain in Clean/ (build_closed_loop never passes it); bitwise; l4_loader passes use_gain=False
- Clean/gantry_dynamic/config.py:114 | lbfgs_history = 50 [ClassVar] | compute, unlabelled | reader: training.py:198 | label | HEURISTIC L-BFGS memory; bitwise
- Clean/gantry_dynamic/config.py:116 | lbfgs_tol_grad = 2.6e-5 [ClassVar] | compute | reader: training.py:199 | keep | labelled HEURISTIC, read by PolishSpec; no change
- Clean/gantry_dynamic/config.py:117 | lbfgs_tol_change = 2.6e-7 [ClassVar] | compute | reader: training.py:199 | keep | labelled; no change
- Clean/gantry_dynamic/config.py:118 | obc_space = 'affine' [ClassVar] | write-only, fixed-option, mainline-compat | reader: obc_gantry.py:261 (print) | literal | the basis is always affine (append_value=True at obc_gantry.py:213); print 'space=affine' literally; bitwise; log text unchanged
- Clean/gantry_dynamic/config.py:119 | obc_refresh = 'epoch' [ClassVar] | write-only, fixed-option, mainline-compat | reader: obc_gantry.py:261 (print) | literal | refresh is always per epoch (interconnect.py:605); print 'refresh=epoch' literally; bitwise
- Clean/gantry_dynamic/config.py:120 | obc_rank_rtol = None [ClassVar] | write-only, mainline-compat, misleading | reader: none; obc.py:44 always uses S_max max(M, N) eps | remove | bitwise; config.json loses OBC_RANK_RTOL
- Clean/gantry_dynamic/config.py:121 | obc_ref_successor = True [ClassVar] | write-only, mainline-compat, misleading | reader: none (config_json_dict only) | remove | Clean/ builds no successors since K15; comment the tuple range in build_reference_set instead (row D obc_gantry.py:86); bitwise; config.json loses the key
- Clean/gantry_dynamic/config.py:122 | obc_ref_chunk = 16384 [ClassVar] | compute, unlabelled | reader: obc_gantry.py:256 | label | "# HEURISTIC: tuples per jacfwd call; memory only, the basis is the same at any chunk" and make it the one source of build_stacked_sensitivity's chunk (row E obc.py:83); bitwise
- Clean/gantry_dynamic/config.py:124 | d = fs_orig // fs_new [ClassVar] | compute | reader: data.py:103, :119, :129, config.py:54 | keep | derived; no change
- Clean/gantry_dynamic/config.py:125 | ts_new [ClassVar] | compute | reader: model.py:58, controller.py:215, data.py:146 | keep | derived; no change
- Clean/gantry_dynamic/config.py:126 | nf = max(1, int(nf_seconds / ts_new)) [ClassVar] | compute | reader: config.py:204 (hp), training.py:221 | keep | derived; the max(1, .) guard is never active (nf = 400) but costs nothing; no change
- Clean/gantry_dynamic/config.py:127 | dtype_np = np.float64 [ClassVar] | compute | reader: data.py:146, :219 | keep | read; no change
- Clean/gantry_dynamic/config.py:128 | dtype_pt = torch.float64 [ClassVar] | compute | reader: model.py:52, :103, controller.py:215, obc_gantry.py:94 | keep | read; no change
- Clean/gantry_dynamic/config.py:132 | `allowed` values of __post_init__ (n_a in (0, 2, 8), noise, device, ...) [guard] | compute | reader: config.py:136 | keep | guards CLI and config.json input with an actionable message; no change
- Clean/gantry_dynamic/config.py:139 | compile_mode needs cuda [guard] | compute | reader: config.py:139 | keep | actionable guard; no change
- Clean/gantry_dynamic/config.py:166 | obc [property] | compute | reader: model.py:119, training.py:208, :212 | keep | read; no change
- Clean/gantry_dynamic/config.py:171 | ann_route_ix = range(6 + n_a) [property] | fixed-option, mainline-compat | reader: model.py:41, obc_gantry.py:250, training.py:245 | remove | routing is always every state in order (mainline's option, D-068); with K32: model.py uses np.arange(nx), GantryOBCCorrection and PS2Opening lose route_ix (rows E); bitwise (expansion_matrix(arange(nx), nx) is the identity it builds now); config.json loses ANN_ROUTE_IX
- Clean/gantry_dynamic/config.py:173 | literal 6 in range(6 + self.n_a) [number] | unlabelled | reader: config.py:173 | literal | use self.nx_phys; bitwise (goes with the row above)
- Clean/gantry_dynamic/config.py:176 | mode [property] "The mainline name of the data set" | mainline-compat | reader: config.py:214 (folder name), :248 (MODE), data.py:172 (print) | remove | print cfg.noise and name the folder after it (row save_dir); bitwise; folder layout of new runs changes
- Clean/gantry_dynamic/config.py:181 | combo_init_detune [property] | compute | reader: model.py:54, gantry_interconnect_dynamic.py:94 | keep | labelled HEURISTIC, read by build_model; no change
- Clean/gantry_dynamic/config.py:197 | hp [property] docstring "keys and order are those of the checkpoint .npz field hp" | compute; comment mainline-compat | reader: gantry_interconnect_dynamic.py:102, evaluation.py:239 | comment | hp stays (K30 ruled); drop the order claim once the .npz hp field goes (row B3); bitwise
- Clean/gantry_dynamic/config.py:212 | save_dir folder name f'{mode}_{encoder_init}' [name] | mainline-compat | reader: gantry_interconnect_dynamic.py:82 | rename | os.path.join(cfg.out_root, cfg.noise); bitwise; harness readers that rebuild the folder (smoke.py, gpu_smoke.py, l4_stage.py, run_l3.py, run_l3bc.py, timing_driver.py) follow; existing folders are passed by path and still work
- Clean/gantry_dynamic/config.py:237 | GIT_SHA, GIT_COMMIT environment fallback [option] | compute | reader: config.py:237 | keep | provenance on a server without git, a user needs it (amendment 9 point 2); no change
- Clean/gantry_dynamic/config.py:235 | [:8] hash digits [number] | unlabelled | reader: config.py:235 | keep | display length of a hash, not a model number; no change

### A4. Keys of `config_json_dict` and of `main`'s config.json
Rule used for every key (P1): (a) read back into RunConfig: keep; (b) a setting Clean/ computes with, not derivable from other kept keys: keep, it is what the guard of A2 compares; (c) derived from or duplicating a kept key: remove; (d) a literal no Clean/ code reads (mainline option or a description not tied to the code): remove; (e) needed only to read thesis folders: move to the converter; (f) provenance: keep. Every remove is bitwise; config.json of new runs loses the key; existing Clean folders still load (they keep their old keys, which the new reader ignores); compare_l3.py compares common keys only, so it reports nothing new.
- Clean/gantry_dynamic/config.py:248 | MODE [config.json key] | mainline-compat | reader: config.py:153 (mainline branch only) | move | (e); NOISE holds it; bitwise
- Clean/gantry_dynamic/config.py:248 | ENCODER_INIT = 'data_lsq' [config.json key] | write-only, fixed-option, mainline-compat | reader: none | remove | (d)
- Clean/gantry_dynamic/config.py:249 | ANN_ACTIVATION = 'tanh' [config.json key] | write-only, mainline-compat, misleading | reader: none; model.py:71 hard-codes torch.nn.Tanh | remove | (d): the literal is not tied to the code, so it would stay 'tanh' if the activation changed
- Clean/gantry_dynamic/config.py:250 | ANN_ARCH = 'mlp' [config.json key] | write-only, mainline-compat | reader: none | remove | (d); mainline's mlp/resnet option
- Clean/gantry_dynamic/config.py:251 | G_INIT = 'live' [config.json key] | write-only, mainline-compat | reader: none | remove | (d); mainline's zero/live option; Clean's init is torch_nets.py:62
- Clean/gantry_dynamic/config.py:252 | LIVE_G_WEIGHT_INIT = 'kaiming_uniform_a_sqrt5' [config.json key] | write-only, mainline-compat | reader: none | remove | (d)
- Clean/gantry_dynamic/config.py:253 | WA_ENCODER_INIT = 'xavier_uniform_gain1' [config.json key] | write-only, mainline-compat | reader: none | remove | (d); Clean's init is pre_encoder.py:69
- Clean/gantry_dynamic/config.py:254 | FS_NEW [config.json key] | compute | reader: config.py:158 (guard) | keep | (b)
- Clean/gantry_dynamic/config.py:254 | D [config.json key] | write-only | reader: guard only; derived from FS_ORIG, FS_NEW | remove | (c)
- Clean/gantry_dynamic/config.py:255 | FS_ORIG [config.json key] | compute | reader: guard | keep | (b)
- Clean/gantry_dynamic/config.py:255 | SEED [config.json key] | compute | reader: guard | keep | (b)
- Clean/gantry_dynamic/config.py:255 | SNR = None [config.json key] | write-only, mainline-compat | reader: none | remove | (d); mainline's synthetic noise option; Clean's noise is the data set
- Clean/gantry_dynamic/config.py:256 | JOINT_ESTIMATION [config.json key] | write-only, fixed-option | reader: none | remove | (d); with the ClassVar (A3)
- Clean/gantry_dynamic/config.py:257 | PHYSICS_PARAMETERIZATION = 'reduced' [config.json key] | write-only, mainline-compat | reader: none | remove | (d); Clean/ has only the reduced block
- Clean/gantry_dynamic/config.py:258 | PARAM_RMSE_BASELINE [config.json key] | write-only, mainline-compat | reader: none | remove | (d); P3
- Clean/gantry_dynamic/config.py:259 | PARAM_INIT_DETUNE = None [config.json key] | write-only, mainline-compat | reader: none | remove | (d); mainline's raw-parameter detune
- Clean/gantry_dynamic/config.py:260 | COMBO_INIT_DETUNE [config.json key] | mainline-compat | reader: config.py:151 (mainline branch); l4_loader.py:81 reads it from staged thesis folders | move | (c) derived from START and SEED, (e) for thesis folders; bitwise; l4_loader reads thesis folders, unaffected
- Clean/gantry_dynamic/config.py:261 | ORTH_BETA = 4.66e-4 [config.json key] | write-only, mainline-compat, unlabelled | reader: none | remove | (d); mainline's soft orthogonality strength, off in every thesis run
- Clean/gantry_dynamic/config.py:262 | ORTH_POINT_STRIDE = 100 [config.json key] | write-only, mainline-compat, unlabelled | reader: none | remove | (d)
- Clean/gantry_dynamic/config.py:263 | ORTH_RANK_TOL = 1e-12 [config.json key] | write-only, mainline-compat, unlabelled | reader: none | remove | (d)
- Clean/gantry_dynamic/config.py:264 | CLOSED_LOOP = True [config.json key] | write-only, fixed-option | reader: none | remove | (d); Clean/ is closed loop only
- Clean/gantry_dynamic/config.py:265 | ADAM_EPS [config.json key] | compute | reader: guard | keep | (b)
- Clean/gantry_dynamic/config.py:266 | ORTH = False [config.json key] | write-only, mainline-compat | reader: none | remove | (d)
- Clean/gantry_dynamic/config.py:267 | ANN_ROUTE_IX [config.json key] | write-only, fixed-option | reader: guard; derived from N_A | remove | (c); with ann_route_ix (A3)
- Clean/gantry_dynamic/config.py:268 | STRIDE [config.json key] | compute | reader: guard | keep | (b)
- Clean/gantry_dynamic/config.py:269 | N_ITS [config.json key] | compute | reader: config.py:155 (_ENV_KEYS) | keep | (a)
- Clean/gantry_dynamic/config.py:270 | ITS_PER_VAL [config.json key] | compute | reader: config.py:155 | keep | (a)
- Clean/gantry_dynamic/config.py:271 | NF_SECONDS [config.json key] | compute | reader: guard | keep | (b)
- Clean/gantry_dynamic/config.py:272 | USE_F64 [config.json key] | write-only, fixed-option | reader: none | remove | (d)
- Clean/gantry_dynamic/config.py:273 | SAVE_FLAG = True [config.json key] | write-only, mainline-compat | reader: none | remove | (d); Clean/ always saves
- Clean/gantry_dynamic/config.py:274 | NF_PROBE_PRINT = False [config.json key] | write-only, mainline-compat | reader: none (skipped at :157) | remove | (d); mainline's per-epoch probe print
- Clean/gantry_dynamic/config.py:275 | NX_ANN [config.json key] | mainline-compat | reader: config.py:152 (mainline branch); l4_loader.py:81 on thesis folders | move | (c) duplicate of N_A, (e); bitwise
- Clean/gantry_dynamic/config.py:276 | N_NODES_PER_LAYER [config.json key] | compute | reader: guard | keep | (b)
- Clean/gantry_dynamic/config.py:277 | N_HIDDEN_LAYERS [config.json key] | compute | reader: guard | keep | (b)
- Clean/gantry_dynamic/config.py:278 | UP_SAMPLE [config.json key] | compute | reader: guard | keep | (b)
- Clean/gantry_dynamic/config.py:279 | BATCH_SIZE [config.json key] | compute | reader: guard | keep | (b)
- Clean/gantry_dynamic/config.py:280 | LR [config.json key] | compute | reader: guard | keep | (b)
- Clean/gantry_dynamic/config.py:281 | EPOCHS [config.json key] | compute | reader: config.py:155 | keep | (a)
- Clean/gantry_dynamic/config.py:282 | NF [config.json key] | write-only | reader: guard; derived from NF_SECONDS, FS_NEW | remove | (c)
- Clean/gantry_dynamic/config.py:283 | NA_NB [config.json key] | compute | reader: guard | keep | (b)
- Clean/gantry_dynamic/config.py:284 | DEVICE [config.json key] | compute | reader: config.py:155 | keep | (a)
- Clean/gantry_dynamic/config.py:285 | CHECKPOINT_CHUNK = 0 [config.json key] | write-only, mainline-compat | reader: none | remove | (d); mainline's gradient checkpointing
- Clean/gantry_dynamic/config.py:286 | COMPILE_MODE [config.json key] | compute | reader: config.py:155 | keep | (a)
- Clean/gantry_dynamic/config.py:287 | LBFGS = True [config.json key] | write-only, fixed-option | reader: none | remove | (d)
- Clean/gantry_dynamic/config.py:288 | LBFGS_MAX_ITER [config.json key] | compute | reader: config.py:155 | keep | (a)
- Clean/gantry_dynamic/config.py:289 | LBFGS_INNER_ITER [config.json key] | compute | reader: config.py:155 | keep | (a)
- Clean/gantry_dynamic/config.py:290 | LBFGS_HISTORY [config.json key] | compute | reader: guard | keep | (b)
- Clean/gantry_dynamic/config.py:291 | LBFGS_WINDOWS [config.json key] | compute | reader: config.py:155 | keep | (a)
- Clean/gantry_dynamic/config.py:292 | LBFGS_CHUNK = None [config.json key] | write-only, mainline-compat | reader: none | remove | (d)
- Clean/gantry_dynamic/config.py:293 | LBFGS_TOL_GRAD [config.json key] | compute | reader: guard | keep | (b)
- Clean/gantry_dynamic/config.py:294 | LBFGS_TOL_CHANGE [config.json key] | compute | reader: guard | keep | (b)
- Clean/gantry_dynamic/config.py:295 | LBFGS_FREEZE = [] [config.json key] | write-only, mainline-compat | reader: none | remove | (d)
- Clean/gantry_dynamic/config.py:296 | LBFGS_TIME_BUDGET_S = None [config.json key] | write-only, mainline-compat | reader: none | remove | (d)
- Clean/gantry_dynamic/config.py:297 | LBFGS_METER_REL_TOL = 0.001 [config.json key] | write-only, mainline-compat, unlabelled | reader: none | remove | (d); mainline's negation-meter gate
- Clean/gantry_dynamic/config.py:298 | START_PHASE [config.json key] | write-only | reader: none (skipped); always 'adam', since main writes config.json for fresh runs only | remove | (d); a rerun's phase is in RESUMES
- Clean/gantry_dynamic/config.py:299 | PARAM_PRIOR [config.json key] | write-only, mainline-compat | reader: none | remove | (d)
- Clean/gantry_dynamic/config.py:300 | STATIC_PASS_BUFFERS = False [config.json key] | write-only, mainline-compat | reader: none | remove | (d)
- Clean/gantry_dynamic/config.py:301 | OBC [config.json key] | mainline-compat | reader: config.py:150 (mainline branch); l4_loader.py:81 on thesis folders | move | (c) duplicate of ARM, (e)
- Clean/gantry_dynamic/config.py:302 | OBC_SPACE [config.json key] | write-only, fixed-option | reader: none | remove | (d)
- Clean/gantry_dynamic/config.py:303 | OBC_REFRESH [config.json key] | write-only, fixed-option | reader: none | remove | (d)
- Clean/gantry_dynamic/config.py:304 | OBC_REF_CHUNK [config.json key] | write-only | reader: guard | remove | (d): memory only, the result is the same at any chunk, so the guard must not refuse on it
- Clean/gantry_dynamic/config.py:305 | OBC_RANK_RTOL [config.json key] | write-only, mainline-compat | reader: none | remove | (d)
- Clean/gantry_dynamic/config.py:306 | TRUNCATE_S [config.json key] | compute | reader: config.py:155 | keep | (a)
- Clean/gantry_dynamic/config.py:307 | CONTROLLER_GAIN [config.json key] | write-only, mainline-compat | reader: none | remove | (d)
- Clean/gantry_dynamic/config.py:308 | PS2 [config.json key] | compute | reader: config.py:147 | keep | (a)
- Clean/gantry_dynamic/config.py:309 | Y_DECIMATION [config.json key] | compute | reader: guard | keep | (b): the only record of AA_ORDER and AA_FC_HZ in config.json
- Clean/gantry_dynamic/config.py:310 | RECORD_TRIM [config.json key] | compute | reader: guard | keep | (b): the only record of AA_TRIM_S
- Clean/gantry_dynamic/config.py:311 | LOSS_WEIGHTING = 'servo' [config.json key] | write-only, fixed-option | reader: none | remove | (d)
- Clean/gantry_dynamic/config.py:312 | VALIDATION_MEASURE [config.json key] | write-only | reader: none | remove | (d): a display name
- Clean/gantry_dynamic/config.py:313 | LBFGS_METERS_GATE = False [config.json key] | write-only, mainline-compat | reader: none | remove | (d)
- Clean/gantry_dynamic/config.py:314 | OBC_REF_SUCCESSOR [config.json key] | write-only, mainline-compat, misleading | reader: none | remove | (d); Clean/ builds no successors
- Clean/gantry_dynamic/config.py:315 | ARM [config.json key] | compute | reader: config.py:147 | keep | (a)
- Clean/gantry_dynamic/config.py:316 | START [config.json key] | compute | reader: config.py:147 | keep | (a)
- Clean/gantry_dynamic/config.py:317 | N_A [config.json key] | compute | reader: config.py:147 | keep | (a)
- Clean/gantry_dynamic/config.py:318 | NOISE [config.json key] | compute | reader: config.py:148 | keep | (a)
- Clean/gantry_dynamic/config.py:319 | BASELINE_FORM [config.json key] | compute | reader: config.py:155 | keep | (a)
- Clean/gantry_dynamic/config.py:320 | DATA_ROOT [config.json key] | write-only | reader: none ("never read back", :84) | remove | (c): data_dir records the folder used
- Clean/gantry_dynamic/config.py:321 | OUT_ROOT [config.json key] | write-only | reader: none | remove | (d): the folder is where the file is
- Clean/gantry_dynamic/config.py:322 | LAST_SAVE = 'every validation' [config.json key] | write-only, mainline-compat | reader: none | remove | (d): marks a difference to mainline's dumps; the behaviour is in interconnect.py:419
- Clean/gantry_dynamic/config.py:323 | CODE = 'clean' [config.json key] | mainline-compat | reader: config.py:146 | remove | (e): only tells Clean folders from mainline folders; with P1 the reader keys on ARM
- Clean/gantry_dynamic/config.py:324 | GIT [config.json key] | compute | reader: user (provenance) | keep | (f)
- Clean/gantry_interconnect_dynamic.py:108 | hp [config.json key] | write-only | reader: none (capture_l12.py reads cfg.hp, not this key) | remove | (c): duplicates N_NODES_PER_LAYER, N_HIDDEN_LAYERS, UP_SAMPLE, NA_NB, BATCH_SIZE, LR, EPOCHS, N_A and NF
- Clean/gantry_interconnect_dynamic.py:108 | run_id [config.json key] | compute | reader: gantry_interconnect_dynamic.py:37, evaluation.py:235 | keep | (f) and read
- Clean/gantry_interconnect_dynamic.py:108 | data_dir [config.json key] | write-only | reader: user; harness compare_l3.py skips it | keep | (f): the data folder a run used
- Clean/gantry_interconnect_dynamic.py:65 | RESUMES [config.json key] | write-only | reader: user; harness run_l3bc.py:241 | keep | (f): the rerun history of a folder
- Clean/gantry_interconnect_dynamic.py:67 | RESUMES entry key code='clean' [config.json key] | mainline-compat | reader: none | remove | tells Clean reruns from mainline reruns only; git holds the provenance; bitwise
- Clean/gantry_interconnect_dynamic.py:66 | RESUMES entry keys time, type, source, git, rerun_id [config.json keys] | write-only | reader: user; run_l3bc.py:241 | keep | (f)
- Clean/gantry_dynamic/config.py:246 | config_json_dict docstring "mainline configuration keys in their original order, then Clean keys" [comment] | mainline-compat | reader: none | comment | "the run's settings, read back by RunConfig.from_config_json"; bitwise

## B. Checkpoints: state_dict keys, the pair, the dumps, the logs

### B1. Registered buffers and parameters (state_dict keys of hfn and encoder)
Cost of every key removal below (P3): the hfn state_dict loses the key, so `_assert_same_architecture` (training.py:96) refuses every existing checkpoint that has it (the thesis checkpoints and all Clean checkpoints written so far, including any step-5 server run made before the change) until the converter of P1 drops it; compare_l3.py and compare_l3bc.py `_cmp_tensors` on hfn need the key in a dropped set (as compare.py:46 DROPPED_IN_CLEAN); r1_check.py and the L4 clean side load converted folders. Outputs bitwise: no computation reads the key.
- Clean/model_augmentation/fit_systems/blocks.py:63 | out_gate [buffer, persistent=False, Jan's code] | compute | reader: blocks.py:78, evaluation.py:55 | keep | the evaluation's baseline copy zeroes it (amendment 9 point 4); not a state_dict key; no change
- Clean/model_augmentation/fit_systems/blocks.py:96 | C [buffer, Jan's code] | compute | reader: blocks.py:113 | keep | the output map C^n; no change
- Clean/model_augmentation/fit_systems/blocks.py:97 | D [buffer, Jan's code] | compute, fixed-option | reader: blocks.py:113 (always zero) | keep | ruled K40; no new evidence
- Clean/model_augmentation/fit_systems/blocks.py:150 | nominal matrices built in GantryStateBlock.__init__ (build_poly_constants, d0, build_G_matrix_entries at theta^0) [computation] | write-only | reader: only the 13 buffers below | remove | lines 150-159 and 161-175 except P_mat; the trainable subclass rebuilds every matrix per objective (blocks.py:698); bitwise; P3
- Clean/model_augmentation/fit_systems/blocks.py:161 | mh [buffer] | write-only | reader: none (blocks.py:153, :156 read gantry_ss.mh, not the buffer) | remove | P3
- Clean/model_augmentation/fit_systems/blocks.py:162 | alpha [buffer] | write-only | reader: none (_deriv_with unpacks alpha from mats) | remove | P3
- Clean/model_augmentation/fit_systems/blocks.py:163 | beta [buffer] | write-only | reader: none | remove | P3
- Clean/model_augmentation/fit_systems/blocks.py:164 | gamma_ [buffer] | write-only | reader: none | remove | P3
- Clean/model_augmentation/fit_systems/blocks.py:165 | N0 [buffer] | write-only | reader: none | remove | P3
- Clean/model_augmentation/fit_systems/blocks.py:166 | N1 [buffer] | write-only | reader: none | remove | P3
- Clean/model_augmentation/fit_systems/blocks.py:167 | N2 [buffer] | write-only | reader: none | remove | P3
- Clean/model_augmentation/fit_systems/blocks.py:169 | Ax [buffer] | write-only | reader: none | remove | P3
- Clean/model_augmentation/fit_systems/blocks.py:170 | Bw [buffer] | write-only | reader: none | remove | P3
- Clean/model_augmentation/fit_systems/blocks.py:171 | Bu [buffer] | write-only | reader: none | remove | P3
- Clean/model_augmentation/fit_systems/blocks.py:172 | A_combined [buffer] | write-only | reader: none (_deriv_with unpacks it from mats) | remove | P3
- Clean/model_augmentation/fit_systems/blocks.py:174 | K_mat [buffer] | write-only | reader: none | remove | P3
- Clean/model_augmentation/fit_systems/blocks.py:175 | C_mat [buffer] | write-only | reader: none | remove | P3
- Clean/model_augmentation/fit_systems/blocks.py:176 | P_mat [buffer] | compute | reader: blocks.py:233, :294 | keep | read by both vector fields; no change
- Clean/model_augmentation/fit_systems/blocks.py:179 | std_x [buffer] | compute | reader: blocks.py:225, :291 | keep | no change
- Clean/model_augmentation/fit_systems/blocks.py:180 | std_x_1d [buffer] | compute | reader: blocks.py:267, :351 | keep | same values as std_x in the shape the output division needs; no change
- Clean/model_augmentation/fit_systems/blocks.py:181 | std_u [buffer] | compute | reader: blocks.py:226, :293 | keep | no change
- Clean/model_augmentation/fit_systems/blocks.py:182 | x_mean [buffer] | compute | reader: blocks.py:225, :292 | keep | no change
- Clean/model_augmentation/fit_systems/blocks.py:183 | u_mean [buffer] | compute | reader: blocks.py:226, :293 | keep | no change
- Clean/model_augmentation/fit_systems/blocks.py:398 | params_init [buffer] | compute | reader: blocks.py:527, :701, evaluation.py:73 | keep | theta^0, read by the gauge section and the true-parameter evaluation; no change
- Clean/model_augmentation/fit_systems/blocks.py:400 | _RMSE_baseline [attribute] | write-only, mainline-compat | reader: blocks.py:535 (Lambda_combo only) | remove | P3; bitwise
- Clean/model_augmentation/fit_systems/blocks.py:399 | comment "the scale of the subclass's Lambda_combo buffer" [comment] | mainline-compat | reader: none | remove | with _RMSE_baseline; bitwise
- Clean/model_augmentation/fit_systems/blocks.py:403 | Lb [buffer] | compute | reader: blocks.py:419, :475, :611 | keep | no change
- Clean/model_augmentation/fit_systems/blocks.py:530 | free_params [parameter] | compute | reader: blocks.py:498, interconnect.py:563 | keep | beta, the trained physical coordinates; no change
- Clean/model_augmentation/fit_systems/blocks.py:531 | combo_init [buffer] | compute | reader: blocks.py:627, :655, :705 | keep | no change
- Clean/model_augmentation/fit_systems/blocks.py:532 | Lambda_combo [buffer] | write-only, mainline-compat | reader: none (its own comment: "read by no loss"); mainline reads it in param_loss, off in the thesis runs | remove | P3
- Clean/model_augmentation/fit_systems/blocks.py:544 | coordinate_version [buffer] | mainline-compat | reader: training.py:88 (_assert_same_architecture) | keep | guards a checkpoint of another coordinate map with an actionable message; today only mainline checkpoints of version 1 differ, so say so in the comment; no change
- Clean/model_augmentation/fit_systems/blocks.py:546 | m_diff_eta0 [buffer] | compute | reader: blocks.py:630, :658 | keep | no change
- Clean/model_augmentation/fit_systems/blocks.py:547 | m_diff_scale [buffer] | compute | reader: blocks.py:630, :658 | keep | no change
- Clean/model_augmentation/fit_systems/blocks.py:53 | net.* (AugmentationNetwork layers) [parameters, Jan's code] | compute | reader: blocks.py:77 | keep | theta_aug; no change
- Clean/model_augmentation/fit_systems/pre_encoder.py:61 | Wb_psi_y, Wb_psi_u [parameters, Jan's code] | compute | reader: pre_encoder.py:95 | keep | W^b; no change
- Clean/model_augmentation/fit_systems/pre_encoder.py:70 | Wa_psi_y, Wa_psi_u [parameters, Jan's code] | compute | reader: pre_encoder.py:96 | keep | W^a; no change
- Clean/model_augmentation/fit_systems/pre_encoder.py:74 | net.* (psi_tilde layers) [parameters, Jan's code] | compute | reader: pre_encoder.py:103 | keep | no change
- Clean/gantry_dynamic/obc_gantry.py:133 | E [buffer] | compute, fixed-option | reader: obc_gantry.py:180 | keep | the physical-row selector; with K32 it is always [I_6; 0]; keep it as the matrix of eq. f_obc (no change), only its construction from route_ix goes (row E)
- Clean/gantry_dynamic/obc_gantry.py:134 | vbar [buffer] | compute | reader: obc_gantry.py:158 | keep | beta_lin for prediction; no change
- Clean/gantry_dynamic/obc_gantry.py:136 | theta_frozen [buffer] | compute | reader: interconnect.py:583, evaluation.py:57 | keep | no change
- Clean/model_augmentation/fit_systems/closed_loop.py:48 | M_state [buffer] | compute | reader: closed_loop.py:118 | keep | in the dumps through the pickled simulator, not in the pair; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:49 | M_error [buffer] | compute | reader: closed_loop.py:118 | keep | no change

### B2. The pair `<base>.pt` / `<base>.npz` and the accepted formats
- Clean/gantry_dynamic/training.py:33 | 'hfn' [.pt key] | compute | reader: training.py:50 | keep | no change
- Clean/gantry_dynamic/training.py:34 | 'encoder' [.pt key] | compute | reader: training.py:50 | keep | no change
- Clean/gantry_dynamic/training.py:37 | 'optimizer' [.pt key] | compute | reader: training.py:50, :136 | keep | Adam moments of the pre-polish pair; no change
- Clean/gantry_dynamic/training.py:313 | done_epochs [.npz key, both pairs] | compute | reader: training.py:288 | keep | the polish-only rerun reads it from the pre_lbfgs pair; no change
- Clean/gantry_dynamic/training.py:313 | bestfit [.npz key, both pairs] | compute | reader: training.py:290; val_check.py:21, l4_loader.py:469 | keep | no change
- Clean/gantry_dynamic/training.py:314 | hp [.npz key, both pairs] | write-only | reader: none (compare_l3.py compares it) | remove | duplicate of config.json's settings (A4); bitwise; compare_l3.py: add 'hp' to NPZ_SKIP
- Clean/gantry_dynamic/training.py:314 | orig_run_id [.npz key, both pairs] | write-only | reader: none (compare_l3.py:18 skips it) | remove | always the folder's run_id (config.json run_id); bitwise
- Clean/gantry_dynamic/training.py:341 | bestfit_selected [.npz key, final] | write-only | reader: none in Clean/; l4_stage.py | remove | (c): the val_hist.npz 'selected' row holds the same score; bitwise; l4_stage.py reads that row
- Clean/gantry_dynamic/training.py:341 | elapsed [.npz key, final] | write-only | reader: none (compare_l3.py skips) | remove | (c): loss.npz 'time' holds the Adam wall time per validation and the log prints it; bitwise
- Clean/gantry_dynamic/training.py:342 | lbfgs_outcome [.npz key, final] | compute | reader: evaluation.py:241; run_l3bc.py, gpu_smoke.py | keep | read by the evaluation header; key max_iter inside is a row in D
- Clean/gantry_dynamic/training.py:343 | optimizer_state_dropped [.npz key, final] | write-only | reader: training.py:140 (one print) | remove | (c): the pair holds no 'optimizer' exactly when it is True; the print tests fmt == 'pair' and optim_sd is None; bitwise
- Clean/gantry_dynamic/training.py:343 | start_phase [.npz key, final] | write-only | reader: none in Clean/; gpu_smoke.py | remove | (c): RESUMES 'type' records a polish-only job; bitwise
- Clean/gantry_dynamic/training.py:344 | resumed_from [.npz key, final] | write-only | reader: none; compare_l3bc.py compares the name | remove | (c): RESUMES 'source'; bitwise; compare_l3bc.py adapts
- Clean/gantry_dynamic/training.py:344 | resumed_at_epoch [.npz key, final] | write-only | reader: none in Clean/; run_l3bc.py:250 | keep | the only file record of the epoch a continuation started from, which a user of a resumed folder needs; no change
- Clean/gantry_dynamic/training.py:279 | resumed_at_epoch = -1 sentinel [number] | unlabelled | reader: training.py:344 | keep | "not resumed" marker, not a model number; no change
- Clean/gantry_dynamic/training.py:345 | ps2_summary [.npz key, final] | write-only | reader: user | keep | the only file record of the PS2 path taken (completed or fallback); no change
- Clean/gantry_dynamic/training.py:312 | file names gantry_ckpt_<run_id>, gantry_ckpt_<run_id>_pre_lbfgs [name] | compute | reader: gantry_interconnect_dynamic.py:39, evaluation.py:240 | keep | no change
- Clean/gantry_dynamic/training.py:45 | 'pair' format [accepted format] | compute | reader: polish-only rerun (gantry_interconnect_dynamic.py:39), evaluate (evaluation.py:241) | keep | Clean's own format; no change
- Clean/gantry_dynamic/training.py:52 | 'whole' format (deepSI dump _best.pth / _last.pth) [accepted format] | compute | reader: --resume (gantry_interconnect_dynamic.py:42) | keep | Clean's own resume reads its deepSI dumps (assumption 6.1 settled: yes); the parts below are residue
- Clean/gantry_dynamic/training.py:60 | d.get('epochs_before', 0), d.get('epoch_counter', 0) [dump keys] | compute | reader: training.py:288 | keep | a dump of a fresh run has no epochs_before; no change
- Clean/gantry_dynamic/training.py:62 | d.get('bestfit', inf) [dump key] | compute | reader: training.py:290 | keep | every Clean dump has bestfit, so the default is never taken, but it is cheap; no change
- Clean/gantry_dynamic/training.py:63 | d.get('ps2_opening') [dump key] | compute | reader: training.py:123, :240 | keep | no change
- Clean/gantry_dynamic/training.py:65 | d.get('optimizer') or d.get('_optimizer') [dump keys] | mainline-compat | reader: training.py:66 | literal | 'optimizer' is the key of mainline dumps written before mainline's optimizer property (mainline interconnect.py:1521); Clean dumps hold '_optimizer' only because of that property. With the property removed (row interconnect.py:645), read d.get('optimizer') only; bitwise for new runs; existing Clean test dumps resume with fresh Adam moments (the message says so)
- Clean/gantry_dynamic/training.py:66 | hasattr(optimizer, 'state_dict') guard [branch] | mainline-compat | reader: training.py:66 | remove | a dump's optimizer is always a torch optimizer; bitwise
- Clean/gantry_dynamic/training.py:88 | coordinate_version comparison [guard] | mainline-compat | reader: training.py:90 | keep | actionable guard (row B1 coordinate_version)
- Clean/gantry_dynamic/training.py:96 | missing / extra / shape comparison [guard] | compute | reader: training.py:101 | keep | actionable guard; P3 makes it refuse old checkpoints, which is right, so the converter drops the keys instead of this guard
- Clean/gantry_dynamic/training.py:116 | message "...or the mainline implementation configured for the checkpoint precision" [message] | misleading, mainline-compat | reader: user | comment | "use a float64 checkpoint"; bitwise
- Clean/gantry_dynamic/training.py:124 | refusal of a dump inside an unfinished PS2 opening [guard] | compute | reader: training.py:124 | keep | actionable guard; no change
- Clean/gantry_dynamic/training.py:143 | print clause "whole-system format: deepSI stores a live optimizer, not a state dict" [message branch] | misleading | reader: user | remove | resolve_checkpoint turns the live optimizer into a state dict (:66), so a whole dump reaches this branch only when it has no optimizer at all; the clause states the opposite; bitwise

### B3. Training logs in the run folder (val_hist.npz, loss.npz) and the deepSI dumps
- Clean/gantry_dynamic/training.py:179 | val_hist_event [val_hist.npz key] | compute | reader: training.py:316, :324 (in memory); val_check.py | keep | the selection log; no change
- Clean/gantry_dynamic/training.py:180 | val_hist_update [val_hist.npz key] | compute | reader: training.py:362 (selected update) | keep | no change
- Clean/gantry_dynamic/training.py:181 | val_hist_epoch [val_hist.npz key] | write-only | reader: user | keep | the epoch of each validation, which the update alone does not give across budgets; no change
- Clean/gantry_dynamic/training.py:182 | val_hist_score [val_hist.npz key] | compute | reader: training.py:318, :330 | keep | no change
- Clean/gantry_dynamic/training.py:183 | val_hist_sim_rms [val_hist.npz key] | write-only, mainline-compat | reader: none | remove | deepSI's length-weighted sim-RMS in metres, the mainline validation measure before D-256; with last_sim_rms (row D closed_loop.py:249); bitwise; val_check.py does not read it
- Clean/gantry_dynamic/training.py:184 | val_hist_rms [val_hist.npz key] | write-only | reader: user | keep | per record and axis RMS of every validation, the data behind the score; no change
- Clean/gantry_dynamic/training.py:185 | val_hist_accepted [val_hist.npz key] | write-only | reader: user | keep | marks the polish verdict in the log; no change
- Clean/gantry_dynamic/training.py:186 | val_sigma_e [val_hist.npz key] | write-only | reader: user; capture_l12.py | keep | turns val_hist_rms into NRMS_e; no change
- Clean/gantry_dynamic/training.py:357 | update, train_loss, val_score, selected_update, polish_score, polish_accepted [loss.npz keys] | compute | reader: evaluation.py:199-207 | keep | E4 ruling (Dirk: save the data of the loss curve); no change
- Clean/gantry_dynamic/training.py:358 | epoch, time [loss.npz keys] | write-only | reader: user | keep | E4 ruling; the data of the loss curve against epoch or wall time; no change
- Clean/model_augmentation/fit_systems/interconnect.py:640 | deepSI dump content (__getstate__: __dict__ without obc, training_phase, fit_history) [dump format] | compute | reader: gantry_interconnect_dynamic.py:42 (resume) | keep | Clean's resume source; no change
- Clean/model_augmentation/fit_systems/interconnect.py:480 | ps2_opening = (phase, updates) [dump attribute] | compute | reader: training.py:63 | keep | no change
- Clean/gantry_dynamic/training.py:294 | epochs_before [dump attribute] | compute | reader: training.py:60 | keep | no change
- Clean/gantry_dynamic/training.py:282 | unique_code [deepSI attribute] | compute | reader: deepSI name (interconnect.py:625 self.name) | keep | no change

## C. The `hp` dict keys (`RunConfig.hp`, config.py:199)
- Clean/gantry_dynamic/config.py:200 | NX_ANN [hp key] | compute, mainline-compat | reader: model.py:39, training.py:298 | rename | n_a, the name of every other n_a in Clean/ (K8 renamed the field, not the key); bitwise; nothing reads the key from a file once the .npz hp goes (B2)
- Clean/gantry_dynamic/config.py:201 | n_nodes_per_layer [hp key] | compute | reader: model.py:67, :100, training.py:245 | keep | no change
- Clean/gantry_dynamic/config.py:202 | n_hidden_layers [hp key] | compute | reader: model.py:68, :101 | keep | no change
- Clean/gantry_dynamic/config.py:203 | up_sample [hp key] | compute, fixed-option | reader: model.py:58 | keep | K34 ruled; no change
- Clean/gantry_dynamic/config.py:204 | nf [hp key] | compute | reader: training.py:221, :263, :297 | keep | no change
- Clean/gantry_dynamic/config.py:205 | na_nb [hp key] | compute | reader: model.py:44 | keep | no change
- Clean/gantry_dynamic/config.py:206 | batch_size [hp key] | compute | reader: training.py:260 | keep | no change
- Clean/gantry_dynamic/config.py:207 | lr [hp key] | compute | reader: model.py:111 | keep | no change
- Clean/gantry_dynamic/config.py:208 | epochs [hp key] | compute | reader: training.py:255, :296, :309 | keep | no change

## D. The other files: constants, attributes, names, comments, branches, numbers

### D1. `gantry_interconnect_dynamic.py`
- Clean/gantry_interconnect_dynamic.py:22 | EVALUATE = True [module constant] | compute | reader: gantry_interconnect_dynamic.py:74, :184 | keep | amendment 9 point 3 (a parameter at the top of the script); no change
- Clean/gantry_interconnect_dynamic.py:23 | EVAL_RUN_DIR = None [module constant] | compute | reader: gantry_interconnect_dynamic.py:152 | keep | same ruling; no change
- Clean/gantry_interconnect_dynamic.py:28 | dump name 'AugmentedModel_<run_id><_best/_last>.pth' [name] | compute | reader: gantry_interconnect_dynamic.py:42, :61 | keep | must match deepSI's name (class name + unique_code); no change
- Clean/gantry_interconnect_dynamic.py:54 | rerun_id from SLURM_JOB_ID or the time [option] | compute | reader: gantry_interconnect_dynamic.py:55 | keep | names the archive of a rerun; no change
- Clean/gantry_interconnect_dynamic.py:81 | run_id from SLURM_JOB_ID or the time [option] | compute | reader: gantry_interconnect_dynamic.py:82 | keep | no change
- Clean/gantry_interconnect_dynamic.py:80 | fresh / rerun branch (rerun is None) [branch] | compute | reader: gantry_interconnect_dynamic.py:80 | keep | no change
- Clean/gantry_interconnect_dynamic.py:91 | polish-only sets start_phase='lbfgs' [branch] | compute | reader: gantry_interconnect_dynamic.py:92 | keep | no change
- Clean/gantry_interconnect_dynamic.py:95 | print ' g_aug=mlp/live' [name] | mainline-compat | reader: log | remove | mainline's ann_arch/g_init values, not options of Clean/; bitwise; log text only
- Clean/gantry_interconnect_dynamic.py:125 | encoder start or restored checkpoint (src is None) [branch] | compute | reader: gantry_interconnect_dynamic.py:125 | keep | no change
- Clean/gantry_interconnect_dynamic.py:137 | evaluate_run branch [branch] | compute | reader: gantry_interconnect_dynamic.py:137 | keep | amendment 9 point 3; no change
- Clean/gantry_interconnect_dynamic.py:146 | preset [CLI argument] | compute | reader: gantry_interconnect_dynamic.py:172 | keep | a user selects the run; no change
- Clean/gantry_interconnect_dynamic.py:147 | noise [CLI argument] | compute | reader: gantry_interconnect_dynamic.py:172 | keep | no change
- Clean/gantry_interconnect_dynamic.py:148 | --cpu [CLI flag] | compute | reader: gantry_interconnect_dynamic.py:182 | keep | no change
- Clean/gantry_interconnect_dynamic.py:150 | --resume [CLI flag] | compute | reader: gantry_interconnect_dynamic.py:170, :181 | keep | no change
- Clean/gantry_interconnect_dynamic.py:151 | --polish-only [CLI flag] | compute | reader: gantry_interconnect_dynamic.py:170 | keep | no change
- Clean/gantry_interconnect_dynamic.py:152 | --evaluate [CLI flag] | compute | reader: gantry_interconnect_dynamic.py:163 | keep | amendment 9 point 3 (the CLI flag may stay); no change
- Clean/gantry_interconnect_dynamic.py:154 | --no-evaluate [CLI flag] | compute | reader: gantry_interconnect_dynamic.py:184 | keep | no change
- Clean/gantry_interconnect_dynamic.py:155 | --from best/last [CLI flag] | compute | reader: gantry_interconnect_dynamic.py:181 | keep | no change
- Clean/gantry_interconnect_dynamic.py:157 | --closed-form [CLI flag] | compute | reader: gantry_interconnect_dynamic.py:173 | keep | amendment 3 toggle; no change
- Clean/gantry_interconnect_dynamic.py:160 | --data-root [CLI flag] | compute | reader: gantry_interconnect_dynamic.py:174, :180 | keep | no change
- Clean/gantry_interconnect_dynamic.py:161 | --out-root [CLI flag] | compute | reader: gantry_interconnect_dynamic.py:174, :180 | keep | no change
- Clean/gantry_interconnect_dynamic.py:164 | --evaluate with preset/noise refused [guard] | compute | reader: gantry_interconnect_dynamic.py:165 | keep | actionable guard; no change
- Clean/gantry_interconnect_dynamic.py:168 | --from without --resume refused [guard] | compute | reader: gantry_interconnect_dynamic.py:169 | keep | no change
- Clean/gantry_interconnect_dynamic.py:177 | preset/noise/--closed-form with a run folder refused [guard] | compute | reader: gantry_interconnect_dynamic.py:178 | keep | no change

### D2. `gantry_dynamic/__init__.py`
- Clean/gantry_dynamic/__init__.py:10 | _here [module constant] | verification | reader: gantry_dynamic/__init__.py:12 | keep | part of the import guard below
- Clean/gantry_dynamic/__init__.py:11 | _ma_root [module constant] | verification | reader: gantry_dynamic/__init__.py:12 | keep | part of the import guard
- Clean/gantry_dynamic/__init__.py:12 | guard: model_augmentation must come from this folder [guard] | verification, mainline-compat | reader: import | keep | refuses a mixed import with an actionable message; needed while Clean/ lives beside a second model_augmentation; no change
- Clean/gantry_dynamic/__init__.py:8 | comment "...would mix it with the mainline one" [comment] | mainline-compat | reader: none | comment | "with another model_augmentation on sys.path"; bitwise

### D3. `gantry_dynamic/data.py`
- Clean/gantry_dynamic/data.py:23 | DATASETS [module constant] | compute | reader: data.py:82 | keep | the data set folder names; no change
- Clean/gantry_dynamic/data.py:25 | _THESIS_SPLIT [module constant] | compute | reader: data.py:35 | keep | the manifest block to split map; no change
- Clean/gantry_dynamic/data.py:27 | SOS [module constant] | compute | reader: data.py:124 | keep | K41; no change
- Clean/gantry_dynamic/data.py:37 | controller == 'K1' row filter [branch] | compute | reader: data.py:37 | keep | selects the records of the thesis split (11 unit-gain test records per data set are other controllers); no change
- Clean/gantry_dynamic/data.py:45 | stored controller fields name, design_Y, gain, fbw_Hz [record format keys] | compute | reader: data.py:47 | keep | read from every record; no change
- Clean/gantry_dynamic/data.py:54 | train_list [DataBundle field] | compute | reader: data.py:215, ps2_opening.py:79, evaluation.py:265 | keep | no change
- Clean/gantry_dynamic/data.py:55 | val_list [DataBundle field] | compute | reader: evaluation.py:265 (getattr split + '_list') | keep | K2 not implemented (amendment 9 point 4); the evaluation reads it; no change
- Clean/gantry_dynamic/data.py:56 | test_list [DataBundle field] | compute | reader: evaluation.py:265 | keep | no change
- Clean/gantry_dynamic/data.py:57 | train_data [DataBundle field] | compute | reader: training.py:220, :259, model.py:111 | keep | no change
- Clean/gantry_dynamic/data.py:58 | val_ckpt_data [DataBundle field] | compute | reader: training.py:211, :222, :259 | keep | the validation records of checkpoint selection; no change
- Clean/gantry_dynamic/data.py:59 | train_files [DataBundle field] | compute | reader: gantry_interconnect_dynamic.py:118, obc_gantry.py:249 | keep | no change
- Clean/gantry_dynamic/data.py:60 | val_files [DataBundle field] | compute | reader: gantry_interconnect_dynamic.py:118, evaluation.py:247 | keep | no change
- Clean/gantry_dynamic/data.py:61 | test_files [DataBundle field] | compute | reader: evaluation.py:247 | keep | no change
- Clean/gantry_dynamic/data.py:62 | e_train [DataBundle field] | compute | reader: controller.py:237 | keep | no change
- Clean/gantry_dynamic/data.py:63 | e_val [DataBundle field] | compute | reader: controller.py:235 | keep | no change
- Clean/gantry_dynamic/data.py:64 | e_test [DataBundle field] | compute | reader: evaluation.py:265 | keep | no change
- Clean/gantry_dynamic/data.py:70 | x_mean [Norm field] | compute | reader: model.py:58, obc_gantry.py:72 | keep | no change
- Clean/gantry_dynamic/data.py:71 | std_x [Norm field] | compute | reader: model.py:57, data.py:241 | keep | no change
- Clean/gantry_dynamic/data.py:72 | std_u [Norm field] | compute | reader: model.py:57, controller.py:215 | keep | no change
- Clean/gantry_dynamic/data.py:73 | u_mean [Norm field] | compute | reader: model.py:58, :90 | keep | no change
- Clean/gantry_dynamic/data.py:74 | ystd [Norm field] | compute | reader: model.py:93, controller.py:237 | keep | no change
- Clean/gantry_dynamic/data.py:75 | y0 [Norm field] | compute | reader: model.py:92 | keep | no change
- Clean/gantry_dynamic/data.py:76 | Cd_norm [Norm field] | compute | reader: model.py:60 | keep | no change
- Clean/gantry_dynamic/data.py:77 | Dd_np [Norm field] | compute, fixed-option | reader: model.py:60 (always zero) | keep | K40 ruled; no new evidence
- Clean/gantry_dynamic/data.py:120 | noise != 'lowpass' point-sampling branch [branch] | compute | reader: data.py:120 | keep | the noise-free data set; no change
- Clean/gantry_dynamic/data.py:135 | truncate_s branch [branch] | verification | reader: data.py:135 | keep | with the truncate_s field (A3)
- Clean/gantry_dynamic/data.py:164 | servo sigma finite and positive [guard] | compute | reader: data.py:164 | keep | actionable guard; no change
- Clean/gantry_dynamic/data.py:218 | fs = 1.0 / data.train_list[0].dt [duplicate source] | compute | reader: data.py:226 | literal | cfg.fs_new; bitwise only if 1.0 / (1.0 / 4000) == 4000.0 in float64, a one-expression check before the change
- Clean/gantry_dynamic/data.py:219 | P_inv_T [local constant] | compute | reader: data.py:223 | literal | the same expression as obc_gantry.P_INV_T64; move P_INV_T64 to data.py (obc_gantry imports data already) and use it in both; bitwise
- Clean/gantry_dynamic/data.py:225 | "# THEORY: var of a first difference of white noise = 2*sigma_n^2, so sigma_v = sigma_n*sqrt(2)*fs" [comment] | misleading, mainline-compat | reader: none | remove | a remnant of mainline's commented-out SNR velocity scale (scripts/gantry/gantry_dynamic/data.py:351); no sigma_n or sigma_v is computed here; keep the line above it (the backward difference); bitwise
- Clean/gantry_dynamic/data.py:234 | + 1e-8 on std_x [number] | unlabelled | reader: data.py:234 | label | "# HEURISTIC: keeps T_x finite for a constant channel" (the comment at :232 says it, without the label); bitwise
- Clean/gantry_dynamic/data.py:235 | + 1e-8 on std_u [number] | unlabelled | reader: data.py:235 | label | same label; bitwise
- Clean/gantry_dynamic/data.py:237 | + 1e-8 on ystd [number] | unlabelled | reader: data.py:237 | label | same label; bitwise

### D4. `gantry_dynamic/controller.py`
- Clean/gantry_dynamic/controller.py:18 | COMPILE_BACKEND = 'inductor' [module constant] | compute | reader: controller.py:228 | keep | the compile backend; no change
- Clean/gantry_dynamic/controller.py:21 | mb [module constant] | compute | reader: controller.py:48 | keep | THEORY gtd_config.m; float64 copy of gantry_ss.mb for the MATLAB double design; no change
- Clean/gantry_dynamic/controller.py:21 | mh [module constant] | compute | reader: controller.py:48, :50 | keep | same
- Clean/gantry_dynamic/controller.py:21 | m1 [module constant] | compute | reader: controller.py:48 | keep | same
- Clean/gantry_dynamic/controller.py:21 | m2 [module constant] | compute | reader: controller.py:48 | keep | same
- Clean/gantry_dynamic/controller.py:21 | Jb [module constant] | compute | reader: controller.py:50 | keep | same
- Clean/gantry_dynamic/controller.py:21 | Jh [module constant] | compute | reader: controller.py:50 | keep | same
- Clean/gantry_dynamic/controller.py:22 | cg1 [module constant] | compute | reader: controller.py:30 | keep | same
- Clean/gantry_dynamic/controller.py:22 | cg2 [module constant] | compute | reader: controller.py:30 | keep | same
- Clean/gantry_dynamic/controller.py:22 | cy [module constant] | compute | reader: controller.py:32 | keep | same
- Clean/gantry_dynamic/controller.py:22 | cb1 [module constant] | compute | reader: controller.py:31 | keep | same
- Clean/gantry_dynamic/controller.py:22 | cb2 [module constant] | compute | reader: controller.py:31 | keep | same
- Clean/gantry_dynamic/controller.py:23 | kb1 [module constant] | compute | reader: controller.py:33 | keep | same
- Clean/gantry_dynamic/controller.py:23 | kb2 [module constant] | compute | reader: controller.py:33 | keep | same
- Clean/gantry_dynamic/controller.py:23 | Lb [module constant] | compute | reader: controller.py:29, :48 | keep | same
- Clean/gantry_dynamic/controller.py:23 | d [module constant] | compute | reader: controller.py:50 | keep | same
- Clean/gantry_dynamic/controller.py:25 | FBW = 100.0 [module constant] | compute | reader: controller.py:26, :146 | keep | labelled THEORY; no change
- Clean/gantry_dynamic/controller.py:26 | W [module constant] | compute | reader: controller.py:84, :102 | keep | no change
- Clean/gantry_dynamic/controller.py:29 | P [module constant] | compute | reader: controller.py:71 | keep | labelled THEORY; no change
- Clean/gantry_dynamic/controller.py:30 | C_DAMP [module constant] | compute | reader: controller.py:68 | keep | no change
- Clean/gantry_dynamic/controller.py:33 | K_STIFF [module constant] | compute | reader: controller.py:68 | keep | no change
- Clean/gantry_dynamic/controller.py:84 | 10, 6, 3 in the controller shape [numbers] | compute | reader: controller.py:84, :85 | keep | labelled by the docstring THEORY ruleOfThumb.m; no change
- Clean/gantry_dynamic/controller.py:141 | _CONTROLLER_METADATA [module constant, registry] | compute | reader: controller.py:153 | keep | K4 ruled; no change
- Clean/gantry_dynamic/controller.py:146 | 1e-9 bandwidth match [number] | unlabelled | reader: controller.py:146 | label | "# HEURISTIC: equality of stored MATLAB doubles"; bitwise
- Clean/gantry_dynamic/controller.py:161 | 1e-12 unit-gain test [number] | unlabelled | reader: controller.py:161 | label | same label; bitwise
- Clean/gantry_dynamic/controller.py:161 | gain refusal unless use_gain [guard] | compute, fixed-option | reader: controller.py:161 | keep | K3 ruled (use_gain stays); no new evidence beyond A3 controller_gain
- Clean/gantry_dynamic/controller.py:188 | (uniq if use_gain else design Y only) return branch [branch] | fixed-option | reader: controller.py:220 (print) | keep | K3 ruled
- Clean/gantry_dynamic/controller.py:202 | docstring "val_data: unused, kept for callers" [comment] | misleading, verification | reader: none | remove | with the parameter (E, build_closed_loop val_data); bitwise
- Clean/gantry_dynamic/controller.py:219 | verbose print branch [branch] | verification | reader: controller.py:219 | keep | K6 ruled; l4_loader passes verbose=False
- Clean/gantry_dynamic/controller.py:227 | compiled step when compile_mode is set [branch] | compute | reader: controller.py:227 | keep | no change

### D5. `gantry_dynamic/model.py`
- Clean/gantry_dynamic/model.py:25 | PHYSICAL_STATE_IX = np.arange(6) [module constant] | compute | reader: model.py:81, :83, :84 | keep | the named physical rows of the wiring; no change
- Clean/gantry_dynamic/model.py:71 | activation=torch.nn.Tanh [option] | compute | reader: blocks.py:58 | keep | the only place that sets the activation; ANN_ACTIVATION (A4) goes
- Clean/gantry_dynamic/model.py:107 | CUDA requested but unavailable [guard] | compute | reader: model.py:107 | keep | actionable guard; no change
- Clean/gantry_dynamic/model.py:113 | Adam eps after construction [guard] | compute | reader: model.py:113 | keep | K39 ruled
- Clean/gantry_dynamic/model.py:119 | attach_obc on the OBC arm [branch] | compute | reader: model.py:119 | keep | no change
- Clean/gantry_dynamic/model.py:128 | on_device CUDA branches [branch] | compute | reader: model.py:128, :133 | keep | no change
- Clean/gantry_dynamic/model.py:207 | _MAX_ENCODER_UNEXPLAINED_FRACTION = 1e-3 [module constant] | compute | reader: model.py:269 | keep | labelled HEURISTIC; no change
- Clean/gantry_dynamic/model.py:237 | zero regressor std set to 1 [branch] | compute | reader: model.py:237 | keep | no change
- Clean/gantry_dynamic/model.py:266 | atol = 1e-6 x min std [number] | unlabelled | reader: model.py:266 | label | "# HEURISTIC: write-back check tolerance"; bitwise

### D6. `gantry_dynamic/obc_gantry.py`
- Clean/gantry_dynamic/obc_gantry.py:30 | P_INV_T64 [module constant] | compute | reader: model.py:176, obc_gantry.py:84, evaluation.py:35 | keep | the one float64 P^-T (D3 makes data.py use it too); no change
- Clean/gantry_dynamic/obc_gantry.py:36 | Z_step [GantryReferenceSet field] | compute | reader: obc_gantry.py:212 | keep | no change
- Clean/gantry_dynamic/obc_gantry.py:37 | Z_field [GantryReferenceSet field] | compute | reader: obc_gantry.py:237 | keep | no change
- Clean/gantry_dynamic/obc_gantry.py:72 | reshape(1, 6), reshape(1, 3) [numbers] | unlabelled | reader: obc_gantry.py:72-75 | keep | state and input dimensions, not tuning numbers; no change
- Clean/gantry_dynamic/obc_gantry.py:86 | k = np.arange(2, len(u) - 3) [tuple range] | mainline-compat, misleading | reader: obc_gantry.py:87 | comment | the - 3 drops the last stencil tuple, which mainline's D-252 successor needed; Clean/ builds no successor but keeps it to keep the thesis tuple set; say so (the docstring's "qdot exists at i and i + 1" states the set, not why); bitwise
- Clean/gantry_dynamic/obc_gantry.py:123 | _phy (list, not a submodule) [attribute] | compute | reader: obc_gantry.py:140, :158, :177 | keep | keeps the physics block out of the correction's state_dict; no change
- Clean/gantry_dynamic/obc_gantry.py:124 | nx, nx_phys [attributes] | compute | reader: obc_gantry.py:126, :130, :174 | keep | no change
- Clean/gantry_dynamic/obc_gantry.py:126 | row_ix [attribute] | compute, fixed-option | reader: obc_gantry.py:131, :256 | keep | the protected rows; with K32 it is tuple(range(nx_phys)); no change
- Clean/gantry_dynamic/obc_gantry.py:127 | field_cols [attribute] | compute, fixed-option | reader: obc_gantry.py:255 | keep | with K32 equal to row_ix; no change
- Clean/gantry_dynamic/obc_gantry.py:207 | builder moves blk64 to the device [branch] | compute | reader: obc_gantry.py:207 | keep | K17 ruled
- Clean/gantry_dynamic/obc_gantry.py:236 | reference-input cache per device and dtype [branch] | compute | reader: obc_gantry.py:236 | keep | no change
- Clean/gantry_dynamic/obc_gantry.py:259 | attach print 'space=%s, refresh=%s' [print] | fixed-option | reader: log | literal | with A3 obc_space, obc_refresh; bitwise

### D7. `gantry_dynamic/training.py`
- Clean/gantry_dynamic/training.py:159 | fit_sys, hist_path, val_hist [ValidationHistory attributes] | compute | reader: training.py:173, :186, :316 | keep | no change
- Clean/gantry_dynamic/training.py:164 | events adam, rebuilt, polish_candidate, selected, final [names] | compute | reader: training.py:316, :324, :326 | keep | the selection rule reads them; no change
- Clean/gantry_dynamic/training.py:170 | getattr(fit_sys, 'batch_counter', -1), epoch_counter nan [defaults] | compute | reader: training.py:170 | keep | a polish-only job records before any fit; no change
- Clean/gantry_dynamic/training.py:202 | empty_cache on CUDA [branch] | compute | reader: training.py:202 | keep | no change
- Clean/gantry_dynamic/training.py:208 | eager validation when no Adam ran on the unconstrained arm [branch] | compute | reader: training.py:208 | keep | completes the lazy initialisation before the compiled probe; no change
- Clean/gantry_dynamic/training.py:212 | rebuilt-basis validation on the OBC arm [branch] | compute | reader: training.py:212 | keep | no change
- Clean/gantry_dynamic/training.py:241 | PS2 start rule (no record: done_epochs == 0; else phase != 'S3') [branch] | compute | reader: training.py:241 | keep | no change
- Clean/gantry_dynamic/training.py:251 | polish-only and spent-budget branches [branch] | compute | reader: training.py:251, :253 | keep | no change
- Clean/gantry_dynamic/training.py:262 | its_per_val = n_its when n_its is set [branch] | verification | reader: training.py:262 | keep | one validation at the end of a run capped in updates; goes with the n_its field (A3)
- Clean/gantry_dynamic/training.py:318 | tie rule: latest of equal scores [branch] | compute | reader: training.py:318 | keep | matches fit's >= at interconnect.py:415; no change
- Clean/gantry_dynamic/training.py:326 | final row: accepted polish, else rebuilt, else selected [branch] | compute | reader: training.py:326 | keep | no change
- Clean/gantry_dynamic/training.py:337 | drop_optimizer_state [flag] | compute | reader: training.py:340 | keep | selects include_optimizer; its .npz copy goes (B2); no change
- Clean/gantry_dynamic/training.py:354 | getattr(fit_sys, 'fit_history', {}) if adam_ran [branch] | compute | reader: training.py:357 | keep | no change

### D8. `gantry_dynamic/evaluation.py`
- Clean/gantry_dynamic/evaluation.py:29 | SPLITS [module constant] | compute | reader: evaluation.py:247, :264 | keep | no change
- Clean/gantry_dynamic/evaluation.py:31 | MODELS (keys, labels, colours) [module constant] | compute | reader: evaluation.py:136, :156, :175 | keep | approved mocks (amendment 9 point 3); no change
- Clean/gantry_dynamic/evaluation.py:34 | FRAMES [module constant] | compute | reader: evaluation.py:120, :279 | keep | no change
- Clean/gantry_dynamic/evaluation.py:37 | TEX [module constant] | compute | reader: evaluation.py:159, :177 | keep | no change
- Clean/gantry_dynamic/evaluation.py:53 | obc_correction None branch [branch] | compute | reader: evaluation.py:56 | keep | no change
- Clean/gantry_dynamic/evaluation.py:92 | noise-free data set: no twins [branch] | compute | reader: evaluation.py:266 | keep | amendment 9 implementation choice 3; no change
- Clean/gantry_dynamic/evaluation.py:124 | stage RMS from the scorer, generalised from the predictions [branch] | compute | reader: evaluation.py:124 | keep | no change
- Clean/gantry_dynamic/evaluation.py:139 | table widths ('%-7s', '%-20s', '-' * 97, '%6.4f') [numbers] | unlabelled | reader: evaluation.py:139-145 | keep | layout of the approved terminal table, not a model number; no change
- Clean/gantry_dynamic/evaluation.py:151 | _bar_figure layout numbers (figsize 7, 0.5, 2.5; offset 0.6; width 0.27; axvline 0.2, lw 0.8; alpha 0.3; rotation 45; legend 0.95; rect 0.92; dpi 110) [numbers] | unlabelled | reader: evaluation.py:151-166 | keep | figure layout, read by the figure code; the label rule covers model and identification numbers; no change
- Clean/gantry_dynamic/evaluation.py:172 | _record_figure layout numbers (figsize 10, 7.8; height ratio 0.6; lw 0.6, 1; text x 1.008; fontsize 7.5; alpha 0.3; rect 0.88; dpi 110) [numbers] | unlabelled | reader: evaluation.py:172-192 | keep | same
- Clean/gantry_dynamic/evaluation.py:197 | _loss_figure layout numbers (figsize 8, 5.5; marker sizes 50, 70; zorder 5; alpha 0.3; dpi 110) [numbers] | unlabelled | reader: evaluation.py:197-215 | keep | same
- Clean/gantry_dynamic/evaluation.py:205 | polish marker branches (polish_score.size, upd.size) [branch] | compute | reader: evaluation.py:205, :206 | keep | no change
- Clean/gantry_dynamic/evaluation.py:253 | print widths '=' * 100, '-' * 66, '%12.5g' [numbers] | unlabelled | reader: evaluation.py:253, :259, :304 | keep | layout; no change
- Clean/gantry_dynamic/evaluation.py:256 | str(cfg.dtype_pt)[6:] [number] | unlabelled | reader: evaluation.py:256 | literal | 'float64' via str(cfg.dtype_pt).replace('torch.', ''); bitwise; print only
- Clean/gantry_dynamic/evaluation.py:300 | fourth value of estimation_parameters ('identifiable combinations') discarded as _ [return value] | write-only, mainline-compat | reader: none | remove | mainline's estimation_parameters tells the raw from the reduced parameterisation; Clean/ has one; return three values (row D10 blocks.py:703); bitwise
- Clean/gantry_dynamic/evaluation.py:312 | '%.17g' [number] | unlabelled | reader: evaluation.py:312, :317 | label | "# THEORY: 17 significant digits round-trip IEEE 754 binary64" (the comment at :308 says it, without the label); bitwise
- Clean/gantry_dynamic/evaluation.py:312 | r[:5] key columns [number] | unlabelled | reader: evaluation.py:312 | keep | the five text columns of scores.csv; no change
- Clean/gantry_dynamic/evaluation.py:311 | scores.csv columns split, record, frame, model, axis, rms, nrms_e, sigma_e, floor_nrms_e [saved data] | write-only | reader: user; val_check.py, r1_check.py | keep | approved output (amendment 9 point 3); no change
- Clean/gantry_dynamic/evaluation.py:313 | predictions.npz keys t, r, e, e_noise_free, e_hat_<model>, k0, P_inv_T [saved data] | write-only | reader: user | keep | approved output; no change
- Clean/gantry_dynamic/evaluation.py:316 | params.csv columns name, true, start, learned [saved data] | write-only | reader: user | keep | approved output; no change
- Clean/gantry_dynamic/evaluation.py:319 | evaluation.json keys checkpoint, run_id, git [saved data] | write-only | reader: user | keep | provenance of the evaluation (E1 moved the val check out); no change
- Clean/gantry_dynamic/evaluation.py:321 | loss.png only if loss.npz exists [branch] | compute | reader: evaluation.py:321 | keep | folders written before E4 have no loss.npz; no change

### D9. `model_augmentation/fit_systems/closed_loop.py`
- Clean/model_augmentation/fit_systems/closed_loop.py:40 | nc, nu [ControllerBank attributes] | compute | reader: closed_loop.py:68, :122 | keep | no change
- Clean/model_augmentation/fit_systems/closed_loop.py:116 | bank follows the data device [branch] | compute | reader: closed_loop.py:117 | keep | no change
- Clean/model_augmentation/fit_systems/closed_loop.py:142 | bank [ClosedLoopSimulator attribute] | compute | reader: closed_loop.py:188, model.py:189 | keep | no change
- Clean/model_augmentation/fit_systems/closed_loop.py:144 | extra_array_names ('ctrl_ix', 'loss_w') [attribute] | compute | reader: interconnect.py:462, lbfgs_polish.py:72 | keep | no change
- Clean/model_augmentation/fit_systems/closed_loop.py:146 | val_sigma_e [attribute] | compute | reader: closed_loop.py:251, training.py:186 | keep | no change
- Clean/model_augmentation/fit_systems/closed_loop.py:148 | _compiled [attribute] | compute | reader: closed_loop.py:183, lbfgs_polish.py:214 | keep | no change
- Clean/model_augmentation/fit_systems/closed_loop.py:149 | train_ctrl_rows [attribute] | compute | reader: closed_loop.py:205, model.py:190, ps2_opening.py:83 | keep | no change
- Clean/model_augmentation/fit_systems/closed_loop.py:151 | record_weights [attribute] | compute | reader: closed_loop.py:212 | keep | no change
- Clean/model_augmentation/fit_systems/closed_loop.py:152 | val_rows [attribute] | compute | reader: closed_loop.py:244 | keep | no change
- Clean/model_augmentation/fit_systems/closed_loop.py:183 | compiled step for gradient-enabled CUDA rollouts [branch] | compute | reader: closed_loop.py:184, :192 | keep | no change
- Clean/model_augmentation/fit_systems/closed_loop.py:199 | kw.get('nf', 25), kw.get('stride', 1) [defaults, numbers] | fixed-option, unlabelled | reader: closed_loop.py:203 | keep | K38 ruled; new detail only: 25 is no setting of any Clean run (nf = 400), so a default would silently misalign if a caller forgot nf; Dirk may revisit with kw['nf']
- Clean/model_augmentation/fit_systems/closed_loop.py:206 | window count alignment [guard] | compute | reader: closed_loop.py:206 | keep | actionable guard; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:225 | window content check [guard] | compute | reader: closed_loop.py:225 | keep | no change
- Clean/model_augmentation/fit_systems/closed_loop.py:245 | last_per_channel [attribute] | compute | reader: closed_loop.py:251, training.py:168 | keep | no change
- Clean/model_augmentation/fit_systems/closed_loop.py:249 | last_sim_rms and its THEORY comment (deepSI multi-record sim-RMS) [attribute] | write-only, mainline-compat | reader: training.py:169 (into val_hist_sim_rms only) | remove | with val_hist_sim_rms (B3); bitwise; the score does not use it
- Clean/model_augmentation/fit_systems/closed_loop.py:274 | np_dtype float32 branch [branch] | fixed-option | reader: closed_loop.py:284 | literal | the model is always float64 (dtype_pt); np_dtype = np.float64; bitwise
- Clean/model_augmentation/fit_systems/closed_loop.py:278 | one controller row per record [guard] | compute | reader: closed_loop.py:278 | keep | K22 ruled
- Clean/model_augmentation/fit_systems/closed_loop.py:317 | return_pred branches [branch] | compute | reader: closed_loop.py:317, :319 | keep | K21 ruled; the evaluation now uses it (evaluation.py:117)

### D10. `model_augmentation/fit_systems/blocks.py` (Jan's code except the @added classes)
- Clean/model_augmentation/fit_systems/blocks.py:12 | "# CHANGED: restricted to the thesis training path; full implementation in model_augmentation/fit_systems/blocks.py at the repository root" [comment, Jan's code] | misleading, mainline-compat | reader: none | comment | a copy of Clean/ for ASMPT has no repository root above it; "full implementation: Jan's public code, commit 8200923 (model_augmentation/__init__.py)"; bitwise
- Clean/model_augmentation/fit_systems/blocks.py:144 | Ts [attribute] | compute | reader: blocks.py:198, :368 | keep | no change
- Clean/model_augmentation/fit_systems/blocks.py:145 | up_sample [attribute] | compute, fixed-option | reader: blocks.py:198, :199 | keep | K34 ruled
- Clean/model_augmentation/fit_systems/blocks.py:146 | nx = 6 [attribute, number] | compute | reader: blocks.py:494, :683 | keep | state dimension; no change
- Clean/model_augmentation/fit_systems/blocks.py:142 | nz=9, nw=6 [numbers] | compute | reader: interconnect.py:253 | keep | block sizes; no change
- Clean/model_augmentation/fit_systems/blocks.py:148 | closed_form [attribute] | compute | reader: blocks.py:254, :332 | keep | the baseline_form option; no change
- Clean/model_augmentation/fit_systems/blocks.py:254 | closed-form / LFR output equation branches [branch] | compute | reader: blocks.py:254, :332 | keep | amendment 3 toggle; no change
- Clean/model_augmentation/fit_systems/blocks.py:259 | comment "kept because the thesis results used it" [comment] | compute | reader: none | keep | states why the default path is the redundant LFR form: it reproduces the thesis checkpoints bitwise; true of this code; no change
- Clean/model_augmentation/fit_systems/blocks.py:389 | PARAM_NAMES [ClassVar] | compute | reader: model.py:51 | keep | no change
- Clean/model_augmentation/fit_systems/blocks.py:506 | COMBO_NAMES [ClassVar] | compute | reader: blocks.py:705 | keep | no change
- Clean/model_augmentation/fit_systems/blocks.py:508 | M_TOTAL_IX = 6 [ClassVar] | compute | reader: blocks.py:612 | keep | named index; no change
- Clean/model_augmentation/fit_systems/blocks.py:509 | M_DIFF_IX = 7 [ClassVar] | compute | reader: blocks.py:541, :625, :644 | keep | named index; no change
- Clean/model_augmentation/fit_systems/blocks.py:510 | J_EFF_IX = 8 [ClassVar] | compute | reader: blocks.py:612 | keep | named index; no change
- Clean/model_augmentation/fit_systems/blocks.py:512 | M_DIFF_RHO = 0.99 [ClassVar] | compute | reader: blocks.py:611 | keep | labelled HEURISTIC; no change
- Clean/model_augmentation/fit_systems/blocks.py:513 | COORDINATE_VERSION = 2 [ClassVar] | mainline-compat | reader: blocks.py:545 | keep | with the coordinate_version guard (B1)
- Clean/model_augmentation/fit_systems/blocks.py:586 | raw_init[10], raw_init[11], raw_init[12] [numbers] | unlabelled | reader: blocks.py:586, :587 | rename | PARAM_NAMES.index('mb'), ('Jb'), ('Jh') as class constants beside M_TOTAL_IX; bitwise
- Clean/model_augmentation/fit_systems/blocks.py:647 | finite and positive combinations [guard] | compute | reader: blocks.py:647 | keep | actionable guard; no change
- Clean/model_augmentation/fit_systems/blocks.py:652 | m_diff inside the bound [guard] | compute | reader: blocks.py:652 | keep | no change
- Clean/model_augmentation/fit_systems/blocks.py:703 | estimation_parameters fourth value "identifiable combinations" [return value] | write-only, mainline-compat | reader: none (evaluation.py:300 discards it) | remove | return (names, start, current); bitwise

### D11. `model_augmentation/fit_systems/interconnect.py` (Jan's code except the @added items)
- Clean/model_augmentation/fit_systems/interconnect.py:2 | "# CHANGED: ... full implementation ... at the repository root" [comment, Jan's code] | misleading, mainline-compat | reader: none | comment | as blocks.py:12; bitwise
- Clean/model_augmentation/fit_systems/interconnect.py:30 | nu, ny, nx [attributes, Jan's code] | compute | reader: interconnect.py:69, :91, :196 | keep | no change
- Clean/model_augmentation/fit_systems/interconnect.py:34 | nr_blocks [attribute, Jan's code] | compute | reader: interconnect.py:122, :250 | keep | no change
- Clean/model_augmentation/fit_systems/interconnect.py:35 | connected_blocks [attribute, Jan's code] | compute | reader: interconnect.py:113, :164 | keep | no change
- Clean/model_augmentation/fit_systems/interconnect.py:37 | input_signal_sizes [attribute, Jan's code] | compute | reader: interconnect.py:224, :238 | keep | no change
- Clean/model_augmentation/fit_systems/interconnect.py:38 | output_signal_sizes [attribute, Jan's code] | write-only | reader: none (interconnect.py:254 appends only) | remove | Jan's code: Dirk decides; bitwise
- Clean/model_augmentation/fit_systems/interconnect.py:40 | signal_connections [attribute, Jan's code] | compute | reader: interconnect.py:132 | keep | no change
- Clean/model_augmentation/fit_systems/interconnect.py:42 | initialized_forward_function [attribute, Jan's code] | compute | reader: interconnect.py:75, :198 | keep | K46 ruled
- Clean/model_augmentation/fit_systems/interconnect.py:44 | obc_correction [attribute] | compute | reader: interconnect.py:87, :559, :580 | keep | no change
- Clean/model_augmentation/fit_systems/interconnect.py:72 | assert x.size(0) == u.size(0) [guard, Jan's code] | compute | reader: interconnect.py:72 | keep | no change
- Clean/model_augmentation/fit_systems/interconnect.py:87 | f_obc subtraction on the OBC arm [branch] | compute | reader: interconnect.py:87 | keep | no change
- Clean/model_augmentation/fit_systems/interconnect.py:114 | block that takes pass_mats [branch] | compute | reader: interconnect.py:114 | keep | no change
- Clean/model_augmentation/fit_systems/interconnect.py:122 | n_output_signals, n_input_signals [attributes, Jan's code] | compute | reader: interconnect.py:99, :101, :148 | keep | no change
- Clean/model_augmentation/fit_systems/interconnect.py:124 | n_nodes = 4 + nr_blocks [number, Jan's code] | compute | reader: interconnect.py:129 | keep | node count of Jan's graph; no change
- Clean/model_augmentation/fit_systems/interconnect.py:126 | array_connection_matrices, output_ix_sorted_input_ix_dependencies [attributes, Jan's code] | compute | reader: interconnect.py:104, :105 | keep | no change
- Clean/model_augmentation/fit_systems/interconnect.py:145 | order_output_signal_computation [attribute, Jan's code] | compute | reader: interconnect.py:80, :181 | keep | no change
- Clean/model_augmentation/fit_systems/interconnect.py:161 | output_cone, _pass_mats_ix [attributes] | compute | reader: interconnect.py:202, :114 | keep | no change
- Clean/model_augmentation/fit_systems/interconnect.py:227 | empty connection matrix means identity [branch, Jan's code] | compute | reader: interconnect.py:227 | keep | build_model connects blocks without a matrix; no change
- Clean/model_augmentation/fit_systems/interconnect.py:261 | default method by signal index [branch, Jan's code] | compute | reader: interconnect.py:261, :262 | keep | no change
- Clean/model_augmentation/fit_systems/interconnect.py:292 | 'concat' alias [option, Jan's code] | compute | reader: model.py:81, :84 | keep | no change
- Clean/model_augmentation/fit_systems/interconnect.py:299 | Signal_Connection attributes [attributes, Jan's code] | compute | reader: interconnect.py:133, :134, :215, :226 | keep | no change
- Clean/model_augmentation/fit_systems/interconnect.py:307 | tries = 20 [number] | unlabelled | reader: interconnect.py:310 | label | "# HEURISTIC: about 5 s of retries for a sync client or scanner holding the file"; bitwise
- Clean/model_augmentation/fit_systems/interconnect.py:316 | sleep 0.25 s [number] | unlabelled | reader: interconnect.py:316 | label | same label; bitwise
- Clean/model_augmentation/fit_systems/interconnect.py:329 | DtypeDataLoader data, ids, rng (seed 0), batch_size [attributes] | compute | reader: deepSI My_Simple_DataLoader.__iter__ (fit_system.py:690) | keep | deepSI's iterator reads them; no change
- Clean/model_augmentation/fit_systems/interconnect.py:339 | simulator [ClassVar default] | compute | reader: interconnect.py:366, :372, :377 | keep | no change
- Clean/model_augmentation/fit_systems/interconnect.py:342 | validation_probes [ClassVar default] | compute | reader: interconnect.py:378 | keep | no change
- Clean/model_augmentation/fit_systems/interconnect.py:345 | training_phase [ClassVar default] | compute | reader: interconnect.py:478 | keep | no change
- Clean/model_augmentation/fit_systems/interconnect.py:351 | hfn, norm [attributes, Jan's code] | compute | reader: interconnect.py:432, :446 | keep | no change
- Clean/model_augmentation/fit_systems/interconnect.py:360 | bestfit [attribute, Jan's code] | compute | reader: interconnect.py:415, training.py:307 | keep | no change
- Clean/model_augmentation/fit_systems/interconnect.py:361 | init_model_done [attribute, Jan's code] | compute | reader: deepSI checkpoint_load_system (fit_system.py:502) | keep | no change
- Clean/model_augmentation/fit_systems/interconnect.py:429 | epoch_counter, batch_counter [attributes, Jan's code] | compute | reader: interconnect.py:544, training.py:170 | keep | no change
- Clean/model_augmentation/fit_systems/interconnect.py:435 | Loss_val, Loss_train, batch_id, time, epoch_id [attributes, Jan's code] | compute | reader: interconnect.py:632 (fit_history), training.py:357 | keep | the loss curve data (E4); no change
- Clean/model_augmentation/fit_systems/interconnect.py:441 | n_its is None [branch, Jan's code] | compute | reader: interconnect.py:441 | keep | Clean passes n_its None in production; no change
- Clean/model_augmentation/fit_systems/interconnect.py:442 | its_per_val == 'epoch' [branch, Jan's code] | fixed-option | reader: interconnect.py:442 (Clean always passes an int) | keep | Jan's code; harmless; Dirk decides
- Clean/model_augmentation/fit_systems/interconnect.py:459 | cuda batch move [branch, Jan's code] | compute | reader: interconnect.py:459 | keep | no change
- Clean/model_augmentation/fit_systems/interconnect.py:473 | NaN loss breaks [branch, Jan's code] | compute | reader: interconnect.py:473 | keep | no change
- Clean/model_augmentation/fit_systems/interconnect.py:478 | PS2 opening after each update [branch] | compute | reader: interconnect.py:478 | keep | no change
- Clean/model_augmentation/fit_systems/interconnect.py:494 | 'sqrt loss' if train_loss >= 0 else 'loss' [branch, Jan's code] | fixed-option | reader: interconnect.py:494 | keep | the else arm needs a negative loss, which V never is; Jan's code; Dirk decides
- Clean/model_augmentation/fit_systems/interconnect.py:497 | KeyboardInterrupt stops early [branch, Jan's code] | compute | reader: interconnect.py:497 | keep | a user stops a run by hand; no change
- Clean/model_augmentation/fit_systems/interconnect.py:511 | no _best dump: keep last [branch, Jan's code] | compute | reader: interconnect.py:511 | keep | no change
- Clean/model_augmentation/fit_systems/interconnect.py:524 | _optimizer = None [ClassVar default] | mainline-compat | reader: interconnect.py:648 | remove | with the property below
- Clean/model_augmentation/fit_systems/interconnect.py:525 | obc = None [ClassVar default] | compute | reader: interconnect.py:557, :603 | keep | no change
- Clean/model_augmentation/fit_systems/interconnect.py:527 | _UNPICKLABLE ('obc', 'training_phase', 'fit_history') [ClassVar] | compute | reader: interconnect.py:633, :642 | keep | no change
- Clean/model_augmentation/fit_systems/interconnect.py:529 | _FIT_HISTORY [ClassVar] | compute | reader: interconnect.py:632 | keep | E4 implementation choice 5; no change
- Clean/model_augmentation/fit_systems/interconnect.py:537 | _ann_ix, _mats_ix [attributes] | compute | reader: interconnect.py:564, :579 | keep | K28 ruled
- Clean/model_augmentation/fit_systems/interconnect.py:544 | + 1e-6 in _epoch_index [number] | unlabelled | reader: interconnect.py:544 | label | "# HEURISTIC: the sum of 1/N per update can end just below an integer" (the comment says it, without the label); bitwise
- Clean/model_augmentation/fit_systems/interconnect.py:562 | rebuild basis on request or when missing [branch] | compute | reader: interconnect.py:562 | keep | no change
- Clean/model_augmentation/fit_systems/interconnect.py:603 | OBC coefficient per objective [branch] | compute | reader: interconnect.py:603 | keep | no change
- Clean/model_augmentation/fit_systems/interconnect.py:632 | fit_history [attribute] | compute | reader: training.py:354 | keep | no change
- Clean/model_augmentation/fit_systems/interconnect.py:644 | comment "deepSI stores the optimizer as _optimizer in the dumps; the property keeps that attribute" [comment] | misleading, mainline-compat | reader: none | remove | deepSI stores self.optimizer under 'optimizer' (deepSI fit_system.py:156); '_optimizer' is mainline's backing field (mainline interconnect.py:1287, :1521); bitwise
- Clean/model_augmentation/fit_systems/interconnect.py:645 | optimizer property and setter [property] | mainline-compat | reader: every fit_sys.optimizer access (model.py:111, training.py:37, :136) | remove | in Clean/ it only renames deepSI's attribute to mainline's '_optimizer'; with B2 training.py:65 reading 'optimizer'; bitwise; existing Clean test dumps resume with fresh moments; thesis dumps are not resumed by Clean/

### D12. `model_augmentation/fit_systems/lbfgs_polish.py`
- Clean/model_augmentation/fit_systems/lbfgs_polish.py:18 | n_windows, max_iter, inner_iter, history_size, tol_grad, tol_change, seed [PolishSpec fields] | compute | reader: lbfgs_polish.py:200, :207, :255 | keep | PolishSpec ruled kept (amendment 9 point 1); no change
- Clean/model_augmentation/fit_systems/lbfgs_polish.py:45 | n_total, effective_stride, n_used, args, kw [FixedWindowBatch attributes] | compute | reader: lbfgs_polish.py:90, :201 | keep | no change
- Clean/model_augmentation/fit_systems/lbfgs_polish.py:58 | seeded subset when more windows than n_windows [branch] | compute | reader: lbfgs_polish.py:58 | keep | no change
- Clean/model_augmentation/fit_systems/lbfgs_polish.py:80 | / 1024 ** 2 (MB) [number] | unlabelled | reader: lbfgs_polish.py:80 | keep | unit conversion in a print; no change
- Clean/model_augmentation/fit_systems/lbfgs_polish.py:116 | [:160], [:200] message truncation [numbers] | unlabelled | reader: lbfgs_polish.py:116, :285 | keep | message length; no change
- Clean/model_augmentation/fit_systems/lbfgs_polish.py:150 | ran, accepted, val_after [lbfgs_outcome keys] | compute | reader: training.py:225, :326, evaluation.py:255 | keep | no change
- Clean/model_augmentation/fit_systems/lbfgs_polish.py:150 | reason, n_iter, n_closure, func_evals, wall_s, loss_first, loss_last, val_before, n_windows_used, n_windows_total, effective_stride, deterministic, used_compiled, params_polished [lbfgs_outcome keys] | write-only | reader: user (the polish log in the final .npz) | keep | what a user reads to judge a polish; no change
- Clean/model_augmentation/fit_systems/lbfgs_polish.py:150 | max_iter [lbfgs_outcome key] | write-only | reader: none | remove | echo of the configured LBFGS_MAX_ITER (config.json); bitwise; compare_l3 compares common fields only
- Clean/model_augmentation/fit_systems/lbfgs_polish.py:214 | compiled probe, eager fallback, OOM release, abort, converged break, cap, non-finite, exception, rollback [branches] | compute | reader: lbfgs_polish.py:214-298 | keep | K36 ruled; no change
- Clean/model_augmentation/fit_systems/lbfgs_polish.py:250 | max(abs(first), 1e-300) [number] | unlabelled | reader: lbfgs_polish.py:250 | label | "# HEURISTIC: division guard of a printed ratio"; bitwise

### D13. `model_augmentation/fit_systems/obc.py`
- Clean/model_augmentation/fit_systems/obc.py:37 | zero column scaled by 1 [branch] | compute | reader: obc.py:37 | keep | no change
- Clean/model_augmentation/fit_systems/obc.py:46 | truncation on rank loss [branch] | compute | reader: obc.py:46 | keep | no change
- Clean/model_augmentation/fit_systems/obc.py:48 | U, S, Vh, column_scale, vbar [OBCBasis attributes] | compute | reader: obc.py:66, :67, interconnect.py:565 | keep | no change
- Clean/model_augmentation/fit_systems/obc.py:51 | n_cols, rank, tol [OBCBasis attributes] | compute | reader: obc.py:157, :160 | keep | rank policy and its message; no change
- Clean/model_augmentation/fit_systems/obc.py:103 | float64 TypeError [guard] | compute | reader: obc.py:103 | keep | K19 ruled
- Clean/model_augmentation/fit_systems/obc.py:116 | append_value branches [branch] | fixed-option | reader: obc.py:116, :120 | keep | K18 ruled; no caller passes False (row E)
- Clean/model_augmentation/fit_systems/obc.py:129 | max_consecutive_rank_loss = 2 [ClassVar, number] | compute, unlabelled | reader: obc.py:165 | label | "# HEURISTIC: abort after two rank-deficient refreshes in a row"; bitwise
- Clean/model_augmentation/fit_systems/obc.py:133 | reference_field, basis_builder, basis, last_refresh_epoch, consecutive_rank_loss [OBCProjection attributes] | compute | reader: obc.py:151, :157, :183, :196 | keep | no change
- Clean/model_augmentation/fit_systems/obc.py:154 | candidate basis follows the old basis device [branch] | compute | reader: obc.py:154 | keep | no change
- Clean/model_augmentation/fit_systems/obc.py:157 | rank policy (abort at first build, keep last, accept) [branch] | compute | reader: obc.py:157-171 | keep | no change

### D14. `model_augmentation/fit_systems/ps2_opening.py`
- Clean/model_augmentation/fit_systems/ps2_opening.py:23 | aux_batch = 64 [PS2Spec field] | compute | reader: ps2_opening.py:233 | keep | labelled HEURISTIC; no change
- Clean/model_augmentation/fit_systems/ps2_opening.py:24 | lr = 1e-3 [PS2Spec field] | compute | reader: ps2_opening.py:163, :297 | keep | labelled THEORY; no change
- Clean/model_augmentation/fit_systems/ps2_opening.py:25 | eval_every = 25 [PS2Spec field] | compute | reader: ps2_opening.py:121 | keep | labelled; no change
- Clean/model_augmentation/fit_systems/ps2_opening.py:26 | s1_min = 100 [PS2Spec field] | compute | reader: ps2_opening.py:125 | keep | labelled; no change
- Clean/model_augmentation/fit_systems/ps2_opening.py:27 | s1_cap = 350 [PS2Spec field] | compute | reader: ps2_opening.py:127 | keep | labelled; no change
- Clean/model_augmentation/fit_systems/ps2_opening.py:28 | s1_tol = 0.02 [PS2Spec field] | compute | reader: ps2_opening.py:126 | keep | labelled; no change
- Clean/model_augmentation/fit_systems/ps2_opening.py:29 | s1_window = 4 [PS2Spec field] | compute | reader: ps2_opening.py:125 | keep | covered by the label of s1_tol; no change
- Clean/model_augmentation/fit_systems/ps2_opening.py:31 | screen_at = 150 [PS2Spec field] | compute | reader: ps2_opening.py:130 | keep | labelled; no change
- Clean/model_augmentation/fit_systems/ps2_opening.py:32 | screen_max = 0.90 [PS2Spec field] | compute, unlabelled | reader: ps2_opening.py:130 | label | "# HEURISTIC: S2 runs only if the decoder explains at least 10 % of the residual" (after_step docstring); bitwise
- Clean/model_augmentation/fit_systems/ps2_opening.py:33 | n_cal = 128 [PS2Spec field] | compute, unlabelled | reader: ps2_opening.py:164 | label | HEURISTIC calibration windows; bitwise
- Clean/model_augmentation/fit_systems/ps2_opening.py:34 | n_snap = 8192 [PS2Spec field] | compute, unlabelled | reader: ps2_opening.py:290 | label | HEURISTIC S2 fit pairs; bitwise
- Clean/model_augmentation/fit_systems/ps2_opening.py:35 | n_snap_cal = 2048 [PS2Spec field] | compute, unlabelled | reader: ps2_opening.py:290 | label | HEURISTIC; bitwise
- Clean/model_augmentation/fit_systems/ps2_opening.py:36 | s2_batch = 512 [PS2Spec field] | compute, unlabelled | reader: ps2_opening.py:300 | label | HEURISTIC; bitwise
- Clean/model_augmentation/fit_systems/ps2_opening.py:37 | s2_check = 100 [PS2Spec field] | compute | reader: ps2_opening.py:313 | keep | labelled; no change
- Clean/model_augmentation/fit_systems/ps2_opening.py:38 | s2_min = 500 [PS2Spec field] | compute, unlabelled | reader: ps2_opening.py:320 | label | HEURISTIC; bitwise
- Clean/model_augmentation/fit_systems/ps2_opening.py:39 | s2_cap = 8000 [PS2Spec field] | compute, unlabelled | reader: ps2_opening.py:300, :307 | label | HEURISTIC; bitwise
- Clean/model_augmentation/fit_systems/ps2_opening.py:40 | s2_tol = 0.01 [PS2Spec field] | compute | reader: ps2_opening.py:320 | keep | covered by the s2_check label; no change
- Clean/model_augmentation/fit_systems/ps2_opening.py:41 | seed = 0 [PS2Spec field] | fixed-option | reader: ps2_opening.py:91, :164, :282, :298 | keep | the only caller passes cfg.seed (training.py:243); the default is never used but a spec without a seed would be odd; no change
- Clean/model_augmentation/fit_systems/ps2_opening.py:64 | cls.count(k) >= 3 [number] | unlabelled | reader: ps2_opening.py:64 | label | extend the HEURISTIC at :61: a class needs three records so one can be left out; bitwise
- Clean/model_augmentation/fit_systems/ps2_opening.py:77 | _added_cols [attribute] | compute, fixed-option | reader: ps2_opening.py:295 | keep | with K32 it is range(6, 6 + n_a); no change
- Clean/model_augmentation/fit_systems/ps2_opening.py:78 | records, names, spec, ctrl_rows, n_xb, n_a, width, fit_ix, cal_ix [attributes] | compute | reader: ps2_opening.py:147, :157, :181, :233 | keep | no change
- Clean/model_augmentation/fit_systems/ps2_opening.py:89 | phase, it, history, s1_end, s2, status [attributes] | compute | reader: ps2_opening.py:97, :125, :144 | keep | no change
- Clean/model_augmentation/fit_systems/ps2_opening.py:89 | phase name 'S3' = the opening has ended [name] | misleading | reader: ps2_opening.py:97, training.py:124, :242 | keep | the thesis has S1 and S2 only, so 'S3' reads as a third stage, but renaming it changes the ps2_opening value in every dump and the resume test; the module docstring defines it; no change
- Clean/model_augmentation/fit_systems/ps2_opening.py:91 | _rng, _decoder, _opt, _cal, _mats, _dev, _dt, _M, _aux [attributes] | compute | reader: ps2_opening.py:119, :163, :233 | keep | no change
- Clean/model_augmentation/fit_systems/ps2_opening.py:86 | empty calibration split [guard] | compute | reader: ps2_opening.py:86 | keep | actionable guard; no change
- Clean/model_augmentation/fit_systems/ps2_opening.py:144 | ps2_summary JSON keys status, phase, updates, s1_end, s2, calibration_records, history, spec [saved data] | write-only | reader: user | keep | the record of the opening (B2 ps2_summary); no change
- Clean/model_augmentation/fit_systems/ps2_opening.py:164 | seed + 321 [number] | unlabelled | reader: ps2_opening.py:164 | label | "# HEURISTIC: an offset so the calibration draw is independent of the S1 stream"; bitwise
- Clean/model_augmentation/fit_systems/ps2_opening.py:282 | seed + 1000 [number] | unlabelled | reader: ps2_opening.py:282 | label | same label; bitwise
- Clean/model_augmentation/fit_systems/ps2_opening.py:320 | plateau window 5 of S2 [number] | unlabelled | reader: ps2_opening.py:320 | rename | a PS2Spec field s2_window = 5 beside s1_window (the comment at :37 names it); bitwise; ps2_summary spec gains the field

### D15. `model_augmentation/fit_systems/pre_encoder.py` (Jan's file; AugmentedEncoder is @added)
- Clean/model_augmentation/fit_systems/pre_encoder.py:6 | "# CHANGED: ... at the repository root" [comment, Jan's code] | misleading, mainline-compat | reader: none | comment | as blocks.py:12; bitwise
- Clean/model_augmentation/fit_systems/pre_encoder.py:47 | nu, ny, na, nb [attributes] | write-only | reader: none | keep | K37 ruled (amendment 9: readable names)
- Clean/model_augmentation/fit_systems/pre_encoder.py:49 | nx, nx_aug [attributes] | compute | reader: pre_encoder.py:100 | keep | no change
- Clean/model_augmentation/fit_systems/pre_encoder.py:56 | n_y, n_u [attributes] | compute | reader: pre_encoder.py:60, :91, :92 | keep | no change

### D16. `model_augmentation/systems/gantry_ss.py`
- Clean/model_augmentation/systems/gantry_ss.py:8 | _D = torch.float32 [module constant] | compute | reader: gantry_ss.py:10-67 | keep | the nominal values enter the float64 model float32-rounded (model.py:52, Lb, P, Cd), and the thesis checkpoints were trained so; float64 constants would be a rounding item; add one comment saying so; bitwise
- Clean/model_augmentation/systems/gantry_ss.py:10 | mb = 22.8 [module constant] | compute, unlabelled | reader: model.py:52 (getattr) | label | one "# THEORY: Garcia-Herreros et al. Table I, kamtin-fp-model main.m" line above the block (the docstring cites it, without the label); bitwise
- Clean/model_augmentation/systems/gantry_ss.py:11 | mh = 10.1 [module constant] | compute, unlabelled | reader: model.py:52, blocks.py:153 | label | same line
- Clean/model_augmentation/systems/gantry_ss.py:12 | m1 = 10.2 [module constant] | compute, unlabelled | reader: model.py:52 | label | same
- Clean/model_augmentation/systems/gantry_ss.py:13 | m2 = 10.7 [module constant] | compute, unlabelled | reader: model.py:52 | label | same
- Clean/model_augmentation/systems/gantry_ss.py:15 | Jb = 1.0 [module constant] | compute, unlabelled | reader: model.py:52 | label | same
- Clean/model_augmentation/systems/gantry_ss.py:16 | Jh = 0.05 [module constant] | compute, unlabelled | reader: model.py:52 | label | same
- Clean/model_augmentation/systems/gantry_ss.py:18 | cg1 = 14.5 [module constant] | compute, unlabelled | reader: model.py:52 | label | same
- Clean/model_augmentation/systems/gantry_ss.py:19 | cg2 = 20.3 [module constant] | compute, unlabelled | reader: model.py:52 | label | same
- Clean/model_augmentation/systems/gantry_ss.py:20 | cy = 10.0 [module constant] | compute, unlabelled | reader: model.py:52 | label | same
- Clean/model_augmentation/systems/gantry_ss.py:22 | cb1 = 9.0 [module constant] | compute, unlabelled | reader: model.py:52 | label | same
- Clean/model_augmentation/systems/gantry_ss.py:23 | cb2 = 9.0 [module constant] | compute, unlabelled | reader: model.py:52 | label | same
- Clean/model_augmentation/systems/gantry_ss.py:25 | kb1 = 1987.5 [module constant] | compute, unlabelled | reader: model.py:52 | label | same
- Clean/model_augmentation/systems/gantry_ss.py:26 | kb2 = 1987.5 [module constant] | compute, unlabelled | reader: model.py:52 | label | same
- Clean/model_augmentation/systems/gantry_ss.py:28 | Lb = 0.725 [module constant] | compute, unlabelled | reader: model.py:53, blocks.py:403 | label | same
- Clean/model_augmentation/systems/gantry_ss.py:29 | d = 0.1 [module constant] | compute, unlabelled | reader: model.py:52 | label | same
- Clean/model_augmentation/systems/gantry_ss.py:31 | _z [module constant] | compute | reader: gantry_ss.py:34-56 | keep | no change
- Clean/model_augmentation/systems/gantry_ss.py:34 | M1 [module constant] | write-only after P3 | reader: blocks.py:159 (nominal build only) | remove | with P3; the trainable path builds M_1 from m_h (blocks.py:478); bitwise
- Clean/model_augmentation/systems/gantry_ss.py:39 | M2 [module constant] | write-only after P3 | reader: blocks.py:159 | remove | with P3
- Clean/model_augmentation/systems/gantry_ss.py:43 | C [module constant] | write-only after P3 | reader: blocks.py:159, :175 | remove | with P3
- Clean/model_augmentation/systems/gantry_ss.py:51 | K [module constant] | write-only after P3 | reader: blocks.py:159, :174 | remove | with P3
- Clean/model_augmentation/systems/gantry_ss.py:56 | P [module constant] | compute | reader: blocks.py:176, obc_gantry.py:30, data.py:219 | keep | no change
- Clean/model_augmentation/systems/gantry_ss.py:64 | Cd [module constant] | compute | reader: data.py:241 | keep | no change
- Clean/model_augmentation/systems/gantry_ss.py:67 | Dd [module constant] | compute, fixed-option | reader: data.py:242 | keep | K40 ruled
- Clean/model_augmentation/systems/gantry_ss.py:109 | Ax, Bw, Bu return values of build_G_matrix_entries [return values] | write-only after P3 | reader: blocks.py:480 discards them | literal | return A_combined only once P3 removes the nominal build; bitwise

### D17. `model_augmentation/utils/` and `model_augmentation/__init__.py` (Jan's code)
- Clean/model_augmentation/utils/deepSI_corrections.py:5 | "# CHANGED: ... at the repository root" [comment, Jan's code] | misleading, mainline-compat | reader: none | comment | as blocks.py:12; bitwise
- Clean/model_augmentation/utils/deepSI_corrections.py:52 | u0, ustd, y0, ystd [attributes, Jan's code] | compute | reader: deepSI_corrections.py:76, :77, closed_loop.py:284 | keep | no change
- Clean/model_augmentation/utils/deepSI_corrections.py:67 | list input transformed record by record [branch] | compute | reader: deepSI_corrections.py:67 | keep | no change
- Clean/model_augmentation/utils/deepSI_corrections.py:70 | type guard [guard] | compute | reader: deepSI_corrections.py:70 | keep | no change
- Clean/model_augmentation/utils/deepSI_corrections.py:74 | assert not normed [guard, Jan's code] | compute | reader: deepSI_corrections.py:74 | keep | no change
- Clean/model_augmentation/utils/deepSI_corrections.py:76 | u is None, y is None branches [branch, Jan's code] | fixed-option | reader: deepSI_corrections.py:76, :77 | keep | Jan's code; every record has u and y; Dirk decides
- Clean/model_augmentation/utils/deepSI_corrections.py:37 | k0_right, na_right, nb_right windows [branch, Jan's code] | fixed-option | reader: deepSI_corrections.py:38-41 | keep | always 0 in Clean/; Jan's code; Dirk decides
- Clean/model_augmentation/utils/torch_nets.py:6 | "# CHANGED: ... at the repository root" [comment, Jan's code] | misleading, mainline-compat | reader: none | comment | as blocks.py:12; bitwise
- Clean/model_augmentation/utils/torch_nets.py:58 | net [AugmentationNetwork attribute] | compute | reader: torch_nets.py:74 | keep | no change
- Clean/model_augmentation/utils/utils.py:2 | "# CHANGED: ... at the repository root" [comment, Jan's code] | misleading, mainline-compat | reader: none | comment | as blocks.py:12; bitwise
- Clean/model_augmentation/__init__.py:1 | attribution to Jan's public code, commit 8200923, BSD 3-Clause [comment] | compute | reader: user | keep | licence notice; it is the reference the CHANGED headers should point to; no change

## E. Function parameters and their defaults (one row each)
Generated from the AST of the 24 files (first reading line in the body); the rows with a verdict other than a plain keep, and the keeps that rest on a ruling, are judged by hand.
- Clean/gantry_dynamic/config.py:51 | cfg [parameter of y_decimation] | compute | reader: config.py:53 | keep | read by the body; no change
- Clean/gantry_dynamic/config.py:58 | cfg [parameter of record_trim] | compute | reader: config.py:60 | keep | read by the body; no change
- Clean/gantry_dynamic/config.py:144 | meta [parameter of RunConfig.from_config_json] | compute | reader: config.py:146 | keep | read by the body; no change
- Clean/gantry_dynamic/config.py:144 | overrides=** [parameter of RunConfig.from_config_json] | compute | reader: config.py:163 | keep | read by the body; no change
- Clean/gantry_dynamic/config.py:212 | cfg [parameter of save_dir] | compute | reader: config.py:214 | keep | read by the body; no change
- Clean/gantry_dynamic/config.py:220 | args=* [parameter of git_provenance._git] | compute | reader: config.py:222 | keep | read by the body; no change
- Clean/gantry_dynamic/config.py:245 | cfg [parameter of config_json_dict] | compute | reader: config.py:248 | keep | read by the body; no change
- Clean/gantry_dynamic/config.py:245 | git=None [parameter of config_json_dict] | verification | reader: config.py:324 | keep | main passes git_provenance(); the harness (hcommon.py:263) passes None; harmless default; no change
- Clean/gantry_dynamic/controller.py:36 | Y_op [parameter of mass_matrix_at] | compute | reader: controller.py:48 | keep | read by the body; no change
- Clean/gantry_dynamic/controller.py:54 | Y_op [parameter of stage_frequency_response] | compute | reader: controller.py:66 | keep | read by the body; no change
- Clean/gantry_dynamic/controller.py:54 | s [parameter of stage_frequency_response] | compute | reader: controller.py:71 | keep | read by the body; no change
- Clean/gantry_dynamic/controller.py:89 | Y_op [parameter of build_cfb_at] | compute | reader: controller.py:102 | keep | read by the body; no change
- Clean/gantry_dynamic/controller.py:89 | ts [parameter of build_cfb_at] | compute | reader: controller.py:111 | keep | read by the body; no change
- Clean/gantry_dynamic/controller.py:116 | Y_op [parameter of controller_ss] | compute | reader: controller.py:129 | keep | read by the body; no change
- Clean/gantry_dynamic/controller.py:116 | ts [parameter of controller_ss] | compute | reader: controller.py:129 | keep | read by the body; no change
- Clean/gantry_dynamic/controller.py:116 | gain=1.0 [parameter of controller_ss] | fixed-option | reader: controller.py:131 | keep | K3 implementation choice 1 (the gain argument stays); always 1.0 in Clean/
- Clean/gantry_dynamic/controller.py:134 | name [parameter of _stem] | compute | reader: controller.py:136 | keep | read by the body; no change
- Clean/gantry_dynamic/controller.py:144 | name [parameter of register_stored_controller] | compute | reader: controller.py:146 | keep | read by the body; no change
- Clean/gantry_dynamic/controller.py:144 | ctrl_name [parameter of register_stored_controller] | compute | reader: controller.py:147 | keep | read by the body; no change
- Clean/gantry_dynamic/controller.py:144 | design_y [parameter of register_stored_controller] | compute | reader: controller.py:146 | keep | read by the body; no change
- Clean/gantry_dynamic/controller.py:144 | gain [parameter of register_stored_controller] | compute | reader: controller.py:146 | keep | read by the body; no change
- Clean/gantry_dynamic/controller.py:144 | fbw_hz [parameter of register_stored_controller] | compute | reader: controller.py:146 | keep | read by the body; no change
- Clean/gantry_dynamic/controller.py:150 | name [parameter of controller_design] | compute | reader: controller.py:152 | keep | read by the body; no change
- Clean/gantry_dynamic/controller.py:150 | use_gain=False [parameter of controller_design] | fixed-option | reader: controller.py:161 | keep | K3 ruled; always False in Clean/ (controller_gain of A3 is not wired to it)
- Clean/gantry_dynamic/controller.py:167 | record_names [parameter of build_controller_bank] | compute | reader: controller.py:182 | keep | read by the body; no change
- Clean/gantry_dynamic/controller.py:167 | ts [parameter of build_controller_bank] | compute | reader: controller.py:185 | keep | read by the body; no change
- Clean/gantry_dynamic/controller.py:167 | ystd [parameter of build_controller_bank] | compute | reader: controller.py:187 | keep | read by the body; no change
- Clean/gantry_dynamic/controller.py:167 | std_u [parameter of build_controller_bank] | compute | reader: controller.py:187 | keep | read by the body; no change
- Clean/gantry_dynamic/controller.py:167 | dtype=torch.float32 [parameter of build_controller_bank] | fixed-option, misleading | reader: controller.py:187 | remove | drop the float32 default: both callers (controller.py:215, evaluation.py:248) pass cfg.dtype_pt = float64, and a caller that forgot it would build a float32 bank; bitwise; l4_loader passes dtype already
- Clean/gantry_dynamic/controller.py:167 | use_gain=False [parameter of build_controller_bank] | fixed-option | reader: controller.py:182 | keep | K3 ruled
- Clean/gantry_dynamic/controller.py:191 | fs [parameter of build_closed_loop] | compute | reader: controller.py:228 | keep | read by the body; no change
- Clean/gantry_dynamic/controller.py:191 | norm [parameter of build_closed_loop] | compute | reader: controller.py:215 | keep | read by the body; no change
- Clean/gantry_dynamic/controller.py:191 | cfg [parameter of build_closed_loop] | compute | reader: controller.py:215 | keep | read by the body; no change
- Clean/gantry_dynamic/controller.py:191 | train_files [parameter of build_closed_loop] | compute | reader: controller.py:209 | keep | read by the body; no change
- Clean/gantry_dynamic/controller.py:191 | val_files [parameter of build_closed_loop] | compute | reader: controller.py:209 | keep | read by the body; no change
- Clean/gantry_dynamic/controller.py:191 | val_data [parameter of build_closed_loop] | write-only, verification | reader: none | remove | K6 kept it as a useful option, but it is no option: unused, and its docstring keeps it "for callers", an interface reason the yardstick excludes; the evaluation added since does not pass it; bitwise; main (gantry_interconnect_dynamic.py:118) and l4_loader.py build() drop the keyword
- Clean/gantry_dynamic/controller.py:191 | e_train [parameter of build_closed_loop] | compute | reader: controller.py:237 | keep | read by the body; no change
- Clean/gantry_dynamic/controller.py:191 | e_val [parameter of build_closed_loop] | compute | reader: controller.py:235 | keep | read by the body; no change
- Clean/gantry_dynamic/controller.py:191 | verbose=True [parameter of build_closed_loop] | verification | reader: controller.py:219 | keep | K6 ruled; only l4_loader passes False
- Clean/gantry_dynamic/data.py:30 | cfg [parameter of record_files] | compute | reader: data.py:33 | keep | read by the body; no change
- Clean/gantry_dynamic/data.py:42 | d [parameter of _register_controller] | compute | reader: data.py:45 | keep | read by the body; no change
- Clean/gantry_dynamic/data.py:42 | filename [parameter of _register_controller] | compute | reader: data.py:47 | keep | read by the body; no change
- Clean/gantry_dynamic/data.py:80 | cfg [parameter of traj_dir] | compute | reader: data.py:82 | keep | read by the body; no change
- Clean/gantry_dynamic/data.py:85 | d [parameter of _load_u] | compute | reader: data.py:87 | keep | read by the body; no change
- Clean/gantry_dynamic/data.py:90 | u [parameter of _resample_u] | compute | reader: data.py:104 | keep | read by the body; no change
- Clean/gantry_dynamic/data.py:90 | cfg [parameter of _resample_u] | compute | reader: data.py:103 | keep | read by the body; no change
- Clean/gantry_dynamic/data.py:108 | d [parameter of _decimate_y] | compute | reader: data.py:121 | keep | read by the body; no change
- Clean/gantry_dynamic/data.py:108 | cfg [parameter of _decimate_y] | compute | reader: data.py:119 | keep | read by the body; no change
- Clean/gantry_dynamic/data.py:127 | d [parameter of _record_at_fs_new] | compute | reader: data.py:130 | keep | read by the body; no change
- Clean/gantry_dynamic/data.py:127 | cfg [parameter of _record_at_fs_new] | compute | reader: data.py:129 | keep | read by the body; no change
- Clean/gantry_dynamic/data.py:127 | fields=() [parameter of _record_at_fs_new] | compute | reader: data.py:132 | keep | read by the body; no change
- Clean/gantry_dynamic/data.py:140 | filename [parameter of _load_record] | compute | reader: data.py:142 | keep | read by the body; no change
- Clean/gantry_dynamic/data.py:140 | cfg [parameter of _load_record] | compute | reader: data.py:142 | keep | read by the body; no change
- Clean/gantry_dynamic/data.py:140 | folder=None [parameter of _load_record] | compute | reader: data.py:142 | keep | K11 (unruled) settled by new evidence: the evaluation loads the noise-free twins with folder=cfg.data_root (evaluation.py:267)
- Clean/gantry_dynamic/data.py:150 | e_list [parameter of servo_sigma] | compute | reader: data.py:163 | keep | read by the body; no change
- Clean/gantry_dynamic/data.py:150 | k0 [parameter of servo_sigma] | compute | reader: data.py:163 | keep | read by the body; no change
- Clean/gantry_dynamic/data.py:170 | cfg [parameter of load_datasets] | compute | reader: data.py:172 | keep | read by the body; no change
- Clean/gantry_dynamic/data.py:177 | files [parameter of load_datasets._split] | compute | reader: data.py:179 | keep | read by the body; no change
- Clean/gantry_dynamic/data.py:200 | cfg [parameter of compute_normalization] | compute | reader: data.py:219 | keep | read by the body; no change
- Clean/gantry_dynamic/data.py:200 | data [parameter of compute_normalization] | compute | reader: data.py:215 | keep | read by the body; no change
- Clean/gantry_dynamic/evaluation.py:40 | fit_sys [parameter of _without_learned] | compute | reader: evaluation.py:50 | keep | read by the body; no change
- Clean/gantry_dynamic/evaluation.py:40 | free=None [parameter of _without_learned] | compute | reader: evaluation.py:58 | keep | read by the body; no change
- Clean/gantry_dynamic/evaluation.py:63 | phy [parameter of _true_combinations] | compute | reader: evaluation.py:73 | keep | read by the body; no change
- Clean/gantry_dynamic/evaluation.py:76 | phy [parameter of _true_free] | compute | reader: evaluation.py:86 | keep | read by the body; no change
- Clean/gantry_dynamic/evaluation.py:89 | cfg [parameter of _noise_free_twins] | compute | reader: evaluation.py:92 | keep | read by the body; no change
- Clean/gantry_dynamic/evaluation.py:98 | models [parameter of _split_scores] | compute | reader: evaluation.py:117 | keep | read by the body; no change
- Clean/gantry_dynamic/evaluation.py:98 | sdl [parameter of _split_scores] | compute | reader: evaluation.py:117 | keep | read by the body; no change
- Clean/gantry_dynamic/evaluation.py:98 | e_list [parameter of _split_scores] | compute | reader: evaluation.py:121 | keep | read by the body; no change
- Clean/gantry_dynamic/evaluation.py:98 | e_nf_list [parameter of _split_scores] | compute | reader: evaluation.py:128 | keep | read by the body; no change
- Clean/gantry_dynamic/evaluation.py:98 | bank [parameter of _split_scores] | compute | reader: evaluation.py:117 | keep | read by the body; no change
- Clean/gantry_dynamic/evaluation.py:98 | rows [parameter of _split_scores] | compute | reader: evaluation.py:117 | keep | read by the body; no change
- Clean/gantry_dynamic/evaluation.py:98 | k0 [parameter of _split_scores] | compute | reader: evaluation.py:118 | keep | read by the body; no change
- Clean/gantry_dynamic/evaluation.py:134 | split [parameter of _table] | compute | reader: evaluation.py:139 | keep | read by the body; no change
- Clean/gantry_dynamic/evaluation.py:134 | names [parameter of _table] | compute | reader: evaluation.py:141 | keep | read by the body; no change
- Clean/gantry_dynamic/evaluation.py:134 | records [parameter of _table] | compute | reader: evaluation.py:139 | keep | read by the body; no change
- Clean/gantry_dynamic/evaluation.py:134 | s [parameter of _table] | compute | reader: evaluation.py:136 | keep | read by the body; no change
- Clean/gantry_dynamic/evaluation.py:149 | path [parameter of _bar_figure] | compute | reader: evaluation.py:166 | keep | read by the body; no change
- Clean/gantry_dynamic/evaluation.py:149 | title [parameter of _bar_figure] | compute | reader: evaluation.py:164 | keep | read by the body; no change
- Clean/gantry_dynamic/evaluation.py:149 | names [parameter of _bar_figure] | compute | reader: evaluation.py:155 | keep | read by the body; no change
- Clean/gantry_dynamic/evaluation.py:149 | records [parameter of _bar_figure] | compute | reader: evaluation.py:151 | keep | read by the body; no change
- Clean/gantry_dynamic/evaluation.py:149 | s [parameter of _bar_figure] | compute | reader: evaluation.py:157 | keep | read by the body; no change
- Clean/gantry_dynamic/evaluation.py:169 | path [parameter of _record_figure] | compute | reader: evaluation.py:192 | keep | read by the body; no change
- Clean/gantry_dynamic/evaluation.py:169 | title [parameter of _record_figure] | compute | reader: evaluation.py:189 | keep | read by the body; no change
- Clean/gantry_dynamic/evaluation.py:169 | names [parameter of _record_figure] | compute | reader: evaluation.py:174 | keep | read by the body; no change
- Clean/gantry_dynamic/evaluation.py:169 | t [parameter of _record_figure] | compute | reader: evaluation.py:176 | keep | read by the body; no change
- Clean/gantry_dynamic/evaluation.py:169 | residual [parameter of _record_figure] | compute | reader: evaluation.py:176 | keep | read by the body; no change
- Clean/gantry_dynamic/evaluation.py:169 | ref [parameter of _record_figure] | compute | reader: evaluation.py:181 | keep | read by the body; no change
- Clean/gantry_dynamic/evaluation.py:169 | nrms [parameter of _record_figure] | compute | reader: evaluation.py:179 | keep | read by the body; no change
- Clean/gantry_dynamic/evaluation.py:169 | floor [parameter of _record_figure] | compute | reader: evaluation.py:179 | keep | read by the body; no change
- Clean/gantry_dynamic/evaluation.py:195 | path [parameter of _loss_figure] | compute | reader: evaluation.py:215 | keep | read by the body; no change
- Clean/gantry_dynamic/evaluation.py:195 | title [parameter of _loss_figure] | compute | reader: evaluation.py:213 | keep | read by the body; no change
- Clean/gantry_dynamic/evaluation.py:195 | loss [parameter of _loss_figure] | compute | reader: evaluation.py:199 | keep | read by the body; no change
- Clean/gantry_dynamic/evaluation.py:218 | run_dir [parameter of evaluate] | compute | reader: evaluation.py:233 | keep | read by the body; no change
- Clean/gantry_dynamic/evaluation.py:218 | data_root=DATA_ROOT [parameter of evaluate] | compute | reader: evaluation.py:236 | keep | read by the body; no change
- Clean/gantry_dynamic/model.py:28 | hp [parameter of build_model] | compute | reader: model.py:39 | keep | K30 ruled (amendment 9: readable signature); hp is always cfg.hp
- Clean/gantry_dynamic/model.py:28 | cfg [parameter of build_model] | compute | reader: model.py:40 | keep | read by the body; no change
- Clean/gantry_dynamic/model.py:28 | data [parameter of build_model] | compute | reader: model.py:111 | keep | read by the body; no change
- Clean/gantry_dynamic/model.py:28 | norm [parameter of build_model] | compute | reader: model.py:57 | keep | read by the body; no change
- Clean/gantry_dynamic/model.py:126 | fit_sys [parameter of on_device] | compute | reader: model.py:129 | keep | read by the body; no change
- Clean/gantry_dynamic/model.py:126 | cfg [parameter of on_device] | compute | reader: model.py:128 | keep | read by the body; no change
- Clean/gantry_dynamic/model.py:137 | fit_sys [parameter of prefit_physical_encoder] | compute | reader: model.py:151 | keep | read by the body; no change
- Clean/gantry_dynamic/model.py:137 | cfg [parameter of prefit_physical_encoder] | compute | reader: model.py:151 | keep | read by the body; no change
- Clean/gantry_dynamic/model.py:137 | data [parameter of prefit_physical_encoder] | compute | reader: model.py:162 | keep | read by the body; no change
- Clean/gantry_dynamic/model.py:137 | norm [parameter of prefit_physical_encoder] | compute | reader: model.py:163 | keep | read by the body; no change
- Clean/gantry_dynamic/model.py:180 | x [parameter of prefit_physical_encoder.recorder] | compute | reader: model.py:182 | keep | read by the body; no change
- Clean/gantry_dynamic/model.py:180 | u [parameter of prefit_physical_encoder.recorder] | compute | reader: model.py:183 | keep | read by the body; no change
- Clean/gantry_dynamic/model.py:180 | args=* [parameter of prefit_physical_encoder.recorder] | compute | reader: model.py:184 | keep | read by the body; no change
- Clean/gantry_dynamic/model.py:210 | enc [parameter of least_squares_start] | compute | reader: model.py:228 | keep | read by the body; no change
- Clean/gantry_dynamic/model.py:210 | nx [parameter of least_squares_start] | compute | reader: model.py:256 | keep | read by the body; no change
- Clean/gantry_dynamic/model.py:210 | uh [parameter of least_squares_start] | compute | reader: model.py:231 | keep | read by the body; no change
- Clean/gantry_dynamic/model.py:210 | yh [parameter of least_squares_start] | compute | reader: model.py:231 | keep | read by the body; no change
- Clean/gantry_dynamic/model.py:210 | xt [parameter of least_squares_start] | compute | reader: model.py:231 | keep | read by the body; no change
- Clean/gantry_dynamic/obc_gantry.py:40 | q [parameter of fourth_order_central_difference] | compute | reader: obc_gantry.py:52 | keep | read by the body; no change
- Clean/gantry_dynamic/obc_gantry.py:40 | ts [parameter of fourth_order_central_difference] | compute | reader: obc_gantry.py:52 | keep | read by the body; no change
- Clean/gantry_dynamic/obc_gantry.py:55 | cfg [parameter of build_reference_set] | compute | reader: obc_gantry.py:71 | keep | read by the body; no change
- Clean/gantry_dynamic/obc_gantry.py:55 | norm [parameter of build_reference_set] | compute | reader: obc_gantry.py:72 | keep | read by the body; no change
- Clean/gantry_dynamic/obc_gantry.py:55 | nx_ann [parameter of build_reference_set] | compute, mainline-compat | reader: obc_gantry.py:97 | rename | n_a, as K8 did for RunConfig (mainline NX_ANN); bitwise; harness keyword callers capture_l12.py:102, l4_loader.py:329, :360 follow
- Clean/gantry_dynamic/obc_gantry.py:55 | files [parameter of build_reference_set] | compute | reader: obc_gantry.py:79 | keep | read by the body; no change
- Clean/gantry_dynamic/obc_gantry.py:55 | verbose=True [parameter of build_reference_set] | verification | reader: obc_gantry.py:99 | keep | K6 ruled; only the harness passes it
- Clean/gantry_dynamic/obc_gantry.py:108 | phy_block [parameter of GantryOBCCorrection.__init__] | compute | reader: obc_gantry.py:123 | keep | read by the body; no change
- Clean/gantry_dynamic/obc_gantry.py:108 | nx [parameter of GantryOBCCorrection.__init__] | compute | reader: obc_gantry.py:124 | keep | read by the body; no change
- Clean/gantry_dynamic/obc_gantry.py:108 | nx_phys [parameter of GantryOBCCorrection.__init__] | compute | reader: obc_gantry.py:124 | keep | read by the body; no change
- Clean/gantry_dynamic/obc_gantry.py:108 | route_ix [parameter of GantryOBCCorrection.__init__] | fixed-option, mainline-compat | reader: obc_gantry.py:125 | remove | K32 (unruled, re-raised): always range(6 + n_a), so row_ix = field_cols = range(nx_phys); build E from nx_phys; bitwise; with ann_route_ix (A3)
- Clean/gantry_dynamic/obc_gantry.py:108 | dtype [parameter of GantryOBCCorrection.__init__] | compute | reader: obc_gantry.py:130 | keep | read by the body; no change
- Clean/gantry_dynamic/obc_gantry.py:143 | theta [parameter of GantryOBCCorrection.set_frozen] | compute | reader: obc_gantry.py:145 | keep | read by the body; no change
- Clean/gantry_dynamic/obc_gantry.py:143 | vbar [parameter of GantryOBCCorrection.set_frozen] | compute | reader: obc_gantry.py:146 | keep | read by the body; no change
- Clean/gantry_dynamic/obc_gantry.py:148 | theta [parameter of GantryOBCCorrection.tangent_pair] | compute | reader: obc_gantry.py:158 | keep | read by the body; no change
- Clean/gantry_dynamic/obc_gantry.py:162 | x [parameter of GantryOBCCorrection.forward] | compute | reader: obc_gantry.py:174 | keep | read by the body; no change
- Clean/gantry_dynamic/obc_gantry.py:162 | u [parameter of GantryOBCCorrection.forward] | compute | reader: obc_gantry.py:174 | keep | read by the body; no change
- Clean/gantry_dynamic/obc_gantry.py:162 | obc_pass [parameter of GantryOBCCorrection.forward] | compute | reader: obc_gantry.py:175 | keep | read by the body; no change
- Clean/gantry_dynamic/obc_gantry.py:183 | phy_block [parameter of make_float64_block] | compute | reader: obc_gantry.py:185 | keep | read by the body; no change
- Clean/gantry_dynamic/obc_gantry.py:188 | blk64 [parameter of make_basis_builder] | compute | reader: obc_gantry.py:207 | keep | read by the body; no change
- Clean/gantry_dynamic/obc_gantry.py:188 | ref [parameter of make_basis_builder] | compute | reader: obc_gantry.py:212 | keep | read by the body; no change
- Clean/gantry_dynamic/obc_gantry.py:188 | row_ix [parameter of make_basis_builder] | compute | reader: obc_gantry.py:212 | keep | read by the body; no change
- Clean/gantry_dynamic/obc_gantry.py:188 | chunk [parameter of make_basis_builder] | compute | reader: obc_gantry.py:212 | keep | read by the body; no change
- Clean/gantry_dynamic/obc_gantry.py:188 | device=None [parameter of make_basis_builder] | fixed-option | reader: obc_gantry.py:206 | keep | K17 ruled
- Clean/gantry_dynamic/obc_gantry.py:204 | vbar64 [parameter of make_basis_builder.build] | compute | reader: obc_gantry.py:206 | keep | read by the body; no change
- Clean/gantry_dynamic/obc_gantry.py:218 | ref [parameter of make_reference_field] | compute | reader: obc_gantry.py:237 | keep | read by the body; no change
- Clean/gantry_dynamic/obc_gantry.py:218 | field_cols [parameter of make_reference_field] | compute | reader: obc_gantry.py:228 | keep | read by the body; no change
- Clean/gantry_dynamic/obc_gantry.py:232 | ann [parameter of make_reference_field.field] | compute | reader: obc_gantry.py:234 | keep | read by the body; no change
- Clean/gantry_dynamic/obc_gantry.py:245 | fit_sys [parameter of attach_obc] | compute | reader: obc_gantry.py:247 | keep | read by the body; no change
- Clean/gantry_dynamic/obc_gantry.py:245 | cfg [parameter of attach_obc] | compute | reader: obc_gantry.py:249 | keep | read by the body; no change
- Clean/gantry_dynamic/obc_gantry.py:245 | data [parameter of attach_obc] | compute | reader: obc_gantry.py:249 | keep | read by the body; no change
- Clean/gantry_dynamic/obc_gantry.py:245 | norm [parameter of attach_obc] | compute | reader: obc_gantry.py:249 | keep | read by the body; no change
- Clean/gantry_dynamic/training.py:17 | obj [parameter of _json_safe] | compute | reader: training.py:19 | keep | read by the body; no change
- Clean/gantry_dynamic/training.py:30 | fit_sys [parameter of save_checkpoint_pair] | compute | reader: training.py:33 | keep | read by the body; no change
- Clean/gantry_dynamic/training.py:30 | base_path [parameter of save_checkpoint_pair] | compute | reader: training.py:38 | keep | read by the body; no change
- Clean/gantry_dynamic/training.py:30 | include_optimizer=True [parameter of save_checkpoint_pair] | compute | reader: training.py:36 | keep | read by the body; no change
- Clean/gantry_dynamic/training.py:30 | meta=** [parameter of save_checkpoint_pair] | compute | reader: training.py:39 | keep | read by the body; no change
- Clean/gantry_dynamic/training.py:42 | base_path [parameter of resolve_checkpoint] | compute | reader: training.py:44 | keep | read by the body; no change
- Clean/gantry_dynamic/training.py:71 | optimizer [parameter of load_optimizer_moments] | compute | reader: training.py:73 | keep | read by the body; no change
- Clean/gantry_dynamic/training.py:71 | saved [parameter of load_optimizer_moments] | compute | reader: training.py:74 | keep | read by the body; no change
- Clean/gantry_dynamic/training.py:86 | target_sd [parameter of _assert_same_architecture] | compute | reader: training.py:88 | keep | read by the body; no change
- Clean/gantry_dynamic/training.py:86 | ckpt_sd [parameter of _assert_same_architecture] | compute | reader: training.py:89 | keep | read by the body; no change
- Clean/gantry_dynamic/training.py:86 | what [parameter of _assert_same_architecture] | compute | reader: training.py:94 | keep | read by the body; no change
- Clean/gantry_dynamic/training.py:109 | fit_sys [parameter of _assert_same_dtype] | compute | reader: training.py:111 | keep | read by the body; no change
- Clean/gantry_dynamic/training.py:109 | hfn_sd [parameter of _assert_same_dtype] | compute | reader: training.py:112 | keep | read by the body; no change
- Clean/gantry_dynamic/training.py:120 | fit_sys [parameter of load_checkpoint] | compute | reader: training.py:130 | keep | read by the body; no change
- Clean/gantry_dynamic/training.py:120 | base_path [parameter of load_checkpoint] | compute | reader: training.py:122 | keep | read by the body; no change
- Clean/gantry_dynamic/training.py:120 | joint_estimation [parameter of load_checkpoint] | write-only | reader: none | remove | K7 (unruled, re-raised): never read; bitwise; callers main:129, evaluation.py:241, l4_loader.py:432, :468, run_l3bc.py:94 drop it
- Clean/gantry_dynamic/training.py:120 | start_phase='adam' [parameter of load_checkpoint] | compute | reader: training.py:124 | keep | every caller passes it; the default is the training start; no change
- Clean/gantry_dynamic/training.py:157 | fit_sys [parameter of ValidationHistory.__init__] | compute | reader: training.py:159 | keep | read by the body; no change
- Clean/gantry_dynamic/training.py:157 | hist_path [parameter of ValidationHistory.__init__] | compute | reader: training.py:160 | keep | read by the body; no change
- Clean/gantry_dynamic/training.py:163 | event [parameter of ValidationHistory.record] | compute | reader: training.py:172 | keep | read by the body; no change
- Clean/gantry_dynamic/training.py:163 | fit_sys [parameter of ValidationHistory.record] | compute | reader: training.py:167 | keep | read by the body; no change
- Clean/gantry_dynamic/training.py:163 | score [parameter of ValidationHistory.record] | compute | reader: training.py:169 | keep | read by the body; no change
- Clean/gantry_dynamic/training.py:163 | row=None [parameter of ValidationHistory.record] | compute | reader: training.py:166 | keep | read by the body; no change
- Clean/gantry_dynamic/training.py:163 | extra=** [parameter of ValidationHistory.record] | compute | reader: training.py:172 | keep | read by the body; no change
- Clean/gantry_dynamic/training.py:188 | fit_sys [parameter of ValidationHistory.__call__] | compute | reader: training.py:190 | keep | read by the body; no change
- Clean/gantry_dynamic/training.py:188 | val_sys_data [parameter of ValidationHistory.__call__] | compute | reader: interconnect.py:379 (probe protocol) | keep | the probe signature cal_validation_error calls; no change
- Clean/gantry_dynamic/training.py:188 | value [parameter of ValidationHistory.__call__] | compute | reader: training.py:190 | keep | read by the body; no change
- Clean/gantry_dynamic/training.py:193 | fit_sys [parameter of run_lbfgs_polish] | compute | reader: training.py:201 | keep | read by the body; no change
- Clean/gantry_dynamic/training.py:193 | hp [parameter of run_lbfgs_polish] | compute | reader: training.py:221 | keep | K30 ruled (amendment 9: readable signature); hp is always cfg.hp
- Clean/gantry_dynamic/training.py:193 | cfg [parameter of run_lbfgs_polish] | compute | reader: training.py:197 | keep | read by the body; no change
- Clean/gantry_dynamic/training.py:193 | data [parameter of run_lbfgs_polish] | compute | reader: training.py:211 | keep | read by the body; no change
- Clean/gantry_dynamic/training.py:193 | bestfit [parameter of run_lbfgs_polish] | compute | reader: training.py:207 | keep | read by the body; no change
- Clean/gantry_dynamic/training.py:193 | history [parameter of run_lbfgs_polish] | compute | reader: training.py:216 | keep | read by the body; no change
- Clean/gantry_dynamic/training.py:193 | adam_ran=True [parameter of run_lbfgs_polish] | compute | reader: training.py:208 | keep | train passes it always; harmless default; no change
- Clean/gantry_dynamic/training.py:231 | fit_sys [parameter of _adam_phase] | compute | reader: training.py:235 | keep | read by the body; no change
- Clean/gantry_dynamic/training.py:231 | hp [parameter of _adam_phase] | compute | reader: training.py:245 | keep | K30 ruled (amendment 9: readable signature); hp is always cfg.hp
- Clean/gantry_dynamic/training.py:231 | cfg [parameter of _adam_phase] | compute | reader: training.py:241 | keep | read by the body; no change
- Clean/gantry_dynamic/training.py:231 | data [parameter of _adam_phase] | compute | reader: training.py:244 | keep | read by the body; no change
- Clean/gantry_dynamic/training.py:231 | checkpoint_dir [parameter of _adam_phase] | compute | reader: training.py:248 | keep | read by the body; no change
- Clean/gantry_dynamic/training.py:231 | meta [parameter of _adam_phase] | compute | reader: training.py:240 | keep | read by the body; no change
- Clean/gantry_dynamic/training.py:231 | done_epochs [parameter of _adam_phase] | compute | reader: training.py:242 | keep | read by the body; no change
- Clean/gantry_dynamic/training.py:231 | epochs_remaining [parameter of _adam_phase] | compute | reader: training.py:253 | keep | read by the body; no change
- Clean/gantry_dynamic/training.py:231 | polish_only [parameter of _adam_phase] | compute | reader: training.py:241 | keep | read by the body; no change
- Clean/gantry_dynamic/training.py:275 | fit_sys [parameter of train] | compute | reader: training.py:282 | keep | read by the body; no change
- Clean/gantry_dynamic/training.py:275 | hp [parameter of train] | compute | reader: training.py:296 | keep | K30 ruled (amendment 9: readable signature); hp is always cfg.hp
- Clean/gantry_dynamic/training.py:275 | cfg [parameter of train] | compute | reader: training.py:280 | keep | read by the body; no change
- Clean/gantry_dynamic/training.py:275 | data [parameter of train] | compute | reader: training.py:304 | keep | read by the body; no change
- Clean/gantry_dynamic/training.py:275 | checkpoint_dir [parameter of train] | compute | reader: training.py:304 | keep | read by the body; no change
- Clean/gantry_dynamic/training.py:275 | run_id [parameter of train] | compute | reader: training.py:282 | keep | read by the body; no change
- Clean/gantry_dynamic/training.py:275 | resume=None [parameter of train] | compute | reader: training.py:286 | keep | read by the body; no change
- Clean/gantry_interconnect_dynamic.py:26 | run_id [parameter of _dumps] | compute | reader: gantry_interconnect_dynamic.py:28 | keep | read by the body; no change
- Clean/gantry_interconnect_dynamic.py:32 | kind [parameter of _rerun_source] | compute | reader: gantry_interconnect_dynamic.py:38 | keep | read by the body; no change
- Clean/gantry_interconnect_dynamic.py:32 | run_dir [parameter of _rerun_source] | compute | reader: gantry_interconnect_dynamic.py:35 | keep | read by the body; no change
- Clean/gantry_interconnect_dynamic.py:32 | source [parameter of _rerun_source] | compute | reader: gantry_interconnect_dynamic.py:42 | keep | read by the body; no change
- Clean/gantry_interconnect_dynamic.py:51 | kind [parameter of _archive_rerun] | compute | reader: gantry_interconnect_dynamic.py:60 | keep | read by the body; no change
- Clean/gantry_interconnect_dynamic.py:51 | run_dir [parameter of _archive_rerun] | compute | reader: gantry_interconnect_dynamic.py:55 | keep | read by the body; no change
- Clean/gantry_interconnect_dynamic.py:51 | cj [parameter of _archive_rerun] | compute | reader: gantry_interconnect_dynamic.py:65 | keep | read by the body; no change
- Clean/gantry_interconnect_dynamic.py:51 | run_id [parameter of _archive_rerun] | compute | reader: gantry_interconnect_dynamic.py:61 | keep | read by the body; no change
- Clean/gantry_interconnect_dynamic.py:51 | src [parameter of _archive_rerun] | compute | reader: gantry_interconnect_dynamic.py:64 | keep | read by the body; no change
- Clean/gantry_interconnect_dynamic.py:74 | cfg [parameter of main] | compute | reader: gantry_interconnect_dynamic.py:82 | keep | read by the body; no change
- Clean/gantry_interconnect_dynamic.py:74 | run_id=None [parameter of main] | verification | reader: gantry_interconnect_dynamic.py:81 | keep | the CLI never passes it, the harness does; a user naming a run from Python would too; no change
- Clean/gantry_interconnect_dynamic.py:74 | rerun=None [parameter of main] | compute | reader: gantry_interconnect_dynamic.py:80 | keep | read by the body; no change
- Clean/gantry_interconnect_dynamic.py:74 | evaluate_run=EVALUATE [parameter of main] | compute | reader: gantry_interconnect_dynamic.py:137 | keep | amendment 9 point 3 and implementation choice 4 (harness passes False); no change
- Clean/model_augmentation/fit_systems/blocks.py:25 | nz [parameter of Block.__init__, Jan's code] | compute | reader: blocks.py:30 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:25 | nw [parameter of Block.__init__, Jan's code] | compute | reader: blocks.py:29 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:25 | kwargs=** [parameter of Block.__init__, Jan's code] | compute | reader: blocks.py:27 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:35 | nz [parameter of Static_ANN_Block.__init__, Jan's code] | compute | reader: blocks.py:50 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:35 | nw [parameter of Static_ANN_Block.__init__, Jan's code] | compute | reader: blocks.py:50 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:35 | net [parameter of Static_ANN_Block.__init__, Jan's code] | compute | reader: blocks.py:53 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:35 | n_nodes_per_layer [parameter of Static_ANN_Block.__init__, Jan's code] | compute | reader: blocks.py:56 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:35 | n_hidden_layers [parameter of Static_ANN_Block.__init__, Jan's code] | compute | reader: blocks.py:57 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:35 | activation [parameter of Static_ANN_Block.__init__, Jan's code] | compute | reader: blocks.py:58 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:35 | net_kwargs [parameter of Static_ANN_Block.__init__, Jan's code] | compute | reader: blocks.py:59 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:65 | z [parameter of Static_ANN_Block.forward, Jan's code] | compute | reader: blocks.py:77 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:85 | C [parameter of Linear_Output_Block.__init__, Jan's code] | compute | reader: blocks.py:87 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:85 | D [parameter of Linear_Output_Block.__init__, Jan's code] | compute | reader: blocks.py:88 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:99 | z [parameter of Linear_Output_Block.forward, Jan's code] | compute | reader: blocks.py:111 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:119 | z [parameter of Discrete_Nonlinear_Function_Block.forward, Jan's code] | compute | reader: blocks.py:122 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:119 | mats=None [parameter of Discrete_Nonlinear_Function_Block.forward, Jan's code] | misleading | reader: blocks.py:122 | remove | drop the None default: since K14 a call without matrices fails inside _rk4; Jan's code (CHANGED line); bitwise
- Clean/model_augmentation/fit_systems/blocks.py:130 | std_x [parameter of GantryStateBlock.__init__] | compute | reader: blocks.py:179 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:130 | std_u [parameter of GantryStateBlock.__init__] | compute | reader: blocks.py:181 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:130 | x_mean [parameter of GantryStateBlock.__init__] | compute | reader: blocks.py:182 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:130 | u_mean [parameter of GantryStateBlock.__init__] | compute | reader: blocks.py:183 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:130 | Ts [parameter of GantryStateBlock.__init__] | compute | reader: blocks.py:144 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:130 | up_sample [parameter of GantryStateBlock.__init__] | compute | reader: blocks.py:145 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:130 | closed_form=False [parameter of GantryStateBlock.__init__] | compute | reader: blocks.py:148 | keep | the default is the thesis form; build_model passes it; no change
- Clean/model_augmentation/fit_systems/blocks.py:130 | args=* [parameter of GantryStateBlock.__init__] | compute | reader: blocks.py:142 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:130 | kwargs=** [parameter of GantryStateBlock.__init__] | compute | reader: blocks.py:142 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:185 | x [parameter of GantryStateBlock._rk4] | compute | reader: blocks.py:201 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:185 | u [parameter of GantryStateBlock._rk4] | compute | reader: blocks.py:201 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:185 | mats [parameter of GantryStateBlock._rk4] | compute | reader: blocks.py:201 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:208 | x [parameter of GantryStateBlock._deriv_with] | compute | reader: blocks.py:225 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:208 | u [parameter of GantryStateBlock._deriv_with] | compute | reader: blocks.py:226 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:208 | mats [parameter of GantryStateBlock._deriv_with] | compute | reader: blocks.py:222 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:270 | x [parameter of GantryStateBlock._deriv_with_tangent] | compute | reader: blocks.py:292 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:270 | sx [parameter of GantryStateBlock._deriv_with_tangent] | compute | reader: blocks.py:291 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:270 | u [parameter of GantryStateBlock._deriv_with_tangent] | compute | reader: blocks.py:293 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:270 | mats [parameter of GantryStateBlock._deriv_with_tangent] | compute | reader: blocks.py:285 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:270 | dmats [parameter of GantryStateBlock._deriv_with_tangent] | compute | reader: blocks.py:287 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:355 | x [parameter of GantryStateBlock._rk4_with_tangent] | compute | reader: blocks.py:371 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:355 | sx [parameter of GantryStateBlock._rk4_with_tangent] | compute | reader: blocks.py:371 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:355 | u [parameter of GantryStateBlock._rk4_with_tangent] | compute | reader: blocks.py:371 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:355 | mats [parameter of GantryStateBlock._rk4_with_tangent] | compute | reader: blocks.py:371 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:355 | dmats [parameter of GantryStateBlock._rk4_with_tangent] | compute | reader: blocks.py:371 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:392 | RMSE_baseline=1.0 [parameter of _TrainableGantryStateBlock.__init__] | write-only, mainline-compat | reader: blocks.py:400 | remove | P3: only feeds Lambda_combo; bitwise; model.py:59 drops the keyword
- Clean/model_augmentation/fit_systems/blocks.py:392 | params_init=None [parameter of _TrainableGantryStateBlock.__init__] | misleading | reader: blocks.py:396 | remove | drop the None default: to_tensor(None) fails since its CHANGED line (utils.py:15); bitwise
- Clean/model_augmentation/fit_systems/blocks.py:392 | args=* [parameter of _TrainableGantryStateBlock.__init__] | compute | reader: blocks.py:394 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:392 | kwargs=** [parameter of _TrainableGantryStateBlock.__init__] | compute | reader: blocks.py:394 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:405 | p [parameter of _TrainableGantryStateBlock._build_KC] | compute | reader: blocks.py:417 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:436 | mh [parameter of _TrainableGantryStateBlock._build_M1M2] | compute | reader: blocks.py:446 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:461 | p [parameter of _TrainableGantryStateBlock._mats_from_raw] | compute | reader: blocks.py:473 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:483 | z [parameter of _TrainableGantryStateBlock.nonlinear_function] | compute | reader: blocks.py:494 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:483 | mats [parameter of _TrainableGantryStateBlock.nonlinear_function] | compute | reader: blocks.py:494 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:515 | combo_init=None [parameter of IdentifiableGantryStateBlock.__init__] | misleading | reader: blocks.py:528 | remove | drop the None default: to_tensor(None) fails; bitwise
- Clean/model_augmentation/fit_systems/blocks.py:515 | args=* [parameter of IdentifiableGantryStateBlock.__init__] | compute | reader: blocks.py:524 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:515 | kwargs=** [parameter of IdentifiableGantryStateBlock.__init__] | compute | reader: blocks.py:524 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:550 | raw [parameter of IdentifiableGantryStateBlock.combos_of] | compute | reader: blocks.py:562 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:550 | Lb [parameter of IdentifiableGantryStateBlock.combos_of] | compute | reader: blocks.py:563 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:571 | combo [parameter of IdentifiableGantryStateBlock.gauge_section] | compute | reader: blocks.py:584 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:571 | raw_init [parameter of IdentifiableGantryStateBlock.gauge_section] | compute | reader: blocks.py:586 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:571 | Lb [parameter of IdentifiableGantryStateBlock.gauge_section] | compute | reader: blocks.py:585 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:600 | combo [parameter of IdentifiableGantryStateBlock._m_diff_bound] | compute | reader: blocks.py:611 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:614 | free [parameter of IdentifiableGantryStateBlock.combinations_from_free] | compute | reader: blocks.py:627 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:633 | combo [parameter of IdentifiableGantryStateBlock.free_from_combinations] | compute | reader: blocks.py:645 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:671 | free [parameter of IdentifiableGantryStateBlock.transition_from_free] | compute | reader: blocks.py:682 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:671 | z [parameter of IdentifiableGantryStateBlock.transition_from_free] | compute | reader: blocks.py:683 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:685 | free [parameter of IdentifiableGantryStateBlock.mats_and_tangent_from_free] | compute | reader: blocks.py:696 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:685 | dfree [parameter of IdentifiableGantryStateBlock.mats_and_tangent_from_free] | compute | reader: blocks.py:696 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/blocks.py:698 | free [parameter of IdentifiableGantryStateBlock._mats_at] | compute | reader: blocks.py:701 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:21 | A [parameter of ControllerBank.__init__] | compute | reader: closed_loop.py:38 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:21 | B [parameter of ControllerBank.__init__] | compute | reader: closed_loop.py:38 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:21 | C [parameter of ControllerBank.__init__] | compute | reader: closed_loop.py:38 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:21 | D [parameter of ControllerBank.__init__] | compute | reader: closed_loop.py:38 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:21 | ystd [parameter of ControllerBank.__init__] | compute | reader: closed_loop.py:39 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:21 | std_u [parameter of ControllerBank.__init__] | compute | reader: closed_loop.py:39 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:21 | dtype=torch.float32 [parameter of ControllerBank.__init__] | fixed-option, misleading | reader: closed_loop.py:37 | remove | drop the float32 default; the only caller passes the model dtype (controller.py:187); bitwise
- Clean/model_augmentation/fit_systems/closed_loop.py:51 | xc [parameter of ControllerBank.step] | compute | reader: closed_loop.py:67 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:51 | e_norm [parameter of ControllerBank.step] | compute | reader: closed_loop.py:66 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:51 | ctrl [parameter of ControllerBank.step] | compute | reader: closed_loop.py:64 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:71 | hfn [parameter of closed_loop_step] | compute | reader: closed_loop.py:90 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:71 | bank [parameter of closed_loop_step] | compute | reader: closed_loop.py:92 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:85 | x [parameter of closed_loop_step.step] | compute | reader: closed_loop.py:90 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:85 | xc [parameter of closed_loop_step.step] | compute | reader: closed_loop.py:92 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:85 | u_t [parameter of closed_loop_step.step] | compute | reader: closed_loop.py:94 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:85 | y_t [parameter of closed_loop_step.step] | compute | reader: closed_loop.py:92 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:85 | ctrl [parameter of closed_loop_step.step] | compute | reader: closed_loop.py:92 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:85 | obc_pass [parameter of closed_loop_step.step] | compute | reader: closed_loop.py:94 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:85 | pass_mats [parameter of closed_loop_step.step] | compute | reader: closed_loop.py:94 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:99 | step [parameter of closed_loop_rollout] | compute | reader: closed_loop.py:126 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:99 | u_data [parameter of closed_loop_rollout] | compute | reader: closed_loop.py:116 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:99 | y_data [parameter of closed_loop_rollout] | compute | reader: closed_loop.py:125 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:99 | x0 [parameter of closed_loop_rollout] | compute | reader: closed_loop.py:124 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:99 | bank [parameter of closed_loop_rollout] | compute | reader: closed_loop.py:116 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:99 | ctrl_ix [parameter of closed_loop_rollout] | compute | reader: closed_loop.py:118 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:99 | obc_pass=None [parameter of closed_loop_rollout] | compute | reader: closed_loop.py:126 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:99 | pass_mats=None [parameter of closed_loop_rollout] | misleading | reader: closed_loop.py:126 | remove | drop the None default: required since K14 (every caller passes objective_matrices); bitwise
- Clean/model_augmentation/fit_systems/closed_loop.py:131 | fit_sys [parameter of window_counts] | compute | reader: closed_loop.py:133 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:131 | sdl [parameter of window_counts] | compute | reader: closed_loop.py:134 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:131 | nf [parameter of window_counts] | compute | reader: closed_loop.py:133 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:131 | stride [parameter of window_counts] | compute | reader: closed_loop.py:134 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:140 | bank [parameter of ClosedLoopSimulator.__init__] | compute | reader: closed_loop.py:142 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:140 | train_ctrl_rows [parameter of ClosedLoopSimulator.__init__] | compute | reader: closed_loop.py:149 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:140 | val_rows [parameter of ClosedLoopSimulator.__init__] | compute | reader: closed_loop.py:152 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:140 | compiled [parameter of ClosedLoopSimulator.__init__] | compute | reader: closed_loop.py:148 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:140 | train_weights [parameter of ClosedLoopSimulator.__init__] | compute | reader: closed_loop.py:151 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:140 | val_sigma_e [parameter of ClosedLoopSimulator.__init__] | compute | reader: closed_loop.py:146 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:169 | fit_sys [parameter of ClosedLoopSimulator.__call__] | compute | reader: closed_loop.py:188 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:169 | x [parameter of ClosedLoopSimulator.__call__] | compute | reader: closed_loop.py:190 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:169 | ufuture [parameter of ClosedLoopSimulator.__call__] | compute | reader: closed_loop.py:183 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:169 | yfuture [parameter of ClosedLoopSimulator.__call__] | compute | reader: closed_loop.py:190 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:169 | ctrl_ix [parameter of ClosedLoopSimulator.__call__] | compute | reader: closed_loop.py:190 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:169 | obc_pass=None [parameter of ClosedLoopSimulator.__call__] | compute | reader: closed_loop.py:191 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:169 | pass_mats=None [parameter of ClosedLoopSimulator.__call__] | misleading | reader: closed_loop.py:191 | remove | drop the None default, as closed_loop_rollout; bitwise
- Clean/model_augmentation/fit_systems/closed_loop.py:169 | kw=** [parameter of ClosedLoopSimulator.__call__] | compute | reader: none (absorbs nf and stride of loss_kwargs) | comment | say it absorbs the window keywords loss() passes on; bitwise
- Clean/model_augmentation/fit_systems/closed_loop.py:197 | data [parameter of ClosedLoopSimulator.augment_training_data] | compute | reader: closed_loop.py:206 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:197 | sys_data [parameter of ClosedLoopSimulator.augment_training_data] | compute | reader: closed_loop.py:201 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:197 | fit_sys [parameter of ClosedLoopSimulator.augment_training_data] | compute | reader: closed_loop.py:202 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:197 | kw=** [parameter of ClosedLoopSimulator.augment_training_data] | compute, fixed-option | reader: closed_loop.py:199 | keep | K38 ruled (see D9 on the default nf = 25)
- Clean/model_augmentation/fit_systems/closed_loop.py:216 | data [parameter of ClosedLoopSimulator.verify_by_content] | compute | reader: closed_loop.py:218 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:216 | sdl [parameter of ClosedLoopSimulator.verify_by_content] | compute | reader: closed_loop.py:220 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:216 | counts [parameter of ClosedLoopSimulator.verify_by_content] | compute | reader: closed_loop.py:220 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:216 | k0 [parameter of ClosedLoopSimulator.verify_by_content] | compute | reader: closed_loop.py:222 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:216 | nf [parameter of ClosedLoopSimulator.verify_by_content] | compute | reader: closed_loop.py:224 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:216 | stride [parameter of ClosedLoopSimulator.verify_by_content] | compute | reader: closed_loop.py:222 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:231 | fit_sys [parameter of ClosedLoopSimulator.validation_error] | compute | reader: closed_loop.py:244 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:231 | val_sys_data [parameter of ClosedLoopSimulator.validation_error] | compute | reader: closed_loop.py:243 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:254 | fit_sys [parameter of closed_loop_free_run_rms_batch] | compute | reader: closed_loop.py:268 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:254 | sdl [parameter of closed_loop_free_run_rms_batch] | compute | reader: closed_loop.py:278 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:254 | bank [parameter of closed_loop_free_run_rms_batch] | compute | reader: closed_loop.py:307 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:254 | ctrl_rows [parameter of closed_loop_free_run_rms_batch] | compute | reader: closed_loop.py:278 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/closed_loop.py:254 | return_pred=False [parameter of closed_loop_free_run_rms_batch] | compute | reader: closed_loop.py:317 | keep | K21 ruled; the evaluation uses True (evaluation.py:117)
- Clean/model_augmentation/fit_systems/interconnect.py:26 | nx [parameter of Interconnect.__init__, Jan's code] | compute | reader: interconnect.py:32 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:26 | nu [parameter of Interconnect.__init__, Jan's code] | compute | reader: interconnect.py:30 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:26 | ny [parameter of Interconnect.__init__, Jan's code] | compute | reader: interconnect.py:31 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:47 | fn [parameter of Interconnect._apply] | compute | reader: interconnect.py:49 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:47 | args=* [parameter of Interconnect._apply] | compute | reader: interconnect.py:49 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:47 | kwargs=** [parameter of Interconnect._apply] | compute | reader: interconnect.py:49 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:55 | x [parameter of Interconnect.forward, Jan's code] | compute | reader: interconnect.py:69 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:55 | u [parameter of Interconnect.forward, Jan's code] | compute | reader: interconnect.py:70 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:55 | obc_pass=None [parameter of Interconnect.forward, Jan's code] | compute | reader: interconnect.py:88 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:55 | pass_mats=None [parameter of Interconnect.forward, Jan's code] | misleading | reader: interconnect.py:80 | remove | drop the None default: every caller of forward passes the matrices (closed_loop.py:94, model.py:184); Jan's code (CHANGED line); bitwise
- Clean/model_augmentation/fit_systems/interconnect.py:96 | x [parameter of Interconnect._propagate] | compute | reader: interconnect.py:99 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:96 | u [parameter of Interconnect._propagate] | compute | reader: interconnect.py:99 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:96 | order [parameter of Interconnect._propagate] | compute | reader: interconnect.py:103 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:96 | pass_mats [parameter of Interconnect._propagate] | compute | reader: interconnect.py:115 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:172 | target_output_ix [parameter of Interconnect._signal_cone] | compute | reader: interconnect.py:174 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:184 | x [parameter of Interconnect.output_only] | compute | reader: interconnect.py:194 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:205 | output_signal_ix [parameter of Interconnect.init_connection_matrices, Jan's code] | compute | reader: interconnect.py:230 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:205 | signal_connections [parameter of Interconnect.init_connection_matrices, Jan's code] | compute | reader: interconnect.py:214 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:247 | new_block [parameter of Interconnect.add_block, Jan's code] | compute | reader: interconnect.py:250 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:256 | input_signal [parameter of Interconnect.connect_signals, Jan's code] | compute | reader: interconnect.py:258 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:256 | output_signal [parameter of Interconnect.connect_signals, Jan's code] | compute | reader: interconnect.py:259 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:256 | connection_function_method=None [parameter of Interconnect.connect_signals, Jan's code] | compute | reader: interconnect.py:261 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:256 | connection_matrix=torch.empty(0, 0) [parameter of Interconnect.connect_signals, Jan's code] | compute | reader: interconnect.py:266 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:268 | block [parameter of Interconnect.connect_block_signals, Jan's code] | compute | reader: interconnect.py:271 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:268 | input_signal_list [parameter of Interconnect.connect_block_signals, Jan's code] | compute | reader: interconnect.py:270 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:268 | output_signal_list [parameter of Interconnect.connect_block_signals, Jan's code] | compute | reader: interconnect.py:273 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:276 | signal [parameter of Interconnect.determine_signal_ix, Jan's code] | compute | reader: interconnect.py:278 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:283 | signal_name [parameter of Interconnect.convert_signal_name_to_ix, Jan's code] | compute | reader: interconnect.py:285 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:290 | connection_function_method [parameter of Interconnect.parse_connection_function_method, Jan's code] | compute | reader: interconnect.py:293 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:297 | input_signal_ix [parameter of Signal_Connection.__init__, Jan's code] | compute | reader: interconnect.py:299 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:297 | output_signal_ix [parameter of Signal_Connection.__init__, Jan's code] | compute | reader: interconnect.py:300 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:297 | connection_function_method [parameter of Signal_Connection.__init__, Jan's code] | compute | reader: interconnect.py:302 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:297 | connection_matrix [parameter of Signal_Connection.__init__, Jan's code] | compute | reader: interconnect.py:303 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:307 | src [parameter of _replace_with_retry] | compute | reader: interconnect.py:312 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:307 | dst [parameter of _replace_with_retry] | compute | reader: interconnect.py:312 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:307 | tries=20 [parameter of _replace_with_retry] | compute | reader: interconnect.py:310 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:323 | data [parameter of DtypeDataLoader.__init__] | compute | reader: interconnect.py:329 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:323 | batch_size [parameter of DtypeDataLoader.__init__] | compute | reader: interconnect.py:333 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:323 | dtype [parameter of DtypeDataLoader.__init__] | compute | reader: interconnect.py:328 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:325 | d [parameter of DtypeDataLoader.__init__._conv] | compute | reader: interconnect.py:327 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:347 | interconnect [parameter of SSE_Interconnect.__init__, Jan's code] | compute | reader: interconnect.py:349 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:347 | na=5 [parameter of SSE_Interconnect.__init__, Jan's code] | fixed-option | reader: interconnect.py:349 | keep | Jan's default 5, never used (build_model passes na); Dirk decides
- Clean/model_augmentation/fit_systems/interconnect.py:347 | nb=5 [parameter of SSE_Interconnect.__init__, Jan's code] | fixed-option | reader: interconnect.py:349 | keep | Jan's default, never used; Dirk decides
- Clean/model_augmentation/fit_systems/interconnect.py:347 | na_right=0 [parameter of SSE_Interconnect.__init__, Jan's code] | fixed-option | reader: interconnect.py:349 | keep | always 0 in Clean/; passed to deepSI; Dirk decides
- Clean/model_augmentation/fit_systems/interconnect.py:347 | nb_right=0 [parameter of SSE_Interconnect.__init__, Jan's code] | fixed-option | reader: interconnect.py:349 | keep | always 0 in Clean/; Dirk decides
- Clean/model_augmentation/fit_systems/interconnect.py:356 | sys_data=None [parameter of SSE_Interconnect.init_model, Jan's code] | misleading | reader: interconnect.py:358 | keep | the None default fails at sys_data.nu; Jan's signature; Dirk decides
- Clean/model_augmentation/fit_systems/interconnect.py:356 | optimizer_kwargs={} [parameter of SSE_Interconnect.init_model, Jan's code] | compute | reader: interconnect.py:359 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:364 | x [parameter of SSE_Interconnect.simulate] | compute | reader: interconnect.py:366 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:364 | ufuture [parameter of SSE_Interconnect.simulate] | compute | reader: interconnect.py:366 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:364 | yfuture=None [parameter of SSE_Interconnect.simulate] | compute | reader: interconnect.py:366 | keep | K25 ruled
- Clean/model_augmentation/fit_systems/interconnect.py:364 | Loss_kwargs=** [parameter of SSE_Interconnect.simulate] | compute | reader: interconnect.py:366 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:369 | sys_data [parameter of SSE_Interconnect.make_training_data] | compute | reader: interconnect.py:371 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:369 | Loss_kwargs=** [parameter of SSE_Interconnect.make_training_data] | compute | reader: interconnect.py:371 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:375 | val_sys_data [parameter of SSE_Interconnect.cal_validation_error] | compute | reader: interconnect.py:377 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:375 | validation_measure='sim-NRMS' [parameter of SSE_Interconnect.cal_validation_error] | write-only, misleading | reader: none | remove | never read, default 'sim-NRMS' names a measure Clean/ does not compute (the simulator's sim-NRMSe); bitwise; callers interconnect.py:408, lbfgs_polish.py:171, training.py:211, :213 drop the argument
- Clean/model_augmentation/fit_systems/interconnect.py:384 | train_sys_data [parameter of SSE_Interconnect.fit, Jan's code] | compute | reader: interconnect.py:432 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:384 | val_sys_data [parameter of SSE_Interconnect.fit, Jan's code] | compute | reader: interconnect.py:408 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:384 | epochs=30 [parameter of SSE_Interconnect.fit, Jan's code] | fixed-option | reader: interconnect.py:441 | keep | Jan's default 30, always passed; Dirk decides
- Clean/model_augmentation/fit_systems/interconnect.py:384 | n_its=None [parameter of SSE_Interconnect.fit, Jan's code] | compute | reader: interconnect.py:441 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:384 | batch_size=256 [parameter of SSE_Interconnect.fit, Jan's code] | fixed-option | reader: interconnect.py:439 | keep | Jan's default 256, always passed; Dirk decides
- Clean/model_augmentation/fit_systems/interconnect.py:384 | loss_kwargs={} [parameter of SSE_Interconnect.fit, Jan's code] | compute | reader: interconnect.py:432 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:384 | validation_measure='sim-NRMS' [parameter of SSE_Interconnect.fit, Jan's code] | misleading | reader: interconnect.py:408 | comment | read by prints only; Jan's default 'sim-NRMS' is not Clean's measure; set the default to 'sim-NRMSe' or drop it (every caller passes it); bitwise
- Clean/model_augmentation/fit_systems/interconnect.py:384 | its_per_val='epoch' [parameter of SSE_Interconnect.fit, Jan's code] | fixed-option | reader: interconnect.py:442 | keep | Jan's default 'epoch', never used (D11 branch); Dirk decides
- Clean/model_augmentation/fit_systems/interconnect.py:384 | cuda=False [parameter of SSE_Interconnect.fit, Jan's code] | compute | reader: interconnect.py:459 | keep | always passed; no change
- Clean/model_augmentation/fit_systems/interconnect.py:403 | train_loss=None [parameter of SSE_Interconnect.fit.validation, Jan's code] | compute | reader: interconnect.py:411 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:403 | time_elapsed_total=None [parameter of SSE_Interconnect.fit.validation, Jan's code] | compute | reader: interconnect.py:412 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:531 | args=* [parameter of AugmentedModel.__init__] | compute | reader: interconnect.py:533 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:531 | kwargs=** [parameter of AugmentedModel.__init__] | compute | reader: interconnect.py:533 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:546 | refresh_basis=False [parameter of AugmentedModel._sync_obc_coefficient] | compute | reader: interconnect.py:562 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:568 | theta_aux=None [parameter of AugmentedModel.objective_matrices] | compute | reader: interconnect.py:583 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:586 | uhist [parameter of AugmentedModel.loss] | compute | reader: interconnect.py:600 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:586 | yhist [parameter of AugmentedModel.loss] | compute | reader: interconnect.py:600 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:586 | ufuture [parameter of AugmentedModel.loss] | compute | reader: interconnect.py:615 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:586 | yfuture [parameter of AugmentedModel.loss] | compute | reader: interconnect.py:615 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:586 | Loss_kwargs=** [parameter of AugmentedModel.loss] | compute | reader: interconnect.py:612 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:621 | name='_best' [parameter of AugmentedModel.checkpoint_save_system] | compute | reader: interconnect.py:625 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:629 | args=* [parameter of AugmentedModel.checkpoint_load_system] | compute | reader: interconnect.py:634 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:629 | kwargs=** [parameter of AugmentedModel.checkpoint_load_system] | compute | reader: interconnect.py:634 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/interconnect.py:651 | opt [parameter of AugmentedModel.optimizer] | compute | reader: interconnect.py:653 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/lbfgs_polish.py:27 | fit_sys [parameter of _restore] | compute | reader: lbfgs_polish.py:34 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/lbfgs_polish.py:27 | params [parameter of _restore] | compute | reader: lbfgs_polish.py:32 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/lbfgs_polish.py:27 | snap [parameter of _restore] | compute | reader: lbfgs_polish.py:32 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/lbfgs_polish.py:41 | fit_sys [parameter of FixedWindowBatch.__init__] | compute | reader: lbfgs_polish.py:45 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/lbfgs_polish.py:41 | sys_data [parameter of FixedWindowBatch.__init__] | compute | reader: lbfgs_polish.py:45 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/lbfgs_polish.py:41 | loss_kwargs [parameter of FixedWindowBatch.__init__] | compute | reader: lbfgs_polish.py:43 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/lbfgs_polish.py:41 | n_windows [parameter of FixedWindowBatch.__init__] | compute | reader: lbfgs_polish.py:47 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/lbfgs_polish.py:41 | seed [parameter of FixedWindowBatch.__init__] | compute | reader: lbfgs_polish.py:59 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/lbfgs_polish.py:66 | a [parameter of FixedWindowBatch.__init__._to_tensor] | compute | reader: lbfgs_polish.py:68 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/lbfgs_polish.py:84 | fit_sys [parameter of _make_closure] | compute | reader: lbfgs_polish.py:90 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/lbfgs_polish.py:84 | batch [parameter of _make_closure] | compute | reader: lbfgs_polish.py:90 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/lbfgs_polish.py:84 | opt [parameter of _make_closure] | compute | reader: lbfgs_polish.py:89 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/lbfgs_polish.py:84 | counter [parameter of _make_closure] | compute | reader: lbfgs_polish.py:88 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/lbfgs_polish.py:97 | opt [parameter of _lbfgs_counts] | compute | reader: lbfgs_polish.py:99 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/lbfgs_polish.py:97 | params [parameter of _lbfgs_counts] | compute | reader: lbfgs_polish.py:99 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/lbfgs_polish.py:103 | exc [parameter of _is_oom] | compute | reader: lbfgs_polish.py:105 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/lbfgs_polish.py:109 | closure [parameter of _probe_closure] | compute | reader: lbfgs_polish.py:112 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/lbfgs_polish.py:109 | label [parameter of _probe_closure] | compute | reader: lbfgs_polish.py:116 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/lbfgs_polish.py:125 | out [parameter of _acceptance] | compute | reader: lbfgs_polish.py:127 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/lbfgs_polish.py:125 | validation_measure [parameter of _acceptance] | compute | reader: lbfgs_polish.py:132 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/lbfgs_polish.py:136 | fit_sys [parameter of lbfgs_polish] | compute | reader: lbfgs_polish.py:159 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/lbfgs_polish.py:136 | train_sys_data [parameter of lbfgs_polish] | compute | reader: lbfgs_polish.py:200 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/lbfgs_polish.py:136 | loss_kwargs [parameter of lbfgs_polish] | compute | reader: lbfgs_polish.py:200 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/lbfgs_polish.py:136 | spec [parameter of lbfgs_polish] | compute | reader: lbfgs_polish.py:150 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/lbfgs_polish.py:136 | val_sys_data [parameter of lbfgs_polish] | compute | reader: lbfgs_polish.py:171 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/lbfgs_polish.py:136 | val_before=None [parameter of lbfgs_polish] | compute | reader: lbfgs_polish.py:196 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/lbfgs_polish.py:136 | validation_measure [parameter of lbfgs_polish] | compute | reader: lbfgs_polish.py:171 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/lbfgs_polish.py:181 | reason [parameter of lbfgs_polish._abort] | compute | reader: lbfgs_polish.py:184 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/lbfgs_polish.py:181 | oom [parameter of lbfgs_polish._abort] | compute | reader: lbfgs_polish.py:187 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/lbfgs_polish.py:181 | word [parameter of lbfgs_polish._abort] | compute | reader: lbfgs_polish.py:189 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/obc.py:21 | J [parameter of OBCBasis.__init__] | compute | reader: obc.py:33 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/obc.py:21 | vbar [parameter of OBCBasis.__init__] | compute | reader: obc.py:50 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/obc.py:53 | F [parameter of OBCBasis.coefficient] | compute | reader: obc.py:64 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/obc.py:69 | device [parameter of OBCBasis.to] | compute | reader: obc.py:71 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/obc.py:82 | step [parameter of build_stacked_sensitivity] | compute | reader: obc.py:114 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/obc.py:82 | vbar [parameter of build_stacked_sensitivity] | compute | reader: obc.py:103 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/obc.py:82 | Z [parameter of build_stacked_sensitivity] | compute | reader: obc.py:111 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/obc.py:82 | row_ix [parameter of build_stacked_sensitivity] | compute | reader: obc.py:106 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/obc.py:82 | chunk=16384 [parameter of build_stacked_sensitivity] | fixed-option, unlabelled | reader: obc.py:111 | literal | drop the default 16384, a second copy of RunConfig.obc_ref_chunk; the builder passes it (obc_gantry.py:212); equation_tests.py:371 relies on the default and passes chunk explicitly after the change; bitwise
- Clean/model_augmentation/fit_systems/obc.py:82 | device=None [parameter of build_stacked_sensitivity] | fixed-option | reader: obc.py:105 | keep | K17 ruled
- Clean/model_augmentation/fit_systems/obc.py:82 | append_value=False [parameter of build_stacked_sensitivity] | fixed-option | reader: obc.py:116 | keep | K18 ruled; new evidence: no caller anywhere passes False (obc_gantry.py:213 and equation_tests.py:371 pass True), so the default names the variant nobody builds; Dirk may revisit
- Clean/model_augmentation/fit_systems/obc.py:131 | reference_field [parameter of OBCProjection.__init__] | compute | reader: obc.py:133 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/obc.py:131 | basis_builder [parameter of OBCProjection.__init__] | compute | reader: obc.py:134 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/obc.py:140 | vbar [parameter of OBCProjection.refresh] | compute | reader: obc.py:150 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/obc.py:140 | epoch [parameter of OBCProjection.refresh] | compute | reader: obc.py:160 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/obc.py:174 | epoch [parameter of OBCProjection.maybe_refresh] | compute | reader: obc.py:183 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/obc.py:174 | vbar_fn [parameter of OBCProjection.maybe_refresh] | compute | reader: obc.py:184 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/obc.py:186 | learning_component [parameter of OBCProjection.coefficient] | compute | reader: obc.py:196 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/pre_encoder.py:22 | nx [parameter of AugmentedEncoder.__init__] | compute | reader: pre_encoder.py:49 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/pre_encoder.py:22 | nu [parameter of AugmentedEncoder.__init__] | compute | reader: pre_encoder.py:47 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/pre_encoder.py:22 | ny [parameter of AugmentedEncoder.__init__] | compute | reader: pre_encoder.py:48 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/pre_encoder.py:22 | na [parameter of AugmentedEncoder.__init__] | compute | reader: pre_encoder.py:51 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/pre_encoder.py:22 | nb [parameter of AugmentedEncoder.__init__] | compute | reader: pre_encoder.py:52 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/pre_encoder.py:22 | nx_aug [parameter of AugmentedEncoder.__init__] | compute | reader: pre_encoder.py:50 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/pre_encoder.py:22 | n_nodes_per_layer [parameter of AugmentedEncoder.__init__] | compute | reader: pre_encoder.py:74 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/pre_encoder.py:22 | n_hidden_layers [parameter of AugmentedEncoder.__init__] | compute | reader: pre_encoder.py:74 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/pre_encoder.py:22 | dtype [parameter of AugmentedEncoder.__init__] | compute | reader: pre_encoder.py:61 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/pre_encoder.py:81 | uhist [parameter of AugmentedEncoder.forward] | compute | reader: pre_encoder.py:91 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/pre_encoder.py:81 | yhist [parameter of AugmentedEncoder.forward] | compute | reader: pre_encoder.py:92 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/ps2_opening.py:44 | values [parameter of _plateaued] | compute | reader: ps2_opening.py:46 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/ps2_opening.py:44 | window [parameter of _plateaued] | compute | reader: ps2_opening.py:46 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/ps2_opening.py:44 | tol [parameter of _plateaued] | compute | reader: ps2_opening.py:46 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/ps2_opening.py:49 | names [parameter of calibration_split] | compute | reader: ps2_opening.py:63 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/ps2_opening.py:72 | fit_sys [parameter of PS2Opening.__init__] | compute | reader: ps2_opening.py:80 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/ps2_opening.py:72 | data [parameter of PS2Opening.__init__] | compute | reader: ps2_opening.py:79 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/ps2_opening.py:72 | n_xb [parameter of PS2Opening.__init__] | compute | reader: ps2_opening.py:77 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/ps2_opening.py:72 | n_a [parameter of PS2Opening.__init__] | compute | reader: ps2_opening.py:77 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/ps2_opening.py:72 | width [parameter of PS2Opening.__init__] | compute | reader: ps2_opening.py:84 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/ps2_opening.py:72 | spec [parameter of PS2Opening.__init__] | compute | reader: ps2_opening.py:83 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/ps2_opening.py:72 | route_ix [parameter of PS2Opening.__init__] | fixed-option, mainline-compat | reader: ps2_opening.py:76 | remove | K32: _added_cols = range(n_xb, n_xb + n_a); bitwise
- Clean/model_augmentation/fit_systems/ps2_opening.py:99 | fs [parameter of PS2Opening.after_step] | compute | reader: ps2_opening.py:112 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/ps2_opening.py:149 | fs [parameter of PS2Opening._start] | compute | reader: ps2_opening.py:152 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/ps2_opening.py:168 | fs [parameter of PS2Opening._sample] | compute | reader: ps2_opening.py:179 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/ps2_opening.py:168 | rec_ix [parameter of PS2Opening._sample] | compute | reader: ps2_opening.py:181 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/ps2_opening.py:168 | B [parameter of PS2Opening._sample] | compute | reader: ps2_opening.py:182 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/ps2_opening.py:168 | rng [parameter of PS2Opening._sample] | compute | reader: ps2_opening.py:182 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/ps2_opening.py:168 | pair=False [parameter of PS2Opening._sample] | compute | reader: ps2_opening.py:183 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/ps2_opening.py:200 | fs [parameter of PS2Opening._residual] | compute | reader: ps2_opening.py:213 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/ps2_opening.py:200 | b [parameter of PS2Opening._residual] | compute | reader: ps2_opening.py:213 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/ps2_opening.py:200 | x [parameter of PS2Opening._residual] | compute | reader: ps2_opening.py:213 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/ps2_opening.py:219 | x [parameter of PS2Opening._added_states] | compute | reader: ps2_opening.py:221 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/ps2_opening.py:223 | fs [parameter of PS2Opening._s1_step] | compute | reader: ps2_opening.py:233 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/ps2_opening.py:251 | fs [parameter of PS2Opening._cal_unexplained] | compute | reader: ps2_opening.py:262 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/ps2_opening.py:266 | fs [parameter of PS2Opening._fit_added_state_transition] | compute | reader: ps2_opening.py:278 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/ps2_opening.py:284 | ix [parameter of PS2Opening._fit_added_state_transition.snap] | compute | reader: ps2_opening.py:286 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/ps2_opening.py:284 | n [parameter of PS2Opening._fit_added_state_transition.snap] | compute | reader: ps2_opening.py:286 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/ps2_opening.py:292 | X [parameter of PS2Opening._fit_added_state_transition.state_map_loss] | compute | reader: ps2_opening.py:295 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/ps2_opening.py:292 | U [parameter of PS2Opening._fit_added_state_transition.state_map_loss] | compute | reader: ps2_opening.py:295 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/ps2_opening.py:292 | T [parameter of PS2Opening._fit_added_state_transition.state_map_loss] | compute | reader: ps2_opening.py:295 | keep | read by the body; no change
- Clean/model_augmentation/fit_systems/ps2_opening.py:292 | t_var [parameter of PS2Opening._fit_added_state_transition.state_map_loss] | compute | reader: ps2_opening.py:295 | keep | read by the body; no change
- Clean/model_augmentation/systems/gantry_ss.py:69 | m1_ [parameter of build_poly_constants] | compute | reader: gantry_ss.py:84 | keep | read by the body; no change
- Clean/model_augmentation/systems/gantry_ss.py:69 | m2_ [parameter of build_poly_constants] | compute | reader: gantry_ss.py:84 | keep | read by the body; no change
- Clean/model_augmentation/systems/gantry_ss.py:69 | mb_ [parameter of build_poly_constants] | compute | reader: gantry_ss.py:84 | keep | read by the body; no change
- Clean/model_augmentation/systems/gantry_ss.py:69 | mh_ [parameter of build_poly_constants] | compute | reader: gantry_ss.py:84 | keep | read by the body; no change
- Clean/model_augmentation/systems/gantry_ss.py:69 | Jb_ [parameter of build_poly_constants] | compute | reader: gantry_ss.py:86 | keep | read by the body; no change
- Clean/model_augmentation/systems/gantry_ss.py:69 | Jh_ [parameter of build_poly_constants] | compute | reader: gantry_ss.py:86 | keep | read by the body; no change
- Clean/model_augmentation/systems/gantry_ss.py:69 | Lb_ [parameter of build_poly_constants] | compute | reader: gantry_ss.py:85 | keep | read by the body; no change
- Clean/model_augmentation/systems/gantry_ss.py:69 | d_ [parameter of build_poly_constants] | compute | reader: gantry_ss.py:92 | keep | read by the body; no change
- Clean/model_augmentation/systems/gantry_ss.py:109 | N0_ [parameter of build_G_matrix_entries] | compute | reader: gantry_ss.py:123 | keep | read by the body; no change
- Clean/model_augmentation/systems/gantry_ss.py:109 | d0_ [parameter of build_G_matrix_entries] | compute | reader: gantry_ss.py:129 | keep | read by the body; no change
- Clean/model_augmentation/systems/gantry_ss.py:109 | M1_ [parameter of build_G_matrix_entries] | compute | reader: gantry_ss.py:132 | keep | read by the body; no change
- Clean/model_augmentation/systems/gantry_ss.py:109 | M2_ [parameter of build_G_matrix_entries] | compute | reader: gantry_ss.py:133 | keep | read by the body; no change
- Clean/model_augmentation/systems/gantry_ss.py:109 | K_ [parameter of build_G_matrix_entries] | compute | reader: gantry_ss.py:130 | keep | read by the body; no change
- Clean/model_augmentation/systems/gantry_ss.py:109 | C_ [parameter of build_G_matrix_entries] | compute | reader: gantry_ss.py:131 | keep | read by the body; no change
- Clean/model_augmentation/utils/deepSI_corrections.py:14 | u=None [parameter of WindowedSystemData.__init__, Jan's code] | compute | reader: deepSI_corrections.py:16 | keep | read by the body; no change
- Clean/model_augmentation/utils/deepSI_corrections.py:14 | y=None [parameter of WindowedSystemData.__init__, Jan's code] | compute | reader: deepSI_corrections.py:16 | keep | read by the body; no change
- Clean/model_augmentation/utils/deepSI_corrections.py:14 | x=None [parameter of WindowedSystemData.__init__, Jan's code] | write-only | reader: none | keep | System_data's signature (SystemDataNormalizer passes x); Jan's code; Dirk decides
- Clean/model_augmentation/utils/deepSI_corrections.py:14 | cheat_n=0 [parameter of WindowedSystemData.__init__, Jan's code] | compute | reader: deepSI_corrections.py:16 | keep | read by the body; no change
- Clean/model_augmentation/utils/deepSI_corrections.py:14 | normed=False [parameter of WindowedSystemData.__init__, Jan's code] | compute | reader: deepSI_corrections.py:16 | keep | read by the body; no change
- Clean/model_augmentation/utils/deepSI_corrections.py:14 | dt=None [parameter of WindowedSystemData.__init__, Jan's code] | compute | reader: deepSI_corrections.py:16 | keep | read by the body; no change
- Clean/model_augmentation/utils/deepSI_corrections.py:18 | na=10 [parameter of WindowedSystemData.to_hist_future_data, Jan's code] | fixed-option | reader: deepSI_corrections.py:36 | keep | Jan's default 10, deepSI always passes it
- Clean/model_augmentation/utils/deepSI_corrections.py:18 | nb=10 [parameter of WindowedSystemData.to_hist_future_data, Jan's code] | fixed-option | reader: deepSI_corrections.py:36 | keep | Jan's default 10, always passed
- Clean/model_augmentation/utils/deepSI_corrections.py:18 | nf=5 [parameter of WindowedSystemData.to_hist_future_data, Jan's code] | fixed-option | reader: deepSI_corrections.py:37 | keep | Jan's default 5, always passed
- Clean/model_augmentation/utils/deepSI_corrections.py:18 | na_right=0 [parameter of WindowedSystemData.to_hist_future_data, Jan's code] | compute | reader: deepSI_corrections.py:37 | keep | read by the body; no change
- Clean/model_augmentation/utils/deepSI_corrections.py:18 | nb_right=0 [parameter of WindowedSystemData.to_hist_future_data, Jan's code] | compute | reader: deepSI_corrections.py:37 | keep | read by the body; no change
- Clean/model_augmentation/utils/deepSI_corrections.py:18 | stride=1 [parameter of WindowedSystemData.to_hist_future_data, Jan's code] | compute | reader: deepSI_corrections.py:38 | keep | read by the body; no change
- Clean/model_augmentation/utils/deepSI_corrections.py:18 | force_multi_u=False [parameter of WindowedSystemData.to_hist_future_data, Jan's code] | write-only | reader: none | keep | deepSI's method signature; Jan's code
- Clean/model_augmentation/utils/deepSI_corrections.py:18 | force_multi_y=False [parameter of WindowedSystemData.to_hist_future_data, Jan's code] | write-only | reader: none | keep | deepSI's method signature; Jan's code
- Clean/model_augmentation/utils/deepSI_corrections.py:18 | online_construct=False [parameter of WindowedSystemData.to_hist_future_data, Jan's code] | write-only | reader: none | keep | deepSI passes it (deepSI encoders.py:241); Jan's code
- Clean/model_augmentation/utils/deepSI_corrections.py:50 | u0=0 [parameter of SystemDataNormalizer.__init__, Jan's code] | compute | reader: deepSI_corrections.py:52 | keep | read by the body; no change
- Clean/model_augmentation/utils/deepSI_corrections.py:50 | ustd=1 [parameter of SystemDataNormalizer.__init__, Jan's code] | compute | reader: deepSI_corrections.py:53 | keep | read by the body; no change
- Clean/model_augmentation/utils/deepSI_corrections.py:50 | y0=0 [parameter of SystemDataNormalizer.__init__, Jan's code] | compute | reader: deepSI_corrections.py:54 | keep | read by the body; no change
- Clean/model_augmentation/utils/deepSI_corrections.py:50 | ystd=1 [parameter of SystemDataNormalizer.__init__, Jan's code] | compute | reader: deepSI_corrections.py:55 | keep | read by the body; no change
- Clean/model_augmentation/utils/deepSI_corrections.py:57 | sys_data [parameter of SystemDataNormalizer.transform, Jan's code] | compute | reader: deepSI_corrections.py:67 | keep | read by the body; no change
- Clean/model_augmentation/utils/torch_nets.py:15 | n_in [parameter of build_mlp] | compute | reader: torch_nets.py:27 | keep | read by the body; no change
- Clean/model_augmentation/utils/torch_nets.py:15 | n_out [parameter of build_mlp] | compute | reader: torch_nets.py:32 | keep | read by the body; no change
- Clean/model_augmentation/utils/torch_nets.py:15 | n_nodes_per_layer [parameter of build_mlp] | compute | reader: torch_nets.py:27 | keep | read by the body; no change
- Clean/model_augmentation/utils/torch_nets.py:15 | n_hidden_layers [parameter of build_mlp] | compute | reader: torch_nets.py:28 | keep | read by the body; no change
- Clean/model_augmentation/utils/torch_nets.py:15 | activation [parameter of build_mlp] | compute | reader: torch_nets.py:27 | keep | read by the body; no change
- Clean/model_augmentation/utils/torch_nets.py:47 | n_in [parameter of AugmentationNetwork.__init__] | compute | reader: torch_nets.py:58 | keep | read by the body; no change
- Clean/model_augmentation/utils/torch_nets.py:47 | n_out [parameter of AugmentationNetwork.__init__] | compute | reader: torch_nets.py:58 | keep | read by the body; no change
- Clean/model_augmentation/utils/torch_nets.py:47 | n_nodes_per_layer [parameter of AugmentationNetwork.__init__] | compute | reader: torch_nets.py:58 | keep | read by the body; no change
- Clean/model_augmentation/utils/torch_nets.py:47 | n_hidden_layers [parameter of AugmentationNetwork.__init__] | compute | reader: torch_nets.py:58 | keep | read by the body; no change
- Clean/model_augmentation/utils/torch_nets.py:47 | activation [parameter of AugmentationNetwork.__init__] | compute | reader: torch_nets.py:58 | keep | read by the body; no change
- Clean/model_augmentation/utils/torch_nets.py:47 | nx_phys [parameter of AugmentationNetwork.__init__] | compute | reader: torch_nets.py:62 | keep | read by the body; no change
- Clean/model_augmentation/utils/torch_nets.py:64 | X [parameter of AugmentationNetwork.forward] | compute | reader: torch_nets.py:74 | keep | read by the body; no change
- Clean/model_augmentation/utils/utils.py:8 | obj [parameter of added, Jan's code] | compute | reader: utils.py:10 | keep | read by the body; no change
- Clean/model_augmentation/utils/utils.py:13 | input [parameter of to_tensor, Jan's code] | compute | reader: utils.py:16 | keep | read by the body; no change
- Clean/model_augmentation/utils/utils.py:19 | ix [parameter of expansion_matrix, Jan's code] | compute | reader: utils.py:29 | keep | read by the body; no change
- Clean/model_augmentation/utils/utils.py:19 | n [parameter of expansion_matrix, Jan's code] | compute | reader: utils.py:29 | keep | read by the body; no change
- Clean/model_augmentation/utils/utils.py:33 | ix [parameter of selection_matrix, Jan's code] | compute | reader: utils.py:43 | keep | read by the body; no change
- Clean/model_augmentation/utils/utils.py:33 | n [parameter of selection_matrix, Jan's code] | compute | reader: utils.py:43 | keep | read by the body; no change
