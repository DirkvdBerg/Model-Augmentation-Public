# Design: clean minimal implementation of the gantry augmentation pipeline (step 2 of 4)

2026-10-19. Handoff `tasks/handoffs/2026-10-19-clean-code-design.md`. Dirk's words: "make a new self contained
folder only needing the required python files and to be written with minimal code", for the
`scripts/gantry/gantry_interconnect_dynamic.py` pipeline. This document is what step 4 implements and what the step-3
critics (one Codex session, one Claude session) check. Sections 1 to 7 follow the handoff's items 1 to 7.

Inputs: `comparison.md` (agreed step-1 results and Dirk's rulings), `inventory-claude.md` (A.1, A.2, A.3, Part C),
`inventory-codex.md` (A2, A4, A5, Part C), D-274 and its amendment (`docs/decisions.md`), the entry's THESIS block,
`Thesis-writeup/Code/Evaluation/*.py`, and targeted reads of the mainline code.

**Oracle.** Mainline = the working tree of 2026-10-19 on branch `Augmentation` (last commit 4271a31) with the
uncommitted D-274 edits. Line numbers in this document are mainline lines of that tree. Source fingerprint of the
copied files: the SHA256 values of `inventory-codex.md` C7 hold for every copied file except the three D-274 files,
which are now `GD/data.py` 866e0152...7566, `GD/training.py` a50b18fb...2a30, `MA/fit_systems/ps2_opening.py`
a1e291cb...d85 (full hashes in the golden `meta.json`, section 6.2).

**Spec.** runs.tsv rows 221 to 230 (no 228), each on `lowpass` and on `noisefree` data. Varied: arm (obc, u), start
(detuned, correct), n_a (0, 2, 8), ps2 (0, 1), noise (lowpass, noisefree). Fixed: seed 1, n_f 0.1 s, tanh MLP 16x2,
live g_aug initialisation, encoder `data_lsq`, everything in `_THESIS_FIXED` (EP:342-352).

Abbreviations: EP = `gantry_interconnect_dynamic.py`, GD = `gantry_dynamic/`, MA = `model_augmentation/`,
CM = `scripts/gantry/common/`. H1 to H11 are the equivalence hazards of section 6.1.

## 0. Implementation rules

These rules decide every case the mapping (section 2) does not spell out.

1. **Copy first.** A kept definition is copied from mainline statement by statement. "copy" = no statement changes
   (comments may be shortened); "trim" = only the listed branches and statements are removed; "merge" = the
   statements move into the named function unchanged; "rewrite" = new code with the contract stated in the row.
   Nothing else changes: no renamed locals, no reordered arithmetic, no "cleaner" equivalent formula (H4 to H6).
2. **Removed branches.** A branch is removed when the spec selects the other side through a constant (an option
   value, a fixed dimension, a feature flag that is always off). Branches whose side is decided by data or by the
   device (GPU moves, the compiled rollout, OBC rank loss, polish failure, NaN break, PS2 fallback) are kept even
   when the CPU trace never took them.
3. **Guards (rule G).** A guard is kept when it checks something the code does not construct itself (files on disk,
   solver results, training numerics) or when the evaluation relies on it. A guard that only re-checks values the
   clean code builds from its own constants (validation of removed options, argument shapes between two internal
   calls) is removed. Every removed guard is named in the mapping.
4. **Markers (Dirk).** The clean `model_augmentation/` keeps the CLAUDE.md markers exactly where mainline has them:
   `__project_origin__ = "added"` at the top of every file that has it, `@added` on every kept class or function
   that has it, and every `# CHANGED: <reason>` comment that sits on a kept statement (a comment whose block is
   removed goes with it). Deletions carry no marker; this mapping is their record. `gantry_dynamic/` files keep
   their `__project_origin__` line.
5. **Comments.** Kept: every `# THEORY:` and `# HEURISTIC:` label on a numerical formula, constant or threshold
   (CLAUDE.md), and comments that state a non-obvious reason on a kept statement. Dropped: history narratives
   (job numbers, measured incidents of removed options), long docstrings of removed alternatives.
6. **No run-time patching.** Every behaviour is in the copied code; nothing wraps or patches a class at run time
   (rejected by Dirk). The harness (section 6) imports mainline and the clean code side by side in separate
   processes.
7. **Prints.** Kept: the run line, the data lines, the bank line ("one bank over N records, k distinct
   controllers", a sbatch sanity grep), "training rollout COMPILED", the encoder-start line, one progress line per
   validation, the PS2 summary, the polish outcome lines. Everything printed only by a dropped diagnostic goes.

## 1. Folder layout

**Location and name: `Thesis-writeup/Code/Training/`** (open choice 1). It sits beside the thesis code that already
exists there (`Data/` generates the records, `Evaluation/` scores the checkpoints, `Figures/`, `Server/`), so the
thesis code reads data generation, training, evaluation, figures in one tree, and the data root
`Thesis-writeup/Data` is a fixed relative path (`Training/../../Data`) on the PC and on the server clone alike.

```
Thesis-writeup/Code/Training/
  gantry_interconnect_dynamic.py   entry: CLI (preset, noise, optional cpu, smoke) and main(cfg)
  gantry_dynamic/
    __init__.py                    __project_origin__ only (no sys.path insertion: Training/ is the import root)
    config.py                      RunConfig (5 axes + run environment), PRESETS, constants, config.json, save_dir
    data.py                        DATA_ROOT, DATASETS (lowpass, noisefree), MANIFEST split, record loading, normalisation
    controller.py                  K1 controller bank, closed-loop simulator construction, torch.compile
    model.py                       build_model, least-squares encoder start
    obc_gantry.py                  OBC reference set, correction module, basis builder, attach_obc
    training.py                    train (Adam, PS2, selection, polish, checkpoints), ValHistory, checkpoint loading
    closed_loop_report.py          _without_learned, _true_free only (evaluation interface, open choice 10)
  model_augmentation/
    __init__.py                    empty (mainline eagerly imports benchmark modules; nothing needs that)
    fit_systems/
      __init__.py                  empty
      interconnect.py              Interconnect, SSE_Interconnect (Jan, trimmed), SSE_Interconnect_Composed, Dtype_DataLoader
      blocks.py                    Block, Static_ANN_Block, Linear_Output_Block, Gantry/Trainable/Reduced state blocks
      closed_loop.py               ControllerBank, closed-loop rollout, ClosedLoopSimulator, batch free-run scorer
      lbfgs_polish.py              PolishSpec, FixedBatch, lbfgs_polish
      obc.py                       OBCBasis, OBCLifecycle, build_stacked_sensitivity
      pre_encoder.py               linear_encoder_init_aug
      ps2_opening.py               PS2Spec, PS2Opening
    systems/
      __init__.py                  empty
      gantry_ss.py                 nominal float32 constants, P, Cd, Dd, LFR builders
    utils/
      __init__.py                  empty
      utils.py                     added, to_tensor, detect_algebraic_loop, expansion_matrix, selection_matrix
      torch_nets.py                live_g_feed_forward_nn
      deepSI_corrections.py        fixed_System_data, fixed_System_data_norm
```

Twenty-four files, every one of them a copy-and-trim of the mainline file with the same package path (1:1, open
choice 3), so a reviewer diffs each against its mainline original. Size: the four mainline locations hold 15,750 lines in 42
files; the kept definitions span 5,945 of them, about 4,900 of which are code (blank, comment and docstring lines
excluded, rough tokenizer count). The branch removals of section 2 come on top, so the folder should land between
3,500 and 4,500 lines (estimate from these counts; the implementer reports the measured number).

**Outside the folder, by design:**
- Data: `DATA_ROOT = <repo>/Thesis-writeup/Data` (section 4). Read only.
- Run output: `OUT_ROOT = <repo>/simulations/gantry_clean/`, run folder `OUT_ROOT/{mode}_data_lsq/<run_id>/`, with
  `mode` = `thesis_taf_lowpass_noise` or `thesis_taf_noisefree` (the mainline mode strings, so a family folder has the
  name the evaluation already expects) and `run_id` = `SLURM_JOB_ID`, else a timestamp. A separate root keeps clean
  runs out of the mainline family folders, where `discover_runs` would raise on two runs of one design (open
  choice 15).
- deepSI's own checkpoints `<Class>_<run_id>_best.pth` and `_last.pth` stay where deepSI writes them
  (`~/.deepSI/checkpoints`, or `%LOCALAPPDATA%/deepSI/checkpoints` on Windows); fit() needs them (H8).
- The installed `deepSI` package, pinned at 0.3.29, with torch 2.5.1, numpy 2.0.1, scipy 1.17.0 (open choice 2).
- The harness (section 6) in `Thesis-writeup/Code/Verification/clean-code/` and its golden outputs in
  `<repo>/simulations/clean-golden/`.

**Imports.** Running the entry puts `Training/` on `sys.path[0]`, so `gantry_dynamic` and `model_augmentation`
resolve inside the folder with the same package and module names as mainline. Consequence: one process must never
have both the mainline root and `Training/` on its path (same package names); section 5 states what this means for
the evaluation.

## 2. File-by-file mapping

Every definition of the four code locations (647, the full syntax ledger of `inventory-codex.md` A4, nested
functions included) and every module or class constant (154) appears below exactly once. 262 definitions are kept,
merged or rewritten; 385 are dropped. The table was generated from a decision list and checked by script: every
definition labelled required or interface in either inventory (Codex "required", "required interface"; Claude
"run (pipeline)") has an explicit row or an explicit drop reason, and no row names a definition that does not exist.
Unlisted dropped definitions carry the reason of their inventory label ("diagnostic (Dirk excluded)" or "not on the
spec path").

Columns: source definition (mainline line) | destination in the new folder (bare name = same file as the heading
arrow) | action (section 0, rule 1) | removed branches with mainline lines, kept items that a trim could be expected
to remove, interface notes.

**D-274 at its destinations:** change 1 (S2 keeps the start as a candidate) `MA/fit_systems/ps2_opening.py
PS2Opening._fit_map` lines 233-235, plus the agreed comment on `T.var(0)` in `_fit_map.ee`; change 2 (K1 records
only) `GD/data.py record_files` lines 89-90; change 3 as amended (the polish must beat `bestfit`; the rebuilt model is
kept on rejection) `GD/training.py run_lbfgs_polish` lines 784-796 and the final-event rule in `GD/training.py train`.

**New definitions (no mainline source):** `GD/config.py` `PRESETS`, `ROOT`, `OUT_ROOT`, the RunConfig axis
properties (`obc`, `nx_ann`, `ann_route_ix`, `mode`, `combo_init_detune`) and ClassVar constants (section 3);
`GD/data.py` `DATA_ROOT`, `DATASETS`; `GD/training.py` `train`, `ValHistory`; EP `cli()`.

#### `EP`

| Source (line) | Destination | Action | Removed branches, reason, interface |
|-|-|-|-|
| `main` (382) | `EP main(cfg, run_id=None)` | rewrite | same call order as EP:383-453: run dir; seed (np, torch); load_datasets; compute_normalization; config.json; seed again; build_model; build_closed_loop; prefit_physical_encoder; train. Removed: print_run_banner (397), RESUME_CHECKPOINT and resolve_resume_hp (399-400), save_flag tests (always save), open-loop branch (446-447), post-run evaluate/diagnostics (456-471) |

- Dropped (THESIS_* env parsing replaced by the CLI (section 3)): `_thesis_env`.
- Module and class constants: `CFG`: replaced -> GD/config.py PRESETS + RunConfig; `_dataset_ab_mode`, `_dataset_ab_seed`, `_lbfgs_ab_freeze`, `_obc_arm`, `_thesis_arm`: drop; `_LBFGS_TOL_GRAD`, `_LBFGS_TOL_CHANGE`: move -> GD/config.py RunConfig.lbfgs_tol_grad, lbfgs_tol_change (ClassVar).

#### `GD/closed_loop_report.py` -> `GD/closed_loop_report.py`

| Source (line) | Destination | Action | Removed branches, reason, interface |
|-|-|-|-|
| `_without_learned` (38) | `_without_learned` | copy | evaluation interface; module imports reduced to Static_ANN_Block, copy, torch |
| `_true_free` (67) | `_true_free` | copy | evaluation interface |

- Dropped (post-run report): `_score`, `closed_loop_table`.
- Module and class constants: `__project_origin__`: copy (marker); `ROWS`: drop.

#### `GD/config.py` -> `GD/config.py`

| Source (line) | Destination | Action | Removed branches, reason, interface |
|-|-|-|-|
| `y_decimation` (46) | `y_decimation` | copy | anti-alias rule for noise=lowpass, point sampling for noisefree |
| `record_trim` (53) | `record_trim` | trim | always round(AA_TRIM_S * 4000) = 20 (both modes trim); evaluation imports it |
| `RunConfig` (60) | `RunConfig` | rewrite | five axis fields + run-environment fields; fixed settings as ClassVar constants under the mainline names (section 3) |
| `RunConfig.__post_init__` (370) | `RunConfig.__post_init__` | rewrite | validates the axis and environment values only (allowed sets); the 30 checks of removed options go |
| `RunConfig.fs_new_hz` (542) | `RunConfig.fs_new_hz` | copy |  |
| `RunConfig.d` (546) | `RunConfig.d` | copy |  |
| `RunConfig.ts_new` (550) | `RunConfig.ts_new` | copy |  |
| `RunConfig.dtype_np` (554) | `RunConfig.dtype_np` | copy |  |
| `RunConfig.dtype_pt` (558) | `RunConfig.dtype_pt` | copy |  |
| `RunConfig.nf` (562) | `RunConfig.nf` | trim | nf_override branch (571-572) removed; round(NF_SECONDS / ts_new) = 400 |
| `RunConfig.na_nb` (576) | `RunConfig.na_nb` | trim | returns NA_NB = 30; the dimension rule and override go |
| `RunConfig.hp` (583) | `RunConfig.hp` | copy | hp keeps key names and order (NX_ANN, n_nodes_per_layer, n_hidden_layers, up_sample, nf, na_nb, batch_size, lr, epochs): the .npz hp contract |
| `save_dir` (603) | `save_dir` | trim | OUT_ROOT/{mode}_data_lsq (section 1) |
| `git_provenance` (610) | `git_provenance` | copy | provenance only; cwd = the repository holding the folder |
| `git_provenance._git` (628) | `git_provenance._git` | copy |  |
| `config_json_dict` (668) | `config_json_dict` | rewrite | mainline keys in mainline order with the spec values, then ARM, START, N_A, NOISE, PS2_SMOKE, then GIT (section 5) |

- Dropped (never called): `default_hp`.
- Dropped (config_json_dict writes every field explicitly; the import-time guard served evolving fields): `_assert_json_coverage`.
- Module and class constants: `__project_origin__`: copy (marker); `_PKG_DIR`, `REPO_ROOT`: replaced -> GD/config.py ROOT (the folder) and REPO_ROOT = ROOT/../../.. (git cwd, OUT_ROOT); `ANTI_ALIAS_MODES`: replaced -> GD/config.py: anti-alias rule iff noise == "lowpass"; `AA_ORDER`, `AA_FC_HZ`, `AA_TRIM_S`: copy (HEURISTIC/THEORY labels kept) -> GD/config.py; `TRIM_MODES`: drop (both noise entries trim); `_JSON_EXEMPT`: drop.

#### `GD/controller.py` -> `GD/controller.py`

| Source (line) | Destination | Action | Removed branches, reason, interface |
|-|-|-|-|
| `M_op` (72) | `M_op` | copy |  |
| `sys_stage_frf` (81) | `sys_stage_frf` | copy |  |
| `cnorm_coeffs` (92) | `cnorm_coeffs` | copy |  |
| `cnorm_at` (99) | `cnorm_at` | copy |  |
| `build_cfb_at` (104) | `build_cfb_at` | copy |  |
| `_tf_to_ss_batch` (124) | `_tf_to_ss_batch` | copy |  |
| `controller_ss` (143) | `controller_ss` | copy |  |
| `_stem` (181) | `_stem` | copy |  |
| `register_stored_controller` (198) | `register_stored_controller` | copy | the unsupported-controller branch (205-207) stays: the evaluation skips a non-K1 record through y_op_for raising KeyError |
| `y_op_for` (210) | `y_op_for` | trim | legacy RECORD_Y_OP branch (218-219) removed; stored and unsupported branches stay (evaluation interface) |
| `gain_for` (232) | `gain_for` | copy |  |
| `build_controller_bank` (246) | `build_controller_bank` | copy | evaluation imports it |
| `build_closed_loop` (269) | `build_closed_loop` | trim | removed: val-count guard (304-305), loss_weighting test and e_train-is-None guard (350-352, always servo weights), physical_D print (316-317); kept: duplicate-name guard (292-293), compile block (329-341, GPU), keyword-only signature incl. verbose, e_train, e_val (evaluation calls it) |

- Module and class constants: `__project_origin__`: copy (marker); `mb`, `mh`, `m1`, `m2`, `Jb`, `Jh`, `cg1`, `cg2`, `cy`, `cb1`, `cb2`, `kb1`, `kb2`, `Lb`, `d`, `FBW`, `W`, `TS`, `P`, `C_DAMP`, `K_STIFF`: copy (Python doubles, independent of gantry_ss float32) -> GD/controller.py; `RECORD_Y_OP`: drop (legacy modes); `_STORED_DESIGN_Y`, `_STORED_GAIN`, `_UNSUPPORTED`: copy (registry filled by _register_controller) -> GD/controller.py.

#### `GD/data.py` -> `GD/data.py`

| Source (line) | Destination | Action | Removed branches, reason, interface |
|-|-|-|-|
| `record_files` (79) | `record_files` | trim | legacy branch (81-82) removed; D-274 K1 filter (89-90) copied; manifest order; 18/6/11 |
| `_register_controller` (94) | `_register_controller` | copy |  |
| `DataBundle` (104) | `DataBundle` | trim | fields kept: train_list, val_list, test_list, train_data, val_ckpt_data, train_files, val_files, test_files, e_train, e_val, e_test; dropped: val_data, test_data, val_x_logical, val_x_aug, test_x_logical, test_x_aug, sigma_n |
| `Norm` (129) | `Norm` | trim | fields kept: x_mean, std_x, std_u, u_mean, ystd, y0, Cd_norm, Dd_np; dropped: u_all, y_all, x_all (linear_map only) |
| `traj_dir` (143) | `traj_dir` | trim | DATA_ROOT / DATASETS[cfg.noise]; legacy branch (146) removed; evaluation imports it |
| `_load_u` (149) | `_load_u` | copy | evaluation imports it |
| `_resample_u` (154) | `_resample_u` | trim | D == 1 branch (168-169) removed; float64 block mean kept |
| `_decimate_y` (174) | `_decimate_y` | trim | 20 -> 4 kHz guard (179-181) removed (constants); both rules kept |
| `_record_at_fs_new` (187) | `_record_at_fs_new` | copy | truncate_s branch kept for the smoke and the harness |
| `_load_record` (205) | `_load_record` | trim | is_thesis test and has_r branch removed (every thesis record holds r_sim); folder= kept (evaluation twins) |
| `servo_sigma` (224) | `servo_sigma` | trim | None guard (228-229) removed; finite-positive guard kept (data) |
| `load_datasets` (260) | `load_datasets` | trim | removed: SNR guard and noise branch (272-293), val_data and test_data (299-300), primary val/test truth loads (302-313); kept prints: data dir, decimation rule, split counts (315-317) |
| `load_datasets._split` (278) | `load_datasets._split` | copy |  |
| `compute_normalization` (331) | `compute_normalization` | trim | u_all/y_all/x_all no longer returned; arithmetic unchanged incl. inv(P.numpy().T) in float32 then cast (hazard H6) |

- Dropped (every mode is a thesis mode): `is_thesis`.
- Dropped (not used (comparison.md)): `load_traj`.
- Dropped (diagnostic ground truth (comparison.md)): `load_mat_aug`.
- Module and class constants: `__project_origin__`: copy (marker); `TRAIN_FILES`, `VAL_FILES`, `TEST_FILES`: drop (legacy lists); `THESIS_DATASETS`: replaced -> GD/data.py DATASETS = {lowpass, noisefree} below DATA_ROOT; `_THESIS_SPLIT`: copy -> GD/data.py _THESIS_SPLIT.

#### `GD/diagnostics.py`

- Dropped (diagnostic (Dirk excluded)): `r2_per_channel`, `best_affine_r2`, `encoder_state_estimates`, `state_recovery_diagnostic`, `compute_gradient_norms`, `encoder_init_state`, `zeroed_ann_validation`.
- Dropped (printing only (both inventories)): `print_run_banner`.
- Module and class constants: `__project_origin__`: drop (file dropped).

#### `GD/model.py` -> `GD/model.py`

| Source (line) | Destination | Action | Removed branches, reason, interface |
|-|-|-|-|
| `get_encoder_dims` (39) | (merged) | merge | into GD/model.py build_model: na = nb = cfg.na_nb, right extensions 0 (the data_lsq branch, 47-51) |
| `build_model` (60) | `build_model` | trim | removed: raw/non-joint branches (104-109, 124-129), static_pass_buffers (113-115), zero/resnet g (139, 141-142), ANN_REZERO_GATE (157-164), linear_map/default encoder (210-245), else-cast (302-304), orth (312-314); kept in order (hazard H1): reduced block (detuned combos), Linear_Output_Block, Static_ANN_Block with live_g_feed_forward_nn, connections, SSE_Interconnect_Composed, norm fields, data_lsq encoder, cuda-availability guard, init_model, eps guard, hfn.to(float64), attach_obc for arm=obc. PARAM_NAMES read from Reduced_Gantry_State_Block (Parameterized_Gantry_State_Block is dropped) |
| `prefit_physical_encoder` (328) | `prefit_physical_encoder` | copy | cuda moves kept |
| `_prefit` (352) | `_prefit` | trim | simulator-is-None guard (355-356) removed |
| `_prefit.recorder` (389) | `_prefit.recorder` | copy |  |
| `least_squares_start` (415) | `least_squares_start` | copy | every guard kept (data-dependent readiness, D-271) |
| `train_model` (467) | (merged) | merge | into GD/training.py train: eps guard, cuda move, fit(...) with the same arguments, cpu move; its_per_val = n_its if n_its else its_per_val (501) |

- Module and class constants: `__project_origin__`: copy (marker); `COMPILE_BACKEND`: move -> GD/controller.py COMPILE_BACKEND (where torch.compile is called); `PHY_IX`, `_START_MAX_UNEXPLAINED`: copy -> GD/model.py.

#### `GD/obc_gantry.py` -> `GD/obc_gantry.py`

| Source (line) | Destination | Action | Removed branches, reason, interface |
|-|-|-|-|
| `GantryReferenceSet` (49) | `GantryReferenceSet` | trim | fields kept: Z_step, Z_field, record_ix, sample_ix, record_names, x_phys_next_rec; dropped: x_phys, u_stage, x_phys_next, x_abs, n_per_record, ts (diagnostic or unused) |
| `GantryReferenceSet.n` (68) | `GantryReferenceSet.n` | copy |  |
| `fourth_order_central_difference` (79) | `fourth_order_central_difference` | copy |  |
| `_load_record_float64` (91) | `_load_record_float64` | trim | returns (u, y) only: the truth fields x_logical, delta_a, vdelta_a are not read |
| `build_reference_set` (106) | `build_reference_set` | trim | removed: snr guard (114-115), stride option (136-137), files default TRAIN_FILES (files required), truth arrays, the Y-coverage print; signature (cfg, norm, nx_ann, files, verbose) kept for the evaluation |
| `GantryOBCCorrection` (175) | `GantryOBCCorrection` | trim | affine space only |
| `GantryOBCCorrection.__init__` (189) | `GantryOBCCorrection.__init__` | trim | space guard (193-194), empty-route guard (209-210), _static and use_jvp attributes removed; buffers E, vbar, theta_frozen unchanged (state_dict names) |
| `GantryOBCCorrection.expansion_point` (225) | `GantryOBCCorrection.expansion_point` | copy |  |
| `GantryOBCCorrection.set_frozen` (230) | `GantryOBCCorrection.set_frozen` | copy |  |
| `GantryOBCCorrection.tangent_pair` (250) | `GantryOBCCorrection.tangent_pair` | trim | _static branches (262-263, 267-268) and tangent-space branch (264-265) removed |
| `GantryOBCCorrection.forward` (286) | `GantryOBCCorrection.forward` | trim | use_jvp branch (297-298) and tangent branch (301-302) removed |
| `make_float64_block` (312) | `make_float64_block` | copy | evaluation imports it |
| `make_basis_builder` (319) | `make_basis_builder` | trim | space argument fixed to affine; signature (blk64, ref, row_ix, chunk, device=) kept for the evaluation |
| `make_basis_builder.build` (323) | `make_basis_builder.build` | trim | tangent-only branch (332->339) removed; device move (327-328) kept |
| `make_reference_field` (344) | `make_reference_field` | copy |  |
| `make_reference_field.field` (350) | `make_reference_field.field` | copy |  |
| `attach_obc` (362) | `attach_obc` | trim | removed: ref/ref_stride arguments, static_pass_buffers (385-389), TRAIN_FILES fallback, name_columns; kept: block-type guard (372-373), expected_rank 11, float64 block, lifecycle, fit_sys.obc_reference |

- Dropped (print only): `GantryReferenceSet.memory_bytes`.
- Dropped (jvp reference path): `GantryOBCCorrection.step`, `GantryOBCCorrection.correction_from_theta`.
- Module and class constants: `__project_origin__`: copy (marker); `COMBO_NAMES`: drop (name_columns print only).

#### `GD/training.py` -> `GD/training.py`

| Source (line) | Destination | Action | Removed branches, reason, interface |
|-|-|-|-|
| `_json_safe` (26) | `_json_safe` | copy |  |
| `save_checkpoint_weights` (39) | `save_checkpoint_weights` | copy | .pt keys hfn, encoder, optimizer (omitted after an accepted polish) |
| `resolve_checkpoint` (67) | `resolve_checkpoint` | copy | both formats (pair and deepSI _best.pth): the evaluation loads either |
| `_assert_same_architecture` (158) | `_assert_same_architecture` | copy | evaluation interface |
| `_assert_same_dtype` (248) | `_assert_same_dtype` | copy | evaluation interface |
| `load_checkpoint` (277) | `load_checkpoint` | trim | Adam-moment restore (295-297) and the fresh-optimizer note (305-315) removed; signature (fit_sys, base_path, joint_estimation, start_phase) kept for the evaluation call |
| `_NfProbe` (319) | `ValHistory` | rewrite | the D-267 history only (open choice 9): val_hist, hist_path, record, hist_arrays, __call__ |
| `_NfProbe.__init__` (362) | `ValHistory.__init__` | rewrite | stores fit_sys and hist_path; no window batch, no WindowErrorStats, no combo meters (no global RNG either way, hazard H2) |
| `_NfProbe.record` (593) | `ValHistory.record` | copy | open-loop early return (600-601) removed |
| `_NfProbe.hist_arrays` (610) | `ValHistory.hist_arrays` | copy | val_hist.npz keys unchanged |
| `_NfProbe.__call__` (627) | `ValHistory.__call__` | trim | only record("adam", fit_sys, value); train/val nf statistics and prints removed |
| `_install_nf_val_probe` (671) | (merged) | merge | into GD/training.py train (append the probe, keep a reference, restore the tuple in finally) |
| `run_lbfgs_polish` (753) | `run_lbfgs_polish` | trim | removed: lbfgs-off return (764-765), meters_fn, start_phase test; kept: PolishSpec from the config, cuda moves and empty_cache (776-782, 820-822), D-274 rule (784-796): val_before = bestfit, OBC rebuilt score measured and recorded as "rebuilt", used only if bestfit is not finite; polish_candidate record |
| `train_model_with_diagnostics` (825) | `train` | rewrite | same sequence as 825-1017 without resume and start_phase=lbfgs: unique_code = run_id; bestfit = inf; PS2 attach (final or smoke spec, 871-879); history install; Adam (merged train_model); finally restore probes and training_phase; pre-polish pair (932-938); selected; polish; final = accepted polish, else rebuilt (OBC), else selected (D-274); bestfit = final score; .pt and .npz with all 12 keys (resumed_from "", resumed_at_epoch -1, start_phase "adam") |

- Dropped (resume only): `load_optimizer_moments`.
- Dropped (resume only; not used by the evaluation): `infer_arch_from_hfn`, `config_for_checkpoint`.
- Dropped (nf/joint/OBC probe printing (Dirk excluded)): `_NfProbe._val_stats`, `_NfProbe._fmt`, `_NfProbe._joint_probe`, `_NfProbe._obc_probe`.
- Dropped (polish meters, no gate (meters_gate=False)): `_gantry_meters`.
- Dropped (no resume; hp = cfg.hp): `resolve_resume_hp`.
- Module and class constants: `__project_origin__`: copy (marker); `_NfProbe.N_VAL_WINDOWS`: drop.

#### `MA/fit_systems/blocks.py` -> `MA/fit_systems/blocks.py`

| Source (line) | Destination | Action | Removed branches, reason, interface |
|-|-|-|-|
| `Block` (12) | `Block` | trim | Jan |
| `Block.__init__` (15) | `Block.__init__` | copy |  |
| `Block.init_block` (24) | `Block.init_block` | copy |  |
| `Static_ANN_Block` (31) | `Static_ANN_Block` | copy | Jan; evaluation imports it (type identity, out_gate) |
| `Static_ANN_Block.__init__` (32) | `Static_ANN_Block.__init__` | copy | net constructed here: first global draws (hazard H1) |
| `Static_ANN_Block.forward` (69) | `Static_ANN_Block.forward` | trim | out_gate re-home (76-77) removed (non-persistent buffer follows .to()) |
| `Linear_Output_Block` (120) | `Linear_Output_Block` | copy | @added |
| `Linear_Output_Block.__init__` (121) | `Linear_Output_Block.__init__` | trim | ny == 0 branch (130-131) removed; to_tensor float32 materialisation kept (hazard H6) |
| `Linear_Output_Block.forward` (138) | `Linear_Output_Block.forward` | copy |  |
| `Discrete_Nonlinear_Function_Block` (251) | `Discrete_Nonlinear_Function_Block` | trim | Jan |
| `Discrete_Nonlinear_Function_Block.__init__` (252) | `Discrete_Nonlinear_Function_Block.__init__` | copy |  |
| `Discrete_Nonlinear_Function_Block.forward` (260) | `Discrete_Nonlinear_Function_Block.forward` | copy |  |
| `Gantry_State_Block` (664) | `Gantry_State_Block` | trim | @added |
| `Gantry_State_Block.__init__` (699) | `Gantry_State_Block.__init__` | trim | Y_op-at-construction branch (757-763) removed (Y_op = None; the evaluation sets Y_op, N_op, d_op afterwards); nominal float32 buffers unchanged (hazard H6) |
| `Gantry_State_Block.nonlinear_function` (779) | `Gantry_State_Block.nonlinear_function` | copy |  |
| `Gantry_State_Block._rk4` (791) | `Gantry_State_Block._rk4` | copy |  |
| `Gantry_State_Block._deriv_with` (814) | `Gantry_State_Block._deriv_with` | trim | terms capture (871-874) removed; frozen-Y branch (843-845) kept: evaluation frozen_y |
| `Gantry_State_Block._deriv_both_with` (902) | `Gantry_State_Block._deriv_both_with` | trim | terms/sterms capture (968-972) removed; frozen-Y guard (909-910, raises NotImplementedError) kept |
| `Gantry_State_Block._rk4_with_tangent` (979) | `Gantry_State_Block._rk4_with_tangent` | copy |  |
| `_Trainable_Gantry_State_Block` (996) | `_Trainable_Gantry_State_Block` | trim | @added; static_pass class attribute removed; PARAM_NAMES kept |
| `_Trainable_Gantry_State_Block.__init__` (1008) | `_Trainable_Gantry_State_Block.__init__` | trim | params_init None branch (1024-1025) removed; positivity guard (1030-1031) kept; buffers unchanged (state_dict names) |
| `_Trainable_Gantry_State_Block._build_KC` (1042) | `_Trainable_Gantry_State_Block._build_KC` | copy |  |
| `_Trainable_Gantry_State_Block._build_M1M2` (1064) | `_Trainable_Gantry_State_Block._build_M1M2` | copy |  |
| `_Trainable_Gantry_State_Block._mats_from_raw` (1079) | `_Trainable_Gantry_State_Block._mats_from_raw` | copy |  |
| `_Trainable_Gantry_State_Block.nonlinear_function` (1102) | `_Trainable_Gantry_State_Block.nonlinear_function` | copy |  |
| `_Trainable_Gantry_State_Block.pass_mats` (1112) | `_Trainable_Gantry_State_Block.pass_mats` | trim | static_pass branch removed |
| `_Trainable_Gantry_State_Block._mats` (1122) | `_Trainable_Gantry_State_Block._mats` | copy |  |
| `_Trainable_Gantry_State_Block.__getstate__` (1147) | `_Trainable_Gantry_State_Block.__getstate__` | copy |  |
| `_Trainable_Gantry_State_Block.__setstate__` (1152) | `_Trainable_Gantry_State_Block.__setstate__` | copy |  |
| `Reduced_Gantry_State_Block` (1250) | `Reduced_Gantry_State_Block` | trim | @added; COMBO_NAMES and index constants kept |
| `Reduced_Gantry_State_Block.__init__` (1304) | `Reduced_Gantry_State_Block.__init__` | trim | combo_init None branch (1308-1316) removed; value guards kept; Lambda_combo and RMSE buffers kept (state_dict names) |
| `Reduced_Gantry_State_Block.combos_of` (1350) | `Reduced_Gantry_State_Block.combos_of` | copy | evaluation interface |
| `Reduced_Gantry_State_Block.gauge_section` (1365) | `Reduced_Gantry_State_Block.gauge_section` | copy |  |
| `Reduced_Gantry_State_Block._m_diff_bound` (1385) | `Reduced_Gantry_State_Block._m_diff_bound` | copy |  |
| `Reduced_Gantry_State_Block.combinations_from_free` (1397) | `Reduced_Gantry_State_Block.combinations_from_free` | copy | evaluation interface |
| `Reduced_Gantry_State_Block.free_from_combinations` (1409) | `Reduced_Gantry_State_Block.free_from_combinations` | copy | evaluation interface (_true_free) |
| `Reduced_Gantry_State_Block.recover_combinations` (1424) | `Reduced_Gantry_State_Block.recover_combinations` | copy |  |
| `Reduced_Gantry_State_Block.transition_from_free` (1429) | `Reduced_Gantry_State_Block.transition_from_free` | copy | evaluation interface |
| `Reduced_Gantry_State_Block.mats_and_tangent_from_free` (1461) | `Reduced_Gantry_State_Block.mats_and_tangent_from_free` | copy |  |
| `Reduced_Gantry_State_Block.mats_and_tangent_from_free._mats` (1463) | `Reduced_Gantry_State_Block.mats_and_tangent_from_free._mats` | copy |  |
| `Reduced_Gantry_State_Block._recover_params` (1482) | `Reduced_Gantry_State_Block._recover_params` | copy |  |
| `Reduced_Gantry_State_Block.admissibility` (1494) | `Reduced_Gantry_State_Block.admissibility` | copy |  |
| `Reduced_Gantry_State_Block.estimation_parameters` (1551) | `Reduced_Gantry_State_Block.estimation_parameters` | copy | evaluation interface |

- Dropped (not on the spec path): `Block.forward`, `Cubic_Block`, `Cubic_Block.__init__`, `Cubic_Block.forward`, `Linear_State_Block`, `Linear_State_Block.__init__`, `Linear_State_Block.forward`, `Parameterized_Linear_State_Block`, `Parameterized_Linear_State_Block.__init__`, `Parameterized_Linear_State_Block.forward`, `Parameterized_Linear_State_Block.param_loss`, `Parameterized_Linear_Output_Block`, `Parameterized_Linear_Output_Block.__init__`, `Parameterized_Linear_Output_Block.forward`, `Parameterized_Linear_Output_Block.param_loss`, `Cascaded_Tanks_State_Block`, `Cascaded_Tanks_State_Block.__init__`, `Cascaded_Tanks_State_Block.nonlinear_function`, `Parameterized_Cascaded_Tanks_State_Block`, `Parameterized_Cascaded_Tanks_State_Block.__init__`, `Parameterized_Cascaded_Tanks_State_Block.nonlinear_function`, `Parameterized_MSD_State_Block`, `Parameterized_MSD_State_Block.__init__`, `Parameterized_MSD_State_Block.nonlinear_function`, `Nonlinear_MSD_State_Block`, `Nonlinear_MSD_State_Block.__init__`, `Nonlinear_MSD_State_Block.nonlinear_function`, `Nonlinear_MSD_State_Block.deriv`.
- Dropped (not used): `Discrete_Nonlinear_Function_Block.init_norm`, `Gantry_State_Block.deriv`, `Gantry_State_Block._deriv_tangent_with`, `Reduced_Gantry_State_Block.transition_and_tangent_from_free`.
- Dropped (abstract; overridden): `Discrete_Nonlinear_Function_Block.nonlinear_function`.
- Dropped (overridden by _Trainable_Gantry_State_Block._mats): `Gantry_State_Block._mats`.
- Dropped (reporting and meters; the evaluation uses estimation_parameters only (comparison.md: _combos_from_raw)): `_Trainable_Gantry_State_Block.physical_params`, `_Trainable_Gantry_State_Block._combos_from_raw`, `_Trainable_Gantry_State_Block.identifiable_combinations`, `_Trainable_Gantry_State_Block._initial_identifiable_combinations`, `_Trainable_Gantry_State_Block._combination_training_label`, `_Trainable_Gantry_State_Block.param_table`.
- Dropped (raw coordinates; never instantiated (PARAM_NAMES read from the reduced block)): `Parameterized_Gantry_State_Block`.
- Dropped (raw coordinates): `Parameterized_Gantry_State_Block.__init__`, `Parameterized_Gantry_State_Block._recover_params`, `Parameterized_Gantry_State_Block.param_loss`, `Parameterized_Gantry_State_Block.estimation_parameters`.
- Dropped (returns the float 0.0 (param_prior=False); the loss sweep that called it is removed): `Reduced_Gantry_State_Block.param_loss`.
- Dropped (meters and the excluded run report): `Reduced_Gantry_State_Block.identifiable_combinations`, `Reduced_Gantry_State_Block._initial_identifiable_combinations`, `Reduced_Gantry_State_Block._combination_training_label`, `Reduced_Gantry_State_Block.physical_params`.
- Module and class constants: `_Trainable_Gantry_State_Block.PARAM_NAMES`, `Reduced_Gantry_State_Block.COMBO_NAMES`, `Reduced_Gantry_State_Block.M_TOTAL_IX`, `Reduced_Gantry_State_Block.M_DIFF_IX`, `Reduced_Gantry_State_Block.J_EFF_IX`, `Reduced_Gantry_State_Block.M_DIFF_RHO`, `Reduced_Gantry_State_Block.COORDINATE_VERSION`: copy -> MA/fit_systems/blocks.py; `_Trainable_Gantry_State_Block._IDENT_ORDER`: drop (param_table).

#### `MA/fit_systems/closed_loop.py` -> `MA/fit_systems/closed_loop.py`

| Source (line) | Destination | Action | Removed branches, reason, interface |
|-|-|-|-|
| `ControllerBank` (76) | `ControllerBank` | trim |  |
| `ControllerBank.__init__` (122) | `ControllerBank.__init__` | copy | shape guards (132-136) kept (inputs from the stored controller files) |
| `ControllerBank.n_controllers` (150) | `ControllerBank.n_controllers` | copy | the bank print in build_closed_loop (sbatch sanity line) |
| `ControllerBank.gather` (158) | `ControllerBank.gather` | copy |  |
| `ControllerBank.zero_state` (162) | `ControllerBank.zero_state` | copy |  |
| `ControllerBank.step` (167) | `ControllerBank.step` | copy |  |
| `_rollout_segment` (216) | `_rollout_segment` | copy | hazard H4 (output, controller, corrected step) |
| `closed_loop_rollout` (245) | `closed_loop_rollout` | trim | checkpointed branch (295-311, checkpoint_chunk=0) and the chunk argument removed; bank re-home (269-270) kept (GPU) |
| `window_controller_index` (314) | `window_controller_index` | copy |  |
| `WindowControllerIndex` (354) | `WindowControllerIndex` | copy |  |
| `WindowControllerIndex.__init__` (370) | `WindowControllerIndex.__init__` | copy |  |
| `WindowControllerIndex.build` (377) | `WindowControllerIndex.build` | copy | count guards kept (deepSI windowing) |
| `WindowControllerIndex.verify_by_content` (427) | `WindowControllerIndex.verify_by_content` | copy |  |
| `ClosedLoopSimulator` (445) | `ClosedLoopSimulator` | trim |  |
| `ClosedLoopSimulator.__init__` (482) | `ClosedLoopSimulator.__init__` | trim | checkpoint_chunk and the timing history removed |
| `ClosedLoopSimulator.__getstate__` (535) | `ClosedLoopSimulator.__getstate__` | copy | drops the compiled pair: the polish after fit is eager (hazard H8) |
| `ClosedLoopSimulator.__call__` (559) | `ClosedLoopSimulator.__call__` | trim | ctrl_ix-None guard (563-567) and the update timer removed; compiled branch with cudagraph_mark_step_begin and clones (584-604) kept (GPU) |
| `ClosedLoopSimulator.augment_training_data` (640) | `ClosedLoopSimulator.augment_training_data` | trim | last_window_counts removed |
| `ClosedLoopSimulator._check_val_records` (647) | `ClosedLoopSimulator._check_val_records` | copy |  |
| `ClosedLoopSimulator.validation_error` (667) | `ClosedLoopSimulator.validation_error` | trim | sim-NRMSe only: measure guards (684-692) and the sim-RMS return (716) removed; last_per_channel and last_sim_rms kept (ValHistory) |
| `closed_loop_free_run_rms_batch` (719) | `closed_loop_free_run_rms_batch` | copy | evaluation interface incl. return_pred; grouping by length kept (hazard H5) |

- Dropped (print and check_units only): `ControllerBank.physical_D`.
- Dropped (not used): `ControllerBank.check_units`.
- Dropped (only used by the polish compiled/eager switch, unreachable after fit): `ClosedLoopSimulator.eager`.
- Dropped (print only): `ClosedLoopSimulator._note_update_time`.
- Dropped (one-record wrapper, not used): `closed_loop_free_run_rms`.
- Module and class constants: `__project_origin__`: copy (marker).

#### `MA/fit_systems/interconnect.py` -> `MA/fit_systems/interconnect.py`

| Source (line) | Destination | Action | Removed branches, reason, interface |
|-|-|-|-|
| `Interconnect` (21) | `Interconnect` | trim | Jan |
| `Interconnect.__init__` (22) | `Interconnect.__init__` | trim | debugging, save_signals, first_step_eval removed |
| `Interconnect._apply` (77) | `Interconnect._apply` | copy | keeps connection matrices on the module device and dtype (makes the per-call heals redundant) |
| `Interconnect._module_device_dtype` (88) | `Interconnect._module_device_dtype` | copy |  |
| `Interconnect.init_model` (93) | `Interconnect.init_model` | copy |  |
| `Interconnect.forward` (100) | `Interconnect.forward` | trim | removed: 3-D input branches (112-113, 119-120), per-call device heal (137-141), save_signals (172-182); kept: lazy init_forward (125-127, the evaluation calls hfn before any rollout), summation order, OBC subtraction (hazard H4) |
| `Interconnect.init_forward` (199) | `Interconnect.init_forward` | copy |  |
| `Interconnect._signal_cone` (280) | `Interconnect._signal_cone` | copy |  |
| `Interconnect.output_only` (303) | `Interconnect.output_only` | trim | per-call device heal (330-333) removed |
| `Interconnect.init_connection_matrices` (354) | `Interconnect.init_connection_matrices` | trim | add_to connections (372-373, 428-429) and debugging print (375-378) removed |
| `Interconnect.add_block` (444) | `Interconnect.add_block` | copy | small helper, copied whole |
| `Interconnect.connect_signals` (457) | `Interconnect.connect_signals` | copy | small helper, copied whole |
| `Interconnect.connect_block_signals` (474) | `Interconnect.connect_block_signals` | copy | small helper, copied whole |
| `Interconnect.determine_signal_ix` (483) | `Interconnect.determine_signal_ix` | copy | small helper, copied whole |
| `Interconnect.convert_signal_name_to_ix` (493) | `Interconnect.convert_signal_name_to_ix` | copy | small helper, copied whole |
| `Interconnect.parse_connection_function_method` (525) | `Interconnect.parse_connection_function_method` | copy | small helper, copied whole |
| `Signal_Connection` (535) | `Signal_Connection` | trim |  |
| `Signal_Connection.__init__` (537) | `Signal_Connection.__init__` | copy |  |
| `Dtype_DataLoader` (568) | `Dtype_DataLoader` | copy | default_rng(0) and the in-place shuffle of the deepSI parent (hazard H3) |
| `Dtype_DataLoader.__init__` (594) | `Dtype_DataLoader.__init__` | copy |  |
| `Dtype_DataLoader.__init__._conv` (595) | `Dtype_DataLoader.__init__._conv` | copy |  |
| `SSE_Interconnect` (604) | `SSE_Interconnect` | trim | Jan; class attributes simulator, validation_probes, training_phase kept |
| `SSE_Interconnect.__init__` (628) | `SSE_Interconnect.__init__` | trim | e_net/e_net_kwargs arguments removed (the encoder is always injected); multi_loss_val removed |
| `SSE_Interconnect.init_model` (651) | `SSE_Interconnect.init_model` | trim | sys_data-None branch (653-655), auto_fit_norm branch (658-659), init_nets call removed; to_device, optimizer, scheduler, history arrays, hfn.init_model kept in order |
| `SSE_Interconnect.simulate` (678) | `SSE_Interconnect.simulate` | trim | open-loop rollout (693-705) removed: always the simulator |
| `SSE_Interconnect.make_training_data` (746) | `SSE_Interconnect.make_training_data` | trim | no-simulator branch removed |
| `SSE_Interconnect.cal_validation_error` (753) | `SSE_Interconnect.cal_validation_error` | trim | deepSI fallback (764-765) removed; probes loop kept (ValHistory) |
| `SSE_Interconnect.fit` (808) | `SSE_Interconnect.fit` | trim | removed: init_model_done=False branch (915-922), scheduler warning (929-930), Dataset/DataLoader branch (960-963), concurrent_val (973-976, 1062-1066, 1109-1118), timeout (985, 989-990, 1098-1101), list_val_measures, print_full_time_profile, the extra-array name guards (1010-1020), Tictoctimer and tqdm; kept in order: _check_and_refresh_optimizer_if_needed, dt, cuda move, train(), counters, make_training_data(norm.transform(...)), N_batch_updates_per_epoch = N // batch_size, Dtype_DataLoader, initial validation, loop over loop_dataset with batch .cuda(), closure, optimizer.step(closure), NaN break (1041-1043), training_phase.after_step, counters (epoch_counter += 1/N), validation every its_per_val, end: train(); cpu(); _last save; history keep; _best load; refresh_basis sync and _best re-save (1136-1149). One progress line per validation kept |
| `SSE_Interconnect.fit.validation` (881) | `SSE_Interconnect.fit.validation` | trim | cpu flip (883-884, 909-910) and list_val_measures (892-896) removed; tie rule bestfit >= Loss_val kept |
| `SSE_Interconnect.fit.closure` (1023) | `SSE_Interconnect.fit.closure` | trim | timer calls removed; loss, zero_grad, backward order kept |
| `SSE_Interconnect_Composed` (1257) | `SSE_Interconnect_Composed` | trim | ours (@added); class attributes kept: obc; dropped: orth_penalty, loss_stats, traj_penalty, _optimizer, _traj_*, obc_validation_diagnostics, the optimizer property |
| `SSE_Interconnect_Composed.__init__` (1305) | `SSE_Interconnect_Composed.__init__` | trim | Parameterized_MSD guard (1309-1317) and _param_ix removed; _ann_ix, _mats_ix kept |
| `SSE_Interconnect_Composed._sync_prediction_state_for_validation` (1332) | `SSE_Interconnect_Composed._sync_prediction_state_for_validation` | trim | diagnostic dict (1367-1380) removed: theta = obc.prediction_coefficient(ann), set_frozen(theta, basis.vbar); refresh rules (1350-1359) and the epoch floor(+1e-6) kept |
| `SSE_Interconnect_Composed.loss` (1383) | `SSE_Interconnect_Composed.loss` | trim | removed: wiring guards (1402-1405), w-is-None branch (1421-1422), loss_stats (1430-1431), param_loss sweep (1435-1436: adds the float 0.0, exact), orth penalty (1440-1445); kept in order: encoder, pass_mats hoist, OBC maybe_refresh/coefficient/set_frozen/tangent_pair, weighted MSE (hazard H5) |
| `SSE_Interconnect_Composed.checkpoint_save_system` (1505) | `SSE_Interconnect_Composed.checkpoint_save_system` | trim | stash = obc, obc_reference, training_phase |
| `SSE_Interconnect_Composed.checkpoint_load_system` (1514) | `SSE_Interconnect_Composed.checkpoint_load_system` | trim | stash and reinstall obc, obc_reference, training_phase; optimizer migration, adapter rebind and re-wrap (1522-1567) removed |
| `SSE_Interconnect_Composed.__getstate__` (1570) | `SSE_Interconnect_Composed.__getstate__` | trim | drops obc, obc_reference, training_phase (deepcopy in the evaluation) |

- Dropped (save_signals path): `Interconnect.reset_saved_signals`.
- Dropped (not used): `Interconnect.convert_signal_ix_to_name`.
- Dropped (printing): `Signal_Connection.__str__`, `Signal_Connection.__repr__`.
- Dropped (not on the spec path): `modified_encoder_net`, `modified_encoder_net.__init__`, `modified_encoder_net.forward`, `SSE_Interconnect_ParamLoss.loss`.
- Dropped (only builds an encoder when none is injected (647-649); init_model no longer calls it): `SSE_Interconnect.init_nets`.
- Dropped (overridden by SSE_Interconnect_Composed.loss): `SSE_Interconnect.loss`.
- Dropped (base no-op; the instance is always SSE_Interconnect_Composed): `SSE_Interconnect._sync_prediction_state_for_validation`.
- Dropped (open-loop deepSI paths): `SSE_Interconnect._model_dtype`, `SSE_Interconnect.init_state_multi`, `SSE_Interconnect.measure_act_multi`.
- Dropped (replaced by SSE_Interconnect_Composed): `SSE_Interconnect_ParamLoss`.
- Dropped (nf probe statistics (no numeric effect)): `WindowErrorStats`, `WindowErrorStats.__init__`, `WindowErrorStats.reset`, `WindowErrorStats.update`, `WindowErrorStats.summary`.
- Dropped (legacy optimizer migration only): `SSE_Interconnect_Composed.__setstate__`.
- Dropped (property for the trajectory penalty (getter and setter); optimizer becomes a plain attribute): `SSE_Interconnect_Composed.optimizer`, `SSE_Interconnect_Composed.optimizer`.
- Dropped (trajectory penalty (fit delegates to the parent when it is None)): `SSE_Interconnect_Composed._assert_optimizer_supported`, `SSE_Interconnect_Composed._wrap_optimizer_step`, `SSE_Interconnect_Composed._wrap_optimizer_step.step_with_penalty`, `SSE_Interconnect_Composed._wrap_optimizer_step.step_with_penalty.wrapped`, `SSE_Interconnect_Composed._unwrap_optimizer_step`, `SSE_Interconnect_Composed.fit`.
- Module and class constants: `SSE_Interconnect_Composed._TRAJ_UNPICKLABLE`, `SSE_Interconnect_Composed._TRAJ_COUNTERS`, `SSE_Interconnect_Composed._TRAJ_ALLOWED_OPTIMIZERS`: drop (trajectory penalty); `SSE_Interconnect_Composed._OBC_UNPICKLABLE`, `SSE_Interconnect_Composed._PHASE_UNPICKLABLE`: copy -> MA/fit_systems/interconnect.py.

#### `MA/fit_systems/lbfgs_polish.py` -> `MA/fit_systems/lbfgs_polish.py`

| Source (line) | Destination | Action | Removed branches, reason, interface |
|-|-|-|-|
| `PolishSpec` (50) | `PolishSpec` | trim | fields kept: n_windows, max_iter, inner_iter, history_size, tol_grad, tol_change, seed; dropped: freeze, meter_rel_tol, time_budget_s, chunk, meters_gate |
| `decide` (97) | (merged) | merge | into lbfgs_polish: accepted = finite(loss_last) and finite(val_before) and finite(val_after) and val_after < val_before (the meters_gate=False, accept_without_validation=False path of 110-127) |
| `_named_params` (134) | `_named_params` | copy | parameter order = deepSI parameters_with_names order (hazard H7) |
| `_snapshot` (155) | `_snapshot` | copy |  |
| `_restore` (165) | `_restore` | copy |  |
| `_resync` (171) | `_resync` | copy |  |
| `window_count` (206) | `window_count` | trim | stride == 1 branch (226-227) removed |
| `FixedBatch` (234) | `FixedBatch` | trim | one chunk; yields (uh, yh, uf, yf, kwargs) once |
| `FixedBatch.__init__` (258) | `FixedBatch.__init__` | trim | chunk machinery and extra-array guards (309-318) removed; stride scaling, window_count guard (277-283), default_rng(seed).choice + sort (288-289) kept |
| `FixedBatch.__init__._to_tensor` (296) | `FixedBatch.__init__._to_tensor` | copy |  |
| `FixedBatch.__iter__` (334) | `FixedBatch.__iter__` | trim | single chunk; the weight 1.0 multiply is dropped (x * 1.0 is exact) |
| `_make_closure` (464) | `_make_closure` | trim |  |
| `_make_closure.closure` (466) | `_make_closure.closure` | trim | zero_grad(set_to_none=False), loss, backward, return torch.as_tensor(float(loss)) (default dtype float32, hazard H7); grad-None fill (476-478) removed |
| `_loss_value` (483) | `_loss_value` | trim | single chunk |
| `_lbfgs_counts` (500) | `_lbfgs_counts` | copy |  |
| `_is_oom` (515) | `_is_oom` | copy | GPU failure path (reason text, empty_cache) |
| `_probe_closure` (522) | `_probe_closure` | copy | two closures before stepping; a failure or non-reproducible loss aborts the polish (weights unchanged) |
| `lbfgs_polish` (549) | `lbfgs_polish` | trim | removed: meters_fn, force_eager, accept_without_validation, freeze checks (621-630), compiled_available and the compiled try with its fallback (657-686) reduced to one eager probe, time budget (757-759), peak-memory stats; kept in order: snapshot, val_before (passed, else measured), FixedBatch, _new_opt, probe, n_outer blocks of opt.step(closure) with exception restore (721-741), convergence break done < inner_iter (749-756), iteration cap, loss_last, ran, val_after, accept rule, restore + _resync on rejection |
| `lbfgs_polish._validate` (597) | `lbfgs_polish._validate` | trim | val_sys_data None branch and _quiet_probes removed; eval(), _resync, cal_validation_error, train() kept |
| `lbfgs_polish._new_opt` (650) | `lbfgs_polish._new_opt` | copy | LBFGS lr 1, strong_wolfe |

- Dropped (meters do not gate (meters_gate=False)): `meter_regression`.
- Dropped (meters): `generic_meters`, `lbfgs_polish._all_meters`.
- Dropped (the polish after fit is always eager (hazard H8)): `_eager`.
- Dropped (freeze = () (no-op)): `_frozen`.
- Dropped (no probes or loss_stats are attached during the polish (train restores the probe tuple before it)): `_quiet_probes`.
- Dropped (lazy init is done by fit; the call only re-solves the same OBC coefficient): `warm_up_lazy_init`.
- Module and class constants: `__project_origin__`: copy (marker).

#### `MA/fit_systems/obc.py` -> `MA/fit_systems/obc.py`

| Source (line) | Destination | Action | Removed branches, reason, interface |
|-|-|-|-|
| `OBCRankLossAbort` (118) | `OBCRankLossAbort` | copy |  |
| `RankReport` (124) | `RankReport` | trim | dataclass fields kept (rank, tol, n_cols, cond, singular values) |
| `OBCBasis` (160) | `OBCBasis` | trim |  |
| `OBCBasis.__init__` (169) | `OBCBasis.__init__` | copy |  |
| `OBCBasis.rank_tolerance` (183) | `OBCBasis.rank_tolerance` | copy |  |
| `OBCBasis.from_matrix` (191) | `OBCBasis.from_matrix` | copy | float64 SVD, column scaling, rank cutoff (hazard H9) |
| `OBCBasis.coefficient` (242) | `OBCBasis.coefficient` | copy |  |
| `OBCBasis.to` (322) | `OBCBasis.to` | copy | GPU path (basis and field on different devices) |
| `OBCBasis.device` (330) | `OBCBasis.device` | copy |  |
| `build_stacked_sensitivity` (341) | `build_stacked_sensitivity` | copy | evaluation interface; dtype guard kept |
| `evaluate_step_rows` (367) | `evaluate_step_rows` | copy |  |
| `OBCLifecycle` (392) | `OBCLifecycle` | trim |  |
| `OBCLifecycle.__init__` (410) | `OBCLifecycle.__init__` | trim | refresh-policy guard, verbose, name_columns, history/timing/counters removed; rank policy state (basis, last_refresh_epoch, consecutive_rank_loss, max 2) kept |
| `OBCLifecycle.refresh` (446) | `OBCLifecycle.refresh` | trim | basis0, r_J drift, history entry and prints removed; rank policy kept whole: retain previous basis on one loss, abort on two consecutive or with no basis (460-478); device move (454-455) kept |
| `OBCLifecycle.maybe_refresh` (501) | `OBCLifecycle.maybe_refresh` | copy |  |
| `OBCLifecycle.coefficient` (512) | `OBCLifecycle.coefficient` | trim | bookkeeping (last_theta, last_field_norm, timing, n_objective_evals) removed; basis-None guard and device move kept |
| `OBCLifecycle.validation_coefficient` (543) | `OBCLifecycle.prediction_coefficient` | rewrite | returns theta only: the no-grad solve of _prediction_solve(diagnostics=False) (558-568); theta is computed before the diagnostics in mainline, so it is the same tensor |
| `OBCLifecycle._prediction_solve` (558) | (merged) | merge | into OBCLifecycle.prediction_coefficient (lines 559-568) |

- Dropped (static_pass_buffers=False): `_StaticCopy`, `_StaticCopy.forward`, `_StaticCopy.backward`, `StaticPassBuffers`, `StaticPassBuffers.__init__`, `StaticPassBuffers._alloc`, `StaticPassBuffers.__call__`, `StaticPassBuffers.reset`.
- Dropped (refresh print (comparison.md: diagnostic)): `RankReport.lines`.
- Dropped (diagnostic (comparison.md)): `OBCBasis.apply`, `OBCBasis.jt`, `OBCBasis.project_onto`, `OBCBasis.project_out`, `OBCBasis.r_perp`, `OBCBasis.r_overlap`, `OBCBasis.frobenius`, `OBCBasis.relative_change`, `OBCBasis.memory_bytes`.
- Dropped (jvp path): `directional_correction`.
- Dropped (its name is reused by the merged validation path above): `OBCLifecycle.prediction_coefficient`.
- Dropped (diagnostic): `OBCLifecycle.stacked_diagnostics`, `OBCLifecycle.memory_bytes`, `OBCLifecycle.state_summary`.
- Module and class constants: `__project_origin__`: copy (marker).

#### `MA/fit_systems/pre_encoder.py` -> `MA/fit_systems/pre_encoder.py`

| Source (line) | Destination | Action | Removed branches, reason, interface |
|-|-|-|-|
| `linear_encoder_init_aug` (306) | `linear_encoder_init_aug` | trim | @added; state_dict names Wb_psi_y, Wb_psi_u, Wa_psi_y, Wa_psi_u, net.{0,2,4} |
| `linear_encoder_init_aug.__init__` (321) | `linear_encoder_init_aug.__init__` | trim | only wb_init="random", xavier_uniform_gain1, no offsets: removed the reconstructability map (378-413), wb_init guard (369-370), legacy Kaiming (422-427), D-055 offset buffers (470-486) and unused A, B, C, D, flag_linear_only arguments; draw order kept: Wb uniform_ (376-377), Wa xavier (432-433), net Linear x3 (446-451), then zeroing (hazard H1) |
| `linear_encoder_init_aug.forward` (488) | `linear_encoder_init_aug.forward` | trim | fix_enabled offsets (498-500, 508-509) and flag_linear_only (518-519) removed; input order [u, y] for the net, W_u uh + W_y yh for the linear part kept |

- Dropped (not on the spec path): `convert_to_same_type`, `System_data_with_x`, `System_data_with_x.__init__`, `System_data_with_x.to_hist_future_data`, `System_data_norm_with_x`, `System_data_norm_with_x.__init__`, `System_data_norm_with_x.transform`, `System_data_norm_with_x.inverse_transform`, `linear_encoder_init`, `linear_encoder_init.__init__`, `linear_encoder_init.forward`, `SS_pre_encoder`, `SS_pre_encoder.__init__`, `SS_pre_encoder.init_nets`, `SS_pre_encoder.loss`, `SS_pre_encoder.n_step_error`.

#### `MA/fit_systems/ps2_opening.py` -> `MA/fit_systems/ps2_opening.py`

| Source (line) | Destination | Action | Removed branches, reason, interface |
|-|-|-|-|
| `PS2Spec` (37) | `PS2Spec` | copy |  |
| `calibration_split` (59) | `calibration_split` | copy |  |
| `PS2Opening` (69) | `PS2Opening` | copy |  |
| `PS2Opening.__init__` (70) | `PS2Opening.__init__` | copy |  |
| `PS2Opening.for_pipeline` (87) | `PS2Opening.for_pipeline` | copy |  |
| `PS2Opening.done` (98) | `PS2Opening.done` | copy |  |
| `PS2Opening.after_step` (101) | `PS2Opening.after_step` | copy |  |
| `PS2Opening.summary` (130) | `PS2Opening.summary` | copy |  |
| `PS2Opening.summary_json` (135) | `PS2Opening.summary_json` | copy | summary feeds the .npz ps2_summary |
| `PS2Opening._device` (139) | `PS2Opening._device` | copy |  |
| `PS2Opening._start` (143) | `PS2Opening._start` | copy |  |
| `PS2Opening._sample` (161) | `PS2Opening._sample` | copy |  |
| `PS2Opening._residual` (183) | `PS2Opening._residual` | copy |  |
| `PS2Opening._xa` (191) | `PS2Opening._xa` | copy |  |
| `PS2Opening._s1_step` (194) | `PS2Opening._s1_step` | copy |  |
| `PS2Opening._cal_unexplained` (207) | `PS2Opening._cal_unexplained` | copy |  |
| `PS2Opening._fit_map` (213) | `PS2Opening._fit_map` | copy | D-274 change 1 (233-235): the start parameters are the first candidate |
| `PS2Opening._fit_map.snap` (220) | `PS2Opening._fit_map.snap` | copy |  |
| `PS2Opening._fit_map.ee` (226) | `PS2Opening._fit_map.ee` | copy | D-274: add the comment that T.var(0) divides by B-1, i.e. (B-1)/B times eq:ps2_s2, same minimiser |

- Module and class constants: `__project_origin__`: copy (marker).

#### `MA/systems/gantry_ss.py` -> `MA/systems/gantry_ss.py`

| Source (line) | Destination | Action | Removed branches, reason, interface |
|-|-|-|-|
| `build_poly_constants` (150) | `build_poly_constants` | copy |  |
| `build_G_matrix_entries` (184) | `build_G_matrix_entries` | copy |  |

- Module and class constants: `__project_origin__`: copy (marker); `_D`, `mb`, `mh`, `m1`, `m2`, `Jb`, `Jh`, `cg1`, `cg2`, `cy`, `cb1`, `cb2`, `kb1`, `kb2`, `Lb`, `d`, `_z`, `M1`, `M2`, `C`, `K`, `P`, `Cd`, `Dd`: copy (float32, hazard H6) -> MA/systems/gantry_ss.py; `fs`, `ts`, `M0`: drop (not read); `_kx_art`, `_ky_art`: drop (GANTRY_KX_ART/KY_ART env overrides; the K[0,0], K[2,2] branch goes with them).

#### `MA/utils/deepSI_corrections.py` -> `MA/utils/deepSI_corrections.py`

| Source (line) | Destination | Action | Removed branches, reason, interface |
|-|-|-|-|
| `fixed_System_data` (6) | `fixed_System_data` | copy |  |
| `fixed_System_data.__init__` (7) | `fixed_System_data.__init__` | copy |  |
| `fixed_System_data.to_hist_future_data` (10) | `fixed_System_data.to_hist_future_data` | copy | window arithmetic (hazard H5) |
| `fixed_System_data_norm` (30) | `fixed_System_data_norm` | trim |  |
| `fixed_System_data_norm.__init__` (44) | `fixed_System_data_norm.__init__` | copy |  |
| `fixed_System_data_norm.is_id` (51) | `fixed_System_data_norm.is_id` | copy |  |
| `fixed_System_data_norm.transform` (69) | `fixed_System_data_norm.transform` | copy |  |

- Dropped (not on the spec path): `fixed_System_data_norm.make_training_data`, `fixed_System_data_norm.fit`, `fixed_System_data_norm.inverse_transform`, `fixed_System_data_norm.u_transform`, `fixed_System_data_norm.y_transform`, `fixed_System_data_norm.u_inv_transform`, `fixed_System_data_norm.y_inv_transform`, `fixed_System_data_norm.__repr__`, `convert_to_same_type`.
- Module and class constants: `__project_origin__`: copy (marker).

#### `MA/utils/torch_nets.py` -> `MA/utils/torch_nets.py`

| Source (line) | Destination | Action | Removed branches, reason, interface |
|-|-|-|-|
| `live_g_feed_forward_nn` (170) | `live_g_feed_forward_nn` | copy | three nn.Linear draws at float32 (hazard H1) |
| `live_g_feed_forward_nn.__init__` (178) | `live_g_feed_forward_nn.__init__` | copy |  |
| `live_g_feed_forward_nn.forward` (203) | `live_g_feed_forward_nn.forward` | copy |  |

- Dropped (not on the spec path): `linear_encoder_net`, `linear_encoder_net.__init__`, `linear_encoder_net.forward`, `feed_forward_nn`, `feed_forward_nn.__init__`, `feed_forward_nn.forward`, `identity_init_simple_res_net`, `identity_init_simple_res_net.__init__`, `identity_init_simple_res_net.forward`, `linear_mapping`, `linear_mapping.__init__`, `linear_mapping.forward`, `zero_init_linear_mapping`, `zero_init_linear_mapping.__init__`, `zero_init_linear_mapping.forward`, `zero_init_feed_forward_nn`, `zero_init_feed_forward_nn.__init__`, `zero_init_feed_forward_nn.forward`, `paper_resnet`, `paper_resnet.__init__`, `paper_resnet.forward`, `zero_init_resnet`, `zero_init_resnet.__init__`, `zero_init_resnet.forward`, `unit_variance_feed_forward_nn`, `unit_variance_feed_forward_nn.__init__`, `unit_variance_feed_forward_nn.forward`, `HybridGantryEncoder`, `HybridGantryEncoder.__init__`, `HybridGantryEncoder.forward`, `LinearInitEncoderWrapper`, `LinearInitEncoderWrapper.__init__`, `LinearInitEncoderWrapper.forward`, `positive_default_encoder_net`, `positive_default_encoder_net.__init__`, `positive_default_encoder_net.forward`.

#### `MA/utils/utils.py` -> `MA/utils/utils.py`

| Source (line) | Destination | Action | Removed branches, reason, interface |
|-|-|-|-|
| `added` (7) | `added` | copy | the @added marker |
| `to_tensor` (15) | `to_tensor` | copy | torch.Tensor(ndarray) = float32 materialisation (hazard H6) |
| `detect_algebraic_loop` (31) | `detect_algebraic_loop` | copy |  |
| `expansion_matrix` (157) | `expansion_matrix` | copy | real matmuls in the summation (hazard H4) |
| `selection_matrix` (162) | `selection_matrix` | copy |  |

- Dropped (not on the spec path): `to_ndarray`, `plot_fft_freq`, `plot_sys_data`, `get_sys_data_dimensions`, `determine_sys_data_signal_dimension`, `determine_std_T`, `determine_std_T_sys_data`, `normalize_linear_ss_matrices`, `determine_yes_no_query_binary_output`, `compute_noisy_y_from_SNR`.

#### Files dropped whole

- `GD/baselines.py` (diagnostic (Dirk excluded)): `stepwise_rollout`, `compute_baseline_fp_nrms`, `compute_baseline_fp_nrms._fp_step`, `compute_all_baselines`; constants `__project_origin__`.
- `GD/evaluation.py` (diagnostic (Dirk excluded)): `capture_loss_history`, `report_joint_estimation`, `_record_mean`, `evaluate_and_save`, `_make_plots`, `_build_save_dict`, `per_record_nrms`; constants `__project_origin__`.
- `GD/obc_diagnostics.py` (diagnostic (Dirk excluded)): `qualify_basis`, `print_qualification`, `principal_angles`, `absorber_truth_one_step`, `absorber_truth_one_step.M`, `absorber_truth_one_step.deriv`, `check_truth_parameters`, `discrepancy_fields`, `rho_table`; constants `__project_origin__`.
- `CM/feedback_curriculum.py` (not on the spec path): `FixedWindowBatch`, `build_fixed_window_batch`, `save_training_snapshot`, `load_training_snapshot`, `FeedbackTargetSelector`, `FeedbackTargetSelector.__init__`, `FeedbackTargetSelector.initial_strength`, `FeedbackTargetSelector._evaluate`, `FeedbackTargetSelector._checkpoint_path`, `FeedbackTargetSelector._write_state`, `FeedbackTargetSelector.__call__`, `local_dynamics_diagnostic`, `local_dynamics_diagnostic.state_from_x`, `local_dynamics_diagnostic.state_from_u`, `local_dynamics_diagnostic.output_from_x`; constants `FeedbackTargetSelector.SCHEDULE`.
- `CM/oracle.py` (diagnostic (Dirk excluded)): `_M`, `_deriv`, `simulate`, `oracle_open_loop`, `truth_from_meta`, `_deriv_truth`, `oracle_closed_loop`; constants `__project_origin__`, `m1`, `m2`, `mb`, `mh`, `Jb`, `Jh`, `cg1`, `cg2`, `cy`, `cb1`, `cb2`, `kb1`, `kb2`, `Lb`, `d`, `P_np`, `MA_FRAC`, `MA`, `MH_RIGID`, `L0`, `FA`, `KA`, `ZETA_A`, `CA`, `_C4`, `_K4`, `_E43`.
- `CM/orth_penalty.py` (not on the spec path): `theta_bar_for`, `_cache_path`, `_x_logical_from_data`, `_make_block`, `_compute`, `build_orth_penalty`; constants `__project_origin__`, `_F64`.
- `CM/rezero_gate.py` (not on the spec path): `_Gate`, `_Gate.__init__`, `_Gate.gate`, `_Gate.forward`, `apply_rezero_gate`; constants `__project_origin__`.
- `MA/fit_systems/augmented_dynamics.py` (not on the spec path): `AugmentedDynamics`, `AugmentedDynamics.__init__`, `AugmentedDynamics.net`, `AugmentedDynamics.rho`, `AugmentedDynamics.gamma`, `AugmentedDynamics.effective_B`, `AugmentedDynamics.trainable_parameters`, `AugmentedDynamics.pole_table`, `AugmentedDynamics.forward`, `find_ann_block`, `find_wrapper`, `final_linear`, `reset_final_layer`, `empirical_input_scale`, `draw_input_map`; constants `__project_origin__`, `GATE_ZERO`, `OUTPUT_ZERO`.
- `MA/fit_systems/continuous_augmentation.py` (not on the spec path): `rk4_step`, `CombinedContinuousGantryBlock`, `CombinedContinuousGantryBlock.__init__`, `CombinedContinuousGantryBlock.correction_field`, `CombinedContinuousGantryBlock.derivative`, `CombinedContinuousGantryBlock.forward`, `CombinedContinuousGantryBlock.param_loss`, `CombinedContinuousGantryBlock.__getattr__`.
- `MA/fit_systems/gyorok_orth_projection.py` (not on the spec path): `_block_step`, `gantry_orth_matrix`, `gantry_orth_matrix.f_single`, `finite_difference_check`, `orth_basis`, `spectrum_report`, `GyorokOrthPenalty`, `GyorokOrthPenalty.__init__`, `GyorokOrthPenalty.Q`, `GyorokOrthPenalty.penalty_of_field`, `GyorokOrthPenalty.forward`; constants `__project_origin__`, `_F64`.
- `MA/fit_systems/multiple_shooting.py` (not on the spec path): `SSE_Interconnect_MultipleShooting`, `SSE_Interconnect_MultipleShooting._norm`, `SSE_Interconnect_MultipleShooting._defect_penalty`, `SSE_Interconnect_MultipleShooting.loss`; constants `__project_origin__`.
- `MA/fit_systems/obc_projection.py` (not on the spec path): `reduced_step`, `stacked_basis`, `stacked_basis.f`, `directional`, `rank_report`, `principal_angles`, `RankLossAbort`, `OBCBasis`, `OBCBasis.__init__`, `OBCBasis._check_rank`, `OBCBasis.range_basis`, `OBCBasis.coefficient`, `OBCBasis.rho`, `OBCBasis.residual_ratio`, `OBCBasis.spectrum`, `OBC_Projected_ANN_Block`, `OBC_Projected_ANN_Block.__init__`, `OBC_Projected_ANN_Block.phys`, `OBC_Projected_ANN_Block.init_block`, `OBC_Projected_ANN_Block.phys_input`, `OBC_Projected_ANN_Block.rebuild_basis`, `OBC_Projected_ANN_Block.clear_basis`, `OBC_Projected_ANN_Block._stack_physical_write`, `OBC_Projected_ANN_Block.coefficient`, `OBC_Projected_ANN_Block.forward`; constants `__project_origin__`, `DEFAULT_RCOND`.
- `MA/fit_systems/orth_projection.py` (not on the spec path): `OrthProjectionPenalty`, `OrthProjectionPenalty.__init__`, `OrthProjectionPenalty.penalty_of_field`, `OrthProjectionPenalty.forward`, `SSE_Interconnect_OrthLoss`, `SSE_Interconnect_OrthLoss.loss`; constants `__project_origin__`.
- `MA/fit_systems/white_box_models.py` (not on the spec path): `Discrete_White_Box_Model`, `Discrete_White_Box_Model.__init__`, `Discrete_White_Box_Model.forward`, `Discrete_White_Box_Model.f`, `Discrete_White_Box_Model.h`, `Discrete_Cascaded_Tanks`, `Discrete_Cascaded_Tanks.__init__`, `Discrete_Cascaded_Tanks.init_model`, `Discrete_Cascaded_Tanks.f`, `Discrete_Cascaded_Tanks.h`.
- `MA/systems/cascaded_tanks.py` (not on the spec path): `Cascaded_Tanks`, `Cascaded_Tanks.__init__`, `Cascaded_Tanks.deriv`, `Cascaded_Tanks.h`, `Extended_Cascaded_Tanks`, `Extended_Cascaded_Tanks.__init__`, `Extended_Cascaded_Tanks.deriv`, `Extended_Cascaded_Tanks.h`.
- `MA/systems/gantry_linearization.py` (not on the spec path): `gantry_linearize_and_discretize`, `verify_linearization`, `verify_linearization.deriv`; constants `__project_origin__`.
- `MA/systems/mass_spring_damper.py` (not on the spec path): `Mass_spring_damper`, `Mass_spring_damper.__init__`, `Mass_spring_damper.deriv`, `Mass_spring_damper.h`, `Msd_2dof`, `Msd_2dof.__init__`, `Msd_2dof.deriv`, `Msd_2dof.h`, `Msd_ndof`, `Msd_ndof.__init__`, `Msd_ndof.cast_parameter_to_array`, `Msd_ndof.determine_mask`, `Msd_ndof.deriv`, `Msd_ndof.h`, `Msd_ndof.expand_with_ix`, `Msd_ndof_lpf_output`, `Msd_ndof_lpf_output.__init__`, `Msd_ndof_lpf_output.cast_parameter_to_array`, `Msd_ndof_lpf_output.determine_mask`, `Msd_ndof_lpf_output.deriv`, `Msd_ndof_lpf_output.h`, `Msd_ndof_lpf_output.expand_with_ix`, `Msd_ndof_input_saturation`, `Msd_ndof_input_saturation.__init__`, `Msd_ndof_input_saturation.cast_parameter_to_array`, `Msd_ndof_input_saturation.determine_mask`, `Msd_ndof_input_saturation.deriv`, `Msd_ndof_input_saturation.h`, `Msd_ndof_input_saturation.expand_with_ix`.
- `GD/__init__.py` (no definitions): `__project_origin__` copy (marker).

## 3. Config design

One frozen dataclass. The five axes are its fields; the run environment (device, compilation, budgets that only the
smoke test and the harness change) are fields with production defaults; every fixed setting the kept code reads is
a `ClassVar` under its mainline attribute name, so copied code (`cfg.stride`, `cfg.adam_eps`, `hp['lr']`) needs no
edit, and a setting that only selected a removed branch is not defined at all (an `AttributeError` then exposes a
missed trim).

```python
@dataclass(frozen=True)
class RunConfig:
    # the five axes
    arm: str                                   # 'obc' | 'u'
    start: str                                 # 'detuned' | 'correct'
    n_a: int                                   # added states (spec: 0, 2, 8)
    ps2: bool                                  # PS2 opening of the added-state part
    noise: str                                 # 'lowpass' | 'noisefree'
    # run environment: production defaults; only the cpu and smoke words and the harness change them
    device: str = 'cuda'                       # 'cuda' | 'cpu'
    compile_mode: Optional[str] = 'reduce-overhead'   # None = eager; needs device 'cuda'
    truncate_s: Optional[float] = None         # seconds kept per record (tests only)
    epochs: int = 200
    n_its: Optional[int] = None                # Adam update cap; then one validation at the cap (mainline rule, model.py:501)
    its_per_val: int = 1300
    lbfgs_max_iter: int = 100
    lbfgs_inner_iter: int = 20
    lbfgs_windows: int = 16384
    ps2_smoke: bool = False                    # the short PS2Spec of the mainline smoke (training.py:873-876)
```

`ClassVar` constants (values of the spec; labels as in mainline): `seed = 1`, `nx_phys, nu, ny = 6, 3, 3`,
`fs_orig = 20000`, `fs_new = 4000`, `stride = 10`, `nf_seconds = 0.1`, `na_nb = 30`, `n_nodes_per_layer = 16`,
`n_hidden_layers = 2`, `up_sample = 1`, `batch_size = 512`, `lr = 1e-5`, `adam_eps = 1e-16`, `param_prior = False`,
`param_rmse_baseline = 0.01` (HEURISTIC, D-076: still sets the `Lambda_combo` buffer, a state_dict key),
`detune = 0.10` (HEURISTIC, D-222), `encoder_init = 'data_lsq'`, `validation_measure = 'sim-NRMSe'`,
`controller_gain = False`, `start_phase = 'adam'`, `lbfgs_history = 50`, `lbfgs_tol_grad = 2.6e-5`,
`lbfgs_tol_change = 2.6e-7` (HEURISTIC, D-263), `obc_space = 'affine'`, `obc_refresh = 'epoch'`,
`obc_rank_rtol = None`, `obc_ref_successor = True`, `obc_ref_chunk = 16384`. The implementer checks the list with a
grep of `cfg\.\w+` over the kept code.

Properties: `obc` (arm == 'obc'), `nx_ann` (n_a), `ann_route_ix` (`tuple(range(6 + n_a))`), `mode`
(`thesis_taf_lowpass_noise` | `thesis_taf_noisefree`), `combo_init_detune` (None for 'correct'; for 'detuned'
`[1.0 + 0.10 * s for s in np.random.default_rng(1).choice([-1.0, 1.0], size=10)]`, the EP:336-337 draw from an
independent generator, H1), and the mainline derived properties `fs_new_hz`, `d`, `ts_new`, `dtype_np` (float64),
`dtype_pt` (float64), `nf` (400), `hp` (same keys, same order). `__post_init__` checks the axis and environment
values against their allowed sets and `compile_mode is None or device == 'cuda'` (mainline config.py:480-481).

**Presets** (`GD/config.py PRESETS`, named by the thesis and evaluation labels, open choice 5):

| Preset | runs.tsv id | arm | start | n_a | ps2 |
|-|-|-|-|-|-|
| R1 | 221 | obc | detuned | 2 | 1 |
| R2 | 222 | obc | correct | 2 | 1 |
| R3 | 223 | u | correct | 2 | 0 |
| R4 | 224 | u | detuned | 2 | 1 |
| R5 | 225 | u | correct | 2 | 1 |
| R6 | 226 | u | correct | 0 | 0 |
| R7 | 227 | u | correct | 8 | 1 |
| R13 | 229 | obc | detuned | 0 | 0 |
| R14 | 230 | u | detuned | 0 | 0 |

Checked against runs.tsv rows 221 to 230 and `eval_common.FINAL_DESIGN` (identical). A preset fixes four axes; noise,
the fifth, is given separately, so 18 runs come from 9 presets.

**Launch (CLI, open choice 4).** `python -u gantry_interconnect_dynamic.py <preset> <noise> [cpu] [smoke]`: two
positional arguments and two optional words, parsed with `argparse` choices.
- Server: `srun python -u Thesis-writeup/Code/Training/gantry_interconnect_dynamic.py R1 lowpass` (a new sbatch is
  step-4 or later work; the mainline `run_thesis.sbatch` environment block applies unchanged: threads, inductor and
  triton caches, CUDA_VISIBLE_DEVICES).
- Local: `... R1 lowpass cpu` (`device='cpu'`, `compile_mode=None`).
- Launch test: the word `smoke` applies the mainline THESIS_SMOKE values (EP:368-372 and training.py:873-876):
  `truncate_s=2.0, compile_mode=None, n_its=4, its_per_val=4, lbfgs_inner_iter=1, lbfgs_max_iter=2,
  lbfgs_windows=256, ps2_smoke=True`.
- The harness builds `RunConfig(...)` directly and calls `main(cfg, run_id)`.
- Environment variables the clean code reads: `SLURM_JOB_ID` (run id) and `GIT_SHA` / `GIT_COMMIT` (provenance
  fallback in `git_provenance`). No `THESIS_*`, `RESUME_CHECKPOINT`, `DATASET_AB_*`, `OBC_ARM`, `ANN_REZERO_GATE`,
  `GANTRY_KX_ART` or `GANTRY_KY_ART`.

**config.json** (`main`, as EP:403-406): `config_json_dict(cfg, git)` writes every mainline key in mainline order with
the value of the run (constants for the fixed settings), then the new keys `ARM, START, N_A, NOISE, PS2_SMOKE`,
`CODE: "clean"`, then `GIT`; `main` adds `hp`, `run_id`, `data_dir`. One key changes meaning: `NF_PROBE_PRINT` is
written `False` (the probe no longer exists, open choice 20). All keys the evaluation reads are present (section 5).

## 4. Data

- `DATA_ROOT = <repo>/Thesis-writeup/Data`, reached as `ROOT/../../Data` from the folder. Two named entries,
  `DATASETS = {'lowpass': 'Coulomb-tanh-and-MSD-lowpass-noise', 'noisefree': 'Coulomb-tanh-and-MSD-noise-free'}`;
  `cfg.noise` picks one. Nothing else in `Thesis-writeup/Data` is read (the twins are evaluation-only, section 5).
- **Split** from each entry's `MANIFEST.csv`, in manifest order: `block` training to train, validation to val,
  test-I and test-E to test, `frf` ignored; only rows with `controller == 'K1'` (D-274). Checked on both manifests on
  2026-10-19: training 18 K1, validation 6 K1, test-I 6 K1, test-E 5 K1 plus E5 (K2-g+, dropped), frf 12. Result
  18/6/11. File path `<folder>/<file>.mat` from the manifest columns.
- **Per record** (`_load_record`, unchanged arithmetic): `loadmat`; register the record's stored controller
  (`meta.controller`, into the controller registry); `u` = block mean of `u_total` over D = 5 samples in float64
  (D-087); `y` by the noise entry: `lowpass` = `r[::5] + sosfiltfilt(butter(8, 1600 Hz, fs 20 kHz, sos), y - r)[::5]`
  (D-232), `noisefree` = `y[::5]` (point sampling); `r` point-sampled; every array trimmed by `record_trim` = 20
  samples at both ends; then `truncate_s` (tests only); `e = y - r` in float64; `System_data(u, y, dt=2.5e-4)` in
  float64.
- **Derived**: `servo_sigma(e, k0 = 30)` (population std from k0, D-262) per record and axis; training loss weights
  `ystd / sigma_e`; validation scales for `sim-NRMSe`. Normalisation over the concatenated training records
  (`compute_normalization`, H6).
- The noise entry changes exactly two things: the y rule above and `mode` (folder family name, `MODE` and
  `Y_DECIMATION` in config.json). The step-1 trace found no other line that depends on it (`inventory-claude.md`
  A.0).
- Truth fields in the records (`x_logical`, `delta_a`, `vdelta_a`, `meta.truth`) are not read (open choice 11).

## 5. Evaluation interface

**Assumption of the handoff, settled.** `eval_common.py`, `eval_greybox.py`, `eval_overlap.py`, `eval_negation.py`,
`eval_frf.py` and `eval_blackbox.py` were read on 2026-10-19 (imports and every attribute access). They reach the
pipeline only through the interface of `inventory-claude.md` C6 and `inventory-codex.md` C4, with two additions found
now: `eval_overlap.deployed_basis` reads `corr.space` (default 'tangent' when absent) and passes `space=` to
`make_basis_builder`; `eval_blackbox` builds its config with the black-box `bbcl.setup.cfg_for`, a mainline
`RunConfig` with about fifteen fields the clean config does not have.

What the clean code keeps for the evaluation:
1. **Package and module names**: `gantry_dynamic.{config, data, model, controller, training, closed_loop_report,
   obc_gantry}` and `model_augmentation.fit_systems.{closed_loop, obc, blocks}` with the imported names
   `RunConfig, record_trim, load_datasets, compute_normalization, _load_record, servo_sigma, traj_dir,
   _record_at_fs_new, _load_u, _resample_u, record_files, build_model, build_closed_loop (keywords train_files,
   val_files, val_data, verbose, e_train, e_val), build_controller_bank, y_op_for, load_checkpoint,
   _without_learned, _true_free, build_reference_set (cfg, norm, nx_ann=, files=, verbose=), make_float64_block,
   make_basis_builder (blk, ref, row_ix, chunk, device=, space=), closed_loop_free_run_rms_batch (return_pred=),
   build_stacked_sensitivity, Static_ANN_Block`.
2. **Config round trip**: `eval_greybox.run_config` rebuilds `RunConfig` from the upper-cased field names in
   config.json; every clean field has its key there (`ARM, START, N_A, PS2, NOISE, DEVICE, COMPILE_MODE,
   TRUNCATE_S, EPOCHS, N_ITS, ITS_PER_VAL, LBFGS_MAX_ITER, LBFGS_INNER_ITER, LBFGS_WINDOWS, PS2_SMOKE`).
   `discover_runs` reads `TRUNCATE_S, N_ITS, run_id, OBC, NX_ANN, COMBO_INIT_DETUNE, PS2`; `checkpoint_status`
   reads `EPOCHS`; `run_config` reads `NA_NB, NF`; `make_results` reads `OBC, COMBO_INIT_DETUNE` from the cached
   config. All written. Config attributes read by the evaluation: `seed, nx_ann, nx_phys, obc, obc_ref_chunk,
   ann_route_ix, controller_gain, dtype_pt, ts_new, hp`: all provided (section 3).
3. **Files**: run folder `OUT_ROOT/{mode}_data_lsq/<run_id>/` with `config.json`, `gantry_ckpt_<rid>.pt` (keys
   `hfn`, `encoder`, `optimizer` unless dropped after an accepted polish), `gantry_ckpt_<rid>.npz` with the twelve
   mainline keys (`done_epochs, bestfit, bestfit_selected, elapsed, hp, orig_run_id, lbfgs_outcome,
   optimizer_state_dropped, start_phase, resumed_from, resumed_at_epoch, ps2_summary`; `resumed_from = ''`,
   `resumed_at_epoch = -1`, `start_phase = 'adam'`), the `_pre_lbfgs` pair (`done_epochs, bestfit, hp,
   orig_run_id`), `val_hist.npz` (mainline keys). `bestfit` is the final event's score, read by the evaluation as
   `ref_score`. deepSI's `<Class>_<rid>_best.pth` stays loadable by `resolve_checkpoint` for an unfinished run. No
   `gantry_results_<rid>.npz`: the report is excluded, and `eval_greybox.py:178-179` already skips its check when
   the file is absent. `lbfgs_outcome` keeps the keys `ran, accepted, reason, n_iter, max_iter, n_closure,
   func_evals, wall_s, loss_first, loss_last, val_before, val_after, n_windows_used, n_windows_total,
   effective_stride, deterministic, params_polished` (no reader outside the run).
4. **state_dict names and shapes**: unchanged, because modules, attribute names, buffer registrations and block
   order (physics 0, output 1, ANN 2) are unchanged. For n_a = 2 (221 smoke checkpoint, C6):
   `connected_blocks.0.{free_params, mh, alpha, beta, gamma_, N0, N1, N2, Ax, Bw, Bu, A_combined, K_mat, C_mat,
   P_mat, std_x, std_x_1d, std_u, x_mean, u_mean, params_init, Lb, combo_init, Lambda_combo, coordinate_version,
   m_diff_eta0, m_diff_scale}`, `connected_blocks.1.{C, D}`, `connected_blocks.2.net.net.{0,2,4}.{weight, bias}`,
   `obc_correction.{E, vbar, theta_frozen}` (obc arm), `encoder.{Wb_psi_y, Wb_psi_u, Wa_psi_y, Wa_psi_u,
   net.{0,2,4}.{weight, bias}}`. Every buffer stays even where the trained block does not read it (the nominal LFR
   buffers): they are keys that `load_checkpoint` checks exactly. `out_gate` stays a non-persistent buffer.
   Layer 1 checks key sets and shapes against mainline for all 16 model cases.
5. **Objects and behaviour**: `fs.hfn.connected_blocks`, `fs.encoder`, `fs.na`, `fs.nb`, `na_right`, `nb_right`,
   `fs.norm` (`transform`, `u0`, `ustd`, `y0`, `ystd`), `fs.simulator`; `hfn.obc_correction.{theta_frozen, vbar,
   space}` (`space = 'affine'` stays an attribute although it is constant); physics block `free_params,
   params_init, combo_init, Lb, PARAM_NAMES, COMBO_NAMES, combos_of, pass_mats()` (the ten-tuple), the frozen-Y
   attributes `Y_op, N_op, d_op` that `frozen_y` sets, `estimation_parameters, combinations_from_free,
   free_from_combinations, transition_from_free`; reference set `Z_step, Z_field, n, record_ix, sample_ix,
   record_names, x_phys_next_rec`; `Norm` and `DataBundle` fields of section 2. `load_checkpoint(fs, path,
   joint_estimation=True, start_phase='lbfgs')` restores the saved OBC buffers and nothing re-syncs them;
   `y_op_for` raises `KeyError` for a non-K1 record (the evaluation's skip); `Interconnect.forward` initialises
   itself lazily (`eval_frf` calls `hfn(x, u)` before any rollout).

**Evaluation edits needed when the evaluation moves to the clean code** (not made now; the handoff leaves the
evaluation scripts out of scope):
(a) `eval_common.py:19-21` puts `Thesis-writeup/Code/Training` on the path instead of the repository root and
`scripts/gantry`, keeping `scripts/gantry/thesis-results` (`inspect_checkpoint.clamped`, `added_poles`), and
`THESIS_RUNS_DIR` points at the clean family folder. (b) `eval_greybox.run_config` drops
`na_nb_override` and `nf_override` from its `opts.update` (open choice 8). (c) `eval_blackbox` stays on the mainline
root (the black box R8 is out of scope and its `cfg_for` needs the mainline `RunConfig`); since one process must not
hold both roots, the root becomes a per-script choice in `eval_common`.

## 6. Equivalence plan

### 6.1 Hazards (what a faithful rewrite must reproduce)

Agreed by both inventories (Part C), made specific to the mapping:
- **H1 global draw order.** After the second seed (EP:409-410) the global torch stream is consumed by, in order:
  the live g_aug MLP (three `nn.Linear` at float32, input to output, in `Static_ANN_Block.__init__`, also for
  n_a = 0); `Wb_psi_y` then `Wb_psi_u` `uniform_` at float64 (values later overwritten by the least-squares start,
  the draws are not); one joint Xavier draw for W^a at float64; the encoder net's three `nn.Linear` at float32 then
  cast; on PS2 rows the decoder's two `nn.Linear` at the first `after_step`. Nothing may draw from a global stream
  between the second seed and the first update, and the first seed before the data (EP:390-391) stays. The detuning
  signs come from an independent `default_rng(1)` (section 3).
- **H2 dropped code draws nothing global.** The nf probe used `default_rng(cfg.seed)` (training.py:402) and the
  post-run diagnostics run after the final checkpoint; dropping them moves no stream (C1 of both inventories).
- **H3 Adam batch order.** `Dtype_DataLoader` keeps deepSI's `default_rng(0)` and the in-place `shuffle(ids)` of
  `My_Simple_DataLoader.__iter__` at every pass, consumed through `loop_dataset`, full batches only,
  `N_batch_updates_per_epoch = N // 512`, independent of the run seed. PS2 uses its own seeded generators (spec.seed,
  +321, +1000, a CPU `torch.Generator`); the polish subset is `default_rng(seed).choice` then sorted.
- **H4 step order.** Interconnect sums through the connection matrices (`expansion_matrix`, `selection_matrix` are
  real matmuls, `+=` in `order_output_signal_computation`), then `xp - obc_correction(x, u, obc_pass)`; the closed
  loop computes `output_only(x)`, then `baddbmm(bmm(M_error, e), M_state, xc)`, then `hfn(x, u + u_fb)`, with the
  normalisation folded into the bank matrices.
- **H5 reductions and windows.** Loss `((y_pred - yfuture) * w[:, None, :]).square().mean()`; weights and sigmas in
  float64 with `np.std` (ddof 0) from k0; windows from `fixed_System_data.to_hist_future_data` per record in manifest
  order with stride 10, `ctrl_ix` and `loss_w` aligned per record; validation grouped by post-k0 length, score
  `mean(per_channel / val_sigma_e)`.
- **H6 float32 materialisation, reproduced, not fixed.** `gantry_ss` tensors are float32 (`_D`); `to_tensor` is
  `torch.Tensor(ndarray)` (float32) for `std_x, std_u, x_mean, u_mean, C, D`; nominal LFR buffers are built in
  float32 before the cast; `compute_normalization` inverts `P.numpy().T` in float32 and casts; the OBC reference set
  and the pre-fit cast `P` to float64 before inverting; `nn.Linear` defaults are drawn at float32 and cast.
- **H7 polish numerics.** Parameter order = deepSI `parameters_with_names` (it walks `dir(self)`, so `encoder` then
  `hfn`); one parameter group; the closure returns `torch.as_tensor(float(loss))`, a default-dtype (float32) scalar,
  so the strong-Wolfe search compares float32-rounded values; `zero_grad(set_to_none=False)`; LBFGS lr 1,
  `max_iter = inner_iter`, history 50, the two tolerances; blocks until the cap or `done < inner_iter`.
- **H8 state across the end of fit().** `checkpoint_load_system('_best')` replaces `__dict__` with an unpickled copy:
  the simulator comes back without its compiled pair (so the polish is eager), the optimizer is the pickled one, the
  OBC lifecycle and PS2 phase are carried across by the stash; the loss history is restored after; then the basis is
  rebuilt at the restored parameters and `_best` re-saved. The clean code keeps this mechanism (no in-memory
  best-state shortcut).
- **H9 OBC numerics.** Float64 deep copy of the physics block; `jacfwd` stack in chunks of 16384, sample-major then
  physical row; column scaling; SVD rank cutoff `max(rows, cols) * eps * s_max`; coefficient `V S^-1 U^T F` in float64;
  refresh at `floor(epoch_counter + 1e-6)`; the pre-fit builds the first basis; PS2 re-syncs the frozen coefficient
  between updates; rank-loss policy (retain once, abort on two).
- **H10 least-squares encoder start.** Centred, column-scaled regressors, `gelsd`, `rcond = eps * max(N, m)`,
  records grouped by length in dict insertion order, stride-10 windows. The solution is rank-truncated (the 2 s
  smoke reported rank 141 of 180), so any change in the window set or order can move the rank: the highest-risk
  single item.
- **H11 libraries.** deepSI 0.3.29, torch 2.5.1, numpy 2.0.1, scipy 1.17.0 (local env and
  `environment.server-train.yml`). deepSI's `checkpoint_load_system` calls `torch.load(file)` without
  `weights_only`, which torch 2.6 and later default to True: the pin is part of the method.

### 6.2 Harness environment

- One conda env (H11), CPU, `device='cpu'`, `compile_mode=None`, float64 (the method). `torch.set_num_threads(1)`
  and `OMP_NUM_THREADS = MKL_NUM_THREADS = 1` in every harness process, because multi-threaded BLAS and OpenMP
  reductions may block differently from run to run, and bitwise comparison needs the same operation order.
- Mainline and the clean code run in separate processes (same package names). Mainline is imported unmodified:
  the runner sets the row's `THESIS_*` variables plus `THESIS_SMOKE=1` and `THESIS_DEVICE=cpu`, imports EP, replaces
  `EP.CFG` with the harness budget and calls `EP.main()` (the method of the step-1 trace runner; the smoke variable
  is needed for the PS2 smoke spec, training.py:873). deepSI checkpoints go to a scratch folder per run (LOCALAPPDATA
  or HOME redirect).
- Harness budget, identical on both sides: `truncate_s = 0.6`, `ps2_smoke = True`, and for layer 3 `epochs = 3`,
  `n_its = None`, `its_per_val = 5`, `lbfgs_max_iter = 2`, `lbfgs_inner_iter = 1`, `lbfgs_windows = 256`.
  Why 0.6 s: the shortest truncation at which both noise entries have a positive servo sigma (noise-free TR-T and
  VA-T stand still until 0.495 s, A.0) and every window type is valid. It gives 198 windows per record (first window
  end 430, stride 10, 2,400 samples), 3,564 in total, so 6 Adam updates per epoch and 18 in three epochs, with OBC
  refreshes at the two epoch boundaries inside the run, validations at update 0 (the initial one), 5, 10 and 15, and three
  unvalidated updates at the end (as in production).
- Seeds: every run uses the spec seed 1 (both seeding points, EP:390-391 and 409-410, the PS2 generators and the
  polish subset follow from it); the layer 1-2 batch indices come from `default_rng(0)`; no other seed is introduced.
- Golden outputs: `<repo>/simulations/clean-golden/L<n>/<row>-<noise>/`, each with `meta.json` (SHA256 of the 19
  copied mainline files, library versions, thread settings, `git_provenance()`, the budget). `compare.py` refuses a
  golden whose mainline hashes differ from the current files.

### 6.3 Layers

| Layer | Cases | Golden content (mainline) | Comparison | Tolerance and reason |
|-|-|-|-|-|
| 1 construction and forward pass | the 8 distinct models (221, 222, 223 = 225 for this layer, 224, 226, 227, 229, 230) x lowpass and noisefree = 16 | data per split (u, y, e, file lists), `Norm` fields, servo sigmas, loss weights; training window arrays (uh, yh, uf, yf, ctrl_ix, loss_w); controller bank matrices; OBC reference set fields; state_dict right after `build_model` (H1) with key and shape lists; after the pre-fit: state_dict, LS rank and singular values, first OBC basis (U, S, column scale, vbar), theta_frozen; then, with the mainline post-pre-fit state_dict loaded into the clean model: encoder x0 on a fixed 64-window batch (`default_rng(0).choice(N, 64, replace=False)`, sorted), `pass_mats()`, OBC coefficient and tangent pair, the 400-step closed-loop prediction, `cal_validation_error` and `last_per_channel` | `np.array_equal` / `torch.equal` per array; equal key sets and shapes | Bitwise. Same statements in the same order on the same inputs, single-threaded float64: any difference marks a changed statement. Only for a difference traced to a named cause and accepted by the judge: max abs difference over max abs reference at most 1e-12 per tensor (eps 1.1e-16 times the about 1e4 terms of the longest reduction: 400-step rollout sums, about 4e4-row OBC stack) |
| 2 loss and gradients | the same 16 cases, same batch and state | `fit_sys.loss(uh, yh, uf, yf, nf=400, stride=10, ctrl_ix=..., loss_w=...)` and, after `backward()`, the gradient of every named parameter in `parameters_with_names` order (for obc through theta, H9) | per tensor | Bitwise, same reason and the same documented 1e-12 fallback |
| 3 short seeded training | 221-lowpass (obc, detuned, n_a 2, PS2), 229-noisefree (obc, n_a 0, plain), 224-noisefree (u, detuned, 2, PS2), 226-lowpass (u, correct, 0, plain), 227-lowpass (u, correct, 8, PS2) | the run folder of `main()` with the layer-3 budget: config.json, `_pre_lbfgs` pair, final pair, val_hist.npz, the PS2 summary, `lbfgs_outcome` | state_dicts of both pairs (hfn, encoder, optimizer when present); `.npz` scalars and strings except `elapsed` and timings; the common `lbfgs_outcome` fields except `wall_s`; val_hist arrays; config.json on the shared keys except GIT, run_id, data_dir and NF_PROBE_PRINT | Bitwise, no fallback. Deterministic CPU with the same draws (H1 to H3); the PS2 S2 Adam (lr 1e-3, up to 100 steps) and the LS rank decision (H10) amplify any earlier difference, so a mismatch is localised (bisect with `n_its` 1, 2, 4, 8 on both sides, then layers 1-2), never toleranced |
| 3b polish rejection | 221-lowpass and 226-lowpass, from the golden layer-3 `_pre_lbfgs` state loaded into a freshly built model on each side | `lbfgs_polish(..., val_before=0.0)`, which forces the reject path (restore and `_resync`) | outcome fields and parameters (must equal the start state) | Bitwise; restore is a copy |
| 4 final checkpoints | the 9 final rows on lowpass (no noise-free final run exists): jobs 89792 (221), 89796 (222), 89797 (223), 89793 (224), 89798 (225), 89799 (226), 89800 (227), 89794 (229), 89795 (230), from the thesis-results-4 logs | mainline evaluation `eval_greybox.py R1 R2 R3 R4 R5 R6 R7 R13 R14` with the split option `train,val,test`, with `THESIS_EVAL=final` on CPU, rows `model` for all and `physics, base_start, base_true` for R1, R4, R6, R13 (covers `_without_learned`, `_true_free`); caches `preds_<R>_<split>.npz` | the same evaluation on the clean code (section 5 edits): per record from k0, `y`, `sigma_e` and every system's prediction; derived RMS and NRMS_e per split, record and axis; `params_learned` | Bitwise against mainline (same weights, same eager scoring code). Both sides must also pass the evaluation's own check of the validation score against the checkpoint's `bestfit` at relative 1e-3 (`eval_greybox.py:196-199`, existing acceptance for CPU eager against GPU compiled; reused, not introduced) |

Rows chosen so that every axis value occurs in layer 3: arm obc (221, 229) and u (224, 226, 227); n_a 0 (229, 226)
and > 0 (221, 224, 227); PS2 on (221, 224, 227) and off (229, 226); lowpass (221, 226, 227) and noisefree (229,
224); detuned (221, 229, 224) and correct (226, 227).

**Cost.** No harness timing exists yet. The only measured reference is the step-1 trace (`inventory-claude.md` A.0):
218 to 447 s per u row and 826 to 1249 s per obc row at 0.3 to 0.6 s truncation and 4 updates, post-run
diagnostics included. Layer 3 has 18 updates, so obc rows will take longer; the first case of each layer is timed
(and its memory measured) before the rest are launched, and all runs go to the background with streamed output.
Layer 4 is best run where the checkpoints are (server CPU), at about 4 min per evaluation run
(`Thesis-writeup/Code/Figures/README.md`).

### 6.4 Not covered by the layers

GPU and compiled paths (compiled rollout, `cudagraph_mark_step_begin`, device moves, the CUDA OOM path of the polish),
the OBC rank-loss branches, the polish exception path, the NaN break, and the PS2 plateau and fallback with the real
thresholds. Covered instead by (a) a GPU smoke of the clean code on the server (word `smoke`, R1 and R6, lowpass):
it must finish, log "training rollout COMPILED", log no "skipping cudagraphs", and write the checkpoint pair; next to
a mainline smoke on the same node, the initial validation score and the four training losses must agree within
1e-9 relative (identical graphs on identical inputs; CUDA kernels with atomic accumulation may reorder sums); and
(b) review: every one of these branches is copied verbatim (section 2).

### 6.5 Inputs from the server

- The nine final run folders under `simulations/gantry_subnet/thesis_taf_lowpass_noise_data_lsq/<job>/` (config.json,
  both checkpoint pairs, `gantry_results_<job>.npz` for the evaluation's report check). They are not local.
- The `[lbfgs]` lines of 221 and 222 (the local logs end before the polish). They decide only which saved file is
  the layer-4 reference for those two rows, under D-274 change 3 as amended: polish ACCEPTED with val_after below
  the selected checkpoint's `bestfit`: `gantry_ckpt_<job>` (same under both rules); ACCEPTED with val_after at or
  above `bestfit` but below the rebuilt score (accepted only under the D-253 rule): the D-274 final model is the
  rebuilt one, saved as `gantry_ckpt_<job>_pre_lbfgs`, and that row's thesis numbers move by about 2e-6 relative;
  ROLLED BACK or ABORTED: `gantry_ckpt_<job>`. Known from the local logs: 229 accepted (2.925875e-01 below the
  checkpoint's 2.926736e-01, D-274), 224 and 227 rolled back, 223, 225, 226 and 230 aborted by CUDA OOM in the eager
  determinism probe with weights unchanged.

## 7. Open choices

Each lists the choice taken, the alternatives, and the reason.

1. **Folder.** Taken: `Thesis-writeup/Code/Training/`. Alternatives: repository root `gantry_clean/`;
   `scripts/gantry/clean/`. Reason: beside the thesis Data, Evaluation, Figures and Server code, data root at a fixed
   relative path on PC and server.
2. **deepSI.** Taken: installed dependency, pinned 0.3.29. Alternative: vendor the used subset (System_data and
   System_data_list, the windowing, `SS_encoder_general` init and `make_training_data`, `System_torch`
   `parameters_with_names`, `init_optimizer`, `_check_and_refresh_optimizer_if_needed`, checkpoint save and load,
   `My_Simple_DataLoader` with its iterator, `loop_dataset`; a few hundred lines). Reason: H3 and H8 live in deepSI,
   the evaluation and the black box import deepSI anyway; vendoring can follow later with this harness as its test.
3. **How far "minimal" goes inside Jan's classes.** Taken: keep Jan's classes, names and files; delete unused
   methods and option branches; copy small helpers whole; keep `SSE_Interconnect_Composed` as our subclass.
   Alternatives: merge `SSE_Interconnect` and `SSE_Interconnect_Composed` into one class and trim the helpers too
   (fewer lines, blurs the Jan/ours markers, no 1:1 diff); keep Jan's classes intact (most lines, least risk).
   Reason: markers keep meaning and every file diffs against its mainline original.
4. **Launch.** Taken: CLI `<preset> <noise> [cpu] [smoke]`. Alternative: `THESIS_*` environment
   variables as now. Reason: the run is visible in the command line and the log, and nothing is inherited through
   the launcher's export-all setting (Codex C3 point 16).
5. **Preset names.** Taken: R labels (R1 to R7, R13, R14). Alternatives: runs.tsv ids 221 to 230; both as aliases.
   Reason: the thesis and the evaluation use the R labels; the id is in the preset table.
6. **RunConfig fields.** Taken: five axes plus ten run-environment fields with production defaults. Alternatives:
   axes plus one `smoke` flag (the harness could not set 18 updates and 4 validations without patching); a separate
   budget object (the evaluation's flat field-name round trip and the flat config.json prefer flat fields).
7. **config.json.** Taken: every mainline key with the run's value, plus ARM, START, N_A, NOISE, PS2_SMOKE, CODE.
   Alternative: only the keys the evaluation reads (about fifteen). Reason: complete provenance; the layer-3 check is
   a plain equality on shared keys.
8. **Evaluation round trip.** Taken: when the evaluation switches, `eval_greybox.run_config` drops two keywords.
   Alternatives: accept inert `nf_override` and `na_nb_override` fields in the clean config (no evaluation edit, two
   meaningless fields); a `RunConfig.from_run_dir()` in the clean code that the evaluation calls.
9. **Validation history.** Taken: `ValHistory` (record, hist_arrays, the 'adam' probe) and `val_hist.npz`.
   Alternative: drop it and compute the final `bestfit` from three scalars (selected score, rebuilt score, polish
   val_after). Reason: about 30 lines; the final event follows the same rows as mainline; it is the only record of
   the validation curve now that the run report is excluded.
10. **Interface helpers.** Taken: `gantry_dynamic/closed_loop_report.py` with `_without_learned` and `_true_free`
    only. Alternative: move them into `model.py` and change three evaluation imports. Reason: zero import changes.
11. **Truth fields.** Taken: not read (data loader and OBC reference set). Alternative: keep reading them for later
    diagnostics. Reason: no method use (D-255), less I/O, less code.
12. **Resume and polish-only.** Taken: dropped (`RESUME_CHECKPOINT`, `start_phase='lbfgs'`, `resolve_resume_hp`,
    `load_optimizer_moments`, the polish warm-up and compiled try). Consequence: the polish-only reruns that the D-274
    amendment allows for 223, 225, 226 and 230 run on mainline. Alternative: keep a polish-only entry (brings back
    those functions). Reason: not in the spec.
13. **Guards.** Taken: rule G (section 0). Alternative: keep every mainline guard (more lines, no numeric
    difference). Reason: a guard that re-checks the code's own constants cannot fire; the data and numerics guards
    stay.
14. **Per-call device self-heals** (`Interconnect.forward` 137-141, `output_only` 330-333,
    `Static_ANN_Block.forward` 76-77). Taken: removed; `Interconnect._apply` and the buffer mechanism keep devices and
    dtypes in sync. Alternative: keep them (about 12 defensive lines). Covered by the GPU smoke.
15. **Output root.** Taken: `<repo>/simulations/gantry_clean/`. Alternatives: the mainline family folders (the
    evaluation's `discover_runs` raises on two runs of one design); inside the folder (mixes outputs into code).
16. **Polish simplification.** Taken: eager only after fit, one determinism probe, no warm-up, no chunk loop, no
    meters. Each removal is exact or unreachable after fit (H7, H8). Alternative: copy the mainline polish whole
    (needed only for a polish-only resume, see 12).
17. **param_loss.** Taken: removed with the loss sweep (it adds the Python float 0.0, which leaves the loss and its
    gradient bit for bit unchanged; `Lambda_combo` stays a buffer, a state_dict key). Alternative: keep both (about 8
    lines).
18. **Harness tolerances.** Taken: bitwise for every layer, with a documented 1e-12 fallback for layers 1-2 only.
    Alternative: relative tolerances everywhere. Reason: a tolerance would let a changed statement pass unnoticed.
19. **Harness location.** Taken: `Thesis-writeup/Code/Verification/clean-code/`, golden outputs in
    `simulations/clean-golden/`. Alternative: `tasks/clean-code/harness/`. Reason: Verification exists (empty) for
    exactly this; `tasks/` holds documents.
20. **NF_PROBE_PRINT.** Taken: written `False`. Alternative: the mainline value `True` (would claim prints that
    do not happen).
21. **GantryReferenceSet fields.** Taken: six fields kept (training and evaluation readers). Alternative: keep all
    twelve so the class matches mainline field for field. Reason: the other six hold truth or bookkeeping that no
    kept code and no evaluation script reads.
