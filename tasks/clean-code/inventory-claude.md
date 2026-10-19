# Clean-code inventory (Claude, dynamic trace), 2026-10-19

Step 1 of 4 of `tasks/handoffs/2026-10-19-clean-code-inventory-claude.md`. Spec: runs.tsv rows 221 to 230 (no 228),
`THESIS_NOISE` lowpass, plus 221 and 224 noisefree. Code locations: `scripts/gantry/gantry_interconnect_dynamic.py` (EP),
`scripts/gantry/gantry_dynamic/` (GD), `scripts/gantry/common/` (CM), `model_augmentation/` (MA).

Evidence tags: **trace** = executed (or not) in the coverage runs below; **reading** = classified by reading, with the
reason (real-run, GPU or compile code that a CPU smoke cannot reach, or code the trace cannot distinguish).

## 0. How the trace was made

- `coverage 7.16.2` (installed 2026-10-19 with Dirk's approval), branch coverage, rc file and one data file per row in
  `tasks/clean-code/coverage/` (`.coverage.<row>-<noise>`), combined report `combined-report.txt` and `combined.json`.
- Each row ran with its runs.tsv `THESIS_*` values plus `THESIS_SMOKE=1`, `THESIS_DEVICE=cpu`, through a trace-only
  runner (scratchpad `trace_runner.py`) that imports EP, sets `CFG.truncate_s = 0.3` and calls `EP.main()`. No repo file
  was edited. Why 0.3 s instead of the smoke's 2 s (agreed with Dirk 2026-10-19): coverage records which lines run,
  not how often; 0.3 s keeps every window type valid (30-sample encoder window, 400-step training window, PS2 H = 30)
  and gives about 1400 training windows, i.e. 2 Adam updates per epoch, so the 4 smoke updates cross two epoch
  boundaries and the per-epoch OBC basis refresh executes (it does not at 2 s). The first 2 s trace of row 221
  (3305 s) is kept as `trunc2s-221-lowpass.coverage` for comparison.
- deepSI checkpoints went to the scratchpad (`LOCALAPPDATA` redirect); run folders went to
  `simulations/gantry_subnet/thesis_taf_<mode>_data_lsq/covsmoke*-<row>-<noise>/` (Dirk 2026-10-19). Evaluation's
  `discover_runs` ignores them (TRUNCATE_S and N_ITS set, run_id not numeric). Dirk deletes them when done.

Runtime per row: see section A.0.

### Smoke versus real run (handoff section 5)

The smoke path differs from the real run only in these places (all read in EP:368-374, TRAIN:867-872, MODEL:501):

| Item | Smoke | Real run | Code reached only in the real run |
|-|-|-|-|
| `truncate_s` | 2.0 (trace: 0.3) | None | `DATA:198-199` is skipped in the real run; everything else in loading is the same |
| `compile_mode` | None | `reduce-overhead` | `CTRL:329-341` (torch.compile), `CL:584-591, 600-604` (compiled branch, `cudagraph_mark_step_begin`, clones) |
| `device` | cpu | cuda | `MODEL:343-344, 348-349, 483-484, 518-519`, `TRAIN:776-782, 817-818`, `EVAL:111` (GD/evaluation.py), `IC:933-934, 998-999` (batch `.cuda()`) |
| `n_its`, `its_per_val` | 4, 4 | None (33400 updates), 1300 | same lines, other values; in the real run `its_per_val` comes from `cfg.its_per_val` (`MODEL:501`) |
| PS2 spec | `TRAIN:869-872` override | `PS2Spec(seed)` defaults | the S1 plateau/screen logic with real thresholds (`PS:108-121`, plateau branch), S2 plateau break (`PS:246-247`) |
| polish | 2 iterations, inner 1, 256 windows | 100, inner 20, 16384 | multi-iteration blocks, tolerance exits inside torch LBFGS, `lbfgs_chunk` path stays None at nf 0.1 |

## Part A. Required-code inventory

### A.1 `RunConfig` fields (GD `config.py:60-367`)

Values are those of the real run (EP `_THESIS_FIXED` and the base `CFG`); "varied" means the spec rows differ in it.
"read, inert" = read on the required path but its value cannot change a number in any spec row (still recorded in
`config.json`, and some still shape a checkpoint buffer).

| Field | Spec | Value(s) | Read on the required path at | Note |
|-|-|-|-|-|
| mode | varied | `thesis_taf_lowpass_noise`, `thesis_taf_noisefree` | DATA:141-144, 175, CFG:48, 56 | selects folder, y rule (anti-alias only for lowpass) and trim |
| encoder_init | constant | `data_lsq` | MODEL:42-51, 210-261, 300, EP:444, CFG:605 | also names the run folder |
| ann_activation | constant | `tanh` | MODEL:136 | |
| ann_arch | constant | `mlp` | MODEL:139 | |
| g_init | constant | `live` | MODEL:138 | |
| wa_encoder_init | constant | `xavier_uniform_gain1` | MODEL:259 -> PRE:428 | |
| joint_estimation | constant | True | MODEL:81, TRAIN:849 (resume only) | |
| physics_parameterization | constant | `reduced` | MODEL:96 | |
| param_prior | constant | False | MODEL:94 (`flag_loss_reg`) | `param_loss` returns 0 |
| param_rmse_baseline | constant | 0.01 | MODEL:90 | read, inert in the loss; sets the `Lambda_combo` buffer, which is in the checkpoint |
| param_init_detune | constant | None | only MODEL:106 (raw branch) and `__post_init__` | never read on the path |
| combo_init_detune | varied | 10-factor list (detuned) or None (correct) | MODEL:98-99 | |
| snr | constant | None | DATA:269-284, OBCG:114 | guards only |
| seed | constant | 1 | EP:390-391, 409-410, TRAIN:402, 868, 774 | detuning signs are drawn in EP:336 from THESIS_SEED before the config exists |
| closed_loop | constant | True | EP:432 | |
| controller_gain | constant | False | only `closed_loop_report.py` (post) and the evaluation | never read on the path; `build_closed_loop` uses the default `use_gain=False` |
| fs_orig | constant | 20000 | CFG:547 (`d`), DATA:177-181 | |
| fs_new | constant | 4000 | via `fs_new_hz`, `d`, `ts_new` (CFG:541-551) | |
| stride | constant | 10 | MODEL:357, model `train_model` loss_kwargs (MODEL:503), TRAIN:795 | |
| truncate_s | constant | None | DATA:198 | smoke only |
| use_f64 | constant | True | via `dtype_np`, `dtype_pt` (CFG:553-559) | |
| save_flag | constant | True | EP:403, 450, 466 | |
| nf_probe_print | constant | True | TRAIN:676-679 | diagnostic printing only |
| device | constant | `cuda` | MODEL:286, 343, 483, 518, TRAIN:776, 817, CTRL (compile) | CPU in the smoke |
| checkpoint_chunk | constant | 0 | CTRL:355 -> CL:291 | off |
| compile_mode | constant | `reduce-overhead` | CTRL:329 | None in the smoke |
| nx_ann | varied | 0, 2, 8 | via `hp['NX_ANN']` (MODEL:71), TRAIN:421, 873, OBCG:379 | |
| ann_route_ix | varied | `range(6 + nx_ann)` | MODEL:73, OBCG:382 | always every state, so `expansion_matrix` is the identity |
| n_nodes_per_layer | constant | 16 | via `hp` (MODEL:145, 198, 233, 255), TRAIN:874 | also the PS2 decoder width |
| n_hidden_layers | constant | 2 | via `hp` | |
| up_sample | constant | 1 | via `hp` (MODEL:90) | |
| batch_size | constant | 512 | via `hp` (MODEL:487) | |
| lr | constant | 1e-5 | via `hp` (MODEL:294, 504) | |
| adam_eps | constant | 1e-16 | MODEL:294-299, 471-476, 504 | |
| epochs | constant | 200 | via `hp` (TRAIN:854) | |
| n_its | constant | None | MODEL:490, 501 | 4 in the smoke |
| its_per_val | constant | 1300 | MODEL:501 | |
| nf_seconds | constant | 0.1 | via `nf` (CFG:561-573) | |
| nf_override | constant | None | CFG:571 | |
| na_nb_override | constant | 30 | CFG:577 | |
| loss_weighting | constant | `servo` | CTRL:350 | |
| validation_measure | constant | `sim-NRMSe` | EP:454, TRAIN:858, 865, 788, 809 | |
| lbfgs | constant | True | TRAIN:764, 928 | |
| lbfgs_max_iter, lbfgs_inner_iter, lbfgs_history, lbfgs_windows | constant | 100, 20, 50, 16384 | TRAIN:769-775 | smoke: 2, 1, 50, 256 |
| lbfgs_chunk | constant | None | TRAIN:774 | EP:364 sets it only for nf > 0.1 s |
| lbfgs_tol_grad, lbfgs_tol_change | constant | 2.6e-5, 2.6e-7 | TRAIN:772 | |
| lbfgs_freeze | constant | () | TRAIN:773 | |
| lbfgs_time_budget_s | constant | None | TRAIN:774 | |
| lbfgs_meter_rel_tol | constant | 1e-3 | TRAIN:773 -> LB:782 | read, inert (`meters_gate=False`) |
| lbfgs_meters_gate | constant | False | TRAIN:775 | |
| start_phase | constant | `adam` | TRAIN:847-881, 786, 1024 | |
| ps2 | varied | True, False | TRAIN:866 | |
| orth | constant | False | MODEL:312 | gate only |
| orth_beta, orth_point_stride, orth_rank_tol | constant | 4.66e-4, 100, 1e-12 | only CM `orth_penalty.py` | never read on the path |
| obc | varied | True (obc arm), False (u arm) | MODEL:321, TRAIN:787 | |
| obc_space | constant | `affine` | OBCG:383, 388, 394-399 | |
| obc_ref_successor | constant | True | OBCG:135, 139 | |
| obc_refresh | constant | `epoch` | OBCG:395 | |
| obc_ref_chunk | constant | 16384 | OBCG:393 | memory only |
| obc_rank_rtol | constant | None | OBCG:395 | |
| static_pass_buffers | constant | False | MODEL:113, OBCG:385 | gate only |
| nx_phys, nu, ny | constant | 6, 3, 3 | throughout | |

Derived properties needed: `fs_new_hz`, `d`, `ts_new`, `dtype_np`, `dtype_pt`, `nf`, `na_nb`, `hp` (its key names and
order are the `.npz` `hp` contract). Module-level constants needed: `REPO_ROOT`, `ANTI_ALIAS_MODES`, `AA_ORDER`,
`AA_FC_HZ`, `TRIM_MODES`, `AA_TRIM_S`, `y_decimation`, `record_trim`; `save_dir`, `config_json_dict` (the evaluation's
config round trip, C6). `git_provenance` and `_assert_json_coverage` do not change numbers.

Environment switches outside `RunConfig` that the spec never sets (all not needed): `DATASET_AB_MODE`,
`DATASET_AB_SEED`, `LBFGS_AB_FREEZE`, `OBC_ARM` (EP:218-288), `RESUME_CHECKPOINT` (EP:399, resume paths in TRAIN),
`ANN_REZERO_GATE` (MODEL:157-164), `THESIS_NITS` (EP:366-367).

### A.0 Trace runs

| Row | Arm | Start | n_a | PS2 | Noise | truncate_s | Seconds | rc |
|-|-|-|-|-|-|-|-|-|
| 221 | obc | detuned | 2 | 1 | lowpass | 0.3 | 985 | 0 |
| 224 | u | detuned | 2 | 1 | lowpass | 0.3 | 218 | 0 |
| 222 | obc | correct | 2 | 1 | lowpass | 0.3 | 826 | 0 |
| 229 | obc | detuned | 0 | 0 | lowpass | 0.3 | 1177 | 0 |
| 223 | u | correct | 2 | 0 | lowpass | 0.3 | 308 | 0 |
| 225 | u | correct | 2 | 1 | lowpass | 0.3 | 323 | 0 |
| 226 | u | correct | 0 | 0 | lowpass | 0.3 | 316 | 0 |
| 227 | u | correct | 8 | 1 | lowpass | 0.3 | 447 | 0 |
| 230 | u | detuned | 0 | 0 | lowpass | 0.3 | 368 | 0 |
| 221 | obc | detuned | 2 | 1 | noisefree | 0.6 | 1249 | 0 |
| 224 | u | detuned | 2 | 1 | noisefree | 0.6 | 304 | 0 |

Total about 1.7 h. The OBC arm dominates the cost (the coefficient over the whole reference set at every objective
plus the correction at every step). The noise-free rows needed 0.6 s: at 0.3 s they stop in `servo_sigma`
(DATA:232-233) because the TR-T and VA-T records are at standstill until sample 1980 (0.495 s) and the noise-free
servo error is exactly zero there. That is a truncation artefact; full records have a positive sigma_e. With 0.6 s a
noise-free epoch has 6 updates, so the epoch refresh (OBC:507-508) is not reached in those two rows; it is reached in
the lowpass OBC rows, and the refresh code does not depend on the noise mode.

**Lowpass versus noise-free** (same row, pipeline context): the only lines that differ are the y rule, DATA:177,
180-182 and CFG:49 (lowpass) against DATA:176 (noise-free), plus the epoch refresh and a timing print noted above.
Nothing else in the four locations depends on `THESIS_NOISE`.

**Pipeline versus post** (coverage contexts set by the scratchpad `trace_runner.py`): `pipeline` = from import to the
return of `train_model_with_diagnostics`, which writes the final `gantry_ckpt_<rid>.pt/.npz` (TRAIN:960-1005);
`post` = `evaluate_and_save`, `state_recovery_diagnostic`, `compute_gradient_norms` (EP:457-471). A function that runs
only in `post` cannot change a saved number.

**Row-specific code** (from A.3): OBC objects only in 221, 222, 229 and 221-noisefree; PS2 only in the PS2 = 1 rows;
`lbfgs_polish._restore` only in 230 (the one row whose polish was rejected); `meter_regression` in every row but 230.

### A.2 Classification per file

Categories: **required** (changes numbers on the spec path; trace unless marked reading); **required, no numeric
effect** (guards, printing, bookkeeping on the required path that the clean code may keep or drop without changing
numbers, listed so Dirk decides); **needed by the evaluation** (not on the training path, but imported by
`Thesis-writeup/Code/Evaluation/`, see C6); **not needed**. Every function and class of the four locations is in A.3
with its trace status; this section gives the decision and the reason.

#### EP `scripts/gantry/gantry_interconnect_dynamic.py`
- required: the base `CFG` values the THESIS block does not override (EP:38-212: adam_eps, up_sample, fs,
  its_per_val, lbfgs_inner_iter, lbfgs_windows, device, compile_mode, checkpoint_chunk, static_pass_buffers,
  save_flag); the THESIS block EP:290-379 (env parsing, `_THESIS_FIXED`, detuning signs, smoke and device switches);
  `main()` EP:382-454 (seeding twice, data, normalisation, `resolve_resume_hp`, config.json, `build_model`,
  `build_closed_loop`, `prefit_physical_encoder`, `train_model_with_diagnostics`).
- not needed: the four experiment blocks EP:214-288 (`DATASET_AB_*`, `LBFGS_AB_FREEZE`, `OBC_ARM`), `THESIS_NITS`
  (EP:366-367), the `print_run_banner` call (EP:397, printing), and EP:457-471 (`evaluate_and_save`,
  `state_recovery_diagnostic`, `compute_gradient_norms`, post).

#### GD `config.py`
- required: `RunConfig` (fields in A.1), `__post_init__` (validation; required, no numeric effect), the derived
  properties, `y_decimation`, `record_trim`, the D-232 constants, `save_dir`, `config_json_dict` (config.json, which
  the evaluation reads back, C6).
- required, no numeric effect: `git_provenance` (+ `_git`), `_assert_json_coverage`, `_JSON_EXEMPT`.
- not needed: `default_hp` (never called).

#### GD `data.py`
- required: `THESIS_DATASETS`, `_THESIS_SPLIT`, `is_thesis`, `record_files`, `_register_controller`, `DataBundle`,
  `Norm`, `traj_dir`, `_load_u`, `_resample_u`, `_decimate_y` (both branches: lowpass and noise-free),
  `_record_at_fs_new`, `_load_record`, `servo_sigma`, `load_datasets` (+ `_split`), `compute_normalization`.
- required, no numeric effect: `load_mat_aug` and the `val_x_*`/`test_x_*` fields it fills (diagnostic ground truth,
  DATA:300-311); also imported by `closed_loop_report` (needed by the evaluation).
- not needed: `load_traj` (not run; the evaluation uses `_load_record`), the legacy `TRAIN_FILES`/`VAL_FILES`/
  `TEST_FILES` constants and the non-thesis branches of `record_files`/`traj_dir` (DATA:81-82, 144), the SNR noise
  branch (DATA:284-291).

#### GD `model.py`
- required: `COMPILE_BACKEND`, `PHY_IX`, `get_encoder_dims` (`data_lsq` branch), `build_model` (reduced joint branch,
  live MLP, `data_lsq` encoder, `init_model`, OBC attach for the obc arm), `prefit_physical_encoder`, `_prefit`,
  `_START_MAX_UNEXPLAINED`, `least_squares_start` (its R^2 and reproduction checks are guards), `train_model`.
- required by reading (GPU, real run): the `cfg.device == 'cuda'` branches MODEL:286-292, 343-349, 483-484, 518-519.
- not needed: the `linear_map` and `default` encoder branches (MODEL:42-46, 52-56, 210-245, 302-304), the raw
  joint-estimation branch (MODEL:104-109), the non-joint branch (MODEL:124-129), `static_pass_buffers`
  (MODEL:113-115), the zero-init and resnet g networks (MODEL:139-142), `ANN_REZERO_GATE` (MODEL:157-164), the orth
  penalty (MODEL:312-314).

#### GD `controller.py`
- required: `M_op`, `sys_stage_frf`, `cnorm_coeffs`, `cnorm_at`, `build_cfb_at`, `_tf_to_ss_batch`, `controller_ss`,
  `_stem`, `register_stored_controller`, `y_op_for`, `gain_for`, `build_controller_bank`, `build_closed_loop`, and the
  module constants they read (all run in every row).
- required by reading (real run): the compile block CTRL:329-341.

#### GD `training.py`
- required: `_json_safe`, `save_checkpoint_weights`, `run_lbfgs_polish`, `train_model_with_diagnostics`,
  `resolve_resume_hp` (no-resume branch). `_gantry_meters` is the polish `meters_fn`; with `meters_gate=False` it
  cannot change the outcome (required, no numeric effect).
- required although it is a diagnostic: `_NfProbe` and `_install_nf_val_probe`. The probe's `record` and
  `hist_arrays` (D-267) hold the validation history from which `train_model_with_diagnostics` picks the `selected` and
  `final` events, and `bestfit` in the checkpoint `.npz` is the `final` event's score (TRAIN:933-958); the evaluation
  reads it as `ref_score` (C6). The other probe parts (`_val_stats`, `_fmt`, `_joint_probe`, `_obc_probe`, the
  `WindowErrorStats` train statistics, the `[nf gap]` prints) have no numeric effect: they run under `no_grad` on a
  local window set drawn with a local generator.
- needed by the evaluation (not run on the training path): `load_checkpoint`, `resolve_checkpoint`,
  `load_optimizer_moments`, `_assert_same_architecture`, `_assert_same_dtype`. `infer_arch_from_hfn` and
  `config_for_checkpoint` are used by neither the pipeline nor `Thesis-writeup/Code/Evaluation/` (not needed).
- required by reading (GPU): TRAIN:776-782, 817-818.
- not needed: the resume branches (TRAIN:847-855, 1024-1067), the `start_phase='lbfgs'` branches, the smoke PS2 spec
  TRAIN:869-872.

#### GD `obc_gantry.py` (obc arm)
- required: `GantryReferenceSet` (+ `n`), `fourth_order_central_difference`, `_load_record_float64`,
  `build_reference_set` (successor branch), `GantryOBCCorrection` (`__init__`, `expansion_point`, `set_frozen`,
  `tangent_pair` affine branch, `forward`), `make_float64_block`, `make_basis_builder` (+ `build`, affine column),
  `make_reference_field` (+ `field`), `attach_obc`.
- required, no numeric effect: `GantryReferenceSet.memory_bytes` (print).
- needed by the evaluation: `build_reference_set`, `make_float64_block`, `make_basis_builder` (eval_overlap,
  eval_negation).
- not needed: `GantryOBCCorrection.step` and `correction_from_theta` (jvp reference path, tests), the tangent-only
  branches, the `static_pass_buffers` branches (OBCG:385-389), the `use_jvp` branch.

#### GD `evaluation.py`, `diagnostics.py`, `obc_diagnostics.py`, `closed_loop_report.py`, `baselines.py`
- not needed (Dirk: diagnostics are not a deliverable): every function runs only in `post` (`evaluate_and_save`,
  `capture_loss_history`, `report_joint_estimation`, `_record_mean`, `_make_plots`, `_build_save_dict`,
  `state_recovery_diagnostic`, `encoder_state_estimates`, `r2_per_channel`, `best_affine_r2`,
  `compute_gradient_norms`, `closed_loop_table`, `_score`) or never (`obc_diagnostics.py` whole, `baselines.py` whole,
  `encoder_init_state`, `zeroed_ann_validation`, `per_record_nrms`). One exception runs in the pipeline context:
  `print_run_banner` (printing only).
- needed by the evaluation: `closed_loop_report._without_learned` and `_true_free` (eval_greybox, eval_overlap), which
  pull in `common.oracle` and `data.load_mat_aug` through the module's imports. `evaluation.py:111` holds a GPU
  `fit_sys.cuda()` (post only).
- Part C check: none of them draws from a global RNG before the final checkpoint is written, so removing them changes
  no number (C1).

#### GD `__init__.py`
- required: puts `REPO_ROOT` on `sys.path` so `model_augmentation` and `common` import (GD/__init__.py:12-15).

#### CM `scripts/gantry/common/`
- not needed for training: `orth_penalty.py` (orth=False), `rezero_gate.py` (env gate), `feedback_curriculum.py`
  (never imported). `oracle.py`: only `truth_from_meta`, `_deriv_truth`, `oracle_closed_loop` run, in `post`
  (closed-loop report); the module is needed only because `closed_loop_report`, which the evaluation imports, imports
  it.

#### MA `fit_systems/interconnect.py`
- required: `Interconnect` (`__init__`, `_apply`, `_module_device_dtype`, `init_model`, `forward`, `init_forward`,
  `_signal_cone`, `output_only`, `init_connection_matrices`, `add_block`, `connect_signals`, `connect_block_signals`,
  `determine_signal_ix`, `convert_signal_name_to_ix`, `parse_connection_function_method`), `Signal_Connection.__init__`,
  `Dtype_DataLoader` (+ `_conv`), `SSE_Interconnect` (`__init__`, `init_nets`, `init_model`, `simulate`,
  `make_training_data`, `cal_validation_error`, `fit` + `validation`), `SSE_Interconnect_Composed` (`__init__`,
  `_sync_prediction_state_for_validation`, `loss`, `checkpoint_save_system`, `checkpoint_load_system`, `__getstate__`,
  `__setstate__`, the `optimizer` property, the delegating branch of `fit`).
- required, no numeric effect: `WindowErrorStats` (nf probe statistics accumulated in `loss`).
- required by reading (GPU): the `.cuda()` lines in `fit` (IC:933-934, 998-999). `concurrent_val` stays False, so
  IC:973-976, 1062-1066, 1109-1118 are not needed.
- not needed: `modified_encoder_net` (an encoder is always injected), `SSE_Interconnect.loss`, the base
  `_sync_prediction_state_for_validation` (overridden), `_model_dtype`, `init_state_multi`, `measure_act_multi`
  (open-loop deepSI paths), `SSE_Interconnect_ParamLoss`, the trajectory-penalty machinery of
  `SSE_Interconnect_Composed` (`_assert_optimizer_supported`, `_wrap_optimizer_step`, `_unwrap_optimizer_step`, the
  `traj_penalty` branches of `fit` and `checkpoint_load_system`, the `_TRAJ_*` constants), `reset_saved_signals`,
  `convert_signal_ix_to_name`, `Signal_Connection.__str__/__repr__`, the `save_signals` branch of `forward`
  (IC:173-182), the orth branch of `loss` (IC:1440-1445).

#### MA `fit_systems/blocks.py`
- required: `Block` (`__init__`, `init_block`), `Static_ANN_Block` (+ `forward`, `out_gate`), `Linear_Output_Block`,
  `Discrete_Nonlinear_Function_Block` (`__init__`, `forward`), `Gantry_State_Block` (`__init__`, `nonlinear_function`,
  `_rk4`, `_deriv_with`; `_deriv_both_with`, `_rk4_with_tangent` for OBC), `_Trainable_Gantry_State_Block`
  (`__init__`, `_build_KC`, `_build_M1M2`, `_mats_from_raw`, `nonlinear_function`, `pass_mats`, `_mats`,
  `__getstate__`, `__setstate__`, `_combos_from_raw`), `Reduced_Gantry_State_Block` (`__init__`, `combos_of`,
  `gauge_section`, `_m_diff_bound`, `combinations_from_free`, `recover_combinations`, `_recover_params`,
  `admissibility` (guard), `param_loss` (returns 0), `identifiable_combinations` (meters, no numeric effect); for OBC
  `transition_from_free`, `mats_and_tangent_from_free` (+ `_mats`)).
- needed by the evaluation (post only in the trace): `estimation_parameters`, `free_from_combinations`,
  `_initial_identifiable_combinations`, `_combination_training_label`, `param_table`; and the frozen-scheduling
  attributes `N_op`, `d_op`, `Y_op` read by `_deriv_with` (eval_common `frozen_y`).
- not needed: `Cubic_Block`, `Linear_State_Block`, `Parameterized_Linear_State_Block`,
  `Parameterized_Linear_Output_Block`, `Cascaded_Tanks_State_Block`, `Parameterized_Cascaded_Tanks_State_Block`,
  `Parameterized_MSD_State_Block`, `Nonlinear_MSD_State_Block`, `Parameterized_Gantry_State_Block` (raw),
  `Block.forward`, `Discrete_Nonlinear_Function_Block.init_norm` and `.nonlinear_function`, `Gantry_State_Block._mats`,
  `deriv`, `_deriv_tangent_with`, the overridden `_Trainable_Gantry_State_Block.physical_params`,
  `identifiable_combinations`, `_initial_identifiable_combinations`, `_combination_training_label`,
  `transition_and_tangent_from_free`, `Reduced_Gantry_State_Block.physical_params`. Note:
  `SSE_Interconnect_Composed.__init__` tests `isinstance(m, Parameterized_MSD_State_Block)` (IC:1309-1310); dropping
  the class means dropping that guard.

#### MA `fit_systems/closed_loop.py`
- required: `ControllerBank` (`__init__`, `gather`, `zero_state`, `step`; `n_controllers`, `physical_D` print only),
  `_rollout_segment`, `closed_loop_rollout` (un-checkpointed branch), `window_controller_index`,
  `WindowControllerIndex` (`build` with weights, `verify_by_content` guard), `ClosedLoopSimulator` (`__init__`,
  `__getstate__`, `eager`, `__call__`, `augment_training_data`, `_check_val_records` guard, `validation_error`
  `sim-NRMSe` branch), `closed_loop_free_run_rms_batch`.
- required, no numeric effect: `_note_update_time`.
- required by reading (real run): the compiled branch CL:584-591, 600-604.
- needed by the evaluation: `closed_loop_free_run_rms_batch` (also with `return_pred=True`).
- not needed: `ControllerBank.check_units`, `closed_loop_free_run_rms` (one-record wrapper), the checkpointed rollout
  CL:295-311, the `sim-RMS` return (CL:716).

#### MA `fit_systems/lbfgs_polish.py`
- required: `PolishSpec`, `decide`, `_named_params`, `_snapshot`, `_restore` (rejection; trace in 230 only), `_resync`,
  `window_count`, `FixedBatch`, `_eager`, `_frozen` (empty freeze), `_quiet_probes`, `warm_up_lazy_init`,
  `_make_closure` (+ `closure`), `_loss_value`, `_lbfgs_counts`, `_probe_closure`, `lbfgs_polish` (+ `_validate`).
- required, no numeric effect: `meter_regression`, `generic_meters`, `_all_meters` (meters do not gate).
- required by reading (real run): `_is_oom` and the exception paths LB:666-741 (an OOM or exception restores the
  weights and marks the polish as not run; this changes the result only when it fires), the multi-block loop with
  tolerance exits, `time_budget_s=None`.

#### MA `fit_systems/obc.py` (obc arm)
- required: `RankReport`, `OBCBasis` (`__init__`, `rank_tolerance`, `from_matrix`, `coefficient`, `apply`, `jt`;
  `r_perp`, `frobenius`, `relative_change` for the refresh print and the validation meters), `build_stacked_sensitivity`,
  `evaluate_step_rows`, `OBCLifecycle` (`__init__`, `refresh` with the rank policy, `maybe_refresh`, `coefficient`,
  `validation_coefficient`, `_prediction_solve`), `OBCRankLossAbort`.
- required by reading (GPU, real run): `OBCBasis.to` (OBC:322-328, called at OBC:525, 564 when basis and field are on
  different devices); the rank-loss branches OBC:460-478 (fire only if a refresh loses rank).
- needed by the evaluation: `build_stacked_sensitivity`.
- not needed: `_StaticCopy`, `StaticPassBuffers` (static_pass_buffers off), `project_onto`, `project_out`,
  `r_overlap`, `memory_bytes`, `directional_correction` (jvp path), `prediction_coefficient`, `stacked_diagnostics`,
  `state_summary`.

#### MA `fit_systems/pre_encoder.py`
- required: `linear_encoder_init_aug` (`__init__` with `wb_init='random'` and `xavier_uniform_gain1`, `forward`).
- not needed: `convert_to_same_type`, `System_data_with_x`, `System_data_norm_with_x`, `linear_encoder_init`,
  `SS_pre_encoder`, the reconstructability branch PRE:379-413, the D-055 offset buffers (no mean arguments are passed,
  PRE:448-450).

#### MA `fit_systems/ps2_opening.py` (PS2 rows)
- required: `PS2Spec`, `calibration_split`, `PS2Opening` (all methods); `summary`/`summary_json` feed the `.npz`
  `ps2_summary` field (no numeric effect).

#### MA other `fit_systems/` files
- not needed: `augmented_dynamics.py`, `continuous_augmentation.py`, `gyorok_orth_projection.py`,
  `multiple_shooting.py`, `obc_projection.py`, `orth_projection.py`, `white_box_models.py` (no function runs).

#### MA `systems/`
- required: `gantry_ss.py` (nominal parameters, `P`, `Cd`, `Dd`, `build_poly_constants`, `build_G_matrix_entries`).
- not needed: `gantry_linearization.py` (imported by `model.py` at module level, called only by the `linear_map`
  encoder), `cascaded_tanks.py`, `mass_spring_damper.py`.

#### MA `utils/`
- required: `utils.py` `added` (the marker decorator), `to_tensor`, `detect_algebraic_loop`, `expansion_matrix`,
  `selection_matrix`; `torch_nets.py` `live_g_feed_forward_nn`; `deepSI_corrections.py` `fixed_System_data`
  (`__init__`, `to_hist_future_data`), `fixed_System_data_norm` (`__init__`, `is_id`, `transform`).
- not needed: the rest of `utils.py` (plotting, `normalize_linear_ss_matrices` for linear_map,
  `compute_noisy_y_from_SNR`, ...), every other network in `torch_nets.py`, `fixed_System_data_norm.fit`,
  `inverse_transform`, the `*_transform` helpers, `make_training_data`, `__repr__`, `convert_to_same_type`.

#### Outside the four locations (flag)
- The pipeline depends on the installed `deepSI` package: `System_data`, `System_data_list`, `to_hist_future_data`,
  `SS_encoder_general` (base of `SSE_Interconnect`: `na`, `nb`, windowing in `make_training_data`), `System_fittable`
  (`checkpoint_save_system`/`checkpoint_load_system`, `parameters_with_names`, `init_optimizer`,
  `_check_and_refresh_optimizer_if_needed`), `My_Simple_DataLoader`, `loop_dataset`, `Tictoctimer`. A self-contained
  folder either keeps `deepSI` as a dependency or vendors these; vendoring changes the C1 batch order unless
  `default_rng(0)` and the in-place shuffle are reproduced.

### A.3 Every function and class (generated from the trace)

Status: `run (pipeline)` = executed before the final checkpoint in at least one row; `run (post only)` = executed only
after it; `not run`; `class` = class statement (see its methods). Rows: `all` = all eleven traced rows. The last
column lists branch arcs inside an executed function that no row took (from `.coverage.combined`): other modes,
guards that did not fire, GPU or compile branches, real-run cadence. Read together with A.2.

Rows traced: 221-lowpass, 221-noisefree, 222-lowpass, 223-lowpass, 224-lowpass, 224-noisefree, 225-lowpass, 226-lowpass, 227-lowpass, 229-lowpass, 230-lowpass


##### scripts/gantry/gantry_interconnect_dynamic.py

| line | name | trace | rows | unexecuted branch arcs |
|-|-|-|-|-|
| 382 | main | run (pipeline) | all | 403->409, 432->447, 444->450, 466->exit |

##### scripts/gantry/gantry_dynamic/baselines.py

| line | name | trace | rows | unexecuted branch arcs |
|-|-|-|-|-|
| 18 | stepwise_rollout | not run | - |  |
| 33 | compute_baseline_fp_nrms | not run | - |  |
| 91 | compute_baseline_fp_nrms._fp_step | not run | - |  |
| 120 | compute_all_baselines | not run | - |  |

##### scripts/gantry/gantry_dynamic/closed_loop_report.py

| line | name | trace | rows | unexecuted branch arcs |
|-|-|-|-|-|
| 38 | _without_learned | run (post only) | all | 58->60 |
| 67 | _true_free | run (post only) | all |  |
| 81 | _score | run (post only) | all |  |
| 86 | closed_loop_table | run (post only) | all | 105->106, 115->118, 120->122, 130->131 |

##### scripts/gantry/gantry_dynamic/config.py

| line | name | trace | rows | unexecuted branch arcs |
|-|-|-|-|-|
| 46 | y_decimation | run (pipeline) | all |  |
| 53 | record_trim | run (pipeline) | all |  |
| 60 | RunConfig (class) | class | all |  |
| 370 | RunConfig.__post_init__ | run (pipeline) | all | 376->377, 379->380, 381->382, 386->387, 389->390, 391->392, 393->394, 396->397, 398->399, 401->402, 403->404, 407->409, 410->412, 415->417, 420->421, 431->432, 436->437, 444->445, 450->451, 452->453, 454->455, 456->457, 463->464, 470->471, 475->476, 480->481, 490->493, 501->502, 506->507, 511->512, 515->516, 519->520, 524->525, 529->530, 535->536 |
| 542 | RunConfig.fs_new_hz | run (pipeline) | all |  |
| 546 | RunConfig.d | run (pipeline) | all |  |
| 550 | RunConfig.ts_new | run (pipeline) | all |  |
| 554 | RunConfig.dtype_np | run (pipeline) | all |  |
| 558 | RunConfig.dtype_pt | run (pipeline) | all |  |
| 562 | RunConfig.nf | run (pipeline) | all | 571->572 |
| 576 | RunConfig.na_nb | run (pipeline) | all |  |
| 583 | RunConfig.hp | run (pipeline) | all |  |
| 598 | default_hp | not run | - |  |
| 603 | save_dir | run (pipeline) | all |  |
| 610 | git_provenance | run (pipeline) | all | 635->636, 651->652, 651->654 |
| 628 | git_provenance._git | run (pipeline) | all |  |
| 668 | config_json_dict | run (pipeline) | all |  |
| 793 | _assert_json_coverage | run (pipeline) | all | 805->806 |

##### scripts/gantry/gantry_dynamic/controller.py

| line | name | trace | rows | unexecuted branch arcs |
|-|-|-|-|-|
| 72 | M_op | run (pipeline) | all |  |
| 81 | sys_stage_frf | run (pipeline) | all |  |
| 92 | cnorm_coeffs | run (pipeline) | all |  |
| 99 | cnorm_at | run (pipeline) | all |  |
| 104 | build_cfb_at | run (pipeline) | all |  |
| 124 | _tf_to_ss_batch | run (pipeline) | all |  |
| 143 | controller_ss | run (pipeline) | all |  |
| 181 | _stem | run (pipeline) | all |  |
| 198 | register_stored_controller | run (pipeline) | all | 200->205 |
| 210 | y_op_for | run (pipeline) | all | 218->219, 220->223, 223->224, 223->227 |
| 232 | gain_for | run (pipeline) | all |  |
| 246 | build_controller_bank | run (pipeline) | all |  |
| 269 | build_closed_loop | run (pipeline) | all | 292->293, 304->305, 311->329, 330->331, 339->340, 339->346, 350->354, 351->352 |

##### scripts/gantry/gantry_dynamic/data.py

| line | name | trace | rows | unexecuted branch arcs |
|-|-|-|-|-|
| 75 | is_thesis | run (pipeline) | all |  |
| 79 | record_files | run (pipeline) | all | 81->82 |
| 92 | _register_controller | run (pipeline) | all |  |
| 102 | DataBundle (class) | class | all |  |
| 127 | Norm (class) | class | all |  |
| 141 | traj_dir | run (pipeline) | all | 142->144 |
| 147 | _load_u | run (pipeline) | all |  |
| 152 | _resample_u | run (pipeline) | all | 166->167 |
| 172 | _decimate_y | run (pipeline) | all | 177->178 |
| 185 | _record_at_fs_new | run (pipeline) | all | 196->198, 198->200 |
| 203 | _load_record | run (pipeline) | all | 209->211 |
| 218 | load_traj | not run | - |  |
| 222 | servo_sigma | run (pipeline) | all | 226->227, 232->233 |
| 237 | load_mat_aug | run (pipeline) | all |  |
| 258 | load_datasets | run (pipeline) | all | 266->269, 270->271, 284->285, 285->286, 285->289, 286->287, 286->289, 287->288, 287->289, 289->290, 289->291 |
| 276 | load_datasets._split | run (pipeline) | all |  |
| 329 | compute_normalization | run (pipeline) | all |  |

##### scripts/gantry/gantry_dynamic/diagnostics.py

| line | name | trace | rows | unexecuted branch arcs |
|-|-|-|-|-|
| 19 | r2_per_channel | run (post only) | all |  |
| 27 | best_affine_r2 | run (post only) | all |  |
| 40 | encoder_state_estimates | run (post only) | all |  |
| 65 | state_recovery_diagnostic | run (post only) | all | 116->exit |
| 124 | compute_gradient_norms | run (post only) | all | 163->164, 178->179, 184->187, 194->197 |
| 208 | encoder_init_state | not run | - |  |
| 224 | zeroed_ann_validation | not run | - |  |
| 264 | print_run_banner | run (pipeline) | all | 280->281, 308->309 |

##### scripts/gantry/gantry_dynamic/evaluation.py

| line | name | trace | rows | unexecuted branch arcs |
|-|-|-|-|-|
| 31 | capture_loss_history | run (post only) | all | 42->47 |
| 53 | report_joint_estimation | run (post only) | all | 60->80, 67->75 |
| 83 | _record_mean | run (post only) | all | 90->88 |
| 95 | evaluate_and_save | run (post only) | all | 101->102, 110->111, 116->117, 131->133, 139->exit |
| 152 | _make_plots | run (post only) | all | 168->172, 172->176, 176->179, 179->180, 225->237 |
| 240 | _build_save_dict | run (post only) | all | 265->271, 271->275 |
| 278 | per_record_nrms | not run | - |  |

##### scripts/gantry/gantry_dynamic/model.py

| line | name | trace | rows | unexecuted branch arcs |
|-|-|-|-|-|
| 39 | get_encoder_dims | run (pipeline) | all | 42->43, 47->53 |
| 60 | build_model | run (pipeline) | all | 81->125, 96->105, 106->107, 106->108, 113->114, 116->132, 120->121, 138->142, 157->158, 158->159, 158->161, 210->212, 246->286, 286->287, 296->297, 300->303, 303->304, 303->312, 312->313 |
| 328 | prefit_physical_encoder | run (pipeline) | all | 343->344, 348->349 |
| 352 | _prefit | run (pipeline) | all | 355->356, 368->370 |
| 415 | least_squares_start | run (pipeline) | all | 430->431, 439->440, 455->456, 458->459, 461->463 |
| 467 | train_model | run (pipeline) | all | 472->473, 483->484, 518->519 |

##### scripts/gantry/gantry_dynamic/obc_diagnostics.py

| line | name | trace | rows | unexecuted branch arcs |
|-|-|-|-|-|
| 38 | qualify_basis | not run | - |  |
| 95 | print_qualification | not run | - |  |
| 122 | principal_angles | not run | - |  |
| 131 | absorber_truth_one_step | not run | - |  |
| 159 | absorber_truth_one_step.M | not run | - |  |
| 172 | absorber_truth_one_step.deriv | not run | - |  |
| 192 | check_truth_parameters | not run | - |  |
| 215 | discrepancy_fields | not run | - |  |
| 247 | rho_table | not run | - |  |

##### scripts/gantry/gantry_dynamic/obc_gantry.py

| line | name | trace | rows | unexecuted branch arcs |
|-|-|-|-|-|
| 49 | GantryReferenceSet (class) | class | 221-lowpass 221-noisefree 222-lowpass 229-lowpass |  |
| 68 | GantryReferenceSet.n | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 229-lowpass |  |
| 71 | GantryReferenceSet.memory_bytes | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 229-lowpass | 73->74 |
| 79 | fourth_order_central_difference | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 229-lowpass |  |
| 91 | _load_record_float64 | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 229-lowpass |  |
| 106 | build_reference_set | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 229-lowpass | 114->115, 136->137, 139->141, 163->171 |
| 175 | GantryOBCCorrection (class) | class | 221-lowpass 221-noisefree 222-lowpass 229-lowpass |  |
| 189 | GantryOBCCorrection.__init__ | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 229-lowpass | 193->194, 209->210 |
| 222 | GantryOBCCorrection.step | not run | - |  |
| 225 | GantryOBCCorrection.expansion_point | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 229-lowpass |  |
| 230 | GantryOBCCorrection.set_frozen | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 229-lowpass |  |
| 250 | GantryOBCCorrection.tangent_pair | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 229-lowpass | 261->263, 264->265, 267->268 |
| 271 | GantryOBCCorrection.correction_from_theta | not run | - |  |
| 286 | GantryOBCCorrection.forward | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 229-lowpass | 297->298, 301->302 |
| 312 | make_float64_block | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 229-lowpass |  |
| 319 | make_basis_builder | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 229-lowpass |  |
| 323 | make_basis_builder.build | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 229-lowpass | 327->328, 332->339 |
| 344 | make_reference_field | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 229-lowpass |  |
| 350 | make_reference_field.field | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 229-lowpass |  |
| 362 | attach_obc | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 229-lowpass | 372->373, 376->381, 385->387, 388->389, 388->390, 403->408 |

##### scripts/gantry/gantry_dynamic/training.py

| line | name | trace | rows | unexecuted branch arcs |
|-|-|-|-|-|
| 26 | _json_safe | run (pipeline) | all | 32->33, 34->35 |
| 39 | save_checkpoint_weights | run (pipeline) | all |  |
| 67 | resolve_checkpoint | not run | - |  |
| 124 | load_optimizer_moments | not run | - |  |
| 158 | _assert_same_architecture | not run | - |  |
| 198 | infer_arch_from_hfn | not run | - |  |
| 236 | config_for_checkpoint | not run | - |  |
| 248 | _assert_same_dtype | not run | - |  |
| 277 | load_checkpoint | not run | - |  |
| 319 | _NfProbe (class) | class | all |  |
| 362 | _NfProbe.__init__ | run (pipeline) | all | 393->399, 411->421, 427->440, 440->exit |
| 446 | _NfProbe._val_stats | run (pipeline) | all | 453->454, 463->464 |
| 490 | _NfProbe._fmt | run (pipeline) | all |  |
| 498 | _NfProbe._joint_probe | run (pipeline) | all | 505->514, 516->517, 533->534, 545->548, 551->558 |
| 560 | _NfProbe._obc_probe | run (pipeline) | all | 580->581 |
| 593 | _NfProbe.record | run (pipeline) | all | 600->601, 607->exit |
| 610 | _NfProbe.hist_arrays | run (pipeline) | all | 612->613 |
| 627 | _NfProbe.__call__ | run (pipeline) | all | 633->635, 638->663, 641->647, 649->659, 651->653, 655->657, 657->659, 663->exit |
| 671 | _install_nf_val_probe | run (pipeline) | all | 679->700 |
| 703 | _gantry_meters | run (pipeline) | all | 725->726, 728->729, 728->737, 738->739 |
| 753 | run_lbfgs_polish | run (pipeline) | all | 764->765, 776->777, 791->793, 811->815 |
| 821 | train_model_with_diagnostics | run (pipeline) | all | 843->845, 848->849, 859->860, 869->873, 882->886, 903->906, 907->910, 911->918, 928->937, 939->941, 947->953, 954->956, 956->957, 956->958, 974->982, 986->1009 |
| 1016 | resolve_resume_hp | run (pipeline) | all | 1031->1032, 1035->1051, 1053->1054, 1053->1069, 1063->1064, 1063->1066, 1066->1067, 1066->1077 |

##### scripts/gantry/common/feedback_curriculum.py

| line | name | trace | rows | unexecuted branch arcs |
|-|-|-|-|-|
| 21 | FixedWindowBatch (class) | class | - |  |
| 30 | build_fixed_window_batch | not run | - |  |
| 66 | save_training_snapshot | not run | - |  |
| 76 | load_training_snapshot | not run | - |  |
| 84 | FeedbackTargetSelector (class) | class | - |  |
| 94 | FeedbackTargetSelector.__init__ | not run | - |  |
| 125 | FeedbackTargetSelector.initial_strength | not run | - |  |
| 128 | FeedbackTargetSelector._evaluate | not run | - |  |
| 165 | FeedbackTargetSelector._checkpoint_path | not run | - |  |
| 168 | FeedbackTargetSelector._write_state | not run | - |  |
| 187 | FeedbackTargetSelector.__call__ | not run | - |  |
| 274 | local_dynamics_diagnostic | not run | - |  |

##### scripts/gantry/common/oracle.py

| line | name | trace | rows | unexecuted branch arcs |
|-|-|-|-|-|
| 61 | _M | not run | - |  |
| 73 | _deriv | not run | - |  |
| 83 | simulate | not run | - |  |
| 101 | oracle_open_loop | not run | - |  |
| 144 | truth_from_meta | run (post only) | all | 153->154 |
| 160 | _deriv_truth | run (post only) | all |  |
| 186 | oracle_closed_loop | run (post only) | all |  |

##### scripts/gantry/common/orth_penalty.py

| line | name | trace | rows | unexecuted branch arcs |
|-|-|-|-|-|
| 53 | theta_bar_for | not run | - |  |
| 63 | _cache_path | not run | - |  |
| 81 | _x_logical_from_data | not run | - |  |
| 107 | _make_block | not run | - |  |
| 117 | _compute | not run | - |  |
| 180 | build_orth_penalty | not run | - |  |

##### scripts/gantry/common/rezero_gate.py

| line | name | trace | rows | unexecuted branch arcs |
|-|-|-|-|-|
| 46 | _Gate (class) | class | - |  |
| 49 | _Gate.__init__ | not run | - |  |
| 62 | _Gate.gate | not run | - |  |
| 65 | _Gate.forward | not run | - |  |
| 69 | apply_rezero_gate | not run | - |  |

##### model_augmentation/fit_systems/augmented_dynamics.py

| line | name | trace | rows | unexecuted branch arcs |
|-|-|-|-|-|
| 98 | AugmentedDynamics (class) | class | - |  |
| 118 | AugmentedDynamics.__init__ | not run | - |  |
| 172 | AugmentedDynamics.net | not run | - |  |
| 176 | AugmentedDynamics.rho | not run | - |  |
| 195 | AugmentedDynamics.gamma | not run | - |  |
| 199 | AugmentedDynamics.effective_B | not run | - |  |
| 206 | AugmentedDynamics.trainable_parameters | not run | - |  |
| 212 | AugmentedDynamics.pole_table | not run | - |  |
| 229 | AugmentedDynamics.forward | not run | - |  |
| 257 | find_ann_block | not run | - |  |
| 263 | find_wrapper | not run | - |  |
| 273 | final_linear | not run | - |  |
| 282 | reset_final_layer | not run | - |  |
| 302 | empirical_input_scale | not run | - |  |
| 349 | draw_input_map | not run | - |  |

##### model_augmentation/fit_systems/blocks.py

| line | name | trace | rows | unexecuted branch arcs |
|-|-|-|-|-|
| 12 | Block (class) | class | all |  |
| 15 | Block.__init__ | run (pipeline) | all |  |
| 24 | Block.init_block | run (pipeline) | all |  |
| 27 | Block.forward | not run | - |  |
| 31 | Static_ANN_Block (class) | class | all |  |
| 32 | Static_ANN_Block.__init__ | run (pipeline) | all |  |
| 69 | Static_ANN_Block.forward | run (pipeline) | all | 76->77 |
| 83 | Cubic_Block (class) | class | all |  |
| 84 | Cubic_Block.__init__ | not run | - |  |
| 88 | Cubic_Block.forward | not run | - |  |
| 94 | Linear_State_Block (class) | class | all |  |
| 95 | Linear_State_Block.__init__ | not run | - |  |
| 109 | Linear_State_Block.forward | not run | - |  |
| 120 | Linear_Output_Block (class) | class | all |  |
| 121 | Linear_Output_Block.__init__ | run (pipeline) | all | 130->131 |
| 138 | Linear_Output_Block.forward | run (pipeline) | all |  |
| 146 | Parameterized_Linear_State_Block (class) | class | all |  |
| 147 | Parameterized_Linear_State_Block.__init__ | not run | - |  |
| 178 | Parameterized_Linear_State_Block.forward | not run | - |  |
| 187 | Parameterized_Linear_State_Block.param_loss | not run | - |  |
| 199 | Parameterized_Linear_Output_Block (class) | class | all |  |
| 200 | Parameterized_Linear_Output_Block.__init__ | not run | - |  |
| 232 | Parameterized_Linear_Output_Block.forward | not run | - |  |
| 239 | Parameterized_Linear_Output_Block.param_loss | not run | - |  |
| 251 | Discrete_Nonlinear_Function_Block (class) | class | all |  |
| 252 | Discrete_Nonlinear_Function_Block.__init__ | run (pipeline) | all |  |
| 255 | Discrete_Nonlinear_Function_Block.init_norm | not run | - |  |
| 260 | Discrete_Nonlinear_Function_Block.forward | run (pipeline) | all |  |
| 268 | Discrete_Nonlinear_Function_Block.nonlinear_function | not run | - |  |
| 274 | Cascaded_Tanks_State_Block (class) | class | all |  |
| 275 | Cascaded_Tanks_State_Block.__init__ | not run | - |  |
| 301 | Cascaded_Tanks_State_Block.nonlinear_function | not run | - |  |
| 361 | Parameterized_Cascaded_Tanks_State_Block (class) | class | all |  |
| 362 | Parameterized_Cascaded_Tanks_State_Block.__init__ | not run | - |  |
| 388 | Parameterized_Cascaded_Tanks_State_Block.nonlinear_function | not run | - |  |
| 448 | Parameterized_MSD_State_Block (class) | class | all |  |
| 449 | Parameterized_MSD_State_Block.__init__ | not run | - |  |
| 503 | Parameterized_MSD_State_Block.nonlinear_function | not run | - |  |
| 557 | Nonlinear_MSD_State_Block (class) | class | all |  |
| 558 | Nonlinear_MSD_State_Block.__init__ | not run | - |  |
| 581 | Nonlinear_MSD_State_Block.nonlinear_function | not run | - |  |
| 595 | Nonlinear_MSD_State_Block.deriv | not run | - |  |
| 664 | Gantry_State_Block (class) | class | all |  |
| 699 | Gantry_State_Block.__init__ | run (pipeline) | all | 757->758 |
| 779 | Gantry_State_Block.nonlinear_function | run (pipeline) | all |  |
| 791 | Gantry_State_Block._rk4 | run (pipeline) | all |  |
| 800 | Gantry_State_Block._mats | not run | - |  |
| 810 | Gantry_State_Block.deriv | not run | - |  |
| 814 | Gantry_State_Block._deriv_with | run (pipeline) | all | 843->844, 871->872, 873->874, 873->875 |
| 893 | Gantry_State_Block._deriv_tangent_with | not run | - |  |
| 902 | Gantry_State_Block._deriv_both_with | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 229-lowpass | 909->910, 968->969, 971->972 |
| 979 | Gantry_State_Block._rk4_with_tangent | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 229-lowpass |  |
| 996 | _Trainable_Gantry_State_Block (class) | class | all |  |
| 1008 | _Trainable_Gantry_State_Block.__init__ | run (pipeline) | all | 1024->1025, 1030->1031 |
| 1042 | _Trainable_Gantry_State_Block._build_KC | run (pipeline) | all |  |
| 1064 | _Trainable_Gantry_State_Block._build_M1M2 | run (pipeline) | all |  |
| 1079 | _Trainable_Gantry_State_Block._mats_from_raw | run (pipeline) | all |  |
| 1102 | _Trainable_Gantry_State_Block.nonlinear_function | run (pipeline) | all |  |
| 1112 | _Trainable_Gantry_State_Block.pass_mats | run (pipeline) | all |  |
| 1122 | _Trainable_Gantry_State_Block._mats | run (pipeline) | all | 1123->1124 |
| 1147 | _Trainable_Gantry_State_Block.__getstate__ | run (pipeline) | all |  |
| 1152 | _Trainable_Gantry_State_Block.__setstate__ | run (pipeline) | all |  |
| 1156 | _Trainable_Gantry_State_Block.physical_params | not run | - |  |
| 1169 | _Trainable_Gantry_State_Block._combos_from_raw | run (pipeline) | all |  |
| 1183 | _Trainable_Gantry_State_Block.identifiable_combinations | not run | - |  |
| 1187 | _Trainable_Gantry_State_Block._initial_identifiable_combinations | not run | - |  |
| 1191 | _Trainable_Gantry_State_Block._combination_training_label | not run | - |  |
| 1194 | _Trainable_Gantry_State_Block.param_table | run (post only) | all |  |
| 1214 | Parameterized_Gantry_State_Block (class) | class | all |  |
| 1222 | Parameterized_Gantry_State_Block.__init__ | not run | - |  |
| 1231 | Parameterized_Gantry_State_Block._recover_params | not run | - |  |
| 1235 | Parameterized_Gantry_State_Block.param_loss | not run | - |  |
| 1243 | Parameterized_Gantry_State_Block.estimation_parameters | not run | - |  |
| 1250 | Reduced_Gantry_State_Block (class) | class | all |  |
| 1304 | Reduced_Gantry_State_Block.__init__ | run (pipeline) | all | 1308->1316, 1318->1319, 1322->1323, 1338->1339 |
| 1350 | Reduced_Gantry_State_Block.combos_of | run (pipeline) | all |  |
| 1365 | Reduced_Gantry_State_Block.gauge_section | run (pipeline) | all |  |
| 1385 | Reduced_Gantry_State_Block._m_diff_bound | run (pipeline) | all |  |
| 1397 | Reduced_Gantry_State_Block.combinations_from_free | run (pipeline) | all |  |
| 1409 | Reduced_Gantry_State_Block.free_from_combinations | run (post only) | all | 1414->1415, 1417->1418 |
| 1424 | Reduced_Gantry_State_Block.recover_combinations | run (pipeline) | all |  |
| 1429 | Reduced_Gantry_State_Block.transition_from_free | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 229-lowpass |  |
| 1461 | Reduced_Gantry_State_Block.mats_and_tangent_from_free | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 229-lowpass |  |
| 1463 | Reduced_Gantry_State_Block.mats_and_tangent_from_free._mats | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 229-lowpass |  |
| 1470 | Reduced_Gantry_State_Block.transition_and_tangent_from_free | not run | - |  |
| 1482 | Reduced_Gantry_State_Block._recover_params | run (pipeline) | all |  |
| 1494 | Reduced_Gantry_State_Block.admissibility | run (pipeline) | all |  |
| 1528 | Reduced_Gantry_State_Block.param_loss | run (pipeline) | all | 1530->1532 |
| 1536 | Reduced_Gantry_State_Block.identifiable_combinations | run (pipeline) | all |  |
| 1541 | Reduced_Gantry_State_Block._initial_identifiable_combinations | run (post only) | all |  |
| 1544 | Reduced_Gantry_State_Block._combination_training_label | run (post only) | all |  |
| 1547 | Reduced_Gantry_State_Block.physical_params | not run | - |  |
| 1551 | Reduced_Gantry_State_Block.estimation_parameters | run (post only) | all |  |

##### model_augmentation/fit_systems/closed_loop.py

| line | name | trace | rows | unexecuted branch arcs |
|-|-|-|-|-|
| 76 | ControllerBank (class) | class | all |  |
| 122 | ControllerBank.__init__ | run (pipeline) | all | 132->133, 135->136 |
| 150 | ControllerBank.n_controllers | run (pipeline) | all |  |
| 153 | ControllerBank.physical_D | run (pipeline) | all |  |
| 158 | ControllerBank.gather | run (pipeline) | all |  |
| 162 | ControllerBank.zero_state | run (pipeline) | all |  |
| 167 | ControllerBank.step | run (pipeline) | all |  |
| 187 | ControllerBank.check_units | not run | - |  |
| 216 | _rollout_segment | run (pipeline) | all |  |
| 245 | closed_loop_rollout | run (pipeline) | all | 269->270, 291->301, 302->303, 302->311 |
| 314 | window_controller_index | run (pipeline) | all | 340->341 |
| 354 | WindowControllerIndex (class) | class | all |  |
| 370 | WindowControllerIndex.__init__ | run (pipeline) | all |  |
| 377 | WindowControllerIndex.build | run (pipeline) | all | 406->407, 420->421 |
| 427 | WindowControllerIndex.verify_by_content | run (pipeline) | all | 437->438 |
| 445 | ClosedLoopSimulator (class) | class | all |  |
| 482 | ClosedLoopSimulator.__init__ | run (pipeline) | all | 488->490 |
| 535 | ClosedLoopSimulator.__getstate__ | run (pipeline) | all |  |
| 541 | ClosedLoopSimulator.eager | run (pipeline) | all |  |
| 559 | ClosedLoopSimulator.__call__ | run (pipeline) | all | 563->564, 585->586, 600->604 |
| 608 | ClosedLoopSimulator._note_update_time | run (pipeline) | all | 633->634 |
| 640 | ClosedLoopSimulator.augment_training_data | run (pipeline) | all |  |
| 647 | ClosedLoopSimulator._check_val_records | run (pipeline) | all | 650->651, 658->660 |
| 667 | ClosedLoopSimulator.validation_error | run (pipeline) | all | 684->685, 686->687, 714->716 |
| 719 | closed_loop_free_run_rms_batch | run (pipeline) | all | 769->770 |
| 817 | closed_loop_free_run_rms | not run | - |  |

##### model_augmentation/fit_systems/continuous_augmentation.py

| line | name | trace | rows | unexecuted branch arcs |
|-|-|-|-|-|
| 20 | rk4_step | not run | - |  |
| 29 | CombinedContinuousGantryBlock (class) | class | - |  |
| 39 | CombinedContinuousGantryBlock.__init__ | not run | - |  |
| 62 | CombinedContinuousGantryBlock.correction_field | not run | - |  |
| 70 | CombinedContinuousGantryBlock.derivative | not run | - |  |
| 77 | CombinedContinuousGantryBlock.forward | not run | - |  |
| 88 | CombinedContinuousGantryBlock.param_loss | not run | - |  |
| 93 | CombinedContinuousGantryBlock.__getattr__ | not run | - |  |

##### model_augmentation/fit_systems/gyorok_orth_projection.py

| line | name | trace | rows | unexecuted branch arcs |
|-|-|-|-|-|
| 48 | _block_step | not run | - |  |
| 53 | gantry_orth_matrix | not run | - |  |
| 79 | gantry_orth_matrix.f_single | not run | - |  |
| 104 | finite_difference_check | not run | - |  |
| 138 | orth_basis | not run | - |  |
| 151 | spectrum_report | not run | - |  |
| 188 | GyorokOrthPenalty (class) | class | - |  |
| 202 | GyorokOrthPenalty.__init__ | not run | - |  |
| 223 | GyorokOrthPenalty.Q | not run | - |  |
| 226 | GyorokOrthPenalty.penalty_of_field | not run | - |  |
| 229 | GyorokOrthPenalty.forward | not run | - |  |

##### model_augmentation/fit_systems/interconnect.py

| line | name | trace | rows | unexecuted branch arcs |
|-|-|-|-|-|
| 21 | Interconnect (class) | class | all |  |
| 22 | Interconnect.__init__ | run (pipeline) | all |  |
| 77 | Interconnect._apply | run (pipeline) | all |  |
| 88 | Interconnect._module_device_dtype | run (pipeline) | all | 89->91 |
| 93 | Interconnect.init_model | run (pipeline) | all |  |
| 100 | Interconnect.forward | run (pipeline) | all | 109->113, 116->120, 125->126, 137->138, 138->139, 138->143, 139->138, 139->140, 173->174, 176->177, 176->181 |
| 193 | Interconnect.reset_saved_signals | not run | - |  |
| 199 | Interconnect.init_forward | run (pipeline) | all | 273->exit |
| 280 | Interconnect._signal_cone | run (pipeline) | all | 291->292 |
| 303 | Interconnect.output_only | run (pipeline) | all | 317->319, 320->322, 322->323, 322->325, 330->331, 331->332, 331->336, 332->331, 332->333 |
| 354 | Interconnect.init_connection_matrices | run (pipeline) | all | 372->373, 375->376, 385->386, 414->415, 421->422, 428->429, 436->exit |
| 444 | Interconnect.add_block | run (pipeline) | all |  |
| 457 | Interconnect.connect_signals | run (pipeline) | all |  |
| 474 | Interconnect.connect_block_signals | run (pipeline) | all |  |
| 483 | Interconnect.determine_signal_ix | run (pipeline) | all | 486->487, 488->491 |
| 493 | Interconnect.convert_signal_name_to_ix | run (pipeline) | all | 496->498, 498->499, 498->503 |
| 505 | Interconnect.convert_signal_ix_to_name | not run | - |  |
| 525 | Interconnect.parse_connection_function_method | run (pipeline) | all | 530->532, 532->exit, 532->533 |
| 535 | Signal_Connection (class) | class | all |  |
| 537 | Signal_Connection.__init__ | run (pipeline) | all |  |
| 545 | Signal_Connection.__str__ | not run | - |  |
| 549 | Signal_Connection.__repr__ | not run | - |  |
| 552 | modified_encoder_net (class) | class | all |  |
| 553 | modified_encoder_net.__init__ | not run | - |  |
| 561 | modified_encoder_net.forward | not run | - |  |
| 568 | Dtype_DataLoader (class) | class | all |  |
| 594 | Dtype_DataLoader.__init__ | run (pipeline) | all |  |
| 595 | Dtype_DataLoader.__init__._conv | run (pipeline) | all |  |
| 604 | SSE_Interconnect (class) | class | all |  |
| 628 | SSE_Interconnect.__init__ | run (pipeline) | all |  |
| 644 | SSE_Interconnect.init_nets | run (pipeline) | all | 647->648 |
| 651 | SSE_Interconnect.init_model | run (pipeline) | all | 653->654, 658->659 |
| 678 | SSE_Interconnect.simulate | run (pipeline) | all | 691->693, 699->700, 699->705, 700->701, 700->703 |
| 707 | SSE_Interconnect.loss | not run | - |  |
| 746 | SSE_Interconnect.make_training_data | run (pipeline) | all | 749->751 |
| 753 | SSE_Interconnect.cal_validation_error | run (pipeline) | all | 762->765 |
| 770 | SSE_Interconnect._sync_prediction_state_for_validation | not run | - |  |
| 779 | SSE_Interconnect._model_dtype | not run | - |  |
| 785 | SSE_Interconnect.init_state_multi | not run | - |  |
| 800 | SSE_Interconnect.measure_act_multi | not run | - |  |
| 808 | SSE_Interconnect.fit | run (pipeline) | all | 915->916, 920->921, 920->922, 929->930, 933->934, 956->960, 960->961, 973->974, 979->982, 986->989, 989->990, 998->999, 1008->1023, 1010->1011, 1017->1018, 1041->1042, 1049->1050, 1062->1063, 1063->1064, 1063->1071, 1076->1077, 1080->1081, 1094->1095, 1099->1100, 1100->996, 1100->1101, 1109->1110, 1111->1112, 1111->1117, 1113->1114, 1113->1117, 1150->exit |
| 881 | SSE_Interconnect.fit.validation | run (pipeline) | all | 883->884, 892->893, 909->910 |
| 1155 | SSE_Interconnect_ParamLoss (class) | class | all |  |
| 1173 | SSE_Interconnect_ParamLoss.loss | not run | - |  |
| 1182 | WindowErrorStats (class) | class | all |  |
| 1204 | WindowErrorStats.__init__ | run (pipeline) | all |  |
| 1209 | WindowErrorStats.reset | run (pipeline) | all |  |
| 1220 | WindowErrorStats.update | run (pipeline) | all |  |
| 1241 | WindowErrorStats.summary | run (pipeline) | all |  |
| 1257 | SSE_Interconnect_Composed (class) | class | all |  |
| 1305 | SSE_Interconnect_Composed.__init__ | run (pipeline) | all | 1311->1312 |
| 1332 | SSE_Interconnect_Composed._sync_prediction_state_for_validation | run (pipeline) | all | 1343->1344 |
| 1383 | SSE_Interconnect_Composed.loss | run (pipeline) | all | 1390->1400, 1402->1403, 1421->1422, 1440->1441, 1441->1442, 1441->1445 |
| 1505 | SSE_Interconnect_Composed.checkpoint_save_system | run (pipeline) | all |  |
| 1514 | SSE_Interconnect_Composed.checkpoint_load_system | run (pipeline) | all | 1523->1524, 1558->1559, 1560->1561, 1560->1563, 1563->1564, 1563->1565, 1565->1567 |
| 1570 | SSE_Interconnect_Composed.__getstate__ | run (pipeline) | all |  |
| 1599 | SSE_Interconnect_Composed.__setstate__ | run (pipeline) | all | 1611->1612 |
| 1622 | SSE_Interconnect_Composed.optimizer | run (pipeline) | all |  |
| 1626 | SSE_Interconnect_Composed.optimizer | run (pipeline) | all | 1628->1630 |
| 1635 | SSE_Interconnect_Composed._assert_optimizer_supported | not run | - |  |
| 1664 | SSE_Interconnect_Composed._wrap_optimizer_step | not run | - |  |
| 1672 | SSE_Interconnect_Composed._wrap_optimizer_step.step_with_penalty | not run | - |  |
| 1682 | SSE_Interconnect_Composed._wrap_optimizer_step.step_with_penalty.wrapped | not run | - |  |
| 1699 | SSE_Interconnect_Composed._unwrap_optimizer_step | not run | - |  |
| 1704 | SSE_Interconnect_Composed.fit | run (pipeline) | all | 1705->1711, 1712->1713, 1712->1719, 1720->1721, 1720->1725, 1725->1726, 1725->1732, 1754->1755, 1754->1762, 1762->1763, 1762->1767 |

##### model_augmentation/fit_systems/lbfgs_polish.py

| line | name | trace | rows | unexecuted branch arcs |
|-|-|-|-|-|
| 50 | PolishSpec (class) | class | all |  |
| 78 | meter_regression | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 223-lowpass 224-lowpass 224-noisefree 225-lowpass 226-lowpass 227-lowpass 229-lowpass | 87->88, 89->90, 89->91, 92->87, 92->93 |
| 97 | decide | run (pipeline) | all | 110->111, 113->114, 114->115, 114->117, 118->119, 118->120, 125->126 |
| 134 | _named_params | run (pipeline) | all | 149->148 |
| 155 | _snapshot | run (pipeline) | all |  |
| 165 | _restore | run (pipeline) | 230-lowpass |  |
| 171 | _resync | run (pipeline) | all | 177->exit |
| 181 | generic_meters | run (pipeline) | all | 197->199 |
| 206 | window_count | run (pipeline) | all | 226->227 |
| 234 | FixedBatch (class) | class | all |  |
| 258 | FixedBatch.__init__ | run (pipeline) | all | 277->278, 287->291, 309->310, 315->316, 328->exit |
| 296 | FixedBatch.__init__._to_tensor | run (pipeline) | all |  |
| 334 | FixedBatch.__iter__ | run (pipeline) | all |  |
| 347 | _eager | run (pipeline) | all |  |
| 373 | _frozen | run (pipeline) | all | 386->387, 391->392 |
| 396 | _quiet_probes | run (pipeline) | all |  |
| 420 | warm_up_lazy_init | run (pipeline) | all |  |
| 464 | _make_closure | run (pipeline) | all |  |
| 466 | _make_closure.closure | run (pipeline) | all | 477->478 |
| 483 | _loss_value | run (pipeline) | all |  |
| 500 | _lbfgs_counts | run (pipeline) | all |  |
| 515 | _is_oom | not run | - |  |
| 522 | _probe_closure | run (pipeline) | all | 538->539, 538->541, 543->546 |
| 549 | lbfgs_polish | run (pipeline) | all | 623->624, 627->628, 647->648, 673->674, 682->683, 687->688, 696->697, 696->698, 701->713, 737->738, 737->739, 739->740, 739->741, 745->749, 749->753, 757->758, 767->768, 769->770, 790->802, 799->802 |
| 588 | lbfgs_polish._all_meters | run (pipeline) | all | 590->595 |
| 597 | lbfgs_polish._validate | run (pipeline) | all | 598->599 |

##### model_augmentation/fit_systems/multiple_shooting.py

| line | name | trace | rows | unexecuted branch arcs |
|-|-|-|-|-|
| 45 | SSE_Interconnect_MultipleShooting (class) | class | - |  |
| 96 | SSE_Interconnect_MultipleShooting._norm | not run | - |  |
| 104 | SSE_Interconnect_MultipleShooting._defect_penalty | not run | - |  |
| 166 | SSE_Interconnect_MultipleShooting.loss | not run | - |  |

##### model_augmentation/fit_systems/obc.py

| line | name | trace | rows | unexecuted branch arcs |
|-|-|-|-|-|
| 44 | _StaticCopy (class) | class | 221-lowpass 221-noisefree 222-lowpass 229-lowpass |  |
| 67 | _StaticCopy.forward | not run | - |  |
| 74 | _StaticCopy.backward | not run | - |  |
| 78 | StaticPassBuffers (class) | class | 221-lowpass 221-noisefree 222-lowpass 229-lowpass |  |
| 87 | StaticPassBuffers.__init__ | not run | - |  |
| 91 | StaticPassBuffers._alloc | not run | - |  |
| 98 | StaticPassBuffers.__call__ | not run | - |  |
| 114 | StaticPassBuffers.reset | not run | - |  |
| 118 | OBCRankLossAbort (class) | class | 221-lowpass 221-noisefree 222-lowpass 229-lowpass |  |
| 124 | RankReport (class) | class | 221-lowpass 221-noisefree 222-lowpass 229-lowpass |  |
| 139 | RankReport.lines | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 229-lowpass | 145->148 |
| 160 | OBCBasis (class) | class | 221-lowpass 221-noisefree 222-lowpass 229-lowpass |  |
| 169 | OBCBasis.__init__ | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 229-lowpass |  |
| 183 | OBCBasis.rank_tolerance | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 229-lowpass | 186->188 |
| 191 | OBCBasis.from_matrix | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 229-lowpass | 194->195, 196->197, 209->212, 234->238 |
| 242 | OBCBasis.coefficient | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 229-lowpass | 244->245 |
| 250 | OBCBasis.apply | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 229-lowpass | 253->255 |
| 257 | OBCBasis.jt | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 229-lowpass |  |
| 262 | OBCBasis.project_onto | not run | - |  |
| 267 | OBCBasis.project_out | not run | - |  |
| 272 | OBCBasis.r_perp | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 229-lowpass |  |
| 280 | OBCBasis.r_overlap | not run | - |  |
| 289 | OBCBasis.frobenius | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 229-lowpass | 290->291 |
| 295 | OBCBasis.relative_change | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 229-lowpass | 298->299 |
| 322 | OBCBasis.to | not run | - |  |
| 330 | OBCBasis.device | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 229-lowpass |  |
| 333 | OBCBasis.memory_bytes | not run | - |  |
| 341 | build_stacked_sensitivity | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 229-lowpass | 352->353 |
| 367 | evaluate_step_rows | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 229-lowpass |  |
| 380 | directional_correction | not run | - |  |
| 392 | OBCLifecycle (class) | class | 221-lowpass 221-noisefree 222-lowpass 229-lowpass |  |
| 410 | OBCLifecycle.__init__ | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 229-lowpass | 415->416 |
| 446 | OBCLifecycle.refresh | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 229-lowpass | 454->455, 460->461, 465->466, 465->469, 469->470, 469->473, 473->474, 473->478, 476->477, 476->478, 487->494 |
| 501 | OBCLifecycle.maybe_refresh | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 229-lowpass | 503->504 |
| 512 | OBCLifecycle.coefficient | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 229-lowpass | 519->520, 524->525 |
| 533 | OBCLifecycle.prediction_coefficient | not run | - |  |
| 543 | OBCLifecycle.validation_coefficient | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 229-lowpass |  |
| 558 | OBCLifecycle._prediction_solve | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 229-lowpass | 559->560, 563->564, 567->568 |
| 594 | OBCLifecycle.stacked_diagnostics | not run | - |  |
| 607 | OBCLifecycle.memory_bytes | not run | - |  |
| 610 | OBCLifecycle.state_summary | not run | - |  |

##### model_augmentation/fit_systems/obc_projection.py

| line | name | trace | rows | unexecuted branch arcs |
|-|-|-|-|-|
| 91 | reduced_step | not run | - |  |
| 100 | stacked_basis | not run | - |  |
| 143 | directional | not run | - |  |
| 161 | rank_report | not run | - |  |
| 185 | principal_angles | not run | - |  |
| 193 | RankLossAbort (class) | class | - |  |
| 204 | OBCBasis (class) | class | - |  |
| 211 | OBCBasis.__init__ | not run | - |  |
| 238 | OBCBasis._check_rank | not run | - |  |
| 266 | OBCBasis.range_basis | not run | - |  |
| 270 | OBCBasis.coefficient | not run | - |  |
| 274 | OBCBasis.rho | not run | - |  |
| 285 | OBCBasis.residual_ratio | not run | - |  |
| 293 | OBCBasis.spectrum | not run | - |  |
| 299 | OBC_Projected_ANN_Block (class) | class | - |  |
| 326 | OBC_Projected_ANN_Block.__init__ | not run | - |  |
| 360 | OBC_Projected_ANN_Block.phys | not run | - |  |
| 363 | OBC_Projected_ANN_Block.init_block | not run | - |  |
| 367 | OBC_Projected_ANN_Block.phys_input | not run | - |  |
| 371 | OBC_Projected_ANN_Block.rebuild_basis | not run | - |  |
| 385 | OBC_Projected_ANN_Block.clear_basis | not run | - |  |
| 389 | OBC_Projected_ANN_Block._stack_physical_write | not run | - |  |
| 398 | OBC_Projected_ANN_Block.coefficient | not run | - |  |
| 402 | OBC_Projected_ANN_Block.forward | not run | - |  |

##### model_augmentation/fit_systems/orth_projection.py

| line | name | trace | rows | unexecuted branch arcs |
|-|-|-|-|-|
| 24 | OrthProjectionPenalty (class) | class | - |  |
| 54 | OrthProjectionPenalty.__init__ | not run | - |  |
| 61 | OrthProjectionPenalty.penalty_of_field | not run | - |  |
| 70 | OrthProjectionPenalty.forward | not run | - |  |
| 76 | SSE_Interconnect_OrthLoss (class) | class | - |  |
| 93 | SSE_Interconnect_OrthLoss.loss | not run | - |  |

##### model_augmentation/fit_systems/pre_encoder.py

| line | name | trace | rows | unexecuted branch arcs |
|-|-|-|-|-|
| 23 | convert_to_same_type | not run | - |  |
| 37 | System_data_with_x (class) | class | all |  |
| 38 | System_data_with_x.__init__ | not run | - |  |
| 43 | System_data_with_x.to_hist_future_data | not run | - |  |
| 104 | System_data_norm_with_x (class) | class | all |  |
| 119 | System_data_norm_with_x.__init__ | not run | - |  |
| 124 | System_data_norm_with_x.transform | not run | - |  |
| 158 | System_data_norm_with_x.inverse_transform | not run | - |  |
| 193 | linear_encoder_init (class) | class | all |  |
| 194 | linear_encoder_init.__init__ | not run | - |  |
| 281 | linear_encoder_init.forward | not run | - |  |
| 306 | linear_encoder_init_aug (class) | class | all |  |
| 321 | linear_encoder_init_aug.__init__ | run (pipeline) | all | 369->370, 371->382, 383->384, 383->399, 384->383, 384->385, 387->388, 387->393, 400->401, 400->402, 407->408, 407->411, 422->424, 428->437, 443->467, 455->461, 470->474 |
| 488 | linear_encoder_init_aug.forward | run (pipeline) | all | 498->499, 508->509, 518->519 |
| 527 | SS_pre_encoder (class) | class | all |  |
| 536 | SS_pre_encoder.__init__ | not run | - |  |
| 571 | SS_pre_encoder.init_nets | not run | - |  |
| 587 | SS_pre_encoder.loss | not run | - |  |
| 600 | SS_pre_encoder.n_step_error | not run | - |  |

##### model_augmentation/fit_systems/ps2_opening.py

| line | name | trace | rows | unexecuted branch arcs |
|-|-|-|-|-|
| 37 | PS2Spec (class) | class | all |  |
| 59 | calibration_split | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 224-lowpass 224-noisefree 225-lowpass 227-lowpass |  |
| 69 | PS2Opening (class) | class | all |  |
| 70 | PS2Opening.__init__ | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 224-lowpass 224-noisefree 225-lowpass 227-lowpass | 74->75, 79->80 |
| 87 | PS2Opening.for_pipeline | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 224-lowpass 224-noisefree 225-lowpass 227-lowpass |  |
| 98 | PS2Opening.done | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 224-lowpass 224-noisefree 225-lowpass 227-lowpass |  |
| 101 | PS2Opening.after_step | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 224-lowpass 224-noisefree 225-lowpass 227-lowpass | 102->103, 118->119 |
| 130 | PS2Opening.summary | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 224-lowpass 224-noisefree 225-lowpass 227-lowpass |  |
| 135 | PS2Opening.summary_json | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 224-lowpass 224-noisefree 225-lowpass 227-lowpass |  |
| 139 | PS2Opening._device | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 224-lowpass 224-noisefree 225-lowpass 227-lowpass |  |
| 143 | PS2Opening._start | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 224-lowpass 224-noisefree 225-lowpass 227-lowpass | 152->155 |
| 161 | PS2Opening._sample | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 224-lowpass 224-noisefree 225-lowpass 227-lowpass |  |
| 183 | PS2Opening._residual | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 224-lowpass 224-noisefree 225-lowpass 227-lowpass |  |
| 191 | PS2Opening._xa | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 224-lowpass 224-noisefree 225-lowpass 227-lowpass |  |
| 194 | PS2Opening._s1_step | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 224-lowpass 224-noisefree 225-lowpass 227-lowpass |  |
| 207 | PS2Opening._cal_unexplained | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 224-lowpass 224-noisefree 225-lowpass 227-lowpass |  |
| 213 | PS2Opening._fit_map | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 224-lowpass 224-noisefree 225-lowpass 227-lowpass | 244->246, 246->247, 249->252 |
| 220 | PS2Opening._fit_map.snap | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 224-lowpass 224-noisefree 225-lowpass 227-lowpass |  |
| 226 | PS2Opening._fit_map.ee | run (pipeline) | 221-lowpass 221-noisefree 222-lowpass 224-lowpass 224-noisefree 225-lowpass 227-lowpass |  |

##### model_augmentation/fit_systems/white_box_models.py

| line | name | trace | rows | unexecuted branch arcs |
|-|-|-|-|-|
| 9 | Discrete_White_Box_Model (class) | class | all |  |
| 10 | Discrete_White_Box_Model.__init__ | not run | - |  |
| 18 | Discrete_White_Box_Model.forward | not run | - |  |
| 33 | Discrete_White_Box_Model.f | not run | - |  |
| 36 | Discrete_White_Box_Model.h | not run | - |  |
| 39 | Discrete_Cascaded_Tanks (class) | class | all |  |
| 40 | Discrete_Cascaded_Tanks.__init__ | not run | - |  |
| 57 | Discrete_Cascaded_Tanks.init_model | not run | - |  |
| 66 | Discrete_Cascaded_Tanks.f | not run | - |  |
| 80 | Discrete_Cascaded_Tanks.h | not run | - |  |

##### model_augmentation/systems/cascaded_tanks.py

| line | name | trace | rows | unexecuted branch arcs |
|-|-|-|-|-|
| 5 | Cascaded_Tanks (class) | class | - |  |
| 6 | Cascaded_Tanks.__init__ | not run | - |  |
| 14 | Cascaded_Tanks.deriv | not run | - |  |
| 24 | Cascaded_Tanks.h | not run | - |  |
| 27 | Extended_Cascaded_Tanks (class) | class | - |  |
| 28 | Extended_Cascaded_Tanks.__init__ | not run | - |  |
| 37 | Extended_Cascaded_Tanks.deriv | not run | - |  |
| 49 | Extended_Cascaded_Tanks.h | not run | - |  |

##### model_augmentation/systems/gantry_linearization.py

| line | name | trace | rows | unexecuted branch arcs |
|-|-|-|-|-|
| 52 | gantry_linearize_and_discretize | not run | - |  |
| 111 | verify_linearization | not run | - |  |
| 146 | verify_linearization.deriv | not run | - |  |

##### model_augmentation/systems/gantry_ss.py

| line | name | trace | rows | unexecuted branch arcs |
|-|-|-|-|-|
| 150 | build_poly_constants | run (pipeline) | all |  |
| 184 | build_G_matrix_entries | run (pipeline) | all |  |

##### model_augmentation/systems/mass_spring_damper.py

| line | name | trace | rows | unexecuted branch arcs |
|-|-|-|-|-|
| 5 | Mass_spring_damper (class) | class | all |  |
| 6 | Mass_spring_damper.__init__ | not run | - |  |
| 14 | Mass_spring_damper.deriv | not run | - |  |
| 25 | Mass_spring_damper.h | not run | - |  |
| 29 | Msd_2dof (class) | class | all |  |
| 30 | Msd_2dof.__init__ | not run | - |  |
| 41 | Msd_2dof.deriv | not run | - |  |
| 62 | Msd_2dof.h | not run | - |  |
| 66 | Msd_ndof (class) | class | all |  |
| 67 | Msd_ndof.__init__ | not run | - |  |
| 83 | Msd_ndof.cast_parameter_to_array | not run | - |  |
| 90 | Msd_ndof.determine_mask | not run | - |  |
| 98 | Msd_ndof.deriv | not run | - |  |
| 158 | Msd_ndof.h | not run | - |  |
| 163 | Msd_ndof.expand_with_ix | not run | - |  |
| 169 | Msd_ndof_lpf_output (class) | class | all |  |
| 170 | Msd_ndof_lpf_output.__init__ | not run | - |  |
| 184 | Msd_ndof_lpf_output.cast_parameter_to_array | not run | - |  |
| 191 | Msd_ndof_lpf_output.determine_mask | not run | - |  |
| 199 | Msd_ndof_lpf_output.deriv | not run | - |  |
| 245 | Msd_ndof_lpf_output.h | not run | - |  |
| 258 | Msd_ndof_lpf_output.expand_with_ix | not run | - |  |
| 264 | Msd_ndof_input_saturation (class) | class | all |  |
| 265 | Msd_ndof_input_saturation.__init__ | not run | - |  |
| 277 | Msd_ndof_input_saturation.cast_parameter_to_array | not run | - |  |
| 284 | Msd_ndof_input_saturation.determine_mask | not run | - |  |
| 292 | Msd_ndof_input_saturation.deriv | not run | - |  |
| 339 | Msd_ndof_input_saturation.h | not run | - |  |
| 344 | Msd_ndof_input_saturation.expand_with_ix | not run | - |  |

##### model_augmentation/utils/deepSI_corrections.py

| line | name | trace | rows | unexecuted branch arcs |
|-|-|-|-|-|
| 6 | fixed_System_data (class) | class | all |  |
| 7 | fixed_System_data.__init__ | run (pipeline) | all |  |
| 10 | fixed_System_data.to_hist_future_data | run (pipeline) | all |  |
| 30 | fixed_System_data_norm (class) | class | all |  |
| 44 | fixed_System_data_norm.__init__ | run (pipeline) | all |  |
| 51 | fixed_System_data_norm.is_id | run (pipeline) | all |  |
| 55 | fixed_System_data_norm.make_training_data | not run | - |  |
| 61 | fixed_System_data_norm.fit | not run | - |  |
| 69 | fixed_System_data_norm.transform | run (pipeline) | all | 74->75, 78->85 |
| 87 | fixed_System_data_norm.inverse_transform | not run | - |  |
| 101 | fixed_System_data_norm.u_transform | not run | - |  |
| 103 | fixed_System_data_norm.y_transform | not run | - |  |
| 105 | fixed_System_data_norm.u_inv_transform | not run | - |  |
| 107 | fixed_System_data_norm.y_inv_transform | not run | - |  |
| 110 | fixed_System_data_norm.__repr__ | not run | - |  |
| 114 | convert_to_same_type | not run | - |  |

##### model_augmentation/utils/torch_nets.py

| line | name | trace | rows | unexecuted branch arcs |
|-|-|-|-|-|
| 9 | linear_encoder_net (class) | class | all |  |
| 10 | linear_encoder_net.__init__ | not run | - |  |
| 18 | linear_encoder_net.forward | not run | - |  |
| 22 | feed_forward_nn (class) | class | all |  |
| 23 | feed_forward_nn.__init__ | not run | - |  |
| 37 | feed_forward_nn.forward | not run | - |  |
| 40 | identity_init_simple_res_net (class) | class | all |  |
| 41 | identity_init_simple_res_net.__init__ | not run | - |  |
| 58 | identity_init_simple_res_net.forward | not run | - |  |
| 64 | linear_mapping (class) | class | all |  |
| 65 | linear_mapping.__init__ | not run | - |  |
| 78 | linear_mapping.forward | not run | - |  |
| 81 | zero_init_linear_mapping (class) | class | all |  |
| 82 | zero_init_linear_mapping.__init__ | not run | - |  |
| 93 | zero_init_linear_mapping.forward | not run | - |  |
| 96 | zero_init_feed_forward_nn (class) | class | all |  |
| 97 | zero_init_feed_forward_nn.__init__ | not run | - |  |
| 116 | zero_init_feed_forward_nn.forward | not run | - |  |
| 121 | paper_resnet (class) | class | all |  |
| 124 | paper_resnet.__init__ | not run | - |  |
| 165 | paper_resnet.forward | not run | - |  |
| 170 | live_g_feed_forward_nn (class) | class | all |  |
| 178 | live_g_feed_forward_nn.__init__ | run (pipeline) | all | 182->183 |
| 203 | live_g_feed_forward_nn.forward | run (pipeline) | all |  |
| 207 | zero_init_resnet (class) | class | all |  |
| 208 | zero_init_resnet.__init__ | not run | - |  |
| 231 | zero_init_resnet.forward | not run | - |  |
| 235 | unit_variance_feed_forward_nn (class) | class | all |  |
| 236 | unit_variance_feed_forward_nn.__init__ | not run | - |  |
| 263 | unit_variance_feed_forward_nn.forward | not run | - |  |
| 267 | HybridGantryEncoder (class) | class | all |  |
| 274 | HybridGantryEncoder.__init__ | not run | - |  |
| 303 | HybridGantryEncoder.forward | not run | - |  |
| 338 | LinearInitEncoderWrapper (class) | class | all |  |
| 358 | LinearInitEncoderWrapper.__init__ | not run | - |  |
| 407 | LinearInitEncoderWrapper.forward | not run | - |  |
| 434 | positive_default_encoder_net (class) | class | all |  |
| 435 | positive_default_encoder_net.__init__ | not run | - |  |
| 444 | positive_default_encoder_net.forward | not run | - |  |

##### model_augmentation/utils/utils.py

| line | name | trace | rows | unexecuted branch arcs |
|-|-|-|-|-|
| 7 | added | run (pipeline) | all |  |
| 15 | to_tensor | run (pipeline) | all | 16->17 |
| 24 | to_ndarray | not run | - |  |
| 31 | detect_algebraic_loop | run (pipeline) | all | 38->39 |
| 43 | plot_fft_freq | not run | - |  |
| 65 | plot_sys_data | not run | - |  |
| 98 | get_sys_data_dimensions | not run | - |  |
| 105 | determine_sys_data_signal_dimension | not run | - |  |
| 112 | determine_std_T | not run | - |  |
| 122 | determine_std_T_sys_data | not run | - |  |
| 137 | normalize_linear_ss_matrices | not run | - |  |
| 147 | determine_yes_no_query_binary_output | not run | - |  |
| 157 | expansion_matrix | run (pipeline) | all |  |
| 162 | selection_matrix | run (pipeline) | all |  |
| 168 | compute_noisy_y_from_SNR | not run | - |  |


## Part B. Thesis-to-code table

Produced by one Explore subagent (handoff section 14) reading the four sections and the referenced appendix parts
against the active path; I spot-checked the rows marked (v) myself against the code. Where thesis and code disagree,
both are reported and nothing is decided (handoff section 8).

Abbreviations: thesis T2..T5 = `Thesis-writeup/Writing/sections/02_system_baseline.tex` .. `05_setup.tex`,
TA = `99_appendix.tex`. Code: ENTRY = EP; CFG, DATA, MODEL, CTRL, TRAIN, OBCG, OBCD = GD `config.py`, `data.py`,
`model.py`, `controller.py`, `training.py`, `obc_gantry.py`, `obc_diagnostics.py`; SS = MA `systems/gantry_ss.py`,
BLK/IC/CL/PRE/LB/PS/OBC = MA `fit_systems/blocks.py`, `interconnect.py`, `closed_loop.py`, `pre_encoder.py`,
`lbfgs_polish.py`, `ps2_opening.py`, `obc.py`; NETS = MA `utils/torch_nets.py`; TDG =
`Thesis-writeup/Code/Data/tdg_config.m` (data generator, cross-check only).

### Mismatches (both sides stated)

| Thesis | Code | Difference |
|-|-|-|
| TA:448-462 eq:setup_decim (y = r(5k) + (B*(y-r))(5k) for the data) | CFG:33, DATA:172-182 (v) | `thesis_taf_noisefree` point-samples y (`y[::5]`); only the noisy and lowpass modes use the r + filtered servo error rule. Matters for the noise-free axis of the spec |
| TA:455-456 "zero-phase eighth-order Butterworth" | CFG:34-37, DATA:181-182 (v) | `butter(8)` applied by `sosfiltfilt`: magnitude 1/(1+x^16), i.e. squared eighth order (-6 dB at 1.6 kHz) |
| T3:1112-1126 eq:ps2_s2 cost normalised by sum (x - m)^2 | PS:226-227 (v) | `T.var(0)` is the unbiased variance (divides by B-1): code value = (B-1)/B times the thesis ratio; same minimiser |
| T3:1134-1135 keep the S2 parameters with the lowest held-out cost | PS:233-251 (v) | when no check beats the starting cost, `state` stays None and the LAST S2 iterate is kept, not the start |
| T5:199-213 "11 test records" | DATA:79-89 (v) | MANIFEST blocks test-I + test-E give 12 (E5, generated with 1.2 K1, included); the closed-loop report skips E5 (D-227) |
| T4:366-392 eq:obc_overlap, Pi onto ran Phi (10 columns) | OBC:571-582 | the logged `rho_overlap` projects onto ran[Phi Gamma] (11 columns); diagnostic only |
| T4:177-197 eq:obc_offset, column gamma = f_base(beta_lin) - Phi beta_lin | OBCG:332-338 | eleventh column is f_base(beta_lin) itself; same span, same F tilde and correction; only the split of the first 10 coefficients differs by alpha beta_lin (judged equivalent) |

### Section II (T2)

| thesis file:line | statement | code file:line | verdict | note |
|-|-|-|-|-|
| T2:120-125 | u_act = [F_X1 F_X2 F_Y], q_act = [X1 X2 Y] | DATA:147-149, 212-214 | match | `u_total` stage forces, y stage positions |
| T2:129-135 eq:system | CT plant, y_k = q_act(kT_s) + v_k | DATA:185-215 | match (data side) | plant is the data generator |
| T2:136-138 | closed loop with known LTI controller; D_N = {u_act, y} | CTRL:269-356, CL:216-242 | match | residual form |
| T2:160-168 | Garcia lumped model, X = (X1+X2)/2, Theta = (X1-X2)/L_b | SS:120-125, DATA:339-342 | match | |
| T2:174-188 eq:Ptransform | q = P^-T q_act, u = P u_act | SS:120-136, BLK:832 | match | |
| T2:200-202 | small-angle, no Coriolis | BLK:836-838 | match | |
| T2:207-210 eq:eom | M(Y) qdd + C qd + K q = u | BLK:814-875 | match | |
| T2:214-223 eq:mass_matrix | entries of M(Y) | SS:70-84, 150-181 | match | |
| T2:224-241 eq:damping_stiffness_matrices | C, K | SS:89-100, BLK:1042-1061 | match | |
| T2:255-256, TA:19-51 tab:nominal_parameters | nominal values | SS:37-56, CTRL:56-58 | match | |
| T2:260-262 | L_b fixed | BLK:1036 | match | buffer |
| T2:267-268 | LPV in measured Y | MODEL:88 (`Y_op=None`), BLK:846-856 | match | Y at every RK4 stage |
| T2:269-271 | frozen M(0) raises X error up to 3.7x | none | thesis-only | result claim |
| T2:287-295 | self-scheduled LFR | BLK:858-867 | match | |
| T2:312-315 eq:Msplit | M0 + Y M1 + Y^2 M2 | SS:70-84, BLK:1064-1077 | match | |
| T2:318-338 eq:lfr_mass_factorization, eq:lfr_chain, eq:lfr_constant | LFR chain | BLK:836-867, SS:184-234 | match | |
| T2:343-350 eq:delta | Delta(Y) = Y I6 | BLK:862 | match | |
| T2:354-361 eq:lfr_G | G in CT | SS:214-232 | match | |
| T2:363-372 | well-posed iff M(Y) nonsingular, closed-form inverse | BLK:847-855 | match | loop solved in closed form |
| T2:385-387 | theta_base estimated jointly | ENTRY:343, MODEL:96-103 | match | |
| T2:399-405, 420-430 | identifiability argument, rank 10 | BLK:1256-1262 (docstring) | thesis-only | no executed check |
| T2:433-440 eq:phi | ten combinations, order | BLK:1296-1297, 1350-1362 | match | |
| T2:442-445 | estimate theta_base not raw theta | BLK:1325, 1365-1383 (`gauge_section`) | match | raw values are a gauge |
| T2:459-469 eq:lfr_admissibility | m_Sigma J_eff > (L_b^2/4) m_Delta^2; log + tanh map | BLK:1385-1407 | match | bound with 0.99 margin |

### Section III (T3)

| thesis file:line | statement | code file:line | verdict | note |
|-|-|-|-|-|
| T3:202-211, 262-284, 309-311 | one FNN for physical corrections and added states, additive in the state update | MODEL:136-151, 179-183 | match | nw = 6 + n_a |
| T3:286-296 eq:aug_transition | x~ = f_base + f_aug, xbar = g_aug, y = P^T q | IC:143-160, MODEL:132, 180-186, SS:135-136 | match | |
| T3:305-318 | added states free; output map not augmented | MODEL:132, 185-186 | match | |
| T3:331-346 eq:aug_ct | CT model, one RK4 step per T_s, ZOH, M(Y) per stage | BLK:791-798, 814-875, ENTRY:127 (`up_sample=1`) | match | |
| T3:348-354 | well-posedness (todo) | none | thesis-only | |
| T3:356-375 eq:aug_norm | normalisation from training data; velocities by finite difference | DATA:329-375 | match | code-only: +1e-8 on std_x, std_u, ystd (DATA:371-374); velocity is a backward difference with the first row duplicated (DATA:365-366) |
| T3:376-410 eq:aug_norm_maps | xbar not normalised; normalised maps; C^n | BLK:824-825, 870, DATA:378 | match | |
| T3:415-432 | phi_aug: tanh hidden, linear out; inputs 6+n_a states + 3 forces; Y an input | MODEL:136-151, 179, NETS:170-204, ENTRY:345, 357 | match | ANN input is the loop-corrected u (CL:239-241) |
| T3:434-444 | closed-loop simulation because X, Y are marginal | ENTRY:349 | match | |
| T3:475-505 | motivation | none | thesis-only | |
| T3:509-560 eq:aug_two_loops, eq:aug_residual_loop | residual loop, x^c = 0 per window, controller at T_s | CL:167-184, 216-242, 284-285, CTRL:116, 299 | match | Tustin at 4 kHz |
| T3:572-609 eq:aug_cl_step | step order output, error, input, state; normalisation folded into the bank; validation encoder once | CL:139-147, 235-241, 787-797 | match | |
| T3:637-664 eq:mdelta_map | theta_base,j = theta0_j exp(beta_j); bounded m_Delta | BLK:1325, 1337-1346, 1385-1407, BLK:1301 | match | eta0, s from the detuned start |
| T3:666-699 eq:aug_loss | W_r = sigma_y / sigma_e,r,j, mean over windows, n_f, n_y | IC:1419-1426 (v), CTRL:345-353 (v), DATA:222-234 (v) | match | sigma_e population std over k0..N-1, k0 = 30; prior off (ENTRY:344) |
| T3:701-754 eq:aug_encoder, eq:aug_enc_lin | past-only window n+1 = 30, psi = [W^b; W^a] chi + psi~(chi) | MODEL:47-51 (v), 251-261 (v), PRE:488-524 | match | class takes n = na - 1 |
| T3:759-761 | split non-unique | none | thesis-only | |
| T3:773-786 | Adam minibatches, then L-BFGS on fixed windows, kept only if validation improves; selection by mean NRMS_e | IC:1023-1039, LB:97-127, 258-341, CL:667-716 (v), ENTRY:350 | match | code-only: polish windows = stride-scaled grid plus a seeded random subset (LB:263-291) |
| T3:807-879 eq:aug_enc_sim, eq:aug_enc_fit | data-based encoder start: closed-loop baseline simulation from q = P^-T y0, LS over stride-10 windows, intercept in psi~ bias | MODEL:328-464 (v) | match | code-only: windows run to the record end and include the zero-velocity start transient; R^2 >= 0.999 gate (MODEL:410-412, 458-460) |
| T3:882-920 eq:aug_init | beta = 0; physical rows and biases zero; added rows nn.Linear draw; W^a Xavier gain 1 | BLK:1325, NETS:192-201, PRE:428-435, 454-459 | match | |
| T3:935-1010 | PS = S1 + S2, S1 after each V update with its own Adam; calibration split | IC:1044-1046, PS:59-66, 101-128, 146, 156, 161-181 | match | code-only: S1 tau drawn at random per record weighted by length (PS:168-173) |
| T3:1014-1091 eq:ps2_s1, eq:ps2_nu | S1 loss, decoder, nu on fixed set C, stop rule | PS:148-157, 183-211, 108-128 | match | code-only: OBC arm residual uses the re-synced theta_frozen (PS:186) |
| T3:1095-1135 eq:ps2_s2 | S2 pairs, one-step regression | PS:213-255 | see mismatches | |
| T3:1136-1174 eq:ps2_handover | S2 updates all theta_aug; hand-over | PS:216-229, IC:1044-1046 | match | code-only: training Adam moments not reset after PS2 |

### Section IV (T4)

| thesis file:line | statement | code file:line | verdict | note |
|-|-|-|-|-|
| T4:83-127 eq:obc_gyorok | theta_aux = (Phi^T Phi)^-1 Phi^T F | OBC:242-248 | match | code-only: rank-truncated SVD, tolerance S_max max(M,N) eps, column equilibration in the affine arm (OBC:183-188, 205-211, 234-238) |
| T4:157-176 eq:obc_sens | Jacobian w.r.t. beta, 6x10, float64 | OBCG:319-331, OBC:341-364, BLK:1429-1445 | match | |
| T4:177-197 eq:obc_offset | eleventh column | OBCG:332-338 | match (equivalent) | see mismatches |
| T4:211-230 eq:obc_tuples | q = P^-T y, 4th-order difference, x_a = 0, successor exists | OBCG:79-88, 106-171 (v), ENTRY:344 | match | |
| T4:232-291 eq:obc_coef, eq:obc_corr | coefficient on the stack, correction at every step, physical rows only, differentiated | OBCG:286-308, 344-359, OBC:512-531, IC:168-170, 1400-1416 (v) | match | |
| T4:302-322 | re-solved every objective; basis per epoch; rebuild at the selected checkpoint; polish with fixed basis | IC:1136-1149 (v), IC:1406-1409, OBC:501-509, TRAIN:786-792 (v) | match | code-only: also re-solved before every validation (IC:1364-1366); the rebuilt model replaces the selected one without a comparison |
| T4:325-364 eq:obc_eps, eq:obc_recovery | first-order recovery | OBCD:215-244 | thesis-only (training path) | evaluation/diagnostics |
| T4:366-392 eq:obc_overlap | overlap | OBC:571-582 | mismatch (diagnostic) | see mismatches |

### Section V (T5)

| thesis file:line | statement | code file:line | verdict | note |
|-|-|-|-|-|
| T5:45-163 eq:setup_fric, eq:setup_abs | data generation (friction, absorber, RK4 20 kHz, noise) | TDG:28-130 | thesis-only (data generator) | values consistent with TDG |
| T5:150-154 | all data with K1 | CTRL:198-229 | match | |
| T5:181-213 | records, split 18/6/11 | DATA:79-89 | mismatch (count) | see mismatches |
| T5:216-218 | T_s 0.25 ms, windows 0.1 s | ENTRY:69-70, CFG:561-573 | match | |
| T5:231-269 tab:setup_models | M_U, M_OBC, detuning 10 % with seed signs | ENTRY:324-361, 334-337 (v), MODEL:97-99 | match | |
| T5:274-281 eq:setup_bb | black box | none | thesis-only | out of scope |
| T5:302-321 eq:setup_rms | RMS, NRMS_e over k = n+1..N-1, sigma_e population std | CL:756, 801-809, CTRL:347-348, DATA:231 | match | |
| T5:323-342 eq:setup_delta | delta_j, overlap on splits | TRAIN:737-749, OBC:574-575 | match (definition) | logged, not gating; val/test overlap in the evaluation |

### Appendix (TA), parts referenced by II to V

| thesis file:line | statement | code file:line | verdict | note |
|-|-|-|-|-|
| TA:55-124 | M1, M2, G partition, M(Y)^-1 by Horner | SS:79-84, 150-234, BLK:847-855, 1064-1077 | match | C_z, D_zw, D_zu not formed (loop solved in closed form) |
| TA:127-153 | positive-definiteness proof | BLK:1494-1525 | thesis-only (proof) | code-only gate: abort if inadmissible on Y in [-0.30, 0.30] (MODEL:116-123) |
| TA:155-184 | eta0, s; inverse map | BLK:1337-1346, 1409-1422 | match / thesis-only (inverse from matrix entries not in code) | |
| TA:196-216 app:ps_stop | S1 stop rule constants | PS:41-47, 108-126 | match | |
| TA:307-316 eq:setup_K1 | K1 design, Tustin | CTRL:60, 72-119, 299 | match | pipeline discretises at T_s |
| TA:342-429 | multisine, records | TDG:93-105, MANIFEST.csv | thesis-only (data) | |
| TA:448-462 eq:setup_decim | input block mean, y rule, trim | DATA:152-200, CFG:33-56 | match lowpass; mismatch noisefree and filter order | see mismatches |
| TA:494-495 | training loop re-discretises K1 at T_s | CTRL:299 | match | |
| TA:525-532 | one seed per model | ENTRY:328-337, 390-391, 409-410 | match | the seed also sets detuning signs, PS2 streams and the polish subset |
| TA:533-587 tab:hyper | every grey-box hyperparameter | ENTRY:70, 127, 137, 177, 186, 190, 311-312, 345-349, 357, CFG:573, BLK:1301, PS:39-55, TRAIN:874 | match (all grey-box rows) | black-box rows out of scope |

### Code-only on the active path (changes numbers or gates the run, not stated in the thesis)

- Initial (pre-training) validation enters selection; ties (`>=`) go to the later checkpoint (IC:902, 977-978).
- Validation every 1300 updates: the last 33400 mod 1300 = 900 Adam updates are never validated, so cannot be selected
  (IC:953-955, 1061; count assumes 4754 windows per record x 18 = 85572, 167 updates per epoch).
- With OBC the final model is the rebuilt-basis model even when its score is worse than the selected checkpoint's
  (IC:1148-1149, TRAIN:942-958).
- L-BFGS determinism probe with eager fallback; on OOM or exception weights restored and the polish counts as not run
  (LB:666-741).
- OBC rank policy: on rank loss keep the previous basis, abort after two consecutive losses (OBC:460-478).
- OBC coefficient solved in float64, then cast to the pipeline dtype (OBC:526-531; float64 here, so a no-op).
- Batch order of Adam: deepSI `default_rng(0)` shuffle, independent of the run seed (see Part C).

## Part C. Equivalence hazards

What would make a faithful rewrite give different numbers, for the layered harness (forward, loss and gradients,
seeded short training, final metrics per split; CPU, float64).

### C1. Every RNG use in the four code locations

Position: **R** = on the required path and changes numbers; **R-inert** = on the required path, consumes nothing that
matters; **D** = diagnostic only (never changes a saved number); **N** = not reached by the spec.


| file:line | RNG | position | what depends on it |
|-|-|-|-|
| EP:336 | `np.random.default_rng(seed).choice([-1, 1], 10)` | R | detuning signs (detuned rows only) |
| EP:390-391 | global `np.random.seed`, `torch.manual_seed` before data | R-inert | nothing draws between here and EP:409 (data noise DATA:290 is off, snr None); keep it anyway, it is cheap |
| EP:409-410 | global seeds before `build_model` | R | every draw below from the GLOBAL torch stream, in this order |
| NETS:186-190 via MODEL:143 | `nn.Linear` defaults of the live g_aug MLP (3 layers), drawn at float32, cast to float64 at MODEL:301 | R | ANN weights; physical rows and biases zeroed after the draw (NETS:197-201), added-state rows keep it |
| PRE:376-377 via MODEL:251 | `uniform_` of `Wb_psi_y`, `Wb_psi_u` at float64 | R (stream position only) | the VALUES are overwritten by `least_squares_start` (MODEL:445-446), but the draw advances the stream; skipping it changes W^a and the encoder net |
| PRE:432-433 | `xavier_uniform_` of one [W_y W_u] block at float64 | R | W^a |
| PRE:446-453 | `nn.Linear` defaults of the encoder correction net, float32, cast by `.to(DTYPE_PT)` (MODEL:261) | R | hidden layers kept; final physical rows overwritten (MODEL:447-449), added rows zeroed (PRE:458-459) |
| PS:148-149 (first `after_step`, i.e. after Adam update 1) | `nn.Linear` defaults of the PS2 decoder, float32 then cast | R (PS2 rows) | decoder hidden layer; last layer zeroed (PS:150). Drawn from the global stream wherever it stands after the four draws above: anything inserted between EP:409 and the first update that draws from the global torch stream changes it |
| IC:600 (`Dtype_DataLoader`) | `np.random.default_rng(0)`, `shuffle(ids)` once per pass (deepSI `My_Simple_DataLoader.__iter__`) | R | Adam batch order; seed 0 regardless of `cfg.seed`; one shuffle per epoch pass, in place |
| PS:83 | `default_rng(spec.seed)` | R (PS2 rows) | S1 minibatches (record choice weighted by length, tau uniform) |
| PS:157 | `default_rng(seed + 321)` | R (PS2 rows) | fixed calibration set C |
| PS:218 | `default_rng(seed + 1000)` | R (PS2 rows) | S2 snapshot pairs |
| PS:230 | `torch.Generator().manual_seed(seed)`, `randint` | R (PS2 rows) | S2 minibatch indices |
| LB:288 | `default_rng(seed).choice` of the polish windows | R | the fixed L-BFGS batch (taken because n_used < n_built in both smoke and real runs) |
| TRAIN:402 | `default_rng(seed).choice` of 256 validation windows (nf probe) | D | printed nf-window RMS only; local generator, no global state |
| DATA:290 | global `np.random.normal` output noise | N | snr None |
| MA `fit_systems/augmented_dynamics.py:293, 355` | seeded torch Generators | N | module not on the path (Part A) |
| MA `systems/gantry_linearization.py:122` | `default_rng(42)` | N | linearisation is used by `encoder_init='linear_map'` only |
| MA `systems/mass_spring_damper.py:26`, `utils/utils.py:173` | global `np.random.normal` | N | benchmarks / noise helper |
| CM `rezero_gate.py:95` | `nn.init.normal_` | N | only under env `ANN_REZERO_GATE` (MODEL:157) |
| deepSI `systems/system.py:54` | `secrets.token_urlsafe` | R-inert | `unique_code`, overwritten by the run id (TRAIN:842-843); file names only |

No diagnostic on the required path draws from a GLOBAL stream before training ends: the nf probe uses its own
generator, the post-run report and diagnostics run after the final checkpoint is written (TRAIN:960-1005). Removing the diagnostics therefore cannot shift any RNG, with one condition: the clean code must create the
same modules in the same order between EP:409 and the first Adam update (MLP, encoder W^b draw, W^a, encoder net,
then the PS2 decoder at update 1).

### C2. State mutated as a side effect

- **PS2 S2 overwrites the ANN parameters inside `fit()`** (PS:248-253, called from `after_step`, IC:1045-1046) while the
  training Adam keeps its moments; the next Adam step uses moments from before the overwrite. PS2 also sets
  `p.grad` of encoder parameters temporarily and restores them (PS:200-205).
- **PS2 re-syncs the OBC coefficient** (`_sync_prediction_state_for_validation`, PS:186, 228) between updates, writing
  `hfn.obc_correction.theta_frozen` and `vbar` and `obc_validation_diagnostics`. The next `loss()` overwrites them
  (IC:1409), so training is unaffected, but any prediction made before the next objective sees the re-synced value.
- **The encoder pre-fit builds the first OBC basis** (`_prefit` calls `_sync_prediction_state_for_validation`, MODEL:363)
  before `fit()`; `fit()`'s initial validation then finds `obc.basis` set and does not rebuild (IC:1353). Same basis
  point (theta^0), but the build happens there, not in fit.
- **`fit()` ends with `checkpoint_load_system('_best')`**, `self.__dict__ = torch.load(...)` (IC:1139): the returned
  system is an unpickled copy (exact in float64), its simulator's compiled pair is dropped (CL:535-538) so the polish
  is eager, `obc`, `obc_reference`, `training_phase` are carried across by stash (IC:1514-1527), and the loss history
  is restored afterwards (IC:1136-1143). Then the basis is rebuilt at the restored parameters and `_best` re-saved
  (IC:1148-1149). A rewrite that keeps the best weights in memory instead must reproduce exactly this state.
- **OBC refresh is keyed on `floor(epoch_counter + 1e-6)`**, where `epoch_counter` is a float accumulating
  `1/N_batch_updates_per_epoch` per update (IC:1058, 1406-1407). An integer epoch counter refreshes at the same updates
  unless the float sum crosses an integer one update early or late; at 167 updates per epoch and 33400 updates the
  accumulated error (about 1e-12) is far inside the 1e-6 guard, so equivalent in practice, but keep the rule.
- **Validation order inside `validation()`** (IC:881-912): sync coefficient, selection value, probes (nf probe runs an
  extra encoder + rollout under `no_grad`; read-only), append histories, save checkpoint on `bestfit >= Loss_val`.
- **The nf probe swaps `simulator.indexer.record_rows/record_weights`** for one `make_training_data` call and restores
  them (TRAIN:391-412); it also overwrites `simulator.last_window_counts`. `loss_stats` accumulates inside `loss()`
  (IC:1430-1431) under the graph; it does not enter the loss.
- **Lazy initialisation on first forward**: `Interconnect.init_forward` (IC:125-127), connection matrices re-homed to the
  input device/dtype (IC:137-141), `out_gate` re-homed (BLK:76-77), controller bank re-homed (CL:269-270),
  `self.nb` written per call (IC:123). The first forward is the eager initial validation, which is why compile works.
- **Polish**: `_quiet_probes` disables probes and `loss_stats`; parameters restored on rejection with `_resync`
  (LB:162-178); D-266 re-solve before every polish validation.
- **`bestfit` and `fit_sys.bestfit`** are rewritten to the final event's score after the polish (TRAIN:942-958); the
  `.npz` meta `bestfit` is what `eval_common.checkpoint_status` returns as `ref_score`.

### C3. Operation order that changes float results

- **Interconnect summation order** (IC:143-160): `xp` is accumulated as `+= matmul(connection, signal)` in
  `order_output_signal_computation`, physical and ANN contributions added in that order, then
  `xp - obc_correction(x, u, obc_pass)` (IC:168-170). A rewrite as `f_base + f_aug - corr` in another order or with the
  identity matrices removed (`expansion_matrix`, `selection_matrix` are real matmuls) changes the last bits.
- **Closed-loop step** (CL:232-241): `y = output_only(x)`, then `baddbmm(bmm(M_error, e), M_state, xc)`, then
  `hfn(x, u_t + u_fb)`. The normalisation is folded into the bank matrices (CL:139-147); unfolding it changes bits.
- **Loss** (IC:1419-1426): weighted `((y_pred - yfuture) * w[:, None, :]).square().mean()`, one reduction over the stack;
  `mse_loss` only when `w is None`. Weights `ystd / servo_sigma(e_train, k0)` with `np.std` (ddof 0) over k0..N-1
  (CTRL:345-353, DATA:222-234), computed in float64.
- **Validation batching** (CL:779-809): records grouped by post-k0 length and rolled out as one batch per group; the
  aggregate is `sqrt(mean(mse_per_channel))`, the record mean weighted by full record length, NRMS_e =
  `mean(per_channel / val_sigma_e)`. Different grouping changes BLAS blocking (documented 2.6e-10 relative in float32).
- **Hoisted physics structure**: `pass_mats()` once per objective (IC:1390-1393) and `tangent_pair` after `set_frozen`
  (IC:1409-1416); computing them per step is the same math in a different order.
- **Encoder least squares** (MODEL:415-464): centred, column-scaled `torch.linalg.lstsq(..., driver='gelsd')` with
  `rcond = eps * max(N, m)`. The 2 s smoke reported rank 141 of 180 at condition 9.1e15: the solution is a
  rank-truncated one and the truncation decision depends on singular values near the cutoff, so any change in the
  pre-fit rollout, window assembly order (record groups by length, MODEL:378-405) or regressor scaling can move the
  rank and change W. Highest-risk single item for the harness.
- **OBC basis and coefficient**: float64 `jacfwd` stack in chunks of `obc_ref_chunk` (OBC:341-364), column scaling for
  the affine arm, SVD rank tolerance (OBC:183-238), coefficient `V S^-1 U^T F` (OBC:242-248). Chunk size is declared
  memory-only, but chunked jacfwd evaluates the same rows, so it is safe.
- **Normalisation** (DATA:329-375): statistics over the concatenated training records, `std + 1e-8`, velocities by a
  backward first difference with the first row duplicated; `Cd_norm = Cd * std_x / ystd`.
- **Initial weights**: networks are drawn at float32 and cast to float64 (MODEL:301, 261), while W^b and W^a are drawn
  directly at float64 (PRE:376-377, 432). Building an `nn.Linear` at float64 directly gives different initial values
  from the same seed.
- **Detuning** `float(1.0 + 0.10 * s)` and `combo * combo.new_tensor(detune)` (EP:337, MODEL:97-99), then the bounded
  m_Delta coordinate built from the detuned start (BLK:1337-1346).
- **Library versions**: torch Adam (foreach or not), LBFGS strong-Wolfe, deepSI windowing (`to_hist_future_data`),
  scipy `sosfiltfilt`, `loadmat` are part of the numbers; the harness should pin the env.

### C4. GPU and compile-only paths

- `torch.compile(..., backend='inductor', mode='reduce-overhead', fullgraph=True)` of `hfn` and `hfn.output_only` for
  the TRAINING rollout only (CTRL:329-341), selected per call when `ufuture.is_cuda and torch.is_grad_enabled()`
  (CL:584-604), with `cudagraph_mark_step_begin` and output clones. Not bit-identical to eager (max |dg| 7.5e-10, job
  80610). Validation, the nf probe, PS2 residuals (no_grad), the pre-fit and the polish (simulator unpickled, compiled
  pair dropped) are eager.
- Device moves: `fit_sys.cuda()` around `fit()` (MODEL:483-484, 518-519), the pre-fit (MODEL:343-349), the polish with
  `torch.cuda.empty_cache()` (TRAIN:776-782, 817-818); batch `.cuda()` in fit (IC:998-999); no CPU flip for validation
  with a simulator attached (IC:879), so validation runs on the GPU in the real run and on the CPU in the smoke.
  CPU and GPU reductions differ, so the golden outputs must be produced on CPU (as the harness plans).
- `static_pass_buffers=False` and `checkpoint_chunk=0`: `StaticPassBuffers` (OBC:44-116) and the checkpointed rollout
  branch (CL:295-311) are off in every spec row.
- `_note_update_time` (CL:608-638) prints only.

### C5. Dtype switches

- Everything float64 (`use_f64=True`): data cast at DATA:214, `Dtype_DataLoader` keeps float64 and leaves integer
  `ctrl_ix` alone (IC:594-601), the bank at `cfg.dtype_pt` (CTRL:299). Integer index arrays must stay integer.
- OBC: float64 deep copy of the physics block for the basis (OBCG:312-316), reference field cast to the ANN's dtype
  (OBCG:353-354), coefficient cast to the pipeline dtype (OBC:526-531): all float64 here.
- Network construction at float32 then cast (C3, initial weights).
- The `.npz` and `config.json` are float64/JSON; `theta_frozen`, `vbar`, `E` are float64 buffers in the checkpoint.

### C6. Interface the evaluation scripts use (must stay loadable)

Read in `Thesis-writeup/Code/Evaluation/` (`eval_common.py`, `eval_greybox.py`, `eval_overlap.py`,
`eval_negation.py`, `eval_frf.py`, `eval_blackbox.py`) and `scripts/gantry/thesis-results/inspect_checkpoint.py`,
which they import.

- **sys.path and package names**: `eval_common.py:19-21` puts the repo root, `scripts/gantry` and
  `scripts/gantry/thesis-results` on the path; the scripts import the packages `gantry_dynamic`, `model_augmentation`
  and (through `closed_loop_report`) `common`.
- **Imports**: `gantry_dynamic.config.RunConfig`, `record_trim`; `gantry_dynamic.data.load_datasets`,
  `compute_normalization`, `_load_record`, `servo_sigma`, `traj_dir`, `_record_at_fs_new`, `_load_u`, `_resample_u`,
  `record_files`; `gantry_dynamic.model.build_model`; `gantry_dynamic.controller.build_closed_loop` (keywords
  `train_files`, `val_files`, `val_data`, `verbose`, `e_train`, `e_val`), `build_controller_bank`, `y_op_for`;
  `gantry_dynamic.training.load_checkpoint`; `gantry_dynamic.closed_loop_report._without_learned`, `_true_free`
  (this module imports `common.oracle` and `data.load_mat_aug`, so both are needed by the evaluation even though
  Dirk excluded the report from training); `gantry_dynamic.obc_gantry.build_reference_set`, `make_float64_block`,
  `make_basis_builder`; `model_augmentation.fit_systems.closed_loop.closed_loop_free_run_rms_batch`;
  `model_augmentation.fit_systems.obc.build_stacked_sensitivity`; `model_augmentation.fit_systems.blocks.Static_ANN_Block`.
- **Config round trip**: `eval_greybox.run_config` rebuilds a `RunConfig` from `config.json` by upper-casing every field
  name (so every field kept in the clean `RunConfig` must keep its `config_json_dict` key), tuples for `ann_route_ix`
  and `lbfgs_freeze`, and reads `NA_NB`, `NF`; `discover_runs` reads `TRUNCATE_S`, `N_ITS`, `run_id`, `OBC`, `NX_ANN`,
  `COMBO_INIT_DETUNE`, `PS2`; `checkpoint_status` reads `EPOCHS`. The build sequence it replays is EP's: seed, load,
  normalise, seed, `build_model`, `build_closed_loop` (no pre-fit; the checkpoint carries the encoder).
- **Files**: run folder `<repo>/simulations/gantry_subnet/{mode}_{encoder_init}/<run_id>/` (`save_dir`, CFG:603-606;
  `eval_common.py:65-66` expects `thesis_taf_lowpass_noise_data_lsq`); `gantry_ckpt_<rid>.pt` with top keys `hfn`,
  `encoder` (and `optimizer` unless dropped) and `.npz` with `done_epochs`, `bestfit`, `bestfit_selected`, `elapsed`,
  `hp`, `orig_run_id`, `lbfgs_outcome`, `optimizer_state_dropped`, `start_phase`, `resumed_from`, `resumed_at_epoch`,
  `ps2_summary` (TRAIN:987-1005; seen in the 221 smoke checkpoint). An unfinished run falls back to deepSI's whole-object
  pickle `<Class>_<rid>_best.pth` (`eval_common.py:113-122`), which unpickles classes by module path
  (`model_augmentation.fit_systems.interconnect.SSE_Interconnect_Composed`, `...closed_loop.ClosedLoopSimulator`,
  `ControllerBank`, `WindowControllerIndex`, the blocks, `gantry_dynamic.obc_gantry.GantryOBCCorrection`) and reads
  `batch_counter`, `epoch_counter`, `bestfit`. The final campaign finished, so only the pair format is needed for it.
- **state_dict names** (221 smoke checkpoint, n_a = 2): `hfn.connected_blocks.0.{free_params, mh, alpha, beta, gamma_,
  N0, N1, N2, Ax, Bw, Bu, A_combined, K_mat, C_mat, P_mat, std_x, std_x_1d, std_u, x_mean, u_mean, params_init, Lb,
  combo_init, Lambda_combo, coordinate_version, m_diff_eta0, m_diff_scale}`, `connected_blocks.1.{C, D}`,
  `connected_blocks.2.net.net.{0,2,4}.{weight,bias}`, `obc_correction.{E, vbar, theta_frozen}` (OBC arm only);
  `encoder.{Wb_psi_y, Wb_psi_u, Wa_psi_y, Wa_psi_u, net.{0,2,4}.{weight,bias}}`. Block order (physics 0, output 1,
  ANN 2) is fixed by `add_block` order (MODEL:133-151); `load_checkpoint` checks names and shapes exactly
  (TRAIN:158-195) and reads `coordinate_version` and `free_params`. `infer_arch_from_hfn` parses `*.net.net.<i>.weight`.
- **Object attributes read**: `fs.hfn.connected_blocks`, `fs.encoder`, `fs.na`, `fs.nb`, `fs.norm`, `fs.simulator`,
  `fs.hfn.obc_correction.{theta_frozen, vbar}`; physics block `free_params`, `params_init`, `PARAM_NAMES`,
  `COMBO_NAMES`, `pass_mats()` returning the 10-tuple `(K, C, A, mh, alpha, beta, gamma, N0, N1, N2)`, `N_op`, `d_op`,
  `Y_op` (frozen-scheduling path), `estimation_parameters()`, `combinations_from_free`, `transition_from_free`;
  `Static_ANN_Block.out_gate` (clamped model, `inspect_checkpoint.py:84-94`); reference set `Z_field`, `Z_step`, `n`,
  `record_ix`, `record_names`, `sample_ix`, `x_phys_next_rec`; `Norm` fields `Cd_norm`, `std_u`, `std_x`, `u_mean`,
  `x_mean`, `ystd`; `DataBundle` `train_files`, `val_files`, `test_files`, `val_list`, `<split>_list`, `e_<split>`,
  `val_ckpt_data`; `RunConfig` `hp`, `seed`, `nx_ann`, `nx_phys`, `obc`, `obc_ref_chunk`, `ann_route_ix`,
  `controller_gain`, `dtype_pt`, `ts_new`.
- **Behaviour the evaluation relies on**: `load_checkpoint(..., start_phase='lbfgs')` restores the saved OBC buffers
  and no re-sync follows (`eval_greybox.py:117-118`); `_load_record(f, cfg, folder=...)` for the twins.
