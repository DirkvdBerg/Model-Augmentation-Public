# Keep audit, merged (D-284 step 4b, amendment 8)

2026-10-30. Inputs: `tasks/clean-code/keep-audit-claude.md` (CA IDs) and `tasks/clean-code/keep-audit-codex.md` part A
(X IDs), written independently on snapshot `final` (manifest `98d23ae2...1f99`) by `06c-audit-criteria.md`. Merged
IDs `K<n>`. Rule of the merge: both keep stands; same change takes the more concrete text; a disagreement goes to
Dirk with one line per side and is not settled here.

## 1. Totals
- **Site lists**: identical. Both list the same 273 sites (24 files, 31 classes, 209 callables, 9 lambdas); no site
  appears in only one audit. Both list all 28 speed sites.
- **Per site** (section 5): 190 keep in both; 42 non-keep in both; 7 differ only because one audit also marked a
  call site of an agreed change; 11 differ where Claude changed its verdict to Codex's after reading (K41 to K45);
  23 differ on a real disagreement (section 3).
- **Changes**: 47 K items. 26 agreed (21 found by both, 5 found by Codex that Claude accepts after reading); 21
  disagreements (19 Claude only, 2 Codex only). All are bitwise; neither audit
  proposes a rounding item, so the rounding list is empty.
- **Jan's code**: K20 (both; the gate in `Static_ANN_Block`), K28 (Claude; `init_forward`), K43 (agreed; the CHANGED
  discovery line in `init_forward`), K46, K47 (Codex; `Interconnect.forward`, `determine_signal_ix`,
  `convert_signal_name_to_ix`).
- **Correction to the Claude audit**: the unused test records are 11 per data set (6 test-I, 5 test-E with
  controller K1), not 12; Claude counted the manifest blocks without the K1 filter. Codex's count holds.
- **Coverage**: both read the same `combined.json` (87.6 % statements, 62.5 % branches) and give every miss a
  disposition; they differ only on the three checks of K19, K22, K39.

## 2. Agreed changes (26), ranked by gain
Change text: the more concrete of the two. Gain: what each removes; wall times are unmeasured unless stated.

| K | Change | Sources | Gain | Label | Layer |
|-|-|-|-|-|-|
| K2 | `load_datasets` loads train and val only; `DataBundle` without `test_list`, `test_files`, `e_test` and the redundant `val_list` field (local `val_list` kept for `val_ckpt_data`); `record_files` keeps returning test (`Clean/blackbox`); load summary and clean harness readers adapted | CA7, X-S25 | 11 of 35 record loads (loadmat, block mean, filter) per run | bitwise | L1-2, CLI smoke |
| K1 | delete `closed_loop_report.py`, `free_from_combinations`, `recover_combinations`, `estimation_parameters`; adapt clean harness consumers; all state_dict buffers stay | CA1 to CA3, X-M13 | 1 file, 5 callables, about 86 lines | bitwise | L1-2 (harness L1/L4 consumers adapted) |
| K15 | `GantryReferenceSet(Z_step, Z_field)`: no `record_ix`, `sample_ix`, `record_names`, `x_phys_next_rec` and their rec/samp/xnr construction; no `fit_sys.obc_reference` and its `_UNPICKLABLE` entry; `n` inlined as `int(reference.Z_step.shape[0])` in the print; tuple eligibility `k = arange(2, len(u) - 3)` unchanged | CA20, CA24, X-S18, X-M24 | per OBC run 48N bytes of successors and 16N of indices not built; about 15 lines | bitwise (harness capture key dropped, `DROPPED_IN_CLEAN` with K15) | L1-2, L3 OBC |
| K3 | unit-gain controllers: `use_gain` gone from `y_op_for` and `build_controller_bank`, the gain refusal merged into `y_op_for`, one lookup per record, `controller_ss` without `gain` (identity scales); the `abs(g - 1) > 1e-12` refusal and the other metadata messages stay | CA10 (part), X-S20 | one lookup per record instead of two; about 15 lines | bitwise (x * 1.0 == x) | L1-2 |
| K13 | `GantryStateBlock` without `Y_op` and the frozen `N_op`/`d_op` arm, and the frozen-Y guard in the tangent | CA4, X-M14 | about 12 lines; hfn ops unchanged (estimate) | bitwise | L1-2 |
| K20 | `Static_ANN_Block` without `out_gate` and its multiply (after K1 removes its only writer); `net.*` keys unchanged | CA5, X-S17 | hfn -1 multiply per step, forward and backward (872 to 871, 242 to 241 ops, estimate) | bitwise, touches Jan's code | L1-2, L3 PS2 |
| K17 | `make_basis_builder.build` returns J only (no timer, its seconds were discarded), no `device` override (`dev = vbar64.device`); `build_stacked_sensitivity` infers `dev = vbar.device`; `refresh`: `J = self.basis_builder(vbar64)` | CA22, X-S21, X-M31 | two clock reads per refresh, two unused parameters | bitwise | L1-2, L3 refresh |
| K18 | `build_stacked_sensitivity` always appends the baseline column; `append_value` and both conditionals gone | CA23 (part), X-M29 | one parameter, two branches | bitwise | L1-2 |
| K14 | per-step matrix fallbacks removed: `nonlinear_function` (`mats is None`), `GantryOBCCorrection.forward` (`obc_pass is None`); direct clean harness calls adapted to `objective_matrices` | CA6, X-S19 | no rollout can rebuild matrices per step; about 4 lines | bitwise | L1-2, L3 PS2 and polish |
| K21 | `closed_loop_free_run_rms_batch` without `return_pred`, `preds` and their branches | CA30 (part), X-M15 | radon C 15 to about 13 | bitwise | L1-2 validation |
| K6 | `build_closed_loop` without `val_data` (and `main`'s keyword) and `verbose`; `build_reference_set` without `verbose` | CA12, CA20 (part), X-M16, X-M17 | 2 parameters, 3 print conditionals | bitwise | L1-2, CLI smoke |
| K7 | `load_checkpoint` without `joint_estimation` (and `main`'s argument); `JOINT_ESTIMATION` in config.json stays | CA16, X-M19 | one unused parameter | bitwise | L3c |
| K8 | `cfg.n_a` instead of `cfg.nx_ann`; `RunConfig.nx_ann` deleted; `NX_ANN`/`N_A` keys and hp keys unchanged | CA14, X-M20 | one alias | bitwise | L1-2, L3c |
| K9 | `save_dir` inlined in `main`: `os.path.join(cfg.out_root, f'{cfg.mode}_{cfg.encoder_init}')` | CA18, X-M21 | one forward | bitwise | L1-2, L3c, CLI smoke |
| K10 | `_load_u` inlined: `_resample_u(d['u_total'], cfg)` in `_record_at_fs_new` | CA9, X-M22 | one forward | bitwise | L1-2 |
| K11 | `_load_record` without `folder`: `os.path.join(traj_dir(cfg), filename)` | CA8 (part), X-M18 | one unused input path | bitwise | L1-2 |
| K16 | `make_float64_block` inlined: `baseline64 = copy.deepcopy(phy).to(torch.float64)` | CA21, X-M23 | one forward | bitwise | L1-2 |
| K23 | `ControllerBank.n_controllers` inlined as `bank.M_state.shape[0]` in the print | CA31, X-M30 | one property | bitwise | L1-2 |
| K24 | `_signal_cone` without `if k in needed: continue` | CA44, X-M33 | 2 lines (never reached) | bitwise | L1-2 |
| K25 | `SSE_Interconnect.simulate` removed; `fs.simulator(fs, x, ...)` in `AugmentedModel.loss` and the PS residual | CA29, X-M26 | one forward (deepSI defines no `simulate`) | bitwise | L1-2, PS2 smoke |
| K26 | encoder start: `u0, ustd, y0, ystd` vectors prepared once before the length groups | CA42, X-S23 | 4 conversions per record | bitwise | L1-2 |
| K41 | `_decimate_y`: module-level `SOS = butter(AA_ORDER, AA_FC_HZ, fs=RunConfig.fs_orig, output='sos')`, reused | X-S22; Claude agrees after reading | one filter design per run instead of one per lowpass record (training, val and the OBC reload) | bitwise (same deterministic design) | L1-2 |
| K42 | `IdentifiableGantryStateBlock.__init__`: `bound = self._m_diff_bound(combo_init)` once, used in `r0` and `scale` | X-S24; Claude agrees after reading | one repeated computation | bitwise | L1-2 |
| K43 | `pass_mats` deleted; `objective_matrices` calls `phy._mats_at(phy.free_params)`; both discovery predicates test `_mats_at`; the `pass_mats` keyword of the rollout stays | X-M25; Claude agrees after reading (Claude had kept it as the discovery marker; the predicate can test `_mats_at`) | one forward | bitwise, touches the CHANGED line of Jan's `init_forward` | L1-2 |
| K44 | `GantryOBCCorrection.expansion_point` removed; `maybe_refresh` takes the beta tensor, `_sync_obc_coefficient` reads it directly; detach stays inside `refresh` | X-M27; Claude agrees after reading | one getter callback | bitwise | L1-2 OBC |
| K45 | `OBCBasis.device` removed; `basis.U.device` at its readers | X-M28; Claude agrees after reading | one property alias | bitwise | L1-2 OBC, GPU smoke |

## 3. Disagreements for Dirk (21)
One line per side. Gain in brackets.

| K | Change | Claude | Codex |
|-|-|-|-|
| K4 | one controller registration: `register_stored_controller(d, filename)` in `controller.py` reads meta.controller and stores (`_register_controller` and its lambda go) [2 callables to 1] | two layers in two modules for one registration; the outer only unpacks and forwards | `_register_controller` decodes the MATLAB scalar fields, `register_stored_controller` holds the supported/unsupported policy; merging obscures the metadata policy |
| K5 | inline `build_controller_bank` into `build_closed_loop` step 1 [one called-once function] | after K3 its body is 5 lines and step 1 is already titled "Controller bank" | keeps it (simplified by K3): it builds the bank of eq:aug_cl_step |
| K12 | one `_load_split(files, cfg)` replaces `_load_record` plus the nested `_split` [2 callables to 1] | a called-once loader plus a nested unzip helper for one task | `_split` is the split rule used three times and `_load_record` real per-record I/O, not forwarding |
| K19 | drop the float64 `TypeError` in `build_stacked_sensitivity` [2 lines] | unreachable: `refresh` always passes `vbar.double()` | actionable guard: refuses a non-float64 beta and names the repair (ML-085) |
| K22 | drop the record/controller-row count check in `closed_loop_free_run_rms_batch` [3 lines] | `validation_error` builds both lists from the same split; no input reaches it | actionable guard: reports both lengths and the fix (ML-065) |
| K27 | inline `mats_and_tangent_from_free` into `tangent_pair` [one method] | its body is one `torch.func.jvp` call, one caller | the jvp is the matrix tangent of eq:obc_corr; inlining obscures the tangent preparation |
| K28 | one lookup of the pass_mats block (`AugmentedModel._mats_ix` passed to the interconnect) [low] | the same lookup runs in `init_forward` and in `AugmentedModel.__init__` | keeps both (K43 changes only the predicate); `init_forward` is Jan's |
| K29 | 8 ClassVars written only to config.json (`controller_gain`, `param_prior`, `lbfgs`, `use_f64`, `obc_rank_rtol`, `obc_ref_successor`, after K7 `joint_estimation`, and the print-only `obc_space`, `obc_refresh`) become literals in `config_json_dict` [about 9 lines] | state that exists only to be written once, like `ANN_ACTIVATION='tanh'` already is | part of the CLI/config.json schema of `RunConfig` |
| K30 | drop the `hp` parameter from `build_model`, `train`, `_adam_phase`, `run_lbfgs_polish`; read `cfg.*` (hp stays as `cfg.hp` in the checkpoint) [4 parameters] | `hp` is always `cfg.hp`: every value is passed twice | keeps the signatures; only `nx_ann` (K8) changes |
| K31 | drop `np.random.seed`/`torch.manual_seed` at `main:94-95` [2 lines] | overwritten at `:109-110` with no draw in between | no proposal; keeps main's seed sequence (it notes the reseed after loading) |
| K32 | drop `route_ix` from `GantryOBCCorrection` and `PS2Opening` (always all states) [about 6 lines] | no thesis row routes otherwise; `row_ix = field_cols = range(6)`, `_added_cols = range(6, 6 + n_a)` | the constructors build the protected-row selector and the added-output map from it |
| K33 | merge `_TrainableGantryStateBlock` into `IdentifiableGantryStateBlock` [one class level] | never instantiated and cannot work alone (it calls `_mats_at` of the subclass) | it is the raw-parameter matrix base (eq:matrices) with persistent buffers |
| K34 | one RK4 step in `_rk4`, `_rk4_with_tangent` (`up_sample` always 1) [low] | the loop runs `range(1)`, `h = Ts / 1` | the RK4 recurrence stays; `up_sample` is an hp/config key |
| K35 | inline `ValidationHistory.hist_arrays` into `record` [one method] | called once | it is the val_hist.npz layout; inlining makes recording murkier |
| K36 | split `lbfgs_polish`: `_determinism_probe(...)` for the compiled probe and eager fallback [E 31 to about D] | the probe/fallback is infrastructure, not a thesis step (06c: split unless each branch maps to a thesis step) | every branch is a named polish phase or an F5 solver guard; a split only lowers the grade |
| K37 | drop the unread encoder attributes `nu`, `ny`, `na`, `nb` [4 lines] | stored, never read | keeps the constructor (widths, initialisation draws); does not address the attributes |
| K38 | `kw['nf']`, `kw['stride']` instead of `kw.get` with unused defaults in `augment_training_data` [none] | the defaults are never used | part of the deepSI extra-array hook |
| K39 | drop the Adam-eps check after `init_model` [5 lines] | unreachable: the eps is passed two lines above and deepSI forwards it | actionable guard naming the configured and observed eps (ML-016) |
| K40 | `Norm` without `Dd_np`; `build_model` passes `gantry_ss.Dd.numpy()` [low] | D = 0 is a model constant, not a normalisation value | `Norm` aggregates the normalisation and output maps (eq:aug_norm_maps) |
| K46 | remove the lazy initialisation from `Interconnect.forward` (lines 75-77) [3 lines] | Jan's `forward` would then fail when called first, a call-order dependency without a message; the lines are Jan's | never reached: every rollout calls `output_only` first, which keeps the initialisation |
| K47 | `determine_signal_ix` and `convert_signal_name_to_ix` without the implicit `None` exit (dict lookup) [none] | rewrites two Jan bodies for no runtime or clarity gain; no line is removed, only an exit arc | unsupported input silently returns None; a lookup that raises is the supported mapping only |

## 4. Speed sites (28)
Both audits give every problem the same disposition: no new proposal on the training update; the measured round
gains stand (M1 validation 166 to 47 s on 221, 30 to 14 s on 226; M19 layer 3 on 221 548 to 413 s; M11, M12
hfn 882 to 872 and 252 to 242 ops); M5 (reverted on the measured polish drift), M7, M9, M10, M13, M15, M16, M25, M28
stay rejected and M18 deferred to step 5. Proposals attached to speed sites: P1 and P9 K20; P5 K14; P6 and P19 K21;
P12 K17, K18; P16 K36 (disagreement); P17 K30 (disagreement); P20 K6; P21 K15; P27 K6, K7, K30. P4, P11 (M17) are
judged only by the step-5 server timing.

## 5. Site rows (union of both lists)
"Claude" and "Codex" give each audit's verdict and its K items. Status: `keep (both)`; `both non-keep`; `same K;
call-site edit noted by one audit` (an agreed change that one audit also marked at its caller); `agreed after
reading` (K41 to K45); `disagreement` (section 3).

| site | name | Claude | Codex | K | status |
|-|-|-|-|-|-|
| `Clean/gantry_dynamic/__init__.py:1` | `<file>` | keep | keep | - | keep (both) |
| `Clean/gantry_dynamic/closed_loop_report.py:1` | `<file>` | remove (K1) | remove (K1) | K1 | both non-keep |
| `Clean/gantry_dynamic/closed_loop_report.py:11` | `_without_learned [callable]` | remove (K1) | remove (K1) | K1 | both non-keep |
| `Clean/gantry_dynamic/closed_loop_report.py:34` | `_true_free [callable]` | remove (K1) | remove (K1) | K1 | both non-keep |
| `Clean/gantry_dynamic/config.py:1` | `<file>` | keep | keep | - | keep (both) |
| `Clean/gantry_dynamic/config.py:51` | `y_decimation [callable]` | keep | keep | - | keep (both) |
| `Clean/gantry_dynamic/config.py:58` | `record_trim [callable]` | keep | keep | - | keep (both) |
| `Clean/gantry_dynamic/config.py:64` | `RunConfig` | simplify (K29) | keep | K29 | disagreement (section 3) |
| `Clean/gantry_dynamic/config.py:131` | `RunConfig.__post_init__ [callable]` | keep | keep | - | keep (both) |
| `Clean/gantry_dynamic/config.py:145` | `RunConfig.from_config_json [callable]` | keep | keep | - | keep (both) |
| `Clean/gantry_dynamic/config.py:167` | `RunConfig.obc [property]` | keep | keep | - | keep (both) |
| `Clean/gantry_dynamic/config.py:172` | `RunConfig.nx_ann [property]` | remove (K8) | inline (K8) | K8 | both non-keep |
| `Clean/gantry_dynamic/config.py:177` | `RunConfig.ann_route_ix [property]` | keep (K32) | keep | K32 | keep (both) |
| `Clean/gantry_dynamic/config.py:182` | `RunConfig.mode [property]` | keep | keep | - | keep (both) |
| `Clean/gantry_dynamic/config.py:187` | `RunConfig.combo_init_detune [property]` | keep | keep | - | keep (both) |
| `Clean/gantry_dynamic/config.py:203` | `RunConfig.hp [property]` | keep | simplify (K8) | K8 | same K; call-site edit noted by one audit |
| `Clean/gantry_dynamic/config.py:218` | `save_dir [callable]` | inline (K9) | inline (K9) | K9 | both non-keep |
| `Clean/gantry_dynamic/config.py:224` | `git_provenance [callable]` | keep | keep | - | keep (both) |
| `Clean/gantry_dynamic/config.py:226` | `git_provenance._git [callable]` | keep | keep | - | keep (both) |
| `Clean/gantry_dynamic/config.py:251` | `config_json_dict [callable]` | keep | simplify (K8) | K8 | same K; call-site edit noted by one audit |
| `Clean/gantry_dynamic/controller.py:1` | `<file>` | keep | keep | - | keep (both) |
| `Clean/gantry_dynamic/controller.py:36` | `mass_matrix_at [callable]` | keep | keep | - | keep (both) |
| `Clean/gantry_dynamic/controller.py:54` | `stage_frequency_response [callable]` | keep | keep | - | keep (both) |
| `Clean/gantry_dynamic/controller.py:74` | `normalized_controller_coefficients [callable]` | keep | keep | - | keep (both) |
| `Clean/gantry_dynamic/controller.py:89` | `build_cfb_at [callable]` | keep | keep | - | keep (both) |
| `Clean/gantry_dynamic/controller.py:116` | `controller_ss [callable]` | simplify (K3) | simplify (K3) | K3 | both non-keep |
| `Clean/gantry_dynamic/controller.py:134` | `_stem [callable]` | keep | keep | - | keep (both) |
| `Clean/gantry_dynamic/controller.py:144` | `register_stored_controller [callable]` | merge into _register_controller (K4) | keep | K4 | disagreement (section 3) |
| `Clean/gantry_dynamic/controller.py:150` | `y_op_for [callable]` | merge into one lookup (K3) | simplify (K3) | K3 | both non-keep |
| `Clean/gantry_dynamic/controller.py:165` | `gain_for [callable]` | merge into design_y_for (K3) | merge into y_op_for (K3) | K3 | both non-keep |
| `Clean/gantry_dynamic/controller.py:176` | `build_controller_bank [callable]` | inline (K3, K5) | simplify (K3) | K3, K5 | both non-keep |
| `Clean/gantry_dynamic/controller.py:200` | `build_closed_loop [callable]` | simplify (K6) | simplify (K6, K23) | K6, K23 | both non-keep |
| `Clean/gantry_dynamic/data.py:1` | `<file>` | keep | keep | - | keep (both) |
| `Clean/gantry_dynamic/data.py:28` | `record_files [callable]` | keep | simplify (K2) | K2 | same K; call-site edit noted by one audit |
| `Clean/gantry_dynamic/data.py:40` | `_register_controller [callable]` | merge into register_stored_controller (K4) | keep | K4 | disagreement (section 3) |
| `Clean/gantry_dynamic/data.py:44` | `_register_controller.<lambda>` | merge (K4) | keep | K4 | disagreement (section 3) |
| `Clean/gantry_dynamic/data.py:50` | `DataBundle` | simplify (K2) | simplify (K2) | K2 | both non-keep |
| `Clean/gantry_dynamic/data.py:66` | `Norm` | simplify (K40) | keep | K40 | disagreement (section 3) |
| `Clean/gantry_dynamic/data.py:78` | `traj_dir [callable]` | keep | keep | - | keep (both) |
| `Clean/gantry_dynamic/data.py:83` | `_load_u [callable]` | inline (K10) | inline (K10) | K10 | both non-keep |
| `Clean/gantry_dynamic/data.py:88` | `_resample_u [callable]` | keep | keep | - | keep (both) |
| `Clean/gantry_dynamic/data.py:106` | `_decimate_y [callable]` | keep | simplify (K41) | K41 | agreed after reading (Claude changed verdict) |
| `Clean/gantry_dynamic/data.py:126` | `_record_at_fs_new [callable]` | keep | simplify (K10) | K10 | same K; call-site edit noted by one audit |
| `Clean/gantry_dynamic/data.py:139` | `_load_record [callable]` | merge with _split (K11, K12) | simplify (K11) | K11, K12 | both non-keep |
| `Clean/gantry_dynamic/data.py:149` | `servo_sigma [callable]` | keep | keep | - | keep (both) |
| `Clean/gantry_dynamic/data.py:169` | `load_datasets [callable]` | simplify (K2) | simplify (K2) | K2 | both non-keep |
| `Clean/gantry_dynamic/data.py:176` | `load_datasets._split [callable]` | merge (K12) | keep | K12 | disagreement (section 3) |
| `Clean/gantry_dynamic/data.py:199` | `compute_normalization [callable]` | keep | keep | - | keep (both) |
| `Clean/gantry_dynamic/model.py:1` | `<file>` | keep | keep | - | keep (both) |
| `Clean/gantry_dynamic/model.py:28` | `build_model [callable]` | simplify (K30, K13, K39) | simplify (K13) | K13, K30, K39 | both non-keep |
| `Clean/gantry_dynamic/model.py:126` | `on_device [callable]` | keep | keep | - | keep (both) |
| `Clean/gantry_dynamic/model.py:137` | `prefit_physical_encoder [callable]` | simplify (K26) | simplify (K8, K26) | K8, K26 | both non-keep |
| `Clean/gantry_dynamic/model.py:155` | `prefit_physical_encoder.<lambda>` | keep | keep | - | keep (both) |
| `Clean/gantry_dynamic/model.py:156` | `prefit_physical_encoder.<lambda>` | keep | keep | - | keep (both) |
| `Clean/gantry_dynamic/model.py:178` | `prefit_physical_encoder.recorder [callable]` | keep | keep | - | keep (both) |
| `Clean/gantry_dynamic/model.py:208` | `least_squares_start [callable]` | keep | keep | - | keep (both) |
| `Clean/gantry_dynamic/model.py:228` | `least_squares_start.<lambda>` | keep | keep | - | keep (both) |
| `Clean/gantry_dynamic/obc_gantry.py:1` | `<file>` | keep | keep | - | keep (both) |
| `Clean/gantry_dynamic/obc_gantry.py:34` | `GantryReferenceSet` | simplify (K15) | simplify (K15) | K15 | both non-keep |
| `Clean/gantry_dynamic/obc_gantry.py:44` | `GantryReferenceSet.n [property]` | inline (K15) | inline (K15) | K15 | both non-keep |
| `Clean/gantry_dynamic/obc_gantry.py:49` | `fourth_order_central_difference [callable]` | keep | keep | - | keep (both) |
| `Clean/gantry_dynamic/obc_gantry.py:64` | `build_reference_set [callable]` | simplify (K15, K6) | simplify (K6, K15) | K6, K15 | both non-keep |
| `Clean/gantry_dynamic/obc_gantry.py:123` | `GantryOBCCorrection` | keep | keep | - | keep (both) |
| `Clean/gantry_dynamic/obc_gantry.py:127` | `GantryOBCCorrection.__init__ [constructor]` | simplify (K32) | keep | K32 | disagreement (section 3) |
| `Clean/gantry_dynamic/obc_gantry.py:157` | `GantryOBCCorrection.expansion_point [callable]` | keep | inline (K44) | K44 | agreed after reading (Claude changed verdict) |
| `Clean/gantry_dynamic/obc_gantry.py:162` | `GantryOBCCorrection.set_frozen [callable]` | keep | keep | - | keep (both) |
| `Clean/gantry_dynamic/obc_gantry.py:167` | `GantryOBCCorrection.tangent_pair [callable]` | simplify (K27) | keep | K27 | disagreement (section 3) |
| `Clean/gantry_dynamic/obc_gantry.py:181` | `GantryOBCCorrection.forward [callable]` | simplify (K14) | simplify (K14) | K14 | both non-keep |
| `Clean/gantry_dynamic/obc_gantry.py:204` | `make_float64_block [callable]` | inline (K16) | inline (K16) | K16 | both non-keep |
| `Clean/gantry_dynamic/obc_gantry.py:209` | `make_basis_builder [callable]` | simplify (K17) | simplify (K17) | K17 | both non-keep |
| `Clean/gantry_dynamic/obc_gantry.py:225` | `make_basis_builder.build [callable]` | keep | simplify (K17, K18) | K17, K18 | same K; call-site edit noted by one audit |
| `Clean/gantry_dynamic/obc_gantry.py:239` | `make_reference_field [callable]` | keep | keep | - | keep (both) |
| `Clean/gantry_dynamic/obc_gantry.py:253` | `make_reference_field.field [callable]` | keep | keep | - | keep (both) |
| `Clean/gantry_dynamic/obc_gantry.py:266` | `attach_obc [callable]` | simplify (K15) | simplify (K15, K16) | K15, K16 | both non-keep |
| `Clean/gantry_dynamic/training.py:1` | `<file>` | keep | keep | - | keep (both) |
| `Clean/gantry_dynamic/training.py:17` | `_json_safe [callable]` | keep | keep | - | keep (both) |
| `Clean/gantry_dynamic/training.py:30` | `save_checkpoint_pair [callable]` | keep | keep | - | keep (both) |
| `Clean/gantry_dynamic/training.py:42` | `resolve_checkpoint [callable]` | keep | keep | - | keep (both) |
| `Clean/gantry_dynamic/training.py:71` | `load_optimizer_moments [callable]` | keep | keep | - | keep (both) |
| `Clean/gantry_dynamic/training.py:86` | `_assert_same_architecture [callable]` | keep | keep | - | keep (both) |
| `Clean/gantry_dynamic/training.py:109` | `_assert_same_dtype [callable]` | keep | keep | - | keep (both) |
| `Clean/gantry_dynamic/training.py:120` | `load_checkpoint [callable]` | simplify (K7) | simplify (K7) | K7 | both non-keep |
| `Clean/gantry_dynamic/training.py:151` | `ValidationHistory` | keep | keep | - | keep (both) |
| `Clean/gantry_dynamic/training.py:157` | `ValidationHistory.__init__ [constructor]` | keep | keep | - | keep (both) |
| `Clean/gantry_dynamic/training.py:163` | `ValidationHistory.record [callable]` | keep | keep | - | keep (both) |
| `Clean/gantry_dynamic/training.py:175` | `ValidationHistory.hist_arrays [callable]` | inline (K35) | keep | K35 | disagreement (section 3) |
| `Clean/gantry_dynamic/training.py:188` | `ValidationHistory.__call__ [callable]` | keep | keep | - | keep (both) |
| `Clean/gantry_dynamic/training.py:193` | `run_lbfgs_polish [callable]` | simplify (K30) | keep | K30 | disagreement (section 3) |
| `Clean/gantry_dynamic/training.py:231` | `_adam_phase [callable]` | simplify (K30) | simplify (K8) | K8, K30 | both non-keep |
| `Clean/gantry_dynamic/training.py:277` | `train [callable]` | simplify (K30) | keep | K30 | disagreement (section 3) |
| `Clean/gantry_dynamic/training.py:320` | `train.<lambda>` | keep | keep | - | keep (both) |
| `Clean/gantry_interconnect_dynamic.py:1` | `<file>` | keep | keep | - | keep (both) |
| `Clean/gantry_interconnect_dynamic.py:21` | `_dumps [callable]` | keep | keep | - | keep (both) |
| `Clean/gantry_interconnect_dynamic.py:27` | `_rerun_source [callable]` | keep | keep | - | keep (both) |
| `Clean/gantry_interconnect_dynamic.py:46` | `_archive_rerun [callable]` | keep | keep | - | keep (both) |
| `Clean/gantry_interconnect_dynamic.py:69` | `main [callable]` | simplify (K31) | simplify (K6, K7, K9) | K6, K7, K9, K31 | both non-keep |
| `Clean/gantry_interconnect_dynamic.py:135` | `cli [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/__init__.py:1` | `<file>` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/__init__.py:1` | `<file>` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/blocks.py:1` | `<file>` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/blocks.py:22` | `Block` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/blocks.py:25` | `Block.__init__ [constructor]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/blocks.py:33` | `Static_ANN_Block` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/blocks.py:35` | `Static_ANN_Block.__init__ [constructor]` | simplify (K20) | simplify (K1, K20) | K1, K20 | both non-keep |
| `Clean/model_augmentation/fit_systems/blocks.py:65` | `Static_ANN_Block.forward [callable]` | simplify (K20) | simplify (K1, K20) | K1, K20 | both non-keep |
| `Clean/model_augmentation/fit_systems/blocks.py:82` | `Linear_Output_Block` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/blocks.py:85` | `Linear_Output_Block.__init__ [constructor]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/blocks.py:99` | `Linear_Output_Block.forward [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/blocks.py:117` | `Discrete_Nonlinear_Function_Block` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/blocks.py:119` | `Discrete_Nonlinear_Function_Block.forward [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/blocks.py:126` | `GantryStateBlock` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/blocks.py:130` | `GantryStateBlock.__init__ [constructor]` | simplify (K13, K34) | simplify (K13) | K13, K34 | both non-keep |
| `Clean/model_augmentation/fit_systems/blocks.py:188` | `GantryStateBlock._rk4 [callable]` | simplify (K34) | keep | K34 | disagreement (section 3) |
| `Clean/model_augmentation/fit_systems/blocks.py:211` | `GantryStateBlock._deriv_with [callable]` | simplify (K13) | simplify (K13) | K13 | both non-keep |
| `Clean/model_augmentation/fit_systems/blocks.py:279` | `GantryStateBlock._deriv_with_tangent [callable]` | simplify (K13) | simplify (K13) | K13 | both non-keep |
| `Clean/model_augmentation/fit_systems/blocks.py:369` | `GantryStateBlock._rk4_with_tangent [callable]` | simplify (K34) | keep | K34 | disagreement (section 3) |
| `Clean/model_augmentation/fit_systems/blocks.py:400` | `_TrainableGantryStateBlock` | merge into IdentifiableGantryStateBlock (K33) | keep | K33 | disagreement (section 3) |
| `Clean/model_augmentation/fit_systems/blocks.py:406` | `_TrainableGantryStateBlock.__init__ [constructor]` | merge (K33) | keep | K33 | disagreement (section 3) |
| `Clean/model_augmentation/fit_systems/blocks.py:419` | `_TrainableGantryStateBlock._build_KC [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/blocks.py:450` | `_TrainableGantryStateBlock._build_M1M2 [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/blocks.py:475` | `_TrainableGantryStateBlock._mats_from_raw [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/blocks.py:497` | `_TrainableGantryStateBlock.nonlinear_function [callable]` | simplify (K14) | simplify (K14) | K14 | both non-keep |
| `Clean/model_augmentation/fit_systems/blocks.py:512` | `_TrainableGantryStateBlock.pass_mats [callable]` | keep | inline (K43) | K43 | agreed after reading (Claude changed verdict) |
| `Clean/model_augmentation/fit_systems/blocks.py:518` | `IdentifiableGantryStateBlock` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/blocks.py:531` | `IdentifiableGantryStateBlock.__init__ [constructor]` | keep | simplify (K42) | K42 | agreed after reading (Claude changed verdict) |
| `Clean/model_augmentation/fit_systems/blocks.py:565` | `IdentifiableGantryStateBlock.combos_of [callable]` | keep (K1) | keep | K1 | keep (both) |
| `Clean/model_augmentation/fit_systems/blocks.py:586` | `IdentifiableGantryStateBlock.gauge_section [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/blocks.py:615` | `IdentifiableGantryStateBlock._m_diff_bound [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/blocks.py:629` | `IdentifiableGantryStateBlock.combinations_from_free [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/blocks.py:648` | `IdentifiableGantryStateBlock.free_from_combinations [callable]` | remove (K1) | remove (K1) | K1 | both non-keep |
| `Clean/model_augmentation/fit_systems/blocks.py:676` | `IdentifiableGantryStateBlock.recover_combinations [callable]` | remove (K1) | remove (K1) | K1 | both non-keep |
| `Clean/model_augmentation/fit_systems/blocks.py:686` | `IdentifiableGantryStateBlock.transition_from_free [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/blocks.py:700` | `IdentifiableGantryStateBlock.mats_and_tangent_from_free [callable]` | inline (K27) | keep | K27 | disagreement (section 3) |
| `Clean/model_augmentation/fit_systems/blocks.py:713` | `IdentifiableGantryStateBlock._mats_at [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/blocks.py:718` | `IdentifiableGantryStateBlock.estimation_parameters [callable]` | remove (K1) | remove (K1) | K1 | both non-keep |
| `Clean/model_augmentation/fit_systems/closed_loop.py:1` | `<file>` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/closed_loop.py:17` | `ControllerBank` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/closed_loop.py:21` | `ControllerBank.__init__ [constructor]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/closed_loop.py:37` | `ControllerBank.__init__.<lambda>` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/closed_loop.py:52` | `ControllerBank.n_controllers [property]` | inline (K23) | inline (K23) | K23 | both non-keep |
| `Clean/model_augmentation/fit_systems/closed_loop.py:56` | `ControllerBank.step [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/closed_loop.py:76` | `closed_loop_step [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/closed_loop.py:90` | `closed_loop_step.step [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/closed_loop.py:104` | `closed_loop_rollout [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/closed_loop.py:136` | `window_counts [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/closed_loop.py:142` | `ClosedLoopSimulator` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/closed_loop.py:145` | `ClosedLoopSimulator.__init__ [constructor]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/closed_loop.py:159` | `ClosedLoopSimulator.__getstate__ [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/closed_loop.py:166` | `ClosedLoopSimulator.eager [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/closed_loop.py:174` | `ClosedLoopSimulator.__call__ [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/closed_loop.py:202` | `ClosedLoopSimulator.augment_training_data [callable]` | simplify (K38) | keep | K38 | disagreement (section 3) |
| `Clean/model_augmentation/fit_systems/closed_loop.py:221` | `ClosedLoopSimulator.verify_by_content [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/closed_loop.py:236` | `ClosedLoopSimulator.validation_error [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/closed_loop.py:259` | `closed_loop_free_run_rms_batch [callable]` | simplify (K21, K22) | simplify (K21) | K21, K22 | both non-keep |
| `Clean/model_augmentation/fit_systems/closed_loop.py:280` | `closed_loop_free_run_rms_batch.<lambda>` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/closed_loop.py:281` | `closed_loop_free_run_rms_batch.<lambda>` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/interconnect.py:1` | `<file>` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/interconnect.py:24` | `Interconnect` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/interconnect.py:26` | `Interconnect.__init__ [constructor]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/interconnect.py:47` | `Interconnect._apply [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/interconnect.py:55` | `Interconnect.forward [callable]` | keep | simplify (K46) | K46 | disagreement (section 3) |
| `Clean/model_augmentation/fit_systems/interconnect.py:96` | `Interconnect._propagate [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/interconnect.py:120` | `Interconnect.init_forward [callable]` | keep | simplify (K43) | K43 | agreed after reading (Claude changed verdict) |
| `Clean/model_augmentation/fit_systems/interconnect.py:172` | `Interconnect._signal_cone [callable]` | simplify (K24) | simplify (K24) | K24 | both non-keep |
| `Clean/model_augmentation/fit_systems/interconnect.py:186` | `Interconnect.output_only [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/interconnect.py:207` | `Interconnect.init_connection_matrices [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/interconnect.py:249` | `Interconnect.add_block [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/interconnect.py:258` | `Interconnect.connect_signals [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/interconnect.py:270` | `Interconnect.connect_block_signals [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/interconnect.py:278` | `Interconnect.determine_signal_ix [callable]` | keep | simplify (K47) | K47 | disagreement (section 3) |
| `Clean/model_augmentation/fit_systems/interconnect.py:285` | `Interconnect.convert_signal_name_to_ix [callable]` | keep | simplify (K47) | K47 | disagreement (section 3) |
| `Clean/model_augmentation/fit_systems/interconnect.py:292` | `Interconnect.parse_connection_function_method [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/interconnect.py:297` | `Signal_Connection` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/interconnect.py:299` | `Signal_Connection.__init__ [constructor]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/interconnect.py:309` | `_replace_with_retry [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/interconnect.py:322` | `DtypeDataLoader` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/interconnect.py:325` | `DtypeDataLoader.__init__ [constructor]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/interconnect.py:327` | `DtypeDataLoader.__init__._conv [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/interconnect.py:338` | `SSE_Interconnect` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/interconnect.py:349` | `SSE_Interconnect.__init__ [constructor]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/interconnect.py:358` | `SSE_Interconnect.init_model [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/interconnect.py:366` | `SSE_Interconnect.simulate [callable]` | inline (K25) | inline (K25) | K25 | both non-keep |
| `Clean/model_augmentation/fit_systems/interconnect.py:371` | `SSE_Interconnect.make_training_data [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/interconnect.py:377` | `SSE_Interconnect.cal_validation_error [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/interconnect.py:386` | `SSE_Interconnect.fit [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/interconnect.py:405` | `SSE_Interconnect.fit.validation [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/interconnect.py:466` | `SSE_Interconnect.fit.closure [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/interconnect.py:522` | `AugmentedModel` | keep | simplify (K15) | K15 | same K; call-site edit noted by one audit |
| `Clean/model_augmentation/fit_systems/interconnect.py:531` | `AugmentedModel.__init__ [constructor]` | simplify (K28) | simplify (K43) | K28, K43 | both non-keep |
| `Clean/model_augmentation/fit_systems/interconnect.py:541` | `AugmentedModel._epoch_index [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/interconnect.py:546` | `AugmentedModel._sync_obc_coefficient [callable]` | keep | simplify (K44) | K44 | agreed after reading (Claude changed verdict) |
| `Clean/model_augmentation/fit_systems/interconnect.py:568` | `AugmentedModel.objective_matrices [callable]` | keep | simplify (K43) | K43 | agreed after reading (Claude changed verdict) |
| `Clean/model_augmentation/fit_systems/interconnect.py:586` | `AugmentedModel.loss [callable]` | keep | simplify (K25, K44) | K25, K44 | agreed after reading (Claude changed verdict) |
| `Clean/model_augmentation/fit_systems/interconnect.py:621` | `AugmentedModel.checkpoint_save_system [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/interconnect.py:629` | `AugmentedModel.checkpoint_load_system [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/interconnect.py:638` | `AugmentedModel.__getstate__ [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/interconnect.py:644` | `AugmentedModel.optimizer [property]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/interconnect.py:649` | `AugmentedModel.optimizer [property setter]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/lbfgs_polish.py:1` | `<file>` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/lbfgs_polish.py:16` | `PolishSpec` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/lbfgs_polish.py:27` | `_restore [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/lbfgs_polish.py:37` | `FixedWindowBatch` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/lbfgs_polish.py:41` | `FixedWindowBatch.__init__ [constructor]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/lbfgs_polish.py:66` | `FixedWindowBatch.__init__._to_tensor [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/lbfgs_polish.py:84` | `_make_closure [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/lbfgs_polish.py:86` | `_make_closure.closure [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/lbfgs_polish.py:97` | `_lbfgs_counts [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/lbfgs_polish.py:103` | `_is_oom [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/lbfgs_polish.py:109` | `_probe_closure [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/lbfgs_polish.py:125` | `_acceptance [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/lbfgs_polish.py:136` | `lbfgs_polish [callable]` | split (K36) | keep | K36 | disagreement (section 3) |
| `Clean/model_augmentation/fit_systems/lbfgs_polish.py:165` | `lbfgs_polish._validate [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/lbfgs_polish.py:176` | `lbfgs_polish._record_counts [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/lbfgs_polish.py:181` | `lbfgs_polish._abort [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/lbfgs_polish.py:204` | `lbfgs_polish._new_opt [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/obc.py:1` | `<file>` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/obc.py:13` | `OBCRankLossAbort` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/obc.py:17` | `OBCBasis` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/obc.py:21` | `OBCBasis.__init__ [constructor]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/obc.py:53` | `OBCBasis.coefficient [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/obc.py:69` | `OBCBasis.to [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/obc.py:77` | `OBCBasis.device [property]` | keep | inline (K45) | K45 | agreed after reading (Claude changed verdict) |
| `Clean/model_augmentation/fit_systems/obc.py:82` | `build_stacked_sensitivity [callable]` | simplify (K18, K19) | simplify (K17, K18) | K17, K18, K19 | both non-keep |
| `Clean/model_augmentation/fit_systems/obc.py:114` | `build_stacked_sensitivity.<lambda>` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/obc.py:126` | `OBCProjection` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/obc.py:131` | `OBCProjection.__init__ [constructor]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/obc.py:140` | `OBCProjection.refresh [callable]` | simplify (K17) | simplify (K17, K45) | K17, K45 | both non-keep |
| `Clean/model_augmentation/fit_systems/obc.py:174` | `OBCProjection.maybe_refresh [callable]` | keep | simplify (K44) | K44 | agreed after reading (Claude changed verdict) |
| `Clean/model_augmentation/fit_systems/obc.py:186` | `OBCProjection.coefficient [callable]` | keep | simplify (K45) | K45 | agreed after reading (Claude changed verdict) |
| `Clean/model_augmentation/fit_systems/pre_encoder.py:1` | `<file>` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/pre_encoder.py:18` | `AugmentedEncoder` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/pre_encoder.py:22` | `AugmentedEncoder.__init__ [constructor]` | simplify (K37) | keep | K37 | disagreement (section 3) |
| `Clean/model_augmentation/fit_systems/pre_encoder.py:81` | `AugmentedEncoder.forward [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/ps2_opening.py:1` | `<file>` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/ps2_opening.py:21` | `PS2Spec` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/ps2_opening.py:44` | `_plateaued [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/ps2_opening.py:49` | `calibration_split [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/ps2_opening.py:68` | `PS2Opening` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/ps2_opening.py:72` | `PS2Opening.__init__ [constructor]` | simplify (K32) | keep | K32 | disagreement (section 3) |
| `Clean/model_augmentation/fit_systems/ps2_opening.py:95` | `PS2Opening.done [property]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/ps2_opening.py:99` | `PS2Opening.after_step [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/ps2_opening.py:142` | `PS2Opening.summary_json [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/ps2_opening.py:149` | `PS2Opening._start [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/ps2_opening.py:168` | `PS2Opening._sample [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/ps2_opening.py:200` | `PS2Opening._residual [callable]` | keep | simplify (K25) | K25 | same K; call-site edit noted by one audit |
| `Clean/model_augmentation/fit_systems/ps2_opening.py:219` | `PS2Opening._added_states [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/ps2_opening.py:223` | `PS2Opening._s1_step [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/ps2_opening.py:251` | `PS2Opening._cal_unexplained [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/ps2_opening.py:266` | `PS2Opening._fit_added_state_transition [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/ps2_opening.py:284` | `PS2Opening._fit_added_state_transition.snap [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/fit_systems/ps2_opening.py:292` | `PS2Opening._fit_added_state_transition.state_map_loss [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/systems/__init__.py:1` | `<file>` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/systems/gantry_ss.py:1` | `<file>` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/systems/gantry_ss.py:69` | `build_poly_constants [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/systems/gantry_ss.py:109` | `build_G_matrix_entries [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/utils/__init__.py:1` | `<file>` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/utils/deepSI_corrections.py:1` | `<file>` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/utils/deepSI_corrections.py:12` | `WindowedSystemData` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/utils/deepSI_corrections.py:14` | `WindowedSystemData.__init__ [constructor]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/utils/deepSI_corrections.py:18` | `WindowedSystemData.to_hist_future_data [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/utils/deepSI_corrections.py:48` | `SystemDataNormalizer` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/utils/deepSI_corrections.py:50` | `SystemDataNormalizer.__init__ [constructor]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/utils/deepSI_corrections.py:57` | `SystemDataNormalizer.transform [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/utils/torch_nets.py:1` | `<file>` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/utils/torch_nets.py:15` | `build_mlp [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/utils/torch_nets.py:43` | `AugmentationNetwork` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/utils/torch_nets.py:47` | `AugmentationNetwork.__init__ [constructor]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/utils/torch_nets.py:64` | `AugmentationNetwork.forward [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/utils/utils.py:1` | `<file>` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/utils/utils.py:8` | `added [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/utils/utils.py:13` | `to_tensor [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/utils/utils.py:19` | `expansion_matrix [callable]` | keep | keep | - | keep (both) |
| `Clean/model_augmentation/utils/utils.py:33` | `selection_matrix [callable]` | keep | keep | - | keep (both) |
