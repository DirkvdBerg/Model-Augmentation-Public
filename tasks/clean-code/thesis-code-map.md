# Thesis-to-code map, merged (D-284 step 4 item 3)

Merged from `thesis-code-map-claude.md` (C) and `thesis-code-map-codex.md` (X). The comments of item 4 are written
from this file. Thesis revision `6b77c9a` (working tree equal). Code line numbers refer to the 24 grey-box files
before the comment rewrite (the `implemented` snapshot plus the MM16 fix); the rewrite moved lines, not code.
Path abbreviations and the symbol table are those of the Claude map, section 0. Equation tests
(`Thesis-writeup/Code/Verification/clean-code/equation_tests.py`), current state 2026-10-28: 71 passed, 1 failed
(`P_s2_cost`, whose expected value is corrected in item 8, MM3). Before the MM16 fix: 69 passed, 3 failed.

## 0. Scope and how disagreements are decided
- Included appendix subsections: `app:params`, `app:lfr-details`, `app:lfr-wellposed`, `app:combos`,
  `app:setup_system`, `app:hyper` (grey-box rows). Excluded: `app:splits`, `app:setup_records`, `app:setup_checks`,
  `app:setup_bb` and the black-box rows of `app:hyper`.
- Scope: both maps cover the same active displays (X counts 61) and Section II to VI text. They differ only on
  `app:setup_records` and `app:setup_checks`: X includes them (rows `S_records`, `S_processing`, all external or
  procedure), C excludes them as subsections. **Decided**: excluded as subsections; their only method content (block
  mean, error-only filter, trim, manifest split) is mapped through the active Section VI text 05:323 to 05:365, which
  both maps cover.
- "Agree" = both maps give the same thesis entry (wording or row granularity aside). A disagreement is decided from
  the thesis text and the code, in one line, or listed for Dirk in section 4.
- A failing equation test is a mismatch (section 3).

## 1. Forward: thesis entry, code, both mappings

| Thesis ID | Code (merged) | C | X | Agree / decision |
|-|-|-|-|-|
| eq:system | external (data generation) | ext | B_system ext | agree |
| eq:system_loop | `ctrl.controller_ss`, `ctrl.build_cfb_at`; `cl.ControllerBank` | K_fb matrices | A_controller_state/force (PASS) | agree |
| 02:197 hold | `blocks._rk4` (u held), `data._resample_u` | yes | in A_rk4 | agree |
| eq:Ptransform | `ss.P`, `ss.Cd`, `obcg.P_INV_T64`, `data.compute_normalization`, `blocks._deriv_with` | yes | B_transform (PASS) | agree |
| eq:eom, eq:base_ss | `blocks.GantryStateBlock._deriv_with` (+ `_deriv_with_tangent`) | yes | B_vector_field, B_state (PASS) | agree |
| eq:mass_matrix | `ctrl.mass_matrix_at`; `ss.build_poly_constants`; `blocks._mats_from_raw` | yes | B_mass (PASS, 2e-7 tol) | agree; tolerance is MM16 |
| eq:damping_matrix, eq:stiffness_matrix | `ss.C`, `ss.K`, `blocks._build_KC`, `ctrl.C_DAMP`, `ctrl.K_STIFF` | yes | B_damping, B_stiffness (PASS) | agree |
| eq:Msplit, 99:63 | `ss.M1`, `ss.M2`, `blocks._build_M1M2` | yes | B_mass_split, L_matrices (PASS) | agree |
| 02:745 | derivation only | yes | B_identifiability ext | agree |
| eq:phi | `blocks.combos_of`, `COMBO_NAMES` | yes | B_combinations (PASS) | agree |
| 02:706 L_b fixed | `ss.Lb`, `blocks._TrainableGantryStateBlock.__init__` | yes | none | X has no row; C kept (L_b is a buffer, not trained) |
| eq:lfr_admissibility | `blocks._m_diff_bound` with `combinations_from_free` | yes | B_admissibility (PASS) | agree |
| 02:854 log coordinates | `blocks.combinations_from_free`, `__init__` (beta = 0) | yes | A_coordinates (PASS) | agree |
| eq:mdelta_map | `blocks.combinations_from_free`, `_m_diff_bound`, `M_DIFF_RHO` | yes | B_mdelta (PASS) | agree |
| eq:aug_transition | `model.build_model` wiring, `ic.Interconnect.forward`, `._propagate`, `blocks.Linear_Output_Block` | yes | A_physical, A_added, A_output (PASS) | agree |
| 03:450 RK4 | `blocks._rk4`, `_deriv_with` (Y per stage) | yes | A_rk4 (PASS) | agree |
| 03:475 network | `nets.AugmentationNetwork`, `nets.build_mlp`, `blocks.Static_ANN_Block`, `utils.expansion_matrix` | yes | A_network (PASS) | agree |
| eq:aug_norm, 03:521, 03:637 | `data.compute_normalization`, `dsc.SystemDataNormalizer.transform` | yes | A_normalization (PASS after the MM16 fix) | MM16, fixed |
| eq:aug_norm_maps | `blocks._deriv_with` (denormalise, xdot / sigma_x), `data.compute_normalization` (C^n), `blocks.Linear_Output_Block` | yes | A_norm_transition, A_norm_inverse, A_norm_output (PASS after the MM16 fix) | MM16, fixed |
| 03:665 | `blocks.Static_ANN_Block.forward`, `nets.AugmentationNetwork.forward` | yes | A_network (PASS) | agree |
| eq:aug_two_loops | `cl.closed_loop_step` through the residual form | derivation, residual form | A_model_loop_state/force (PASS) | agree (X tests the equivalence C stated) |
| eq:aug_residual_loop | `cl.ControllerBank.step`, `cl.closed_loop_step`, `cl.closed_loop_rollout` (x^c = 0) | yes | A_residual_state/force (PASS) | agree |
| eq:aug_residual_split | derivation only | yes | A_residual_split ext | agree |
| 03:943 step order, eq:aug_cl_step | `cl.closed_loop_step.step`, `cl.ControllerBank.__init__`, `.step` | yes | A_cl_output, A_cl_input, A_cl_step (PASS) | agree |
| eq:aug_loss, 03:1149 W_r | `ic.AugmentedModel.loss`, `ctrl.build_closed_loop`, `data.servo_sigma`, `cl.augment_training_data`, `dsc.to_hist_future_data` | yes | A_loss, A_loss_physical/added/initial (PASS) | agree |
| eq:aug_encoder, eq:aug_enc_lin | `enc.AugmentedEncoder.forward`, `dsc.to_hist_future_data`, `model.build_model` (na - 1) | yes | A_encoder_history, A_encoder_linear (PASS) | agree; (u, y) order of psi_tilde: MM10 |
| 03:1266 Adam | `ic.SSE_Interconnect.fit`, `.init_model` | yes | A_optimizer (not testable) | agree |
| 03:1291 selection | `cl.ClosedLoopSimulator.validation_error`, `cl.closed_loop_free_run_rms_batch`, `ic.fit.validation` | yes | A_optimizer, S_metric | agree |
| 03:1304 polish | `lbfgs.lbfgs_polish`, `FixedWindowBatch`, `_acceptance`, `_restore`, `train.run_lbfgs_polish` | yes | A_optimizer | agree; basis held fixed during polish: MM2 |
| eq:aug_enc_sim, 03:1387, 03:1413 | `model.prefit_physical_encoder` | yes | A_encoder_sim (PASS) | agree; MM11 |
| eq:aug_enc_fit, 03:1432 | `model.least_squares_start` | yes | A_encoder_fit (PASS) | agree |
| eq:aug_init, 03:1506 | `blocks.IdentifiableGantryStateBlock.__init__`, `nets.AugmentationNetwork.__init__`, `nets.build_mlp`, `enc.AugmentedEncoder.__init__`, `model.least_squares_start` | yes | A_init_beta/network/encoder (PASS) | agree |
| eq:ps2_s1, 03:1671, 03:1736 | `ps2._s1_step`, `._residual`, `._start`, `._added_states`, `._sample`, `ps2.calibration_split` | yes | P_s1_loss, P_residual (PASS) | agree |
| 03:1811 stops | `ps2.after_step`, `ps2._plateaued`, `._cal_unexplained`, `PS2Spec` | yes | P_stops (PASS) | agree |
| eq:ps2_alt | `ic.fit` 430 to 432, `ps2.after_step` | yes | P_alternation (not testable) | agree |
| eq:ps2_s2, 03:1913 | `ps2._fit_added_state_transition` (`snap`, `state_map_loss`) | yes | P_pairs (PASS), P_s2_cost (**FAIL**) | mismatch MM3; 03:1913 only in C, kept |
| eq:ps2_handover | `ps2.after_step` 103 to 113 | yes | P_handover (not testable) | agree |
| eq:obc_gyorok | derivation only | yes | O_gyorok ext | agree |
| eq:obc_sens | `obc.build_stacked_sensitivity`, `blocks.transition_from_free`, `obcg.make_basis_builder` | yes | O_sensitivity (PASS) | agree |
| eq:obc_offset | `obcg.make_basis_builder.build` (column f^n_base(beta_lin)) | MM1, side none | O_offset (PASS), M3 side thesis | decided MM1: none |
| eq:obc_tuples, 04:255 | `obcg.fourth_order_central_difference`, `obcg.build_reference_set` | yes | O_difference, O_tuple_position (PASS) | agree |
| 04:263 rank | `obc.OBCBasis.__init__`, `obc.OBCProjection.refresh`, `obc.OBCRankLossAbort` | yes | none | X has no row; C kept |
| eq:obc_coef | `obc.OBCBasis.coefficient`, `obc.OBCProjection.coefficient`, `obcg.make_reference_field` | yes | O_basis, O_field, O_coefficient, O_projected, O_orthogonality (PASS) | agree |
| eq:obc_corr, 04:325 | `obcg.GantryOBCCorrection.forward`, `.tangent_pair`, `blocks._rk4_with_tangent`, `ic.Interconnect.forward`, `ic.AugmentedModel.loss` | yes | O_correction, O_physical_step, O_added_step (PASS) | agree |
| 04:341 refresh | `ic.AugmentedModel.loss`, `obc.maybe_refresh`, `ic._epoch_index`, `obcg.expansion_point` | yes | O_refresh (not testable) | agree; rebuild after Adam: MM2 |
| eq:obc_eps, eq:obc_recovery (first line), eq:obc_overlap | external (`Thesis-writeup/Code/Evaluation/eval_overlap.py`); `report._true_free`, `obcg.GantryReferenceSet.x_phys_next_rec` | yes | O_residual, O_recovery_projected, O_overlap ext | agree |
| eq:obc_recovery (second line, beta_hat_np) | derivation only | ext | O_recovery_unprojected, derivation | agree (Codex check F007) |
| eq:setup_truth, eq:setup_fric, eq:setup_abs, 99:371 | external (data generation) | yes | S_friction, S_absorber, S_extended_matrices ext | agree |
| 05:229, eq:setup_K1, 99:393 | `ctrl.normalized_controller_coefficients`, `ctrl.build_cfb_at`, `ctrl.stage_frequency_response`, `ctrl.controller_ss` | yes | S_controller (PASS, 2e-9 tol) | agree |
| 05:323 processing | `data._decimate_y`, `data._resample_u`, `data._record_at_fs_new`, `cfg.AA_ORDER`, `cfg.AA_TRIM_S` | yes | S_processing (not testable) | agree; MM7, MM8 |
| 05:365 split | `data.record_files` | yes | S_records ext | agree |
| tab:setup_models | `cfg.PRESETS`; M_LPV: `report._without_learned`, `report._true_free` (Ext `eval_greybox.py:142` to 151) | yes | none | X has no row; C kept |
| 05:497 detune, 05:502 seed | `cfg.RunConfig.combo_init_detune`, `model.build_model` 39, `cfg.seed`, `entry.main` | yes | S_defaults | agree |
| eq:setup_rms | `cl.validation_error`, `cl.closed_loop_free_run_rms_batch`, `data.servo_sigma`; test metrics external | yes | S_metric (PASS) | agree |
| tab:nominal_parameters | `ss` 9 to 28, `ctrl` 19 to 21 | yes | module constants | agree; float32 storage: MM16 |
| eq:lfr_mass_factorization, eq:lfr_chain, eq:lfr_constant, eq:delta | `blocks._deriv_with` 184 to 188 | yes | L_factorization, L_chain, L_constant, L_delta (PASS) | agree |
| eq:lfr_G, eq:app_lfr_blocks | first row `ss.build_G_matrix_entries` in `_deriv_with`; C_z, D_zw, D_zu not built; output row C_y = [I 0] (q) is the physical-state slice `x2[:, :3]` (derivation level); `ss.Cd` = P^T [I 0] belongs to eq:aug_transition | yes | L_G_state/latent, L_blocks (PASS); L_blocks_latent/output ext | agree; C_y corrected (Codex check F006) |
| 99:199, 99:209, 99:254, 99:269, 99:311 | derivation only | yes | L_schur, L_wellposed, L_sylvester, L_raw_bound, L_inverse_entries ext | agree |
| 99:245, 99:258 | `ss.build_poly_constants`, d(Y) in `blocks._deriv_with` 165 | yes | L_reduced_mass, L_det (PASS) | agree |
| 99:279 | `blocks.IdentifiableGantryStateBlock.__init__` 379 to 381 | yes | L_eta (PASS) | agree |
| 99:297 | `blocks.gauge_section` then `_mats_from_raw` | yes | L_entries (PASS) | agree |
| tab:hyper | `cfg.RunConfig`, `ps2.PS2Spec`, `lbfgs.PolishSpec` | all match | all match | agree; Adam eps not in the table: MM6 |

## 2. Backward: callables
Rows where both maps give the same entry (or both say infrastructure) are summarised per file; every disagreement
has its own row. X's thesis rows are translated to the IDs of section 1.

**Agree, summarised.** `entry`: all infrastructure (main also 05:502). `cfg`: `y_decimation`, `record_trim` (05:323),
`ann_route_ix` (03:486), the rest infrastructure. `ctrl`: `mass_matrix_at`, `stage_frequency_response`,
`normalized_controller_coefficients`, `build_cfb_at`, `controller_ss` (eq:setup_K1, eq:system_loop), `build_closed_loop`
(W_r, sigma_e), metadata helpers infrastructure. `data`: `_resample_u`, `_decimate_y`, `_record_at_fs_new` (05:323),
`servo_sigma` (eq:setup_rms, W_r), `compute_normalization` (eq:aug_norm, C^n), `record_files`, `load_datasets` (05:365),
I/O infrastructure. `model`: `build_model` (eq:aug_transition plus eq:aug_init, 05:497), `prefit_physical_encoder` with
`recorder` (eq:aug_enc_sim), `least_squares_start` (eq:aug_enc_fit, eq:aug_init), `on_device` infrastructure. `obcg`:
`fourth_order_central_difference`, `build_reference_set` (eq:obc_tuples), `GantryOBCCorrection.*` (eq:obc_corr,
04:341), `make_basis_builder` with `build` (eq:obc_sens, eq:obc_offset), `make_reference_field` with `field`
(eq:obc_coef), `attach_obc`, `make_float64_block` infrastructure. `train`: `run_lbfgs_polish` (03:1304), `_adam_phase`,
`train` (03:1266 to 1309), checkpoint and history infrastructure. `blocks`: `Static_ANN_Block.*` (03:665),
`Linear_Output_Block.*` (eq:aug_norm_maps output), `GantryStateBlock.__init__`, `_rk4`, `_deriv_with`,
`_deriv_with_tangent`, `_rk4_with_tangent` (eq:base_ss, RK4, eq:obc_corr), `_build_KC`, `_build_M1M2`,
`_mats_from_raw` (eq:matrices), `nonlinear_function` (f_base), `IdentifiableGantryStateBlock.__init__`, `combos_of`,
`gauge_section`, `_m_diff_bound`, `combinations_from_free`, `free_from_combinations`, `recover_combinations`,
`transition_from_free`, `mats_and_tangent_from_free`, `_mats_at` (eq:phi, eq:mdelta_map, eq:obc_sens), `Block.__init__`
and `estimation_parameters` infrastructure. `cl`: `ControllerBank.*`, `closed_loop_step` with `step`,
`closed_loop_rollout`, `ClosedLoopSimulator.__call__`, `.validation_error`, `.augment_training_data`,
`closed_loop_free_run_rms_batch` (eq:aug_cl_step, eq:aug_residual_loop, eq:aug_loss, eq:setup_rms), the rest
infrastructure. `ic`: `Interconnect.forward`, `._propagate`, `.output_only` (eq:aug_transition, eq:obc_corr), Jan's
wiring methods infrastructure, the added `Interconnect._apply` 46, `._signal_cone` 160, `_replace_with_retry` 284,
`DtypeDataLoader.__init__` 300 with `_conv` 301 infrastructure, `SSE_Interconnect.fit` with `validation`, `closure` (03:1266, 03:1291, eq:ps2_alt),
`AugmentedModel._epoch_index`, `._sync_obc_coefficient`, `.loss` (04:341, eq:obc_coef, eq:aug_loss), dumps and
`optimizer` property infrastructure. `obc`: `OBCBasis.__init__`, `.coefficient`, `build_stacked_sensitivity`,
`OBCProjection.refresh`, `.maybe_refresh`, `.coefficient` (eq:obc_coef, eq:obc_sens, 04:341), `to`, `device`,
`OBCProjection.__init__` infrastructure. `enc`: `AugmentedEncoder.*` (eq:aug_enc_lin, eq:aug_init). `ps2`: every
callable as section 1 (eq:ps2_s1, eq:ps2_s2, 03:1811, eq:ps2_alt), `summary_json`, `done` infrastructure. `lbfgs`:
`FixedWindowBatch`, `_make_closure`, `_acceptance`, `lbfgs_polish` (03:1304), counters, OOM and probe infrastructure.
`ss`: `build_poly_constants` (eq:base_ss M(Y)^-1, 99:245, 99:258), `build_G_matrix_entries` (eq:app_lfr_blocks), constants
(tab:nominal_parameters). `dsc`: `to_hist_future_data` (eq:aug_loss windows, eq:aug_encoder), `transform` (eq:aug_norm),
`__init__`s infrastructure. `nets`: `build_mlp`, `AugmentationNetwork.*` (03:475, eq:aug_init, 03:665). `utils`:
`expansion_matrix`, `selection_matrix` (eq:aug_transition routing), `added`, `to_tensor` infrastructure. Generated
dataclass constructors (`RunConfig`, `DataBundle`, `Norm`, `GantryReferenceSet`, `PS2Spec`, `PolishSpec`) and lambdas:
infrastructure, as X lists them.

**Disagreements (16).**

| Callable | C | X | Decision |
|-|-|-|-|
| `report._without_learned` 11 | tab:setup_models (M_LPV), Ext | A_encoder_sim, O_overlap | C: it disables the network and f_obc for `eval_greybox.py:142` to 151 (baseline systems); the encoder prefit simulates the full model, and the overlap uses only `_true_free` |
| `report._true_free` 26 | eq:obc_eps (beta_star), Ext | B_mdelta, O_residual | both: the inverse of eq:mdelta_map at the true combinations, used for Xi |
| `cfg.RunConfig.nx_ann` 171 | infrastructure | A_added | X: it is n_xbar of eq:aug_transition |
| `cfg.RunConfig.hp` 192 | infrastructure | S_defaults | X: it carries tab:hyper values into the checkpoint |
| `cfg.RunConfig.combo_init_detune` 183 | 05:497 | B_combinations, S_defaults | C plus tab:hyper: it is the detuned start of Section VI, multiplied onto eq:phi |
| `ctrl.build_controller_bank` 122 | eq:aug_cl_step | S_controller | both: builds K_fb per record (eq:setup_K1) for the bank of eq:aug_cl_step |
| `blocks.Discrete_Nonlinear_Function_Block.forward` 73 | infrastructure | A_rk4 | C: Jan's interface; it only forwards to `nonlinear_function`, which carries A_rk4 |
| `blocks._TrainableGantryStateBlock.__init__` 282 | 02:706 | B_combinations | C plus tab:nominal_parameters: it stores theta^0 raw and L_b; the combinations live in the subclass |
| `blocks._TrainableGantryStateBlock.pass_mats` 346 | infrastructure | B_mass | C: it builds the eq:matrices once per objective; no formula of its own |
| `cl.window_counts` 77 | infrastructure | A_encoder_history | C: counts the windows of eq:aug_loss; no formula |
| `ic.SSE_Interconnect.simulate` 338 | infrastructure | A_cl_step | C: deepSI hook that forwards to the simulator |
| `ic.SSE_Interconnect.make_training_data` 343 | infrastructure | A_encoder_history | X: it builds the windows (chi_tau, n_f) of eq:aug_loss and eq:aug_encoder |
| `ic.SSE_Interconnect.cal_validation_error` 349 | infrastructure | S_metric | X: it returns the 03:1291 validation error |
| `ic.AugmentedModel.objective_matrices` 507 | infrastructure | B_mass, O_correction | C: bundles the per-objective matrices and tangent pair; formulas are in the callees |
| `ps2.PS2Opening.__init__` 56 | infrastructure | S_defaults, P_stops | X: it applies the left-out split of 03:1672 |
| `lbfgs._restore` 26 | 03:1305 | infrastructure | C: the rollback is the "only kept if the validation error drops" step |

Also only in one map (no conflict, kept): C maps `obc.OBCRankLossAbort` and `obcg.GantryReferenceSet` to 04:263 and
04:255; X lists them as infrastructure. Kept as C, since they carry the rank assumption and the tuple set.

## 3. Mismatches, merged
Side: **C** code would change, **T** thesis would change, **none** equivalent (the comment states the code's form).
"Test" = what item 8 does with the equation test.

| ID | Source | Mismatch | Decision |
|-|-|-|-|
| MM1 | C MM1, X M3 | eq:obc_offset / eq:obc_coef: code's extra column is f^n_base(beta_lin), not gamma; same span, identical F_tilde and f^n_obc everywhere; theta_aux in other coordinates ((a, c) code = (a + beta_lin c, c) thesis). X calls it thesis side. | **none**: theta_aux is never reported in the thesis and the algorithm is the same; the comment must state the code's coordinates. Test `O_offset` already passes on the translation. |
| MM2 | C only | basis rebuilt at the selected checkpoint after Adam, held fixed in the polish (`ic.fit` 465); thesis sentence struck (04:360) | **T**, for Dirk |
| MM3 | C MM3, X M1 | eq:ps2_s2: code cost = thesis cost x (B - 1)/B (`T.var(0)` divides by B - 1); same minimiser and same kept parameters; logged S2 costs scaled. X calls it code side. | **none**: changing the code changes the run (Adam epsilon scaling), and no S2 cost value is reported in the thesis; the comment states the factor. Test `P_s2_cost`: update to the code's formula (thesis quotient x (B - 1)/B). |
| MM4 | C, X notation | symbols: `M_DIFF_RHO` = gamma; `gamma_` = J_eff; `alpha`, `beta` = m_Sigma + m_h, (L_b/2) m_Delta | none (comments name them) |
| MM5 | merged into MM16 | | |
| MM6 | C only | Adam epsilon 1e-16 not in tab:hyper | **T**, for Dirk |
| MM7 | C only | 5 ms trim at both record ends not in the active text (struck 99:573) | **T**, for Dirk |
| MM8 | C only | noise-free records point sampled, not filtered (05:325) | **T conditional**: the revnote (99:563) already rules it; no action unless a noise-free training run is reported |
| MM10 | C, X notation | psi_tilde reads (u, y), chi is (y, u) | none |
| MM11 | C, X test | eq:aug_enc_sim simulated with the whole start model (f^n_aug = 0 and f^n_obc = 0 on the physical rows) | none (X `A_encoder_sim` PASS) |
| MM12 | C only | `# HEURISTIC` on the x^c_tau = 0 reset (`cl` 66), which the thesis derives | **C (marker only)**, ruled and relabelled `# THEORY` (D-284 amendment 6) |
| MM13 to MM15 | C only | closed-form toggle (D-284 amendment 3), `Lambda_combo` buffer (checkpoint key), `Y_op` branch (evaluation interface) | none |
| MM16 | X M2, C MM5 | Precision of the normalisation and nominal constants: `data.compute_normalization` 175 inverts the float32 `P` before promotion, so mu_x, sigma_x use a float32-accurate P^-T while the tuples and the encoder start use the float64 `P_INV_T64`; P^T mu_x = mu_y then holds to about 1e-7 relative, so y_hat^n = C^n x^n carries a tiny constant offset; sigma carries + 1e-8; `ss` stores the nominal parameters in float32 (theta^0 differs from Table I by about 1e-7 relative). Tests `A_normalization` (1.0e-5), `A_norm_output` (9.1e-9) fail on it. C missed the P^-T part. | **C, fixed** (Dirk 2026-10-28, D-284 amendment 6): measured offset 1.1e-13 m on X1, X2 and 0 on Y (1.2e-8 of the smallest servo-error sigma); `data.compute_normalization` 175 now inverts after promotion, a rounding item. Both tests pass (71 of 72). The float32 storage of the nominal constants in `ss` and the + 1e-8 stay (not ruled; the nominal constants are the model's theta^0 in every thesis run). |

Counts: code side 2 (MM12 marker and MM16 precision, both ruled and done); thesis side 3 plus 1 conditional (MM2, MM6, MM7;
MM8); equivalent or ruled 8 (MM1, MM3, MM4, MM10, MM11, MM13, MM14, MM15). Map rows the two models disagreed on: 16
backward rows (section 2), 1 scope row (section 0), 4 forward rows only in C (02:706, 04:263, tab:setup_models,
03:1913), 2 mismatch side classifications (MM1, MM3) and 1 mismatch only X found (MM16 P^-T part); 5 mismatches only C
found (MM2, MM6, MM7, MM8, MM12).

## 4. For Dirk
1. MM16: ruled 2026-10-28, fixed (D-284 amendment 6).
2. MM12: ruled 2026-10-28, relabelled `# THEORY` (D-284 amendment 6).
3. Thesis side (no edit here): MM2 one sentence on the basis rebuilt at the selected checkpoint and held in the polish;
   MM6 Adam epsilon 1e-16 in tab:hyper; MM7 the 5 ms record trim.

## 5. Planned test edits (item 8)
`P_s2_cost`: expected x (B - 1)/B (MM3, decided). `A_normalization`, `A_norm_output`: no edit, they pass after the
MM16 fix.
`equation_tests.py` is Codex's step-4 file; the edits are made in item 8 and named in the implementation report.

## 6. Codex comment check (`comment-check2-codex.md`, 138 findings): applied and answered
Applied 122, agreed but open or deferred 5 (F008 to F011, F013), answered 11; comments, docstrings, markers and this
map only, no code changed:
- Q1: F001 (scope labels, section 0), F003, F004 (added interconnect callables, section 2).
- Q2: F006 (C_y row), F007 (second recovery line derivation only), F012 (current test and ruling status). F008 to F011
  stay open for Dirk (MM2, MM6, MM7, MM8; section 4). F013 is item 8 (MM3, section 5).
- Q3, all four: F014 (backward-difference velocity with qdot_0 = qdot_1), F015 (the implemented J and its theta_aux
  coordinates in `obc.py` and the loss comment), F016 (S2 docstring with (B - 1)/B), F017 (the reloaded model carries
  the history saved with the best dump; deepSI's load replaces `__dict__`).
- Q4: F020 to F028 (module conventions), F029 to F080 (52 missing docstrings), F081 to F086, F088, F090, F092 to F098,
  F100 to F128, F130, F133 (formula labels and docstrings).
- Q5, all five: F134 `Linear_Output_Block` is Jan's (`@added` removed, `# CHANGED:` on its changed parts); F135
  `deepSI_corrections.py` holds Jan's renamed classes (`__project_origin__` replaced by a `# CHANGED:` header, transform
  marked); F136 to F138 (import-group reasons corrected). `jan-changes-audit.md` (step-3 file, not edited) lists
  neither class; flagged for Dirk.

Answered, not applied (11), one line each:
- F002: the per-callable inventory with current anchors is X's backward table (`thesis-code-map-codex.md`); section 2
  overrides it in the 16 decided rows, so it is not repeated here; anchors are pre-rewrite by design (header).
- F005: column X already splits these displays per line (A_norm_*, A_init_*, O_basis to O_orthogonality, L_G_*), each
  with its test; the code column lists the callables per part in the same order.
- F018, F019: R2 forbids thesis equation numbers; Kessels Eqs. (5.12), (5.13) and Hoekstra Eq. (8) are paper sources,
  which a `# THEORY:` label must name.
- F087, F089, F091, F099, F131, F132: the docstring already states the formula of a one-statement body; a label would
  repeat it, and spelling out the adjugate entries of F099 would duplicate the code.
- F129: the docstring now states chi_tau and the futures; the slices beneath are those definitions.

Effect on item 8: besides the two removed decorators, the AST comparison against `implemented` must also accept the
removed `@added` of `Linear_Output_Block` and the removed `__project_origin__` assignment in `deepSI_corrections.py`
(F134, F135), and the MM16 fix in `data.py`.

### Codex comment check 4 (`comment-check4-codex.md`, 40 findings), 2026-10-30
All 40 applied; four in other words, one line each:
- F009: `sample_ix` indexes the record as `_record_at_fs_new` returns it (no claim about trimming or truncation).
- F010: the input stays `u` (the code's `u_stage`), not `u_act`.
- F019: the existing xdot^n = xdot / sigma_x stays, with "time derivative of x_tilde^n in 1/s"; no new F^n = T_x F.
- F021: split into the two branches as proposed, but the LFR branch keeps "kept because the thesis results used it"
  (Dirk 2026-10-28).
Also merged four further stacked header pairs found by a scan for a numbered header followed by a comment repeating its
label: `model.py` 3. Check, `blocks.py` 1. Nominal matrices, `closed_loop.py` 1. Normalise, `obc.py` 1. Column scale.
Proof: stripped AST (docstrings removed, comments not in the AST) equal on all 24 files against the pre-edit `final`
copy; `equation_tests.py` 72 of 72; ruff F clean.

## 7. Item 5b (D-284 amendment 7), 2026-10-28
71 math and public callables rewritten to name, formula, role, `Args:`/`Returns:` (shapes, normalised or physical,
thesis symbols), `least_squares_start` exactly as the agreed example; 29 numbered phase headers in the multi-stage
functions (`least_squares_start`, `prefit_physical_encoder`, `build_model`, `compute_normalization`,
`build_reference_set`, `closed_loop_free_run_rms_batch`, `AugmentedModel.loss`, `lbfgs_polish`, `PS2Opening.after_step`,
`_fit_added_state_transition`). Private helpers with obvious arguments and infrastructure keep one sentence. Proof:
stripped AST of all 24 files identical to the tree after item 5 and the Codex fixes; ruff F clean; imports exit 0;
equation tests 71 of 72 (P_s2_cost, item 8).

## 8. Codex comment check 3 (`comment-check3-codex.md`, 42 findings): all applied
Q3 F001 (encoder bounds in thesis symbols, m = (n + 1)(n_y + n_u), code n_y, n_u are the flattened widths) and F002
(PS ends at the screen_at check too, keeping S1's encoder changes); Q4 F003 to F042 (Args/Returns with shapes and
units, the matrix-tuple schema and the units convention once in `blocks.py`, numbered phases in 15 functions).
Nothing answered or declined. Proof: stripped AST of all 24 files identical to the tree after item 5; ruff F clean;
imports exit 0.
