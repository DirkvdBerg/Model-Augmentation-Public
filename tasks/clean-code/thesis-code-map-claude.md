# Thesis-to-code map (Claude, D-284 step 4 item 2)

Independent of `thesis-code-map-codex.md` (not read). Merged later into `thesis-code-map.md`.

## 0. Revision and scope
- Thesis: `git log -1 --format=%h Thesis-writeup/Writing` = `6b77c9a`; `git status --short Thesis-writeup/Writing` empty
  (working tree equals `6b77c9a`).
- Code: the 24 grey-box files of `Clean/` as on 2026-10-28 (all comments still the D-282 ones; step 4 has changed no
  `.py` under `Clean/`). Line numbers below are of these files.
- Active text only: lines starting with `%`, `\ifdraft` blocks, `\del{...}`, `\delmv{...}{...}` and struck displays
  are out. `\mv{...}{...}`, `\add`, `\chg{old}{new}` (new part) are in.
- Thesis IDs: the label, or `<file>:<line>` for an unlabelled display or an algorithm step stated in text (file
  `02` = `02_system_baseline.tex`, `03` = `03_augmentation.tex`, `04` = `04_obc.tex`, `05` = `05_setup.tex`,
  `99` = `99_appendix.tex`; line = first line of the display or sentence).
- Appendix subsections included: `app:params`, `app:lfr-details`, `app:lfr-wellposed`, `app:combos`,
  `app:setup_system`, `app:hyper`. Excluded (results, data generation design, black box): `app:splits`,
  `app:setup_records`, `app:setup_checks`, `app:setup_bb` (`eq:setup_bb`, D-285 session).
- Paths below are relative to `Clean/`: `blocks` = `model_augmentation/fit_systems/blocks.py`, `ic` =
  `.../interconnect.py`, `cl` = `.../closed_loop.py`, `obc` = `.../obc.py`, `enc` = `.../pre_encoder.py`, `ps2` =
  `.../ps2_opening.py`, `lbfgs` = `.../lbfgs_polish.py`, `ss` = `model_augmentation/systems/gantry_ss.py`, `nets` =
  `model_augmentation/utils/torch_nets.py`, `utils` = `.../utils/utils.py`, `dsc` = `.../utils/deepSI_corrections.py`;
  `cfg`, `data`, `ctrl`, `model`, `obcg`, `train`, `report` = `gantry_dynamic/{config,data,controller,model,
  obc_gantry,training,closed_loop_report}.py`; `entry` = `gantry_interconnect_dynamic.py`.

Code symbols that differ from the thesis (used in the comments as "code name is symbol"): `alpha` = m_Sigma + m_h
(= M_0,11), `beta` = (L_b/2) m_Delta (the a of `app:lfr-wellposed`), `gamma_` = J_eff, `M_DIFF_RHO` = gamma (the
margin), `m_total`/`m_diff`/`kb_sum`/`cb_sum` = m_Sigma/m_Delta/k_b,Sigma/c_b,Sigma, `free_params`/`free` = beta
(training coordinates), `vbar` = beta_lin, `theta_frozen` = theta_aux, `fnet` = u - C qdot - K q, `a` = qddot,
`E` = the physical-row selector, `xc` = x^c, `M` (PS2) = H, `nu` (`_cal_unexplained`) = nu, `std_x`/`x_mean` =
sigma_x/mu_x, `Cd_norm` = C^n, `n_a`/`NX_ANN` = n_xbar, `na_nb` = n + 1.

## 1. Forward: thesis entry to code

### 02_system_baseline.tex (Sections II, III)
| ID | Content | Code | Note |
|-|-|-|-|
| eq:system (02:168) | xdot = f(x, u_act), y_k = q_act(kT_s) + v_k | external: `Clean/data_generation/` (Simulink truth) | problem statement |
| eq:system_loop (02:193) | discrete controller K_fb (A_fb, B_fb, C_fb, D_fb) on r - y, plus u_ff | `ctrl.controller_ss` 73, `ctrl.build_cfb_at` 59; loop itself external (data generation) | K_fb matrices only |
| 02:197 (hold) | u_act held over T_s | `blocks.GantryStateBlock._rk4` 128 (u fixed over the RK4 step); `data._resample_u` 84 | |
| eq:Ptransform (02:409) | q = P^-T q_act, u = P u_act, P | `ss.P` 52, `ss.Cd` 60 (P^T); `obcg.P_INV_T64` 24; `data.compute_normalization` 175; `model.prefit_physical_encoder` 143; `blocks._deriv_with` 151 (u_log = P u); `ctrl.P` 27 | |
| 02:420 (assumptions) | small yaw, no Coriolis | derivation only | |
| eq:eom (02:437) | M(Y) qddot + C qdot + K q = u | `blocks.GantryStateBlock._deriv_with` 139 | solved for qddot, see eq:base_ss |
| eq:mass_matrix (02:454) | M(Y) entries | `ctrl.mass_matrix_at` 34 (float, controller design); in the model only through `ss.build_poly_constants` 65 (alpha, beta, gamma_, adj M) | |
| eq:damping_matrix (02:464) | C entries | `ss.C` 41; `blocks._TrainableGantryStateBlock._build_KC` 293; `ctrl.C_DAMP` 28 | |
| eq:stiffness_matrix (02:473) | K entries | `ss.K` 48; `blocks._build_KC` 293; `ctrl.K_STIFF` 31 | |
| eq:Msplit (02:557) | M(Y) = M_0 + Y M_1 + Y^2 M_2 | `ss.M1`, `ss.M2` 33 to 38; `blocks._build_M1M2` 313; `M_0^-1 = N0 / d0` in `ss.build_G_matrix_entries` 99 | M_0 never built explicitly |
| eq:base_ss (02:585) | xdot = F(x, u) = (qdot, M(Y)^-1 (u - C qdot - K q)) | `blocks.GantryStateBlock._deriv_with` 139 (M(Y)^-1 = (N0 + Y N1 + Y^2 N2)/d(Y), `ss.build_poly_constants` 65); tangent `_deriv_with_tangent` 193 | default path returns (qdot, a) through the LFR output equation, MM16 |
| 02:615 (LPV, self-scheduled) | Y is state and scheduling variable | `_deriv_with` 164 (Y = x2[:, 2] per RK4 stage) | |
| 02:645 (LFR exists) | pointer to app:lfr-details | see 99 rows | |
| 02:745 (equation*) | (M - M') qddot + (C - C') qdot + (K - K') q = 0 | derivation only | |
| eq:phi (02:796) | theta_base = (k_b,Sigma, c_g1, c_g2, c_y, c_b,Sigma, m_h, m_Sigma, m_Delta, J_eff, d) and its definitions | `blocks.IdentifiableGantryStateBlock.COMBO_NAMES` 356, `.combos_of` 388 | order identical |
| 02:706 (L_b fixed) | L_b at nominal | `ss.Lb` 27; `blocks._TrainableGantryStateBlock.__init__` 291 (buffer, not trained) | |
| eq:lfr_admissibility (02:840) | m_Sigma J_eff > (L_b^2/4) m_Delta^2 | enforced by `blocks._m_diff_bound` 415 with `combinations_from_free` 420 | |
| 02:854 (log coords) | theta_base,j = theta^0_base,j exp(beta_j), beta = 0 at start | `blocks.combinations_from_free` 423; `IdentifiableGantryStateBlock.__init__` 371 (zeros) | |
| eq:mdelta_map (02:867) | m_Delta = gamma (2/L_b) sqrt(m_Sigma J_eff) tanh(eta_0 + s beta_mDelta) | `blocks.combinations_from_free` 424, `._m_diff_bound` 415, `M_DIFF_RHO` 362 | gamma is `M_DIFF_RHO`, MM4 |

### 03_augmentation.tex (Section IV)
| ID | Content | Code | Note |
|-|-|-|-|
| eq:aug_transition (03:377) | x_{k+1} = f_base + f_aug, xbar_{k+1} = g_aug, y_hat = P^T q | `model.build_model` 59 to 68 (interconnect wiring), `ic.Interconnect.forward` 54, `._propagate` 85 | |
| 03:450 to 457 (RK4) | f_base = one RK4 step of F over T_s, u held, M(Y) at each stage's Y | `blocks._rk4` 128 (`up_sample` = 1), `_deriv_with` 164 | |
| 03:475 to 502 (network) | phi_aug MLP, tanh hidden, linear output; inputs (x, xbar, u_act); first six outputs correct x, rest are xbar_{k+1} | `nets.AugmentationNetwork` 29, `nets.build_mlp` 11, `blocks.Static_ANN_Block` 22; routing `utils.expansion_matrix` 19 with `cfg.ann_route_ix` 175 | |
| eq:aug_norm (03:536) | x^n = T_x (x - mu_x), u^n, y^n; T_i = diag(sigma_i)^-1 | `data.compute_normalization` 169; `dsc.SystemDataNormalizer.transform` 41 (u, y); `cl.closed_loop_free_run_rms_batch` 195 | sigma + 1e-8, MM5 |
| 03:521 to 524 | mu, sigma from D_N; q = P^-T y, velocities by finite difference / T_s | `data.compute_normalization` 177 to 184 | D_N read as the training records |
| 03:556 | xbar not normalised | `obcg.build_reference_set` 80 (zeros), model state layout | |
| eq:aug_norm_maps (03:613) | normalised model, C^n = T_y P^T [I 0] T_x^-1, f^n_base = T_x (f_base - mu_x) | `blocks._deriv_with` 145 (denormalise), 190 (xdot / sigma_x); `data.compute_normalization` 194 (C^n); `blocks.Linear_Output_Block` 48 | RK4 in x^n equals T_x(f_base - mu_x) exactly (affine equivariance) |
| 03:637 (mu_x positions) | position means = P^-T mu_y, so y offset cancels | `data.compute_normalization` 186 (mean of P^-T y = P^-T mu_y, to rounding) | |
| 03:665 (display) | (f^n_aug, g^n_aug) = phi_aug(x^n, u^n) | `blocks.Static_ANN_Block.forward` 40, `nets.AugmentationNetwork.forward` 40 | network outputs f^n_aug directly |
| eq:aug_two_loops (03:828) | model controller on r - y_hat | derivation; implemented in residual form eq:aug_residual_loop | |
| eq:aug_residual_loop (03:896) | x^c_{k+1} = A x^c + B (y - y_hat), x^c_tau = 0, u_hat = u + C x^c + D (y - y_hat) | `cl.ControllerBank.step` 37, `cl.closed_loop_step` 46, `cl.closed_loop_rollout` 68 (x^c = 0) | |
| eq:aug_residual_split (03:927) | u_hat = closed-loop model + (u - u_check) | derivation only | |
| 03:943 (order) | output, then error and input, then state update | `cl.closed_loop_step.step` 49 | |
| eq:aug_cl_step (03:965) | normalised: eps = T_y^-1 (y^n - y_hat^n), u_hat^n = u^n + T_u (C x^c + D eps), x^c update | `cl.ControllerBank.__init__` 26 to 31 (B diag(sigma_y), diag(1/sigma_u) C, diag(1/sigma_u) D diag(sigma_y)), `.step` 37 | x^c kept physical |
| eq:aug_loss (03:1135) | V = mean over windows, steps, axes of ||W_r (y^n - y_hat^n)||^2, constraints, encoder start | `ic.AugmentedModel.loss` 517; windows `dsc.WindowedSystemData.to_hist_future_data` 12, `ic.SSE_Interconnect.make_training_data` 343 | |
| 03:1149 (W_r) | W_r = diag(sigma_y / sigma_e,r), sigma_e of y - r | `ctrl.build_closed_loop` 161, `data.servo_sigma` 130, `cl.ClosedLoopSimulator.augment_training_data` 147 | sigma_e over k0 .. N-1 |
| eq:aug_encoder (03:1166) | x_hat^n_tau = psi(chi_tau), chi = n + 1 past y^n and u^n, up to tau - 1 | `enc.AugmentedEncoder.forward` 61; history windows `dsc.to_hist_future_data` 25 to 26; `model.build_model` 79 (na - 1) | |
| eq:aug_enc_lin (03:1193) | psi = [W^b; W^a] chi + psi_tilde(chi) | `enc.AugmentedEncoder.forward` 67 to 77 | psi_tilde reads (u, y), MM10 |
| 03:1266 (Adam) | Adam on minibatches of windows, gradient over the full window | `ic.SSE_Interconnect.fit` 357 (closure 418), `ic.SSE_Interconnect.init_model` 330 | |
| 03:1291 to 1300 (selection) | validation on D_val in closed loop, encoder once on first n + 1 samples, lowest validation error kept | `cl.ClosedLoopSimulator.validation_error` 166, `cl.closed_loop_free_run_rms_batch` 178, `ic.SSE_Interconnect.fit.validation` 361 | |
| 03:1304 to 1309 (polish) | L-BFGS on a fixed set of windows from the selected parameters, kept only if validation error drops; unconstrained | `lbfgs.lbfgs_polish` 131, `lbfgs.FixedWindowBatch` 35, `lbfgs._acceptance` 120, `train.run_lbfgs_polish` 192 | |
| 03:1342 (start from baseline) | physical rows from the baseline | see eq:aug_init | |
| eq:aug_enc_sim (03:1368) | closed-loop baseline simulation at theta^0 with eq:aug_cl_step | `model.prefit_physical_encoder` 118 | full start model, MM11 |
| 03:1387 to 1390 | record start: q = P^-T y, zero velocity, xbar, x^c | `model.prefit_physical_encoder` 142 to 144; x^c in `cl.closed_loop_rollout` 68 | |
| eq:aug_enc_fit (03:1401) | (W^b0, b^b0) = argmin mean ||W^b chi^0 + b - x^n,0||^2 | `model.least_squares_start` 175 (centred, standardised lstsq, gelsd) | |
| 03:1413 (chi^0, D) | chi^0 from y_hat^n,0 and u_hat^n,0; D = fit samples | `model.prefit_physical_encoder` 147 to 165 (stride `cfg.stride`) | |
| 03:1432 to 1436 | psi_tilde physical output bias = b^b0, weights 0 | `model.least_squares_start` 203 to 205 | |
| eq:aug_init (03:1486) | beta = 0; W^xtilde_L = 0, b_L = 0, W^xbar_L ~ U(-1/sqrt n_h, 1/sqrt n_h); encoder W^b = W^b0, psi_tilde rows; W^a Xavier, a = sqrt(6/(n_xbar + 6(n+1))) | `blocks.IdentifiableGantryStateBlock.__init__` 371; `nets.AugmentationNetwork.__init__` 33 to 38; `nets.build_mlp` 24 (b_L = 0); `enc.AugmentedEncoder.__init__` 40 to 59; `model.least_squares_start` 199 to 205 | |
| 03:1506 (hidden draws) | hidden layers of phi_aug and psi_tilde ~ U(-1/sqrt m, 1/sqrt m) | `nets.build_mlp` 14 (nn.Linear default) | |
| eq:ps2_s1 (03:1710) | min over theta_p, theta^a_enc of mean ||p(psi_a(chi)) - e_tau||^2, e over H = n + 1 samples / sigma_e | `ps2.PS2Opening._s1_step` 177, `._residual` 166, `._start` 130 (decoder, one hidden layer), `._added_states` 173 | |
| 03:1671 to 1672 | S1 minibatch of tau from D_N; left-out records | `ps2.PS2Opening._sample` 141, `ps2.calibration_split` 42 | |
| 03:1811 to 1813 | S1 ends when left-out error stops improving; S2 only if decoder explains at least 10 % | `ps2.PS2Opening.after_step` 82, `ps2._plateaued` 37, `._cal_unexplained` 192, `PS2Spec` 14 | |
| eq:ps2_alt (03:1829) | one Adam step on V then one on S1, theta^a_enc from S1 replaces | `ic.SSE_Interconnect.fit` 430 to 432 calling `ps2.PS2Opening.after_step` 82 | |
| eq:ps2_s2 (03:1890) | P = {(x_hat^n_k, u^n_k, xbar_{k+1})}, min over theta_aug of normalised one-step error | `ps2.PS2Opening._fit_added_state_transition` 199 (`snap` 207, `state_map_loss` 213) | scale (B - 1)/B, MM3 |
| 03:1913 | keep theta_aug with lowest left-out cost | `ps2._fit_added_state_transition` 227 to 245 | start is a candidate |
| eq:ps2_handover (03:1952) | theta_PS = (beta^(i1), theta^S2_aug, theta^(i1)_enc); S2 skipped: after S1 | `ps2.PS2Opening.after_step` 103 to 113 (phase S3) | |

### 04_obc.tex (Section V)
| ID | Content | Code | Note |
|-|-|-|-|
| eq:obc_gyorok (04:144) | Gyorok projection, linear in theta | derivation only (literature) | |
| eq:obc_sens (04:194) | varphi = d f^n_base / d beta at beta_lin | `obc.build_stacked_sensitivity` 56 (jacfwd), `blocks.transition_from_free` 449, `obcg.make_basis_builder` 145 | |
| eq:obc_offset (04:211) | f^n_base ~ varphi beta + gamma, gamma = f^n_base(beta_lin) - varphi beta_lin | `obcg.make_basis_builder.build` 153 to 158 (column f^n_base(beta_lin), not gamma) | MM1 |
| eq:obc_tuples (04:252) | x_i = (q_i, qdot_i), q = P^-T y, fourth-order central difference | `obcg.fourth_order_central_difference` 42, `obcg.build_reference_set` 49 | |
| 04:255 to 261 | tuples with recorded u_i, xbar = 0, normalised, once before training, all samples with qdot at i and i + 1 | `obcg.build_reference_set` 63 to 80; `obcg.attach_obc` 185 | training records |
| 04:263 (rank) | Phi_Gamma full column rank (assumed) | `obc.OBCBasis.__init__` 18 (rank check), `obc.OBCProjection.refresh` 94 (abort rule) | |
| eq:obc_coef (04:278) | Phi_Gamma, F^n_aug at (x^n_i, 0, u^n_i), theta_aux = (Phi^T Phi)^-1 Phi^T F, F_tilde, orthogonality | `obc.OBCBasis.coefficient` 38, `obc.OBCProjection.coefficient` 124, `obcg.make_reference_field` 163 | SVD form; equals the normal equations at full rank |
| eq:obc_corr (04:316) | f^n_obc = [varphi gamma] theta_aux; x_{k+1} = f_base + f_aug - f_obc on physical rows; xbar = g_aug | `obcg.GantryOBCCorrection.forward` 128, `.tangent_pair` 122, `blocks._rk4_with_tangent` 258, `blocks._deriv_with_tangent` 193, `ic.Interconnect.forward` 76 to 77 | varphi a as a directional derivative |
| 04:325 | theta_aux differentiated through Gyorok Eq. (8) | `ic.AugmentedModel.loss` 526 (graph through `OBCBasis.coefficient`) | |
| 04:341 to 346 | projected update in V; theta_aux at every V; beta_lin = beta at each epoch start, fixed in the epoch | `ic.AugmentedModel.loss` 521 to 527, `obc.OBCProjection.maybe_refresh` 119, `ic.AugmentedModel._epoch_index` 489, `obcg.GantryOBCCorrection.expansion_point` 112 | |
| eq:obc_eps (04:380) | Xi = stacked one-step residual at beta_star | external: `Thesis-writeup/Code/Evaluation/eval_overlap.py`; inputs from `obcg.GantryReferenceSet.x_phys_next_rec`, `report._true_free` 26 | |
| eq:obc_recovery (04:398) | first-order estimates | derivation only | |
| eq:obc_overlap (04:456) | rho(Z) = ||Pi Z|| / ||Z|| | external: `Thesis-writeup/Code/Evaluation/eval_overlap.py` | |

### 05_setup.tex (Section VI)
| ID | Content | Code | Note |
|-|-|-|-|
| eq:setup_truth, eq:setup_fric (05:117, 05:123), eq:setup_abs (05:156) | data-generating system | external: `Clean/data_generation/` (D-286) | |
| 05:229 | K_fb = K_1 by Tustin at T_s | `ctrl.build_cfb_at` 68 (bilinear), `ctrl.controller_ss` 73 | |
| 05:235 | u_ff = injected force only | external (data generation) | |
| 05:323 to 326 | downsample to T_s = 0.25 ms, servo error zero-phase Butterworth order 8 at 1.6 kHz, force averaged per sample | `data._decimate_y` 96, `data._resample_u` 84, `cfg.AA_ORDER` 22, `cfg.fs_new` 96 | noise-free: point sampling, MM8; trim, MM7 |
| 05:365 | 18 train, 6 val, 11 test | `data.record_files` 24 (manifest split) | |
| tab:setup_models (05:464) | M_LPV, M_U, M_OBC | `cfg.PRESETS` 29 (arms); `report._without_learned` 11, `report._true_free` 26 (M_LPV, evaluation) | |
| 05:497 | detuned start: 10 % larger or smaller, same signs | `cfg.RunConfig.combo_init_detune` 183, `model.build_model` 39 to 40 | |
| 05:502 | one seed | `cfg.RunConfig.seed` 91, `entry.main` 92 to 107 | |
| eq:setup_rms (05:596) | RMS over k = n + 1 .. N - 1, NRMS_e = RMS / sigma_e | validation: `cl.ClosedLoopSimulator.validation_error` 166, `data.servo_sigma` 130; test metrics external (evaluation) | |

### 99_appendix.tex (included subsections)
| ID | Content | Code | Note |
|-|-|-|-|
| tab:nominal_parameters (99:28) | nominal values (upper block) | `ss` 9 to 28; `ctrl` 19 to 21 | lower block external |
| 99:63 (equation*) | M_1, M_2 | `ss.M1`, `ss.M2` 33 to 38; `blocks._build_M1M2` 313 | |
| eq:lfr_mass_factorization (99:94), eq:lfr_chain (99:101), eq:lfr_constant (99:108) | LFR construction | derivation; z = (a, Y a), w = Y z in `blocks._deriv_with` 184 to 185 | |
| eq:delta (99:121) | w = Delta(Y) z, Delta = Y I_6 | `blocks._deriv_with` 185 | |
| eq:lfr_G (99:129), eq:app_lfr_blocks (99:186) | G blocks A_x, B_w, B_u (first row), C_z, D_zw, D_zu, C_y | first block row: `ss.build_G_matrix_entries` 91 (A_x, B_w, B_u, A_combined, R = N0 / d0), used in `blocks._deriv_with` 187 to 188; C_z, D_zw, D_zu not built (loop solved in closed form); C_y = `ss.Cd` | |
| 99:199, 99:209 | I - D_zw Delta; det = det M(Y) / det M_0 | derivation only | |
| 99:245 (equation*) | M(Y) in theta_base | `ss.build_poly_constants` 67 to 69 (alpha, beta, gamma_) | symbols, MM4 |
| 99:252 to 267 | Sylvester; det M(Y)/m_h = ... | `blocks._deriv_with` 165 to 167: d(Y) = m_h (alpha gamma - beta^2 + 2 beta m_h Y + m_h (alpha - m_h) Y^2), equal to det M(Y) | |
| 99:276 to 287 | b, r_0, eta_0 = atanh(r_0), s = m_Delta,0 / (b_0 (1 - r_0^2)) | `blocks.IdentifiableGantryStateBlock.__init__` 379 to 381 | |
| 99:297 (equation*) | matrices in theta_base | `blocks.gauge_section` 399 then `_mats_from_raw` 328 (any raw vector with these combinations gives the same matrices) | |
| 99:311 (equation*) | theta_base from the matrix entries | derivation; numerically `blocks.combos_of` 388 on the raw vector | |
| eq:setup_K1 (99:391) | K_1,j(s) = kappa_j (s + w_b/6)/s (s + w_b/3)/(s + 3 w_b) 10 w_b/(s + 10 w_b) | `ctrl.normalized_controller_coefficients` 52 | |
| 99:393 (kappa_j) | unit open-loop gain at w_b, nominal baseline at Y = 0 | `ctrl.build_cfb_at` 66 to 67, `ctrl.stage_frequency_response` 43 at `ctrl.y_op_for` 96 (stored design Y, 0 for every record: `Clean/data_generation/tdg_config.m:55`) | |
| 99:371 (M_e) | data-generating matrices | external | |
| tab:hyper (99:693) | settings | `cfg.RunConfig` 72 to 129, `ps2.PS2Spec` 14; see the defaults check below | |

Defaults check against `tab:hyper`: 4 kHz (`fs_new`), 1 RK4 step (`up_sample`), n_f = 400 (`nf`), stride 10, n + 1 = 30
(`na_nb`), n_xbar in {0, 2, 8} (`__post_init__`), 2 / 16 / tanh, gamma = 0.99 (`M_DIFF_RHO`), detune 10 %, Adam 1e-5,
batch 512, validation every 1300 (`its_per_val`), L-BFGS 100 / 50 / 20 / 16384, tolerances 2.6e-5 / 2.6e-7, seed 1;
PS: lr 1e-3, S1 batch 64, decoder width 16 (`width` = n_h), H = n + 1, check every 25 on 128 windows, S1 100 to 350,
2 % over 4 checks, 10 % (`screen_max` 0.90), check at 150, pairs 8192 / 2048, S2 batch 512, S2 500 to 8000, 1 % over 5
checks 100 apart. All match. Updates 33 400 = `epochs` 200 x updates per epoch (data dependent, not checked here).

## 2. Backward: code to thesis
Infrastructure = I/O, logging, CLI, checkpointing, framework plumbing, checks. "Ext" = used outside `Clean/`.

### gantry_interconnect_dynamic.py
| Callable | Entry |
|-|-|
| `_dumps` 21, `_rerun_source` 27, `_archive_rerun` 46, `cli` 131 | infrastructure (rerun folders, CLI) |
| `main` 69 | infrastructure (run order); seeds: 05:502 |

### gantry_dynamic/
| Callable | Entry |
|-|-|
| `__init__` (module guard) | infrastructure |
| `report._without_learned` 11 | tab:setup_models (M_LPV), Ext evaluation |
| `report._true_free` 26 | eq:obc_eps (beta_star), Ext evaluation |
| `cfg.y_decimation` 51, `cfg.record_trim` 58 | 05:323 (record), MM7 |
| `cfg.RunConfig` 64, `.__post_init__` 131, `.from_config_json` 145, `.obc` 167, `.nx_ann` 171, `.mode` 179, `.hp` 192 | infrastructure; fields: tab:hyper |
| `cfg.RunConfig.ann_route_ix` 175 | 03:486 (outputs to all states in order) |
| `cfg.RunConfig.combo_init_detune` 183 | 05:497 |
| `cfg.save_dir` 207, `cfg.git_provenance` 213 (`_git` 215), `cfg.config_json_dict` 239 | infrastructure |
| `ctrl.mass_matrix_at` 34 | eq:mass_matrix (at Y_op) |
| `ctrl.stage_frequency_response` 43 | 99:393 (P^T G(s) P of eq:base_ss frozen at Y) |
| `ctrl.normalized_controller_coefficients` 52 | eq:setup_K1 |
| `ctrl.build_cfb_at` 59 | eq:setup_K1, 99:393, 05:229 |
| `ctrl.controller_ss` 73 | eq:system_loop (A_fb, B_fb, C_fb, D_fb) |
| `ctrl._stem` 80, `ctrl.register_stored_controller` 90, `ctrl.y_op_for` 96, `ctrl.gain_for` 111 | infrastructure (controller metadata of a record) |
| `ctrl.build_controller_bank` 122 | eq:aug_cl_step (one controller per record) |
| `ctrl.build_closed_loop` 133 | 03:1149 (W_r), eq:setup_rms (sigma_e of validation); compile set-up infrastructure |
| `data.record_files` 24, `data._register_controller` 36, `data.traj_dir` 74, `data._load_u` 79, `data._load_record` 120, `data.load_datasets` 140 (`_split` 147) | infrastructure (I/O); 05:365 |
| `data._resample_u` 84 | 05:326 (force averaged), 02:197 |
| `data._decimate_y` 96 | 05:325 |
| `data._record_at_fs_new` 107 | 05:323 |
| `data.servo_sigma` 130 | eq:setup_rms (sigma_e), 03:1149 |
| `data.compute_normalization` 169 | eq:aug_norm, 03:521, 03:637, eq:aug_norm_maps (C^n) |
| `model.build_model` 24 | eq:aug_transition, eq:aug_enc_lin, 05:497, eq:aug_init (beta = 0) |
| `model.on_device` 107 | infrastructure |
| `model.prefit_physical_encoder` 118 (`recorder` 147) | eq:aug_enc_sim, 03:1387, 03:1413 |
| `model.least_squares_start` 175 | eq:aug_enc_fit, 03:1432, eq:aug_init (encoder rows) |
| `obcg.GantryReferenceSet` 28, `.n` 38 | 04:255 (tuple set) |
| `obcg.fourth_order_central_difference` 42 | eq:obc_tuples |
| `obcg.build_reference_set` 49 | eq:obc_tuples, 04:255 |
| `obcg.GantryOBCCorrection.__init__` 94 | eq:obc_corr (physical rows only) |
| `obcg.GantryOBCCorrection.expansion_point` 112 | 04:345 (beta_lin = beta) |
| `obcg.GantryOBCCorrection.set_frozen` 117 | eq:obc_coef (theta_aux stored for validation) |
| `obcg.GantryOBCCorrection.tangent_pair` 122, `.forward` 128 | eq:obc_corr |
| `obcg.make_float64_block` 140 | infrastructure (float64 basis) |
| `obcg.make_basis_builder` 145 (`build` 148) | eq:obc_sens, eq:obc_offset (MM1), eq:obc_coef (Phi_Gamma) |
| `obcg.make_reference_field` 163 (`field` 169) | eq:obc_coef (F^n_aug at xbar = 0) |
| `obcg.attach_obc` 181 | 04:255 to 261 (once before training), infrastructure wiring |
| `train._json_safe` 17, `train.save_checkpoint_pair` 30, `train.resolve_checkpoint` 42, `train.load_optimizer_moments` 71, `train._assert_same_architecture` 86, `train._assert_same_dtype` 109, `train.load_checkpoint` 120 | infrastructure (checkpoints, resume) |
| `train.ValidationHistory` 151 (`__init__` 157, `record` 162, `hist_arrays` 174, `__call__` 187) | infrastructure (logging of 03:1291 validations) |
| `train.run_lbfgs_polish` 192 | 03:1304 |
| `train._adam_phase` 230 | 03:1266, eq:ps2_alt (opening attached) |
| `train.train` 276 | 03:1291 to 1309 (Adam, selection, polish, final choice) |

### model_augmentation/
| Callable | Entry |
|-|-|
| `utils.added` 8, `utils.to_tensor` 14 | infrastructure |
| `utils.expansion_matrix` 19, `utils.selection_matrix` 25 | eq:aug_transition (routing to x_tilde, xbar rows; selection of x_tilde) |
| `nets.build_mlp` 11 | 03:475 (MLP), eq:aug_init (b_L = 0), 03:1506 |
| `nets.AugmentationNetwork.__init__` 33 | eq:aug_init (W^xtilde_L = 0) |
| `nets.AugmentationNetwork.forward` 40 | 03:665 |
| `dsc.WindowedSystemData.__init__` 9 | infrastructure |
| `dsc.WindowedSystemData.to_hist_future_data` 12 | eq:aug_loss (windows tau, n_f, stride), eq:aug_encoder (chi up to tau - 1) |
| `dsc.SystemDataNormalizer.__init__` 35, `.transform` 41 | eq:aug_norm (u^n, y^n) |
| `ss` constants 9 to 28 | tab:nominal_parameters |
| `ss.M1`, `ss.M2`, `ss.C`, `ss.K`, `ss.P`, `ss.Cd`, `ss.Dd` | 99:63, eq:damping_matrix, eq:stiffness_matrix, eq:Ptransform, eq:aug_transition (y = P^T q, D = 0) |
| `ss.build_poly_constants` 65 | eq:base_ss (M(Y)^-1 = (N0 + Y N1 + Y^2 N2) / d(Y)), 99:245 |
| `ss.build_G_matrix_entries` 91 | eq:app_lfr_blocks (first block row) |
| `blocks.Block.__init__` 15 | infrastructure (Jan's block interface) |
| `blocks.Static_ANN_Block.__init__` 24, `.forward` 40 | 03:665; `out_gate`: Ext evaluation (`report._without_learned`) |
| `blocks.Linear_Output_Block.__init__` 50, `.forward` 63 | eq:aug_norm_maps (y_hat^n = C^n x_tilde^n) |
| `blocks.Discrete_Nonlinear_Function_Block.forward` 73 | infrastructure (Jan's block interface) |
| `blocks.GantryStateBlock.__init__` 84 | eq:base_ss set-up (nominal matrices, sigma_x, mu_x) |
| `blocks.GantryStateBlock._rk4` 128 | 03:450 (f_base = RK4) |
| `blocks.GantryStateBlock._deriv_with` 139 | eq:base_ss, eq:aug_norm_maps, eq:delta, eq:app_lfr_blocks |
| `blocks.GantryStateBlock._deriv_with_tangent` 193 | eq:obc_sens / eq:obc_corr (directional derivative of F) |
| `blocks.GantryStateBlock._rk4_with_tangent` 258 | eq:obc_corr (f^n_base(beta_lin) and varphi a together) |
| `blocks._TrainableGantryStateBlock.__init__` 282 | 02:706 (L_b fixed); `Lambda_combo` scale: none, MM14 |
| `blocks._TrainableGantryStateBlock._build_KC` 293 | eq:damping_matrix, eq:stiffness_matrix |
| `blocks._TrainableGantryStateBlock._build_M1M2` 313 | eq:Msplit, 99:63 |
| `blocks._TrainableGantryStateBlock._mats_from_raw` 328 | eq:matrices via eq:Msplit (the matrices of one objective) |
| `blocks._TrainableGantryStateBlock.nonlinear_function` 340, `.pass_mats` 346 | eq:aug_transition (f_base of one objective), infrastructure (matrices once per objective) |
| `blocks.IdentifiableGantryStateBlock.__init__` 365 | eq:aug_init (beta = 0), 99:276 to 287 (eta_0, s); `Lambda_combo`, `coordinate_version`: infrastructure, MM14 |
| `blocks.IdentifiableGantryStateBlock.combos_of` 388 | eq:phi |
| `blocks.IdentifiableGantryStateBlock.gauge_section` 399 | 99:297 (any raw vector with the combinations) |
| `blocks.IdentifiableGantryStateBlock._m_diff_bound` 415 | eq:mdelta_map (b = gamma (2/L_b) sqrt(m_Sigma J_eff)), eq:lfr_admissibility |
| `blocks.IdentifiableGantryStateBlock.combinations_from_free` 420 | 02:854, eq:mdelta_map |
| `blocks.IdentifiableGantryStateBlock.free_from_combinations` 428 | inverse of eq:mdelta_map and 02:854; Ext evaluation (`report._true_free`) |
| `blocks.IdentifiableGantryStateBlock.recover_combinations` 445, `.estimation_parameters` 463 | eq:phi (reported estimate); Ext evaluation |
| `blocks.IdentifiableGantryStateBlock.transition_from_free` 449 | eq:obc_sens (f^n_base as a function of beta) |
| `blocks.IdentifiableGantryStateBlock.mats_and_tangent_from_free` 454 | eq:obc_corr (matrix tangent along a) |
| `blocks.IdentifiableGantryStateBlock._mats_at` 458 | eq:phi to eq:matrices (beta to matrices) |
| `cl.ControllerBank.__init__` 18, `.step` 37 | eq:aug_cl_step |
| `cl.ControllerBank.n_controllers` 34 | infrastructure |
| `cl.closed_loop_step` 46 (`step` 49) | eq:aug_cl_step, 03:943 |
| `cl.closed_loop_rollout` 59 | eq:aug_residual_loop (x^c_tau = 0), eq:aug_loss constraints |
| `cl.window_counts` 77 | infrastructure (window count of eq:aug_loss) |
| `cl.ClosedLoopSimulator.__init__` 87, `.__getstate__` 100, `.eager` 107, `.augment_training_data` 132, `.verify_by_content` 151 | infrastructure; `augment_training_data` also attaches W_r (03:1149) |
| `cl.ClosedLoopSimulator.__call__` 115 | eq:aug_loss rollout (compiled or eager) |
| `cl.ClosedLoopSimulator.validation_error` 166 | 03:1291, eq:setup_rms |
| `cl.closed_loop_free_run_rms_batch` 178 | 03:1296 (closed-loop free run, encoder once), eq:setup_rms |
| `ic.Interconnect.__init__` 26, `._apply` 46, `.init_forward` 109, `._signal_cone` 160, `.init_connection_matrices` 187, `.add_block` 228, `.connect_signals` 237, `.connect_block_signals` 249, `.determine_signal_ix` 257, `.convert_signal_name_to_ix` 263, `.parse_connection_function_method` 269, `Signal_Connection.__init__` 275 | infrastructure (Jan's interconnect) |
| `ic.Interconnect.forward` 54, `._propagate` 85 | eq:aug_transition, eq:obc_corr (subtract f_obc) |
| `ic.Interconnect.output_only` 174 | eq:aug_cl_step (y_hat before the update) |
| `ic._replace_with_retry` 284, `ic.DtypeDataLoader.__init__` 300 (`_conv` 301) | infrastructure |
| `ic.SSE_Interconnect.__init__` 322, `.init_model` 330, `.simulate` 338, `.make_training_data` 343, `.cal_validation_error` 349 | infrastructure (deepSI hooks); `init_model`: 03:1266 (Adam) |
| `ic.SSE_Interconnect.fit` 357 (`validation` 361, `closure` 418) | 03:1266, 03:1291 to 1300, eq:ps2_alt |
| `ic.AugmentedModel.__init__` 480, `.checkpoint_save_system` 535, `.checkpoint_load_system` 543, `.__getstate__` 552, `.optimizer` 558 / 562 | infrastructure |
| `ic.AugmentedModel._epoch_index` 489 | 04:345 (epoch of the refresh) |
| `ic.AugmentedModel._sync_obc_coefficient` 494 | eq:obc_coef (theta_aux for validation), MM2 (rebuild at the selected checkpoint) |
| `ic.AugmentedModel.objective_matrices` 507 | infrastructure (matrices once per objective) |
| `ic.AugmentedModel.loss` 517 | eq:aug_loss, 04:341 to 346 |
| `obc.OBCRankLossAbort` 11 | 04:263 (rank failure) |
| `obc.OBCBasis.__init__` 18 | eq:obc_coef (factorisation), 04:263 |
| `obc.OBCBasis.coefficient` 38 | eq:obc_coef (theta_aux) |
| `obc.OBCBasis.to` 45, `.device` 52 | infrastructure |
| `obc.build_stacked_sensitivity` 56 | eq:obc_sens stacked; with `append_value` eq:obc_offset column (MM1) |
| `obc.OBCProjection.__init__` 85 | infrastructure |
| `obc.OBCProjection.refresh` 94 | 04:345, 04:263 |
| `obc.OBCProjection.maybe_refresh` 119 | 04:345 |
| `obc.OBCProjection.coefficient` 124 | eq:obc_coef |
| `enc.AugmentedEncoder.__init__` 17 | eq:aug_init (encoder), 03:1506 |
| `enc.AugmentedEncoder.forward` 61 | eq:aug_enc_lin |
| `ps2.PS2Spec` 14 | tab:hyper (PS rows) |
| `ps2._plateaued` 37 | 03:1811, tab:hyper (stop criteria) |
| `ps2.calibration_split` 42 | 03:1672, tab:hyper |
| `ps2.PS2Opening.__init__` 56, `.done` 79, `.summary_json` 115 | infrastructure |
| `ps2.PS2Opening.after_step` 82 | eq:ps2_alt, 03:1811, eq:ps2_handover |
| `ps2.PS2Opening._start` 122 | eq:ps2_s1 (decoder p, theta^a_enc, own Adam) |
| `ps2.PS2Opening._sample` 141 | 03:1671 (S1 batch), eq:ps2_s2 (pairs) |
| `ps2.PS2Opening._residual` 166 | eq:ps2_s1 (e_tau, sigma_e) |
| `ps2.PS2Opening._added_states` 173 | eq:ps2_s1 (psi_a) |
| `ps2.PS2Opening._s1_step` 177 | eq:ps2_s1 |
| `ps2.PS2Opening._cal_unexplained` 192 | 03:1811 (nu) |
| `ps2.PS2Opening._fit_added_state_transition` 199 (`snap` 207, `state_map_loss` 213) | eq:ps2_s2, 03:1913 |
| `lbfgs.PolishSpec` 15 | tab:hyper (L-BFGS rows) |
| `lbfgs._restore` 26 | 03:1305 (rollback when not kept) |
| `lbfgs.FixedWindowBatch.__init__` 39 (`_to_tensor` 63) | 03:1304 (fixed set of windows) |
| `lbfgs._make_closure` 80 (`closure` 82) | 03:1304 (minimise V) |
| `lbfgs._lbfgs_counts` 92, `lbfgs._is_oom` 98, `lbfgs._probe_closure` 104 | infrastructure (counters, OOM, determinism check) |
| `lbfgs._acceptance` 120 | 03:1305 (kept only if validation error drops) |
| `lbfgs.lbfgs_polish` 131 (`_validate` 150, `_record_counts` 160, `_abort` 164, `_new_opt` 186) | 03:1304 to 1309 |

No callable is left without an entry or an infrastructure role.

## 3. Mismatches
Side: **C** = code would change, **T** = thesis would change, **none** = equivalent, comments state it. Nothing is
changed here.

| ID | Where | Mismatch | Side |
|-|-|-|-|
| MM1 | eq:obc_offset, eq:obc_coef vs `obcg.make_basis_builder.build` 153 | Extra column is f^n_base(beta_lin), not gamma = f^n_base(beta_lin) - varphi beta_lin. Same span, so F_tilde and f^n_obc are identical at every (x, u); theta_aux coordinates differ: code (a, c) = thesis (a + beta_lin c, c) shifted. An equation test of theta_aux must use the code's basis. | none |
| MM2 | 04:345 vs `ic.SSE_Interconnect.fit` 465 (`_sync_obc_coefficient(refresh_basis=True)`) and the polish | After Adam the basis is rebuilt at the selected checkpoint's beta and held fixed during the polish; the thesis sentence saying so was struck (04:360). | T |
| MM3 | eq:ps2_s2 vs `ps2.state_map_loss` 213 | Denominator `T.var(0).mean()` divides by B - 1: code cost = thesis cost x (B - 1)/B. Same minimiser and same kept parameters (one constant per set); logged S2 costs scaled. | none |
| MM4 | eq:mdelta_map, 99:245 vs `blocks` | Margin gamma is `M_DIFF_RHO`; `gamma_` is J_eff; `alpha`, `beta` are m_Sigma + m_h and (L_b/2) m_Delta. Code rho is not the thesis rho (overlap). | none (comments name it) |
| MM5 | eq:aug_norm vs `data.compute_normalization` 187 to 190 | sigma_x, sigma_u, sigma_y carry + 1e-8. | none |
| MM6 | tab:hyper vs `cfg.adam_eps` 105 | Adam epsilon 1e-16 (default 1e-8) is not in the table. | T |
| MM7 | 05:323 vs `cfg.AA_TRIM_S` 25, `data._record_at_fs_new` 113 | 5 ms trimmed at both record ends; the sentence was struck (99:573). | T |
| MM8 | 05:325 vs `data._decimate_y` 99 | Noise-free data are point sampled, not filtered; the thesis revnote (99:563) already rules: add a sentence only if a noise-free training run is reported. | T (conditional) |
| MM9 | (withdrawn while writing: W_r and the validation sigma_e both use k0 = n + 1 .. N - 1, as eq:setup_rms) | | |
| MM10 | eq:aug_enc_lin vs `enc.AugmentedEncoder.forward` 75 to 77 | psi_tilde reads (u history, y history), chi is (y, u): a fixed permutation of a learned network's input. | none |
| MM11 | eq:aug_enc_sim vs `model.prefit_physical_encoder` 151 | Code simulates the whole start model (with f^n_aug = 0 on the physical rows and, on the OBC arm, f^n_obc = 0), not f^n_base alone; the physical states are the same. | none |
| MM12 | eq:aug_residual_loop vs `cl.closed_loop_rollout` 66 | The x^c_tau = 0 reset carries `# HEURISTIC`; the thesis derives it (x^c = 0 makes the model's controller state the system's, 03:862). The marker stays (step-4 rule); relabelling to `# THEORY` is a code change for Dirk. | C (marker) |
| MM13 | 02:645, eq:base_ss vs `blocks._deriv_with` 175 to 188 and `RunConfig.baseline_form` | Default path evaluates xdot through the LFR output equation on the closed-form a (equal to eq:base_ss up to rounding); `closed_form` / `--closed-form` is an option no thesis row uses. | none (M8 toggle, D-284 amendment 3) |
| MM14 | none vs `blocks._TrainableGantryStateBlock.__init__` 289, `IdentifiableGantryStateBlock.__init__` 373, `cfg.param_rmse_baseline` 108 | `Lambda_combo` (and `param_rmse_baseline`) is read by no loss (`param_prior` False); kept only as a checkpoint key. | none (interface, design.md section 5) |
| MM15 | 02:517 (struck frozen-LTI comparison) vs `blocks._deriv_with` 158 to 161 (`Y_op`, `N_op`, `d_op`) | The frozen-scheduling branch has no active thesis entry; `N_op`, `d_op` are set outside `Clean/` by the evaluation (an interface, implementation report "evaluation interface"). | none (Ext) |

Counts: code side 1 (MM12, a marker), thesis side 3 plus 1 conditional (MM2, MM6, MM7; MM8), equivalent or ruled 9
(MM1, MM3, MM4, MM5, MM10, MM11, MM13, MM14, MM15).

## 4. Item 1 proof (harness narrowed to the grey box)
`Thesis-writeup/Code/Verification/clean-code/hcommon.py`: `grey_box_files()` lists the 24 grey-box `.py` files;
`clean_tree_hashes()` (H2) hashes only those. `check_imports.py` counts only grey-box `.py` files. Results
2026-10-28: `check_imports.py` exit 0 ("imported 24 modules; 24 grey-box .py files; outside Clean/: none"); fast
gate `run_l12_all.sh clean 221-lowpass 227-lowpass` then `compare.py l12 221-lowpass 227-lowpass`: L1 PASS, L2 PASS
on both, `COMPARE_L12 PASS`.
