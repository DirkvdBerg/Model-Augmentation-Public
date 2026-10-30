# Independent Codex thesis-to-code map

2026-10-28. Thesis revision: `6b77c9a`, from `git log -1 --format=%h -- Thesis-writeup/Writing`. `git status --porcelain -- Thesis-writeup/Writing` returned no entries. The overview and item 2 of the Claude handoff were read first. Claude's independent map was not read.

## Scope

Active displayed equations and method procedures in `Thesis-writeup/Writing/sections/02_system_baseline.tex`, `03_augmentation.tex`, `04_obc.tex`, `05_setup.tex` and the method subsections of `99_appendix.tex`. Comments, deleted text (`del`, `delmv`), old arguments of `chg`, and draft-only outlines are excluded. Additions, moved text and new arguments of `chg` are included. Unlabelled displays use the opening delimiter's source line.

Appendix included: `app:params` (nominal table context), `app:lfr-details`, `app:lfr-wellposed`, `app:combos`, `app:setup_system`, `app:setup_records` (excitation procedure), `app:setup_checks`, and grey-box rows of `app:hyper`. Excluded: `app:splits` (results table), `app:setup_bb` (black-box model and search) and black-box hyperparameters. `app:setup` is the enclosing section. Included data-generation formulas are explicitly external. No appendix subsection describing the grey-box method is silently omitted.

Code: the 24 Python files under `Clean/gantry_dynamic/`, `Clean/model_augmentation/`, and `Clean/gantry_interconnect_dynamic.py`. Named functions, nested closures, methods, properties, explicit constructors and generated dataclass constructors are inventoried below. No training, harness run, thesis compilation, mainline edit or git write. Only this map and `Thesis-writeup/Code/Verification/clean-code/equation_tests.py` are written.

## Notation and test conventions

In the forward table, callable names resolve to the fully qualified names and exact `Clean/...:line` locations in the backward table. Thus each forward row has both its qualified callable and source anchor without repeating long paths. A test cell names `test_<row>` in the companion script; shared tests identify the exact common formula. Per-test results and largest differences follow below.

Thesis baseline `u` is logical force; callable `u` is normalised actuator force, denormalised then multiplied by P. Batch rows write Ax as `x @ A.T`. `free_params` is beta; `m_total` is m_Sigma (excludes mh); `m_diff` is m_Delta. Polynomial `alpha`, `beta`, `gamma_` are m_Sigma+mh, Lb*m_Delta/2 and J_eff, not the training beta or affine offset gamma. `nx_aug` is n_barx. Arguments na/nb of AugmentedEncoder are one less than the history sample count. The MLP reads (u,y) while the encoder's linear map reads thesis chi=(y,u), an explicit input permutation, not a sign or index error.

Tests: fixed seed 284, small synthetic batches, float64 CPU, rtol=1e-12 and atol=1e-12 for theoretical zeros. `B_mass` uses rtol=atol=2e-7 because the controller starts with decimal float64 constants while the physics block promotes stored float32 constants. That provenance exception is explicit; it is not applied to OBC or integration tests. Tests retain actual formula failures and return nonzero. Numerical comparison is a formula check, not a proof of training convergence.

## Forward map

### Baseline and physical parameters

| Row | Thesis entry and mathematical part | Callable(s) | Test or boundary |
|---|---|---|---|
| B_system | eq:system, true vector field and sampled noisy positions | external | **External/input boundary**. `Thesis-writeup/Code/Data/gantrySystemExtendedTanh.m`, simulated truth, not fitted physics |
| A_controller_state | eq:system_loop, state row | ControllerBank.__init__, ControllerBank.step | **PASS**, `test_A_controller_state`. A_controller_state |
| A_controller_force | eq:system_loop, force row; bank returns feedback part, recorded input includes feedforward | ControllerBank.step, closed_loop_step | **PASS**, `test_A_controller_force`. A_controller_force, A_cl_step |
| B_transform | eq:Ptransform, P matrix, q=P^-T q_act and u=P u_act | Linear_Output_Block.forward, GantryStateBlock._deriv_with | **PASS**, `test_B_transform`. B_transform |
| B_vector_field | eq:eom, M(Y)qdd=P u_act-C qdot-Kq | GantryStateBlock._deriv_with | **PASS**, `test_B_vector_field`. B_vector_field |
| B_mass | eq:mass_matrix, expanded M(Y) | mass_matrix_at, _TrainableGantryStateBlock._mats_from_raw | **PASS**, `test_B_mass`. B_mass |
| B_damping | eq:damping_matrix, C entries | _TrainableGantryStateBlock._build_KC | **PASS**, `test_B_damping`. B_damping |
| B_stiffness | eq:stiffness_matrix, K_22=kb1+kb2 | _TrainableGantryStateBlock._build_KC | **PASS**, `test_B_stiffness`. B_stiffness |
| B_mass_split | eq:Msplit, M0+Y M1+Y^2 M2 | _TrainableGantryStateBlock._build_M1M2, _TrainableGantryStateBlock._mats_from_raw | **PASS**, `test_B_mass_split`. B_mass_split |
| B_state | eq:base_ss, F=(qdot,M^-1 netforce) | GantryStateBlock._deriv_with | **PASS**, `test_B_state`. B_vector_field |
| B_identifiability | 02_system_baseline.tex:745, (M-Mprime)qdd+(C-Cprime)qdot+(K-Kprime)q=0 | external | **External/input boundary**. derivation only, local identifiability argument in that subsection |
| B_combinations | eq:phi, ordered ten-vector, raw combinations | IdentifiableGantryStateBlock.combos_of, IdentifiableGantryStateBlock.gauge_section | **PASS**, `test_B_combinations`. B_combinations |
| B_admissibility | eq:lfr_admissibility, m_Sigma J_eff>(Lb m_Delta/2)^2 | IdentifiableGantryStateBlock._m_diff_bound, IdentifiableGantryStateBlock.combinations_from_free | **PASS**, `test_B_admissibility`. B_admissibility |
| B_mdelta | eq:mdelta_map, bounded signed coordinate | IdentifiableGantryStateBlock.combinations_from_free, IdentifiableGantryStateBlock.free_from_combinations | **PASS**, `test_B_mdelta`. B_mdelta |

### Augmentation, controller, encoder and initialisation

| Row | Thesis entry and mathematical part | Callable(s) | Test or boundary |
|---|---|---|---|
| A_physical | eq:aug_transition, physical next-state row | Interconnect.forward, build_model | **PASS**, `test_A_physical`. A_physical |
| A_added | eq:aug_transition, added next-state row, no identity bypass | Interconnect.forward, build_model | **PASS**, `test_A_added`. A_added |
| A_output | eq:aug_transition, yhat=P^T q | Linear_Output_Block.forward, Interconnect.output_only | **PASS**, `test_A_output`. B_transform |
| A_normalization | eq:aug_norm, all three affine maps and their training statistics | SystemDataNormalizer.transform, compute_normalization | **FAIL**, `test_A_normalization`. A_normalization |
| A_norm_transition | eq:aug_norm_maps, physical/added transition rows and f_base affine conjugacy | GantryStateBlock._rk4, Interconnect.forward | **PASS**, `test_A_norm_transition`. A_norm_transition, A_physical, A_added |
| A_norm_output | eq:aug_norm_maps, yhat^n=Cn x^n and Cn=Ty P^T[I 0]Tx^-1 | compute_normalization, Linear_Output_Block.forward | **FAIL**, `test_A_norm_output`. A_norm_output |
| A_norm_inverse | eq:aug_norm_maps, final row recovering physical x,u | GantryStateBlock._deriv_with | **PASS**, `test_A_norm_inverse`. B_vector_field |
| A_network | 03_augmentation.tex:665, stacked (f_aug^n,g_aug^n)=phi_aug | Static_ANN_Block.forward, AugmentationNetwork.forward | **PASS**, `test_A_network`. A_network |
| A_model_loop_state | eq:aug_two_loops, model-controller state row | ControllerBank.step, closed_loop_step | **PASS**, `test_A_model_loop_state`. A_controller_state; implemented through equivalent residual form |
| A_model_loop_force | eq:aug_two_loops, model-controller force row | closed_loop_step | **PASS**, `test_A_model_loop_force`. A_cl_step; reference and feedforward cancelled algebraically |
| A_residual_state | eq:aug_residual_loop, xc+=A xc+B(y-yhat), xc_tau=0 | ControllerBank.step, closed_loop_rollout | **PASS**, `test_A_residual_state`. A_controller_state; zero reset inspected in rollout |
| A_residual_force | eq:aug_residual_loop, uhat=u_act+C xc+D(y-yhat) | closed_loop_step | **PASS**, `test_A_residual_force`. A_cl_step |
| A_residual_split | eq:aug_residual_split, force identity exposing controller replay error | external | **External/input boundary**. derivation only, obtained by subtracting the controller equations; no separate runtime calculation |
| A_cl_output | eq:aug_cl_step, yhat and physical epsilon row | closed_loop_step, ControllerBank.__init__ | **PASS**, `test_A_cl_output`. A_cl_step |
| A_cl_input | eq:aug_cl_step, normalised corrected input row | closed_loop_step, ControllerBank.step | **PASS**, `test_A_cl_input`. A_cl_step |
| A_cl_step | eq:aug_cl_step, controller state row and step order | closed_loop_step, closed_loop_rollout | **PASS**, `test_A_cl_step`. A_cl_step |
| A_loss | eq:aug_loss, objective sum/(windows nf ny), Wr=sigma_y/sigma_e | AugmentedModel.loss, servo_sigma, build_closed_loop | **PASS**, `test_A_loss`. A_loss |
| A_loss_physical | eq:aug_loss, physical equality constraint | Interconnect.forward | **PASS**, `test_A_loss_physical`. A_physical |
| A_loss_added | eq:aug_loss, added equality constraint | Interconnect.forward | **PASS**, `test_A_loss_added`. A_added |
| A_loss_initial | eq:aug_loss, encoder initial condition and loop quantities | AugmentedEncoder.forward, closed_loop_step | **PASS**, `test_A_loss_initial`. A_encoder_linear, A_cl_step |
| A_encoder_history | eq:aug_encoder, both initial state and history chi ending tau-1 | WindowedSystemData.to_hist_future_data, AugmentedEncoder.forward | **PASS**, `test_A_encoder_history`. A_encoder_history, A_encoder_linear |
| A_encoder_linear | eq:aug_enc_lin, [Wb;Wa]chi+psi_tilde | AugmentedEncoder.forward | **PASS**, `test_A_encoder_linear`. A_encoder_linear |
| A_encoder_sim | eq:aug_enc_sim, physical baseline rollout and output rows | prefit_physical_encoder, closed_loop_rollout | **PASS**, `test_A_encoder_sim`. A_encoder_sim, synthetic prefit target/history assembly and least squares; A_rk4 separately checks nonlinear physics |
| A_encoder_fit | eq:aug_enc_fit, affine least-squares start | least_squares_start | **PASS**, `test_A_encoder_fit`. A_encoder_fit |
| A_init_beta | eq:aug_init, beta=0 | IdentifiableGantryStateBlock.__init__ | **PASS**, `test_A_init_beta`. A_init_beta |
| A_init_network | eq:aug_init, physical final weights zero, added weights uniform, final bias zero | AugmentationNetwork.__init__, build_mlp | **PASS**, `test_A_init_network`. A_init_network |
| A_init_encoder | eq:aug_init, fitted Wb,b; Xavier Wa; nonlinear final weights zero; added bias zero | AugmentedEncoder.__init__, least_squares_start | **PASS**, `test_A_init_encoder`. A_init_encoder, A_encoder_fit |
| A_rk4 | 03_augmentation.tex:449, RK4 with held actuator input, endogenous Y recomputed at each stage | GantryStateBlock._rk4 | **PASS**, `test_A_rk4`. A_rk4 |
| A_coordinates | 03_augmentation.tex identification prose, nine log coordinates and bounded tenth | IdentifiableGantryStateBlock.combinations_from_free | **PASS**, `test_A_coordinates`. B_mdelta |
| A_optimizer | 03_augmentation.tex:1266, Adam minibatches, record validation, restore best, fixed-window L-BFGS and acceptance | SSE_Interconnect.fit, train, run_lbfgs_polish, lbfgs_polish | not testable here: optimizer/selection trajectory requires training, explicitly forbidden; call order inspected |

### Predictive-state opening

| Row | Thesis entry and mathematical part | Callable(s) | Test or boundary |
|---|---|---|---|
| P_s1_loss | eq:ps2_s1, decoder criterion | PS2Opening._s1_step, PS2Opening._cal_unexplained | **PASS**, `test_P_s1_loss`. P_s1_loss, real S1 gradients captured by a no-op optimizer, no update |
| P_residual | eq:ps2_s1, residual target divided by batch RMS | PS2Opening._residual | **PASS**, `test_P_residual`. P_residual |
| P_alternation | eq:ps2_alt, V Adam then separate S1 Adam; shared encoder coordinates replaced by S1 result | SSE_Interconnect.fit, PS2Opening.after_step, PS2Opening._s1_step | not testable here: needs optimizer updates; inspected optimizer.step before after_step |
| P_pairs | eq:ps2_s2, pair set and encoder at k_i and k_i+1; recorded input | PS2Opening._sample, PS2Opening._fit_added_state_transition.snap | **PASS**, `test_P_pairs`. P_pairs |
| P_s2_cost | eq:ps2_s2, regression quotient | PS2Opening._fit_added_state_transition.state_map_loss | **FAIL**, `test_P_s2_cost`. P_s2_cost; zero update budget evaluates the actual nested callable |
| P_handover | eq:ps2_handover, retain beta/encoder, updated network, discard decoder and continue V | PS2Opening.after_step, PS2Opening._fit_added_state_transition, SSE_Interconnect.fit | not testable here: follows trained S1/S2, no optimizer run permitted; source order inspected |
| P_stops | 03_augmentation.tex:1811 and app:hyper, held-out classes, plateau/cap, 10 percent screen, S2 plateau | calibration_split, _plateaued, PS2Opening.after_step | **PASS**, `test_P_stops`. P_stops tests split and plateau; full state-machine sequence not testable here without training |

### Orthogonal construction

| Row | Thesis entry and mathematical part | Callable(s) | Test or boundary |
|---|---|---|---|
| O_gyorok | eq:obc_gyorok, four rows: linear model, projected field, LS coefficient, orthogonality | external | **External/input boundary**. background derivation; gantry adaptation below, not another runtime path |
| O_sensitivity | eq:obc_sens, Jacobian of complete normalised discrete transition w.r.t. beta | build_stacked_sensitivity, IdentifiableGantryStateBlock.transition_from_free | **PASS**, `test_O_sensitivity`. O_sensitivity |
| O_offset | eq:obc_offset, affine approximation and gamma=fbar-Phi beta_lin | make_basis_builder, GantryOBCCorrection.forward | **PASS**, `test_O_offset`. O_offset verifies basis-coordinate transformation |
| O_tuple_position | eq:obc_tuples, q=P^-T y and state concatenation | build_reference_set | **PASS**, `test_O_tuple_position`. O_tuple_position, actual build_reference_set with synthetic in-memory records |
| O_difference | eq:obc_tuples, five-point fourth-order qdot | fourth_order_central_difference | **PASS**, `test_O_difference`. O_difference |
| O_basis | eq:obc_coef, first row, stacked Phi_Gamma | make_basis_builder, build_stacked_sensitivity | **PASS**, `test_O_basis`. O_sensitivity, O_offset |
| O_field | eq:obc_coef, second row, physical network field at xbar=0, sample-major stack | make_reference_field | **PASS**, `test_O_field`. O_field |
| O_coefficient | eq:obc_coef, third row, full-rank LS coefficient | OBCBasis.__init__, OBCBasis.coefficient | **PASS**, `test_O_coefficient`. O_coefficient |
| O_projected | eq:obc_coef, fourth row, F-J theta | GantryOBCCorrection.forward, Interconnect.forward | **PASS**, `test_O_projected`. O_coefficient, O_physical_step |
| O_orthogonality | eq:obc_coef, fifth row, J.T(F-J theta)=0 | OBCBasis.coefficient | **PASS**, `test_O_orthogonality`. O_coefficient |
| O_correction | eq:obc_corr, first row, Jacobian-vector product plus affine-column correction | GantryOBCCorrection.forward, GantryStateBlock._rk4_with_tangent | **PASS**, `test_O_correction`. O_correction |
| O_physical_step | eq:obc_corr, second row, base+network-correction | Interconnect.forward | **PASS**, `test_O_physical_step`. O_physical_step |
| O_added_step | eq:obc_corr, third row, g_aug unchanged | Interconnect.forward | **PASS**, `test_O_added_step`. O_physical_step checks unchanged added row too |
| O_refresh | 04_obc.tex:343, coefficient each objective, basis/expansion point each epoch | AugmentedModel.loss, OBCProjection.maybe_refresh, OBCProjection.refresh | not testable here: evolving epoch/optimizer procedure; predicate and call order inspected |
| O_residual | eq:obc_eps, reconstructed successor minus true-parameter baseline | external | **External/input boundary**. `Thesis-writeup/Code/Evaluation/eval_overlap.py:evaluate`, lines 113-130 |
| O_recovery_projected | eq:obc_recovery, first row, predicted first-order bias | external | **External/input boundary**. `Thesis-writeup/Code/Evaluation/eval_overlap.py:evaluate`, line 132 onward; not training optimum |
| O_recovery_unprojected | eq:obc_recovery, second row, subtract raw field | external | **External/input boundary**. derivation only of first-order surrogate; not an alternative estimator in Clean |
| O_overlap | eq:obc_overlap, rho and Pi rows | external | **External/input boundary**. `Thesis-writeup/Code/Evaluation/eval_overlap.py:overlap:43`, least squares without forming dense Pi |

### Setup and appendix realisation

| Row | Thesis entry and mathematical part | Callable(s) | Test or boundary |
|---|---|---|---|
| S_friction | eq:setup_truth (same display also labels eq:setup_fric), truth EOM and smooth Coulomb force rows | external | **External/input boundary**. `Thesis-writeup/Code/Data/gantrySystemExtendedTanh.m` |
| S_absorber | eq:setup_abs, extended truth EOM | external | **External/input boundary**. same data-generating system, not fitted baseline |
| S_metric | eq:setup_rms, per-axis RMS and NRMS_e rows, samples n+1 through N-1 | closed_loop_free_run_rms_batch, ClosedLoopSimulator.validation_error, servo_sigma | **PASS**, `test_S_metric`. S_metric |
| L_matrices | 99_appendix.tex:63, expanded M1 and M2 | _TrainableGantryStateBlock._build_M1M2 | **PASS**, `test_L_matrices`. B_mass_split |
| L_factorization | eq:lfr_mass_factorization, polynomial mass equation | GantryStateBlock._deriv_with | **PASS**, `test_L_factorization`. B_vector_field |
| L_chain | eq:lfr_chain, a1=Y qdd and a2=Y a1 | GantryStateBlock._deriv_with | **PASS**, `test_L_chain`. L_chain |
| L_constant | eq:lfr_constant, M0 qdd+M1 a1+M2 a2=netforce | GantryStateBlock._deriv_with | **PASS**, `test_L_constant`. B_vector_field |
| L_delta | eq:delta, w=Y z, z=(qdd,Y qdd) | GantryStateBlock._deriv_with | **PASS**, `test_L_delta`. L_chain |
| L_G_state | eq:lfr_G, state row of G | build_G_matrix_entries, GantryStateBlock._deriv_with | **PASS**, `test_L_G_state`. L_blocks, B_vector_field |
| L_G_latent | eq:lfr_G, z row, built from rational acceleration | GantryStateBlock._deriv_with | **PASS**, `test_L_G_latent`. L_chain |
| L_G_output | eq:lfr_G, logical q row followed by P^T for measured output | Linear_Output_Block.forward | **PASS**, `test_L_G_output`. B_transform |
| L_blocks | eq:app_lfr_blocks, first row Ax,Bw,Bu | build_G_matrix_entries | **PASS**, `test_L_blocks`. L_blocks |
| L_blocks_latent | eq:app_lfr_blocks, second row Cz,Dzw,Dzu | external | **External/input boundary**. derivation only; no stored latent matrices, equivalent signals constructed in GantryStateBlock._deriv_with |
| L_blocks_output | eq:app_lfr_blocks, third row Cy=[I 0], Dyw=Dyu=0 | external | **External/input boundary**. logical-output derivation; actual measured-output block composes P^T, B_transform |
| L_schur | 99_appendix.tex:199, I-Dzw Delta block matrix | external | **External/input boundary**. derivation only, well-posedness proof |
| L_wellposed | 99_appendix.tex:209, determinant equalities | external | **External/input boundary**. derivation only, well-posedness proof |
| L_reduced_mass | 99_appendix.tex:245, M in reduced coordinates | _TrainableGantryStateBlock._mats_from_raw | **PASS**, `test_L_reduced_mass`. B_mass_split |
| L_sylvester | 99_appendix.tex:254, three positive minors in reversed coordinate order | external | **External/input boundary**. derivation only, parameter map enforces admissibility |
| L_det | 99_appendix.tex:258, determinant expansion and completed square | build_poly_constants, GantryStateBlock._deriv_with | **PASS**, `test_L_det`. L_det |
| L_raw_bound | 99_appendix.tex:269, positivity of raw-parameter admissibility expression | external | **External/input boundary**. derivation only |
| L_eta | 99_appendix.tex:279, r0,eta0,s initial signed-coordinate calibration | IdentifiableGantryStateBlock.__init__ | **PASS**, `test_L_eta`. L_eta |
| L_entries | 99_appendix.tex:297, all entry-to-combination identities, including repeated mh entries | IdentifiableGantryStateBlock.combos_of, _TrainableGantryStateBlock._build_KC, _TrainableGantryStateBlock._build_M1M2 | **PASS**, `test_L_entries`. L_entries |
| L_inverse_entries | 99_appendix.tex:311, inverse map from matrix entries to combinations | external | **External/input boundary**. derivation only; runtime maps parameters to matrices, not inferred matrices to parameters |
| S_extended_matrices | 99_appendix.tex:371, Me,Ma,Ce,Ke matrices | external | **External/input boundary**. `Thesis-writeup/Code/Data/gantrySystemExtendedTanh.m` |
| S_controller | eq:setup_K1, continuous transfer function and gain at bandwidth; bilinear discretisation in surrounding method | normalized_controller_coefficients, stage_frequency_response, build_cfb_at, controller_ss | **PASS**, `test_S_controller`. S_controller, continuous shape, frozen frequency response and discrete state-space response; see explicit conditioning tolerance below |
| S_records | app:setup_records and Section V records, manifest split and K1 selection | record_files, load_datasets | **External/input boundary**. external inputs: MANIFEST.csv under configured dataset, no record read in tests |
| S_processing | app:setup_checks, force block means, error-only AA, decimation, trim, training-only normalisation | _resample_u, _decimate_y, _record_at_fs_new, compute_normalization | not testable here: processing procedure has no displayed coefficient formula; source inspected; affine/statistics formulas tested in A_normalization |
| S_defaults | app:hyper, grey-box and PS settings | RunConfig.hp, PS2Spec, train | not testable here: table/config correspondence, not an executable mathematical formula; defaults inspected |

Default check: model rate 4 kHz, one RK4 step, nf=400, stride=10, encoder history=30, tanh network and correction 2x16, gamma=.99, detuning 10 percent, Adam lr=1e-5, batch=512, 200 epochs, validation every 1300 updates. L-BFGS outer/inner/history=100/20/50, windows=16384, tolerances=2.6e-5/2.6e-7. PS lr=.001, S1 batch=64 and 100..350 updates, 25-update checks, 4-check 2 percent plateau; S2 batch=512, pairs=8192/2048, 500..8000 updates, 100-update checks and 5-check 1 percent plateau. Class holds out its last record when count>=3. These agree with app:hyper. Stated 33,400 Adam updates depends on the full dataset's 167 batches/epoch, not a universal constant in RunConfig. Smoke budgets are intentionally different CLI/test settings. Decoder width is one hidden layer of 16, distinct from the two-layer augmentation network.

## Mismatches

1. **M1, code side, P_s2_cost:** thesis quotient uses the sum of centred target squares. Code divides MSE by `T.var(0).mean()` with sample correction=1. Its value is `(B-1)/B` times the displayed objective. Same minimizer at fixed B, different reported value and Adam epsilon scaling. Batch and calibration B differ. To implement the display, code would use population variance. To preserve thesis runs, thesis would state the sample-variance convention and factor. Neither changed here.

2. **M2, code side, A_normalization/A_norm_output:** `compute_normalization` inverts stored P in float32, then promotes the inverse. The float64 formula with the same stored P gives different position/velocity statistics. `C mu_x=mu_y` consequently holds only approximately, and the zero-intercept normalised output is not the exact affine conjugacy at float64 precision. Code also adds 1e-8 to population scales, an explicit numerical guard not written in the ideal formula. To make the formula exact, code would invert after promotion and account for any residual output offset. To preserve recorded runs, thesis/implementation notes would state the precision convention. The float32 nominal physics constants versus decimal-double controller constants (B_mass tolerance) are the same precision-provenance class, not another mass sign or physical model mismatch. Neither changed here.

3. **M3, thesis side, O_offset/O_basis/O_correction:** thesis theta_aux uses `[Phi,Gamma]`; code stores coefficients in `[Phi,fbar]`. With code coefficients `(a,c)`, thesis coefficients are `(a+beta_lin*c,c)`. The spans and corrected fields are identical; O_offset tests this translation. A comment treating the coefficient vectors as identical would be wrong. The thesis implementation note or symbol translation should change, not the algorithm.

Counts: **2 code-side mismatch classes, 1 thesis-side representation difference**. No unexplained mathematical callable or missing grey-box algorithm was found. External data generation, evaluation-only quantities and derivations are intentional scope boundaries. Small differences within explicit numerical tolerances are not counted individually.

## Equation-test results

Initial development runs exposed test-fixture issues (scalar dtype conversion and in-place reference assembly under jacfwd); those were fixed. The normalization and S2 formula failures are retained, without expected-failure suppression. No production code is changed.

### Final CPU results

`conda run -n GraduationProject python -B Thesis-writeup/Code/Verification/clean-code/equation_tests.py`, 2026-10-28: **69 passed, 3 failed**, 72 named tests. The script returns 1 because the three formula failures are retained. There are 6 procedure rows explicitly not testable here, plus 19 external or input/I/O rows. P_stops has a passing split/plateau test; its complete trained state-machine sequence is deferred.

No training was performed: S1 captures actual loss gradients with a no-op optimizer; S2 uses a zero-update budget solely to evaluate its real nested objective. A_encoder_sim executes the prefit history/target construction with an in-memory observable synthetic baseline, plus its least-squares solve. No dataset or checkpoint is read or written.

Controller numerical exception: the factored continuous K1 formula, mapped through the bilinear s(z), is compared with the expanded discrete polynomial followed by tf2ss. The observed relative difference is 8.838560466491345e-10 (absolute 0.0066097763 on responses of order 1e7). This differently ordered, ill-conditioned coefficient conversion uses rtol=2e-9, atol=1e-11. Continuous shape and physical frequency response retain the default tolerance. It is numerical conditioning, with no change to controller design. B_mass uses its separately stated float32-provenance tolerance.

| Test/map row | Result | Largest absolute difference |
|---|---|---|
| `test_B_transform` | PASS | 5.55111512e-17 |
| `test_B_mass` | PASS | 7.62939457e-7 |
| `test_B_damping` | PASS | 0.00000000e+0 |
| `test_B_stiffness` | PASS | 0.00000000e+0 |
| `test_B_mass_split` | PASS | 0.00000000e+0 |
| `test_B_vector_field` | PASS | 3.55271368e-15 |
| `test_B_combinations` | PASS | 0.00000000e+0 |
| `test_B_admissibility` | PASS | predicate checks; no numeric difference |
| `test_B_mdelta` | PASS | 2.49800181e-16 |
| `test_A_rk4` | PASS | 1.08420217e-19 |
| `test_A_normalization` | FAIL | 1.00835850e-5 |
| `test_A_norm_transition` | PASS | 1.08420217e-19 |
| `test_A_norm_output` | FAIL | 9.06455359e-9 |
| `test_A_physical` | PASS | 6.93889390e-18 |
| `test_A_added` | PASS | 6.93889390e-18 |
| `test_A_network` | PASS | 0.00000000e+0 |
| `test_A_controller_state` | PASS | 8.88178420e-16 |
| `test_A_controller_force` | PASS | 8.88178420e-16 |
| `test_A_cl_step` | PASS | 3.55271368e-15 |
| `test_A_loss` | PASS | 0.00000000e+0 |
| `test_A_encoder_history` | PASS | 0.00000000e+0 |
| `test_A_encoder_linear` | PASS | 3.33066907e-16 |
| `test_A_encoder_fit` | PASS | 1.42108547e-14 |
| `test_A_encoder_sim` | PASS | 2.66453526e-15 |
| `test_A_init_beta` | PASS | 5.55111512e-17 |
| `test_A_init_network` | PASS | 0.00000000e+0 |
| `test_A_init_encoder` | PASS | 0.00000000e+0 |
| `test_P_residual` | PASS | 0.00000000e+0 |
| `test_P_s1_loss` | PASS | 0.00000000e+0 |
| `test_P_pairs` | PASS | 0.00000000e+0 |
| `test_P_s2_cost` | FAIL | 5.67650042e-1 |
| `test_O_sensitivity` | PASS | 1.73472348e-18 |
| `test_O_offset` | PASS | 3.60822483e-15 |
| `test_O_difference` | PASS | 0.00000000e+0 |
| `test_O_coefficient` | PASS | 1.07653436e-14 |
| `test_O_correction` | PASS | 6.77626358e-21 |
| `test_O_field` | PASS | 0.00000000e+0 |
| `test_O_tuple_position` | PASS | 2.84217094e-14 |
| `test_O_physical_step` | PASS | 0.00000000e+0 |
| `test_L_blocks` | PASS | 2.22044605e-16 |
| `test_L_chain` | PASS | 3.55271368e-15 |
| `test_L_det` | PASS | 4.54747351e-13 |
| `test_L_eta` | PASS | 2.49800181e-16 |
| `test_L_entries` | PASS | 0.00000000e+0 |
| `test_S_metric` | PASS | 0.00000000e+0 |
| `test_S_controller` | PASS | 6.60977628e-3 |
| `test_P_stops` | PASS | predicate checks; no numeric difference |
| `test_B_state` | PASS | 3.55271368e-15 |
| `test_A_output` | PASS | 5.55111512e-17 |
| `test_A_norm_inverse` | PASS | 3.55271368e-15 |
| `test_A_coordinates` | PASS | 2.49800181e-16 |
| `test_A_model_loop_state` | PASS | 8.88178420e-16 |
| `test_A_model_loop_force` | PASS | 3.55271368e-15 |
| `test_A_residual_state` | PASS | 3.55271368e-15 |
| `test_A_residual_force` | PASS | 3.55271368e-15 |
| `test_A_cl_output` | PASS | 3.55271368e-15 |
| `test_A_cl_input` | PASS | 3.55271368e-15 |
| `test_A_loss_physical` | PASS | 6.93889390e-18 |
| `test_A_loss_added` | PASS | 6.93889390e-18 |
| `test_A_loss_initial` | PASS | 3.33066907e-16 |
| `test_O_basis` | PASS | 1.73472348e-18 |
| `test_O_projected` | PASS | 0.00000000e+0 |
| `test_O_orthogonality` | PASS | 1.07653436e-14 |
| `test_O_added_step` | PASS | 0.00000000e+0 |
| `test_L_matrices` | PASS | 0.00000000e+0 |
| `test_L_factorization` | PASS | 3.55271368e-15 |
| `test_L_constant` | PASS | 3.55271368e-15 |
| `test_L_delta` | PASS | 3.55271368e-15 |
| `test_L_G_state` | PASS | 2.22044605e-16 |
| `test_L_G_latent` | PASS | 3.55271368e-15 |
| `test_L_G_output` | PASS | 5.55111512e-17 |
| `test_L_reduced_mass` | PASS | 0.00000000e+0 |

The failures are A_normalization (1.00835850e-5), A_norm_output (9.06455359e-9), and P_s2_cost (0.5676500417 for a batch of four, exactly the 3/4 sample-variance scaling). These are the two code-side mismatch classes M2 and M1. OBC basis-coordinate equivalence M3 passes.


## Active display index and backward map

Generated inventories below cover the active display source anchors and every callable. Source fingerprints bind line anchors to the tree read during this independent map; Claude's later comment rewrite can move lines without changing the map's mathematics.

### Complete backward inventory

Each name includes its module; nested closures remain under their parent. Property accessors have separate rows. Infrastructure rows state their actual source role.

| Qualified callable | Source | Thesis rows or infrastructure role |
|---|---|---|
| `gantry_interconnect_dynamic._dumps` | `Clean/gantry_interconnect_dynamic.py:21` | infrastructure: CLI, configuration, run folders and orchestration. Paths of the deepSI dumps _best.pth and _last.pth of a run; deepSI names them after the model class. |
| `gantry_interconnect_dynamic._rerun_source` | `Clean/gantry_interconnect_dynamic.py:27` | infrastructure: CLI, configuration, run folders and orchestration. (config.json, run id, source checkpoint) of a rerun, which continues in its run folder under the |
| `gantry_interconnect_dynamic._archive_rerun` | `Clean/gantry_interconnect_dynamic.py:46` | infrastructure: CLI, configuration, run folders and orchestration. Move the previous outcome, and for a resume both deepSI dumps, to previous/<rerun_id>/ and add a |
| `gantry_interconnect_dynamic.main` | `Clean/gantry_interconnect_dynamic.py:69` | infrastructure: CLI, configuration, run folders and orchestration. Build and train a fresh run or the requested resume/polish-only rerun; return its final validation score. |
| `gantry_interconnect_dynamic.cli` | `Clean/gantry_interconnect_dynamic.py:131` | infrastructure: CLI, configuration, run folders and orchestration. Parse the command line into a RunConfig and a rerun, then start main. |
| `gantry_dynamic.closed_loop_report._without_learned` | `Clean/gantry_dynamic/closed_loop_report.py:11` | A_encoder_sim, O_overlap |
| `gantry_dynamic.closed_loop_report._true_free` | `Clean/gantry_dynamic/closed_loop_report.py:26` | B_mdelta, O_residual |
| `gantry_dynamic.config.y_decimation` | `Clean/gantry_dynamic/config.py:51` | infrastructure: configuration, validation, provenance and serialized metadata. The rule that brings y to FS_NEW, as recorded in config.json. |
| `gantry_dynamic.config.record_trim` | `Clean/gantry_dynamic/config.py:58` | infrastructure: configuration, validation, provenance and serialized metadata. Return samples trimmed at each FS_NEW record end; sample i maps to FS_ORIG index (i + trim) D. |
| `gantry_dynamic.config.RunConfig.__post_init__` | `Clean/gantry_dynamic/config.py:131` | infrastructure: configuration, validation, provenance and serialized metadata. Check the axis and environment values, which arrive from the CLI, a config.json or the harness. |
| `gantry_dynamic.config.RunConfig.from_config_json` | `Clean/gantry_dynamic/config.py:145` | infrastructure: configuration, validation, provenance and serialized metadata. Read a supported run configuration, taking roots and start_phase from defaults or overrides. |
| `gantry_dynamic.config.RunConfig.obc` | `Clean/gantry_dynamic/config.py:167` | infrastructure: configuration, validation, provenance and serialized metadata |
| `gantry_dynamic.config.RunConfig.nx_ann` | `Clean/gantry_dynamic/config.py:171` | A_added |
| `gantry_dynamic.config.RunConfig.ann_route_ix` | `Clean/gantry_dynamic/config.py:175` | A_physical, A_added |
| `gantry_dynamic.config.RunConfig.mode` | `Clean/gantry_dynamic/config.py:179` | infrastructure: configuration, validation, provenance and serialized metadata |
| `gantry_dynamic.config.RunConfig.combo_init_detune` | `Clean/gantry_dynamic/config.py:183` | B_combinations, S_defaults |
| `gantry_dynamic.config.RunConfig.hp` | `Clean/gantry_dynamic/config.py:192` | S_defaults |
| `gantry_dynamic.config.save_dir` | `Clean/gantry_dynamic/config.py:207` | infrastructure: configuration, validation, provenance and serialized metadata. Output folder of the run family; the caller creates the run folder below it. |
| `gantry_dynamic.config.git_provenance` | `Clean/gantry_dynamic/config.py:213` | infrastructure: configuration, validation, provenance and serialized metadata. The code that produced this run, e.g. '0bded3d Augmentation dirty:302 diff:3b437e0c'. |
| `gantry_dynamic.config.git_provenance._git` | `Clean/gantry_dynamic/config.py:215` | infrastructure: configuration, validation, provenance and serialized metadata |
| `gantry_dynamic.config.config_json_dict` | `Clean/gantry_dynamic/config.py:239` | infrastructure: configuration, validation, provenance and serialized metadata. Return mainline configuration keys in their original order, then Clean keys and GIT. |
| `gantry_dynamic.controller.mass_matrix_at` | `Clean/gantry_dynamic/controller.py:34` | B_mass |
| `gantry_dynamic.controller.stage_frequency_response` | `Clean/gantry_dynamic/controller.py:43` | S_controller |
| `gantry_dynamic.controller.normalized_controller_coefficients` | `Clean/gantry_dynamic/controller.py:52` | S_controller |
| `gantry_dynamic.controller.build_cfb_at` | `Clean/gantry_dynamic/controller.py:59` | S_controller |
| `gantry_dynamic.controller.controller_ss` | `Clean/gantry_dynamic/controller.py:73` | S_controller |
| `gantry_dynamic.controller._stem` | `Clean/gantry_dynamic/controller.py:80` | infrastructure: controller metadata lookup and validation. A record's file name without folder and .mat suffix: the key of _CONTROLLER_METADATA. |
| `gantry_dynamic.controller.register_stored_controller` | `Clean/gantry_dynamic/controller.py:90` | infrastructure: controller metadata lookup and validation. Register a record's stored controller: (design Y, gain) if the bank can build it, else its description. |
| `gantry_dynamic.controller.y_op_for` | `Clean/gantry_dynamic/controller.py:96` | infrastructure: controller metadata lookup and validation. Controller design position Y of a record, by file name with or without .mat. |
| `gantry_dynamic.controller.gain_for` | `Clean/gantry_dynamic/controller.py:111` | infrastructure: controller metadata lookup and validation. Stored gain of a record's controller, 1.0 unless stored otherwise. |
| `gantry_dynamic.controller.build_controller_bank` | `Clean/gantry_dynamic/controller.py:122` | S_controller |
| `gantry_dynamic.controller.build_closed_loop` | `Clean/gantry_dynamic/controller.py:133` | A_loss |
| `gantry_dynamic.data.record_files` | `Clean/gantry_dynamic/data.py:24` | S_records |
| `gantry_dynamic.data._register_controller` | `Clean/gantry_dynamic/data.py:36` | infrastructure: data paths, manifest/loading and controller registration. Register the record's stored controller (meta.controller) for the closed-loop bank. |
| `gantry_dynamic.data.traj_dir` | `Clean/gantry_dynamic/data.py:74` | infrastructure: data paths, manifest/loading and controller registration. Folder of the data set of cfg.noise. |
| `gantry_dynamic.data._load_u` | `Clean/gantry_dynamic/data.py:79` | infrastructure: data paths, manifest/loading and controller registration. The plant input u_total = [F_X1, F_X2, F_Y] of a record. |
| `gantry_dynamic.data._resample_u` | `Clean/gantry_dynamic/data.py:84` | S_processing |
| `gantry_dynamic.data._decimate_y` | `Clean/gantry_dynamic/data.py:96` | S_processing |
| `gantry_dynamic.data._record_at_fs_new` | `Clean/gantry_dynamic/data.py:107` | S_processing |
| `gantry_dynamic.data._load_record` | `Clean/gantry_dynamic/data.py:120` | infrastructure: data paths, manifest/loading and controller registration. Load a record once and return (System_data, servo error), using folder when supplied. |
| `gantry_dynamic.data.servo_sigma` | `Clean/gantry_dynamic/data.py:130` | A_loss, S_metric |
| `gantry_dynamic.data.load_datasets` | `Clean/gantry_dynamic/data.py:140` | S_records |
| `gantry_dynamic.data.load_datasets._split` | `Clean/gantry_dynamic/data.py:147` | infrastructure: data paths, manifest/loading and controller registration |
| `gantry_dynamic.data.compute_normalization` | `Clean/gantry_dynamic/data.py:169` | A_normalization, A_norm_output, S_processing |
| `gantry_dynamic.model.build_model` | `Clean/gantry_dynamic/model.py:24` | A_physical, A_added |
| `gantry_dynamic.model.on_device` | `Clean/gantry_dynamic/model.py:107` | infrastructure: temporary device context. Move the model, built and kept on the CPU, to cfg.device for the work inside the block. |
| `gantry_dynamic.model.prefit_physical_encoder` | `Clean/gantry_dynamic/model.py:118` | A_encoder_sim |
| `gantry_dynamic.model.prefit_physical_encoder.recorder` | `Clean/gantry_dynamic/model.py:147` | A_encoder_sim |
| `gantry_dynamic.model.least_squares_start` | `Clean/gantry_dynamic/model.py:175` | A_encoder_fit, A_init_encoder |
| `gantry_dynamic.obc_gantry.GantryReferenceSet.n` | `Clean/gantry_dynamic/obc_gantry.py:38` | infrastructure: tuple count |
| `gantry_dynamic.obc_gantry.fourth_order_central_difference` | `Clean/gantry_dynamic/obc_gantry.py:42` | O_difference |
| `gantry_dynamic.obc_gantry.build_reference_set` | `Clean/gantry_dynamic/obc_gantry.py:49` | O_tuple_position |
| `gantry_dynamic.obc_gantry.GantryOBCCorrection.__init__` | `Clean/gantry_dynamic/obc_gantry.py:94` | O_correction |
| `gantry_dynamic.obc_gantry.GantryOBCCorrection.expansion_point` | `Clean/gantry_dynamic/obc_gantry.py:112` | O_refresh |
| `gantry_dynamic.obc_gantry.GantryOBCCorrection.set_frozen` | `Clean/gantry_dynamic/obc_gantry.py:117` | O_coefficient, O_refresh |
| `gantry_dynamic.obc_gantry.GantryOBCCorrection.tangent_pair` | `Clean/gantry_dynamic/obc_gantry.py:122` | O_correction |
| `gantry_dynamic.obc_gantry.GantryOBCCorrection.forward` | `Clean/gantry_dynamic/obc_gantry.py:128` | O_offset, O_projected, O_correction |
| `gantry_dynamic.obc_gantry.make_float64_block` | `Clean/gantry_dynamic/obc_gantry.py:140` | infrastructure: float64 model copy for OBC basis. A float64 copy of the physics block for the basis build; the live block keeps the training dtype. |
| `gantry_dynamic.obc_gantry.make_basis_builder` | `Clean/gantry_dynamic/obc_gantry.py:145` | O_offset, O_basis |
| `gantry_dynamic.obc_gantry.make_basis_builder.build` | `Clean/gantry_dynamic/obc_gantry.py:148` | O_basis |
| `gantry_dynamic.obc_gantry.make_reference_field` | `Clean/gantry_dynamic/obc_gantry.py:163` | O_field |
| `gantry_dynamic.obc_gantry.make_reference_field.field` | `Clean/gantry_dynamic/obc_gantry.py:169` | O_field |
| `gantry_dynamic.obc_gantry.attach_obc` | `Clean/gantry_dynamic/obc_gantry.py:181` | O_basis, O_field, O_correction |
| `gantry_dynamic.training._json_safe` | `Clean/gantry_dynamic/training.py:17` | infrastructure: checkpoint I/O, validation history or resume metadata. Convert NumPy integer and floating values recursively to native values for JSON metadata. |
| `gantry_dynamic.training.save_checkpoint_pair` | `Clean/gantry_dynamic/training.py:30` | infrastructure: checkpoint I/O, validation history or resume metadata. Save component state dictionaries to <base>.pt and metadata arrays to <base>.npz. |
| `gantry_dynamic.training.resolve_checkpoint` | `Clean/gantry_dynamic/training.py:42` | infrastructure: checkpoint I/O, validation history or resume metadata. Read either checkpoint format to (hfn_sd, encoder_sd, optimizer_sd or None, meta, format). |
| `gantry_dynamic.training.load_optimizer_moments` | `Clean/gantry_dynamic/training.py:71` | infrastructure: checkpoint I/O, validation history or resume metadata. Restore the optimizer's moment estimates and keep this run's hyperparameters. |
| `gantry_dynamic.training._assert_same_architecture` | `Clean/gantry_dynamic/training.py:86` | infrastructure: checkpoint I/O, validation history or resume metadata. Raise naming the mismatch, rather than load_state_dict's list of keys. |
| `gantry_dynamic.training._assert_same_dtype` | `Clean/gantry_dynamic/training.py:109` | infrastructure: checkpoint I/O, validation history or resume metadata. Refuse a checkpoint whose dtype differs from the run's. |
| `gantry_dynamic.training.load_checkpoint` | `Clean/gantry_dynamic/training.py:120` | infrastructure: checkpoint I/O, validation history or resume metadata. Restore the weights, and on an 'adam' start the Adam moments, from either format; returns the meta. |
| `gantry_dynamic.training.ValidationHistory.__init__` | `Clean/gantry_dynamic/training.py:157` | infrastructure: checkpoint I/O, validation history or resume metadata |
| `gantry_dynamic.training.ValidationHistory.record` | `Clean/gantry_dynamic/training.py:162` | infrastructure: checkpoint I/O, validation history or resume metadata. Append a row: from the simulator's last validation (events adam, rebuilt, polish_candidate), or a |
| `gantry_dynamic.training.ValidationHistory.hist_arrays` | `Clean/gantry_dynamic/training.py:174` | infrastructure: checkpoint I/O, validation history or resume metadata. The history as arrays, one entry per row, for val_hist.npz. |
| `gantry_dynamic.training.ValidationHistory.__call__` | `Clean/gantry_dynamic/training.py:187` | infrastructure: checkpoint I/O, validation history or resume metadata. Validation-probe entry: record the validation that just ran. |
| `gantry_dynamic.training.run_lbfgs_polish` | `Clean/gantry_dynamic/training.py:192` | A_optimizer |
| `gantry_dynamic.training._adam_phase` | `Clean/gantry_dynamic/training.py:230` | A_optimizer, P_alternation |
| `gantry_dynamic.training.train` | `Clean/gantry_dynamic/training.py:276` | A_optimizer, S_defaults |
| `model_augmentation.fit_systems.blocks.Block.__init__` | `Clean/model_augmentation/fit_systems/blocks.py:15` | infrastructure: framework construction, dimensions and stored data |
| `model_augmentation.fit_systems.blocks.Static_ANN_Block.__init__` | `Clean/model_augmentation/fit_systems/blocks.py:24` | A_network, A_init_network |
| `model_augmentation.fit_systems.blocks.Static_ANN_Block.forward` | `Clean/model_augmentation/fit_systems/blocks.py:40` | A_network |
| `model_augmentation.fit_systems.blocks.Linear_Output_Block.__init__` | `Clean/model_augmentation/fit_systems/blocks.py:50` | A_output |
| `model_augmentation.fit_systems.blocks.Linear_Output_Block.forward` | `Clean/model_augmentation/fit_systems/blocks.py:63` | B_transform, A_output, A_norm_output, L_G_output |
| `model_augmentation.fit_systems.blocks.Discrete_Nonlinear_Function_Block.forward` | `Clean/model_augmentation/fit_systems/blocks.py:73` | A_rk4 |
| `model_augmentation.fit_systems.blocks.GantryStateBlock.__init__` | `Clean/model_augmentation/fit_systems/blocks.py:84` | B_vector_field, A_normalization |
| `model_augmentation.fit_systems.blocks.GantryStateBlock._rk4` | `Clean/model_augmentation/fit_systems/blocks.py:128` | A_norm_transition, A_rk4 |
| `model_augmentation.fit_systems.blocks.GantryStateBlock._deriv_with` | `Clean/model_augmentation/fit_systems/blocks.py:139` | B_transform, B_vector_field, B_state, A_norm_inverse, L_factorization, L_chain, L_constant, L_delta, L_G_state, L_G_latent, L_det |
| `model_augmentation.fit_systems.blocks.GantryStateBlock._deriv_with_tangent` | `Clean/model_augmentation/fit_systems/blocks.py:193` | O_sensitivity, O_correction |
| `model_augmentation.fit_systems.blocks.GantryStateBlock._rk4_with_tangent` | `Clean/model_augmentation/fit_systems/blocks.py:258` | O_correction |
| `model_augmentation.fit_systems.blocks._TrainableGantryStateBlock.__init__` | `Clean/model_augmentation/fit_systems/blocks.py:282` | B_combinations |
| `model_augmentation.fit_systems.blocks._TrainableGantryStateBlock._build_KC` | `Clean/model_augmentation/fit_systems/blocks.py:293` | B_damping, B_stiffness, L_entries |
| `model_augmentation.fit_systems.blocks._TrainableGantryStateBlock._build_M1M2` | `Clean/model_augmentation/fit_systems/blocks.py:313` | B_mass_split, L_matrices, L_entries |
| `model_augmentation.fit_systems.blocks._TrainableGantryStateBlock._mats_from_raw` | `Clean/model_augmentation/fit_systems/blocks.py:328` | B_mass, B_mass_split, L_reduced_mass |
| `model_augmentation.fit_systems.blocks._TrainableGantryStateBlock.nonlinear_function` | `Clean/model_augmentation/fit_systems/blocks.py:340` | A_rk4 |
| `model_augmentation.fit_systems.blocks._TrainableGantryStateBlock.pass_mats` | `Clean/model_augmentation/fit_systems/blocks.py:346` | B_mass |
| `model_augmentation.fit_systems.blocks.IdentifiableGantryStateBlock.__init__` | `Clean/model_augmentation/fit_systems/blocks.py:365` | A_init_beta, L_eta |
| `model_augmentation.fit_systems.blocks.IdentifiableGantryStateBlock.combos_of` | `Clean/model_augmentation/fit_systems/blocks.py:388` | B_combinations, L_entries |
| `model_augmentation.fit_systems.blocks.IdentifiableGantryStateBlock.gauge_section` | `Clean/model_augmentation/fit_systems/blocks.py:399` | B_combinations |
| `model_augmentation.fit_systems.blocks.IdentifiableGantryStateBlock._m_diff_bound` | `Clean/model_augmentation/fit_systems/blocks.py:415` | B_admissibility |
| `model_augmentation.fit_systems.blocks.IdentifiableGantryStateBlock.combinations_from_free` | `Clean/model_augmentation/fit_systems/blocks.py:420` | B_admissibility, B_mdelta, A_coordinates |
| `model_augmentation.fit_systems.blocks.IdentifiableGantryStateBlock.free_from_combinations` | `Clean/model_augmentation/fit_systems/blocks.py:428` | B_mdelta |
| `model_augmentation.fit_systems.blocks.IdentifiableGantryStateBlock.recover_combinations` | `Clean/model_augmentation/fit_systems/blocks.py:445` | B_mdelta |
| `model_augmentation.fit_systems.blocks.IdentifiableGantryStateBlock.transition_from_free` | `Clean/model_augmentation/fit_systems/blocks.py:449` | O_sensitivity |
| `model_augmentation.fit_systems.blocks.IdentifiableGantryStateBlock.mats_and_tangent_from_free` | `Clean/model_augmentation/fit_systems/blocks.py:454` | O_sensitivity, O_correction |
| `model_augmentation.fit_systems.blocks.IdentifiableGantryStateBlock._mats_at` | `Clean/model_augmentation/fit_systems/blocks.py:458` | B_mass, B_mdelta |
| `model_augmentation.fit_systems.blocks.IdentifiableGantryStateBlock.estimation_parameters` | `Clean/model_augmentation/fit_systems/blocks.py:463` | infrastructure: B_combinations parameter-report metadata. Names, start values and current values of the identifiable combinations. |
| `model_augmentation.fit_systems.closed_loop.ControllerBank.__init__` | `Clean/model_augmentation/fit_systems/closed_loop.py:18` | A_controller_state, A_cl_output |
| `model_augmentation.fit_systems.closed_loop.ControllerBank.n_controllers` | `Clean/model_augmentation/fit_systems/closed_loop.py:34` | infrastructure: controller count |
| `model_augmentation.fit_systems.closed_loop.ControllerBank.step` | `Clean/model_augmentation/fit_systems/closed_loop.py:37` | A_controller_state, A_controller_force, A_model_loop_state, A_residual_state, A_cl_input |
| `model_augmentation.fit_systems.closed_loop.closed_loop_step` | `Clean/model_augmentation/fit_systems/closed_loop.py:46` | A_controller_force, A_model_loop_state, A_model_loop_force, A_residual_force, A_cl_output, A_cl_input, A_cl_step, A_loss_initial |
| `model_augmentation.fit_systems.closed_loop.closed_loop_step.step` | `Clean/model_augmentation/fit_systems/closed_loop.py:49` | A_cl_step |
| `model_augmentation.fit_systems.closed_loop.closed_loop_rollout` | `Clean/model_augmentation/fit_systems/closed_loop.py:59` | A_residual_state, A_cl_step, A_encoder_sim |
| `model_augmentation.fit_systems.closed_loop.window_counts` | `Clean/model_augmentation/fit_systems/closed_loop.py:77` | A_encoder_history |
| `model_augmentation.fit_systems.closed_loop.ClosedLoopSimulator.__init__` | `Clean/model_augmentation/fit_systems/closed_loop.py:87` | infrastructure: simulator metadata, compilation context, window alignment or serialization |
| `model_augmentation.fit_systems.closed_loop.ClosedLoopSimulator.__getstate__` | `Clean/model_augmentation/fit_systems/closed_loop.py:100` | infrastructure: simulator metadata, compilation context, window alignment or serialization. The state without the compiled step, a cache that cannot be pickled. |
| `model_augmentation.fit_systems.closed_loop.ClosedLoopSimulator.eager` | `Clean/model_augmentation/fit_systems/closed_loop.py:107` | infrastructure: simulator metadata, compilation context, window alignment or serialization. Run through the uncompiled step inside the block, then restore. |
| `model_augmentation.fit_systems.closed_loop.ClosedLoopSimulator.__call__` | `Clean/model_augmentation/fit_systems/closed_loop.py:115` | A_cl_step |
| `model_augmentation.fit_systems.closed_loop.ClosedLoopSimulator.augment_training_data` | `Clean/model_augmentation/fit_systems/closed_loop.py:132` | A_loss, A_encoder_history |
| `model_augmentation.fit_systems.closed_loop.ClosedLoopSimulator.verify_by_content` | `Clean/model_augmentation/fit_systems/closed_loop.py:151` | infrastructure: simulator metadata, compilation context, window alignment or serialization. Each record's first and last window against its raw data at the derived offset. |
| `model_augmentation.fit_systems.closed_loop.ClosedLoopSimulator.validation_error` | `Clean/model_augmentation/fit_systems/closed_loop.py:166` | S_metric |
| `model_augmentation.fit_systems.closed_loop.closed_loop_free_run_rms_batch` | `Clean/model_augmentation/fit_systems/closed_loop.py:178` | S_metric |
| `model_augmentation.fit_systems.interconnect.Interconnect.__init__` | `Clean/model_augmentation/fit_systems/interconnect.py:26` | infrastructure: signal-graph construction, routing matrices, device movement and dependency order |
| `model_augmentation.fit_systems.interconnect.Interconnect._apply` | `Clean/model_augmentation/fit_systems/interconnect.py:46` | infrastructure: signal-graph construction, routing matrices, device movement and dependency order. nn.Module._apply that also moves array_connection_matrices, a plain list, with the module. |
| `model_augmentation.fit_systems.interconnect.Interconnect.forward` | `Clean/model_augmentation/fit_systems/interconnect.py:54` | A_physical, A_added, A_norm_transition, A_loss_physical, A_loss_added, O_projected, O_physical_step, O_added_step |
| `model_augmentation.fit_systems.interconnect.Interconnect._propagate` | `Clean/model_augmentation/fit_systems/interconnect.py:85` | A_physical, A_added, A_output |
| `model_augmentation.fit_systems.interconnect.Interconnect.init_forward` | `Clean/model_augmentation/fit_systems/interconnect.py:109` | infrastructure: signal-graph construction, routing matrices, device movement and dependency order |
| `model_augmentation.fit_systems.interconnect.Interconnect._signal_cone` | `Clean/model_augmentation/fit_systems/interconnect.py:160` | infrastructure: signal-graph construction, routing matrices, device movement and dependency order. Output-signal indices that must be computed to obtain output signal target_output_ix. |
| `model_augmentation.fit_systems.interconnect.Interconnect.output_only` | `Clean/model_augmentation/fit_systems/interconnect.py:174` | A_output |
| `model_augmentation.fit_systems.interconnect.Interconnect.init_connection_matrices` | `Clean/model_augmentation/fit_systems/interconnect.py:187` | infrastructure: signal-graph construction, routing matrices, device movement and dependency order |
| `model_augmentation.fit_systems.interconnect.Interconnect.add_block` | `Clean/model_augmentation/fit_systems/interconnect.py:228` | infrastructure: signal-graph construction, routing matrices, device movement and dependency order. Add a block; its output and input get the next signal index. |
| `model_augmentation.fit_systems.interconnect.Interconnect.connect_signals` | `Clean/model_augmentation/fit_systems/interconnect.py:237` | infrastructure: signal-graph construction, routing matrices, device movement and dependency order. Connect two signals using the selected concatenation or additive mapping. |
| `model_augmentation.fit_systems.interconnect.Interconnect.connect_block_signals` | `Clean/model_augmentation/fit_systems/interconnect.py:249` | infrastructure: signal-graph construction, routing matrices, device movement and dependency order. Connect every signal of input_signal_list into block, and block into every signal of output_signal_list. |
| `model_augmentation.fit_systems.interconnect.Interconnect.determine_signal_ix` | `Clean/model_augmentation/fit_systems/interconnect.py:257` | infrastructure: signal-graph construction, routing matrices, device movement and dependency order |
| `model_augmentation.fit_systems.interconnect.Interconnect.convert_signal_name_to_ix` | `Clean/model_augmentation/fit_systems/interconnect.py:263` | infrastructure: signal-graph construction, routing matrices, device movement and dependency order |
| `model_augmentation.fit_systems.interconnect.Interconnect.parse_connection_function_method` | `Clean/model_augmentation/fit_systems/interconnect.py:269` | infrastructure: signal-graph construction, routing matrices, device movement and dependency order |
| `model_augmentation.fit_systems.interconnect.Signal_Connection.__init__` | `Clean/model_augmentation/fit_systems/interconnect.py:275` | infrastructure: connection descriptor |
| `model_augmentation.fit_systems.interconnect._replace_with_retry` | `Clean/model_augmentation/fit_systems/interconnect.py:284` | infrastructure: atomic checkpoint replacement retries. os.replace, retried while another process (a sync client, a virus scanner) briefly holds dst open, which |
| `model_augmentation.fit_systems.interconnect.DtypeDataLoader.__init__` | `Clean/model_augmentation/fit_systems/interconnect.py:300` | infrastructure: preserve dtype and deterministic seeded window shuffle |
| `model_augmentation.fit_systems.interconnect.DtypeDataLoader.__init__._conv` | `Clean/model_augmentation/fit_systems/interconnect.py:301` | infrastructure: preserve dtype and deterministic seeded window shuffle |
| `model_augmentation.fit_systems.interconnect.SSE_Interconnect.__init__` | `Clean/model_augmentation/fit_systems/interconnect.py:322` | infrastructure: fit-system construction and optimizer interface |
| `model_augmentation.fit_systems.interconnect.SSE_Interconnect.init_model` | `Clean/model_augmentation/fit_systems/interconnect.py:330` | infrastructure: fit-system construction and optimizer interface. Set nu and ny, build the optimizer over every parameter and reset bestfit. |
| `model_augmentation.fit_systems.interconnect.SSE_Interconnect.simulate` | `Clean/model_augmentation/fit_systems/interconnect.py:338` | A_cl_step |
| `model_augmentation.fit_systems.interconnect.SSE_Interconnect.make_training_data` | `Clean/model_augmentation/fit_systems/interconnect.py:343` | A_encoder_history |
| `model_augmentation.fit_systems.interconnect.SSE_Interconnect.cal_validation_error` | `Clean/model_augmentation/fit_systems/interconnect.py:349` | S_metric |
| `model_augmentation.fit_systems.interconnect.SSE_Interconnect.fit` | `Clean/model_augmentation/fit_systems/interconnect.py:357` | A_optimizer, P_alternation, P_handover |
| `model_augmentation.fit_systems.interconnect.SSE_Interconnect.fit.validation` | `Clean/model_augmentation/fit_systems/interconnect.py:361` | S_metric, A_optimizer |
| `model_augmentation.fit_systems.interconnect.SSE_Interconnect.fit.closure` | `Clean/model_augmentation/fit_systems/interconnect.py:418` | A_loss, P_alternation |
| `model_augmentation.fit_systems.interconnect.AugmentedModel.__init__` | `Clean/model_augmentation/fit_systems/interconnect.py:480` | infrastructure: framework block indexing, checkpoint serialization or optimizer property |
| `model_augmentation.fit_systems.interconnect.AugmentedModel._epoch_index` | `Clean/model_augmentation/fit_systems/interconnect.py:489` | O_refresh |
| `model_augmentation.fit_systems.interconnect.AugmentedModel._sync_obc_coefficient` | `Clean/model_augmentation/fit_systems/interconnect.py:494` | O_coefficient, O_refresh |
| `model_augmentation.fit_systems.interconnect.AugmentedModel.objective_matrices` | `Clean/model_augmentation/fit_systems/interconnect.py:507` | B_mass, O_correction |
| `model_augmentation.fit_systems.interconnect.AugmentedModel.loss` | `Clean/model_augmentation/fit_systems/interconnect.py:517` | A_loss, O_refresh |
| `model_augmentation.fit_systems.interconnect.AugmentedModel.checkpoint_save_system` | `Clean/model_augmentation/fit_systems/interconnect.py:535` | infrastructure: framework block indexing, checkpoint serialization or optimizer property. deepSI's dump of the model, written atomically. |
| `model_augmentation.fit_systems.interconnect.AugmentedModel.checkpoint_load_system` | `Clean/model_augmentation/fit_systems/interconnect.py:543` | infrastructure: framework block indexing, checkpoint serialization or optimizer property. deepSI's dump load, keeping the training-time state of this process. |
| `model_augmentation.fit_systems.interconnect.AugmentedModel.__getstate__` | `Clean/model_augmentation/fit_systems/interconnect.py:552` | infrastructure: framework block indexing, checkpoint serialization or optimizer property. The state without the training-time keys, as every dump holds it. |
| `model_augmentation.fit_systems.interconnect.AugmentedModel.optimizer` | `Clean/model_augmentation/fit_systems/interconnect.py:558` | infrastructure: framework block indexing, checkpoint serialization or optimizer property |
| `model_augmentation.fit_systems.interconnect.AugmentedModel.optimizer` (setter) | `Clean/model_augmentation/fit_systems/interconnect.py:562` | infrastructure: framework block indexing, checkpoint serialization or optimizer property |
| `model_augmentation.fit_systems.lbfgs_polish._restore` | `Clean/model_augmentation/fit_systems/lbfgs_polish.py:26` | infrastructure: A_optimizer: closure probes, optimizer counts, recovery and restore. Copy the snapshot back into the parameters, then pair theta_frozen of the OBC arm with them, keeping the |
| `model_augmentation.fit_systems.lbfgs_polish.FixedWindowBatch.__init__` | `Clean/model_augmentation/fit_systems/lbfgs_polish.py:39` | A_loss, A_optimizer |
| `model_augmentation.fit_systems.lbfgs_polish.FixedWindowBatch.__init__._to_tensor` | `Clean/model_augmentation/fit_systems/lbfgs_polish.py:63` | infrastructure: A_optimizer: closure probes, optimizer counts, recovery and restore |
| `model_augmentation.fit_systems.lbfgs_polish._make_closure` | `Clean/model_augmentation/fit_systems/lbfgs_polish.py:80` | A_loss, A_optimizer |
| `model_augmentation.fit_systems.lbfgs_polish._make_closure.closure` | `Clean/model_augmentation/fit_systems/lbfgs_polish.py:82` | A_loss, A_optimizer |
| `model_augmentation.fit_systems.lbfgs_polish._lbfgs_counts` | `Clean/model_augmentation/fit_systems/lbfgs_polish.py:92` | infrastructure: A_optimizer: closure probes, optimizer counts, recovery and restore. (iterations, closure evaluations) as torch counts them, cumulative over step() calls. |
| `model_augmentation.fit_systems.lbfgs_polish._is_oom` | `Clean/model_augmentation/fit_systems/lbfgs_polish.py:98` | infrastructure: A_optimizer: closure probes, optimizer counts, recovery and restore. Recognise CUDA OOM exceptions and RuntimeErrors containing 'out of memory'. |
| `model_augmentation.fit_systems.lbfgs_polish._probe_closure` | `Clean/model_augmentation/fit_systems/lbfgs_polish.py:104` | infrastructure: A_optimizer: closure probes, optimizer counts, recovery and restore. Evaluate the closure twice without a step; returns (identical, first value, error or None). |
| `model_augmentation.fit_systems.lbfgs_polish._acceptance` | `Clean/model_augmentation/fit_systems/lbfgs_polish.py:120` | A_optimizer |
| `model_augmentation.fit_systems.lbfgs_polish.lbfgs_polish` | `Clean/model_augmentation/fit_systems/lbfgs_polish.py:131` | A_optimizer |
| `model_augmentation.fit_systems.lbfgs_polish.lbfgs_polish._validate` | `Clean/model_augmentation/fit_systems/lbfgs_polish.py:150` | infrastructure: A_optimizer: closure probes, optimizer counts, recovery and restore |
| `model_augmentation.fit_systems.lbfgs_polish.lbfgs_polish._record_counts` | `Clean/model_augmentation/fit_systems/lbfgs_polish.py:160` | infrastructure: A_optimizer: closure probes, optimizer counts, recovery and restore |
| `model_augmentation.fit_systems.lbfgs_polish.lbfgs_polish._abort` | `Clean/model_augmentation/fit_systems/lbfgs_polish.py:164` | infrastructure: A_optimizer: closure probes, optimizer counts, recovery and restore. Abandon the polish: restore and resync the start weights, and release the CUDA cache after an |
| `model_augmentation.fit_systems.lbfgs_polish.lbfgs_polish._new_opt` | `Clean/model_augmentation/fit_systems/lbfgs_polish.py:186` | infrastructure: A_optimizer: closure probes, optimizer counts, recovery and restore |
| `model_augmentation.fit_systems.obc.OBCBasis.__init__` | `Clean/model_augmentation/fit_systems/obc.py:18` | O_coefficient |
| `model_augmentation.fit_systems.obc.OBCBasis.coefficient` | `Clean/model_augmentation/fit_systems/obc.py:38` | O_coefficient, O_orthogonality |
| `model_augmentation.fit_systems.obc.OBCBasis.to` | `Clean/model_augmentation/fit_systems/obc.py:45` | infrastructure: factor device movement or device property |
| `model_augmentation.fit_systems.obc.OBCBasis.device` | `Clean/model_augmentation/fit_systems/obc.py:52` | infrastructure: factor device movement or device property |
| `model_augmentation.fit_systems.obc.build_stacked_sensitivity` | `Clean/model_augmentation/fit_systems/obc.py:56` | O_sensitivity, O_basis |
| `model_augmentation.fit_systems.obc.OBCProjection.__init__` | `Clean/model_augmentation/fit_systems/obc.py:85` | infrastructure: store basis and reference callbacks. Store the reference-field evaluator and the basis builder returning (J, seconds). |
| `model_augmentation.fit_systems.obc.OBCProjection.refresh` | `Clean/model_augmentation/fit_systems/obc.py:94` | O_refresh |
| `model_augmentation.fit_systems.obc.OBCProjection.maybe_refresh` | `Clean/model_augmentation/fit_systems/obc.py:119` | O_refresh |
| `model_augmentation.fit_systems.obc.OBCProjection.coefficient` | `Clean/model_augmentation/fit_systems/obc.py:124` | O_coefficient |
| `model_augmentation.fit_systems.pre_encoder.AugmentedEncoder.__init__` | `Clean/model_augmentation/fit_systems/pre_encoder.py:17` | A_init_encoder |
| `model_augmentation.fit_systems.pre_encoder.AugmentedEncoder.forward` | `Clean/model_augmentation/fit_systems/pre_encoder.py:61` | A_loss_initial, A_encoder_history, A_encoder_linear |
| `model_augmentation.fit_systems.ps2_opening._plateaued` | `Clean/model_augmentation/fit_systems/ps2_opening.py:37` | P_stops |
| `model_augmentation.fit_systems.ps2_opening.calibration_split` | `Clean/model_augmentation/fit_systems/ps2_opening.py:42` | P_stops |
| `model_augmentation.fit_systems.ps2_opening.PS2Opening.__init__` | `Clean/model_augmentation/fit_systems/ps2_opening.py:56` | S_defaults, P_stops |
| `model_augmentation.fit_systems.ps2_opening.PS2Opening.done` | `Clean/model_augmentation/fit_systems/ps2_opening.py:79` | infrastructure: stage completion or progress serialization |
| `model_augmentation.fit_systems.ps2_opening.PS2Opening.after_step` | `Clean/model_augmentation/fit_systems/ps2_opening.py:82` | P_alternation, P_handover, P_stops |
| `model_augmentation.fit_systems.ps2_opening.PS2Opening.summary_json` | `Clean/model_augmentation/fit_systems/ps2_opening.py:115` | infrastructure: stage completion or progress serialization. The opening's record, written as ps2_summary. |
| `model_augmentation.fit_systems.ps2_opening.PS2Opening._start` | `Clean/model_augmentation/fit_systems/ps2_opening.py:122` | P_s1_loss, A_init_encoder |
| `model_augmentation.fit_systems.ps2_opening.PS2Opening._sample` | `Clean/model_augmentation/fit_systems/ps2_opening.py:141` | P_pairs |
| `model_augmentation.fit_systems.ps2_opening.PS2Opening._residual` | `Clean/model_augmentation/fit_systems/ps2_opening.py:166` | P_residual |
| `model_augmentation.fit_systems.ps2_opening.PS2Opening._added_states` | `Clean/model_augmentation/fit_systems/ps2_opening.py:173` | P_s1_loss, P_pairs |
| `model_augmentation.fit_systems.ps2_opening.PS2Opening._s1_step` | `Clean/model_augmentation/fit_systems/ps2_opening.py:177` | P_s1_loss, P_alternation |
| `model_augmentation.fit_systems.ps2_opening.PS2Opening._cal_unexplained` | `Clean/model_augmentation/fit_systems/ps2_opening.py:192` | P_s1_loss |
| `model_augmentation.fit_systems.ps2_opening.PS2Opening._fit_added_state_transition` | `Clean/model_augmentation/fit_systems/ps2_opening.py:199` | P_handover |
| `model_augmentation.fit_systems.ps2_opening.PS2Opening._fit_added_state_transition.snap` | `Clean/model_augmentation/fit_systems/ps2_opening.py:207` | P_pairs |
| `model_augmentation.fit_systems.ps2_opening.PS2Opening._fit_added_state_transition.state_map_loss` | `Clean/model_augmentation/fit_systems/ps2_opening.py:213` | P_s2_cost |
| `model_augmentation.systems.gantry_ss.build_poly_constants` | `Clean/model_augmentation/systems/gantry_ss.py:65` | L_det |
| `model_augmentation.systems.gantry_ss.build_G_matrix_entries` | `Clean/model_augmentation/systems/gantry_ss.py:91` | L_G_state, L_blocks |
| `model_augmentation.utils.deepSI_corrections.WindowedSystemData.__init__` | `Clean/model_augmentation/utils/deepSI_corrections.py:9` | infrastructure: framework construction, dimensions and stored data |
| `model_augmentation.utils.deepSI_corrections.WindowedSystemData.to_hist_future_data` | `Clean/model_augmentation/utils/deepSI_corrections.py:12` | A_encoder_history |
| `model_augmentation.utils.deepSI_corrections.SystemDataNormalizer.__init__` | `Clean/model_augmentation/utils/deepSI_corrections.py:35` | infrastructure: framework construction, dimensions and stored data |
| `model_augmentation.utils.deepSI_corrections.SystemDataNormalizer.transform` | `Clean/model_augmentation/utils/deepSI_corrections.py:41` | A_normalization |
| `model_augmentation.utils.torch_nets.build_mlp` | `Clean/model_augmentation/utils/torch_nets.py:11` | A_init_network |
| `model_augmentation.utils.torch_nets.AugmentationNetwork.__init__` | `Clean/model_augmentation/utils/torch_nets.py:33` | A_init_network |
| `model_augmentation.utils.torch_nets.AugmentationNetwork.forward` | `Clean/model_augmentation/utils/torch_nets.py:40` | A_network |
| `model_augmentation.utils.utils.added` | `Clean/model_augmentation/utils/utils.py:8` | infrastructure: identity attribution decorator. Mark a class or function as added by this project, not part of Jan's framework (grep "@added"). |
| `model_augmentation.utils.utils.to_tensor` | `Clean/model_augmentation/utils/utils.py:14` | infrastructure: tensor conversion. input as a tensor; a tensor is returned as it is. |
| `model_augmentation.utils.utils.expansion_matrix` | `Clean/model_augmentation/utils/utils.py:19` | A_physical, A_added |
| `model_augmentation.utils.utils.selection_matrix` | `Clean/model_augmentation/utils/utils.py:25` | A_physical, A_output |

### Generated constructors and anonymous closures

- `gantry_dynamic.config.RunConfig.__init__` (generated), `Clean/gantry_dynamic/config.py:64`: infrastructure, typed configuration/data record; S_defaults for RunConfig/PS2Spec/PolishSpec, A_normalization/O_tuple_position for Norm/DataBundle/GantryReferenceSet.
- `gantry_dynamic.data.DataBundle.__init__` (generated), `Clean/gantry_dynamic/data.py:46`: infrastructure, typed configuration/data record; S_defaults for RunConfig/PS2Spec/PolishSpec, A_normalization/O_tuple_position for Norm/DataBundle/GantryReferenceSet.
- `gantry_dynamic.data.Norm.__init__` (generated), `Clean/gantry_dynamic/data.py:62`: infrastructure, typed configuration/data record; S_defaults for RunConfig/PS2Spec/PolishSpec, A_normalization/O_tuple_position for Norm/DataBundle/GantryReferenceSet.
- `gantry_dynamic.obc_gantry.GantryReferenceSet.__init__` (generated), `Clean/gantry_dynamic/obc_gantry.py:28`: infrastructure, typed configuration/data record; S_defaults for RunConfig/PS2Spec/PolishSpec, A_normalization/O_tuple_position for Norm/DataBundle/GantryReferenceSet.
- `model_augmentation.fit_systems.lbfgs_polish.PolishSpec.__init__` (generated), `Clean/model_augmentation/fit_systems/lbfgs_polish.py:15`: infrastructure, typed configuration/data record; S_defaults for RunConfig/PS2Spec/PolishSpec, A_normalization/O_tuple_position for Norm/DataBundle/GantryReferenceSet.
- `model_augmentation.fit_systems.ps2_opening.PS2Spec.__init__` (generated), `Clean/model_augmentation/fit_systems/ps2_opening.py:14`: infrastructure, typed configuration/data record; S_defaults for RunConfig/PS2Spec/PolishSpec, A_normalization/O_tuple_position for Norm/DataBundle/GantryReferenceSet.
- Lambda under `gantry_dynamic.data._register_controller`, `Clean/gantry_dynamic/data.py:40`: implementation detail inheriting its parent mapping (conversion, key or evaluator).
- Lambda under `gantry_dynamic.model.prefit_physical_encoder`, `Clean/gantry_dynamic/model.py:125`: implementation detail inheriting its parent mapping (conversion, key or evaluator).
- Lambda under `gantry_dynamic.model.prefit_physical_encoder`, `Clean/gantry_dynamic/model.py:126`: implementation detail inheriting its parent mapping (conversion, key or evaluator).
- Lambda under `gantry_dynamic.model.least_squares_start`, `Clean/gantry_dynamic/model.py:179`: implementation detail inheriting its parent mapping (conversion, key or evaluator).
- Lambda under `gantry_dynamic.training.train`, `Clean/gantry_dynamic/training.py:316`: implementation detail inheriting its parent mapping (conversion, key or evaluator).
- Lambda under `model_augmentation.fit_systems.closed_loop.ControllerBank.__init__`, `Clean/model_augmentation/fit_systems/closed_loop.py:21`: implementation detail inheriting its parent mapping (conversion, key or evaluator).
- Lambda under `model_augmentation.fit_systems.closed_loop.closed_loop_free_run_rms_batch`, `Clean/model_augmentation/fit_systems/closed_loop.py:187`: implementation detail inheriting its parent mapping (conversion, key or evaluator).
- Lambda under `model_augmentation.fit_systems.closed_loop.closed_loop_free_run_rms_batch`, `Clean/model_augmentation/fit_systems/closed_loop.py:188`: implementation detail inheriting its parent mapping (conversion, key or evaluator).
- Lambda under `model_augmentation.fit_systems.obc.build_stacked_sensitivity`, `Clean/model_augmentation/fit_systems/obc.py:70`: implementation detail inheriting its parent mapping (conversion, key or evaluator).

Module-level nominal constants in `Clean/model_augmentation/systems/gantry_ss.py` implement B_transform, B_mass_split, B_damping, B_stiffness and app:params. The package __init__.py files have no callable definitions. Inherited torch/deepSI constructors are framework plumbing, not new definitions.

### Source fingerprints

| File | SHA256 |
|---|---|
| `Clean/gantry_interconnect_dynamic.py` | `573d1482d65ed5e9a04403788803eb06a4432e4dc307057bdae5f37ae2ac3e2d` |
| `Clean/gantry_dynamic/__init__.py` | `ac3d61d4c5ea371f40f29c12b9588e256bc8bf6bbf01199891169cc83e3551bd` |
| `Clean/gantry_dynamic/closed_loop_report.py` | `c1c35bce99f5de1a09fded53b3044098ce0c4ca66d0a36b2f81c98a876caeccc` |
| `Clean/gantry_dynamic/config.py` | `d644d1c2cbecf6f834d63c62ee0267259bafa9466ed9460a182b713b1b2365d6` |
| `Clean/gantry_dynamic/controller.py` | `340dcc76743fb61868ce40509c47315fd1113ec5f1d8cbf96cb21435039e046b` |
| `Clean/gantry_dynamic/data.py` | `832220b0555a43b461d4bdcbef3373200c31d4e4d638e6ac654fc931c0d7ae40` |
| `Clean/gantry_dynamic/model.py` | `4d7f4e49d00bca9e24ba71e0589376638eb55fa20c24765cc2fb18cc2b3357c3` |
| `Clean/gantry_dynamic/obc_gantry.py` | `df8a4cfa282ec7754adf36290465e961965e48d08d7a0ac3375fae100c94e572` |
| `Clean/gantry_dynamic/training.py` | `af7f5cf1f32739310621b9a80ae825a61fe6d0f8e7ec3406dd556877e3fc9668` |
| `Clean/model_augmentation/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `Clean/model_augmentation/fit_systems/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `Clean/model_augmentation/fit_systems/blocks.py` | `65e35e44e15dcb0fef44271f7d323c197fcf16831d08be0ff039c7ed7b20c299` |
| `Clean/model_augmentation/fit_systems/closed_loop.py` | `167b57a7719633458cd1f31d5ac21cb65943f42b96db005a126ff1a4e25ba77d` |
| `Clean/model_augmentation/fit_systems/interconnect.py` | `afb19314b804996b7d254631f8a5742584f7288f8a156a4b5423eff2f5c162e8` |
| `Clean/model_augmentation/fit_systems/lbfgs_polish.py` | `bbb681f1ff047887ca9391f79023f3d40117d0494ed92c8e8acb9f45bb08662e` |
| `Clean/model_augmentation/fit_systems/obc.py` | `9c91981d60c8e0d03d39ce57bf849b2761947beabc492470e74be688d2d3e0a6` |
| `Clean/model_augmentation/fit_systems/pre_encoder.py` | `a6edb3b595ded43b26cfa09e3da18a93bf7647aca39c465cff47aef34b6dbb33` |
| `Clean/model_augmentation/fit_systems/ps2_opening.py` | `855af7681ca5b6dd6517f4741e7b0d5dd38ccf7b5d319d2c16068d999413ef04` |
| `Clean/model_augmentation/systems/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `Clean/model_augmentation/systems/gantry_ss.py` | `d64a42ad4168ffaf6736b82f350040c28110db1baec79a419aaa6e31c6dcfc29` |
| `Clean/model_augmentation/utils/__init__.py` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `Clean/model_augmentation/utils/deepSI_corrections.py` | `c13d607eb65a1661123af663907b67195688c505f35e41b1cbd99731f849cf28` |
| `Clean/model_augmentation/utils/torch_nets.py` | `fcb2f7ccc519d3bb36c3d95e90db8d92edd3221314ebd1f67c45f9ff85589539` |
| `Clean/model_augmentation/utils/utils.py` | `30dddd776b9eb78ed1a561fd938b330011ec2685f73492ee04d303300d50ca0a` |

Inventory: 24 files, 209 explicit callable definitions, 9 anonymous closures, 97 forward rows.

### Active display coverage index

Labels apply to the mathematical rows separated in the forward map. Matrix entry rows stay with their enclosing matrix.

| Source | Label(s) or unlabelled ID | Forward rows |
|---|---|---|
| `Thesis-writeup/Writing/sections/02_system_baseline.tex:166` | `eq:system` | B_system, A_controller_state, A_controller_force |
| `Thesis-writeup/Writing/sections/02_system_baseline.tex:188` | `eq:system_loop` | A_controller_state, A_controller_force |
| `Thesis-writeup/Writing/sections/02_system_baseline.tex:397` | `eq:Ptransform` | B_transform |
| `Thesis-writeup/Writing/sections/02_system_baseline.tex:435` | `eq:eom` | B_vector_field |
| `Thesis-writeup/Writing/sections/02_system_baseline.tex:446` | `eq:mass_matrix` | B_mass |
| `Thesis-writeup/Writing/sections/02_system_baseline.tex:456` | `eq:damping_matrix` | B_damping |
| `Thesis-writeup/Writing/sections/02_system_baseline.tex:466` | `eq:stiffness_matrix` | B_stiffness |
| `Thesis-writeup/Writing/sections/02_system_baseline.tex:555` | `eq:Msplit` | B_mass_split |
| `Thesis-writeup/Writing/sections/02_system_baseline.tex:582` | `eq:base_ss` | B_state |
| `Thesis-writeup/Writing/sections/02_system_baseline.tex:745` | `02_system_baseline.tex:745` | B_identifiability |
| `Thesis-writeup/Writing/sections/02_system_baseline.tex:793` | `eq:phi` | B_combinations |
| `Thesis-writeup/Writing/sections/02_system_baseline.tex:838` | `eq:lfr_admissibility` | B_admissibility |
| `Thesis-writeup/Writing/sections/02_system_baseline.tex:864` | `eq:mdelta_map` | B_mdelta |
| `Thesis-writeup/Writing/sections/03_augmentation.tex:368` | `eq:aug_transition` | A_physical, A_added, A_output |
| `Thesis-writeup/Writing/sections/03_augmentation.tex:532` | `eq:aug_norm` | A_normalization, A_norm_transition, A_norm_output, A_norm_inverse |
| `Thesis-writeup/Writing/sections/03_augmentation.tex:599` | `eq:aug_norm_maps` | A_norm_transition, A_norm_output, A_norm_inverse |
| `Thesis-writeup/Writing/sections/03_augmentation.tex:665` | `03_augmentation.tex:665` | A_network |
| `Thesis-writeup/Writing/sections/03_augmentation.tex:823` | `eq:aug_two_loops` | A_model_loop_state, A_model_loop_force |
| `Thesis-writeup/Writing/sections/03_augmentation.tex:891` | `eq:aug_residual_loop` | A_residual_state, A_residual_force |
| `Thesis-writeup/Writing/sections/03_augmentation.tex:923` | `eq:aug_residual_split` | A_residual_split |
| `Thesis-writeup/Writing/sections/03_augmentation.tex:957` | `eq:aug_cl_step` | A_cl_output, A_cl_input, A_cl_step |
| `Thesis-writeup/Writing/sections/03_augmentation.tex:1119` | `eq:aug_loss` | A_loss, A_loss_physical, A_loss_added, A_loss_initial |
| `Thesis-writeup/Writing/sections/03_augmentation.tex:1159` | `eq:aug_encoder` | A_encoder_history |
| `Thesis-writeup/Writing/sections/03_augmentation.tex:1189` | `eq:aug_enc_lin` | A_encoder_linear |
| `Thesis-writeup/Writing/sections/03_augmentation.tex:1364` | `eq:aug_enc_sim` | A_encoder_sim |
| `Thesis-writeup/Writing/sections/03_augmentation.tex:1397` | `eq:aug_enc_fit` | A_encoder_fit |
| `Thesis-writeup/Writing/sections/03_augmentation.tex:1474` | `eq:aug_init` | A_init_beta, A_init_network, A_init_encoder |
| `Thesis-writeup/Writing/sections/03_augmentation.tex:1702` | `eq:ps2_s1` | P_s1_loss, P_residual |
| `Thesis-writeup/Writing/sections/03_augmentation.tex:1823` | `eq:ps2_alt` | P_alternation |
| `Thesis-writeup/Writing/sections/03_augmentation.tex:1881` | `eq:ps2_s2` | P_pairs, P_s2_cost |
| `Thesis-writeup/Writing/sections/03_augmentation.tex:1949` | `eq:ps2_handover` | P_handover |
| `Thesis-writeup/Writing/sections/04_obc.tex:137` | `eq:obc_gyorok` | O_gyorok |
| `Thesis-writeup/Writing/sections/04_obc.tex:190` | `eq:obc_sens` | O_sensitivity |
| `Thesis-writeup/Writing/sections/04_obc.tex:203` | `eq:obc_offset` | O_offset |
| `Thesis-writeup/Writing/sections/04_obc.tex:247` | `eq:obc_tuples` | O_tuple_position, O_difference |
| `Thesis-writeup/Writing/sections/04_obc.tex:266` | `eq:obc_coef` | O_basis, O_field, O_coefficient, O_projected, O_orthogonality |
| `Thesis-writeup/Writing/sections/04_obc.tex:307` | `eq:obc_corr` | O_correction, O_physical_step, O_added_step |
| `Thesis-writeup/Writing/sections/04_obc.tex:377` | `eq:obc_eps` | O_residual |
| `Thesis-writeup/Writing/sections/04_obc.tex:393` | `eq:obc_recovery` | O_recovery_projected, O_recovery_unprojected |
| `Thesis-writeup/Writing/sections/04_obc.tex:449` | `eq:obc_overlap` | O_overlap |
| `Thesis-writeup/Writing/sections/05_setup.tex:118` | `eq:setup_fric` | S_friction |
| `Thesis-writeup/Writing/sections/05_setup.tex:153` | `eq:setup_abs` | S_absorber |
| `Thesis-writeup/Writing/sections/05_setup.tex:591` | `eq:setup_rms` | S_metric |
| `Thesis-writeup/Writing/sections/99_appendix.tex:63` | `99_appendix.tex:63` | L_matrices |
| `Thesis-writeup/Writing/sections/99_appendix.tex:92` | `eq:lfr_mass_factorization` | L_factorization |
| `Thesis-writeup/Writing/sections/99_appendix.tex:99` | `eq:lfr_chain` | L_chain |
| `Thesis-writeup/Writing/sections/99_appendix.tex:106` | `eq:lfr_constant` | L_constant |
| `Thesis-writeup/Writing/sections/99_appendix.tex:115` | `eq:delta` | L_delta |
| `Thesis-writeup/Writing/sections/99_appendix.tex:126` | `eq:lfr_G` | L_G_state, L_G_latent, L_G_output |
| `Thesis-writeup/Writing/sections/99_appendix.tex:174` | `eq:app_lfr_blocks` | L_blocks, L_blocks_latent, L_blocks_output |
| `Thesis-writeup/Writing/sections/99_appendix.tex:199` | `99_appendix.tex:199` | L_schur |
| `Thesis-writeup/Writing/sections/99_appendix.tex:209` | `99_appendix.tex:209` | L_wellposed |
| `Thesis-writeup/Writing/sections/99_appendix.tex:245` | `99_appendix.tex:245` | L_reduced_mass |
| `Thesis-writeup/Writing/sections/99_appendix.tex:254` | `99_appendix.tex:254` | L_sylvester |
| `Thesis-writeup/Writing/sections/99_appendix.tex:258` | `99_appendix.tex:258` | L_det |
| `Thesis-writeup/Writing/sections/99_appendix.tex:269` | `99_appendix.tex:269` | L_raw_bound |
| `Thesis-writeup/Writing/sections/99_appendix.tex:279` | `99_appendix.tex:279` | L_eta |
| `Thesis-writeup/Writing/sections/99_appendix.tex:297` | `99_appendix.tex:297` | L_entries |
| `Thesis-writeup/Writing/sections/99_appendix.tex:311` | `99_appendix.tex:311` | L_inverse_entries |
| `Thesis-writeup/Writing/sections/99_appendix.tex:371` | `99_appendix.tex:371` | S_extended_matrices |
| `Thesis-writeup/Writing/sections/99_appendix.tex:388` | `eq:setup_K1` | S_controller |

Active display count: 61. No active in-scope display left unmapped.
