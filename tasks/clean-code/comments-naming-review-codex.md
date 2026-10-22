# Review of the Clean comment and naming plan

2026-10-22. Reviewed `comments-naming-plan.md` against all 24 Python files in `Clean/`, the evaluation imports and attribute reads, the verification harness, Jan's `c9f913a`, and the method notation in `03_augmentation.tex` and `04_obc.tex`.

Only this review is written. No training, harness execution, imports of project modules, git writes, or protected-data reads were performed. Checks used text searches, Python AST parsing and read-only `git show`. Existing workspace changes were left in place.

The pass is worth doing, but revise the names and sample mathematics before approval. The most consequential comment error is equating the implemented affine basis with the thesis's offset-column basis: their spans agree, their columns and coefficients do not.

## 1. Naming convention and R1 to R12

Use PascalCase for our classes, including subclasses of Jan's classes. Preserve Jan's original spelling exactly. Inheriting `SSE_Interconnect` does not require a new class to inherit its underscore convention. Use snake_case for functions and locals, uppercase for constants, and a leading underscore for private module helpers and implementation attributes. A temporary local generally needs no leading underscore. Mathematical locals can remain short when their meaning is established beside the calculation.

Prefer the method's terms: augmented model, physical baseline, added states, encoder, orthogonal projection, polish and servo error. Do not make a class name an inventory of all its features.

| Row | Verdict and recommended name | Reason |
| - | - | - |
| R1 | Replace `SSE_Interconnect_Composed` with `AugmentedModel`, rather than `SSE_Interconnect_Weighted`. | It is the fitted augmented state-space model, with encoder, objective, OBC integration and checkpoint handling. Weighted names only one detail of its objective and preserves the mixed class convention. The shared class also serves the unconstrained arm, so an OBC-only name would be wrong. |
| R2 | Replace `OBCLifecycle` with `OBCProjection`. | It owns the projection basis, rank policy and coefficient solve. `OBCProjector` is understandable, but suggests it directly returns a projected field; the actual state correction remains in `GantryOBCCorrection`. Rename local `life` to `projection`. |
| R3 | Accept `AugmentedEncoder`. | It estimates both physical and added states and remains trainable; the old name misleadingly suggests an initialisation routine. |
| R4 | Use `AugmentationNetwork` instead of `AugmentationNet`. | It supplies both physical corrections and added-state updates. Remove `live_g` from the class name; explain the deliberately different initialisation of its output rows in one comment. |
| R5 | Accept `ValidationHistory`. | Clear and consistent with the other class names. |
| R6 | Accept `_sync_obc_coefficient`. | All callers synchronise the frozen prediction coefficient, beyond validation. Its docstring must also say that a requested refresh, or a missing basis, rebuilds the basis. |
| R7 | Accept `_deriv_with_tangent`. | It returns the derivative and its directional tangent, matching `_rk4_with_tangent`. Keep both returns explicit in its description. |
| R8 | Use `_CONTROLLER_METADATA` instead of `_STORED_CONTROLLERS`. | Values are design-position/gain tuples or rejection descriptions, not stored controller systems or matrices. Uppercase is appropriate for the module-level registry name even though its dictionary is populated at runtime. |
| R9 | Accept `_tf_to_block_diag_ss`. | It assembles one block-diagonal state-space system from per-channel transfer functions, not a batch of systems. |
| R10 | Accept `n_a`, `nx`, `raw_names`, `raw_nominal`; prefer `initial_combinations` to `combo_start`. | These distinguish the added-state count, total state count, nominal raw parameters and possibly detuned starting combinations. Preserve the literal hyperparameter key `NX_ANN`. In comments, bridge `n_a` to the thesis's `n_{bar x}` once. |
| R11 | Accept removing the listed temporary-local underscores, with the mappings below. | This is a readability pass, not a rule that every underscore is meaningless. Preserve private module names and real implementation attributes. |
| R12 | Use `state_map_loss` for nested function `ee`; accept `use_compiled` for `use_c`. | The nested function is the normalised added-state transition-fitting loss, not an unexplained stage cost. `use_compiled` states the actual branch condition clearly. |

For R11, use these explicit mappings rather than a blind underscore deletion:

| Location | Mapping |
| - | - |
| config | `_signs` to `signs` |
| training checkpoint reader | `_opt` to `optimizer`, `_opt_sd` to `optimizer_state`, `_dropped` to `optimizer_state_dropped` |
| training history selection | `_adam` to `adam_rows`, `_last` to `last_by_event` |
| training final save | `_polished` to `polish_accepted`, `_no_adam_ran` to `polish_only`, `_drop_optimizer` to `drop_optimizer_state`, `_why` to `drop_reason` |
| closed-loop scorer | `_p` to `parameter` |
| interconnect | `_model_dtype` to `model_dtype`, `_selected_checkpoint` to `selected_checkpoint`, `_corr` to `correction`, `_theta` to `theta_aux` |
| encoder | `_ny_hist` to `n_output_history_values`, `_nu_hist` to `n_input_history_values` |
| entry point | `_f` to `fh` |

The proposed `polish_only` correction is especially useful: the old local tests `start_phase == 'lbfgs'`, while the separate `adam_ran` also considers an exhausted epoch budget. Do not conflate the two.

## 2. Verdict on every considered-and-kept group

| Name or group | Verdict |
| - | - |
| `hp` | Keep the parameter, property and stored field. It is familiar in this code and used by evaluation. Replacing its contents with direct config reads changes the interface and is outside this pass. |
| `RunConfig`, `nx_ann`, `combo_init_detune`, ClassVar constants | Keep. `RunConfig` already follows the chosen convention. The evaluation maps dataclass field names to uppercase JSON keys and reads several properties; preserve this contract. These names do not all become keys automatically: `config_json_dict` explicitly constructs most keys, so document the actual dependency rather than claiming every attribute spelling is inherently a JSON key. |
| `DataBundle`, including `val_ckpt_data`, all split lists/files and `e_train`, `e_val`, `e_test` | Keep. Evaluation and harness consumers use these interfaces, including dynamic split attribute construction in `eval_greybox.split_records`. `val_ckpt_data` is less clear than `validation_data`, but a rename is forbidden by the evaluation boundary. |
| `Norm` and its fields | Keep. Evaluation reads normalisation fields and the fixed output map. `Normalization` would be clearer in isolation, but preserving this established external surface is preferable; field renames are prohibited where read. |
| `closed_loop_report`, `_without_learned`, `_true_free` | Keep all three. Evaluation imports both helpers from that exact module. Private spelling does not mean internal-only use here. |
| `_load_u`, `_resample_u`, `_record_at_fs_new`, `_load_record` | Keep. Evaluation imports them directly, including the private names. |
| `_ann_ix`, `_mats_ix` | Keep. Harness reads both block indices. These are private attributes, not temporary locals to strip under R11. |
| `obc_reference` | Keep. Harness reads the reference-set object. |
| `epochs_before`, `ps2_opening` | Keep. They are whole-dump metadata keys read on continuation and checked by the harness. They are not module parameter names. |
| `record_weights`, `val_rows`, `train_ctrl_rows`, `last_per_channel`, `last_sim_rms`, `val_sigma_e` | Keep each. Harness reads these fields; the names are adequate once record ordering, units and the servo-error denominator are explained. |
| `least_squares_start`, `prefit_physical_encoder` | Keep both. The former is patched by `capture_l12.py`; the latter is called there. The existing names communicate their purpose well enough. |
| `obc_correction` | Keep. Registered module path and direct evaluation/harness access. |
| `Wb_psi_y`, `Wb_psi_u`, `Wa_psi_y`, `Wa_psi_u` | Keep every parameter name. They are checkpoint keys and map to the thesis's physical/added encoder rows. |
| `free_params`, `theta_frozen`, `vbar` | Keep. They are trainable parameter or persistent buffer names, and evaluation uses them. Explain `beta`, `theta_aux` and `beta_lin` beside them instead. |
| `out_gate` | Keep, but correct the stated reason: it is registered with `persistent=False`, so it is absent from `state_dict`. It is used by `_without_learned`; its intervention role is clear. |
| `Y_op`, `N_op`, `d_op` | Keep. Evaluation assigns all three for frozen scheduling. `Y_op` is an ordinary attribute; the latter two are assigned tensors in that evaluation path, not constructor-registered persistent buffers in Clean. |
| `Interconnect`, `SSE_Interconnect`, `Signal_Connection`, `Block`, `Static_ANN_Block`, `Discrete_Nonlinear_Function_Block`, `Linear_Output_Block` | Keep every name. Confirmed in Jan's commit. A rewritten implementation or an `@added` marker does not transfer ownership of an established name. |
| `to_tensor`, `expansion_matrix`, `selection_matrix` | Keep all three. Confirmed as Jan's original function names, including the rewritten `to_tensor`. |
| `fixed_System_data`, `fixed_System_data_norm` | Reject the keep decision. Use `WindowedSystemData` and `SystemDataNormalizer`. These classes are ours in a new file, absent from Jan's commit, and neither class name is imported/read by the evaluation or harness. Their relationship to deepSI belongs in the base class and docstring; `fixed` names a historical repair. Keep inherited method names. |
| `PS2Opening`, `PS2Spec`, every spec field | Keep. PS2 is the named procedure in the code and run interface, not merely a job label. Define it once as predictive-state initialisation of the added states. Preserve summary keys from `asdict(spec)`, stage/status strings and the `ps2_summary` payload. |

## 3. Additional names the plan missed

This list comes from inspecting definitions, assignments, registered parameters/buffers and their readers across the whole folder. It lists the additional changes I recommend; it does not propose replacing every conventional mathematical abbreviation.

| Current name | Recommendation | Safety and reason |
| - | - | - |
| `Gantry_State_Block` | `GantryStateBlock` | Our `@added` class; no evaluation/harness name dependency. Preserve all buffer names. This is a physical transition block, not the complete augmented model. |
| `_Trainable_Gantry_State_Block` | `_TrainableGantryStateBlock` | Our private base class; no external name dependency. Retain the private prefix and use the same class convention. |
| `Reduced_Gantry_State_Block` | `IdentifiableGantryStateBlock` | Our `@added` class; no evaluation/harness class import. It trains the ten identifiable parameter combinations, not a reduced-order state model: the physical state dimension remains six. Update imports and class-qualified calls in model construction and OBC code. |
| `Dtype_DataLoader` | `DtypeDataLoader` | Our `@added` class, absent in Jan's commit; only its internal construction changes. Preserve deepSI-required instance fields and iteration interface. |
| `mlp_stack` | `build_mlp` | Our `@added` function; no evaluation/harness import. It constructs and initialises a network, returning both the sequential module and final layer. Update both network and encoder callers. |
| `_replace` | `_replace_with_retry` | Our `@added` helper; only the checkpoint-save call changes. The retry behaviour is the useful distinction from `os.replace`. |
| controller `M_op` | `mass_matrix_at` | Own new-file function, no evaluation/harness reader. Names the physical matrix and evaluation point. |
| controller `sys_stage_frf` | `stage_frequency_response` | Own function, no external reader; returns a frequency-response matrix, not a system object. |
| controller `cnorm_coeffs` | `normalized_controller_coefficients` | Own function, no external reader. Clarifies that these are controller-shape polynomials, not data-normalisation coefficients. |
| model `PHY_IX` | `PHYSICAL_STATE_IX` | Own module constant, no external reader. Describes the selected state rows directly. |
| model `_START_MAX_UNEXPLAINED` | `_MAX_ENCODER_UNEXPLAINED_FRACTION` | Own constant, no external reader. Names the encoder-fit acceptance quantity. |
| OBC `cand`, `blk64`, `corr`, `ref` in longer construction/refresh functions | `candidate_basis`, `baseline64`, `correction`, `reference` | Locals only; improve the long functions without changing corresponding attributes or helper arguments used elsewhere. |
| model `_prefit` locals `T`, `rv`, `uh_all`, `yh_all`, `xt_all` | `as_tensor`, `as_vector`, `input_windows`, `output_windows`, `state_targets` | Locals only; make the simulation-to-regression path easier to follow. Short `X`, `U`, `W`, `b` can stay inside the actual algebra. |
| PS2 `_head`, `_fit_map`, `_xa` | `_decoder`, `_fit_added_state_transition`, `_added_states` | Own implementation attributes/methods, no evaluation/harness reader. `_head` belongs to the plain PS2 helper object, not a registered parent module, and that training object is excluded from dumps. Preserve its optimizer parameter list and summary strings. |
| polish `FixedBatch` | `FixedWindowBatch` | Own new-file class, no external reader. Identifies the deterministic simulation-window batch used for polish. |

Do not rename `servo_sigma`: it already uses the method's term and evaluation imports it. Do not rename `build_stacked_sensitivity`, `evaluate_step_rows`, `build_reference_set`, `make_float64_block`, `make_basis_builder`, `transition_from_free`, `combinations_from_free`, `free_from_combinations`, `pass_mats`, `estimation_parameters`, `PARAM_NAMES`, `COMBO_NAMES` or reference-set fields: evaluation imports or reads these interfaces. The ten combination-name strings are used to locate entries, notably `m_diff`, and should stay.

The other own class names (`OBCBasis`, `OBCRankLossAbort`, `ControllerBank`, `ClosedLoopSimulator`, `GantryReferenceSet`, `GantryOBCCorrection`, `PolishSpec`, `DataBundle`, `RunConfig`) need no additional rename. `OBCRankLossAbort` is unusual but accurately signals termination under the rank policy. Keep the remaining module filenames: several are externally imported, Jan's filenames must stay, and no additional own filename change provides enough clarity to justify a second naming layer. Keep private module constants, package import aliases and module-initialisation locals private. In the gantry constructor, `_alpha` through `_A_combined`, `_C` and `_D` can remain consistently distinguished from destination buffer names; do not mechanically strip them.

## 4. Rename safety audit

I searched the old names across `Clean/`, all evaluation Python files and all harness Python files. I also inspected AST attribute accesses and literal `getattr`/`hasattr` calls, plus the dynamic split-name construction and config field mapping. Ownership was checked by markers and by parsing the classes/functions in Jan's commit, rather than trusting the plan's ownership statement.

| Row | Ownership and consumers checked | Safety verdict |
| - | - | - |
| R1 | `@added` subclass, absent in Jan's commit. Internal model import/call and entry-point dump pattern. Harness `gpu_smoke.py`, `run_l3bc.py`, `compare_l3bc.py`. No evaluation name reference. | Safe only with the stated harness adaptations and regenerated clean dumps. The plan correctly identifies the filename side effect. Keep the old class on the mainline branch of each shared harness path. |
| R2 | New-file class; import, construction and docstring in `obc_gantry.py`. Harness accesses `fit_sys.obc` and its basis, not `OBCLifecycle`. | Safe. Preserve `obc` and the basis interface/fields. |
| R3 | `@added` class; model import/construction and class name in `super(...)`. Not in Jan's commit or evaluation/harness imports. | Safe. Update the explicit `super` reference too. Preserve all parameter/module attribute names. |
| R4 | `@added` class; model import/network argument and explicit `super(...)`. Not in Jan's commit or evaluation/harness imports. | Safe. Update `super`; keep `self.net` and all configuration literals. |
| R5 | New-file class and construction inside `train`. No external name reference. | Safe. Preserve NPZ keys and event strings. |
| R6 | Method in our subclass; six calls across interconnect, model, polish and PS2. No external name reference. | Safe. The plan's six-call count is correct: two in interconnect, one in model, one in polish and two in PS2. |
| R7 | Method in our `@added` physical block; four RK4 calls. No external reference. | Safe. Four calls plus the definition. |
| R8 | Own new-file module global; registration and two lookup functions. No external reference. | Safe. Three uses plus the assignment, matching the plan if its count excludes the definition. |
| R9 | Own new-file helper; one call in `controller_ss`. No external reference. | Safe. |
| R10 | Function locals in our new module; `NX_ANN` also occurs as a literal key. | Safe as a scoped identifier rename. A text replacement of the literal key is forbidden. |
| R11 | The listed names are temporary locals in our functions/classes, or our added `fit` method. | Safe as scoped renames. `_optimizer`, `_UNPICKLABLE`, `_compiled` and PS2 implementation attributes are not covered by this row. Preserve dictionary literals such as `optimizer_state_dropped`. |
| R12 | Nested function definition/calls in PS2 and local branch variable in the own closed-loop simulator. | Safe. `ee` is a function, not merely a scalar local as the row implies. |

The class and method renames do not change `state_dict` keys: parameter/module paths come from attributes such as `connected_blocks`, `net`, `encoder`, `obc_correction`, `Wb_psi_y` and `free_params`, all preserved. Class names do affect pickle lookup and deepSI dump filenames, so the general statement that class names are never in keys is insufficient as the whole safety argument. The accepted regenerated-dump policy resolves pickle compatibility for this task; it does not remove the filename adaptations.

Grep confirms the frozen strings singled out by the plan: `compare_gpu.py` searches for `training rollout COMPILED`; `run_l3bc.py` requires `PS2` in the unfinished-opening error. PS2 summaries/status, validation event names, polish reason text and config values participate in recorded or compared outputs. Keep these strings unchanged under the agreed contract. R1 is the explicit filename exception, handled with a class-prefix mapping in `compare_l3bc.py` in addition to its run-id normalisation. No harness or evaluation edits were made in this review.

## 5. Marker plan

The marker rule is about provenance, independent of a naming or comment preference. Apply it to the copied framework in Clean using Jan's commit as the provenance reference.

| Item | Verdict |
| - | - |
| 1, retain existing markers | Accept. Preserve `@added` even on rewritten Jan-named objects such as `Linear_Output_Block` and `to_tensor`; preserve their names separately. New-file markers cover new files and their contents. |
| 2, replace 26 trim comments with five file headers | Accept the consolidation. The 26 count is correct: 21 in interconnect and five in blocks. Put the module docstring first, then the header. Describe the current restriction, for example `# CHANGED: restricted to the thesis training path; full implementation in model_augmentation/fit_systems/interconnect.py at the repository root`. This deliberate provenance explanation is an exception to avoiding historical clutter. |
| 3, keep markers inside Jan's definitions | Accept with a complete inventory, not an approximate count. Keep short reasons at every changed block in inherited code, including module imports needed for the additions and the changed initialisation of `norm`. `# CHANGED: subtract f_obc` merely restates the line; say why the baseline-direction component is removed. |
| 4, mark added methods | Add `@added` to the seven unmarked methods: `Interconnect._apply`, `_signal_cone`, `output_only` and `SSE_Interconnect.simulate`, `make_training_data`, `cal_validation_error`, `fit`. They are absent from Jan's classes; `_propagate` is already decorated. There are eight added methods in this group, requiring seven new decorators. The inherited deepSI methods do not make them Jan-authored definitions. The decorator is an identity function, so this is a provenance edit with no intended numerical change. |
| 5, remove redundant markers | Accept inside our marked files, our `@added` definitions and newly decorated methods. Reject deleting all three import markers without preserving their provenance: they are changed blocks in Jan's module. Consolidate them into one marked import group if desired. A header that says only trimmed does not explain added dependencies. |
| 6, theory and heuristic labels | Retain the labels where required, with short accurate formulas/sources and explicit engineering choices. Correct the counts: Clean has 20 literal `# THEORY:` occurrences and 12 literal `# HEURISTIC:` occurrences, not 23 and 16. The plan's sample adds two theory labels, but that still does not justify its stated current counts. Labels inside existing docstrings are included in these counts. |
| 7, edit CLAUDE.md | Do not add a new rule. The existing changed-block marker can already describe a restricted copy. The current task authorizes only this review. |

The nine-file premise is also wrong. `Clean/model_augmentation/` contains 11 nonempty Python files, plus four empty package initialisers. Five nonempty files retain Jan's original filenames; six have new-file markers: closed loop, polish, OBC, PS2, gantry constants and deepSI corrections. Across Clean there are 24 Python files, four empty. The total 87 changed-marker occurrences, comprising 26 trim and 61 other occurrences, is correct. Do not treat the plan's approximate target of 14 markers as an acceptance criterion.

## 6. Sample comments and formulas

The samples remove much clutter, but neither is ready to set the standard for the remaining files. Several externally called functions still have multiline docstrings; factual formula corrections matter more than polishing their wording.

### OBC sample

1. The two-line module docstring meets the length rule, but baseline Jacobian is incomplete for the configured affine basis. Describe orthogonal projection onto baseline sensitivity and offset directions. The generic code supplies the factorisation and coefficient; the gantry code supplies those directions. Naming Gyorok once is reasonable if it distinguishes the adopted projection from the local nonlinear-baseline construction.
2. `OBCBasis` does not always contain a full-column-rank Jacobian. Its constructor also forms a truncated candidate so the caller can reject it. Say it factors a column-scaled projection basis and records its numerical rank. A valid basis is normally retained for an epoch, but explicit refresh and checkpoint restoration can rebuild it.
3. The sample's coefficient formula is wrong as written for the stored SVD. Let `D = diag(column_scale)`. The code factors `J D^-1 = U S V^T`, so for a retained full-rank basis the implemented coefficient is `theta_aux = D^-1 V S^-1 U^T F = (J^T J)^-1 J^T F`. The sample omits `D^-1` from the SVD expression and implicitly identifies its factors with the SVD of unscaled J. The old comment already had this problem. Include the full-rank condition; the inverse formula does not describe a rank-deficient candidate.
4. More seriously, `J = Phi_Gamma` is not literally true for the deployed affine path. The thesis defines `gamma = f_base(beta_lin) - varphi beta_lin` and stacks `[Phi Gamma]`. `make_basis_builder` appends the baseline transition itself to the sensitivity, giving `J_code = [Phi F_base(beta_lin)]`. Their column spans coincide because `F_base = Phi beta_lin + Gamma`. Consequently the projection agrees, but the fitted coordinates differ: if the code coefficient is `[a; c]`, the thesis coefficient is `[a + beta_lin c; c]`. `theta_frozen` contains code coordinates, and the correction implements `varphi a + c f_base`, not `varphi a + c gamma`. Explain this basis change once in the gantry-specific code; do not claim identical columns or coefficients, and do not change the numerical basis in a comment pass.
5. `build_stacked_sensitivity` builds only `Phi`, before the extra column is appended. Its derivative and sample-major shape statement are correct, but use `beta`, `beta_lin` and `f_base^n` when mapping it to the thesis. The proposed `v` and `vbar` formula remains implementation notation. A one-line docstring can describe stacking sensitivities, with a short mathematical comment if needed.
6. `evaluate_step_rows` correctly describes row selection, sample-major stacking, float64 and no gradient. This returns a transition value, not the thesis offset gamma by itself.
7. The numerical rank threshold `tol = S_max max(M,N) eps` matches the code, with S belonging to the column-scaled matrix. Keep that qualification. It is a numerical rank criterion, not a proof of physical identifiability.
8. `refresh` keeps the last valid basis after one failed refresh and raises on the configured consecutive-failure limit, default two. The sample hardcodes second time in its docstrings despite the mutable `max_consecutive_rank_loss`. State the policy through that setting and shorten the public docstring. `coefficient` differentiates through F with the basis held fixed; that is the useful fact to retain.
9. Both revised rank errors still fail the requested error-message standard. The unchanged prefix ends a sentence before the new suffix; neither suffix gives an action. Use one sentence that reports the rank failure and asks the reader to check reference excitation or redundant protected directions. The first-build explanation must include the affine baseline column, not only training coordinates. Do not prescribe lowering the tolerance or accepting a truncated basis as a cosmetic repair.

### Model sample

1. The module docstring meets the one-line rule. `build_model` should have a one-line action docstring; move any indispensable architecture explanation to a short comment. Its proposed four-line docstring is too long and omits the actual OBC subtraction while mentioning an OBC arm.
2. The separate physical and added-state updates are structurally right: `x_tilde_next = f_base + f_aug`, `x_bar_next = g_aug`. They describe normalised quantities here, so use `x_tilde^n`, augmented `x_hat^n`, `u^n`, `f_base^n`, `f_aug^n`, `g_aug^n` and `C^n`, with output `y_hat^n`. The actual plain-symbol `y_k = C x_tilde_k` looks like physical units and omits that these are predicted outputs. The configured D matrix is zero.
3. The connection comment adding a six-dimensional baseline directly to full augmented x is dimensionally incomplete. Use the two row-wise updates, or explicitly embed the baseline as `[f_base^n; 0]` and the network as `[f_aug^n; g_aug^n]`. Include `-f_obc^n` on physical rows for the OBC arm. `phi_aug` is the whole network, not just its physical correction.
4. The comment on `PHY_IX` identifies the physical-state rows correctly, but they contain normalised positions and velocities. Do not equate those stored values directly with unnormalised `(q, qdot)`.
5. The history statement is correct: model-level `na = nb = 30` uses the 30 samples before the current sample; the encoder constructor receives 29 because it reads n+1. In the thesis that is `n+1`, not the added-state count `n_a`. Remove the equation number and repeated Hoekstra name; one attribution for this encoder method is sufficient.
6. The encoder identity `psi = [W^b; W^a] chi + psi_tilde(chi)` matches the thesis. The linear term combines output and input histories in that order for the thesis chi; the actual nonlinear network is fed input history then output history. This is a fixed permutation absorbed into its function, not a reason to claim the network's literal input tensor is chi in thesis order.
7. The statement that the initial physical correction is zero is correct for freshly built models: the network's physical output weights and final bias start at zero. Added-state transition weights remain live. Describe that current mechanism once rather than adding explanatory comments at every constructor argument.
8. Starting from `q_0 = P^-T y_0`, zero physical velocities and zero added states is correct. Code uses row batches, hence multiplication by `P_INV_T64.T`. Position conversion occurs before physical-state normalisation. The recorded initial y is not `norm.y0`, despite their similar spelling.
9. The recorder comment correctly identifies the actual closed-loop input `u + u_fb`; use hats/superscript n if giving the thesis formula. The regression windows correctly take simulated input/output at s through s+na-1 and target the physical state at s+na because na equals nb here. Use normalised target notation and `max(na, nb)` when stating the general implementation.
10. The least-squares formula is correct as an affine fit at initialisation: `x_tilde^{n,0} approximately W^{b,0} chi^0 + b^{b,0}`. It is not a claim that every target is exactly reproduced. Returned W has shape regressors by physical states, transposed into the encoder parameters; distinguish the returned array from the thesis matrix W^b.
11. Standardising regressors and mapping back is described correctly. `rcond = eps max(M,N)` matches the code, but the solve uses centred, standardised Phi. The claim that gelsd is repeatable to the last bit on the CPU while gelsy is not is too absolute; this code does not establish a platform-independent guarantee. Say `# use the SVD driver for the rank-truncated least-squares fit` if a comment is needed.
12. Zeroing physical output-layer weights and putting b in its bias correctly describes the effective result; the code subtracts the encoder's zero-input output when setting that bias. `unexplained = 1 - R^2` is correct per state for nonzero target variance, since it divides mean squared residual by population variance. The acceptance threshold is `1e-3`, or `R^2 >= 0.999`.
13. The proposed heuristic comment retains experimental history about earlier starts and their residuals. Remove it. A sufficient comment is `# HEURISTIC: require R^2 >= 0.999 for every physical state on the simulated encoder-fit windows`. Keep the actual threshold.
14. The fixed-output-map and same-length-grouping comments mostly restate their code. The non-obvious zero initial conditions and encoder sample-count convention merit comments; routine stacking/grouping generally does not.
15. The CUDA message has an actionable remedy, but the Adam-eps message still lacks one. Ask the reader to make optimizer construction use the configured eps. The literal CLI option spellings in the plan are flags, not prose dashes; preserve real CLI syntax in any eventual code edit.

### Syntax-tree and string claim

I reconstructed both proposed files in memory from the fenced diffs, matching each hunk's old/context lines against the current file. All nine OBC hunks and eight model hunks applied uniquely. Both reconstructed files parse, as do all 24 current Clean Python files.

With docstrings removed and the listed identifier renames normalised, the changes outside error arguments are identical ASTs. Exactly four raised error-message expressions change: two rank-loss errors, CUDA availability and Adam eps. No dictionary key or ordinary data string changes in these samples.

The literal claim about setting aside strings needs precision. Replacing string constants with placeholders gives equal OBC ASTs, but unequal model ASTs: the Adam message has an extra literal segment and reverses the order of its two formatted expressions. This is still confined to formatting that error. Replacing complete raised-message arguments with placeholders gives equal trees for both files; inspection of the four arguments confirms only message construction changed. An implementation check should explicitly allow these four messages, not discard every string across the folder, since that would conceal forbidden key or event changes. These are static findings, not numerical equivalence tests.

## 7. Three open decisions

1. Name Kessels once, but attribute the known-controller rollout idea, not the supplied controller itself or our complete residual formulation. The controller comes from the repository's design recipe. A concise module description can say `Closed-loop model simulation with a known controller, following Kessels.` Keep the residual equations and zero controller-difference initial state clear at the rollout; do not imply Kessels supplied this exact initialisation. Retain any mandatory local theory labels without repeating full paper references or equation numbers.
2. Move the gantry constants docstring before `__project_origin__`. This deliberately changes `__doc__` and statement order, not model calculations. Use one or two lines and retain the immutable MATLAB-source provenance. It needs an explicit exception in the AST acceptance check, not a claim of identical trees.
3. Add the seven missing `@added` method decorators listed in marker item 4. This is the existing repository rule, not a numerical feature decision. Check the decorated objects remain callable and retain the existing decorator on `_propagate`; no training run is needed for this review.

## 8. Corrections required before approving the plan

The naming table needs the R1, R2, R4, R8 and R12 revisions above, consistent spelling for all our classes, and the additional own-name inventory. Keep the external evaluation surface fixed.

Correct the framework-file and theory/heuristic counts, retain provenance for changed import groups, and count seven new decorators. Replace the sample's unscaled SVD identity and the false literal identification of the affine basis/coefficient with the thesis symbols. Tighten public docstrings, remove remaining equation references, repeated attribution and experimental history, and make every revised error actionable. Qualify the AST claim as stated above and protect all data-bearing strings during the eventual check.

The plan's basic rename-safety conclusion survives the audit. The naming choices and mathematical comments need revision before they become the template for the rest of Clean.
