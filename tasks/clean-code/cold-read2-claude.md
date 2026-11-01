# Cold read 2 (Claude): comment understandability of the grey-box files in `Clean/`

Sources used: only `Clean/` (the 24 grey-box `.py` files and `Clean/README.md`). A question is listed when the answer
needs another file, the thesis or the history; questions are asked once, at the first place they arise.

## Counts per file

| File | Questions | Noise lines |
|-|-|-|
| Clean/gantry_interconnect_dynamic.py | 8 | 3 |
| Clean/gantry_dynamic/__init__.py | 0 | 0 |
| Clean/gantry_dynamic/config.py | 12 | 11 |
| Clean/gantry_dynamic/controller.py | 4 | 6 |
| Clean/gantry_dynamic/data.py | 3 | 4 |
| Clean/gantry_dynamic/evaluation.py | 5 | 0 |
| Clean/gantry_dynamic/model.py | 5 | 0 |
| Clean/gantry_dynamic/obc_gantry.py | 4 | 2 |
| Clean/gantry_dynamic/training.py | 5 | 2 |
| Clean/model_augmentation/__init__.py | 0 | 0 |
| Clean/model_augmentation/fit_systems/__init__.py | 0 | 0 |
| Clean/model_augmentation/fit_systems/blocks.py | 1 | 24 |
| Clean/model_augmentation/fit_systems/closed_loop.py | 0 | 3 |
| Clean/model_augmentation/fit_systems/interconnect.py | 2 | 11 |
| Clean/model_augmentation/fit_systems/lbfgs_polish.py | 1 | 6 |
| Clean/model_augmentation/fit_systems/obc.py | 0 | 7 |
| Clean/model_augmentation/fit_systems/pre_encoder.py | 0 | 4 |
| Clean/model_augmentation/fit_systems/ps2_opening.py | 3 | 6 |
| Clean/model_augmentation/systems/__init__.py | 0 | 0 |
| Clean/model_augmentation/systems/gantry_ss.py | 0 | 3 |
| Clean/model_augmentation/utils/__init__.py | 0 | 0 |
| Clean/model_augmentation/utils/deepSI_corrections.py | 0 | 4 |
| Clean/model_augmentation/utils/torch_nets.py | 0 | 4 |
| Clean/model_augmentation/utils/utils.py | 0 | 3 |
| **Total** | **53** | **103** |

## List A: questions

```
Q001 | Clean/gantry_interconnect_dynamic.py:40 | step | the "pre-polish pair": what do the .pt and the .npz file each hold, and why does a polish-only rerun need both?
Q002 | Clean/gantry_interconnect_dynamic.py:69 | variable | code='clean' in the RESUMES entry: what does 'clean' mean, and what other values can this field take?
Q003 | Clean/gantry_interconnect_dynamic.py:118 | parameter | e_train, e_val: what are these (servo error y - r?), in which unit and at which rate?
Q004 | Clean/gantry_interconnect_dynamic.py:122 | variable | "the controller bank": what is the bank, one controller per record or per design, and what does it hold?
Q005 | Clean/gantry_interconnect_dynamic.py:128 | return | load_checkpoint returns meta: which keys does it hold, and which of them does train use?
Q006 | Clean/gantry_interconnect_dynamic.py:133 | return | bestfit / SCORE_NAME: which score is this (on which split, normalised how), and is lower better?
Q007 | Clean/gantry_interconnect_dynamic.py:154 | step | a resume carries "neither the batch stream nor an unfinished PS2 opening": does a resumed run then restart the opening, skip it, or refuse to start?
Q008 | Clean/gantry_interconnect_dynamic.py:158 | variable | "A_combined, w = Y z": what are A_combined, w and z, and is Y here the head position or a matrix?
Q009 | Clean/gantry_dynamic/config.py:34 | setting | PRESETS jumps from R7 to R13: what are R8 to R12, and why are they not presets here?
Q010 | Clean/gantry_dynamic/config.py:70 | setting | arm 'obc': what is beta (its relation to the physical parameters, its unit), what is f_obc^n, and what does "take over a change of beta" mean?
Q011 | Clean/gantry_dynamic/config.py:73 | setting | start 'detuned': the "ten combinations" of which physical parameters, and what do c_g1, c_g2, m_Sigma, m_Delta, J_eff and d (line 208) stand for?
Q012 | Clean/gantry_dynamic/config.py:76 | setting | n_a: why does __post_init__ (line 164) allow only 0, 2 and 8; does another value break something or is it only untested?
Q013 | Clean/gantry_dynamic/config.py:78 | setting | ps2: what are "PS2", "S1" and "S2", and what happens with ps2=True and n_a=0, which the check admits?
Q014 | Clean/gantry_dynamic/config.py:91 | setting | epochs: with n_its None, does training stop exactly after epochs times the updates per epoch, and is a last interval shorter than its_per_val validated?
Q015 | Clean/gantry_dynamic/config.py:128 | setting | na_nb: "the n + 1 past samples": what is n, and is the encoder window na_nb or na_nb + 1 samples long?
Q016 | Clean/gantry_dynamic/config.py:130 | setting | "the R^2 check of the encoder start": what threshold, and where is it applied?
Q017 | Clean/gantry_dynamic/config.py:148 | setting | lbfgs_tol_grad, lbfgs_tol_change "absolute on V": what is the typical scale of V, so I can tell when they need rescaling?
Q018 | Clean/gantry_dynamic/config.py:159 | setting | dtype_np, dtype_pt: can a run be made in float32, or does other code assume float64?
Q019 | Clean/gantry_dynamic/config.py:180 | step | CODE != 'clean': which code wrote the "other folders", and where is clean-up-verification/convert_run.py (it is not in Clean/)?
Q020 | Clean/gantry_dynamic/config.py:186 | step | why is config_json_dict passed through json.dumps and json.loads before the comparison?
Q021 | Clean/gantry_dynamic/controller.py:31 | variable | P "stage-to-generalised transform": which quantities does P map and in which direction (forces u = P u_act, positions y = P^T q?), given line 75 uses P^T G P?
Q022 | Clean/gantry_dynamic/controller.py:201 | step | why is the loss weighted by W_r = diag(sigma_y / sigma_e,r) per record, and not equally over records?
Q023 | Clean/gantry_dynamic/controller.py:227 | variable | bank.M_state: what is this matrix, and why is its first dimension the number of distinct controllers?
Q024 | Clean/gantry_dynamic/controller.py:241 | variable | k0 = max(fs.na, fs.nb) "= n + 1": are fs.na and fs.nb the window length na_nb or na_nb - 1?
Q025 | Clean/gantry_dynamic/data.py:26 | variable | test-I, test-E: interpolation and extrapolation in what (Y position, motion profile, amplitude), and what does "one test split scored per record" mean?
Q026 | Clean/gantry_dynamic/data.py:79 | variable | "which the closed loop needs (output_only)": what is output_only and why does it need D = 0?
Q027 | Clean/gantry_dynamic/data.py:96 | variable | u_total: what does it contain (feedback force, feedforward, injected multisine, their sum)?
Q028 | Clean/gantry_dynamic/evaluation.py:43 | variable | "the network gate": what is it, is it trained, and what value does it have in training?
Q029 | Clean/gantry_dynamic/evaluation.py:44 | step | why do the two baseline-only models keep the augmented model's trained encoder for their initial state, and does that favour or penalise them?
Q030 | Clean/gantry_dynamic/evaluation.py:69 | variable | "(float32-rounded)": why are the true parameters float32-rounded in a float64 run, and how large is the effect on the "true" baseline?
Q031 | Clean/gantry_dynamic/evaluation.py:121 | return | closed_loop_free_run_rms_batch: run[0] and run[2] are used; what is run[1]?
Q032 | Clean/gantry_dynamic/evaluation.py:282 | variable | records = file name up to the first '_': what does a record name encode (e.g. the 'TR-S' class in ps2_opening)?
Q033 | Clean/gantry_dynamic/model.py:52 | variable | "m_b and J_b : J_h for the gauge section": what is the gauge section, and why does it need m_b and the ratio J_b : J_h?
Q034 | Clean/gantry_dynamic/model.py:86 | variable | signal names "x", "u", "xp", "y": is "xp" the next state, and is "y" the model output y_hat^n?
Q035 | Clean/gantry_dynamic/model.py:124 | step | "the network block and the output map are built in float32": why, and does anything else stay float32 after .to(cfg.dtype_pt)?
Q036 | Clean/gantry_dynamic/model.py:170 | variable | pass_mats and obc_pass, "the matrices of one objective": what matrices, and what is an "objective"?
Q037 | Clean/gantry_dynamic/model.py:185 | variable | "x^c_0 = 0 in the rollout": what is x^c?
Q038 | Clean/gantry_dynamic/obc_gantry.py:9 | step | beta_lin "renewed at the first objective of every epoch": what counts as an objective (one loss evaluation, a validation)?
Q039 | Clean/gantry_dynamic/obc_gantry.py:94 | step | the stencil gives qdot at 2 .. N - 3, but the tuples use i = 2 .. N - 4: why is the last sample dropped?
Q040 | Clean/gantry_dynamic/obc_gantry.py:68 | variable | "(Euclidean, all rows equal)": does "all rows equal" mean every state row and tuple has weight 1?
Q041 | Clean/gantry_dynamic/obc_gantry.py:266 | setting | "space=affine, refresh=epoch": are these fixed, or settable somewhere, and what are the alternatives?
Q042 | Clean/gantry_dynamic/training.py:89 | variable | coordinate_version: which values exist and which map beta -> theta_base does each name?
Q043 | Clean/gantry_dynamic/training.py:241 | step | "selection restarts in this job": after a resume, can the job end with a model worse than the previous job's best, since bestfit is reset to inf?
Q044 | Clean/gantry_dynamic/training.py:250 | variable | "decoder p", "theta^a_enc" and "encoder pairs": what are they?
Q045 | Clean/gantry_dynamic/training.py:300 | variable | fit_sys.unique_code: what reads it (the deepSI dump names?)
Q046 | Clean/gantry_dynamic/training.py:325 | step | "after an n_its cap or an early stop": which early stop exists, given config.py:97 says there is no stopping?
Q047 | Clean/model_augmentation/fit_systems/blocks.py:12 | variable | "one evaluation of V": what is V (defined only in interconnect.py)?
Q048 | Clean/model_augmentation/fit_systems/interconnect.py:384 | variable | "(N, n + 1, 3)": what is n here, given SSE_Interconnect gets na = na_nb while the encoder gets na - 1?
Q049 | Clean/model_augmentation/fit_systems/interconnect.py:490 | step | on a NaN training loss the loop breaks: what is then selected and saved, and does training go on to the polish?
Q050 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:156 | return | n_closure and func_evals: how do they differ (probe calls, line-search evaluations)?
Q051 | Clean/model_augmentation/fit_systems/ps2_opening.py:58 | variable | "excitation class": which classes do the record names encode, and which records fall in each?
Q052 | Clean/model_augmentation/fit_systems/ps2_opening.py:168 | step | S1 updates W^a and psi_tilde's output layer with its own Adam while the main Adam updates the same tensors: why a separate optimizer, and are the main Adam moments affected?
Q053 | Clean/model_augmentation/fit_systems/ps2_opening.py:279 | variable | m in the S2 cost "(xbar_{k+1|k+1} - m)^2": the mean of the target over the batch, per state?
```

## List B: noise

```
N001 | Clean/gantry_interconnect_dynamic.py:99 | repeats-code | "2. Data and normalisation." names the two calls below
N002 | Clean/gantry_interconnect_dynamic.py:111 | repeats-code | "3. Model and closed loop." names the calls below
N003 | Clean/gantry_interconnect_dynamic.py:131 | repeats-code | "5. Training." before train()
N004 | Clean/gantry_dynamic/config.py:91 | empty-tag | "HEURISTIC: thesis budget" says where 200 came from, not why
N005 | Clean/gantry_dynamic/config.py:97 | empty-tag | "HEURISTIC: thesis value" tells nothing about 1300
N006 | Clean/gantry_dynamic/config.py:102 | empty-tag | "HEURISTIC: thesis budget" tells nothing about the L-BFGS values
N007 | Clean/gantry_dynamic/config.py:126 | empty-tag | "HEURISTIC: thesis value" tells nothing about 4 kHz
N008 | Clean/gantry_dynamic/config.py:131 | empty-tag | "HEURISTIC: thesis values" tells nothing about stride, nf_seconds, na_nb
N009 | Clean/gantry_dynamic/config.py:136 | empty-tag | "HEURISTIC: thesis values" tells nothing about the network size
N010 | Clean/gantry_dynamic/config.py:142 | empty-tag | "HEURISTIC: thesis values" (the eps parenthesis is informative, the tag is not)
N011 | Clean/gantry_dynamic/config.py:149 | empty-tag | "HEURISTIC: fixed values" says nothing about the tolerances
N012 | Clean/gantry_dynamic/config.py:196 | repeats-code | "True on the OBC arm." restates return self.arm == 'obc'
N013 | Clean/gantry_dynamic/config.py:213 | empty-tag | "HEURISTIC: 10 % detuning" restates 0.10, gives no reason
N014 | Clean/gantry_dynamic/config.py:242 | repeats-code | "stdout of one git command in the repository." restates the subprocess call
N015 | Clean/gantry_dynamic/controller.py:71 | repeats-code | A, B formula restated on the two lines below
N016 | Clean/gantry_dynamic/controller.py:87 | repeats-doc | numerator and denominator already in the docstring (line 81)
N017 | Clean/gantry_dynamic/controller.py:111 | repeats-doc | "2. Gain kappa_j." already defined in the docstring (line 96)
N018 | Clean/gantry_dynamic/controller.py:114 | repeats-code | "3. Tustin at ts." above cont2discrete(..., 'bilinear')
N019 | Clean/gantry_dynamic/controller.py:230 | repeats-code | "2. Compiled step." above torch.compile, explained by line 231
N020 | Clean/gantry_dynamic/controller.py:243 | repeats-doc | W_r formula already in the docstring (line 201)
N021 | Clean/gantry_dynamic/data.py:134 | repeats-doc | error-only filtering already in the docstring (lines 121 to 123)
N022 | Clean/gantry_dynamic/data.py:175 | repeats-doc | sigma_e formula already in the docstring (line 165)
N023 | Clean/gantry_dynamic/data.py:234 | repeats-doc | q = P^-T y and the mean relation already in the docstring (lines 216, 217)
N024 | Clean/gantry_dynamic/data.py:253 | repeats-doc | C^n formula and no offset already in the docstring (lines 216, 217)
N025 | Clean/gantry_dynamic/obc_gantry.py:186 | repeats-doc | f_obc^n formula already in the docstring (line 172)
N026 | Clean/gantry_dynamic/obc_gantry.py:218 | repeats-doc | J formula already in the inner docstring (line 213)
N027 | Clean/gantry_dynamic/training.py:162 | repeats-doc | "An empty history, written to hist_path at every entry." repeats the class docstring (line 159)
N028 | Clean/gantry_dynamic/training.py:303 | repeats-doc | resume = (path, meta) already in the docstring (line 289)
N029 | Clean/model_augmentation/fit_systems/blocks.py:14 | history | "restricted to the thesis training path; Jan's full code ... commit 8200923" is provenance (lines 14, 15)
N030 | Clean/model_augmentation/fit_systems/blocks.py:19 | history | "unused imports removed" describes an earlier version
N031 | Clean/model_augmentation/fit_systems/blocks.py:28 | repeats-code | "Store the block input size nz and output size nw." restates the two assignments
N032 | Clean/model_augmentation/fit_systems/blocks.py:38 | repeats-doc | restates the class docstring (line 36)
N033 | Clean/model_augmentation/fit_systems/blocks.py:87 | history | "no empty-matrix cases" describes what Jan's version did
N034 | Clean/model_augmentation/fit_systems/blocks.py:105 | repeats-doc | output map with D = 0 and the controller reason already in the class docstring (lines 85, 86)
N035 | Clean/model_augmentation/fit_systems/blocks.py:113 | history | "CHANGED: no size assertion" says what was removed
N036 | Clean/model_augmentation/fit_systems/blocks.py:180 | repeats-doc | RK4 stages restated from the docstring (line 168) and the code
N037 | Clean/model_augmentation/fit_systems/blocks.py:204 | repeats-code | denormalisation formula restated by the two lines below
N038 | Clean/model_augmentation/fit_systems/blocks.py:212 | repeats-code | "Generalised force: u = P u_act." above u2 @ P_mat.T (also line 155)
N039 | Clean/model_augmentation/fit_systems/blocks.py:215 | repeats-code | fnet formula restated by the lines below
N040 | Clean/model_augmentation/fit_systems/blocks.py:248 | repeats-code | "Normalise: xdot^n = xdot / sigma_x." above the division
N041 | Clean/model_augmentation/fit_systems/blocks.py:295 | repeats-code | "s_d = dm_h poly + m_h s_poly." restates line 297
N042 | Clean/model_augmentation/fit_systems/blocks.py:316 | repeats-code | closed-form tangent restated by the two lines below
N043 | Clean/model_augmentation/fit_systems/blocks.py:333 | repeats-code | normalisation restated by the two lines below
N044 | Clean/model_augmentation/fit_systems/blocks.py:353 | repeats-doc | stage formula already in the docstring (line 341)
N045 | Clean/model_augmentation/fit_systems/blocks.py:363 | repeats-code | tangent update restated by line 364
N046 | Clean/model_augmentation/fit_systems/blocks.py:403 | repeats-doc | K = diag(0, k_b,Sigma, 0) already in the docstring (line 390)
N047 | Clean/model_augmentation/fit_systems/blocks.py:428 | repeats-doc | M_1 already in the docstring (line 420)
N048 | Clean/model_augmentation/fit_systems/blocks.py:434 | repeats-doc | M_2 already in the docstring (line 420)
N049 | Clean/model_augmentation/fit_systems/blocks.py:577 | repeats-doc | equal splits and the J ratio already in the docstring (line 558) and line 572 (lines 577, 578)
N050 | Clean/model_augmentation/fit_systems/blocks.py:610 | repeats-doc | exp map already in the docstring (line 601)
N051 | Clean/model_augmentation/fit_systems/blocks.py:612 | repeats-doc | tanh map already in the docstring (line 601)
N052 | Clean/model_augmentation/fit_systems/blocks.py:640 | repeats-doc | inverse map already in the docstring (line 620)
N053 | Clean/model_augmentation/fit_systems/closed_loop.py:70 | repeats-doc | the one-product step already stated at line 52 and in the docstring (line 59)
N054 | Clean/model_augmentation/fit_systems/closed_loop.py:94 | repeats-doc | "Output: y_hat^n = C^n x_tilde^n." already in the docstring (line 91)
N055 | Clean/model_augmentation/fit_systems/closed_loop.py:98 | repeats-doc | u_hat^n = u^n + u_fb^n already in the docstring (line 91)
N056 | Clean/model_augmentation/fit_systems/interconnect.py:2 | history | "restricted to the thesis training path; Jan's full code ... commit 8200923" is provenance (lines 2, 3)
N057 | Clean/model_augmentation/fit_systems/interconnect.py:9 | history | which imports were added for the changed fit and which are "as in Jan's" (lines 9, 10)
N058 | Clean/model_augmentation/fit_systems/interconnect.py:18 | history | "instead of the parameterised blocks and detect_algebraic_loop" describes the replaced imports (lines 18, 19)
N059 | Clean/model_augmentation/fit_systems/interconnect.py:27 | repeats-code | "An empty interconnect with state size nx, ..." restates the assignments
N060 | Clean/model_augmentation/fit_systems/interconnect.py:173 | repeats-code | "the connection matrices on the module's device and dtype" restates the .to(...) below
N061 | Clean/model_augmentation/fit_systems/interconnect.py:305 | repeats-code | "Object to hold information regarding connection between two signals" restates the class name
N062 | Clean/model_augmentation/fit_systems/interconnect.py:307 | repeats-code | "Store the two signal indices, the method and the connection matrix." restates the assignments
N063 | Clean/model_augmentation/fit_systems/interconnect.py:397 | history | the list of arguments removed from Jan's fit (lines 397, 398)
N064 | Clean/model_augmentation/fit_systems/interconnect.py:442 | history | "(Jan's continues an earlier fit)" describes the original
N065 | Clean/model_augmentation/fit_systems/interconnect.py:461 | repeats-doc | why DtypeDataLoader exists is already in its docstring (line 331)
N066 | Clean/model_augmentation/fit_systems/interconnect.py:638 | repeats-doc | V formula already in the docstring (line 607)
N067 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:45 | repeats-doc | restates the class docstring (lines 40 to 42)
N068 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:108 | repeats-code | "Recognise CUDA OOM exceptions and RuntimeErrors containing 'out of memory'." restates the return line
N069 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:184 | repeats-code | "Store the L-BFGS iteration and closure counts in out." restates the two lines
N070 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:212 | repeats-code | "A fresh L-BFGS optimizer with the spec's settings, strong-Wolfe line search." restates the constructor
N071 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:263 | repeats-doc | the ceil(max_iter / inner_iter) blocks and early stop already in PolishSpec (lines 17 to 19)
N072 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:302 | repeats-doc | acceptance rule already in _acceptance's docstring (line 130)
N073 | Clean/model_augmentation/fit_systems/obc.py:36 | repeats-doc | column scale D already in the docstring (line 26)
N074 | Clean/model_augmentation/fit_systems/obc.py:41 | repeats-doc | "J D^-1 = U S V^T." already in the docstring (line 26)
N075 | Clean/model_augmentation/fit_systems/obc.py:67 | repeats-doc | theta_aux formula already in the docstring (line 58)
N076 | Clean/model_augmentation/fit_systems/obc.py:72 | repeats-code | "Move the factors to device." restates the method body
N077 | Clean/model_augmentation/fit_systems/obc.py:80 | repeats-code | "Device of the factors." restates return self.U.device
N078 | Clean/model_augmentation/fit_systems/obc.py:115 | repeats-doc | varphi definition already in the docstring (line 89)
N079 | Clean/model_augmentation/fit_systems/obc.py:123 | repeats-doc | J = [Phi col(step)] already in the docstring (line 99)
N080 | Clean/model_augmentation/fit_systems/pre_encoder.py:6 | history | "restricted to the thesis training path; Jan's full code ... commit 8200923" is provenance (lines 6, 7)
N081 | Clean/model_augmentation/fit_systems/pre_encoder.py:12 | history | "CHANGED: the @added marker and the layer stack of the added encoder" describes the import change
N082 | Clean/model_augmentation/fit_systems/pre_encoder.py:34 | repeats-doc | psi formula already in the class docstring (line 19)
N083 | Clean/model_augmentation/fit_systems/pre_encoder.py:96 | repeats-doc | "Linear part: (W^b chi, W^a chi)." already in the docstring (line 84)
N084 | Clean/model_augmentation/fit_systems/ps2_opening.py:101 | repeats-code | "True once the opening has ended." restates return self.phase == 'S3'
N085 | Clean/model_augmentation/fit_systems/ps2_opening.py:224 | repeats-doc | residual over H samples already in the docstring (line 213)
N086 | Clean/model_augmentation/fit_systems/ps2_opening.py:226 | repeats-code | "Scale: e_tau / sigma_e." above the division
N087 | Clean/model_augmentation/fit_systems/ps2_opening.py:248 | repeats-doc | S1 cost already in the docstring (line 236)
N088 | Clean/model_augmentation/fit_systems/ps2_opening.py:250 | repeats-code | "g = grad_(theta_p, theta^a_enc) V_S1." restates torch.autograd.grad below
N089 | Clean/model_augmentation/fit_systems/ps2_opening.py:333 | repeats-code | "3. Restore the best parameters." above the copy loop
N090 | Clean/model_augmentation/systems/gantry_ss.py:70 | repeats-doc | alpha, beta, gamma already in the module docstring (line 2) and the Returns (line 67)
N091 | Clean/model_augmentation/systems/gantry_ss.py:116 | repeats-doc | R = M_0^-1 = N0 / d0 already in the docstring (line 99)
N092 | Clean/model_augmentation/systems/gantry_ss.py:123 | repeats-doc | A_x, B_w, B_u already in the docstring (line 99)
N093 | Clean/model_augmentation/utils/deepSI_corrections.py:5 | history | Jan's class names and the restriction to the thesis path (lines 5 to 7)
N094 | Clean/model_augmentation/utils/deepSI_corrections.py:54 | repeats-code | "Store mu_u, sigma_u, mu_y, sigma_y as u0, ustd, y0, ystd." restates the assignments and line 3
N095 | Clean/model_augmentation/utils/deepSI_corrections.py:72 | history | "CHANGED: lists handled record by record; errors name the expected input" describes the change, not the code
N096 | Clean/model_augmentation/utils/deepSI_corrections.py:78 | repeats-doc | normalisation formula already in the docstring (line 63)
N097 | Clean/model_augmentation/utils/torch_nets.py:6 | history | "restricted to the thesis training path; Jan's full code ... commit 8200923" is provenance (lines 6, 7)
N098 | Clean/model_augmentation/utils/torch_nets.py:11 | history | "CHANGED: the marker of the added definitions" on an import
N099 | Clean/model_augmentation/utils/torch_nets.py:37 | repeats-doc | "b_L = 0." already in the docstring (line 19)
N100 | Clean/model_augmentation/utils/torch_nets.py:62 | repeats-doc | f_aug^n = 0 at the start already in the class docstring (line 44) and line 50
N101 | Clean/model_augmentation/utils/utils.py:2 | history | "restricted to the thesis training path; Jan's full code ... commit 8200923" is provenance (lines 2, 3)
N102 | Clean/model_augmentation/utils/utils.py:14 | repeats-code | "input as a tensor; a tensor is returned as it is." restates the one-line body
N103 | Clean/model_augmentation/utils/utils.py:15 | history | "None is no longer passed through (no caller passes it)" describes the earlier behaviour
```
