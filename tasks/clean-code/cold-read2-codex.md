| File | Questions | Noise lines |
|---|---:|---:|
| Clean/gantry_interconnect_dynamic.py | 0 | 5 |
| Clean/gantry_dynamic/__init__.py | 0 | 0 |
| Clean/gantry_dynamic/config.py | 7 | 8 |
| Clean/gantry_dynamic/controller.py | 2 | 15 |
| Clean/gantry_dynamic/data.py | 6 | 12 |
| Clean/gantry_dynamic/evaluation.py | 11 | 12 |
| Clean/gantry_dynamic/model.py | 6 | 14 |
| Clean/gantry_dynamic/obc_gantry.py | 7 | 19 |
| Clean/gantry_dynamic/training.py | 8 | 9 |
| Clean/model_augmentation/__init__.py | 0 | 3 |
| Clean/model_augmentation/fit_systems/__init__.py | 0 | 0 |
| Clean/model_augmentation/fit_systems/blocks.py | 7 | 66 |
| Clean/model_augmentation/fit_systems/closed_loop.py | 5 | 21 |
| Clean/model_augmentation/fit_systems/interconnect.py | 11 | 38 |
| Clean/model_augmentation/fit_systems/lbfgs_polish.py | 8 | 16 |
| Clean/model_augmentation/fit_systems/obc.py | 4 | 23 |
| Clean/model_augmentation/fit_systems/pre_encoder.py | 3 | 11 |
| Clean/model_augmentation/fit_systems/ps2_opening.py | 20 | 26 |
| Clean/model_augmentation/systems/__init__.py | 0 | 0 |
| Clean/model_augmentation/systems/gantry_ss.py | 4 | 5 |
| Clean/model_augmentation/utils/__init__.py | 0 | 0 |
| Clean/model_augmentation/utils/deepSI_corrections.py | 3 | 8 |
| Clean/model_augmentation/utils/torch_nets.py | 2 | 9 |
| Clean/model_augmentation/utils/utils.py | 3 | 7 |
| **Total** | **117** | **327** |

## List A, questions

```text
Q001 | Clean/gantry_dynamic/config.py:95 | setting | n_its: is this a fresh update budget or a cumulative budget on a resume, and what do zero and negative values mean?
Q002 | Clean/gantry_dynamic/config.py:103 | setting | lbfgs_max_iter and lbfgs_inner_iter: what values are allowed, what disables polishing, and how is a total budget that is not divisible by the inner budget handled?
Q003 | Clean/gantry_dynamic/config.py:105 | setting | lbfgs_windows: must this count be strictly positive, and is there a special value for using every eligible window?
Q004 | Clean/gantry_dynamic/config.py:117 | setting | seed: which sources of randomness remain uncontrolled by this value, and should repeating a run on the same device reproduce its weights and scores exactly?
Q005 | Clean/gantry_dynamic/config.py:134 | setting | na_nb: why is the encoder history fixed at 30 samples, and how should this duration be chosen for a different stage or sampling rate?
Q006 | Clean/gantry_dynamic/config.py:164 | setting | n_a: why are only 0, 2 and 8 supported?
Q007 | Clean/gantry_dynamic/config.py:164 | setting | ps2: is True meaningful when n_a is zero, and what does that combination request?
Q008 | Clean/gantry_dynamic/controller.py:46 | parameter | Y_op: what range of design positions is supported by the controller construction?
Q009 | Clean/gantry_dynamic/controller.py:129 | parameter | gain: must this output scale be positive, or do zero and negative values have a supported meaning?
Q010 | Clean/gantry_dynamic/data.py:25 | setting | DATASETS: what noise amplitude and spectrum, friction law parameters and hidden absorber dynamics distinguish these data sets from the rigid baseline?
Q011 | Clean/gantry_dynamic/data.py:34 | parameter | MANIFEST.csv: what do the record-name components and folder names mean, and which parts are relied on when identifying records?
Q012 | Clean/gantry_dynamic/data.py:47 | parameter | _register_controller: what exact MATLAB structure layout and scalar/string representation must meta.controller have?
Q013 | Clean/gantry_dynamic/data.py:138 | parameter | fields: which record fields can be requested here, and must all of them be sample-aligned arrays at fs_orig?
Q014 | Clean/gantry_dynamic/data.py:143 | step | Why are extra fields point-sampled while the force is block-averaged and y is filtered, and how should a caller choose the rule for another physical signal?
Q015 | Clean/gantry_dynamic/data.py:174 | step | Which Beintema et al. reference and definition justify using the standard deviation of the servo error as the NRMS denominator?
Q016 | Clean/gantry_dynamic/evaluation.py:44 | step | Why do both baseline comparisons retain the augmented model's trained encoder rather than use an encoder fitted to each baseline?
Q017 | Clean/gantry_dynamic/evaluation.py:69 | variable | The 'true' parameters are float32-rounded: are these exactly the parameters used to generate the records, or a rounded reference for comparison?
Q018 | Clean/gantry_dynamic/evaluation.py:106 | variable | floor: does this represent only measurement noise, or also differences between the independently closed-loop noisy and noise-free trajectories?
Q019 | Clean/gantry_dynamic/evaluation.py:138 | parameter | _table: what are the required structures and ordering of names, records and s?
Q020 | Clean/gantry_dynamic/evaluation.py:153 | parameter | _bar_figure: what keys, array shapes and record ordering must s contain?
Q021 | Clean/gantry_dynamic/evaluation.py:173 | parameter | _record_figure: what are the required units, frame, time origin and dictionary structure of residual, ref, nrms and floor?
Q022 | Clean/gantry_dynamic/evaluation.py:200 | parameter | _loss_figure: what keys and event values must loss and hist contain, including for a polish-only run?
Q023 | Clean/gantry_dynamic/evaluation.py:238 | return | evaluate: what are the exact keys, array shapes and units in each split and frame of the returned dictionary?
Q024 | Clean/gantry_dynamic/evaluation.py:282 | variable | records: which part of a record filename is intended to be its public identifier, and must that prefix be unique within a split?
Q025 | Clean/gantry_dynamic/evaluation.py:290 | variable | predictions.npz: which arrays include the encoder-history samples, which begin at k0, and which frame and units do the saved r, e, e_noise_free and e_hat arrays use?
Q026 | Clean/gantry_dynamic/evaluation.py:308 | step | Why are the per-frame residual arrays removed from results after they have been plotted?
Q027 | Clean/gantry_dynamic/model.py:83 | parameter | connect_signals: what are the source and destination meanings of x, u, xp and y in this wiring?
Q028 | Clean/gantry_dynamic/model.py:85 | step | Why does the augmentation correct the position rows as well as the velocity rows of the physical state?
Q029 | Clean/gantry_dynamic/model.py:101 | step | Why use both a linear history map and a nonlinear correction in the encoder rather than one network?
Q030 | Clean/gantry_dynamic/model.py:185 | step | Why is zero velocity a suitable initial condition after the record has already been trimmed at both ends?
Q031 | Clean/gantry_dynamic/model.py:252 | step | Why standardise the least-squares regressors a second time after the signals have already been normalised?
Q032 | Clean/gantry_dynamic/model.py:255 | step | Why use a rank-truncated SVD fit for the physical encoder, and which history directions is it intended to discard?
Q033 | Clean/gantry_dynamic/obc_gantry.py:49 | step | Why is fourth-order differentiation required for the reference tuples, and how does measurement noise affect these velocity estimates?
Q034 | Clean/gantry_dynamic/obc_gantry.py:68 | step | Why are all reference tuples evaluated at xbar = 0, rather than at encoder-estimated added states?
Q035 | Clean/gantry_dynamic/obc_gantry.py:68 | step | Why is the projection defined with equal weights on normalised physical-state rows rather than the servo-error weights used by the training objective?
Q036 | Clean/gantry_dynamic/obc_gantry.py:94 | step | Why is the last complete central-difference sample, at index N - 3, omitted from the reference set?
Q037 | Clean/gantry_dynamic/obc_gantry.py:129 | step | Why is the physical block stored in a one-element Python list rather than directly as an attribute of the correction module?
Q038 | Clean/gantry_dynamic/obc_gantry.py:172 | step | Why protect the baseline transition column in addition to its ten parameter-sensitivity columns?
Q039 | Clean/gantry_dynamic/obc_gantry.py:191 | step | Why must the basis use a separate float64 copy of the physical block when the configured training dtype is already float64?
Q040 | Clean/gantry_dynamic/training.py:33 | parameter | save_checkpoint_pair: what metadata must a caller supply for a pair to support loading, continuing training and evaluation?
Q041 | Clean/gantry_dynamic/training.py:73 | step | Why does restoring Adam moments retain the current run's hyperparameters rather than restore the checkpoint's parameter-group settings?
Q042 | Clean/gantry_dynamic/training.py:168 | variable | ValidationHistory events: what distinguishes adam, rebuilt, polish_candidate, selected and final, and what extra fields can each carry?
Q043 | Clean/gantry_dynamic/training.py:208 | step | Why must an OBC polish beat the old selected score rather than the validation score of the rebuilt-basis model it actually starts from?
Q044 | Clean/gantry_dynamic/training.py:230 | step | After a rejected polish restores the original weights, where do the candidate's per-channel validation values used for this history row come from?
Q045 | Clean/gantry_dynamic/training.py:249 | step | Why does done_epochs == 0 decide whether to start PS when no opening state was saved, rather than the number of completed Adam updates?
Q046 | Clean/gantry_dynamic/training.py:325 | step | Why does a checkpoint pair mark the entire epoch budget as completed after an update cap or an early stop?
Q047 | Clean/gantry_dynamic/training.py:353 | step | Why is Adam state omitted for a polish-only job even when polishing is rejected and the weights are restored?
Q048 | Clean/model_augmentation/fit_systems/blocks.py:25 | parameter | Block: is this a general interface for arbitrary signals, or must every subclass follow the (B, channels, 1) convention and discrete-time interpretation of this gantry model?
Q049 | Clean/model_augmentation/fit_systems/blocks.py:49 | parameter | net_kwargs: which keys must a supplied network constructor accept, and what forward input and output contract must that network satisfy?
Q050 | Clean/model_augmentation/fit_systems/blocks.py:145 | setting | closed_form: what numerical accuracy or performance benefit should be expected from switching off the LFR output equation?
Q051 | Clean/model_augmentation/fit_systems/blocks.py:509 | parameter | combo_init: what positivity, mass-difference and nonzero constraints must the starting combinations satisfy before the signed-coordinate calibration can be constructed?
Q052 | Clean/model_augmentation/fit_systems/blocks.py:523 | step | Why match the signed mass-difference coordinate's initial slope to m_Delta,0, and is a zero initial mass difference supported?
Q053 | Clean/model_augmentation/fit_systems/blocks.py:558 | step | Why is this particular gauge chosen, with equal joint splits and fixed m_b and J_b:J_h, rather than another raw-parameter representative?
Q054 | Clean/model_augmentation/fit_systems/blocks.py:559 | return | gauge_section: are the returned individual masses and inertias guaranteed to remain physically meaningful, or are only their combined inertia matrices constrained?
Q055 | Clean/model_augmentation/fit_systems/closed_loop.py:59 | variable | x^c: what physical assumption makes setting the controller-state difference to zero at each independently encoded training window suitable?
Q056 | Clean/model_augmentation/fit_systems/closed_loop.py:146 | parameter | ClosedLoopSimulator: what are the required shapes, ordering and units of train_weights and val_sigma_e, and must their record order match the corresponding controller-row lists?
Q057 | Clean/model_augmentation/fit_systems/closed_loop.py:207 | setting | The direct-call default nf = 25: why is this rollout length used instead of the run configuration's horizon?
Q058 | Clean/model_augmentation/fit_systems/closed_loop.py:218 | step | Why does content verification inspect only each record's first and last window before assigning the controller and loss weights to all its windows?
Q059 | Clean/model_augmentation/fit_systems/closed_loop.py:257 | parameter | closed_loop_free_run_rms_batch: must fit_sys already be in evaluation mode and have synchronised OBC coefficients before this function is called?
Q060 | Clean/model_augmentation/fit_systems/interconnect.py:99 | return | _propagate: how are the returned output_signals entries ordered, and which entries may be None when an output dependency cone is requested?
Q061 | Clean/model_augmentation/fit_systems/interconnect.py:123 | parameter | init_forward: what restrictions must the connected graph satisfy, including disconnected signals and cycles, before computation order can be resolved?
Q062 | Clean/model_augmentation/fit_systems/interconnect.py:212 | parameter | init_connection_matrices: how are concatenated and additive signals ordered and dimensioned when both target the same destination?
Q063 | Clean/model_augmentation/fit_systems/interconnect.py:265 | parameter | connect_signals: what are the supported input_signal and output_signal values and which source/destination uses of x, xp, u and y are meaningful?
Q064 | Clean/model_augmentation/fit_systems/interconnect.py:265 | parameter | connection_matrix: what does an empty matrix mean for each connection method, and what shapes must a supplied matrix have?
Q065 | Clean/model_augmentation/fit_systems/interconnect.py:277 | parameter | connect_block_signals: can the same source or destination occur more than once in a connection list, and how are repeated connections interpreted?
Q066 | Clean/model_augmentation/fit_systems/interconnect.py:368 | parameter | init_model: can sys_data actually be None, and which optimizer_kwargs are supported by the inherited optimizer factory?
Q067 | Clean/model_augmentation/fit_systems/interconnect.py:377 | parameter | simulate: can yfuture be None for the attached closed-loop simulator, and which Loss_kwargs are required?
Q068 | Clean/model_augmentation/fit_systems/interconnect.py:517 | step | Why rebuild the OBC basis after selecting and reloading the best checkpoint, even though doing so changes the selected model's score?
Q069 | Clean/model_augmentation/fit_systems/interconnect.py:625 | step | Why refresh the OBC basis once per epoch rather than every objective or when beta has moved by a threshold?
Q070 | Clean/model_augmentation/fit_systems/interconnect.py:636 | step | Why does Adam minimise the mean squared scaled window error while checkpoint selection uses the mean unsquared NRMS_e of complete records?
Q071 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:19 | setting | PolishSpec: which values of n_windows, max_iter and inner_iter disable the polish, and which must be strictly positive?
Q072 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:93 | step | Why does the polish closure zero existing gradient tensors rather than set gradients to None?
Q073 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:97 | return | The detached closure value: why is a new CPU tensor returned instead of a detached value on the model's device?
Q074 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:123 | step | Why must the determinism probe reproduce the loss exactly rather than within a numerical tolerance?
Q075 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:144 | step | Why use strong-Wolfe line search with a fixed learning rate of 1.0 for this polish?
Q076 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:151 | return | ran: does True require the iteration budget to finish, a successful optimizer step, or merely reaching final validation?
Q077 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:157 | variable | n_closure versus func_evals: which evaluations enter each count, including the determinism probes, retries and the final forward pass?
Q078 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:165 | step | Why keep the OBC expansion point fixed throughout polishing while the physical parameters continue changing?
Q079 | Clean/model_augmentation/fit_systems/obc.py:26 | step | Why scale the basis columns by their Euclidean norms before factorisation?
Q080 | Clean/model_augmentation/fit_systems/obc.py:27 | step | Why reject a rank-deficient basis instead of using its truncated-SVD projection onto the identifiable subspace?
Q081 | Clean/model_augmentation/fit_systems/obc.py:131 | setting | max_consecutive_rank_loss: why permit one failed refresh but abort after two consecutive failures?
Q082 | Clean/model_augmentation/fit_systems/obc.py:134 | parameter | reference_field and basis_builder: what callable signatures, row ordering, dtype, device and gradient contracts must their implementations satisfy?
Q083 | Clean/model_augmentation/fit_systems/pre_encoder.py:67 | step | Why initialise all added linear rows in one Xavier draw rather than initialise the input-history and output-history maps independently?
Q084 | Clean/model_augmentation/fit_systems/pre_encoder.py:76 | step | Why initialise the nonlinear added-state encoder rows to zero while keeping the linear added-state rows random?
Q085 | Clean/model_augmentation/fit_systems/pre_encoder.py:104 | step | Why does the nonlinear encoder read u history before y history while the linear encoder's chi convention is y history before u history?
Q086 | Clean/model_augmentation/fit_systems/ps2_opening.py:21 | setting | PS2Spec: which combinations of evaluation cadence, screen time, minimum counts, caps and plateau-window sizes are valid?
Q087 | Clean/model_augmentation/fit_systems/ps2_opening.py:24 | setting | aux_batch: are auxiliary windows drawn with replacement, and how does this batch size affect the residual normalisation and the S1 update?
Q088 | Clean/model_augmentation/fit_systems/ps2_opening.py:25 | setting | lr: does this single value control the decoder, the auxiliary encoder updates and the S2 network fit, and are the Adam epsilon and beta settings the same as the main optimizer's?
Q089 | Clean/model_augmentation/fit_systems/ps2_opening.py:33 | setting | screen_max: why is explaining 10 percent of the residual the threshold for continuing to S2?
Q090 | Clean/model_augmentation/fit_systems/ps2_opening.py:34 | setting | n_cal, n_snap and n_snap_cal: are these totals across their record subsets or counts per record, and can sampled windows or pairs repeat?
Q091 | Clean/model_augmentation/fit_systems/ps2_opening.py:50 | step | Why compare the best of the recent checks with the best of all preceding checks to detect a plateau?
Q092 | Clean/model_augmentation/fit_systems/ps2_opening.py:78 | parameter | PS2Opening: what restrictions on n_a, encoder history lengths and network output ordering must a caller satisfy?
Q093 | Clean/model_augmentation/fit_systems/ps2_opening.py:104 | step | Why place each S1 update after the main Adam update rather than complete predictive-state initialisation before fitting the main objective?
Q094 | Clean/model_augmentation/fit_systems/ps2_opening.py:157 | step | Why initialise the decoder's final weight and bias to zero?
Q095 | Clean/model_augmentation/fit_systems/ps2_opening.py:162 | setting | H = n + 1: why must the residual-prediction horizon equal the encoder history length, and are those durations intended to be changed together?
Q096 | Clean/model_augmentation/fit_systems/ps2_opening.py:168 | step | Why update only W^a and the encoder's final nonlinear layer in S1, keeping its hidden representation fixed?
Q097 | Clean/model_augmentation/fit_systems/ps2_opening.py:191 | step | Why reserve M + 1 future samples for S2 pairs that use only one next-step history, and exclude the remaining eligible samples from both draws?
Q098 | Clean/model_augmentation/fit_systems/ps2_opening.py:213 | step | Why divide the S1 residual by one RMS over all samples and axes rather than use the record-specific servo-error weighting of the main objective?
Q099 | Clean/model_augmentation/fit_systems/ps2_opening.py:227 | step | S1 residual scaling: must the current model's residual RMS be strictly positive, and what is the intended meaning of a zero-residual case?
Q100 | Clean/model_augmentation/fit_systems/ps2_opening.py:253 | step | Why keep separate Adam moment estimates for S1 and the main objective while they update shared encoder parameters?
Q101 | Clean/model_augmentation/fit_systems/ps2_opening.py:279 | step | Why divide the S2 error by the mean sample variance of the added-state targets, rather than normalise each learned coordinate separately?
Q102 | Clean/model_augmentation/fit_systems/ps2_opening.py:289 | step | Why allow shared hidden layers to change the physical network correction during S2, when the S2 loss only fits the added-state transition?
Q103 | Clean/model_augmentation/fit_systems/ps2_opening.py:298 | step | Why fit the S2 transition with the recorded force u_k when the main objective drives it with the closed-loop corrected force u_hat_k?
Q104 | Clean/model_augmentation/fit_systems/ps2_opening.py:311 | step | Why recompute target variance for every S2 minibatch rather than use a fixed variance from the full fitting set?
Q105 | Clean/model_augmentation/fit_systems/ps2_opening.py:321 | step | S2 variance scaling: must the added-state targets have a strictly positive mean variance in every minibatch?
Q106 | Clean/model_augmentation/systems/gantry_ss.py:10 | setting | _D: why are nominal parameters and geometry created in float32 before a float64 training run?
Q107 | Clean/model_augmentation/systems/gantry_ss.py:19 | variable | Jb and Jh: are these inertias about each component's own centre of mass or about the stage Theta axis at a particular head position?
Q108 | Clean/model_augmentation/systems/gantry_ss.py:36 | variable | d: which physical direction and sign define this head offset relative to the beam and its rotation axis?
Q109 | Clean/model_augmentation/systems/gantry_ss.py:69 | step | With inertia varying with Y, which velocity-dependent inertial effects are included in this baseline and which modelling assumption accounts for their absence from the stated force balance?
Q110 | Clean/model_augmentation/utils/deepSI_corrections.py:15 | parameter | WindowedSystemData: are absent u or y signals supported by its window-construction method, and what sample/channel layout must supplied arrays use?
Q111 | Clean/model_augmentation/utils/deepSI_corrections.py:54 | parameter | SystemDataNormalizer: may the scales be scalars or channel arrays, what broadcasting layout is required, and must all scales be strictly positive?
Q112 | Clean/model_augmentation/utils/deepSI_corrections.py:81 | step | Why discard a supplied state trajectory x during normalisation instead of carrying or normalising it?
Q113 | Clean/model_augmentation/utils/torch_nets.py:18 | parameter | activation: must this be tanh, or is another no-argument module class supported by build_mlp and AugmentationNetwork?
Q114 | Clean/model_augmentation/utils/torch_nets.py:74 | return | g_aug^n is called normalised here: are the added-state outputs scaled by any fixed normalisation, or only expressed in learned coordinates?
Q115 | Clean/model_augmentation/utils/utils.py:13 | parameter | to_tensor: what non-tensor input types and shapes are supported, and are dtype, device and sharing with a NumPy input preserved?
Q116 | Clean/model_augmentation/utils/utils.py:25 | parameter | expansion_matrix: must ix contain distinct nonnegative indices smaller than n, and can ix be empty?
Q117 | Clean/model_augmentation/utils/utils.py:39 | parameter | selection_matrix: are repeated indices allowed, and are negative or empty index lists supported?
```

## List B, noise

```text
N001 | Clean/gantry_interconnect_dynamic.py:79 | repeats-code | "1. Run folder." - The section label restates the immediately following operations.
N002 | Clean/gantry_interconnect_dynamic.py:99 | repeats-code | "2. Data and normalisation." - The section label restates the immediately following operations.
N003 | Clean/gantry_interconnect_dynamic.py:111 | repeats-code | "3. Model and closed loop." - The section label restates the immediately following operations.
N004 | Clean/gantry_interconnect_dynamic.py:121 | repeats-code | "4. Encoder start, or the restored checkpoint." - The section label restates the immediately following operations.
N005 | Clean/gantry_interconnect_dynamic.py:131 | repeats-code | "5. Training." - The section label restates the immediately following operations.
N006 | Clean/gantry_dynamic/config.py:48 | repeats-code | "config.json keys of the run environment fields, read back by from_config_json" - The dictionary directly maps these field names to the JSON keys.
N007 | Clean/gantry_dynamic/config.py:69 | repeats-doc | "One thesis run: the five axes, the run environment and the fixed settings." - Repeats the module description and the field documentation below.
N008 | Clean/gantry_dynamic/config.py:126 | empty-tag | "HEURISTIC: thesis value" - The source is only 'thesis value(s)', with no explanation of what selected the value.
N009 | Clean/gantry_dynamic/config.py:131 | empty-tag | "HEURISTIC: thesis values" - The source is only 'thesis value(s)', with no explanation of what selected the value.
N010 | Clean/gantry_dynamic/config.py:136 | empty-tag | "HEURISTIC: thesis values" - The source is only 'thesis value(s)', with no explanation of what selected the value.
N011 | Clean/gantry_dynamic/config.py:196 | repeats-code | "True on the OBC arm." - The one-line predicate says exactly this.
N012 | Clean/gantry_dynamic/config.py:213 | repeats-doc | "HEURISTIC: 10 % detuning" - Repeats the 10 percent detuning already defined in this property's formula.
N013 | Clean/gantry_dynamic/config.py:242 | repeats-code | "stdout of one git command in the repository." - The subprocess call and stdout access say exactly this.
N014 | Clean/gantry_dynamic/controller.py:40 | repeats-doc | "Inertia matrix for the controller design." - The formula or detailed paragraph immediately below already identifies this matrix, response, controller or simulator.
N015 | Clean/gantry_dynamic/controller.py:59 | repeats-doc | "Plant frequency response for the controller tuning." - The formula or detailed paragraph immediately below already identifies this matrix, response, controller or simulator.
N016 | Clean/gantry_dynamic/controller.py:71 | repeats-doc | "A = [0 I; -R K -R C], B = [0; R], R = M(Y_op)^-1." - Repeats the state-space formula described in the frequency-response docstring.
N017 | Clean/gantry_dynamic/controller.py:79 | repeats-doc | "Shape of the controller K_1." - The formula or detailed paragraph immediately below already identifies this matrix, response, controller or simulator.
N018 | Clean/gantry_dynamic/controller.py:87 | repeats-doc | "Numerator 10 w_b (s + w_b/6)(s + w_b/3), denominator s (s + 3 w_b)(s + 10 w_b)." - Repeats the numerator and denominator already given in the same function's docstring.
N019 | Clean/gantry_dynamic/controller.py:94 | repeats-doc | "Per-axis controller K_1." - The formula or detailed paragraph immediately below already identifies this matrix, response, controller or simulator.
N020 | Clean/gantry_dynamic/controller.py:105 | repeats-code | "1. Plant and controller shape at w_b." - The section label restates the following frequency evaluation, gain computation or bilinear discretisation.
N021 | Clean/gantry_dynamic/controller.py:111 | repeats-code | "2. Gain kappa_j." - The section label restates the following frequency evaluation, gain computation or bilinear discretisation.
N022 | Clean/gantry_dynamic/controller.py:114 | repeats-code | "3. Tustin at ts." - The section label restates the following frequency evaluation, gain computation or bilinear discretisation.
N023 | Clean/gantry_dynamic/controller.py:121 | repeats-doc | "Controller K_fb as a state space." - The formula or detailed paragraph immediately below already identifies this matrix, response, controller or simulator.
N024 | Clean/gantry_dynamic/controller.py:199 | repeats-doc | "Closed-loop simulator of training and validation." - The formula or detailed paragraph immediately below already identifies this matrix, response, controller or simulator.
N025 | Clean/gantry_dynamic/controller.py:214 | repeats-doc | "1. Controller bank over the training and validation records." - The function docstring already describes the bank, compiled step and loss scales.
N026 | Clean/gantry_dynamic/controller.py:230 | repeats-doc | "2. Compiled step." - The function docstring already describes the bank, compiled step and loss scales.
N027 | Clean/gantry_dynamic/controller.py:239 | repeats-doc | "3. Loss and validation scales." - The function docstring already describes the bank, compiled step and loss scales.
N028 | Clean/gantry_dynamic/controller.py:243 | repeats-doc | "Loss weight: W_r = diag(sigma_y / sigma_e,r) per training record." - Repeats W_r and the record-dependent scaling already stated in this function's docstring.
N029 | Clean/gantry_dynamic/data.py:101 | repeats-doc | "Actuator force at the model rate." - The detailed paragraph immediately below already specifies this resampling, scale or normalisation.
N030 | Clean/gantry_dynamic/data.py:119 | repeats-doc | "Measured positions at the model rate." - The detailed paragraph immediately below already specifies this resampling, scale or normalisation.
N031 | Clean/gantry_dynamic/data.py:134 | repeats-doc | "Error-only filtering: y_k = r_k + [B(y - r)]_{k D}; r is unfiltered." - Repeats the error-only filtering formula and the unfiltered reference from the function docstring.
N032 | Clean/gantry_dynamic/data.py:163 | repeats-doc | "Servo-error scale sigma_e." - The detailed paragraph immediately below already specifies this resampling, scale or normalisation.
N033 | Clean/gantry_dynamic/data.py:175 | repeats-doc | "sigma_e,j = (mean over k >= k0 of (e_j,k - mean e_j)^2)^(1/2)." - Restates the population-standard-deviation definition already given in the docstring.
N034 | Clean/gantry_dynamic/data.py:184 | repeats-code | "Load the train, val and test records of the manifest split." - The function name and immediately following split loading already say this.
N035 | Clean/gantry_dynamic/data.py:214 | repeats-doc | "Normalisation of the augmented model." - The detailed paragraph immediately below already specifies this resampling, scale or normalisation.
N036 | Clean/gantry_dynamic/data.py:227 | repeats-code | "1. Physical states of the training records." - The section label merely names the following data collection or statistics.
N037 | Clean/gantry_dynamic/data.py:234 | repeats-doc | "Positions: q = P^-T y (so the position means are P^-T mu_y)." - The positions transform and its mean relation are already explained in the function docstring.
N038 | Clean/gantry_dynamic/data.py:243 | repeats-code | "2. Statistics." - The section label merely names the following data collection or statistics.
N039 | Clean/gantry_dynamic/data.py:244 | repeats-code | "mu_i = mean(i), sigma_i = (mean (i - mu_i)^2)^(1/2) + 1e-8, i in {x_tilde, u_act, y}." - The following mean and standard-deviation assignments directly state the same formula.
N040 | Clean/gantry_dynamic/data.py:253 | repeats-doc | "3. Output map: C^n = T_y P^T [I 0] T_x^-1, with no offset since P^T mu_q = mu_y." - Repeats C^n and the no-offset relation from the function docstring.
N041 | Clean/gantry_dynamic/evaluation.py:36 | repeats-code | "axis name in the figures" - The dictionary directly supplies the figure's axis-name substitution.
N042 | Clean/gantry_dynamic/evaluation.py:41 | repeats-doc | "Baseline-only copy of a model." - The detailed paragraph immediately below already explains the baseline-only copy.
N043 | Clean/gantry_dynamic/evaluation.py:66 | repeats-doc | "Identifiable combinations of the data-generating parameters." - The next paragraph already identifies the returned parameter combinations.
N044 | Clean/gantry_dynamic/evaluation.py:80 | repeats-doc | "Training coordinate of the true parameters." - The next paragraph already defines the true-parameter training coordinate.
N045 | Clean/gantry_dynamic/evaluation.py:102 | repeats-doc | "Scores of one split in both frames." - The detailed paragraph and return description already specify the split scores and frames.
N046 | Clean/gantry_dynamic/evaluation.py:228 | repeats-doc | "Evaluate the final checkpoint of a run folder on every split." - The following paragraph already explains the final-checkpoint evaluation and its outputs.
N047 | Clean/gantry_dynamic/evaluation.py:241 | repeats-code | "1. The run on the CPU: configuration, records and the final checkpoint." - The following calls plainly load the configuration, data, model and checkpoint on the CPU.
N048 | Clean/gantry_dynamic/evaluation.py:250 | repeats-doc | "start_phase 'lbfgs': the weights only, no Adam moments" - The start_phase setting and checkpoint loader already document this weights-only mode.
N049 | Clean/gantry_dynamic/evaluation.py:272 | repeats-doc | "2. Per split: the free runs, the scores of both frames, the figures; one simulation per record and model." - The evaluate docstring and _split_scores contract already describe these computations.
N050 | Clean/gantry_dynamic/evaluation.py:311 | repeats-doc | "3. Physical parameters: true, start and learned identifiable combinations, in COMBO_NAMES order and the physical" - The parameter-reporting API and surrounding output already give the order and physical units.
N051 | Clean/gantry_dynamic/evaluation.py:312 | repeats-doc | "units of theta_base." - The parameter-reporting API and surrounding output already give the order and physical units.
N052 | Clean/gantry_dynamic/evaluation.py:322 | repeats-code | "4. Data files and the loss curves." - The following writes and plotting call plainly save data and draw the curves.
N053 | Clean/gantry_dynamic/model.py:29 | repeats-doc | "The untrained augmented model." - The following paragraph already describes the untrained augmented model and its start.
N054 | Clean/gantry_dynamic/model.py:48 | repeats-code | "1. Baseline block and output map." - The section label names the immediately following construction or wiring.
N055 | Clean/gantry_dynamic/model.py:67 | repeats-code | "2. Network block phi_aug." - The section label names the immediately following construction or wiring.
N056 | Clean/gantry_dynamic/model.py:78 | repeats-code | "3. Wiring." - The section label names the immediately following construction or wiring.
N057 | Clean/gantry_dynamic/model.py:85 | repeats-doc | "one network output per state, in state order" - The immediately preceding wiring explanation already specifies state-order network outputs.
N058 | Clean/gantry_dynamic/model.py:93 | repeats-code | "4. Model, normaliser and encoder." - The section label names the immediately following construction or wiring.
N059 | Clean/gantry_dynamic/model.py:111 | repeats-code | "5. Optimizer and, on the OBC arm, the projection." - The section label names the immediately following construction or wiring.
N060 | Clean/gantry_dynamic/model.py:147 | repeats-doc | "Encoder start from a baseline simulation." - The following paragraph already explains the simulated-baseline encoder start.
N061 | Clean/gantry_dynamic/model.py:199 | repeats-code | "closed_loop_step calls the model's output_only" - The next assignment plainly exposes the model's output_only method on the recorder.
N062 | Clean/gantry_dynamic/model.py:216 | repeats-code | "2. Fit the encoder to the simulated states." - The next call explicitly fits the encoder to the collected state targets.
N063 | Clean/gantry_dynamic/model.py:228 | repeats-doc | "Least-squares start of the encoder." - The following formula and paragraph already specify the least-squares encoder start.
N064 | Clean/gantry_dynamic/model.py:244 | repeats-code | "1. Solve." - The section label simply names the subsequent solve or parameter copies.
N065 | Clean/gantry_dynamic/model.py:258 | repeats-doc | "W_s = argmin ||((chi - mu) / s) W_s - (X - mean X)||_F^2." - The least-squares objective is already given above; this line rewrites it in standardised coordinates.
N066 | Clean/gantry_dynamic/model.py:265 | repeats-code | "2. Write into the encoder." - The section label simply names the subsequent solve or parameter copies.
N067 | Clean/gantry_dynamic/obc_gantry.py:34 | repeats-doc | "the float64 P^-T: measured positions y to generalised positions q = P^-T y" - The module conventions already define the same measured-to-generalised coordinate transform.
N068 | Clean/gantry_dynamic/obc_gantry.py:47 | repeats-doc | "Velocity of the OBC tuples." - The following formula and argument/return descriptions already identify the tuple velocity.
N069 | Clean/gantry_dynamic/obc_gantry.py:64 | repeats-doc | "OBC tuples of the training records." - The following paragraph already defines the training reference tuples.
N070 | Clean/gantry_dynamic/obc_gantry.py:85 | repeats-code | "1. Tuples per record, in physical units." - The next loop plainly loads each training record.
N071 | Clean/gantry_dynamic/obc_gantry.py:88 | repeats-code | "one training record in float64, resampled as for training" - The next loop plainly loads each training record.
N072 | Clean/gantry_dynamic/obc_gantry.py:91 | repeats-doc | "x_tilde_i = (q_i, qdot_i), q_i = P^-T y_i, at i = 2 .. N - 4 (the stencil gives qdot at 2 .. N - 3)" - Repeats the position map and sample range in the reference-set docstring.
N073 | Clean/gantry_dynamic/obc_gantry.py:98 | repeats-code | "2. Normalise and stack: x_tilde^n = T_x (x_tilde - mu_x), u^n = T_u (u - mu_u)." - The two following affine normalisation expressions say exactly this.
N074 | Clean/gantry_dynamic/obc_gantry.py:117 | repeats-doc | "OBC correction on the protected physical rows." - Repeats the class description and the next paragraph.
N075 | Clean/gantry_dynamic/obc_gantry.py:131 | repeats-doc | "every physical row is protected; the network output of state i is column i (one output per state, in order)" - The class and constructor already say all six physical rows are protected and state-ordered.
N076 | Clean/gantry_dynamic/obc_gantry.py:135 | repeats-doc | "E: E_ii = 1 on the protected physical rows, 0 elsewhere" - E and its protected rows have already been specified in the constructor's buffer documentation.
N077 | Clean/gantry_dynamic/obc_gantry.py:141 | repeats-doc | "theta_aux = (a, c): p sensitivity coefficients and the coefficient of the baseline transition" - Repeats the theta_aux layout already given by the module and constructor docstrings.
N078 | Clean/gantry_dynamic/obc_gantry.py:156 | repeats-doc | "Matrices of the OBC correction." - The following paragraph and return description already specify the tangent matrices.
N079 | Clean/gantry_dynamic/obc_gantry.py:170 | repeats-doc | "OBC correction f_obc^n." - The following formula and return description already identify the correction.
N080 | Clean/gantry_dynamic/obc_gantry.py:186 | repeats-doc | "Correction on the physical rows: f_obc^n = E (varphi a + c f_base^n)." - Repeats the correction formula from this function's docstring.
N081 | Clean/gantry_dynamic/obc_gantry.py:197 | repeats-doc | "Builder of the OBC basis." - The following paragraph already identifies the basis builder.
N082 | Clean/gantry_dynamic/obc_gantry.py:213 | repeats-doc | "J = [Phi  col(f_base^n(beta_lin, z_i))] at beta_lin = vbar64, with its build time." - Repeats the outer builder's exact formula and return contract.
N083 | Clean/gantry_dynamic/obc_gantry.py:218 | repeats-doc | "J = [Phi  F_base^n(beta_lin)]." - Repeats the basis formula already stated in the builder docstring.
N084 | Clean/gantry_dynamic/obc_gantry.py:227 | repeats-doc | "Network field on the OBC tuples." - The following formula and return description already specify the reference network field.
N085 | Clean/gantry_dynamic/obc_gantry.py:241 | repeats-doc | "F^n_aug = col(f^n_aug(theta_aug, (x_tilde^n_i, 0), u^n_i)), sample-major." - Repeats the enclosing function's field formula and stacking order.
N086 | Clean/gantry_dynamic/training.py:162 | repeats-doc | "An empty history, written to hist_path at every entry." - Repeats the class docstring's empty-history and write-on-entry contract.
N087 | Clean/gantry_dynamic/training.py:181 | repeats-code | "The history as arrays, one entry per row, for val_hist.npz." - The method name and dictionary construction already show that the history becomes arrays.
N088 | Clean/gantry_dynamic/training.py:193 | repeats-code | "Validation-probe entry: record the validation that just ran." - The next call simply records the just-computed validation value.
N089 | Clean/gantry_dynamic/training.py:295 | repeats-code | "1. Resume state." - The section label names the immediately following stage.
N090 | Clean/gantry_dynamic/training.py:303 | repeats-doc | "resume = (path, meta) of the checkpoint main has loaded into fit_sys" - The train docstring already defines the resume pair and says that main loaded it.
N091 | Clean/gantry_dynamic/training.py:319 | repeats-code | "2. Adam with PS." - The section label names the immediately following stage.
N092 | Clean/gantry_dynamic/training.py:330 | repeats-code | "3. Pre-polish checkpoint and the selected row." - The section label names the immediately following stage.
N093 | Clean/gantry_dynamic/training.py:342 | repeats-code | "4. Polish and final selection." - The section label names the immediately following stage.
N094 | Clean/gantry_dynamic/training.py:352 | repeats-code | "5. Final checkpoint." - The section label names the immediately following stage.
N095 | Clean/model_augmentation/__init__.py:1 | history | "Adapted from Model-Augmentation-Public by Jan H. Hoekstra (https://github.com/JanHHoekstra/Model-Augmentation-Public," - Describes upstream authorship, revision and change markers rather than model or runtime behaviour.
N096 | Clean/model_augmentation/__init__.py:2 | history | "commit 8200923), BSD 3-Clause License, see LICENSE. Changes to his code are marked # CHANGED, additions @added or" - Describes upstream authorship, revision and change markers rather than model or runtime behaviour.
N097 | Clean/model_augmentation/__init__.py:3 | history | "__project_origin__." - Describes upstream authorship, revision and change markers rather than model or runtime behaviour.
N098 | Clean/model_augmentation/fit_systems/blocks.py:14 | history | "CHANGED: restricted to the thesis training path; Jan's full code: Model-Augmentation-Public," - Describes the upstream version and changes to its imports.
N099 | Clean/model_augmentation/fit_systems/blocks.py:15 | history | "commit 8200923 (model_augmentation/__init__.py)" - Describes the upstream version and changes to its imports.
N100 | Clean/model_augmentation/fit_systems/blocks.py:19 | history | "CHANGED: the @added marker and gantry_ss for the added gantry blocks; unused imports removed" - Describes the upstream version and changes to its imports.
N101 | Clean/model_augmentation/fit_systems/blocks.py:25 | filler | "Basic block implementation with variables that need to be defined for interconnect." - The generic description supplies no block variables or contract beyond the class name.
N102 | Clean/model_augmentation/fit_systems/blocks.py:28 | repeats-code | "Store the block input size nz and output size nw." - The constructor only assigns the two size arguments.
N103 | Clean/model_augmentation/fit_systems/blocks.py:38 | repeats-doc | "Network block: phi_aug with a gate per output." - Repeats the class description and the following paragraph.
N104 | Clean/model_augmentation/fit_systems/blocks.py:47 | repeats-code | "n_hidden_layers: hidden layers." - The documented hidden-layer count and activation simply repeat the argument names or fixed caller values.
N105 | Clean/model_augmentation/fit_systems/blocks.py:54 | repeats-code | "CHANGED: net_kwargs reach the network's constructor (the number of physical rows)" - The following constructor call plainly forwards net_kwargs.
N106 | Clean/model_augmentation/fit_systems/blocks.py:68 | repeats-doc | "Network block of the augmented model." - The formula immediately below already describes the network block.
N107 | Clean/model_augmentation/fit_systems/blocks.py:87 | repeats-doc | "CHANGED: C and D required and registered as buffers (they follow .to(), not trained); no empty-matrix cases" - The class docstring already states that C and D are fixed buffers; the removed empty cases are history.
N108 | Clean/model_augmentation/fit_systems/blocks.py:89 | repeats-code | "C and D as fixed buffers; ny, nx, nu from their shapes." - The buffer registrations and size accesses directly state this.
N109 | Clean/model_augmentation/fit_systems/blocks.py:103 | repeats-doc | "Output map of the augmented model." - The next paragraph already defines the output map.
N110 | Clean/model_augmentation/fit_systems/blocks.py:113 | history | "CHANGED: no size assertion" - Only records that an assertion was removed.
N111 | Clean/model_augmentation/fit_systems/blocks.py:125 | repeats-doc | "CHANGED: mats, the matrices built once per objective, are passed on to nonlinear_function" - The forward docstring already explains how mats is passed to the nonlinear map.
N112 | Clean/model_augmentation/fit_systems/blocks.py:135 | repeats-doc | "Baseline transition f_base^n." - The class description and following paragraph already identify this transition.
N113 | Clean/model_augmentation/fit_systems/blocks.py:154 | repeats-code | "1. Force transform u = P u_act." - The following buffer stores the force transform; its physical meaning is supplied on the next comment line.
N114 | Clean/model_augmentation/fit_systems/blocks.py:158 | repeats-code | "2. Normalisation buffers." - The following registrations are visibly the normalisation buffers.
N115 | Clean/model_augmentation/fit_systems/blocks.py:166 | repeats-doc | "Baseline transition f_base^n: one RK4 step of the vector field over T_s." - The class and next paragraph already describe the baseline RK4 transition.
N116 | Clean/model_augmentation/fit_systems/blocks.py:180 | repeats-code | "Stages: k1 = h F(x, u), k2 = h F(x + k1/2, u), k3 = h F(x + k2/2, u), k4 = h F(x + k3, u)." - The four immediately following expressions state these RK4 stages.
N117 | Clean/model_augmentation/fit_systems/blocks.py:189 | repeats-doc | "Baseline vector field F." - The following equation already identifies the vector field.
N118 | Clean/model_augmentation/fit_systems/blocks.py:204 | repeats-code | "1. Denormalise: x = x^n sigma_x + mu_x, u = u^n sigma_u + mu_u." - Restates the immediately following denormalisation, force transform, force balance, derivative or normalisation.
N119 | Clean/model_augmentation/fit_systems/blocks.py:211 | repeats-code | "2. Net force." - Restates the immediately following denormalisation, force transform, force balance, derivative or normalisation.
N120 | Clean/model_augmentation/fit_systems/blocks.py:212 | repeats-code | "Generalised force: u = P u_act." - Restates the immediately following denormalisation, force transform, force balance, derivative or normalisation.
N121 | Clean/model_augmentation/fit_systems/blocks.py:215 | repeats-code | "Net force: fnet = u - K q - C qdot." - Restates the immediately following denormalisation, force transform, force balance, derivative or normalisation.
N122 | Clean/model_augmentation/fit_systems/blocks.py:234 | repeats-code | "4. State derivative." - Restates the immediately following denormalisation, force transform, force balance, derivative or normalisation.
N123 | Clean/model_augmentation/fit_systems/blocks.py:236 | repeats-doc | "Closed form: xdot = (qdot, a) (moves results by rounding)." - The closed_form argument and the branch expression already state this form and rounding effect.
N124 | Clean/model_augmentation/fit_systems/blocks.py:248 | repeats-code | "5. Normalise: xdot^n = xdot / sigma_x." - Restates the immediately following denormalisation, force transform, force balance, derivative or normalisation.
N125 | Clean/model_augmentation/fit_systems/blocks.py:253 | repeats-doc | "Baseline vector field F and its directional derivative." - The next paragraph already explains the vector field and its tangent.
N126 | Clean/model_augmentation/fit_systems/blocks.py:273 | repeats-code | "1. Denormalise: x = x^n sigma_x + mu_x, s_x = sx sigma_x; u = u^n sigma_u + mu_u, u = P u_act." - Restates the immediately following denormalisation or force/tangent equations.
N127 | Clean/model_augmentation/fit_systems/blocks.py:279 | repeats-code | "2. Net force and its tangent: fnet = u - K q - C qdot, s_fnet = -K s_q - dK q - C s_qdot - dC qdot." - Restates the immediately following denormalisation or force/tangent equations.
N128 | Clean/model_augmentation/fit_systems/blocks.py:284 | repeats-code | "3. Acceleration a = num / d." - The subsequent numerator-over-denominator expression directly gives the acceleration.
N129 | Clean/model_augmentation/fit_systems/blocks.py:285 | repeats-code | "Determinant and its tangent: d(Y) = m_h poly(Y), poly = alpha gamma - beta^2 + 2 beta m_h Y" - The following poly and s_poly expressions spell out this determinant and its product-rule derivative.
N130 | Clean/model_augmentation/fit_systems/blocks.py:286 | repeats-code | "+ m_h (alpha - m_h) Y^2 (product rule term by term)." - The following poly and s_poly expressions spell out this determinant and its product-rule derivative.
N131 | Clean/model_augmentation/fit_systems/blocks.py:295 | repeats-code | "s_d = dm_h poly + m_h s_poly." - The next two assignments give precisely this product rule.
N132 | Clean/model_augmentation/fit_systems/blocks.py:299 | repeats-code | "Numerator and its tangent: num = N0 fnet + Y (N1 fnet + Y N2 fnet) (Horner)." - The immediately following numerator expression is already in Horner form.
N133 | Clean/model_augmentation/fit_systems/blocks.py:302 | repeats-code | "s_(N_j fnet) = dN_j fnet + N_j s_fnet; s_num by the product rule on the Horner form." - The following three products and s_num expression directly state this product rule.
N134 | Clean/model_augmentation/fit_systems/blocks.py:310 | repeats-code | "Acceleration and its tangent: a = num / d, s_a = (s_num - a s_d) / d (quotient rule)." - The next two assignments state the acceleration and quotient-rule tangent.
N135 | Clean/model_augmentation/fit_systems/blocks.py:314 | repeats-code | "4. State derivative." - The branch immediately below directly constructs these state derivatives.
N136 | Clean/model_augmentation/fit_systems/blocks.py:316 | repeats-code | "Closed form: (xdot, s_xdot) = ((qdot, a), (s_qdot, s_a))." - The branch immediately below directly constructs these state derivatives.
N137 | Clean/model_augmentation/fit_systems/blocks.py:333 | repeats-code | "5. Normalise: xdot^n = xdot / sigma_x, likewise the tangent." - The next two divisions visibly normalise the derivative and its tangent.
N138 | Clean/model_augmentation/fit_systems/blocks.py:339 | repeats-doc | "Baseline transition f_base^n and its directional derivative." - The next paragraph already specifies the RK4 transition and tangent.
N139 | Clean/model_augmentation/fit_systems/blocks.py:353 | repeats-doc | "Stages: k_i = h F(x_i), s_i = h DF(x_i)[sx_i, dmats], x_i and sx_i as in _rk4." - Repeats the stage/tangent formula in this function's docstring.
N140 | Clean/model_augmentation/fit_systems/blocks.py:363 | repeats-code | "Tangent update: sx_next = sx + (s1 + 2 s2 + 2 s3 + s4) / 6." - The next assignment is exactly the stated tangent update.
N141 | Clean/model_augmentation/fit_systems/blocks.py:403 | repeats-code | "Stiffness: K = diag(0, k_b,Sigma, 0)." - The following small stack directly constructs this diagonal stiffness matrix.
N142 | Clean/model_augmentation/fit_systems/blocks.py:418 | repeats-doc | "Y-dependent parts of the inertia matrix." - The formula immediately below already defines the Y-dependent inertia parts.
N143 | Clean/model_augmentation/fit_systems/blocks.py:428 | repeats-doc | "M_1 = -m_h (e1 e2^T + e2 e1^T)." - Repeats the M_1 and M_2 formulas in the same function's docstring.
N144 | Clean/model_augmentation/fit_systems/blocks.py:434 | repeats-doc | "M_2 = m_h e2 e2^T." - Repeats the M_1 and M_2 formulas in the same function's docstring.
N145 | Clean/model_augmentation/fit_systems/blocks.py:443 | repeats-doc | "Baseline matrices of one objective." - The next paragraph already describes the objective's baseline matrix tuple.
N146 | Clean/model_augmentation/fit_systems/blocks.py:466 | repeats-doc | "Baseline transition f_base^n as a block." - The next equation and return description already specify the block transition.
N147 | Clean/model_augmentation/fit_systems/blocks.py:514 | repeats-doc | "1. beta = 0 at theta^0_base." - The constructor docstring already states beta = 0 at combo_init.
N148 | Clean/model_augmentation/fit_systems/blocks.py:521 | repeats-code | "2. Calibrate the signed coordinate." - The following bound, ratio, atanh and scale calculations plainly perform this calibration.
N149 | Clean/model_augmentation/fit_systems/blocks.py:535 | repeats-doc | "Identifiable combinations theta_base(theta)." - The next paragraph already defines the identifiable combinations.
N150 | Clean/model_augmentation/fit_systems/blocks.py:556 | repeats-doc | "A raw parameter vector with given combinations." - The next paragraph already explains the raw representative and gauge.
N151 | Clean/model_augmentation/fit_systems/blocks.py:572 | repeats-code | "m_1 + m_2 = m_Sigma - m_b, J_b + J_h = J_eff - (L_b^2/4)(m_1 + m_2), r = J_b0 / (J_b0 + J_h0)." - The next three assignments directly state these three formulas.
N152 | Clean/model_augmentation/fit_systems/blocks.py:577 | repeats-doc | "Equal splits of k_b,Sigma, c_b,Sigma; m_1,2 = (m_1 + m_2 +- m_Delta) / 2;" - The gauge docstring and immediately following stack already give these splits.
N153 | Clean/model_augmentation/fit_systems/blocks.py:578 | repeats-doc | "J_b = r J_sum, J_h = (1 - r) J_sum." - The gauge docstring and immediately following stack already give these splits.
N154 | Clean/model_augmentation/fit_systems/blocks.py:585 | repeats-doc | "Bound on |m_Delta|." - The formula immediately below already specifies the mass-difference bound.
N155 | Clean/model_augmentation/fit_systems/blocks.py:599 | repeats-doc | "Parameter map theta_base(beta)." - The following equation already defines the parameter map.
N156 | Clean/model_augmentation/fit_systems/blocks.py:610 | repeats-doc | "Positive combinations: theta_base,j = theta^0_base,j exp(beta_j)." - Repeats the positive and signed mapping equations already given in this docstring.
N157 | Clean/model_augmentation/fit_systems/blocks.py:612 | repeats-doc | "Signed mass difference: m_Delta = b tanh(eta_0 + s beta_mDelta)." - Repeats the positive and signed mapping equations already given in this docstring.
N158 | Clean/model_augmentation/fit_systems/blocks.py:618 | repeats-doc | "Inverse parameter map beta(theta_base)." - The following equations already identify the inverse map.
N159 | Clean/model_augmentation/fit_systems/blocks.py:634 | repeats-code | "Ratio: r = m_Delta / b, |r| < 1 required." - The next ratio calculation and range check state this requirement.
N160 | Clean/model_augmentation/fit_systems/blocks.py:640 | repeats-doc | "Inverse: beta_j = log(theta_base,j / theta^0_base,j), beta_mDelta = (atanh(r) - eta_0) / s." - Repeats the inverse formulas already given in the function's docstring.
N161 | Clean/model_augmentation/fit_systems/blocks.py:646 | repeats-doc | "Current parameter estimate." - The next line and return contract already identify the current estimate.
N162 | Clean/model_augmentation/fit_systems/blocks.py:656 | repeats-doc | "Baseline transition as a function of beta." - The next paragraph already defines the beta-dependent transition.
N163 | Clean/model_augmentation/fit_systems/blocks.py:670 | repeats-doc | "Baseline matrices and their directional derivative in beta." - The following jvp description already defines the matrices and their derivative.
N164 | Clean/model_augmentation/fit_systems/closed_loop.py:27 | repeats-doc | "Controller bank, one per distinct design." - The class and following paragraph already identify the controller bank.
N165 | Clean/model_augmentation/fit_systems/closed_loop.py:57 | repeats-doc | "Controller step in residual form." - The immediately following equations already specify the controller step.
N166 | Clean/model_augmentation/fit_systems/closed_loop.py:70 | repeats-doc | "Both rows in one product: (u_fb^n, x^c_{k+1}) = [C^n; A] x^c + [D^n; B^n] e^n." - Repeats the combined-product layout documented above at buffer construction.
N167 | Clean/model_augmentation/fit_systems/closed_loop.py:77 | repeats-doc | "Closed-loop model step." - The next paragraph already specifies the closed-loop model step.
N168 | Clean/model_augmentation/fit_systems/closed_loop.py:91 | repeats-doc | "One step: y_hat^n = C^n x_tilde^n, u_hat^n = u^n + u_fb^n, then x_hat^n_{k+1}." - Repeats the outer function's step sequence.
N169 | Clean/model_augmentation/fit_systems/closed_loop.py:94 | repeats-code | "Output: y_hat^n = C^n x_tilde^n." - Restates the following output evaluation, feedback update or state update.
N170 | Clean/model_augmentation/fit_systems/closed_loop.py:96 | repeats-code | "Feedback on e^n = y^n - y_hat^n, and x^c_{k+1}." - Restates the following output evaluation, feedback update or state update.
N171 | Clean/model_augmentation/fit_systems/closed_loop.py:98 | repeats-code | "State update with the model input u_hat^n = u^n + u_fb^n." - Restates the following output evaluation, feedback update or state update.
N172 | Clean/model_augmentation/fit_systems/closed_loop.py:105 | repeats-doc | "Closed-loop simulation of a window." - The following paragraph already defines the window simulation.
N173 | Clean/model_augmentation/fit_systems/closed_loop.py:144 | repeats-doc | "The closed loop of training and validation; assigned to fit_sys.simulator." - Repeats the module description and constructor role.
N174 | Clean/model_augmentation/fit_systems/closed_loop.py:151 | repeats-doc | "sigma_e per validation record and axis (n_val, ny), the denominator of sim-NRMSe" - These fields are already described by the constructor summary and the rollout/loss contracts.
N175 | Clean/model_augmentation/fit_systems/closed_loop.py:153 | repeats-doc | "the closed_loop_step compiled by torch.compile, or None for eager" - These fields are already described by the constructor summary and the rollout/loss contracts.
N176 | Clean/model_augmentation/fit_systems/closed_loop.py:156 | repeats-doc | "W_r per training record, one (ny,) row each, appended to every window as loss_w" - These fields are already described by the constructor summary and the rollout/loss contracts.
N177 | Clean/model_augmentation/fit_systems/closed_loop.py:176 | repeats-doc | "Closed-loop rollout of a batch of windows (the simulator of V)." - The next paragraph and return contract already specify the batch rollout.
N178 | Clean/model_augmentation/fit_systems/closed_loop.py:189 | repeats-doc | "use the compiled step for gradient-enabled CUDA rollouts, including the polish" - Repeats the compiled-rollout condition in the same method's docstring.
N179 | Clean/model_augmentation/fit_systems/closed_loop.py:240 | repeats-doc | "Validation error for checkpoint selection." - The next paragraph already defines the checkpoint-selection score.
N180 | Clean/model_augmentation/fit_systems/closed_loop.py:258 | repeats-doc | "Closed-loop free run of full records." - The following paragraph already defines the complete-record free run.
N181 | Clean/model_augmentation/fit_systems/closed_loop.py:288 | repeats-code | "1. Normalise, u^n = (u - mu_u) / sigma_u, y^n = (y - mu_y) / sigma_y, and group records by length." - The immediately following array expressions perform exactly this normalisation.
N182 | Clean/model_augmentation/fit_systems/closed_loop.py:298 | repeats-code | "filled with return_pred only" - The return_pred conditional plainly controls whether predictions are filled.
N183 | Clean/model_augmentation/fit_systems/closed_loop.py:302 | repeats-code | "2. Free run per group, then RMS per record." - The next loop plainly runs groups and computes record RMS values.
N184 | Clean/model_augmentation/fit_systems/closed_loop.py:307 | repeats-doc | "x_hat^n_{k0} = psi(chi_{k0}); the scored samples are k0 .. N - 1" - Repeats the encoder start and scored range from the function docstring.
N185 | Clean/model_augmentation/fit_systems/interconnect.py:1 | history | "Jan's interconnect of blocks and his encoder fit system, with the augmented model of the thesis." - Describes upstream ownership, revision or changes to imports and the inherited implementation.
N186 | Clean/model_augmentation/fit_systems/interconnect.py:2 | history | "CHANGED: restricted to the thesis training path; Jan's full code: Model-Augmentation-Public," - Describes upstream ownership, revision or changes to imports and the inherited implementation.
N187 | Clean/model_augmentation/fit_systems/interconnect.py:3 | history | "commit 8200923 (model_augmentation/__init__.py)" - Describes upstream ownership, revision or changes to imports and the inherited implementation.
N188 | Clean/model_augmentation/fit_systems/interconnect.py:9 | history | "CHANGED: os, math and get_work_dirs added for the changed fit (the epoch index of the OBC refresh and the checkpoint" - Describes upstream ownership, revision or changes to imports and the inherited implementation.
N189 | Clean/model_augmentation/fit_systems/interconnect.py:10 | history | "saving); time and deepSI's loader helpers as in Jan's" - Describes upstream ownership, revision or changes to imports and the inherited implementation.
N190 | Clean/model_augmentation/fit_systems/interconnect.py:18 | history | "CHANGED: the @added marker, the blocks the added code looks for (Static_ANN_Block) and the renamed normaliser" - Describes upstream ownership, revision or changes to imports and the inherited implementation.
N191 | Clean/model_augmentation/fit_systems/interconnect.py:19 | history | "SystemDataNormalizer instead of the parameterised blocks and detect_algebraic_loop" - Describes upstream ownership, revision or changes to imports and the inherited implementation.
N192 | Clean/model_augmentation/fit_systems/interconnect.py:57 | repeats-doc | "Augmented-model transition." - The next equations already define the augmented transition.
N193 | Clean/model_augmentation/fit_systems/interconnect.py:82 | repeats-code | "CHANGED: the signal loop is _propagate, shared with output_only, on the device and dtype of x" - The next call visibly delegates signal propagation to _propagate.
N194 | Clean/model_augmentation/fit_systems/interconnect.py:103 | repeats-code | "signals outside order stay None" - The following list initialisation makes the uncomputed entries None.
N195 | Clean/model_augmentation/fit_systems/interconnect.py:115 | repeats-doc | "the physics block takes the matrices built once per objective" - The forward argument contract and _propagate code already identify the block receiving pass_mats.
N196 | Clean/model_augmentation/fit_systems/interconnect.py:167 | repeats-code | "CHANGED: the output's dependency cone, resolved once for output_only" - The next call explicitly computes and stores the output's signal cone.
N197 | Clean/model_augmentation/fit_systems/interconnect.py:169 | repeats-code | "CHANGED: the block that takes the per-objective matrices pass_mats, resolved once" - The next lookup explicitly locates the block with pass_mats.
N198 | Clean/model_augmentation/fit_systems/interconnect.py:173 | repeats-code | "CHANGED: the connection matrices on the module's device and dtype" - The following conversion explicitly uses the module parameter's device and dtype.
N199 | Clean/model_augmentation/fit_systems/interconnect.py:192 | repeats-doc | "Output of the augmented model without a state update." - The next paragraph and return contract already specify the output-only operation.
N200 | Clean/model_augmentation/fit_systems/interconnect.py:278 | repeats-code | "Connect every signal of input_signal_list into block, and block into every signal of output_signal_list." - The two short loops directly connect each listed source and destination.
N201 | Clean/model_augmentation/fit_systems/interconnect.py:286 | repeats-code | "Signal index of a block or a signal name." - The two following type branches directly implement this index lookup.
N202 | Clean/model_augmentation/fit_systems/interconnect.py:293 | repeats-code | "0 for x and xp, 1 for u and y." - The immediately following conditions directly list these indices.
N203 | Clean/model_augmentation/fit_systems/interconnect.py:300 | repeats-code | "The canonical name of a connection method." - The following tiny dictionary directly states the canonical mapping.
N204 | Clean/model_augmentation/fit_systems/interconnect.py:307 | repeats-code | "Store the two signal indices, the method and the connection matrix." - The constructor merely assigns the named arguments to identically named fields.
N205 | Clean/model_augmentation/fit_systems/interconnect.py:338 | repeats-code | "A floating array cast to dtype, other arrays unchanged." - The tiny helper directly checks floating dtype and converts it.
N206 | Clean/model_augmentation/fit_systems/interconnect.py:365 | repeats-doc | "CHANGED: a normaliser whose windows keep the data's dtype; build_model sets its values" - The constructor docstring already identifies the dtype-preserving normaliser.
N207 | Clean/model_augmentation/fit_systems/interconnect.py:397 | history | "CHANGED: arguments of the thesis path only (no norm fit, optimizer setup, concurrent validation, timeout," - Lists features removed from the upstream fit signature.
N208 | Clean/model_augmentation/fit_systems/interconnect.py:398 | history | "scheduler, Dataset loader, verbose levels or time profile); init_model is called by the pipeline" - Lists features removed from the upstream fit signature.
N209 | Clean/model_augmentation/fit_systems/interconnect.py:401 | repeats-doc | "Adam phase with checkpoint selection." - The following paragraph and arguments already define the Adam phase.
N210 | Clean/model_augmentation/fit_systems/interconnect.py:441 | repeats-code | "1. Windows, data loader and the initial validation." - The following calls visibly construct windows, the loader and initial validation.
N211 | Clean/model_augmentation/fit_systems/interconnect.py:442 | history | "CHANGED: counters and history always start empty (Jan's continues an earlier fit)" - Compares the counter reset with the upstream implementation.
N212 | Clean/model_augmentation/fit_systems/interconnect.py:461 | repeats-doc | "CHANGED: DtypeDataLoader (same seed-0 shuffle), since My_Simple_DataLoader casts the windows to float32" - The DtypeDataLoader class already explains the fixed shuffle and dtype preservation.
N213 | Clean/model_augmentation/fit_systems/interconnect.py:471 | repeats-doc | "2. Adam updates, each followed by the PS step; validation every its_per_val." - Repeats the update/PS/validation sequence in the fit docstring.
N214 | Clean/model_augmentation/fit_systems/interconnect.py:517 | repeats-code | "3. Select: reload the best dump." - The next checkpoint-loading code visibly selects and reloads the best dump.
N215 | Clean/model_augmentation/fit_systems/interconnect.py:550 | repeats-doc | "Locate the network block and the block that takes pass_mats." - The next two lookups explicitly locate the network and per-objective matrix blocks.
N216 | Clean/model_augmentation/fit_systems/interconnect.py:564 | repeats-doc | "Coefficient of the OBC correction for prediction." - The following paragraph already explains the prediction coefficient and basis policy.
N217 | Clean/model_augmentation/fit_systems/interconnect.py:587 | repeats-doc | "Matrices of one objective." - The next paragraph already defines the objective's matrices.
N218 | Clean/model_augmentation/fit_systems/interconnect.py:605 | repeats-doc | "Training objective V." - The following equation and paragraph already define V.
N219 | Clean/model_augmentation/fit_systems/interconnect.py:617 | repeats-code | "1. Initial states: x_hat^n_{tau|tau} = psi(theta_enc, chi_tau)." - The section label repeats the following encoder call, coefficient solve or rollout.
N220 | Clean/model_augmentation/fit_systems/interconnect.py:619 | repeats-code | "2. OBC coefficient of this objective." - The section label repeats the following encoder call, coefficient solve or rollout.
N221 | Clean/model_augmentation/fit_systems/interconnect.py:631 | repeats-code | "3. Closed-loop rollout and cost." - The section label repeats the following encoder call, coefficient solve or rollout.
N222 | Clean/model_augmentation/fit_systems/interconnect.py:638 | repeats-doc | "Objective: V = mean (W_r (y^n - y_hat^n))^2." - Repeats the objective formula already given in the same function's docstring.
N223 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:45 | repeats-doc | "The windows at the scaled stride, and a seeded subset of n_windows if there are more." - Repeats the class docstring's scaled stride and seeded subset.
N224 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:70 | repeats-code | "The selected windows of one array, on the model's device and dtype." - The short helper directly selects these windows and converts their dtype and device.
N225 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:91 | repeats-doc | "Write grad V to the parameters' .grad slots and return a detached scalar V on the fixed batch." - The enclosing closure factory already states the objective, gradient and batch.
N226 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:108 | repeats-code | "Recognise CUDA OOM exceptions and RuntimeErrors containing 'out of memory'." - The one-line predicate directly recognises exactly these exceptions.
N227 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:130 | repeats-doc | "(keep, why): keep the polished weights only if V is finite and the validation error drops." - The main polish docstring already states the finite-loss and improving-validation acceptance rule.
N228 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:142 | repeats-doc | "Polish of the selected model." - The following paragraph already specifies the selected-model polish.
N229 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:145 | filler | "refines the Adam result without risking a worse model." - The claim about avoiding risk adds no contract beyond the explicit rollback condition on the previous line.
N230 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:167 | repeats-code | "the start weights, for an exact rollback" - The next list comprehension plainly clones the starting parameters.
N231 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:184 | repeats-code | "Store the L-BFGS iteration and closure counts in out." - The following two assignments visibly record these counters.
N232 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:202 | repeats-code | "1. Score to beat and the fixed batch." - The section label simply names the following validation/batch, optimizer/probe or iteration operations.
N233 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:218 | repeats-code | "2. Optimizer and determinism probe." - The section label simply names the following validation/batch, optimizer/probe or iteration operations.
N234 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:227 | repeats-code | "a failed compiled probe is retried once on the eager rollout" - The following force_eager branch visibly retries a failed compiled probe.
N235 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:262 | repeats-code | "3. L-BFGS iterations." - The section label simply names the following validation/batch, optimizer/probe or iteration operations.
N236 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:263 | repeats-doc | "allow ceil(max_iter / inner_iter) blocks, stopping early if a block uses fewer than inner_iter iterations" - PolishSpec already documents outer calls and early termination.
N237 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:281 | repeats-code | "the objective at the final weights: one forward pass over the batch, without gradient" - The next no_grad block explicitly evaluates the loss at the final weights.
N238 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:302 | repeats-doc | "4. Keep the polish only if the validation error dropped." - Repeats the module, main function and acceptance docstrings.
N239 | Clean/model_augmentation/fit_systems/obc.py:20 | repeats-doc | "J D^-1 = U S V^T, D_ii = ||J_:,i|| (1 for a zero column): the economy SVD of the column-scaled basis, with its" - The constructor immediately repeats the SVD, column scaling and numerical-rank contract.
N240 | Clean/model_augmentation/fit_systems/obc.py:21 | repeats-doc | "numerical rank." - The constructor immediately repeats the SVD, column scaling and numerical-rank contract.
N241 | Clean/model_augmentation/fit_systems/obc.py:24 | repeats-doc | "Factorised OBC basis." - The following formula already identifies the factorised basis.
N242 | Clean/model_augmentation/fit_systems/obc.py:36 | repeats-doc | "1. Column scale D = diag(||J_:,i||), 1 for a zero column." - Repeats the column-scale definition in this constructor's docstring.
N243 | Clean/model_augmentation/fit_systems/obc.py:40 | repeats-code | "2. Factorise." - The following linalg.svd call plainly factorises the scaled matrix.
N244 | Clean/model_augmentation/fit_systems/obc.py:41 | repeats-doc | "J D^-1 = U S V^T." - Repeats the SVD equation already specified in the docstring.
N245 | Clean/model_augmentation/fit_systems/obc.py:43 | repeats-doc | "3. Numerical rank; truncate on rank loss." - Repeats the numerical-rank and truncation contract already specified above.
N246 | Clean/model_augmentation/fit_systems/obc.py:56 | repeats-doc | "OBC coefficient theta_aux." - The next formula and return description already identify the coefficient.
N247 | Clean/model_augmentation/fit_systems/obc.py:67 | repeats-doc | "theta_aux = D^-1 V S^-1 U^T F." - Repeats the coefficient formula from the same function's docstring.
N248 | Clean/model_augmentation/fit_systems/obc.py:72 | repeats-code | "Move the factors to device." - The following .to calls directly move the factors.
N249 | Clean/model_augmentation/fit_systems/obc.py:80 | repeats-code | "Device of the factors." - The property simply returns U.device.
N250 | Clean/model_augmentation/fit_systems/obc.py:87 | repeats-doc | "Stacked baseline sensitivity Phi." - The following formula already defines the stacked sensitivity.
N251 | Clean/model_augmentation/fit_systems/obc.py:104 | filler | "1. Prepare." - The generic 'Prepare' label adds no information about the preparation's purpose.
N252 | Clean/model_augmentation/fit_systems/obc.py:112 | repeats-code | "2. Jacobian per chunk of tuples." - The next loop visibly computes a Jacobian per chunk.
N253 | Clean/model_augmentation/fit_systems/obc.py:115 | repeats-doc | "varphi_i = d step(beta, z_i) / d beta at beta_lin, rows row_ix, stacked sample-major." - Repeats the derivative and row-order contract already given in the function docstring.
N254 | Clean/model_augmentation/fit_systems/obc.py:120 | repeats-code | "3. Stack, with the optional f_base^n column." - The following concatenations visibly stack the chunks and optional values.
N255 | Clean/model_augmentation/fit_systems/obc.py:123 | repeats-doc | "J = [Phi  col(step(beta_lin, z_i))]." - Repeats the append_value argument's formula.
N256 | Clean/model_augmentation/fit_systems/obc.py:135 | repeats-code | "Store the reference-field evaluator and the basis builder returning (J, seconds)." - The constructor only stores the two supplied callables.
N257 | Clean/model_augmentation/fit_systems/obc.py:144 | repeats-doc | "Rebuild the OBC basis at beta_lin." - The following paragraph already describes the rebuild and rank policy.
N258 | Clean/model_augmentation/fit_systems/obc.py:152 | repeats-code | "1. Candidate basis at beta_lin." - The next calls plainly build the candidate basis at the detached expansion point.
N259 | Clean/model_augmentation/fit_systems/obc.py:159 | repeats-doc | "2. Rank policy: keep the last full-rank basis or abort on rank loss, else accept." - Repeats the constructor's rank policy and this method's existing explanation.
N260 | Clean/model_augmentation/fit_systems/obc.py:178 | repeats-doc | "Per-epoch refresh of beta_lin." - The following paragraph already defines the per-epoch refresh.
N261 | Clean/model_augmentation/fit_systems/obc.py:192 | repeats-doc | "theta_aux of the current network." - The following equation and return description already identify the network coefficient.
N262 | Clean/model_augmentation/fit_systems/pre_encoder.py:6 | history | "CHANGED: restricted to the thesis training path; Jan's full code: Model-Augmentation-Public," - Records upstream provenance or the imports added for this project.
N263 | Clean/model_augmentation/fit_systems/pre_encoder.py:7 | history | "commit 8200923 (model_augmentation/__init__.py)" - Records upstream provenance or the imports added for this project.
N264 | Clean/model_augmentation/fit_systems/pre_encoder.py:12 | history | "CHANGED: the @added marker and the layer stack of the added encoder" - Records upstream provenance or the imports added for this project.
N265 | Clean/model_augmentation/fit_systems/pre_encoder.py:34 | repeats-doc | "Encoder psi = [W^b; W^a] chi + psi_tilde(chi)." - The class and following paragraph already specify the encoder.
N266 | Clean/model_augmentation/fit_systems/pre_encoder.py:55 | repeats-code | "1. Physical linear rows W^b." - The section label merely names the physical rows, added rows or nonlinear network constructed next.
N267 | Clean/model_augmentation/fit_systems/pre_encoder.py:64 | repeats-code | "2. Added linear rows W^a (Xavier)." - The section label merely names the physical rows, added rows or nonlinear network constructed next.
N268 | Clean/model_augmentation/fit_systems/pre_encoder.py:73 | repeats-code | "3. Nonlinear correction psi_tilde." - The section label merely names the physical rows, added rows or nonlinear network constructed next.
N269 | Clean/model_augmentation/fit_systems/pre_encoder.py:82 | repeats-doc | "Encoder psi." - The next equation already identifies the encoder.
N270 | Clean/model_augmentation/fit_systems/pre_encoder.py:96 | repeats-code | "Linear part: (W^b chi, W^a chi)." - The next two matrix products visibly compute the linear part.
N271 | Clean/model_augmentation/fit_systems/pre_encoder.py:104 | repeats-doc | "Nonlinear correction: + psi_tilde, which reads (u history, y history), a fixed permutation of chi (Jan's" - The module docstring already states the nonlinear input permutation; the upstream origin adds only history.
N272 | Clean/model_augmentation/fit_systems/pre_encoder.py:105 | repeats-doc | "encoder input order)." - The module docstring already states the nonlinear input permutation; the upstream origin adds only history.
N273 | Clean/model_augmentation/fit_systems/ps2_opening.py:51 | repeats-code | "Plateau: len(values) > window and min(values[-window:]) > (1 - tol) min(values[:-window])." - The function's one-line predicate states exactly the same condition.
N274 | Clean/model_augmentation/fit_systems/ps2_opening.py:56 | repeats-doc | "Left-out records of PS." - The following paragraph already defines the calibration records.
N275 | Clean/model_augmentation/fit_systems/ps2_opening.py:66 | repeats-doc | "a class is the record name without its trailing number, 'TR-S5' -> 'TR-S'" - Repeats the excitation-class naming rule from the same docstring.
N276 | Clean/model_augmentation/fit_systems/ps2_opening.py:75 | repeats-doc | "The opening after every Adam update until it ends: S1 fits a decoder of the residual on the added states," - Repeats the module's S1/S2 description and the after_step contract.
N277 | Clean/model_augmentation/fit_systems/ps2_opening.py:76 | repeats-doc | "S2 the network's added-state transition." - Repeats the module's S1/S2 description and the after_step contract.
N278 | Clean/model_augmentation/fit_systems/ps2_opening.py:81 | repeats-code | "the network outputs of the added states: one output per state, in state order" - The next range expression plainly chooses the added-state output columns.
N279 | Clean/model_augmentation/fit_systems/ps2_opening.py:101 | repeats-doc | "True once the opening has ended." - The module already defines S3 as the end of the opening, which is the complete predicate below.
N280 | Clean/model_augmentation/fit_systems/ps2_opening.py:105 | repeats-doc | "PS step after each Adam step on V." - The next paragraph already explains the PS step after Adam.
N281 | Clean/model_augmentation/fit_systems/ps2_opening.py:123 | repeats-code | "1. One S1 step." - The following call and cadence branch plainly perform the S1 step and calibration check.
N282 | Clean/model_augmentation/fit_systems/ps2_opening.py:125 | repeats-code | "2. On the check grid: nu on the left-out records, then end S1 or continue." - The following call and cadence branch plainly perform the S1 step and calibration check.
N283 | Clean/model_augmentation/fit_systems/ps2_opening.py:211 | repeats-doc | "S1 residual target e_tau." - The following formula already defines the S1 target.
N284 | Clean/model_augmentation/fit_systems/ps2_opening.py:224 | repeats-doc | "Residual over H samples of the closed-loop rollout from x_hat^n_{tau|tau}." - Repeats the residual and its scaling from the function docstring.
N285 | Clean/model_augmentation/fit_systems/ps2_opening.py:226 | repeats-doc | "Scale: e_tau / sigma_e." - Repeats the residual and its scaling from the function docstring.
N286 | Clean/model_augmentation/fit_systems/ps2_opening.py:230 | repeats-code | "xbar = psi_a(chi): the n_xbar rows after the physical ones." - The returned slice directly selects the added rows after the physical rows.
N287 | Clean/model_augmentation/fit_systems/ps2_opening.py:234 | repeats-doc | "S1 step." - The following minimisation and paragraph already define the S1 step.
N288 | Clean/model_augmentation/fit_systems/ps2_opening.py:242 | repeats-code | "1. Sample windows and encode." - The section label names the immediately following sampling, target, gradient or Adam operations.
N289 | Clean/model_augmentation/fit_systems/ps2_opening.py:245 | repeats-code | "2. Frozen residual target." - The section label names the immediately following sampling, target, gradient or Adam operations.
N290 | Clean/model_augmentation/fit_systems/ps2_opening.py:247 | repeats-code | "3. Decoder cost and gradient." - The section label names the immediately following sampling, target, gradient or Adam operations.
N291 | Clean/model_augmentation/fit_systems/ps2_opening.py:248 | repeats-doc | "S1 cost: mean ||p(xbar_{tau|tau}) - e_tau||^2 over the batch, H and the axes." - Repeats the S1 objective and averaging convention already specified above.
N292 | Clean/model_augmentation/fit_systems/ps2_opening.py:250 | repeats-code | "g = grad_(theta_p, theta^a_enc) V_S1." - The next autograd.grad call plainly computes this gradient.
N293 | Clean/model_augmentation/fit_systems/ps2_opening.py:252 | repeats-code | "4. S1's own Adam step." - The section label names the immediately following sampling, target, gradient or Adam operations.
N294 | Clean/model_augmentation/fit_systems/ps2_opening.py:262 | repeats-doc | "S1 check nu." - The following formula already defines the calibration check.
N295 | Clean/model_augmentation/fit_systems/ps2_opening.py:277 | repeats-doc | "S2: one-step fit of the added-state transition." - The following objective already explains the one-step S2 fit.
N296 | Clean/model_augmentation/fit_systems/ps2_opening.py:299 | repeats-code | "1. Fixed pairs from the frozen encoder." - The section label names the immediately following pair construction, optimization or parameter restore.
N297 | Clean/model_augmentation/fit_systems/ps2_opening.py:306 | repeats-code | "2. Adam on the S2 cost, checked on the left-out pairs." - The section label names the immediately following pair construction, optimization or parameter restore.
N298 | Clean/model_augmentation/fit_systems/ps2_opening.py:333 | repeats-code | "3. Restore the best parameters." - The section label names the immediately following pair construction, optimization or parameter restore.
N299 | Clean/model_augmentation/systems/gantry_ss.py:70 | repeats-doc | "alpha = m_Sigma + m_h, beta = (L_b/2) m_Delta, gamma = J_eff." - Repeats alpha, beta and gamma from both the module and function return documentation.
N300 | Clean/model_augmentation/systems/gantry_ss.py:77 | repeats-doc | "Adjugate coefficients of Y^0, Y^1, Y^2." - The adjugate polynomial coefficients are already defined by the function docstring.
N301 | Clean/model_augmentation/systems/gantry_ss.py:97 | repeats-doc | "State row of the LFR G." - The following formula and return description already identify the LFR state row.
N302 | Clean/model_augmentation/systems/gantry_ss.py:116 | repeats-doc | "R = M_0^-1 = N0 / d0." - Repeats R = N0/d0 from the same function's docstring.
N303 | Clean/model_augmentation/systems/gantry_ss.py:123 | repeats-doc | "A_x = [0 I; -R K -R C], B_w = [0 0; -R M_1 -R M_2], B_u = [0; R]." - Repeats the three matrix formulas already supplied in the same function's docstring.
N304 | Clean/model_augmentation/utils/deepSI_corrections.py:5 | history | "CHANGED: Jan's fixed_System_data and fixed_System_data_norm, renamed WindowedSystemData and SystemDataNormalizer;" - Records the old class names, upstream revision and restriction of the upstream code.
N305 | Clean/model_augmentation/utils/deepSI_corrections.py:6 | history | "restricted to the thesis training path; Jan's full code: Model-Augmentation-Public," - Records the old class names, upstream revision and restriction of the upstream code.
N306 | Clean/model_augmentation/utils/deepSI_corrections.py:7 | history | "commit 8200923 (model_augmentation/__init__.py)" - Records the old class names, upstream revision and restriction of the upstream code.
N307 | Clean/model_augmentation/utils/deepSI_corrections.py:21 | repeats-doc | "Training windows of a record." - The next paragraph already specifies the training windows.
N308 | Clean/model_augmentation/utils/deepSI_corrections.py:54 | repeats-code | "Store mu_u, sigma_u, mu_y, sigma_y as u0, ustd, y0, ystd." - The following four assignments store exactly these normalisation parameters.
N309 | Clean/model_augmentation/utils/deepSI_corrections.py:61 | repeats-doc | "Normalisation of the records." - The next equation and argument/return contract already define record normalisation.
N310 | Clean/model_augmentation/utils/deepSI_corrections.py:72 | repeats-code | "CHANGED: lists handled record by record; errors name the expected input" - The surrounding branches plainly handle lists and raise the named input-type error.
N311 | Clean/model_augmentation/utils/deepSI_corrections.py:78 | repeats-doc | "Normalise: u^n = (u - u0) / ustd, y^n = (y - y0) / ystd." - Repeats the normalisation formula in the class and method docstrings.
N312 | Clean/model_augmentation/utils/torch_nets.py:6 | history | "CHANGED: restricted to the thesis training path; Jan's full code: Model-Augmentation-Public," - Only records upstream provenance or the origin of the added marker import.
N313 | Clean/model_augmentation/utils/torch_nets.py:7 | history | "commit 8200923 (model_augmentation/__init__.py)" - Only records upstream provenance or the origin of the added marker import.
N314 | Clean/model_augmentation/utils/torch_nets.py:11 | history | "CHANGED: the marker of the added definitions" - Only records upstream provenance or the origin of the added marker import.
N315 | Clean/model_augmentation/utils/torch_nets.py:37 | repeats-doc | "b_L = 0." - Repeats b_L = 0 from the function docstring.
N316 | Clean/model_augmentation/utils/torch_nets.py:48 | repeats-doc | "Augmentation network phi_aug at its start." - The class and following paragraph already define the augmentation network's start.
N317 | Clean/model_augmentation/utils/torch_nets.py:60 | repeats-doc | "the routed outputs are the states in order, so the physical rows are the first nx_phys" - The module conventions and constructor argument documentation already specify this output order.
N318 | Clean/model_augmentation/utils/torch_nets.py:62 | repeats-doc | "W^xtilde_L = 0, so f_aug^n = 0 at the start." - Repeats the zero-physical-output initialisation and its effect from the class and constructor docstrings.
N319 | Clean/model_augmentation/utils/torch_nets.py:66 | repeats-doc | "Augmentation network phi_aug." - The class description and argument/return contract already specify this network mapping.
N320 | Clean/model_augmentation/utils/torch_nets.py:68 | repeats-doc | "(f_aug^n, g_aug^n) = phi_aug(x_hat^n, u^n)." - The class description and argument/return contract already specify this network mapping.
N321 | Clean/model_augmentation/utils/utils.py:2 | history | "CHANGED: restricted to the thesis training path; Jan's full code: Model-Augmentation-Public," - Records the upstream source and revision.
N322 | Clean/model_augmentation/utils/utils.py:3 | history | "commit 8200923 (model_augmentation/__init__.py)" - Records the upstream source and revision.
N323 | Clean/model_augmentation/utils/utils.py:9 | history | "Mark a class or function as added by this project, not part of Jan's framework (grep \"@added\")." - Describes a provenance marker that returns its argument unchanged.
N324 | Clean/model_augmentation/utils/utils.py:14 | repeats-code | "input as a tensor; a tensor is returned as it is." - The one-line conditional directly states this conversion and tensor pass-through.
N325 | Clean/model_augmentation/utils/utils.py:15 | history | "CHANGED: None is no longer passed through (no caller passes it)" - Only describes a removed None case and the previous caller situation.
N326 | Clean/model_augmentation/utils/utils.py:20 | repeats-doc | "Routing matrix E." - The following matrix definition already supplies the routing or selection meaning.
N327 | Clean/model_augmentation/utils/utils.py:34 | repeats-doc | "Selection matrix S." - The following matrix definition already supplies the routing or selection meaning.
```
