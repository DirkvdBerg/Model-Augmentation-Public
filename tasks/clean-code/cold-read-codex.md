| File | Questions | Noise lines |
| --- | ---: | ---: |
| `Clean/gantry_interconnect_dynamic.py` | 6 | 11 |
| `Clean/gantry_dynamic/__init__.py` | 0 | 0 |
| `Clean/gantry_dynamic/config.py` | 24 | 9 |
| `Clean/gantry_dynamic/controller.py` | 13 | 10 |
| `Clean/gantry_dynamic/data.py` | 17 | 13 |
| `Clean/gantry_dynamic/evaluation.py` | 12 | 6 |
| `Clean/gantry_dynamic/model.py` | 18 | 18 |
| `Clean/gantry_dynamic/obc_gantry.py` | 17 | 16 |
| `Clean/gantry_dynamic/training.py` | 21 | 13 |
| `Clean/model_augmentation/__init__.py` | 0 | 3 |
| `Clean/model_augmentation/fit_systems/__init__.py` | 0 | 0 |
| `Clean/model_augmentation/fit_systems/blocks.py` | 18 | 73 |
| `Clean/model_augmentation/fit_systems/closed_loop.py` | 11 | 20 |
| `Clean/model_augmentation/fit_systems/interconnect.py` | 32 | 49 |
| `Clean/model_augmentation/fit_systems/lbfgs_polish.py` | 8 | 23 |
| `Clean/model_augmentation/fit_systems/obc.py` | 8 | 23 |
| `Clean/model_augmentation/fit_systems/pre_encoder.py` | 8 | 12 |
| `Clean/model_augmentation/fit_systems/ps2_opening.py` | 18 | 29 |
| `Clean/model_augmentation/systems/__init__.py` | 0 | 0 |
| `Clean/model_augmentation/systems/gantry_ss.py` | 14 | 4 |
| `Clean/model_augmentation/utils/__init__.py` | 0 | 0 |
| `Clean/model_augmentation/utils/deepSI_corrections.py` | 3 | 11 |
| `Clean/model_augmentation/utils/torch_nets.py` | 3 | 11 |
| `Clean/model_augmentation/utils/utils.py` | 0 | 10 |
| **Total** | **251** | **364** |

## List A, questions

```text
Q001 | Clean/gantry_interconnect_dynamic.py:129 | step | Which parts of training state does load_checkpoint restore, and what does start_phase change about restoration?
Q002 | Clean/gantry_interconnect_dynamic.py:133 | step | On resume, do epochs and n_its specify a new budget or the total budget including work already completed?
Q003 | Clean/gantry_interconnect_dynamic.py:134 | return | bestfit: what validation metric and aggregation does this scalar represent, and is it measured before or after L-BFGS?
Q004 | Clean/gantry_interconnect_dynamic.py:151 | setting | --polish-only: does polishing use the best Adam parameters, the last Adam parameters, or another saved state?
Q005 | Clean/gantry_interconnect_dynamic.py:156 | setting | --from best versus last: which validation criterion defines best, and which optimizer, scheduler and random-number states are retained?
Q006 | Clean/gantry_interconnect_dynamic.py:158 | setting | --closed-form: what LFR quantities are bypassed, and does this change computational cost or differentiability as well as rounding?
Q007 | Clean/gantry_dynamic/config.py:23 | variable | The 0.01 transient criterion: which noise amplitude or statistic is the reference, and which 36 records and filtering conditions establish the 4.6 ms bound?
Q008 | Clean/gantry_dynamic/config.py:55 | step | Why is record_trim applied to noise-free records as well as to records that undergo antialias filtering?
Q009 | Clean/gantry_dynamic/config.py:63 | setting | arm='obc' versus 'u': which part of the augmentation is constrained, and what model behavior or training cost changes?
Q010 | Clean/gantry_dynamic/config.py:64 | setting | start='correct': correct relative to which physical parameter vector, and does it include the unmodelled plant dynamics?
Q011 | Clean/gantry_dynamic/config.py:65 | setting | n_a: what dynamics and units do the added states represent, and why are only 0, 2 and 8 allowed?
Q012 | Clean/gantry_dynamic/config.py:66 | setting | ps2: what are the two initialization stages, what does enabling them change, and what happens when n_a is zero?
Q013 | Clean/gantry_dynamic/config.py:70 | setting | compile_mode='reduce-overhead' versus None: what work is compiled, and what startup, memory and execution-time tradeoffs does this introduce?
Q014 | Clean/gantry_dynamic/config.py:71 | setting | truncate_s: which part of which records is retained, at which sampling rate is the cutoff applied, and what does None mean?
Q015 | Clean/gantry_dynamic/config.py:73 | setting | epochs: what constitutes one epoch with strided, overlapping simulation windows?
Q016 | Clean/gantry_dynamic/config.py:74 | setting | n_its: iterations of which optimizer operation, how does this interact with epochs, and what does None mean?
Q017 | Clean/gantry_dynamic/config.py:76 | setting | its_per_val: does halving it only evaluate and select checkpoints more often, or also affect stopping or learning-rate scheduling?
Q018 | Clean/gantry_dynamic/config.py:79 | setting | lbfgs_max_iter and lbfgs_inner_iter: how do these limits combine, and are line-search evaluations counted as iterations?
Q019 | Clean/gantry_dynamic/config.py:81 | setting | lbfgs_windows: how are these windows chosen, does changing their count change the objective or just memory, and are they reused at every step?
Q020 | Clean/gantry_dynamic/config.py:82 | setting | start_phase: which preparation, training and checkpoint-selection steps are skipped or repeated for 'lbfgs'?
Q021 | Clean/gantry_dynamic/config.py:90 | setting | seed: which random choices share this seed, and which training, PS and polish random streams use other seeds?
Q022 | Clean/gantry_dynamic/config.py:91 | variable | nx_phys, nu and ny: what is the physical state ordering, and which three force and displacement axes are counted?
Q023 | Clean/gantry_dynamic/config.py:97 | setting | fs_new: what sampling-rate choices are supported, and which integration, controller, encoder and filter time scales change with this rate?
Q024 | Clean/gantry_dynamic/config.py:99 | setting | stride: measured in original samples, training samples or windows, and what does reducing it change about the number and overlap of training examples?
Q025 | Clean/gantry_dynamic/config.py:100 | setting | nf_seconds: does the rollout duration include the encoder history, and which physical timescale is the chosen 0.100 s intended to cover?
Q026 | Clean/gantry_dynamic/config.py:101 | setting | na_nb: which input and output samples form this window, is the current sample included, and why is its length described as n + 1?
Q027 | Clean/gantry_dynamic/config.py:104 | setting | n_hidden_layers: does this count only nonlinear hidden layers or also input/output projections, and do encoder and augmentation use the same activation?
Q028 | Clean/gantry_dynamic/config.py:105 | setting | up_sample: what is upsampled, which values are supported, and how does changing it affect integration, observations and the loss?
Q029 | Clean/gantry_dynamic/config.py:114 | setting | lbfgs_tol_grad and lbfgs_tol_change: which gradient norm and loss/parameter changes are tested, and are these absolute or relative tolerances?
Q030 | Clean/gantry_dynamic/config.py:166 | variable | theta_base,j: which ten parameter combinations and ordering receive these detuning multipliers, and are they dimensional or scaled parameters?
Q031 | Clean/gantry_dynamic/controller.py:21 | variable | mb, mh, m1, m2, Jb and Jh: which bodies and axes do these inertias refer to, and what are their units?
Q032 | Clean/gantry_dynamic/controller.py:22 | variable | cg1, cg2, cy, cb1 and cb2: which physical dampers do these denote, and which coefficients are translational versus rotational?
Q033 | Clean/gantry_dynamic/controller.py:23 | variable | kb1, kb2, Lb and d: which springs and geometric distances are these, and what are their units and positive directions?
Q034 | Clean/gantry_dynamic/controller.py:29 | variable | P: what are the ordered stage and generalized coordinates, and which direction maps positions versus forces?
Q035 | Clean/gantry_dynamic/controller.py:36 | parameter | Y_op: relative to which origin and positive direction is the payload position measured, and what range is supported?
Q036 | Clean/gantry_dynamic/controller.py:77 | step | Why are the controller zeros and poles placed at w_b/6, w_b/3, 3 w_b and 10 w_b?
Q037 | Clean/gantry_dynamic/controller.py:119 | parameter | gain: are zero or negative gains meaningful supported settings, and does changing gain also change the nominal controller tuning?
Q038 | Clean/gantry_dynamic/controller.py:144 | parameter | ctrl_name: is this merely a display label, or must it identify a particular controller family in order to reproduce the stored controller?
Q039 | Clean/gantry_dynamic/controller.py:151 | parameter | use_gain=False: why must a caller opt into stored nonunit gains, and where is that option appropriate during training and evaluation?
Q040 | Clean/gantry_dynamic/controller.py:168 | parameter | ystd and std_u: are these centered signal standard deviations, RMS scales or other normalization factors, and must they be strictly positive?
Q041 | Clean/gantry_dynamic/controller.py:200 | parameter | norm: which fields and normalization conventions must this object provide?
Q042 | Clean/gantry_dynamic/controller.py:234 | step | Why are the first max(na, nb) samples excluded when computing the training loss and validation scales?
Q043 | Clean/gantry_dynamic/controller.py:237 | variable | train_weights: how do these per-record servo-error scales enter the loss, and how are axes and records aggregated?
Q044 | Clean/gantry_dynamic/data.py:23 | variable | The dataset names: what does MSD represent, and does lowpass refer to the generated measurement noise, the resampling filter, or both?
Q045 | Clean/gantry_dynamic/data.py:25 | variable | test-I and test-E: what experiments do these blocks represent, and why are they combined into one test split?
Q046 | Clean/gantry_dynamic/data.py:30 | parameter | MANIFEST.csv: what columns, filename conventions and block/controller values must a supplied data set contain?
Q047 | Clean/gantry_dynamic/data.py:36 | step | Why is only K1 accepted when the controller bank can represent multiple designs and gains?
Q048 | Clean/gantry_dynamic/data.py:45 | variable | meta.controller: what MATLAB structure layout and scalar types are required for name, design_Y, gain and fbw_Hz?
Q049 | Clean/gantry_dynamic/data.py:57 | variable | train_data and val_ckpt_data: what distinguishes these aggregate objects from train_list and val_list, and which consumer expects each representation?
Q050 | Clean/gantry_dynamic/data.py:76 | variable | Cd_norm and Dd_np: which normalized or physical input/output coordinates do these matrices map, especially for a nonzero D?
Q051 | Clean/gantry_dynamic/data.py:93 | step | How should model-interval averaged actuator forces be timed relative to the point-sampled positions and references?
Q052 | Clean/gantry_dynamic/data.py:108 | parameter | d: which record fields, shapes, units and sample alignment are required beyond y and r_sim?
Q053 | Clean/gantry_dynamic/data.py:112 | step | Why is only y - r filtered rather than the entire measured position, and what assumption about the reference makes this appropriate?
Q054 | Clean/gantry_dynamic/data.py:127 | parameter | fields: which optional record fields can be resampled this way, and must they all be arrays at the original simulation rate?
Q055 | Clean/gantry_dynamic/data.py:140 | parameter | What is the intended scope of a record's filename identity when loading files from different folders into the controller registry?
Q056 | Clean/gantry_dynamic/data.py:157 | parameter | k0: why is the first scored sample n + 1, and does that convention match the encoder's excluded current sample?
Q057 | Clean/gantry_dynamic/data.py:203 | variable | q and qdot: what is their ordered physical coordinate vector and the unit of each component?
Q058 | Clean/gantry_dynamic/data.py:223 | step | Why use a backward difference of measured, potentially noisy positions to set the velocity normalization, and why duplicate the first velocity?
Q059 | Clean/gantry_dynamic/data.py:230 | step | Why are normalization statistics weighted per sample across concatenated records rather than equally per record, given the later per-record normalization?
Q060 | Clean/gantry_dynamic/data.py:231 | setting | The +1e-8 floor: what physical scale does this represent for mixed translational and rotational channels, and what happens to a nearly constant channel?
Q061 | Clean/gantry_dynamic/evaluation.py:43 | variable | theta_aux and beta: what do these coordinates represent physically, and why does removing the learned part require zeroing theta_aux in addition to the network gate?
Q062 | Clean/gantry_dynamic/evaluation.py:55 | step | When n_a is positive, how are baseline-only initial states obtained, and can the trained augmented encoder still affect the baseline comparison?
Q063 | Clean/gantry_dynamic/evaluation.py:73 | step | Why is phy.params_init guaranteed to equal the data-generating parameters rather than the potentially detuned start or values restored from a checkpoint?
Q064 | Clean/gantry_dynamic/evaluation.py:98 | parameter | models: which model attributes and normalization are required for the shared bank and record list to be valid?
Q065 | Clean/gantry_dynamic/evaluation.py:101 | step | How is each closed-loop free run initialized, including the encoder and controller state, and is measured output used after initialization?
Q066 | Clean/gantry_dynamic/evaluation.py:103 | variable | floor: under what assumptions is the noisy-versus-noise-free difference an attainable prediction-error floor in this closed-loop experiment?
Q067 | Clean/gantry_dynamic/evaluation.py:181 | step | Why are only reference axes 0 and 2 drawn for each frame?
Q068 | Clean/gantry_dynamic/evaluation.py:205 | step | How should selected checkpoints be interpreted when a resume produces repeated Adam update numbers?
Q069 | Clean/gantry_dynamic/evaluation.py:245 | step | Why is start_phase='lbfgs' used to load every evaluation checkpoint, including runs without polishing?
Q070 | Clean/gantry_dynamic/evaluation.py:272 | step | Why is a noise-free twin loaded using the noisy run's configuration and resampling rule?
Q071 | Clean/gantry_dynamic/evaluation.py:278 | variable | t: is zero the start of the original record, the first sample after edge trimming, or the beginning of the scored free run?
Q072 | Clean/gantry_dynamic/evaluation.py:305 | variable | The physical-parameter table: what are the definitions, ordering and units of the reported identifiable combinations?
Q073 | Clean/gantry_dynamic/model.py:28 | parameter | hp versus cfg: which values take precedence when these contain different n_a, history, sampling or network settings?
Q074 | Clean/gantry_dynamic/model.py:32 | variable | f_aug^n, g_aug^n and f_obc^n: are these discrete state increments or continuous derivatives, and how do their units relate to the sample time?
Q075 | Clean/gantry_dynamic/model.py:65 | variable | The augmentation's concatenated input and output: what is the exact physical-state, added-state and actuator ordering?
Q076 | Clean/gantry_dynamic/model.py:70 | setting | Tanh: why is this activation selected for both networks, and what changes if it is replaced?
Q077 | Clean/gantry_dynamic/model.py:78 | step | connect_block_signals and connect_signals: how do their concat/additive conventions determine block input order and next-state sums?
Q078 | Clean/gantry_dynamic/model.py:95 | variable | W^b and W^a: how are the history coordinates ordered within each linear encoder map?
Q079 | Clean/gantry_dynamic/model.py:111 | step | What does init_model initialize or reset after the encoder and normalizer have already been assigned?
Q080 | Clean/gantry_dynamic/model.py:120 | step | Why is the OBC reference basis attached before fitting the physical encoder, and which initialized parameter values does that basis depend on?
Q081 | Clean/gantry_dynamic/model.py:126 | parameter | on_device: must the optimizer, simulator bank and OBC tensors already be attached for the context manager to move all necessary state?
Q082 | Clean/gantry_dynamic/model.py:157 | step | Why must the OBC coefficient be synchronized before the baseline simulation, even though the physical augmentation initially vanishes?
Q083 | Clean/gantry_dynamic/model.py:173 | step | Why are initial generalized velocities, added states and controller residual states zero, and what assumption about record startup supports this?
Q084 | Clean/gantry_dynamic/model.py:181 | parameter | recorder's *args: which precomputed matrices or correction terms are passed through to the model step?
Q085 | Clean/gantry_dynamic/model.py:194 | step | Why is the encoder fitted from simulated histories and simulated closed-loop inputs instead of measured histories and inputs?
Q086 | Clean/gantry_dynamic/model.py:196 | step | Why can the prefit use all available history-to-state windows instead of requiring the full nf rollout horizon used in training?
Q087 | Clean/gantry_dynamic/model.py:206 | setting | The R^2 >= 0.999 threshold: why is this reconstruction level required to begin training, and how does changing it affect admissible encoder windows?
Q088 | Clean/gantry_dynamic/model.py:213 | step | What observability or approximation assumption makes a linear history map sufficient for the scheduled nonlinear baseline's physical states?
Q089 | Clean/gantry_dynamic/model.py:258 | step | Why must the physical nonlinear output bias be set by evaluating enc at zero after zeroing its existing bias?
Q090 | Clean/gantry_dynamic/model.py:274 | step | Why must the added-state nonlinear output rows remain exactly zero after the physical encoder fit?
Q091 | Clean/gantry_dynamic/obc_gantry.py:5 | variable | Phi, Gamma and varphi: which are single-tuple sensitivities versus stacked matrices, and with respect to which parameter coordinate are they differentiated?
Q092 | Clean/gantry_dynamic/obc_gantry.py:7 | variable | a and c: how are these projection coefficients determined and how does their coordinate transformation preserve f_obc?
Q093 | Clean/gantry_dynamic/obc_gantry.py:36 | variable | Z_step versus Z_field: what differs between their signal layouts and which computation consumes each?
Q094 | Clean/gantry_dynamic/obc_gantry.py:59 | step | Why are reference tuples taken from measured positions with added states fixed to zero instead of from the model's evolving state estimates?
Q095 | Clean/gantry_dynamic/obc_gantry.py:60 | step | Why is the reference set fixed throughout training even as the baseline parameters and added-state dynamics change?
Q096 | Clean/gantry_dynamic/obc_gantry.py:86 | step | Why use a fourth-order central velocity difference for OBC when normalization uses a backward difference?
Q097 | Clean/gantry_dynamic/obc_gantry.py:87 | step | Why is the last valid central-difference tuple excluded from the reference set?
Q098 | Clean/gantry_dynamic/obc_gantry.py:112 | variable | Protected rows: what is being protected from what, and what orthogonality guarantee is imposed on these rows?
Q099 | Clean/gantry_dynamic/obc_gantry.py:138 | step | When is beta copied into beta_lin, and for which operations is the expansion point detached?
Q100 | Clean/gantry_dynamic/obc_gantry.py:142 | parameter | When may set_frozen be called so that theta_aux and beta_lin refer to the same projection basis?
Q101 | Clean/gantry_dynamic/obc_gantry.py:165 | variable | Orthogonal to J: which inner product and sample weighting define orthogonality, and does the guarantee extend beyond the fixed tuples at xbar=0?
Q102 | Clean/gantry_dynamic/obc_gantry.py:187 | parameter | row_ix: may callers select arbitrary or reordered physical rows, and how must this agree with the network field columns?
Q103 | Clean/gantry_dynamic/obc_gantry.py:200 | return | J's rows: how are tuple indices and protected state rows interleaved, and what does each of its eleven columns represent?
Q104 | Clean/gantry_dynamic/obc_gantry.py:210 | step | Why append the baseline transition itself to the sensitivity columns rather than use sensitivities alone?
Q105 | Clean/gantry_dynamic/obc_gantry.py:217 | parameter | field_cols: must these match row_ix exactly, and can added-state network outputs participate in the projection?
Q106 | Clean/gantry_dynamic/obc_gantry.py:244 | parameter | Which existing block ordering and model attributes does attach_obc require before attaching the reference set and projection?
Q107 | Clean/gantry_dynamic/obc_gantry.py:258 | variable | space='affine' and refresh='epoch': what is affine in this construction, and exactly when within an epoch is the basis refreshed?
Q108 | Clean/gantry_dynamic/training.py:31 | parameter | meta: which metadata keys and scalar/array types must be supplied for the saved pair to be resumable?
Q109 | Clean/gantry_dynamic/training.py:61 | variable | done_epochs: why is the deepSI epoch counter rounded, and how should partial epochs and an n_its-limited run contribute to the remaining budget?
Q110 | Clean/gantry_dynamic/training.py:63 | variable | ps2_opening: what do S1, S2 and S3 mean, and is updates a phase-local or cumulative count?
Q111 | Clean/gantry_dynamic/training.py:70 | parameter | saved: which parameter ordering must saved optimizer groups share with the live model for their moments to be matched correctly?
Q112 | Clean/gantry_dynamic/training.py:87 | variable | coordinate_version: which parameter representations do the version values distinguish?
Q113 | Clean/gantry_dynamic/training.py:118 | parameter | start_phase: are 'adam' and 'lbfgs' the only supported values, and does loading restore OBC basis state as well as its frozen coefficient buffers?
Q114 | Clean/gantry_dynamic/training.py:144 | return | meta: which keys are guaranteed for each checkpoint format, and are scalar values NumPy arrays or native Python values?
Q115 | Clean/gantry_dynamic/training.py:164 | variable | per_channel and score: are these RMS, normalized RMS or squared errors, and how are records and axes ordered and aggregated?
Q116 | Clean/gantry_dynamic/training.py:165 | variable | update and epoch: are these cumulative over continuations or local to the current fit call, and what do -1 and NaN indicate?
Q117 | Clean/gantry_dynamic/training.py:198 | step | Why is a polish judged against the previously selected score when rebuilding the OBC basis changes the starting model's score?
Q118 | Clean/gantry_dynamic/training.py:203 | step | Which lazy initialization must an eager validation complete before the compiled polish objective can run?
Q119 | Clean/gantry_dynamic/training.py:220 | step | When a polish is rejected, does the history row's per-channel data describe the candidate or the restored model?
Q120 | Clean/gantry_dynamic/training.py:229 | step | Why is fit_sys.bestfit reset to infinity when resuming, rather than continuing selection against the saved best score?
Q121 | Clean/gantry_dynamic/training.py:238 | step | What work does PS2Opening perform before normal Adam training, and which parameters are allowed to update in each phase?
Q122 | Clean/gantry_dynamic/training.py:256 | setting | When n_its is supplied, why does it also replace its_per_val, and when are intermediate validations then performed?
Q123 | Clean/gantry_dynamic/training.py:264 | step | If Adam's update budget ends before PS reaches S3, how is that partially initialized state treated during polish, saving and a later resume?
Q124 | Clean/gantry_dynamic/training.py:301 | step | Why is done_epochs set to the full requested epoch budget even when n_its may have stopped Adam earlier?
Q125 | Clean/gantry_dynamic/training.py:306 | step | Which exact weights, optimizer moments and OBC expansion point are stored in the pre-polish pair after fit reloads its best checkpoint?
Q126 | Clean/gantry_dynamic/training.py:310 | step | How is the selected validation-history row matched to the checkpoint that fit actually restored?
Q127 | Clean/gantry_dynamic/training.py:333 | variable | ps2_summary versus ps2_opening: which is needed to determine whether initialization is complete when a checkpoint pair is resumed?
Q128 | Clean/gantry_dynamic/training.py:346 | variable | loss.npz time: what is its unit and origin, and does it include validation, PS initialization or polish time?
Q129 | Clean/model_augmentation/fit_systems/blocks.py:9 | variable | K, C and the inverse-inertia coefficients N0, N1 and N2: what are the units of their entries in the mixed translational/rotational coordinate system?
Q130 | Clean/model_augmentation/fit_systems/blocks.py:25 | parameter | nz and nw: must every interconnect block accept and return a trailing singleton dimension, and which additional attributes or methods must subclasses provide?
Q131 | Clean/model_augmentation/fit_systems/blocks.py:85 | parameter | C and D: must these already include input/output normalization, and does this block support nonzero feedthrough in a closed-loop rollout?
Q132 | Clean/model_augmentation/fit_systems/blocks.py:119 | parameter | mats: which subclasses require this argument, and how does the interconnect distinguish them from blocks that take only z?
Q133 | Clean/model_augmentation/fit_systems/blocks.py:137 | parameter | std_x, x_mean, std_u and u_mean: which coordinate ordering and physical units must callers provide?
Q134 | Clean/model_augmentation/fit_systems/blocks.py:140 | setting | up_sample: must it be a strictly positive integer, and are inputs and the learned discrete correction held fixed throughout the internal RK4 substeps?
Q135 | Clean/model_augmentation/fit_systems/blocks.py:151 | variable | P_mat: which actuator and generalized axis orderings and force signs does this fixed transform assume?
Q136 | Clean/model_augmentation/fit_systems/blocks.py:193 | parameter | mats: which entries depend on trainable parameters, and must callers regenerate the tuple after every optimizer update?
Q137 | Clean/model_augmentation/fit_systems/blocks.py:216 | step | Why use a rational polynomial inverse of M(Y) rather than a linear solve at each integration stage?
Q138 | Clean/model_augmentation/fit_systems/blocks.py:234 | variable | The LFR signals z and w: which scheduling/uncertainty interconnection do these represent?
Q139 | Clean/model_augmentation/fit_systems/blocks.py:246 | step | Why is the state/parameter tangent propagated explicitly instead of obtaining the entire transition derivative using automatic differentiation?
Q140 | Clean/model_augmentation/fit_systems/blocks.py:365 | variable | PARAM_NAMES: which physical elements, axes and SI units correspond to each of the fourteen raw parameters?
Q141 | Clean/model_augmentation/fit_systems/blocks.py:369 | parameter | params_init: must it contain strictly positive physical parameters, and what admissibility assumptions are required?
Q142 | Clean/model_augmentation/fit_systems/blocks.py:477 | variable | Identifiable combinations: which experimental or model assumptions make these ten combinations identifiable?
Q143 | Clean/model_augmentation/fit_systems/blocks.py:489 | setting | M_DIFF_RHO=0.99: why use this particular margin, and how does reducing it affect representable plants and optimization conditioning?
Q144 | Clean/model_augmentation/fit_systems/blocks.py:491 | variable | COORDINATE_VERSION=2: which coordinate map does this version identify and when must the value change?
Q145 | Clean/model_augmentation/fit_systems/blocks.py:496 | parameter | combo_init: what finite, positivity and mass-difference conditions must it satisfy, including whether m_Delta may be zero?
Q146 | Clean/model_augmentation/fit_systems/blocks.py:549 | return | The gauge-section raw parameters: are negative representative masses or inertias admissible if the combined inertia matrix remains positive definite?
Q147 | Clean/model_augmentation/fit_systems/closed_loop.py:5 | variable | e^n=y^n-y_hat^n: how does this residual relate to the servo error e=y-r used elsewhere, and why does the reference disappear from the controller equation?
Q148 | Clean/model_augmentation/fit_systems/closed_loop.py:25 | parameter | A, B, C and D: what actuator-axis ordering and controller realization coordinates must these matrices use?
Q149 | Clean/model_augmentation/fit_systems/closed_loop.py:85 | parameter | obc_pass and pass_mats: what are their tuple layouts and when may either be None?
Q150 | Clean/model_augmentation/fit_systems/closed_loop.py:91 | step | Which plant feedthrough assumptions permit output_only before the feedback calculation, and what would change for a plant with nonzero D?
Q151 | Clean/model_augmentation/fit_systems/closed_loop.py:111 | parameter | u_data: is this recorded total actuator input, feedforward input or externally injected excitation?
Q152 | Clean/model_augmentation/fit_systems/closed_loop.py:124 | step | Why is equal model/system controller state at every window start a suitable initialization when the encoder estimates only the plant state?
Q153 | Clean/model_augmentation/fit_systems/closed_loop.py:148 | variable | extra_array_names: which training-array ordering and keyword interface does the simulator assume?
Q154 | Clean/model_augmentation/fit_systems/closed_loop.py:202 | parameter | data and sys_data: what is the ordering of data's history/future arrays, and are both arguments already normalized?
Q155 | Clean/model_augmentation/fit_systems/closed_loop.py:204 | setting | nf=25 and stride=1: why are these the fallback window settings, and which caller normally overrides them?
Q156 | Clean/model_augmentation/fit_systems/closed_loop.py:221 | parameter | verify_by_content: are sdl records physical or normalized, and why are only first/last future-input windows checked?
Q157 | Clean/model_augmentation/fit_systems/closed_loop.py:239 | step | Why select checkpoints by the arithmetic mean of per-record, per-axis NRMS values while training minimizes squared weighted window errors?
Q158 | Clean/model_augmentation/fit_systems/interconnect.py:37 | variable | input_signal_sizes and output_signal_sizes: why do these names include block outputs and block inputs respectively, and what do indices 0 and 1 mean on each side?
Q159 | Clean/model_augmentation/fit_systems/interconnect.py:55 | parameter | obc_pass and pass_mats: what exact structures must callers construct before calling the interconnect?
Q160 | Clean/model_augmentation/fit_systems/interconnect.py:75 | step | Why is connection resolution deferred to the first forward/output_only call, and may blocks or connections be added afterwards?
Q161 | Clean/model_augmentation/fit_systems/interconnect.py:96 | parameter | order: which signal indices are valid, and what dependency guarantees must a caller satisfy?
Q162 | Clean/model_augmentation/fit_systems/interconnect.py:118 | return | output_signals: which entries denote external outputs versus block inputs, and which entries can remain None?
Q163 | Clean/model_augmentation/fit_systems/interconnect.py:124 | variable | n_nodes=4+nr_blocks: what are the four non-block graph nodes and how do their indices map to signal indices?
Q164 | Clean/model_augmentation/fit_systems/interconnect.py:136 | step | Why are output indices 0 and 1 shifted by n_nodes-2 before constructing the dependency graph?
Q165 | Clean/model_augmentation/fit_systems/interconnect.py:142 | step | What does slicing and rolling the directional connection matrix accomplish in the computation-order calculation?
Q166 | Clean/model_augmentation/fit_systems/interconnect.py:148 | parameter | The interconnect graph: must it be acyclic, fully connected and free of algebraic loops before init_forward runs?
Q167 | Clean/model_augmentation/fit_systems/interconnect.py:164 | step | Why does only the first block exposing pass_mats receive per-objective matrices, and is a single physics block a required invariant?
Q168 | Clean/model_augmentation/fit_systems/interconnect.py:196 | step | Why may output_only substitute zero input and None matrices, and which restrictions does this place on output connections?
Q169 | Clean/model_augmentation/fit_systems/interconnect.py:220 | step | Why is the first additive connection promoted to a concatenation when no concatenation exists?
Q170 | Clean/model_augmentation/fit_systems/interconnect.py:235 | step | Why are connection matrices padded in two passes, and what determines the final concatenated signal order?
Q171 | Clean/model_augmentation/fit_systems/interconnect.py:256 | parameter | input_signal and output_signal: which strings or block objects are accepted, and is the direction from an external input/block output to a block input/external output?
Q172 | Clean/model_augmentation/fit_systems/interconnect.py:256 | parameter | connection_function_method=None: what default is selected for each destination, and which explicit strings are supported?
Q173 | Clean/model_augmentation/fit_systems/interconnect.py:256 | parameter | connection_matrix: what are its row/column semantics and required shape for additive versus concatenated mappings?
Q174 | Clean/model_augmentation/fit_systems/interconnect.py:268 | parameter | input_signal_list and output_signal_list: does list order fix block input order, and are outputs copied or summed when several connections share a destination?
Q175 | Clean/model_augmentation/fit_systems/interconnect.py:324 | parameter | data: what is the expected ordering and meaning of its window arrays and appended simulator arrays?
Q176 | Clean/model_augmentation/fit_systems/interconnect.py:333 | setting | The fixed loader seed 0: why is it independent of cfg.seed, and how does it affect fresh versus resumed minibatch sequences?
Q177 | Clean/model_augmentation/fit_systems/interconnect.py:348 | parameter | na, nb, na_right and nb_right: what are the history offsets, is the current sample included, and do nonzero right histories use future observations?
Q178 | Clean/model_augmentation/fit_systems/interconnect.py:360 | step | Which named parameters are included in parameters_with_names, and how are optimizer groups and optimizer type chosen?
Q179 | Clean/model_augmentation/fit_systems/interconnect.py:365 | parameter | yfuture=None: is omitting future measured output supported by the attached closed-loop simulator?
Q180 | Clean/model_augmentation/fit_systems/interconnect.py:370 | return | make_training_data: what are the returned arrays, their order and normalization, and how are histories aligned with future targets?
Q181 | Clean/model_augmentation/fit_systems/interconnect.py:442 | step | Why does an epoch use floor(N_training_samples/batch_size) updates, and are remainder windows discarded or carried into another pass?
Q182 | Clean/model_augmentation/fit_systems/interconnect.py:452 | step | Why validate and permit checkpoint selection before any Adam or PS initialization step?
Q183 | Clean/model_augmentation/fit_systems/interconnect.py:478 | step | Why is an auxiliary PS step placed after each normal objective update rather than completing PS initialization first?
Q184 | Clean/model_augmentation/fit_systems/interconnect.py:516 | step | Why rebuild the OBC basis after loading the best dump without repeating checkpoint selection using the rebuilt model's score?
Q185 | Clean/model_augmentation/fit_systems/interconnect.py:528 | variable | _UNPICKLABLE: what state must the current process reconstruct or retain when loading a dump that lacks obc, training_phase and fit_history?
Q186 | Clean/model_augmentation/fit_systems/interconnect.py:550 | step | Why synchronize the OBC coefficient at validation time without refreshing the expansion point, but force a basis refresh after checkpoint restoration?
Q187 | Clean/model_augmentation/fit_systems/interconnect.py:576 | parameter | theta_aux=None: when is theta_frozen guaranteed to match the current ANN and basis, and which caller ensures that?
Q188 | Clean/model_augmentation/fit_systems/interconnect.py:606 | step | Is the OBC expansion point updated on the first objective of each integer epoch, and what changes when that refresh frequency changes?
Q189 | Clean/model_augmentation/fit_systems/interconnect.py:608 | step | Which gradient paths through the solved coefficient, baseline parameters and frozen basis are retained during loss.backward?
Q190 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:19 | setting | max_iter and inner_iter: how do these limits interact if max_iter is zero or not divisible by inner_iter?
Q191 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:48 | step | Why first increase the window stride and then sample a subset instead of uniformly sampling from all original training windows?
Q192 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:75 | variable | batch.kw['stride']: why does the stored loss keyword retain base_stride after windows were constructed using effective_stride?
Q193 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:119 | step | Why require bitwise-equal objective values from two probes rather than use a tolerance, and are gradients also required to reproduce?
Q194 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:147 | return | The outcome dictionary: what distinguish ran, accepted, deterministic and used_compiled, and what do None, NaN or zero mean after an abort?
Q195 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:158 | step | Why polish every optimizer parameter in one group, and which encoder and baseline parameters remain trainable?
Q196 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:170 | step | Is the OBC expansion point fixed for the entire polish, or can loss calls refresh it during line search?
Q197 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:255 | step | What does a step block using fewer than inner_iter iterations indicate about convergence and the remaining polish budget?
Q198 | Clean/model_augmentation/fit_systems/obc.py:2 | variable | J, Phi, F_base^n and Phi_Gamma: which baseline coordinate, tuple distribution and inner product define the projection space?
Q199 | Clean/model_augmentation/fit_systems/obc.py:21 | parameter | J and vbar: must J always have eleven columns and vbar ten entries, or can this model-independent class handle another parameter count?
Q200 | Clean/model_augmentation/fit_systems/obc.py:28 | variable | r protected physical rows: which rows are protected and what physical dimensions have been normalized out before stacking J?
Q201 | Clean/model_augmentation/fit_systems/obc.py:91 | parameter | step: what exact batch and trailing-dimension shape must this callable return, and must it be free of state mutation for jacfwd?
Q202 | Clean/model_augmentation/fit_systems/obc.py:107 | step | Why detach vbar and build the basis under no_grad while using jacfwd, and which parameter gradients does that exclude during training?
Q203 | Clean/model_augmentation/fit_systems/obc.py:129 | setting | max_consecutive_rank_loss=2: why permit one failed refresh, and how does a retained old basis change the current model's orthogonality constraint?
Q204 | Clean/model_augmentation/fit_systems/obc.py:155 | step | Why move a newly built candidate basis to the old basis's device rather than keep it on the build device?
Q205 | Clean/model_augmentation/fit_systems/obc.py:187 | parameter | learning_component: what network/block interface is expected by the supplied reference_field?
Q206 | Clean/model_augmentation/fit_systems/pre_encoder.py:41 | parameter | na and nb: are unequal histories supported, must they be nonnegative, and what does zero mean?
Q207 | Clean/model_augmentation/fit_systems/pre_encoder.py:65 | step | Why use Xavier initialization for W^a while W^b uses the nn.Linear distribution?
Q208 | Clean/model_augmentation/fit_systems/pre_encoder.py:67 | setting | The Xavier gain and joint draw: what effect do these choices have on the initial added-state scale?
Q209 | Clean/model_augmentation/fit_systems/pre_encoder.py:74 | step | Why does psi_tilde share hidden layers between physical and added states, and which parts can PS initialization subsequently update?
Q210 | Clean/model_augmentation/fit_systems/pre_encoder.py:76 | step | Why initialize the added-state nonlinear rows to zero while retaining random physical rows until prefit?
Q211 | Clean/model_augmentation/fit_systems/pre_encoder.py:84 | parameter | The history arrays: must samples be ordered oldest-to-newest with time-major channel flattening?
Q212 | Clean/model_augmentation/fit_systems/pre_encoder.py:89 | return | The physical and added state coordinates: which normalization and units apply to each part of the returned vector?
Q213 | Clean/model_augmentation/fit_systems/pre_encoder.py:102 | step | Why does the nonlinear map read the histories in the opposite order to chi and the linear maps?
Q214 | Clean/model_augmentation/fit_systems/ps2_opening.py:5 | variable | M=H=n+1: why is the residual prediction horizon tied to output-history length rather than the training rollout horizon?
Q215 | Clean/model_augmentation/fit_systems/ps2_opening.py:22 | setting | PS2Spec defaults: what behavior do these settings assume about sampling rate and record length if they are described as plant-independent?
Q216 | Clean/model_augmentation/fit_systems/ps2_opening.py:24 | setting | lr=1e-3: why is the auxiliary learning rate a hundred times the main Adam rate, and which Adam epsilon/dtype convention do these stages use?
Q217 | Clean/model_augmentation/fit_systems/ps2_opening.py:26 | setting | s1_min and s1_cap: must these lie on the evaluation grid, and how are off-grid limits intended to be interpreted?
Q218 | Clean/model_augmentation/fit_systems/ps2_opening.py:42 | setting | seed: which draws use this seed versus the model's torch seed, including decoder initialization and S2 minibatches?
Q219 | Clean/model_augmentation/fit_systems/ps2_opening.py:53 | step | Why choose the last member of each excitation class for calibration, and does 'last' mean manifest order or the highest numeric suffix?
Q220 | Clean/model_augmentation/fit_systems/ps2_opening.py:54 | step | Why leave calibration records in the main objective V while holding them out of S1 and S2?
Q221 | Clean/model_augmentation/fit_systems/ps2_opening.py:73 | parameter | data, n_xb, n_a and width: which fields and encoder/network structure are required, and is n_a=0 a supported use of this class?
Q222 | Clean/model_augmentation/fit_systems/ps2_opening.py:89 | variable | s1_end, s2 and status: what does None mean in the summary for a screened-out, running or completed opening?
Q223 | Clean/model_augmentation/fit_systems/ps2_opening.py:103 | step | Why keep encoder changes when the predictability screen rejects S2, and how does that differ from starting the plain method?
Q224 | Clean/model_augmentation/fit_systems/ps2_opening.py:149 | step | Why initialize the decoder's output layer and bias to zero, and why use one hidden layer of the augmentation width?
Q225 | Clean/model_augmentation/fit_systems/ps2_opening.py:181 | step | Why exclude an additional M+1 samples when sampling one-step S2 pairs that only require adjacent histories?
Q226 | Clean/model_augmentation/fit_systems/ps2_opening.py:204 | step | Why normalize the entire residual batch with a single RMS rather than the per-record, per-axis servo scales used by V?
Q227 | Clean/model_augmentation/fit_systems/ps2_opening.py:243 | step | Why retain a separate Adam state for shared encoder parameters, and how do these auxiliary updates interact with the main optimizer's moment estimates?
Q228 | Clean/model_augmentation/fit_systems/ps2_opening.py:279 | step | Why allow shared hidden-layer updates to change physical corrections while fitting only the added-state transition?
Q229 | Clean/model_augmentation/fit_systems/ps2_opening.py:288 | variable | The S2 input U: why use the recorded force rather than the corrected closed-loop force used in the main rollout?
Q230 | Clean/model_augmentation/fit_systems/ps2_opening.py:301 | step | Why normalize each S2 minibatch with its own target variance rather than one fixed training-pair variance?
Q231 | Clean/model_augmentation/fit_systems/ps2_opening.py:305 | step | Why select S2 by the left-out state-map cost rather than the main model's validation score?
Q232 | Clean/model_augmentation/systems/gantry_ss.py:1 | variable | Which parts of the simulated plant are represented by theta^0, and which dynamics are intentionally excluded from this baseline?
Q233 | Clean/model_augmentation/systems/gantry_ss.py:8 | setting | _D=torch.float32: why create nominal values and geometric transforms in float32 when the training model uses float64, and which precision is intended for reported truth?
Q234 | Clean/model_augmentation/systems/gantry_ss.py:11 | variable | mb, mh, m1 and m2: which physical components are these masses?
Q235 | Clean/model_augmentation/systems/gantry_ss.py:16 | variable | Jb and Jh: which rotation axes and bodies do these moments of inertia describe?
Q236 | Clean/model_augmentation/systems/gantry_ss.py:19 | variable | cg1, cg2 and cy: which physical dampers and coordinates do they act on, and what are their units?
Q237 | Clean/model_augmentation/systems/gantry_ss.py:23 | variable | cb1 and cb2: which joints do these dampers represent, and are the units N s/m or N m s/rad?
Q238 | Clean/model_augmentation/systems/gantry_ss.py:26 | variable | kb1 and kb2: which springs do these represent, and are they translational or torsional stiffnesses?
Q239 | Clean/model_augmentation/systems/gantry_ss.py:29 | variable | Lb and d: which exact distances do they measure, and where are their geometric origins?
Q240 | Clean/model_augmentation/systems/gantry_ss.py:32 | variable | q_act, q and u: what is the ordered coordinate/force vector, positive direction and mixed-unit convention for each?
Q241 | Clean/model_augmentation/systems/gantry_ss.py:46 | parameter | The inertia parameters: what positivity and nonsingularity conditions are required for the rational inverse to be meaningful at all scheduling positions?
Q242 | Clean/model_augmentation/systems/gantry_ss.py:59 | variable | Y: which payload position and origin schedule the inertia, and what physical range is assumed?
Q243 | Clean/model_augmentation/systems/gantry_ss.py:86 | variable | G, w and the LFR: what is the scheduling block and its signal ordering, and why does this helper return only the state row of G?
Q244 | Clean/model_augmentation/systems/gantry_ss.py:95 | parameter | M1_ and M2_: do they multiply Y and Y^2 respectively, and which units and coordinate ordering do their entries have?
Q245 | Clean/model_augmentation/systems/gantry_ss.py:98 | return | A_x, B_w, B_u and A_combined: do these act on physical generalized forces or stage actuator forces, and are they continuous-time derivative maps?
Q246 | Clean/model_augmentation/utils/deepSI_corrections.py:14 | parameter | cheat_n, normed and dt: what do these System_data constructor arguments mean, and what units and values are accepted?
Q247 | Clean/model_augmentation/utils/deepSI_corrections.py:18 | parameter | na_right and nb_right: which current/future samples do these add to the histories, and how do they change the first eligible window?
Q248 | Clean/model_augmentation/utils/deepSI_corrections.py:19 | parameter | force_multi_u, force_multi_y and online_construct: what do these settings mean in this wrapper, and do they affect its returned arrays?
Q249 | Clean/model_augmentation/utils/torch_nets.py:16 | variable | The decoder: does PS use this shared builder or a separately initialized network, and which initialization guarantees apply to it?
Q250 | Clean/model_augmentation/utils/torch_nets.py:44 | step | Why zero only the physical output weights while leaving added-state dynamics random at initialization?
Q251 | Clean/model_augmentation/utils/torch_nets.py:72 | return | Physical corrections versus added-state updates: are the former increments added to an RK4 state transition and the latter entire next states, and how do both scale with sample time?
```

## List B, noise

```text
N001 | Clean/gantry_interconnect_dynamic.py:22 | repeats-code | "# Evaluation" — adds nothing to the two evaluation settings immediately below.
N002 | Clean/gantry_interconnect_dynamic.py:35 | filler | "folder's run id; nothing moves here.\"\"\"" — adds no return or input contract to the read-only helper.
N003 | Clean/gantry_interconnect_dynamic.py:78 | repeats-code | "# 1. Run folder." — restates the folder setup below.
N004 | Clean/gantry_interconnect_dynamic.py:80 | repeats-code | "# run folder: a fresh run never writes into an existing one" — restates the explicit existence check and error.
N005 | Clean/gantry_interconnect_dynamic.py:99 | repeats-code | "# 2. Data and normalisation." — restates the two calls below.
N006 | Clean/gantry_interconnect_dynamic.py:111 | repeats-code | "# 3. Model and closed loop." — restates build_model and build_closed_loop.
N007 | Clean/gantry_interconnect_dynamic.py:112 | repeats-code | "# seed the model's initial draws" — restates the immediately following seed calls.
N008 | Clean/gantry_interconnect_dynamic.py:122 | repeats-code | "# 4. Encoder start, or the restored checkpoint." — restates the two branches below.
N009 | Clean/gantry_interconnect_dynamic.py:132 | repeats-code | "# 5. Training." — restates train below.
N010 | Clean/gantry_interconnect_dynamic.py:136 | repeats-code | "# 6. Evaluation of the final checkpoint." — repeats the main docstring and evaluate call.
N011 | Clean/gantry_interconnect_dynamic.py:143 | repeats-code | "\"\"\"Parse the command line into a RunConfig and a rerun, then start main.\"\"\"" — restates the function's plainly visible argparse construction and main call.
N012 | Clean/gantry_dynamic/config.py:1 | history | "\"\"\"Run configuration: the five axes of a thesis run, the run environment and the fixed settings.\"\"\"" — ties the configuration to the thesis rather than explaining the axes.
N013 | Clean/gantry_dynamic/config.py:41 | repeats-code | "# config.json keys of the run environment fields, read back by from_config_json" — The configuration-key mapping and its reading function are explicit in _ENV_KEYS and from_config_json.
N014 | Clean/gantry_dynamic/config.py:62 | repeats-doc | "\"\"\"One thesis run: the five axes, the run environment and the fixed settings.\"\"\"" — repeats the module docstring.
N015 | Clean/gantry_dynamic/config.py:68 | repeats-code | "# run environment: production defaults, changed by the CLI flags and by main (start_phase)" — repeats the defaults and callers.
N016 | Clean/gantry_dynamic/config.py:89 | repeats-doc | "# fixed settings of the method; config_json_dict records those the code computes with" — repeats that function's docstring.
N017 | Clean/gantry_dynamic/config.py:127 | repeats-code | "\"\"\"Check the axis and environment values, which arrive from the CLI or a config.json.\"\"\"" — restates the validation below.
N018 | Clean/gantry_dynamic/config.py:159 | repeats-code | "\"\"\"True on the OBC arm.\"\"\"" — restates self.arm == 'obc'.
N019 | Clean/gantry_dynamic/config.py:174 | repeats-doc | "# HEURISTIC: s_j in {-1, +1}, multiplier_j = 1 + 0.1 s_j." — repeats the property's formula.
N020 | Clean/gantry_dynamic/config.py:203 | repeats-code | "\"\"\"stdout of one git command in the repository.\"\"\"" — restates the subprocess call and stdout access.
N021 | Clean/gantry_dynamic/controller.py:67 | repeats-code | "# A = [0 I; -R K -R C], B = [0; R], R = M(Y_op)^-1." — restates the matrix construction directly below.
N022 | Clean/gantry_dynamic/controller.py:83 | repeats-doc | "# Numerator 10 w_b (s + w_b/6)(s + w_b/3), denominator s (s + 3 w_b)(s + 10 w_b)." — The numerator and denominator formula repeats the formula in the same function's docstring.
N023 | Clean/gantry_dynamic/controller.py:101 | repeats-code | "# 1. Plant and controller shape at w_b." — restates the frequency-response calls below.
N024 | Clean/gantry_dynamic/controller.py:107 | repeats-code | "# 2. Gain kappa_j." — restates the scalar gain calculation.
N025 | Clean/gantry_dynamic/controller.py:108 | repeats-doc | "# THEORY: k_j = 1 / |G_jj(jW) Cnorm(jW)|, unit loop gain at the bandwidth (ruleOfThumb.m)" — The unit-loop-gain formula repeats the build_cfb_at docstring.
N026 | Clean/gantry_dynamic/controller.py:110 | repeats-code | "# 3. Tustin at ts." — repeats method='bilinear' and the docstring.
N027 | Clean/gantry_dynamic/controller.py:208 | repeats-code | "# 1. Controller bank over the training and validation records." — restates the concatenation and bank call.
N028 | Clean/gantry_dynamic/controller.py:224 | repeats-code | "# 2. Compiled step." — restates the torch.compile branch.
N029 | Clean/gantry_dynamic/controller.py:233 | repeats-code | "# 3. Loss and validation scales." — restates the scale calculations below.
N030 | Clean/gantry_dynamic/controller.py:236 | repeats-doc | "# Loss weight: W_r = diag(sigma_y / sigma_e,r) per training record." — repeats the build_closed_loop docstring.
N031 | Clean/gantry_dynamic/data.py:22 | history | "# the thesis data sets below cfg.data_root, one folder per noise setting" — adds project provenance to the explicit dataset paths.
N032 | Clean/gantry_dynamic/data.py:24 | repeats-code | "# manifest block -> split" — restates the mapping's keys and values.
N033 | Clean/gantry_dynamic/data.py:26 | repeats-code | "# the anti-aliasing filter B of _decimate_y, designed once" — restates the module-level butter call and its use.
N034 | Clean/gantry_dynamic/data.py:123 | repeats-doc | "# Error-only filtering: y_k = r_k + [B(y - r)]_{k D}; r is unfiltered." — The error-only-filtering equation repeats _decimate_y's docstring.
N035 | Clean/gantry_dynamic/data.py:162 | repeats-doc | "# sigma_e,j = (mean over k >= k0 of (e_j,k - mean e_j)^2)^(1/2)." — The sigma_e standard-deviation formula repeats servo_sigma's docstring.
N036 | Clean/gantry_dynamic/data.py:171 | repeats-doc | "\"\"\"Load the train, val and test records of the manifest split.\"\"\"" — repeats the module and DataBundle descriptions.
N037 | Clean/gantry_dynamic/data.py:178 | repeats-code | "\"\"\"(System_data, servo error) lists of the records of one split.\"\"\"" — restates _load_record and the returned pair of lists.
N038 | Clean/gantry_dynamic/data.py:214 | repeats-code | "# 1. Physical states of the training records." — restates the transformations below.
N039 | Clean/gantry_dynamic/data.py:221 | repeats-doc | "# Positions: q = P^-T y (so the position means are P^-T mu_y)." — and the mean identity repeat the module conventions and normalization docstring.
N040 | Clean/gantry_dynamic/data.py:223 | repeats-code | "# Velocities: qdot_k = (q_k - q_{k-1}) / T_s for k >= 1, qdot_0 = qdot_1." — The backward-difference formula and duplicated first sample restate np.diff and np.vstack immediately below.
N041 | Clean/gantry_dynamic/data.py:229 | repeats-code | "# 2. Statistics." — adds nothing to the mean and std calculations.
N042 | Clean/gantry_dynamic/data.py:230 | repeats-code | "# mu_i = mean(i), sigma_i = (mean (i - mu_i)^2)^(1/2) + 1e-8, i in {x_tilde, u_act, y}." — The mean and population-standard-deviation formula restates the NumPy operations below.
N043 | Clean/gantry_dynamic/data.py:239 | repeats-doc | "# 3. Output map: C^n = T_y P^T [I 0] T_x^-1, with no offset since P^T mu_q = mu_y." — The normalized-output-map formula and zero-offset reason repeat compute_normalization's docstring.
N044 | Clean/gantry_dynamic/evaluation.py:36 | repeats-code | "# axis name in the figures" — restates the TEX lookup's use.
N045 | Clean/gantry_dynamic/evaluation.py:66 | repeats-doc | "theta_base(theta^0), the combinations of the raw parameters theta^0 the data were generated with." — repeats the preceding description and return contract.
N046 | Clean/gantry_dynamic/evaluation.py:236 | repeats-code | "# 1. The run on the CPU: configuration, records and the final checkpoint." — restates the configuration override and loading calls.
N047 | Clean/gantry_dynamic/evaluation.py:266 | repeats-code | "# 2. Per split: the free runs, the scores of both frames, the figures; one simulation per record and model." — repeats the loop and _split_scores docstring.
N048 | Clean/gantry_dynamic/evaluation.py:303 | repeats-code | "# 3. Physical parameters: true, start and learned identifiable combinations." — restates the computations and printed table.
N049 | Clean/gantry_dynamic/evaluation.py:313 | repeats-code | "# 4. Data files and the loss curves." — restates the file-writing calls below.
N050 | Clean/gantry_dynamic/model.py:45 | repeats-code | "# 1. Baseline block and output map." — restates the block construction.
N051 | Clean/gantry_dynamic/model.py:48 | repeats-doc | "# Start: theta^0_base = theta_base(theta^0) on the correct arm, times (1 +- 0.1) per combination on the detuned" — The correct/detuned start formula repeats combo_init_detune's documented rule and the multiplication below.
N052 | Clean/gantry_dynamic/model.py:49 | repeats-doc | "# arm; beta = 0 there." — repeats the build_model docstring's initial beta.
N053 | Clean/gantry_dynamic/model.py:63 | repeats-code | "# 2. Network block phi_aug." — restates Static_ANN_Block construction.
N054 | Clean/gantry_dynamic/model.py:74 | repeats-code | "# 3. Wiring." — restates the connect calls below.
N055 | Clean/gantry_dynamic/model.py:79 | repeats-code | "# one network output per state, in state order" — restates np.arange(nx) and the identity expansion.
N056 | Clean/gantry_dynamic/model.py:87 | repeats-code | "# 4. Model, normaliser and encoder." — restates the constructions below.
N057 | Clean/gantry_dynamic/model.py:105 | repeats-code | "# 5. Optimizer and, on the OBC arm, the projection." — restates init_model and attach_obc.
N058 | Clean/gantry_dynamic/model.py:106 | repeats-code | "# no fallback to the CPU: config.json records the device the run asked for" — repeats the explicit unavailable-CUDA error.
N059 | Clean/gantry_dynamic/model.py:161 | repeats-code | "# 1. Simulate the start model in closed loop, record by record (equal lengths batched)." — restates the grouping and rollout calls.
N060 | Clean/gantry_dynamic/model.py:200 | repeats-code | "# 2. Fit the encoder to the simulated states." — restates least_squares_start below.
N061 | Clean/gantry_dynamic/model.py:211 | repeats-doc | "\"\"\"Least-squares start of the encoder." — repeats the function name and the precise objective that follows.
N062 | Clean/gantry_dynamic/model.py:227 | repeats-code | "# 1. Solve." — adds nothing to the least-squares operations.
N063 | Clean/gantry_dynamic/model.py:240 | repeats-code | "# the SVD driver, for the rank-truncated least-squares fit" — restates driver='gelsd' and the adjacent rcond description for a reader familiar with least squares.
N064 | Clean/gantry_dynamic/model.py:241 | repeats-doc | "# W_s = argmin ||((chi - mu) / s) W_s - (X - mean X)||_F^2." — The standardized least-squares objective repeats the preceding centered-regressor comment and lstsq call.
N065 | Clean/gantry_dynamic/model.py:248 | repeats-code | "# 2. Write into the encoder." — restates the copy and assignment operations.
N066 | Clean/gantry_dynamic/model.py:260 | repeats-code | "# 3. Check: 1 - R_j^2 = mean((pred_j - X_j)^2) / var(X_j) per state; a large value means the linear window" — The unexplained-fraction formula restates the immediately following calculation.
N067 | Clean/gantry_dynamic/model.py:261 | repeats-code | "# map does not reconstruct that state." — restates the meaning of the adjacent R^2 calculation.
N068 | Clean/gantry_dynamic/obc_gantry.py:29 | repeats-code | "# the float64 P^-T: measured positions y to generalised positions q = P^-T y" — and its position transformation restate the inverse immediately below and the module conventions.
N069 | Clean/gantry_dynamic/obc_gantry.py:77 | repeats-code | "# 1. Tuples per record, in physical units." — restates the loop below.
N070 | Clean/gantry_dynamic/obc_gantry.py:80 | repeats-code | "# one training record in float64, resampled as for training" — restates loadmat, _record_at_fs_new and np.asarray.
N071 | Clean/gantry_dynamic/obc_gantry.py:83 | repeats-doc | "# x_tilde_i = (q_i, qdot_i), q_i = P^-T y_i, at i = 2 .. N - 4: the stencil gives qdot at 2 .. N - 3, and" — The tuple equation and index range repeat build_reference_set's docstring.
N072 | Clean/gantry_dynamic/obc_gantry.py:84 | history | "# the last of these is left out to keep the tuple set of the thesis runs" — gives historical compatibility as its only rationale.
N073 | Clean/gantry_dynamic/obc_gantry.py:91 | repeats-code | "# 2. Normalise and stack: x_tilde^n = T_x (x_tilde - mu_x), u^n = T_u (u - mu_u)." — The normalization and stacking formula restates the arithmetic and hstack below.
N074 | Clean/gantry_dynamic/obc_gantry.py:124 | repeats-code | "# every physical row is protected; the network output of state i is column i (one output per state, in order)" — Protecting every physical row and matching its ANN column restate range(nx_phys) and field_cols = row_ix.
N075 | Clean/gantry_dynamic/obc_gantry.py:128 | repeats-doc | "# E: E_ii = 1 on the protected physical rows, 0 elsewhere" — repeats the constructor's buffer description.
N076 | Clean/gantry_dynamic/obc_gantry.py:134 | repeats-doc | "# theta_aux = (a, c): p sensitivity coefficients and the coefficient of the baseline transition" — and its coefficient count repeat the class and constructor contracts.
N077 | Clean/gantry_dynamic/obc_gantry.py:175 | repeats-doc | "# f_base^n(beta_lin) and varphi(beta_lin) a, as the tangent of RK4 along beta direction a (zero state tangent)." — The RK4 tangent explanation repeats tangent_pair's contract and _rk4_with_tangent's documented purpose.
N078 | Clean/gantry_dynamic/obc_gantry.py:178 | repeats-doc | "# Correction on the physical rows: f_obc^n = E (varphi a + c f_base^n)." — The correction equation repeats the forward docstring.
N079 | Clean/gantry_dynamic/obc_gantry.py:183 | repeats-code | "\"\"\"A float64 copy of the physics block for the basis build; the live block keeps the training dtype.\"\"\"" — The float64 copy and unchanged live block restate deepcopy(...).to(float64); the builder below identifies its use.
N080 | Clean/gantry_dynamic/obc_gantry.py:204 | repeats-doc | "\"\"\"J = [Phi  col(f_base^n(beta_lin, z_i))] at beta_lin = vbar64, with its build time.\"\"\"" — The inner builder's J equation and timing contract repeat make_basis_builder's docstring.
N081 | Clean/gantry_dynamic/obc_gantry.py:209 | repeats-doc | "# J = [Phi  F_base^n(beta_lin)]: same span as [Phi Gamma] (module docstring)." — explicitly repeats the module docstring.
N082 | Clean/gantry_dynamic/obc_gantry.py:228 | repeats-code | "# the reference inputs, cached per device and dtype" — restates the cache key and conversion below.
N083 | Clean/gantry_dynamic/obc_gantry.py:232 | repeats-doc | "\"\"\"F^n_aug = col(f^n_aug(theta_aug, (x_tilde^n_i, 0), u^n_i)), sample-major.\"\"\"" — The inner field's stacking equation repeats make_reference_field's docstring.
N084 | Clean/gantry_dynamic/training.py:19 | repeats-code | "\"\"\"Convert NumPy integer and floating values recursively to native values for JSON metadata.\"\"\"" — The recursive NumPy-to-native JSON conversion description restates the straightforward type branches.
N085 | Clean/gantry_dynamic/training.py:86 | filler | "\"\"\"Raise naming the mismatch, rather than load_state_dict's list of keys.\"\"\"" — describes error presentation without adding an input or state contract.
N086 | Clean/gantry_dynamic/training.py:109 | repeats-code | "\"\"\"Refuse a checkpoint whose dtype differs from the run's.\"\"\"" — restates the explicit dtype comparison and error.
N087 | Clean/gantry_dynamic/training.py:154 | repeats-doc | "\"\"\"An empty history, written to hist_path at every entry.\"\"\"" — repeats the class's history contract.
N088 | Clean/gantry_dynamic/training.py:171 | repeats-code | "\"\"\"The history as arrays, one entry per row, for val_hist.npz.\"\"\"" — restates the array comprehensions and filename already documented.
N089 | Clean/gantry_dynamic/training.py:183 | repeats-code | "\"\"\"Validation-probe entry: record the validation that just ran.\"\"\"" — restates self.record('adam', ...).
N090 | Clean/gantry_dynamic/training.py:248 | repeats-code | "# skip fit's validation and checkpoint cycle when the Adam budget is spent" — restates the branch that omits fit.
N091 | Clean/gantry_dynamic/training.py:271 | repeats-code | "# 1. Resume state." — adds nothing to the initialization below.
N092 | Clean/gantry_dynamic/training.py:295 | repeats-code | "# 2. Adam with PS." — restates _adam_phase.
N093 | Clean/gantry_dynamic/training.py:304 | repeats-code | "# 3. Pre-polish checkpoint and the selected row." — restates the save and selection below.
N094 | Clean/gantry_dynamic/training.py:314 | repeats-code | "# 4. Polish and final selection." — restates run_lbfgs_polish and the following choice.
N095 | Clean/gantry_dynamic/training.py:317 | repeats-code | "# final row: the accepted polish, otherwise the rebuilt OBC model, otherwise the selected Adam row" — The final-row priority restates the immediately following conditional expression.
N096 | Clean/gantry_dynamic/training.py:325 | repeats-code | "# 5. Final checkpoint." — restates save_checkpoint_pair below.
N097 | Clean/model_augmentation/__init__.py:1 | history | "\"\"\"Adapted from Model-Augmentation-Public by Jan H. Hoekstra (https://github.com/JanHHoekstra/Model-Augmentation-Public," — The upstream author and repository are provenance already supplied by Clean/README.md.
N098 | Clean/model_augmentation/__init__.py:2 | history | "commit 8200923), BSD 3-Clause License, see LICENSE. Changes to his code are marked # CHANGED, additions @added or" — The upstream revision and attribution scheme repeat the package attribution in Clean/README.md.
N099 | Clean/model_augmentation/__init__.py:3 | history | "__project_origin__.\"\"\"" — This continuation names the project-origin marker already explained in Clean/README.md.
N100 | Clean/model_augmentation/fit_systems/blocks.py:1 | history | "\"\"\"The blocks of the augmented model: Jan's block types and the gantry physics blocks of the baseline." — gives authorship rather than block behavior.
N101 | Clean/model_augmentation/fit_systems/blocks.py:12 | history | "# CHANGED: restricted to the thesis training path; Jan's full code: Model-Augmentation-Public," — is implementation history.
N102 | Clean/model_augmentation/fit_systems/blocks.py:13 | history | "# commit 8200923 (model_augmentation/__init__.py)" — The upstream commit reference is provenance already covered by package attribution.
N103 | Clean/model_augmentation/fit_systems/blocks.py:17 | history | "# CHANGED: the @added marker and gantry_ss for the added gantry blocks; unused imports removed" — The import-change annotation describes additions and removed imports.
N104 | Clean/model_augmentation/fit_systems/blocks.py:23 | filler | "\"\"\"Basic block implementation with variables that need to be defined for interconnect.\"\"\"" — supplies no actual interface contract.
N105 | Clean/model_augmentation/fit_systems/blocks.py:26 | repeats-code | "\"\"\"Store the block input size nz and output size nw.\"\"\"" — restates the two assignments.
N106 | Clean/model_augmentation/fit_systems/blocks.py:36 | repeats-doc | "\"\"\"Network block: phi_aug with a gate per output." — repeats the class docstring.
N107 | Clean/model_augmentation/fit_systems/blocks.py:52 | repeats-code | "# CHANGED: net_kwargs reach the network's constructor (the number of physical rows)" — restates **net_kwargs below.
N108 | Clean/model_augmentation/fit_systems/blocks.py:61 | repeats-doc | "# CHANGED: a gate per output, 1 in training, applied in forward; _without_learned zeroes it to remove the" — The gate's training/evaluation use repeats Static_ANN_Block.forward's docstring.
N109 | Clean/model_augmentation/fit_systems/blocks.py:62 | repeats-code | "# network (persistent=False: not in the state_dict)" — restates the explicit buffer option.
N110 | Clean/model_augmentation/fit_systems/blocks.py:66 | repeats-doc | "\"\"\"Network block of the augmented model." — repeats the class description.
N111 | Clean/model_augmentation/fit_systems/blocks.py:84 | history | "# CHANGED: C and D required and registered as buffers (they follow .to(), not trained); no empty-matrix cases" — The CHANGED line describes the prior optional-matrix interface; the fixed-buffer behavior is already documented.
N112 | Clean/model_augmentation/fit_systems/blocks.py:86 | repeats-code | "\"\"\"C and D as fixed buffers; ny, nx, nu from their shapes.\"\"\"" — restates the assignments and registrations.
N113 | Clean/model_augmentation/fit_systems/blocks.py:100 | repeats-doc | "\"\"\"Output map of the augmented model." — repeats the class docstring.
N114 | Clean/model_augmentation/fit_systems/blocks.py:110 | history | "# CHANGED: no size assertion" — records a removed check.
N115 | Clean/model_augmentation/fit_systems/blocks.py:118 | repeats-code | "\"\"\"w = nonlinear_function(z), defined by the subclass.\"\"\"" — restates the forwarding call.
N116 | Clean/model_augmentation/fit_systems/blocks.py:120 | repeats-code | "\"\"\"Evaluate the subclass's nonlinear map with the matrices of this objective.\"\"\"" — restates the following call.
N117 | Clean/model_augmentation/fit_systems/blocks.py:121 | repeats-code | "# CHANGED: mats, the matrices built once per objective, are passed on to nonlinear_function" — restates the explicit argument.
N118 | Clean/model_augmentation/fit_systems/blocks.py:131 | repeats-doc | "\"\"\"Baseline transition f_base^n." — repeats the class docstring.
N119 | Clean/model_augmentation/fit_systems/blocks.py:148 | repeats-doc | "# True: xdot = (qdot, a) directly; False: through the LFR output equation (see _deriv_with)" — The closed_form flag explanation repeats the constructor argument documentation.
N120 | Clean/model_augmentation/fit_systems/blocks.py:154 | repeats-code | "# 2. Normalisation buffers." — restates the register_buffer calls.
N121 | Clean/model_augmentation/fit_systems/blocks.py:162 | repeats-doc | "\"\"\"Baseline transition f_base^n: one RK4 step of the vector field over T_s." — The RK4 baseline-transition description repeats the GantryStateBlock class docstring.
N122 | Clean/model_augmentation/fit_systems/blocks.py:176 | repeats-code | "# Stages: k1 = h F(x, u), k2 = h F(x + k1/2, u), k3 = h F(x + k2/2, u), k4 = h F(x + k3, u)." — The four RK4 stage equations restate the four immediately following equations.
N123 | Clean/model_augmentation/fit_systems/blocks.py:185 | repeats-doc | "\"\"\"Baseline vector field F." — The generic vector-field heading adds nothing to the precise equation immediately below.
N124 | Clean/model_augmentation/fit_systems/blocks.py:200 | repeats-code | "# 1. Denormalise: x = x^n sigma_x + mu_x, u = u^n sigma_u + mu_u." — The denormalization formula restates the arithmetic below.
N125 | Clean/model_augmentation/fit_systems/blocks.py:207 | repeats-code | "# 2. Net force." — restates the following force computation.
N126 | Clean/model_augmentation/fit_systems/blocks.py:208 | repeats-code | "# Generalised force: u = P u_act." — restates the matrix product and constructor force-transform comment.
N127 | Clean/model_augmentation/fit_systems/blocks.py:211 | repeats-code | "# Net force: fnet = u - K q - C qdot." — restates the immediately following arithmetic.
N128 | Clean/model_augmentation/fit_systems/blocks.py:229 | repeats-code | "# 4. State derivative." — restates the following state-derivative branch.
N129 | Clean/model_augmentation/fit_systems/blocks.py:231 | repeats-doc | "# Closed form: xdot = (qdot, a) (moves results by rounding)." — The closed-form equation and rounding note repeat the constructor and adjacent LFR explanation.
N130 | Clean/model_augmentation/fit_systems/blocks.py:235 | history | "# since a is already the closed form; kept because the thesis results used it." — is historical justification; the equivalence is already stated on the preceding line.
N131 | Clean/model_augmentation/fit_systems/blocks.py:242 | repeats-code | "# 5. Normalise: xdot^n = xdot / sigma_x." — restates the division below.
N132 | Clean/model_augmentation/fit_systems/blocks.py:247 | repeats-doc | "\"\"\"Baseline vector field F and its directional derivative." — The generic directional-derivative heading repeats the precise derivative definition immediately below.
N133 | Clean/model_augmentation/fit_systems/blocks.py:266 | repeats-code | "# 1. Denormalise: x = x^n sigma_x + mu_x, s_x = sx sigma_x; u = u^n sigma_u + mu_u, u = P u_act." — The denormalization and force-transform equations restate the following tensor operations.
N134 | Clean/model_augmentation/fit_systems/blocks.py:272 | repeats-code | "# 2. Net force and its tangent: fnet = u - K q - C qdot, s_fnet = -K s_q - dK q - C s_qdot - dC qdot." — The force and tangent equations restate the following explicit sums.
N135 | Clean/model_augmentation/fit_systems/blocks.py:277 | repeats-code | "# 3. Acceleration a = num / d." — restates the later acceleration division.
N136 | Clean/model_augmentation/fit_systems/blocks.py:278 | repeats-doc | "# Determinant and its tangent: d(Y) = m_h poly(Y), poly = alpha gamma - beta^2 + 2 beta m_h Y" — The determinant polynomial repeats _deriv_with's determinant explanation.
N137 | Clean/model_augmentation/fit_systems/blocks.py:279 | repeats-doc | "# + m_h (alpha - m_h) Y^2 (product rule term by term)." — The remaining determinant term and product-rule description repeat the preceding vector-field calculation and explicit derivatives.
N138 | Clean/model_augmentation/fit_systems/blocks.py:288 | repeats-code | "# s_d = dm_h poly + m_h s_poly." — restates the immediately following tangent product rule.
N139 | Clean/model_augmentation/fit_systems/blocks.py:292 | repeats-code | "# Numerator and its tangent: num = N0 fnet + Y (N1 fnet + Y N2 fnet) (Horner)." — The Horner numerator equation restates num below and the module's Horner convention.
N140 | Clean/model_augmentation/fit_systems/blocks.py:295 | repeats-code | "# s_(N_j fnet) = dN_j fnet + N_j s_fnet; s_num by the product rule on the Horner form." — The matrix-product derivative rule restates the three s_n*f assignments.
N141 | Clean/model_augmentation/fit_systems/blocks.py:303 | repeats-code | "# Acceleration and its tangent: a = num / d, s_a = (s_num - a s_d) / d (quotient rule)." — The acceleration quotient-rule equation restates the following two assignments.
N142 | Clean/model_augmentation/fit_systems/blocks.py:307 | repeats-code | "# 4. State derivative." — adds nothing to the derivative branches.
N143 | Clean/model_augmentation/fit_systems/blocks.py:309 | repeats-code | "# Closed form: (xdot, s_xdot) = ((qdot, a), (s_qdot, s_a))." — The closed-form state/tangent pair restates the two torch.cat operations.
N144 | Clean/model_augmentation/fit_systems/blocks.py:313 | repeats-doc | "# LFR output equation of _deriv_with, differentiated:" — repeats the function's directional-derivative contract.
N145 | Clean/model_augmentation/fit_systems/blocks.py:314 | repeats-code | "# s_xdot = A_combined s_(x, w, u) + dA_combined (x, w, u), s_w = sY z + Y s_z, s_u = 0." — The tangent product-rule equations restate the explicit s_z, s_w and s_xdot_phys calculations.
N146 | Clean/model_augmentation/fit_systems/blocks.py:326 | repeats-code | "# 5. Normalise: xdot^n = xdot / sigma_x, likewise the tangent." — The normalized-derivative division restates the following two divisions.
N147 | Clean/model_augmentation/fit_systems/blocks.py:332 | repeats-doc | "\"\"\"Baseline transition f_base^n and its directional derivative." — The generic transition/tangent heading repeats the precise RK4 tangent description immediately below.
N148 | Clean/model_augmentation/fit_systems/blocks.py:346 | repeats-doc | "# Stages: k_i = h F(x_i), s_i = h DF(x_i)[sx_i, dmats], x_i and sx_i as in _rk4." — The RK4 tangent stage rule repeats _rk4_with_tangent's docstring.
N149 | Clean/model_augmentation/fit_systems/blocks.py:356 | repeats-code | "# Tangent update: sx_next = sx + (s1 + 2 s2 + 2 s3 + s4) / 6." — The tangent RK4 update equation restates the next line verbatim.
N150 | Clean/model_augmentation/fit_systems/blocks.py:395 | repeats-doc | "# Stiffness: K = diag(0, k_b,Sigma, 0)." — The diagonal stiffness equation repeats _build_KC's docstring.
N151 | Clean/model_augmentation/fit_systems/blocks.py:401 | repeats-doc | "# Damping: C of the docstring." — explicitly repeats the docstring.
N152 | Clean/model_augmentation/fit_systems/blocks.py:421 | repeats-doc | "# M_1 = -m_h (e1 e2^T + e2 e1^T)." — The M1 formula repeats _build_M1M2's docstring.
N153 | Clean/model_augmentation/fit_systems/blocks.py:427 | repeats-doc | "# M_2 = m_h e2 e2^T." — The M2 formula repeats _build_M1M2's docstring.
N154 | Clean/model_augmentation/fit_systems/blocks.py:450 | repeats-code | "# d0 = det M_0 = m_h (alpha gamma - beta^2), so M_0^-1 = N0 / d0." — The determinant and N0/d0 relation repeat the scalar formula below and the inverse-inertia contract.
N155 | Clean/model_augmentation/fit_systems/blocks.py:458 | repeats-doc | "\"\"\"Baseline transition f_base^n as a block." — The baseline-transition heading repeats the GantryStateBlock description and the equation below.
N156 | Clean/model_augmentation/fit_systems/blocks.py:494 | repeats-doc | "\"\"\"Baseline block in the training coordinates beta." — The baseline-in-beta heading repeats the class docstring.
N157 | Clean/model_augmentation/fit_systems/blocks.py:504 | repeats-doc | "# 1. beta = 0 at theta^0_base." — repeats the class and constructor docstrings.
N158 | Clean/model_augmentation/fit_systems/blocks.py:511 | repeats-code | "# 2. Calibrate the signed coordinate." — adds nothing to the specific explanation on the next lines.
N159 | Clean/model_augmentation/fit_systems/blocks.py:525 | repeats-doc | "\"\"\"Identifiable combinations theta_base(theta)." — The identifiable-combinations heading repeats the function name and the precise definitions below.
N160 | Clean/model_augmentation/fit_systems/blocks.py:562 | repeats-code | "# m_1 + m_2 = m_Sigma - m_b, J_b + J_h = J_eff - (L_b^2/4)(m_1 + m_2), r = J_b0 / (J_b0 + J_h0)." — The m_sum, J_sum and r equations restate the three assignments below.
N161 | Clean/model_augmentation/fit_systems/blocks.py:567 | repeats-doc | "# Equal splits of k_b,Sigma, c_b,Sigma; m_1,2 = (m_1 + m_2 +- m_Delta) / 2;" — Equal splits and the mass-pair formula repeat gauge_section's docstring and the returned arithmetic.
N162 | Clean/model_augmentation/fit_systems/blocks.py:568 | repeats-code | "# J_b = r J_sum, J_h = (1 - r) J_sum." — The J_b/J_h ratio split restates the returned products below.
N163 | Clean/model_augmentation/fit_systems/blocks.py:575 | repeats-doc | "\"\"\"Bound on |m_Delta|." — The bound heading repeats the precise bound equation below.
N164 | Clean/model_augmentation/fit_systems/blocks.py:589 | repeats-doc | "\"\"\"Parameter map theta_base(beta)." — The parameter-map heading repeats the precise map below.
N165 | Clean/model_augmentation/fit_systems/blocks.py:600 | repeats-doc | "# Positive combinations: theta_base,j = theta^0_base,j exp(beta_j)." — The positive-coordinate exponential map repeats combinations_from_free's docstring.
N166 | Clean/model_augmentation/fit_systems/blocks.py:602 | repeats-doc | "# Signed mass difference: m_Delta = b tanh(eta_0 + s beta_mDelta)." — The signed tanh map repeats combinations_from_free's docstring.
N167 | Clean/model_augmentation/fit_systems/blocks.py:608 | repeats-doc | "\"\"\"Inverse parameter map beta(theta_base)." — The inverse-map heading repeats the precise inverse equations below.
N168 | Clean/model_augmentation/fit_systems/blocks.py:624 | repeats-code | "# Ratio: r = m_Delta / b, |r| < 1 required." — repeats the division and explicit rejection below.
N169 | Clean/model_augmentation/fit_systems/blocks.py:630 | repeats-doc | "# Inverse: beta_j = log(theta_base,j / theta^0_base,j), beta_mDelta = (atanh(r) - eta_0) / s." — The inverse coordinate equations repeat free_from_combinations's docstring.
N170 | Clean/model_augmentation/fit_systems/blocks.py:636 | repeats-doc | "\"\"\"Current parameter estimate." — adds nothing to the precise parameter-reporting sentence below.
N171 | Clean/model_augmentation/fit_systems/blocks.py:646 | repeats-doc | "\"\"\"Baseline transition as a function of beta." — The baseline-in-beta heading repeats the precise transition contract below.
N172 | Clean/model_augmentation/fit_systems/blocks.py:660 | repeats-doc | "\"\"\"Baseline matrices and their directional derivative in beta." — The generic matrices/tangent heading repeats the precise jvp equation below.
N173 | Clean/model_augmentation/fit_systems/closed_loop.py:26 | repeats-doc | "\"\"\"Controller bank, one per distinct design." — repeats the class docstring.
N174 | Clean/model_augmentation/fit_systems/closed_loop.py:46 | repeats-doc | "# THEORY: eps = T_y^-1 e^n and u_fb^n = T_u u_fb give B^n = B diag(sigma_y), C^n = diag(1/sigma_u) C," — The Bn/Cn normalization identities repeat the module conventions; the THEORY label does not add another source.
N175 | Clean/model_augmentation/fit_systems/closed_loop.py:47 | repeats-doc | "# D^n = diag(1/sigma_u) D diag(sigma_y)." — The Dn normalization identity repeats the module conventions.
N176 | Clean/model_augmentation/fit_systems/closed_loop.py:69 | repeats-doc | "# Both rows in one product: (u_fb^n, x^c_{k+1}) = [C^n; A] x^c + [D^n; B^n] e^n." — The fused controller-product equation repeats the constructor's one-product explanation.
N177 | Clean/model_augmentation/fit_systems/closed_loop.py:90 | repeats-doc | "\"\"\"One step: y_hat^n = C^n x_tilde^n, u_hat^n = u^n + u_fb^n, then x_hat^n_{k+1}.\"\"\"" — The inner one-step equation repeats closed_loop_step's docstring.
N178 | Clean/model_augmentation/fit_systems/closed_loop.py:93 | repeats-code | "# Output: y_hat^n = C^n x_tilde^n." — restates output_only and the nearby docstring.
N179 | Clean/model_augmentation/fit_systems/closed_loop.py:95 | repeats-code | "# Feedback on e^n = y^n - y_hat^n, and x^c_{k+1}." — restates bank.step's arguments and returns.
N180 | Clean/model_augmentation/fit_systems/closed_loop.py:97 | repeats-code | "# State update with the model input u_hat^n = u^n + u_fb^n." — restates the next call.
N181 | Clean/model_augmentation/fit_systems/closed_loop.py:145 | repeats-code | "\"\"\"Store the bank, the controller row and W_r per training record, sigma_e per validation record, the step.\"\"\"" — restates the constructor assignments.
N182 | Clean/model_augmentation/fit_systems/closed_loop.py:151 | repeats-code | "# the closed_loop_step compiled by torch.compile, or None for eager" — repeats build_closed_loop and the use_compiled condition.
N183 | Clean/model_augmentation/fit_systems/closed_loop.py:166 | repeats-code | "\"\"\"Run through the uncompiled step inside the block, then restore.\"\"\"" — restates the context-manager assignments.
N184 | Clean/model_augmentation/fit_systems/closed_loop.py:187 | repeats-code | "# use the compiled step for gradient-enabled CUDA rollouts, including the polish" — restates the immediately following condition and __call__ docstring.
N185 | Clean/model_augmentation/fit_systems/closed_loop.py:216 | repeats-code | "# same record order and counts as ctrl_ix, so the checks above cover it" — restates the parallel concatenations above and below.
N186 | Clean/model_augmentation/fit_systems/closed_loop.py:250 | repeats-doc | "# (n_val, ny), metres" — The inline shape and metres note repeats closed_loop_free_run_rms_batch's per-channel return contract.
N187 | Clean/model_augmentation/fit_systems/closed_loop.py:283 | repeats-code | "# 1. Normalise, u^n = (u - mu_u) / sigma_u, y^n = (y - mu_y) / sigma_y, and group records by length." — The normalization equations and grouping description restate the operations below.
N188 | Clean/model_augmentation/fit_systems/closed_loop.py:293 | repeats-code | "# filled with return_pred only" — restates the later conditional assignment.
N189 | Clean/model_augmentation/fit_systems/closed_loop.py:295 | repeats-code | "# the matrices at the current parameters, once for every group and step" — restates the single objective_matrices call outside the group loop.
N190 | Clean/model_augmentation/fit_systems/closed_loop.py:297 | repeats-code | "# 2. Free run per group, then RMS per record." — restates the nested loops.
N191 | Clean/model_augmentation/fit_systems/closed_loop.py:302 | repeats-doc | "# x_hat^n_{k0} = psi(chi_{k0}); the scored samples are k0 .. N - 1" — The encoder start and scored sample range repeat the free-run docstring.
N192 | Clean/model_augmentation/fit_systems/closed_loop.py:311 | repeats-code | "# y_hat = y_hat^n sigma_y + mu_y; RMS_j = (mean_k e_j,k^2)^(1/2), record RMS = (mean_j RMS_j^2)^(1/2)." — The denormalization and RMS equations restate the arithmetic below.
N193 | Clean/model_augmentation/fit_systems/interconnect.py:1 | history | "\"\"\"Jan's interconnect of blocks and his encoder fit system, with the augmented model of the thesis.\"\"\"" — The module description identifies Jan and the thesis rather than adding a behavioral contract beyond the class docstrings.
N194 | Clean/model_augmentation/fit_systems/interconnect.py:2 | history | "# CHANGED: restricted to the thesis training path; Jan's full code: Model-Augmentation-Public," — describes the fork's origin and scope reduction.
N195 | Clean/model_augmentation/fit_systems/interconnect.py:3 | history | "# commit 8200923 (model_augmentation/__init__.py)" — The upstream commit citation repeats package provenance.
N196 | Clean/model_augmentation/fit_systems/interconnect.py:9 | history | "# CHANGED: os, math and get_work_dirs added for the changed fit (the epoch index of the OBC refresh and the checkpoint" — The os/math/get_work_dirs addition annotation records import history.
N197 | Clean/model_augmentation/fit_systems/interconnect.py:10 | history | "# saving); time and deepSI's loader helpers as in Jan's" — is provenance.
N198 | Clean/model_augmentation/fit_systems/interconnect.py:18 | history | "# CHANGED: the @added marker, the blocks the added code looks for (Static_ANN_Block) and the renamed normaliser" — The marker/import/renaming annotation describes source changes.
N199 | Clean/model_augmentation/fit_systems/interconnect.py:19 | history | "# SystemDataNormalizer instead of the parameterised blocks and detect_algebraic_loop" — The old normalizer and removed block/loop-detection names are historical details.
N200 | Clean/model_augmentation/fit_systems/interconnect.py:27 | repeats-code | "\"\"\"An empty interconnect with state size nx, input size nu and output size ny.\"\"\"" — restates initialization.
N201 | Clean/model_augmentation/fit_systems/interconnect.py:43 | repeats-code | "# CHANGED: the OBC correction f_obc^n, set by attach_obc on the OBC arm, else None" — The optional obc_correction assignment and attaching caller are visible in code.
N202 | Clean/model_augmentation/fit_systems/interconnect.py:56 | repeats-doc | "\"\"\"Augmented-model transition." — The generic transition heading adds nothing to the state/output equations immediately below.
N203 | Clean/model_augmentation/fit_systems/interconnect.py:79 | repeats-code | "# CHANGED: the signal loop is _propagate, shared with output_only, on the device and dtype of x" — restates the calls in both methods.
N204 | Clean/model_augmentation/fit_systems/interconnect.py:100 | repeats-code | "# signals outside order stay None" — restates initialization and the loop over order.
N205 | Clean/model_augmentation/fit_systems/interconnect.py:112 | repeats-code | "# the physics block takes the matrices built once per objective" — repeats the branch and argument contract.
N206 | Clean/model_augmentation/fit_systems/interconnect.py:121 | repeats-code | "\"\"\"Resolve the computation order and the connection matrices once.\"\"\"" — restates init_forward's operations and guarded call sites.
N207 | Clean/model_augmentation/fit_systems/interconnect.py:160 | repeats-code | "# CHANGED: the output's dependency cone, resolved once for output_only" — restates _signal_cone(1) and output_only's docstring.
N208 | Clean/model_augmentation/fit_systems/interconnect.py:162 | repeats-code | "# CHANGED: the block that takes the per-objective matrices pass_mats, resolved once" — restates the next expression.
N209 | Clean/model_augmentation/fit_systems/interconnect.py:166 | repeats-code | "# CHANGED: the connection matrices on the module's device and dtype" — restates the conversion below.
N210 | Clean/model_augmentation/fit_systems/interconnect.py:269 | repeats-code | "\"\"\"Connect every signal of input_signal_list into block, and block into every signal of output_signal_list.\"\"\"" — The description of connecting each input and output signal restates the two simple loops.
N211 | Clean/model_augmentation/fit_systems/interconnect.py:284 | repeats-code | "\"\"\"0 for x and xp, 1 for u and y.\"\"\"" — restates the two branches.
N212 | Clean/model_augmentation/fit_systems/interconnect.py:298 | repeats-code | "\"\"\"Store the two signal indices, the method and the connection matrix.\"\"\"" — restates the four assignments.
N213 | Clean/model_augmentation/fit_systems/interconnect.py:327 | repeats-code | "\"\"\"A floating array cast to dtype, other arrays unchanged.\"\"\"" — restates the ternary return.
N214 | Clean/model_augmentation/fit_systems/interconnect.py:332 | repeats-doc | "# the batch order of My_Simple_DataLoader: a generator fixed at seed 0" — The seed-0 generator description repeats DtypeDataLoader.__init__'s docstring.
N215 | Clean/model_augmentation/fit_systems/interconnect.py:354 | repeats-doc | "# CHANGED: a normaliser whose windows keep the data's dtype; build_model sets its values" — The dtype-preserving normalizer description repeats SSE_Interconnect.__init__'s docstring.
N216 | Clean/model_augmentation/fit_systems/interconnect.py:358 | repeats-code | "'''Set nu and ny, build the optimizer over every parameter and reset bestfit.'''" — Setting sizes, building the optimizer and resetting bestfit plainly repeat the short method body.
N217 | Clean/model_augmentation/fit_systems/interconnect.py:366 | repeats-code | "\"\"\"Return (y_pred, x_final) through the attached simulator, with y_pred shaped (batch, nf, ny).\"\"\"" — Forwarding through the attached simulator repeats its single return call; the shape is already documented by the simulator.
N218 | Clean/model_augmentation/fit_systems/interconnect.py:371 | repeats-code | "\"\"\"deepSI's training windows plus the arrays the attached simulator appends per window.\"\"\"" — The superclass windows plus appended simulator arrays restate the two calls.
N219 | Clean/model_augmentation/fit_systems/interconnect.py:377 | repeats-code | "\"\"\"The simulator's validation score, then the validation probes (side effects only).\"\"\"" — The simulator score followed by probe callbacks restates the method body.
N220 | Clean/model_augmentation/fit_systems/interconnect.py:383 | history | "# CHANGED: arguments of the thesis path only (no norm fit, optimizer setup, concurrent validation, timeout," — The list of removed fit arguments describes the prior implementation.
N221 | Clean/model_augmentation/fit_systems/interconnect.py:384 | history | "# scheduler, Dataset loader, verbose levels or time profile); init_model is called by the pipeline" — The remaining removed features and previous setup arrangement are historical context.
N222 | Clean/model_augmentation/fit_systems/interconnect.py:405 | repeats-code | "\"\"\"Validate, save the best and the last dump, and return the validation error.\"\"\"" — Validate, save best/last and return error restates the nested function body.
N223 | Clean/model_augmentation/fit_systems/interconnect.py:407 | repeats-code | "# the score is the simulator's, so no measure is passed" — restates cal_validation_error's implementation.
N224 | Clean/model_augmentation/fit_systems/interconnect.py:419 | repeats-doc | "# the dump holds the tensors on the training device; resolve_checkpoint maps them to the CPU" — The training-device tensors and CPU remapping repeat validation's device note and resolve_checkpoint's explicit map_location.
N225 | Clean/model_augmentation/fit_systems/interconnect.py:427 | repeats-code | "# 1. Windows, data loader and the initial validation." — restates the calls below.
N226 | Clean/model_augmentation/fit_systems/interconnect.py:428 | history | "# CHANGED: counters and history always start empty (Jan's continues an earlier fit)" — The contrast with Jan's continuation behavior is source history; counter resets are explicit below.
N227 | Clean/model_augmentation/fit_systems/interconnect.py:447 | repeats-doc | "# CHANGED: DtypeDataLoader (same seed-0 shuffle), since My_Simple_DataLoader casts the windows to float32" — The dtype loader and seed description repeat DtypeDataLoader's docstrings.
N228 | Clean/model_augmentation/fit_systems/interconnect.py:456 | repeats-doc | "# 2. Adam updates, each followed by the PS step; validation every its_per_val." — Adam/PS/validation ordering repeats fit's docstring.
N229 | Clean/model_augmentation/fit_systems/interconnect.py:463 | repeats-code | "# CHANGED: the simulator's extra arrays (controller row, W_r) reach loss() by name" — The extra arrays reaching loss by name restates dict(zip(...)) below.
N230 | Clean/model_augmentation/fit_systems/interconnect.py:467 | repeats-code | "\"\"\"V and its gradient on the current minibatch.\"\"\"" — restates loss and backward below.
N231 | Clean/model_augmentation/fit_systems/interconnect.py:502 | repeats-code | "# 3. Select: reload the best dump." — restates checkpoint_load_system.
N232 | Clean/model_augmentation/fit_systems/interconnect.py:533 | repeats-code | "\"\"\"Locate the network block and the block that takes pass_mats.\"\"\"" — Locating the ANN and pass_mats blocks restates the two next expressions.
N233 | Clean/model_augmentation/fit_systems/interconnect.py:537 | repeats-doc | "# the network block and the block that takes pass_mats" — The network/pass_mats lookup description repeats the constructor docstring.
N234 | Clean/model_augmentation/fit_systems/interconnect.py:548 | repeats-doc | "\"\"\"Coefficient of the OBC correction for prediction." — The generic coefficient heading repeats the precise coefficient synchronization description below.
N235 | Clean/model_augmentation/fit_systems/interconnect.py:570 | repeats-doc | "\"\"\"Matrices of one objective." — The generic objective-matrices heading repeats the specific tuple construction described below.
N236 | Clean/model_augmentation/fit_systems/interconnect.py:588 | repeats-doc | "\"\"\"Training objective V." — The generic objective heading adds nothing to the precise weighted objective immediately below.
N237 | Clean/model_augmentation/fit_systems/interconnect.py:600 | repeats-code | "# 1. Initial states: x_hat^n_{tau|tau} = psi(theta_enc, chi_tau)." — The initial-state encoder equation restates self.encoder below.
N238 | Clean/model_augmentation/fit_systems/interconnect.py:602 | repeats-code | "# 2. OBC coefficient of this objective." — adds nothing to the branch below.
N239 | Clean/model_augmentation/fit_systems/interconnect.py:612 | repeats-code | "# 3. Closed-loop rollout and cost." — restates the simulation and return below.
N240 | Clean/model_augmentation/fit_systems/interconnect.py:619 | repeats-doc | "# Objective: V = mean (W_r (y^n - y_hat^n))^2." — The objective formula repeats loss's docstring.
N241 | Clean/model_augmentation/fit_systems/interconnect.py:642 | repeats-code | "\"\"\"The state without the training-time keys, as every dump holds it.\"\"\"" — restates the dictionary filter and _UNPICKLABLE contract.
N242 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:17 | filler | "\"\"\"The settings of the polish.\"\"\"" — gives no meaning beyond the class name.
N243 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:42 | repeats-doc | "\"\"\"The windows at the scaled stride, and a seeded subset of n_windows if there are more.\"\"\"" — The scaled-stride and seeded-subset description repeats the FixedWindowBatch class docstring.
N244 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:53 | repeats-doc | "# augment_training_data checks the built windows against window_counts at this stride" — The alignment-check note repeats augment_training_data's contract.
N245 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:67 | repeats-code | "\"\"\"The selected windows of one array, on the model's device and dtype.\"\"\"" — Selecting windows and converting to model dtype/device restates the two following lines.
N246 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:87 | repeats-doc | "\"\"\"Write grad V to the parameters' .grad slots and return a detached scalar V on the fixed batch.\"\"\"" — The nested closure's gradient-and-loss description repeats _make_closure's docstring.
N247 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:104 | repeats-code | "\"\"\"Recognise CUDA OOM exceptions and RuntimeErrors containing 'out of memory'.\"\"\"" — The OOM type/string description restates the short Boolean expression.
N248 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:126 | repeats-code | "\"\"\"(keep, why): keep the polished weights only if V is finite and the validation error drops.\"\"\"" — The finite-loss/improved-validation acceptance description restates the three explicit checks.
N249 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:138 | repeats-doc | "\"\"\"Polish of the selected model." — The generic polish heading repeats the module docstring and precise algorithm description below.
N250 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:141 | filler | "refines the Adam result without risking a worse model." — repeats the preceding rollback criterion with a broader assurance.
N251 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:160 | repeats-code | "# the start weights, for an exact rollback" — restates the detached clones and _restore contract.
N252 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:166 | repeats-code | "\"\"\"The validation error at the current weights, theta_aux synced first.\"\"\"" — The validation and coefficient-sync description restates the calls below.
N253 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:177 | repeats-code | "\"\"\"Store the L-BFGS iteration and closure counts in out.\"\"\"" — Storing the iteration and closure counts restates the two assignments.
N254 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:195 | repeats-code | "# 1. Score to beat and the fixed batch." — restates val_before and FixedWindowBatch below.
N255 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:205 | repeats-code | "\"\"\"A fresh L-BFGS optimizer with the spec's settings, strong-Wolfe line search.\"\"\"" — The fresh optimizer and strong-Wolfe description restates torch.optim.LBFGS below.
N256 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:211 | repeats-code | "# 2. Optimizer and determinism probe." — restates the next calls.
N257 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:212 | repeats-doc | "# determinism probe, compiled first if available: the strong-Wolfe line search needs a loss that" — The compiled-first reproducibility rationale repeats the detailed eager-fallback explanation below.
N258 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:213 | repeats-doc | "# reproduces" — completes a rationale already stated in the fallback explanation below.
N259 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:220 | repeats-code | "# a failed compiled probe is retried once on the eager rollout" — The eager retry after a failed compiled probe restates force_eager and the retry branch.
N260 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:227 | repeats-code | "# the failed attempt's optimizer and allocations go before the eager retry" — Releasing optimizer/closure before eager retry restates the assignment below.
N261 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:254 | repeats-code | "# 3. L-BFGS iterations." — restates the optimization loop.
N262 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:255 | repeats-code | "# allow ceil(max_iter / inner_iter) blocks, stopping early if a block uses fewer than inner_iter iterations" — The ceil block count and short-block stopping restate n_outer and done < spec.inner_iter.
N263 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:273 | repeats-code | "# the objective at the final weights: one forward pass over the batch, without gradient" — The final forward pass without gradients restates torch.no_grad and the loss call.
N264 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:294 | repeats-doc | "# 4. Keep the polish only if the validation error dropped." — Keeping the polish only on improved validation repeats the module and function docstrings.
N265 | Clean/model_augmentation/fit_systems/obc.py:18 | repeats-doc | "\"\"\"J D^-1 = U S V^T, D_ii = ||J_:,i|| (1 for a zero column): the economy SVD of the column-scaled basis, with its" — The scaled-SVD equation repeats the constructor's more complete factorization and rank contract.
N266 | Clean/model_augmentation/fit_systems/obc.py:19 | repeats-doc | "numerical rank.\"\"\"" — repeats the constructor's rank explanation.
N267 | Clean/model_augmentation/fit_systems/obc.py:22 | repeats-doc | "\"\"\"Factorised OBC basis." — repeats the class name and precise formula below.
N268 | Clean/model_augmentation/fit_systems/obc.py:34 | repeats-doc | "# 1. Column scale D = diag(||J_:,i||), 1 for a zero column." — The column-scale definition repeats the constructor docstring.
N269 | Clean/model_augmentation/fit_systems/obc.py:38 | repeats-code | "# 2. Factorise." — adds nothing to the SVD call.
N270 | Clean/model_augmentation/fit_systems/obc.py:39 | repeats-doc | "# J D^-1 = U S V^T." — The scaled-SVD equation repeats the constructor docstring.
N271 | Clean/model_augmentation/fit_systems/obc.py:41 | repeats-code | "# 3. Numerical rank; truncate on rank loss." — Numerical-rank calculation and truncation restate the comparison and slicing below.
N272 | Clean/model_augmentation/fit_systems/obc.py:54 | repeats-doc | "\"\"\"OBC coefficient theta_aux." — The generic coefficient heading repeats the precise coefficient equation immediately below.
N273 | Clean/model_augmentation/fit_systems/obc.py:65 | repeats-doc | "# theta_aux = D^-1 V S^-1 U^T F." — The coefficient equation repeats coefficient's docstring.
N274 | Clean/model_augmentation/fit_systems/obc.py:70 | repeats-code | "\"\"\"Move the factors to device.\"\"\"" — restates the .to calls.
N275 | Clean/model_augmentation/fit_systems/obc.py:78 | repeats-code | "\"\"\"Device of the factors.\"\"\"" — restates self.U.device.
N276 | Clean/model_augmentation/fit_systems/obc.py:85 | repeats-doc | "\"\"\"Stacked baseline sensitivity Phi." — The generic sensitivity heading repeats the precise derivative and stacking definition below.
N277 | Clean/model_augmentation/fit_systems/obc.py:102 | repeats-code | "# 1. Prepare." — is an uninformative stage heading.
N278 | Clean/model_augmentation/fit_systems/obc.py:110 | repeats-code | "# 2. Jacobian per chunk of tuples." — restates the range/chunk loop and jacfwd call.
N279 | Clean/model_augmentation/fit_systems/obc.py:113 | repeats-doc | "# varphi_i = d step(beta, z_i) / d beta at beta_lin, rows row_ix, stacked sample-major." — The sensitivity definition and stacking order repeat build_stacked_sensitivity's docstring.
N280 | Clean/model_augmentation/fit_systems/obc.py:118 | repeats-code | "# 3. Stack, with the optional f_base^n column." — restates cat and append_value below.
N281 | Clean/model_augmentation/fit_systems/obc.py:121 | repeats-doc | "# J = [Phi  col(step(beta_lin, z_i))]." — The appended-value J formula repeats the append_value argument contract.
N282 | Clean/model_augmentation/fit_systems/obc.py:133 | repeats-code | "\"\"\"Store the reference-field evaluator and the basis builder returning (J, seconds).\"\"\"" — Storing two callables restates the two assignments; the builder return is already evident at its invocation.
N283 | Clean/model_augmentation/fit_systems/obc.py:142 | repeats-doc | "\"\"\"Rebuild the OBC basis at beta_lin." — The generic rebuild heading repeats the detailed refresh contract below.
N284 | Clean/model_augmentation/fit_systems/obc.py:150 | repeats-code | "# 1. Candidate basis at beta_lin." — restates the basis construction below.
N285 | Clean/model_augmentation/fit_systems/obc.py:157 | repeats-doc | "# 2. Rank policy: keep the last full-rank basis or abort on rank loss, else accept." — The keep/abort rank policy repeats refresh's docstring and the nearby heuristic.
N286 | Clean/model_augmentation/fit_systems/obc.py:176 | repeats-doc | "\"\"\"Per-epoch refresh of beta_lin." — repeats the class description and precise schedule on the next lines.
N287 | Clean/model_augmentation/fit_systems/obc.py:188 | repeats-doc | "\"\"\"theta_aux of the current network." — repeats the precise coefficient formula below.
N288 | Clean/model_augmentation/fit_systems/pre_encoder.py:1 | repeats-doc | "\"\"\"The encoder of the augmented model." — repeats the AugmentedEncoder class description.
N289 | Clean/model_augmentation/fit_systems/pre_encoder.py:6 | history | "# CHANGED: restricted to the thesis training path; Jan's full code: Model-Augmentation-Public," — is fork history.
N290 | Clean/model_augmentation/fit_systems/pre_encoder.py:7 | history | "# commit 8200923 (model_augmentation/__init__.py)" — The upstream commit reference repeats package provenance.
N291 | Clean/model_augmentation/fit_systems/pre_encoder.py:12 | history | "# CHANGED: the @added marker and the layer stack of the added encoder" — The added marker/layer-stack import annotation records source history.
N292 | Clean/model_augmentation/fit_systems/pre_encoder.py:34 | repeats-doc | "\"\"\"Encoder psi = [W^b; W^a] chi + psi_tilde(chi)." — The encoder decomposition repeats the class docstring.
N293 | Clean/model_augmentation/fit_systems/pre_encoder.py:54 | repeats-code | "# 1. Physical linear rows W^b." — restates Wb_psi_y and Wb_psi_u below.
N294 | Clean/model_augmentation/fit_systems/pre_encoder.py:55 | repeats-doc | "# the windows hold na + 1 output and nb + 1 input samples" — The na+1/nb+1 window lengths repeat the constructor argument contract.
N295 | Clean/model_augmentation/fit_systems/pre_encoder.py:64 | repeats-code | "# 2. Added linear rows W^a (Xavier)." — restates the named parameters and initialization below.
N296 | Clean/model_augmentation/fit_systems/pre_encoder.py:73 | repeats-code | "# 3. Nonlinear correction psi_tilde." — restates build_mlp and self.net below.
N297 | Clean/model_augmentation/fit_systems/pre_encoder.py:82 | repeats-doc | "\"\"\"Encoder psi." — adds nothing to the class and function formulas.
N298 | Clean/model_augmentation/fit_systems/pre_encoder.py:94 | repeats-doc | "# Linear part: (W^b chi, W^a chi)." — The linear-part decomposition repeats the encoder formula and explicit matrix products.
N299 | Clean/model_augmentation/fit_systems/pre_encoder.py:102 | repeats-doc | "# Nonlinear correction: + psi_tilde, which reads (u history, y history), a fixed permutation of chi." — The nonlinear input permutation repeats the module conventions.
N300 | Clean/model_augmentation/fit_systems/ps2_opening.py:46 | repeats-code | "\"\"\"Plateau: len(values) > window and min(values[-window:]) > (1 - tol) min(values[:-window]).\"\"\"" — The plateau inequality restates the next line exactly.
N301 | Clean/model_augmentation/fit_systems/ps2_opening.py:61 | repeats-doc | "# a class is the record name without its trailing number, 'TR-S5' -> 'TR-S'" — The excitation-class suffix-stripping rule repeats calibration_split's docstring.
N302 | Clean/model_augmentation/fit_systems/ps2_opening.py:70 | repeats-doc | "\"\"\"The opening after every Adam update until it ends: S1 fits a decoder of the residual on the added states," — S1 decoder and S2 transition descriptions repeat the module docstring.
N303 | Clean/model_augmentation/fit_systems/ps2_opening.py:71 | repeats-doc | "S2 the network's added-state transition.\"\"\"" — The remainder of the S2 transition description repeats the module docstring.
N304 | Clean/model_augmentation/fit_systems/ps2_opening.py:75 | repeats-code | "# the network outputs of the added states: one output per state, in state order" — The added-state output range restates range(n_xb, n_xb+n_a).
N305 | Clean/model_augmentation/fit_systems/ps2_opening.py:95 | repeats-code | "\"\"\"True once the opening has ended.\"\"\"" — repeats phase == 'S3' and the module's S3 definition.
N306 | Clean/model_augmentation/fit_systems/ps2_opening.py:99 | repeats-doc | "\"\"\"PS step after each Adam step on V." — repeats the class description.
N307 | Clean/model_augmentation/fit_systems/ps2_opening.py:117 | repeats-code | "# 1. One S1 step." — restates _s1_step.
N308 | Clean/model_augmentation/fit_systems/ps2_opening.py:119 | repeats-code | "# 2. On the check grid: nu on the left-out records, then end S1 or continue." — The check-grid description restates the modulo guard, calibration call and end conditions.
N309 | Clean/model_augmentation/fit_systems/ps2_opening.py:214 | repeats-doc | "# Residual over H samples of the closed-loop rollout from x_hat^n_{tau|tau}." — The H-sample residual description repeats _residual's docstring.
N310 | Clean/model_augmentation/fit_systems/ps2_opening.py:216 | repeats-code | "# Scale: e_tau / sigma_e." — repeats the following division and the docstring.
N311 | Clean/model_augmentation/fit_systems/ps2_opening.py:220 | repeats-code | "\"\"\"xbar = psi_a(chi): the n_xbar rows after the physical ones.\"\"\"" — The added-state slice description repeats the explicit slice and documented state ordering.
N312 | Clean/model_augmentation/fit_systems/ps2_opening.py:224 | repeats-doc | "\"\"\"S1 step." — adds nothing to the objective that follows.
N313 | Clean/model_augmentation/fit_systems/ps2_opening.py:232 | repeats-code | "# 1. Sample windows and encode." — restates the next two calls.
N314 | Clean/model_augmentation/fit_systems/ps2_opening.py:235 | repeats-code | "# 2. Frozen residual target." — restates _residual and x.detach().
N315 | Clean/model_augmentation/fit_systems/ps2_opening.py:237 | repeats-code | "# 3. Decoder cost and gradient." — restates the loss and autograd calls.
N316 | Clean/model_augmentation/fit_systems/ps2_opening.py:238 | repeats-doc | "# S1 cost: mean ||p(xbar_{tau|tau}) - e_tau||^2 over the batch, H and the axes." — The S1 cost formula repeats _s1_step's docstring.
N317 | Clean/model_augmentation/fit_systems/ps2_opening.py:240 | repeats-code | "# g = grad_(theta_p, theta^a_enc) V_S1." — The gradient expression restates torch.autograd.grad.
N318 | Clean/model_augmentation/fit_systems/ps2_opening.py:242 | repeats-code | "# 4. S1's own Adam step." — repeats self._opt.step and the detailed comment below.
N319 | Clean/model_augmentation/fit_systems/ps2_opening.py:252 | repeats-doc | "\"\"\"S1 check nu." — adds nothing to the precise calibration equation below.
N320 | Clean/model_augmentation/fit_systems/ps2_opening.py:267 | repeats-doc | "\"\"\"S2: one-step fit of the added-state transition." — repeats the module and class descriptions.
N321 | Clean/model_augmentation/fit_systems/ps2_opening.py:285 | repeats-code | "\"\"\"Pairs (psi(chi_k), u^n_k, psi_a(chi_{k+1})) of n sampled windows.\"\"\"" — The sampled encoder-pair tuple restates _sample and the return below.
N322 | Clean/model_augmentation/fit_systems/ps2_opening.py:289 | repeats-code | "# 1. Fixed pairs from the frozen encoder." — restates the no_grad helper and two sample calls.
N323 | Clean/model_augmentation/fit_systems/ps2_opening.py:293 | repeats-doc | "\"\"\"Mean squared added-state prediction error divided by t_var, the mean sample variance of the target" — The normalized state-map error repeats _fit_added_state_transition's objective.
N324 | Clean/model_augmentation/fit_systems/ps2_opening.py:294 | repeats-doc | "coordinates.\"\"\"" — The target-variance description repeats that objective's denominator and variance convention.
N325 | Clean/model_augmentation/fit_systems/ps2_opening.py:296 | repeats-code | "# 2. Adam on the S2 cost, checked on the left-out pairs." — restates the optimizer and loop.
N326 | Clean/model_augmentation/fit_systems/ps2_opening.py:301 | repeats-code | "# t_var = (1/n_xbar) sum_j var(T_j), var with 1/(B - 1), for the left-out set here and per batch below." — The mean sample-variance equation restates Tc.var(0).mean() and the minibatch variant.
N327 | Clean/model_augmentation/fit_systems/ps2_opening.py:305 | repeats-code | "# the start is a candidate too, so S2 never ends worse than it began on the calibration pairs" — Treating the start as a candidate restates best = r0 and the initial parameter snapshot below.
N328 | Clean/model_augmentation/fit_systems/ps2_opening.py:323 | repeats-code | "# 3. Restore the best parameters." — restates the parameter copy loop.
N329 | Clean/model_augmentation/systems/gantry_ss.py:60 | repeats-doc | "# alpha = m_Sigma + m_h, beta = (L_b/2) m_Delta, gamma = J_eff." — The alpha/beta/gamma definitions repeat the module conventions and build_poly_constants return contract.
N330 | Clean/model_augmentation/systems/gantry_ss.py:67 | repeats-doc | "# Adjugate coefficients of Y^0, Y^1, Y^2." — repeats the rational-inverse equation in the docstring.
N331 | Clean/model_augmentation/systems/gantry_ss.py:105 | repeats-doc | "# R = M_0^-1 = N0 / d0." — The R=N0/d0 identity repeats build_G_matrix_entries's docstring.
N332 | Clean/model_augmentation/systems/gantry_ss.py:112 | repeats-doc | "# A_x = [0 I; -R K -R C], B_w = [0 0; -R M_1 -R M_2], B_u = [0; R]." — The A_x/B_w/B_u equations repeat build_G_matrix_entries's docstring.
N333 | Clean/model_augmentation/utils/deepSI_corrections.py:5 | history | "# CHANGED: Jan's fixed_System_data and fixed_System_data_norm, renamed WindowedSystemData and SystemDataNormalizer;" — The previous class names and renaming annotation are source history.
N334 | Clean/model_augmentation/utils/deepSI_corrections.py:6 | history | "# restricted to the thesis training path; Jan's full code: Model-Augmentation-Public," — The thesis-path restriction and upstream repository reference are provenance.
N335 | Clean/model_augmentation/utils/deepSI_corrections.py:7 | history | "# commit 8200923 (model_augmentation/__init__.py)" — The upstream commit citation repeats package attribution.
N336 | Clean/model_augmentation/utils/deepSI_corrections.py:15 | empty-tag | "\"\"\"deepSI's System_data.\"\"\"" — names a dependency without explaining the constructor arguments.
N337 | Clean/model_augmentation/utils/deepSI_corrections.py:20 | repeats-doc | "\"\"\"Training windows of a record." — The generic window heading adds nothing to the precise history/future slices below.
N338 | Clean/model_augmentation/utils/deepSI_corrections.py:23 | history | "stride (deepSI's loop, for every stride)." — records implementation origin rather than additional window behavior.
N339 | Clean/model_augmentation/utils/deepSI_corrections.py:51 | repeats-code | "\"\"\"Store mu_u, sigma_u, mu_y, sigma_y as u0, ustd, y0, ystd.\"\"\"" — Storing the four normalization fields restates the four assignments and module conventions.
N340 | Clean/model_augmentation/utils/deepSI_corrections.py:58 | repeats-doc | "\"\"\"Normalisation of the records." — repeats the class description and formula below.
N341 | Clean/model_augmentation/utils/deepSI_corrections.py:60 | repeats-doc | "u^n = (u - mu_u) / sigma_u, y^n = (y - mu_y) / sigma_y, record lengths and signal shapes unchanged." — The normalization equation repeats the class docstring; preserved array shapes follow directly from the elementwise operation.
N342 | Clean/model_augmentation/utils/deepSI_corrections.py:69 | repeats-code | "# CHANGED: lists handled record by record; errors name the expected input" — Handling lists record by record and naming expected inputs restates the preceding list branch and following type error.
N343 | Clean/model_augmentation/utils/deepSI_corrections.py:75 | repeats-doc | "# Normalise: u^n = (u - u0) / ustd, y^n = (y - y0) / ystd." — The normalization equation repeats both class and transform docstrings.
N344 | Clean/model_augmentation/utils/torch_nets.py:6 | history | "# CHANGED: restricted to the thesis training path; Jan's full code: Model-Augmentation-Public," — The thesis-path restriction and upstream repository reference are provenance.
N345 | Clean/model_augmentation/utils/torch_nets.py:7 | history | "# commit 8200923 (model_augmentation/__init__.py)" — The upstream commit citation repeats package attribution.
N346 | Clean/model_augmentation/utils/torch_nets.py:11 | history | "# CHANGED: the marker of the added definitions" — The inline CHANGED annotation identifies an added marker import.
N347 | Clean/model_augmentation/utils/torch_nets.py:24 | repeats-code | "(Sequential, output layer)." — Returning Sequential and its final layer restates the construction and return net, final_layer.
N348 | Clean/model_augmentation/utils/torch_nets.py:37 | repeats-doc | "# b_L = 0." — repeats build_mlp's docstring and the zero-bias initialization below.
N349 | Clean/model_augmentation/utils/torch_nets.py:48 | repeats-doc | "\"\"\"Augmentation network phi_aug at its start." — repeats the class description.
N350 | Clean/model_augmentation/utils/torch_nets.py:50 | repeats-doc | "An MLP (build_mlp) whose first nx_phys output rows are f_aug^n, set to zero, and the rest g_aug^n, kept random." — The zero physical/random added output-row initialization repeats the class docstring.
N351 | Clean/model_augmentation/utils/torch_nets.py:59 | repeats-doc | "# the routed outputs are the states in order, so the physical rows are the first nx_phys" — The routed state ordering repeats the module conventions.
N352 | Clean/model_augmentation/utils/torch_nets.py:61 | repeats-doc | "# W^xtilde_L = 0, so f_aug^n = 0 at the start." — The zero physical output-layer equation repeats the class docstring.
N353 | Clean/model_augmentation/utils/torch_nets.py:65 | repeats-doc | "\"\"\"Augmentation network phi_aug." — repeats the class description.
N354 | Clean/model_augmentation/utils/torch_nets.py:67 | repeats-doc | "(f_aug^n, g_aug^n) = phi_aug(x_hat^n, u^n)." — The phi_aug input/output equation repeats the class docstring.
N355 | Clean/model_augmentation/utils/utils.py:1 | repeats-doc | "\"\"\"Small helpers: the @added marker, tensor conversion and the selection and expansion matrices.\"\"\"" — The module's helper inventory repeats the immediately visible function definitions and their docstrings.
N356 | Clean/model_augmentation/utils/utils.py:2 | history | "# CHANGED: restricted to the thesis training path; Jan's full code: Model-Augmentation-Public," — The thesis-path restriction and upstream repository reference are provenance.
N357 | Clean/model_augmentation/utils/utils.py:3 | history | "# commit 8200923 (model_augmentation/__init__.py)" — The upstream commit citation repeats package attribution.
N358 | Clean/model_augmentation/utils/utils.py:9 | history | "\"\"\"Mark a class or function as added by this project, not part of Jan's framework (grep \"@added\").\"\"\"" — The marker docstring only records project/upstream authorship and a grep convention; the decorator returns its argument unchanged.
N359 | Clean/model_augmentation/utils/utils.py:14 | repeats-code | "\"\"\"input as a tensor; a tensor is returned as it is.\"\"\"" — Returning a tensor unchanged or converting input repeats the one-line conditional.
N360 | Clean/model_augmentation/utils/utils.py:15 | history | "# CHANGED: None is no longer passed through (no caller passes it)" — describes a removed behavior.
N361 | Clean/model_augmentation/utils/utils.py:20 | repeats-doc | "\"\"\"Routing matrix E." — adds nothing to the precise routing equation below.
N362 | Clean/model_augmentation/utils/utils.py:27 | repeats-doc | "(n, len(ix)) E." — The returned routing-matrix shape repeats the exact shape given in the same docstring's equation.
N363 | Clean/model_augmentation/utils/utils.py:34 | repeats-doc | "\"\"\"Selection matrix S." — adds nothing to the precise selection equation below.
N364 | Clean/model_augmentation/utils/utils.py:41 | repeats-doc | "(len(ix), n) S." — The returned selection-matrix shape repeats the exact shape given in the same docstring's equation.
```

