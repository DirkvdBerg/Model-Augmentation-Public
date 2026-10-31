# Cold read of the grey-box code (Claude)

Sources: the 24 files of the task and `Clean/README.md` only. 95 questions, 95 noise lines.

| File | Questions | Noise |
|-|-|-|
| Clean/gantry_interconnect_dynamic.py | 9 | 3 |
| Clean/gantry_dynamic/__init__.py | 0 | 0 |
| Clean/gantry_dynamic/config.py | 24 | 8 |
| Clean/gantry_dynamic/controller.py | 4 | 4 |
| Clean/gantry_dynamic/data.py | 10 | 5 |
| Clean/gantry_dynamic/evaluation.py | 2 | 0 |
| Clean/gantry_dynamic/model.py | 12 | 1 |
| Clean/gantry_dynamic/obc_gantry.py | 7 | 4 |
| Clean/gantry_dynamic/training.py | 5 | 1 |
| Clean/model_augmentation/__init__.py | 0 | 0 |
| Clean/model_augmentation/fit_systems/__init__.py | 0 | 0 |
| Clean/model_augmentation/fit_systems/blocks.py | 5 | 27 |
| Clean/model_augmentation/fit_systems/closed_loop.py | 3 | 5 |
| Clean/model_augmentation/fit_systems/interconnect.py | 5 | 9 |
| Clean/model_augmentation/fit_systems/lbfgs_polish.py | 2 | 0 |
| Clean/model_augmentation/fit_systems/obc.py | 1 | 5 |
| Clean/model_augmentation/fit_systems/pre_encoder.py | 1 | 5 |
| Clean/model_augmentation/fit_systems/ps2_opening.py | 2 | 4 |
| Clean/model_augmentation/systems/__init__.py | 0 | 0 |
| Clean/model_augmentation/systems/gantry_ss.py | 1 | 5 |
| Clean/model_augmentation/utils/__init__.py | 0 | 0 |
| Clean/model_augmentation/utils/deepSI_corrections.py | 2 | 3 |
| Clean/model_augmentation/utils/torch_nets.py | 0 | 4 |
| Clean/model_augmentation/utils/utils.py | 0 | 2 |

## List A, questions

```
Q001 | Clean/gantry_interconnect_dynamic.py:28 | return | _dumps: in which folder does deepSI's get_work_dirs()['checkpoints'] put the _best and _last dumps, what sets that folder, and why do they live outside the run folder?
Q002 | Clean/gantry_interconnect_dynamic.py:40 | variable | gantry_ckpt_<run_id>_pre_lbfgs (.pt and .npz): which weights does it hold (the Adam checkpoint selected by validation, or the last update), and how does it differ from the deepSI _best dump?
Q003 | Clean/gantry_interconnect_dynamic.py:119 | parameter | e_train, e_val: what are they (servo errors y - r per record, in metres?), and what does the closed loop use them for?
Q004 | Clean/gantry_interconnect_dynamic.py:129 | return | load_checkpoint returns meta: which keys does it hold, and what does train do with each?
Q005 | Clean/gantry_interconnect_dynamic.py:130 | variable | resume = (archived source path, meta): what does train use the path for?
Q006 | Clean/gantry_interconnect_dynamic.py:133 | return | bestfit: which score is it (mean validation NRMS_e over records and axes?), is lower better, and is it the score after the polish or of the Adam selection?
Q007 | Clean/gantry_interconnect_dynamic.py:138 | step | evaluate(sdir): which checkpoint is "the final checkpoint" (polished if accepted, else the selected Adam one?), and the evaluation rebuilds the model on the CPU after a CUDA run: does that change the numbers?
Q008 | Clean/gantry_interconnect_dynamic.py:144 | setting | reduce-overhead compilation: does it change the results against eager or the cpu flag (bit-identical, or rounding), or only speed and memory?
Q009 | Clean/gantry_interconnect_dynamic.py:155 | setting | the `from` flag, best or last: when should a resume start from last instead of the already selected best?
Q010 | Clean/gantry_dynamic/config.py:16 | setting | DATA_ROOT and OUT_ROOT sit outside Clean/ (Thesis-writeup/Data, simulations/): when Clean/ is shipped on its own, where are the data sets expected, and are they included?
Q011 | Clean/gantry_dynamic/config.py:20 | setting | AA filter formula: is fs the 20 kHz simulation rate or the 4 kHz training rate, "error" of what (deviation of the magnitude from 1?), and why are 300 Hz and 1 kHz the frequencies quoted?
Q012 | Clean/gantry_dynamic/config.py:22 | setting | AA_FC_HZ = 1600 is fixed while the comment ties it to 0.8 of the fs_new Nyquist frequency: must it be changed by hand when fs_new changes (fs_new = 2000 would put fc above Nyquist)?
Q013 | Clean/gantry_dynamic/config.py:25 | setting | AA_TRIM_S: is it trimmed at both ends of every record also on the noise-free set, where no filter runs? "0.01 of the in-band noise": which band, and what goes wrong at 0?
Q014 | Clean/gantry_dynamic/config.py:27 | setting | PRESETS: rows R8 to R12 are absent; is any other combination of the axes valid (OBC with n_a = 8, ps2 with n_a = 0), or only these nine?
Q015 | Clean/gantry_dynamic/config.py:30 | setting | ps2, "PS2 opening": what is it, what does the 2 stand for, and what does ps2 = True do with n_a = 0?
Q016 | Clean/gantry_dynamic/config.py:30 | setting | start 'correct' versus 'detuned': does 'correct' mean the exact data-generating baseline parameters (evaluation.py calls them 'true')?
Q017 | Clean/gantry_dynamic/config.py:30 | setting | n_a: why only 0, 2 or 8 (line 128); what breaks with another value?
Q018 | Clean/gantry_dynamic/config.py:67 | setting | noise 'lowpass': does it name the noise colouring of the data set or the anti-aliasing low-pass applied in data.py?
Q019 | Clean/gantry_dynamic/config.py:71 | setting | truncate_s: what is it (each record cut to its first truncate_s seconds?), what is it for, and does None mean no truncation?
Q020 | Clean/gantry_dynamic/config.py:73 | setting | epochs = 200: what is one epoch (one pass over all training windows at stride 10?), and how many Adam updates is that?
Q021 | Clean/gantry_dynamic/config.py:74 | setting | n_its: what is it, how does it relate to epochs, and what does None mean?
Q022 | Clean/gantry_dynamic/config.py:76 | setting | its_per_val: training.py uses n_its as its_per_val when n_its is set; is its_per_val then ignored and validation run only at the end?
Q023 | Clean/gantry_dynamic/config.py:79 | setting | lbfgs_max_iter versus lbfgs_inner_iter: is max_iter the total over all step() calls, and what does a different inner_iter change (convergence test, memory)?
Q024 | Clean/gantry_dynamic/config.py:81 | setting | lbfgs_windows = 16384: how are they drawn from the training windows, and do fewer windows only save memory or also change the result?
Q025 | Clean/gantry_dynamic/config.py:91 | setting | nx_phys = 6: which six states, in which order and units?
Q026 | Clean/gantry_dynamic/config.py:92 | setting | nu = ny = 3: which three inputs and outputs, in which order and frame?
Q027 | Clean/gantry_dynamic/config.py:99 | setting | stride = 10: in samples at fs_new? What changes with stride 1 (ten times more windows, longer epochs)?
Q028 | Clean/gantry_dynamic/config.py:101 | setting | na_nb = 30: 30 past samples of both y and u, excluding the current sample? What happens if it is shorter (the encoder start R^2 check)?
Q029 | Clean/gantry_dynamic/config.py:105 | setting | up_sample = 1: what is it?
Q030 | Clean/gantry_dynamic/config.py:109 | setting | lr = 1e-5: one rate for the network, the encoder and the physical training coordinates beta together?
Q031 | Clean/gantry_dynamic/config.py:114 | setting | lbfgs_tol_grad, lbfgs_tol_change: what is "the servo-weighted objective" (V of interconnect.py?), relative to which loss scale were 2.6e-5 and 2.6e-7 chosen, and do they need changing for other data?
Q032 | Clean/gantry_dynamic/config.py:145 | step | convert_run.py of clean-up-verification/ is not in Clean/: what is the CODE key of config.json (line 250), which other values exist, and how do I read a run folder written by other code?
Q033 | Clean/gantry_dynamic/config.py:166 | return | combo_init_detune: which ten parameters get the multipliers, in which order?
Q034 | Clean/gantry_dynamic/controller.py:20 | variable | mb, mh, m1, m2, Jb, Jh, cg1, cg2, cy, cb1, cb2, kb1, kb2, Lb, d: why a second copy of the nominal parameters instead of gantry_ss, and must the two copies stay equal?
Q035 | Clean/gantry_dynamic/controller.py:20 | variable | gtd_config.m, getss.m, gtd_build_plant.m and ruleOfThumb.m are cited but not in Clean/ (the README names tdg_config.m): where are they?
Q036 | Clean/gantry_dynamic/controller.py:43 | parameter | Y_op, the design Y: how is a record's design Y chosen (its operating Y?), and how many distinct designs does the bank hold?
Q037 | Clean/gantry_dynamic/controller.py:125 | parameter | gain: which records have a stored gain other than 1, and why must use_gain be passed explicitly?
Q038 | Clean/gantry_dynamic/data.py:3 | variable | q and y: which generalised coordinates are q and which stage positions are y, with units (q = (X, Theta, Y) appears only in obc_gantry.py and evaluation.py)?
Q039 | Clean/gantry_dynamic/data.py:23 | setting | DATASETS, "Coulomb-tanh-and-MSD": which dynamics are in the data but not in the baseline, i.e. what is the augmentation expected to learn?
Q040 | Clean/gantry_dynamic/data.py:25 | variable | test-I and test-E: what do I and E stand for, and why are both merged into one test split?
Q041 | Clean/gantry_dynamic/data.py:36 | step | K1 records only: which other controllers appear in the manifest, and what are those records for?
Q042 | Clean/gantry_dynamic/data.py:58 | variable | val_ckpt_data: what does "ckpt" mean (the validation set of checkpoint selection?), and why is there no System_data_list for test?
Q043 | Clean/gantry_dynamic/data.py:77 | variable | Dd_np: is D always zero, and why is it carried through Norm and the output block?
Q044 | Clean/gantry_dynamic/data.py:112 | step | why is only the servo error y - r filtered and r passed unfiltered?
Q045 | Clean/gantry_dynamic/data.py:120 | step | why is the noise-free set point-sampled without an anti-aliasing filter?
Q046 | Clean/gantry_dynamic/data.py:130 | step | u_k is the mean over [k T_s, (k + 1) T_s) and y_k the sample at k T_s: is that the intended alignment (u_k acts after y_k), and do the encoder and the model assume it?
Q047 | Clean/gantry_dynamic/data.py:224 | step | velocities for mu_x, sigma_x by a backward difference of the measured y: on the lowpass set, does the noise dominate sigma of the velocities?
Q048 | Clean/gantry_dynamic/evaluation.py:66 | step | 'true' = params_init = the gantry_ss nominal values: are these exactly the data generator's parameters (gantry_ss stores float32), and does 'true' silently change if gantry_ss is edited?
Q049 | Clean/gantry_dynamic/evaluation.py:245 | step | start_phase='lbfgs': passed only to skip restoring the Adam moments?
Q050 | Clean/gantry_dynamic/model.py:31 | variable | beta in "beta = 0 and f_aug^n = 0": the training coordinates of blocks.py, or the mass coupling (L_b/2) m_Delta, both called beta there?
Q051 | Clean/gantry_dynamic/model.py:56 | parameter | params_init (14 raw nominal values) and combo_init (10 combinations): why both, and which parts of the raw nominal enter the model?
Q052 | Clean/gantry_dynamic/model.py:78 | step | connect_block_signals and connect_signals with "additive" and "concat": what does each call build, and what does an empty output list mean?
Q053 | Clean/gantry_dynamic/model.py:90 | step | fit_sys.norm is set by hand before init_model: does init_model or fit recompute it from the data?
Q054 | Clean/gantry_dynamic/model.py:117 | step | why fit_sys.hfn.to(dtype) again after init_model, when the blocks were already cast?
Q055 | Clean/gantry_dynamic/model.py:157 | step | _sync_obc_coefficient() before the start simulation: what does it set (the basis at the start beta, theta_aux of the start network?), and why does the start simulation need it?
Q056 | Clean/gantry_dynamic/model.py:159 | return | objective_matrices(): what does step_kw hold?
Q057 | Clean/gantry_dynamic/model.py:173 | step | qdot_0 = 0 at the first (trimmed) sample: is the stage at rest there in every record?
Q058 | Clean/gantry_dynamic/model.py:185 | variable | recorder.output_only: why must the wrapper carry hfn.output_only?
Q059 | Clean/gantry_dynamic/model.py:194 | step | the encoder start fits windows of the simulated y_hat and closed-loop u_hat, not the measured y and recorded u the encoder reads in training: why, and what does that imply on measured data?
Q060 | Clean/gantry_dynamic/model.py:225 | return | least_squares_start returns W and b, which the caller ignores: who uses them?
Q061 | Clean/gantry_dynamic/model.py:258 | step | b - enc(0, 0): after the physical rows' weight and bias are zeroed, what non-zero term remains in enc(0, 0)[:nx] that b must cancel?
Q062 | Clean/gantry_dynamic/obc_gantry.py:4 | variable | vbar = beta_lin, varphi, Phi_Gamma, Gamma: notation defined outside the code; what is beta_lin (the expansion point of beta, refreshed every epoch?), and what do a and c of theta_aux stand for?
Q063 | Clean/gantry_dynamic/obc_gantry.py:35 | variable | Z_step and Z_field: why two copies of the tuples, one with and one without the xbar = 0 columns?
Q064 | Clean/gantry_dynamic/obc_gantry.py:59 | step | the tuples are taken at xbar = 0: what does the orthogonality guarantee when the added states are non-zero during the rollout?
Q065 | Clean/gantry_dynamic/obc_gantry.py:59 | step | why are the tuples built from the measured (noisy) y and differentiated velocities rather than from model states, and from every sample of every training record?
Q066 | Clean/gantry_dynamic/obc_gantry.py:86 | step | why a fourth-order central difference here and a first-order backward difference in compute_normalization (data.py:224)?
Q067 | Clean/gantry_dynamic/obc_gantry.py:119 | variable | theta_frozen: "frozen" in what sense (a detached copy for prediction, overwritten at every objective)?
Q068 | Clean/gantry_dynamic/obc_gantry.py:258 | variable | "space=affine, refresh=epoch": what is the affine space here, and which other options are there?
Q069 | Clean/gantry_dynamic/training.py:59 | variable | ps2_opening = (phase, updates): what are the phases S1, S2 and S3?
Q070 | Clean/gantry_dynamic/training.py:87 | variable | coordinate_version: which coordinates, what was version 1, and when does it change?
Q071 | Clean/gantry_dynamic/training.py:203 | step | "one eager validation completes the lazy initialisation": which lazy initialisation, and why only on the unconstrained arm?
Q072 | Clean/gantry_dynamic/training.py:238 | parameter | n_xb = cfg.nx_phys: does xb mean the physical (baseline) states, while xbar elsewhere means the added states?
Q073 | Clean/gantry_dynamic/training.py:302 | step | done_epochs = hp['epochs'] also when fit stopped early (NaN loss, KeyboardInterrupt): what is this step for, and can such a run still be resumed?
Q074 | Clean/model_augmentation/fit_systems/blocks.py:4 | variable | alpha, beta, gamma_ (mass terms), beta = free_params (training coordinates) and gamma = M_DIFF_RHO (margin): which beta and which gamma does each later docstring mean (e.g. lines 478, 496, 591)?
Q075 | Clean/model_augmentation/fit_systems/blocks.py:234 | variable | the LFR of the 'lfr' form, z = (a, Y a), w = Y z: which LPV-LFR representation is this, and where is it defined?
Q076 | Clean/model_augmentation/fit_systems/blocks.py:365 | variable | PARAM_NAMES: which physical element is each of kb1, kb2, cg1, cg2, cy, cb1, cb2, mh, m1, m2, mb, Jb, Jh, d, and in which units?
Q077 | Clean/model_augmentation/fit_systems/blocks.py:439 | variable | "one objective" (matrices built once per objective): one minibatch loss V, one validation, or one rollout?
Q078 | Clean/model_augmentation/fit_systems/blocks.py:491 | setting | COORDINATE_VERSION = 2: what was version 1, and when must it be raised?
Q079 | Clean/model_augmentation/fit_systems/closed_loop.py:124 | step | x^c_tau = 0 at every window start: why may the model's controller state be taken equal to the system's there, when the system's controller state is not in the data?
Q080 | Clean/model_augmentation/fit_systems/closed_loop.py:204 | setting | defaults nf = 25, stride = 1: when do they apply?
Q081 | Clean/model_augmentation/fit_systems/closed_loop.py:254 | step | the "free run" feeds the measured y into the residual controller (e = y - y_hat): in what sense is it a free run, and how much of the measurement noise reaches the score?
Q082 | Clean/model_augmentation/fit_systems/interconnect.py:37 | variable | input_signal_sizes, output_signal_sizes: input and output from whose side (input_signal_sizes receives block.nw, the block's output size)? Likewise input_signal and output_signal of connect_signals.
Q083 | Clean/model_augmentation/fit_systems/interconnect.py:124 | step | init_forward: what do the 4 + nr_blocks nodes, the shift of signal indices 0 and 1 to the end and the roll by 2 represent?
Q084 | Clean/model_augmentation/fit_systems/interconnect.py:220 | step | why does the first additive connection become a concatenation when a signal has no concatenation?
Q085 | Clean/model_augmentation/fit_systems/interconnect.py:348 | parameter | na_right, nb_right: what are they, and are they ever non-zero here?
Q086 | Clean/model_augmentation/fit_systems/interconnect.py:460 | step | the batch order comes from a seed-0 generator, independent of cfg.seed: does every run see the same batch sequence, and is it reshuffled every epoch?
Q087 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:159 | step | does the OBC basis refresh during the polish, or stay at the beta of the selected checkpoint while beta moves?
Q088 | Clean/model_augmentation/fit_systems/lbfgs_polish.py:248 | variable | "relative to the opening loss": the loss at the start of the polish, unrelated to the PS2 opening?
Q089 | Clean/model_augmentation/fit_systems/obc.py:178 | step | beta_lin is fixed for an epoch while beta changes every update: how far from orthogonal does the corrected field get within an epoch, i.e. what does a per-epoch refresh trade against a per-update one?
Q090 | Clean/model_augmentation/fit_systems/pre_encoder.py:36 | step | W^a is Xavier (non-zero), so the added states start non-zero, while psi_tilde's added rows start at zero: what is the non-zero start for?
Q091 | Clean/model_augmentation/fit_systems/ps2_opening.py:37 | setting | s2_check = 100: its comment describes the plateau rule; is s2_check the check cadence in S2 steps?
Q092 | Clean/model_augmentation/fit_systems/ps2_opening.py:154 | variable | _M = fs.na: why is the S1 residual horizon M = H equal to the encoder window n + 1?
Q093 | Clean/model_augmentation/systems/gantry_ss.py:1 | variable | nominal parameters in float32 while the model runs in float64: does the float32 rounding (e.g. of L_b = 0.725) reach the results, and is it intended?
Q094 | Clean/model_augmentation/utils/deepSI_corrections.py:14 | parameter | x and cheat_n: x is accepted but not passed to System_data; what are x and cheat_n?
Q095 | Clean/model_augmentation/utils/deepSI_corrections.py:18 | parameter | force_multi_u, force_multi_y, online_construct: accepted but unused; are they ignored on purpose?
```

## List B, noise

```
N001 | Clean/gantry_interconnect_dynamic.py:22 | filler | "# Evaluation" heads two settings whose names and comments already say it
N002 | Clean/gantry_interconnect_dynamic.py:80 | repeats-code | "a fresh run never writes into an existing one" is what the FileExistsError below says
N003 | Clean/gantry_interconnect_dynamic.py:112 | repeats-code | "seed the model's initial draws" restates np.random.seed and torch.manual_seed
N004 | Clean/gantry_dynamic/config.py:72 | empty-tag | "of the thesis runs" says where the value came from, not what it does
N005 | Clean/gantry_dynamic/config.py:75 | empty-tag | "of the thesis runs" says where the value came from, not what it does
N006 | Clean/gantry_dynamic/config.py:96 | empty-tag | "of the thesis runs" says where the value came from, not what it does
N007 | Clean/gantry_dynamic/config.py:98 | empty-tag | "of the thesis runs" says where the values came from, not what they do
N008 | Clean/gantry_dynamic/config.py:102 | empty-tag | "of the thesis runs" says where the values came from, not what they do
N009 | Clean/gantry_dynamic/config.py:106 | empty-tag | "of the thesis runs" says where the values came from, not what they do
N010 | Clean/gantry_dynamic/config.py:159 | repeats-code | "True on the OBC arm." restates self.arm == 'obc'
N011 | Clean/gantry_dynamic/config.py:174 | repeats-doc | the HEURISTIC line repeats the docstring formula of line 166
N012 | Clean/gantry_dynamic/controller.py:67 | repeats-code | A, B formula restates the np.block lines below
N013 | Clean/gantry_dynamic/controller.py:83 | repeats-doc | numerator and denominator repeat the docstring of line 77
N014 | Clean/gantry_dynamic/controller.py:108 | repeats-doc | the kappa_j formula repeats the docstring of line 92
N015 | Clean/gantry_dynamic/controller.py:236 | repeats-doc | W_r formula repeats the docstring of line 195
N016 | Clean/gantry_dynamic/data.py:123 | repeats-doc | error-only filtering repeats the docstring of lines 111-112
N017 | Clean/gantry_dynamic/data.py:162 | repeats-doc | the sigma_e formula repeats the docstring of line 153
N018 | Clean/gantry_dynamic/data.py:221 | repeats-doc | q = P^-T y and the position means repeat the docstring of line 204
N019 | Clean/gantry_dynamic/data.py:230 | repeats-code | mu_i, sigma_i formula restates lines 232-237
N020 | Clean/gantry_dynamic/data.py:239 | repeats-doc | C^n formula and "no offset" repeat the docstring of lines 203-204
N021 | Clean/gantry_dynamic/model.py:235 | repeats-code | W = W_s / s, b = mean(X) - mean(chi) W restates lines 243-244
N022 | Clean/gantry_dynamic/obc_gantry.py:84 | history | "to keep the tuple set of the thesis runs" says how earlier runs did it
N023 | Clean/gantry_dynamic/obc_gantry.py:128 | repeats-doc | E_ii = 1 on the protected rows repeats the Buffers line 119
N024 | Clean/gantry_dynamic/obc_gantry.py:178 | repeats-doc | f_obc^n formula repeats the docstring of line 164
N025 | Clean/gantry_dynamic/obc_gantry.py:209 | repeats-doc | same span as [Phi Gamma] repeats the module docstring, as the line itself says
N026 | Clean/gantry_dynamic/training.py:317 | repeats-code | the final-row rule restates the conditional expression of lines 318-319
N027 | Clean/model_augmentation/fit_systems/blocks.py:12 | repeats-doc | restriction and Jan's commit repeat README Attribution and model_augmentation/__init__.py
N028 | Clean/model_augmentation/fit_systems/blocks.py:17 | history | "unused imports removed" describes earlier code
N029 | Clean/model_augmentation/fit_systems/blocks.py:84 | repeats-doc | buffers, not trained, repeats the class docstring of line 83; "no empty-matrix cases" is history
N030 | Clean/model_augmentation/fit_systems/blocks.py:110 | history | "no size assertion" describes removed code
N031 | Clean/model_augmentation/fit_systems/blocks.py:121 | repeats-doc | repeats the docstring of line 120
N032 | Clean/model_augmentation/fit_systems/blocks.py:148 | repeats-doc | repeats the closed_form argument of line 141
N033 | Clean/model_augmentation/fit_systems/blocks.py:176 | repeats-code | the four stage formulas restate lines 177-180
N034 | Clean/model_augmentation/fit_systems/blocks.py:200 | repeats-code | the denormalisation formula restates lines 201-202
N035 | Clean/model_augmentation/fit_systems/blocks.py:211 | repeats-code | fnet formula restates lines 212-214 (and the module docstring)
N036 | Clean/model_augmentation/fit_systems/blocks.py:242 | repeats-code | xdot^n = xdot / sigma_x restates line 243 and the docstring of line 195
N037 | Clean/model_augmentation/fit_systems/blocks.py:266 | repeats-code | the denormalisation formulas restate lines 267-270
N038 | Clean/model_augmentation/fit_systems/blocks.py:272 | repeats-code | fnet and s_fnet formulas restate lines 273-275
N039 | Clean/model_augmentation/fit_systems/blocks.py:288 | repeats-code | s_d formula restates line 290
N040 | Clean/model_augmentation/fit_systems/blocks.py:292 | repeats-code | num in Horner form restates line 299
N041 | Clean/model_augmentation/fit_systems/blocks.py:303 | repeats-code | a and s_a formulas restate lines 304-305
N042 | Clean/model_augmentation/fit_systems/blocks.py:309 | repeats-code | restates lines 310-311
N043 | Clean/model_augmentation/fit_systems/blocks.py:326 | repeats-code | restates lines 327-328
N044 | Clean/model_augmentation/fit_systems/blocks.py:346 | repeats-doc | stage formulas repeat the docstring of line 334
N045 | Clean/model_augmentation/fit_systems/blocks.py:356 | repeats-code | tangent update restates line 357
N046 | Clean/model_augmentation/fit_systems/blocks.py:395 | repeats-doc | K = diag(0, k_b,Sigma, 0) repeats the docstring of line 382
N047 | Clean/model_augmentation/fit_systems/blocks.py:401 | repeats-doc | "C of the docstring" points back to the docstring
N048 | Clean/model_augmentation/fit_systems/blocks.py:421 | repeats-doc | M_1 formula repeats the docstring of line 413
N049 | Clean/model_augmentation/fit_systems/blocks.py:427 | repeats-doc | M_2 formula repeats the docstring of line 413
N050 | Clean/model_augmentation/fit_systems/blocks.py:600 | repeats-doc | exp map repeats the docstring of line 591
N051 | Clean/model_augmentation/fit_systems/blocks.py:602 | repeats-doc | tanh map repeats the docstring of lines 591-592
N052 | Clean/model_augmentation/fit_systems/blocks.py:624 | repeats-code | r = m_Delta / b and the check restate lines 625-626
N053 | Clean/model_augmentation/fit_systems/blocks.py:630 | repeats-doc | inverse map repeats the docstring of line 610
N054 | Clean/model_augmentation/fit_systems/closed_loop.py:69 | repeats-doc | the one-product step repeats line 51 and the docstring of line 58
N055 | Clean/model_augmentation/fit_systems/closed_loop.py:93 | repeats-code | y_hat^n = C^n x_tilde^n restates hfn.output_only(x) and the docstring of line 90
N056 | Clean/model_augmentation/fit_systems/closed_loop.py:95 | repeats-code | restates bank.step(xc, y_t - y_model, ctrl)
N057 | Clean/model_augmentation/fit_systems/closed_loop.py:97 | repeats-code | restates hfn(x, u_t + u_fb, ...) and the docstring of line 90
N058 | Clean/model_augmentation/fit_systems/closed_loop.py:311 | repeats-code | the denormalisation and RMS formulas restate lines 312-316
N059 | Clean/model_augmentation/fit_systems/interconnect.py:2 | repeats-doc | restriction and Jan's commit repeat README Attribution and model_augmentation/__init__.py
N060 | Clean/model_augmentation/fit_systems/interconnect.py:9 | history | which imports were added for which change (lines 9-10)
N061 | Clean/model_augmentation/fit_systems/interconnect.py:18 | history | which imports replaced which removed ones (lines 18-19)
N062 | Clean/model_augmentation/fit_systems/interconnect.py:332 | repeats-doc | seed-0 batch order repeats the docstring of line 325
N063 | Clean/model_augmentation/fit_systems/interconnect.py:383 | history | list of removed fit arguments (lines 383-384)
N064 | Clean/model_augmentation/fit_systems/interconnect.py:428 | history | "Jan's continues an earlier fit" describes the original code
N065 | Clean/model_augmentation/fit_systems/interconnect.py:447 | repeats-doc | repeats the DtypeDataLoader docstring of line 322
N066 | Clean/model_augmentation/fit_systems/interconnect.py:537 | repeats-doc | repeats the docstring of line 533
N067 | Clean/model_augmentation/fit_systems/interconnect.py:619 | repeats-doc | V formula repeats the docstring of line 590
N068 | Clean/model_augmentation/fit_systems/obc.py:34 | repeats-doc | column scale D repeats the docstring of line 24
N069 | Clean/model_augmentation/fit_systems/obc.py:39 | repeats-doc | J D^-1 = U S V^T repeats the docstrings of lines 18 and 24
N070 | Clean/model_augmentation/fit_systems/obc.py:65 | repeats-doc | theta_aux formula repeats the docstring of line 56
N071 | Clean/model_augmentation/fit_systems/obc.py:113 | repeats-doc | varphi_i definition repeats the docstring of line 87
N072 | Clean/model_augmentation/fit_systems/obc.py:121 | repeats-doc | J = [Phi col(...)] repeats the append_value argument of line 97
N073 | Clean/model_augmentation/fit_systems/pre_encoder.py:6 | repeats-doc | restriction and Jan's commit repeat README Attribution and model_augmentation/__init__.py
N074 | Clean/model_augmentation/fit_systems/pre_encoder.py:12 | history | why these imports were added
N075 | Clean/model_augmentation/fit_systems/pre_encoder.py:55 | repeats-doc | window sizes repeat the na, nb argument of line 41
N076 | Clean/model_augmentation/fit_systems/pre_encoder.py:94 | repeats-code | "Linear part" restates lines 95-96
N077 | Clean/model_augmentation/fit_systems/pre_encoder.py:102 | repeats-doc | the fixed permutation repeats the module docstring of line 3
N078 | Clean/model_augmentation/fit_systems/ps2_opening.py:214 | repeats-doc | residual over H samples repeats the docstring of line 203
N079 | Clean/model_augmentation/fit_systems/ps2_opening.py:216 | repeats-code | "Scale: e_tau / sigma_e" restates line 217
N080 | Clean/model_augmentation/fit_systems/ps2_opening.py:238 | repeats-doc | S1 cost repeats the docstring of line 226
N081 | Clean/model_augmentation/fit_systems/ps2_opening.py:240 | repeats-code | "g = grad V_S1" restates line 241
N082 | Clean/model_augmentation/systems/gantry_ss.py:2 | empty-tag | "kamtin-fp-model/03 Simulink gantry/main.m (the immutable ground truth)" names a file not in Clean/ and says nothing about the values
N083 | Clean/model_augmentation/systems/gantry_ss.py:10 | repeats-doc | the THEORY source repeats the module docstring of lines 1-2
N084 | Clean/model_augmentation/systems/gantry_ss.py:60 | repeats-doc | alpha, beta, gamma repeat the Returns line 57 and the module docstring
N085 | Clean/model_augmentation/systems/gantry_ss.py:105 | repeats-doc | R = M_0^-1 = N0 / d0 repeats the docstring of line 89
N086 | Clean/model_augmentation/systems/gantry_ss.py:112 | repeats-doc | A_x, B_w, B_u formulas repeat the docstring of line 89
N087 | Clean/model_augmentation/utils/deepSI_corrections.py:5 | history | Jan's class names and the rename (lines 5-7)
N088 | Clean/model_augmentation/utils/deepSI_corrections.py:69 | history | describes the change, not the behaviour
N089 | Clean/model_augmentation/utils/deepSI_corrections.py:75 | repeats-doc | normalisation formula repeats the docstring of line 60
N090 | Clean/model_augmentation/utils/torch_nets.py:6 | repeats-doc | restriction and Jan's commit repeat README Attribution and model_augmentation/__init__.py
N091 | Clean/model_augmentation/utils/torch_nets.py:11 | filler | "the marker of the added definitions" adds nothing to the import
N092 | Clean/model_augmentation/utils/torch_nets.py:37 | repeats-code | "b_L = 0." restates line 38 and the docstring of line 19
N093 | Clean/model_augmentation/utils/torch_nets.py:61 | repeats-doc | W^xtilde_L = 0 repeats the class docstring of line 44
N094 | Clean/model_augmentation/utils/utils.py:2 | repeats-doc | restriction and Jan's commit repeat README Attribution and model_augmentation/__init__.py
N095 | Clean/model_augmentation/utils/utils.py:15 | history | "None is no longer passed through" describes earlier code
```
