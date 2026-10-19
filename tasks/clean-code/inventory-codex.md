# Independent static inventory for the final gantry method

Date: 2026-10-19. Scope: handoff `tasks/handoffs/2026-10-19-clean-code-inventory-codex.md`, step 1 only. No training, simulation, tracing, git writes, or edits to existing files. The Claude inventory and coverage directory were not read. The only created file is this inventory. Source references below refer to the current working tree, including its existing uncommitted changes, not an assumed clean commit.

Method: read the entry's override block, launcher and run table; propagate the resulting configuration through the data loader, model builder, controller, training loop, OBC and polish; compare the thesis and its referenced appendix; inspect evaluation consumers. AST enumeration is used only to check definition coverage and list source locations, never as an execution trace. All class and function definitions in the four requested locations, including nested definitions, appear in the symbol ledger. A class being required does not make all its methods required.

Terminology: **required** means executed method, necessary construction, selection, serialization, or load compatibility. **not needed, diagnostic** includes functions that actually execute today but are explicitly excluded by the handoff. **interface** means needed by the existing evaluation consumers even when not part of training. Keeping or relocating such an interface is a later design decision, not proposed here. Imported definitions whose bodies are never called are not method dependencies. Error guards are distinguished from successful-path operations.

## Part A. Required-code inventory

### A1. Exact campaign and shared configuration

`Thesis-writeup/Code/Server/tables/runs.tsv:88` to its final row 96 contain run IDs, not physical line numbers, 221 to 230 with 228 absent. `submit.sh:54` resolves the table and `:113` onward selects IDs by tier; `greybox/run_thesis.sbatch:42` selects an ID, `:44` reads the eleven columns, `:45` to `:50` export them, `:93` invokes the entry. Submission uses the launcher's export-all setting, so unrelated inherited experimental environment variables must be absent for this spec. The lowpass branch below is repeated with noise set to noisefree, making 18 configuration cases, not a Cartesian product of every listed axis.

| Run ID | Evaluation label | Arm | Start | Added states | PS2 | Required paths |
| :- | :- | :- | :- | :- | :- | :- |
| 221 | R1 | obc | detuned | 2 | 1 | ALL, OBC, PS, DET |
| 222 | R2 | obc | correct | 2 | 1 | ALL, OBC, PS |
| 223 | R3 | u | correct | 2 | 0 | ALL |
| 224 | R4 | u | detuned | 2 | 1 | ALL, PS, DET |
| 225 | R5 | u | correct | 2 | 1 | ALL, PS |
| 226 | R6 | u | correct | 0 | 0 | ALL, ZERO |
| 227 | R7 | u | correct | 8 | 1 | ALL, PS |
| 229 | R13 | obc | detuned | 0 | 0 | ALL, OBC, ZERO, DET |
| 230 | R14 | u | detuned | 0 | 0 | ALL, ZERO, DET |

ALL means every row, on each noise mode. OBC means 221, 222, 229; PS means 221, 222, 224, 225, 227; ZERO means 226, 229, 230; DET means 221, 224, 229, 230. GPU means every production row, since device=cuda and compile_mode=reduce-overhead; CPU eager is useful for the later equivalence harness but THESIS_DEVICE/THESIS_SMOKE are not method axes.

The successful path is:

1. Entry `:312` to `:379`: validate/export choices; independent NumPy seed-1 detuning generator if DET; replace CFG using `_THESIS_FIXED`, then row overrides. Fixed init is live, arch mlp, seed 1, horizon 0.1 s.
2. `main:382`: run directory; seed NumPy and Torch; `load_datasets`; `compute_normalization`; resolve hp without resume; config write; repeat both seeds; `build_model`.
3. `build_model:60`: reduced trainable physics at nominal or detuned combinations, fixed output, one live-g tanh MLP, all-state routing, composed fit system, data_lsq encoder, Adam initialization on CPU; attach OBC on OBC rows.
4. Entry `:428`: `build_closed_loop` on the 18 training and 6 validation records; compile both hfn and output_only for the training rollout. `prefit_physical_encoder` runs the baseline closed loop from measured positions, zero velocities, zero added states and zero controller state; `least_squares_start` installs affine physical encoder rows.
5. `train_model_with_diagnostics:821`: assign run ID, no resume; attach PS if PS; install validation history/probes; `train_model:467` calls composed fit, then base fit. Initial eager validation, shuffled full batches of weighted windows with full BPTT, per-epoch OBC refresh, PS after each Adam step, validation every 1300 updates, last-on-tie best checkpoint, restore best and rebuild OBC basis.
6. Save pre-polish checkpoint. `run_lbfgs_polish:753` re-scores OBC after rebuild. `lbfgs_polish:549` uses a fixed sampled/coarsened batch, a fixed OBC basis, fresh differentiable coefficient per loss, reproducibility checks, strong-Wolfe L-BFGS, validation-only acceptance with finite final loss, restore and re-sync on rejection. Save final component weights and metadata.
7. Existing entry evaluates/reports/saves and executes state/gradient diagnostics after training. These diagnostics are excluded from the deliverable. Their interfaces and numerical side effects are covered in Part C.

No resume, DATASET_AB, LBFGS_AB_FREEZE, OBC_ARM, ANN_REZERO_GATE, THESIS_NITS, THESIS_SMOKE, THESIS_DEVICE, GANTRY_KX_ART or GANTRY_KY_ART override belongs to the spec. Loading checkpoints for evaluation does belong to the interface. Error handling for unavailable GPU, invalid config, rank loss, finite checks, failed polish and incompatible checkpoint remains behavior to account for; a static audit cannot say those guards never trigger on an actual run.

### A2. Branch decisions and mathematical dependencies

| File and source | Required branch and call path | Not needed for this spec |
| :- | :- | :- |
| `gantry_dynamic/data.py:79` | Manifest order, block mapping training/validation/test-I/test-E; ignores frf. Both manifests contain 18 train, 6 val, 12 test and 12 frf rows. | Legacy fixed lists as data choices; other datasets. E5 is loaded but default controller policy excludes it from reported scores. |
| `data.py:152` | D=5 force block mean in float64, discard incomplete hold; `_record_at_fs_new:185` aligns all fields, trims 20 samples from each end after decimation. | D=1 and truncate_s branches. |
| `data.py:172` | Lowpass: order-8 SOS Butterworth 1600 Hz at 20 kHz on y-r, forward/backward, then point sample and add r. Noisefree: point sample y directly. | Original noisy mode; no fresh noise injection because snr=None. |
| `data.py:222` | Population std of e=y-r over k0=30 onward; record-axis positive finite checks; controller weights=ystd/sigma_e. | RMS(e) denominator, per-window weights, oracle states as method inputs. |
| `data.py:329` | Concatenate training arrays in manifest order. Logical position from inverse of stored P transpose; first differences times fs for velocity, prepend first difference. Mean/std and +1e-8 scales. | Stored truth states for normalization; analytic linearization scaling. |
| `model.py:81` | joint_estimation=True, reduced coordinates; detune combos only; flag_loss_reg=False; admissibility guard. | Raw 14-parameter training, fixed physical model, parameter prior, soft orth penalty. |
| `model.py:136` | tanh, live_g_feed_forward_nn; route all 6+n_a rows. With n_a=0 the complete final layer is zero but its constructor still consumes random draws. | Identity activation, zero init variant as deployed g, paper ResNet, ReZero, Lipschitz cap. |
| `model.py:179` | ANN input is [x_b,x_a,u], physical input [x_b,u]; fixed C/D block; ANN and physics summed into next state; output uses physical positions only. | Learned output matrix, limited routing, continuous augmentation, separate added-dynamics wrappers. |
| `model.py:246` | data_lsq uses class argument na=nb=29, which means 30 history samples; SSE na=nb=30, right extensions zero. Wb random draw occurs before overwrite by least squares; Wa joint Xavier draw; nonlinear hidden layers default PyTorch, output added rows zero. | Reconstructability map and mean-restoration offsets, default learned encoder, linear-only encoder. |
| `model.py:352` | Prefit at declared theta0 via pipeline rollout; groups equal-length records; collects pre-step physical states and corrected inputs; stride 10 simulated history. | Gradient pretraining, true-state fitting, old 20-epoch data_fit budget, model-based Wb warm start. |
| `model.py:415` | CPU double gelsd solve on standardized centred regressors [flatten(y), flatten(u)], with intercept and rcond=eps64*max(Phi.shape); maps solution back. Retains Wa and hidden weights; physical output weights zero, bias carries intercept. Readiness R2>=0.999 and exact-map check. | Ridge fitting, no-intercept solve, QR default solver, nonlinear prefit. |
| `controller.py:72` | Rigid nominal plant at design_Y=0, independent of trained/detuned physics; bandwidth 100 Hz; rule-of-thumb per-axis gains; bilinear conversion at 4 kHz; tf2ss block bank. | Changing controller with learned physics or scheduling it at instantaneous Y. |
| `controller.py:198` | Load stored controller metadata, register by basename; K1 at gain1 for train/val; deduplicate sorted (designY,gain) bank keys. | Legacy RECORD_Y_OP lookup as method dataset choice; alternate FBW; controller_gain=True. |
| `controller.py:330` | Compile BOTH hfn and hfn.output_only, inductor, fullgraph=True, reduce-overhead. | Other compile modes/backends; replacing hfn by compiled object for validation. |
| `blocks.py:791` | One normalized-state RK4 step per Ts, physical denormalization inside derivatives; input held; Y varies at RK stages. Closed rational solve feeds explicit z,w and LFR A_combined product. | Frozen-Y method path; direct mass-matrix solve as replacement; up_sample>1. Frozen-Y remains evaluation interface. |
| `blocks.py:1304` | Coordinate version2; nine positive combos exponential; signed m_diff tanh bound with rho=.99 and eta0/scale from start; gauge reconstruction feeds physical matrices. | Raw log parameter clamp, free-coordinate redefinition, treating m_diff as positive. |
| `closed_loop.py:216` | Output first, controller residual next, corrected force then state step. Zero residual-controller state at each window start. Bank physical states with input/output normalization folded into matrices. | Plant feedthrough algebraic solve; open-loop simulation; truncated BPTT; chunked gradient checkpoint branch because chunk=0. |
| `interconnect.py:942` | Four window arrays plus int64 ctrl_ix and loss_w. Dtype_DataLoader, seed fixed at 0, full-batch count floor N/512, remainder dropped. 200 epochs means 200*floor(N/512) updates. | Dataset/DataLoader branch, concurrent validation, timeout, extra measures, scheduler-specific run changes. |
| `interconnect.py:1383` | Hoist pass_mats once per objective; compose weighted MSE; param_loss is called but returns 0. OBC path additionally solves reference coefficient with gradient and threads tangent pair. | Parameter regularization term beyond adding zero; orth/traj penalties and optimizer wrappers. |
| `obc_gantry.py:106` | OBC only: fourth-order centred velocities, k=2..N-4 inclusive, successor q[k+1],qd[k-1], added states zero, all training records and stride1. | Legacy endpoint without successor; optional reference stride; diagnostic true states/absorber as mathematical inputs. Current loader still reads those fields for diagnostics. |
| `obc_gantry.py:319` | OBC only: float64 pure transition jacfwd in chunks16384; row-major sample then physical-row stack; append fbar column, NOT gamma column in deployed coordinates. | Tangent-only space; gamma-coordinate coefficient serialization. Same protected span as [J,gamma]. |
| `obc.py:191` | Thin SVD of column-normalized [J,fbar], relative rank cutoff=max(rows,cols)*eps64, expected rank11; one failed refresh retains previous valid basis, second consecutive failure aborts; no valid initial basis aborts. | Continuing on permanently truncated basis, normal-equation inverse. |
| `obc.py:501` | Basis at first need and epoch change (floor(epoch_counter+1e-6)); vbar detached. Differentiable SVD-coefficient application through ANN field; no differentiation through basis construction. Prediction solves under no_grad and stores frozen theta/vbar. | StaticPassBuffers false; fixed-initial-basis experiment; jvp correction alternative use_jvp=False. |
| `obc_gantry.py:250` | Per-pass mats and their JVP; per-step hand-coded primal+tangent RK4. Affine corr=J*theta_first10+alpha*fbar; extend correction with zeros in added rows. | Per-step jacfwd/JVP alternative; projecting g_aug; subtracting a continuous derivative before RK4. |
| `ps2_opening.py:59` | PS only; hold out last record in each stripped class with >=3 members. Main V still uses all records. S1 after Adam update, separate optimizer only Wa, encoder final layer, temporary head; target is detached current residual with batch RMS scale. | PS on n_a=0; PS on rows223/226/229/230; physical encoder hidden-layer update by auxiliary optimizer. |
| `ps2_opening.py:101` | Check every25; plateau after100 with last4, relative.02; cap350; screen150 and threshold.90; fallback retains S1 encoder changes. Both success and fallback are reachable data-dependent paths. | Assuming every spec run reaches S2 or skipping calibration when noisefree. |
| `ps2_opening.py:213` | Conditional S2: frozen 8192 fit/2048 cal snapshot pairs; recorded u_k; ANN-only Adam at.001, batch512; cal every100, min500, cap8000, last5 plateau at.01. | Teacher forcing main loss; rebuilding S2 targets every update; resetting main Adam moments. |
| `lbfgs_polish.py:258` | All rows: coarsen stride by floor(total/desired) then choose up to16384 windows without replacement, sort indices; single accumulation chunk. | Random subset directly from all stride10 windows; chunk/full-data/freezing experiment branches. |
| `lbfgs_polish.py:641` | Eager no_grad warmup, twice-backprop closure reproducibility check, try compiled first on GPU then eager if needed; fixed full parameter group, strong_wolfe lr1,20 inner steps,100 maximum, history50, tolerances2.6e-5/2.6e-7. | Blind unconditional polish; using Adam eps as L-BFGS tolerance; restoring only parameters without frozen OBC re-sync. |
| `lbfgs_polish.py:778` | Final loss finite and strictly lower validation score. meters_gate=False means no oracle/prior meter veto. All rejection/rollback/error paths affect final weights and must be represented. | Meter-gated acceptance; time cap, accept-without-validation, forced eager as declared method. |

### A3. Every RunConfig field

Values below are resolved from dataclass defaults, the entry CFG constructor and THESIS replacements in that order. A field may be constant and only read for logging/guards; that is distinguished from an active method setting. Because config_json_dict serializes every nonexempt field, almost none is literally never read. Here 'inactive' means its alternative behavior is never reached, not that its attribute is never accessed.

| Source | Field | Spec value / variation | Use |
| :- | :- | :- | :- |
| `config.py:63` | `mode` | `varies: thesis_taf_lowpass_noise / thesis_taf_noisefree` | Active data path, filter branch and output family. |
| `config.py:68` | `encoder_init` | `'data_lsq'` | Active construction and least-squares prefit. |
| `config.py:70` | `ann_activation` | `'tanh'` | Active tanh branch. |
| `config.py:74` | `ann_arch` | `'mlp'` | Selector read: mlp only. |
| `config.py:75` | `g_init` | `'live'` | Live output-row initialization. |
| `config.py:77` | `wa_encoder_init` | `'xavier_uniform_gain1'` | Joint Xavier Wa initialization. |
| `config.py:78` | `joint_estimation` | `True` | Reduced physical block trained. |
| `config.py:80` | `physics_parameterization` | `'reduced'` | Reduced block selector. |
| `config.py:92` | `param_prior` | `False` | Read by param_loss: returns zero. |
| `config.py:93` | `param_rmse_baseline` | `0.01` | Read to construct Lambda_combo; objective strength inactive. |
| `config.py:97` | `param_init_detune` | `None` | Guard and metadata; raw-detuning branch inactive. |
| `config.py:102` | `combo_init_detune` | `varies: None (correct); seed-1 signed 10% factors (DET)` | Multiplies nominal combos before coordinate setup. |
| `config.py:105` | `snr` | `None` | Read as None; no added-noise RNG or noise floor. |
| `config.py:106` | `seed` | `1` | Both main seed resets; local detuning/probe/polish/PS generators. |
| `config.py:112` | `closed_loop` | `True` | Active simulator attachment and objective. |
| `config.py:116` | `controller_gain` | `False` | Report/evaluation policy only; default skips nonunit-gain E5. |
| `config.py:119` | `fs_orig` | `20000` | 20k data/filter rate. |
| `config.py:120` | `fs_new` | `4000` | 4k model rate. |
| `config.py:121` | `stride` | `10` | Training and prefit stride10; not OBC reference stride1. |
| `config.py:124` | `truncate_s` | `None` | Inactive smoke branch; no truncation. |
| `config.py:125` | `use_f64` | `True` | Select dtype; does not prevent earlier float32 rounding, Part C. |
| `config.py:126` | `save_flag` | `True` | Writes configuration/checkpoints/results; base fit also always writes deepSI checkpoints. |
| `config.py:127` | `nf_probe_print` | `True` | Diagnostic only; setting False only suppresses prints, not probe execution. |
| `config.py:148` | `device` | `'cuda'` | CPU construction then CUDA prefit/fit/polish/evaluation transitions. |
| `config.py:159` | `checkpoint_chunk` | `0` | Read; value0 selects uncheckpointed full BPTT. |
| `config.py:180` | `compile_mode` | `'reduce-overhead'` | Active GPU compilation for hfn and output_only. |
| `config.py:183` | `nx_ann` | `varies: 0 / 2 / 8` | Active model dimension, network size, optimizer or polish setting passed downstream. |
| `config.py:188` | `ann_route_ix` | `varies: tuple(range(6 + nx_ann))` | Active model dimension, network size, optimizer or polish setting passed downstream. |
| `config.py:189` | `n_nodes_per_layer` | `16` | Active model dimension, network size, optimizer or polish setting passed downstream. |
| `config.py:190` | `n_hidden_layers` | `2` | Active model dimension, network size, optimizer or polish setting passed downstream. |
| `config.py:191` | `up_sample` | `1` | One RK4 step per sample. |
| `config.py:192` | `batch_size` | `512` | Active model dimension, network size, optimizer or polish setting passed downstream. |
| `config.py:193` | `lr` | `1e-05` | Active model dimension, network size, optimizer or polish setting passed downstream. |
| `config.py:197` | `adam_eps` | `1e-16` | Active model dimension, network size, optimizer or polish setting passed downstream. |
| `config.py:198` | `epochs` | `200` | Active model dimension, network size, optimizer or polish setting passed downstream. |
| `config.py:205` | `n_its` | `None` | None; epochs define update count. |
| `config.py:216` | `its_per_val` | `1300` | 1300 batch updates, initial validation also occurs. |
| `config.py:217` | `nf_seconds` | `0.1` | Active model dimension, network size, optimizer or polish setting passed downstream. |
| `config.py:219` | `nf_override` | `None` | None; formula branch chosen. |
| `config.py:220` | `na_nb_override` | `30` | Overrides dimension-derived history to30. |
| `config.py:223` | `loss_weighting` | `'servo'` | Adds record-axis loss_w to every window. |
| `config.py:225` | `validation_measure` | `'sim-NRMSe'` | Active selection, history and polish scoring. |
| `config.py:237` | `lbfgs` | `True` | Enables terminal polish. |
| `config.py:243` | `lbfgs_max_iter` | `100` | Active model dimension, network size, optimizer or polish setting passed downstream. |
| `config.py:244` | `lbfgs_inner_iter` | `20` | Active model dimension, network size, optimizer or polish setting passed downstream. |
| `config.py:245` | `lbfgs_history` | `50` | Active model dimension, network size, optimizer or polish setting passed downstream. |
| `config.py:258` | `lbfgs_windows` | `16384` | Active model dimension, network size, optimizer or polish setting passed downstream. |
| `config.py:263` | `lbfgs_chunk` | `None` | Active model dimension, network size, optimizer or polish setting passed downstream. |
| `config.py:276` | `lbfgs_tol_grad` | `2.6e-5` | Active model dimension, network size, optimizer or polish setting passed downstream. |
| `config.py:277` | `lbfgs_tol_change` | `2.6e-7` | Active model dimension, network size, optimizer or polish setting passed downstream. |
| `config.py:283` | `lbfgs_freeze` | `()` | Empty; freeze context touches no parameters. |
| `config.py:284` | `lbfgs_time_budget_s` | `None` | None; no time-budget termination. |
| `config.py:286` | `lbfgs_meter_rel_tol` | `0.001` | Read into spec but no acceptance effect with meter gate off. |
| `config.py:289` | `lbfgs_meters_gate` | `False` | False; meter diagnostics logged but no veto. |
| `config.py:293` | `start_phase` | `'adam'` | adam; skip-Adam alternative inactive. |
| `config.py:296` | `ps2` | `varies: True for PS rows, False otherwise` | Active model dimension, network size, optimizer or polish setting passed downstream. |
| `config.py:323` | `orth` | `False` | False; no soft penalty construction. |
| `config.py:324` | `orth_beta` | `0.000466` | Metadata only for this objective; positive field does not enable penalty. |
| `config.py:326` | `orth_point_stride` | `100` | Metadata only; penalty builder never called. |
| `config.py:327` | `orth_rank_tol` | `1e-12` | Metadata only; not the OBC rank tolerance. |
| `config.py:339` | `obc` | `varies: True for OBC, False for U` | Active model dimension, network size, optimizer or polish setting passed downstream. |
| `config.py:343` | `obc_space` | `'affine'` | Read for OBC rows, affine only; metadata/guards on U. |
| `config.py:346` | `obc_ref_successor` | `True` | Active reference endpoint on OBC; evaluation also uses it for U. |
| `config.py:349` | `obc_refresh` | `'epoch'` | Active epoch branch on OBC; metadata on U. |
| `config.py:352` | `obc_ref_chunk` | `16384` | Basis construction memory chunks on OBC; evaluation interface. |
| `config.py:355` | `obc_rank_rtol` | `None` | None => shape-dependent eps64 cutoff; OBC only. |
| `config.py:362` | `static_pass_buffers` | `False` | Read False; no static-copy buffers. |
| `config.py:365` | `nx_phys` | `6` | Active model dimension, network size, optimizer or polish setting passed downstream. |
| `config.py:366` | `nu` | `3` | Active model dimension, network size, optimizer or polish setting passed downstream. |
| `config.py:367` | `ny` | `3` | Active model dimension, network size, optimizer or polish setting passed downstream. |

Derived properties `config.py:542` fs_new_hz=4000, `:546` d=5, `:550` ts_new=0.00025, `:554` dtype_np=float64, `:558` dtype_pt=torch.float64, `:562` nf=400, `:576` na_nb=30 and `:583` hp are required. hp keys and insertion order: NX_ANN, n_nodes_per_layer, n_hidden_layers, up_sample, nf, na_nb, batch_size, lr, epochs. No default_hp wrapper call is needed.

### A4. Complete file, class and function ledger

Each definition is listed separately. Parent-qualified nested functions are included. Markers use the source file marker or the nearest decorated class; CHANGED numbers identify the actual comments to preserve, including comments preceding a new method. Original/unmarked is an observation, not a request to invent a marker. Rows marked diagnostic are excluded even when current code executes them.

#### `scripts/gantry/gantry_interconnect_dynamic.py`

| Source | Symbol | Classification | Spec and call path / exclusion | Project marker |
| :- | :- | :- | :- | :- |
| `scripts/gantry/gantry_interconnect_dynamic.py:318` | `_thesis_env` | required | ALL: module THESIS block -> _thesis_env; __main__ -> main | original/unmarked |
| `scripts/gantry/gantry_interconnect_dynamic.py:382` | `main` | required | ALL: module THESIS block -> _thesis_env; __main__ -> main | original/unmarked; CHANGED:426 |

#### `scripts/gantry/gantry_dynamic/__init__.py`
No function or class definitions. Import/bootstrap status: required package bootstrap; imports define framework classes but do not instantiate benchmark models. See A5.

#### `scripts/gantry/gantry_dynamic/baselines.py`

| Source | Symbol | Classification | Spec and call path / exclusion | Project marker |
| :- | :- | :- | :- | :- |
| `scripts/gantry/gantry_dynamic/baselines.py:18` | `stepwise_rollout` | not needed, diagnostic | Diagnostic module explicitly excluded; baselines.py no longer used by entry; no training call path | file added:6 |
| `scripts/gantry/gantry_dynamic/baselines.py:33` | `compute_baseline_fp_nrms` | not needed, diagnostic | Diagnostic module explicitly excluded; baselines.py no longer used by entry; no training call path | file added:6 |
| `scripts/gantry/gantry_dynamic/baselines.py:91` | `compute_baseline_fp_nrms._fp_step` | not needed, diagnostic | Diagnostic module explicitly excluded; baselines.py no longer used by entry; no training call path | file added:6 |
| `scripts/gantry/gantry_dynamic/baselines.py:120` | `compute_all_baselines` | not needed, diagnostic | Diagnostic module explicitly excluded; baselines.py no longer used by entry; no training call path | file added:6 |

#### `scripts/gantry/gantry_dynamic/closed_loop_report.py`

| Source | Symbol | Classification | Spec and call path / exclusion | Project marker |
| :- | :- | :- | :- | :- |
| `scripts/gantry/gantry_dynamic/closed_loop_report.py:38` | `_without_learned` | not needed in method; required interface | Post-run report excluded; eval_greybox imports _without_learned/_true_free (Part C) | file added:19 |
| `scripts/gantry/gantry_dynamic/closed_loop_report.py:67` | `_true_free` | not needed in method; required interface | Post-run report excluded; eval_greybox imports _without_learned/_true_free (Part C) | file added:19 |
| `scripts/gantry/gantry_dynamic/closed_loop_report.py:81` | `_score` | not needed, diagnostic | Post-run report excluded; eval_greybox imports _without_learned/_true_free (Part C) | file added:19 |
| `scripts/gantry/gantry_dynamic/closed_loop_report.py:86` | `closed_loop_table` | not needed, diagnostic | Post-run report excluded; eval_greybox imports _without_learned/_true_free (Part C) | file added:19 |

#### `scripts/gantry/gantry_dynamic/config.py`

| Source | Symbol | Classification | Spec and call path / exclusion | Project marker |
| :- | :- | :- | :- | :- |
| `scripts/gantry/gantry_dynamic/config.py:46` | `y_decimation` | required | ALL: entry -> cfg/hp/config serialization and import coverage guard | file added:12 |
| `scripts/gantry/gantry_dynamic/config.py:53` | `record_trim` | required | ALL: entry -> cfg/hp/config serialization and import coverage guard | file added:12 |
| `scripts/gantry/gantry_dynamic/config.py:60` | `RunConfig` | required | ALL: entry -> cfg/hp/config serialization and import coverage guard | file added:12 |
| `scripts/gantry/gantry_dynamic/config.py:370` | `RunConfig.__post_init__` | required | ALL: entry -> cfg/hp/config serialization and import coverage guard | file added:12 |
| `scripts/gantry/gantry_dynamic/config.py:542` | `RunConfig.fs_new_hz` | required | ALL: entry -> cfg/hp/config serialization and import coverage guard | file added:12 |
| `scripts/gantry/gantry_dynamic/config.py:546` | `RunConfig.d` | required | ALL: entry -> cfg/hp/config serialization and import coverage guard | file added:12 |
| `scripts/gantry/gantry_dynamic/config.py:550` | `RunConfig.ts_new` | required | ALL: entry -> cfg/hp/config serialization and import coverage guard | file added:12 |
| `scripts/gantry/gantry_dynamic/config.py:554` | `RunConfig.dtype_np` | required | ALL: entry -> cfg/hp/config serialization and import coverage guard | file added:12 |
| `scripts/gantry/gantry_dynamic/config.py:558` | `RunConfig.dtype_pt` | required | ALL: entry -> cfg/hp/config serialization and import coverage guard | file added:12 |
| `scripts/gantry/gantry_dynamic/config.py:562` | `RunConfig.nf` | required | ALL: entry -> cfg/hp/config serialization and import coverage guard | file added:12 |
| `scripts/gantry/gantry_dynamic/config.py:576` | `RunConfig.na_nb` | required | ALL: entry -> cfg/hp/config serialization and import coverage guard | file added:12 |
| `scripts/gantry/gantry_dynamic/config.py:583` | `RunConfig.hp` | required | ALL: entry -> cfg/hp/config serialization and import coverage guard | file added:12 |
| `scripts/gantry/gantry_dynamic/config.py:598` | `default_hp` | not needed | default_hp compatibility wrapper is not called; direct cfg.hp used. | file added:12 |
| `scripts/gantry/gantry_dynamic/config.py:603` | `save_dir` | required | ALL: entry -> cfg/hp/config serialization and import coverage guard | file added:12 |
| `scripts/gantry/gantry_dynamic/config.py:610` | `git_provenance` | required | ALL: entry -> cfg/hp/config serialization and import coverage guard | file added:12 |
| `scripts/gantry/gantry_dynamic/config.py:628` | `git_provenance._git` | required | ALL: entry -> cfg/hp/config serialization and import coverage guard | file added:12 |
| `scripts/gantry/gantry_dynamic/config.py:668` | `config_json_dict` | required | ALL: entry -> cfg/hp/config serialization and import coverage guard | file added:12 |
| `scripts/gantry/gantry_dynamic/config.py:793` | `_assert_json_coverage` | required | ALL: entry -> cfg/hp/config serialization and import coverage guard | file added:12 |

#### `scripts/gantry/gantry_dynamic/controller.py`

| Source | Symbol | Classification | Spec and call path / exclusion | Project marker |
| :- | :- | :- | :- | :- |
| `scripts/gantry/gantry_dynamic/controller.py:72` | `M_op` | required | ALL: load records -> controller registration; main -> build_closed_loop -> bank | file added:44 |
| `scripts/gantry/gantry_dynamic/controller.py:81` | `sys_stage_frf` | required | ALL: load records -> controller registration; main -> build_closed_loop -> bank | file added:44 |
| `scripts/gantry/gantry_dynamic/controller.py:92` | `cnorm_coeffs` | required | ALL: load records -> controller registration; main -> build_closed_loop -> bank | file added:44 |
| `scripts/gantry/gantry_dynamic/controller.py:99` | `cnorm_at` | required | ALL: load records -> controller registration; main -> build_closed_loop -> bank | file added:44 |
| `scripts/gantry/gantry_dynamic/controller.py:104` | `build_cfb_at` | required | ALL: load records -> controller registration; main -> build_closed_loop -> bank | file added:44 |
| `scripts/gantry/gantry_dynamic/controller.py:124` | `_tf_to_ss_batch` | required | ALL: load records -> controller registration; main -> build_closed_loop -> bank | file added:44 |
| `scripts/gantry/gantry_dynamic/controller.py:143` | `controller_ss` | required | ALL: load records -> controller registration; main -> build_closed_loop -> bank | file added:44 |
| `scripts/gantry/gantry_dynamic/controller.py:181` | `_stem` | required | ALL: load records -> controller registration; main -> build_closed_loop -> bank | file added:44 |
| `scripts/gantry/gantry_dynamic/controller.py:198` | `register_stored_controller` | required | ALL: load records -> controller registration; main -> build_closed_loop -> bank | file added:44 |
| `scripts/gantry/gantry_dynamic/controller.py:210` | `y_op_for` | required | ALL: load records -> controller registration; main -> build_closed_loop -> bank | file added:44 |
| `scripts/gantry/gantry_dynamic/controller.py:232` | `gain_for` | required | ALL: load records -> controller registration; main -> build_closed_loop -> bank | file added:44 |
| `scripts/gantry/gantry_dynamic/controller.py:246` | `build_controller_bank` | required | ALL: load records -> controller registration; main -> build_closed_loop -> bank | file added:44 |
| `scripts/gantry/gantry_dynamic/controller.py:269` | `build_closed_loop` | required | ALL: load records -> controller registration; main -> build_closed_loop -> bank | file added:44 |

#### `scripts/gantry/gantry_dynamic/data.py`

| Source | Symbol | Classification | Spec and call path / exclusion | Project marker |
| :- | :- | :- | :- | :- |
| `scripts/gantry/gantry_dynamic/data.py:75` | `is_thesis` | required | ALL: main -> load_datasets/normalization; evaluation -> load_traj | file added:8 |
| `scripts/gantry/gantry_dynamic/data.py:79` | `record_files` | required | ALL: main -> load_datasets/normalization; evaluation -> load_traj | file added:8 |
| `scripts/gantry/gantry_dynamic/data.py:92` | `_register_controller` | required | ALL: main -> load_datasets/normalization; evaluation -> load_traj | file added:8 |
| `scripts/gantry/gantry_dynamic/data.py:102` | `DataBundle` | required | ALL: main -> load_datasets/normalization; evaluation -> load_traj | file added:8 |
| `scripts/gantry/gantry_dynamic/data.py:127` | `Norm` | required | ALL: main -> load_datasets/normalization; evaluation -> load_traj | file added:8 |
| `scripts/gantry/gantry_dynamic/data.py:141` | `traj_dir` | required | ALL: main -> load_datasets/normalization; evaluation -> load_traj | file added:8 |
| `scripts/gantry/gantry_dynamic/data.py:147` | `_load_u` | required | ALL: main -> load_datasets/normalization; evaluation -> load_traj | file added:8 |
| `scripts/gantry/gantry_dynamic/data.py:152` | `_resample_u` | required | ALL: main -> load_datasets/normalization; evaluation -> load_traj | file added:8 |
| `scripts/gantry/gantry_dynamic/data.py:172` | `_decimate_y` | required | ALL: main -> load_datasets/normalization; evaluation -> load_traj | file added:8 |
| `scripts/gantry/gantry_dynamic/data.py:185` | `_record_at_fs_new` | required | ALL: main -> load_datasets/normalization; evaluation -> load_traj | file added:8 |
| `scripts/gantry/gantry_dynamic/data.py:203` | `_load_record` | required | ALL: main -> load_datasets/normalization; evaluation -> load_traj | file added:8 |
| `scripts/gantry/gantry_dynamic/data.py:218` | `load_traj` | required interface | Evaluation loads individual records and twins; main load_datasets uses _load_record directly. | file added:8 |
| `scripts/gantry/gantry_dynamic/data.py:222` | `servo_sigma` | required | ALL: main -> load_datasets/normalization; evaluation -> load_traj | file added:8 |
| `scripts/gantry/gantry_dynamic/data.py:237` | `load_mat_aug` | not needed, diagnostic | Called by load_datasets for primary truth and by excluded closed-loop report; no training numeric input. | file added:8 |
| `scripts/gantry/gantry_dynamic/data.py:258` | `load_datasets` | required | ALL: main -> load_datasets/normalization; evaluation -> load_traj | file added:8 |
| `scripts/gantry/gantry_dynamic/data.py:276` | `load_datasets._split` | required | ALL: main -> load_datasets/normalization; evaluation -> load_traj | file added:8 |
| `scripts/gantry/gantry_dynamic/data.py:329` | `compute_normalization` | required | ALL: main -> load_datasets/normalization; evaluation -> load_traj | file added:8 |

#### `scripts/gantry/gantry_dynamic/diagnostics.py`

| Source | Symbol | Classification | Spec and call path / exclusion | Project marker |
| :- | :- | :- | :- | :- |
| `scripts/gantry/gantry_dynamic/diagnostics.py:19` | `r2_per_channel` | not needed, diagnostic | Diagnostic module explicitly excluded; baselines.py no longer used by entry; no training call path | file added:7 |
| `scripts/gantry/gantry_dynamic/diagnostics.py:27` | `best_affine_r2` | not needed, diagnostic | Diagnostic module explicitly excluded; baselines.py no longer used by entry; no training call path | file added:7 |
| `scripts/gantry/gantry_dynamic/diagnostics.py:40` | `encoder_state_estimates` | not needed, diagnostic | Diagnostic module explicitly excluded; baselines.py no longer used by entry; no training call path | file added:7 |
| `scripts/gantry/gantry_dynamic/diagnostics.py:65` | `state_recovery_diagnostic` | not needed, diagnostic | Diagnostic module explicitly excluded; baselines.py no longer used by entry; no training call path | file added:7 |
| `scripts/gantry/gantry_dynamic/diagnostics.py:124` | `compute_gradient_norms` | not needed, diagnostic | Diagnostic module explicitly excluded; baselines.py no longer used by entry; no training call path | file added:7 |
| `scripts/gantry/gantry_dynamic/diagnostics.py:208` | `encoder_init_state` | not needed, diagnostic | Diagnostic module explicitly excluded; baselines.py no longer used by entry; no training call path | file added:7 |
| `scripts/gantry/gantry_dynamic/diagnostics.py:224` | `zeroed_ann_validation` | not needed, diagnostic | Diagnostic module explicitly excluded; baselines.py no longer used by entry; no training call path | file added:7 |
| `scripts/gantry/gantry_dynamic/diagnostics.py:264` | `print_run_banner` | not needed, diagnostic | Diagnostic module explicitly excluded; baselines.py no longer used by entry; no training call path | file added:7 |

#### `scripts/gantry/gantry_dynamic/evaluation.py`

| Source | Symbol | Classification | Spec and call path / exclusion | Project marker |
| :- | :- | :- | :- | :- |
| `scripts/gantry/gantry_dynamic/evaluation.py:31` | `capture_loss_history` | not needed, diagnostic | Post-run evaluate_and_save/report/plots/results; output schema retained as interface in C | file added:15 |
| `scripts/gantry/gantry_dynamic/evaluation.py:53` | `report_joint_estimation` | not needed, diagnostic | Post-run evaluate_and_save/report/plots/results; output schema retained as interface in C | file added:15 |
| `scripts/gantry/gantry_dynamic/evaluation.py:83` | `_record_mean` | not needed, diagnostic | Post-run evaluate_and_save/report/plots/results; output schema retained as interface in C | file added:15 |
| `scripts/gantry/gantry_dynamic/evaluation.py:95` | `evaluate_and_save` | not needed, diagnostic | Post-run evaluate_and_save/report/plots/results; output schema retained as interface in C | file added:15 |
| `scripts/gantry/gantry_dynamic/evaluation.py:152` | `_make_plots` | not needed, diagnostic | Post-run evaluate_and_save/report/plots/results; output schema retained as interface in C | file added:15 |
| `scripts/gantry/gantry_dynamic/evaluation.py:240` | `_build_save_dict` | not needed, diagnostic | Post-run evaluate_and_save/report/plots/results; output schema retained as interface in C | file added:15 |
| `scripts/gantry/gantry_dynamic/evaluation.py:278` | `per_record_nrms` | not needed, diagnostic | Post-run evaluate_and_save/report/plots/results; output schema retained as interface in C | file added:15 |

#### `scripts/gantry/gantry_dynamic/model.py`

| Source | Symbol | Classification | Spec and call path / exclusion | Project marker |
| :- | :- | :- | :- | :- |
| `scripts/gantry/gantry_dynamic/model.py:39` | `get_encoder_dims` | required | ALL: main -> build_model/prefit -> training.train_model | file added:2 |
| `scripts/gantry/gantry_dynamic/model.py:60` | `build_model` | required | ALL: main -> build_model/prefit -> training.train_model | file added:2 |
| `scripts/gantry/gantry_dynamic/model.py:328` | `prefit_physical_encoder` | required | ALL: main -> build_model/prefit -> training.train_model | file added:2 |
| `scripts/gantry/gantry_dynamic/model.py:352` | `_prefit` | required | ALL: main -> build_model/prefit -> training.train_model | file added:2 |
| `scripts/gantry/gantry_dynamic/model.py:389` | `_prefit.recorder` | required | ALL: main -> build_model/prefit -> training.train_model | file added:2 |
| `scripts/gantry/gantry_dynamic/model.py:415` | `least_squares_start` | required | ALL: main -> build_model/prefit -> training.train_model | file added:2 |
| `scripts/gantry/gantry_dynamic/model.py:467` | `train_model` | required | ALL: main -> build_model/prefit -> training.train_model | file added:2 |

#### `scripts/gantry/gantry_dynamic/obc_diagnostics.py`

| Source | Symbol | Classification | Spec and call path / exclusion | Project marker |
| :- | :- | :- | :- | :- |
| `scripts/gantry/gantry_dynamic/obc_diagnostics.py:38` | `qualify_basis` | not needed, diagnostic | Diagnostic module explicitly excluded; baselines.py no longer used by entry; no training call path | file added:19 |
| `scripts/gantry/gantry_dynamic/obc_diagnostics.py:95` | `print_qualification` | not needed, diagnostic | Diagnostic module explicitly excluded; baselines.py no longer used by entry; no training call path | file added:19 |
| `scripts/gantry/gantry_dynamic/obc_diagnostics.py:122` | `principal_angles` | not needed, diagnostic | Diagnostic module explicitly excluded; baselines.py no longer used by entry; no training call path | file added:19 |
| `scripts/gantry/gantry_dynamic/obc_diagnostics.py:131` | `absorber_truth_one_step` | not needed, diagnostic | Diagnostic module explicitly excluded; baselines.py no longer used by entry; no training call path | file added:19 |
| `scripts/gantry/gantry_dynamic/obc_diagnostics.py:159` | `absorber_truth_one_step.M` | not needed, diagnostic | Diagnostic module explicitly excluded; baselines.py no longer used by entry; no training call path | file added:19 |
| `scripts/gantry/gantry_dynamic/obc_diagnostics.py:172` | `absorber_truth_one_step.deriv` | not needed, diagnostic | Diagnostic module explicitly excluded; baselines.py no longer used by entry; no training call path | file added:19 |
| `scripts/gantry/gantry_dynamic/obc_diagnostics.py:192` | `check_truth_parameters` | not needed, diagnostic | Diagnostic module explicitly excluded; baselines.py no longer used by entry; no training call path | file added:19 |
| `scripts/gantry/gantry_dynamic/obc_diagnostics.py:215` | `discrepancy_fields` | not needed, diagnostic | Diagnostic module explicitly excluded; baselines.py no longer used by entry; no training call path | file added:19 |
| `scripts/gantry/gantry_dynamic/obc_diagnostics.py:247` | `rho_table` | not needed, diagnostic | Diagnostic module explicitly excluded; baselines.py no longer used by entry; no training call path | file added:19 |

#### `scripts/gantry/gantry_dynamic/obc_gantry.py`

| Source | Symbol | Classification | Spec and call path / exclusion | Project marker |
| :- | :- | :- | :- | :- |
| `scripts/gantry/gantry_dynamic/obc_gantry.py:49` | `GantryReferenceSet` | required | OBC: build_model -> attach_obc -> reference/basis/correction; evaluation uses same API for U | file added:22 |
| `scripts/gantry/gantry_dynamic/obc_gantry.py:68` | `GantryReferenceSet.n` | required | OBC: build_model -> attach_obc -> reference/basis/correction; evaluation uses same API for U | file added:22 |
| `scripts/gantry/gantry_dynamic/obc_gantry.py:71` | `GantryReferenceSet.memory_bytes` | not needed, diagnostic | May execute in current reporting/refresh monitoring, but excluded numeric diagnostic; OBC: build_model -> attach_obc -> reference/basis/correction; evaluation uses same API for U | file added:22 |
| `scripts/gantry/gantry_dynamic/obc_gantry.py:79` | `fourth_order_central_difference` | required | OBC: build_model -> attach_obc -> reference/basis/correction; evaluation uses same API for U | file added:22 |
| `scripts/gantry/gantry_dynamic/obc_gantry.py:91` | `_load_record_float64` | required | OBC: build_model -> attach_obc -> reference/basis/correction; evaluation uses same API for U | file added:22 |
| `scripts/gantry/gantry_dynamic/obc_gantry.py:106` | `build_reference_set` | required | OBC: build_model -> attach_obc -> reference/basis/correction; evaluation uses same API for U | file added:22 |
| `scripts/gantry/gantry_dynamic/obc_gantry.py:175` | `GantryOBCCorrection` | required | OBC: build_model -> attach_obc -> reference/basis/correction; evaluation uses same API for U | file added:22 |
| `scripts/gantry/gantry_dynamic/obc_gantry.py:189` | `GantryOBCCorrection.__init__` | required | OBC: build_model -> attach_obc -> reference/basis/correction; evaluation uses same API for U | file added:22 |
| `scripts/gantry/gantry_dynamic/obc_gantry.py:222` | `GantryOBCCorrection.step` | not needed | Only alternative use_jvp correction calls this; basis builder calls block transition directly. | file added:22 |
| `scripts/gantry/gantry_dynamic/obc_gantry.py:225` | `GantryOBCCorrection.expansion_point` | required | OBC: build_model -> attach_obc -> reference/basis/correction; evaluation uses same API for U | file added:22 |
| `scripts/gantry/gantry_dynamic/obc_gantry.py:230` | `GantryOBCCorrection.set_frozen` | required | OBC: build_model -> attach_obc -> reference/basis/correction; evaluation uses same API for U | file added:22 |
| `scripts/gantry/gantry_dynamic/obc_gantry.py:250` | `GantryOBCCorrection.tangent_pair` | required | OBC: build_model -> attach_obc -> reference/basis/correction; evaluation uses same API for U | file added:22 |
| `scripts/gantry/gantry_dynamic/obc_gantry.py:271` | `GantryOBCCorrection.correction_from_theta` | not needed | Alternative jvp/diagnostic API; production forward uses tangent_pair. | file added:22 |
| `scripts/gantry/gantry_dynamic/obc_gantry.py:286` | `GantryOBCCorrection.forward` | required | OBC: build_model -> attach_obc -> reference/basis/correction; evaluation uses same API for U | file added:22 |
| `scripts/gantry/gantry_dynamic/obc_gantry.py:312` | `make_float64_block` | required | OBC: build_model -> attach_obc -> reference/basis/correction; evaluation uses same API for U | file added:22 |
| `scripts/gantry/gantry_dynamic/obc_gantry.py:319` | `make_basis_builder` | required | OBC: build_model -> attach_obc -> reference/basis/correction; evaluation uses same API for U | file added:22 |
| `scripts/gantry/gantry_dynamic/obc_gantry.py:323` | `make_basis_builder.build` | required | OBC: build_model -> attach_obc -> reference/basis/correction; evaluation uses same API for U | file added:22 |
| `scripts/gantry/gantry_dynamic/obc_gantry.py:344` | `make_reference_field` | required | OBC: build_model -> attach_obc -> reference/basis/correction; evaluation uses same API for U | file added:22 |
| `scripts/gantry/gantry_dynamic/obc_gantry.py:350` | `make_reference_field.field` | required | OBC: build_model -> attach_obc -> reference/basis/correction; evaluation uses same API for U | file added:22 |
| `scripts/gantry/gantry_dynamic/obc_gantry.py:362` | `attach_obc` | required | OBC: build_model -> attach_obc -> reference/basis/correction; evaluation uses same API for U | file added:22 |

#### `scripts/gantry/gantry_dynamic/training.py`

| Source | Symbol | Classification | Spec and call path / exclusion | Project marker |
| :- | :- | :- | :- | :- |
| `scripts/gantry/gantry_dynamic/training.py:26` | `_json_safe` | required | ALL: main -> training/history/checkpoint/polish orchestration; diagnostic subbranches excluded in A2/C | file added:10 |
| `scripts/gantry/gantry_dynamic/training.py:39` | `save_checkpoint_weights` | required | ALL: main -> training/history/checkpoint/polish orchestration; diagnostic subbranches excluded in A2/C | file added:10 |
| `scripts/gantry/gantry_dynamic/training.py:67` | `resolve_checkpoint` | required interface | Evaluation -> load_checkpoint -> compatibility checks; not fresh-campaign execution | file added:10 |
| `scripts/gantry/gantry_dynamic/training.py:124` | `load_optimizer_moments` | not needed | Resume/architecture convenience helper; fresh spec has no resume and evaluation loads with start_phase=lbfgs without this call. | file added:10 |
| `scripts/gantry/gantry_dynamic/training.py:158` | `_assert_same_architecture` | required interface | Evaluation -> load_checkpoint -> compatibility checks; not fresh-campaign execution | file added:10 |
| `scripts/gantry/gantry_dynamic/training.py:198` | `infer_arch_from_hfn` | not needed | Resume/architecture convenience helper; fresh spec has no resume and evaluation loads with start_phase=lbfgs without this call. | file added:10 |
| `scripts/gantry/gantry_dynamic/training.py:236` | `config_for_checkpoint` | not needed | Resume/architecture convenience helper; fresh spec has no resume and evaluation loads with start_phase=lbfgs without this call. | file added:10 |
| `scripts/gantry/gantry_dynamic/training.py:248` | `_assert_same_dtype` | required interface | Evaluation -> load_checkpoint -> compatibility checks; not fresh-campaign execution | file added:10 |
| `scripts/gantry/gantry_dynamic/training.py:277` | `load_checkpoint` | required interface | Evaluation -> load_checkpoint -> compatibility checks; not fresh-campaign execution | file added:10 |
| `scripts/gantry/gantry_dynamic/training.py:319` | `_NfProbe` | required | ALL: main -> training/history/checkpoint/polish orchestration; diagnostic subbranches excluded in A2/C | file added:10 |
| `scripts/gantry/gantry_dynamic/training.py:362` | `_NfProbe.__init__` | required | ALL: main -> training/history/checkpoint/polish orchestration; diagnostic subbranches excluded in A2/C | file added:10 |
| `scripts/gantry/gantry_dynamic/training.py:446` | `_NfProbe._val_stats` | not needed, diagnostic | ALL: installed nf/joint/OBC probe or polish meters; no selection veto; see C | file added:10 |
| `scripts/gantry/gantry_dynamic/training.py:490` | `_NfProbe._fmt` | not needed, diagnostic | ALL: installed nf/joint/OBC probe or polish meters; no selection veto; see C | file added:10 |
| `scripts/gantry/gantry_dynamic/training.py:498` | `_NfProbe._joint_probe` | not needed, diagnostic | ALL: installed nf/joint/OBC probe or polish meters; no selection veto; see C | file added:10 |
| `scripts/gantry/gantry_dynamic/training.py:560` | `_NfProbe._obc_probe` | not needed, diagnostic | ALL: installed nf/joint/OBC probe or polish meters; no selection veto; see C | file added:10 |
| `scripts/gantry/gantry_dynamic/training.py:593` | `_NfProbe.record` | required | ALL: main -> training/history/checkpoint/polish orchestration; diagnostic subbranches excluded in A2/C | file added:10 |
| `scripts/gantry/gantry_dynamic/training.py:610` | `_NfProbe.hist_arrays` | required | ALL: main -> training/history/checkpoint/polish orchestration; diagnostic subbranches excluded in A2/C | file added:10 |
| `scripts/gantry/gantry_dynamic/training.py:627` | `_NfProbe.__call__` | required | ALL: main -> training/history/checkpoint/polish orchestration; diagnostic subbranches excluded in A2/C | file added:10 |
| `scripts/gantry/gantry_dynamic/training.py:671` | `_install_nf_val_probe` | required | ALL: main -> training/history/checkpoint/polish orchestration; diagnostic subbranches excluded in A2/C | file added:10 |
| `scripts/gantry/gantry_dynamic/training.py:703` | `_gantry_meters` | not needed, diagnostic | ALL: installed nf/joint/OBC probe or polish meters; no selection veto; see C | file added:10 |
| `scripts/gantry/gantry_dynamic/training.py:753` | `run_lbfgs_polish` | required | ALL: main -> training/history/checkpoint/polish orchestration; diagnostic subbranches excluded in A2/C | file added:10 |
| `scripts/gantry/gantry_dynamic/training.py:821` | `train_model_with_diagnostics` | required | ALL: main -> training/history/checkpoint/polish orchestration; diagnostic subbranches excluded in A2/C | file added:10 |
| `scripts/gantry/gantry_dynamic/training.py:1016` | `resolve_resume_hp` | required | ALL: main -> training/history/checkpoint/polish orchestration; diagnostic subbranches excluded in A2/C | file added:10 |

#### `scripts/gantry/common/feedback_curriculum.py`

| Source | Symbol | Classification | Spec and call path / exclusion | Project marker |
| :- | :- | :- | :- | :- |
| `scripts/gantry/common/feedback_curriculum.py:21` | `FixedWindowBatch` | not needed | oracle only post-run report; other common modules have no required import/call with orth=False and no ReZero/curriculum | original/unmarked |
| `scripts/gantry/common/feedback_curriculum.py:30` | `build_fixed_window_batch` | not needed | oracle only post-run report; other common modules have no required import/call with orth=False and no ReZero/curriculum | original/unmarked |
| `scripts/gantry/common/feedback_curriculum.py:66` | `save_training_snapshot` | not needed | oracle only post-run report; other common modules have no required import/call with orth=False and no ReZero/curriculum | original/unmarked |
| `scripts/gantry/common/feedback_curriculum.py:76` | `load_training_snapshot` | not needed | oracle only post-run report; other common modules have no required import/call with orth=False and no ReZero/curriculum | original/unmarked |
| `scripts/gantry/common/feedback_curriculum.py:84` | `FeedbackTargetSelector` | not needed | oracle only post-run report; other common modules have no required import/call with orth=False and no ReZero/curriculum | original/unmarked |
| `scripts/gantry/common/feedback_curriculum.py:94` | `FeedbackTargetSelector.__init__` | not needed | oracle only post-run report; other common modules have no required import/call with orth=False and no ReZero/curriculum | original/unmarked |
| `scripts/gantry/common/feedback_curriculum.py:125` | `FeedbackTargetSelector.initial_strength` | not needed | oracle only post-run report; other common modules have no required import/call with orth=False and no ReZero/curriculum | original/unmarked |
| `scripts/gantry/common/feedback_curriculum.py:128` | `FeedbackTargetSelector._evaluate` | not needed | oracle only post-run report; other common modules have no required import/call with orth=False and no ReZero/curriculum | original/unmarked |
| `scripts/gantry/common/feedback_curriculum.py:165` | `FeedbackTargetSelector._checkpoint_path` | not needed | oracle only post-run report; other common modules have no required import/call with orth=False and no ReZero/curriculum | original/unmarked |
| `scripts/gantry/common/feedback_curriculum.py:168` | `FeedbackTargetSelector._write_state` | not needed | oracle only post-run report; other common modules have no required import/call with orth=False and no ReZero/curriculum | original/unmarked |
| `scripts/gantry/common/feedback_curriculum.py:187` | `FeedbackTargetSelector.__call__` | not needed | oracle only post-run report; other common modules have no required import/call with orth=False and no ReZero/curriculum | original/unmarked |
| `scripts/gantry/common/feedback_curriculum.py:274` | `local_dynamics_diagnostic` | not needed | oracle only post-run report; other common modules have no required import/call with orth=False and no ReZero/curriculum | original/unmarked |
| `scripts/gantry/common/feedback_curriculum.py:314` | `local_dynamics_diagnostic.state_from_x` | not needed | oracle only post-run report; other common modules have no required import/call with orth=False and no ReZero/curriculum | original/unmarked |
| `scripts/gantry/common/feedback_curriculum.py:317` | `local_dynamics_diagnostic.state_from_u` | not needed | oracle only post-run report; other common modules have no required import/call with orth=False and no ReZero/curriculum | original/unmarked |
| `scripts/gantry/common/feedback_curriculum.py:320` | `local_dynamics_diagnostic.output_from_x` | not needed | oracle only post-run report; other common modules have no required import/call with orth=False and no ReZero/curriculum | original/unmarked |

#### `scripts/gantry/common/oracle.py`

| Source | Symbol | Classification | Spec and call path / exclusion | Project marker |
| :- | :- | :- | :- | :- |
| `scripts/gantry/common/oracle.py:61` | `_M` | not needed, diagnostic | oracle only post-run report; other common modules have no required import/call with orth=False and no ReZero/curriculum | file added:17 |
| `scripts/gantry/common/oracle.py:73` | `_deriv` | not needed, diagnostic | oracle only post-run report; other common modules have no required import/call with orth=False and no ReZero/curriculum | file added:17 |
| `scripts/gantry/common/oracle.py:83` | `simulate` | not needed, diagnostic | oracle only post-run report; other common modules have no required import/call with orth=False and no ReZero/curriculum | file added:17 |
| `scripts/gantry/common/oracle.py:101` | `oracle_open_loop` | not needed, diagnostic | oracle only post-run report; other common modules have no required import/call with orth=False and no ReZero/curriculum | file added:17 |
| `scripts/gantry/common/oracle.py:144` | `truth_from_meta` | not needed, diagnostic | oracle only post-run report; other common modules have no required import/call with orth=False and no ReZero/curriculum | file added:17 |
| `scripts/gantry/common/oracle.py:160` | `_deriv_truth` | not needed, diagnostic | oracle only post-run report; other common modules have no required import/call with orth=False and no ReZero/curriculum | file added:17 |
| `scripts/gantry/common/oracle.py:186` | `oracle_closed_loop` | not needed, diagnostic | oracle only post-run report; other common modules have no required import/call with orth=False and no ReZero/curriculum | file added:17 |

#### `scripts/gantry/common/orth_penalty.py`

| Source | Symbol | Classification | Spec and call path / exclusion | Project marker |
| :- | :- | :- | :- | :- |
| `scripts/gantry/common/orth_penalty.py:53` | `theta_bar_for` | not needed | oracle only post-run report; other common modules have no required import/call with orth=False and no ReZero/curriculum | file added:34 |
| `scripts/gantry/common/orth_penalty.py:63` | `_cache_path` | not needed | oracle only post-run report; other common modules have no required import/call with orth=False and no ReZero/curriculum | file added:34 |
| `scripts/gantry/common/orth_penalty.py:81` | `_x_logical_from_data` | not needed | oracle only post-run report; other common modules have no required import/call with orth=False and no ReZero/curriculum | file added:34 |
| `scripts/gantry/common/orth_penalty.py:107` | `_make_block` | not needed | oracle only post-run report; other common modules have no required import/call with orth=False and no ReZero/curriculum | file added:34 |
| `scripts/gantry/common/orth_penalty.py:117` | `_compute` | not needed | oracle only post-run report; other common modules have no required import/call with orth=False and no ReZero/curriculum | file added:34 |
| `scripts/gantry/common/orth_penalty.py:180` | `build_orth_penalty` | not needed | oracle only post-run report; other common modules have no required import/call with orth=False and no ReZero/curriculum | file added:34 |

#### `scripts/gantry/common/rezero_gate.py`

| Source | Symbol | Classification | Spec and call path / exclusion | Project marker |
| :- | :- | :- | :- | :- |
| `scripts/gantry/common/rezero_gate.py:46` | `_Gate` | not needed | oracle only post-run report; other common modules have no required import/call with orth=False and no ReZero/curriculum | file added:39 |
| `scripts/gantry/common/rezero_gate.py:49` | `_Gate.__init__` | not needed | oracle only post-run report; other common modules have no required import/call with orth=False and no ReZero/curriculum | file added:39 |
| `scripts/gantry/common/rezero_gate.py:62` | `_Gate.gate` | not needed | oracle only post-run report; other common modules have no required import/call with orth=False and no ReZero/curriculum | file added:39 |
| `scripts/gantry/common/rezero_gate.py:65` | `_Gate.forward` | not needed | oracle only post-run report; other common modules have no required import/call with orth=False and no ReZero/curriculum | file added:39 |
| `scripts/gantry/common/rezero_gate.py:69` | `apply_rezero_gate` | not needed | oracle only post-run report; other common modules have no required import/call with orth=False and no ReZero/curriculum | file added:39 |

#### `model_augmentation/__init__.py`
No function or class definitions. Import/bootstrap status: required package bootstrap; imports define framework classes but do not instantiate benchmark models. See A5.

#### `model_augmentation/fit_systems/__init__.py`
No function or class definitions. Import/bootstrap status: required package bootstrap; imports define framework classes but do not instantiate benchmark models. See A5.

#### `model_augmentation/fit_systems/augmented_dynamics.py`

| Source | Symbol | Classification | Spec and call path / exclusion | Project marker |
| :- | :- | :- | :- | :- |
| `model_augmentation/fit_systems/augmented_dynamics.py:98` | `AugmentedDynamics` | not needed | No AugmentedDynamics wrapper is built; live-g MLP handles added states. | file added:85 |
| `model_augmentation/fit_systems/augmented_dynamics.py:118` | `AugmentedDynamics.__init__` | not needed | No AugmentedDynamics wrapper is built; live-g MLP handles added states. | file added:85 |
| `model_augmentation/fit_systems/augmented_dynamics.py:172` | `AugmentedDynamics.net` | not needed | No AugmentedDynamics wrapper is built; live-g MLP handles added states. | file added:85 |
| `model_augmentation/fit_systems/augmented_dynamics.py:176` | `AugmentedDynamics.rho` | not needed | No AugmentedDynamics wrapper is built; live-g MLP handles added states. | file added:85 |
| `model_augmentation/fit_systems/augmented_dynamics.py:195` | `AugmentedDynamics.gamma` | not needed | No AugmentedDynamics wrapper is built; live-g MLP handles added states. | file added:85 |
| `model_augmentation/fit_systems/augmented_dynamics.py:199` | `AugmentedDynamics.effective_B` | not needed | No AugmentedDynamics wrapper is built; live-g MLP handles added states. | file added:85 |
| `model_augmentation/fit_systems/augmented_dynamics.py:206` | `AugmentedDynamics.trainable_parameters` | not needed | No AugmentedDynamics wrapper is built; live-g MLP handles added states. | file added:85 |
| `model_augmentation/fit_systems/augmented_dynamics.py:212` | `AugmentedDynamics.pole_table` | not needed | No AugmentedDynamics wrapper is built; live-g MLP handles added states. | file added:85 |
| `model_augmentation/fit_systems/augmented_dynamics.py:229` | `AugmentedDynamics.forward` | not needed | No AugmentedDynamics wrapper is built; live-g MLP handles added states. | file added:85 |
| `model_augmentation/fit_systems/augmented_dynamics.py:257` | `find_ann_block` | not needed | No AugmentedDynamics wrapper is built; live-g MLP handles added states. | file added:85 |
| `model_augmentation/fit_systems/augmented_dynamics.py:263` | `find_wrapper` | not needed | No AugmentedDynamics wrapper is built; live-g MLP handles added states. | file added:85 |
| `model_augmentation/fit_systems/augmented_dynamics.py:273` | `final_linear` | not needed | No AugmentedDynamics wrapper is built; live-g MLP handles added states. | file added:85 |
| `model_augmentation/fit_systems/augmented_dynamics.py:282` | `reset_final_layer` | not needed | No AugmentedDynamics wrapper is built; live-g MLP handles added states. | file added:85 |
| `model_augmentation/fit_systems/augmented_dynamics.py:302` | `empirical_input_scale` | not needed | No AugmentedDynamics wrapper is built; live-g MLP handles added states. | file added:85 |
| `model_augmentation/fit_systems/augmented_dynamics.py:349` | `draw_input_map` | not needed | No AugmentedDynamics wrapper is built; live-g MLP handles added states. | file added:85 |

#### `model_augmentation/fit_systems/blocks.py`

| Source | Symbol | Classification | Spec and call path / exclusion | Project marker |
| :- | :- | :- | :- | :- |
| `model_augmentation/fit_systems/blocks.py:12` | `Block` | required | ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | original/unmarked |
| `model_augmentation/fit_systems/blocks.py:15` | `Block.__init__` | required | ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | original/unmarked |
| `model_augmentation/fit_systems/blocks.py:24` | `Block.init_block` | required | ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | original/unmarked |
| `model_augmentation/fit_systems/blocks.py:27` | `Block.forward` | not needed | Abstract method is overridden by every constructed concrete block. | original/unmarked |
| `model_augmentation/fit_systems/blocks.py:31` | `Static_ANN_Block` | required | ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | original/unmarked; CHANGED:46,56 |
| `model_augmentation/fit_systems/blocks.py:32` | `Static_ANN_Block.__init__` | required | ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | original/unmarked; CHANGED:46,56 |
| `model_augmentation/fit_systems/blocks.py:69` | `Static_ANN_Block.forward` | required | ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | original/unmarked |
| `model_augmentation/fit_systems/blocks.py:83` | `Cubic_Block` | not needed | Different physical/ANN block or raw-parameter alternative; reduced gantry branch only. | original/unmarked |
| `model_augmentation/fit_systems/blocks.py:84` | `Cubic_Block.__init__` | not needed | Different physical/ANN block or raw-parameter alternative; reduced gantry branch only. | original/unmarked |
| `model_augmentation/fit_systems/blocks.py:88` | `Cubic_Block.forward` | not needed | Different physical/ANN block or raw-parameter alternative; reduced gantry branch only. | original/unmarked |
| `model_augmentation/fit_systems/blocks.py:94` | `Linear_State_Block` | not needed | Different physical/ANN block or raw-parameter alternative; reduced gantry branch only. | original/unmarked |
| `model_augmentation/fit_systems/blocks.py:95` | `Linear_State_Block.__init__` | not needed | Different physical/ANN block or raw-parameter alternative; reduced gantry branch only. | original/unmarked |
| `model_augmentation/fit_systems/blocks.py:109` | `Linear_State_Block.forward` | not needed | Different physical/ANN block or raw-parameter alternative; reduced gantry branch only. | original/unmarked |
| `model_augmentation/fit_systems/blocks.py:120` | `Linear_Output_Block` | required | ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | @added on Linear_Output_Block |
| `model_augmentation/fit_systems/blocks.py:121` | `Linear_Output_Block.__init__` | required | ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | @added on Linear_Output_Block |
| `model_augmentation/fit_systems/blocks.py:138` | `Linear_Output_Block.forward` | required | ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | @added on Linear_Output_Block |
| `model_augmentation/fit_systems/blocks.py:146` | `Parameterized_Linear_State_Block` | not needed | Different physical/ANN block or raw-parameter alternative; reduced gantry branch only. | original/unmarked |
| `model_augmentation/fit_systems/blocks.py:147` | `Parameterized_Linear_State_Block.__init__` | not needed | Different physical/ANN block or raw-parameter alternative; reduced gantry branch only. | original/unmarked |
| `model_augmentation/fit_systems/blocks.py:178` | `Parameterized_Linear_State_Block.forward` | not needed | Different physical/ANN block or raw-parameter alternative; reduced gantry branch only. | original/unmarked |
| `model_augmentation/fit_systems/blocks.py:187` | `Parameterized_Linear_State_Block.param_loss` | not needed | Different physical/ANN block or raw-parameter alternative; reduced gantry branch only. | original/unmarked |
| `model_augmentation/fit_systems/blocks.py:199` | `Parameterized_Linear_Output_Block` | not needed | Different physical/ANN block or raw-parameter alternative; reduced gantry branch only. | original/unmarked |
| `model_augmentation/fit_systems/blocks.py:200` | `Parameterized_Linear_Output_Block.__init__` | not needed | Different physical/ANN block or raw-parameter alternative; reduced gantry branch only. | original/unmarked |
| `model_augmentation/fit_systems/blocks.py:232` | `Parameterized_Linear_Output_Block.forward` | not needed | Different physical/ANN block or raw-parameter alternative; reduced gantry branch only. | original/unmarked |
| `model_augmentation/fit_systems/blocks.py:239` | `Parameterized_Linear_Output_Block.param_loss` | not needed | Different physical/ANN block or raw-parameter alternative; reduced gantry branch only. | original/unmarked |
| `model_augmentation/fit_systems/blocks.py:251` | `Discrete_Nonlinear_Function_Block` | required | ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | original/unmarked; CHANGED:261 |
| `model_augmentation/fit_systems/blocks.py:252` | `Discrete_Nonlinear_Function_Block.__init__` | required | ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | original/unmarked |
| `model_augmentation/fit_systems/blocks.py:255` | `Discrete_Nonlinear_Function_Block.init_norm` | not needed | Manual normalization; no init_norm call. | original/unmarked |
| `model_augmentation/fit_systems/blocks.py:260` | `Discrete_Nonlinear_Function_Block.forward` | required | ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | original/unmarked; CHANGED:261 |
| `model_augmentation/fit_systems/blocks.py:268` | `Discrete_Nonlinear_Function_Block.nonlinear_function` | not needed | Overridden by gantry implementation. | original/unmarked; CHANGED:261 |
| `model_augmentation/fit_systems/blocks.py:274` | `Cascaded_Tanks_State_Block` | not needed | Different physical/ANN block or raw-parameter alternative; reduced gantry branch only. | original/unmarked |
| `model_augmentation/fit_systems/blocks.py:275` | `Cascaded_Tanks_State_Block.__init__` | not needed | Different physical/ANN block or raw-parameter alternative; reduced gantry branch only. | original/unmarked |
| `model_augmentation/fit_systems/blocks.py:301` | `Cascaded_Tanks_State_Block.nonlinear_function` | not needed | Different physical/ANN block or raw-parameter alternative; reduced gantry branch only. | original/unmarked |
| `model_augmentation/fit_systems/blocks.py:361` | `Parameterized_Cascaded_Tanks_State_Block` | not needed | Different physical/ANN block or raw-parameter alternative; reduced gantry branch only. | original/unmarked |
| `model_augmentation/fit_systems/blocks.py:362` | `Parameterized_Cascaded_Tanks_State_Block.__init__` | not needed | Different physical/ANN block or raw-parameter alternative; reduced gantry branch only. | original/unmarked |
| `model_augmentation/fit_systems/blocks.py:388` | `Parameterized_Cascaded_Tanks_State_Block.nonlinear_function` | not needed | Different physical/ANN block or raw-parameter alternative; reduced gantry branch only. | original/unmarked |
| `model_augmentation/fit_systems/blocks.py:448` | `Parameterized_MSD_State_Block` | not needed | Different physical/ANN block or raw-parameter alternative; reduced gantry branch only. | original/unmarked |
| `model_augmentation/fit_systems/blocks.py:449` | `Parameterized_MSD_State_Block.__init__` | not needed | Different physical/ANN block or raw-parameter alternative; reduced gantry branch only. | original/unmarked |
| `model_augmentation/fit_systems/blocks.py:503` | `Parameterized_MSD_State_Block.nonlinear_function` | not needed | Different physical/ANN block or raw-parameter alternative; reduced gantry branch only. | original/unmarked |
| `model_augmentation/fit_systems/blocks.py:557` | `Nonlinear_MSD_State_Block` | not needed | Different physical/ANN block or raw-parameter alternative; reduced gantry branch only. | @added on Nonlinear_MSD_State_Block |
| `model_augmentation/fit_systems/blocks.py:558` | `Nonlinear_MSD_State_Block.__init__` | not needed | Different physical/ANN block or raw-parameter alternative; reduced gantry branch only. | @added on Nonlinear_MSD_State_Block |
| `model_augmentation/fit_systems/blocks.py:581` | `Nonlinear_MSD_State_Block.nonlinear_function` | not needed | Different physical/ANN block or raw-parameter alternative; reduced gantry branch only. | @added on Nonlinear_MSD_State_Block |
| `model_augmentation/fit_systems/blocks.py:595` | `Nonlinear_MSD_State_Block.deriv` | not needed | Different physical/ANN block or raw-parameter alternative; reduced gantry branch only. | @added on Nonlinear_MSD_State_Block |
| `model_augmentation/fit_systems/blocks.py:664` | `Gantry_State_Block` | required | ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | @added on Gantry_State_Block; CHANGED:784,815,816,877,898,976 |
| `model_augmentation/fit_systems/blocks.py:699` | `Gantry_State_Block.__init__` | required | ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | @added on Gantry_State_Block |
| `model_augmentation/fit_systems/blocks.py:779` | `Gantry_State_Block.nonlinear_function` | required | ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | @added on Gantry_State_Block; CHANGED:784 |
| `model_augmentation/fit_systems/blocks.py:791` | `Gantry_State_Block._rk4` | required | ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | @added on Gantry_State_Block; CHANGED:784 |
| `model_augmentation/fit_systems/blocks.py:800` | `Gantry_State_Block._mats` | not needed | Overridden by _Trainable_Gantry_State_Block._mats; inherited nonlinear_function dispatches to that override. | @added on Gantry_State_Block |
| `model_augmentation/fit_systems/blocks.py:810` | `Gantry_State_Block.deriv` | not needed | Production _rk4 calls _deriv_with directly. | @added on Gantry_State_Block |
| `model_augmentation/fit_systems/blocks.py:814` | `Gantry_State_Block._deriv_with` | required | ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | @added on Gantry_State_Block; CHANGED:815,816 |
| `model_augmentation/fit_systems/blocks.py:893` | `Gantry_State_Block._deriv_tangent_with` | not needed | Production tangent uses fused _deriv_both_with. | @added on Gantry_State_Block |
| `model_augmentation/fit_systems/blocks.py:902` | `Gantry_State_Block._deriv_both_with` | required | ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | @added on Gantry_State_Block; CHANGED:898 |
| `model_augmentation/fit_systems/blocks.py:979` | `Gantry_State_Block._rk4_with_tangent` | required | ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | @added on Gantry_State_Block; CHANGED:976 |
| `model_augmentation/fit_systems/blocks.py:996` | `_Trainable_Gantry_State_Block` | required | ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | @added on _Trainable_Gantry_State_Block; CHANGED:1032,1080,1092,1129 |
| `model_augmentation/fit_systems/blocks.py:1008` | `_Trainable_Gantry_State_Block.__init__` | required | ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | @added on _Trainable_Gantry_State_Block; CHANGED:1032 |
| `model_augmentation/fit_systems/blocks.py:1042` | `_Trainable_Gantry_State_Block._build_KC` | required | ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | @added on _Trainable_Gantry_State_Block |
| `model_augmentation/fit_systems/blocks.py:1064` | `_Trainable_Gantry_State_Block._build_M1M2` | required | ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | @added on _Trainable_Gantry_State_Block |
| `model_augmentation/fit_systems/blocks.py:1079` | `_Trainable_Gantry_State_Block._mats_from_raw` | required | ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | @added on _Trainable_Gantry_State_Block; CHANGED:1080 |
| `model_augmentation/fit_systems/blocks.py:1102` | `_Trainable_Gantry_State_Block.nonlinear_function` | required | ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | @added on _Trainable_Gantry_State_Block |
| `model_augmentation/fit_systems/blocks.py:1112` | `_Trainable_Gantry_State_Block.pass_mats` | required | ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | @added on _Trainable_Gantry_State_Block |
| `model_augmentation/fit_systems/blocks.py:1122` | `_Trainable_Gantry_State_Block._mats` | required | ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | @added on _Trainable_Gantry_State_Block |
| `model_augmentation/fit_systems/blocks.py:1147` | `_Trainable_Gantry_State_Block.__getstate__` | required | ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | @added on _Trainable_Gantry_State_Block |
| `model_augmentation/fit_systems/blocks.py:1152` | `_Trainable_Gantry_State_Block.__setstate__` | required | ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | @added on _Trainable_Gantry_State_Block |
| `model_augmentation/fit_systems/blocks.py:1156` | `_Trainable_Gantry_State_Block.physical_params` | not needed, diagnostic | May execute in current reporting/refresh monitoring, but excluded numeric diagnostic; ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | @added on _Trainable_Gantry_State_Block |
| `model_augmentation/fit_systems/blocks.py:1169` | `_Trainable_Gantry_State_Block._combos_from_raw` | not needed, diagnostic | May execute in current reporting/refresh monitoring, but excluded numeric diagnostic; ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | @added on _Trainable_Gantry_State_Block |
| `model_augmentation/fit_systems/blocks.py:1183` | `_Trainable_Gantry_State_Block.identifiable_combinations` | not needed | Overridden by reduced implementation. | @added on _Trainable_Gantry_State_Block |
| `model_augmentation/fit_systems/blocks.py:1187` | `_Trainable_Gantry_State_Block._initial_identifiable_combinations` | not needed | Overridden by reduced implementation. | @added on _Trainable_Gantry_State_Block |
| `model_augmentation/fit_systems/blocks.py:1191` | `_Trainable_Gantry_State_Block._combination_training_label` | not needed | Overridden by reduced implementation. | @added on _Trainable_Gantry_State_Block |
| `model_augmentation/fit_systems/blocks.py:1194` | `_Trainable_Gantry_State_Block.param_table` | not needed, diagnostic | May execute in current reporting/refresh monitoring, but excluded numeric diagnostic; ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | @added on _Trainable_Gantry_State_Block |
| `model_augmentation/fit_systems/blocks.py:1214` | `Parameterized_Gantry_State_Block` | not needed | Different physical/ANN block or raw-parameter alternative; reduced gantry branch only. | @added on Parameterized_Gantry_State_Block |
| `model_augmentation/fit_systems/blocks.py:1222` | `Parameterized_Gantry_State_Block.__init__` | not needed | Different physical/ANN block or raw-parameter alternative; reduced gantry branch only. | @added on Parameterized_Gantry_State_Block |
| `model_augmentation/fit_systems/blocks.py:1231` | `Parameterized_Gantry_State_Block._recover_params` | not needed | Different physical/ANN block or raw-parameter alternative; reduced gantry branch only. | @added on Parameterized_Gantry_State_Block |
| `model_augmentation/fit_systems/blocks.py:1235` | `Parameterized_Gantry_State_Block.param_loss` | not needed | Different physical/ANN block or raw-parameter alternative; reduced gantry branch only. | @added on Parameterized_Gantry_State_Block |
| `model_augmentation/fit_systems/blocks.py:1243` | `Parameterized_Gantry_State_Block.estimation_parameters` | not needed | Different physical/ANN block or raw-parameter alternative; reduced gantry branch only. | @added on Parameterized_Gantry_State_Block |
| `model_augmentation/fit_systems/blocks.py:1250` | `Reduced_Gantry_State_Block` | required | ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | @added on Reduced_Gantry_State_Block; CHANGED:1428,1447,1468 |
| `model_augmentation/fit_systems/blocks.py:1304` | `Reduced_Gantry_State_Block.__init__` | required | ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | @added on Reduced_Gantry_State_Block |
| `model_augmentation/fit_systems/blocks.py:1350` | `Reduced_Gantry_State_Block.combos_of` | required | ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | @added on Reduced_Gantry_State_Block |
| `model_augmentation/fit_systems/blocks.py:1365` | `Reduced_Gantry_State_Block.gauge_section` | required | ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | @added on Reduced_Gantry_State_Block |
| `model_augmentation/fit_systems/blocks.py:1385` | `Reduced_Gantry_State_Block._m_diff_bound` | required | ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | @added on Reduced_Gantry_State_Block |
| `model_augmentation/fit_systems/blocks.py:1397` | `Reduced_Gantry_State_Block.combinations_from_free` | required | ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | @added on Reduced_Gantry_State_Block |
| `model_augmentation/fit_systems/blocks.py:1409` | `Reduced_Gantry_State_Block.free_from_combinations` | required interface | Report/evaluation estimation and true-baseline helpers; ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | @added on Reduced_Gantry_State_Block |
| `model_augmentation/fit_systems/blocks.py:1424` | `Reduced_Gantry_State_Block.recover_combinations` | required | ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | @added on Reduced_Gantry_State_Block |
| `model_augmentation/fit_systems/blocks.py:1429` | `Reduced_Gantry_State_Block.transition_from_free` | required | ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | @added on Reduced_Gantry_State_Block; CHANGED:1428 |
| `model_augmentation/fit_systems/blocks.py:1461` | `Reduced_Gantry_State_Block.mats_and_tangent_from_free` | required | ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | @added on Reduced_Gantry_State_Block |
| `model_augmentation/fit_systems/blocks.py:1463` | `Reduced_Gantry_State_Block.mats_and_tangent_from_free._mats` | required | ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | @added on Reduced_Gantry_State_Block |
| `model_augmentation/fit_systems/blocks.py:1470` | `Reduced_Gantry_State_Block.transition_and_tangent_from_free` | not needed | Production correction uses tangent_pair then _rk4_with_tangent directly. | @added on Reduced_Gantry_State_Block; CHANGED:1468 |
| `model_augmentation/fit_systems/blocks.py:1482` | `Reduced_Gantry_State_Block._recover_params` | required | ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | @added on Reduced_Gantry_State_Block |
| `model_augmentation/fit_systems/blocks.py:1494` | `Reduced_Gantry_State_Block.admissibility` | required | ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | @added on Reduced_Gantry_State_Block |
| `model_augmentation/fit_systems/blocks.py:1528` | `Reduced_Gantry_State_Block.param_loss` | required | ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | @added on Reduced_Gantry_State_Block |
| `model_augmentation/fit_systems/blocks.py:1536` | `Reduced_Gantry_State_Block.identifiable_combinations` | not needed, diagnostic | May execute in current reporting/refresh monitoring, but excluded numeric diagnostic; ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | @added on Reduced_Gantry_State_Block |
| `model_augmentation/fit_systems/blocks.py:1541` | `Reduced_Gantry_State_Block._initial_identifiable_combinations` | not needed, diagnostic | May execute in current reporting/refresh monitoring, but excluded numeric diagnostic; ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | @added on Reduced_Gantry_State_Block |
| `model_augmentation/fit_systems/blocks.py:1544` | `Reduced_Gantry_State_Block._combination_training_label` | not needed, diagnostic | May execute in current reporting/refresh monitoring, but excluded numeric diagnostic; ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | @added on Reduced_Gantry_State_Block |
| `model_augmentation/fit_systems/blocks.py:1547` | `Reduced_Gantry_State_Block.physical_params` | not needed, diagnostic | May execute in current reporting/refresh monitoring, but excluded numeric diagnostic; ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | @added on Reduced_Gantry_State_Block |
| `model_augmentation/fit_systems/blocks.py:1551` | `Reduced_Gantry_State_Block.estimation_parameters` | required interface | Report/evaluation estimation and true-baseline helpers; ALL: build_model -> reduced physics / fixed output / ANN; OBC adds tangent methods | @added on Reduced_Gantry_State_Block |

#### `model_augmentation/fit_systems/closed_loop.py`

| Source | Symbol | Classification | Spec and call path / exclusion | Project marker |
| :- | :- | :- | :- | :- |
| `model_augmentation/fit_systems/closed_loop.py:76` | `ControllerBank` | required | ALL: build_closed_loop -> simulator; composed loss/validation -> rollout | file added:60 |
| `model_augmentation/fit_systems/closed_loop.py:122` | `ControllerBank.__init__` | required | ALL: build_closed_loop -> simulator; composed loss/validation -> rollout | file added:60 |
| `model_augmentation/fit_systems/closed_loop.py:150` | `ControllerBank.n_controllers` | required | ALL: build_closed_loop -> simulator; composed loss/validation -> rollout | file added:60 |
| `model_augmentation/fit_systems/closed_loop.py:153` | `ControllerBank.physical_D` | required | ALL: build_closed_loop -> simulator; composed loss/validation -> rollout | file added:60 |
| `model_augmentation/fit_systems/closed_loop.py:158` | `ControllerBank.gather` | required | ALL: build_closed_loop -> simulator; composed loss/validation -> rollout | file added:60 |
| `model_augmentation/fit_systems/closed_loop.py:162` | `ControllerBank.zero_state` | required | ALL: build_closed_loop -> simulator; composed loss/validation -> rollout | file added:60 |
| `model_augmentation/fit_systems/closed_loop.py:167` | `ControllerBank.step` | required | ALL: build_closed_loop -> simulator; composed loss/validation -> rollout | file added:60 |
| `model_augmentation/fit_systems/closed_loop.py:187` | `ControllerBank.check_units` | not needed | Only a diagnostic helper, never called by builder. | file added:60 |
| `model_augmentation/fit_systems/closed_loop.py:216` | `_rollout_segment` | required | ALL: build_closed_loop -> simulator; composed loss/validation -> rollout | file added:60 |
| `model_augmentation/fit_systems/closed_loop.py:245` | `closed_loop_rollout` | required | ALL: build_closed_loop -> simulator; composed loss/validation -> rollout | file added:60 |
| `model_augmentation/fit_systems/closed_loop.py:314` | `window_controller_index` | required | ALL: build_closed_loop -> simulator; composed loss/validation -> rollout | file added:60 |
| `model_augmentation/fit_systems/closed_loop.py:354` | `WindowControllerIndex` | required | ALL: build_closed_loop -> simulator; composed loss/validation -> rollout | file added:60 |
| `model_augmentation/fit_systems/closed_loop.py:370` | `WindowControllerIndex.__init__` | required | ALL: build_closed_loop -> simulator; composed loss/validation -> rollout | file added:60 |
| `model_augmentation/fit_systems/closed_loop.py:377` | `WindowControllerIndex.build` | required | ALL: build_closed_loop -> simulator; composed loss/validation -> rollout | file added:60 |
| `model_augmentation/fit_systems/closed_loop.py:427` | `WindowControllerIndex.verify_by_content` | required | ALL: build_closed_loop -> simulator; composed loss/validation -> rollout | file added:60 |
| `model_augmentation/fit_systems/closed_loop.py:445` | `ClosedLoopSimulator` | required | ALL: build_closed_loop -> simulator; composed loss/validation -> rollout | file added:60 |
| `model_augmentation/fit_systems/closed_loop.py:482` | `ClosedLoopSimulator.__init__` | required | ALL: build_closed_loop -> simulator; composed loss/validation -> rollout | file added:60 |
| `model_augmentation/fit_systems/closed_loop.py:535` | `ClosedLoopSimulator.__getstate__` | required | ALL: build_closed_loop -> simulator; composed loss/validation -> rollout | file added:60 |
| `model_augmentation/fit_systems/closed_loop.py:541` | `ClosedLoopSimulator.eager` | required | ALL: build_closed_loop -> simulator; composed loss/validation -> rollout | file added:60 |
| `model_augmentation/fit_systems/closed_loop.py:559` | `ClosedLoopSimulator.__call__` | required | ALL: build_closed_loop -> simulator; composed loss/validation -> rollout | file added:60 |
| `model_augmentation/fit_systems/closed_loop.py:608` | `ClosedLoopSimulator._note_update_time` | not needed | Timing/recompile diagnostic, called but no method numerical output. | file added:60 |
| `model_augmentation/fit_systems/closed_loop.py:640` | `ClosedLoopSimulator.augment_training_data` | required | ALL: build_closed_loop -> simulator; composed loss/validation -> rollout | file added:60 |
| `model_augmentation/fit_systems/closed_loop.py:647` | `ClosedLoopSimulator._check_val_records` | required | ALL: build_closed_loop -> simulator; composed loss/validation -> rollout | file added:60 |
| `model_augmentation/fit_systems/closed_loop.py:667` | `ClosedLoopSimulator.validation_error` | required | ALL: build_closed_loop -> simulator; composed loss/validation -> rollout | file added:60 |
| `model_augmentation/fit_systems/closed_loop.py:719` | `closed_loop_free_run_rms_batch` | required | ALL: build_closed_loop -> simulator; composed loss/validation -> rollout | file added:60 |
| `model_augmentation/fit_systems/closed_loop.py:817` | `closed_loop_free_run_rms` | not needed | No successful spec call path to this definition. | file added:60 |

#### `model_augmentation/fit_systems/continuous_augmentation.py`

| Source | Symbol | Classification | Spec and call path / exclusion | Project marker |
| :- | :- | :- | :- | :- |
| `model_augmentation/fit_systems/continuous_augmentation.py:20` | `rk4_step` | not needed | Discrete additive correction after baseline RK4; no continuous correction block. | original/unmarked |
| `model_augmentation/fit_systems/continuous_augmentation.py:29` | `CombinedContinuousGantryBlock` | not needed | Discrete additive correction after baseline RK4; no continuous correction block. | original/unmarked |
| `model_augmentation/fit_systems/continuous_augmentation.py:39` | `CombinedContinuousGantryBlock.__init__` | not needed | Discrete additive correction after baseline RK4; no continuous correction block. | original/unmarked |
| `model_augmentation/fit_systems/continuous_augmentation.py:62` | `CombinedContinuousGantryBlock.correction_field` | not needed | Discrete additive correction after baseline RK4; no continuous correction block. | original/unmarked |
| `model_augmentation/fit_systems/continuous_augmentation.py:70` | `CombinedContinuousGantryBlock.derivative` | not needed | Discrete additive correction after baseline RK4; no continuous correction block. | original/unmarked |
| `model_augmentation/fit_systems/continuous_augmentation.py:77` | `CombinedContinuousGantryBlock.forward` | not needed | Discrete additive correction after baseline RK4; no continuous correction block. | original/unmarked |
| `model_augmentation/fit_systems/continuous_augmentation.py:88` | `CombinedContinuousGantryBlock.param_loss` | not needed | Discrete additive correction after baseline RK4; no continuous correction block. | original/unmarked |
| `model_augmentation/fit_systems/continuous_augmentation.py:93` | `CombinedContinuousGantryBlock.__getattr__` | not needed | Discrete additive correction after baseline RK4; no continuous correction block. | original/unmarked |

#### `model_augmentation/fit_systems/gyorok_orth_projection.py`

| Source | Symbol | Classification | Spec and call path / exclusion | Project marker |
| :- | :- | :- | :- | :- |
| `model_augmentation/fit_systems/gyorok_orth_projection.py:48` | `_block_step` | not needed | orth=False; no soft-penalty builder. | file added:37 |
| `model_augmentation/fit_systems/gyorok_orth_projection.py:53` | `gantry_orth_matrix` | not needed | orth=False; no soft-penalty builder. | file added:37 |
| `model_augmentation/fit_systems/gyorok_orth_projection.py:79` | `gantry_orth_matrix.f_single` | not needed | orth=False; no soft-penalty builder. | file added:37 |
| `model_augmentation/fit_systems/gyorok_orth_projection.py:104` | `finite_difference_check` | not needed | orth=False; no soft-penalty builder. | file added:37 |
| `model_augmentation/fit_systems/gyorok_orth_projection.py:138` | `orth_basis` | not needed | orth=False; no soft-penalty builder. | file added:37 |
| `model_augmentation/fit_systems/gyorok_orth_projection.py:151` | `spectrum_report` | not needed | orth=False; no soft-penalty builder. | file added:37 |
| `model_augmentation/fit_systems/gyorok_orth_projection.py:188` | `GyorokOrthPenalty` | not needed | orth=False; no soft-penalty builder. | file added:37 |
| `model_augmentation/fit_systems/gyorok_orth_projection.py:202` | `GyorokOrthPenalty.__init__` | not needed | orth=False; no soft-penalty builder. | file added:37 |
| `model_augmentation/fit_systems/gyorok_orth_projection.py:223` | `GyorokOrthPenalty.Q` | not needed | orth=False; no soft-penalty builder. | file added:37 |
| `model_augmentation/fit_systems/gyorok_orth_projection.py:226` | `GyorokOrthPenalty.penalty_of_field` | not needed | orth=False; no soft-penalty builder. | file added:37 |
| `model_augmentation/fit_systems/gyorok_orth_projection.py:229` | `GyorokOrthPenalty.forward` | not needed | orth=False; no soft-penalty builder. | file added:37 |

#### `model_augmentation/fit_systems/interconnect.py`

| Source | Symbol | Classification | Spec and call path / exclusion | Project marker |
| :- | :- | :- | :- | :- |
| `model_augmentation/fit_systems/interconnect.py:21` | `Interconnect` | required | ALL: build_model -> composed system -> base fit; simulator -> hfn/output_only | original/unmarked; CHANGED:17,18,43,51,84,101,136,148,162,237,257,261,267,279,299 |
| `model_augmentation/fit_systems/interconnect.py:22` | `Interconnect.__init__` | required | ALL: build_model -> composed system -> base fit; simulator -> hfn/output_only | original/unmarked; CHANGED:17,18,43 |
| `model_augmentation/fit_systems/interconnect.py:77` | `Interconnect._apply` | required | ALL: build_model -> composed system -> base fit; simulator -> hfn/output_only | original/unmarked |
| `model_augmentation/fit_systems/interconnect.py:88` | `Interconnect._module_device_dtype` | required | ALL: build_model -> composed system -> base fit; simulator -> hfn/output_only | original/unmarked; CHANGED:84 |
| `model_augmentation/fit_systems/interconnect.py:93` | `Interconnect.init_model` | required | ALL: build_model -> composed system -> base fit; simulator -> hfn/output_only | original/unmarked |
| `model_augmentation/fit_systems/interconnect.py:100` | `Interconnect.forward` | required | ALL: build_model -> composed system -> base fit; simulator -> hfn/output_only | original/unmarked; CHANGED:101,136,148,162 |
| `model_augmentation/fit_systems/interconnect.py:193` | `Interconnect.reset_saved_signals` | not needed | No signal-capture request; save_signals=False. | original/unmarked |
| `model_augmentation/fit_systems/interconnect.py:199` | `Interconnect.init_forward` | required | ALL: build_model -> composed system -> base fit; simulator -> hfn/output_only | original/unmarked; CHANGED:237,257,261,267 |
| `model_augmentation/fit_systems/interconnect.py:280` | `Interconnect._signal_cone` | required | ALL: build_model -> composed system -> base fit; simulator -> hfn/output_only | original/unmarked; CHANGED:279 |
| `model_augmentation/fit_systems/interconnect.py:303` | `Interconnect.output_only` | required | ALL: build_model -> composed system -> base fit; simulator -> hfn/output_only | original/unmarked; CHANGED:299 |
| `model_augmentation/fit_systems/interconnect.py:354` | `Interconnect.init_connection_matrices` | required | ALL: build_model -> composed system -> base fit; simulator -> hfn/output_only | original/unmarked |
| `model_augmentation/fit_systems/interconnect.py:444` | `Interconnect.add_block` | required | ALL: build_model -> composed system -> base fit; simulator -> hfn/output_only | original/unmarked |
| `model_augmentation/fit_systems/interconnect.py:457` | `Interconnect.connect_signals` | required | ALL: build_model -> composed system -> base fit; simulator -> hfn/output_only | original/unmarked |
| `model_augmentation/fit_systems/interconnect.py:474` | `Interconnect.connect_block_signals` | required | ALL: build_model -> composed system -> base fit; simulator -> hfn/output_only | original/unmarked |
| `model_augmentation/fit_systems/interconnect.py:483` | `Interconnect.determine_signal_ix` | required | ALL: build_model -> composed system -> base fit; simulator -> hfn/output_only | original/unmarked |
| `model_augmentation/fit_systems/interconnect.py:493` | `Interconnect.convert_signal_name_to_ix` | required | ALL: build_model -> composed system -> base fit; simulator -> hfn/output_only | original/unmarked |
| `model_augmentation/fit_systems/interconnect.py:505` | `Interconnect.convert_signal_ix_to_name` | not needed | Used only inside debugging=True printing, false in spec. | original/unmarked |
| `model_augmentation/fit_systems/interconnect.py:525` | `Interconnect.parse_connection_function_method` | required | ALL: build_model -> composed system -> base fit; simulator -> hfn/output_only | original/unmarked |
| `model_augmentation/fit_systems/interconnect.py:535` | `Signal_Connection` | required | ALL: build_model -> composed system -> base fit; simulator -> hfn/output_only | original/unmarked |
| `model_augmentation/fit_systems/interconnect.py:537` | `Signal_Connection.__init__` | required | ALL: build_model -> composed system -> base fit; simulator -> hfn/output_only | original/unmarked |
| `model_augmentation/fit_systems/interconnect.py:545` | `Signal_Connection.__str__` | not needed | Debug printing only. | original/unmarked |
| `model_augmentation/fit_systems/interconnect.py:549` | `Signal_Connection.__repr__` | not needed | Debug printing only. | original/unmarked |
| `model_augmentation/fit_systems/interconnect.py:552` | `modified_encoder_net` | not needed | Alternative wrapper/default encoder or statistics helper; composed data_lsq path only. | original/unmarked |
| `model_augmentation/fit_systems/interconnect.py:553` | `modified_encoder_net.__init__` | not needed | Alternative wrapper/default encoder or statistics helper; composed data_lsq path only. | original/unmarked |
| `model_augmentation/fit_systems/interconnect.py:561` | `modified_encoder_net.forward` | not needed | Alternative wrapper/default encoder or statistics helper; composed data_lsq path only. | original/unmarked |
| `model_augmentation/fit_systems/interconnect.py:568` | `Dtype_DataLoader` | required | ALL: build_model -> composed system -> base fit; simulator -> hfn/output_only | @added on Dtype_DataLoader |
| `model_augmentation/fit_systems/interconnect.py:594` | `Dtype_DataLoader.__init__` | required | ALL: build_model -> composed system -> base fit; simulator -> hfn/output_only | @added on Dtype_DataLoader |
| `model_augmentation/fit_systems/interconnect.py:595` | `Dtype_DataLoader.__init__._conv` | required | ALL: build_model -> composed system -> base fit; simulator -> hfn/output_only | @added on Dtype_DataLoader |
| `model_augmentation/fit_systems/interconnect.py:604` | `SSE_Interconnect` | required | ALL: build_model -> composed system -> base fit; simulator -> hfn/output_only | original/unmarked; CHANGED:605,614,673,694,710,742,801,807,858,904,965,1001,1044,1123 |
| `model_augmentation/fit_systems/interconnect.py:628` | `SSE_Interconnect.__init__` | required | ALL: build_model -> composed system -> base fit; simulator -> hfn/output_only | original/unmarked |
| `model_augmentation/fit_systems/interconnect.py:644` | `SSE_Interconnect.init_nets` | required | ALL: build_model -> composed system -> base fit; simulator -> hfn/output_only | original/unmarked |
| `model_augmentation/fit_systems/interconnect.py:651` | `SSE_Interconnect.init_model` | required | ALL: build_model -> composed system -> base fit; simulator -> hfn/output_only | original/unmarked |
| `model_augmentation/fit_systems/interconnect.py:678` | `SSE_Interconnect.simulate` | required | ALL: build_model -> composed system -> base fit; simulator -> hfn/output_only | original/unmarked; CHANGED:673,694 |
| `model_augmentation/fit_systems/interconnect.py:707` | `SSE_Interconnect.loss` | not needed | Overridden by composed.loss without super loss call. | original/unmarked; CHANGED:710 |
| `model_augmentation/fit_systems/interconnect.py:746` | `SSE_Interconnect.make_training_data` | required | ALL: build_model -> composed system -> base fit; simulator -> hfn/output_only | original/unmarked; CHANGED:742 |
| `model_augmentation/fit_systems/interconnect.py:753` | `SSE_Interconnect.cal_validation_error` | required | ALL: build_model -> composed system -> base fit; simulator -> hfn/output_only | original/unmarked |
| `model_augmentation/fit_systems/interconnect.py:770` | `SSE_Interconnect._sync_prediction_state_for_validation` | not needed | Overridden by composed method. | original/unmarked |
| `model_augmentation/fit_systems/interconnect.py:779` | `SSE_Interconnect._model_dtype` | not needed | Only open-loop init_state/measure interface uses it here. | @added on _model_dtype |
| `model_augmentation/fit_systems/interconnect.py:785` | `SSE_Interconnect.init_state_multi` | not needed | No apply_experiment/n_step_error in required closed-loop method. | @added on init_state_multi |
| `model_augmentation/fit_systems/interconnect.py:800` | `SSE_Interconnect.measure_act_multi` | not needed | No open-loop apply_experiment in method. | original/unmarked; CHANGED:801 |
| `model_augmentation/fit_systems/interconnect.py:808` | `SSE_Interconnect.fit` | required | ALL: build_model -> composed system -> base fit; simulator -> hfn/output_only | original/unmarked; CHANGED:801,807,858,904,965,1001,1044,1123 |
| `model_augmentation/fit_systems/interconnect.py:881` | `SSE_Interconnect.fit.validation` | required | ALL: build_model -> composed system -> base fit; simulator -> hfn/output_only | original/unmarked; CHANGED:904 |
| `model_augmentation/fit_systems/interconnect.py:1023` | `SSE_Interconnect.fit.closure` | required | ALL: build_model -> composed system -> base fit; simulator -> hfn/output_only | original/unmarked |
| `model_augmentation/fit_systems/interconnect.py:1155` | `SSE_Interconnect_ParamLoss` | not needed | Alternative wrapper/default encoder or statistics helper; composed data_lsq path only. | @added on SSE_Interconnect_ParamLoss |
| `model_augmentation/fit_systems/interconnect.py:1173` | `SSE_Interconnect_ParamLoss.loss` | not needed | Alternative wrapper/default encoder or statistics helper; composed data_lsq path only. | @added on SSE_Interconnect_ParamLoss |
| `model_augmentation/fit_systems/interconnect.py:1182` | `WindowErrorStats` | not needed | Alternative wrapper/default encoder or statistics helper; composed data_lsq path only. | @added on WindowErrorStats; CHANGED:1222 |
| `model_augmentation/fit_systems/interconnect.py:1204` | `WindowErrorStats.__init__` | not needed | Alternative wrapper/default encoder or statistics helper; composed data_lsq path only. | @added on WindowErrorStats |
| `model_augmentation/fit_systems/interconnect.py:1209` | `WindowErrorStats.reset` | not needed | Alternative wrapper/default encoder or statistics helper; composed data_lsq path only. | @added on WindowErrorStats |
| `model_augmentation/fit_systems/interconnect.py:1220` | `WindowErrorStats.update` | not needed | Alternative wrapper/default encoder or statistics helper; composed data_lsq path only. | @added on WindowErrorStats; CHANGED:1222 |
| `model_augmentation/fit_systems/interconnect.py:1241` | `WindowErrorStats.summary` | not needed | Alternative wrapper/default encoder or statistics helper; composed data_lsq path only. | @added on WindowErrorStats |
| `model_augmentation/fit_systems/interconnect.py:1257` | `SSE_Interconnect_Composed` | required | ALL: build_model -> composed system -> base fit; simulator -> hfn/output_only | @added on SSE_Interconnect_Composed; CHANGED:1410,1417,1507,1517,1596,1619 |
| `model_augmentation/fit_systems/interconnect.py:1305` | `SSE_Interconnect_Composed.__init__` | required | ALL: build_model -> composed system -> base fit; simulator -> hfn/output_only | @added on SSE_Interconnect_Composed |
| `model_augmentation/fit_systems/interconnect.py:1332` | `SSE_Interconnect_Composed._sync_prediction_state_for_validation` | required | ALL: build_model -> composed system -> base fit; simulator -> hfn/output_only | @added on SSE_Interconnect_Composed |
| `model_augmentation/fit_systems/interconnect.py:1383` | `SSE_Interconnect_Composed.loss` | required | ALL: build_model -> composed system -> base fit; simulator -> hfn/output_only | @added on SSE_Interconnect_Composed; CHANGED:1410,1417 |
| `model_augmentation/fit_systems/interconnect.py:1505` | `SSE_Interconnect_Composed.checkpoint_save_system` | required | ALL: build_model -> composed system -> base fit; simulator -> hfn/output_only | @added on SSE_Interconnect_Composed; CHANGED:1507 |
| `model_augmentation/fit_systems/interconnect.py:1514` | `SSE_Interconnect_Composed.checkpoint_load_system` | required | ALL: build_model -> composed system -> base fit; simulator -> hfn/output_only | @added on SSE_Interconnect_Composed; CHANGED:1507,1517 |
| `model_augmentation/fit_systems/interconnect.py:1570` | `SSE_Interconnect_Composed.__getstate__` | required | ALL: build_model -> composed system -> base fit; simulator -> hfn/output_only | @added on SSE_Interconnect_Composed; CHANGED:1596 |
| `model_augmentation/fit_systems/interconnect.py:1599` | `SSE_Interconnect_Composed.__setstate__` | required | ALL: build_model -> composed system -> base fit; simulator -> hfn/output_only | @added on SSE_Interconnect_Composed; CHANGED:1596,1619 |
| `model_augmentation/fit_systems/interconnect.py:1622` | `SSE_Interconnect_Composed.optimizer` | required | ALL: build_model -> composed system -> base fit; simulator -> hfn/output_only | @added on SSE_Interconnect_Composed; CHANGED:1619 |
| `model_augmentation/fit_systems/interconnect.py:1626` | `SSE_Interconnect_Composed.optimizer` | required | ALL: build_model -> composed system -> base fit; simulator -> hfn/output_only | @added on SSE_Interconnect_Composed; CHANGED:1619 |
| `model_augmentation/fit_systems/interconnect.py:1635` | `SSE_Interconnect_Composed._assert_optimizer_supported` | not needed | traj_penalty=None. | @added on SSE_Interconnect_Composed |
| `model_augmentation/fit_systems/interconnect.py:1664` | `SSE_Interconnect_Composed._wrap_optimizer_step` | not needed | traj_penalty=None. | @added on SSE_Interconnect_Composed |
| `model_augmentation/fit_systems/interconnect.py:1672` | `SSE_Interconnect_Composed._wrap_optimizer_step.step_with_penalty` | not needed | traj_penalty=None. | @added on SSE_Interconnect_Composed |
| `model_augmentation/fit_systems/interconnect.py:1682` | `SSE_Interconnect_Composed._wrap_optimizer_step.step_with_penalty.wrapped` | not needed | traj_penalty=None. | @added on SSE_Interconnect_Composed |
| `model_augmentation/fit_systems/interconnect.py:1699` | `SSE_Interconnect_Composed._unwrap_optimizer_step` | not needed | traj_penalty=None fast return in fit. | @added on SSE_Interconnect_Composed |
| `model_augmentation/fit_systems/interconnect.py:1704` | `SSE_Interconnect_Composed.fit` | required | ALL: build_model -> composed system -> base fit; simulator -> hfn/output_only | @added on SSE_Interconnect_Composed |

#### `model_augmentation/fit_systems/lbfgs_polish.py`

| Source | Symbol | Classification | Spec and call path / exclusion | Project marker |
| :- | :- | :- | :- | :- |
| `model_augmentation/fit_systems/lbfgs_polish.py:50` | `PolishSpec` | required | ALL: train_model_with_diagnostics -> run_lbfgs_polish -> lbfgs_polish | file added:33 |
| `model_augmentation/fit_systems/lbfgs_polish.py:78` | `meter_regression` | required | ALL: train_model_with_diagnostics -> run_lbfgs_polish -> lbfgs_polish | file added:33 |
| `model_augmentation/fit_systems/lbfgs_polish.py:97` | `decide` | required | ALL: train_model_with_diagnostics -> run_lbfgs_polish -> lbfgs_polish | file added:33 |
| `model_augmentation/fit_systems/lbfgs_polish.py:134` | `_named_params` | required | ALL: train_model_with_diagnostics -> run_lbfgs_polish -> lbfgs_polish | file added:33 |
| `model_augmentation/fit_systems/lbfgs_polish.py:155` | `_snapshot` | required | ALL: train_model_with_diagnostics -> run_lbfgs_polish -> lbfgs_polish | file added:33 |
| `model_augmentation/fit_systems/lbfgs_polish.py:165` | `_restore` | required | ALL: train_model_with_diagnostics -> run_lbfgs_polish -> lbfgs_polish | file added:33 |
| `model_augmentation/fit_systems/lbfgs_polish.py:171` | `_resync` | required | ALL: train_model_with_diagnostics -> run_lbfgs_polish -> lbfgs_polish | file added:33 |
| `model_augmentation/fit_systems/lbfgs_polish.py:181` | `generic_meters` | not needed | No successful spec call path to this definition. | file added:33 |
| `model_augmentation/fit_systems/lbfgs_polish.py:206` | `window_count` | required | ALL: train_model_with_diagnostics -> run_lbfgs_polish -> lbfgs_polish | file added:33 |
| `model_augmentation/fit_systems/lbfgs_polish.py:234` | `FixedBatch` | required | ALL: train_model_with_diagnostics -> run_lbfgs_polish -> lbfgs_polish | file added:33 |
| `model_augmentation/fit_systems/lbfgs_polish.py:258` | `FixedBatch.__init__` | required | ALL: train_model_with_diagnostics -> run_lbfgs_polish -> lbfgs_polish | file added:33 |
| `model_augmentation/fit_systems/lbfgs_polish.py:296` | `FixedBatch.__init__._to_tensor` | required | ALL: train_model_with_diagnostics -> run_lbfgs_polish -> lbfgs_polish | file added:33 |
| `model_augmentation/fit_systems/lbfgs_polish.py:334` | `FixedBatch.__iter__` | required | ALL: train_model_with_diagnostics -> run_lbfgs_polish -> lbfgs_polish | file added:33 |
| `model_augmentation/fit_systems/lbfgs_polish.py:347` | `_eager` | required | ALL: train_model_with_diagnostics -> run_lbfgs_polish -> lbfgs_polish | file added:33 |
| `model_augmentation/fit_systems/lbfgs_polish.py:373` | `_frozen` | required | ALL: train_model_with_diagnostics -> run_lbfgs_polish -> lbfgs_polish | file added:33 |
| `model_augmentation/fit_systems/lbfgs_polish.py:396` | `_quiet_probes` | required | ALL: train_model_with_diagnostics -> run_lbfgs_polish -> lbfgs_polish | file added:33 |
| `model_augmentation/fit_systems/lbfgs_polish.py:420` | `warm_up_lazy_init` | required | ALL: train_model_with_diagnostics -> run_lbfgs_polish -> lbfgs_polish | file added:33 |
| `model_augmentation/fit_systems/lbfgs_polish.py:464` | `_make_closure` | required | ALL: train_model_with_diagnostics -> run_lbfgs_polish -> lbfgs_polish | file added:33 |
| `model_augmentation/fit_systems/lbfgs_polish.py:466` | `_make_closure.closure` | required | ALL: train_model_with_diagnostics -> run_lbfgs_polish -> lbfgs_polish | file added:33 |
| `model_augmentation/fit_systems/lbfgs_polish.py:483` | `_loss_value` | required | ALL: train_model_with_diagnostics -> run_lbfgs_polish -> lbfgs_polish | file added:33 |
| `model_augmentation/fit_systems/lbfgs_polish.py:500` | `_lbfgs_counts` | required | ALL: train_model_with_diagnostics -> run_lbfgs_polish -> lbfgs_polish | file added:33 |
| `model_augmentation/fit_systems/lbfgs_polish.py:515` | `_is_oom` | required | ALL: train_model_with_diagnostics -> run_lbfgs_polish -> lbfgs_polish | file added:33 |
| `model_augmentation/fit_systems/lbfgs_polish.py:522` | `_probe_closure` | required | ALL: train_model_with_diagnostics -> run_lbfgs_polish -> lbfgs_polish | file added:33 |
| `model_augmentation/fit_systems/lbfgs_polish.py:549` | `lbfgs_polish` | required | ALL: train_model_with_diagnostics -> run_lbfgs_polish -> lbfgs_polish | file added:33 |
| `model_augmentation/fit_systems/lbfgs_polish.py:588` | `lbfgs_polish._all_meters` | required | ALL: train_model_with_diagnostics -> run_lbfgs_polish -> lbfgs_polish | file added:33 |
| `model_augmentation/fit_systems/lbfgs_polish.py:597` | `lbfgs_polish._validate` | required | ALL: train_model_with_diagnostics -> run_lbfgs_polish -> lbfgs_polish | file added:33 |
| `model_augmentation/fit_systems/lbfgs_polish.py:650` | `lbfgs_polish._new_opt` | required | ALL: train_model_with_diagnostics -> run_lbfgs_polish -> lbfgs_polish | file added:33 |

#### `model_augmentation/fit_systems/multiple_shooting.py`

| Source | Symbol | Classification | Spec and call path / exclusion | Project marker |
| :- | :- | :- | :- | :- |
| `model_augmentation/fit_systems/multiple_shooting.py:45` | `SSE_Interconnect_MultipleShooting` | not needed | Composed system, no multiple-shooting wrapper; closed-loop method. | file added:37 |
| `model_augmentation/fit_systems/multiple_shooting.py:96` | `SSE_Interconnect_MultipleShooting._norm` | not needed | Composed system, no multiple-shooting wrapper; closed-loop method. | file added:37 |
| `model_augmentation/fit_systems/multiple_shooting.py:104` | `SSE_Interconnect_MultipleShooting._defect_penalty` | not needed | Composed system, no multiple-shooting wrapper; closed-loop method. | file added:37 |
| `model_augmentation/fit_systems/multiple_shooting.py:166` | `SSE_Interconnect_MultipleShooting.loss` | not needed | Composed system, no multiple-shooting wrapper; closed-loop method. | file added:37 |

#### `model_augmentation/fit_systems/obc.py`

| Source | Symbol | Classification | Spec and call path / exclusion | Project marker |
| :- | :- | :- | :- | :- |
| `model_augmentation/fit_systems/obc.py:44` | `_StaticCopy` | not needed | No successful spec call path to this definition. | file added:34 |
| `model_augmentation/fit_systems/obc.py:67` | `_StaticCopy.forward` | not needed | No successful spec call path to this definition. | file added:34 |
| `model_augmentation/fit_systems/obc.py:74` | `_StaticCopy.backward` | not needed | No successful spec call path to this definition. | file added:34 |
| `model_augmentation/fit_systems/obc.py:78` | `StaticPassBuffers` | not needed | No successful spec call path to this definition. | file added:34 |
| `model_augmentation/fit_systems/obc.py:87` | `StaticPassBuffers.__init__` | not needed | No successful spec call path to this definition. | file added:34 |
| `model_augmentation/fit_systems/obc.py:91` | `StaticPassBuffers._alloc` | not needed | No successful spec call path to this definition. | file added:34 |
| `model_augmentation/fit_systems/obc.py:98` | `StaticPassBuffers.__call__` | not needed | No successful spec call path to this definition. | file added:34 |
| `model_augmentation/fit_systems/obc.py:114` | `StaticPassBuffers.reset` | not needed | No successful spec call path to this definition. | file added:34 |
| `model_augmentation/fit_systems/obc.py:118` | `OBCRankLossAbort` | required | OBC: attach_obc -> lifecycle -> refresh/coefficient -> basis | file added:34 |
| `model_augmentation/fit_systems/obc.py:124` | `RankReport` | required | OBC: attach_obc -> lifecycle -> refresh/coefficient -> basis | file added:34 |
| `model_augmentation/fit_systems/obc.py:139` | `RankReport.lines` | not needed, diagnostic | May execute in current reporting/refresh monitoring, but excluded numeric diagnostic; OBC: attach_obc -> lifecycle -> refresh/coefficient -> basis | file added:34 |
| `model_augmentation/fit_systems/obc.py:160` | `OBCBasis` | required | OBC: attach_obc -> lifecycle -> refresh/coefficient -> basis | file added:34 |
| `model_augmentation/fit_systems/obc.py:169` | `OBCBasis.__init__` | required | OBC: attach_obc -> lifecycle -> refresh/coefficient -> basis | file added:34 |
| `model_augmentation/fit_systems/obc.py:183` | `OBCBasis.rank_tolerance` | required | OBC: attach_obc -> lifecycle -> refresh/coefficient -> basis | file added:34 |
| `model_augmentation/fit_systems/obc.py:191` | `OBCBasis.from_matrix` | required | OBC: attach_obc -> lifecycle -> refresh/coefficient -> basis | file added:34 |
| `model_augmentation/fit_systems/obc.py:242` | `OBCBasis.coefficient` | required | OBC: attach_obc -> lifecycle -> refresh/coefficient -> basis | file added:34 |
| `model_augmentation/fit_systems/obc.py:250` | `OBCBasis.apply` | not needed, diagnostic | Used to calculate validation/standalone projected-field diagnostics; coefficient solve does not use apply. | file added:34 |
| `model_augmentation/fit_systems/obc.py:257` | `OBCBasis.jt` | not needed, diagnostic | Used by r_perp diagnostic only. | file added:34 |
| `model_augmentation/fit_systems/obc.py:262` | `OBCBasis.project_onto` | not needed | Used by standalone overlap diagnostics; production validation uses U.T directly. | file added:34 |
| `model_augmentation/fit_systems/obc.py:267` | `OBCBasis.project_out` | not needed | Standalone diagnostic helper. | file added:34 |
| `model_augmentation/fit_systems/obc.py:272` | `OBCBasis.r_perp` | not needed, diagnostic | Validation/standalone orthogonality meter; does not affect coefficient or rank decision. | file added:34 |
| `model_augmentation/fit_systems/obc.py:280` | `OBCBasis.r_overlap` | not needed | Standalone diagnostic helper (rho alias has same status). | file added:34 |
| `model_augmentation/fit_systems/obc.py:289` | `OBCBasis.frobenius` | not needed, diagnostic | Denominator of r_perp and relative-change monitoring only. | file added:34 |
| `model_augmentation/fit_systems/obc.py:295` | `OBCBasis.relative_change` | not needed, diagnostic | May execute in current reporting/refresh monitoring, but excluded numeric diagnostic; OBC: attach_obc -> lifecycle -> refresh/coefficient -> basis | file added:34 |
| `model_augmentation/fit_systems/obc.py:322` | `OBCBasis.to` | required | OBC: attach_obc -> lifecycle -> refresh/coefficient -> basis | file added:34 |
| `model_augmentation/fit_systems/obc.py:330` | `OBCBasis.device` | required | OBC: attach_obc -> lifecycle -> refresh/coefficient -> basis | file added:34 |
| `model_augmentation/fit_systems/obc.py:333` | `OBCBasis.memory_bytes` | not needed, diagnostic | May execute in current reporting/refresh monitoring, but excluded numeric diagnostic; OBC: attach_obc -> lifecycle -> refresh/coefficient -> basis | file added:34 |
| `model_augmentation/fit_systems/obc.py:341` | `build_stacked_sensitivity` | required | OBC: attach_obc -> lifecycle -> refresh/coefficient -> basis | file added:34 |
| `model_augmentation/fit_systems/obc.py:367` | `evaluate_step_rows` | required | OBC: attach_obc -> lifecycle -> refresh/coefficient -> basis | file added:34 |
| `model_augmentation/fit_systems/obc.py:380` | `directional_correction` | not needed | No successful spec call path to this definition. | file added:34 |
| `model_augmentation/fit_systems/obc.py:392` | `OBCLifecycle` | required | OBC: attach_obc -> lifecycle -> refresh/coefficient -> basis | file added:34 |
| `model_augmentation/fit_systems/obc.py:410` | `OBCLifecycle.__init__` | required | OBC: attach_obc -> lifecycle -> refresh/coefficient -> basis | file added:34 |
| `model_augmentation/fit_systems/obc.py:446` | `OBCLifecycle.refresh` | required | OBC: attach_obc -> lifecycle -> refresh/coefficient -> basis | file added:34 |
| `model_augmentation/fit_systems/obc.py:501` | `OBCLifecycle.maybe_refresh` | required | OBC: attach_obc -> lifecycle -> refresh/coefficient -> basis | file added:34 |
| `model_augmentation/fit_systems/obc.py:512` | `OBCLifecycle.coefficient` | required | OBC: attach_obc -> lifecycle -> refresh/coefficient -> basis | file added:34 |
| `model_augmentation/fit_systems/obc.py:533` | `OBCLifecycle.prediction_coefficient` | not needed | PS residual synchronization currently calls validation_coefficient, not this optimization. | file added:34 |
| `model_augmentation/fit_systems/obc.py:543` | `OBCLifecycle.validation_coefficient` | required | OBC: attach_obc -> lifecycle -> refresh/coefficient -> basis | file added:34 |
| `model_augmentation/fit_systems/obc.py:558` | `OBCLifecycle._prediction_solve` | required | OBC: attach_obc -> lifecycle -> refresh/coefficient -> basis | file added:34 |
| `model_augmentation/fit_systems/obc.py:594` | `OBCLifecycle.stacked_diagnostics` | not needed | Standalone diagnostic API. | file added:34 |
| `model_augmentation/fit_systems/obc.py:607` | `OBCLifecycle.memory_bytes` | not needed | Not called by attached pipeline. | file added:34 |
| `model_augmentation/fit_systems/obc.py:610` | `OBCLifecycle.state_summary` | not needed | Not called by attached pipeline. | file added:34 |

#### `model_augmentation/fit_systems/obc_projection.py`

| Source | Symbol | Classification | Spec and call path / exclusion | Project marker |
| :- | :- | :- | :- | :- |
| `model_augmentation/fit_systems/obc_projection.py:91` | `reduced_step` | not needed | Legacy projected-block implementation; active OBC is obc.py plus obc_gantry.py. | file added:3 |
| `model_augmentation/fit_systems/obc_projection.py:100` | `stacked_basis` | not needed | Legacy projected-block implementation; active OBC is obc.py plus obc_gantry.py. | file added:3 |
| `model_augmentation/fit_systems/obc_projection.py:134` | `stacked_basis.f` | not needed | Legacy projected-block implementation; active OBC is obc.py plus obc_gantry.py. | file added:3 |
| `model_augmentation/fit_systems/obc_projection.py:143` | `directional` | not needed | Legacy projected-block implementation; active OBC is obc.py plus obc_gantry.py. | file added:3 |
| `model_augmentation/fit_systems/obc_projection.py:161` | `rank_report` | not needed | Legacy projected-block implementation; active OBC is obc.py plus obc_gantry.py. | file added:3 |
| `model_augmentation/fit_systems/obc_projection.py:185` | `principal_angles` | not needed | Legacy projected-block implementation; active OBC is obc.py plus obc_gantry.py. | file added:3 |
| `model_augmentation/fit_systems/obc_projection.py:193` | `RankLossAbort` | not needed | Legacy projected-block implementation; active OBC is obc.py plus obc_gantry.py. | file added:3 |
| `model_augmentation/fit_systems/obc_projection.py:204` | `OBCBasis` | not needed | Legacy projected-block implementation; active OBC is obc.py plus obc_gantry.py. | file added:3 |
| `model_augmentation/fit_systems/obc_projection.py:211` | `OBCBasis.__init__` | not needed | Legacy projected-block implementation; active OBC is obc.py plus obc_gantry.py. | file added:3 |
| `model_augmentation/fit_systems/obc_projection.py:238` | `OBCBasis._check_rank` | not needed | Legacy projected-block implementation; active OBC is obc.py plus obc_gantry.py. | file added:3 |
| `model_augmentation/fit_systems/obc_projection.py:266` | `OBCBasis.range_basis` | not needed | Legacy projected-block implementation; active OBC is obc.py plus obc_gantry.py. | file added:3 |
| `model_augmentation/fit_systems/obc_projection.py:270` | `OBCBasis.coefficient` | not needed | Legacy projected-block implementation; active OBC is obc.py plus obc_gantry.py. | file added:3 |
| `model_augmentation/fit_systems/obc_projection.py:274` | `OBCBasis.rho` | not needed | Legacy projected-block implementation; active OBC is obc.py plus obc_gantry.py. | file added:3 |
| `model_augmentation/fit_systems/obc_projection.py:285` | `OBCBasis.residual_ratio` | not needed | Legacy projected-block implementation; active OBC is obc.py plus obc_gantry.py. | file added:3 |
| `model_augmentation/fit_systems/obc_projection.py:293` | `OBCBasis.spectrum` | not needed | Legacy projected-block implementation; active OBC is obc.py plus obc_gantry.py. | file added:3 |
| `model_augmentation/fit_systems/obc_projection.py:299` | `OBC_Projected_ANN_Block` | not needed | Legacy projected-block implementation; active OBC is obc.py plus obc_gantry.py. | file added:3 |
| `model_augmentation/fit_systems/obc_projection.py:326` | `OBC_Projected_ANN_Block.__init__` | not needed | Legacy projected-block implementation; active OBC is obc.py plus obc_gantry.py. | file added:3 |
| `model_augmentation/fit_systems/obc_projection.py:360` | `OBC_Projected_ANN_Block.phys` | not needed | Legacy projected-block implementation; active OBC is obc.py plus obc_gantry.py. | file added:3 |
| `model_augmentation/fit_systems/obc_projection.py:363` | `OBC_Projected_ANN_Block.init_block` | not needed | Legacy projected-block implementation; active OBC is obc.py plus obc_gantry.py. | file added:3 |
| `model_augmentation/fit_systems/obc_projection.py:367` | `OBC_Projected_ANN_Block.phys_input` | not needed | Legacy projected-block implementation; active OBC is obc.py plus obc_gantry.py. | file added:3 |
| `model_augmentation/fit_systems/obc_projection.py:371` | `OBC_Projected_ANN_Block.rebuild_basis` | not needed | Legacy projected-block implementation; active OBC is obc.py plus obc_gantry.py. | file added:3 |
| `model_augmentation/fit_systems/obc_projection.py:385` | `OBC_Projected_ANN_Block.clear_basis` | not needed | Legacy projected-block implementation; active OBC is obc.py plus obc_gantry.py. | file added:3 |
| `model_augmentation/fit_systems/obc_projection.py:389` | `OBC_Projected_ANN_Block._stack_physical_write` | not needed | Legacy projected-block implementation; active OBC is obc.py plus obc_gantry.py. | file added:3 |
| `model_augmentation/fit_systems/obc_projection.py:398` | `OBC_Projected_ANN_Block.coefficient` | not needed | Legacy projected-block implementation; active OBC is obc.py plus obc_gantry.py. | file added:3 |
| `model_augmentation/fit_systems/obc_projection.py:402` | `OBC_Projected_ANN_Block.forward` | not needed | Legacy projected-block implementation; active OBC is obc.py plus obc_gantry.py. | file added:3 |

#### `model_augmentation/fit_systems/orth_projection.py`

| Source | Symbol | Classification | Spec and call path / exclusion | Project marker |
| :- | :- | :- | :- | :- |
| `model_augmentation/fit_systems/orth_projection.py:24` | `OrthProjectionPenalty` | not needed | orth=False; no OrthProjectionPenalty or OrthLoss. | file added:15 |
| `model_augmentation/fit_systems/orth_projection.py:54` | `OrthProjectionPenalty.__init__` | not needed | orth=False; no OrthProjectionPenalty or OrthLoss. | file added:15 |
| `model_augmentation/fit_systems/orth_projection.py:61` | `OrthProjectionPenalty.penalty_of_field` | not needed | orth=False; no OrthProjectionPenalty or OrthLoss. | file added:15 |
| `model_augmentation/fit_systems/orth_projection.py:70` | `OrthProjectionPenalty.forward` | not needed | orth=False; no OrthProjectionPenalty or OrthLoss. | file added:15 |
| `model_augmentation/fit_systems/orth_projection.py:76` | `SSE_Interconnect_OrthLoss` | not needed | orth=False; no OrthProjectionPenalty or OrthLoss. | file added:15 |
| `model_augmentation/fit_systems/orth_projection.py:93` | `SSE_Interconnect_OrthLoss.loss` | not needed | orth=False; no OrthProjectionPenalty or OrthLoss. | file added:15 |

#### `model_augmentation/fit_systems/pre_encoder.py`

| Source | Symbol | Classification | Spec and call path / exclusion | Project marker |
| :- | :- | :- | :- | :- |
| `model_augmentation/fit_systems/pre_encoder.py:23` | `convert_to_same_type` | not needed | Only linear_encoder_init_aug is constructed; no x-supervision/pre-encoder/legacy encoder. | original/unmarked |
| `model_augmentation/fit_systems/pre_encoder.py:37` | `System_data_with_x` | not needed | Only linear_encoder_init_aug is constructed; no x-supervision/pre-encoder/legacy encoder. | original/unmarked |
| `model_augmentation/fit_systems/pre_encoder.py:38` | `System_data_with_x.__init__` | not needed | Only linear_encoder_init_aug is constructed; no x-supervision/pre-encoder/legacy encoder. | original/unmarked |
| `model_augmentation/fit_systems/pre_encoder.py:43` | `System_data_with_x.to_hist_future_data` | not needed | Only linear_encoder_init_aug is constructed; no x-supervision/pre-encoder/legacy encoder. | original/unmarked |
| `model_augmentation/fit_systems/pre_encoder.py:104` | `System_data_norm_with_x` | not needed | Only linear_encoder_init_aug is constructed; no x-supervision/pre-encoder/legacy encoder. | @added on System_data_norm_with_x |
| `model_augmentation/fit_systems/pre_encoder.py:119` | `System_data_norm_with_x.__init__` | not needed | Only linear_encoder_init_aug is constructed; no x-supervision/pre-encoder/legacy encoder. | @added on System_data_norm_with_x |
| `model_augmentation/fit_systems/pre_encoder.py:124` | `System_data_norm_with_x.transform` | not needed | Only linear_encoder_init_aug is constructed; no x-supervision/pre-encoder/legacy encoder. | @added on System_data_norm_with_x |
| `model_augmentation/fit_systems/pre_encoder.py:158` | `System_data_norm_with_x.inverse_transform` | not needed | Only linear_encoder_init_aug is constructed; no x-supervision/pre-encoder/legacy encoder. | @added on System_data_norm_with_x |
| `model_augmentation/fit_systems/pre_encoder.py:193` | `linear_encoder_init` | not needed | Only linear_encoder_init_aug is constructed; no x-supervision/pre-encoder/legacy encoder. | original/unmarked |
| `model_augmentation/fit_systems/pre_encoder.py:194` | `linear_encoder_init.__init__` | not needed | Only linear_encoder_init_aug is constructed; no x-supervision/pre-encoder/legacy encoder. | original/unmarked |
| `model_augmentation/fit_systems/pre_encoder.py:281` | `linear_encoder_init.forward` | not needed | Only linear_encoder_init_aug is constructed; no x-supervision/pre-encoder/legacy encoder. | original/unmarked |
| `model_augmentation/fit_systems/pre_encoder.py:306` | `linear_encoder_init_aug` | required | ALL: build_model(data_lsq) -> linear_encoder_init_aug -> encoder.forward | @added on linear_encoder_init_aug; CHANGED:337,343,349,352,412,416,456,463,471,489,497,506 |
| `model_augmentation/fit_systems/pre_encoder.py:321` | `linear_encoder_init_aug.__init__` | required | ALL: build_model(data_lsq) -> linear_encoder_init_aug -> encoder.forward | @added on linear_encoder_init_aug; CHANGED:337,343,349,352,412,416,456,463,471 |
| `model_augmentation/fit_systems/pre_encoder.py:488` | `linear_encoder_init_aug.forward` | required | ALL: build_model(data_lsq) -> linear_encoder_init_aug -> encoder.forward | @added on linear_encoder_init_aug; CHANGED:489,497,506 |
| `model_augmentation/fit_systems/pre_encoder.py:527` | `SS_pre_encoder` | not needed | Only linear_encoder_init_aug is constructed; no x-supervision/pre-encoder/legacy encoder. | original/unmarked |
| `model_augmentation/fit_systems/pre_encoder.py:536` | `SS_pre_encoder.__init__` | not needed | Only linear_encoder_init_aug is constructed; no x-supervision/pre-encoder/legacy encoder. | original/unmarked |
| `model_augmentation/fit_systems/pre_encoder.py:571` | `SS_pre_encoder.init_nets` | not needed | Only linear_encoder_init_aug is constructed; no x-supervision/pre-encoder/legacy encoder. | original/unmarked |
| `model_augmentation/fit_systems/pre_encoder.py:587` | `SS_pre_encoder.loss` | not needed | Only linear_encoder_init_aug is constructed; no x-supervision/pre-encoder/legacy encoder. | original/unmarked |
| `model_augmentation/fit_systems/pre_encoder.py:600` | `SS_pre_encoder.n_step_error` | not needed | Only linear_encoder_init_aug is constructed; no x-supervision/pre-encoder/legacy encoder. | original/unmarked |

#### `model_augmentation/fit_systems/ps2_opening.py`

| Source | Symbol | Classification | Spec and call path / exclusion | Project marker |
| :- | :- | :- | :- | :- |
| `model_augmentation/fit_systems/ps2_opening.py:37` | `PS2Spec` | required | PS: training -> PS2Opening.for_pipeline -> base fit.after_step; S2 conditional | file added:25 |
| `model_augmentation/fit_systems/ps2_opening.py:59` | `calibration_split` | required | PS: training -> PS2Opening.for_pipeline -> base fit.after_step; S2 conditional | file added:25 |
| `model_augmentation/fit_systems/ps2_opening.py:69` | `PS2Opening` | required | PS: training -> PS2Opening.for_pipeline -> base fit.after_step; S2 conditional | file added:25 |
| `model_augmentation/fit_systems/ps2_opening.py:70` | `PS2Opening.__init__` | required | PS: training -> PS2Opening.for_pipeline -> base fit.after_step; S2 conditional | file added:25 |
| `model_augmentation/fit_systems/ps2_opening.py:87` | `PS2Opening.for_pipeline` | required | PS: training -> PS2Opening.for_pipeline -> base fit.after_step; S2 conditional | file added:25 |
| `model_augmentation/fit_systems/ps2_opening.py:98` | `PS2Opening.done` | required | PS: training -> PS2Opening.for_pipeline -> base fit.after_step; S2 conditional | file added:25 |
| `model_augmentation/fit_systems/ps2_opening.py:101` | `PS2Opening.after_step` | required | PS: training -> PS2Opening.for_pipeline -> base fit.after_step; S2 conditional | file added:25 |
| `model_augmentation/fit_systems/ps2_opening.py:130` | `PS2Opening.summary` | required | PS: training -> PS2Opening.for_pipeline -> base fit.after_step; S2 conditional | file added:25 |
| `model_augmentation/fit_systems/ps2_opening.py:135` | `PS2Opening.summary_json` | required | PS: training -> PS2Opening.for_pipeline -> base fit.after_step; S2 conditional | file added:25 |
| `model_augmentation/fit_systems/ps2_opening.py:139` | `PS2Opening._device` | required | PS: training -> PS2Opening.for_pipeline -> base fit.after_step; S2 conditional | file added:25 |
| `model_augmentation/fit_systems/ps2_opening.py:143` | `PS2Opening._start` | required | PS: training -> PS2Opening.for_pipeline -> base fit.after_step; S2 conditional | file added:25 |
| `model_augmentation/fit_systems/ps2_opening.py:161` | `PS2Opening._sample` | required | PS: training -> PS2Opening.for_pipeline -> base fit.after_step; S2 conditional | file added:25 |
| `model_augmentation/fit_systems/ps2_opening.py:183` | `PS2Opening._residual` | required | PS: training -> PS2Opening.for_pipeline -> base fit.after_step; S2 conditional | file added:25 |
| `model_augmentation/fit_systems/ps2_opening.py:191` | `PS2Opening._xa` | required | PS: training -> PS2Opening.for_pipeline -> base fit.after_step; S2 conditional | file added:25 |
| `model_augmentation/fit_systems/ps2_opening.py:194` | `PS2Opening._s1_step` | required | PS: training -> PS2Opening.for_pipeline -> base fit.after_step; S2 conditional | file added:25 |
| `model_augmentation/fit_systems/ps2_opening.py:207` | `PS2Opening._cal_unexplained` | required | PS: training -> PS2Opening.for_pipeline -> base fit.after_step; S2 conditional | file added:25 |
| `model_augmentation/fit_systems/ps2_opening.py:213` | `PS2Opening._fit_map` | required | PS: training -> PS2Opening.for_pipeline -> base fit.after_step; S2 conditional | file added:25 |
| `model_augmentation/fit_systems/ps2_opening.py:220` | `PS2Opening._fit_map.snap` | required | PS: training -> PS2Opening.for_pipeline -> base fit.after_step; S2 conditional | file added:25 |
| `model_augmentation/fit_systems/ps2_opening.py:226` | `PS2Opening._fit_map.ee` | required | PS: training -> PS2Opening.for_pipeline -> base fit.after_step; S2 conditional | file added:25 |

#### `model_augmentation/fit_systems/white_box_models.py`

| Source | Symbol | Classification | Spec and call path / exclusion | Project marker |
| :- | :- | :- | :- | :- |
| `model_augmentation/fit_systems/white_box_models.py:9` | `Discrete_White_Box_Model` | not needed | Definitions imported by package initializer, never instantiated by gantry pipeline. | original/unmarked |
| `model_augmentation/fit_systems/white_box_models.py:10` | `Discrete_White_Box_Model.__init__` | not needed | Definitions imported by package initializer, never instantiated by gantry pipeline. | original/unmarked |
| `model_augmentation/fit_systems/white_box_models.py:18` | `Discrete_White_Box_Model.forward` | not needed | Definitions imported by package initializer, never instantiated by gantry pipeline. | original/unmarked |
| `model_augmentation/fit_systems/white_box_models.py:33` | `Discrete_White_Box_Model.f` | not needed | Definitions imported by package initializer, never instantiated by gantry pipeline. | original/unmarked |
| `model_augmentation/fit_systems/white_box_models.py:36` | `Discrete_White_Box_Model.h` | not needed | Definitions imported by package initializer, never instantiated by gantry pipeline. | original/unmarked |
| `model_augmentation/fit_systems/white_box_models.py:39` | `Discrete_Cascaded_Tanks` | not needed | Definitions imported by package initializer, never instantiated by gantry pipeline. | original/unmarked |
| `model_augmentation/fit_systems/white_box_models.py:40` | `Discrete_Cascaded_Tanks.__init__` | not needed | Definitions imported by package initializer, never instantiated by gantry pipeline. | original/unmarked |
| `model_augmentation/fit_systems/white_box_models.py:57` | `Discrete_Cascaded_Tanks.init_model` | not needed | Definitions imported by package initializer, never instantiated by gantry pipeline. | original/unmarked |
| `model_augmentation/fit_systems/white_box_models.py:66` | `Discrete_Cascaded_Tanks.f` | not needed | Definitions imported by package initializer, never instantiated by gantry pipeline. | original/unmarked |
| `model_augmentation/fit_systems/white_box_models.py:80` | `Discrete_Cascaded_Tanks.h` | not needed | Definitions imported by package initializer, never instantiated by gantry pipeline. | original/unmarked |

#### `model_augmentation/systems/__init__.py`
No function or class definitions. Import/bootstrap status: required package bootstrap; imports define framework classes but do not instantiate benchmark models. See A5.

#### `model_augmentation/systems/cascaded_tanks.py`

| Source | Symbol | Classification | Spec and call path / exclusion | Project marker |
| :- | :- | :- | :- | :- |
| `model_augmentation/systems/cascaded_tanks.py:5` | `Cascaded_Tanks` | not needed | Different benchmark system, never instantiated. | original/unmarked |
| `model_augmentation/systems/cascaded_tanks.py:6` | `Cascaded_Tanks.__init__` | not needed | Different benchmark system, never instantiated. | original/unmarked |
| `model_augmentation/systems/cascaded_tanks.py:14` | `Cascaded_Tanks.deriv` | not needed | Different benchmark system, never instantiated. | original/unmarked |
| `model_augmentation/systems/cascaded_tanks.py:24` | `Cascaded_Tanks.h` | not needed | Different benchmark system, never instantiated. | original/unmarked |
| `model_augmentation/systems/cascaded_tanks.py:27` | `Extended_Cascaded_Tanks` | not needed | Different benchmark system, never instantiated. | original/unmarked |
| `model_augmentation/systems/cascaded_tanks.py:28` | `Extended_Cascaded_Tanks.__init__` | not needed | Different benchmark system, never instantiated. | original/unmarked |
| `model_augmentation/systems/cascaded_tanks.py:37` | `Extended_Cascaded_Tanks.deriv` | not needed | Different benchmark system, never instantiated. | original/unmarked |
| `model_augmentation/systems/cascaded_tanks.py:49` | `Extended_Cascaded_Tanks.h` | not needed | Different benchmark system, never instantiated. | original/unmarked |

#### `model_augmentation/systems/gantry_linearization.py`

| Source | Symbol | Classification | Spec and call path / exclusion | Project marker |
| :- | :- | :- | :- | :- |
| `model_augmentation/systems/gantry_linearization.py:52` | `gantry_linearize_and_discretize` | not needed | Imported by model.py, but encoder_init=data_lsq never calls analytical map. | file added:1 |
| `model_augmentation/systems/gantry_linearization.py:111` | `verify_linearization` | not needed | Imported by model.py, but encoder_init=data_lsq never calls analytical map. | file added:1 |
| `model_augmentation/systems/gantry_linearization.py:146` | `verify_linearization.deriv` | not needed | Imported by model.py, but encoder_init=data_lsq never calls analytical map. | file added:1 |

#### `model_augmentation/systems/gantry_ss.py`

| Source | Symbol | Classification | Spec and call path / exclusion | Project marker |
| :- | :- | :- | :- | :- |
| `model_augmentation/systems/gantry_ss.py:150` | `build_poly_constants` | required | ALL: gantry block construction/matrix rebuild | file added:1 |
| `model_augmentation/systems/gantry_ss.py:184` | `build_G_matrix_entries` | required | ALL: gantry block construction/matrix rebuild | file added:1 |

#### `model_augmentation/systems/mass_spring_damper.py`

| Source | Symbol | Classification | Spec and call path / exclusion | Project marker |
| :- | :- | :- | :- | :- |
| `model_augmentation/systems/mass_spring_damper.py:5` | `Mass_spring_damper` | not needed | Benchmark system imported by systems initializer, never instantiated. | original/unmarked |
| `model_augmentation/systems/mass_spring_damper.py:6` | `Mass_spring_damper.__init__` | not needed | Benchmark system imported by systems initializer, never instantiated. | original/unmarked |
| `model_augmentation/systems/mass_spring_damper.py:14` | `Mass_spring_damper.deriv` | not needed | Benchmark system imported by systems initializer, never instantiated. | original/unmarked |
| `model_augmentation/systems/mass_spring_damper.py:25` | `Mass_spring_damper.h` | not needed | Benchmark system imported by systems initializer, never instantiated. | original/unmarked |
| `model_augmentation/systems/mass_spring_damper.py:29` | `Msd_2dof` | not needed | Benchmark system imported by systems initializer, never instantiated. | original/unmarked |
| `model_augmentation/systems/mass_spring_damper.py:30` | `Msd_2dof.__init__` | not needed | Benchmark system imported by systems initializer, never instantiated. | original/unmarked |
| `model_augmentation/systems/mass_spring_damper.py:41` | `Msd_2dof.deriv` | not needed | Benchmark system imported by systems initializer, never instantiated. | original/unmarked |
| `model_augmentation/systems/mass_spring_damper.py:62` | `Msd_2dof.h` | not needed | Benchmark system imported by systems initializer, never instantiated. | original/unmarked |
| `model_augmentation/systems/mass_spring_damper.py:66` | `Msd_ndof` | not needed | Benchmark system imported by systems initializer, never instantiated. | original/unmarked |
| `model_augmentation/systems/mass_spring_damper.py:67` | `Msd_ndof.__init__` | not needed | Benchmark system imported by systems initializer, never instantiated. | original/unmarked |
| `model_augmentation/systems/mass_spring_damper.py:83` | `Msd_ndof.cast_parameter_to_array` | not needed | Benchmark system imported by systems initializer, never instantiated. | original/unmarked |
| `model_augmentation/systems/mass_spring_damper.py:90` | `Msd_ndof.determine_mask` | not needed | Benchmark system imported by systems initializer, never instantiated. | original/unmarked |
| `model_augmentation/systems/mass_spring_damper.py:98` | `Msd_ndof.deriv` | not needed | Benchmark system imported by systems initializer, never instantiated. | original/unmarked |
| `model_augmentation/systems/mass_spring_damper.py:158` | `Msd_ndof.h` | not needed | Benchmark system imported by systems initializer, never instantiated. | original/unmarked |
| `model_augmentation/systems/mass_spring_damper.py:163` | `Msd_ndof.expand_with_ix` | not needed | Benchmark system imported by systems initializer, never instantiated. | original/unmarked |
| `model_augmentation/systems/mass_spring_damper.py:169` | `Msd_ndof_lpf_output` | not needed | Benchmark system imported by systems initializer, never instantiated. | original/unmarked |
| `model_augmentation/systems/mass_spring_damper.py:170` | `Msd_ndof_lpf_output.__init__` | not needed | Benchmark system imported by systems initializer, never instantiated. | original/unmarked |
| `model_augmentation/systems/mass_spring_damper.py:184` | `Msd_ndof_lpf_output.cast_parameter_to_array` | not needed | Benchmark system imported by systems initializer, never instantiated. | original/unmarked |
| `model_augmentation/systems/mass_spring_damper.py:191` | `Msd_ndof_lpf_output.determine_mask` | not needed | Benchmark system imported by systems initializer, never instantiated. | original/unmarked |
| `model_augmentation/systems/mass_spring_damper.py:199` | `Msd_ndof_lpf_output.deriv` | not needed | Benchmark system imported by systems initializer, never instantiated. | original/unmarked |
| `model_augmentation/systems/mass_spring_damper.py:245` | `Msd_ndof_lpf_output.h` | not needed | Benchmark system imported by systems initializer, never instantiated. | original/unmarked |
| `model_augmentation/systems/mass_spring_damper.py:258` | `Msd_ndof_lpf_output.expand_with_ix` | not needed | Benchmark system imported by systems initializer, never instantiated. | original/unmarked |
| `model_augmentation/systems/mass_spring_damper.py:264` | `Msd_ndof_input_saturation` | not needed | Benchmark system imported by systems initializer, never instantiated. | original/unmarked |
| `model_augmentation/systems/mass_spring_damper.py:265` | `Msd_ndof_input_saturation.__init__` | not needed | Benchmark system imported by systems initializer, never instantiated. | original/unmarked |
| `model_augmentation/systems/mass_spring_damper.py:277` | `Msd_ndof_input_saturation.cast_parameter_to_array` | not needed | Benchmark system imported by systems initializer, never instantiated. | original/unmarked |
| `model_augmentation/systems/mass_spring_damper.py:284` | `Msd_ndof_input_saturation.determine_mask` | not needed | Benchmark system imported by systems initializer, never instantiated. | original/unmarked |
| `model_augmentation/systems/mass_spring_damper.py:292` | `Msd_ndof_input_saturation.deriv` | not needed | Benchmark system imported by systems initializer, never instantiated. | original/unmarked |
| `model_augmentation/systems/mass_spring_damper.py:339` | `Msd_ndof_input_saturation.h` | not needed | Benchmark system imported by systems initializer, never instantiated. | original/unmarked |
| `model_augmentation/systems/mass_spring_damper.py:344` | `Msd_ndof_input_saturation.expand_with_ix` | not needed | Benchmark system imported by systems initializer, never instantiated. | original/unmarked |

#### `model_augmentation/utils/__init__.py`
No function or class definitions. Import/bootstrap status: required package bootstrap; imports define framework classes but do not instantiate benchmark models. See A5.

#### `model_augmentation/utils/deepSI_corrections.py`

| Source | Symbol | Classification | Spec and call path / exclusion | Project marker |
| :- | :- | :- | :- | :- |
| `model_augmentation/utils/deepSI_corrections.py:6` | `fixed_System_data` | required | ALL: composed system -> fixed norm -> transform -> fixed window arrays | file added:1 |
| `model_augmentation/utils/deepSI_corrections.py:7` | `fixed_System_data.__init__` | required | ALL: composed system -> fixed norm -> transform -> fixed window arrays | file added:1 |
| `model_augmentation/utils/deepSI_corrections.py:10` | `fixed_System_data.to_hist_future_data` | required | ALL: composed system -> fixed norm -> transform -> fixed window arrays | file added:1 |
| `model_augmentation/utils/deepSI_corrections.py:30` | `fixed_System_data_norm` | required | ALL: composed system -> fixed norm -> transform -> fixed window arrays | file added:1 |
| `model_augmentation/utils/deepSI_corrections.py:44` | `fixed_System_data_norm.__init__` | required | ALL: composed system -> fixed norm -> transform -> fixed window arrays | file added:1 |
| `model_augmentation/utils/deepSI_corrections.py:51` | `fixed_System_data_norm.is_id` | required | ALL: composed system -> fixed norm -> transform -> fixed window arrays | file added:1 |
| `model_augmentation/utils/deepSI_corrections.py:55` | `fixed_System_data_norm.make_training_data` | not needed | Only norm.fit uses this; auto_fit_norm=False. | file added:1 |
| `model_augmentation/utils/deepSI_corrections.py:61` | `fixed_System_data_norm.fit` | not needed | auto_fit_norm=False; scales installed manually. | file added:1 |
| `model_augmentation/utils/deepSI_corrections.py:69` | `fixed_System_data_norm.transform` | required | ALL: composed system -> fixed norm -> transform -> fixed window arrays | file added:1 |
| `model_augmentation/utils/deepSI_corrections.py:87` | `fixed_System_data_norm.inverse_transform` | not needed | No open-loop apply_experiment in required path. | file added:1 |
| `model_augmentation/utils/deepSI_corrections.py:101` | `fixed_System_data_norm.u_transform` | not needed | Not called by required rollout; it normalizes arrays explicitly. | file added:1 |
| `model_augmentation/utils/deepSI_corrections.py:103` | `fixed_System_data_norm.y_transform` | not needed | Not called by required rollout. | file added:1 |
| `model_augmentation/utils/deepSI_corrections.py:105` | `fixed_System_data_norm.u_inv_transform` | not needed | Not called by required rollout. | file added:1 |
| `model_augmentation/utils/deepSI_corrections.py:107` | `fixed_System_data_norm.y_inv_transform` | not needed | Not called by required rollout. | file added:1 |
| `model_augmentation/utils/deepSI_corrections.py:110` | `fixed_System_data_norm.__repr__` | not needed | Not used for method computation. | file added:1 |
| `model_augmentation/utils/deepSI_corrections.py:114` | `convert_to_same_type` | not needed | No successful spec call path to this definition. | file added:1 |

#### `model_augmentation/utils/torch_nets.py`

| Source | Symbol | Classification | Spec and call path / exclusion | Project marker |
| :- | :- | :- | :- | :- |
| `model_augmentation/utils/torch_nets.py:9` | `linear_encoder_net` | not needed | Alternative network/encoder class never constructed: mlp/live data_lsq path. | @added on linear_encoder_net |
| `model_augmentation/utils/torch_nets.py:10` | `linear_encoder_net.__init__` | not needed | Alternative network/encoder class never constructed: mlp/live data_lsq path. | @added on linear_encoder_net |
| `model_augmentation/utils/torch_nets.py:18` | `linear_encoder_net.forward` | not needed | Alternative network/encoder class never constructed: mlp/live data_lsq path. | @added on linear_encoder_net |
| `model_augmentation/utils/torch_nets.py:22` | `feed_forward_nn` | not needed | Alternative network/encoder class never constructed: mlp/live data_lsq path. | original/unmarked |
| `model_augmentation/utils/torch_nets.py:23` | `feed_forward_nn.__init__` | not needed | Alternative network/encoder class never constructed: mlp/live data_lsq path. | original/unmarked |
| `model_augmentation/utils/torch_nets.py:37` | `feed_forward_nn.forward` | not needed | Alternative network/encoder class never constructed: mlp/live data_lsq path. | original/unmarked |
| `model_augmentation/utils/torch_nets.py:40` | `identity_init_simple_res_net` | not needed | Alternative network/encoder class never constructed: mlp/live data_lsq path. | original/unmarked |
| `model_augmentation/utils/torch_nets.py:41` | `identity_init_simple_res_net.__init__` | not needed | Alternative network/encoder class never constructed: mlp/live data_lsq path. | original/unmarked |
| `model_augmentation/utils/torch_nets.py:58` | `identity_init_simple_res_net.forward` | not needed | Alternative network/encoder class never constructed: mlp/live data_lsq path. | original/unmarked |
| `model_augmentation/utils/torch_nets.py:64` | `linear_mapping` | not needed | Alternative network/encoder class never constructed: mlp/live data_lsq path. | original/unmarked |
| `model_augmentation/utils/torch_nets.py:65` | `linear_mapping.__init__` | not needed | Alternative network/encoder class never constructed: mlp/live data_lsq path. | original/unmarked |
| `model_augmentation/utils/torch_nets.py:78` | `linear_mapping.forward` | not needed | Alternative network/encoder class never constructed: mlp/live data_lsq path. | original/unmarked |
| `model_augmentation/utils/torch_nets.py:81` | `zero_init_linear_mapping` | not needed | Alternative network/encoder class never constructed: mlp/live data_lsq path. | original/unmarked |
| `model_augmentation/utils/torch_nets.py:82` | `zero_init_linear_mapping.__init__` | not needed | Alternative network/encoder class never constructed: mlp/live data_lsq path. | original/unmarked |
| `model_augmentation/utils/torch_nets.py:93` | `zero_init_linear_mapping.forward` | not needed | Alternative network/encoder class never constructed: mlp/live data_lsq path. | original/unmarked |
| `model_augmentation/utils/torch_nets.py:96` | `zero_init_feed_forward_nn` | not needed | Alternative network/encoder class never constructed: mlp/live data_lsq path. | original/unmarked |
| `model_augmentation/utils/torch_nets.py:97` | `zero_init_feed_forward_nn.__init__` | not needed | Alternative network/encoder class never constructed: mlp/live data_lsq path. | original/unmarked |
| `model_augmentation/utils/torch_nets.py:116` | `zero_init_feed_forward_nn.forward` | not needed | Alternative network/encoder class never constructed: mlp/live data_lsq path. | original/unmarked |
| `model_augmentation/utils/torch_nets.py:121` | `paper_resnet` | not needed | Alternative network/encoder class never constructed: mlp/live data_lsq path. | @added on paper_resnet |
| `model_augmentation/utils/torch_nets.py:124` | `paper_resnet.__init__` | not needed | Alternative network/encoder class never constructed: mlp/live data_lsq path. | @added on paper_resnet |
| `model_augmentation/utils/torch_nets.py:165` | `paper_resnet.forward` | not needed | Alternative network/encoder class never constructed: mlp/live data_lsq path. | @added on paper_resnet |
| `model_augmentation/utils/torch_nets.py:170` | `live_g_feed_forward_nn` | required | ALL: build_model -> Static_ANN_Block -> live_g_feed_forward_nn | @added on live_g_feed_forward_nn |
| `model_augmentation/utils/torch_nets.py:178` | `live_g_feed_forward_nn.__init__` | required | ALL: build_model -> Static_ANN_Block -> live_g_feed_forward_nn | @added on live_g_feed_forward_nn |
| `model_augmentation/utils/torch_nets.py:203` | `live_g_feed_forward_nn.forward` | required | ALL: build_model -> Static_ANN_Block -> live_g_feed_forward_nn | @added on live_g_feed_forward_nn |
| `model_augmentation/utils/torch_nets.py:207` | `zero_init_resnet` | not needed | Alternative network/encoder class never constructed: mlp/live data_lsq path. | @added on zero_init_resnet |
| `model_augmentation/utils/torch_nets.py:208` | `zero_init_resnet.__init__` | not needed | Alternative network/encoder class never constructed: mlp/live data_lsq path. | @added on zero_init_resnet |
| `model_augmentation/utils/torch_nets.py:231` | `zero_init_resnet.forward` | not needed | Alternative network/encoder class never constructed: mlp/live data_lsq path. | @added on zero_init_resnet |
| `model_augmentation/utils/torch_nets.py:235` | `unit_variance_feed_forward_nn` | not needed | Alternative network/encoder class never constructed: mlp/live data_lsq path. | original/unmarked |
| `model_augmentation/utils/torch_nets.py:236` | `unit_variance_feed_forward_nn.__init__` | not needed | Alternative network/encoder class never constructed: mlp/live data_lsq path. | original/unmarked |
| `model_augmentation/utils/torch_nets.py:263` | `unit_variance_feed_forward_nn.forward` | not needed | Alternative network/encoder class never constructed: mlp/live data_lsq path. | original/unmarked |
| `model_augmentation/utils/torch_nets.py:267` | `HybridGantryEncoder` | not needed | Alternative network/encoder class never constructed: mlp/live data_lsq path. | @added on HybridGantryEncoder |
| `model_augmentation/utils/torch_nets.py:274` | `HybridGantryEncoder.__init__` | not needed | Alternative network/encoder class never constructed: mlp/live data_lsq path. | @added on HybridGantryEncoder |
| `model_augmentation/utils/torch_nets.py:303` | `HybridGantryEncoder.forward` | not needed | Alternative network/encoder class never constructed: mlp/live data_lsq path. | @added on HybridGantryEncoder |
| `model_augmentation/utils/torch_nets.py:338` | `LinearInitEncoderWrapper` | not needed | Alternative network/encoder class never constructed: mlp/live data_lsq path. | @added on LinearInitEncoderWrapper |
| `model_augmentation/utils/torch_nets.py:358` | `LinearInitEncoderWrapper.__init__` | not needed | Alternative network/encoder class never constructed: mlp/live data_lsq path. | @added on LinearInitEncoderWrapper |
| `model_augmentation/utils/torch_nets.py:407` | `LinearInitEncoderWrapper.forward` | not needed | Alternative network/encoder class never constructed: mlp/live data_lsq path. | @added on LinearInitEncoderWrapper |
| `model_augmentation/utils/torch_nets.py:434` | `positive_default_encoder_net` | not needed | Alternative network/encoder class never constructed: mlp/live data_lsq path. | original/unmarked |
| `model_augmentation/utils/torch_nets.py:435` | `positive_default_encoder_net.__init__` | not needed | Alternative network/encoder class never constructed: mlp/live data_lsq path. | original/unmarked |
| `model_augmentation/utils/torch_nets.py:444` | `positive_default_encoder_net.forward` | not needed | Alternative network/encoder class never constructed: mlp/live data_lsq path. | original/unmarked |

#### `model_augmentation/utils/utils.py`

| Source | Symbol | Classification | Spec and call path / exclusion | Project marker |
| :- | :- | :- | :- | :- |
| `model_augmentation/utils/utils.py:7` | `added` | required | ALL: model/interconnect/block construction | original/unmarked |
| `model_augmentation/utils/utils.py:15` | `to_tensor` | required | ALL: model/interconnect/block construction | @added on to_tensor |
| `model_augmentation/utils/utils.py:24` | `to_ndarray` | not needed | Utility unused by chosen data_lsq and closed-loop paths. | original/unmarked |
| `model_augmentation/utils/utils.py:31` | `detect_algebraic_loop` | required | ALL: model/interconnect/block construction | original/unmarked |
| `model_augmentation/utils/utils.py:43` | `plot_fft_freq` | not needed | Utility unused by chosen data_lsq and closed-loop paths. | original/unmarked |
| `model_augmentation/utils/utils.py:65` | `plot_sys_data` | not needed | Utility unused by chosen data_lsq and closed-loop paths. | original/unmarked |
| `model_augmentation/utils/utils.py:98` | `get_sys_data_dimensions` | not needed | Utility unused by chosen data_lsq and closed-loop paths. | original/unmarked |
| `model_augmentation/utils/utils.py:105` | `determine_sys_data_signal_dimension` | not needed | Utility unused by chosen data_lsq and closed-loop paths. | original/unmarked |
| `model_augmentation/utils/utils.py:112` | `determine_std_T` | not needed | Utility unused by chosen data_lsq and closed-loop paths. | original/unmarked |
| `model_augmentation/utils/utils.py:122` | `determine_std_T_sys_data` | not needed | Utility unused by chosen data_lsq and closed-loop paths. | original/unmarked |
| `model_augmentation/utils/utils.py:137` | `normalize_linear_ss_matrices` | not needed | Utility unused by chosen data_lsq and closed-loop paths. | original/unmarked |
| `model_augmentation/utils/utils.py:147` | `determine_yes_no_query_binary_output` | not needed | Utility unused by chosen data_lsq and closed-loop paths. | original/unmarked |
| `model_augmentation/utils/utils.py:157` | `expansion_matrix` | required | ALL: model/interconnect/block construction | original/unmarked |
| `model_augmentation/utils/utils.py:162` | `selection_matrix` | required | ALL: model/interconnect/block construction | original/unmarked |
| `model_augmentation/utils/utils.py:168` | `compute_noisy_y_from_SNR` | not needed | Utility unused by chosen data_lsq and closed-loop paths. | @added on compute_noisy_y_from_SNR |

Definition coverage: 42 Python source files, 647 definitions (classes, functions and methods, including nested ones). Classifications: not needed: 308, not needed in method; required interface: 2, not needed, diagnostic: 58, required: 272, required interface: 7. This count is based on source syntax, not runtime coverage.

### A5. Required module-level data and external dependencies

- `gantry_dynamic/config.py:25`: REPO_ROOT and path semantics; AA_ORDER=8, AA_FC_HZ=1600, AA_TRIM_S=.005 and mode sets. `data.py:61`: exactly the lowpass and noise-free named dataset entries and split map, manifest order. `model.py:31`: COMPILE_BACKEND=inductor and physical indices0..5. `model.py:412`: least-squares readiness threshold1e-3.
- `controller.py:56`: nominal Python double constants, P/C_DAMP/K_STIFF, FBW=100, plus mutable stored-controller dictionaries `:193`. Training controller is independent of the float32-derived nominal physics constants in gantry_ss.
- `gantry_ss.py:32`: `_D=torch.float32`, nominal raw tensors, Lb, P, Cd and Dd, M/C/K and pure build functions. Import-time KX/KY overrides are outside spec. Do not replace nominal tensors with full-precision decimal doubles and claim exact equivalence.
- `blocks.py:1005` PARAM_NAMES raw order; `:1296` COMBO_NAMES reduced order; indices, M_DIFF_RHO=.99, COORDINATE_VERSION=2. Base-class nominal buffers and the gauge section still execute even if a more compact mathematical representation could avoid them.
- `Interconnect` metadata and connection lists/matrices, Block IDs, output dependency order and output-only cone; `Static_ANN_Block.out_gate` nonpersistent buffer of ones; `SSE_Interconnect_Composed` class defaults None for penalties/OBC/phase; parameter-group and serialization seams.
- Package initializers currently import utility/framework modules eagerly. `model_augmentation/__init__.py` imports utils and fit_systems; fit_systems imports interconnect, blocks, white_box_models, utils; systems imports mass_spring_damper. These make definitions importable and can demand dependencies, but they do not run benchmark constructors or RNG draws. `gantry_dynamic/__init__.py:13` inserts REPO_ROOT into sys.path.
- External runtime boundary: Python, NumPy, SciPy MAT I/O, scipy.signal butter/sosfiltfilt/cont2discrete/tf2ss, Torch nn/autograd/func/linalg/optim/compile, deepSI System_data/System_data_list, SS_encoder_general and System_torch superclass services, My_Simple_DataLoader iteration and loop_dataset. tqdm and deepSI timing/size helpers support existing fit logging; matplotlib is reporting/import overhead rather than a mathematical dependency.
- The active shell's Python has no installed NumPy or deepSI (`find_spec` returned None). No source-backed claim is made here about its inherited implementations beyond the repository calls and the fixed loader override. Exact installed deepSI/PyTorch source/version is a later equivalence input. No training import was attempted. `environment.server-train.yml` is a declared environment, not proof of the environment of a particular checkpoint.
- No Python code under common is mathematically necessary for training. oracle.py is used by the excluded report only; orth_penalty.py requires orth=True; rezero_gate.py requires ANN_REZERO_GATE; feedback_curriculum.py is not called. Evaluation interfaces currently import helpers from an excluded diagnostic module; this is a compatibility dependency, not grounds to include its complete reporting implementation.

## Part B. Thesis-to-code comparison

S2, S3, S4, S5 mean `Thesis-writeup/Writing/sections/02_system_baseline.tex`, `03_augmentation.tex`, `04_obc.tex`, `05_setup.tex`. App means `Thesis-writeup/Writing/sections/99_appendix.tex`. GD, FS, U mean `scripts/gantry/gantry_dynamic/`, `model_augmentation/fit_systems/`, `model_augmentation/utils/`. EV, Data, Setup mean `Thesis-writeup/Code/Evaluation/`, `Thesis-writeup/Code/Data/`, `Thesis-writeup/Code/Setup/`. Equation labels identify numbered equations independently of TeX build numbering. Rows also cover grouped method paragraphs. Derivations and empirical motivation need not be executable algorithms; these are explicitly distinguished from implemented procedures. Numerical result claims were not reproduced by this static audit.

### B1. Section II

| Thesis source and statement/equation | Code source | Verdict and qualification |
| :- | :- | :- |
| S2:96 to :144 hardware, force/position coordinates, known-controller problem; `eq:system` :129 | GD/data.py:203; GD/controller.py:269; Data/gantrySystemExtendedTanh.m:27 | Match data interface u_total,y,stored controller. Unknown continuous f is the identification problem, not a Python training truth model. Hardware/confidentiality statements are in thesis but not code. |
| S2:160 and :200 small-angle and negligible Coriolis assumptions; `eq:eom` :207 | FS/blocks.py:814; gantry_ss.py:184 | Match M(Y), linear C/K, no Coriolis/centripetal terms. Rail friction and absorber absent from baseline, as S2:244 todo requests. |
| `eq:Ptransform` S2:175 and coordinate definitions :164 | gantry_ss.py:120; GD/data.py:339; FS/blocks.py:832; GD/obc_gantry.py:131 | Match algebraically. P is created in float32, while controller.py:65 uses Python double decimal constants. Precision difference is unstated in thesis. |
| `eq:mass_matrix` S2:214 | gantry_ss.py:70; FS/blocks.py:1079, :1350, :1365 | Match via reduced combinations and gauge reconstruction; closed rational inverse rather than per-step dense solve. |
| `eq:damping_stiffness_matrices` S2:224 | FS/blocks.py:1042; gantry_ss.py:89 | Match kb/cb sums. Artificial KX/KY environment overrides outside spec. |
| S2:246 to :262 meanings, nominal values and fixed Lb | gantry_ss.py:37 to :56; FS/blocks.py:1036 | Match nominal design values and fixed coordinate frame, with float32 rounding caveat. Lb is not trained. |
| S2:267 to :291 self-scheduled LPV Y, no assumption on dotY | FS/blocks.py:847, :791 | Match use of model state Y at every RK stage. S2:268 calls Y measured: a literal measured-Y rollout would differ, since simulation uses predicted Y; measurements are used in reference tuples. |
| S2:269 empirical3.7 LPV/LTI error factor | EV/eval_greybox.py:72, :140; App:253 | In thesis but not core code as an asserted numeric result. Evaluation supplies LTI intervention; factor not revalidated. |
| `eq:Msplit` S2:312 | gantry_ss.py:70 to :84; FS/blocks.py:1064 | Match. |
| `eq:lfr_mass_factorization` S2:318; `eq:lfr_chain` :325 | FS/blocks.py:836 to :862 | Match acceleration solve and z=[a,Y*a]. |
| `eq:lfr_constant` S2:334; `eq:delta` :343 | FS/blocks.py:861 to :867 | Match w=Y*z and explicit LFR state-row multiplication. |
| `eq:lfr_G` S2:354; closure/wellposedness :363 to :372 | gantry_ss.py:184; FS/blocks.py:814, :1494 | Match state row and algebraic solution. Full15x15 G and proof are in thesis but not instantiated as a single runtime matrix; Ax/Bw/Bu only. Construction samples Y for an admissibility check; the coordinate map encodes the global analytic condition. |
| S2:385 to :430 ten independent functions, determinant mh*Lb^2/2, four lost directions | FS/blocks.py:1350, :1365, :1296 | Match representation; structural identifiability proof is in thesis but not an executed estimator or runtime proof. |
| `eq:phi` S2:433 and :438 definitions | FS/blocks.py:1296, :1350 | Match order kb_sum,cg1,cg2,cy,cb_sum,mh,m_total,m_diff,J_eff,d. Raw14 names are not trained coordinates. |
| `eq:lfr_admissibility` S2:461; log/tanh choice :466 | FS/blocks.py:1385, :1397, :1337 | Match analytic condition with rho=.99. Finite precision/overflow can still invalidate numerical values; code does not guarantee finite outputs for arbitrary enormous free vectors. |

### B2. Section III

| Thesis source and statement/equation | Code source | Verdict and qualification |
| :- | :- | :- |
| S3:210 to :228 overview, :262 missing dynamics and :276 parallel choice | GD/model.py:71 to :151, :179 to :201 | Match six physical plus n_a latent states, one discrete parallel ANN. Motivation is thesis reasoning, not an executable necessity test. |
| `eq:aug_transition` S3:286 and :297 to :318 output/latent meaning | GD/model.py:179 to :186; FS/interconnect.py:143 to :170; FS/blocks.py:138 | Match U model; OBC separately replaces physical correction. g_aug gives next x_a, not an increment; output map is fixed physical positions only. |
| `eq:aug_ct` S3:333; RK4/ZOH/intermediate Y :331 to :346 | FS/blocks.py:791, :814; GD/model.py:88 | Match one step per Ts=.00025, held input, Y at each RK stage. Explicit LFR state row retained after rational solve. |
| S3:348 wellposed acyclic interconnection draft todo | FS/interconnect.py:199; U/utils.py:31; FS/blocks.py:1397 | Match implementation conditions. The differentiability/wellposedness theorem is thesis-only analysis. |
| S3:356 to :365 normalization, velocity finite differences; `eq:aug_norm` :366 | GD/data.py:329 to :378; FS/blocks.py:824 | Match intent and training-only statistics. Unmentioned +1e-8 scale stabilization, repeated first velocity difference and float32 materialization remain. |
| `eq:aug_norm_maps` S3:385 and :398 to :410 SI inversion/output offset | FS/blocks.py:824 to :870; GD/data.py:378; GD/model.py:132 | Match intended affine conjugacy. Numeric caveat: state/input scales and C pass through default-precision to_tensor before casting to float64. Separately computed means align only to rounding; no explicit output bias enforces exact identity. |
| S3:376 added states not separately normalized | FS/pre_encoder.py:438; GD/model.py:143 | Match no latent mean/std. |
| S3:412 to :432 shared tanh MLP, full inputs/all-six routing | U/torch_nets.py:170; FS/blocks.py:31; GD/model.py:136, :179 | Match two hidden16 layers, no bypass, physical position corrections included. |
| S3:434 to :444 integrator drift motivates feedback | FS/blocks.py:1042; FS/closed_loop.py:216 | Match stiffness-free X/Y and chosen loop. Stability/drift rationale is thesis analysis; no drift regularizer/test runs in final method. |
| S3:475 to :505 closed-loop/noise motivation and consistency limitation | GD/controller.py:269; FS/closed_loop.py:216 | Match chosen simulation; no consistency proof or identified stochastic noise model in code. |
| `eq:aug_two_loops` S3:518, cancellation :535 to :544 | GD/controller.py:269; FS/closed_loop.py:167, :216 | Match residual realization. Two absolute controllers/reference/feedforward cancellation are not separately run. Training-rate versus generator-rate controller means equivalence to recorded absolute loop is approximate, as thesis acknowledges. |
| `eq:aug_residual_loop` S3:553 and zero initial xc :546 | FS/closed_loop.py:284, :235 to :241 | Match residual controller reset per window and carried through record validation; no estimated controller state. |
| S3:572 output-before-input and no feedthrough | FS/closed_loop.py:235; FS/interconnect.py:303; FS/blocks.py:138 | Match D=0; output-only cone avoids ANN/RK4 evaluation to compute output. |
| `eq:aug_cl_step` S3:588 | FS/closed_loop.py:139 to :145, :167, :216 | Match scaling folded into controller matrices, controller state physical. Fused baddbmm is current operation order, not separate C/D/A/B products. |
| S3:601 to :609 reset window versus once-per-record validation | FS/interconnect.py:1384; FS/closed_loop.py:787 to :797 | Match encoder at k0=30 then complete remainder free run. |
| S3:637 joint estimation of physics/ANN/encoder | GD/model.py:194, :293; FS/interconnect.py:664; FS/lbfgs_polish.py:134 | Match inherited optimizer parameter groups include trainable encoder and hfn. |
| S3:643 exponential coordinates; `eq:mdelta_map` :647 and :652 to :664 | FS/blocks.py:1337, :1385, :1397 | Match eta0,local scale,rho=.99,beta0=0. Gauge reconstruction into raw14 is an additional implementation step unstated in main thesis. |
| `eq:aug_loss` S3:673, weights :690 to :695 | GD/controller.py:347; FS/closed_loop.py:377; FS/interconnect.py:1383 to :1436 | Match weighted MSE, all400 outputs including first, weights sigma_y/sigma_e. param_loss called but adds0. Actual minibatches drop remainder; full-sum equation is objective, not every update. |
| S3:701 to :716 encoder/scalability/full gradients | FS/interconnect.py:1384, :1023; FS/closed_loop.py:291 | Match full BPTT, no detachment or truncation. Smoothness rationale is thesis analysis. |
| `eq:aug_encoder` S3:717, :726 to :733 | GD/model.py:47 to :51; FS/pre_encoder.py:488; U/deepSI_corrections.py:18 | Match history[tau-30,tau),future[tau,tau+400), no current y_tau used by encoder. |
| `eq:aug_enc_lin` S3:739, parameter roles :745 to :754 | FS/pre_encoder.py:503 to :523 | Match architecture with column permutation: linear Wb/Wa use u/y blocks; nonlinear net receives [u_flat,y_flat], while displayed chi is[y_flat,u_flat]. Exact function reparameterization, but saved weights require actual ordering. No D-055 offsets in data_lsq. |
| S3:759 to :770 nonunique split/first-order qualifier todo | FS/obc.py:242; FS/interconnect.py:1400 | U enforces no uniqueness. OBC protects local reference span, not a proven unique closed-loop optimum. Thesis-only analysis/todo, not an extra optimizer constraint. |
| S3:773 to :786 Adam/fixed-batch L-BFGS, min validation/strict acceptance | FS/interconnect.py:808, :902; FS/lbfgs_polish.py:258, :650, :778; GD/training.py:786 | Match with finite-loss guard, coarsened/sampled fixed batch and last-on-tie Adam checkpoint; OBC comparison re-scored after rebuild. No oracle meter veto. |
| S3:807 to :820 nonlinear-baseline data-based route | GD/model.py:246, :328 | Match current linear restriction described next. Older nonlinear Adam pretraining in fix docs is superseded. |
| `eq:aug_enc_sim` S3:826, initial states :832 to :841 | GD/model.py:352 to :407 | Match theta0 baseline, measured first positions mapped to q,zero velocity/added/controller states, simulated u_cl/y_hat histories and pre-step states. Physical ANN output zero during simulation. |
| `eq:aug_enc_fit` S3:848, :854 to :878 linear restriction/intercept/SVD | GD/model.py:415 to :464 | Match standardized centred affine LS,CPU double gelsd,explicit rank cutoff,physical output weights zero and intercept bias; hidden layers/Wa retained. Code additionally requires R2>=.999,map reproduction,finite values. |
| `eq:aug_init` S3:889, distributions/equality :899 to :920 | U/torch_nets.py:178 to :201; FS/pre_encoder.py:375 to :459; GD/model.py:443 | Match beta0=0,physical final rows zero,added rows fan-in live,final biases zero,Wa joint Xavier(fan_in180,fan_out=n_a),physical encoder LS. Hidden biases default uniform. Wb/output draws later overwritten still consume RNG. |
| S3:935 to :974 PS motivation/two-stage regression,V continues S1 | FS/ps2_opening.py:101; FS/interconnect.py:1045 | Match auxiliary update after each main update. PS flag means attempted opening, not guaranteed S2 completion. |
| S3:995 to :1013 calibration/history/future/H=n+1 | FS/ps2_opening.py:59, :87, :143, :161 | Match H30,history ends tau-1,last class member held out. Sampling is length-weighted with replacement and leaves extra tail indices unused. |
| `eq:ps2_s1` S3:1014, detached target/decoder/parameters :1037 to :1054 | FS/ps2_opening.py:143 to :157, :183 to :205 | Match e=(yf-ypred) flattened/batch RMS, no servo sigma_e weighting. All encoder final rows passed to auxiliary optimizer,physical rows receive zero auxiliary gradient; head output initially zero. |
| S3:1056 conditional-mean rationale; :1065 to :1077 no reuse/no independence | FS/ps2_opening.py:174, :183 | Match disjoint indices and detached targets; conditional-expectation rationale is not a finite-network guarantee. No whitening correction. |
| `eq:ps2_nu` S3:1083 and :1079 to :1091 fixed calibration/fallback | FS/ps2_opening.py:110 to :128, :207 | Match ratio of same-tensor means equals sum ratio; residual recalculated at current model and RMS scale. Fallback retains encoder changes; threshold<=.90 permits S2. |
| S3:1095 to :1115 frozen pairs,recorded input,V paused S2 | FS/ps2_opening.py:213 to :235 | Match snapshot under no_grad,main loop blocked during fit,ANN-only optimizer,recorded u_k. |
| `eq:ps2_s2` S3:1116, :1127 to :1132 variance | FS/ps2_opening.py:226 to :227 | Mismatch normalization: thesis uses sum squared deviations with population normalization; code T.var(0).mean has default correction1. For B windows,code objective is(B-1)/B times displayed fraction:511/512 on batch512,2047/2048 on cal2048. State map/targets match. |
| S3:1134 lowest calibration-cost S2 parameters kept | FS/ps2_opening.py:233 to :251 | Match if a checked point beats initial r0. Mismatch otherwise: state=None,no initial snapshot,so no improving check leaves final S2 iterate in place while reporting best=r0. |
| S3:1136 to :1150 recorded/closed-loop input relation | FS/ps2_opening.py:223; FS/closed_loop.py:237 | Match conditional algebra; code does not assert encoder reproduces y at start. |
| S3:1151 to :1157 physical final rows unchanged/shared hidden effect | FS/ps2_opening.py:216 to :238; U/torch_nets.py:186 | Match added-output-only loss zeroes physical final-row gradient; shared hidden changes can still affect f_aug. |
| `eq:ps2_handover` S3:1160, :1158 to :1174 decoder discarded/same model/loss | FS/ps2_opening.py:124 to :128, :248 to :255; GD/training.py:892 | Match no main Adam reinitialization; S2 clears ANN gradients; main moments retained across auxiliary updates. Fallback retains S1 result. |

### B3. Section IV

| Thesis source and statement/equation | Code source | Verdict and qualification |
| :- | :- | :- |
| S4:83 to :100 uniqueness to first order and nonlinear state-space extension | FS/obc.py:242; GD/obc_gantry.py:319 | Match construction intent; no global identifiability/true-recovery guarantee implemented. Local to fixed one-step stack. |
| `eq:obc_gyorok` S4:119 and :135 to :146 full rank | FS/obc.py:191, :242 | Match algebraic LS projection,computed with SVD rather than inverse normal equations. Cited input-output predecessor is theory,not another runtime model. |
| `eq:obc_sens` S4:169 and :157 to :176 beta derivative | FS/obc.py:341; GD/obc_gantry.py:330; FS/blocks.py:1429 | Match jacfwd of normalized discrete transition with respect to free beta at fixed tuples,float64. No gradient through tuple construction. |
| `eq:obc_offset` S4:179 and :177 to :200 affine inclusion | GD/obc_gantry.py:332 to :338, :305 to :307 | Equivalent span,different coefficients: thesis gamma=fbar-J*vbar,code appends fbar and subtracts J*a+alpha*fbar. Gamma coefficients would be(a+alpha*vbar,alpha). Do not relabel saved theta_frozen without conversion. |
| `eq:obc_tuples` S4:216 and :223 to :230 successor/alignment | GD/obc_gantry.py:79, :106 to :162 | Match measured positions,fourth-order velocities,k=2..N-4 inclusive,zero added state,all training records,no encoder. Simulator truth fields read only for diagnostics. |
| `eq:obc_coef` S4:234 and :248 to :256 stacked slice guarantee | FS/obc.py:191 to :248, :512; GD/obc_gantry.py:344 | Match projector mathematically,with above affine coordinates and column-scaled SVD. Expected rank11; bad refresh can retain old valid basis once,an unstated failure behavior. Orthogonality over stack at x_a=0,not every individual tuple or simulated trajectory. |
| `eq:obc_corr` S4:272; physical-only :258 to :269,differentiation :289 | FS/interconnect.py:1406 to :1416, :168 to :170; GD/obc_gantry.py:250, :286 | Match coefficient gradient through ANN reference field,basis/vbar detached,current state and corrected input; g_aug unchanged. Hand-coded RK4 tangent computes J*a,alpha*fbar gives equivalent affine direction. |
| S4:302 to :318 reuse,PS,coefficient per objective,epoch refresh | FS/interconnect.py:1406; FS/obc.py:501 | Match OBC+PS rows221/222;229 deliberately no PS/n_a0. Epoch=floor(counter+1e-6),not validation cadence. First basis can be built during prefit; epoch0 reuses it. |
| S4:319 to :322 restore,rebuild,fixed-basis polish,lower validation | FS/interconnect.py:1139 to :1149; GD/training.py:786; FS/lbfgs_polish.py:171, :603, :778 | Match re-scoring after rebuild,coefficient changes during polish,basis fixed,rollback re-sync. Selected and final scores can differ. |
| `eq:obc_eps` S4:328; reconstructed successor/noise/discrepancy | GD/obc_gantry.py:140; EV/eval_overlap.py:109 to :125; GD/obc_diagnostics.py:215 | Match final reconstructed successor. Diagnostic 'data' still means simulator successor;'reconstructed' matches equation. Neither is training target. |
| `eq:obc_recovery` S4:342; :337 to :364 true iff Phi^T Xi=0 and limitation | EV/eval_overlap.py:128 to :129, :93 | Match first-order diagnostic beta shift,converted by combination Jacobian. Training does not solve this surrogate; it optimizes closed-loop V. No separate unconstrained one-step estimator runs in training. |
| `eq:obc_overlap` S4:368; cosine/rho/Pi/c_neg :366 to :392 | FS/obc.py:262, :280; EV/eval_overlap.py:41; EV/eval_negation.py:71 to :89 | Match projection/cosine formulas. Mismatch operating point for OBC c_neg: thesis baseline increment uses current beta,evaluator uses v_hat=corr.vbar. U uses current free_params. Evaluation measures corrected deployed F; displayed F_aug notation could mean raw field. |

### B4. Section V

| Thesis source and statement/equation | Code source | Verdict and qualification |
| :- | :- | :- |
| S5:45 to :52 simulated data/industrial limits; :67 to :78 omissions | GD/data.py:61, :203; Data/tdg_config.m:23, :82 | Match external-data arrangement. Generator/confidentiality are not algorithms training folder needs. No prohibited real records read. |
| `eq:setup_truth` parent S5:89; `eq:setup_fric` :90 | Data/gantrySystemExtendedTanh.m:69 to :80; tdg_config.m:28 to :37; common/oracle.py:178 | Match tanh rail friction,gain1000,v_fr=.001m/s,stage velocities P.T*qdot. Not a trained-baseline term. Parent subequations groups friction/absorber. |
| S5:104 smoothing sweep/regime | Data/tdg_config.m:37; Setup/rail_velocity.py:23 | Value1mm/s matches. Sweep/adequacy are external evidence,not executed final method or statically verified numerical conclusion. |
| `eq:setup_abs` S5:113 and unnumbered M_e/M_a/C_e/K_e :122 to :137 | Data/gantrySystemExtendedTanh.m:34 to :64; common/oracle.py:160 | Match expansion with mh_rigid=mh-ma,ma=.5mh,L0=.1,fa150,zeta.03 (tdg_config.m:41 to :50). Legacy oracle.py:36 constants MA_FRAC=.10,ZETA_A=.05 are unused old oracle,not current meta-driven truth. |
| S5:138 to :149 generatorRK4/20k/parameters/resonance/crossover | Data/tdg_config.m:45 to :64; gtd_run_simulation.m:1; Setup/band_frf.py:41,loop_memory.py:26 | Match declared settings,distinct from trainingRK4/4k. Resonance/crossover adequacy numerical evidence not re-run. |
| S5:150 to :164 K1,force feedforward,colored noise,twins,SNR | Data/tdg_config.m:69, :124 to :139; tdg_noise_draw.m:16 to :23; GD/data.py:203 | Match stored data. cfg.snr=None means no new training noise. Noisy-minus-noisefree output includes changed feedback response,not only raw encoder noise. |
| S5:181 to :197 five motion types,12s,multisine,P^-1,band rationale | Data/tdg_config.m:73 to :101; gtd_make_multisine.m:62 to :100; manifests | Match declared configuration/imported lists. SNR/band adequacy are external design evidence,not required training algorithms. |
| S5:199, 18/6/11 split and :203 to :213 held-out settings/selection | GD/data.py:69, :79; manifests; GD/controller.py:232; closed_loop_report.py:95; EV/eval_greybox.py:123 | Qualified match: loader18/6/12; E5,K2-g+ excluded under controller_gain=False,leaving11 scored tests. Thesis's 'every record K1' omits this distinction for complete loaded manifest. |
| S5:201 seven multisine twins | EV/eval_common.py:142; eval_greybox.py:89; GD/data.py:203 folder override | Match evaluation-only manifest,never in train/normalization/reference tuples/validation selection. |
| S5:216, 4kHz/0.1s | entry.py:345; GD/config.py:542 to :573 | Match Ts.25ms,nf400. |
| S5:231 to :269 compared models,LPV truth,detuning,PS,shared setup | runs.tsv:88 to :96; entry.py:334 to :365; closed_loop_report.py:38, :67 | Match main R1/R4 and appendix ablations. Truth LPV is reporting intervention,not trained. Shared seed1 signs. Row229 is OBC/n_a0/noPS,223 plain U. |
| S5:274 to :281 black-box search | EV/eval_blackbox.py; Server/submit.sh:78 final-bb | In thesis but not this pipeline by scope; shared loader/metrics only. No R8 implementation inventoried. |
| S5:302 full records and servo comparison; `eq:setup_rms` :308 | FS/closed_loop.py:719; GD/data.py:222; EV/eval_greybox.py:154 | Match RMS and population std on k0=30..N-1. Shared r cancels from prediction difference. Noise floor is external paired evaluation,not post-run cfg.snr noise level. |
| S5:319, 100% means equal performance to yhat=r | GD/data.py:231; FS/closed_loop.py:807 | Mismatch as exact identity: yhat=r gives RMS(e)/std(e)=sqrt(1+mean(e)^2/var(e)),100% only if mean(e)=0. Numerator not centred. May be negligible here but no code identity. |
| `eq:setup_delta` S5:324 and :334 m_Delta scale | GD/training.py:439, :746; EV/eval_overlap.py:57 | Match mean nominal X mass for m_diff,absolute truth for other combos. param_table/report_joint_estimation print true/start-relative percentages instead,not this metric. |
| S5:337 to :342 overlap train/val/testI1-I6,corrected field,Xi | EV/eval_overlap.py:97 to :136; eval_negation.py:66 | Match external evaluation,train slice x_a0. Training post-run report itself does not calculate every thesis metric. |

### B5. Referenced appendix parts

| Appendix source and statement/equation | Code source | Verdict and qualification |
| :- | :- | :- |
| App:19 to :48 nominal/addition parameter table | gantry_ss.py:37; Data/tdg_config.m:23 to :50 | Match design decimals,float32 materialization caveat. Current oracle reads truth metadata. |
| App:53 to :64 M1/M2 and constant mass equation | FS/blocks.py:1064; gantry_ss.py:184 | Match. |
| `eq:app_G_partition` App:65; `eq:app_lfr_blocks` :77 | gantry_ss.py:184 to :234; FS/blocks.py:861 | Match Ax/Bw/Bu state row. Cz/Dzw/Dzu/Cy/Dyw/Dyu and full partition proof are in thesis but not separately instantiated by runtime. Output block maps actuator frame. |
| App:96 to :123 loop determinant ratio,rational inverse | gantry_ss.py:150; FS/blocks.py:847 to :855 | Match inverse numerator/determinant and z,w closure. Schur/determinant proof not executed. |
| App:124 both polynomials evaluated by Horner | FS/blocks.py:848 to :855 | Mismatch literally: numerator n0f+Y*(n1f+Y*n2f),denominator expanded alpha*gamma-beta^2+2*beta*mh*Y+mh*(alpha-mh)*Y^2. Rewriting denominator Horner changes operation order. |
| App:127 to :153 positivity/completed square/raw positivity proof | FS/blocks.py:1385, :1397, :1494 | Match represented global condition; proof not evaluated every update. Construction additionally samples201Y in[-.30,.30]. |
| App:155 to :169 eta0,scale,inverse beta_Delta | FS/blocks.py:1337 to :1346, :1409 | Match nonzero start and contracted domain; no near-zero fallback. |
| App:172 to :184 inverse combinations from matrix entries | FS/blocks.py:1350, :1365 | Match forward combos/gauge; inverse-from-entry formulas are thesis-only derivation,not separately executed identification. |
| App:186 to :194 repeated Y/known selection/wellposedness draft | FS/blocks.py:847, :861 | Match state selector and repeatedY. Comparison with general sufficient theorem is thesis-only. |
| App:196 to :216 PS stopping equations | FS/ps2_opening.py:110 to :125 | Match threshold/cadence/cap/strict plateau for defaults. Absent previous checks represented as None; thesis infinity. Code blocks plateau until>4 checks; with min100 same stopping effect. Early screen150 unless an earlier plateau endsS1. |
| App:218 to :232 full FRF/positions | EV/eval_frf.py:35, :56, :76 | Match hfn interface,saved frozen OBC correction,C,norms,sevenY[-.39,-.30,-.15,0,.15,.30,.39]. Resonance/coincidence statements empirical,not revalidated. |
| App:235 to :244 all-split results | EV/eval_greybox.py:89; eval_common.py:31 | Match external consumers; training evaluation.py itself val/test only. |
| App:246 to :290 baseline/structure/PS/no-added-state recovery/grid comparisons | runs.tsv:88; EV/eval_greybox.py:140; eval_common.py:40 | Match specified ablations/interfaces. Table/figure production excluded from deliverable. |
| `eq:setup_K1` App:308 and :313 to :316 gain/Tustin | GD/controller.py:72 to :151 | Match bandwidth100,nominalY0,per-axis gain. Generator20k,training4k rediscretization. |
| `eq:setup_noise` App:319 and :323 to :329 spectrum/SNR | Data/tdg_config.m:124 to :130; tdg_noise_draw.m:16; Setup/snr.py:15 | Match external noise generation. Spectrum fitting/numerical evidence not re-run; no training RNG injection. |
| `eq:setup_ms` App:344 and :349 to :358 phases/crest | Data/gtd_make_multisine.m:62 to :100; tdg_config.m:94 | Match external synthesis,.5Hz106..297,30trials per component,RMS normalization,yaw response criterion. Not training dependency. |
| App:359 to :370 band/known truth/uncertainty | Setup/band_edge.py:37,band_frf.py:41 | In thesis but not final-method code; external design workflow,numeric macros not revalidated. |
| App:383 to :427 records/held-out limits/twin amplitudes | GD/data.py:79; EV/eval_common.py:142; Data/tdg_manifest.m | Match manifest ingestion,no runtime limit-based splitting beyond block labels. E5 qualification above applies. |
| `eq:setup_decim` App:450 and :455 to :462 processing | GD/data.py:152 to :200; GD/config.py:33 to :55 | Match lowpass servoSOS forward/backward/20sample trim/force mean. Noisefree extension intentionally point-samples y and still trims5ms; this equation does not describe it. 'Exact ZOH input' is stated averaging choice; code does not prove exact nonlinear response equivalence for all intra-hold force variation. |
| App:465 tanh.5%/checks table :479 to :490 | Setup/rail_velocity.py:23,rate.py:42,loop_memory.py:52,filter_edge.py:28,filter_leak.py:39,gap_snr.py:34,controller_rate.py:36 | In thesis but not training code,external design checks; displayed numeric evidence not revalidated. |
| App:494 to :505 controller force-change correction todo | GD/controller.py:116; GD/data.py:169; Setup/controller_rate.py:36 | Match4k controller/blockmean distinction. Thesis correction todo separates same-instant controller difference from hold variation; no preprocessing change in final method. |
| `eq:setup_bb` App:515 and :520 to :523 | EV/eval_blackbox.py,separate black-box pipeline | In thesis but not this pipeline; shared grey-box loading/scales are interface only. |
| App:525 to :564 model/data/optimizer hyperparameters/single seed | entry.py:342 to :365 and CFG; GD/config.py:583; GD/model.py:435; FS/lbfgs_polish.py:650 | Match4k,1RK4,400horizon,stride10,history30,LS cutoff,0/2/8states,2x16tanh,float64,.99margin,10%detune,Adam1e-5/eps1e-16,batch512,200ep,val1300,LBFGS100/history50/inner20/windows16384/tols2.6e-5,2.6e-7. One seed=1. |
| App:565 to :577 every PS hyperparameter | FS/ps2_opening.py:37 to :59 | Match lr.001;S1batch64/cal128;head16/H30;cadence25;plateau4/.02;100/350;screen150/.90;last/class>=3;S2pairs8192/2048;batch512/check100;plateau5/.01;500/8000. Auxiliary Adam default eps differs from main eps,not stated in table. |
| App:578 to :589 black-box grid/winner | Server/submit.sh final-bb; EV/eval_common.py:69 | External pipeline/discovery,no selected R8 architecture fixed here. |
| App:591 to :600 OBC/WIP suggestions | FS/obc.py:191, :446; GD/config.py:668 | SVD/rank/refresh/metadata already in code. Suggested appendix prose is not an additional algorithm specification. |

### B6. Code beyond, or conflicting with, thesis detail

| Source | Finding | Verdict |
| :- | :- | :- |
| GD/model.py:435 to :463 | gelsd/standardization/cutoff stated in thesis; finite/map-reproduction/R2>=.999 gates and overwritten draws unstated | In code but not fully in thesis. |
| FS/blocks.py:1365 | Gauge raw14: mb fixed,equal kb/cb splits,Jb:Jh ratio fixed; individual gauge values need not remain physical although inertia admissible | In code but not main thesis; trained combination meaning unchanged. |
| FS/obc.py:205, :446 | Column scaling,SVD cutoff,one rank-loss retention then abort | In code but not detailed in thesis. |
| FS/interconnect.py:902, :1136; GD/training.py:935 | Last-on-tie selection,history preservation,selected/rebuilt/candidate/final events | In code but not explicit thesis selection formula. |
| FS/lbfgs_polish.py:258, :522, :673, :725 | Coarsen before subset choice,twice-gradient reproducibility check,compiled/eager fallback,error rollback/OBC re-sync | In code but not detailed thesis polish. |
| FS/ps2_opening.py:183, :230, :248 | No zero-RMS residual guard,dedicated CPU S2 sampler,no-improvement last-iterate behavior,retained main moments | In code but not fully stated; last-iterate edge conflicts with lowest-held-out-cost statement. |
| GD/training.py:39, :67; FS/interconnect.py:1505 | Component/whole checkpoint formats,pickle hooks,optimizer migration,Adam omitted after accepted polish | In code but not mathematical thesis; load compatibility required. |
| GD/data.py:300; GD/obc_gantry.py:91 | Loads simulator truth fields for diagnostics outside training equations | In code but not method's mathematical data requirement. |
| GD/evaluation.py:166 to :181 | Final NRMS_e validation curve called 'Val sim-RMS (selector)',labelled metres,metre baseline overlaid | Mismatch stale diagnostic units. Saved loss_val is still NRMS_e. |

No disagreement is resolved in favor of either source. Current thesis and D-271 agree on least-squares initialization; older D-251/D-261 and October16 fix prose describe superseded nonlinear prefit. D-262 population std supersedes older RMS(e) wording. D-253 plan said val_before=None; current GD/training.py explicitly measures the rebuilt score to record its history event,then passes that value: equivalent comparison behavior.

## Part C. Equivalence hazards and evaluation interfaces

### C1. RNG sequence and scope

There is no explicit stochastic layer in the deployed tanh MLP/encoder: no dropout, stochastic transition or data augmentation on the required path. Randomness is in initialization, window sampling and PS. Removing an unused function definition consumes no RNG. Removing a constructor, a draw that is later overwritten, or changing float32 versus float64 draw types can change all later weights. CPU construction order matters even for GPU training.

| Source | Draw/state change | Position and equivalence consequence |
| :- | :- | :- |
| entry.py:336 | default_rng(seed1).choice([-1,1],10) | DET only,at module config time,before main. Independent Generator; does not advance global NumPy state. Correct starts skip it. Factors are applied to combinations,including signed m_diff. |
| entry.py:390 to :391 | np.random.seed,torch.manual_seed | Before loading data. Torch manual seed includes device generators according to installed Torch semantics. |
| GD/data.py:290 | np.random.normal | Never executed in spec,snr=None. Lowpass noise is loaded,not synthesized. Removing this inactive branch consumes no draw. |
| entry.py:409 to :410 | Repeat both seed resets | After data/normalization/config reporting,before model. An RNG use before this reset cannot shift model initialization. |
| U/torch_nets.py:186, :189, :192 | Three nn.Linear constructors,each weight and bias initialization | ALL during ANN construction,CPU default float32; order input->hidden->output. Output bias and physical output rows subsequently zeroed. Keep the draws that supplied zeros later. Width/input/output sizes depend on n_a,so equal seed across0/2/8 does not imply identical hidden weights everywhere. |
| FS/pre_encoder.py:376 to :377 | Wb_y then Wb_u uniform_ in requested float64 | ALL after ANN construction. Both are overwritten by least_squares_start but their RNG advancement must remain in an equivalent initialization sequence. Skipping them would change Wa/encoder nonlinear weights. |
| FS/pre_encoder.py:432 to :435 | One joint float64 Xavier draw for Wa[y,u] | ALL; zero-row n_a0 also follows construction,though empty init has no weight values. Drawing separate Wa_y and Wa_u with their separate fan-in/out bounds or float32 first is not equivalent. |
| FS/pre_encoder.py:446, :449, :451 | Three nn.Linear weight/bias draws | ALL after Wb/Wa,default float32 then encoder.to(float64). Physical final weights initially live,added final rows zero; all final bias zero. LS later zeros physical weights and installs affine intercept. Hidden biases remain drawn. |
| GD/training.py:402 | local default_rng(cfg.seed).choice without replacement | Fixed256 validation diagnostic batch,before main fit. Independent RNG,does not advance global NumPy or Torch. No model-construction RNG after it before first training update. Removing this diagnostic draw itself cannot change batch permutations. |
| FS/interconnect.py:600 | self.rng=np.random.default_rng(0) | Main Dtype_DataLoader always uses0,ignoring its seed argument. Its inherited iteration shuffles window IDs from this private state; do not replace with cfg.seed or Torch shuffle. Inherited deepSI iteration source is an external verification boundary. |
| FS/interconnect.py:962 | DataLoader(shuffle=True) | Not executed: training data is array list,not Dataset. Do not replace the actual loader by this alternative and expect equivalent order. |
| FS/ps2_opening.py:83 | Persistent local default_rng(spec.seed) | PS only,used for S1 record/window samples. S1 _sample uses rng.choice(:169) and rng.integers(:173); draw order is record vector first,then time per selected record. |
| FS/ps2_opening.py:148 to :150 | Two temporary nn.Linear constructors,then zero final head weight/bias | PS only,lazily after first main optimizer update. CPU default float32 construction then .to(device,dtype). Head draws consume global CPU Torch RNG even for weights subsequently zeroed. They happen after all deployed model weights are initialized; main array loader uses independent NumPy RNG. Removing them while retaining PS would change head training; removing PS intentionally changes method. |
| FS/ps2_opening.py:157 | local default_rng(seed+321) passed into _sample | PS calibration windows fixed once; independent from S1 RNG stream. |
| FS/ps2_opening.py:218 | local default_rng(seed+1000) | Conditional S2;same stream samples fit snapshot then calibration snapshot. Changing snapshot construction order changes both. |
| FS/ps2_opening.py:230, :235 | CPU torch.Generator manual_seed(seed); torch.randint | Conditional S2 independent CPU sampling with replacement,indices moved to device after drawing. GPU generator or randperm would differ. |
| FS/lbfgs_polish.py:288 | local default_rng(seed).choice without replacement,then sort | ALL if coarsened candidate set exceeds desired16384; otherwise arange/no draw. Effective stride chosen before draw,so a uniform16384-subset of original stride10 set differs. Private RNG does not advance training/global streams. |
| FS/pre_encoder.py:426 to :427 | Legacy separate Kaiming draws | Not executed: wa_encoder_init=xavier_uniform_gain1. |
| U/torch_nets.py:162 | ResNet bypass Kaiming draw | Not executed: ann_arch=mlp. Other network constructors and wrappers use nn.Linear draws only when instantiated; none execute just because definitions are imported. Exact sites listed in C5. |
| common/rezero_gate.py:92, :95 | final.reset_parameters and optional normal_ | Not executed: ANN_REZERO_GATE absent; live init would explicitly reject combining it. |
| FS/augmented_dynamics.py:293 to :299 | dedicated Torch generator and final.uniform_ | Not executed:no wrapper or gate reinitialization. |
| FS/augmented_dynamics.py:355 to :356 | dedicated generator and torch.rand | Not executed: no random added-state input map. |
| gantry_linearization.py:122 to :124 | default_rng42,standard_normal for x/u | Verification helper only,not invoked by import/data_lsq. |
| systems/mass_spring_damper.py:26 | np.random.normal output noise | Benchmark system never instantiated,even though module imported. |
| U/utils.py:173 | np.random.normal SNR helper | Never called by chosen loader. |

Comments in systems/cascaded_tanks.py:16, :17, :25, :50 and pre_encoder.py:679 to :680 contain commented-out random draws,not executable RNG operations. torch.empty/empty_like/zeros/ones do not consume pseudorandom draws,although reading uninitialized values would be a separate bug; used construction slots are populated before their required reads. deepSI's random unique checkpoint code is described in training.py:835,then overwritten by run_id at:843. The inherited source/generator type is not available in the current shell and is not claimed to use NumPy/Torch. Timestamp/SLURM run naming changes filenames,not model numbers.

### C2. Diagnostic removal and state side effects

- `print_run_banner` executes before the second seed reset,reads GPU name/provenance and prints config. It uses no mathematical RNG. git_provenance is cached; removing it changes metadata,not weights. The entry still calls provenance despite an older fix-document note saying it was dropped.
- `load_datasets:300` loads primary validation/test truth states and estimates absorber velocity. These are diagnostic-only reads and no RNG. Removing them affects DataBundle diagnostic fields and report availability,not training normalization,which is computed from measured training positions. OBC `_load_record_float64` also reads x_logical,delta_a,vdelta_a for diagnostic fields. Z_step/Z_field depend on u/y only. A replacement requiring only u/y/r/meta may omit truth reads mathematically,but evaluation currently expects reference-set physical/successor fields.
- `_NfProbe.__init__:391` temporarily replaces simulator.indexer.record_rows by validation rows and sets record_weights=None to build its diagnostic batch. The try/finally restores both. It mutates last_counts while building diagnostic windows; training data construction subsequently overwrites counts. The local RNG does not affect global streams. Removing the probe must not remove the real training window-index alignment or weights.
- `_NfProbe` has two roles: excluded nf/joint/OBC meters and required validation history bookkeeping. `record:593`/`hist_arrays:610` feed selected/rebuilt/polish_candidate/final events and final bestfit in train_model_with_diagnostics. Deleting the entire class without accounting for that bookkeeping changes saved bestfit/history even if training weights are unchanged. Setting nf_probe_print=False only suppresses printing; batch construction/diagnostic execution still occurs.
- `_NfProbe._val_stats:446` calls simulate under no_grad at validation. It updates Interconnect.nb,lazy connection state,the physics `_cur` cache,simulator timing history and diagnostic x_a/error statistics. It uses stored theta_frozen; it does not call a fresh differentiable OBC coefficient or rebuild a basis. MLP/encoder have no dropout/batchnorm,so these extra forwards do not change weights or running normalization stats. Their timing/cache side effects can affect compile warmness and wall time but have no intended objective contribution.
- WindowErrorStats updates/reset and `_joint_probe` nominal-combo comparisons are diagnostics. param_loss is read there but prior is disabled; no RNG. `_obc_probe` reads cached scalars. No oracle selection veto is hidden in these probes. Their history arrays are diagnostics; the separate record-axis validation histories are an existing external interface.
- `OBCLifecycle.refresh` computes diagnostic basis-drift comparisons and report correlations even with verbose=False. The candidate SVD/rank decision and retained basis are mathematical behavior; drift r_J/report printing are diagnostic. `_sync_prediction_state_for_validation` currently uses validation_coefficient,which jointly returns the freshly solved coefficient and diagnostic metrics. Removing diagnostics must preserve the exact coefficient solve,set_frozen and refresh policy. There is no extra random draw in the diagnostic calculations.
- `run_lbfgs_polish` always calls `_gantry_meters` and generic_meters before/after. They log truth-based combo error and param drift but meters_gate=False passes empty dicts into decide. Removing these meters cannot change acceptance,provided coefficient synchronization and the actual validation call remain. No RNG use in meters.
- `_probe_closure` in polish is not optional reporting for equivalence: it evaluates loss/backward twice and controls compiled/eager fallback or abort. `warm_up_lazy_init` is also a real state initializer,not merely a diagnostic. No optimizer step occurs during these checks,but OBC refresh/frozen coefficients/cache/gradients can be updated. Preserve these effects or prove the alternative starts from identical state.
- `closed_loop_report._without_learned` uses deepcopy,zeroes the copy's nonpersistent ANN out_gate and OBC theta_frozen,optionally changes its physical free coordinate. It does not alter original weights. Deepcopy invokes serialization hooks dropping OBC lifecycle/reference/PS but retaining inference correction. The controller bank passed to scoring is mutable/moved to device. Complete report is excluded; `_without_learned`/`_true_free` are imported by evaluation and need compatibility treatment.
- `evaluate_and_save` calls eval and writes a saved system before report scoring,temporarily moves model CUDA/CPU,then writes result arrays. It does not reload best and must not discard accepted polish. State-recovery diagnostic calls eval/no_grad; no stochastic operations. Post-run `compute_gradient_norms:124` explicitly calls train,zero_grad,loss/backward and leaves gradients/caches/mode changed; loss can refresh OBC at the current epoch. It runs after final checkpoint/results writes,so cannot alter those saved trained numbers. Comparing the live object after main exits would see its side effects; a golden checkpoint must be captured at the same boundary.

### C3. Numerical and control-flow hazards

1. **Float64 is not double precision throughout construction.** gantry_ss.py:32 creates raw parameters,P,Cd in float32. model.py:86 promotes those rounded raw tensors. Gantry_State_Block.__init__:730 builds nominal polynomial/LFR buffers in that source dtype before module casting; trainable runtime matrices are rebuilt from promoted parameters. Lb stays the promoted float32 nominal. U/utils.py:21 `torch.Tensor(ndarray)` creates default floating tensors,so std_x,std_u,x_mean,u_mean in blocks.py:773 to :777 and C/D in Linear_Output_Block are rounded on construction,then promoted. Encoder nonlinear nn.Linear and deployed ANN draws are default float32 then cast,while Wb/Wa random draws are explicitly float64. Controller constants/matrices are computed from Python/NumPy doubles. Replacing all literals/tensors/draws with direct float64 changes the baseline,normalizations,gradients and random sequence.
2. **Two related but not identical stored scale copies.** fit_sys.norm receives NumPy64 arrays directly,whereas physical block scales/C passed through to_tensor rounding. OBC tuples normalize using original Norm arrays; transition uses block's stored rounded scales. All modules ultimately have float64 dtype,but starting values differ at earlier rounding boundaries. Single-precision-looking last bits are not removable by a later .double().
3. **Data conditioning order.** Full-rate force casts to64,reshapes and means holds; y-r filter is full-record SOS forward/backward with SciPy default padding before decimation; trim after decimation; r/stored fields point-sampled; all modes trim20 at both ends. Swapping trim/filter order,using ordinary b/a filter,filtering y instead of y-r,filtering inputs,or filtering noisefree y changes numbers. Preserve floor handling for force length and y clipping to N. E scales use loaded/filtered y minus point-sampled r over scored interval,not truth servo error or whole-record std.
4. **Normalization is record-length weighted.** Concatenate arrays in manifest order,calculate population mean/std; velocity scale uses backward difference of reconstructed positions with first difference duplicated,not OBC fourth-order stencil. +1e-8 applies to standard deviations,not variances. P inverse in normalization is formed on the stored float32 P NumPy array before casting to requested dtype; OBC builds inverse after converting P to64. These routes can differ,even in a float64 run.
5. **Encoder input ordering.** The affine LS regressor is[y_flat,u_flat],linear encoder computes W_u*uh+W_y*yh,nonlinear net input is[u_flat,y_flat]. Both histories oldest-first/time-major per-axis. Permuting columns mathematically is legal only with corresponding weights; a state_dict load does not perform it. SSE.na30 but encoder.self.na29 means n+1=30. right extensions zero; validation/prefit/PS must all agree.
6. **LS conditioning/rank/ordering.** Standardize each regressor with population std,replace exactly zero scales by1,centre target; rcond=eps64*max(Phi.shape),not augmented-intercept dimension. LAPACK gelsd on CPU,not default gelsy/CUDA gels. Map slopes/intercept back in same order. Predicted-map finite/allclose/readiness gates stop before training. The bias assignment evaluates current zero-input encoder after zeroing physical final weights/bias; offsets absent on data_lsq. Group ordering and stride determine floating reductions in fit; do not add ridge or change sample set silently.
7. **RK4 and LFR operation order.** Code steps normalized x with h-scaled derivatives,retains z,w and A_combined product instead of returning acceleration directly; numerator has Horner structure,denominator expanded. ANN correction is added after complete discrete RK4,not divided by Ts or injected as derivative. Hoisted parameter matrices form one graph per objective shared across all steps; detach or cache across steps/updates changes derivatives. Hand tangent must match all RK stages,including state-dependent Y and normalizations. Reordering sums,matmuls or using solve changes rounding.
8. **OBC stack/coordinates.** Reference stride1 irrespective of training stride10; record-major,sample-major,physical-row-major flatten. Endpoint k=2..N-4; successor q[k+1],qd[k-1]. Affine solve uses[J,fbar],columns normalized by raw2norm,thin SVD64,rank threshold shape*eps*Smax. Normal-equation inverse or unscaled gamma solve is mathematically related but numerically different. Save/load coefficient basis is the deployed fbar convention. Preserve rank-loss fallback's old vbar with old basis,not only old U.
9. **OBC differentiation and refresh.** Basis build/vbar detached,coefficient remains differentiable through ANN on the entire reference set,then tangent_pair uses that direction in differentiable operations. set_frozen detaches a prediction copy but must not detach the tensor threaded to training. Epoch floor includes1e-6. Prefit may initialize epoch0 basis before training; post-best restore forces refresh; polish holds epoch/basis fixed. Predictions after checkpoint loading must use saved frozen correction,not a newly rebuilt basis or new solve against unavailable reference state.
10. **Controller realization.** Fixed nominal ruleOfThumb atY0,100Hz,tf2ss after4kHz bilinear discretization. Using stored20kHz state matrices,learned/detuned physics,another realization,or reordering controller state channels changes numbers. Normalization folded into B/C/D; bank.step fused baddbmm; output-first and xc0 fixed. Model corrects recorded u using measured-model residual,not reference-model directly. Equal-length grouping changes reduction/BLAS batch dimensions; keep it for equivalence.
11. **Window counts/order.** Four arrays built by fixed_System_data.to_hist_future_data,range(max(history)+max(future,right),len+1,stride),not sliding across concatenated record boundaries. ctrl_ix int64 and per-record loss_w must match first/last window content; group and manifest order carry controller identity. Main private sampler seed0,full batches,floor updates/epoch,200epochs; validation initially and every1300updates,not necessarily at last remainder. No guaranteed final validation beyond cadence. Main closure returns MSE,logging 'sqrt loss' does not mean optimizing RMS. Train loss history stores passed accumulated MSE values.
12. **PS changes parameters outside main Adam.** S1 current detached residual re-solved each time,normalization by global batch residual RMS has no zero/finite guard. Main Adam moments retained during S1/S2. S1 separate Adam has default eps1e-8,not main1e-16. S1 head lazily initialized after first update; calibration and fit sampling private seeds. S2 shared hidden weights alter physical f_aug even when physical final rows unchanged. S2 target variance default correction1,calibration per fixed snapshot; no-improvement restoration issue in B2. Replacing auxiliary optimizer parameter list by masked updates can alter state/order. Fallback retains S1 modifications,so 'fallback plain' is not bitwise a run where PS never attached.
13. **L-BFGS batch and lifecycle.** FixedBatch coarsens stride using n_total//want,then draws sorted subset. Same windows/weights every closure. One chunk in spec; alternative chunk accumulation changes rounding even if mathematically same gradient. warmup and two determinism closures run before actual iterations; compiled can fall back eager. strong_wolfe,lr1,max_iter20 per block and100total,history50,absolute tolerances as specified. _make_closure sums Python floats and returns torch.as_tensor(total) with default floating dtype; this scalar is distinct from double loss graphs. Preserve or explicitly test that boundary. params with unused gradients get zero tensors. _loss_value uses no_grad,eager route even when optimizer closures compiled. Acceptance checks finite final loss and strictly lower eager validation score; equality rejects. Failed/aborted phase returns unchanged/restored weights; re-sync frozen coefficient after every restore.
14. **Device/compile differences.** CPU builds,CUDA prefit/fit/polish,model CPU outside phases; bank moves itself as needed. Validation remains eager on current device while main compiled route is used only for CUDA plus grad enabled. Both hfn and output_only compiled; cudagraph_mark_step_begin and cloned outputs are needed for graph replay lifetime. Prefit calls direct eager closed_loop_rollout with no_grad. L-BFGS training closures can compile a new larger batch graph. Different device/hardware/BLAS/SVD/compiler/Torch/CUDA versions/reduction fusion change float outcomes; no all-platform bitwise promise follows from same seed. static_pass_buffers=False excludes custom static-copy autograd; checkpoint_chunk0 excludes gradient-checkpoint recomputation. No AMP/mixed-precision branch in selected path.
15. **Serialization/selection boundary.** Parent fit restores best weights/optimizer,keeps full epoch_id/Loss_val/Loss_train,then rebuilds OBC correction and overwrites deepSI selected checkpoint. The Adam bestfit_selected score can predate this rebuild. Final checkpoint bestfit is final event score; polish accepted -> omit obsolete Adam moments. Never reload _best during final evaluation and discard accepted polish. _cur per-forward cache excluded from pickle; model connection matrices are non-buffer tensor lists following _apply. PyTorch state_dict has buffers and parameters,but not these lists or full Norm; reconstruction must rebuild them compatibly.
16. **Environment and metadata.** the launcher's export-all setting can carry unrelated overrides. Run IDs,paths/timestamps/provenance are nonnumerical except as filename identity. Mode-specific output family must match discovery; final noisefree family is thesis_taf_noisefree_data_lsq,which current evaluator does not auto-select unless THESIS_RUNS_DIR is overridden. Holding training reproducible also requires original manifests/file bytes and ordering,not just code/config. No claim of existing GPU results verified here.

### C4. Evaluation compatibility contract

**Discovery and configuration.** `EV/eval_common.py:35` THESIS_EVAL=final selects discover_runs. Default family is `simulations/gantry_subnet/thesis_taf_lowpass_noise_data_lsq`; override THESIS_RUNS_DIR. It scans `<run_id>/config.json`,requires numeric run_id,ignores non-None TRUNCATE_S/N_ITS,and maps(OBC,NX_ANN,COMBO_INIT_DETUNE is not None,PS2) to R labels. Duplicates raise. It does not include MODE/noise/seed/arch in this discovery key,so both noise modes cannot be mixed under one scanned family without collision risk. Existing separate family directories matter.

`EV/eval_greybox.py:35` rebuilds RunConfig from every recognized uppercase field in config.json,converts ANN_ROUTE_IX and LBFGS_FREEZE lists to tuples,forces cpu/eager/no truncation,and restores na_nb_override=NA_NB,nf_override=NF. It uses cfg.hp rather than importing the entry CFG. It seeds twice,loads same data/normalization,builds model,attaches simulator with e_train/e_val,and loads weights without rerunning prefit. Removing unused fields from RunConfig can affect reconstruction of historical configs even if new math works; new defaults must not silently replace final method settings.

**Pipeline imports consumed by evaluation.**

| Consumer | Required imports/API |
| :- | :- |
| EV/eval_greybox.py:41 to :68 | RunConfig; load_datasets,compute_normalization; build_model; build_closed_loop keyword args train_files,val_files,val_data,e_train,e_val,verbose. |
| EV/eval_greybox.py:93, :106 to :110 | _load_record(filename,cfg,folder) returning(System_data,e); load_checkpoint(fs,path,joint_estimation=True,start_phase='lbfgs'); build_controller_bank,y_op_for; excluded-module helpers _without_learned,_true_free; closed_loop_free_run_rms_batch; servo_sigma. |
| EV/eval_blackbox.py:37 to :38, :112 to :116 | traj_dir,_record_at_fs_new,_load_u,_resample_u,record_trim,record_files,servo_sigma. Grey-box preprocessing is shared even though black-box training is outside scope. |
| EV/eval_overlap.py:50, :67 to :71 | make_basis_builder; load_checkpoint; _true_free; build_reference_set,make_float64_block; build_stacked_sensitivity; Static_ANN_Block type identity. |
| EV/eval_negation.py:39 to :42 | load_checkpoint; build_reference_set,make_float64_block; build_stacked_sensitivity; Static_ANN_Block; deployed_basis from eval_overlap. |
| EV/eval_frf.py:35, :77 | load_checkpoint; callable hfn(x,u)->(y,x_next),differentiable in state/input; norm.Cd_norm,std_u,ystd,x_mean,std_x,u_mean; physical+added state layout. |
| EV imports from scripts/gantry/thesis-results/inspect_checkpoint.py | clamped: deepcopy,find Static_ANN_Block,ann.out_gate added rows zero. added_poles: fs.norm.transform,fs.encoder,fs.hfn,na/nb/right extensions,trainable dtype,connected state layout. Its standalone checkpoint_config expects obsolete AUGMENTATION_INIT; eval_greybox deliberately does not call it. |

**Object contract.** DataBundle.train_list/val_list/test_list,train_files/val_files/test_files,e_train/e_val/e_test; train_data and val_ckpt_data as deepSI lists. Norm fields x_mean,std_x,u_mean,std_u,y0,ystd,Cd_norm,Dd_np and concatenated arrays. Fit system fields encoder,hfn,norm,na,nb,na_right,nb_right,simulator,parameters_with_names,optimizer,bestfit. hfn.connected_blocks remains ordered [reduced physics,fixed output,ANN],physical first6 state coordinates followed by n_a. Evaluators locate physics by free_params,ANN by isinstance. No checkpoint loading can substitute another class identity unnoticed.

OBC evaluators use hfn.obc_correction.space,vbar,theta_frozen; reference fields Z_step,Z_field,record_ix,sample_ix,x_phys_next_rec,record_names,n; physics APIs PARAM_NAMES,COMBO_NAMES,params_init,free_params,combo_init,Lb,combinations_from_free,free_from_combinations,transition_from_free,pass_mats,estimation_parameters. Frozen-Y comparison sets Y_op,N_op,d_op from pass_mats and calls the block's frozen path,which is not training method but is required existing evaluation behavior. out_gate is nonpersistent: interventions must start at ones on a reconstructed unmodified model.

**Checkpoint names and formats.**

- Preferred final: `<family>/<job>/gantry_ckpt_<job>.pt` plus `.npz`. Pre-polish pair uses `_pre_lbfgs` suffix. `.pt` contains hfn and encoder component state_dicts,and optimizer only when it matches weights. Accepted polish deliberately omits optimizer. `.npz` metadata keys: done_epochs,bestfit,bestfit_selected,elapsed,hp(JSON string),orig_run_id,lbfgs_outcome(JSON),optimizer_state_dropped,start_phase,resumed_from,resumed_at_epoch,ps2_summary. Pre-polish meta is smaller: done_epochs,bestfit,hp,orig_run_id.
- Preliminary: deepSI `<class>_<job>_best.pth` (unique_code assigned job). `EV/checkpoint_status` uses exactly one `*_<job>_best.pth`,never automatically falls back to _last. resolve_checkpoint accepts full.pth,path minus.pth,or base to which_best.pth is appended. Whole-file torch.load uses map_location=cpu,weights_only=False. Pair loader uses torch.load default weights-only behavior of installed version. Whole checkpoint's hfn/encoder are live modules and require original importable module/class paths,including frozen correction.
- State_dict prefixes from construction: `connected_blocks.0.*` for physical buffers/parameters,`.1.C`/`.1.D` for fixed output,`.2.net.net.{0,2,4}.{weight,bias}` for deployed2x16 MLP. Physical `free_params`,params_init,combo_init,Lambda_combo,Lb,m_diff_eta0,m_diff_scale,coordinate_version and inherited nominal/LFR/scaling buffers remain named and shaped. OBC adds `obc_correction.E`,`.vbar`,`.theta_frozen` (11vector affine). corr._phy is an ordinary list reference,not another registered parameter subtree. Encoder root keys Wb_psi_y,Wb_psi_u,Wa_psi_y,Wa_psi_u and `net.{0,2,4}.{weight,bias}`; data_lsq has no u_off/y_off/x_off buffers. ANN out_gate is not persistent and absent from state_dict. Model list order is therefore part of compatibility.
- load_checkpoint checks hfn dtype,complete hfn and encoder key/shape equality and coordinate_version2,then loads. Do not drop nominal buffers as 'unused math' while claiming old state_dict loading unchanged. Legacy raw log_params versus reduced free_params are incompatible. infer_arch_from_hfn searches `.net.net.<even>.weight` shape/layer count; no state-count guess from current CFG. resolve_resume_hp owns architecture from saved hp only for real resume,which is not a campaign axis.
- Pickle hooks strip OBC lifecycle,reference and PS/traj training objects,clear `_cur`,strip compiled simulator callables,and keep frozen inference correction. checkpoint_load_system stashes/reinstalls live training auxiliaries around whole-system restore. Composed optimizer property migrates legacy `optimizer` to `_optimizer`. Potential external-version seam: resolve_checkpoint's whole-format optimizer lookup explicitly checks d.get('optimizer'),while composed live storage is `_optimizer`; whether super save materializes a legacy optimizer key needs installed-deepSI confirmation. This does not block evaluation start_phase='lbfgs',which ignores Adam moments,but matters to any later exact resume.

**Report/config/cache arrays.** Training `gantry_results_<job>.npz` contains y_ref,y_hat_cl,t_val,u_val,epoch_id,loss_val,loss_train,cl_rows,cl_val_files,cl_val_rms,cl_test_files,cl_test_rms,std_x,x_mean,std_u,u_mean,ystd,y0,Cd_norm,Dd_np,P_matrix,noise_snr,noise_sigma,cheat_n,dt,na,nb,nf,NX_PHYS,NX_ANN,nxd,hp(JSON),config(JSON),conditional y_hat_test_cl/y_test_ref,param_names,params_init,params_learned. `cl_rows` order is model,physics part,baseline start,baseline true,oracle. cl_rms shape(5,n_records,3),including NaN E5 row. Full prediction arrays have NaN before k0. t_val includes trim offset; saved config here has GIT=None while external config.json includes passed provenance,run_id,data_dir,hp.

`val_hist.npz` and result history keys are val_hist_event,val_hist_update,val_hist_epoch,val_hist_score,val_hist_sim_rms,val_hist_rms,val_hist_accepted,val_sigma_e. Event order includes Adam(initial and each cadence),selected,rebuilt for OBC,polish_candidate if phase ran with finite score,final. Histories survive best reload via an external probe reference. A candidate history entry after rejection still refers to the rejected candidate's last simulator statistics; final event copies selected/rebuilt row. Preserve these event identities if reproducing reports.

EV/eval_greybox.py:180 compares model/physics/base_start/base_true per-record RMS to saved cl_val_rms/cl_test_rms with relative error<1e-6. Validation score against checkpoint uses mean RMS/sigma_e and prints threshold1e-3. It does not re-sync OBC after loading: that would change deployed inference model. Evaluation caches use records,k0,split,y,sigma_e,model plus requested physics/base_start/base_true/clamped/base_lti,run,job,nx_ann,obc,ps2,status,final,update,total,ref_score,val_score,config,param_names,params_init,params_learned. File names preds_Rn.npz for test,preds_Rn_<split>.npz otherwise. make_results checks job/code identity and consumes these cache fields; floor/servo sigma currently also come from black-box paired caches. Eval overlap uses train,val and testI1-I6; negation uses same stacks. No figure/table script itself is required in clean training folder.

### C5. Exhaustive source RNG and implicit-initializer site ledger

The following table enumerates executable source calls that create/advance RNG state or construct random-weight neural layers in the four requested code locations. Alias calls inside `_sample` are included. Status follows caller classification in A4. Constant/zero initialization calls are omitted because they consume no random draws; commented text is not an executable call. Wrappers invoking a child network are included to expose indirect draws in inactive branches. This syntax list complements C1's temporal analysis; neither claims runtime coverage.

| Source | Caller | Call | Path classification |
| :- | :- | :- | :- |
| `model_augmentation/fit_systems/augmented_dynamics.py:293` | `reset_final_layer` | `torch.Generator()` | not needed |
| `model_augmentation/fit_systems/augmented_dynamics.py:293` | `reset_final_layer` | `torch.Generator().manual_seed(int(seed))` | not needed |
| `model_augmentation/fit_systems/augmented_dynamics.py:297` | `reset_final_layer` | `layer.weight.uniform_(-bound, bound, generator=gen)` | not needed |
| `model_augmentation/fit_systems/augmented_dynamics.py:299` | `reset_final_layer` | `layer.bias.uniform_(-bound, bound, generator=gen)` | not needed |
| `model_augmentation/fit_systems/augmented_dynamics.py:355` | `draw_input_map` | `torch.Generator()` | not needed |
| `model_augmentation/fit_systems/augmented_dynamics.py:355` | `draw_input_map` | `torch.Generator().manual_seed(int(seed))` | not needed |
| `model_augmentation/fit_systems/augmented_dynamics.py:356` | `draw_input_map` | `(2.0 * torch.rand((n_aug, nz), generator=gen, dtype=torch.float64) - 1.0).numpy()` | not needed |
| `model_augmentation/fit_systems/augmented_dynamics.py:356` | `draw_input_map` | `torch.rand((n_aug, nz), generator=gen, dtype=torch.float64)` | not needed |
| `model_augmentation/fit_systems/blocks.py:48` | `Static_ANN_Block.__init__` | `net(n_in=self.nz, n_out=self.nw, n_nodes_per_layer=n_nodes_per_layer, n_hidden_layers=n_hidden_layers, activation=activation, **{} if net_kwargs is None else net_kwargs)` | required |
| `model_augmentation/fit_systems/interconnect.py:558` | `modified_encoder_net.__init__` | `simple_res_net(n_in=nb * np.prod(self.nu, dtype=int) + na * np.prod(self.ny, dtype=int), n_out=nx, n_nodes_per_layer=n_nodes_per_layer, n_hidden_layers=n_hidden_layers, activation=activation)` | not needed |
| `model_augmentation/fit_systems/interconnect.py:600` | `Dtype_DataLoader.__init__` | `np.random.default_rng(0)` | required |
| `model_augmentation/fit_systems/interconnect.py:649` | `SSE_Interconnect.init_nets` | `self.e_net(nb=self.nb + nb_right, nu=nu, na=self.na + na_right, ny=ny, nx=self.nx, **self.e_net_kwargs)` | required |
| `model_augmentation/fit_systems/lbfgs_polish.py:288` | `FixedBatch.__init__` | `np.random.default_rng(seed)` | required |
| `model_augmentation/fit_systems/lbfgs_polish.py:288` | `FixedBatch.__init__` | `np.random.default_rng(seed).choice(n_built, size=self.n_used, replace=False)` | required |
| `model_augmentation/fit_systems/pre_encoder.py:267` | `linear_encoder_init.__init__` | `nn.Linear(self.n_in, n_nodes_per_layer)` | not needed |
| `model_augmentation/fit_systems/pre_encoder.py:270` | `linear_encoder_init.__init__` | `nn.Linear(n_nodes_per_layer, n_nodes_per_layer)` | not needed |
| `model_augmentation/fit_systems/pre_encoder.py:273` | `linear_encoder_init.__init__` | `nn.Linear(n_nodes_per_layer, self.n_out)` | not needed |
| `model_augmentation/fit_systems/pre_encoder.py:376` | `linear_encoder_init_aug.__init__` | `torch.empty(nx, (n + 1) * ny, dtype=dtype).uniform_(-bound, bound)` | required |
| `model_augmentation/fit_systems/pre_encoder.py:377` | `linear_encoder_init_aug.__init__` | `torch.empty(nx, (n + 1) * nu, dtype=dtype).uniform_(-bound, bound)` | required |
| `model_augmentation/fit_systems/pre_encoder.py:426` | `linear_encoder_init_aug.__init__` | `nn.init.kaiming_uniform_(wa_y)` | required |
| `model_augmentation/fit_systems/pre_encoder.py:427` | `linear_encoder_init_aug.__init__` | `nn.init.kaiming_uniform_(wa_u)` | required |
| `model_augmentation/fit_systems/pre_encoder.py:433` | `linear_encoder_init_aug.__init__` | `nn.init.xavier_uniform_(wa, gain=1.0)` | required |
| `model_augmentation/fit_systems/pre_encoder.py:446` | `linear_encoder_init_aug.__init__` | `nn.Linear(self.n_in, n_nodes_per_layer)` | required |
| `model_augmentation/fit_systems/pre_encoder.py:449` | `linear_encoder_init_aug.__init__` | `nn.Linear(n_nodes_per_layer, n_nodes_per_layer)` | required |
| `model_augmentation/fit_systems/pre_encoder.py:451` | `linear_encoder_init_aug.__init__` | `nn.Linear(n_nodes_per_layer, self.n_out)` | required |
| `model_augmentation/fit_systems/pre_encoder.py:577` | `SS_pre_encoder.init_nets` | `self.e_net(nb=self.nb + nb_right, nu=nu, na=self.na + na_right, ny=ny, nx=self.nx, **self.e_net_kwargs)` | not needed |
| `model_augmentation/fit_systems/pre_encoder.py:585` | `SS_pre_encoder.init_nets` | `self.hf_net(nx=self.nx, nu=self.nu, ny=self.ny, **self.hf_net_kwargs)` | not needed |
| `model_augmentation/fit_systems/ps2_opening.py:83` | `PS2Opening.__init__` | `np.random.default_rng(spec.seed)` | required |
| `model_augmentation/fit_systems/ps2_opening.py:148` | `PS2Opening._start` | `nn.Linear(self.n_a, self.width)` | required |
| `model_augmentation/fit_systems/ps2_opening.py:149` | `PS2Opening._start` | `nn.Linear(self.width, self._M * ny)` | required |
| `model_augmentation/fit_systems/ps2_opening.py:157` | `PS2Opening._start` | `np.random.default_rng(self.spec.seed + 321)` | required |
| `model_augmentation/fit_systems/ps2_opening.py:169` | `PS2Opening._sample` | `rng.choice(len(rec_ix), size=B, p=lens / lens.sum())` | required |
| `model_augmentation/fit_systems/ps2_opening.py:173` | `PS2Opening._sample` | `rng.integers(k0, len(u) - M - 1)` | required |
| `model_augmentation/fit_systems/ps2_opening.py:218` | `PS2Opening._fit_map` | `np.random.default_rng(sp.seed + 1000)` | required |
| `model_augmentation/fit_systems/ps2_opening.py:230` | `PS2Opening._fit_map` | `torch.Generator(device='cpu')` | required |
| `model_augmentation/fit_systems/ps2_opening.py:230` | `PS2Opening._fit_map` | `torch.Generator(device='cpu').manual_seed(sp.seed)` | required |
| `model_augmentation/fit_systems/ps2_opening.py:235` | `PS2Opening._fit_map` | `torch.randint(len(Xf), (sp.s2_batch,), generator=g)` | required |
| `model_augmentation/fit_systems/ps2_opening.py:235` | `PS2Opening._fit_map` | `torch.randint(len(Xf), (sp.s2_batch,), generator=g).to(Xf.device)` | required |
| `model_augmentation/systems/gantry_linearization.py:122` | `verify_linearization` | `np.random.default_rng(42)` | not needed |
| `model_augmentation/systems/gantry_linearization.py:123` | `verify_linearization` | `rng.standard_normal(nx)` | not needed |
| `model_augmentation/systems/gantry_linearization.py:124` | `verify_linearization` | `rng.standard_normal(nu)` | not needed |
| `model_augmentation/systems/mass_spring_damper.py:26` | `Mass_spring_damper.h` | `np.random.normal(0, self.sigma_n[0])` | not needed |
| `model_augmentation/utils/torch_nets.py:14` | `linear_encoder_net.__init__` | `nn.Linear(nb * np.prod(self.nu, dtype=int) + na * np.prod(self.ny, dtype=int), nx, bias=False)` | not needed |
| `model_augmentation/utils/torch_nets.py:27` | `feed_forward_nn.__init__` | `nn.Linear(n_in, n_nodes_per_layer)` | not needed |
| `model_augmentation/utils/torch_nets.py:30` | `feed_forward_nn.__init__` | `nn.Linear(n_nodes_per_layer, n_nodes_per_layer)` | not needed |
| `model_augmentation/utils/torch_nets.py:32` | `feed_forward_nn.__init__` | `nn.Linear(n_nodes_per_layer, n_out)` | not needed |
| `model_augmentation/utils/torch_nets.py:44` | `identity_init_simple_res_net.__init__` | `nn.Linear(n_in, n_out)` | not needed |
| `model_augmentation/utils/torch_nets.py:48` | `identity_init_simple_res_net.__init__` | `zero_init_feed_forward_nn(n_in, n_out, n_nodes_per_layer=n_nodes_per_layer, n_hidden_layers=n_hidden_layers, activation=activation)` | not needed |
| `model_augmentation/utils/torch_nets.py:68` | `linear_mapping.__init__` | `nn.Linear(n_in, n_out)` | not needed |
| `model_augmentation/utils/torch_nets.py:85` | `zero_init_linear_mapping.__init__` | `nn.Linear(n_in, n_out)` | not needed |
| `model_augmentation/utils/torch_nets.py:101` | `zero_init_feed_forward_nn.__init__` | `nn.Linear(n_in, n_nodes_per_layer)` | not needed |
| `model_augmentation/utils/torch_nets.py:104` | `zero_init_feed_forward_nn.__init__` | `nn.Linear(n_nodes_per_layer, n_nodes_per_layer)` | not needed |
| `model_augmentation/utils/torch_nets.py:107` | `zero_init_feed_forward_nn.__init__` | `nn.Linear(n_nodes_per_layer, n_out)` | not needed |
| `model_augmentation/utils/torch_nets.py:143` | `paper_resnet.__init__` | `nn.Linear(n_in, n_nodes_per_layer)` | not needed |
| `model_augmentation/utils/torch_nets.py:146` | `paper_resnet.__init__` | `nn.Linear(n_nodes_per_layer, n_nodes_per_layer)` | not needed |
| `model_augmentation/utils/torch_nets.py:148` | `paper_resnet.__init__` | `nn.Linear(n_nodes_per_layer, n_out)` | not needed |
| `model_augmentation/utils/torch_nets.py:162` | `paper_resnet.__init__` | `nn.init.kaiming_uniform_(self.W_a[output_row:output_row + 1], a=5 ** 0.5)` | not needed |
| `model_augmentation/utils/torch_nets.py:186` | `live_g_feed_forward_nn.__init__` | `nn.Linear(n_in, n_nodes_per_layer)` | required |
| `model_augmentation/utils/torch_nets.py:189` | `live_g_feed_forward_nn.__init__` | `nn.Linear(n_nodes_per_layer, n_nodes_per_layer)` | required |
| `model_augmentation/utils/torch_nets.py:192` | `live_g_feed_forward_nn.__init__` | `nn.Linear(n_nodes_per_layer, n_out)` | required |
| `model_augmentation/utils/torch_nets.py:212` | `zero_init_resnet.__init__` | `nn.Linear(n_in, n_nodes_per_layer)` | not needed |
| `model_augmentation/utils/torch_nets.py:215` | `zero_init_resnet.__init__` | `nn.Linear(n_nodes_per_layer, n_nodes_per_layer)` | not needed |
| `model_augmentation/utils/torch_nets.py:218` | `zero_init_resnet.__init__` | `nn.Linear(n_nodes_per_layer, n_out)` | not needed |
| `model_augmentation/utils/torch_nets.py:223` | `zero_init_resnet.__init__` | `nn.Linear(n_in, n_out)` | not needed |
| `model_augmentation/utils/torch_nets.py:240` | `unit_variance_feed_forward_nn.__init__` | `nn.Linear(n_in, n_nodes_per_layer)` | not needed |
| `model_augmentation/utils/torch_nets.py:243` | `unit_variance_feed_forward_nn.__init__` | `nn.Linear(n_nodes_per_layer, n_nodes_per_layer)` | not needed |
| `model_augmentation/utils/torch_nets.py:246` | `unit_variance_feed_forward_nn.__init__` | `nn.Linear(n_nodes_per_layer, n_out)` | not needed |
| `model_augmentation/utils/torch_nets.py:294` | `HybridGantryEncoder.__init__` | `zero_init_feed_forward_nn(n_in=n_in, n_out=self.NX_ANN, n_nodes_per_layer=n_nodes_per_layer, n_hidden_layers=n_hidden_layers, activation=activation)` | not needed |
| `model_augmentation/utils/torch_nets.py:398` | `LinearInitEncoderWrapper.__init__` | `zero_init_feed_forward_nn(n_in=n_in, n_out=nx_ann, n_nodes_per_layer=n_nodes_per_layer, n_hidden_layers=n_hidden_layers, activation=activation)` | not needed |
| `model_augmentation/utils/torch_nets.py:440` | `positive_default_encoder_net.__init__` | `simple_res_net(n_in=nb * np.prod(self.nu, dtype=int) + na * np.prod(self.ny, dtype=int), n_out=nx, n_nodes_per_layer=n_nodes_per_layer, n_hidden_layers=n_hidden_layers, activation=activation)` | not needed |
| `model_augmentation/utils/utils.py:173` | `compute_noisy_y_from_SNR` | `np.random.normal(0, sigma_noise, y_true.shape)` | not needed |
| `scripts/gantry/common/rezero_gate.py:92` | `apply_rezero_gate` | `final.reset_parameters()` | not needed |
| `scripts/gantry/common/rezero_gate.py:95` | `apply_rezero_gate` | `nn.init.normal_(final.weight, mean=0.0, std=float(std))` | not needed |
| `scripts/gantry/gantry_dynamic/data.py:290` | `load_datasets` | `np.random.normal(0, sigma_n, sd.y.shape)` | required |
| `scripts/gantry/gantry_dynamic/data.py:290` | `load_datasets` | `np.random.normal(0, sigma_n, sd.y.shape).astype(cfg.dtype_np)` | required |
| `scripts/gantry/gantry_dynamic/training.py:402` | `_NfProbe.__init__` | `np.random.default_rng(cfg.seed)` | required |
| `scripts/gantry/gantry_dynamic/training.py:402` | `_NfProbe.__init__` | `np.random.default_rng(cfg.seed).choice(len(d[0]), size=n, replace=False)` | required |
| `scripts/gantry/gantry_interconnect_dynamic.py:336` | `module` | `np.random.default_rng(_seed)` | module-level configuration path |
| `scripts/gantry/gantry_interconnect_dynamic.py:336` | `module` | `np.random.default_rng(_seed).choice([-1.0, 1.0], size=10)` | module-level configuration path |
| `scripts/gantry/gantry_interconnect_dynamic.py:390` | `main` | `np.random.seed(cfg.seed)` | required |
| `scripts/gantry/gantry_interconnect_dynamic.py:391` | `main` | `torch.manual_seed(cfg.seed)` | required |
| `scripts/gantry/gantry_interconnect_dynamic.py:409` | `main` | `np.random.seed(cfg.seed)` | required |
| `scripts/gantry/gantry_interconnect_dynamic.py:410` | `main` | `torch.manual_seed(cfg.seed)` | required |

83 executable RNG/generator/implicit network-initialization call sites listed. Nested Generator/manual_seed calls on one line are separate call sites. Inactive wrappers invoke child constructors but are never instantiated on the campaign path.

### C6. Remaining static-audit boundaries

- No method was run and no numerical claim was tested. Definition and equation coverage are checked mechanically, but branch reachability is static reasoning conditional on the stated configs and normal successful execution. Data-dependent PS fallback/S2 stopping, rank-loss handling, non-finite losses, compile fallback and polish rejection cannot be selected in advance from config alone; both branches are inventoried.
- The active Python has neither NumPy nor deepSI. A small standalone attempt to print the seed-1 detuning factors failed on NumPy import before any draw. The inventory records the exact deterministic choice expression rather than an unverified numeric sign vector. No environment/package installation was attempted. Installed deepSI iteration/parameter-discovery/pickle behavior and actual server library versions remain external equivalence inputs.
- All52 active equation labels in Sections II to V and99_appendix.tex occur in Part B. Method paragraphs are grouped by subject,including draft/WIP qualifications. Empirical claims such as filter leakage,FRF agreement,SNR,crossover,readiness of an actual run and reported result factors are mapped to producers,not falsely certified by static inspection.
- The complete symbol ledger is syntax-based and includes647 definitions across42 Python files. Abstract/overridden methods,alternative wrappers,diagnostic callbacks,nested functions and methods inherited by the selected objects are distinguished. Parameterized_Gantry_State_Block is not instantiated,although model.py:85 reads its inherited PARAM_NAMES class attribute; the same underlying constant lives on _Trainable_Gantry_State_Block. static_pass_buffers=False excludes _StaticCopy/StaticPassBuffers and all their methods. Importing an unused benchmark class is not executing its body.
- Whole checkpoint compatibility requires the old import paths and all expected keys,even for buffers a compact rewrite might mathematically eliminate. Existing evaluation imports helpers from excluded closed_loop_report.py; this is an interface requirement that must be reconciled in the later design. This inventory makes no design proposal and does not expand scope to implement diagnostics.

### C7. Source fingerprint for later comparison

Repository HEAD at inspection:4271a314d3b2acf5508e37660c0f83accee64366,branch Augmentation. Working tree already contained many user changes; no git write was performed. Hashes below identify the source bytes actually inspected rather than relying on HEAD. This inventory is intentionally not hashed into itself.

| File | SHA256 |
| :- | :- |
| `scripts/gantry/gantry_interconnect_dynamic.py` | `3aaafbb159d25415ef4f6ccde0e5ae9f4d084acd13ce3b151427def7160f56c5` |
| `scripts/gantry/gantry_dynamic/__init__.py` | `72d4cc64cf1d2023bb06ff83f69181307884e4325b8c712c56204e6b36698e4a` |
| `scripts/gantry/gantry_dynamic/baselines.py` | `55c262288bc0e0f644d562da96e34344d1d10f58501b0967d2c501899b67dfd1` |
| `scripts/gantry/gantry_dynamic/closed_loop_report.py` | `6042b7bc3023117aa683a0696a790cf5844a09e58f63bf82bcf9fa30eb35e348` |
| `scripts/gantry/gantry_dynamic/config.py` | `61432505dad81a526801548dfa7318504ca76f847b2a5a30a629069cebeb31db` |
| `scripts/gantry/gantry_dynamic/controller.py` | `1fbee4540801f9b1bf447280a49d28ca023a4a98f263f800f4e49f057a203e03` |
| `scripts/gantry/gantry_dynamic/data.py` | `b4fc791efdbead4dfd09dd91d36f39f889a89622817ec8d0ad2cf28928482766` |
| `scripts/gantry/gantry_dynamic/diagnostics.py` | `0c62a1be3532a376ef3b6d17ab29ae8161d668254d3566d9338d2f6faf918968` |
| `scripts/gantry/gantry_dynamic/evaluation.py` | `aff0de2579899ea1a9db8e20a55f35fa360fe77bea5fa188d44661ddb2935654` |
| `scripts/gantry/gantry_dynamic/model.py` | `763399354570c881ef96700e4991f799af939db41d8342f2c87c007d18d5630a` |
| `scripts/gantry/gantry_dynamic/obc_diagnostics.py` | `576d10b4ba04547d4ffb0b92c9808e3cf82eaf340a8e75b6f7758ee3e95e7fc1` |
| `scripts/gantry/gantry_dynamic/obc_gantry.py` | `19ecf225d4b272c494bc26769cc5a55b6313d3d3f9da665a718bdd7b5650342e` |
| `scripts/gantry/gantry_dynamic/training.py` | `b43e6358b0ee458738af466ec9c15ce96d1045789cb9b239dfa9d93f2c7c2d02` |
| `scripts/gantry/common/feedback_curriculum.py` | `cf9b655c6f9ccc54faf69e9f99ef869de6a41c8c1f484967bb8bfb535fde8371` |
| `scripts/gantry/common/oracle.py` | `bf3b700cf5ddd8bdd8b7e17f202b6d27f3c0b16bef0a1b6dc4a7741e4a28f579` |
| `scripts/gantry/common/orth_penalty.py` | `94e1f07918e457ee69fe93e4701dcfebe3ae03dbbb63158d3da47f24eb26f9df` |
| `scripts/gantry/common/rezero_gate.py` | `e0bee5bf035cf06ef3506125f60309c2e4c3029db0b0c125c5b916c3188915b3` |
| `model_augmentation/__init__.py` | `3ccd35009717f50c0d0756acd3b5b1baa3f0fc8478d6efbc7ed99a7ffdbdc9fa` |
| `model_augmentation/fit_systems/__init__.py` | `07a18a4e372054a040f368eaf1167d843592e8e1b4684bd91deb648b1bffbd6f` |
| `model_augmentation/fit_systems/augmented_dynamics.py` | `2bb9aaf47830182cd892b5f3474a04371fcc2e12cecd22748f6089d47fd97d48` |
| `model_augmentation/fit_systems/blocks.py` | `47004dfc734402ea112b63ce009dec9b854ecb9abe280252cd55f04371128666` |
| `model_augmentation/fit_systems/closed_loop.py` | `b4d8842a144c6b0ee5cdddfe58efff65383ec9ef4b700ae2723622db4874a883` |
| `model_augmentation/fit_systems/continuous_augmentation.py` | `066b70d79cedc1905037c6b11efeffbc5ee888708c981e6a3253a9ba3e3833d5` |
| `model_augmentation/fit_systems/gyorok_orth_projection.py` | `4399dc66526416b1c2f14ab4aa8471b1e2bea5b18c2c5e66212db3935ad2c2b5` |
| `model_augmentation/fit_systems/interconnect.py` | `310e0dee2e1d7cf7e91cc5da81ad570155621c75db3af6996df79a01a5a75138` |
| `model_augmentation/fit_systems/lbfgs_polish.py` | `bf22b78fbfbd4f93d1d99f48c0a6756bf11603f8ecd60c702a6b72d026faabbf` |
| `model_augmentation/fit_systems/multiple_shooting.py` | `e218ab0665e247fec07c423becb132ca3cc27d554e0232045e677e830702ff00` |
| `model_augmentation/fit_systems/obc.py` | `b6b12b0a77362c896960fda9aaae6e6ea7962f3f9777f73adf0c5ca4eaecb1cb` |
| `model_augmentation/fit_systems/obc_projection.py` | `27035f6b8fba8e122fbd7c9973ac7ef77c373dba3d3a1ddcec28a4146f0ba111` |
| `model_augmentation/fit_systems/orth_projection.py` | `92cfb224c759fb355b7b1c9e99c0c57ee8a36fcc4dbdcd852ef61d5f30213197` |
| `model_augmentation/fit_systems/pre_encoder.py` | `86240148327e70801a4838de6524ee90255a14437d57a8d763803b54851a1d82` |
| `model_augmentation/fit_systems/ps2_opening.py` | `6297aae61b98e9d88b6bb49b50e2cdcaba6ff7277ab4d1604805844b82f9a9bc` |
| `model_augmentation/fit_systems/white_box_models.py` | `561c053bbd537a57a9a708d4c764471737f0f4260d8639df7b1879abad312c09` |
| `model_augmentation/systems/__init__.py` | `314e585003425dad69100cd60789f460d8e25cad2b8c1a9f8472a1d7f896519e` |
| `model_augmentation/systems/cascaded_tanks.py` | `66f60076d15a4e7422e92bc7712cecb5fbcf41a1e50809d83676a630a07115a1` |
| `model_augmentation/systems/gantry_linearization.py` | `fdc419ebad7a00ac1d937afd9881a256292bb68e402648278cf08d92d3c5bab0` |
| `model_augmentation/systems/gantry_ss.py` | `40bb3f3c0d936c9ef3a3d52d5a1c1e9ae7816c3e79d0d446321b709ebc0c463b` |
| `model_augmentation/systems/mass_spring_damper.py` | `25f0f4b61bda445f241c84767139feaef3ce3307a2da0bb0a4c9d73ffd8c7e01` |
| `model_augmentation/utils/__init__.py` | `9617c9dbd9c89786efecec8a2725bbc6f77da00c2820c86896812ebc8a9af49d` |
| `model_augmentation/utils/deepSI_corrections.py` | `f56f194b32ebd78faf32cb0f9918fdb8fce2e792b17405030977cfce12408d8e` |
| `model_augmentation/utils/torch_nets.py` | `945b1dfca49666f94276d4e873f0360ba165aecec8361473b7ff41efd3ed775c` |
| `model_augmentation/utils/utils.py` | `0885d55f26c7fbea6f6281f1c70c57335bc2f4b263b93e73150834d7eeada0bb` |
| `Thesis-writeup/Writing/sections/02_system_baseline.tex` | `268fad55784e49597c399799bf02b385fc5373e184513a1ad957a657f1b1659d` |
| `Thesis-writeup/Writing/sections/03_augmentation.tex` | `d0cfe9dea6f356236b8e45508b4136828cf8f97607b2f382262be3e1e86bf5a2` |
| `Thesis-writeup/Writing/sections/04_obc.tex` | `e6b68be11dffc7781207f6e7450abd92602224479073080fbc2fb3ad625348e2` |
| `Thesis-writeup/Writing/sections/05_setup.tex` | `3058d62633454dd07d7191fba6cc0d3a13ff05b51e432b9c3f39a9ed00d03ed6` |
| `Thesis-writeup/Writing/sections/99_appendix.tex` | `c992d6295a3861cc1c5d66e52ae829378212b1d646240054829be4dd2755c937` |
| `Thesis-writeup/Code/Evaluation/eval_blackbox.py` | `88c226ca5ba50e7b7f1b3e0d57439e4deff9ab69c6d48f2e04b6dbedee95727c` |
| `Thesis-writeup/Code/Evaluation/eval_common.py` | `12b19cc1c757717fc998e63c18d5519cc944ce80c22d69217ac7085805d48a4a` |
| `Thesis-writeup/Code/Evaluation/eval_frf.py` | `472102c066fa17597a62bd8e0461ce95b60c1741c941df50d324834390b23d10` |
| `Thesis-writeup/Code/Evaluation/eval_greybox.py` | `5747dd6dc09c0d2c0d9bf0f4ca0d197fe1ab8ee775e0a44713f988f125ccc099` |
| `Thesis-writeup/Code/Evaluation/eval_negation.py` | `8f074c0e6e63ceee3895724fe058155222c605f1528669a027cf930c536c396b` |
| `Thesis-writeup/Code/Evaluation/eval_overlap.py` | `cc8fac33804cf1404441adb108207e72631eb845eaead243874d074529da90e5` |
| `Thesis-writeup/Code/Evaluation/make_results.py` | `da10b571d4474e00a17ecbf2bc1c6b4019a6ab2379841f7cfdfa782a736ad42a` |
| `Thesis-writeup/Code/Server/tables/runs.tsv` | `95340c53e2402e0d9791122a166e6c0957ce443dd86389fcf1cd875236cc8f92` |
| `Thesis-writeup/Code/Server/greybox/run_thesis.sbatch` | `c4cbc5cf1f3c468b705fdd5ad9c9b231062957a8a83fc6d79a6dc276ab71fcd2` |

Final document checks:647/647 source definitions classified exactly once;70/70 dataclass fields tabulated;52/52 active equation labels mapped;source fingerprints unchanged at final check;no forbidden dash sequences. These are document/source checks,not numerical tests.
