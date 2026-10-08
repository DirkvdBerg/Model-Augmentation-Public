# Source map for thesis sections

Purpose: tell an agent writing `Thesis-writeup/Writing/sections/NN_*.tex` where the FINAL thesis method lives in code, decisions and literature.
It separates the thesis campaign (D-222) from the many earlier experiment paths that still sit in the same files.
Traced 2026-10-06 against branch `Augmentation`; the thesis method is the `THESIS_ARM` path, nothing else.

**Rule: verify a pointer still exists (Grep the function or block name) before relying on it.** Function names are stable; line numbers are not and are omitted on purpose.

## 1. Effective thesis configuration

**Precedence (later wins):** `RunConfig` dataclass defaults (`scripts/gantry/gantry_dynamic/config.py`) < base `CFG = RunConfig(...)` (entry file `scripts/gantry/gantry_interconnect_dynamic.py`) < the earlier env blocks (`DATASET_AB_*`, `BURN_IN_AB`, `LBFGS_AB_FREEZE`; inert when unset) < `_THESIS_FIXED` plus the per-run arguments in the same `replace(...)` call < `THESIS_NITS`, `THESIS_SMOKE`, `THESIS_DEVICE`, `THESIS_PS2`. `THESIS_ARM` together with `OBC_ARM` raises. A thesis run sets only the `THESIS_*` variables (`scripts/gantry/thesis-results/run_thesis.sbatch` reads them from `runs.tsv`).

Source column: **R** = RunConfig default, **B** = base CFG, **F** = `_THESIS_FIXED`, **A** = per-run argument in the thesis block.

| Field | Aug-OBC (`THESIS_ARM=obc`) | Aug-U (`THESIS_ARM=u`) | Src | D-number |
|-|-|-|-|-|
| Data mode | `thesis_taf_noisy` or `thesis_taf_noisefree` (`THESIS_NOISE`), folders `Thesis-writeup/Data/Coulomb-tanh-and-MSD[-noise-free]` | same | A | D-221, D-230 |
| Splits | MANIFEST.csv blocks: training 18, validation 6, test-I + test-E 12 | same | code (`data.py::record_files`) | D-221 |
| Rates | 20 kHz records, model at 4 kHz (`fs_new=4000`), `up_sample=1` (one RK4 step per 0.25 ms) | same | B | D-166 |
| y decimation | noisy: `r[::5] + BW8(y-r)[::5]`, zero-phase Butterworth 8 at 1.6 kHz; noise-free: point sampling; 5 ms trimmed at both record ends in both | same | module consts in config.py | D-232 |
| u decimation | block mean over each 4 kHz hold interval of `u_total` (stage forces) | same | code (`data.py::_resample_u`) | D-087 |
| Physical block | `Reduced_Gantry_State_Block`, LPV (`Y_op=None`, Y = state q3 each RK4 stage) | same | F (`joint_estimation=True`, `physics_parameterization='reduced'`) | D-190, D-191, D-229 |
| Start | detuned: factors `1 + 0.10 s_j`, signs from `default_rng(seed)`; correct: `combo_init_detune=None` (`THESIS_START`) | same | A | D-191, D-222 |
| Parameter prior | off (`param_prior=False`) | off | F | D-193 |
| Soft orth penalty | off (`orth=False`) | off | F | D-111 superseded |
| OBC | `obc=True`, `obc_space='tangent'`, `obc_refresh='epoch'`, `obc_rank_rtol=None`, `obc_ref_chunk=16384` | `obc=False` | A, F (space), R (rest) | D-192, D-194, D-195, D-198 |
| Added states n_a | 0, 2, 4 or 8 (`THESIS_NA`; main results 2) | same | A | D-222, TRAINING-DESIGN.md |
| Routing | all `6 + n_a` rows (`ann_route_ix=range(6+n_a)`) | same | A | D-103, D-222 |
| ANN | `Static_ANN_Block`, input `[x (6+n_a), u (3)]`, 16 x 2 tanh, `zero_init_feed_forward_nn` (final layer zero) | same | F (size), B (`ann_activation='tanh'`) | D-222, D-233, D-237 |
| Encoder | `linear_encoder_init_aug`: W^b reconstructability map, W^a Kaiming uniform, correction MLP 16 x 2 tanh with zero final layer; trained jointly | same | B (`encoder_init='linear_map'`), F (size) | D-055, D-119, D-233 |
| Encoder lag | `na = nb = 29` for every n_a, `na_right = nb_right = 1` (window y and u over k-29..k) | same | F (`na_nb_override=29`) | D-222 |
| Rollout | closed loop, residual form, K1 from each record's meta, controller state 0 at each window start | same | B (`closed_loop=True`), F | D-140, D-141, D-142, D-221 |
| Window | `nf_seconds=0.1` (nf = 400) default; sweep tier uses 0.2 and 0.4 (`THESIS_NF`) | same | A | D-220 |
| Stride, burn-in | stride 10, burn_in 0 (whole window scored) | same | F | D-178 not used, D-220 |
| Loss | MSE of normalised y over the window, no penalty terms | same | code | D-193, D-222 |
| Normalisation | training records only; x from `P^-T y` and its first difference; u, y mean and std | same | code (`data.py::compute_normalization`) | D-119 |
| Precision | float64 both phases (`use_f64=True`) | same | F | D-171 |
| Adam | lr 1e-5, eps 1e-16, batch 512, 200 epochs (or `THESIS_NITS` updates), validation every 1300 updates | same | F (lr, batch, epochs), B (eps, its_per_val) | D-148, D-189, D-222 |
| L-BFGS polish | 100 iter, history 50, inner 20, 16384 fixed windows, tol_grad 1e-12, **tol_change 0.0**, freeze none, chunked above 0.1 s | same | F (iter, history), B (rest), A (chunk) | D-171, D-174 |
| Selection | best closed-loop free-run sim-RMS [m] on the 6 validation records, then polish kept only if it improves it and no meter regresses | same | code | D-171, D-172, D-198 |
| PS2 opening | off (`THESIS_PS2` default 0); 1 = separate `ps2` tier, not the thesis method | same | A | D-236 |
| Excluded knobs | device cuda, compile reduce-overhead, checkpoint_chunk 0, static_pass_buffers False: runtime only | | B | D-169, D-199 |

## 2. Superseded paths a reader must ignore

Entry file `scripts/gantry/gantry_interconnect_dynamic.py`:
- Base `CFG` values for joint estimation off, 24 x 3 net, `nx_ann=8` with 14 rows, float32, the fixed `combo_init_detune` list: overwritten by the thesis block; never describe them as the method.
- `DATASET_AB_MODE/SEED`: dataset-learnability pair on legacy `augmentation_ma50*` data.
- `BURN_IN_AB` (D-178): burn-in screen; thesis scores the full window (D-220 rules burn-in out).
- `LBFGS_AB_FREEZE` (D-175): encoder-freeze pair. NOT overwritten by the thesis block, so it must be unset.
- `OBC_ARM` with `noproj | obc | obc_affine` (D-192, D-200): the pre-thesis projection pair; the affine eleven-column arm is not in the campaign (`_THESIS_FIXED` pins `obc_space='tangent'`).
- `THESIS_PS2=1` (D-236): a later opening experiment on top of the thesis protocol, not the reported method unless a section says so.
- `ANN_REZERO_GATE` env in `model.py::build_model` (D-130 era gate): must be unset.

Package and framework:
- `model_augmentation/fit_systems/obc_projection.py` (`OBC_Projected_ANN_Block`, `[J, c]`, D-190): replaced by `obc.py` + `gantry_dynamic/obc_gantry.py` (D-192); not imported by the pipeline.
- `orth_projection.py`, `gyorok_orth_projection.py`, `scripts/gantry/common/orth_penalty.py`: Gyorok 2025 soft penalty (`orth=False`).
- `multiple_shooting.py`, `SSE_Interconnect_ParamLoss`: D-127 retired; pipeline builds `SSE_Interconnect_Composed`.
- `augmented_dynamics.py` (AUG_LRU live linear bypass, D-150/D-151/D-158/D-160): not imported; thesis added states start dead (zero final layer, D-233, D-237).
- `Parameterized_Gantry_State_Block` (raw 14 scalars, D-076): only for `physics_parameterization='raw'`.
- D-167 Y-scheduled encoder `W^b(Y)`: not wired; `build_model` calls `gantry_linearize_and_discretize(dt)` with the default `Y_op=0`.
- `lpv_lfr_baseline/`: reference derivation and verification package, not imported by the training path (`gantry_ss.py` says so).
- `gantry_dynamic/obc_diagnostics.py`, `diagnostics.py`, `baselines.py`: reporting only.
- `scripts/gantry/thesis-results/runs.tsv`: tier `karnopp` is by its name the pre-D-230 Karnopp data under the same mode names (not confirmed in a decision; check the job logs); tier `ps2ctl` is retired (D-236); the thesis tiers are `core` and `4` (window sweep).

## 3. Concept-to-code map (thesis path)

Short paths: `GD` = `scripts/gantry/gantry_dynamic/`, `FS` = `model_augmentation/fit_systems/`, `SYS` = `model_augmentation/systems/`.

| Concept (Section) | Where |
|-|-|
| Parameters, M0/M1/M2, K, C, P, output Cd = [P^T 0], Dd = 0 (II) | `SYS/gantry_ss.py` module constants; `build_poly_constants` (alpha, beta, gamma, N0..N2 of M(Y)^-1 = N(Y)/d(Y)); `build_G_matrix_entries` |
| LPV-LFR derivation and checks (II) | `docs/Thesis-documentation/LPV_LFR_Stepwise_Derivation/main.tex`; `lpv_lfr_baseline/core/physics.py::build_M`, `lfr_matrices.py::build_G_matrix`, `lfr_forward.py::lfr_forward`, `lfr_simulate.py::rk4_step`, `lpv_lfr_baseline/scripts/validate_lfr.py` |
| Deployed f_base: denormalise, P on stage force, fnet, Horner N(Y)/d(Y), z = [a; Y a], w = Y z, xdot via A_combined (II, III) | `FS/blocks.py::Gantry_State_Block._deriv_with` |
| RK4 step (III) | `Gantry_State_Block._rk4` (`up_sample` substeps; thesis 1), input held over the step |
| Ten combinations, training coordinates (II, IV) | `FS/blocks.py::Reduced_Gantry_State_Block`: `COMBO_NAMES`, `_combos_from_raw` (on `_Trainable_Gantry_State_Block`), `combinations_from_free` (9 log, m_diff = rho (2/Lb) sqrt(m_total J_eff) tanh(eta0 + s v)), `gauge_section`, `admissibility`; detuning applied in `GD/model.py::build_model` |
| M(Y) hoist per objective | `_Trainable_Gantry_State_Block.pass_mats`, used in `FS/interconnect.py::SSE_Interconnect_Composed.loss` (D-195) |
| Where f_aug enters | `FS/interconnect.py::Interconnect.forward`: xp = phy rows (RK4 result) + ANN rows via `expansion_matrix(route_ix)`, wiring in `GD/model.py::build_model` (`connect_signals`) |
| Routing f_aug / g_aug (III) | ANN outputs 0..5 add to the physical rows (f_aug); outputs 6.. ARE the added states' next value (g_aug; no identity carry, no A/B for x_a, D-180) |
| Output map (III) | `FS/blocks.py::Linear_Output_Block` with `norm.Cd_norm`, physical rows only; `Interconnect.output_only` for the rollout |
| Zero init (III) | `model_augmentation/utils/torch_nets.py::zero_init_feed_forward_nn` (final weight and bias zero) |
| Normalisation (III, V) | `GD/data.py::compute_normalization`: training records, q = P^-T y, velocity by first difference at 4 kHz, mean/std per channel; set on `fit_sys.norm` in `build_model` (`auto_fit_norm=False`) |
| Data loading, decimation, trim (V) | `GD/data.py::load_datasets`, `_record_at_fs_new`, `_decimate_y`, `_resample_u`; `GD/config.py::y_decimation`, `record_trim` |
| Encoder and W^b (III) | `FS/pre_encoder.py::linear_encoder_init_aug` (W^b = [A^n O_n^+ , r_n - A^n O_n^+ T_n], W^a Kaiming, zero-final-layer MLP, D-017/D-055 offset fix); linearisation `SYS/gantry_linearization.py::gantry_linearize_and_discretize` (ZOH, Y_op = 0, NOMINAL gantry_ss parameters); dims `GD/model.py::get_encoder_dims` |
| Controller K1 (III) | `GD/controller.py::build_cfb_at` (ruleOfThumb gains, Tustin), `controller_ss`, `register_stored_controller` (from record meta via `data.py::_register_controller`), `build_controller_bank`, `build_closed_loop` |
| Closed-loop rollout, x^c init, step order (III) | `FS/closed_loop.py::closed_loop_rollout` (x^c = 0 at window start), `_rollout_segment` (y_hat = h(x); e = y - y_hat; u_fb = Cc xc + Dc e; x+ = step(x, u_data + u_fb); xc+ = Ac xc + Bc e), `ControllerBank.step` (normalisation folded in), `ClosedLoopSimulator.__call__` |
| Loss (III) | `FS/interconnect.py::SSE_Interconnect_Composed.loss`: encoder at window start, rollout via `simulate`, `mse_loss` over batch, nf and the 3 channels; `param_loss` returns 0 with the prior off |
| Training loop, polish (V) | `GD/training.py::train_model_with_diagnostics`, `GD/model.py::train_model` (deepSI `fit`), `GD/training.py::run_lbfgs_polish`, `FS/lbfgs_polish.py::lbfgs_polish`, `FixedBatch`, `decide`; meters `GD/training.py::_gantry_meters` |
| OBC reference tuples (IV) | `GD/obc_gantry.py::build_reference_set`: every training sample with a full stencil, q = P^-T y, q_dot by `fourth_order_central_difference`, x_a = 0, recorded u; normalised |
| OBC stack J (IV) | `obc_gantry.py::make_basis_builder` -> `FS/obc.py::build_stacked_sensitivity` (float64 `jacfwd` of `Reduced_Gantry_State_Block.transition_from_free` over the 10 free coordinates, physical routed rows) |
| OBC coefficient (IV) | `FS/obc.py::OBCBasis.from_matrix` (SVD, rank tol max(M,N) eps), `OBCBasis.coefficient` (theta = V S^-1 U^T F, differentiable in F); field `obc_gantry.py::make_reference_field`; once per objective in `OBCLifecycle.coefficient` |
| OBC refresh (IV) | `OBCLifecycle.maybe_refresh` / `refresh` (at each epoch boundary at the detached current free coordinate; rank loss keeps previous basis, two in a row abort) |
| Corrected model and forward sensitivity (IV) | `obc_gantry.py::GantryOBCCorrection.forward` (subtracts E J(vbar, x_b, u) theta, physical routed rows only, zero on x_a rows), `tangent_pair` -> `Reduced_Gantry_State_Block.mats_and_tangent_from_free`, `Gantry_State_Block._rk4_with_tangent` / `_deriv_tangent_with`; subtraction in `Interconnect.forward` |
| OBC wiring and validation sync (IV) | `obc_gantry.py::attach_obc` (from `build_model` when `cfg.obc`); `SSE_Interconnect_Composed._sync_prediction_state_for_validation` (D-198, D-201) |
| Validation, checkpoint selection (V) | `FS/closed_loop.py::ClosedLoopSimulator.validation_error` -> `closed_loop_free_run_rms_batch` (k0 = max(na, nb), record-length-weighted mean of per-record RMS in m); deepSI restores `_best` at end of `fit` (`FS/interconnect.py::SSE_Interconnect.fit`), then OBC basis rebuilt with `refresh_basis=True` |
| Post-run report (VI) | `GD/evaluation.py::evaluate_and_save`, `GD/closed_loop_report.py::closed_loop_table` (D-227) |

## 4. Using `docs/decisions.md` (about 7600 lines)

- Format: `### [D-NNN] Title`, then `**Date**`, `**What**`, `**Why**`, `**Ruled out**`, `**Constrains**`, often `**Outcome**`, `**Amendment**`, `**Correction**`; template near the `## Decision Template` heading.
- Entries are NOT in numeric order: D-218 to D-209 sit at the top, D-188 to D-120 after the template, older entries further down, D-213 to D-237 at the end. Locate by number: `Grep "^### \[D-192\]" docs/decisions.md`.
- Duplicate numbers exist (D-017, D-076, D-148, D-186 each twice): read both and check the date.
- A topic: Grep the headers first (`Grep -i "^### \[D-.*(obc|orthogonal)" docs/decisions.md`), then the body for the number (`Grep "D-192" docs/decisions.md`) to find later entries that amend it.
- Rule: **a later entry supersedes an earlier one on the same choice**, whether or not it says so; explicit `SUPERSEDED` markers exist only on a few (D-212, D-218). Check the code before trusting either.
- Decisions give reasons and plans, not evidence. Evidence is in the folders they name (REPORT.md, problem log run table in `docs/gantry-augmentation-problem-log.md`), and some entries record plans whose outcome is elsewhere or pending.

Final choices per section:

| Section | Defining D-numbers |
|-|-|
| II Baseline | D-017 (Delta(Y) LFR), D-167 (closed-form frozen-Y linearisation only), D-190 and D-191 (ten combinations, reduced block), D-229 (bounded m_diff, admissible M(Y)) |
| III Augmentation | D-222 (configuration), D-103 (X and Y routed), D-180 (added states are Jacobian blocks of one MLP), D-233 and D-237 (init, W^a, encoder deviations), D-140, D-141, D-142 (closed-loop residual form, K1 stepped at 4 kHz, x^c = 0), D-220 (0.1 s window, no burn-in), D-171 (L-BFGS polish), D-148 and D-189 (Adam eps, lr) |
| IV OBC | D-192 (one-step tangent OBC), D-193 (prior off), D-194 (hand-written forward sensitivity), D-195 (per-objective hoist), D-198 and D-201 (validation sync), D-202 and D-203 (Delta* frame and stencil, diagnostics), D-229 |
| V Setup | D-221 (data modes, splits, K1 from meta), D-224, D-226, D-230 (tanh friction truth, regenerated data), D-225 (encoder noise), D-232 (anti-alias, trim), D-166 (4 kHz), D-227 (closed-loop report), D-090 (run table), TRAINING-DESIGN.md |

Superseded, and by what:

| Old | Replaced by |
|-|-|
| D-068 Theta-only routing | D-103, D-222 (all 6 + n_a rows) |
| D-034, D-076 Lambda prior | D-193, D-222 (off) |
| D-111, D-185, D-186 Gyorok 2025 soft penalty | D-190, D-192 (OBC), `orth=False` |
| D-190 `[J, c]` protected space | D-192 tangent `range(J)`; D-200 affine arm never in campaign |
| D-178 burn-in | D-220 (ruled out), `burn_in=0` |
| D-189 24 x 3, n_a 8 | D-222 (16 x 2, n_a per run) |
| D-150, D-151, D-158, D-160 live added-state bypass | D-233, D-237 (zero final layer, dead start) |
| D-155 Xavier W^a | code uses Kaiming; deviation logged in D-233 |
| D-167 W^b(Y) encoder | not used; W^b at Y = 0 |
| D-127 multiple shooting | retired 2026-08-28 |
| D-146 concurrent validation default | `train_model` does not pass it; deepSI default False |
| D-204, D-209 Karnopp friction truth | D-224, D-226 (tanh), D-230 |
| D-212, D-218 noise injection | D-225 (encoder noise) |
| D-221 data folders and noise wording | D-230 folders, D-225 noise |

## 5. Using `docs/references.md` (about 1400 lines) with `Thesis-writeup/Writing/refs.bib`

- Chain: claim -> cite key in a section table of `docs/references.md` (columns Cite key, File, Description) -> local PDF under `literature/` -> verification status, stated in the description as READ IN FULL, READ IN PART, ABSTRACT ONLY or SECOND-HAND, often with the date of re-verification and exact equation or page numbers. Rows without such a marker are unverified summaries: open the PDF.
- Find a key: `Grep "\`<key>\`" docs/references.md`. Core keys: `garcia2013model`, `drenth2025lpvlfr`, `drenth2025thesis`, `toth2010modeling` (II); `hoekstra2025lfr`, `hoekstra2026lfr`, `hoekstra2026encoder`, `beintema2023subnet`, `kessels2025ai` (III); `gyorok2025l4dc`, `gyorok2026obc` (IV).
- `refs.bib` keys must equal `docs/references.md` keys. Disabled entries start `% DISABLED-` and are skipped by BibTeX. Kessels PDF page = thesis page + 26 (`FS/closed_loop.py` header).
- Known gaps (2026-10-06):
  - `gyorok2026obc`: cited about 12 times in `04_obc.tex`, **no entry in refs.bib**. Metadata in references.md: IFAC J. Systems and Control 35:100376, 2026, DOI 10.1016/j.ifacsc.2026.100376, arXiv 2511.01321.
  - `gyorok2025l4dc`: active entry is a **placeholder** ("Source pending verification"); a full DISABLED entry (L4DC 2025, PMLR 283) sits below it.
  - The DISABLED `hoekstra2026lfr` block carries the 2025 EJC metadata of `hoekstra2025lfr`; the active `hoekstra2026lfr` (arXiv 2602.17297) is correct. Do not re-enable the disabled one.
  - `hoekstra2026encoder`: active arXiv entry; a section todo asks to check for a published version.
  - `kessels2025ai` is the PhD thesis; references.md also lists the Nonlinear Dynamics paper. Equation numbers 5.x refer to the thesis.
  - refs.bib carries keys no section cites yet (gautier1990minimum, raue2009profile, and similar): harmless.

## 6. Contribution map (Sections II to IV)

Sources: `01_introduction.tex` contribution bullets and its gap todo; section header WRITING GUIDE and SOURCE MAP lines (only `03_augmentation.tex` has an explicit SOURCE MAP); the code above.

| Section | Adopted | Adapted (difference) | This work |
|-|-|-|-|
| II | Garcia-Herreros EOM, parameters, P transform; LPV-LFR definitions (Drenth, Toth) | CT LPV-LFR with Delta(Y) = Y I6 built from the polynomial M(Y) (Drenth's DT companion paper defines DT pairs) | Gantry LFR realisation and well-posedness over Y; ten identifiable combinations as coordinates (rank 10 of 14) and the bounded m_diff map |
| III | Dynamic-parallel topology f_aug, g_aug, zero-init learning function (Hoekstra 2025 EJC Table 1, 2026 Automatica Eq. 29, Jan's code); SUBNET truncated simulation (Beintema); W^b reconstructability Eqs. 16 to 17 (Hoekstra encoder 2026) | W^b computed analytically from a ZOH linearisation at Y = 0 instead of fitted (paper Eq. 30); W^a Kaiming instead of Xavier; closed-loop rollout of Kessels Eq. 5.12/5.13 in residual form with x^c = 0 (a definition, not his Remark 5.4 reconstruction, per `closed_loop.py`); objective Hoekstra Eq. 22 but scored in closed loop | Gantry specialisation, self-scheduled RK4 baseline with f_aug added after the step, routing to all physical rows, residual-loop implementation, joint estimation in identifiable coordinates |
| IV | OBC coefficient and projection (Gyorok 2026 Eqs. 8, 10; Lemma 3) | Linear-in-parameters IO regressor replaced by the Jacobian of the nonlinear RK4 state transition at vbar, refreshed per epoch (linearisation precedent: Gyorok 2025 Sec. 4); data-derived state tuples (P^-T y plus fourth-order difference, x_a = 0) since no full state is measured | State-level, LPV, MIMO extension (Gyorok 2026 names state space as future work); per-objective differentiable coefficient through a rollout; added states left unprojected; the true-parameter recovery condition |

Gap statement (intro todo): Hoekstra 2025/2026 have no LPV scheduling and no closed-loop training; Kessels trains through the closed loop but without LFR, LPV, parameter estimation or orthogonality, so closed-loop training alone is not novel; Gyorok OBC is input-output only. "Validated" must be scoped to simulation (real Telica data are out of the main results).

## 7. Known traps

1. W^b is linearised at the NOMINAL `gantry_ss` parameters and Y = 0 even in detuned runs, so encoder and physical block start inconsistent (`GD/model.py::build_model`, `SYS/gantry_linearization.py`).
2. Notation: Section III says u is the actuator force (stage, `u_total` in code), Section II uses u for P u_act; Section IV writes u_i = P u_act,i. The code feeds stage forces and applies P inside `Gantry_State_Block._deriv_with`.
3. Base CFG is not the thesis run: joint estimation off, 24 x 3, n_a 8, float32, and `param_prior` default True in RunConfig. Only the thesis block makes it the method.
4. `lbfgs_tol_change=0.0` in the thesis run is inherited from the base CFG's encoder-freeze diagnostic setting, not chosen for the thesis (TRAINING-DESIGN.md does not mention it).
5. `its_per_val=1300` is commented as "every 10 epochs" for 14 records; with 18 records it is about 7.8 epochs (`GD/config.py` RunConfig.its_per_val comment, entry CFG).
6. L-BFGS acceptance meter `combo_err` compares against the TRUE parameters (`GD/training.py::_gantry_meters`): an oracle gate inside the training procedure.
7. Noisy mode: normalisation velocities and OBC reference velocities are differences of the anti-alias filtered noisy y; the OBC guard checks only `cfg.snr` (`GD/obc_gantry.py::build_reference_set`). TRAINING-DESIGN.md section 3 still argues from force noise (pre D-225).
8. D-142 calls x^c = 0 Kessels' Remark 5.4; `FS/closed_loop.py::closed_loop_rollout` says it is NOT. Section III has a VERIFY todo on this.
9. `GD/model.py::build_model` comment says D-068 stiff routing to K>0 rows; the thesis routes all rows. Stale comment.
10. Encoder lag 29 for every n_a overrides Jan's `2(6 + n_a) + 1` (`GD/config.py::RunConfig.na_nb`); D-222 lists n_a 0, 2, 8 while the code also accepts 4 (`runs.tsv` has a 4).
11. Mode names `thesis_taf_*` pointed to the Karnopp folders before D-230; older run logs under the same mode used other data (`GD/data.py::THESIS_DATASETS`).
12. `gantry_ss._D = torch.float32`: nominal constants are float32-rounded even in float64 runs (noted in D-167).
13. Env vars outside the THESIS set still act: `LBFGS_AB_FREEZE`, `ANN_REZERO_GATE`, `GANTRY_KX_ART` / `GANTRY_KY_ART` (`SYS/gantry_ss.py`), `RESUME_CHECKPOINT`.
14. The loss mean also averages over the three output channels (`mse_loss`), unlike the 1/(|T| n_f) of Section III's V; and Adam sees minibatches of 512 windows, not all of T.
15. During the polish the OBC basis is not refreshed: the epoch counter does not advance after the post-restore refresh and `FS/lbfgs_polish.py` has no OBC logic of its own (read from the code, not measured).
