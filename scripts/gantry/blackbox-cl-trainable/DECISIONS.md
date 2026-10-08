# blackbox-cl-trainable: decisions

Every entry is written BEFORE the code or run it governs; thresholds are not moved after a number
is seen. Source: `tasks/handoffs/2026-09-29-blackbox-closed-loop-trainable.md`.

### BB2-001 BLA check first (user, 2026-09-29)
**Date**: 2026-09-29
- User: "lets check if we can get a proper BLA from the current data. if not we will skip that
  part." Run unattended; the answer is the deliverable ("if it cant work for this data then we
  have our answer").
- User: "since its in simulation we know the real model, use that to your advantage for comparing
  your results but DONT use the oracle / real data to cheat with the black box model".
  So the truth enters ONLY as the scoring reference of the BLA check; nothing from it reaches any
  black-box code, initialisation or setting. The literature phase is not started (scope not
  agreed yet).
- Code in `bla/`; imports repo modules read-only; outputs in `bla/outputs/` (small `.npz`/`.json`).

### BB2-002 BLA check: what is estimated, from what
**Date**: 2026-09-29
- **Data**: the 18 training records of `Thesis-writeup/Data/Coulomb-tanh-and-MSD[-noise-free]/`
  (MANIFEST block `training`), native 20 kHz, `loadmat(mat_dtype=True)`. Validation, test and TF
  records are not read.
- **Target**: the plant BLA G: stage force `u_total` -> stage position q, 3 x 3 (what a black
  box's linear part must be). On the noisy set the measured output is y = q + v.
- **Loop**: u = u_fb + f, u_fb = Cfb (r - y), Cfb = K1 (`controller.build_cfb_at(0, 5e-5)`, known
  to both arms). The conventions are checked on the data before any estimate (C0 below).
- **Band B_tr (106 to 297 Hz), multisine lines** (14 multisine records, 2 s period, 0.5 Hz grid,
  383 lines): per 2 s period DFT at the excited lines; period 1 dropped (transient), periods 2 to 6
  averaged. Estimator: **indirect, f as instrument**, G = S_yf S_uf^-1, # THEORY: Wellstead 1981
  (Automatica 17), Pintelon and Schoukens 2012 ch. 2 (closed-loop FRF from an external reference;
  consistent when f is independent of v). Also the **direct** estimate G = S_yu S_uu^-1 on the same lines, to show the
  closed-loop noise bias (expected on the noisy set only).
  - per standstill record (TR-S1 to S5, one Y each): a single experiment carries 3 inputs on the
    same lines, so S_yf and S_uf are estimated by the local polynomial method over neighbouring
    lines, degree 2, window 2n+1 = 15 lines (# THEORY: Pintelon, Schoukens et al. 2010 LPM,
    MSSP 24; unknowns 3 (R+1) = 9 < 15 lines; # HEURISTIC: n = 7 is 7 Hz, under the 9 Hz
    anti-resonance -3 dB width, degree 2 follows its curvature).
  - pooled (all 14 multisine records, Y varies): per line least squares over the records' F
    vectors plus the same LPM window: the LTI BLA over the training distribution, the object a
    black-box linear init would use.
- **Low band (below 106 Hz)**: no multisine; the only exogenous input is the reference, entering
  as the force w_r = Cfb r. Coverage per 1 Hz bin from 1 to 106 Hz: kappa = sigma_min / sigma_max
  of the exogenous force spectral matrix sum_records W W^H (Hann-windowed 1 s segments, 50 %
  overlap, 4 records' multisine-free TR-T included). Where kappa passes, G is estimated with the
  same indirect formula, r as instrument (S_yw S_uw^-1).
- **Scoring reference (truth, scoring only)**: frictionless T_AF, stage u -> q, ZOH at 20 kHz:
  `excitation-closed-loop/common.py` `truth_mck(Y)` + `frf_zoh` (4 dof incl. the absorber); its
  parameters asserted equal to each record's `meta.truth` (else the reference is rebuilt from
  meta). Per-Y estimates against the truth at that Y; pooled against the truth at Y = 0 and the
  envelope over the training Y range [-0.30, 0.30].

### BB2-003 BLA check: pre-registered criteria and prediction
**Date**: 2026-09-29
- **Error per line**: e(f) = sigma_max(G_hat - G_true) / sigma_max(G_true).
- **B_tr PASS** (per standstill Y, and pooled against the Y envelope): median e <= 0.10 and
  90th percentile e <= 0.25 over the 383 lines, separately for noise-free and noisy.
  # HEURISTIC: 0.10 is the top of the friction-induced shift of the true BLA away from the
  frictionless system measured in session 1 (2 to 11 %, DATA-DESIGN s. B_tr, B11), which is a
  property of the real BLA, not an estimation error; 0.25 at the 90th percentile leaves room for
  the lines at the anti-resonance and the cross-entry notch.
- **Noise effect**: e_noise(f) = sigma_max(G_noisy - G_noisefree) / sigma_max(G_noisefree),
  reported for indirect and direct; no threshold (it is what the check reports).
- **Low band**: identifiable in a bin iff kappa >= 1e-2 (# HEURISTIC: the weakest input direction
  gets at least 1 % of the strongest's amplitude; below that the 3 x 3 inverse amplifies any
  error by more than 100). Accurate where identifiable iff median e <= 0.10 over those bins.
- **Verdict**: "a proper BLA can be obtained from this data" iff B_tr PASS on both versions AND
  the low band is identifiable and accurate from 20 to 106 Hz (# HEURISTIC: K1's crossovers lie at
  66 to 108 Hz, DATA-DESIGN 5.11, so the loop's stability with a model is decided there; a BLA
  that misses that band cannot give a start the controller is known to stabilise). Otherwise the
  answer is "band-limited BLA only" or "no BLA", with the failing part named.
- **Prediction** (before running): B_tr PASS noise-free; noisy indirect PASS, noisy direct biased
  near the loop's peak (264 Hz) where the noise is shaped; low band NOT identifiable as 3 x 3
  (the anti-yaw channel carries almost no reference: only TR-L2 has 1 mm X_anti), so the verdict
  is "band-limited BLA only".

### BB2-004 bla1 outcome read, and the low-band control run bla2
**Date**: 2026-09-29
- bla1 (RUNS.md): B_tr PASS on both versions; low band identifiable by kappa (prediction wrong)
  but inaccurate: median e 0.125 (20 to 40 Hz), 0.18 (40 to 60), 0.48 (60 to 80), 1.13 (80 to
  106), so the pre-registered verdict is "no proper BLA". Noise changes the BLA by ~1e-5
  (encoder noise ~10 nm rms against a 13 to 28 um response), so noisy = noise-free here.
- Lesson on the criterion: kappa is scale-free, so it passes bins where the reference carries
  almost no power; it tests the direction spread, not the excitation level. Kept as reported,
  not re-used as a verdict item.
- **Why bla2**: before calling the low band a property of the data, rule out the estimator (Hann
  1 s Welch on transient-rich, non-periodic moves leaks the 1 to 5 Hz move power, 1e5 above the
  80 to 106 Hz content, into higher bins).
- **bla2 method**: (a) LPM on the full-record DFT (12 s, 1/12 Hz), pooled over all 18 training
  records, inputs W = Cfb R + F, outputs [Y; U], degree 2, window 2n+1 = 15 bins, a transient
  polynomial per record (# THEORY: Pintelon, Schoukens et al. 2010, LPM for arbitrary excitation;
  the transient term absorbs leakage); G = S_yw S_uw^-1; scored 1 to 106 Hz against the truth Y
  envelope. (b) CONTROL: the same estimator on a friction-free, fixed-Y (Y = 0) LTI simulation of
  the truth in the K1 loop at 20 kHz, driven by each record's own r - r(0) and f (truth used as a
  data generator for method validation only, user 2026-09-29; nothing reaches the black box).
- **Criteria** (fixed now): accurate band iff median e <= 0.10 over 20 to 106 Hz (BB2-003).
  Reading: control PASS and data FAIL -> the data's BLA is not the linear plant in that band
  (friction / LPV), no proper BLA from this data; control FAIL -> the training references do not
  excite that band enough; both PASS -> a proper BLA exists with LPM (verdict flips).

### BB2-005 BLA verdict: no proper BLA from the training data; the BLA route is dropped
**Date**: 2026-09-29
- bla2: the estimator on friction-free LTI data with the same references recovers the truth to
  1e-4 (control PASS), so excitation and method are adequate; on the real records it misses the
  linear plant by 14 to 26 % over 20 to 106 Hz and 63 % below 10 Hz, identically noisy and
  noise-free (data FAIL). By the BB2-004 reading the gap is the system (tanh Coulomb friction,
  Y motion), not the estimator or the noise.
- Verdict per BB2-003: B_tr (106 to 297 Hz) BLA is good (median 6 to 8 %); the band that decides
  K1 loop stability (66 to 108 Hz crossovers) is not. So the fallback BLA arm of the handoff
  section 4 is dropped (user 2026-09-29: "if not we will skip that part").
- Not tested (out of the question asked): whether a start built from this 15 to 25 % off BLA would
  still be stabilised by K1.

### BB2-006 The model: Jan's SUBNET inside a K1-derived ICI wrapper, stable S (user "go", 2026-09-29)
**Date**: 2026-09-29
- User direction: build the black box that trains in closed loop, stop and report if it fails;
  a plant model is required (controller transfer, R7 / E5). Defaults agreed in chat: criterion
  BB2-008, local CPU only, no test data, K-tilde from K1 alone, one critic agent at the end.
- **Structure** (# THEORY: Boroujeni et al. CDC 2025, Eq. 4 and Theorem 1, the ICI model
  y = S(u + K y) with S strictly causal and L_p-stable; sign as in our residual form):
  plant model G_hat: u -> y_hat with y_hat = S(w), w = u + Kt y_hat, all in the pipeline's
  normalised coordinates. Kt ("K-tilde") = K1 with its integrator s replaced by s + w_leak, same
  gains, Tustin at the model rate, folded with the pipeline's (ystd, std_u) exactly as the
  ControllerBank folds K1 (closed_loop.py lines 139 to 141).
- **Why the leak** (LITERATURE.md SQ2): Theorem 1 needs K incrementally stable; K1 has a pole at
  z = 1. With Kt stable, ICI holds for the pair (G, Kt) if Kt stabilises the plant (checked in
  Phase B against the truth, as a check only). Under K1 the training loop is then S with the
  feedback dK = K1 - Kt, which is nonzero only below about w_leak: stable in practice, not
  guaranteed (Phase B measures it).
- **w_leak = 2 pi x 1.67 Hz**: # HEURISTIC: one decade below K1's lowest zero (W/6 = 16.7 Hz,
  controller.py cnorm), so Kt equals K1 within 0.6 deg of phase at the 66 to 108 Hz crossovers;
  from K1's own design constants, no plant data.
- **S, stable for every parameter value** (# THEORY: Orvieto et al. ICML 2023, LRU, Appendix A
  code p17): a cascade of L blocks; block b has m complex modes held as real 2 x 2 blocks,
  lambda = exp(-exp(nu_log) + i exp(theta_log)) (so |lambda| < 1 always), input matrix scaled
  by gamma = sqrt(1 - |lambda|^2) (their normalisation), output z = C s, block output
  o = z + MLP(z) (tanh). ŷ = W_out o_L + b_out. No direct term anywhere, so S is strictly causal
  (y_hat[k] depends on states only) and L_p-stable (stable linear recurrences, Lipschitz static
  maps, cascade). The nonlinearity sits between blocks, never inside a state feedback, which is
  what keeps the guarantee (a nonlinear f(x, u) in the recurrence would lose it).
- **Input scaling of S**: w is divided by a fixed per-channel std sigma_w computed once from the
  18 training records as std(u_data + Kt y_data) in normalised units (# HEURISTIC: the folded
  Kt feedthrough is O(1e3), so w is O(1e3) where u is O(1); same kind of data-only normalisation
  as the pipeline's own).
- **Initial state**: the encoder is Jan's (deepSI `default_encoder_net`, 2 x 16 tanh, `.reshape`
  fix as BB-003), na = nb = 29, na_right = nb_right = 1 (the grey box's windows). It outputs the
  S states plus a correction to the Kt state; the Kt state starts at its steady state for the
  last measured output, x_Kt0 = (I - A_Kt)^-1 B_Kt y[k0] + correction (# HEURISTIC: standstill
  consistency; the correction learns the rest).
- **Init, no plant information** (# HEURISTIC, rate and window only): ring init r in
  [0.9, 0.999] (decay 2.5 ms to 0.25 s at 4 kHz, i.e. from 10 samples to 2.5 windows), phase in
  [0, pi] (whole band to Nyquist); B, C Glorot as the LRU listing; MLPs PyTorch default; W_out
  scaled by 0.1 so the start is small but not zero. Defaults L = 2, m = 8 (32 S states), MLP
  width 16, block width 8. State dimension is a search hyperparameter, not the true order.
- **Training**: the grey box's carrier, as BB-003: `SSE_Interconnect_Composed` with this `hfn`
  and encoder, `build_closed_loop` (residual form, K1, xc = 0 per window), `model.train_model`
  (deepSI fit, Adam), data and norm from `load_datasets` / `compute_normalization` in the thesis
  modes, nf 0.1 s, stride 10, burn_in 0, float64 (shared settings). Adam lr 1e-3, eps 1e-8
  (Jan / deepSI default), not the grey box's 1e-5 / 1e-16.
- **Controller transfer**: under K2 the same G_hat is closed with K2 by the same simulator; the
  wiring is checked on training data only (G_hat with K1 in `build_closed_loop` vs an explicit
  recomputation), E5 untouched.
- Rejected: plain Jan SUBNET under K1 (G4 fragility), REN / contraction as S (damps the 264 Hz
  mode, SQ3), open-loop training (user: unfair), closed-loop map without plant (no K2 test).

### BB2-007 Phase B diagnosis: prediction and checks (before running)
**Date**: 2026-09-29
- B1 folded K1 feedthrough max |D'| in the thesis norm. Prediction: O(1e3), as the old 5.9e3.
- B2 plain Jan SUBNET (nx 8 not used: nx = 32 to match S, random seed) closed with K1 on the 6
  validation records: prediction UNSTABLE (non-finite or > 10 max|y| by BB-007's rule).
- B3 the new model at init, seeds 1 to 3: prediction STABLE on 6/6 validation records.
- B4 robustness: 10 random-sign perturbations of all parameters at delta 1e-3 and 1e-2 (Jan's
  step size and 10x): prediction 0 unstable for the new model; the plain SUBNET is unstable at
  init already (B2).
- B5 Kt: its loop with the frictionless truth at Y in {-0.3, 0, 0.3} is stable (truth as a CHECK
  of a design assumption, user 2026-09-29; nothing enters the model) and |Kt - K1| / |K1| < 1 %
  over 50 to 300 Hz. Prediction: both hold.
- B6 wiring: y_hat of G_hat stepped by the pipeline's simulator equals a direct recomputation of
  y = S(u_cl + Kt y) with the same states, to 1e-10 (float64).
- Stop rule (user): if B3 or B4 fails, report with the mechanism instead of continuing.

### BB2-008 Phase D pre-registered trainability criterion
**Date**: 2026-09-29
- Metric: closed-loop free-run servo-error RMS on the 6 validation records, per stage axis, in
  metres, from `closed_loop_free_run_rms_batch` (the pipeline's selection scorer), at every
  validation.
- PASS iff for every one of 3 seeds (1, 2, 3) on BOTH versions (noisy, noise-free): (a) every
  validation finite and stable (BB-007 rule); (b) at the best validation, each axis' RMS is at
  most 1/3 of its epoch-0 value (# HEURISTIC: the factor agreed with the user; per axis, so fitting one channel
  or an offset alone does not pass); (c) the loss
  curve decreases (last-10-update mean < first-10-update mean).
- Local budget per run: fixed number of updates N_upd and validations every N_upd / 6, set in
  RUNS.md before the first run from the measured update time (target <= 15 min per run).
- Reported, not criteria: the FP baseline (grey box epoch 0) on the same validation set for
  context; wall time per update and per validation (R8); every configuration tried.
- If the local budget is too short to show the drop but the curves are finite and decreasing,
  the verdict is "trains, not converged locally" and the server runner is prepared with the same
  criterion.

### BB2-009 Local CPU deviations (measured constraint, 2026-09-29)
**Date**: 2026-09-29
- Machine at launch: 1.6 GB free RAM of 15.8 GB, C: 6.3 GB free, 12 logical cores. At float64 and
  stride 10 the window arrays alone are ~1.6 GB (BB-011 arithmetic x2), which would trip the
  1.0 GB watchdog.
- Local runs only: stride 100 (8.5k windows, ~0.16 GB) and batch 64. Same windows' content,
  same objective, same validation; fewer and smaller updates. Server runners keep stride 10,
  batch 512 (shared settings). Every local number is labelled with these two values.
- Phase B uses the noisy version only (the noise-free twin has identical dynamics and controller;
  the encoder noise does not enter B1 to B6). Phase D runs both.
- Watchdog: `tools/watchdog.ps1` (copy of telica-real), called with RamKillGB 1.0 (user),
  RamAlertGB 1.5, DiskKillGB 0.5, DiskAlertGB 1.0.
- B6 made concrete: the Kt folding is checked against the pipeline's own ControllerBank built
  from the same physical Kt (1e-12), and the trajectory the simulator produces is re-stepped
  outside the simulator with that ControllerBank as the Kt path (1e-10); strict causality by
  perturbing u[k] and checking y[k] unchanged.

### BB2-010 Launching under a RAM shortage (measured, 2026-09-29)
**Date**: 2026-09-29
- pb1 was killed by the user's 1.0 GB emergency watchdog 7.7 s after launch, during Python start
  (available 1.10 -> 0.69 GB; tree 422 MB); no computation ran. Machine: 1.1 GB available, 37.0 of
  39.4 GB committed; 22 `claude` processes hold 5.3 GB, VS Code 3.1 GB (other sessions; not
  touched).
- So each job waits until available RAM >= 2.5 GB before launching (`tools/run_wd.sh`, poll 30 s,
  gives up after 3 h). This is not the conservative launch gate the user ruled out: a launch below
  this level is killed by the user's own emergency rule before it computes anything (measured
  above). The kill level stays 1.0 GB. Python is called directly (no `conda run` wrapper).

### BB2-011 Watchdog off for the local runs (user, 2026-09-29)
**Date**: 2026-09-29
- User: "can you try to run it anyway, ignore the watch dof" (watchdog), at 1.7 GB available.
- Local jobs now run without the watchdog and without the RAM wait of BB2-010, one at a time,
  env python called directly. Everything else unchanged.

### BB2-012 Phase B verdict: FAIL at B3; mechanism located (stop per user rule)
**Date**: 2026-09-29
- B3 fails: the model is not stable under K1 at its random start.
- Mechanism (pbm): with Kt as controller the loop reduces to S and is stable (0.9974, 3/3 seeds),
  so the guaranteed part works; with K1 real poles sit at z = 1.00002 to 1.00007 (angle 0): a
  slow drift through the leftover integrator dK = K1 - Kt (nonzero below ~1.7 Hz) closed over the
  random DC gain of S. Nothing else fails: Kt stabilises the truth (B5), wiring exact (B6).
- Stopped here per user; the fix (a DC path of S fixed from Kt alone) is proposed, not built.

### BB2-013 Design change: Jan's plain SUBNET, closed-loop, window curriculum (user, 2026-09-29)
**Date**: 2026-09-29
- User: stay as close as possible to Jan Hoekstra's simple black box, no structure that needs
  justifying, no monkey patching; "yes please go ahead" on the window-curriculum option
  (LITERATURE.md, second run). The ICI / stable-S design (BB2-006) is dropped, kept here as
  tried and failed (BB2-012).
- **Model**: Jan's (deepSI 0.3.29 `hf_net_default` with `default_state_net` / `default_output_net`,
  f and h 2 x 8 tanh, encoder `default_encoder_net` 2 x 16), unchanged except the interface the
  grey box's carrier asks for (`output_only = h(x)`, `connected_blocks = ()`, no-op
  `init_model`) and `.reshape` for `.view` (BB-003). In `bbcl/jan_model.py`, not imported from
  another folder.
- **nx = 16**: # HEURISTIC: Jan's rule nx = 2 dof uses the plant's degrees of freedom (plant
  information, handoff section 4), so it is not used; 16 is a generous default, the search covers
  6 to 32.
- **Loop**: the grey box's closed loop (K1, residual form, xc = 0 per window), carrier
  `SSE_Interconnect_Composed`, `model.train_model`. Adam lr 1e-3, eps 1e-8 (Jan).
- **Window curriculum** (# THEORY: Decuyper, Tiels, Noel, Schoukens, IFAC-PapersOnLine 2020,
  unconstrained multiple shooting for unstable initialisation, number of segments reduced during
  training; Beintema et al. L4DC 2021 Sec. 2.3, short windows stabilise SUBNET training;
  arXiv 2608.05777 curriculum plus multiple shooting): nf stages 10, 25, 50, 100, 200, 400 samples
  (2.5 ms to 0.1 s; # HEURISTIC: roughly doubling), each stage one `train_model(..., nf=stage)`
  call with a fixed update count. fit() ends every call by reloading its best checkpoint, so each
  stage continues from the best model so far (pipeline behaviour, not changed). The final stage is
  the shared setting nf = 400.
- **Logging without patching**: per-axis RMS through the declared `validation_probes` hook,
  which calls the pipeline's `closed_loop_free_run_rms_batch` itself (a second free run per
  validation, cost reported); losses from fit's own `Loss_train` / `Loss_val` arrays.
- Local deviations as BB2-009 (stride 100, batch 64); watchdog off (BB2-011).

### BB2-014 Pre-registered trainability criterion for BB2-013 (replaces BB2-008 (b), (c))
**Date**: 2026-09-29
- Why the change: the untrained model is unstable in closed loop (B2), so the epoch-0 RMS of
  BB2-008 (b) is not finite and "1/3 of it" is empty. And (c) compared per-update losses, which
  are not comparable across window lengths.
- Metric unchanged: per-axis closed-loop free-run RMS [m] on the 6 validation records.
- PASS iff for every one of 3 seeds (1, 2, 3) on both versions:
  (a) the final stage (nf = 400) has no NaN training loss, and its best validation is finite and
      stable (BB-007 rule); earlier stages may have unstable free runs (recorded, not a failure:
      that is the curriculum's premise);
  (b) at the final model, each axis' RMS <= 1/3 of the mean-model RMS on the same records, the
      mean model being y_hat = the training-record mean y0 (# HEURISTIC: data-only reference that
      any trained model must clearly beat; replaces the non-finite epoch-0 value);
  (c) within the final stage, fit's Loss_train at its last validation < at its first.
- Reported, not criteria: the FP baseline (grey box epoch 0) on the same records, ratio included,
  since (b) is a weak bar next to it; the fraction of stages with unstable free runs; wall time.
- Budget per local run: 100 updates per stage, 6 stages, validations at each stage start and end.

### BB2-015 Clarifications of BB2-014 before any run
**Date**: 2026-09-29
- fit() records one Loss_train value per call (plus a NaN at its initial validation), so (c) is
  evaluated over TWO consecutive nf = 400 calls: stages 10, 25, 50, 100, 200, 400, 400; (c) =
  the second call's Loss_train < the first's.
- "The final model" in (b) is the best stable validation of the nf = 400 stage, because fit()
  reloads its best checkpoint at the end of every call.

### BB2-016 d1 outcome: the pre-registered curriculum fails at its first stage (stop per user)
**Date**: 2026-09-29
- At random init the model + K1 loop grows about 270x per 4 kHz step (training loss 7.3e48 on
  10-step windows); a 10-step window is already far too long for the loss to be usable, so the
  BB2-013 schedule does not keep training finite. Mechanism: K1's folded feedthrough (3.7e3,
  B1) times the random model's one-step input-to-output gain.
- Runner bug found and fixed after the run: deepSI's checkpoint folder under the redirected
  LOCALAPPDATA was not created (FileNotFoundError at the first save).
- Stopped here per the user's rule; next step for the user to choose.

### BB2-017 Variant A, one try (user "yes lets try A once", 2026-09-29)
**Date**: 2026-09-29
- Same model, loop, criterion (BB2-014, BB2-015) and budget per stage as d1; only the schedule
  starts at one-step prediction: nf stages 1, 2, 5, 10, 25, 50, 100, 200, 400, 400.
  # THEORY: one-step model training as in model-based RL (MBPO, Janner et al. 2019) and
  one-step pre-training before simulation error (LITERATURE.md SQ4, SQ6); nf = 1 scores h(x0)
  only, nf = 2 includes one controller step.
- Pre-registered stop rule for the arm: if the run does not reach a stable nf = 400 model that
  meets BB2-014, the plant-model black box is closed and the closed-loop-map option (B) is taken,
  with no further variants.

### BB2-018 Plant-model black box closed; option B (closed-loop map) taken (user rule, 2026-09-29)
**Date**: 2026-09-29
- d2b met the BB2-017 stop rule: training loss 0.10 (nf 1), 299 (nf 2), 3.7e15 (nf 5), free
  runs NaN throughout, so no stable nf = 400 model is reachable. User: "if you can actually run
  and it doesnt work, go to B". The plant-model black box (Jan's SUBNET under K1) is closed:
  three designs tried (random start, window curriculum from 10 and from 1 steps) plus the ICI
  design, all failing by the same loop gain (B1: folded K1 feedthrough 3.7e3). R8 result.
- **Option B**: Jan's SUBNET, deepSI's own model class and fit(), inputs (r, f), output y, on the
  thesis records through the pipeline's resampling (`_record_at_fs_new`, fields point-sampled);
  na = nb = 29 (grey box's lag), na_right = 0 (Jan); nx 16; lr 1e-3; auto_fit_norm on the 18
  training records; nf 400, float32 (deepSI default, Jan; deviation from the float64 shared
  setting, stated); local stride 100, batch 64 (BB2-009). Runner `train_bb_clmap.py`.
  The prediction is the grey box's scored quantity (y of the loop under K1 given r and f).
- **No E5** for this arm: the model has no plant; LITERATURE.md SQ5 (indirect recovery needs the
  exogenous plant-input signal f + K1 r, unbounded under K1's integrator).
- **Criterion** (as BB2-014 where it applies): per seed and version, (a) all validation free
  runs finite; (b) final per-axis RMS <= 1/3 of the mean-model RMS; (c) deepSI's Loss_train
  last < first finite. 3 seeds x 2 versions, local 10 epochs.

### BB2-019 Option B trains on the servo error e = y - r (CRITIC findings 1 to 3; handoff 2026-10-03)
**Date**: 2026-10-03
- Problem: on raw y, auto_fit_norm scales y by its move std (0.05 to 0.15 m), so the loop's own
  behaviour (y - r, ~1e-5 m) is ~1e-4 of the normalised range; b1 to b6 are 30 to 100x worse
  than the zero-parameter predictor y_hat = r.
- Change (one signal, one dtype): the deepSI output signal is e = y - r (computed in float64),
  normalised by deepSI's own auto_fit_norm; prediction y_hat = r + e_hat; score RMS(y_hat - y) =
  RMS(e_hat - e) in metres, the thesis metric (RESULTS-DESIGN.md:77-78). Still Jan's model, Jan's
  fit(), Jan's inputs (r, f); nothing about the plant or K1 enters.
- Float64: deepSI 0.3.29 casts every batch to float32 itself (fit_system.py:684, encoders.py:288,
  298), so a float64 model would need an override of deepSI (not allowed). `--in64` instead hands
  deepSI float64 records: its normalisation then runs in float64 before its own cast, so e keeps
  ~1e-7 relative resolution (~1e-12 m) and r ~6e-8 of its std (3e-9 to 1e-8 m) instead of 3e-8 m.
  This replaces the handoff's "(reading) float64 is needed".
- Flags added to `train_bb_clmap.py` (`--target e --in64`, plus the search axes na, na_right, nf,
  batch, stride, f/h and encoder width and depth); every default reproduces b1 to b6.
- Criterion for the settling run e1 (noisy, seed 1, 10 epochs, all else as b1): NRMS_e < 1 on
  every axis and the validation sim-RMS falling over the last epochs. Section-10 PASS rule
  (handoff) applies to the final configuration only.

### BB2-020 Add the reference increment dr = r(k) - r(k-1) as input (e1 plateau)
**Date**: 2026-10-03
- e1 (BB2-019) plateaued in 4 epochs at normalised loss 0.82 (val 1.66e-5 m vs 2.4e-5 at init).
- Reading: the servo error answers the reference's velocity and acceleration, but r enters as a
  position normalised by its move std (0.05 to 0.15 m). At 4 kHz one-sample increments are about
  v Ts / std ~ 2.5e-4 of the normalised range and second differences ~ 1e-5: the network must
  difference inputs near float32 resolution (6e-8) through tanh layers. HEURISTIC estimate from
  v ~ 0.1 m/s, a ~ 10 m/s^2; checked in e2 by whether the plateau moves.
- Change: three more input channels dr = r(k) - r(k-1), formed in float64 before deepSI's
  normalisation (`--dr`). Same exogenous signal (r is its running sum), no plant or controller
  information, deepSI's own model and fit(); r stays in for the Y-dependent dynamics.
- e2 = e1 + `--dr`, noisy, seed 1, 10 epochs. Criterion as e1.

### BB2-021 Learning rate 3e-3 and validation every 3 epochs (e2: optimisation-limited)
**Date**: 2026-10-03
- e2's training loss still fell at epoch 10 (0.79 -> 0.76) with 1330 updates of batch 64; the
  grey box's design budget is ~33k updates of batch 512. The local wall clock, not the model, is
  the limit.
- lr 3e-3 (REPORT search range 3e-4 to 3e-3; Jan's 1e-3 x 3). HEURISTIC: one step up the grid.
- deepSI's own `its_per_val` = 399 (3 epochs): validation was 60 % of wall time; this gives ~1.7x
  more updates per run. Selection still on the 6 validation records, at a 3-epoch grain.
- e3 = e2 + both, 15 epochs (5 validations at 399 updates, ~30 min).

### BB2-022 Batch 256 with stride 25, and a fixed training wall clock (deepSI `timeout`)
**Date**: 2026-10-04
- e3 still improved ~3 % per 400 updates at the end: more data per wall-clock second is the lever.
- On this CPU an update's cost is dominated by the 400 sequential BPTT steps through 2 x 8 nets,
  not by the batch's FLOPs (HEURISTIC, measured in e4's sec/batch). Batch 256 (REPORT range
  256 to 2000) with stride 25 keeps 133 updates per epoch at 4x the windows per update.
- deepSI's own `timeout` = 1500 s of training: every search run now gets the same wall clock, so
  runs compare at equal cost (R8). Validation every 399 updates as e3; the best validated model
  is kept by deepSI.

### BB2-023 Learning rate 1e-2 at batch 256 (e4: per-update progress unchanged by batch)
**Date**: 2026-10-04
- e4 matched e3 update for update (val 1.37e-5 at 1596 updates both), so gradient noise was not
  the limit; the step size is. A 4x batch makes a larger step safe in the usual linear-scaling
  sense (HEURISTIC, Goyal et al. 2017 rule of thumb, not verified for BPTT through a free run).
- lr 1e-2 = e3's 3e-3 x 3.3; above the REPORT grid (3e-3), stated. deepSI keeps the best
  validated model, so a late divergence cannot spoil the selected result.

### BB2-024 Wider nets: f and h 32 x 2, encoder 64 x 2 (REPORT search grid)
**Date**: 2026-10-04
- e5 (lr 1e-2) is the best so far; the move records (T1, T2, P1) remain 4 to 9x the grey box on X.
  Jan's 8-node nets are sized for a 3-dof MSD (REPORT "Hyperparameters"); this loop carries plant
  plus a 9-state controller, friction and Y-dependent inertia.
- f, h 32 x 2 and encoder 64 x 2, both inside the REPORT grid. On this CPU the per-step overhead
  dominates an update, so width should cost little wall time (HEURISTIC, checked by updates
  reached in the 1500 s budget).

### BB2-025 Encoder window na_right = nb_right = 1, as the grey box (CRITIC finding 6)
**Date**: 2026-10-04
- e6 (wider nets at lr 1e-2) diverged at update 548; width is dropped at this step size.
- The grey box's encoder sees u and y up to and including k0 (na_right 1); option B so far up
  to k0 - 1 (Jan's default 0). deepSI's own `na_right`/`nb_right` arguments; scoring still from
  k0 = 29. Removes one scoring asymmetry; e7 = e5 + this. The better of e5 and e7 is the final
  configuration for the 6-run proof.
- Outcome: not runnable in stock deepSI 0.3.29 (its free run ignores na_right: encoder 348 vs 360
  inputs). Dropped; the na_right asymmetry with the grey box stays and is disclosed.

### BB2-026 Final configuration for the proof = e5, fixed at 1596 updates
**Date**: 2026-10-04
- e5 is the best search result (9.8e-6 / 1.05e-5 / 1.43e-5 m); e6 diverged, e7 not runnable.
- The proof uses a fixed update count (12 epochs = 1596 updates, validation every 399) instead of
  the wall-clock timeout, so the six runs share one budget regardless of machine load. e5 reached
  and selected exactly the 1596-update model, so it stands as noisy seed 1.

### BB2-027 Budget run at lr 3e-3 (proof FAIL on X; lr 1e-2 diverges in 3 of 8 runs)
**Date**: 2026-10-04
- Proof (BB2-026): Y and NRMS_e pass, X1 / X2 at 2.6 / 3.1x the epoch-0 grey box: FAIL. The
  handoff's branch "NRMS_e < 1 and improving with budget" applies, so the open number is the
  budget. lr 1e-2 produced a NaN training loss in e6, noisy s3 and noise-free s2.
- e8: the final configuration at lr 3e-3 (e3's stable step), 9576 updates (~2 h), validation every
  798 updates. Measures the error against budget at a step the server runner can use for 50k
  updates (runner default BB_LR=3e-3).
- Outcome: X 2.0 / 2.2x the grey box at 9576 updates (proof 2.6 / 3.1x at 1596), validation flat
  over the last 3000 updates. Budget helps but flattens; recorded as the overnight verdict.

### BB2-028 Δr ablation at the e8 setting (user 2026-10-06)
**Date**: 2026-10-06
- Δr was added at e2 (3 to 5 % on X) and never re-tested. User: run the e8 setting without it.
- e9 = e8 minus `--dr`; everything else identical (noisy, seed 1, lr 3e-3, 9576 updates).
- Reading: within ~10 % of e8 (HEURISTIC margin, about the seed spread 1.09 to 1.26) means Δr
  is dropped and the method is "r and f in, e out"; clearly worse means Δr stays.

### BB2-029 Hyperparameter search, batch 1: design (DRAFT, user 2026-10-06)
**Date**: 2026-10-06
- User: search the black box's hyperparameters without dr, then a proper run with the winner;
  full-length runs on the server, not short proxies. TRAINING-DESIGN.md "black-box search space:
  deferred" is answered here.
- Plan: batch 1 = 18 configurations, 1 seed (`runners/search_grid.conf`, `runners/run_bb_search.sh`);
  batch 2 = the top 3 with 2 more seeds, chosen on the 3-seed mean (seed spread 1.09 to 1.26 makes
  single-seed differences under ~20 % unreliable); batch 3 = winner, 3 seeds x 2 versions, then one
  test-set evaluation. Selection on the 6 validation records only.
- Shared: noisy, target e, float64 records, inputs r and f only (no dr, user), batch 256, stride
  25, 20000 Adam updates for every configuration (equal budget in updates, TRAINING-DESIGN.md
  section 2), validation every 1000, deepSI keeps the best validated model, 22 h timeout.
- f block-averaged over each 4 kHz interval (`--f-blockmean`, the grey box's u rule, D-087) to
  remove the half-sample timing asymmetry (CRITIC 2nd review, finding 2); index 17 keeps it
  point-sampled to measure the effect. r stays point-sampled (as in the grey box's y rule).
- Centre: nx 16, f/h 16 x 2 (the grey box's network, TRAINING-DESIGN.md section 3), encoder 32 x 2
  (2 x f/h width, Jan's 16 : 8 ratio, HEURISTIC), lr 3e-3 (e8 stable; 1e-2 diverged in 3 of 8),
  nf 0.1 s, na 29 (the grey box's window and lag).
- Axes: A nx {8, 16, 32} x width {8, 16, 32} (9); B depth 3 (1); C lr {1e-3, 1e-2} (2); D nf
  {0.2, 0.4 s} (2, the grey box's sweep values); E na {15, 60} (2); F checks: with dr, f
  point-sampled (2). One-at-a-time around the centre for B to F (HEURISTIC; interactions are
  checked only through batch 2).
- 20000 updates: e8 plateaued near 6400 at lr 3e-3 with 8-wide nets; 3x margin for larger nets,
  which converge more slowly (TRAINING-DESIGN.md section 3 caveat). HEURISTIC.
- CPU partition: the update is dispatch-bound (BB2-022); GPU only as an opt-in test (BB_CUDA=1).
- 2026-10-07 (user): third data version `lowpass` = pipeline mode `thesis_taf_lowpass_noise`
  (Coulomb-tanh-and-MSD-lowpass-noise, same D-232 anti-alias rule and trim as noisy). Runner takes
  `BB_VERSION` (default noisy); outputs go to `outputs/search/<version>/`.
