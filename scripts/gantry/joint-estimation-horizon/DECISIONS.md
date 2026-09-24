# joint-estimation-horizon decisions

Session of 2026-09-24, handoff `tasks/handoffs/2026-09-23-joint-estimation-horizon.md`. Every entry
is written BEFORE the code or run it governs. Thresholds are pre-registered here and never moved
after a number is seen; a change after data is a new entry marked as a deviation.

### JH-001 Infrastructure: folder, vendoring, launcher, compute device
**Date**: 2026-09-24
**What**:
- Layout per handoff section 9 G0: `README.md`, `DECISIONS.md`, `RUNS.md`, `REPORT.md`, `vendor/`,
  `tools/`, `info/` (G1), `staged/` (G2), `runners/` (G3), `outputs/<run>/`.
- Vendored from the WORKING TREE at `1ec57c1` (branch `Augmentation`, dirty; the handoff names
  `a97a577`, which is an ancestor of `1ec57c1`): the whole `model_augmentation` package (python
  files), `scripts/gantry/gantry_dynamic/`, `scripts/gantry/common/`, and the entry file
  `scripts/gantry/gantry_interconnect_dynamic.py`. Provenance table with sha256 and git status in
  `vendor/VENDORED.md`. Only lines marked `VENDOR-PATCH (JH-001)` differ from the sources;
  `tools/check_vendor.py` re-verifies that.
- Patches, each needed for a stated reason and nothing else:
  1. `gantry_dynamic/__init__.py`: put `vendor/` on `sys.path` instead of the repo root, so
     `model_augmentation` resolves to the vendored copy (same patch as blackbox-closed-loop BB-002).
  2. `gantry_dynamic/config.py`: `REPO_ROOT` five levels up (data are read in place from
     `data/gantry/matlab/trajectory/`, never copied); `save_dir` redirected to
     `outputs/sim/` so that no code path can write under `simulations/`.
  3. `gantry_dynamic/data.py`: environment switch `JH_FILESET` = `production` (default, the entry
     file's 18/6/5 lists, exact no-op) or `obc14` (the 14/4/4 lists the D-192 arms 84032-84037
     ran with, before D-206 appended the Telica profiles). Needed because G0(b) and G2 compare
     against numbers measured on `augmentation_ma50_b140-230_a6_z03`, which holds only those
     records.
- The repo is pip-installed in develop mode (telica-real G0), so an unpatched import can silently
  resolve to the repo copy. `jh_env.py` puts `vendor/` first on `sys.path` and `check_no_leak()`
  asserts that every loaded `model_augmentation`, `gantry_dynamic` and `common` module file lies
  inside `vendor/`. Every script calls it.
- Launcher `tools/run.sh`: every run goes through `tools/watchdog.ps1`, copied unchanged from
  `telica-real/tools/watchdog.ps1` (TR-002 thresholds: RAM alert 4.0 GB, kill 2.5 GB; C: alert
  3.0 GB, kill 2.0 GB). It refuses to launch below 3.5 GB available RAM
  (`# HEURISTIC:` the machine idles at 4.2 to 4.9 GB available with nothing of ours running,
  telica-real G0; 3.5 GB keeps 1 GB above the kill line) and prints the GPU process list before
  launch. `PYTHONDONTWRITEBYTECODE=1` so no `__pycache__` is written outside this folder.
- Compute: CPU, float64, for G1 and G2. The rollout is dispatch-bound (D-169), the Quadro P2000
  runs float64 at 1/32 rate and cannot run inductor (sm_61), and the handoff (section 13) asks for
  float64 where cancellation matters. It also keeps the GPU free for other sessions. G0 uses the
  production dtype (float32) because it reproduces a production number.
**Why**: handoff section 0 (files only inside this folder, vendor instead of editing, watchdog on
every run) and section 13 (memory).
**Ruled out**: importing the repo packages in place for the computations (a later edit to the
live pipeline would silently change this session's numbers); a GPU default (no gain, contention).

### JH-002 G0 criteria (pre-registered)
**Date**: 2026-09-24
**What**: G0 PASSES when all four hold.
- (a) The vendored entry file's `CFG` (read from `vendor/gantry_interconnect_dynamic.py`, no
  override except `device='cpu'`, `compile_mode=None`: the local card cannot run inductor and the
  validation free run is eager in production anyway) builds the production model the way `main()`
  does (seed, `load_datasets`, `compute_normalization`, `build_model`, `build_closed_loop`), and
  its epoch-0 closed-loop validation number, `fit_sys.cal_validation_error(val_ckpt_data,
  'sim-RMS')` after `eval()` (what `fit()` stores as `Loss_val[0]`), equals the reference
  **1.4768032959e-05 m** within **1e-6 relative**. Reference: the untrained CFG model's
  `Loss_val[0]` on `augmentation_ma50_z03_b140-230_a6all_telica_coulomb`, measured today by
  blackbox-closed-loop G1(c), identical from the original repo packages in a separate process.
- (a') Independent of that sibling number: the same computation run from the ORIGINAL repo
  packages, imported in place in a separate process (read only, no `main()`, bytecode off), agrees
  with (a) within 1e-6 relative.
- (b) The joint-estimation arm the G2 comparison refers to: `OBC_ARM=noproj` (reduced ten
  combinations, 10 % detuned start, prior off) with `JH_FILESET=obc14` and
  `mode='augmentation_ma50_b140-230_a6_z03'` reproduces run 84032's logged
  `Initial Validation sim-RMS= 2.296473121532472e-05` within **1e-5 relative**
  (`# HEURISTIC:` 84032 ran float32 on an A100; CPU and GPU float32 differ in reduction order,
  D-169, and the same config logged 2.2964694e-05 in 83795/84037, a 1.6e-6 relative spread).
- (c) `check_no_leak()` passes in (a) and (b); `tools/check_vendor.py` reports 0 unmarked
  differences.
**Why**: the handoff's G0 PASS ("the vendored entry builds the production model and reproduces
its epoch-0 closed-loop validation number"), plus (b) because G2 is about the joint-estimation
model on the older dataset, not about the entry file's default (joint estimation off).

### JH-003 G1 design and thresholds, simulated data (pre-registered)
**Date**: 2026-09-24
**Quantity.** For each identifiable combination `theta_j` (the ten free coordinates of
`Reduced_Gantry_State_Block`: nine log, `m_diff` relative-linear, so a standard deviation in free
coordinates IS a relative standard deviation), the Cramer-Rao relative standard deviation
`sd_j = sqrt([F^-1]_jj)` of the closed-loop WINDOWED output model, as a function of window length.
**Model and data.** The production build (vendored `build_model`, `use_f64=True`, joint
estimation on the reduced block, `combo_init_detune=None`, i.e. at the TRUE combinations, prior
off) on the production simulated set (`CFG.mode`, the 18 training records, production loader and
normalisation), closed loop with the production controller bank at 4 kHz in residual form,
`xc = 0` at every window start, window start state from the production `linear_map` encoder. The
ANN is ABSENT: the rollout is physical block + output map + controller, checked equal to the
production `closed_loop_rollout(fit_sys.hfn, ...)` with its zero-output ANN (criterion V1 below).
Sensitivities `dy/dtheta` (10 directions) and `dy/dx0` (6 physical initial states) by FORWARD
MODE (`torch.autograd.forward_ad`), float64, CPU.
**Windows.** Per record the span `[29, 29 + 3200 M)`, `M = floor((N - 29) / 3200)`, is partitioned
into NON-overlapping windows of `nf` samples, `nf` in {100, 200, 400, 800, 1600, 3200}
(25 ms to 0.8 s at 4 kHz). The same samples enter every `nf`, so the comparison across `nf` is at
equal data; only the number of window restarts changes. Non-overlapping because the training
stride-10 windows reuse each sample ~40 times and add no information; with disjoint windows and
independent noise the information of the windowed model is exact. Each `nf` with burn-in 0 and
with burn-in 100 (the first 100 samples of each window unscored, D-178; not defined at nf 100).
**Nuisance.** Primary: the six physical initial states are FREE per window (the production encoder
is trained jointly, so x0 is a per-window nuisance): `F = sum_w R_tt^T R_tt` from the QR factor of
`W^1/2 [S_x0 | S_theta]` per window, i.e. the Schur complement of the x0 block.
`# THEORY:` CRB of a parameter subset in the presence of nuisance parameters = inverse of the Schur
complement of the Fisher information (standard block-inverse identity, e.g. Kay 1993,
Fundamentals of Statistical Signal Processing: Estimation Theory, ch. 3). Secondary: x0 known.
**Noise level (data-derived).** Primary: white, per stage channel, sigma = **8.69 / 10.63 / 5.94 nm**
(X1 / X2 / Y), the Telica closed-loop standstill measured error (telica-real, closed-loop-noise
G0). Why the ERROR level and not the source spectrum: in the residual-form closed loop, at the
true model, the residual is `y_data - y_model = S v` for measurement noise `v` and `= S G d` for a
force disturbance `d` (both to first order), which is exactly the standstill error the encoder
records, so the measured error level is the residual level. Secondary (robustness): coloured, the
pooled measured error PSD of closed-loop-noise G1 (145 logs, `outputs/g1_spectra/g1.npz`),
point-sampled to 4 kHz as the loader samples `y`, per-channel Toeplitz covariance, cross-channel
coherence ignored (stated).
**Threshold.** A combination is INFORMED at `nf` when `sd_j` < **1 %**. Reason: its two-sigma
interval (2 %) is then below the median per-combination bias the projection predicts from data
(`|J+ Delta*|`, median 2.9 % over the ten, `baseline-and-added-dynamics.md` section 4) and a fifth
of the 10 % detuning the recovery runs start from, so the objective can tell the estimate apart
from both. The handoff's example level, **5 %**, is reported as a second column; it only separates
the estimate from the detuned start. `m_diff` is reported on its own scale (0.5 kg), not the
combo-err meter's 10.45 kg (which would read 21x smaller).
**Affordable nf.** `# HEURISTIC:` training cost per update is proportional to nf (dispatch-bound
rollout, D-169); the production schedule, 33,400 updates at nf 400 and 1.24 s/update, takes
21.1 h of a 24 h wall. Affordable = nf 400 at a 24 h wall, nf 800 at a 48 h wall; every longer
window is reported as not affordable at the production update count.
**Validity checks, before any table is read.** V1: the ANN-free rollout equals the production
`closed_loop_rollout` through `fit_sys.hfn` (zero-output ANN) to max |dy| <= 1e-12 of `ystd` on 8
windows of nf 400. V2: forward mode against a central finite difference (float64, step 1e-6 in
free coordinates) on 4 windows at nf 400: max relative difference <= 1e-4 per direction.
V3: `F` symmetric positive definite at every `nf` reported.
**PASS (G1, simulated).** A table per combination of the smallest `nf` at which `sd_j` < 1 % (and
< 5 %), primary variant, with the combinations that never get there within nf 3200 named, `cond(F)`
per `nf`, and the measured compute and memory per `nf`; V1 to V3 pass. The Telica table is JH-004.
**Addendum (2026-09-24, before any G1 number).** The PASS table's primary variant is: white noise,
x0 free, **burn-in 0** (the entry file's current objective, `burn_in=0`); burn-in 100 is the
second table. Coloured-noise autocorrelation: `R(tau) = int P(f) cos(2 pi f tau) df` on the
Welch grid of `g1.npz` at 20 kHz, times a Bartlett taper of the Welch segment length (the lags the
estimate supports; the taper keeps the Toeplitz matrix positive definite), then every fifth lag
(the loader point-samples `y` to 4 kHz). V4: `R(0)` equals the pooled rms 9.848 / 10.400 /
6.338 nm (closed-loop-noise G1 closure) within 2 %.

### JH-005 G2 design, arms and tolerance (pre-registered)
**Date**: 2026-09-24
**Setting.** The dataset the reference was measured on, `augmentation_ma50_b140-230_a6_z03`
(`JH_FILESET=obc14`, the 14 training records), the reduced block in float64, starting from the
OBC arms' detuned start (`combo_init_detune` = [1.10, 1.10, 0.90, 1.10, 0.90, 0.90, 1.10, 1.10,
0.90, 1.10]), prior off, NO ANN anywhere. Reference: `J+ Delta*` at `theta*`, stencil variant,
per combination (`delta_orthogonality.json`, 4.956 % combo-err), percentages in the training
meter's convention (`|true|`, `m_diff` on 10.45 kg; `m_diff` is also reported on its own 0.5 kg).
**What "restricted to range(J)" can mean, decided before fitting.** For the one-step least-squares
objective, R2 says the parameter optimum is `J+ Delta*` whatever the learned part represents,
INCLUDING nothing: a baseline-only one-step fit and its projected version `||P0 r(theta)||^2` have
the same first-order minimiser. For a multi-step objective the literal projection of the rollout
residual onto the span of its own sensitivities has the SAME minimiser as the unrestricted fit
(first-order normal equations are identical), so that reading is vacuous. The non-vacuous reading
is to supply the part of the missing dynamics OBC leaves to the network, `c = (I - P0) r0`
(`r0` the recorded one-step residual at the detuned start, `P0` the projector onto `range(J0)`
over the reference set: data-derived, no oracle), as a time-indexed additive correction on the six
physical rows of the rollout. That is the rollout an ideal orthogonal network would produce.
Stated in advance: at first order this arm lands on `J0-oblique J+ Delta*` for ANY horizon,
because with `c` supplied the model's one-step error vanishes at `theta* + J+ Delta*` sample by
sample; its gap therefore measures curvature (R3), the obliqueness of `P0` (R4, 4.76 to 4.96 %) and
the stencil artefacts in `c` integrated by the rollout, not the decoupling itself. The arm that
tests whether the decoupling carries to the multi-step closed-loop objective is (i): with NO
network, R2 still predicts `J+ Delta*` for the one-step objective, so the gap of the multi-step
baseline-only fit from `J+ Delta*` is exactly the network-dependence of the multi-step parameter
estimate between a null and a perfect orthogonal network.
**Arms.**
- (a) ONE-STEP baseline-only fit, `min ||x_next - f(theta, z)||^2` over all 671,944 reference
  tuples (stride 1, stencil next state, normalised rows), Gauss-Newton from the detuned start with
  the Jacobian of `build_stacked_sensitivity` (production primitive), accumulated in chunks.
- (i) LONG-WINDOW baseline-only fit, unrestricted: closed-loop residual-form rollout, non-overlapping
  windows of nf = 3200 (0.8 s) over `[29, 29 + 3200 M)` of every record, burn-in 100, per-channel
  weight `1/ystd^2` (the training objective's weighting), the six initial states FREE per window
  (variable projection: eliminated per window by the Schur complement of the Gauss-Newton system),
  Levenberg-Marquardt on the ten free coordinates.
- (ii) the same with the correction `c` added at every step.
- (i-400), (ii-400): the same two at nf = 400 (the training horizon), secondary, to show the
  horizon dependence.
Convergence: relative cost decrease < 1e-9 over an accepted step, or step < 1e-6 in free
coordinates, or 30 iterations (reported). Sensitivities by forward mode (jh_model).
**Tolerance.** `# HEURISTIC:` the reference itself cannot be resolved below the construction's
null-test floor, which alone predicts 0.742 % combo-err with per-combination values up to 1.17 pp
(`delta_orthogonality.json`, `floor_bias_pct`). G2 PASSES when the gap of arm (ii) nf 3200 from
`J+ Delta*` is <= **1.2 pp on each combination** and <= **0.74 pp RMS**. The same tolerance is
applied to (a) and (i) and reported as "within / outside" but only (ii) decides PASS, as the
handoff specifies. `m_diff` is judged in the meter convention and reported on its own scale.

### JH-004 G1 on Telica (pre-registered)
**Date**: 2026-09-24
**What**: the JH-003 quantity on the real machine, with telica-real's own pipeline.
- Vendoring: the telica-real code tree (python, `params/`, `friction/`, `controller/` incl. the
  identified 5 kHz controller `telica_sos_identified_5k.npz`, `baseline/` incl.
  `recovered_params_a2.json`, `pipeline/` incl. `norm_frozen.json`, `rate.py`, `data_index.py`,
  `tr_env.py`, and telica-real's own `vendor/`) copied to `vendor/telica_real/`, provenance in
  `vendor/VENDORED.md`. Patches marked `VENDOR-PATCH (JH-004)`: `tr_env.py` repo depth and
  outputs folder only. Telica data are read ONLY through the vendored loader inside a script
  (data access policy); nothing is cached to disk.
- Model: `pipeline/telica_model.py::build_model_real` with `joint_estimation=True` (TR-025: ten
  combinations + three log Coulomb levels, tanh friction), at the G4 attempt-2 values (no
  detune), 5 kHz (TR-022, the rate at which an identified Telica controller exists), float64,
  frozen normalisation. ANN absent (zero output); x0 from the production `linear_map` encoder.
- Data: every readable log of the TRAIN split (all iterations), cut as the pipeline cuts them
  (motion start - 100 ms to the end, 5 kHz). Records last about 0.5 s after the cut, so the grid is
  the durations that fit: nf in {125, 250, 500, 1000, 2000} samples = {25, 50, 100, 200, 400} ms,
  non-overlapping windows over `[29, 29 + 2000)` of every record long enough (count reported);
  burn-in 0 and 125 samples (25 ms, the D-178 duration at 5 kHz).
- Sensitivities: the 13 free coordinates by forward mode through the production closed-loop
  rollout (`closed_loop_rollout` through `fit_sys.hfn`, parameter made a dual tensor), plus the six
  initial states. Primary: x0 free, white noise at the JH-003 sigma (it is Telica's own standstill
  level). The 13x13 information is inverted; the ten combinations are reported (the three cc
  levels are co-estimated in TR-025 and reported beside them).
- Threshold, table and validity checks as JH-003 (1 % primary, 5 % secondary; V2 finite
  difference on 4 windows; V3 SPD). The simulated table (JH-003) is the deciding one; the Telica
  table reports what the real excitation (one positive move per log) supports. G1 is not failed by
  the Telica table; if it cannot be produced after three attempts it is reported FAIL with its
  mechanism beside the simulated PASS.

### JH-006 G3 decision rule (pre-registered before any G1 or G2 number)
**Date**: 2026-09-24
**Inputs.** G1: `n_j` = smallest nf with sd_j < 1 % (primary variant), "affordable" = nf <= 400.
G2: `a_ok`, `i_ok`, `ii_ok` = arm within the JH-005 tolerance.
**Rule, evaluated in order; the first that applies is the recommendation.**
1. `a_ok`: the OBC parameter optimum is computable WITHOUT the network by a one-step baseline-only
   fit, which needs no window at all. Recommend **(b) staged**: stage 1 = one-step baseline-only
   least squares (if also `i_ok`, the long-window fit is an equivalent stage 1 and is named as the
   real-data fallback); stage 2 = the augmentation with the ten combinations FIXED at the stage-1
   values and still protected (OBC basis over all ten columns at that point; confirmed by reading
   that `attach_obc` builds `J` over the full `free_params` vector whatever is trainable).
2. not `a_ok` but `ii_ok`: recommend (b) staged with the long-window fit plus the orthogonal
   correction as stage 1.
3. neither: G1 decides. Every combination with `n_j` <= 400: **(d) separate parameter learning
   rate** on those; combinations with `n_j` in (400, 3200]: **(c) longer windows** for the parameter
   gradient (chunked, exact, at 4 kHz); `n_j` = never: **(a) freeze** at the start value, kept in
   the protected space. If this mixes schemes, the one covering the most combinations is the
   recommendation and the rest are stated as its restrictions.
Coherent multiple shooting (e) is recommended only if G1 shows the x0-free information at nf 400
far below the x0-known information for the slow combinations (the multiple-shooting argument is
that the reset, not the length, loses the information) AND rules 1 and 2 do not apply.
**Server runs (prepared, not submitted).** Arm = the recommended scheme; control = the joint OBC
arm exactly as run 84033 (tangent OBC, detuned start, prior off, 14 records of
`augmentation_ma50_b140-230_a6_z03`), matched on everything except the scheme. The written
prediction states the final combo-err per arm (with its tolerance) and the val sim-RMS ordering.
**Smoke.** One optimizer update for the arm and one for the control (2 in total, the session cap),
CPU, eager, batch 8, nf 400, reference set at a probe stride (memory), under the watchdog. PASS:
both run and the arm's update leaves the fixed parameters bit-identical while the control's moves
them (connectivity check, D-095).

### JH-007 G3 arm mechanism in the vendored entry file (implementation of JH-006)
**Date**: 2026-09-24
**What**: one environment switch `JH_ARM` in `vendor/gantry_interconnect_dynamic.py`, same narrow
form as `OBC_ARM` (allow-list, recorded in config.json because CFG is rewritten before it is
serialised). Shared by both values: `mode='augmentation_ma50_b140-230_a6_z03'` (requires
`JH_FILESET=obc14`, asserted), `epochs=150`, `joint_estimation=True`, reduced block,
`param_prior=False`, `obc=True`, `obc_space='tangent'`, i.e. run 84033's configuration.
- `JH_ARM=control`: the detuned start, parameters trainable (84033 unchanged).
- `JH_ARM=staged`: `combo_init_detune` = stage-1 combinations / nominal combinations, read from
  `JH_STAGE1_JSON` (the G2 stage-1 result), so the free coordinate is 0 AT the stage-1 point; after
  `build_model`, `free_params.requires_grad_(False)`: Adam never sees a gradient for it, the OBC
  basis is still built over all ten columns at that point (JH-006 reading), nothing else changes.
- `JH_SMOKE=1`: CPU, eager, batch 8, `n_its=1`, `lbfgs=False`, `its_per_val=None`, nothing else;
  the OBC reference set is built at a probe stride `JH_REF_STRIDE` (default 97) through a patched
  `attach_obc` call, memory only (JH-006 smoke).
All patch lines marked `VENDOR-PATCH (JH-007)`.
**Addendum (2026-09-24, before the smoke).** Two smoke-only settings, memory and file scope, not
behaviour: training-window `stride=400` (the stride-10 window arrays are ~0.7 GB, more than this
machine has free), and `LOCALAPPDATA` pointed inside the run folder, because deepSI's `fit()`
writes its `_best`/`_last` checkpoints to `%LOCALAPPDATA%/deepSI/checkpoints` on Windows, outside
this folder. Baseline of that directory taken before any run of this session: 636 files, newest
07:18 today (another session); it is re-checked at the end.

### JH-008 G1 V2 attempt 1 FAILED; attempt 2 pre-registered (deviation, before any G1 table)
**Date**: 2026-09-24
**Observed (run g1_checks).** V1 PASS (max |dy| exactly 0.0: the ANN-free rollout is bit-identical
to the production one). V2 FAIL at the pre-registered 1e-4: forward mode against a central
difference with h = 1e-6 disagrees by 3.1e-4 (kb_sum), 5.8e-4 (cg1), 2.0e-4 (cg2), 6.1e-4 (cy);
every mass direction and every initial-state direction agrees to 1.4e-6 or better (mh 7.8e-7,
m_total 5.2e-7, J_eff 3.6e-7).
**Mechanism hypothesised, to be tested, not assumed.** The four failing directions are exactly the
damping and stiffness family, whose closed-loop output sensitivity is small (the loop rejects a
velocity-proportional force), so a 1e-6 step moves the output by an amount comparable to the
rounding of a 400-step float64 rollout: the error should scale like 1/h (rounding), not h^2
(truncation). A forward-mode defect would not depend on h.
**Attempt 2.** Same four windows, a step ladder h in {1e-3, 1e-4, 1e-5, 1e-6}, relative error per
direction and step, and max |dy/dtheta| per direction printed. PASS if, for every direction, the
smallest error over the ladder is <= 1e-4, AND for the four damping and stiffness directions the
error falls when h grows from 1e-6 (the rounding signature). The tolerance is not changed.
