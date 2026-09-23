# telica-real report

```
G0 Infrastructure: PASS. Watchdog CSV 7 samples on a 30 s load (>= 5, TR-002); forced kill removed
  the Python child; 10/10 vendored imports resolve inside telica-real, 0 leaks.
G1 Parameters: PASS. 26 rows (datasheet first, Garcia fallback); vendored gantry_ss, controller
  and loader read params/, 0 old literals (TR-005). Force scale kept as two readings (TR-004).
G2 Friction: PASS. Karnopp port vs MATLAB 6-state copy 8.7e-18 m (tol 1e-12, TR-007), identical
  stick/slip census; vs the original .m (ma -> 0) 1.6e-15 m; cc = 0 bit-exact (0.0).
G3 Controller: PASS after 3 attempts on (i). Held-out replay gain LX1 0.9988 / LX2 1.0025 /
  LY 1.0041 (intervals 1 +- 0.0030 / 0.0088 / 0.0067, TR-009/TR-011). The 1.16/1.33/3.55
  mismatch is not a gain, state or frame error: the controller that ran differs from the zpk file
  (low-frequency tuning, notches); identified on train iter0. Residual form returns u_data
  exactly; units 2.3e-11 in float64, float32 fails (2-4 nm input rounding) -> pipeline is float64.
G4 Baseline: PASS on attempt 2 (attempt 1 FAIL: ccy 170.8 N > 90.3 N bound, and a Karnopp
  stick-slip limit cycle of 0.8-4.6 um). Stable on 159/159 records; held-out NRMSE 1.95 vs 70821
  4.90 (iter0: 0.20 / 0.15 / 0.38); masses 90.9 / 20.95 kg vs datasheet 91 / 19 (TR-014/015).
  DEVIATION: friction is the tanh form (identical to Karnopp while sliding); Karnopp's stick branch
  drives a limit cycle the machine does not have.
G5 Learnability: PASS, GO, server run warranted: yes. Repeatable/non-repeatable residual power
  4.9 / 5.5 / 4.6 (>= 1); held-out MEM headroom 4.8 / 5.4 / 3.1 (>= 1), p < 3e-24, R2 0.89 /
  0.85 / 0.71 (TR-017). ANN: rows X, Theta, Y + aug; nx_ann 2 (order test inconclusive, rule
  floor); horizon 40 ms (lower bound); band 3 Hz-2.5 kHz.
G6 Pipeline: PASS. Smoke test end to end, exactly 2 updates, ANN moved 2.0e-5; peak RAM 676 MB
  (smoke), 1142 MB in-process at the server horizon (probe). server_run.sh written, not submitted:
  80.3 KB/window-step -> batch 256 x 800 steps = 4.1 GB with chunk 200 (16.4 GB unchunked).
5 kHz (2026-09-23, TR-022): PASS, now the server default. Controller refit at 5 kHz (causal, lag 0)
  held-out gain 0.9988 / 1.0025 / 1.0013, VAF 0.998 / 0.998 / 0.9996; residual-form identity
  exact; G4 replay stable 159/159, held-out NRMSE 1.947 (20 kHz 1.952), 3.6x faster; 40 ms = 200
  steps, 4.1 GB graph at batch 256 without checkpointing.
Resources: peak tree RAM 1.21 GB (g6_probe), min available RAM 3.56 GB, C: min free 8.50 GB,
  0 watchdog kills (besides the deliberate G0 kill test), 2 augmentation optimizer updates total.
Next for the user: (1) submit pipeline/runners/server_run.sh (optionally a second arm, e.g.
  NX_ANN=8 NF_SECONDS=0.1); (2) ask Kamtin which controller ran for these logs (G3) and what the
  logged current convention is (TR-004); (3) decide on Karnopp vs tanh stick (G4, TR-015).
```

## G0 Infrastructure: PASS
Thresholds pre-registered in TR-002.
- Folder created (`README.md`, `DECISIONS.md`, `RUNS.md`, `REPORT.md`, `vendor/`, `tools/`,
  `params/`, `friction/`, `controller/`, `baseline/`, `learnability/`, `pipeline/`, `outputs/`).
- Vendored the framework and pipeline from the working tree (`vendor/VENDORED.md`); imports
  resolve inside `telica-real/` for all 10 checked modules, 0 leaks (`tools/check_imports.py`).
  The repo is pip-installed in develop mode, so `tr_env.check_no_leak()` guards every script.
- Watchdog (`tools/watchdog.ps1`): 30 s dummy load exited 0 with 7 CSV samples (criterion >= 5);
  forced RAM kill (`-RamKillGB 100`, grace 8 s) fired at 10.6 s, the first sample after the
  grace, and the Python child (PID 10884) and its tree were gone afterwards (0 survivors).
- Resource context: 4.2 to 4.9 GB available RAM with nothing of ours running, so the 2.5 GB kill
  level leaves ~2 GB for any script.

## G1 Parameters: PASS
Criteria TR-005, all four met (`params/check_params.py`, run g1_params).
- `params/telica_params.py` + `params/PARAMS.md`: 26 rows (value, unit, source, identifiability);
  every parameter the baseline (14 raw + Lb), friction law (cc, V_BRK, v0) and controller path
  (Kt, s_F, Ts) reads has a row.
- Vendored `gantry_ss.py` and `gantry_dynamic/controller.py` now import from `params/` (exact
  match, 0 old literals); the vendored loader takes `s_F` from `params/` and asserts Kt from
  `Telica 1.mat` (109 / 77.6) equals the datasheet Kt.
- Telica values used: X-moving 91.0 kg, mh 19.0 kg, cg 136 / 136 / 98 N/(m/s), cc 43 / 43 / 49 N
  (all datasheet MAXIMA, so upper bounds); Garcia fallback for Jb, Jh, cb, kb, Lb, d.
- Force scale kept as two readings (TR-004): (A) default `s_F = [3.469, 3.469, 3.202]` with
  datasheet masses, (B) `s_F = 1` with 3.5x lighter effective masses. Found while doing this: the
  datasheet names the four X motors X1-L, X2-L, X1-R, X2-R with L/R = the two gantries, which puts
  D-112's "forcer pair per logged channel" in doubt (question for Kamtin, recorded in TR-004).

## G2 Friction: PASS
Criteria TR-007 (a)-(d), all met (`friction/check_friction.py`, runs g2_prep, g2_matlab,
g2_compare; MATLAB R2025a `-nojvm -batch`).
- `friction/karnopp.py`: batched torch port of the Karnopp block of
  `gantrySystemExtendedCoulomb.m` (stage-frame force, Delassus stuck-set solve, active-set
  breakaway, `V_BRK = 2.25e-3 m/s`), injected as `u_eff = u_log - P F` at BOTH u-sites through a
  mixin on `Gantry_State_Block` / `Reduced_Gantry_State_Block`. No hidden MSD.
- (a) cc = 0 vs the frictionless parent: max diff exactly 0.0 on LPV and frozen branches, float64
  and float32 (2000 steps).
- (b) vs MATLAB R2 (6-state copy, friction block verbatim): max stage-position error 8.7e-18 m
  (tol 1e-12), stick/slip census identical at every step (0 mismatches).
- (c) vs the UNMODIFIED `.m` (ma = 1e-6 and 2e-6 kg, Richardson to ma = 0): 1.6e-15 m against a
  tolerance of 2.3e-10 m (the first-order absorber effect).
- (d) census over 8000 steps per rail [X1, X2, Y]: held 1000 / 2160 / 1228, sliding
  6952 / 5136 / 6352, broke away 48 / 704 / 420; all rails held for the first 50 ms of sub-cc
  force. Figure `outputs/g2_friction/g2_trajectory.png`.
- V_BRK and the step (TR-006): valid at 20 kHz (a_max h = 2.5e-3 < 2 V_BRK = 4.5e-3 m/s, margin
  1.8x), NOT at 4 kHz (1.25e-2 > 4.5e-3): a 4 kHz friction model could jump the stick band.
- Forms used downstream: `karnopp` for the G4 replays and recovery, `tanh` only where a gradient
  at v = 0 is needed and then named (TR-006).

## G3 Controller: PASS (float64 training form), after 3 attempts on (i)
Criteria TR-009 (+ TR-011 for attempt 3). Runs g3_ctrl_analyze ... g3_bank_diag.
- **Rate**: 20 kHz, the zpk's own Ts (TR-008): at 4 kHz the LX zeros at 1.87 kHz sit at Nyquist,
  the 2.4 / 3.1 kHz error peaks alias, and the Karnopp band is invalid (TR-006).
- **Sign and units**: `e = r - y = M2 * 1e-6` [m] -> current [A] -> force `diag(Kt s_F) i`
  (reading A: 378.1 / 378.1 / 248.5 N/A). Residual form uses the same K on `y_data - y_model`.
- **Linearity of the logged path** (iter0): feedforward leakage `max|MF30 - MF230| = 0.0 A` on all
  15 records; no saturation (max |MF230| 6.2 A against Ip 27.9 A, no plateau); the 6x6 zpk has 0
  off-diagonal entries and cross-rail replays add only 0.003-0.005 VAF (no decoupling matrix);
  coherence 0.95-1.0 (linear, time-invariant). One record (train xpos_-210_ypos-200 iter0) has
  MF230_Y identically constant and is excluded for Y.
- **The 1.16 / 1.33 / 3.55 mismatch**: reproduced exactly (1.164 / 1.328 / 3.552 on the D-073
  record). It is NOT a gain and NOT the controller's unknown initial state (attempt 1: with the
  zero-input-response basis the gains stay 1.196 / 1.355 / 3.531 on every record). The ratio
  logged/replay is frequency dependent: X 1.0-1.35x with -33 deg extra phase at 5-50 Hz, 0.55x
  at 50-150 Hz, the X2 299 Hz notch and the Y 150-400 Hz notch region absent from the log; Y 3-4x
  below 50 Hz. The "2.5 ms lag" of the schema is that low-frequency phase, not a delay (lag 0 after
  identification). Not a row mix-up and not a frame transform (attempt 2).
- **Mechanism, named and modelled (attempt 3)**: the controller that ran is an LTI operator that
  differs from the zpk file as stored. Its physical origin (retuned controller, another sensor in
  the loop, a drive-side filter) is not identifiable from these channels: a question for Kamtin.
  Modelled by re-fitting the zpk's own 8th-order structure (integrator fixed at z = 1) on TRAIN
  iter0: held-out val + test iter0 pooled gain **0.9988 / 1.0025 / 1.0041** against intervals
  **1 +- 0.0030 / 0.0088 / 0.0067** (TR-009 (i) PASS); held-out VAF 0.998 / 0.998 / 0.940
  (zpk file 0.945 / 0.935 / 0.879). Y keeps a residual mismatch around 200-500 Hz (VAF 0.94):
  carried forward as a known controller-path limitation on Y.
- **(ii) residual form**: `controller/telica_bank.py` builds the framework's `ControllerBank`
  (one row, 20 kHz only). With `y_model = y_data` it returns `u_data` exactly (max diff 0.0) in
  float64 AND float32. Units test (feedback vs an independent scipy filter): 2.3e-11 in float64
  (PASS); 9.1e-3 in float32 against the pre-registered 1e-3 (FAIL). Mechanism confirmed (4.6e-8
  agreement once the reference sees the float32-rounded error): float32 normalised positions
  round the servo error by 2-4 nm, above the 0.98 nm encoder step. Consequence TR-013: the
  real-data pipeline runs in float64; the float32 path is refused in G6.
- Attempt count: 3 on (i) (initial-state model, row/frame mechanisms, identification); 1 numerical
  fix on (ii) (TR-012, falsified) and its diagnosis (TR-013).

## G4 Baseline adequacy: PASS on attempt 2 (attempt 1 FAIL)
Criteria TR-014 (pre-registered), evaluated by `baseline/g4_verdict.py` (run g4_verdict).
Closed-loop replay with the identified controller, float64, 20 kHz, every readable record (159;
`train/xpos_-60_ypos120/iter6.log` is empty), both the direct form (from the reference) and the
residual (training) form.
- **Data facts** (g4_survey, g4_holding): every record is ONE positive move (X 40 mm at 1 m/s,
  Y 80 mm at 1.5 m/s), so Coulomb friction is not separable from a constant force while sliding.
  The holding force at rest flips sign across the move in 159 / 159 / 158 records: friction
  memory, which measures the held friction directly: 83 / 100 / 81 N (reading A), with implied
  static external force ~0.
- **Attempt 1** (regression cc, Karnopp): FAIL. (c) ccy = 170.8 N above the held-friction
  97.5th percentile (90.3 N); and every record ends in a stick-slip limit cycle (X ~0.8 um near
  100 Hz, Y ~4.6 um near 190 Hz) the machine does not show. Mechanism measured (g4_diag_lc): the
  frictionless loop is linearly stable (least damped 186 Hz, zeta 0.011) and settles to
  12 / 12 / 15 nm (measured 11 / 11 / 7 nm); Karnopp's set-valued stick drives the limit cycle
  (0.8 / 0.7 / 4.6 um; still 0.4 / 0.4 / 1.0 um with a 20x narrower band); tanh settles to
  12 / 12 / 8 nm. The oscillation amplitudes sit inside the pre-sliding range, where the rail is
  not a Coulomb relay. Side finding: the zpk file closes an UNSTABLE loop with this plant
  (|lambda| 1.0031 at 276-282 Hz), which explains the 70821 closed-loop divergence.
- **Attempt 2** (TR-015): cc = held friction 82.6 / 100.3 / 80.5 N, tanh stick zone (v0 1e-3;
  identical to Karnopp's slip branch while sliding), linear parameters re-fitted.
  (a) **PASS**: 0 / 159 unstable in either form, no limit cycle.
  (b) **PASS**: held-out median NRMSE_direct **1.95** against **4.90** for 70821 (same loop, same
  code). Held-out iter0 (feedback only): **0.20 / 0.15 / 0.38** (70821: 0.35 / 0.38 / 0.59).
  (c) **PASS**: ten combinations positive, min eig M(Y) 5.44 over +-0.25 m, min |d(Y)| 1.05e4,
  cc within the held-friction bound (86.4 / 113.9 / 90.3 N; met by construction, see TR-015),
  X-moving mass 90.9 kg (-0.1 % vs datasheet), mh 20.95 kg (+10 %).
- **Recovered parameters vs datasheet** (reading A): m_total + mh 90.9 vs 91.0 kg, mh 20.95 vs 19;
  cg1 / cg2 / cy 310 / 322 / 308 N/(m/s) against datasheet MAXIMA 136 / 136 / 98 (2.3-3.1x):
  the viscous terms absorb the one-way-move Coulomb excess and whatever else is linear in
  velocity (cable carrier, seals); not separable with this excitation. Theta-row quantities
  (J_eff, d, kb, cb) are not identifiable from these moves (relative SE 0.3-1.8) and stay at the
  Garcia values.
- **References**: frictionless datasheet parameters (nf) held-out NRMSE_direct 2.21, residual
  form 1.92 (attempt 2: 1.98), iter0 0.21 / 0.15 / 0.40. Friction and recovery improve the
  direct form modestly (2.21 -> 1.95) and leave the residual (training) form about unchanged:
  the rigid-body + friction structure is near what these moves can identify.
- **Residual against the noise floor** (attempt 2, residual form, PSD / standstill floor):
  1-20 Hz 1.2e5 / 7.5e4 / 3.9e4; 20-100 Hz 1.4e4 / 8.4e3 / 3.4e4; 100-300 Hz 1.4e3 / 1.7e3 / 3.0e4;
  300-1000 Hz 75 / 135 / 1042; above 1 kHz at the floor (2.0, 1.4). The residual is far above
  the floor below 1 kHz on all axes, largest relative to the measurement on Y.
- **Deviation from the handoff, flagged**: the baseline's friction is the D-116 tanh form, not
  Karnopp. Karnopp is ported and verified (G2) and was tried first; on real data its stick branch
  produces a limit cycle the machine does not have (measured above). The sliding branch is
  identical.

## G5 Learnability: PASS, verdict GO ("server run warranted: yes")
Metric and thresholds pre-registered in TR-017 (from a `deep-research` literature run); computed by
`learnability/g5_learnability.py` (run g5_learnability) on the closed-loop residual
`y_model - y_data` of the G4 attempt-2 baseline in the training (residual) form.
- **M1, the ceiling** (repeatable residual power; pintelon2012sysid eqs. 2-36/2-37 p50,
  schoppe2016measuring eqs. 14-15): the 15 iter0 records (and the 15 iterETEL records) are exact
  repeats of one reference profile at 15 positions (reference deviation 0.011 um). Repeatable /
  non-repeatable power **X1 4.92, X2 5.54, Y 4.60** (ETEL group 5.20 / 5.92 / 3.70), criterion
  >= 1; 242-827 FDR-significant bins per axis and group. Repeatable residual 346 / 339 / 255 nm rms
  against 156 / 144 / 119 nm non-repeatable. Conservative: position-dependent structure counts as
  non-repeatable here.
- **M2, the go decision** (cross-validated model-error model, ljung1999mem eqs. 12/24, on external
  regressors only per forssell1999closed's closed-loop rule): FIR on the reference acceleration and
  velocity, logged feedforward and Y x acceleration, fitted on train, scored on the 42 held-out
  records. Headroom **H = 4.80 / 5.40 / 3.13** (explained held-out power over the non-repeatable
  power; criterion >= 1), p < 3e-24, held-out R^2 median **0.89 / 0.85 / 0.71**. Against the
  standstill floor instead (9.8 / 10.4 / 6.3 nm) H is ~1000-1200: the standstill floor would
  massively over-declare learnability, which is why M1's in-motion variation is the denominator.
- **Answer to the handoff's open question (section 8)**: (a). The residual is far above the noise
  floor, repeatable across positions and predictable on held-out records from exogenous signals
  alone. It is not a force-scale error in disguise (a pure scale error would be absorbed by the
  recovered masses, which match the datasheet within 0.1 % on X) and not the controller path (the
  identified controller reproduces MF230 within 0.1-0.4 % on held-out records).
- **ANN settings from the residual** (TR-017 rules, TR-018): routing = velocity rows of X, Theta, Y
  (all pass M1 and M2) + the augmented rows -> `ann_route_ix = [3, 4, 5, 6, 7]`; nx_ann = 2 (the
  order test was inconclusive: slow singular-value decay, no gap, under the noise threshold, so 2 is
  the rule's floor, not an estimate); horizon 40 ms (lower bound: the impulse response has not
  decayed at the 20 ms or 40 ms window edge); band of repeatable residual 3 Hz-2.5 kHz. Written to
  `learnability/ann_settings.json`. None of these come from the simulated 212 Hz absorber.
- **M3** (descriptive): the X residual correlates with the reference acceleration (max |xcorr| 0.50),
  the Y residual much less (0.15): the Y residual is mostly not a linear response to its own
  acceleration, consistent with Y's lower R^2.
- **M5** (descriptive, TR-017; run g6_probe, forward + backward only, no optimizer step): over 32
  train windows at the 40 ms horizon, the gradient noise scale is B_simple = 6.7
  (mccandlish2018empirical eq. 2.9), far below the server batch of 256, so the gradient direction
  is well estimated; per-parameter GSNR median 0.20 (ANN 0.49; liu2020gsnr notes GSNR starts low
  at initialisation). Only 75 of the ANN's 1613 weights receive gradient at initialisation: the
  zero-initialised output rows of the two augmented states have no path to the loss until the
  first update (the known W^a dead zone, D-130). Not a blocker; noted for the server run.

## G6 Pipeline ready: PASS
Acceptance TR-019 (+ TR-021 for the evidence reading). Runs g6_norm, g6_smoke, g6_probe, g6_check.
- **The seven section-4 items, adapted** (`pipeline/telica_augment.py`, checked by
  `pipeline/check_g6.py`, 11/11 PASS):
  (1) controller: the identified Telica controller as a one-row `ControllerBank`, residual form,
  float64, refuses any step but 20 kHz; (2) parameters: `params/` + the G4 attempt-2 baseline
  (tanh friction), nothing hard-coded; (3) normalisation frozen from the train split with
  filtered velocities (`pipeline/norm_frozen.json`, TR-016); (4) no decimation: 20 kHz, and a
  4 kHz config is refused by the data layer; (5) no ground-truth diagnostics: the entry point
  calls none, and the vendored training wrapper now skips the absorber-GT diagnostic without GT;
  output-only evaluation (augmented vs ANN-zeroed baseline, per record, metres); (6) the split
  by operating point from `data_index.py`, records via the loader; (7) ANN settings from G5
  (`nx_ann 2`, rows `[3, 4, 5, 6, 7]`, horizon 40 ms).
- **Smoke test** (g6_smoke): 2 train / 2 val / 2 test records (300 ms each), batch 8, one 5 ms
  window, CPU, float64: ran end to end, exit 0. Exactly **2 optimizer updates** (batch_counter 2
  at deepSI's `_last` checkpoint), the ANN moved (max weight change 2.0e-5, `_last` vs `_best`),
  closed-loop validation ran (3.82e-7 m before, 8.04e-7 m after, so deepSI kept the initial
  weights as best), output-only evaluation completed. **Peak RAM: 676 MB** process-tree working
  set (watchdog, 5 s sampling); the in-process counter of that run read 0 (ctypes bug, fixed
  after). The probe (same model, server horizon, forward + backward only) peaked at **1142 MB**
  in-process (1214 MB tree).
- **Server runner** `pipeline/runners/server_run.sh` (NOT submitted). Memory estimate (TR-020):
  80.3 KB per window-step measured, so batch 256 x 800 steps is ~16.4 GB unchunked and ~4.1 GB
  with `checkpoint_chunk = 200`, plus ~2.7 GB of host window arrays; 64 GB host, one GPU, 24 h,
  50 epochs. Runtime on GPU is not measured here (no inductor toolchain on this PC); if the
  compile fails, `COMPILE_MODE=none` runs eager.
- deepSI's automatic checkpoints are redirected into the run folder (they default to
  `%LOCALAPPDATA%\deepSI`), so the pipeline writes nothing outside `telica-real/outputs/`.

## 5 kHz training rate (TR-022, 2026-09-23): PASS, server default
Requested by the user to cut server cost. Rule: positions and reference anti-aliased (Butterworth 8,
2 kHz, zero-phase) and point-sampled, forces and currents block-mean per hold interval (`rate.py`).
Pre-registered acceptance (A)-(D) in TR-022; runs g3_5k, g3_5k_a2, g3_bank_5k,
g4_replay_rec_a2_5k, g6_check_5k(_final), g6_probe_5k.
- **(A) controller at 5 kHz, 2 attempts**: attempt 1 met the gain interval but picked lag -1 (the
  current leading the error), an alignment artifact of block-mean current against point-sampled
  error, not causal and so unusable in the loop. Attempt 2, causal lags only: lag 0, held-out gain
  **0.9988 / 1.0025 / 1.0013** (intervals 1 +- 0.0030 / 0.0088 / 0.0044), held-out VAF
  0.998 / 0.998 / **0.9996** (Y fits better than at 20 kHz, 0.940).
- **(B)** residual form returns u_data exactly (0.0); units 1.5e-11 (float64). float32 fails as at
  20 kHz (input rounding), irrelevant for the float64 pipeline.
- **(C)** G4 attempt-2 baseline at Ts = 2e-4 s: stable on 159/159 in both forms; held-out
  NRMSE_direct **1.947** (limit 2.147; 20 kHz 1.952), residual form **1.963** (limit 2.180);
  iter0 held-out 0.199 / 0.148 / 0.333 (20 kHz 0.198 / 0.148 / 0.376); 57 s against 208 s.
- **(D)** static checks 11/11 at 5 kHz; forward + backward at nf 200 without an optimizer step;
  81 KB per window-step; in-process peak 697 MB.
- **Server**: 40 ms horizon = 200 steps; batch 256 -> ~4.1 GB graph, no checkpointing; stride 5.
  `FS_TRAIN=20000` restores the 20 kHz path.
- Machine note: two launches were blocked first, by RAM (1.9 GB available) and then by disk
  (C: 1.27 GB after the Windows page file grew to 9.3 GB under a 22.9 of 24.9 GB commit load from
  other applications). Both cleared after the browser was closed. This page-file growth on a
  nearly full C: is a plausible mechanism for the earlier overnight window closures.

## Friction on vs off, same parameters (user question, 2026-09-23)
Attempt-2 recovered parameters at 5 kHz, identified controller, all 159 records (run
g4_replay_rec_a2_nf_5k against g4_replay_rec_a2_5k).

| | With friction (tanh) | Friction off |
|-|-|-|
| Held-out NRMSE, direct form (median) | 1.947 | 1.966 |
| Held-out NRMSE, residual (training) form | 1.963 | 2.007 |
| Held-out iter0, X1 / X2 / Y | 0.199 / 0.148 / 0.333 | 0.214 / 0.163 / 0.370 |
| Held-out force NRMSE, X1 / X2 / Y | 0.179 / 0.205 / 0.228 | 0.206 / 0.218 / 0.266 |

Friction lowers the median errors by 1-2 % overall, 7-10 % on the feedback-only iter0 records and
6-14 % on the force; the residual-form PSD averaged over records is mixed (below 100 Hz the direct
form has LESS residual power without friction, carried by the large-error records). Caveat: the
viscous terms were recovered WITH friction in the model, so this isolates the friction term, not a
refit without friction.

## Absolute RMS of the baselines (run g4_rms_table, 2026-09-23)
Held-out median, rms(simulated - measured servo error) [nm], X1 / X2 / Y.

| Baseline | Direct form, all iter. | Training (residual) form, all iter. | Direct form, iter0 |
|-|-|-|-|
| Recovered + friction, 5 kHz (server default) | 421 / 359 / 236 | 367 / 371 / 200 | 364 / 359 / 208 |
| Same parameters, friction off, 5 kHz | 428 / 360 / 234 | 380 / 383 / 208 | 392 / 399 / 231 |
| Recovered + friction, 20 kHz | 423 / 360 / 278 | 368 / 374 / 263 | 362 / 360 / 235 |
| Datasheet parameters, no friction, 20 kHz | 406 / 386 / 300 | 376 / 351 / 283 | 382 / 378 / 249 |
| 70821, no friction, 20 kHz | 682 / 998 / 434 | 622 / 905 / 397 | 647 / 930 / 370 |
| Measured servo error itself | 142 / 195 / 93 | | 1842 / 2436 / 627 |

## Server configuration update (TR-024, 2026-09-23)
nx_ann = 8 (user decision; G5's order test was inconclusive), routing X, Theta, Y velocity rows + 8
augmented rows, encoder window 29 samples. `pipeline/runners/server_run.sh` follows the user's working
GPU runner (partition oahu, existing log folder, `srun --cpu-bind=cores`, toolchain and GPU
diagnostics). Checked without an optimizer step (runs g6_check_5k_nx8, g6_probe_5k_nx8): static
checks 11/11, forward + backward at nf 200 runs, 81.7 KB per window-step, so batch 256 needs ~4.2 GB
of graph. Open risk: float64 speed on a consumer GPU (~1/32 of float32); the log prints the card.

## Projection arms ready (TR-025, 2026-09-23)
Joint estimation of the ten identifiable combinations plus trainable Coulomb levels (13 directions),
starting at the G4 values, no prior; cc learns only while sliding (stick-zone mask). Three arms via
`OBC_ARM` = noproj / obc / obc_affine in `pipeline/runners/server_run.sh`. Gates without any optimizer
step (run g6_joint_a2), all PASS: friction tangent vs jvp 2.3e-16; exact primal; mask exact per
evaluation (the unmasked stick-zone share of the cc gradient would be 5.6 / 11.5 / 3.5 %);
OBC orthogonality 2.2e-13; per-step correction == jvp 6.4e-17; all three arms build and run
forward + backward. Basis at G4: rank 13/13 (cond 6.1e4; kb_sum and cb_sum barely excited),
affine 14/14. To compare the arms afterwards: held-out output error (augment_metrics.json), and the
drift of the 13 parameters from G4 against masses 91 / 19 kg and held friction 86 / 114 / 90 N.
