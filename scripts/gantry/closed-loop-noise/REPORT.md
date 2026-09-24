# closed-loop-noise report

```
G4 Closure (Telica loop): FAIL by rule on (c), closure itself PASS. Simulated Phi_e inside the 95 % CI
  in 97.7 / 95.3 / 95.3 % of bands (>= 85 %, CN-008); rms 9.840 / 10.390 / 6.334 nm vs measured
  9.843 / 10.390 / 6.336 (tol 4.1 / 4.1 / 3.6 %). Analytic = simulated to 0.04 % in rms (MATLAB covar
  9.85 / 10.39 / 6.34), but only 93 / 88 / 88 % of bands inside the simulation's +-1.3 % CI (>= 90 %):
  systematic misses <= 1.8 %, three attempts, FAIL stands.
G5 Generator: PASS. Noise-off bitwise identical on T3, TP1 (17/17 fields). Dataset ..._noise_ss written
  (29 records, 1.4 GB, double). Friction off: generator noise part of e = S_sim Phi_v S_sim^H,
  11.05 / 11.52 / 6.78 vs 11.05 / 11.56 / 6.78 nm, 93 / 93 / 95 % of bands. With Karnopp friction it
  is NOT linear: held rails turn the integrator into a random walk (std x2.6-6.3 between record
  halves in E1 and 6 of 7 profile records) and move the true position by 38-1288 nm there. Stick fraction
  (1 mm/s proxy) unchanged to 0.1 %; peak force changes <= 0.36 N (limits not approached).
G0 PASS (8.69 / 10.63 / 5.94 nm reproduced). G1 PASS (145 logs, rms closure 0.15 %). G2 FAIL after 3
  attempts: the identified plant misses cross-axis coupling (empirical off-diagonal S 2-14x the model)
  and the Y path. G3 PASS: source 12.8 / 13.5 / 9.0 nm (tanh reading, CN-007 deviation), VAR(512)
  shaping filter; white-encoder part <= 4.8 / 4.8 / 4.5 nm. G6 done (formula table, recipe, training).
Resources: peak RAM 4.03 GB (MATLAB, g5b_twin), min available 1.77 GB, C: min free 6.17 GB,
  2 watchdog kills (MATLAB compile at the 2.5 GB line, before CN-010), 0 training updates.
Next for the user: (1) decide whether the low-frequency part of v (< 300 Hz) is encoder noise or a
  0.1-0.5 N force disturbance (not separable from these logs; it changes the generator's error by 2x
  below 300 Hz); (2) before training on the twin: anti-alias y before y[::D] and keep float64
  (G6 items 2 and 4); (3) the Telica plant model needs cross-axis coupling (G2) before its S is
  trusted above the diagonal.
```

The standstill plant reading (sliding, tanh, stuck) decides the source below 300 Hz by up to three
decades (G3). The production choice, tanh, departs from the pre-registered default (CN-007). Every
table below gives it first.

## G0 Infrastructure and literature: PASS
Criteria CN-001 to CN-003. Runs g0_lit_list, g0_lit_write, g0_check, g0_check2.
- Folder, `tools/watchdog.ps1` (copied unchanged from telica-real, TR-002 thresholds),
  `tools/run.sh`; every run of this session has a `RUNS.md` row and went through the watchdog
  (34 runs; MATLAB runs with the CN-010 RAM thresholds). Not wrapped, stated for completeness: three
  shell one-liners with the system Python, two that printed numbers from an existing `g1.json` /
  `g3.json` and one that did the path substitution in the vendored `tr_env.py` and loader. None read
  Telica data or computed a result.
- `LITERATURE.md`: the six selected messages of transcript 8bf42294 (SQ1, SQ4, SQ2, the synthesis,
  and the answers of 2026-09-22 16:49 and 17:11 UTC; the handoff's times turned out to be UTC).
  All ten methods of the register are findable by name.
- Vendored loader (`vendor/telica_real/.../telica_loader.py`, `pre_motion_ms=None`) on
  `train/xpos_-60_ypos-40/iter0`: standstill std **8.6914 / 10.6256 / 5.9395 nm** against the quoted
  8.69 / 10.63 / 5.94 (|diff| 0.0014 / 0.0044 / 0.0005 nm, criterion < 0.005); the default call still
  trims to 50 ms. Attempt 1 missed (up to 0.0097 nm) because the window used the loader's motion
  index, which fires 19 samples after the first setpoint change (0.5 um threshold vs sub-micron
  first steps); every window in this folder now uses `j0` = first `diff(M0) != 0`.

## G1 Measured spectra: PASS
Criteria CN-004. Run g1_spectra2 (g1_spectra crashed after the spectra, see RUNS.md).
- **Feedforward before motion is NOT zero** in the ILC logs: in every one of the 130 ILC logs
  (iter1 to iter10, iterTEST, iter6_1) the feedforward starts 256 to 258 samples (12.8 ms) before the
  first setpoint change (a non-causal ILC); in the 15 iterETEL logs it is nonzero from the first
  sample. Handled by the pre-registered rule: the window ends 200 samples before the feedforward
  onset; the ETEL logs are dropped (window < 2048). `iter6` of `xpos_-60_ypos120` is empty.
  Result: **145 logs pooled**, K = 908 Welch segments, nu = 1720, per-bin 95 % CI factor
  [0.936, 1.070]; no log failed the stationarity screen.
- **Closure**: rms from the pooled PSD **9.848 / 10.400 / 6.338 nm** against the time-domain
  9.838 / 10.415 / 6.340 nm: **+0.10 / -0.15 / -0.03 %** (criterion 2 %). Per-log rms 10-90 %
  range 7.95-11.2 / 8.56-11.9 / 5.75-6.82 nm.
- **Shape**: 91-98 % of the variance above 200 Hz; peak PSD 41 / 43 / 39 dB above quantisation
  (7.9e-24 m^2/Hz); a flat floor of ~2.5e-21 m^2/Hz above 5 kHz on all axes; sharp lines at
  100, 300, 500 Hz (odd multiples of 100 Hz, mains related) and near 1.3, 2.4 and 4.3 kHz.
  Band rms [nm], X1 / X2 / Y: 10-100 Hz 1.83 / 2.36 / 0.65; 200-500 Hz 4.04 / 4.03 / 1.17;
  500-1000 Hz 4.19 / 4.48 / 2.05; 2-5 kHz 5.80 / 5.91 / 3.06; 5-10 kHz 3.44 / 3.48 / 3.26.
- **Cross-axis coherence** is significant (null 1 % threshold 0.0053): X1-X2 up to 0.82 (400-700 Hz,
  and 0.44 at 100 Hz), X2-Y up to 0.70 (1.3 kHz); variance-weighted 0.22 (X1-X2), 0.07, 0.08.
- **Band CIs for G3 to G5**: 43 bands 19.5 Hz to 10 kHz; record-bootstrap 95 % half-width median
  7.7 % (max 37 %).
- **Non-repeatable motion content** (15 exact repeats, 340 ms from motion start, measured error):
  iter0 **200 / 211 / 87 nm**, iterETEL 149 / 157 / 92 nm, i.e. 15-20x the standstill level on X,
  concentrated below 300 Hz (figure panel 4), converging to the standstill spectrum above ~1 kHz.
  This is the motion upper bound of section 8 of the handoff (it includes position-dependent
  structure). Figure: `outputs/g1_spectra/g1_spectra.png`.

## G2 Sensitivity: FAIL after three attempts (mechanism named)
Criteria CN-005, amended CN-005a and CN-005b before attempts 2 and 3. Runs g2_sensitivity2,
g2_sensitivity3, g2_sensitivity4; diagnostics g2_diag_sg, g2_diag_el.
- **Model S** (identified controller, attempt-2 plant, 20 kHz, 15 positions). Sliding reading:
  |S_ii| peaks 2.29 / 2.34 / 6.35 at 400 / 410 / 312 Hz; off-diagonals are large (|S_X1X2| up to
  1.92, |S_YX1| up to 2.10, near 300-400 Hz). tanh-at-rest reading (rail damping cc/v0): peaks
  1.15 / 1.11 / 2.05, off-diagonals below 0.14. The least-damped closed-loop mode, 183-186 Hz with
  zeta 0.009-0.011, belongs to the identified controller (it is there with G = 0 as well), not to
  the loop. The per-axis `|S_ii|^2` de-shaping does NOT suffice: it stays inside the G1 CI in only
  51-63 % of bands (sliding) and 84-100 % (tanh), so G3 uses the full MIMO form.
- **Empirical S** from the ILC feedforward (160 logs, 349 ms from 20 ms before motion; 93-99 % of
  the feedforward-difference energy lies in the motion phase): coherent (multiple coherence
  >= 0.9) only **below 206 / 209 / 232 Hz**; above that the feedforward carries no power.
- **Agreement** (sliding reading) in the 504 coherent (bin, element) pairs: 15.5 % (attempt 1,
  statistical tolerance), 24.4 % (attempt 2, + controller-identification floor), **32.1 %**
  (attempt 3, + the model's validated accuracy) against the 80 % criterion. tanh reading: 9.5 /
  10.9 / 14.1 %.
- **Mechanism, element by element** (attempt 3): `S_X1X1` agrees in 77 % of coherent bins
  (|emp|/|model| median 0.97); `S_X2X2` 47 % (1.74); `S_YY` 16 % (2.19). The **off-diagonal**
  elements carry the failure: the median empirical cross-sensitivities are **2.0-4.0x** the model
  in the X rows (`S_X1X2`, `S_X1Y`, `S_X2X1`, `S_X2Y`; 9-46 % agreement) and **11-14x** in the Y
  row (`S_YX1`, `S_YX2`, 5-21 % agreement). On the well-conditioned process sensitivity, `SG_emp / SG_model` is 0.93-1.08 on the
  X diagonals below 150 Hz but 0.35-0.73 on Y. Named mechanisms: (1) **cross-axis coupling the
  identified plant does not contain** (the rigid-body LPV model couples the axes only through
  M(Y); the machine shows X-to-Y coupling an order of magnitude larger, consistent with the
  significant standstill X1-X2 and X2-Y coherence in G1); (2) **the Y path** (plant or the known
  Y controller mismatch of telica-real G3, VAF 0.94); (3) the low-frequency cancellation floor
  of `I - SG K` (attempt 1 to 2). Not the mechanism: post-motion settling (refuted, 93-99 % of
  the excitation is in motion).
- **What G2 cannot settle**: the empirical S exists only below ~230 Hz and only while sliding. The
  standstill S above 230 Hz (where 91-98 % of the noise variance sits) is model-only. The
  pre-motion feedforward window was too weak to pick the standstill plant (response 13.3 / 14.2 /
  13.0 nm against 9.8 / 10.4 / 6.3 nm noise, below the 3x rule), so the production reading is
  **sliding** by the pre-registered default (superseded by CN-007: production reading tanh, see G3).
  The weak evidence leans the other way (median NRMSE
  sliding 1.13-1.26, i.e. worse than predicting zero; tanh 0.74-0.90; stuck 1.0), so G3 carries
  tanh as the alternative reading. Figures: `outputs/g2_sensitivity/g2_sensitivity.png`,
  `outputs/g2_diag_sg/sg_emp_vs_model.png`.

## G3 Source spectrum and noise model: PASS on attempt 2 (production reading tanh, CN-007)
Criteria CN-006, CN-006a; production reading changed from sliding to tanh by CN-007 (a flagged
post-data deviation, reasons there). Runs g3_source (attempt 1 FAIL), g3_source2.
- **De-shaping** per log with the full 3x3 S at the log's position (G2 showed the per-axis form is
  not enough). Source rms over 19.5 Hz-10 kHz, against the measured error 9.84 / 10.39 / 6.34 nm:

  | reading | v rms [nm] | v / e | 19.5-100 Hz | 100-300 Hz | 0.3-1 kHz | 1-3 kHz | 3-10 kHz |
  |-|-|-|-|-|-|-|-|
  | **tanh (production)** | **12.75 / 13.48 / 9.03** | 1.30 / 1.30 / 1.43 | 8.56 / 9.07 / 6.57 | 3.26 / 3.52 / 0.93 | 5.05 / 5.35 / 2.42 | 4.23 / 5.13 / 4.03 | 5.93 / 5.68 / 3.93 |
  | sliding | 207.7 / 239.6 / 222.4 | 21 / 23 / 35 | 207.5 / 239.3 / 222.3 | 6.10 / 6.47 / 2.47 | 3.42 / 3.60 / 2.15 | 4.17 / 5.06 / 3.76 | 5.93 / 5.68 / 3.93 |
  | stuck (S = I) | 9.84 / 10.39 / 6.34 | 1 | 1.83 / 2.36 / 0.65 | 2.83 / 2.85 / 0.87 | 5.44 / 5.65 / 2.29 | 4.49 / 5.44 / 4.22 | 5.98 / 5.73 / 3.99 |

  Above ~300 Hz the three readings agree within ~15 % (S is close to I there); below 300 Hz they
  differ by up to three decades in PSD, because `Phi_v = Phi_e / |S|^2` squares a small and
  model-dependent |S|. **The source above 300 Hz is determined by the data; the source below
  300 Hz is determined by the assumed standstill plant.**
- **S-only bound on a white encoder noise**: **4.77 / 4.84 / 4.47 nm rms** (density 2.3 / 2.3 /
  2.0e-21 m^2/Hz, set by the flat 7.5-9.4 kHz floor), identical for all readings because S = I at
  those frequencies. That is 17x the quantisation level (0.28 nm): the encoder path has a white
  floor well above quantisation (interpolation, read channel), as the 09-22 register expected.
- **Interpretation B (all input disturbance, `v = G d`)**, force rms per band, tanh reading
  [N]: 19.5-100 Hz 0.14 / 0.20 / 0.08; 100-300 Hz 0.40 / 0.52 / 0.11; 0.3-1 kHz 3.6 / 4.2 / 1.7;
  1-3 kHz 30 / 34 / 13; 3-10 kHz 2000 / 2100 / 890 (sliding reading similar below 1 kHz, 9e4 N
  above 3 kHz). A force disturbance of a few tenths of a newton below 300 Hz is physically ordinary
  (cable carrier, air, force ripple); tens to thousands of newtons above 1 kHz are impossible. So
  the data themselves split the source: **above ~1 kHz v is sensor noise; below ~300 Hz a force
  disturbance of 0.1-0.5 N explains it equally well** (not separable from this data, the 09-22
  hard negative; this is where the two interpretations matter).
- **Cross-axis correlation of v** (variance-weighted coherence, tanh): X1-X2 0.143, X1-Y 0.051,
  X2-Y 0.090. Above the 0.1 rule on X1-X2, so the shaping filter is a correlated VAR.
- **Shaping filter H_v**: attempt 1 (orders <= 256) FAIL, best 91 / 79 / 81 % of bands inside
  (tanh); mechanism: an all-pole model of that order cannot follow the sharp 100 / 300 / 500 Hz and
  1.3 / 2.4 / 4.3 kHz lines within ~8 % band CIs. Attempt 2: **VAR(512), bands inside the Phi_v
  bootstrap CI 100 / 97.7 / 93.0 %** (criterion 90 %), **rms -1.42 / -1.47 / -0.47 %** (criterion
  2 %), largest root modulus 0.99821 (stable). Files `outputs/g3_source/hv_tanh.{npz,mat}` (A, Sigma;
  `v_t = sum A_i v_{t-i} + chol(Sigma) w_t`, w unit white per 50 us sample). The stuck reading also
  passes at 512; the sliding reading passes no order (at 512 the bands pass but the rms is +14-21 %,
  its 19.5-100 Hz content is extrapolated and dominates). Below 19.5 Hz Phi_v is held at the first
  band (extrapolation, not data). Figure `outputs/g3_source/g3_source.png`.

## G4 Closure on the Telica loop: FAIL by the pre-registered rule on (c) after three attempts; the closure against the measurement, (a) and (b), PASSES
Criteria CN-008, CN-008a, CN-008b. Runs g4_closure_tanh, g4_expwelch, g4_closure_tanh3 +
g4_expwelch3. Production reading tanh; v from H_v (VAR 512) at the encoder node of the identified
controller + identified plant, 15 positions x 2 seeds x 16.5 s, nonlinear RK4 block (tanh friction).
- **(a) PASS**: simulated band means inside the measured 95 % CI in **97.7 / 95.3 / 95.3 %** of the
  43 bands (criterion 85 %).
- **(b) PASS**: simulated rms **9.840 / 10.390 / 6.334 nm** against measured **9.843 / 10.390 /
  6.336 nm** (19.5 Hz-10 kHz): **-0.02 / -0.01 / -0.02 %** (tolerance 4.1 / 4.1 / 3.6 %).
- **(c) analytic = simulated**: rms **-0.04 / -0.02 / 0.00 %** (criterion 1 %, met); bands: 74-77 %
  (attempt 1, raw analytic against a Welch estimate: Hann leakage of the sharp lines across band
  edges), **93.0 / 88.4 / 88.4 %** (attempt 2, analytic passed through the Welch expectation) and the
  same with a block-bootstrap CI (attempt 3), against 90 %. The remaining misses are systematic
  (identical on identical seeds) and small: at most 1.8 % in the 263, 83, 376, 234 and 532 Hz bands,
  against a simulated CI of about +-1.3 %. Mechanism not isolated below the 2 % level (candidates:
  the per-segment mean removal, which the expectation formula ignores; the Jacobian linearisation
  of the RK4 step with tanh friction, stage speeds 4e-5 m/s << v0). FAIL stands by the rule.
- Side numbers: injected v 15.0 / 15.9 / 11.0 nm rms over the full band, of which the part below
  19.5 Hz is the held-flat extrapolation of G3 (the loop suppresses it); noise-driven force 1.8 /
  2.4 / 2.2 N rms; controller-state std at most 1.3e-2 (bounded, stationary).
- Limitation stated in CN-008: (a) and (b) test the chain (inversion, filter, node, sign,
  simulator), not S itself; the independent test of S is G2 (FAIL). Figure
  `outputs/g4_closure_tanh/g4_closure.png`.

## G5 In-loop injection in the generator: PASS ((a) bitwise, (b) written, (c) on attempt 2, (d) reported)
Criteria CN-009, CN-009a, CN-010. Runs g5_inspect2, g5a_noiseoff2, g5b_twin, g5c_ref, recipe,
g5_check (attempt 1), g5_check_noE1 (diagnostic), g5c_cc02, g5_check_cc0 (attempt 2).
- **Which code.** The production records predate D-209: T3 (2026-09-21) and TP1 (2026-09-20) were
  written with the OLD stick band; D-209 landed 2026-09-22 20:19. The generator copy therefore
  uses the friction ODE of `ae25789^`; everything else is the working tree
  (`matlab/VENDORED_MATLAB.md`).
- **Node.** The model runs four loops; the generator records only the "Extended ODE" loop. The one
  edit (`matlab/gen/make_noise_model.m`): `Gain4 (P^T q) -> [+ v_enc] -> Sum3 (r - y)`, v as
  sample-and-hold at the controller's rate (Cfb is discrete, Tustin, with an integrator). The
  recorded `y` is the measured `q + v`; `q_true`, `v_enc` are stored; `u_fb = lsim(Cfb, r - y)`.
- **(a) PASS**: noise off through the edited model reproduces `T3_standstill_Y000` and
  `TP1_telica_y000` **bitwise in all 17 production fields**. This also confirms the pre-D-209
  provenance of the production records.
- **(b)**: `data/gantry/matlab/trajectory/augmentation_ma50_z03_b140-230_a6all_telica_coulomb_noise_ss/`,
  29 records (same names and multisine seeds as production), **1.4 GB** (double precision, see the
  precision finding below; production 0.525 GB in single), plus `figures/` and `NOISE_README.md`.
  Noise per record: VAR(512), seed = crc32(id), 15.0 / 15.9 / 10.8 nm rms (full band, includes the
  extrapolated part below 19.5 Hz). C: 10.4 -> 9.0 GB.
- **(c)** noise part of the error, `e_noisy - e_off = -(q_true + v - q_off)`, against
  `S_sim Phi_v S_sim^H` (`S_sim = feedback(eye(3), G_sim Cfb)`, frictionless 8-state linearisation
  with the hidden absorber, c2d zoh, computed in `matlab/recipe_sensitivity.m`):
  - Attempt 1, on the dataset itself (8 standstill-class records): FAIL, 67 / 70 / 56 % of bands,
    rms 121 / 137 / 451 nm against 11.1 / 11.6 / 6.8 nm. Mechanism: Karnopp friction. E1 (the sine
    sweep, stuck on both X rails, as in production) runs with the loop effectively open: the noise
    part is 448 / 519 / 2612 nm and the integrator state grows (std 11 -> 52 between halves).
    Without E1 (diagnostic): 72 / 77 / 56 %, rms +12 / +12 / +31 %, the excess at 24-83 Hz and, on
    Y, at the 210 Hz multisine band (+2900 %): the noise shifts the stick-slip timing at multisine
    reversals and the large multisine response carries that shift.
  - **Attempt 2 (CN-009a), friction off (cc = 0) on T1, T3, T5, V1: PASS**, 93.0 / 93.0 / 95.3 % of
    bands inside the CI (criterion 85 %), **rms 11.047 / 11.521 / 6.784 nm against the analytic
    11.048 / 11.558 / 6.777 nm (-0.01 / -0.32 / +0.10 %)**, per-record covar 11.0-11.1 / 11.5-11.6 /
    6.8 nm. The injection node, sign, sampling and the generator's loop are exactly the linear
    math. With friction the loop is not linear, and the deviation above is the measured size of that.
- **(d) reported**:
  - *Integrator*: in the linear or sliding loop the controller state is stationary (friction off:
    std 0.093 -> 0.095 between record halves; sliding standstill records 0.25-0.41, no trend). When
    Karnopp holds a rail, S = I there and the integrator integrates v: the std grows between halves
    in E1 (11 -> 52) and in 6 of 7 Telica-profile records (e.g. 3.4 -> 8.8, 27 -> 171). So the
    09-22 claim "white noise into the integrating controller is a random walk" is **wrong in the
    sliding loop and right while stuck**. Consequence: in the Telica-profile records the noise moves
    the TRUE position by **38 to 1288 nm rms** (the held rails break away at shifted times).
  - *Stick fraction* (the drivers' proxy, |rail velocity| < 1 mm/s from the true position): identical
    to 3 decimals with and without noise on all 15 checked records (T 5-14 %, E1 100 / 100 / 27 %,
    TP 70-88 %). The proxy cannot see noise-level velocities (~4e-5 m/s); the Karnopp stuck state
    itself is not logged by the ODE, so chatter inside the band is not measured.
  - *Headroom*: peak |u_total| over 29 records 1570 / 1717 / 869 N with noise against 1570 / 1717 /
    869 N without (largest change per record 0.36 / 0.33 / 0.10 N), limits 2000 / 2000 / 1420 N;
    rms at most 405 / 424 / 214 N against 916 / 916 / 656 N. Noise-driven feedback force 0.12-0.18 /
    0.13-0.18 / 0.04 N rms while sliding, up to 5.6 N on held rails.
- **Precision (finding)**: the production generator stores signals as float32. The rounding it
  would add to y on these records is 0.0003 / 0.0003 / 4.1 nm rms (X rails near 0 m, Y up to
  0.3 m), against a noise part of y of ~11 / 12 / 7 nm, so the twin is double throughout.
  Figures: `outputs/g5_check/g5_check.png`, `outputs/g5_check_cc0/g5_check.png`.

## G6 Report for ASMPT and for the training pipeline: done
Numbers from runs recipe (MATLAB) and g6_numbers.

**The formula table** (Ljung and Forssell 1999 Eqs. 13/15; `S = (I + G K)^-1` MIMO in stage axes;
`y_meas = q + v`, `e = r - y_meas`, `u = K e`):

| quantity | node | on Telica (measured) | in the generator (injected) |
|-|-|-|-|
| source v = n + G d | encoder node: the position the controller sees | not directly; `Phi_v = S^-1 Phi_e S^-H` | v from H_v (VAR 512) at that node |
| error e | controller input | `Phi_e = S Phi_v S^H` (standstill logs) | `-S_sim v` (the twin's noise part of y is `S_sim v`) |
| feedback force | controller output | `-K S v` (rank 1 with e: no extra information) | `-Cfb S_sim v`, in u_fb |
| true position | stage | not observable | `-T_sim v`, in q_true |

Per band, nm rms, X1 / X2 / Y:

| band | measured e (Telica) | source v, tanh (sliding) | Telica e, v injected | Telica e, MEASURED Phi_e injected (S twice) | generator e, v | generator e, Phi_e |
|-|-|-|-|-|-|-|
| 19.5-100 Hz | 1.83 / 2.36 / 0.65 | 8.56 / 9.07 / 6.57 (208 / 239 / 222) | 1.83 / 2.36 / 0.65 | 1.20 / 1.51 / 0.84 | 3.51 / 3.84 / 1.77 | 2.36 / 2.57 / 0.96 |
| 100-300 Hz | 2.83 / 2.85 / 0.87 | 3.26 / 3.52 / 0.93 (6.1 / 6.5 / 2.5) | 2.82 / 2.85 / 0.87 | 2.52 / 2.36 / 0.93 | 4.99 / 4.97 / 1.48 | 4.23 / 3.95 / 1.25 |
| 0.3-1 kHz | 5.44 / 5.65 / 2.29 | 5.05 / 5.35 / 2.42 | 5.45 / 5.66 / 2.29 | 5.91 / 6.02 / 2.19 | 5.66 / 5.96 / 2.93 | 6.07 / 6.24 / 2.73 |
| 1-3 kHz | 4.49 / 5.44 / 4.22 | 4.23 / 5.13 / 4.03 | 4.49 / 5.44 / 4.22 | 4.80 / 5.79 / 4.43 | 4.24 / 5.15 / 4.07 | 4.51 / 5.46 / 4.27 |
| 3-10 kHz | 5.98 / 5.73 / 3.99 | 5.93 / 5.68 / 3.93 | 5.98 / 5.73 / 3.99 | 6.03 / 5.78 / 4.05 | 5.93 / 5.68 / 3.93 | 5.98 / 5.73 / 3.99 |
| **19.5 Hz-10 kHz** | **9.84 / 10.39 / 6.34** | 12.75 / 13.48 / 9.03 | 9.85 / 10.39 / 6.34 | 10.11 / 10.53 / 6.51 | 11.07 / 11.56 / 6.78 | 10.79 / 11.12 / 6.64 |

Reading it: applying S twice is wrong in principle, and the table shows where it matters. In total
rms it costs only 1-3 % on this machine at rest, because 91-98 % of the variance sits where
|S| ~ 1. At 19.5-100 Hz it is off by -36 % to +29 % in the Telica loop and by -33 % to -46 % in the
generator (100-300 Hz: -17 % to +7 % and -11 % to -20 %).
Above 1 kHz source, error and "S twice" coincide. The generator's own loop (Garcia plant,
ruleOfThumb Cfb) shapes the same source into 11.1 / 11.6 / 6.8 nm: 7-12 % more than Telica overall,
and about twice the Telica error below 300 Hz. That is the S_sim vs S_telica difference, and the
reason the source, not the measured error, is what transfers between the two loops.

**MATLAB recipe** (`matlab/recipe_sensitivity.m`, run "recipe", 439 s; Control System and Signal
Processing only): `S = feedback(eye(3), G*K)`; the VAR noise model as `ss` (`var2ss`);
`covar(S*Hv, eye(3))` is the exact steady-state error covariance. It reproduces the measured Telica
standstill rms, **9.85 / 10.39 / 6.34 nm** against 9.85 / 10.40 / 6.34, independently of the Python
chain. The same call with the fit of the measured spectrum gives the S-twice result, and with
`G_sim, Cfb` it gives the generator's prediction (confirmed by the friction-off simulation to
0.3 %, G5 c). Data side: `pwelch` / `cpsd` with the G1 settings (Hann 2048, 50 %); simulation side:
`lsim` or the generator, then `pwelch`.

**Consequences for the training pipeline** (reported, not implemented):
1. **Level.** `data.py` adds white output noise at SNR 60 with sigma_n = 1.4e-4 m, i.e. **140 um
   rms**, open loop. The measured standstill error of the machine is **~10 nm**, 14 000x smaller; the
   in-loop noise part of y in the twin is 11 / 12 / 7 nm.
2. **Aliasing.** The twin's noise part of y (`S_sim v`) has **26-37 % of its variance above 2.5 kHz**
   and 34-41 % above 2 kHz. `load_traj` point-samples `y[::D]` (D = 4 or 5), justified in D-087 /
   D-099 by "y band-limited << 2 kHz". That holds for the noiseless y and fails for the noisy one: a
   third of the noise variance folds into [0, Nyquist]. Positions need an anti-alias filter before
   point sampling (telica-real's 5 kHz path already does this, TR-022). u is block-averaged and
   needs no change.
3. **Differencing in `compute_normalization`.** Harmless at this level: the noise velocity from first
   differences of the decimated y is 4.4e-5 / 1.2e-4 / 4.4e-5 m/s (logical X, Theta, Y, D = 4)
   against true velocity stds of 0.082 / 0.031 / 0.35 m/s (train median), about 1e-3 relative. The
   2.7x to 619x inflation documented in data.py belongs to the 140 um SNR table, not to encoder-level
   noise.
4. **float32.** `load_traj` casts y to `cfg.dtype_np`; in float32 the Y channel rounds by ~4 nm rms
   (G5 precision finding), the same effect as telica-real's float32 finding (TR-013). With nm noise
   the pipeline has to stay float64.
5. **Closed-loop bias** (Forssell and Ljung 1999; `LITERATURE.md`): in the twin the logged u contains
   `-Cfb S_sim v`, so the input is correlated with the output noise, and an output-error fit driven
   by the logged u is biased unless the noise model is right. The residual (training) form
   `u = u_data + K (y_data - y_model)` feeds `K v` back through y_data: the same correlation.
6. **Against the `CL_NOISE_CONSISTENT` shortcut** (`cl_train.py`: `y += v`, `u -= C_fb(v)`): the
   shortcut puts v unshaped in y (12.7 / 13.5 / 9.0 nm) where the loop puts `S v` (11.1 / 11.6 /
   6.8 nm); its force noise `K v` is 0.113 / 0.132 / 0.036 N against `K S v` 0.126 / 0.144 / 0.038 N;
   and it leaves the true position untouched, where the loop moves it by `T v` = **10.9 / 11.2 /
   7.7 nm** rms while sliding and by 38-1288 nm on held rails (G5 d). The missing `-T v` is as large
   as the noise itself.

**The open question of handoff section 8, in numbers.** Production level = the standstill source
(tanh reading), 12.8 / 13.5 / 9.0 nm over 19.5 Hz-10 kHz. Its white-sensor part is at most 4.8 /
4.8 / 4.5 nm (S-only bound). Motion upper bound: the non-repeatable content during motion is
200 / 211 / 87 nm (iter0) and 149 / 157 / 92 nm (iterETEL). Read as an encoder-node source it would be
0.86 / 0.61 / 0.22 um (tanh; diagonal de-shaping) or 16 / 13 / 5.4 um (sliding), times 1-2 for the
ILC factor. That is not an encoder: it is disturbance and position-dependent structure below 300 Hz,
so it was not generated as a second dataset (CN-011). The two interpretations of v bound the
transfer. All-sensor (what the twin injects) is one end. Partly `G d` (0.1-0.5 N below 300 Hz) is
the other: that part would enter the Garcia plant as a force, i.e. shaped by `S_sim G_sim` instead
of `S_sim`, and that is the correction to make if ASMPT can attribute the low-frequency part to
forces.
