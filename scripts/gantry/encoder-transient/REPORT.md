```
G1 Diagnosis: Y and dY error is 100 % model mismatch (the absorber on the payload: the planted-model map
  cuts it 29x / 33x, and it is flat in |Y|, 2.165e-05 to 2.178e-05 m over T1-T5, T3 at Y = 0 included);
  X, dX 97 % and Theta, dTheta ~100 % operating point (corr with |Y| +0.92 / +0.90 -> -0.96 / +0.99 with
  the per-window map; the Theta remainder is absorber coupling and goes only with both); window 0 %.
G2 Fix: PASS on attempt 1. Transient share 99.6 % -> -0.1 % (<= 30 %, ET-004); discrimination
  1.275x -> 176.6x RMS (>= 0.85 x 19.23x, the burn-in value re-measured on this dataset). Planted model
  at the exact-state floor (1.1625e-07 vs 1.1638e-07 m). Y-axis: fixed only by a map built from a model
  that contains the absorber; a baseline-source map leaves Y/dY unchanged (the section 8 split).
G3 Structure: PASS. Planted free run with the G2 seed 200x-10000x below the production seeds (H 400 and
  4000); untrained model: exact seed still 10x worse on Y (D-197 compensation reproduced). R1-LS: the
  same linear structure fits the truth 3x-88x better than G2, so G2's state floor is model mismatch,
  not structure; Adam never improves the near-exact planted init (best epoch 0 of 1000).
G4 Telica: PASS on attempt 3. FD reference bound 4.9e-05 m/s vs encoder dX error 7.2e-03 (resolvable,
  every channel). Map built from the model the Telica encoder feeds (recovered parameters + its tanh
  friction + Y schedule) at na 7: dX 7.2e-03 -> 1.3e-03, dY 2.8e-02 -> 4.4e-03 m/s, positions 20x-29x
  lower, every ypos (worst 0.29; held-out 0.31). Attempts 1 (schedule only) and 2 (na 29) FAIL on the
  Theta pair, whose model row is not identifiable. na 7 is the grid edge.
G5 Server: PASS, not submitted. Arms: sim burnin / g2static / g2refresh (model-Jacobian map rebuilt per
  epoch, reproduces G2 exactly without training), R2 on a checkpoint, Telica with the G4 encoder. One
  Adam step raised the window error 8x (sim) / 30x (Telica), grow -> 0.43 / 0.01: encoder linear map
  frozen in the G2 arms (ET-013).
Context: the handoff's 88 % / 1.25x / 3.40x belong to another dataset (ma_frac 0.10); re-measured here:
  99.6 % / 1.275x / 19.23x (ET-002).
Resources: peak tree RAM 1.25 GB, min available RAM 3.54 GB (alerts only), 0 watchdog kills,
  2 training updates.
```

# encoder-transient: report

Handoff `tasks/handoffs/2026-09-23-encoder-transient.md`. Decisions `DECISIONS.md` (ET-001 ...), runs
`RUNS.md` (hypothesis before launch). Every number below is from a run in `RUNS.md`; thresholds were
registered in `DECISIONS.md` before the number they judge existed.

**Context correction first (ET-002).** The handoff's reference numbers (88 % transient, 1.25x, 3.40x
with burn-in) were measured by `cl_burnin_sweep.py` on a different dataset (`mode='augmentation'`,
`ma_frac 0.10`, `zeta_a 0.05`, `nx_ann 2`) with a planted ANN regressed onto `msd-offset/plant.py`
(`ma_frac 0.10`). The handoff's own check "confirm the planted model matches the dataset's absorber"
therefore fails for `augmentation_ma50_b140-230_a6_z03`. This session built an EXACT planted model
(RK4 of the replay-gated 8-state EOM `truth.Plant8` at the dataset's absorber, the model's own `Ts`
and `up_sample`, no fitting) and re-measured today's numbers on this dataset (G1). All G2 thresholds
are relative to those re-measured values.

## G0 Infrastructure: PASS
Folder, vendored code (`vendor/VENDORED.md`, HEAD `1ec57c1`, working tree), watchdog
(`tools/watchdog.ps1`, launched by `tools/run.sh`). Vendored R0 on V1: Y `2.1772e-05 m` (D-197
`2.18e-05`, 0.13 %), dY `2.7857e-02 m/s` (`2.79e-02`, 0.16 %); threshold 1 % (ET-001). Two things
had to be pinned to reproduce D-197: the dataset (the working-tree entry file names the friction
folder) and the record lists (D-206 appended Telica-profile records that the frictionless folder lacks).
New fact from G0: the V1 Y error is zero-mean (bias `-2.3e-09 m` against RMS `2.18e-05 m`), so the
"absolute floor" is not an operating-point OFFSET.

## G1 Diagnosis in simulation: PASS
Runs R003 (state error, no rollout) and R004 (closed loop, no gradients). Thresholds ET-003.

**State error against the exact truth**, pooled over T1-T5 and V1-V4, `na = 29`, stride 5. `P0`
production map (baseline, `Y = 0`); `A` baseline map at the window's measured `Y`; `B` planted
8-state map at `Y = 0`; `AB` both. Shares are the pre-registered two-factor split of the removed MS.

| channel | P0 | A | B | AB | operating point | model mismatch | window | named cause |
|-|-|-|-|-|-|-|-|-|
| X [m] | 1.176e-06 | 2.951e-07 | 1.174e-06 | 3.383e-08 | 0.97 | 0.03 | 0.000 | operating point |
| Theta [rad] | 6.429e-06 | 1.830e-06 | 7.184e-06 | 1.933e-07 | 1.08 | -0.08 | 0.000 | operating point (+ model interaction) |
| Y [m] | 2.176e-05 | 2.171e-05 | 7.519e-07 | 6.368e-08 | 0.00 | 1.00 | 0.000 | model mismatch |
| dX [m/s] | 1.203e-03 | 2.992e-04 | 1.202e-03 | 4.241e-06 | 0.97 | 0.03 | 0.000 | operating point |
| dTheta [rad/s] | 6.072e-03 | 1.743e-03 | 6.594e-03 | 3.206e-05 | 1.05 | -0.05 | 0.000 | operating point (+ model interaction) |
| dY [m/s] | 2.784e-02 | 2.781e-02 | 8.384e-04 | 1.366e-04 | 0.00 | 1.00 | 0.000 | model mismatch |

- **Y and dY are model mismatch, entirely.** The absorber mass is half the payload (`ma_frac 0.50`), it
  sits on the Y axis, and the multisine drives it at its 150 Hz mode in every record; the baseline map
  has no such mode. That is why the error is identical on every record (same excitation) and flat in
  `|Y|` (T1-T5 span 2.165e-05 to 2.178e-05 m, T3 at `Y = 0` included, where the operating point is
  exact). The planted map alone cuts it 29x (Y) and 33x (dY); scheduling adds a further 12x / 6x only
  once the model is right.
- **X and Theta pairs are operating point** (MS1 / D-167 reproduced): `A` cuts them 3.5x to 4x. The
  Theta pair keeps a residual under `A` that is still V-shaped in `Y` (corr +0.988), the open point of
  D-167; the planted map removes it (`AB` 33x / 189x). Consistent with the EOM: the true Theta inertia
  carries `ma (Y + L0 + da)^2` and the coupling `-ma L0`, so half the payload sits at `Y + L0`, not at
  `Y`. This is a candidate for the "second Y-dependent term" D-167 could not name (G1 shows the planted
  map removes the residual; it does not isolate the term). `B` alone is WORSE on Theta
  (the right model at the wrong operating point), hence the negative model share: an interaction.
- **Correlation with `|Y|`**, `corr(record RMS, mean |Y|)` over T1-T5, before -> after `A`: X +0.919 ->
  -0.955, Theta +0.903 -> +0.988, dX +0.923 -> -0.906, dTheta +0.904 -> +0.888, Y -0.795 -> -0.918,
  dY -0.938 -> -0.958. On X/dX the sign flips and the level drops 4x (the dependence is removed, as in
  D-167). On Y/dY the correlation is of a quantity flat within 0.6 %, so it carries no information.
  After `AB` every residual is flat within about 15 % across T1-T5.
- **Window**: with the right map the window buys nothing and longer windows HURT (pooled `AB`, na 15 /
  29 / 59 / 119: Theta 1.73e-07 / 1.93e-07 / 3.71e-07 / 2.70e-06 rad, dY 1.43e-04 / 1.37e-04 /
  1.86e-04 / 2.37e-04 m/s). Untrained `P0` gains at most 1.12x from any window (X); R1's window gains
  appear only after training the encoder against the truth.
- Gates: planted(`ma -> 0`) linearisation equals the baseline map to 1.2e-07; float64 `P0` equals the
  pipeline's float32 encoder to <= 6.0e-04 of the error; grid interpolation <= 4.8e-03 of the error.

**Reconciling P1 ("the window is NOT the constraint") with R1 ("the window is the lever").** They
answer different questions and both hold. P1 fitted an unconstrained least-squares map to the TRUTH
and found the absorber state reconstructable at `L = 9` to `R2 = 0.9998`: the information is in a
short window. R1 started from the baseline map at `Y = 0`, which G1 shows is the wrong model (Y/dY)
at the wrong operating point (X/Theta). A longer window lets the fit average that mismatch down, so the
error falls with the window. Once the map is built from the right model at the right `Y`, the short
window is at the floor and longer windows only add linearisation error. The window was a lever against
model mismatch, not an observability limit.

**Today's transient on this dataset** (R004, V1-V4, 476 windows, `nf = 400`, closed loop, `xc = 0`):

| arm | window RMS [m] | RMS [100:400] | transient share | grow |
|-|-|-|-|-|
| untrained model, production encoder | 2.0548e-05 | 2.0766e-05 | -2.1 % | 1.63 |
| planted model, production encoder, W^a random (today's init) | 1.6118e-05 | 1.0799e-06 | 99.6 % | 0.01 |
| planted model, production W^b, W^a zero | 3.1219e-06 | 1.7400e-07 | 99.7 % | 0.01 |
| planted model, exact 8-state x0 (floor) | 1.1638e-07 | 1.1627e-07 | 0.2 % | (e0 ~ 0) |

Discrimination untrained / planted: 1.275x at `K = 0` and **19.23x with burn-in `K = 100`** (W^a
random); 6.58x / 119.3x with W^a zero. The floor is 0.72 % of the planted `K = 0` RMS (<= 10 %
required), so the planted test measures the encoder, not the discretisation. The untrained model's
error is model mismatch, not transient (share -2 %), which is why the loss cannot rank the two.

## G2 The fix in simulation: PASS on attempt 1
Run R006, thresholds ET-004. The variant `encoder/sched_encoder.py` + `encoder/core.py`:
`ScheduledReconEncoder` = production `linear_encoder_init_aug` as parent (so degree 0 IS production)
plus `sum_k C_k (T_k(Y/0.4) - T_k(0))` on the pure-scaled window, `Y` = measured stage Y at the
window's last sample, `C_k` least squares to exact frozen-Y maps with `W(0)` pinned (D-167 method,
Chebyshev basis). Source `planted`: W^b AND W^a from the planted model (the latent rows are the
absorber pair in its known realisation). Window `na = 29` (G1: no window gain, Theta degrades longer).

- **Collapse gate**: degree 0, baseline source, equals the pipeline encoder bitwise (`torch.equal`)
  on all 47 969 V1 windows, float32 and float64.
- **Degree rule** (T1-T5 only, within 10 % of the exact grid map on every channel): baseline source
  **2** (worst ratio 1.026, map residual 4.5e-04), planted source **6** (1.010, 2.6e-07). Degree 0 of
  the planted map is 345x off on dX: the planted map needs the scheduling far more than the baseline one.
- **Closed loop, planted model, `nf = 400`, `K = 0`**:

| arm | window RMS [m] | share | note |
|-|-|-|-|
| planted, G2 (planted source, deg 6) | **1.1625e-07** | **-0.1 %** | at the exact-state floor 1.1638e-07 |
| planted, exact grid-scheduled planted map | 1.1625e-07 | -0.1 % | the polynomial loses nothing |
| planted, planted map at `Y = 0` (no schedule) | 7.5954e-07 | 97.2 % | model alone is not enough |
| planted, baseline map scheduled, W^a zero | 3.0535e-06 | 99.7 % | operating point alone is not enough |
| planted, baseline map scheduled, W^a random | 1.6084e-05 | 99.6 % | = today |
| untrained, G2 (baseline source, deg 2) | 2.0535e-05 | -2.3 % | vs 2.0548e-05 with the production encoder |

  (a) transient share **-0.1 %** <= 30 %: PASS. (b) discrimination **176.6x** at `K = 0` (178.6x at
  `K = 100`) >= 16.35x: PASS (today 1.275x; burn-in 19.23x). (c) state error vs `P0`, pooled: X 35x,
  Theta 33x, Y 342x, dX 282x, dTheta 191x, dY 204x lower; worst single record and channel 0.255 (T3,
  Theta: at `Y = 0` only the model half of the fix acts): PASS.
- **Mechanism.** The transient is the mismatch between the encoder's `x0` and the state the MODEL
  needs. Both halves are required: the operating point (X/Theta) and a map from the model the encoder
  feeds (Y/dY and the latent pair). Each alone leaves 97 % to 99.7 % of the window as transient.
- **The Y-axis caveat, stated plainly (section 8).** On the planted model Y is fixed. In the
  production pipeline at initialisation the model is the baseline, and a baseline-source map CANNOT
  fix Y/dY (2.172e-05 m, 2.781e-02 m/s, unchanged); only a map rebuilt from a model that has learned
  the absorber can. So for training the fix splits: the scheduled map removes the X/Theta transient
  from the start, and the Y/latent part needs the map refreshed from the current augmented model
  (G5 arm `g2refresh`), or burn-in kept for Y until then.

## G3 Structure check: PASS
Runs R007, R016, R018 (R1), R015 (R2); thresholds ET-005, extra arm ET-007.

**R2, free run (the PASS criterion)**: open loop, V1-V4, 20 starts, pooled RMS [m] on the stage
outputs X1 / X2 / Y:

| model | x0 | H = 400 (0.1 s) | H = 4000 (1 s) |
|-|-|-|-|
| planted | exact 8-state | 9.5e-08 / 9.1e-08 / 1.7e-07 | 1.1e-07 / 1.1e-07 / 1.7e-07 |
| planted | **G2 (planted source)** | **4.4e-07 / 4.6e-07 / 5.1e-07** | **2.2e-06 / 2.2e-06 / 3.4e-06** |
| planted | production, W^a random | 1.0e-04 / 9.5e-05 / 4.6e-03 | 6.7e-04 / 7.1e-04 / 3.4e-02 |
| planted | production, W^a zero | 1.0e-04 / 9.4e-05 / 1.4e-04 | 6.0e-04 / 6.0e-04 / 9.9e-04 |
| untrained | exact physical | 1.2e-04 / 1.4e-04 / 1.5e-03 | 4.0e-04 / 4.8e-04 / 1.3e-02 |
| untrained | production | 1.3e-04 / 1.5e-04 / 1.5e-04 | 6.6e-04 / 7.0e-04 / 9.9e-04 |
| untrained | G2 (baseline source) | 1.3e-04 / 1.5e-04 / 1.5e-04 | 3.4e-04 / 4.1e-04 / 9.9e-04 |

On the planted model the G2 seed is 200x to 10 000x better than the production seeds on every
channel and horizon: PASS (ET-005). The compensation effect, reported honestly: on the UNTRAINED
model the exact seed is still 10x worse on Y than any encoder seed (D-197 R2 reproduced, 1.49e-03
vs 1.45e-04 m). The G2 baseline-source seed leaves Y exactly where the production encoder has it
(same compensation) and halves the X drift at 1 s (3.4e-04 vs 6.6e-04 m). A better physical-state
estimate helps only in a model that does not need the compensation. That is why G2 is judged on
the planted model, and why training needs the map rebuilt from the current model (G5).

**R1, structure headroom (reported, no threshold)**:
- *Adam from the G2 init* (the D-197 recipe, lr 1e-4, 1000 epochs, planted source, 8 channels): best
  epoch 0. The weighted validation MSE is 7.2e-13 at the init and 1e-10 to 5e-6 for the rest of the
  run: Adam's scale-free first steps destroy a near-exact linear init and never recover it. R1 with
  Adam cannot measure headroom from this start.
- *Closed-form best linear structure* (R1-LS, ET-007; the same Chebyshev-scheduled linear structure,
  ridge LS against the exact state, ridge chosen on held-out training records): headroom over the G2
  init on V1-V4 is X 7.9x, Theta 36x, Y 4.6x, dX 3.2x, dTheta 6.4x, dY 83x, absorber pair 82x / 88x.
  My pre-registered expectation (< 3x) is refuted. The G2 state floor is not the structure's
  limit. It is the gap between the linearised, ZOH, block-mean-input model the map is built from and
  the 20 kHz truth, and a truth-supervised map absorbs that gap. For the OBJECTIVE this headroom does
  not matter: the planted closed-loop window with the G2 encoder is already at the model's own floor
  (1.1625e-07 vs 1.1638e-07 m). What the transient needs is consistency with the model, and G2 has it.
  A truth-fitted map would be the D-197 R2 trap again whenever the model is not exact.
- *Adam from the G2 baseline-source init* (arm ii, 6 channels, 1000 epochs, best epoch 882): headroom
  X 4.9x, Theta 8.1x, Y 3.2x, dX 5.7x, dTheta 5.1x, dY 2.0x. Trained Y 6.7e-06 m and dY 1.4e-02 m/s
  land where D-197's R1 landed from the production init (7.3e-06, 1.2e-02): the absorber mismatch is
  only partly learnable by an encoder built on the baseline. Trained X 6.7e-08 m and Theta 2.5e-07 rad
  are 6x to 12x better than D-197's R1 (4.2e-07, 3.1e-06), the value of starting from the scheduled
  map. My expectation of no X/Theta gain is refuted. This start is far from exact, so Adam's first
  steps cost little here.

**What the three R1 numbers say together.** The encoder STRUCTURE is not the limit (R1-LS). The
initialisation is: from the right model's map (planted) there is nothing left for the objective, and
from the wrong model's map (baseline) supervision cuts X/Theta 5x to 8x but Y/dY only 2x to 3x.
Adam cannot be trusted to refine a near-exact linear map (R007 here, and R020/R025 in G5).

## G4 Telica: PASS on attempt 3 (attempts 1 and 2 FAIL, mechanisms below)
Runs R011, R012, R013; thresholds ET-006 (+ ET-009 pooled PSDs), attempts ET-010, ET-011. Records:
iter0 of all 15 operating points, 5 kHz (TR-022), cut [motion - 100 ms, end], 37 673 windows (na 29).

**The reference is resolvable, by 2 to 3 decades.** Positions `P^-T y` from the raw 20 kHz measured
positions (bound = standstill std: 6.9e-09 m, 2.1e-08 rad, 6.1e-09 m); velocities from a zero-phase
Butterworth (order 4, fc chosen by the pooled bound: 1600 Hz on all three) and a central difference
(bound 4.9e-05 m/s, 1.5e-04 rad/s, 5.7e-05 m/s, noise and filter distortion combined, ET-006/ET-009).
The production Telica encoder errs by 6.8e-06 m, 2.7e-05 rad, 2.6e-05 m, 7.2e-03 m/s, 2.9e-02 rad/s,
2.8e-02 m/s: 100x to 4000x the bound. The handoff's worry (FD reference too noisy) does not apply. Cross-check
with `scripts/gantry/closed-loop-noise/REPORT.md` (another session, finished before this one started):
standstill noise 9.84 / 10.42 / 6.34 nm on X1 / X2 / Y (time-domain rms, its G1 closure); the logical bounds here (X
6.9 nm, Y 6.1 nm) are consistent with it.

| attempt | encoder (vs P0-T = production, datasheet map at Y = 0, na 29) | X | Theta | Y | dX | dTheta | dY | worst ypos | verdict |
|-|-|-|-|-|-|-|-|-|-|
| 1 | datasheet map, Y-scheduled (deg 3) | 1.011 | 0.839 | 1.000 | 1.011 | 0.839 | 1.000 | 1.076 (X, +120) | FAIL |
| 2 | map from the model the encoder FEEDS (recovered params, its friction, Y schedule), na 29 | 0.747 | 0.984 | 0.661 | 0.743 | 0.975 | 0.654 | 1.208 (Theta, +40) | FAIL |
| 3 | attempt 2 at na = 7 (1.6 ms), window chosen on the train split | **0.039** | **0.051** | **0.034** | **0.181** | **0.239** | **0.160** | **0.293** (dTheta, +40) | **PASS** |

(Pooled RMS ratio to P0-T over all 15 records; PASS needs every resolvable channel lower and no ypos
level worse than 1.01. Attempt 3 on the 4 held-out val/test records only: worst 0.306, PASS.)

- **Attempt 1, operating point only**: helps the Theta pair (37 % at ypos -200), nothing else.
- **Attempt 2, the model**: the Telica pipeline feeds the encoder into a block with the RECOVERED
  parameters (viscous damping 2.3x to 3.1x the datasheet, mh +10 %) and tanh Coulomb friction of
  80 to 100 N (telica-real TR-015), but builds the encoder from the datasheet constants without
  friction. Building the map from the block's own Jacobians plus its own friction as an input
  correction (sign from the measured window) removes 25 % to 35 % of the X/Y error. Attribution:
  recovered parameters X 0.83 / Y 0.69, plus friction X 0.73 / Y 0.65. The Theta pair stays at
  ~2.7e-05 rad, 100x the measured Theta std (2.9e-07 rad), under EVERY map. The Theta row
  (J_eff, kb, cb) is not identifiable from these one-way moves (telica-real G4), so no available
  model has a Theta row that matches the machine.
- **Attempt 3, the window**: a model error that acts like a force error `dF` enters the reconstruction
  through the window's input path and grows like `dF T^2 / m` on positions and `dF T / m` on
  velocities. Shortening the window from 6.0 ms to 1.6 ms (ratio 0.27, squared 0.071) gives position
  ratios 0.03 to 0.05 and velocity ratios 0.16 to 0.24, the predicted pattern. The encoder's own
  noise barely rises (standstill floor 2.6e-05 m/s at na 29), because Telica is mismatch-limited, not
  noise-limited. The training-split objective falls monotonically to the SMALLEST window tested
  (0.119 at na 7 vs 0.804 at na 29), so na = 7 is the grid edge: the optimum may be shorter still.
- **How far from the measurement**: a kinematic estimator from the same window (positions read
  directly, velocities from a quadratic fit of the last 11 samples) reaches 5.2e-09 m, 1.5e-08 rad,
  5.1e-09 m, 3.5e-04 m/s, 3.5e-04 rad/s, 4.1e-04 m/s. The best model-based map is still 50x to 175x
  above it on positions and 4x to 20x on velocities. On Telica the encoder error is model mismatch,
  not noise. The kinematic arm is a reference, not a candidate: for a mismatched model a true-state
  seed can be the worse seed (D-197 R2, reproduced in G3 R2 below).
- **Contrast with simulation**: in simulation, with the right model, the window bought nothing
  (G1). On Telica, with an imperfect model, a short window is the dominant lever. Both follow from
  the same fact: the window length sets how far the map propagates the model's error.

## G5 Server preparation: PASS (nothing submitted)
Thresholds ET-008 (+ ET-012 Telica encoder, ET-013 freeze). Predictions: `runners/PREDICTIONS.md`.

**The training-time fix exists and is checked without training.** In training the model is baseline +
ANN and changes every update, so the map must come from the model itself: `encoder/model_map.py`
linearises the model's OWN step (autograd Jacobians at rest at each `Y`), builds the same
reconstructability map for ALL rows (W^b and W^a) and fits the Y schedule. Pre-check R017: on the
planted model it reproduces G2 exactly (window RMS 1.1625e-07 m = floor, share -0.0 %, 176.6x). On the
untrained model its latent rows are exactly zero (unobservable latents: the minimum-norm,
model-consistent W^a), and the affine offset at rest is 0.

| arm | script | what | smoke | updates |
|-|-|-|-|-|
| (a) `burnin` | `runners/sim_arm.sh` | production encoder, `burn_in = 100` (the current burn-in arm) | R024 dry, V1 RMS 2.0512e-05 | 0 |
| (a) `g2static` | same | G2 baseline-source map (deg 2), W^a = 0, `burn_in = 0` | R022 dry, 2.0510e-05 | 0 |
| (a) `g2refresh` | same | model-Jacobian map, all 14 rows, rebuilt every epoch, `burn_in = 0` | R020: refresh, 1 update, refresh; R027 dry (freeze) | **1** |
| (b) R2 on a checkpoint | `runners/r2_ckpt.sh` | exact / checkpoint encoder / model-Jacobian map of the trained model; open and closed loop | R023 (untrained model, no checkpoint on this PC) | 0 |
| (c) Telica | `runners/telica_arm.sh` | telica-real production arm with the G4 encoder (RFS, na 7) | R025: 1 update; R028 zero-update hook check (freeze) | **1** |

**Finding from the two allowed updates (ET-013).** After ONE Adam update the closed-loop window
error rose 8x in simulation (`[nf val]` 1.91e-05 -> 1.56e-04 m, `grow` 1.83 -> 0.43) and 30x on
Telica (5.08e-07 -> 1.47e-05 m, `grow` 0.41 -> 0.01). A `grow` below 1 puts the error at the window
start, i.e. in the initial state; a damaged ANN would make the error grow through the window. So the
evidence points at the encoder: Adam's scale-free first step wrecks a near-exact linear map (R007
showed the same in encoder-only training). The two updates cannot separate encoder from ANN further.
Consequence, applied to the prepared arms: in the G2 arms the encoder's linear map is FROZEN
(`requires_grad = False`, D-175) and, in `g2refresh`, maintained by the per-epoch rebuild from the
model instead of by gradient; the correction net stays trainable. `ENC_TRAIN_LINEAR=1` gives the
paired unfrozen arm. Checked without updates (R027: 17 640 frozen / 5 894 trainable; R028: 1 536 /
2 726).

**Defects found and fixed during G5, all before or without spending an update**: R019, the refresh
wrapper was a closure that deepSI's checkpoint pickling rejected. Its partial 855 B checkpoint in
`%LOCALAPPDATA%/deepSI/checkpoints` was deleted, and deepSI's work dir is now redirected into
`outputs/g5/<run>/` (as telica-real does). R021: the smoke nf (20) was below the burn-in (100).
R025: my marker comment on the vendored smoke line commented out the smoke `nf_seconds`, so that
smoke ran at nf 200 instead of 25 (corrected, R026, R028). Also, `vendor/transient/r2_freerun.py
--ckpt` unpacks 4 values from `resolve_checkpoint`, which now returns 5; `runners/r2_ckpt.py` does
not reuse that path.


## Resources and rule compliance
- **Runs**: 30 watchdog-supervised launches in 29 output folders (`RUNS.md`, each with its hypothesis before launch), run
  one at a time, GPU jobs never concurrent. 0 watchdog kills; 5 exited 1 (R002, R005, R009, R019,
  R021, each a documented code or config defect, fixed and re-run; R004's first attempt crashed too,
  its folder was reused). Peak process-tree working set 1.25 GB (R018), minimum available RAM
  3.54 GB (R018, 47 alerts below the 4 GB line, kill line 2.5 GB never reached), minimum C: free
  7.13 GB.
- **Training updates: 2 in total**, one in R020 (simulation `g2refresh` smoke) and one in R025
  (Telica smoke, rolled back by deepSI's best-checkpoint restore). Everything else was encoder-only
  regression (R007, R018 on the GPU; R016 closed form), forward rollouts without gradients, or
  zero-update dry runs.
- **Files**: everything created or edited is inside `scripts/gantry/encoder-transient/`, with one
  exception that was undone: R019's partial deepSI checkpoint (855 B) in
  `%LOCALAPPDATA%/deepSI/checkpoints`, deleted the same minute. Nothing committed.
- **Outside the watchdog**: a handful of interpreter calls that ran no project code and read no
  dataset (listed at the top of `RUNS.md`).
- **Data policy**: Telica logs only through the vendored loader, summaries only; `Telica.mat` never
  touched (the loader reads `Telica 1.mat` for Kt, as telica-real does).

## What is left for the user
1. Submit the prepared arms from one unedited checkout (`runners/*.sh`, commands in each header):
   `ARM=burnin` / `ARM=g2static` / `ARM=g2refresh` for (a), then `r2_ckpt.sh` on the best checkpoint
   of the `burnin` arm for (b), and `telica_arm.sh` next to telica-real's own `server_run.sh` for (c).
2. Decide the Telica window: na = 7 was the SMALLEST window tested and the training objective was
   still falling, so a shorter one may be better. It also changes the SUBNET window of the whole
   Telica pipeline (the encoder window is the model's history), which is a design choice beyond the
   encoder.
3. The Theta row on Telica is not identifiable from one-way moves (telica-real G4). Every encoder's
   Theta estimate is dominated by it until an experiment excites Theta or the ANN learns it.
