# Data design: training, validation and test (interpolation and extrapolation)

Design on paper only; generation starts after the user approves it (?). Numbers marked "hand estimate" are paper arithmetic from the generator's formulas, not simulations. Source precedence: user decisions, then session 1's `scripts/gantry/excitation-closed-loop/EXCITATION-VALIDATION.md`, then `RESULTS-DESIGN.md`, then the slides; result numbers follow the slides (R7 = controller transfer). History: first draft, examiner critique `CRITIC.md` (section 11), independent review `REVIEW.md` (section 12), user decisions and new inputs of 2026-09-26 (section 13).

## 0. Notation
- T_AF: the benchmark truth, physics baseline + Coulomb rail friction + payload absorber (ma 0.50 mh, damping 0.03 through `ZETA_A_OVERRIDE` as in the D-188 datasets, free-free pole 212.13 Hz, anti-resonance 150 Hz; in closed loop the peak sits at 266 Hz, session 1)
- K1: the one controller of every training, validation and test record, designed once at Y = 0 (not per record, as the current generator does); K2-*: changed controllers, R7 test records only (5.11)
- max: the datasheet maximum acceleration, X 30 and Y 50 m/s² (user decision 2026-09-26); level L: a = L x max and v = L x 2 m/s unless stated
- A_prod: the production multisine rms, 240 N symmetric, 87 N m anti, 180 N Y (6x the `gtd_config.m` values, the "a6" datasets; AUDIT M1, session 1)
- B_tr: the training multisine band, 140 to 390 Hz at A_prod, session 1's proposal (?); no separate parameter band (Q2)
- N_T: a force disturbance d on the motor force inside the loop (plant force = u_ff + u_fb + d), white per 20 kHz sample, stage-axis covariance calibrated so the frictionless linear loop at Y_op = 0 reproduces the measured Telica standstill error, 9.8 / 10.4 / 6.3 nm rms (`Thesis-writeup/Documentation/NOISE-INJECTION.md`); y is the true position, u_total excludes d; every record its own realisation
- "own": the record has its own multisine phase realisation; "none": no multisine
- j: peak reference jerk; d: move distance
- Labels: a range label (I or E per axis) and a density label (dense or sparse), both from section 6; tables give the intended range label
- Priority: core (the basics, Quinten) or extra (if time allows); every record is generated in one campaign, priority only orders the evaluation

## 1. Machine values and their sources
| Quantity | X | Y | Source |
|-|-|-|-|
| Total stroke | 750 mm (±0.375 m) | 800 mm (±0.400 m) | datasheet p. 2, dimensional data; `gtd_config.m` lim.pos |
| Operating range | targets ±0.35 m | targets ±0.325 m, pick station at -0.40 m | datasheet p. 4, typical cycle: the full Y stroke is operational |
| Max speed | 2 m/s | 2 m/s | p. 2; p. 4 cycle |
| Max acceleration (max) | 30 m/s² | 50 m/s² | p. 2; the p. 4 cycle runs at it |
| Measured ASMPT (ILC) move | 40 mm, 1.0 m/s, 39.3 m/s² (1.31 x max) | 80 mm, 1.5 m/s, 50.7 m/s² | D-210; a stress record beyond max (5.4) |
| Jerk time | 25 ms (p. 4 cycle); ILC equivalent 13.5 ms | 25 ms; 15.7 ms | p. 4; D-210 |
| Typical moves | 0.1, 50, 200 mm | same | p. 3 |
| Peak / continuous force | 2000 / 916 N per rail | 1420 / 656 N | p. 2; lim.force |
| Yaw limit | 6 mm (X1 - X2) | | lim.diff (no datasheet source, D-206) |
| Rated payload | | 2 kg, listed under Z; maximum payload "application dependent" | p. 2 |
| ILC campaign coverage | -0.21 to -0.02 m | -0.20 to +0.20 m | schema; campaign coverage, not a machine limit |

- Y origin: the p. 4 target grid is centred on the figure's origin, so the stroke is read as symmetric about it; that this origin is the beam centre (the model's Y = 0, the yaw axis) is assumed, a question for Quinten or Jasper (?)
- Modelling assumption, stated once in the setup: the simulated plant is Garcia's parameter set (cross-arm 0.725 m, X moving mass 53.8 kg, Y 10.1 kg) run over the Telica stroke; M(Y) treats the head as a point mass on a lever arm, valid at any Y, so beyond Garcia's own travel it is a synthetic benchmark (section 13 gives the reason and the line it replaces)

## 2. Design principles
- Scoring mode: every prediction, in training and in every test, is a closed-loop simulation of the model inside the known controller, driven by the reference r and the injected force f (multisine or feedforward), never by the logged u (RESULTS-DESIGN Q1); X and Y are free integrators, so open-loop runs from u drift; the noise enters the truth's loop only, so each controller has its own floor
- Training and validation are identification experiments plus the ILC-shape profiles without multisine, as in the current dataset (TP and VP records, D-206: records without multisine keep the multisine-free test records on the training distribution)
- Validation lies inside the training range but is not a copy of training, as in the current dataset (V1 to V4, VP1, VP2; user decision 2026-09-26): every training record type has validation records at Y values, centres and setpoints training does not use, with the same levels and jerk times and independent phase, noise and setpoint realisations; validation selects checkpoints, the model order and the black-box size
- Because validation also interpolates in Y, "equal training and validation error" mixes data sufficiency with interpolation; the data-sufficiency check (Quinten) therefore also runs on I1, a same-setting mirror in the test set that selects nothing
- One training fraction for every continuous axis: training covers up to 0.75 of the machine value; test extrapolation takes 0.875 and 1.0 (1.17x and 1.33x the training maximum); HEURISTIC (?)
  - anchored to the datasheet and to the realised maximum over all training records, so no test value lies below a trained one
  - precedents for the distance: Bolderman 2024 Fig. 13 (sweeps of maximum position, velocity, acceleration); Kessels R3 (peak acceleration +85 %, thesis p. 177); 1.33x is the milder end, fixed by the user decision that tests stop at the machine maximum
  - the one exception is the measured ILC move (1.31 x max on X), a named stress record, because ASMPT runs it
- One factor at a time from a matched interpolation anchor (Bolderman 2024 Fig. 13; Kessels Tables 5.6 and 5.7); records that change several axes are declared; moving extrapolation records are also scored on their out-of-range segments alone
- The unseen-profile claim (Aspect 4 ii) rests on new kinematics, distances and routes (X only, Y only, 0.1 and 5 mm moves, long strokes, the datasheet cycle); both profile shapes (S-curve, sine) are trained
- One controller K1 for every record except the controller tests (user decision), designed once, not per record (5.11)
- One truth (T_AF) for every set; control truths only where a result needs them (7.6)
- Realisations: training and validation in 3 data realisations (new phases and noise; new setpoint draws for moves), used as replicate runs, not pooled; every test record has a noise-only twin (same r and f, new noise) for its floor; every multisine test record a second phase realisation
- Noise N_T in every record; R5 also uses noise-free twins (d = 0) of its training records, since the recovery condition is derived noise-free (RESULTS-DESIGN S4)
- The noise matches the measured level, not its spectrum: the Telica standstill error is mostly sensor noise, over 90 % above 200 Hz, while a force disturbance moves the mass mostly below about 200 Hz; the thesis calls it "a force disturbance calibrated to the measured error level", not "the Telica noise"
- Black box: same data, network size and hyperparameters selected on validation (Quinten: no size limit); training effort (time to converge, cost of the search) reported; limitation stated: B_tr targets what the baseline misses, which favours the grey box (TE-B1 partly counters it)

## 3. The operating-point kind
Every condition the closed-loop linearised response of this truth depends on.

| Member | Enters the truth through | Use |
|-|-|-|
| Payload position Y | M(Y), quadratic in Y; absorber mass at Y + L0 + delta_a | used, main scheduling axis (5.1) |
| X position | nothing (truth is X-invariant), but X is a network input (z = [x̂; u]) | used, invariance test (5.2) |
| Velocity per axis, incl. sign and zero | Coulomb sign and stick band (D-209); viscous damping | used: stick (standstill), slip (moves, reversal share 0.1 to 1.4 %, session 1), reversals (5.3) |
| Stick/slip combination across axes | friction per rail and on Y | covered, not an axis: move axes finish at different times, Lissajous; the density rule checks it |
| Excitation amplitude | friction response depends on force level (Type I: at A_prod/6 the rails stick and the closed-loop peak drops to 227 Hz, session 1) | used (5.9) |
| Controller operating point | K1 design point; realised crossover shifts with Y on the rails and with the absorber on Y | one fixed design (5.11) |
| Yaw angle and rate | linear yaw spring | not an axis; in the density features, since the network reads them |
| Absorber deflection | linear spring | omitted: linear |
| Scheduling rate dY/dt | no dM/dt terms in truth or baseline (RESULTS-DESIGN 4c) | merged into velocity |
| Payload mass | mh in every M(Y) entry; also moves the absorber pole | plant member (5.12) |
| Temperature, wear, second beam head, Z axis | not in the simulator | omitted, not modelled |

## 4. The extrapolation kind
Anything the test data contain beyond what training covered, for this machine and this model.
- Y position beyond the training range, inside the stroke: used (5.1); the stroke ends are operational (pick station), so "beyond training" is also "inside the operating range"
- Y beyond the stroke: omitted, the end stop
- X position beyond the training range: used (5.2)
- Velocity: used (5.3)
- Acceleration up to the machine maximum: used (5.4, user decision); beyond it only the measured ILC move, as a named stress record
- Jerk above the trained maximum: used (5.5); T3 is only the generator's knob for it
- Move distance below the trained minimum: used (5.6); above the trained maximum it needs positions outside the trained range, so it is not a separate axis
- Motion type, axis combination (X only, Y only, XY): used as the cross-coupling group and as new routes (5.7)
- Motion type, commanded yaw: omitted from test; ASMPT commands X1 = X2 (D-206); yaw stays in training for identifiability
- Motion direction and sequence (reversals, dwell): omitted as an axis; random setpoints cover both directions at every Y; dwell only changes the standstill share
- Profile shape: both shapes trained; the shape contrast at equal kinematics stays as a check (5.7)
- Excitation class (multisine, none, swept sine): none is trained through the ILC-shape records; swept sine is E (5.8)
- Excitation amplitude, above and below the trained levels: used (5.9)
- Excitation band (frequencies outside B_tr): used, extra (5.10)
- Controller: gain ±20 % (core), bandwidth 60 and 150 Hz (after the stability gate and Quinten's answer), feedforward (?), notch only if named (5.11)
- Plant, payload mass: used, extra, pending ASMPT confirmation (5.12)
- Plant, changed absorber or friction law: omitted; a missing dynamic that changes independently after training is unknowable to every model
- Parameter detuning: a model initialisation, not a data property (Q2)
- Noise level: omitted; not a model input under closed-loop scoring from r and f, it moves the floor only
- Horizon longer than the training window: not varied; every test is scored over the whole record
- Initial state: omitted; every record starts at rest after the 0.5 s hold that holds the encoder window
- Sample rate: omitted, fixed
- Several axes at once: used, one stress record (5.13)

## 5. Axis rows

### 5.1 Payload position Y (core)
- Training -0.30 to +0.30 m for every record (75 % of the ±0.40 stroke); standstill points -0.30, -0.15, 0, 0.15, 0.30; validation mirrors them
- Validation: standstill at ±0.075 and ±0.225 m, between the training points; sweeps, Lissajous and ILC-shape records on shifted centres and lattices inside ±0.30
- Test I: standstill every 0.025 m between the training points, leaving out the validation Y (16 points, group I2); ASMPT routes inside ±0.20
- Test E: standstill at ±0.325, ±0.35, ±0.375, ±0.39 m; one move record and one ILC-shape record reaching ±0.39 m through explicit setpoints
- Changed 2026-09-26: the line "fallback if the geometry rules out |Y| > 0.3625 m: E points ±0.325, ±0.35, ±0.36 only" is removed; why: the datasheet cycle runs to -0.40 m, and Garcia's own table would limit the head to ±0.24 m (section 13), so the ±0.36 cut had no basis
- Sources: stroke and operating range (p. 2, p. 4); 0.39 keeps the limiter margin; 0.025 m grid from RESULTS-DESIGN R1
- Why: Aspect 4 (i), held-out operating points; R1 (LPV vs LTI) and R6
- Prediction (hand estimates: rigid baseline from M_op of `gtd_build_plant.m`; true system from the same matrix with ma at Y + L0, absorber locked, i.e. below the 150 Hz anti-resonance)
  - from the training edge to ±0.39 m, the baseline's X-yaw coupling grows 28 % (+Y) and 32 % (-Y), its yaw inertia 13 %
  - the true system's X-yaw coupling (-0.686 - 10.1 Y) grows 25 % (+Y) and 39 % (-Y), its yaw inertia 14 % (+Y) and 12 % (-Y); this is what the models must extrapolate; the LPV baseline carries its own share, the LTI baseline none
  - the LPV baseline is not exact on T_AF: it puts all of mh at Y, the truth puts ma at Y + L0, leaving a constant X-yaw offset of 0.505 kg m and a yaw-inertia residual 1.01 Y + 0.05 kg m² (0.44 at 0.39, 8 %), which the augmentation must also extrapolate
  - the LTI residual at 0.39 m is 4.5 to 9x that; inside, all arms are close, because training moves cross every interior Y and move holds stop at random interior Y
- Falsifier: the end rise of the LTI arm not larger than that of the LPV arm, beyond the start spread
- Criterion: end rise per arm = error at each end point minus the mean interior error; compared with the S6 rule of RESULTS-DESIGN; floors from the noise twins
- Label: E for |Y| > 0.30 by range; interior dense or sparse by the density rule

### 5.2 X position, an invariance the model is not told (extra)
- Training X_sym ±0.28 m (move setpoints, Lissajous amplitude; ILC-shape shuttles inside ±0.10); validation mirrors it
- Test I: TI-A2 (X only at Y -0.20, X inside ±0.10), TI-A9 (the same move at X 0.21 to 0.25 m); with TE-X2 they form one ladder: centre, edge, beyond
- Test E: TE-X1, moves at L 0.5, T3 30 ms with an explicit one-sided setpoint list, 0.28 < X ≤ 0.36 m; TE-X2, X-only sine moves between 0.30 and 0.36 m at Y -0.20
- Sources: stroke 0.375 m (p. 2)
- Why: the truth does not depend on X, but X enters the network of the augmented model and of the black box; only a held-out X range reveals a spurious X dependence
- Prediction: B-LPV exactly flat in X; the learned arms flat if they learned the invariance
- Falsifier: an arm's error at TE-X not equivalent to its anchor (it depends on X spuriously); reported per arm, without requiring any arm to fail
- Criterion: S6 equivalence between anchor and E record per arm, out-of-range segments scored alone as well
- Label: E by range

### 5.3 Velocity (core)
- Training up to 1.5 m/s per axis (move level 0.75; sweep 1.41 m/s; Lissajous 1.50 m/s on X); validation mirrors it
- Test I: ASMPT routes up to 1.5 m/s; anchor for E: TI-A6 (200 mm moves, velocity capped at 1.5)
- Test E: long strokes (X 0.50 m, Y 0.56 m, inside the trained positions) at 1.75 and 1.95 m/s, acceleration 0.6 max, jerk about 330 / 920 m/s³ (inside)
- Sources: 2 m/s (p. 2, p. 4); 1.95 keeps a margin for the velocity check on the simulated response
- Why: throughput moves run at the speed limit; velocities are network inputs
- Prediction: viscous damping is in the physics, and Coulomb friction is a sign that a saturating tanh continues correctly, so the augmented model holds; the black box must produce damping linear in velocity beyond its data and saturates
- Falsifier: the augmented error rise from anchor to E at least as large as the black box's
- Criterion: S6 rule on the rises, whole record and above-1.5 m/s segments
- Label: E by range on realised peak reference velocity

### 5.4 Acceleration (core, user decision: datasheet maximum)
- Training reference acceleration up to 0.75 max, X 22.5 and Y 37.5 m/s² (move levels 0.25, 0.5, 0.75; ILC-shape records at 0.45 and 0.75); sweeps and Lissajous stay at or below 8 m/s²; validation mirrors it
- Test I: ASMPT routes at 0.3 max (TI-A7, 9 / 15) and 0.6 max (TI-A1, 18 / 30); the I1 twin of TR-P3
- Test E, acceleration only (jerk inside the trained range): TE-A1, sine at 0.875 max (26.25 / 43.75, j 1690 / 2560); TE-A3, S-curve at 1.0 max, T3 25 ms (j 1200 / 2000); TE-A6, moves at L 1.0 with multisine; TE-A7, the S-curve move at 1.0 near Y 0.30, where the loaded rail is heaviest
- Test E, acceleration and jerk together (declared): TE-A2, sine at 1.0 max (30 / 50, Y velocity-capped at 1.5, j 2060 / 3340, 1.10x / 1.07x the trained jerk); TE-A4, A5 (X only, Y only at 1.0)
- Stress record beyond the maximum: TE-A8, the measured ILC move exactly (D-210 kinematics; X 1.31 x max, j 3090 / 3430); it needs a named exception to the acceleration assert (section 10)
- Why: Quinten counts "a higher maximum value" as extrapolation; the datasheet cycle runs at the maximum
- Prediction
  - the truth is linear apart from friction, so the physics part scales exactly with acceleration
  - force in the network's input space (mixed hand estimate, not a truth force: loaded-rail coefficient from the truth coupling, up to 32.0 kg at Y +0.30, times the 1.37 transient ratio of D-206's pre-check, which ran on the rigid baseline; the post-simulation check of section 10 gives the real values): training moves at L 0.75 reach about 990 N per rail plus the multisine peak, about 500 to 750 N at A_prod, so about 1500 to 1750 N; the E records without multisine reach about 1260 to 1315 N, the measured move about 1725 N; on Y, 180 N rms of multisine already exceeds the motion force at max (505 N)
  - so acceleration E is range-E but expected density-dense: what is new is a sustained low-frequency force over a 90 to 180 ms acceleration phase, not its amplitude; a black box need not fail here, and an equal outcome is reported as "acceleration to the rated maximum is not an extrapolation of the network's inputs at production excitation"
- Falsifier: the augmented error rise from its anchor larger than B-LPV's (the correction degrades under larger motion)
- Criterion: error rise from the anchor per arm, S6 rule; inside training, TI-A7 and TI-A1 equivalent (S6), else the rise starts before the training edge
- Label: E by range on realised reference acceleration; density label beside it

### 5.5 Jerk (extra)
- The axis is the peak reference jerk j, since absorber-band content of an S-curve scales with it (above 1/T3 its acceleration spectrum falls as j / (π² f²)); T3 = a / j is the generator's knob
- Training T3 12, 25, 30, 36 ms; designed peak jerk up to TR-P4: X 1875, Y 3125 m/s³; the generator allows up to 2 jmax on triangular-acceleration moves, so the trained maximum is read from the generated references (section 6)
- Test I: TI-J1 (L 0.5, T3 18 ms, j 830 / 1390); ASMPT routes at 0.3 and 0.6 max
- Test E: TE-J1, moves at L 0.75 with j = 1.5x the realised training maximum (T3 about 8 ms if that maximum is the designed one: 2810 / 4690); TE-J2, S-curve 40 / 80 mm at 0.6 max and T3 8 ms (2250 / 3750); TE-A2 and TE-A8 (5.4)
- Sources: 25 ms (p. 4); ILC equivalents 13.5 / 15.7 ms (D-210) inside the trained T3 range; the shortest T3 the machine runs is pending from Jasper and Dragan, so 8 ms is a synthetic stress value (?)
- D-211 check at 212.13 Hz, f x T3 off integers: 12 ms 2.55, 18 ms 3.82, 25 ms 5.30, 30 ms 6.36, 36 ms 7.64, 8 ms 1.70
- Longer T3 than trained: omitted; it only removes content above 1/T3, and T3 above vmax/amax = 2/max (67 ms X, 40 ms Y) stops a move reaching amax
- Why: jerk decides how much motion energy reaches the loop bandwidth and the absorber
- Prediction: during moves the baseline error in B_tr grows with j; the augmented model follows it only if its absorber correction is a mode driven by the input, not a fit to multisine statistics
- Falsifier: augmented error in B_tr during the E-jerk moves above the baseline's
- Criterion: error PSD in B_tr against the floor PSD from the noise twins, re-read around the 266 Hz closed-loop peak
- Label: E by range on realised peak jerk, X and Y separately

### 5.6 Move distance (core for the ASMPT distances, extra for 0.1 mm)
- Training move distances from fixed logarithmic strata, 1 to 3, 3 to 10, 10 to 30, 30 to 100, 100 to 300 mm and 300 mm to the range, with guaranteed counts per record, random sign and order, reflected at the range bounds (?)
- Test I: TI-A8 (5 mm, at 0.3 max, j 960 / 2060, inside), 40 / 80 mm (ILC), 50 and 200 mm (p. 3)
- Test E, distance out of range with other kinematics inside: TE-D3, 0.1 mm sine moves at 3.8 m/s² (j about 1860, T 12.9 ms, fundamental 78 Hz); acceleration and jerk cannot both be held at the anchor's values, since j = sqrt(2π) a^1.5 / sqrt(d) for a sine move
- Test E, declared multi-axis: TE-D1, 0.1 mm sine moves at 0.6 max: T = 5.9 ms (X) and 4.6 ms (Y), fundamentals 169 and 218 Hz inside B_tr, j about 1.9e4 / 4.1e4 m/s³ (10x / 13x trained); the reference itself drives the absorber, which no trained motion does
- TE-D2, 0.1 mm S-curve moves at T3 25 ms: the generator is the FIR chain of Biagiotti and Melchiorri (`thirdOrderSetpointETEL`, T3 = amax / jmax kept for every distance), so the move lasts at least T3 and its content lies mostly below 1/T3 = 40 Hz
- Sources: p. 3; D-210
- Prediction: TE-D3 close to its anchor for every physics-based arm; on TE-D1 the baseline error is dominated by absorber ringing and the augmented model tracks it; on TE-D2 all models are close
- Falsifier: the augmented model no better than the baseline on TE-D1, or TE-D3 failing (then short distance alone matters)
- Criterion: error PSD in B_tr against the floor; S6 rule between arms
- Label: E by the per-move distance range; pointwise near the origin, so the density label may say dense (expected disagreement, section 6)

### 5.7 Profile shape, routes and cross-coupling (core)
- Training: S-curve moves (Jasper's generator, TR-P), sine-shaped ILC moves without multisine (TR-T), sinusoidal paths (sweeps, Lissajous), standstill; one yaw record
- The ILC-shape records keep the measured strokes (40 / 80 mm) and the sine shape but run at 0.45 and 0.75 max, so their durations are longer than the measured move; this supersedes, for training records only, the D-206 amendment "stroke and move duration are not adjustable" (user decision 2026-09-26); the measured move stays exact in test (TE-A8)
- Test I, new routes at trained kinematics: X only, Y only, 5, 50 and 200 mm, 0.3 max; TI-A4, the S-curve single move at the same kinematics as TI-A1 (shape contrast)
- Cross-coupling group: X-only and Y-only moves (TI-A2, A3, TE-A4, A5), scored on the differential rail error X1 - X2, the part neither move commands
  - hand estimates on T_AF: X only at Y -0.20 and 0.6 max gives a yaw disturbance torque of about 24 N m (M12 = 1.33 kg m with the absorber offset); Y only gives about 30 N m (M23 = -mh d); both of order um in differential error under a 100 Hz loop, an order estimate only; the Y error of an X-only move is a few nm, near the floor, so it is not the metric
- Why: Aspect 4 (ii), unseen routes and kinematics; the S-curve twin at equal kinematics separates shape from route
- Prediction: sine and S-curve at equal kinematics give equivalent augmented error (same continuity class, D-210); the differential error carries the absorber, which acts along Y and couples into yaw, where the baseline misses it
- Falsifier: a sine vs S-curve difference for the augmented model beyond the start spread; no cross-coupling gain over B-LPV above the floor
- Criterion: S6 equivalence between shapes per arm; cross-coupling against the floor and start spread
- Label: I (range and class)

### 5.8 Excitation class (core)
- Training: multisine on the identification records, none on the ILC-shape records (TR-T)
- Test: none on every ASMPT route (I by class); TE-B2, swept sine over B_tr at standstill (extra, sweep law, amplitude and duration to fix) (?)
- Prediction: the ranking of arms by error holds on multisine-off records
- Falsifier: the ranking reverses
- Criterion: S6 rule on pairwise differences between arms, TI-A records against I1
- Label: none I; swept sine E

### 5.9 Excitation amplitude (extra)
- Training 0.5 and 1.0 A_prod on standstill, 1.0 on moving records (the limiter may scale a record down; labels read the realised level)
- Test I: 0.75 A_prod (TI-M1); test E: 1.5 A_prod (TE-M2) and A_prod/6 (TE-M3, session 1's low level)
- Why: friction is Type I (Pintelon 2020 Table I): at A_prod the rails slide and the absorber band holds; at A_prod/6 they stick most of the time, the closed-loop BLA is 45 to 54 % off frictionless and the peak falls to 227 Hz (session 1); what reaches the rails is the injection filtered by the input sensitivity, and the stick band is V_BRK (D-209)
- Prediction: 0.75 interpolates; at 1.5 the physics-based arms improve relative to the output; at A_prod/6 every arm degrades, the augmented one least if its friction correction is a function of velocity rather than of the multisine level
- Falsifier: error at 0.75 outside the span of 0.5 and 1.0
- Criterion: S6 rule
- Label: I at 0.75; E at 1.5 and A_prod/6 by range

### 5.10 Excitation band (extra)
- Training B_tr; test E: TE-B1, standstill multisine over 1 Hz to 1 kHz excluding B_tr, at the same total rms as A_prod (session 1's broadband range) (?)
- Why: at frequencies no training record excites, the model is unconstrained; ASMPT measures FRFs broadband
- Prediction: outside B_tr the augmented model follows the baseline, which is right there apart from the absorber tails; the black box is unconstrained
- Falsifier: augmented error outside B_tr above the baseline's
- Criterion: error PSD outside B_tr against B-LPV and the floor PSD
- Label: E, categorical

### 5.11 Controller (core, R7)
- K1 (user decision 2026-09-26): the generator's rule-of-thumb diagonal controller in stage coordinates, crossover 100 Hz, designed once on the plant at Y = 0 and used unchanged in every training, validation and test record; only the K2 records of R7 use another controller
- Current generator, and why it changes
  - now: `gtd_build_plant.m` designs a new controller for every record, on the mass matrix at that record's Y_op
  - the mass each X rail feels depends on where the head sits along the beam: 17.2 to 25.7 kg over ±0.30 m over both rails (rigid-baseline hand estimate, high-frequency limit; the absorber adds dynamic mass on the truth below 150 Hz)
  - so the per-record gains differ by up to a factor of about 1.5, and the current training data hold a family of controllers, not one
  - problem 1: "train with one controller, then change it" (Quinten) is then not literally true
  - problem 2: a test at Y 0.39 m would get a controller with gains never used in training, so Y extrapolation and a controller change would be mixed in one record
  - problem 3: the real machine runs one fixed controller per axis (`dFeedbackControllersTelica`, schema), not one scheduled on Y
- With one K1, the loop still changes with Y because the plant changes (realised crossovers below); that is a plant effect, as on the machine, not a controller change
- Y_op keeps one role only: the linear limit pre-check of `gtd_enforce_limits` linearises the plant at the record's Y_op (and the record starts there, D-206); the controller no longer depends on it
- Realised loops with one design (hand estimates, approximate)
  - rails, rigid baseline: crossover 74 to 108 Hz over ±0.30 m; on the truth somewhat lower, since the absorber adds dynamic mass below 150 Hz through the yaw lever d and the offset L0; the gate below settles the truth values; about 66 Hz at the heavy rail near 0.39 m (a plant effect of Y extrapolation, not a controller change)
  - Y: below 150 Hz the absorber adds dynamic mass ma / (1 - (f/150)²), so the first Y crossover is about 83 Hz; the closed-loop peak lies at 266 Hz (half-power band about 233 to 299 Hz, session 1)
- Gate before generation: stability and minimum gain and phase margins of K1 and every K2 on T_AF over the full Y grid (±0.39), then the time-domain limit checks; a controller failing the gate gets no records
- Test E, R7 ladder: K2-g+ and K2-g- (all gains ±20 %, Kessels' size, core): +20 % moves every loop together, to about 120 Hz on the rails at Y 0, outside the trained 74 to 108 Hz; K2-lo (design 60 Hz) and K2-hi (design 150 Hz), conditional on the gate and on Quinten's answer to which changes are realistic (?); K2-ff = K1 plus acceleration feedforward with nominal masses through the injection port, an appendix use-case test (?); a notch only if Quinten names one
- K2-ff keeps K1's feedback and the total force stays about M(Y) a, so it tests the error ASMPT sees after feedforward, not an input-space extrapolation
- Applied to TI-A1, TI-A2, TI-A3, TE-A2 and TI-S3 (standstill multisine at Y 0); the absorber band for R7 is re-read per controller, since each controller moves the closed-loop peak (session 1)
- Prediction, per axis: the K1 to K2 change acts mainly on the Y loop around the closed-loop absorber peak, where the baseline misses the absorber: the baseline mispredicts the change of servo error there, the augmented model predicts it; on the rails the change is rigid-body and every physics-based arm predicts it
- Falsifier: the augmented model's error in predicting the change as large as the baseline's (it learned the K1 loop, not the plant)
- Criterion: per controller, the absolute prediction error against that controller's own floor (noise twins); on the multisine-free anchors, the predicted change e_K2 - e_K1 against the true change, in metres; replaces the NRMS ratio of RESULTS-DESIGN R8 (?); a model unstable under K2 fails
- Label: E, categorical

### 5.12 Plant: payload mass (extra, pending ASMPT)
- Training nominal payload; test E +2 kg on the rigid head mass only (a proposed head-mass perturbation; the datasheet's 2 kg is the rated payload under Z, and ASMPT should confirm a realistic change) (?); standstill at Y 0 and the TI-A1 route
- The absorber pole moves from 212 Hz to 150 sqrt(1 + 5.05/7.05) = 197 Hz, through the known rigid mass
- Arms: augmented with updated physics (mh + 2 kg, network unchanged), augmented with frozen physics, B-LPV updated, black box (cannot be updated)
- Why: die and tool changes are routine, and an updatable physical mh is the practical value of interpretability (R4)
- Prediction: below B_tr (rigid-body band) the updated augmented model stays near its nominal error, frozen models miss a 20 % change of the Y mass; in B_tr the updated model moves its absorber mode towards 197 Hz only if the learned mode couples through the physics head mass
- Falsifier: below B_tr, updated no better than frozen (its parameters are not physical)
- Criterion: S6 rule on the error below B_tr; B_tr reported against the predicted pole shift
- Label: E, categorical

### 5.13 Combined: the datasheet typical cycle (extra)
- ETEL S-curve at X 30 / Y 50 m/s², 1.95 m/s, T3 25 ms, along an explicit route: pick (0.04, -0.39), camera (0.09, -0.39), then targets chosen to visit both X ends (±0.35) and the Y top row (+0.325), about 10 s
- Why: the vendor's own cycle, with Y, X, velocity and acceleration at or beyond the training edge at once
- Reported as a stress-test score beside the single-axis E records, without a composition bound (nonlinear errors on several axes need not add)
- Label: E on several axes

## 6. Labelling rule (fixed before generation)
- Two labels per test record, both reported beside its error; the R6 ladder is ordered by the density measure, not by the name of the group
- Range label, per axis, on realised quantities of the generated references and injections (after the limiter's scaling): Y and X visited, peak reference velocity, acceleration and jerk per axis (X and Y separately), move distance, multisine level
  - training range = minimum to maximum over all training records and realisations; E if outside by more than 1 % (HEURISTIC: above the generator's timing quantisation, e.g. the FIR windows rounded up to 50 us)
  - categorical axes (controller, excitation class, band, truth): an untrained value is E
- Density label, on the network's physical inputs z = (X, Y, dX/dt, dY/dt, X1 - X2, d(X1 - X2)/dt, u_X1, u_X2, u_Y) of the truth's logged data, each divided by its machine value (0.375 m, 0.40 m, 2 m/s, 2 m/s, 6 mm, the training maximum of the yaw rate, 2000 N, 2000 N, 1420 N), sampled at 1 kHz
  - d = mean distance to the 5 nearest training samples; tau = 95th percentile of d over the training samples, leaving out every realisation of the same reference
  - per test record: fraction p of samples with d > tau, and median d / tau; one threshold for every test type: the largest p over all validation records, frozen before any test error is read
  - dense if p is at most that threshold, sparse otherwise
  - HEURISTIC, QSAR precedent (Sahigara 2012 Sect. 2.2, 5 neighbours, 95th percentile); the threshold is data-derived (check 8)
  - the logged u_total excludes d, but its feedback part reacts to d; at the calibrated sub-newton level this is negligible for the label
- Expected disagreements, listed before generation: the acceleration E records (range E, force inside the multisine's range, 5.4); the 0.1 mm records (range E, near the origin); TE-K21, K22 (feedforward: controller E, inputs close to K1); the standstill points at ±0.325 m (range E, as near the trained edge as interior points are to theirs)
- Never the convex hull: it counts empty regions as covered (Schweidtmann 2021 Fig. 5)

## 7. Sets
Common to every record unless stated: truth T_AF, controller K1, noise N_T (own realisation), 12 s (0.5 s hold, 10 s active, hold to 12 s), start at rest, multisine B_tr at A_prod with own phases. Kinematics give reference values: a = peak acceleration X / Y [m/s²], v = peak velocity [m/s], T3 = jerk time, j = peak jerk X / Y [m/s³].

### 7.1 Training (3 realisations)
| ID | Type | Y [m] | X_sym [m] | Kinematics | Multisine |
|-|-|-|-|-|-|
| TR-S1 to S5 | standstill | -0.30, -0.15, 0, 0.15, 0.30 | 0 | none | A_prod |
| TR-S6, S7 | standstill | -0.15, 0.15 | 0 | none | 0.5 A_prod |
| TR-Y1 to Y3 | Y sweep | 0 ± 0.30 | 0 | 0.2, 0.5, 0.75 Hz: v 0.38, 0.94, 1.41; a 0.5, 3.0, 6.7 | A_prod |
| TR-P1 | move, S-curve | setpoints in ±0.30 | setpoints in ±0.28 | L 0.25 (7.5 / 12.5, v 0.5), T3 36 ms, log-strata distances | A_prod |
| TR-P2 | move | ±0.30 | ±0.28 | L 0.5 (15 / 25, v 1.0), T3 30 ms | A_prod |
| TR-P3 | move | ±0.30 | ±0.28 | L 0.75 (22.5 / 37.5, v 1.5), T3 25 ms | A_prod |
| TR-P4 | move | ±0.30 | ±0.28 | L 0.75, T3 12 ms (j 1875 / 3125) | A_prod |
| TR-P5 | move with yaw | ±0.30 | ±0.14, X_anti ±1 mm | L 0.5, T3 25 ms | A_prod |
| TR-L1 | Lissajous | 0.30 at 0.35 Hz (v 0.66) | 0.28 at 0.85 Hz (v 1.50, a 8.0) | | A_prod |
| TR-L2 | Lissajous | 0.25 at 0.7 Hz (v 1.10, a 4.8) | 0.08 at 1.5 Hz (v 0.75, a 7.1), X_anti 1 mm at 0.8 Hz | | A_prod |
| TR-T1 | ILC-shape sine shuttle (TP1 lattice, Y_op 0) | -0.24 to 0.24 | inside ±0.10 | 40 / 80 mm, 0.75 max (22.5 / 37.5), v 0.76 / 1.38, j 1340 / 2040, dwell 0.30 s | none |
| TR-T2 | as TR-T1, mirrored directions (TP2) | -0.24 to 0.24 | inside ±0.10 | 0.45 max (13.5 / 22.5), v 0.59 / 1.07, dwell 0.45 s | none |
| TR-T3 | TP3 lattice, Y_op +0.06 | -0.26 to 0.30 | inside ±0.10 | 0.75 max, dwell 0.60 s | none |
| TR-T4 | TP4 lattice, Y_op -0.06 | -0.30 to 0.26 | inside ±0.10 | 0.45 max, dwell 0.30 s | none |

- Results: training data of every arm; R4 parameters; R5 on the noise-free twins
- Session 1: every record class excites all ten parameter combinations above the floor, the ILC profiles at 26 dB or more

### 7.2 Validation (3 realisations)
Inside the training range, not copies of training records (as V1 to V4, VP1, VP2 now); own phases, noise and setpoints.

| ID | Type | Y [m] | X_sym [m] | Kinematics | Multisine |
|-|-|-|-|-|-|
| VA-S1 to S4 | standstill | -0.225, -0.075, 0.075, 0.225 | 0 | none | A_prod |
| VA-S5, S6 | standstill | -0.075, 0.075 | 0 | none | 0.5 A_prod |
| VA-Y1 to Y3 | Y sweep | +0.05 ± 0.25, -0.05 ± 0.25, +0.05 ± 0.25 | 0 | 0.2, 0.5, 0.75 Hz: v 0.31, 0.79, 1.18 | A_prod |
| VA-P1 to P5 | moves as TR-P1 to P5 | new setpoint draws in ±0.30, so holds sit at Y training never holds at | as TR | same levels and jerk times | A_prod |
| VA-L1, L2 | Lissajous as TR-L1, L2 | Y centre +0.05, amplitude 0.25; centre -0.05, amplitude 0.20 | as TR | same X motion | A_prod |
| VA-T1 to T4 | ILC-shape as TR-T1 to T4 | on the lattice residue D-206 held out (0.04 mod 0.08) | inside ±0.10 | same levels | none |

- 20 records, 3 realisations
- Use: checkpoints, the number of added states (R2 order), black-box size and hyperparameters; never a reported score

### 7.3 Test interpolation
| ID | Type | Y [m] | X_sym [m] | Kinematics | Multisine | Results | Label |
|-|-|-|-|-|-|-|-|
| I1 (21) | twin of every TR record (TI-S1 to S7, TI-Y1 to Y3, TI-P1 to P5, TI-L1, L2, TI-T1 to T4) | as TR | as TR | as TR | as TR | R2 curve, R4 accuracy, R6 group I1, sufficiency check, anchors | I |
| TI-G1 to G16 (I2) | standstill grid | ±0.025 to ±0.275 every 0.025, not on training or validation points | 0 | none | A_prod | R1, R6 group I2 | I |
| TI-A1 | ASMPT sine, XY | shuttle inside ±0.20 | inside ±0.10 | 40 / 80 mm, 0.6 max (18 / 30), v 0.68 / 1.24, j 960 / 1460, dwell 0.3 s | none | R6 group I3, R7 anchor | I |
| TI-A2 | ASMPT sine, X only | -0.20 | inside ±0.10 | 40 mm, 0.6 max | none | cross-coupling, R7 anchor, X ladder | I |
| TI-A3 | ASMPT sine, Y only | shuttle inside ±0.20 | 0 | 80 mm, 0.6 max | none | cross-coupling, R7 anchor | I |
| TI-A4 | ASMPT S-curve, XY | inside ±0.20 | inside ±0.10 | as TI-A1, T3 25 ms (j 720 / 1200) | none | shape contrast; anchor of TE-A3, A7, J2 | I |
| TI-A5, A6 | ASMPT sine, XY | inside ±0.20 | inside ±0.28 | 50 / 50 mm; 200 / 200 mm (v capped 1.5), 0.6 max | none | R6 group I3; A6 anchor of TE-V | I |
| TI-A7 | ASMPT sine, XY | shuttle inside ±0.20 | inside ±0.10 | 40 / 80 mm, 0.3 max (9 / 15), v 0.48 / 0.87, j 340 / 520 | none | R6 group I3, lower level | I |
| TI-A8 | ASMPT sine, 5 mm, X and Y alternating | inside ±0.20 | inside ±0.10 | 0.3 max, j 960 / 2060, dwell 0.1 s | none | R6 group I3, short distance | I |
| TI-A9 | ASMPT sine, X only | -0.20 | 0.21 to 0.25 | 40 mm, 0.6 max | none | X ladder (edge) | I |
| TI-J1 | move, new seed | ±0.30 | ±0.28 | L 0.5, T3 18 ms (j 830 / 1390) | A_prod | jerk axis (R6 appendix) | I |
| TI-M1 | standstill | 0.15 | 0 | none | 0.75 A_prod | amplitude axis (R6 appendix) | I |

### 7.4 Test extrapolation
| ID | Type | Y [m] | X_sym [m] | Kinematics | Multisine | Anchor | Results | Label |
|-|-|-|-|-|-|-|-|-|
| TE-Y1 to Y8 | standstill | ±0.325, ±0.35, ±0.375, ±0.39 | 0 | none | A_prod | TI-G | R1, R6 | E (Y) |
| TE-Y9 | move, explicit setpoints | reaching ±0.39 | ±0.28 | L 0.5, T3 30 ms | A_prod | I1 twin of TR-P2 | R1, R6 | E (Y) |
| TE-Y10 | ILC-shape sine, XY | shuttle to ±0.39 | inside ±0.10 | as TI-A1 | none | TI-A1 | R6 | E (Y) |
| TE-X1 | move, explicit one-sided setpoints | ±0.30 | 0.28 < X ≤ 0.36 | L 0.5, T3 30 ms | A_prod | I1 twin of TR-P2 | R6 | E (X) |
| TE-X2 | ASMPT sine, X only | -0.20 | 0.30 to 0.36 | 40 mm, 0.6 max | none | TI-A2, TI-A9 | R6 | E (X) |
| TE-A1 | ASMPT sine, XY | inside ±0.20 | inside ±0.10 | 0.875 max (26.25 / 43.75), v 0.82 / 1.49, j 1690 / 2560 | none | TI-A1 | R6 | E (a) |
| TE-A2 | ASMPT sine, XY | inside ±0.20 | inside ±0.10 | 1.0 max (30 / 50), v 0.87 / 1.5, j 2060 / 3340 | none | TI-A1 | R6, R7 anchor | E (a, j) |
| TE-A3 | ASMPT S-curve, XY | inside ±0.20 | inside ±0.10 | 1.0 max, T3 25 ms (j 1200 / 2000) | none | TI-A4 | R6, acceleration only | E (a) |
| TE-A4, A5 | ASMPT sine, X only; Y only | -0.20; shuttle | inside ±0.10; 0 | 1.0 max | none | TI-A2; TI-A3 | cross-coupling at the maximum | E (a, j) |
| TE-A6 | move | ±0.30 | ±0.28 | L 1.0 (30 / 50), v 1.5, T3 25 ms | A_prod | I1 twin of TR-P3 | R6, acceleration only | E (a) |
| TE-A7 | ASMPT S-curve, XY | shuttle 0.22 to 0.30 | inside ±0.10 | as TE-A3 | none | TI-A4 | R6, largest force | E (a) |
| TE-A8 | the measured ILC move, exact | inside ±0.20 | inside ±0.10 | 39.3 / 50.7 (X 1.31 x max), v 1.0 / 1.5, j 3090 / 3430 | none | TE-A2 | R6 stress record | E (a, j), beyond max |
| TE-V1, V2 | ASMPT sine, XY long | -0.28 to 0.28 | -0.25 to 0.25 | 0.6 max, v 1.75; v 1.95 | none | TI-A6 | R6 | E (v) |
| TE-J1 | move | ±0.30 | ±0.28 | L 0.75, j 1.5x realised training maximum (T3 about 8 ms) | A_prod | I1 twin of TR-P3 | R6 | E (j) |
| TE-J2 | ASMPT S-curve, XY | inside ±0.20 | inside ±0.10 | 40 / 80 mm, 0.6 max, T3 8 ms (j 2250 / 3750) | none | TI-A4 | R6 | E (j) |
| TE-D1 | ASMPT sine, 0.1 mm, X and Y alternating | inside ±0.20 | inside ±0.10 | 0.6 max, dwell 0.1 s | none | TI-A8 | R6, declared multi-axis | E (d, j) |
| TE-D2 | ASMPT S-curve, 0.1 mm | inside ±0.20 | inside ±0.10 | T3 25 ms | none | TI-A8 | R6 | E (d) |
| TE-D3 | ASMPT sine, 0.1 mm | inside ±0.20 | inside ±0.10 | 3.8 m/s², j about 1860 | none | TI-A8 | R6, distance out of range | E (d) |
| TE-K1 to K8 | TI-A1, A2, A3, TE-A2 under K2-g+ and K2-g- | as source | as source | as source | none | the K1 record | R7, core | E (controller) |
| TE-K9, K10 | TI-S3 under K2-g+ and K2-g- | 0 | 0 | none | A_prod, own | TI-S3 and its second phase realisation | R7, core | E (controller) |
| TE-K11 to K20 (?) | the same five under K2-lo and K2-hi | as source | as source | as source | as source | the K1 record | R7, after the gate | E (controller) |
| TE-K21, K22 (?) | TI-A1, TE-A2 under K2-ff | as source | as source | as source | none | the K1 record | R7 appendix | E (controller) |
| TE-M2, M3 | standstill | 0.15; 0 | 0 | none | 1.5 A_prod; A_prod/6 | I1 twins of TR-S4, TR-S3 | amplitude axis (R6 appendix) | E (amplitude) |
| TE-B1 (?) | standstill | 0 | 0 | none | 1 Hz to 1 kHz outside B_tr, total rms A_prod | I1 twin of TR-S3 | band axis (R6 appendix) | E (band) |
| TE-B2 (?) | standstill, swept sine | 0 | 0 | none | swept over B_tr | I1 twin of TR-S3 | class axis (R6 appendix) | E (class) |
| TE-W1, W2 (?) | payload +2 kg: standstill; TI-A1 route | 0; as TI-A1 | 0; as TI-A1 | as source | A_prod; none | I1 twin of TR-S3; TI-A1 | plant axis (R6 appendix) | E (plant) |
| TE-C1 (?) | datasheet cycle, S-curve, explicit route | -0.39 to 0.325 | -0.35 to 0.35 | 30 / 50, v 1.95, T3 25 ms | none | single-axis E records | R6 stress score | E (several) |

- TI-S3 is the I1 twin of TR-S3 (standstill, Y 0); its second phase realisation is the phase-spread reference for its K2 twins, which keep their own phases

### 7.5 FRF references for R3 (test)
| ID | Type | Y [m] | Multisine | Why this Y |
|-|-|-|-|-|
| TF-1 to TF-3 | standstill, periodic multisine, several periods, 4 phase realisations each (session 1's robust method used 4) | 0.15, 0.2, 0.35 | B_tr, A_prod | trained point, between trained points (0.225 is now a validation point), beyond training |
| TF-4 (?) | as above | -0.35 | B_tr, A_prod | the truth is not symmetric in Y (absorber at +L0, m1 ≠ m2) |

- R3 compares the model's frozen-Y response with the truth's best linear approximation at the stated level (Quinten: an FRF is linear, the friction truth is not; friction does not show in it), plus the analytic frictionless FRF as reference
- Number of periods per record: set against the noise floor, which with force injection lies mostly below about 200 Hz, under B_tr (?)

### 7.6 Control truths (Q3)
| Truth | Records | Results |
|-|-|-|
| T_0, baseline only | 7.1 (1 realisation), 7.2, noise-free twins of 7.1 | R5 null control (zero bias); B-refit exact recovery (appendix) |
| T_OA, designed orthogonal addition (D-213, D-214) (?) | 7.1 (1 realisation), 7.2, noise-free twins of 7.1 | R5, only if the addition is truly orthogonal (Quinten); otherwise R5 is omitted |
| T_F, baseline + friction, no absorber (?) | 7.1 (1 realisation), 7.2, I1 | R2 second truth: knee at 0 added states, since friction is static in velocity |

- Own phases for every control-truth record (the user's rule); truths are compared over replicate phase ensembles, not on paired inputs
- An absorber-only truth is not included (user rule: never the absorber alone); Quinten's note "absorber alone first, then the combination?" is open (?)

### 7.7 Every result has its data, every record a result
| Result (slides numbering) | Records |
|-|-|
| R1 position dependence (LPV vs LTI, no augmentation and no black box, Quinten) | TI-G, TE-Y |
| R2 added states | order chosen on 7.2; curve reported on I1, on T_AF and T_F |
| R3 plant check | 7.5 |
| R4 interpretability | 7.1 (parameters), I1 (accuracy) |
| R5 recovery condition | 7.1 on T_0 and T_OA, noise-free twins |
| R6 generalisation, main | groups I1, I2, I3 (TI-A), E groups Y, X, a, v, j, d; cross-coupling (TI-A2, A3, TE-A4, A5); stress records TE-A8, TE-C1 |
| R6 appendix, one row per extra axis | shape (TI-A4), class (TE-B2), amplitude (TI-M1, TE-M2, M3), band (TE-B1), plant (TE-W), jerk (TI-J1, TE-J) |
| R7 controller transfer | TE-K with their K1 records |
| Sufficiency check (Quinten) | 7.1 against 7.2 (includes Y interpolation) and against I1 (same settings) |

### 7.8 Manifest (simulations)
| Block | Count |
|-|-|
| T_AF training, 21 records x 3 realisations | 63 |
| T_AF validation, 20 x 3 | 60 |
| Test interpolation (I1 21, TI-G 16, TI-A 9, TI-J1, TI-M1) | 48 |
| Test extrapolation (Y 10, X 2, A 8, V 2, J 2, D 3, K 10 core + 10 conditional + 2 appendix, M 2, B 2, W 2, C 1) | 56 |
| FRF references, 3 x 4 realisations (+4 for TF-4) | 12 |
| Noise-only twins of every test record except the FRF references | 104 |
| Second phase realisations of the multisine test records | 55 |
| T_AF total | about 398 |
| Control truths T_0, T_OA, T_F, 62 each | 186 |
| Campaign total | about 585 |

- Runtime about 1.4 min per record (D-212 status: 40 min for the current 29-record dataset), about 14 h of MATLAB, one job at a time (?)

## 8. Coverage display (setup section; optional in the thesis, Quinten: "if it helps")
- Figure A, scheduling plane: x = Y / 0.40, y = (dY/dt) / 2 m/s, all samples
  - training occupancy as shaded cells (grid of AUDIT M4); test records numbered as in van Haren 2022 Fig. 6, colour = range label, open marker = sparse by the density label; numbers match the result figures
  - stroke as a solid edge at ±1 (also the operating range, p. 4); training limit ±0.75 as a dashed "model limit" line (Schuet 2021 Fig. 2); ILC range ±0.5 dotted, labelled "historical ILC campaign coverage"
  - moving test records as thin orbits, standstill tests as markers on dY/dt = 0; X as a marginal strip, not a plane (the truth is X-invariant)
- Figure B, kinematic plane, one marker per record, panels X and Y: x = peak reference velocity / 2 m/s, y = peak reference acceleration / max, marker size = peak jerk
  - boxes: training maximum 0.75 (dashed), datasheet maximum (solid); TE-A8 as a named point outside (1.31 on X)
  - marker shape = profile shape, fill = controller, "d" beside 0.1 mm records, whose label comes from the distance range
- Table beside the figures: absolute vs applied limits per axis (Weigand Table 7 form)
- Results link: test error against median d / tau, one labelled point per test record (Wu 2017 Fig. 8)
- Not used: convex hull, 3D views, scatterplot matrices

## 9. Open questions: proposals
**Q1. Holding out operating points when moves traverse Y**
- Proposal: hold out the stroke ends from every training record (no record enters |Y| > 0.30); interior test points are standstill points between the training points, which training moves cross and move holds visit, so they test dense interpolation
- Why: an interior region held out from motion needs every training move to stay inside a band, which bans long strokes; that is the rejected 80 mm gap
- The ends are operational (the p. 4 pick station at -0.40 m), so the Y extrapolation test is directly ASMPT-relevant

**Q2. One data set for prediction and joint estimation**
- One data set; the detuning is a start value of the model, not a property of the data
- Confirmed by session 1: every record class as generated excites all ten combinations above the floor, and point-to-point references alone reach 29 dB or more; no parameter band is needed, B_tr = the absorber band

**Q3. Control truths**
- T_0 kept as the zero point of R5 and for the B-refit exact recovery
- T_OA replaces the absorber-at-10 % truth: R5 is shown only with a truly orthogonal addition (Quinten)
- T_F as R2's second truth (knee at 0 added states)

**Q4. Extrapolation axes that matter most for ASMPT, for Quinten** (?)
- 1 controller change; 2 acceleration and velocity to the datasheet maximum; 3 position to the stroke ends (pick station); 4 the ASMPT distances once confirmed; 5 payload change and X position after the external questions are settled

**Open points**
- B_tr = 140 to 390 Hz at A_prod, session 1's proposal (?)
- noise: on the input force (`NOISE-INJECTION.md`, agreed 2026-09-26); session 1's floor was computed with encoder noise, so its verdicts, mainly the low-frequency margins for joint estimation (smallest about 13 dB), need rechecking against the force-noise floor before the band and the joint-estimation verdict are final (?)
- Y origin relative to the beam centre (section 1) (?)
- the ASMPT profile set from Jasper and Dragan and the K2 set from Quinten: freeze before generation; if timing requires, generate the unaffected core records first (?)
- f_tr = 0.75 (?); log-strata distances (?); 8 ms as a synthetic jerk value (?); payload source (?); T_OA and T_F (?); absorber-only truth (?); FRF periods (?)
- R7 scored on absolute error and predicted change instead of the NRMS ratio, a change to RESULTS-DESIGN R8 (?)

## 10. Feasibility on paper (one line per element; nothing here changes the design)
- Y ±0.39 m: inside lim.pos_Y 0.40; X ±0.36 m inside lim.pos_X 0.375 with the 6 mm yaw limit
- Velocity 1.95 m/s: validate_response checks the simulated velocity against 2.0 m/s; a record whose realised velocity exceeds it is lowered, not forced
- Acceleration assert: `gtd_check_phaseA` asserts 30 / 50 m/s² on the reference; every record complies at 1.0 max up to the FIR discretisation, which may need a 1 % margin (?); TE-A8 needs a named exception; generation itself does not check acceleration (D-206)
- Limit check on the truth, binding: `gtd_enforce_limits` checks force, position, velocity and yaw on the linear loop of `gtd_build_plant`, which is the rigid baseline (full head mass at Y, no absorber, no friction), and scales the multisine on that basis; nothing checks the Simulink truth afterwards; the production multisine puts its power in the absorber band, where the truth departs most from that plant (closed-loop peak 266 Hz); so every record is checked after simulation on the simulated truth (peak and rms force, position, velocity, yaw against lim), a failing record is regenerated with its multisine scaled down, and the realised level is recorded; a generator change (?)
- The force lines below are screening values only (mixed hand estimates, 5.4); the post-simulation truth check decides
- Rail force, training: L 0.75 moves at about 990 N plus the A_prod multisine peak (about 500 to 750 N per rail) give about 1500 to 1750 N, inside 2000 N
- Rail force, test: TE-A6 (L 1.0 with multisine) about 1800 to 2050 N, so its multisine will be scaled and the realised level recorded; ASMPT routes at 1.0 about 1260 to 1315 N; TE-A8 about 1725 N (D-206's pre-check on the rigid baseline: 1685 N)
- RMS force: the A_prod multisine alone gives about 170 to 250 N rms per rail; back-to-back moves at L 0.75 or 1.0 (TR-P3, P4, TE-A6, TE-J1, TE-C1) may approach 916 N; longer holds fix it, with the active-motion time then stated per record (?)
- Standstill at 1.5 A_prod: about 255 N rms per rail and 270 N on Y, inside 916 / 656 N
- Beyond max: the X rail reaches 2000 N near 1.5 max without multisine (2000 / (32.0 x 1.37) = 45.6 m/s²); only TE-A8 goes there, at 1.31
- Y: 10.1 kg x 50 m/s² = 505 N, far inside 1420 N (rigid baseline; on the truth the absorber raises the Y dynamic mass below 150 Hz and lowers it above the pole, so the Y force in the absorber band comes from the truth check)
- Jerk: no jerk limit in the generator; T3 ≤ vmax/amax = 2/max (40 ms on Y) holds for every record, the longest trained T3 being 36 ms; TE-J1's T3 is set after the training references exist
- 0.1 mm moves: the FIR chain handles short moves (vmax reduced to sqrt(d amax), T3 kept); the sine profile's triangular branch as well (D-206)
- ILC-shape at 0.45 and 0.75 max: `gtd_build_records_telica` holds the D-210 kinematics in one `TEL` struct; a per-record acceleration field is needed (?)
- X-only and Y-only ASMPT routes: `gtd_make_reference_telica` asserts every move equals the stroke on both axes; a zero stroke on one axis needs a small change (?)
- ASMPT S-curve single moves with dwell and explicit setpoint lists (TE-Y9, TE-X1, TE-C1): a deterministic setpoint list through the existing move machinery (`ref_aprbs`) (?)
- Log-strata distances: `ref_aprbs` draws setpoints uniformly; a change (?)
- Fixed K1: `gtd_build_plant.m` designs at Y_op; the controller must be designed at Y = 0 while the pre-check plant stays at Y_op (?)
- Start pose: records must start at [0, Y_op] (D-206); TI-A9 and the TE-X records use an explicit equilibrium at their X offset, valid because the linear plant is X-invariant (?)
- Controller gate: stability and margins of K1 and every K2 on T_AF over Y up to ±0.39 before generation (5.11) (?)
- K2-ff: the injection port that carries the multisine carries the feedforward, and the pre-check already includes it
- Payload +2 kg: on the rigid head mass only; `gtd_config.m` sets ma = MA_FRAC x mh, which would otherwise grow the absorber (?)
- Per-record band field for TE-B1 and a sweep law for TE-B2 (?)
- Seeds: seed = 100 x track_id + record index collides beyond 100 records; a collision-free key (truth, set, record, realisation, twin, signal) with every seed stored in the manifest (?)
- Noise: force disturbance at the plant input (`NOISE-INJECTION.md`), sigma 0.31 / 0.34 / 0.10 N per 20 kHz sample, of which 0.05 / 0.06 / 0.02 N below 300 Hz; the servo-error floor it creates lies mostly below about 200 Hz; d is recorded for diagnostics only, never a model input; signals in double; checks: d = 0 reproduces the records bitwise, a friction-off standstill record reproduces 9.8 / 10.4 / 6.3 nm within 2 %
- Noise and the loop: calibrated at Y_op = 0 for every record, so the error level varies up to about 30 % on X over Y ±0.30, and it changes under every K2; the noise-only twins give each record and controller its own floor
- Friction and noise at standstill: Karnopp friction holds a rail at rest below 11 to 18 N, so the sub-newton noise barely moves it; standstill records whose rails stick (TR-S6, S7, VA-S5, S6, TE-M3) carry almost no noise and a floor near zero; at A_prod the rails slide (session 1) and the noise enters

### Implementation to do before generation
Generator changes the design requires; none of them depends on the open questions of section 9. Every new dataset goes to a new folder, with `cfg.out_dir` and `cfg.fig_dir` both overridden.
1. One fixed K1: design the controller at Y = 0 in `gtd_build_plant.m`, keep the pre-check plant at the record's Y_op (5.11)
2. Binding post-simulation check on the simulated truth: peak and rms force, position, velocity and yaw against lim; regenerate a failing record with its multisine scaled down and record the realised level (section 10)
3. Input-force noise block: a Sum block before the input `u` of the Extended ODE, fed by a From Workspace `d_in`, own seed per record; `d_in` recorded for diagnostics only (`NOISE-INJECTION.md` section 4); d = 0 must reproduce the records bitwise
4. Per-record acceleration for the ILC-shape records: `gtd_build_records_telica` holds one `TEL` struct; the levels 0.45 and 0.75 max need a per-record field (5.7)
5. X-only and Y-only routes: `gtd_make_reference_telica` asserts every move equals the stroke on both axes; allow a zero stroke on one axis (5.7)
6. Explicit setpoint lists: deterministic move sequences with dwell through the existing move machinery (`ref_aprbs`), for TE-Y9, TE-X1, TE-C1 and the ASMPT S-curve records (7.4)
7. Log-strata move distances in `ref_aprbs`, replacing uniform setpoints (5.6)
8. Collision-free seed key (truth, set, record, realisation, twin, signal), every seed stored in the manifest; the current `100 x track_id + k` collides beyond 100 records
9. Per-record band field for TE-B1 and a sweep law for TE-B2 (5.8, 5.10)
10. Payload on the rigid head mass only: `gtd_config.m` sets ma = MA_FRAC x mh, which would otherwise grow the absorber (5.12)
11. Acceleration assert: a named exception for TE-A8 in `gtd_check_phaseA`, and a 1 % margin at 1.0 max if the FIR discretisation needs it (section 10)
12. Controller gate: stability and margins of K1 and every K2 on T_AF over Y up to ±0.39, before any controller record is generated (5.11)

## 11. Critic points (`CRITIC.md`, first draft) and how they were handled
1. Labelling rule could not move I to E: two labels, ladder by density, expected disagreements listed (section 6)
2. tau set by twin realisations: every realisation of the same reference left out (section 6)
3. Scoring mode unstated: closed-loop from r and f, per-controller floors (section 2)
4. No test floor: noise-only twins and second phase realisations (section 2)
5. Jerk measured by T3: peak jerk is the axis (5.5)
6. TE-D1 a jerk and force test: declared multi-axis, TE-D3 added (5.6)
7. Absorber on the Y loop ignored: Y loop stated, session 1's 266 Hz peak cited, gate on T_AF (5.11)
8. ±20 % argument wrong: corrected, ±20 % is the core R7 rung (5.11)
9. Payload moves the absorber pole: stated, scored below B_tr (5.12)
10. LPV baseline not exact on T_AF: affine residual stated (5.1)
11. Validation subset and selection set: mirrors, sufficiency check on I1 (section 2)
12. R7 NRMS ratio: absolute error and predicted change (5.11) (?)
13. TI-A labels depend on a no-multisine training record: superseded, the ILC-shape records are in training (5.7, section 13)
14. E records without matched anchors: anchor column, explicit setpoints, segment scoring (7.4)
15. Acceleration E where force distance is weakest: TE-A7 near Y 0.30; force now on the truth coupling (5.4)
16. Generator limits missed: acceleration assert and rms force (section 10)
17. R2 order on test: chosen on validation (7.2)
18. Records without a result: result map with appendix rows (7.7)
19. Friction regime ignored the loop: stated, and session 1's measurement now used (5.9)
20. Numbers: rail masses 17.2 to 25.7 kg over both rails; growth to 0.39 m (5.1, 5.11)
21. Section 0 vs generator: damping override and anti channel stated (section 0)
22. T_AF10 not mass only: T_AF10 dropped for T_OA (7.6)
23. Range rule tolerance: 1 % quantisation tolerance on realised values (section 6)
24. 0.1 mm S-curve generator: FIR chain confirmed (5.6)
25. Yaw in density features: added (section 6)
26. Criteria without failure: X by equivalence, TE-C1 as a stress score, class by ranking (5.2, 5.13, 5.8)
27. Cross-coupling at the floor: differential rail error, torques on the truth (5.7)
28. Stroke-end fallback: superseded, the stroke ends are operational (5.1, section 13)

## 12. Independent review (`REVIEW.md`) and how each finding was handled
1. a_mach from the measured move: accepted by the user decision; datasheet 30 / 50, measured move as stress record TE-A8 (5.4)
2. Session 1 not incorporated: accepted; A_prod, B_tr 140 to 390 Hz (?), no parameter band, force and amplitude statements redone (sections 0, 5.4, 5.9, 10)
3. Density threshold undefined for new types: accepted; one validation threshold for every test type, frozen before errors are read (section 6)
4. Tolerance undefined for jerk, distance, amplitude: accepted; 1 % quantisation tolerance on realised values for every axis (section 6)
5. Phases paired across truths: accepted; own phases (7.6)
6. K1 stability not shown: accepted; the D-188 claim was wrong (per-record controllers); gate on T_AF over the full Y grid (5.11)
7. Records beyond 0.3625 m: rejected with the user; the datasheet cycle operates to -0.40 m and Garcia's own table would already exclude ±0.30; modelling assumption stated (section 1, 13)
8. Anchors and random references: accepted; explicit setpoint lists for TE-Y9, TE-X1, TE-C1; TE-D3 named "distance out of range, other kinematics inside" (5.6, 7.4)
9. Force on the rigid baseline: accepted; truth coupling (32.0 kg), final gate on simulated T_AF forces (5.4, section 10)
10. Campaign size understated: accepted; manifest (7.8)
11. TR-P6 weakens the separation: resolved differently; TR-P6 dropped, the ILC-shape records return to training as in the current dataset (D-206)
12. Profiles and controllers not frozen: accepted; freeze before generation, core first if needed (section 9)
13. Payload source and intervention: accepted; proposed perturbation pending ASMPT, updated and frozen arms (5.12)
14. Seeds and sampling: accepted; tuple key, log strata (sections 5.6, 10)
15. Jerk arithmetic: partly; the measured Y move has a 6.8 ms cruise (D-210), so its jerk is a π / Ta = 3430 m/s³, not the pure-cycloid 3200; X and Y jerk now reported separately everywhere, incl. 0.1 mm Y (4.1e4)
16. Cross-coupling on baseline coupling: accepted; truth values 24 and 30 N m (5.7)
17. X invariance predicted BB failure: accepted; equivalence per arm (5.2)
18. Combined-cycle bound: accepted; stress score (5.13)
19. Inconsistent IDs: accepted; TI-S3, TE-K numbering unified (7.4)
20. Validation wording: accepted (section 2, 7.2)
21. ILC range status: accepted; "historical ILC campaign coverage" (sections 1, 8)

## 13. Changes of 2026-09-26 (user decisions and new inputs)
- Maximum acceleration: the datasheet's 30 / 50 m/s² (p. 2, the p. 4 cycle runs at it), replacing the measured 39.3 / 50.7; every level is recomputed; the measured move is TE-A8
- Y range, old line and why it changed: the draft said "fallback if the geometry rules out |Y| > 0.3625 m: E points ±0.325, ±0.35, ±0.36 only", from Garcia's cross-arm length (0.725 m, rails at ±0.3625); it is replaced because (1) the datasheet cycle (p. 4) puts the pick station at Y = -0.40 m, so the ends are operational, and (2) Garcia's table also lists a 0.25 m payload, which on a 0.725 m arm limits the head to ±0.24 m and his validation move to ±0.2 m, so the geometry argument taken literally would already exclude the current ±0.30 data; the simulator is stated as a synthetic benchmark (section 1)
- ILC-shape profiles back in training and validation without multisine, as in the current dataset (TP and VP, D-206), at 0.45 and 0.75 max; the no-multisine patch record TR-P6 is dropped
- Production multisine level A_prod (6x `gtd_config.m`) and session 1's band; the amplitude axis gains A_prod/6, session 1's stick regime
- Quinten's comments that change data: R3 as a best linear approximation with its own periodic records (7.5); R5 only with a truly orthogonal addition (T_OA); R1 without augmentation and black box (no data change); black-box size by hyperparameter search on validation, training effort reported; noise inside the simulation, on the input
- Noise model: a force disturbance on the motor force, calibrated to the measured error level (`NOISE-INJECTION.md`), replacing the encoder injection of D-212; own realisation per record; the floor moves from mostly above 200 Hz to mostly below it; session 1's floor to be rechecked (section 9)
- Validation, as in the current dataset (V1 to V4, VP1, VP2): inside the training range but not copies of training records (user decision); the earlier record-for-record mirror is replaced; I2 drops the validation Y (16 points) and the R3 "between" point moves from 0.225 to 0.2 m
- Baseline vs true system: the generator's limit pre-check runs on the rigid baseline, so a post-simulation check on the simulated truth becomes the binding limit check (section 10); every hand estimate now says which model it uses; 5.1 gives the true system's growth beyond the training edge (coupling +25 / +39 %, yaw inertia +14 / +12 %) beside the baseline's
