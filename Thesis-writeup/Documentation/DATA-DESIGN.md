# Data design: training, validation and test (interpolation and extrapolation)

Design on paper only; generation starts after the user approves it (?). Numbers marked "hand estimate" are paper arithmetic from the generator's formulas, not simulations. Source precedence: user decisions, then session 1's `scripts/gantry/excitation-closed-loop/EXCITATION-VALIDATION.md`, then `RESULTS-DESIGN.md`, then the slides; result numbers follow the slides (R7 = controller transfer). History: first draft, examiner critique `CRITIC.md` (section 11), independent review `REVIEW.md` (section 12), user decisions and new inputs of 2026-09-26 (section 13).

## 0. Notation
- T_AF: the benchmark truth, physics baseline + Coulomb rail friction + payload absorber (ma 0.50 mh, damping 0.03 through `ZETA_A_OVERRIDE` as in the D-188 datasets, free-free pole 212.13 Hz, anti-resonance 150 Hz; in closed loop the peak sits at 266 Hz, session 1)
- K1: the one controller of every training, validation and test record, designed once at Y = 0 (not per record, as the current generator does); K2-*: changed controllers, R7 test records only (5.11)
- max: the datasheet maximum acceleration, X 30 and Y 50 m/s² (user decision 2026-09-26); level L: a = L x max and v = L x 2 m/s unless stated
- A_prod: the production multisine rms, 240 N symmetric, 87 N m anti, 180 N Y (6x the `gtd_config.m` values, the "a6" datasets; AUDIT M1, session 1)
- B_tr: the training multisine band, 106 to 297 Hz, decided 2026-09-26 (`EXCITATION-VALIDATION.md` section 16, D-219); no separate parameter band (Q2)
  - lines: every 0.5 Hz, 106.0, 106.5, ..., 297.0 Hz (383 lines), i.e. every 6th DFT line of the 12 s record, so the multisine is periodic with 2 s; each logical channel [sym, anti, Y] on all 383 lines with its own random phases, as the current generator does (channels not zippered)
  - level: A_prod per channel, unchanged; 383 lines instead of the current datasets' 1081 (140 to 230 Hz at 1/12 Hz), so 2.8x the power per line
  - lower edge 106 Hz: the highest K1 crossover (first |L_jj| = 1) over the three channels, the five training Y points, the nominal baseline and the measured truth (72 to 106 Hz); the multisine stays out of the controller bandwidth (user)
  - upper edge 297 Hz: the narrowest band above 106 Hz holding 90 % of the resolved FRF differences to T_AF (|E| > 2.45 sigma, full 3 x 3, averaged over Y = -0.30, -0.15, 0, +0.15, +0.30 with K1); nominal baseline 106 to 297 Hz, 10 % detuned baseline 106 to 290 Hz, so one band serves both; 80 % and 95 % give 111 to 276 and 106 to 323 Hz
  - holds the features: anti-resonance 150 Hz (Y from F_Y), cross-entry notch 208 to 213 Hz over Y (X1 from F_Y), closed-loop peak 264 Hz
  - line density: the narrowest feature, the anti-resonance, is about 9 Hz wide at -3 dB and gets 18 lines, against at least 4 per 3 dB width (Geerardyn, Rolain, Schoukens 2013)
  - confirmed on T_AF with this spectrum (B11: Y = 0, K1, A_prod, 10 realisations): the three features stay at the same frequencies, 33 Hz or more inside the edges; the FRF level moves 2 to 11 % from the broadband measurement because the rails stick less (X 8 % of the time against 12 %); B11 used zippered channels (1.5 Hz per channel) at the same rms per channel, the same band and rail power to first order
  - below 106 Hz lies 16 % (nominal) and 21 % (detuned) of the resolved differences; that part is the motion references' job, not yet checked on the references of 7.1 (section 9) (?)
  - replaces the placeholder 140 to 390 Hz and the current datasets' 140 to 230 Hz (`BAND_OVERRIDE` in `generate_trajectory_data_coulomb.m`)
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
- Because validation also interpolates in Y, "equal training and validation error" mixes data sufficiency with interpolation; the data-sufficiency check (Quinten) therefore also runs on I1, a same-setting twin of a training record in the test set that selects nothing
- One training fraction for every continuous axis: training covers up to 0.75 of the machine value; test extrapolation takes 0.875 and 1.0 (1.17x and 1.33x the training maximum); HEURISTIC (?)
  - anchored to the datasheet and to the realised maximum over all training records, so no test value lies below a trained one
  - precedents for the distance: Bolderman 2024 Fig. 13 (sweeps of maximum position, velocity, acceleration); Kessels R3 (peak acceleration +85 %, thesis p. 177); 1.33x is the milder end, fixed by the user decision that tests stop at the machine maximum
  - the one exception is the measured ILC move (1.31 x max on X), a named stress record, because ASMPT runs it
- One factor at a time from a matched interpolation anchor (Bolderman 2024 Fig. 13; Kessels Tables 5.6 and 5.7); records that change several axes are declared; moving extrapolation records are also scored on their out-of-range segments alone
- The unseen-profile claim (Aspect 4 ii) rests on new kinematics, distances and routes (X only, Y only, 0.1 and 5 mm moves, long strokes, the datasheet cycle); both profile shapes (S-curve, sine) are trained
- One controller K1 for every record except the controller tests (user decision), designed once, not per record (5.11)
- One truth (T_AF) for every set; control truths only where a result needs them (7.7)
- Realisations: training and validation in 3 data realisations (new phases and noise; new setpoint draws for moves), used as replicate runs, not pooled; noise-only twins (same r and f, new noise) give the floors: 4 in the first campaign, on E1, I1, I4 and E5; floor-relative criteria apply only to those records, all other records report absolute errors; second phase realisations of test records are deferred
- Noise N_T in every record; R5 also uses noise-free twins (d = 0) of its training records, since the recovery condition is derived noise-free (RESULTS-DESIGN S4)
- The noise matches the measured error below 300 Hz, in level and shape, not above: the Telica standstill error is mostly sensor noise, over 90 % above 200 Hz, which a force disturbance cannot produce; the force is calibrated to the measured spectrum below 300 Hz (3.4 / 3.7 / 1.1 nm rms, D-218), so each record carries about that, not the full 9.8 / 10.4 / 6.3 nm; the thesis calls it "a force disturbance calibrated to the measured error spectrum below 300 Hz", not "the Telica noise"
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
| Scheduling rate dY/dt | no dM/dt terms in truth or baseline (RESULTS-DESIGN 4c, the earlier version) | merged into velocity |
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
- Motion type, axis combination (X only, Y only, XY): a route, not a categorical axis; single-axis motion occurs in training (the Y sweeps are Y-only; in every point-to-point move the shorter axis finishes first and the other moves alone), so whether a route is covered is judged by the density label (section 6); used as the cross-coupling group (5.7)
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
The rows describe the full extrapolation kind. The first campaign generates only the records of section 7 (6 interpolation and 6 extrapolation test records); every other test record named in a row is deferred (7.5). Record IDs used in the rows: TI-A1 = I3, TI-A2 = I4, TI-A4 = I5, TE-Y10 = E2, TE-A3 = E3, TE-V2 = E4, TE-A8 = E6, K2-g+ on TI-A1 = E5; I6 replaces TI-A6 as the velocity anchor.

### 5.1 Payload position Y (core)
- Training -0.30 to +0.30 m for every record (75 % of the ±0.40 stroke); standstill points -0.30, -0.15, 0, 0.15, 0.30
- Validation: standstill at +0.10 (as the current V1); sweep, Lissajous and ILC-shape records on shifted centres and lattices inside ±0.30
- Test I, first campaign: standstill at -0.225 (I2); ASMPT routes inside ±0.20 (I3)
- Test E, first campaign: standstill at -0.39 (E1; on the true system the coupling grows 39 % towards -Y against 25 % towards +Y); I3's route shuttling out to -0.33 and +0.39 on its lattice (E2)
- Deferred: the rest of the between-points grid (every 0.025 m), standstill at ±0.325, ±0.35, ±0.375 and +0.39, and a move record reaching ±0.39
- Changed 2026-09-26: the line "fallback if the geometry rules out |Y| > 0.3625 m: E points ±0.325, ±0.35, ±0.36 only" is removed; why: the datasheet cycle runs to -0.40 m, and Garcia's own table would limit the head to ±0.24 m (section 13), so the ±0.36 cut had no basis
- Sources: stroke and operating range (p. 2, p. 4); 0.39 keeps the limiter margin; 0.025 m grid from the earlier RESULTS-DESIGN R1
- Why: Aspect 4 (i), held-out operating points; R1 (LPV vs LTI) and R6
- Prediction (hand estimates: rigid baseline from M_op of `gtd_build_plant.m`; true system from the same matrix with ma at Y + L0, absorber locked, i.e. below the 150 Hz anti-resonance)
  - from the training edge to ±0.39 m, the baseline's X-yaw coupling grows 28 % (+Y) and 32 % (-Y), its yaw inertia 13 %
  - the true system's X-yaw coupling (-0.686 - 10.1 Y) grows 25 % (+Y) and 39 % (-Y), its yaw inertia 14 % (+Y) and 12 % (-Y); this is what the models must extrapolate; the LPV baseline carries its own share, the LTI baseline none
  - the LPV baseline is not exact on T_AF: it puts all of mh at Y, the truth puts ma at Y + L0, leaving a constant X-yaw offset of 0.505 kg m and a yaw-inertia residual 1.01 Y + 0.05 kg m² (0.44 at 0.39, 8 %), which the augmentation must also extrapolate
  - the LTI residual at 0.39 m is 4.5 to 9x that; inside, all arms are close, because training moves cross every interior Y and move holds stop at random interior Y
- Falsifier: the end rise of the LTI arm not larger than that of the LPV arm, beyond the start spread
- Criterion: end rise per arm = error at each end point minus the mean interior error; compared with the S6 rule of RESULTS-DESIGN; floors from the noise twins
- Label: E for |Y| > 0.30 by range; interior dense or sparse by the density rule

### 5.2 X position, an invariance the model is not told (deferred)
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
- Test I, first campaign: I6, long strokes (X 0.50 m, Y 0.56 m, inside the trained positions) at 0.6 max with the velocity capped at 1.5 m/s
- Test E, first campaign: E4, the same strokes at 1.95 m/s, jerk about 330 / 920 m/s³ (inside); deferred: 1.75 m/s
- Sources: 2 m/s (p. 2, p. 4); 1.95 keeps a margin for the velocity check on the simulated response
- Why: throughput moves run at the speed limit; velocities are network inputs
- Prediction: viscous damping is in the physics, and Coulomb friction is a sign that a saturating tanh continues correctly, so the augmented model holds; the black box must produce damping linear in velocity beyond its data and saturates
- Falsifier: the augmented error rise from anchor to E at least as large as the black box's
- Criterion: S6 rule on the rises, whole record and above-1.5 m/s segments
- Label: E by range on realised peak reference velocity

### 5.4 Acceleration (core, user decision: datasheet maximum)
- Training reference acceleration up to 0.75 max, X 22.5 and Y 37.5 m/s² (move levels 0.25, 0.5, 0.75; ILC-shape records at 0.45 and 0.75); sweeps and Lissajous stay at or below 8 m/s²; validation mirrors it
- First campaign: I5 (S-curve at 0.6 max) with E3 (S-curve at 1.0 max, acceleration only); I3 (sine at 0.6 max) with E6 (the measured ILC move, declared stress record); the other records of this row are deferred
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

### 5.5 Jerk (deferred)
- The axis is the peak reference jerk j, since absorber-band content of an S-curve scales with it (above 1/T3 its acceleration spectrum falls as j / (π² f²)); T3 = a / j is the generator's knob
- Jerk-time rule for every S-curve record: T3 = (l + 0.5) / f_ring, with l chosen per record for the intended jerk level
  - THEORY: the jerk stage of the FIR chain puts zeros of the reference spectrum at k / T3 (Biagiotti and Melchiorri eq. 34); midway between two zeros the reference excites f_ring most for a given T3
  - Jasper's rule T3 = l / f_mode puts a zero on the mode so the machine does not ring after a move, the production goal; it is inverted here because the goal is identification; T3 > 0 always, since the machine never runs 0 (p. 4 cycle 25 ms; Jasper)
  - f_ring is the closed-loop ringing frequency, not the 212.13 Hz open-loop pole: about 264 Hz at the current production multisine (session 1 BLA peak 261.5 to 263 Hz, closed-loop peak 266 Hz), about 227 Hz when the rails stick (A_prod/6); it depends on the multisine's level and spectrum; re-read with B_tr itself (B11, 2026-09-26): closed-loop peak 264.0 Hz, unchanged, so the T3 values below stand
  - tolerance: within 3 dB of the lobe maximum for f_ring ± 0.25 / T3 (±19 Hz at 13.3 ms, ±7 Hz at 36 ms); on a zero at ± 0.5 / T3
  - regimes: the training S-curve records carry the multisine, so the rails slide and 264 Hz applies; the multisine-free test S-curve records (I5, E3) keep 25 ms, between zeros at both 227 Hz (5.68) and 264 Hz (6.60); 13.3 and 36 ms are between zeros at 264 Hz only (3.02 and 8.17 at 227 Hz)
  - proportion: the reference supplies 40 to 90 nm of absorber-band mismatch against about 33 000 nm from the multisine (session 1 J5, current multisine), so the placement is a small improvement, not a structural one; T3 = 0 would excite more, but is not a machine setting
  - changed 2026-09-26: the D-211 check at 212.13 Hz (open-loop pole, f x T3 off integers) is replaced by this rule; at 264 Hz, 12 ms sat near a zero (3.17), 30 ms on one (7.92), 18 ms at the 3 dB edge (4.75) and 8 ms near a zero (2.11)
- Training T3 (?): 13.3 ms (TR-P3, l 3), 25 ms (TR-P4, l 6), 28.4 ms (TR-P2, l 7; VA-P1 and I1 follow it), 36 ms (TR-P1, l 9); designed peak jerk up to TR-P3: X 1690, Y 2820 m/s³; the generator allows up to 2 jmax on triangular-acceleration moves, so the trained maximum is read from the generated references (section 6)
- Test I: TI-J1 (L 0.5, T3 17.0 ms, l 4, j 880 / 1470); ASMPT routes at 0.3 and 0.6 max
- Test E: TE-J1, moves at L 0.75 with T3 from the rule once the training references exist; with the designed maximum, l 2 gives 9.5 ms (2380 / 3960, 1.4x) (?); TE-J2, S-curve 40 / 80 mm at 0.6 max and T3 9.5 ms (1900 / 3170, 1.12x) or 5.7 ms (l 1, 3170 / 5280, 1.87x) (?); TE-A2 and TE-A8 (5.4)
- Sources: 25 ms (p. 4); ILC equivalents 13.5 / 15.7 ms (D-210) inside the trained T3 range; the shortest T3 the machine runs is pending from Jasper and Dragan, so the E jerk times are synthetic stress values (?)
- Longer T3 than trained: omitted; it only removes content above 1/T3, and T3 above vmax/amax = 2/max (67 ms X, 40 ms Y) stops a move reaching amax
- Why: jerk decides how much motion energy reaches the loop bandwidth and the absorber
- Prediction: during moves the baseline error in B_tr grows with j; the augmented model follows it only if its absorber correction is a mode driven by the input, not a fit to multisine statistics
- Falsifier: augmented error in B_tr during the E-jerk moves above the baseline's
- Criterion: error PSD in B_tr against the floor PSD from the noise twins, re-read around the 266 Hz closed-loop peak
- Label: E by range on realised peak jerk, X and Y separately

### 5.6 Move distance (deferred; the 40 / 80 mm ILC distance is in I3)
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
- First campaign: I3 (sine, X and Y together), I4 (alternating X-only and Y-only moves, cross-coupling, the two scored separately), I5 (S-curve twin of I3, shape contrast); deferred: the other routes (5, 50, 200 mm, 0.3 max, X edge)
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

### 5.8 Excitation class (core through the ASMPT records; swept sine deferred)
- Training: multisine on the identification records, none on the ILC-shape records (TR-T)
- Test: none on every ASMPT route (I by class); TE-B2, swept sine over B_tr at standstill (extra, sweep law, amplitude and duration to fix) (?)
- Prediction: the ranking of arms by error holds on multisine-off records
- Falsifier: the ranking reverses
- Criterion: S6 rule on pairwise differences between arms, TI-A records against I1
- Label: none I; swept sine E

### 5.9 Excitation amplitude (deferred)
- Training 1.0 A_prod only (the limiter may scale a record down; labels read the realised level); the half-level standstill records were dropped to keep training at 18 records, so 0.75 is also E by range once this axis is generated
- Test I: 0.75 A_prod (TI-M1); test E: 1.5 A_prod (TE-M2) and A_prod/6 (TE-M3, session 1's low level)
- Why: friction is Type I (Pintelon 2020 Table I): at A_prod the rails slide and the absorber band holds; at A_prod/6 they stick most of the time, the closed-loop BLA is 45 to 54 % off frictionless and the peak falls to 227 Hz (session 1); what reaches the rails is the injection filtered by the input sensitivity, and the stick band is V_BRK (D-209)
- Prediction: 0.75 interpolates; at 1.5 the physics-based arms improve relative to the output; at A_prod/6 every arm degrades, the augmented one least if its friction correction is a function of velocity rather than of the multisine level
- Falsifier: error at 0.75 outside the span of 0.5 and 1.0
- Criterion: S6 rule
- Label: I at 0.75; E at 1.5 and A_prod/6 by range

### 5.10 Excitation band (deferred)
- Training B_tr; test E: TE-B1, standstill multisine over 1 Hz to 1 kHz excluding B_tr, at the same total rms as A_prod (session 1's broadband range) (?)
- Why: at frequencies no training record excites, the model is unconstrained; ASMPT measures FRFs broadband
- Prediction: outside B_tr the augmented model follows the baseline, which is right there apart from the absorber tails; the black box is unconstrained
- Falsifier: augmented error outside B_tr above the baseline's
- Criterion: error PSD outside B_tr against B-LPV and the floor PSD
- Label: E, categorical

### 5.11 Controller (core, R7)
- First campaign: E5 = I3 under K2-g+ (K1 gains +20 %), the more informative direction since it raises the crossover towards the absorber; deferred: K2-g-, K2-lo, K2-hi, K2-ff and the standstill controller records
- First-campaign claim, narrowed: "transfer to +20 % gain on the I3 route", scored as the absolute prediction error on E5 against E5's own floor (its noise twin); no claim across controllers or routes, and no change score e_K2 - e_K1, since I3 and E5 carry independent noise
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
- Criterion: per controller, the absolute prediction error against that controller's own floor (noise twins); on the multisine-free anchors, the predicted change e_K2 - e_K1 against the true change, in metres; replaces the NRMS ratio of the earlier results design; a model unstable under K2 fails
- Label: E, categorical

### 5.12 Plant: payload mass (deferred, pending ASMPT)
- Training nominal payload; test E +2 kg on the rigid head mass only (a proposed head-mass perturbation; the datasheet's 2 kg is the rated payload under Z, and ASMPT should confirm a realistic change) (?); standstill at Y 0 and the TI-A1 route
- The absorber pole moves from 212 Hz to 150 sqrt(1 + 5.05/7.05) = 197 Hz, through the known rigid mass
- Arms: augmented with updated physics (mh + 2 kg, network unchanged), augmented with frozen physics, B-LPV updated, black box (cannot be updated)
- Why: die and tool changes are routine, and an updatable physical mh is the practical value of interpretability (R4)
- Prediction: below B_tr (rigid-body band) the updated augmented model stays near its nominal error, frozen models miss a 20 % change of the Y mass; in B_tr the updated model moves its absorber mode towards 197 Hz only if the learned mode couples through the physics head mass
- Falsifier: below B_tr, updated no better than frozen (its parameters are not physical)
- Criterion: S6 rule on the error below B_tr; B_tr reported against the predicted pole shift
- Label: E, categorical

### 5.13 Combined: the datasheet typical cycle (deferred)
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
  - per test record: fraction p of samples with d > tau, and median d / tau; one global threshold for every test record: the largest p over the 6 validation records pooled, not per record type, frozen before any test error is read (conservative for test types validation does not contain, stated as a limit)
  - dense if p is at most that threshold, sparse otherwise
  - HEURISTIC, QSAR precedent (Sahigara 2012 Sect. 2.2, 5 neighbours, 95th percentile); the threshold is data-derived (check 8)
  - the logged u_total excludes d, but its feedback part reacts to d; at the calibrated sub-newton level this is negligible for the label
- Expected disagreements, listed before generation: the acceleration E records (range E, force inside the multisine's range, 5.4); the 0.1 mm records (range E, near the origin); TE-K21, K22 (feedforward: controller E, inputs close to K1); the standstill points at ±0.325 m (range E, as near the trained edge as interior points are to theirs)
- Never the convex hull: it counts empty regions as covered (Schweidtmann 2021 Fig. 5)

## 7. Sets (first campaign)
Common to every record unless stated: truth T_AF; controller K1 (one design at Y = 0); input-force noise N_T with its own realisation; 12 s at 20 kHz (downsampled to 4 kHz for training), start at rest after the 0.5 s hold; multisine B_tr at A_prod with its own phases. Per training run the load equals the current dataset: 18 training and 6 validation records. Kinematics give reference values: a = peak acceleration X / Y [m/s²], v = peak velocity [m/s], T3 = jerk time, j = peak jerk X / Y [m/s³]; max = the datasheet maximum, 30 / 50 m/s².

### 7.1 Training (18 records, 3 realisations)
| ID | Type | Y [m] | X_sym [m] | Kinematics | Multisine | Current equivalent |
|-|-|-|-|-|-|-|
| TR-S1 to S5 | standstill | -0.30, -0.15, 0, 0.15, 0.30 | 0 | none | A_prod | T1 to T5 |
| TR-Y1 to Y3 | Y sweep | 0 ± 0.30 | 0 | 0.2, 0.5, 0.75 Hz: v 0.38, 0.94, 1.41; a 0.5, 3.0, 6.7 | A_prod | T6 to T8 (T8's X overlay dropped) |
| TR-P1 | move, S-curve (Jasper's generator) | setpoints in ±0.30 | setpoints in ±0.28 | 25 % of max (7.5 / 12.5, v 0.5), T3 36 ms, log-strata distances | A_prod | T9 |
| TR-P2 | move | ±0.30 | ±0.28 | 50 % (15 / 25, v 1.0), T3 30 ms | A_prod | T10 |
| TR-P3 | move | ±0.30 | ±0.28 | 75 % (22.5 / 37.5, v 1.5), T3 12 ms (j 1875 / 3125, the trained jerk maximum) | A_prod | T11 |
| TR-P4 | move with yaw | ±0.30 | ±0.14, X_anti ±1 mm | 50 %, T3 25 ms | A_prod | T12 |
| TR-L1 | Lissajous | 0.30 at 0.35 Hz (v 0.66) | 0.28 at 0.85 Hz (v 1.50, a 8.0) | | A_prod | T13 (wider in X) |
| TR-L2 | Lissajous | 0.25 at 0.7 Hz (v 1.10, a 4.8) | 0.08 at 1.5 Hz (v 0.75, a 7.1), X_anti 1 mm at 0.8 Hz | | A_prod | T14 |
| TR-T1 | ILC-shape sine shuttle, lattice Y_op 0 | -0.24 to 0.24 | inside ±0.10 | 40 / 80 mm, 75 % of max (22.5 / 37.5), v 0.76 / 1.38, dwell 0.30 s | none | TP1 (measured 39.3 / 50.7 replaced) |
| TR-T2 | as TR-T1, mirrored directions | -0.24 to 0.24 | inside ±0.10 | 45 % (13.5 / 22.5), v 0.59 / 1.07, dwell 0.45 s | none | TP2 |
| TR-T3 | lattice Y_op +0.06 | -0.26 to 0.30 | inside ±0.10 | 75 %, dwell 0.60 s | none | TP3 |
| TR-T4 | lattice Y_op -0.06 | -0.30 to 0.26 | inside ±0.10 | 45 %, dwell 0.30 s | none | TP4 |

- Training ranges: Y ±0.30 m, X ±0.28 m, velocity up to 1.5 m/s, acceleration up to 22.5 / 37.5 m/s², jerk up to the realised maximum (about 1875 / 3125 m/s³), move distance 1 mm to 0.6 m, multisine A_prod, controller K1
- Results: training data of every arm; R4 parameters; R5 on the noise-free twins
- Session 1: every record class excites all ten parameter combinations above the floor, the ILC profiles at 26 dB or more

### 7.2 Validation (6 records, 3 realisations)
Inside the training range, not copies of training records, as V1 to V4, VP1 and VP2 in the current dataset; own phases, noise and setpoints.

| ID | Type | Y [m] | X_sym [m] | Kinematics | Multisine | Current equivalent |
|-|-|-|-|-|-|-|
| VA-S1 | standstill | +0.10 | 0 | none | A_prod | V1 |
| VA-Y1 | Y sweep | +0.05 ± 0.25 | 0 | 0.5 Hz, v 0.79 | A_prod | V3 |
| VA-P1 | move, new setpoint draw | setpoints in ±0.30 | ±0.28 | 50 %, T3 30 ms | A_prod | V2 (over the full Y range instead of -0.30 to -0.14) |
| VA-L1 | Lissajous | centre -0.05, 0.20 at 0.7 Hz | 0.08 at 1.5 Hz | | A_prod | V4 |
| VA-T1, T2 | ILC-shape on the held-out lattice | lattices Y_op +0.04 and -0.04 (residue 0.04 mod 0.08, where no training ILC move stops) | inside ±0.10 | 75 % and 45 % of max | none | VP1, VP2 |

- Use: checkpoints, the number of added states (R2 order), black-box size and hyperparameters; never a reported score

### 7.3 Test interpolation (6 records)
Every axis inside its training range; controller K1.

| ID | Record | Y [m] | X_sym [m] | Kinematics | Multisine | Serves |
|-|-|-|-|-|-|-|
| I1 | move, same settings as TR-P2, new phases, noise and setpoints | ±0.30 | ±0.28 | 50 %, T3 30 ms | A_prod | R2 curve, R4 accuracy, sufficiency check |
| I2 | standstill, between training points | -0.225 | 0 | none | A_prod | R1; partner of E1 |
| I3 | ASMPT sine, X and Y together, shuttle | lattice residue 0.07 (Y_op -0.01), stops -0.17 to +0.15 | inside ±0.10 | 40 / 80 mm, 60 % of max (18 / 30), v 0.68 / 1.24, j 960 / 1460, dwell 0.3 s | none | R6 unseen route; partner of E2, E5, E6 |
| I4 | ASMPT sine, alternating X-only (40 mm) and Y-only (80 mm) moves | same lattice as I3 | inside ±0.10 | 60 % of max | none | R6 cross-coupling (differential rail error X1 - X2), X-only and Y-only segments scored separately |
| I5 | ASMPT S-curve, X and Y together | same lattice as I3 | inside ±0.10 | 40 / 80 mm, 60 % of max, T3 25 ms (j 720 / 1200) | none | partner of E3; shape contrast with I3 |
| I6 | ASMPT sine, long strokes | -0.28 to 0.28 | -0.25 to 0.25 | 0.50 / 0.56 m, 60 % of max, velocity capped at 1.5 m/s | none | partner of E4 |

### 7.4 Test extrapolation (6 records)
Exactly one axis beyond its training range, against one I partner that differs only in that axis; E6 is a declared two-axis stress record.

| ID | Record | Axis beyond training | Value (training edge) | Partner | Serves |
|-|-|-|-|-|-|
| E1 | standstill | Y | -0.39 m (-0.30); the true coupling grows 39 % towards -Y | I2 | R1 |
| E2 | I3's route on the same lattice, shuttling further out | Y, while moving | stops at -0.33 and +0.39 m (±0.30); with E1 at -0.39, both ends reach 0.39 | I3 | R1, R6 |
| E3 | ASMPT S-curve, T3 25 ms | acceleration only (j 1200 / 2000, inside) | 30 / 50 m/s², 100 % of max (22.5 / 37.5) | I5 | R6 |
| E4 | long strokes as I6 | velocity | 1.95 m/s (1.5) | I6 | R6 |
| E5 | I3 under K2-g+ (K1 gains +20 %) | controller | rail crossover about 120 Hz (74 to 108 Hz trained) | I3 | R7, narrowed to this one case (5.11) |
| E6 | the measured ILC move, exact (D-210 kinematics) | acceleration and jerk, beyond the datasheet maximum | 39.3 / 50.7 m/s² (X 1.31 x max), j 3090 / 3430 | I3 | R6, ASMPT's own move |

### 7.4b Pair definitions (fixed before generation)
Each pair shares one deterministic reference definition except the named axis; own phases and noise per record (the user's rule), so phase is a nuisance factor between records.
- Lattice for I3, I4, I5, E2, E3, E5, E6: Y residue 0.07 mod 0.08 (Y_op -0.01; chosen over 0.01 because its lattice reaches +0.39, complementing E1 at -0.39), used by neither training (residues 0, 0.02, 0.06) nor validation (0.04); held out are the route and the kinematics, not the operating points, since continuous training and validation motion passes every Y on it
- Common to the lattice records: start at [0, -0.01], first move +X +Y, full-stroke shuttle reflected at the range bound, dwell 0.3 s, X shuttle inside ±0.10, 10 s active
- I2 / E1: standstill multisine at Y -0.225 / -0.39; everything else equal
- I3 / E2: same lattice, start, direction sequence and dwell; I3 reflects inside ±0.20 (stops -0.17 to +0.15), E2 inside ±0.39 (stops -0.33 to +0.39)
- I3 / E5: identical reference; K1 against K2-g+
- I5 / E3: S-curve single moves (Jasper's FIR generator, T3 25 ms), same lattice, start, direction sequence and dwell; amax 60 % against 100 % of max, vmax 1.5 m/s in both; dependent changes (move duration, peak jerk 720 / 1200 to 1200 / 2000, peak velocity) are covariates, all inside training
- I6 / E4: sine long strokes, start at [0, 0], first a half stroke to (-0.25, -0.28), then full strokes of 0.50 m (X) and 0.56 m (Y), same sequence and dwell; velocity cap 1.5 against 1.95 m/s, acceleration 60 % of max in both; dependent changes (acceleration-phase length, cruise length) are covariates
- I3 / E6: E6 is the measured ILC move on I3's lattice, start and dwell; acceleration, jerk and velocity all change, so it is a declared two-axis stress record, not a one-axis pair
- Acceptance after generation: a table of realised differences per pair; every non-target axis must keep range label I, else the pair is reported as multi-axis

### 7.5 Deferred test records (left out of the first campaign)
Designed in section 5, generated and evaluated only if time allows (Quinten: start with the basics). In the thesis each is stated as not shown, for time, not relevance.
- Y position: the rest of the between-points grid (every 0.025 m), standstill at ±0.325, ±0.35, ±0.375 and +0.39, a move record reaching ±0.39 (5.1); R1 therefore rests on one standstill pair at -Y (I2 / E1) and one moving pair reaching -0.33 and +0.39 (I3 / E2), not a curve
- X position: moves on one side at 0.28 to 0.36 m, X-only moves at 0.30 to 0.36 m, the edge record at 0.21 to 0.25 m (5.2)
- Velocity: 1.75 m/s (5.3)
- Acceleration: sine at 0.875 and 1.0 of max, X-only and Y-only at 1.0, moves with multisine at 1.0, the S-curve near Y 0.30, the 30 % level (5.4)
- Jerk: moves at 1.5x the trained jerk, the S-curve at T3 8 ms, moves at T3 18 ms (5.5)
- Move distance: 0.1 mm (three variants), 5, 50 and 200 mm routes (5.6)
- Motion type: Y-only moves, so cross-coupling is shown on X-only moves only (5.7)
- Excitation: swept sine (5.8); multisine level 0.75, 1.5 and 1/6 of A_prod (5.9); band outside B_tr (5.10)
- Controller: -20 % gain, 60 and 150 Hz bandwidth, feedforward, the standstill controller records (5.11); R7 therefore rests on one controller change
- Plant: payload +2 kg (5.12)
- Combined: the datasheet typical cycle (5.13)
- Same-setting twins of the training record types other than the move record (I1)
- Second phase realisations of the multisine test records, and noise-only twins beyond one per record type and controller

### 7.6 FRF references for R3 (test)
| ID | Type | Y [m] | Multisine | Why this Y |
|-|-|-|-|-|
| TF-1 to TF-3 | standstill, periodic multisine, several periods, 4 phase realisations each (session 1's robust method) | 0.15, 0.225, 0.35 | B_tr, A_prod | trained point, between trained points, beyond training |
| TF-4 (?) | as above | -0.35 | B_tr, A_prod | the truth is not symmetric in Y (absorber at +L0, m1 ≠ m2) |

- R3 compares the model's frozen-Y response with the truth's best linear approximation at the stated level (Quinten: an FRF is linear, the friction truth is not; friction does not show in it), plus the analytic frictionless FRF as reference; no model rollouts, so cheap
- Fixed before generation, because R3's criterion depends on them: the number of periods per record, set against the noise floor (below 300 Hz with force injection, under B_tr), and the variance estimator, the robust BLA method over the 4 phase realisations (Pintelon 2020; noise and nonlinear distortion separated) (?)

### 7.7 Control truths (Q3)
| Truth | Records | Results |
|-|-|-|
| T_0, baseline only | 7.1 (1 realisation), 7.2, noise-free twins of 7.1 | R5 null control (zero bias); B-refit exact recovery (appendix) |
| T_OA, designed orthogonal addition (D-213, D-214) (?) | 7.1 (1 realisation), 7.2, noise-free twins of 7.1 | R5, only if the addition is truly orthogonal (Quinten); otherwise R5 is omitted |
| T_F, baseline + friction, no absorber (?) | 7.1 (1 realisation), 7.2, I1 to I6 | R2 second truth: knee at 0 added states, since friction is static in velocity |

- Own phases for every control-truth record (the user's rule)
- An absorber-only truth is not included (user rule: never the absorber alone); Quinten's note "absorber alone first, then the combination?" is open (?)

### 7.8 Every result has its data
| Result (slides numbering) | Records |
|-|-|
| R1 position dependence (LPV vs LTI, no augmentation and no black box, Quinten) | I2 with E1, I3 with E2; training standstill points as reference level |
| R2 added states | order chosen on 7.2; curve reported on I1 to I6, on T_AF and T_F |
| R3 plant check | 7.6 |
| R4 interpretability | 7.1 (parameters), I1 to I6 (accuracy) |
| R5 recovery condition | 7.1 on T_0 and T_OA, noise-free twins |
| R6 generalisation | I1 to I6 against E1 to E4 and E6, pairwise; cross-coupling on I4 |
| R7 controller transfer | E5 against its own floor, with I3 under K1 as reference; one named case (+20 % gain on the I3 route) |
| Sufficiency check (Quinten) | training against validation (includes Y interpolation) and against I1 (same settings) |

### 7.9 Manifest (simulations)
| Block | Count |
|-|-|
| T_AF training, 18 records x 3 realisations | 54 |
| T_AF validation, 6 x 3 | 18 |
| Test interpolation | 6 |
| Test extrapolation | 6 |
| FRF references, 3 Y x 4 realisations | 12 |
| Noise-only twins of E1, I1, I4 and E5 | 4 |
| T_AF total | 100 |
| Control truths, committed: T_0 42 (18 + 6 + 18 noise-free), T_F 30 (18 + 6 + 6) | 72 |
| Committed total | 172 |
| Conditional: T_OA 42 (18 + 6 + 18 noise-free), generated only after its orthogonality check passes | 42 |
| Maximum total | 214 |

- Runtime about 1.4 min per record (D-212 status: 40 min for the current 29-record dataset), about 4 h of MATLAB for the committed set, one job at a time
- Per training run: 18 training and 6 validation records, as with the current dataset; per trained model: 12 test rollouts

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
- Confirmed by session 1: every record class as generated excites all ten combinations above the floor, and point-to-point references alone reach 29 dB or more; no parameter band is needed; B_tr (106 to 297 Hz, section 0) holds the absorber features and the detuning differences above the crossover, and the same band serves the nominal and the detuned baseline

**Q3. Control truths**
- T_0 kept as the zero point of R5 and for the B-refit exact recovery
- T_OA replaces the absorber-at-10 % truth: R5 is shown only with a truly orthogonal addition (Quinten)
- T_F as R2's second truth (knee at 0 added states)

**Q4. Extrapolation axes that matter most for ASMPT, for Quinten** (?)
- 1 controller change; 2 acceleration and velocity to the datasheet maximum; 3 position to the stroke ends (pick station); 4 the ASMPT distances once confirmed; 5 payload change and X position after the external questions are settled

**Open points**
- B_tr decided: 106 to 297 Hz on a 0.5 Hz grid at A_prod (section 0, D-219); open: whether the references of 7.1 cover the resolved differences below 106 Hz (16 % nominal, 21 % detuned) and the friction velocity range (both directions, at least 3 constant-speed levels, reversals, dwell); checked once the references are generated (?)
- noise: on the input force (`NOISE-INJECTION.md`, agreed 2026-09-26); session 1's floor was computed with encoder noise, so its verdicts, mainly the low-frequency margins for joint estimation (smallest about 13 dB), need rechecking against the force-noise floor before the band and the joint-estimation verdict are final (?)
- Y origin relative to the beam centre (section 1) (?)
- the ASMPT profile set from Jasper and Dragan and the K2 set from Quinten: freeze before generation; if timing requires, generate the unaffected core records first (?)
- f_tr = 0.75 (?); log-strata distances (?); 8 ms as a synthetic jerk value (?); payload source (?); T_OA and T_F (?); absorber-only truth (?); FRF periods (?)
- R7 in the first campaign: absolute error on E5 against its own floor, one named case (5.11); the predicted-change score and the earlier NRMS ratio apply only if the deferred controller records are generated (?)
- if R1 needs more than one end point, swap E4 (velocity) for standstill at +0.39 m (?)

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
- Noise and the loop: calibrated at Y_op = 0 for every record, so the error level varies up to about 30 % on X over Y ±0.30, and it changes under every K2; so a floor holds only for the record it was measured on (the twins of E1, I1, I4 and E5)
- Friction and noise at standstill: Karnopp friction holds a rail at rest below 11 to 18 N, so the sub-newton noise barely moves it; standstill records whose rails stick (none in the first campaign, which runs every standstill record at A_prod; the deferred low-level records TE-M3 and the like) carry almost no noise and a floor near zero; at A_prod the rails slide (session 1) and the noise enters

### Implementation to do before generation
Generator changes the design requires; none of them depends on the open questions of section 9. Every new dataset goes to a new folder, with `cfg.out_dir` and `cfg.fig_dir` both overridden.
1. One fixed K1: design the controller at Y = 0 in `gtd_build_plant.m`, keep the pre-check plant at the record's Y_op (5.11)
2. Binding post-simulation check on the simulated truth: peak and rms force, position, velocity and yaw against lim; regenerate a failing record with its multisine scaled down and record the realised level (section 10)
3. Input-force noise block: a Sum block before the input `u` of the Extended ODE, fed by a From Workspace `d_in`, own seed per record; `d_in` recorded for diagnostics only (`NOISE-INJECTION.md` section 4); d = 0 must reproduce the records bitwise
4. Per-record kinematics for the ILC-shape and ASMPT records (TR-T, VA-T, I3 to I6, E2 to E4, E6): `gtd_build_records_telica` holds one `TEL` struct; acceleration level, stroke and velocity cap need per-record fields (5.7)
5. X-only and Y-only routes: `gtd_make_reference_telica` asserts every move equals the stroke on both axes; allow a zero stroke on one axis (5.7)
6. Explicit setpoint lists: deterministic S-curve move sequences with dwell through the existing move machinery (`ref_aprbs`), for I5 and E3 (and the deferred TE-Y9, TE-X1, TE-C1)
7. Log-strata move distances in `ref_aprbs`, replacing uniform setpoints (5.6)
8. Collision-free seed key (truth, set, record, realisation, twin, signal), every seed stored in the manifest; the current `100 x track_id + k` collides beyond 100 records
9. Deferred records only: a per-record band field for TE-B1 and a sweep law for TE-B2 (5.8, 5.10)
10. Deferred records only: payload on the rigid head mass only; `gtd_config.m` sets ma = MA_FRAC x mh, which would otherwise grow the absorber (5.12)
11. Acceleration assert: a named exception for E6 (the measured ILC move) in `gtd_check_phaseA`, and a 1 % margin at 1.0 max for E3 if the FIR discretisation needs it (section 10)
12. Controller gate: stability and margins of K1 and K2-g+ (E5) on T_AF over Y up to ±0.39, before any record is generated (5.11); the other K2 only when they are generated
13. Multisine band B_tr (section 0): `BAND_OVERRIDE = [106, 297]` in `generate_trajectory_data_coulomb.m`, and a 0.5 Hz line step in `gtd_make_multisine.m`, which now puts a line on every DFT bin from f_low to f_high (`freqs = cfg.f_low : df : cfg.f_high`, df = 1/12 Hz); the new step is every 6th bin; the script's assert that the band holds the 212.13 Hz pole still passes; the multisine cache check compares seed, band, record length, Y_op and candidate count but not the line step, so the step must be added to it (and to the saved info), or a 1/12 Hz realisation of the same band is reused

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
18. Records without a result: result map with appendix rows (7.8)
19. Friction regime ignored the loop: stated, and session 1's measurement now used (5.9)
20. Numbers: rail masses 17.2 to 25.7 kg over both rails; growth to 0.39 m (5.1, 5.11)
21. Section 0 vs generator: damping override and anti channel stated (section 0)
22. T_AF10 not mass only: T_AF10 dropped for T_OA (7.7)
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
5. Phases paired across truths: accepted; own phases (7.7)
6. K1 stability not shown: accepted; the D-188 claim was wrong (per-record controllers); gate on T_AF over the full Y grid (5.11)
7. Records beyond 0.3625 m: rejected with the user; the datasheet cycle operates to -0.40 m and Garcia's own table would already exclude ±0.30; modelling assumption stated (section 1, 13)
8. Anchors and random references: accepted; explicit setpoint lists for TE-Y9, TE-X1, TE-C1; TE-D3 named "distance out of range, other kinematics inside" (5.6, 7.4)
9. Force on the rigid baseline: accepted; truth coupling (32.0 kg), final gate on simulated T_AF forces (5.4, section 10)
10. Campaign size understated: accepted; manifest (7.9)
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
- Quinten's comments that change data: R3 as a best linear approximation with its own periodic records (7.6); R5 only with a truly orthogonal addition (T_OA); R1 without augmentation and black box (no data change); black-box size by hyperparameter search on validation, training effort reported; noise inside the simulation, on the input
- Noise model: a force disturbance on the motor force, calibrated to the measured error spectrum below 300 Hz in level and shape (`NOISE-INJECTION.md`, D-218), replacing the encoder injection of D-212; own realisation per record; the floor moves from mostly above 200 Hz to below 300 Hz; session 1's floor to be rechecked (section 9)
- Multisine band B_tr: 106 to 297 Hz on a 0.5 Hz grid at A_prod (section 0, D-219), from the measured FRF of T_AF with K1 at the five training Y points; replaces the placeholder 140 to 390 Hz; the lower edge is the K1 crossover, so the differences inside the controller bandwidth are left to the references (to check, section 9)
- Validation, as in the current dataset (V1 to V4, VP1, VP2): inside the training range but not copies of training records (user decision); the earlier record-for-record mirror is replaced
- First campaign sized like the current dataset (user decision): training 18 records (the two half-level standstill records dropped, the 12 ms jerk time moved onto the 75 % move record), validation 6 records, all 12 s at 20 kHz downsampled to 4 kHz; test 6 interpolation and 6 extrapolation records, each E with one I partner that differs only in the extrapolated axis; everything else deferred (7.5)
- Baseline vs true system: the generator's limit pre-check runs on the rigid baseline, so a post-simulation check on the simulated truth becomes the binding limit check (section 10); every hand estimate now says which model it uses; 5.1 gives the true system's growth beyond the training edge (coupling +25 / +39 %, yaw inertia +14 / +12 %) beside the baseline's
- Audit (`scripts/gantry/data-set-design/DATA-AUDIT.md`), applied: axis combination is a route judged by the density label, not a categorical axis (section 4), so I4 stays interpolation; I4 alternates X-only and Y-only moves; I3 and I4 move to lattice residue 0.07, which training and validation do not use; exact pair definitions (7.4b); R7 narrowed to one named case scored against E5's own floor; noise twins named on E1, I1, I4 and E5; FRF references at 4 realisations with period count and variance estimator fixed; one global density threshold; manifest split into committed (172) and conditional (T_OA, 42); the R1 point count in 7.5 corrected
- Audit, not applied: replacing the velocity pair by a +Y standstill pair (R1 as reframed by Quinten is close to predetermined, E2 already reaches +0.39 while moving, and velocity is an open R6 test); replacing the second ILC validation record (7 of the 12 test records are ILC or ASMPT routes without multisine); extra validation coverage of yaw and the top move level; guaranteed single-axis training moves (training already contains single-axis motion); the result-criteria items (R1, R5, R7 criteria, power and equivalence margin, unit of replication) belong in `RESULTS-DESIGN.md`
