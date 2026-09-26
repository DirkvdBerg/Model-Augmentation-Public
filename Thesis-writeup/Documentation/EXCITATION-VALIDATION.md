# Closed-loop excitation validation: which excitation the data need

Task: `tasks/handoffs/2026-09-25-closed-loop-excitation-validation.md`. Consumer: session 2
(`tasks/handoffs/2026-09-25-data-set-design.md`). Truth = baseline + Coulomb rail friction + payload
absorber (ma 0.50, zeta 0.03), always. Sections 1 to 7 were written before any computation; section 8b
lists what changed after it.

Paths: every relative path in this file (`figures/`, `outputs/`, `matlab/`, `*.py`) is under
`scripts/gantry/excitation-closed-loop/`. The document moved here on 2026-09-26.

## 0. Status for review (2026-09-26): read this section first
Two questions are open, both for the true system Coulomb + MSD (never MSD alone):
- Q1: are the current closed-loop FRFs a sound basis for choosing the multisine band?
- Q2: do the motion references plus the multisine cover the FRF differences, for the nominal baseline and for the 10 % detuned baseline?

Basis now used (user decisions 2026-09-26):
- The FRF differences E_nom = PS_truth - PS_nominal and E_det = PS_truth - PS_detuned, full 3 x 3 in stage coordinates (inputs F_X1, F_X2, F_Y at the plant input; outputs X1, X2, Y servo error); PS_truth is the best linear approximation (BLA) of the Coulomb + MSD truth at the production multisine level, measured through the Coulomb generator (J2)
- The excitation is decided on the noiseless case. The Telica-floor results (sections 5, 6, 10 secondary) stay as information for when noise is added
- The band is what the motion references leave uncovered, placed on the mismatch; the references are checked from their own spectra (a reference r acts as an injection K r, section 14)

Valid now:
- BLA at Y = 0 with the controller designed at Y = 0 (case B1), gated by the friction-off case B0 against the analytic model (median error 2e-5)
- Model-side parameter check J4 (exact, noiseless): with point-to-point references all ten combinations separate (gamma 1.2 to 1.6); J_eff, cb_sum and d take 82 to 98 % of their information from the 140 to 230 Hz multisine (section 10)
- Reference-based prediction J5: passes its gate at 50 to 140 Hz and for Y in 140 to 230 Hz; fails at 20 to 50 Hz and for X above 140 Hz, where friction during motion is not the standstill BLA (section 14)
- Coverage J6 at Y = 0 (fig. 10): references reach about 50 Hz (X) and 70 Hz (Y), the current multisine covers 140 to 230 Hz, and 50 to 140 Hz is covered by neither; that gap holds the peak of the detuning part (60 to 100 Hz) and of the X mismatch (50 to 80 Hz); covered share of E_nom 20 to 47 %, of E_det 27 to 62 % (section 14)

Not valid, to redo:
- The BLA at Y = +/-0.30 (B3, B4) used a controller redesigned at +/-0.30. `DATA-DESIGN.md` section 5.11 freezes one controller K1 designed at Y = 0 for every record, so B3, B4, the Y-dependent cross-entry notch (204 to 224 Hz) and figs. 6 and 7 at +/-0.30 must be redone with K1
- FRF points must follow the training operating range of `DATA-DESIGN.md` section 5.1: Y from -0.30 to +0.30 m, standstill points -0.30, -0.15, 0, +0.15, +0.30 (X plays no role: neither model nor truth depends on X)
- The coverage check used the current records' references; `DATA-DESIGN.md` plans other levels and jerk times, which need the same check
- The record simulations J3 and J4 used the current generator's per-record controller K(Y_op) (standstill T1, T2, T4, T5 at +/-0.15 and +/-0.30; TP3, TP4 at +/-0.06), not K1; the motion records at Y_op = 0 are unaffected

Superseded proposals (kept in the text below for the record): features read from the Y to Y entry alone; the unsourced "+/- 3 half-widths" edge rule; 140 to 365 and 140 to 390 Hz; "keep 140 to 230 Hz" (section 9), which the coverage result (section 14) contradicts for the detuned case and for friction in X.

Current proposal (?): one multisine band of 50 to 230 Hz for both data sets, starting where the references stop; not yet checked in the coverage table, and pending the K1 BLA at the training points.

Planned, not run: the Coulomb + MSD BLA with K1 at Y = -0.30, -0.15, +0.15, +0.30 (four MATLAB runs of `matlab/bla_truth.m`, after a change to use K1 everywhere); then figs. 6, 7, 10 with the largest difference over the five points; then the 50 to 230 Hz band and the planned references in the coverage table.

## 1. What this must establish
- (a) The multisine band the absorber data must excite: where the nominal baseline and the truth differ in closed loop, above the floor
- (b) The frequencies the joint-estimation data must excite: per identifiable combination, where a 10 % detuning changes the closed-loop servo error above the floor, and whether two combinations can be told apart
- (c) Per record class: does it deliver (a) and (b) in closed loop, with an explicit verdict on the slow motion profiles and on the reference alone (no multisine)

## 2. Operating-point kind: every condition the closed-loop linearised response depends on
| member | enters through | used here |
|-|-|-|
| payload position Y | M(Y) quadratic (yaw inertia mh Y^2, X-yaw entry -mh Y); absorber lever arm Y + L0 in the yaw inertia | yes, full stroke |
| controller design point Y_op | Cfb frozen per record at Y_op (`gtd_build_plant.m`); moving records run K(Y_op) at every Y they visit | yes, pairs (Y, Y_op) |
| rail friction regime, per rail | Karnopp law (D-209): stuck (rail locked), reversing (zero-mean velocity, relay-like), sliding at bias velocity (constant force, no small-signal effect) | yes, standstill by BLA, sliding by derivation plus record check |
| excitation amplitude and spectrum | with friction the BLA depends on the input power spectrum and level (Pintelon 2020, Thm 1 item 2) | yes, two levels and two spectra |
| axis combination | which rails slide at the same time (X-only moves leave the Y rail at rest, and the reverse) | yes, as the per-rail regime above, read from the records |
| acceleration during motion | the linearisation of the self-scheduled M(Y) about a moving trajectory adds dM/dY times the acceleration, zero at standstill | yes, only through the time-domain record simulations (no frozen FRF exists) |
| absorber deflection delta_a | enters M only through ma delta_a; `delta_a` rms 4e-5 m against Y + L0 of 0.1 to 0.5 m | no: below 0.05 % of the entry |
| X position, yaw angle | appear nowhere in M, C, K of the model or the truth | no: no dependence exists in the simulated truth |
| velocity (Coriolis) | the truth ODE freezes M and has no Coriolis terms (`gtd_run_simulation.m:59-64`) | no: absent from the truth |
| controller variant (R8) | a changed controller changes S, hence the delivered spectrum | no: the R8 controllers are session 2's design (?) |
| payload mass, noise level | plant variation and noise are session 2's extrapolation kind; nm noise moves stick-slip timing only (closed-loop-noise G5 d) | no |

## 3. Chosen set
- Frozen FRFs (analytic): Y in {-0.40, -0.30, -0.15, 0, +0.15, +0.30, +0.40} m with K(Y); plus Y = +/-0.30 with K(0), the loop of the moving records
- Figures at Y = -0.30, 0, +0.30 (ends and centre of the current training range); the +/-0.40 machine stroke (datasheet p. 2) in the tables
- Truth BLA (Coulomb generator, periodic multisines, section 13 of the handoff):
  - B0: Y 0, level H, cc = 0 (chain check against the analytic frictionless truth)
  - B1: Y 0, level H, broadband 1 Hz to 1 kHz
  - B2: Y 0, level L = H/6, broadband (amplitude dependence, Pintelon 2020 Table I)
  - B3, B4: Y +0.30 and -0.30, level H, broadband
  - B5: Y 0, level H, 140 to 230 Hz only (the regime the absorber records actually run in)
  - H = the production multisine rms per logical channel: 240 N sym, 87 N m anti, 180 N Y (AUDIT M1)
- Sliding regime: derived, not simulated: a rail sliding with constant sign sees a constant Coulomb force, so its small-signal dynamics are the frictionless ones exactly; checked on the records by the share of time each rail reverses while the reference moves
- Records: T1 to T14 and TP1 to TP4 (training classes) plus E4 (APRBS, multisine off), from `augmentation_ma50_z03_b140-230_a6all_telica_coulomb` (D-208); the frictionless twins (`augmentation_ma50_b140-230_a6_z03`, `augmentation_ma50_z03_telica`) split the mismatch into absorber and friction parts
- Provenance caveat: those truth records carry the pre-D-209 stick band (D-209 measured the change at under 1 % on TP1) and TP carries the pre-D-210 kinematics (about 0.2 %); the BLA runs use the current ODE

## 4. The deciding quantity (handoff section 8)
- One identity links the three candidates: with the same r and f, u = K e + f in both loops, so e_m - e_t = S_o,m (y_t - G_m u_t) exactly, S_o,m = (I + G_m K)^-1
  - frequency form: plant mismatch times the delivered truth input u_t, seen through the model's sensitivity; the FRF view (candidate 1) and the servo-error difference (candidate 2) are the same quantity
  - parameter form: e(theta + dtheta) - e(theta) = -S_o G_m (dM s^2 + dC s + dK) G_m u to first order, the closed-loop output sensitivity (candidate 3)
- Proposal (?):
  - (a) decided by the servo-error difference e_t - e_nom simulated on each record (no linearisation, covers friction and transients); the FRF form explains where it lives and must agree with it at standstill
  - (b) decided by the closed-loop output sensitivities per combination, noise-whitened: the only form that says whether two combinations separate; the FRF form gives their frequency location
- Consequence for "delivered spectrum": the relevant input is u_total of the truth (plant input, reference-driven part included), not the injected f (Dirkx 2020 Eq. 2: U = S W)

## 5. Floor (provisional, data-derived)
- Source: Telica encoder-node noise, VAR(512) tanh reading (closed-loop-noise G3, `outputs/g3_source/hv_tanh.npz`), 12.8 / 13.5 / 9.0 nm rms
- Servo-error floor: Phi_e = S_o Phi_v S_o^H with S_o of the frictionless truth at the record's (Y, Y_op); equals the RESULTS-DESIGN S4 floor (difference of two noise realisations over sqrt(2)) in the linear loop, checked there to 0.3 % (G5 c)
- Gate: its integral must reproduce the generator's 11.05 / 11.52 / 6.78 nm (G6)
- Not separable below 300 Hz: encoder noise vs a 0.1 to 0.5 N force disturbance changes this floor up to 2x there (G6 item 1); every margin is reported in dB so it can be re-read (?)

## 6. Criteria, stated before computation
- Detection of a known change in Gaussian noise: deflection d with d^2 = sum over lines 2 |D_k|^2 / E|N_k|^2 (Whittle form; full-record DFT, positive lines)
- Error rates 5 % false alarm, 95 % detection (HEURISTIC choice) give d >= 3.29 (THEORY: Neyman-Pearson detector of a known signal, deflection criterion; textbook source not held in `literature/` (?))
- (a) band: every line of the proposed band must carry the mismatch at single-line detection, |D_k|^2 / E|N_k|^2 >= 5.4 (7.3 dB), at the production per-line force
- (a) band edges: the anti-resonance and the pole, each with n = 3 half-power half-widths (zeta f) on its outer side, widths measured on the level-H BLA (HEURISTIC n: covers most of each feature's phase transition)
- (a) record class verdict: in-band deflection of e_t - e_nom >= 3.29
- (b) per combination, per record set: single detection d_i >= 3.29 for a 10 % change; joint recovery 10 % / sigma_i >= 3.29 with sigma_i from the inverse Fisher matrix of all ten (Pintelon and Schoukens frequency-domain ML, known noise model)
- (b) separability: collinearity index gamma of the whitened sensitivities (Brun 2001 Eq. 13), critical range 5 to 20
- Margin reported everywhere as 20 log10(d / 3.29) dB

## 7. Physics predictions and what would show them wrong
- Absorber, Y channel, no coupling: relative mismatch dG/G_m = -r^2 ma / (mh - mh_r r^2), r = f / 150 Hz
  - -6 % at 50 Hz, -29 % at 100 Hz, -100 % at the 150 Hz anti-resonance, pole at 212.1 Hz, +100 % asymptote above (the payload looks half as heavy)
  - so the mismatch does not end above the pole; the upper edge is set by the floor and the force budget, not by a feature
  - wrong if: the closed-loop BLA shows the zero or pole more than one line off, or no asymptote
- Absorber in X1, X2: only through the yaw row (-ma d, constant) and the X-yaw mass entry; smaller than in Y; wrong if X shows it at Y's size
- Friction at standstill, level H: rails reverse at multisine rate; the Gaussian-input BLA of cc sign(v) is a damper cc sqrt(2/pi) / sigma_v, about 800 N s/m per X rail and 360 N s/m on Y
  - prediction: the 212 Hz peak broadens (added modal damping about 0.01 to 0.02), the 150 Hz zero does not move, changes of a few percent elsewhere
  - level L: the equivalent damping grows about 6x, the peak broadens further and the low-frequency response drops (sticking)
  - wrong if: the zero moves, or H and L give the same BLA (then friction is not Type I here)
- Friction in motion: rails slide with constant sign, so the small-signal dynamics are frictionless; wrong if the records show rails reversing during moves beyond the move ends
- Detuning at standstill (140 to 230 Hz, inertia-dominated): mass combinations excited; damping enters as c / (omega m), about 1e-3, and yaw stiffness as kb / (J omega^2), about 1e-3
  - prediction: kb_sum and cb_sum near or below the criterion on standstill records; cg1, cg2, cy weak
- Detuning with motion: acceleration (hundreds of N at 0.2 to 10 Hz) excites m_total, mh, J_eff, m_diff, d; velocity (up to 1.6 m/s) excites cg1 + cg2 and cy
  - yaw stiffness and damping need yaw motion: only T12 and T14 carry a yaw reference, so kb_sum and cb_sum are predicted weak on every other class and on the reference alone
  - cg1 and cg2 separate only through yaw velocity: predicted collinear (AUDIT C6 found -0.98 one-step)
  - m_diff and mh Y share the X-yaw entry: separable only where Y varies during X acceleration
- Friction mismatch lives at velocity reversals and at low frequency; the absorber-band multisine does not target it

## 8. Computations (one at a time, under the watchdog)
| run | what | gate before trusting it |
|-|-|-|
| J1 | analytic closed-loop FRFs: nominal, detuned (production vector `[1.1, 1.1, 0.9, 1.1, 0.9, 0.9, 1.1, 1.1, 0.9, 1.1]`, D-191), each combination +10 %, frictionless truth; floor | floor integral equals G6; analytic PS equals the replica's line ratio on T3 |
| J2 | truth BLA, B0 to B5, robust method (M 4 random-phase realisations, 1 transient + 2 periods of 2 s, channels zippered on the 0.5 Hz grid) | B0 equals the J1 frictionless truth |
| J3 | replica closed loops of nominal, detuned and ten +10 % models on every record, with and without f; differences, sensitivities, Fisher matrices, delivered spectra | vendored replica reproduces a Simulink baseline record (CL-003 check) |

Run log (all under `tools/wd.sh`; logs in `outputs/<run>.log`; nothing trained, no dataset written):
| run | outcome |
|-|-|
| J1 | killed at 2 s by the disk line (C: 1.81 GB free < 2 GB); J1b with outputs cut to 22 MB and the disk kill line at 1.5 GB: done |
| J1 gate | floor rms 11.10 / 11.55 / 6.78 nm against G6 11.07 / 11.56 / 6.78: PASS |
| J3 gates | G-J3a replica vs Simulink null records, T3 rms below 7e-13 m, T9 rms 1.6e-9 m, max 3.5e-8 m (limits 5e-8 / 2e-7): PASS; G-J1b analytic PS vs replica at 1078 T3 lines, median error 0.3 to 0.5 %: PASS |
| J3 | 19 records, 24 closed loops each, 217 s; every frictionless twin matched its friction record (same r, f) |
| J2_B0 | killed at 56 s by the RAM line lowered to 0.4 GB (MATLAB tree 2.4 GB, 2.8 GB available at launch) |
| J2 | user: ignore the limiter; RAM kill off, disk guard 0.3 GB, launched from the repo root to reuse the model cache; B0, B1, B2 saved, then killed in B3 by the disk guard (C: 0.25 GB, pagefile growth under RAM pressure) |
| J2b | B3, B4, B5 after RAM recovered to 6 GB: done, peak tree 3.1 GB |
| J2 gate | G-J2: B0 (cc = 0) BLA against the J1 analytic truth, median error 2e-5 per column: PASS (first read FAILED at exactly 2.0, a sign convention of the post-processing: injection to error vs PS; fixed to PS) |
| J4 | noiseless detuning check (`j4_noiseless.py`), 19 records, 53 s; run WITHOUT the watchdog on the user's instruction (C: at 0.22 GB free made the watchdog kill it at launch) |
| J5 | reference-based prediction (`j5_reference.py`): first gate FAIL (30 to 10000x too high: point-to-point records end elsewhere than they start, the plain DFT saw that jump); with one Hann window on reference, multisine and simulation: see section 14 |
| J6 | coverage of the FRF differences by references and multisine at Y = 0 (`j6_coverage.py`, fig. 10) |

## 8b. Revisions after the plan (2026-09-26)
- Truth is Coulomb + MSD only (user and supervisor Quinten). The frictionless (MSD-only) truth is kept only as the B0 pipeline gate; every result below uses the Coulomb + MSD records or its BLA
- The system is 3 x 3 (stage inputs F_X1, F_X2, F_Y; outputs X1, X2, Y). The first reading of features from the Y to Y entry alone, with logical inputs, is superseded by the full stage matrix (figs. 6, 7)
- The excitation is decided on the noiseless case (user). The deciding quantities become: separability of the ten combinations (collinearity index gamma, Brun 2001, critical 5 to 20; independent share r_i) with the training loss's own weighting (servo error in m, three stage channels), and the parameter signal against the unmodelled mismatch. The Telica-floor margins of section 6 stay as secondary information for when noise is added
- The band-edge rule of section 6 (features +/- 3 half-widths) had no source (literature search, section 9) and is withdrawn, with the 140 to 365 and 140 to 390 Hz proposals built on it

Figures (`figures/`):
- `fig5_baseline_vs_truth.png`: nominal and detuned baseline against the Coulomb + MSD BLA, Y and X1 channels (subset; superseded by fig. 6)
- `fig6_mimo_Y-0.30.png`, `fig6_mimo_Y+0.00.png`, `fig6_mimo_Y+0.30.png`: full 3 x 3, stage coordinates, magnitude and relative error per entry (`fig6_mimo.py`)
- `fig7_errors_3x3.png`: absolute error per entry, three Y positions, detuning alone and BLA std (`fig7_errors.py`)
- `fig8a_detuning_signal.png`: detuning signal from the reference alone and with the multisine, against the unmodelled mismatch, per class and channel (`fig8_detuned.py`)
- `fig8b_information_by_frequency.png`: cumulative information per combination, reference alone vs reference + multisine
- `fig2`, `fig3`, `fig4` (noise-floor view) stay as secondary; `fig1` uses the frictionless truth and is superseded

## 9. Nominal baseline vs Coulomb + MSD: the absorber data band
- Where they differ (3 x 3, figs. 6 and 7, BLA at production level H):
  - Y from F_Y: dip at 150 Hz at every Y; absolute error largest 100 to 300 Hz, peak about 2e-7 m/N near 260 Hz (the closed-loop peak)
  - cross entries X1, X2 from F_Y and Y from F_X1, F_X2: a notch in the truth at 204 to 224 Hz (the absorber pole in the coupling path), Y dependent (about 210 to 224 Hz at Y -0.30 m, 204 to 216 Hz at +0.30 m); error plateau 1e-8 to 2e-8 m/N from 30 to 230 Hz
  - rail entries X1, X2 from F_X1, F_X2: error peaks at 50 to 100 Hz, about 5x the BLA std: Coulomb friction at the loop bandwidth
  - below about 10 Hz the rail entries sit at the BLA's own uncertainty: not resolved by this measurement
- Friction BLA features, Y from F_Y (J2):

| case | Y [m] | level | stuck X1 / X2 / Y | nonlinear std / BLA | dip | closed-loop peak | upper -3 dB |
|-|-|-|-|-|-|-|-|
| B1 | 0 | H | 12 / 13 / 4 % | 2 to 5 % | 150.5 | 261.5 | 303.5 |
| B2 | 0 | L = H/6 | 83 / 86 / 39 % | 19 to 39 % | 150.5 | 227.0 | 296.0 |
| B3 | +0.30 | H | 13 / 18 / 4 % | 2 to 5 % | 150.5 | 263.0 | 303.5 |
| B4 | -0.30 | H | 16 / 13 / 3 % | 2 to 5 % | 150.5 | 263.0 | 305.0 |
| B5, 140 to 230 only | 0 | H | 8 / 9 / 5 % | 2 to 5 % | 151.5 | band edge | band edge |

- Gate B0 (cc = 0) against the analytic model: 2e-5; responses periodic to 2e-12 in every case, so the BLA framework applies (Pintelon 2020 Sec. III-B)
- Amplitude matters (Pintelon 2020 Table I, Type I): at one sixth of the production level the rails stick most of the time and the closed-loop peak falls from 262 to 227 Hz. The band holds only at the production rms
- Literature on band and power (one deep-research agent, 2026-09-26):
  - no sourced rule sets band edges at a number of half-widths around a mode (medium coverage; modal-testing books not read)
  - line density: about four excited lines inside the 3 dB bandwidth of each second-order mode (Geerardyn, Rolain, Schoukens, IEEE TIM 62(5):1364-1372, 2013, DOI 10.1109/TIM.2012.2232474; read via Geerardyn's 2016 VUB PhD thesis, ch. 2; the same count in modal testing for spectral bias)
  - fixed power budget: "the energy will be concentrated at the frequencies where it contributes most to the knowledge about the model parameters" (Pintelon and Schoukens, 2nd ed. 2012, p. 152); spreading is advised only for uncertainty in where the mode sits and for detecting model errors
  - closed loop: the design is posed on the plant-input spectrum (Bombois et al. 2011 eq. 5.4; Dirkx 2020 Theorem 3), so plant information sits at the plant features
- **Decision (superseded 2026-09-26 by section 14, kept for the record): keep 140 to 230 Hz at the production multisine rms**
  - it holds the dip (150 Hz) and every cross-entry notch (204 to 224 Hz at Y -0.30 to +0.30 m); the upper edge sits about 6 Hz above the highest notch, one pole half-width
  - line density: at 1/12 Hz spacing about 100 lines in the dip's 3 dB width and about 150 in the pole's, against the rule's four
  - widening to 390 Hz costs 4.4 dB per line and adds no plant feature
- Two conditions to state:
  - the 262 Hz peak (Y from F_Y only) lies outside the band; it follows from plant and controller, so a correct plant model reproduces it (derivation, not a sourced rule). Verification (?): one test record excited at 230 to 300 Hz; if the trained model predicts the peak it never saw, this holds and doubles as a frequency-extrapolation result
  - the friction mismatch in X (50 to 100 Hz, and below 20 Hz between the rails) is not targeted by this band; the motion records carry it (section 10)
- Delivered vs injected: in 140 to 230 Hz on the standstill records u_total is 251 / 247 / 117 N rms per stage rail against 170 / 170 / 180 N injected (1.47 / 1.45 / 0.65): feedback reshapes the band

## 10. 10 % detuned baseline vs Coulomb + MSD: the joint-estimation data
- Noiseless, training-loss weighting (J4). Separability per record set; gamma critical 5 to 20; r_i = independent share (1 orthogonal, 0 confounded):

| set | with multisine: gamma, lowest r_i | reference alone: gamma, lowest r_i |
|-|-|-|
| standstill | 6.3; J_eff 0.22, kb_sum 0.23 (kb_sum / J_eff -0.97) | no information |
| Y sweep | 6.3; J_eff 0.22 | 1.8; kb_sum 0.71 |
| APRBS | 1.3; d 0.91 | 1.3; kb_sum 0.92 |
| Lissajous | 1.4; d 0.87 | 1.4; mh 0.88 |
| ASMPT profiles | 1.6; kb_sum 0.78 (no multisine) | same |
| E4 (APRBS, no ms) | 1.5; kb_sum 0.80 | same |
| all training | 1.3; J_eff 0.92 | 1.2; kb_sum 0.95 |

- Where each combination's information comes from (all training, fig. 8b):
  - from the reference, below about 30 Hz: kb_sum 96 %, cg1 100 %, cg2 100 %, cy 100 %, m_total 97 %, m_diff 69 %, mh 54 %
  - from the 140 to 230 Hz multisine: J_eff 98 %, cb_sum 95 %, d 82 %, mh 45 %, m_diff 30 %
- Size against the unmodelled mismatch (fig. 8a):
  - point-to-point moves (APRBS, ASMPT): at 2 to 30 Hz the detuning signal from the reference alone is as large as truth minus detuned
  - Y sweep, Lissajous: the reference-alone detuning signal is 10 to 100x below the unmodelled mismatch at low frequency
  - above about 30 Hz the unmodelled mismatch exceeds the detuning signal in every class, in band by 5 to 10x (absorber)
- **Answer 1: the reference profile is enough for the low-frequency dynamics when the set holds point-to-point moves**: all ten separable on the reference alone (gamma 1.2 to 1.6), the low-frequency combinations take 96 to 100 % of their information from it, and its signal matches the unmodelled mismatch at 2 to 30 Hz. Smooth sweeps and Lissajous alone are weak (signal 10 to 100x below the mismatch)
- **Answer 2: the multisine band stays 140 to 230 Hz for the detuned data**: the multisine is needed only for J_eff, cb_sum and d (yaw inertia, yaw damping, payload offset), and 140 to 230 Hz supplies 82 to 98 % of their information; no low-frequency lines are needed. One multisine design serves both data sets
- Caveats: noiseless decides identifiability, not precision; standstill records alone are marginal (gamma 6.3); the unmodelled part is larger than the parameter signal above 30 Hz, a bias risk for joint estimation (orthogonality work), not an excitation gap
- Secondary, with Telica-level noise (J3, measured floor range >= 19.5 Hz): every class as generated detects all ten jointly with at least 12.5 dB; reference alone passes for APRBS and ASMPT (at least 25.7 dB) and fails for sweeps and Lissajous on kb_sum, cg1, cg2, cb_sum (fig. 3)

## 11. Verdict per record class
| class | absorber band 140 to 230 Hz | detuned case, reference alone | detuned case, with multisine |
|-|-|-|-|
| standstill + ms | delivers it | nothing | marginal alone (gamma 6.3, kb_sum / J_eff) |
| Y sweep + ms | delivers it | separable, weak against the mismatch | multisine adds J_eff, cb_sum, d |
| APRBS + ms | delivers it | enough (gamma 1.3) | enough |
| Lissajous + ms | delivers it | separable, weak against the mismatch | multisine adds J_eff, cb_sum, d |
| ASMPT profiles | absorber content far below its mismatch: no absorber record | enough (gamma 1.6) | (no multisine) |
- Rail regime while moving: reversals in 0.1 to 1.4 % of moving time per rail: the motion records see the frictionless small-signal dynamics apart from the Coulomb offset

## 12. Predictions against outcome
| prediction (section 7) | outcome |
|-|-|
| absorber dG/G: -6 % at 50 Hz, -29 % at 100 Hz | -5.7 %, -27.8 %: held |
| zero 150 Hz, pole 212.1 Hz, within one line | 150.85 and 213.3 Hz: missed by 1.7 and 2.4 lines (yaw coupling through d) |
| no feature above the pole | wrong: the closed loop puts a peak at 262 Hz in Y from F_Y |
| absorber in X smaller than in Y | held in magnitude; but the cross entries carry the pole as a Y-dependent notch (204 to 224 Hz), missed by the first single-entry reading |
| friction broadens the peak, dip stays | held: upper half-width 33 to 42 Hz, dip within one line |
| H and L give different BLAs (Type I) | held: peak 261.5 vs 227 Hz |
| motion: rails slide | held: reversals 0.1 to 1.4 % of moving time |
| kb_sum, cb_sum weak on standstill | held noiseless (r_i 0.23), not against the Telica floor (21 and 30 dB) |
| kb_sum, cb_sum weak on reference alone except yaw references | wrong for point-to-point moves (r_i 0.92 and 1.00) |
| cg1, cg2 collinear | wrong on motion sets (r_i 0.97 to 1.00); held on standstill alone (r_i 0.35) |

## 13. Conflicts and open points
- Truth: Coulomb + MSD throughout (user decision wins over RESULTS-DESIGN S1, R2, R3, R5)
- RESULTS-DESIGN S7: "absorber excited by the multisine only" holds; "detuning mostly by the motion records" holds for the low-frequency combinations, while J_eff, cb_sum and d come from the multisine; "rail damping pair weakly separable" is wrong in closed loop
- (?) Keep 140 to 230 Hz at the production rms, with the 230 to 300 Hz verification record
- (?) Friction mismatch in X at 50 to 100 Hz: carried by the motion records only; whether that suffices for the augmentation is untested
- (?) Multisine level is an operating-point member (E2 uses 0.80); session 2 should treat it as an extrapolation axis
- (?) R8 controllers move the 262 Hz closed-loop peak; the plant features (150 Hz, 204 to 224 Hz) do not
- (?) Noise added later: re-read section 10 against the fixed noise level; the Telica-floor margins (at least 12.5 dB) suggest no verdict flips

## 14. Reference-based checks: do references plus multisine cover the FRF differences? (J5, J6, 2026-09-26)
- Why this route: the references are designed, so what they excite can be read from them before any data exist. In a linear loop a reference r acts on the servo error like an injection w = K r at the plant input (e = S_o r - PS f with S_o = I - PS K), so the truth-model difference it produces is E K r; the multisine produces E f
- Gate G-J5 (predicted |E K r| vs simulated truth minus nominal, on E4 and TP1 to TP4, the records whose truth exists without multisine; pass = band rms ratio within a factor 2):

| band | ratio predicted / simulated, X1 / X2 / Y | verdict |
|-|-|-|
| 20 to 50 Hz | 3.9 to 8.9 | fails: over-predicts; rails slide during moves, the standstill friction BLA does not apply |
| 50 to 140 Hz | 0.9 to 1.8 | passes |
| 140 to 230 Hz | X 0.15 to 0.30, Y 0.84 to 0.91 | Y passes; X fails: stick-slip at move ends adds broadband X content no linear prediction holds |
| 230 to 1000 Hz | 0.0 to 0.27 | fails, same mechanism |

- Absorber (Y channel, where the gate passes): the references produce 40 to 90 nm of mismatch in 140 to 230 Hz, the multisine about 33 000 nm: the references do not excite the absorber features; a multisine is needed there
- Friction during motion is nonlinear (sliding, stick-slip at move ends) and is not predictable from the references by this route; it is a velocity-coverage question (both directions, several speed levels to separate Coulomb from viscous as in Garcia's constant-velocity identification, reversals, dwells), not a frequency-band question (?)
- Coverage (J6, fig. 10, Y = 0, K1): a source covers frequency f on input i when its force ASD there is within 40 dB of its own maximum over 20 to 1000 Hz (HEURISTIC; at 20 dB the references cover almost nothing above 10 Hz)
  - references cover up to about 50 Hz on X and about 70 Hz on Y; the multisine 140 to 230 Hz; 50 to 140 Hz by neither
  - the detuning part peaks at 60 to 100 Hz in every entry; E_nom in the X entries at 50 to 80 Hz; Y from F_Y rises from 100 Hz
  - covered share of int |E|^2 df over 20 to 1000 Hz (references + multisine): E_nom 20 to 47 % per entry, E_det 27 to 62 %, detuning part 29 to 43 %
- Reading: for the absorber the 140 to 230 Hz multisine holds the plant features; for the detuned case and for friction in X the largest FRF differences sit in the uncovered 50 to 140 Hz gap; J4 still found all ten combinations separable there because noiseless small signals suffice
- Consequence (?): a multisine band of about 50 to 230 Hz would start where the references stop; cost 3 dB per line at the same rms (180 Hz instead of 90 Hz); not yet checked in the table
- Limits of this section: one Y point (Y = 0); references of the current records, not the planned ones of `DATA-DESIGN.md`; the 40 dB threshold is a heuristic; the gate passes only at 50 to 140 Hz and for Y in band
