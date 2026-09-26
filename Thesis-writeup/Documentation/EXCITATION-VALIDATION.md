# Closed-loop excitation validation: which excitation the data need

Task: `tasks/handoffs/2026-09-25-closed-loop-excitation-validation.md`. Consumer: session 2
(`tasks/handoffs/2026-09-25-data-set-design.md`). Truth = baseline + Coulomb rail friction + payload
absorber (ma 0.50, zeta 0.03), always. Sections 1 to 7 were written before any computation; section 8b
lists what changed after it.

One data set. The data are generated once, from the truth under K1; the truth is never detuned. "Nominal" and
"10 % detuned" (D-191 vector) are two versions of the BASELINE MODEL: the detuned one is the parameter start value of
joint estimation, not a property of the data (`DATA-DESIGN.md` Q2). Each baseline model is compared with the same
truth FRF, and the excitation must serve both comparisons on the one data set. Where older sections below say
"detuned data" or "joint-estimation data", they mean this second comparison, not a second data set.

| comparison (same data, same truth, same K1) | 90 % band (T2) |
|-|-|
| nominal baseline model vs Coulomb + MSD | 106 to 297 Hz |
| 10 % detuned baseline model vs Coulomb + MSD | 106 to 290 Hz |

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

Current approach (section 15, agreed 2026-09-26): measure the FRF (BLA) of the Coulomb + MSD truth with K1 at the five training Y points, broadband at production level, robust method; take the differences to the nominal and the detuned baseline per Y and averaged over Y. The training multisine stays at or above the controller bandwidth (user); differences inside the bandwidth are the motion references' job. The earlier "50 to 230 Hz" proposal is withdrawn because it puts the multisine inside the bandwidth.

Done (section 16): the K1 truth FRFs at the five training Y points (B6 to B10) and the band analysis J7: crossover 106 Hz; 90 % band 106 to 297 Hz (nominal) and 106 to 290 Hz (detuned); candidate 106 to 297 Hz on a 0.5 Hz line grid at production rms, confirmed by B11 (features inside the band).

Done (section 17): T3 and friction coverage on the planned references of `DATA-DESIGN.md` 7.1, built without simulation: friction PASS; T3 passes 70 % (nominal) and 85 % (detuned) of the resolved weight below the crossover, the diagonal entries resolved, the X-Y cross entries not.

Done (section 18): T4 for the detuned case, gamma 1.27 on the planned data (limit 5), also at the detuned start; the cross-entry gap does not cost separability.

Noise (section 19): the decision is noiseless; the argument that adding noise does not change it is written, but its numbers come from the superseded white-force calibration and MUST BE RECOMPUTED once the shaped noise spectrum exists.

## 1. What this must establish
- (a) The multisine band the absorber data must excite: where the nominal baseline and the truth differ in closed loop, above the floor
- (b) The frequencies the same data must excite for joint estimation from the 10 % detuned baseline model: per identifiable combination, where a 10 % detuning changes the closed-loop servo error above the floor, and whether two combinations can be told apart
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
| J2c | B6 to B10: K1, five training Y points, M = 10; user: do not let it be killed, so RAM kill off; 17 min, lowest free disk 0.85 GB (OneDrive sync database, not the run); all saved |
| J7 | band analysis (`j7_band.py`): first version took the last |L| = 1 crossing (264 Hz, the absorber re-crossing on Y), fixed to the first crossing (106 Hz); see section 16 |
| J2d | B11: confirmation with the candidate spectrum (106 to 297 Hz, 0.5 Hz grid, A_prod, K1, Y = 0, M = 10); RAM and disk kill off, disk monitored by hand; 187 s, saved; comparison `j8_confirm.py` |
| J9ref | the 18 planned training references (`matlab/ref_planned.m`, generator shape functions, no Simulink): 10 s, saved; gate: ILC and S-curve v, a pass; sweeps and Lissajous exceed the 7.1 peaks (generator fade), S-curve jerk 2x (ETEL), see 17b |
| J9 | T3 and friction coverage (`j9_refcov.py`), seconds; section 17b |
| J10 | T4 Brun collinearity (`j10_brun.py`; references at 20 kHz from `ref_planned(out20)`, scratchpad): 18 records, 63 s; section 18b |

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

## 10. 10 % detuned baseline model vs Coulomb + MSD: joint estimation on the same data
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
- **Answer 2: the multisine band stays 140 to 230 Hz for the detuned baseline model**: the multisine is needed only for J_eff, cb_sum and d (yaw inertia, yaw damping, payload offset), and 140 to 230 Hz supplies 82 to 98 % of their information; no low-frequency lines are needed. One multisine design serves both baseline models on the one data set
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

## 15. FRF measurement approach (agreed with the user, 2026-09-26; supersedes the measurement settings of section 3 for the band decision)
Purpose: measure the closed-loop FRF of the true system, compare it with the nominal and the 10 % detuned baseline, and read from the differences which frequencies the data must excite.

Set-up
- Truth: Coulomb + MSD (T_AF), never MSD alone
- Controller: K1, designed once at Y = 0 and frozen (`DATA-DESIGN.md` 5.11)
- Operating points: the five training standstill points, Y = -0.30, -0.15, 0, +0.15, +0.30 m (`DATA-DESIGN.md` 5.1); X plays no role, neither model nor truth depends on it
- Measurement signal: random-phase periodic multisine, 1 Hz to 1 kHz, at the production level A_prod, injected at the plant input; a measurement signal only, not the training excitation, so it also covers the loop-bandwidth region
- Channels: zippered over the three input channels (one experiment gives all columns, at 1.5 Hz per column; L9 slides 49 to 50); the full 3 x 3 in stage coordinates (inputs F_X1, F_X2, F_Y; outputs X1, X2, Y)
- Averaging: robust method (L13 slides 48 to 58), 1 transient period discarded, 2 measured periods, M random-phase realisations; M = 10 proposed (the current runs have 4) (?)
- Result: the differences E_nom = PS_truth - PS_nominal and E_det = PS_truth - PS_detuned per entry and per Y, and averaged over the five Y points (user choice); only differences above the measurement's own uncertainty count

Why the result is a best linear approximation (BLA)
- Coulomb friction makes the truth nonlinear, so it has no single FRF: the output/input ratio depends on the input
- The expected FRF over random-phase multisine realisations is the definition of the BLA (L13 slide 42); the spread over realisations is the nonlinear distortion, here friction; noiseless periods differ only by transients (about 1e-12 measured), so all spread is friction
- The BLA is the same for every excitation with the same power spectrum (L13 slide 46): it holds for A_prod and this spectrum, which is why the level is fixed at the production level
- The same procedure on a linear system returns its exact FRF with zero spread; the friction-off case B0 confirms the pipeline to 2e-5

Why this basis for the band (lectures 5SMB0, `literature/experiment-design/System-identification/`)
- The identification cost weights the model error by the input spectrum, int |G_o - G|^2 Phi_u dw, so an FRF difference counts only where the data excite it, and input power belongs where accuracy is needed (resonances, bandwidth) (L9 slide 13)
- Closed-loop data carry the plant mainly around the crossover (L11 slide 43), which is where the detuning difference peaks (60 to 100 Hz, section 14)
- Where the FRF uncertainty is too large, add input power at those frequencies (L7 slide 38; L8 slides 50 to 57)

Split between the two excitation sources
- The training multisine is not placed inside the controller bandwidth (user, 2026-09-26): its lower edge is at or above the realised crossover (rails 74 to 108 Hz, Y about 83 Hz with K1, `DATA-DESIGN.md` 5.11); it targets the differences above it (the absorber: dip near 150 Hz, cross-entry notch near 204 to 224 Hz)
- Differences inside the bandwidth (detuning part, friction mismatch in X) must come from the motion references (their jerk content); checked afterwards with the reference-coverage method of section 14 on the planned references

State and next steps (not run)
- `scripts/gantry/excitation-closed-loop/matlab/bla_truth.m` is half-edited and does not run: the case table has the K1 cases B6 to B9 and the controller source, but the simulation helper still takes the old arguments
- Next, on the user's go: finish the K1 change, set M, run the truth at the four missing Y points (and Y = 0 again if M changes) one at a time, then compute the differences and their averages over Y

### 15b. Thresholds, fixed before the runs (user, 2026-09-26)
Notation: E(f) = one entry of the difference PS_truth - PS_model (model = nominal or detuned), after averaging over the five Y points; sigma(f) = the standard deviation of the truth FRF from the robust method.
- T1, resolved difference: |E(f)| > 2.45 sigma(f), the 95 % bound for a complex-valued FRF (L8 slide 47); only resolved differences count in T2 and T3 (the 90 % bound would be 2.15 sigma, derived from the same distribution but not on the slide; 95 % kept)
- T2, multisine band, per comparison (nominal, detuned):
  - weight per frequency: the sum over the 9 entries of |E(f)|^2, the model-error term of the identification cost int |G_o - G|^2 Phi_u dw (L9 slide 13); a weight, not a threshold
  - band: inside the frequencies at or above the controller crossover (user: no training multisine inside the bandwidth), the range that holds at least 90 % of the resolved weight there (HEURISTIC, user-agreed); sensitivity reported at 80 and 95 %
  - it must include the plant features (the dip near 150 Hz, the cross-entry notch near 204 to 224 Hz)
  - line density: at least 4 excited lines per 3 dB width of each feature (Geerardyn, Rolain, Schoukens, IEEE TIM 2013)
  - confirmation: the truth FRF remeasured with the candidate band's own spectrum keeps the features inside the band (the BLA depends on the input spectrum, L13 slide 46)
- T3, references good enough, below the crossover (user, 2026-09-26): the difference must stay resolved under the excitation the data actually deliver
  - sigma without noise: the spread of the truth FRF over the phase realisations is friction's nonlinear distortion, which behaves like noise (L13)
  - rescale sigma to the data's excitation: variance scales as 1 / input power (L7; L11 slide 37), so sigma_data(f) = sigma(f) |U_meas(f)| / |U_ref(f)|, with U_meas the measurement multisine and U_ref the force the planned motion references deliver (the equivalent injection K1 r)
  - criterion: at every resolved difference below the crossover, |E(f)| > 2.45 sigma_data(f), the same 95 % bound as T1 (L8 slide 47)
  - caveat (?): friction's distortion itself changes with the excitation level and spectrum (L13), so the 1 / input power rescaling is an approximation
- T4, deferred (user, 2026-09-26): the claim for now is only that the data excite the detuning differences, judged by T1 to T3; whether separability of the ten combinations is also needed is decided after the results. If it is: collinearity index gamma < 5 with K1, the planned references and the chosen band (noiseless, training-loss weighting, as J4)
  - source: Brun, Reichert, Kuensch, "Practical identifiability analysis of large environmental simulation models", Water Resources Research 37(4), 2001, DOI 10.1029/2000wr900350; gamma = 1/sqrt(smallest eigenvalue of the column-normalised sensitivity matrix)
  - caveat: the critical range 5 to 20 is the authors' experience with environmental models, not a derived threshold
  - why: excitation alone does not show whether two combinations (for example cg1 and cg2) produce distinguishable output changes; gamma does
- Friction: judged by velocity coverage of the references (directions, speed levels, reversals, dwell), not by T1 to T4; its criterion is still open (?)
- With noise added later: the FRF criterion sigma < 0.1 |G| in the band (L8 slide 49) can replace T2 and T3

## 16. Results: the multisine band from the K1 FRFs (J7, 2026-09-26)
Runs B6 to B10 (`matlab/bla_truth.m`): Coulomb + MSD truth, K1, Y = -0.30, -0.15, 0, +0.15, +0.30 m, production level, broadband 1 Hz to 1 kHz, M = 10; not killed (17 min, lowest free disk 0.85 GB; the disk drain during the run was OneDrive's sync database, not the run). Analysis `j7_band.py`, output `outputs/j7_band.json`, figures `fig11_differences_avg.png`, `fig12_band_weight.png`.
- Periodic to 1e-12 in every case; rails stuck 12 to 17 % (X) and about 3.5 % (Y) at every Y
- Crossover with K1 (the bandwidth, first |L_jj| = 1): 72 to 106 Hz over channels, Y points and both loops (nominal baseline, measured truth); highest 106.3 Hz (rail X1 at Y = +0.30) -> f_c = 106 Hz
  - correction made during the analysis: the first version took the last |L| = 1 crossing and got 264 Hz on the truth's Y loop, where the absorber resonance lifts |L| above 1 again; that is not the bandwidth
  - one truth value (X1 at Y = -0.30, 9 Hz) comes from inverting the measured FRF at low frequency; it does not affect f_c

| comparison | 90 % band (T2) | 80 % | 95 % | resolved weight below f_c (references' job) |
|-|-|-|-|-|
| nominal baseline model vs Coulomb + MSD | 106 to 297 Hz | 111 to 276 Hz | 106 to 323 Hz | 16 % |
| 10 % detuned baseline model vs Coulomb + MSD | 106 to 290 Hz | 106 to 267 Hz | 106 to 314 Hz | 21 % |

- Both comparisons give almost the same band, about 106 to 300 Hz; it holds the dip (150 Hz), the cross-entry notch (208 to 213 Hz over Y) and the closed-loop peak (about 262 Hz)
- Why the two bands agree, and why this is expected rather than luck:
  - the lower edge is the K1 crossover (106 Hz), a property of the controller and the plant, the same in both comparisons; the multisine stays above it by design (user)
  - above the crossover the difference to the truth comes from what both baseline models lack, the absorber (dip 150 Hz, cross-entry notch about 210 Hz, closed-loop peak 264 Hz) and friction; the 10 % detuning is small next to it, so the two difference curves nearly coincide there
  - the detuning shows mainly near and below the crossover (60 to 100 Hz, section 14): 21 % of the detuned comparison's resolved differences lie below 106 Hz against 16 % for the nominal one; that part is the references' job (section 17), and T4 (section 18) confirms the references carry kb_sum, cg1, cg2, cy and m_total while the band carries J_eff, cb_sum and d
  - so B_tr = 106 to 297 Hz, which holds both 90 % bands, serves both baseline models on one data set
- The lower edge is set by the crossover, not by the weight (the resolved weight continues below 106 Hz)

Why the broader band is defensible (discussed with the user)
- Every part of it carries a resolved difference training will see: 106 to 140 Hz the detuning and friction differences near the crossover; 140 to 230 Hz the absorber's plant features; 230 to 300 Hz the closed-loop peak, the largest difference in the matrix
- Training fits the closed-loop servo error, which is largest at 230 to 300 Hz; with 140 to 230 Hz only, the model has to extrapolate there
- Pintelon and Schoukens (2012, p. 152): concentrate power where it informs the model; here that is where the model is wrong

Energy per line: a sparser line grid instead of a narrower band (user concern: spreading the energy over many frequencies)
- At the records' 1/12 Hz spacing the band has about 2290 lines against about 1080 today: power per line halves (-3.3 dB) at the same rms
- About 4 lines per 3 dB bandwidth suffice (Geerardyn, Rolain, Schoukens 2013); the narrowest feature (the dip, about 9 Hz at -3 dB) holds about 100 lines at 1/12 Hz
- Candidate: excite every 6th line, a 0.5 Hz grid: about 18 lines in the dip's width, and about 3x today's power per line despite the wider band; the empty lines show the friction distortion directly (odd-multisine gaps, L13)
- Candidate training multisine (?): 106 to 297 Hz, 0.5 Hz grid, production rms (240 N sym, 87 N m anti, 180 N Y)

Confirmation (T2, stated before the result): the truth FRF measured with the candidate spectrum itself (case B11: Y = 0, K1, 106 to 297 Hz on the 0.5 Hz grid, A_prod, M = 10), compared with B10 inside the band
- Pass: the dip, the cross-entry notch and the closed-loop peak stay inside 106 to 297 Hz
- Reported beside it: the in-band change of the FRF against B10, and the stick fractions (friction regime)
- If it fails: the 80 % band (111 to 276 Hz) is the fallback, re-checked the same way

Confirmation result (B11, `j8_confirm.py`): PASS
- Periodic to 4e-14; 127 lines per channel from 106.5 to 295.5 Hz, shared with B10
- Features with the candidate spectrum (B10 broadband in brackets): dip 150.0 Hz (150.0), notch X1 from F_Y 210.0 Hz (208.5), closed-loop peak 264.0 Hz (264.0); closest to a band edge 33 Hz
- The FRF itself moves: median change 2 to 11 % per entry (Y from F_Y 2 %, diagonals X 5 %, cross entries about 10 %), beyond 2.45 sigma on 17 to 80 % of the lines
- Cause: friction regime; with the band-limited input the rails stick less (X 8 to 9 % of the time against 12 to 13 %, Y 2 % against 4 %). This is the BLA's dependence on the input spectrum (L13), not a moved feature
- Consequence: the band holds; the difference sizes of J7 are for the broadband standstill input and change by about 10 % with the training spectrum. In motion the friction regime differs again (J5, section 14); that is part of T3 on the planned references

## 17. Do the planned references cover what lies below the crossover? T3 and friction coverage (J9, plan written before the run, 2026-09-26)
Question: below f_c = 106 Hz the multisine does not excite (section 16); 16 % (nominal) and 21 % (detuned) of the resolved differences lie there. Do the 18 training references of `DATA-DESIGN.md` 7.1 excite them (T3), and do they cover the friction velocity range?

References (no simulation; `matlab/ref_planned.m`, output `outputs/j9_refs.mat`)
- Built with the generator's own shape functions: standstill and sinusoidal paths through `gtd_make_reference`, ILC-shape cycloids through `gtd_make_reference_telica` (per-record level, dwell and directions of 7.1; velocity cap not binding, so the stated peaks 0.76 / 1.38 and 0.59 / 1.07 m/s are reached), S-curve moves with `thirdOrderSetpointETEL` at the 7.1 levels and jerk times
- S-curve distances from the log strata of `DATA-DESIGN.md` 5.6 (1 to 3, 3 to 10, 10 to 30, 30 to 100, 100 to 300 mm, 300 mm to the range), which the generator does not have yet; details open in 5.6, taken here as (HEURISTIC): each axis cycles through the six strata in shuffled order, log-uniform inside a stratum, random sign, reflected at the range bound, the top stratum up to the largest move that fits; 0.1 s hold between moves as the generator; one setpoint draw
- Gate before use: realised peak velocities of the sweeps, Lissajous and ILC-shape records equal the 7.1 values within 0.01 m/s; S-curve peaks at or below the level's v, a and j

T3 (section 15b), how it is evaluated
- Reference as injection: w = K1 r at the plant input, stage coordinates (section 14); logical channels f = P w
- Reference spectrum without leakage: each record starts and ends at rest, so the difference sequence of r has finite support; R(f) = D(f) / (e^(j 2 pi f dt) - 1), D the DFT of the first difference (exact for f > 0; no window)
- Excitation compared as energy per hertz (one-sided energy spectral density, summed over the 18 records, averaged over each 1.5 Hz FRF line cell): measurement ESD_meas,c = A_c^2 T_meas / B, with A = (240 N, 87 N m, 180 N) per logical channel, T_meas = 10 realisations x 2 periods x 2 s = 40 s, B = 999 Hz
- sigma_data^2 = sigma^2 ESD_meas,c / ESD_ref,c per logical column (variance proportional to 1 / input energy, L7, L11 slide 37), then to stage entries as in J7; E and sigma averaged over the five Y points as in J7
- Pass (T3): at each resolved difference below f_c, |E| > 2.45 sigma_data; reported as the share of the resolved weight below f_c that passes, per comparison, per entry and per band (1 to 20, 20 to 50, 50 to 106 Hz)
- Known limit, stated before the result: in motion the friction response is not the standstill BLA (J5 over-predicted 4 to 9x at 20 to 50 Hz); below 50 Hz the verdict is approximate and is to be confirmed on the truth where it matters

Friction coverage, from the reference velocities (pass criteria fixed now, per stage axis X1, X2, Y, over the training set)
- both directions: motion above the stick band (|v| > 1 mm/s, V_BRK of D-209) in each direction
- at least 3 constant-speed levels per direction: segments with |a| < 0.05 m/s^2 and |v| > 1 mm/s lasting at least 20 ms (HEURISTIC: inertial force at most about 1.6 N on the heaviest rail, 32 kg, against 11.6 to 18.4 N Coulomb; corrected before the run from 0.5 m/s^2, which would count the whole 0.2 Hz Y sweep, peak 0.47 m/s^2, as constant speed), levels distinct by 0.1 m/s or more; reversals through a dwell when the stick-band gap lasts 20 ms or more, else direct (HEURISTIC); basis: Garcia's constant-velocity identification separates Coulomb from viscous friction with several speeds (L9 slide 29 staircase idea)
- reversals: velocity sign changes, both through a dwell and direct
- dwell: holds at |v| below the stick band
- also reported: time share per speed bin, to show the spread of speeds the friction model is fitted on

### 17b. Results (J9, 2026-09-26; `j9_refcov.py`, `outputs/j9_refcov.json`, `outputs/J9.log`, fig. 13)
Gate (reference construction against the 7.1 table)
- Passes: ILC shapes (0.757 / 1.382 and 0.586 / 1.070 m/s), S-curve v and a at the level values, sweep TR-Y3 and Lissajous X peaks
- Fails, generator properties, not the construction:
  - the generator's 0.5 s half-cosine fade on sinusoidal paths raises the slow ones' peaks (TR-Y1 0.60 m/s and 3.8 m/s^2 against 0.38 and 0.5; TR-Y2 1.09 against 0.94; TR-L1 Y 0.91 against 0.66) and leaves acceleration steps of 1.7 to 3.0 m/s^2 at the fade edges
  - `thirdOrderSetpointETEL` doubles the jerk on moves that do not reach amax (its own header): TR-P3 3750 / 6246 m/s^3 against the 1875 / 3125 of 7.1, likewise P1, P2, P4
- Consequence for this check: none; sweeps and Lissajous deliver 0 to 0.2 % of the reference energy above 20 Hz (below); the 7.1 peak values and the trained jerk maximum are corrected in `DATA-DESIGN.md` 7.1 (user, 2026-09-26)

T3: share of the resolved weight below f_c = 106 Hz that the planned references resolve (|E| > 2.45 sigma_data)

| band | share of the resolved weight below f_c (nominal / detuned) | passes, nominal | passes, detuned |
|-|-|-|-|
| 1 to 20 Hz | 0.9 / 0.6 % | 100 % | 100 % |
| 20 to 50 Hz | 13.6 / 10.5 % | 83 % | 85 % |
| 50 to 106 Hz | 85.5 / 88.9 % | 68 % | 85 % |
| total | | 70 % | 85 % |

- Per entry (nominal / detuned): diagonals X1, X2 70 to 78 %, Y from F_Y 77 / 97 %; X1 from F_X2 and X2 from F_X1 32 to 49 %; X from F_Y and Y from F_X 0 to 2 %
- Where the energy comes from, 20 to 106 Hz: S-curve moves 89 to 100 % per channel (TR-P3 alone about 90 % at 30 to 106 Hz), ILC shapes 6 to 11 %; sweeps, Lissajous and standstill about 0. Relative to the FRF measurement the references deliver 10 to 30x its energy per hertz at 20 to 50 Hz (sym, Y) but 0.4 to 2 % at 50 to 106 Hz
- Yaw is the weak channel: 99 % of its reference energy comes from TR-P4 (the only S-curve record with X_anti, 1 mm), whose jerk time of 25 ms puts spectral zeros at 40 and 80 Hz; that gives the sigma_data peaks in the X-input columns of fig. 13 and most of the shortfall in the X1 / X2 split
  - the 25 ms follows the jerk-time rule of `DATA-DESIGN.md` 5.5, T3 = (l + 0.5) / f_ring with l = 6: 264 x 0.025 = 6.6, so the 264 Hz closed-loop ringing lies between the zeros at 240 and 280 Hz, as designed
  - the jerk stage is a moving average of length T3, so the reference spectrum has zeros at every multiple of 1/T3 = 40 Hz (Biagiotti and Melchiorri eq. 34); the rule places one pair of them around 264 Hz and cannot avoid the ones at 40 and 80 Hz, below the crossover, which it does not consider; measured: TR-P4's spectrum falls by a factor 30 to 50 at 39 to 41 and 79 to 81 Hz against the neighbouring frequencies
  - the sym and Y channels do not show these zeros because records with other jerk times (12, 30, 36 ms) fill each other's zeros; yaw has TR-P4 as its only source

Reading, against the criterion
- The references resolve the direct (diagonal) differences below the crossover, and those carry most of the weight: 85 % of the detuned case and 70 % of the nominal case pass
- They do not resolve the X-Y cross-coupling differences below the crossover (X from F_Y, Y from F_X), and they resolve the X1 / X2 split only partly, because only one record excites yaw
- T3 therefore passes for the diagonal entries and fails for the cross entries; whether the cross-entry gap matters for the ten parameter combinations is exactly what T4 (Brun collinearity) decides; T4 stays deferred until the user decides
- 50 to 106 Hz, where 86 to 89 % of the weight lies, is inside the range where J5's linear prediction passed its gate (50 to 140 Hz); the 20 to 50 Hz part (11 to 14 % of the weight) is approximate, so a truth confirmation there has low value

Friction coverage: PASS on X1, X2 and Y
- Both directions: X about 24 s each way, Y about 39 s each way (18 records, 10 s active each)
- Constant-speed levels, each direction: X 0.5, 1.0, 1.5 m/s; Y 0.38, 0.5, 0.66, 1.0, 1.5 m/s (S-curve cruise, sweep and Lissajous velocity peaks)
- Reversals: X 75 to 77 through a dwell, 47 direct; Y 69 through a dwell, 50 direct
- Dwell: about 195 holds of 20 ms or more per axis
- Speeds spread from 1 mm/s to 1.5 m/s, none above 1.5 m/s (the trained maximum by design)
- Limit: one setpoint draw of the S-curve records

## 18. T4 for the detuned case: can the planned data separate the ten combinations? (J10, plan written before the run, 2026-09-26)
Why now: T3 (17b) leaves the X-Y cross entries and part of the X1 / X2 split unresolved below the crossover; T4 decides whether that matters for joint estimation (user: run Brun for the detuned case).
- Data: the 18 training references of 17 (`matlab/ref_planned.m`, 20 kHz) with the chosen multisine on every record that carries one (106 to 297 Hz, 0.5 Hz grid, A_prod per logical channel, random phases, own seed per record; crest-factor selection skipped, HEURISTIC: it does not change the spectrum); the ILC-shape records without multisine; controller K1; noiseless
- Sensitivities, as J4: the baseline closed loop (vendored replica) at theta_0 and with each combination +10 %; S_i = e(theta_0 + 10 % on i) - e(theta_0), servo error in metres on the three stage channels, unweighted (the training loss's own weighting); Gram matrix summed over the records (Parseval)
- Index: Brun, Reichert, Kuensch 2001 Eq. 13, gamma = 1 / sqrt(smallest eigenvalue of the column-normalised Gram matrix); also the independent share r_i = 1 / sqrt((Gn^-1)_ii) per combination and the most collinear pair
- Pass: gamma < 5 (15b; the lower end of Brun's critical range 5 to 20, the authors' experience, not a derived threshold)
- Reported beside it: gamma for the references alone (no multisine) and for the multisine alone (standstill records), to show which source separates which combination; and gamma at the detuned start point (sensitivities around theta_0 x D-191 vector), since joint estimation starts there

### 18b. Results (J10, 2026-09-26; `j10_brun.py`, `outputs/j10_brun.json`, `outputs/J10.log`; 18 records, 63 s)

| data | gamma | verdict (< 5) | most collinear pair | lowest r_i |
|-|-|-|-|-|
| planned data (references + multisine), at theta_0 | 1.27 | PASS | J_eff / d, -0.28 | J_eff 0.93 |
| planned data, at the detuned start | 1.27 | PASS | J_eff / d, -0.28 | J_eff 0.92 |
| references alone | 1.31 | PASS | kb_sum / J_eff, -0.41 | kb_sum, J_eff 0.91 |
| multisine alone (standstill records) | 4.55 | PASS, near the limit | kb_sum / J_eff, -0.94 | cb_sum 0.29 |

- Where each combination's information comes from (planned data, diagonal Gram): below 106 Hz (references) kb_sum 94 %, cg1, cg2, cy 99 %, m_total 87 %; in 106 to 297 Hz (multisine) J_eff 100 %, cb_sum 92 %, d 91 %, mh 69 %, m_diff 60 %
- The two sources complement each other: the multisine alone confounds kb_sum, cb_sum, J_eff and the guide dampers (r_i 0.29 to 0.44), the references alone separate all ten, and together every r_i is 0.92 or more

Reading, against the criterion
- T4 passes with a wide margin: gamma 1.27 against 5, the same at the detuned start; the ten combinations are separable on the planned data
- So the T3 gap (X-Y cross entries and part of the X1 / X2 split unresolved below the crossover) does not cost separability: what the cross entries carry is also carried by entries the data resolve
- Limits: noiseless and linear-in-parameter (secant sensitivities of +10 %), baseline model only; gamma judges the shape of the ten effects, not their size against friction distortion (that is T3); whether the augmentation can absorb baseline effects is a separate question (negation, projection)

## 19. Why the noiseless decision holds once noise is added (argument; numbers to recompute, 2026-09-26)
The excitation (B_tr, references) was decided on the noiseless case (user). The data are generated with and without noise; the argument below is why adding noise should not change the decision.
- The noise is a force at the plant input (`NOISE-INJECTION.md`), the same point where the multisine enters and where a reference acts (as K1 r), so the comparison is a plain force-spectrum ratio
- In the band: the multisine puts about 170 to 300 N^2/Hz per logical channel into 106 to 297 Hz; the noise is 70 to 80 dB below that
- Below the crossover: the references deliver about 40 dB more force energy per hertz than the noise at 50 to 106 Hz (training set, 18 records of 12 s)
- Against the friction distortion: the noise-driven FRF error in the band is about 100x smaller than the friction distortion sigma used in T1 to T3, so the thresholds, and hence T1 to T3, are set by friction, not by noise
- T4: Brun's gamma uses column-normalised sensitivities; it measures how alike the ten effects are and does not depend on the noise level; noise only widens the parameter standard errors
- So noise changes the results (floors, parameter accuracy), not which excitation is needed; the with-noise and noise-free data serve the results (R5, floors), not this decision

MUST BE RECOMPUTED before the thesis cites these numbers:
- The dB figures above are estimates from the SUPERSEDED white-force calibration (sigma 0.312 / 0.342 / 0.099 N per 20 kHz sample, about 0.05 / 0.06 / 0.02 N rms below 300 Hz; `NOISE-INJECTION.md` section 3, "superseded")
- The agreed calibration is shaped to the measured error spectrum below 300 Hz and puts about 1/9 of that power there (3.4 against about 9.8 nm rms), but it can concentrate it at some frequencies; its force spectrum Phi_d is not computed yet (`NOISE-INJECTION.md` status)
- To recompute once Phi_d exists: (1) Phi_d against the multisine PSD in 106 to 297 Hz; (2) Phi_d x 216 s against the references' energy spectral density below 106 Hz (`j9_refcov.py`); (3) the noise-driven FRF variance against the friction distortion variance of B6 to B10 per line; the argument holds if (1) and (2) stay above about 20 dB and (3) stays below 1
- Below about 20 Hz the argument is weakest (the noise model holds its lowest band value there, and the shaped noise is largest at low frequency); that range carries under 1 % of the resolved weight below the crossover (17b)
