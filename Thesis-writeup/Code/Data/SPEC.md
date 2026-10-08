# Specification: thesis data, T_AF truth, with and without noise

What the generator in this folder must produce, reconciled from `Thesis-writeup/Documentation/` (DATA-DESIGN, NOISE-INJECTION, EXCITATION-VALIDATION, EXCITATION-REVIEW, RESULTS-DESIGN) and decisions D-209 to D-219. Written 2026-09-26 from the handoff `tasks/handoffs/2026-09-26-data-generation.md`.

## 1. Precedence applied
- Order (DATA-DESIGN opening lines): user decisions, then session 1 (`EXCITATION-VALIDATION.md`), then `RESULTS-DESIGN.md`, then the slides; the later dated decision wins (D-219 over D-216, D-218 over D-212)
- User decisions of 2026-09-27: one realisation of training and validation instead of the design's three replicate realisations (GENERATION-REPORT U6); the floor of every record is its noisy minus noise-free difference, so the 4 noise-only twins leave the data set (U7)
- User decisions of 2026-09-26 (handoff): one truth T_AF, never detuned; two versions, with and without noise, identical except the noise force; noise on the motor force (superseded 2026-09-28: noise on the encoder, D-225); one K1 at Y = 0; datasheet maximum acceleration 30 / 50 m/s²; own phases and own noise per record
- Each conflict in section 6 names the rule that decided it

## 2. Versions and folders
- With noise: `Thesis-writeup/Data/Coulomb-and-MSD/{Training,Validation,Test}/`, 48 records (DATA-DESIGN 7.9, T_AF block; one realisation of training and validation and no noise-only twins, user decisions 2026-09-27)
- Without noise: `Thesis-writeup/Data/Coulomb-and-MSD-noise-free/{Training,Validation,Test}/`, the same 48 records with v = 0 (encoder noise off; d = 0 under D-218), everything else identical (handoff reading, confirmed by the user 2026-09-26); each record's noise floor is the difference between its two versions (user decision 2026-09-27)
- The noise-free set is the user's addition of 2026-09-26; the design had noise-free twins only for R5's control truths (DATA-DESIGN section 2, 7.7)
- `Thesis-writeup/Data/Detuned-with-Coulomb-and-MSD/` is not written: the detuning is a start value of the baseline model, not a data property (DATA-DESIGN Q2, user 2026-09-26)
- `MANIFEST.csv` per version; each record's file names its twin in the other version (`pair`)

## 3. Common settings (every record unless stated)
| Setting | Value | Source |
|-|-|-|
| Truth | baseline + Coulomb rails (tanh-smoothed, D-224) + payload absorber (T_AF) | DATA-DESIGN section 0 |
| Masses, damping, stiffness, geometry | mb 22.8, mh 10.1, m1 10.2, m2 10.7 kg; Jb 1.0, Jh 0.05 kg m²; cg1 14.5, cg2 20.3, cy 10, cb1 = cb2 = 9 N s/m; kb1 = kb2 = 1987.5 N/m; Lb 0.725, d 0.1 m | `gtd_config.m` (Garcia 2013 set) |
| Coulomb | cc1 16.80, cc2 18.35, ccy 11.60 N | Garcia 2013 via `generate_trajectory_data_coulomb.m`; not a knob |
| Friction law | F_i = cc_i tanh(g v_i) per stage rail, g = 1000 s/m (v0 = 1/g = 1 mm/s; D-226, was 100 in D-224), `gantrySystemExtendedTanh.m`; the law and g are stored in every record (`meta.truth.friction`, `meta.truth.friction_gain`) | D-223 (gain sweep), D-224 (port); replaces the Karnopp stick band V_BRK 1e-3 m/s of D-209, which drove a micrometre stick-slip limit cycle at standstill |
| Absorber mass | ma = 0.50 x mh = 5.05 kg; rigid head 5.05 kg | DATA-DESIGN section 0; D-188 |
| Absorber stiffness | ka = ma (2 pi 150)² = 4.4857e6 N/m (recomputed) | `gtd_config.m`, fa = 150 Hz (the anti-resonance) |
| Absorber damping | zeta 0.03 (`ZETA_A_OVERRIDE`), ca = 2 zeta sqrt(ka ma) = 285.6 N s/m (recomputed) | DATA-DESIGN section 0; D-188 |
| Free-free pole | 150 sqrt(1 + 5.05 / 5.05) = 212.13 Hz (recomputed) | DATA-DESIGN section 0 |
| Absorber offset | L0 = 0.10 m along +Y | `gtd_config.m` |
| Solver | ode4, fixed step 5e-5 s; controller discrete at 20 kHz | `make_coulomb_model.m`, copied driver |
| Record | 12 s at 20 kHz: 0.5 s hold, 10 s active, hold, padded; start at rest at [0, Y_op] | `gtd_config.m`, DATA-DESIGN 7 |
| Controller K1 | ruleOfThumb per stage axis, crossover 100 Hz, rigid nominal plant at Y = 0, tustin; same in every record | DATA-DESIGN 5.11 (user 2026-09-26) |
| K2-g+ (E5 only) | 1.2 x K1 (all gains +20 %) | DATA-DESIGN 5.11, 7.4 |
| Multisine band B_tr | 106 to 297 Hz, 0.5 Hz grid (every 6th DFT line), 383 lines per logical channel (recomputed), own random phases per channel, crest factor best of 30 | D-219, DATA-DESIGN section 0 |
| Multisine level A_prod | 240 N sym, 87 N m anti (= 0.5 x 240 x 0.725), 180 N Y (6 x `gtd_config.m`); anti capped by the 2 mm yaw budget (cap not reached, C4a) | DATA-DESIGN section 0 |
| Limits | X ±0.375 m, Y ±0.400 m, yaw 6 mm, 2 m/s, 30 / 50 m/s² (reference), force peak 2000 / 2000 / 1420 N, rms 916 / 916 / 656 N | datasheet via `gtd_config.m` lim |
| Noise | encoder noise v, one independent generator per stage axis, measured shape mapped above 50 Hz with K1 at the record's Y_op (section 5) | D-225 (amended), NOISE-INJECTION |
| Saved precision | double, every signal | NOISE-INJECTION section 5 |

## 4. Manifest
Levels: a = L x max (30 / 50 m/s²), v = L x 2 m/s. Multisine records carry A_prod on B_tr; "none" records carry no injection. One realisation of training and validation (user decision 2026-09-27, GENERATION-REPORT U6): every record has its own multisine phases, noise and (moves) setpoint draws; the FRF records keep their 4 phase realisations each (part of the robust BLA method). Every seed is a hash of its key (truth, set, record, realisation, twin, signal) and stored in the file.

### 4.1 Training, 18 records (folder Training)
| ID | Class | Y_op | Settings | Multisine |
|-|-|-|-|-|
| TR-S1 to S5 | standstill | -0.30, -0.15, 0, 0.15, 0.30 | hold | A_prod |
| TR-Y1 to Y3 | oscillatory (fade 0.5 s) | 0 | Y 0 ± 0.30 at 0.2, 0.5, 0.75 Hz | A_prod |
| TR-P1 | S-curve moves, log strata | 0 | L 0.25, T3 36 ms, X_sym ±0.28, Y ±0.30 | A_prod |
| TR-P2 | same | 0 | L 0.50, T3 30 ms | A_prod |
| TR-P3 | same | 0 | L 0.75, T3 12 ms | A_prod |
| TR-P4 | same, yaw | 0 | L 0.50, T3 25 ms, X_sym ±0.14, X_anti ±1 mm | A_prod |
| TR-L1 | Lissajous | 0 | Y 0.30 at 0.35 Hz, X 0.28 at 0.85 Hz | A_prod |
| TR-L2 | Lissajous | 0 | Y 0.25 at 0.7 Hz, X 0.08 at 1.5 Hz, X_anti 1 mm at 0.8 Hz | A_prod |
| TR-T1 | ILC-shape cycloid shuttle | 0 | 40 / 80 mm, L 0.75, dirs (Y, X) +,+ , dwell 0.30 s, X ±0.10, Y ±0.30 | none |
| TR-T2 | same | 0 | L 0.45, dirs -,-, dwell 0.45 | none |
| TR-T3 | same | +0.06 | L 0.75, dirs +,-, dwell 0.60 | none |
| TR-T4 | same | -0.06 | L 0.45, dirs -,+, dwell 0.30 | none |

### 4.2 Validation, 6 records (folder Validation)
| ID | Class | Y_op | Settings | Multisine |
|-|-|-|-|-|
| VA-S1 | standstill | +0.10 | hold | A_prod |
| VA-Y1 | oscillatory | +0.05 | Y 0.05 ± 0.25 at 0.5 Hz | A_prod |
| VA-P1 | S-curve moves, log strata | 0 | L 0.50, T3 30 ms, X_sym ±0.28, Y ±0.30, own draw | A_prod |
| VA-L1 | Lissajous | -0.05 | Y 0.20 at 0.7 Hz, X 0.08 at 1.5 Hz | A_prod |
| VA-T1 | ILC-shape | +0.04 | L 0.75, dirs +,+, dwell 0.40 (READING) | none |
| VA-T2 | ILC-shape | -0.04 | L 0.45, dirs +,-, dwell 0.35 (READING) | none |

### 4.3 Test, 12 + 12 FRF = 24 (folder Test)
| ID | Class | Y_op | Settings | Multisine | Controller |
|-|-|-|-|-|-|
| I1 | S-curve moves | 0 | as TR-P2, own draw | A_prod | K1 |
| I2 | standstill | -0.225 | hold | A_prod | K1 |
| I3 | ILC-shape route | -0.01 | 40 / 80 mm, L 0.60, dirs +,+, dwell 0.30, X ±0.10, Y ±0.20 (stops -0.17 to +0.15) | none | K1 |
| I4 | route, alternating X-only / Y-only | -0.01 | as I3, X-only 40 mm then Y-only 80 mm; Y ±0.20 (READING) | none | K1 |
| I5 | route, S-curve | -0.01 | I3 sequence, amax 18 / 30, vmax 1.5, T3 25 ms | none | K1 |
| I6 | route, long strokes | 0 | half stroke to (-0.25, -0.28), then 0.50 / 0.56 m, L 0.60, v cap 1.5, dwell 0.30 (READING) | none | K1 |
| E1 | standstill | -0.39 | hold | A_prod | K1 |
| E2 | ILC-shape route | -0.01 | I3 with Y ±0.39 (stops -0.33 to +0.39) | none | K1 |
| E3 | route, S-curve | -0.01 | I5 at L 1.0 (30 / 50), 1 % acceleration margin flag | none | K1 |
| E4 | route, long strokes | 0 | I6 with v cap 1.95 | none | K1 |
| E5 | ILC-shape route | -0.01 | I3 | none | K2-g+ |
| E6 | ILC-shape route | -0.01 | D-210 kinematics exact (X 1.000562817 m/s, 39.31412412 m/s²; Y 1.499812523 m/s, 50.65906898 m/s²), acceleration assert exception | none | K1 |
| TF-1 to TF-3 x r1 to r4 | standstill | 0.15, 0.225, 0.35 | periodic B_tr multisine, 6 periods of 2 s in the 12 s record (READING) | A_prod | K1 |
| (E1, I1, I4, E5 noise-only twins `_nt1`) | | | dropped from the data set 2026-09-27 (U7); the 4 generated ones are a cross-check in `scripts/gantry/thesis-data-verification/noise-twins/` | | |

## 5. Noise (D-225 and its amendment, NOISE-INJECTION; supersedes D-218's input force)
- Injection: encoder noise v on the branch into the error sum (model block EncoderSum); the controller sees y = q + v; saved `y` is that measured signal, `y_true` the true q, `v_enc` the noise; u_total = u_ff + u_fb is the plant force (no noise force)
- Target: measured Telica standstill error, `g1.npz`, full band: 9.8 / 10.4 / 6.3 nm rms; each axis's auto-spectrum, cross-spectra dropped (one independent generator per axis, supervisors 2026-09-28)
- THEORY: phi_j = Phi_e,jj / |S_jj|^2 above 50 Hz, Phi_e,jj below, S = (I + G K1)^-1 per DFT line of the 12 s record up to 10 kHz (e = -S v; the shape of the measured error is mapped, Jasper)
- G: T_AF linearised at rest at the record's Y_op, including the tanh friction slope cc g per rail, ZOH 20 kHz (`tdg_truth_lin.m` with g; with g omitted it is the friction-off model C11 checked)
- Calibration loop: K1 at each record's own Y_op, for every record including E5
- Phi_e between Welch bins: linear interpolation (HEURISTIC); below 9.77 Hz held at that bin; DC and Nyquist zero
- Corner 50 Hz and shape tolerance 10 % (HEURISTIC, agreed with the user 2026-09-28, to confirm with Jasper)
- Realisation: per axis a frequency-domain shaping filter on unit complex Gaussian noise, own seed per axis derived from the record's noise seed (`tdg_noise_draw.m`)
- Superseded and not used: the D-218 force (T^-1 Phi_e T^-H below 300 Hz with a 300 to 330 Hz roll-off) and the white force 0.312 / 0.342 / 0.099 N
- Shape check: standstill at Y 0, friction on, band ratios above 50 Hz within 10 % of one level (`scripts/gantry/coulomb-tanh-gain/check_encoder_noise.m` B, PASS 2026-09-28)

## 6. Conflicts and stale statements, and how each was resolved
1. Training jerk times: DATA-DESIGN 5.5 lists 13.3, 25, 28.4, 36 ms (marked (?)); tables 7.1 to 7.3 give 36, 30, 12, 25 ms (TR-P1 to P4; VA-P1 and I1 30 ms)
   - Decided by precedence (corrected after REVIEW.md finding 4: both lists were written in the same commit 54d7096, so commit order decides nothing): 12 ms on TR-P3 is a user decision (DATA-DESIGN section 13: "the 12 ms jerk time moved onto the 75 % move record"); 30 ms on TR-P2, VA-P1 and I1 rests on session 1 (precedence 2), whose J9 / J10 evidence (T3 70 / 85 %, Brun gamma 1.27) was computed on 36 / 30 / 12 / 25 ms; C10a shows the generator builds references by the same rule (bitwise for the same seeds); the generated TR-P records are other draws of that rule
   - 30 ms against 5.5's 28.4 ms: confirmed by the user 2026-09-27 (GENERATION-REPORT U9)
   - Flag for the design docs: 5.5's own rule says 12 ms and 30 ms put reference spectral zeros near the 264 Hz closed-loop peak; 5.5 also states the placement is a small effect (40 to 90 nm against 33 000 nm from the multisine); 5.5's T3 list was aligned with 7.1 on 2026-09-27 (U9)
2. Noise calibration: DATA-DESIGN section 0 (N_T "white per 20 kHz sample", "at Y_op = 0", "9.8 / 10.4 / 6.3 nm") and section 10 (sigma 0.31 / 0.34 / 0.10 N, "calibrated at Y_op = 0 for every record", check "reproduces 9.8 / 10.4 / 6.3 nm") against D-218 and NOISE-INJECTION (shaped, below 300 Hz, per record Y_op)
   - Decided by the later dated decision: D-218 (2026-09-26, supersedes the white-force version); the DATA-DESIGN lines are stale pre-D-218 text; RESULTS-DESIGN S4 likewise
   - Which loop: NOISE-INJECTION assumed a controller per Y_op, DATA-DESIGN 5.11 fixes K1; both hold together: G at the record's Y_op (D-218 item 5), K1 (user decision); E5 gets the K1-calibrated force, so its floor moves with K2 as DATA-DESIGN sections 2 and 10 require ("each controller has its own floor")
3. Band: D-216 (140 to 230 Hz), EXCITATION-REVIEW (50 to 230 Hz provisional), RESULTS-DESIGN S7 (140 to 390 Hz (?)), DATA-DESIGN section 12 item 2 (140 to 390 Hz (?)), `Augmentation-coulomb/README.md` band section: all superseded by D-219 (later dated; B11 confirmation) and D-217 (user: no multisine inside the controller bandwidth)
4. Noise-free twins: design section 2 (R5 control truths only) against the user's complete noise-free T_AF set: user decision wins
5. ILC-shape acceleration: `gtd_build_records_telica.m` header "acceleration follows the logs, not the datasheet" and the D-206 amendment "stroke and move duration not adjustable": superseded for training and validation by the user decision of 2026-09-26 (datasheet maximum, 0.45 and 0.75 max, DATA-DESIGN 5.7); the measured move stays exact in E6
6. NOISE-INJECTION check 1 "d = 0 reproduces the existing records bitwise": the existing datasets use per-record controllers and the old band, so no new record can equal them; implemented as C1 (d = 0 in the generator model equals the copied model bitwise on the same inputs) and C3 (each noise-free record equals its noisy twin in every field outside the noise path)
7. Stale values, not used for generation, reported: 7.2 VA-Y1 "v 0.79" is the pure-sine value (realised with the fade 0.907 m/s, C9); 7.3 I5 "j 720 / 1200" and 7.4 E3 "j 1200 / 2000" are amax / T3, realised 1439 / 2399 and 2395 / 3999 because the FIR generator doubles the jerk when the constant-velocity phase is shorter than T3 (|T1 - T2| < T3 in the FIR chain; those moves do reach amax; REVIEW.md finding 6; 7.1 and EXCITATION-VALIDATION 17b word it as "moves too short to reach amax"); both stay inside the trained 3750 / 6250; 7.4 E4 "j about 330 / 920": the first half stroke does not reach 1.95 m/s and realises 383 / 923; 5.4 and 7.4 E6 "j 3430" on Y: the D-210 values give pi a / Ta = 3422 (realised 3422)
8. Section 9 open point "freeze the ASMPT profile set (Jasper, Dragan) and the K2 set (Quinten) before generation": the first-campaign routes are frozen in 7.4b (pair definitions "fixed before generation") and generated as written; if ASMPT's set differs, I3 to I6 and E2 to E6 are regenerated (?)
9. Section 7.1 open consequence of the 0.5 s fade (jerk range label set by fade steps) (?): a labelling question; the 7.1 table adopts the generator's fade for the data, so generation follows it
10. Friction and noise (found in the pilot, discussed with the user 2026-09-26): the design expects each record to carry about 3.4 / 3.7 / 1.1 nm of noise (DATA-DESIGN section 2) and "stuck rails almost none" (NOISE-INJECTION check 3); with the Karnopp truth the noise instead shifts stick-slip breakaway, so sliding multisine records carry 2 to 3 x the target in band and stick-dominated multisine-free records diverge from their noise-free twins by micrometres (2 to 12 % of the servo error); decision (user, option 1 of the discussion): keep the Karnopp truth, generate as designed, judge stick-dominated records against the noise-twin floors (I4, E5); the design statements are to be corrected (?), numbers in GENERATION-REPORT

## 7. Readings (not fixed by the design; confirmed by the user 2026-09-26)
- Noise-free scope and folder names (handoff section 4)
- TF records: 12 s holds 6 periods of the 2 s multisine (1 transient + 5 measured available); DATA-DESIGN 7.6 leaves the period count open; this reading is confirmed by the user
- I4 Y range ±0.20 (I3's, "same lattice as I3")
- I6 / E4 dwell 0.30 s (the measured dwell, D-206; 7.4b gives no value)
- VA-T1 / VA-T2 directions and dwell from their current equivalents VP1 (+,+, 0.40 s) and VP2 (+,-, 0.35 s); Y range ±0.30
- Velocity cap of the ILC-shape and I3 / E2 / E5 records: 2 m/s (not binding; the pure-cycloid peak is reached, as in `ref_planned.m`)
- Noise roll-off above 300 Hz: raised cosine 300 to 330 Hz (user decision, section 5)
- (Dropped with U6: the duplicate noise-free copies of the deterministic ILC records existed only because of realisations 2 and 3)

## 8. Generator changes (implementation list, IMPLEMENTATION.md has the detail)
From DATA-DESIGN section 10 "Implementation to do before generation":
1. One fixed K1 at Y = 0; pre-check plant at Y_op (`gtd_build_plant.m`)
2. Binding post-simulation limit check on the simulated truth, multisine scaled down on failure, realised level recorded (`tdg_post_check.m`, driver)
3. Noise block, own seed, v = 0 no-op: since D-225 the encoder noise (EncoderSum, `v_in`, `v_enc` and `y_true` recorded; `tdg_make_model.m`, `gtd_run_simulation.m`, `gtd_save_record.m`, `tdg_noise_*.m`); was the force block NoiseSum with `d_in` (D-218)
4. Per-record ILC kinematics (record table sets stroke, vmax, amax per record; `gtd_make_reference_telica.m` unchanged for 'telica')
5. X-only / Y-only routes (`gtd_make_reference_telica.m`, class 'route', pattern 'alternate')
6. Explicit setpoint lists: the deterministic lattice shuttle, profiled by the S-curve generator for I5 / E3 (class 'route', profile 'scurve')
7. Log-strata move distances (`gtd_make_reference.m`)
8. Collision-free seed key, all seeds stored (`tdg_seed.m`, `tdg_manifest.m`, `gtd_save_record.m`)
9. and 10. Deferred records only: not implemented
11. Acceleration assert: named E6 exception, 1 % margin flag for E3 (`gtd_validate_ref.m`)
12. Controller gate for K1 and K2-g+ on T_AF over ±0.39 (`scripts/gantry/thesis-data-verification/checks/check_controller_gate.m`)
13. B_tr on the 0.5 Hz grid, line step in the cache key and saved info (`gtd_make_multisine.m`)
Also needed, not in that list:
- Double precision, new fields (`gtd_save_record.m`)
- New record table (`tdg_manifest.m`), config (`tdg_config.m`), one driver for both versions (`generate_thesis_data.m`)
- Model copy with only the recorded loop (runtime 107 s to 6 s per record, C1 bitwise) and a plant-force log (C12)
- Short Simulink cache folder (Windows 260-character build limit on this repo path)
