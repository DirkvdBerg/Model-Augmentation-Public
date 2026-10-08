# Generation report: thesis data, T_AF truth, with and without noise

The current data are the regeneration of 2026-09-28 (section 0). Sections 1 to 8 report the first generation (session 2026-09-26, handoff `tasks/handoffs/2026-09-26-data-generation.md`; audit and follow-up of 2026-09-27 in section 8): Karnopp friction and a force noise, superseded, its folders kept untouched until the user decides on them.

## 0. The current data: regeneration of 2026-09-28 (D-230, D-231)
Status: complete. 48 records in both versions, same blocks, record names and MANIFEST pairing as section 1; nothing failed a check.

What changed against the first generation
- Friction in the truth: F_i = cc_i tanh(g v_i), g = 1000 s/m (D-224, D-226), instead of the Karnopp stick; the stick drove a micrometre stick-slip limit cycle the machine does not show
- Noise on the encoder (D-225 and its amendment, `NOISE-INJECTION.md`): the controller sees y = q + v; one generator per axis; the measured standstill spectrum's shape mapped through the loop above 50 Hz, full band to 10 kHz; replaces the input force d of D-218
- Record fields: `y` = q + v (what the controller saw), `y_true` = q, `v_enc` = v, `u_fb` = K1 (r - y); `d_in` removed; `meta.truth` holds the friction law and gain
- Folders: `Thesis-writeup/Data/Coulomb-tanh-and-MSD/` (noisy) and `Coulomb-tanh-and-MSD-noise-free/`; the first generation stays in `Coulomb-and-MSD[-noise-free]`
- Simulator: 20 records with Simulink, the other 28 with the replica of the recorded loop `tdg_replica.m` (D-231: about 1 GB of RAM instead of 2.85 GB); `meta.generator.simulator` says which (absent = Simulink)

| Check (IMPLEMENTATION.md) | Result |
|-|-|
| C3, C4b, C8, C14 (`check_records`) | PASS on all 48, both versions (C8 now checks the tanh law and gain; C14 regenerates `v_enc` bitwise from the per-axis seeds) |
| C16 replica against Simulink | PASS bitwise in every field outside `meta.generator`, TR-T1, TR-S3, TR-P3, both versions |
| C17 tanh chart against the replica | PASS, max difference 0, both versions |
| C18 noise shape above 50 Hz | PASS, band ratios 0.97 to 1.03 |
| C19 encoder noise in the model against the replica, u_plant = u_total | PASS, max difference 0, 1.3e-8 N |
| C20 closed-loop oracle against the pilot records | PASS |
| D-221 gate: the pipeline's K1 reproduces `u_fb` from r - y (2026-09-29) | PASS, 1e-11 to 5e-10 relative, E5 (K2-g+) included |
| EXCITATION-VALIDATION section 19 on these data (2026-09-29) | holds everywhere |

- Sizes: 2.3 GB noisy, 2.0 GB noise-free (96 files)
- Run: the generator's disk rule stopped the campaign after 20 of 48 records when the page file filled C:; after the user cleared the conda cache, the other 28 ran with the replica in 34 min (20:23 to 20:57)
- Training reads these data through the D-232 loader rule: y anti-alias filtered in the noisy mode, 5 ms dropped at both record ends in both modes (`NOISE-INJECTION.md` section 8)

## 1. What was generated
Status: complete. All 48 records exist in both versions (one realisation, U6; floors from the noise-free twins, U7); the last 10 FRF references were generated on 2026-09-27 after the audit (section 8). Nothing generated failed a check.

| Block (DATA-DESIGN 7.9) | Designed (noisy / noise-free) | On disk (noisy / noise-free) |
|-|-|-|
| Training | 18 / 18 | 18 / 18 |
| Validation | 6 / 6 | 6 / 6 |
| Test interpolation I1 to I6 | 6 / 6 | 6 / 6 |
| Test extrapolation E1 to E6 | 6 / 6 | 6 / 6 |
| FRF references TF-1 to TF-3 x 4 | 12 / 12 | 12 / 12 |
| Total | 48 / 48 | 48 / 48 |

- The 4 noise-only twins generated earlier are not part of the data set any more (U7); they are kept in `scripts/gantry/thesis-data-verification/noise-twins/`

- Folders: `Thesis-writeup/Data/Coulomb-and-MSD/` (noisy) and `Thesis-writeup/Data/Coulomb-and-MSD-noise-free/`, each with `Training`, `Validation`, `Test` and a `MANIFEST.csv` that pairs every record with its twin in the other version
- Every file: 12 s at 20 kHz, every signal in double, the copied schema plus `d_in`, `seed_ms`, `seed_sp`, `noise_seed`, `version`, `pair` and `meta` (settings, truth and controller parameters, band, realised multisine level, post-check, noise model, code hashes)
- Not written: `Thesis-writeup/Data/Detuned-with-Coulomb-and-MSD/` (empty, left as found; the detuning is a model start value, U1)
- The last 10 FRF records were generated on 2026-09-27 with `generate_thesis_data('frf')`, after the reproduction test C16 (section 8)

## 2. Check results (all on the final code; criteria in each check's header, stated before its first run)
| Check | Result |
|-|-|
| C1 generator model with d = 0 equals the original model, bitwise | PASS, max difference 0 (run time 107 s -> 5.9 s per 12 s run) |
| C2 noise level: friction off, standstill, K1, at all 17 Y_op of the manifest, band 19.5 to 300 Hz | PASS, every axis within 1.4 % of 3.370 / 3.696 / 1.086 nm; figure `scripts/gantry/thesis-data-verification/checks/fig_noise_calibration.png` (level and shape match below 300 Hz) |
| C3 noise-free twin identical outside the noise path | PASS on every saved record (38 after the move; 42 before it, the noise-only twins then included and sharing r and f with their records) |
| C4a / C4b multisine: 383 lines on 106 : 0.5 : 297 Hz, off-grid energy 1e-31, rms 240 / 87 / 180 (constructed and delivered) | PASS |
| C5 post-simulation limits on the simulated truth | PASS on every record at full level (scale 1.00); peak force at most 0.80 and rms force at most 0.43 of the limits; E4 kept its 1.95 m/s cap |
| C6 controller gate, K1 and K2-g+ over Y = ±0.39 on T_AF | PASS (stable); Y-loop phase margin 28.5 deg (K1) and 25.0 deg (K2-g+), flagged (?) |
| C7 seeds: 48 records, 144 distinct seeds (multisine, setpoint and noise per record) | PASS (after U7, `logs/after_move.log`; earlier: 100 records and 292 seeds before U6, 52 records and 148 seeds between U6 and U7) |
| C8 truth parameters in every file (ma 5.05 kg, zeta 0.03, ka, ca, pole 212.13 Hz, cc, V_BRK 1e-3) | PASS |
| C9 references within limits, realised kinematics | PASS; E3 needed no margin (29.94 / 49.98 m/s^2), E6 realises 39.31 / 50.66 m/s^2 by its named exception |
| C10a / C10b reference rules equal the excitation evidence's; route class equals telica class | PASS, bitwise |
| C11 truth linearisation equals the ODE Jacobian | PASS, 1e-16 |
| C12 plant force = u_total + d, every record | PASS, at most 2e-8 N |
| C13 noise draw covariance and spread | PASS |
| C14 schema, double precision, lengths, stored noise seed regenerates d bitwise | PASS on every saved record (38 after the move) |
| C15 MATLAB with and without its JVM give the same simulation checksums (sums of abs(q), abs(delta_a), abs(u_total) of TR-P3_r1, 17 digits) | PASS, the three checksums are equal; equal checksums, not a bitwise comparison of every array (audit A5) |

Rechecks of handoff section 5, with values
- Noise: Phi_d recomputed from `g1.npz` (target 3.370 / 3.696 / 1.086 nm, 11.7 / 12.6 / 2.9 % of the measured variance); check 2 passes (C2); loop: T_AF at the record's Y_op with K1 (M2); the superseded white-force numbers are not used (M3); roll-off 300 to 330 Hz (U3); noise force per sample 0.10 to 0.16 / 0.11 to 0.19 / 0.02 N (X1 / X2 / Y) depending on Y_op; 35 % of the X force variance lies in the 300 to 330 Hz roll-off (70 % with the 400 Hz roll-off first tried)
- Noise-free twins: identical outside the noise path, record by record (C3, C12)
- Multisine: 106 to 297 Hz on the 0.5 Hz grid (D-219 over D-216), 383 lines per channel, A_prod per channel, checked on the generated signal (C4a, C4b); 53 to 68 dB above the noise force inside B_tr at every Y_op (EXCITATION-VALIDATION section 19 item 1)
- Levels and limits: levels recomputed from 30 / 50 m/s^2 and 2 m/s (realised peaks printed in `scripts/gantry/thesis-data-verification/logs/checks_final/attempt1.log`, stored per file in `meta.ref_peaks`); binding post-check on the truth (C5)
- Controller gate: C6
- Absorber and friction: C8 (and C2 end to end)
- Seeds: C7

## 3. Findings to take back to the design documents (?)
- F1 friction and noise (discussed with the user, U4): the noise does not add "about 3.4 / 3.7 / 1.1 nm" once friction is on
  - multisine records (rails slide): the noise adds 4 to 10 nm rms (median 6.7 / 7.6 / 2.6 nm, all frequencies), 2 to 3 x the target
  - multisine-free records (rails stuck about 75 % of the time): noisy and noise-free twins differ by 0.4 to 3.2 um rms (median 1.2 / 1.0 / 0.9 um), because the noise shifts stick-slip breakaway (in the pilot the twins' commanded forces differed by up to 42 N)
  - floors (noisy minus noise-free, the definition since U7): E1 8.0 / 7.4 / 4.5 nm, I1 5.9 / 6.9 / 1.7 nm (0.03 to 0.1 % of their servo error); I4 2.39 / 2.10 / 2.51 um, E5 1.61 / 1.62 / 0.87 um (3 to 12 % of their servo error); the earlier noise-only twins gave E1 9.7 / 8.9 / 5.2 nm, I1 7.3 / 8.2 / 3.4 nm, I4 2.08 / 1.95 / 1.90 um, E5 1.62 / 1.63 / 1.54 um
  - to correct: DATA-DESIGN section 2 ("each record carries about that") and NOISE-INJECTION check 3 ("stuck rails show almost none"); the R6 and R7 criteria on the multisine-free routes should be read against the I4 and E5 floors
- F2 the calibrated noise force is concentrated: on Y 51 % of its variance lies at 145 to 155 Hz (the calibration inverts the absorber's anti-resonance notch); on X it rises with frequency to the roll-off (peak at 302.8 Hz); a property of calibrating through the model's own loop (D-218), stated, not changed
- F3 stale design text: DATA-DESIGN section 0 (N_T "white per 20 kHz sample", "9.8 / 10.4 / 6.3 nm") and section 10 (white-force sigma 0.31 / 0.34 / 0.10 N, calibration at Y_op = 0), RESULTS-DESIGN S4 and S7; superseded by D-218 and D-219
- F4 DATA-DESIGN 5.5 lists training jerk times 28.4 and 13.3 ms against 7.1's 30 and 12 ms; the data use 7.1 (M1); 30 ms confirmed by the user (U9)
- F5 stated values that differ from the realised references: VA-Y1 v 0.79 is the pure-sine value (0.907 realised with the fade); I5 and E3 jerks are 2 x amax / T3 (1439 / 2399 and 2395 / 3999; inside the trained 3750 / 6250); the doubling condition is a constant-velocity phase shorter than T3, not "moves too short to reach amax" (DATA-DESIGN 7.1, EXCITATION-VALIDATION 17b); E4 first half stroke 383 / 923; E6 Y jerk 3422, not 3430
- F6 controller: Y-loop phase margin 28.5 deg with K1 and 25.0 deg with K2-g+, below the usual 30 deg; the design names no margin number (?)
- F7 C10a proves the reference rules, not the records: the excitation evidence (J9, J10) was computed on ref_planned's draws of the S-curve records, the data use their own draws of the same rule; rerunning J9 / J10 on the data's draws is optional (?)
- F8 EXCITATION-VALIDATION section 19 items (2) and (3): recomputed after the audit, both claims hold (section 8, A3); EXCITATION-VALIDATION section 19 and RESULTS-DESIGN S7 updated with the numbers (user, 2026-09-27)
- F9 machine: on this PC the page file grows to about 14 GB under memory pressure and takes back freed disk; MATLAB with Simulink needs 2 to 3.5 GB of RAM, which the other sessions left only intermittently (section 5)

## 4. Decisions taken in this session, with reasons
Decided by the user (2026-09-26)
- U1. Data folders `Coulomb-and-MSD/` (with noise) and `Coulomb-and-MSD-noise-free/` (new); noise-free = every record except the 4 noise-only twins (since U7 there are no noise-only twins in the data set). Reason: handoff reading, confirmed
- U2. TF records are ordinary 12 s standstill records: the 0.5 Hz line grid makes the multisine periodic in 2 s, so each holds 6 periods (1 transient + 5 measured available); I4 Y range ±0.20 (I3's); I6 / E4 dwell 0.30 s (the measured ASMPT dwell); VA-T1 / VA-T2 directions and dwell from their current equivalents VP1 / VP2; identical noise-free copies of the deterministic ILC realisations are written. Reason: readings of gaps in the design, confirmed
- U3. Noise roll-off above 300 Hz: raised cosine to zero at 330 Hz. Reason: the calibrated force rises with frequency (mass line), so a wide roll-off puts most of the X force above 300 Hz (70 % of its variance at 300 to 400 Hz: X force std 0.18 N against 0.12 N at 330 Hz), while the design wants the noise to match the measured error below 300 Hz and not above; the error below 300 Hz is identical in both
- U4. Keep the Karnopp friction truth and generate as designed, although the noise shifts stick-slip breakaway (finding F1). Reason: the thesis claims concern the absorber and parameter interpretability, not friction realism; T_AF is declared a synthetic benchmark; the band, the D-209 stick threshold and the excitation evidence were all computed on this truth; the noise twins of I4 and E5 measure the resulting floors
- U5. Disk: the user freed space (14 GB free after); generation in stages with the 2 GB reserve
- U6. One realisation of training and validation instead of three (user, 2026-09-27). Reason (user): the three replicate realisations add complexity the thesis does not need; the spread in results comes from the model starts. Consequences: realisations 2 and 3 are not generated and not in the record table (`tdg_manifest.m` then held 52 noisy + 48 noise-free records; 48 records in both versions since U7); the realisation-1 records keep their names (`_r1`) and seeds, so the files generated before the change stay valid (C7 and C3 / C4b / C8 / C14 rerun on the new table, PASS); the code hashes stored in those files (`meta.generator.code_sha256`) refer to the code before this change (`tdg_manifest.m` and `generate_thesis_data.m` were edited; the records' data do not depend on the edited lines); the design documents were updated on the user's instruction, only where they described the replicates (DATA-DESIGN.md sections 2, 6, 7.1, 7.2, 7.9, 13; RESULTS-DESIGN.md S1, S5, S6, the R6 figure note, the run count, B4), which the original handoff had put out of scope; the FRF records keep their 4 phase realisations each, because those belong to the robust BLA method of R3, not to replicates
- U7. The noise floor of every record is the difference between its noisy and its noise-free version; the 4 noise-only twins (E1, I1, I4, E5 `_nt1`) leave the data set (user, 2026-09-27). Reason: with a noise-free twin of every record the floor is available everywhere without extra files, and the two data folders match one to one (38 / 38). The 4 generated twins are kept as a cross-check in `scripts/gantry/thesis-data-verification/noise-twins/` (a twin difference is about sqrt(2) x the noisy minus noise-free difference, as two independent draws should be). DATA-DESIGN.md (sections 2, 5.1, 5.5, 5.11, 7.5, 7.9, 10, 13) and RESULTS-DESIGN.md (S4, S6, R6 figure, R7, D8) were updated on the user's instruction. `tdg_manifest.m` holds 48 records
- U8. The verification material (checks with results and figure, run logs, the watchdog and `.sh` launchers, the review, the untouched source copies) was moved to `scripts/gantry/thesis-data-verification/` (user, 2026-09-27), so `Thesis-writeup/Code/Data/` holds only the generator and its documentation (`REFERENCE.md` points to the moved material). Paths in the code were adjusted; the run logs of future generation go to `scripts/gantry/thesis-data-verification/logs/`; a check run from the new location passed (section 2)
- U9. Jerk time 30 ms, not 28.4 ms, for TR-P2, VA-P1 and I1 (user, 2026-09-27, after audit finding A8). Reason: DATA-DESIGN 5.5's rule T3 = (l + 0.5) / f_ring gives 28.4 ms, because 30 ms puts a spectral zero of the reference at the 264 Hz closed-loop peak (264 x 0.030 = 7.92); but 5.5 rates the effect small (the move supplies 40 to 90 nm of absorber-band excitation against about 33 000 nm from the multisine in the same records); 30 ms matches the section 7 tables and the J9 / J10 excitation evidence, and needs no regeneration; DATA-DESIGN 5.5's list was aligned to the tables

Decided by me, within the design (each can be reverted; the reason is the one that decided it)
- M1. Jerk times 36 / 30 / 12 / 25 ms for TR-P1 to P4 (DATA-DESIGN 7.1), not 5.5's 36 / 28.4 / 13.3 / 25 ms. Reason: 12 ms on TR-P3 is a user decision (DATA-DESIGN section 13), and session 1's excitation evidence (J9, J10; precedence 2) was computed on 36 / 30 / 12 / 25 ms; the generator builds its references by the same rules (C10a). A first reason given here (commit order) was wrong and is withdrawn (REVIEW.md finding 4). 5.5 itself rates the effect of 30 against 28.4 ms small (40 to 90 nm against 33 000 nm); 30 ms confirmed by the user 2026-09-27 (U9)
- M2. Noise calibration loop: the T_AF linearisation at each record's own Y_op with K1, for every record including E5 (K2-g+). Reason: D-218 calibrates per Y_op and DATA-DESIGN 5.11 fixes one K1; the noise is a force of the machine, not of the controller, so under K2 the same force gives a different error floor, which is what DATA-DESIGN sections 2 and 10 expect ("each controller has its own floor")
- M3. The white-force numbers of DATA-DESIGN sections 0 and 10 (0.31 / 0.34 / 0.10 N, "reproduces 9.8 / 10.4 / 6.3 nm") are not used. Reason: superseded by D-218 (later dated), and the handoff says so
- M4. Noise realisation by a frequency-domain spectral factor (per DFT line of the 12 s record: D_k = sqrt(N fs / 2) L_k w_k, L_k L_k^H = Phi_d) instead of a Yule-Walker VAR shaping filter. Reason: it is a shaping filter with unit white input as NOISE-INJECTION asks, reproduces the target spectrum on the record's own grid without a fit, while a VAR fitted to a spectrum that is exactly zero above 330 Hz of a 10 kHz band is near-singular; check C13 shows the draws have the target covariance
- M5. Measured spectrum between its 9.77 Hz Welch bins: linear interpolation of the 3 x 3 matrix entries. Reason: a convex combination of positive definite matrices stays positive definite, so Phi_d stays a valid spectral matrix
- M6. Noise below 19.53 Hz: Phi_d held at its 19.53 Hz value (NOISE-INJECTION's own rule), DC bin zero (a zero-mean force over the record)
- M7. Model copy with only the recorded "Extended ODE" loop. Reason: the three other plant loops of the copied model share no signal with it, so they cannot change the data (C1: bitwise identical), and removing them cuts a 12 s run from 107 s to 6 s and the Simscape Multibody dependency
- M8. One simulation per record version. Reason: the copied generator also ran every record without its multisine, only to print a diagnostic; no saved field used it
- M9. Log-strata move distances built with exactly the rule of the excitation evidence (`ref_planned.m`: shuffled strata per axis, log-uniform, random sign, reflection). Reason: DATA-DESIGN 5.6 leaves the details open, and the design's excitation verdict (T3, Brun gamma 1.27) was computed on this rule
- M10. Post-simulation check on the plant force u_total + d (not u_total alone), and the noisy run decides the multisine scale for both versions. Reason: conservative (the plant receives u_total + d), and one scale keeps the two versions identical except the noise
- M11. Controller gate: PASS on stability (the design's requirement); margins reported and flagged below GM 6 dB / PM 30 deg (Skogestad and Postlethwaite 2005, section 2.4.3); the Y-loop phase margin is 28.5 deg with K1 and 25.0 deg with K2-g+, flagged, generation not stopped. Reason: the design names no margin number, and both loops are stable over Y = ±0.39
- M12. Check criteria revised after a first failure, both documented in the check headers: C12 tolerance 1e-9 N to 1e-6 N (the 1.5e-8 N found is rounding in the controller states, a plumbing error would show 1 to 40 N); C13 from "one draw within 5 %" to a 40-draw test against the predicted spread (the first criterion assumed a flat spectrum; the force spectrum is concentrated). The C1 and C2 criteria were never changed
- M13. Deleted and regenerated my own pilot files: the 14 noisy files made with the 300 to 400 Hz roll-off (U3), and the 13 noise-free files made by an earlier revision of the driver (data identical, stored code hashes different), so every file of the final set comes from the final code
- M14. Resources: the emergency RAM kill stays at 1.0 GB (user); the disk emergency kill is 0.5 GB and the driver stops a stage below 2 GB free (handoff); a job the RAM kill stops is retried after 2 min (no RAM launch gate); a job waits while any other MATLAB runs; writes are atomic (`*_part.mat` then rename), so a killed job never leaves a truncated record
- M15. MATLAB without its JVM when check C15 shows bitwise-identical simulations (about 0.5 GB less RAM on a PC whose other programs hold most of its memory); Java calls replaced by `certutil` (SHA-256) and PowerShell (free disk); the noise figure is drawn by Python
- M16. No per-record figures, multisine disk cache off. Reason: disk (the checks produce the figures that prove something)
- M17. Saved format MAT v7 (compressed), every signal double. Reason: the training loaders use `scipy.io.loadmat`, which cannot read v7.3; double per NOISE-INJECTION section 5
- M18. The independent review (REVIEW.md) was run on the code and the first pilot before the campaign, not after it, so its fixes reach every record; its 13 findings and their handling are in REVIEW.md
- M19. A multisine-free route that fails only the velocity post-check gets both axis caps lowered in 0.05 m/s steps, not below the trained 1.5 m/s; the realised cap is stored (`meta.vcap_lowered`). Reason: DATA-DESIGN section 10, "a record whose realised velocity exceeds it is lowered, not forced"; the step is a HEURISTIC
- M20. A failing record is logged (`scripts/gantry/thesis-data-verification/logs/generation_failures.csv`) and the stage continues with the next record. Reason: one record must not block the others in an unattended run; failures are listed in section 1
- M21. The multisine scale is tied across a record's two versions and its noise twin even when they are generated in different runs (the stored scale is reused and may not change). Reason: REVIEW.md finding 3; "identical except the noise" must hold by construction
- M22. Process hygiene: stopping a background task in this environment does not stop the scripts it started; two old pilot pipelines kept running and were killed by process id before they could interleave with the new one (no record was written by them). Pipelines are now stopped by process id
- M23. The campaign pipeline opens only when every check on the final code has a PASS line (REVIEW.md finding 1); a check that fails stops the campaign before any record is written
- M24. MATLAB R2025a finished its work but did not exit (0 % CPU for 10 min once without the JVM; later both modes exceeded 20 s at exit, under heavy memory pressure; earlier runs exited normally). Every job therefore ends by printing a completion marker; if MATLAB has not exited 20 s after the marker, the launcher ends its process tree and counts the job as done (the marker proves the statement completed; an erroring statement prints no marker and is reported as failed). Reason: keeps the lower-RAM mode usable on a memory-starved PC without leaving a hung job blocking the one-MATLAB rule
- M25. (Superseded by U6: with one realisation the campaign runs in block order again.) Campaign order by replicate instead of by block: realisation 1 of training and validation, then every test record, then realisations 2 and 3. Reason: the disk ran short (the page file grew to 13.9 GB under memory pressure and took back the space the user freed); DATA-DESIGN section 2 uses the three realisations as separate replicate runs, not pooled, so realisation 1 plus the test set is a complete experiment, and it gets the disk first. The first campaign run (block order) had finished training realisation 1 when it was stopped for this; its one interrupted write was removed by the atomic-save rule
- M26. The three noise twins I1, I4, E5 failed at first (a driver bug: the twin looked up its record in the selected block only; fixed, the full table is now passed). To fit them above the 2 GB reserve, the 3 partial records of training realisation 2 (TR-S1 to S3, 6 files, 0.27 GB) were deleted: a replicate is only usable when complete, and these files are regenerated bitwise from their seeds with realisation 2 (realisation 2 was later dropped altogether, U6). Reason: the twins are the floors the friction finding (F1) makes necessary
- M27. After generation, comment separator lines made of hyphens were replaced by "==" in `tdg_make_model.m` and two check scripts (the project's no-dash rule); comments only, so `tdg_make_model.m`'s SHA-256 in the stored `meta.generator.code_sha256` of the 42 / 38 files is `2a3a569e13b097d9...`, the current file `46a8f73a114ed033...`; the built model `model/gantry_tdg_2025a.slx` is unchanged

## 5. Sizes and runtimes
- Files: noisy 2.1 GB and noise-free 1.8 GB for the 48 records (before the FRF completion: 1.7 and 1.5 GB for 38); per record with multisine about 48 to 52 MB noisy and 42 to 47 MB noise-free, without multisine about 38 to 41 MB and 32 to 34 MB (double precision, MAT v7 compressed)
- Full data set: 96 files (48 noisy + 48 noise-free), 2.1 GB noisy and 1.8 GB noise-free (measured 2026-09-27); an FRF record is 47.9 MB noisy and 42.4 MB noise-free
- Time: one 12 s simulation takes 6 s on the generator model (107 s on the copied four-loop model); a record in both versions took a median 18 s under this PC's memory pressure; all 45 generation log lines together took 14 min of MATLAB
- Disk: C: had 2.7 GB free at the start, 14 GB after the user freed space, then fell to about 2 GB as the page file grew to 13.9 GB; the driver's 2 GB stage rule stopped the campaign at 42 / 38 files
- RAM: available RAM was 3 to 4.5 GB beside the user's other sessions; MATLAB with Simulink needs 2 to 3.5 GB, so several attempts were stopped by the 1.0 GB emergency kill and retried (M14); no record was lost or truncated (atomic writes)

## 6. Where to look
- All verification material is in `scripts/gantry/thesis-data-verification/` (see `REFERENCE.md`): check scripts, results and the noise figure in `checks/`; every run's log in `logs/<run>/attempt*.log` and the sequences `logs/pipeline_pilot.log`, `logs/pipeline_campaign*.log`; per record `logs/generation_log.csv`; failures `logs/generation_failures.csv` (the 3 twin failures of M26, since regenerated); runs with settings later changed in `logs/superseded/`
- The independent review and how each point was handled: `scripts/gantry/thesis-data-verification/REVIEW.md`
- Reading the files in Python: `scipy.io.loadmat(path, mat_dtype=True)`. MAT v7 stores integer-valued doubles in a compact integer type, so without `mat_dtype=True` scipy returns `fs` as uint16, `amp_rms` as uint8 and all-zero arrays (`d_in` of a noise-free file, `f_sim` of a multisine-free record) as uint8; the values are exact, MATLAB reads them as double, and every non-integer signal (y, u_total, x_logical, ...) is float64 either way

## 7. Proposed decision-log entry (for `docs/decisions.md`, not written: outside this task's folders)
- D-220 Thesis data generated from DATA-DESIGN 7 with the generator in `Thesis-writeup/Code/Data/`: 48 records in a noisy and a noise-free version, one realisation of training and validation (user 2026-09-27), one K1, B_tr 106 to 297 Hz on 0.5 Hz, A_prod, D-218 force noise calibrated per Y_op with K1 and rolled off at 300 to 330 Hz (user), jerk time 30 ms (user); the Karnopp truth is kept although the noise shifts stick-slip breakaway (user), so every record's floor is its noisy minus noise-free difference (user); decisions U1 to U9 and M1 to M27, findings F1 to F9 and the audit in GENERATION-REPORT.md

## 8. Audit of 2026-09-27 (independent, read-only, against the original design documents) and how each finding was handled
- A1 FRF 10 of 12 missing: generated 2026-09-27 (`logs/repro_and_frf/`); 48 / 48 files; C3 / C4b / C8 / C14 PASS on all 48
- A2 friction-on floors above the target: accepted by U4; the design statements that each record carries the target were corrected (DATA-DESIGN sections 0, 2, 10; RESULTS-DESIGN S4)
- A3 EXCITATION-VALIDATION section 19 items (2) and (3), recomputed from the data (`scripts/gantry/thesis-data-verification/checks/a3_*.py`):
  - (2) reference force K1 r against the realised noise force, training set, 216 s: median 42 to 59 dB at 50 to 106 Hz (minimum 27 dB, anti channel), 66 to 98 dB at 20 to 50 Hz, 98 to 141 dB below 20 Hz; no 1.5 Hz cell below 10 dB; the claim (about 40 dB) holds
  - (3) friction distortion against the noise-driven FRF error on TF-1 to TF-3: median sigma ratio 105 to 580 per output and Y (49 to 99 on X at 240 to 297 Hz); the claim (about 100x) holds, so friction, not noise, sets R3's threshold; noise-free records periodic to 1e-10
  - note for R3 (?): with 4 realisations and 3 inputs the distortion estimate has 1 degree of freedom per line; medians over lines are solid, a per-line threshold is not
- A4 provenance: C16, the current code reproduces TF-1_r1, TR-P3_r1, E5_r1 and E6_r1 bitwise (every field except `meta.generator`) although 9 code files changed after generation; model hash unchanged
- A5 C15 overclaim: wording corrected to "three equal checksums"
- A6 stale counts: corrected (48 records, 144 seeds, 96 files; U6 line, REVIEW note, D-220 draft, driver header)
- A7 design documents: RESULTS-DESIGN data count (120 committed records), S4 noise text and S7 band corrected; DATA-DESIGN pre-D-218 noise text replaced (user approval 2026-09-27)
- A8 jerk time: 30 ms confirmed by the user (U9); DATA-DESIGN 5.5 aligned
- A9, A10: notes, no change (truth acceleration is not a design limit; Y-loop margins reported, stability is the gate)
