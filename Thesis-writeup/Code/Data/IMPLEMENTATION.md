# Implementation: the thesis data generator

What was built in this folder, readable without the code. The specification is `SPEC.md`; provenance of copied files is `VENDORED.md`; results and every decision taken during generation are in `GENERATION-REPORT.md`; `REFERENCE.md` points to the verification material (checks, logs, launchers, review, source copies), which was moved to `scripts/gantry/thesis-data-verification/` on 2026-09-27 so that this folder holds only the generator and its documentation.

## 1. Folder layout
| Path | What it does |
|-|-|
| `generate_thesis_data.m` | Driver. Per record (local function `gen_one`): build inputs once, simulate the noisy version (own encoder-noise draw) and the noise-free version (v = 0), post-check both on the simulated truth, scale the multisine down if a limit fails (same scale for both; a scale already stored in a saved twin is reused and may not change), lower the velocity cap of a multisine-free route that fails only the velocity limit, save both, log one line; a failing record is logged in `generation_failures.csv` (in `scripts/gantry/thesis-data-verification/logs/`) and the stage continues; ends by writing `MANIFEST.csv` per version. Resumable: skips finished files, removes interrupted `*_part.mat` writes, stops a stage below 2 GB free disk |
| `tdg_config.m` | Every constant (truth, controller, record timing, limits, B_tr, A_prod, noise settings, output folders), each with its source |
| `tdg_manifest.m` | The record table: 48 records (DATA-DESIGN 7.9; one realisation of training and validation, no noise-only twins, user decisions 2026-09-27), each generated noisy and noise-free; every seed from its key |
| `tdg_seed.m` | Seed = FNV-1a hash of the key (truth, set, record, realisation, twin, signal) |
| `tdg_build_inputs.m` | Reference, reference check, multisine and linear pre-check for one manifest row (deterministic) |
| `tdg_post_check.m` | Binding limit check on the simulated truth (position, yaw, velocity, peak and rms plant force) |
| `tdg_truth_lin.m` | T_AF linearised at rest at Y; friction off, or with the tanh friction slope cc g at rest when g is given (the loop the noise is shape-corrected on, D-225 amendment) |
| `tdg_noise_psd.m` | Encoder-noise auto-spectra, one per axis: phi_j = Phi_e,jj / abs(S_jj)^2 above the 50 Hz corner, Phi_e,jj below, at a record's Y_op with K1 (D-225 amendment) |
| `tdg_noise_draw.m` | One realisation of the three independent encoder noises, own seed per axis derived from the record's noise seed |
| `tdg_make_model.m` | Builds `model/gantry_tdg_2025a.slx` from the vendored model (recorded loop only, noise block, plant-force log, chart repointed to the tanh ODE with Parameter `g_fric`, D-224) |
| `gantrySystemExtendedTanh.m` | The truth ODE the recorded loop calls (D-224): the vendored ODE's mass, damping and stiffness sections verbatim, friction `cc_i tanh(g v_i)` per stage rail; byte-identical to `scripts/gantry/coulomb-tanh-gain/gantrySystemExtendedTanh.m`, where D-223 tested it |
| `tdg_init.m`, `tdg_init_paths.m` | Path, short Simulink cache folder, model loading |
| `tdg_write_manifest.m` | `MANIFEST.csv` per version from the files on disk |
| `gtd_*.m` (6 files) | Modified copies of the copied generator; every change marked `THESIS-DATA` |
| `vendor/` | Verbatim runtime copies (ODE, controller design, setpoint generator, unchanged `gtd_*`) and the source Simulink model `gantry_additional_state_coulomb_2025a.slx` that `tdg_make_model.m` builds from |
| `model/` | The generator's Simulink model (built, not copied) |
| `noise/` | `noise_target.mat` (measured spectrum) and the script that exports it from `g1.npz` |
| `REFERENCE.md` | Where the verification material is and what it holds |

## 2. Generator changes
Numbering follows DATA-DESIGN section 10 ("Implementation to do before generation"); C-numbers are the checks of section 4.

| # | What changed | Where | Why | Check |
|-|-|-|-|-|
| 1 | One fixed controller K1 (ruleOfThumb, 100 Hz, rigid nominal plant at Y = 0) in every record; K2-g+ = 1.2 K1 for E5; the pre-check plant stays at Y_op | `gtd_build_plant.m` | DATA-DESIGN 5.11, user decision | C6 (gate) |
| 2 | Binding limit check on the simulated truth; on failure the multisine is scaled down in 0.1 steps and both versions re-simulated; a multisine-free route failing only the velocity limit gets its cap lowered in 0.05 m/s steps (not below 1.5 m/s); realised scale and cap stored | `tdg_post_check.m`, driver | DATA-DESIGN section 10 item 2 and the feasibility line "lowered, not forced" | C5 (every record) |
| 3 | Encoder noise v on the branch into the error sum (the controller sees q + v); `y` = q + v, `y_true` = q, `v_enc` = v recorded; u_total is the plant force; own seed per axis; v = 0 for the noise-free version (D-225; was the force d of D-218, block NoiseSum, `d_in`) | `tdg_make_model.m` (EncoderSum, EncoderNoise), `gtd_run_simulation.m`, `gtd_save_record.m`, `tdg_noise_psd.m`, `tdg_noise_draw.m` | D-225 and amendment, NOISE-INJECTION sections 3 to 5 | C12, C18, C19 |
| 4 | Per-record kinematics of the ILC-shape and route records | `tdg_manifest.m` (the reference builder already read them per record) | DATA-DESIGN 5.7 | C9, C10a |
| 5 | X-only / Y-only alternating routes (I4) | `gtd_make_reference_telica.m`, class `route`, pattern `alternate` | DATA-DESIGN 5.7, 7.3 | C9 |
| 6 | Deterministic setpoint sequences: the lattice shuttle of I3 profiled by the S-curve generator (I5, E3); long strokes with a first half stroke (I6, E4) | `gtd_make_reference_telica.m`, class `route` | DATA-DESIGN 7.4b | C9, C10b |
| 7 | Log-strata move distances for the S-curve move records (TR-P, VA-P1, I1), by the rule of the excitation evidence (`ref_planned.m`) | `gtd_make_reference.m`, `ref_logstrata` | DATA-DESIGN 5.6 | C10a (same rule; the records themselves are other draws of it) |
| 8 | Collision-free seeds from a text key; every seed stored in the file and the manifest | `tdg_seed.m`, `tdg_manifest.m`, `gtd_save_record.m` | DATA-DESIGN section 10 item 8 | C7 |
| 11 | Acceleration assert: named exception for E6 (1.31 x max), 1 % margin flag for E3 | `gtd_validate_ref.m` | DATA-DESIGN section 10 item 11 | C9 (E3 did not need the margin) |
| 12 | Controller gate: stability and margins of K1 and K2-g+ on T_AF over Y = ±0.39 | `scripts/gantry/thesis-data-verification/checks/check_controller_gate.m` | DATA-DESIGN 5.11, item 12 | C6 |
| 13 | Multisine lines on the 0.5 Hz grid (every 6th DFT line), band 106 to 297 Hz; line step in the cache key and in the saved info; disk cache off by default | `gtd_make_multisine.m`, `tdg_config.m` | D-219 | C4a, C4b |
| n1 | Every signal saved in double; new fields `v_enc` and `y_true` (D-225; `d_in` under D-218), `seed_ms`, `seed_sp`, `noise_seed`, `version`, `pair`, `meta` (settings, truth and controller parameters, band, scale, post-check, noise model, code hashes); atomic write | `gtd_save_record.m` | NOISE-INJECTION section 5; handoff (seeds, settings and truth parameters in each file) | C14 |
| n2 | Model copy keeps only the recorded "Extended ODE" loop (the three other plant loops share no signal with it) | `tdg_make_model.m` | runtime 107 s -> 6 s per 12 s run; no Simscape Multibody dependency | C1 (bitwise) |
| n3 | One simulation per record version (the copied generator also ran each record without its multisine, for a printed diagnostic only) | `gtd_run_simulation.m` | runtime; no saved field depended on it | C1 |
| n4 | Plant-force log `u_plant` in the model | `tdg_make_model.m` | proves u_total + d is what the plant received | C12 (every record) |
| n5 | Record table, config, driver replaced; two version folders | `tdg_manifest.m`, `tdg_config.m`, `generate_thesis_data.m` | DATA-DESIGN section 7, user decisions | C7 |
| n6 | Simulink cache in a short temp folder | `tdg_init.m` | the repo path exceeds the Windows 260-character build limit | runs |
| n7 | No per-record figures (`gtd_plot_record.m` not used) | driver | disk (about 0.2 MB per figure, 96 files); the checks produce the figures that prove something | none |

## 3. What differs from the copied generator (`scripts/gantry/thesis-data-verification/vendor/original/`)
- Same: the truth ODE's mass, damping, stiffness and absorber (copied verbatim into `gantrySystemExtendedTanh.m`), the controller rule, the setpoint generators (`thirdOrderSetpointETEL.m`, `cyc_profile`, the oscillatory fade), the multisine synthesis and crest-factor selection, the anti-channel yaw cap, the linear pre-check, the saved schema of the copied fields, the solver
- Different: the friction law, tanh instead of Karnopp (D-224; the vendored Karnopp ODE is used only with cc = 0, by `tdg_truth_lin.m`, where both laws are the frictionless ODE); everything in section 2; in short: one controller, the band and line grid, the level, the record table and seeds, the noise, double precision, the post-check, the model loops, the output folders
- Numerical identity where nothing was meant to change: the model with d = 0 equals the original bitwise (C1, for the Karnopp build before D-224; since D-224 the chart differs by design and C17 checks the tanh model against the D-223 replica); the generator builds the 18 training references by the same rules as the design's excitation evidence (J9, J10), bitwise for the same seeds (C10a); the generated S-curve records use their own hash seeds, so they are other draws of the same rule

## 4. How to regenerate
- Prerequisites: MATLAB R2025a with Simulink and Control System Toolbox; Python only for `noise/export_noise_target.py` and the figure script
- Start MATLAB in this folder and run `tdg_init_paths` (path and the short Simulink cache folder)
- Once: `tdg_make_model` (builds the model); `noise/export_noise_target.py` (only if `noise/noise_target.mat` is missing)
- One record, both versions: `generate_thesis_data({'TR-P3_r1'})`; several: a cell of file names or record IDs (`{'I3'}` gives I3_r1)
- A block: `generate_thesis_data('training')` (also `validation`, `test-I`, `test-E`, `frf`)
- Into another folder (reproduction test, nothing written into `Thesis-writeup/Data`): `generate_thesis_data({'TF-1_r1'}, struct('out_dir', D))`, then `check_reproduction('TF-1_r1', D)` from the verification folder
- Everything: `generate_thesis_data()`
- Only one version: `generate_thesis_data(sel, struct('versions', {{'noise-free'}}))`; normally generate both together (the noisy run decides the multisine scale); when the other version's file already exists, its stored scale is reused and may not change, so a pair can never split (REVIEW.md finding 3)
- Existing files are never overwritten; to regenerate a record, delete its file(s) first
- One MATLAB at a time. Optionally from a shell under the RAM watchdog: `scripts/gantry/thesis-data-verification/tools/run_matlab.sh <name> "generate_thesis_data(...)"` (retries after an emergency RAM kill)
- Run log: one line per generated record in `scripts/gantry/thesis-data-verification/logs/generation_log.csv` (`cfg.log_dir`)
- Output: `Thesis-writeup/Data/Coulomb-and-MSD[-noise-free]/{Training,Validation,Test}/<file>.mat` and `MANIFEST.csv`
- Reading the files in Python: `scipy.io.loadmat(path, mat_dtype=True)`. MAT v7 stores integer-valued doubles in a compact integer type, so without `mat_dtype=True` scipy returns `fs` as uint16, `amp_rms` as uint8 and all-zero arrays (`d_in` of a noise-free file, `f_sim` of a multisine-free record) as uint8; the values are exact, MATLAB reads them as double, and every non-integer signal (y, u_total, x_logical, ...) is float64 either way

## 5. Checks (criteria in each script's header, stated before its first run; scripts, results and logs in `scripts/gantry/thesis-data-verification/`)
| ID | Script | What it proves | Result |
|-|-|-|-|
| C1 | `check_model_equivalence.m` | generator model with d = 0 equals the original model bitwise (motion + multisine + friction record) | PASS, max difference 0 (Karnopp build; superseded by design since D-224, see C17) |
| C17 | `scripts/gantry/coulomb-tanh-gain/check_tanh_model.m` | generator model with the tanh chart (D-224) equals the D-223 replica of the recorded loop bitwise, on the TR-T1 inputs, noise-free and noisy; the replica itself reproduces the saved Karnopp records bitwise (D-223 gate, TR-T1 both versions and TR-S3) | PASS, max difference 0 in both versions; plumbing 1.2e-8 N (2026-09-28) |
| C12 | same, and per record in the driver | u_plant = u_total + d (since D-225: u_plant = u_total, i.e. the saved y is the signal the controller saw) | PASS at 1e-6 N (first criterion 1e-9 N failed at 1.5e-8 N, rounding in the controller states; revised, see the script header) |
| C18 | `scripts/gantry/coulomb-tanh-gain/check_encoder_noise.m` B | encoder noise shape: standstill Y 0, friction on at g = 1000, band ratios simulated / measured above 50 Hz within 10 % of one level (D-225 amendment) | PASS 2026-09-28: 0.97 to 1.03, level 1.00, total 9.82 / 10.34 / 6.33 nm vs measured 9.85 / 10.40 / 6.34; 10 to 50 Hz 0.68 to 0.79 (stated limit) |
| C19 | `scripts/gantry/coulomb-tanh-gain/check_encoder_noise.m` A | generator model with encoder noise equals the replica bitwise (TR-T1 inputs, both versions), u_plant = u_total | PASS 2026-09-28: max difference 0 in both versions, plumbing 1.3e-8 N |
| C20 | `scripts/gantry/coulomb-tanh-gain/check_oracle_pilot.py` | the closed-loop oracle of the post-run report (D-227) reproduces the noise-free pilot records (TR-T1, TR-S3, VA-P1, generated to a scratch folder) at the 4 kHz training rate | PASS 2026-09-28: 2 nm (TR-T1), 0.1 to 0.95 um (multisine records; the 4 kHz discretisation) |
| C11 | `check_noise_calibration.m` | truth linearisation = finite-difference Jacobian of the ODE | PASS, 1e-16 |
| C2 (D-218, superseded by C18) | `check_noise_calibration.m` | friction off, standstill, every Y_op of the manifest: simulated error 19.5 to 300 Hz within 2 % of 3.370 / 3.696 / 1.086 nm | PASS at the final 300 to 330 Hz roll-off, all 17 Y_op within 1.4 %; `scripts/gantry/thesis-data-verification/checks/fig_noise_calibration.png` |
| C13 (D-218 draw; the per-axis draw of D-225 is checked by C18) | `check_noise_draw.m` | the noise draw has the target covariance and spread | PASS (first criterion wrong and failed; revised before the rerun, see header) |
| C6 | `check_controller_gate.m` | K1 and K2-g+ stable on T_AF over ±0.39; margins | PASS (stability); Y-loop PM 28.5 / 25.0 deg flagged below 30 deg |
| C7 | `check_references.m` | 48 records, distinct seeds | PASS (earlier runs: 100 records before the one-realisation decision, 52 before the noise-only twins were dropped) |
| C9 | `check_references.m` | every reference within limits (E6 exception, E3 margin) and the realised kinematics | PASS |
| C10a | `check_references.m` | the training references follow the same rules as the J9 / J10 evidence: bitwise at 4 kHz for r1, the S-curve records with ref_planned's seeds 9001 to 9004 (a rule check, not a record check) | PASS |
| C10b | `check_references.m` | route(cyc, together) = telica, bitwise | PASS |
| C4a | `check_references.m` | constructed multisine: 383 lines on the grid, rms 240 / 87 / 180 (the rms part is partly self-referential: synth normalises to unit rms; the line grid is the informative part) | PASS |
| C3, C4b, C8, C14 | `check_records.m` | on saved files: the noise-free twin identical outside the noise path, delivered excitation, truth parameters, schema, precision, stored seed regenerates d (to update before the regenerated data: it asserts V_BRK and d_in, now friction_gain and v_enc with per-axis seeds) | PASS on every saved record (38 records after the move; before it 42, the 4 noise-only twins included) |
| C5 | driver, every record | post-simulation limits on the truth | per record in `scripts/gantry/thesis-data-verification/logs/generation_log.csv` |
| C15 | `nojvm_smoke.m` + `tools/pipeline_pilot.sh` (verification folder) | MATLAB without its JVM simulates a real record (TR-P3_r1, multisine, noise, friction) with the same three 17-digit checksums (sums of abs(q), abs(delta_a), abs(u_total)) as MATLAB with its JVM; each mode writes its own file; equal checksums, not a bitwise comparison of every array | PASS |
| C16 | `check_reproduction.m` (verification folder) | the current code regenerates a saved record (into a scratch folder, `out_dir` option) bitwise equal in both versions, every field except `meta.generator` (audit A4, 2026-09-27) | PASS on TF-1_r1, TR-P3_r1, E5_r1, E6_r1 (FRF, moves with multisine, K2-g+, ILC exact kinematics), although 9 code files changed after those records were saved; model hash unchanged |

What the checks do not prove (REVIEW.md findings 9 and 11): C8 compares configuration with design constants; that the Simulink truth really uses these ka, ca and ma is shown end to end by C2 (48 % of the Y noise force sits at 145 to 155 Hz, so a wrong absorber would fail C2 on Y). C12's "+ d" holds by construction of the log point; it proves the in-loop controller and feedforward (E5 ran with 1.2 K1) and the sign and timing of d.

## 6. Parts of the design not implemented, and why
- Realisations 2 and 3 of training and validation: dropped by the user (2026-09-27, GENERATION-REPORT U6)
- Deferred records of DATA-DESIGN 7.5 and control truths T_0, T_F, T_OA (7.7): out of scope (handoff section 2)
- Items 9 and 10 of DATA-DESIGN section 10 (band field for TE-B1, sweep law for TE-B2, payload on the rigid head mass): deferred records only
- TF-4 at Y = -0.35: marked (?) in 7.6 and not in the 7.9 count
- The range and density labels (DATA-DESIGN section 6) and the pair acceptance table (7.4b): evaluation steps on generated data, not generation; the realised reference peaks they need are stored in every file (`meta.ref_peaks`) and listed in GENERATION-REPORT
- EXCITATION-VALIDATION section 19 recomputations: (1) printed by C2; (2) and (3) done after the audit from the generated data (`scripts/gantry/thesis-data-verification/checks/a3_references_vs_noise.py`, `a3_frf_noise_vs_friction.py`), both claims hold (GENERATION-REPORT section 8)
- Rerunning J9 / J10 on the generated TR-P draws (instead of ref_planned's draws): an optional analysis step (?)
