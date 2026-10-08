# Independent review of the thesis data generator

Moved here from `Thesis-writeup/Code/Data/` on 2026-09-27 (GENERATION-REPORT U8). Paths below refer to the layout at review time: `checks/`, `logs/`, `tools/` and `vendor/original/` are now in this folder; the generator code and `SPEC.md`, `IMPLEMENTATION.md`, `GENERATION-REPORT.md`, `VENDORED.md` stay in `Thesis-writeup/Code/Data/`.

Verdict: no blocker found; the code does what SPEC.md says and every design number recomputes, but two majors must be closed before the campaign data are used (the noise checks at the final 300 to 330 Hz roll-off have not run and nothing gates the campaign on them; the JVM smoke test can compare a checksum with itself), plus a latent scale-pairing gap.

- Scope: SPEC.md, IMPLEMENTATION.md, VENDORED.md, GENERATION-REPORT.md, every `.m` in this folder and `checks/`, `tools/*.sh`, diffs against `vendor/original/`, logs up to 22:30 on 2026-09-26; design documents and D-209, D-210, D-218, D-219
- Method: read-only; no MATLAB; recomputations in Python in the session scratchpad (not in the repo); every quoted line below passed `scripts/verify_quote.py` (MATCH OK)
- Data files under `Thesis-writeup/Data/` were not read

## Findings

### 1. Noise checks at the final setting have not run, and the campaign does not wait for them (major)
- Where: `logs/C2_C11_C13_noise_rolloff330.log:5`; `tools/pipeline_campaign.sh:17` to `20`; `GENERATION-REPORT.md` section 2 ("PENDING"); `IMPLEMENTATION.md:75`, `76`, `83`
- What
  - C2 (noise level), C11 and C13 completed only with the superseded 300 to 400 Hz roll-off; the one run at 300 to 330 Hz was killed by the RAM watchdog after 17.6 s, before any output
  - `check_records` (C3, C4b, C8, C14) ran once, on the superseded pilot, and FAILED (C3 on I4 and I6: `isequal` on the NaN `p.T3`); the fix (`isequaln`, `checks/check_records.m:72`) is right but was never re-run
  - the static checks (C4a, C7, C9, C10, log time 21:16) predate the last edits of `tdg_config.m` (21:51), `gtd_save_record.m` and `generate_thesis_data.m` (21:59)
  - `pipeline_campaign.sh` runs every stage without reading any check result; D-218 lists C2 among the "Checks before use"
- Evidence
  - `WATCHDOG KILL available RAM 0.94 GB < 1 GB; killing tree of PID 2412 (12 processes, 2,107 MB)` (rolloff330 log, line 5)
  - `C3 / C4b / C8 / C14 on 14 saved records -> FAIL` (`logs/superseded/pilot_rolloff300-400.log:77`)
  - `for stage in training validation test-I test-E frf noise-twin; do` ... `MATLAB_FLAGS=$FLAGS run "gen_$stage" "generate_thesis_data('$stage')"` (`pipeline_campaign.sh:17` to `19`), no condition
  - my prediction of the C2 statistic (expected Welch band rms: target spectrum, linear interpolation, roll-off weight, Hann 2048 leakage): 3.3786 / 3.7067 / 1.0866 nm at 300 to 400 Hz against 3.3782 / 3.7063 / 1.0866 nm at 300 to 330 Hz; the roll-off moves C2 by 0.01 %, so a pass like the earlier one (worst 1.7 %) is expected, but it is not shown
- Fix: start `pipeline_campaign.sh` only when the latest logs hold `C2 ... -> PASS`, `C13 ... -> PASS` and a PASS of `check_records` on the 14 pilot records made with the final code; re-run `check_references` on the final code; fill GENERATION-REPORT section 2 from those logs

### 2. The JVM smoke test can compare the JVM checksum with itself (major, check integrity)
- Where: `tools/pipeline_pilot.sh:20` to `22`; `tools/run_matlab.sh:33`; consumer `tools/pipeline_campaign.sh:12` (reads `logs/matlab_flags.txt`)
- What: A is the first `sim ... sum|q|` line within 40 lines after ANY line containing "smoke_nojvm". If every -nojvm attempt is killed (7 of 7 so far in `logs/pipeline_pilot.log`), the last such line is run_matlab's "finished with code 137" line, and the JVM run's own `sim` line follows within about 6 lines. A and B are then the same line, the test reports agreement and writes -nojvm for the whole campaign
- Evidence
  - `A=$(grep -A40 "smoke_nojvm" "$LOG" | grep -m1 "^sim .* sum|q|" | sed 's/^sim [0-9.]* s, //')`
  - `B=$(grep -A40 "smoke_jvm" "$LOG" | grep "^sim .* sum|q|" | tail -1 | sed 's/^sim [0-9.]* s, //')`
  - `if [ -n "$A" ] && [ "$A" = "$B" ]; then FLAGS=-nojvm; ...`
  - `echo "[run] $NAME finished with code $rc after $attempt attempt(s)"` (`run_matlab.sh:33`, contains the NAME smoke_nojvm)
- Consequence: the campaign would run in a mode never shown equal to the one C1 and C12 ran in; data risk low (same solver), but the check passes vacuously
- Fix: let `nojvm_smoke.m` write its checksum line to a file named by `usejava('jvm')` (e.g. `logs/smoke_jvm0.txt`, `logs/smoke_jvm1.txt`); compare the two files; both must exist and come from runs that printed `JVM running: 0` and `JVM running: 1`

### 3. The multisine scale is not tied between versions outside one joint run (minor, latent)
- Where: `generate_thesis_data.m:81` to `89`; `IMPLEMENTATION.md:64`; noise twins built at `tdg_manifest.m:108` to `114`
- What
  - with `versions = {'noise-free'}`, `want_n` is false, so the noisy record is not simulated; the noise-free scale is set by its own post-check from `in.chk.scale` and never reads `scale_realised` of the existing noisy file
  - the documented workaround "generate the noisy version first" is exactly the order that can split a pair (noisy scaled to 0.9 because of its noise, noise-free later passing at 1.0)
  - a noise-only twin (`free = false`) sets its scale from its own noisy run only, while its base record's scale comes from the base noisy plus noise-free runs; "same r and f" then holds only if both happen to agree
- Evidence
  - `if need_n || (want_n && need_f)   % the noisy run decides the scale for both` and `if e.free && (need_f || need_n)` (lines 85, 89)
  - `... the noisy run decides the multisine scale for both, so generate the noisy version first or both together` (`IMPLEMENTATION.md:64`)
  - not triggered so far: every pilot scale 1.00, E1 force peak 0.48 of the limit; C3 compares `f_sim` and would catch both cases after the fact
- Fix: in a noise-free-only call load `meta.scale_realised` of the noisy file and require the noise-free run to pass at it; for a twin, read the base record's `scale_realised` (twins are generated after their bases in the campaign order) and assert equality; correct `IMPLEMENTATION.md:64` to "both together only"

### 4. SPEC conflict 1 gives a wrong reason; the jerk-time choice needs its real basis and a user confirmation (minor, (?))
- Where: `SPEC.md:101`; `GENERATION-REPORT.md:23`
- What
  - claim: `7.1 tables (last edited in commit 3f27e25 with the user-approved realised values, after the 5.5 line of 54d7096)`
  - git: the 7.1 T3 values (36 / 30 / 12 / 25 ms) and 5.5's rule list (36 / 28.4 / 13.3 / 25 ms) were both written in commit 54d7096 (20:01); 3f27e25 (20:12) only added the realised jerks to the 7.1 rows, T3 unchanged (`git log -S`, `git show 3f27e25`); "later dated" decides nothing here
  - real support: DATA-DESIGN section 13 records the user decision "the 12 ms jerk time moved onto the 75 % move record" (TR-P3); EXCITATION-VALIDATION 17 and 18 (precedence 2) computed J9 and J10 on 12 and 30 ms
  - 30 ms (TR-P2, VA-P1, I1) against 5.5's 28.4 ms rests on the EV evidence only, and 5.5's own rule places 30 ms on a reference spectral zero at 264 Hz (7.92); 5.5 also rates the whole effect small (40 to 90 nm against 33 000 nm)
- Fix: replace the commit-order argument by the section 13 user decision and the EV evidence; drop "C10a shows the generated references equal those" (finding 5); ask the user to confirm 30 ms over 28.4 ms (?)

### 5. C10a is a rule check, not a record check; IMPLEMENTATION and SPEC overclaim it (minor)
- Where: `checks/check_references.m:50`, `56` to `58`; `IMPLEMENTATION.md:56`; `SPEC.md:101`; `GENERATION-REPORT.md:23`
- What: for TR-P1 to P4 the check swaps the manifest seed for ref_planned's seeds 9001 to 9004 and compares every 5th sample (4 kHz), realisation r1 only; it proves the generator reproduces ref_planned for the same seed. The generated TR-P records use hash seeds, so their setpoint draws are not the J9 and J10 evidence draws
- Evidence
  - `jseed = containers.Map({'TR-P1','TR-P2','TR-P3','TR-P4'}, {9001, 9002, 9003, 9004});`
  - `if isKey(jseed, e.rec), rec.seed = jseed(e.rec); end` and `dr = max(abs(r(1:5:end, :) - squeeze(J.R(jk, :, :))), [], 'all');`
  - `... the 18 training references equal those of the design's excitation evidence bitwise (C10a)` (`IMPLEMENTATION.md:56`)
- Consequence: T3 70 / 85 % and Brun gamma 1.27 hold for another draw of the same rule (EV 17b already lists "one setpoint draw" as a limit)
- Fix: word it as "same rule; bitwise at 4 kHz for ref_planned's seeds"; optionally rerun J9 and J10 on the generated TR-P draws (?)

### 6. Jerk doubling is attributed to the wrong condition (minor, wording)
- Where: `SPEC.md:110`; also `DATA-DESIGN.md:311` and `EXCITATION-VALIDATION.md:410` (flag only, not edited)
- What: the texts say `thirdOrderSetpointETEL` doubles the jerk on moves "too short to reach amax". In the FIR chain the jerk is d / (T1 T2 T3) times the net weight of the impulses at 0, T1, T2, T1 + T2 inside one T3 window, so it doubles when |T1 - T2| < T3 (constant-velocity phase shorter than T3, including every move that does not reach vmax). The doubled moves do reach amax
- Evidence (my Python replica of the ETEL chain reproduces the C9 values exactly)
  - I5 X: T1 = T2 = 47.15 ms, a 17.99, j 1439 (= 2 x 720); I5 Y: |T1 - T2| = 3.35 ms < 25 ms, a 29.99, j 2399
  - E3 X: a 29.94, j 2395; E3 Y: a 49.98, j 3999; TR-P3 (C9): a 22.50 / 37.50, j 3750 / 6250
  - a 0.5 m move (cruise 267 ms) gives j = amax / T3
- Fix: correct the wording in SPEC; numbers unaffected

### 7. Controller gate made margins non-binding; Y-loop phase margins are below the limit the check cites (minor, (?))
- Where: `checks/check_controller_gate.m:42` to `43` (pass = stability only; header lines 8 to 14 flag margins); `DATA-DESIGN.md:251`; `GENERATION-REPORT.md` M11
- What: the design's gate is `stability and minimum gain and phase margins of K1 and every K2 on T_AF over the full Y grid (±0.39) ... a controller failing the gate gets no records`; the check passes on `stable = all(rho < 1);` alone; Y-loop PM 28.5 deg (K1, Y -0.05) and 25.0 deg (K2-g+, Y -0.02) against the 30 deg it cites; recorded as "decided by me" (M11), not as a user decision
- Fix: put both PM values to the user as an explicit acceptance item, or have the design state margin thresholds, before the campaign (?)

### 8. Stale roll-off text and missing labels in the noise code and docs (minor)
- Where and what
  - `SPEC.md:89`: `... per DFT line of the 12 s record up to 400 Hz`; the grid now ends at 330 Hz (`fmax = cfg.noise_edge + cfg.noise_rolloff`)
  - `SPEC.md:95`: the VAR argument cites a target "exactly zero above 400 Hz"
  - `tdg_noise_psd.m:20` to `21`: `shape is open (?) in the design`, decided by the user (U3)
  - `checks/check_noise_draw.m:9` ("300 to 400 Hz roll-off band") and `:50` (print label "300-400 Hz (roll-off)")
  - `tdg_noise_psd.m:14` to `16`: the linear interpolation between Welch bins has no HEURISTIC label in the code (it has one in SPEC.md:92); `check_noise_draw.m:40` (`okB = all(sr >= 0.6 * srel & sr <= 1.5 * srel);`) has none either; CLAUDE.md requires the inline label
- Fix: update the texts; add `% HEURISTIC:` with the reasons already in SPEC and GENERATION-REPORT M5

### 9. C12 revision is justified; what C12 does and does not prove (note)
- Where: `checks/check_model_equivalence.m:10` to `17`; `gtd_run_simulation.m:47` to `48`; `tdg_make_model.m:65` to `76`
- Justified: 1e-9 N failed at 1.2e-8 / 1.5e-8 N (lsim against the Simulink LTI block, 1.7e-11 of the peak force); the calibrated d has a per-sample std of 0.02 to 0.27 N (300 to 400 Hz run; X about 0.12 N at 330 Hz per GENERATION-REPORT U3), so a missing, shifted or sign-flipped d stays four to five orders above 1e-6 N; the per-record assert held in the pilot at 2.1e-10 to 1.9e-8 N
- Limit: `u_plant` is logged at the NoiseSum output, so "+ d" holds by construction; C12 proves that the in-loop controller plus feedforward equals `lsim(plant.Cfb, r - q) + f` (so E5 really ran with 1.2 K1: 9.5e-9 N in the pilot) and that `d_in` enters with the right sign and sample alignment; that d reaches the plant input in stage coordinates is shown only end to end by C2 (finding 1)

### 10. C13 revision is justified (note)
- The first criterion (one draw within 5 %) assumed a flat spectrum; predicted single-draw spread 0.025 / 0.026 / 0.048, so the failing 8.9 % on Y is about 1.9 sigma; the revised 40-draw criterion was saved (21:21:55) before its run ended (21:22) and passed at 300 to 400 Hz (mean ratios 0.991 to 1.008)
- Independent check: my replica of `tdg_noise_draw` (one-sided Phi, `D_k = sqrt(N fs / 2) L_k w_k`, inverse real FFT) gives var(d) / integral of Phi = 0.998 / 1.000 / 0.994 ± 0.005 over 200 draws, and the cross-spectrum phase follows Phi, not its conjugate
- Convention: g1.npz uses `P = np.einsum('kfi,kfj->fij', X, X.conj()) / (fs * np.sum(w ** 2))` (`closed-loop-noise/measure/spec.py:68`), i.e. E[X_i X_j^*], the one `Hi \ Pe / Hi'` needs; Phi_d = T^-1 Phi_e T^-H with T = feedback(Gd, K1) matches D-218 item 4
- Not re-run at 300 to 330 Hz (finding 1)

### 11. C4a, C4b and C8 are partly self-referential (note)
- C4a / C4b rms part: `synth` normalises each channel to unit rms and the check compares rms with the same A it was multiplied by; the line-grid and off-grid parts are the informative tests
- C8 compares `meta.truth` (copied from cfg) with design constants and `meta.controller.gain` with `e.K_gain` (both from the manifest): a configuration check; that the Simulink truth uses these ka, ca and ma is shown only by C2 (48 % of the Y force variance lies at 145 to 155 Hz, so a wrong absorber would fail C2 on Y)

### 12. A post-check failure on a multisine-free record aborts the stage (note, (?))
- Where: `generate_thesis_data.m:96` (`assert(has_ms && s > 0.05, ...)`); `DATA-DESIGN.md:489` (`a record whose realised velocity exceeds it is lowered, not forced`)
- The job ends and `pipeline_campaign.sh` moves to the next stage; the likeliest case, E4 (reference 1.95 m/s against 2.0), was not in the pilot (?)
- Fix: none if E4 passes; otherwise decide with the user (lower the cap, or accept)

### 13. Small defects (note)
- `meta.pair` / `pair` use `fullfile` (`generate_thesis_data.m:129`, `139`), so the `.mat` holds Windows backslashes while MANIFEST.csv has "/"; Linux loaders reading `pair` get backslashes; use `strrep(..., '\', '/')`
- `generation_log.csv` column "version" logs the requested versions (`generate_thesis_data.m:158`), not the files written: E1_r1_nt1 was logged "noisy+noise-free" with no noise-free file
- the folder is untracked in git (`?? Thesis-writeup/Code/Data/`), so `meta.generator.git_head` does not identify the code; `code_sha256` does; commit when the user decides (?)
- the copied driver's assert that the band holds the 212.13 Hz pole (DATA-DESIGN section 10 item 13 expects it to "still pass") is not in `tdg_config.m`; true for 106 to 297 Hz, just not asserted
- EXCITATION-VALIDATION section 19 recomputations (2) (references against Phi_d below 106 Hz) and (3) (noise-driven FRF variance against friction distortion) are neither done nor listed in IMPLEMENTATION section 6; (1) is printed by C2 (multisine over noise at least 52.7 dB in B_tr, at 300 to 400 Hz) (?)
- DATA-DESIGN 5.4 and 7.4 give E6 j 3430 on Y; the D-210 values give a pi / Ta = 3422 (realised 3422); cosmetic

## Verified correct (no finding)
- Noise node: NoiseSum sits between Sum2 (controller + feedforward, stage) and Gain3 (P), fed by `d_in` in stage coordinates, interpolation off, sample time ts; `u_total = u_fb + f_ms` excludes d; `y` is the true position; post-check uses `u_total + d`
- Coordinates: `tdg_truth_lin` maps stage force through B P and outputs P' x (stage), matching the ODE (C11, 1e-16) and the stage-axis target; A_anti is a torque, mapped to stage by P^-1
- Seeds: `seed_sp` to the reference (`tdg_build_inputs.m:231`), `seed_ms` to the multisine (`:239`), `seed_noise` to an own RandStream; keys by FNV-1a (exact in uint64); 292 distinct seeds (C7)
- Pairs: both versions share one `in`, one d draw and one scale per joint run; noise-free uses d = 0; twins take `nt0` keys for ms and sp, a new noise key, and copy E5's `K_gain = 1.2`
- Controllers: K1 designed once on the rigid nominal plant at Y = 0 (`gtd_build_plant.m:23` to `27`), K2-g+ = 1.2 K1 only on E5 and its twin; noise calibrated with K1 for every record, E5 included, as decided
- Manifest: 54 / 18 / 6 / 6 / 12 / 4 = 100 noisy, 96 noise-free; file names unique (asserted); MANIFEST.csv pair and twin_of columns right
- SPEC section 4 against DATA-DESIGN 7.1 to 7.4b, 7.6, 7.9: levels (a = L max, v = L 2 m/s), T3, ranges, Y_op, dwell, directions (TR-T from TP1 to TP4 in `vendor/original/gtd_build_records_telica.m`), lattice residue 0.07, E2 stops, E6 kinematics equal to D-210's `TEL` to all digits; every READING is one the user confirmed
- SPEC section 6: conflicts 2, 3, 5, 6, 9 follow the precedence rule; 4 and 10 are user decisions; 7 correct in its numbers (wording in finding 6); 1 in finding 4
- Vendored copies: `vendor/*.m` byte-identical to `vendor/original/`; ODE equals the working tree (`V_BRK = 1.0e-3`); `ruleOfThumb.m` equals kamtin-fp-model; SHA-256 prefixes in VENDORED.md match those I computed
- Resume: interrupted `*_part.mat` removed; a record whose noisy file exists is re-simulated (deterministic) when its noise-free file is missing in a joint run

## Numbers recomputed
| Quantity | Recomputed | Claimed |
|-|-|-|
| ka = ma (2 pi 150)^2 | 4.4857e6 N/m | 4.4857e6 |
| ca = 2 zeta sqrt(ka ma) | 285.57 N s/m | 285.6 |
| Free-free pole 150 sqrt(1 + 5.05 / 5.05) | 212.132 Hz | 212.13 |
| A_anti = 0.5 x 240 x 0.725 | 87.0 N m | 87 |
| Lines 106 : 0.5 : 297 Hz | 383; 12 s bins k = 1272 : 6 : 3564 (MATLAB 1273 : 6 : 3565), all integer | 383 |
| Target band rms, 29 Welch bins 19.53 to 292.97 Hz, sum x 9.766 Hz | 3.3701 / 3.6955 / 1.0857 nm | 3.370 / 3.696 / 1.086 |
| Share of measured variance (total 9.848 / 10.400 / 6.338 nm) | 11.71 / 12.63 / 2.93 % | 11.7 / 12.6 / 2.9 |
| g1.npz SHA-256 prefix | 84f4e9a427f4b57b | same |
| Synthesis var(d) / integral of Phi (replica, 200 draws) | 0.998 / 1.000 / 0.994 ± 0.005 | 1 |
| I3 cycloid v = sqrt(2 D a / pi), j = 2 a^2 / v | 0.677 / 1.236 m/s, 957 / 1456 | 0.68 / 1.24, 960 / 1460 (realised 960 / 1456) |
| TR-T1, TR-T2 | 0.757 / 1.382, 1338 / 2035; 0.586 / 1.070, 622 / 946 | realised 1338 / 2035; 622 / 951 |
| E4 | full strokes j 332 / 923; X first half stroke pure cycloid v 1.693, j 383 | 383 / 923 |
| E6 | X v 1.0006 (cycloid peak equals the cap), j 3090; Y Ta 46.50 ms, cruise 6.84 ms, j 3422 | 3090 / 3422 |
| Lattice stops | I3 -0.17 to +0.15; E2 -0.33 to +0.39 (residue 0.07); TR-T residues 0, 0.06, 0.02; VA-T 0.04 | same |
| ETEL S-curve jerks | I5 1439 / 2399, E3 2395 / 3999 (replica) | same |
| Measured X1-X2 coherence below 300 Hz | 0.439 at 97.7 Hz | "up to 0.44 at 100 Hz" |

- The band convention matters: a trapezoid integral gives 3.311 / 3.613 / 1.071 nm; target and C2 both use the bin sum, so they are consistent

## Not verified
- C2, C11, C13 at 300 to 330 Hz; C3, C4b, C8, C14 on the final code; C4a, C7, C9, C10 after the 21:51 `tdg_config.m` edit (untracked folder, no diff available)
- MATLAB's `ifft(..., 'symmetric')` with a zero upper half: inferred from C13's measured variance ratios (0.99 to 1.01), not tested directly
- Simulink block parameters beyond what C1 (bitwise) and C12 constrain, e.g. Feedforward1's interpolation
- Whether E4 passes the velocity post-check (finding 12)
- The generated data themselves (not read, per instruction)

## How each finding was handled (implementer, 2026-09-26)
1. Fixed. `tools/pipeline_campaign.sh` now starts only when `logs/pipeline_pilot.log` holds a PASS line for C2, C4a, C6, C7, C9, C10a, C10b, C11, C13 and C3/C4b/C8/C14 on the 14 pilot records, all run on the final code in the new pilot pipeline (one MATLAB job runs every static and noise check); otherwise it logs the missing lines and generates nothing. Results in GENERATION-REPORT section 2
2. Fixed. `checks/nojvm_smoke.m` writes its checksum to `logs/smoke_jvm0.txt` or `logs/smoke_jvm1.txt` (named by `usejava('jvm')` inside the run); the pipeline deletes both before the smoke runs and uses -nojvm only if both exist and are identical (`cmp`)
3. Fixed. `gen_one` in `generate_thesis_data.m`: a noise-only twin reads its record's `scale_realised` (the record must exist first); a run that writes only one version reads the scale stored in the other version's file; in both cases the scale may not change (post-check failure is an error). IMPLEMENTATION section 4 no longer suggests generating one version first
4. Fixed in SPEC section 6 conflict 1 and GENERATION-REPORT M1: basis is the user decision of DATA-DESIGN section 13 (12 ms on TR-P3) and session 1's J9 / J10 evidence (30 ms); the commit-order argument is withdrawn. 30 ms against 28.4 ms stays marked (?) for the user, who asked not to be consulted during this run; generation uses 30 ms
5. Fixed. C10a is described everywhere as a rule check (same rules, bitwise for the same seeds; the generated TR-P records are other draws); the C10a print line says so. Rerunning J9 / J10 on the generated draws is listed as an optional analysis (?)
6. Fixed in SPEC section 6 item 7 (condition |T1 - T2| < T3); the DATA-DESIGN and EXCITATION-VALIDATION wording is flagged for the user, not edited (outside this folder)
7. Kept as it is, stated in GENERATION-REPORT M11 as a decision taken without the user: the design names no margin number, and making 30 deg binding would stop K1 itself, the controller the user fixed for every record; both loops are stable over ±0.39; the Y-loop PM of 28.5 / 25.0 deg is a (?) for the user
8. Fixed: SPEC lines on 400 Hz, `tdg_noise_psd.m` header (roll-off decided), `check_noise_draw.m` print label; HEURISTIC labels added to the Welch-bin interpolation and to C13's [0.6, 1.5] spread band. The history note in `check_noise_draw.m` keeps "300 to 400 Hz" on purpose: it describes the setting of that first run
9. and 10. Agreed; the limits of C12 are stated in IMPLEMENTATION section 5
11. Agreed; stated in IMPLEMENTATION section 5 (C2 is the end-to-end proof that the Simulink truth uses the configured absorber)
12. Fixed. A failing record is logged in `logs/generation_failures.csv` and the stage continues; a multisine-free route that fails only the velocity limit gets its cap lowered in 0.05 m/s steps, not below 1.5 m/s, and the realised cap is stored (`meta.vcap_lowered`), per DATA-DESIGN section 10 "lowered, not forced"
13. `pair` now uses "/"; the log column records the versions actually written; the band-holds-pole assert is back in `tdg_config.m`; EXCITATION-VALIDATION section 19 items (2) and (3) listed as not done in IMPLEMENTATION section 6; E6 "3430" noted in SPEC section 6 item 7; the untracked folder: not committed (commits only on the user's request), `code_sha256` in every file identifies the code

## Update after the review (2026-09-27)
- The user dropped realisations 2 and 3 of training and validation (GENERATION-REPORT U6). Counts in the findings above (54 / 18 / 6 / 6 / 12 / 4 = 100 noisy, 96 noise-free) describe the record table at review time; after U6 it held 18 / 6 / 6 / 6 / 12 / 4 = 52 noisy and 48 noise-free records, and since U7 (noise-only twins dropped) 18 / 6 / 6 / 6 / 12 = 48 records, each in both versions. The review's findings are otherwise unaffected
