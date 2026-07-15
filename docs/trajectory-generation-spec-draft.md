# Trajectory Generation Specification — DRAFT (per-record detail)

**Date:** 2026-07-06. **Status: DRAFT, nothing implemented.** Built on
`generate_oscillatory_multisine_data.m` and `generate_trajectory_data_without_multisine.m`
(both read in full), `docs/data-generation-design-draft.md`, and the donor recipes in
`docs/excitation-recipe-extraction.md`.

## 0. Discrepancies found in the existing scripts (must be resolved, flagged here)

1. **Yaw limit:** multisine script `lim.diff = 6e-3` m; trajectory script
   `lim.diff = sin(0.1)*Lb ≈ 0.0724` m (Garcia 2013). Factor 12 apart. **This spec uses
   6 mm (user instruction).** Consequence: the old anti-symmetric trajectories
   (X_anti_amp up to 0.030 m → 60 mm |X1-X2|) are NOT usable; anti-symmetric reference
   amplitudes are capped at 1 mm here (|X1-X2| = 2*X_anti = 2 mm).
2. **ma_frac:** multisine script default MA_FRAC = 0.50; trajectory script ma_frac = 0.10.
   **RESOLVED (user, 2026-07-06): MA_FRAC = 0.50** (oscillatory multisine script value).
   ma = 5.05 kg, mh_rigid = 5.05 kg. Consistent with the ~157 Hz observed coupled peak
   (isolated fa = 150 Hz parameter).
3. **Multisine period:** currently 1 s tiled ~11x and one realization shared per split.
   Replaced here (Section 1.4).

## 1. Common machinery (all records, both tracks)

| Item | Specification | Basis |
|---|---|---|
| 1.1 Model & rate | `gantry_additional_state_2025a`, fs_sim = 20 kHz; downsample to 4 kHz for training (existing pipeline, no degradation) | existing scripts; user |
| 1.2 Parameters | As in both scripts (mb 22.8, mh 10.1, m1 10.2, m2 10.7, Jb 1.0, Jh 0.05, kb 2x1987.5, Lb 0.725, d 0.1); MSD: fa = 150 Hz, zeta_a = 0.05, ma = MA_FRAC*mh, mh_rigid = mh-ma, L0 = 0.10 | scripts |
| 1.3 Controller | `ruleOfThumb(100 Hz)` per diagonal channel, designed at frozen Y_op of the record (existing convention; controller stays frozen during Y-sweeps) | scripts |
| 1.4 Record timing | 0.5 s hold + 10 s active + 0.5 s hold, padded to 12.0 s total. Multisine period = full 12 s record (grid f0 = 1/12 ≈ 0.083 Hz). NO tiling. | period=record: donor A function; no tiling: noiseless (lessons rule) |
| 1.5 Multisine construction | 3 independent random-phase multisines drawn per record in LOGICAL channels [f_sym, f_anti, f_Y]; phases uniform [0,2pi); flat amplitude on the track band; low-crest-factor selection (best of N random draws), seeded and cached so records are not regenerated; unique seed per record (seed = 100*track + record index), so realizations are independent per record AND per split (train/val/test never share a draw). CF is scored on the post-P STAGE force (and resulting yaw), NOT the raw logical channel (P mixes channels; the peak/yaw budgets live in physical coordinates). | random phase + per-split independence: donor A / Jan; CF selection: current gantry scripts, justified by our peak-force + 6 mm yaw constraints (absent in Jan's open-loop MSD); logical coords + CF-on-stage: HEURISTIC |
| 1.6 Coordinate transform | RESOLVED (2026-07-06): stage forces from logical (generalized) forces via **F_stage = P^{-1} f_logical** (virtual-work dual of q_stage = P' q_logical; plant is sys = P'*G*P). Explicitly F_X1 = 0.5*f_sym + f_anti/Lb, F_X2 = 0.5*f_sym - f_anti/Lb, F_Y = f_Y. NB f_anti is a yaw TORQUE [N*m] (logical coord 2 is the tilt angle theta ~ (X1-X2)/Lb), NOT a force; the naive f_sym +/- f_anti map is both mis-normalized (2x on sym) and dimensionally wrong. Verified 5 ways by `gtd_check_transform` (P^{-1} vs analytic, sym->equal rails, anti->opposite rails, work invariance, plant DC consistency). | derived from P convention; value-correctness verified (lessons rule) |
| 1.7 Amplitudes (per logical channel RMS, fixed across records of a track) | A_sym = 40 N, A_Y = 30 N (≈5% of 916/656 N RMS limits, D-056 lineage); A_anti from the yaw budget: target multisine-induced |X1-X2| peak ≤ 2 mm via closed-loop yaw compliance at Y_op with peak factor 4 (placeholder value: compute per record; expected order: a few N). Final levels via GATE 2 (delta_a activation). | fixed-absolute-amplitude: donor A; levels HEURISTIC + GATE |
| 1.8 Yaw budget split (6 mm total) | Reference yaw ≤ 2 mm (X_anti_amp ≤ 1 mm) + multisine-induced yaw ≤ 2 mm + 2 mm margin (transients, coupling) | HEURISTIC |
| 1.9 Injection | f added to plant input in parallel with Cfb output (unchanged from multisine script) | donor D (Bolderman), citable rationale |
| 1.10 Validation & enforced limits | All limits taken from the generator's `lim` struct: pos_X 0.375 m, pos_Y 0.400 m, yaw \|X1-X2\| 6 mm, velocity 2.0 m/s (all axes), acc_X 30, acc_Y 50 m/s^2 (checked on r), force peak [2000 2000 1420] N, RMS [916 916 656] N. `validate_ref` (pos, yaw, vel, acc on r), `validate_response` (q: pos, yaw, vel), `validate_forces` (u_total peak+RMS); proportional scale-down loop retained as safety | enforced `lim` struct (existing scripts) |
| 1.11 Jerk profiles | All point-to-point motion via `thirdOrderSetpointETEL(dist, vmax, amax, amax/jerkTime, Inf, ts)`; jerkTime in [0.025, 0.050] s as in existing records. Rationale: machine-realistic third-order profiles keep reference spectral content below ~1/jerkTime = 20-40 Hz, so the resonance band is excited ONLY by the multisine (clean attribution of delta_a activation; also avoids ref-induced ringing, cf. spike hypothesis 2) | existing scripts; rationale HEURISTIC |
| 1.12 Saved signals | u_total, u_fb, f_sim, y, x_logical (6 baseline states, logical coords), delta_a + vdelta_a (= x_aug, the 2 MSD states; vdelta_a differentiated), r_sim, Y_trajectory, t_sim, fs, dt, split, amp_rms (= [A_sym, A_anti, A_Y] in [N, N*m, N]) + multisine seed and track id. Full 8-state ground truth [x_logical, x_aug] for encoder pre-training. | existing + provenance + full-state (D-085) |
| 1.13 Informativeness diagnostic | Per record: with/without-multisine delta_a RMS ratio + residual PSD peak in [145, 165] Hz (window follows the OBSERVED coupled peak once 157-vs-150 is confirmed) | existing diagnostic |

## 2. Tracks (same 22 trajectories, two multisine variants)

| Track | Band | Used for | Lines (f0 = 1/12 Hz) |
|---|---|---|---|
| JE | [1, 200] Hz | Joint estimation (baseline params drift-regularized + ANN) | ~2390 |
| AUG | [130, 180] Hz | Augmentation-only (baseline frozen) | ~600 |

Matched pairs: same trajectory, same seed index per record; only the band (and hence the
drawn line set) differs. **Implementation: a script toggle `TRACK = 'joint' | 'augmentation'`
mapping directly onto the existing `MULTISINE_BAND` switch (`'joint'` -> `'broadband'` [1,200] Hz,
`'augmentation'` -> `'narrowband'` [130,180] Hz), alongside the existing `USE_MSD` toggle
(user requirement); each track writes to its own output subfolder.** Rationale: joint estimation needs data informative for theta_base
(low band) AND the ANN everywhere; augmentation-only concentrates power on the target with
low-band behavior constrained by the trajectory content (donor A full-grid precedent for
JE; AUG concentration is OURS, mid-band risk covered by the matched JE records + E4).

## 3. Training records (T1-T14)

All Y values in m, amplitudes in m, X_anti_amp ≤ 0.001 everywhere (Section 0.1).

**T1-T5 — standstill multisine (frozen-Y).** r = [0, 0, Y_op] constant, no motion.
Y_op = -0.30 (T1), -0.15 (T2), 0 (T3), +0.15 (T4), +0.30 (T5).
Purpose: local LPV information at 5 scheduling points (>= 3 required by quadratic M(Y);
Ghosh 2018); cleanest delta_a activation (no trajectory force masking); JE track: baseline
sensitivity at operating points; AUG track: resonance learning.
Note: multisine is the ONLY excitation; feedback reacts to multisine-induced error only.

**T6-T8 — Y-sweep (scheduling rate).** Oscillatory class (half-cosine fade 0.5 s, as
existing `make_ref_oscillatory`): Y = Y_center + A_y*sin(2*pi*f_y*t), Y_center = 0,
A_y = 0.30.
- T6: f_y = 0.2 Hz (peak Ydot 0.38 m/s, peak acc 0.47 m/s^2)
- T7: f_y = 0.7 Hz (peak Ydot 1.32 m/s, peak acc 5.8 m/s^2; within 2.0 / 50 limits)
- T8: f_y = 0.35 Hz + X_sym overlay 0.05 @ 1.1 Hz (rate + coupling mix)
Purpose: Ydot-dependent terms (dM/dt): invisible in frozen-Y data (LPV lit / Toth 2010,
lit log §10); two distinct rates separate effect order; T8 adds X-Y coupling under sweep.

**T9-T12 — randomized jerk-limited setpoint sequences (position-level APRBS).**
Structure: sequence of point-to-point moves (Section 1.11 profiles) with 0.1 s holds
between moves, filling the 10 s active window; setpoints drawn uniformly per record:
X_sym in [-0.10, +0.10], Y in [-0.30, +0.30], X_anti in [-0.001, +0.001]; unique seed per
record.
Ladder (base = 75% of the enforced per-axis limits, HEURISTIC margin set by user:
vmax_X = vmax_Y = 0.75*2.0 = 1.5 m/s, amax_X = 0.75*30 = 22.5, amax_Y = 0.75*50 = 37.5 m/s^2;
levels below are fractions of this base; jerkTime per record):
- T9: 30% (vmax_X 0.45, amax_X 6.75, vmax_Y 0.45, amax_Y 11.25), jerkTime 0.050
- T10: 60% (0.90, 13.5, 0.90, 22.5), jerkTime 0.035
- T11: 100% (1.50, 22.5, 1.50, 37.5), jerkTime 0.025
- T12: 60% with X_anti active (setpoints +-0.001) and X_sym reduced to [-0.05, 0.05],
  jerkTime 0.040 (yaw-mode information within the 6 mm budget)
Purpose: operating-range and off-orbit coverage; amplitude ladder (Schoukens & Ljung:
cover amplitude range); randomized setpoints kill "all records alike"; APRBS adaptation
flagged PARTIAL (Nelles standard, position-level version OURS).

**T13-T14 — lissajous (multi-axis simultaneous).**
- T13: X_sym 0.08 @ 1.5 Hz, Y 0.25 @ 0.4 Hz, X_anti 0 (vel/acc: X 0.75 m/s / 7.1 m/s^2;
  Y 0.63 / 1.6 — within limits)
- T14: X_sym 0.06 @ 1.3 Hz, Y 0.30 @ 0.7 Hz, X_anti 0.001 @ 0.8 Hz (Y: 1.32 m/s,
  5.8 m/s^2 — within limits; |X1-X2| ref = 2 mm max)
Purpose: MIMO coupling with all logical channels simultaneously active (Colin
informativity; MUMI-style independent per-channel content); visits (X, Y, Xdot, Ydot)
combinations single-axis records never reach; F1Tenth lemniscate analog.

## 4. Validation records (V1-V4) — separate generation runs (Jan's protocol)

Fresh multisine realizations AND fresh setpoint seeds; unseen interior Y; same classes
and amplitude range as training. Purpose: checkpoint selection only.
- V1: standstill multisine at Y = +0.10
- V2: randomized setpoints (60% ladder), Y setpoints restricted to [-0.30, -0.14]
  (centered ~ -0.22), new seed
- V3: Y-sweep A_y = 0.15 around Y_center = +0.10, f_y = 0.2 Hz (trained rate, unseen
  Y-range/center)
- V4: lissajous X_sym 0.07 @ 1.4 Hz, Y 0.20 @ 0.5 Hz around Y_center = -0.10

## 5. Test records (E1-E4) — untouched until final evaluation

- **E1 — resonance sweep (cross-class):** standstill at Y = 0; force sinesweep
  130 -> 180 Hz over the 10 s active window (5 Hz/s; ISO 7626-2 rule S_max ~ BW^2 with
  BW ~ 15 Hz gives ample margin; Gloth & Sinapius as analysis source), amplitude 80% of
  track levels. Applied on f_Y. RESOLVED (user + L0 comment in
  `generate_trajectory_data_without_multisine.m`, "+Y direction"): the hidden MSD acts
  along Y (the beam axis), so f_Y is the correct sweep channel.
  Purpose: does the learned augmentation reproduce the resonance under a never-trained
  signal class (Bouc-Wen function).
- **E2 — independent multisine at unseen Y:** standstill at Y = +0.22, fresh realization,
  80% amplitude. Purpose: same-class generalization + Y-interpolation (donor A + E).
- **E3 — above-ladder randomized setpoints:** new seed; base = 90% of the enforced per-axis
  limits (vmax_X = vmax_Y = 1.8, amax_X = 27, amax_Y = 45), HEURISTIC margin, above the T11
  training top (75%) but a deliberate 10% below the hardware limits (2.0 / 30 / 50);
  jerkTime 0.030 (unseen value). Purpose: amplitude/rate extrapolation beyond the training
  ladder (mirrors Jan's extrapolate set).
- **E4 — multisine OFF:** T11-class trajectory, new seed, f = 0. Purpose: regression
  check that the augmentation does not distort baseline behavior when the resonance is
  barely excited; also the mid-band no-harm check for the AUG track.

## 6. Consistency-with-literature summary

Fresh realization per record, period = record, no CF optimization, transient handling,
fixed absolute amplitudes: donor A (Hoekstra EJC 2025 + released script). Reference-class
diversity x ladder, record-level splits, separate val generation: donor C (F1Tenth) with
val protocol per Jan's ECC script (independent draws, never segments). Injection point:
donor D (Bolderman, quoted rationale). Test suite functions: donor E (Bouc-Wen) +
extrapolate set (Jan's repo). Frozen-Y count: Ghosh 2018 + quadratic M(Y) (>= 3, using 5).
Y-rate records: LPV dynamic-dependence argument (Toth 2010, section to be quoted).
Logical-coordinate multisine + yaw budget: HEURISTIC (no precedent, 4 search rounds;
GVT force appropriation analogy). Band split JE/AUG: JE full-grid per donors; AUG
narrowband concentration OURS with named mitigations (matched JE records, E4).

## 7. Open before implementation

1. ~~ma_frac~~ RESOLVED: 0.50. Diagnostics window [145, 165] Hz retained (brackets both
   150 and the coupled ~157 Hz peak). The ~157 Hz figure comes from the user's MATLAB
   FRF of the linearized Simulink model (observed coupled peak; isolated parameter
   fa = 150 Hz), consistent with the reduced-mass shift at ma_frac = 0.50.
2. ~~A_anti value~~ RESOLVED (2026-07-06): `gtd_size_anti_amp` sizes A_anti (a torque, N*m)
   so the anti-driven closed-loop peak |X1-X2| equals the 2 mm budget exactly, via the SISO
   yaw transfer H_yaw = [1 -1 0]*sys_cl*([1;-1;0]/Lb). Sym/Y coupling absorbed by the 2 mm
   margin + hard 6 mm (Phase 4).
3. ~~E1 sweep channel~~ RESOLVED: f_Y (MSD acts along Y).
4. GATE 2 amplitude confirmation (delta_a activation at 40/30 N, both tracks).
5. ~~Transform normalization check (1.6)~~ RESOLVED (2026-07-06): F_stage = P^{-1} f_logical
   (see 1.6); f_anti is a yaw torque. Verified by `gtd_check_transform`.
6. ~~Multisine phase/realization~~ RESOLVED (2026-07-06): random phase + low-crest-factor
   selection (best of N random draws), seeded and cached; independent realization per record
   AND per split (train/val/test never share a draw). CF scored on the post-P stage force +
   yaw, not the raw logical channel. Deviates from Jan (who uses no CF optimization),
   justified by our peak-force + 6 mm yaw constraints.
7. ~~Ladder margins~~ RESOLVED (2026-07-06): p2p training top (T11) = 75% of the enforced
   per-axis limits; test extrapolation (E3) = 90%; no record sits at the hardware limit.
