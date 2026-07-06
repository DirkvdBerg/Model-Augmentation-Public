# Implementation Instructions: Two-Track Record Generator

**Date:** 2026-07-06. **Audience:** a fresh session/agent implementing the new data
generator. This document is the single instruction source; it assumes NO access to the
design conversation. Read in this order before writing any code:
1. `CLAUDE.md` and `tasks/lessons.md` (active constraints, non-negotiable)
2. This document
3. `docs/trajectory-generation-spec-draft.md` (the record-by-record specification)
4. `Matlab-scripts/Augmentation/data/generate_oscillatory_multisine_data.m` (the base
   script whose skeleton you will preserve)

Supporting rationale (do not need for implementation, cite in decisions log):
`docs/data-generation-design-draft.md`, `docs/excitation-recipe-extraction.md`,
`docs/excitation-design-literature.md` Sections 8-10.

---

## 1. Mission

Create ONE new MATLAB script,
`Matlab-scripts/Augmentation/data/generate_track_record_data.m`, adapted from
`generate_oscillatory_multisine_data.m`, that generates 22 records (14 train, 4 val,
4 test) of closed-loop gantry data with per-record logical-coordinate force multisines,
in two selectable band tracks. Do NOT modify the two existing generator scripts (they
stay for diff/regression comparison) and do NOT touch anything in `kamtin-fp-model/`.

## 2. Hard rules (from CLAUDE.md / lessons.md, repeated because they bind this task)

- Preserve the base script's section order, naming style, and local functions. Adapt
  minimally; append new local functions at the bottom. A reviewer must be able to diff
  the new script against the base script and follow every change.
- No operational scaffolding: no env-var hooks, no smoke-test modes, no suppression of
  existing prints/plots. The only new toggle is TRACK (Section 3).
- Every numerical constant in signal-processing code carries `# THEORY:` or
  `# HEURISTIC:` style comments (MATLAB: `% THEORY: ...` / `% HEURISTIC: ...`).
  The labels for each constant are given below; copy them.
- Shape checks are not verification. The coordinate-transform check in Section 6 must
  compare VALUES against an independently computed reference.
- Log the design decisions (Section 12) to `docs/decisions.md` BEFORE writing code.
- The user runs MATLAB, not you. The script must print every validation result so the
  user can inspect; you must give the user a run command and an acceptance checklist
  (Section 11).

## 3. Script header and toggles

Keep the base script's header structure. Toggles at top:

```matlab
USE_MSD = true;    % keep; this script only supports true (assert it)
MA_FRAC = 0.50;    % RESOLVED by user 2026-07-06: 0.50 (ma = 5.05 kg, mh_rigid = 5.05 kg)
TRACK   = 'joint'; % 'joint'        -> multisine band [1, 200] Hz  (joint estimation)
                   % 'augmentation' -> multisine band [130, 180] Hz (augmentation-only)
```

- fa = 150 Hz, zeta_a = 0.05, L0 = 0.10 (unchanged). Note in a comment: observed coupled
  peak ~157 Hz (user's linearized-Simulink FRF); diagnostics window Section 9.
- Physical parameters, controller construction (`ruleOfThumb`, fbw = 100), P matrix,
  hardware limits struct: copy VERBATIM from the base script. lim.diff = 6e-3 m
  (% THEORY: TELICA spec / generate_data_correct_max_theta.m). Do NOT use the
  sin(0.1)*Lb value found in generate_trajectory_data_without_multisine.m; that
  discrepancy is documented in the spec Section 0.1.
- Output: `data/gantry/matlab/records/joint/` or `.../records/augmentation/` per TRACK.

## 4. Record timing (replaces the 1 s period + tiling of the base script)

- fs = 20e3. Record layout: 0.5 s start hold + 10.0 s active window + 0.5 s end hold,
  then pad the end hold to exactly T_rec = 12.0 s total (N_rec = 240000 samples).
  Reuse/adapt `pad_to_periods` with N_period = round(12*fs) so every record is one
  period long.
- Multisine period = T_rec (12 s), so grid spacing f0 = 1/12 Hz.
  % THEORY: period-long multisine, Pintelon & Schoukens 2012 (leakage-free periodic
  % excitation); single period per record because repetition adds zero information in
  % noiseless simulation (see docs/data-generation-design-draft.md Section 3).
- NO tiling. NO shared realizations. Every record draws its own multisine (Section 5).

## 5. Multisine construction (new local function, replaces generate_cached_multisine)

Signature suggestion: `f_logical = make_logical_multisine(fs, T_rec, band, seed)`.

- For each logical channel c in {sym, anti, Y}: set `rng(seed + c)`, draw phases
  uniform in [0, 2pi) for every integer grid line k with k*f0 inside `band`
  (joint: [1, 200] Hz -> k = 12..2400; augmentation: [130, 180] Hz -> k = 1560..2160).
  Flat amplitude spectrum. Build the time signal (iFFT or sum of cosines), normalize to
  unit RMS over the record.
- NO crest-factor optimization (single draw).
  % HEURISTIC-free copy: donor protocol, Hoekstra released script uses
  % n_crest_factor_optim = 1 (docs/excitation-recipe-extraction.md, Donor A).
- Seed formula (deterministic, collision-free, record-unique):
  `seed = 10000*track_id + 100*record_index + channel_index`, track_id: joint = 1,
  augmentation = 2; record_index 1..22 in the order of Section 8. Save the seed in the
  output file.
- The multisine runs over the ENTIRE 12 s record including holds (matches base script
  behavior; standstill records are all-hold by construction).

## 6. Logical-to-stage transform and its mandatory verification

- Nominal mapping: `F_X1 = f_sym + f_anti; F_X2 = f_sym - f_anti; F_Y = f_Y.`
  % HEURISTIC: logical-coordinate excitation design, no literature precedent
  % (docs/excitation-design-literature.md Sections 9.2, 10); analogy: GVT force
  % appropriation.
- MANDATORY VERIFICATION (lessons rule: value-correctness, not shape): before the main
  loop, at Y_op = 0, build sys_cl exactly as the base script does and lsim it with a
  pure f_anti test signal (f_sym = f_Y = 0). Verify: (a) the yaw response
  q(:,1) - q(:,2) is non-trivial, (b) the symmetric response (q(:,1)+q(:,2))/2 is
  smaller than the yaw response by at least a factor 10 in RMS, (c) repeat with pure
  f_sym and verify the reverse. Print both ratios. If the decoupling does not hold,
  STOP and reconcile the mapping with P = [1,1,0; Lb/2,-Lb/2,0; 0,0,1] before
  proceeding (the P convention may require scaling by Lb/2 on the anti channel).
  Report the outcome to the user either way.

## 7. Amplitudes

- A_sym = 40 N RMS, A_Y = 30 N RMS, fixed for ALL records of a track.
  % HEURISTIC: ~5 percent of the 916/656 N RMS limits (D-056 lineage); final level
  % subject to the activation gate (Section 9). Fixed-absolute-amplitude policy copied
  % from Hoekstra released script (amp_scale constant across splits).
- A_anti per record, computed from the yaw budget:
  1. Build sys_cl at the record's Y_op (as base script).
  2. lsim sys_cl with the record's unit-RMS f_anti mapped to stage forces.
  3. yaw_per_unit = max|q(:,1) - q(:,2)|.
  4. `A_anti = 0.8 * (2e-3 / yaw_per_unit)`.
     % HEURISTIC: yaw budget split 2 mm reference + 2 mm multisine + 2 mm margin of the
     % 6 mm limit; 0.8 safety factor. RESOLVED by user 2026-07-06.
  5. Cap A_anti at 40 N RMS; print A_anti for every record.
- Test records use 80 percent of the track amplitudes (% copy: Bouc-Wen tests at
  reduced amplitude, 40 of 50 N).
- Keep the base script's proportional scale-down loop as a safety net, unchanged.

## 8. The 22 records

Reference trajectories: reuse the base script's `make_ref_oscillatory` (with its 0.5 s
half-cosine fade) and `sp1d`/`thirdOrderSetpointETEL` machinery. Two NEW builders are
needed: `make_ref_standstill` (constant [0, 0, Y_op]) and `make_ref_random_p2p`
(Section 8.1). All oscillatory frequencies below give an integer number of cycles in the
10 s active window (base script convention). X_anti reference amplitudes never exceed
0.001 m (|X1 - X2| = 2*X_anti = 2 mm reference budget).

### Train

| id | class | parameters |
|---|---|---|
| T1_hold_Ym30 | standstill | Y_op = -0.30 |
| T2_hold_Ym15 | standstill | Y_op = -0.15 |
| T3_hold_Y000 | standstill | Y_op = 0 |
| T4_hold_Yp15 | standstill | Y_op = +0.15 |
| T5_hold_Yp30 | standstill | Y_op = +0.30 |
| T6_Ysweep_slow | osc | Y_center 0, A_y 0.30, f_y 0.2; no X motion |
| T7_Ysweep_fast | osc | Y_center 0, A_y 0.30, f_y 0.7; no X motion |
| T8_Ysweep_mix | osc | Y_center 0, A_y 0.30, f_y 0.4 + X_sym 0.05 @ 1.1 Hz |
| T9_p2p_30 | p2p_random | ladder 30 percent: vmax_X 0.45, amax_X 6, vmax_Y 0.54, amax_Y 12.6, jerkTime 0.050 |
| T10_p2p_60 | p2p_random | 60 percent: 0.90, 12, 1.08, 25.2, jerkTime 0.035 |
| T11_p2p_100 | p2p_random | 100 percent: 1.50, 20, 1.80, 42, jerkTime 0.025 |
| T12_p2p_anti | p2p_random | 60 percent, X_sym setpoints limited to [-0.05, 0.05], X_anti setpoints in [-0.001, 0.001], jerkTime 0.040 |
| T13_lissajous_a | osc | X_sym 0.08 @ 1.5, Y 0.25 @ 0.4 around Y_center 0, X_anti 0 |
| T14_lissajous_b | osc | X_sym 0.06 @ 1.3, Y 0.30 @ 0.7 around Y_center 0, X_anti 0.001 @ 0.8 |

Setpoint bounds for p2p_random: X_sym in [-0.10, +0.10], Y in [-0.30, +0.30]
(T12 exception above). Ladder percentages relative to base vmax_X 1.5, amax_X 20,
vmax_Y 1.8, amax_Y 42. % HEURISTIC: ladder covers amplitude range (Schoukens & Ljung
% 2019); base values from existing validated records.

### Validation (fresh seeds and realizations, unseen interior Y; Jan's protocol:
### separate generation runs, never data slices)

| id | class | parameters |
|---|---|---|
| V1_hold_Yp10 | standstill | Y_op = +0.10 |
| V2_p2p_Ym22 | p2p_random | 60 percent ladder, Y setpoints in [-0.30, -0.14], new seed |
| V3_Ysweep_off | osc | Y_center +0.10, A_y 0.15, f_y 0.2 |
| V4_lissajous_v | osc | Y_center -0.10, X_sym 0.07 @ 1.4, Y 0.20 @ 0.5 |

### Test (untouched until final evaluation)

| id | class | parameters |
|---|---|---|
| E1_sweep_reso | sweeptest | standstill Y_op = 0; f_Y = linear sinesweep 130 -> 180 Hz over the 10 s active window (5 Hz/s), amplitude 0.8*A_Y; f_sym = f_anti = 0. Sweep phase: phi(t) = 2*pi*(130*t + 2.5*t^2), zero during holds. % THEORY: sweep-through-own-resonance test class (Bouc-Wen); rate within ISO 7626-2 bound (BW^2 ~ 250 Hz^2 scale for BW ~ 15 Hz). MSD acts along Y (user + L0 comment), so f_Y is the sweep channel. |
| E2_hold_Yp22 | standstill | Y_op = +0.22, independent multisine, 0.8x amplitudes |
| E3_p2p_extra | p2p_random | vmax_Y 1.9, amax_Y 45, vmax_X 1.5, amax_X 22, jerkTime 0.030, new seed. Above the training ladder, deliberately below hardware limits (user decision). |
| E4_traj_only | p2p_random, multisine OFF | T11 parameters, new seed, f identically zero |

### 8.1 make_ref_random_p2p algorithm

```
rng(setpoint_seed)   % separate from multisine seed: setpoint_seed = seed + 50
pos = [0, 0, Y_start]  (Y_start = first drawn Y setpoint; start record there)
while active time used < 10 s:
    draw target: X_sym ~ U(bounds), X_anti ~ U(bounds or 0), Y ~ U(bounds)
    per axis, build jerk-limited move via sp1d(dist, vmax, amax, jerkTime, ts)
    pad axes to the longest axis duration (pvpad), append move
    append 0.1 s hold (n_hold_short, base script constant)
truncate to exactly 10 s active, end in hold
```
Prepend/append the 0.5 s holds and pad to 12 s as for all records. Position, velocity,
acceleration, and yaw limits are enforced by the existing `validate_ref` on the result;
if a drawn sequence violates them, redraw with seed+1 (print when this happens).

## 9. Simulation, diagnostics, and saving (per record)

Follow the base script's main loop structure exactly:
1. Controller at frozen Y_op (base code).
2. Build reference (Section 8), validate_ref with lim.diff = 6e-3.
3. Build logical multisine (Section 5), map to stage (Section 6), scale (Section 7).
4. lsim pre-check + proportional scale-down loop (base code).
5. Simulink WITH multisine; Simulink WITHOUT multisine (keep the informativeness
   baseline exactly as in the base script).
6. Informativeness diagnostic: delta_a RMS with/without ratio; residual PSD peak in
   [145, 165] Hz. % window brackets fa = 150 and the coupled ~157 Hz peak (user FRF).
7. Save: all base-script fields PLUS `multisine_seed`, `setpoint_seed`, `track`,
   `band`, `A_ch = [A_sym, A_anti, A_Y]`.
8. Keep the base script's two figures per record (presentation + diagnostics),
   unchanged in structure.

E4 skips steps 3-4 (f = 0) but still runs both simulations (they coincide; keep the
save format identical with zero f_sim). E1 replaces the multisine with the sweep in
step 3; the without-multisine baseline still runs.

## 10. What NOT to do

- No crest-factor optimization, no noise injection, no windowing/filtering of saved
  signals.
- No modification of `generate_oscillatory_multisine_data.m`,
  `generate_trajectory_data_without_multisine.m`, the Simulink models, or anything in
  `kamtin-fp-model/`.
- No downsampling in this script (the Python pipeline consumes 20 kHz and downsamples
  to 4 kHz; unchanged).
- No new toggles beyond TRACK, no env-var hooks, no removal of prints or progress
  output.
- Do not mark the task complete without the user having run both tracks and the
  acceptance checklist below passing (verification lesson).

## 11. Acceptance checklist (user runs, agent reviews printed output)

Run per track: `run('Matlab-scripts/Augmentation/data/generate_track_record_data.m')`
with TRACK set to each value.

1. Transform check (Section 6) printed and passing (>= 10x decoupling both directions).
2. 22 records generated per track, every `r OK` line printed, zero validation failures
   or documented scale-downs.
3. A_anti printed per record; multisine-induced yaw stays <= 2 mm in the lsim
   pre-check (print max yaw per record).
4. delta_a with/without ratio: highest for T1-T5 and E2 (standstill records); report
   the table. If the AUG track's ratios are not clearly above 1 for standstill records,
   flag GATE 2 failure to the user (amplitude too low).
5. Joint-track PSD of u_total shows content across [1, 200] Hz; augmentation-track
   content confined to [130, 180] Hz plus trajectory band.
6. No unexplained spikes in the presentation figures for p2p records (spike history:
   docs/gantry-augmentation-problem-log.md).
7. File count and naming: `<id>.mat` under `records/<track>/`.

## 12. Decisions to log in docs/decisions.md before coding (one entry each)

D-xxx entries: (1) two-track band scheme with TRACK toggle (joint [1,200], augmentation
[130,180], rationale + mid-band mitigation via matched joint records and E4);
(2) logical-coordinate multisine with 6 mm yaw budget split 2/2/2 and 0.8 safety
(HEURISTIC, no precedent, GVT analogy); (3) period = record, no tiling, fresh
realization per record (noiseless justification, donor A function); (4) fixed absolute
amplitudes 40/30 N + computed A_anti (donor A policy + budget); (5) split design:
14/4/4 records, val = separate runs at unseen interior Y (Jan's ECC protocol), test
axes per record E1-E4; (6) MA_FRAC = 0.50 and the 150-parameter vs ~157-coupled-peak
bookkeeping (user FRF). Cite `docs/trajectory-generation-spec-draft.md` and
`docs/excitation-recipe-extraction.md` in each entry.

## 13. Known open items the implementer must NOT silently resolve

- If the Section 6 transform check fails, stop and consult the user (P-convention
  scaling question).
- If any record cannot satisfy limits even after scale-down, report; do not redesign
  trajectories.
- GATE 2 (amplitude sufficiency) is decided by the user from the checklist output, not
  by the implementer.
