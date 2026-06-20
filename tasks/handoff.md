# Session Handoff — Baseline Multisine Data Quality + Downsampling Validation

**Last written**: 2026-06-20

---

## Summary

This session diagnosed and partially fixed fundamental problems with the baseline multisine training data. The existing baseline data is unusable for identification (wrong frequency range, unrealistic forces, loose safety limits). Supervisor feedback adds two more issues: not enough data, and the data lacks oscillatory content.

---

## Problem 1: Baseline multisine bandwidth wrong (FIXED in code, data not regenerated)

**Root cause**: `generate_multisine_data.m` used `f_high = 200 Hz` for both baseline and MSD modes. The baseline system only has dynamics up to ~7 Hz (single oscillatory theta mode at 4.71-5.13 Hz).

**Evidence**: `multisine_frequency_range_baseline.m` (created this session) computes eigendecomposition of the 6-state baseline. Output confirmed by user running in MATLAB:
- One oscillatory mode at 4.71 Hz (Y=0.3) to 5.13 Hz (Y=0.0)
- Bode plots confirm flat/rolling-off response above 7 Hz
- Recommended: f_low=1, f_high=7 Hz

**Fix applied**: Lines 101-107 of `generate_multisine_data.m`:
```matlab
if USE_MSD
    f_high = 200;  % MSD resonance at ~150 Hz
else
    f_high = 7;    % baseline: single oscillatory mode at ~5 Hz
end
```

**Status**: Code changed, baseline data NOT yet regenerated with correct f_high. Existing data in `data/gantry/matlab/multisine/baseline/` was generated with f_high=200 and is useless.

---

## Problem 2: Yaw/diff limit too loose (FIXED in code)

**Root cause**: `lim.diff = sin(0.1) * Lb` = 72.4 mm. Should be 6 mm per `generate_data_correct_max_theta.m`.

**Fix applied**: Line 94: `lim.diff = 6e-3;`

---

## Problem 3: X_anti_amp too large (FIXED in code)

**Fix applied** (matching `generate_data_correct_max_theta.m`):
- T4: 0.030 -> 0.0025
- T7: 0.030 -> 0.0025
- T8: 0.020 -> 0.0018
- E1: 0.015 -> 0.0013

---

## Problem 4: Force cap broken (NEEDS REWORK)

**What we tried**: Added a 40% force cap (`amp_max = min(amp_max, 0.40 * min(traj_rms(traj_rms > 0))`). This failed because `min` picks up near-zero channels. T2, T3, T4 all got `amp_max = 0 N` (no multisine at all).

**Root cause**: Single scalar `amp_max` applied to all 3 channels, but trajectory force varies per channel. `min` picks the weakest channel.

**Agreed approach**: Per-channel amplitude scaling:
```matlab
for j = 1:3
    if traj_rms(j) > threshold  % active channel
        amp_ch(j) = min(amp_hw, 0.40 * traj_rms(j));
    else                         % inactive channel
        amp_ch(j) = 0;           % don't excite unused channels
    end
end
f(:,j) = amp_ch(j) * f_tiled(:,j);  % per-channel scaling
```

**Design decision**: Inactive channels (traj RMS ~ 0) get zero multisine. Rationale: trajectories are designed with intent (T2/T3 isolate X dynamics at fixed Y). Adding Y excitation would perturb the scheduling variable and confound LPV identification. `generate_data_correct_max_theta.m` does the same via explicit `ms_modes` per trajectory.

**`generate_cached_multisine.m` already supports per-channel `amp_rms`** (line 9: "scalar or (1 x n_ch)"). Infrastructure exists, just not used yet.

**Status**: Current force cap code in `generate_multisine_data.m` (lines 244-253) needs to be replaced with per-channel logic. NOT yet implemented.

---

## Problem 5: Supervisor feedback — not enough data

Supervisor said current dataset is insufficient for identification. Only 10 trajectories (T1-T8 train, V1 val, E1 test), each ~2 seconds at 20 kHz.

**To address**: Need more trajectories, and/or longer trajectories, and/or more diverse operating conditions.

---

## Problem 6: Supervisor feedback — need oscillations

Current trajectories are point-to-point moves (ramp up, hold, ramp down). No sustained oscillatory motion. Supervisor explicitly said to add oscillations.

**To address**: Add trajectory profiles with back-and-forth / oscillatory motion (e.g., repeated sinusoidal sweeps, multi-period point-to-point cycles). This would also help with Problem 8.

---

## Problem 7: Too much dead time in trajectories

Each trajectory has 0.5s hold at start + 0.5s hold at end + the motion phase. For short motions, the hold periods dominate. Most of the data samples are "no movement," wasting data budget.

**To address**: Reduce hold times and/or design trajectories with continuous motion (oscillations solve this naturally).

---

## Problem 8: Downsampling validation (BLOCKED on data regeneration)

Script `scripts/gantry/parameter-diagnostics/downsampling_rk4_validation.py` exists and runs, but:
- First run used LTI model (wrong), rewritten to use LPV (`Gantry_State_Block(Y_op=None)`)
- Takes several minutes locally, cancelled by user
- Should be run after baseline data is regenerated with correct f_high
- With f_high=7 Hz, minimum viable sampling rate should be much lower than with 200 Hz data

---

## Scripts created/modified this session

| Script | Action | Status |
|--------|--------|--------|
| `scripts/gantry/encoder/system_dynamics_analysis.m` | Created | Done — Nyquist bounds for baseline + MSD, saves JSON |
| `scripts/gantry/parameter-diagnostics/downsampling_rk4_validation.py` | Created | Done — LPV downsampling sweep, not yet run on correct data |
| `Matlab-scripts/Augmentation/diagnostics/multisine_frequency_range_baseline.m` | Created | Done — baseline freq range analysis, confirmed f_high=7 |
| `Matlab-scripts/Augmentation/data/generate_multisine_data.m` | Modified | Partially done — f_high, lim.diff, X_anti_amp fixed; force cap needs rework |

---

## Execution order for next session

1. **Fix force cap** (Problem 4): Replace single-scalar cap with per-channel logic
2. **Regenerate baseline data**: Run `generate_multisine_data.m` (USE_MSD=false) in MATLAB
3. **Verify plots**: Check force levels and position effects look reasonable
4. **Design oscillatory trajectories** (Problems 5, 6, 7): Plan new trajectory profiles
5. **Run downsampling validation** (Problem 8): On regenerated data, locally or on cluster

---

## Key files

| File | Role |
|------|------|
| `Matlab-scripts/Augmentation/data/generate_multisine_data.m` | Main data generation script (toggle USE_MSD) |
| `Matlab-scripts/parameter-recovery/generate_data_correct_max_theta.m` | Reference for correct limits and per-mode amplitudes |
| `Matlab-scripts/Augmentation/diagnostics/multisine_frequency_range_baseline.m` | Baseline frequency range analysis (f_high=7) |
| `Matlab-scripts/Augmentation/diagnostics/multisine_frequency_range_MSD.m` | MSD frequency range analysis (f_high~165) |
| `Matlab-scripts/Augmentation/diagnostics/generate_cached_multisine.m` | Multisine generation (supports per-channel amp_rms) |
| `scripts/gantry/parameter-diagnostics/downsampling_rk4_validation.py` | Python downsampling sweep |
| `simulations/gantry_subnet/diagnostics/system_dynamics.json` | Nyquist bounds from MATLAB |
| `data/gantry/matlab/multisine/baseline/` | Baseline multisine data (currently useless, needs regeneration) |
