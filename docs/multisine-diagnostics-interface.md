# Multisine Generation — Theory Basis and Diagnostics Interface

**Purpose:** Theory foundation for multisine design in `export_param_recovery_inject_ref.m`, and the correct interface between `experiment_diagnostics.py` outputs and multisine design parameters.  
**Companion documents:**  
- `docs/diagnostics-theory-basis.md` — theory for the Python diagnostics  
- `docs/trajectory-design-param-recovery.md` — full T1–T8 design rationale  
- `docs/experiment-design-closed-loop.md` — closed-loop ID session notes  
- `literature/experiment-design/System-identification/sysid-experiment-design-notes.md` — lecture synthesis

---

## The Fundamental Problem with the Current Workflow

**Current (backwards):**
```
MATLAB generates multisine with ad-hoc frequency bands
    → closed-loop simulation
    → Python diagnostics discover f_99 problem (195 Hz in T4)
    → apply post-hoc cap as workaround
```

**Correct (theory-first):**
```
Python diagnostics (physics only, no data needed)
    → f_osc_min, fs_new, model band
    → MATLAB reads these values
    → designs multisine within model band
    → closed-loop simulation
    → Python diagnostics verify (no surprises)
```

The diagnostics can run **before** any trajectory data exists, because Diagnostics 2 and 3 use only the physics model. Diagnostic 1 (FFT/f_99) is a post-hoc signal check — it should verify, not drive.

---

## Interface: What Diagnostics Output Feeds Multisine Design

| Diagnostic output | Source | Drives multisine parameter |
|---|---|---|
| `f_osc_min` [Hz] | Diag 2: slowest oscillatory pole | Upper frequency of ALL channels: `f_high ≤ 10 × f_osc_min` |
| `fs_new` [Hz] | Diag 2: `10 × f_osc_min` rule | Multisine design sampling rate |
| `tau_max` [s] | Diag 2: dominant time constant | Minimum trajectory duration: `T ≥ 10 × tau_max` |
| `segment_len` [samples] | Diag 2: max(period, 10×τ, 10×n_params) | Minimum usable data length per trajectory |
| `f_osc_min` [Hz] | Diag 2 | Frequency of interest for resonance-targeted shaping |

At current system parameters:
- `f_osc_min ≈ 4.94 Hz`
- `fs_new = 1000 Hz`
- `tau_max ≈ 1.57 s` → minimum trajectory `T ≥ 15.7 s`
- Model band upper bound: `10 × 4.94 = 49.4 Hz`

---

## Multisine Design Parameters — THEORY vs HEURISTIC

### Frequency bounds

**Upper bound — f_high**

```matlab
% THEORY: Lecture 9 slides 10-12 (5SMB0) — "10ωb ≤ ωs ≤ 30ωb" applied to model band
% THEORY: Pintelon & Schoukens (2001/2012) — excite model band, not excitation band
f_high_max = 10 * f_osc_min;   % ≈ 49 Hz at current parameters
```

All channels: `f_high ≤ 49 Hz`. Currently:
- Common X (T2, T3): `f_high = 100 Hz` → **above model band, drives 2000 Hz sampling** → change to 49 Hz
- Diff X (T4, T7, T8): `f_high = 20 Hz` → within model band ✓
- Y (T1, T6): `f_high = 20 Hz` → within model band ✓

**Lower bound — f_low**

```matlab
% HEURISTIC: f_low = 1 Hz — engineering choice to include slow drift/rigid-body modes
% No specific lecture source for f_low = 1 Hz
f_low = 1;   % Hz
```

Acceptable heuristic — just below f_osc_min (4.94 Hz) to cover rigid-body and slow modes.

**Resonance-targeted shaping for (F1−F2)**

```matlab
% THEORY: Lecture 8 slides 50-56 (5SMB0) — "increase input power near uncertain/resonant regions"
% THEORY: trajectory-design-param-recovery.md Section 6.8 — stiffness dominant below resonance,
%         inertia dominant above resonance, resonance peak most sensitive for separation
% HEURISTIC: specific shaping factors (0.5×, 1×, 0.25×) — no exact lecture source
```

The principle of concentrating power near resonance is theory (Lecture 8). The specific amplitude ratios are heuristic.

---

### Phase design — Schroeder

```matlab
% THEORY: Lecture 9 slides 22-24 (5SMB0) — Schroeder phases minimise crest factor
% THEORY: phi_k = -k(k-1)*pi/F — directly from lecture formula
phi = -(idx .* (idx - 1)) * pi / F;
```

Crest factor ≈ 1.58 vs ~3.5 for random phases. **This is fully theory-supported.**

---

### Odd-only harmonics

```matlab
% THEORY: Lecture 13 slides 28-40, 59-61 (5SMB0)
% "Odd-only lines help detect/separate even and odd nonlinear distortion"
% Even harmonics (2f0, 4f0, ...) remain empty → even-order nonlinear distortion detectable
k_odd = k_all(mod(k_all, 2) == 1);
```

**Fully theory-supported.** Gantry has potential quadratic terms (Coriolis: Ẋ·Ẏ) — odd-only multisine lets you detect whether these are significant.

---

### Frequency resolution — f0 = 1 Hz

```matlab
% THEORY: Lecture 3 (5SMB0) — "Only then x(t) is exactly periodic: spectrum is exact"
%         Frequencies must lie on DFT bins: f_k = k * (fs/N)
% HEURISTIC: f0 = 1 Hz (1-second period) — engineering choice, not a lecture-specified value
f0 = 1;         % Hz
N_period = fs;  % samples per period at fs
```

The principle (integer DFT bins) is theory. The specific choice f0=1 Hz is a heuristic. It is a reasonable choice: 1 Hz gives adequate spectral resolution, 1-second periods divide cleanly into most trajectory durations.

---

### Minimum number of frequency lines — PE guard

```matlab
% THEORY: Lecture 6 slides 17-21 (5SMB0) — PE order = 2F for multisine with F lines
%         For 14 parameters: 2F ≥ 14 → F ≥ 7
if F < 7
    warning('PE order 2F < 14 — may not excite all parameter directions');
end
```

**Fully theory-supported.** Note: PE is a necessary but not sufficient condition for identifiability. FIM/Jacobian rank check (Section 9 of trajectory-design doc) is the sufficient condition.

---

### Injection point — reference vs force

```matlab
% THEORY: Lecture 11 (5SMB0) — closed-loop identification requires external excitation
%         independent of feedback/noise
% THEORY: T(jω) = GC/(1+GC) ≈ 1 below bandwidth → reference injection not attenuated
% THEORY: S(jω) = 1/(1+GC) ≪ 1 below bandwidth → force injection severely attenuated
% CONCLUSION: reference injection (r_ms) is the correct closed-loop approach
r_total = r_traj + r_ms;   % reference injection
% f = 0 always — no force injection
```

**Fully theory-supported.** This is also validated in `sysid-experiment-design-notes.md` Section 3.

---

### Amplitude — kinematics constraint

```matlab
% THEORY: Lecture 9 slides 22-24 (5SMB0) — amplitude must stay below actuator/response limits
% HEURISTIC: accel-limited formula F_equiv = M_eff * (2π·f)² * A
%            — derived from F=ma, not a direct lecture formula
% THEORY: amplitude sweep + validate_response — lecture principle of sweeping to find max safe level
```

The sweep procedure is theory-backed (Lecture 9 sweep recommendation). The specific kinematic formula is a heuristic derived from Newton's second law — acceptable but should be labelled.

---

### Amplitude ratio check: rms(r_ms) / rms(r_traj) < 0.3

```matlab
% HEURISTIC: ratio < 0.3 ensures multisine is a perturbation, not the dominant signal
% Source: sysid-experiment-design-notes.md Section 2 — "rule of thumb from lectures"
% No direct lecture slide for 0.3; it is a practical guard
```

Heuristic. Acceptable engineering guard.

---

## What is Currently Wrong — Summary

| Parameter | Current value | Theory says | Fix |
|---|---|---|---|
| `f_high` for common X (T2, T3) | 100 Hz | ≤ 49 Hz (10 × f_osc_min) | Change to 49 Hz |
| `f_high` for Y (T1, T6) | 20 Hz | ≤ 49 Hz | Already within bound — acceptable |
| Source of `f_high` | Ad-hoc band assignment | `f_osc_min` from Diagnostic 2 | Read from diagnostics output |
| Frequency band for broadband channels | Fixed 1-100 Hz | Tied to physics via f_osc_min | Compute at design time |

The 100 Hz upper bound on common X is the source of the T2 `f_99 = 115 Hz` finding (base dataset). It also explains why a broadband multisine forces `fs_new = 2000 Hz` — the excitation content exists above the model band.

---

## Correct Design Procedure for Next Experiment Generation

```
Step 1: Run Python diagnostics (physics only — no data needed)
    conda run -n GraduationProject python -m lpv_lfr_baseline.scripts.experiment_diagnostics
    Reads: f_osc_min, fs_new, tau_max, segment_len
    Writes: simulations/.../diagnostics/diagnostics_report.txt

Step 2: Export diagnostics parameters to JSON for MATLAB
    {
        "f_osc_min_hz": 4.94,
        "fs_new_hz": 1000,
        "model_band_upper_hz": 49.4,
        "tau_max_s": 1.572,
        "min_trajectory_duration_s": 15.72
    }

Step 3: MATLAB reads JSON, designs multisine
    f_high = model_band_upper_hz = 49.4 Hz  (not hard-coded 100 Hz)
    T_min  = min_trajectory_duration_s       (not hard-coded duration)
    All channels: f_high ≤ 49 Hz

Step 4: Closed-loop simulation → data

Step 5: Run Python diagnostics again on data
    Verify: f_99 ≤ model_band_upper_hz for all channels and trajectories
    If f_99 > model_band: WARN — signal content above model band (see González 2024)
    fs_new should be 1000 Hz — if not, something is wrong with Step 3
```

---

## What the Literature Does NOT Support

| Current implementation | Status |
|---|---|
| `f_high = 100 Hz` for X channels | **HEURISTIC** — above model band, not theory-justified |
| `f_99` as driver of `fs_new` | **HEURISTIC** — f_99 is not defined in any cited source |
| `_FS_RULE_FACTOR = 8` | **WRONG** — lecture says 10× lower bound |
| `[::D]` stride for decimation | **WRONG** — no anti-aliasing filter applied |
| `_F99_PHYSICAL_CAP_FACTOR = 10` | **HEURISTIC mixing** — applies 10× rule to wrong variable (f_99 instead of f_BW) → do not implement |

---

## References

| Claim | Source |
|---|---|
| fs = 10 × system bandwidth | Lecture 9 slides 10-12 (5SMB0) |
| Model band vs excitation band | Pintelon & Schoukens (2001/2012) |
| Consistency with out-of-band aliasing | González et al., arXiv:2410.19629 / IEEE TAC 2024 |
| Schroeder phases, crest factor | Lecture 9 slides 22-24 (5SMB0) |
| Odd-only harmonics, nonlinear distortion | Lecture 13 slides 28-40, 59-61 (5SMB0) |
| Reference injection T≈1, force injection S≪1 | Lecture 11 (5SMB0) |
| Power near resonance for uncertain parameters | Lecture 8 slides 50-56 (5SMB0) |
| Amplitude sweep to max safe level | Lecture 9 slides 22-24 (5SMB0) |
| Integer DFT bins, leakage-free | Lecture 3 (5SMB0) |
| PE order = 2F, F ≥ 7 guard | Lecture 6 slides 17-21 (5SMB0) |
