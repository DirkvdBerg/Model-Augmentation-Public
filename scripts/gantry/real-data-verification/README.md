# Real-Data Verification — Telica Gantry

This folder contains scripts and results for verifying whether the FP (first-principles)
gantry model structure is compatible with real Telica ILC measurement data.

---

## Goal

Determine whether the FP model can fit real Telica data at a single operating point
before committing to full LPV parameter estimation across all scheduling positions.
The primary tool is Simulation Error Method (SEM) using `train_param_recovery.py`.

---

## System: Telica Dual-Axis Gantry (ASMPT)

Two beam heads (Left = BHL, Right = BHR), each with two X-axes and one Y-axis.
Six axes total. The FP model covers one beam head (BHL or BHR) with 3 axes:
GTRX1 (X1), GTRX2 (X2), GTRY (Y).

---

## Data Location

```
kamtin-data/Data Telica/06 40 mm XL 80 mm YL/
    train/
        xpos_-60_ypos-40/      iter0.log ... iter8.log, iterETEL.log
        xpos_-60_ypos120/      ...
        xpos_-60_ypos-120/
        xpos_-60_ypos-200/
        xpos_-135_ypos40/
        xpos_-135_ypos-40/
        xpos_-135_ypos-200/
        xpos_-210_ypos40/
        xpos_-210_ypos120/
        xpos_-210_ypos-120/
        xpos_-210_ypos200/
    test/
        xpos_-60_ypos40/       iter0.log ... iter8.log, iterETEL.log, iterTEST.log
        xpos_-135_ypos-120/
    validation/
        xpos_-135_ypos120/
        xpos_-210_ypos-40/
```

The `xpos_` / `ypos_` folder names encode the LPV scheduling variable (stage position
in mm) at which that dataset was collected. Start with a single train folder.

---

## File Format: `iter*.log`

Tab-separated text. First row is the column header.

| File        | Content |
|-------------|---------|
| `iter0`     | Feedback only, feedforward off |
| `iter1`-`iterN` | ILC iterations (feedforward improves each step) |
| `iterETEL`  | ETEL default feedforward, no ILC |
| `iterTEST`  | Final ML-generated feedforward — target output |

### Header format

```
TimeStamp   BHL_GTRX1.M0:0.0   BHL_GTRX1.M2:0.0   BHL_GTRX1.MF230:0.0   BHL_GTRX1.MF30:0.0   ...
```

The MATLAB loader strips `:0.x` and replaces `.` with `_`, giving field names like
`BHL_GTRX1_M0`, `BHL_GTRX1_M2`, etc.

---

## Signal Definitions

Source: `literature/gantry/accuret-monitoring-signals.md` (ETEL AccurET manual, pp. 139-141, 398).

| Signal  | Name | Description | Unit (raw) |
|---------|------|-------------|------------|
| `M0`    | Theoretical position | Position setpoint `Xc` | dpi |
| `M2`    | Position control error | Tracking error `Xe = M0 - M1` | dpi |
| `MF230` | Feedback current | Controller-side current after advanced filters, before feedforward/cogging additions | ci |
| `MF30`  | Total current command | Full current command after feedforward + cogging + offset + KF60 saturation | ci |

### Current-command chain (from AccurET monitoring diagram, p. 139)

```
MF230  (feedback, after advanced filters)
+ MF231  (feedforward — NOT logged)
+ MF250  (cogging compensation — NOT logged)
+ torque offset  (NOT logged)
= MF233  (before saturation)
--> sat_KF60 --> MF30  (total current command, after saturation)
```

So:
- `MF30` is the **total plant input** — feedback + feedforward + cogging, after limiting.
- `MF230` is only the **feedback branch**.
- `MF30 - MF230` approximates feedforward only when cogging is off and KF60 is not saturating.

---

## Unit Conversions

| Signal | Raw unit | Conversion | Result |
|--------|----------|------------|--------|
| `M0`, `M2` | dpi (drive position increments) | x 1e-6 (MATLAB loader convention: treats as µm) | metres |
| `MF30`, `MF230` | ci (current increments) | / 481.882 | Amperes |
| Current to force | A | x K_f [N/A] | Newtons |

**K_f (motor force constant)** is not in the logged data. It lives in
`machine-parameter-files/output/Telica.mat` (external, not in this repo).
Source for 481.882: `kamtin-data/runFDILCAllHostSwLog.m` line 439 comment and line 324.

### Actual stage position

```
M1 (real position) = M0 - M2        [dpi -> m after conversion]
```

Use `M1` as the measured stage position for the FP model output.

---

## Preprocessing Pipeline (raw log -> model-compatible .mat)

Goal: produce a `.mat` file with the same fields that `precompute._load_trajectory()` expects.

| Field in .mat | Shape  | Content | Source in log |
|---------------|--------|---------|---------------|
| `u_q1`        | (T, 3) | Stage-coordinate force input [A or N] | `[BHL_GTRX1_MF30, BHL_GTRX2_MF30, BHL_GTRY_MF30] / 481.882` |
| `q1`          | (T, 3) | Stage positions [m] | `[BHL_GTRX1_M0 - BHL_GTRX1_M2, BHL_GTRX2_M0 - BHL_GTRX2_M2, BHL_GTRY_M0 - BHL_GTRY_M2]` |
| `fs`          | scalar | Sampling rate [Hz] | 20000 (after MATLAB resampling) |

Notes:
- Use BHL (left beam head) for all three axes. BHR is a separate body not modelled.
- `f_sim` field can be omitted (absent = zero, handled by `precompute._load_trajectory()`).
- If K_f is unknown, use input in Amperes. The FP model is linear in force, so K_f
  folds into all mass/stiffness/damping parameters as a uniform scale factor. Structural
  validity is unaffected; recovered parameters will have units scaled by 1/K_f.

---

## Validation Approach

Method: **Simulation Error Method (SEM)** using `train_param_recovery.py`.

The FP model is simulated open-loop from the measured input `u`. Parameters are
optimized (Adam) to minimize simulation error `||y_sim - q1||`. If the structure is
compatible, Adam converges to low NRMSE. This is the correct method for structural
validation: SEM forces the model to propagate autonomously, so a wrong structure
cannot hide behind state corrections (unlike PEM).

Decision thresholds (Schoukens & Ljung 2011; Paduart et al. 2018):
- NRMSE < 15% on held-out data: structure compatible
- NRMSE > 30% after training: structural mismatch or data/force-signal problem
- 15-30%: ambiguous, inspect residual spectrum for structured peaks

### Which iteration to use first

`iterETEL` (ETEL default feedforward): nonzero excitation, well-characterized baseline.
Good first candidate. Avoid `iter0` (feedback only, no feedforward excitation) for initial
structural check.

### Starting point for parameters

Use the detuned initial parameters already in `ParameterizedLFRBlock` — they are
physically motivated and within the right order of magnitude for a gantry system.
If Adam fails to converge, consider multi-start from parameter sets sampled around
the nominal before concluding structural mismatch.

---

## Key Source Files

| File | Role |
|------|------|
| `kamtin-data/runFDILCAllHostSwLog.m` | MATLAB loading pipeline, unit conversions |
| `literature/gantry/accuret-monitoring-signals.md` | Signal definitions, current-command chain |
| `literature/gantry/AccurET-Oper&Soft-VerV.pdf` | Source manual (pp. 139-141, 398) |
| `docs/kamtin-telica-schema.md` | Folder structure and column naming |
| `lpv_lfr_baseline/scripts/train_param_recovery.py` | SEM training loop |
| `lpv_lfr_baseline/scripts/precompute.py` | Data loading — defines required .mat fields |
