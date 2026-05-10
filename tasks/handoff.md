# Session Handoff

_Previous sessions archived to `archive/sessions/`._

**Last written**: 2026-05-10 by Claude (Sonnet 4.6)

---

## Current Goal

Run and verify `generate_identification_experiment.m`. Both scripts are complete.

---

## What Was Completed This Session

### diagnostics_system.m — complete, verified working

Full 3-mode nonparametric pre-analysis. Probes all 3 modes (common, diff, y) across 5 Y
operating points. Results from last run:
- f_low  = 43.0 Hz  (|S|^2 > 0.1 threshold, worst-case across modes and Y)
- f_high = 195.0 Hz (tail |G×S|^2 < 5% of total, worst-case across modes and Y)
- A_max  = [366, 366, 262] N RMS (40% of hardware RMS limits per mode)

f_high uses cumulative energy criterion on |G×S|^2 (= |Q/F_sim|^2 via feedback algebra).
This replaced findpeaks — kb1+kb2 resonance at ~5 Hz is below f_low=43 Hz (controller
suppresses it), so peak search produced no peaks above f_low.

### generate_identification_experiment.m — complete, not yet run

New file at `Matlab-scripts/generate_identification_experiment.m`. Key design:
- Loads f_low, f_high, A_max from step0_outputs.mat
- T1-T8 trajectories with exact parameter values from export_param_recovery_multisine.m
- Simultaneous multi-mode Schroeder multisine injection per trajectory
- Validation: position+velocity on q1, acceleration on r only (not q1 — fs^2 artifact)
- All design choices labeled THEORY/HEURISTIC inline

**Next step: run the script in MATLAB and verify all 8 trajectories execute without errors.**

---

## Open Blockers (carried forward)

- **Sample rate**: D-012 — 16 kHz (main.m) vs 20 kHz (ETEL spec), unresolved
- **Float32 acceptability**: Run training in both dtypes, compare param_table()
- **MIMO decorrelation**: Phase offset insufficient per Pintelon et al. (2011) — declared limitation

---

## Key Files

| What | Where |
|------|-------|
| Theory validation | `docs/theory-validation.md` |
| Design decisions | `docs/decisions.md` |
| Pre-analysis script (complete) | `Matlab-scripts/diagnostics_system.m` |
| Identification experiment script (complete, unrun) | `Matlab-scripts/generate_identification_experiment.m` |
| Pre-analysis outputs | `Matlab-output/step0_outputs.mat` |
| Old multisine script (reference only) | `Matlab-scripts/export_param_recovery_multisine.m` |
| Lessons (read before anything) | `tasks/lessons.md` |
