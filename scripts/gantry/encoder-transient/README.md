# encoder-transient

Fix the encoder's initial-state error that dominates the 400-step training objective: diagnosed and
fixed in simulation against the exact state, then checked on Telica against measured positions and
filtered finite-difference velocities. Handoff: `tasks/handoffs/2026-09-23-encoder-transient.md`.
Verdicts: `REPORT.md`. Decisions (ET-001, ...): `DECISIONS.md`. Runs with pre-launch hypotheses: `RUNS.md`.

| Path | What |
|-|-|
| `et_paths.py` | import bootstrap: `vendor/` first, repo paths removed, `check()` asserts the resolution |
| `vendor/` | working-tree copies (`vendor/VENDORED.md`), patched only by `tools/apply_vendor_patches.py` (`# VENDORED:`) |
| `encoder/core.py` | pure algebra: reconstructability map, Chebyshev Y-schedule, `ScheduledReconEncoder` |
| `encoder/recon.py` | baseline and planted (8-state) linearisations, physical maps, grid-scheduled map |
| `encoder/sched_encoder.py` | the G2 variant's parents (`baseline` = production, `planted` = W^b and W^a from the planted model) |
| `encoder/model_map.py` | the same map built from ANY model's own Jacobians (training-time refresh, G5) |
| `diagnose/` | G0 reproduction, G1 attribution and transient, G2 fix, G3 R1 / R1-LS / R2, G5 pre-check |
| `diagnose/cl_common.py` | exact planted model (`PlantedHfn`), closed-loop windows, metrics |
| `telica/g4_telica*.py` | G4 attempts 1 to 3 on telica-real's vendored stack (`telica/vendor_telica/`) |
| `telica/et_hook.py`, `telica/check_hook.py` | the G4 encoder hook for the Telica arm, and its zero-update check |
| `runners/` | G5 server arms (`*.sh`, not submitted), `PREDICTIONS.md` |
| `tools/` | `watchdog.ps1` (copy of telica-real's), `run.sh`, `apply_vendor_patches.py` |
| `outputs/<run>/` | run artefacts, `resources.csv`, `watchdog_summary.txt` |

Every script runs through the watchdog, one at a time, from the repo root:
```
bash scripts/gantry/encoder-transient/tools/run.sh <run_name> <script relative to encoder-transient> [args]
```
