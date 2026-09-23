# telica-real

Self-contained setup that takes the augmentation project from simulation to the real Telica
gantry (beam head BHL: rails X1, X2 and Y). Handoff: `tasks/handoffs/2026-09-22-telica-real-gantry.md`.
Verdicts per gate: `REPORT.md`. Decisions (TR-001, ...): `DECISIONS.md`. Every run with its
pre-launch hypothesis: `RUNS.md`.

## Structure
| Path | What |
|-|-|
| `tr_env.py` | import bootstrap: `vendor/` then this folder on `sys.path`, 4 threads, leak guard |
| `data_index.py` | the train / validation / test split by operating point (from the schema doc) |
| `vendor/` | working-tree copies of the framework and pipeline (`VENDORED.md`); edits marked `# TELICA-REAL:` |
| `tools/watchdog.ps1`, `tools/run.sh` | launch wrapper: resource CSV, alert and kill thresholds (TR-002) |
| `params/` | Telica parameters, single source (`telica_params.py`, `PARAMS.md`) (G1) |
| `friction/` | Karnopp friction port + MATLAB references (G2) |
| `controller/` | Telica controller: zpk export, replay, identified logged path, `ControllerBank` (G3) |
| `baseline/` | record access, survey, holding-force test, recovery, closed-loop replay (G4) |
| `learnability/` | residual learnability metric and the derived ANN settings (G5) |
| `pipeline/` | real-data augmentation pipeline, smoke test, server runner (G6) |
| `outputs/<run>/` | run artefacts only (resources.csv, watchdog_summary.txt, metrics, PNGs) |

## Running anything
Every script runs through the watchdog, from any directory:
```
bash telica-real/tools/run.sh <run_name> <script relative to telica-real> [args]
```
It streams output live and writes `telica-real/outputs/<run_name>/resources.csv`.
MATLAB steps use the watchdog directly, e.g.
`powershell -NoProfile -ExecutionPolicy Bypass -File telica-real/tools/watchdog.ps1 -Run "matlab -nojvm -sd telica-real/friction/matlab -batch make_ref_g2" -Out telica-real/outputs/g2_matlab`.

## Gates, in order
| Gate | Command(s) |
|-|-|
| G0 | `tools/dummy_load.py 30`; kill test with `WD_FLAGS="-RamKillGB 100 -MinKillS 8"`; `tools/check_imports.py` |
| G1 | `params/check_params.py` (and `params/telica_params.py` to print the table) |
| G2 | `friction/check_friction.py prep`, MATLAB `make_ref_g2`, `friction/check_friction.py compare` |
| G3 | `controller/analyze_controller.py`, MATLAB `export_zpk_bhl`, `controller/iter0_replay.py`, `controller/mechanism.py`, `controller/identify_logged.py`, `controller/check_bank.py identified` |
| G4 | `baseline/survey.py`, `baseline/holding.py`, `baseline/recover.py` (attempt 1) and `baseline/recover.py --cc-from-holding` (attempt 2), `baseline/g4_replay.py` with variant `rec`, `rec_a2`, `nf` or `70821`, `baseline/diag_limit_cycle.py`, `baseline/g4_verdict.py rec_a2 _a2` |
| G5 | `learnability/make_residuals.py`, `learnability/g5_learnability.py`, `learnability/order_sensitivity.py`; M5 in `pipeline/probe.py` |
| G6 | `pipeline/telica_augment.py --build-norm`; `TELICA_SMOKE=1 MEM_PROBE=1` + `pipeline/telica_augment.py` (max 2 updates); `pipeline/probe.py` (no updates); `pipeline/check_g6.py`; server: `sbatch telica-real/pipeline/runners/server_run.sh` (not submitted) |

## Rules this folder keeps
Data are read only through the vendored loader (`kamtin-data/Data Telica/` is never listed or
copied); nothing outside `telica-real/` is written; real-data runs are float64 at 20 kHz.
