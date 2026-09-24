# joint-estimation-horizon

How to estimate the slow physical parameters jointly with the augmentation when the training
window (0.1 s, sized for the 150 Hz absorber) is 3x to 13x shorter than the slow time constants.
Handoff: `tasks/handoffs/2026-09-23-joint-estimation-horizon.md`. Verdicts: `REPORT.md`.
Decisions and pre-registered thresholds: `DECISIONS.md` (JH-xxx). Every run: `RUNS.md`.

| Path | Content |
|-|-|
| `jh_env.py` | import bootstrap: `vendor/` first on sys.path, `check_no_leak()` (JH-001) |
| `jh_model.py` | production build at chosen parameters, ANN-free closed-loop rollout, forward-mode sensitivities |
| `vendor/` | vendored framework and pipeline (`VENDORED.md`, only `VENDOR-PATCH (JH-001)` lines differ) |
| `tools/` | `watchdog.ps1` (telica-real, unchanged), `run.sh`, `wait_run.sh`, `check_vendor.py`, `g0_epoch0.py` |
| `info/` | G1: Cramer-Rao bound per combination against window length |
| `staged/` | G2: baseline-only fits against `J+ Delta*` |
| `runners/` | G3: server arm and control (not submitted) |
| `outputs/<run>/` | `run.log`, `resources.csv`, `watchdog_summary.txt`, results |

Run anything from the repo root, always through the watchdog:
```
bash scripts/gantry/joint-estimation-horizon/tools/run.sh <run> <script relative to this folder> [args]
bash scripts/gantry/joint-estimation-horizon/tools/wait_run.sh <run> <script> [args]   # waits for RAM
```
Live log: `outputs/<run>/run.log`.
