# closed-loop-noise

Estimate the Telica encoder-node source noise from the closed-loop ILC logs, inject it inside the
simulated loop of the data generator, and check that the sensitivity math closes on both loops.
Handoff: `tasks/handoffs/2026-09-23-closed-loop-noise.md`. Verdicts: `REPORT.md`. Decisions
(CN-001, ...): `DECISIONS.md`. Every run with its hypothesis: `RUNS.md`. The 09-22 literature
synthesis: `LITERATURE.md`.

## Structure
| Path | What |
|-|-|
| `cn_env.py` | import bootstrap (vendored telica-real first on sys.path, leak guard, `out_dir`) |
| `vendor/` | copies of telica-real (controller, plant, simulator, loader with `pre_motion_ms`); `VENDORED.md` |
| `tools/` | `watchdog.ps1` (from telica-real), `run.sh` (launcher), `extract_literature.py`, `check_g0.py` |
| `measure/` | `spec.py` (records, windows, Welch matrices, bands, CIs), `g1_spectra.py` (G1) |
| `sensitivity/` | `loop.py` (linear Telica loop, three standstill readings), `g2_sensitivity.py` (G2) + diagnostics |
| `model/` | `noise_model.py` (Yule-Walker AR/VAR shaping filter), `g3_source.py` (G3), `make_noise_files.py` (G5 noise) |
| `closure/` | `g4_closure.py` + `g4_expected_welch.py` (G4), `g5_check.py` (G5 c/d), `export_recipe_inputs.py`, `g6_numbers.py` (G6) |
| `matlab/recipe_sensitivity.m` | the MATLAB recipe: `feedback`, `bode`, `covar`, VAR noise as `ss` (G6) |
| `matlab/gen/` | vendored generator (pre-D-209 friction ODE, gtd_* helpers, drivers as functions), the encoder-noise model `gantry_cn_noise_2025a.slx` and its builder; `VENDORED_MATLAB.md` |
| `matlab/noise_ss/` | per-record encoder noise v for the twin (from H_v, seeded by record id) |
| `outputs/<run>/` | run artefacts (resources.csv, watchdog_summary.txt, json, npz, png) |

## Running
Python: `bash scripts/gantry/closed-loop-noise/tools/run.sh <run> <script relative to this folder> [args]`.
MATLAB: `CN_MATLAB=matlab/gen WD_FLAGS="-RamKillGB 1.2 -RamAlertGB 2.0" bash .../tools/run.sh <run> <function>`
(CN-010: Simulink needs ~3 GB). Order: G0 `tools/check_g0.py`; G1 `measure/g1_spectra.py`; G2
`sensitivity/g2_sensitivity.py`; G3 `model/g3_source.py`; G4 `closure/g4_closure.py tanh 16.5 2`
then `closure/g4_expected_welch.py tanh`; G5 `model/make_noise_files.py tanh 1.0 ss`, MATLAB
`g5a_noiseoff`, `g5b_twin`, `g5c_ref`, `closure/export_recipe_inputs.py`, MATLAB (from `matlab/`)
`recipe_sensitivity`, `closure/g5_check.py`; G6 `closure/g6_numbers.py`.

## Rules this folder keeps
Telica logs are read only through the vendored loader, summaries only; nothing outside this folder
is written except the one new dataset folder
`data/gantry/matlab/trajectory/augmentation_ma50_z03_b140-230_a6all_telica_coulomb_noise_ss/`;
no training, no commits.
