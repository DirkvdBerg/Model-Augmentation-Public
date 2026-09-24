# orthogonal-addition

Handoff `tasks/handoffs/2026-09-23-orthogonal-addition.md`: derive an addition to the gantry
baseline with `J^T Delta* = 0`, put it in the MATLAB truth, regenerate closed-loop data, and test
with the production OBC Jacobian whether the predicted parameter bias `J^+ Delta*` is at the floor.
Outcome in `REPORT.md` (G0 to G3 and G5 PASS, G4 FAIL after three attempts per route with the
mechanism measured).

| file | what |
|-|-|
| `REPORT.md` | gate verdicts G0 to G5, with numbers |
| `DERIVATION.md` | the derivation, every equation with its machine check |
| `DECISIONS.md` | OA-001 to OA-023, logged before implementing; thresholds pre-registered there |
| `RUNS.md` | every run, hypothesis before launch, outcome after |
| `vendor/` | vendored measurement and training path (`VENDORED.md`: sources, md5, edits) |
| `measure/measure_delta.py` | `rho*`, `J^+ Delta*`, floor (env `OA_MODE`, `OBC_REF_STRIDE`, `OA_LEAN`, `OA_TRAIN_ONLY`) |
| `measure/g4_verdict.py` | the OA-008 verdict from a null and an addition JSON |
| `measure/scan_summary.py` | table of several measurement JSONs |
| `design/oa_core.py` | production Jacobian, fast force operator `G[k]`, per-tuple signals, measured `Delta*` |
| `design/columns.py`, `design/addition_io.py` | legitimate slots (OA-006), physical terms; one definition of an addition for MATLAB and Python |
| `design/screen_a.py`, `screen_b.py`, `screen_c.py` | G2 screens of routes A, B, C |
| `design/correct.py` | compensator solve on a dataset (model or measured target) |
| `design/design_u_aware.py`, `design_quadratic.py`, `continuation.py`, `tradeoff.py` | the closed-loop (quadratic) conditions, OA-014 to OA-021 |
| `design/make_absorber.py`, `scale_addition.py` | addition files |
| `derive/d_symbolic.py`, `derive/d_numeric.py` | machine checks D1 to D11, N1 to N6 |
| `matlab/` | `gantrySystemOA.m` (truth EOM), `make_oa_model.m`, `check_oa_noop.m` (Class A), `check_oa_reaches_plant.m` (Class B), `generate_oa_data.m`, `generate_scan.m`, `oa_save_record.m`, the models `gantry_oa.slx`, `gantry_oa_ref.slx` |
| `runners/` | `oa_entry.py` (vendored production entry, `OA_MODE`, `OA_SMOKE`), `run_oa_arm.sh` (server, not submitted) |
| `tools/` | `watchdog.ps1` (copy), `wd.sh` (every run goes through it), `inspect_slx.py`, `show_addition.py` |
| `outputs/` | logs `<run>.log`, watchdog folders `<run>/`, result JSONs, `R-033c_table.md` |

Every script runs through the watchdog, from this folder: `bash tools/wd.sh <run-id> "<command>"`.

## Datasets
Kept (handoff sect. 13): `data/gantry/matlab/trajectory/augmentation_oa_null_b140-230_a6` (null
control) and `augmentation_oa_demo_b140-230_a6` (route B member at `||Delta*||` 1.09). The others
were measured and removed (R-036); regenerate any of them, from this folder, with
`bash tools/wd.sh <id> "matlab -batch \"addpath('matlab'); generate_oa_data('NAME','<name>','ADDITION','<file>')\""`:

| name | addition file | measured in |
|-|-|-|
| `augmentation_oa_b_b140-230_a6` | `design/addition_route_b_R-009b_null_s1.mat` | R-019 |
| `augmentation_oa_b3_b140-230_a6` | `design/addition_b3.mat` | R-023 |
| `augmentation_oa_c1_b140-230_a6` | `design/addition_c1_K.mat` | R-017 |
| `augmentation_oa_cabs_b140-230_a6` | `design/addition_c_abs.mat` | R-014 |
| `augmentation_oa_s<f>` (14 training records) | `design/addition_scan_f<f>.mat`, via `generate_scan` | R-027 |
