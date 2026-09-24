# Vendored copies (ET-001)

Copied 2026-09-24 from the working tree at HEAD `1ec57c1` (branch `Augmentation`, tree dirty; the
handoff names `a97a577`, HEAD has moved since). Nothing here is imported from the repo: `et_paths.py`
puts this folder first on `sys.path` and `et_paths.check()` asserts the resolution.

| here | source | notes |
|-|-|-|
| `transient/*.py`, `*.sh` | `scripts/gantry/transient/code/` | D-197 harness (truth, R0, R1, R2, figures) |
| `gantry_dynamic/*.py` | `scripts/gantry/gantry_dynamic/` | production pipeline package |
| `model_augmentation/**.py` | `model_augmentation/` | framework (no `__pycache__`) |
| `gantry_interconnect_dynamic.py` | `scripts/gantry/` | entry file; its `CFG` is what `enc_common.build` uses |
| `closed_loop/cl_burnin_sweep.py`, `cl_capability.py`, `cl_pipeline.py`, `cl_controller.py` | `scripts/gantry/closed-loop-controller/` | D-178 planted-model measurement; kept for provenance (ET-002: its planted ANN is for another dataset) |
| `closed_loop/demo_common.py` | `scripts/gantry/drift-demo/` | dependency of the two above |
| `closed_loop/drift_common.py` | `scripts/gantry/diagnostics-drift/` | dependency of `demo_common` |
| `closed_loop/plant.py` | `scripts/gantry/msd-offset/` | the planted ANN's target plant (`MA_FRAC 0.10`, `ZETA_A 0.05`) |
| `common/oracle.py` | `scripts/gantry/common/` | `truth.selfcheck` reference |

## Patches (`tools/apply_vendor_patches.py`, each line marked `# VENDORED:`)
1. `gantry_dynamic/__init__.py`: puts `vendor/` (not the repo root) on `sys.path`.
2. `gantry_dynamic/config.py`: `REPO_ROOT` five levels up, so data paths are unchanged.
3. `gantry_dynamic/data.py`: `TRAIN/VAL/TEST_FILES` reset to the 22-record lists of `c82ff9b` (D-197);
   the D-206 Telica-profile records are not in the frictionless folder.
4. `transient/truth.py`: `vendor/` on `sys.path`; `REPO` for the data path; `CACHE` READ from
   `scripts/gantry/transient/cache/` (62 MB, never rebuilt here).
5. `transient/enc_common.py`: `vendor/` on `sys.path`; outputs to `outputs/vendored_r0/`; `build()`
   pins `mode='augmentation_ma50_b140-230_a6_z03'` (the working-tree entry file names the friction
   folder; the only other entry-file change since `c82ff9b` is `epochs`).

G0 (`diagnose/g0_reproduce.py`, run R001): V1 Y `2.1772e-05 m`, dY `2.7857e-02 m/s`, within 0.16 % of
D-197.

## Telica vendor (ET-006, ET-012): `telica/vendor_telica/`
Copies of telica-real (HEAD `1ec57c1`, working tree): `tr_env.py`, `data_index.py`, `rate.py`, `vendor/`
(telica-real's own vendored framework with the Telica `gantry_ss`), `params/`, `friction/`,
`controller/`, `baseline/{__init__,records}.py` + `recovered_params{,_a2}.json`,
`pipeline/{__init__,real_data,telica_model,telica_augment,memory}.py` + `norm_frozen.json`,
`learnability/ann_settings.json`. Patches (same tool, marked `# VENDORED:` / `# VENDORED-ET:`):
`tr_env.py` repo root and outputs (into `outputs/telica/`); `vendor/gantry_dynamic/config.py` repo root;
`vendor/real_data_verification/telica_loader.py` repo root (for `Telica 1.mat`);
`pipeline/telica_augment.py` smoke update count (`SMOKE_N_ITS`), the G4 encoder hook (`ET_G2_ENCODER`,
`telica/et_hook.py`), the window (`ET_NA_NB`), and the R025 correction of the smoke line.
