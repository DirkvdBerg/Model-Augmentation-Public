# Vendored code (TR-001)

Copied 2026-09-22 from the WORKING TREE of branch `Augmentation`, HEAD `9372b09`
(the handoff names `a97a577`, its parent; `9372b09` only changes comments per its message).
The tree was dirty ("working tree, dirty"): these copies may differ from any commit.
Edits made here are marked `# TELICA-REAL:` inline (`grep -rn "TELICA-REAL" telica-real/vendor`).

| Vendored path | Source path | Commit | State | Date |
|-|-|-|-|-|
| `model_augmentation/` (whole package, no `__pycache__`) | `model_augmentation/` | 9372b09 | working tree, dirty | 2026-09-22 |
| `gantry_dynamic/*.py` | `scripts/gantry/gantry_dynamic/*.py` | 9372b09 | working tree, dirty | 2026-09-22 |
| `gantry_interconnect_dynamic.py` | `scripts/gantry/gantry_interconnect_dynamic.py` | 9372b09 | working tree, dirty | 2026-09-22 |
| `common/*.py` | `scripts/gantry/common/*.py` | 9372b09 | working tree, dirty | 2026-09-22 |
| `real_data_verification/telica_loader.py` | `scripts/gantry/real-data-verification/telica_loader.py` | 9372b09 | working tree, dirty | 2026-09-22 |
| `real_data_verification/telica_controller.py` | `scripts/gantry/real-data-verification/telica_controller.py` | 9372b09 | working tree, dirty | 2026-09-22 |
| `real_data_verification/telica_residual_spectrum.py` | `scripts/gantry/real-data-verification/telica_residual_spectrum.py` | 9372b09 | working tree, dirty | 2026-09-22 |
| `real_data_verification/plant_coulomb_ref.py` | `scripts/gantry/coulomb-offset/plant_coulomb.py` (renamed: reference for the friction port, pre-D-209 band) | 9372b09 | working tree, dirty | 2026-09-22 |
| `real_data_verification/cl_headroom.py`, `cl_residual_spectrum.py` | `scripts/gantry/closed-loop-controller/` (reference for G5 only, not imported) | 9372b09 | working tree, dirty | 2026-09-22 |

Not vendored, and why:
- `lpv_lfr_baseline/`: nothing on the augmentation path imports it (checked with grep on
  `model_augmentation/`, `gantry_dynamic/`, `common/`). The baseline used here is
  `Gantry_State_Block` from the vendored `model_augmentation`, which is the realization the
  augmentation trains against (D-138 note). `coulomb_lfr.py` / `lfr_param_block_coulomb.py` belong
  to the lpv_lfr_baseline thread and are superseded here by `friction/`.
- `deepSI`: an installed package of the `GraduationProject` env (site-packages), not repo code.

Edits log (every `# TELICA-REAL:` marker):
- `gantry_dynamic/__init__.py`: sys.path root is `telica-real/vendor`, not the repo root.
- `model_augmentation/systems/gantry_ss.py`: the Garcia/main.m constants replaced by imports from
  `params/telica_params.py` (G1, TR-003).
- `gantry_dynamic/controller.py`: the gtd_config constants replaced by imports from `params/`
  (G1); the ruleOfThumb bank itself is not used on real data (`controller/telica_bank.py`).
- `real_data_verification/telica_loader.py`: force scale from `params/` with Kt / fs consistency
  asserts (G1, TR-004); new `load_telica_log_full(path, engine)` (untrimmed, numpy, G3/G4).
- `model_augmentation/fit_systems/closed_loop.py`: `ControllerBank(compute_dtype=...)` keeps the
  controller in float64 inside a float32 rollout (TR-012; a no-op in the float64 pipeline).
- `gantry_dynamic/training.py`: skips the absorber-GT diagnostic `aug_state_r2` when the data carry
  no ground truth (G6 item 5).
