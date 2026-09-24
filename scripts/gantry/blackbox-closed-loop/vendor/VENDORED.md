# Vendored sources (BB-002)

Copied 2026-09-24 from the working tree at commit `1ec57c1` (branch
`Augmentation`, tree dirty; the handoff names `a97a577`, which is an ancestor). Status column:
`clean` = identical to HEAD, `M` = modified in the working tree, `??` = untracked; the copy is
of the WORKING TREE file in every case. sha256 (first 12 hex) is of the SOURCE at copy time.
Only lines marked `VENDOR-PATCH (BB-002)` differ from the source (path resolution);
`adapter/bbcl.py::check_vendor()` re-verifies that at run time (G0 criterion 3).

| Source | sha256 | Status | Copy |
|-|-|-|-|
| `scripts/gantry/gantry_dynamic/__init__.py` | 72d4cc64cf1d | clean | `vendor/gantry_dynamic/__init__.py` |
| `scripts/gantry/gantry_dynamic/baselines.py` | 55c262288bc0 | clean | `vendor/gantry_dynamic/baselines.py` |
| `scripts/gantry/gantry_dynamic/config.py` | cc52d1448c8b | clean | `vendor/gantry_dynamic/config.py` |
| `scripts/gantry/gantry_dynamic/controller.py` | 643d5e1872f7 | clean | `vendor/gantry_dynamic/controller.py` |
| `scripts/gantry/gantry_dynamic/data.py` | 0a4b6f4057b0 | clean | `vendor/gantry_dynamic/data.py` |
| `scripts/gantry/gantry_dynamic/diagnostics.py` | 48cba98e1f51 | clean | `vendor/gantry_dynamic/diagnostics.py` |
| `scripts/gantry/gantry_dynamic/evaluation.py` | f18027018c62 | clean | `vendor/gantry_dynamic/evaluation.py` |
| `scripts/gantry/gantry_dynamic/model.py` | 5119c21172c6 | clean | `vendor/gantry_dynamic/model.py` |
| `scripts/gantry/gantry_dynamic/obc_diagnostics.py` | c2fa74df98d2 | clean | `vendor/gantry_dynamic/obc_diagnostics.py` |
| `scripts/gantry/gantry_dynamic/obc_gantry.py` | ffe22fe019b5 | clean | `vendor/gantry_dynamic/obc_gantry.py` |
| `scripts/gantry/gantry_dynamic/training.py` | a255866cce17 | clean | `vendor/gantry_dynamic/training.py` |
| `model_augmentation/__init__.py` | 3ccd35009717 | clean | `vendor/model_augmentation/__init__.py` |
| `model_augmentation/fit_systems/__init__.py` | 07a18a4e3720 | clean | `vendor/model_augmentation/fit_systems/__init__.py` |
| `model_augmentation/fit_systems/augmented_dynamics.py` | 2bb9aaf47830 | ?? | `vendor/model_augmentation/fit_systems/augmented_dynamics.py` |
| `model_augmentation/fit_systems/blocks.py` | 9e3e3294e9c4 | clean | `vendor/model_augmentation/fit_systems/blocks.py` |
| `model_augmentation/fit_systems/closed_loop.py` | 00e80d9c645a | clean | `vendor/model_augmentation/fit_systems/closed_loop.py` |
| `model_augmentation/fit_systems/continuous_augmentation.py` | 066b70d79ced | ?? | `vendor/model_augmentation/fit_systems/continuous_augmentation.py` |
| `model_augmentation/fit_systems/gyorok_orth_projection.py` | 4399dc665264 | clean | `vendor/model_augmentation/fit_systems/gyorok_orth_projection.py` |
| `model_augmentation/fit_systems/interconnect.py` | eb032c450428 | clean | `vendor/model_augmentation/fit_systems/interconnect.py` |
| `model_augmentation/fit_systems/lbfgs_polish.py` | 31a7436e3251 | clean | `vendor/model_augmentation/fit_systems/lbfgs_polish.py` |
| `model_augmentation/fit_systems/multiple_shooting.py` | e218ab0665e2 | clean | `vendor/model_augmentation/fit_systems/multiple_shooting.py` |
| `model_augmentation/fit_systems/obc.py` | b6b12b0a7736 | clean | `vendor/model_augmentation/fit_systems/obc.py` |
| `model_augmentation/fit_systems/obc_projection.py` | 27035f6b8fba | ?? | `vendor/model_augmentation/fit_systems/obc_projection.py` |
| `model_augmentation/fit_systems/orth_projection.py` | 92cfb224c759 | clean | `vendor/model_augmentation/fit_systems/orth_projection.py` |
| `model_augmentation/fit_systems/pre_encoder.py` | c396b992aa4b | clean | `vendor/model_augmentation/fit_systems/pre_encoder.py` |
| `model_augmentation/fit_systems/white_box_models.py` | 561c053bbd53 | clean | `vendor/model_augmentation/fit_systems/white_box_models.py` |
| `model_augmentation/systems/__init__.py` | 314e58500342 | clean | `vendor/model_augmentation/systems/__init__.py` |
| `model_augmentation/systems/cascaded_tanks.py` | 66f60076d15a | clean | `vendor/model_augmentation/systems/cascaded_tanks.py` |
| `model_augmentation/systems/gantry_linearization.py` | fdc419ebad7a | M | `vendor/model_augmentation/systems/gantry_linearization.py` |
| `model_augmentation/systems/gantry_ss.py` | 40bb3f3c0d93 | clean | `vendor/model_augmentation/systems/gantry_ss.py` |
| `model_augmentation/systems/mass_spring_damper.py` | 25f0f4b61bda | clean | `vendor/model_augmentation/systems/mass_spring_damper.py` |
| `model_augmentation/utils/__init__.py` | 9617c9dbd9c8 | clean | `vendor/model_augmentation/utils/__init__.py` |
| `model_augmentation/utils/deepSI_corrections.py` | f56f194b32eb | clean | `vendor/model_augmentation/utils/deepSI_corrections.py` |
| `model_augmentation/utils/torch_nets.py` | eb24b0160e59 | clean | `vendor/model_augmentation/utils/torch_nets.py` |
| `model_augmentation/utils/utils.py` | 0885d55f26c7 | clean | `vendor/model_augmentation/utils/utils.py` |
| `scripts/gantry/ann-blackbox/ann_blackbox.py` | 2fd6cd311c3f | M | `vendor/ann_blackbox/ann_blackbox.py` |
| `scripts/gantry/ann-blackbox/bla_init.py` | d00f322fc8dd | M | `vendor/ann_blackbox/bla_init.py` |
| `scripts/gantry/ann-blackbox/frf_init.py` | 8eb45a08a48e | ?? | `vendor/ann_blackbox/frf_init.py` |
| `scripts/gantry/ann-blackbox/data.py` | fd9fa50362ee | clean | `vendor/ann_blackbox/data.py` |
| `scripts/gantry/ann-blackbox/vector_fit.py` | 34e6e9d20347 | ?? | `vendor/ann_blackbox/vector_fit.py` |
| `scripts/gantry/ann-blackbox/lpm_frf.py` | 03b406787bb0 | ?? | `vendor/ann_blackbox/lpm_frf.py` |
| `scripts/gantry/msd-offset/plant.py` | 05826815d414 | ?? | `vendor/msd_offset/plant.py` |
| `telica-real/tools/watchdog.ps1` | 17e1df82373e | clean | `tools/watchdog.ps1` (unchanged) |
| `scripts/gantry/common/oracle.py` | 0e2cf8e30270 | clean | `vendor/common/oracle.py` |
| `scripts/gantry/common/orth_penalty.py` | 94e1f07918e4 | clean | `vendor/common/orth_penalty.py` |
| `scripts/gantry/common/rezero_gate.py` | e0bee5bf035c | clean | `vendor/common/rezero_gate.py` |
| (none) | | vendor-only | `vendor/common/__init__.py` (makes the vendored `common` a regular package; see its docstring) |

Not vendored, read in place (read-only): the grey-box entry file
`scripts/gantry/gantry_interconnect_dynamic.py` (its `CFG` is the configuration, BB-002), the
production data folder `data/gantry/matlab/trajectory/<CFG.mode>/`, the old `augmentation/` folder
and the four checkpoints (G2 only, BB-007), and
`scripts/gantry/ann-blackbox/results/frf_init_joint_lowf_ma50_a5/frf_init_ss.npz` (BB-008).
