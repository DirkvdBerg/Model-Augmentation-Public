# Vendored sources (JH-001)

Copied 2026-09-24 from the WORKING TREE at commit `1ec57c1` (branch `Augmentation`, tree dirty;
the handoff names `a97a577`, an ancestor). sha256 (first 12 hex) is of the SOURCE at copy time;
git status: `clean` = identical to HEAD, `M` = modified in the working tree, `??` = untracked.
Only lines marked `VENDOR-PATCH (JH-001)` differ from the source; `tools/check_vendor.py`
re-verifies that. Patched files: `gantry_dynamic/__init__.py` (vendor on sys.path),
`gantry_dynamic/config.py` (REPO_ROOT depth, save_dir into outputs/sim), `gantry_dynamic/data.py`
(JH_FILESET switch). Data are read in place from `data/gantry/matlab/trajectory/`.

| Source | sha256 (12) | git status | Copy |
|-|-|-|-|
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
| `scripts/gantry/common/feedback_curriculum.py` | cf9b655c6f9c | clean | `vendor/common/feedback_curriculum.py` |
| `scripts/gantry/common/oracle.py` | 0e2cf8e30270 | clean | `vendor/common/oracle.py` |
| `scripts/gantry/common/orth_penalty.py` | 94e1f07918e4 | clean | `vendor/common/orth_penalty.py` |
| `scripts/gantry/common/rezero_gate.py` | e0bee5bf035c | clean | `vendor/common/rezero_gate.py` |
| `scripts/gantry/gantry_interconnect_dynamic.py` | 1d722fa4678c | clean | `vendor/gantry_interconnect_dynamic.py` |
| `telica-real/tools/watchdog.ps1` | 17e1df82373e | clean | `tools/watchdog.ps1` |

## telica-real tree (JH-004)

Copied 2026-09-24 from `telica-real/` (working tree). Binary data files (npz, mat) are copied byte for byte and checked by sha256 only.

| Source | sha256 (12) | git status | Copy |
|-|-|-|-|
| `telica-real/baseline/__init__.py` | f085cf55025f | clean | `vendor/telica_real/baseline/__init__.py` |
| `telica-real/baseline/cl_sim.py` | 11fa4028ac55 | clean | `vendor/telica_real/baseline/cl_sim.py` |
| `telica-real/baseline/diag_limit_cycle.py` | df14bc83dc9a | clean | `vendor/telica_real/baseline/diag_limit_cycle.py` |
| `telica-real/baseline/g4_replay.py` | f4c7d9e0d3db | clean | `vendor/telica_real/baseline/g4_replay.py` |
| `telica-real/baseline/g4_verdict.py` | 5cf53315ab16 | clean | `vendor/telica_real/baseline/g4_verdict.py` |
| `telica-real/baseline/holding.py` | 19a39d50ca89 | clean | `vendor/telica_real/baseline/holding.py` |
| `telica-real/baseline/records.py` | da73115b6c5a | clean | `vendor/telica_real/baseline/records.py` |
| `telica-real/baseline/recover.py` | 6edf40366fe8 | clean | `vendor/telica_real/baseline/recover.py` |
| `telica-real/baseline/recovered_params.json` | 8321a52cff5a | clean | `vendor/telica_real/baseline/recovered_params.json` |
| `telica-real/baseline/recovered_params_a2.json` | bd92a4fa19c0 | clean | `vendor/telica_real/baseline/recovered_params_a2.json` |
| `telica-real/baseline/rms_table.py` | 9283b45a1046 | clean | `vendor/telica_real/baseline/rms_table.py` |
| `telica-real/baseline/survey.py` | a882e089f793 | clean | `vendor/telica_real/baseline/survey.py` |
| `telica-real/controller/__init__.py` | 805dea1ec458 | clean | `vendor/telica_real/controller/__init__.py` |
| `telica-real/controller/analyze_controller.py` | 179c46c6327b | clean | `vendor/telica_real/controller/analyze_controller.py` |
| `telica-real/controller/check_bank.py` | a2a397d37d8b | clean | `vendor/telica_real/controller/check_bank.py` |
| `telica-real/controller/identify_5k.py` | 4135a289b791 | clean | `vendor/telica_real/controller/identify_5k.py` |
| `telica-real/controller/identify_logged.py` | e8dc7bae3f45 | clean | `vendor/telica_real/controller/identify_logged.py` |
| `telica-real/controller/iter0_replay.py` | ff933ee3f697 | clean | `vendor/telica_real/controller/iter0_replay.py` |
| `telica-real/controller/mechanism.py` | e27fc7e9ff59 | clean | `vendor/telica_real/controller/mechanism.py` |
| `telica-real/controller/telica_bank.py` | 89381e071085 | clean | `vendor/telica_real/controller/telica_bank.py` |
| `telica-real/controller/telica_ctrl.py` | da6b737b25a8 | clean | `vendor/telica_real/controller/telica_ctrl.py` |
| `telica-real/controller/telica_sos_identified.npz` | 6fbbcd969232 | clean | `vendor/telica_real/controller/telica_sos_identified.npz` |
| `telica-real/controller/telica_sos_identified_5k.npz` | 0d049465f148 | clean | `vendor/telica_real/controller/telica_sos_identified_5k.npz` |
| `telica-real/controller/telica_zpk_bhl.mat` | 5de86f55a133 | clean | `vendor/telica_real/controller/telica_zpk_bhl.mat` |
| `telica-real/data_index.py` | ba1ec47623c4 | clean | `vendor/telica_real/data_index.py` |
| `telica-real/friction/__init__.py` | c4587739a2de | clean | `vendor/telica_real/friction/__init__.py` |
| `telica-real/friction/check_friction.py` | 68bcbd3dff22 | clean | `vendor/telica_real/friction/check_friction.py` |
| `telica-real/friction/karnopp.py` | 4a8d110b11b9 | clean | `vendor/telica_real/friction/karnopp.py` |
| `telica-real/learnability/__init__.py` | 3786ac1a8043 | clean | `vendor/telica_real/learnability/__init__.py` |
| `telica-real/learnability/ann_settings.json` | 10a4a69f32e0 | clean | `vendor/telica_real/learnability/ann_settings.json` |
| `telica-real/learnability/g5_learnability.py` | e319ff2583f6 | clean | `vendor/telica_real/learnability/g5_learnability.py` |
| `telica-real/learnability/make_residuals.py` | ec4ef9680f45 | clean | `vendor/telica_real/learnability/make_residuals.py` |
| `telica-real/learnability/order_sensitivity.py` | 149fa6abccbc | clean | `vendor/telica_real/learnability/order_sensitivity.py` |
| `telica-real/params/__init__.py` | 1e55323e8f34 | clean | `vendor/telica_real/params/__init__.py` |
| `telica-real/params/check_params.py` | 001543a0305e | clean | `vendor/telica_real/params/check_params.py` |
| `telica-real/params/telica_params.py` | 10a185188322 | clean | `vendor/telica_real/params/telica_params.py` |
| `telica-real/pipeline/__init__.py` | aa872cfe4e4d | clean | `vendor/telica_real/pipeline/__init__.py` |
| `telica-real/pipeline/check_g6.py` | 8eda6c288671 | clean | `vendor/telica_real/pipeline/check_g6.py` |
| `telica-real/pipeline/check_joint.py` | fc09fdc238e8 | clean | `vendor/telica_real/pipeline/check_joint.py` |
| `telica-real/pipeline/memory.py` | bc24a5c7a8e1 | clean | `vendor/telica_real/pipeline/memory.py` |
| `telica-real/pipeline/norm_frozen.json` | 7bb6f2c40a7c | clean | `vendor/telica_real/pipeline/norm_frozen.json` |
| `telica-real/pipeline/probe.py` | c68fd0a17a64 | clean | `vendor/telica_real/pipeline/probe.py` |
| `telica-real/pipeline/real_data.py` | effbf5a73c47 | clean | `vendor/telica_real/pipeline/real_data.py` |
| `telica-real/pipeline/telica_augment.py` | df9320321e81 | clean | `vendor/telica_real/pipeline/telica_augment.py` |
| `telica-real/pipeline/telica_model.py` | 93a23922201e | clean | `vendor/telica_real/pipeline/telica_model.py` |
| `telica-real/pipeline/telica_obc.py` | f29f656bbc3c | clean | `vendor/telica_real/pipeline/telica_obc.py` |
| `telica-real/rate.py` | 05182c15ab23 | clean | `vendor/telica_real/rate.py` |
| `telica-real/tools/check_imports.py` | 67eefc768c6f | clean | `vendor/telica_real/tools/check_imports.py` |
| `telica-real/tools/compile_all.py` | 161023bf5788 | clean | `vendor/telica_real/tools/compile_all.py` |
| `telica-real/tools/dummy_load.py` | 2570259b8236 | clean | `vendor/telica_real/tools/dummy_load.py` |
| `telica-real/tools/resources_summary.py` | 37223515a064 | clean | `vendor/telica_real/tools/resources_summary.py` |
| `telica-real/tr_env.py` | a455dc8c59a0 | clean | `vendor/telica_real/tr_env.py` |
| `telica-real/vendor/common/feedback_curriculum.py` | cf9b655c6f9c | clean | `vendor/telica_real/vendor/common/feedback_curriculum.py` |
| `telica-real/vendor/common/oracle.py` | 0e2cf8e30270 | clean | `vendor/telica_real/vendor/common/oracle.py` |
| `telica-real/vendor/common/orth_penalty.py` | 94e1f07918e4 | clean | `vendor/telica_real/vendor/common/orth_penalty.py` |
| `telica-real/vendor/common/rezero_gate.py` | e0bee5bf035c | clean | `vendor/telica_real/vendor/common/rezero_gate.py` |
| `telica-real/vendor/gantry_dynamic/__init__.py` | a074ecc3cf74 | clean | `vendor/telica_real/vendor/gantry_dynamic/__init__.py` |
| `telica-real/vendor/gantry_dynamic/baselines.py` | 55c262288bc0 | clean | `vendor/telica_real/vendor/gantry_dynamic/baselines.py` |
| `telica-real/vendor/gantry_dynamic/config.py` | cc52d1448c8b | clean | `vendor/telica_real/vendor/gantry_dynamic/config.py` |
| `telica-real/vendor/gantry_dynamic/controller.py` | a378ff000eb2 | clean | `vendor/telica_real/vendor/gantry_dynamic/controller.py` |
| `telica-real/vendor/gantry_dynamic/data.py` | 0a4b6f4057b0 | clean | `vendor/telica_real/vendor/gantry_dynamic/data.py` |
| `telica-real/vendor/gantry_dynamic/diagnostics.py` | 48cba98e1f51 | clean | `vendor/telica_real/vendor/gantry_dynamic/diagnostics.py` |
| `telica-real/vendor/gantry_dynamic/evaluation.py` | f18027018c62 | clean | `vendor/telica_real/vendor/gantry_dynamic/evaluation.py` |
| `telica-real/vendor/gantry_dynamic/model.py` | 5119c21172c6 | clean | `vendor/telica_real/vendor/gantry_dynamic/model.py` |
| `telica-real/vendor/gantry_dynamic/obc_diagnostics.py` | c2fa74df98d2 | clean | `vendor/telica_real/vendor/gantry_dynamic/obc_diagnostics.py` |
| `telica-real/vendor/gantry_dynamic/obc_gantry.py` | ffe22fe019b5 | clean | `vendor/telica_real/vendor/gantry_dynamic/obc_gantry.py` |
| `telica-real/vendor/gantry_dynamic/training.py` | 2d080753abc1 | clean | `vendor/telica_real/vendor/gantry_dynamic/training.py` |
| `telica-real/vendor/gantry_interconnect_dynamic.py` | 1d722fa4678c | clean | `vendor/telica_real/vendor/gantry_interconnect_dynamic.py` |
| `telica-real/vendor/model_augmentation/__init__.py` | 3ccd35009717 | clean | `vendor/telica_real/vendor/model_augmentation/__init__.py` |
| `telica-real/vendor/model_augmentation/fit_systems/__init__.py` | 07a18a4e3720 | clean | `vendor/telica_real/vendor/model_augmentation/fit_systems/__init__.py` |
| `telica-real/vendor/model_augmentation/fit_systems/augmented_dynamics.py` | 2bb9aaf47830 | clean | `vendor/telica_real/vendor/model_augmentation/fit_systems/augmented_dynamics.py` |
| `telica-real/vendor/model_augmentation/fit_systems/blocks.py` | 9e3e3294e9c4 | clean | `vendor/telica_real/vendor/model_augmentation/fit_systems/blocks.py` |
| `telica-real/vendor/model_augmentation/fit_systems/closed_loop.py` | 092b993c0d75 | clean | `vendor/telica_real/vendor/model_augmentation/fit_systems/closed_loop.py` |
| `telica-real/vendor/model_augmentation/fit_systems/continuous_augmentation.py` | 066b70d79ced | clean | `vendor/telica_real/vendor/model_augmentation/fit_systems/continuous_augmentation.py` |
| `telica-real/vendor/model_augmentation/fit_systems/gyorok_orth_projection.py` | 4399dc665264 | clean | `vendor/telica_real/vendor/model_augmentation/fit_systems/gyorok_orth_projection.py` |
| `telica-real/vendor/model_augmentation/fit_systems/interconnect.py` | eb032c450428 | clean | `vendor/telica_real/vendor/model_augmentation/fit_systems/interconnect.py` |
| `telica-real/vendor/model_augmentation/fit_systems/lbfgs_polish.py` | 31a7436e3251 | clean | `vendor/telica_real/vendor/model_augmentation/fit_systems/lbfgs_polish.py` |
| `telica-real/vendor/model_augmentation/fit_systems/multiple_shooting.py` | e218ab0665e2 | clean | `vendor/telica_real/vendor/model_augmentation/fit_systems/multiple_shooting.py` |
| `telica-real/vendor/model_augmentation/fit_systems/obc.py` | b6b12b0a7736 | clean | `vendor/telica_real/vendor/model_augmentation/fit_systems/obc.py` |
| `telica-real/vendor/model_augmentation/fit_systems/obc_projection.py` | 27035f6b8fba | clean | `vendor/telica_real/vendor/model_augmentation/fit_systems/obc_projection.py` |
| `telica-real/vendor/model_augmentation/fit_systems/orth_projection.py` | 92cfb224c759 | clean | `vendor/telica_real/vendor/model_augmentation/fit_systems/orth_projection.py` |
| `telica-real/vendor/model_augmentation/fit_systems/pre_encoder.py` | c396b992aa4b | clean | `vendor/telica_real/vendor/model_augmentation/fit_systems/pre_encoder.py` |
| `telica-real/vendor/model_augmentation/fit_systems/white_box_models.py` | 561c053bbd53 | clean | `vendor/telica_real/vendor/model_augmentation/fit_systems/white_box_models.py` |
| `telica-real/vendor/model_augmentation/systems/__init__.py` | 314e58500342 | clean | `vendor/telica_real/vendor/model_augmentation/systems/__init__.py` |
| `telica-real/vendor/model_augmentation/systems/cascaded_tanks.py` | 66f60076d15a | clean | `vendor/telica_real/vendor/model_augmentation/systems/cascaded_tanks.py` |
| `telica-real/vendor/model_augmentation/systems/gantry_linearization.py` | fdc419ebad7a | clean | `vendor/telica_real/vendor/model_augmentation/systems/gantry_linearization.py` |
| `telica-real/vendor/model_augmentation/systems/gantry_ss.py` | 1c0c46b49e41 | clean | `vendor/telica_real/vendor/model_augmentation/systems/gantry_ss.py` |
| `telica-real/vendor/model_augmentation/systems/mass_spring_damper.py` | 25f0f4b61bda | clean | `vendor/telica_real/vendor/model_augmentation/systems/mass_spring_damper.py` |
| `telica-real/vendor/model_augmentation/utils/__init__.py` | 9617c9dbd9c8 | clean | `vendor/telica_real/vendor/model_augmentation/utils/__init__.py` |
| `telica-real/vendor/model_augmentation/utils/deepSI_corrections.py` | f56f194b32eb | clean | `vendor/telica_real/vendor/model_augmentation/utils/deepSI_corrections.py` |
| `telica-real/vendor/model_augmentation/utils/torch_nets.py` | eb24b0160e59 | clean | `vendor/telica_real/vendor/model_augmentation/utils/torch_nets.py` |
| `telica-real/vendor/model_augmentation/utils/utils.py` | 0885d55f26c7 | clean | `vendor/telica_real/vendor/model_augmentation/utils/utils.py` |
| `telica-real/vendor/real_data_verification/cl_headroom.py` | 98db74d76ed6 | clean | `vendor/telica_real/vendor/real_data_verification/cl_headroom.py` |
| `telica-real/vendor/real_data_verification/cl_residual_spectrum.py` | 8948a03f9937 | clean | `vendor/telica_real/vendor/real_data_verification/cl_residual_spectrum.py` |
| `telica-real/vendor/real_data_verification/plant_coulomb_ref.py` | 69f83ddee415 | clean | `vendor/telica_real/vendor/real_data_verification/plant_coulomb_ref.py` |
| `telica-real/vendor/real_data_verification/telica_controller.py` | f0c49be94dd5 | clean | `vendor/telica_real/vendor/real_data_verification/telica_controller.py` |
| `telica-real/vendor/real_data_verification/telica_loader.py` | 6fa94d38c6ae | clean | `vendor/telica_real/vendor/real_data_verification/telica_loader.py` |
| `telica-real/vendor/real_data_verification/telica_residual_spectrum.py` | 8d7f7cc3a89d | clean | `vendor/telica_real/vendor/real_data_verification/telica_residual_spectrum.py` |
| `telica-real/vendor/VENDORED.md` | 831d7a13421f | clean | `vendor/telica_real/vendor/VENDORED.md` |
