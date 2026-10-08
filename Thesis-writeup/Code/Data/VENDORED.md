# Vendored files

Copied 2026-09-26 from repo HEAD `d641059` (branch Augmentation), working tree as found. SHA-256 prefixes (16 hex) are of the copied bytes; the full hash is reproducible with `sha256sum`.

## Layout
- `scripts/gantry/thesis-data-verification/vendor/original/`: verbatim copies of every source file, never edited, not on the generator's path; the reference for diffs and for the bitwise model check (C1); moved there from this folder on 2026-09-27
- `vendor/`: verbatim runtime copies (identical bytes to those in `vendor/original/`), on the path, plus the source Simulink model `gantry_additional_state_coulomb_2025a.slx` (moved here from `vendor/original/`, since `tdg_make_model.m` builds from it)
- `scripts/gantry/thesis-data-verification/vendor/`: `gantrySystem.m`, `gantrySystemCoriolisCentripetal.m`, verbatim, needed only to load the original four-loop model in check C1
- In the table below, "original only" means: kept in `scripts/gantry/thesis-data-verification/vendor/original/`, not used by the generator
- `./gtd_*.m`: modified copies; every change is marked `THESIS-DATA` in the file and listed in `IMPLEMENTATION.md`

## Files
| File | Source path | Source state | SHA-256 (16) | Used as |
|-|-|-|-|-|
| `gantry_additional_state_coulomb_2025a.slx` | `Matlab-scripts/Augmentation-coulomb/` | clean | `de2a85857446c451` | `vendor/`, verbatim (moved from `vendor/original/` 2026-09-27); `model/gantry_tdg_2025a.slx` is built from it by `tdg_make_model.m` |
| `gantrySystemExtendedCoulomb.m` | `Matlab-scripts/Augmentation-coulomb/` | working tree differs from HEAD: `V_BRK = 1.0e-3` (D-209 swept value, in force) against HEAD's superseded `2.25e-3`; the working tree is the one to trust (handoff) | `675260373f477c07` | `vendor/`, verbatim; since D-224 used only with cc = 0 (friction off, `tdg_truth_lin.m`); the recorded loop calls the generator-owned `gantrySystemExtendedTanh.m`, which copies its mass, damping and stiffness sections |
| `generate_trajectory_data_coulomb.m` | `Matlab-scripts/Augmentation-coulomb/` | working tree differs from HEAD (SELECT and out_dir only) | `efc1529ebf23dda3` | original only; its knobs are folded into `tdg_config.m` |
| `make_coulomb_model.m` | `Matlab-scripts/Augmentation-coulomb/` | clean | `d17fa2e46abd062b` | original only (how the .slx was built) |
| `coulomb_v_brk.m` | `Matlab-scripts/Augmentation-coulomb/` | clean | `533a8f0179d9d638` | original only; its parse rule is used in `generate_thesis_data.m` (truth_params) |
| `generate_trajectory_data_telica_coulomb.m` | `Matlab-scripts/Augmentation-telica-profiles-coulomb/` | working tree differs from HEAD | `fdeac6286cb047b3` | original only (the ILC-shape driver the copied generator ran) |
| `gtd_build_records_telica.m` | `Matlab-scripts/Augmentation-telica-profiles/` | working tree differs from HEAD: the D-210 `TEL` kinematics (in force) | `84b915de3bc50fd4` | original only; replaced by `tdg_manifest.m`; E6 uses its `TEL` values |
| `gtd_make_reference_telica.m` | `Matlab-scripts/Augmentation-telica-profiles/` | clean | `9e9f26dfc6d09d99` | modified copy `./gtd_make_reference_telica.m` |
| `gtd_config.m` | `Matlab-scripts/Augmentation/data/` | clean | `8be57ef96c29ac11` | original only; replaced by `tdg_config.m` |
| `gtd_build_plant.m` | same | clean | `6bdef7c8243dfa3b` | modified copy |
| `gtd_build_records.m` | same | clean | `39ac8162c456daf0` | original only; replaced by `tdg_manifest.m` |
| `gtd_make_reference.m` | same | clean | `336e8a1d484f16ad` | modified copy |
| `gtd_make_multisine.m` | same | clean | `ddb510078897dd57` | modified copy |
| `gtd_run_simulation.m` | same | clean | `00d915cbd3bb8e4f` | modified copy |
| `gtd_save_record.m` | same | clean | `d104339fc26b90d5` | modified copy |
| `gtd_validate_ref.m` | same | clean | `33b576bcd811f657` | modified copy |
| `gtd_enforce_limits.m` | same | clean | `64f3f90b878f1fa0` | `vendor/`, verbatim |
| `gtd_size_anti_amp.m` | same | clean | `abef453bbb70a521` | `vendor/`, verbatim |
| `gtd_logical_to_stage.m` | same | clean | `af1c692237622b68` | `vendor/`, verbatim |
| `gtd_check_phaseA.m` | same | clean | `24cd89a01c9f538f` | original only; its asserts live on in `gtd_validate_ref.m` and `scripts/gantry/thesis-data-verification/checks/check_references.m` |
| `gtd_plot_record.m` | same | clean | `ab99006519e48fb3` | original only; per-record figures are not produced (disk, see `IMPLEMENTATION.md`) |
| `getss.m`, `ruleOfThumb.m`, `thirdOrderSetpointETEL.m` | `kamtin-fp-model/03 Simulink gantry/functions/` (read-only source, copied, never modified) | clean | `97734cfd7970c3d0`, `c4bcc59c4abbf7a7`, `16407c64c3edf95f` | `vendor/`, verbatim |
| `gantrySystem.m`, `gantrySystemCoriolisCentripetal.m` | same | clean | `578f4ee1f0c19e98`, `2d6ce80f6f0ce990` | `scripts/gantry/thesis-data-verification/vendor/`, verbatim; needed only to load the ORIGINAL model in check C1 |

## Inputs read, not copied
| Input | Path | SHA-256 (16) | Use |
|-|-|-|-|
| Measured standstill error spectrum | `scripts/gantry/closed-loop-noise/outputs/g1_spectra/g1.npz` | `84f4e9a427f4b57b` | exported by `noise/export_noise_target.py` to `noise/noise_target.mat` (full hash stored inside) |
| Planned training references of the excitation evidence (J9, J10) | `scripts/gantry/excitation-closed-loop/outputs/j9_refs.mat` | read only in `scripts/gantry/thesis-data-verification/checks/check_references.m` | bitwise comparison C10a |

## Not copied
- `Matlab-scripts/Augmentation-coulomb/check_*.m` (friction gates of D-204 and D-209): they validate `gantrySystemExtendedCoulomb.m`, which is copied byte for byte, and they read datasets under `data/`; not needed to run the generator
- `Matlab-scripts/Augmentation-coulomb/README.md`: its band section is stale (handoff); D-209 and D-219 are used instead
