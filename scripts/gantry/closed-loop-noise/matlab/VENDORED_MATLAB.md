# Vendored MATLAB (CN-009)

Copied 2026-09-24 into `matlab/gen/`. HEAD `1ec57c1`, working tree dirty. Edits marked
`% CLOSED-LOOP-NOISE:`.

| File in `matlab/gen/` | Source | State |
|-|-|-|
| `gtd_build_plant.m`, `gtd_build_records.m`, `gtd_enforce_limits.m`, `gtd_logical_to_stage.m`, `gtd_make_multisine.m`, `gtd_make_reference.m`, `gtd_plot_record.m`, `gtd_size_anti_amp.m` | `Matlab-scripts/Augmentation/data/` | working tree = HEAD (unchanged since 2026-09-19), verbatim |
| `gtd_config.m` | same | edited: repo root 5 levels up; `Matlab-scripts/Augmentation(/diagnostics)` no longer added (everything needed is vendored); `kamtin-fp-model/03 Simulink gantry` still added, read only, as in production |
| `gtd_run_simulation.m` | same | edited: `v_enc = [t, v]` pushed to base (zeros when no noise); `u_fb = lsim(Cfb, r - (q + v))`; `out.v_enc` |
| `gtd_save_record.m` | same | edited: `y = q + v` (measured), `Y_trajectory = y(:,3)`, precision switch `cfg.save_double`, extra fields `q_true`, `v_enc`, `noise_seed`, `noise_model` when double |
| `gtd_build_records_telica.m`, `gtd_make_reference_telica.m` | `Matlab-scripts/Augmentation-telica-profiles/` | verbatim |
| `gantrySystemExtendedCoulomb.m` | `Matlab-scripts/Augmentation-coulomb/` at git `ae25789^` | verbatim, the PRE-D-209 stick band (`v_eps = 9 (cc1+cc2)/m_total ts`) that generated the production records |
| `gantry_cn_base_2025a.slx` | `Matlab-scripts/Augmentation-coulomb/gantry_additional_state_coulomb_2025a.slx` (md5 e7df4cbf007e...) | byte copy, renamed; never edited |
| `gantry_cn_noise_2025a.slx` | built from the base by `make_noise_model.m` | one edit: encoder-noise Sum on `Gain4 -> Sum3` |
| `cn_gen_multisine.m` | `Matlab-scripts/Augmentation-coulomb/generate_trajectory_data_coulomb.m` (working tree) | function copy: same knobs and call sequence; mode / select / out_dir / noise_dir arguments |
| `cn_gen_telica.m` | `Matlab-scripts/Augmentation-telica-profiles-coulomb/generate_trajectory_data_telica_coulomb.m` (HEAD `548e112`) | function copy, same edits |
| `attach_noise.m`, `make_noise_model.m`, `g5*_*.m` | new | CN-009 |

Not vendored, used read only from the repo's `kamtin-fp-model/03 Simulink gantry/functions/`:
`getss.m`, `ruleOfThumb.m`, `thirdOrderSetpointETEL.m`, `gantrySystem.m`,
`gantrySystemCoriolisCentripetal.m` (the model's other chart functions).
