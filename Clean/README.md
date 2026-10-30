# Clean code of the thesis

Minimal code that reproduces the thesis "Model Augmentation for a Dual-Gantry High-Precision Motion System": the
simulated data, the grey-box augmented model with orthogonal projection (OBC), and the black-box reference model.
Section names below refer to the thesis.

## Order of use
1. Generate the data (MATLAB): `data_generation/`.
2. Train the grey-box rows: `gantry_interconnect_dynamic.py`.
3. Train the black-box grid and select one configuration: `blackbox/`.
4. Evaluate both on train, validation and test (`blackbox/eval_bb.py` for the black box).

## Grey box: augmented model and OBC
Entry point `gantry_interconnect_dynamic.py`: a fresh run, `--resume RUN_DIR` (continue Adam) or
`--polish-only RUN_DIR` (polish from the pre-polish checkpoint). Rows R1 to R7, R13, R14 are presets in
`gantry_dynamic/config.py` (`PRESETS`: arm, baseline start, number of added states, PS initialisation).
```
python -u Clean/gantry_interconnect_dynamic.py R6 lowpass --cpu --smoke   # launch test on CPU: 2 s records, 4 updates
python -u Clean/gantry_interconnect_dynamic.py R1 lowpass                 # full run, CUDA and torch.compile
```
Data are read from `Thesis-writeup/Data/<data set>`, results go to `simulations/gantry_clean/` (`--data-root`,
`--out-root`).

| File | Role | Thesis |
|-|-|-|
| `gantry_interconnect_dynamic.py` | command line: fresh run, resume, polish-only rerun | |
| `gantry_dynamic/config.py` | run configuration: the row axes, run environment, fixed settings, detuned start | Models; Hyperparameters |
| `gantry_dynamic/data.py` | records of one data set, split, resampling to the training rate, normalisation | Records; Processing and checks |
| `gantry_dynamic/controller.py` | the known controller K_fb per stage axis, Tustin-discretised | Closed loop and excitation |
| `gantry_dynamic/model.py` | builds the augmented model and its encoder; least-squares start of the encoder | Augmented model; Initialisation |
| `gantry_dynamic/obc_gantry.py` | gantry part of OBC: reference tuples, basis Phi_Gamma, correction f_obc | Extension to the gantry |
| `gantry_dynamic/training.py` | Adam with PS, checkpoint selection, L-BFGS polish, checkpoint files | Identification |
| `gantry_dynamic/closed_loop_report.py` | evaluation helpers: model without its learned part, true-parameter coordinate | Metrics |
| `model_augmentation/fit_systems/blocks.py` | model blocks: baseline vector field f_base (LPV-LFR, Y-scheduled M(Y)), RK4 | Baseline state-space model |
| `model_augmentation/fit_systems/interconnect.py` | interconnect of blocks, encoder fit system, the training loss | Augmented model; Identification |
| `model_augmentation/fit_systems/closed_loop.py` | closed-loop simulation with K_fb in residual form | Closed-loop simulation |
| `model_augmentation/fit_systems/pre_encoder.py` | encoder psi = [W^b; W^a] chi + psi_tilde(chi) | Initialisation |
| `model_augmentation/fit_systems/ps2_opening.py` | PS initialisation of the added states (stages S1, S2) | Added-state initialisation |
| `model_augmentation/fit_systems/obc.py` | model-independent OBC: factorised basis J, coefficient theta_aux | Orthogonal-by-construction parametrisation |
| `model_augmentation/fit_systems/lbfgs_polish.py` | L-BFGS polish on fixed windows, kept only if validation improves | Identification |
| `model_augmentation/systems/gantry_ss.py` | nominal parameters theta^0 and baseline matrices | Nominal physical parameters |
| `model_augmentation/utils/torch_nets.py` | network layer stack and augmentation network phi_aug | Augmented model |
| `model_augmentation/utils/deepSI_corrections.py` | deepSI data classes that keep the data dtype | |
| `model_augmentation/utils/utils.py` | `@added` marker, tensor conversion, selection matrices | |

## Black box (R8)
SUBNET on the servo error e = y - r (deepSI 0.3.29). `run_r8.sh` trains the 11 grid configurations on the server
(`sbatch --array=9` for the selected one only).

| File | Role | Thesis |
|-|-|-|
| `blackbox/bb.py` | shared parts: model M_BB, data via `gantry_dynamic`, sigma_e | Black-box model |
| `blackbox/jan_model.py` | encoder psi and state map f_bb (names fixed by the saved checkpoints) | Black-box model |
| `blackbox/train_bb.py` | trains one configuration, weighted loss, selection by validation NRMS_e | Black-box model |
| `blackbox/select_bb.py` | selects the configuration with the lowest validation NRMS_e | Black-box model |
| `blackbox/eval_bb.py` | predictions on every split: `python -u Clean/blackbox/eval_bb.py <run folder> <output folder>` | Metrics |
| `blackbox/run_r8.sh` | SLURM array over the grid | |

## Data generation (MATLAB)
`generate_thesis_data()` writes all 48 records (18 training, 6 validation, 12 test, 12 FRF), each noisy and
noise-free, to `Thesis-writeup/Data/Coulomb-tanh-and-MSD-lowpass-noise` and `Coulomb-tanh-and-MSD-noise-free`.
Simulink model `data_generation/model/gantry_tdg_2025a.slx`; option `simulator 'replica'` steps the same loop in
plain MATLAB. Needs Jasper's unchanged functions in `Thesis-writeup/Code/Data/vendor/` (not in `Clean/`).

| File | Role | Thesis |
|-|-|-|
| `generate_thesis_data.m` | runs the manifest, saves records and provenance | Records |
| `tdg_config.m`, `tdg_init.m` | all generator constants and paths; loads the model | Simulated system |
| `tdg_manifest.m`, `tdg_write_manifest.m` | record table; `MANIFEST.csv` per data set | Records |
| `tdg_build_inputs.m` | plant, reference and multisine of one record | Closed loop and excitation |
| `gantrySystemExtendedTanh.m` | truth: gantry with hidden absorber and tanh Coulomb friction | Data-generating system |
| `gtd_build_plant.m` | nominal plant at Y_op and the fixed controller | Closed loop and excitation |
| `gtd_make_reference.m`, `gtd_validate_ref.m` | stage reference of a record; hardware-limit check | Excitation and records |
| `gtd_make_multisine.m` | injected multisine stage force | Excitation and records |
| `gtd_run_simulation.m`, `tdg_replica.m` | closed-loop simulation (Simulink, or the MATLAB replica) | Simulated system |
| `tdg_noise_psd.m`, `tdg_noise_draw.m`, `noise/noise_target.mat` | encoder-noise spectra and one realisation | Processing and checks |
| `tdg_post_check.m` | limit check on the simulated truth | Processing and checks |
| `gtd_save_record.m`, `tdg_seed.m` | record file; deterministic seeds from text keys | Records |

## Requirements
Python with PyTorch 2.5.1 (`environment.server-train.yml` in the repository root) and deepSI 0.3.29, installed
separately; CUDA for full grey-box runs. MATLAB with Simulink R2025a for data generation.

## Attribution
`model_augmentation/` is adapted from Model-Augmentation-Public by Jan H. Hoekstra (commit 8200923, BSD 3-Clause,
see `model_augmentation/LICENSE`). Changes to his code are marked `# CHANGED:`, additions `@added` or
`__project_origin__`. The black-box nets are his deepSI nets.
