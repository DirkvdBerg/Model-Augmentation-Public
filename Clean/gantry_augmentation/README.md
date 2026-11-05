# Model Augmentation for a Dual-Gantry High-Precision Motion System

Code of the MSc thesis of the same name: the grey-box augmented model with orthogonal-by-construction projection
(OBC), and the black-box reference model. Brackets name the thesis section.

## Installation
    conda env create -f environment.yml
    conda activate gantry-augmentation
Pins the environment of the thesis results: Python 3.11.14, PyTorch 2.5.1 with CUDA 12.1, deepSI 0.3.29 (GitHub
legacy branch). Full grey-box runs need a CUDA GPU (OBC polish: about 28 GB).

## Data
Not included; `data_generation/` generates it in MATLAB. Set the data and output folders in
`DATA_ROOT` and `OUT_ROOT` of `gantry_dynamic/config.py`.

## Reproducing the thesis
1. **Grey box**: `sbatch run_grey_box.sh` trains the nine thesis models (table below); each run ends with its
   evaluation on train, validation and test in `<run folder>/evaluation/`.
2. **Black box**: `sbatch blackbox/run_black_box.sh` trains the 11 grid configurations; then
   `python -m blackbox.select_bb <grid folder>` selects one (thesis: index 9).
3. **Black-box evaluation**: `python -m blackbox.eval_bb <run folder>`.

Fill in the `<...>` placeholders of both `.sh` files first. Black-box commands run from this folder.

## Grey-box settings
All settings are at the top of `gantry_interconnect_dynamic.py`, commented, with the thesis values; the default is
M_OBC. Five axes select the model and can be set on the command line:

| Setting | Values | Flag |
|-|-|-|
| arm | `obc` (projection), `u` (unconstrained) | `--arm` |
| start | `detuned` (10 %), `correct` | `--start` |
| n_a | added states: 0, 2, 8 | `--n-a` |
| ps2 | PS initialisation of the added states | `--ps2` / `--no-ps2` |
| noise | `noisy`, `noisefree` | `--noise` |

The nine thesis models, as in `run_grey_box.sh`:

| Name | Arguments |
|-|-|
| M_OBC | `--arm obc --start detuned --n-a 2 --ps2` |
| M_U | `--arm u --start detuned --n-a 2 --ps2` |
| obc_correct_na2_ps | `--arm obc --start correct --n-a 2 --ps2` |
| obc_detuned_na0 | `--arm obc --start detuned --n-a 0 --no-ps2` |
| u_correct_na2 | `--arm u --start correct --n-a 2 --no-ps2` |
| u_correct_na2_ps | `--arm u --start correct --n-a 2 --ps2` |
| u_correct_na0 | `--arm u --start correct --n-a 0 --no-ps2` |
| u_correct_na8_ps | `--arm u --start correct --n-a 8 --ps2` |
| u_detuned_na0 | `--arm u --start detuned --n-a 0 --no-ps2` |

On a finished run folder (settings are read from its `config.json`):
- `--resume RUN_DIR`: continue Adam
- `--polish-only RUN_DIR`: redo the L-BFGS polish
- `--evaluate RUN_DIR`: evaluation only (CPU is enough)

# Project files
- **gantry_interconnect_dynamic.py**: all settings; fresh run, resume, polish-only, evaluation (Models; Hyperparameters)
- **run_grey_box.sh**: SLURM array over the nine grey-box models
## gantry_dynamic
Gantry-specific parts.
- **config.py**: run configuration, checks, `config.json`, detuned start
- **data.py**: records, split, resampling, normalisation (Records; Processing and checks)
- **controller.py**: known controller K_fb per axis, Tustin (Closed loop and excitation)
- **model.py**: augmented model and encoder; least-squares encoder start (Augmented model; Initialisation)
- **obc_gantry.py**: gantry part of OBC: reference tuples, basis, f_obc (Extension to the gantry)
- **training.py**: Adam with PS, checkpoint selection, L-BFGS polish (Identification)
- **evaluation.py**: NRMS_e of the model and both baselines per split (Metrics)
## model_augmentation
Jan Hoekstra's framework, with additions.
- **fit_systems/blocks.py**: baseline f_base (LPV-LFR, M(Y)), RK4 (Baseline state-space model)
- **fit_systems/interconnect.py**: block interconnect, encoder fit system, loss (Augmented model; Identification)
- **fit_systems/closed_loop.py**: closed-loop simulation with K_fb (Closed-loop simulation)
- **fit_systems/pre_encoder.py**: encoder psi = [W^b; W^a] chi + psi_tilde(chi) (Initialisation)
- **fit_systems/ps2_opening.py**: PS initialisation, stages S1 and S2 (Added-state initialisation)
- **fit_systems/obc.py**: model-independent OBC: basis J, theta_aux (Orthogonal-by-construction parametrisation)
- **fit_systems/lbfgs_polish.py**: L-BFGS polish, kept only if validation improves (Identification)
- **systems/gantry_ss.py**: nominal parameters theta^0, baseline matrices (Nominal physical parameters)
- **utils/torch_nets.py**: layer stack, augmentation network phi_aug (Augmented model)
- **utils/deepSI_corrections.py**, **utils/utils.py**: dtype-safe deepSI data, `@added` marker, helpers
## blackbox
SUBNET on the servo error e = y - r, on the grey box's samples and windows (Black-box model).
- **bb.py**: model M_BB, data loader, whole-record prediction
- **train_bb.py**: one configuration, weighted loss, best validation NRMS_e
- **select_bb.py**: lowest validation NRMS_e of the grid
- **eval_bb.py**: predictions on every split (Metrics)
- **run_black_box.sh**: SLURM array over the grid
## data_generation
MATLAB generation of the data sets, with Simulink or the plain-MATLAB `tdg_replica.m` (Records).

# License and attribution
`model_augmentation/` is adapted from Model-Augmentation-Public by Jan H. Hoekstra (BSD 3-Clause,
see `model_augmentation/LICENSE`). Changes to his code are marked `# CHANGED:`, additions `@added` or
`__project_origin__`. The black-box nets are deepSI's default nets.

The OBC in `fit_systems/obc.py` and `gantry_dynamic/obc_gantry.py` adapts the orthogonal-by-construction method of
B. M. Györök et al. (code: https://github.com/AIMotionLab-SZTAKI/orthogonal-IO-augm, MIT License) from input-output
to state-space baselines, reimplemented in PyTorch without code from his repository.
