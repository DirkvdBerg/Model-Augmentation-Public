# Model Augmentation for a Dual-Gantry High-Precision Motion System

Code of the MSc thesis of the same name: the grey-box augmented model with orthogonal-by-construction projection
(OBC), and the black-box reference model. Brackets name the thesis section.

## Installation
    conda env create -f environment.yml
    conda activate gantry-augmentation
Pins the environment of the thesis results: Python 3.11.14, PyTorch 2.5.1 with CUDA 12.1, deepSI 0.3.29 (GitHub
legacy branch). Full grey-box runs need a CUDA GPU (OBC polish: about 28 GB).

## Data
`data/` holds the thesis data sets, with and without encoder noise: 18 training, 6 validation and 11 test records,
and the 7 test records repeated with the multisine (block `test-M` in `MANIFEST.csv`, evaluation only).
`data_generation/` reproduces them bitwise in MATLAB. Set the data and output folders in `DATA_ROOT` and
`OUT_ROOT` of `gantry_dynamic/config.py`.

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
MATLAB and Simulink generation of the data sets (Records). Jasper's `getss.m`, `ruleOfThumb.m`,
`thirdOrderSetpointETEL.m` and `model/gantry_2025a.slx` are unchanged copies.
- **generate_data.m**: run it to write all 42 records, with and without encoder noise, to `data/`
- **build_model.m**: builds the truth loop `model/gantry_truth.slx` from Jasper's model
- **data_config.m**: all constants, with their sources
- **record_table.m**: the records and their seeds
- **make_reference.m**, **make_multisine.m**, **encoder_noise.m**: reference, injected force, encoder noise
- **build_loop.m**, **precheck_limits.m**: controller K1 and the linear limit check that sets the multisine scale
- **run_simulation.m**, **gantrySystemExtendedTanh.m**: one closed-loop run; the truth plant

**Motion profiles** (`make_reference.m`). Each record is 12 s: a 0.5 s hold at [0, 0, Y_op], 10 s of motion and a
final hold. Five types:
- standstill at a fixed Y; the multisine is the only excitation
- slow sweeps of Y and X-Y Lissajous paths: sinusoids that fade in and out with a half cosine
- point-to-point moves with S-curve profiles (Jasper's `thirdOrderSetpointETEL.m`); each distance is drawn from
  one of six bands between 1 mm and the range, so small and large moves occur equally often
- operational motion profiles: sine-shaped moves of 40 mm on X and 80 mm on Y with X and Y together, the shape of
  the ASMPT machine's logged setpoints (half-sine acceleration up to a velocity limit)
- the test routes I3 to I6 and E2 to E6 vary these: other ranges, alternating X and Y moves, S-curve profiles,
  long strokes

Every reference is checked against the position, yaw, speed and acceleration limits before it is simulated.

**Encoder noise** (`encoder_noise.m`). Independent coloured noise on each measured stage position X1, X2 and Y:
a second-order low-pass on a white floor. Its parameters in `data_config.m` are set so that at standstill the
simulated servo error matches the noise level and spectral shape of the ASMPT gantry; the machine's resonances
are left out. Each record and axis has a fixed seed. The noise-free version is the same simulation without the noise.

# License and attribution
`model_augmentation/` is adapted from Model-Augmentation-Public by Jan H. Hoekstra (BSD 3-Clause,
see `model_augmentation/LICENSE`). Changes to his code are marked `# CHANGED:`, additions `@added` or
`__project_origin__`. The black-box nets are deepSI's default nets.

The OBC in `fit_systems/obc.py` and `gantry_dynamic/obc_gantry.py` adapts the orthogonal-by-construction method of
B. M. Györök et al. (code: https://github.com/AIMotionLab-SZTAKI/orthogonal-IO-augm, MIT License) from input-output
to state-space baselines, reimplemented in PyTorch without code from his repository.
