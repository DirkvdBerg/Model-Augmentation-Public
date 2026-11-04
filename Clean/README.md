# Clean code of the thesis

Minimal code that reproduces the thesis "Model Augmentation for a Dual-Gantry High-Precision Motion System": the
grey-box augmented model with orthogonal projection (OBC) and the black-box reference model.
Section names below refer to the thesis.

## Order of use
1. Train the grey-box rows: `gantry_interconnect_dynamic.py`.
2. Train the black-box grid and select one configuration: `blackbox/`.
3. Evaluate both on train, validation and test: the grey box at the end of each run (`gantry_dynamic/evaluation.py`),
   the black box with `blackbox/eval_bb.py`.

## Grey box: augmented model and OBC
Entry point `gantry_interconnect_dynamic.py`. Every setting is at its top, with the thesis values: `MODE` and
`RUN_DIR` (a fresh run; `resume`, continue Adam; `polish-only`, polish from the pre-polish checkpoint; `evaluate`, a
finished run only), `EVALUATE`, and `SETTINGS` (the five axes arm, baseline start, number of added states, PS
opening and data set; sampling and windows; networks; Adam; L-BFGS; PS opening; environment). The default is the
thesis model M_OBC; the other thesis models change the axes. A run's `config.json` records its settings, and a
resume or polish-only run reads them from there. Training ends with the evaluation of the final checkpoint into
`RUN_DIR/evaluation/`.
```
python -u Clean/gantry_interconnect_dynamic.py                                   # SETTINGS: M_OBC, CUDA, torch.compile
python -u Clean/gantry_interconnect_dynamic.py --arm u --start correct --n-a 0 --no-ps2   # another model
python -u Clean/gantry_interconnect_dynamic.py --evaluate RUN_DIR               # evaluation of a finished run, CPU
```

| File | Role | Thesis |
|-|-|-|
| `gantry_interconnect_dynamic.py` | all settings; fresh run, resume, polish-only rerun, evaluation | Models; Hyperparameters |
| `run_grey_box.sh` | SLURM array over the nine grey-box runs of the thesis | Models |
| `gantry_dynamic/config.py` | run configuration: the record of the settings, their checks, config.json, detuned start | |
| `gantry_dynamic/data.py` | records of one data set, split, resampling to the training rate, normalisation | Records; Processing and checks |
| `gantry_dynamic/controller.py` | the known controller K_fb per stage axis, Tustin-discretised | Closed loop and excitation |
| `gantry_dynamic/model.py` | builds the augmented model and its encoder; least-squares start of the encoder | Augmented model; Initialisation |
| `gantry_dynamic/obc_gantry.py` | gantry part of OBC: reference tuples, basis Phi_Gamma, correction f_obc | Extension to the gantry |
| `gantry_dynamic/training.py` | Adam with PS, checkpoint selection, L-BFGS polish, checkpoint files | Identification |
| `gantry_dynamic/evaluation.py` | NRMS_e of the augmented model and both baselines per split, figures and data | Metrics |
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

## Black box
SUBNET on the servo error e = y - r (deepSI 0.3.29), on the grey box's samples and windows (its `SETTINGS`, noisy
data set). The scripts run from `Clean/` as modules, e.g. `python -u -m blackbox.train_bb`. `run_black_box.sh` trains
the 11 grid configurations on the server (`sbatch --array=9` for the selected one only).

| File | Role | Thesis |
|-|-|-|
| `blackbox/bb.py` | shared parts: model M_BB, the grey box's data settings and loader, free-run prediction | Black-box model |
| `blackbox/train_bb.py` | trains one configuration, weighted loss, selection by validation NRMS_e | Black-box model |
| `blackbox/select_bb.py` | selects the configuration with the lowest validation NRMS_e | Black-box model |
| `blackbox/eval_bb.py` | predictions on every split: `python -u -m blackbox.eval_bb <run folder> <output folder>` | Metrics |
| `blackbox/run_black_box.sh` | SLURM array over the grid | |

## Requirements
`environment.yml` pins the environment the thesis results were produced with (Python 3.11.14, PyTorch 2.5.1 with
CUDA 12.1, deepSI 0.3.29 from its GitHub legacy branch): `conda env create -f environment.yml`. CUDA for full
grey-box runs.

## Attribution
`model_augmentation/` is adapted from Model-Augmentation-Public by Jan H. Hoekstra (commit 8200923, BSD 3-Clause,
see `model_augmentation/LICENSE`). Changes to his code are marked `# CHANGED:`, additions `@added` or
`__project_origin__`. The black-box nets are deepSI's default nets, as in his deepSI black box.

The orthogonal projection in `model_augmentation/fit_systems/obc.py` and `gantry_dynamic/obc_gantry.py` adapts the
method of B. M. Györök et al., "Orthogonal projection-based regularization for efficient model augmentation" (L4DC
2025; code: https://github.com/AIMotionLab-SZTAKI/orthogonal-augmentation, MIT License). Györök adds the projection as
a penalty on the training loss; here the network's part along the baseline's directions is removed by construction.
