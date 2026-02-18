# Scripts Guide (Critical)

This file is a practical and critical map of the Python scripts in this repository.

Scope:
- Explain what each script actually does.
- Clarify where scripts are robust vs brittle/hardcoded.
- Provide execution order for each benchmark family.
- Make adaptation to a new setup explicit.

Legend:
- `data-gen`: generate/simulate datasets
- `train`: train and optionally save models
- `eval`: compute performance and compare models
- `plot`: create diagnostics/figures
- `baseline`: non-augmented reference model path

## Quick Reality Check

- This codebase is **state-space architecture trained mostly from input-output data (`u,y`)**.
- Most shipped experiments are **simulation/benchmark driven**, not plug-and-play for a new physical setup.
- Several scripts have **hardcoded paths/model names**, and some are exploratory.
- `model_augmentation/fit_systems/white_box_models.py` is reusable, but it is **not the main path used by most scripts**.

## Quick Reference Table

| File | Primary role | Key APIs/classes | Usually used by | Risk level |
|---|---|---|---|---|
| `model_augmentation/fit_systems/interconnect.py` | Graph-based hybrid model composition + training wrapper | `Interconnect`, `connect_signals`, `SSE_Interconnect` | Most `scripts/ecc_2025/*`, `scripts/bouc_wen/*`, `scripts/cascaded_tanks/*` | Medium |
| `model_augmentation/fit_systems/blocks.py` | Baseline/augmentation building blocks | `Linear_*`, `Parameterized_*`, `Static_ANN_Block` | All hybrid training scripts | Medium |
| `model_augmentation/fit_systems/pre_encoder.py` | Supervised state pre-encoder path | `System_data_with_x`, `SS_pre_encoder` | `*_pre_encoder.py` scripts | Medium-High |
| `model_augmentation/utils/utils.py` | Routing + normalization helpers | `selection_matrix`, `expansion_matrix`, `normalize_linear_ss_matrices` | Nearly all scripts | Medium |
| `model_augmentation/fit_systems/white_box_models.py` | Standalone white-box templates | `Discrete_White_Box_Model`, `Discrete_Cascaded_Tanks` | Rare in current script pipeline | Low-Medium |
| `scripts/ecc_2025/msd_ndof_data_generation_dynamic.py` | Simulated data generation | MSD simulator setup + save block | ECC workflow start | High (hardcoded settings) |
| `scripts/ecc_2025/msd_ndof_interconnect_dynamic.py` | Main ECC hybrid training | structure flags (`parallel/series`, `dynamic/static`) | ECC core training | High (path/name coupling) |
| `scripts/ecc_2025/msd_ndof_evaluate_fit_systems.py` | ECC model comparison | hardcoded model lists + metrics plots | ECC evaluation | High (manual model list sync) |
| `scripts/bouc_wen/bouc_wen_pre_encoder.py` | Main Bouc-Wen hybrid training | baseline branches + optional pre-encoder | Bouc-Wen core training | High (many branches) |
| `scripts/bouc_wen/bouc_wen_evaluate_fit_systems.py` | Bouc-Wen comparison | `test_type` toggle + model list | Bouc-Wen evaluation | Medium-High |
| `scripts/cascaded_tanks/interconnect_state_aug.py` | Cascaded-tanks augmentation training | encoder reuse + custom wiring | Cascaded-tanks training | Medium-High |
| `scripts/cascaded_tanks/test.py` | Utility/demo plotting | exploratory snippets | Manual experiments | High (non-production script) |

## Start Here (By Benchmark)

- `ecc_2025` (MSD):
  1. `scripts/ecc_2025/msd_ndof_data_generation_dynamic.py`
  2. `scripts/ecc_2025/msd_ndof_interconnect_dynamic.py`
  3. `scripts/ecc_2025/msd_ndof_deepSI_encoder.py`
  4. `scripts/ecc_2025/msd_ndof_evaluate_fit_systems.py`
  5. `scripts/ecc_2025/msd_ndof_state_comparison.py`

- `bouc_wen`:
  1. `scripts/bouc_wen/bouc_wen_pre_encoder.py`
  2. `scripts/bouc_wen/bouc_wen_ANN_SS.py`
  3. `scripts/bouc_wen/bouc_wen_evaluate_fit_systems.py`
  4. `scripts/bouc_wen/bouc_wen_state_comparison.py`

- `cascaded_tanks`:
  1. `scripts/cascaded_tanks/interconnect_pre_encoder.py`
  2. `scripts/cascaded_tanks/interconnect_state_aug.py`
  3. `scripts/cascaded_tanks/interconnect_evaluate_fit_sys.py`
  4. `scripts/cascaded_tanks/test.py` (utility/demo)

## Core Modules (What matters most)

- `model_augmentation/fit_systems/interconnect.py`
  - Dynamic signal-graph interconnect engine.
  - Handles block wiring, forward order, algebraic-loop detection, and output/state propagation.
  - `SSE_Interconnect` wraps deepSI encoder training around the interconnect.
  - Includes parameter regularization terms in loss for parameterized FP blocks.

- `model_augmentation/fit_systems/blocks.py`
  - Library of baseline and augmentation blocks:
    - baseline: `Linear_State_Block`, `Linear_Output_Block`, parameterized variants
    - augmentation: `Static_ANN_Block`
    - physics-specific blocks for MSD/cascaded tanks

- `model_augmentation/fit_systems/pre_encoder.py`
  - Optional pre-encoder path using known states (`x`) where available.

- `model_augmentation/utils/utils.py`
  - Important helpers: `selection_matrix`, `expansion_matrix`, normalization utilities.

- `model_augmentation/fit_systems/white_box_models.py`
  - Standalone white-box model classes (currently cascaded tanks).
  - Useful as a template for custom discrete FP models.
  - Not the dominant training path in the provided benchmark scripts.

## Script Registry

### ECC 2025 (Mass-Spring-Damper)

#### `scripts/ecc_2025/msd_ndof_data_generation_dynamic.py`
- Category: `data-gen`
- Purpose:
  - Build multisine train/val/test datasets from an MSD simulator.
  - Optionally compare with baseline linear model.
- Inputs required:
  - MSD parameters and baseline `.mat` model files.
- Outputs:
  - `.npz` datasets (save lines currently commented).
- Critical notes:
  - Many values are hardcoded (DOF, amplitudes, lengths).
  - Save block is commented; easy to forget.

#### `scripts/ecc_2025/msd_ndof_interconnect_dynamic.py`
- Category: `train`
- Purpose:
  - Main hybrid model training (baseline FP + ANN augmentation).
- Key switches:
  - `FP_type`: `ideal` or `approximate`
  - `dynamic_aug`: dynamic vs static augmentation
  - `type_aug`: `parallel` vs `series`
  - `linear_parallel`: linear residual option
  - `SNR`, `nf`, `epochs`, `batch_size`
- Outputs:
  - saved model in `models/ecc_corrected/...`
- Critical notes:
  - High coupling to expected directory structure and naming.
  - Easy to train a variant and forget to evaluate matching filename.

#### `scripts/ecc_2025/msd_ndof_pre_encoder.py`
- Category: `train` (pre-encoder + hybrid)
- Purpose:
  - Train state pre-encoder using state labels, then inject into interconnect model.
- Critical notes:
  - Requires `x` in data; not purely I/O mode.
  - Uses staged fit and manual bestfit/encoder replacement.

#### `scripts/ecc_2025/msd_ndof_deepSI_encoder.py`
- Category: `train` (black-box baseline)
- Purpose:
  - Train ANN-only encoder state-space model for comparison.
- Critical notes:
  - Needed for fair ablation against augmented model.

#### `scripts/ecc_2025/msd_ndof_evaluate_fit_systems.py`
- Category: `eval` + `plot`
- Purpose:
  - Load model list, compute RMS/NRMS, plot prediction error and validation losses.
- Critical notes:
  - Model lists are hardcoded; easy mismatch with what is actually trained.
  - Baseline comparison path assumes specific FP model files.

#### `scripts/ecc_2025/msd_ndof_state_comparison.py`
- Category: `plot`
- Purpose:
  - Compare internal state and augmentation signal contributions.
- Critical notes:
  - Mostly diagnostic; not a pass/fail benchmark.

### Bouc-Wen

#### `scripts/bouc_wen/bouc_wen_pre_encoder.py`
- Category: `train`
- Purpose:
  - Main hybrid Bouc-Wen training script with multiple baseline options (`FP`, `FP_nonlin`, `BLA_2`, `BLA_3`).
- Critical notes:
  - Dense script with many branches and assumptions on `.mat` structure.
  - Strongest script for seeing parameterized FP + ANN residual in one place.

#### `scripts/bouc_wen/bouc_wen_ANN_SS.py`
- Category: `train` (black-box baseline)
- Purpose:
  - Train ANN-only state-space benchmark for Bouc-Wen.

#### `scripts/bouc_wen/bouc_wen_evaluate_fit_systems.py`
- Category: `eval` + `plot`
- Purpose:
  - Evaluate candidate models on multisine/sinesweep tests.
- Critical notes:
  - Model lists and test type are manual toggles.

#### `scripts/bouc_wen/bouc_wen_state_comparison.py`
- Category: `plot`
- Purpose:
  - Visualize internal state vs augmentation channels.

### Cascaded Tanks

#### `scripts/cascaded_tanks/interconnect_pre_encoder.py`
- Category: `train/eval` (mostly load-and-check in current form)
- Purpose:
  - Work with encoder baseline model and compare against simulation states.
- Critical notes:
  - Contains significant commented code; acts as mixed notebook-style script.

#### `scripts/cascaded_tanks/interconnect_state_aug.py`
- Category: `train` + `eval`
- Purpose:
  - Train state-augmentation around cascaded-tank baseline and save best model.
- Critical notes:
  - Uses loaded encoder baseline and custom wiring.
  - Validation split handling is unusual (`val_train_split = 0` by default).

#### `scripts/cascaded_tanks/interconnect_evaluate_fit_sys.py`
- Category: `eval` + `plot`
- Purpose:
  - Evaluate saved augmented model and produce paper-like plots.

#### `scripts/cascaded_tanks/test.py`
- Category: `utility/demo`
- Purpose:
  - Signal plotting and exploratory checks.
- Critical notes:
  - Not a production/evaluation script.

## Parameter Cheat Sheet

- `nx`, `nu`, `ny`: baseline state/input/output dimensions.
- `nxd`: modeled state dimension in interconnect (can include extension).
- `na`, `nb`: history lengths for encoder.
- `nf`: simulation horizon in training loss.
- `epochs`, `batch_size`: training schedule.
- `SNR` / `sigma_n`: noise level.
- `FP_type`: idealized vs intentionally mismatched first-principles baseline.
- `type_aug`: `parallel` or `series` augmentation wiring.
- `dynamic_aug`: whether augmentation includes extended state dynamics.

## Decision Rules (Pragmatic)

- Start with `parallel + dynamic_aug=True`.
- Use `series` only if parallel residual cannot remove structured errors.
- Use pre-encoder only when state labels are credible and available.
- Keep FP parameter regularization enabled for identifiability.

## Minimum Validation Checklist

- Report RMS + NRMS on train/val/test.
- Compare against:
  - FP-only baseline
  - ANN-only baseline
  - hybrid variants
- Check long-horizon rollout stability (not just 1-step fit).
- Run at least one extrapolation test (new excitation).

## Common Failure Modes

- Dimension mismatches in signal routing (`selection_matrix`, `expansion_matrix`).
- Wrong model filename/path in evaluation lists.
- Training noise/SNR mismatch between train and eval scripts.
- Hidden assumptions in `.mat` variable names.
- Overfitting with large `nxd` and weak regularization.

## Adaptation Checklist (New System)

1. Define plant: `dt`, `nu`, `ny`, baseline state definition.
2. Create/choose baseline FP model (linear or nonlinear block).
3. Generate or load I/O data with informative excitation.
4. Normalize consistently and check units.
5. Build interconnect with clear FP and augmentation paths.
6. Train hybrid + ANN-only baselines.
7. Evaluate with RMS/NRMS and extrapolation tests.
8. Inspect state/augmentation diagnostics.
9. Lock naming and reproducibility.

## Paper Traceability

- `1.pdf` concepts implemented in this repo:
  - encoder-based identification from I/O data
  - baseline + augmentation interconnection
  - static/dynamic augmentation variants
  - benchmark style comparison on MSD/Bouc-Wen
- Practical implementation here is graph-wiring based (Interconnect) rather than a single explicit LFR matrix object.

## Function and Class Reference (Core Files)

### Signal Flow (Interconnect)

Conceptual path in `model_augmentation/fit_systems/interconnect.py`:

`x_k, u_k -> [connection matrices] -> z_i -> block_i(z_i)=w_i -> ... -> xp_k, y_k`

- `x` and `u` are always signal indices `0` and `1`.
- Each added block introduces one input-side signal (`w_i`) and one output-side signal (`z_i`).
- `selection_matrix(...)` and `expansion_matrix(...)` from `model_augmentation/utils/utils.py` are the main tools to map between full state and FP substate slices.

### `model_augmentation/fit_systems/interconnect.py`

- `Interconnect.__init__` (`model_augmentation/fit_systems/interconnect.py:12`)
  - Does: initializes graph metadata and signal containers.
  - Inputs: `nx, nu, ny`.
  - Common misuse: giving `nx` inconsistent with block wiring dimensions.

- `Interconnect.forward` (`model_augmentation/fit_systems/interconnect.py:41`)
  - Does: computes one-step `y, xp` through current graph.
  - Expects: `x` shaped `(batch,nx)` or `(batch,nx,1)`, `u` shaped `(batch,nu)` or `(batch,nu,1)`.
  - Returns: `y` and `xp` with batch-first shapes.
  - Common misuse: shape mismatch caused by wrong `selection/expansion` matrices.

- `Interconnect.init_forward` (`model_augmentation/fit_systems/interconnect.py:108`)
  - Does: topological scheduling + connection matrix assembly.
  - Critical behavior: detects algebraic loops via `detect_algebraic_loop(...)`.

- `Interconnect.init_connection_matrices` (`model_augmentation/fit_systems/interconnect.py:158`)
  - Does: constructs additive/concatenation routing matrices per output signal.
  - Critical behavior: if only additive links exist, one is auto-converted to concat (`model_augmentation/fit_systems/interconnect.py:186`).
  - Caveat: `add_to` path is not implemented (`model_augmentation/fit_systems/interconnect.py:233`).

- `Interconnect.connect_signals` (`model_augmentation/fit_systems/interconnect.py:261`)
  - Does: registers an edge between signals/blocks.
  - Defaults:
    - internal block outputs (`z_i`) use concat,
    - global outputs (`xp,y`) use additive.

- `SSE_Interconnect` (`model_augmentation/fit_systems/interconnect.py:371`)
  - Does: deepSI encoder wrapper around `Interconnect` dynamics.
  - `loss(...)` (`model_augmentation/fit_systems/interconnect.py:416`): simulation MSE + optional FP parameter regularization.
  - Caveat: regularization only applies to specific parameterized block classes.

### `model_augmentation/fit_systems/blocks.py`

- `Block` (`model_augmentation/fit_systems/blocks.py:10`)
  - Base API for interconnect-compatible modules.
  - Contract: `forward(z)` returns `w`, with dimensions matching `nz -> nw`.

- `Static_ANN_Block` (`model_augmentation/fit_systems/blocks.py:27`)
  - Does: learned static mapping used for residual augmentation.
  - Expects: `z` with width `nz`.

- `Linear_State_Block` / `Linear_Output_Block` (`model_augmentation/fit_systems/blocks.py:53`, `model_augmentation/fit_systems/blocks.py:75`)
  - Do: baseline linear FP state/output equations.
  - Common misuse: swapping state and input order in concatenated `z`.

- `Parameterized_Linear_State_Block` / `Parameterized_Linear_Output_Block` (`model_augmentation/fit_systems/blocks.py:96`, `model_augmentation/fit_systems/blocks.py:137`)
  - Do: trainable FP matrices with regularization toward initial values.
  - `param_loss()` is consumed by `SSE_Interconnect.loss(...)`.

- `Discrete_Nonlinear_Function_Block` (`model_augmentation/fit_systems/blocks.py:177`)
  - Template for physics-inspired nonlinear blocks with `nonlinear_function(z)`.

- `Parameterized_MSD_State_Block` (`model_augmentation/fit_systems/blocks.py:312`)
  - Does: parameterized MSD dynamics and discretization in normalized coordinates.
  - Caveat: contains hardcoded normalization matrices/initial parameters tied to benchmark assumptions.

### `model_augmentation/fit_systems/pre_encoder.py`

- `System_data_with_x` (`model_augmentation/fit_systems/pre_encoder.py:20`)
  - Extends deepSI `System_data` with state history/future windows.
  - Use when supervised state data `x` is available.

- `System_data_norm_with_x` (`model_augmentation/fit_systems/pre_encoder.py:43`)
  - Normalizes `u,y` while carrying `x` through.
  - Caveat: currently implemented only for `System_data_with_x` type.

- `SS_pre_encoder` (`model_augmentation/fit_systems/pre_encoder.py:106`)
  - Does: trains encoder state estimate against provided `x` labels.
  - `loss(...)` (`model_augmentation/fit_systems/pre_encoder.py:138`) currently asserts `nf=1` behavior.
  - Caveat: `n_step_error(...)` implementation is partial/debug-like and not a robust reporting utility.

### `model_augmentation/utils/utils.py`

- `selection_matrix(ix,n)` (`model_augmentation/utils/utils.py:153`)
  - Returns matrix selecting indexed components from a full vector.
  - Typical use: feed FP block only the physical substate.

- `expansion_matrix(ix,n)` (`model_augmentation/utils/utils.py:148`)
  - Returns matrix expanding a subvector into full-state coordinates.
  - Typical use: add FP output back into `xp` channels.

- `normalize_linear_ss_matrices(...)` (`model_augmentation/utils/utils.py:128`)
  - Normalizes `A,B,C,D` based on data statistics.
  - Caveat: requires representative data scaling, otherwise FP baseline is distorted.

- `detect_algebraic_loop(...)` (`model_augmentation/utils/utils.py:22`)
  - Used in interconnect graph validation.

### `model_augmentation/fit_systems/white_box_models.py`

- `Discrete_White_Box_Model` (`model_augmentation/fit_systems/white_box_models.py:9`)
  - Base class for discrete FP models with `f(x,u)` and `h(x,u)` API.
  - `forward(...)` (`model_augmentation/fit_systems/white_box_models.py:18`) reshapes and returns `y,xp`.

- `Discrete_Cascaded_Tanks` (`model_augmentation/fit_systems/white_box_models.py:39`)
  - Implements cascaded tanks FP model with trainable parameters and normalization.
  - Caveat: assumes model form and scaling structure specific to tank benchmark.

## High-Risk Caveats (Read Before Editing)

- Hardcoded benchmark normalization/parameter defaults exist in some blocks.
- Some scripts are notebook-like and include commented branches that are easy to misuse.
- Filename conventions in evaluation scripts are manual; mismatches are common.
- `add_to` connection mode exists in API but is not implemented in matrix assembly.
- `white_box_models.py` is valid, but most training scripts rely on `Interconnect + blocks` path.

## ECC Repro Runbook (Practical)

Use this runbook when you need a repeatable baseline for `scripts/ecc_2025`.

1. Generate data:
   - `scripts/ecc_2025/msd_ndof_data_generation_dynamic.py`
   - Ensure save lines are enabled (train/val/test `.npz`).
2. Train hybrid model:
   - `scripts/ecc_2025/msd_ndof_interconnect_dynamic.py`
   - Pick one variant first: `type_aug="parallel"`, `dynamic_aug=True`.
3. Train ANN-only baseline:
   - `scripts/ecc_2025/msd_ndof_deepSI_encoder.py`
4. Evaluate all trained models:
   - `scripts/ecc_2025/msd_ndof_evaluate_fit_systems.py`
   - Sync `fit_sys_file_name_list` with actually saved model names.
5. Run diagnostics:
   - `scripts/ecc_2025/msd_ndof_state_comparison.py`

Done criteria:
- Train/val/test data files exist in `data/mass_spring_damper`.
- At least one hybrid + one ANN-only model are saved.
- RMS and NRMS are reported for baseline and trained models.
- Loss curves and prediction-error plots are generated.

## ECC Config Matrix (Most Used Flags)

| Script | Flag | Typical value | Notes |
|---|---|---|---|
| `scripts/ecc_2025/msd_ndof_interconnect_dynamic.py` | `FP_type` | `approximate` | `ideal` and `approximate` are both used in experiments; keep eval path aligned. |
| `scripts/ecc_2025/msd_ndof_interconnect_dynamic.py` | `dynamic_aug` | `True` | Enables state extension beyond baseline FP state size. |
| `scripts/ecc_2025/msd_ndof_interconnect_dynamic.py` | `type_aug` | `parallel` | Start with parallel before series. |
| `scripts/ecc_2025/msd_ndof_interconnect_dynamic.py` | `linear_parallel` | `True/False` | `True` gives linear residual branch, `False` uses nonlinear activation. |
| `scripts/ecc_2025/msd_ndof_interconnect_dynamic.py` | `SNR` | `20/30/60` | Keep train/eval SNR consistent. |
| `scripts/ecc_2025/msd_ndof_interconnect_dynamic.py` | `nf` | `200` | Simulation horizon in loss; affects runtime and stability. |
| `scripts/ecc_2025/msd_ndof_interconnect_dynamic.py` | `epochs` | `3000` | Paper-style ECC runs use long training; adjust for quick checks. |
| `scripts/ecc_2025/msd_ndof_deepSI_encoder.py` | `epochs` | `10000` | ANN-only baseline is typically trained longer. |
| `scripts/ecc_2025/msd_ndof_data_generation_dynamic.py` | `System_dof` / `FP_dof` | `3` / `2 or 3` | Keep intended mismatch scenario explicit (plant vs baseline). |

## Paper Traceability: `1.pdf` to Code

`1.pdf` (European Journal of Control 2025) maps to this repository as follows.

| `1.pdf` concept | Repo location | Practical meaning |
|---|---|---|
| LFR-style baseline + augmentation interconnection | `model_augmentation/fit_systems/interconnect.py` | Implemented as graph-wiring blocks/signals, not one monolithic LFR matrix object. |
| Static vs dynamic augmentation | `scripts/ecc_2025/msd_ndof_interconnect_dynamic.py` (`dynamic_aug`) | Dynamic adds model states; static does not. |
| Parallel vs series augmentation | `scripts/ecc_2025/msd_ndof_interconnect_dynamic.py` (`type_aug`) | Two augmentation interconnection classes from the paper are directly selectable. |
| Encoder-based I/O identification (SUBNET-style) | `model_augmentation/fit_systems/interconnect.py` (`SSE_Interconnect`) | Encoder estimates latent state from input/output history. |
| Joint FP + augmentation estimation with regularization | `model_augmentation/fit_systems/interconnect.py`, `model_augmentation/fit_systems/blocks.py` | Parameterized FP blocks can be regularized during training. |
| MSD benchmark workflow | `scripts/ecc_2025/*` | Data generation, hybrid training, ANN baseline training, evaluation. |

## Thesis Traceability: Kessels EA (`20250206_Kessels_hf.pdf`)

Chapter 5 introduces Extension and Augmentation-based (EA) model updating:
- Augmentation: learned terms correct existing FP equations.
- Extension: extra states capture missing dynamics/time-scales.
- Encoder: initialize latent/extended states from measured data.
- Training: truncated output prediction error over simulation windows.

Closest mapping in this repo:

| Kessels EA concept | Repo location | Notes |
|---|---|---|
| Retain FP core and add learned corrections | `model_augmentation/fit_systems/blocks.py` + `model_augmentation/fit_systems/interconnect.py` | Grey-box behavior through explicit FP and ANN blocks. |
| State extension for missing dynamics | `scripts/ecc_2025/msd_ndof_interconnect_dynamic.py` (`nxd`, `dynamic_aug`) | Dynamic mode increases modeled state dimension. |
| Encoder-initialized latent state training | `model_augmentation/fit_systems/interconnect.py` (`SSE_Interconnect`) | Same high-level role as EA encoder concept. |
| Black-box comparator | `scripts/ecc_2025/msd_ndof_deepSI_encoder.py` | ANN-only baseline for ablation/comparison. |

Relation to `scripts/ecc_2025/msd_ndof_data_generation_dynamic.py`:
- This script is not EA training itself.
- It creates excitation/response datasets and baseline sanity checks that are consumed by EA-like hybrid training scripts.
- If you want to study extension/augmentation effects, this script is where you control excitation richness, noise setup, and train/val/test consistency.

## Naming and Path Contract (Avoid Common Breaks)

- Data files expected by ECC scripts:
  - `data/mass_spring_damper/msd_3dof_multisine_train.npz`
  - `data/mass_spring_damper/msd_3dof_multisine_val.npz`
  - `data/mass_spring_damper/msd_3dof_multisine_test.npz`
- Saved model roots expected by eval:
  - `models/ecc_corrected/ideal/SNR{SNR}/...`
  - `models/ecc_corrected/approximate/SNR{SNR}/...`
- Figures path expected by eval:
  - `figures/ecc_corrected/`

If you change filenames or folder layout, update:
- `scripts/ecc_2025/msd_ndof_interconnect_dynamic.py` save-name logic.
- `scripts/ecc_2025/msd_ndof_evaluate_fit_systems.py` model list and load paths.
