# Project Status Report

**Project**: LPV-LFR Grey-Box Augmentation for a Dual-Gantry Motion System
**Date**: 2026-06-16
**Repository**: `Baseline-LPV-Augmentation`, branch `Augmentation`

---

## Executive Summary

The **CT LPV-LFR baseline model** of the dual-gantry (physics from Garcia-Herreros et al.) is mathematically derived, implemented in Python/PyTorch, and verified against MATLAB to near-machine-precision. An SVD-reduced LFR realization (6-channel to 4-channel, lossless) is also complete. **Parameter recovery training** (10-13 trainable physical parameters, no ANN) has been implemented, run on synthetic MATLAB data via a custom training pipeline, and converges, though final parameter accuracy has not been reported as a definitive result. **Grey-box augmentation** (dynamic parallel ANN alongside the LFR baseline) is wired into Jan Hoekstra's `SSE_Interconnect` framework with three encoder variants and has been run on a university compute server, but systematic evaluation and comparison are not yet complete. **All data is synthetic** (MATLAB Simulink), generated both with and without a hidden mass-spring-damper disturbance. No real hardware data from the ASMPT dual-gantry has been used. The thesis-critical contributions (orthogonal projection regularization, settling-time cost function, local FRF integration) are at the conceptual/theoretical stage with no implementation.

---

## Repository Map

| Directory | Language | Purpose |
|-----------|----------|---------|
| `kamtin-fp-model/` | MATLAB | **READ-ONLY** reference FP model (Garcia-Herreros). Simulink models + support functions. |
| `LPV/` | LaTeX + MATLAB | LFR derivation documents, well-posedness proof, supervisor feedback, SVD derivation. |
| `Matlab-scripts/` | MATLAB | Data-generation scripts (trajectory export, multisine, augmented-model Simulink). |
| `Matlab-output/` | .mat files | Exported simulation data: trajectories, LPV matrices, parameter-recovery datasets. |
| `model_augmentation/` | Python | Jan Hoekstra's augmentation framework: `Interconnect`, `SSE_Interconnect`, blocks, encoder, utilities. |
| `lpv_lfr_baseline/` | Python | Gantry-specific LPV-LFR baseline: physics, LFR matrices, RK4 simulation, SVD reduction, param recovery. |
| `scripts/gantry/` | Python | Training, validation, encoder, and diagnostic scripts for the gantry system. |
| `scripts/ecc_2025/`, `scripts/bouc_wen/`, `scripts/cascaded_tanks/` | Python | Reference benchmark scripts (MSD, Bouc-Wen, Cascaded Tanks). Not the target system. |
| `data/gantry/` | .mat, .npz | Loaded training/validation data, MATLAB-exported multisine and trajectory datasets. |
| `simulations/` | .pt, .npz, .png, .db | Outputs: param recovery checkpoints, augmentation results, diagnostics plots. |
| `docs/` | Markdown | 53 design decisions (D-001 to D-053), interface docs, experiment design notes. |
| `tasks/` | Markdown | Task tracking (`todo.md`), lessons (`lessons.md`), session handoff (`handoff.md`). |
| `planning/` | Python | Gantt chart script. |
| `literature/` | PDF | Reference papers (Drenth thesis, Hoekstra EJC, etc.). |

**Entry points**:
- `scripts/gantry/gantry_interconnect_dynamic.py` -- main augmentation training script (runs on SLURM)
- `lpv_lfr_baseline/scripts/train_param_recovery.py` -- parameter recovery training (runs as module)
- `lpv_lfr_baseline/scripts/validate_lfr.py` -- 5-plot visual validation of the LFR baseline
- `Matlab-scripts/Augmentation/main_augmentation.m` -- MATLAB Simulink with hidden MSD for data generation

---

## Component Assessment

### 1. Baseline LPV-LFR Derivation and Well-Posedness

**Status**: DONE

**Key files**:
- `LPV/supporting/derivations/LFR-derivation.tex` -- full derivation
- `LPV/supporting/derivations/M-invertibility.tex` -- positive definiteness proof (Sylvester's criterion)
- `LPV/LFR-derivation-supervisor.tex` -- supervisor-reviewed version
- `LPV/LFR-SVD-derivation.tex` -- SVD reduction from 6 to 4 latent channels
- `LPV/supporting/verification/verify_lfr_reduction.m` -- MATLAB algebraic verification

**What exists**:
- CT quasi-LPV model with M(Y) = M0 + M1*Y + M2*Y^2, scheduling variable Y (payload position).
- LFR realization with Delta(Y) = Y*I_6, explicit algebraic loop solved via N(Y)/d(Y) (polynomial adjugate/determinant).
- Well-posedness proved globally (all Y in R) via M(Y) positive definite (Sylvester's criterion), not via rho(D_zw) < 1.
- SVD reduction: rank(D_zw) = 4, so the 6-channel LFR reduces losslessly to a 4-channel LFR.
- All derivation claims verified by an external Claude instance and cross-checked against Drenth thesis Sections 2.1, 2.1.1, 2.2.

**What remains**: Nothing for the derivation itself. The question of whether the baseline LFR should also be realized via the Drenth D_zw = exp(-N) parameterization (vs. the current analytical approach) is an open design decision (D-036).

---

### 2. Python LFR Implementation and Simulation

**Status**: DONE

**Key files**:
- `lpv_lfr_baseline/core/physics.py` -- physical constants as torch.float64 tensors, M0/M1/M2 decomposition
- `lpv_lfr_baseline/core/lfr_matrices.py` -- G matrix construction (Ax, Bw, Bu, Cz, Dzw, Dzu, Cy)
- `lpv_lfr_baseline/core/lfr_forward.py` -- genuine LFR-first forward pass (6-step signal flow)
- `lpv_lfr_baseline/core/lfr_simulate.py` -- RK4 integration with BPTT mode control, gradient checkpointing
- `lpv_lfr_baseline/blocks/lfr_block.py` -- Jan-compatible Block wrapper (nz=9, nw=18)
- `lpv_lfr_baseline/svd/` -- complete SVD-reduced variant (4-channel, lfr_svd_block.py, lfr_svd_simulate.py)
- `lpv_lfr_baseline/tests/test_jan_compat.py` -- integration tests for Interconnect wiring

**How D_zw / algebraic loop is handled**: The loop is solved analytically, not iteratively. `lfr_forward.py` computes z = L(Y)^{-1} * rhs via N(Y)/d(Y) (Horner evaluation of the polynomial adjugate). D_zw != 0 is the structural reason for the loop; well-posedness is guaranteed by M(Y) > 0. The state update then flows through G (xdot = Ax@x + Bw@w + Bu@u), preserving the LFR structure.

**Algebraic loop in Jan's framework**: The Interconnect's `detect_algebraic_loop` check passes because Y is extracted from the state (x[:,2]), not routed as a named external signal. This avoids a feedback loop in the block graph.

**Verification results**:
- Python RK4 vs MATLAB ode45: BFR 99.99%+ on all channels (sub-nanometre residual)
- LPV vs frozen LTI comparison demonstrates scheduling benefit
- Integration tests (test_jan_compat.py): 8 checks covering shapes, gradients, algebraic loop detection, z_lfr slot access, augmentation wiring, and value correctness

**What remains**: torch.compile optimization deferred due to hardware constraints (needs Volta+ GPU, D-040).

---

### 3. Gantry State Block in the Augmentation Framework

**Status**: DONE

**Key files**:
- `model_augmentation/fit_systems/blocks.py` (lines 639-820) -- `Gantry_State_Block`
- `model_augmentation/systems/gantry_ss.py` -- physical constants + LFR matrix builders (float32 variant)
- `model_augmentation/systems/gantry_linearization.py` -- frozen-Y linearization for encoder init

**What exists**: `Gantry_State_Block` is a `Discrete_Nonlinear_Function_Block` subclass that performs CT RK4 integration using the LFR rational structure. Supports both frozen-Y (Phase 1/2, precomputed N_op/d_op) and LPV self-scheduling (Y_op=None, Horner per step). Handles denormalization/renormalization internally. Upsampling factor configurable.

**Note**: There are two copies of the physics: `lpv_lfr_baseline/core/physics.py` (float64, for standalone LFR work) and `model_augmentation/systems/gantry_ss.py` (float32, for the augmentation framework). Both mirror main.m exactly but diverge in dtype.

---

### 4. Parameter Identification (Trainable Physical Parameters)

**Status**: IN PROGRESS (pipeline works, results not finalized)

**Key files**:
- `lpv_lfr_baseline/blocks/lfr_param_block.py` -- `ParameterizedLFRBlock` with 13 trainable scalars via log/exp reparameterization
- `lpv_lfr_baseline/blocks/lfr_fit_system.py` -- `LFRFitSystem` (SSE_Interconnect subclass with generic param_loss)
- `lpv_lfr_baseline/scripts/train_param_recovery.py` -- training script with windowed BPTT, multi-trajectory, sigma-normalized loss
- `lpv_lfr_baseline/scripts/precompute.py` -- precomputes fixed data (trajectories, sigma, segment pools)
- `lpv_lfr_baseline/scripts/data_utils.py` -- data loading utilities
- `lpv_lfr_baseline/scripts/analyze_param_recovery.py`, `eval_param_recovery.py`, `parameter_evaluation.py` -- analysis scripts

**What exists**:
- 13 trainable physical parameters (masses, damping, stiffness, inertias, geometry d) with log/exp positivity guarantee.
- Detuned initialization (2-10% off true values).
- Split regularization for degenerate parameter pairs (kb1+kb2, cb1+cb2, Jb+Jh).
- Multiple dataset variants: `base` (6 trajectories, no multisine), `base_extended_6mm` (8 trajectories, 6mm range), `multisine`, `ref_injection`.
- Training outputs in `simulations/param_recovery/`: checkpoints up to 600 epochs, convergence plots, diagnostic outputs.
- Per-channel sigma normalization (D-042), multi-trajectory binary masking (D-044).

**What remains**:
- Final parameter accuracy report (how close recovered parameters are to true values).
- Go/no-go decision for augmentation: parameter recovery was meant to be the gate (D-023), but the augmentation pipeline has been developed in parallel.
- Measurement noise addition (deferred, noted in todo.md).

---

### 5. Grey-Box Augmentation (ANN + LFR Baseline)

**Status**: IN PROGRESS (wired and running, not systematically evaluated)

**Key files**:
- `scripts/gantry/gantry_interconnect_dynamic.py` -- main training script (configurable: mode, encoder, augmentation)
- `model_augmentation/fit_systems/interconnect.py` -- Jan's `Interconnect` + `SSE_Interconnect`
- `model_augmentation/fit_systems/blocks.py` -- `Static_ANN_Block` (parallel augmentation)
- `model_augmentation/fit_systems/pre_encoder.py` -- `linear_encoder_init` (reconstructability-based, Hoekstra 2026 Eq. 16-17)
- `model_augmentation/utils/torch_nets.py` -- `LinearInitEncoderWrapper`, `HybridGantryEncoder`

**Architecture**:
- Dynamic parallel: `Gantry_State_Block` (physics, 6 states) + `Static_ANN_Block` (learned, NX_ANN states) wired additively through Jan's Interconnect.
- Three encoder variants: `linear_map` (reconstructability init, preferred), `hybrid` (analytical physical + learned augmented), `default` (standard deepSI).
- Data: multisine or trajectory data from MATLAB Simulink (see Section 7).
- Training: Adam optimizer via SSE_Interconnect.fit(), rollout horizon nf, batch_size configurable.
- Optuna hyperparameter search infrastructure present (USE_OPTUNA flag).

**What has been run** (from server logs in `data/gantry/server-logs/`):
- Multiple runs with different configurations (job IDs 65555, 66090-66093, 66495).
- Encoder validation pipeline (step0_init, step1_baseline, step2_msd).
- Initial sim-RMS improving from ~0.28 to ~0.09 within 6 epochs (from log).

**What remains**:
- Step 5 systematic evaluation plan (drafted in todo.md but not executed):
  - Phase 1: Hyperparameter sweeps (fs, nf, na_nb)
  - Phase 2: Baseline mismatch quantification (NX_ANN=0)
  - Phase 3: Controlled 2x2 comparison (encoder type x data type)
  - Phase 4: Analysis and conclusions
- Baseline-vs-augmented comparison with proper metrics and statistical rigor.
- State recovery analysis (R2_raw, R2_linmap per channel) -- diagnostic code exists but not systematically applied.
- Encoder off-by-one bug was found and fixed (D-053), but post-fix training results not reported.

---

### 6. Interpretability and Regularization

**Status**: NOT STARTED (theoretical concepts only)

**What exists**:
- Design decision D-003: augmentation structure is parallel dynamic LFR, chosen specifically for compatibility with orthogonal projection regularization (Gyorok et al.).
- Novelty 1 (todo.md): orthogonal projection regularization. Supervisor note: "extra states not really thought about yet, will also need theoretical development."
- The z_lfr/w_lfr latent signals are explicitly routed in the Interconnect (test_jan_compat.py Check 5), so augmentation has structural access to the LFR feedback path.

**What remains**: All implementation. The theory for projecting onto the complement of the baseline's column space has not been formalized. No code exists for this.

---

### 7. Data: Datasets, Sources, and Gaps

**Status**: Synthetic data complete; no real hardware data

All training and validation data is generated by MATLAB Simulink simulations. There are two model variants:

| Dataset type | MATLAB model | Hidden MSD? | Location |
|---|---|---|---|
| Baseline trajectories | `kamtin-fp-model` (Garcia-Herreros) | No | `Matlab-output/parameter-recovery/` (T1-T6) |
| Extended baseline | Same | No | `Matlab-output/identification-trajectories-no-multisine-6mm/` (T1-T8 + V1 + E1) |
| Augmented (with MSD) | `Matlab-scripts/Augmentation/main_augmentation.m` | Yes (2-DOF, Dahl friction) | `data/gantry/matlab/` (trajectories + multisine subdirs) |
| Multisine (augmented) | Same + multisine injection | Yes | `data/gantry/matlab/multisine/` (T1-T8, V1, E1 + baseline/ + m50/) |
| LPV matrix sweep | `export_lpv_matrices.m` | No | `Matlab-output/lpv_matrices.mat` |
| Varying-Y simulation | `export_lpv_sim.m` | No | `Matlab-output/lpv_sim_varying_y.mat` |

**The hidden MSD** (design decision D-038): extends the 3-DOF gantry to 4-DOF (8 states) by adding a hidden mass `ma` attached to the payload via a spring-damper. Implemented in `Matlab-scripts/Augmentation/gantrySystemExtended.m`. Parameters: ma = 0.1 * mh = 1.01 kg, ka = 500 N/m, ca = 2 Ns/m, L0 = 0.1 m (equilibrium offset). The extra state `delta_a` (relative displacement of ma from mh) enters the mass matrix M(Y, delta_a), coupling all DOFs. This is the synthetic "unmodeled dynamics" the augmentation must learn. The Simulink model has Coulomb friction disabled (`cc1`, `cc2`, `ccy` set but not active).

**Real gantry data**: One ETEL monitoring log exists (`data/gantry/iterETEL.log`, 16429 rows at ~0.1s, ~1666s duration). This contains setpoints and current commands from the BHL (left beam head), not the high-rate position/force signals needed for system identification. **No encoder-resolution position data or force data from the ASMPT dual-gantry hardware is present in the repository.** The entire pipeline operates on MATLAB simulation data.

**m50 variant**: A subdirectory `data/gantry/matlab/multisine/m50/` contains datasets dated 2026-06-16 (today), suggesting an additional model variant with a different hidden mass (50 kg?). This is the most recent data generation.

---

### 8. Validation and Comparison Scripts

**Status**: Partial

| Script | Purpose | Status |
|--------|---------|--------|
| `lpv_lfr_baseline/scripts/validate_lfr.py` | 5-plot LFR baseline validation (trajectory, Bode, nat. freq., LPV vs frozen) | Done, runnable |
| `lpv_lfr_baseline/scripts/plot_lpv_vs_frozen.py` | LPV scheduling benefit visualization | Done |
| `lpv_lfr_baseline/scripts/analyze_param_recovery.py` | Parameter convergence analysis | Done |
| `lpv_lfr_baseline/scripts/eval_param_recovery.py` | Checkpoint evaluation | Done |
| `scripts/gantry/gantry_baseline_validation.py` | Baseline simulation validation | Exists |
| `scripts/gantry/gantry_state_comparison.py` | State trajectory comparison | Exists |
| `scripts/gantry/gantry_evaluate.py` | Post-training evaluation | Exists |
| `scripts/gantry/encoder/encoder_baseline_standalone.py` | Encoder standalone on baseline data | Tested locally |
| `scripts/gantry/encoder/encoder_msd_standalone.py` | Encoder standalone on MSD data | Written, not fully evaluated |
| `scripts/gantry/encoder/step1_baseline_equals_system.py` | Encoder regression to baseline states | Exists |
| `scripts/gantry/verification/diagnose_dynamic_parallel.py` | Dynamic parallel augmentation diagnostic | Exists |
| `scripts/gantry/verification/encoder_state_recovery.py` | State recovery R2 diagnostic | Exists |

**Metrics used**: sim-RMS (primary, per-channel), BFR (best fit rate, per-channel), NRMS, R2 (raw and via linear map), max|error|.

**What remains**: No script exists that produces a single unified "baseline vs augmented" comparison table or figure with proper train/val/test splits and multiple seeds.

---

### 9. SVD Reduction

**Status**: DONE

**Key files**:
- `lpv_lfr_baseline/svd/lfr_svd_reduction.py` -- two-stage SVD (Method 2 from LFR-SVD-derivation.tex)
- `lpv_lfr_baseline/svd/lfr_svd_block.py` -- Jan-compatible `LFRReducedBlock` (nz=9, nw=14)
- `lpv_lfr_baseline/svd/lfr_svd_simulate.py` -- RK4 simulation with reduced G
- `lpv_lfr_baseline/svd/lfr_svd_forward.py` -- reduced LFR forward pass

**What exists**: Complete lossless reduction from 6-channel to 4-channel LFR. The reduced block is a drop-in replacement for `LFRBaselineBlock`. Derivation in `LPV/LFR-SVD-derivation.tex`.

**What remains**: The reduced block is not yet used in the main augmentation pipeline (`gantry_interconnect_dynamic.py` uses `Gantry_State_Block`, not `LFRReducedBlock`). No comparison of training speed or accuracy between full and reduced has been done.

---

## Dependencies and Critical Path

### What can run today (simulation only)

```
MATLAB Simulink → .mat data files → Python training pipeline → results
```

1. **LFR baseline validation** (`validate_lfr.py`): standalone, needs only `Matlab-output/*.mat`.
2. **Parameter recovery** (`train_param_recovery.py`): needs `Matlab-output/parameter-recovery*/` or `identification-trajectories*/`.
3. **Augmentation training** (`gantry_interconnect_dynamic.py`): needs `data/gantry/matlab/` datasets.
4. **Encoder standalone validation**: needs `data/gantry/matlab/` and `data/gantry/baseline_simulations/`.
5. **All MATLAB data generation scripts**: need MATLAB with Simulink + `kamtin-fp-model/`.

### What blocks what

| Blocked item | Blocked by | Notes |
|---|---|---|
| Augmentation evaluation (Step 5) | Systematic sweep runs on compute server | Plan drafted, not executed |
| Orthogonal projection regularization | Theoretical development | No implementation exists |
| Settling-time cost function | Design + implementation | Not started |
| Local FRF integration | Experiment design + theory | Not started |
| Hardware validation | Real gantry data | No data in repo |
| Measurement noise study | Parameter recovery on clean data first | Explicitly deferred |
| torch.compile speedup | Volta+ GPU | Current P2000 is Pascal (CC 6.1) |

### Hardware/gantry data dependency

**Everything in the current pipeline runs on synthetic MATLAB data.** The transition to real data requires:
1. High-rate position measurements from gantry encoders (X1, X2, Y at >= 4 kHz).
2. Corresponding force/current measurements (F_X1, F_X2, F_Y).
3. Adaptation of data loading code (currently expects `.mat` files with specific field names: `u`, `u_total`, `y`, `q1`, `q_simscape`).
4. Re-computation of normalization statistics from real data.
5. The controller model (Cfb) may need updating if the real controller differs from the simulated one.

---

## Hardware and Data Risk

### What cannot be done without the gantry

- Validating whether the augmentation learns the correct real-world unmodeled dynamics (as opposed to the synthetic MSD).
- Determining if the hidden MSD model (D-038) is a reasonable proxy for the actual flexible-mode coupling.
- Settling-time cost function validation on real closed-loop data.
- Local FRF experiments (Novelty 3) -- requires on-machine probing.
- Final thesis claims about real-world performance improvement.

### What the synthetic data can cover

- Full augmentation pipeline development and debugging.
- Parameter recovery validation (can the optimizer find the right physics?).
- Encoder design evaluation (which initialization strategy works best?).
- Augmentation architecture comparison (NX_ANN, network size, encoder type).
- Regularization development and testing.
- All figures for the "method" chapters of the thesis.
- Proof-of-concept: augmented model outperforms baseline on synthetic data where the ground truth is known.

### Risk level

**HIGH**: The thesis claims grey-box augmentation for a dual-gantry system, but all results to date are on simulated data from a model augmented with a synthetic disturbance. If real gantry data becomes available late, there may not be time to debug the data pipeline, re-tune hyperparameters, and produce convincing results. The synthetic MSD may not be representative of the real unmodeled effects.

---

## TODOs, FIXMEs, and Incomplete Code

### In Python files

| File | Line | Type | Note |
|------|------|------|------|
| `model_augmentation/fit_systems/blocks.py` | 635 | TODO | "Not certain we need the output block" |
| `model_augmentation/fit_systems/blocks.py` | 600-610 | Commented | Large block of commented-out MSD state equations |
| `model_augmentation/systems/gantry_linearization.py` | 49-50 | NotImplementedError | `Y_op != 0.0` linearization not implemented |
| `scripts/gantry/gantry_interconnect_dynamic.py` | ~62 | Config | `USE_OPTUNA = False` -- Optuna infrastructure present but disabled |
| `scripts/gantry/encoder_initialisation/interconnect_fit.py` | 82 | TODO | "automatically create snr dataset if it does not exist yet" |
| `scripts/gantry/encoder_initialisation/interconnect_fit.py` | 91-92 | TODO | "add nonlinear_map" and "add noisy_map" encoder types not implemented |
| `scripts/gantry/encoder_initialisation/interconnect_fit.py` | 208 | TODO | "pass physical parameters into nonlinear msd block" (typo in original) |
| `docs/lfr-structure.md` | 252 | TODO | "Determining eta for the gantry (TODO)" -- not documented |
| `scripts/journal_model_augmentation/msd_ndof_data_generation.py` | 48 | BLOCKING | "The LPF input MSD system has a bug in it" -- entire code path disabled (benchmark only, not gantry) |
| `lpv_lfr_baseline/scripts/experiment_diagnostics_old.py` | whole file | Old version | Superseded by `experiment_diagnostics.py` |
| `model aug-wafer/tests/test_utils.py` | whole file | STUB | Contains only `print("Hello world")` |

### Dual physics modules (not a bug, but a maintenance risk)

Two files define the same physical constants from `main.m`:
- `model_augmentation/systems/gantry_ss.py` -- float32, used by `Gantry_State_Block` in the augmentation pipeline
- `lpv_lfr_baseline/core/physics.py` -- float64, used by the standalone LFR baseline and parameter recovery

Both must stay in sync with `kamtin-fp-model/03 Simulink gantry/main.m`. A parameter change in one but not the other would produce silent inconsistencies.

### In task tracking

| Item | Status | Notes |
|------|--------|-------|
| Task 2.4 (Python vs MATLAB matrix comparison) | Unchecked in todo.md | Likely done (BFR 99.99% reported) but not marked |
| Task 2.5 (rectangular approximation error) | Unchecked | Possibly superseded by CT+RK4 decision |
| Task 3.0 (CT model write-up) | Unchecked | No `docs/ct-model-writeup.md` found |
| Task 3.1 (M matrix invertibility validation) | Unchecked | Analytical proof exists in LPV/ but no `validate_rank_m.py` |
| Task 3.2 (RK4 integrator) | Unchecked in todo | Implementation exists in blocks.py and lfr_simulate.py |
| Task 3.3 Blocker A (LFR discretization paper) | Unchecked | Superseded by CT+RK4 approach |
| Task 3.3 Blocker B (rational LFR realization) | Partially resolved | Analytical realization exists |
| Task 3.4-3.7 (LFR structure, wiring, augmentation) | Mixed | Implementation exists but tasks not formally completed in todo.md |
| Step 5 (systematic evaluation) | Not started | Plan drafted, no runs |
| Gantt chart (MEET-01) | Exists (`planning/gantt_chart.py`) but unknown if presented |
| Novelty 1-3 | Not started | Theory only |

### Untracked/throwaway files at top level

`_smoke_test.py`, `_smoke1.py`, `test.py`, `temp_extract.py`, `throwaway_optuna_output.md`, `dynamic_parallel_output.md`, `frf_*.png`, `poles_augmented.png` -- likely temporary/diagnostic artifacts that should be cleaned up.

---

## Completeness Estimates per Component

| Component | Completeness | What remains |
|-----------|-------------|--------------|
| CT LPV-LFR derivation | 100% | Nothing |
| Well-posedness proof | 100% | Nothing |
| SVD reduction | 100% | Not yet used in training pipeline |
| Python LFR implementation | 95% | torch.compile deferred; two physics.py copies (float32/float64) |
| Parameter recovery pipeline | 75% | Final results not reported; noise study deferred; go/no-go gate not passed |
| Augmentation pipeline wiring | 90% | Works end-to-end but not systematically evaluated |
| Encoder initialization | 85% | Three variants exist; off-by-one fixed; systematic comparison not done |
| Systematic augmentation evaluation | 10% | Plan drafted (todo.md Step 5), no runs completed |
| Orthogonal projection regularization | 5% | Design decision made; no code |
| Settling-time cost function | 0% | Concept only |
| Local FRF integration | 0% | Concept only |
| Baseline vs augmented comparison figures | 15% | Scripts exist; no unified comparison artifact |
| Real hardware data integration | 0% | No real data in repo |
| MATLAB data generation | 90% | Multiple dataset variants; m50 variant being generated |
| Documentation (decisions.md) | 90% | 53 decisions logged; some open (D-036) |

---

## Open Questions for Supervisors

1. **Go/no-go on parameter recovery**: The parameter recovery pipeline (Step 3b) was intended as the gate before augmentation (D-023), but augmentation development proceeded in parallel. Is the parameter recovery result sufficient, or does it need to be finalized first?

2. **Real gantry data timeline**: When will high-rate encoder + force data from the ASMPT gantry be available? The entire pipeline currently runs on synthetic MATLAB data. Late arrival of real data is a significant risk.

3. **Hidden MSD model fidelity**: The synthetic augmentation target (D-038: Dahl friction with position-dependent stiffness) may not be representative of the real unmodeled flexible-mode dynamics. Has the fidelity of this proxy been assessed?

4. **Orthogonal projection theory**: Supervisor noted (2026-03-20) that the extra states introduced by augmentation are "not really thought about yet." Has there been progress on the theoretical development needed before implementation?

5. **LFR vs state-space decision (D-036)**: Open question whether to commit to LFR structure (needed for mu-synthesis control design, valued by ASMPT) or accept state-space-only (simpler, sufficient for identification). This affects the augmentation architecture.

6. **SVD reduction in training**: The 4-channel reduced LFR is derived and implemented but not used in training. Should it replace the 6-channel version? This could reduce computational cost and the augmentation network's input dimension.

7. **Scope of thesis results**: Given that all data is synthetic, what is the minimum acceptable result for the thesis? Is a simulation-only proof-of-concept acceptable, or is hardware validation required for a passing grade?

8. **Settling-time cost function and local FRF integration**: These were identified as research novelties (2026-03-20). Are they still in scope given the current timeline, or should effort focus on completing the augmentation evaluation on synthetic data?

9. **Multi-seed statistical evaluation**: The current pipeline runs single seeds. For thesis-quality results, how many random initializations are needed? Jan's reference approach uses 100 random inits with best BFR on validation.

10. **Encoder architecture for augmented states**: The `LinearInitEncoderWrapper` uses two separate sub-networks (physical + augmented) with no shared layers, meaning augmented states get no gradient from standalone training (handoff.md). Should the architecture be changed, or is pipeline-only training of augmented states acceptable?
