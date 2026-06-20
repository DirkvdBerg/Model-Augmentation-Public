# Session Handoff — Encoder Standalone Validation + Augmented State Discussion

**Last written**: 2026-06-14

---

## Discussion to continue

**Question**: Can `NX_ANN=2` (augmented states) outperform `NX_ANN=0` in standalone encoder training (no pipeline)?

**Current conclusion**: No, because of the architecture. The `LinearInitEncoderWrapper` has two **separate** subnetworks:
- `phys_encoder` (linear_encoder_init): outputs 6 physical states, gets gradient from state MSE loss
- `ann` (zero_init_feed_forward_nn): outputs NX_ANN augmented states, gets **no gradient** because the loss is only on physical states

Since they share no parameters, the augmented states stay at ~0.

**Options for making 6+2 meaningful in standalone**:
1. **Supervise augmented states on delta_a** — user does NOT want this (we don't know delta_a on real system)
2. **Single shared-layer network** with 8 outputs — then loss on 6 physical states also trains hidden layers, and the 2 extra outputs get indirect gradient through shared representations
3. **I/O loss in the pipeline** — this is how Jan does it. The state propagation uses all 8 states, output prediction depends on propagated state, and backprop reaches augmented states indirectly.

**User wants**: to compare 6+2 vs 6 and see if extra states help. This comparison only makes sense in the pipeline (option 3), OR if the architecture is changed (option 2).

---

## How Jan uses the encoder initialization

### Reference code
`scripts/gantry/encoder_initialisation/interconnect_fit.py` (lines 318-530)

### Jan's pipeline flow:
1. **Build linear_encoder_init** from normalized (Ad_bar, Bd_bar, Cd_bar, Dd_bar)
2. **Optionally pre-train** via `SS_pre_encoder` (Eq. 35): minimize `||encoder(u_hist, y_hist) - x_baseline||^2`
   - Uses JAX static model for optimization (adam 50-200 epochs + L-BFGS 200 epochs)
   - `x_baseline` comes from **forward simulation of the FP model** (NOT from measurements)
   - Copies trained weights back into a PyTorch encoder
3. **Inject** into `SSE_Interconnect`: `fit_sys.encoder = initialised_encoder`
4. **Train full pipeline** via `fit_sys.fit()` with I/O loss (nf-step-RMS)
   - This is where augmented states get indirect gradient
   - Validation measures: 1-step, 5-step, 20-step, 50-step, 200-step RMS

### Key: Jan's encoder has `nx_model` outputs total
- In his MSD example: `nx_model = 4` (2 physical + 2 augmented for MSD)
- The `linear_encoder_init` is built for ALL 4 states because the linearized system includes MSD states
- His linearization already includes the MSD as part of the model → so the reconstruction map handles all states

### Our situation is different:
- Our linearization is the gantry ONLY (6 states, no MSD)
- The MSD is the unknown disturbance we want to capture
- `linear_encoder_init` only knows about 6 gantry states
- The 2 augmented states have no model-based initialization available

### Architecture details

**`linear_encoder_init`** (pre_encoder.py:191-300):
- ResNet: `x = Wb_psi_y @ y_hist + Wb_psi_u @ u_hist + psi_tilde(y_hist, u_hist)`
- `Wb_psi_y = A^n @ O_n^{-1}` — from reconstructability theory (THEORY: Hoekstra 2026 Eq. 16-17)
- `Wb_psi_u = -A^n @ O_n^{-1} @ Gamma_n + gamma_n` — input correction
- `psi_tilde` = zero-init MLP for nonlinear corrections (starts at 0, so init is purely linear)
- All weights are `nn.Parameter` — fully trainable in pipeline

**`LinearInitEncoderWrapper`** (torch_nets.py:244-338):
- `phys_encoder`: linear_encoder_init instance (6 outputs)
- `ann`: zero_init_feed_forward_nn (NX_ANN outputs) — **separate network, no shared layers**
- Convention fix: adds/subtracts mean offsets to bridge pipeline↔pure-scaled conventions
- Forward: `cat(phys_encoder(u,y), ann(u,y))`

---

## What was done this session

### Encoder standalone scripts created and tested

| Script | Data | NX_ANN | Status |
|--------|------|--------|--------|
| `encoder_baseline_standalone.py` | baseline (no MSD) | 0 | TESTED locally, works |
| `encoder_msd_standalone.py` | MSD | 2 | Written, not yet run |

### Baseline result (local run):
- Velocities: encoder beats analytical (dq2: 5x better, dq3: 8x better)
- Positions: encoder slightly worse than analytical (analytical is exact: P_inv@y = 0 error)
- I/O check: 1-step NRMS < 0.1% (pipeline-compatible)
- Velocity verification: Python central-diff matches MATLAB within 0.77%
- Training: 200 epochs, 115s, converges at ~epoch 60

### Script features (both):
- Target states computed in Python (P_inv@y + central finite-diff), not from MATLAB directly
- CHECK 1: velocity verification (Python vs MATLAB)
- CHECK 2: I/O check (one RK4 step)
- MSD additionally: CHECK 3 augmented state correlation with delta_a
- Plots: loss, comparison (logical + stage), difference (logical + stage), NRMS bar
- All plots show NRMS and RMS (with SI prefix) in legend
- Coordinate names: logical (X, theta, Y, dX, dtheta, dY), stage (x1, x2, Y, dx1, dx2, dY)
- Saves: .pt weights, .json results, .npz trajectories

### Shell script updated
`scripts/gantry/encoder/run_encoder_validation.sh`:
- step0_init → encoder_baseline → encoder_msd → (step1_pipeline | step2_pipeline)

---

## Key files

| File | Role |
|------|------|
| `scripts/gantry/encoder/encoder_baseline_standalone.py` | Encoder standalone, baseline data |
| `scripts/gantry/encoder/encoder_msd_standalone.py` | Encoder standalone, MSD data (NX_ANN=2) |
| `scripts/gantry/encoder/run_encoder_validation.sh` | SLURM runner for all encoder scripts |
| `model_augmentation/utils/torch_nets.py:244-338` | LinearInitEncoderWrapper (phys + ann) |
| `model_augmentation/fit_systems/pre_encoder.py:191-300` | linear_encoder_init (Wb matrices) |
| `scripts/gantry/encoder_initialisation/interconnect_fit.py:318-530` | Jan's reference: how he wires encoder → pipeline |
| `model_augmentation/fit_systems/blocks.py:639-820` | Gantry_State_Block (RK4 + LFR) |
| `model_augmentation/systems/gantry_ss.py` | P matrix, Cd, physical parameters |

### Literature
- `literature/hoekstra2025_lfr-augmentation-ejc.pdf` — Eq. 8 (encoder ResNet), Eq. 16-17 (Wb init from O_n), Eq. 35 (pre-training loss)
- `literature/drenth2025_lpv-lfr-thesis.pdf` — full derivation and MSD examples

---

## Open question for next session

The user wants 6+2 to outperform 6 in standalone. For this to work, the architecture needs to allow gradient flow to augmented states. Options:

1. **Shared-layer architecture**: Replace two separate networks with one MLP that outputs 8 states. Physical state loss trains shared hidden layers → augmented outputs get meaningful hidden representations. But: loses the clean model-based initialization on the linear part.

2. **I/O loss addition**: Add a secondary loss term: feed encoder states through one RK4 step, compare y_hat vs y_measured. This creates a gradient path for augmented states (since the state model could benefit from knowing MSD state). This is essentially a lightweight version of the pipeline.

3. **Accept pipeline-only**: Augmented states only make sense in the full pipeline. Standalone validates the 6 physical states; pipeline validates the 6+2.

The user explicitly does NOT want to supervise on delta_a (unknown in practice). They want to see the encoder discover MSD dynamics from I/O data alone.
