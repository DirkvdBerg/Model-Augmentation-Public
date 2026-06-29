# Session Handoff: Baseline Encoder Diagnostic

**Last written**: 2026-06-21

---

## Why this matters

This is **Step 1 of baseline verification**: proving the encoder works when the model IS the system (NX_ANN=0, ANN frozen, only the encoder is trainable). If the encoder can't maintain physical state quality in this ideal case (no model mismatch), then model augmentation (Step 2) won't work either. This is a decision gate for the entire project.

---

## The problem: training destroys state quality despite loss converging

The linear encoder initialization (Hoekstra 2026, Eq. 16-17) gives excellent state reconstruction at init. But training with output MSE loss degrades it:

| Configuration | q2 init | q2 after training | Ratio | Notes |
|---|---|---|---|---|
| ANN unfrozen, 100 ep, nf=20, lr=1e-3 | 0.052 | 1301 | 25,000x | Co-adaptation bug (FIXED: freeze ANN) |
| ANN frozen, 100 ep, nf=20, lr=1e-3 | 0.052 | 1.72 | 33x | Loss converged fine |
| ANN frozen, 1 ep, nf=10, lr=1e-3 | 0.052 | 2.64 | 51x | Instant explosion |

Output prediction loss converges in all cases. The model predicts outputs correctly but the internal states become non-physical.

**Root cause**: Theta (q2) contributes ~3.4 microns to X1/X2 outputs vs ~49 mm from X states. That's 0.007%. Output MSE gradients for theta are essentially zero, so the encoder drifts from its initialization. The loss function doesn't "see" theta.

---

## Architecture

```
Encoder (LinearInitEncoderWrapper)
    Input: past I/O windows (u_past, y_past)
    Output: normalized state estimate x_hat (6 states)
    Init: linear map from reconstructability matrix (Hoekstra Eq. 16-17)
    Trainable: yes (the only trainable component in Step 1)

State Block (Gantry_State_Block)
    RK4 integration of LPV nonlinear dynamics
    Y_op=None (self-scheduled), Ts=1/fs_new, up_sample=1
    NOT trainable (physics block)

ANN Block (Static_ANN_Block)
    Frozen (all params requires_grad=False) when NX_ANN=0
    Zero-initialized, outputs zero

Output Block (Linear_Output_Block)
    y = C @ x + D @ u (linear, Cd/Dd from gantry_ss)
```

Framework: deepSI/SUBNET via `SSE_Interconnect`. Loss: nf-step output MSE.

---

## Current state of diagnostic script

File: `scripts/gantry/encoder/diagnostic_nf_lr.py`

### Two stages

**Stage 1: Encoder init quality vs sampling rate**
- Sweeps FS_SWEEP = [20000, 4000, 2000, 1000, 500, 400, 200]
- Compares encoder init NRMS at each rate vs 20 kHz native
- Auto-selects lowest acceptable rate for Stage 2

**Stage 2: nf x lr grid at selected rate**
- NF_VALUES = [20, 40, 80, 160, 200], LR_VALUES = [5e-4, 1e-4, 5e-5]
- N_DIAG_EPOCHS = 10, per-epoch state tracking
- Early stopping if q2 ratio > 10x
- Outputs: per-combination table, heatmaps, loss curves, state evolution, JSON

### CRITICAL BUG in Stage 1: uses apply_experiment instead of direct encoder call

Stage 1 currently calls `evaluate_encoder_states()` which runs `apply_experiment()` (full sequential model simulation, 200k timesteps at 20 kHz). This hangs for minutes.

**The fix**: Call the encoder directly on I/O windows. The encoder init is a linear map, testable with a batched forward pass in milliseconds.

How to evaluate encoder init directly:
1. Downsample validation data: u_val, y_val, x_logical_val at rate fs
2. Normalize: `u_norm = (u - u_mean) / std_u`, `y_norm = (y - y0) / ystd`
3. Construct sliding windows: for each timestep t >= na+na_right:
   - `u_window[t] = u_norm[t-nb-nb_right+1 : t+1]`  shape (nb+nb_right, nu)
   - `y_window[t] = y_norm[t-na-na_right+1 : t+1]`  shape (na+na_right, ny)
4. Stack into batch, call `encoder.forward(u_batch, y_batch)` -> (batch, 6) normalized states
5. Un-normalize: `x_phys = x_enc * std_x + x_mean`
6. Compare to `x_logical_val` at corresponding timesteps, compute per-state NRMS

The encoder class is `LinearInitEncoderWrapper` at `model_augmentation/utils/torch_nets.py:244`. Its `forward(upast, ypast)` handles internal convention conversion (pipeline mean-subtracted to pure-scaled and back).

**Important**: Stage 2 (nf/lr sweep) should KEEP using `evaluate_encoder_states()` with `apply_experiment`, because there we test the full pipeline (encoder + model propagation) after training. Only Stage 1 needs the direct call.

### Other issues

- **ENCODER_INIT_MAX_RATIO = 5.0 is arbitrary**: No justification for this threshold. Print the table, let the user decide based on the data.
- **deepSI epochs are cumulative**: `fit(epochs=N)` means "train to epoch N total." Script handles this (calls `fit(epochs=epoch)` with epoch=1,2,3,...).
- **`(0,)` debug prints**: From deepSI internals. Harmless.
- **Add `flush=True`** to print statements so output appears immediately via conda run.

---

## step1_baseline_equals_system.py current parameters

File: `scripts/gantry/encoder/step1_baseline_equals_system.py`

| Parameter | Current value | Notes |
|---|---|---|
| FS_NEW | 400 | Needs validation from diagnostic Stage 1 |
| up_sample | 2 | Should be 1 (validated by downsampling Test B) |
| nf | 20 | Needs validation from diagnostic Stage 2 |
| lr | 1e-3 | Too aggressive (51x q2 explosion at nf=10). Needs Stage 2 |
| epochs | 100 | |
| batch_size | 256 | |
| NX_ANN | 0 | Baseline = system |
| ANN frozen | yes | Fixed this session |

After the diagnostic produces results, update these parameters and run full training.

---

## Established facts (from prior work)

| Fact | Evidence |
|---|---|
| RK4 model passes at 200 Hz | Downsampling validation: 0.82% NRMS |
| up_sample=1 sufficient at 200+ Hz | Downsampling Test B: NRMS < 5.6e-5 |
| Encoder init fails at 200 Hz | q2 NRMS = 2.078 (unusable) |
| Encoder init works at 400 Hz | q2 NRMS = 0.052 |
| Native 20 kHz encoder init | Unknown (Stage 1 will measure) |
| Jan's hyperparameters | nf=int(0.100/Ts), lr=1e-4 to 5e-4, batch_size=256-3000 |
| Jan's paper on state convergence | "Deferred to future research" (no solution given) |

---

## Execution order for next session

1. **Fix Stage 1** in `diagnostic_nf_lr.py`: Replace `evaluate_encoder_states()` call with direct encoder evaluation (construct I/O windows, batch forward pass, compare to x_logical). Keep Stage 2 using `evaluate_encoder_states()` as-is.

2. **Run diagnostic**: `conda run -n GraduationProject python scripts/gantry/encoder/diagnostic_nf_lr.py`

3. **Read Stage 1 output**: Does 20 kHz encoder init work well for q2? If yes, the sampling rate sweep tells us which rates preserve that quality. If 20 kHz is also bad for q2, the problem is fundamental observability, not sampling rate.

4. **Read Stage 2 output**: Any (nf, lr) with BETTER or STABLE verdict? Look at the per-epoch state curves to see if q2 initially holds and then diverges, or immediately explodes.

5. **Decision gate**:
   - If a good (nf, lr) exists: update step1 parameters, run full training
   - If ALL combinations degrade q2: output MSE cannot constrain theta. Need loss modification (state regularization, auxiliary loss on theta, or accept non-physical states)

6. **Update step1**: Set validated FS_NEW, up_sample=1, nf, lr from diagnostic. Run overnight.

---

## Key files

| File | Role |
|------|------|
| `scripts/gantry/encoder/diagnostic_nf_lr.py` | Diagnostic script (NEEDS Stage 1 fix) |
| `scripts/gantry/encoder/step1_baseline_equals_system.py` | Full training script (needs parameter update) |
| `model_augmentation/utils/torch_nets.py:244` | `LinearInitEncoderWrapper` (encoder to call directly) |
| `model_augmentation/fit_systems/pre_encoder.py` | `linear_encoder_init` (reconstructability-based encoder) |
| `model_augmentation/systems/gantry_linearization.py` | `gantry_linearize_and_discretize(dt)` |
| `model_augmentation/systems/gantry_ss.py` | Cd, Dd, P matrices |
| `data/gantry/matlab/multisine/baseline-v2/` | Training data (T1-T10, V1-V2, E1-E2, 10s each, 20 kHz) |
| `simulations/gantry_subnet/diagnostics/` | Output directory |
| `tasks/lessons.md` | Active rules (read first every session) |
| `.claude/plans/elegant-hugging-fox.md` | Data generation plan (baseline-v2, separate from encoder work) |
