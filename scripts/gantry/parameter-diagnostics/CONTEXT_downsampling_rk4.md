# Context: Downsampling & RK4 Step-Size Validation Script

## Goal

Write a Python script that determines the minimum viable sampling rate (FS_NEW)
for the gantry model, and validates that the RK4 integration is accurate at that
rate. Output should go to `simulations/gantry_subnet/diagnostics/`.

The script should answer two questions:
1. At what FS_NEW does the discretized model start to deviate from the reference?
2. How many RK4 sub-steps (up_sample) are needed at each FS_NEW?

## Why this matters

The gantry system is sampled at FS_ORIG = 20 kHz. We currently downsample to
FS_NEW = 4 kHz (decimation factor D=5). Training uses RK4 integration at
dt = 1/FS_NEW with `up_sample` sub-steps per sample.

Lower FS_NEW means:
- Longer effective rollout windows for the same nf (e.g., nf=400 at 1 kHz = 0.4s
  vs 0.1s at 4 kHz). This matters for augmentation training where nf should
  cover the system's settling time (~1.3s for theta).
- Less memory per trajectory (fewer samples).
- But risks aliasing fast dynamics and RK4 integration error.

The supervisor's guidance: "check number of RK4 steps. One RK4 step should be
fine. Can check this with FP model with RK4, can compare the two step sizes."

## System dynamics (from baseline_dynamics_analysis.m)

The baseline gantry at Y_op=0 has these modes:

| Mode | Frequency | Damping | Settling time |
|------|-----------|---------|---------------|
| Theta oscillation | 5.2 Hz | zeta=0.092 | ~1.34 s |
| X velocity decay | - | overdamped | tau ~ 1.01 s |
| Y velocity decay | - | overdamped | tau ~ 1.55 s |
| X position | integrator | - | - |
| Y position | integrator | - | - |

With MSD augmentation (not yet in baseline, future target):
- MSD resonance at ~150 Hz. This sets the upper frequency bound.
- Nyquist for 150 Hz = 300 Hz, practically need >= 1.5 kHz.

For the baseline model alone (no MSD), the fastest dynamics are theta at 5.2 Hz,
so sampling rates as low as ~100 Hz would suffice from a Nyquist perspective.
But RK4 accuracy and transient behavior set a tighter lower bound.

## What the script should do

### Test A: Downsampling sweep (integration accuracy)

1. Generate a reference trajectory at FS_ORIG = 20 kHz:
   - Load a training .mat file (e.g., T3_X_sym_Y000.mat)
   - Extract u (stage forces) and x_logical (true states) at 20 kHz
   - This is the ground truth

2. For each candidate FS_NEW in [10000, 4000, 2000, 1000, 500, 250, 100]:
   a. Downsample the input u by factor D = FS_ORIG / FS_NEW
   b. Discretize the baseline model: Ad, Bd = gantry_linearize_and_discretize(dt=1/FS_NEW)
   c. Forward-simulate: x[k+1] = Ad @ x[k] + Bd @ u[k], x[0] = 0
   d. Also simulate using Gantry_State_Block (nonlinear LPV, RK4) with up_sample=1
   e. Compare both simulations against x_logical downsampled to FS_NEW
   f. Compute per-channel NRMS

3. Plot: NRMS vs FS_NEW (one line per state channel). Mark where NRMS exceeds
   a threshold (e.g., 1% = 0.01).

### Test B: RK4 sub-step sweep (at each FS_NEW)

For a few key sampling rates (e.g., 4000, 1000, 500 Hz):

1. Simulate using Gantry_State_Block with up_sample = [1, 2, 5, 10, 20]
2. Use up_sample=20 as the reference
3. Compute NRMS of each up_sample against the reference
4. Plot: NRMS vs up_sample for each FS_NEW

This validates the supervisor's claim that "one RK4 step should be fine."

### Test C: Impulse/step response settling time

1. At FS_NEW = 4 kHz, apply a unit step to each input channel (F1, F2, FY)
2. Simulate the baseline model forward for 2 seconds (8000 samples)
3. Determine when each output channel has settled (e.g., within 2% of final value)
4. Report settling times. These determine the minimum nf for augmentation training.

## Key files

| File | Role |
|------|------|
| `model_augmentation/systems/gantry_linearization.py` | `gantry_linearize_and_discretize(dt)` returns (Ad, Bd, Cd, Dd) for LTI baseline at Y_op=0 |
| `model_augmentation/fit_systems/blocks.py` lines 639+ | `Gantry_State_Block`: nonlinear LPV CT-ODE integrated with RK4. Constructor takes `Ts`, `up_sample`, `Y_op`, normalization params. `nonlinear_function(z)` does one sample step with `up_sample` RK4 sub-steps. |
| `model_augmentation/systems/gantry_ss.py` | Physical parameters (m1, m2, mb, mh, etc.), matrices K, C, P, Cd |
| `data/gantry/matlab/multisine/baseline/` | Training .mat files with `u_total` (stage forces), `y` (stage positions), `x_logical` (logical states), all at FS_ORIG=20 kHz |
| `scripts/gantry/encoder/baseline_dynamics_analysis.m` | MATLAB script with eigenvalue/settling analysis (already run, results above) |

## Coordinate conventions

- **Stage coordinates**: u = [F1, F2, FY] (forces), y = [x1, x2, Y] (positions)
- **Logical coordinates**: x = [X, theta, Y, dX, dtheta, dY]
- **Transform**: y_stage = P^T @ q_logical, u_logical = P @ u_stage
- P is defined in `gantry_ss.py`

## How to use Gantry_State_Block for simulation

```python
from model_augmentation.fit_systems.blocks import Gantry_State_Block
import torch, numpy as np

# Normalization: compute from training data (see encoder_io_validation.py)
state_block = Gantry_State_Block(
    Y_op=None,         # LPV mode (Y read from state)
    std_x=std_x,       # (6,1) state std
    std_u=std_u,       # (3,1) input std
    x_mean=x_mean,     # (6,1) state mean
    u_mean=u_mean,     # (3,1) input mean
    Ts=1/FS_NEW,
    up_sample=1,       # number of RK4 sub-steps per sample
)

# Forward pass: z = [x_norm; u_norm] concatenated, shape (batch, 9, 1)
# x_norm = (x - x_mean) / std_x
# u_norm = (u - u_mean) / std_u
x_next_norm = state_block.nonlinear_function(z)  # (batch, 6, 1)
```

## How to use the linearized model for simulation

```python
from model_augmentation.systems.gantry_linearization import gantry_linearize_and_discretize
import numpy as np

Ad, Bd, Cd, Dd = gantry_linearize_and_discretize(dt=1/FS_NEW)
# Ad: (6,6), Bd: (6,3) maps stage forces to logical states
# Simulate: x[k+1] = Ad @ x[k] + Bd @ u_stage[k]
```

## Output

Save to `simulations/gantry_subnet/diagnostics/`:
- `downsampling_sweep.json`: NRMS per channel per FS_NEW
- `rk4_substep_sweep.json`: NRMS per channel per up_sample per FS_NEW
- `step_response_settling.json`: settling times per I/O channel
- `downsampling_nrms_vs_fs.png`: main result plot
- `rk4_substep_nrms.png`: sub-step validation plot
- `step_response.png`: step response with settling time markers

## Python environment

```bash
conda run -n GraduationProject python scripts/gantry/parameter-diagnostics/downsampling_rk4_validation.py
```

## Important notes

- The linearized model is only valid at Y_op=0. For trajectories with large Y
  excursions, use Gantry_State_Block (nonlinear) as the reference instead.
- The .mat files contain data at 20 kHz. Downsample with simple decimation
  (take every D-th sample). The MATLAB data generation already uses ZOH inputs,
  so decimation of u is correct.
- For Gantry_State_Block, all inputs and states must be normalized. See
  `encoder_io_validation.py:compute_normalization()` for how to compute the
  normalization constants from training data.
- The script should NOT modify any existing files. It is a standalone diagnostic.
