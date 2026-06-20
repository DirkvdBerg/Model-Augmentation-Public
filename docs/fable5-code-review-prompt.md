# Fable 5 Code Review Prompt: Gantry Augmentation Implementation

Copy everything below the line into Fable 5.

---

## Context

I am a master's student implementing dynamic model augmentation for a dual-gantry positioning system. The framework was built by my supervisor Jan (his reference examples work correctly on MSD benchmarks). I adapted it to a 3-DOF gantry with a hidden mass-spring-damper (MSD) on the payload.

**The problem**: the default learned encoder cannot recover certain physical states (theta, velocities) from I/O history, even though the output loss (sim-RMS) looks reasonable. I am trying to determine whether this is:
1. An implementation bug (wrong indices, wrong normalization, wrong physics)
2. A fundamental observability limitation of the system/encoder architecture
3. A training configuration issue (nf too short, na_nb too small, wrong learning rate)

**Your task**: Compare my gantry implementation against Jan's reference MSD implementation. Look for implementation bugs that could explain poor state recovery. Be specific: cite file paths, line numbers, and what the correct value should be.

## System description

**Gantry baseline (6-state)**:
- States: x = [q1, q2, q3, dq1, dq2, dq3] in logical coordinates
- Inputs: u = [F_X1, F_X2, F_Y] in stage coordinates (3 forces)
- Outputs: y = [X1, X2, Y] in stage coordinates (3 positions)
- Coordinate transform: P maps stage forces to logical forces, P^T maps logical positions to stage positions
- y = P^T @ q (positions only, no velocities in output)
- M(Y) inertia matrix depends on Y position (LPV)

**Augmented system (8-state, MATLAB ground truth)**:
- Adds hidden MSD: delta_a (displacement), vdelta_a (velocity)
- MSD mass ma = fraction of mh, spring ka, damper ca
- MSD couples to gantry through 4x4 mass matrix
- Output equation unchanged: still y = P^T @ [q1, q2, q3] (MSD states are hidden)

**Key dimensions**:
- FS_ORIG = 20000 Hz, FS_NEW = 4000 Hz (decimation factor D=5)
- NX_PHYS = 6, NX_ANN = 2 (augmented states), nxd = 8
- nf = 1200 samples (300 ms rollout at 4 kHz)
- na_nb = 400 samples (100 ms encoder history)
- RK4 with 10x upsampling inside each dt step

## Files to read (in order of priority)

### 1. PHYSICS COMPARISON (highest priority)

**MATLAB ground truth** (what the data was generated from):
- `Matlab-scripts/Augmentation/gantrySystemExtended.m` — 8-state ODE with MSD
- `Matlab-scripts/Augmentation/data/generate_multisine_data.m` (lines 1-100) — data generation, parameter values, which Simulink model is used
- `Matlab-scripts/Augmentation/additional_state_lagrangian.m` — symbolic Lagrangian derivation

**Python physics block** (what the training uses):
- `model_augmentation/systems/gantry_ss.py` — physical constants, M(Y) decomposition, LFR polynomial constants, P transform, Cd/Dd output matrices
- `model_augmentation/fit_systems/blocks.py` lines 641-820 — `Gantry_State_Block`: RK4 integration, deriv() method, denormalization, LFR signal flow

**What to check**:
- Do the physical parameter values in gantry_ss.py match generate_multisine_data.m exactly?
- Does the Python deriv() produce the same xdot as the MATLAB gantrySystemExtended.m ODE for the baseline (no MSD) case? The Python block only models the 6-state baseline; the MSD is supposed to be learned by the ANN.
- Is the P transform applied correctly? MATLAB uses logical-coordinate forces as input to the ODE. Python receives stage-coordinate forces and transforms via P.
- Is M(Y)^{-1} computed correctly via the LFR polynomial rational form N(Y)/d(Y)?
- The RK4 uses 10x upsampling (line 761). Is the step size dt/10 correct for a 150 Hz mode?

### 2. NORMALIZATION (high priority, subtle bugs hide here)

**Jan's reference** (working):
- `scripts/ecc_2025/msd_ndof_interconnect_dynamic.py` line 67 — uses `normalize_linear_ss_matrices()` with `auto_fit_norm=True`
- `model_augmentation/utils/utils.py` — `normalize_linear_ss_matrices` function

**Gantry** (manual normalization):
- `scripts/gantry/gantry_interconnect_dynamic.py` lines 123-155 — computes x_mean, std_x from finite-difference velocities, normalizes Cd manually
- Same file lines 170-173 — passes std_x, std_u, x_mean, u_mean to Gantry_State_Block
- Same file lines 204-207 — manually sets fit_sys.norm.u0, ustd, y0, ystd
- `blocks.py` lines 769-818 — deriv() denormalizes x and u, computes xdot in physical units, renormalizes

**What to check**:
- Jan uses `auto_fit_norm=True`. The gantry uses `auto_fit_norm=False` with manual normalization. Is the manual normalization equivalent? Specifically:
  - Does the Gantry_State_Block correctly undo normalization before physics and reapply after?
  - Is Cd_norm computed correctly? It should satisfy: y_norm = Cd_norm @ x_norm (where x_norm = (x - x_mean) / std_x and y_norm = (y - y0) / ystd)
  - Is x_mean[2] = mean(Y positions)? This matters because Y ~ 0.3m, so the mean offset is significant.
  - Are the u_mean and std_u computed from the correct input signal? (u_total for multisine, u for trajectories)
- The state block receives normalized x, denormalizes to physical, computes xdot_phys, then renormalizes via xdot_norm = xdot_phys / std_x. Is this correct? Note: d(x_norm)/dt = d((x_phys - x_mean)/std_x)/dt = dxdot_phys / std_x (since x_mean is constant). This looks correct but verify.

### 3. INTERCONNECT WIRING (medium priority)

**Jan's reference**:
- `scripts/ecc_2025/msd_ndof_interconnect_dynamic.py` lines 75-98

**Gantry**:
- `scripts/gantry/gantry_interconnect_dynamic.py` lines 161-227 (build_model function)

**What to check**:
- The selection_matrix and expansion_matrix use PHY_IX = [0,1,2,3,4,5] to route 6 physical states within the 8-dimensional (nxd=8) total state. Is this correct?
- The ANN block has nz=nxd+nu=11 inputs and nw=nxd=8 outputs. It receives all 8 states + 3 inputs, and outputs corrections to all 8 state derivatives. The physical block only touches indices [0:6]. The ANN corrections at indices [6:8] are the only thing driving the augmented states. Is this the intended behavior?
- The output block uses selection_matrix(PHY_IX, nxd) to pick the 6 physical states, then applies Cd_norm (3x6). The augmented states [6:8] do not appear in the output. Is this correct for a hidden MSD?

### 4. ENCODER (medium priority)

**Hybrid encoder**:
- `model_augmentation/utils/torch_nets.py` lines 174-232

**What to check**:
- Positions: `pos = y_denorm @ P_inv` where P_inv = inv(P). Is this the right direction? We need logical positions from stage outputs: q_logical = inv(P^T) @ y_stage. Since y = P^T @ q, we need q = inv(P^T) @ y = (P^{-T})^T @ y... verify the transpose chain is correct.
- Velocities: backward finite difference using last two y samples, multiplied by fs. Is fs correct (4000 Hz after decimation, not 20000 Hz)?
- The physical states are detached (`.detach()`) so no gradient flows through them. Only the ANN branch for augmented states has gradients. Is this the right design choice?
- When NX_ANN=0: the encoder returns only x_phys_norm (6 states). When NX_ANN=2: it appends 2 learned states. The zero-init ANN starts at zero, so initial augmented states are zero. Correct?

### 5. DATA LOADING (lower priority)

- `scripts/gantry/gantry_interconnect_dynamic.py` lines 100-117
- Does the decimation `[::D]` correctly downsample from 20 kHz to 4 kHz?
- Is the input signal `u_total` (trajectory + multisine) or `u` (trajectory only)?
- The MATLAB data has x_logical (6 states) and delta_a (MSD displacement). These are NOT loaded during training. They are only used for normalization statistics. Correct?

## Specific questions to answer

1. **Is there a sign error or transpose error in the P transform chain?** Follow the forces from stage input u through P to logical forces, through the ODE, back through P^T to stage outputs. Any error here would make the physics block compute wrong dynamics.

2. **Does the denormalization in deriv() correctly recover physical values?** x_phys = x * std_x + x_mean. If x_mean[2] is the mean Y position (~0.3m), then the denormalized Y position should be correct. But if x_mean is computed from training data that has Y varying between 0 and 0.4m, x_mean[2] ~ 0.2m, which is NOT Y_op=0.3m. Does this cause a mismatch when Y_op is used for frozen-Y mode?

3. **Is the Cd normalization consistent with the state normalization?** Cd_norm = Cd * std_x / ystd. The output block computes y_norm = Cd_norm @ x_norm. Expanding: y_norm = Cd * std_x / ystd @ (x_phys - x_mean) / std_x = Cd @ (x_phys - x_mean) / ystd = (Cd @ x_phys - Cd @ x_mean) / ystd. For this to equal (y - y0) / ystd, we need y0 = Cd @ x_mean. Check: is y0 computed as Cd @ x_mean in the training script?

4. **In LPV mode (Y_op=None), is the Y value correctly extracted from the normalized state?** The deriv() denormalizes x before extracting Y = x_phys[2]. But x_phys = x * std_x + x_mean. If x_mean and std_x are correct, this should give the true Y position. Verify this chain.

5. **Are there any dimension mismatches when NX_ANN=0?** The ANN block still exists with nw=nxd=6 when NX_ANN=0 and nxd=NX_PHYS=6. Does Static_ANN_Block work correctly when its output dimension equals the physical state dimension?

## Output format

For each of the 5 sections above, report:
- **OK**: if no issues found, with a one-line summary of what you verified
- **ISSUE**: description, file:line, what it is vs what it should be
- **UNCLEAR**: if you cannot determine correctness without running code, state what test would resolve it

End with a prioritized list of the top 3 most likely bugs (if any found) that could explain poor encoder state recovery.
