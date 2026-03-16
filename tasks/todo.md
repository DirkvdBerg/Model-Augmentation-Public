# Task Tracking

---

## Step 1: Frozen LTI Baseline — Physics Validation

**Goal**: Confirm the Python translation of the MATLAB FP model is correct.
Y is fixed at 0.3 m, standard ZOH discretization. This is not the final model —
it is a validation tool to rule out physics bugs before moving to LPV.

**Decisions in scope**: D-006 (stage coordinates), D-007 (fixed baseline first)

---

### Task 1.1 — Compute discrete A, B, C, D in stage coordinates ✅
**File**: `scripts/gantry/gantry_ss.py`
- [x] Physical parameters from `main.m`
- [x] M(Y), C, K matrices matching `gantrySystem.m`
- [x] Continuous-time SS in logical coordinates (getss.m)
- [x] Stage coordinate transform via P
- [x] ZOH discretization at fs=16 kHz via `scipy.signal.cont2discrete`

### Task 1.2 — Validate discrete matrices against MATLAB G ✅
**File**: `scripts/gantry/gantry_ss.py` (`__main__` block)
- [x] Dimensions correct (6×6, 6×3, 3×6, 3×3)
- [x] D matrix is zero
- [x] Eigenvalues on/inside unit circle (marginally stable — correct, rigid body modes)
- [x] A, B, C, D match MATLAB G matrices to < 1e-10 (actual error ~1e-19)

### Task 1.3 — Standalone simulation (no augmentation)
**File**: `scripts/gantry/gantry_sim.py`
- [ ] Simulate N steps using plain matrix recursion:
      `x_{k+1} = A·x_k + B·u_k`,  `y_k = C·x_k`
- [ ] Use same synthetic input trajectory as `main.m` (`thirdOrderSetpointETEL`)
      OR a simple step input for quick visual check
- [ ] Export MATLAB `lsim` output (`q3`) from `main.m` to `Matlab-output/`
- [ ] Compare Python simulation vs MATLAB `lsim` — confirm outputs match
- [ ] Plot X1, X2, Y over time — confirm physically reasonable behaviour

**Pass criterion**: Python lsim output matches MATLAB q3 to numerical tolerance.
If mismatch: physics bug. Fix before proceeding.

---

## Step 2: LPV Extension — Tóth Discretization

**Goal**: Replace frozen ZOH with proper discrete-time LPV model where A(Y), B(Y)
vary with the scheduling variable Y. This is the model that goes into augmentation.

**Reference**: `toth2010discretization` (cited in research-methods.md)
**Decisions needed**: D-009 (LPV discretization method choice)

### Task 2.1 — Study Tóth LPV discretization method
- [ ] Read `toth2010discretization` — understand what changes vs standard ZOH
- [ ] Determine: does M(Y) vary linearly or nonlinearly in Y?
      (from main.m: M[0,1] = M[1,0] linear in Y, M[1,1] quadratic in Y)
- [ ] Identify which matrices become scheduling-dependent: A(Y), B(Y) or both

### Task 2.2 — Implement LPV discretization
**File**: `scripts/gantry/gantry_lpv_ss.py`
- [ ] Implement `gantry_lpv_discrete_ss(Y, fs=16e3)` returning A(Y), B(Y), C, D
- [ ] Verify reduces to frozen LTI at Y=0.3 (matches Task 1.2 result)

### Task 2.3 — Validate LPV model across operating points
- [ ] Evaluate A(Y), B(Y) at multiple Y values (e.g. Y = 0.1, 0.2, 0.3, 0.4, 0.5 m)
- [ ] Export MATLAB G at each Y from `main.m` to `Matlab-output/`
- [ ] Compare Python vs MATLAB at each operating point
- [ ] Verify eigenvalues remain stable across full operational range
- [ ] Verify invertibility of M(Y) across full operational range

**Pass criterion**: Python LPV matrices match MATLAB at all tested Y values.

---

## Step 3: Combine with Augmentation Framework

**Goal**: Wire the validated LPV baseline into the augmentation interconnect
and prepare for training on real experimental data.

**Blocked on**: Step 2 complete + real experimental gantry data available

### Task 3.1 — Wire into augmentation interconnect
**File**: `scripts/gantry/gantry_baseline.py`
- [ ] Instantiate `Interconnect(nx=6, nu=3, ny=3)`
- [ ] Instantiate `Linear_State_Block(A, B)` and `Linear_Output_Block(C, D)`
- [ ] Wire with `selection_matrix` / `expansion_matrix` (FP_state_ix = np.arange(6))
- [ ] Run forward pass — confirm no shape errors

### Task 3.2 — Normalization
- [ ] Apply `normalize_linear_ss_matrices()` once real data is available
- [ ] Compute T_x, T_u, T_y from training data statistics

### Task 3.3 — Promote to trainable baseline
- [ ] Replace `Linear_State_Block` with `Parameterized_Linear_State_Block`
- [ ] Replace `Linear_Output_Block` with `Parameterized_Linear_Output_Block`
- [ ] Set RMSE_baseline from Step 2 simulation results

### Task 3.4 — Add augmentation network and train
- [ ] Add `Static_ANN_Block` or `Dynamic_ANN_Block` as parallel augmentation
- [ ] Train on experimental data
- [ ] Evaluate on held-out data (unseen Y positions, unseen motion profiles)

---

## Deferred

- LPV-LFR augmentation adaptation (D-005)
- Orthogonal projection regularization (Aspect 3, research-methods.md)
