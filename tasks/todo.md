# Task Tracking

---

## Step 1: Fixed FP Baseline in Augmentation Interconnect

**Goal**: Implement the gantry FP model as `Linear_State_Block` + `Linear_Output_Block` wired into the augmentation interconnect. Validate end-to-end before adding trainability.

**Decisions in scope**: D-006 (stage coordinates), D-007 (fixed baseline first)
**Reference script**: `scripts/bouc_wen/bouc_wen_pre_encoder.py`

---

### Task 1.1 — Compute discrete A, B, C, D in stage coordinates
**File**: `scripts/gantry/gantry_ss.py`

- [ ] Encode physical parameters from `main.m` (masses, dimensions, stiffness, damping)
- [ ] Implement `M(Y)`, `C`, `K` matrices as Python functions, matching `gantrySystem.m` exactly
- [ ] Build continuous-time state-space in logical coordinates:
  - `A_c = [0, I; -M(Y)\K, -M(Y)\C]`
  - `B_c = [0; M(Y)⁻¹]`
  - `C_c = [I₃, 0₃]`, `D_c = 0`
- [ ] Apply stage coordinate transform: `A_stage = P' A_c P`, `B_stage = P' B_c`, `C_stage = C_c P`, `D_stage = D_c`
- [ ] Discretize with ZOH at `ts = 1/16000` using `scipy.signal.cont2discrete`
- [ ] Return `A, B, C, D` as numpy arrays

**Fixed operating point**: `Y = 0.3 m` (matches `main.m`)
**Dimensions**: nx=6, nu=3, ny=3

---

### Task 1.2 — Validate discrete matrices against MATLAB
**File**: `scripts/gantry/gantry_ss.py` (add a `__main__` block)

- [ ] Print A, B, C, D and verify dimensions (6×6, 6×3, 3×6, 3×3)
- [ ] Check all eigenvalues of A are inside the unit circle (stable system)
- [ ] Verify C = [I₃ | 0₃] structure is preserved after discretization
- [ ] Verify D is zero (no direct feedthrough in this model)
- [ ] Cross-check A, B values against MATLAB `G.A`, `G.B` — manually compare at least the (1,4), (2,5), (3,6) entries (near-identity block from ZOH of double integrator)

**Pass criterion**: eigenvalues inside unit circle, dimensions correct, A/B entries match MATLAB to < 1e-6

---

### Task 1.3 — Wire into augmentation interconnect (baseline only, no ANN)
**File**: `scripts/gantry/gantry_baseline.py`

- [ ] Instantiate `Interconnect(nx=6, nu=3, ny=3)`
- [ ] Instantiate `Linear_State_Block(A, B)` and `Linear_Output_Block(C, D)`
- [ ] Add blocks and wire with `selection_matrix` / `expansion_matrix`
  - FP_state_ix = np.arange(6) (all states are baseline states)
- [ ] Normalization: skip for now — note as placeholder, will require real data statistics
- [ ] Run one forward pass with a zero input tensor to confirm shapes are correct
- [ ] Simulate N=1000 steps with a unit step input on F_X1, plot X1, X2, Y

**Pass criterion**: forward pass succeeds with no shape errors; positions rise and stabilize

---

### Normalization note (blocking Step 2)
`normalize_linear_ss_matrices()` requires training data statistics (std of states, inputs, outputs). This cannot be applied until real experimental gantry data is available. Step 2 (trainable baseline) is blocked on data acquisition.

---

## Step 2: Trainable Baseline (deferred — blocked on data)

Promote to `Parameterized_Linear_State_Block` / `Parameterized_Linear_Output_Block` once:
- Step 1 validation passes
- Real experimental data is available for normalization

---

## Step 3: LPV Extension (deferred — see D-005)
