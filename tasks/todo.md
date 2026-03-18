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

### Task 1.3 — Standalone simulation and comparison against Simscape
**File**: `scripts/gantry/gantry_sim.py`

**Sub-task A — Export MATLAB data** ✅
- [x] `Matlab-output/gantry_input.mat`  — `u` (force input), `r` (reference), `t`
- [x] `Matlab-output/gantry_output.mat` — `q` (Simscape nonlinear ground truth), `t`

**Sub-task B — Choose simulation approach**

Two options were considered:

*Option A — Open-loop with closed-loop u*
Feed the closed-loop controller output `u` directly into the Python model open-loop.
**Problem**: without feedback the linearisation error accumulates freely — the Python
output will drift from `q` in a way that reflects the missing feedback, not just
linearisation error. Not a fair comparison.

*Option B — Closed-loop Python simulation* ✅ **chosen**
Replicate the full closed-loop in Python: implement `Cfb`, feed reference `r`,
simulate the feedback loop. Compare Python closed-loop output vs Simscape `q`.
**Why**: both systems see the same reference under the same controller — any
residual is purely linearisation error, which is exactly what we want to quantify.

**Sub-task C — Export Cfb from MATLAB**
- [ ] Save `Cfb` matrices from MATLAB:
      ```matlab
      [Cfb_num, Cfb_den] = tfdata(Cfb);
      save('Matlab-output/gantry_controller.mat', 'Cfb_num', 'Cfb_den', 'ts')
      ```

**Sub-task D — Implement and run `gantry_sim.py`** ✅
- [x] Load `gantry_input.mat` (u already in deviation coordinates from lsim)
- [x] Run open-loop simulation in deviation coordinates (x_0=0), add Y_op back
- [x] Save y (N,3) and t to `simulations/frozen_lti/y.npz`

**Sub-task E — Comparison script** ✅
- [x] `scripts/gantry/gantry_compare.py` created
- [x] Loads `simulations/frozen_lti/y.npz`, `gantry_q3_lsim.mat`, `gantry_q_simscape.mat`
- [x] Plots y, q3, q on same axes per channel (3 panels)
- [x] Prints RMS and max absolute residual (in µm) for: y vs q3, q3 vs q, y vs q
- [x] **Results**: y vs q3 = 0.00 µm (PASS), q3 vs q: X1 4.86 µm / X2 3.19 µm / Y 0.10 µm

**Step 1 PASS** — Python model matches MATLAB lsim to numerical zero. Linearisation gap is 4–16 µm, bounded and small. Dominant dynamics captured.

**Why not compare against lsim (q3)?**
A, B, C, D already match MATLAB G to 1e-19 (Task 1.2) — lsim comparison
is redundant. Simscape is the nonlinear ground truth and shows the physically
meaningful linearisation gap the augmentation must learn to correct.

**Pass criterion**: Simscape comparison shows a visible but bounded residual —
confirms the linearised model captures dominant dynamics.

---

## Step 2: LPV Extension — Frozen-at-sampling-instant ZOH

**Goal**: Implement and validate the discrete-time LPV model where A(Y), B(Y) vary with
scheduling variable Y.

**Method**: Frozen-at-sampling-instant ZOH (Tóth Section III-B) — call standard ZOH at each
Y value. Zero local truncation error within the ZOH assumption (justified at 16 kHz, ΔY small).

**Key decisions**: D-012 (discretization method), D-014 (numpy vs torch files), D-015 (augmented
matrix exponential for B_d), D-016 (matrix comparison validation strategy)

**What the LPV model captures and what it does not**:
- ✓ Y-dependent inertia M(Y): M[0,1] linear in Y, M[1,1] quadratic in Y — this is the LPV part
- ✗ Coriolis/centripetal terms: dropped in linearization (velocity-dependent, not in G)
- ✗ Velocity-dependent friction: linearized away
This is a quasi-LPV model. The augmentation must learn the rest from data.

**What each validation step proves**:
- Task 2.4 (matrix comparison) — proves Python correctly implements the same physics as
  MATLAB G(Y). Implementation correctness only.
- Export 2 + simulation comparison — proves the LPV simulation is a *better baseline* than
  the frozen LTI when Y varies. Comparison target is q1 (not q directly — see below).
- The augmentation closes what neither frozen LTI nor LPV captures: Coriolis, friction, etc.

**Correct comparison target for LPV superiority: q1, not q**
q1 (from gantrySystem.m in Simulink) is a continuous-time quasi-LPV simulation — M(Y) is
re-evaluated at each integration step as Y evolves. It uses identical physics to our LPV
model (same M(Y), C, K, no Coriolis). This makes it the cleanest reference:

  Frozen LTI ──┐
               ├── vs q1  →  proves LPV beats frozen LTI (Y-varying inertia effect)
  Python LPV ──┘

  q1  ── vs q (Simscape)  →  defines augmentation target (Coriolis + Coulomb gap)

Comparison chain (supervisor-confirmed):
1. q3 (frozen LTI lsim) vs q1 — intermediate step, shows frozen LTI error
2. Python LPV sim vs q1       — validates LPV implementation (discrete vs continuous)
3. LPV vs LTI vs q1           — proves LPV benefit when Y varies
4. q1 vs q (Simscape)         — quantifies what augmentation must learn

### Task 2.1 — Decisions and method 🔄 (Tóth justification in progress)
- [x] Tóth (2010) assessed via assess-paper skill
- [x] Method chosen: frozen-at-sampling-instant ZOH for validation; augmented matrix_exp for training
- [x] Drenth (2025) assessed — Architecture 1 confirmed, SSE_Interconnect unchanged
- [x] Augmented matrix exponential formula documented (D-015)
- [x] D-012 updated with quasi-LPV dynamic dependence caveat: Tóth states ZOH is "only reasonable
      for static dependence"; our system has dynamic dependence (Y=x(3)); residual intra-sample
      error accepted as small due to 220:1 timescale separation (ΔY ≤ 0.125 mm/sample).
      Exact Tóth quotes and Assumptions 1 & 2 added. Numerical confirmation deferred to Task 2.5.

### Task 2.2 — MATLAB export script
**File**: `Matlab-scripts/export_lpv_matrices.m` (new script — does not modify immutable files)

Cannot call `main.m` in a loop — it is a script that runs Simulink, figures, setpoint generation.
Instead: duplicate only the physics setup from `main.m` and call `getss.m` directly (immutable
function, safe to call). This is the same computation main.m does at lines 12–88 + 103 + 218.

**Export 1 — LPV matrix sweep** → `Matlab-output/lpv_matrices.mat` ✅
Compares Python A(Y), B(Y) against MATLAB at each operating point (core matrix validation).
**File**: `Matlab-scripts/export_lpv_matrices.m`
- [x] Y sweep: `Y_values = linspace(0.05, 0.75, 50)` (50 points, ~14 mm spacing, D-016)
- [x] At each Y: build M(Y), call `getss(n,M,C,K)`, apply P transform, `c2d(...,'zoh')`
- [x] Save per Y: `A` (6×6), `B` (6×3), `C` (3×6), `D` (3×3), `Y_values` (50×1)
- [x] Save: `det_M` (50×1) — physics health check, confirms M(Y) positive definite across range

**Export 2 — Varying-Y Simulink simulation** → `Matlab-output/lpv_sim_varying_y.mat`
Provides the reference signals needed to prove LPV is a better baseline than frozen LTI.
Primary comparison target is q1 (continuous-time quasi-LPV, same physics as our model).
q (Simscape) is the secondary target — quantifies the augmentation gap.
- [ ] NOTE: first check Step 1 Simscape data — if Y varies significantly there, reuse it
      instead of running a new simulation (load q_simscape.mat, plot Y channel vs time)
- [ ] If new simulation needed: design reference where Y axis sweeps 0.1 → 0.7 m slowly
      while X1/X2 do normal motion — requires a new reference signal r_lpv
- [ ] Run Simulink model (gantry_2025a) with this reference — saves q, q1, q2 automatically
- [ ] Save the following variables:
      - `t`              (N×1)   — time vector [s]
      - `r`              (N×3)   — reference signals [X1_ref, X2_ref, Y_ref] in stage coords [m]
      - `u`              (N×3)   — controller force inputs [F_X1, F_X2, F_Y] [N]
                                   (must be from linear feedback — NOT Simscape nonlinear
                                    friction forces; see Step 1 u/q3 fix for why)
      - `q1`             (N×3)   — gantrySystem.m output [X1, X2, Y] [m]
                                   PRIMARY comparison target: continuous-time quasi-LPV,
                                   same physics as our LPV model (M(Y) updated each step,
                                   no Coriolis, no Coulomb). Proves LPV beats frozen LTI.
      - `q_simscape`     (N×3)   — Simscape nonlinear output [X1, X2, Y] [m]
                                   SECONDARY target: quantifies augmentation gap (q1 vs q
                                   = Coriolis + Coulomb contribution)
      - `Y_trajectory`   (N×1)   — absolute Y position over time [m], extracted from q1
                                   (used by Python LPV sim to set scheduling variable)
      - `fs`             (1×1)   — sample frequency [Hz] = 16000

### Task 2.3 — Torch reimplementation ✅
**Files**: `scripts/gantry/gantry_lpv_torch.py`, `scripts/gantry/gantry_lpv_sim_torch.py`

This is a **full torch reimplementation** of `gantry_discrete_ss` — NOT a wrapper around it.
Every value (physical parameters, M(Y), A_c, B_c, P transform) is defined as a torch tensor
from the start so that gradients flow through the entire computation. The only structural
difference from `gantry_ss.py` is the numerical backend: `cont2discrete` is replaced by
`torch.linalg.matrix_exp` on the 9×9 augmented matrix (required for differentiability and
to handle singular A_c — see D-015).

Defining everything in torch from the start (not converting from numpy) also means physical
parameters (mb, mh, m1, m2, ...) can optionally be made trainable later with no refactoring.

- [x] Implement `gantry_lpv_matrices_torch(Y: torch.Tensor, fs: torch.Tensor = _DEFAULT_FS)`
      - All physical parameters defined as `torch.tensor` scalars (float64)
      - M(Y), C_mat, K built as torch tensors using tensor arithmetic
      - A_c(Y), B_c(Y) assembled as torch tensors
      - Stage coordinate transform P as torch tensor; B_c_stage = B_c @ P, C_c_stage = P.T @ C_c
      - 9x9 augmented matrix: `M_aug[:6,:6] = A_c; M_aug[:6,6:] = B_c_stage`
      - `EM = torch.linalg.matrix_exp(M_aug * ts)`
      - Returns A_d=EM[:6,:6], B_d=EM[:6,6:], C_c_stage (constant), D=zeros — all torch tensors
      - fs changed from float to torch.Tensor for consistency; module-level `_DEFAULT_FS` buffer
- [x] Add `__main__` block in `gantry_lpv_torch.py`: torch vs scipy < 1e-10 at Y=0.3 (actual 5.5e-15)
- [x] Implement `GantryLPVSimulator(nn.Module)` in `gantry_lpv_sim_torch.py`
      - `forward(x0, u)`: self-scheduling loop p[k]=x[k][2], list+torch.stack, out-of-place updates
      - `simulate(x0, u)`: torch.no_grad wrapper around forward()
      - C_d computed once in __init__ (constant, no Y-dependence), registered as buffer
      - `__main__` block: Test 1 (frozen Y free-response vs scipy, error 5.5e-15 PASS),
        Test 2 (BPTT gradient test, grad norm 2.24e2 PASS)

### Task 2.4 — Validation: Python vs MATLAB matrix comparison
**File**: `scripts/gantry/gantry_lpv_validate.py`

- [ ] Load `Matlab-output/lpv_matrices.mat`
- [ ] At each Y in the sweep:
      - Call `gantry_discrete_ss(Y)` → A_py, B_py, C_py, D_py
      - Compare to MATLAB A, B, C, D: `max_err = np.max(np.abs(A_py - A_mat))`
      - Check max_err < 1e-10 for all matrices
- [ ] Plot max error vs Y for each matrix (A, B, C) — should be flat near 1e-19
- [ ] Verify M(Y) positive definite at all Y (det > 0, or all eigenvalues > 0)
- [ ] Verify all eigenvalues of A(Y) inside unit circle across full sweep
- [ ] Save summary plot to `simulations/lpv_validation/matrix_errors.png`

### Task 2.5 — Quantify rectangular approximation error (Option D vs Option E)
**File**: `scripts/gantry/gantry_lpv_validate.py` (add section)

- [ ] At each Y: compare `A_d_rect = I + ts·A_c(Y)` vs `A_d_exact = expm(A_c(Y)·ts)`
- [ ] Plot relative error vs Y — establishes the O(ts) bound numerically
- [ ] Confirms Option E (matrix_exp) is preferred over Option D (rectangular)

**Pass criterion**: Python LPV matrices match MATLAB to < 1e-10 at all Y values in the sweep.
Eigenvalues stable and M(Y) positive definite across full operational range.

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
- `torch.compile` on `LPV_Linear_State_Block.forward()`: one-line optimisation once eager-mode
  implementation is validated. Static matrix shapes (6×6, 6×3) and no data-dependent control
  flow make it a good candidate. Must test compatibility with `torch.utils.checkpoint`
  (stable from PyTorch 2.1+). Do not add until correctness is confirmed in eager mode.
- Orthogonal projection regularization (Aspect 3, research-methods.md)
