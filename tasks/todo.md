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

*Option A — Open-loop with closed-loop u* ✅ **what was actually implemented**
Load u from MATLAB closed-loop simulation, drive Python model open-loop.
**Result**: y vs q3 = 0.00 µm — proves Python matrices match MATLAB (same as Task 1.2 but via simulation).
**Note**: Option B (full Python closed-loop) was discussed but not implemented — Cfb was never exported
and the meaningful comparison is q3 vs Simscape, not y vs q3.

**Sub-task D — Implement and run `gantry_sim.py`** ✅
- [x] Load `gantry_input.mat` (u from MATLAB closed-loop, deviation coordinates)
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
- ✗ Coriolis/centripetal terms: dropped at linearization (velocity-product terms vanish)
- ✗ Coulomb friction: non-differentiable, excluded from state-space model by construction
  (cc1=16.8 N, cc2=18.35 N, ccy=11.6 N appear in main.m but are marked "not in SS model")
- ✗ Velocity-dependent friction: linearized away
This is a quasi-LPV model. The augmentation must learn the rest from data.

**Why Simscape is the ground truth reference, not the baseline**:
Simscape captures M(Y) + Coriolis + Coulomb. However, it cannot be expressed as differentiable
discrete-time state-space matrices. The augmentation framework requires A(Y)*x + B(Y)*u in
closed form, differentiable through PyTorch for training. Simscape cannot be called from Python
and cannot be backpropagated through. The linearized state-space model is the best physics
expressible in the required form. Simscape is used only as the evaluation ground truth after
training — it is the target to measure against, not the model to train with.

**What each validation step proves**:
- Task 2.4 (matrix comparison) — proves Python A(Y), B(Y) match MATLAB G(Y) exactly.
  Implementation correctness only. Does NOT prove LPV is better than frozen LTI.
- Simulation comparison (Export 2) — layered validation chain, see below.
- The augmentation closes what neither baseline captures: Coriolis + Coulomb.

**Layered comparison chain — what each step isolates**:

Each comparison isolates exactly one effect:

  DT-LPV vs q1 (CT-LPV):
    Both have identical physics (same M(Y), C, K, no Coriolis).
    Residual = ZOH discretization error only.
    Purpose: validates that the ZOH discretization was done correctly.
    Expected: small residual (16 kHz, ΔY ≤ 0.125 mm/sample, 220:1 timescale separation).

  Frozen LTI vs q1 (CT-LPV):
    Residual = ZOH discretization error + frozen M(Y) error.
    When Y varies, this is larger than the DT-LPV residual above.
    Purpose: shows the cost of freezing M(Y) at Y=0.3.

  Gap between the two residuals above:
    = frozen M(Y) error alone (discretization cancels).
    Purpose: quantifies the LPV improvement over frozen LTI.

  DT-LPV vs q (Simscape):
    Residual = Coriolis + Coulomb + ZOH discretization error.
    Purpose: defines the augmentation target — what the network must learn.

  NOTE: Y must vary significantly during the simulation for any difference between
  DT-LPV and frozen LTI to appear. If Y stays near 0.3 m, both use the same matrices
  and produce the same output. The comparison is only meaningful with a trajectory
  where Y sweeps the operational range.

  NOTE: Y=0.3 is the main.m design point — not an arbitrary choice. The frozen LTI
  represents the model you would deploy without any knowledge that Y matters.

Comparison chain steps (supervisor-confirmed):
1. DT-LPV sim vs q1        — ZOH discretization validation (discrete vs continuous, same physics)
2. Frozen LTI vs q1        — shows frozen M(Y) error on top of discretization
3. LPV vs frozen LTI vs q1 — gap = LPV benefit from Y-varying inertia
4. LPV vs q (Simscape)     — augmentation target (Coriolis + Coulomb gap)

### Task 2.1 — Decisions and method ✅
- [x] Tóth (2010) assessed via assess-paper skill
- [x] Method chosen: frozen-at-sampling-instant ZOH for validation; augmented matrix_exp for training
- [x] Drenth (2025) assessed — Architecture 1 confirmed, SSE_Interconnect unchanged
- [x] Augmented matrix exponential formula documented (D-015)
- [x] D-012 updated with quasi-LPV dynamic dependence caveat: Tóth states ZOH is "only reasonable
      for static dependence"; our system has dynamic dependence (Y=x(3)); residual intra-sample
      error accepted as small due to 220:1 timescale separation (ΔY ≤ 0.125 mm/sample).
      Exact Tóth quotes and Assumptions 1 & 2 added. Numerical confirmation deferred to Task 2.5.

### Task 2.2 — MATLAB export scripts
**Files**: `Matlab-scripts/export_lpv_matrices.m`, `Matlab-scripts/export_lpv_sim.m`

Cannot call `main.m` in a loop — it is a script that runs Simulink, figures, setpoint generation.
Instead: duplicate only the physics setup from `main.m` and call `getss.m` directly (immutable
function, safe to call). This is the same computation main.m does at lines 12–88 + 103 + 218.

**Export 1 — LPV matrix sweep** → `Matlab-output/lpv_matrices.mat` ✅
Compares Python A(Y), B(Y) against MATLAB at each operating point (core matrix validation).
**File**: `Matlab-scripts/export_lpv_matrices.m`
- [x] Y sweep: `Y_values = linspace(-0.35, 0.35, 50)` (50 points, within physical range ±400 mm)
- [x] At each Y: build M(Y), call `getss(n,M,C,K)`, apply P transform, `c2d(...,'zoh')`
- [x] Save per Y: `A` (6×6), `B` (6×3), `C` (3×6), `D` (3×3), `Y_values` (50×1)
- [x] Save: `det_M` (50×1) — physics health check, confirms M(Y) positive definite across range

**Export 2 — Varying-Y Simulink simulation** → `Matlab-output/lpv_sim_varying_y.mat`
Provides the reference signals needed for two validations:
  (a) DT-LPV vs q1: proves ZOH discretization was implemented correctly (primary goal)
  (b) Frozen LTI vs q1: shows the cost of freezing M(Y) at Y=0.3 (LPV benefit)
Primary comparison target is q1 (CT quasi-LPV, same physics as our model). q (Simscape) is
the secondary target for the augmentation gap.

**Why Y must vary:** With constant Y both models run the same frozen LTI and ZOH error
is zero by construction -- nothing to measure. ZOH error only appears when M(Y) is changing
between samples. The comparison is only meaningful when Y moves through the operational range.

**Why external scheduling (Y_schedule):** The validation runs DT-LPV with Y_schedule=q1(:,3).
This isolates the ZOH error only. Self-scheduling (x_k[2]) would add a second approximation
on top, confounding the ZOH comparison.

**Trajectory design (choice between two options):**

  Option A -- Reuse existing main.m trajectory (X step + Y sweep):
    r(:,1:2) = 400mm X step; r(:,3) = -400mm Y sweep from 0.3 to -0.1 m.
    Pros: realistic, exercises coupling, no new design work.
    Cons: Y reaches -0.1 m which may be near physical limit; coupled X+Y motion
          makes individual channel residuals harder to interpret.

  Option B -- Dedicated Y ramp, X at rest (RECOMMENDED for ZOH validation):
    r(:,1:2) = 0 (X stays at rest); r(:,3) = smooth Y step from 0.3 to 0.1 m.
    Direction: negative (same as main.m convention: r(:,3) = -pvajs + 0.3).
    Moving positive (toward 0.5 m) risks reaching the physical beam end-stop.
    Pros: isolates Y dynamics cleanly, Y stays within safe range (0.1-0.3 m),
          coupling terms are zero when X motion is absent.
    Cons: less representative of real operation.

  Use Option B for ZOH discretization validation. Option A can be added later
  for the full LPV-vs-frozen-LTI comparison (more realistic conditions).

**MATLAB implementation for export_lpv_sim.m** (new file, does NOT modify kamtin-fp-model):

```matlab
% Matlab-scripts/export_lpv_sim.m
% Exports q1, u, Y_trajectory for LPV ZOH validation.
% Does not modify any file in kamtin-fp-model/.

addpath(genpath('../kamtin-fp-model'))

% --- 1. Physics parameters (identical to main.m lines 12-49) ---
mb = 22.8; mh = 10.1; m1 = 10.2; m2 = 10.7;
Jb = 1.0;  Jh = 0.05;
cg1 = 14.5; cg2 = 20.3; cy = 10;
cb1 = 9;    cb2 = 9;
kb1 = 1987.5; kb2 = 1987.5;
Lb = 0.725; d = 0.1;
Y_op = 0.3;  % main.m design operating point (frozen LTI reference)

M_op = [m1+m2+mb+mh, (m1-m2)*Lb/2-mh*Y_op, 0;
        (m1-m2)*Lb/2-mh*Y_op, Jb+Jh+(m1+m2)*Lb^2/4+mh*d^2+mh*Y_op^2, -mh*d;
        0, -mh*d, mh];
C_mat = [cg1+cg2, (cg1-cg2)*Lb/2, 0;
         (cg1-cg2)*Lb/2, cb1+cb2+(cg1+cg2)*Lb^2/4, 0;
         0, 0, cy];
K_mat = [0,0,0; 0,kb1+kb2,0; 0,0,0];

% --- 2. Build state-space and controller (identical to main.m lines 88-207) ---
n = 3;
sys = getss(n, M_op, C_mat, K_mat);
P = [1, 1, 0; Lb/2, -Lb/2, 0; 0, 0, 1];
StageCoordinatesSystem = P.' * sys * P;
fs = 16e3;  ts = 1/fs;

fbw = 100;
Cfb = tf(num2cell(zeros(3)), num2cell(ones(3)));
for j = 1:3
    Cfb(j,j) = ruleOfThumb(fbw, StageCoordinatesSystem(j,j), ts);
end

% --- 3. Test trajectory: Y step from 0.3 to 0.5 m, X at rest (Option B) ---
% Use thirdOrderSetpointETEL for smooth acceleration-limited Y motion.
% Parameters chosen for Y axis (slower than X: vmax_Y=0.3 m/s, amax_Y=3 m/s^2).
pmax_Y = 0.2;    % [m] Y displacement: 0.3 -> 0.5 m
vmax_Y = 0.3;    % [m/s]
amax_Y = 3.0;    % [m/s^2]
jerkTime_Y = 0.05;  % [s]
jmax_Y = amax_Y / jerkTime_Y;

[pvajs_Y] = thirdOrderSetpointETEL(pmax_Y, vmax_Y, amax_Y, jmax_Y, Inf, ts);
n_move = size(pvajs_Y, 1);

% Add 0.5 s hold at start (system settles at Y=0.3) and 0.5 s at end.
n_hold = round(0.5 / ts);
nt = n_hold + n_move + n_hold;
t = ts * (0:nt-1)';

% Reference: X1=X2 hold at zero, Y ramps from 0.3 to 0.5 m.
r = zeros(nt, 3);
r(:, 3) = 0.3;  % Y reference starts and holds at 0.3 m
r(n_hold + (1:n_move), 3) = 0.3 + pvajs_Y(:, 1);  % Y moves to 0.5 m
r(n_hold + n_move + 1 : end, 3) = 0.5;             % Y holds at 0.5 m

f = zeros(nt, 3);  % no feedforward forces

% --- 4. Run Simulink ---
% r, f, Cfb are set in workspace -- Simulink FromWorkspace blocks read them.
mdl = 'gantry_2025a';
sim(mdl, t(end));
% After sim(): q1, q, q2 are automatically in workspace via ToWorkspace blocks.

% --- 5. Reconstruct u from q1 ---
% u applied to q1 path = Cfb * (r - q1)  (f=0, so no feedforward term).
% Cfb is a discrete diagonal 3x3 TF. lsim handles it channel by channel.
e_q1 = r - q1;          % (N x 3) tracking error
u_q1 = lsim(ss(Cfb), e_q1, t);  % (N x 3) force applied to q1 path

% --- 6. Extract Y trajectory and rename Simscape output ---
Y_trajectory = q1(:, 3);   % (N x 1) absolute Y position [m]
q_simscape = q;            % rename for clarity

% --- 7. Save ---
save('../Matlab-output/lpv_sim_varying_y.mat', ...
     't', 'fs', 'r', 'u_q1', 'q1', 'q_simscape', 'Y_trajectory');
disp('Saved Matlab-output/lpv_sim_varying_y.mat')
```

**Notes on u reconstruction:**
`sim()` does not export u directly (no ToWorkspace block for u in the model). The u applied
to the gantrySystem.m path is `Cfb * (r - q1)` because: (a) the feedback controller reads
the q1 path output, (b) feedforward f=0. Using `lsim(ss(Cfb), r-q1, t)` recovers this
exactly at discrete-time steps. This is the u we pass to the Python DT-LPV simulator.

**Initial conditions:**
Simulink integrators start from zero (default ICs). The initial reference is r(1) = [0, 0, 0.3],
so the feedback will drive X1=X2 to 0 and Y to 0.3 during the first 0.5 s hold period.
Python DT-LPV starts at x0 = [0, 0, 0, 0, 0, 0] matching Simulink ICs. The initial tracking
transient is NOT trimmed -- it is additional valid data (Y is changing during it).

**Key design notes:**
- [ ] NOTE: LPV improvement is expected primarily in X1, X2 channels (M[0,1] and M[1,1] are
      Y-dependent). The Y channel dynamics are largely decoupled (M[2,2]=mh constant). Report
      results per channel; do not expect uniform improvement across all three.
- [ ] NOTE: CT vs DT error floor. q1 is CT; Python LPV is DT at 16 kHz.
      An irreducible ZOH discretization error exists. Task 2.5 quantifies this floor so it
      is not confused with model error.
- [ ] RESOLVE: Controller stability across Y range. Before finalizing 0.3->0.5 m sweep,
      verify in Simulink that the closed-loop (Cfb designed at Y=0.3) remains stable at Y=0.5.
      If controller performance degrades significantly, reduce the Y range.

**Metric:** BFR per channel (primary), RMS in µm (secondary).

**Variables saved:**
  - `t`              (N x 1)   time vector [s]
  - `fs`             (1 x 1)   sample frequency = 16000 Hz
  - `r`              (N x 3)   reference [X1_ref, X2_ref, Y_ref] stage coords [m]
  - `u_q1`           (N x 3)   reconstructed force [F_X1, F_X2, F_Y] [N]
  - `q1`             (N x 3)   CT quasi-LPV output [X1, X2, Y] [m] -- PRIMARY target
  - `q_simscape`     (N x 3)   Simscape nonlinear output [X1, X2, Y] [m] -- secondary
  - `Y_trajectory`   (N x 1)   absolute Y position = q1(:,3) [m]

**Export 2 script written:** `Matlab-scripts/export_lpv_sim.m` ✅
- [x] Run the script in MATLAB: Y_trajectory sweeps 0.3 -> 0.1 m, N=29068 samples, 1.817 s
- [x] q1 and q_simscape populated, u_q1 non-zero (F_Y RMS > 1 N confirmed by verify_exports.m)
- [x] All verify_exports.m checks PASS

**ZOH validation result (gantry_lpv_compare.py):** ✅
- DT-LPV (Python, matrix_exp) vs q1 (CT, ode45): BFR X1=99.99%, X2=99.98%, Y=100.00%
- Residual sub-nanometre across all channels — ZOH discretization confirmed correct
- **What this proves:** matrix_exp ZOH formula and ode45 CT integration agree to numerical precision
- **What this does NOT prove:** model quality vs physical reality (need y_lpv vs q_simscape next)
- Script: `scripts/gantry/gantry_lpv_compare.py`, output: `simulations/lpv_zoh/`

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

## Step 3: CT Model Write-up and RK4 Integration

**Goal**: Write up the CT model equations in full, implement RK4 integration in torch,
and establish the foundation for the LFR-based LPV augmentation.

**Key decision**: D-018 (confirmed 2026-03-20 by supervisor). The model is kept in continuous
time. RK4 with fixed step replaces the pre-discretized ZOH approach in the training loop.
The LFR structure is used for LPV scheduling (D-005, confirmed 2026-03-20).

**Important clarification after derivation review**:
Step 3 currently mixes two different kinds of statements:
- what is already mathematically established
- what is still an implementation decision

What is established:
- the CT quasi-LPV baseline can be written as an explicit LPV-LFR using the
latent-variable construction in `LPV/supporting/derivations/LFR-derivation.tex`
- the resulting algebraic loop is well-posed because it reduces to solving
  `M(Y) v = f_gen` for this specific construction

What is still not fixed:
- whether the implemented baseline state should live in logical or stage coordinates
- whether runtime RK4 should evaluate the explicit LFR loop or the equivalent
  collapsed CT vector field
- whether the chosen LFR repetition count is accepted as "valid and sufficient"
  or needs a stronger minimality argument

These decision checkpoints should be resolved before Task 3.4 code starts.

**Prerequisites**:
- [x] Step 2 complete (LPV baseline validated, physics confirmed)
- [ ] Paper on discretizing LFRs found and reviewed (supervisor action item, 2026-03-20)
- [ ] M matrix rank validated across different trajectories

**Why CT + RK4 instead of pre-discretized ZOH**:
Steps 1 and 2 used ZOH discretization to validate that the Python physics matches MATLAB
exactly. That goal is now achieved. For the augmentation training loop, the supervisor
confirmed: keep the model in CT, apply RK4 with a fixed time step. Pre-discretizing first
gets messy and is not needed when RK4 is available. ZOH holds the input constant within
each interval; RK4 is the integration method used inside that interval.

---

### Task 3.0 — CT model write-up (prerequisite for all of Step 3)
**File**: `docs/ct-model-writeup.md` (new document)

This is a prerequisite task. No implementation begins until this write-up is complete.

- [ ] State the CT ODE in logical coordinates: A_c(Y), B_c in terms of M(Y), C, K, P
- [ ] State the coordinate transform from logical to stage (P matrix, numerical values)
- [ ] List every physical quantity with its symbol, value, dimension, and unit
      (mb, mh, m1, m2, Jb, Jh, cg1, cg2, cy, cb1, cb2, kb1, kb2, Lb, d, fs)
- [ ] Explain what q = [X1, X2, Y, dX1, dX2, dY] means physically (stage positions and velocities)
- [ ] Explain where in the signal chain feedforward (ff) enters (feedforward entry point is
      where excitation/disturbances would be added)
- [ ] Document the RK4 formula applied to the CT ODE:
      k1 = f(x_k, u_k),  k2 = f(x_k + ts/2 * k1, u_k),
      k3 = f(x_k + ts/2 * k2, u_k),  k4 = f(x_k + ts * k3, u_k)
      x_{k+1} = x_k + ts/6 * (k1 + 2*k2 + 2*k3 + k4)
- [ ] Explain why fixed step (RK4) is chosen over variable step (ODE45) and Euler (see D-018)

### Task 3.1 — Validate M matrix invertibility across the operational range
**File**: `scripts/gantry/validate_rank_m.py` (new script)

Supervisor noted (2026-03-20): "computing rank of m matrix for different trajectories.
Can see if there is rank drop/can happen. Will probably not happen."

**Method (two-part, rigorous):**

Part A — Analytical rank/positive-definiteness check (complete, no sweep needed):
M(Y) is a polynomial in Y, so det(M(Y)) is a degree-2 polynomial (parabola) in Y.
This means the check can be done analytically without any sampling:
- [ ] Compute det(M(Y)) symbolically as a function of Y (numpy.poly1d or sympy)
- [ ] Find all real roots of det(M(Y)) = 0
- [ ] Verify no real root falls inside the ETEL operational range [-0.35, 0.35] m
- [ ] Confirm det(M(0)) > 0 (sign check at one point, combined with no roots in range, proves
      det > 0 everywhere in the range)
- [ ] Apply Sylvester's criterion: verify all three leading principal minors are positive
      across the range (M[0,0] is constant; 2x2 minor and 3x3 det are both polynomials in Y)
This approach is complete and rigorous. If no roots exist in the range, M(Y) is positive
definite (and therefore invertible) everywhere the gantry operates.

Part B — Condition number sweep (numerical health check):
The condition number is not a polynomial, so use a dense sweep here:
- [ ] Compute cond(M(Y)) for Y in linspace(-0.35, 0.35, 200)
- [ ] Plot condition number vs Y; report the maximum
- [ ] Pass criterion: condition number stays below a reasonable threshold (e.g. < 1e4)
      A high condition number would not break invertibility but would cause numerical issues
      in computing M(Y)^{-1} and therefore A_c(Y)

**Literature note**: Find a paper or textbook reference that formally states inertia matrices
from the Lagrangian formulation are positive definite by construction (T = 0.5 q_dot^T M q_dot > 0
for any non-zero q_dot implies M positive definite). This provides the theoretical backing
for why rank drop is not expected, and can be cited in the thesis.
- [ ] Find and log the reference (add to `docs/references.md`)

- [ ] Save summary plots and analytical result to `simulations/rank_validation/`

### Task 3.2 — RK4 integrator in torch
**File**: `scripts/gantry/gantry_rk4_torch.py` (new file)

Implements the CT ODE integration using RK4 with fixed step ts = 1/fs. This replaces
the role of `gantry_lpv_torch.py` (which used matrix_exp ZOH) in the training loop.

- [ ] Implement `gantry_ct_ode(x, u, Y=None)` returning dxdt:
      Uses A_c(Y), B_c from physical parameters (all torch tensors)
      Y defaults to x[2] (self-scheduling) when not provided externally
- [ ] Implement `rk4_step(f, x, u, ts)` as a standalone function
      k1 = f(x, u), k2 = f(x + ts/2 * k1, u), k3 = f(x + ts/2 * k2, u),
      k4 = f(x + ts * k3, u); return x + ts/6 * (k1 + 2k2 + 2k3 + k4)
- [ ] Implement `GantryRK4Simulator(nn.Module)`:
      `forward(x0, u)`: loop over time steps, call rk4_step at each step
      Self-scheduling: Y from x[k][2] at each step
      Output: y_k = C * x_k (C constant, same as before)
- [ ] Verify gradient flow (BPTT test, same pattern as Task 2.3)
- [ ] Cross-check against ZOH results from Step 2: BFR should remain above 99.9% (both
      methods integrate the same CT ODE; small numerical difference expected)

### Task 3.3 — Literature: two open blockers for LFR implementation
**Output**: Notes in `docs/decisions.md` and/or `docs/lfr-discretization-notes.md`

Both items below must be resolved before Task 3.4 can begin.

**Blocker A — Paper on discretizing LFRs (supervisor action item, 2026-03-20):**
- [ ] Search literature for a paper on discretizing LFRs
      (supervisor explicitly mentioned this as an action item in the meeting)
- [ ] Understand how the LFR structure is handled when integrating with a CT baseline:
      does the LFR itself need to be discretized separately from the CT ODE,
      or does applying RK4 to the full CT system subsume the LFR?
- [ ] Log design implications in `docs/decisions.md` (new decision D-019 if needed)

**Blocker A interpretation note**:
This blocker is now narrower than it first appeared. Because the project moved
to CT+RK4, the missing literature is no longer needed to justify the existence
of the baseline LFR itself. It is mainly needed to answer a more specific
implementation question: whether there is any reason the explicit LFR loop must
also be treated as a separately discretized object, or whether RK4 on the full
CT realization is sufficient.

**Blocker B — Realizing the LPV model with rational parameter dependence as an LFR:**
The gantry CT model contains M(Y)^{-1}, which makes the entries of A_c(Y) rational
functions of Y (not polynomial). Converting this to a proper LFR form requires expressing
the rational Y-dependence as a linear fractional transformation with a structured Δ(Y).

PARTIALLY RESOLVED (checked 2026-03-22):

What IS resolved (augmentation well-posedness):
- drenth2025_lpv-lfr-rational.pdf covers well-posedness for the LFR structure.
- Well-posedness condition: Definition 1 (IFAC) defines det(I - Dzw * Δ(p(k))) ≠ 0.
  Theorem 2.5 (thesis) / Theorem 6 (IFAC) gives sufficient conditions.
- Direct parameterization: Dzw = exp(-N) with N ≻ 0 guarantees ρ(Dzw) < 1 by
  construction. Well-posedness satisfied automatically during training.

What is NOT resolved (baseline LFR realization):
- Drenth's papers address LPV-LFR **identification** (learning M and Δ from data).
  They do NOT address converting a known physics model (with rational M(Y)^{-1}) into
  LFR form. The baseline is assumed to already be in LFR form (eq. 5.1 in thesis).
- Converting A_c(Y) with rational Y-entries to a proper LFR {M_lfr, Δ(Y)} requires an
  LFT realization procedure. The standard reference is Zhou, Doyle & Glover (1996),
  "Robust and Optimal Control", Chapter 10. Both Drenth and Hoekstra cite this textbook.
- Alternative tools: MATLAB Robust Control Toolbox (`lftdata`), LPVcore, lpvtools.
- [ ] Obtain Zhou, Doyle & Glover (1996) Ch. 10 for LFT realization procedure
- [ ] Alternatively: ask supervisors for the specific conversion method they recommend
- [ ] Read Drenth thesis Chapter 5 (pages 29-34): confirm how the baseline LFR is
      assumed to be structured, and whether any guidance on conversion is given
- [ ] Log any remaining gaps in `docs/decisions.md`

**Blocker B interpretation note**:
The derivation document has now changed the shape of this blocker. It is no
longer accurate to treat the project as if there were no baseline realization
method at all. There is now a direct algebraic realization available.

The remaining question is narrower:
- do we accept the latent-variable realization from `LPV/supporting/derivations/LFR-derivation.tex` as
  the project baseline, or
- do we still require a textbook or tool-based LFT realization for comparison,
  minimality, or supervisor preference?

This should be written explicitly before implementation, otherwise the task can
 drift between "derive any valid realization" and "derive a canonical one".

### Task 3.4 — LFR structure for LPV baseline and augmentation (supervisor confirmed 2026-03-20)
**File**: `scripts/gantry/gantry_lfr_lpv.py` (new file)

Implements the LFR structure for both the baseline and augmentation, following Drenth's
notation (thesis eq. 5.1-5.2, IFAC paper eq. 6-7).

**Notation** (Drenth convention, NOT the generic M11/M12/M21/M22):
- M_lfr = [[A_x, B_w, B_u], [C_z, D_zw, D_zu], [C_y, D_yw, D_yu]]
  (the constant interconnection matrix; called "M_lfr" to avoid collision with M(Y) inertia)
- Δ(p) = diag(p * I_η) where p = Y (scheduling variable), η = repetition count
- The repetition count η is a design parameter: higher η allows richer rational dependence
  on Y but increases the latent dimension. Start with η determined by the rational degree
  of M(Y)^{-1} entries. Document choice as a new decision (D-019 or D-020).

**Two LFR subsystems** (Drenth thesis eq. 5.2):
1. Baseline LFR: captures the known rational Y-dependence from M(Y)^{-1} in A_c(Y).
A valid latent-variable realization now exists in `LPV/supporting/derivations/LFR-derivation.tex`.
   Remaining decision: implement that realization directly, or replace/compare
   it with a textbook/tool-based LFT realization if one is obtained later.
2. Augmentation LFR: learned from data, adds correction on top of baseline.
   Uses Drenth's direct parameterization for well-posedness (D_zw = exp(-N)).

**Decision checkpoint before implementation**:
- [ ] Decide whether the baseline will be implemented in logical coordinates and
      transformed around the data interface, or similarity-transformed fully to
      stage coordinates before coding
- [ ] Decide whether `gantry_lfr_lpv.py` represents a runtime simulation object
      or a representation/proof object that feeds an equivalent RK4 vector field
- [ ] Decide whether the current repetition count is accepted as sufficient, or
      whether a minimality/canonical-form argument is required

- [ ] Determine η for the baseline LFR (from the rational structure of M(Y)^{-1})
- [ ] Implement the baseline LFR realization (blocked on Task 3.3 Blocker B)
- [ ] Implement Δ(Y) block: maps scheduling variable Y to diag(Y * I_η)
- [ ] Implement M_lfr as trainable parameters using Drenth's notation
- [ ] Implement augmentation LFR with direct parameterization (D_zw = exp(-N))
- [ ] Wire both into the SSE_Interconnect alongside the CT+RK4 integration
- [ ] Validate well-posedness: ρ(D_zw) < 1 holds by construction
- [ ] Blocked on: Task 3.3 (both blockers) and Task 3.2 (RK4 baseline)

**Notation collision warning**: Throughout this project, "M" refers to two different things:
- M(Y): the 3x3 inertia matrix from the gantry Lagrangian (physics)
- M_lfr: the constant interconnection matrix in the LFR structure (control theory)
Always use M(Y) or M_lfr to disambiguate. Never use bare "M" without context.

### Task 3.5 — Wire CT + RK4 baseline into augmentation interconnect
**File**: `scripts/gantry/gantry_baseline.py` (updated from original plan)

The CT + RK4 baseline requires a custom block class, not the existing `Linear_State_Block`.

**Important scope note**:
Task 3.5 should not start until Task 3.4 has answered the representation-versus-
runtime question. Otherwise there is a risk of building `CT_RK4_State_Block`
against the wrong abstraction boundary.

- [ ] Implement `CT_RK4_State_Block(Block)`:
      Takes x_k and u_k, performs one RK4 step using `gantry_ct_ode`, returns x_{k+1}
      Computes A_c(Y), B_c at each forward call (Y from x_k[2])
- [ ] Implement `Linear_Output_Block(C, D)` (unchanged, C is constant)
- [ ] Instantiate `Interconnect(nx=6, nu=3, ny=3)`
- [ ] Wire with selection/expansion matrices (FP_state_ix = np.arange(6))
- [ ] Run forward pass: confirm no shape errors and gradient flow works

### Task 3.6 — Normalization
- [ ] Apply normalization once real experimental data is available
- [ ] Compute T_x, T_u, T_y from training data statistics
- [ ] Note: normalization must be compatible with the CT ODE (normalizing the state changes
      the ODE coefficients; document how this is handled)

### Task 3.7 — Add augmentation network and train
- [ ] Add `Static_ANN_Block` or `Dynamic_ANN_Block` as parallel augmentation
- [ ] Step 1: fit to least squares (minimize MSE on output)
- [ ] Step 2: adjust cost function for settling time (see Step 4)
- [ ] Evaluate on held-out data (unseen Y positions, unseen motion profiles)

**KEEP IN MIND — M(Y) invertibility under trainable parameters:**
If any inertia parameters (mb, mh, m1, m2, Jb, Jh, Lb, d) are made trainable during
augmentation, the pre-training invertibility verification (Task 3.1) no longer holds.
The roots of det(M(Y)) shift with every parameter update. Options to handle this:
  1. Keep inertia parameters fixed; only allow damping (cg1, cg2, cy, cb1, cb2) and
     stiffness (kb1, kb2) to be trainable. These do not enter M(Y) so invertibility is
     unaffected. This is the recommended starting point.
  2. If inertia parameters must be trained: add a regularization term or hard constraint
     that keeps the minimum eigenvalue of M(Y) above a safe threshold.
  3. Monitor cond(M(Y)) during training and stop/warn if it degrades.
Decide which parameters are trainable before starting Task 3.7 and document in decisions.md.

---

## Step 4: Research Novelties Development

These are confirmed research contributions (supervisor meeting 2026-03-20).

### Novelty 1: Orthogonal projection regularization
**Status**: Theoretical development ongoing. Theory must be developed before implementation.
- [ ] Formalize the orthogonal projection approach (Gyorok et al. base)
- [ ] The extra states introduced by the augmentation are not fully theorized yet
      (supervisor note: "extra states not really thought about yet, will also need theoretical development")

### Novelty 2: Settling time cost function (supervisor identified, 2026-03-20)
**Status**: Planned, not started.

Supervisor noted: "first step fit model to least square fit. then settling time. change cost
function for that. another novelty: tuning cost function for settling error bound."

- [ ] Design cost function that emphasizes settling time and settling error bound
      (not just squared loss, which treats all time steps equally)
- [ ] Literature review: existing cost functions for settling time in system ID
- [ ] Implement and test on gantry simulation data first

### Novelty 3: Local FRF integration into global fitting
**Status**: Identified as interesting, open question on experiment design.

Supervisor note: "use local measurements, small excitations, local frf. can include in total
fitting of the model. their expectation: if you take local frf response will describe system
really well. in squared loss function hard to emphasise that. combine local data, if not
able to estimate global model -- about experiment design."

- [ ] Investigate how to combine local FRF measurements with global trajectory fitting
- [ ] Design: how to weight local FRF vs global trajectory loss
- [ ] This is also tied to experiment design (what excitations to use)

---

## Step 3b: Baseline Parameter Training on Synthetic MATLAB Data (D-023)

**Goal**: Before adding any augmentation black box, demonstrate that the baseline with free physical parameters can recover the correct parameter values from MATLAB-generated synthetic data. This is Jan's parameter update method applied to the gantry baseline.

**Why first**: Roland specified this phasing (2026-03-31 meeting). Validates the training pipeline in isolation before augmentation complexity is added.

### Task 3b.1 — Generate synthetic training data from MATLAB
- [ ] Generate synthetic data for a range of Y values and parameter volumes (use export scripts or extend them)
- [ ] Data must cover representative operational range of Y

### Task 3b.2 — Implement trainable baseline parameters
- [ ] Make physical parameters (mb, mh, m1, m2, etc.) trainable in the torch model
- [ ] Initialize close to true values
- [ ] Document which parameters are made trainable (decisions.md)

### Task 3b.3 — Train and validate parameter recovery
- [ ] Train on MATLAB synthetic data (no augmentation block)
- [ ] Verify that estimated parameters converge to the MATLAB ground-truth values
- [ ] This is the go/no-go gate before augmentation begins

### Task 3b.4 — Compare LPV-LFR model with Jasper's MATLAB result
- [ ] Compare the Python LPV-LFR simulation output against Jasper's MATLAB LPV-LFR implementation
- [ ] Raised by ASMPT in the 2026-03-31 meeting as a cross-check

---

## April 9 Meeting Preparation

Next meeting: April 9, afternoon (online or on campus), supervisor preference confirmed.

- [ ] Clearly explain the MATLAB model: what each file does, what each q signal means,
      what the physical quantities are (units, dimensions)
- [ ] Prepare slides with a picture of the model structure (block diagram of gantry, signals)
- [ ] Prepare Gantt chart with absolute dates (not relative dates)
- [ ] Find and review paper on discretizing LFRs (see Task 3.3) and summarize findings
- [ ] Write up CT conversion (Task 3.0) so it can be presented
- [ ] Be ready to answer: what is q? What are X, Y positions physically? Where does ff enter?

---

## Deferred

- `torch.compile` on the RK4 state block: one-line optimization once eager-mode
  implementation is validated. Static input shapes and no data-dependent control flow
  make it a good candidate. Must test compatibility with `torch.utils.checkpoint`
  (stable from PyTorch 2.1+). Do not add until correctness is confirmed in eager mode.
- Orthogonal projection regularization (full implementation, blocked on theory in Novelty 1 above)
- F1Tenth application: supervisor noted (2026-03-20) this is simulation-only.
