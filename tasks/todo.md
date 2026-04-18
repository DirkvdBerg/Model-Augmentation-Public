# Task Tracking

_Step 1 (Frozen LTI Baseline) completed and archived to `archive/sessions/2026-04-03-todo-step1.md`._

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

## Step 3b: Baseline Parameter Training on Synthetic MATLAB Data (D-023, D-030–D-033)

**Goal**: Demonstrate that the baseline with free physical parameters can recover the correct parameter values from MATLAB-generated data. No ANN augmentation in this step — parameter recovery only. This is the go/no-go gate before augmentation complexity is added.

**Why first**: Roland specified this phasing (2026-03-31 meeting). Validates the training pipeline in isolation.

**Key decisions**: D-030 (parameter set + identifiability), D-031 (separate file), D-032 (SSE_Interconnect subclass), D-033 (data strategy).

**Future extension (not in this step)**:
- Option B data: Python `simulate()` with multisine input, controlled SNR, explicit train/val/test splits (mirrors Jan's MSD setup exactly)
- ANN augmentation on z_lfr slot (parallel, additive to xp) after parameter recovery is proven

---

### Task 3b.1 — Decisions logged ✅
- [x] Parameter set chosen and identifiability justified (D-030)
- [x] File structure decided (D-031)
- [x] SSE_Interconnect integration strategy decided (D-032)
- [x] Data strategy decided: Option A MATLAB, Option B future (D-033)

---

### Task 3b.2 — Implement `lpv_lfr_baseline/lfr_param_block.py`

**Trainable scalars** (nn.Parameter, 10 total):
`kb_sum`, `cg1`, `cg2`, `cy`, `cb_sum`, `mh`, `m1`, `m2`, `mb`, `J_sum`

**Fixed buffers**: `Lb`, `d` (see D-030 for rationale)

**Detuned initial values**:
| Scalar | True value | Detuned init | Δ |
|--------|-----------|--------------|---|
| kb_sum | 3975.0    | 3776.25      | −5% |
| cg1    | 14.5      | 13.05        | −10% |
| cg2    | 20.3      | 18.27        | −10% |
| cy     | 10.0      | 9.00         | −10% |
| cb_sum | 18.0      | 16.20        | −10% |
| mh     | 10.1      | 9.595        | −5% |
| m1     | 10.2      | 9.690        | −5% |
| m2     | 10.7      | 10.165       | −5% |
| mb     | 22.8      | 22.344       | −2% |
| J_sum  | 1.05      | 0.9975       | −5% |

**Implementation checklist**:
- [ ] `class ParameterizedLFRBlock(_BASE)` with `nz=9, nw=18`
- [ ] `self.params` as `nn.Parameter` (10 scalars, detuned init)
- [ ] `self.params_init` as frozen buffer (same detuned values — regularization anchor)
- [ ] `self.Lambda` as buffer — weighted per-parameter (tight for `mb`, standard for rest)
- [ ] `_build_matrices()` — differentiable reconstruction of M0, M1, M2, K, C from `self.params` + fixed `Lb`, `d`
- [ ] `forward()` — same structure as `LFRBaselineBlock.forward()`, calls `_build_matrices()` then `rk4_step()`
- [ ] `param_loss()` — Lambda-weighted L2 regularization toward `params_init`
- [ ] Verification checks (shape, autograd through `_build_matrices`, `param_loss` non-negative, gradient flows to `self.params`)

**RMSE_baseline**: Compute from one no-gradient forward pass of detuned baseline on MATLAB data before training (D-034). Pass result to `ParameterizedLFRBlock.__init__()`.

**Positivity constraint**: Log/exp reparameterization (D-035). Store `self.log_params = nn.Parameter(torch.log(params_init))`. Recover `params = torch.exp(self.log_params).clamp(min=1e-6)` in forward and param_loss.

---

### Task 3b.3 — Subclass SSE_Interconnect

**File**: `lpv_lfr_baseline/lfr_fit_system.py` (new)

- [ ] `class LFRFitSystem(SSE_Interconnect)` that overrides `loss()`
- [ ] Generic sweep: `for m in self.hfn.connected_blocks: if hasattr(m, 'param_loss'): loss_theta += m.param_loss()`
- [ ] All other loss logic (simulation MSE, encoder) inherited unchanged from `SSE_Interconnect`
- [ ] Smoke test: instantiate with a `ParameterizedLFRBlock`, check `loss()` calls `param_loss()`

---

### Task 3b.4 — Training script

**File**: `lpv_lfr_baseline/train_param_recovery.py` (new)

- [ ] Load `Matlab-output/lpv_sim_varying_y.mat` → convert `u_q1`, `q1` to deepSI `System_data`
- [ ] Build Interconnect with `ParameterizedLFRBlock` (same wiring as `build_baseline_interconnect()` in `test_jan_compat.py`)
- [ ] Instantiate `LFRFitSystem`, call `init_model()` and `fit()`
- [ ] Set RMSE_baseline (from Task 3b.2 open item)

---

### Task 3b.5 — Proof / evaluation

- [ ] After training: print `params` (learned) vs `params_init` (detuned) vs true values from `physics.py`
- [ ] Compute RMS prediction error: detuned baseline (no training) vs trained `ParameterizedLFRBlock`
- [ ] Show parameters moved toward true values — this is the go/no-go criterion
- [ ] If parameters do NOT recover: diagnose (data richness, Lambda tuning, RMSE_baseline)

---

### Task 3b.6 — Compare LPV-LFR model with Jasper's MATLAB result
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

- **Measurement noise in `multisine_muli_traject.m`**: Add realistic position measurement noise
  to `q1` outputs before claiming results reflect real-hardware performance. Current simulation
  is noise-free → overly optimistic parameter recovery. Suggested: encoder noise ~1–10 nm RMS
  at 20 kHz, additive Gaussian on `q1` after simulation. Reference: `tasks/handoff.md`
  section "No measurement noise yet". Do NOT add until parameter recovery pipeline is validated
  on clean data first.

- `torch.compile` on the RK4 state block: one-line optimization once eager-mode
  implementation is validated. Static input shapes and no data-dependent control flow
  make it a good candidate. Must test compatibility with `torch.utils.checkpoint`
  (stable from PyTorch 2.1+). Do not add until correctness is confirmed in eager mode.
- Orthogonal projection regularization (full implementation, blocked on theory in Novelty 1 above)
- F1Tenth application: supervisor noted (2026-03-20) this is simulation-only.

---

## Step 3c: Training Speed Optimization — GMatrix Refactor + torch.compile

**Goal**: Reduce per-epoch wall-clock time of `train_param_recovery.py` without
changing the LPV-LFR structure, the polynomial loop solve, or the BPTT gradient flow.

**Decided approach**: Two-phase. Phase 1 is a structural prerequisite; Phase 2 is the
compilation. CUDA graphs are explicitly NOT used (they bypass autograd, incompatible
with gradient-based parameter recovery). bptt_mode='full' stays.

**Why two phases and not just adding @torch.compile now:**
Phase 1 is motivated by arithmetic intensity, not by compilation correctness.
TorchDynamo natively flattens Python dataclasses (including `GMatrix`) into PyTrees
when they are passed as function arguments, referencing the underlying tensor memory
pointers directly. Attribute access on a dataclass is therefore free in a compiled
graph, and value changes after an optimizer step do NOT trigger recompilation.

The reason to replace `GMatrix` with a single `(15, 15)` tensor is GPU arithmetic
efficiency: seven separate submatrix tensors require seven independent global memory
loads, seven micro-matmuls, and seven intermediate writes back to HBM. A single
contiguous tensor lets TorchInductor load the entire G block into SM shared memory
once and process all four matrix products in a single fused Triton kernel -- higher
arithmetic intensity, lower memory bandwidth pressure, better SM utilization.

Phase 1 also simplifies the codebase (one buffer instead of seven, one argument
instead of a structured object) which makes the code easier to read and extend.

**What is NOT changed (hard constraints):**
- LFR signal flow: Steps 1-6 in `lfr_forward.py` remain exactly as specified in
  `literature/lpv-lfr/Additional notes/LPV-LFR-Implementation-Spec.md`
- Polynomial loop solve: N(Y)/d(Y) Horner form stays -- no torch.linalg.solve
- BPTT mode: bptt_mode='full' stays -- full gradient tape, no truncation
- All physical parameters remain trainable via `log_params` (nn.Parameter)
- All verification checks in every `__main__` block must still pass after each task

---

### Phase 1: GMatrix Refactor (prerequisite for Phase 2)

#### Task P1.1 — Replace GMatrix dataclass with (15,15) tensor in lfr_matrices.py

**File**: `lpv_lfr_baseline/core/lfr_matrices.py`

**What to remove:**
- The entire `@dataclass class GMatrix` definition (current lines 63-86), including
  all 7 field annotations (Ax, Bw, Bu, Cz, Dzw, Dzu, Cy)
- The `from dataclasses import dataclass` import (line 52)
- The module-level singleton `G = build_G_matrix(...)` (current lines 190) -- this
  singleton is only used in the `__main__` block and is dead code in the pipeline

**What to change in `build_G_matrix()`:**
- Return type changes from `GMatrix` to `torch.Tensor` with shape `(15, 15)`
- Assemble Ax, Bw, Bu, Cz, Dzw, Dzu, Cy exactly as now (same formulas)
- Then pack into one tensor using `torch.zeros(15, 15, dtype=dtype, device=device)`:

```python
G_mat = torch.zeros(15, 15, dtype=dtype, device=device)
G_mat[:6,   :6]   = Ax   # (6,6)  state rows,  state cols
G_mat[:6,  6:12]  = Bw   # (6,6)  state rows,  w cols
G_mat[:6, 12:15]  = Bu   # (6,3)  state rows,  u cols
G_mat[6:12,  :6]  = Cz   # (6,6)  z rows,      state cols
G_mat[6:12, 6:12] = Dzw  # (6,6)  z rows,      w cols
G_mat[6:12,12:15] = Dzu  # (6,3)  z rows,      u cols
G_mat[12:15, :6]  = Cy   # (3,6)  y rows,      state cols
# G_mat[12:15, 6:15] = 0 by default (Dyw=0, Dyu=0)
return G_mat
```

**Block layout (rows: xdot/z/y; cols: x/w/u):**
```
G_mat (15×15):
         x(0:6)   w(6:12)  u(12:15)
xdot(0:6)  [ Ax   |  Bw   |  Bu  ]
z(6:12)    [ Cz   |  Dzw  |  Dzu ]
y(12:15)   [ Cy   |   0   |   0  ]
```

**What to update in `__main__` block:**
- Replace all `G.Ax`, `G.Bw`, etc. with `G_mat[:6,:6]`, `G_mat[:6,6:12]`, etc.
- The `_alpha_s, _beta_s, ...` module-level precomputation stays (needed for the
  `__main__` block to construct `G_mat` for verification)
- All `check(...)` calls verify submatrix slices against the expected values -- same
  checks, just accessed via slicing instead of attribute

**Verification run after this task:**
```
conda run -n GraduationProject python -m lpv_lfr_baseline.core.lfr_matrices
```
All shape checks and algebraic assembly checks must pass.

---

#### Task P1.2 — Update lfr_forward.py

**File**: `lpv_lfr_baseline/core/lfr_forward.py`

**What to remove:**
- `from lpv_lfr_baseline.core.lfr_matrices import GMatrix` import (line 30)

**What to change in the `lfr_forward()` signature:**
- `G: GMatrix` parameter becomes `G: torch.Tensor  # (15, 15) LFR interconnection`

**What to add at the top of the function body (before Step 1):**
```python
# Extract submatrices from the packed (15,15) G tensor.
# These are strided views -- no data copy, same storage.
# torch.compile traces these as part of the computation graph.
Ax  = G[:6,   :6]    # (6,6)
Bw  = G[:6,  6:12]   # (6,6)
Bu  = G[:6, 12:15]   # (6,3)
Cy  = G[12:15, :6]   # (3,6)
```

**What stays unchanged:**
- All 6 LFR signal-flow steps (the computation is identical; only variable names change
  from attribute access to local tensor variables)
- `xdot = (x @ Ax.T) + (w @ Bw.T) + (u @ Bu.T)` -- same formula
- `y = x @ Cy.T` -- same formula
- The docstring (update the G parameter description only)

**What to update in `__main__` block:**
- `G_true` is now a `(15, 15)` tensor returned by `build_G_matrix()`
- Replace `G_true.Ax`, `G_true.Bw` etc. with slice notation throughout
- All 6 verification checks (loop residual, xdot vs collapsed, w=Y*z, structural audit,
  autograd through Y, collapsed pattern absent) must still pass

**Verification run after this task:**
```
conda run -n GraduationProject python -m lpv_lfr_baseline.core.lfr_forward
```
All 6 checks must pass (PASS status for each).

---

#### Task P1.3 — Update lfr_simulate.py

**File**: `lpv_lfr_baseline/core/lfr_simulate.py`

**What to remove:**
- Any `from lpv_lfr_baseline.core.lfr_matrices import GMatrix` import if present

**What to change in `rk4_step()` signature (line 55):**
- `G: GMatrix` → `G: torch.Tensor  # (15, 15)`
- All other parameters and the function body are unchanged
- G is only passed through to `_lfr_forward()` -- no attribute access on G in rk4_step

**What to change in `simulate()` signature (line 98):**
- `G: GMatrix` → `G: torch.Tensor  # (15, 15)`
- Function body unchanged (G is passed to `rk4_step` only)

**What to change in `SimResult` / type annotations:**
- Any GMatrix type hint in comments or annotations → `torch.Tensor (15, 15)`

No logic changes. This is a signature-only update.

---

#### Task P1.4 — Update LFRBaselineBlock in lfr_block.py

**File**: `lpv_lfr_baseline/blocks/lfr_block.py`

**What to remove from `__init__`:**
- All 7 individual buffer registrations: `_G_Ax`, `_G_Bw`, `_G_Bu`, `_G_Cz`,
  `_G_Dzw`, `_G_Dzu`, `_G_Cy` (7 separate `self.register_buffer(...)` calls)

**What to add in `__init__`:**
```python
# Single (15,15) G matrix -- packed LFR interconnection, stored as one buffer.
# Submatrices are extracted by lfr_forward() via slicing at call time.
self.register_buffer('_G', G_true)   # G_true is now the (15,15) tensor
```

**What to update in `forward()`:**
- Replace any unpacking of `self._G_Ax`, `self._G_Bw` etc. with `self._G`
- The call to `rk4_step(...)` passes `self._G` directly as the G argument
- No reconstruction needed (G never changes for the baseline block -- it is constant)

**Dead code to remove:**
- Any helper that unpacks the old buffer set into a GMatrix instance (if any)

**Verification:** run the block's `__main__` or existing tests.

---

#### Task P1.5 — Update ParameterizedLFRBlock in lfr_param_block.py

**File**: `lpv_lfr_baseline/blocks/lfr_param_block.py`

**What to update:**
- `build_G_matrix()` now returns a `(15, 15)` tensor -- any code that previously
  accessed `G.Ax`, `G.Bw` etc. on its return value must use slices instead
- In `forward()` (if it calls `build_G_matrix()` directly): the returned tensor is
  passed directly to `rk4_step()` -- no unpacking needed
- Any type annotation `G: GMatrix` becomes `G: torch.Tensor`

**What does NOT change:**
- `_recover_params()`, `param_loss()`, `param_table()` -- completely unaffected
- `log_params` (nn.Parameter) -- unchanged
- All buffers (`_Lb`, `_d`, `_P`, `_ts`) -- unchanged
- The call pattern: G is rebuilt inside `_SimWrapper.forward()` (not inside
  `ParameterizedLFRBlock.forward()`) -- this separation stays

---

#### Task P1.6 — Update _SimWrapper in train_param_recovery.py

**File**: `lpv_lfr_baseline/scripts/train_param_recovery.py`

**Location**: `class _SimWrapper` (line 599), `forward()` method (line 606)

**What to update:**
- `build_G_matrix()` call at line 615 now returns a `(15, 15)` tensor
- `simulate(...)` call at line 616 receives the tensor directly -- no change to
  the call signature (the G argument is just passed through)
- Remove any GMatrix import at the top of the file

**What does NOT change:**
- The structure of `_SimWrapper.forward()`: still recovers params once, builds G
  once, calls `simulate()` once per forward -- this is correct and stays
- `bptt_mode='full'` at line 631 stays

---

#### Task P1.7 — Remove GMatrix import from all files and verify

**Files to scan**: grep for `GMatrix` across the entire `lpv_lfr_baseline/` folder:
```
conda run -n GraduationProject grep -r "GMatrix" lpv_lfr_baseline/
```
Every occurrence should be gone. Any remaining reference is dead code to remove.

**Full verification suite -- run all __main__ blocks in order:**
```
conda run -n GraduationProject python -m lpv_lfr_baseline.core.lfr_matrices
conda run -n GraduationProject python -m lpv_lfr_baseline.core.lfr_forward
conda run -n GraduationProject python -m lpv_lfr_baseline.core.lfr_simulate   (if exists)
conda run -n GraduationProject python -m lpv_lfr_baseline.blocks.lfr_block    (if exists)
```

**Pass criterion for Phase 1**: all checks pass, no GMatrix references remain,
`build_G_matrix()` returns a `(15, 15)` tensor, `lfr_forward` accepts it and produces
numerically identical output to before the refactor (compare xdot, z, w, y values
against a reference run before changes).

---

#### Task P1.8 — Update test_lfr_structural.py

**File**: `lpv_lfr_baseline/tests/test_lfr_structural.py`

**Why this task exists:**
After the GMatrix refactor, `build_G_matrix()` returns a `(15, 15)` tensor instead of
a GMatrix dataclass. The test file has 7 uses of GMatrix (import, constructor, attribute
access). None of the mathematical properties tested change -- only the syntax for
accessing submatrices changes. This task updates syntax without changing any assertion,
tolerance threshold, or test logic.

**Critical rule**: Do NOT write bad code to match the test. Update the test to match the
clean (15,15) tensor API. Every change is a direct mechanical substitution.

---

**Change 1 — Import (line 55): remove `GMatrix` from the import**

Old:
```python
from lpv_lfr_baseline.core.lfr_matrices import build_G_matrix, GMatrix
```
New:
```python
from lpv_lfr_baseline.core.lfr_matrices import build_G_matrix
```
Rationale: `GMatrix` class no longer exists after Task P1.1. The `build_G_matrix` function
import stays; only the class name is removed.

---

**Change 2 — Module-level setup (after line 68): add 7 named slice views**

After the line:
```python
_G  = build_G_matrix(_N0, _d0, M1, M2, K, C)
```
Insert the following 7 lines immediately below it:
```python
_G_Ax  = _G[:6,    :6]    # (6,6)  state map
_G_Bw  = _G[:6,   6:12]   # (6,6)  w input
_G_Bu  = _G[:6,  12:15]   # (6,3)  u input
_G_Cz  = _G[6:12,   :6]   # (6,6)  z output, state columns
_G_Dzw = _G[6:12,  6:12]  # (6,6)  z output, w columns
_G_Dzu = _G[6:12, 12:15]  # (6,3)  z output, u columns
_G_Cy  = _G[12:15,  :6]   # (3,6)  y output
```
These are strided views into the (15,15) contiguous tensor -- no data copy, zero overhead.
They replace every attribute access `_G.Ax` → `_G_Ax`, etc. throughout the file.

---

**Change 3 — `_xdot_from_w` helper (line 82): replace 3 attribute accesses**

Old:
```python
    return (_x @ _G.Ax.T) + (w_in @ _G.Bw.T) + (_u @ _G.Bu.T)
```
New:
```python
    return (_x @ _G_Ax.T) + (w_in @ _G_Bw.T) + (_u @ _G_Bu.T)
```

---

**Change 4 — Test 1, line 109: replace `_G.Bw.T`**

Old:
```python
    expected_delta = (w_fake - w_nom) @ _G.Bw.T    # (1, 6)
```
New:
```python
    expected_delta = (w_fake - w_nom) @ _G_Bw.T    # (1, 6)
```

---

**Change 5 — Test 3, lines 191-204: replace GMatrix constructor + attribute access**

Old block (lines 191-204):
```python
    dBw = torch.randn_like(_G.Bw) * 0.1     # (6, 6)  random perturbation

    # Build perturbed G (replace only Bw)
    G_pert = GMatrix(
        Ax=_G.Ax,
        Bw=_G.Bw + dBw,
        Bu=_G.Bu,
        Cz=_G.Cz,
        Dzw=_G.Dzw,
        Dzu=_G.Dzu,
        Cy=_G.Cy,
    )

    xdot_pert = (_x @ G_pert.Ax.T) + (w_nom @ G_pert.Bw.T) + (_u @ G_pert.Bu.T)
```
New block:
```python
    dBw    = torch.randn_like(_G_Bw) * 0.1   # (6, 6)  random perturbation

    # Build perturbed G (clone the (15,15) tensor, modify only the Bw slice)
    G_pert = _G.clone()
    G_pert[:6, 6:12] = G_pert[:6, 6:12] + dBw

    xdot_pert = (_x @ G_pert[:6, :6].T) + (w_nom @ G_pert[:6, 6:12].T) + (_u @ G_pert[:6, 12:15].T)
```
Mathematical meaning is identical: the perturbed xdot uses the full G with only the Bw
block modified. The slice `G_pert[:6, 6:12]` is exactly `Bw`. The `.clone()` ensures the
original `_G` is not mutated.

---

**Change 6 — Test 4, `f_w` inner function (line 240): replace 3 attribute accesses**

Old:
```python
        return (_x @ _G.Ax.T) + (w_in @ _G.Bw.T) + (_u @ _G.Bu.T)
```
New:
```python
        return (_x @ _G_Ax.T) + (w_in @ _G_Bw.T) + (_u @ _G_Bu.T)
```

---

**Change 7 — Test 4, lines 246 and 253: replace `_G.Bw`**

Line 246 old:
```python
    max_err = (J_mat - _G.Bw).abs().max().item()
```
Line 246 new:
```python
    max_err = (J_mat - _G_Bw).abs().max().item()
```

Line 253 old:
```python
    n_wrong = (J_mat - _G.Bw).abs().gt(1e-12).sum().item()
```
Line 253 new:
```python
    n_wrong = (J_mat - _G_Bw).abs().gt(1e-12).sum().item()
```

---

**Change 8 — Test 5, line 289: replace `_G.Bw.T`**

Old:
```python
    expected_delta = ts * (w_fake - w_nom) @ _G.Bw.T    # (1, 6)
```
New:
```python
    expected_delta = ts * (w_fake - w_nom) @ _G_Bw.T    # (1, 6)
```

---

**Change 9 — Test 6, line 347: replace `_G.Dzw`**

Old:
```python
        L_Y = eye6 - y_val * _G.Dzw                # (6, 6)
```
New:
```python
        L_Y = eye6 - y_val * _G_Dzw                # (6, 6)
```
Note: the same pattern appears again at line 417 (second loop in 6c cross-check). Both
occurrences must be updated.

---

**Change 10 — Test 6, lines 364-365: replace `_G.Cz.T` and `_G.Dzu.T`**

Old:
```python
        rhs   = (_x @ _G.Cz.T) + (_u @ _G.Dzu.T)  # (1, 6)
```
New:
```python
        rhs   = (_x @ _G_Cz.T) + (_u @ _G_Dzu.T)  # (1, 6)
```

---

**Change 11 — Test 6, line 417 (cross-check loop): replace `_G.Dzw`**

Old:
```python
        L_Y  = eye6 - y_val * _G.Dzw
```
New:
```python
        L_Y  = eye6 - y_val * _G_Dzw
```

---

**Change 12 — Test 7c (lines 501-505): simplify device check**

Old block:
```python
    G_cpu  = build_G_matrix(N0_cpu, d0_cpu, M1.cpu(), M2.cpu(), K.cpu(), C.cpu())
    ok_7c  = all(
        getattr(G_cpu, f).device.type == 'cpu'
        for f in ['Ax', 'Bw', 'Bu', 'Cz', 'Dzw', 'Dzu', 'Cy']
    )
    print(f"    All G entries on CPU : {ok_7c}   {'PASS' if ok_7c else 'FAIL'}")
```
New block:
```python
    G_cpu  = build_G_matrix(N0_cpu, d0_cpu, M1.cpu(), M2.cpu(), K.cpu(), C.cpu())
    ok_7c  = G_cpu.device.type == 'cpu'
    print(f"    G on CPU : {ok_7c}   {'PASS' if ok_7c else 'FAIL'}")
```
Rationale: `G_cpu` is now a `(15, 15)` tensor. One `.device.type` check on the tensor is
equivalent to checking all 7 submatrices -- they all live in the same storage. The
multi-attribute loop was only needed because GMatrix was a struct of separate tensors.

**What this test still proves after the change:**
The regression being tested is: "no hardcoded `torch.eye(3)` on CPU inside `build_G_matrix`
that would put part of G on CPU when N0 is on CUDA". That bug manifests as the returned
tensor being on CPU when inputs are on CUDA. Checking `G_cpu.device.type == 'cpu'` when all
inputs are on CPU confirms the device-following contract holds. The test description in the
comment above (`7c`) remains accurate.

---

**Summary of all lines changed in test_lfr_structural.py:**

| Location | Old text (key fragment) | New text (key fragment) |
|----------|------------------------|------------------------|
| Line 55 | `import build_G_matrix, GMatrix` | `import build_G_matrix` |
| After line 68 | (nothing) | Add 7 `_G_Xx = _G[rows, cols]` views |
| Line 82 | `_G.Ax.T ... _G.Bw.T ... _G.Bu.T` | `_G_Ax.T ... _G_Bw.T ... _G_Bu.T` |
| Line 109 | `_G.Bw.T` | `_G_Bw.T` |
| Line 191 | `torch.randn_like(_G.Bw)` | `torch.randn_like(_G_Bw)` |
| Lines 193-203 | `G_pert = GMatrix(Ax=..., Bw=_G.Bw+dBw, ...)` | `G_pert = _G.clone(); G_pert[:6,6:12] += dBw` |
| Line 204 | `G_pert.Ax.T ... G_pert.Bw.T ... G_pert.Bu.T` | `G_pert[:6,:6].T ... G_pert[:6,6:12].T ...` |
| Line 240 | `_G.Ax.T ... _G.Bw.T ... _G.Bu.T` | `_G_Ax.T ... _G_Bw.T ... _G_Bu.T` |
| Line 246 | `_G.Bw` | `_G_Bw` |
| Line 253 | `_G.Bw` | `_G_Bw` |
| Line 289 | `_G.Bw.T` | `_G_Bw.T` |
| Line 347 | `_G.Dzw` | `_G_Dzw` |
| Lines 364-365 | `_G.Cz.T ... _G.Dzu.T` | `_G_Cz.T ... _G_Dzu.T` |
| Line 417 | `_G.Dzw` | `_G_Dzw` |
| Lines 501-505 | `all(getattr(G_cpu, f).device.type ... for f in [...])` | `G_cpu.device.type == 'cpu'` |

**No changes to:**
- Any test assertion (tolerances, expected values, pass/fail logic)
- Any call to `lfr_forward` (signature unchanged)
- Any physics constant (`_alpha`, `_beta`, `_gamma`, `_N0`, `_N1`, `_N2`, `_d0`)
- Test 7a (uses `_N0`, `_d0` directly -- no GMatrix)
- Test 7b (builds G_p locally via `build_G_matrix`, passes to `lfr_forward` -- G_p is now a tensor, lfr_forward now takes a tensor, so this is automatically correct)
- Module-level `_x`, `_u`, `_u_s`, `_Y` definitions

**Verification run after this task:**
```
conda run -n GraduationProject python -m lpv_lfr_baseline.tests.test_lfr_structural
```
Expected output: all 7 tests PASS, `Overall: ALL PASS`.

**Why Test 7b is automatically correct without changes:**
Test 7b does:
```python
G_p  = build_G_matrix(N0_p, d0_p, M1, M2, K, C)
xdot_p, _, _, _ = lfr_forward(_x, _u, _Y, G_p, ...)
```
After Phase 1: `build_G_matrix` returns a `(15,15)` tensor; `lfr_forward` accepts a
`(15,15)` tensor. The call pattern is unchanged. Autograd flows through the tensor
slice operations inside `lfr_forward`. No syntax change needed here.

---

### Phase 2: torch.compile (requires Phase 1 complete)

#### Task P2.1 — Benchmark baseline (before compile)

**File**: `lpv_lfr_baseline/scripts/train_param_recovery.py`

Before adding any compilation, measure the current per-epoch wall-clock time:
- Run 20 epochs with current code (post Phase 1, pre compile)
- Record: `t_fwd` (forward pass time), `t_bwd` (backward pass time) from the existing
  `_sync_time()` instrumentation already in the training loop
- Record: total epoch time
- Save this as the baseline for comparison

This measurement establishes the speedup denominator. Without it we cannot claim
any particular improvement.

---

#### Task P2.2 — Add @torch.compile to rk4_step

**File**: `lpv_lfr_baseline/core/lfr_simulate.py`

**What to add:**
```python
import torch

@torch.compile
def rk4_step(
    x:       torch.Tensor,
    ...
```

Place `@torch.compile` directly above the `def rk4_step(...)` line.

**Why this is the right granularity:**
`rk4_step` contains all four `_lfr_forward` calls (k1, k2, k3, k4) in a single
Python function. The compiler traces through all four in one shot and produces a
single fused computation graph. Inter-stage tensor operations (e.g. `x + ts/2 * k1`)
are also inside this graph. The result: 4 Python function dispatches become 1-2
compiled kernel launches per time step.

**Why not @torch.compile on lfr_forward:**
If compile is applied to `lfr_forward` alone, the compiler sees each call in
isolation -- it cannot fuse across the k1→k2→k3→k4 chain. `rk4_step` is the right
boundary because it holds the full RK4 accumulation in scope.

**Why not @torch.compile on simulate:**
`simulate` contains the Python `for k in range(N)` loop. When TorchDynamo compiles
a function containing a Python loop it attempts to **fully unroll** the loop into a
flat FX graph. For N=4000-8000 steps this generates millions of graph nodes, can
exhaust system RAM during tracing, and compilation can take hours. Do NOT apply
`torch.compile` directly to `simulate`. The correct solution for the loop is
`torch.func.scan` (see Task P2.4).

**After Phase 1, all arguments to rk4_step are tensors:**
- G is a `(15, 15)` tensor argument
- K, C, mh, alpha, beta, gamma, N0, N1, N2, ts are all tensors
- x, u_logical are tensors
- `_lfr_forward` contains no Python-level data-dependent branching on Y

This means `fullgraph=True` is achievable. Use these compile flags:
```python
@torch.compile(fullgraph=True, dynamic=False)
def rk4_step(...):
```
- `fullgraph=True`: raises an error on any graph break rather than silently degrading
- `dynamic=False`: prevents generation of generic shape-polymorphic kernels; since
  state dimension (6), batch size, and number of RK4 stages are all fixed, the
  compiler generates shape-specialised Triton kernels which are significantly faster

**Compatibility with BPTT:**
`torch.compile` preserves autograd. The compiled graph still records operations for
backward. bptt_mode='full' works unchanged. No gradient flow is lost.

**Compatibility with float64:**
`torch.compile` works with float64 on both CPU and GPU. No dtype changes needed.

**Compatibility with the checkpoint bptt_mode:**
The existing `grad_checkpoint` call in `simulate` wraps a `_step` closure. If
`rk4_step` is compiled, the closure captures the compiled function -- this is
compatible with `grad_checkpoint`.

---

#### Task P2.3 — Benchmark after compile and decide on next step

Run the same 20-epoch benchmark from Task P2.1 with `@torch.compile` on `rk4_step`.
Compare `t_fwd`, `t_bwd`, total epoch time.

**Expected outcome:**
- First epoch: slower than baseline (compilation warmup, ~1-5 s overhead once)
- Subsequent epochs: 2-5x faster on GPU; modest speedup on CPU (Python dispatch
  overhead is lower on CPU so the relative gain is smaller)

**Decision gate:**
- If speedup is sufficient for practical training (epoch time acceptable): stop here.
- If forward pass time is still dominant: proceed to Task P2.4.
- If backward pass time is dominant: the bottleneck is BPTT memory, not dispatch;
  consider enabling bptt_mode='checkpoint' for long segments (already implemented).

---

#### Task P2.4 (conditional) — Replace Python loop with torch.func.scan

**Only do this if Task P2.3 shows insufficient speedup.**

**What it does**: `torch.func.scan` (PyTorch 2.4+) is the PyTorch equivalent of
JAX's `lax.scan`. Instead of unrolling a Python for-loop into a massive FX graph,
it traces the loop body **once**, then executes a device-side loop directly on the
GPU. This gives O(1) compilation memory and time regardless of N, with the same
autograd support as a full BPTT Python loop.

**Why this and not torch.compile(simulate):**
Compiling `simulate` would cause TorchDynamo to unroll the N-step for-loop into a
monolithic FX graph with millions of nodes -- O(hours) to compile, potential OOM.
`torch.func.scan` avoids unrolling entirely: the RK4 body is traced once.

**File**: `lpv_lfr_baseline/core/lfr_simulate.py`

**Restructuring required:**
`torch.func.scan` expects a pure transition function of the form:
```python
def step_fn(carry, input_t):
    # carry: current state x  -- (batch, 6)
    # input_t: u at this timestep -- (batch, 3)
    # returns: (next_carry, output_t)
    x_next, z, w, y = rk4_step(carry, input_t, G, K, C, ...)
    return x_next, y  # (carry_out, scan_output)
```
Then replace the `for k in range(N)` loop in `simulate()` with:
```python
# u_seq_logical shape: (N, batch, 3) -- time-first, already the correct layout
x_final, Y_seq = torch.func.scan(step_fn, x0, u_seq_logical)
# Y_seq: (N, batch, 3) -- outputs stacked along the time dimension
```

**Constraints to verify before implementing:**
- `torch.func.scan` requires PyTorch >= 2.4. Check installed version:
  `conda run -n GraduationProject python -c "import torch; print(torch.__version__)"`
- The carry (x) and inputs (u) must be plain tensors or pytrees of tensors -- satisfied
- G, K, C and poly constants must be captured in the closure of `step_fn`, not passed
  as scan inputs -- this is fine since they are fixed within one `simulate()` call
- Autograd through scan is supported but was marked experimental in PyTorch 2.4.
  Test gradient flow with the same BPTT verification used in Task P2.1.
- The `return_latents=True` path (storing Z, W per step) requires returning z and w
  as scan outputs alongside y -- restructure the output tuple accordingly.
- `bptt_mode='truncated'` and `bptt_mode='checkpoint'` paths in `simulate()` become
  irrelevant once scan is used (scan handles its own backward); remove or guard them.

**Known issue (from Gemini source ref #6):**
`torch._higher_order_ops.scan` has a reported interaction with `clamp` operations
under `torch.compile` + `autograd` in early PyTorch 2.x. The `_recover_params()`
call uses `.clamp(min=1e-6)`. If this is captured in the scan closure and causes
issues, move the clamp outside the scan call (compute params once before calling
`simulate`, pass as closed-over tensors).

**Verification after implementing:**
- Run 20-epoch benchmark and compare against Task P2.3 result
- Verify `param_table()` matches Task P2.1 reference (same seed) to within 1e-6
- Confirm gradient flows to `log_params`: `block.log_params.grad` must be non-zero
  after `loss.backward()`

---

### Verification: numerical correctness after all changes

After Phase 1 + Phase 2 are complete, run a short training run (20 epochs) with the
same seed as a reference run (eager mode, post Phase 1, pre compile) and compare:
- `param_table()` output: learned parameters must be identical (or within float
  tolerance -- compilation may change floating-point evaluation order slightly)
- `val_rmse` per epoch: must match within 1e-6 relative tolerance

If values diverge beyond tolerance, suspect float reordering in the compiled graph.
Acceptable fix: use `torch.compile(..., options={"triton.cudagraph_trees": False})`
to disable aggressive graph optimization.

---

### Dead code removed by this plan (summary)

| What | Where | Why removed |
|------|-------|-------------|
| `@dataclass class GMatrix` | `lfr_matrices.py` | Replaced by (15,15) tensor |
| `from dataclasses import dataclass` | `lfr_matrices.py` | No longer needed |
| Module-level `G = build_G_matrix(...)` singleton | `lfr_matrices.py` | Only in __main__, not in pipeline |
| 7 individual buffer registrations `_G_Ax` ... `_G_Cy` | `lfr_block.py` | Replaced by single `_G` buffer |
| All `import GMatrix` / `from ... import GMatrix` | all files | Class removed |
| Any GMatrix type annotations | all files | Type is now `torch.Tensor` |

---

## Supervisor Meeting Notes — 2026-04-09

Notes and action items from meeting with TUe + ASMPT supervisors. Items are flagged here so they are encountered at the right step.

---

### [MEET-01] URGENT: Gantt chart / planning required for next meeting
**Raised by**: Maarten (third time — this is a hard requirement)
- [ ] **Before next meeting**: prepare a Gantt chart / planning covering remaining project milestones
- Include: parameter recovery completion, augmentation implementation, validation, thesis writing
- Supervisors need this to steer the project; do not attend the next meeting without it

---

### [MEET-02] Parameter updating — norm to tune cost function landscape
**Relevant at**: Step 3 (parameter recovery training)
- When parameters are not individually identifiable but their sum is, add a **norm term** to the
  cost function to tune the landscape and make individual parameters more identifiable
- This is a regularization strategy, not a penalty on the sum itself — it shapes the landscape
  so the optimizer can distinguish the components

**Logarithmic parameterization for Adam** (follow-up on D-035):
- If log-domain gradients are used, **centre and normalize** the log-parameters (around ~1)
  before computing the gradient update — Roland's suggestion
- Near zero / near 1 in log-space the gradient magnitude can be very different across parameters
  with different scales; centring corrects for this
- Alternative to log: `params * params` (square, always positive) or `abs(params * params)` —
  ensures positivity without log but gradient behaviour near zero still needs checking
- **Open question**: what is the cleanest guarantee that a parameter never reaches zero?
  → see D-035 for current log/exp approach; revisit if instability observed near small values
- **Open question (MEET-06 below)**: how important is uniqueness for the parameter updating?

---

### [MEET-03] LPV baseline is a state-space model, NOT LFR — LFR not exploited
**Relevant at**: Step 2 (LPV) and Step 3+ (augmentation)
- **Current state**: `torch.linalg.solve` is used to invert `M(Y)` at every step — this is
  numerically correct but gives **zero benefit from LFR structure**
- **LFR benefit for ASMPT**: LFR structure is almost essential for control design, even with a
  black-box augmentation on top — the structured plant model allows structured H-inf / mu-synthesis
- **Path to LFR**: express `M(Y)` invertibility as a rational function symbolically
  (MATLAB can do this), then the full forward pass does not require a matrix inverse at every
  timestep — the rational form is the LFR representation
- **Jan's interconnect framework**: does allow pure state-space (no LFR), but this trades away
  the control-design benefit; decision depends on project scope
- **SVD**: primarily beneficial for control design (fewer latent signals / lower-rank channel)
  — open question: how does SVD affect interpretability of learned states?
- **Unresolved**: in the parallel augmentation structure (D-003), the additive augmentation is
  one option; parallel in Jan's framework would allow orthogonality regularization — it is not
  yet clear whether switching to state-space (not LFR) loses that orthogonality benefit
- **Decision needed (see D-036 placeholder in decisions.md)**: commit to state-space only vs.
  invest in symbolic M(Y) inversion to recover LFR structure

---

### [MEET-04] Augmentation — physical interpretability of learned states
**Relevant at**: Step 4 (augmentation training)
- The augmentation result must be **physically meaningful** — not just low residual
- Additional learned states should be interpretable (e.g. map to a physical mode)
- **Open question**: how to enforce this? Options to investigate:
  - Regularization on the magnitude or structure of the additional states
  - Constrain state-space matrices of the augmentation block (e.g. passivity, sparsity)
  - Compare learned states against known unmodeled effects (Coriolis, Coulomb)

---

### [MEET-05] BFR low for X1 and X2 — expected, not a bug
**Relevant at**: Step 2/3 validation
- X1 and X2 have low BFR; reference is 0 (the Y-axis excites them only weakly via coupling)
- Signal amplitude is small relative to the error scale → BFR is noisy/low by construction
- Not an indicator of a model problem — document this in any results/thesis section that
  reports BFR per channel

---

### [MEET-06] Uniqueness of parameter updating
**Relevant at**: Step 3 (parameter recovery)
- Open question raised in meeting: how important is uniqueness (identifiability) for the
  parameter updating procedure?
- Directly related to MEET-02: if uniqueness is not guaranteed, norm regularization may be
  the primary tool to shape the cost landscape toward a unique minimum
- **Action**: review identifiability theory for the specific parameter set; check which
  parameter combinations appear only as sums in M(Y) and whether the trajectory excitation
  is rich enough to separate them
