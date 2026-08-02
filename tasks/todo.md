# Task Tracking

_Step 1 (Frozen LTI Baseline) completed and archived to `archive/sessions/2026-04-03-todo-step1.md`._
_Tasks 2.1, 2.3, 3b.1 and the 2026-06-10 code-review section archived to `archive/sessions/2026-07-02-todo.md`._

---

## ACTIVE 2026-08-02 — True-init augmentation: can the ANN learn the absorber at all?

Handoff: `tasks/handoffs/2026-08-02-true-init-augmentation.md`. Autonomous session (section 15).
Running log: `scripts/gantry/true-init-augmentation/IMPLEMENTATION-LOG.md`.
Not run: the todo/handoff archival sweep (would churn ~1000 stale lines and is outside the
handoff's scope); noted here so it is not mistaken for an oversight.

### A. Infrastructure
- [ ] A1 Create `scripts/gantry/true-init-augmentation/` + open the implementation log.
- [ ] A2 `truth_exact.py`: exact 8-state truth in Python (RK4, 20 kHz, from the rest IC
      `[0,0,Y_op,0,0,0,0,0]`), decimated to 4 kHz. Gives ANALYTIC velocities (integrator
      states), not `gradient()` finite differences. Gate: reproduces the record's positions
      at or below the known `5.37e-10 m` X level.
- [ ] A3 `plant_cog.py`: LFR baseline with the truth's static mass distribution at
      `delta_a = 0`. Derive the corrected `N0,N1,N2` and `d(Y)` analytically (the LFR
      rational form survives the correction; only `M[0,1]`, `M[1,1]` change).
      Gates: (a) `ma_cog = 0` reproduces the parent bit-identically;
      (b) `N_cog(Y)/d_cog(Y) == inv(M_cog(Y))` over a Y sweep;
      (c) cross-realization vs the collapsed 3-DOF `M3(Y, cog=True)`.

### B. Task item (i) — per-window target check (no training)
- [ ] B1 Free-run floor at THIS configuration (4 kHz, block-mean u, exact IC at t=0,
      chunked into 0.100 s windows) for ALL SIX physical states. Velocity thresholds are
      not in the literature; derive and state them before use (handoff section 10).
- [ ] B2 `diag_window_target.py`: re-seed each window from the exact 6-state truth IC over a
      grid of starts, roll the baseline forward `nf = 400`, report per-window mean and
      scatter on all six states. Arms: CoG off / CoG on, float64 / float32.
- [ ] B3 Verdict against the acceptance criterion (Y `2.979e-08`, X `9.147e-08`,
      Theta `3.730e-09` from the 20 kHz free run, plus the B1 floors). If any state lands
      materially above its floor: diagnose (code defect vs genuine property of per-window
      re-seeding), record, do not stop.
- [ ] B4 Second record class for error bars (backlog item 2, pulled forward if cheap).

### C. Task item (ii) — training arm
- [ ] C1 `true_init_train.py`: pipeline mirror with the SUBNET encoder REPLACED by the exact
      6-state truth IC per window (rows 6-7 stay at zero, never seeded: handoff section 8).
      Encoder bypass via a `make_training_data`/`loss` override in the new folder; nothing
      in `model_augmentation/` is modified.
- [ ] C2 Validation measure consistent with the true-init target (windowed free run from the
      exact IC), so checkpoint selection is not made on an encoder metric that no longer exists.
- [ ] C3 Hypothesis row in `docs/gantry-augmentation-problem-log.md` section 12 BEFORE launch
      (run-discipline rule, not overridden by section 15).
- [ ] C4 Run ANN-on vs ANN-off on the same seeds; report whether the ANN learns.
- [ ] C5 Attribute the outcome to one of the section-8 candidates (persistence / coordinate
      pinning / capacity). Do NOT implement either fix.

### D. Reporting
- [ ] D1 `IMPLEMENTATION-LOG.md` written as the work happens, opening with
      `## 0. Read this first`.
- [ ] D2 Draft (do NOT apply) the D-130 amendment inside the new folder's log.
- [ ] D3 Commit working increments on `Augmentation`. No push, no PR.

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

_Step 3c (speed optimization, reverted) and Task 3b.4b (multi-traj loss fix, done) archived to `archive/sessions/2026-06-10-todo.md`._

---

## Step 5: Systematic Augmentation Evaluation (2026-06-10)

**NOTE TO NEXT SESSION**: This plan was drafted at the end of a long code-review session.
You are free to argue with it, restructure it, or reject parts of it. Be critical. The
reasoning below may have blind spots. Challenge the sweep ranges, the run matrix, and
especially any implicit assumptions before implementing. If something doesn't make sense,
push back on the user and discuss before coding.

### Goal

Justify hyperparameter choices (downsampling rate, nf, na_nb) empirically, then run a
controlled comparison of encoder types x data types to characterize what the augmentation
learns and why.

### Phase 1: Hyperparameter justification (sweeps)

Justify each choice with a physics argument AND an empirical sweep. The physics argument
sets the expected range; the sweep confirms it and picks the value.

**1a. Downsampling rate (fs)**
- Physics argument: highest mode is MSD at 150 Hz. Nyquist requires >300 Hz. At fs=4000 Hz
  we have 13x oversampling. The multisine excitation band is [1, 200] Hz, so nothing above
  200 Hz is excited.
- Empirical: sweep fs in {1000, 2000, 4000} Hz. Run baseline-only (NX_ANN=0) forward sim
  on one trajectory, report sim-RMS. No training needed. Shows where decimation starts to
  lose information.
- Challenge this: is 1000 Hz even worth testing given the 150 Hz mode? Is the RK4 substep
  count (10x) still adequate at lower fs?

**1b. Rollout horizon nf**
- Physics argument: MSD settling time tau = 1/(2*pi*150*0.05) ~ 21 ms. 5*tau ~ 106 ms.
  The MSD ring-down must be visible in the rollout for the loss to learn it. Current
  nf = 1200 samples = 300 ms at 4 kHz (~ 14*tau), which is conservative.
- Empirical: sweep nf in {200, 400, 800, 1200} samples (50, 100, 200, 300 ms). Train
  with default encoder, NX_ANN=2, multisine data. Report sim-RMS and state recovery R2.
- Challenge this: does very long nf cause vanishing gradients? Is the settling time
  argument even the right one for BPTT (see lessons.md rule on context mismatch)?

**1c. Encoder history na_nb**
- Physics argument: encoder needs to see at least one full MSD oscillation period
  (1/150 Hz ~ 6.7 ms) and ideally some decay. 100 ms = 400 samples captures ~15 periods.
- Empirical: sweep na_nb in {50, 100, 200, 400} samples (12.5, 25, 50, 100 ms). Same
  training setup as 1b.
- Challenge this: does the default ANN encoder even use the temporal structure, or does
  it just flatten everything? If flattened, more history = more parameters = harder to train.

**Sweep logistics**: Each sweep varies one parameter, holds others at default. Use short
training (e.g. 50 epochs) to see trends, not convergence. Save sim-RMS and R2 per run.

### Phase 2: Baseline model mismatch quantification

Before any augmentation training, quantify what the baseline model gets wrong.

- [ ] Forward-simulate the 6-state physics-only model (NX_ANN=0, no ANN, no training) on
      both multisine and trajectory validation data. Report per-channel sim-RMS.
- [ ] This is the "model mismatch floor": the error the ANN must absorb.
- [ ] Compare multisine vs trajectory mismatch. If multisine data excites the MSD more,
      the mismatch should be larger (the hidden mode is more visible).
- [ ] Print std_x per channel for both data types. Quantifies the normalization conditioning
      (if std_x[1] for theta is 1000x smaller than std_x[0], that explains poor theta recovery
      regardless of encoder choice).

### Phase 3: Controlled 2x2 comparison

With hyperparameters fixed from Phase 1, run the full matrix:

|                  | Multisine data | Trajectory data |
|------------------|----------------|-----------------|
| Default encoder  | Run A          | Run B           |
| Hybrid encoder   | Run C          | Run D           |

Also include the physics-only baseline from Phase 2 as reference (no encoder, no ANN).

**Per run, report:**
1. sim-RMS (val) per channel (X1, X2, Y)
2. State recovery R2 diagnostic (R2_raw, R2_linmap per physical state channel)
3. ANN latent state RMS (are the augmented states active?)
4. Loss convergence curve
5. Per-state gradient norm during training (encoder ANN params + augmentation ANN params,
   logged per epoch). If theta's gradient is orders of magnitude below others, that is a
   quantitative explanation for poor recovery.

**Interpretation matrix:**
- Run A vs B: does multisine excitation improve augmentation learning?
- Run C vs D: same question for the hybrid encoder
- Run A vs C: does fixing the encoder basis (hybrid) improve state recovery?
- Run B vs D: same question on trajectory data
- All vs baseline: how much does augmentation reduce sim-RMS vs physics-only?

### Phase 4: Analysis and conclusions

- [ ] Which states are recovered, which are not, and why (gradient magnitude, std_x
      conditioning, output sensitivity via Cd_norm)
- [ ] Does the default encoder rotate the basis (R2_linmap >> R2_raw)?
- [ ] Does the hybrid encoder eliminate the rotation?
- [ ] Does multisine data make the MSD states more identifiable?
- [ ] Is theta recovery fundamentally limited by output sensitivity (PBH/observability),
      or just poorly conditioned (fixable with better normalization/loss weighting)?

### Open questions for the next session to resolve

- Should the gradient logging be per-parameter-group or per-state-channel? Per-channel
  is more informative but may require hooking into the backward pass.
- Is 50 epochs enough for the sweeps, or do some hyperparameters only show their effect
  at convergence?
- Should we include a "masked ANN" variant (ANN corrections only on augmented state
  channels 6:8, not on physical channels 0:6) as a fifth run? This directly tests the
  basis-rotation hypothesis.
- The PBH observability test scripts exist in `Matlab-scripts/Augmentation/diagnostics/`.
  Should we run them first to get an analytical answer on theta observability before
  spending compute on training runs?

---

## Telica Training: Remaining Items (2026-07-04)

_CLOE-prerequisite diagnostics (C.1-C.3, resolved 2026-07-03 per D-069/D-073/D-074) and the
completed train/val/test split wiring (D-075) archived to `archive/sessions/2026-07-04-todo.md`._

- [ ] Server run: sync dFeedbackControllersTelica_ba.mat + Telica 1.mat, then launch
- [ ] NOTE: measured epoch time on laptop CPU is ~5 min (11 gradient steps x 650 BPTT samples,
      batch 22); 40 epochs + validation ~= 4 h laptop. Expect faster on server (GPU preload built in).

---

## Joint Estimation — Parameterized Gantry Block (2026-07-04, D-076)

**Goal**: trainable damping+stiffness `[kb_sum, cg1, cg2, cy, cb_sum]` in the multisine pipeline
(`gantry_interconnect_dynamic.py`); machinery reusable for the absorber gray-box block (Option A).
Diagnostics are part of the phases; each gate blocks the next phase. Full design in D-076.

### Phase 0 — Bookkeeping
- [x] D-076 logged in docs/decisions.md
- [x] This plan added to tasks/todo.md

### Phase 1 — Block + Gate 1 (DONE 2026-07-04)
- [x] 1.1 Reference capture BEFORE any code change -> `ref_fixed_block.npz` (100x64x6 rollout, seeded)
- [x] 1.2 Parent refactor: `_mats()` hook in `Gantry_State_Block` (verified behavior-neutral: A0)
- [x] 1.3 `Parameterized_Gantry_State_Block` (@added, below parent in blocks.py)
- [x] 1.4 Gate 1 PASS: A0 max|diff| = 0.0 (bitwise), A1 max|diff| = 0.0 (bitwise),
      B all 5 grads finite/nonzero, FD-vs-autograd max rel err 2.5e-7 (tol 1e-5).
      Results: `simulations/gantry_subnet/diagnostics/joint_estimation/gate1_results.json`

### Phase 2 — Trainer + Gate 2 (DONE 2026-07-05)
- [x] 2.1 `SSE_Interconnect_ParamLoss` in `interconnect.py` (@added delegating loss + one
      `# CHANGED:` import line)
- [x] 2.2 Gate 2 PASS: C exact (rel diff 4e-10); D recovery with state-readout harness:
      gap ratios kb_sum 0.000, cg1 0.211, cg2 0.189, cy 0.001, cb_sum 0.055 (pass <= 0.5).
      Forward overhead of param block: +11.7% (fwd+bwd 1.09 s/batch at nf=100, batch 128).
      Results: gate2_results.json; logs gate_run*.log (attempts 1-2 preserved).
- [x] FINDING (Phase 3/4 input, discuss with Jan): with position-only outputs, short-window
      BPTT and a co-trained encoder, the ENCODER absorbs parameter error (params practically
      unidentifiable: attempt 1 = checkpoint trap best@epoch0 under sim-RMS val, the known
      K=0 horizon mismatch; attempt 2 = loss down 1000x with params frozen at init). Also:
      param_loss anchored to the INIT values pins params there once the MSE landscape
      flattens. Real JE runs need deliberate design: windowed validation, encoder
      warm-start/freeze or longer nf, and awareness that reg anchors to init.

### Phase 3 — Script flag + rehearsal (DONE 2026-07-05)
- [x] 3.1 `gantry_interconnect_dynamic.py`: `JOINT_ESTIMATION` flag (gates block class only),
      `PARAM_RMSE_BASELINE = 0.01` (# HEURISTIC, jobs 68675/68676; verified = Jan's own
      RMSE_baseline pattern, cf. his hardcoded 0.2 in Parameterized_MSD_State_Block),
      unconditional `SSE_Interconnect_ParamLoss`, param table + npz fields
      (`params_init`/`params_learned` + flags in config json), pre-JE resume guard
- [x] 3.2 Rehearsals passed (epochs=1/nf=10, reverted after): flag ON exit 0 with param table,
      npz fields, `log_params` in checkpoint; flag OFF exit 0 with LOCAL anchor
      0.00056488684 == flag-ON value bit-exact (cluster anchor 6.4948e-4 not reproducible
      locally: cluster used the finite-diff normalization fallback, local has
      baseline_states.npz). Logs: diagnostics/joint_estimation/rehearsal_on|off.log
- [ ] NOTE cluster sync: commit first; verify deployed copy with
      `grep -n "JOINT_ESTIMATION" scripts/gantry/gantry_interconnect_dynamic.py` and
      `grep -n "class Parameterized_Gantry_State_Block" model_augmentation/fit_systems/blocks.py`
      and `grep -n "class SSE_Interconnect_ParamLoss" model_augmentation/fit_systems/interconnect.py`

### Phase 4 — Runs (decision point; run design agreed 2026-07-05)
- [x] Run design: pilot pair on cluster at 5 epochs (~2.5 h each), then full 30-epoch versions
      if pilots are clean. Run T = `JOINT_ESTIMATION=True`, `PARAM_INIT_DETUNE=None` (start at
      TRUE values; drift measures absorber-induced parameter bias; reg anchors AT truth).
      Run D = same + `PARAM_INIT_DETUNE=[1.10,1.10,0.90,1.10,0.90]` (lpv_lfr_baseline detuning
      signs; in-pipeline recovery test; NOTE reg anchors at the DETUNED init — Jan's prior
      semantics, honest hardware simulation). Both otherwise identical to 68675 (seed, ANN,
      nf=400, sim-RMS val) for comparability.
- [x] `PARAM_INIT_DETUNE` knob added to gantry_interconnect_dynamic.py; verified via
      build_model: detuned block init = [4372.5, 15.95, 18.27, 11.0, 16.2] exactly (PASS)
- [ ] Commit (scoped D-076 arc) -> sync -> verify greps (above) -> launch pilots T and D
- [ ] Align routing question (Option A/B, P1 ladder) with Jan (draft note prepared 2026-07-05);
      no-ANN bias ablation (R-B) decided AFTER Jan reply — may be superseded by Option A

### Joint Estimation v2 — all 14 raw physical params (2026-07-05, D-077)
**Goal**: extend the block from 5 damping/stiffness sums to ALL 14 raw physical params
(mirror `train_param_recovery`: train raw, trust the 10 identifiable combinations). Enabled
by the M(Y)-invertibility proof (positive params -> PD for all Y).
- [x] Block: parent `_mats()` hook widened 3 -> 10 (carries mh/alpha/beta/gamma_/N0/N1/N2);
      child rewritten to 14 raw log-params with full per-timestep M(Y) rebuild
      (`build_poly_constants` + `build_G_matrix_entries`); `identifiable_combinations()` /
      `param_table()` report the 10 combos; `m_diff` signed, derived from logged m1,m2.
- [x] Gate 1 PASS: A0/A1 bitwise 0.0 (full nominal rebuild exact); B all 14 grads under
      gradcheck-style atol+rtol tol; NEW check M `max|M·N/d - I|=6.66e-16`, `min eig 2.97>0`
      over 200 positive-orthant samples.
- [x] Gate 2 PASS: C rel 1.16e-9; D recovered ALL 10 combinations (v1 subset <=0.141 gated;
      masses mh 0.001 / m_total 0.000 / m_diff 0.002 (signed +1.59->-0.496) / J_eff 0.028 /
      d 0.000 reported). Overhead: fwd +15.9% vs fixed; fwd+bwd 1.49 s/batch (+37% vs v1 in
      isolation, less in the full pipeline).
- [x] Script: `build_model` 14-nominal from `gantry_ss` + `param_table()` report; validated
      via no-train build rehearsal (params_init exact, combination table correct).
- [x] D-077 logged (design + gate results).
- [ ] Commit the D-076/D-077 arc (scoped) -> cluster sync -> verify greps.
- [ ] Phase 4 run design now uses a 14-vector `PARAM_INIT_DETUNE` (aligned to PARAM_NAMES);
      Run T = None (start at truth), Run D = 14-vector detune. Runtime ~+15-40% over v1
      depending on ANN/window share.

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
