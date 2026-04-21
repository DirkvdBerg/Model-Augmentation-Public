# Design Decisions

Decisions are logged here before implementation. Each entry states what was decided, why, what was ruled out, and what it constrains going forward.

---

## Decision Template

```
### [D-NNN] Title
**Date**: YYYY-MM-DD
**What**: What was decided.
**Why**: The reason — constraint, evidence, or trade-off that drove the choice.
**Ruled out**: Alternatives considered and why they were rejected.
**Constrains**: What future decisions or implementations this locks in.
```

---

## Decisions

### [D-001] Target system is the ASMPT dual-gantry (García-Herreros et al.)
**Date**: 2026-03-16
**What**: The sole target system for this project is the ASMPT dual-gantry stage modeled by García-Herreros et al.
**Why**: This is the industrial use case for the graduation project. All other benchmarks (MSD, Bouc-Wen, Cascaded Tanks) are reference implementations of the augmentation framework only.
**Ruled out**: Using MSD or other benchmarks as the target system.
**Constrains**: All new code, data pipelines, and model structures must be built around the gantry system.

---

### [D-002] MATLAB files in `kamtin-fp-model/` are immutable
**Date**: 2026-03-16
**What**: The MATLAB model files defining the FP model structure are the ground truth and must never be modified.
**Why**: They represent the validated physical model from García-Herreros et al. and are the hard constraint that the Python implementation must conform to.
**Ruled out**: Adapting the MATLAB model to fit the Python code. The direction of adaptation is always MATLAB → Python, never the reverse.
**Constrains**: Any Python state-space implementation must reproduce the structure defined in the MATLAB files exactly.

---

### [D-003] Augmentation structure is parallel dynamic LFR
**Date**: 2026-03-16
**What**: The augmentation architecture is a parallel dynamic structure within the LFR framework.
**Why**: Parallel structure is required for orthogonal projection-based regularization (Gyorok et al.), which prevents the learned component from capturing dynamics already described by the baseline. Dynamic (not static) augmentation is needed because cross-coupling and position-dependent flexible dynamics require additional learned states beyond the baseline.
**Ruled out**: Series interconnection (incompatible with orthogonal projection regularization); static augmentation (cannot capture dynamics requiring additional states).
**Constrains**: The LFR interconnection must be realized as a parallel structure. Regularization implementation follows Gyorok et al.

---

### [D-004] Scheduling variable is payload position Y
**Date**: 2026-03-16
**What**: The LPV scheduling variable is the payload position Y.
**Why**: Y enters the inertia matrix algebraically in the García-Herreros model, making it the natural scheduling variable. Since Y is a system state (not an exogenous signal), the formulation is quasi-LPV. Y is directly available from the physical model and does not need to be identified from data.
**Ruled out**: Data-driven scheduling variable identification (not needed here since Y follows from the physics).
**Constrains**: The LPV discretization must handle Y as a state-dependent scheduling variable. Invertibility of the position-dependent inertia matrix must be verified across the full operational range.

---

### [D-006] Python implementation uses stage coordinates
**Date**: 2026-03-16
**What**: The Python discrete-time state-space model is implemented in stage coordinates: states q = [X1, X2, Y, dX1, dX2, dY], inputs u = [F_X1, F_X2, F_Y], outputs y = [X1, X2, Y].
**Why**: Real experimental gantry data is measured in stage coordinates (X1, X2, Y from encoders; F_X1, F_X2, F_Y from amplifiers). The model must match the data — the model is coordinate-independent, so the data determines the choice. The MATLAB model also discretizes in stage coordinates (`c2d(StageCoordinatesSystem, ts, 'zoh')`), providing a direct reference.
**Ruled out**: Logical coordinates [X, Θ, Y] — the augmentation framework trains on measured data, which is in stage coordinates. Working in logical coordinates would require transforming every data sample and adds no benefit.
**Constrains**: The A, B, C, D matrices passed to the augmentation blocks must be in stage coordinates. Normalization statistics (T_x, T_u, T_y) must also be computed from stage-coordinate data.

---

### [D-009] One file per responsibility — scripts import from gantry_ss.py, not duplicate it
**Date**: 2026-03-17
**What**: Each script in `scripts/gantry/` has a single responsibility. `gantry_ss.py` is the sole definition of the model (physics → discrete A, B, C, D). All other scripts (simulation, validation, augmentation wiring) import `gantry_discrete_ss()` from it rather than redefining the matrices.
**Why**: Avoids parameter duplication — if a physical parameter changes, it changes in one place only. Makes the boundary between "model definition" and "model use" explicit.
**Ruled out**: Copying A, B, C, D into each script — creates silent inconsistencies if parameters are updated.
**Constrains**: Any script that needs the discrete model must import from `gantry_ss.py`. Extensions (LPV variant, different Y) are added as new functions in `gantry_ss.py`, not in the calling scripts.

---

### [D-008] Fixed SISO-only bug in modified_encoder_net; kept local copy over deepSI default
**Date**: 2026-03-16
**What**: Uncommented line 361 in `model_augmentation/fit_systems/interconnect.py` so `self.ny` is set from the `ny` argument instead of hardcoded to `tuple()`.
**Why**: The original code forced `np.prod(self.ny) = 1` regardless of actual ny, making the encoder input `nb·nu + na·1` even for MIMO systems. For the gantry (ny=3) this would silently drop output channels 2 and 3 from encoder history, giving input size 40 instead of the correct 60.
**Verified**: Unit test confirmed SISO (ny=1) input unchanged at 20; MIMO (ny=3) input now 60 (was 40).
**Ruled out**: Replacing `modified_encoder_net` with deepSI's `default_encoder_net` — kept local copy to allow gantry-specific encoder extensions later. The two are now functionally identical.
**Constrains**: Nothing locked in — local copy can still be extended independently of deepSI upstream.

---

### [D-007] Implement fixed baseline first, add trainability in a second step
**Date**: 2026-03-16
**What**: The Python FP model is first implemented as a fixed (non-trainable) baseline using `Linear_State_Block` and `Linear_Output_Block`. Trainability (`Parameterized_Linear_State_Block` / `Parameterized_Linear_Output_Block`) is added only after the fixed baseline is validated end-to-end in the augmentation interconnect.
**Why**: Stepwise approach reduces the number of failure modes at each stage. A fixed baseline is easier to verify (output is deterministic and can be compared directly against the MATLAB `G` matrices). Trainability introduces regularization and gradient flow, which should only be debugged once the structural wiring is confirmed correct.
**Ruled out**: Going straight to parameterized blocks — adds trainable parameters and param_loss complexity before the block shapes, wiring, and normalization are validated.
**Constrains**: Validation milestone required before promoting to parameterized blocks: simulated output from the Python baseline must match the MATLAB `c2d` matrices to numerical tolerance.

---

### [D-005] LFR structure confirmed for the LPV augmentation
**Date**: 2026-03-16 (updated 2026-03-20)
**What**: The augmentation framework will use an LFR structure for the LPV scheduling. This was initially deferred but was confirmed as the right approach by the supervisor in the meeting of 2026-03-20.
**Why**: The supervisor stated: "LFR gives more flexibility. Can always compute a state-space representation if we want to remap. Suggestion: start with LFR structure for scheduling/LPV." The LFR parameterization allows the learned correction to vary with Y in a principled way through the delta-p block (see D-017). Rank of the M matrix across different trajectories should be computed to confirm no rank drop occurs (expected to be fine, but must be verified).
**Ruled out**: Pure state-space augmentation without LFR structure. Deferring LFR indefinitely (supervisor explicitly suggested it as the starting point for the LPV scheduling).
**Constrains**: Step 3 implementation targets the LFR structure for LPV scheduling. A paper on discretizing LFRs must be found and reviewed before implementation (supervisor action item from 2026-03-20 meeting). The CT conversion must be written up first before the LFR structure is implemented (see D-018).

---

### [D-010] LPV baseline and LPV augmentation are separate concerns
**Date**: 2026-03-17
**What**: The LPV extension has two distinct parts that must not be conflated:
  1. **LPV baseline** — the FP model with A(Y[k]), B(Y[k]) recomputed each step from physics. This is what Step 2 builds and validates.
  2. **LPV augmentation** — a data-driven network on top of the baseline that also varies with Y. This is a Step 3+ concern.
**Why**: Jan's original augmentation framework has no LPV support. The `Parameterized_LPV_Affine_Linear_State_Block` found in the codebase is a user-added augmentation component, not a baseline block. Treating it as the LPV baseline would conflate two separate responsibilities.
**Ruled out**: Using `Parameterized_LPV_Affine_Linear_State_Block` as the LPV baseline block — it is trainable, augmentation-side, and uses an affine-in-Y² approximation that does not represent the full physics.
**Constrains**: Step 2 validates the LPV baseline purely in Python (no framework). Step 3 requires a new `LPV_Linear_State_Block` (see D-011).

---

### [D-011] Framework integration of LPV baseline requires a new block type
**Date**: 2026-03-17 (updated 2026-03-22)
**What**: Wiring the LPV baseline into the augmentation interconnect requires a new block, `CT_RK4_State_Block`, that reads Y from the current state at each forward call and integrates the CT ODE using one RK4 step.
**Why**: The existing `Linear_State_Block` stores A and B as fixed attributes set at init, so it cannot update them per step. The LPV baseline needs physics that change every timestep as Y evolves. No existing block in the framework supports this.
**Ruled out**: Reusing `Linear_State_Block` with a single frozen operating point (that is the frozen LTI). Reusing `Parameterized_LPV_Affine_Linear_State_Block` (wrong structure: affine-in-Y², trainable, augmentation-side).
**Constrains**: The block computes A_c(Y), B_c(Y) from physics at each step and applies RK4 with dt=ts (see D-018). The baseline should also be expressed in LFR form for compatibility with Drenth's augmentation procedure (see D-005, updated 2026-03-22). Y is read from state index 2 in stage coordinates (self-scheduled).

**Update 2026-03-22**: Changed from `LPV_Linear_State_Block` calling `gantry_discrete_ss(Y)` (pre-discretized DT) to `CT_RK4_State_Block` integrating the CT ODE with RK4 (per D-018). Additionally, the baseline should be expressed in LFR form per supervisor confirmation (D-005).

---

### [D-012] LPV discretization: frozen ZOH for validation, exact ZOH via matrix_exp for training
**Date**: 2026-03-17 (updated 2026-03-18)
**What**: Two discretization approaches are used, one per use case:
  1. **Validation (Step 2)**: frozen-at-sampling-instant — call `cont2discrete(A_c(Y[k]), ts)` at each step.
  2. **Training loop (Step 3)**: exact ZOH via `torch.linalg.matrix_exp(A_c(Y) * ts)`. Fully torch-differentiable (confirmed by test).

**Theoretical status — quasi-LPV caveat (important)**:
  Tóth (2010) states the ZOH setting is *"only reasonable for the discretization of LPV-SS
  representation with static dependence as dynamic dependence requires a higher-order hold
  approach"* (Section I, page 2).
  Our system is **quasi-LPV with dynamic dependence**: Y = x(3) is a system state, not an
  exogenous signal. Within each sampling interval, Y evolves continuously as the state
  integrates — it is not truly held constant by ZOH. Consequently:
  - The "errorless" property (Tóth Section IV-A: *"The complete method theoretically provides
    errorless discretization in terms of the ZOH setting"*) applies strictly to static
    dependence only.
  - For our system there is a **small but nonzero residual intra-sample error** from the
    within-interval variation of Y.

**Formal requirements from Tóth (Assumptions 1 and 2, page 5–6)**:
  - Assumption 1 (ZOH setting): *"We are given a CT-LPV system S, with CT input signal uc,
    scheduling signal pc, and output signal yc, where uc and pc are generated by an ideal ZOH
    device and yc is sampled in a perfectly synchronized manner with Td > 0 as the sampling
    period or discretization time-step."*
    Satisfied: our 20 kHz discrete control loop holds u_c and p_c (=Y) constant within each
    50 µs sample interval ✓
  - Assumption 2 (Switching effects): *"The switching behavior of the ZOH actuation has no
    effect on the CT plant, i.e. the switching of the signals is assumed to take place smoothly."*
    Tóth notes: *"this assumption is automatically satisfied in most numerical simulations of
    LPV systems, like in the implemented numerical approaches of SIMULINK in MATLAB."*
    Satisfied: Y changes continuously — no discontinuous jumps; our Python numerical simulation
    mirrors the SIMULINK approach Tóth explicitly endorses ✓
  Note: Tóth provides no quantitative bound on dp/dt. The qualitative remark on page 20
  (*"p_c changes smoothly and relatively slowly with respect to the actual dynamics of the
  plant"*) is motivating prose, not a formal condition.
  Closed-loop applicability: *"The presented ZOH setting is also applicable for closed-loop
  controllers in the structure given in Figure 2"* — our closed-loop Python simulation is
  within the scope Tóth explicitly covers.

**Self-scheduling vs external scheduling**:
  Tóth's Assumption 1 requires p[k] to be held by an ideal ZOH device -- it must be
  *measurable* (externally available) at each step k, not predicted from internal state.
  This implies external scheduling: Y[k] is read from the encoder at step k and held for
  that interval.

  Using Y[k] = x_predicted[k][2] from the model's own state (self-scheduling) introduces
  a further approximation on top of the dynamic dependence caveat already accepted above:
  - Dynamic dependence caveat: Y is a state, not an exogenous signal -- ZOH is approximate.
  - Self-scheduling: Y[k] itself is approximate (from predicted state, not measured). If the
    open-loop state drifts, the scheduling variable is wrong, compounding the error.

  External scheduling (Y[k] from measurement) is more consistent with Tóth and is used
  wherever measurements are available:
  - Training loop: Y[k] = x_measured[k][2] from real data (external, consistent with Tóth).
  - Validation against q1: Y[k] = Y_trajectory[k] from the MATLAB reference (external).
  Self-scheduling is reserved for autonomous simulation with no external measurements and
  carries the additional compounding approximation noted above.

**A_c invertibility (Tóth footnote 2)**:
  Tóth writes the complete discretization formula assuming A_c invertible *"for convenience"*
  but footnote 2 states: *"To compute the resulting matrix functions of this discretization
  approach, Ac(p) is not required to be invertible, but if it is, we can write the resulting
  DT description of the state-evolution conveniently as (9a)."*
  Our A_c is singular (rigid body modes → top-left 3×3 block is zero). The naive formula
  B_d = A_c⁻¹(A_d − I)B_c is therefore undefined. The augmented matrix exponential (D-015)
  is the correct general form — directly supported by Tóth's own footnote.

**Practical justification for small residual error**:
  The intra-sample Y variation is bounded by ΔY ≤ 0.100 mm/sample
  (= v_max × ts = 2 m/s × 50 µs; v_max from ETEL datasheet and main.m vmax=2).
  Physical timescale argument: Y traverses its full 700 mm operational range (ETEL datasheet,
  5% margin from 800 mm stroke) at maximum speed in ≥ 350 ms = 5600 samples, while the
  plant's fastest relevant dynamics act on the closed-loop bandwidth timescale
  ~1/(2π×100 Hz) ≈ 1.6 ms (fbw=100, main.m) — a ~220:1 timescale separation. This makes
  the intra-sample Y variation negligible in practice. Rigorous numerical confirmation:
  ‖A(Y+ΔY) − A(Y)‖/‖A(Y)‖ at ΔY = 0.125 mm is verified in Task 2.5.

**RESOLVED: sample rate set to 20 kHz (matching PLTI spec)**:
  AccurET-Oper&Soft-VerV.pdf confirms PLTI = 50 µs (20 kHz), matching the position control
  loop rate. main.m, export_lpv_sim.m, export_lpv_matrices.m, and physics.py all updated
  to fs = 20e3 (T_d = 50 µs). ΔY_max = 2 × 50e-6 = 0.1 mm — strengthens the slowly-varying
  argument relative to the old 0.125 mm/sample at 16 kHz.

**Why**:
  - Validation: `cont2discrete` is exact for the frozen ODE and fast enough for a one-off
    simulation. The residual quasi-LPV error is accepted as small (see above).
  - Training: `torch.linalg.matrix_exp` is a native PyTorch op — autograd traces through it,
    gradients flow back to Y[k]. The rectangular approximation (Option D, O(ts) error) is a
    valid fallback but is strictly inferior — there is no reason to accept approximation error
    when the matrix exponential is differentiable.
**Ruled out**:
  - Polynomial expansion (Option A): A_c(Y) is rational (from M(Y)⁻¹), so no exact polynomial A_d(Y) exists.
  - Linear-affine approximation (Option B): drops dominant Y² term in M[1,1].
  - Grid interpolation (Option C): not natively torch-differentiable.
  - Rectangular approximation (Option D): O(ts) error — valid fallback only. Superseded by Option E.
  - scipy `cont2discrete` in training loop: not inside autograd graph.
**Constrains**: `LPV_Linear_State_Block.forward()` must compute A_c(Y) analytically from M(Y)⁻¹ using tensor ops, then apply `torch.linalg.matrix_exp(A_c(Y) * ts)`. See `docs/lpv-discretization.md` for full rationale and option comparison table.

**Update 2026-03-20 (supervisor meeting)**: For the augmentation training loop (Step 3+), the discretization approach shifts from pre-discretized ZOH to CT model with RK4 integration. The ZOH approach remains valid for Step 2 validation (completed). See D-018, which supersedes the "training loop" part of this decision. Read D-012 as: Steps 1-2 validation used ZOH (done); Step 3+ training loop uses RK4 on the CT model (see D-018).

---

### [D-013] LPV baseline uses LFR form with CT+RK4 integration
**Date**: 2026-03-17 (updated 2026-03-22)
**What**: The LPV baseline must be *available* in LFR form {M^b, Δ^b(Y)} and integrated using RK4 inside a custom `CT_RK4_State_Block`. The `SSE_Interconnect` wiring machinery is used unchanged. Internally, the forward simulation may collapse to evaluating an equivalent CT vector field A_c(Y)x + B_c(Y)u (as Drenth Ch. 2 eq. 2.29 confirms), but the baseline must remain representable in LPV-LFR form for compatibility with Drenth's augmentation framework (Ch. 5 eq. 5.1-5.2).
**Why**: Supervisor confirmed (2026-03-22) that the baseline itself should use the LFR structure. Drenth Ch. 5 eq. 5.1 assumes the baseline is available in LPV-LFR form. Self-scheduled quasi-LPV (Y from state) is supported. The LFR representation of the baseline requires converting A_c(Y) with its rational M(Y)^{-1} entries into LFR form using standard LFT realization methods (Zhou, Doyle & Glover, 1996).
**Ruled out**: Computing A_c(Y) directly without LFR form (originally chosen, but revised per supervisor guidance). New `SSE_Interconnect` subclass (existing class is sufficient).
**Constrains**: The baseline LFR must be realized from the known physics. Normalization is handled by Drenth eq. 5.5: T_x, T_u, T_y scaling applies to all LFR submatrices. The conversion requires choosing η (repetition count in Δ) and verifying LFR well-posedness. One implementation detail remains open: whether runtime code evaluates the explicit LFR loop or the equivalent collapsed CT vector field. See `docs/lpv-lfr-interconnect.md` for the original assessment (partially superseded by this update).

**Update 2026-03-22**: Major revision. Original decision said LFR is NOT required for the baseline. Supervisor confirmed the opposite: use LFR structure for the baseline. Also changed from pre-discretized A_d(Y), B_d(Y) to CT+RK4 (per D-018). Normalization question is answered by Drenth eq. 5.5.

---

### [D-014] gantry_discrete_ss stays numpy; torch version lives in a separate file
**Date**: 2026-03-17
**What**: `gantry_ss.py` / `gantry_discrete_ss()` is not modified to support PyTorch. A separate file `scripts/gantry/gantry_lpv_torch.py` holds a torch-native implementation that mirrors `gantry_discrete_ss` in structure but uses tensor ops and `torch.linalg.matrix_exp` throughout.
**Why**: Two entirely different use cases with different dependencies and contracts:
  - `gantry_discrete_ss`: numpy in, numpy out, scipy `cont2discrete`, validation and MATLAB comparison only. Pure, simple, zero framework dependency.
  - torch version: torch tensor in, torch tensor out, differentiable, lives inside the training loop. Must stay inside the autograd graph.
  Adding a `use_torch=True` flag to `gantry_discrete_ss` would mix two concerns, add a conditional dependency on torch in a validation-only file, and violate D-009 (one file per responsibility).
**Ruled out**: Modifying `gantry_discrete_ss` to support a torch mode via flag — mixes validation and training concerns in one function.
**Constrains**: `gantry_lpv_torch.py` is a full torch reimplementation — NOT a wrapper around `gantry_discrete_ss`. Every value (physical parameters, M(Y), A_c, B_c, P transform, A_d, B_d) is defined as a `torch.tensor` from the start. No numpy intermediates, no conversion. This ensures gradients flow through the entire computation and physical parameters can optionally be made trainable later without refactoring. The only structural change from `gantry_ss.py` is replacing `cont2discrete` with `torch.linalg.matrix_exp` on the 9×9 augmented matrix (see D-015).

---

### [D-015] B_d(Y) must use augmented matrix exponential — naive formula fails
**Date**: 2026-03-17
**What**: Computing B_d(Y) via the naive formula `B_d = A_c⁻¹ · (A_d − I) · B_c` is forbidden. The correct formula uses the augmented matrix exponential:
```
M_aug = [[A_c(Y),  B_c(Y)],    # (n+m) × (n+m) = 9×9 for gantry
         [  0,        0   ]]

[A_d, B_d] = expm(M_aug · ts)[:n, :], split at column n
```

**Mathematical background**:
  The general ZOH formula for B_d (Tóth complete method, always valid) is:

    B_d = [∫₀^{T_d} exp(A_c · τ) dτ] · B_c

  This integral has no simple closed form when A_c is singular.
  When A_c is invertible, the integral simplifies algebraically to:

    ∫₀^{T_d} exp(A_c · τ) dτ  =  A_c⁻¹ · (exp(A_c · T_d) − I)  =  A_c⁻¹ · (A_d − I)

  giving the convenient form:  B_d = A_c⁻¹ · (A_d − I) · B_c   [Tóth eq. 9a]

  Tóth footnote 2: *"To compute the resulting matrix functions of this discretization
  approach, Ac(p) is not required to be invertible, but if it is, we can write the
  resulting DT description of the state-evolution conveniently as (9a)."*

  The augmented matrix exponential (Van Loan 1978) computes the integral numerically
  without any inversion:

    exp([[A_c, B_c], [0, 0]] · T_d)  =  [[A_d, B_d], [0, I]]

  B_d drops out of the top-right block directly. No A_c⁻¹ anywhere.
  This is what scipy cont2discrete(method='zoh') uses internally.

**Why A_c is singular for our system**:
  The gantry A_c has block structure:

    A_c = [[  0,    I  ],
           [-M⁻¹K, -M⁻¹C]]

  The top-left 3×3 block is identically zero. The K matrix has zero rows for X and Y
  (rigid body modes — no spring restoring force in those directions), so det(K) = 0,
  which propagates to det(A_c) = 0. A_c⁻¹ does not exist.
  Note: B_c itself is not the problem — it is well-defined as [0; M⁻¹].
  The singularity is entirely in A_c, and only affects the shortcut for B_d.

**Complexity increase vs invertible case**:
  - Invertible A_c: compute A_d = expm(A_c · ts) [6×6], then B_d algebraically — two steps.
  - Singular A_c: must form 9×9 augmented matrix and compute one expm — A_d and B_d
    obtained together. Cannot be separated. Computationally more expensive but exact.

**Why**: The gantry A_c(Y) is singular — the top-left 3×3 block is all zeros (position states
  have no velocity-independent dynamics; rigid body modes give zero eigenvalues). `A_c⁻¹`
  does not exist, so the naive formula is undefined. The augmented exponential sidesteps the
  singularity and is mathematically identical to what scipy `cont2discrete(method='zoh')`
  does internally. Both scipy and the torch version must use this formula — any discrepancy
  between them is a numerical precision issue only.
**Ruled out**: `B_d = A_c⁻¹ · (A_d − I) · B_c` — undefined for singular A_c. `B_d = ts · B_c` (rectangular fallback) — O(ts) error, only valid as Option D fallback.
**Constrains**: Both `gantry_lpv_torch.py` and any future `LPV_Linear_State_Block` must form the 9×9 augmented matrix before calling `torch.linalg.matrix_exp`. See `docs/lpv-discretization.md` for the full derivation.

---

### [D-016] Step 2 validation is matrix comparison, not trajectory simulation
**Date**: 2026-03-17
**What**: Step 2 validation compares discrete A(Y), B(Y) matrices directly against MATLAB output at 5 operating points (Y = 0.1, 0.2, 0.3, 0.4, 0.5 m). It does not require simulating a full trajectory.
**Why**: A(Y), B(Y) already match MATLAB to 1e-19 at Y=0.3 (Task 1.2). The LPV question is whether the same holds at other Y values. If the matrices match at every Y, the physics is correct — no trajectory needed to confirm that. Trajectory simulation would add complexity (need input data, initial conditions, etc.) without providing additional information about the correctness of the physics parameterization.
**Ruled out**: Running a full closed-loop trajectory simulation at each Y — unnecessary for validating the LPV matrix computation. The trajectory simulation in Step 1 already validated the dynamics at Y=0.3.
**Constrains**: Requires a new MATLAB script `Matlab-scripts/export_lpv_matrices.m` (does not modify immutable files — calls existing functions) that evaluates G at each Y and saves A, B, C, D per operating point to `Matlab-output/lpv_matrices.mat`. Python comparison script `gantry_lpv_validate.py` checks max absolute error < 1e-10 per matrix per Y. Validation sweep: Y = linspace(0.05, 0.75, 50) — confirmed from ETEL Telica datasheet (total Y stroke = 800 mm, 5% margin from hard limits). 5 points is insufficient: M(Y)⁻¹ is rational in Y and could have non-monotone error behaviour between sparse samples. Dense 50-point sweep allows plotting error vs Y to confirm uniformity across the full operational range.

**Important distinction — what matrix comparison proves vs simulation comparison**:
Matrix comparison (Task 2.4) proves implementation correctness only: Python A(Y), B(Y) match
the same physics as MATLAB G(Y). It does NOT prove that the LPV simulation is a better baseline
than the frozen LTI. That requires Export 2 (Task 2.2) on a varying-Y trajectory.

**Correct simulation comparison target: q1, not q (Simscape).**
q1 (gantrySystem.m in Simulink) is a continuous-time quasi-LPV simulation — M(Y) is
re-evaluated each integration step as Y evolves. It uses identical physics to the LPV model
(same M(Y), C, K; no Coriolis, no Coulomb). Comparing LPV vs frozen LTI both against q1
isolates the Y-varying inertia effect cleanly, without Coriolis/Coulomb interference.
q (Simscape) is the secondary target: q1 vs q quantifies the augmentation gap (Coriolis +
Coulomb). The model is quasi-LPV: captures Y-dependent inertia only — Coriolis, centripetal,
and velocity-dependent friction are dropped and must be learned by the augmentation.

---

### [D-017] Both baseline and augmentation use LFR Δ(Y) structure
**Date**: 2026-03-19 (updated 2026-03-22)
**What**: Both the FP LPV baseline and the learned augmentation use the LFR Δ(Y) structure, as required by Drenth Ch. 5 eq. 5.1-5.2. The baseline has its own Δ^b(Y) block derived from the known physics (M(Y)^{-1}). The augmentation has a separate Δ^a(Y) block with trainable parameters. The two Δ blocks are block-diagonal (no cross-coupling in Δ), but the interconnection between baseline and augmentation happens through the combined M matrix (Drenth eq. 5.2, the `ab` and `ba` submatrices).
**Why**: Supervisor confirmed (2026-03-22) that the baseline should use LFR structure. Drenth Ch. 5 eq. 5.1 explicitly assumes the baseline is in LPV-LFR form. The baseline's Δ^b(Y) is fixed (derived from physics, not trained). The augmentation's Δ^a(Y) has trainable parameters. Well-posedness of the combined LFR is guaranteed by Drenth's direct parameterization (D_zw = exp(-N), Theorem 2.5).
**Open questions**:
- Whether parameter refinement of the FP baseline (making mb, mh, etc. trainable) changes the baseline's Δ^b structure during training. To be confirmed with supervisor at April 9 meeting.
- Whether the baseline implementation should live internally in logical coordinates or be similarity-transformed to stage coordinates before coding, given D-006.
- Whether the current latent-variable realization is accepted as the project baseline or treated as an intermediate realization pending a canonical/minimal LFT realization.
**Ruled out**: Original decision that the baseline does not need LFR (revised per supervisor guidance 2026-03-22).
**Constrains**: The baseline LFR realization must be derived from M(Y)^{-1}. A latent-variable realization now exists and is acceptable as a valid candidate baseline unless a stronger canonical/minimal realization requirement is imposed. This determines the baseline's Δ^b structure and the practical η (repetition count). The combined well-posedness (baseline + augmentation) must be ensured.

**Update 2026-03-22**: Major revision. Original decision said baseline does NOT need Δ(Y). Supervisor confirmed the opposite. Both baseline and augmentation now use LFR structure, per Drenth Ch. 5.

---

### [D-018] CT model kept in continuous time; RK4 used for integration at fixed step
**Date**: 2026-03-20
**What**: The gantry FP model is implemented and maintained as a continuous-time (CT) ODE. Simulation and augmentation training both integrate the CT equations using RK4 with a fixed time step equal to the sampling period (ts = 1/fs). The model is not pre-discretized before the integration step in the training loop.
**Why**: Supervisor confirmed in meeting (2026-03-20), quoting directly from notes: "write up the ct conversion. dont do discretization first will get messy." and "use rk4 not euler discretization. better to not precompute." Key reasoning:
  - RK4 with fixed step always takes the same dt, so it responds correctly to the sampling period and is compatible with the discrete control loop.
  - RK4 is a sum of 4 terms (4 evaluations with weighting), strictly more accurate than Euler (1st order) at the same step size.
  - ODE45 uses variable step sizes (cannot enforce a consistent sampling period by default). The ode4 variant forces a fixed step, but that is equivalent to RK4 directly.
  - ZOH pre-discretization is kept only for Steps 1-2 validation (already completed) where exact MATLAB matrix comparison was the goal. It is not used in the augmentation training loop.
  - When using system identification with a CT baseline, the same RK4 approach applies: keep the model in CT, apply RK4 alongside it.
  - ZOH (zero-order hold) holds the input constant within each interval but says nothing about how the ODE is integrated inside the interval. RK4 is the integration method used inside that interval.
**Ruled out**:
  - Euler discretization: O(h) truncation error, inferior accuracy for the same step size. Supervisor confirmed: "use rk4 not euler."
  - ODE45 with variable step: incompatible with a fixed sampling period in a discrete control loop. Acceptable only as the ode4 variant (fixed step), but RK4 achieves the same result directly.
  - Pre-discretizing with ZOH for the training loop: supervisor explicitly said not to pre-compute. Write up CT first, apply RK4 at runtime.
**Constrains**:
  - The CT model equations must be written up in full before integration is applied. This means: coordinate transforms, all physical quantities with dimensions and units, the full state-space ODE in logical and stage coordinates. This write-up is a prerequisite for Step 3.
  - A paper on discretizing LFRs must be found and reviewed (supervisor action item from 2026-03-20). The LFR structure also operates on the CT equations; understanding how LFRs are discretized informs the Step 3 implementation.
  - The torch training loop integrates the CT ODE using RK4 with dt=ts. The `LPV_Linear_State_Block` planned in D-011 is revised: instead of computing and storing A_d(Y), B_d(Y), it computes A_c(Y), B_c(Y) and applies one RK4 step.
  - The LFR structure for LPV augmentation (D-005, confirmed 2026-03-20) also builds on the CT formulation.
  - Rank of the M matrix should be computed across different trajectories to confirm no rank drop occurs across the operational range.

---

### [D-020] Two methods for rational LPV dependency; Method 2 (state-space form) chosen
**Date**: 2026-03-29 (resolved 2026-03-31, Roland Tóth meeting)
**What**: Two methods exist for handling the rational LPV dependency introduced by M(Y)⁻¹. Method 2 is chosen.

**Method 1 — Online resolve (what Roel implemented):**
Keep the full LFR structure live at runtime. G and Δ(Y) remain as separate blocks. During training, the backward pass propagates through the matrix inverse, implemented either by differentiating through the explicit inverse or via fixed-point iteration. Benefits: stays in true LFR form; LTI and parameter-varying blocks remain separated (useful for control design); potentially faster inference. Disadvantage: must deal with the rational symbolic form of M(Y)⁻¹ explicitly; more complex to implement.

**Method 2 — State-space form (chosen):**
Take M(Y)⁻¹ analytically and absorb it into Ac(Y), Bc(Y). Runtime evaluates `ẋ = Ac(Y)x + Bc(Y)u` directly via RK4. Rational dependency on Y is retained (do NOT rewrite to affine). LFR is used for derivation and structural analysis only, not as a live runtime loop. The augmentation block operates on the same collapsed signals; its black box component can remain affine.

**Why Method 2**: Roland confirmed in 2026-03-31 meeting that this is acceptable. The "algebraic loop" concern was a misapplication of the definition: M(Y) being invertible means the system is well-posed and no true algebraic loop exists. Need to stick to the original parameter structure of M(Y) (augmentation can be added on top without changing the baseline structure). Simpler to implement.
**Ruled out**: Method 1 for now. Not blocked, but not needed: the simpler SS form suffices and Method 1 can be revisited if control design or faster inference become priorities.
**Note — third option not pursued (delay)**: ASMPT mentioned a third approach: introduce a unit delay into the scheduling loop to break the algebraic dependency, rather than collapsing it analytically (Method 2) or resolving it online during training (Method 1). Not chosen because Method 2 is simpler and sufficient, but recorded here for completeness.
**Constrains**: Implement `CT_RK4_State_Block` using Ac(Y), Bc(Y) with rational-in-Y entries (from M(Y)⁻¹). Do not rewrite to affine. Verify M(Y) invertibility numerically: compute singular values of M(Y) across the full Y operational range and confirm they remain bounded away from zero. Check that maximum signal values in M(Y) are below 1 (or 1/0.75) to bound remaining concern.

---

### [D-021] Verify M(Y) invertibility numerically across the Y operational range
**Date**: 2026-03-31
**What**: Before relying on M(Y)⁻¹ in the runtime implementation, numerically verify that M(Y) remains invertible across the full operational Y range. Compute singular values of M(Y) for Y swept across [0, 0.7] m. Confirm all singular values stay bounded away from zero. Also check that maximum signal values in M(Y) are below 1 (or 1/0.75) to bound any remaining well-posedness concern.
**Why**: Roland noted this as a concrete verification step. Y range is also relevant for centering the scheduling variable: centering Y (e.g., Y_c = Y - Y_mean) improves numerical conditioning and avoids potential singularities near the boundary of the operational range.
**Ruled out**: Assuming invertibility without verification.
**Constrains**: This is a prerequisite check before implementing `CT_RK4_State_Block`. Script can be a short standalone MATLAB or Python check. Results should confirm M(Y) is positive definite (physical mass matrix) throughout the range.

---

### [D-022] Non-baseline physics go in augmentation, not in baseline
**Date**: 2026-03-31
**What**: Physical effects not present in the García-Herreros first-principles equations must not be added to the baseline model. They belong in the augmentation component and can be parametrized there.
**Why**: Confirmed by Roland in the 2026-03-31 meeting, specifically in response to the ASMPT-raised question about hysteresis. The concrete example: using sign(dY/dt) as an additional scheduling variable to capture hysteresis direction is a good idea, but it goes in the augmentation, not the baseline. Hysteresis is the motivating example that established this rule. The baseline must remain the exact FP model as derived. Adding extra physics to the baseline would conflate the known physics with the learned correction, making it harder to isolate what the augmentation is doing.
**Ruled out**: Extending the baseline state-space equations with additional physical terms (hysteresis, Coriolis, resonance, etc.).
**Constrains**: The baseline is frozen at the García-Herreros equations. Additional dynamics, forces, and scheduling variables (including sign(dY/dt) for hysteresis) are added in the augmentation block only.

---

### [D-023] Training roadmap: validate parameter estimation on synthetic MATLAB data before adding augmentation
**Date**: 2026-03-31
**What**: The training proceeds in two phases before full augmentation:
  1. Generate synthetic data from MATLAB for various Y values and parameter volumes.
  2. Train the baseline model with free (trainable) physical parameters only — no augmentation black box (Jan's parameter update method). Initialize parameters close to the true values. Show that the parameter estimation recovers the correct parameters from MATLAB-generated data.
  Only after this is demonstrated does augmentation (extra states, Coriolis, etc.) get added.
**Why**: Roland specified this phasing in the 2026-03-31 meeting. Validating the parameter update step in isolation (no black box) proves the baseline training pipeline works before adding augmentation complexity. This mirrors Jan's original method.
**Ruled out**: Jumping straight to augmentation training without first showing the baseline parameter estimation works on synthetic data.
**Constrains**: Synthetic data must cover a representative range of Y and other parameter volumes. The parameter initialization must be close enough to the true values for convergence. The "show it works" milestone (baseline parameters converge to MATLAB ground truth) is required before Step 4 (augmentation) begins.

---

### [D-024] Augmentation ordering: resonance first, Coriolis second
**Date**: 2026-03-31 (ASMPT meeting)
**What**: The augmentation is built up in two steps: first catch resonance dynamics, then add Coriolis as a second step. Coriolis is the more complex effect and should not be targeted before resonance is demonstrated to work.
**Why**: ASMPT guidance from the 2026-03-31 meeting. Resonance is the simpler and more immediate correction; Coriolis requires additional states and is a larger modelling step.
**Ruled out**: Adding Coriolis in the first augmentation step.
**Constrains**: The augmentation milestones in D-023 (training roadmap) follow this ordering.

---

### [D-025] Hysteresis: significant effect, sign(dY/dt) scheduling variable in augmentation
**Date**: 2026-03-31 (ASMPT meeting)
**What**: Hysteresis is a significant unmodelled effect in the gantry. The current scheduling structure (Y-only) cannot capture hysteresis direction because that requires the sign of velocity. Proposed approach: add sign(dY/dt) as an additional scheduling variable, or add a simple explicit hysteresis sub-model. Both approaches belong in the augmentation, not the baseline (see D-022). If hysteresis is not addressed at all, the network will absorb it through black-box fitting, which may reduce interpretability.
**Why**: Raised by ASMPT in the 2026-03-31 meeting. Confirmed by Roland as a good idea for the augmentation side.
**Open**: Whether to apply cost function weighting for hysteresis-dominated regions. Whether a dedicated simple hysteresis sub-model is better than the scheduling variable approach.
**Ruled out**: Adding hysteresis handling to the baseline model.
**Constrains**: When designing the augmentation scheduling structure, include sign(dY/dt) as a candidate scheduling variable. Revisit after resonance augmentation is validated (D-024 ordering).

---

### [D-026] Remove G from lfr_forward — replace G-matrix steps with direct physics expressions
**Date**: 2026-04-02

#### What was decided

The `G` argument is removed from `lfr_forward`. Steps 6 and 7 of the forward pass are replaced with direct physics expressions:

**Before (removed):**
```python
def lfr_forward(x, u, Y, G, M0, M1, M2, K, C):
    ...
    # Step 6: state derivative via G matrix  →  (batch, 6)
    xdot = x @ G.Ax.T + w @ G.Bw.T + u @ G.Bu.T

    # Step 7: output  →  (batch, 3)
    y = x @ G.Cy.T
```

**After (implemented):**
```python
def lfr_forward(x, u, Y, M0, M1, M2, K, C):
    ...
    # Step 6: state derivative — direct from physics (no G needed)
    xdot = torch.cat([x[:, 3:], v], dim=-1)   # (batch, 6)

    # Step 7: output — positions in logical coordinates
    y = x[:, :3]   # (batch, 3)
```

The `G` argument is also removed from `rk4_step` in `lfr_simulate.py`, from the `simulate` function signature, and from `LFRBaselineBlock` in `lfr_block.py` (the `self._G` attribute is removed; `rk4_step` no longer needs it).

---

#### Why this is a valid change — mathematical justification

**The physical state equations.** The gantry equation of motion in logical coordinates is:

```
M(Y) q̈ = -K q - C q̇ + u
```

The state is `x = [q; q̇] ∈ R⁶`, with `x[0:3] = q` (positions) and `x[3:6] = q̇` (velocities). The continuous-time state derivative is therefore:

```
ẋ = [q̇; q̈] = [x[3:6];  M(Y)⁻¹ fnet]       (equation 1)
```

where `fnet = -K x[0:3] - C x[3:6] + u` is the net generalized force. After **step 3** of `lfr_forward`, the quantity `v = M(Y)⁻¹ fnet` is already computed via `torch.linalg.solve(M_Y, fnet)`. Equation (1) then gives directly:

```
xdot = cat([x[:, 3:], v], dim=-1)            (equation 2)
```

This is always exactly correct, for any value of Y, because it is derived directly from the physical equations of motion.

**What G.Ax/Bw/Bu encode.** The LFR G-matrix representation expresses the same state equation as:

```
xdot = G.Ax @ x + G.Bw @ w + G.Bu @ u
```

where `w = [v₁; v₂] = [Y·v; Y²·v]` are the LFR latent signals (already computed in steps 4–5). The entries G.Ax, G.Bw, G.Bu are **constant matrices**, constructed by `build_G_matrix()` using `M₀⁻¹` (the mass matrix at a nominal point). The Y-dependence is captured through the latent signals `w`, not through G directly.

This G-matrix expression is algebraically identical to equation (2) — the LFR G matrices were derived precisely to encode the physical state equations in the LFR framework. The identity holds because the LFR structure is exact: the LFR is not a linearization or approximation; it is an exact rewriting of the rational-in-Y equations using the Δ(Y) = Y·I₆ block (verified in `lfr_forward.py` Check 2 against the collapsed form A_c(Y)@x + B_c(Y)@u).

**Why the G-matrix expression is inferior to the direct expression.** Even though the two forms are algebraically equivalent, the G-matrix form has a hidden dependency: it is only correct when G.Ax/Bw/Bu are consistent with the current values of M0/M1/M2/K/C. G is precomputed at import time in `lfr_matrices.py` by calling `build_G_matrix(M0, M1, M2, K, C)`. If M0/M1/M2/K/C are updated during parameter estimation, but G is not rebuilt, the G-matrix expression silently produces incorrect gradients and incorrect dynamics. The direct expression (equation 2) has no such dependency: it is always correct for whatever M0/M1/M2/K/C are passed to `lfr_forward` at that call.

**Why G.Cy = [I₃ | 0₃] is also removed.** The output `y = x @ G.Cy.T` selects the first 3 state components (logical positions). G.Cy is always `[I₃ | 0_{3×3}]` by the gantry output definition (output = position in logical coordinates). This is directly `x[:, :3]`. Unlike G.Ax/Bw/Bu, G.Cy would not become stale during parameter estimation (it does not depend on M0). However, replacing it with `x[:, :3]` is simpler, removes the G dependency entirely, and is more readable.

**Autograd implications.** The gradient path for physical parameters (M0, M1, M2) flows through `torch.linalg.solve(M_Y, fnet)` → `v` → `xdot`. This path exists in both the old and new implementation. The G-matrix form additionally has gradient paths through G.Ax/Bw/Bu entries when G is built dynamically from M0 inside the forward context. These extra paths disappear with the G removal. However, the physically correct gradient path (through the solve) is the one that was always present and is the one required for parameter estimation. The extra G-entry gradient paths in the old implementation were an artifact of redundant parameterization, not a feature.

---

#### What was ruled out

**Option A: Keep G in the signature but always rebuild it inside forward.**
`G = build_G_matrix(M0, M1, M2, K, C)` at each forward call, then use `G.Ax/Bw/Bu`. This adds unnecessary matrix computation at every forward step (linalg.solve inside build_G_matrix) and computes the same result as equation (2) through a much more expensive path. Rejected: unnecessary overhead, no benefit over the direct expression.

**Option B: Keep G and require the caller to always pass a freshly built G.**
Documented as a constraint ("caller must keep G consistent"). This is error-prone: the interface has two representations of the same physics, and nothing prevents them from diverging silently. Rejected: fragile by design, no benefit over the direct expression.

**Option C: Keep G only for documentation/clarity.**
G was never purely documentary — it participates directly in computation and autograd. Keeping a live computational dependency on G for readability reasons is not justified. Rejected.

---

#### What this constrains

- **lfr_forward signature** is now `(x, u, Y, M0, M1, M2, K, C)`. Any call site must be updated.
- **rk4_step and simulate** no longer accept or pass G. All call sites updated accordingly.
- **LFRBaselineBlock** does not store `self._G`. `build_G_matrix` is not called inside `forward()`.
- **G and build_G_matrix** remain in `lfr_matrices.py` — they are still useful for numerical analysis, LFR structure inspection, and offline verification. They are not deleted.
- **SVD-reduced forward pass** (`svd/lfr_svd_forward.py`) must NOT apply this shortcut. In the reduced realization the state and latent vectors are rotated by the SVD transformation matrices; the physical structure (positions first, velocities last) no longer holds, so `cat([x[:,3:], v])` is incorrect for the reduced system. The SVD-reduced forward must retain its G_reduced.Ax/Bw/Bu parameterization.
- **Check F in test_jan_compat.py** (trainable physical parameter gradient test) is simplified: only the solve-path gradient path exists. The distinction between "static G" and "dynamic G" is removed. The updated check verifies that M0.grad is non-None after backward — which is guaranteed by the linalg.solve gradient — and reports the gradient norm.

---

### [D-027] Fix y-output coordinate mismatch in the Interconnect connection matrix
**Date**: 2026-04-02
**What**: The `S_y` selection matrix in `build_baseline_interconnect` and `build_augmented_interconnect` (in `test_jan_compat.py`) was:
```python
S_y = selection_matrix(np.arange(3), 18)    # (3, 18) — selects logical positions
```
This routes `x_next[0:3]` (logical positions [X, Θ, Y]) directly as the Interconnect output `y`. The reference and training data use stage coordinates [X1, X2, Y]. The fix embeds the logical→stage transform into the connection matrix:
```python
S_y = P.numpy() @ selection_matrix(np.arange(3), 18)    # (3, 18) — logical → stage
```
In row-vector convention used throughout the Python code, `y_stage = y_logical @ P` (see `simulate()`: `Y_list.append(y_k @ P)`). For the Interconnect where the connection matrix acts as `y = S_y @ w_block` (column-vector convention), the correct transform is `S_y = P.numpy().T @ selection_matrix(np.arange(3), 18)`.

Wait — the Interconnect uses column-vector convention (w_block is (batch, nw, 1)), so `y = S_y @ w_block` computes (3, 18) @ (18, 1) = (3, 1). To obtain `y_stage = P.T @ y_logical` (column-vector form), `S_y = P.numpy().T @ selection_matrix(np.arange(3), 18)`.

**Why**: The MATLAB reference data (`q3`, simulation outputs) are in stage coordinates [X1, X2, Y]. The `lfr_forward` output `y = x[:, :3]` is in logical coordinates [X, Θ, Y]. The two coordinate systems differ in the X1/X2 vs X/Θ representation — they are related by `y_stage = P.T @ y_logical` (column-vector form). Without the P-transform in S_y, the Interconnect would output logical positions as training targets, causing incorrect loss computation when compared against stage-coordinate reference data.
**Ruled out**: Embedding the P-transform in `lfr_block.py` (adding y-routing logic to the block output, changing nw). The connection matrix is the correct place for coordinate transforms in Jan's framework — the block output format is fixed by the nw=18 contract.
**Constrains**: `build_baseline_interconnect` and `build_augmented_interconnect` in `test_jan_compat.py` apply this fix. Any future Interconnect wiring for the gantry baseline must use `P.numpy().T @ selection_matrix(np.arange(3), 18)` for the y connection matrix, not a plain selection matrix.

---

### [D-028] Add BPTT mode toggle to simulate()
**Date**: 2026-04-03
**What**: `simulate()` in `lfr_simulate.py` gains a `bptt_mode` parameter with three options: `"full"` (default, unchanged behaviour — retains entire graph), `"truncated"` (detach state every `segment_len` steps), and `"checkpoint"` (use `torch.utils.checkpoint` for exact gradients at O(sqrt(N)) memory). `simulate_frozen()` moved from `validate_lfr.py` to `lfr_simulate.py`.
**Why**: The full computation graph across N RK4 steps is O(N) in memory. For realistic training horizons (N > 1000), this becomes impractical. Jan's framework handles this implicitly via `nf`-bounded windows (typical nf=200), but our standalone `simulate()` had no such bound. The three modes give callers explicit control: `"truncated"` matches Jan's nf pattern (cheap, biased gradients); `"checkpoint"` gives exact gradients at ~1.3x compute; `"full"` remains the default for backward compatibility and short horizons.
**Ruled out**: Adjoint method (torchdiffeq) — exact O(1) memory but numerically unstable for stiff systems and adds an external dependency. Hardcoding a single BPTT strategy — different training scenarios benefit from different trade-offs.
**Constrains**: Training scripts should choose `bptt_mode` explicitly based on horizon length and gradient quality requirements. `segment_len` for truncated mode should cover the system's settling time (~200-1000 steps at 20 kHz).

### [D-029] LPV-LFR baseline code cleanup: performance and CUDA readiness
**Date**: 2026-04-05
**What**: Cleaned up the lpv_lfr_baseline package based on a line-by-line code review. Changes: (1) Pre-transform u_seq from stage to logical coords once before the simulate() loop instead of N times inside it. (2) Pre-allocate output tensors in simulate() and simulate_frozen() instead of list+stack. (3) Removed `_rk4_step_for_checkpoint` wrapper (identical to `rk4_step`; checkpoint calls `rk4_step` directly now). (4) Added `Y_override` parameter to `rk4_step` so `simulate_frozen` reuses the same RK4 logic instead of duplicating it. (5) Made lfr_block.py dtype cast conditional (skip when already float64). (6) Fixed CUDA device bug in simulate_frozen (`torch.full` was missing `device=x0.device`). (7) Pre-allocated tensors use `x0.new_empty()` to inherit device and dtype. (8) Trimmed module docstrings in lfr_forward.py and lfr_simulate.py. (9) Fixed test_jan_compat.py S_y construction to avoid unnecessary numpy round-trip.
**Why**: Preparing for GPU training. The original code had N redundant P.T matmuls per trajectory, N+1 tensor object allocations in Python lists, and a device bug that would crash on CUDA.
**Ruled out**: Deleting lfr_matrices.py (still used by svd/). Switching from torch.linalg.solve to Cholesky (negligible difference for 3x3 matrices).
**Constrains**: `rk4_step` now has an optional `Y_override` keyword argument. Callers using positional args are unaffected. `simulate_frozen` is now a thin wrapper around `simulate`-style logic with `Y_override`.

### [D-030] Trainable physical parameter set for ParameterizedLFRBlock
**Date**: 2026-04-06
**What**: 10 trainable scalars, 2 fixed scalars, in `ParameterizedLFRBlock`. Trainable: `kb_sum` (=kb1+kb2), `cg1`, `cg2`, `cy`, `cb_sum` (=cb1+cb2), `mh`, `m1`, `m2`, `mb`, `J_sum` (=Jb+Jh). Fixed buffers: `Lb`, `d`.
**Why**: Identifiability analysis on the matrix structure of M(Y), C, K:
- `kb1`, `kb2` appear only as their sum in K[1,1] → not individually identifiable; train sum.
- `cg1`, `cg2` appear as both sum and difference in C → individually identifiable.
- `cy` appears isolated in C[2,2] → directly identifiable.
- `cb1`, `cb2` appear only as sum in C[1,1] → train sum.
- `mh` is the sole LPV parameter (enters M0, M1, M2) → strongest signal, must train.
- `m1`, `m2` appear individually via M0[0,1]=(m1-m2)*Lb/2 → identifiable.
- `mb` appears only in M0[0,0] sum with m1+m2+mh → weakest signal; train with tight Lambda.
- `Jb`, `Jh` appear only as sum in M0[1,1] → train sum.
- `Lb` appears in M0, C, and the P coordinate transform; changing P corrupts stage↔logical mapping during training → fixed.
- `d` appears only in products mh*d and mh*d² alongside trainable mh → not separately identifiable; fixed.
All 10 trainable scalars are simultaneously trained from the start (same pattern as `Parameterized_MSD_State_Block`). Lambda regularization weights handle the varying identifiability — tighter for `mb` (2% detuning), standard for others (5–10% detuning).
**Ruled out**: Training `Lb` (corrupts P transform), training `d` (unidentifiable alongside mh), training `Jb`/`Jh` individually (only sum is identifiable), phased training (Jan trains all params at once; regularization handles weak identifiability).
**Constrains**: `_build_matrices()` in `lfr_param_block.py` must reconstruct M0, M1, M2, K, C from these 10 scalars plus fixed `Lb`, `d`. Detuning amounts: kb_sum −5%, cg1/cg2/cy/cb_sum −10%, mh/m1/m2/J_sum −5%, mb −2%.

---

### [D-031] Implement ParameterizedLFRBlock in a separate file lfr_param_block.py
**Date**: 2026-04-06
**What**: The trainable-parameter LFR block lives in `lpv_lfr_baseline/lfr_param_block.py`, not in `lfr_block.py`.
**Why**: `lfr_block.py` has a single well-tested responsibility (stateless frozen-parameter wrapper). The parameterized variant adds substantial new logic: scalar parameter management, `_build_matrices()` differentiable reconstruction, and `param_loss()` regularization. Mixing these two concerns would make both files harder to read and test independently. The existing module follows a one-concern-per-file pattern.
**Ruled out**: Extending `lfr_block.py` with a subclass (same file becomes bloated); creating a generic `parameterized_block.py` (too abstract for one use case).
**Constrains**: `lfr_block.py` stays untouched as the frozen baseline reference. `lfr_param_block.py` imports `rk4_step` from `lfr_simulate.py` and scalar constants from `physics.py` as initial values only.

---

### [D-032] Subclass SSE_Interconnect to handle ParameterizedLFRBlock.param_loss()
**Date**: 2026-04-06
**What**: A thin subclass of `SSE_Interconnect` (living in `lpv_lfr_baseline/`) overrides `loss()` to add a generic `hasattr(m, 'param_loss')` sweep over connected blocks. Jan's `model_augmentation/` code is not modified.
**Why**: `SSE_Interconnect.loss()` calls `param_loss()` only on hard-coded `isinstance` checks for its own block types. `model_augmentation/` is read-only (CLAUDE.md). A subclass override is the minimal, non-invasive extension.
**Ruled out**: Editing Jan's `interconnect.py` (violates read-only constraint); monkey-patching at runtime (fragile).
**Constrains**: The subclass must call `super().loss()` minus the block-type sweep, then add its own generic sweep — or replicate the loss structure with the generic check. It lives in `lpv_lfr_baseline/` and is the entry point for all training scripts in this project.

---

### [D-033] Data strategy: Option A (MATLAB) for first experiment, Option B (Python simulate) future
**Date**: 2026-04-06
**What**: The first parameter-recovery experiment uses the existing `Matlab-output/lpv_sim_varying_y.mat` as training data (Option A). Option B — generating fresh synthetic data via Python `simulate()` with a multisine input, controlled noise (SNR), and explicit train/val/test splits — is deferred to a future experiment.
**Why**: The MATLAB trajectory was generated with the true physical parameters and provides the ground-truth output we need to train against. It exercises varying Y (0.3→0.1 m), which is exactly the range where M(Y) variation is observable. Option B is more rigorous and mirrors Jan's experimental design exactly, but requires additional scripting (input design, noise model, data splits) that is not needed to prove the concept.
**Ruled out**: Using frozen-Y data (LPV parameter mh not identifiable without Y variation); skipping Option B entirely (it is the right long-term approach for a rigorous benchmark).
**Constrains**: The training script must load and convert `lpv_sim_varying_y.mat` to deepSI format. When Option B is implemented, the training script should be parameterizable to switch data sources without changing the model structure.

---

### [D-019] Use Drenth thesis for CT LPV-LFR citations; treat IFAC paper as DT companion
**Date**: 2026-03-24
**What**: For any continuous-time LPV-LFR definition, notation, or generic interconnection equations used in the gantry write-up, the primary source is Drenth's thesis (`literature/books/drenth2025_lpv-lfr-thesis.pdf`). The IFAC paper (`literature/lpv-lfr/drenth2025_lpv-lfr-rational.pdf`) is treated as the discrete-time companion paper and cited as such.
**Why**: The two local Drenth sources are not interchangeable. The thesis explicitly gives the LPV-LFR pair `(G, Delta(p))` in continuous time with `x_dot(t)`, `z(t)`, `w(t)`, `y(t)` and the equivalent rational LPV-SS form. The IFAC paper defines the LPV-LFR pair `{M, Delta(p)}` in discrete time. Citing the IFAC paper as if it were the primary CT definition overstates the DT-to-CT adaptation and obscures the notation difference between the two sources.
**Ruled out**: Treating the thesis and IFAC paper as equivalent sources for Section 2-style CT LPV-LFR definitions. Citing IFAC eq. 6-9 as if it were the primary CT source.
**Constrains**: `docs/references.md`, `docs/lfr-structure.md`, and future LaTeX source notes should cite the thesis for CT LPV-LFR definitions. The IFAC paper remains useful for DT LPV-LFR context, rational-dependency motivation, and well-posedness discussion, but should be labeled as the DT companion when referenced.

---

### [D-034] RMSE_baseline for Lambda regularization computed from detuned baseline on MATLAB data
**Date**: 2026-04-06 (updated 2026-04-20)
**What**: Before training begins, compute the per-trajectory RMSE of `ParameterizedLFRBlock` with `params = params_init` (detuned values) on the active MATLAB trajectories. Two quantities are derived from this:

1. `rmse_baseline` — group-balanced RMSE **in metres** (physical units). Used only for reporting and to instantiate the block when the loss is in physical units (not the current training setup).
2. `rmse_baseline_normalized` — the same RMSE expressed **in sigma-normalized units** (dimensionless), computed via `_aggregate_normalized_rmse_baseline()`. This is what is actually passed to `ParameterizedLFRBlock.__init__()` as `RMSE_baseline`.

The distinction matters because the training loss is normalized by sigma (see D-042):
```
mse_loss = mean(((Y_pred - q1) / sigma)²)    # dimensionless, O(1)
```
Lambda must be calibrated in the same unit system as `mse_loss`. Passing the metre-space value would make Lambda ~450× too small, effectively disabling regularization.

Inside the block, Lambda is computed as:
```python
Lambda[i] = RMSE_baseline_normalized / params_init[i]
```
This ensures the regularization cost is comparable to the simulation MSE when parameters have moved enough to reduce the (normalized) prediction error by one `RMSE_baseline_normalized` unit.

**Why**: RMSE_baseline_normalized scales the regularization relative to the simulation loss in the same unit system. Computing it from the actual detuned baseline on actual data gives principled, automatic calibration. Jan's fixed constant (0.2) is only valid because his data is already normalized to O(1) — our raw data is in metres and sigma-normalization must be applied first.
**Ruled out**: Passing `rmse_baseline` (metres) to the block — Lambda would be ~450× too small and regularization would be ineffective. Manual constant without sigma normalization — arbitrary and unit-dependent.
**Constrains**: `train_param_recovery.py` must compute both `rmse_baseline` (for logging) and `rmse_baseline_normalized` (for the block). The block always receives the sigma-normalized value. Both values should be logged in the saved `.pt` file for reproducibility. See D-042 for the sigma normalization itself.

---

### [D-036] OPEN — Augmentation training: state initialisation and mini-batch strategy
**Date**: 2026-04-08
**Status**: Deferred — decide when implementing augmentation training.
**What**: Two coupled design choices must be made when extending from parameter recovery to augmentation training:

**Choice A — State initialisation for segment start states:**

Option 1 (data-derived, current): positions from measured q1, velocities from central finite differences. Cached as `state_traj_n{N}.pt`. Works for parameter recovery because all states are observable (q, q̇ from positions). **Will not generalise to augmentation**: the augmentation block introduces latent states (e.g. hidden flexible modes) that cannot be read from measured positions or computed by finite differences.

Option 2 (encoder, Jan's approach — `model_augmentation/fit_systems/interconnect.py` line 417): `x = self.encoder(uhist, yhist)`. A learned neural network maps a window of past inputs and outputs to the full augmented state. The encoder is trained jointly with the physics parameters. This is the only correct approach when latent states exist.

**Recommendation**: Keep data-derived states for parameter recovery (current code). Switch to an encoder when augmentation is added. The encoder architecture Jan used is `modified_encoder_net` in `interconnect.py` — a `simple_res_net` mapping `[uhist, yhist]` → `x0`.

**Choice B — Segmentation strategy (overlapping vs non-overlapping):**

Current (parameter recovery): non-overlapping segments, stride = segment_len. Batch = n_seg = N // segment_len (e.g. 70). One gradient update per epoch = full-batch GD.

Jan's approach (augmentation): overlapping sliding windows, stride controlled by deepSI data loader (typically stride=1 or small). Many more gradient updates per epoch — effectively mini-batch SGD. More diverse gradient signal; helps generalisation and can escape local minima.

Trade-off: overlapping windows require the encoder to re-estimate state at every window start (batch × encoder forward pass per epoch). Non-overlapping is cheaper but less diverse. For noisy real data with a learned augmentation, mini-batch SGD over overlapping windows is the standard choice (confirmed by Jan's code).

**Recommendation**: For augmentation training, adopt Jan's overlapping strategy with encoder-based state init. The precomputed `state_traj` cache is still useful for the physical (observable) state components as a warm-start or validation reference.

**Ruled out at this stage**: None — decision deferred until augmentation implementation begins.
**Constrains**: Augmentation training script design. Encoder architecture and hyperparameters (nb, na window lengths) must be chosen at that time.

---

### [D-035] Physical parameter positivity enforced via log/exp reparameterization
**Date**: 2026-04-06
**What**: Physical scalars in `ParameterizedLFRBlock` are stored as `self.log_params = nn.Parameter(torch.log(params_init))`. Physical values are recovered as `params = torch.exp(self.log_params).clamp(min=1e-6)` inside `forward()` and `param_loss()`. The clamp is a numerical crash guard only, not an optimization mechanism.
**Why**: If any physical parameter goes zero or negative during training, `M(Y) = M0 + M1*Y + M2*Y²` becomes singular and `torch.linalg.solve` crashes or produces garbage. L2 regularization alone provides no hard guarantee. Log/exp reparameterization maps the unconstrained real line to `(0, ∞)` — the optimizer trains `log_params` freely in ℝ and positivity is guaranteed by construction. Literature survey (GPyTorch, Stan, neural ODE grey-box models, PINN parameter ID papers) confirms log/exp is the dominant choice for positive scalar physical parameters. Initialisation is trivial: `log(params_init)` exactly inverts the exp transform, so training starts at the correct physical values.
**Ruled out**:
- *Softplus*: `params = log(1 + exp(raw))`. Functionally equivalent to log/exp at our parameter magnitudes (all ≥ 1.05 kg) — softplus saturates to identity for large inputs so the two are numerically indistinguishable. Softplus is GPyTorch's default because it prevents overflow during large hyperparameter searches; this concern does not apply here since L2 regularization keeps params near init. Rejected in favour of log/exp for simplicity (no `softplus_inverse` needed at init) and because it is the more standard choice in the system identification literature.
- *Projected gradient / clamping as training strategy*: `params.clamp_(min=1e-6)` after each optimizer step. Creates a discontinuous gradient at the boundary — the optimizer sees a flat landscape and cannot recover. Parameters cluster at the clip value. Widely considered an antipattern (cf. WGAN weight clipping critique). Retained only as a numerical safety net after exp, not as a constraint mechanism.
- *Log-barrier term*: Add `-λ · Σ log(params)` to the loss. Requires scheduling λ toward 0 (interior point method) to be principled; in stochastic gradient training with Adam this scheduling is difficult to get right. Adds a hyperparameter with no clear benefit when L2 regularization already anchors parameters near positive initial values.
- *Unconstrained training relying on regularization alone*: L2 regularization provides a soft pull toward positive init values but no hard guarantee. For a small detuning (5-10%) and well-calibrated Lambda this would likely work in practice, but provides no protection against edge cases (aggressive learning rates, long training, poor RMSE_baseline calibration).

---

### [D-036] OPEN: LFR structure vs. state-space-only for LPV baseline and augmentation
**Date**: 2026-04-09 (raised in supervisor meeting, not yet decided)
**What**: Decide whether to express the LPV baseline as a true LFR (with M(Y) invertibility as a
rational/symbolic expression) or remain in state-space form (current: `torch.linalg.solve` at
every step).
**Why this matters**:
- Current `linalg.solve` approach is numerically correct but gives zero LFR structural benefit.
- LFR structure is almost essential for control design (H-inf, mu-synthesis) — a primary interest
  of ASMPT even when a black-box augmentation is added on top.
- Expressing M(Y)^{-1} symbolically as a rational function (MATLAB can do this) means no per-step
  matrix inversion; the forward pass becomes matrix-vector products only — computationally cheaper
  and structurally a proper LFR.
- Jan's interconnect framework supports state-space directly (no LFR required), but this trades
  away the control-design benefit.
**Open sub-questions**:
1. Does the parallel augmentation (D-003) still provide the orthogonality regularization benefit
   if the baseline is in state-space form rather than LFR? (I.e., what exactly is traded away?)
2. SVD on the LFR channels: reduces latent signals (good for control), but how does it affect
   interpretability of the learned augmentation states?
3. Identifiability / uniqueness of parameter updating: which parameter combinations only appear
   as sums in M(Y)? Can trajectory excitation separate them, or is norm regularization needed?
**Decision path**:
- If project scope includes control design deliverable → invest in symbolic M(Y)^{-1} (MATLAB)
  to recover LFR structure before augmentation.
- If scope is simulation/prediction only → state-space form is acceptable; note the limitation
  explicitly in the thesis.
**Ruled out**: Nothing ruled out yet — decision deferred pending scope clarification with supervisors.
**Constrains**: LPV model implementation (`lpv_lfr_baseline/`), augmentation interconnect structure,
and any control design work downstream.

---

### [D-037] OPEN: Norm regularization on cost function to improve parameter identifiability
**Date**: 2026-04-09 (raised in supervisor meeting, not yet decided)
**What**: When two or more physical parameters are only identifiable as a sum (e.g. M[i,j] = a + b
where only a+b enters M(Y)), add a norm term to the cost function to shape the landscape so that
individual components can be recovered.
**Why**: The standard RMSE + L2 loss may have a degenerate valley along directions where a+b is
constant — any (a, b) pair on that line gives the same loss. A norm term (e.g. L1 or L2 on the
individual values) breaks the degeneracy by preferring sparse or small-magnitude decompositions,
making the optimizer converge to a specific point rather than sliding along the valley.
**Connection to log-domain gradients (D-035)**:
- If Adam operates in log-domain (log/exp reparameterization), gradients near zero/small values
  are amplified — parameters at very different scales receive unfair updates.
- Roland's suggestion: **centre and normalize** log-parameters around ~1 before the gradient step
  to equalize the effective step sizes across parameters.
- Alternative to log: `p² = p * p` (always positive, smooth near zero gradient) or `|p * p|` —
  avoids log singularity near zero but still requires positivity guarantee.
**Open question**: What is the cleanest guarantee that a parameter never hits zero during training?
Current answer: log/exp (D-035). Revisit if instability seen near small parameter values.
**Ruled out**: Nothing ruled out yet.
**Constrains**: `train_param_recovery.py` loss function and optimizer configuration; see also
MEET-02 and MEET-06 in `tasks/todo.md`.
**Constrains**: `ParameterizedLFRBlock` stores `self.log_params` as `nn.Parameter`. All reads of physical parameter values — in `_build_matrices()`, `param_loss()`, and any diagnostic printout — must go through `torch.exp(self.log_params)`. The regularization reference `self.params_init` remains in physical space (not log space) for interpretability.

---

### [D-038] Simulation study extra state: Y-position-dependent Dahl friction states [z₁, z₂]
**Date**: 2026-04-10
**What**: The 8-state data-generating model for the augmentation simulation study adds two Dahl friction states [z₁, z₂] — bristle deflections on the X₁ and X₂ guides — to the 6-state LPV baseline. The baseline remains unmodified (6 states, constant C and K). The augmentation must discover the extra states and their coupling.

Data-generating model dynamics (extra states):
```
ż₁ = Ẋ₁ − (|Ẋ₁|/g) · z₁     where Ẋ₁ = Ẋ + (Lb/2)·Θ̇
ż₂ = Ẋ₂ − (|Ẋ₂|/g) · z₂     where Ẋ₂ = Ẋ − (Lb/2)·Θ̇

Y-dependent Coulomb amplitudes:
  Fc₁(Y) = Fc · (Lb/2 − Y) / Lb
  Fc₂(Y) = Fc · (Lb/2 + Y) / Lb

Modified force equations in data generator:
  F_X_friction = Fc₁(Y)·z₁ + cg1·Ẋ₁ + Fc₂(Y)·z₂ + cg2·Ẋ₂
  τ_Θ_friction = (Fc₁(Y)·z₁ − Fc₂(Y)·z₂) · Lb/2 + (cg1·Ẋ₁ − cg2·Ẋ₂) · Lb/2
```

**Why**: Five candidates were evaluated; the friction states were the only choice satisfying all criteria simultaneously:
1. Genuine dynamic states (own ODE, memory — not computable from current [X,Θ,Y,Ẋ,Θ̇,Ẏ])
2. Creates coupling: asymmetric Fc₁(Y) ≠ Fc₂(Y) when Y ≠ 0 generates Y-dependent torque on Θ from X motion
3. Position-dependent: coupling amplitude varies with Y, enriching the LPV structure (C(Y) alongside M(Y))
4. Direction-sensitive: z₁, z₂ carry history through direction reversals (pre-sliding transient)
5. Physically motivated: load distribution N₁(Y), N₂(Y) on X-guides changes with payload Y — documented in gantry literature
6. Directly connects to D-025 (supervisor's hysteresis observation) as the proper dynamic formulation of sign(Ẏ) scheduling
7. Exact Jan-analogy: extra states in data generator (absent from baseline), augmentation must rediscover them

**Ruled out**:
- *Support structure resonance [x_b, ẋ_b]*: Garcia's 37.7 Hz die-cast base resonance is specific to his rig; Telica uses granite/polymer-concrete frame with first resonance >100 Hz, above control bandwidth. No Y-dependence — does not enrich LPV structure.
- *Cross-arm bending mode [δ, δ̇]*: Garcia explicitly calls cross-arm vibration "negligible in comparison to the coupling between actuators." Building the simulation study on a phenomenon the original paper dismisses is a weak foundation.
- *Coriolis coupling (Ẏ·Θ̇ terms)*: Not a state — a static nonlinear function of existing states. A non-dynamic augmentation could capture it without extra states. Reserved for second augmentation step (D-024).
- *sign(Ẏ)*: Not a state — a static (memoryless) nonlinearity. Already approximately modelled as Coulomb friction in the baseline. The friction states [z₁, z₂] are the correct dynamic version that captures the hysteresis memory sign(Ẏ) approximates.

**Constrains**:
- Data generator implementation extends `rk4_step` / `lfr_simulate.py` to an 8-state variant; the 6-state baseline code is NOT modified.
- Augmentation interconnect uses `nxd=2` extra states (analogous to Jan's `nxd=2` for m₃ in MSD).
- Verification: true z₁(t), z₂(t) from the data generator are saved and compared against the augmentation's learned states.
- Key metric: Θ prediction error as a function of Y-position and motion direction.
- Parameter g (Dahl stiffness) and Fc (nominal Coulomb amplitude) must be chosen to produce a physically plausible but clearly observable effect — suggested range: g ≈ 1–5 μm (pre-sliding displacement), Fc ≈ 10–30 N.
- Cross-references: D-022 (extra states in augmentation, not baseline), D-023 (validate parameter recovery before augmentation), D-024 (friction study is the first augmentation demonstration), D-025 (friction states are the dynamic formulation of hysteresis scheduling).

---

### [D-039] Feedback controller operating point per trajectory: Y_initial
**Date**: 2026-04-17
**What**: In `export_lpv_multi_traj.m`, the feedback controller `Cfb` and frozen LTI `G`
are designed at `Y_op = sp.Y_initial` for each trajectory — the Y position at the start
of the main motion. This replaces the previous single frozen choice of `Y_op = 0.3` for
all trajectories.

| Trajectory | Y_initial | Cfb designed at |
|---|---|---|
| T1 | 0.3 | Y = 0.3 |
| T2 | 0.3 | Y = 0.3 |
| T3 | 0.0 | Y = 0.0 |
| T4 | 0.2 | Y = 0.2 |
| T5 | 0.2 | Y = 0.2 |
| T6 | 0.3 | Y = 0.3 |

**Why**: Designing at `Y_op = 0.3` for all trajectories is unnecessarily wrong for T3
(Y=0.0), T4 (Y=0.2), T5 (starts at Y=0.2). Using `Y_initial` gives each trajectory a
controller optimally matched to its operating condition without requiring any Simulink
changes — `Cfb` and `G` are still plain workspace variables.

**Ruled out**:
- *Single Y=0.3 for all*: unnecessarily off-design for T3/T4/T5.
- *Gain-scheduled LPV controller Cfb(Y)*: the correct solution for trajectories where
  Y varies during motion (T1, T5, T6). Requires replacing the fixed LTI `Cfb` block in
  Simulink with an online-scheduled controller (S-function or MATLAB function block).
  Not implemented because it requires modifying the Simulink model, which is out of
  scope for the current parameter recovery phase.

**Constrains**:
- For T1, T5, T6 where Y actively sweeps during the main motion, `Cfb` at `Y_initial`
  is still an approximation — the controller is off-design-point as Y moves. This is
  accepted for now; the recorded `(u_q1, q1)` pair remains a valid input-output dataset
  for parameter recovery regardless of controller quality, since both signals are saved
  exactly as simulated.
- If gain-scheduled control is added later, `Cfb` computation must move inside the
  trajectory loop and be evaluated online using the current Y state.

---

### [D-040] torch.compile on rk4_step deferred — hardware constraint
**Date**: 2026-04-18
**What**: `@torch.compile(fullgraph=True, dynamic=False)` was added to `rk4_step` as
Phase 2 of the Step 3c training speed optimization. It has been removed and deferred.
**Why**:
- Training GPU is a Quadro P2000 (CUDA Capability 6.1). Triton requires CC ≥ 7.0 (Volta+).
  `backend='inductor'` fails with: *"Found Quadro P2000 which is too old to be supported
  by the triton GPU compiler"*.
- CPU path also blocked: MSVC `cl.exe` is not installed on this Windows machine; TorchInductor
  cannot compile C++ kernels for the CPU fallback.
- `backend='aot_eager'` works on both but provides no kernel fusion — only Python dispatch
  overhead reduction, which is negligible on a GPU-bound workload.
**What WAS completed (kept)**: Phase 1 (GMatrix → (15,15) tensor refactor) is complete
and stays. It reduces the buffer count from 7 to 1 in `lfr_block.py`, simplifies the API,
and is the necessary prerequisite for Triton kernel fusion once hardware is upgraded.
**Ruled out**: `aot_eager` as a permanent solution — it provides ~0% speedup on CUDA.
**Re-enable when**: Training moves to a Volta/Turing/Ampere GPU (CC ≥ 7.0). The code
comment in `lfr_simulate.py` contains the exact decorator to uncomment.
**Known issue to fix on re-enable**: `rk4_step` is called in both gradient (training loop)
and no-grad (eval pass) contexts. With the default `cache_size_limit=8`, this triggers
`GLOBAL_STATE changed: grad_mode` recompilations that eventually raise `CacheLimitExceeded`.
Fix: use `options={"cache_size_limit": 4}` in the decorator — allows grad/no-grad × dtype
specializations without restructuring the call sites. No logic change needed.
**Constrains**: `lfr_simulate.py` — the commented-out decorator block must not be removed;
it documents the intended optimization for future hardware.

---

### [D-041] Physics computation kept in float64 — float32 not precise enough
**Date**: 2026-04-18
**What**: All physics in `rk4_step` and `lfr_forward` (the polynomial loop solve, RK4
integration, matrix products) is computed in float64. The Jan framework uses float32
throughout; explicit casts are applied at the block boundary in `lfr_block.py` and
`lfr_param_block.py` (float32 → float64 on entry, float64 → float32 on exit).
**Why**:
- The polynomial loop solve `N(Y)/d(Y)` uses Horner evaluation of the adjugate matrix
  (N0, N1, N2) and the scalar determinant polynomial d(Y). These involve subtraction
  of near-equal terms and division by a scalar that can be small near the limits of the
  Y operational range. float32 provides only ~7 decimal digits of precision — insufficient
  to guarantee numerical accuracy of the solve across the full Y range and over long
  trajectories (4000 RK4 steps per segment).
- RK4 integration accumulates truncation error per step; float32 rounding adds a second
  error source on top. Over 4000 steps at ts = 1/16 kHz the accumulated float32 error
  has not been validated against the required parameter recovery accuracy.
- Physical parameters (masses ~10–25 kg, stiffnesses ~2000 N/m) span two orders of
  magnitude. float32 relative error (~1e-7) translates to absolute errors that may not
  be negligible for gradient-based parameter recovery where small parameter deltas matter.
**Ruled out**: float32 physics — not validated, risk of gradient degradation during
parameter recovery training. The Quadro P2000 has 1/32 fp64-to-fp32 throughput ratio
(Pascal), so float32 would be significantly faster, but correctness must come first.
**Future investigation**: If training speed becomes a bottleneck after moving to better
hardware (or if float64 remains slow), run a controlled experiment:
1. Train with float64 (reference), record `param_table()` and val RMSE per epoch.
2. Remove the two cast lines in `lfr_block.py` to run entirely in float32.
3. Compare `param_table()` — if parameters agree to within ~0.1% and RMSE curves match,
   float32 is acceptable and the cast lines can be removed permanently.
The comment in `lfr_block.py` marks the exact two lines to change.
**Constrains**: `lfr_block.py` and `lfr_param_block.py` — the float32↔float64 cast lines
must not be removed without the above validation. `lfr_forward.py` and `lfr_simulate.py`
need no changes; they operate on whatever dtype the caller passes.

---

### [D-042] Training loss normalized by per-channel output standard deviation (sigma)
**Date**: 2026-04-20
**What**: The MSE training loss in `train_param_recovery.py` is computed in sigma-normalized space:
```python
sigma = std of q1 across all 6 TRAJ_SPECS trajectories, per channel  # (3,) float64 tensor
err   = (Y_pred - q1_seg) / sigma
mse_loss = err.pow(2).mean()                                           # dimensionless
```
`sigma` is computed over the **full trajectory set** (all TRAJ_SPECS, not just ACTIVE_TRAJ_IDS) and cached to disk. It does not change when the active trajectory subset is changed.

**Why**: The three output channels [X1, X2, Y] are in metres but have different signal amplitudes. Without normalization the Y channel (largest excursion) dominates the loss, pulling parameter gradients toward Y-related parameters (mh, cy) at the expense of X-related ones (m1, m2, cg1, cg2). Dividing by sigma gives each channel unit variance, so MSE contribution is proportional to relative prediction error, not absolute channel amplitude.

Using the full TRAJ_SPECS for sigma (not the active subset) means:
- Sigma is stable regardless of which trajectories are active — no cache invalidation when ACTIVE_TRAJ_IDS changes.
- Sigma represents the full operating envelope of the system, not just the subset being trained on.

**Connection to D-034 (RMSE_baseline_normalized)**: Because the loss is dimensionless, the RMSE_baseline passed to `ParameterizedLFRBlock` must also be in sigma-normalized units. `rmse_baseline_normalized` is computed by `_aggregate_normalized_rmse_baseline()`, which applies the same per-channel sigma division to the per-trajectory RMSE before aggregating. This is the value passed to the block — not the metre-space `rmse_baseline`. See D-034 for the full Lambda calibration rationale.

**Ruled out**:
- *No normalization*: Y channel dominates; X1/X2 parameter gradients are suppressed.
- *Global scalar normalization*: a single scalar (e.g. overall std) does not correct the per-channel imbalance.
- *Normalizing by active-subset sigma*: sigma would shift when ACTIVE_TRAJ_IDS changes, making Lambda (which is fixed at block construction) inconsistent across runs.

**Constrains**: `train_param_recovery.py` — `sigma` must always be computed from the full TRAJ_SPECS, not the active subset. The `SIGMA_CACHE_VERSION` constant must be incremented if TRAJ_SPECS itself changes. Any future training script for this system must apply the same sigma normalization and pass `rmse_baseline_normalized` (not metres) to the block.

---

### [D-043] Checkpoint/epoch selection strategy for parameter recovery training
**Date**: 2026-04-20
**What**: Three decisions about which parameter vector to save and how to track convergence:

1. **Current phase (clean MATLAB data):** Use **Polyak-Ruppert tail averaging** over the plateau phase. Start averaging on the first LR reduction event from `ReduceLROnPlateau` — this trigger is automatic and requires no additional hyperparameter. The averaged `log_params` are saved alongside the final-epoch `log_params` in the `.pt` file. This is not yet implemented.

2. **Convergence tracking:** Run a full-trajectory eval (same as step 5) every `PARAM_LOG_INTERVAL` epochs. Save the result in `history`. This gives a clean convergence curve comparable to the final step 5 result, and provides the signal for best-epoch tracking if needed. This is not yet implemented.

3. **Future phase (measurement noise):** Polyak averaging over the late plateau becomes harmful — the late iterates are corrupted by semi-convergence (the optimizer fits noise after exhausting the clean signal). Switch to early stopping:
   - Known noise variance → **Morozov Discrepancy Principle**: halt when the smoothed training residual hits the noise floor `τ·δ²`.
   - Unknown noise variance → **L-curve method**: log `(residual norm, solution norm)` at each epoch; find the corner post-training. This requires logging `‖log_params‖` (or deviation from init) alongside the loss in `history`. The `log_params_snapshot` already saved at `PARAM_LOG_INTERVAL` supports this.

**Why:**
- **Saving last epoch is not principled.** The last epoch may not be optimal: the stochastic 8-segment train loss has high variance, and `ReduceLROnPlateau` does not guarantee the last iterate is the best. Last ≈ best only if LR has fully decayed to `min_lr` — which may not happen within 2000 epochs.
- **Best-epoch on stochastic train loss is actively wrong.** It rewards lucky random batches, not genuine parameter improvement. Confirmed by both the subagent research and Gemini Deep Research.
- **Polyak tail averaging is theoretically optimal for clean data.** For a 13-parameter, physics-constrained, locally convex problem, iterate averaging achieves the Cramér-Rao lower bound. It cancels the zero-mean batch noise algebraically without any additional computation beyond a running sum of 13 scalars.
- **Full-trajectory eval every PARAM_LOG_INTERVAL solves two problems at once:** the convergence plot becomes directly comparable to the step 5 final result, and it provides a stable signal for best-epoch tracking that is immune to batch sampling noise.
- **Semi-convergence is a real risk when noise is added.** With only 13 parameters, structural overfitting cannot occur. But the optimizer will eventually start fitting measurement noise rather than physics — "clean priority learning" means accuracy peaks mid-training, not at the end. Polyak averaging the corrupted plateau would amplify this effect.

**Ruled out:**
- *Best-epoch on stochastic train loss:* rewards sampling variance; statistically invalid for epoch selection.
- *Best-epoch on fixed held-out segment set:* computationally wasteful per epoch; vulnerable to trajectory divergence and the same noise issue as the train set (just with a fixed random seed instead of a varying one). Less principled than full-trajectory eval.
- *Schedule-Free optimizer (Defazio 2024):* eliminates epoch selection entirely by unifying momentum and iterate averaging — promising but not implemented. Would remove `ReduceLROnPlateau` and its associated patience/factor hyperparameters. Deferred as a future experiment.
- *Stochastic Weight Averaging (SWA) with cyclical LR:* correct in principle but requires replacing `ReduceLROnPlateau` with a cyclical schedule. More disruptive to the current setup than Polyak tail averaging which re-uses the existing scheduler trigger.

**Constrains:**
- `train_param_recovery.py`: add `averaging_active` flag, `AveragedModel` from `torch.optim.swa_utils`, triggered by first LR reduction. Save `averaged_log_params` in the `.pt` file.
- `train_param_recovery.py`: add full-trajectory eval loop inside the `PARAM_LOG_INTERVAL` block. Save per-trajectory RMSE snapshots in `history`.
- When noise is added: `history` must log solution norm `‖log_params − log(params_init)‖` per epoch to support L-curve analysis post-training. The `log_params_snapshot` at `PARAM_LOG_INTERVAL` already provides this at coarser resolution.
- Both `params_learned` (last epoch) and `params_learned_avg` (Polyak average) must appear in the final `.pt` save so results can be compared.

---

### [D-044] Multi-trajectory loss function: binary masking + per-trajectory per-channel sigma
**Date**: 2026-04-21
**What**: Replace the current global-sigma unweighted MSE loss with a loss that applies
binary channel masks per trajectory group, normalizes by per-trajectory per-channel signal
std, and averages per segment before averaging over the batch.

**The six problems with the current implementation (global sigma, no masking):**

1. **Dormant channels included in the loss.** On T1/T6 (Y-only), X1 and X2 are actively
   suppressed by the feedback controller but contribute equally to the MSE. The optimizer
   receives gradient signal from controller suppression dynamics rather than plant physics,
   pulling physical parameters away from their true values.

2. **Global sigma dilutes Y, inflates X.** sigma[Y] is computed from all 6 trajectories
   including T2/T3/T4 where Y is constant → sigma[Y] is artificially small → Y is
   over-weighted. sigma[X1] is computed across all 6 trajectories including T1/T6 where
   X1 ≈ 0 → sigma[X1] is artificially large → X1 is under-weighted on trajectories where
   it is actually active. Both biases compound simultaneously.

3. **Within-trajectory amplitude imbalance.** On T5 (X + Y both active), if Y sweeps much
   more than X1/X2, Y dominates the loss. Parameters primarily identified by X motion
   (m1, m2, cg1, cg2) are undertrained relative to Y-related parameters (mh, cy).

4. **Cross-trajectory amplitude imbalance.** Trajectories with the same active channels
   can have very different amplitudes (T1 conservative vs T6 aggressive Y sweep). A single
   global sigma[Y] does not capture this: T6 segments always dominate T1 segments in the
   loss, even though both are Y-only trajectories contributing equal information about Y.

5. **Denominator is inconsistent across segments.** Different segments have different numbers
   of active channels (T1: 1 active, T2/T3/T4: 2 active, T5: 3 active). A fixed global
   denominator gives unequal weight per active channel-step across trajectory groups. No
   single global denominator is correct for all segments simultaneously.

6. **Adam sees inconsistent loss scale across batches.** With 8 segments sampled from
   different trajectory groups per batch, the loss magnitude depends on which groups appear.
   Without per-segment normalization, Adam's second moment estimate v_t cannot stabilize,
   making its adaptive learning rate unreliable.

**Why**: Problems 1–6 compound. Problems 1 and 2 corrupt the gradient direction. Problems
3 and 4 create systematic undertraining of specific parameter subsets. Problems 5 and 6
make Adam's adaptation unreliable across epochs. The combination means the optimizer is
simultaneously given wrong gradient directions AND wrong step sizes.

**Chosen solution:**
```
For each segment in the batch:
  1. Binary mask:  zero out dormant channels for this trajectory group
  2. Normalize:    divide residual by sigma[traj_id][channel]
                   (sigma computed from that trajectory individually, active channel only)
  3. Per-segment loss = masked_normalized_err².mean() over (active_channels × T)
Average segment losses over the batch.
```

Formally:
```
loss = (1/B) Σ_i [ (1 / (n_active_i · T)) Σ_c Σ_t  m_{g,c} · ((ŷ_c - y_c) / σ_{traj,c})² ]
```

where m_{g,c} ∈ {0,1} is the binary mask for channel c in trajectory group g,
and σ_{traj,c} is the std of channel c computed from that trajectory only.

**Why per-trajectory sigma solves problems 3 and 4:** Each trajectory's sigma reflects
its own excitation amplitude. T6's sigma[Y] ≈ 300 mm; T1's sigma[Y] ≈ 50 mm. After
normalization, a 30 mm residual on T6 contributes (30/300)² = 0.01 — equal to a 5 mm
residual on T1 contributing (5/50)² = 0.01. Equal relative contribution regardless of
absolute excitation amplitude.

**Why per-segment averaging solves problems 5 and 6:** Each segment contributes O(1) to
the loss regardless of how many active channels it has. Adam sees a consistent loss
magnitude across all batches regardless of trajectory group composition. The second
moment estimate v_t stabilizes correctly.

**Forward compatibility (future hardware data):** When moving to real measurements with
additive noise, per-trajectory sigma transitions directly to the principled Λ⁻¹ weighting
(Ljung 1999 §7.4, Gautier, Janot & Vandanjon 2013). At high SNR (gantry encoders:
signal mm–cm, noise µm), signal std ≈ noise-floor-independent scale → per-trajectory
sigma is the high-SNR approximation of Λ⁻¹ weighting. No architectural change required
at the transition to hardware data; only the interpretation of sigma changes.

**Literature support:**

*Problem 1 — Dormant channel masking in gradient-based SysID (verified by direct quote):*
- **Werling et al., "Trajectory-based actuator identification via differentiable
  simulation"** (PDF p. 5, Eq. 2 and p. 12, Appendix B): loss `L = (1/MN) Σ ‖W(s'−s)‖²`
  with `W = diag(w_q, w_qdot)`; set to `diag(1, 0)` so velocity remains in the rollout
  but *"velocity residuals are not penalized because the measured velocity signal is
  noticeably noisier than position."* Directly confirms: mask in the loss, keep in the
  dynamics. Optimizer: Adam (Appendix B).
- **Gautier & Khalil (1990)** — dormant joints produce structural zeros in the regressor
  (classical least-squares analog). Forssell & Ljung (1999) additionally applies when
  measurement noise is present (closed-loop bias-pull mechanism).

*Problems 2 & 3 — Amplitude normalization across channels in gradient-based SysID (verified):*
- **Lutter et al., "Dynamic Modeling of Robotic Manipulator via an Augmented Deep
  Lagrangian Network"** (PDF p. 4, Eq. 8): Mahalanobis norm with diagonal covariance
  matrix W_τ; explicit justification: *"It is necessary to normalize the loss function
  using covariance matrix since the torque magnitude may vary greatly from joint to joint."*
- **Lutter et al., "Combining Physics and Deep Learning to learn Continuous-Time Dynamics
  Models" (Deep Lagrangian Networks, IJRR)** (PDF p. 7, Eq. 12): same Mahalanobis norm
  with diagonal W_τ; *"It is beneficial to normalize the loss using the covariance matrix
  because magnitude of the residual might vary between different joints."*
- **"Constrained Gray-Box Identification of Electromechanical Systems Under Unfiltered
  Step-Response Data"** (PDF pp. 6–7, Eq. 3): normalized composite residual dividing
  trajectory errors by `RMS(signal)` per channel; *"naturally balances the relative
  contribution of current and velocity; thus α_ω = α_i = 1 is sufficient and avoids
  additional manual scaling."*

*Problems 5 & 6 — Segmented minibatch objective for Adam consistency (verified):*
- **Werling et al. (above)**, Eq. 2: loss averaged over M segments and N timesteps as
  `(1/MN) Σ_j Σ_i ‖W(s'_{i,j} − s_{i,j})‖²` — each segment normalized independently
  before batch average. Adam confirmed as optimizer (Appendix B).

*Problem 4 — Cross-trajectory amplitude imbalance:*
- **No exact citable method found** that matches all of: multiple trajectories + same
  active channels + different amplitudes + joint gradient-based physical parameter ID +
  trajectory-specific normalization in the training loss.
- **Citable principle — experiment-balanced weighting:** adjacent inverse-identification
  literature explicitly supports the broader principle that multiple experiments should
  contribute in a balanced or uncertainty-weighted way to the cost function, rather than
  in proportion to raw residual magnitude:
  - **Zhang et al., Int. J. Solids Struct. (2023), doi:10.1016/j.ijsolstr.2023.112534**:
    explicitly states that good inverse-identification results depend on *"maintaining
    equal contribution of the strain states from each experiment to the cost function"*
    — the clearest paper-level support for equal cross-experiment contribution.
  - **Neggers et al., Mech. Mater. (2019), doi:10.1016/j.mechmat.2019.03.001**:
    when combining multiple experiments and data sources, weighting should follow
    measurement uncertainty derived from a Bayesian formulation — citable basis for
    experiment-wise balancing rather than raw aggregation.
- **Framing for thesis:** per-trajectory sigma normalization is an engineering
  realization of experiment-balanced weighting — supported in adjacent inverse-
  identification literature as a principle, but not a canonical standard method in
  robot gradient-based SysID. It is not "uncited" but it is also not "established."

*Supporting context — gradient-based physical SysID as established paradigm (verified):*
- **Muratore et al., "Differentiable Simulation for Physical System Identification"
  (RA-L 2021)** (PDF p. 6, Sec. IV-B): friction and mass estimated by backpropagating
  MSE loss through differentiable simulator via PyTorch AD; Adam optimizer.
- **Saveriano et al., "Physics-informed online learning of gray-box models by moving
  horizon estimation" (EJC 2023, 100861)** (PDF pp. 3–4): physical submodel + neural
  network trained via BPTT; arrival cost covariance *"can be seen as an adaptive
  learning-rate."*
- **Ljung (1999) §7.4 eq. (7.27)** — Λ⁻¹ weighting of multi-output prediction errors
  (classical PEM; per-trajectory sigma is the high-SNR approximation of this).
- **Gautier, Janot & Vandanjon (2013), IEEE TCST** — per-joint inverse-std normalization
  *"normalises the errors"* in closed-loop robot ID (regressor analog).

**Ruled out:**
- *Global sigma (D-042):* contaminated by inactive-channel samples for every channel
  (Problems 1–4). Documented as the identified flaw in D-042.
- *Per-channel-global sigma (no per-trajectory split):* solves Problems 1–2 partially
  but not Problems 3–4. T6 still dominates T1 after normalization.
- *Per-segment sigma (normalize each segment by its own std):* independently normalizes
  each segment but breaks Adam — momentum estimates are built from segments with
  incompatible normalization bases, corrupting gradient direction across batches.
- *GradNorm (Chen et al. 2018):* correct in principle but requires computing ‖∂L_i/∂θ‖
  through the RK4 graph at every step — expensive and unverified on physical grey-box
  sensitivity Jacobians.

**Constrains:**
- `train_param_recovery.py`: precompute `sigma[traj_id][channel]` from each trajectory's
  active samples before training. Pass trajectory ID with each segment in the batch.
- Loss function must use per-segment averaging (Option B), not global averaging (Option A).
- When hardware data is available: replace sigma computation with noise std estimated from
  static measurements; loss architecture unchanged.
