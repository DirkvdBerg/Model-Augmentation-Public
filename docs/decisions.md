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
**Date**: 2026-04-06
**What**: Before training begins, run one no-gradient forward simulation of `ParameterizedLFRBlock` with `params = params_init` (detuned values) on `lpv_sim_varying_y.mat`. Measure the RMS prediction error in stage coordinates. Use this value as `RMSE_baseline` in the Lambda computation: `Lambda[i] = RMSE_baseline / params_init[i]`.
**Why**: RMSE_baseline scales the regularization relative to the simulation loss. If it is set correctly, the optimizer naturally balances simulation MSE against parameter deviation — when parameters have moved enough to reduce prediction error by one RMSE_baseline unit, the regularization cost is comparable to the simulation benefit. Computing it from the actual detuned baseline on actual data gives an automatic, principled calibration that does not require guessing. Our data is in physical units (metres), so a pre-chosen constant (Jan's value of 0.2 for his normalised MSD system) would be arbitrary and likely wrong.
**Ruled out**: Manual constant (Jan's approach, e.g. 0.2 for MSD) — only valid because Jan pre-normalises his data to dimensionless units; our data is in metres and the appropriate scale is unknown a priori without a simulation. Setting it too small over-regularises (parameters cannot move); too large under-regularises (parameters may overshoot true values).
**Constrains**: The training script must run a forward-pass RMSE computation before calling `init_model()`. This value is passed to `ParameterizedLFRBlock.__init__()` as `RMSE_baseline`. It should be logged alongside training results for reproducibility.

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
**Constrains**: `ParameterizedLFRBlock` stores `self.log_params` as `nn.Parameter`. All reads of physical parameter values — in `_build_matrices()`, `param_loss()`, and any diagnostic printout — must go through `torch.exp(self.log_params)`. The regularization reference `self.params_init` remains in physical space (not log space) for interpretability.
