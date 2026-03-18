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

### [D-005] LFR-LPV augmentation adaptation is deferred
**Date**: 2026-03-16
**What**: Extending the augmentation framework to the full LPV-LFR setting is deferred to a later phase.
**Why**: The immediate priority is getting the FP model into the correct discrete-time state-space form compatible with the existing augmentation code. LPV-LFR augmentation adds complexity that should not be introduced before the baseline integration is validated.
**Ruled out**: Attempting LPV-LFR augmentation before the FP model baseline is working.
**Constrains**: Current work focuses on FP model conversion and compatibility with the existing LFR interconnect. LPV scheduling in the augmentation layer comes after.

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
**Date**: 2026-03-17
**What**: Wiring the LPV baseline into the augmentation interconnect requires a new block — `LPV_Linear_State_Block` — that reads Y from the current state at each forward call and recomputes A(Y), B(Y) via `gantry_discrete_ss(Y)`.
**Why**: The existing `Linear_State_Block` stores A and B as fixed attributes set at init — it cannot update them per step. The LPV baseline needs matrices that change every timestep as Y evolves. No existing block in the framework supports this.
**Ruled out**: Reusing `Linear_State_Block` with a single frozen operating point — that is the frozen LTI, not the LPV baseline. Reusing `Parameterized_LPV_Affine_Linear_State_Block` — wrong structure (affine-in-Y², trainable, augmentation-side).
**Constrains**: Implementation of `LPV_Linear_State_Block` is a Step 3 task, blocked on Step 2 validation. The block must expose Y as a self-scheduled variable (read from state index 2 in stage coordinates) and call `gantry_discrete_ss(Y)` internally.

---

### [D-012] LPV discretization: frozen ZOH for validation, exact ZOH via matrix_exp for training
**Date**: 2026-03-17 (updated 2026-03-17 after torch.linalg.matrix_exp fact-check)
**What**: Two discretization approaches are used, one per use case:
  1. **Validation (Step 2)**: frozen-at-sampling-instant — call `cont2discrete(A_c(Y[k]), ts)` at each step. Theoretically exact within ZOH (zero local truncation error, Tóth Section III-B).
  2. **Training loop (Step 3)**: exact ZOH via `torch.linalg.matrix_exp(A_c(Y) * ts)`. Zero local truncation error, fully torch-differentiable (confirmed by test).
**Why**:
  - Validation: `cont2discrete` is exact and fast enough for a one-off simulation. At 16 kHz with ΔY ≤ 0.125 mm/sample the within-sample variation of A(Y) is negligible — confirmed by Tóth (2010).
  - Training: `torch.linalg.matrix_exp` is a native PyTorch op — autograd traces through it, gradients flow back to Y[k]. This gives the same zero truncation error as scipy `cont2discrete` without leaving the autograd graph. The rectangular approximation (Option D, O(ts) error) is a valid fallback but is strictly inferior — there is no reason to accept approximation error when exact ZOH is differentiable.
**Ruled out**:
  - Polynomial expansion (Option A): A_c(Y) is rational (from M(Y)⁻¹), so no exact polynomial A_d(Y) exists.
  - Linear-affine approximation (Option B): drops dominant Y² term in M[1,1].
  - Grid interpolation (Option C): not natively torch-differentiable.
  - Rectangular approximation (Option D): O(ts) error — valid fallback only. Superseded by Option E.
  - scipy `cont2discrete` in training loop: not inside autograd graph.
**Constrains**: `LPV_Linear_State_Block.forward()` must compute A_c(Y) analytically from M(Y)⁻¹ using tensor ops, then apply `torch.linalg.matrix_exp(A_c(Y) * ts)`. See `docs/lpv-discretization.md` for full rationale and option comparison table.

---

### [D-013] LPV baseline uses Architecture 1 (direct forward) — Drenth confirms
**Date**: 2026-03-17
**What**: `LPV_Linear_State_Block` computes `A_d(Y), B_d(Y)` directly inside `forward(z)` (Architecture 1). The formal LFR Δ(p) structure is NOT required for the baseline. `SSE_Interconnect` and all existing wiring machinery are used unchanged.
**Why**: Drenth's thesis (Chapter 2, eq. 2.29) confirms that the forward simulation loop collapses to direct A(p)x + B(p)u at each step, even when the model is parameterized as a Δ(p)-LFR internally. The Δ(p) structure applies to the learned augmentation, not the physics baseline. Self-scheduled quasi-LPV (Y from state) is explicitly supported — Y is read from the state inside `forward()`, not routed through S. Confirmed by assess-paper assessment of Drenth (2025) + Hoekstra et al. (2025).
**Ruled out**: Architecture 2 (formal Δ(p) scheduling block with separate wiring through S) — not required for the physics baseline. New `SSE_Interconnect` subclass — existing class is sufficient.
**Constrains**: `LPV_Linear_State_Block` wires into `SSE_Interconnect` identically to `Linear_State_Block`. One open question remains: how to apply normalization (Tx/Tx⁻¹) when A_d(Y) is computed at runtime via `matrix_exp` rather than stored as a constant matrix. Must be resolved before Step 3 implementation. See `docs/lpv-lfr-interconnect.md` for full assessment.

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
**Why**: The gantry A_c(Y) is singular — the top-left 3×3 block is all zeros (position states have no velocity-independent dynamics; rigid body modes give zero eigenvalues). `A_c⁻¹` does not exist, so the naive formula is undefined. The augmented exponential sidesteps the singularity and is mathematically identical to what scipy `cont2discrete(method='zoh')` does internally. Both scipy and the torch version must use this formula — any discrepancy between them is a numerical precision issue only.
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
