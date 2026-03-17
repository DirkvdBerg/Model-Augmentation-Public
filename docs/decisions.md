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
