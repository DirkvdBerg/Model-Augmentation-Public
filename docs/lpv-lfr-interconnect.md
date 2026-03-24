# LPV-LFR Interconnect — Specification and Open Questions

This document defines the use case and specific questions that Drenth's thesis must answer
before Step 3 (wiring the LPV baseline into the augmentation interconnect) is implemented.

**Source clarification update (2026-03-24)**:
- `literature/books/drenth2025_lpv-lfr-thesis.pdf` is the primary **continuous-time** LPV-LFR source. It defines the pair `(G, Delta(p))` with `x_dot(t)`, `z(t)`, `w(t)`, `y(t)`.
- `literature/lpv-lfr/drenth2025_lpv-lfr-rational.pdf` is the **discrete-time** companion IFAC paper. It defines `{M, Delta(p)}` in DT.
- For any CT LPV-LFR definition used in the gantry derivation, cite the thesis first and treat the IFAC paper as supporting DT context.

---

## Context — what we have now (LTI baseline)

Jan's augmentation framework uses a fixed LFR interconnect. For the LTI case, the baseline
provides constant A, B, C, D matrices. Two blocks wire into `SSE_Interconnect`:

```
Linear_State_Block(A, B)    →  x[k+1] = A @ x[k] + B @ u[k]
Linear_Output_Block(C, D)   →  y[k]   = C @ x[k] + D @ u[k]
```

The augmentation (data-driven correction) adds additively to x[k+1] and y[k] in parallel.
Signals flow through a pre-computed interconnection matrix S (LFR structure). Algebraic loops
are checked at init. Normalization is applied to A, B, C, D before block construction.

This is validated and working for the gantry frozen LTI (Step 1 ✅).

---

## What we need to add — LPV baseline

The gantry FP model has Y-dependent matrices:

```
x[k+1] = A_d(Y[k]) @ x[k] + B_d(Y[k]) @ u[k]
y[k]   = C @ x[k]
```

where Y[k] = x[k, 2] (payload Y-position, state index 2 in stage coordinates).
A_d(Y) and B_d(Y) change every timestep as Y evolves. C and D are constant.

Our planned block (`LPV_Linear_State_Block`, D-011) reads Y from the state at each forward
call and recomputes A_d(Y), B_d(Y). This is the *direct* approach — it fits naturally into
the existing `Block` interface since `forward(z)` already receives the current state.

**The critical unknown**: Is this compatible with how Drenth represents LPV-LFR structure,
or does his framework require a fundamentally different interconnect architecture?

---

## Two possible architectures

### Architecture 1 — Direct (our current plan)

```
LPV_Linear_State_Block.forward(z):
    Y    = z[2]                         # read Y from state
    A_d  = matrix_exp(A_c(Y) * ts)      # exact ZOH, torch-differentiable
    B_d  = ...
    return A_d @ x + B_d @ u
```

The block is stateful only in the sense that Y changes. It wires into `SSE_Interconnect`
the same way as `Linear_State_Block`. No changes to the interconnect itself.

### Architecture 2 — Δ(p) scheduling block (formal LFR-LPV)

In the formal LFR representation of LPV systems, the scheduling parameter enters as a
separate block Δ(p) in the feedback path of the LFR. The plant is decomposed as:

```
M = [A0 + ΔA(p), B0 + ΔB(p)]
```

where ΔA(p) = Δ(p) @ L_A (structured uncertainty block). This is the standard LFR-LPV
representation (Tóth, Schoukens & Tóth 2018). It requires:
- A different factorization of the plant matrices
- A dedicated Δ block in the interconnect (scheduling function block)
- Potentially different wiring topology — the scheduling variable enters as a separate signal

If Drenth uses Architecture 2, our `LPV_Linear_State_Block` plan (Architecture 1) would
need to be redesigned. This is the most important thing the thesis must resolve.

---

## Questions Drenth's thesis must answer

### Q1 — Which architecture for the LPV baseline?

Does Drenth represent the LPV baseline as:
- (a) A block that directly computes A(p)x + B(p)u in `forward()`, or
- (b) A formal LFR Δ(p) structure with separate scheduling and plant blocks?

Which sections/theorems define this? Does he give a concrete `forward()` or equivalent?

### Q2 — Does `SSE_Interconnect` still work, or is there a new class?

Does the LPV extension use the same `SSE_Interconnect` class with the same
`connect_signals` wiring, or does it introduce a new interconnect class / different
connection topology?

### Q3 — How does the scheduling variable Y enter the interconnect?

In our system, Y is a state (quasi-LPV). In the LFR structure, it must be available
to the LPV block at each forward call. Does Drenth:
- (a) Pass it as part of the state vector z (same as Architecture 1), or
- (b) Treat it as a separate scheduling signal that routes through the S matrix?

### Q4 — Normalization for LPV

In the LTI case, `normalize_linear_ss_matrices()` applies T_x, T_u, T_y to constant
A, B, C, D. For LPV, A(Y) is computed from physics at runtime. How does Drenth handle
normalization — is it applied to A_c(Y) before the matrix exponential, to A_d(Y) after,
or not at all for the physics-based LPV baseline?

### Q5 — Discretization approach in Drenth

Does Drenth specify how the continuous-time LPV model is discretized? Does he use:
- Exact ZOH (via matrix_exp or similar),
- Rectangular approximation,
- Or something else?

Is this compatible with our chosen Option E (`torch.linalg.matrix_exp`, D-012)?

### Q6 — Parallel augmentation still applies?

Does the parallel LFR augmentation structure (data-driven correction additive to baseline)
still hold in the LPV case? Or does the LPV structure change how the augmentation
connects to the baseline?

### Q7 — Existing LPV block in the codebase

The codebase has `Parameterized_LPV_Affine_Linear_State_Block` with:
- `A(p) = A0 + p·A1` (affine in p)
- `sched_state_ix` — index of scheduling variable in the state vector
- `p = x[sched_state_ix]²` (quadratic scheduling function)

Is this block consistent with Drenth's architecture, or is it user-added and inconsistent?
Does Drenth's thesis define the affine-in-p structure or a more general rational structure?

### Q8 — Block interface contract for LPV

Does the `Block` base class interface (`forward(z: Tensor) -> Tensor`) remain the same
for the LPV block, or does Drenth's framework add new interface requirements (e.g., a
separate `sched_signal` argument, or a `forward_lpv(z, p)` signature)?

---

## Decision gate — RESOLVED (2026-03-17)

*Assessed via `assess-paper` skill against `literature/drenth2025_lpv-lfr-thesis.pdf` and companion papers.*
Full Q-by-Q output: `assess-paper-workspace/iteration-1/drenth-assessment/with_skill/outputs/assessment.md`
Decision logged: `docs/decisions.md` D-013

### Q1 — Architecture ✅ Architecture 1 confirmed

Drenth eq. (2.29) collapses to direct A(p)x + B(p)u in the forward loop. The Δ(p) structure is a parameterization of the learned augmentation, not a constraint on the physics baseline. `LPV_Linear_State_Block.forward(z)` reading Y from state and computing A_d(Y), B_d(Y) is fully consistent.

### Q2 — SSE_Interconnect ✅ Unchanged

Hoekstra et al. (EJC 2025) uses fixed S matrices. Drenth Chapter 5 extends within the same framework. No new interconnect class is needed.

### Q3 — Scheduling variable routing ✅ From state vector

Y is read from the state inside `forward()`. It does NOT route through S. Drenth Section 2.4 explicitly supports self-scheduled quasi-LPV.

### Q4 — Normalization ⚠️ OPEN QUESTION

Drenth eq. (5.5) normalizes constant DT matrices (Tx·A·Tx⁻¹). For A_d(Y) computed via matrix_exp at runtime, this procedure must be adapted. Two options:
- **(i) Pre-scale A_c**: normalize the continuous-time system before physics computation — preserves physics interpretation
- **(ii) Post-wrap A_d**: apply Tx/Tx⁻¹ around A_d(Y) at each forward call — matches Drenth eq. (5.5) pattern

**Must be resolved before Step 3 implementation.** Option (i) is likely more natural.

### Q5 — Discretization ⚠️ Not addressed by Drenth

Drenth uses Euler for his examples. Option E (exact ZOH via `torch.linalg.matrix_exp`) is not contradicted — Drenth's framework accepts any DT baseline. Option E remains the correct choice (D-012) on engineering grounds.

### Q6 — Parallel augmentation ✅ Confirmed

Drenth Chapter 5.2 explicitly states "we consider only the parallel augmentation case." Additive correction to x[k+1] and y[k] applies.

### Q7 — Existing LPV block ✅ Architecturally consistent, simplified

`Parameterized_LPV_Affine_Linear_State_Block` is consistent with Drenth's affine-dependency case (Dzw=0). It uses a fixed quadratic scheduling function rather than a learned ResNet, but is not architecturally incompatible.

### Q8 — Block interface ✅ Unchanged

`forward(z: Tensor) -> Tensor` is the correct interface. Scheduling variable is computed inside `forward()` from the state. No new interface requirements.

---

## Why the Δ(p) block is needed for the augmentation but not the baseline

This distinction is easy to conflate — both the baseline and the augmentation vary with Y, but
the source of that variation is fundamentally different.

### Physics baseline — Y-dependency is known analytically

The FP model gives the exact formula for how Y enters the mass matrix M(Y), and therefore
A_d(Y) and B_d(Y). At each timestep, you plug in the current Y and get exact matrices.
Nothing needs to be learned about the Y-dependency structure — physics provides it.

```
LPV_Linear_State_Block.forward(z):
    Y    = z[2]                             # read from state
    A_d  = matrix_exp(A_c(Y) * ts)          # exact physics formula
    return A_d @ x + B_d @ u
```

The Y-dependency is **explicit and closed-form**. No Δ(p) block needed.

### Learned augmentation — Y-dependency must be parameterized

The augmentation learns an unknown correction to the baseline from data. That correction
also varies with Y (e.g. flexible dynamics shift with payload position) — but the formula is
unknown. A neural network must approximate it from measurements.

The problem: how do you parameterize a **matrix-valued function of Y** that a network can learn?

A naive free-matrix approach gives an unstructured black box. The **Δ(p) LFR structure**
is a principled factorization:

```
augmentation output = [M11  M12] [   x_aug   ]
                      [M21  M22] [ Δ(Y)·z_aug ]
```

where Δ(Y) is a block-diagonal matrix of Y values (fixed structure), and M11/M12/M21/M22
are **constant** learnable matrices. This gives:

- The network only trains constant matrices (simpler optimization)
- Y-dependency enters through a fixed structured channel
- Well-posedness (no algebraic loops) can be enforced via constraints on M22
- Rational dependencies on Y (like M(Y)⁻¹ in the baseline) can be represented exactly
  by stacking multiple Δ channels

### Summary

| | Physics baseline | Learned augmentation |
|--|--|--|
| Y-dependency | Known from physics | Unknown, must be learned from data |
| How Y enters | Explicit formula A_c(Y) | Via structured Δ(Y) factorization |
| What is learned | Nothing (fixed) | Constant matrices M11/M12/M21/M22 |
| Why Δ(p) structure | Not needed — physics is explicit | Makes learning tractable and well-posed |

The Δ(p) block is a **design choice for the augmentation** that makes the learning problem
tractable. The baseline doesn't need it because physics already specifies the Y-dependency exactly.

---

## Paper to assess

Drenth, J. (2025). *Gradient-Based Learning of LPV-LFR Models.*
Master thesis, TU Eindhoven.
File: `literature/drenth2025_lpv-lfr-thesis.pdf`

Companion paper (conference):
File: `literature/drenth2025_lpv-lfr-rational.pdf`
(May contain a more compact formulation — check both)

---

## Relevant existing code locations

| File | What it contains |
|------|-----------------|
| `model_augmentation/fit_systems/blocks.py` | All block classes including `Parameterized_LPV_Affine_Linear_State_Block` |
| `model_augmentation/fit_systems/interconnect.py` | `SSE_Interconnect`, `connect_signals`, algebraic loop check |
| `model_augmentation/utils/utils.py` | `normalize_linear_ss_matrices()` |
| `scripts/ecc_2025/msd_ndof_interconnect_dynamic.py` | MSD reference (Architecture 1 style — `Parameterized_MSD_State_Block` computes physics in forward()) |
| `docs/fp-augmentation-interface.md` | LTI baseline interface contract |
| `docs/decisions.md` D-010, D-011, D-012 | LPV baseline decisions logged so far |
