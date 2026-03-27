# LPV-LFR Interconnect — Architecture and Decisions

*Last updated: 2026-03-27. Reflects D-011, D-012, D-013, D-017, D-018 (all updated 2026-03-20 to 2026-03-22).*

---

## Sources

- `literature/books/drenth2025_lpv-lfr-thesis.pdf` — primary CT LPV-LFR source. Defines `(G, Δ(p))` with `ẋ(t)`, `z(t)`, `w(t)`, `y(t)`.
- `literature/lpv-lfr/drenth2025_lpv-lfr-rational.pdf` — discrete-time companion IFAC paper. Defines `{M, Δ(p)}` in DT.
- `LPV/LFR-derivation-supervisor.tex` — completed gantry LPV-LFR derivation. Derives the constant G matrix and Δ(Y) = Y·I₆ explicitly from the dual-gantry physics.

---

## Current architecture (as of 2026-03-22)

### Baseline: CT_RK4_State_Block (D-011, D-013, D-018)

The physics baseline is implemented as a **continuous-time ODE integrated with RK4** at each forward call. It is NOT pre-discretized:

```python
class CT_RK4_State_Block(Block):
    def forward(self, z):
        Y   = z[state_ix_Y]              # read Y from state (self-scheduled)
        x   = z[:nx]
        u   = z[nx:]
        # CT vector field: ẋ = A_c(Y)·x + B_c(Y)·u
        # Integrate one RK4 step with dt = ts
        xp  = rk4_step(A_c(Y), B_c(Y), x, u, ts)
        return xp
```

Where `A_c(Y)`, `B_c(Y)` are computed from `M(Y)⁻¹` each step — the LFR derivation shows these reduce to M0⁻¹ in the G matrix entries, with Y-dependence isolated in Δ(Y).

**Why RK4 over ZOH (D-018, D-012 update 2026-03-20)**:
- Supervisor confirmed RK4 for the augmentation training loop (Step 3+)
- ZOH pre-discretization remains valid for Steps 1–2 validation (already completed)
- RK4 integrates the CT ODE directly — no pre-discretization, stays in continuous time
- Avoids the dynamic dependence caveat of ZOH for quasi-LPV systems (Tóth, D-012)

### Both baseline and augmentation use LFR structure (D-017)

Per supervisor confirmation (2026-03-22):
- **Baseline** has its own `Δ^b(Y)` derived from known physics. Fixed, not trained.
- **Augmentation** has a separate `Δ^a(Y)` with trainable parameters.
- Both follow Drenth Ch. 5 eq. 5.1–5.2.
- `SSE_Interconnect` is used unchanged — no new interconnect class needed (Q2 ✅).

---

## Completed LFR derivation

The gantry LPV-LFR realization is derived in `LPV/LFR-derivation-supervisor.tex`. Key results:

**Scheduling**: `Δ(Y) = Y·I₆` — a single scalar Y repeated 6 times.

**Latent variables**: `z = [v; v₁]`, `w = [v₁; v₂]` where `v = q̈`, `v₁ = Y·v`, `v₂ = Y·v₁`.

**Constant G matrix** (built from M₀, the Y-independent part of the mass matrix):

| | `x` | `w` | `u` |
|---|---|---|---|
| `ẋ` | `Ax = [0, I; -M₀⁻¹K, -M₀⁻¹C]` | `Bw = [0,0; -M₀⁻¹M₁, -M₀⁻¹M₂]` | `Bu = [0; M₀⁻¹]` |
| `z` | `Cz = [-M₀⁻¹K, -M₀⁻¹C; 0, 0]` | `Dzw = [-M₀⁻¹M₁, -M₀⁻¹M₂; I₃, 0]` | `Dzu = [M₀⁻¹; 0]` |
| `y` | `Cy = [I₃, 0]` | `Dyw = 0` | `Dyu = 0` |

Collapsing the loop recovers `M(Y)⁻¹` exactly — verified algebraically in the derivation.

All G matrix entries involve only **M₀⁻¹**, which is constant and precomputed once.

---

## Algebraic loop problem: why the explicit G + Δ wiring cannot enter the Interconnect

### The algebraic loop in the gantry LFR

The LFR loop equations from the derivation are:

```
z = Cz·x + Dzw·w + Dzu·u       (output equation of G)
w = Δ(Y)·z = Y·I₆·z            (scheduling block)
```

where:

```
Dzw = [-M₀⁻¹M₁,  -M₀⁻¹M₂]     ← NOT zero
      [ I₃,        0      ]
```

Because `Dzw ≠ 0`, substituting `w = Y·z` into the first equation gives:

```
z = Cz·x + Y·Dzw·z + Dzu·u
(I - Y·Dzw)·z = Cz·x + Dzu·u
```

**z depends on itself** through `Dzw`. This is an algebraic loop.

This is not a defect in the derivation — it is the correct LFR for a system with *rational* scheduling dependency (M(Y)⁻¹). The loop is well-posed because M(Y) is invertible (proven in `docs/m-matrix-invertibility.md`), which implies `(I - Y·Dzw)` is invertible, satisfying Drenth's well-posedness condition.

### Why Jan's Interconnect rejects it

`interconnect.py:135`:
```python
assert not detect_algebraic_loop(directional_signal_connection_matrix)
```

The framework unconditionally rejects algebraic loops regardless of well-posedness. The comment at `blocks.py:145` for `Parameterized_LPV_Affine_Linear_State_Block` explicitly acknowledges this constraint:

> *"p is computed from state to avoid algebraic loops in the Interconnect graph."*

Wiring G and Δ(Y) as two separate blocks with the `z → Δ → w → G` signal path creates exactly the loop the assertion guards against.

### What the algebraic loop resolves to

Solving `(I - Y·Dzw)·z = Cz·x + Dzu·u` analytically:

```
v = M₀⁻¹·fnet - Y·M₀⁻¹M₁·v - Y²·M₀⁻¹M₂·v
(I + Y·M₀⁻¹M₁ + Y²·M₀⁻¹M₂)·v = M₀⁻¹·fnet
M₀⁻¹·(M₀ + Y·M₁ + Y²·M₂)·v = M₀⁻¹·fnet
M(Y)·v = fnet
v = M(Y)⁻¹·fnet
```

**Resolving the algebraic loop recovers M(Y)⁻¹ exactly.** The LFR does not avoid computing M(Y)⁻¹ — it restructures so G uses only the constant M₀⁻¹, with the Y-dependence in Δ. At runtime, the loop resolution is equivalent to evaluating M(Y)⁻¹.

### Adaptation to the no-algebraic-loop constraint

#### Route 1 — Single CT block with RK4 (exact, current approach)

Compute the CT vector field `A_c(Y)·x + B_c(Y)·u` directly inside `CT_RK4_State_Block.forward()`, then integrate with RK4. The Interconnect never sees the loop.

```python
class CT_RK4_State_Block(Block):
    def forward(self, z):
        Y   = z[state_ix_Y]
        # A_c(Y), B_c(Y) use M(Y)⁻¹ — algebraic loop resolved analytically
        xp  = rk4_step(A_c(Y), B_c(Y), x, u, ts)
        return xp
```

The LFR derivation's value is theoretical: it proves valid LPV-LFR structure, identifies the constant G matrix, and provides the baseline for Drenth's augmentation framework. Runtime simulation uses the resolved form.

#### Route 2 — Affine approximation, Dzw = 0 (approximate, not chosen)

Approximate `A_d(Y) ≈ A₀ + Y·A₁`. Affine dependency gives `Dzw = 0` — no algebraic loop. Fits directly into `Parameterized_LPV_Affine_Linear_State_Block`.

Not chosen: the gantry has rational dependency (M(Y)⁻¹), not affine. Approximation degrades over large Y excursions.

### Summary

| | Explicit G + Δ wiring | Route 1: CT_RK4_State_Block | Route 2: affine approx |
|---|---|---|---|
| Dzw | ≠ 0 (rational) | hidden inside block | = 0 by approximation |
| Loop visible to Interconnect | yes — assertion fails | no | no |
| Exactness | exact | exact | approximate |
| Fits Interconnect | no | yes | yes |
| Discretization | — | RK4 on CT model | A₀ + Y·A₁ DT |

**Route 1 (CT_RK4_State_Block) is the current decision (D-011, D-018).**

---

## Why the Δ(p) block is needed for the augmentation but not the baseline

Both the baseline and the augmentation vary with Y, but for different reasons.

### Physics baseline — Y-dependency is known analytically

The FP model gives the exact formula for how Y enters M(Y), and therefore `A_c(Y)`, `B_c(Y)`. At each timestep, plug in Y and get the exact CT vector field. Nothing needs to be learned.

The Y-dependency is **explicit and closed-form** from the physics.

### Learned augmentation — Y-dependency must be parameterized

The augmentation learns an unknown correction from data. That correction also varies with Y, but the formula is unknown. A neural network must approximate it.

The **Δ(p) LFR structure** is a principled factorization:

```
augmentation output = [M11  M12] [   x_aug   ]
                      [M21  M22] [ Δ(Y)·z_aug ]
```

where Δ(Y) is a block-diagonal matrix of Y values (fixed structure), and M11/M12/M21/M22 are **constant** learnable matrices. This gives:

- The network only trains constant matrices (simpler optimization)
- Y-dependency enters through a fixed structured channel
- Well-posedness enforced via constraints on M22 (Drenth Theorem 2.5)
- Rational dependency on Y can be represented by stacking Δ channels

### Summary

| | Physics baseline | Learned augmentation |
|---|---|---|
| Y-dependency | Known from physics | Unknown, must be learned |
| How Y enters | Explicit CT formula A_c(Y) | Via structured Δ(Y) factorization |
| What is learned | Nothing (fixed) | Constant matrices M11/M12/M21/M22 |
| Why Δ(p) structure | Not needed — physics is exact | Makes learning tractable and well-posed |

---

## Resolved Q&A (historical, from 2026-03-17)

These questions were open before Drenth's thesis was assessed. All resolved.

| Q | Question | Resolution |
|---|---|---|
| Q1 | Which architecture? | Architecture 1 (direct A(p)x + B(p)u in forward). LFR is for augmentation parameterization, not a runtime constraint on the baseline. Drenth eq. 2.29. |
| Q2 | SSE_Interconnect unchanged? | Yes. No new class needed. |
| Q3 | How does Y enter? | From state inside forward(). Does not route through S. Drenth Sec. 2.4. |
| Q4 | Normalization? | Drenth eq. 5.5: Tx·A·Tx⁻¹. Applied to all LFR submatrices. |
| Q5 | Discretization? | Drenth uses Euler. Decision: RK4 on CT model (D-018), not ZOH. |
| Q6 | Parallel augmentation? | Confirmed. Drenth Ch. 5.2 explicitly uses parallel case. |
| Q7 | Parameterized_LPV_Affine_Linear_State_Block? | Consistent with affine case (Dzw=0). Augmentation-side block, not baseline. |
| Q8 | Block interface? | forward(z: Tensor) unchanged. Y computed inside forward() from state. |

---

## Relevant code locations

| File | What it contains |
|------|-----------------|
| `LPV/LFR-derivation-supervisor.tex` | Complete gantry CT LPV-LFR derivation — G matrix, Δ(Y) = Y·I₆ |
| `model_augmentation/fit_systems/blocks.py` | All block classes including `Parameterized_LPV_Affine_Linear_State_Block` |
| `model_augmentation/fit_systems/interconnect.py` | `SSE_Interconnect`, `connect_signals`, algebraic loop check |
| `scripts/gantry/gantry_lpv_torch.py` | Torch ZOH implementation — used for Steps 1–2 validation |
| `scripts/ecc_2025/msd_ndof_interconnect_dynamic.py` | MSD reference — `Parameterized_MSD_State_Block` computes physics in forward() |
| `docs/decisions.md` D-011, D-012, D-013, D-017, D-018 | All LPV baseline and discretization decisions |
| `docs/m-matrix-invertibility.md` | Proof that M(Y) is invertible across the operational range |
