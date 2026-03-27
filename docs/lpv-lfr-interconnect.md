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
- **Baseline** has its own `Δ^b(Y)` derived from known physics.
- **Augmentation** has a separate `Δ^a(Y)` with trainable parameters.
- Both follow Drenth Ch. 5 eq. 5.1–5.2.
- A nominal baseline can be kept fixed, but joint refinement of baseline and augmentation parameters remains possible in principle if the baseline is implemented as a parameterized differentiable block.
- `SSE_Interconnect` is used unchanged unless explicit algebraic-loop execution is introduced.

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

### Possible implementations under the no-algebraic-loop constraint

#### Collapsed baseline execution inside `CT_RK4_State_Block`

One possible implementation is to keep the baseline LPV-LFR available analytically as the pair `{G^b, Δ^b(Y)}`, but to execute it in resolved form inside a single differentiable state block.

The explicit baseline interconnection

```
z = Cz·x + Dzw·w + Dzu·u
w = Δ(Y)·z
```

reduces analytically to the original rational CT state equation

```
x_dot = A_c(Y)·x + B_c(Y)·u
```

where the exact dependence on `M(Y)⁻¹` is preserved. In this implementation, `CT_RK4_State_Block.forward()`:

1. reads the current scheduling value `Y` from the state,
2. evaluates the exact CT vector field `A_c(Y)·x + B_c(Y)·u`,
3. integrates one sampling step with RK4.

```python
class CT_RK4_State_Block(Block):
    def forward(self, z):
        Y   = z[state_ix_Y]
        # A_c(Y), B_c(Y) use M(Y)⁻¹ — algebraic loop resolved analytically
        xp  = rk4_step(A_c(Y), B_c(Y), x, u, ts)
        return xp
```

The LPV-LFR derivation remains essential in this case: it proves that the baseline is a valid LPV-LFR, identifies the constant interconnection matrices, and shows that solving the latent loop recovers the original rational mechanics exactly. Runtime simulation simply uses the resolved form of that same model.

#### Explicit algebraic-loop support in the interconnect

A second possible implementation is to keep the baseline as an explicit `G^b ↔ Δ^b(Y)` interconnection and extend the interconnect so it can execute algebraic loops directly.

This is not just a matter of allowing cyclic signal graphs. The framework would have to:

1. detect which connected blocks belong to an algebraic loop,
2. assemble the corresponding implicit loop equations at runtime,
3. solve the internal latent variables during each forward pass,
4. propagate gradients through that solve,
5. preserve well-posedness of the combined baseline-plus-augmentation interconnection.

This would turn the current acyclic signal-flow engine into an implicit solver for well-posed LFR loops.

### Why collapsed baseline execution is the preferred starting point

At the moment, collapsed baseline execution is the preferred starting point for the gantry for four reasons.

1. It preserves the exact rational baseline. No affine approximation is introduced, and the full `M(Y)⁻¹` dependence is retained.
2. It fits the current `SSE_Interconnect` assumption that the signal graph is acyclic. The interconnect only sees a standard differentiable state-update block.
3. It leaves the augmentation wiring unchanged. The augmentation can still be added in parallel to the same `x_{k+1}` and `y` signals.
4. It is much lower risk than redesigning the framework around implicit algebraic-loop solves.

So this option is preferred as the implementation starting point, even though explicit loop execution remains a conceivable future extension.

### Problems to keep in mind for augmentation

The choice above does **not** remove the key augmentation questions. It only changes how the baseline is executed internally.

#### Parallel augmentation still has to remain well-structured

The project baseline is combined with the learned correction through a parallel dynamic augmentation structure. That means the augmentation still has to:

- add its contribution to `x_{k+1}` and `y` through the existing additive wiring,
- remain differentiable through multi-step rollout,
- avoid capturing dynamics that are already explained by the baseline,
- preserve the intended separation between known physics and learned correction.

Collapsed execution of the baseline does not solve these issues automatically. It only prevents the baseline algebraic loop from appearing in the graph.

#### Baseline-parameter updates and augmentation-parameter updates are different things

Two meanings of "updating the baseline" must be kept separate:

- **Recomputing from the current scheduling value `Y`**: this happens every forward call and is not learning. The matrices `A_c(Y)` and `B_c(Y)` change because `Y` changes.
- **Learning baseline physical parameters during training**: this means masses, dampings, stiffnesses, or other physical coefficients are trainable variables optimized together with the augmentation parameters.

This distinction matters because Jan's framework and paper do allow the second case in principle.

#### What Jan's paper and code show

The augmentation paper explicitly discusses **joint identification of the baseline parameters and the learning-component parameters**. The MSD reference code does the same in practice:

- `Parameterized_MSD_State_Block` stores the baseline physical parameters as trainable `nn.Parameter`s,
- the MSD interconnect script uses that parameterized baseline block,
- the augmentation block is then added in parallel on top of it,
- the training loss includes regularization terms for the trainable baseline block.

So joint baseline-plus-augmentation learning is not hypothetical in the framework. It already exists in the MSD example.

#### Why collapsed baseline execution is still compatible with joint learning

Collapsed baseline execution does **not** prevent joint learning. A collapsed baseline block can still be parameterized.

The key point is that the interconnect only requires a differentiable block mapping `(x_k, u_k)` to `x_{k+1}`. It does **not** require the internal LPV-LFR loop of the baseline to appear explicitly in the graph. Therefore, in principle, a parameterized `CT_RK4_State_Block` can:

- refine selected baseline physical parameters, and
- be trained together with the augmentation parameters and augmented states.

So resolving the baseline algebraic loop internally is compatible with joint training. It only changes the runtime representation, not the optimization logic.

#### Specific gantry risks if baseline parameters are made trainable

The gantry is more delicate than the MSD example because the rational dependence comes from `M(Y)⁻¹`.

If baseline parameters entering `M(Y)` are made trainable, then:

- the exact rational dependence changes during training,
- the baseline `Δ^b(Y)` realization changes implicitly with those parameters,
- invertibility of `M(Y)` can no longer be taken for granted,
- well-posedness of the combined baseline-plus-augmentation LFR may need to be rechecked during training.

This is why trainable baseline parameters for the gantry should be treated cautiously. The safest staged interpretation is:

1. keep the inertia-related parameters fixed first,
2. allow augmentation parameters and augmented states to train in parallel,
3. only then consider constrained baseline-parameter refinement if needed.

For the gantry, the main augmentation-side problems to keep in mind are therefore:

- preserving the exact rational baseline while avoiding a visible algebraic loop,
- keeping the baseline/augmentation responsibility split interpretable,
- ensuring gradients remain well behaved in long rollouts,
- protecting invertibility of `M(Y)` if any baseline physical parameters are refined,
- and preserving well-posedness once baseline and augmentation are trained together.

**Important refinement on invertibility.** The warning above should be read as applying to **unconstrained** baseline-parameter updates. If the parameters entering `M(Y)` are parameterized so they remain physically meaningful by construction, then `M(Y)` should remain positive definite and therefore invertible. In that case, the main issue is no longer singularity of `M(Y)`, but rather:

- enforcing the physically meaningful parameterization throughout training,
- keeping the refined baseline interpretable as a valid mechanical model,
- and monitoring numerical conditioning of `M(Y)` even when invertibility is structurally guaranteed.

So for the gantry, the sharper statement is:

1. joint training of baseline and augmentation parameters is possible in principle,
2. unconstrained updates of parameters entering `M(Y)` are risky,
3. constrained physically meaningful updates can preserve invertibility,
4. conditioning and combined well-posedness still need attention during training.

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
