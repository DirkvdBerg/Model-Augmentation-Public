# LFR Baseline Implementation — Candidate Method: Resolve-and-Retain

*Written 2026-03-27. Proposes a stronger alternative to the xp-only collapsed baseline in `docs/lpv-lfr-interconnect.md`. This is a candidate method — open questions are listed explicitly in the Summary.*

---

## The requirement

The supervisor explicitly required (D-005, D-013, D-017) that **the baseline is in LFR form**.
Drenth Ch. 5 eq. 5.1–5.2 assumes the baseline is available as `{G^b, Δ^b(Y)}` — the constant
interconnection matrix and the scheduling block — not merely as a black-box function `A_c(Y)x + B_c(Y)u`.

The reason is that the augmentation in Drenth's framework couples with the baseline through the
combined LFR structure. The augmentation's learned correction interacts with the baseline's internal
latent signals `z^b` and `w^b`. If the baseline is collapsed to a direct state update, those signals
are never computed, and the structured coupling is lost. The augmentation degrades to a dumb additive
correction — which is exactly what the supervisor wanted to avoid by specifying LFR.

---

## The obstacle

The gantry LFR derivation (`LPV/LFR-derivation-supervisor.tex`) produces:

```
z = Cz·x + Dzw·w + Dzu·u        (G internal output)
w = Δ(Y)·z = Y·I₆·z             (scheduling block)
```

with `Dzw ≠ 0`:

```
Dzw = [-M₀⁻¹M₁,  -M₀⁻¹M₂]
      [ I₃,        0      ]
```

Naively wiring G and Δ(Y) as two separate blocks in Jan's `SSE_Interconnect` would create an
algebraic loop (`z → Δ → w → G → z`). Jan's framework explicitly rejects algebraic loops:

```python
# interconnect.py:135
assert not detect_algebraic_loop(directional_signal_connection_matrix)
```

---

## The alternative: the xp-only collapsed baseline

An alternative implementation resolves the loop by computing `A_c(Y)x + B_c(Y)u` directly inside
`CT_RK4_State_Block` and **never exposing z or w**. The block is a black box: inputs `(x, u)`,
output `xp`. z and w are computed internally and immediately discarded.

This is not wrong — it produces exact physics and satisfies the no-algebraic-loop constraint.
It is however **structurally weaker** for the augmentation objective: without exposing z and w,
the augmentation has nothing structured to connect to from the baseline side.

Specifically:
- Drenth Ch. 5 eq. 5.1 defines the combined system with cross-coupling between `z^b/w^b` (baseline)
  and `z^a/w^a` (augmentation) through M matrix off-diagonal blocks M_ab and M_ba.
- If z^b and w^b are never computed at runtime, the augmentation cannot receive them as structured inputs.
- The augmentation is then limited to additive corrections to `x[k+1]` and `y[k]` — the same
  structure as a non-LFR baseline.

This fails to meet the supervisor's requirement that the baseline should remain structurally available
in LFR form. It does not use the LFR derivation at the point where it matters: at runtime, where
the augmentation connects to it.

---

## The correct solution: explicit forward resolution

The key insight is: **the algebraic loop can be resolved analytically in a forward sequence**.
The result is not a black box — it produces explicit values of z and w at every step, without
ever creating a cycle in the signal graph.

### Step-by-step computation at each timestep

All steps below are computed **sequentially and explicitly** from known quantities. There is no
circularity at any point.

**Step 1. Read current state and input.**

```
x[k] ∈ R⁶     (current state: [q; q̇] in logical coordinates)
u[k] ∈ R³     (current input: generalized forces in logical coordinates)
Y[k] = x[k,2] (scheduling variable: payload Y-position, self-scheduled from state)
```

**Step 2. Compute fnet — the net generalized force.**

```
fnet = [-K, -C] · x[k] + u[k]   ∈ R³
```

This depends only on x[k] and u[k], both known. No loop.
K and C are constant matrices from the FP model.

**Step 3. Compute v — the resolved acceleration.**

```
v = M(Y[k])⁻¹ · fnet   ∈ R³
```

`M(Y[k])` is the 3×3 mass matrix evaluated at the current Y. It is invertible for all Y in
the operational range (proven in `docs/m-matrix-invertibility.md`). This is the only step that
uses `M(Y)⁻¹`, and it is explicit — no loop.

This is physically `v = q̈[k]`, the acceleration.

**Step 4. Compute the latent chain — v₁ and v₂.**

```
v₁ = Y[k] · v   ∈ R³       (first latent: Y · q̈)
v₂ = Y[k] · v₁  ∈ R³       (second latent: Y² · q̈)
```

Each depends only on quantities already computed. No loop.

**Step 5. Assemble z and w explicitly.**

From the derivation (`LFR-derivation-supervisor.tex`, eq. (zw)):

```
z[k] = [v;  v₁]  ∈ R⁶      (G internal output signal)
w[k] = [v₁; v₂]  ∈ R⁶      (scheduling block output: w = Δ(Y)·z = Y·z verified)
```

**z and w are now available as concrete tensors.** They are not hidden inside a black box.

**Step 6. Evaluate the CT state derivative using the constant G matrix.**

```
ẋ = Ax·x[k] + Bw·w[k] + Bu·u[k]
```

where the matrices are the constant G entries (all built from M₀⁻¹, the Y-independent part):

```
Ax = [0,       I₃      ]     (6×6, constant)
     [-M₀⁻¹K, -M₀⁻¹C  ]

Bw = [0,       0       ]     (6×6, constant)
     [-M₀⁻¹M₁,-M₀⁻¹M₂]

Bu = [0  ]                   (6×3, constant)
     [M₀⁻¹]
```

Note: M₀, M₁, M₂ are the constant coefficient matrices from the decomposition
`M(Y) = M₀ + M₁Y + M₂Y²`. They are precomputed once at block initialization.

**This is the G matrix acting on the signals — explicit, structured, using the LFR form.**

**Step 7. Integrate one timestep with RK4.**

At each RK4 sub-step, steps 1–6 are repeated for the intermediate state `x_i`:

```
For each RK4 substep (with intermediate state x_i):
    Y_i   = x_i[2]
    fnet_i = [-K,-C]·x_i + u[k]   (u held constant over interval)
    v_i   = M(Y_i)⁻¹ · fnet_i
    v1_i  = Y_i · v_i
    v2_i  = Y_i · v1_i
    w_i   = [v1_i; v2_i]
    ẋ_i   = Ax·x_i + Bw·w_i + Bu·u[k]

x[k+1] = RK4(x[k], u[k], ts)
```

The z and w at the full step k (Step 5 above) are from the initial `x[k]`, not the RK4 sub-steps.
They represent the latent signals at the measurement instant.

**Step 8. Return xp, z, w — as a single stacked tensor.**

The current `Block` contract (`forward(z) -> w`) requires a **single tensor output**, not a tuple.
Returning `(x[k+1], z[k], w[k])` directly is not compatible with the existing interface.

The practical solution is to concatenate:

```
output = cat([x[k+1], z[k], w[k]])   ∈ R^(6+6+6) = R^18
```

and use `connection_matrix` arguments in `connect_signals` to route the appropriate slices:
- indices 0:6  → the `xp` signal
- indices 6:12 → z^b, available as an input to the augmentation block
- indices 12:18 → w^b, available as an input to the augmentation block

This is an implementation detail but a real one — it requires care when setting up the wiring.
It does not require changes to `SSE_Interconnect` itself, but it does require explicit use of
selection/connection matrices when calling `connect_signals`.

---

## Why this is stronger than the xp-only approach

More precisely: this method is a **partial resolution** — it resolves the algebraic loop internally
but retains the resolved latent signals z and w as explicit runtime quantities, rather than
discarding them along with the loop.

The algebraic loop `z = f(z)` has a unique fixed point (because M(Y) is invertible). We compute
that fixed point directly in closed form via `v = M(Y)⁻¹·fnet`, rather than wiring it as a cyclic
signal path. The result is:

- z[k] and w[k] are concrete tensors, available after the forward call.
- The G matrix is applied explicitly in Step 6 using the constant entries from the derivation.
  They are not folded into an opaque A_c(Y) and discarded.
- The relationship `w[k] = Δ(Y[k])·z[k] = Y[k]·I₆·z[k]` holds exactly.

This is not "no collapse" — the loop IS resolved analytically. The distinction from the xp-only
approach is that resolution stops at z/w, not at xp. The latent signals survive.

---

## Why this does NOT create an algebraic loop in the Interconnect

Jan's `SSE_Interconnect` detects algebraic loops by checking whether any block's output feeds back
into its own input within the same timestep — a cyclic dependency in the signal graph.

In our implementation, the G block computes `(xp, z, w)` from `(x, u)` only. It does not receive z
or w as an input — it produces them. The scheduling Δ(Y) is not a separate block wired in a feedback
path. It is applied analytically inside the block as part of Steps 3–5.

**From the Interconnect's perspective, the baseline is a single block:**

```
Inputs:  x[k], u[k]
Outputs: x[k+1], z[k], w[k]
```

No feedback. No loop. The `detect_algebraic_loop` assertion passes.

---

## What exposing z and w enables — and what remains open

Exposing z^b and w^b is a **necessary condition** for structured LFR augmentation coupling.
It is not yet sufficient. This section is precise about what is established and what is not.

### What is established

Drenth Ch. 5 defines the combined baseline-plus-augmentation LFR with interconnection matrix M:

```
M = [M_bb  M_ba]    (baseline receives baseline signals; augmentation provides to baseline)
    [M_ab  M_aa]    (baseline provides to augmentation; augmentation receives augmentation signals)
```

By exposing z^b and w^b at runtime, the M_ab block becomes realizable: the augmentation block
can receive z^b and w^b as inputs via `connect_signals`. This is more than the xp-only approach
provides.

### What is not yet established

The M_ba block — augmentation signals feeding **back into** the baseline — is a different matter.
If the baseline block needs to receive augmentation signals as inputs within the same timestep,
this creates a same-step dependency that could reintroduce an algebraic loop in the signal graph.

Whether M_ba coupling is needed depends on the specific augmentation structure:

- **For strictly parallel augmentation** (D-003): the augmentation adds corrections to xp and y
  additively. The baseline does not receive augmentation signals as inputs. M_ba may be zero or
  absent. In this case, the concern does not apply.
- **For general LFR augmentation** (full Drenth Ch. 5): M_ba may be nonzero. In that case,
  same-step realizability in the current acyclic `SSE_Interconnect` is not yet verified.

**The claim in this document is therefore:**
- Exposing z^b and w^b is a meaningful improvement over the xp-only block.
- It enables at minimum the augmentation-receives-baseline direction of coupling (M_ab).
- Whether the full combined Drenth LFR (including M_ba) is realizable in the current
  `SSE_Interconnect` without further framework changes **has not yet been proven**.

This is an open question to be resolved before Step 3 implementation.

---

## Comparison with the xp-only collapsed approach

| Property | xp-only (collapsed) | Resolve-and-retain (this doc) |
|---|---|---|
| z and w available at runtime | No — discarded | Yes — concrete tensors |
| Algebraic loop in Interconnect | No | No |
| G matrix used explicitly | No — folded into A_c(Y) | Yes — Ax, Bw, Bu applied in Step 6 |
| Augmentation receives z^b/w^b | No | Yes — via connect_signals (M_ab direction) |
| Baseline receives augmentation signals | N/A | Not yet verified (M_ba direction) |
| Supervisor LFR requirement | Does not meet it | Candidate method — meets necessary condition |
| Full Drenth Ch. 5 coupling established | No | Not yet — open question |
| Block API compatible as-is | Yes | No — requires stacked output + selection matrices |
| Computational cost | M(Y)⁻¹ each step | M(Y)⁻¹ each step (same) |
| Correctness of xp | Exact | Exact (identical result) |

The two approaches produce the **same x[k+1]** — the physics is identical. The difference is
entirely in what structure is exposed and available for the augmentation to connect to.

---

## Verification: the G matrix recovers the original physics

We can verify that Step 6 (`ẋ = Ax·x + Bw·w + Bu·u`) with w from Steps 3–5 is equivalent to the
original equation `M(Y)·q̈ = -Kq - Cq̇ + u`.

The lower block of ẋ is the acceleration:

```
ẍ_lower = -M₀⁻¹K·q - M₀⁻¹C·q̇              (from Ax·x)
         + (-M₀⁻¹M₁)·v₁ + (-M₀⁻¹M₂)·v₂    (from Bw·w)
         + M₀⁻¹·u                            (from Bu·u)
```

Substituting v₁ = Y·v and v₂ = Y²·v:

```
= M₀⁻¹·(-Kq - Cq̇ - M₁Yv - M₂Y²v + u)
= M₀⁻¹·(-Kq - Cq̇ + u) - M₀⁻¹·(M₁Y + M₂Y²)·v
```

But v = M(Y)⁻¹·(-Kq - Cq̇ + u), so M(Y)·v = -Kq - Cq̇ + u, which gives
(M₀ + M₁Y + M₂Y²)·v = -Kq - Cq̇ + u, so:

```
-Kq - Cq̇ + u = M₀·v + M₁Yv + M₂Y²v
```

Substituting back:

```
ẍ_lower = M₀⁻¹·(M₀v + M₁Yv + M₂Y²v) - M₀⁻¹·(M₁Yv + M₂Y²v) = M₀⁻¹·M₀·v = v = q̈  ✓
```

The G matrix computation gives exactly `q̈ = M(Y)⁻¹·fnet`. The physics is recovered exactly.

---

## Critical architectural finding: resolve-and-retain is not LFR augmentation

*Added 2026-03-29, based on research in `LPV/Algebraic loops in LPV-LFR systems/`.*

This section states explicitly what the resolve-and-retain approach does and does not provide for the augmentation, and why the alternative is not currently achievable.

### The two architectures

**Architecture 1 — our current implementation (resolve-and-retain):**

```
Inside LFRBaselineBlock.forward():
    z₁ = torch.linalg.solve(M(Y), fnet)      ← loop resolved here
    z₂ = Y · z₁
    w₂ = Y · z₂
    z = [z₁; z₂],  w = [z₂; w₂]             ← post-resolution signals
    ẋ  = Ax·x + Bw·w + Bu·u
    x_next = RK4(...)

Output to Interconnect: [x_next | z | w]     ← augmentation sees these
```

The augmentation block Δ^b receives z and w **after** the algebraic loop has already been fully resolved. By the time augmentation sees z₁, it already equals M(Y)⁻¹·fnet — the full rational scheduling effect is baked in. The augmentation block adds a correction to xp on top of a physics model that has already run to completion.

**This is structurally equivalent to augmenting the collapsed explicit ODE:**
```
ẋ = Ac(Y)·x + Bc(Y)·u + [Δ^b correction]
```
The LFR structure (G, Δ) is not present at augmentation time. It was used only to derive Ac(Y) and Bc(Y) analytically.

---

**Architecture 2 — proper LFR augmentation (what Drenth Ch. 5 describes):**

```
G and Δ(Y) are separate wired blocks in the Interconnect.
Δ^b is wired in parallel or in series with Δ(Y).

The combined scheduling equation is:
    z = Cz·x + Dzw·[Δ(Y) + Δ^b]·z + Dzu·u

This is a joint implicit equation. Δ^b participates in computing z —
it can change how the rational dependence resolves, not just correct
the output after it has resolved.
```

In this architecture, z and w are genuinely live LFR signals. The augmentation is part of the closed loop, not an additive correction on top of it.

---

### Why Architecture 2 is not achievable for this system

Architecture 2 requires G and Δ(Y) as separate runtime blocks. That requires the signal path `z → Δ(Y) → w → G → z` to be a live wired loop in Jan's Interconnect. Jan's framework explicitly rejects this via the acyclicity assertion (`assert not detect_algebraic_loop(...)`).

The root cause is `Dzw ≠ 0`. If `Dzw = 0` the loop disappears and Architecture 2 becomes straightforward. But `Dzw = 0` is impossible for this system:

- `Dzw ≠ 0` is the structural mechanism that encodes rational parameter dependence in an LFR. The upper LFT formula is `Fu(G, Δ) = G₂₂ + G₂₁·Δ·(I − Dzw·Δ)⁻¹·G₁₂`. When `Dzw = 0` this collapses to `G₂₂ + G₂₁·Δ·G₁₂` — affine in Δ, which cannot represent `M(Y)⁻¹`.
- Setting `Dzw = 0` does not eliminate the rational structure — it loses it entirely. There is no equivalent LFR with `Dzw = 0` that produces the same input-output behavior for this system.
- This was confirmed in both AI research documents (`LPV/Algebraic loops in LPV-LFR systems/`) and is consistent with Zhou, Doyle & Glover (1996) and the LPVcore toolbox documentation from Tóth's group.

There are three workarounds, none of which preserve the intended LFR structure cleanly:

| Workaround | What it does | Why it falls short |
|---|---|---|
| Extended scheduling vector (e.g. pre-compute L(Y) = Cholesky factor) | Makes remaining interconnect affine | Δ is no longer Y·I₆; it encodes M(Y) knowledge, defeating the purpose |
| Descriptor/mass-matrix form | Hides the solve inside an implicit ODE solver | Still requires a solve at each step; doesn't expose z/w to augmentation |
| Loop shifting | Moves Dzw terms elsewhere | Introduces new rational dependencies or changes block definitions |

---

### What post-resolution z and w can still do

Exposing z and w is not useless — it is better than the xp-only collapsed approach. What it provides:

- The augmentation can **read** z^b and w^b as structured inputs — it sees signals that carry information about the current scheduling state (z₁ = q̈, z₂ = Y·q̈) rather than just the full state x.
- The M_ab coupling (augmentation receives baseline latent signals) is enabled and implemented — see `test_jan_compat.py` checks C and D.
- The physical interpretation of z and w is preserved: z₁ = q̈, z₂ = Y·q̈, w₂ = Y²·q̈.

What it does not provide:

- The augmentation cannot **participate in resolving** the rational scheduling effect. It only sees the result after M(Y) has been inverted.
- The M_ba coupling (augmentation output feeds back into the baseline's loop) is not realizable in the current framework without re-introducing an algebraic loop.
- Drenth Ch. 5's full combined-LFR coupling (where Δ^b modifies the scheduling channel itself) is not implemented.

---

### Open question for supervisor

The supervisor explicitly required LFR form (D-005, D-013, D-017). There are two possible interpretations:

1. **Derivation in LFR form, execution can resolve it**: The LFR derivation is used to understand structure, verify well-posedness, and derive the constant G matrices — but at runtime the loop is resolved analytically before augmentation. This is what we currently do.

2. **Augmentation interacts with the live LFR structure**: G and Δ(Y) are separate runtime blocks and Δ^b participates in the loop. This is Architecture 2 — which is not achievable with Jan's current framework for rational-LPV systems.

This distinction should be confirmed with Roland Tóth before implementing the training pipeline. If interpretation 2 is required, a different implementation strategy or framework extension is needed.

---

## Summary

This document proposes a **candidate implementation method** for the gantry LFR baseline that is
stronger than an xp-only collapsed block for the augmentation objective.

The method:

1. **Resolves the algebraic loop analytically** at each step by computing `v = M(Y)⁻¹·fnet` directly.
2. **Computes z and w explicitly** as `z = [v; v₁]`, `w = [v₁; v₂]` — available as tensors.
3. **Applies the constant G matrix** using the entries from the LFR derivation.
4. **Integrates with RK4** on the CT ODE.
5. **Exposes z and w** from the block output (as a stacked tensor with selection matrices for routing).

What is established:
- No algebraic loop in Interconnect: **yes** — z, w computed in forward sequence
- Exact physics: **yes** — verified algebraically, identical xp to the collapsed approach
- Baseline latent signals available at runtime: **yes** — necessary condition for LFR augmentation
- Augmentation can receive z^b/w^b as inputs: **yes** — M_ab direction enabled
- Supervisor's LFR requirement met: **candidate** — meets the necessary structural condition

What remains open:
- Full Drenth Ch. 5 combined coupling (M_ba direction): **not yet verified**
- Whether parallel augmentation needs M_ba at all: **to be confirmed**
- Exact block API and connection matrix wiring: **to be designed**
- Coordinate system of z/w (logical vs stage): **to be confirmed against D-006**
