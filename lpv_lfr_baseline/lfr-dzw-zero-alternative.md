# Alternative LFR Realization: Extended Scheduling Variables to Achieve Dzw = 0

*Written 2026-03-29. Documents an alternative LFR construction that eliminates the algebraic loop by absorbing M(Y)⁻¹ into the scheduling signal. Compare with the current resolve-and-retain approach in `docs/lfr-baseline-implementation-method.md`.*

> **Observation on Roland's feedback**: Roland Tóth is the author of the theory on rational-to-affine
> LPV conversion (Tóth 2010, Ch. 4) and is therefore aware of this alternative. His supervisor
> feedback did not suggest moving to the affine realization — it suggested SVD reduction on the
> existing Dzw ≠ 0 LFR. This may be an implicit signal that the current LFR form is acceptable for
> his purposes. This document is kept for theoretical completeness; whether the affine realization
> is actually required should be confirmed with Roland directly.

---

## Motivation

The current LFR realization (`LPV/LFR-derivation-supervisor.tex`) uses scheduling block Δ(Y) = Y·I₆
and has Dzw ≠ 0. This creates an algebraic loop when G and Δ(Y) are wired as separate blocks,
which prevents the augmentation block Δ^b from participating in the loop (Architecture 2 in
`docs/lfr-baseline-implementation-method.md`).

The root cause: Dzw ≠ 0 is structurally mandatory for any LFR that uses Y as a single scalar
scheduling variable to represent M(Y)⁻¹. Setting Dzw = 0 forces affine dependence on Y, which
cannot represent M(Y)⁻¹.

**The alternative**: instead of using Y as the scheduling variable, absorb the denominator
`det(M(Y))` into the scheduling signal. This produces three new scheduling variables that are
rational in Y, but the resulting LFR — with these variables as inputs — has Dzw = 0.

---

## Step 1: Factor M(Y)⁻¹ Explicitly

By Cramer's rule:

```
M(Y)⁻¹ = adj(M(Y)) / det(M(Y))
```

**det(M(Y))** is quadratic in Y (the 3×3 Lagrangian mass matrix has entries of degree ≤ 2):

```
det(M(Y)) = α + βY + γY²
```

where α, β, γ are scalar constants depending on the physical parameters.

**adj(M(Y))** is a 3×3 polynomial matrix. From the sparsity structure of M(Y) (with M[0,2]=M[2,0]=0
and M[2,2]=mh constant), its entries have the following Y-degree structure:

```
adj(M(Y)) entry degrees:
    [0,0]: degree 2    [0,1]: degree 1    [0,2]: degree 1
    [1,0]: degree 1    [1,1]: degree 0    [1,2]: degree 0
    [2,0]: degree 1    [2,1]: degree 0    [2,2]: degree 2
```

So adj(M(Y)) = Adj₀ + Adj₁·Y + Adj₂·Y² where Adj₀, Adj₁, Adj₂ are 3×3 constant matrices with
the following rank structure:

- **Adj₀** = constant part of adj(M(Y)) → full rank 3 (M₀ is invertible, so adj(M₀) is full rank)
- **Adj₁** = linear coefficient → non-zeros at entries [0,1], [0,2], [1,0], [2,0], [2,2] → rank 3
- **Adj₂** = quadratic coefficient → non-zeros only at entries [0,0] and [2,2] → **rank 2**

Therefore:

```
M(Y)⁻¹ = σ(Y)·Adj₀  +  Y·σ(Y)·Adj₁  +  Y²·σ(Y)·Adj₂
```

where `σ(Y) = 1/det(M(Y))`.

---

## Step 2: Define Three Scheduling Variables

```
ρ₁(Y) = σ(Y)    = 1 / det(M(Y))          [units: 1/(kg²·m²) or similar]
ρ₂(Y) = Y·σ(Y)  = Y / det(M(Y))
ρ₃(Y) = Y²·σ(Y) = Y² / det(M(Y))
```

These are **rational** functions of Y. They are NOT polynomial in Y.

However, since Y = x[2] is a system state (quasi-LPV), all three are **computable at every
timestep** from the current state: evaluate det(M(Y)) → compute reciprocal → multiply by Y and Y².

The only singularity is at det(M(Y)) = 0. This is the same well-posedness condition as the current
approach (M(Y) must be invertible). The two representations share the same domain of validity.

---

## Step 3: The State-Space Model is Affine in ρ

Substituting M(Y)⁻¹ = ρ₁·Adj₀ + ρ₂·Adj₁ + ρ₃·Adj₂ into the equations of motion:

```
ẋ = [0    I₃  ] x + [0] u  +  [0         0     ] x·ρ₁  +  [0] u·ρ₁
    [0    0   ]     [0]        [-Adj₀·K  -Adj₀·C]          [Adj₀]

                            +  [0         0     ] x·ρ₂  +  [0] u·ρ₂
                               [-Adj₁·K  -Adj₁·C]          [Adj₁]

                            +  [0         0     ] x·ρ₃  +  [0] u·ρ₃
                               [-Adj₂·K  -Adj₂·C]          [Adj₂]
```

Compactly:

```
ẋ = (A₀ + A₁ρ₁ + A₂ρ₂ + A₃ρ₃)·x + (B₀ + B₁ρ₁ + B₂ρ₂ + B₃ρ₃)·u
```

where:

```
A₀ = [0    I₃]     B₀ = [0  ]
     [0    0 ]          [0  ]

         [0         0     ]               [0    ]
A₁ =     [-Adj₀·K  -Adj₀·C]     B₁ =     [Adj₀]

         [0         0     ]               [0    ]
A₂ =     [-Adj₁·K  -Adj₁·C]     B₂ =     [Adj₁]

         [0         0     ]               [0    ]
A₃ =     [-Adj₂·K  -Adj₂·C]     B₃ =     [Adj₂]
```

**There is no M(Y)⁻¹ anywhere in these matrices.** The Y-dependence is entirely carried by
ρ = [ρ₁, ρ₂, ρ₃], and the A, B matrices are constant. This is a standard affine LPV-SS model.

---

## Step 4: The Resulting LFR Has Dzw = 0

The LFR for any affine LPV-SS model always has Dzw = 0 — the scheduling Δ appears only in the
feed-forward path, not in a feedback loop. The scheduling block is:

```
Δ(ρ) = diag(ρ₁·I_{n₁},  ρ₂·I_{n₂},  ρ₃·I_{n₃})
```

where nᵢ is the number of latent channels for scheduling variable ρᵢ, determined by the rank of Adjᵢ:

```
n₁ = rank(Adj₀) = 3
n₂ = rank(Adj₁) = 3
n₃ = rank(Adj₂) = 2

Total latent dimension: n₁ + n₂ + n₃ = 8
```

This is the non-minimal realization. A minimal LFR (e.g. via the SVD method in
`LPV/Feedback Supervisor/Roland_Toth_Reduction_Notes.md`) may reduce this further.

**Key consequence: no algebraic loop.** With Dzw = 0, the signal graph is feed-forward:

```
z → Δ(ρ) → w → G → z   becomes   u → G → z → Δ(ρ) → w → G (no feedback)
```

Jan's `SSE_Interconnect` acyclicity assertion passes by construction. G and Δ(ρ) can be wired
as genuinely separate runtime blocks.

---

## Step 5: What This Enables

Because Dzw = 0 and there is no algebraic loop, the augmentation block Δ^b can be placed in
the scheduling channel as a proper second block:

```
Δ_combined(ρ) = Δ(ρ) + Δ^b
```

The combined scheduling equation is:

```
w = (Δ(ρ) + Δ^b) · z
```

This is a joint equation. Δ^b modifies how the scheduling is resolved — it can correct errors in
how M(Y) enters the dynamics, not only correct the output after M(Y) has already been applied.
This is Architecture 2 as described in `docs/lfr-baseline-implementation-method.md`.

---

## Why This Approach Is Not in the Algebraic Loop Literature

The algebraic loop research documents (in `LPV/Algebraic loops in LPV-LFR systems/`) do not describe
this construction. The reason is that those documents answer a different question: *given a rational
LFR with Dzw ≠ 0, how do you handle the loop?* Their framing assumes the loop already exists and
asks how to manage it (resolve analytically, use descriptor form, shift it out, etc.).

The extended scheduling approach sidesteps that question entirely by reformulating the *model* before
building the LFR. It belongs to LPV-SS modeling theory — specifically the problem of converting a
rational parameter dependence into an equivalent affine one by absorbing the denominator into the
scheduling signal. This is treated in the LPV system identification and modeling literature, in
particular Tóth (2010), *Modeling and Identification of Linear Parameter-Varying Systems* (Springer),
Chapter 4, where rational LPV representations are converted to affine ones via scheduling variable
lifting. The two bodies of literature (loop handling vs. model reformulation) ask different questions
and do not cross-reference.

---

## Reduction via Roland's SVD Method

Roland's SVD dimension-reduction method (documented in
`LPV/Feedback Supervisor/Roland_Toth_Reduction_Notes.md`) is applicable here, but in a different
form than for the Dzw ≠ 0 case.

### Why Method 2 does not apply directly

Roland's Method 2 specifically exploits the SVD of Dzw to find a rotation matrix P that compresses
the latent channels. If Dzw = 0, the SVD of Dzw has rank 0 and the decomposition trivializes —
there is no Dzw structure to exploit.

### The applicable reduction: joint SVD across scheduling directions

For the affine-in-ρ LFR with Dzw = 0, the latent dimension per scheduling direction ρᵢ is bounded
by `rank([Aᵢ | Bᵢ])`. For this system:

```
For ρ₁:  [A₁ | B₁] = [ 0         0      | 0    ]
                      [-Adj₀·K  -Adj₀·C  | Adj₀]

For ρ₂:  [A₂ | B₂] = [ 0         0      | 0    ]
                      [-Adj₁·K  -Adj₁·C  | Adj₁]

For ρ₃:  [A₃ | B₃] = [ 0         0      | 0    ]
                      [-Adj₂·K  -Adj₂·C  | Adj₂]
```

The bottom block of each [Aᵢ | Bᵢ] has column space spanned entirely by Adjᵢ (since K and C are
fixed, the columns of Adjᵢ·K, Adjᵢ·C, Adjᵢ all lie in range(Adjᵢ)). Therefore:

```
rank([A₁|B₁]) = rank(Adj₀) = 3
rank([A₂|B₂]) = rank(Adj₁) = 3
rank([A₃|B₃]) = rank(Adj₂) = 2
Total (independent) = 8
```

However, **across scheduling directions** there may be shared structure: Adj₀, Adj₁, Adj₂ all arise
from the same M(Y) decomposition and multiply the same physical matrices K and C. The *joint* column
space of the combined matrix:

```
[[A₁|B₁] | [A₂|B₂] | [A₃|B₃]]
```

may be smaller than 8 if the adjugate matrices share a common subspace. Roland's Method 1 SVD —
applied jointly across all scheduling directions rather than per-direction — finds exactly this
minimal latent dimension. The result is the numerical rank of the combined matrix, which must be
computed from the actual gantry parameters to determine whether reduction below 8 is possible.

Given the sparsity of M(Y) and the diagonal structure of K and C, a reduction to 5–6 channels is
plausible but cannot be confirmed analytically. This is step 3 in "What Would Need to Be Done" below.

---

## Comparison with the Current Approach

| Property | Current (Y, Dzw ≠ 0) | Alternative (ρ = [σ, Yσ, Y²σ], Dzw = 0) |
|---|---|---|
| Scheduling signal | scalar Y (physical, simple) | vector ρ ∈ ℝ³, rational in Y |
| Scheduling block Δ | Y·I₆ — one scalar, 6 channels | diag(ρ₁I₃, ρ₂I₃, ρ₃I₂) — 3 scalars, 8 channels |
| Dzw | ≠ 0 (algebraic loop) | = 0 (no loop) |
| Jan's framework compatible | Not natively (acyclicity violated) | Yes — feed-forward by construction |
| Δ^b can join the loop | No — loop resolved before augmentation | **Yes** — Δ^b is a genuine second block |
| Latent dimension | 6 (4 after Roland's reduction) | 8 (possibly less via minimal realization) |
| Physical interpretation of z/w | z₁ = q̈, z₂ = Y·q̈ (clear) | zᵢ = Rᵢ·[x; u] (less direct) |
| Singularity condition | det(M(Y)) ≠ 0 (same) | det(M(Y)) ≠ 0 (same) |
| Derivation effort | Complete (LPV/LFR-derivation-supervisor.tex) | Not yet derived |

---

## What Would Need to Be Done

1. **Compute Adj₀, Adj₁, Adj₂ numerically** — verify rank structure claimed above
2. **Derive the LFR** for the affine-in-ρ system — construct Cz, Bw, Dzu for each scheduling direction
3. **Check minimality** — apply Roland's SVD method to find the minimal latent dimension
4. **Verify collapse** — confirm that closing the loop Δ(ρ)·z recovers Ac(Y)x + Bc(Y)u exactly
5. **Implement in lfr_matrices.py** — new `build_G_matrix_dzw_zero()` function
6. **Update lfr_block.py** — scheduling block now takes ρ = [σ, Yσ, Y²σ] as input, not just Y
7. **Confirm with Roland** — is this the intended LFR form, or is the resolve-and-retain approach acceptable?

---

## Open Question for Supervisor

This alternative achieves Dzw = 0 and makes proper LFR augmentation (Architecture 2) possible.
However it comes at a cost: the scheduling signal is no longer the simple physical variable Y, but
a vector of rational functions of Y. The scheduling block Δ(ρ) is more complex and the latent
dimension increases from 6 to 8 before any reduction.

The question for Roland Tóth is: **which representation was intended when requesting LPV-LFR form?**

- If the goal is to keep G and Δ as separate live runtime blocks with Dzw = 0, this alternative
  is the path forward.
- If the goal is to use the LFR to understand and derive the structure, and resolution at runtime
  is acceptable, the current approach (resolve-and-retain) is simpler and already complete.

The answer determines which implementation path to pursue.
