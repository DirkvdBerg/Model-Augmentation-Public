# Roland Tóth — LFR Latent Variable Dimension Reduction
**Document**: LPV_LFR_Roland_notes.pdf
**Extracted**: 2026-03-28
**Context**: Roland confirms "The derivation is correct." These notes give a method to reduce n_z = n_w = 6 to a smaller value.

---

## Current Situation

The derived LFR has:
- z ∈ ℝ^6, w ∈ ℝ^6, Δ(Y) = Y·I₆
- z = [z₁; z₂] = [q̈; Y·q̈], w = [w₁; w₂] = [Y·q̈; Y²·q̈]

Roland noted (in LPV_LFR_Stepwise_Derivation_Feedback.pdf, page 6):
> "Actually now you read out a w_1 which has dimension 3. You could reduce this further."

The reduction notes give the systematic SVD-based approach.

---

## Method 1 — SVD of [Bw; I₆] · [Cz | Dzw | Dzu]

**Idea**: Find the effective rank ñ of the scheduling-channel coupling. If ñ < 6, the latent dimension can be reduced.

**Step 1**: Form the 12×15 matrix:
```
M = [Bw; I₆] · [Cz | Dzw | Dzu]
```
and compute its SVD: M = U·Σ·V^T. Count non-zero singular values → ñ ≤ 6.

**Step 2**: Split the factors:
```
L = U·Σ^(1/2) ∈ ℝ^(12×ñ),    split as L = [Lx; Lz]  (6×ñ each)
R = Σ^(1/2)·V^T ∈ ℝ^(ñ×15),  split as R = [Rx | Rw | Ru]  (ñ×6, ñ×6, ñ×3)
```

**Step 3**: New reduced LFR (dimension ñ):
```
z̃ = Cz̃·x + Dzw̃·w̃ + Dzu̇u        where  C̃z = Rx,  D̃zw = Rw·Lz,  D̃zu = Ru
ẋ  = Ax·x + B̃w·w̃ + Bu·u           where  B̃w = Lx  (top 6 rows of L)
w̃  = Y·Iñ·z̃
```

**Result for dual-gantry**: ñ = **6** (no reduction with Method 1).

Reason: rank([Cz|Dzw|Dzu]) = 6 → rank(M) = 6. Method 1 does not help here.

---

## Method 2 — SVD of Dzw first, then SVD (if Method 1 gives ñ = 6)

**Step 1**: Compute SVD of Dzw:
```
Dzw = Û·Σ̂·V̂^T,    r = rank(Dzw)
S₂ = Û·Σ̂ ∈ ℝ^(6×r),    P = V̂^T ∈ ℝ^(r×6)
```
This decomposes Dzw·w = S₂·ŵ  where  ŵ = P·w,  and  ẑ = P·z.

**Step 2**: Form the (6+r)×(6+r+3) matrix using the pre-rotated variables:
```
M₂ = [Bw; P] · [Cz | S₂ | Dzu]
```
and compute its SVD: M₂ = U₂·Σ₂·V₂^T. Count non-zero singular values → ñ ≤ r.

**Step 3**: Apply the same L/R splitting as Method 1, with S₂ replacing Dzw.

**Result for dual-gantry**:
- rank(Dzw) = **4**  (Dzw ∈ ℝ^(6×6) has 4 non-zero singular values)
- After applying Method 2: ñ = **4**

The latent dimension reduces from 6 to **4**.

---

## Numerical Verification (dual-gantry matrices)

| Matrix | Shape | Rank |
|--------|-------|------|
| Bw | 6×6 | **2** (top 3 rows are zero; M₁, M₂ span 2D) |
| Dzw | 6×6 | **4** |
| [Cz\|Dzw\|Dzu] | 6×15 | 6 |
| [Bw;I₆]·[Cz\|Dzw\|Dzu] | 12×15 | 6 → Method 1 gives ñ=6 |
| [Bw;P]·[Cz\|S₂\|Dzu] | 10×13 | **4** → Method 2 gives ñ=4 |

Singular values of `[Bw;I₆]·[Cz|Dzw|Dzu]`:
```
[1073, 2.90, 1.00, 1.00, 0.99, 0.87, ~0, ~0, ...]   → rank 6
```

Singular values of `[Bw;P]·[Cz|S₂|Dzu]` (Method 2):
```
[1073, 2.90, 0.99, 0.87, ~0, ~0, ...]   → rank 4
```

**Bw has rank 2** because:
- Top 3 rows of Bw = 0 (q̇ dynamics don't depend on w)
- Bottom 3 rows = [-M₀⁻¹M₁ | -M₀⁻¹M₂], and range(M₁) ∪ range(M₂) is 2D

Despite Bw having rank 2, the z-equation structure forces ñ = 4 (not 2), because the z readout requires more directions to correctly represent all scheduling interactions.

---

## What ñ = 4 Means

Instead of Δ(Y) = Y·I₆ (6 latent channels), the reduced realization has:
- Δ̃(Y) = Y·I₄  (4 latent channels)
- z̃ ∈ ℝ^4, w̃ ∈ ℝ^4
- B̃w ∈ ℝ^(6×4), C̃z ∈ ℝ^(4×6), D̃zw ∈ ℝ^(4×4), D̃zu ∈ ℝ^(4×3)

The G matrix reduces from ℝ^(15×15) to ℝ^(13×13).

---

## Implementation Plan

1. Compute SVD of Dzw → get S₂ ∈ ℝ^(6×4), P ∈ ℝ^(4×6)
2. Form M₂ = [Bw; P] · [Cz | S₂ | Dzu]
3. SVD of M₂ → L₂ ∈ ℝ^(10×4), R₂ ∈ ℝ^(4×13)
4. Split: L₂ = [Lx̃; L_P] (6×4 and 4×4), R₂ = [R̃x | R̃w | R̃u]
5. New matrices:
   - B̃w = Lx̃
   - C̃z = R̃x,  D̃zw = R̃w · L_P,  D̃zu = R̃u
   - Ax, Bu, Cy unchanged
6. Verify: collapse Δ̃ → recovers original Ac(Y), Bc(Y)

---

## Priority

| Step | Status |
|------|--------|
| Understand method | Done |
| Verify ñ=4 numerically | Done |
| Implement reduced G matrix | Pending |
| Verify collapse still exact | Pending |
| Update LaTeX derivation | Pending |
