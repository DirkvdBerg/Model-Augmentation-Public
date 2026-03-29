# Supervisor Feedback — LPV_LFR_Stepwise_Derivation
**Reviewer**: Roland Tóth
**Document**: LPV_LFR_Stepwise_Derivation_Feedback.pdf
**Extracted**: 2026-03-28

---

## Summary of Major Themes

| Theme | Pages | Priority |
|-------|-------|----------|
| Conceptually incorrect derivation path via `v = M(Y)^{-1} fnet` | 3 | Critical |
| Standard z/w LFR notation not used — use direct scheduling structure | 3 | Critical |
| Wrong section title: "Interconnection Matrices" | 4 | High |
| Missing `(t)` time-dependence on all state variables | 1 | High |
| "logical coordinates" unexplained | 1 | High |
| Word choice fixes ("framework", "carried by", etc.) | 1, 2 | Medium |
| MATLAB verification suggestions (lft, mussv) | 6 | Informational |
| Possible dimension reduction of w_1 readout | 6 | Informational |

---

## Page 1

### 1. Strikeout + Replacement
- **Struck text**: `"derive a continuous-time LPV-LFR realization model within the framework of Drenth [2]."`
- **Caret comment**: `"according to"`
- **Action**: Replace "within the framework of Drenth [2]" → "according to Drenth [2]"

---

### 2. Strikeout + Replacement
- **Struck text**: `"plant-specific is carried by"`
- **Caret comment**: `"contained"`
- **Action**: Replace "carried by" → "contained"

---

### 3. Highlight + Comment
- **Highlighted**: `"variables and a repeated scheduling structure such the interconnection matrices remain constant."`
- **Comment**: `"G corresponds to an LTI filter"`
- **Action**: Revise description — G is an LTI filter, not just interconnection matrices (see also Page 4 comment).

---

### 4. Highlight + Comment
- **Highlighted**: `"is carried by Δ, if eliminating"`
- **Comment**: `"collapsing Delta into G..."`
- **Action**: Roland notices the collapse issue. This ties to the conceptual comment on Page 3 — the derivation should NOT eliminate/collapse Δ into G, but use the standard z/w structure.

---

### 5. Highlight + Caret
- **Highlighted**: `"In logical"`
- **Caret comment**: `"???? explain what they are...."`
- **Action**: Add explicit definition of "logical coordinates" (X = [X1, Theta, Y] vs stage [X1, X2, Y], the P-transform).

---

### 6. Five Caret comments: `"(t)"`
- **Location**: On state variables in the equations (five separate insertions)
- **Action**: All state variables need explicit time dependence — write `x(t)`, `u(t)`, `y(t)`, etc. throughout.

---

### 7. Highlight + Comment
- **Highlighted**: `"M(Y)"`
- **Comment**: `"So it has 2nd order polynomial dependence"`
- **Highlighted entries** (three separate highlights, no comment): `-mhY`, `mhY^2`, `-mhY`
- **Action**: Explicitly state M(Y) has 2nd order polynomial dependence in Y. Roland highlights the specific Y-dependent entries to confirm he sees them.

---

## Page 2

### 8. Highlight (no comment)
- **Highlighted**: `"denote the generalized input in logical coordinates. The MATLAB-derived continuous-time state-space model is then"`
- **Note**: Passage flagged, likely relates to the next comment below.

---

### 9. Text Note
- **Comment**: `"How? You can do this symbolically... or you need to compute it for every time instant..."`
- **Action**: Clarify how M(Y)^{-1} is computed — is it computed symbolically (using adj/det, giving rational expressions in Y) or numerically at each time step? The document must state this explicitly. The symbolic approach gives rational Y-dependence; the numerical approach requires a solve at each instant.

---

### 10. Highlight + Comment
- **Highlighted**: `"where adj(M(Y)) and det(M(Y)) are polynomial in Y, so the entries of M(Y)^{-1} are rational in Y. In Drenth's framework, the LPV-LFR interconnection is equivalent to a LPV-SS representation with rational"`
- **Comment**: `"work, approach etc. but not framework..."`
- **Action**: Replace "Drenth's framework" → "Drenth's work" or "Drenth's approach". Roland does not want the word "framework" used in this context.

---

### 11. Caret comment
- **Comment**: `"ith element of the"`
- **Action**: Insert "ith element of the" at the marked location (indexing clarification in a sentence).

---

## Page 3 — MOST IMPORTANT

### 12. Highlight (no comment)
- **Highlighted**: `"logical"`
- **Comment**: `"translational ?"`
- **Action**: Check whether "logical" should be "translational" in this context.

---

### 13. Strikeout + Comment
- **Struck text**: `"[−K, −C]"`
- **Comment**: `"Don't use commas in vectors and matrices..."`
- **Action**: Remove commas from matrix/vector notation throughout — write `[-K  -C]` not `[-K, -C]`.

---

### 14. Highlight + Major Comment — CRITICAL
- **Highlighted**: `"This motivates the first latent variable v := q̈ = M(Y)^{-1} fnet.  (eq. 19)"`
- **Comment** (full text):

> "This is conceptually incorrect, but the results you got are ok.
>
> You do the following:
>
> M(Y) q̈ = ...
>
> (M_0 + M_1 Y + M_2 Y^2) q̈ = ...
>
> M_0 q̈ + M_1 w_1 + M_2 w_2 = ...
>
> where
>   w_1 = Y * q̈ = Y * z_1
>   w_2 = Y * w_1 = Y * z_2
>
> where
>   z_1 = q̈
>   z_2 = w_1"

- **Action (Critical)**: The derivation path must change. Do NOT introduce v = M(Y)^{-1} fnet as the first latent variable. Instead, use the direct scheduling structure:
  1. Start from M(Y) q̈ = fnet
  2. Expand: (M0 + M1*Y + M2*Y^2) q̈ = fnet
  3. Rearrange: M0 q̈ = fnet - M1*w1 - M2*w2
  4. Define z1 = q̈, z2 = w1 (the LFR latent variables directly)
  5. Define w1 = Y*z1, w2 = Y*z2 (the Δ(Y) = Y*I structure)
  - The results are the same, but the derivation follows the standard LFR z/w notation from the start.

---

### 15. Highlight + Comment
- **Highlighted**: `"M(Y) v = fnet."`
- **Comment**: `"This is not correct..."`
- **Action**: Remove this step. The algebraic loop should not be presented as "solve M(Y)*v = fnet" — this is a collapse/resolution of the loop, not the standard LFR derivation. See comment 14 for the correct approach.

---

### 16. Text Note
- **Comment**: `"This is ok, but please use the standard notation in terms of w and z."`
- **Action**: Keep the mathematical content but rewrite using standard z/w notation consistently.

---

## Page 4

### 17. Text Note
- **Comment**: `"Correct."`
- **Note**: G matrix section verified correct.

---

### 18. Text Note
- **Comment**: `"Correct."`
- **Note**: Second item on this page verified correct.

---

### 19. Highlight + Comment
- **Highlighted**: `"Constant Interconnection Matrices"`
- **Comment**: `"They are not interconnection matrices, but the matrices of the SS representation that describes the LTI part."`
- **Action**: Rename section — "Constant Interconnection Matrices" → "Constant State-Space Matrices of the LTI Part G" (or similar). The G matrix entries (Ax, Bw, Bu, Cz, Dzw, Dzu, Cy) are SS matrices of the LTI filter G, not interconnection matrices.

---

## Page 5

### 20. Text Note
- **Comment**: `"Correct."`

### 21. Text Note
- **Comment**: `"Correct"`
- **Note**: Verification section passes. Both items on this page are confirmed correct.

---

## Page 6

### 22. Text Note (Verification suggestion)
- **Comment**:

> "But you can also do this in Matlab. You can compute (50) directly for some values of Y.
> You can use `lft(Y*eye(6), G)` to compute what the LFR form does....
> You can also prove this directly by computing the star product of Δ(Y) with G."

- **Action (Optional/Informational)**: Add a MATLAB verification step using `lft(Y*eye(6), G)` to independently confirm the LFR realization. The star product Δ(Y) ★ G gives the closed-loop system and can be compared against A_c(Y), B_c(Y) directly.

---

### 23. Highlight + Comment (Well-posedness)
- **Highlighted**: `"−Dzw Δ(Y)"`
- **Comment**:

> "D_zw is upper triangular. This can be checked when that critical value happens.
> Can be computed with mussv in Matlab.
> `mussv(D_zw, [-6, 0])`
> then you take the 1/upperbound that you computed to see how large Y will make this expression singular."

- **Action (Informational)**: Add MATLAB verification of well-posedness using `mussv(D_zw, [-6,0])`. The inverse of the upper bound gives the critical Y value at which `det(I - Dzw*Δ(Y))` could become singular. This is a rigorous way to confirm the loop is well-posed over the operational range.

---

### 24. Text Note
- **Comment**: `"Actually now you read out a w_1 which has dimension 3. You could reduce this further."`
- **Action (Informational)**: Roland suggests the w_1 readout (dimension 3) could potentially be reduced. This may point toward a lower-order LFR realization. Worth investigating whether the latent dimension can be reduced from 6 to a smaller value.

---

## Action List (Ordered by Priority)

| # | Priority | Page | Action |
|---|----------|------|--------|
| 1 | Critical | 3 | Rewrite derivation: use z1=q̈, z2=w1, w1=Y*z1, w2=Y*z2 directly. Remove v=M(Y)^{-1}fnet path. |
| 2 | Critical | 3 | Remove "M(Y)*v = fnet" step — replace with correct LFR z/w formulation. |
| 3 | Critical | 3 | Use standard z/w notation throughout (not v, v1, v2). |
| 4 | High | 4 | Rename section: "Constant Interconnection Matrices" → "Constant SS Matrices of the LTI Part" |
| 5 | High | 1 | Add explicit definition of "logical coordinates". |
| 6 | High | 1 | Add `(t)` to all state variables throughout the document. |
| 7 | Medium | 1 | "within the framework of" → "according to" |
| 8 | Medium | 1 | "carried by" → "contained" |
| 9 | Medium | 2 | "Drenth's framework" → "Drenth's work/approach" |
| 10 | Medium | 3 | Remove commas from matrix/vector notation |
| 11 | Medium | 2 | Clarify how M(Y)^{-1} is computed (symbolic vs per time instant) |
| 12 | Medium | 2 | Add "ith element of the" at marked location |
| 13 | Low | 3 | Check: "logical" → "translational"? |
| 14 | Info | 6 | Add MATLAB verification via `lft(Y*eye(6), G)` |
| 15 | Info | 6 | Add well-posedness check via `mussv(D_zw, [-6,0])` |
| 16 | Info | 6 | Investigate dimension reduction of w_1 (dim 3 → possibly smaller) |
