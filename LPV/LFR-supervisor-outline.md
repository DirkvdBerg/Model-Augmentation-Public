# LPV-LFR Supervisor Version Outline

This file gives the recommended structure for the clean supervisor-facing
derivation. It should be used together with:

- `LPV/LFR-supervisor-inclusion-guide.md`
- `LPV/LFR-derivation-verification.tex`

The goal is concise wording with complete mathematical steps.

## 1. Purpose And Scope

Short paragraph stating:

- goal: derive an exact CT LPV-LFR realization of the dual-gantry baseline
- source boundary: Drenth provides the generic LPV-LFR framework; the specific
  gantry realization is plant-specific
- exactness goal: the realization must collapse back to the MATLAB-derived CT
  state-space model

## 2. Starting Point: Dual-Gantry CT Model

Include:

- mechanical model
  `M(Y) ddot(q) + C dot(q) + K q = f_l`
- logical coordinates
  `q = [X, Theta, Y]^T`
- first-order state choice
  `x = [q^T, dot(q)^T]^T`
- MATLAB-derived CT state-space model
  `dot(x) = A_c(Y) x + B_c(Y) u`
- output map
  `y = C_c x`, `C_c = [I_3 0]`, hence `y = q`

This section should justify the output choice `y = q` by matching the MATLAB
output map.

## 3. Why The Model Is Rational And Why An LPV-LFR Is Needed

Include:

- definition of affine dependency
- definition of rational dependency
- explanation that `A_c(Y)` and `B_c(Y)` contain `M(Y)^{-1}`
- short rationality justification
  `M(Y)^{-1} = adj(M(Y)) / det(M(Y))`

Conclude that the exact gantry model is not naturally affine in `Y`, so the
rational LPV-LFR class is the natural target.

## 4. Drenth's CT LPV-LFR Target Form

Write Drenth's generic CT LPV-LFR equations:

- `dot(x) = A x + B_w w + B_u u`
- `z = C_z x + D_zw w + D_zu u`
- `y = C_y x + D_yw w + D_yu u`
- `w = Delta(p) z`

Also define the constant interconnection matrix `G`.

This is the main place to cite Drenth for:

- the CT LPV-LFR structure
- the block role of `G`
- the fact that `Delta(p)` carries the scheduling dependence

## 5. Scheduling Choice And Repeated-Delta Structure

Explain:

- the physical scheduling variable is the single coordinate `Y`
- `Y^2` is not introduced as a second independent scheduler
- richer dependence is represented by repeated action of the same scheduler
  inside `Delta`

Then state and justify:

- `Delta(Y) = Y I_6`
- one independent scheduler, repeated six times

Explicitly justify the repetition count `6` from the chosen realization:

- `v in R^3`
- `v_1 in R^3`
- `z = [v; v_1] in R^6`
- `w = [v_1; v_2] in R^6`

## 6. Plant-Specific Latent Variables And Loop Signals

Define:

- `f_gen = [-K, -C] x + u`
- `v = M(Y)^{-1} f_gen`
- `v = ddot(q) = [ddot(X), ddot(Theta), ddot(Y)]^T`
- `v_1 = Y v`
- `v_2 = Y^2 v = Y v_1`

State clearly:

- `v`, `v_1`, and `v_2` are whole vectors in `R^3`
- `v_1` and `v_2` are derived latent vectors, not components of `v`
- they are internal helper variables, not physical states

Then define:

- `z = [v; v_1]`
- `w = [v_1; v_2]`
- `w = Delta(Y) z`

Also include:

- `M(Y) = M_0 + Y M_1 + Y^2 M_2`

## 7. Derivation Of The Constant LPV-LFR Interconnection

This is the main derivation section and should keep the intermediate algebra.

### 7.1 Rewrite The Acceleration Equation

Show:

- `M_0 v = f_gen - M_1 v_1 - M_2 v_2`
- substitute `f_gen = [-K,-C] x + u`
- multiply by `M_0^{-1}`
- rewrite in terms of `w = [v_1; v_2]`

### 7.2 Derive The State Equation

Show:

- `dot(x) = [dot(q); v]`
- `dot(q) = [0 I_3] x`
- identify `A`, `B_w`, `B_u`

### 7.3 Derive The z-Equation

Show:

- `z = [v; v_1]`
- `v_1 = [I_3 0] w`
- identify `C_z`, `D_zw`, `D_zu`

### 7.4 Derive The Output Equation

Show:

- `y = q = [I_3 0] x`
- identify `C_y`, `D_yw`, `D_yu`

End the section by presenting the full constant matrix `G`.

## 8. Exact Recovery Of The Original CT Model

Show the collapse explicitly:

- substitute `w = Delta(Y) z = Y z`
- derive `w_1 = Y v`, `w_2 = Y v_1 = Y^2 v`
- substitute into the first block of the `z`-equation
- reduce to `M(Y) v = f_gen`
- recover `dot(x) = A_c(Y) x + B_c(Y) u`
- recover `y = q = C_c x`

Conclude that the LPV-LFR collapses exactly to the original MATLAB-derived CT
state-space model.

## 9. Well-Posedness

Keep this separate from the realization derivation.

### 9.1 Drenth's Exact Criterion

State:

- `I - D_zw Delta(Y)` must be nonsingular

This is the exact generic criterion from Drenth.

### 9.2 Plant-Specific Reduction

Write explicitly:

- `(I - D_zw Delta(Y)) z = C_z x + D_zu u`

Then substitute the chosen realization and reduce the block system to:

- `M(Y) v = f_gen`
- `v_1 = Y v`

### 9.3 Unique Solvability And Well-Posedness

State carefully:

- fix admissible `x`, `u`, and `Y`
- if `M(Y)` is invertible, then `v = M(Y)^{-1} f_gen` is unique
- then `v_1 = Y v` and `v_2 = Y^2 v` are unique
- therefore `z` and `w` are unique
- hence the algebraic loop is well-posed

This should be stated as a realization-specific equivalence, not as a generic
LPV-LFR fact.

### 9.4 Global Well-Posedness

Reference the companion proof that:

- `M(Y) > 0` for all real `Y`

Hence `M(Y)` is invertible for all real `Y`, so the chosen realization is
globally well-posed in `Y`.

## 10. Final Remarks And Scope Of Claims

Keep this short and explicit:

- the realization is exact
- it uses one physical scheduling variable `Y`
- it preserves the original MATLAB input-output model after collapse
- no minimality claim is made
- the well-posedness proof used here is plant-specific, not Drenth's generic
  sufficient theorem from Section 2.2

## Writing Style Notes

- keep the prose concise
- do not skip algebraic steps in the main derivations
- cite Drenth where the framework or structural class is borrowed
- clearly label plant-specific derivation steps when they begin
- avoid the internal-audit labels from the verification note
