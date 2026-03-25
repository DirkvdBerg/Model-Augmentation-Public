# LPV-LFR Supervisor Inclusion Guide

This note records the content that should be carried into the clean
supervisor-facing LPV-LFR derivation. It is a writing guide distilled from the
current clarification session and should be used together with:

- `LPV/LFR-derivation-verification.tex`
- the eventual clean derivation draft

It is not itself the final derivation. Its role is to make sure the clean
version keeps the important arguments explicit without inheriting the full
internal-audit style of the verification note.

## Primary Reference File

Use `LPV/LFR-derivation-verification.tex` as the detailed source of truth,
especially these sections:

- Step 1: CT LPV-LFR target class
- Step 3: collapsed CT model to be recovered
- Step 4: rational vs affine classification
- Step 5: scheduling-variable choice
- Step 6: repeated-Delta structure
- Step 7: latent-variable definitions
- Step 8: loop-signal construction
- Step 9: derivation of the constant blocks of `G`
- Step 10: collapse back to the original CT model
- Step 11: dependence-class confirmation
- Step 14: plant-specific well-posedness route

Useful anchors in `LPV/LFR-derivation-verification.tex`:

- Step 1 starts around line 223
- Step 3 starts around line 279
- Step 4 starts around line 302
- Step 5 starts around line 327
- Step 6 starts around line 350
- Step 7 starts around line 378
- Step 8 starts around line 420
- Step 9 starts around line 462
- Step 10 starts around line 531
- Step 11 starts around line 577
- Step 14 starts around line 682

## Content To Include Explicitly

### 1. Boundary Between Drenth And The Dual-Gantry Derivation

State clearly:

- Drenth provides the generic CT LPV-LFR framework.
- Drenth provides the exact loop-solvability condition and a generic sufficient
  well-posedness theorem.
- The specific dual-gantry realization, latent-variable choice, `G` matrices,
  and reduction to `M(Y)v=f_gen` are plant-specific and are our own derivation.

This boundary should remain explicit in the clean version.

### 2. The Target CT Model And Why The LFR Must Collapse Back To It

State that the LPV-LFR is only acceptable if, after eliminating the latent
loop, it reproduces the original MATLAB-derived continuous-time state-space
model.

This exact-recovery requirement should remain central.

### 3. Define Rational Dependency Before Using It

Define the class first:

- affine dependence means matrix dependence linear in the scheduler
- rational dependence means dependence through ratios of polynomials in the
  scheduler

Then justify why the gantry model is rational:

- `A_c(Y)` and `B_c(Y)` contain `M(Y)^{-1}`
- `M(Y)` is polynomial in `Y`
- therefore `M(Y)^{-1}` is rational in `Y`

The clean version should not compute the full inverse, but it should state
explicitly that

`M(Y)^{-1} = adj(M(Y)) / det(M(Y))`

so the entries of `M(Y)^{-1}` are rational wherever `M(Y)` is invertible.

### 4. Explain Why The Physical Scheduler Is Only `Y`

State explicitly:

- the physical scheduling quantity is the single coordinate `Y`
- `Y^2` appears in the model but is not introduced as a second independent
  scheduler
- this preserves the physical coupling instead of overbounding the plant with
  unnecessary independent scheduling variables

This is where Drenth's rational-vs-affine scheduling argument is relevant.

### 5. Explain Why Repeated Delta Is Used

State explicitly:

- richer dependence is represented by repeated use of the same scheduler inside
  `Delta`
- the clean realization uses `Delta(Y) = Y I_6`
- this does not mean six independent schedulers; it means one scalar scheduler
  repeated over six latent channels

### 6. Explicitly Justify Why The Dimension Is `6`

This should be stated directly for the chosen realization:

- `v` is a 3-vector
- `v_1 = Yv` is a 3-vector
- `z = [v; v_1]` is therefore a 6-vector
- `w = [v_1; v_2]` is therefore a 6-vector
- hence `Delta(Y)` must act on `R^6`, giving `Delta(Y) = Y I_6`

Also state that this is specific to the chosen realization and is not a
minimality claim.

### 7. Define `v`, `v_1`, `v_2` Very Clearly

This should be kept explicit in the clean version:

- `v = M(Y)^{-1} f_gen`
- `v` is the logical-coordinate acceleration vector
- `v = ddot(q) = [ddot(X), ddot(Theta), ddot(Y)]^T`
- `v_1 = Yv`
- `v_2 = Y^2 v = Y v_1`

Also state:

- `v`, `v_1`, and `v_2` are whole vectors in `R^3`
- `v_1` and `v_2` are derived latent vectors, not components of `v`
- they are internal helper variables, not physical states

### 8. Show The Choice Of `z` And `w`

Keep explicit:

- `z = [v; v_1]`
- `w = [v_1; v_2]`
- `w = Delta(Y) z = Y z`

The clean version should briefly explain that repeated action of the same scalar
`Y` generates both the `Y` and `Y^2` terms.

### 9. Show The Full Derivation Of The Constant Blocks Of `G`

The supervisor version should explicitly derive:

- `A`
- `B_w`
- `B_u`
- `C_z`
- `D_zw`
- `D_zu`
- `C_y`
- `D_yw`
- `D_yu`

It is worth keeping the intermediate steps:

- start from `M_0 v = f_gen - M_1 v_1 - M_2 v_2`
- substitute `f_gen = [-K,-C]x + u`
- solve for `v`
- rewrite in terms of `w = [v_1; v_2]`
- use `dot(x) = [dot(q); v]`
- use `z = [v; v_1]`
- use `y = q = [I_3 0] x`

This was one of the main clarification points in the session and is good to
show explicitly in the clean version.

### 10. Explicitly Justify The Output Choice `y = q`

State directly:

- the output is chosen as `y = q`
- this matches the MATLAB output map
- with `x = [q^T, dot(q)^T]^T` and `C_c = [I_3 0]`, one has `y = C_c x = q`

This is a good short justification to keep.

### 11. Keep The Collapse Check Explicit

The clean version should still show that the LPV-LFR collapses back exactly to
the original state-space model:

- substitute `w = Delta(Y) z`
- reduce the loop to `M(Y) v = f_gen`
- use `v = M(Y)^{-1} f_gen`
- recover `dot(x) = A_c(Y) x + B_c(Y) u`

This is the exactness check and should stay explicit.

### 12. Write The Well-Posedness Bridge Explicitly

The clean version should not jump directly from Drenth's criterion to
invertibility of `M(Y)` without the bridge.

State the generic equation:

`(I - D_zw Delta(Y)) z = C_z x + D_zu u`

Then show that, for the chosen realization, this becomes the block system in
`v` and `v_1`, which reduces to

`M(Y) v = f_gen`, `v_1 = Yv`

This should be written explicitly, not only implied.

### 13. State Well-Posedness Carefully

The clean version should state:

- for the chosen dual-gantry realization, the internal loop has a unique
  solution if and only if `M(Y)` is invertible

The logic should be explicit:

- fix admissible `x`, `u`, and `Y`
- if `M(Y)` is invertible, then `v = M(Y)^{-1} f_gen` is unique
- then `v_1 = Yv` and `v_2 = Y^2 v` are unique
- hence `z` and `w` are unique
- this is exactly well-posedness for this realization

Also state clearly that this is not a generic LPV-LFR fact in isolation; it is
true because the chosen loop has already been reduced to that specific
mass-matrix equation.

### 14. Make The Proof Route Used In The Clean Version Explicit

State explicitly that:

- Drenth's Section 2.2 gives a generic sufficient theorem
- the clean dual-gantry proof instead uses the plant-specific exact reduction
  through `M(Y)`
- the companion `M`-invertibility proof then closes the argument

This avoids making it sound like the clean version is applying Theorem 2.5 when
it is actually using a different route.

### 15. No Minimality Claim

Keep explicit:

- the realization is exact
- no claim is made that the chosen latent dimension or repeated-block structure
  is minimal

## Content That Can Be Kept Brief

These helped during the internal clarification session but do not need to be
expanded much in the supervisor version:

- why `x` is both a signal and a state
- why `dot(q) = [0 I_3] x`
- why `Y I_3 v = Y v`
- general discussion of alternative latent-variable choices
- detailed explanation of the `l_infinity` ball, unless the generic sufficient
  theorem is discussed in some depth

## Suggested Tone For The Clean Version

The clean version should:

- stay mathematically explicit where the derivation would otherwise feel like a
  jump
- avoid the internal audit style of repeating every boundary label
- keep Drenth references precise but not overly defensive
- separate generic LPV-LFR framework statements from plant-specific realization
  steps

## Minimum Checklist Before Calling The Clean Version Complete

- rational dependency defined before use
- `y = q` justified and matched to MATLAB
- `v`, `v_1`, `v_2` defined clearly in logical coordinates
- `Delta(Y) = Y I_6` justified for the chosen realization
- `G` blocks derived with intermediate steps
- collapse back to the original CT model shown explicitly
- well-posedness bridge from `I - D_zw Delta(Y)` to `M(Y) v = f_gen` shown
- uniqueness / well-posedness logic stated carefully
- no minimality claim made
