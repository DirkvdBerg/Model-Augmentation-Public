# Dual-Gantry Mapping of the Generalized CT LPV-LFR Recipe

## Purpose

This note maps the generalized recipe in:

- `docs/drenth/ch2-generalized-recipe.md`

onto the dual-gantry continuous-time model documented in:

- `LPV/LFR-derivation.tex`

The purpose is **verification**, not presentation quality. For each step, the note records:

1. the exact Drenth support available,
2. the generalized step extracted from Drenth,
3. the dual-gantry application,
4. the status of the step:
   - `Direct from Drenth`
   - `Generalized from Drenth`
   - `Own dual-gantry derivation`
5. what is **not yet justified** at this stage.

This revision explicitly incorporates Drenth Section `2.2`, so the well-posedness part is now split into:

- Drenth's **generic sufficient well-posedness route**, and
- the dual-gantry **sharper plant-specific route** based on reduction to `M(Y) v = f_gen`.

## Scope Boundary

This note intentionally uses:

- `docs/drenth/ch2-sec21-source.md`
- `docs/drenth/ch2-sec211-source.md`
- `docs/drenth/ch2-sec22-source.md`
- `docs/drenth/ch2-generalized-recipe.md`
- `LPV/LFR-derivation.tex`
- `LPV/M-invertibility.tex`

This note intentionally does **not** yet use:

- Drenth Sections `2.3` and later identification material,
- any augmentation material from Chapter 5,
- any external source beyond the current project notes and derivations.

## Reading Rule

Steps `1`--`12` mirror the stricter generalized recipe and map it onto the dual-gantry model.

Steps `13`--`15` are then added explicitly to separate the two well-posedness routes:

- the exact basic solvability condition from Section `2.1`,
- Drenth's sufficient theorem from Section `2.2`,
- the sharper plant-specific reduction used for the gantry.

Nothing is merged for brevity.

## Step 1. Fix the target representation class as a CT LPV-LFR

### 1. Exact Drenth support

From `docs/drenth/ch2-sec21-source.md`:

- Section `2.1` explicitly defines an LPV-LFR as a pair `(G, Delta(p))`.
- Eq. `(2.1)` gives the continuous-time interconnection.

This is direct framework support.

### 2. Generalized step

Adopt the CT LPV-LFR pair `(G, Delta(p))` as the target representation class.

### 3. Dual-gantry application

In `LPV/LFR-derivation.tex`, the target class is taken to be a CT LPV-LFR for the dual-gantry baseline:

- state: `x = [q^T, qdot^T]^T in R^6`
- input: `u := f_ell`
- output: `y = C_c x = q`

The intent is to represent the dual-gantry baseline as a constant interconnection matrix `G` together with a scheduling block depending only on `Y`.

### 4. Status

- `Direct from Drenth` for the choice of CT LPV-LFR as the target class
- `Own dual-gantry derivation` for the particular physical state/input/output assignment

### 5. What is not yet justified

At this stage, Drenth does **not** justify:

- the particular choice of gantry state,
- the particular output selection,
- or the actual existence of a convenient realization for this plant.

It only justifies the structural target class.

## Step 2. Treat the model as a constant interconnection plus a scheduling block

### 1. Exact Drenth support

From `docs/drenth/ch2-sec21-source.md`:

- Section `2.1` presents the LPV-LFR as a nominal constant interconnection `G` combined with a parameter-varying block `Delta(p)`.
- It is a safe inference from Section `2.1` that the dependence is intended to be isolated in `Delta(p)`.

### 2. Generalized step

Organize the reformulation so that:

- all constant linear structure lives in `G`,
- all scheduling dependence is pushed into `Delta(p)`.

### 3. Dual-gantry application

The current gantry derivation already adopts this organizational goal:

- constant quantities:
  - `M_0`, `M_1`, `M_2`
  - `K`, `C`
  - the eventual blocks of `G`
- scheduling quantity:
  - the scalar coordinate `Y`

The entire reformulation is built so that `Y` enters only through the final chosen Delta block.

### 4. Status

- `Direct from Drenth` for the structural division
- `Generalized from Drenth` for using this as a design rule during reformulation

### 5. What is not yet justified

This step does **not** yet tell us:

- how to push the `Y`-dependence into the loop,
- or whether the chosen plant algebra allows a clean isolation of `Y`.

## Step 3. State the collapsed CT model that the LPV-LFR must reproduce

### 1. Exact Drenth support

From `docs/drenth/ch2-sec21-source.md`:

- Eq. `(2.3)` gives the collapsed LPV-SS form.
- Eq. `(2.4)` gives the latent-variable elimination formula.

The generalized recipe therefore treats exact recovery of the collapsed CT model as the central correctness criterion.

### 2. Generalized step

Write down the continuous-time model that must be recovered exactly after collapse of the latent loop.

### 3. Dual-gantry application

The target CT model is the MATLAB-derived baseline in `LPV/LFR-derivation.tex`, Section `Starting Point: CT State-Space from MATLAB`:

- `x_dot(t) = A_c(Y) x(t) + B_c(Y) u(t)`
- `y(t) = C_c x(t)`

with

- `A_c(Y) = [[0, I_3]; [-M(Y)^(-1) K, -M(Y)^(-1) C]]`
- `B_c(Y) = [[0]; [M(Y)^(-1)]]`
- `C_c = [I_3, 0]`

This is the model the LPV-LFR must reproduce after loop elimination.

### 4. Status

- `Generalized from Drenth` for the verification viewpoint
- `Own dual-gantry derivation` for the specific target equations

### 5. What is not yet justified

At this point, Drenth does **not** tell us how to build the realization.

This step only establishes the target of the verification.

## Step 4. Decide whether the target should be affine or rational

### 1. Exact Drenth support

From `docs/drenth/ch2-sec21-source.md`:

- affine dependency is the special case `D_zw = 0`,
- rational dependency is admitted by the general LPV-LFR collapse.

From `docs/drenth/ch2-sec211-source.md`:

- rational dependency can reduce overbounding,
- and can reduce the number of independent scheduling variables in some examples.

### 2. Generalized step

Decide whether the plant can be represented adequately in the affine special case, or whether the model should retain genuinely rational dependency.

### 3. Dual-gantry application

In `LPV/LFR-derivation.tex`, the baseline CT matrices depend on `M(Y)^(-1)`.

Since `M(Y)` is polynomial in `Y`, but `A_c(Y)` and `B_c(Y)` depend on the inverse `M(Y)^(-1)`, the collapsed dependence is rational in `Y`.

So the gantry model is naturally placed in the rational LPV-LFR class rather than the affine special case.

### 4. Status

- `Generalized from Drenth` for the modeling decision
- `Own dual-gantry derivation` for concluding that this particular plant is rational

### 5. What is not yet justified

This step does **not** yet show:

- how to realize the rational dependence,
- or that the chosen rational realization is the best or minimal one.

## Step 5. Choose scheduling variables with coupling loss in mind

### 1. Exact Drenth support

From `docs/drenth/ch2-sec211-source.md`:

- overbounding can arise when coupling between scheduling variables is discarded,
- the MSD affine example introduces independent variables `p_1 = x` and `p_2 = x^2`,
- the rational example instead keeps one scheduling variable and uses repeated structure.

### 2. Generalized step

Choose scheduling variables so that unnecessary independent scheduling freedom is avoided.

### 3. Dual-gantry application

For the gantry baseline, the natural physical scheduling quantity is the single coordinate `Y`.

The dependence on `Y^2` appears inside `M(Y)`, but it is not introduced as a second independent scheduling variable. Instead, it is treated as structured dependence induced by the same physical quantity `Y`.

### 4. Status

- `Generalized from Drenth`

### 5. What is not yet justified

This step does **not** yet justify:

- how the `Y^2` dependence will be represented inside an LFR,
- only that introducing a separate free scheduling variable for `Y^2` would be structurally less faithful to the original physics.

## Step 6. Use repeated Delta-block structure to express richer dependency without introducing more independent scheduling variables

### 1. Exact Drenth support

From `docs/drenth/ch2-sec21-source.md`:

- Eq. `(2.2)` defines repeated block-diagonal Delta structure.

From `docs/drenth/ch2-sec211-source.md`:

- Eqs. `(2.8)`--`(2.9)` show repeated use of one scheduling variable inside `Delta`.

### 2. Generalized step

If richer dependency must be represented, use repetition of the same scheduling variable inside `Delta(p)` instead of automatically increasing the number of independent scheduling variables.

### 3. Dual-gantry application

In `LPV/LFR-derivation.tex`, the eventual scheduling block is chosen as:

- `Delta(Y) = Y I_6`

This is a repeated-scalar structure: the same physical scheduling variable `Y` acts repeatedly on the latent loop signals.

### 4. Status

- `Direct from Drenth` for the admissibility of repeated Delta structure
- `Own dual-gantry derivation` for the specific choice `Y I_6` and the dimension `6`

### 5. What is not yet justified

Drenth does **not** justify:

- why the repetition count should be exactly `6`,
- or why this is a minimal Delta dimension.

That belongs to the plant-specific realization choice.

## Step 7. Introduce latent loop variables so the scheduling action is expressed through `w = Delta(p) z`

### 1. Exact Drenth support

From `docs/drenth/ch2-sec21-source.md`:

- Section `2.1` explicitly introduces latent variables `z` and `w`.

From `docs/drenth/ch2-sec211-source.md`:

- repeated scheduling action in the rational MSD example is carried through the latent loop.

### 2. Generalized step

Introduce internal latent variables so that the scheduling dependence is expressed through the Delta-loop relation `w = Delta(p) z`.

### 3. Dual-gantry application

In `LPV/LFR-derivation.tex`, the plant-specific latent variables are chosen as:

- `f_gen = [-K, -C] x + u`
- `v = M(Y)^(-1) f_gen`
- `v_1 = Y v`
- `v_2 = Y^2 v = Y v_1`

Then the loop signals are defined as:

- `z = [v; v_1]`
- `w = [v_1; v_2]`

so that the relation `w = Delta(Y) z = Y z` holds.

### 4. Status

- `Direct from Drenth` for the existence and role of latent loop variables
- `Own dual-gantry derivation` for the actual variables `v, v_1, v_2`, `z`, and `w`

### 5. What is not yet justified

This is one of the most important boundaries:

- Drenth does **not** tell us to choose `v, v_1, v_2`,
- Drenth does **not** give a general algorithm for choosing latent variables,
- so the concrete latent-variable design here is fully plant-specific.

## Step 8. Rewrite the model so that all scheduling dependence is pushed into the Delta loop

### 1. Exact Drenth support

From `docs/drenth/ch2-sec21-source.md`:

- Section `2.1` supports the structural goal that `G` should be constant and scheduling dependence should act through `Delta(p)`.

### 2. Generalized step

Rewrite the plant equations so that the parameter dependence appears only through the latent loop.

### 3. Dual-gantry application

The crucial preparatory algebra in `LPV/LFR-derivation.tex` is:

- `M(Y) = M_0 + Y M_1 + Y^2 M_2`

Then, with `v = M(Y)^(-1) f_gen`, pre-multiplying by `M(Y)` yields:

- `M_0 v = f_gen - Y M_1 v - Y^2 M_2 v`

Using the latent definitions:

- `v_1 = Y v`
- `v_2 = Y^2 v`

this becomes:

- `M_0 v = f_gen - M_1 v_1 - M_2 v_2`

so the `Y`-dependence has been converted into dependence on the loop variables.

### 4. Status

- `Generalized from Drenth` for the design objective
- `Own dual-gantry derivation` for the actual algebraic rewrite

### 5. What is not yet justified

This step is **not** a direct Drenth construction.

Drenth does not tell us:

- how to decompose `M(Y)`,
- how to derive this latent-variable substitution,
- or why this specific rewrite is the best one.

## Step 9. Read off the constant block matrix `G`

### 1. Exact Drenth support

From `docs/drenth/ch2-sec21-source.md`:

- Eq. `(2.1)` gives the constant block partition of `G`.

### 2. Generalized step

Once the equations are written in LPV-LFR form, identify the constant blocks of `G` by collecting coefficients of `x`, `w`, and `u`.

### 3. Dual-gantry application

In `LPV/LFR-derivation.tex`, this produces:

- state equation blocks:
  - `A = [[0, I_3]; [-M_0^(-1) K, -M_0^(-1) C]]`
  - `B_w = [[0, 0]; [-M_0^(-1) M_1, -M_0^(-1) M_2]]`
  - `B_u = [[0]; [M_0^(-1)]]`

- loop equation blocks:
  - `C_z = [[M_0^(-1)[-K, -C]]; [0]]`
  - `D_zw = [[-M_0^(-1) M_1, -M_0^(-1) M_2]; [I_3, 0]]`
  - `D_zu = [[M_0^(-1)]; [0]]`

- output equation blocks:
  - `C_y = [I_3, 0]`
  - `D_yw = 0`
  - `D_yu = 0`

### 4. Status

- `Direct from Drenth` for the block structure that must be matched
- `Own dual-gantry derivation` for every explicit matrix formula listed above

### 5. What is not yet justified

Drenth does **not** justify:

- these specific formulas,
- their uniqueness,
- or their minimality.

This is plant-specific coefficient matching.

## Step 10. Collapse the latent loop and verify exact recovery of the target CT model

### 1. Exact Drenth support

From `docs/drenth/ch2-sec21-source.md`:

- Eq. `(2.4)` gives the latent-variable elimination idea,
- Eq. `(2.3)` states that the result is the collapsed LPV-SS model.

### 2. Generalized step

Use loop collapse as the main correctness check: substitute the Delta relation, eliminate the latent variables, and verify that the resulting CT model matches the target.

### 3. Dual-gantry application

In `LPV/LFR-derivation.tex`, the loop collapse is checked explicitly:

- `w = Delta(Y) z = Y z`
- hence `w_1 = Y v`
- and `w_2 = Y v_1 = Y^2 v`

Substituting into the loop equation gives:

- `M(Y) v = f_gen`

Substituting this into the state equation gives:

- `x_dot = [qdot; v]`
- `= [qdot; M(Y)^(-1) f_gen]`
- `= A_c(Y) x + B_c(Y) u`

which matches the target CT model from Step 3.

### 4. Status

- `Direct from Drenth` for the collapse-as-verification principle
- `Own dual-gantry derivation` for the explicit reduction to `M(Y) v = f_gen`

### 5. What is not yet justified

Drenth does **not** justify:

- the specific form of this reduction,
- only that collapse is the right verification step.

So the identity `M(Y) v = f_gen` is a plant-specific correctness result, not a Drenth theorem.

## Step 11. Identify whether the collapsed result is affine or rational

### 1. Exact Drenth support

From `docs/drenth/ch2-sec21-source.md`:

- Section `2.1` explicitly states that the collapsed LPV-LFR admits rational dependency,
- and that affine dependency is the `D_zw = 0` special case.

### 2. Generalized step

After collapse, classify the resulting model as affine or rational.

### 3. Dual-gantry application

Because the collapsed matrices depend on `M(Y)^(-1)`, the final gantry model is rational in `Y`, not affine.

### 4. Status

- `Direct from Drenth` for the classification framework
- `Own dual-gantry derivation` for classifying this specific plant

### 5. What is not yet justified

This step does **not** yet say anything about:

- whether another approximate affine embedding might also be useful,
- or whether the current rational realization is optimal.

It only classifies the exact baseline realization.

## Step 12. State the exact basic loop-solvability condition

### 1. Exact Drenth support

From `docs/drenth/ch2-sec21-source.md`:

- Section `2.1` explicitly states that well-posedness requires `I - D_zw Delta(p)` to be nonsingular for all admissible scheduling values.

### 2. Generalized step

Every candidate realization must first be checked against the exact algebraic-loop solvability condition.

### 3. Dual-gantry application

For the dual-gantry realization, the exact basic condition is:

- `I - D_zw Delta(Y)` nonsingular for all admissible `Y`.

This is the exact LFR-level condition before any sufficient theorem or plant-specific simplification is introduced.

### 4. Status

- `Direct from Drenth`

### 5. What is not yet justified

At this step alone, we do **not** yet have:

- a sufficient theorem that is easy to verify,
- or a plant-specific simplification of the condition.

Those are two different next moves, and they must be kept separate.

## Step 13. State Drenth's generic sufficient well-posedness route from Section 2.2

### 1. Exact Drenth support

From `docs/drenth/ch2-sec22-source.md`:

- Assumption `2.1`: `Delta(p(t))` has diagonal structure.
- Assumption `2.2` and eq. `(2.10)`: the scheduling set is contained in the closed unit `l_infinity` ball.
- Remark `2.3`: scaling and shifting can be used to normalize an arbitrary bounded scheduling set.
- Condition `2.4` and eq. `(2.11)`: `rho(D_zw) < 1`.
- Theorem `2.5`: under these assumptions, `I - D_zw Delta(p(t))` is nonsingular for all admissible `p(t)`.

This is the generic sufficient route Drenth provides.

### 2. Generalized step

One way to establish well-posedness is:

1. verify the required Delta structure,
2. normalize the admissible scheduling set if needed,
3. verify the spectral-radius bound on `D_zw`,
4. apply Theorem `2.5`.

This is a sufficient route, not the exact criterion itself.

### 3. Dual-gantry application

Applied to the dual-gantry realization:

- Assumption `2.1` is structurally compatible with the chosen block
  - `Delta(Y) = Y I_6`
  - because this is diagonal repeated-scalar structure.
- Assumption `2.2` can be discussed in two ways:
  - on the operational range `Y in [0.05, 0.75]`, the raw scheduling variable already satisfies `|Y| <= 1`,
  - for another bounded interval, Remark `2.3` says scaling and shifting could be used.
- The unresolved part is Condition `2.4`:
  - we would need to show `rho(D_zw) < 1` for the specific gantry `D_zw`.

So Section `2.2` gives a viable **generic sufficient route in principle**, but the route is not completed until the spectral-radius condition is actually checked.

### 4. Status

- `Direct from Drenth` for the sufficient theorem and its assumptions
- `Generalized from Drenth` for the workflow "check assumptions, then apply theorem"
- `Own dual-gantry derivation` for identifying how the gantry realization fits or does not yet fit those assumptions

### 5. What is not yet justified

This route does **not** yet justify:

- global well-posedness for all `Y in R`,
- application of Theorem `2.5` without verifying `rho(D_zw) < 1`,
- or the claim that Drenth's theorem is the sharpest route for this plant.

At best, it gives a bounded-range sufficient theorem once its assumptions are verified.

## Step 14. State the sharper plant-specific well-posedness route for the dual gantry

### 1. Exact Drenth support

From `docs/drenth/ch2-sec21-source.md`:

- the exact basic well-posedness condition is the nonsingularity of `I - D_zw Delta(p)`.

That exact condition is direct from Drenth.

The rest of this step is **not** from Drenth:

- it comes from the plant-specific algebra in `LPV/LFR-derivation.tex`,
- together with the positivity/invertibility proof in `LPV/M-invertibility.tex`.

### 2. Generalized step

Instead of applying only a generic sufficient theorem, attempt to use the plant structure to reduce the exact loop-solvability condition to a simpler and sharper condition that is specific to the model at hand.

### 3. Dual-gantry application

For the dual gantry, `LPV/LFR-derivation.tex` derives the loop equations:

- `v = M_0^(-1) [-K, -C] x - M_0^(-1) M_1 (Y v) - M_0^(-1) M_2 (Y v_1) + M_0^(-1) u`
- `v_1 = Y v`

Substituting `v_1 = Y v` gives:

- `(M_0 + Y M_1 + Y^2 M_2) v = f_gen`
- hence `M(Y) v = f_gen`

So the algebraic loop has a unique solution if and only if `M(Y)` is invertible.

Then `LPV/M-invertibility.tex` proves, using Sylvester's criterion, that under positive physical masses, inertias, and geometric parameters:

- `M(Y)` is positive definite for all `Y in R`,
- therefore `M(Y)` is invertible for all `Y in R`.

This yields a sharper plant-specific result:

- the dual-gantry LFR is well-posed for all real `Y`,
- not merely on a normalized bounded scheduling set.

### 4. Status

- `Direct from Drenth` for the fact that one must solve the exact algebraic loop
- `Own dual-gantry derivation` for the reduction to `M(Y) v = f_gen`
- `Own dual-gantry derivation` for the proof route through `LPV/M-invertibility.tex`

### 5. What is not yet justified

This route does **not** come from Drenth's Section `2.2`, so it should not be described as an application of Theorem `2.5`.

Also, if physical parameters are later made trainable, this proof continues to apply only if the optimization preserves the positivity assumptions used in `LPV/M-invertibility.tex`.

## Step 15. Compare the two well-posedness routes explicitly

### 1. Exact Drenth support

From `docs/drenth/ch2-sec22-source.md`:

- Theorem `2.5` is a sufficient result under extra assumptions.

From `docs/drenth/ch2-sec21-source.md`:

- the exact starting point is still the nonsingularity of `I - D_zw Delta(p)`.

### 2. Generalized step

When both a generic sufficient theorem and a plant-specific structural reduction are available, compare them rather than treating them as interchangeable.

### 3. Dual-gantry application

For the dual gantry:

- Drenth's Section `2.2` route is:
  - generic,
  - sufficient,
  - bounded-range in setup,
  - dependent on checking `rho(D_zw) < 1`,
  - useful as theory background and especially relevant for learnable LPV-LFR parameterizations.

- The gantry-specific route is:
  - exact for this plant,
  - structurally tied to the chosen `D_zw`,
  - reduced to the mechanical matrix `M(Y)`,
  - and global in `Y` once `LPV/M-invertibility.tex` is invoked.

So the honest conclusion is:

- Drenth Section `2.2` supplies a **generic sufficient theorem**,
- while the gantry derivation supplies the **sharper plant-specific proof** that is more informative for this baseline.

### 4. Status

- `Generalized from Drenth` for the comparison viewpoint
- `Own dual-gantry derivation` for concluding which route is sharper here

### 5. What is not yet justified

This comparison does **not** imply that Drenth's route is irrelevant.

It remains useful if:

- one wants a generic theorem stated in the verification document,
- one later introduces learnable `D_zw` parameterizations,
- or one wants a bounded-range sufficient condition aligned with Drenth's identification framework.

## Boundary Summary

### Directly supported by Drenth 2.1 / 2.1.1 / 2.2

- the CT LPV-LFR target class `(G, Delta(p))`
- the constant-plus-Delta structural view
- the repeated Delta-block structure
- the affine special case `D_zw = 0`
- the fact that the collapsed model can be rational
- the motivation for rational dependency through coupling preservation and reduced overbounding
- the use of repeated scheduling structure as a modeling device
- collapse of the latent loop as the main correctness check
- the exact basic loop-solvability condition
- the generic sufficient well-posedness route based on:
  - diagonal `Delta`,
  - bounded scheduling set,
  - `rho(D_zw) < 1`

### Generalized from Drenth 2.1 / 2.1.1 / 2.2

- using collapse as the central verification strategy when constructing a new realization
- choosing scheduling variables with coupling preservation as a design principle
- treating repeated copies of one physical scheduling variable as preferable to introducing unnecessary independent scheduling variables
- using the LPV-LFR form as a design target for a plant-specific reformulation
- treating Theorem `2.5` as one available well-posedness route rather than the only route

### Own dual-gantry derivation

- the target CT gantry model itself
- the decomposition `M(Y) = M_0 + Y M_1 + Y^2 M_2`
- the choice of latent variables `v, v_1, v_2`
- the choice `z = [v; v_1]`, `w = [v_1; v_2]`
- the specific Delta block `Delta(Y) = Y I_6`
- the explicit constant matrices `A, B_w, B_u, C_z, D_zw, D_zu, C_y, D_yw, D_yu`
- the reduction of the collapsed loop to `M(Y) v = f_gen`
- the sharper well-posedness proof via invertibility of `M(Y)`

## Short Conclusion

With Drenth Sections `2.1`, `2.1.1`, and `2.2` all explicitly available, the dual-gantry mapping can now be stated more cleanly:

- Sections `2.1` and `2.1.1` justify the **framework choice**, the **rational modeling choice**, and the **verification logic**.
- Section `2.2` adds a **generic sufficient well-posedness theorem**.
- The dual-gantry derivation then goes further and gives a **sharper plant-specific proof** by reducing the loop to `M(Y) v = f_gen` and invoking `LPV/M-invertibility.tex`.

So the verification note should preserve the following distinction:

- Drenth provides the **generic LPV-LFR framework** and a **generic sufficient well-posedness route**.
- The actual dual-gantry realization and the strongest well-posedness proof for this baseline remain **our own derivation**.

