# Drenth Thesis Chapter 2.1 Source-Only Extraction

## Purpose

This note is a **source-only extraction** of Section `2.1` of:

- `literature/books/drenth2025_lpv-lfr-thesis.pdf`

It is intentionally narrow:

- only Section `2.1` is considered,
- only eqs. `(2.1)`--`(2.4)` and the immediately following affine-special-case paragraph are used,
- no dual-gantry interpretation is included,
- no augmentation context is included,
- no material from Section `2.1.1` or `2.2` is used here.

The goal is to record, with maximum explicitness, what Section `2.1` itself does and does not justify.

## Section Boundary

This note covers exactly the following part of Chapter 2:

- the introductory paragraph under `2.1 | Introduction`,
- Drenth's definition of an LPV-LFR as a pair `(G, Delta(p))`,
- eq. `(2.1)`,
- eq. `(2.2)`,
- eq. `(2.3)`,
- eq. `(2.4)`,
- the short paragraph immediately after `(2.4)` up to the start of `2.1.1`.

## Reading Discipline

To keep the note auditable, statements are divided into three classes:

- `Direct from Section 2.1`
- `Safe inference from Section 2.1`
- `Not justified by Section 2.1 alone`

## Direct from Section 2.1

### 1. The Linear Fractional Representation is introduced as a general model representation

Drenth states that the Linear Fractional Representation:

- is a general model representation,
- can represent systems with a wide range of properties,
- originally gained popularity in robust control,
- is used in the LPV setting to incorporate parameter-varying behavior.

This is the high-level motivation of Section `2.1`.

### 2. For LPV systems, the LFR is described as an interconnection

Drenth states that LPV systems are represented by an interconnection between:

- a nominal system `G`,
- parameter-varying components represented by `Delta(p)`,

where `Delta(p)` depends linearly on the scheduling signal `p`.

This is a conceptual statement preceding the formal definition.

### 3. Drenth defines an LPV-LFR as a pair `(G, Delta(p))`

This is the formal object defined in Section `2.1`.

The definition is explicitly given in **continuous time**.

That matters because:

- the IFAC paper uses a discrete-time formulation,
- but Section `2.1` of the thesis is already written in CT,
- so no DT-to-CT adaptation is required to use this definition.

### 4. Equation `(2.1)` gives the standard CT LPV-LFR interconnection

Eq. `(2.1)` introduces the interconnection

- `x_dot(t)`
- `z(t)`
- `y(t)`
- `w(t) = Delta(p(t)) z(t)`

with constant block matrices:

- `A_x`
- `B_w`
- `B_u`
- `C_z`
- `D_zw`
- `D_zu`
- `C_y`
- `D_yw`
- `D_yu`

The important direct facts are:

- the interconnection is continuous-time,
- the matrix `G` is constant,
- the scheduling dependence appears through `Delta(p(t))`,
- `z(t)` and `w(t)` are latent variables.

### 5. Section `2.1` explicitly gives signal meanings

Right after eq. `(2.1)`, Drenth explicitly defines:

- `u(t) in U subset R^{n_u}` as the input signal,
- `x(t) in X subset R^{n_x}` as the state,
- `w(t), z(t) in R^{n_w}` as the latent variables,
- `y(t) in Y subset R^{n_y}` as the output,
- `p(t) in P subset R^{n_p}` as the scheduling variable,
- `t in R` as continuous time.

He also states that the matrices of `G` are real matrices of appropriate dimensions.

### 6. Equation `(2.2)` defines the Delta-block structure

Eq. `(2.2)` defines `Delta(p(t))` as a block-diagonal matrix with repeated scheduling variables:

- `p_1(t) I_{eta_1}`
- ...
- `p_{n_p}(t) I_{eta_{n_p}}`

Drenth explicitly says:

- `eta` is a vector of integers,
- each entry indicates the number of repetitions per scheduling variable,
- the identity blocks are sized to realize the correct total dimension of `Delta(p(t))`.

### 7. Equation `(2.3)` states equivalence to an LPV-SS model with rational dependency

Drenth explicitly states that the interconnection is equivalent to an LPV state-space representation

- `x_dot = A(p(t)) x + B(p(t)) u`
- `y = C(p(t)) x + D(p(t)) u`

and explicitly states that the resulting dependency on `p(t)` is **rational**.

This is one of the most important claims in Section `2.1`.

### 8. Equation `(2.4)` gives the elimination formula for the latent variables

Drenth writes that the rational LPV-SS form is obtained by eliminating the latent variables `w(t)` and `z(t)` in `(2.1)` by substitution of

- `z(t) = (I - D_zw Delta(p(t)))^{-1} (C_z x(t) + D_zu u(t))`

This is the explicit algebraic-collapse formula given in Section `2.1`.

### 9. Section `2.1` explicitly states the basic well-posedness requirement

Immediately after `(2.4)`, Drenth states that well-posedness requires:

- `I - D_zw Delta(p(t))` to be nonsingular for all values of `p(t)`.

This is the generic algebraic-loop solvability condition arising from the collapse formula.

### 10. Section `2.1` explicitly positions rational LPV-LFR as a generalization of affine LPV-SS

Drenth states that because the representation admits rational LPV-SS models, it is a generalization of commonly used affine LPV-SS models.

This is a direct conceptual claim of Section `2.1`.

### 11. Section `2.1` explicitly identifies affine dependency as a special case

Drenth states that affine-dependency models are represented by taking `(2.1)` with:

- `D_zw = 0`

He also states that this affine special case always satisfies the well-posedness condition.

### 12. Section `2.1` explicitly says affine LPV-SS can be converted to LPV-LFR by SVD methods

Drenth states that, given an affine LPV-SS model, an equivalent LPV-LFR realization can easily be recovered using SVD methods.

This is the only constructive remark in Section `2.1`, and it is explicitly attached to the **affine** case.

## Safe Inferences from Section 2.1

These are not written as standalone theorems in Section `2.1`, but they are immediate and low-risk consequences of the stated equations and text.

### 1. All scheduling dependence is intended to be isolated in `Delta(p)`

Why this is a safe inference:

- `G` is presented as the nominal interconnection matrix,
- the matrices inside `G` are described as constant real matrices,
- the scheduling dependence enters through `w = Delta(p) z`,
- the collapsed matrices `A(p), B(p), C(p), D(p)` are then induced by this loop.

So it is safe to say that the representation is organized so that parameter dependence is isolated in `Delta(p)`.

### 2. The dimensions of `z` and `w` must match the size of `Delta(p)`

Why this is a safe inference:

- Drenth states `w(t), z(t) in R^{n_w}`,
- and `Delta(p(t)) : P -> R^{n_w x n_w}`.

Therefore:

- `z` and `w` must have the same dimension,
- and that dimension must match the Delta-block size.

### 3. The collapsed state-space dependence can be non-affine even though `Delta(p)` is linear in `p`

Why this is a safe inference:

- `Delta(p)` itself is block-diagonal and linear in `p`,
- but `(2.4)` contains the inverse term `(I - D_zw Delta(p))^{-1}`,
- so after elimination the dependence becomes rational.

This is fully consistent with Drenth's explicit claim that `(2.3)` admits rational dependency.

### 4. The affine special case corresponds to eliminating the feedback term through `D_zw = 0`

Why this is a safe inference:

- Drenth states affine dependency is obtained by taking `D_zw = 0`,
- and `(2.4)` then simplifies because the inverse term becomes `I^{-1}`.

So it is safe to interpret `D_zw` as the structural ingredient that enables the genuinely rational loop effect.

## Not Justified by Section 2.1 Alone

This section is important because many tempting statements are **not** actually established by Section `2.1` itself.

### 1. Section `2.1` does not give a general realization algorithm for arbitrary CT plants

Section `2.1` defines the framework and states equivalence after elimination, but it does **not** explain:

- how to choose the scheduling variables for an arbitrary plant,
- how to choose latent variables for an arbitrary plant,
- how to construct `G` for an arbitrary rational CT model,
- how to find a minimal realization.

### 2. Section `2.1` does not give a symbolic recipe for selecting `z` and `w`

It tells us what role `z` and `w` play, but it does not say how to invent them for a new mechanical model.

That step belongs to the plant-specific realization work, not to Section `2.1` itself.

### 3. Section `2.1` does not prove well-posedness

It states the well-posedness condition, but it does **not** yet provide sufficient conditions or parameterizations to guarantee it.

That comes only in Section `2.2`.

### 4. Section `2.1` does not claim every rational CT LPV model has an easy constructive LPV-LFR realization

It states equivalence of the interconnection to a rational LPV-SS model after collapse.

That is not the same as claiming:

- every rational LPV-SS description comes with an obvious or easy construction of `G` and `Delta(p)`,
- every such realization is unique,
- every such realization is minimal.

### 5. Section `2.1` does not yet discuss learning, scheduling-map parameterization, or augmentation

Those topics appear later in the thesis. They should not be imported back into the interpretation of Section `2.1`.

## Equation-by-Equation Audit

### Equation `(2.1)`

What it is:

- the defining CT LPV-LFR interconnection.

What it establishes directly:

- the model is written in continuous time,
- `G` is partitioned into state, latent-loop, and output blocks,
- `z` and `w` are internal latent variables,
- the latent variables are connected through `w = Delta(p) z`.

What it does **not** establish by itself:

- how to compute `G`,
- what physical meaning `z` and `w` should have in a new plant,
- whether the interconnection is well-posed.

### Equation `(2.2)`

What it is:

- the structural definition of the Delta block.

What it establishes directly:

- Delta is block diagonal,
- each scheduling variable enters linearly,
- repeated occurrences are handled through identity repetitions.

What it does **not** establish by itself:

- how many repetitions are needed,
- how many scheduling variables are best,
- whether the chosen Delta structure is minimal.

### Equation `(2.3)`

What it is:

- the collapsed LPV state-space representation.

What it establishes directly:

- the interconnection admits an LPV-SS representation,
- the dependency is rational in the scheduling variable.

What it does **not** establish by itself:

- explicit formulas for `A(p), B(p), C(p), D(p)` in terms of the blocks of `G`,
- how difficult it is to recover `G` from a given LPV-SS model.

### Equation `(2.4)`

What it is:

- the explicit elimination formula for `z(t)`.

What it establishes directly:

- the rational dependence comes from solving the algebraic loop,
- well-posedness depends on invertibility of `I - D_zw Delta(p(t))`.

What it does **not** establish by itself:

- a sufficient well-posedness condition,
- a constructive parameterization of `D_zw`,
- an application-specific simplification of the loop.

## Immediate Reusable Takeaways from Section 2.1

If we restrict ourselves strictly to what Section `2.1` supports, then the following reusable takeaways are safe:

1. We may use the CT LPV-LFR pair `(G, Delta(p))` as the target representation class.
2. We may use the block-structured CT interconnection of eq. `(2.1)`.
3. We may use a repeated block-diagonal Delta structure as in eq. `(2.2)`.
4. We may treat `z` and `w` as latent internal loop variables.
5. We may use collapse of the latent loop as the main exactness check.
6. We may state that the collapsed representation admits rational dependency.
7. We may treat affine dependency as the special case `D_zw = 0`.

## What Must Be Deferred to Later Sections or to Our Own Derivation

Section `2.1` alone is **not enough** for:

1. the affine-vs-rational overbounding argument in full detail
   - this belongs to Section `2.1.1`
2. a sufficient well-posedness theorem
   - this belongs to Section `2.2`
3. any learnable well-posed parameterization of `D_zw`
   - this belongs to Section `2.2`
4. choosing plant-specific latent variables
   - this belongs to the application-specific derivation
5. converting a rational mechanical model into a concrete LPV-LFR realization
   - this also belongs to the application-specific derivation

## Short Conclusion

Section `2.1` of Drenth's thesis gives:

- the **continuous-time LPV-LFR framework**,
- the **standard latent-loop interconnection**,
- the **Delta-block structure**,
- the fact that collapsing the loop yields a **rational LPV-SS model**,
- and the **basic algebraic-loop solvability condition**.

Section `2.1` does **not** give:

- a general constructive recipe for arbitrary plants,
- a plant-specific realization algorithm,
- or a well-posedness proof beyond the generic nonsingularity requirement.

So if we use Section `2.1` later in a verification document, the honest phrasing is:

- it justifies the **framework**,
- but not yet the **specific realization steps** for a concrete plant.
