# Drenth Thesis Chapter 2.2 Source-Only Extraction

## Purpose

This note is a **source-only extraction** of Section `2.2` of:

- `literature/books/drenth2025_lpv-lfr-thesis.pdf`

It follows the same discipline as:

- `docs/drenth/ch2-sec21-source.md`
- `docs/drenth/ch2-sec211-source.md`

This note is intentionally narrow:

- only Section `2.2` is considered,
- no dual-gantry interpretation is included,
- no identification application beyond what Section `2.2` itself states is imported,
- no augmentation context is included.

The goal is to record, with maximum explicitness, what Section `2.2` itself does and does not justify about well-posed LPV-LFR models.

## Section Boundary

This note covers exactly the following material from Chapter 2:

- the subsection `2.2 | Well-posed LPV-LFR models`,
- Assumption `2.1`,
- Assumption `2.2`,
- Remark `2.3`,
- Condition `2.4`,
- Theorem `2.5` and its proof sketch as written in the text,
- eqs. `(2.10)`--`(2.13)`,
- the explanatory paragraphs around the matrix-exponential parameterization and its overparameterization discussion,
- stopping at the start of Section `2.3`.

## Reading Discipline

To keep the note auditable, statements are divided into three classes:

- `Direct from Section 2.2`
- `Safe inference from Section 2.2`
- `Not justified by Section 2.2 alone`

## Direct from Section 2.2

### 1. Section `2.2` starts from the generic well-posedness condition introduced earlier

Drenth explicitly begins Section `2.2` by recalling that, from eq. `(2.4)`, an LPV-LFR is well-posed if:

- `I - D_zw Delta(p(t))` is nonsingular,

equivalently,

- `det(I - D_zw Delta(p(t))) != 0`

for all possible realizations of the scheduling variables.

So Section `2.2` is explicitly positioned as a development of the basic loop-solvability condition already introduced in Section `2.1`.

### 2. Drenth explicitly states that Section `2.2` proposes a method for guaranteeing well-posedness

Drenth writes that he proposes a novel method for guaranteeing well-posedness of LPV-LFR models.

So the subsection is not merely descriptive; it is intended to provide a constructive sufficient route.

### 3. Assumption `2.1` requires the Delta-block to have diagonal structure

Drenth explicitly states:

- Assumption `2.1`: `Delta(p(t))` has diagonal structure.

This is one of the foundational structural assumptions used in the subsequent theorem.

### 4. Assumption `2.2` bounds the scheduling set in the `l_infinity` unit ball

Drenth explicitly states:

- Assumption `2.2`: the scheduling set `P` is contained in the closed `n_p`-dimensional `l_infinity` unit ball.

This is formalized in eq. `(2.10)`.

### 5. Remark `2.3` explicitly says Assumption `2.2` is not restrictive

Drenth states that Assumption `2.2` is not restrictive because, under Assumption `2.1`, an arbitrary well-posed LFR `{G, Delta(p)}` can be transformed to an equivalent LFR satisfying the bounded-scheduling assumption by scaling and shifting the scheduling variables.

This is an explicit statement in the subsection.

### 6. Condition `2.4` imposes a spectral-radius bound on `D_zw`

Drenth explicitly states:

- Condition `2.4`: `rho(D_zw) < 1`

This is formalized in eq. `(2.11)`.

### 7. Theorem `2.5` gives a sufficient well-posedness result

Drenth explicitly states:

- if Assumptions `2.1`, `2.2`, and Condition `2.4` hold,
- then the LPV-LFR model `(2.1)` is well-posed,
- i.e. `I - D_zw Delta(p(t))` is nonsingular for all `p(t) in P`.

So Theorem `2.5` is a sufficient well-posedness theorem.

### 8. Drenth explicitly gives the proof logic behind Theorem `2.5`

The proof sketch in the text uses:

- `rho(Delta(p(t))) <= 1` from Assumption `2.2` and the diagonal structure of Assumption `2.1`,
- `rho(D_zw Delta(p(t))) <= rho(D_zw) rho(Delta(p(t)))`,
- and then concludes positivity and nonsingularity of `I - D_zw Delta(p(t))`.

Whether one would want to inspect the proof in more detail is a separate matter, but the proof idea is explicitly part of the subsection.

### 9. Equation `(2.12)` defines a matrix-exponential parameterization of `D_zw`

Drenth explicitly introduces

- `D_zw = e^{-N}`, with `N â‰» 0`

as a way to ensure Condition `2.4`.

This is one of the central constructive moves of Section `2.2`.

### 10. Drenth explicitly states the role of `(2.12)`

Drenth states that:

- the matrix exponential maps negative definite matrices to matrices satisfying `D_zw â‰º I`,
- connecting the well-posedness condition to positive-definiteness of `N` yields a simpler condition,
- and `N` need not be symmetric.

This is the interpretation attached to eq. `(2.12)`.

### 11. Equation `(2.13)` introduces a direct parameterization of `N`

Drenth explicitly defines

- `N = Psi (D_A^T D_A + D_B - D_B^T + epsilon I)`

with

- `Psi = diag(e^{D_d})`

and free variables:

- `D_A`,
- `D_B`,
- `D_d`,

plus a fixed `epsilon > 0`.

This is the second central constructive ingredient of Section `2.2`.

### 12. Drenth explicitly states why `(2.13)` guarantees `N â‰» 0`

Drenth explains that `(2.13)` combines:

- a symmetric positive-semi-definite term,
- a skew-symmetric term,
- a strictly positive-definite term,
- and positive scaling through `Psi`,

so that `N` is guaranteed to be strictly positive-definite by construction.

This is explicitly stated in the explanatory paragraph after `(2.13)`.

### 13. Drenth explicitly states the optimization motivation of `(2.13)`

Drenth states that, in identification, parameterizing `N` according to `(2.13)` enables unconstrained optimization without risk of obtaining ill-posed models.

This is a direct motivation given in Section `2.2`.

### 14. Drenth explicitly discusses overparameterization

Drenth states that replacing a direct parameterization of `D_zw` with the transformed free variables in `(2.13)` increases the number of parameters.

He then explicitly notes that this overparameterization can be reduced by populating only:

- lower triangular elements of `D_A`,
- strictly upper triangular elements of `D_B`.

So Section `2.2` directly acknowledges the cost of the well-posed parameterization.

## Safe Inferences from Section 2.2

These are not stated as standalone theorems beyond the subsection, but they are immediate and low-risk consequences of what Drenth explicitly writes.

### 1. Theorem `2.5` is a sufficient result, not an "if and only if" characterization

Why this is a safe inference:

- the theorem is presented as a condition under which well-posedness holds,
- not as a characterization of all well-posed LPV-LFR models.

So it is safe to read Section `2.2` as giving a sufficient route, not a necessary-and-sufficient one.

### 2. The purpose of Assumption `2.2` is normalization of the scheduling range

Why this is a safe inference:

- Drenth explicitly says the assumption is not restrictive,
- and says equivalent scaling and shifting can enforce it.

So it is safe to interpret Assumption `2.2` as a normalization condition used to make the theorem convenient.

### 3. The matrix-exponential parameterization is designed for learnable models, not just theoretical existence

Why this is a safe inference:

- Drenth explicitly motivates `(2.13)` in terms of unconstrained optimization,
- and discusses parameter counts and implementation choices.

So it is safe to infer that Section `2.2` is strongly identification-oriented even though it begins with a generic well-posedness theorem.

### 4. Section `2.2` moves from a generic solvability condition to a parameterization-friendly sufficient condition

Why this is a safe inference:

- the subsection starts from the exact nonsingularity requirement,
- then adds assumptions and a spectral-radius condition,
- and finally introduces parameterizations that guarantee those conditions.

So it is safe to summarize the flow of Section `2.2` in those three layers.

## Not Justified by Section 2.2 Alone

This part is especially important, because Section `2.2` is easy to overuse.

### 1. Section `2.2` does not prove that every well-posed LPV-LFR must satisfy Condition `2.4`

Theorem `2.5` is presented as a sufficient condition.

So Section `2.2` does **not** prove that all well-posed LPV-LFR models must have:

- diagonal `Delta`,
- bounded scheduling in the unit `l_infinity` ball,
- `rho(D_zw) < 1`.

### 2. Section `2.2` does not prove that the parameterization `(2.12)`--`(2.13)` is the only useful one

Drenth proposes this parameterization, but the subsection does **not** prove uniqueness or optimality of this choice.

### 3. Section `2.2` does not yet prove anything about a specific physical plant

Everything in Section `2.2` is still generic:

- no plant-specific matrix structure is used,
- no mechanical interpretation is used,
- no special reduction of the loop is used.

So Section `2.2` alone does **not** yield a plant-specific well-posedness proof.

### 4. Section `2.2` does not prove minimality or numerical superiority of the parameterized realization

It discusses feasibility of optimization and acknowledges overparameterization, but it does not prove:

- minimality,
- best numerical conditioning,
- or best computational efficiency in all settings.

### 5. Section `2.2` does not replace the exact basic condition from Section `2.1`

The subsection gives a sufficient route, but it does **not** invalidate or replace the original exact condition:

- `I - D_zw Delta(p)` nonsingular.

So one should not confuse the sufficient theorem with the underlying exact criterion.

### 6. Section `2.2` does not by itself justify importing the theorem into a plant-specific derivation without checking assumptions

If one wants to use Theorem `2.5` for a concrete plant, one must still verify:

- diagonal structure of `Delta`,
- normalized scheduling range,
- spectral-radius condition on `D_zw`.

Section `2.2` does not remove the need to check those assumptions in the concrete case.

## Equation-by-Equation Audit

### Equation `(2.10)`

What it is:

- the bounded-scheduling assumption.

What it establishes directly:

- the scheduling trajectory lies in the unit `l_infinity` ball.

What it does **not** establish by itself:

- that every LPV-LFR naturally satisfies this,
- or that this assumption is necessary for well-posedness.

### Equation `(2.11)`

What it is:

- the spectral-radius condition `rho(D_zw) < 1`.

What it establishes directly:

- the specific sufficient bound Drenth uses in Theorem `2.5`.

What it does **not** establish by itself:

- that this bound is necessary,
- or that every well-posed model must satisfy it.

### Equation `(2.12)`

What it is:

- the matrix-exponential parameterization `D_zw = e^{-N}` with `N â‰» 0`.

What it establishes directly:

- a constructive way to enforce the spectral-radius-based sufficient condition through positivity of `N`.

What it does **not** establish by itself:

- how to parameterize `N`,
- how many free variables are needed,
- or that this is the most efficient parameterization.

### Equation `(2.13)`

What it is:

- Drenth's direct free-variable parameterization of `N`.

What it establishes directly:

- a specific constructive map from unconstrained variables to a positive-definite `N`.

What it does **not** establish by itself:

- that this map is unique,
- that it is minimal,
- or that it is numerically optimal.

## Immediate Reusable Takeaways from Section 2.2

If we restrict ourselves strictly to what Section `2.2` supports, then the following takeaways are safe:

1. The exact basic well-posedness condition from Section `2.1` can be strengthened by a sufficient theorem under additional assumptions.
2. A diagonal Delta structure is central to Drenth's sufficient route.
3. Bounding the scheduling range is part of the theorem setup and can be enforced by scaling/shifting.
4. The spectral radius of `D_zw` is the key matrix quantity in Drenth's sufficient condition.
5. A learnable `D_zw` can be parameterized through a positive-definite auxiliary matrix `N`.
6. Drenth's parameterization is designed for unconstrained optimization of rational LPV-LFR models.

## What Must Be Deferred Beyond Section 2.2

Section `2.2` alone is **not enough** for:

1. the generic CT LPV-LFR framework
   - this comes from Section `2.1`
2. the affine-vs-rational overbounding trade-off
   - this comes from Section `2.1.1`
3. any plant-specific realization strategy
   - this belongs to the application-specific derivation
4. any claim that the theorem is sharper than a plant-specific well-posedness proof
   - that requires comparison with the concrete application

## Short Conclusion

Section `2.2` of Drenth's thesis gives:

- a generic sufficient well-posedness route for LPV-LFR models,
- normalization assumptions on the scheduling set,
- a spectral-radius condition on `D_zw`,
- a theorem guaranteeing nonsingularity of `I - D_zw Delta(p)`,
- and a constructive parameterization of `D_zw` suited for unconstrained optimization.

Section `2.2` does **not** give:

- a necessary-and-sufficient characterization of well-posedness,
- a plant-specific proof,
- or a realization algorithm for a concrete mechanical model.

So if we use Section `2.2` later in a verification document, the honest phrasing is:

- it supplies a **generic sufficient theorem and optimization-oriented parameterization**,
- but not yet the **sharpest proof** for a specific plant.

