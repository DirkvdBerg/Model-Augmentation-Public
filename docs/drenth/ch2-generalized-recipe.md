# Generalized CT LPV-LFR Recipe from Drenth Sections 2.1 and 2.1.1 Only

## Purpose

This note gives a **generalized CT LPV-LFR recipe** using **only** the following source-only notes:

- `docs/drenth/ch2-sec21-source.md`
- `docs/drenth/ch2-sec211-source.md`

This means the recipe is intentionally restricted:

- it uses Section `2.1` for the generic CT LPV-LFR framework,
- it uses Section `2.1.1` for the affine-vs-rational trade-off,
- it does **not** use Section `2.2`,
- it does **not** use any identification material from later parts of Chapter 2,
- it does **not** yet include any dual-gantry application.

The point of this note is to answer:

- what generalized modeling procedure is reasonably supported by Drenth `2.1` and `2.1.1`,
- and where that support stops.

## Reading Discipline

Each step is labeled as one of:

- `Directly supported by 2.1 / 2.1.1`
- `Generalized from 2.1 / 2.1.1`
- `Requires material beyond 2.1 / 2.1.1`

The purpose of the labels is to keep the recipe honest. This note is **not** allowed to use:

- Section `2.2` well-posedness theorems or parameterizations,
- any dual-gantry-specific algebra,
- any constructive claims not justified by the two source-only notes.

## What This Recipe Is Allowed to Assume

From `docs/drenth/ch2-sec21-source.md`, the following are available:

- the CT LPV-LFR object `(G, Delta(p))`,
- the interconnection in eq. `(2.1)`,
- the repeated Delta-block structure in eq. `(2.2)`,
- the collapsed rational LPV-SS form in eq. `(2.3)`,
- the latent-variable elimination formula in eq. `(2.4)`,
- the basic nonsingularity requirement,
- the affine special case `D_zw = 0`.

From `docs/drenth/ch2-sec211-source.md`, the following are available:

- rational dependency can reduce overbounding relative to affine dependency,
- overbounding can be caused by loss of coupling between scheduling variables,
- repeated use of one scheduling variable in `Delta` can encode richer dependency,
- richer dependency may be obtained without increasing the number of independent scheduling variables,
- the affine-vs-rational trade-off is structural, not merely numerical.

## Generalized Recipe

### Step 1. Fix the target representation class as a CT LPV-LFR

**Support status**

- `Directly supported by 2.1`

**Why this step is justified**

Section `2.1` explicitly defines an LPV-LFR as a continuous-time pair `(G, Delta(p))` with the standard interconnection in eq. `(2.1)`.

**What the generalized step is**

If you want to reformulate a continuous-time parameter-dependent model in Drenth's framework, the first step is to adopt the LPV-LFR pair `(G, Delta(p))` as the target model class.

**What this step does not yet give**

- it does not tell you how to choose `G`,
- it does not tell you how to choose the latent variables,
- it does not tell you how to choose the scheduling variables.

### Step 2. Treat the model as a constant interconnection plus a scheduling block

**Support status**

- `Directly supported by 2.1`

**Why this step is justified**

Section `2.1` explicitly organizes the LPV-LFR as:

- a constant nominal interconnection `G`,
- together with a scheduling block `Delta(p)`.

It is also a safe inference from Section `2.1` that scheduling dependence is intended to be isolated in `Delta(p)`.

**What the generalized step is**

Reformulation should be organized so that:

- all constant linear structure lives in `G`,
- all parameter dependence is pushed into `Delta(p)`.

**What this step does not yet give**

- it does not explain how to move an arbitrary plant into that structure,
- it only defines the structural target.

### Step 3. State the collapsed CT model that the LPV-LFR must reproduce

**Support status**

- `Generalized from 2.1`

**Why this step is justified**

Section `2.1` says that the LPV-LFR interconnection is equivalent to a collapsed LPV-SS model after eliminating the latent variables via eq. `(2.4)`.

That does not explicitly say "start from the target CT model and work backwards," but it strongly supports treating exact recovery of the collapsed CT model as the central correctness criterion.

**What the generalized step is**

Before constructing the LPV-LFR, write down the continuous-time model that must be recovered after collapsing the latent loop.

**What this step does not yet give**

- it does not tell you how to derive the realization,
- but it does tell you what the realization must be verified against.

### Step 4. Decide whether the target should be affine or rational

**Support status**

- `Directly supported by 2.1 / 2.1.1`

**Why this step is justified**

Section `2.1` explicitly identifies:

- affine dependency as the special case `D_zw = 0`,
- rational dependency as the more general class admitted by the collapsed LPV-LFR.

Section `2.1.1` explicitly motivates why rational dependency may be preferable:

- reduced overbounding,
- reduced conservatism,
- fewer independent scheduling variables in some cases.

**What the generalized step is**

Before constructing the realization, decide whether:

- the affine special case is sufficient,
- or the model should retain genuinely rational dependency.

**What this step does not yet give**

- it does not prove that rational dependency is always better,
- it only justifies treating this as a real modeling choice.

### Step 5. Choose scheduling variables with coupling loss in mind

**Support status**

- `Directly supported by 2.1.1`

**Why this step is justified**

Section `2.1.1` explicitly states that overbounding can arise when coupling between scheduling variables is discarded and independent scheduling freedom is introduced.

**What the generalized step is**

Choose scheduling variables in a way that avoids unnecessary loss of coupling between quantities that are not truly independent in the original model.

**What this step does not yet give**

- it does not provide a universal algorithm for choosing the best scheduling variables,
- but it does justify using preservation of coupling as a design criterion.

### Step 6. Use repeated Delta-block structure to express richer dependency without introducing more independent scheduling variables

**Support status**

- `Directly supported by 2.1`
- `Safely reinforced by 2.1.1`

**Why this step is justified**

Section `2.1` gives the repeated block-diagonal structure of `Delta(p)` in eq. `(2.2)`.
Section `2.1.1` shows, through the MSD example, that repeated use of the same scheduling variable can be used instead of introducing additional independent scheduling variables.

**What the generalized step is**

If richer dependency must be represented, consider using repeated copies of the same scheduling variable inside `Delta(p)` rather than automatically introducing new independent scheduling variables.

**What this step does not yet give**

- it does not prove the repetition count is unique or minimal,
- it does not give an algorithm for choosing the repetition count.

### Step 7. Introduce latent loop variables so the scheduling action is expressed through `w = Delta(p) z`

**Support status**

- `Directly supported by 2.1` for the existence of latent variables
- `Generalized from 2.1 / 2.1.1` for using them as a realization strategy

**Why this step is justified**

Section `2.1` explicitly uses latent variables `z` and `w` in the LPV-LFR interconnection.
Section `2.1.1` shows, via the MSD example, that the model structure may use repeated scheduling action in the Delta loop.

Together, these support the generalized view that latent variables are the internal mechanism by which scheduling dependence is routed through the Delta block.

**What the generalized step is**

Introduce internal latent variables so that the model's parameter dependence can be expressed through the relation

- `w = Delta(p) z`.

**What this step does not yet give**

- Section `2.1` does not provide a symbolic recipe for inventing `z` and `w` for a new plant,
- so the concrete latent-variable choice remains outside the support of `2.1` and `2.1.1`.

### Step 8. Rewrite the model so that all scheduling dependence is pushed into the Delta loop

**Support status**

- `Generalized from 2.1`

**Why this step is justified**

Section `2.1` presents the LPV-LFR as a constant interconnection `G` combined with `Delta(p)`.
It is therefore a safe and necessary generalization that a successful reformulation must move the parameter dependence into the Delta loop.

**What the generalized step is**

Re-express the model so that:

- the interconnection matrices are constant,
- and the scheduling dependence appears through the latent loop only.

**What this step does not yet give**

- Section `2.1` does not tell you how to do this algebraically for an arbitrary plant,
- so this is a design objective, not a constructive method.

### Step 9. Read off the constant block matrix `G`

**Support status**

- `Directly supported by 2.1` for the block structure
- `Generalized from 2.1` for the workflow "rewrite, then read off"

**Why this step is justified**

Eq. `(2.1)` gives the exact constant block partition that the reformulated model must match.

**What the generalized step is**

Once the equations have been rewritten in LPV-LFR form, identify the constant blocks:

- state equation blocks,
- latent-loop blocks,
- output equation blocks.

**What this step does not yet give**

- it does not provide formulas for those blocks in a new plant,
- only the structural slots they must occupy.

### Step 10. Collapse the latent loop and verify exact recovery of the target CT model

**Support status**

- `Directly supported by 2.1`

**Why this step is justified**

Section `2.1` explicitly gives the collapse formula in eq. `(2.4)` and states that the resulting LPV-SS model is obtained by eliminating the latent variables.

**What the generalized step is**

Use the elimination of `z` and `w` as the main verification step:

- collapse the loop,
- recover the continuous-time state-space form,
- check that it matches the target model from Step 3.

**What this step does not yet give**

- it does not guarantee the realization exists,
- it only gives the correct verification criterion once a candidate realization is proposed.

### Step 11. Identify whether the collapsed result is affine or rational

**Support status**

- `Directly supported by 2.1`

**Why this step is justified**

Section `2.1` explicitly distinguishes:

- the affine special case `D_zw = 0`,
- the more general rational case obtained through the latent-loop elimination.

**What the generalized step is**

After collapsing the loop, classify the resulting dependency:

- affine, if it falls into the `D_zw = 0` special case,
- rational otherwise.

**What this step does not yet give**

- it does not quantify which choice is better,
- that motivation comes from Section `2.1.1`.

### Step 12. Check the basic loop-solvability condition

**Support status**

- `Directly supported by 2.1`

**Why this step is justified**

Immediately after eq. `(2.4)`, Section `2.1` states that the interconnection is well-posed if `I - D_zw Delta(p)` is nonsingular for all admissible scheduling values.

**What the generalized step is**

Any candidate LPV-LFR realization must be checked against the basic nonsingularity condition of the algebraic loop.

**What this step does not yet give**

- no sufficient theorem,
- no scaling assumptions,
- no parameterization of `D_zw`,
- no constructive guarantee.

Those all require material beyond `2.1` and `2.1.1`.

## What This Recipe Can Legitimately Claim

Using only Sections `2.1` and `2.1.1`, the following generalized claims are defensible:

1. A CT LPV-LFR should be viewed as a constant interconnection `G` plus a structured Delta block.
2. The collapsed LPV-LFR can admit rational scheduling dependence.
3. Affine dependency is a special case, not the only possible target.
4. Choosing scheduling variables poorly can introduce overbounding by discarding coupling.
5. Repeated use of the same scheduling variable in `Delta` can be structurally preferable to introducing more independent scheduling variables.
6. The correctness of a candidate realization should be checked by collapsing the loop and recovering the target CT model.

## What This Recipe Cannot Legitimately Claim Yet

Using only Sections `2.1` and `2.1.1`, the following are **not** yet justified:

1. a sufficient well-posedness theorem,
2. a generic algorithm for choosing latent variables,
3. a generic algorithm for converting an arbitrary rational CT plant into LPV-LFR form,
4. a method for enforcing well-posedness during optimization,
5. a minimality result,
6. a proof that rational dependency is always the best choice.

## Short Conclusion

Sections `2.1` and `2.1.1` together support a **structural modeling recipe**, not a full construction algorithm.

They justify:

- the CT LPV-LFR target class,
- the role of the Delta-block,
- the use of latent loop variables,
- the affine-versus-rational modeling choice,
- the importance of preserving coupling in scheduling design,
- and the use of loop collapse as the main correctness check.

They do **not** yet justify:

- how to build the realization for a specific plant,
- or how to guarantee well-posedness beyond the basic nonsingularity requirement.

That is exactly where later source material or plant-specific derivation must begin.

