# Drenth Thesis Chapter 2.1.1 Source-Only Extraction

## Purpose

This note is a **source-only extraction** of Section `2.1.1` of:

- `literature/books/drenth2025_lpv-lfr-thesis.pdf`

It follows the same discipline as `docs/drenth/ch2-sec21-source.md`:

- only Section `2.1.1` is considered,
- no dual-gantry interpretation is included,
- no material from Section `2.1` beyond what is explicitly used inside `2.1.1` is imported,
- no material from Section `2.2` or later chapters is used here.

The goal is to record, with maximum explicitness, what Section `2.1.1` itself does and does not justify about the trade-off between affine and rational dependency models.

## Section Boundary

This note covers exactly the following material from Chapter 2:

- the subsection `2.1.1 | The trade-off between affine and rational dependency models`,
- the explanatory paragraphs before the MSD example,
- the nonlinear MSD example and its two LPV embeddings,
- eqs. `(2.5)`--`(2.9)`,
- the paragraph interpreting the scheduling sets and the resulting LTI families,
- the comparison statement immediately before Section `2.2`.

## Reading Discipline

To keep the note auditable, statements are divided into three classes:

- `Direct from Section 2.1.1`
- `Safe inference from Section 2.1.1`
- `Not justified by Section 2.1.1 alone`

## Direct from Section 2.1.1

### 1. Drenth explicitly frames the issue as a trade-off between affine and rational dependency

The subsection is introduced as an explanation of the trade-off made when constructing LPV models using:

- affine dependency, versus
- rational dependency.

So the stated purpose of the subsection is comparative, not merely descriptive.

### 2. Drenth explicitly claims that rational dependency can reduce overbounding

At the start of the subsection, Drenth states that rational dependency LPV-LFR models can reduce:

- overbounding relative to affine dependency models,
- and therefore conservatism in controllers.

This is a direct motivating claim of the subsection.

### 3. Drenth explicitly explains where the overbounding comes from

Drenth states that in the LPV embedding process, specifically the last step where the scheduling variable is disconnected from the nonlinear system and treated as free, knowledge of the scheduling map is discarded except for its range.

He then states more specifically that:

- relationships between scheduling variables,
- and inherent coupling between scheduling variables,

are discarded.

This is the core conceptual mechanism emphasized in the subsection.

### 4. Drenth gives a specific example of lost coupling

Drenth explicitly says that if multiple scheduling variables depend on the same state, then treating them independently can lead to combinations of scheduling variables that could not occur in the original nonlinear system.

This is the first concrete statement of the overbounding mechanism.

### 5. Drenth explicitly states the role of rational dependency

Drenth writes that by allowing rational dependency structures, instead of capturing coupling only in the scheduling map, the coupling can be captured in the dependency on the scheduling signal itself.

He further states that this coupling is then not ignored during controller synthesis.

This is the central conceptual argument for rational dependency in Section `2.1.1`.

### 6. Drenth explicitly states a second claimed benefit of rational dependency

Drenth states that moving complexity from the scheduling map to the dependency structure can also reduce the required number of scheduling variables, improving usability in controller synthesis.

This is a separate direct claim from the overbounding/conservatism claim.

### 7. Drenth introduces an explicit nonlinear MSD example

Drenth then illustrates the trade-off using a nonlinear Mass-Spring-Damper system.

The model is described as:

- a lumped mass `m`,
- connected to the fixed world by a nonlinear spring `k(x)`,
- and a nonlinear damper `d(x)`,
- actuated by an input force `u`,
- with displacement `x` as output.

### 8. Equation `(2.5)` gives the nonlinear MSD dynamics

Eq. `(2.5)` is the nonlinear differential equation:

- involving the linear spring term,
- cubic spring term,
- linear damping term,
- and state-dependent damping term.

The important direct fact is that Drenth uses this equation as the common nonlinear plant for both embeddings.

### 9. Drenth explicitly considers two LPV embeddings of the same nonlinear model

He says:

- first, an affine-dependency LPV embedding is considered,
- second, an unrestricted dependency embedding is considered, which in this example is polynomial.

So the subsection is not comparing two different plants, but two different embeddings of the same plant.

### 10. In the affine embedding, Drenth explicitly chooses two scheduling variables

For the affine model, Drenth introduces:

- `p_1 = x`,
- `p_2 = x^2`.

He states that these capture position-dependent nonlinearities in the nonlinear spring.

### 11. Equation `(2.6)` gives the affine LPV embedding

Drenth writes the affine LPV embedding using the scheduling variables `p_1` and `p_2`.

This is the first explicit side of the comparison.

### 12. Equations `(2.6)`--`(2.7)` show the affine LPV-LFR form

After introducing state coordinates `x_1 = x`, `x_2 = x_dot`, Drenth writes the affine LPV-LFR realization.

The directly relevant structural point is that:

- the affine embedding uses two scheduling variables,
- and the corresponding Delta block uses these as separate diagonal entries.

### 13. In the rational/polynomial embedding, Drenth explicitly uses only one scheduling variable

Drenth then states that if rational dependency is allowed, an embedding can be obtained using only:

- `p_1 = x`.

This is one of the most important direct comparisons in the subsection.

### 14. Equation `(2.8)` gives the rational/polynomial embedding

Drenth writes the nonlinear MSD again, now embedded using only `p_1`.

The direct point is that the `x^2` dependence is no longer treated as a separate scheduling variable.

### 15. Equations `(2.8)`--`(2.9)` show a rational LPV-LFR with repeated use of the same scheduling variable

In the rational LPV-LFR example:

- the same scheduling variable `p_1` appears multiple times in the Delta block,
- rather than introducing a second independent scheduling variable.

This repeated structure is directly visible in eq. `(2.9)`.

### 16. Drenth explicitly compares the admissible scheduling sets of the two embeddings

Drenth then considers the system on a bounded subset of the state space and computes the resulting scheduling ranges.

He states that:

- in the rational case: `-1 <= p_1 <= 1.2`,
- in the affine case: the set additionally includes `0 <= p_2 <= 1.44`.

This is an explicit, concrete comparison of the scheduling descriptions induced by the two embeddings.

### 17. Drenth explicitly states that the coupling between `p_1` and `p_2` is lost in the affine case

This is one of the strongest direct claims in the subsection:

- the affine embedding loses the inherent coupling between `p_1` and `p_2`.

He then uses this loss of coupling to interpret the scheduling-set comparison.

### 18. Drenth explicitly compares the induced families of local LTI models

Drenth states that, by sampling the scheduling set of each embedding, local LTI models can be determined.

He then states that the set of LTI systems corresponding to the affine-dependency model is much larger than the set corresponding to the rational-dependency model.

### 19. Drenth explicitly attributes this difference to overbounding

Drenth states that the larger LTI family of the affine model is caused by overbounding.

He makes the further direct claim that the underlying nonlinear system can never admit the behavior represented by the difference between the two sets.

This is the final concrete conclusion of the subsection.

## Safe Inferences from Section 2.1.1

These are not stated as theorems in the subsection, but they are immediate and low-risk consequences of what Drenth explicitly writes.

### 1. Repetition in the Delta block can be used to encode richer dependency without introducing more independent scheduling variables

Why this is a safe inference:

- Drenth's affine example uses two independent scheduling variables,
- Drenth's rational example uses one scheduling variable repeated several times in `Delta`,
- and the subsection presents this as part of the trade-off.

So it is safe to infer that repetition in `Delta` is one mechanism for expressing richer dependency while keeping the scheduling-variable count lower.

### 2. The subsection treats "polynomial" dependency as a special case within the broader rational-dependency viewpoint

Why this is a safe inference:

- Drenth says the second embedding is "unrestricted" and, in this example, polynomial,
- but he still places it under the rational-dependency trade-off discussion.

So the example supports the safe reading that polynomial dependence is being used as an illustrative subclass inside the broader rational-dependency discussion.

### 3. The main comparative advantage of the rational example in this subsection is structural, not merely numerical

Why this is a safe inference:

- the subsection is not just saying one curve looks better,
- it ties the difference to preserved coupling and reduced scheduling freedom.

So it is safe to infer that the claimed advantage is structural: preserving admissible dependency relations inside the model rather than discarding them into a free scheduling set.

### 4. The affine-vs-rational trade-off is not only about expressiveness, but also about how much scheduling freedom is introduced

Why this is a safe inference:

- Drenth repeatedly links the issue to the set of admissible scheduling values,
- the set of local LTI systems,
- and controller conservatism.

So it is safe to read the subsection as saying that the real issue is not only "model complexity" but "how much artificial freedom the embedding introduces."

## Not Justified by Section 2.1.1 Alone

This is the part that needs the most discipline.

### 1. Section `2.1.1` does not prove that rational dependency is always better than affine dependency

The subsection gives:

- a conceptual argument,
- and one explicit nonlinear MSD example.

It does **not** prove a universal theorem that rational dependency always gives lower conservatism for every system or every embedding.

### 2. Section `2.1.1` does not prove that using fewer scheduling variables is always possible

In the MSD example, rational dependency reduces the number of scheduling variables from two to one.

The subsection does **not** prove that the same reduction is always achievable for arbitrary nonlinear systems.

### 3. Section `2.1.1` does not give a general constructive algorithm for converting affine embeddings into rational LPV-LFR embeddings

It provides one worked comparison, not a generic algorithm.

So Section `2.1.1` alone does not tell us how to build a rational LPV-LFR for an arbitrary plant.

### 4. Section `2.1.1` does not discuss well-posedness

The subsection is about:

- overbounding,
- coupling,
- scheduling-variable count,
- and induced LTI families.

It does **not** address whether the rational example is well-posed in the generic algebraic-loop sense.

That topic belongs to Section `2.2`.

### 5. Section `2.1.1` does not justify a plant-specific latent-variable choice for a new application

The MSD example shows one affine LPV-LFR and one rational LPV-LFR realization, but it does not provide a rule for choosing latent variables in a different plant.

That remains application-specific work.

### 6. Section `2.1.1` does not prove minimality of the rational example

The subsection shows a lower scheduling-variable count in the example, but it does not prove minimality of the realization in any formal sense.

### 7. Section `2.1.1` does not prove that the rational structure alone guarantees improved controller synthesis in every setting

Drenth motivates reduced conservatism, but the subsection does not include controller design or a theorem on controller performance.

So any controller-level claim should remain careful and conditional.

## Equation-by-Equation Audit

### Equation `(2.5)`

What it is:

- the nonlinear MSD differential equation used as the common starting point.

What it establishes directly:

- the plant contains nonlinear spring and damping terms,
- there is one underlying nonlinear system for the comparison.

What it does **not** establish by itself:

- why one embedding is better than the other,
- how to build a general LPV-LFR for arbitrary nonlinear systems.

### Equations `(2.6)`--`(2.7)`

What they are:

- the affine LPV embedding and corresponding affine LPV-LFR realization.

What they establish directly:

- the affine example uses `p_1 = x` and `p_2 = x^2`,
- the affine example treats these as separate scheduling coordinates,
- the affine Delta block contains separate entries for these scheduling variables.

What they do **not** establish by themselves:

- that this affine embedding is unique,
- that every affine embedding of the same plant would behave similarly.

### Equations `(2.8)`--`(2.9)`

What they are:

- the rational/polynomial embedding and corresponding LPV-LFR realization.

What they establish directly:

- the rational example uses only `p_1 = x`,
- repeated copies of `p_1` are used in the Delta block,
- richer dependency is carried in the model structure rather than in additional free scheduling variables.

What they do **not** establish by themselves:

- a universal recipe for constructing such embeddings,
- or that repetition is always the best choice in other systems.

## Immediate Reusable Takeaways from Section 2.1.1

If we restrict ourselves strictly to what Section `2.1.1` supports, then the following takeaways are safe:

1. Rational dependency can reduce overbounding relative to affine dependency in at least some LPV embeddings.
2. A major source of overbounding is loss of coupling between scheduling variables when they are treated as independently free.
3. Repeated use of one scheduling variable inside a Delta block can express richer dependency than introducing only affine dependence on separate scheduling variables.
4. The number of scheduling variables and the structure of the dependency are both design choices that affect conservatism.
5. A comparison between embeddings should be understood not only in terms of equations, but also in terms of:
   - admissible scheduling sets,
   - induced local LTI families,
   - and unattainable behaviors introduced by the embedding.

## What Must Be Deferred Beyond Section 2.1.1

Section `2.1.1` alone is **not enough** for:

1. the generic CT LPV-LFR framework
   - this belongs to Section `2.1`
2. the generic well-posedness condition and sufficient theorem
   - this belongs to Section `2.2`
3. any learnable well-posed parameterization
   - this belongs to Section `2.2`
4. any application-specific realization of a new plant
   - this belongs to the plant-specific derivation

## Short Conclusion

Section `2.1.1` of Drenth's thesis gives:

- the conceptual argument for why rational dependency can be preferable to affine dependency,
- the explanation that overbounding arises from discarded coupling between scheduling variables,
- an explicit nonlinear MSD comparison,
- and a concrete example where the affine embedding induces a larger admissible family of local LTI models than the rational embedding.

Section `2.1.1` does **not** give:

- a universal theorem that rational is always better,
- a general construction algorithm,
- or a plant-specific realization recipe for a new system.

So if we use Section `2.1.1` later in a verification document, the honest phrasing is:

- it justifies the **motivation and modeling preference** for rational dependency,
- but not yet the **specific realization steps** for a concrete plant.

