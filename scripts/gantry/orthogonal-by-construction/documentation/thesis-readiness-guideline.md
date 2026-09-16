# Thesis-readiness guideline

A checklist for turning a working derivation into a chapter that pastes into the thesis
unchanged. Settled 2026-09-14/15 and applied first to `obc-io-additional-state-chapter.tex`.

## Register

Connected mathematical prose, in the manner of the source paper's own derivation. A lead-in
sentence states what is being done, the mathematics is displayed, a following sentence states the
consequence. Not a step ladder, not bare equations, not paragraphs of commentary around small
results.

Exemplar, which fixes the density, the sentence length and the ratio of prose to mathematics:

> **3.1 Learned memory**
>
> Gyorok's learning component is a static map of the regressor. We extend it with an internal
> state, so the correction at time `k` may depend on information not present in the finite
> history:
>
>     y_k = phi(s_k,u_k) th_b + g_y(s_k,u_k,x_a,k)
>     x_a,k+1 = g_a(s_k,u_k,x_a,k)
>
> The baseline defines no equation for `x_a`. Collecting the stored history and the learned state
> into `xi_k = col(s_k, x_a,k)` and using the routing of (9), the predictor becomes
>
>     xi_k+1 = [ A s_k + B_u u_k + B_x phi th_b ]   [ B_x g_y ]
>              [              0                 ] + [   g_a   ]
>
> The baseline contributes zero rows on the learned-state block, which Section 4 uses throughout.

Prose carries the derivation forward rather than annotating it. A result is stated once, without a
disclaimer appended.

## Length rules

1. **A sentence following a displayed equation must add something the equation does not contain**:
   what a newly introduced object means, a consequence, a condition, or a pointer forward. Never a
   restatement of what the equation displays.

   *Keep.* After shift and insertion matrices: "The shift moves older samples down one slot. The
   two insertion matrices independently write the current input and the new output." Neither
   sentence can be read off the entry definitions, and without them the matrices are opaque. The
   sentence saying why the routing identities hold is the same kind.

   *Cut.* "where the first term is the baseline contribution and the second the learned
   contribution", which restates the visible block layout.

   The rule behind both: **explain the role whenever the definition is structural.** A matrix
   defined by its entries and used for what it does needs a bridging sentence. An identity that
   holds for a structural reason needs that reason. A clause restating a block layout bridges
   nothing. Where the rule is unclear, match the two examples rather than reasoning from it.

2. **One sentence before and one after is the norm.** A second sentence after is allowed where a
   newly introduced object needs its role explained, as in the shift-matrix case.

3. **No connective filler.** Not "it is important to note that", "as we can see", "in other
   words", "this allows us to", "it should be emphasised". If the sentence survives deleting the
   opener, delete the opener.

4. **Deduplicate the qualifications into one limitations section.** Each distinct qualification
   appears once, in that section, in its strongest form. Repetitions are deleted; a qualification
   appearing nowhere else is moved, never dropped. A hedge repeated near its result is the single
   largest source of bloat.

5. **Hard cap on length: no longer than the working document, and shorter is better.** Same
   mathematics in fewer pages is the goal, not a reorganisation of the same volume of text.

Applied to a whole document, these mostly delete.

## Displayed versus inline

Formulas that carry meaning are displayed. Inline mathematics is for a single symbol or a
dimension referred to in passing. Anything containing a relation or an operator is displayed.
Number only equations referred to later.

## Structure

Sections named for content, not for actions. A `Step N: <imperative>` sequence is a task list, not
a chapter. The shape that worked:

1. Setting: the problem, why the development is in these coordinates, one paragraph on what is
   inherited and what is ours, and what the main result will be.
2. The inherited construction. Notation table here. Nothing of ours.
3. The machinery being built, ending with its exactness statement.
4. The contribution, starting visibly here.
5. The main results, with side properties as short remarks.
6. What is and is not established: every hedge, collected.

Appendix: verification as one table of checks and outcomes.

Sections 2 and 3 are setup and should be the fastest. The weight belongs in 4 and 5.

**No per-section discussion subsections and no worked simulation examples.** A thesis that
demonstrates algorithms needs simulations; a document that proves propositions needs proofs.
Adding either pads the document and breaks the line.

## Defects to clear

The categories a working derivation carries into a chapter:

1. **Step ladder.** Imperative-titled subsections with prose annotating each step instead of
   carrying the argument between them.
2. **Provenance scaffolding in the body.** A "relation to the source" column in the notation
   table, or prose marking which line is the source's and which is ours. Attribution in a thesis
   is a citation made once.
3. **Meta-commentary about the project.** Handoffs, which notation was replaced, errata in the
   source's prose. That is a review log, not a chapter.
4. **Hedging density.** Most paragraphs closing with a disclaimer. A chapter states its
   assumptions once and proceeds; limitations go in one section.
5. **Inline mathematics in running prose.** See above.
6. **Repository paths in the body.** A pasted chapter cannot refer to a path in a private
   repository.
7. **No figure**, where the structure being described is a block diagram.
8. **No bridge to the thesis.** The chapter must say why the development is in these coordinates
   and what it does and does not give for the target system.

## Acceptance

- Compiles with no undefined references and no missing citations.
- Each defect above addressed, one line per item saying how.
- No mathematics stating a relation, definition or operator inline in running prose.
- No repository path, no session or handoff reference, no "ours versus the source" commentary in
  the body.
- The mathematics unchanged: every proposition, lemma and limitation survives with the same
  content.
- The real test: the body pastes into the thesis and nothing changes.
