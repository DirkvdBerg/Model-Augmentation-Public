# Style profile: method sections in Jan Hoekstra's papers

This profile describes how the method sections of the three papers with Jan
Hoekstra as first author are structured. Our thesis is read by the same group
(Schoukens, Toth, Hoekstra), next to these papers, in the same notation. Use the
papers for structure only: section order, how subsections open, where citations
and values go. For wording, follow the Concise section of SKILL.md, not Jan's
sentences, and never copy his wording.

Sources (local text extraction, read 2026-10-05):

- EJC 2025, `literature/augmentation/hoekstra2025_lfr-augmentation-ejc.pdf`,
  Section 3 (Method, about 1,340 words, 5 displayed equations, 6 subsections).
- 2026 first-principles paper (arXiv 2602.17297),
  `literature/closed-loop-id/hoekstra2026_lfr-augmentation-fp-models.pdf`,
  Sections 3 and 5 (Section 5 is about 2,000 words).
- Encoder initialisation paper (arXiv 2602.13108),
  `literature/augmentation/Encoder initialisation methods in the model augmentation setting.pdf`,
  Sections 2 and 3.

Use the method sections as the model. His introductions are generic and nearly
identical across papers; they are not the model.

## Contents

1. A roadmap the subsections follow
2. One job per subsection, opened from the problem
3. Equations inside sentences
4. Depth goes to the own contribution
5. Attribution where the component appears
6. Explicit scope
7. Structure in the method, values in the setup
8. Paragraphs that carry the logic
9. Plain limitations
10. A worked contrast from our own Section 3

## 1. A roadmap the subsections follow

A section opens with one paragraph that states its goal and names its parts in
order. The subsections then appear in exactly that order, so the reader never
wonders where a subsection came from.

> "Given the proposed model augmentation structure (6), our goal is to estimate
> the parameters of the model structure based on measured data in order to
> capture the behaviour of the data-generating system (1). To this end, we
> specify an identification algorithm consisting of identification criterion,
> baseline parameter regularisation, data and baseline model normalisation, and
> parameter initialisation." (2026, Sec. 5)

Sections 5.1 to 5.4 are then those four items in that order.

## 2. One job per subsection, opened from the problem

Each subsection introduces one component of the method. Its first paragraph
runs: problem, consequence, remedy with its source, equation, meaning of the
equation and of its tuning knob. The reader learns why the component exists
before seeing what it is.

> "The LFR-based model (4) is overparameterized. The joint identification of
> (4) can therefore result in learning components canceling out part of the
> baseline model. As a result, the baseline model parameters θbase can be
> significantly altered by the optimization process, leading to excessively
> large deviations from the initialized values θ*base. This can lead to a
> loss of interpretability of the baseline model. To address this issue, we
> adapt the regularization cost term from Bolderman (2024), which is given as"
> [equation] (EJC, Sec. 3.4)

The same shape opens the initialisation subsection of the 2026 paper: common
practice, why it fails here ("for baseline models, random initialisation can be
unstable"), what prior information offers, then the proposal.

## 3. Equations inside sentences

An equation is part of a sentence. The lead-in ends with "is given as", "can be
written as" or "results in"; the equation follows; the sentence continues with
"where" and defines every new symbol once. At most one further sentence says
what the equation does or what its parameter controls.

> "where Λ is a weighting matrix specifying the relative importance of each
> baseline model parameter." (EJC, Sec. 3.4)

> "The tunable variable λ determines how much the baseline parameters can
> deviate from the nominal parameter set θ⁰_base, relative to the change in
> T-step-ahead prediction cost (22)." (2026, Sec. 5.2)

A displayed equation is never left standing without prose that tells the reader
what to take from it.

## 4. Depth goes to the own contribution

Standard machinery gets a phrase and, where useful, a citation. RK4 appears as
"discretised using the RK4 scheme"; Adam gets one sentence. Full derivations
appear only where the derivation is the paper's contribution: the
reconstructability map in the encoder paper (Eqs. 10 to 17), the well-posedness
proof in the 2026 paper. A borrowed result gets a sentence and a pointer:

> "The existence of the encoder ψ has been shown in [4] for state-space models.
> We give a brief overview of the underlying mechanism." (2026, Sec. 5.1)

For the thesis: the prose and the derivations go where this project differs
from the cited work (the gantry realisation, the closed-loop rollout, the
orthogonality construction). The thesis departs from Jan's papers in one
respect: an equation that defines a quantity of this model stays displayed even
when it is borrowed (the reconstructability map applied to our matrices, the
zero initialisation), because the examiners want the mathematics visible. Only
textbook machinery (RK4 stages, ZOH integral, MLP layer formula) becomes a
phrase. A borrowed result is displayed without its derivation.

## 5. Attribution where the component appears

Each borrowed component is cited once, in the subsection that uses it, with a
verb that states the relation: "we adapt ... from", "is adapted", "based on the
work in", "as in". Sections never open with a row of attributions. Jan lists
the contributions in the Introduction, but his method subsections still say
plainly what is his ("we propose", "we introduce"). The thesis does the same:
each section shows which parts are this work's (see Contributions in SKILL.md).

## 6. Explicit scope

Restrictions are stated in one plain sentence where they apply.

> "In this work, we consider fixed interconnection matrices S." (EJC, Sec. 3.2)

> "It is possible to co-estimate the baseline model parameters with θaug and
> θenc in the model augmentation setting [...]. This is, however, outside the
> scope of this paper." (Encoder paper, footnote 1)

## 7. Structure in the method, values in the setup

The method section describes structure and leaves numbers out. Layer counts,
widths, encoder lags, horizons, regularisation weights and epochs appear in the
experiment section, collected in a hyperparameter table. Each choice that could
be questioned gets a one-clause reason that is honest about its basis:

- "This is the minimum number of states required to model the 3-DOF MSD system."
- "Based on previous black-box identification results on the same dataset [...],
  an encoder lag of na = nb = 12 has been applied."
- "The augmented state dimension has been selected based on physical insight
  and a short trial-and-error period as nxa = 2."
- "A regularisation coefficient of λ = 0.01 has been applied, as a result of a
  line-search."

A tuned value is reported as tuned. It is never dressed up as derived.

## 8. Paragraphs that carry the logic

Each paragraph makes one point, and its sentences follow from each other. The
links are real logical relations ("therefore", "to address this", "hence"), not
decoration. A paragraph of two isolated claims, each with its own citation, does
not occur. Paragraph length follows the point: some of Jan's are two sentences.

## 9. Plain limitations

A limitation or an unwanted result is stated in one direct sentence and left
there.

> "This is not desired behaviour for the approximate initialisations,
> indicating that the learning components are learning parts of the system
> dynamics that could be represented by the baseline model." (2026, Sec. 6.3)

> "Ensuring that output augmentations model the LPF exactly is an
> identifiability problem left to future research." (2026, Sec. 6.5)

## 10. A worked contrast from our own Section 3

The intro paragraph of the 2026-10 draft of Section 3 listed six attributions in
six sentences (Hoekstra, Beintema, Kessels, Drenth, contribution,
orthogonality). Following patterns 1 and 5, it becomes a short opening that
names the subsections, and each citation moves into the subsection whose
component it supports. The sentence-level counterpart (the RK4 paragraph) is the
reference example in SKILL.md.
