# Writing

Self-contained LaTeX project for the 12-page IEEE paper. Compile `main.tex`,
nothing else. No `.tex` file here may contain a `..` path: everything LaTeX
needs lives under `Writing/`, so this directory can be zipped and handed to
Overleaf or a supervisor as is.

| Path | Role |
|-|-|
| `main.tex` | class, inputs, title block. Roughly 70 lines, no content. |
| `util/include.tex` | packages only. Swap templates by touching this file. |
| `util/format.tex` | macros, notation, draft switch. Notation matches `state-level-obc-derivation.tex`. |
| `sections/NN_*.tex` | one file per section, flat, numeric prefix. Each header states its page budget and what it owns. |
| `figures/` | generated PDFs, `\graphicspath` points here and only here. Never hand-edited. |
| `tables/` | generated tabular fragments, `\input` from a section. Numbers are never retyped. |
| `refs.bib` | own entries, cite keys matching `docs/references.md`. |
| `IEEEtran.cls`, `IEEEtran.bst`, `IEEEabrv.bib` | from the shipped IEEE template, kept at project root so the build is standalone. |
| `build/` | aux, log and output PDF. |
| `reference/` | the original untouched IEEE template and BST distribution, for lookup only. Not part of the build. |

## Section writing workflow

Write one section at a time using the `WRITING GUIDE` at the top of its file.
For each section, inspect sources in this order:

1. Read the research-plan anchor and identify wording that can be preserved, wording that needs a method update, and explicit departures.
2. Read `docs/Thesis-documentation/Meeting-audit/candidate-thesis-impact.md` and the relevant theme in `thematic-thesis-audit.md`. Use Quinten's 18 September outline feedback to set priorities: realistic ASMPT motion, settling, controller transfer, justified choices, and final evidence rather than development history.
3. Read the newest relevant entries in `docs/decisions.md` to establish why the final choices were made. If an experimental choice is still open, keep it open rather than converting supervisor advice into a completed decision.
4. Audit the final code path to establish what was actually implemented. The main boundaries are:
   - `lpv_lfr_baseline/`: independent physics and LPV-LFR baseline;
   - `model_augmentation/`: reusable augmentation, rollout, encoder, closed-loop, and OBC framework;
   - `scripts/gantry/gantry_dynamic/`: gantry-specific configuration, data, model assembly, controller, training, diagnostics, OBC adapter, and evaluation;
   - `scripts/gantry/gantry_interconnect_dynamic.py`: thin run entry point and final run knobs.
5. Trace empirical claims to final configurations, saved artefacts, figure data, and evaluation code. Plans and design notes are not evidence.
6. Read the relevant primary papers from the mandatory set below and add literature only where an external claim, method origin, theorem, or comparison needs support.
7. Turn the outline bullets into prose, then check the section's completion criterion and page budget.

Before drafting, write a short section brief with six questions: what must the reader understand; which choices need a reason; which equations are load-bearing; which figure carries the argument; which claims need code, result, or literature verification; and which tempting claims are out of scope. Parameter-recovery tests, failed routes, burn-in screens, and similar development steps stay out of the main narrative unless they are required to establish a final claim.

## Validation story

The application claim is built in layers and must not be reduced to an unseen multisine phase:

1. Interpolation uses held-out phase realizations and operating points inside the training region.
2. Operational validation uses ASMPT-relevant, multisine-free motion profiles and evaluates both tracking and a fixed post-move settling interval.
3. Controller-transfer validation trains with one known controller and evaluates a credible changed controller as the digital-twin use case, while a plant-domain check tests whether feedback merely masks model error.
4. Operating-range extrapolation varies quantities such as stroke, acceleration, amplitude, or scheduling position beyond the represented training region. A controller change is reported separately as a closed-loop distribution shift, not automatically labelled plant extrapolation.

The exact motion profiles, controller variant, seed count, noise level, and primary metric remain open until fixed in `docs/decisions.md`. Current defaults do not settle those scientific choices. When selected, justify them from machine practice, data, or a controlled validation study.

Use physical RMS error as the default primary candidate for tracking and settling because it remains interpretable in metres and is well-defined on low-variation settling segments. Do not mix RMS, NRMS, and BFR as co-equal headline measures. If another primary metric is selected, record the reason before writing. Multisine phase seeds and neural-initialisation seeds answer different questions and must be reported separately.

## Derivation policy

Use derivations to expose the reasoning that is specific to this work, establish
correctness, or connect a cited method to the implemented model. Do not add
algebra merely to make a section appear more mathematical.

For each derivation:

1. State the assumptions and define every signal before manipulating it.
2. Show the load-bearing steps a reader needs to verify the construction.
3. Distinguish an adopted result from this project's own derivation or adaptation.
4. End by connecting the result to the implemented equations, signal routing, or numerical evaluation.
5. Put routine algebra, expanded coefficient matrices, and longer proofs in an appendix or technical supplement.
6. Cite the original source for an established method; do not reproduce its full proof unless the proof is changed or needed for a project-specific claim.

The required depth is section-dependent. The LPV-LFR baseline needs a
project-specific derivation because its exact realization is central to the thesis.
The coordinate transformation, closed-loop residual rollout, RK4 boundary,
orthogonality construction, and loss function need only the short steps required
to remove ambiguity. Data-design and hyperparameter arguments are methodological
justifications, not derivations.

For every nontrivial design or hyperparameter choice, leave a short choice record
before writing prose:

1. State the selected value or construction.
2. Name the realistic alternatives.
3. Classify the basis as physics-derived, literature-supported, inherited, tuned on validation data, safety-constrained, or compute-constrained.
4. Explain which failure or capability the choice addresses.
5. State the consequence or limitation introduced by the choice.
6. Point to a sensitivity study, diagnostic, decision entry, or source where one exists.

Not every numerical setting needs a theoretical optimum. A transparent tuning or
computational rationale is preferable to a false physics justification. Choices that
change expressivity, identifiability, stability, data coverage, or comparison fairness
must be discussed in the paper; routine software settings can remain in the appendix or
configuration record.

## Mandatory methodological reading

Every thesis-writing session must read the relevant parts of the following local
papers directly. Project summaries, decision records, and code comments do not
replace these papers.

| Paper | Required role in the thesis |
|-|-|
| `literature/augmentation/hoekstra2025_lfr-augmentation-ejc.pdf` | Original encoder-based LFR augmentation framework and interconnection terminology. |
| `literature/closed-loop-id/hoekstra2026_lfr-augmentation-fp-models.pdf` | Extended first-principles augmentation formulation, identification procedure, initialization, and evaluation context. |
| `literature/augmentation/Encoder initialisation methods in the model augmentation setting.pdf` | Model-informed encoder initialization, assumptions, and the boundary between physical and added-state initialization. |
| `literature/lpv-lfr/drenth2025_lpv-lfr-rational.pdf` | Affine and rational LPV-LFR identification, scheduling dependence, and well-posedness context. |
| `literature/Orthogonality/Hoekstra - Orthogonal projection-based regularization for efficient model.pdf` | Projection-based regularization, protected subspace, negation problem, and limits of the penalty formulation. |
| `literature/Orthogonality/gyorok2026_orthogonal-by-construction-augmentation_IFAC-JSC_arXiv2511.01321.pdf` | Orthogonal-by-construction parametrization, uniqueness and recovery assumptions, and the distinction from regularization. |

For each thesis claim based on these papers:

1. Read the actual definition, proposition, assumption, or experiment in the PDF.
2. Record which part is adopted unchanged, which part is extended, and which part does not transfer to the gantry setting.
3. Do not count a later extension and its earlier paper as two independent precedents for the same claim.
4. Check the exact paper version before finalizing metadata or wording.
5. Keep project-specific choices, implementation evidence, and empirical findings attributed to this project rather than to the papers.

The roles of the sources are deliberately different:

| Source | Question it answers |
|-|-|
| Research plan | What was originally proposed and promised? |
| Meeting audit and dated supervisor feedback | What must not be forgotten, which priorities changed, and which historical alternatives must not be mistaken for the final method? |
| `docs/decisions.md` | Why was the final choice made? |
| Final pipeline code | What was actually implemented? |
| Run artefacts and evaluation code | What was actually observed? |
| Verified literature | Which external claims and method origins are supported? |

## Reference verification gate

`docs/references.md` is a candidate map, not evidence that an entry is correct.
Before enabling a citation or bibliography entry:

1. Locate the original paper, thesis, book, or official publication record.
2. Verify title, authors, year, venue, pages, and DOI or stable repository identifier.
3. Read the passage that supports the specific sentence being cited.
4. Record whether the thesis sentence is a direct source claim, an inference, or this project's own derivation.
5. Replace the placeholder in `refs.bib` only after all checks pass.

The bibliography calls in `main.tex` are enabled (user decision 2026-09-22). Entries
copied 1:1 from the research plan's `references.bib` are accepted as-is; a `% CHECK:`
comment above an entry records a known metadata fault. Other entries still pass this gate. Own implementation details and measured results use
code and artefact provenance; they do not acquire external citations merely to
make the text appear sourced.

## Figure and table pipeline

One stem, three folders:

```
Code/Figures/obc_pareto.py   reads   Results/obc_pareto.csv
                             writes  Writing/figures/obc_pareto.pdf
```

Run `make figures` from `Thesis-writeup/` to regenerate, `make paper` to build
the PDF end to end. Scripts import `_fig_common.py` for the IEEE column widths
and body font size, so no figure ever needs rescaling in LaTeX.

## Draft switch

`util/format.tex` sets `\drafttrue`. Flip it to `\draftfalse` for the
submission build and every `\todo{}` and `\note{}` disappears.
