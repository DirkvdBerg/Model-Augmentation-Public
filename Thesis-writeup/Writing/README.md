# Writing

Self-contained LaTeX project for the 12-page IEEE paper. Compile `main.tex`,
nothing else. No `.tex` file here may contain a `..` path: everything LaTeX
needs lives under `Writing/`, so this directory can be zipped and handed to
Overleaf or a supervisor as is.

## Editor build and SyncTeX invariant

The repository VS Code settings at the root and in `Writing/.vscode/`
must preserve the following workflow:

1. Saving any included `.tex` file (Ctrl+S) rebuilds `main.tex`, never the
   section file in isolation. Changes written from outside the editor do not
   start a build (`onSave`): with `onFileChange`, AI edits, OneDrive and git
   started overlapping builds that aborted on a locked `main.aux` and left no
   `main.synctex.gz`.
2. The only editor PDF is `build/main.pdf`, produced together with
   `build/main.synctex.gz` using `-synctex=1`.
3. The PDF is opened with LaTeX Workshop's tab viewer; double-clicking it must
   navigate back to the corresponding source line.

Do not change the root-file pattern, output directory, recipe, SyncTeX flag, or
viewer configuration without testing both an included-section save and reverse
SyncTeX. Do not add per-section root directives: `main.tex` is the sole root.

AI edits of any `.tex` file here must leave this workflow working. Dirk relies
on it for every save.

- Edit content only. Never add `% !TEX root`, `\documentclass`,
  `\begin{document}`, or a `..` path to a section file, and never change
  `main.tex`'s structure, `.latexmkrc`, or either `.vscode/settings.json` unless
  asked.
- Leave the file compiling: braces and environments balanced, and `%`, `&`,
  `_`, `#` escaped outside math. A failed build leaves `build/main.pdf` stale,
  so saving no longer refreshes the PDF and double-clicking lands on old lines.
- Only the editor recipe writes build output, so there is exactly one PDF and
  one SyncTeX file. Test compiles go to a scratch directory, never to
  `Writing/` or `build/`.
- Dirk saves the file before asking for an AI edit. An edit on disk while the
  editor holds unsaved changes causes a save conflict, and that save then
  triggers no rebuild.

Done when: after the automatic rebuild, `build/main.log` has no line starting
with `!`.

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

## Writing a section

Write one section at a time, starting from the `WRITING GUIDE` at the top of its
file. Inspect sources in this order:

1. Read the research-plan anchor and identify wording that can be preserved, wording that needs a method update, and explicit departures. Reuse research-plan wording where it is accurate and concise; otherwise write new text, concise and to the point. The research plan is a source, not a constraint.
2. Read `docs/Thesis-documentation/Meeting-audit/candidate-thesis-impact.md` and the relevant theme in `thematic-thesis-audit.md`. Use Quinten's 18 September outline feedback to set priorities: realistic ASMPT motion, settling, controller transfer, justified choices, and final evidence rather than development history.
3. Read the newest relevant entries in `docs/decisions.md` to establish why the final choices were made. If an experimental choice is still open, keep it open rather than converting supervisor advice into a completed decision.
4. Audit the final code path to establish what was actually implemented. The main boundaries are:
   - `lpv_lfr_baseline/`: independent physics and LPV-LFR baseline;
   - `model_augmentation/`: reusable augmentation, rollout, encoder, closed-loop, and OBC framework;
   - `scripts/gantry/gantry_dynamic/`: gantry-specific configuration, data, model assembly, controller, training, diagnostics, OBC adapter, and evaluation;
   - `scripts/gantry/gantry_interconnect_dynamic.py`: thin run entry point and final run knobs.

   The audit informs claims; it is not content. Include only what the reader
   needs to follow the argument.
5. Trace empirical claims to final configurations, saved artefacts, figure data, and evaluation code. Plans and design notes are not evidence.
6. Read the relevant primary papers from the mandatory set below and add literature only where an external claim, method origin, theorem, or comparison needs support.

### Order of work

The reader's argument comes before every style rule below. Work in this
order, and do not start a step before Dirk approves the previous one:

1. **Section brief.** Answer six questions: what must the reader understand;
   which choices need a reason; which equations are load-bearing; which figure
   carries the argument; which claims need code, result, or literature
   verification; and which tempting claims are out of scope. Parameter-recovery
   tests, failed routes, burn-in screens, and similar development steps stay
   out of the main narrative unless they are required to establish a final
   claim.
2. **Claim list.** Per subsection, one governing question and the ordered
   claims that answer it, each with its source and its verified strength.
3. **Equations.** Every load-bearing construction displayed, checked against
   the preceding sections and the code.
4. **Prose,** one paragraph at a time.

Claim rules, applied from step 2 onward:

- **Claim strength.** State the weakest claim the evidence supports.
  Wording in `docs/decisions.md`, plans, or code comments is not evidence for
  a thesis claim; re-derive it or verify it at the source.
- **Method versus experiment.** A method section describes the method and its
  gantry realisation. The simulated truth, controller design, numerical
  settings, and measured numbers belong to Setup or Results.
- **Checks must be able to fail.** A proposed verification that passes by
  construction is not a check.

Then run the four phases below. Dirk names the phase ("structure", "mark",
"rewrite", "check"); the default is structure. The AI supplies content,
structure, and checks; Dirk writes every sentence of the final text. An AI's
word choice and sentence structure carry its fingerprint and survive
paraphrasing, and rewrite prompts do not remove it (evidence basis below).

### Phase 1: Structure

AI role: co-author of content. Produce the brief, claim list, and equations of
the order of work above in chat. Write prose into the section's `.tex` file only
on request, one paragraph at a time, following the authorial-voice rules below
and the editor build invariant above, until Dirk approves the content.

- Write one claim per sentence, connected by the reasoning between them, so the
  section can be rewritten sentence by sentence without becoming a list of
  disconnected statements. Where a dependency must stay attached, voice rule 9
  takes precedence.
- Display every load-bearing construction as an equation; never compress a
  formula into prose to save space. Prose gives the purpose, assumption, or
  consequence of the equation.
- When a choice or assumption needs a reason that is not in `docs/decisions.md`,
  the code, or a cited source, write `because: ?` in its place. An invented
  reason is either wrong or generic, and reasons are what the examiners test.

Done when: Dirk approves the content.

### Phase 2: Mark

AI role: marker. Wrap each technical term in `\kw{}` and change nothing else,
style issues included. A technical term is a noun phrase that names a quantity,
model object, component, or method defined in this thesis or its cited
literature. Mark every occurrence; a multiword term is one unit. Math, symbols,
and `\cite`/`\ref`/`\eqref` are fixed by default and stay unmarked. `\kw{}`
prints teal in the draft build and plain with `\draftfalse`.

Example (Section II-B):

```latex
The measured \kw{payload position} $Y$ is both a \kw{state} and the \kw{scheduling variable},
making the \kw{baseline} a \kw{self-scheduled LPV model}.
```

Done when: deleting every `\kw{` and its closing brace gives back the approved
text exactly. Dirk then commits the marked file; that commit is the reference
for phase 4.

### Phase 3: Rewrite (Dirk only)

Dirk rewrites each paragraph in place around the `\kw{}` terms: read it, cover
it, write it from memory. Useful moves: split a sentence, start from a
different subject, use a verb instead of a noun, use "we" with an active verb,
move the reason. In this phase the AI does not edit the section files and does
not write or suggest sentences. Asked about a sentence, it answers with a
question or a fix of at most five words.

### Phase 4: Check

1. `git diff --word-diff <phase-2 commit> -- <file>`. Uncoloured
   words were kept from the draft; long uncoloured runs outside `\kw{}` terms
   show where the draft's wording survived.
2. AI role: checker. Compare the phase-2 commit with Dirk's version and report
   every instance as this table, nothing else:

   | # | type | problem | fix (max 5 words) |
   |-|-|-|-|
   | 3 | term | `\kw{scheduling variable}` missing | restore term |
   | 5 | meaning | "only if" became "if" | restore "only if" |

   Types: meaning (a claim changed), term (a `\kw{}` term, symbol, or reference
   changed or missing), grammar.

Done when: the table has no meaning or term rows, and the section meets its
completion criterion and page budget.

## Authorial voice and non-generic prose

The aim is clear, recognisably project-specific technical writing, not imitation
of a detector's idea of human prose. Automated authorship detectors are not a
validation tool: they can misclassify human text and perform poorly on short
passages. Review the argument and evidence instead.

Apply these rules to every AI draft (phase 1) and use them as the checklist for
Dirk's rewrite (phase 3):

1. Preserve Dirk's accepted wording and sentence rhythm where it is accurate.
   Edit the smallest necessary span; do not rewrite an entire paragraph merely
   to make it sound more polished.
2. Give every paragraph one concrete job: introduce a physical fact, make a
   modelling choice, define a construction, verify it, or state a consequence.
   Delete sentences that could be pasted unchanged into an unrelated paper.
3. Name the actual object and action. Prefer “closing the loop recovers
   \(M(Y)\ddot q=f_{\mathrm{net}}\)” over “this highlights the effectiveness of
   the proposed framework.”
4. Use plain technical verbs. Avoid inflated, generic vocabulary such as
   “delve,” “intricate,” “pivotal,” “crucial,” “meticulous,” “seamless,”
   “comprehensive,” “underscore,” “showcase,” “leverage,” “utilize,” “enhance,”
   “subsequently,” or “additionally” unless that word is genuinely the most
   precise choice.
5. Do not manufacture transitions. Use “however,” “therefore,” “because,” and
   “consequently” only when the stated logical relation is real. Avoid routine
   “first/next/finally” scaffolding and concluding summaries that repeat the
   paragraph.
6. Do not paraphrase an equation line by line. Prose around an equation must
   explain its purpose, assumption, interpretation, or consequence.
7. Keep technical terms stable rather than cycling through synonyms. Define a
   symbol or acronym once, then use it consistently.
8. Prefer direct claims with an explicit subject and a finite verb. Give a
   consequence its own clause instead of a trailing participle (“…, making the
   baseline …”), use the verb instead of a noun made from it (“we estimate,” not
   “the estimation of … is performed”), and write “is” rather than “serves as.”
   Avoid throat-clearing phrases such as “it is important to note,” “it should be
   emphasized,” “in order to,” and “it can be seen that.”
9. Vary sentence length only as the reasoning requires. Use a short sentence
   for a decisive result and a longer sentence when a dependency or qualification
   must remain attached. Do not force every paragraph into the same cadence.
10. State uncertainty and limitations where the evidence requires them. Do not
    replace a qualified project claim with smooth but stronger generic prose.
11. Check attribution sentence by sentence. Literature supplies general methods
    and definitions; gantry-specific algebra, implementation choices, and results
    remain explicitly this project's work.
12. In the final voice pass, flag repeated sentence openings, repeated transition
    words, trailing participle clauses, unnecessary three-part lists, “not only …
    but also” constructions, promotional adjectives, and claims with no concrete
    noun, equation, datum, or source. Revise for meaning, not merely for lexical
    variation.

These rules implement IEEE's guidance to use clear, simple sentences without
unnecessary words. They also address measured tendencies of LLM-written and
LLM-modified text: a small set of overused words, noun-heavy grammar with
trailing participle clauses, formulaic structure, and wording that is more
homogeneous across authors. Lexical diversity within one text is not a reliable
marker; studies disagree on its direction. The rules are quality controls, not
evidence about who authored a passage.

Evidence basis: [IEEE technical-English guidance](https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/write-in-technical-english/),
[linguistic profiling across human and LLM text](https://aclanthology.org/2025.emnlp-main.1163/),
[changes in LLM-modified scientific prose](https://aclanthology.org/2026.lrec-1.142/),
[excess vocabulary in LLM-era abstracts](https://arxiv.org/abs/2406.07016),
[grammatical and rhetorical style of LLMs](https://doi.org/10.1073/pnas.2422455122),
[cues used by expert human detectors](https://arxiv.org/abs/2501.15654),
[model fingerprints that survive paraphrasing](https://arxiv.org/abs/2502.12150),
[detection of guideline-rewritten text](https://arxiv.org/abs/2607.27183) (detector vendor report),
and [documented limitations of AI-text classification](https://openai.com/index/new-ai-classifier-for-indicating-ai-written-text/).

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
submission build: every `\todo{}` and `\note{}` disappears, and every `\kw{}`
prints as plain text.
