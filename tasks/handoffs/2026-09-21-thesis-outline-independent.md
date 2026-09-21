# Handoff: write an independent bullet-point thesis outline into the Thesis-writeup section files
**From**: session of 2026-09-21 | **Branch**: Augmentation | **Effort suggested**: high (wide reading, judgment on structure; no derivations)

## 1. Task
Build, from the sources in section 11 and your own judgment, a bullet-point outline of the MSc
thesis (a 12-page, two-column IEEE paper) and write it into the existing LaTeX project
`Thesis-writeup/Writing/`. Each file `sections/NN_*.tex` gets: a comment header with its page
budget and what the section owns (the README convention), the `\section`/`\subsection` headings,
under each subsection 3 to 8 short bullets saying what the section must discuss, and the author's
open questions and doubts for that subsection as `\todo{...}` lines. The outline is the agenda for
supervisor meetings: the author presents the overall section and page split first, then goes
through each section's bullets and doubts. Form your own view of the story and structure from the
sources; do not reconstruct an earlier session's outline.

## 2. Out of scope
- Writing thesis prose. Bullets only.
- Running experiments, training, or any code in `scripts/`, `model_augmentation/`, `lpv_lfr_baseline/`.
- Figures, tables, `Code/`, `Data/`, `Results/`, `Makefile`: the pipeline exists, leave it.
- `main.tex` title/author block, `util/include.tex`, `util/format.tex`: do not modify.
- Adding or renaming section files. The structure is fixed at `00_abstract` to `08_conclusion` plus
  `99_appendix`. If you think a topic has no good home, put it under the best-fitting file and
  raise the placement as a `\todo`.
- Earlier outline attempts: do NOT read `docs/thesis-outline.tex` or
  `docs/thesis-outline-supervisor-ready.md`. They are superseded and would bias you.
- Editing `docs/decisions.md` or any source document you read.

## 3. Where things stand
- Branch `Augmentation`, tree dirty in many directories (unrelated work; do not commit or clean).
- `Thesis-writeup/Writing/` is a self-contained IEEE project (`main.tex` compiles to
  `build/main.pdf`); all ten section files are EMPTY.
- No run in flight that this task depends on.

## 4. Established and verified
- Section files, in input order: `00_abstract, 01_introduction, 02_system_baseline,
  03_augmentation, 04_obc, 05_setup, 06_results, 07_discussion, 08_conclusion, 99_appendix`
  (`Thesis-writeup/Writing/main.tex`).
- `util/format.tex` defines `\todo{}` (red) and `\note{}` (blue), hidden when `\draftfalse`;
  notation there matches `state-level-obc-derivation.tex`.
- No `.tex` file under `Writing/` may contain a `..` path (`Writing/README.md`).
- `docs/decisions.md` (205 entries, about 700 KB, newest first) has duplicate IDs: D-017, D-036,
  D-077, D-148 and D-186 each name two different decisions. Cite by ID plus title.
- Several decisions reverse earlier ones (e.g. open-loop vs closed-loop training; the affine
  offset column in the orthogonality construction changed across D-186, D-190, D-192, D-200).
  Always use the latest dated entry and treat reversals as points the thesis must argue.

## 5. Assumed but not verified
- That the 12-page limit excludes references and appendices. Unknown; raise as a `\todo` in
  `01_introduction` or wherever the overview lands.
- Current status of experiments. Design documents do not mean an experiment was run; check
  `tasks/todo.md`, the newest `tasks/handoffs/2026-09-18-*.md`, and `docs/gantry-augmentation-problem-log.md`
  Section 12 before calling anything done.

## 6. Tried and failed
Format attempts rejected by the author in the 2026-09-18 session (content was not the issue):
- An 80 KB "supervisor-ready" document with a nine-field template per subsection, experiment
  matrix, traceability table, risk register -> rejected -> buried the outline; not what was asked.
- Adding meta sections (purpose, deviations table, cross-cutting layers, defence prep) -> rejected.
- `Discuss / To determine / Ask` label blocks with sentence-long bullets -> "way too much words".
- One-word bullets ("Gap", "Noise", "Monte Carlo") -> rejected: carried no content.
- Accepted: headers, short but meaningful one-line bullets, doubts listed per section. See section 12 example.

## 7. Achieved
None for this task (empty section files).

## 8. The open question
Nothing blocked.

## 9. Next action
Read the sources in section 11 (plus the pointers below as needed), decide the story and the page
split, then fill all ten section files in one pass and compile.

Requirements the author has set, which the outline must satisfy:
- It must read as a master thesis in IEEE paper format, so supervisors see the format was
  followed. Use the IEEE conventions of the shipped template (`Writing/reference/`, including
  `IEEEtran_HOWTO.pdf`): abstract plus Index Terms, numbered sections with subsections where they
  help, introduction ending in contributions and a paper-organisation paragraph, conclusion,
  references, appendix. Where a TU/e thesis expectation might differ from a plain IEEE paper,
  raise it as a `\todo`.
- Per subsection, bullets of what to discuss. Mark content that only gets ONE sentence with
  `\note{one sentence}` (example given by the supervisor: parameter recovery without augmentation
  is stated in one sentence). Decide which other items deserve only a sentence.
- Results: per subsection, name the figure or table that shows it (type, axes or columns, what it
  compares) and the claim it supports, as a bullet starting `Figure:` or `Table:`. Keep the total
  figure count realistic for 12 two-column pages.
- Sources: where a statement needs a reference, name it with the cite key from
  `docs/references.md` (e.g. `\cite{key}` inside the bullet). If no source is known yet, add
  `\todo{source needed: <claim>}`.
- Page budgets over the section files that sum to 12 pages.
- The research plan is the approved baseline: where the current work departs from it, the
  relevant bullet says so, and the departure is argued or raised as a `\todo`.
- A black-box comparison is part of the thesis.
- Results bullets follow the supervisor's rule (Maarten): never only show a result; state the
  expectation or criterion, the verdict, and why.
- Every item in the author's raw notes (`../../Thesis-niet-vergeten-te-schrijven.md`, Dutch and
  English, two lists: "Thesis" and "Verdedigen") appears somewhere, as a bullet or a `\todo`.
  Items that are not thesis content (e.g. "make a folder", "look at decisions.md") may be skipped.
- Read `docs/decisions.md` critically: titles first, then the entries relevant to each section;
  note reversals and superseded entries.

Further sources, read on demand:
- `Research-Plan/research-methods.md`, `Research-Plan/literature-survey.md`: plan fragments.
- `scripts/gantry/orthogonal-by-construction/documentation/decisions.md` and
  `thesis-readiness-guideline.md` (same folder): orthogonality decisions; math-writing rules.
- `scripts/gantry/orthogonal-by-construction/documentation/state-level-obc-derivation.tex`,
  `obc-io-additional-state-chapter.tex`, `obc-gantry-physical-state-extension.tex`.
- `LPV/LFR-derivation-supervisor.tex`, `LPV/LFR-SVD-derivation.tex`: baseline and LFR.
- `scripts/gantry/meeting/meeting-26-08-2026/controller-defence.md`: closed-loop controller rationale.
- `docs/augmentation-validation-design.md`: validation design (marked unbuilt; plan, not evidence).
- `docs/writeup/README.md`: earlier supervisor write-up, notation and figure style.
- `docs/kamtin-telica-schema.md`: real-data split and signals; `docs/references.md`: cite keys.

## 10. Acceptance criterion
- `main.tex` compiles without errors (`cd Thesis-writeup/Writing && latexmk -pdf -outdir=build main.tex`).
- All ten section files are filled; every header comment states a page budget; budgets sum to 12.
- Every raw-note item is findable in the outline or deliberately skipped as non-content.
- Every bullet is one line and carries a concrete point.
- Every results subsection names at least one `Figure:` or `Table:`; one-sentence items are marked;
  claims needing literature carry a cite key or a `source needed` todo.

## 11. Read these first
1. `Research-Plan/research-plan-dirk-van-den-berg.pdf`: approved plan, research question, four aspects.
2. `../../Thesis-niet-vergeten-te-schrijven.md` (in `Graduation Project/`): the author's raw notes and doubts.
3. `docs/decisions.md`: the rationale for nearly every design choice; newest first.
4. `docs/control-reasoning.md`: project identity and control reasoning (parts may be outdated; decisions.md wins).
5. `Thesis-writeup/Writing/README.md` + `main.tex`: the structure you write into.

## 12. Do not
- Write sentences or paragraphs under a heading. Wanted format, one subsection:
  ```latex
  \subsection{Example subsection}
  \begin{itemize}
    \item Short phrase stating one concrete point \cite{somekey}
    \item Minor supporting check \note{one sentence}
    \item Figure: quantity A vs quantity B for model X and Y; supports claim Z
  \end{itemize}
  \todo{Open doubt phrased as a question the supervisor can answer}
  ```
- Add summary tables, traceability tables, risk lists or status legends to the paper files.
- Read or write anything in `kamtin-data/Data Telica/` or `kamtin-data/Telica.mat`.
- Use em-dashes or double hyphens anywhere (standing rule).

## 13. Operational
- Compile: `cd Thesis-writeup/Writing && latexmk -pdf -outdir=build main.tex` (pdflatex also works).
- No conda env, cluster or data dependency.
- In the final chat message, give the author the page-split overview (section, pages, one-line
  message) so it can be shown first in the meeting.

## 14. Delegation
At most one Explore subagent, only for sweeping `docs/decisions.md` for entries per section.
Everything else inline.
