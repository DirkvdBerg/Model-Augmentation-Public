# Handoff: finish the thesis introduction (Section I) and the abstract with Dirk, paragraph by paragraph, from all supervisor feedback (Maarten's first)
**From**: session of 2026-10-20 | **Branch**: Augmentation | **Effort suggested**: high (judgment on wording and
narrative with Dirk; every edit is his decision)

## 1. Task
Bring `Thesis-writeup/Writing/sections/01_introduction.tex` and `00_abstract.tex` to a submittable state together with
Dirk, in three passes. **Pass 1 (no edits):** read all feedback on the introduction and abstract yourself from the
original files (section 13: render the annotated pages and read each comment with the passage it marks), map every
comment by its wording to the current text (`file:line`), and give each a status: done, open, or conflict between
reviewers (Maarten's comment wins; show the conflict anyway). Post to Dirk only the open comments and the conflicts,
Maarten's first, each with ID, reviewer, page, the verbatim comment (Dutch as written) and the location; list the done
ones in one short block. **Pass 2:** settle the section-wide points with Dirk before touching paragraphs: length
(introduction plus abstract about 1.25 pages), the opening (societal positioning), which paragraphs merge, the
research question with its aspects, and whether a contribution list returns. **Pass 3:** walk through the section in
reading order. Dirk reads a paragraph; you show the open comments that belong to it plus your own findings (forward
references, AI-sounding phrasing, citation pinpoints); propose the smallest fix in his wording; apply only what he
approves, with change marks; move to the next paragraph. Finish with the abstract and a closing list of what is done
and what is deferred.

## 2. Out of scope
- Sections II to VIII and the appendix: other sessions. If an introduction change needs a change elsewhere (for
  example a term or a section reference), list it, do not edit it.
- The final numbers in the abstract and any contribution that depends on them: the results are being completed in a
  separate session (`tasks/handoffs/2026-10-20-final-results-polish-blackbox.md`). Write the abstract's structure and
  sentences; put the numbers as `\revnote{}` placeholders with the source CSV named.
- The reference check of the bibliography: a separate Codex session
  (`tasks/handoffs/2026-10-20-thesis-reference-check.md`, report `Thesis-writeup/Documentation/REFERENCE-CHECK-2026-10-20.md`).
  Use its report for citations if it exists; do not redo it.
- `Thesis-writeup/Writing/build/` (Dirk's editor rebuilds on save) and `util/format.tex` (use its macros, do not change
  them).

## 3. Where things stand
- Branch `Augmentation`, last commit `e588e63`; tree dirty, Dirk commits himself. Dirk edits the thesis in parallel:
  re-read a file right before each edit.
- The paper was restructured on 2026-10-19/20 (D-273 and its two amendments): I Introduction, II Identification
  problem, III Physics baseline, IV Augmentation, V OBC, VI Experiment design, VII Results with Discussion (VII-D),
  VIII Conclusion. Every section with subsections opens with a roadmap sentence ("In this section, ...").
- Introduction now (`01_introduction.tex`): motivation, white-box/black-box/physics-informed paragraphs, grey-box
  modelling and the gap, research question with three aspects (`:264-289`), an outline paragraph in Hoekstra's form
  ("The paper is organised as follows: first, ...", `:294`, `\kwtwo`), two `\revnote`s (`:304`, `:309`).
- Known defect: `:152` begins "Physical interpretability means that Physical interpretability means that ..." (a paste
  duplicate). Fix it first, with Dirk's OK.
- Abstract: `00_abstract.tex` holds only outline bullets inside a `wip` block (not shown in the review build), plus a
  todo to write it after the results are frozen. The review build prints an empty "Abstract" heading.
- Stale todo at `:353` (whether references and appendices count towards 12 pages) is answered by Maarten's mail
  (section 4): remove it with Dirk's OK.

## 4. Established and verified
- Feedback files: `Thesis-writeup/Feedback/v2/Feedback-Maarten-Main-supervisor-v2-...pdf` (annotated; introduction on
  PDF pages 2-3) and `Feedback-Maarten-General-v2.txt` (mail); `Thesis-writeup/Feedback/v1/` Jan-1 PDF (introduction
  p1-2), Jasper-1 PDF (p1-2) with `Feedback-Jasper-General.txt`, Quinten-1 PDF (p1-2). The v1 PDFs show older builds:
  map by wording, never by equation or line number.
- Maarten's mail (verbatim): "Van mijn moet het niet absoluut 12 pagina's worden (dat is geen vereiste in de study
  guide bij mijn weten), maar het kan wel wat korter." and "Ik heb niet naar de appendix gekeken."
- Maarten's introduction comments (v2 PDF, verbatim; check status yourself in pass 1):
  p2 "missing" at the empty Abstract; p2 "I would start with a bit of positioning the societal aspect; i.e. why do we
  need such systems, and why do we need model augmentation for them?"; p2 "This can be more compact." (controller and
  root-cause paragraph); p2 "good paragraphs, but can be merged into one paragraph, and shortened a bit." (white-box and
  black-box paragraphs); p2 "You can try to make this more compact. There exists more than PINNS, e.g. PGNNs, GP based
  methods, ..."; p3 "I left some suggestions on what to shorten. Introduction + abstract can be of the same total
  length of what you have now (~1.25 page)" (on Dirk's revnote "shorten the introduction?"); p3 "provide a short outline
  of the paper at the end of the introduction." (done: `:294`).
- Earlier processing: v1 introduction feedback was handled on 2026-10-18 (source comments tagged `FEEDBACK` in
  `01_introduction.tex`; commits `963264c`, `07f5b20`). Treat those as candidates for "done", but confirm each in pass 1.
- Dirk's working requirements for finishing a section (this session, 2026-10-20):
  - Maarten has final say; feedback is read from the original files, in context.
  - Dirk's wording first: move or trim his sentences; fixes of at most about five words; new text only where needed,
    marked `\kwtwo{}` for him to reword, plain and short.
  - Every change visible in the editor build: `\mv[tag]{from}{text}` (moved, blue) with `\delmv[tag]{to}{text}` at the
    old place; `\chg[tag]{old}{new}`, `\del[tag]{..}`, `\add[tag]{..}`; tag = comment ID (for example `M-p2`, `Jan4`,
    `Js9`); `\changelegend` once at the top of the section; titles and captions carry no marks (source comment instead).
  - Point out sentences that read as AI-written (stacked qualifiers, words Dirk would not use such as "admissible",
    connective stock phrases); Dirk decides.
  - Concept before use; no forward references except roadmap or outline sentences; why before how.
  - Text for Dirk to paste: LaTeX in a code block, one sentence per line, wrapped at about 75 characters.
  - Published version of a source where it exists; arXiv only as a preprint (Dirk, 2026-10-20).
- Hoekstra et al. 2026 (`literature/augmentation/hoekstra2026_lfr-augmentation-fp-models.pdf`) is the structural model:
  introduction with the gap, a bullet list of contributions, and an outline paragraph at the end.

## 5. Assumed but not verified
- That every v1 comment marked `FEEDBACK` in the source is still satisfied by the current text after the 2026-10-19/20
  restructure (for example the aspects were renumbered, interpretability was redefined). Settle in pass 1.
- The current length of the introduction: about 1.23 pages in the 2026-10-19 review build, with an empty abstract.
  Settle by measuring the heading positions in a scratch build (section 13) before pass 2.
- Whether Dirk wants the contribution list back (it is commented out at about `:311-325`); ask in pass 2.

## 6. Tried and failed
- Rewording Dirk's sentences into "clearer" drafts: he rejects text that does not sound like him (several times on
  2026-10-20). Start from his own sentence; if none exists for a needed point, say so and ask him to write that clause.
- Pointer-style references in the interpretability definition ("identifiable combinations ... (Section II-C)"): a
  forward reference; resolved by striking the pointer and keeping his words (`:152-154`).

## 7. Achieved
- Outline paragraph at the end of the introduction (Maarten p3), Hoekstra's form, `\kwtwo`.
- LPV definition moved out of the introduction's concern: Section III-B now carries the LPV reason; the introduction
  keeps the motivation sentence.

## 8. The open question
Nothing blocked. Pass 2 decides the shape: how the about 0.25-0.3 page for the abstract is found within Maarten's 1.25
pages, given that the societal opening he asks for adds one or two sentences.

## 9. Next action
Pass 1: render the annotated introduction and abstract pages of the four v1 PDFs and the v2 Maarten PDF (section 13),
read every comment with its marked passage, map each to `01_introduction.tex` or `00_abstract.tex`, and post the
status list to Dirk (open and conflicts in full, done in one short block), Maarten first.

## 10. Acceptance criterion
- Every comment on the introduction and abstract, from all four reviewers, has a final status (done, or deferred with
  Dirk's reason) recorded in the closing list.
- Introduction plus abstract measure at most about 1.25 pages in a scratch review build (Maarten's figure).
- No forward reference in the introduction except the outline paragraph; no em-dashes; no open `\todo` that belongs
  to the introduction.
- Every change in the editor build is visible through the marks of section 4.

## 11. Read these first
1. `Thesis-writeup/Writing/sections/01_introduction.tex` and `00_abstract.tex` (the text and its `FEEDBACK` comments).
2. `Thesis-writeup/Feedback/v2/` (Maarten: PDF pages 2-3 and the mail).
3. `Thesis-writeup/Feedback/v1/` (Jan-1, Jasper-1 with his mail, Quinten-1: their introduction pages).
4. `Thesis-writeup/Writing/util/format.tex` lines 62-100: the change-mark macros.
5. `docs/decisions.md` D-273 (restructure) and D-241 (research question and aspects).

## 12. Do not
- Do not edit a paragraph before Dirk has read it and approved the change.
- Do not rewrite Dirk's sentences wholesale; do not add text without `\kwtwo{}`.
- Do not compile into `build/`; do not change `util/format.tex`; no git write operations.
- Do not edit other sections (section 2).

## 13. Operational
- Render annotated pages (PyMuPDF 1.27 in the env): write a script to the scratchpad that opens each PDF in
  `Thesis-writeup/Feedback/v1` and `v2`, and for each page with non-link annotations saves
  `page.get_pixmap(dpi=130, annots=True)`; also dump each annotation's content and marked text
  (`page.get_textbox(quad rect)`); write output with `open(..., encoding='utf-8')` inside the script, because the
  console encoding fails on Greek letters. Run with `conda run -n GraduationProject python -I <script>`.
- Scratch build: copy `Thesis-writeup/Writing` without `build/` to the scratchpad
  (`tar --exclude=./build -cf - . | (cd <copy> && tar -xf -)`), then in the copy
  `export BIBINPUTS="$(pwd -W);" BSTINPUTS="$(pwd -W);"; latexmk -f -pdf -interaction=nonstopmode main.tex` and the
  same for `review.tex`; errors are lines starting with `!` in `build/<name>.log`. Measure section starts with
  PyMuPDF `get_toc(simple=False)`.

## 14. Delegation
None: the files are known and the work is a dialogue with Dirk.
