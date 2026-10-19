# Handoff: plan how the thesis paper goes from about 15 to 12 pages and states its problem more clearly, grounded in the v1 supervisor feedback
**From**: session of 2026-10-19 | **Branch**: Augmentation | **Effort suggested**: high (judgment on structure and on what the story needs; every decision is Dirk's)

## 1. Task
Prepare and lead a discussion with Dirk on shortening the thesis paper
(`Thesis-writeup/Writing/main.tex`, sections in `Thesis-writeup/Writing/sections/`) from its
current length to the 12-page maximum, and on outlining the problem it solves more clearly
(Jan Hoekstra's main point). Work in three gated steps:
(1) READ the v1 feedback yourself, from the original files, in full: both mail files and every
annotated page of the four PDFs rendered as images (section 13). Do not work from section 5 of
this file; it is the previous session's reading and is there only so you can spot where you and
it disagree.
(2) READING: post in chat, per reviewer, what their feedback implies for length and for the
problem outline, each point with the verbatim comment (Dutch as written) and the page of the
reviewed PDF. Then map each to the CURRENT text (section, `file:line`) and say whether the
current text already addressed it. Wait for Dirk to confirm the reading.
(3) PLAN: post a page budget per section of the current review build (measured, section 13) and
a numbered list of candidate cuts, each tagged MOVE TO APPENDIX, CONDENSE, or DELETE, with the
estimated page saving, the feedback comment that supports it (or "no feedback, own proposal"),
and what the story loses. Lead with the problem-outline change, because it decides what the rest
of the paper must carry. Wait for Dirk's answer per item. This session discusses; it edits a
`.tex` file only for items Dirk explicitly approves for implementation here.

## 2. Out of scope
- Implementing cuts Dirk has not approved item by item. A general "sounds good" on the plan is
  not approval to edit; ask which items to implement.
- Processing the v1 line comments (grammar, notation, typos) that do not affect length or the
  problem outline. Sections I to III were already processed in earlier sessions (commits
  `963264c`, `07f5b20`, `66f1569`, `5713cb3`, `a44a15c`); list any you find still open, do not fix them.
- Feedback round 2 (`Thesis-writeup/Feedback/v2/` holds the PDF sent out, no comments yet).
- New results, new runs, new figures from data. Moving or replotting an EXISTING figure into the
  appendix is in scope as a proposal (Jasper suggested it).
- `Thesis-writeup/Writing/build/` (Dirk's editor rebuilds on save), `kamtin-fp-model/`, memory
  files, git write operations, `util/format.tex` change-mark macros (use them, do not redesign).

## 3. Where things stand
- Last commit `4271a31` "Make ready for thesis feedback round 2" (2026-10-19). `Thesis-writeup/`
  had no uncommitted changes at handoff time. Dirk edits the thesis while you work: re-read a
  file right before any edit.
- Current review build `Thesis-writeup/Writing/build/review.pdf` (identical in page layout to
  `Feedback/v2/*.pdf`): 23 pages. Section V starts p10, VI Results p14, VII/VIII and References
  p15, Appendix p17. So the body plus references is about 16 pages, body alone about 15 (Dirk's
  "15 pages"). Measured with pypdf text search, so section starts are approximate to the page.
- The v1 PDFs show OLDER builds (10 to 14 pages). Jasper reviewed an older build than Jan:
  equation numbers differ between them. Map comments to the current text by wording, never by
  equation number, and list a comment whose text no longer exists as such.
- Nothing running.

## 4. Established and verified
- Feedback files, all in `Thesis-writeup/Feedback/v1/`:
  - `Feedback-Jan-General-1-and-2.txt` (mail, Dutch): on Section III; technical details and
    figures fine, more emphasis needed on why things are done, to make it one whole; Jan had
    not yet checked III-E (added-state initialisation).
  - `Feedback-Jan-1-...pdf` (10 p, annotations on p1 to p4: Sections I to III-A) and
    `Feedback-Jan-2-...pdf` (14 p, annotations on p5 to p7: Sections III-B to III-D).
  - `Feedback-Jasper-1-...v2.pdf` (13 p, annotations p1 to p10 and p12) and
    `Feedback-Jasper-General.txt` (mail, Dutch, five numbered points; point 4 refers to a figure
    embedded in his mail that is NOT in the txt file, ask Dirk which figure).
  - `Feedback-Quinten-1-...pdf` (10 p, annotations p1 to p4, p7, p9).
- The annotations were extracted as text with pypdf (about 270 entries). Text extraction loses
  context: highlights without a comment, carets, strike-outs, and the exact passage marked. That
  is why step 1 renders the pages.
- Earlier per-section feedback handoffs hold verbatim comment lists with IDs (Jan1, Js3, ...):
  `tasks/handoffs/2026-10-18-section2-feedback.md` (section 4) and
  `tasks/handoffs/2026-10-18-section3-feedback.md`. Reuse their IDs so Dirk sees one numbering.
- The Writing README calls the paper "the 12-page IEEE paper" (`Thesis-writeup/Writing/README.md:3`).

## 5. Previous session's reading (cross-check only, not a source)
Length and problem-outline points as the previous session read them:
- Jan: problem statement missing; system and baseline are mixed, no dataset D_N, no
  identification objective; order should be system plus identification objective, then baseline,
  then augmentation (Jan-1 p2 sticky notes). Closed-loop windows and encoder belong in the problem
  statement (p3). Introduction: start from control design, then digital twin, then the need for an
  accurate model; after what the method does, say why it is chosen (p1). "If you only use
  parallel augmentation you can cut the LFR explanation" (p4). Minimality remark: expand or cut
  (p3). "Keep to the core story" (Jan-2 p6). Non-uniqueness discussion belongs in Section IV (p7).
  Cost function before encoder, closed-loop identification first (Jan-2 p5).
- Jasper: experiment set too complicated, leave out what the story does not need (do all Table
  I models serve a contribution?); explain experiment design by plots in an appendix instead of
  text; Sections V to VII read like notes, "a few less impressive results explained well" over
  all results (p9); results look strong in the figure but not in Table I, relate them to a
  realistic RMS; show parameter recovery working without added states (adds content).
- Quinten: asks for hyperparameters, train/val/test RMS, overfitting, and why the black box is
  worse than the detuned baseline in interpolation. These ADD length; the plan must say where
  they go (appendix table is one candidate).
Reconcile your own reading with this list in step 2; where they differ, yours from the source wins.

## 6. Tried and failed
- pypdf annotation text extraction as the only reading -> misses unannotated highlights and the
  marked passage for many entries -> PDF markup is positional, not textual -> step 1 renders pages.

## 7. Achieved
None for this task.

## 8. The open question
Does the 12-page maximum include references and appendices? Candidate answers: (a) body only,
references and appendix extra; (b) body plus references, appendix extra; (c) everything. It sets
the cut size (about 3, about 4, or about 11 pages). Ask Dirk in step 2; do not assume.

## 9. Next action
Render every annotated page of the four v1 PDFs (section 13), read each image and both mail
files, then post the step-2 reading in chat.

## 10. Acceptance criterion
Step 3 is done when Dirk has a numbered cut list whose estimated savings sum to at least the
measured overshoot (current page count minus 12, under the definition chosen in section 8), every
item has a decision from Dirk, and the problem-outline change is agreed. Savings are estimated
from the measured page budget, not guessed.

## 11. Read these first
1. `Thesis-writeup/Feedback/v1/` all six files, rendered (the substance of this task).
2. `Thesis-writeup/Writing/build/review.pdf` (what the supervisors see now).
3. `Thesis-writeup/Writing/sections/01_introduction.tex` and `02_system_baseline.tex` (where the
   problem statement must live).
4. `tasks/handoffs/2026-10-18-section2-feedback.md` section 4 (comment IDs, how Jan23 was handled).
5. `Thesis-writeup/Writing/sections/05_setup.tex` and `06_results.tex` (Jasper's main length target).

## 12. Do not
- Do not treat section 5 as the feedback.
- Do not edit `.tex` files without per-item approval; do not compile into `build/`.
- Do not propose cutting content a supervisor explicitly asked for without saying who asked.

## 13. Operational
- Render (PyMuPDF 1.27 is in the env): write to the scratchpad and run
  `conda run -n GraduationProject python -I <scratchpad>/render.py`:
  ```python
  import fitz, glob, os
  src = r"Thesis-writeup/Feedback/v1"; out = r"<scratchpad>/v1png"; os.makedirs(out, exist_ok=True)
  for f in glob.glob(os.path.join(src, "*.pdf")):
      d = fitz.open(f); tag = os.path.basename(f).split("-Model")[0]
      for p in d:
          if any(a.type[0] not in (1, 16) for a in p.annots()):  # skip Link/Popup-only pages
              p.get_pixmap(dpi=130, annots=True).save(f"{out}/{tag}-p{p.number+1}.png")
  ```
  Then Read each PNG. Use a relative path from the repo root or quote absolute ones (spaces).
- Page budget per section: open `build/review.pdf` with PyMuPDF and locate each section heading
  (`p.search_for("II. ")` etc., or the TOC via `d.get_toc()` if hyperref bookmarks exist) to get
  the fraction of a page each section takes.
- Test builds, only if an approved edit is made: the scratch-copy recipe in
  `tasks/handoffs/2026-10-18-section3-feedback.md` section 13.

## 14. Delegation
None. All reading is targeted and the judgment must stay in the session that talks to Dirk.
