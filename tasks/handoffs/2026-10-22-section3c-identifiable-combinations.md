# Handoff: rework thesis III-C ("Physical parameters and identifiable combinations") with Dirk, starting from Maarten's feedback read in its own context
**From**: III-B session of 2026-10-22 | **Branch**: Augmentation | **Effort suggested**: high (judging an identifiability argument against supervisor feedback and the papers; replies stay short)

## 1. Task
Rework `\subsection{Physical parameters and identifiable combinations}\label{sec:params}` in `Thesis-writeup/Writing/sections/02_system_baseline.tex` (from `\label{sec:params}`, about `:683`, to the end of the file) with Dirk. Maarten's p5 notes question the argument itself, not its wording, so content comes before sentences. First read Maarten's v2 feedback yourself in its PDF context (method in 13): render p5 (and p4, p14, p15) as images with the annotations, read the passage beside each note, and form your own reading of every comment. Then post to Dirk, in one short block: (i) your reading of each Maarten comment that bears on III-C, quoted verbatim, with what it asks for; (ii) what the current III-C argues, step by step, and which step each comment hits; (iii) one recommended argument structure for III-C. Wait for Dirk. After he agrees the structure, go paragraph by paragraph: the final text with marks applied, the comments that belong to it, your findings, the smallest fix in Dirk's wording as a marked LaTeX block, then "Apply?". Use the `thesis-section` skill for structure (Hoekstra-style: problem, construction, consequence). Apply only on Dirk's explicit "apply" / "yes, apply".

## 2. Out of scope
- III-A, III-B and Appendix A.2 (`app:lfr-details`): finished on 2026-10-21/22 (III-B and A.2 committed in `af25d9c`). Edit only a cross-reference that a III-C change breaks, and say so.
- Section II, Section I, Sections IV onwards: other sessions. Edit only when Dirk says so in that message. If III-C's new argument needs a change elsewhere (Section I's interpretability definition, the conclusion's "unique estimate", IV-C's joint-estimation sentence), list it and ask.
- The `\ifdraft` draft bullets: never mention or edit them (Dirk 2026-10-22).
- LaTeX builds, git writes, `kamtin-fp-model/`.

## 3. Where things stand
Branch `Augmentation`, last commit `af25d9c` ("Fix section 3-B ..."); thesis sections clean at handoff time. Dirk edits the same file in his editor during the session: re-read the passage right before every edit. No runs in flight.

## 4. Established and verified

### 4.1 Maarten v2 comments that bear on III-C (verbatim; from `Thesis-writeup/Feedback/v2/Feedback-Maarten-Main-supervisor-v2-...pdf`, read 2026-10-22)
In his build III-C was "II-C" and Section IV was "III"; map by wording, never by number.
- **p5, four notes on one icon**, left margin beside the identity `(M(Y)-M'(Y))q'' + (C-C')q' + (K-K')q = 0` and the sentence before it ("Two parameter sets theta and theta' that produce the same motion for every input force satisfy"):
  - (a) "reference for this statement? This is a large jump from the previous statement. Actually, don't you also need the acceleration to be known too?"
  - (b) "i.e. state derivative."
  - (c) "Finally, identifiability should start from model structure + cost function that you consider. I don't think you use the velocity and acceleration in your cost at any point. So this argumentation confuses me, I'd need to understand this better."
  - (d) "To keep it short, I'd just talk about uniqueness in the parameters, explain why the original case is not unique, and what unique vector you have selected."
- **p4**, highlight on "Appendix A-A." (nominal values sentence): "A-A?" (appendix numbering; main.tex now numbers A.1, A.2; III-C still has Dirk's todo "needs to have appendix in the name of the table").
- **p14**, on the parameter-result paragraph: "what is worth showing here is that without OBC, we can every time converge to different values. With OBC we should be much more clustered." (uniqueness as Maarten means it: repeatable estimates).
- **p15**, highlight on "unique estimate" (conclusion): "what do you mean by this? OBC introduces uniqueness, just not the same as the physical interpretation you started from." And on the OBC limitation sentence: "and OBC will only recover the true values if the original system had missing dynamics that are orthogonal on the baseline."
- **Mail** (`Feedback/v2/Feedback-Maarten-General-v2.txt`): rethink order and narrative, far too many forward references ("erg vaak forward references"), it can be somewhat shorter.
- Maarten's other p5 notes (right column: network inputs, normalisation) and p6 to p9 concern Section IV: not III-C.

### 4.2 v1 comments on old II-C (index: `tasks/handoffs/2026-10-18-section2-feedback.md` section 4; statuses in the `% FEEDBACK` source comments)
Jan21 ("This does not flow well in reading"), Js51 (unit), Js52 (theta-dependency in the equation), Js53 ("Some words about observability and controllability would be helpful here."), Js54, Js55, Js56 (separate inequalities), Js57/Js72 ("combinations": "base parameters"/"minimal parameter set"?), Q34 ("Define M', C' and K'"), Q35 ("Define 0,33"), Js150, Js151 (appendix). Dirk's open revnote "Do I need to state symbols like Quinten asked?" is Q34/Q35.

### 4.3 What the current III-C argues (read the file; order as of `af25d9c`)
Opening: joint estimation with the augmentation, interpretability needs unique estimates (Section I). Fourteen parameters not all identifiable. q measured via (2), forces known; L_b fixed (Js33). Full actuation and observability sentence (`\kw`). Identity in q, q', q'' for theta, theta' (unnumbered display). Hence K=K', C=C', M(Y)=M'(Y), then M_0, M_1, M_2 equal (uses `eq:Msplit` from III-B). Twelve entries, three equal m_h, ten functions, Jacobian determinant m_h L_b^2/2, "exactly ten" (Ovchinnikov Def. 7, `\todo{check source}`). `eq:phi` theta_base, Gautier base parameters (`\todo{check source}`), one-to-one for m_h>0 (`app:combos`). Positive theta_base does not give M(Y) > 0; `eq:lfr_admissibility` (proof `app:lfr-wellposed`). Parametrisation in beta (log for nine, tanh-bounded m_Delta, `eq:mdelta_map`), moved here from IV-C for the forward-reference complaint (M-mail).
The four lost directions (k_b1-k_b2, c_b1-c_b2, J_b-J_h, carriage-to-cross-arm mass transfer with the compensating J_b+J_h change) exist only in a commented block under `eq:phi`; this is the "why the original case is not unique" of note (d).

### 4.4 What III-C may build on (backward only)
- Section I (`01_introduction.tex:144-152`): "Physical interpretability means that the identifiable combinations of the physical parameters are estimated uniquely [gyorok2026obc]." III-C is the target of that definition.
- Section II: data set D_N of actuator forces and noisy measured positions (`eq:system`), closed loop with known controller, goal "the identifiable combinations of the physical parameters are estimated uniquely", SUBNET with windows and encoder.
- III-A: matrices (4a) to (4c) (`eq:matrices`), q from (2).
- III-B (finished 2026-10-22): explicit state equation `eq:base_ss`, x~' = F(x~,u) = [q'; M(Y)^{-1}(u - Cq' - Kq)], so q'' is computed from the model (this is what Maarten's (b) "i.e. state derivative." points at); split `eq:Msplit` M(Y)=M_0+YM_1+Y^2M_2 with M_0, M_1, M_2 in Appendix A.2; M(Y) > 0 for positive masses and inertias (`app:lfr-wellposed`).
- The cost Maarten's (c) asks about is defined LATER, in IV-C: `eq:aug_loss` (`03_augmentation.tex` about `:791-804`), the normalised output (position) simulation error over windows. Citing it from III-C is a forward reference; how to respect (c) without one is part of the open question (8).

### 4.5 Sources (`Thesis-writeup/References/`, `index.csv` maps cite key to file)
`Ovchinnikov-2021-Identifiable-functions-ODE-models.pdf` (Def. 7), `Gautier-1990-Minimum-inertial-parameters-serial-robots.pdf` (base parameters), `Gyorok-2026-Orthogonal-by-construction.pdf` (uniqueness and physical meaning, Sec. 3.2), `Hoekstra-2026-LFR-model-augmentation-arXiv-v1.pdf` (identification setting). Citation checks are logged in `Thesis-writeup/Documentation/CITATION-LOG.md`.

## 5. Assumed but not verified
- That note (c) means: the identifiability claim must be stated for the estimator actually used (this model structure plus a position-only output-error cost), not argued from q' and q''. Settled by reading p5 in context and by Dirk.
- That (a) and (c) are answered by one added step: equal positions q(t) for all inputs imply equal q' and q'' (derivatives of the same signal), so the identity needs no measured velocity or acceleration. Plausible, but whether Maarten accepts a noise-free structural argument, versus wanting the statement tied to the cost, is his intent; settled by Dirk.
- That Maarten's (d) "to keep it short" means the derivation (identity, Jacobian) leaves the main text, with the four lost directions and theta_base kept. Settled by Dirk.
- Whether "unique" in III-C should be separated from "physically interpretable" (p15): OBC gives a unique estimate in its own sense. Possibly a Section I or conclusion matter; settled by Dirk.

## 6. Tried and failed (sessions 2026-10-18 to 10-22)
- Readings taken from a handoff instead of the PDF -> feedback misread repeatedly -> a sticky note depends on the passage beside it -> render the page and read it in context first (13).
- Justifications that point forward ("Section IV gives ...", "obtained by one RK4 step") -> Dirk rejected, Maarten's mail criticises them -> justify only from Sections I, II, III-A, III-B.
- A reason that only restated our own choice ("as a model of (1), written in the same form") -> Dirk: circular -> a reason must come from outside the choice (a cited method, an earlier fact).
- A reason that was not true as written ("additional states and initial states both need state-space form", while (3) with q, q' already has a state) -> Dirk: "does this actually justify it?" -> test every reason for truth before proposing it.
- Rewording Dirk's sentences beyond a few words, or wordier replacements of a short correct pointer -> "you only make it worse", "doesn't feel like me" -> fixes of at most about five words, new text only where needed, in `\kwtwo{}`, his structure (short sentences, no semicolons).
- Applying after a pasted phrase or a question -> wrong -> a paste or question is feedback; only "apply" authorises an edit.

## 7. Achieved
III-B: state equation `eq:base_ss` with its reason (Hoekstra assumes a state-space baseline, Eq. (2), PDF p3), LPV and self-scheduled (Drenth) sentences, exactness sentence answering Js36, rational dependence leading to the LFR pointer; no forward references. A.2: complete LFR derivation, ends on well-posedness iff M(Y) nonsingular. Committed `af25d9c`.

## 8. The open question
What argument should III-C make, given Maarten's p5 notes? Candidates: (i) keep the structural derivation and add the missing step (equal positions for all inputs give equal derivatives, so no measured acceleration is needed), stated as structural identifiability of the model output q, which is what the position cost compares; (ii) Maarten's short form (d): the fourteen parameters are not unique (name the four lost directions), theta_base is the unique vector chosen, the derivation in one or two sentences or in Appendix `app:combos`. Evidence that chooses: Maarten's notes read in context (especially (c) "start from model structure + cost function" and (d) "To keep it short"), his mail (shorter), and Dirk's decision.

## 9. Next action
Run the feedback dump in 13 on Maarten's v2 PDF, read p5 (both columns) and p4, p14, p15 images in context, read the current III-C from `\label{sec:params}` to the end of `02_system_baseline.tex` with its source comments, then post to Dirk your reading of each Maarten comment in 4.1, what the current argument does step by step, and one recommended structure for III-C, before any LaTeX.

## 10. Acceptance criterion
Every Maarten comment in 4.1 that concerns III-C, and every v1 ID in 4.2, has a final status (done, or deferred with Dirk's reason) in a closing list; III-C makes the argument Dirk chose in 8; every citation in III-C is checked against its page (Ovchinnikov Def. 7, Gautier) or its `\todo` stays; no forward reference in live text; changes visible through the marks; Dirk says it reads clearly in his build.

## 11. Read these first
1. Maarten v2 PDF p5 (and p4, p14, p15) rendered with annotations, plus his mail: the task depends on reading these in context.
2. `02_system_baseline.tex` from `\label{sec:params}` to the end, with its source comments.
3. `02_system_baseline.tex` III-B (`\label{sec:lfr}`): what III-C may cite.
4. `03_augmentation.tex` `eq:aug_loss` (about `:791-804`): the cost Maarten's (c) refers to (read for understanding; cite it only if Dirk accepts that reference).
5. `.claude/skills/thesis-section/SKILL.md`.

## 12. Do not
- No forward references in live text (Sections IV to VII), not even "Section IV estimates ...".
- Do not reword Dirk's sentences beyond about five words; no semicolons; no new openers he did not write.
- Do not edit or mention the draft bullets.
- No edits without "apply"; no LaTeX builds; no git writes.

## 13. Operational
Feedback dump: write a PyMuPDF script to the session scratchpad that opens `Thesis-writeup/Feedback/v2/Feedback-Maarten-Main-supervisor-v2-*.pdf`, saves `page.get_pixmap(dpi=130, annots=True)` as PNG for p4, p5, p14, p15, and writes per annotation its type, `info["content"]` and the marked text (highlights: `page.get_textbox(fitz.Quad(vertices[i:i+4]).rect)`; sticky notes: text in a rect about 200 pt either side of `annot.rect`) to a file opened with `encoding="utf-8"`. Run `conda run -n GraduationProject python -I <script> ...`; write all output to files from inside Python (conda's stdout re-encoding fails on Greek letters). Then Read the PNGs. For a source equation, render that PDF page at dpi 150 and read the image, because text extraction garbles math. Mark macros: `util/format.tex:85-103` (`\chg[tag]{old}{new}`, `\del`, `\add`, `\mv`, `\delmv`; no nested marks inside struck text, `\mbox` around `\ref`/`\eqref` there). Reply shape that worked: diagnosis in one line, smallest fix as a marked LaTeX block, "Apply?".

## 14. Delegation
None: known files, a dialogue with Dirk.
