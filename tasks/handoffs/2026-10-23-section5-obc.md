# Handoff: finish thesis Section V (Interpretability by Orthogonal Construction) with Dirk, paragraph by paragraph, on supervisor feedback, including the math
**From**: Section II session (2026-10-20 to 10-23) | **Branch**: Augmentation | **Effort suggested**: high (the section is mostly derivation; judging comments and checking each equation against Gyorok and the code; replies stay short)

## 1. Task
Finish `\section{Interpretability by Orthogonal Construction}` (`Thesis-writeup/Writing/sections/04_obc.tex`, `\label{sec:obc}` at `:75` to the end: intro `:75-108`, V-A `sec:obc_construction` `:109`, V-B `sec:obc_map` `:155`, V-C `sec:obc_ident` `:302`, V-D `sec:obc_measures` `:396`; line numbers of 2026-10-23) with Dirk, the way Sections I and II were finished. Numbering: this is Section V; the experiment design (`05_setup.tex`) is Section VI. First read the supervisor feedback on this section yourself, from the original PDFs and in its own context (method in 13): Maarten's v2 note on his p11 and his later notes that point back to OBC (p14, p15, and p5 on uniqueness), Jasper's eleven v1 notes on his p8, Jan's v1 note on his Jan-2 p7, and the mails (4.1). Unlike Sections III and IV, almost nothing here has been handled yet (only three FEEDBACK or RESTRUCTURE comments in the file). Post your reading of the comments to Dirk in one short block for confirmation. Then walk the section in reading order, one paragraph per turn: the paragraph as it reads in the final text (marks applied), its comments with status, your own findings (undefined symbols, repeats of Sections II to IV, forward references, AI-sounding phrasing, claims the papers do not support, notation that no longer matches Sections II and IV), and the smallest fix as an exact replace-with LaTeX block with change marks. Apply only on Dirk's explicit "yes", "apply", "do it". When Dirk asks whether an equation is right or matches the paper, open the paper in `Thesis-writeup/References/` and the code, and answer with the paper's equation number and PDF page, symbol by symbol. End with a closing list: every comment done, or deferred with Dirk's reason.

## 2. Out of scope
- Sections I to IV (`01_introduction.tex`, `02_system_baseline.tex`, `03_augmentation.tex`): finished or in other sessions (Section IV: `tasks/handoffs/2026-10-22-section4-dynamic-augmentation.md`, which may be editing `03_augmentation.tex` now). Edit them only if a Section V decision changes a term they use, and only after Dirk says so in that message.
- Sections VI onwards and the appendix: edit only when Dirk says so in that message. Maarten's p14 and p15 notes sit on Results and Conclusion: decide with Dirk whether Section V must carry something for them; the Results text itself is not yours.
- Results numbers and the `\revnote` "Final OBC results pending (polish reruns ...)" (`:334`): data, not text work (see `memory` project_final_run_priority).
- `build/`, `util/format.tex`, git writes, thesis LaTeX builds, the `\ifdraft` bullet blocks.

## 3. Where things stand
Branch `Augmentation`, last commit `e3a91c9` "Update section 4-E". `Thesis-writeup/Writing/sections/` clean at handoff (only `build/` modified). Dirk edits in parallel and commits himself: re-read a paragraph right before editing it, and check that a block does not exist twice (it happened once in Section II). No runs in flight.

## 4. Established and verified

### 4.1 Comments that concern Section V
Verified: verbatim text and page (annotation dumps of 2026-10-20 and 2026-10-23). NOT verified: which current paragraph each sits on (the text changed since), and the "Reading" column. Map by wording, never by line or equation number.

**Maarten v2** (`Thesis-writeup/Feedback/v2/Feedback-Maarten-Main-supervisor-v2-...pdf`; final say). In his build Section IV (OBC) is p10 to p12; p10 and p12 carry no notes.
| Page, position | Verbatim | Reading (mine) |
|-|-|-|
| p11 left, highlight "them" in "evaluate the Jacobian at these positions with the velocities reconstructed from them" (now `:225`) | "velocities are reconstructed directly from these measured positions." | unclear referent: rewrite the clause so "them" is named |
| p15, note beside the Discussion sentence on the truth overlap and the recovery condition | "and OBC will only recover the true values if the original system had missing dynamics that are orthogonal on the baseline." | the recovery condition of V-C (`eq:obc_recovery`, Phi^T Xi = 0) must be stated as exactly this |
| p15, highlight "unique estimate" in the Conclusion | "what do you mean by this? OBC introduces uniqueness, just not the same as the physical interpretation you started from." | uniqueness (what OBC gives) differs from the true values; V must say which unique vector OBC selects |
| p14, notes on Results B | "reuse the notation introduced in Table I, also Mu is missing here?"; "Mu, MOBC?"; "what is worth showing here is that without OBC, we can every time converge to different values. With OBC we should be much more clustered." | Results; check that V names the models the way Results does |
| p5, notes on the identifiability argument (now III-C) | "To keep it short, I'd just talk about uniqueness in the parameters, explain why the original case is not unique, and what unique vector you have selected." | III-C's session; for V: say which unique vector OBC selects, consistent with III-C |
Mail (`Feedback-Maarten-General-v2.txt`): order and narrative, no forward references, "het kan wel wat korter", appendix not read.

**Jasper v1** (`Thesis-writeup/Feedback/v1/Feedback-Jasper-1-...v2.pdf`, p8, his Section IV "Orthogonal correction of the physical rows" and "Identification of the corrected model"; marked text in quotes). Not in the 2026-10-18 index except the first (indexed there as Js122, wrongly under III-E); give the rest IDs JsV1 to JsV10 in this order.
| ID | Marked | Verbatim |
|-|-|-|
| Js122 | "set P" | "which set P? refer to 28?" |
| JsV1 | "β" | "I still don't understand what is beta, some kind of internal representation of the parameters or what?" |
| JsV2 | "∂β" | "presumably this is actually a jacobian, since beta is a vector? so you have to write it properly with einstein summation or so?" |
| JsV3 | "ki" | "what is i?" |
| JsV4 | "Φ" | "why is this bold? it's a column right? so it's not a matrix or so. and other columns are not bold?" |
| JsV5 | "I" | "what is I?" |
| JsV6 | "I 0" | "is this one zero? or multiple? if it's multiple you need to make that clear somehow." |
| JsV7 | "f n obc" | "what does f^n_{obc} represent?" |
| JsV8 | "θaux" | "what does this represent?" |
| JsV9 | "MOBC is MPS" | "what is M, what is M_{PS}" |
| JsV10 | "correction" | "which correction?" |
Jasper's mail point 1: "Er zijn vaak variabelen die niet duidelijk geintroduceerd zijn" (this section is where most of his undefined-symbol notes sit). Point 5: "Misschien kun je nog laten zien dat de parameter recovery wel werkt voor het model zonder extra states?" (Results; mention to Dirk, do not act).

**Jan v1** (`Feedback-Jan-2-...pdf` p7, on old III-E): highlight "Without enforcing uniqueness the estimated theta_base is not interpretable as defined in Section I. Section IV ..." with "Write this in Section IV. Distracts from story here". So the motivation "without uniqueness theta_base is not interpretable" belongs in this section's opening; check it is there and not repeated in IV. Jan's mail: every subsection says why it exists.

### 4.2 Decisions elsewhere that Section V must stay consistent with
1. Interpretability is defined in the Introduction (`01_introduction.tex:246-247`): "the identifiable combinations of the physical parameters are estimated uniquely" (Gyorok OBC); it does not guarantee the true parameters (Js15). Section II's goal sentence (2026-10-22): estimate the physical and learned parameters from D_N so that the model captures the missing dynamics and predicts accurately, "while the identifiable combinations of the physical parameters are estimated uniquely".
2. Section II states the setting at the single rate T_s with the controller $K_{\mathrm{fb}}$ (matrices $A_{\mathrm{fb}},\dots,D_{\mathrm{fb}}$), the zero-order hold, measured positions $y_k=\qact(kT_s)+v_k$; Section VI names the data's deviations; `f_s` was renamed `f_sim` (20 kHz) because $1/T_s$ is 4 kHz.
3. Section IV uses the normalised quantities $\tilde x^{\mathrm n}$, $u^{\mathrm n}$ (eq:aug_norm), the training coordinates $\beta$ (III-C), the residual closed loop `eq:aug_residual_loop` and the window notation $k|\tau$. V-B builds its tuples from measured positions with a finite-difference velocity (`:225-229`, also `04_obc.tex:229` uses $T_s$).
4. SUBNET is introduced in Section II; IV-C states its two assumptions. Do not reintroduce SUBNET here.

### 4.3 Dirk's working requirements (held 2026-10-20 to 23)
Maarten has final say; feedback read from the original files, in context. Dirk's wording first; changes visible with `\chg[tag]{old}{new}`, `\del`, `\add`, `\mv[tag]{from}{..}` with `\delmv[tag]{to}{..}` (`util/format.tex`); tag = comment ID, `own`, or `Dirk`; no marks in captions, figures or inside math; no marks nested in struck text (`\mbox` around `\ref`). Every change gets a source comment naming the comment and page.
His style: short sentences, few commas, "we" structure, no semicolons if avoidable, plain verbs, no colon-plus-which tails, no "denote". Show proposals as "Replace: ... with: ..." against the current file plus how it reads. One recommendation per turn; say plainly when you change your view. A reply that worked:
> One fix: the comma after $\mathcal D_N$ joins two full sentences. Your original sentence had the word that fixes it, "where":
> ```latex
> ... on short windows of $\mathcal D_N$, where each window is simulated with the controller $K_{\mathrm{fb}}$.
> ```

### 4.4 Papers and code for the math
`Thesis-writeup/References/` (`index.csv` cite key to file): Gyorok et al. 2026 OBC (`gyorok2026obc`, the orthogonal-by-construction method; Sec. 3.2 on non-unique estimates losing physical meaning), Gyorok et al. 2025 L4DC orthogonal projection regularisation (`gyorok2025l4dc`, `Gyorok-2025-Orthogonal-projection-regularization.pdf`; it does NOT use SUBNET or an encoder: measured states), Hoekstra 2026 (`hoekstra2026lfr`, Sec. 5.2 regularisation, PDF p9). The source comments at the top of `04_obc.tex` (`:1-74`) name the decisions D-252 to D-255 and what each equation adapts. Code: the OBC projection in `Clean/` (grep `theta_aux`, `obc`) and `docs/decisions.md` D-252 to D-255.

## 5. Assumed but not verified
- The readings and current locations in 4.1. Settled by reading each note beside its passage.
- That V-C states the recovery condition the way Maarten's p15 note demands (true values only if the missing dynamics are orthogonal to the baseline), and says which unique vector OBC selects (p15 and p5). Settled by reading V-C and the Discussion.
- That every symbol Jasper asked about (beta, i, I, Phi, f^n_obc, theta_aux, M, M_PS, "correction") is now defined before use, or still is not. Settled by the walk.
- That the `\kwtwo` title "Orthogonality measures" (V-D, added 2026-10-20) and its symbols match Results. Settled by grep in `06_results.tex`.

## 6. Tried and failed (Section II session; transfer the lessons)
- Several rewordings in a row -> Dirk lost track -> one recommendation per turn, with the reason.
- Claims about how the code or data work, made before reading them -> wrong proposals withdrawn -> read the code and the setup text before claiming.
- Claimed a later section "builds on" an argument that had been struck -> grep the live text before saying what a section says.
- Drafts with many commas, semicolons, "denote", "This is the ..." -> "not my style" -> 4.3.
- Text read from a handoff instead of the PDFs -> misread feedback (introduction sessions) -> read each note beside its passage.

## 7. Achieved
Nothing in Section V was walked yet. Sections I and II are finished; Section IV is in progress in another session.

## 8. The open question
How far should Section V go in answering Jasper's undefined-symbol notes against Maarten's "korter"? Candidates: (a) define every symbol in the main text where it first appears (longer, clearer); (b) keep the main text to the construction and the recovery condition, and move the tuple stacking and the polish rules to the appendix. Evidence that chooses: how many of Jasper's symbols are still undefined (5), and Dirk.

## 9. Next action
Run the dump in 13 on Maarten's v2 PDF (p11, p14, p15, p5) and Jasper's v1 PDF (p8), place each note on its current paragraph in `04_obc.tex`, and post to Dirk your reading of each with its status (4.1 is my unconfirmed version). After he confirms, post the Section V intro paragraph (`:75-108`) as it reads in the final text.

## 10. Acceptance criterion
Every comment in 4.1 has a final status (done, or deferred with Dirk's reason) in the closing list; every symbol in the live text of Section V is defined before its first use; Section V contradicts nothing in 4.2; no forward reference in its live text; no em-dashes; no open `\todo` unless Dirk defers it; every change visible through the marks.

## 11. Read these first
1. The feedback in context (13): Maarten v2 p11, p14, p15, p5; Jasper v1 p8; Jan-2 v1 p7.
2. `04_obc.tex` in full, with the source comments at its top (`:1-74`).
3. `01_introduction.tex:240-250` (interpretability definition) and Section II's goal sentence (`02_system_baseline.tex`, end of Section II).
4. The Gyorok OBC paper in `Thesis-writeup/References/`, at the equations V-A adapts.
5. Section 4.2 above.

## 12. Do not
- Do not edit before an explicit yes; a pasted phrase or a question gets feedback only.
- Do not run a thesis LaTeX build.
- Do not mention or edit the `\ifdraft` bullet blocks.
- Do not change results numbers or act on the polish-rerun revnote.
- Do not cut a sentence without naming the comment that asks for it or the nearby sentence it repeats.
- Do not reword Dirk's sentences beyond about five words; new text in `\kwtwo`.

## 13. Operational
Feedback dump (one quick script, not a build): write to your session scratchpad a PyMuPDF script that opens the feedback PDFs and, for every page with non-link annotations, saves `page.get_pixmap(dpi=130, annots=True)` as PNG and writes per annotation: page, type, rect, author, `info["content"]`, and the marked text (`page.get_textbox(fitz.Quad(vertices[i:i+4]).rect)` for highlights and strike-outs; for sticky notes the text in a rect about 200 pt either side of `annot.rect`), to a file opened with `encoding="utf-8"` (the console is cp1252: never print PDF text to stdout). Run with `conda run -n GraduationProject python -I <script> <args>`. PDF y in points times 130/72 gives the PNG y. For an equation in a paper, render that page as PNG (dpi 150) and read the image; quote PDF pages.

## 14. Delegation
None: the files are known and the work is a dialogue with Dirk. If a claim about code versus thesis needs an independent check, Dirk may ask for a read-only second-opinion session.
