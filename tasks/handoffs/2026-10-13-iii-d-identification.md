# Handoff: write thesis subsection III-D "Identification" with Dirk, concise, written for the final runs
**From**: session of 2026-10-13 | **Branch**: Augmentation | **Effort suggested**: high (math, code and sources must agree; prose for Dirk to rewrite)

## 1. Task
Rewrite subsection III-D (`Thesis-writeup/Writing/sections/03_augmentation.tex`, currently
`\subsection{Identification problem}\label{sec:aug_closed_loop}`, l.754 to 896, up to
`\subsection{Added-state initialisation}`) with Dirk, using the `thesis-section` skill. Rename the title to
"Identification" (Dirk, 2026-10-13); keep the label `sec:aug_closed_loop` (Sections II, III and IV reference it).
Do skill steps 2a and 2b with Dirk before drafting: start 2a from the five claims and the cuts in sec. 4 below,
show only what you would change, and wait for his answer; then the paragraph outline; then draft. The current text
is a draft by an earlier session, not Dirk's prose, so it may be rewritten freely; once Dirk starts pasting his own
versions, keep his sentences (sec. 12). The result must be more concise than the current text with nothing
required removed, and describe the final runs (`Thesis-writeup/Documentation/FINAL-RUN-FIXES.md`), with every
difference from the 2026-10-11 runs as a `\todo`, as III-E does.

## 2. Out of scope
- III-C (`sec:aug_loop`) and III-E (`sec:aug_ps2`): under review in a separate session
  (`tasks/handoffs/2026-10-13-iii-c-iii-e-review.md`). Do not edit them. III-D must keep the symbols, labels and
  facts they use: `eq:aug_loss`, `eq:aug_init`, $\vartheta=(\xi,\theta_{\mathrm{aug}},\theta_{\mathrm{enc}})$,
  $\mathcal T$, $n_f$, the free-run validation (whole records, one encoder start, $x^c=0$ once), $\mathcal M_{\mathrm U}$.
- III-A, III-B, Sections I, II, IV to VIII, the appendix: read only. A needed change there is proposed in chat
  and made only on Dirk's request.
- Renaming the label `sec:aug_closed_loop` (cosmetic, touches every `\ref`).
- `refs.bib`, `util/format.tex`, `main.tex`, `.latexmkrc`, `.vscode/`, `build/`. Code changes and runs.
- New docs. `Thesis-writeup/Documentation/CITATION-LOG.md` gets a row per passage read.

## 3. Where things stand
- Last commit `b32b9ca`; `Thesis-writeup/` clean at handoff, but Dirk edits III-C in parallel: read the file as it
  is at session start.
- III-D has draft bullets (`\ifdraft`), `outline` lines and `wip` prose around three displays: `eq:mdelta_map`
  (l.~782), `eq:aug_loss` (l.~807), `eq:aug_init` (l.~867). Open todos: polish acceptance meters
  (FINAL-RUN-FIXES item 2), attribution of the zero output layer (D-237, D-240).
- Section II-C hands the $m_\Delta$ map and its justification to this subsection ("the map and its justification
  are in Section~\ref{sec:aug_closed_loop}", `02_system_baseline.tex` l.~360 to 363).

## 4. Established and verified (agreed with Dirk, 2026-10-13)
Claims III-D must establish (the starting point of 2a):
1. $\theta_{\mathrm{base}}$ is estimated jointly with network and encoder, as in Hoekstra 2026 Eq. (22); Kessels
   keeps the first-principles parameters fixed (Eq. (5.12)), which is the gap the Introduction names (l.~187).
2. Training coordinates $\xi$: nine positive combinations logarithmic, signed $m_\Delta$ through `eq:mdelta_map`,
   so every iterate gives $M(Y)\succ0$ without clamp or penalty (this work, D-229).
3. Criterion `eq:aug_loss`: Hoekstra Eq. (22) adapted in two respects (loop of III-C; encoder reads sample $\tau$);
   controller fixed; a closed-loop fit can mask plant error, tested in Results (this masking sentence stays in
   III-D, Dirk 2026-10-13).
4. Optimiser and selection: Adam, then an L-BFGS polish (adapted from Drenth Sec. 5, which uses bounded L-BFGS-B;
   ours needs no bounds because every $\xi$ is admissible); checkpoint on the closed-loop validation error over
   whole records.
5. Initialisation `eq:aug_init`: the model reproduces the baseline at the start (Hoekstra Eq. (29));
   $\mathcal M_{\mathrm U}$ named, input $u_{\mathrm{act}}$, output $q_{\mathrm{act}}$.
Agreed cuts:
- Non-uniqueness paragraph (l.~835 to 843) repeats the Introduction, which defines interpretability as a unique
  estimate (`01_introduction.tex` l.142 to 143, citing Gyorok). Shrink to two sentences: the split is not unique
  (Hoekstra Sec. 5.2), so $\theta_{\mathrm{base}}$ is not interpretable in the sense of Section I; Section IV makes
  it unique.
- Delete "the controller matrices are fixed, but ... the gradient passes through the controller recursion"
  (optimisation mechanics; follows from the display).
- Optimiser paragraph: drop "so no part of the window is truncated from the gradient"; split the selection
  sentence.
- "...but not positive raw masses": candidate to cut (raw masses are not identifiable, II-C).
- The where clause of `eq:aug_init` defines symbols: tighten, do not remove.
Verified sources this session (rows in `CITATION-LOG.md`): Hoekstra 2026 Eq. (22) p. 8 (joint parameter vector
includes $\theta_{\mathrm{base}}$; encoder reads up to $k-1$), Sec. 5.4 p. 9 to 10 (Eqs. (29) to (31)); Kessels
Eq. (5.12) p. 156 (only NN weights trained).

## 5. Assumed but not verified
- `eq:aug_init` for the final runs: the current display sets the whole output layer to zero, so
  $g_{\mathrm{aug}}=0$. D-248 (`docs/decisions.md` l.7771) and FINAL-RUN-FIXES item 6 make the added-state rows
  start live for the final runs; III-E's todo says `eq:aug_init` changes accordingly. Read D-248 and item 6 and
  the runner default before writing the display.
- Polish acceptance: FINAL-RUN-FIXES item 2 removes the oracle meter `combo_err` (and possibly `param_loss`);
  check `model_augmentation/fit_systems/lbfgs_polish.py::decide` (l.94) and
  `scripts/gantry/gantry_dynamic/training.py::run_lbfgs_polish` (l.702) for what the final runs accept on.
- The $m_\Delta$ map: `model_augmentation/fit_systems/blocks.py::_m_diff_bound` (l.1385) and
  `combinations_from_free` (l.1397), D-229 (l.7179); confirm $\rho$, $\eta_0$, $s$ match `eq:mdelta_map` and
  Appendix `app:lfr-wellposed`.
- Drenth Sec. 5 (Adam then L-BFGS-B), Hoekstra Sec. 5.2 (non-unique split), Gyorok 2026 Sec. 3: logged
  2026-10-09, not re-read this session; read the passages before citing.
- Attribution of the zero output layer (todo l.~890): open question for Jan, not for this session to settle.

## 6. Tried and failed
- Proposals ending in `% CHANGED` / `% NEW` comment out the rest of the line and broke Dirk's build twice. Give
  pasteable lines without markers.
- A `\\` as the first line of a paragraph, or a nested `\todo{\todo{...}}`, stops latexmk and leaves "??".
- Rewriting Dirk's sentences when he asked to "improve" was rejected repeatedly on 2026-10-13. See sec. 12.
- Stating source content from unread pages had to be retracted on 2026-10-13. Every claim about a source names
  the page read this session.
- "Identification algorithm" as title: rejected by Dirk (sounds generated). Use "Identification".

## 7. Achieved
None for III-D. III-C written with Dirk on 2026-10-13 (displays `eq:aug_two_loops`, `eq:aug_residual_loop`,
`eq:aug_cl_step`); III-E finished 2026-10-12/13.

## 8. The open question
Nothing blocked. Decide with Dirk in step 2a, from sec. 4.

## 9. Next action
Read `03_augmentation.tex` in full (header MUST ESTABLISH blocks, III-A to III-E), `01_introduction.tex` and
`02_system_baseline.tex` II-C, then the code and decisions of sec. 5, then present the 2a list to Dirk as changes
against sec. 4.

## 10. Acceptance criterion
III-D is titled "Identification", carries the three displays (`eq:mdelta_map`, `eq:aug_loss`, `eq:aug_init`) in
the final-run form with every difference from the 2026-10-11 runs as a `\todo`; each display matches the code
named in sec. 5; no sentence repeats the Introduction, II-C or III-C; every `\cite` locator was read in the PDF this
session and logged; labels used elsewhere still resolve; a scratchpad copy of `Writing/` builds `main.tex` and
`review.tex` with no `!` lines.

## 11. Read these first
1. `Thesis-writeup/Writing/sections/03_augmentation.tex` in full, with the header MUST ESTABLISH blocks.
2. `.claude/skills/thesis-section/SKILL.md` and `Thesis-writeup/Writing/README.md` ("Math standard", III-C and
   III-D source-map rows; the III-C row there still describes the old combined subsection).
3. `Thesis-writeup/Documentation/FINAL-RUN-FIXES.md` items 2 and 6, with D-229, D-237, D-240, D-248.
4. `model_augmentation/fit_systems/lbfgs_polish.py`, `blocks.py` l.1385 to 1420, `gantry_dynamic/training.py` l.702.
5. `literature/augmentation/hoekstra2026_lfr-augmentation-fp-models.pdf` Sec. 5 (pp. 8 to 10).

## 12. Do not
Dirk's working rules, confirmed repeatedly:
1. Once Dirk pastes his own sentences, keep them and his style. A fix is at most a few words, or full pasteable
   lines reusing his words. Example (his sentence, one change):
   ```latex
   The controller uses the output error to correct the model input
   within the training window, as in the closed loop of the system.
   ```
2. Answer exactly the question asked; a yes/no question gets yes or no.
3. No file edit unless he asks; ask "saved?" before every `.tex` edit, then tell him to run "File: Revert File".
4. Do not compile in `Writing/`; build tests only in a scratchpad copy (copy `Writing/` without `build/`, put
   `refs.bib` in its `build/`, `latexmk -pdf -interaction=nonstopmode main.tex` and `review.tex`).
5. No "algorithm", "evolve" or similar generated-sounding words; write "system" for the data-generating gantry.

## 13. Operational
- Python: `conda run -n GraduationProject python <script>`; multi-line snippets via a script file in the scratchpad.
- PDFs: Read tool with `pages` (one page or one range per call).

## 14. Delegation
None.
