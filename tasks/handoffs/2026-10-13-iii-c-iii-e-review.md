# Handoff: critically review thesis III-C and III-E (math first), then give Dirk a per-citation source checklist
**From**: session of 2026-10-13 | **Branch**: Augmentation | **Effort suggested**: xhigh (re-deriving displays against code and checking every claim against its evidence)

## 1. Task
Review two subsections of `Thesis-writeup/Writing/sections/03_augmentation.tex` that Dirk has written:
III-C "Closed-loop simulation" (`\label{sec:aug_loop}`, currently l.589 to 751) and III-E "Added-state
initialisation" (`\label{sec:aug_ps2}`, l.896 to end). Work in two phases with a stop for Dirk in between.
**Phase 1 (review, no edits):** check that both subsections are correct, the math above all. Re-derive every
display and check it against the code that implements it; check dimensions, indices, normalisation and notation
against III-A, III-B and the parts of III-D that III-C and III-E use; check that every claim is true and stated no
stronger than its evidence; check that nothing repeats an earlier section; check every `\todo` is still open.
Report all findings in chat as one list ranked by severity: `file:line`, what is wrong, why (derivation, code line,
or source passage), and a fix in Dirk's style (see sec. 12). Stop and wait for Dirk's decisions.
**Phase 2 (source checklist, after Dirk accepted or rejected the Phase 1 fixes):** for every `\cite` in III-C
and III-E, write in chat: the thesis sentence it supports, the exact locator (PDF file, PDF page and printed page,
equation, remark or condition number), what Dirk must check in that passage for the sentence to hold, and whether
`Thesis-writeup/Documentation/CITATION-LOG.md` already has a row and whether that row matches the sentence.
Dirk checks the sources himself; the session does not mark anything verified on his behalf.

## 2. Out of scope
- III-D (`sec:aug_closed_loop`, l.752 to 895): Dirk has not written it yet. Use it only where III-C or III-E
  depend on it (`eq:aug_loss`, `eq:aug_init`, the free-run validation). Do not
  review its prose or propose changes to it.
- III-A, III-B, Sections I, II, IV to VIII and the appendix: read for context only. Flag a conflict in text;
  do not propose rewrites there.
- Any file edit during Phase 1 and Phase 2, including `CITATION-LOG.md` (the checklist stays in chat, Dirk's
  choice 2026-10-13). An edit to a `.tex` file only after Dirk asks for that specific edit and confirms he saved.
- Code changes and runs (no numerical checks; the controller compatibility check is a separate, planned job).
- `refs.bib`, `util/format.tex`, `main.tex`, `.latexmkrc`, `.vscode/`, `build/`.
- Rewriting Dirk's sentences for style (sec. 12).

## 3. Where things stand
- Last commit `b32b9ca` "Update section 3 closed-loop"; `Thesis-writeup/` clean at handoff. Dirk keeps editing
  III-C, so read the file as it is when the session starts; do not assume any earlier draft.
- III-C was drafted in this session from agreed claims, then rewritten by Dirk sentence by sentence. The agreed
  claims are the `% MUST ESTABLISH, III-C (agreed with Dirk, 2026-10-13)` block in the header of
  `03_augmentation.tex` (just above `\section{Dynamic Augmentation ...}`). Judge III-C against that block.
- III-E was finished with Dirk on 2026-10-12/13; its agreed claims are the `% MUST ESTABLISH, III-E` block in
  the same header. It is written for the final runs; differences with the 2026-10-11 runs are `\todo`s.
- No runs in flight.

## 4. Established and verified (this session, 2026-10-13)
- Closed-loop rollout code: `model_augmentation/fit_systems/closed_loop.py::_rollout_segment` (l.216 to 242):
  per step `y_model = output_only(x)`, then `u_fb, xc = step(xc, y_t - y_model, ctrl)`, then
  `hfn(x, u_t + u_fb)`. `closed_loop_rollout` sets `xc = 0` at every window start (l.276 to 285).
  `ControllerBank` (l.76 to 185) folds the normalisation into the controller: `B' = B diag(ystd)`,
  `C' = diag(1/stdu) C`, `D' = diag(1/stdu) D diag(ystd)`; so the controller state is in physical scale.
- `ClosedLoopSimulator.validation_error` (l.649) and `closed_loop_free_run_rms_batch` (l.693): whole validation
  records, one encoder start, `xc = 0` once, record-length-weighted mean of per-record RMS in metres.
- Controller (`scripts/gantry/gantry_dynamic/controller.py`): per stage axis
  `Cnorm(s) = 10w (s+w/6)(s+w/3) / [s (s+3w)(s+10w)]`, `w = 2*pi*100` (l.92 to 96), so it has an integrator;
  diagonal, three SISO controllers on the actuator positions `q_act = (X1, X2, Y)` (`build_cfb_at`, l.104);
  Tustin over the training `ts` (4 kHz), while the records were generated at 20 kHz, `Dc` differs by 2.832
  (module header l.21 to 27). All thesis records use K1, designed at Y = 0 (l.186 to 191, D-221).
- Residual-form derivation (this work): subtracting the system loop from the model loop of `eq:aug_two_loops`
  gives `eq:aug_residual_loop` exactly when the recorded `u_act` was produced by this controller from the
  recorded `y`; `x^c_tau = 0` means the model controller starts in the system's controller state. With a
  different simulated controller K', `u_hat = K(r - y_hat) + u_ff + (K' - K)(y - y_hat)`: the last term carries
  measurement noise into the input. III-C states this limitation; Setup has the size as a todo.
- Kessels (`literature/augmentation/kessels2025_ai-control.pdf`, PDF page = printed page + 27 in Ch. 5):
  Eqs. (5.12), (5.13a to d) printed pp. 156 to 157 (rollout driven by r and u_FF, error r - y_hat against the
  model output, encoder reads y, r, e, u up to tau-1, only NN weights trained); Remark 5.4 p. 157 (FB state read
  from the machine or reconstructed from y, r and the controller from the record start); Remark 5.1 p. 152 (output
  augmentation independent of the input, causality through the FB controller); Remark 5.6 p. 174 (open-loop
  identification of a SUBNET model on the wire bonder unsuccessful, "may be explained" by noise-induced bias from
  input/noise correlation and the FB integrator). Rows added to `CITATION-LOG.md` 2026-10-13.
- Beintema 2023 (`literature/experiment-design/Papers/Beintema_Schoukens_Toth_2023_Deep_subspace_encoders.pdf`):
  Condition 1 (incremental exponential output stability) on printed p. 6; consistency defined in Sec. 4.1.
- Hoekstra 2026 (`literature/augmentation/hoekstra2026_lfr-augmentation-fp-models.pdf`): Eq. (22) p. 8
  (encoder reads up to k-1, joint parameter vector includes theta_base); Sec. 5.5 p. 10 (consistency under
  SUBNET Conditions 2.1 to 2.4, 2.1 = incremental exponential output stability).
- Forssell and Ljung (`literature/closed-loop-id/forssell1998_cl_revisited_liu2021.pdf`, LiU report): Sec. 5.2.1,
  pp. 13 to 14 (indirect approach needs a known regulator and measurable r; r and noise uncorrelated, so an
  output-error model stays consistent); p. 28 example (indirect method fails when a regulator nonlinearity is
  not modelled). Journal (Automatica 1999) numbering unchecked; III-C has a todo for it.

## 5. Assumed but not verified
- Kessels Eqs. (5.1) to (5.3) (his FP model and controller definition) and Sec. 5.3.2.1 (how the wire-bonder case
  obtains the FB states) were NOT read. In chat it was claimed that his controller may be nonlinear (general
  f_FB, g_FB in (5.13d)) and that he uses the machine's own controller. If III-C states either, check the PDF.
- K1 stabilises the baseline at Y = 0 by design; stability over the whole Y range of the records is not shown.
  III-C was agreed to claim only that the controller removes the free-axis offset, not that the closed loop
  satisfies Beintema Condition 1.
- III-E code correspondence (`model_augmentation/fit_systems/ps2_opening.py`: `PS2Spec` l.37,
  `calibration_split` l.59, `_sample` l.161, `_residual` l.183, `_s1_step` l.194, `_cal_unexplained` l.207,
  `_fit_map` l.213) was not re-read this session. Known differences of the 2026-10-11 runs with the text:
  S1 target starts at tau not tau+1, and g_aug starts at zero (`Thesis-writeup/Documentation/FINAL-RUN-FIXES.md`
  items 5 and 6, D-248); these are todos in III-E, not errors.
- Whether the D-140 "Ruled out (4)" ban on citing Kessels Remark 5.6 has lapsed now that the thesis records are
  noisy: todo in III-C, Dirk's decision.

## 6. Tried and failed
- Proposals with `% CHANGED` or `% NEW` after a line -> commented out the rest of the line -> broke Dirk's build
  twice. Give pasteable lines without markers.
- A `\\` as the first line of a paragraph, or a nested `\todo{\todo{...}}` -> latexmk stops ("There's no line here
  to end"), "??" references. Flag both if present in III-C or III-E (III-C had a nested todo near the controller
  rate check on 2026-10-13).
- Rewriting Dirk's sentences in my own words when he asked to "improve" -> rejected repeatedly in this session
  ("keep my style, I won't repeat myself"). See sec. 12.
- Claiming more than was read: in this session two Kessels claims were made from unread pages (sec. 5) and had to
  be retracted. Every statement about a source names the page it was read on.

## 7. Achieved
- III-C exists in the file with three displays (`eq:aug_two_loops`, `eq:aug_residual_loop`, `eq:aug_cl_step`)
  and `fig:aug_closed_loop`; labels used by III-D and III-E resolve; a scratchpad build of `main.tex` and
  `review.tex` had no `!` lines on 2026-10-13 (before Dirk's later edits).
- Setup (`05_setup.tex`, around l.110) has the controller-mismatch bullet with a size todo.
- III-E: displays `eq:ps2_s1`, `eq:ps2_nu`, `eq:ps2_s2`, `eq:ps2_handover`, stop rule in `app:ps_stop`.

## 8. The open question
Nothing blocked.

## 9. Next action
Read `03_augmentation.tex` from l.1 to the end (header blocks, III-A, III-B, III-C, the III-D dependencies, III-E),
then the code of sec. 4 and `ps2_opening.py`, then re-derive each III-C and III-E display and write the Phase 1
findings list in chat.

## 10. Acceptance criterion
Phase 1 is done when every display of III-C and III-E has been re-derived or matched to a named code line, every
claim sentence has been checked against a source passage, the code or an earlier section, and the findings list
(possibly empty per category) is in chat with one fix per finding. Phase 2 is done when every `\cite` in III-C and
III-E has a checklist entry with sentence, locator, what to check and log status.

## 11. Read these first
1. `Thesis-writeup/Writing/sections/03_augmentation.tex` in full, including both MUST ESTABLISH blocks.
2. `model_augmentation/fit_systems/closed_loop.py` l.76 to 310 and l.649 to 720.
3. `model_augmentation/fit_systems/ps2_opening.py` in full, with `Thesis-writeup/Documentation/FINAL-RUN-FIXES.md`.
4. `Thesis-writeup/Writing/README.md` "Math standard", and the III-C row of the source map.
5. `Thesis-writeup/Documentation/CITATION-LOG.md` (rows dated 2026-10-09 to 2026-10-13) for Phase 2.

## 12. Do not
Dirk's working rules, confirmed repeatedly:
1. Keep Dirk's sentences and style. A fix is at most a few words, or full pasteable lines that reuse his own words.
   Example of a good fix (his sentence, one change):
   ```latex
   The controller uses the output error to correct the model input
   within the training window, as in the closed loop of the system.
   ```
2. Answer exactly the question asked; a yes/no question gets yes or no.
3. No file edit unless he asks; ask "saved?" before every `.tex` edit, then tell him to run "File: Revert File".
4. Do not compile in `Writing/`; build tests only in a scratchpad copy (copy `Writing/` without `build/`, put
   `refs.bib` in its `build/`, `latexmk -pdf -interaction=nonstopmode main.tex` and `review.tex`).
5. Write "system" for the data-generating gantry in III-C, not "machine" (Dirk, 2026-10-13).

## 13. Operational
- Python: `conda run -n GraduationProject python <script>`; multi-line snippets via a script file in the scratchpad.
- PDFs: Read tool with `pages` (one page or one range per call). The Forssell text layer is ligature-corrupted;
  read it as images.
- Quotes from code with line numbers follow the code-quote verification rule.

## 14. Delegation
None. All checks need the full context of the section and the code; inline reading is cheaper.
