# Handoff: discuss five thesis-text points (OBC recovery estimates, test-set count, LPV/LTI comparison, theta_aux size, an overlap sentence)
**From**: session of 2026-10-18 | **Branch**: Augmentation | **Effort suggested**: high (the points mix derivation
checks and text decisions; the answers must agree with the code and the final runs)

## 1. Task
Discuss with Dirk, one point at a time and in this order, the five points of section 8, and for each give him a
clear answer with its evidence and, where the text should change, a concrete proposal in the copy format (LaTeX in a
code block, one sentence per line, wrapped at about 75 characters). Apply a change only when Dirk asks for it, and
then only the agreed lines. The thesis is `Thesis-writeup/Writing/sections/*.tex`; the method text describes the
final method (D-271 campaign), current-run differences go in a `\revnote`.

## 2. Out of scope
- Any code, run, evaluation or figure change. The results tables and figures are produced in another session
  (`tasks/handoffs/2026-10-18-final-results-figures.md`); if a point needs a new number, say which and leave it.
- Rewriting Dirk's sentences beyond the agreed point (memory: never rewrite his wording; at most five-word fixes or a
  proposal he pastes himself).
- Compiling the thesis (only on request).
- Other open revnotes and todos than the five points.

## 3. Where things stand
- Branch `Augmentation`, dirty tree (Dirk commits). The thesis sections were edited today (2026-10-18) by the previous
  session: III-B encoder start rewritten to the least-squares form, III-D loss weight `W_r` and NRMS_e selection, III-E
  PS2 windows (encoder up to tau-1, residual tau..tau+H-1) and figure `figures/tex/ps2_windows.tex`, IV-B scope
  sentence of the recovery analysis, V metrics (`eq:setup_rms` from k = n+1), appendix table rows.
- Final runs (logs `Thesis-writeup/Results/18-10-2026/thesis-results-4/`): R3 to R7, R13, R14 done; R1, R2 (OBC)
  still running on 2026-10-18.

## 4. Established and verified
- `eq:obc_recovery` (`04_obc.tex`, subsection "Identification and recovery"): beta_hat = beta* + (Phi^T Phi)^-1 Phi^T Xi
  (with projection), beta_hat_np = beta* + (Phi^T Phi)^-1 Phi^T (Xi - F_aug) (without). Derived by minimising the
  first-order one-step error on the tuples at x_bar = 0. Its scope sentence now reads: a first-order result for the
  one-step least-squares problem on the reference slice x_bar = 0; it does not predict the parameter bias from the
  closed-loop loss with active added states (edited 2026-10-18 after an independent review).
- The "OBC pred." columns (predicted first-order bias) were dropped from the results by Dirk (revnote at
  `06_results.tex` near `tab:res_obc`): predicted 100 to 1300 % against observed a few to 30 %.
- Truth overlap rho(Xi) = 0.59 (results table `tables/res_obc_combinations*.tex`); rho is a norm ratio, so the share of
  the squared norm of Xi in ran Phi is 0.59^2, about 35 %; Xi also holds tuple reconstruction error and noise.
- theta_aux has 11 entries because the projection basis is Phi_Gamma = [Phi Gamma]: 10 columns for the ten
  combinations beta plus one offset column gamma (Gyorok 2025 Eq. (18)); D-257, `obc_space='affine'`; every OBC log
  prints `basis rank 11/11` and `11 coefficients`. Text: `04_obc.tex` where Phi_Gamma is in R^{6N x 11} and
  theta_aux in R^11.
- Overlap computation (`Thesis-writeup/Code/Evaluation/eval_overlap.py` about :104-106): for OBC the deployed
  correction is subtracted before rho is computed, so the table's OBC overlap is that of the corrected network; zero on
  the training tuples by construction, 0.13 on the test tuples.
- Test data on disk: `Thesis-writeup/Data/Coulomb-tanh-and-MSD-lowpass-noise/Test` holds E1 to E6 and I1 to I6 (12
  records); E5 (controller K2) is skipped in the evaluation, so 11 are scored, matching "11 for testing" at
  `05_setup.tex` about :199. The multisine twins (D-258, evaluation only, never trained or validated on) are in
  `Thesis-writeup/Data/Multisine-test-twins/.../Test`: 7 records, E2M, E3M, E4M, E6M, I3M, I5M, I6M.
- LPV/LTI comparison appears in: `01_introduction.tex` revnote about :290 (Aspect 1 removed, comparison to the
  appendix, LTI error 1.5 to 3.7 times the LPV error on X); `05_setup.tex` about :228-256 (model pairs including
  M_LPV and M_LTI, and their rows in `tab:setup_models`); `99_appendix.tex` about :255 (figure `res_lpv_lti.pdf`).

## 5. Assumed but not verified
- Whether the folder listing above also contains other files (for example TF records) in `Test`: confirm the test
  count from `Thesis-writeup/Data/Coulomb-tanh-and-MSD-lowpass-noise/MANIFEST.csv` before answering point 2.
- Which test records have a multisine and which twins they pair with: read the two MANIFEST.csv files; D-258 has the
  rule.
- Where the sentence of point 5 stands: it is not found verbatim in the sections; the nearest text is
  `08_conclusion.tex` about :44-52 ("The truth overlap of 0.59 means Phi^T Xi != 0, so the recovery condition ... is
  not met", then `\todo{Use this?}`). Ask Dirk where he wants it if it is new.

## 6. Tried and failed
- The first-order bias prediction of `eq:obc_recovery` against the trained OBC model -> 100 to 1300 % predicted vs a
  few to 30 % observed -> the expansion assumes x_bar = 0 and a one-step fit, while training is closed loop with added
  states that carry the absorber, and the correction f_obc depends on (x~, u) only, never on x_bar -> revnote at
  `06_results.tex` tab:res_obc. Do not reintroduce a quantitative bias prediction.

## 7. Achieved
None for these five points; they are open.

## 8. The open question
Five points, in Dirk's words, each to be answered and, where needed, turned into a text proposal:
1. "look at beta hat and beta hat np": what `eq:obc_recovery` still says after the scope change; whether beta_hat_np
   is still needed in the text (it shows why the projection removes the network term Phi^T F_aug from the estimate);
   whether the pair should stay, be shortened, or move. Candidate answers: keep both as the qualitative contrast on
   the slice x_bar = 0; keep only the contrast sentence; move to the appendix.
2. "11? and the multisine doubles?" (revnote at `05_setup.tex` about :201): confirm 11 scored test records (12 on disk,
   E5 skipped and why), and whether the text should name the 7 multisine twins and how they are counted (not part of
   the 11).
3. "remove lpv lti comparison, just reference it. discuss where": options are one sentence in Section II (the LPV form
   follows from the equations of motion, with a pointer to the appendix figure), one sentence in Section V, or only the
   appendix figure; consequences: drop M_LPV/M_LTI as a model pair in `05_setup.tex` and from `tab:setup_models`, but
   M_LPV may still be the baseline row of the generalisation table (check `06_results.tex`).
4. "theta aux is eleven?? due to gamma?": yes (section 4); check that every place in `04_obc.tex` that states the size
   or the projector uses Phi_Gamma consistently (R^{6N x 11}, theta_aux in R^11), and that rho in `eq:obc_overlap`
   uses ran Phi (10 columns) on purpose: it measures overlap with the parameter directions only.
5. The sentence "According to \eqref{eq:obc_recovery}, this overlap can bias the physical parameters even with the
   projection." with `\todo{Use this?}`: decide whether it is consistent with the new scope sentence. It is defensible
   only qualitatively and on the slice x_bar = 0 (Phi^T Xi != 0 means the one-step estimate is biased there); it must
   not suggest the formula predicts the size of the observed drift.

## 9. Next action
Start with point 1: open `04_obc.tex` at `eq:obc_recovery`, read the surrounding derivation and the new scope
sentence, and give Dirk the answer and, if a change is needed, the proposal.

## 10. Acceptance criterion
Each of the five points has an answer Dirk accepts, consistent with the code and the logged numbers in section 4, and
any agreed text change is applied exactly as agreed, with no other line touched.

## 11. Read these first
1. `Thesis-writeup/Writing/sections/04_obc.tex`, subsections "Extension to the gantry" and "Identification and
   recovery": points 1, 4, 5.
2. `Thesis-writeup/Writing/sections/05_setup.tex`, data and model-pair paragraphs: points 2, 3.
3. `Thesis-writeup/Writing/sections/06_results.tex` around `tab:res_obc` and `08_conclusion.tex` about :40-55: point 5.
4. `docs/decisions.md` D-257 (affine basis, Gamma), D-258 (multisine twins), D-268 (overlap stacks).
5. `Thesis-writeup/Writing/sections/01_introduction.tex` revnote about :290: point 3.

## 12. Do not
- Do not edit before Dirk asks; do not rewrite his sentences; no em-dashes.
- Do not reintroduce the predicted-bias numbers or claim the formula explains the observed drift (section 6).
- Do not touch code, data, results or figures.

## 13. Operational
Text only. If Dirk asks for a compile, the editor rebuilds on save; check `build/main.log` instead of running latexmk.

## 14. Delegation
None.
