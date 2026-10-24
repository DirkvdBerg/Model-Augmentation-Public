# Handoff: critique the final Results floats, then write the Section VII text in Dirk's style, on the model of Hoekstra et al. 2026
**From**: results-floats session 2026-10-24 | **Branch**: Augmentation | **Effort suggested**: medium (Dirk's choice: high is too slow; the floats and numbers are fixed, the work is judgement per float and per paragraph)

**Dirk's current message always overrides this plan. When he asks a question, answer that question first, in plain words, in a few sentences. Replies stay short.**

## 1. Task
Two parts, in this order, each confirmed by Dirk before the next.
1. **Critique the floats.** For every table and figure in Section VII (`Thesis-writeup/Writing/sections/06_results.tex`) and in the
   result parts of the appendix (`99_appendix.tex`: `app:frf`, `app:splits`, `app:choices`), give in one or two lines: the one claim
   it supports, the criterion it is judged against, whether it is readable and honest, and how it compares to the matching float in
   Hoekstra et al. 2026 (4.4). Verdict per float: keep, change (exactly how), move or cut. Do not edit anything in this part.
2. **Write the Section VII text**, subsections `sec:res_structure`, `sec:res_obc`, `sec:res_generalisation` and the section
   opening, NOT the Discussion. First a short outline per subsection (what the text must state, in order: criterion, float pointer,
   two to four sentences reading the float, verdict against the criterion, as Hoekstra 2026 Sec. 6 does). After Dirk approves the
   outline, one paragraph per turn: rendered Now / After, applied with marks only on his "yes", "apply" or "do it".
Use the `thesis-section` skill for the order and the deletion test, but Dirk's wording (4.6) wins over the skill's.

## 2. Out of scope
- The Discussion subsection (`sec:discussion`, inside `06_results.tex`): a later session (Dirk 2026-10-24).
- Sections I to VI and the Conclusion (`08_conclusion.tex`). If a result changes a sentence there, name it in one line and leave it.
- New evaluations, training runs, server jobs. All numbers come from `Thesis-writeup/Results/24-10-2026/CSV` and the regenerated
  tables (4.2). If a claim needs a number no CSV holds, say so in one line with the script that would produce it.
- Figure and table scripts in `Thesis-writeup/Code/Figures/`: change one only after Dirk agrees to a change in part 1.
- `\ifdraft` bullet blocks: never mention or edit them. Thesis builds (his editor rebuilds on save). Git writes.

## 3. Where things stand
Branch `Augmentation`, dirty tree (Dirk commits himself). Dirk edits the thesis in parallel and saves often: re-read a paragraph
right before proposing a change to it. All floats of Section VII and its appendix were regenerated today from the final CSVs in the
design Dirk approved (4.1); the running text of Section VII is still the 2026-10-18 preliminary text and is wrong in places (4.3).

## 4. Established and verified

### 4.1 The floats (final, agreed with Dirk 2026-10-24)
`\FloatBarrier` (package `placeins`, `util/include.tex`) sits before each `\subsection` of Section VII, so every float stays in
its own subsection. Scripts and sources: `Thesis-writeup/Code/Figures/README.md`, top table.

| Subsection | Float | Label | Content |
|-|-|-|-|
| VII-A structure (Aspect 1) | Table A | `tab:res_structure` | NRMS_e X1/X2/Y of baseline M_LPV, n_xbar=0, n_xbar=2 plain, n_xbar=2 PS, n_xbar=8 PS (all true start), final M_OBC (detuned), black box M_BB; all 11 test records, with multisine (I1, I2, E1) and without (8) |
| | Time trace | `fig:res_time_trace` | I1, Y; top: simulation error, RMS over 50 ms, % of sigma_e; bottom: servo error 7.84 to 7.95 s, % of sigma_e; system as thick light band, models thin lines |
| | Validation curves | `fig:res_val_curves` | validation NRMS_e against optimiser updates, n_xbar=0, plain, PS, n_xbar=8 PS (from the run logs) |
| | FRF | `fig:res_frf` | F_Y to Y at Y = 0, dB relative to the system at 20 Hz; band: final model over Y; system and baseline vary < 0.3 dB over Y |
| VII-B interpretability (Aspect 2) | Table C | `tab:res_obc` (+ temporary `tab:app_obc_full`) | relative error of the ten combinations, M_U and M_OBC, detuned and true start; norm to truth; start difference |
| | Table D | `tab:res_obc_constraint` | rho(F_aug) on training and test records, rho(Xi), NRMS_e of full model and physical part, M_U vs M_OBC |
| VII-C generalisation (Aspect 3) | Table E | `tab:res_generalisation` (+ temporary `fig:res_generalisation`) | NRMS_e per model and axis, mean over I and over E records: with multisine, without, multisine added |
| Appendix | | `fig:res_frf_full`, `tab:app_splits`, `fig:res_lpv_lti`, `tab:app_structure` (Table A rows on validation), `tab:app_obc_na0`, `tab:app_bb_search` | PS spectrum and `tab:app_obc_full` were cut |

Not yet referenced in the text: `fig:res_val_curves`, `tab:res_obc_constraint`. The two temporary second labels keep old references
resolving; remove them when the text no longer uses them.

### 4.2 The numbers (final CSVs; also in the `.tex` tables under `Writing/tables/`)
- Table A, test, Y with multisine: M_LPV 121.5, n_xbar=0 84.7, plain 76.5, PS 6.1, n_xbar=8 6.6, M_OBC 6.6, M_BB 27.2 %.
  Without multisine Y: 3.3 / 19.9 / 13.4 / 4.0 / 4.0 / 3.7 / 127.4 %. So n_xbar=0 and plain are WORSE than the baseline without
  multisine. Validation curves: PS separates from the others within the first 1300 updates; plain still decreases slowly at the end
  (29.6 to 25.2 % over 31000 updates).
- FRF: learned mode 211 to 214 Hz against the system's 212 Hz; learned damping varies with Y (0.017 to 0.092) against the system's
  constant 0.043 (`res_frf_modes.csv`). 3x3 FRF: the final model adds a narrow resonance near 210 Hz in the X-Y cross terms that the
  system lacks (up to 34 dB deviation at the outer positions, 5 dB at Y = 0).
- Table C: norm to truth, detuned start: M_U 83.1, M_OBC 69.1 (initial 30.0); true start: 58.3 and 59.5. Start difference: M_U 36.4,
  M_OBC 31.3. OBC shortens the start difference for k_b,Sigma, c_g1, c_g2; about equal for c_y, c_b,Sigma, J_eff; longer for m_h, d.
  Dampers drift most (up to +47 %), masses within 3 %. Without added states (appendix) U ends closer: 48.8 vs 68.4.
- Table D: rho(F_aug) M_U 0.85 / 0.84, M_OBC 0.00 on training and 0.24 on test records; rho(Xi) 0.59. Full model test X1/X2/Y:
  M_U 5.7/5.6/4.8, M_OBC 6.3/6.2/4.6. Physical part Y about 42 % for both.
- Table E, final model on Y, E minus I: with multisine +1.3, without -0.3, multisine added 0.0 points. With multisine added the
  final model is 6.0 % against baseline 64.6 to 71.3 % and black box 85.9 to 90.9 %. The black box is above the baseline on every
  record without multisine (about 100 to 128 %); it had not converged (D-264).
- Preliminary text that is now wrong: "M_U deviates close to double that of M_OBC" and "From the correct start, OBC also ends closer"
  (`06_results.tex`, VII-B). The OBC paragraph of the Discussion was already updated today (69.1 against 83.1).

### 4.3 What each aspect asks and its criterion (agreed 2026-10-24)
Research question and aspects: `01_introduction.tex:400-427`. Maarten: every result is argued against a criterion stated before it
(`feedback_thesis_argue_results.md`). The current header of `06_results.tex` ("MUST ESTABLISH ... No verdicts ... those are in the
Discussion") is superseded: verdicts go in the Results, as Hoekstra does (4.4).
- Aspect 1 (structure and training method): captured = the Y error with multisine falls to the level of the records without
  multisine, and the learned mode matches the system's. Answer: dynamic parallel with added states and PS; neither alone suffices;
  eight states add nothing.
- Aspect 2 (interpretability): the Introduction defines interpretability as a UNIQUE estimate of the identifiable combinations, not
  the true values (`01_introduction.tex:246-250`). Criteria in order: the constraint holds (overlap near 0), uniqueness (start
  difference), equal accuracy, and only in simulation the distance to the truth (not expected, Phi^T Xi != 0, Section IV-E).
  Answer: to a limited extent.
- Aspect 3 (generalisation): E not clearly worse than its I partner; lower than baseline and black box. Pairs: E1-I2, E2-I3, E3-I5,
  E4-I6, E6-I3 (`99_appendix.tex` tab:setup_records). The margin for "clearly" is open (5).

### 4.4 Hoekstra et al. 2026 (`Thesis-writeup/References/Hoekstra-2026-LFR-model-augmentation-arXiv-v1.pdf`)
Sec. 6 simulation study PDF p. 10-12, Sec. 7 experiment p. 12-14. Pattern: each subsection opens with what is estimated and compared,
points to one table with all structures plus baseline and black box (Table 6 p. 13, Table 8 p. 14), reads it in one to three
sentences including the verdict, then a convergence figure (Fig. 8, validation loss against epochs, noise floor dashed, p. 12) and
figures that show what was learned (Fig. 9 states, Fig. 10 Bode of the learned filter against the true one, p. 12). Physical
parameters are reported in words only. No results in an appendix. Gyorok et al. 2026 (`Gyorok-2026-Orthogonal-by-construction.pdf`
p. 6-9) show uniqueness with estimates from 10 starts (Fig. 2 p. 8) and state openly that their data design does not fully guarantee
orthogonality (p. 8). Gyorok et al. 2025 Table 1 (p. 10) reports the physical part alone. Quote a paper (a few words, page) before
any claim about it.

### 4.5 How to propose (worked today)
Rendered Now / After, never LaTeX before apply; change only the words Dirk names; one change per turn. Marks of
`Writing/util/format.tex`: `\chg[own]{old}{\kwtwo{new}}`, `\add[own]{\kwtwo{...}}`, `\del[own]`, `\mv`/`\delmv`; a
`% FEEDBACK (own, 2026-10-2x): <reason>` line above each change; no marks inside math, captions or `\cite[...]`; no mark nested in
struck text (`\mbox{\ref{...}}` there). Details and an accepted example: `tasks/handoffs/2026-10-24-section6-setup.md` sections 4.1
and 4.3. Edit `.tex` with a small Python script written to the scratchpad (a bash heredoc halves the backslashes); check that the
braces balance after each edit.

### 4.6 Dirk's style, by example
His sentences, approved today in Section VI-B (`05_setup.tex`):
> We have chosen both frequencies to be above the crossover of the loop with $K_1$, so that the feedback does not suppress them in the data.

> The multisine band runs from the highest crossover frequency of the loop with $K_1$ at 106 Hz up to 297 Hz (Fig. setup_band).
> Above the crossover, this band covers 90 % of the squared difference between the frequency responses of the system and the baseline.

What these show: "we" for every choice; the reason after "so that" or "because"; the observation before the consequence
("The motion profiles excite ... and barely excite the added mass. We therefore add ..."); concrete names ("the added mass", "the
loop with $K_1$"); one fact per sentence at a natural length. He has also said the earlier sections (I to VI) are well written:
read two paragraphs of `03_augmentation.tex` and `05_setup.tex` as further examples before writing.

## 5. Assumed but not verified
- Table C (6 columns) and Table E (8 columns) fit one column: not compiled. Settled when Dirk's editor rebuilds; if one overflows it
  becomes `table*`.
- `placeins` is installed in Dirk's MiKTeX: settled by his first rebuild.
- That four floats before the first `\FloatBarrier` do not leave a float page with white space: settled by the rebuild.
- The margin for "E not clearly worse than I" (Aspect 3): candidate the spread of the final model over the I records without
  multisine on Y (3.1 to 4.3 %); Dirk's decision.

## 6. Tried and failed (this session; each cost Dirk time)
- Wording with comma chains, packed sentences or semicolons -> "100 commas in one sentence", "reads AI" -> one fact per sentence in
  his pattern (4.6).
- The over-correction: chopping into fragments ("The band ends at 297 Hz.") -> "dots for a sentence of 6 words" -> natural sentence
  length, two facts may share a sentence when they belong together.
- Answering a terminology question with computations -> anger -> answer a definition question with the definition only.
- Dotted model lines, stars on curves, a grey window band in the same grey as the system, a triangle marker -> rejected as ugly or
  "AI" -> the styles in 4.1.
- Axis labels "Moving NRMS", "e_Y/sigma_e,Y", "Optimiser update (x10^3)" -> vague or unclean -> "% of sigma_e", full update numbers.
- Errors in metres, dB re 1 m/N, frequencies over f_b -> confidentiality rule is errors only (NRMS_e), Hz allowed
  (`feedback_units_scientific_metres.md`).
- A dot plot of the combinations and a per-record generalisation figure -> Dirk prefers tables (Tables C and E).
- Judging OBC by the distance to the truth only -> the Introduction's criterion is uniqueness (4.3).

## 7. Achieved
All floats of 4.1 are in `06_results.tex` and `99_appendix.tex` with final numbers and agreed captions. VI-B of `05_setup.tex` was
restructured and applied today; the OBC paragraph of the Discussion now states the final numbers and why the selection on validation
cannot protect the parameters. Nothing of the Section VII running text has been rewritten.

## 8. The open question
Nothing blocked. Part 1 may lead Dirk to change a float; then change the script in `Code/Figures`, regenerate with the command in 13,
and show the result before touching the text.

## 9. Next action
Read 11.1 to 11.4, then post part 1: per float of 4.1 (main text and appendix), one or two lines with claim, criterion, readability,
the Hoekstra comparison and the verdict (keep, change how, move, cut).

## 10. Acceptance criterion
Dirk has approved the per-float verdicts and an outline per subsection; every paragraph of the opening and of VII-A to VII-C is
applied with marks on his yes; each subsection states its criterion before its float and its verdict after; every float is
referenced; no number in Section VII differs from the CSVs of 4.2; the temporary labels are removed or still needed by a named reference.

## 11. Read these first
1. `Thesis-writeup/Writing/sections/06_results.tex` up to `\subsection{Discussion}`, and the appendix floats (`99_appendix.tex` app:frf, app:splits, app:choices).
2. The regenerated tables in `Thesis-writeup/Writing/tables/` (`res_structure`, `res_obc_combinations`, `res_obc_constraint`, `res_generalisation`, `res_all_splits`) and the figures `Writing/figures/res_time_trace.pdf`, `res_val_curves.pdf`, `res_frf.pdf`, `res_frf_full.pdf`, `res_lpv_lti.pdf`.
3. Hoekstra et al. 2026, PDF pages 10 to 14 (4.4).
4. `01_introduction.tex:240-260` and `400-427` (interpretability definition, research question, aspects).
5. `tasks/handoffs/2026-10-24-section6-setup.md` sections 4.1 and 4.3 (how to propose, his wording).

## 12. Do not
- Edit before an explicit yes; a pasted phrase or a question gets an answer, not an edit.
- Show LaTeX before apply. Use semicolons, comma chains or sentence fragments (6).
- Use the 2026-10-18 CSVs (`Results/18-10-2026/CSV`) or the preliminary numbers in the current text.
- Reopen decisions of 4.1 and 4.3 unless Dirk raises them.

## 13. Operational
Regenerate one output from the final CSVs (seconds), from the repository root:
```bash
THESIS_RESULTS_DIR="$PWD/Thesis-writeup/Results/24-10-2026/CSV" conda run -n GraduationProject python Thesis-writeup/Code/Figures/<stem>.py
```
To show Dirk a figure, render the PDF to PNG in the scratchpad (`fitz`, 200 dpi) and send it; a large HTML page with embedded images
does not open in his side panel.

## 14. Delegation
None: a dialogue with Dirk on known files.
