# Handoff: decide with Dirk what the Results section (06_results.tex) states, using Hoekstra's results style and the v1 supervisor feedback
**From**: session of 2026-10-18 | **Branch**: Augmentation | **Effort suggested**: high (careful reading of four
annotated PDFs and three papers, then a reasoned content proposal; no code)

## 1. Task
Prepare and lead a discussion with Dirk on WHAT the Results section of the thesis must state, subsection by
subsection. First read every comment in the v1 supervisor feedback (`Thesis-writeup/Feedback/v1/`, four annotated
PDFs plus two general notes) and extract what it says about how results must be presented. Then read the results
sections of Hoekstra's two augmentation papers (primary model for presentation) and Gyorok's two projection papers
(secondary, for the parameter-recovery result) and record how they show results: what each result opens with, which
metric, what is in a table versus a figure, how a verdict is phrased, what is left to the appendix. Combine both with
the final-campaign numbers into ONE proposed content outline for `Thesis-writeup/Writing/sections/06_results.tex`:
per subsection the criterion it is judged against, the numbers that answer it, the verdict, the mechanism, the figure
or table that carries it, and which feedback comment or paper convention motivates each choice. Present it to Dirk
in chat and discuss; write it into the file only after he agrees (section 9).

## 2. Out of scope
- Drafting finished LaTeX prose for 06_results. This session decides content; drafting is a later step with the
  `thesis-section` skill.
- Any evaluation run, figure-script or table-script change, or new figure (`Thesis-writeup/Code/`). The figures and
  tables exist (section 7). If the discussion asks for a different figure, write the request down; do not build it.
- Editing any section other than 06_results (05_setup and 99_appendix included). Flag conflicts in text.
- Rewriting Dirk's sentences: per the user memory on Dirk's wording, propose; edit only the outline blocks, only after
  his go.
- `Thesis-writeup/Feedback/v2/` (empty as of 2026-10-18; the user named v1).
- A literature search. The papers are named and on disk; do not invoke `deep-research` or search the web.
- The final rerun of R1 and R2 (server, after training): Dirk launches it.

## 3. Where things stand
- Branch `Augmentation`, last commit `4207001`; tree dirty (Dirk commits). No run of this session is in flight.
- Training of R1 (job 89792_221) and R2 (89794_222, run folder 89796) still running on the server, at update 22,100
  of 33,400 on 2026-10-18 evening, about 14 h left. Their numbers in the thesis are PRELIMINARY (update 19,500;
  dagger in tables, red stamp in figures). Content decisions must not depend on R1/R2 digits; verdicts on them are
  provisional (especially the OBC verdict, section 5).
- `06_results.tex` currently holds per subsection an `outline` block (Dirk's bullets, partly outdated, section 4) and
  a `wip` block (older generic bullets), plus the floats.

## 4. Established and verified
Presentation decisions (Dirk, binding; `Thesis-writeup/Documentation/THESIS-RESULTS.md` section 3, "Presentation
agreed 2026-10-16" and "Update 2026-10-18"):
- Main text shows only what answers the research question, one result per aspect, each opened by its criterion;
  anything that justifies a choice goes to the appendix. Aspect 1 (LPV vs LTI) is REMOVED as an aspect (2026-10-18);
  its figure is in Appendix `app:choices`.
- Main-text results: B (Aspect 2: Table `tab:res_structure`, test I1 to I6, baseline at true theta / final model /
  black box / floor, with and without multisine; FRF `fig:res_frf`; I1 time trace `fig:res_time_trace`), C (Aspect 3:
  Table `tab:res_obc`, detuned start only, R4 unconstrained vs R1 OBC; the predicted-bias column was DROPPED
  2026-10-18), D (Aspect 4: `fig:res_generalisation`, I/E pairs plus multisine twins, baseline at TRUE parameters).
- Appendix: `app:splits` (`tab:app_splits`, all models on train/val/test), `app:choices` (LPV vs LTI, full structure
  table on validation, PS vs plain spectrum, Table C both starts, n_a = 0 pair, black-box search, revnote for the
  missing validation curves), `app:frf` (full 3x3 FRF), `app:setup` (coverage, band, checks).
- Metric: NRMS_e per axis in % (D-256, D-262, `05_setup.tex` eq:setup_rms), RMS in m and the noise floor beside it;
  records with and without multisine reported separately. Errors in metres in scientific notation (user memory).
- Test is for final results only; choices are decided on validation (D-269).
Numbers (final campaign, `Thesis-writeup/Results/18-10-2026/CSV/`, tables in `Thesis-writeup/Writing/tables/`):
- Acceptance: every run reproduces its training-log validation score to <= 4e-11 relative (`res_checks.csv`).
- B, test I1 to I6, NRMS_e X1/X2/Y %: with multisine, baseline 18.8/20.7/116.6, final 9.7/9.5/6.2, black box
  47.8/47.8/35.9, floor 0.06/0.06/0.01; without multisine baseline 4.5/4.6/3.4, final 4.7/4.8/3.8, black box
  104.8/97.0/127.7. Train/val/test agree (final Y with multisine 6.8/7.1/6.2).
- Structure (validation, with multisine, Y): baseline 121.8, n_a 0 86.5, plain 78.1, PS 6.4, PS n_a 8 6.3. Without
  multisine n_a 0 and plain are WORSE than the baseline on Y (18.4 and 14.6 vs 3.7).
- FRF modes of the final model 211.2 to 213.4 Hz, zeta 0.036 to 0.080 over Y; truth 211.9 Hz, zeta 0.043
  (`res_frf_modes.csv`).
- C (test stack, detuned start): combination-error norm 83.1 % unconstrained vs 47.7 % OBC (correct start 58.3 vs
  35.4); full model NRMS_e 5.7/5.6/4.8 vs 6.4/6.4/4.6; physical part alone 10.3/9.3/41.8 vs 9.7/10.2/42.7 (no
  resolved difference); network overlap 0.84 vs 0.13; truth overlap 0.59 (train 0.51, val 0.57). Without added states
  (R14 vs R13) it reverses: norm 48.8 vs 68.4, OBC overlap 0.42.
- D: final model 6 to 14 % on multisine records, 2.5 to 6 % on the rest (equal to or slightly worse than the baseline,
  worst E4 and I6); black box 75 to 190 % on every record without multisine (no better than y_hat = r), close to the
  final model on I2 and E1, better on I2 X2 (10.2 vs 11.4); twins final 4 to 9 % on Y vs baseline 48 to 87 %.
- Each E record leaves the training range in exactly one direction: E1, E2 in Y; E3, E6 in acceleration; E4 in
  velocity (`fig:setup_coverage`, rebuilt 2026-10-18).

## 5. Assumed but not verified
- The R1/R2 verdicts hold after training ends. The OBC-drifts-less result (C) rests on preliminary R1/R2 parameters,
  which keep moving; settle after the final rerun. Write the C verdict as provisional.
- The v1 PDFs carry their comments as PDF annotations (text in highlights/notes). Not opened in this session; if
  annotations are empty, the comments are drawn into the page and must be read from the rendered pages.
- That Hoekstra 2026 (Automatica submission) has a results section structured comparably to his 2025 EJC paper; read
  both and say if they differ.
- Single-seed rule: differences under about 1 point NRMS_e are not resolved (THESIS-RESULTS section 2); the seed
  spread itself was never measured for this campaign.

## 6. Tried and failed
- Main text with every model (11-row Table I) -> Jasper (v1 general note, point 2): too complicated to explain, ask
  whether all models are needed -> reduced to four rows on 2026-10-18; the full table is in the appendix.
- Table I in metres only -> Jasper point 4: the strong result visible in the time trace "is not clear in table 1";
  relate it to a realistic achievable RMS -> NRMS_e (relative to the servo error) became the primary metric (D-256);
  check whether the table now shows it.
- First-order predicted parameter bias (eq:obc_recovery) beside the observed one -> 100 to 1300 % predicted vs a few
  to 30 % observed: outside the range of the first-order expansion, and it holds for a one-step fit at x_a = 0, not
  closed-loop training -> column dropped (revnote under Table C). The sentence in `05_setup.tex` (end of the metric
  paragraph) that compares predicted and observed delta_j is now stale: flag it, do not edit.

## 7. Achieved
- All agreed figures and tables exist and are wired into 06_results / 99_appendix (generated 2026-10-18 from the
  18-10 CSVs with `THESIS_RESULTS_DIR=Thesis-writeup/Results/18-10-2026/CSV`). Not compiled; Dirk's editor rebuilds.
- Captions of Table B and Table C were adjusted to the reduced content; the coverage-figure caption replaced.

## 8. The open question
What does each Results subsection state, and in which order? Inputs that decide it: (a) the feedback comments
(section 1 reading), (b) Hoekstra's convention for opening a result, the metric shown and the table/figure split, (c)
the binding presentation rules in section 4. Known tensions to put to Dirk explicitly, each with the evidence:
1. The current outline bullets contradict the numbers: "OBC not less drift" (now OBC drifts less, provisionally),
   "PS lowers Y about 9 times" (now about 12x vs plain), "mode 212.0 to 212.2 Hz, zeta about 0.06" (now 211.2 to
   213.4 Hz over Y), "on X the LTI error is 1.5 to 3.7 times" (Aspect 1 removed; X2 on I2 is equal).
2. The leftover "Baseline form" subsection in 06_results belongs to the removed Aspect 1.
3. Criterion of D ("at least as good as the black box on I") fails narrowly on I2 X2; how to state that honestly.
4. Without multisine the augmentation is equal to or slightly worse than the baseline: state it (THESIS-RESULTS
   section 2 requires both sides).
5. Jasper point 5 (parameter recovery without added states) is answered by R13/R14 in the appendix, and it shows OBC
   is WORSE there; decide whether that belongs in the main text of C.

## 9. Next action
Read the six v1 feedback files and the results sections of the four papers, then post in chat ONE proposed outline
of 06_results: per subsection (Augmentation structure, Interpretability, Generalisation, plus any opening paragraph),
3 to 6 one-line bullets (criterion, number, verdict, mechanism, float), each bullet tagged with its source
(feedback comment, paper convention, or decision number), followed by the tensions of section 8 as questions. Format
per the user memory on outlines (headers plus short one-line bullets, open points marked (?)). Example of one bullet:
`- Criterion: below the baseline on the multisine records (Table~\ref{tab:res_structure}) [Maarten: argue against a pre-stated criterion; Hoekstra 2025 <section where you found the convention>]`
After Dirk agrees, replace the `outline` blocks of 06_results with the agreed bullets (leave the `wip` blocks and all
floats alone unless he says otherwise).

## 10. Acceptance criterion
Dirk agrees with a subsection-by-subsection outline in which every bullet (i) names a number from section 4 or a
table/figure label, (ii) carries a source tag, and (iii) every v1 comment about the Results or about presentation in
general is either reflected in a bullet or listed as deliberately not followed, with the reason.

## 11. Read these first
1. `Thesis-writeup/Feedback/v1/` (all six): `Feedback-Jasper-1-...v2.pdf`, `Feedback-Jan-1-...pdf`,
   `Feedback-Jan-2-...pdf`, `Feedback-Quinten-1-...pdf`, `Feedback-Jasper-General.txt`,
   `Feedback-Jan-General-1-and-2.txt`. Extract every annotation, not only the Results pages: Jan's general note asks
   for more emphasis on WHY things are done and on one coherent story, which applies to Results too.
2. `literature/augmentation/hoekstra2025_lfr-augmentation-ejc.pdf` and
   `literature/augmentation/hoekstra2026_lfr-augmentation-fp-models.pdf`: results sections (primary style model).
3. `literature/Orthogonality/Hoekstra - Orthogonal projection-based regularization for efficient model.pdf`
   (Gyorok 2025 L4DC, Table 1: full model and physical part alone) and
   `literature/Orthogonality/gyorok2026_orthogonal-by-construction-augmentation_IFAC-JSC_arXiv2511.01321.pdf`
   (Table 1 style used for Table C).
4. `Thesis-writeup/Documentation/THESIS-RESULTS.md` sections 1 to 3 (aspects, rules, agreed presentation, 2026-10-18
   update).
5. `Thesis-writeup/Writing/sections/06_results.tex` (current outline and floats) and the generated tables in
   `Thesis-writeup/Writing/tables/` (`res_structure.tex`, `res_obc_combinations.tex`, `res_all_splits.tex`).

## 12. Do not
- Do not quote R1/R2 numbers as final, and do not quote R1/R2 training-split or overlap numbers at all: they come
  from update 18,200 while labelled 19,500.
- Do not bring back the predicted-bias column, the 11-row main table, or LPV vs LTI as a main result (Dirk's
  decisions 2026-10-18).
- Do not compile the thesis (user memory).
- Do not edit floats, captions, the appendix or 05_setup.

## 13. Operational
No runs. Reading PDFs: the `pdf` skill or PyMuPDF in `conda run -n GraduationProject python` (annotations:
`page.annots()`); write any helper script to the session scratchpad. Numbers: CSVs in
`Thesis-writeup/Results/18-10-2026/CSV/` (`res_structure.csv`, `res_generalisation.csv`, `res_obc_*.csv`,
`res_errors.csv` for per-record values). Feedback is in Dutch and English; reply to Dirk in English.

## 14. Delegation
None. The files are named; read them inline so the reasoning that links feedback to paper conventions stays in one
context.
