# Handoff: decide with Dirk which results the thesis shows and how (Results section first, then the appendix), on the model of Hoekstra et al. 2026 and Györök et al.
**From**: polish-rerun and final-evaluation session (2026-10-20 to 10-24) | **Branch**: Augmentation | **Effort suggested**: high (judgement on what each result must argue, against reference papers and real numbers; replies stay short)

**Dirk's current message always overrides this plan. When he asks a question, answer that question first, in plain words, in a few sentences.**

## 1. Task
Work out with Dirk, the author, which results the thesis shows and how they are shown, for
`\section{Results}\label{sec:results}` in `Thesis-writeup/Writing/sections/06_results.tex` (Section VII in the
build) and, second, the result parts of the appendix (`99_appendix.tex`: `app:splits`, `app:choices`, `app:frf`).
His words (2026-10-24): "I'll need to discuss ... what results to actually show in the results section ... I think we
can learn a lot from Hoekstra his paper, and Györök his papers. I want your help in ... discussing the results to show,
and how to show them." On scope: "also the appendix, but the most important part is the results." Use the
`thesis-section` skill. Work in three steps, each confirmed by Dirk before the next:
1. **What the section shows now.** Read `06_results.tex` and the appendix result parts in their final reading (marks
   applied). For every table, figure and claim, give in one line: which research aspect it answers (4.2), what it
   shows (rows, columns, axes, split), and what is unclear, redundant or missing. Use the final numbers (4.3), not the
   text's: the text and every figure and table PDF are from the preliminary 2026-10-18 evaluation.
2. **The models.** Read the results (experiment) sections of `hoekstra2026lfr` and `gyorok2026obc` first, then
   `gyorok2025l4dc` and `hoekstra2026encoder` (files in 4.4). For each, give in a few lines with PDF pages: which
   experiments, which tables and figures, which metric, what is in the main text and what in an appendix, and how each
   result is argued. Quote the paper (a few words, with the page) before any claim about it.
3. **The proposal.** An outline per research aspect: which result shows it (table or figure), main text or appendix,
   exactly what it contains (models as rows or curves, columns or axes, split, records), the criterion it is judged
   against, and the one claim it supports. For every current item, say kept, changed (how), moved (where) or cut (why).
   Open points marked (?). After Dirk approves the outline, carry it out one item per turn: the figure or table script
   in `Thesis-writeup/Code/Figures/` first, regenerated from the final CSVs (13), then the text, shown as rendered
   Now / After and applied with marks only on his "yes", "apply" or "do it".

## 2. Out of scope
- Any training, evaluation or server run. The final CSVs (4.3) are the data. If a proposed result needs a number that
  no CSV holds, say so in one line with the evaluation script that would produce it, and let Dirk decide.
- Sections I to VI and the Conclusion. If a result change needs a change elsewhere (for example a claim in the
  Introduction), name the sentence in one line and leave it.
- The Discussion subsection (`sec:discussion`, inside `06_results.tex`) beyond noting what a changed result changes
  there.
- The `\revnote` and `\todo` items in `06_results.tex`: working notes, not requirements. Do not plan around them;
  Dirk raises one if he wants it handled.
- The `\ifdraft` bullet blocks: never mention or edit them.
- Thesis LaTeX builds (his editor rebuilds on save), git writes, `kamtin-data/`.

## 3. Where things stand
Branch `Augmentation`, dirty tree (Dirk commits himself). The final campaign is complete: every grey-box run has its
final model under the polish rule, and `final-eval` (server jobs 90467, 90468, 2026-10-23/24) produced the final
CSVs, now local (4.3). Figure and table outputs in `Thesis-writeup/Writing/figures/res_*.pdf` and
`Writing/tables/res_*.tex` are still the 2026-10-18 preliminary versions (R1 and R2 at a preliminary checkpoint, four
polishes missing). Dirk edits the thesis in parallel and saves often: re-read a paragraph right before editing it.

## 4. Established and verified

### 4.1 How to propose
Rendered Now / After, never LaTeX before apply; change only the words Dirk names; his wording pattern; marks of
`Thesis-writeup/Writing/util/format.tex` (`\chg[own]`, `\add[own]{\kwtwo{...}}`, `\del`, `\mv`/`\delmv`), a
`% FEEDBACK (<who>, 2026-10-2x): <reason>` line above each change. All of this is written out, with an accepted
example, in `tasks/handoffs/2026-10-24-section6-setup.md` sections 4.1, 4.3, 4.4 and 6: read those four sections.
For the outline format: `feedback_outline_format.md` (headers, short one-line bullets, open points (?)).

### 4.2 What the results must answer
Research question and three aspects, `01_introduction.tex:400-417` (final reading): (1) which augmentation structure
and training method let the learned component capture the missing dynamics from closed-loop data; (2) to what extent
orthogonal-by-construction augmentation keeps the physical parameters interpretable, compared to unconstrained
estimation; (3) how the augmented model generalises to held-out operating points and unseen motion profiles, compared
to the physics baseline and a black-box model. The former Aspect 1 (LPV vs LTI) is no longer an aspect; its figure is
in the appendix (`fig:res_lpv_lti`).
Standing rules for results: argue every result against a criterion stated before it (`feedback_thesis_argue_results.md`,
Maarten); every metric exists for training, validation and test, test in the main text (D-269); the primary metric is
NRMS_e per axis (`05_setup.tex` eq:setup_rms); errors in metres in scientific notation
(`feedback_units_scientific_metres.md`); lead with the achieved number beside its reference
(`feedback_lead_with_the_number.md`).

### 4.3 The data: final CSVs
`Thesis-writeup/Results/24-10-2026/CSV/` (16 files; bundle `final-results-2026-10-24.tgz` beside it). Checked
2026-10-24: `res_checks.csv` ok for all nine grey-box runs (evaluated validation NRMS_e equals the final checkpoint's
score, relative difference at most 1.9e-11); no "preliminary" label in any file; every grey-box run and the black box
on train (18 records), validation (6), test (11, I1 to I6 and E1 to E6 without E5) and the multisine twins (7); the
OBC files (`res_obc_accuracy`, `_bias`, `_overlap`, `res_negation`) on train, validation and test tuple stacks.
Runs (folder ids in `docs/gantry-augmentation-problem-log.md` Section 12, rows "FINAL CAMPAIGN"):

| Run | Arm | Start | n_a | Opening | Final validation NRMS_e |
|-|-|-|-|-|-|
| R1 (final model) | OBC | detuned | 2 | PS | 0.077464 |
| R2 | OBC | correct | 2 | PS | 0.078344 |
| R3 | U | correct | 2 | plain | 0.252132 |
| R4 | U | detuned | 2 | PS | 0.073463 |
| R5 | U | correct | 2 | PS | 0.073821 |
| R6 | U | correct | 0 | plain | 0.286834 |
| R7 | U | correct | 8 | PS | 0.071448 |
| R13 | OBC | detuned | 0 | plain | 0.292587 |
| R14 | U | detuned | 0 | plain | 0.287024 |
| R8 (black box) | SUBNET, n_x 16, width 16, lr 1e-3 | n/a | n/a | n/a | 0.4819 (best validation) |

Per run the CSVs also hold the physics part alone, the baseline at the detuned start and at the true parameters,
and for R5 the frozen-Y LTI baseline (`res_errors.csv` column `model`).
Polish (D-255, D-276): kept in R1, R3, R5, R13 (validation change at most -0.065 %); rolled back in R2, R4, R6, R7, R14;
in R3 and R5 the ten combinations did not change at 4 printed decimals.
Black box (D-264, problem log row "FINAL CAMPAIGN R8"): the 11 configurations the thesis states; best validation at
update 78000 of 80000, not plateaued (unresolved convergence); indices 4 and 10 failed (non-finite loss), 1 and 8
diverged with a finite loss but are listed "completed" in `res_bb_selection.csv`; on its training records NRMS_e 0.31
(multisine) and 0.86 (motion only), against 0.087 and 0.042 for R4 (2026-10-18 evaluation; recheck on the final CSV).

### 4.4 Reference papers (local PDFs)
`Thesis-writeup/References/`: `Hoekstra-2026-LFR-model-augmentation-arXiv-v1.pdf` (`hoekstra2026lfr`, the
framework the thesis follows), `Gyorok-2026-Orthogonal-by-construction.pdf` (`gyorok2026obc`, the projection the
thesis extends), `Gyorok-2025-Orthogonal-projection-regularization.pdf` (`gyorok2025l4dc`),
`Hoekstra-2026-Encoder-initialisation-arXiv-v1.pdf` (`hoekstra2026encoder`). Map of all keys: `docs/references.md`.

### 4.5 The current set and its record
Main text (`06_results.tex`): intro (test set, NRMS_e against the noise floor, training and validation in the appendix);
`sec:res_structure` with `tab:res_structure`, `fig:res_time_trace`, `fig:res_frf`; `sec:res_obc` with `tab:res_obc`;
`sec:res_generalisation` with `fig:res_generalisation`; `sec:discussion`. Appendix: `app:frf` (`fig:res_frf_full`),
`app:splits` (`tab:app_splits`), `app:choices` (`fig:res_lpv_lti`, `tab:app_structure`, `fig:res_ps_spectrum`,
`tab:app_obc_full`, `tab:app_obc_na0`, `tab:app_bb_search`). Each output comes from one script in
`Thesis-writeup/Code/Figures/` that reads one CSV (`Figures/README.md`, `_fig_common.py`). Earlier plans: D-247
(2026-10-11, one figure per aspect) and `Thesis-writeup/Documentation/THESIS-RESULTS.md` section 3 (what each run
answers). They are the starting point, not a constraint.

## 5. Assumed but not verified
- That the statements in `06_results.tex` still hold with the final numbers (they were written from the 2026-10-18
  preliminary set). Settled in step 1 from the final CSVs.
- Which baseline the generalisation comparison uses: `05_setup.tex` names the true parameters, `06_results.tex` the
  detuned start; both rows are in `res_generalisation.csv` and the figure draws the detuned one (D-272 item 4). A
  "what to show" decision for Dirk in step 3.
- The noise-floor reference in the results (noisy minus noise-free twin, only where a twin exists): its definition is
  an open point in `05_setup.tex` (Metrics). Treat as (?) in the outline.
- That the black-box training-record numbers of 4.3 hold on the final CSV.

## 6. Tried and failed
- A results design made from the list of runs (inventory) instead of from what the thesis must show -> rejected
  (`feedback_design_not_inventory.md`) -> derive each result from an aspect and the claim it supports.
- Showing a result without arguing it against a stated criterion -> Maarten's objection
  (`feedback_thesis_argue_results.md`) -> every proposed item names its criterion.
- For the proposal and wording failures of earlier sessions, see section 6 of `2026-10-24-section6-setup.md`.

## 7. Achieved
The data are final and checked (4.3). Nothing of the results presentation has been discussed with Dirk since the
final campaign.

## 8. The open question
Nothing blocked.

## 9. Next action
Read `06_results.tex` and the appendix result parts in their final reading, and the final CSVs they draw on, then
post step 1 of section 1: per table, figure and claim, the aspect it answers, what it shows, and what is unclear,
redundant or missing, with the final numbers where they differ from the text.

## 10. Acceptance criterion
Dirk has approved an outline in which every research aspect has the results that answer it, each with its criterion,
its split and its place (main text or appendix); every current table, figure and claim is kept, changed, moved or cut
by his decision; the approved items are carried out (scripts regenerated from `Results/24-10-2026/CSV`, text applied
with marks); no number in the Results text differs from the final CSVs.

## 11. Read these first
1. `06_results.tex` in full, then the appendix result parts (4.5).
2. `tasks/handoffs/2026-10-24-section6-setup.md` sections 4.1, 4.3, 4.4 and 6 (how to propose, Dirk's wording).
3. The results sections of `hoekstra2026lfr` and `gyorok2026obc` (4.4).
4. `01_introduction.tex:400-417` (the question and the three aspects).
5. `Thesis-writeup/Code/Figures/README.md` (which script makes which output from which CSV).

## 12. Do not
- Edit before an explicit yes; a pasted phrase or a question gets an answer, not an edit.
- Show LaTeX before apply.
- Claim what a paper does before quoting it with the page.
- Use the 2026-10-18 CSVs (`Results/18-10-2026/`) or the preliminary PDFs as data.
- Start a server run or rerun an evaluation locally on full data.

## 13. Operational
Regenerate one output from the final CSVs (seconds, no simulation; overwrites the preliminary PDF or table in
`Thesis-writeup/Writing/`), from the repository root:
```bash
THESIS_RESULTS_DIR="$PWD/Thesis-writeup/Results/24-10-2026/CSV" conda run -n GraduationProject python Thesis-writeup/Code/Figures/<stem>.py
```
Regenerate an output only when its item is being discussed or after Dirk approved its change. Reading a CSV for
numbers needs no regeneration.

## 14. Delegation
None: a dialogue with Dirk on known files.
