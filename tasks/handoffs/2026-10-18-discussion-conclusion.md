# Handoff: write the "Discussion and Conclusion" section (08_conclusion.tex) from the frozen Results
**From**: session of 2026-10-18 | **Branch**: Augmentation | **Effort suggested**: high (argument and
interpretation from existing numbers; no code, no runs)

## 1. Task
Write Section VIII of the thesis, `Thesis-writeup/Writing/sections/08_conclusion.tex`, as one section titled
"Discussion and Conclusion": a discussion first, then a short conclusion. Use the `thesis-section` skill (v8 rules:
deletion test, expert register, one fact per sentence). The Results section (`06_results.tex`) is observational by
Dirk's decision: it states what each float shows and gives no verdicts and no mechanisms. This section therefore
carries (a) the explanation of every result that does not speak for itself, (b) the limits of each claim, (c) the
departures from the research plan, and (d) a concise conclusion that answers the research question with one
answer per aspect. Section 4 lists the discussion points found so far as EXAMPLES of these classes; read Results
and the method sections yourself and complete the list per class. Dirk wants the conclusion concise and the
discussion only as long as the points require.

## 2. Out of scope
- Editing any other section (06 Results included, 07_discussion.tex included: it is a hidden `wip` file since
  2026-10-11; its bullets are source material only). Flag conflicts in text.
- New runs, evaluation, figures or tables. Every number comes from Results, the appendix tables or the CSVs.
- The final R1/R2 rerun (Dirk launches it). Do not wait for it.
- A literature search (`deep-research`). Cite only entries already in the bibliography whose passage you read.
- The abstract (written last, after this section).

## 3. Where things stand
- Branch `Augmentation`, last commit `8e30b34`; tree dirty, including `06_results.tex` and `03_augmentation.tex`
  (Dirk commits). No run of this session in flight.
- R1 (final model, OBC) and R2 are still training on the server; their numbers in the thesis are PRELIMINARY
  (update 19500 of 33400, dagger in tables). Write R1-dependent claims so they survive small changes, and add a
  `\todo` where a claim depends on an R1 digit.
- `08_conclusion.tex` holds the heading "Conclusions and discussion", an `outline` block per part (partly outdated:
  it still lists the removed LPV aspect) and `wip` bullets. `main.tex` inputs 07 and 08; 07 is all `wip`.

## 4. Established and verified
Decisions (Dirk, 2026-10-18, this session):
- Option 2: Results stays observational; explanations, limits and departures go here. Hoekstra's papers put a
  one-sentence interpretation in Results and only summary plus one or two limit sentences in the conclusion
  (Hoekstra 2025 EJC Sec. 6); this thesis deliberately departs from that because it has more to argue.
- Heading "Discussion and Conclusion", discussion first.
- Aspects as numbered in the Introduction (`01_introduction.tex` l.274 to 289): 1 structure and training, 2 OBC
  interpretability, 3 generalisation. Interpretability is defined as a UNIQUE estimate of the identifiable
  combinations, explicitly not true-parameter recovery (`01_introduction.tex` l.152 to 157).
Numbers (test records, NRMS_e in %, `Thesis-writeup/Results/18-10-2026/CSV/`, tables in `Writing/tables/`):
- Aspect 1 (`tab:res_structure`): with multisine, baseline at true parameters 18.8/20.7/116.6 (X1/X2/Y), final model
  9.7/9.5/6.2, black box 47.8/47.8/35.9, floor 0.06/0.06/0.01. Without multisine baseline 4.5/4.6/3.4, final
  4.7/4.8/3.8, black box 104.8/97.0/127.7. Final model resonance 2.11 to 2.13 f_b, zeta 0.036 to 0.080 over Y;
  system 2.12 f_b, zeta 0.043 (`res_frf_modes.csv`). Validation (appendix): Y with multisine 6.4 (PS) vs 78.1
  (plain) vs 86.5 (no added states).
- Aspect 2 (`tab:res_obc`, detuned start): combination-error norm 30.0 start, 83.1 U, 47.7 OBC; correct start 58.3 U,
  35.4 OBC. Dampers c_g1, c_g2, c_y move most (U +43 to +46, OBC +23 to +30), mass combinations within 3.4 % of
  truth. Full model and physical part equal within 0.9 points. Network overlap U 0.84/0.85, OBC much lower; truth
  overlap 0.59. Without added states (R14 U vs R13 OBC) the order reverses: 48.8 vs 68.4. Difference between the
  two starts' estimates, hand-computed from `tables/res_obc_combinations_test.tex`: U 36.4, OBC 25.1 (revnote
  request at the end of 06; preliminary).
- Aspect 3 (`fig:res_generalisation`, per record in `res_generalisation.csv`): final model 6.2 to 14.1 on records
  with multisine, 2.5 to 6.2 without; each E record within 2.0 points of its I record; without multisine up to 1.2
  points above the baseline (E4, I6). Black box 74 to 193 without multisine, below the final model only on I2 X2
  (10.2 vs 11.4). Multisine twins: final below 10 on Y, baseline 48 to 87, black box 75 to 112.
- Black box also fails on TRAINING records without multisine: TR-T1 to TR-T4 at 72 to 106 % (train rows, model `bb`,
  `res_generalisation.csv`). So it is not overfitting: it does not fit those records at all. Its selected search
  configuration reached validation NRMS_e 0.482 at update 78000 of 80000 (`res_bb_selection.csv`,
  `tab:app_bb_search`, D-264; corrected 2026-10-18, the earlier 0.501 had no source in the thesis).
- Each E record leaves the training range in one direction (`fig:setup_coverage`, `tab:setup_records`).
- Single seed per model; the seed spread was never measured (`99_appendix.tex` hyperparameter paragraph).
Discussion points found so far (EXAMPLES; generalise per class):
- Unexplained result: black box underperforming (Quinten v1 p9 asked for an explanation).
- Unexplained result: augmentation slightly worse than the baseline without multisine.
- Unexplained result: the OBC ordering reverses without added states (Jasper v1 point 5 asked for this case).
- Limit: OBC does not recover the true values; truth overlap 0.59 means Xi is not orthogonal to ran Phi, so the
  first-order estimate is biased (`04_obc.tex` eq:obc_recovery, l.343 to 381); OBC holds only at x_bar = 0 for the
  one-step fit, not the closed-loop optimum with active added states.
- Limit: uniqueness itself is only partly shown (the two starts still differ, 25.1 vs 36.4).
- Limit: root cause analysis (Introduction motivation) compares unique estimates over time; valid only while the
  missing dynamics do not change (`08_conclusion.tex` outline, Dirk 2026-10-18).
- Limit: closed-loop training can hide plant error behind the controller (`03_augmentation.tex` l.841 todo).
- Limits: single seed, simulation only, one absorber plus friction as missing dynamics.
- Departures from the research plan: Aspect 1 (LPV vs LTI) removed, now a modelling choice in the appendix;
  controller transfer (E5) dropped although the Introduction motivates controller evaluation (l.63 to 69);
  real Telica data out of the results (user memory, 2026-09-24); settling dropped (D-241).

## 5. Assumed but not verified
- Why the black box fails. Candidates, none tested: it models the whole loop (r, u_ms) -> e without the controller
  and must learn the rigid-body and controller dynamics from servo errors of 1e-6 m; its capacity or training budget
  is too small; the objective weights records with multisine more. State the observed facts (fails on training
  records too) and give candidates as candidates.
- Why the augmentation is slightly worse without multisine. Candidate: the learned correction adds small errors
  where the baseline is already near its limit; not tested.
- Why OBC reverses without added states. Candidate: without added states the network must absorb the absorber
  statically, and much of that lies in the parameter directions (truth overlap 0.59), so the projection forces it
  into the parameters; not tested.
- Coulomb friction violates the recovery condition by construction (`07_discussion.tex` l.41, D-204): argued from
  the sensitivity of c_g1, not measured on the final data.
- Whether the final runs still build the encoder from true parameters (old revnotes in `03_augmentation.tex`
  l.592 to 612 say the final runs fix this, D-271). Check `docs/decisions.md` D-271 before mentioning it; if fixed,
  it is not a limitation.

## 6. Tried and failed
- Predicted first-order bias column beside the observed delta_j -> 100 to 1300 % predicted vs a few to 30 %
  observed -> outside the range of the expansion, and it holds for a one-step fit at x_bar = 0, not closed-loop
  training -> dropped (revnote at the end of `06_results.tex`). Do not use it as evidence.
- Results with verdicts and mechanisms in the paragraphs -> Dirk: "results section is not for conclusions" and
  "too messy" -> Results rewritten observational, numbers left to the floats. Keep explanations here, not there.

## 7. Achieved
- `06_results.tex` final in content (Dirk edited the wording 2026-10-18): opening plus three short subsections, one
  per aspect, floats after each, todos and revnotes at the end.

## 8. The open question
Nothing blocked. Order of the discussion: by aspect (1, 2, 3, then departures and general limits) or by class
(unexplained results, limits, departures). Recommendation: by aspect, so each paragraph pairs with a Results
subsection and the conclusion follows the same order.

## 9. Next action
Read the files of section 11, then write `08_conclusion.tex` in full under the heading "Discussion and Conclusion":
draft bullets per subsection (`\ifdraft` block, as in 06), the discussion, then the conclusion. Replace the
outdated `outline` blocks; keep the `wip` bullets until Dirk removes them.

Target style for a discussion paragraph (claim, evidence, limit, what would test it):
> The black box fails on every record without multisine, including the training records (72 to 106 %), so it does
> not fit these records at all rather than overfitting. It models the whole closed loop from the reference and the
> injected force, and must learn the rigid-body dynamics from servo errors of about 1e-6 m. A black box of the plant
> inside the known loop would separate these effects.

Target style for the conclusion: one or two sentences per aspect, a number only where it carries the answer, then
one sentence of the central limit and one recommendation.

## 10. Acceptance criterion
- Every discussion point is tied to a reported result (a float label or an appendix table) and states its limit.
- Every candidate explanation of section 5 is phrased as a candidate unless the section cites evidence for it.
- The conclusion answers each of the three aspects and the research question; it is shorter than the discussion.
- No sentence states a number that is not in section 4 or a float.

## 11. Read these first
1. `Thesis-writeup/Writing/sections/06_results.tex`: what the reader has seen.
2. `Thesis-writeup/Writing/sections/01_introduction.tex` l.149 to 289: interpretability definition, gap, aspects.
3. `Thesis-writeup/Writing/sections/08_conclusion.tex` and `07_discussion.tex`: existing outline and source bullets.
4. `Thesis-writeup/Writing/sections/04_obc.tex` l.343 to 410: what OBC guarantees and does not.
5. `Thesis-writeup/Documentation/THESIS-RESULTS.md` sections 1 to 3: aspects, rules, agreed presentation.

## 12. Do not
- Do not edit 06 or any section other than 08; do not delete `07_discussion.tex`.
- Do not bring back LPV vs LTI as a research result, or the predicted-bias comparison.
- Do not quote the OBC network-overlap digit (R1 label mismatch: update 18200 labelled 19500).
- Do not compile the thesis (user memory).

## 13. Operational
No runs. Numbers from `Thesis-writeup/Results/18-10-2026/CSV/` if a per-record value is needed; small helper
scripts go to the session scratchpad.

## 14. Delegation
None. All files are named; read them inline.
