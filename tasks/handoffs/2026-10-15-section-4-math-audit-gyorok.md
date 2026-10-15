# Handoff: audit the maths of thesis Section IV against the two Gyorok papers (and the code)
**From**: session of 2026-10-15 (Section IV restructure) | **Branch**: Augmentation | **Effort suggested**: xhigh (equation-level reading of two papers, derivations, small numerical checks)

## 1. Task
Dirk's words: "check the current math, especially against the gyorok sources ... make sure we use the math correctly."
Audit every display, every symbol and every sentence that states or cites mathematics in
`Thesis-writeup/Writing/sections/04_obc.tex`. For each, establish that (a) the mathematics is correct, (b) every
statement attributed to Gyorok et al. matches the cited equation, assumption, lemma, theorem or passage EXACTLY in
scope and conditions, and (c) the display matches the code where the code implements it. Read-only: report to Dirk in
chat, most severe first; edit nothing.

YOU MUST READ THE SOURCES FROM THEIR RENDERED PDF PAGES IN YOUR SESSION BEFORE JUDGING ANY ROW. Do not rely on memory,
on the citation log, on the decision log, or on this handoff for what a paper says. Required reading, in full:
1. `literature/Orthogonality/gyorok2026_orthogonal-by-construction-augmentation_IFAC-JSC_arXiv2511.01321.pdf`:
   pages 1 to 6 (abstract, Secs. 2, 3, 4 including Eqs. (1) to (25), Assumptions 1, 6, 8, Lemma 3, Condition 4,
   Remark 5, Theorem 7, Sec. 4.2) and pages 9 to 10 (Sec. 5.2 end, Sec. 6). Read p. 6 (Sec. 4.4, Eq. (39)) as well.
2. `literature/Orthogonality/Hoekstra - Orthogonal projection-based regularization for efficient model.pdf`
   (Gyorok et al. 2025, L4DC): pages 4 to 8 (Secs. 2.2, 2.3, 3, 4: Eqs. (4) to (19), Example 1, Remarks 1, 2, the
   paragraph before Remark 2, the paragraph after Eq. (19)).
3. `literature/augmentation/kessels2025_ai-control.pdf`: PDF page 193 (thesis p. 166, Sec. 5.3.1.3).
For every row you report, quote or transcribe the exact passage or equation you checked against, with page and number.

## 2. Out of scope
- Editing any file. Dirk writes the thesis text; you report.
- Prose style, concision, wording. Correctness only.
- Re-opening Dirk's decisions of 2026-10-14/15: two subsections (IV-A Gyorok's construction as adopted, IV-B
  extension to the gantry); no regularisation/penalty discussion (the 2025 paper is cited only where its concepts are
  used); no roadmap; added states in IV-B, not the opening; Gamma left out (D-254 item 1 as amended); reference tuples
  from data (D-254 item 2, D-252); per-epoch refresh (D-254 item 3); no separate identification-problem display (the
  cost is unchanged); no pseudoinverse "+" notation, (Phi^T Phi)^{-1} Phi^T throughout; the measures in
  eq:obc_overlap (cos angle, c_j, rho) as asked by the supervisor. Report a problem with any of these only if the
  MATHS or the CITATION is wrong, not the choice.
- Dirk's own text: the section opening (the paragraph before `\subsection`) and his `\begin{outline}` blocks. Check
  their maths too, but report fixes as errors only (never reworded versions).
- Other sections, except the displays IV uses: `eq:eom`, `eq:aug_ct`, `eq:aug_norm`, `eq:aug_norm_maps`,
  `eq:mdelta_map`, `eq:aug_loss`, `eq:aug_cl_step`, `sec:params`.

## 3. Where things stand
Branch `Augmentation`, last commit `0246725`, tree dirty (Dirk commits). Section IV was restructured on 2026-10-15:
- IV-A `sec:obc_construction`: `eq:obc_gyorok` (Gyorok 2026 Eqs. (8), (9), (11), Lemma 3 in his notation) with
  claims on least squares, full column rank (Assumption 1, Eq. (6)), data-driven (Sec. 3.2), uniqueness under
  Assumptions 1 and 6 (Sec. 4.2).
- IV-B `sec:obc_map`: `eq:obc_sens` (Jacobian, from 2025 Eq. (15)); full rank through the ten identifiable
  combinations; `eq:obc_tuples`; `eq:obc_coef`; `eq:obc_orth` (our definition); `eq:obc_overlap` (cos angle, c_j,
  rho, Pi); `eq:obc_corr`; identification in prose; `eq:obc_eps`; `eq:obc_recovery`.
Not compiled (Dirk's editor rebuilds); displays fit 252 pt in a scratch compile.

## 4. Established and verified (re-check freely; these are not a substitute for reading the papers)
- Two earlier audits (2026-10-14) confirmed against the code: the normalised RK4 Jacobian (2.8e-16), the coordinate
  map determinant nonzero for finite beta (2.2e-16), centring invariance of the parameter Jacobian, the stencil
  moments, and the common-stack recovery identity (4.9e-16); script `.codex-tmp/section4-math-check/check_math.py`.
- Code: `GD/obc_gantry.py::build_reference_set`, `::GantryOBCCorrection.forward`, `FS/obc.py::OBCBasis.coefficient`
  (V S^-1 U^T F), `::OBCLifecycle.maybe_refresh`, `FS/interconnect.py::forward` (`xp = xp - _obc(x, u, obc_pass)`),
  `Thesis-writeup/Code/Evaluation/eval_overlap.py` (rho on the CORRECTED network output on held-out tuples, and rho of
  eps; no c_j yet). `GD` = `scripts/gantry/gantry_dynamic/`, `FS` = `model_augmentation/fit_systems/`.

## 5. Assumed but not verified (check these first)
- IV-A: that "theta_aux is the least-squares fit of F by Phi", "the remainder is orthogonal on the training data",
  "theta_aux exists for full column rank", "needs an identifiable baseline and a weakly persistently exciting input
  (Eq. (6))", "data-driven ... the projection depends on their distribution (Sec. 3.2)", "these data may be the
  training data, a subset or a synthetic set (Sec. 3.2)", and "unique under full column rank and an exact fit
  (Assumption 6, Sec. 4.2)" each match Gyorok 2026 exactly, including which assumptions each result needs.
- IV-B: the 2025 concepts as cited: state transition stacked over the data (Secs. 3, 4, Eqs. (9), (15)); Jacobian
  (Eq. (15)); offset column Gamma (Eqs. (17), (18)); simulated states when not measured (Sec. 3, paragraph before
  Remark 2); per-epoch recomputation (Sec. 3, Remark 2: for state estimates, not the linearisation point);
  linearisation point updated per cost evaluation or fixed at nominal (Sec. 4, after Eq. (19)).
- IV-B: "with the fourteen raw parameters the Jacobian has four structural null directions" and "in beta the map is
  invertible for m_Delta,0 != 0, so Phi spans the directions of the ten combinations" and "full column rank then
  depends only on the excitation, as in Eq. (6)": is the analogy with Gyorok's Eq. (6) (identifiability plus weak
  persistence of excitation) stated correctly for a Jacobian of a nonlinear state transition?
- IV-B: "evaluates its regressor at the measured data (Eq. (5))": Gyorok's x_k are lagged measured IO; is the
  analogy with reconstructed states correct as worded?
- `eq:obc_orth` "which eq:obc_coef satisfies for every theta_aug, as in Lemma 3": correct?
- `eq:obc_overlap`: cos angle definition; c_j = cos angle(phi_j, d); rho(d) = max over ran Phi of cos angle(v, d) =
  ||Pi d|| / ||d|| (verify the max identity, including sign and the case d in ran Phi or d = 0); Pi the orthogonal
  projector; "Pi f_aug = Phi theta_aux"; "rho(d) = 0 iff every c_j(d) = 0 iff orthogonal in the sense of eq:obc_orth".
  Is anything missing for d = 0 or for c_j's sign?
- `eq:obc_corr` and "As in their prediction (Eqs. (9), (13))": Gyorok subtract phi(x_k) theta_aux at the prediction
  regressor with theta_aux fixed after training (Eq. (13)) and as a function of theta_a in training (Eq. (9)); is the
  thesis statement consistent with both?
- "As in Eqs. (9), (10), the coefficient is a function of the network parameters and is differentiated": supported?
- `eq:obc_eps`, `eq:obc_recovery` and the paragraph after: first-order minimisers with and without the correction on
  ONE common stack; "equals beta* iff Phi^T eps = 0"; attribution "adapted from Thm. 7, which in addition assumes
  exact noise-free reproduction (Assumption 6), Cond. 4, Eq. (21)"; Remark 5 as cited; "a surrogate, not a statement
  on the trained model". Check the analogy eps <-> Gyorok's Delta (their Eq. (15) also has noise E; ours holds
  reconstruction error).
- The supervisor `\revnote` after `eq:obc_sens` (outside wip): its maths ("exactly the condition under which the
  estimate does not depend on the network"; Gamma "would exclude only one direction ... uniform scaling") correct?
- Notation consistency: Phi (single tuple, 6x10) vs bold Phi (stack, 6N x 10); theta_aux; beta_lin; frames
  (normalised, centred coordinates of eq:aug_norm); dimensions in every where-clause.

## 6. Tried and failed
- Citing from memory gave wrong section numbers twice in earlier sessions -> read the rendered page (this is why
  section 1 commands it).
- Remark 2 of the 2025 paper was once placed in Sec. 4; it is in Sec. 3.
- Earlier text claimed uniqueness from Assumption 1 alone; Gyorok 2026 shows uniqueness of theta_b with Assumptions 1
  AND 6 (Sec. 4.2); Assumption 1 alone gives existence of theta_aux and a unique baseline-only solution (text after
  Eq. (7)).
- A recovery statement on a subset of the coefficient's stack was wrong (counterexample in D-252); the stacks are now
  one set I.

## 7. Achieved
Section IV restructured and displays width-checked on 2026-10-15; earlier audits applied (D-252 to D-255,
FINAL-RUN-FIXES items 11, 12).

## 8. The open question
Nothing blocked.

## 9. Next action
Read the three sources of section 1 from their rendered pages. Then read `04_obc.tex` in full and write one row per
display and per sentence with a mathematical statement or a Gyorok/Kessels citation: location, claim, the transcribed
source passage (page, equation/assumption/lemma number) or `file::function`, verdict (OK / WRONG / IMPRECISE /
OVERSTATED / UNSUPPORTED), and for each non-OK the concrete correction. Verify every derived formula by derivation or a
scratchpad script and state the result. Deliver in chat, most severe first.

## 10. Acceptance criterion
Every display and every mathematical or cited sentence in `04_obc.tex` has a row with a transcribed source passage
read in your session and a verdict; every attribution to Gyorok states the same conditions as the paper; every derived
formula called OK was checked.

## 11. Read these first
1. The three PDFs of section 1 (mandatory, rendered pages).
2. `Thesis-writeup/Writing/sections/04_obc.tex` (all, with the header MUST ESTABLISH block).
3. `docs/decisions.md` entries D-190, D-192, D-252 to D-255 (context only, never as a source for what Gyorok says).

## 12. Do not
- Judge a Gyorok attribution without the passage read in your session.
- Edit any file; propose reworded versions of Dirk's own sentences (errors only).
- Re-open the decisions listed in section 2.

## 13. Operational
PDF pages: Read tool with `pages`. Numerical checks: a script in your scratchpad, run with
`conda run -n GraduationProject python <file>` (write scripts with the Write tool; shell heredocs have mangled
backslashes in this environment). Code quotes with line numbers only after `scripts/verify_quote.py` prints MATCH OK.

## 14. Delegation
None. One context must hold the papers, the section and the code together.
