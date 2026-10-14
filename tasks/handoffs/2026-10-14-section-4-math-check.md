# Handoff: independent check of thesis Section IV maths against its cited sources and the code, revnote-aware
**From**: session of 2026-10-14 (Section IV drafting) | **Branch**: Augmentation | **Effort suggested**: xhigh (equation-level reading of two Gyorok papers and Kessels, plus code tracing; the section is the thesis's own contribution)

## 1. Task
Dirk's words: "check your math against the sources it references and our code, while it takes the revnotes in mind."
Check every equation, mathematical claim and citation in `Thesis-writeup/Writing/sections/04_obc.tex` (Section IV,
drafted 2026-10-14) against (a) the cited passage, read from the rendered PDF page in your session, and (b) the code
that implements it. Read-only critique: report findings to Dirk in chat; edit nothing. A difference between the text and
the code is a finding UNLESS the text marks it with a `\revnote` (the text describes the FINAL method; a revnote says
the current runs differ) or a `\todo`; in that case check instead that the revnote states the difference correctly and
that it has a matching item in `Thesis-writeup/Documentation/FINAL-RUN-FIXES.md`. Quote every source you rely on: the
paper passage (page and equation), the code (`file::function`, line quotes only per the Code Quote Verification rule),
or the decision entry (`D-` number).

## 2. Out of scope
- Editing any file. Findings go to chat; Dirk decides what changes (he writes the thesis sentences himself).
- Prose style, concision, wording. Report only correctness: maths, frames, citations, code agreement, revnote accuracy.
- Re-opening decisions Dirk made on 2026-10-14: Section IV has two subsections (no IV-C, no proposition or proof); the
  recovery result is one short paragraph with `eq:obc_eps` displayed; the offset column is omitted (open `\todo`).
- Sections I to III and V to VIII, except where IV cites them (`eq:mdelta_map`, `eq:aug_norm`, `eq:aug_norm_maps`,
  `eq:aug_loss`, `eq:aug_cl_step`, `eq:aug_enc_sim`, `eq:eom`, `sec:params`) and the two refs repointed to `sec:obc_map`
  (`01_introduction.tex` line 146, `06_results.tex` line 95).
- Code changes, reruns, compiling `main.tex`, git writes.

## 3. Where things stand
Branch `Augmentation`, last commit `0246725`, tree dirty (thesis sections, docs; Dirk commits). `04_obc.tex` (about 373
lines) was edited 2026-10-14: opening; IV-A paragraphs 1 to 8 including the new recovery paragraph and `eq:obc_eps`;
IV-B with one new `\revnote` (polish acceptance). The header's `% MUST ESTABLISH` block (top of the file) is the agreed
content. No run in flight.

## 4. Established and verified (2026-10-14, re-check freely)
- Recovery formula, numerically (scratchpad `prop_check.py` of session 9d7704bb): with a baseline linear in beta and
  theta_aux = Phi^+ F on the same stack, the one-step minimiser is beta* + Phi^+ eps to 3e-16 for three network fields;
  uncorrected it is beta* + Phi^+(eps - F); for a curved map the relative deviation scales with the step (8.9e-4,
  8.9e-5, 8.9e-6 at scales 1e-1, 1e-2, 1e-3).
- Passages read from rendered pages: Gyorok 2026 (arXiv 2511.01321v3) Eqs. (2), (4) to (13), Assumption 1, Lemma 3,
  Sec. 3.2 text after Eq. (12), Sec. 4.1 Eqs. (14) to (21), Cond. 4, Remark 5, Assumption 6, Thm. 7, Sec. 5.2 end
  (p. 9), Sec. 6 (pp. 9 to 10); Gyorok 2025 L4DC Sec. 2.2, 2.3 (Eq. 7 normalisation), Sec. 3 Eqs. (8) to (14), Remark 1,
  text before Remark 2, Remark 2, Sec. 4 Eqs. (15) to (19) and the paragraph after (p. 8); Kessels thesis p. 166 (PDF
  p. 193) and p. 183 (PDF p. 210). Logged in `Thesis-writeup/Documentation/CITATION-LOG.md` (rows 34 to 52, 71 to 83,
  and three rows added 2026-10-14).
- Code read 2026-10-14 and found to match the text: `GD/obc_gantry.py` (`build_reference_set`: q = P^-T y, fourth-order
  central difference, two samples trimmed per record end, x_a = 0, recorded u, centred normalisation;
  `GantryOBCCorrection.forward`: subtracts the tangent at the step's own state and input, zero rows on x_a;
  `make_basis_builder`), `FS/obc.py` (`OBCBasis.coefficient` = V S^-1 U^T F; `OBCLifecycle.maybe_refresh` per epoch at a
  detached free coordinate), `FS/interconnect.py` (`forward` line with `xp - _obc(...)`; `loss`;
  `_sync_prediction_state_for_validation`, basis rebuilt with `refresh_basis=True` after the best checkpoint is
  restored). Thesis config (`scripts/gantry/gantry_interconnect_dynamic.py`, `_THESIS_FIXED`): `obc_space='tangent'`,
  `ann_route_ix=tuple(range(6 + n_a))`, `obc_refresh` default `'epoch'` (`GD/config.py`).
- Prefixes as in the README source map: `GD` = `scripts/gantry/gantry_dynamic/`, `FS` = `model_augmentation/fit_systems/`.

## 5. Assumed but not verified (check these first)
- The first-order sensitivity sentence of IV-A paragraph 1: each column of df/dbeta is the RK4 propagation of
  -M(Y)^{-1}(dM/dbeta_j qdd + dC/dbeta_j qd + dK/dbeta_j q). Derived by hand for the CT vector field; check it against
  `eq:eom`, `eq:aug_ct` and the normalisation `eq:aug_norm_maps` (T_x scaling, centring).
- "The map `eq:mdelta_map` has a nonsingular Jacobian for m_Delta,0 != 0" (nine log coordinates plus the tanh map with
  dm/dbeta_Delta = m_Delta,0 at beta = 0). Check for every beta, not only beta = 0.
- The offset sentence: Gyorok's Gamma = f - Phi theta_bar adds the direction of f_base to the protected range, so
  span[Phi Gamma] = span[Phi f_base]. Also the two omission reasons (D-186 2026-09-10 reasons 1, 2): reason 2 (a parameter
  derivative is invariant to the centring mu_x, the offset is not) must hold in the centred coordinates of D-249.
- The recovery paragraph: stacks over I' (tuples with a successor) versus the code's coefficient over all of I; whether
  "unique to first order" and "for every theta_aug" are stated no stronger than the linear-algebra result supports;
  whether the "adapted from Thm. 7, which in addition assumes exact fit (Assumption 6)" attribution is exact.
- Frames: Phi, f_aug, eps, theta_aux all in the normalised centred coordinates of `eq:aug_norm`; the tuple input is the
  recorded actuator force in the same frame as u in `eq:aug_norm_maps`; the code's `u_stage` naming ("stage forces")
  versus the text's "recorded input" / actuator force.
- IV-B per-phase paragraph (S1, S2, Adam, validation, restore, polish) against `FS/ps2_opening.py`, `FS/interconnect.py`
  and `FS/lbfgs_polish.py` (the polish has no OBC logic of its own, so the basis stays at the restored checkpoint's beta;
  the final model's beta_lin is then the pre-polish point).

## 6. Tried and failed
- Citing a passage from memory produced wrong section numbers twice in earlier sessions -> read the rendered page.
- Gyorok 2025 Remark 2 was once mislabelled as the simulated-states fallback (D-185 notes it); the fallback is the
  paragraph before Remark 2 in Sec. 3; Remark 2 is the per-epoch state-estimate update.
- Bash lost `cat`/`awk`/`conda` mid-session before; use PowerShell or small Python scripts if that happens.

## 7. Achieved
Section IV drafted and self-reviewed 2026-10-14 (environments balanced, no `obc_scope` refs left). Not compiled.

## 8. The open question
Nothing blocked. Open `\todo`s in IV are Dirk's decisions, not findings: omitted offset (supervisor confirmation), data
tuples versus the closed-loop baseline simulation `eq:aug_enc_sim`, the projection figure, the refresh measurement and
the reason against a fixed point, model notation. Report only if a `\todo` misstates its own facts.

## 9. Next action
Read `04_obc.tex` in full, then for every display and every sentence with a citation or a mathematical claim, write one
row: location, claim, source passage (quoted or transcribed with page and equation), code (`file::function`), verdict
(OK / WRONG / IMPRECISE / UNSUPPORTED / REVNOTE OK / REVNOTE WRONG), and for every non-OK the concrete fix. Also check
each `\revnote` in IV and, for each FINAL-RUN-FIXES item (1, 2, 5 to 10), whether IV states anything it changes without a
revnote. Deliver the table in chat, most severe first.

## 10. Acceptance criterion
Every display and every cited or mathematical sentence in `04_obc.tex` has a row with a quoted source and a verdict;
every text-code difference is either covered by a correctly worded `\revnote` with a FINAL-RUN-FIXES item or reported as
a finding; every derived formula you call OK was checked by derivation or a scratchpad script, with the result stated.

## 11. Read these first
1. `Thesis-writeup/Writing/sections/04_obc.tex` (all, including the header MUST ESTABLISH block).
2. `literature/Orthogonality/gyorok2026_orthogonal-by-construction-augmentation_IFAC-JSC_arXiv2511.01321.pdf` (pp. 2 to 5, 9 to 10).
3. `literature/Orthogonality/Hoekstra - Orthogonal projection-based regularization for efficient model.pdf` (Gyorok et al. 2025, pp. 4 to 8).
4. `Thesis-writeup/Writing/sections/03_augmentation.tex` lines 235 to 942 (the displays IV builds on and its revnotes).
5. `Thesis-writeup/Documentation/FINAL-RUN-FIXES.md`.

## 12. Do not
- Edit any `.tex`, doc or code file; report only.
- State "Gyorok says" or "Kessels says" without the passage read in your session.
- Flag the removed IV-C, the removed proposition proof, or prose style as findings.

## 13. Operational
PDF pages: Read tool with `pages`. Kessels: `literature/augmentation/kessels2025_ai-control.pdf`, PDF page = thesis page
+ 27 for Ch. 5 (p. 166 is PDF 193). Numerical checks: a script in your scratchpad, run with
`conda run -n GraduationProject python <file>`. Code quotes with line numbers only after `scripts/verify_quote.py`
prints MATCH OK.

## 14. Delegation
None. The check needs one context holding the section, the papers and the code together.
