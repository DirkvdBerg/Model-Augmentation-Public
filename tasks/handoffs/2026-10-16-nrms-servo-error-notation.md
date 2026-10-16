# Handoff: find whether the literature has a standard name and symbol for a simulation error normalised by the servo error
**From**: session of 2026-10-16 (Sections V and VI) | **Branch**: Augmentation | **Effort suggested**: high (literature
search over system identification and motion control; no code)

## 1. Task
Find out, from the literature, how the following error measure is named and written, and whether our name `NRMS_e` is
standard or should be replaced. The measure, per test record and stage axis j of a closed-loop record:
NRMS_e,j = RMS(y_hat_j - y_j) / RMS(y_j - r_j), with y the measured position, y_hat the model's simulated (free-run,
closed-loop) position, r the reference, and e = y - r the servo (tracking) error. It is a normalised RMS of the
simulation error with the servo error as the reference instead of the output itself, because in a tracking record RMS(y)
is dominated by the reference motion (about 0.17 m against RMS(e) of 1e-5 to 6e-5 m). Use the `deep-research` skill
(CLAUDE.md workflow rule D-121), with its step 0 FRAME. Report in chat for Dirk: (a) the usual definitions of NRMS / NRMSE
and BFR / fit in system identification (with their normalisation: by RMS(y), std(y), range, or y - mean(y)), each with a
verified source and equation number; (b) any paper or textbook that normalises a model or prediction error by the
tracking / servo error, or evaluates closed-loop models on the servo error, and what symbol or name it uses (motion
control, closed-loop identification, digital twins of servo systems, ILC); (c) a recommendation: keep `NRMS_e`, or use a
name/symbol found in (b), with the reason; (d) the Research Log the skill requires.

## 2. Out of scope
- Editing any thesis file, `docs/decisions.md` or code; this is advice only.
- Whether NRMS_e is the right metric (decided, D-256); only its name, symbol and literature support are in question.
- Other metrics (BFR as primary metric, settling metrics).

## 3. Where things stand
Branch `Augmentation`, uncommitted thesis edits. No run in flight. The metric is defined in
`Thesis-writeup/Writing/sections/05_setup.tex` (eq:setup_rms, Section V-D) with a `\todo` asking for NRMS/BFR citations.

## 4. Established and verified
- Hoekstra et al. 2026 (`literature/closed-loop-id/hoekstra2026_lfr-augmentation-fp-models.pdf`) and Hoekstra et al.
  2025 EJC report the absolute simulation RMSE; the encoder paper defines RMSE as sqrt(1/N sum ||y_k - y_hat_k||^2), Eq. 36.
- Drenth et al. 2025 (`literature/lpv-lfr/drenth2025_lpv-lfr-rational.pdf`) Eq. 22: BFR normalised by ||y - y_bar||
  (verified by the cross-check session 2026-10-16).
- Gyorok et al. 2025 L4DC (`literature/Orthogonality/Hoekstra - Orthogonal projection-based regularization for efficient model.pdf`)
  reports test NRMS %, citing Beintema 2021 for the definition (not defined in the paper itself).
- The subscript `e` is this project's own notation (`Thesis-writeup/Documentation/THESIS-RESULTS.md` Sec. 2, D-256).

## 5. Assumed but not verified
- Beintema et al. 2021 (SUBNET) defines NRMS as RMS(y - y_hat) / std(y): check the paper.
- That no standard name exists for normalisation by the servo error: the point of this task.

## 6. Tried and failed
None.

## 7. Achieved
None for this question.

## 8. The open question
Is there an established name/symbol for "simulation error relative to the servo error"? Candidates: keep NRMS_e (own
notation, defined in V-D); a name from motion control or closed-loop identification literature if one exists.

## 9. Next action
Run the deep-research skill on section 1's question (step 0 FRAME first), then report (a) to (d) in chat.

## 10. Acceptance criterion
Every definition and name in (a) and (b) carries a source that was opened and read (page, equation or section);
(c) is one recommendation with its reason; (d) is present.

## 11. Read these first
1. `.claude/skills/deep-research/SKILL.md` (the procedure).
2. `Thesis-writeup/Writing/sections/05_setup.tex`, subsection Metrics (the definition as written).
3. `docs/references.md` (cite keys; Beintema 2021/2023 entries).

## 12. Do not
- Edit files; change the metric; cite a source not opened.

## 13. Operational
Read-only; web search and PDFs only. No conda needed.

## 14. Delegation
As the deep-research skill prescribes for one question (one sub-question per agent, at most a few).
