# Handoff: second independent audit of thesis Section IV against its cited sources and the code
**From**: session of 2026-10-14 (Section IV drafting, after the first critique) | **Branch**: Augmentation | **Effort suggested**: xhigh (equation-level reading of two Gyorok papers and Kessels, code tracing, small numerical checks)

## 1. Task
Dirk's words: "give me a new prompt to audit this section against the sources."
Audit every equation, mathematical claim and citation in `Thesis-writeup/Writing/sections/04_obc.tex` against (a) the
cited passage, read from the rendered PDF page in your session, and (b) the code that implements it. Read-only: report
findings to Dirk in chat and edit nothing. The text describes the FINAL method: a text-code difference is a finding
unless a `\revnote` covers it; for each `\revnote`, check that it states the difference correctly and has a matching
item in `Thesis-writeup/Documentation/FINAL-RUN-FIXES.md` and a decision entry. Every reason the text gives for a
choice must trace to a quoted paper passage, a `D-` entry or the code; report any reason that does not. Quote every
source you rely on (paper passage with page and equation; code as `file::function`, line quotes only per the Code Quote
Verification rule; decision by number).

## 2. Out of scope
- Editing any file. Findings go to chat; Dirk writes the thesis sentences.
- Prose style and concision. Report correctness only: maths, coordinate frames, citations, code agreement, revnotes,
  traceability of reasons.
- Re-opening Dirk's 2026-10-14 decisions: two subsections (no IV-C, no proposition or proof); `eq:obc_eps` displayed;
  reference tuples from the data (D-254 item 2); per-epoch refresh (D-254 item 3); every tuple has a successor (D-252);
  polish compared after the basis rebuild (D-253). The offset column is an OPEN question for the supervisors (D-254
  item 1, `\todo` after `eq:obc_sens`): check that the `\todo` states both sides correctly, do not settle it.
- The remaining `\todo`s (offset question, projection figure, model notation) as findings in themselves.
- Other sections, except the displays IV cites (`eq:eom`, `eq:aug_ct`, `eq:aug_norm`, `eq:aug_norm_maps`,
  `eq:mdelta_map`, `eq:aug_loss`, `eq:aug_cl_step`, `eq:aug_enc_sim`, `sec:params`).
- Code changes, reruns, compiling `main.tex`, git writes.

## 3. Where things stand
Branch `Augmentation`, last commit `0246725`, tree dirty (Dirk commits). `04_obc.tex` was drafted 2026-10-14, audited
once (findings below, all applied), then edited again. It holds three `\revnote`s (reference set without successor;
polish compared with the pre-rebuild score; oracle polish gate, 0.1 %) and three `\todo`s. The header's
`% MUST ESTABLISH` block is the agreed content. Not compiled. No run in flight.

## 4. Established and verified
- First audit (2026-10-14, other session) confirmed, with numbers: the normalised RK4 sensitivity against the code
  (Jacobian agreement 2.8e-16); the coordinate map determinant nonzero for every finite beta (2.2e-16); span[Phi Gamma]
  = span[Phi f_base] (3.4e-17); centring invariance of the parameter Jacobian (2.8e-17); the fourth-order stencil
  moments [0,1,0,0,0]; the common-stack recovery identity (4.9e-16); RK4 Taylor remainder second order. Its script:
  `.codex-tmp/section4-math-check/check_math.py`.
- Findings of the first audit, all applied on 2026-10-14: recovery stack mismatch (now every tuple has a successor,
  stacks over one set I, D-252, FINAL-RUN-FIXES item 11); stale polish comparison (D-253, item 12); "no gradient through
  beta_lin" (was "through Phi"); sensitivity as fixed-state physical acceleration change scaled by T_x; dimension error in
  the offset reason (now "depends on the data only through their projection onto the range"); detached-coefficient claim
  narrowed to per-step derivative; full rank; Assumption 6 as noise-free reproduction; revnote threshold 0.1 %; "column
  span" instead of "affine range"; Remark 2 located in Gyorok 2025 Sec. 3.
- Passages logged in `Thesis-writeup/Documentation/CITATION-LOG.md` (gyorok2026obc, gyorok2025l4dc, kessels2025ai rows,
  including four added 2026-10-14).

## 5. Assumed but not verified (check these first; they changed after the first audit)
- IV-A reference-set paragraph: "as their later construction evaluates its regressor at the measured data" cited to
  Gyorok 2026 Eq. (5); "the tuples lie on the trajectories of the system, where the omitted dynamics act, rather than on
  those of the baseline at theta^0"; "they hold the measured next states that define eps"; "built once, before training,
  and depend neither on the trained parameters nor on the encoder" (`GD/obc_gantry.py::build_reference_set`,
  `::attach_obc`).
- "The tuples cover every sample at which this difference exists, both at the sample and at the next sample": the FINAL
  method per D-252; the current code is `k = np.arange(2, len(u) - 2)`; the revnote must say exactly this difference.
- The recovery paragraph after `eq:obc_eps`: stacks over I with x_{i+1}; "eps also holds the error of this
  reconstruction"; "to first order ... minimised by beta* + Phi^+ eps for every theta_aug"; the uncorrected minimiser;
  the Thm. 7 / Assumption 6 attribution.
- IV-A refresh paragraph: "each epoch thus minimises a fixed objective, while the linearisation point follows the
  estimate between epochs" against `FS/obc.py::OBCLifecycle.maybe_refresh` and `FS/interconnect.py::loss` (does anything
  else change the objective within an epoch, e.g. PS2 S2 or validation?); "adapts the per-epoch recomputation they allow
  for the state estimates in their linear case" against Gyorok 2025 Remark 2.
- The offset `\todo`: both sides as stated (Gyorok 2026 Eq. (39); Gyorok 2025 Eq. (18); D-190 "57 percent outside");
  D-186 reasons 1, 2.
- IV-B: "The polish is compared with the validation error of the model with the rebuilt basis" (final, D-253) and its
  revnote against `GD/training.py::run_lbfgs_polish` and `FS/lbfgs_polish.py::lbfgs_polish`.
- The opening: Gyorok 2026 Sec. 3.2 (loss of physical meaning) and Sec. 6 (state-space future work).

## 6. Tried and failed
- Citing from memory gave wrong section numbers twice in earlier sessions -> read the rendered page.
- A recovery statement on a subset of the coefficient's stack was wrong for the implemented model (first audit
  counterexample: Phi = [1,2,3]^T, F = [1,0,0]^T, two-row subset, shift -9/70 at eps = 0) -> sets now aligned.
- "Fixed before training" alone does not separate data tuples from a closed-loop baseline simulation at theta^0 (that
  simulation is also fixed); the text therefore rests the choice on the data reasons in section 5 (D-254).

## 7. Achieved
Section IV drafted, audited once, all findings applied; decisions D-252, D-253, D-254 and FINAL-RUN-FIXES items 11, 12
recorded on 2026-10-14.

## 8. The open question
Nothing blocked. The offset column is the supervisors' question (D-254 item 1), not yours.

## 9. Next action
Read `04_obc.tex` in full. Write one row per display and per sentence with a citation, a mathematical claim or a stated
reason: location, claim, source (quoted or transcribed passage with page and equation, or `file::function`, or `D-`),
verdict (OK / WRONG / IMPRECISE / UNSUPPORTED / REVNOTE OK / REVNOTE WRONG), and for each non-OK the concrete fix. Then
one table for FINAL-RUN-FIXES items 1, 2, 5 to 12: does IV state anything the item changes, and is it covered by a
correct revnote. Deliver in chat, most severe first.

## 10. Acceptance criterion
Every display and every cited, mathematical or reason-giving sentence of `04_obc.tex` has a row with a quoted source and
a verdict; every text-code difference is either covered by a correctly worded `\revnote` with a FINAL-RUN-FIXES item
and a decision entry, or reported; every derived formula called OK was checked by derivation or a scratchpad script,
with the result stated.

## 11. Read these first
1. `Thesis-writeup/Writing/sections/04_obc.tex` (all, including the header).
2. `literature/Orthogonality/gyorok2026_orthogonal-by-construction-augmentation_IFAC-JSC_arXiv2511.01321.pdf` (pp. 1 to 6, 9 to 10).
3. `literature/Orthogonality/Hoekstra - Orthogonal projection-based regularization for efficient model.pdf` (Gyorok et al. 2025, pp. 4 to 8).
4. `docs/decisions.md` entries D-186 (2026-09-10, trajectory basis), D-190, D-192, D-252, D-253, D-254.
5. `Thesis-writeup/Documentation/FINAL-RUN-FIXES.md`.

## 12. Do not
- Edit any file; report only.
- Say "Gyorok says" or "Kessels says" without the passage read in your session.
- Re-report the applied first-audit findings unless the current text still has them.

## 13. Operational
PDF pages: Read tool with `pages`. Kessels: `literature/augmentation/kessels2025_ai-control.pdf`, Sec. 5.3.1.3 is thesis
p. 166 = PDF p. 193. Code prefixes: `GD` = `scripts/gantry/gantry_dynamic/`, `FS` = `model_augmentation/fit_systems/`.
Numerical checks: a script in your scratchpad, run with `conda run -n GraduationProject python <file>`. Code quotes with
line numbers only after `scripts/verify_quote.py` prints MATCH OK.

## 14. Delegation
None. One context must hold the section, the papers and the code together.
