# Handoff: second critical math audit of thesis Section III-B (encoder), every claim checked against a quoted paper passage
**From**: session of 2026-10-14 | **Branch**: Augmentation | **Effort suggested**: xhigh (equation-level reading of three papers, index-by-index checks; Dirk: "it is of critical importance we get this right")
**Supersedes**: `tasks/handoffs/2026-10-14-section-3b-math-audit.md` (its audit ran on the text before D-250; its findings are listed in section 4).

## 1. Task
Dirk's words: "analyze the current version based on the papers, make sure it actually reads and quotes the papers,
not just from memory. It is of critical importance we get this right."
Audit every equation, symbol definition and mathematical claim of Section III-B, "Encoder and initial state"
(`Thesis-writeup/Writing/sections/03_augmentation.tex`, line 444 to line 620), plus the III-A normalisation it builds
on (`eq:aug_norm` line 358, `eq:aug_norm_maps` line 376) and the two later uses of the encoder (`eq:aug_loss` line 843,
`eq:aug_init` line 901). For each item decide: (a) is it mathematically correct; (b) does it match the cited passage
(same quantity, indices, ordering, assumptions); (c) does it match the code of the thesis run, or is the difference
covered by a `\revnote` or `\todo`. **Every row of the report carries the source passage itself**: a verbatim quote of at
most two sentences, or a transcription of the equation in LaTeX, with paper, page and equation or section number, read
in this session from the rendered PDF page. A row whose source column is a paraphrase or a recollection is not
acceptable; if the passage cannot be found, the row says "not found in <paper, pages searched>". The deliverable is the
findings report in chat. Reading and analysis only: no `.tex`, code or doc edits.

## 2. Out of scope
- Prose quality and wording (Dirk writes every sentence, memory `feedback_keep_dirk_wording.md`). Report wording only
  where it makes a mathematical statement false or contradicts a display.
- III-C, III-E, Sections IV to VIII, except the encoder's use in `eq:aug_loss` and `eq:aug_init`.
- Re-opening the decisions in section 4; report only if one is mathematically wrong or misattributed.
- Deciding row 10 (model-based vs data-based W^b, section 8); report facts relevant to it.
- Code changes, reruns, compiling `main.tex`. Small read-only numerical checks in the scratchpad are welcome for any
  numerical claim (one at a time, foreground, under a minute).

## 3. Where things stand
Branch `Augmentation`, last commit `0246725`. Uncommitted: this session's edits to `03_augmentation.tex` (III-B maths),
`docs/decisions.md` (D-249, D-250), `Thesis-writeup/Documentation/FINAL-RUN-FIXES.md` (items 7, 8). Dirk edits the
`.tex` in VS Code; audit the file as it is on disk. No run in flight.

## 4. Established and verified (do not re-derive; check only that the text states them correctly)
Decisions (Dirk, 2026-10-14):
- **Jan's order and notation.** Window and map follow the encoder paper's Eqs. (10), (13): newest sample first,
  O_n = col(C A^n, ..., C), calligraphic T_n block upper triangular, r_n = [0 B AB ... A^{n-1}B]. Window set of
  `eq:aug_loss` renamed calligraphic W (calligraphic I is taken by Section IV).
- **D-249, zero mean enforced** (generality: nonzero means on real machines). Encoder = enc. paper Eq. (8) on the centred
  window chi_tau plus the fixed bias b^b = W^b chi_0 - T_x mu_x (a function of W^b, not a parameter); W^a on chi_tau, no
  bias. Basis: enc. paper text after Eq. (32) (deviations from the equilibrium) and footnote 2 (bias terms).
- **D-250, window ends at tau-1** as in Hoekstra 2026 Eq. (22e). W^b = Eq. (15) at tau-1 propagated one step (Eq. 9a):
  W^b = [A^{n+1} O_n^+ , rbar_n - A^{n+1} O_n^+ T_n], rbar_n = A r_n + B E_1 = [B AB ... A^n B]. Loss sum from tau.
  Numerically exact on a noiseless LTI trajectory (6e-17, `scratchpad/wb_shift_check.py`).
- Linearisation at rest at Y = 0 at theta^0_base, ZOH over T_s (K = diag(0, 3975, 0) in the code).
Justified deviations (findings only if stated wrongly): marginally stable but observable baseline (two eigenvalues of A_d
at 1, rank O_n = 6, cond 1.3e3, n = 29); co-estimation of theta_base (enc. footnote 1 vs Hoekstra 2026 Eq. 22); W^a
Xavier gain 1 on the linear rows, adapted from Hoekstra 2026 Eq. (31) (which Xavier-initialises a separate psi_aug);
coloured measurement noise, so the noiseless map is a starting value.
Revnotes in III-B, current runs differ from the text (not findings): window up to tau (line 490); W^a on the uncentred
window (line 519); W^b at the true parameters (line 603, with a `\todo`).
First audit (2026-10-14, on the pre-D-250 text), findings and status:
- F1: "with white Gaussian measurement noise it is the maximum-likelihood estimate" needs equal noise variance per
  channel in the scaled coordinates (0.19 % between weighted and unweighted map). Proposed: delete the sentence. Open.
- F2: encoder matrices use sigma, rollout uses sigma + 1e-8 (2e-5 relative). Proposed: code fix only. Open.
- F3: lead-in "the linear part acts on the scaled but uncentred window" contradicts `eq:aug_enc_lin` (only the physical
  branch). Proposed fix: "The physical linear branch". Open.
- Confirmed correct: III-A mean handling (inverse maps, mu_q = P^{-T} mu_y cancels), Coriolis terms neglected in
  `eq:eom` (frozen-Y linearity holds), b^b formula (matches code to 1.4e-15), Beintema Sec. 3.1 for the cost claim.
Code path of the thesis run: `GD/model.py::get_encoder_dims`, `build_model`; `SYS/gantry_linearization.py::
gantry_linearize_and_discretize`; `model_augmentation/utils/utils.py::normalize_linear_ss_matrices`;
`GD/data.py::compute_normalization`; `FS/pre_encoder.py::linear_encoder_init_aug`;
`model_augmentation/utils/deepSI_corrections.py::to_hist_future_data`; `FS/closed_loop.py::closed_loop_rollout`.

## 5. Assumed but not verified
- The D-250 text is new: the lead-in "one step of the linear model [Eq. (9a)] gives the state at tau", the definitions of
  E_1 and rbar_n, and the claim that the encoder paper's encoder (Eqs. 4c, 8) reads up to k-1 while its map (Eq. 15) uses
  up to k. Check each index against the PDF.
- Whether citing the encoder paper Eq. (9a) is right for the one-step propagation (Eq. 9a is the LTI state equation;
  Eq. 9b in the paper has a typo, x_{b,k+1} for x_{b,k}).
- Whether "Frozen at Y=0, eq:eom is linear, so the linearisation is the baseline itself at that position" is exactly
  right given Jan's Eq. (31) differentiates the discrete f_base (here: CT linearisation with ZOH vs the RK4 step; measured
  difference 8e-12, `scratchpad/mathcheck.py`).
- Whether the encoder paper's Sec. 4.4 statements are cited for exactly what they show (MSD example, static parallel
  augmentation, n_xa = 0).

## 6. Tried and failed
- Citing Jan from memory -> wrong section numbers twice (2025 EJC vs 2026; Sec. 2 vs 3) -> quote the passage from the
  rendered page in the same session.
- Treating the encoder paper as internally consistent -> Sec. 2 stacks u^{k-1}_{k-n_b} oldest first, Eqs. (10), (13)
  newest first; Eq. (9b) has x_{b,k+1} -> say which equation each thesis line follows.
- A symbol in the wrong coordinate frame survived review (x_tau instead of T_x x_tau) -> check every symbol for physical,
  scaled, or scaled-and-centred coordinates.
- The first audit ran on a text that changed the same day -> state the file's modification time and the lines audited
  at the top of the report.

## 7. Achieved
III-B maths as of 2026-10-14: `eq:aug_encoder` (window tau-1 to tau-n-1, newest first), `eq:aug_enc_lin` (with b^b),
`eq:aug_lin`, `eq:aug_wb` (three lines: map at tau-1, one step, W^b with rbar_n), where-clauses for O_n, T_n, r_n, E_1,
rbar_n. Display widths measured against the 252 pt column (chi_tau 204 pt). Not yet compiled in the editor.

## 8. The open question
Row 10, not for this session to decide: a closed-loop simulation of the baseline on the 18 training records (pre-D-250
window) gave model-based W^b reconstruction errors on dTheta of 4 % of its std at |Y| < 0.05 m rising to 43 % at
|Y| 0.2 to 0.3 m, against about 1e-5 for the data-based least-squares map (enc. Eq. 33) fitted on noise-free simulated
windows (`scratchpad/recon_check.py`). A check on noisy validation records against true states is pending. The audit
may note which III-B statements this affects ("negligible computation", "better starting values").

## 9. Next action
Read in full from the rendered PDF pages (Read tool with `pages`, equations as images): the encoder paper pp. 2 to 6
(Eqs. 1 to 37, footnotes 1 and 2); Hoekstra 2026 pp. 8 to 10 (Secs. 5.1 to 5.4, Eqs. 22 to 33); Beintema 2023 pp. 2 to 5
and Sec. 4.2. Then read `03_augmentation.tex` lines 340 to 620, 830 to 930, and the code path of section 4. Then write
the report: first the file's modification time; then one row per item in the format below; then the findings ordered
by severity, each with the line, what is wrong, the correct form with its quoted source, and the kind of fix (text
fix, `\revnote`, or decision for Dirk); then the section 5 items, each resolved as true or false with evidence.

Example rows:
| # | Thesis item (line) | Thesis states | Source, quoted or transcribed (paper, page, eq.) | Correct? | As in source? | Code? | Finding |
|-|-|-|-|-|-|-|-|
| 1 | `eq:aug_wb` line 1 | T_x x_{tau-1} = A^n O^+ (Y - T_n U) + r_n U | Enc. p. 3, Eq. (15): `x_{b,k} = A^n \mathcal{O}_n^{-1}(y^k_{k-n} - \mathcal{T}_n u^k_{k-n}) + r_n u^k_{k-n}` | yes | yes, index shifted to tau-1 | yes (item 8 pending) | none |
| 2 | encoder window (line ~470) | encoder reads up to tau-1 as in Hoekstra 2026 | L p. 8, Eq. (22e): `\hat x_{k|k} := \psi(\theta_{encoder}, y^{k-1}_{k-n_a}, u^{k-1}_{k-n_b})` | yes | yes | no, `na_right = 1` (revnote line 490) | covered |

## 10. Acceptance criterion
Every equation, symbol definition and mathematical claim in lines 444 to 620, plus `eq:aug_norm`, `eq:aug_norm_maps`
and the encoder terms of `eq:aug_loss` and `eq:aug_init`, has a row whose source column is a verbatim quote or equation
transcription with page and number, read in this session. Every section 5 item is resolved. Every finding names line,
error, correct form with quoted source, and fix kind. F1 to F3 are re-checked on the current text and reported as still
open, resolved, or changed.

## 11. Read these first
1. `literature/augmentation/Encoder initialisation methods in the model augmentation setting.pdf` (pp. 2 to 6).
2. `literature/closed-loop-id/hoekstra2026_lfr-augmentation-fp-models.pdf` (pp. 8 to 10).
3. `Thesis-writeup/Writing/sections/03_augmentation.tex` lines 340 to 620.
4. `literature/experiment-design/Papers/Beintema_Schoukens_Toth_2023_Deep_subspace_encoders.pdf` (pp. 2 to 5, Sec. 4.2).
5. `docs/decisions.md` D-249 and D-250 (last two entries).

## 12. Do not
- Edit any `.tex`, code or doc file; fixes are proposed in chat.
- State "the paper says" without the quote or transcription in the same row.
- Quote code with line numbers without the code-quote verification rule.
- Count a section 4 decision, a justified deviation, or a `\revnote`/`\todo`-covered difference as a finding.

## 13. Operational
PDF pages: Read tool with `pages`. Text search in PDFs: PyMuPDF in a scratchpad script, run with
`conda run -n GraduationProject python <file>` from PowerShell (Bash has no conda on PATH). Numerical checks can reuse
`scratchpad/mathcheck.py` and `wb_shift_check.py` of session 3be8b0ae (imports `gantry_linearize_and_discretize`,
`linear_encoder_init_aug`). Training data: `gantry_interconnect_dynamic.CFG` with `THESIS_ARM=u`, `THESIS_NOISE=noisy`,
`THESIS_START=correct`, `THESIS_NA=2`, `THESIS_SEED=1`, `THESIS_DEVICE=cpu` (about 30 s to load).

## 14. Delegation
None. One reader holds the papers, the text and the code together.
