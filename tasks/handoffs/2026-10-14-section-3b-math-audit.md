# Handoff: critical math audit of thesis Section III-B (encoder) against the cited papers and the code
**From**: session of 2026-10-14 | **Branch**: Augmentation | **Effort suggested**: xhigh (equation-level reading of three papers plus code tracing; errors here were found only by reading the PDFs line by line)

## 1. Task
Dirk's words: "read the relevant papers and critically analyze whether the current math is correct, and stated as in
Jan or other work we reference here, besides justified deviations or the revnotes we state."
Audit every equation, symbol definition and mathematical claim of Section III-B, "Encoder and initial state"
(`Thesis-writeup/Writing/sections/03_augmentation.tex`, from `\subsection{Encoder and initial state}`, line 444, to
`\subsection{Closed-loop simulation}`, line 617), plus the III-A normalisation it depends on (`eq:aug_norm`, line 358;
`eq:aug_norm_maps`, line 376). For each item decide: (a) is it mathematically correct; (b) does it match the cited
passage of the cited paper (same quantity, same indices, same ordering, same assumptions); (c) does it match what the
thesis code computes. Any difference that is neither listed in section 4 below as a justified deviation nor covered
by a `\revnote` or `\todo` in the text is a finding. The deliverable is a findings report in chat. This is reading and
analysis, not editing: no `.tex`, code or doc changes.

## 2. Out of scope
- Prose quality, wording, structure, page budget: Dirk writes every sentence (memory `feedback_keep_dirk_wording.md`).
  Report a wording problem only when it makes a mathematical statement false.
- III-C, III-D, III-E, Sections IV to VIII, except where a III-B symbol is used there (`eq:aug_loss` line 840 uses the
  encoder and the window set; `eq:aug_init` line 898 initialises W^a and psi_tilde).
- Code changes, reruns, compiling `main.tex`. Small read-only numerical checks in the scratchpad are allowed and
  welcome when a claim is numerical (one script at a time, foreground, under a minute).
- Re-opening the decisions in section 4 (Dirk decided them on 2026-10-14). Report only if one is mathematically wrong.
- The open D5 question (why model-based rather than data-based W^b, the `\todo{Revise: ...}` after the W^b paragraph):
  Dirk has not decided it; report the facts relevant to it, do not decide it.

## 3. Where things stand
Branch `Augmentation`, last commit `0246725` (Remove outline bullets). The tree is dirty: this session's edits to
`03_augmentation.tex` (III-B maths), `docs/decisions.md` (D-249) and `Thesis-writeup/Documentation/FINAL-RUN-FIXES.md`
(item 7) are uncommitted. Dirk edits `03_augmentation.tex` in VS Code; the file on disk is the version to audit. No run
in flight.

## 4. Established and verified (2026-10-12 to 10-14, PDFs read, code traced, numbers measured)
Decisions (do not re-open; check only that they are stated correctly):
- **Jan's order.** Window and reconstructability map follow encoder paper Eqs. (10) and (13): newest sample first,
  O_n = col(C A^n, ..., C A, C), the Toeplitz block upper triangular, r_n = [0 B AB ... A^{n-1}B]. Notation: Jan's
  calligraphic T_n for the Toeplitz matrix; the window set of `eq:aug_loss` renamed to calligraphic W to avoid the clash.
- **D-249: zero mean enforced.** The normalisation subtracts the training means (III-A). Reason: generality (data with
  nonzero means, e.g. force offsets on a real machine), Hoekstra 2026 Sec. 5.3 aims at zero mean and its Eq. (28)
  assumes it. The encoder therefore carries the fixed bias b^b = W^b chi_0 - T_x mu_x (a function of W^b, not a
  parameter): Jan's map works in deviations from the equilibrium (encoder paper, text after Eq. 32; footnote 2 on bias
  terms), the model in deviations from the data mean. W^a acts on the centred window, no bias.
- **Linearisation** at rest at Y = 0 (an equilibrium for zero input because K = diag(0, 3975, 0) in the code's units),
  at the start parameters theta^0_base (final runs), ZOH over T_s.
Justified deviations from Jan, already stated or to be stated in the text (not findings unless stated wrongly):
- Window includes sample tau (encoder paper Eqs. 11 to 17), whereas Hoekstra 2026 Eq. (22e) and encoder paper Eqs. (4c),
  (8) read up to tau-1. Note: the loss of `eq:aug_loss` then starts at k = tau, a sample the encoder saw.
- Marginally stable baseline (two eigenvalues of A_d at exactly 1) vs the stable baseline the encoder paper assumes
  (Sec. 2); justified by observability (rank O_n = 6, cond 1.3e3, n = 29 >= n_x).
- Co-estimation of theta_base and W^b (encoder paper footnote 1 puts it out of scope; Hoekstra 2026 Eq. 22 does it).
- W^a Xavier gain 1 on the linear rows of the Eq. (8) encoder, psi_tilde rows zero: adapted from Hoekstra 2026 Eq. (31),
  which Xavier-initialises a separate nonlinear psi_aug (weights and biases).
- Coloured measurement noise: the noiseless map (Eqs. 15 to 17) is used as a starting value, not as the ML estimator.
Revnotes and todos in III-B (code differs from text; not findings):
- `\revnote` line 520: current runs feed W^a the uncentred window (FINAL-RUN-FIXES item 7).
- `\revnote` line 600 and the preceding `\todo`: current runs build W^b from the true parameters (item 1).
Numerical facts (scratchpad scripts `mathcheck.py`, `bb_offset.py`, session 3be8b0ae):
- Code W^b (`FS/pre_encoder.py::linear_encoder_init_aug`, oldest first) equals Jan-order Eqs. (16), (17) after reversing
  the sample blocks: max difference 4e-14; the map reconstructs x_tau of a noiseless linear trajectory to 3e-17.
- ZOH linearisation vs Jacobian of the one-step RK4 baseline (`up_sample=1`): 8e-12 (A), 3e-11 relative (B).
- On the 18 noisy training records: mu/sigma <= 0.053 (x, y), <= 5.5e-4 (u); |b^b| <= 3.5e-3 std; W^a chi_0 about 3 %
  of W^a chi^n.
Code path (thesis run, `THESIS_*` block of `scripts/gantry/gantry_interconnect_dynamic.py`, n = `na_nb_override=29`):
`GD/model.py::get_encoder_dims` and `build_model` (encoder injection); `SYS/gantry_linearization.py::
gantry_linearize_and_discretize`; `model_augmentation/utils/utils.py::normalize_linear_ss_matrices`;
`GD/data.py::compute_normalization`; `FS/pre_encoder.py::linear_encoder_init_aug` (`__init__`, `forward` with the
D-055 offsets); `model_augmentation/utils/deepSI_corrections.py::to_hist_future_data` (window and loss alignment).

## 5. Assumed but not verified
- The III-A normalisation displays (`eq:aug_norm_maps`) are mathematically consistent with the centred encoder and with
  `eq:aug_lin`'s "scaling as eq:aug_norm without the means". Jan marked two III-A passages with "?" (the inversion of
  (15), and P^{-T} mu_y); the method is kept (D-249) but the math there was not re-derived.
- The closed-loop rollout (`FS/closed_loop.py`) uses the same window alignment as `to_hist_future_data` (encoder window
  up to tau, loss from tau). Only the window builder was read.
- The claim "Frozen at Y = 0, eq:eom is linear, so the linearisation is the baseline itself at that position" assumes
  eq:eom has no velocity-product (Coriolis) terms; Jan's feedback on Section II mentions a Coriolis term. Check eq:eom.
- Whether the maximum-likelihood statement carries over to scaled coordinates (per-axis noise variances differ, so the
  pseudoinverse is not the weighted least-squares estimate); the text already calls the map a starting value.
- Beintema citation `[Secs. 2, 3]` for "cost independent of record length and stabilises the optimisation": the
  session read the claim in Sec. 3.1 (and 4.2 per the paper's own pointer, not read).

## 6. Tried and failed
- Citing Jan from memory -> wrong section numbers twice (2025 EJC vs 2026; Sec. 2 vs 3) -> read the PDF passage in the
  session before stating "Jan says".
- Treating the encoder paper as internally consistent -> it is not: Sec. 2 defines u^{k-1}_{k-n_b} oldest first, Eqs. (10),
  (13) stack newest first; Eq. (9b) reads y_k = C x_{b,k+1} + D u_k, a typo for x_{b,k} -> state which equation a thesis
  line follows.
- Writing the map's left-hand side as x_tau -> wrong: the map works in scaled uncentred coordinates, so T_x x_tau
  (fixed 2026-10-14). Check every symbol for its coordinate frame (physical, scaled, scaled and centred).
- Arguing "simulation drifts" without the loop -> the model runs in closed loop; state open or closed loop for every
  stability or drift claim.

## 7. Achieved
III-B maths as of 2026-10-14: `eq:aug_encoder` (newest first, includes tau), `eq:aug_enc_lin` (with b^b), `eq:aug_lin`,
`eq:aug_wb` (restored, Jan's calligraphic T_n), where-clauses for O_n, T_n, r_n; window set calligraphic W in
`eq:aug_loss`. Not yet compiled after these edits.

## 8. The open question
Nothing blocked.

## 9. Next action
Read, in full and from the PDF pages (equations as images, not text extraction): the encoder paper Secs. 2 to 4
(pp. 2 to 6, all equations 1 to 37 and footnotes 1, 2); Hoekstra 2026 Secs. 5.1 to 5.4 (pp. 8 to 10, Eqs. 22 to 33);
Beintema 2023 Secs. 2 and 3 (pp. 2 to 5). Then read III-B and III-A's normalisation on disk, and the code path of
section 4. Then write the report: one row per equation, symbol definition and mathematical claim of III-B, in the
format of the example below, followed by a short list of findings ordered by severity.

Example rows:
| # | Thesis item (line) | Claim | Source checked | Correct? | As in source? | Code? | Finding |
|-|-|-|-|-|-|-|-|
| 4 | `eq:aug_wb` (565) | T_x x_tau = A^n O^+ (Y - T_n U) + r_n U | Enc. Eq. 15 | yes | yes, O^{-1} read as left inverse (after Eq. 12) | yes, `pre_encoder.py` W^b, oldest first | none |
| 9 | where-clause b^b (513) | b^b = W^b chi_0 - T_x mu_x | Enc. after Eq. 32, fn. 2 | ? | own derivation (D-249) | yes, D-055 offsets | ... |

## 10. Acceptance criterion
Every equation, symbol definition and mathematical claim between lines 444 and 617, plus `eq:aug_norm` and
`eq:aug_norm_maps`, has a row whose source column names a passage read in that session. Every item in section 5 is
resolved as true or false with its evidence. Every finding states the line, what is wrong, the correct form with its
source, and whether it needs a text fix, a `\revnote`, or a decision from Dirk.

## 11. Read these first
1. `literature/augmentation/Encoder initialisation methods in the model augmentation setting.pdf` (pp. 2 to 6): the
   source of W^b and of Eq. (8).
2. `literature/closed-loop-id/hoekstra2026_lfr-augmentation-fp-models.pdf` (pp. 8 to 10): criterion, normalisation,
   initialisation, Xavier psi_aug.
3. `Thesis-writeup/Writing/sections/03_augmentation.tex` lines 340 to 617.
4. `literature/experiment-design/Papers/Beintema_Schoukens_Toth_2023_Deep_subspace_encoders.pdf` (pp. 2 to 5).
5. `docs/decisions.md` D-249 (last entry) and `Thesis-writeup/Documentation/FINAL-RUN-FIXES.md` items 1 and 7.

## 12. Do not
- Edit any `.tex`, code, or doc file; propose fixes in chat only (Dirk applies them, or asks for them after saving).
- Quote code with line numbers without the code-quote verification rule.
- Count a deviation in section 4 or a `\revnote`/`\todo` as a finding.

## 13. Operational
PDF pages with equations: Read tool with `pages`. Text search: PyMuPDF in a scratchpad script,
`conda run -n GraduationProject python <scratch file>` (PowerShell; Bash has no conda on PATH). Numerical checks may reuse
the thesis code imports as in this session's `mathcheck.py` (loads `gantry_linearize_and_discretize` and
`linear_encoder_init_aug`); training data load via `gantry_interconnect_dynamic.CFG` with `THESIS_ARM=u`,
`THESIS_NOISE=noisy`, `THESIS_START=correct`, `THESIS_NA=2`, `THESIS_SEED=1`, `THESIS_DEVICE=cpu` (about 30 s).

## 14. Delegation
None. The audit needs one reader holding the papers, the text and the code together.
