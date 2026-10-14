# Handoff: critical review of the rewritten thesis Section III-B (encoder, data-based initialisation)
**From**: session of 2026-10-14 | **Branch**: Augmentation | **Effort suggested**: xhigh (index-level checks against three papers, the code and the writing rules; Dirk: "it is of critical importance we get this right")
**Supersedes**: `tasks/handoffs/2026-10-14-section-3b-math-audit-v2.md` (it audited the model-based version; III-B has since been rewritten around the data-based initialisation, D-251).

## 1. Task
Review Section III-B, "Encoder and initial state" (`Thesis-writeup/Writing/sections/03_augmentation.tex`, line 445 to
line 607), as rewritten on 2026-10-14, plus the new appendix subsection `app:enc_model` (`sections/99_appendix.tex`,
from line 206) and the two places that now use it (`eq:aug_init` line 888, the III-E sentence near line 930). Answer
three questions for every equation, symbol definition and claim: (a) is it mathematically correct; (b) is every cited
passage stated correctly, i.e. same quantity, indices, assumptions and scope as in the paper; (c) does it match the code,
or is the difference covered by a `\revnote`/`\todo`. Then review the subsection as writing against the thesis-section
skill (`.claude/skills/thesis-section/SKILL.md`, its "Review loop" checklist) and the README's "Math standard" and
"Paper structure and ownership". **Every source cell carries the passage itself**: a verbatim quote of at most two
sentences or an equation transcribed in LaTeX, with paper, printed page and number, read in this session from the
rendered PDF page. A paraphrase or a recollection is not acceptable; an unfound passage is reported as "not found in
<paper, pages searched>". Deliverable: the report in chat. Reading and analysis only, no edits.

## 2. Out of scope
- Re-opening decided points (section 4): Jan's order and notation, D-249 (zero mean enforced), D-250 (window ends at
  tau-1), D-251 (data-based W^b), the forward reference to III-C in III-B (Dirk keeps this order, 2026-10-14). Report
  one only if it is mathematically wrong or misattributed.
- Rewriting sentences: Dirk writes the prose. Give fixes of at most five words, or "move"/"delete" with the reason.
- III-C, III-D and III-E beyond the encoder's uses; Sections IV to VIII.
- Code changes, reruns, compiling `main.tex`. Small read-only numerical checks in the scratchpad are welcome for any
  numerical claim (one at a time, foreground, under a minute).

## 3. Where things stand
Branch `Augmentation`, last commit `0246725`; uncommitted edits in `03_augmentation.tex`, `99_appendix.tex`,
`08_conclusion.tex` (one outline bullet), `docs/decisions.md` (D-249, D-250, D-251) and
`Thesis-writeup/Documentation/FINAL-RUN-FIXES.md` (items 7 to 10). Audit the files as they are on disk; state their
modification times at the top of the report. No run in flight.

## 4. Established and verified (build on these; check only that the text states them correctly)
The current III-B, paragraph by paragraph:
1. Only positions measured; encoder estimates the window-start state (Beintema Sec. 3.1); window ends at tau-1 as in
   Hoekstra 2026 Eq. (22e) -> `eq:aug_encoder` (line 479), chi_tau newest first over tau-1 .. tau-n-1, 6(n+1) entries.
2. Encoder paper Eq. (8), linear map plus psi_tilde; window and state centred separately, so the physical rows carry
   a bias b^b -> `eq:aug_enc_lin` (line 506).
3. Physical rows fitted to baseline-simulated states following Hoekstra 2026 Eq. (30); open-loop simulation drifts on
   X and Y (III-A), so the baseline is simulated in closed loop with the controller of III-C ->
   `eq:aug_enc_sim` (line 541), notation x^{n,0}, y^{n,0}, u^{n,0}.
4. Least squares as encoder paper Eq. (33), plus a bias because the signals are centred -> `eq:aug_enc_ls` (line 564);
   W^b, b^b then trained jointly with theta_base (encoder paper footnote 1 vs Hoekstra 2026 Eq. 22).
5. Not model-based (encoder paper Sec. 3.1): the baseline is LPV in Y; evidence in `app:enc_model`.
Decisions: D-249 (`docs/decisions.md`, zero mean for generality; Hoekstra 2026 Sec. 5.3 aims at zero mean, Eq. 28
assumes it), D-250 (window tau-1; Beintema Eqs. 12, 13b include y_t, so the precedent is Hoekstra only), D-251
(data-based; evidence below; open: the initial state of `eq:aug_enc_sim` per record, a `\todo` in the text).
Evidence for D-251 (`scratchpad/recon_val_check.py`, session 3be8b0ae; baseline at the TRUE parameters, noisy
validation records, D-250 window): RMS error in units of state std, dTheta 0.34 model-based vs 0.11 data-based;
model-based 0.16 at |Y| < 0.05 m rising to 0.56 at 0.2 to 0.3 m, data-based 0.10 to 0.13; Y and dY equal for both
(0.0001, 0.073); data-based row norms about 1e3 larger (noise gain). The fit used windows of (y_hat, u_cl) from the
closed-loop simulation, with a bias column.
Revnotes in III-B (current runs differ; not findings): window up to tau (line 487), W^a uncentred (517), W^b at the true
parameters (580, with Dirk's `\todo`, kept on purpose), model-based map (598).
Code path that the FINAL-RUN-FIXES items 8 to 10 will change (current code = model-based): `GD/model.py::
get_encoder_dims`, `build_model`; `FS/pre_encoder.py::linear_encoder_init_aug`; `FS/closed_loop.py::closed_loop_rollout`
(the residual loop used for `eq:aug_enc_sim`); `GD/data.py::compute_normalization`.
Earlier audits (2026-10-14) confirmed: III-A mean handling correct; Coriolis neglected in `eq:eom`; the D-250 indices,
E_1 and rbar_n correct (now in the appendix); encoder paper Sec. 4.4 holds only for its static-parallel MSD example.

## 5. Assumed but not verified
- `eq:aug_enc_sim` with "u^{n,0}_k the closed-loop input of `eq:aug_cl_step` with y^{n,0}_k as the model output and the
  controller state zero at the start of the record" is a correct use of `eq:aug_cl_step` (which is written per window
  with x^c_tau = 0 and normalised signals).
- "Following Hoekstra et al., they are fitted to states simulated with the baseline [2026 Eq. (30)]": Eq. (30) fits an
  ANN encoder psi_base on (y_hat, u) histories up to k-1 with the recorded input; ours fits only the linear rows, on
  closed-loop inputs. Is "following" accurate, and is the adaptation visible enough?
- "Least-squares fit ... [enc. Eq. (33)], with a bias because the signals are centred": Eq. (33) fits W^b on simulated
  outputs and inputs; check what its Phi contains (indices, which signals) against `eq:aug_enc_ls` and its where-clause.
- "The baseline is LPV in Y, so a linearisation at one position misrepresents the Theta mode at others": is this the
  right mechanism for the measured growth of the dTheta error with |Y|, and is "the data-based fit covers the range of
  Y in the training records" supported (a single linear map fitted across Y)?
- Whether the where-clause of `eq:aug_enc_lin` ("window and state are centred separately, so the map between them is
  affine") is correct for a linear model whose data means are not an equilibrium.
- Whether the appendix comparison paragraph states the evidence without overclaiming (baseline at the true parameters;
  numbers from a scratch script, flagged by a `\todo`).

## 6. Tried and failed
- Citing papers from memory -> wrong section numbers twice -> quote from the rendered page in the same session.
- Attributing the past-only window to Beintema -> his Eqs. (12), (13b) include y_t -> check every "as in" against the
  exact equation.
- Symbols in the wrong coordinate frame survived review (x_tau instead of T_x x_tau) -> check physical / scaled /
  scaled-and-centred for every symbol, including the new x^{n,0}.
- An audit ran on a file that changed during the audit -> record modification times first.

## 7. Achieved
III-B rewritten 2026-10-14 (five paragraphs, four displays, four `\revnote`s), step-4 review findings applied (bias
reason in the lead-in, noise-gain sentence deleted, joint training its own paragraph, W^a remark in the where-clause,
D defined as the samples tau with a complete window, outline items before their paragraphs, x^{n,0} notation).
Model-based derivation and the comparison in `app:enc_model`. Display widths measured (sim 188 pt, ls 180 pt, column
252 pt). Not compiled in the editor yet.

## 8. The open question
Nothing blocked for the review. Open in the text, not to be decided here: the initial state of `eq:aug_enc_sim`
(`\todo`, candidate measured positions with zero velocity).

## 9. Next action
Read from the rendered PDF pages: encoder paper pp. 2 to 6 (Eqs. 1 to 37, footnotes 1, 2; Sec. 3.2 and Eqs. 33 to 35 in
full); Hoekstra 2026 pp. 8 to 10 (Secs. 5.1 to 5.4, Eqs. 22 to 33); Beintema 2023 printed pp. 2 to 5 and Sec. 4.2. Read
`03_augmentation.tex` lines 340 to 610 and 640 to 760 (the loop that `eq:aug_enc_sim` reuses), lines 870 to 935, and
`99_appendix.tex` from line 206. Then write the report:
1. File modification times and the lines reviewed.
2. One row per item in the format below.
3. Findings by severity: line, what is wrong, the correct form with its quoted source, fix kind (text fix of at most
   five words, move/delete, `\revnote`, or decision for Dirk).
4. Section 5 items, each resolved true or false with evidence.
5. The skill's review-loop checklist, each item pass or fail with the line.

Example rows:
| # | Thesis item (line) | Thesis states | Source, quoted or transcribed (paper, page, eq.) | Correct? | As in source? | Code? | Finding |
|-|-|-|-|-|-|-|-|
| 1 | `eq:aug_enc_ls` (564) | LS over windows chi^0_tau with bias b | Enc. p. 5, Eq. (33): `\min_{W^b_{\psi,y},W^b_{\psi,u}} \|\Phi [W^{b\top}_{\psi,y}\; W^{b\top}_{\psi,u}]^\top - \hat X\|_2^2` | yes | adapted: bias added, closed-loop inputs | no, item 10 pending (revnote 598) | covered |

## 10. Acceptance criterion
Every equation, symbol definition and claim in lines 445 to 607, in `app:enc_model`, and in the encoder lines of
`eq:aug_init` and III-E has a row whose source cell is a verbatim quote or transcription with page and number read in
this session. Every section 5 item is resolved with evidence. Every review-loop item of the skill is marked pass or
fail with a line. Every finding has line, error, correct form with quoted source, and fix kind.

## 11. Read these first
1. `literature/augmentation/Encoder initialisation methods in the model augmentation setting.pdf` (pp. 2 to 6).
2. `literature/closed-loop-id/hoekstra2026_lfr-augmentation-fp-models.pdf` (pp. 8 to 10).
3. `Thesis-writeup/Writing/sections/03_augmentation.tex` lines 340 to 610.
4. `.claude/skills/thesis-section/SKILL.md` (Rules, Order, Equations, Content, Review loop).
5. `docs/decisions.md` D-249, D-250, D-251 (last three entries).

## 12. Do not
- Edit any `.tex`, code or doc file; fixes go in chat.
- State "the paper says" without the quote or transcription in the same row.
- Count a section 4 decision, the kept forward reference to III-C, or a `\revnote`/`\todo`-covered difference as a
  finding.
- Quote code with line numbers without the code-quote verification rule.

## 13. Operational
PDF pages: Read tool with `pages`. PDF text search: PyMuPDF in a scratchpad script run with
`conda run -n GraduationProject python <file>` from PowerShell (in this environment Bash lacks conda and several Unix
tools). Reusable scripts in session 3be8b0ae's scratchpad: `recon_val_check.py` (the D-251 evidence),
`wb_shift_check.py`, `mathcheck.py`. Training data: `gantry_interconnect_dynamic.CFG` with `THESIS_ARM=u`,
`THESIS_NOISE=noisy`, `THESIS_START=correct`, `THESIS_NA=0`, `THESIS_SEED=1`, `THESIS_DEVICE=cpu`.

## 14. Delegation
None. One reader holds the papers, the text, the code and the skill together.
