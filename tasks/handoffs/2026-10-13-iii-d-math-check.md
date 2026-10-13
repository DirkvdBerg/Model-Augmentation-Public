# Handoff: check the math of thesis subsection III-D "Identification" against its sources and the code
**From**: session of 2026-10-13 | **Branch**: Augmentation | **Effort suggested**: high (three displays, five cited passages, one code path; no derivation of new results)

## 1. Task
Review subsection III-D of `Thesis-writeup/Writing/sections/03_augmentation.tex` (`\subsection{Identification}\label{sec:aug_closed_loop}`,
l.773 to 886 at handoff; the header block `% MUST ESTABLISH, III-D` at l.170 to 189 states what it must establish). Check
every mathematical statement in it: the three displays `eq:mdelta_map`, `eq:aug_loss`, `eq:aug_init`, their where clauses,
and every sentence that states a mathematical or source fact. Check against (a) the cited passage in the PDF, (b) the
code that implements it, (c) the symbols and equations of the sections it builds on (II-C, III-A to III-C, Appendix
`app:lfr-wellposed`), and (d) general correctness (dimensions, bounds, strict versus non-strict inequalities, index
ranges, normalisation constants, implications that only hold under extra conditions). Report every finding in chat to
Dirk. Do not edit the file; Dirk decides which findings to apply.

## 2. Out of scope
- Edits to any `.tex` file, `refs.bib`, code, or docs, unless Dirk asks for a specific edit (then sec. 12 rule 3 applies).
- Prose style, concision and structure of III-D: approved by Dirk on 2026-10-13; flag only where wording states wrong math.
- III-C (`sec:aug_loop`) and III-E (`sec:aug_ps2`): under review in another session
  (`tasks/handoffs/2026-10-13-iii-c-iii-e-review.md`). Read them only for symbols III-D uses.
- Model naming (Aug-U, $\mathcal M$ symbols): decided 2026-10-13, III-D names no model; the label moves to Experiment design.
- The final-run code fixes (`Thesis-writeup/Documentation/FINAL-RUN-FIXES.md`); the text describes the final runs on purpose
  and marks 2026-10-11 differences as `\todo`. A mismatch between text and current code is a finding only if it is not
  covered by such a todo.
- Runs, compiling in `Writing/` (build tests only in a scratchpad copy, sec. 13).

## 3. Where things stand
- Last commit `b32b9ca`; `Thesis-writeup/` dirty (III-D rewritten this session, III-C edited by Dirk). Read the file as it
  is at session start; line numbers may have shifted.
- A scratchpad copy of `Writing/` built `main.tex` and `review.tex` with no `!` lines on 2026-10-13. Undefined references
  that remain are outside III-D (`eq:aug_wb`, `sec:obc_scope`, `eq:obc_overlap`, `sec`).
- No run in flight.

## 4. Established and verified (read this session, 2026-10-13; rows in `Thesis-writeup/Documentation/CITATION-LOG.md`)
- Hoekstra 2026 (`literature/augmentation/hoekstra2026_lfr-augmentation-fp-models.pdf`): Eq. (22) p. 8, joint vector
  col(theta_base, theta_aug, theta_LFR, theta_encoder), criterion 1/(N-T+1) * 1/T * sum ||y_hat - y||^2, encoder (22e) reads
  up to k-1; Sec. 5.2 pp. 8 to 9, minimisers not unique; Eq. (29) p. 9, baseline behaviour for x_b and y_hat, x_a evolves by
  phi_aug(theta_aug^0); Eq. (31) p. 9, psi_aug Xavier; Sec. 5.4.3 pp. 9 to 10, phi_aug = 0 + W_a z_a, matrices not needed
  for (29) drawn U(-1,1).
- Drenth et al. 2025 (`literature/lpv-lfr/drenth2025_lpv-lfr-rational.pdf`) p. 4: Sec. 4.2, D_zw = e^{-N}, N > 0 by
  construction, "enabling unconstrained optimization"; Sec. 5.1, Adam then L-BFGS-B.
- Kessels thesis (`literature/augmentation/kessels2025_ai-control.pdf`) Eq. (5.12), thesis p. 156 (PDF p. 183):
  minimisation over w0 to w3 only.
- Code: `model_augmentation/fit_systems/blocks.py::_m_diff_bound` = rho (2/Lb) sqrt(m_total J_eff), rho 0.99 (D-229);
  `combinations_from_free`: nine combos combo_init*exp(free), m_diff = bound(c(free)) * tanh(eta0 + s*free_i). Appendix
  `99_appendix.tex` l.~168 uses the same bound b(theta_base).
- Code: `model_augmentation/utils/torch_nets.py::live_g_feed_forward_nn` (l.170): final bias zero, physical rows of the final
  weight zero, added-state rows keep nn.Linear's draw. Thesis runner `scripts/gantry/gantry_interconnect_dynamic.py`
  `_THESIS_FIXED` (l.331): `wa_encoder_init='xavier_uniform_gain1'`, `n_nodes_per_layer=16`, `n_hidden_layers=2`,
  `na_nb_override=29`, `param_prior=False`, `lbfgs=True`. `pre_encoder.py` l.415 to 441: W^a Xavier gain 1 on one
  [W_y W_u] draw; psi_tilde final weight and bias zero.
- Code: `lbfgs_polish.py::decide` (l.94) keeps the polish if validation improves and no meter regresses by > 0.1 %;
  `param_loss` returns 0.0 when `flag_loss_reg` (= `param_prior`) is off; `combo_err` from `training.py::_gantry_meters`
  (l.652) is still in the acceptance (FINAL-RUN-FIXES item 2 not implemented; text covers it with a todo).

## 5. Assumed but not verified (the checker's main targets)
1. `eq:aug_loss` index range: the sum runs k = tau to tau+n_f-1 with the encoder reading sample tau. Confirm in
   `model_augmentation/fit_systems/interconnect.py::SSE_Interconnect_Composed.loss` and deepSI's hist/future split
   (`na_right = 1`, `GD/model.py::get_encoder_dims`) which samples are scored, and that the mean is over windows, samples
   and the three channels of the normalised output (1/(|T| n_f n_y)).
2. `eq:mdelta_map` where clause: "eta_0 and s reproduce m_Delta,0 and its sensitivity at xi = 0". D-229 sets
   d m_diff / d free = m_diff_init (a relative-change convention, at fixed m_total, J_eff). Check that "its sensitivity"
   is a correct and sufficient statement, and that the appendix (l.~160 to 175) defines eta_0 and s the same way.
3. Admissibility claim "every xi satisfies eq:lfr_admissibility": holds because |tanh| < 1 and rho < 1 give a strict
   inequality, and the nine log coordinates keep m_h, m_Sigma, J_eff > 0, which the iff in II-C needs. Confirm both
   conditions are stated or implied in the text.
4. Drenth Sec. 4.2 sentence: "every value of its free parameters gives a well-posed model". Their Theorem 6 also needs
   Assumptions 2 and 3 (scheduling bounds). Decide whether the sentence overstates the source.
5. Kessels sentence: "trains only the network". Eq. (5.12) minimises over w0 (encoder) and w1 to w3 (augmentation
   networks). Decide whether "the network" is accurate or should read "only the network weights".
6. Sec. 5.4.3 sentence: "adds a random linear bypass to a zero network". Check on pp. 9 to 10 that W_a is among the
   randomly drawn matrices (it enters Eqs. (32), (33)), as D-248 reads it.
7. `eq:aug_init`: the added-state rows bound U(-1/sqrt(n_h), 1/sqrt(n_h)) equals PyTorch's nn.Linear default
   (kaiming_uniform a = sqrt(5)) for fan-in n_h; the Xavier bound a = sqrt(6/(n_xbar + 6(n+1))) needs fan-in 6(n+1),
   i.e. n+1 samples of 3 outputs and 3 inputs. Check both against the code and the encoder window of `eq:aug_encoder`.
8. "From the same physical initial state, x_tilde and y_hat reproduce the baseline": in the closed loop of III-C this needs
   f_aug = 0 exactly (zero rows and zero bias) and holds for the closed-loop baseline with the same controller. Confirm no
   other path (normalisation offsets, x_bar entering f_base) breaks it.
9. Selection rule: "during Adam we keep the parameters with the lowest validation error ... over whole validation
   records, each simulated from one encoder start with x^c = 0". Confirm in `closed_loop.py::ClosedLoopSimulator.validation_error`
   and the fit loop (`interconnect.py::SSE_Interconnect.fit`, `GD/training.py::train_model`).

## 6. Tried and failed
- Quoting source content from pages not read in the session -> retracted on 2026-10-13 -> claims drifted from the PDF ->
  every finding about a source names the PDF page read in this session.
- Proposed lines ending in `% CHANGED` or `% NEW` -> broke Dirk's build twice -> the comment swallows the rest of the line ->
  give pasteable lines without markers.
- A `\\` as the first line of a paragraph or a nested `\todo{\todo{...}}` -> latexmk stops, "??" in the PDF.
- Redirecting latexmk output to `main.out` in the build copy -> fatal errors -> it overwrote hyperref's `main.out`
  bookmark file -> write logs outside the copied folder.

## 7. Achieved
III-D drafted and building (sec. 3); MUST ESTABLISH block written (l.170 to 189); eight citation-log rows added.

## 8. The open question
Nothing blocked.

## 9. Next action
Read III-D and its header block, then the passages and code of secs. 4 and 5, and present the findings list to Dirk in
the format of sec. 12 rule 6.

## 10. Acceptance criterion
Every display, where clause and math-bearing sentence of III-D, and each of the nine items of sec. 5, has a verdict
(correct / finding) with its evidence (PDF page read this session, `file:line`, or a short derivation). Every finding has a
pasteable correction. New passages read are added as rows to `CITATION-LOG.md`.

## 11. Read these first
1. `Thesis-writeup/Writing/sections/03_augmentation.tex` l.170 to 189 (III-D header) and l.773 to 886 (III-D), plus the
   displays it uses: `eq:aug_transition`, `eq:aug_norm`, `eq:aug_norm_maps`, `eq:aug_encoder`, `eq:aug_cl_step`.
2. `Thesis-writeup/Writing/sections/02_system_baseline.tex` l.~330 to 366 (`eq:phi`, `eq:lfr_admissibility`) and
   `99_appendix.tex` `app:lfr-wellposed`.
3. Hoekstra 2026 pp. 7 to 10; Drenth 2025 p. 4; Kessels p. 156 (PDF p. 183).
4. `model_augmentation/fit_systems/blocks.py` l.1385 to 1420; `interconnect.py::SSE_Interconnect_Composed.loss`.
5. `docs/decisions.md` D-229 (l.7179), D-248 (l.7771).

## 12. Do not
Dirk's working rules:
1. Keep Dirk's sentences and style; a fix is at most a few words, or full pasteable lines reusing his words.
2. Answer exactly the question asked; a yes/no question gets yes or no.
3. No file edit unless he asks; ask "saved?" before every `.tex` edit, then tell him to run "File: Revert File".
4. No em dashes; no "algorithm", "evolve" or similar generated-sounding words in proposed text.
5. Cite Hoekstra Sec. 5.2 only for non-uniqueness, never for its regulariser (Eq. (26)).
6. Report format, one block per finding:
   ```
   F3 (eq:mdelta_map, where clause, l.795) Source: Appendix l.~168; code blocks.py l.1405.
   Issue: "its sensitivity" is ambiguous; the code fixes d m_Delta / d xi_Delta = m_Delta,0 at xi = 0.
   Fix: and $\eta_0$ and $s$ give $m_{\Delta,0}$ and $\partial m_\Delta/\partial\xi_\Delta=m_{\Delta,0}$ at $\xi=0$
   ```
   (The example illustrates the format; its content is item 2 of sec. 5, not a verified finding.)

## 13. Operational
- Python: `conda run -n GraduationProject python <script>`; multi-line snippets via a script file in the scratchpad.
- PDFs: Read tool with `pages` (one page or one range per call).
- Build test (only if a proposed fix changes LaTeX): copy `Writing/` without `build/` to the scratchpad, put `refs.bib` in
  its `build/`, run `latexmk -pdf -interaction=nonstopmode main.tex` and `review.tex` with logs written outside the copy.

## 14. Delegation
None. All reads are targeted.
