# Handoff: finish thesis Section IV-B and IV-C with Dirk (orthogonal-by-construction on the gantry)
**From**: session 2026-10-14 to 16 (Section IV) | **Branch**: Augmentation | **Effort suggested**: high (careful reading of two papers, short exact LaTeX, many small decisions with Dirk)

## 1. Task
Dirk's words: "start a new session for writing section 4 ... important the other session reads gyorok his papers ...
help with finishing the current section 4, we're mainly focused on 4-B and 4-C now."
Help Dirk finish IV-B (`\subsection{Extension to the gantry}`, `sec:obc_map`) and IV-C (`\subsection{Identification
and recovery}`, `sec:obc_ident`) of `Thesis-writeup/Writing/sections/04_obc.tex`. Interactive: Dirk writes the
sentences; you explain, check maths and sources, and propose. FIRST read the two Gyorok papers from their rendered
pages (section 11), then the whole of `04_obc.tex`, then answer Dirk.

## 2. Out of scope
- The section opening and IV-A: Dirk's finished text. Touch only on request.
- Other sections, code, reruns, compiling `main.tex`, git writes.
- Re-opening the decisions in section 4.

## 3. Where things stand
`04_obc.tex` (2026-10-16): opening (Dirk) + IV-A (Dirk, Gyorok's construction, `eq:obc_gyorok` with the projected
network output F~^ANN) + IV-B + IV-C. IV-B begins with Dirk's own paragraph ("Our baseline violates three
requirements of the projection in eq:obc_gyorok. ..."). Dirk's old `\begin{outline}` bullet blocks were moved to the
end of the file on purpose (do not move them back). Several `\revnote`s (supervisor notes) exist; `\revnote` inside
`wip` is hidden in the review build. `wip` = text hidden in the review build; Dirk removes the markers once text is his.
Not compiled; Dirk's editor rebuilds on save (check `Thesis-writeup/Writing/build/main.log` for `!` lines).

Pending, agreed but NOT yet in the file (Dirk may have done some; check):
1. "taken at linearisation point" -> "taken at a linearisation point".
2. After the where-clause of `eq:obc_sens`: one sentence that the offset column Gamma of Gyorok 2025 Eq. (18) is left
   out. Dirk asked for its justification; agreed content: to first order the estimate depends only on components in
   ran(Phi), which the correction already removes, so Gamma would only constrain the network. Use "first-order"
   with hyphen; point to `\eqref{eq:obc_recovery}`. Dirk must be able to defend it (see section 4, Gamma).
3. In Dirk's sentence "...with the velocities reconstructed from them." add ", rather than simulating the baseline as
   in~\cite[Sec.~3]{gyorok2025l4dc}".
4. After the where-clause of `eq:obc_tuples`: "As Gy\"or\"ok et al.\ assume of their data~\cite[Assumption~1]{gyorok2026obc},
   we assume that the tuples give the stacked Jacobian full column rank."
5. Move Dirk's two sentences ("We use a state-space model predicting the next state instead of the output." / "The
   projection is removed from the neural network correction of the physical state at every step in the closed-loop
   simulation.") to just before `eq:obc_corr`.
Then IV-C: identification paragraph, recovery (`eq:obc_eps` = Xi, `eq:obc_recovery`), measures (`eq:obc_overlap`) are
drafted (Claude's prose in `wip`) and need Dirk's review the same way IV-B got it.

## 4. Established and verified (decided with Dirk; build on it)
- Structure: IV-A Gyorok's construction adopted; IV-B our extension (three requirements of eq:obc_gyorok violated:
  (1) state update nonlinear in the parameters through M(Y)^-1 in eq:aug_ct -> Jacobian w.r.t. beta as regressor, as
  in Gyorok 2025 Sec. 4 Eq. (15); (2) regressor needs the state, only positions measured -> Jacobian evaluated at
  measured positions with reconstructed velocities (tuples, eq:obc_tuples); (3) state-space model predicts the next
  state, not the output -> the fitted part is removed from the network correction of the physical state at every
  step of the closed-loop simulation, eq:obc_corr; added states not corrected). IV-C: identification (cost unchanged;
  theta_aux re-solved every evaluation; beta_lin per epoch; rebuild after restore; polish revnote), recovery, measures.
- We use orthogonality BY CONSTRUCTION, not regularisation: the 2025 penalty is never discussed; the 2025 paper is
  cited only where a concept is used (Jacobian Eq. (15), simulated states Sec. 3, Gamma Eq. (18), per-epoch Remark 2).
- Orthogonality = the stacked projected network output on the reference tuples is orthogonal to the stacked Jacobian
  (eq:obc_coef last line); holds for the stack at xbar = 0, not per tuple. Interpretability (Section I) = unique
  estimate; unique only to first order (one-step surrogate), not shown for the nonlinear closed-loop optimum.
- Gamma left out (D-254 item 1 as amended): Gamma protects the baseline's OUTPUT (Gyorok 2025's goal: "the subspace
  where f_theta can generate outputs"); our goal is the PARAMETERS. To first order the beta estimate
  beta* + (Phi^T Phi)^{-1} Phi^T Xi depends only on components in ran(Phi); both with and without Gamma the correction
  removes the network's component there, so Gamma does not change it; it only removes one extra direction
  (I-Pi)F_base outside ran(Phi), which blocks a uniform cancellation of the baseline response (part of Kessels'
  negation). Not true for the nonlinear fit (audit counterexample). Negation is used ONLY in Kessels' sense, defined
  inline at first use; no "parameter negation" term (dropped on purpose).
- Full column rank: theta_aux needs (Phi^T Phi)^{-1}, which exists iff Phi has full column rank (linear algebra;
  Gyorok assumes it, Assumption 1). For a linear-in-parameters baseline, persistence of excitation = full column rank
  of the regressor (Gyorok 2026 Sec. 4.3 after Condition 14). We use beta because full rank needs identifiable
  parameters (only the ten combinations of Section II-C; beta defined in III-C, eq:mdelta_map). Rank also depends on
  the data and beta_lin; not verified for the real data in the thesis -> stated as an assumption like Gyorok's.
- Notation (Dirk): Gyorok's case convention, no bold, no "+": per tuple varphi(beta_lin, x, u) (6x10), stack Phi
  (6N x 10), column Phi_j; network f_aug -> stack F_aug, projected F~_aug; baseline stack F_base; next states X_+;
  discrepancy Xi (not Delta: clashes with Delta(Y), m_Delta, beta_Delta; not eps: output error in III); (Phi^T Phi)^{-1}
  Phi^T throughout. Measures (supervisor: "orthogonality is measured by the inner product"): cos angle(a,b),
  c_j(d) = cos angle(Phi_j, d), rho(d) = max over ran Phi\{0} = ||Pi d||/||d||, nonzero vectors.
- Code agrees with all displays (two audits 2026-10-14, one 2026-10-15; FINAL-RUN-FIXES items 11, 12; D-252 to D-255).

## 5. Assumed but not verified
- That Dirk has or has not yet applied items 1 to 5 of section 3: read the file before proposing.
- `wip` balance around `eq:obc_sens` (a `\begin{wip}` may be missing after `\end{equation}`): check with grep.

## 6. Tried and failed (how to work with Dirk)
- Rewording Dirk's sentences, "improving" them, merging or restructuring: rejected repeatedly and angrily. Copy his
  sentence verbatim, change only errors (grammar, LaTeX, wrong fact) or what he asks; give the fixed sentence in a
  code block. See memory `feedback_keep_dirk_wording.md`.
- Dropping his stated requirements or context from a proposal (e.g. returning only part of his paragraph): always
  show his full text with only the marked changes when proposing edits to a paragraph.
- Citing a passage for a claim it does not make (e.g. Gyorok Assumption 1 after a claim about the Jacobian or
  identifiability; "Remark 5" for the whole of Xi). Cite only what the read passage says.
- Long lists of facts after a display read as an information dump; Dirk wants each sentence to follow from the
  previous one, math first, concise, and every choice justified (paper, decision or gantry property).
- Vague phrases Dirk rejected: "move the baseline", "different models", "behaviour", "transition" (use "state
  update" / "next state predicted by the baseline"), "corrected network output" (use "projected").
- Editing while his buffer was unsaved: ask "is 04_obc.tex saved?" before every edit; afterwards he runs
  "File: Revert File".

## 7. Achieved
Opening and IV-A done by Dirk. IV-B partly rewritten by Dirk; IV-C drafted.

## 8. The open question
Nothing blocked. The Gamma question has a recorded decision (omit); a mail to Jan asks for confirmation, not blocking.

## 9. Next action
Read the papers (section 11) and `04_obc.tex` in full; then tell Dirk in a few lines which of the pending items 1 to 5
are still open in the file, and ask which he wants to do first.

## 10. Acceptance criterion
IV-B and IV-C accepted by Dirk sentence by sentence; every Gyorok attribution matches a passage read in the session;
no unbalanced environments; no em dashes.

## 11. Read these first (mandatory, rendered pages via the Read tool with `pages`)
1. `literature/Orthogonality/gyorok2026_orthogonal-by-construction-augmentation_IFAC-JSC_arXiv2511.01321.pdf`,
   pages 1 to 6 and 9 to 10 (Eqs. (1) to (25), Assumption 1, Lemma 3, Condition 4, Remark 5, Assumption 6, Thm. 7,
   Sec. 4.3 Condition 14 and the PE paragraph, Sec. 6).
2. `literature/Orthogonality/Hoekstra - Orthogonal projection-based regularization for efficient model.pdf` (Gyorok
   et al. 2025, `gyorok2025l4dc`), pages 4 to 8 (Sec. 3 Eqs. (8) to (14), paragraph before Remark 2, Remark 2, Sec. 4
   Eqs. (15) to (19), paragraph after Eq. (19)).
3. `Thesis-writeup/Writing/sections/04_obc.tex` (all, including the header MUST ESTABLISH and NOTATION notes).
4. `docs/decisions.md` D-252 to D-255 (context only).

## 12. Do not
- Reword Dirk's sentences; propose only fixes he can accept or reject.
- Edit without a "saved?" confirmation; touch the outline blocks at the end of the file.
- Claim more than first-order, stacked, one-step results.

## 13. Operational
Edit with the Edit tool (LF endings; OneDrive may lock briefly, retry). Width-check changed displays in a scratch
IEEEtran file (column 252 pt) written with the Write tool (shell heredocs mangle backslashes here).

## 14. Delegation
None.
