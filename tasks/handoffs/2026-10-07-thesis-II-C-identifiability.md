# Handoff: write thesis Section II-C (physical parameters and identifiable combinations) with the identifiability proof in the main text
**From**: session of 2026-10-05 to 2026-10-07 | **Branch**: Augmentation | **Effort suggested**: high (a short proof must be exact; the rest is drafting)

## 1. Task
Invoke the `thesis-section` skill and write subsection II-C, `\subsection{Physical parameters and identifiable combinations}\label{sec:params}` in `Thesis-writeup/Writing/sections/02_system_baseline.tex` (starts at line 257, currently a bullet list). Start at the skill's step 2a: present the claim brief of section 4 below to Dirk as your proposal, adopt his corrections, then step 2b (paragraph outline), then draft. Dirk's own words for this subsection: "I would actually prefer to show the proof for this, but in a sophisticated way." So the structural-identifiability argument goes into the main text as a stated result with a compact proof, not into the appendix.

## 2. Out of scope
- II-B (`sec:lfr`) of the same file: finished in this session with Dirk, sentence by sentence. Do not edit it, except to delete the closed-form clause from its draft bullets and `% MUST ESTABLISH` header block if Dirk confirms he deleted that sentence (see section 8).
- Sections 03, 04 and the u-notation clash between II and III (`03_augmentation.tex` line 86 `\todo`): Dirk's decision, not this task. `04_obc.tex` has uncommitted edits by Dirk; do not touch it.
- Recovery metrics, relative-combination-error normalisation, the value rho = 0.99, and practical recoverability from the closed-loop data: Setup/Results, not II-C (existing header comment above the II-C bullets says so).
- The SVD six-to-four LFR reduction: not used, not to be added (Dirk's decision this session).
- Compiling the thesis: do not. Dirk saves; the editor rebuilds; then check `build/main.log` for lines starting with `!`.

## 3. Where things stand
Branch `Augmentation`, last commit `a656a55`. `02_system_baseline.tex` and `04_obc.tex` have uncommitted edits (Dirk's and this session's). No runs in flight. Dirk edits the .tex files between turns: re-read before every edit.

## 4. Established and verified
- Code definitions (`model_augmentation/fit_systems/blocks.py`): combination order `COMBO_NAMES` line 1292 = `kb_sum, cg1, cg2, cy, cb_sum, mh, m_total, m_diff, J_eff, d`; `m_total = m1+m2+mb` (line 1173), `J_eff = Jb+Jh+(m1+m2)Lb^2/4` without `mh d^2` (line 1175), `m_diff = m1-m2`. Thesis symbols: `m_\Sigma`, `m_\Delta`, `J_{\mathrm{eff}}`, `k_{b,\Sigma}`, `c_{b,\Sigma}` (II-C bullets; `99_appendix.tex` lines 137 to 150).
- Training coordinates (D-229, `blocks.py` lines 1246 to 1381): log for the nine positive combinations; `m_diff = rho (2/Lb) sqrt(m_total J_eff) tanh(eta0 + s*free)`, rho = 0.99. With `mh, m_total, J_eff > 0`, `m_total J_eff > Lb^2 m_diff^2/4` is necessary and sufficient for `M(Y) > 0` at every Y; it guarantees positive-definite inertia, NOT positive raw masses (D-229 amendment, `docs/decisions.md` line 7219ff). Reduced coordinates are what training uses (D-190 item 1, D-191).
- Appendix already holds: entry list and explicit read-back of theta_base from the matrices, the frozen-Y transfer-matrix argument, and the four null directions (`99_appendix.tex` `app:params`, line 97ff); the completed-square proof of `M(Y) > 0` (`app:lfr-wellposed`, line 173ff); the m_Delta map constants eta0, s and its inverse.
- Labels `eq:phi`, `eq:lfr_admissibility`, `eq:mdelta_map` are referenced in `99_appendix.tex` (lines 137, 189, 200) but defined nowhere: they print as "??". II-C must define them (theta_base display, admissibility inequality, m_Delta map).
- Consumers of II-C: `03_augmentation.tex` lines 121 and 488 (theta_base estimated "in the combinations of Section~\ref{sec:params}"); `04_obc.tex` line 80 ("ten combinations ... remove that ambiguity", OBC protects ten directions).
- II-B now ends with: LFR exact; well posed iff M(Y) nonsingular; M(Y) > 0 for every Y for positive masses and inertias. Dirk REJECTED putting "positive reduced parameters do not guarantee this" in II-B: that point belongs to II-C only.

## 5. Assumed but not verified
- Proof structure proposed to Dirk (not yet approved): (if) the map theta (14) -> theta_base (10) -> (M0, M1, M2, C, K) factors, so equal theta_base gives equal dynamics; (only if) every point with qdot = 0, Theta = 0 is an equilibrium for any X0, Y0 (only yaw has stiffness), so equal input-output behaviour gives equal small-signal transfer matrices H(s) = P^T Z(s)^{-1} P at every Y0; P invertible gives Z(s) = M(Y0)s^2 + Cs + K = P H(s)^{-1} P^T, hence M(Y0), C, K; three distinct Y0 give M0, M1, M2; the explicit left inverse gives theta_base. Conclusion: indistinguishable iff equal theta_base; the four null directions are exactly the fibres; Jacobian rank 10 = 14 - 4 is a corollary. Settle by Dirk's 2a answer, then check each step against the EOM in II-A (not the pipeline).
- Proposed 2a claims (Dirk has not approved them; present them as the brief):
  1. The fourteen raw parameters enter the dynamics only through theta_base (display `eq:phi`); this work's own result.
  2. Proposition: equal input-output behaviour iff equal theta_base (proof above), so theta_base is structurally identifiable and the four null directions are not.
  3. Positivity of theta_base does not give M(Y) > 0; that holds iff m_Sigma J_eff > Lb^2 m_Delta^2 / 4 (`eq:lfr_admissibility`).
  4. Training coordinates: log for nine, tanh-bounded m_Delta (`eq:mdelta_map`), so the LFR of II-B stays well posed at every optimiser iterate; positive-definite inertia, not positive raw masses.
- Page cost: section budget 1.35 pages; a stated proposition with proof adds roughly 0.3 page (estimate, not measured). Report the overrun; Dirk decides.

## 6. Tried and failed
- II-B: AI-drafted wording that described G in words ("takes ... returns ...") -> Dirk could not follow it -> prose paraphrased what one display shows -> replaced by one displayed equation plus one sentence. Lesson for II-C: display the load-bearing objects (theta_base, the inequality, the map), keep prose to what the display cannot say.
- II-B: symbols used before definition (D_zw in the main text) and non-standard terms ("constant block", "constant coefficients", "inputs" for latent variables) -> Dirk rejected them -> use Drenth's terms and define every symbol before use.

## 7. Achieved
II-B rewritten and agreed in this session (`02_system_baseline.tex`, `sec:lfr`); appendix `app:lfr-details` extended with the closed-form M(Y)^{-1} = N(Y)/d(Y), Horner remark and a `\todo` on printing N_i. Not compiled.

## 8. The open question
Nothing blocks II-C. One loose end in II-B: Dirk asked whether he can delete the closed-form sentence at the end of II-B; the answer given was yes. If he did, the II-B draft bullet ("closed-form M(Y)^{-1}, no numerical solve") and the `% MUST ESTABLISH` item 3 in the header still mention it; ask Dirk once whether to update them.

## 9. Next action
Invoke `thesis-section` for II-C and present step 2a: the four claims of section 5 with what needs each, the left-out list (Jacobian rank dropped as corollary, which resolves the existing `\todo` above the bullets; rho and metrics to Setup; eta0, s stay in the appendix; `app:params` shrinks to the left-inverse list and the parameter table), and the source issues (undefined labels; structural vs practical identifiability with closed-loop data). Wait for Dirk.

## 10. Acceptance criterion
Dirk approves the II-C text; the labels `eq:phi`, `eq:lfr_admissibility`, `eq:mdelta_map` are defined; every existing II-C bullet is either covered by the prose or explicitly moved; after Dirk's save, `build/main.log` has no line starting with `!` and no undefined-reference warning for these labels.

## 11. Read these first
1. `.claude/skills/thesis-section/SKILL.md` and its `reference/` files: the workflow and register Dirk approved.
2. `Thesis-writeup/Writing/sections/02_system_baseline.tex` in full: II-A defines the EOM and matrices; II-B the LFR; II-C bullets are the content inventory.
3. `Thesis-writeup/Writing/sections/99_appendix.tex` lines 97 to 216: the existing identifiability and positive-definiteness material.
4. `docs/decisions.md` D-229 (line 7179) and D-190/D-191 (lines 1263 to 1310): why reduced coordinates and the bounded map.
5. `model_augmentation/fit_systems/blocks.py` lines 1160 to 1180 and 1246 to 1385: the implemented combinations and map.

## 12. Do not
- Do not edit II-B prose, Section 03 or 04, `main.tex`, or the build settings.
- Do not compile or run latexmk.
- Do not move the identifiability proof to the appendix: Dirk wants it in the main text.
- Do not add the SVD LFR reduction or the Jacobian printout.
- Do not use the "(eom)" shorthand in chat or text; use `\eqref{eq:eom}`.

## 13. Operational
No runs. Symbolic checks, if needed, as a short sympy script in the session scratchpad, run with `conda run -n GraduationProject python <file>` (quick, foreground). Edit .tex files with exact-string edits; files may have CRLF line endings, so Python string replacement that assumes `\n` fails (happened this session).

## 14. Delegation
None. All sources are named above; no subagent.
