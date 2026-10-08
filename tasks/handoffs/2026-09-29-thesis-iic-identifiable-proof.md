# Handoff: work out the algebraic proof of the identifiable parameter combinations for thesis Section II-C
**From**: session of 2026-09-29 | **Branch**: Augmentation | **Effort suggested**: xhigh (derivation plus careful writing; thinking on paper, no computation)

## 1. Task
The user's words: "focus on the proof for II-C and work that out." Work out, with the user, a clean
algebraic derivation that the fourteen raw gantry parameters enter the model only through ten
combinations phi, that phi is structurally identifiable, and that positive definiteness of M(Y)
imposes the extra condition m_Sigma J_eff > L_b^2 m_Delta^2 / 4 in reduced coordinates. Then write
Section II-C (`Thesis-writeup/Writing/sections/02_system_baseline.tex`, subsection
`\label{sec:params}`, currently an 8-bullet outline) and the matching appendix subsections
(`sections/99_appendix.tex`: `app:params` and `app:lfr-wellposed`) as Phase 1 (Structure) of
`Thesis-writeup/Writing/README.md`. The deliverable is a derivation and text, not code or runs.
Discuss the derivation structure with the user in text first and write into the files only after
the user agrees with it.

## 2. Out of scope
- **Section II-B**: the user is rewriting it by hand (README Phase 3). Do not edit, restructure
  or comment on its sentences. II-B already states: well posed iff M(Y) nonsingular; for positive
  masses and inertias M(Y) > 0 for every Y (Appendix B); "Section II-C treats trained, reduced
  parameters". II-C must connect to exactly that sentence.
- **Section II-A**, Sections III to VIII: not this task.
- **Literature searches**: do not launch any literature agent or search without first agreeing
  with the user which claim needs a source and which candidates (memory
  `feedback_no_agent_fanout`, 2026-09-29 addition). Candidate sources already verified are in
  section 4; the user decides whether and where to cite them.
- **Code, runs, numerical scripts**: none. The rank-ten numerical check already exists (section 4).
- `refs.bib` edits beyond what the user asks for.

## 3. Where things stand
Last commit `023bbeb`. Uncommitted: `sections/02_system_baseline.tex` (II-B marked with `\kw{}`;
the user will commit it as the Phase 2 reference) and build outputs. `99_appendix.tex`, `refs.bib`,
`figures/lfr_loop.tex`, `05_setup.tex` were edited this session (check `git status`).
II-C current content (`02_system_baseline.tex` from `\subsection{Physical parameters...}` on): an
itemize outline plus a commented list of Section V items and the user's `\todo` about printing
the Jacobian. Three appendix references are undefined until II-C defines them: `eq:phi`,
`eq:lfr_admissibility`, `eq:mdelta_map` (appendix text in `app:lfr-wellposed` uses them).
No runs in flight.

## 4. Established and verified
- **Decision (user, 2026-09-29): derive the combinations algebraically, not with a Jacobian.** The
  Jacobian (numerical, local, tolerance-dependent) only counts directions; the algebra names the
  combinations and holds for every positive parameter vector. The trajectory Jacobian belongs to
  Section IV (OBC) and practical identifiability to Section V.
- **The three algebraic steps** (discussed and agreed in outline, not yet written cleanly):
  1. The 12 distinct nonzero entries of M0, M1, M2, C, K are functions of phi only, e.g.
     M0_11 = m_Sigma + m_h, M0_12 = L_b m_Delta / 2, M0_22 = J_eff + m_h d^2, M0_23 = -m_h d,
     M0_33 = m_h, M1_12 = -m_h, M2_22 = m_h, C_11 = c_g1 + c_g2, C_12 = L_b(c_g1 - c_g2)/2,
     C_22 = c_b,Sigma + L_b^2(c_g1 + c_g2)/4, C_33 = c_y, K_22 = k_b,Sigma. Hence raw vectors with
     equal phi give identical dynamics: at most phi is identifiable.
  2. phi is recovered entry by entry (m_h = M0_33, d = -M0_23/m_h, m_Sigma = M0_11 - m_h,
     m_Delta = 2 M0_12 / L_b, J_eff = M0_22 - m_h d^2, c_g1 and c_g2 from C_11 and C_12,
     c_y = C_33, c_b,Sigma = C_22 - L_b^2(c_g1 + c_g2)/4, k_b,Sigma = K_22). Twelve entries carry
     ten quantities because M0_33, M1_12, M2_22 all equal +-m_h.
  3. The matrices are fixed by input-output data: at an equilibrium Y0 the linearized transfer
     matrix is H(s) = P^T (M(Y0) s^2 + C s + K)^{-1} P, P invertible and known (L_b fixed), so
     H(s) gives M(Y0), C, K; three distinct Y0 give M0, M1, M2.
  Definitions: m_Sigma = m1 + m2 + mb, m_Delta = m1 - m2, J_eff = Jb + Jh + (L_b^2/4)(m1 + m2),
  k_b,Sigma = kb1 + kb2, c_b,Sigma = cb1 + cb2. These three steps already exist in rough form in
  `99_appendix.tex` subsection `app:params` (the user judged them "not well worked out").
- **Null directions** (the four raw changes leaving phi unchanged): kb1 - kb2, cb1 - cb2, Jb - Jh,
  and (dm1, dm2, dmb, dJb) = t(-2/L_b^2, -2/L_b^2, 4/L_b^2, 1). Numerically confirmed:
  `scripts/gantry/orthogonal-by-construction/investigation-20260915/results/s12_parameterization_and_rank.log`
  (rank 10 at rel tol 1e-4 to 1e-14 on 24 design points; predicted vs numerical null space
  principal angle 2.6e-8 rad).
- **Positive definiteness**: for m_h, m_Sigma, J_eff > 0, M(Y) > 0 for all real Y iff
  m_Sigma J_eff > L_b^2 m_Delta^2 / 4 (Schur complement on the m_h block plus completed square,
  written in `99_appendix.tex` subsection `app:lfr-wellposed`). For positive raw parameters the gap
  equals m_Sigma(Jb + Jh) + (L_b^2/4)[4 m1 m2 + mb(m1 + m2)] > 0. Violation implies M(Y) singular
  at some real Y (det M(Y) -> +inf as |Y| -> inf).
- **Implementation** (`model_augmentation/fit_systems/blocks.py`, class `Reduced_Gantry_State_Block`,
  D-190, D-191, D-229): ten free coordinates, zero at the start; nine positive combinations
  phi_i = phi_i0 exp(xi_i); m_Delta = rho (2/L_b) sqrt(m_Sigma J_eff) tanh(eta0 + s xi_Delta),
  rho = 0.99 fixed; eta0 and s reproduce the start with the log-coordinate local scaling;
  requires m_Delta,0 != 0. Guarantees M(Y) > 0 at every iterate, NOT positive raw masses (positive
  m1, m2 need |m_Delta| < m_Sigma - mb, mb unidentifiable). Nominal bound 35.18 kg, nominal
  |m_Delta| = 0.5 kg (1.42 % of the bound). The thesis campaign uses this reduced block (D-222).
- **Verified candidate sources** (read in primary text 2026-09-29 unless marked; none contains the
  gantry bound, which is this project's derivation):
  - Gautier & Khalil 1990 (`gautier1990minimum`, in refs.bib): minimum set of inertial parameters
    = "the only identifiable parameters by the use of the dynamic model" (Sec. VI p.371). Its II-C
    sentence is currently commented out by the user until the derivation is settled.
  - Sutanto et al. 2020 L4DC (`sutanto2020encoding`): unconstrained parameterization that always
    satisfies physical constraints, sigmoid-bounded coordinate (closest precedent for the tanh).
  - Drenth et al. Sec. 4.2 (`drenth2025lpvlfr`): direct parameterization guaranteeing
    well-posedness.
  - Wensing et al. 2018 (`wensing2018physical`): physical-consistency constraints as LMIs (SDP).
  - Leboutet et al. 2021 (`leboutet2021survey`): unconstrained estimates can give
    non-positive-definite inertia. Yoshida & Khalil 2000 (`yoshida2000verification`): same, ABSTRACT
    ONLY.
  These keys were added to refs.bib this session (uncited); the user has not decided on them.
  Raue 2009/2013 and the CTSM guide were also added but are about log coordinates, which the
  user said was NOT what they wanted sourced.

## 5. Assumed but not verified
- That the transfer-matrix argument (step 3) is the argument the user wants in the thesis, and
  at which level (structural, ideal data). It relies on linearization at an equilibrium; K has
  stiffness only in yaw, so every (X0, 0, Y0) with zero velocity and input is an equilibrium, and
  the dM/dY q_dd term vanishes there. Settle by discussing with the user.
- Whether II-C should print the entry list (step 1) in the main text or only state the
  factorization and put the list in the appendix. User decision.
- Whether to keep the user's `\todo` about printing the Jacobian; with the algebraic route it is
  probably obsolete. Ask.

## 6. Tried and failed
- Drafting II-C prose before agreeing the derivation -> user: "we first need to determine how we
  derive the combinations" -> the draft mixed Jacobian wording ("raw-to-model Jacobian has rank
  ten") with the algebraic argument -> replaced by the current outline.
- Launching literature agents without agreeing scope -> one agent sourced log parameterization
  while the user meant only the m_Delta (tanh) constraint -> reports clouded the context.
- Writing sections by volume, then cutting -> for II-B this took a full day of iterations. The
  cause: symbols introduced without the reason they are needed, remarks the reader never asked
  for (overbounding, Thm 6, minimality), derivation steps in the main text where papers state a
  result, and circular arguments (re-substituting into an equation derived from the one being
  "recovered").

## 7. Achieved
- II-B final content (user-approved 2026-09-29) and marked for Phase 3; Appendix A holds the LFR
  construction, all G blocks, the loop reduction and the det identity.
- II-C outline (8 bullets) in the file; `app:lfr-wellposed` holds the completed-square proof, the
  raw-positivity identity, the tanh constants eta0, s and the inverse map.

## 8. The open question
How should the identifiability argument be presented: (a) main text states the factorization
phi = h(theta) and the four null directions in words, appendix gives steps 1 to 3 in full; or
(b) main text shows step 1 as a compact table or entry list. Evidence to choose: the user's
page budget (Section II total 1.35 pages, II-A and II-B already long) and the user's preference.
Ask; do not decide.

## 9. Next action
Present the three-step derivation (section 4) to the user in plain text, not LaTeX (the user
cannot read raw LaTeX), as a proposed II-C structure: what II-C states, what goes to the
appendix, and which claims (if any) need a source. Wait for agreement before writing.

## 10. Acceptance criterion
The user approves II-C content (README Phase 1 "Done when: Dirk approves the content") and the
build has no line starting with `!` (test compiles in the scratchpad, never in `build/`). Every
symbol in II-C is introduced at the point of use with its reason; no step of the argument is
circular; the three undefined appendix references resolve.

## 11. Read these first
1. `Thesis-writeup/Writing/README.md`: phases, authorial-voice rules, derivation policy.
2. `Thesis-writeup/Writing/sections/02_system_baseline.tex`: II-B (for the handover sentence) and
   the II-C outline.
3. `Thesis-writeup/Writing/sections/99_appendix.tex`: `app:params` and `app:lfr-wellposed`.
4. `model_augmentation/fit_systems/blocks.py`, `Reduced_Gantry_State_Block` docstring.
5. `docs/decisions.md` D-229 (tanh coordinate, limitation, rho fixed).

## 12. Do not
- Do not edit II-B or II-A.
- Do not launch literature agents or searches without the user's agreement on the claim.
- Do not write prose before the derivation structure is agreed; do not paste raw LaTeX as your
  explanation to the user.
- Do not present the Jacobian as the identifiability argument.
- Do not claim the tanh bound guarantees positive raw masses.

## 13. Operational
Test compile: copy `Thesis-writeup/Writing/{main.tex,util,sections,figures,tables,refs.bib,IEEEabrv.bib,IEEEtran.bst,IEEEtran.cls}`
to a scratchpad folder and run `latexmk -pdf -interaction=nonstopmode main.tex` there; grep the
log for lines starting with `!`. The user saves the file before asking for an AI edit; files may
change on disk while you work (the user edits in parallel), so re-read before each edit.

## 14. Delegation
None. Derivation and writing stay in the main session. Literature agents only after the user
agrees on the claim to source.
