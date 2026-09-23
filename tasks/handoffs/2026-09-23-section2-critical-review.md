# Handoff: critical review of thesis Section II (gantry system and physics baseline) against the whole thesis
**From**: session of 2026-09-22/23 | **Branch**: Augmentation | **Effort suggested**: high

## 1. Task
Critically review `Thesis-writeup/Writing/sections/02_system_baseline.tex` (Section II, prose in
II-A, II-B, II-C plus its appendix part in `sections/99_appendix.tex`, subsection
"Identifiability of the physical parameters") as a component of the full 12-page IEEE thesis,
and report to the user in text, before any edit. The report has three parts, in this order:
(1) **Decisions to defend**: every modelling or presentation choice in Section II, classified as
"justification already adequate", "needs a stated reason in Section II now", or "defend later,
in section N"; for each one needing a reason, give the one-sentence reason and its evidence
(paper passage, `D-` number, code, derivation). (2) **Missing**: content Section II must contain
for Sections III to VI to stand on it (notation, definitions, claims later sections cite), with
the later section that needs it. (3) **Should not be here**: content that belongs in another
section, the appendix, or nowhere, with the destination. Also flag every claim that is wrong,
unsupported, or inconsistent with the code or with Sections III to VI. Then wait for the user to
choose what to change; edit only what they approve.

## 2. Out of scope
- Rewriting II-A, II-B or II-C wholesale: the user edited all three after this session drafted
  them; their wording is the baseline. Propose line-level changes only.
- Deleting the `\begin{itemize}` bullet lists in any section: the user deletes those personally.
- Writing Sections III to VI: read their outlines for consistency only.
- Running training, sims, or new numerical scripts (session-limit rule in memory).
- `kamtin-fp-model/` (read only), `kamtin-data/` (blocked).
- The introduction (`01_introduction.tex`): drafted in this session from the research plan with
  `\todo` notes; reviewed separately.

## 3. Where things stand
Branch `Augmentation`, last commit `380350c`. Tree dirty in `Thesis-writeup/Writing/`
(sections 02, 04, 05, 99, `refs.bib`, `figures/gantry_system_with_absorber.tex`, build files)
and untracked `Thesis-writeup/Code/physical_parameter_identifiability.py`. Nothing running.
Build note: `Writing/main.aux` (stale, tracked) sits next to `main.tex`, so pdflatex reads it
instead of `build/main.aux` and reports every citation and reference as undefined. Judge
undefined references only from a clean copy of `Writing/` built in the scratchpad
(pdflatex, `bibtex build/main` run from the `Writing` root, pdflatex twice).

## 4. Established and verified
- Garcia-Herreros 2013 (local `literature/gantry/garcia.txt`): coordinates (X, Theta, Y) avoid
  closed kinematic chains and expose load-distribution coupling (Sec. 2, lines ~395-400).
  Neglected: cross-arm vibration on the flexible-plate support, PMLSM detent forces, friction
  variation along the stroke (lines ~1660-1680). Separately a 37.7 Hz first resonance of the
  supporting structure along X, handled by a 26.5 ms jerk time (lines 1644, 2693). Both exist;
  the current II-A text says "supporting-structure vibration", check it against this.
- Toth 2010 book, Def. 7.2 (PDF p. ~193): quasi-LPV = scheduling variable not free. Research-plan
  sentence "Since Y is a system state rather than an exogenous signal ... quasi-LPV" is reused.
- Drenth 2025 SYSID paper (`literature/lpv-lfr/drenth2025_lpv-lfr-rational.pdf`), Definition 1:
  LPV-LFR well posed iff I - D_zw Delta(p) nonsingular for all p in P (discrete time). Cite
  Drenth for the definition and model class only; the equivalence with det M(Y) != 0 is this
  project's derivation (`docs/Thesis-documentation/LPV_LFR_internal_loop_Well_posedness`).
  Drenth uses "self-scheduled" for a jointly estimated scheduling map; here the map p = Y is known.
- Roland Toth feedback (`LPV/Feedback Supervisor/Roland_Toth_Feedback.md`): use the z/w
  construction, not v = M(Y)^-1 f; call G "the LTI part", not "interconnection matrices"; define
  "logical coordinates" (thesis now says generalized q = [X Theta Y], actuator q_act = [X1 X2 Y]).
- Identifiability (`Thesis-writeup/Code/physical_parameter_identifiability.py`, symbolic): the 12
  nonzero entries of M0, M1, M2, C, K have Jacobian rank 10, nullity 4 in the 14 raw parameters;
  null directions: kb1 vs kb2, cb1 vs cb2, Jb vs Jh, and (dm1, dm2, dmb, dJb) = t(-2/Lb^2,
  -2/Lb^2, 4/Lb^2, 1). phi is recovered entry by entry (injective). The frozen small-signal
  argument (linearize at any equilibrium [X0 0 Y0], P invertible) makes it input-output.
- Admissibility (symbolic, session): with m_h, m_Sigma = m1+m2+mb, J_eff > 0,
  min_Y det M(Y) = m_h (m_h+m_Sigma)(4 J_eff m_Sigma - Lb^2 m_Delta^2)/(4 m_Sigma), so M(Y) > 0
  for all Y iff m_Sigma J_eff > Lb^2 m_Delta^2 / 4. The user wrote this into the appendix.
- Code: `Gantry_State_Block` (`model_augmentation/fit_systems/blocks.py:660`) computes the rational
  inverse N(Y)/d(Y) in Horner form, then z, w, and xdot through G; Y = x[2] at every RK4 stage.
  Entry script: `fs_new=4000`, `up_sample=1` (one RK4 step per 0.25 ms sample; config default
  is 2). Output map in the augmentation pipeline: `Cd = [P^T 0]` (`model_augmentation/systems/
  gantry_ss.py:133`); `lpv_lfr_baseline` uses [I3 0]. Section II now writes the LFR output as q
  (C_q) with q_act = P^T q at the interface.
- Reduced block (`blocks.py:1246` onward): 9 combinations in log coordinates, m_diff signed with
  relative-linear coordinate; docstring states the stacked RK4 transition has rank 10, nullity 4
  and "positivity is not admissibility".
- Generator limits (`Matlab-scripts/Augmentation/data/gtd_config.m:98-99`): pos_X 0.375, pos_Y
  0.400 m; datasets cover Y in [-0.30, 0.30] m (D-206).
- Kessels thesis Ch. 6 (PDF p. ~222): augmentation compensates incorrect FP parameters and
  orthogonality is missing beyond linear-in-parameter cases (supports the gap; intro only).

## 5. Assumed but not verified
- Gautier & Khalil 1990 (DOI 10.1109/70.56655) as the "base parameters" analogue: metadata from
  Crossref only, passage unread; entry `gautier1990minimum` in `refs.bib` is currently uncited.
- Stacked-Jacobian rank on the training data: docstring only, no saved artefact.
- Frozen-LTI comparator result (Section VI, `06_results.tex:21`): no artefact exists.
- Whether any runtime admissibility check exists: none found; II-C carries a `\todo`.
- Parameter provenance: values come from `kamtin-fp-model` main.m; no identification report found.
- Friction stick threshold: the `05_setup.tex` bullet (v_brk = 2.25e-3 m/s, Lee et al. 2020)
  conflicts with D-204 (`V_EPS=(cc1+cc2)/m_total*ts`, ~3.3e-5 m/s before its 3-10x margin).

## 6. Tried and failed
- Inverse-first LFR derivation (v = M(Y)^-1 f) -> rejected by Roland as a collapsed inverse
  interpretation -> use z/w construction -> Roland feedback file, page 3.
- Equation counting to show 10 identifiable parameters -> 12 entries vs 14 unknowns suggests
  nullity 2, true nullity is 4 -> three entries all equal +-m_h -> use rank/injectivity instead.
- One-sample transition Jacobian to show identifiability -> 6x14, rank at most 6 -> cannot show
  10; only stacked or coefficient Jacobians can.
- Gantry figure traced 1:1 from Garcia Fig. 2 -> Elsevier reproduction, not adaptation, and too
  wide -> own adapted schematic `figures/gantry_system_with_absorber.tex` (absorber shaded).

## 7. Achieved
- Section II prose II-A to II-C written and user-edited; Fig. with panels (a) schematic
  (`figures/gantry_system_with_absorber.tex`) and (b) LFR loop (`figures/lfr_loop.tex`) in a
  `figure*`. Appendix identifiability proof and admissibility condition written.
- Labels: `sec:lfr`, `sec:params`, `sec:setup` (05), `sec:obc` (04), `sec:results` (06),
  `app:params`. Macros `\qact \uact \fnet \Dsch` in `util/format.tex`; tikz in `util/include.tex`.
- `refs.bib`: 11 introduction entries copied 1:1 from the research-plan bib (two with `% CHECK`),
  plus `lee2020feeddrive`, `gautier1990minimum`. Bibliography enabled in `main.tex`.

## 8. The open question
Nothing blocked. The review itself is the open work; the user wants it critical.

## 9. Next action
Read the files in section 11, then deliver the three-part review of section 1 in text. Known
candidates to confirm or reject (not verdicts):
- Defend: why Y is the scheduling variable physically (payload moves mass along the arm), why
  continuous time instead of pre-discretized frozen models (the text states the choice but not
  the reason), why the full six-channel loop (now justified in II-B), why report in phi.
- Missing: justified Y operating range (`\todo` in II-A); the role of the baseline for Section III
  (initial model and fixed/estimated part); consistency of the output symbol (II-B uses q and
  C_q, other sections may use y); Sylvester's criterion citation (`\todo` in II-A).
- Not here: the II-C bullet on relative combination error (agreed to move to Section V or OBC
  results); "exact-model parameter recovery" bullet (scope note, not prose); Horner-form
  implementation detail (keep one clause or move to the appendix).
- Citations: II-B cites `drenth2025thesis`, which is disabled in `refs.bib` (renders undefined);
  the verified source for Definition 1 is the SYSID paper (`drenth2025lpvlfr`, still a
  placeholder entry). Decide the key and add a verified entry. `toth2010modeling` and
  `drenth2025lpvlfr` are placeholders.
Example of the wanted format for one item:
"Continuous time instead of frozen ZOH models | needs a reason in II-B now | Reason: the
scheduler Y changes within a sample during 1.5 m/s moves, and RK4 re-evaluates M(Y) at every
stage, so no frozen-per-sample approximation is introduced | Evidence: blocks.py:660, D-206
kinematics."

## 10. Acceptance criterion
Every sentence and equation of Section II and its appendix part is covered by one of: defended
(with evidence), flagged (wrong/unsupported/inconsistent, with the fix), or relocated (with the
destination); and every Section II item referenced by Sections III to VI outlines is present.

## 11. Read these first
1. `Thesis-writeup/Writing/sections/02_system_baseline.tex` : the object of review, with its writing guide.
2. `Thesis-writeup/Writing/sections/99_appendix.tex` : the Section II proofs.
3. `Thesis-writeup/Writing/README.md` : section workflow, choice-record rule, derivation policy.
4. Outlines in `sections/03_augmentation.tex`, `04_obc.tex`, `05_setup.tex`, `06_results.tex` : what later sections expect from Section II.
5. `docs/Thesis-documentation/Meeting-audit/thematic-thesis-audit.md` themes 1, 2, 7 : supervisor-derived safeguards for this section.

## 12. Do not
- Do not reintroduce the inverse-first derivation, "logical coordinates", "interconnection
  matrices", the six-to-four SVD reduction, or equation counting as the identifiability argument.
- Do not cite papers for claims not read in the primary source (README reference gate).
- Do not delete bullet lists or `\todo` notes the user has not approved removing.
- No em-dashes or double hyphens in any output (CLAUDE.md).

## 13. Operational
Clean build: copy `main.tex util sections figures tables refs.bib IEEEtran.cls IEEEtran.bst
IEEEabrv.bib` from `Thesis-writeup/Writing/` to a scratchpad folder, then
`pdflatex -output-directory=build main.tex`, `bibtex build/main`, pdflatex twice. Symbolic check:
`conda run -n GraduationProject python Thesis-writeup/Code/physical_parameter_identifiability.py`.

## 14. Delegation
None. Targeted reading of about ten known files; do it inline.
