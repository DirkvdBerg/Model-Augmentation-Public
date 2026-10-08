# Handoff: rewrite thesis Section IV (orthogonal by construction) cleaner and more concise, keeping every decision the user already made
**From**: session of 2026-09-30 | **Branch**: Augmentation | **Effort suggested**: high (writing with technical judgment; no derivations to redo)

## 1. Task
Rewrite the prose of `Thesis-writeup/Writing/sections/04_obc.tex` so it is cleaner and more concise, subsection by subsection (IV-A, then IV-B, then IV-C). After each subsection, reflect on it against the user's standpoints in section 4 below, fix what fails, and only then continue. The current file already contains a complete prose draft with all decisions applied; your job is to tighten wording and structure, not to change content decisions. Keep every formula that carries meaning as a displayed equation. Report back per subsection with a short sentence-to-content check (which sentence states what, and its source: ours, a Györök citation, or a D- number).

## 2. Out of scope
- Changing any content decision listed in section 4 (notation, Γ exclusion, x̄ = 0 definition, input frame, what is shown where). If you disagree, say so in one sentence and leave the text as decided.
- Sections II, III, V, VI, VII, appendix: do not edit. Section II and the appendix were already renamed to θ_base and x̃ this session. Section V line 51/52 still use φ (user has not approved renaming it; flag, do not edit).
- `refs.bib`: the `gyorok2026obc` entry is missing; do not add it (reference verification gate in `Thesis-writeup/Writing/README.md`). Leave the existing `\note`.
- Compiling: do not run latexmk or pdflatex. `Writing/.latexmkrc` forces output into `build/` and a test compile this session overwrote the editor build. The user's editor rebuilds on save.
- Phase 2/3/4 of the Writing README (marking `\kw{}`, rewrite, check): this is phase 1 (structure/content) only.

## 3. Where things stand
Branch `Augmentation`, dirty tree (many unrelated files). Files touched this session: `Thesis-writeup/Writing/sections/04_obc.tex` (full prose draft), `02_system_baseline.tex` (x to x̃ at lines ~203 and ~218; φ to θ_base in Sec. II-C bullets), `99_appendix.tex` (x to x̃ in the LFR block; φ to θ_base incl. the tanh map), `05_setup.tex` (one bullet added: parameter prior), `07_discussion.tex` (two Limitations bullets: pointwise no-go, Coulomb inner product). No runs in flight.

## 4. Established and verified (the user's standpoints; build on these)
Notation (user decision, follows Hoekstra et al. 2025 EJC `literature/augmentation/hoekstra2025_lfr-augmentation-ejc.pdf` Table 1 and Eq. 2 to 3, and the slides `scripts/gantry/meeting/meeting-18-09-2026/slides/obc-slide-panels*.tex`):
- x̃ physical state col(q, q̇), x̄ added states, x̂ = col(x̃, x̄); f_base(θ_base, x̃, u), f_aug(θ_aug, x̂, u) into physical rows, g_aug(θ_aug, x̂, u) into added rows.
- θ_base = the ten identifiable combinations (replaces φ, which in Györök is the regressor and in Hoekstra the component functions), θ_aug = ANN weights, θ̄_base = expansion point.
- Φ_θ̄(x̃, u) = ∂f_base/∂θ_base at θ̄_base (Györök's regressor symbol; replaces my earlier J), stack Φ_θ̄(X̃, U), F_aug = stacked f_aug on the tuples, θ_aux = Φ⁺F_aug (Györök's symbol).
- No S for the protected space: write ran Φ_θ̄(X̃, U). No ξ in Section IV (optimizer coordinate; same range; appendix only). No ρ for overlap: ρ is already the tanh bound (0.99) and a spectral radius; use cos∠(v, ran Φ) and cos∠(v, F_base) with ⟨a, b⟩ = aᵀb.
- u = P u_act (generalized force, consistent with Sec. II). Verified in code: tuples use recorded `u_total` [F_X1, F_X2, F_Y] (`scripts/gantry/gantry_dynamic/data.py` ~line 143) and the block applies P internally (`model_augmentation/fit_systems/blocks.py` lines 828, 916); state is logical, q = P⁻ᵀy (`scripts/gantry/gantry_dynamic/obc_gantry.py`, `build_reference_set`).

Content standpoints:
- Our contribution is Györök's orthogonal by construction (2026, `literature/Orthogonality/gyorok2026_...arXiv2511.01321.pdf`) adapted to our nonlinear, LPV, closed-loop, state-space gantry model with added states, AND its application to the gantry. Both must be visible. The user rejected a generic IV-A: show gantry-specific formulas.
- Show the OBC formulas we work with, compactly and cited (Györök 2026 Eqs. 8, 10, 11, 13), without the derivation. Rule agreed: display the working formulas of the method we use, inherited ones cited; no proofs; do not repeat displays of Sections II/III. "Display only what is ours" was rejected as too strict.
- Formulas are displayed, never compressed into prose to save space. No word budgets; say what needs to be said, concisely.
- Regressor comes from the regularization paper (Györök 2025 L4DC, `literature/Orthogonality/Hoekstra - Orthogonal projection-based regularization for efficient model.pdf`): Eq. 15 Jacobian. Do not display the Taylor expansion itself; define Φ directly as the Jacobian. Use ≈ (as Györök), not O(·) remainders.
- Γ (affine offset, 2025 Eqs. 16 to 18) is NOT protected. One sentence in the text: it multiplies no estimated parameter. Do not display Γ's definition (user found showing it incoherent with excluding it). Keep the `\todo` that the choice (parameter uniqueness, 2026 Eq. 39, vs output overlap, 2025 Eq. 18) must be confirmed with supervisors (D-190 records an output-overlap condition). The affine arm is removed from the campaign (`Thesis-writeup/Documentation/RESULTS-DESIGN.md` change 30; D-222 arms are obc and u only).
- Orthogonality is DEFINED one-step with respect to the baseline; the baseline has no added states, so the reference tuples are baseline states with x̄ = 0 by definition. Added states affect later physical rows through the recursion, which the one-step definition does not cover (this is what the slide "Where the two components are added" conveys).
- The f_aug / g_aug split and the corrected augmented state equation must be shown in state-space form (central display, IV-B).
- Gantry sensitivity display: ∂q̈/∂θ_j = −M(Y)⁻¹[∂_jM q̈ + ∂_jC q̇ + ∂_jK q]; mass group (m_h, m_Σ, m_Δ, J_eff, d) via q̈, damping (c_g1, c_g2, c_y, c_b,Σ) via q̇, stiffness k_b,Σ via q; negation = learned write acting as added mass/damping/stiffness force; Y enters via M(Y)⁻¹. Verified: `scripts/gantry/orthogonal-by-construction/baseline-and-added-dynamics/baseline-and-added-dynamics.md` §8.1 to 8.2 (terms vanish exactly for the continuous-time derivative at q̇ = 0; through RK4 only to leading order in T_s).
- Scope and recovery condition stay in IV-C (method section, per IEEE practice, like Györök 2026 Sec. 4); Discussion refers back, does not restate. Limitations stated once, strongest form.
- Recovery: bias Φ⁺Δ\* and Φᵀ Δ\* = 0 are the state-level counterpart of Györök 2026 Cond. 4 / Eq. 21, valid only noise-free, first order, Φ at θ\*, one-step least-squares surrogate (training is closed-loop multistep). No nonzero missing force is orthogonal at a single tuple, so the condition needs designed excitation (Remark 5); measurement details are in Discussion.
- Parameter prior: now a bullet in Section V, not in IV. The 87 % / 4.5 % Γ evidence and ρ\* = 0.205 / 4.96 % are dropped (earlier absorber-only dataset, not thesis truth T_AF).
- Keep all four `\todo{}` in 04_obc.tex verbatim; keep the `\note{}` lines.

Implementation facts behind IV-B (all verified in code/decisions): reference tuples from every training sample, fourth-order stencil inside each record, x̄ = 0 (D-192, `obc_gantry.py`); θ_aux computed once per objective in float64 via SVD and differentiated through; Φ and θ̄ refreshed per epoch at the current θ_base, rank loss aborts after two consecutive refreshes (`model_augmentation/fit_systems/obc.py`, D-192); correction applied as a hand-written forward sensitivity through the RK4 stages, `torch.func.jvp` is only the reference (D-194); range(Φ) is invariant to the ξ reparameterization (D-229); validation re-synchronizes θ_aux (D-198, appendix detail only).

## 5. Assumed but not verified
- That the current displays fit an IEEE column: not compiled. Every multi-term display was split with `aligned`; the user's editor rebuild will show overfull boxes in `build/main.log`.
- "The stack has full column rank, checked at every rebuild": true for the code path and measured rank 10 on the earlier dataset (PROGRESS.md stage 8 preflight); not yet measured on the thesis data (belongs in Results R4).
- 4th-order stencil gain error below 0.2 % in the thesis band 106 to 297 Hz: estimated about 0.12 % at 297 Hz, not independently checked. Not currently stated in the text.

## 6. Tried and failed (do not repeat)
- A generic IV-A (Taylor expansion, stack, Γ discussion, no gantry physics) -> user: "does not seem like a proper state space contribution" -> it read as Györök restated -> add gantry sensitivity and the state-space split.
- Displaying Γ's definition while saying it is excluded -> user: "incoherent" -> one sentence only.
- Naming the protected space S -> user: undefined, unnecessary -> use ran Φ.
- Word budgets per subsection -> user rejected; focus on content and concision.
- Writing LaTeX through bash heredocs -> the tool collapsed every `\\` into `\` (11 broken line breaks, fixed) -> use the Edit/Write tools only.
- latexmk test compile with `-outdir` -> `.latexmkrc` forced `build/`, overwrote the editor build and emptied the bibliography -> do not compile.

## 7. Achieved
`04_obc.tex` holds a complete prose draft (implemented, not user-approved as final): lead-in; IV-A with displays eq:obc_sens, eq:obc_negation, eq:obc_gantry_sens; IV-B with eq:obc_gyorok, eq:obc_tuples, eq:obc_ours (stack, F_aug, θ_aux, orthogonality, gradient relation), eq:obc_model (corrected augmented state equation); IV-C with eq:obc_delta, eq:obc_bias, eq:obc_overlap. Labels sec:obc_negation, sec:obc_map, sec:obc_scope. The user's verdict: content right, text "a lot", should be cleaner and more concise.

## 8. The open question
Nothing blocked. Supervisor-level question left as `\todo`: protect range Φ (parameter uniqueness) or range [Φ Γ] (output overlap).

## 9. Next action
Rewrite IV-A in place in `04_obc.tex`: same displays (sensitivity, negation, gantry sensitivity), fewer and tighter sentences, one claim per sentence, gantry meaning kept; then reflect against section 4 and report the check table before moving to IV-B.

## 10. Acceptance criterion
The user approves each subsection. Per subsection: every sentence maps to one item of section 4 or a cited source; no sentence restates a display; all displays from section 7 remain (or a merge is named and justified); all `\todo{}` present verbatim; no em-dash or double hyphen; braces and environments balanced.

## 11. Read these first
1. `Thesis-writeup/Writing/sections/04_obc.tex`: the draft to tighten.
2. `Thesis-writeup/Writing/README.md` sections "Authorial voice" and "Derivation policy": the style rules.
3. `scripts/gantry/meeting/meeting-18-09-2026/slides/obc-slide-panels.tex`: the user's preferred notation and the state-space framing.
4. `literature/Orthogonality/gyorok2026_orthogonal-by-construction-augmentation_IFAC-JSC_arXiv2511.01321.pdf` Sec. 3 to 4: the cited construction and Cond. 4.
5. `Thesis-writeup/Writing/sections/02_system_baseline.tex` Sec. II-C: θ_base definition and ten combinations.

## 12. Do not
- Compile, or write into `Thesis-writeup/Writing/build/`.
- Edit LaTeX via bash heredocs (backslash collapse).
- Reintroduce φ, J, S, ξ, ρ, η, Taylor-expansion displays, O(·) terms, the Γ definition display, the affine arm, or the old-dataset numbers.
- Remove or reword any `\todo{}`.

## 13. Operational
No runs. Editing only. Python env `GraduationProject` not needed.

## 14. Delegation
None. Targeted editing of one file.
