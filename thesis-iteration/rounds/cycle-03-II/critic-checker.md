# Critic (checker lens), cycle 03, Section II

Score: 7 / 10

## What was checked
- Compile: scratch copy of `thesis-iteration/Writing` with the draft as `sections/02_system_baseline.tex` (identical to the sandbox file), `latexmk -pdf`: no line starting with `!`; no undefined citation or reference from Section II (the undefined keys are `gyorok2026obc`, `forssell1999revisited`, `hefny2015supervised`, `downey2017psrnn` in later sections). One 5.3 pt overfull box comes from the gantry figure input, not from the prose. Submission build (`\draftfalse`): about 1.95 pages against the 1.35 budget (1.44x, figures and the matrix displays included), so under the 1.5x trigger.
- Code: `model_augmentation/systems/gantry_ss.py` (M0, C, P, Lb = 0.725 fixed); `FS/blocks.py::Gantry_State_Block._deriv_with` (closed-form N(Y)/d(Y) loop solve in Horner form; `Y_op` frozen branch = M_LTI capability); `Reduced_Gantry_State_Block.COMBO_NAMES`, `combos_of`, `gauge_section`. Order and definitions of eq:phi match `COMBO_NAMES`/`combos_of` exactly.
- Decisions: D-018, D-022, D-204, D-242, D-243, D-244, the Lb entry (decisions.md l. 5393: Lb fixed because it defines P); THESIS-RESULTS step 1 (LPV and frozen-LTI at mid-stroke, true parameters, no training).
- Algebra: twelve nonzero entries up to symmetry (M0: 5, M1: 1, M2: 1, C: 4, K: 1), three equal to m_h: correct. Jacobian with the stated variable order is block lower triangular with a 2x2 (cg1, cg2) block; det = 1 * (-m_h) * (L_b/2) * 1 * 1 * (-L_b) * 1 * 1 * 1 = m_h L_b^2/2: correct.
- Citations: every key and location in the draft is in `citation-log.md` (garcia2013model Sec. 2.2, 2.3, Eq. 6, Table I; toth2010modeling Def. 7.2; drenth2025thesis Eqs. 2.1, 2.2, Sec. 2.1.1; drenth2025lpvlfr Def. 1, Sec. 3.2, Thm. 6; ovchinnikov2021computing Def. 7) and supports the sentence it sits on.

Strong points: one argument with a short roadmap; II-B runs problem (rational M(Y)^-1), construction (eqs. 6 to 8), consequence (exact, well posed iff M(Y) nonsingular); II-C argues toward the conclusion before naming theta_base, as agreed; every signal has its space; the frozen-LTI model is defined by its equation and its training status; contribution kinds (adopted from Garcia, adapted, own) are visible.

## Issues

1. MAJOR, correctness (F). l. 371 to 372: "a transfer of mass from the X movers to the cross-arm" is not a null direction. Moving eps from m1 and m2 (eps/2 each) to m_b keeps m_Sigma and m_Delta but changes J_eff by -L_b^2/4 eps; the lost direction is that transfer together with a compensating change L_b^2/4 eps of J_b + J_h. The code's gauge section confirms the coupling (`gauge_section`: J_sum = J_eff - (m_Sigma - m_b) L_b^2/4 depends on the fixed m_b). Fix: "a transfer of mass from the X movers to the cross-arm with a compensating change of J_b + J_h".
   Rule cause: skill Workflow 4 "Every reason traces to a source, decision, code ... every claim is as strong as its display or evidence" failed to prevent it; missing: a check that every own mathematical claim made in prose (not only displays) is verified by algebra or against the code.

2. MINOR, agreed content (B, C). l. 339 to 345: the unnumbered display with primed matrices M', C', K' introduces new symbols, while the header MUST ESTABLISH (II-C item 2, agreed 2026-10-07) says "no new symbols". The argument reads without the display ("two parameter vectors with the same behaviour satisfy the difference of their equations of motion along every trajectory").
   Rule cause: missing. The skill Workflow 4 review loop has no check that the draft honours every MUST ESTABLISH item including its restrictions (no new symbols, no display, no environment); Workflow 2a only says to "start from it".

3. MINOR, prose around the math (A, D). l. 301 to 311: after eq:lfr_det the paragraph appends four sentences (iff condition, contrast with Drenth Thm. 6, M(Y) > 0 for all Y, closed-form solve). README "Prose around the math" allows one consequence; the Drenth contrast belongs in the lead-in, and "M(Y) > 0 for every Y, so the loop is solved in closed form" is its own consequence paragraph.
   Rule cause: README "Prose around the math" exists and failed; the header MUST ESTABLISH II-B item 3 bundles "well posed iff ... true for every Y; hence closed form" into one item, which invites one run of sentences. Missing: a note that one MUST ESTABLISH item may need several paragraphs.

4. MINOR, restatement (E). l. 284 to 285: "All blocks of G are constant." repeats l. 230 ("a constant LTI part G"); "Its first block row is the state equation." paraphrases the display (the first row has dx/dt on the left). Both come from MUST ESTABLISH II-B item 1.
   Rule cause: conflict between skill "Say each fact once" / README voice rule 6 (no paraphrase) and a MUST ESTABLISH item worded as sentences; missing: a precedence line that MUST ESTABLISH fixes claims, not sentences, so a claim already visible in a display or an earlier sentence is satisfied without a sentence.

5. MINOR, notation (F). l. 304: rho(D_zw) is the spectral radius, while Section III eq:mdelta_map uses rho as the margin and Section IV's todo already relies on "rho is the margin". One symbol, two meanings.
   Rule cause: skill "Notation" (one symbol, one meaning) failed; its review check lists only symbols the section introduces, so an operator symbol borrowed from a cited theorem slipped through. Missing: the symbol check covers every symbol the section uses, including operators quoted from a cited result.

6. MINOR, contribution claim (F). l. 94 to 95: "The realisation and the combinations ... form the first contribution of Section I", but the Introduction's first contribution names only the LPV-LFR realisation. Needs a `\todo` (Introduction wording) rather than a silent claim.
   Rule cause: skill "Contributions" says the opening states which contribution it delivers but not that a mismatch with the Introduction's wording becomes a `\todo`; missing.

7. MINOR, todo candidate basis (D, F). l. 206: the Coulomb todo cites D-022 as basis for leaving dry friction to the augmentation. D-022 says the baseline "must remain the exact FP model as derived" and covers effects absent from Garcia's equations; Garcia Eq. (6) contains the Coulomb terms, so D-022 argues against omitting them, and the source map lists D-204 under "not" for Section V. The candidate should cite D-204's "Python baseline stays frictionless" only, and note the tension with D-022.
   Rule cause: skill "Reasons and claims" ("A `\todo` candidate meets the same rules") failed; README source map row II-A lists D-022 without saying it does not cover Garcia's own friction term.

8. MINOR, deletion (E). l. 320 to 321: "It is exact only at Y = Y0 and elsewhere misses the terms -m_hY and m_hY^2 of M(Y)." restates "removes the position dependence" (l. 314) and l. 196 to 197.
   Rule cause: skill "Deletion test" / "Say each fact once" failed (no new rule needed).

9. MINOR, packing (E). l. 356 to 358: one sentence holds three facts (Jacobian block triangular, its determinant, independence).
   Rule cause: skill review "Packing pass" failed (no new rule needed).

10. MINOR, attribution (F). l. 147: "In operation the yaw angle stays within tens of microradians." is Garcia's empirical statement (Sec. 2.3) but carries no citation; the citation sits on the next sentence, which takes the approximation.
    Rule cause: README voice rule 11 (check attribution sentence by sentence) failed.
