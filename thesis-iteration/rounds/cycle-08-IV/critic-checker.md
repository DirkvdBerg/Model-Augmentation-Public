# Critic (checker lens), cycle 08, Section IV

Score: 7 / 10

## What was checked
- Compile: `latexmk -pdf main.tex` in `thesis-iteration/Writing`. `build/main.log` has no line starting with "!". There are no overfull boxes on `04_obc.tex` lines, only underfull boxes at lines 139 to 152 and 276 to 282. The undefined `eq:obc_overlap` comes from `06_results.tex` line 40, which the header already names. G passes.
- No em-dashes found (Unicode, `---` or `--`).
- Code claims confirmed in this session:
  - `obc_gantry.py::build_reference_set`: q = P^-T y, fourth-order stencil for k in [2, N-3] inside each record, x_a = 0, u = recorded `u_total` (actuator forces), normalised with the training statistics.
  - `obc.py::OBCBasis.from_matrix`: rank tolerance max(M,N) eps S0, with no column scaling on the tangent arm.
  - `OBCLifecycle.refresh`: rank loss keeps the previous basis, and two in a row abort.
  - `interconnect.py::loss`: the coefficient is computed once per objective with autograd, and `vbar` is a detached buffer, so xi gets no gradient through the correction.
  - `_sync_prediction_state_for_validation`: re-solves the coefficient for validation, and `refresh_basis=True` rebuilds after the selected checkpoint is restored.
  - `GantryOBCCorrection.forward`: zero rows on the added states.
- Citations: all of them match `citation-log.md` rows (gyorok2026obc Eq. 2, Sec. 3.2, Eqs. 8, 10, Lemma 3, Eq. 13, Assumption 1, Cond. 4, Thm. 7, Eq. 21, Assumption 6, Remark 5, Thms. 16, 18, Cond. 10, Sec. 6; gyorok2025l4dc Ex. 1, Eqs. 13, 15, 18, Remark 1, Sec. 3, Sec. 4; kessels2025ai Sec. 5.3.1.3, Eq. 5.12).
- The algebra of Proposition 1 and the first-order negation claim were checked by hand and are correct. Every Jacobian/full-rank claim in IV-B is correct as qualified ("at xi^(nu)", "on the tuples").
- Strengths: the roadmap is in Hoekstra style, the identification problem is displayed with the substitution in its own display, trained, fixed and initialisation are explicit, and the D-186 offset caveat and the D-192 slice limits are stated.

## Issues

1. MAJOR (A, E). IV-B, lines 165 to 179 and 208 to 231. The subsection opens with a seven-sentence Gy\"or\"ok positioning block before any construction. Two of its facts then return after `eq:obc_coef`:
   - "fit the stacked network output to the stacked regressor and subtract the fit [Eqs. (8), (10)]" comes back as "fit and subtract as in [Eqs. (8), (10)]";
   - "orthogonal to the regressor on the training data [Lemma 3]" comes back as "By [Lemma 3], Phi'(...)=0, so ... no component along the directions".
   The block should shrink to the problem (IO, linear in parameters, state space named as future work, no penalty weight). The adopted fit and subtract and Lemma 3 should be cited once, at the display.
   Rule cause: skill "Order" ("Literature positioning sits at the component it contrasts ... never as a block before the construction") and "Say each fact once" / "Citations and values" (cite each borrowed component once, where it is used) failed to prevent it.

2. MINOR (A). IV-A, lines 135 to 137 and 147 to 151. Two paragraphs end on a different point from their topic sentence:
   - the Kessels contrast ("Kessels' notion is broader ...") closes the first-order cancellation paragraph, but it belongs with the Kessels sentence that opens IV-A;
   - "We protect the ten combinations ... rank deficient" closes the "each direction has a physical meaning" paragraph. It is a separate choice (protected coordinates) and uses "protect" before IV-B constructs the protection.
   Rule cause: skill "Review loop" topic-sentence check ("a closing sentence on another point moves to the paragraph it serves or starts its own").

3. MINOR (D, C). IV-B, lines 192 to 202. "The velocity uses a fourth-order stencil, because the damping directions multiply it" does not justify the fourth order over a lower order. No decision gives this reason: D-192 item 2 states the stencil without one, and the code cites Fornberg (1988) as THEORY. The display also writes the stencil weights out, which is textbook machinery. Naming "fourth-order central difference" with its source in the lead-in would do.
   Rule cause: skill "Reasons and claims" (every stated reason traces to a source) and the "Math" check (a standard step's result is one sentence, not a display) failed to prevent it.

4. MINOR (F, D). IV-B, lines 233 to 239. The refresh contrast says Gy\"or\"ok et al. update the expansion point at every evaluation or fix it. However, `citation-log.md` records gyorok2025l4dc Remark 2: the projection matrix may be updated with current state estimates, recomputing the SVD at the start of each epoch. That is a precedent for the epoch-wise rebuild, so leaving it out makes the per-epoch choice look like this work's own. It should be cited as adapted (Remark 2 refreshes the state estimates, while here the expansion point is refreshed).
   Rule cause: skill "Reasons and claims" ("Before writing a \todo for a missing reason, search the part's source-map row (... citation-log.md)") and "Contributions" (adopted/adapted/own marking). The IV-B source-map row does not list Remark 2, so the map is incomplete for this choice.

5. MINOR (C, D). IV-D, lines 382 to 412 and 423 to 445. Two problems with the assumptions of Proposition 1:
   - Assumption (i), noise-free records, is never used in the proof, which is purely algebraic. Its only role is to make (iv) plausible (Gy\"or\"ok Assumption 6), so it belongs in the following prose, not in the hypothesis list.
   - Assumption (ii), affine on the segment that contains xi^(nu), holds only approximately for the gantry, since the baseline is rational in theta_base through M(Y)^{-1} (D-190 Why). The scope paragraph never names (ii) as local or approximate.
   Rule cause: skill "Order" Method ("An assumption the gantry meets only locally (affine, linearised, a slice) is named as such in its scope"). No rule requires that every listed hypothesis is used in the proof, so that rule is missing.

6. MINOR (F). IV-D, lines 379 to 380. "The stacks of (obc_coef) are restricted to the tuples that have a successor" silently redefines the construction. The trained model builds Phi and theta_aux over every stencil tuple (`build_reference_set`, k up to N-3). Only the D-203 measurement script drops the last tuple of each record. The proof then uses a theta_aux the trained model does not compute. The sentence should say the proposition concerns the restricted construction (or measurement), not the trained one.
   Rule cause: skill "1. Read" ("Describe the method as the code defines it"). No rule requires that a formal statement's object is the implemented one or is named as a variant.

7. MINOR (F, D). IV-D, lines 448 to 454. "Section VI therefore tests whether the estimate is independent of the start, and reports recovery against this condition" has two problems:
   - "therefore" does not follow: needing xi* does not imply a start-independence test;
   - `DOC/THESIS-RESULTS.md` step 5 compares OBC against U (drift from the correct start, combination errors from the detuned start) and lists no metric for Phi'delta. So "reports recovery against this condition" claims something Results does not plan to report.
   Rule cause: skill "Reasons and claims" (claim as strong as its evidence). The README rule that THESIS-RESULTS wins over a consumer section was not applied to forward pointers made in a method section.

8. MINOR (F). Lines 262 to 270 and 313 to 321. The corrected-model and problem displays mix coordinates:
   - the physical line uses (hat x^n_k, u^n_k);
   - the x-bar line of `eq:obc_model` and the referenced unchanged lines of `eq:aug_loss` use (hat x_k, u_k) and hat y_k = h_base(tilde x_k).
   The maps are consistent through `eq:aug_norm`, but within one display the state appears in both frames.
   Rule cause: skill "Review loop" notation check ("after a normalisation display, every display and both sides of a coordinate-change line use the normalised symbols"). This is partly inherited from `eq:aug_loss` in Section III.

9. MINOR (A). IV-B, lines 181 to 257. The construction's realisation choices sit in separate multi-sentence paragraphs between the tuple, coefficient and model displays, so the subsection partly reads as formula-and-reason blocks:
   - tuple source and drift, with a \todo;
   - refresh, with a \todo;
   - rank rule plus offset, with a \todo.
   Rule cause: skill "Content" ("the others form the one realisation-choices paragraph of their component ... never one paragraph per item"). This conflicts with the skill's own packing example, which presents the three-sentence refresh unit as the target. The skill does not say how the two combine.

10. MINOR (E). Lines 223, 233, 337. That Phi and xi^(nu) are fixed within an epoch is stated three times: the where clause of `eq:obc_coef`, the refresh paragraph, and IV-C ("Within an epoch, Phi and xi^(nu) are held fixed"). Also, "The orthogonality of (obc_coef) then holds for every theta_aug" (line 332) restates the display, where theta_aux is already a function of theta_aug. The consequence is the next clause.
    Rule cause: skill "Say each fact once" (no restatement of an earlier subsection or a display).

11. MINOR (D). Line 257. The Gamma \todo does not use the required "Missing: ... Candidate: ... (basis)." form. It also uses Gamma, which the thesis has not defined for the offset and which Section III-B already uses for Gamma_n (input-to-state map).
    Rule cause: skill "Open points are \todo{}s" (form) and "Notation" (one symbol, one meaning, todos included).
