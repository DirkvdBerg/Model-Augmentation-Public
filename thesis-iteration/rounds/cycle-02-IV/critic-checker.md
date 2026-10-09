# Critic (checker lens), cycle 02, Section IV, 2026-10-09

Score: 7 / 10

Checked: draft.tex in full (identical to Writing/sections/04_obc.tex); Sections II and III for notation and handover;
code `GD/obc_gantry.py` (`build_reference_set`, `GantryOBCCorrection`, `attach_obc`), `FS/obc.py` (`OBCBasis.from_matrix`,
`OBCLifecycle.refresh`), the `_THESIS_FIXED` block of the entry file; decisions D-190, D-192, D-198, D-201 to D-203, D-232
(OBC velocity), D-224, D-226; THESIS-RESULTS step 5; every citation against citation-log.md, plus Gyorok 2026 Sec. 4.3
(Eq. 27, one-step-ahead predictor, Thms. 16, 18) read in the PDF. Compile: `latexmk -pdf main.tex` exits 0, no `!` line
in build/main.log; the three undefined citations are in Section III (forssell, hefny, downey), not in this section.

Verified correct against code and decisions: reference tuples (q = P^-T y, fourth-order central difference, stencil inside
the record, x_a = 0, recorded normalised input, controller absent); six protected rows (`ann_route_ix = range(6 + n_a)`);
economy SVD and the tolerance max(M, N) eps S[0]; rank loss keeps the previous basis and two consecutive losses abort;
coefficient once per objective with gradient through the solve; basis per epoch, detached; D-198 validation sync and the
final rebuild at the selected checkpoint; offset excluded (D-192 item 1); delta* in normalised coordinates with the same
stencil for the successor (D-202, D-203); M_OBC starts from PS2 (THESIS-RESULTS R1, R2). Proposition 1 and the derivation
of Eq. (obc_bias) are correct. Every citation location is in the log and supports its sentence. No em-dashes.

Strengths: roadmap in Hoekstra's form, one job per subsection, adopted/adapted/own visible, the scope paragraph and the
unique-is-not-true argument are tight and correct, the proof is short and is the section's own contribution.

## Issues

1. (major, D/A) IV-B splits the realisation choices of one component into one paragraph per choice: rank and tolerance
   (lines 197 to 206), refresh rates (233 to 247) and offset (249 to 258), each a short block after the displays. README
   "Prose around the math" says the realisation choices of one component form one paragraph, not one per choice. The
   subsection therefore reads as construction, then a list of choice blocks, the pattern Section context rejects.
   Rule cause: README "Prose around the math" states the rule, but the skill's Review loop has no check for it, and the
   header "Choices to justify" list (skill "The header": treat as content to consider) pulls toward one block per item
   (writer note 3).

2. (major, F) The velocity `\todo` (line 173) could have been resolved: docs/decisions.md "D-232 OBC velocity result"
   (2026-09-29) measured the noisy-mode reference basis against the noise-free one, sin of the largest principal angle
   1.94e-3 against the 0.01 criterion, PASS, no change. The candidate ("build the tuples from the noise-free twin") is
   therefore wrong, and the skill forbids a `\todo` that the session could resolve. Rule cause: README "Source map per
   subsection" row IV-B omits D-232 (OBC velocity) from its Decisions cell, and `reference/sources.md` trap 7 states the
   `cfg.snr` guard without the D-232 result, so a writer following the map finds the trap and not its resolution.

3. (minor, E) IV-A opens by restating Section III: "The baseline alone is not ambiguous ... The network makes the split
   non-unique" (lines 71 to 73) repeats the last paragraph of Section III-A and the section's own roadmap, although the
   writer's MUST ESTABLISH item 1 says not to repeat it. Only the Kessels sentence (the term "negation") is new. Rule
   cause: skill "Say each fact once" exists; the Review loop check "Nothing restates ... an earlier section" was not
   applied to topic sentences that carry a pointer, and no rule says that a pointer does not license a restatement.

4. (minor, F) "negation has an exact first-order form" and "hides any displacement p ... from the data" (lines 78, 94 to
   95) overstate the display, which holds up to O(||p||^2): the hiding is to first order only. Rule cause: skill
   "Reasons and claims" (state every claim as strongly as its evidence supports) was not checked against the display's
   own remainder term; the Review loop has no item comparing a claim with the display that supports it.

5. (minor, D) The refresh-rate paragraph gives why not every evaluation ("Updating rebuilds Phi_R at every evaluation")
   only as a cost fact, without the reason D-190 records for per-epoch sufficiency: the expansion-point sensitivity is
   mild (principal angle 1.5e-3 rad at 1 % displacement). The epoch choice is left as an unjustified compromise. Rule
   cause: skill "Justify only what an examiner would question" is applied, but the source map row IV-B lists D-190 under
   "not" ([J, c]) without saying its refresh rationale still holds, so the writer could not use it.

6. (minor, D) The alternative to reconstructed reference states is argued only against an open-loop baseline
   simulation ("drifts on the free X and Y axes"); Section III-C itself simulates the baseline in closed loop, which does
   not drift. An examiner will ask why that is not used. D-190 (3) records that a baseline-only reference set was
   declined by the user, and reconstruction from measured positions also avoids states from a detuned baseline. Rule
   cause: skill "Justify only what an examiner would question" lacks a prompt to test the stated reason against
   alternatives the thesis itself already uses.

7. (minor, E) Reader test: "theta_aux is computed from an economy SVD of Phi_R, which avoids squaring its condition
   number" (lines 200 to 201) is implementation detail: the model is the same either way. The tolerance formula is the
   numpy matrix_rank convention, standard and not part of the argument. Keep only the rank requirement and the rank-loss
   rule (method-defining). Rule cause: skill "The reader test" covers it; the header "Choices to justify" lists "rank
   tolerance", which the skill tells the writer to cover, so the two rules pull in opposite directions without a
   precedence.

8. (minor, E) Packing and deletion: line 359 to 361 joins two independent facts with "and" ("gamma = 0 exactly when the
   condition holds, and for the corrected network on held-out tuples, gamma shows ..."); line 278 "so the normalisation
   sets its metric" restates the clause before it. Rule cause: skill Review loop "Packing pass" lists semicolons and
   sentences over 30 words only, so an "and" joining two facts in a sentence under 30 words is not caught, although the
   skill's own example forbids it.

9. (minor, F) The overlap gamma (Eq. obc_overlap) is a metric displayed in a method section with no confirmed consumer:
   README ownership gives metrics to Experiment design, and THESIS-RESULTS step 5 reports only combination errors (the
   writer's own `\todo` says so). Either Results uses it and Experiment design lists it, or the display goes. Rule cause:
   README "Paper structure and ownership" assigns metrics to V but says nothing about a check quantity that belongs to a
   method's own guarantee; the skill has no rule for a display whose consumer is undecided.

10. (minor, F) The friction `\todo` candidate (line 362) states the truth friction as "-c sign(qdot)" with inner product
    "-c sum|qdot_k|" against the damping column. The thesis truth is `cc_i tanh(g v_i)` per actuator (D-224, D-226,
    g = 1000), and the damping column of Phi is M(Y)^-1 C-derivative times qdot through RK4 in normalised coordinates, so
    the closed-form inner product is not that sum. The conclusion (Condition 4 fails) is plausible, the stated mechanism
    is not exact. Rule cause: skill "Reasons and claims" and the Review loop check on citations inside `\todo` candidates
    do not extend to mechanism claims inside a candidate.

11. (minor, F) Notation: the section introduces Phi_R, delta*, gamma while Sections VI, VII and the appendix use J,
    Delta*, rho* for the same quantities. The writer's reason is sound (rho is the m_Delta margin, Delta the scheduling
    block, J also J_val and J_eff), and a `\todo` records it. The draft is acceptable; the rules are not. Rule cause:
    skill "Notation" requires both "keep the symbols the other sections use" and "one symbol has one meaning" and gives
    no precedence when an existing later symbol already clashes (writer note 5).

## Not counted
Length 1.9 pages against 1.55 (1.2x, under the 1.5x trigger). The training-coordinate Jacobian (code differentiates in the
free coordinates, D-192 item 1) is left out; ran Phi is the same under the diffeomorphic reparametrisation, so the
corrected model is identical and the reader test supports the omission.
