# Critic (examiner lens, Maarten), cycle 11, Section IV

Score: 7 / 10

Verdict: an examiner sees what is built (sensitivity Phi in xi, tuples, least-squares coefficient, corrected write on the six physical rows), what is trained (vartheta unchanged, theta_aux not a parameter) and against which cost (problem display with the substituted line). The roadmap, four one-job subsections and the short Proposition 1 with proof follow Hoekstra/Drenth. What keeps it from 8 or more: IV-B is a long run that builds tuples before saying what they feed, the orthogonality property is stated four times, IV-C does not say how the correction enters the PS2 stages the model is trained with, and a notation slip between the correction and the problem display. Compiles (build/main.log has no "!" line); no em-dashes.

## Major

1. B, "which model on which problem for each phase" (IV-C, lines 315-316). "Adam, the L-BFGS polish and the PS2 stages ... run as for M_PS2, with this transition" does not say whether S1 (temporary network on the closed-loop residual) and S2 (fit of the uncorrected g_aug) see the corrected write, nor what each stage changes under OBC. M_OBC is defined with PS2 (line 327), so the reader cannot reconstruct its training. Rule cause: skill "Order", Method bullet ("each phase of a training or initialisation states which parameters it changes ... and so what the next phase starts from") exists but is not enforced when a model reuses another's phases; README "Math standard" item 3 covers only the V problem, not reused initialisation phases.

2. A/E, one fact stated four times (IV-B). The orthogonality Phi^T(f_aug - Phi theta_aux) = 0 is line 4 of eq:obc_coef, then "The last line is Lemma 3 ... holds for every theta_aug" (226-227), then lines 250-255, then again 275-278 ("its derivative ... vanishes too. An update of the network therefore never moves the corrected stack into ran Phi"). The display line is a consequence (Lemma 3), not a definition. Rule cause: skill "Rules that produce this style", "Say each fact once"; README "Prose around the math" ("at most one consequence") does not say that a consequence written as a display line counts as that consequence.

3. A, order inside IV-B (lines 152-200). The subsection runs: problem (partial states), tuples display with stencil argument, then three todos, then Gyorok's linear construction, penalty history, nonlinear extension, offset exclusion, and only then the coefficient display that the tuples feed. The tuples are built before the reader knows they are the auxiliary evaluations of a least-squares fit; the literature contrast is a block of five sentences before the construction. Rule cause: skill "Order" ("that method's facts appear once, in the lead-in of the display that adopts or adapts them"; "Literature positioning ... never as a block before the construction") is violated, and nothing in the rules orders two displays of one component by dependency (the display a later display consumes is motivated by that later one).

## Minor

4. F, notation (eq:obc_corr vs eq:obc_problem). The correction is written with u^n_k (recorded input), the problem display with hat u^n_k (closed-loop input of eq:aug_loss). Rule cause: README "Math standard" item 4 (all lines of one display, and a changed line, use one set of signals) does not cover a component display whose signal argument changes when substituted into the problem.

5. B, fixed target not marked (eq:obc_problem). xi-circ is held per epoch and receives no gradient; this is said in prose (260-274) but not in the display's "where" clause. Rule cause: README PS2 example ("a target computed from the current model and held fixed during the update is marked so in one where clause") is stated only inside the PS2 example, not as a general Math standard item.

6. D, guarantee sentence lacks its rank qualifier (line 256). "To first order, the stacked one-step map therefore determines xi" needs full column rank of the stack, which only Proposition 1 (ii) and the todo at 228 introduce. Rule cause: skill "Order" Method bullet ("every prose sentence that states the guarantee carries that qualifier") names local qualifiers only, not rank or other assumptions.

7. F, Proposition 1 on R_+ while the code projects over R (line 366-368). The restricted set is used but not named as a variant of the trained construction. Rule cause: skill "Order" ("names the variant it concerns ... as such") exists; failed to prevent.

8. C/E, restatements of earlier sections. IV-A opens (98-101) with the non-uniqueness the roadmap and Sec. III-D already state; eq:obc_tuples line 1 rewrites eq:aug_norm's T_x(col(q, qdot) - mu_x); the selection paragraph (318-323) restates that theta_aux is recomputed, which follows from it being a function of theta_aug. Rule cause: skill "Say each fact once"; README "Prose around the math" ("an implementation fallback stays out") does not cover bookkeeping such as recompute-before-validation and stack-at-restore.

9. D, paraphrase of an inline formula (lines 134-138). Three sentences read back which combination acts through qddot, qdot, q after the formula shows it. Rule cause: skill "Equations carry the content" (paraphrase); writer note 11 shows the packing rule pushed the writer to split a list into sentences instead of deleting it.

10. E, scope paragraph (411-434) is twelve short sentences with an ambiguous "Their inner product" and an obscure item (cancellation of the nominal transition constrained only through its range component). Rule cause: skill "Order" scope rule ("only what an examiner would question ... each item once") failed to prevent; no rule limits the item count.

11. D, todo density: nine todos, three stacked directly after eq:obc_tuples. Each is legitimate (no source for records vs forward simulation, xbar = 0, stencil mismatch), but they break the argument where the reader needs it most. Rule cause: missing rule on merging related todos of one display into one (writer note 9).

12. Length about 2.9 pages against the 1.55 budget; the displays are justified, the excess is in items 2, 3, 8, 10. Rule cause: README "Paper structure and ownership" page budgets predate the Math standard (writer note 12).
