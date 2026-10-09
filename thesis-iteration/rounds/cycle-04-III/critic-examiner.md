# Critic (examiner lens, Maarten Schoukens), cycle 04, Section III

Score: 7 / 10

## Overall
Read once as an examiner, the section is one argument: a short roadmap, four subsections with one job each
(model, initial state, identification problem, added-state initialisation), each opened from its problem.
Every part is displayed with its spaces: the S-DP transition with domains and codomains, the normalisation
with the stage/actuator frame visible (Hoekstra 2026 Eq. 28 pattern), the network, the zero output layer
(Eq. 29 pattern), the encoder with W^b from the reconstructability map, both controller loops and the
residual loop, the criterion, the admissible parameter coordinates, and the S1 and S2 objectives. What is
trained (vartheta = (xi, theta_aug, theta_enc)), what is fixed (controller), the start and the cost are all
recoverable, and M_U and M_PS2 are named where their equations are. Gantry specifics (free X and Y axes,
equilibrium at Theta = 0, M(Y) per RK4 stage, admissibility by construction) are presented as this work's.
Citations checked against citation-log.md all support their sentences. Compiles (no "!" lines).

What keeps it from 8 or higher: the identification problem is not one self-contained display; III-D is
about 2.5 times the agreed PS2 sketch and carries optimiser schedule, calibration and gradient detail the
README excludes; the PS2 motivation states only the zero-gradient gate, which opens after one update, so
the need for PS2 is not argued; and most realisation choices in III-A and III-B are stated without reasons
(left to todos), which is exactly Maarten's "justify choices" point.

## Issues

1. MAJOR (B, A). eq:aug_loss is a cost display with the constraints given in prose by reference
   ("subject to (aug_transition) with u = u_hat, (aug_encoder) and (aug_loop)"), and xi is defined only
   after it ("the training coordinates of theta_base below", eq:mdelta_map). The reader must assemble the
   problem from four displays across two subsections. Hoekstra 2026 Eq. 22 (22a to 22e) and Drenth Eq. 19
   put model, encoder and initial state in the one display. Rule: README "Math standard" item 3 states it;
   the writer did not follow it, and Skill "Order" ("Known before new") failed to prevent the forward
   reference to xi.

2. MAJOR (A, D). III-D motivation: "At the zero start ... the gradient of V with respect to W^a and to the
   added-state rows of W_L is zero. We therefore initialise the added states from data." The gate holds only
   at the exact start; after the first update of the physical rows of W_L the gradient is nonzero
   (DOC/PS2-METHOD.md Sec. 2.2: "an initial gradient gate, not a proof that ordinary training can never
   activate the states"). An examiner asks at once why one update does not fix it. The documented reason
   (plain training learns static corrections or real lags and leaves the mode; evidence in Results) is
   missing, even as a pointer. Rule: README "Math standard", PS2 example, fixes the problem as "the zero
   output layer leaves the added states without influence", i.e. it prescribes the incomplete argument;
   Skill "Reasons and claims" (claim no stronger than its evidence) failed to catch it.

3. MAJOR (C, E). III-D runs to about 16 sentences plus an eight-item where-clause against the agreed
   "about two displays and six sentences". Content the sketch excludes: "one S1 step updates omega and
   theta^a_enc with a separate optimiser" after every Adam update (optimiser schedule), "windows of training
   records withheld from the fit" and pairs "from the withheld records" (calibration bookkeeping), "from the
   encoder state without gradient", "zero output layer" of d_omega and the theta^a_enc row selection
   (gradient mechanics), and "at the end of S1 or at an earlier screening update". Rule: README PS2 sketch
   ("Optimiser schedules, calibration bookkeeping and gradient mechanics stay out") was not followed; Skill
   "Length" calls the sentence count only a target, which weakens the check.

4. MAJOR (D). Most realisation choices are stated without a reason: h_aug = 0, correcting the position
   rows, an unrestricted network instead of an LPV-structured one (title still says LPV-LFR), centring and
   measured-position statistics instead of Hoekstra's baseline-simulation scaling, Y = 0 and nominal
   parameters for W^b, model-based rather than data-based W^b. Each is a todo. For h_aug = 0 a verified
   source exists (Kessels Remark 5.1 and 5.3, citation-log rows 68, 69) and could have been stated as an
   adopted reason. The choices paragraph in III-A also ends on a risk ("a persistent correction accumulates
   in position") that argues against the choice and is left unresolved. Rule: Skill "Content / Reasons and
   claims" ("A reason you derive yourself becomes an open point") turns every non-cited reason into a todo,
   so the rules systematically produce unjustified choices; Skill "Justify only what an examiner would
   question" is then unmet. Missing rule: when a verified source supports a candidate, state it in prose.

5. MINOR (F, notation). eq:mdelta_map uses rho in (0,1) as a margin, while Section II uses rho(D_zw) as the
   spectral radius quoted from Drenth Thm. 6. Also theta^0_base here versus theta_base,0 in the appendix
   (app:lfr-wellposed). Rule: Skill "Notation" ("An operator or symbol quoted from a cited result (a
   spectral radius ...) is checked like an introduced one") failed to prevent it.

6. MINOR (B, notation). eq:aug_norm normalises q_act (y^n = T_y(q_act - mu_y)) while the where-clause
   defines y as the recorded output and eq:aug_loss compares y^n with y_hat^n "normalised as in
   (aug_norm)"; which signal is scaled is ambiguous. "Its first difference" for the velocity statistics
   omits the division by T_s that a velocity scale needs. Rule: README "Math standard" item 4 ("A
   normalised signal gets its own symbol ... which input is scaled").

7. MINOR (D, F). Selection: "kept only if it lowers this score and no monitored quantity regresses".
   "Monitored quantity" is undefined, and the todo shows it includes the oracle combination error. A rule
   that defines the method must give its condition as the code states it. Rule: Skill "Content / reader
   test" exception (switching or acceptance rule: one sentence with its condition) not followed.

8. MINOR (E, register). "PS2" is never expanded in prose (only in a todo); "Kessels' direct form" two
   paragraphs after the indirect approach invites confusion with the direct closed-loop approach. Rule:
   Skill "Order" ("Known before new") and Skill "Notation" / README voice item 7 (stable terms).

9. MINOR (A). Two paragraphs carry two jobs: the last III-A paragraph mixes the added-state gauge with the
   baseline/network split and the interpretability definition; the III-C "residual form needs no controller
   initialisation" paragraph also carries the step order and algebraic-loop argument. The III-D contrasts
   with Hefny ("Unlike the first stage ...", "Unlike the second stage ...") are appended after displays.
   Rule: README "Prose around the math" ("never a run of sentences appended to a display").

10. MINOR (E). Draft build spans pages 6 to 10, roughly twice the 1.55-page budget in the submission build;
    the overrun comes mainly from III-D and III-C appended sentences (issues 3, 9). Rule: Skill "Length"
    (second deletion pass above 1.5 times the budget).
