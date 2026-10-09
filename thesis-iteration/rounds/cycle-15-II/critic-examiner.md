# Critic (examiner lens, Maarten Schoukens), cycle 15, Section II

Score: 7 / 10

Overall: the section reads as one argument. The roadmap is short and in Hoekstra's form, each subsection has one job, and each opens with the need. The math shows what is built on the gantry: coordinates and frames (eq:Ptransform), EOM with matrices, the scheduling block, G with u = P u_act and q_act = P^T q inside the display, the exact well-posedness determinant, theta_base, the admissibility condition and the frozen model. Every display has a where clause with spaces. No textbook machinery is displayed. The algebra I rechecked holds: P and det P = -L_b; the Schur complement gives det M(Y)/det M0; twelve entries with three equal to m_h; and the min over Y of det of the payload-reduced M(Y) is ((m_Sigma+m_h)/m_Sigma)(m_Sigma J_eff - L_b^2 m_Delta^2/4), so eq:lfr_admissibility is iff. No em-dashes. What keeps the score at 7 is that Section II-B's reason for existing is circular (A). Smaller faults remain in prose and correctness (C to F).

## Issues

1. MAJOR (A, coherence/justification), Section II-B, first paragraph and the implementation paragraph (draft lines 261-263, 360-372).
   The need is stated as "a form of eq:eom whose well-posedness can be checked for every payload position". II-A already states M(Y) > 0 for every Y. II-B's own result is that the LFR is well posed "exactly where M(Y) is nonsingular, which holds for every Y by Section II-A". The implementation then bypasses the loop through M(Y)^{-1}. Section III-A also states M(Y) invertibility directly in its where clause (03 line 237). An examiner reading once asks what the LFR adds. As written, it adds nothing the reader did not have after II-A. The real need sits in the source map: D-017 (supervisor: baseline in LPV-LFR form), research-plan Aspect 1 (explicit position dependence), and Hoekstra's interconnection taking the baseline as a block. That need is only hinted at through the hoekstra2025lfr citation.
   Rule cause: README "Source map per subsection", row II-B ("the result it yields is eq:wellposed, which III-A cites"). Also SKILL "Order", Method bullet ("'still for' names a display or result of this thesis that needs the construction (a condition a later section cites)"). Together they license justifying a construction by a result that only restates an earlier statement. Missing rule: the need in a subsection's first sentence must be one that no earlier sentence of the section already meets.

2. MINOR (C/E, reader test), Section II-B, implementation paragraph (lines 363-369).
   "It then evaluates the first block row of G at w = Delta(Y) col(qdd, Y qdd)" describes how the code computes a model that is the same either way. This is implementation detail. The paragraph has four sentences where the rule allows one equivalence sentence plus what the construction is still for.
   Rule cause: SKILL "Content", reader test ("how the code stores a signal, when the model is the same either way"). This is not enforced against the SKILL "Order" Method bullet, which invites an implementation paragraph.

3. MINOR (F, correctness of cited object), Section II-C, second paragraph (lines 398-401).
   "Up to symmetry, eq:mass_matrix and eq:damping_stiffness_matrices have twelve nonzero entries" is false for the cited displays: M(Y), C and K have ten nonzero entries up to symmetry. The count of twelve holds for M0, M1, M2 of eq:Msplit, C and K, as the next sentence's M_{1,12} and M_{2,22} show.
   Rule cause: SKILL "Workflow" step 4 ("checked ... against the exact object the sentence names (the equation it cites)"). The rule exists but was not applied. No rule failed here.

4. MINOR (F, attribution), Section II-B, line 317: "We keep the LFR in continuous time, as Drenth does [drenth2025thesis Sec. 2.1]".
   Sec. 2.1 gives the CT LPV-LFR form as background. Drenth's identification and augmentation are discrete time (citation-log row 76; the paper's Eq. 6 is DT). The citation supports "the CT LPV-LFR form of [X]", not "as Drenth does". The reason given (theta keeps the structure of M, C, K; D-018, D-020) is this work's choice.
   Rule cause: SKILL "Contributions" and the review check "Every 'as in', 'following' and 'as [X] do' is checked against the cited display". The rule exists. The README source-map row II-B ("cite the CT form from drenth2025thesis") primes the "as Drenth does" wording because it does not say that Drenth works in DT.

5. MINOR (D, justification of a choice), Section II-A, lines 195-198 (Coulomb omission).
   The sentence gives the consequence ("an effect the baseline lacks by construction") but not the reason D-204 records: the learned part is tested on an unmodelled effect. This is the departure from Garcia's Eq. (6) that an examiner would question first.
   Rule cause: README source-map row II-A states the reason but tells the writer to give only "the omission with the section's one pointer". SKILL "Justify only what an examiner would question" is overridden by that row. The row should state that the reason clause travels with the pointer.

6. MINOR (B, completeness of a compared model), Section II-D (lines 446-463).
   M_LPV and M_LTI do not state their initial state or whether they run open loop or in the closed loop. This is correctly carried by a \todo with a candidate. Until it is resolved, the reader cannot tell what "the size of the position-dependent prediction error" is measured on.
   Rule cause: none failed (SKILL "Order", Method: "else a \todo"). This is an open decision, not a writing fault.

7. MINOR (E, packing), Section II-C, lines 418-421: "Four directions of theta leave the model unchanged, ..., so estimates are reported in theta_base."
   The sentence is about 35 words and holds three facts (four lost directions, the list, the reporting consequence).
   Rule cause: SKILL "Workflow" step 4 packing pass. The rule exists but was not applied. The MUST ESTABLISH item "Four lost directions in one sentence" pushed toward one sentence. The SKILL says a block fixes claims, not sentences, so the packing rule should have won.

8. MINOR (E, read-back), Section II-B, line 274-275 ("It enters through M(Y), which carries the load distribution over the actuators and the yaw inertia") and line 459-460 ("so that Delta(Y_op) = Y_op I6 is constant").
   Both read the display back. The first restates eq:mass_matrix, and the second restates what freezing Y in eq:lti means.
   Rule cause: SKILL "Equations carry the content" (read-back sentence deleted). The rule exists but was not applied.

9. MINOR (F, convention), Section II-C, lines 402-404: "Their Jacobian ... has determinant m_h L_b^2/2".
   The order of the ten functions is not stated, so the sign depends on an unstated convention. Only nonzero is needed.
   Rule cause: README "Derivation policy" item 7 (magnitude when the sign depends on an unstated order). This is not in the SKILL review checklist, so the check was missed.

## Not issues (checked)
- Length (about 2.0 pages against 1.35) is caused by required displays and is reported in the header with a budget proposal.
- The well-posedness "if" in line 347, beside the "exactly where" consequence, is acceptable.
- The forward pointer to Section III-D for the training coordinates is agreed (MUST ESTABLISH II-C item 5).
- The roadmap \todos (contribution wording, terminology) follow SKILL "Contributions" and "The header".
