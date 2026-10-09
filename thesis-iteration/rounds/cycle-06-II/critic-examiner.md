# Critic (examiner lens, Maarten Schoukens), cycle 06, Section II

Score: 7 / 10

Read once as an examiner: the section is one argument (roadmap, II-A model, II-B realisation, II-C identifiable combinations), the gantry math is displayed and correct, and the own claims hold when checked (P-transform, M(Y) split, det(I - Dzw Delta) = det M(Y)/det M0, twelve entries and three copies of m_h giving ten combinations, four lost directions, admissibility iff m_Sigma J_eff > (L_b^2/4) m_Delta^2 via the Schur complement, minimum at Y = -a/m_Sigma). Citations match citation-log rows. build/main.log has no "!" line. Length about 1.7 pages against 1.35 (writer note 12), under the 1.5x trigger.

What keeps it from 8 or higher: the examiner is not told what the LFR is for, and the last paragraph of II-B carries a second job.

## Issues

1. MAJOR (A, coherence). II-B motivates the LPV-LFR by the rational dependence of M(Y)^{-1} (lines 238 to 244), builds G and Delta, then its consequence paragraph (lines 317 to 323) says the implementation solves ddq = M(Y)^{-1}(u - C dq - K q) in closed form. Section III uses f_base as an RK4 step of eq:eom, not G (03_augmentation.tex line 217). The reader never learns what the realisation buys the thesis (exactness and well-posedness statement in the form of the Hoekstra/Drenth framework, the form the augmentation is positioned against, the first contribution). Maarten would ask "why build the LFR if you do not use it". No sentence and no todo. Rule cause: skill "Order" (problem, construction, consequence) is followed formally, but no rule requires a construction to state its role downstream when the implementation bypasses it; header "Choices to justify" lists "full six-channel LFR realization", and skill "Content", "The header" lets such a candidate be dropped silently by the reader test.

2. minor (A, one job per subsection). The last paragraph of II-B (lines 325 to 337) names M_LPV, points to discretisation in Section III, and introduces the frozen-LTI comparator with its own display eq:lti. Three points under one topic sentence, and the comparator is not part of "Self-scheduled LPV-LFR representation". Rule cause: README "Paper structure and ownership" gives II the models M_LPV, M_LTI but no rule says where a comparator model goes (writer note 1); skill "Order" ("One job per subsection", "Topic sentences") did not prevent it.

3. minor (B, math shows how it works). eq:lfr_G displays only the block pattern of G; A_x, B_w, B_u, C_z, D_zw, D_zu, C_y are left to the appendix (line 300). The two claims that follow, "its first block row is the state equation" and eq:wellposed, depend on D_zw = [-M0^{-1}M1, -M0^{-1}M2; I, 0], so the reader cannot check them from the main text, although the blocks are one line. Rule cause: README "Derivation policy" item 5 ("expanded coefficient matrices ... in an appendix") does not distinguish expanded entries from the compact block realisation that a main-text claim relies on.

4. minor (B, signals in and out). G takes the generalised force u and outputs q, while the drives act and are measured in the actuator frame (lines 134 to 136). The section never displays the model's measured output P^T q or input u = P u_act as the model's ports; Section III adds h_base. "The model predicts all three measured positions" (line 215) is therefore stated, not shown. Rule cause: README "Math standard" item 1 asks for spaces, not for the model's input and output to be stated in the measured signals; no rule requires the ports of a baseline to match the data frame in its own section.

5. minor (D, choices justified). Two header "Choices to justify" have no reason and no todo: the continuous-time formulation (Drenth et al. 2025, cited here, is discrete time; an SI examiner asks why CT) and the full six-channel Delta (only "We do not claim this realisation to be minimal"). The Caution about the SVD reduction is honoured only by silence. Rule cause: skill "Content", "The header" ("each passes the reader test or is left out") permits silent omission of a choice that differs from the cited method, in tension with skill "Justify only what an examiner would question".

6. minor (E, register). II-C opening paragraph (lines 351 to 359) splits one inference into six sentences and loses its referent: "Hence they give the same matrices" ("they" is two sentences back). Rule cause: skill "Rules that produce this style" and the packing pass say "split what remains into one fact per sentence" without a check that each split sentence keeps an explicit subject.

7. minor (E, deletion test). Garcia-Herreros is cited five times in II-A, twice in consecutive sentences with the same location (lines 155, 157 to 161). Rule cause: conflict between skill "Citations and values" ("Cite each borrowed component once, where it is used") and the review item "Every sentence that states a fact ... taken from a paper carries that citation itself".

8. minor (D, no paraphrase). The one consequence after eq:damping_stiffness_matrices, "it shifts the coupling between X and Theta and adds m_hY^2 to the yaw inertia" (lines 208 to 209), reads the matrix back, and II-B line 262 restates "Y enters only through Y ddq and Y^2 ddq". Rule cause: skill "Equations carry the content" / "Say each fact once" failed to prevent it; the "at most one consequence" slot has no test that the consequence adds something the display does not show.

## Not issues (checked)
- Open todos (contribution wording, dual-drive term, Coulomb conflict D-022 vs D-204, model notation, missing LTI runner, Gautier, m_Delta clash) are genuine open decisions in the required form.
- No training explanation and no values in the section; Y_op and the Coulomb reason correctly pointed to Section V; m_Delta map correctly left to III-C.
- No intermediate textbook derivation displayed; proofs in the appendix (C passes).
