# Critic (checker lens), cycle 06, Section II

Score: 8 / 10

## What was verified
- Compile: `latexmk -pdf main.tex` in `thesis-iteration/Writing`, no line starting with `!` in `build/main.log`. All Section II citations resolve (the undefined keys are gyorok2026obc, forssell1999revisited, hefny2015supervised, downey2017psrnn, all outside II). One Section II warning: overfull hbox 8.6 pt at `eq:Ptransform` (draft line 150).
- Code (`SYS/gantry_ss.py`): M0, M1, M2, C, K entries, P, and the output map `Cd = [P^T 0]` match eq:mass_matrix, eq:damping_stiffness_matrices, eq:Ptransform. `FS/blocks.py::Gantry_State_Block._deriv_with`: closed-form N(Y)/d(Y) in Horner form, input u_act mapped by P; frozen `Y_op` branch exists. `Reduced_Gantry_State_Block`: COMBO_NAMES order equals eq:phi; admissibility condition and the strict rho-contracted region (D-229) match the last paragraph; the four flat directions match the docstring.
- Own math checked by algebra this session: P relations (X=(X1+X2)/2, Theta=(X1-X2)/Lb); dimensions of G (15x15); det(I6 - Dzw Delta) = det M(Y)/det M0 via the Schur complement; N(Y) and det M(Y) quadratic in Y; twelve nonzero entries, three equal to m_h, ten functions of rank ten; the mass-transfer lost direction (dm1 = dm2 = -delta, dmb = 2 delta, d(Jb+Jh) = Lb^2 delta / 2); eq:lfr_admissibility iff M(Y) > 0 for all Y given m_h, m_Sigma, J_eff > 0 (minimum of the Schur complement at Y = -b/m_Sigma).
- Citations against citation-log.md and the PDFs: garcia2013model Sec. 2.2, 2.3, Eqs. 6, 11, Table I; toth2010modeling Def. 7.2; drenth2025thesis Sec. 2.1 Eqs. 2.1, 2.4, Sec. 2.1.1, Eq. 2.9 (read: the LFR "is equivalent to a LPV-SS representation with rational dependency", supports the sentence); drenth2025lpvlfr Sec. 3.2 (read: "self-scheduled LPV-LFR models" with p = psi(x, u), supports the term); ovchinnikov2021computing Def. 7. All support their sentences.
- No em-dashes. Notation consistent with Section III (tilde x, u = P u_act, theta_base, h_base = P^T q). D-242 / THESIS-RESULTS step 1 support the frozen-LTI comparator (no training, true parameters).

## Issues (all minor)
1. **F/B, eq:lfr_G and M_LPV.** G's output row is q with C_y = [I 0], and M_LPV is named as eq:delta, eq:lfr_G, so the named compared model maps u to q. The code and Section III (h_base = P^T q, input u = P u_act) define the model from u_act to the measured q_act. The reader of II alone does not see that M_LPV is evaluated in the actuator frame. One clause naming q_act = P^T q as the model output (or the output row as P^T q) would close it.
   Rule cause: skill "Order by kind of section", Method: "Each model ends complete" asks for trained, fixed, start, cost, but not for the input and output signals of a named model in the frame the data use; README "Math standard" item 1 defines spaces, not frames.
2. **F, Coulomb pointer.** "for a data-design reason given in Section~V" points to a reason Section V does not state (05_setup.tex still describes the superseded Karnopp friction), and the todo candidate cites D-204 while the source-map row "V Data-generating system" lists D-204 under "not"; the current truth friction is the tanh law of D-224/D-226, which keeps friction in the truth only.
   Rule cause: README "Source map per subsection": row II-A cites D-204 as the governing reason while row V excludes it; the map's update rule did not catch the conflict.
3. **F/D, "positive entries of theta_base".** m_Delta is a signed entry of theta_base (nominal -0.5 kg in the code), so "positive entries of theta_base do not imply M(Y) > 0" states a premise that cannot hold; the next sentence's condition (m_h, m_Sigma, J_eff > 0) is the correct one.
   Rule cause: the header MUST ESTABLISH II-C item 4 wording ("Positive theta_base") was carried into prose; skill "2a" says a block fixes claims, not sentences, but no check asks to verify a block's wording against the code's sign conventions.
4. **E/D, paraphrase.** "it shifts the coupling between X and Theta and adds m_h Y^2 to the yaw inertia" restates eq:mass_matrix; II-B then restates "Y enters only through Y qdd and Y^2 qdd". Keep "The payload position enters only M(Y)" and delete the colon clause.
   Rule cause: skill "Equations carry the content" (do not paraphrase) and "Say each fact once" were not applied in the deletion pass.
5. **A/E, one job per subsection.** The last paragraph of II-B carries three jobs: naming M_LPV, a signpost "Section III-A discretises this continuous-time model" that is not on the topic of the paragraph, and the frozen-LTI comparator with its display. The LTI comparator is not part of the LPV-LFR realisation.
   Rule cause: README "Paper structure and ownership" gives M_LPV and M_LTI to Section II and the source map has a separate row "II Frozen-LTI baseline", but neither says where the comparator sits; skill "Order" "One job per subsection" and the topic-sentence check did not catch it.
6. **F, contribution wording.** "We do not claim this realisation to be minimal" stands beside the Introduction's "compact continuous-time LPV-LFR realisation"; the existing contribution todo does not name this tension.
   Rule cause: skill "Contributions" (opening claims what the Introduction states, else a todo) checks scope, not adjectives that the section then disclaims.
7. **E, pronoun and packing.** In II-C, "Hence they give the same matrices" refers back over an intervening sentence ("they" = the two parameter vectors). In II-B, "The implementation therefore solves the loop in closed form, ..., with M^{-1} = N/det M and both polynomials quadratic in Y" holds three facts and repeats the appendix.
   Rule cause: skill "Rules that produce this style" packing pass (three facts in one sentence) not applied to the implementation sentence; no rule covers ambiguous back-references.
8. **G, layout.** Overfull hbox of 8.6 pt at eq:Ptransform (two-column width).
   Rule cause: task compile rule checks only lines starting with "!"; no rule asks to check overfull boxes in the section's own lines.

## Not issues (checked)
- The where clause of eq:eom is long but is a where clause; the skill exempts it from the 30-word count.
- Two pointers to Section V (Coulomb reason, Y_op value): one is a reason pointer, one a values pointer; within the skill's "Reasons and claims" and "Expert register" rules.
- No identification problem or initialisation is owned by Section II, so criterion B asks only for the realisation, its signals and the comparator; these are displayed.
