# Outline: Section III, Dynamic LPV-LFR Augmentation (run 13, pre-approved)

Opening (one paragraph): the section delivers Introduction contribution 2 (dynamic-parallel augmentation extended to a self-scheduled, closed-loop setting, with theta_base estimated jointly) and names III-A model, III-B encoder, III-C closed-loop objective; Section IV constrains the split.

## III-A Augmented model
Job: define the model that is identified, and where the physics ends and the network begins.

1. Topic: the baseline omits effects of the machine, and dynamic omitted effects need states of their own.
   - Garcia Sec. 2.4 neglected effects; Coulomb omitted from (eom) although Garcia Sec. 2.2 has it.
   - General argument (no specific mode): refitting theta_base or a static correction adds no states; dynamic augmentation does (EJC Sec. 2).
   - Equation: S-DP model (aug_transition), generic input u_k, output h_base = C_n. Adopted: EJC Table 1, 2026 Table 1.
   - Reason for parallel (research plan Aspect 2): OBC of Section IV is formulated for additive augmentation (Gyorok 2026 Sec. 1 to 2).
2. Topic: f_base is one RK4 step of (eom), self-scheduled within the step; f_aug enters after the step (RK4 boundary). This work's.
   - No RK4 display (textbook); input held over the step; normalisation from training records (2026 Sec. 5.3, adapted: state statistics from P^-T y and differences, not a baseline simulation).
   - Positioning, adapted: Drenth Sec. 5.2 augments a DT LPV-LFR baseline with LPV-LFR learning blocks; here an unrestricted network acts on a CT self-scheduled baseline.
3. Topic: one tanh MLP supplies f_aug and g_aug (2026 Eq. 14).
   - Equation: output stacking (aug_mlp); layer formula as phrase.
   - Routing: all six physical rows, as the S-DP f_aug acts on the whole baseline state (EJC Sec. 2); X and Y rows kept because the correction must reach the coupled X and Y response (D-103). g_aug is the next added state itself (D-180).
4. Topic: output map and well-posedness. h_aug = 0; no learned function feeds another (D_zw = 0) so the graph is acyclic; with C2 functions the model is well posed (2026 Thm. 9, Remark 10). Added states have no assigned physical meaning and are judged at the output (Caution).
5. Topic: training starts from the baseline (2026 Eq. 29).
   - Equation: zero output layer (aug_zero_init). Adapted: whole output layer zero, no random linear bypass as in 2026 Sec. 5.4.3; follows the published S-DP code (D-233, D-237); the added states start inactive.
   - Consequence: closed-loop start is the baseline loop (Kessels Sec. 5.3.2.2); split not unique (2026 Sec. 5.2); Section IV.
   - Figure: fig:aug_structure referenced at the model equation.

## III-B Encoder and initial state
Job: give each window an initial state when only positions are measured.

1. Topic: encoder estimates the window's initial state from I/O history (Beintema Sec. 3, adopted). Equation (aug_encoder); history includes sample tau, as the reconstructability map (encoder paper Eq. 15) uses.
2. Topic: linear map plus nonlinear correction (encoder paper Eq. 8). Equation (aug_enc_lin).
3. Topic: W^b from the baseline, linearised at rest (encoder paper Eqs. 30 to 31, adapted: analytic W^b instead of the 2026 Eq. 30 fit).
   - Equation: A_c, B_c at Y = 0, C_d (aug_lin); ZOH as phrase.
   - Equation: W^b reconstructability map (aug_wb), encoder paper Eqs. 16 to 17; observability gives a left inverse; the linearisation only sets the start.
4. Topic: W^a random (Kaiming, adapted from Xavier of 2026 Sec. 5.4.2), psi-tilde zero; by the zero output layer the encoded added states do not affect the output at the start.

## III-C Closed-loop rollout and objective
Job: define what is fitted, and why it is fitted in closed loop.

1. Topic: records and use are under feedback, so we fit the closed-loop response (departure from open-loop Eq. 22 of 2026). Adopted from Kessels Eqs. 5.12, 5.13c, 5.13d; controller outside the model so it can be replaced (Kessels Sec. 5.3.2.3, D-140).
2. Topic: the baseline gives a second reason: free integrators on X and Y (K of Sec. II); open-loop error accumulates, the loop acts on it (D-139, D-142). Limit (Caution): feedback also suppresses model error, so closed-loop fit does not prove an accurate open-loop plant (Kessels Sec. 5.3.2.3); Results test this.
3. Topic: residual form (this work's realisation). Equations: recorded and model loops (aug_direct_controller), difference (aug_controller); condition for exactness. Figure fig:aug_closed_loop.
4. Topic: x^c = 0 gives s-hat = s without reconstruction (adapted: Kessels Remark 5.4 reconstructs); step order from zero feedthrough.
5. Topic: objective (aug_loss) = 2026 Eq. 22 with each window in closed loop (Kessels Eq. 5.12); joint estimation in the Sec. II-C coordinates; every sample scored as in both; window spans closed-loop memory (D-220); validation over whole records (Kessels Eq. 5.14). One pointer to Setup for all values.

## Moved or left out
- RK4 stage display: textbook machinery, phrase instead.
- MLP layer formula and ZOH integral: textbook, phrase.
- All values (n_xa, width, n, n_f, optimiser, rates, controller K1, absorber, window and validation values): Setup.
- Absorber schematic: stays in Setup (the omitted dynamics are introduced there).
- All todos and VERIFY notes: removed from the prose, listed as open points in the delivery report.
- Implementation detail (network and encoder receive actuator forces in code; P absorbed by the first layer): left out, open point.
