# Outline: Section III, Dynamic LPV-LFR Augmentation (run 7, pre-approved)

Opening (one paragraph): the baseline omits effects Garcia-Herreros names (Sec. 2.4) plus Coulomb friction; we augment it with a learned dynamic component and estimate theta_base, theta_aug, eta jointly from closed-loop records; names III-A, III-B, III-C in order.

## III-A Augmented model
Job: define the model that is identified, with the RK4 boundary and well-posedness explicit.
- P1. Topic: an omitted vibration mode adds model order, which neither refitting theta_base nor a static correction can supply. Eq: S-DP transition plus output map (eq:aug_transition). Sources: hoekstra2025lfr Sec. 2, hoekstra2026lfr Table 1. Defines x-tilde, x-bar, x-hat, u = P u_act, h_base = [P^T 0]; routing to all six physical rows stated as the S-DP definition.
- P2. Topic: f_base is one RK4 step of (eom), so it stays self-scheduled. No display (textbook RK4; approved example sentence). Sources: hoekstra2026lfr Sec. 7.2 (RK4), Sec. 5.3 (normalisation). Consequence: network acts after the step (discrete-time correction); contrast with Drenth thesis Sec. 5.2, whose augmentation enters the baseline LFR.
- P3. Topic: one MLP supplies both learned functions. Eq: eq:aug_mlp (output split only, no layer formula). Points to Fig. aug_structure.
- P4. Topic: the augmented model is well posed. No display. Sources: hoekstra2026lfr Cond. 6, Cond. 8, Thm. 9; Sec. II-B for M(Y) > 0 at every Y. Also: x-bar reaches the output only through f_aug and is not the truth absorber (Caution).
- P5. Topic: training starts from the baseline. Eq: zero output layer (eq:aug_zero_init). Source: hoekstra2026lfr Eq. (29).
- P6. Topic: the split between f_aug and theta_base is not unique. No display. Source: hoekstra2026lfr Sec. 5.2, Eq. (26) not used (D-193); hand-off to Section IV.

## III-B Encoder and initial state
Job: give each window its initial state from data and initialise it from the baseline.
- P1. Topic: only positions are measured, so an encoder estimates the window's initial state. Eq: encoder with linear and nonlinear part (eq:aug_encoder). Sources: beintema2023subnet, hoekstra2026encoder Eq. (8), Eq. (15) for the current sample.
- P2. Topic: W^b is initialised from the linearised baseline. Eqs: linearisation at Y = 0 with M_0 (eq:aug_lin), reconstructability map (eq:aug_wb). Sources: hoekstra2026encoder Eqs. (16), (17), Sec. 3.1 (left inverse); contrast with hoekstra2026lfr Eq. (30). ZOH integral as a phrase.
- P3. Topic: W^a random, psi-tilde zero, so encoder plus model reproduce the baseline at initialisation; W^b is only a starting value.

## III-C Closed-loop rollout and objective
Job: state why and how each window is simulated inside the known loop, and the objective.
- P1. Topic: the model must predict the machine under feedback, so we fit the closed-loop response, not the open-loop simulation of hoekstra2026lfr Eq. (22) (research-plan departure).
- P2. Topic: the free axes give a second reason. Builds on K in (damping_stiffness); stability proviso cited to kessels2025ai Sec. 5.2.3.
- P3. Topic: we adopt Kessels' closed-loop truncated training with the controller outside the model. Sources: kessels2025ai Eqs. (5.12), (5.13c), (5.13d), Sec. 5.3.2.3; difference: theta_base joint.
- P4. Topic: subtracting the recorded loop gives the residual form. Eqs: both loops (eq:aug_direct_controller), residual form (eq:aug_controller). Own derivation. Exactness conditions; controller re-discretised at T_s is the one approximation (D-141).
- P5. Topic: x^c_tau = 0 is Kessels' controller initialisation without reconstruction; sample order without algebraic loop. Source: kessels2025ai Remark 5.4, D-142. Points to Fig. aug_closed_loop.
- P6. Topic: the objective. Eq: eq:aug_loss. Sources: hoekstra2026lfr Eq. (22), kessels2025ai Eq. (5.12); every sample scored as in both.
- P7. Topic: consequence of the loop for the window and for interpretation. Closed-loop memory bounds n_f (D-220); a small V does not by itself show open-loop plant recovery (Caution). Single Setup pointer.

## Moves and omissions
- Out: the RK4 stage display (textbook machinery; phrase plus citation), the MLP layer formula (phrase), the ZOH integral (phrase).
- Out to Introduction: the contribution sentences of the old opening (contributions are listed once, in Section I).
- To Setup: n_x_a, network width and depth, encoder history n, n_f, validation horizon, W^a distribution, controller re-discretisation rate and its residual check.
- Left out: earlier routing variants (Caution: superseded; development history), xc reset cost measurements (D-142), the burn-in alternative (not used; full-window scoring follows the cited objectives).
- Open points (reasons not traceable, figure and notation conflicts) go to the delivery report, not into the prose.
