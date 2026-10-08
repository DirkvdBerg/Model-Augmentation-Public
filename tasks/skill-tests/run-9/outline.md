# Section III outline (run 9, pre-approved)

Opening (one paragraph): the baseline omits dynamics (Garcia Sec. 2.4 list, plus the Coulomb friction that (eom) leaves out); this section adds a learned dynamic component, estimated jointly with theta_base from closed-loop records; names III-A, III-B, III-C.

## III-A Augmented model (`sec:aug_structure`)
Job: define the model that is trained, and its state at initialisation.
- P1. Topic: an omitted mode needs added model order, which neither a refit of theta_base nor a static correction gives; additional states do. Eq. (aug_transition): S-DP form, generic input u_k, output h_base. Sources: hoekstra2025lfr Sec. 2, hoekstra2026lfr Table 1.
- P2. Topic: parallel, not series, because the orthogonal construction of Section IV needs an additive learned term (gyorok2025l4dc Sec. 2; research plan Aspect 2). Positioning against Drenth (discrete-time LPV-LFR augmentation, LFR matrices, Sec. 5.2): our baseline is continuous-time physics and the learned part is an unstructured network, so the S-DP form is used. No equation.
- P3. Topic: f_base is one RK4 step of (eom) with the input held; each stage evaluates M(Y) at its own state, so f_base stays self-scheduled; normalisation with training statistics (hoekstra2026lfr Sec. 5.3). No equation (RK4 stages are textbook; follows the approved reference example).
- P4. Topic: one tanh MLP supplies f_aug and g_aug. Eq. (aug_mlp) as the output split only (no layer formula). f_aug writes all six physical rows, as in the S-DP form where the learning component affects the entire baseline state (hoekstra2025lfr Sec. 2, hoekstra2026lfr Table 1). The network corrects the result of the RK4 step: discrete-time correction (the RK4 boundary).
- P5. Topic: h_base reads positions from the physical state only; acyclic graph, smooth f_base and tanh give well-posedness (hoekstra2026lfr Thm. 9, Conds. 6, 8). Additional states have no physical meaning, and are not the simulated absorber's states (Caution). Fig. aug_structure referenced here.
- P6. Topic: training starts from the baseline. Eq. (aug_zero_init). Consequence: split not unique (hoekstra2026lfr Sec. 5.2), pointer to Section IV.

## III-B Encoder and initial state (`sec:aug_encoder`)
Job: give each window an initial state from measured data.
- P1. Topic: only positions are measured, so the window state is estimated from the I/O history (beintema2023subnet). Eq. (aug_encoder), one history length n including the current sample, as in the reconstructability map (hoekstra2026encoder Eqs. (10) to (15)). Full state encoded, positions included.
- P2. Topic: the encoder is a linear map plus a nonlinear correction (hoekstra2026encoder Eq. (8)). Eq. (aug_enc_lin).
- P3. Topic: the physical rows start from the reconstructability map of the linearised baseline (hoekstra2026encoder Sec. 3.1, Eqs. (16), (17), (30) to (32)). Eq. (aug_lin) CT linearisation at rest at Y = 0 with the nominal parameters, ZOH as a phrase; Eq. (aug_wb). Observability from measured positions. The linearisation sets only the starting value.
- P4. Topic: W^a random, psi_tilde zero; with (aug_zero_init) the encoded additional states do not reach the output at initialisation.

## III-C Closed-loop rollout and objective (`sec:aug_closed_loop`)
Job: define how a window is simulated and what is minimised.
- P1. Topic: data and use case are closed loop, so the closed-loop response is fitted, not the open-loop simulation of hoekstra2026lfr Eq. (22).
- P2. Topic: second reason, the free integrators: A(Y) has two eigenvalues at 0, so open-loop position errors persist; the loop acts on them (D-139).
- P3. Topic: adopted from Kessels (5.12), (5.13c), (5.13d); controller kept outside the model so it can be replaced (Sec. 5.3.2.3).
- P4. Topic: both loops share r, u_ff and controller. Eq. (aug_direct_controller), controller realisation (A_c, B_c, C_c, D_c), forces mapped by P.
- P5. Topic: subtraction removes r and u_ff. Eq. (aug_controller). Exactness conditions; the model loop runs the controller at the model sample period (D-141), quantified in Setup. Fig. aug_closed_loop referenced.
- P6. Topic: x^c_tau = 0 gives the model's controller the recorded controller state, which Kessels reconstructs (Remark 5.4); no model error before tau enters the window; per-sample evaluation order (no feedthrough).
- P7. Topic: limitation (Caution): the loop suppresses model error inside its bandwidth, so a closed-loop fit alone does not show the open-loop plant is recovered (D-139).
- P8. Topic: the objective. Eq. (aug_loss). Truncated objective of hoekstra2026lfr Eq. (22) with closed-loop windows (kessels (5.12)); every sample scored as in both; joint estimation, controller fixed.
- P9. Topic: horizons kept separate: window spans the closed-loop memory (D-220); checkpoint selection on a closed-loop simulation over complete validation records from one encoder estimate (kessels (5.14)); values to Setup.

## Moves and removals
- Setup: n_x_a, network width and depth, encoder history n, window length and its sweep, stride, optimiser and L-BFGS polish, controller re-discretisation rate and its effect on the loop (D-141), W^a distribution, residual check of the re-discretised controller, sensitivity of the window to slower dynamics.
- Intro: the contribution paragraph (realisation on the gantry, identification under feedback, OBC as separate contribution).
- Results: evidence that open-loop position errors dominate; added-state contribution measured by zeroing x_a (D-180).
- Removed as textbook machinery: RK4 stage equations (old eq:aug_rk4), MLP layer formula, ZOH integral for B_d.
- Left out (history, Caution): earlier routing variants.
