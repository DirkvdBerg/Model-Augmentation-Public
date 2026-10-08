# Outline: Section III, Dynamic LPV-LFR Augmentation (skill test run 2)

Roadmap (one line): the baseline misses dynamics that need extra model order; we define the
augmented model (III-A), the encoder that supplies each window's initial state (III-B), the
closed-loop rollout and objective (III-C), and the initialisation from the baseline (III-D).

Notation decision: u keeps its Section II meaning (generalised force P u_act). The controller
produces actuator forces u_act; the model receives u = P u_act. Output y = measured positions
q_act; output map h_base, as in Fig. aug_structure. Controller matrices (A_K, B_K, C_K, D_K).

## III intro (roadmap paragraph)
- Purpose: what the baseline omits (Garcia Sec. 2.4; Coulomb left out by us), what this section
  estimates jointly (theta_base, theta_aug, eta), then the four parts in order.
  Sources: garcia2013model Sec. 2.4, Section II.

## III-A Augmented model (job: define the model and the RK4 boundary)
- P1 Problem: an omitted mode needs extra order; refitting theta_base or a static correction
  cannot add it; additional states can. Remedy: S-DP of Hoekstra (Table 1), LPV precedent in
  Drenth. Display eq:aug_transition (S-DP with h_base). Where clause. Meaning.
  Sources: hoekstra2025lfr Sec. 2, hoekstra2026lfr Table 1, drenth2025thesis Sec. 5.2.
- P2 Baseline transition: one RK4 step of eq:eom over T_s, input held (no display, RK4 is
  standard); every stage evaluates M(Y) at its own state so the baseline stays self-scheduled;
  f_aug acts after the step, so it is a discrete-time correction; normalisation as in Hoekstra
  Sec. 5.3 changes coordinates, not dynamics. Sources: hoekstra2026lfr Sec. 5.3, code
  (Gantry block, up_sample=1 is a Setup value).
- P3 Learned functions: one tanh MLP supplies both (display eq:aug_mlp, no layer formula);
  routing to all six physical rows: missing dynamics couple into X and Y (D-103), and a force
  held over one sample moves the position within the sample, so a discrete-time correction
  needs position rows too (own derivation from the ZOH input map). Figure ref fig:aug_structure.
  Sources: D-103, D-222, model.py.
- P4 Output and well-posedness: h_base = [P^T 0] fixed, h_aug = 0; acyclic graph plus smooth
  F and tanh gives well-posedness (Thm. 9); additional states reach the output only through
  f_aug, carry no assigned physical meaning and are not the states of the simulated absorber.
  Sources: hoekstra2026lfr Cond. 6, 8, Thm. 9; D-180.

## III-B Encoder (job: initial state of each window)
- P1 Problem: each window needs x_hat at its start; only positions are measured; encoder as in
  SUBNET; display eq:aug_encoder merged with the linear-plus-correction form (encoder paper
  Eq. 8); histories include the current sample as in the reconstructability map; the encoder
  estimates every state, the positions included. Sources: beintema2023subnet,
  hoekstra2026encoder Eq. 8, Eqs. 6 and 15.

## III-C Closed-loop rollout and objective (job: the training criterion)
- P1 Problem: data under feedback, use is prediction under feedback with a known controller;
  open-loop simulation from the recorded input (Hoekstra Eq. 22) drifts on X and Y because K has
  zero X and Y rows. Remedy: Kessels' closed-loop truncated rollout with gradients through the
  controller, controller kept outside the learned model so it can be replaced.
  Sources: Section II eq:damping_stiffness_matrices, kessels2025ai Eqs. (5.12), (5.13c), (5.13d),
  Sec. 5.3.2.3.
- P2 Model loop display (one line pair, Kessels form with r, u_ff); recorded loop has the same
  form with y, s, u; subtraction gives the residual form, display eq:aug_controller; exactness
  conditions. Sources: kessels2025ai (5.13c,d); own derivation; closed_loop.py.
- P3 Window start and sample order: x^c_tau = 0 means s_hat = s at the start, without
  reconstruction (contrast Remark 5.4); evaluation order; Fig. aug_closed_loop; limitation:
  the controller attenuates model error, so a closed-loop fit is not proof of an accurate plant.
- P4 Objective: display eq:aug_loss with T_y weighting (normalised outputs); truncated
  objective of Hoekstra Eq. (22) with closed-loop windows; joint estimation in the coordinates of
  Section II-C, controller fixed; no parameter penalty (Eq. 26), Section IV replaces it; every
  sample scored; four horizons named, values in Setup. Sources: hoekstra2026lfr Eqs. (22), (26),
  TRAINING-DESIGN.md, D-193, interconnect.py loss.

## III-D Initialisation from the baseline (job: start at baseline behaviour)
- P1 Problem: random network initialisation can destabilise the model (Hoekstra Sec. 5.4;
  in closed loop, Kessels Sec. 5.2.3); remedy: zero output layer gives f_aug = g_aug = 0, so the
  model reproduces the baseline (Eq. 29); the split is not unique from this start (Sec. 5.2),
  Section IV constrains it. Sources: hoekstra2026lfr Sec. 5.4, Eq. (29), Sec. 5.2; kessels2025ai
  Sec. 5.2.3; D-233, D-237.
- P2 W^b: linearise eq:eom at rest at Y = 0 (display A_c, B_c; ZOH as a phrase), reconstructability
  map (encoder paper Eqs. 16, 17, no display), observability from the positions, local
  linearisation only sets the start (encoder paper Sec. 3.1). Sources: hoekstra2026encoder,
  gantry_linearization.py, pre_encoder.py.
- P3 W^a random, psi_tilde zero; at initialisation the encoded additional states do not reach the
  output. Sources: pre_encoder.py, D-233.

## Moved or left out
- RK4 stage display: replaced by a phrase (standard machinery, style profile Sec. 10).
- MLP layer formula: replaced by a phrase.
- Reconstructability map display (W^b formula): borrowed unchanged, pointer to Eqs. 16, 17.
- ZOH integral: phrase.
- Recorded-loop controller equations: one sentence instead of a second display line.
- To Setup: added-state count, network size, encoder lag, window length, stride, rate (4 kHz,
  re-discretised controller bias D-141), W^a distribution, residual check of the controller.
- Left out (development history or mechanics): gradient flow into the added states after zero
  init, Kaiming versus Xavier, D-142 wording dispute.
