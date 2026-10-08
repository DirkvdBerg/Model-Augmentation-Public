# Section III outline (run 11, pre-approved)

Opening (one paragraph): the rigid-body baseline omits effects named by Garcia-Herreros and the Coulomb friction; Section III adds learned dynamics and estimates all parameters from closed-loop records; names III-A, III-B, III-C.

## III-A Augmented model
Job: define the model and its boundary with the physics.
- P1. If omitted effects are dynamic, the six baseline states cannot represent them; refitting or a static correction adds no states, so the augmentation needs its own. Eq: dynamic-parallel transition with output (`eq:aug_transition`). Signals and frames: y = q_act, u = P u_act, normalised. Sources: Hoekstra 2025 Sec. 2, Hoekstra 2026 Table 1; Drenth Sec. 5.2 for positioning (LPV-LFR learning part vs MLP here).
- P2. f_base is one RK4 step of (eom), input held, self-scheduled; normalisation with training statistics. No display (RK4 stages are textbook machinery). Source: Hoekstra 2026 Sec. 5.3.
- P3. One MLP gives f_aug and g_aug. Eq: output split (`eq:aug_mlp`), layer formula dropped. Routing to all six physical rows, reason: the learning component acts on the entire state (Hoekstra 2025 Sec. 2) and X/Y authority (D-103). RK4 boundary: f_aug is a discrete-time correction. Fig. `fig:aug_structure` referenced here.
- P4. Output map not learned (Kessels Remark 5.3); acyclic graph and smooth parts give well-posedness (Hoekstra 2026 Thm. 9, Cond. 6, 8). Additional states defined only up to a state transformation: no assigned physical meaning (D-180; covers the absorber caution in the general case).
- P5. Zero output layer reproduces the baseline at the start. Eq: `eq:aug_zero_init`. Source: Hoekstra 2026 Eq. (29); split non-unique (Sec. 5.2), pointer to Section IV.

## III-B Encoder and initial state
Job: give each window its initial state.
- P1. Only positions are measured, so an encoder estimates the state from a history ending at the current sample. Eq: `eq:aug_encoder` with one lag n. Source: Beintema 2023; encoder paper Eq. (15) for the current sample.
- P2. Linear map plus nonlinear correction. Eq: `eq:aug_enc_lin`. Source: encoder paper Eq. (8).
- P3. W^b from the linearised baseline at nominal parameters, rest, Y = 0; ZOH as a phrase. Eq: `eq:aug_lin` (A_c(Y), B_c(Y) only, reused in III-C). Then `eq:aug_wb` (encoder paper Eqs. 16, 17); observability gives a left inverse; the linearisation only sets a start.
- P4. W^a random, nonlinear correction zero; encoded added states inactive at start by (`eq:aug_zero_init`).

## III-C Closed-loop rollout and objective
Job: state what is fitted and how each window is simulated.
- P1. Records and intended use are closed loop (Introduction), so the closed-loop response is fitted, departing from the open-loop simulation of Hoekstra 2026 Eq. (22).
- P2. Second reason: X and Y have no stiffness, two eigenvalues at the origin, velocity errors persist as position errors in open loop; the loop acts on them. Limitation: feedback also attenuates model error, so closed-loop fit does not show open-loop accuracy (Caution).
- P3. Kessels' closed-loop truncated rollout with gradients through the controller; controller outside the model so it can be replaced (Kessels (5.12), (5.13c), (5.13d), Sec. 5.3.2.3).
- P4. Recorded and model loops share r, u_ff and controller. Eq: `eq:aug_direct_controller`; subtracting gives the residual form, exact under same LTI controller, same feedforward, no saturation. Eq: `eq:aug_controller`. Fig. `fig:aug_closed_loop`.
- P5. x^c_tau = 0 starts the model controller in the recorded controller state; Kessels reconstructs it instead (Remark 5.4); no pre-window model error enters (D-140, D-142).
- P6. Fixed per-sample order because the model has no feedthrough.
- P7. Objective. Eq: `eq:aug_loss`; Hoekstra 2026 Eq. (22) with closed-loop windows as in Kessels (5.12); every sample scored as in both; joint estimation with fixed controller; one pointer to Section V for n_xa, network size, n, n_f and the free-run horizon (kept as four separate quantities).

## Moves and omissions
- To Introduction (already there): the contribution sentence and the separate-contribution pointer to OBC.
- To Setup: all values (n_xa, 16 x 2 network, lag 29, n_f = 0.1 s and its closed-loop-memory bound D-220, 4 kHz, validation free-run horizon and checkpoint rule), W^a distribution (Kaiming, deviation from Xavier), state-normalisation source, the controller residual check todo, the horizon sensitivity checks.
- To Results: evidence that open-loop position errors dominate; the closed-loop masking check.
- Left out: RK4 stage equations, ZOH integral and MLP layer formula (textbook machinery); earlier routing variants and burn-in history (development history; Caution "earlier routing plans superseded").
- Absorber schematic stays in Setup (Fig. `fig:gantry_absorber`), where the absorber is introduced.
