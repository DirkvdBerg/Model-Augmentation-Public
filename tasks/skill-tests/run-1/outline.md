# Section III outline (thesis-section skill, test run 1, pre-approved)

Roadmap (one line): the baseline omits effects that need extra model order; we add a learned dynamic-parallel component and estimate it jointly with the ten physical combinations from closed-loop records; III-A the augmented model, III-B the initial state of each window, III-C the closed-loop rollout and objective.

## Opening paragraph
- Purpose: what the baseline leaves out (Garcia Sec. 2.4 effects, plus the Coulomb friction our baseline drops), what is estimated (theta_base, theta_aug, eta), the roadmap. No attribution row, no contribution list (Introduction owns it). Sources: garcia2013model Sec. 2.4; Sec. II.

## III-A Augmented model (job: define the model structure and its starting point)
- P1 Problem and structure: missing modes need order; refit and static correction cannot add it, additional states can; dynamic-parallel structure; Drenth's LPV-LFR precedent; signal frames (u = P u_act generalised force, y = q_act). Eq. `eq:aug_transition`. Sources: hoekstra2025lfr Sec. 2; hoekstra2026lfr Table 1; drenth2025thesis Sec. 5.2.
- P2 Baseline transition: one RK4 step of eq:eom with the input held (phrase, no display); every stage evaluates M(Y) at its own Y, so the baseline stays self-scheduled in the step; normalised coordinates (Hoekstra Sec. 5.3, Eq. 28), state scale from measured positions. No equation. Sources: hoekstra2026lfr Sec. 5.3; code `data.py::compute_normalization`.
- P3 Learned functions: one tanh MLP supplies both f_aug and g_aug. Eq. `eq:aug_mlp` reduced to the one-line output split (no layer formula). Routing: all physical rows, reason = X/Y coupling (D-103). Discrete-time correction on the RK4 result. Sources: D-180, D-103, D-222, model.py.
- P4 Output map and well-posedness: C_n = [P^T 0], output not augmented (Kessels Remark 5.3), acyclic graph plus C2 functions gives well-posedness (Thm. 9); added states reach the output only through f_aug. No new equation.
- P5 Initialisation: zero output layer, Eq. `eq:aug_zero_init`; baseline behaviour (Eq. 29); added states inactive at start, activated by training the output layer; non-unique split leads to Sec. IV. Sources: hoekstra2026lfr Eq. 29, Sec. 5.2; D-233, D-237.

## III-B Encoder and initial state (job: how each window gets its initial state)
- P1 Problem: only positions measured; per-window free initial states would multiply the parameters; encoder from history (Beintema). Eq. `eq:aug_encoder` (one history length n, current sample included because the reconstructability map uses it). Encoder estimates positions too (fact only). Sources: hoekstra2026encoder Sec. 1, Eq. 15; beintema2023subnet.
- P2 Encoder structure and W^b: Eq. `eq:aug_enc_lin` (Eq. 8); linearisation at rest at Y = 0, an equilibrium (encoder paper Eqs. 30, 31), Eq. `eq:aug_lin` with ZOH in prose; Eq. `eq:aug_wb` (Eqs. 16, 17); observability gives the left inverse; linearisation only sets a starting value.
- P3 Added-state rows: W^a random, nonlinear part zero; inactive at start by Eq. aug_zero_init. Sources: pre_encoder.py, D-233.

## III-C Closed-loop rollout and objective (job: what each window simulates and what is minimised)
- P1 Problem: records and use are closed loop; open-loop simulation of the recorded input fits a different object, and the free integrators on X and Y keep any position error. Sources: hoekstra2026lfr Eq. 22; eq:damping_stiffness_matrices, eq:aug_lin.
- P2 Remedy: Kessels' closed-loop truncated rollout with the known controller and gradients through it; controller kept outside the learned model so it can be swapped in evaluation. Sources: kessels2025ai Eqs. 5.12, 5.13c, 5.13d, Sec. 5.3.2.3.
- P3 Recorded and model loops, then the residual form (own derivation). Eqs. `eq:aug_direct_controller`, `eq:aug_controller`; exactness conditions; direct form needed for a changed controller. Figure `fig:aug_closed_loop`.
- P4 Controller initial condition and step order: x^c_tau = 0 means the model controller starts in the recorded controller's state, no reconstruction (contrast Kessels Remark 5.4); unique value giving zero correction for an exact model; evaluation order without feedthrough. Source: closed_loop.py.
- P5 Objective: Eq. `eq:aug_loss` on normalised outputs, scored over the full window; Hoekstra Eq. 22 with closed-loop windows (Kessels 5.12); joint estimation, controller fixed; four horizons named separately (encoder history, window, validation free run), values in Setup.

## Moved or left out
- RK4 stage display `eq:aug_rk4`: removed (standard; style profile Sec. 10).
- MLP layer formula: removed (standard), the output split kept.
- Contribution sentences and attribution row in the opening: Introduction.
- Controller residual check (rediscretised controller on recorded r - y), rediscretisation bias at 4 kHz (D-141), window length, encoder lag, network size, added-state count, sample rate: Setup.
- Kessels Remark 5.6 noise-bias argument: left out (not measured on the thesis data); open point.
- Burn-in development history: left out (final method scores the full window).
