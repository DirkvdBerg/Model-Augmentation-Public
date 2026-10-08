# Section 3 outline (run 3, pre-approved)

Roadmap (one line): the baseline omits effects the machine has; we add a dynamic parallel learned component and identify it jointly with the physical parameters from closed-loop records, in four parts: augmented model, encoder, closed-loop rollout, identification criterion.

## III (opening paragraph)
- P0. Problem and roadmap: Garcia neglects base resonance, cross-arm vibration, detent, friction variation; our baseline also drops Coulomb friction; this section adds a learned component and estimates theta_base, theta_aug, eta jointly from records measured under feedback; names A to D in order. No equation. Sources: garcia2013model Sec. 2.4, Section II.

## III-A Augmented model (`sec:aug_structure`). Job: define the model that is identified and why it has this form.
- P1. Missing order needs added states; static corrections and refitting cannot add order; S-DP structure with notation (u_act, y = q_act). Eq: `eq:aug_transition`. Sources: hoekstra2025lfr Sec. 2, hoekstra2026lfr Table 1, drenth2025thesis Sec. 5.2 (LPV-LFR precedent, augmentation itself an LPV-LFR).
- P2. f_base is one RK4 step of eq:eom, input held over the step (ZOH), each stage evaluates M(Y) at its own Y, so self-scheduled within the step; continuous physics vs discrete correction. No display (textbook RK4). Source: Gantry_State_Block, Section II.
- P3. Normalisation: signal scales differ by orders; zero mean, unit std from training records, velocities from first differences; baseline step evaluated in physical coordinates and mapped back; output map C_n, offsets cancel. Eq: `eq:aug_norm` (new). Sources: hoekstra2026lfr Sec. 5.3, data.py::compute_normalization, model.py.
- P4. One MLP supplies f_aug and g_aug; writes all six physical rows (absorber coupling reaches X and Y, D-103; EJC Sec. 2 learning components act on the whole state); risk: free-integrator rows accumulate corrections in open loop, which Section III-C returns to. Eq: `eq:aug_mlp` (split only, layer formula dropped). Sources: model.py, run entry `ann_route_ix=range(6+n_a)`, D-068, D-103.
- P5. Output map not learned (h_aug = 0); additional states reach output only through f_aug; acyclic graph plus C2 functions gives well-posedness; added states have no physical meaning, their dynamics are Jacobian blocks of phi_aug. No display. Sources: hoekstra2026lfr Cond. 6, 8, Thm. 9; D-180.
- P6. Zero output layer reproduces the baseline at initialisation (Eq. 29 with g_aug = 0); split not unique, Section IV constrains it; sizes in Setup. Eq: `eq:aug_zero_init`. Sources: hoekstra2026lfr Eq. (29), Sec. 5.2; D-237.

## III-B Encoder and initial state (`sec:aug_encoder`). Job: supply the state at each window start.
- P1. Only positions measured; encoder from I/O history including current sample (as in the reconstructability map); encodes all states including positions. Eq: `eq:aug_encoder`. Sources: beintema2023subnet, hoekstra2026encoder Eq. (15), model.py (na_right = 1).
- P2. Linear part plus nonlinear correction; physical rows from the baseline linearised at rest, Y = 0, nominal parameters, ZOH. Eqs: `eq:aug_enc_lin`, `eq:aug_lin` (ZOH integral to a phrase). Sources: hoekstra2026encoder Eq. (8), gantry_linearization.py.
- P3. W^b as the reconstructability map; full column rank from measured positions; linearisation only sets the start. Eq: `eq:aug_wb`. Source: hoekstra2026encoder Eqs. (10), (13) to (17), pre_encoder.py.
- P4. W^a random, psi-tilde zero output layer; encoded added states do not affect the output at initialisation. No display. Sources: pre_encoder.py, D-233.

## III-C Closed-loop rollout (`sec:aug_closed_loop`). Job: define how a window is simulated under the known controller.
- P1. Records and use case are closed loop; Hoekstra's criterion drives the recorded input; Kessels closes the known controller inside each window and backpropagates through it; we adopt it, controller outside the model so it can be replaced. Sources: hoekstra2026lfr Eq. (22), kessels2025ai Eqs. (5.12), (5.13c), (5.13d), Sec. 5.3.2.3.
- P2. Free integrators X, Y: open-loop position errors persist; in the loop the controller acts on them; risk that feedback also attenuates model error, so closed-loop fit alone is not plant evidence. Source: eq:damping_stiffness_matrices, Meeting audit.
- P3. Recorded and model loops with one LTI controller, error sign convention. Eq: `eq:aug_direct_controller`; Fig. `fig:aug_closed_loop`. Source: controller.py, D-221.
- P4. Subtracting gives the residual form, exactness conditions. Eq: `eq:aug_controller`. Sources: closed_loop.py, D-140, D-141.
- P5. x^c_tau = 0 means s-hat_tau = s_tau without reconstruction (contrast Kessels Remark 5.4); step order forced by no plant feedthrough and controller feedthrough, no delay. No display. Sources: closed_loop.py, kessels2025ai Remark 5.4.

## III-D Identification criterion (`sec:aug_objective`, new). Job: state what is minimised and over what.
- P1. Truncated output error over closed-loop windows, joint in theta_base, theta_aug, eta, controller fixed, no regularisation term toward nominal values. Eq: `eq:aug_loss`. Sources: hoekstra2026lfr Eqs. (22), (26), (27); kessels2025ai Eq. (5.12); D-193.
- P2. Every sample scored, so the encoded initial state stays in the objective; encoder history, window and validation horizon are distinct; window must span the closed-loop memory; values in Setup. No display. Sources: D-178 notes, D-220, config.

## Moved or left out
- To Setup: number of added states, network width and depth, encoder history n, window n_f, stride, RK4 substeps, model rate 4 kHz and controller re-discretisation (D-141), optimiser (Adam then L-BFGS), detuned start, the window bound from closed-loop memory (D-220 values).
- Left out: RK4 stage equations and MLP layer formula (textbook), Kaiming versus Xavier distribution of W^a (implementation detail, open point), the six-attribution intro paragraph (attributions moved to their subsections, contributions stay in the Introduction), D-142 versus code wording of x^c = 0 (open point).
