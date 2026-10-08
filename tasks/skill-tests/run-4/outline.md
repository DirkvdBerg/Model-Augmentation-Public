# Outline: Section III, Dynamic LPV-LFR Augmentation (run 4, pre-approved)

Roadmap (one line): the baseline omits dynamics that need extra model order; III-A defines the
dynamic-parallel augmented model, III-B closes the known controller around it, III-C states the
identification criterion with the encoder, III-D the initialisation that starts training from the
baseline.

## III-A Augmented model (`sec:aug_structure`, label kept: Section II refers to it)
Job: define the model that is trained, with every signal's frame and the RK4 boundary.
1. Problem and structure: missing vibration order cannot come from refitting theta_base or a static
   correction; additional states can; dynamic-parallel model. Eq. `eq:aug_transition`.
   Sources: Garcia Sec. 2.4 via roadmap, Hoekstra EJC Sec. 2, Hoekstra 2026 Table 1.
2. Relation to the research plan and to Drenth: plan proposed dynamic parallel and LPV-LFR
   embedding; Drenth keeps the augmented model in LPV-LFR form (Sec. 5.2); we keep the baseline LFR
   exact and put an unrestricted network in parallel to its discretised step. Source: research plan
   Aspect 2, Drenth thesis Sec. 5.2.
3. Baseline step and normalisation: one RK4 step with held input (phrase, no display), M(Y)
   re-evaluated at each stage; normalised transition displayed. Eq. `eq:aug_fbase` (new).
   Sources: Hoekstra 2026 Sec. 5.3, code (Gantry_State_Block, data.py via D-129).
4. Learned functions: one network, both outputs; all physical rows written; reason (coupling into X
   and Y, D-103) and the alternative it rules out (rows with stiffness only); free-integrator risk,
   handled in III-B. Eq. `eq:aug_mlp` (layer formula removed). Sources: Hoekstra 2026 Eq. (22d),
   EJC Sec. 2, D-103, D-222, code.
5. Output, well-posedness, meaning of added states: h_base = C_n, h_aug = 0, acyclic + C2 gives
   Thm. 9; added states have no physical identity and are not the simulated absorber; figure
   `fig:aug_structure` referenced here. Sources: Hoekstra 2026 Thm. 9, Sec. II, D-180.

## III-B Closed-loop simulation (`sec:aug_closed_loop`)
Job: define the input the model receives during training and why it is closed loop.
1. Problem: data under feedback, use case is prediction under a (possibly changed) controller; free
   X and Y integrate velocity errors in open loop; departure from open-loop (22) of Hoekstra; adopt
   Kessels (5.12) to (5.13d), controller outside the model (Sec. 5.3.2.3); closed-loop fit is not
   proof of open-loop plant accuracy (Caution). Sources: Kessels, Hoekstra 2026 Eq. (22), Sec. II K.
2. Two loops sharing reference and feedforward, `eq:aug_direct_controller`, `fig:aug_closed_loop`.
   Controller realisation renamed (A_s,B_s,C_s,D_s), LTI, P absorbed so its output is a generalised
   force (notation of Sec. II and IV).
3. Difference form `eq:aug_controller`; exactness conditions; re-discretisation at the model rate
   as a stated approximation (D-141, D-166, values to Setup).
4. Initial controller state x^c = 0 equals s_hat = s, same initial condition as Kessels Remark 5.4
   without reconstruction; evaluation order per sample (no feedthrough). Source: Kessels Remark 5.4,
   closed_loop.py.

## III-C Identification criterion (`sec:aug_objective`, new label)
Job: state what is minimised, over which windows, from which initial state.
1. Encoder: estimates the full state from input-output history including the current sample,
   `eq:aug_encoder`; contrast with Kessels Remark 5.3. Sources: Beintema 2023, Hoekstra 2026 (22e),
   encoder paper Eq. (15).
2. Encoder parameterisation `eq:aug_enc_lin`. Source: encoder paper Eq. (8).
3. Objective `eq:aug_loss`, joint over (theta_base, theta_aug, eta), controller fixed, full window
   scored as in (22a) and (5.12). Sources: Hoekstra 2026 Eq. (22), Kessels Eq. (5.12).
4. Three horizons kept apart (encoder history, scored window, free-run validation); window bound
   from closed-loop memory (D-220); values in Setup.

## III-D Initialisation (`sec:aug_init`, new label)
Job: start training from the baseline.
1. Zero output layer `eq:aug_zero_init` gives baseline behaviour (Hoekstra 2026 Eq. (29)); the
   paper also admits a random linear part (Sec. 5.4.3), not used; non-uniqueness, pointer to
   Section IV. Sources: Hoekstra 2026 Sec. 5.2, 5.4, D-233, D-237.
2. Encoder physical rows from the baseline linearised at rest at Y = 0, ZOH (phrase), `eq:aug_lin`
   without the ZOH integral; reconstructability map `eq:aug_wb`; observability. Sources: encoder
   paper Eqs. (10) to (17), gantry_linearization.py, pre_encoder.py.
3. W^a random, psi_tilde zero, added states inactive until the output layer trains.

## Moved or left out
- RK4 stage equations (`eq:aug_rk4`): textbook, phrase only (style profile, worked contrast).
- MLP layer formula in `eq:aug_mlp` and the ZOH integral in `eq:aug_lin`: textbook, phrase only.
- Contribution sentences of the old intro paragraph: Introduction owns contributions.
- Residual controller check, Remark 5.6 noise question, open-loop evidence: Setup / Results (open
  points).
- Window length, encoder lag, network size, added-state count, rates, optimiser: Setup.
- All draft \todo markers: resolved in prose where a source exists, otherwise open points.
