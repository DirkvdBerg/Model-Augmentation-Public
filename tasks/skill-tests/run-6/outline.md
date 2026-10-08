# Outline: Section III, Dynamic LPV-LFR Augmentation (run 6)

Status: pre-approved for this test run.

**Roadmap (one line):** the baseline misses effects of the machine; we augment it with a
dynamic-parallel learned part (A), give each training window an encoder initial state (B), fit
the closed-loop response with the known controller (C), and start training from the baseline (D).

Change of structure against the current draft: initialisation (zero output layer, W^b, W^a,
non-uniqueness) moves out of III-A and III-B into a new III-D, following the order of
Hoekstra 2026 Sec. 5 (criterion before initialisation). The attribution paragraph of the
section opening is dissolved: each citation moves to the subsection that uses it.

## Section opening (roadmap paragraph)
- Topic: the baseline of II-B describes the rigid-body dynamics, not every effect of the machine.
  Garcia Sec. 2.4 neglected effects; our baseline also omits Coulomb friction; this section adds a
  learned dynamic part and estimates theta_base, theta_aug, eta jointly from closed-loop records;
  names III-A to III-D. Sources: garcia2013model Sec. 2.4.

## III-A Augmented model (job: define the model structure)
1. Topic: an omitted vibration mode needs additional model order. Refit or static correction
   cannot add it; additional states can (hoekstra2025lfr Sec. 2); parallel because Section IV acts
   on an additive correction (research plan Aspect 2, Sec. IV). Equation: eq:aug_transition
   (hoekstra2026lfr Table 1, S-DP), where-clause defines x~, x-bar, x-hat, u-hat, u_k, y_k.
2. Topic: f_base is one classical RK4 step of the continuous-time model with the input held.
   Consequence: M(Y) evaluated per stage, self-scheduled within the step; Drenth precedent
   (drenth2025thesis Sec. 5.2); normalisation of signals and states (hoekstra2026lfr Sec. 5.3).
   No display (RK4 stages are textbook machinery).
3. Topic: one MLP supplies both learned functions. Equation: eq:aug_mlp reduced to the output
   split (layer formula dropped as textbook). Routing: all six physical rows, because the missing
   dynamics couple into X and Y (D-103); Theta-only routing named as the insufficient alternative
   (research-plan anchor: departure from earlier routing). Discrete-time nature of f_aug.
   Width and depth to Setup.
4. Topic: the output map is not learned. C_n = [P^T 0] (eq:Ptransform), acyclic graph, smooth
   f_base and tanh give well-posedness (hoekstra2026lfr Thm. 9). Additional states have no
   assigned meaning and are not the states of the truth-only absorber (header Caution).
   Figure fig:aug_structure referenced here.

## III-B Encoder and initial state (job: define the window initial state)
1. Topic: each window needs an initial state, but only positions are measured. Encoder
   (beintema2023subnet), eq:aug_encoder; histories include the current sample; all states encoded,
   positions included; trained jointly. History length to Setup.
2. Topic: the encoder is a linear map with a nonlinear correction (hoekstra2026encoder Eq. 8),
   eq:aug_enc_lin; the linear part can be initialised from the baseline (III-D), psi~ learns the rest.

## III-C Closed-loop rollout and objective (job: define the training criterion)
1. Topic: we fit the closed-loop response, because the records are measured and the model is used
   under feedback; departure from open-loop Eq. 22 (hoekstra2026lfr). Second reason: two
   eigenvalues at the origin on X and Y (eq:damping_stiffness_matrices), so open-loop position
   errors persist; the controller acts on them if the loop is stable.
2. Topic: we adopt Kessels' rollout with the known controller inside, gradients through it,
   controller outside the model (kessels2025ai Eqs. 5.12, 5.13c, 5.13d; Sec. 5.3.2.3).
3. Topic: recorded and model loop share r, u^ff and controller. Equation: eq:aug_direct_controller.
   Figure fig:aug_closed_loop.
4. Topic: subtracting the loops removes r and u^ff. Equation: eq:aug_controller. Exactness
   conditions; needs only recorded u and y (D-141).
5. Topic: x^c_tau = 0 sets s-hat_tau = s_tau without reconstruction; consequence; Kessels
   Remark 5.4 as the alternative; evaluation order (no model feedthrough).
6. Topic: the parameters minimise the closed-loop output error. Equation: eq:aug_loss;
   truncated objective (hoekstra2026lfr Eq. 22) in closed loop (kessels2025ai Eq. 5.12); joint
   estimation, fixed controller; every sample scored as in Eq. 22a.
7. Topic: three horizons are set separately: encoder history, window n_f (must span closed-loop
   memory, D-220), validation over complete records in closed loop (config). Values to Setup.
8. Topic: a small closed-loop error does not show open-loop accuracy (header Caution); Section VI
   tests this separately.

## III-D Initialisation (job: start training from the baseline)
1. Topic: training starts from the baseline, because random learned parts can make the start
   unstable (hoekstra2026lfr Sec. 5.4). Equation: eq:aug_zero_init; reproduces the baseline
   (hoekstra2026lfr Eq. 29).
2. Topic: W^b is initialised from a linearisation of the baseline (hoekstra2026lfr Sec. 5.4.2
   for the reason). Equation: eq:aug_lin (A_c, B_c, C_d; ZOH integral as a phrase).
3. Topic: for this linear model W^b is the reconstructability map. Equation: eq:aug_wb
   (hoekstra2026encoder Eqs. 16, 17); observability; normalisation; only a starting value.
4. Topic: W^a random (Kaiming uniform, not Xavier as in hoekstra2026lfr Sec. 5.4.2, D-233),
   psi~ zero; added states inactive at the start, activated by training the output layer.
5. Topic: the split between f_base and f_aug is not unique (hoekstra2026lfr Sec. 5.2);
   Section IV constrains it. Sizes to Setup.

## Moved or left out
- Contribution sentences of the opening: left out (Introduction owns the contribution list).
- RK4 stage display (eq:aug_rk4): replaced by a phrase (textbook machinery, style profile Sec. 10).
- MLP layer formula in eq:aug_mlp and the ZOH integral in eq:aug_lin: replaced by phrases.
- All numerical settings (n_x_a, width, depth, n_a = n_b, n_f, validation records): Setup.
- All \todo notes: to the open-points list of the delivery report.
