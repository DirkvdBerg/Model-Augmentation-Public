# Implementation plan: residual-predictive opening for the added states (draft for fact-check, 2026-10-02)

Status: PROPOSAL. Nothing below is implemented in this form. The closest tested variant (linear head, fixed switch
counts, large opening learning rates; probes E5 and E9 in `DECISIONS.md` section 4) worked in 2 of 4 local runs.
Paper references are to Hoekstra et al. 2026, arXiv:2602.17297 (`literature/closed-loop-id/hoekstra2026_lfr-augmentation-fp-models.pdf`);
pages read by the author of this plan: 3 to 7 and 9 to 10. Appendix A is cited only through a third-party audit and
must be checked.

## 1. Goal and claim

Goal: make the added states of the thesis augmentation learn missing dynamics (on the gantry benchmark: the payload
absorber, 212 Hz, zeta 0.043) from the zero start, without system-specific information, inside Hoekstra's framework.

Claim the method may make if it passes section 7: "A temporary, training-only residual-predictive objective,
followed by a fixed-target fit of the state map and ordinary output-error training, lets the added states of a
zero-initialised dynamic-parallel augmentation acquire missing dynamics on this benchmark." It is an EXTENSION of
Hoekstra's identification procedure (a temporary pre-phase), not part of the paper's method.

## 2. What stays exactly as in the paper / Jan's code

| Element | Paper | Gantry pipeline (unchanged by this plan) |
|-|-|-|
| Augmentation class | dynamic parallel S-DP, Table 1 p.3: x_b+ = f_base + f_aug(x_b, x_a, u), x_a+ = g_aug(x_b, x_a, u) | `Static_ANN_Block` writes routed physical rows additively and the added rows (`scripts/gantry/gantry_dynamic/model.py`, `build_model`) |
| Learning function | plain ANN Eq. (14) or ResNet Eq. (15), p.7; Sec. 6.3 p.11: parallel experiments use feedforward networks | one zero-init 2x16 tanh MLP (`zero_init_feed_forward_nn`), as in Jan's examples (D-233) |
| Initialisation | baseline behaviour at init, Eq. (29) p.9 | output equals the baseline at init (final layer zero; added states start at x_a+ = 0, D-233) |
| Encoder | psi_b, psi_a, Eqs. (30), (31) p.9 | `linear_encoder_init_aug` (W^b analytic, W^a kaiming; known deviations, D-233) |
| Cost | truncated simulation cost plus parameter regularisation, Eqs. (22), (26), (27) p.9 | thesis closed-loop rollout loss (nf 400) + `param_loss`; OBC arm: forward projection |
| Final model | | identical class and size; the opening's head is discarded |

The opening adds only temporary loss terms and a schedule. No model parameter, routing or initialisation changes.

## 3. The method, step by step

Notation: x_hat(k) = [x_hat_b(k), x_hat_a(k)] = encoder(past window ending at k); F = one model step (`fs.hfn`);
y_sim = closed-loop rollout of the CURRENT model from x_hat(k); L_thesis = the unchanged training loss.

**Phase A: residual-predictive encoding (shapes x_a).**
- Target: e(k) = y(k..k+M-1) - y_sim(k..k+M-1), the current model's next M output residuals, detached, divided by
  its batch RMS. # HEURISTIC: M = 40 samples (10 ms at 4 kHz, about two periods of the gantry mode); the general
  rule ("a few periods of the slowest unmodelled band of interest") is itself a choice that needs justification.
- Head: a small training-only MLP h (1 hidden layer, 16 tanh units, zero-initialised output) maps x_hat_a(k) to e(k).
  NONLINEAR by user decision (the linear head borrows linear subspace theory, section 6).
- Loss term: L_A = ||h(x_hat_a(k)) - e(k)||^2 / numel. Gradient reaches the head and the encoder's added-state
  outputs (W^a and the x_a rows of the encoder's zero-init NL net); nothing else.
- Total loss in Phase A: L_thesis + lambda_A L_A, lambda_A set once so both are equal at the first evaluation (as E3).

**Phase B: fixed-target state-map fit (teaches the ANN the dynamics in x_a).**
- Loss term: L_B = ||x_hat_a(k+1) - F_a(x_hat(k), u(k))||^2, with x_hat(k) and x_hat(k+1) DETACHED (the encoder is
  the teacher; detaching prevents collapse of x_a to a trivially predictable signal). Gradient reaches only the ANN
  (the physics block writes no added row). x_hat(k+1) uses the window shifted by one sample (as `innovation_opening.py`).
- Total loss in Phase B: L_thesis + lambda_A L_A + lambda_B L_B (L_A continues so the target keeps forming).
- How the state map is given enough optimisation to fit L_B (DECISION REQUIRED, section 5):
  - B1 (tested): opening-phase learning-rate factors on the ANN (hidden layers x100, added-row output x1000 on the
    thesis lr 1e-5), via a per-row/group factor in Adam (`RowScaledAdam`, `innovation_opening.py`).
  - B2 (untested): K inner supervised steps on L_B alone per training update, on the ANN parameters only, at a
    moderate fixed lr (the regression is one-step and cheap; AB1 fitted it in minutes).

**Phase C: output-error refinement.** lambda_A, lambda_B decay linearly to 0 over D updates; afterwards the loss is
exactly L_thesis, all rate factors 1, the head is discarded. This phase learns the readout (how x_a enters the
physical rows) and maps the closed-loop object of Phases A and B back to the plant (section 6, L2).

**Switch rules (data-derived, thresholds fixed before any run, never tuned on the gantry pole).**
- Every W training updates, evaluate on a fixed held-out batch (validation windows): the head's unexplained fraction
  U_A = L_A / var(e), and the state-map residual U_B = L_B / var(x_hat_a(k+1)).
- A -> B when U_A improved by less than delta_A (relative) over the last 2 checks, or at a cap N_A_max.
- B -> C when U_B improved by less than delta_B over the last 2 checks and U_B < u_B_max, or at a cap N_B_max.
- # HEURISTIC: W, delta_A, delta_B, u_B_max and the caps have no literature value. Proposal: W = 25, delta = 2 %,
  u_B_max = 0.05, caps 1000 / 1000 updates, D = 150 updates. Local data for reference: the linear-head E3 run
  plateaued at U_A about 0.75 to 0.8 by update 300 to 400; AB1 reached U_B 0.002 to 0.008 at its optimum.

## 4. How it is added to the gantry pipeline (no edits to protected files)

Protected, not edited: `scripts/gantry/gantry_interconnect_dynamic.py`, `scripts/gantry/gantry_dynamic/`,
Jan's code in `model_augmentation/`, and the prepared E3 server files (`innovation_opening.py`,
`model_innovation.py`, `gantry_innovation_opening.py`, `run_gantry_innovation_opening.sh`,
`test_innovation_opening.py`, `smoke_innovation_opening.cmd`).

New files (folder `scripts/gantry/absorber-learning-diagnosis/`, marked `__project_origin__ = "added"`):
1. `residual_opening.py`: class `ResidualOpening(SSE_Interconnect_Composed)` (same pattern as `InnovationOpening`):
   overrides `loss()` to add L_A and L_B according to the phase; holds the head, the phase state, the held-out
   batch and the switch logic; counts only gradient-enabled training evaluations (validation runs under no_grad are
   excluded, as in `InnovationOpening.loss`); after Phase C returns exactly `SSE_Interconnect_Composed.loss`.
   For B1 it reuses `RowScaledAdam` (imported, not modified); for B2 an optimizer wrapper whose `step()` first runs
   K inner L_B steps on the ANN parameters, then the normal Adam step.
2. `model_residual.py`: copy of the model builder that returns `ResidualOpening` instead of the composed class (same
   pattern as `model_innovation.py`, changes marked).
3. `gantry_residual_opening.py`: copy of `gantry_interconnect_dynamic.py` whose only changes (marked) configure the
   opening and rebuild the optimizer. Env switches for M, head size, B1/B2, thresholds.
4. `test_residual_opening.py`: pre-flight checks (section 7.1).
5. `run_gantry_residual_opening.sh`, `smoke_residual_opening.cmd`: server array and local smoke runs.

Interactions to keep correct:
- Joint estimation and `param_loss`: unchanged; L_A and L_B do not reach the physics parameters (L_A: detached
  rollout, head on x_a only; L_B: added rows, written only by the ANN). To verify in the pre-flight.
- OBC arm: the projection acts in the forward pass on the ANN's physical-row output; its basis rows for the added
  states are exactly zero (`model_augmentation/fit_systems/obc_projection.py` module docstring, "ADDITIONAL STATES"),
  so L_B on the added rows is unaffected. The readout learned in Phase C is projected as usual.
- L-BFGS polish at the end of the run calls the loss many times per step; the opening must have ended long before
  (the caps guarantee it); the pre-flight asserts the loss equals L_thesis after the opening.
- Checkpoint selection: unchanged (free-run validation sim-RMS). Reported in addition, not used for selection: the
  quiet-record check (validation records where the model's sim-RMS exceeds block-off, AD1).
- Compile mode / CUDA: the head and the inner steps must live on the model's device and dtype (E3 server files hit
  this; same fix).

## 5. Open decisions (to settle before implementation)

1. Head: nonlinear MLP (this plan) vs linear. A linear head has the subspace-identification rationale, a nonlinear
   one has none (section 6, L1). Decide after the literature sweep `tasks/handoffs/2026-10-02-lit-nonlinear-residual-predictive-state.md`.
2. B1 vs B2. B1 is the only tested form but confounds "target" with "large learning rates"; B2 removes that confound
   by design but is untested and adds an inner loop to the optimizer.
3. Whether to also offer the ResNet linear bypass of Eq. (15) (a paper-supported alternative parameterisation of the
   learning function, D-233 note) as a separate arm. This changes the deployed model; keep it out of this plan's core.
4. Threshold values (section 3).

## 6. Limitations (critical)

- L1 No theory for the nonlinear version. The only formal support is linear: Phase A is then reduced-rank
  past-to-future regression of the residual (subspace identification), recovering the error system's predictor
  state up to similarity, for linear, open-loop, exogenous non-periodic input, exact rigid-body baseline, fixed
  target. With a nonlinear head, x_a is identified at best up to an arbitrary invertible nonlinear map; only Phase B
  (propagation by F_a) keeps it state-like, and nothing proves it does.
- L2 Closed loop. The residual of a closed-loop simulation carries closed-loop poles (true and nominal); measured
  data add noise feedback. Phases A and B therefore learn a closed-loop object (AA1: residual optimum 232 Hz vs plant
  212 Hz). Phase C is relied on to map it back to the plant; this is observed (E5, E9: 208 to 209 Hz), not proven.
- L3 Moving target. e(k) is the CURRENT model's residual; as training explains it, the target shrinks toward noise.
  It may also help (static and rigid-body errors are removed first, leaving the mode). Unquantified.
- L4 Detuned baseline and friction. The residual is not only the absorber: mass errors add double-integrator terms,
  Coulomb friction is nonlinear (AA1: rank 2 explains 48 % of the linearly predictable residual, rank 4 77 %). With
  n_a = 2 the bottleneck may pick the wrong content on another system.
- L5 Order n_a is not selected by the method; with n_a larger than the missing order, the extra states are
  unidentified and may self-oscillate (E6 self-oscillated on quiet records even with n_a matched).
- L6 Heuristics: M, head size, thresholds, caps, decay length, and (B1) the rate factors. None has a literature value.
- L7 Evidence: only the linear-head, fixed-count, B1 variant was tested: 2 of 4 local runs complete, 2 partial;
  batch 64, at most 1100 updates, arm U only. Nonlinear head, switch rules and B2 are untested. Nothing at server
  scale (batch 512, 200 epochs, lr 1e-5), nothing on the OBC arm.
- L8 Damping drift: under long pure-loss training the learned pair moved to marginal damping (E7, local |z| 1.000);
  it stayed stable over 12 s records, but a 26k-update run may differ.
- L9 Periodic multisine data: a future-input term in the head would absorb the mode (shortcut); this plan
  deliberately has none, which means the head must also model input-predictable residual content.
- L10 Generality: one system, one missing linear mode. Claims beyond "on this benchmark" need another missing
  dynamic with the same hyperparameters.
- L11 Paper/code ambiguity: whether the dead start is Jan's intended method or an artefact of his example code is
  open (D-233); if Jan used a live init for S-DP, the opening may be unnecessary.

## 7. Verification and acceptance

7.1 Pre-flight (local, `test_residual_opening.py`, must pass before any run):
- model equals the baseline at start (update-0 loss equals the zero-init pipeline's, both arms);
- L_A gradient reaches only head and encoder added-state outputs; L_B only the ANN; physics parameters receive no
  gradient from either term;
- after the opening the loss equals `SSE_Interconnect_Composed.loss` exactly and all rate factors are 1;
- state_dict of the final model has the same keys as a thesis run (head excluded);
- switch rules fire on a synthetic plateau and respect the caps;
- OBC arm builds its basis and the projection leaves added rows unchanged.

7.2 Local screen (before the server), each with prediction first in `DECISIONS.md`:
- 3 seeds of the plan (zero start, same batches) with paired zero-init controls; for B1 additionally a matched-rate
  control (same rate factors and schedule, opening weights 0) to separate target from learning rate.
- Measured: added poles at encoder points, steal check (`g2_steal.py`), held-out band removal (`ac_band.py`),
  whole-record free run including quiet records (`ad_valfree.py`).

7.3 Pass / fail (server, pre-registered): added pair in 150 to 300 Hz with zeta <= 0.15 at >= 90 % of encoder points,
AND more than 22 % of the Y error removed in 230 to 297 Hz on the full multisine validation records (run 42's metric),
AND final validation sim-RMS no worse than run 42, AND no validation record above its block-off value. Fail otherwise.
Arms U and OBC, at least 2 seeds each; the submitted E3 array (job 87446) serves as the unordered control.

## 8. Cost

Local: pre-flight and smoke about 1 h; screen about 3 seeds x (plan + control) x 1 h. Server: 4 runs (2 arms x
2 seeds) of about run-42 wall time (<= 24 h each, 1 GPU); opening overhead estimated at a few percent (M-step
rollouts during Phase A and B, held-out checks every W updates, B2 inner steps).
