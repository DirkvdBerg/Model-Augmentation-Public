# Control Reasoning Reference

Purpose: give any session the project's domain identity and the control-engineering
reasoning behind it. This is the expanded version of the "Control Engineering Stance"
checklist in CLAUDE.md. It describes what exists and why; task state lives in
`tasks/todo.md`, decisions in `docs/decisions.md`, failure history in
`docs/gantry-augmentation-problem-log.md`.

## 1. Project identity

Master thesis, TU/e Control Systems group (AI&ES track, with ASMPT): "Model
Augmentation for a Dual-Gantry High-Precision Motion System". Supervisors:
M. Schoukens, R. Toth, J. Hoekstra (TU/e); Q. van den Elsen, J. Gerritsen (ASMPT).
Full plan: `presentations/Research_plan___Graduation_project_AI_ES___Feedback_Processed (3).pdf`.

Main research question: how should physics-based plant models of the dual-gantry be
augmented to improve parameter interpretability and settling prediction accuracy,
particularly for cross-coupling and position-dependent dynamics?

Four aspects:
1. Position-dependent baseline: Garcia-Herreros dual-gantry model in LPV form with
   payload position Y as scheduling variable (quasi-LPV: Y is a state), discretized;
   compared against frozen-LTI alternatives at fixed Y.
2. Augmentation structure: dynamic parallel LFR augmentation (Hoekstra EJC 2025),
   SUBNET encoder estimation, extended to the LPV setting (Drenth 2025),
   restricted to well-posed realizations.
3. Interpretability preservation: orthogonal projection-based regularization
   (Gyorok et al., L4DC 2025), which requires three extensions here and is the
   thesis's scientific contribution (Section 5).
4. Generalization: free-run simulation, BFR as primary measure, held-out Y
   positions and unseen motion profiles, compared against a black-box model.

## 2. Pipelines and signal chains

Three parallel pipelines. Capabilities are often already built in a sibling
pipeline; search all three before declaring a change necessary.

| Pipeline | Entry point | Role |
|---|---|---|
| `model_augmentation/` (Jan's framework) + `scripts/gantry/` | `scripts/gantry/gantry_interconnect_dynamic.py` | Augmentation training: SUBNET encoder + Interconnect + parallel ANN, joint parameter estimation |
| `lpv_lfr_baseline/` | `lpv_lfr_baseline/scripts/train_param_recovery.py` | Baseline LPV-LFR simulation (CT physics, RK4) and physical parameter recovery via windowed BPTT |
| `scripts/gantry/real-data-verification/` | `run_telica_param_recovery.py` | Parameter recovery on real Telica machine data (closed-loop logs) |

Signal chain facts that repeatedly matter:
- Measured outputs y are stage coordinates [X1, X2, Y]. Physical states are logical
  coordinates [X, Theta, Y] plus velocities. The P matrix
  (`model_augmentation/systems/gantry_ss.py`) maps between them;
  `gantry_interconnect_dynamic.py` builds logical states as
  `pos_logical = (inv(P.T) @ y.T).T`. Any output-side check that skips the
  P-transform is wrong even when shapes match.
- Augmentation pipeline: 20 kHz mat files, downsampled (FS_NEW, currently 4 kHz),
  normalized per channel (x by std of finite-difference logical states, u and y by
  training std). The ANN lives entirely in normalized coordinates.
- Baseline pipeline: per-trajectory per-channel sigma normalizes the loss; inputs
  are stage forces, targets are stage positions q1.
- Telica pipeline: 20 kHz native logs, closed loop (controller in the loop);
  training runs open-loop simulation, validation includes closed-loop checks.
  Schema: `docs/kamtin-telica-schema.md`.

## 3. Research plan vs code (status, descriptive)

| Aspect | In code |
|---|---|
| 1. LPV baseline | Baseline simulation and parameter recovery implemented (`lpv_lfr_baseline/`). Frozen-LTI comparison not built. |
| 2. Augmentation | In progress in `gantry_interconnect_dynamic.py`: stiffness-selective routing, joint estimation, noise-floor acceptance. Failure history in the problem log. |
| 3. Orthogonal projection | Not implemented. Both pipelines implement Lambda parameter-anchoring instead (Section 5); the projection penalty exists nowhere in the codebase. |
| 4. Generalization | Held-out val/test records exist in both pipelines (V/E records). BFR reporting and the black-box comparison baseline not built. |

## 4. Control reasoning checklist, expanded

Each item: the check, the project incident behind it, where to look.

1. **Loop.** Is the data open- or closed-loop, and what does the method assume?
   A controller in the loop correlates input with noise and reshapes what a fit
   means; excitation injected at the reference reaches the plant through T which
   is ~1 in-band, while plant-input injection is attenuated by S.
   Incident: the ref-injection dataset is incompatible with open-loop training
   (`docs/ref-injection-openloop-incompatibility.md`); Telica work needed
   controller reconstruction before closed-loop evaluation was meaningful.
2. **Coordinates.** Which frame is each signal in (stage vs logical), and where
   does the P-transform sit? The data decides the frame, not "cleaner physics".
   Incident: an output-path bug survived a whole session because the test checked
   only shapes; the Y channel is identical in both frames, hiding the error.
3. **Identifiability.** Is the quantity identifiable from this data, or degenerate?
   Only combinations may be identifiable (Section 6). F = m*a cannot separate mass
   scale from force scale. Present all physically consistent interpretations and
   name the external reference that would decide; never call one "impossible".
   Incident: Telica effective mass recovered at ~half the FP-model nominal was
   declared a force-scaling bug; the FP model may simply describe a different
   machine.
4. **Excitation.** Does the input excite what must be learned: frequency band,
   amplitude, and scheduling (Y) coverage? A dynamic invisible in the data cannot
   be recovered, and no optimizer setting fixes that. Incident: the absorber
   period is ~100 ms, so rollouts shorter than nf = 400 samples at 4 kHz contain
   no absorber information; short-window curricula trained on nothing.
5. **Noise setting.** Noiseless simulation and real data admit different
   arguments. SNR budgets, period averaging, and realization variance transfer
   zero information to noiseless data; conversely, on real data, quantities must
   be derived from measurements, never from model matrices or oracle simulations.
   Incident: multiple corrected proposals, including an FRF segment-length rule
   applied to BPTT training and model-derived "noise floors".
6. **Negation.** Can the learning component absorb dynamics the baseline already
   describes? Parameter-anchoring does not structurally prevent this; only the
   projection does (Section 5). Check it with the standalone-baseline test, not
   with parameter deltas alone.
7. **Well-posedness.** The LFR algebraic loop must be solvable: M(Y) invertible
   over the whole operating range, not just Y = 0
   (`docs/m-matrix-invertibility.md`), and the interconnection graph acyclic
   (Hoekstra EJC, Thm. 1: acyclicity equals existence of a topological ordering).
8. **Threshold defensibility.** Acceptance criteria must be data-derived and
   standard: the measurement noise floor (sigma_n = rms(y) * 10^(-SNR/20),
   Jan's convention) is defensible on hardware; an oracle-model floor is not,
   because no oracle exists on the real machine.

Additional standing checks that do not fit one item:
- Marginally stable baseline modes (K = 0 double integrators on X and Y) make
  additive state corrections accumulate without bound over long horizons; routing,
  training horizon, and validation horizon must be chosen with the baseline's
  spectrum in mind. This produced the epoch-0-best failure (problem log, Sections
  2-3) and Jan's stiffness-selective routing fix.
- Model discretization accuracy and encoder/observer quality have separate
  sampling-rate requirements; a passing downsampling validation says nothing
  about encoder initialization quality at that rate.

## 5. Interpretability: negation, Lambda vs Pi

The central distinction of the thesis. Two different regularizers answer two
different questions:

- **Lambda (parameter anchoring, Bolderman via Hoekstra EJC Eq. 6-7).**
  V_reg = || Lambda (theta - theta_init) ||^2 with
  Lambda = (1/(eps * V_MSE(theta_init)))^(1/2) * diag(theta_init)^(-1).
  Answers: "did the physical parameters drift from init?" This IS implemented:
  `lpv_lfr_baseline/blocks/lfr_param_block.py` (param_loss) and the
  `RMSE_baseline` / `PARAM_RMSE_BASELINE` arguments in both training scripts.
  It bounds parameter drift but does not prevent the ANN from reproducing
  baseline dynamics while parameters stay near init.
- **Pi (orthogonal projection, Gyorok et al. 2025).** Stack the
  linear-in-parameters regressor Phi(X,U) over the training data, take the
  reduced SVD Phi = Q Sigma V^T, project with Pi = Q Q^T, and add
  beta * ||Pi f_ANN(X,U)||^2 to the loss, computed cheaply as
  ||Q^T f_ANN(X,U)||^2. This penalizes exactly the ANN output that lies in the
  baseline's span. Not implemented anywhere in this repo.

Facts from the paper that shape the implementation:
- Nonlinear-in-theta baselines are handled by first-order Taylor expansion around
  a linearization point theta_bar (nominally theta_0), with the offset absorbed
  into an extended regressor [Phi_theta_bar, Gamma_theta_bar]. The Jacobian
  d f_base / d theta along FP-simulated trajectories is obtainable by autograd.
- Pi can be precomputed once at theta_0 from FP-simulated states (their Remark 2
  discusses per-epoch recomputation as an optional refinement); the gantry
  pipelines already produce FP-simulated state trajectories.
- Their beta was tuned over decades (1e-7 to 1e-5 worked in their study); the
  scale is task-specific and does not transfer as a number.
- Their evaluation of interpretability is the **standalone-baseline test**: strip
  the ANN, run the FP model alone with the learned theta-hat on test data.
  In their Table 1 (noiseless): nominal baseline 37.91% NRMS; co-estimated
  without regularization 118.01% (baseline destroyed by negation); with
  projection 36.07% (better than nominal). Parameter-delta tables cannot detect
  this failure; the standalone test can.

The three extensions the research plan claims, confirmed absent from the paper:
1. MIMO with structural cross-coupling: their case study has no coupled outputs;
   the dual-gantry has coupled (X, Theta) dynamics.
2. Scheduling-dependent subspace: their Pi is built from one fixed dataset; here
   the baseline output subspace varies with Y. Open design question: one global
   Pi over a Y-sweeping dataset vs Y-binned projections.
3. LFR interconnection: their derivation is for a plain additive state-space
   structure; here the ANN enters through the Interconnect with restricted
   routing. In particular, stiffness-selective routing means the ANN writes only
   to K > 0 state rows, so the projection must be built for the restricted output
   space, an interaction covered by neither paper.

Measurability precondition: augmentation (and negation) is only measurable when
the baseline residual is well above the noise floor. The simulation study's FP
baseline is near-perfect (~1.6e-4 windowed RMS, problem log Section 7), which
makes the legitimate ANN signal small and raises the fraction of learned behavior
that is baseline cancellation. On the real machine the baseline residual is large,
which is where both augmentation and the projection earn their keep.

## 6. Identifiable parameter combinations (gantry baseline)

From `train_param_recovery.py` / `lfr_param_block.py`: 14 raw physical parameters
are trained (log-reparameterized for positivity), but the physics only determines
10 combinations. The pairs enter the dynamics only as sums:

| Identifiable | Raw parameters |
|---|---|
| kb_sum | kb1 + kb2 |
| cb_sum | cb1 + cb2 |
| J_sum | Jb + Jh |
| individually | cg1, cg2, cy, mh, m1, m2, mb |

Splits within each pair are held near init by a split regularization
(`split_loss`); recovery results are judged on the combinations, not the raw
values ("train raw, trust combinations"). Any claim about a raw split parameter
is a claim about the regularizer, not about the data.

## 7. Diagnosing training failures

The full gantry failure history, with diagnostics and outcomes, is owned by
`docs/gantry-augmentation-problem-log.md`; read it before re-deriving any failure
hypothesis. The transferable principle: every failure found so far was in the
signal path, not the hyperparameters. Check in this order:
1. Gradient routing (does the loss gradient reach the component: C_aug dead zone),
2. Encoder initialization quality (state NRMS vs the native-rate reference),
3. Discretization/sampling rate (model accuracy and encoder quality separately),
4. Identifiability and excitation (is the target dynamic in the data at all),
5. Only then learning rates, schedulers, and network sizes.
