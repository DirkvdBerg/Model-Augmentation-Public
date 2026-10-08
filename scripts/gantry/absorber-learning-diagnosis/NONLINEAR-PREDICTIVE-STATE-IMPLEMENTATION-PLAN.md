# Implementation plan: nonlinear predictive-state pretraining for dynamic parallel augmentation

Status: PROPOSAL FOR FACT-CHECK. Nothing in this document has been implemented or tested. The
existing `IMPLEMENTATION-PLAN.md` remains the description of the earlier residual-head opening.

Paper references to Hoekstra et al. refer to *Learning-based augmentation of first-principle
models: A linear fractional representation-based approach*, arXiv:2602.17297v1. References to
SUBNET refer to Beintema, Schoukens, and Toth, *Deep Subspace Encoders for Nonlinear System
Identification*, Automatica 2023, arXiv:2210.14816.

## 1. Objective and proposed claim

The objective is to make the additional states of Hoekstra's dynamic parallel state augmentation
learn missing dynamics without prescribing their poles, frequency, damping, or a linear state
transition. The method must train the components that remain in the deployed model. It must not
depend on a temporary residual-prediction head that is discarded after training.

The proposed method is **nonlinear predictive-state pretraining**, abbreviated NPSP in this plan.
It first estimates only the nonlinear augmentation and its added-state encoder while the baseline
is frozen. It then uses the resulting model as the initialization for the existing joint thesis
training procedure.

If all acceptance tests in Section 10 pass, the narrow claim is:

> Nonlinear predictive-state pretraining can initialize the encoder, transition, and physical
> coupling of a dynamic parallel augmentation from measured input-output trajectories, improving
> the acquisition of missing dynamics relative to ordinary zero-initialized training on the
> tested systems.

The method must not claim that the learned states are physical absorber coordinates, that the
estimated realization is unique, or that it is statistically consistent for arbitrary
closed-loop noisy data.

## 2. Relationship to Hoekstra's framework

### 2.1 Elements retained

1. **Augmentation class.** Hoekstra's Table 1 on p. 3 defines dynamic parallel state augmentation as

   \[
   x_{b,k+1}=f_{\mathrm{base}}(x_{b,k},u_k)
              +f_{\mathrm{aug}}(x_{b,k},x_{a,k},u_k),
   \]

   \[
   x_{a,k+1}=g_{\mathrm{aug}}(x_{b,k},x_{a,k},u_k).
   \]

   NPSP uses exactly this deployed model class. The additional states remain nonlinear latent
   states because both \(f_{\mathrm{aug}}\) and \(g_{\mathrm{aug}}\) are neural networks.

2. **LFR representation.** Hoekstra's Eqs. (5) and (6) on pp. 4 to 5 allow the learning component to affect the
   state through the LFR interconnection. NPSP changes neither the baseline block nor the
   well-posed routing. It changes the internal parameterization and initialization of the S-DP
   learning component.

3. **Learning functions.** Hoekstra's Eq. (14) permits a feedforward ANN and Eq. (15) on p. 7 permits a
   ResNet with a linear bypass. The core NPSP proposal uses feedforward ANNs. An Eq. (15) bypass
   is an ablation, not a requirement and not the main source of the learned dynamics.

4. **Encoder.** Hoekstra's Eq. (31) on p. 9 has separate baseline and augmented encoder outputs,
   \(\psi_b\) and \(\psi_a\). NPSP retains that decomposition and trains \(\psi_a\) to initialize
   the deployed added states.

5. **Final estimator.** After pretraining, the final model is refined with the existing truncated
   simulation objective and parameter regularization corresponding to Hoekstra's Eq. (22) on
   p. 8 and Eqs. (26) and (27) on p. 9, as adapted by the thesis pipeline.

### 2.2 Extension beyond Hoekstra

Hoekstra initializes the augmented encoder weights in Eq. (31) by Xavier initialization and gives
a general model-structure initialization in Section 5.4. The paper does not provide a separate
data-driven procedure that makes the added-state encoder, the nonlinear transition
\(g_{\mathrm{aug}}\), and the nonlinear physical coupling \(f_{\mathrm{aug}}\) mutually
consistent before joint estimation.

NPSP adds such an initialization procedure. It is inspired by SUBNET's use of an encoder and
truncated simulation for nonlinear state-space identification, but applies that idea only to the
missing dynamics conditional on a known baseline. The baseline model therefore remains explicit
rather than being replaced by a black-box ANN state-space model.

This is an extension of the identification procedure, not a claim that Hoekstra's structure is
incorrect.

### 2.3 Relationship to existing project implementations

The repository already contains `model_augmentation/fit_systems/augmented_dynamics.py`. That
module makes the added states live with a stable linear recurrence and a driven input map while
retaining a zero-output correction network. Its documentation reports that the placed poles moved
very little in the gantry experiments and concludes that the recurrence behaved as a fixed basis
whose span mattered, rather than as an adaptive resonator.

NPSP must not claim novelty for making the added states nonzero at initialization. Its distinct
hypothesis is that the actual nonlinear transition and its encoder can be trained together from
multi-step data without selecting recurrence poles. `AugmentedDynamics` is therefore a required
linear-basis control, not the implementation of NPSP.

The current model builder also has an optional ReZero-style scalar gate. That gate changes the
initial gradient path but does not train an explicit nonlinear transition and matching encoder.
It is another initialization control rather than a substitute for NPSP.

## 3. Deployed model structure

### 3.1 Separate transition and correction networks

The current gantry model uses one zero-output-initialized MLP to write both physical and added
state rows. NPSP replaces that single network internally by two networks:

\[
\begin{aligned}
z_k &= [x_{b,k}^{\mathsf T},x_{a,k}^{\mathsf T},u_k^{\mathsf T}]^{\mathsf T},\\
\Delta x_{b,k+1} &= f_{\theta_f}(z_k),\\
x_{a,k+1} &= g_{\theta_g}(z_k).
\end{aligned}
\]

The output supplied to the existing interconnect is assembled in `ann_route_ix` order. Physical
rows receive \(f_{\theta_f}\); every additional-state row receives \(g_{\theta_g}\). The physics
block continues to write zero to the additional-state rows, so the latter equation is an absolute
next-state map, not an increment.

This separation follows the two functions shown for S-DP in Hoekstra's Table 1. It prevents the
initialization rule for the physical correction from also forcing the added-state transition to
zero.

### 3.2 Initialization

The initialization is:

1. \(f_{\theta_f}\): hidden layers use the framework's ordinary random initialization; the output
   layer is zero. The complete model therefore equals the baseline at update zero, consistent with
   the baseline-behaviour objective in Hoekstra Eq. (29).

2. \(g_{\theta_g}\): all layers, including the output layer, use Xavier initialization. The added
   states consequently have a live nonlinear transition from the first rollout. Output activation
   is `tanh` in the first implementation so a random transition cannot produce unbounded
   one-step state values. This is a **HEURISTIC** bounded-coordinate choice, not a stability
   theorem.

3. Encoder: introduce `SplitAugmentedEncoder`, matching the two-output form of Hoekstra Eq. (31).
   Its baseline head is the existing baseline-informed linear map plus a separate zero-output
   nonlinear correction. Its added-state head has its own linear and nonlinear parameters. The
   added linear rows use Xavier initialization, and the added nonlinear head is initially zero so
   that the ablation can distinguish the random linear encoder contribution. The complete added
   output is passed through `tanh` in the first implementation so its scale matches the bounded
   added-state coordinates. This is not identical to the current shared encoder and must be
   treated as an architectural factor.

4. \(\psi_b\): retain the existing baseline-informed initialization. Freeze both its linear map
   and its separate nonlinear correction during pretraining. Phase 2 may train them in the same
   way as the ordinary joint pipeline.

No absorber frequency, pole, damping ratio, or target waveform is used in these rules.

The `tanh` state coordinate limits the representable latent coordinate to a compact set. On a
bounded experimental domain, an invertible coordinate compression may exist, but this is not a
general proof of equivalent representation. A no-output-activation transition must therefore be
included as an ablation if the bounded version succeeds.

## 4. Pretraining data and split

### 4.1 Signals and coordinates

Use the same normalized signals, sampling rate, stage-to-logical coordinate conversion, and
record boundaries as the ordinary gantry pipeline. The model and encoder must never mix measured
stage positions with logical baseline states. No data are concatenated across record boundaries.

Each pretraining example starting at time \(k\) contains:

1. the causal past input and output windows required by the existing encoder;
2. the future measured input \(u_{k:k+H-1}\);
3. the future measured output \(y_{k:k+H-1}\), plus the sample and past context required to
   construct the causal encoder window ending at \(k+H\);
4. causal encoder windows ending at \(k+1,\ldots,k+H\), used only by the state-consistency term.

The shifted windows must be implemented as views or indexed gathers from complete records. They
must not be materialized as \(H\) duplicated copies of the dataset.

### 4.2 Calibration split

Split the existing training records by complete periods or complete trajectory segments into:

1. an NPSP fitting subset;
2. an NPSP calibration subset.

The official validation and test records remain untouched. The calibration subset selects the
pretraining checkpoint and the consistency weight. It must not be reused as evidence for final
generalization.

Randomly splitting overlapping windows is forbidden because nearly identical past and future
segments would appear on both sides.

### 4.3 Primary target

The primary target is measured output over a truncated simulation. It is not an estimated
physical-state innovation.

An earlier conceptual version proposed

\[
d_k=\hat x_{b,k+1}-f_{\mathrm{base}}(\hat x_{b,k},u_k).
\]

That quantity is useful as a diagnostic, but it is not a defensible supervised target here:
\(x_b\) is not measured, the baseline encoder is approximate, and its error would be presented to
the augmentation as missing plant dynamics. NPSP therefore avoids treating \(d_k\) as ground
truth.

## 5. Pretraining objective

### 5.1 Truncated output simulation

For a section starting at \(k\), initialize

\[
[\hat x_{b,k},\hat x_{a,k}]
=\psi(u_{k-n_b:k-1},y_{k-n_a:k}).
\]

Roll out the actual S-DP model for \(H\) samples with the recorded plant input. The baseline
parameters and \(\psi_b\) remain frozen, while \(f_{\theta_f}\), \(g_{\theta_g}\), and \(\psi_a\)
are trainable. Define

\[
L_{\mathrm{sim}}
=\frac{1}{BHn_y}
\sum_{i=1}^{B}\sum_{j=0}^{H-1}
\left\|W_y\left(y_{i,k+j}-\hat y_{i,k+j}\right)\right\|_2^2,
\]

where \(W_y\) is exactly the output normalization used by the ordinary thesis objective.

This is a residual identification problem because the baseline is present and frozen, but the
loss is evaluated on the complete simulated output. The target is fixed measured data; it does
not shrink as the current augmentation improves.

The default \(H\) is the existing `cfg.nf`. This is a **HEURISTIC** engineering choice made to
avoid inserting the known absorber period into the method. Dynamics with relevant memory longer
than \(H\) remain weakly identifiable or invisible.

### 5.2 Added-state consistency

For each future sample, independently evaluate the same deployed encoder on the causal window
ending at that sample:

\[
x^{\mathrm{enc}}_{a,k+j}=\psi_a(\text{past ending at }k+j).
\]

Compare it with the added state produced by rolling out the deployed transition:

\[
L_{\mathrm{cons}}
=\frac{1}{BHn_a}
\sum_{i=1}^{B}\sum_{j=1}^{H}
\left\|
x^{\mathrm{roll}}_{a,i,k+j}-x^{\mathrm{enc}}_{a,i,k+j}
\right\|_2^2.
\]

Both sides receive gradients in the core implementation. This encourages the encoder and
transition to use one coordinate system. It does not prove that the coordinate is minimal,
observable, or physically meaningful.

The core pretraining loss is

\[
L_{\mathrm{NPSP}}=L_{\mathrm{sim}}+\lambda_c L_{\mathrm{cons}}.
\]

The candidate set for \(\lambda_c\) must be declared before any result is inspected and selected
only on the calibration subset. The initial proposal is \(\{0,0.1,1\}\). This set is
**HEURISTIC**. The \(\lambda_c=0\) member is required because it determines whether explicit
state consistency contributes beyond ordinary augmentation-only pretraining.

### 5.3 Why collapse is still possible

The solution \(x_a=0\) is not excluded algebraically. If a static map of \((x_b,u)\) can explain
the data, \(f_{\theta_f}\) may ignore \(x_a\). This is not automatically a failure because the
data may not require additional memory. It is a failure only when a held-out dynamic discrepancy
requires memory and the full model does not outperform the same model with \(x_a\) clamped to
zero.

Do not add a variance-forcing state regularizer in the first implementation. Such a penalty can
manufacture active-looking latent states without making them dynamically useful.

## 6. Explicit training procedure

### Phase 0: construction and preflight

1. Build the ordinary baseline, output block, interconnect, and encoder.
2. Replace the single augmentation MLP with the split nonlinear S-DP network from Section 3.
3. Assert that every additional-state row occurs exactly once in `ann_route_ix`.
4. Assert that the update-zero input-output trajectory equals the existing baseline trajectory to
   numerical tolerance.
5. Assert that the added states are nonzero and time varying at update zero for at least one
   nonzero training window.
6. Check gradients separately:
   a. `L_sim` reaches the physical correction output layer at update zero;
   b. `L_cons` reaches \(g_{\theta_g}\) and \(\psi_a\) at update zero;
   c. neither loss reaches frozen baseline parameters or \(\psi_b\).

### Phase 1: nonlinear predictive-state pretraining

1. Freeze the physical baseline parameters and \(\psi_b\).
2. Train \(f_{\theta_f}\), \(g_{\theta_g}\), and \(\psi_a\) using
   \(L_{\mathrm{NPSP}}\).
3. Evaluate `L_sim` on the non-overlapping calibration segments once per epoch.
4. Save the best calibration checkpoint. Do not implement a phase counter or plateau switch
   inside the loss function.
5. Use a declared maximum epoch budget. Early stopping may stop computation, but the selected
   weights are always the best calibration checkpoint, not the last epoch.
6. Save pretraining optimizer state, split metadata, \(H\), \(\lambda_c\), and architecture
   metadata in a separate initialization artifact.

### Phase 2: ordinary joint refinement

1. Reload the best pretraining weights.
2. Unfreeze the parameters that the corresponding thesis arm normally estimates.
3. Remove `L_cons` completely. Use the unchanged thesis loss, parameter regularization, routing,
   and validation selector.
4. Rebuild Adam after changing `requires_grad`; do not load the pretraining optimizer into the
   joint phase.
5. Run the existing Adam training and optional L-BFGS polish.
6. Select the final checkpoint using the existing free-run validation criterion.

The auxiliary objective therefore affects initialization only. All trained networks remain part
of the deployed model, but no auxiliary loss, phase state, or extra prediction head remains.

## 7. Addition to the gantry pipeline

### 7.1 Isolated implementation for the first experiment

The first implementation must not modify Jan's framework files or the established gantry entry
point. Add the following files under `scripts/gantry/absorber-learning-diagnosis/`, each marked
with `__project_origin__ = "added"` where it is a Python module:

1. `predictive_state_net.py`

   Define `SplitDynamicParallelNet`. It accepts the same normalized \([x_b,x_a,u]\) tensor and
   returns outputs in the current route order. It owns `f_net` and `g_net`, exposes their
   parameters separately, and validates the routing contract.

2. `predictive_state_encoder.py`

   Define `SplitAugmentedEncoder`. Copy the already constructed baseline linear maps without
   changing their coordinate convention, give baseline and added states separate nonlinear
   heads, and expose separate parameter groups. The test must show that its baseline rows equal
   the existing encoder's baseline rows at initialization.

3. `predictive_state_data.py`

   Define a record-aware window dataset that returns initial histories, future inputs and
   outputs, and shifted causal histories without crossing record boundaries.

4. `predictive_state_pretrain.py`

   Implement an explicit PyTorch training loop for Phase 1. Do not hide phase transitions in
   `SSE_Interconnect_Composed.loss` and do not wrap the ordinary optimizer with inner steps.

5. `model_predictive_state.py`

   Copy the current gantry model builder and change only the augmentation network construction.
   Return the ordinary `SSE_Interconnect_Composed` fit system so Phase 2 uses the established
   training implementation.

6. `gantry_predictive_state.py`

   Copy the current entry script. Add Phase 1 before the ordinary `train_model` call, plus the
   calibration split and initialization-artifact handling.

7. `test_predictive_state.py`

   Implement all preflight, gradient-routing, checkpoint round-trip, record-boundary, dtype, and
   device tests.

8. `run_gantry_predictive_state.sh` and `smoke_predictive_state.cmd`

   Provide server and local launchers with all new settings written to the run metadata.

### 7.2 Promotion to a framework contribution

Promotion happens only after the isolated experiment passes. The reusable API should then be
small:

```python
initializer = NonlinearPredictiveStateInitializer(
    horizon=cfg.nf,
    consistency_weight=lambda_c,
)

report = initializer.fit(
    fit_system=fit_sys,
    records=train_records,
    calibration_records=calibration_records,
)
```

The model-side reusable component is `SplitDynamicParallelNet`. New framework files receive the
project's `__project_origin__` marker; new symbols in existing framework files receive `@added`.
Promotion also requires a generic test on at least one non-gantry benchmark. Otherwise it is a
gantry experiment, not a framework contribution.

### 7.3 Configuration additions

The copied entry point must record:

1. `NPSP_ENABLED`;
2. `NPSP_EPOCHS`;
3. `NPSP_HORIZON`, defaulting to the ordinary `nf`;
4. `NPSP_LAMBDA_CONS`;
5. calibration record or segment identifiers;
6. transition output activation;
7. separate learning rates for `f_net`, `g_net`, and \(\psi_a\);
8. initialization artifact path when Phase 1 is skipped and a saved initialization is loaded.

All new numerical settings must be labelled `THEORY` or `HEURISTIC` in code according to the
repository rules.

## 8. Required interactions with existing features

### 8.1 Joint physical-parameter estimation

Physical parameters are frozen in Phase 1. This prevents the baseline from moving while the
augmentation target is being established. Phase 2 restores the existing joint-estimation choice
and parameter loss.

This does not prevent negation after unfreezing. The standalone-baseline test remains required.

### 8.2 Orthogonal projection arm

The OBC projection must act only on `f_net` outputs routed to physical rows. It must not alter
`g_net` outputs routed to added-state rows. The current gantry OBC expansion already has zero rows
for additional states, but its reference-field adapter must be tested against the split network's
route ordering.

Run the unprojected arm first. The projected arm is eligible only after these exact identities
hold in a unit test:

\[
g_{\mathrm{projected}}(z)=g_{\mathrm{unprojected}}(z),
\]

and the physical output equals the established projected field for identical weights.

OBC does not solve latent-state collapse and NPSP does not solve baseline negation. They address
different failure modes.

### 8.3 L-BFGS

L-BFGS runs only in Phase 2 after `L_cons` has been removed. Its closure must call the ordinary
thesis loss. No pretraining counters, shifted-history dataset, or calibration state may be
reachable from that closure.

### 8.4 Checkpoints

The split network changes state-dict keys and is not architecture-compatible with existing
single-MLP checkpoints. Loading must fail with an explicit architecture message rather than
partially loading weights.

The Phase 1 initialization artifact and Phase 2 training checkpoint are different objects. The
final checkpoint contains only deployed model and encoder weights plus the ordinary training
metadata. It does not contain duplicated training records or shifted windows.

### 8.5 Device, dtype, and compilation

All shifted encoder calls and rollout tensors must follow the model's device and dtype. First
verify eager CPU and eager CUDA parity. Compilation is attempted only after numerical parity;
the extra encoder evaluations in `L_cons` may not compile efficiently and are not part of the
Phase 2 rollout.

## 9. Experimental design

### 9.1 Minimum controls

Use paired seeds, identical data, identical Phase 2 budgets, and identical checkpoint selection
for:

1. the current zero-initialized single-MLP S-DP control;
2. the current ReZero-gated control;
3. the existing `AugmentedDynamics` linear-recurrence control;
4. split nonlinear S-DP without Phase 1;
5. split nonlinear S-DP with Phase 1 and \(\lambda_c=0\);
6. complete NPSP with the selected nonzero \(\lambda_c\);
7. the earlier linear-head opening.

These arms separate four explanations: architecture, live added-state initialization,
augmentation-only pretraining, and explicit state consistency.

### 9.2 Local screen

Run at least three paired seeds through a shortened but identical budget. The local screen only
rejects clearly broken methods; it does not support a robustness claim.

Record at every validation point:

1. ordinary simulation RMS and BFR;
2. added-state RMS and maximum magnitude;
3. `L_cons` and calibration `L_sim`;
4. the loss increase when \(x_a\) is clamped to zero;
5. added-state transition Jacobian eigenvalues along validation trajectories, reported only as a
   local diagnostic;
6. quiet-record self-oscillation;
7. physical parameter movement and standalone-baseline performance;
8. OBC overlap and projected-field norms where applicable.

### 9.3 Server experiment

Use at least five paired seeds for the surviving NPSP arm and its primary control. Two seeds are
not enough to support a claim about reliability for a stochastic method that has already shown
strong seed dependence.

Predeclare the primary endpoint as held-out free-run simulation performance. Frequency and
damping estimates of the known absorber are secondary diagnostics and may not select checkpoints
or hyperparameters.

Report all unstable and failed runs. Do not silently resample an initialization.

### 9.4 Generality experiment

Success on the gantry absorber supports only a single-system claim. Before promotion to the
framework, use the same algorithm and selection procedure on at least:

1. a missing linear dynamic of unknown order;
2. a missing nonlinear dynamic with memory;
3. a static mismatch, where the method should not invent useful added-state dependence;
4. a no-mismatch control, where it should remain close to the baseline;
5. a changed excitation spectrum or amplitude not used in pretraining.

The nonlinear benchmark is essential. Without it, the experiment cannot establish an advantage
over linear realization or fixed basis-function methods.

## 10. Acceptance and falsification criteria

The proposal passes only if all of the following hold:

1. update-zero predictions equal the baseline while added states are live;
2. the exact deployed transition, correction, and encoder receive the intended Phase 1 gradients;
3. NPSP improves held-out free-run performance over both the current control and split-network
   no-pretraining control across paired seeds;
4. clamping \(x_a\) to zero materially degrades the NPSP result on systems known to require
   additional memory;
5. the improvement survives unseen excitation and quiet-record checks;
6. the standalone baseline and physical-parameter diagnostics show that improvement is not only
   baseline cancellation;
7. the nonlinear benchmark demonstrates value beyond a fitted linear realization;
8. OBC retains its exact routing and projection properties;
9. final Phase 2 training and L-BFGS use the unchanged thesis objective.

Reject or narrow the method if any of the following occurs:

1. the split network alone explains the gain;
2. \(\lambda_c=0\) performs as well as the full method, in which case state consistency is not a
   supported contribution;
3. the added-state clamp has no held-out effect;
4. gains disappear outside the periodic multisine phases seen during training;
5. a linear residual model matches NPSP on every tested discrepancy;
6. pretraining damages quiet-record or long-horizon stability;
7. results require absorber-specific horizon or threshold tuning.

## 11. Limitations

1. **No general state recovery theorem.** The learned latent state is not unique. Even under ideal
   conditions it is identifiable only up to an invertible coordinate transformation.

2. **No physical-state target.** The primary output loss avoids pretending that encoded baseline
   states are ground truth, but it also provides a weaker and more indirect learning signal for
   the added transition.

3. **Closed-loop bias.** The gantry records are collected in closed loop. Recorded plant input is
   correlated with disturbances and measurement noise. NPSP as specified is a direct nonlinear
   output-error initialization, not a proven consistent closed-loop estimator. Classical
   closed-loop subspace methods address this using innovation estimation, high-order predictors,
   instruments, or controller information; incorporating an equivalent nonlinear noise treatment
   is outside the first implementation.

4. **Teacher-forced input.** Phase 1 uses recorded future plant input. It does not expose the model
   to the way its own output errors change controller actions. Phase 2 and closed-loop validation
   must test whether the initialization transfers.

5. **Finite horizon.** Missing dynamics slower than \(H\), weakly excited within a section, or
   outside the excitation band cannot be learned reliably.

6. **State collapse remains possible.** The consistency loss admits collapsed encoder and
   transition states. The simulation loss and added-state clamp diagnose usefulness but do not
   guarantee it.

7. **Shortcut through baseline states.** The correction network may approximate the missing
   effect as a static function of \(x_b,u\). This is acceptable if it generalizes, but then the
   evidence does not support a claim about learned additional dynamics.

8. **Bounded-coordinate restriction.** A `tanh` transition avoids immediate numerical blow-up but
   restricts the latent coordinate and does not guarantee incremental or closed-loop stability.

9. **No automatic order selection.** The first implementation takes `NX_ANN` as a declared model
   choice. A framework-level method still needs record-level validation over candidate dimensions
   or a separately justified order-selection method.

10. **Negation is separate.** Freezing the baseline during pretraining does not prevent the ANN
    from learning baseline dynamics, and unfreezing in Phase 2 reintroduces the usual negation
    risk. OBC and the standalone-baseline test remain necessary.

11. **Encoder leakage risk.** Periodic data and overlapping windows can let the encoder infer
    phase rather than a transferable state. Causal windows, record-level splits, unseen phases,
    and non-periodic tests are mandatory.

12. **Added complexity.** The method introduces a second training stage, shifted-window data
    access, a consistency weight, and a changed network architecture. A gain must be large and
    reproducible enough to justify that complexity.

13. **Not yet a framework contribution.** Until it works on a nonlinear missing-dynamics case
    outside the gantry and exposes a system-independent API, it is an experiment in this thesis,
    not an addition to Hoekstra's general framework.

14. **Novelty may be narrow.** Encoder-based truncated simulation already exists in SUBNET, and
    separate nonlinear state transition and readout maps are standard in neural state-space
    identification. The potentially new contribution is their baseline-conditional formulation,
    baseline-preserving initialization, and integration into LFR S-DP. A dedicated literature
    and citation review is required before describing that combination as novel.

## 12. Primary references and scope of support

1. J. H. Hoekstra, B. M. Gyorok, R. Toth, and M. Schoukens, *Learning-based augmentation of
   first-principle models: A linear fractional representation-based approach*, 2026,
   arXiv:2602.17297, https://arxiv.org/abs/2602.17297. Supports the S-DP structure, ANN and ResNet learning components, encoder-based
   truncated simulation, baseline-behaviour initialization, and LFR framing. It does not establish
   NPSP's consistency loss or its effectiveness.

2. G. I. Beintema, M. Schoukens, and R. Toth, *Deep Subspace Encoders for Nonlinear System
   Identification*, Automatica, 2023, arXiv:2210.14816,
   https://arxiv.org/abs/2210.14816. Supports encoder-based initialization of
   truncated nonlinear simulations and discusses local consistency under its assumptions. Those
   results do not automatically transfer to a frozen-baseline residual subsystem or to the
   explicit consistency term proposed here.

3. G. I. Beintema, R. Toth, and M. Schoukens, *Nonlinear state-space identification using deep
   encoder networks*, L4DC, 2021, PMLR 144:241-250,
   https://proceedings.mlr.press/v144/beintema21a.html. Supports the use of encoder-estimated initial
   states and subsection simulation for scalable nonlinear state-space identification.

4. J. A. Ramos, G. Mercere, and I. Markovsky, *Innovation-based subspace identification in open-
   and closed-loop*, CDC, 2016, https://imarkovs.github.io/papers/cdc16.pdf. Supports the warning that closed-loop input-noise correlation
   requires explicit treatment. It does not validate NPSP, which is nonlinear and uses a different
   objective.

5. M. Forgione and D. Piga, *Continuous-time system identification with neural networks: Model
   structures and fitting criteria*, European Journal of Control, 2021,
   https://doi.org/10.1016/j.ejcon.2021.01.008. Provides related evidence
   for jointly enforcing output fit and consistency of latent states with learned dynamics. Its
   continuous-time formulation and optimized state sequences are not the same algorithm as NPSP.

## 13. Recommended first action

Implement only the split nonlinear S-DP network and the preflight gradient tests before building
the pretraining dataset or launching a run. If the model cannot simultaneously reproduce the
baseline at update zero, keep the added transition live, and route OBC exactly, the remaining
NPSP implementation has no valid foundation.
