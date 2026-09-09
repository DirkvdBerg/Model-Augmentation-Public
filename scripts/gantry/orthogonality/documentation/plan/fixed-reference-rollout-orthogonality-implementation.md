# Gantry implementation specification: fixed-reference trajectory orthogonality

Revised: 2026-09-08. Status: proposed implementation specification, not completed gantry validation.
This replaces the 2026-09-07 version of this same plan. It owns the candidate algorithm,
mathematical contract and acceptance ladder, without creating another theory record.

## 1. Objective, scope and evidence

Extend stacked orthogonal-projection regularization to the **total closed-loop prediction
effect** of a dynamic augmentation, including trainable latent initialization. Test whether
directional regularization preserves useful correction better than simply shrinking the
augmentation, while retaining physically interpretable parameter estimates.

The candidate uses frozen reference geometry, a finite penalty and selective gradients.
It is not a theorem of true-parameter recovery, a hard constraint or an H2 construction.
It includes latent effects in the penalized contribution; it does not separately constrain
every latent route or every possible ANN training direction.

Supporting records retain these roles:

- [gaps.md](../gaps.md): live theory/evidence and claim ledger, Sections 7 and 9–10.
- [decisions.md](../../../../../docs/decisions.md): D-180 through D-184 and historical decisions.
- [source audit](../theory-source-audit-2026-09-06.md): per-paper assessments; its older proposed
  theory is not an alternative implementation specification.
- [closed-loop verification report](../proofs/nonlinear-closed-loop-orthogonality.md):
  independent MATLAB/SymPy checks of a small nonlinear example.
- [RULES.md](../RULES.md) and [algebra-tooling.md](../algebra-tooling.md):
  evidence categories and independent-source requirements for FP structural proofs.

Every numbered equation has a warrant. Distinguish evidence status from data requirements:

| Status | Meaning |
|---|---|
| Definition / design | Specifies an object or algorithm, not its efficacy |
| Derived | Follows from the stated assumptions and argument |
| Symbolically checked | CAS checked the specified identity/example; scope matters |
| Code-inspected | Matches inspected source; runtime parity is a separate check |
| Numerically checked / measured | Depends on example, data, precision and settings |
| Open | Required result or applicability remains unestablished |

Separately label quantities structural, data-computable, or truth-requiring validation.
Known simulation truth may validate a method but may not select deployable geometry, beta
or checkpoints. A numerical toy optimum is not a structural theorem.

### 1.1 Decisions to implement

| Item | Candidate | Qualification |
|---|---|---|
| Simulator | Production residual closed-loop simulator only | No open-loop gantry acceptance requirement |
| Effect | Full minus augmentation-off scored output stack | Includes route interaction and controller response |
| Geometry | Full-predictor physical sensitivity at frozen reference | Current geometry can drift |
| Latent initialization | Trainable augmentation parameters, including penalty gradients | Resolve current shared nonlinear encoder ownership |
| Physical initialization | Trainable nuisance parameters, profiled across actual shared windows | May remove needed physical directions |
| Physical updates | Base objective only; physical parameter anchor OFF in primary comparison | Anchor-on is a separately labeled matched ablation |
| Penalty | Scalar-MSE-normalized trajectory projection | Finite beta does not enforce exact orthogonality |
| Core | Existing shared core plus production adapter | Generic tests do not certify gantry integration |
| Intervention | Non-persistent ANN output gate indexed through routing | Must survive backward/checkpoint replay correctly |
| Basis refresh | Frozen initially, with drift monitoring | D-183 re-anchoring is a separate ablation |
| Controls | None, projection, raw-size, profiled-size | Paired starts, data, architecture and budgets |
| Reference | Output-preserving raw encoder conversion of one fixed checkpoint; no post-conversion warm-up | Warm-up is a separately declared reference choice |
| Gradient integration | Explicit accumulation hook after base backward and before optimizer step | No custom backward in the first implementation |
| Initial window budget | Four-window smoke probe; start pilot geometry at 32 training windows | Coverage/cost checks can require a documented pre-training revision |

The geometry probe determines whether the profiled physical space is useful. It does not
determine parameter ownership: that is a declared design contract. No stage is claimed
completed merely because this specification defines it.

### 1.2 Implementer's ordered checklist

1. Select the actual checkpoint/configuration and verify dimensions, routes, scoring, controller
   and anchor flags; save the manifest before any run.
2. On a disposable output-preserving encoder conversion, run the early four-window physical
   and initialization-sensitivity probe. This uses the existing full production simulator and
   precedes gate and optimizer integration. A small-set failure is diagnostic, not a conclusion
   about all possible regularization windows.
3. Apply the predeclared rank/resource decision in Sections 6.5 and 13. Implement production
   encoder conversion, gates and parity checks only after reviewing this early result.
4. Integrate the explicit gradient hook, verify full-stack/chunked updates and measure cost.
5. Run the primary anchor-off comparison only after its acceptance gates pass. Reserve
   tangent-capability claims for the additional diagnostic in Section 7.4.

Normative choices are summarized here and in Section 1.1. The derivations justify them;
the evidence index does not replace this task order.

## 2. Relation to Györök

Györök 2025 already uses a state-space baseline, multistep fitting and encoder initialization.
Its augmentation is a static one-step correction, and its regularizer acts on stacked
evaluations of that correction. The extension is **not** the introduction of state space,
an encoder, or trajectory stacking in general.

Here the correction stack becomes a closed-loop output difference through the actual dynamic
augmented predictor. It includes latent initialization, writing, persistence, readout and
feedback. Its physical regressor becomes the sensitivity of that scored predictor, with
initialization nuisance treatment made explicit.

The project's zero-latent evaluation slice leaves an affine latent readout K unpenalized.
That is a structural counterexample to the slice's coverage; Györök's original static ANN
has no latent states. The K=0 control shows propagation alone also changes the rollout
condition. That observation is not claimed as the learned-state novelty.

Györök's nonlinear extended regressor includes an affine-offset direction alongside parameter
tangents. This candidate deliberately uses parameter tangents only. Removing latent states
does not therefore make it a literal reproduction of the original penalty.

Warrant: source audit, Györök 2025 Sections 2–4, Eqs. (3),(4),(9)–(19); gaps Sections 7b/9.5.
Attributions use the existing local source audit, not a new full publication review.
Section 15 records source limits. No unconditional novelty or recovery theorem is asserted.

## 3. Gantry instantiation and source owners

| Contract | Authoritative source |
|---|---|
| Model, routing, encoder construction | [model.py](../../../gantry_dynamic/model.py) |
| Dimensions and flags | [config.py](../../../gantry_dynamic/config.py) |
| Physical log parameters, RK4/LFR, output block | [blocks.py](../../../../../model_augmentation/fit_systems/blocks.py) |
| Encoder maps and shared correction | [pre_encoder.py](../../../../../model_augmentation/fit_systems/pre_encoder.py) |
| Loss, scoring and optimizer closure | [interconnect.py](../../../../../model_augmentation/fit_systems/interconnect.py) |
| Controller and authoritative rollout | [closed_loop.py](../../../../../model_augmentation/fit_systems/closed_loop.py) |
| Closed-loop signal notation | [closed-loop-form-v4.tex](../../../../../docs/writeup/closed-loop-form-v4.tex) |
| Normalization | [coordinates-normalisation-v1.tex](../../../../../docs/writeup/coordinates-normalisation-v1.tex) |
| FP source structure | [fp-model-structure.md](../../../../../docs/fp-model-structure.md) |

Use the write-up partition x=[tilde x; bar x]. Let a=bar x be shorthand, and x_c the
residual-controller state. Thus chi=[tilde x; a; x_c].

Code-inspected dimensions:

- tilde x: six normalized physical rows [X, Theta, Y, dX, dTheta, dY].
- u: three normalized stage-force channels.
- y: three normalized measured stage-output channels in current RunConfig.
- a: hp['NX_ANN'] rows. The checkpoint of interest has eight; current config defaults to two.
- ANN output: len(ann_route_ix), not automatically six plus n_a.
- x_c: dimension from ControllerBank.

Current config defaults to all eight state rows for its two-state augmentation; older
comments mention restricted routing. Read the actual run artifact, not those comments.
An eight-latent-state full-routing run needs its 14-row routing explicitly verified. Do not
copy the two-output illustrative calculation from a conversation into this three-output pipeline.

### 3.1 Coordinates and physical parameters

With training-data standard-deviation matrices and means,

    tilde x = D_x^{-1}(x_SI-mu_x),
    u = D_u^{-1}(u_SI-mu_u),   y = D_y^{-1}(y_SI-mu_y).              (1)

Warrant: coordinate definition and normalization contract. Verify actual buffers and offsets.
Learned states have no independently measured physical scale. Predictions must use the
actual output block, not the first three physical-state entries or velocity estimates.

Let lambda denote raw trainable log coordinates, theta physical parameters:

    theta_i(lambda)=max(theta_init,i exp(lambda_i),1e-6).            (2)

Warrant: Parameterized_Gantry_State_Block._recover_params, code-inspected.
Smooth arguments apply away from clamp boundaries. Reference construction requires no active
clamps in the intended physical coordinates. Otherwise stop and explicitly redesign the
coordinate set; do not call a clamp-induced zero column structural nonidentifiability.
Monitor clamp activation during training, not only at geometry builds. Activation invalidates
the unchanged-coordinate sensitivity claim and triggers a paused geometry review. Report the
unclamped parameter values and distance to the boundary as well as the active mask.

Raw order:
kb1, kb2, cg1, cg2, cy, cb1, cb2, mh, m1, m2, mb, Jb, Jh, d.

The block's ten combination coordinates are

    kappa(theta)=(
      kb1+kb2, cg1, cg2, cy, cb1+cb2, mh,
      m1+m2+mb, m1-m2,
      Jb+Jh+(m1+m2)Lb^2/4, d
    ).                                                             (3)

Warrant: inspected _combos_from_raw mapping; FP structural proofs remain in their existing
independent-source records. These are not ten arbitrarily selected raw parameters.
The signed m1-m2 combination must not be log-transformed indiscriminately.

Initially differentiate the supported production block with respect to all 14 lambda entries,
retaining supported singular directions. If the predictor factors through kappa with
initialization independent of lambda, S_lambda=S_kappa D_lambda kappa by the chain rule.
Equality of spans also needs this combination map locally onto. Do not implement a new
ten-coordinate model without a validated parameter-to-matrices map. Structural identifiability
does not guarantee rank ten in this scored, encoder-profiled experiment.

### 3.2 Scheduling and discrete physics

f_lambda is the actual RK4 map of the normalized continuous LFR, including up_sample.
Y is the denormalized third physical state. At a substage,

    D_x f_c(x,u,Y(x),theta)
      = partial_x f_c |_Y + partial_Y f_c D_x Y.                    (4)

Warrant: chain rule. Differentiate every RK4 substage, LFR solve and denormalization.
Freezing Y omits a first-order derivative. Subsequent movement of the sensitivity subspace
is a separate nonlinear geometry issue.

Never remove log_params from the Parameter registry to accommodate forward-mode duals.
Use an explicit pure parameter evaluation or tested functional substitution, without a stale
_cur cache. Compare against production, not merely against another private physical block.

## 4. Closed-loop prediction, initialization and interventions

### 4.1 Production recurrence

Let B_r expand ANN outputs into routed state rows, partitioned B_r,p and B_r,a.
For ANN N_eta_d and intervention mask m,

    w_k=diag(m) N_eta_d([tilde x_k; a_k; hat u_k]),
    tilde x_{k+1}=f_lambda(tilde x_k,hat u_k)+B_r,p w_k,
    a_{k+1}=B_r,a w_k.                                             (5)

Warrant: code-inspected additive routing. Latent outputs replace the next latent state;
they are not automatically increments. The production interconnect executes this equation.
Active wrappers and trainable gates must be declared in the run manifest.

Using the write-up's normalized signal convention,

    hat y_k=h(tilde x_k),       e_k=y_k-hat y_k,
    u_fb,k=C_c x_c,k+D_c e_k,
    x_c,k+1=A_c x_c,k+B_c e_k,
    hat u_k=u_k+u_fb,k.                                            (6)

Warrant: closed-loop-form-v4 and closed_loop._rollout_segment, with actual bank normalization
and per-record matrices. Current D_d=0 means output is evaluated BEFORE the model state update.
A future direct input/output path requires a revised well-posedness contract.

Recorded u,y, controller selection and controller initialization match across interventions.
Each intervention evolves its own controller state. Production zero residual-controller state
at window start is an experiment convention, not an estimate of the true controller state.

The difference below is an effect in this residual closed-loop prediction experiment.
Replaying full-rollout controller inputs/states into off would define a different effect.
Closed-loop feedback is included deliberately; the result is not an uncontrolled plant discrepancy.

### 4.2 Parameter ownership and encoder migration

Use disjoint groups

    physical: lambda;
    augmentation: eta=(eta_d,eta_a,active augmentation gates);
    encoder nuisance: xi_p.                                       (7)

Initialize each window with

    tilde x_w,0=e_p(H_w;xi_p),      a_w,0=e_a(H_w;eta_a).             (8)

Warrant: selected design. Latent initialization remains trainable under BOTH prediction and
penalty updates; it is not profiled away. Physical initialization remains trainable under the
base loss and is profiled as nuisance. That does not eliminate initial-state ambiguity.

The current linear_encoder_init_aug has separate Wb_psi_* and Wa_psi_* tensors but a shared
nonlinear correction net for physical and latent rows. Slicing outputs or detaching one does
not make its trainable dependencies disjoint.

Default migration: retain the linear maps, create two independent nonlinear correction
branches, copy the original hidden layers into both, and copy the corresponding final-layer
rows and biases. For the current feedforward architecture this preserves initial encoder
outputs while separating subsequent updates. Verify architecture, activations, dtype, buffers
and output parity. Use fresh paired optimizer states after conversion; do not silently retain
incompatible moments. If the selected encoder differs, explicitly implement its conversion
or reject the configuration.

This changes parameter sharing even when initial outputs match. All primary penalty controls
must use the same converted encoder. Run an unregularized converted-versus-original control
separately. Freezing physical initialization is an optional different method, not the default.

Penalty differentiation holds tilde x_0 constant and keeps a_0 live through eta_a.
Base-loss differentiation keeps both live. Tensor identity checks alone are insufficient:
perturb eta_a and verify physical initialization does not change, and conversely for xi_p.
Within each evaluation of a window, encode its history once and reuse the physical initial
value across interventions. Two-pass recomputation encodes again at identical parameter and
stochastic states; it does not fit a separate initial condition for each mode.

### 4.3 Full, off and clamped modes

- full: learned a_0, all ANN write gates one.
- off: same physical initialization, a_0=0, all ANN write gates zero.
- clamped: same physical initialization, a_0=0, physical-write gates one and latent-write
  gates zero. Under current routing a then stays zero.

Classify ANN output j by route_ix[j]<6 or >=6. route_cols means ANN-output indices,
not full-state row indices. Validate mappings. Suppress all learned output paths if a
future architecture includes direct output augmentation.

Assert at runtime that the clamped trajectory has zero latent state at initialization and
after every transition, including active wrappers. Incremental dynamics alone do not break
this: a_next=a+g with a_0=0 and g gated to zero also stays zero. An ungated state-write path
can break it and must be disabled explicitly or the configuration rejected.

Use one fixed-shape non-persistent buffer on the existing ANN output path, not a wrapper or
weight mutation. Scope modes with exception-safe restoration; loading initializes all-one mode.
The buffer is not a trainable parameter.

Backward may save a gate tensor: changing its storage can invalidate gradients. Capture a
safe gate value as needed. Checkpoint recomputation must replay the original mode, not the
current global buffer mode. Initially keep the graph-bearing full mode valid through its
backward; off is no-grad. Additional graph-bearing modes require explicit replay tests.
Compiled-mode parity, restoration on exceptions and checkpoint safety are acceptance gates.

Assert off-independence for every declared augmentation parameter: perturb eta_d, eta_a and
trainable augmentation gates while holding physical parameters and physical initialization
fixed, and require unchanged off predictions. Also check an AD-enabled off evaluation for
absence of those dependencies; a zero derivative at one special parameter value is insufficient
on its own. This is the premise licensing no-grad off, and a future ownership change must fail
this test rather than silently inherit that optimization.

With identically scored output stacks,

    d=p_full-p_off,
    d_lat=p_full-p_clamped,   d_dir=p_clamped-p_off,
    d=d_lat+d_dir.                                                  (9)

Warrant: algebraic identity for three vectors, not evidence that the simulator interventions
are correct. The split is intervention-based, not a unique separation of nonlinear mechanisms.
Projected direct/latent contributions may cancel.

## 5. Scoring and scalar loss weight

Each window predicts nf outputs and scores k=B,...,nf-1, T=nf-B>0.
B is the existing loss burn_in value; preserve its slicing and preceding differentiated
transitions. No burn-in sweep or encoder mitigation is proposed here.

For n_W fixed regularization windows, stack window/time/channel in a saved order:

    p=vec([hat y_w,k]),    N=n_W T n_y,
    J_pred=||p-y_stack||^2/N.                                      (10)

Warrant: code-inspected plain mse_loss for equal-length, equally weighted normalized outputs.
History right extensions and per-window controller identities must match the production loss.
Existing anchors are separate base terms.

Whitening is L=I/sqrt(N). It leaves column spans unchanged, so do not instantiate L.
The implementation penalty is

    V=(beta/N)||Q^T d||^2.                                        (11)

Warrant: definition, equivalent to beta||Q^T Ld||^2.
Scalar whitening does NOT allow omission of 1/N from the penalty value. Save the convention;
old sum-based beta values require conversion.

Channel weights, masks or unequal averaging invalidate this scalar contract and require an
explicit revised metric. Dense weights are outside the first adapter. General weighted
projection remains valid, but whitening may then couple row chunks.

## 6. Geometry and encoder computation

### 6.1 Reference

Choose a training-data-only checkpoint after encoder conversion and fixed windows G.
Paired conditions share checkpoint, geometry and normalization. At q_0=(lambda_0,eta_0,xi_p,0),

    S=D_lambda p_full(q_0) in R^{N x 14},
    E=D_xi_p p_full(q_0) in R^{N x n_xi_p}.                         (12)

Warrant: definitions. Full-predictor S includes the current augmentation's state dependence.
Baseline-only S is another method. AD is enabled to construct derivatives, then outputs are
detached and stored. “Frozen geometry” does not mean disabling all differentiation during construction.

Default reference selection is the raw, output-preserving conversion of one preselected
checkpoint, without any post-conversion warm-up. Reset optimizer states identically in all
primary arms. This selects a reproducible reference, not a calibrated optimum under the new
ownership. A stabilization phase is permissible only as a separately declared shared protocol
completed before constructing all paired references; do not warm up each penalty arm separately.

### 6.2 Initialization chain rule: D-184

If xi_p enters predictions only through physical initialization,

    Gamma_w=D_tilde_x_w,0 p_w in R^{T n_y x 6},
    J_w=D_xi_p e_p(H_w) in R^{6 x n_xi_p},
    E_w=Gamma_w J_w,
    E=blockdiag_w(Gamma_w) stack_w(J_w).                            (13)

Warrant: exact chain rule under the stated dependency. Perturb xi_p while explicitly holding
initialization fixed and require unchanged predictions. Gamma_w includes propagation through
latent dynamics and controller feedback.

D-184's earlier 14 initial-state columns referred to both partitions before the ownership
revision. Nuisance propagation now has six columns per window. D_a0 p is useful for latent
gradient verification but is not part of E.

Compute Gamma_w with batched independent-window JVPs or an equivalent recurrence. Six
direction labels can be vectorized across windows because windows do not interact. Globally
Gamma has 6 n_W columns, not six. Do not differentiate whole rollouts over thousands of
encoder weights, and do not claim a six-dimensional global nuisance space.

    rank(E)<=min(N,n_xi_p,6 n_W),
    range(E) subseteq range(blockdiag Gamma_w).                    (14)

Warrant: product rank/range identities. Profiling free per-window initial states may remove
more than profiling the actual shared encoder. Factorization alone does not solve memory:
explicit J or E can still be large.

### 6.3 Exact compressed construction

For each window obtain an orthonormal basis U_w for range(Gamma_w), rank r_w<=6.
Let C_w=U_w^T Gamma_w, Z_w=C_w J_w, Z=stack_w Z_w and U_G=blockdiag_w U_w.
Then, before numerical truncation,

    E=U_G Z,   U_G^T U_G=I,
    Z=U_Z Sigma_Z V_Z^T,
    Q_E=U_G U_Z[:,nonzero singular directions].                    (15)

Warrant: Gamma_w=U_w C_w, followed by SVD. Derived here; gantry implementation is open.
Numerical rank truncation is an approximation with a recorded threshold, not an exact identity.

Build Z_w by differentiating the small encoder output C_w e_p(H_w), holding C_w fixed.
No rollout depends on differentiated encoder weights in this construction. Store U_w and
window blocks of U_Z; form Q_E blocks when needed. r_G=sum r_w<=6 n_W, so Z is
r_G-by-n_xi_p rather than N-by-n_xi_p. It may still be too large: measure before scale-up.

When every r_w=6, Z has the same dimensions as the stacked encoder Jacobian J_w. The saving
is relative to E, whose row count includes the scored trajectory, not necessarily relative
to J. Orthogonal factors enable this representation; they do not guarantee affordable storage.

Use direct SVD initially. Z Z^T squares conditioning; it is not a harmless alternative.
If matrix-free computation is needed, validate it separately rather than dropping encoder
parameters or substituting independent initial states. On a tiny set compare against explicit E.

### 6.4 Profiling and rank policy

    M_E v=v-Q_E(Q_E^T v),
    Sbar=M_E S,     Q=orth(Sbar).                                  (16)

Warrant: projection definitions. No dense M_E or QQ^T is necessary.
Accumulate Q_E^T S across windows, then form Sbar blocks in a second pass. Q has at most
14 columns; zero or reduced supported rank is possible.

Use float64 for initial geometry verification and save spectra. A physical rank threshold
tau_S=max(atol,rtol||S||_2) avoids declaring residual profiling roundoff a real direction
when Sbar should vanish. It may also discard weak real directions: save threshold sweeps,
absolute spectra and separate tolerances for Gamma_w, Z and Sbar. Do not force rank ten.

Under the factorization through (3), and with independent initialization and fixed external
scalings as specified, rank(S)<=10. The generic core's 14-column capacity is not an expected
gantry rank. Derive reference log-coordinate null vectors from D_lambda kappa and check
||S n|| relative to ||S|| ||n||. Those structural null tests complement the rank integer.
An extra mass/force-scale degeneracy must be proved for the actual free parameters and force
normalization before adding it: it cannot be imported from a model with different freedoms.

Zero profiled physical rank stops the protection experiment. Reduced rank narrows its claim;
diagnose excitation, observations, encoder overlap, clamping and precision.
The dimension fraction 6/(n_y T) is only a count, not an overlap estimate. A ten-dimensional
encoder space can erase ten physical directions inside a huge output space. Extra windows
can change shared-encoder geometry. Shrinking E does not shrink its exact projector:
P_(epsilon E)=P_E for every nonzero scalar epsilon.

### 6.5 Early probe and predeclared rank decisions

Before gate/closure integration, use four reproducibly selected training windows spanning
available records/excitation regimes, with the intended physical encoder branch on a disposable
conversion. Check its initialization-output parity and a few S/Gamma directional derivatives
against the existing full simulator before interpreting spectra. No off intervention is needed.

If shared-encoder construction is not ready, free physical initial-state profiling is an early
bound only. Its nuisance range contains the shared-encoder range: full physical rank surviving
that larger nuisance space is reassuring; collapse there does NOT prove shared-encoder collapse.
Do not substitute the whole unsplit encoder and call it the proposed geometry.

Record r_S=rank(S), r_bar=rank(Sbar), supported spectra and threshold sensitivity. Apply:

- r_S=0 or r_bar=0: no protection pilot on this geometry. Diagnose before integration scale-up.
- 0<r_bar<r_S: the encoder removes supported physical directions. A separately labeled
  reduced-span diagnostic pilot is permissible, but not the primary all-supported-directions
  comparison. First assess a more representative predeclared window set and rank robustness.
- r_bar=r_S<10: a supported-subspace pilot is permissible, with an explicitly reduced physical
  claim. Do not claim all ten combinations are protected or identifiable.
- r_bar=r_S=10 with robust spectra: pass the rank prerequisite for a ten-combination pilot;
  this alone does not prove tangent separation or recovery.

Four-window decisions concern that set. Confirm the same gates on the final G before training.
Do not require a favorable outcome or quietly revise tolerances until a desired rank appears.

### 6.6 Coordinate robustness of numerical profiling

For an invertible encoder-coordinate change xi'=T xi, E'=E T^{-1}. Exact ranges, hence exact
nuisance projectors, are equal; truncated singular-value ranges need not be. This is a derived
invariance statement with a numerical-applicability limitation, not another efficacy theorem.

On a small explicit or compressed E case, test fixed-seed diagonal rescalings with positive
entries log-spaced between 0.1 and 10, and orthogonal right-coordinate transformations where
affordable. These are proposed diagnostic ranges, not guarantees under arbitrary conditioning.
Repeat rank thresholds by factors 0.1, 1 and 10 around the declared default, above a justified
roundoff floor. Report projector distance, r_bar and the physical singular spectrum. Document
base tolerances and residual checks before viewing the resulting ranks.

Material changes flag numerical geometry uncertainty and block an unqualified rank claim.
Do not silently interpret weak encoder directions as absent. If a stable numerical range cannot
be resolved, use better arithmetic/scaling or explicitly redefine a perturbation metric/budget.
The latter is regularized nuisance treatment, not unrestricted profiling. Apply corresponding
robustness checks to Gamma_w compression and physical coordinates as well.

## 7. Mathematical justification and limits

### 7.1 Local least squares

At a fixed reference, with augmentation held fixed, consider a fixed output perturbation d:

    min_delta_lambda,delta_xi ||S delta_lambda+E delta_xi+d||^2
      = min_delta_lambda ||Sbar delta_lambda+M_E d||^2              (17)

where equality means equality of the minimized nuisance-eliminated objective.
Warrant: orthogonal least-squares profiling. This is a local unanchored prediction calculation,
not the full nonlinear training problem with existing residuals, anchors and freely moving ANN.

For full column rank in the selected physical coordinates,

    delta_lambda_*=-Sbar^dagger M_E d,
    delta_lambda_*=0 iff Sbar^T M_E d=0 iff Q^T d=0.                (18)

Warrant: normal equations and range(Q)=range(Sbar).
For rank-deficient raw coordinates, this describes the minimum-norm solution only; arbitrary
physical null directions remain. It does not establish unique raw parameters or true recovery.

### 7.2 Value orthogonality versus enforcement

Since Q^T M_E=Q^T,

    V=0 iff Q^T d=0 iff Sbar^T M_E d=0   (beta>0),
    ||Q^T d||/sqrt(N)=sqrt(V/beta).                                 (19)

Warrant: projector algebra, also checked in the generic CAS report.
This quantifies an attained penalty, not what training will attain. Finite beta or stationarity
does not imply V=0. Total d orthogonality does not imply separate d_lat orthogonality.

The hard constraint c(eta)=Q^T d/sqrt(N)=0 is a different algorithm. Toy solvers retained
a useful nonzero correction at numerical tolerance with physical/encoder parameters fixed.
That is example-level feasibility evidence, not exact symbolic existence on the gantry,
global optimality or joint-training recovery. A projected free output vector need not have a
causal recurrent realization on unseen inputs.

### 7.3 Values versus training directions

Let R=D_eta p_full. Orthogonality of the current d does not imply Q^T R=0.
An exact identity c(eta)=0 throughout an open parameter family, with Q fixed and p_off
independent of eta, would imply Q^T R=0 by differentiation. At a single feasible point
of a constrained manifold, only its feasible tangent vectors v satisfy Q^T R v=0.
Finite regularization supplies neither identity.

Local non-substitution after nuisance profiling requires sufficient physical rank and

    range(M_E S) intersection range(M_E R)={0}.                    (20)

Warrant: linearized cancellation geometry. Orthogonal tangent spaces are sufficient and
stronger than trivial intersection. Encoder overlap already removed from S is not protected.
This proof gap remains even if V becomes very small.

At Q^T d=0, with Q fixed and off independent of eta, the penalty Hessian is

    D_eta^2 V=(2 beta/N) R^T Q Q^T R.                              (27)

Warrant: twice differentiate (11); the residual-weighted second-derivative term vanishes.
The penalty can add curvature against aligned directions without removing them from the
model class. Tangent overlap is not by itself a reason to reject a soft-regularization pilot.
It is a reason to withhold a claim of local inability to substitute.

### 7.4 Capability diagnostic: required only for a capability claim

Restrict raw physical variations to supported coordinates. If S=U_s Sigma_s V_s^T,
let A=V_s[:,supported] diag(1/sigma_supported), so S A is orthonormal. Record this local
output-normalized coordinate choice and its threshold; it is not a physical-unit convention.
For Nuis=[E,R], evaluate

    F=(I-P_Nuis) S A,       G_cap=F^T F.                           (28)

Warrant: least-squares elimination of encoder and augmentation tangent freedom. Positive
definite G_cap means local first-order distinguishability of these supported physical
directions; a null direction supplies a local cancellation. It is not a nonlinear recovery
theorem. Using fourteen raw coordinates makes this Gram structurally singular already.

On a manageable window set, obtain each F column from a nuisance least-squares solve using
E/R JVPs and VJPs. Check convergence, normal residuals, tolerance sweeps and a small explicit
reference problem. Incomplete solves overestimate distinguishability. Uncertain accuracy gives
an inconclusive diagnostic, not a positive eigenvalue certificate.

This diagnostic is optional for the realized-contribution pilot and required before claiming
reduced first-order confounding or inability to substitute. Full tangent separation is not
a prerequisite for studying whether curvature (27) helps.

### 7.5 Full-predictor versus baseline physical directions

With identical physical initialization and controller conventions,

    S_full=S_off+D_lambda d.                                      (29)

Warrant: differentiate p_full=p_off+d. S_full is the appropriate derivative for the local
compensation problem of the predictor actually trained. S_off describes the physical-only
intervention. Neither is universally the correct target for every thesis claim.

On the small reference set measure relative ||D_lambda d||, supported ranks and projector
distance. Compare unprofiled spaces first, then use the SAME nuisance projector to isolate
the effect of changing S. Rebuilding E_off is another change and must be labeled separately.
Keep output/log-coordinate conventions identical. Small differences support only an empirical
approximation argument. Record the denominator convention for relative norms.

A baseline-space training ablation is optional for the full-predictor claim; require it if
interpreting the method as preserving standalone-baseline directions. A frozen chosen
checkpoint is not automatically an interior optimum of a defined calibration loss. Define
that target separately before importing calibration-optimality conclusions.

## 8. Derivatives and gradient routing

For chi_{k+1}=F_k(chi_k;q), hat y_k=H_k(chi_k;q), any parameter group v satisfies

    X^v_{k+1}=A_k X^v_k+B^v_k,   Y^v_k=C_k X^v_k+D^v_k,
    A_k=D_chi F_k, B^v_k=D_v F_k,
    C_k=D_chi H_k, D^v_k=D_v H_k.                                 (21)

Warrant: exact first-order chain rule along a smooth nonlinear trajectory, not an LTI
assumption. Initialize derivatives from the actual encoder. Include controller and scheduling,
then slice scored outputs. The nonlinear toy verifies this recurrence; production parity is open.

Let J_base include prediction loss and any explicitly selected base terms. The physical
parameter anchor is off in the primary comparison (Section 11). Apply

    g_lambda=D_lambda J_base,
    g_xi_p=D_xi_p J_base,
    g_eta=D_eta J_base+(2 beta/N)(D_eta d)^T Q Q^T d.               (22)

Warrant: algorithm choice plus exact eta derivative of (11).
Detach physical parameter leaves and tilde x_0 for penalty evaluation, or request only eta
gradients. Keep a_0 live through eta_a, and keep every subsequent physical, latent, controller
and scheduling path reached from eta live. Off is independent of eta under the specified
ownership, so it may be no-grad; its VALUES still use current physical parameters and physical
initialization and must be refreshed after they change.

A geometry-buffer no-graph assertion does not verify this policy. Test actual added gradients
and optimizer steps for all groups. Finite-difference eta, including latent initialization,
with other groups fixed.

This update is not the gradient of J_base+V. To prove it is not the gradient of ANY scalar
objective requires a nonzero mixed derivative D_lambda D_eta V or D_xi_p D_eta V.
Nonzero D_lambda V alone is insufficient. Existing report checks establish the weaker
statement; their stronger nonconservativity interpretation needs correction.

Use the existing Adam-compatible route initially. A joint L-BFGS line search on J_base+V
with (22) supplied as its gradient is inconsistent. Do not generalize that to a prohibition on
all blockwise optimization or a claim that no convergence theory could ever apply.

### 8.1 Exact two-pass window chunking

For consistently partitioned complete windows,

    r=sum_b r_b,   r_b=Q_b^T d_b/sqrt(N),   V=beta||r||^2.          (23)

Warrant: matrix multiplication. Generally ||sum r_b||^2 differs from sum||r_b||^2.
Pass 1 accumulates r with no graph. Pass 2 freezes r, model values, data, geometry and
stochastic states, recomputes each chunk and accumulates eta gradients of

    surrogate_b=2 beta stop_gradient(r)^T r_b(eta).                 (24)

Warrant: chain rule for squared norm. Log V from pass 1, not surrogate values.
Take one optimizer step after accumulation. A rows argument that computes all windows and
then slices saves no rollout memory: select histories/signals/controllers before simulation.

The production fit closure clears gradients after computing loss. Do not prepopulate gradients
inside loss and expect them to survive. Implement an explicit accumulation hook after
zero_grad/base backward and before optimizer step; custom backward is deferred. The order is:
zero_grad; evaluate/backward J_base; no-grad residual pass; recompute chunks and add eta-only
gradients; apply configured clipping to the combined gradients; optimizer step. With mixed
precision, gradient scaling must be consistent; start with the verified non-AMP path.
Do not scale beta by chunk count.
Replay dropout/stateful buffers/checkpoint modes identically or reject the unsupported configuration.

## 9. Shared core and main-pipeline integration

### 9.1 Responsibilities

[trajectory_orth_projection.py](../../../../../model_augmentation/fit_systems/trajectory_orth_projection.py)
owns generic geometry algebra, projections, diagnostics and the accumulator.
[trajectory_adapter.py](../../../../../model_augmentation/fit_systems/trajectory_adapter.py)
defines the contract. The gantry adapter calls the production builder, encoder and closed-loop
simulator. It must not implement another private gantry rollout.

The API names physical/augmentation/encoder can remain, but augmentation includes eta_a
and encoder means xi_p only. Existing generic adapters do not automatically satisfy this
revised ownership or the compressed D-184 construction.

Required adapter operations:

1. Describe dimensions, routing, readout/offsets, normalization, histories, score slicing,
   controller selection, wrappers, dtype and parameter-group dependencies.
2. Produce encoded initial values and the frozen-physical/live-latent penalty view.
3. Score full/off/clamped modes on selected complete windows through production simulation.
4. Evaluate physical JVPs and per-window initial-state Jacobians functionally.
5. Compute encoder-only products and compressed nuisance geometry.
6. Return detached geometry artifacts with verified row order and provenance.
7. Integrate (22) and (24) into the actual optimizer closure.

### 9.2 Parity and functional parameters

The primary parity target is fit_sys's own prediction and scored loss from identical histories
and signals. Matching saved and parameterized blocks inside a private loop is insufficient.

With the feature disabled, compare outputs, base loss, gradients and one optimizer step.
For encoder conversion, compare initialization outputs separately before resetting optimizer
state. Full gate must match unmodified production predictions. Off must match the production
physical-only intervention semantics, including controller evolution.

Resolve forward-mode incompatibility without registry mutations. Test module registration,
state_dict structure, parameter/buffer values and caches after a Jacobian call. Constant-looking
float32 output offsets do not justify looser derivative or subspace tolerances. Inspect time
accumulation, directional derivative errors and weak singular directions separately.

### 9.3 Configuration and artifacts

Keep the feature disabled by default until the verification ladder passes. Persist:

- Algorithm/version and encoder migration/ownership version.
- Kind: none, projection, raw_size, profiled_size; beta and normalization N.
- Reference hashes for model groups, normalization, controller bank and active wrappers.
- Dimensions, route_ix/route_cols, channels, nf/B, history right extensions.
- Fixed training-window identities, controller identities and exact stack order.
- Gamma/Z/S/Sbar thresholds, spectra, factorization residuals and basis factors.
- Dtype, AD mode, checkpoint/compile configuration, RNG seeds and optimizer-state policy.
- Geometry, penalty and base-update timing and peak CPU/GPU memory.

Do not register the live model as a child of the geometry module or duplicate optimizer
ownership. On resume verify provenance instead of silently rebuilding geometry at current
parameters. The non-persistent intervention gate must initialize to full mode.

## 10. Diagnostics and current-reference mismatch

For v equal to total, direct or latent contribution, compute

    vbar=M_E v,
    a_par=||Q^T v||/sqrt(N),
    a_perp=||vbar-Q(Q^T v)||/sqrt(N),
    ell=||Q^T v||/||vbar||.                                       (25)

Warrant: orthogonal decomposition. ell is undefined for a below-floor denominator; do not
report zero as success. Also report ||v||/sqrt(N) and ||Q_E^T v||/sqrt(N).
Contribution norm ratios are not percentages of an additive energy budget.

These are data-computable diagnostics. Old checkpoint latent-path measurements motivate the
work but do not certify this geometry. Withdrawn open-loop/state-readout values in gaps.md
must not be reused as production acceptance targets.

A consistent invertible linear latent change a'=Ta, with encoder, dynamics and readout all
transformed, preserves predictions and zero-clamp interventions, hence (25). Neither ||a||
nor ||K|| alone measures contribution. Translations require transforming the clamp reference.

At documented checkpoints rebuild current S_q/E_q on the same windows. Compare against the
ACTUAL saved reference, not a reconstructed “initial” model using current eta or xi_p.
Let Pi_q=Q_q Q_q^T:

    ||Pi_q d||/sqrt(N)
      <= ||Pi_0 d||/sqrt(N)+||Pi_q-Pi_0||_2 ||d||/sqrt(N).          (26)

Warrant: triangle inequality. At equal rank, projector distance is sin of the largest principal
angle. At different ranks use projector distance and rank reporting, not only common-rank
angles. Small consecutive angles do not bound accumulated drift.

Report current and reference alignment separately, along with rank before/after profiling and
conditioning. There is no established universal acceptable drift cutoff. Physical, dynamic
and initialization updates can all move the basis; attribution to the encoder requires an
isolation experiment. Refreshing the basis is a separately named algorithm.

## 11. Strength selection and controls

**Primary decision:** disable the physical parameter-anchor term in every primary arm.
Assert its evaluated contribution and added gradient are zero; do not infer this from a flag
whose meaning differs between block classes. Inventory other base penalties and keep them
identical across arms. This is a choice to test projection without active physical anchoring,
not a claim that combining the two is mathematically invalid.

Primary conditions use identical converted encoders, starts, fixed windows, optimization
budgets and eta-only penalty routing:

1. J_base only.
2. Trajectory projection (11).
3. Raw size: lambda_s ||d||^2/N.
4. Profiled size: lambda_s ||M_E d||^2/N.

Named efficacy risk: apparent parameter improvement may come entirely from suppressing the
augmentation. Until size controls establish a directional-selection benefit, the method has
not demonstrated an advantage over shrinkage. Verify the historical comparison's exact
quantity, scaling and gradient policy before calling these controls a replication of it.

Secondary mechanistic comparisons, separately budgeted:

- Anchor-on: repeat matched none/projection/size conditions with the same fixed anchor.
  This measures incremental benefit conditional on anchoring; one unmatched extra arm
  cannot isolate the interaction.
- Full-gradient projection: optimize J_base+V with consistent gradients into all selected
  groups and Q frozen. This tests routing. It is not necessary to isolate projection versus
  size when those primary arms already share routing.
- Baseline-space projection: as scoped in Section 7.5, with other conventions matched.

The profiled-size control matches nuisance treatment. Raw size additionally shrinks
encoder-space effects. Same numerical strength does not mean matched regularization.
Compare validation tradeoffs and, where informative, matched alignment suppression or correction size.

Pointwise-only and both penalties are secondary ablations. Single-seed differences do not
prove harm or redundancy everywhere. Off-trajectory coverage requires its own evaluation
conditions, not an assumption that same-distribution experiments were incapable of finding benefit.

No closed-form beta rule is established for this penalty. Bolderman Eq. (38) weights a physical
parameter anchor, not orthogonality. Its separate network L-curve is tuning precedent.
Predeclare a training/validation sweep, budgets and data-computable selection rule. Include
beta=0, grid-boundary reporting and correction-preservation diagnostics. An L-curve-inspired
plot under selective updates is exploratory, not a theorem of optimal beta.

Current RULES.md labels beta without its required L-curve a chosen constant. Report it that
way; adopting another accepted rule would require an explicit rule revision. Do not import
an old numeric beta without the N convention, geometry and gradient policy.

Run a paired multi-seed pilot before strong conclusions. Evaluate separate records and longer
closed-loop sequences without overlapping-history leakage. Do not select hyperparameters or
checkpoints by simulated parameter truth. Report physical combinations, seed/record variability
and held-out fit; these are data-computable, not proof of truth. Known-parameter simulations
may validate recovery separately. Good prediction does not bound physical-parameter bias.

At augmentation-off d=0, V=0 and D_eta V=0. That is expected for squared residuals and does
not establish that near-zero initialization prevents learning or explain a tie with size
regularization. Measure actual gradient reachability, including latent initialization and
active gates. Warm-start or ReZero changes must be shared across primary controls.

## 12. Verification ladder using production code

Use a focused driver around the production builder, encoder, simulator and shared core.
Existing generic checks are references, not replacements for these gates.

| Stage | Required check | Acceptance / limit |
|---|---|---|
| P0 early probe | Actual configuration; disposable encoder parity; four-window S/Gamma/E ranks and null-vector tests | Before gate/closure work; repeat rank decision on final G |
| V0 ownership | Encoder conversion output parity; disjoint dependencies; eta_a perturbations do not change physical initialization | Required for the revised algorithm |
| V1 simulator | Full production output/scoring parity; off semantics; independent controller evolution; gate restoration | Compare against fit_sys, not a private loop |
| V2 derivatives | Directional finite differences for lambda, eta_d, eta_a, xi_p; initial-state chain rule; scheduling/controller negative controls | Save absolute/relative error across step sizes |
| V3 geometry | Explicit/compressed E parity; rank/null checks; coordinate rescaling; shared/free-state comparison; clamp gate | Apply Section 6.5 decisions, including reduced-span labels |
| V4 gradient policy | Penalty reaches eta_a and eta_d; no added lambda/xi_p gradients; one-step parity | Buffer graph checks alone do not pass |
| V5 chunking | Stacked versus differently chunked V, gradients and optimizer update; deliberate cross-chunk cancellation | Actually recompute window subsets |
| V6 execution | Eager/compile parity; checkpoint on/off gradients; gate replay; peak-memory profiling | Optimize only verified paths |
| V7 interpretation | Direct/latent cancellation; latent-coordinate rescaling; zero contribution overlap with nonzero tangent overlap | Prevent overclaiming from a zero metric |
| V8 pilot | Anchor-off paired none/projection/size; both reference and current alignment; rank/clamp monitoring | Suppression only under Q_0 supports only the reference claim |

For claims about the final model's current physical tangent, improvement of the current-geometry
aligned amplitude must be shown against paired controls, alongside correction size and fit.
Report current metrics in each model's own geometry and common-reference metrics separately;
rank loss is not evidence of improved separation. Tangent-capability claims additionally require
Section 7.4 and are not an automatic consequence of passing V8.

Keep the MATLAB/SymPy scripts as independent small references. Their shared-encoder example
does not verify the newly separated latent-initialization gradient policy or actual gantry
readout. Add a small separated-ownership check and direct-versus-factorized encoder Jacobian
comparison. Verify (15) against explicit construction before its gantry use.

If retaining the stronger nonconservativity claim, add an actual mixed-derivative witness.
A nonzero physical-gradient check is not enough. For intended symbolic zero identities,
unknown results fail the check. Three rational witness evaluations remain point checks.

For FP-specific structural claims follow algebra-tooling.md: independent MATLAB transcription
from kamtin-fp-model and SymPy transcription from Python. Two engines on the same abstract
toy are not independent-source FP validation. No new FP theorem is claimed in this revision.

Tolerances must reflect dtype, scale and perturbation convergence. Float64 is the initial
reference, not a precision guarantee. Preserve spectra and failed controls. Unexplained OOM,
rank collapse, output mismatch or stale functional caches block scale-up.

## 13. Cost and execution order

No runtime multiplier is established for the revised gantry method. Historical eager iteration
times, one-step AD speedups or dense-matrix OOM explanations do not predict its total cost.

Let C_fwd denote forward-only evaluation of the regularization set and C_grad its graph-bearing
forward/backward cost. Two-pass evaluation includes full/off forwards in pass 1, full recomputation
and backward in pass 2, and cached off outputs or another off forward. Base training is additional.
Benchmark O(N) off-output caching versus recomputation. Latent initialization stays live in pass 2.

Geometry cost includes physical directions, six per-window physical-initial-state directions,
encoder-only products and factorizations. Batched directions do not imply an exact sixfold
runtime. Benchmark AD tangent chunk sizes on the complete closed-loop map. The historical
145x one-step benchmark is not an expected speedup for this geometry.

Storage excluding rollout activations:

| Object | Order |
|---|---|
| S and physical basis Q | O(14 N), or streamed equivalents |
| Window bases U_w | O(sum_w T n_y r_w) |
| Compressed encoder matrix Z | O(r_G n_xi_p) |
| U_Z | O(r_G rank(E)) |
| Nuisance projection of S | O(14 rank(E)) |
| Off outputs if cached | O(N) |
| Dense N-by-N weight/projector | Prohibited in this scalar implementation |

Profile CPU/GPU peaks by stage: AD tangent batches, encoder factorization, graph creation,
checkpoint replay and optimizer state. Checkpointing addresses activations, not oversized
geometry matrices. Report compile latency, steady-state speed and graph breaks separately.
A shorter code-generation time or smaller generated file is not evidence of faster evaluation.

Execution order:

1. Select the actual checkpoint/configuration, resolve latent dimensions and routing, and save
   the manifest, anchor-off policy, raw-conversion reference and window-selection seed.
2. Run P0 on a disposable converted model using existing full production simulation. Check
   small directional derivatives and shared-encoder geometry before gate/closure engineering.
3. Review the predeclared rank/resource outcomes; implement verified ownership conversion and
   gates, then full/off/clamped production parity.
4. Build/test compressed geometry on the final G; confirm V2/V3 and off-independence.
5. Integrate the explicit gradient hook and verify a complete optimizer step and replay.
6. Measure cost and compiled parity; optimize observed bottlenecks without changing the stack.
7. Run the anchor-off paired pilot, including final-current geometry evaluation.

Initial G budget (design, not an adequacy result): four windows for P0, then a starting target
of max(32, number of represented training records) for the pilot, with at least one window per
record and remaining windows spread reproducibly over usable time/excitation. nf and B retain
the chosen prediction contract. Require sufficient valid windows and disclose unavoidable
history overlap; never silently duplicate windows to meet the count.

Before fixing pilot G, compare spectra/ranks and cost on nested small/expanded sets. Spaces
on different row sets cannot be compared by principal angles directly; use common evaluation
rows for any such comparison. If coverage, rank or memory is inadequate, revise the budget
and record the reason BEFORE paired training. Suggested initial rollout chunk size is four
windows, reduced for memory if needed; it must not redefine G or N.

Start eager with fixed shapes. A different regularization batch/horizon can trigger separate
compiled/CUDA-graph specializations; measure captures, compile latency and peak memory rather
than assuming one existing graph covers both prediction and penalty. For Gamma/S, initially
use forward JVPs because per-window differentiated dimensions (6 and 14) are usually fewer
than scored outputs. Validate and benchmark tangent batching against alternatives on this
closed-loop map; neither this heuristic nor the historical 145x figure guarantees speed.

No long gantry experiment should be launched merely to discover an already-testable gradient
or simulator contract error.

## 14. Single implementation-to-evidence index

| Statement / decision | Basis | Status / missing work |
|---|---|---|
| Zero-latent pointwise term misses affine latent readout | gaps 7b/9.5; nonlinear report blind-spot check | Structural example established; not a universal novelty claim |
| Delayed learned-state and controller effects enter rollout sensitivities | Eq. (21); report direct/recursive comparison | Derived for smooth maps; toy checked; gantry V2 open |
| Projection and profiling identities | Eqs. (16)–(19); gaps Section 9; report geometry checks | Derived, generic symbolic checks; data rank remains open |
| Local zero physical displacement condition | Normal equations (17)–(18) | Conditional local statement, not nonlinear joint recovery |
| Latent initialization belongs to penalized augmentation | User requirement and (7)–(8) | Design; current shared net needs verified conversion |
| Actual shared physical encoder is nuisance | Local compensation problem (17) | Conditional modeling choice; excludes ambiguous directions from protection |
| D-184 chain rule | Eq. (13), initialization-only dependency | Derived; dependency and production parity checks open |
| Compressed E basis | Factorization argument (15) | Derived proposed computation; rank truncation and memory unverified |
| MSE scalar weighting | Production composed loss; (10)–(11) | Code-inspected; runtime score/metadata parity required |
| Gate mechanism | Actual routing and Section 4.3 | Design; backward/replay/compiled parity open |
| Production rollout parity | fit_sys and closed_loop contracts | Required; old private-loop parity insufficient |
| Selective update | Eq. (22), D-181/D-182 | Design with correct eta derivative; stronger curl assertion not yet checked |
| Exact two-pass gradient | Eqs. (23)–(24); generic core/report checks | Derived; production closure and replay tests open |
| Latent scale invariance | Consistent coordinate change; toy check | Structural argument; configured pipeline test open |
| Total alignment can conceal route cancellation | Eq. (9) and linear projection | Derived; log both route projections |
| Nonzero hard-feasible correction | Nonlinear report, fixed-parameter solver example | Numerically checked only; no global optimum or gantry feasibility proof |
| Reference/current mismatch bound | Eq. (26), D-183 diagnostics | Derived; saved-reference drift must be measured correctly |
| Rank 2-of-3 at ANN-off | Nonlinear toy | Example-specific numerical result, not gantry evidence |
| Better physical estimates than shrinkage | Planned paired controls | Open |
| Exact orthogonality from finite beta | No implication supports it | Not claimed |
| Separate latent-route orthogonality or tangent separation | Not implied by total value constraint | Not claimed; (20) is a separate target |
| Burn-in magnitude decay removes encoder projector | Projector scale invariance contradicts this inference | Not used |
| Validation fit bounds encoder-induced parameter bias | No supporting bound | Not claimed |
| Bolderman anchor formula selects beta | Different regularization term | Rejected attribution |
| Primary anchor-off policy | Section 11 experimental isolation choice | Design; assert inactive term/gradient in every primary arm |
| Early rank triage | Section 6.5, product-range inclusion | Derived bound plus proposed probe; four windows are not final-G evidence |
| Nuisance-coordinate robustness | E'=E T^{-1}, Section 6.6 | Exact invariance derived; truncated range robustness unverified |
| Soft penalty adds curvature without removing directions | Eq. (27) | Derived at zero projected residual; no attainable-space separation claim |
| Tangent-confounding diagnostic | Eq. (28), supported-coordinate nuisance elimination | Derived; iterative solve accuracy and gantry outcome open |
| Full/off sensitivity difference | Eq. (29) | Derived; measure mechanism; training ablation conditional on target claim |
| Projection-versus-shrinkage confound | Section 11 named risk and matched size controls | Open efficacy risk; historical control equivalence must be checked |
| Raw-conversion reference and initial G budget | Sections 6.1/13 | Explicit design choices, not calibrated optimum or coverage proof |

A count of passing tests does not replace these scope distinctions. In particular, the
report's nonzero physical-gradient check is not a mixed-partial proof, and its rational
witness evaluations are not universal identities. Historical numerical tables remain in the
evidence records with their corrections; do not copy withdrawn values into a new result.

## 15. Source use and limitations

| Source / locator recorded in existing audit | Actual role | Limit |
|---|---|---|
| Györök 2025, Sections 2–4, Eqs. (3),(4),(9)–(19) | Stacked physical-regressor projection | Already state-space/encoder-based fitting; no recovery theorem for this recurrent penalty |
| Györök 2026 local arXiv v3, Sections 2–4 pp. 3–6, Condition 4/Theorem 7, conclusions pp. 9–10 | Conditional recovery; by-construction versus soft distinction | The by-construction tradeoff discussion does not prove finite penalties invalid; no unconditional gantry recovery |
| Kon 2022, Section IV pp. 4–6, Theorems 17/20 | Soft versus hard separation | Hard disjoint-optimization theorem does not transfer to selective recurrent training |
| Kon 2023, Sections 3.3–4 pp. 3–4, Eqs. (16)–(22) | Trajectory SVD projection and frozen-reference precedent | No proof frozen geometry stays adequate on this gantry |
| Hoekstra 2026, Table 1, Sections 5–6.3, Eqs. (16)–(17),(22),(26)–(27) | Dynamic augmentation, encoder initialization and multistep fitting | No efficacy theorem for the proposed penalty |
| Méndez-Blanco et al. 2021, Section 3, Eqs. (12)–(14), as assessed in the source audit | Initial-state nuisance/Schur geometry background | Equation (17) is derived here, not asserted to be a quoted identical encoder theorem |
| Bolderman 2024, Eqs. (19),(38), Remark 3.1 and tuning discussion | Physical-anchor scaling; separate network L-curve precedent | No closed-form trajectory-orthogonality beta |
| Plumlee/projected calibration literature in audit | Sensitivity-orthogonal discrepancy prior art and truth/calibration distinction | No unconditional physical recovery |
| Chain rule, SVD, normal equations and projection algebra | Derivations in this plan | No optimizer convergence or generalization theorem |

These locators refer to versions in the existing source audit; confirm final-publication
numbering before thesis submission. This revision did not reread every PDF. The elementary
derivations above do not require importing an uncited external theorem.

H2/Stein theory, Hankel reduction, hard constrained optimization and a globally orthogonal
causal architecture are outside the first implementation. Preserve them as separate research
questions rather than claiming them as mechanisms of this penalty.

## 16. Definition of done and revision scope

Ready for pilot means V0–V7 pass against the actual configured gantry, nuisance profiling
passes the applicable full/reduced-span decision in Section 6.5, cost fits the declared budget, and saved artifacts
reproduce the computation.

Efficacy requires V8 with appropriate controls. The stronger claim that physical parameters
remain true during joint training requires additional assumptions or evidence beyond contribution
orthogonality. Keep it explicit even if predictions and alignment look good.

This revision consolidates documentation and inspects code. It does not change production
training, run a geometry probe, or rerun MATLAB/SymPy. The encoder migration and compressed
construction are implementation requirements, not completed results.

The 2026-09-07 plan's whole-encoder profiling/detachment policy is superseded by (7)–(8):
trainable latent initialization is part of eta. Older D-184 dimension/cost comments must be read
with the per-window and parameter-ownership qualifications in Section 6. Existing evidence
records retain ownership of measurements; this document owns the revised implementation contract.

Critique follow-up, 2026-09-08: added P0 early geometry triage, explicit anchor-off primary
conditions, raw-conversion reference, accumulation-hook choice, initial G/chunk budgets,
clamp/off/clamped-mode assertions, coordinate-robustness checks, supported-coordinate tangent
diagnostics and current-geometry efficacy criteria. These are specification improvements;
none of the new probes, assertions or comparisons has been executed by this documentation edit.
