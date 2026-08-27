# Orthogonal Projection Regularization: Method and Approval Plan

Status: Stage A COMPLETE (2026-07-12), all diagnostics pass on pre-stated
criteria -- Steps 0-4 and 6 PASS, Step 5 reported (coverage sufficient, LOO
<= 1.7e-15 all bins). Awaiting GATE A user approval. Nothing enters
`model_augmentation/` until Stage A and Stage B are approved.
Results: `simulations/gantry_subnet/diagnostics/orth_projection/step<N>/`.

Sources used throughout:
- [GYOROK] Gyorok, Hoekstra, Kon, Peni, Schoukens, Toth, "Orthogonal projection-based
  regularization for efficient model augmentation", L4DC 2025.
  `literature/Orthogonality/Hoekstra - Orthogonal projection-based regularization for efficient model.pdf`
- [REPO] Their public implementation: `orthogonal-augmentation-main/`
  (key files: `model_augmentation/utils.py`, `model_augmentation/augmentation_structures.py`,
  `f1tenth_augmentation/car_models.py`, `scripts/orth_training.py`)
- [EJC] Hoekstra, Verhoek, Toth, Schoukens, "Learning-based model augmentation with LFRs",
  EJC 2025. `literature/augmentation/hoekstra2025_lfr-augmentation-ejc.pdf`
- Project code: `model_augmentation/fit_systems/blocks.py` (`Parameterized_Gantry_State_Block`),
  `scripts/gantry/gantry_dynamic/` (config, model, training).

Claims marked [SYNTHESIS] are our own reasoning, not stated in any source.

---

## 1. Problem statement

Joint estimation (trainable physical parameters theta alongside the ANN) is
non-identifiable: the additive structure lets the ANN reproduce dynamics the
baseline could supply by moving theta, so the (theta, ANN) split is not unique.
[GYOROK] Sect. 3 calls this non-identifiability; [EJC] Sect. 3.4 describes it as
"learning components canceling out part of the baseline model"; project term:
negation ([EJC] Sect. 4.3, "the ideal initializations are not negated").

Evidence that the existing Lambda anchor (`param_loss`, [EJC] Eqs. 6-7,
implemented in `Parameterized_Gantry_State_Block.param_loss`) does not prevent
negation:
- [GYOROK] Table 1 (noiseless): baseline at theta_0 = 37.91% NRMS; after
  co-estimation without projection the standalone baseline degrades to 118.01%;
  with projection 36.07%.
- [EJC] Sect. 4.3: with Lambda active and parameters near init, "the learning
  components are learning part of the system dynamics that could be represented
  by the baseline model".

Goal: add the orthogonal projection penalty Pi ([GYOROK]) to the gantry joint
estimation so the ANN is penalized exactly for output in the baseline's
parameter-response subspace, alongside the existing Lambda term.

Scope guard (D-103 untouched): Theta-only routing (`ann_route_ix=(1,4,6,7)`) is
the machinery-development and validation environment, not a deliverable
configuration. All projection code takes the routed-row set from
`cfg.ann_route_ix`; the later X+Theta+Y extension must be a config change, not
surgery.

---

## 2. Theoretical method

### 2.1 Which variant of [GYOROK] applies: the Section 4 (Taylor) variant

The Section 3 (direct) variant requires the baseline to be exactly linear in
the parameters ([GYOROK] Eq. 8): f_theta(x,u) = phi(x,u) * theta. This fails
for the discretized gantry baseline on two counts, both established from code:

1. Masses and inertias (mh, m1, m2, mb, Jb, Jh) enter through M(Y)^-1: the
   explicit state derivative is M(Y)^-1 (-C ẋ - K x + F u), rational in these
   parameters (`blocks.py`, `nonlinear_function` rebuilds d0 = det(M0) and the
   rational structure per step).
2. The block integrates with RK4 substeps (`blocks.py` lines ~763-768,
   up_sample substeps per sample): the discrete map composes the dynamics,
   so even the stiffness/damping parameters (linear in the continuous-time
   descriptor form) appear polynomially in the discrete transition.

Note: LPV (Y-scheduling) is NOT the reason the Taylor variant is needed. The
decision criterion in [GYOROK] is nonlinearity in theta only. Y-dependence is
handled by per-sample evaluation (Sect. 2.3 below).

Taylor and SVD are not alternatives: both variants end in the same SVD-based
projection. The Taylor expansion only changes which matrix the SVD is taken of
([GYOROK], sentence after Eq. 18: "instead of taking the reduced SVD of
Phi(X,U), now it should be calculated for Phi_tilde(X,U)").

### 2.2 The method, step by step

All at linearization point theta_bar = theta_0 (the nominal/init values;
[GYOROK] p. 8 argues theta_0 is a valid fixed choice, enabling one-time
precompute). Let R = the routed row set from `cfg.ann_route_ix`, n_r = |R|.

1. Simulate the FP baseline at theta_0 on the training records to obtain the
   state trajectory X_hat ([GYOROK] end of Sect. 3: when full-state
   measurements are unavailable, use FP-simulated states; [REPO]
   `calculate_orthogonalisation` implements exactly this fallback).
2. Per sample k, compute the parameter Jacobian of the discrete transition,
   restricted to routed rows:
       Phi_k = d f_theta(x_k, u_k) / d theta |_(theta_0)   in R^(n_r x 14)
   by autograd through `Parameterized_Gantry_State_Block` (the block is already
   differentiable w.r.t. `log_params`; [REPO] `car_models.py` __main__ block
   demonstrates `torch.autograd.functional.jacobian` on the FP model).
   Parameterization invariance [SYNTHESIS]: d f / d log(theta) = theta * d f / d theta
   is a nonzero column scaling; column span, hence Pi, is identical. We may
   differentiate w.r.t. log_params directly.
3. Per sample k, the offset column ([GYOROK] Eqs. 16-17):
       Gamma_k = f_theta0(x_k, u_k) - Phi_k * theta_0   in R^(n_r)
4. Stack sample-major over all N training samples ([REPO] convention,
   `calculate_orth_matrix`: row n_r*k + i):
       Phi_tilde = [Phi | Gamma]   in R^(N*n_r x 15)
5. Reduced (economy) SVD once, before training ([GYOROK] Eq. 10, Remark after;
   [REPO] `torch.linalg.svd(Matrix, full_matrices=False)`):
       Phi_tilde = Q Sigma V^T,   Q in R^(N*n_r x 15)
   Rank truncation (ours; [GYOROK] assumes full column rank, Eq. 10 context):
   our stack is exactly rank-deficient (four structural degeneracies, Step 4
   prediction), so keep only columns with sigma_i > 1e-12 * sigma_0
   (standard numerical-rank criterion). Penalizing along the ~1e-16 noise
   directions would penalize legitimate ANN output. Measured (Step 2):
   sigma[10]/sigma[0] ~ 4e-6, sigma[11]/sigma[0] ~ 4e-18; any threshold in
   between yields the same 11 columns.
6. Penalty term added to the training loss ([GYOROK] Eqs. 13-14, 19; [REPO]
   `calculate_orthogonalisation` in `augmentation_structures.py`):
       V_orth = beta * || Q^T f_ANN(X_hat, U) ||^2
   where f_ANN(X_hat, U) is the ANN evaluated on the same fixed point set,
   restricted to routed rows, stacked with the same convention.
7. Total loss: existing SUBNET truncated loss + existing Lambda param_loss +
   V_orth. Lambda and Pi coexist ([REPO] `fit_system.py` loss: mse +
   (parm_regularization + orthCost)/N_batch_updates_per_epoch).

Frame convention (must be consistent, our delta from [REPO]): [REPO] evaluates
both the regressor and the ANN output in physical units (`calculate_xnet`
de-normalizes with Tx_inv before projecting). Our interconnect and routing live
in the normalized frame, so we build Phi_tilde AND evaluate f_ANN in the
normalized frame. What matters is that both sides use the same frame
[SYNTHESIS, but forced by the same-frame requirement implicit in the inner
product Q^T f_ANN].

### 2.3 LPV handling: why precompute + one SVD stays valid

The subspace is NOT a projector in the per-sample state space built at one
frozen Y. It is the column span of the stacked matrix Phi_tilde in R^(N*n_r):
column j is the signature trajectory of parameter j over the whole dataset,
evaluated at each sample's actual state, and Y is a state. The Jacobian at
sample k is therefore automatically evaluated at that sample's operating point;
the Y-dependence of the parameter directions is contained inside the columns.
Only theta_bar is frozen at precompute time, never Y.

[SYNTHESIS] Why one global Pi is the principled default: negation replaces
(theta_0 + delta_theta, f_ANN) by (theta_0, f_ANN + Phi * delta_theta) with
delta_theta a constant vector. The compensating ANN output is therefore a
global signature Phi * delta_theta over the entire dataset, which is exactly a
direction in span(Phi_tilde). An ANN output mimicking a parameter effect only
in one Y-region corresponds to no constant-theta change, is not negation of
this baseline, and is correctly left unpenalized by the global Pi. A Y-binned
Pi is stricter (it also penalizes legitimate Y-local dynamics that locally
resemble parameter effects) and becomes a thesis comparison experiment, not a
validity requirement.

Precondition (data, not formula): the training records must sweep Y over the
operating range; otherwise span(Phi_tilde) only encodes the visited operating
points and the projection is blind to parameter mimicry at unvisited Y.
Checked by diagnostic D5, not by changing the method.

[SYNTHESIS] Row restriction, the two-construction choice: restricting the
regressor ROWS to `ann_route_ix` and then taking the SVD is NOT the same
projector as taking the full-space SVD and projecting the expanded ANN output;
they agree exactly on the negation-feasible directions (theta drifts whose
signatures are supported on the routed rows -- the only drifts the ANN can
compensate; off-row signature components are blocked by the data fit itself)
and differ on mixed signatures, where restrict-then-SVD is strictly more
conservative (bounded by the soft beta). We take restrict-then-SVD. Status of
the reference code: CONSISTENT WITH this choice (its regressor rows equal the
ANN-written rows, nx=3 velocity-only model), but it cannot adjudicate the
choice -- its excluded rows (kinematic positions) carry zero parameter
signature, so both constructions coincide there. Our excluded X/Y rows DO
carry signatures; the choice and its justification are ours (thesis
extension 3). Rows 6,7 (absorber) have zero baseline signature, so the
restricted stack automatically leaves ANN output there unpenalized -- the
theoretically desired behavior.

### 2.4 Known deltas versus [GYOROK]/[REPO]

| Aspect | [GYOROK]/[REPO] | Ours | Why |
|---|---|---|---|
| Jacobian | hand-derived symbolic, hardcoded per system | autograd through the block | 14 params, RK4, M(Y)^-1 make symbolic derivation infeasible; [REPO] itself demos autograd jacobian |
| Rows | full state (ANN writes all states) | restricted to `cfg.ann_route_ix` | stiffness-selective routing; the penalty must live in the space the ANN can write |
| Frame | physical units | normalized frame | interconnect and routing are normalized; consistency is what matters |
| Scheduling | none (LTI-structure case study) | Y enters per sample | Sect. 2.3 |
| Penalty point set | full fixed training set every batch (their N approx 8e3), optional mini_batch_size subsampling | OPEN: decimated fixed set vs subsampling | our records are much longer; decision in Sect. 5 |
| theta_bar | theta_0 fixed, precompute once (per-epoch recompute = [GYOROK] Remark 2, optional) | same default; Remark 2 as fallback | Lambda anchor holds theta near theta_0 |

---

## 3. Diagnostics: Stage A (subspace correctness, standalone)

Rules of the game: every diagnostic constructs Phi_tilde from scratch (no
trained model, no full pipeline), states its falsifiable prediction BEFORE the
run, and saves data + prediction-vs-measurement plots under
`simulations/gantry_subnet/diagnostics/orth_projection/`. A diagnostic passes
only on its stated criterion; no post-hoc threshold adjustment.

### D1: Autograd vs finite-difference Jacobian
- What: column j of Phi from autograd vs central finite differences on
  theta_j, at a spread of samples (different Y, different velocities).
- Prediction: relative error per column < 1e-4 (float64) at FD step near the
  central-difference optimum ~ eps^(1/3) (two-term balance, truncation O(h^2)
  vs round-off O(eps/h); Numerical Recipes Sect. 5.7); error curve vs FD step
  shows the expected V-shape (truncation vs roundoff).
  [CORRECTED 2026-07-12: earlier draft said "sqrt-eps", which is the FORWARD
  difference rule; the script uses central differences.]
- Catches: wiring errors (parameter ordering, log vs physical mix-ups,
  wrong row selection), autograd graph breaks.

### D2: Perturbation-reconstruction (the formula check)
- What: for small delta_theta in random directions, compute the true response
  change Delta_f = f_(theta_0 + delta) - f_(theta_0) stacked over the same
  samples; measure the out-of-subspace fraction
  r = ||(I - Pi_tilde) Delta_f|| / ||Delta_f||.
- Prediction: r scales quadratically in ||delta_theta|| (pure Taylor
  remainder): slope 2 on log-log over >= 3 decades of ||delta_theta||.
  Slope 1 means the Jacobian or the offset column is wrong.
- This is the strongest single check: it tests Phi, Gamma, stacking, SVD, and
  Pi together against the true nonlinear model.

### D3: Rank vs identifiability
- What: singular value spectrum of Phi_tilde.
- Prediction (stated before computing): a cliff after approximately 11
  significant values (10 identifiable combinations per D-077 + 1 offset
  column); the kb1/kb2 columns identical, the cb1/cb2 columns identical
  (K and C depend only on the sums; `blocks.py` `_build_KC`).
- A different rank falsifies either the regressor construction or the D-077
  identifiability analysis; both outcomes are informative.

### D4: Projector sanity
- What: ||Q^T Q - I||, ||Pi^2 - Pi||, symmetry of Pi (on a decimated stack if
  the full Pi is too large; the Q-based checks need no explicit Pi).
- Prediction: machine-precision small. Catches reshaping and numerical bugs.

### D5: Y-coverage (the data-derived scheduling diagnostic)
- What: split the training samples into Y bins; build per-bin stacked
  regressors Phi_tilde^(b); measure how much of each bin's subspace is
  captured by the global Q: for each bin, the residual
  rho_b = ||(I - Q Q^T) Phi_tilde^(b)||_F / ||Phi_tilde^(b)||_F,
  and the principal angles between span(Phi_tilde^(b)) and span(Q).
- Prediction: if the training Y-sweep covers the operating range, rho_b stays
  small and roughly uniform across bins, including bins near the held-out Y
  positions used in validation. A bin with large rho_b marks a Y-region where
  the projection is blind; the remedy is data (extend the Y sweep), never a
  formula change.
- This diagnostic is empirical by construction: it uses only the training
  records and the theta_0 FP model, no oracle quantities.
- [MEASURED OUTCOME + FORMULA CORRECTION, 2026-07-12: the column-space rho_b
  as specified above (ours, not the paper's) turned out to be dominated by a
  trivial support-fraction artifact: a bin's embedded columns can align with
  the globally-spread Q columns only up to ~sqrt(n_b/N), so rho_b ~ 0.92-0.98
  tracks sample counts, not coverage (verified: bin 5 predicted 0.911 vs
  measured 0.923; bin 7 predicted 0.984 vs 0.974). The informative coverage
  statement is the ROW-SPACE leave-one-bin-out residual: project each bin's
  regressor rows onto the row-space basis (V) of the all-other-bins stack.
  Measured: rho_b_LOO <= 1.7e-15 in every one of 12 bins -- no Y-region
  contributes an irreplaceable direction; the 11-dim identifiable structure
  is Y-uniform and the projection extrapolates across the sweep. Combined
  with every bin being represented in the decimated penalty subset (blocking
  check PASS) and held-out Y positions (-0.10, +0.10, +0.22) in
  well-populated bins: coverage is sufficient; D5 raises no data action.]

### D6: Synthetic negation detector (penalty end-to-end)
- What: construct two fake ANN output fields on the fixed point set:
  (a) f_ANN := Phi * delta_theta for a known delta_theta (pure negation);
  (b) f_ANN := a field projected into the orthogonal complement (pure
  legitimate output).
- Prediction: penalty ratio ||Q^T f||^2 / ||f||^2 approx 1 for (a) and approx 0
  for (b). This is the negation detector tested on synthetic negation before
  it ever sees training.

Stage A exit criterion: D1-D4 and D6 pass on their stated predictions; D5
reported with its Y-coverage map (D5 informs data design, it does not block the
machinery). User approves before Stage B.

---

## 4. Stage B (hook design) and Stage C (validation runs)

Stage B, only after Stage A approval:
- Verify how the training loss is assembled in Jan's
  `model_augmentation/fit_systems/interconnect.py` (the `SSE_Interconnect_ParamLoss`
  subclass already sweeps blocks for `param_loss`; confirm the same
  non-invasive pattern carries a `projection_loss` that needs the fixed
  (X_hat, U) tensors and the ANN block forward). Three-pipelines rule: check
  `lpv_lfr_baseline/blocks/lfr_fit_system.py` (D-032) for the existing hook
  pattern before touching anything.
- Deliverable: a written hook design (which subclass, which tensors cached
  where, config knobs: `orth_beta`, penalty point set size), approved before
  any edit to `model_augmentation/`. Markers per CLAUDE.md tracking rules when
  edits happen.
- Unit test: D6 rerun through the actual hook (same numbers expected).

Stage C, pre-declared validation pair (D-090 rows in the problem log BEFORE
launch):
- Config: joint_estimation=True, detuned start (`param_init_detune`, run-D
  style), Theta routing (1,4,6,7), lr re-checked for this configuration (the
  current CFG lr=1e-7 belongs to the X/Y de-confound run, not this one).
- Run pair: identical except beta=0 vs beta swept over decades (bracket from
  [GYOROK] Fig. 3: 1e-7..1e-5 as starting range only, explicitly not
  transferable).
- Criteria (both on simulation data where truth is known):
  1. Recovery error of the 10 identifiable combinations vs truth
     (`param_table`).
  2. Standalone-baseline test: strip the ANN, run the FP model alone with the
     learned theta_hat on held-out data; compare NRMS vs the theta_0 baseline
     ([GYOROK] Sect. 5.4 evaluation; "standalone-baseline test" is our project
     term).
- Expected signature if the method works: beta=0 shows degraded standalone
  baseline (negation), swept beta restores it without destroying augmented
  accuracy ([GYOROK] Table 1 pattern).

---

## 5. Open decisions (to settle during Stage A/B, logged to decisions.md)

1. Penalty point set for long records: decimated fixed subset vs [REPO]-style
  random mini-batch per SVD; and its size. Constraint: the point set must
  retain the Y sweep (D5 rerun on the chosen subset).
2. beta sweep design and selection rule (selected on the Stage C criteria, not
  on training loss).
3. Whether Remark 2 (per-epoch recompute of theta_bar and the SVD) is needed:
  triggered only if Stage C shows the projection losing grip as theta moves.
4. Y-binned Pi comparison experiment: thesis-level, after the global-Pi
  machinery is validated.

---

## 6. Goals: what "approved" means

The orthogonality regularization is approved for the FP model when:
1. Stage A diagnostics pass on pre-stated predictions (formulas verified
   against the true nonlinear model, rank consistent with identifiability,
   penalty detects synthetic negation).
2. Stage B hook design approved and unit-tested without invasive framework
   edits.
3. Stage C shows, on the detuned-start joint estimation: standalone-baseline
   NRMS with projection at or better than the theta_0 baseline, while the
   augmented model retains its accuracy advantage; both vs the beta=0 control.
4. All decisions logged (decisions.md), run rows in the problem log (D-090).

---

## 7. Stepwise validation ladder (execution plan)

Ordering principle: dependency order, not paper order. Each step's machinery is
an input to the next; a step starts only when the previous step's pass
criterion is met and, at the two gates, when the user has approved.

Conventions for every step:
- Scripts live in `scripts/gantry/orth-projection/`, one script per step,
  runnable standalone (each constructs what it tests from scratch; no trained
  model, no full pipeline run required).
- Results (JSON/npz + figures) go to
  `simulations/gantry_subnet/diagnostics/orth_projection/step<N>/`.
- Every figure is prediction vs measurement with the deviation quantified; the
  pass criterion is stated in the script header and printed with the result.
- Precision: all Stage A diagnostics run in float64 (`use_f64` equivalent);
  the pipeline's float32 behavior is checked once, in Step 8.
- Data: the training records loaded exactly as `gantry_dynamic/data.py` loads
  them (fs_new = 4000 Hz, same normalization via `compute_normalization`);
  theta_0 = nominal values from `model_augmentation/systems/gantry_ss.py`.

### Step 0: Fixed point set (X_hat, U)
- Build: simulate the FP baseline at theta_0 over the training records
  (normalized frame, `up_sample` as in the pipeline) and store the point set
  (X_hat, U) that all later steps consume.
- Verify: the rollout must reproduce the existing pipeline baseline. Cross-check
  the simulated output trajectory against `compute_baseline_fp_nrms`
  (`gantry_dynamic/baselines.py`) on the same record, same init, same start
  index.
- Goal achieved when: max abs deviation between the two output trajectories is
  at float64 round-off level (prediction: < 1e-10 relative; both code paths
  implement the same rollout).
- Why first: if the point set is built with a different convention (frame,
  up_sample, sample alignment) than the pipeline, every later diagnostic
  validates the wrong object.

### Step 1: Parameter Jacobian at one sample (D1)
- Build: Phi_k for a single sample via `torch.autograd.functional.jacobian`
  through `Parameterized_Gantry_State_Block` w.r.t. log_params, converted to
  physical-theta columns; central finite differences as reference.
- Run at >= 5 samples spread over the record (different Y, different velocity
  signs) and for all 14 columns.
- Goal achieved when: per-column relative error < 1e-4 at the sqrt-eps FD
  step, and the error vs FD-step curve shows the V-shape (truncation left,
  round-off right). A flat or monotone curve fails the step.

### Step 2: Stacked regressor, offset column, SVD, projector (D4)
- Build: Phi_tilde = [Phi | Gamma] over the point set (row convention
  n_r*k + i, routed rows from `cfg.ann_route_ix`), economy SVD -> Q.
- Point set size: start with a decimated subset; measure wall time and memory
  for the full set BEFORE deciding the production size (no cost claims without
  measurement; log the measured numbers in the step output).
- Goal achieved when: ||Q^T Q - I||_max < 1e-12, projector idempotency and
  symmetry at the same level (via Q-based identities, no explicit Pi), and the
  stacking convention is verified by reconstructing one sample's Phi_k block
  from the stack and matching it to Step 1's output exactly.

### Step 3: Perturbation-reconstruction, the formula check (D2)
- Build: for >= 10 random unit directions dir in RELATIVE theta-space
  (theta = theta_0 (1 + eps * dir), realized exactly via
  log_params = log(1 + eps * dir)) and eps swept over >= 3 decades, compute
  the true stacked response change Delta_f = f_(theta) - f_(theta_0) on the
  decimated point set, restricted rows, same stacking as Q.
- [CORRECTED 2026-07-12, before the diagnostic ran: the original criterion
  mixed the absolute residual and the relative fraction. Taylor:
  Delta_f = Phi delta + O(delta^2); the projector kills the first-order term
  exactly, so the ABSOLUTE residual ||(I - Q Q^T) Delta_f|| ~ eps^2 (slope 2),
  while the RATIO r = ||(I - Q Q^T) Delta_f|| / ||Delta_f|| ~ eps (slope 1),
  since ||Delta_f|| ~ eps. A wrong Jacobian/offset gives absolute slope 1 and
  a PLATEAUING ratio (slope 0).]
- Goal achieved when, for all directions, over eps in [1e-5, 1e-2]:
  P1: log-log slope of the absolute residual ||(I - Q Q^T) Delta_f|| vs eps
      is 2.0 +/- 0.1 (pure Taylor remainder);
  P2: log-log slope of the ratio r vs eps is 1.0 +/- 0.1 (no plateau);
  P3: r at eps = 1e-2 is < 0.1 (HEURISTIC sanity bound: second-order/first-
      order ~ eps for a smooth reparameterized map, so r ~ 1e-2 expected;
      r ~ O(1) would mean the subspace misses first-order content).
  Slope ~1 in the absolute residual (equivalently a ratio plateau) in any
  direction fails the step (wrong Jacobian or offset column).

### Step 4: Rank vs identifiability (D3)
- Build: singular value spectrum of Phi_tilde from Step 2.
- Prediction stated now, before computing (refined 2026-07-12 for the
  ROW-RESTRICTED stack, from the gantry_ss.py structure, lines 62-166):
  cliff after EXACTLY 11 significant values. Basis: four exact column
  degeneracies reduce 14 parameter columns to 10 independent, + 1 offset:
    (i)   kb1 ~ kb2      (K carries only kb1+kb2)
    (ii)  cb1 ~ cb2      (C carries only cb1+cb2)
    (iii) Jb ~ Jh        (gamma depends only on Jb+Jh, gantry_ss.py line 147)
    (iv)  mb in span{m1, m2, Jb}  (mb enters only through alpha; the
          alpha-direction is reachable from m1, m2, Jb columns since gamma
          carries (m1+m2)*Lb^2/4)
  Validity under row restriction: M0[1,2] = -mh*d and beta couple Y- and
  X-row forces into the Theta row (M(Y)^-1 is full), so EVERY parameter,
  including cy, retains Theta-row content. An earlier hypothesis that cy
  drops under restriction is REFUTED by this structure (checked before any
  spectrum was computed).
- Goal achieved when: the spectrum shows a gap of >= 3 orders of magnitude
  after the predicted count, and the predicted column collinearities hold. If
  the count differs, STOP: either the regressor or the D-077 analysis is
  wrong; diagnose which before proceeding (both outcomes are informative, but
  neither may be waved through).

### Step 5: Y-coverage map (D5)
- Build: bin the point set by Y (>= 8 bins over the training sweep), per-bin
  stacked regressor Phi_tilde^(b), and the ROW-SPACE leave-one-bin-out
  residual rho_b_LOO (see D5 [FORMULA CORRECTION 2026-07-12]: the
  column-space rho_b originally specified here is a support-fraction
  artifact, not a coverage measure, and rho_b_LOO replaces it as the
  criterion). The column-space rho_b and principal angles vs span(Q) are
  still computed and reported, for the record only.
- Goal achieved when: the rho_b map is produced and reported, including bins
  nearest the held-out Y positions. This step INFORMS (data design, penalty
  point-set decimation in Sect. 5.1) but does not block: a large rho_b in some
  bin is a data finding, remedied by extending the Y sweep, not by changing
  formulas. It blocks only the choice of a decimated penalty set that would
  drop a covered bin.

### Step 6: Synthetic negation detector (D6)
- Build: on the point set, two synthetic ANN output fields:
  (a) Phi * delta_theta for known delta_theta (pure negation);
  (b) a random field with its span(Q) component removed (pure legitimate).
- Goal achieved when: penalty ratio ||Q^T f||^2 / ||f||^2 > 0.99 for (a) and
  < 1e-6 for (b), across >= 10 random draws of each.

### GATE A (user approval)
Deliverable for review: the six step results (figures + numbers), the measured
cost of the full-set SVD from Step 2, and the proposed penalty point set
(size + decimation, respecting Step 5's coverage map). No `model_augmentation/`
edit has happened yet. Approval unlocks Stage B.

### Step 7 DELIVERABLE (2026-07-12): the hook design, for approval

Read basis: `model_augmentation/fit_systems/interconnect.py` lines 433-457
(SSE_Interconnect.loss: encoder -> nf rollout -> mse + isinstance-based Lambda
for Jan's own parameterized blocks) and lines 723-747
(SSE_Interconnect_ParamLoss, @added: super().loss() + hasattr sweep of
param_loss, exact no-op when absent). Three-pipelines check: no data-dependent
loss term exists in any pipeline; new machinery is required and follows the
ParamLoss pattern.

D7.1 Placement: ONE new file `model_augmentation/fit_systems/orth_projection.py`
  with `__project_origin__ = "added"` (whole-file ownership marker), containing:
  - `OrthProjectionPenalty(nn.Module)`: buffers Q (rank-truncated, cast to
    pipeline dtype), Z_pts (the fixed normalized penalty inputs
    [x_phys, x_aug=0, u], see D7.4), `route_cols` (which ANN output columns
    map to rows < 6, preserving ann_route_ix order), scalar beta.
    `forward(ann_block)`: evaluate the ANN on Z_pts (one batched forward),
    take route_cols, stack sample-major, return beta * ||Q^T f||^2.
  - `SSE_Interconnect_OrthLoss(SSE_Interconnect_ParamLoss)`: attribute
    `orth_penalty = None`;
    loss() = super().loss() + (self.orth_penalty(ann_block) if
    self.orth_penalty is not None else 0).
    Used unconditionally by build_model (mirrors the D-076 ParamLoss
    precedent); `orth_penalty` stays None unless cfg.orth_beta > 0.

D7.2 No-op guarantee: with orth_penalty None (default; cfg.orth_beta = 0) the
  loss() code path is identical to SSE_Interconnect_ParamLoss. Verified by
  Step 8 parity check 2 (bit-identical short run).

D7.3 Config knobs (RunConfig): `orth_beta: float = 0.0` (0 = off),
  `orth_point_stride: int = 100` (penalty point set decimation; Step 5
  coverage verified at 100), `orth_rank_tol: float = 1e-12`.

D7.4 Penalty point set at training time: built by a pipeline-side function
  `build_orth_penalty(cfg, data, norm)` in `gantry_dynamic/`.
  [REVISED 2026-07-12, D-111, after Step 7b FAIL + Step 7c ablation:]
  States are DATA-DERIVED (q = P^-T y exact static inversion, qdot by FD --
  the data.py construction), NOT FP-simulated at theta_bar as originally
  designed. Measured basis: rollout states leaked up to 0.164 of true
  negation-signature energy (7b: open-loop drift on K=0 axes at detuned
  theta_bar dominates); data/truth states reduce this to the theta_bar-only
  level 3.8e-3..1.7e-2 (7c), matching the curvature prediction. This is the
  paper's PRIMARY full-state-measurement setting ([REPO] x_meas=True), not
  its simulated-states fallback (p. 7), whose stay-near-the-data assumption
  fails for marginally stable systems. Build: Jacobian stack at theta_bar on
  the data-derived points (measured ~6 min at stride 100), SVD (0.03 s);
  cached to an npz keyed by (mode, fs_new, stride, route_ix, up_sample,
  theta_bar hash, states='data'). ANN inputs at the penalty points use
  x_aug = 0 (no pre-training absorber estimate; the baseline regressor does
  not depend on the absorber rows). Remark-2-style refresh (rebuild at
  current theta/states) stays the documented escalation, not the default.

D7.5 theta_bar = the run's params_init (the Lambda anchor: nominal, or the
  DETUNED init in recovery runs). Never the simulation truth: in the Stage C
  recovery test the experimenter's knowledge is the detuned init; using the
  true values would smuggle oracle information into the regularizer. Stage A
  validated the machinery at nominal theta_0; all scripts take theta_bar from
  the block's params_init, so this is a parameter choice, not new code.

D7.6 Loss-scale delta vs [REPO]: the reference divides the (full-fixed-set)
  penalty by N_batch_updates_per_epoch so it counts once per epoch; our
  param_loss precedent adds its term every batch undivided. We follow OUR
  precedent (penalty added per batch, undivided): the constant factor is
  absorbed into the swept beta, and the penalty then enters the loss exactly
  like the existing Lambda term (including any sqrt_train treatment).
  Flagged so the beta bracket is not read as transferable from [GYOROK].

D7.7 Frames and dtype: Q was built from Jacobians of the block's NORMALIZED
  transition (Step 2), and the ANN input/output live in the same normalized
  interconnect frame; no conversion enters the penalty. Q is computed in f64
  (Stage A) and cast to the pipeline dtype (f32) in the buffer; Step 8 parity
  runs at pipeline dtype.

D7.8 Checkpoint impact: the penalty module's buffers add ~1 MB (Q at f32) to
  the saved system; resume of pre-hook checkpoints must still work (Step 8
  check; orth_penalty is attached after construction, not read from old
  checkpoints).

D7.9 beta determination (added 2026-07-12 after checking both sources).
  Status in the sources: NO principled rule exists. [GYOROK] Sect. 5.3 tunes
  beta empirically over decades (Fig. 3, judged on TEST NRMS; beta = 1e-7
  noiseless, range [1e-7, 1e-5]); Remark 1 notes several decades work.
  [REPO] hardcodes `beta = 1e-7` in orth_training.py (USER DEFINITIONS);
  no sweep logic exists; the penalty is an UN-normalized sum over their
  ~8e3-point set divided by batches/epoch, so their number is welded to
  their dataset size, units, and conventions -- non-transferable.
  Our rule (ours, not the sources'):
  (1) Bracket center [HEURISTIC, Lambda-analogous dimensional argument in
      the spirit of Bolderman/EJC Eq. 7]: an ANN absorbing a permitted-size
      parameter drift should pay a penalty comparable to the mse it can fake:
          beta_center ~ V_MSE(theta_init) / E_drift,
          E_drift = mean ||Phi_tilde [delta; 0]||^2 over random relative
          drifts delta of the anchor scale (10%, the detune magnitude),
      both computable from saved Stage-A artifacts + the baseline mse; no
      truth needed. Sweep >= +-2 decades around beta_center.
  (2) Selection rule, pre-declared and truth-free (improves on the sources'
      select-on-test practice): choose beta on VALIDATION data only:
      augmented val NRMS within a pre-declared margin of the beta=0 run, and
      standalone-baseline val NRMS best / no worse than FP at theta_init.
      Simulation truth (parameter recovery error) is used ONLY to verify the
      selection afterwards, never to make it.
  (3) Note: the penalty and its gradient are exactly zero at the ANN's
      zero init (quadratic in f_ANN), so beta does not distort early
      training; it prices absorption as it emerges.

Approval of this design (GATE: user) unlocks the Step 8 implementation.

### Step 7b: theta_bar sensitivity (the "wrong parameters" robustness check)

Empirical test of D7.5: how much does the projection lose when the subspace
is built at the experimenter's (detuned) parameter knowledge instead of the
truth? This is exactly the Stage C production situation.

- Build the full PRODUCTION basis at the detuned anchor: FP rollout of the
  point set at theta_d = theta_0 * detune (the RunConfig default 14-vector,
  +-10%), Jacobian stack at theta_d on the detuned states, SVD, rank-truncate
  -> Q_d. (Both the states and the linearization point are detuned: the
  production object, not a half-way construction.)
- Compare against the Stage-A truth basis Q (Step 2): principal angles
  between the two 11-dim column spaces (same row indexing).
- Crossed Step 3: perturbation responses generated around the TRUTH on the
  truth manifold (as in Step 3), projected against Q_d. The residual
  fraction r(eps) can no longer fall to zero; it floors at the
  mismatch-induced level.
- Pre-stated prediction (from measured Step-3 curvature, ratio 3.5e-4 at
  eps = 1e-2 -> curvature scale c2/c1 ~ 0.035 per unit eps; mismatch 0.1):
      r_floor ~ 3.5e-3, expected band [1e-3, 1e-2];
      max principal angle O(0.2 deg), pass < 5 deg.
  PASS: r_floor < 5e-2 for all directions (detuned basis still catches
  > 99.75% of true first-order signature energy) and angles < 5 deg.
  FAIL consequence: Remark 2 (per-epoch recompute) is promoted from fallback
  to requirement before Stage C.

  [OUTCOME 2026-07-12: FAIL as originally built -- max angle 56.7 deg, floors
  3.4e-2..1.64e-1. The prediction modeled the wrong perturbation: it covered
  the linearization point only, while the production recipe also re-simulated
  the STATES at theta_d (open-loop drift on K=0 axes dominates).
  Step 7c ablation (same theta_d linearization, truth-manifold states):
  max angle 11.1 deg, floors 3.8e-3..1.69e-2 -- the theta_bar-only cost lands
  in the predicted band; state drift was the dominant contributor.
  RESOLUTION (D-111): penalty states switched to data-derived (P^-T y + FD),
  which eliminates the dominant contributor, realizes the paper's primary
  x_meas setting, and keeps precompute-once. Remark 2 remains escalation
  only. Residual accepted risk: theta_bar-only leakage <= 1.7e-2 on random
  directions at 10% detune; worst-aligned direction ~sin(11 deg) ~ 0.19.]

### Step 7: Hook design on paper (original spec)
- Read the loss assembly in `model_augmentation/fit_systems/interconnect.py`
  and the existing non-invasive pattern in
  `lpv_lfr_baseline/blocks/lfr_fit_system.py` (D-032) and
  `SSE_Interconnect_ParamLoss` (three-pipelines rule: confirm nothing existing
  already carries a data-dependent loss term).
- Deliverable: a short written design in this document (new subsection):
  which subclass, where the fixed (X_hat, U) tensors and Q are cached, the
  config knobs (`orth_beta`, point-set size), the exact loss line
  (mirroring [REPO]: mse + (param_loss + orth_cost)/N_batches), and the
  no-op guarantee (beta = 0 must be bit-identical to the current pipeline,
  same property `SSE_Interconnect_ParamLoss` has for absent param_loss).
- Goal achieved when: the user approves the written design. Only then is code
  written into `model_augmentation/`, with the CLAUDE.md ownership markers.

### Step 8: Hook implementation + parity checks
- Implement per the approved design.
- Three checks, all must pass:
  1. D6 parity: the synthetic negation detector rerun THROUGH the hook
     reproduces Step 6's numbers (same point set, same Q) to float32
     round-off.
  2. No-op regression: a short training run (few epochs, fixed seed) with
     beta = 0 produces bit-identical loss trajectory to the same run on the
     pre-hook code.
  3. Gradient flow: with beta > 0, `compute_gradient_norms` shows a nonzero
     gradient contribution from the penalty into the ANN parameters and zero
     into theta (the penalty is a function of the ANN only; theta enters only
     through the precomputed Q).
- Goal achieved when: all three pass. Any training run here uses the existing
  entry point (`gantry_interconnect_dynamic.py`) with its per-epoch
  train/val nf-RMS printing active; no new training scripts.

### Step 9: Smoke run with penalty active
- One short training run (Theta routing, joint_estimation=True, detuned start,
  lr re-swept for THIS configuration, beta at the middle of the [GYOROK]
  bracket) purely to observe behavior: penalty magnitude relative to mse and
  param_loss logged per epoch, no NaN, no optimizer collapse.
- Goal achieved when: the run completes with all three loss components
  finite and the penalty term visibly responding to training (changing, not
  frozen). Model quality is explicitly NOT judged here.

### GATE B (user approval)
Deliverable: parity results, smoke-run loss decomposition. Approval unlocks
Stage C.

### Step 10: Pre-declared validation pair (Stage C)
- D-090 rows in the problem log BEFORE launch, one per run: hypothesis for
  the beta = 0 control ("detuned joint estimation negates: standalone-baseline
  NRMS degrades vs theta_0 baseline") and for the beta sweep ("projection
  restores standalone-baseline NRMS without destroying augmented accuracy").
- Config: identical pair except beta; joint_estimation=True, detuned init,
  Theta routing, epochs/nf per the then-current known-good training config.
- Judged on (both from simulation truth):
  1. Identifiable-combination recovery error (param_table) vs truth.
  2. Standalone-baseline test on held-out records: FP alone with learned
     theta_hat vs FP at theta_0.
- Goal achieved when: the [GYOROK] Table 1 signature is reproduced in
  direction (control degrades the standalone baseline, some beta in the sweep
  restores it to <= theta_0-baseline NRMS while augmented NRMS stays within
  10% of the control's augmented NRMS). If no beta achieves both, the result
  is a finding, not a failure of the plan: escalate to Remark 2 (per-epoch
  recompute) per Sect. 5.3.

### Step 11: Closure
- Decision entries in decisions.md (method choice, hook design, beta
  selection rule), run-table outcomes filled in, this document's status
  updated, and the Y-binned-Pi comparison experiment scoped as the follow-up
  thesis experiment.
- Goal achieved when: a reader can reproduce the approval chain from this
  document alone.
