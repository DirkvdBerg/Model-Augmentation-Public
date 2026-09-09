# STATE OF THE DERIVATION

## READ FIRST

This file is long and will be retrieved in fragments. These five hold regardless
of which fragment you receive.

1. The true parameter value is NOT well defined under a nonparametric
   discrepancy. Never give an unconditional true-parameter-recovery proof. A
   conditional result under stated assumptions is the useful form.
2. Trajectory-level orthogonal projection for a STATIC learned term is prior art
   (Kon 2023), as is sensitivity-orthogonal discrepancy in calibration (Plumlee).
   The candidate contribution is the case where the learned term carries its own
   internal state, and that boundary is not yet established.
3. The VALUE of the augmentation's contribution and its TANGENT (the directions
   training can move in) are different objects. Do not infer one from the other
   without a theorem.
4. Synthetic testbed results are evidence about the synthetic system only.
   Truth-dependent diagnostics are validation tools, never deployable ingredients.
5. An evidence label is not evidence. SYMBOLICALLY VERIFIED means a computer
   algebra system checked a given identity; it supplies no assumptions,
   quantifiers, or argument, and is not a theorem.

Nothing has been implemented in the real training pipeline, and no run exists
with joint estimation and the proposed penalty enabled. Gantry efficacy is
entirely untested.

---

Updated 2026-09-07. This is the current working record, not an authority over
mathematics, primary sources, or the user's current instructions. Flag conflicts
explicitly. Experimental numbers below are preserved from the previous record;
this editorial revision did not rerun experiments or reverify source PDFs.

**Evidence categories.**
- THEOREM: a precise statement with assumptions and a complete argument.
- SYMBOLICALLY VERIFIED: an identity or finite example checked computationally;
  this supports, but does not replace, a general proof.
- MEASURED: a numerical observation with the stated experimental scope.
- TESTBED: evidence about a specified synthetic system, not gantry efficacy.
- CITED: verified in a primary source with a precise supporting location.
- OPEN: a target, conjecture, unresolved assumption, or missing validation.
- RETRACTED: a previous claim withdrawn or narrowed.

Truth-dependent diagnostics are validation tools, not deployable ingredients.
Labels alone are not evidence; retain proof, source, and experiment references
when available. Missing artifact identifiers must be supplied before thesis use.

---

## 0. Run and artifact references

Supplied 2026-09-07 to close the "artifact identifiers needed" flag. The
external session cannot access this repository or these files; they are given so
results can be traced by whoever can.

```
Gantry measurements   checkpoint SSE_Interconnect_Composed_p2LPDA_best.pth,
                      run 81262, best validation 5.774e-06, nx_ann = 8,
                      orth = OFF, joint = False, encoder linear_map,
                      ann_route_ix = 0..13, dataset
                      augmentation_ma50_b140-230_a6_z03 at 4 kHz.
                      Script leakage_measure.py, 2026-09-04.
Symbolic latent path  affine_leakage.py (SymPy 1.14.0) and affine_leakage.m
                      (MATLAB R2025a), 2026-09-04. Both use null-space
                      parameterisation, NOT solve(): solve() returned the
                      trivial branch on a homogeneous system and the two
                      engines disagreed until this was changed.
Rank / identifiability nullspace_check.py, claimA_symbolic.py and .m, 2026-09-03.
Discrepancy alignment recovery_floor.py, 2026-09-03. Truth-requiring.
Bias relation         bias_relation.py, 2026-09-03.
Synthetic testbed     system.py, geometry.py, train_compare.py, 2026-09-06/07.
```

**The deployed penalty, as actually implemented.** The basis is built from the
per-sample one-step state Jacobian `Phi = d x_next / d theta` in NORMALISED
STATE coordinates, stacked over roughly 6,700 decimated samples with an extended
regressor column, economy SVD, relative rank truncation at 1e-12. The penalty is
`beta ||Q^T f_ANN(x, xa = 0, u)||^2` on the routed physical rows, with `Q` frozen
at the anchor `theta_bar`. It is therefore a state-space, one-step, value
penalty evaluated off the realised latent trajectory. Connecting the proposed
output-space windowed penalty to this implementation is outstanding work.

---

## 1. Objective and candidate contribution

**Target, not yet established.** Develop a trajectory-level regularizer that
reduces parameter-aligned contributions from a dynamic augmentation while
retaining useful predictive correction. Establish separately what this implies
for local physical-parameter distinguishability, regularized estimation, and a
defined calibration target. Neither improvement over baseline-alone parameter
error nor predictive improvement is presently guaranteed. OPEN

The deployed adaptation of a Györök-style pointwise penalty evaluates the
augmentation on the slice xa = 0. It therefore misses an explicit affine
latent-to-physical map K. This is a limitation of applying that construction to
our dynamic augmentation, not a claim that Györök's original static model
contained this unpenalized latent path.

Learned internal-state feedback is the candidate contribution boundary.
Trajectory-level orthogonality and sensitivity-orthogonal discrepancy are
already prior-art topics; novelty of the precise extension remains to be
positioned. The learned states need not have a unique physical interpretation.

Keep distinct: contribution orthogonality, tangent separation, local parameter
identifiability, selection by a regularized objective, calibration-target
recovery, true-parameter recovery, and prediction quality.

---

## 2. Mathematical setup and established restricted results

### 2.1 Geometry to be used

Stack the full predictor outputs over the samples and windows used by the loss:
    yhat = p(kappa, eta, xi),
    delta yhat = S delta kappa + R delta eta + E delta xi + remainder.

Kappa denotes independent physical coordinates where justified; eta denotes
learned-model parameters; xi denotes actual shared encoder parameters. Specify
which physical and latent initial states the encoder initializes. Raw-parameter
rank deficiency must not be treated as identifiable information.

For a positive-definite loss weight W, choose L with L^T L = W. Define
    A = L S, B = L R, D = L E, M_D = I - D D†,
    Sbar = M_D A, Rbar = M_D B.
Use orthonormal bases of the numerically supported column spaces, with the rank
tolerance reported. Equivalently, treat [D B] as joint nuisance directions.

For the stated linearization, no nonzero physical change can be cancelled by
nuisance changes exactly when
    rank([D B A]) - rank([D B]) = dim(kappa).
Equivalently, A^T (I - N N†) A is positive definite, N = [D B].
Reason: v^T A^T (I - N N†) A v = ||(I - N N†) A v||², which vanishes exactly
when A v belongs to range(N). THEOREM, finite-dimensional linear algebra.

After encoder profiling, this requires full physical rank and trivial
intersection of range(Sbar) and range(Rbar). Orthogonality is sufficient but
stronger than necessary. This is a first-order distinguishability result, not
a global uniqueness or true-parameter recovery theorem.

All recurrence, initial-condition, observation, and state-dependent scheduling
derivatives must be included. Specify reference parameters, trajectories,
horizons, and whether inputs are exogenous or controller-generated.

### 2.2 Affine latent-path counterexample

- In the affine construction, the explicit map K from latent state to physical
  correction is multiplied by zero on xa = 0. The pointwise penalty therefore
  has zero derivative with respect to K. This says nothing about gradients
  from the prediction loss or other penalties. The constraint was imposed as
  an identity over the whole (x,u) slice, stronger than a finite point set.
  SYMBOLICALLY VERIFIED in SymPy and MATLAB, as recorded previously.

- In the constant-matrix affine recurrence, the latent-path contribution has
  kernel
    Lambda_T(i) = sum_{j=i+1}^{T-1} A^(T-1-j) K G^(j-1-i),
  giving sum_i Lambda_T(i) g_i for the specified writes and zero initial
  latent contribution. Nonzero latent initialization adds its own term.
  The parameter-aligned part requires applying the specified sensitivity
  projection to the contribution. Matrix powers are not an exact formula for
  a general nonlinear or time-varying rollout.
  SYMBOLICALLY VERIFIED against the affine rollout, as recorded previously.

- The K = 0 control still has nonzero projected cumulative direct contribution.
  Thus propagation alone can break the proposed transfer from one-step
  orthogonality to rollout orthogonality. The additional K route is the
  candidate extension; the propagation-only issue is not claimed as novel.
  SYMBOLICALLY VERIFIED for the constructed example.

An affine counterexample also disproves a universal guarantee for a richer
model class containing it; it does not prove every richer network behaves so.

### 2.3 Conditional baseline-estimate invariance

Consider min_{k,d} ||b - A k - d||² with fixed full-column-rank A, unrestricted k,
and A^T d = 0 for every admissible d. Assume a minimizer exists and admissibility
of d does not depend on k. Then every joint minimizer has
    k_hat = (A^T A)^(-1) A^T b,
the baseline-alone least-squares estimate. Proof: the normal equation in k is
A^T(A k + d - b) = 0, and A^T d = 0 removes the correction term.

This also describes a consistently weighted, encoder-profiled affine problem
when its assumptions hold. Equality of estimates implies equal error relative
to any fixed target in this restricted problem. THEOREM with proof above.

The expression (Phi^T Phi)^(-1) Phi^T Delta describes displacement due to a
discrepancy only under the corresponding affine reference model and rank
assumptions. It is not an unconditional formula for the recurrent estimator.
Extension to a moving basis, a soft penalty, parameter-dependent correction,
nonlinear encoder, or nonlinear joint training is OPEN.

---

## 3. Gantry checkpoint observations

Trained checkpoint, nx_ann = 8, orth = OFF, joint = False; 48 windows,
open-loop rollout from data-derived x0 with xa = 0. The previous record's phrase
"stable at 25 steps" was loose wording and is replaced by the figures. The
measurement was repeated at two horizons:

```
window length T      25        100
latent-path share    0.826     0.905
projected overlap    0.677     0.745
total overlap        0.738     0.728
```

The quantities are horizon-dependent, not identical, and any citation should
state which horizon it uses.

- Reported latent-path contribution norm is approximately 90% of the total
  cumulative augmentation contribution norm. This is a norm ratio, not an
  additive energy share; correlated paths may cancel. MEASURED
- Reported projection overlap is 0.745 against a randomisation null of 0.193,
  approximately 3.9x and above its 99th percentile. The isotropic RMS overlap
  benchmark sqrt(10/288) is approximately 0.186; it is not generally the exact
  mean overlap. Preserve the original projection, weighting, and null definition
  when reproducing these numbers. MEASURED
- The stacked sensitivity matrix is 288 x 14 with numerical rank 10. Individual
  endpoint state blocks have rank 6 of 6. Projection onto their full R^6 space
  is the identity, so exact orthogonality would force a zero correction.
  This makes that formulation overly restrictive, not vacuous. It does not
  rule out a stacked trajectory formulation. MEASURED
- Reported true-discrepancy overlap is 0.826. This is a truth-dependent geometric
  diagnostic, not by itself a parameter-recovery floor. In an applicable affine
  problem, displacement depends on A† Delta, including singular values and
  parameter coordinates, not only a normalized overlap. MEASURED,
  truth-requiring; metric definition and artifact needed before thesis use.

These measurements show realized contribution alignment in the recorded
state-space diagnostic. They do not establish tangent overlap in the actual
output loss, parameter drift (joint estimation was disabled), or suppression
by the proposed regularizer.

---

## 4. Synthetic testbed observations

Reported setup: 1-DOF baseline plus hidden mass-spring-damper absorber, 0.5% mass
ratio, discrepancy approximately 10% of output standard deviation; raw
rank deficiency k = ka + kb; partial observation and a trained encoder.
Reported sweep duration is about three minutes.

The following numbers are preserved from the latest supplied record. Associate
them with the encoder/scaling fixes, configuration, seed, and saved run before
using them as reproducible results. TESTBED throughout.

- Free augmentation: prediction RMSE 1.09e-3 versus baseline-alone 1.02e-3,
  with parameter error 1.8x worse. This is consistent with substitution, but
  does not alone isolate its mechanism or show the latent route caused it.
- Penalizing only the latent path reduced its overlap while total overlap
  remained approximately unchanged. This motivates measuring both paths and
  testing a total-contribution penalty; it is not a universal impossibility
  theorem for path-specific penalties.
- Total-contribution penalty: overlap ell fell from 0.832 to 0.250; parallel
  amplitude decreased about 25x and perpendicular amplitude about 4x.
  This supports preferential suppression of aligned contribution alongside
  overall shrinking, not rotation with no shrinking.
- A plain contribution-size penalty gave comparable parameter error:
  0.0932 versus 0.0950. Reported perpendicular amplitudes were 1.89e-4 for
  size penalization and 3.51e-3 for projection, a ratio of about 18.6.
  Retaining more perpendicular contribution does not itself show that the
  retained contribution improves prediction.
- LATENT-PATH SHARE AFTER THE SCALING FIX, which decides whether this testbed
  exercises the mechanism at all. In-span amplitudes for the free augmentation:
  a_parallel(latent path) = 3.47e-2 against a_parallel(total) = 1.91e-2, so the
  latent route is now a leading contributor. Before the fix the ratio
  ||d_latent|| / ||d|| was about 2.5e-4, against roughly 0.90 on the gantry, so
  the pre-fix testbed did not exercise the mechanism and its results are void.
  Amplitudes exceeding the total is possible because the two routes are
  correlated and partially cancel; this is a norm ratio, not an energy share.
- Prediction RMSE: baseline-alone 1.02e-3, free augmentation 1.09e-3,
  constrained augmentation 1.03e-3. These runs do not demonstrate predictive
  benefit from augmentation. Memory must be shown useful before claiming
  useful memory was preserved.

Diagnostic definitions for each total or latent-path contribution d:
    dbar = M_D L d,
    ell = ||Q_S^T dbar|| / ||dbar||,
    a_parallel = ||Q_S^T dbar||,
    a_perp = ||(I - Q_S Q_S^T) dbar||,
where Q_S spans Sbar. Report ||dbar|| and ||L d|| too. Ell is undefined or
unreliable for negligible dbar; use a declared threshold. A numerator using
encoder projection and a denominator without it conflates overlap with removal
of encoder directions.

Record whether d is a nonlinear ablation difference or a first-order propagated
contribution. Compare penalties over prediction/retained-correction tradeoffs;
identical beta values need not represent comparable strengths. Distinguish
truth-based oracle hyperparameter selection from a deployable selection rule.

---

## 5. Open questions and candidate claims

**Q1. Value versus tangent.** What does small ||Q_S^T M_D L d|| imply about
attainable predictor variations? A value constraint at one point is insufficient
in general. Determine useful additional hypotheses or construct counterexamples.
If a fixed projection identity holds throughout a parameter neighborhood,
differentiate that identity; if the basis or encoder projector moves, include
its derivatives. Establish the relationship between derivatives of d and R.
OPEN

**Q2. Separation versus regularized selection.** Historical geometry reported
three principal cosines printed as 1.0000, null 0.332, rank 151, ambient 1600.
These numbers are unvalidated: they predate the encoder fix, describe untrained
initialization, and lack a checked rank tolerance. Recompute at trained
checkpoints with singular spectra and tolerance sensitivity.

Confirmed overlap would establish failure of the local separation condition at
those points. It would not prove separation impossible for every parameter
setting or architecture. A value penalty may select a decomposition despite a
confounded prediction map. Analyze the full regularized objective, including
penalty derivatives with respect to physical, network, and encoder parameters.
Candidate empirical target: reduced aligned contribution at matched retained
correction and acceptable prediction. OPEN

**Q3. Prior-art boundary.** Sensitivity-orthogonal discrepancy already appears
in the calibration literature. Earlier project records logged Plumlee 2017 and
Plumlee and Joseph 2018 and assessed related work under P7. Do not present that
principle as newly discovered or erase the existing audit. Consult the reference
record and source audit first. If theorem-level details remain unresolved,
verify the exact subspace, inner product, assumptions, and estimand in the PDFs.
Do not transfer a statement about point estimation or uncertainty from one paper
to another without checking. Exact positioning of our dynamic extension is OPEN.

**Q4. Enforcement.** Compare hard constraints, by-construction projection,
quadratic penalties, and augmented-Lagrangian methods. A coefficient beta in
beta ||c||² is a penalty weight, not generally a Lagrange multiplier. Finite
quadratic penalties need not satisfy c = 0 exactly. Feasibility, deployment,
moving projections, and optimization behavior remain to be established.
Claims about a paper's motivation need a precise source location. OPEN

**Q5. Nonlinear connection.** Bound or otherwise clearly delimit the difference
between exact nonlinear rollout ablations and first-order propagated effects.
State-dependent scheduling already contributes through the first-order chain
rule; variation of the linearization introduces further terms. No applicable
remainder bound is currently established for the deployed model. OPEN

---

## 6. Retractions and corrections

Retain these to prevent recurrence of the same overclaims.

- General "orthogonality gives no worse physical parameters than baseline-alone
  while improving prediction": RETRACTED. Only the restricted affine
  invariance theorem in Section 2.3 is established here.
- "0.826 is a parameter-recovery floor": RETRACTED. The number is preserved as
  a reported discrepancy-alignment diagnostic. Neither general harm nor general
  harmlessness of the constraint follows from it.
- "Identity projection is vacuous": CORRECTED. Orthogonality to the full space
  forces zero contribution. The previous per-endpoint condition
  Psi_T^T A^a K = 0 does not rule out stacked or by-construction alternatives.
- "The encoder is required and cannot absorb parameter error": RETRACTED.
  Least-squares initialization used zero targets, so the frozen-encoder
  comparison did not test the intended initialization.
- Historical geometry values, including rank(E) = 26 and encoder principal
  cosines 0.87, 0.75, 0.61: WITHDRAWN pending rerun after the encoder fix.
- The single-scale architecture suppressed latent feedback near initialization
  to O(eps²) versus O(eps) direct correction, under bounded network derivatives.
  The prior record reports separate scales now implemented. Earlier results
  describe the old architecture and must not be pooled with corrected runs
  or used to establish useful latent-path preservation.
- Saturated overlap at selected checkpoints proves universal unreachability:
  RETRACTED. Scope is local and numerical unless a general argument is supplied.

---

## 7. Outstanding work

- No extension is reported implemented in the gantry training pipeline.
  The deployed penalty still uses stacked one-step state sensitivities and
  network values on xa = 0.
- No gantry run with joint estimation and the proposed penalty enabled is
  reported. Gantry efficacy is therefore untested, despite existing structural
  counterexamples and synthetic observations.
- Verify the corrected testbed can benefit from memory on held-out predictions.
- Rerun geometry after the encoder fix and at trained checkpoints; document
  ranks, tolerances, whitening, and whether the basis is fixed or recomputed.
- Supply reproducible run and source references for the preserved observations.
- Connect the implemented nonlinear contribution penalty to the mathematical
  claim, including nuisance channels, moving bases, and approximation error.
