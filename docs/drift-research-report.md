# Free-run drift on a marginal mode: extended literature report and method candidates

**Date**: 2026-07-24. **Phase 2 of analysis session #1.**
**Grounded in**: `docs/drift-critical-analysis.md` (Phase 1, this session). The questions below are the
REFINED ones from that analysis, not the original brief's, because Phase 1 refuted two of the brief's
premises (encoder-init as the DC cause, and "the drift lives in a loss-flat direction").
**Deliverable**: cited method candidates, each marked against the project's five hard constraints, with
trade-offs and build cost.

---

## 0. How to read this report (verification levels and honesty guard)

Every claim carries a level. Do not cite anything below PRIMARY-READ in the thesis without re-reading the
source.

| Level | Meaning |
|---|---|
| **PRIMARY-READ (local)** | the PDF is on disk in `literature/` and the project has read it |
| **ABSTRACT-VERIFIED** | I fetched the arXiv abstract/landing page this session and the statement is in it |
| **SEARCH-LEVEL** | the statement comes from search-result snippets of the paper's own text; plausible but not read at source |
| **INFERRED** | my reasoning connecting a source to our setting; not the source's claim |

### Adversarial checks that actually changed something this session
1. **A PDF fetch produced a fabricated-sounding claim, and I discarded it.** Fetching `arXiv:1705.08209`
   (ARTBP) as a PDF returned "the divergence of truncated BPTT increases with larger learning rates", which
   CONTRADICTS the project's own primary read (`baseline-null/diagnostics-literature.md`: the divergence is
   learning-rate INDEPENDENT). A separate search returned the paper's own wording: the influence-balancing
   experiment uses a DECREASING schedule `eta_t = eta_0 / sqrt(1+t)` and "reducing the learning rate will not
   prevent divergence". **The project's read is correct; the PDF fetch was unreliable.** Consequence for us
   below (Q4): ARTBP's own convergence demonstration is at a DECREASING step size, which is exactly the open
   question the project flagged and has never resolved at constant lr.
2. **A second PDF fetch failed outright** (`arXiv:2201.00144`, binary not decoded), so the nonlinear-NI
   free-body claims below are SEARCH-LEVEL only, not primary-read. Flagged in Q5.
3. **The project's "orthogonal-by-construction is IO-only" claim is CONFIRMED** at the abstract:
   `arXiv:2511.01321` says "physics-based **input-output** models"; state-space is not mentioned. The
   state-space extension really is open. (ABSTRACT-VERIFIED.)

---

## 1. The refined questions

- **Q1** Why does a curvature-blind adaptive optimizer park a STIFF direction away from its minimizer, and
  what is known about this on near-unit-root modes? (Phase 1 refuted the "flat direction" framing.)
- **Q2** *(the sharpest)* How can a subspace-selective regularizer BITE under a scale-invariant optimizer,
  given that `beta` saturates from 1e3 to 1e12 under Adam?
- **Q3** What separates a spurious persistent force from a genuine dissipative one on a marginal mode, when
  direction fails (rank-1 pins are dodged) and information fails (d16)?
- **Q4** Unbiased truncated-BPTT corrections on a mode with NO geometric memory decay: does unbiasedness help
  at CONSTANT step size, and how is variance controlled at rho = 1?
- **Q5** Grey-box learning that PRESERVES pole 1 while keeping full expressivity and admitting a DC-carrying
  dissipative residual.
- **Q6** Integrator factoring (Tustin-Net family): can it be modified to admit a net impulse?
- **Q7** Exponential-versus-marginal discrimination, and stability constraints with an exact marginal carve-out.
- **Q8** Encoder/state-estimator bias on integrator modes, scoped as a FLOOR problem.

### The five hard constraints used to mark every candidate
- **C1** preserves `|lambda| = 1` on X/Y (no artificial damping)
- **C2** full expressivity (no for-all-weights class restriction as the DELIVERABLE)
- **C3** does not forbid a DC-carrying / net-impulse friction
- **C4** acts on the estimator/training (keeps X/Y routing, full `[x,u]` input, position-domain loss)
- **C5** knowledge-free target (data-derived or known-FP-subspace, never an assumption on the unknown residual)

---

## 2. Q1: optimizer geometry on a stiff, badly-approached direction

**Established, and it matches our measurement.** Adam is normalized steepest descent with respect to the
`l_infinity` norm and behaves like sign descent; its implicit bias is `l_infinity`-margin maximization where
(S)GD's is `l_2` (SEARCH-LEVEL: arXiv:2406.10650; arXiv:2505.24022; arXiv:2602.16340). A sign-like step has
magnitude ~lr regardless of gradient magnitude or local curvature, so on a direction with large curvature and
a minimizer at `b* = -g/H` of order 1e-11, the iterate cannot settle into the well and parks at ~lr. This is
exactly the `curvature_sensitivity` + `gain_vs_dc` picture, and it explains the 2000x Adam-versus-SGD DC gap
at matched lr without invoking flatness at all.

Two further supports the project already holds: constant-step-size SGD/Adam do not converge to a minimizer
but fluctuate in a stationary law whose scale vanishes only as the step size does (SEARCH-LEVEL, via the
project's own agent report: arXiv:2607.16384; Barakat and Bianchi arXiv:1810.02263); and finite-step GD
carries an implicit step-size-scaled bias (Barrett and Dherin arXiv:2009.11162).

**Consequence for method selection (INFERRED but tight).** Since the operating point is set by STEP SIZE and
not by curvature, any cure that works by ADDING curvature (a quadratic penalty) has bounded effect under
Adam. That is precisely what `step4_orth_projection_null.py` measured. Cures must instead change the STEP
(preconditioning, decoupled application, projection of the update) or change the GRADIENT (debias it).

**Gap**: no source reports this specifically for a marginal/integrator mode inside a simulation-error loss.
Our combination remains unreported, which is a thesis-positive negative result and is consistent with the
project's earlier D-108 conclusion.

---

## 3. Q2 (the decisive question): making a subspace penalty bite under a sign-like optimizer

This is where the literature is richest and where the project has tried the least. Four families, in
increasing strength.

### 3.1 Decoupled (proximal) application of the penalty
The canonical precedent is decoupled weight decay: with Adam, an `l_2` penalty inside the loss has its
gradient divided by the same `sqrt(v_hat)` as the data gradient, so the EFFECTIVE regularization strength
becomes gradient-history-dependent and, in the limit, scale-free. AdamW fixes it by applying the decay
directly to the parameters, OUTSIDE the adaptive preconditioner:
`theta <- (1 - eta*lambda) theta - eta * m_hat / (sqrt(v_hat) + eps)`
(PRIMARY-ADJACENT: Loshchilov and Hutter, arXiv:1711.05101 / ICLR 2019; the mechanism is stated verbatim in
the paper and reproduced in multiple secondary sources fetched this session).
**Direct transfer to us**: apply the orthogonal-projection penalty PROXIMALLY, i.e. after the Adam step,
shrink the component of the ANN's routed output (or of the responsible parameters) along the pinned subspace
by a fixed factor. This restores a monotone dependence on `beta` and removes the saturation ceiling.
**This is the single cheapest untried fix for the step-4 failure, and it is a 3-line change.**
Marks: C1 yes, C2 yes (soft), C3 yes (only the pinned subspace is shrunk), C4 yes, C5 yes.
Risk: proximal shrinkage of a state-dependent output is only well defined if you shrink PARAMETERS whose
output lies in the subspace; for a nonlinear ANN this needs the same weight-space back-mapping the limits doc
flags as C3 (`data-silent-regularization-limits.md`).

### 3.2 Projection of the UPDATE instead of penalization of the loss
Continual learning has an entire mature toolset for "do not move in this subspace": Orthogonal Gradient
Descent projects each update onto the orthogonal complement of a protected subspace (Farajtabar et al.,
arXiv:1910.07104); Gradient Projection in Common Null Space and GNSP project into the null space of prior
tasks (arXiv:2507.19839; GPCNS, ACM MM 2024); Restricted Orthogonal Gradient Projection relaxes the hard
projection to keep learning capacity (arXiv:2301.12131). Orthogonal Natural Gradient combines projection with
a natural-gradient preconditioner (arXiv:2508.17169). All SEARCH-LEVEL / ABSTRACT-VERIFIED.
**Transfer**: replace `V = beta ||Q^T f||^2` with "take the Adam step, then remove its component along the
protected direction(s)". This cannot saturate, because it is not a penalty at all; and it composes with Adam
rather than fighting it.
Marks: C1 yes, C2 yes if the protected subspace is genuinely the non-identifiable one, C3 conditional (see
Q3: the protected subspace must not be all velocity-row DC), C4 yes, C5 yes.
Risk: exactly the step-4 dodge. Projection removes motion along `Q` but the optimizer will use the orthogonal
complement, which for a rank-1 `Q` still drifts. **Projection fixes the SATURATION problem, not the SUBSPACE
problem.** Both must be solved.

### 3.3 Hard output-space projection layers
If a subspace must be exactly annihilated, closed-form differentiable projection layers exist with universal
approximation preserved under the constraint: HardNet (parallel projection layer, closed-form forward pass,
input-dependent affine equality and inequality constraints, universal approximation under the constrained
architecture), KKT-hPINN (closed-form projection for linear equality constraints), ENFORCE
(arXiv:2502.06774), LMI-Net (differentiable projection onto LMI constraints, arXiv:2604.05374). All
SEARCH-LEVEL.
**Transfer**: a projection layer on the ANN's routed output that annihilates the protected direction exactly.
Marks: C1 yes, C4 yes, C5 yes, **C2/C3 depend entirely on what you annihilate**. Annihilating "all velocity
DC" is the mean penalty, i.e. C3 fail. Annihilating a measured, data-derived low-rank direction keeps C3 but
is dodgeable (step-4 measured the dodge on the hard projection too).

### 3.4 Per-parameter-group optimizers and curvature-aware steps on the protected direction only
Since the pathology is that Adam's step is curvature-blind while SGD's is curvature-aware, the natural hybrid
is to run a NON-adaptive (or second-order-preconditioned) step on the parameters carrying the protected
direction and Adam elsewhere. Precedents: K-FAC and its per-layer/mini-block variants give tractable
curvature preconditioning per block (SEARCH-LEVEL: K-FAC literature, two-level and mini-block Fisher
variants), and PyTorch natively supports per-parameter-group optimizers.
**Transfer**: since our measured Hessian in the 6-D output-DC subspace is available exactly (autograd
double-backward, positive definite, eigenvalues 1.95e-2 to 7.43e4, `curvature_sensitivity.py`), a Newton step
in THAT tiny subspace is essentially free and lands at `b* ~ 1e-11` rather than ~lr.
Marks: C1 yes, C2 yes, C3 yes, C4 yes, C5 yes.
**This is the most constraint-clean candidate in the whole report and it has never been tried.** Caveat: it
addresses the DC displacement component (about 80% in the null, X-dominant on real data), not the Y
anti-damping feedback.

---

## 4. Q3: separating a spurious persistent force from a genuine friction impulse

Phase 1 established that DIRECTION fails (rank-1 pins are dodged) and INFORMATION fails (d16: on a K=0 axis a
DC is the most loss-informed direction per unit amplitude). Four candidate separators remain, ranked by how
well they survive the constraints.

### 4.1 Power sign (`F . v`) as a SOFT steering term. Best separator, and it must NOT be hard.
This is the only criterion that provably distinguishes the two: a sustained force along motion injects energy
(`F . v > 0`), friction opposes motion (`F . v < 0`). Imposed HARD it is a class restriction (C2/C3 fail as a
deliverable, and the project already proved passivity bounds velocity, not position: Cauchy-Schwarz gives
`O(sqrt(T))` position growth, `p1_drift_probe`). Imposed SOFT it is a steering term, exactly the shape the
project's own `steering-not-blunt-bound` rule asks for.
Literature support for the soft version is explicit: "system models satisfying desired properties like
stability or passivity can be obtained by imposing SOFT constraints in the form of regularization terms ...
such approaches may not possess theoretical guarantees ... but nonetheless often yield property-preserving
models in practice" (SEARCH-LEVEL: Safe Physics-informed Machine Learning for Dynamics and Control,
arXiv:2504.12952, ACC 2025 tutorial; and the MERL physics-informed ML review TR2023-052).
Marks: C1 yes (no restoring term added), C2 yes (soft), **C3 yes (this is the point: friction is REWARDED,
not penalized)**, C4 yes, C5 yes (power sign needs no knowledge of the residual).
**Assessment: this is the best-founded untried separator in the project.** It is not in the current plan at
all, and it is the natural soft counterpart of the dissipativity route the project rejected in hard form.

### 4.2 Free-run CONSEQUENCE penalties (price the accumulation, not the force)
Rather than penalizing a force component, penalize what it does over a horizon. The 2026 rollout literature
has converged on exactly this shape: Time Increment Consistency, "a finite-lag regularizer that directly
constrains temporal covariance and mixing structure of the rollout, supplying long-horizon constraints that
pointwise state-reconstruction losses provably cannot" (SEARCH-LEVEL, arXiv:2605.05540); and commutativity /
non-normality regularization, which identifies "transient amplification of perturbations along rollout
trajectories, driven by non-normal and non-commuting latent Jacobians" as the structural mechanism of
long-horizon error growth and adds JVP-based penalties on per-step Jacobian normality and across-step
commutativity (SEARCH-LEVEL, arXiv:2605.08856; already flagged in the project's own sweep Direction 6).
Marks: C1 yes, C2 yes, C3 **yes, and this is its key advantage**: a friction impulse and a spurious DC differ
in what they do to the free-run only through the sign of their work, so a consequence penalty that is
referenced to the TRUE trajectory (available in sim, and in the windowed loss on real data) prices the error,
not the mechanism. C4 yes, C5 yes.
Cost: these are gradient-through-rollout penalties and inherit the same ill-conditioning as long-horizon BPTT
on a marginal mode. Pair with Q4.

### 4.3 Frequency-selective priors on the residual
The project's current Layer-2 target is "near-DC frequency selectivity" (d16). The principled home for this
is kernel-based regularized identification with FREQUENCY-DOMAIN SIDE INFORMATION, where a prior is shaped in
frequency without restricting the model class (SEARCH-LEVEL: arXiv:2111.00410; the broader stable-spline /
RKHS framework, Pillonetto and De Nicolao 2010, Pillonetto et al. Automatica 50(3), and arXiv:1511.01543).
Marks: C1 yes, C2 soft yes, **C3 NO in the naive form**: a Coulomb friction impulse IS near-DC, so a pure
near-DC penalty suppresses it. This is the open tension the project's own step-2 flagged in code and did not
solve. It becomes viable only combined with 4.1 (penalize near-DC content that also has `F . v > 0`).

### 4.4 Adaptive-control robustification (the classical home of this exact failure)
"Parameter drift along an unexcited direction" is the founding problem of robust adaptive control, with a
standard toolkit: sigma-modification (leakage toward a prior; "stops parameter drift in the presence of
persistent disturbance but induces an asymptotic error"), e-modification, the projection operator (keeps
parameters in a convex set, requires a known bound), and dead-zone (stop adapting below a threshold; "only
local robustness") (SEARCH-LEVEL, consistent across several sources this session).
Marks: sigma-mod and dead-zone are C1/C4/C5-clean and C2-soft, but **C3 is the problem again**: leakage
toward zero on a velocity-row DC is the mean penalty under another name. The projection operator is the most
interesting for us because it bounds rather than shrinks, which admits a nonzero friction DC while forbidding
runaway, at the price of needing a bound (C5 weakens: where does the bound come from?).

---

## 5. Q4: unbiased truncated BPTT at rho = 1

**What the source says (SEARCH-LEVEL, cross-checked against the project's PRIMARY-READ).** Truncated BPTT
"favors short-term dependencies", is biased, and "does not benefit from the convergence guarantees from
stochastic gradient theory". ARTBP restores unbiasedness by random truncation with compensation factors. In
the influence-balancing experiment truncated BPTT "diverges even for truncation ranges largely above the
intrinsic temporal scale", "estimates the overall gradient with a wrong sign", and "reducing the learning rate
will not prevent divergence"; ARTBP converges reliably. Critically, that experiment runs a DECREASING
schedule `eta_t = eta_0/sqrt(1+t)`, `eta_0 = 3e-4`. Variance is controlled by making the probability of a
length-L subsequence decay like `L^-alpha` (the poly-tail, Eq. 14), which turns exponentially growing
compensation factors into polynomially growing ones (finite variance for `alpha > 3`).

**Two consequences specific to us, both of which the project has half-recorded and neither of which is
resolved.**
1. **The variance argument presumes decaying memory that a pole-1 mode does not have.** Our Phase-D grid
   measured the poly-tail buying only 2 to 5x over geometric (after the 1-seed 24 to 47x was corrected), and
   the baseline-null run shows ARTBP making a low-signal case WORSE. So ARTBP on `z = 1` is unbiased but
   heavy-tailed, and that is a measured property of OUR system, not a literature deficiency.
2. **Unbiasedness buys convergence only with a vanishing step size.** The paper's own demonstration uses a
   decreasing schedule; the constant-step-size literature (Q1) says a constant step never converges to a
   point. **Nobody has tested ARTBP with a decaying lr schedule on this problem.** Given that our drift is
   proportional to lr, a decreasing schedule is the theoretically indicated companion to ARTBP, and it is
   free to try.
Marks for ARTBP + poly-tail + decaying lr: C1 yes, C2 yes, C3 yes, C4 yes, C5 yes. **All five clean.**
Cost: measured, already built (`train_artbp.py`), gate-2 script ready.

**Multiple shooting** (Turan and Jaschke arXiv:2109.06786; Forgione and Piga truncated simulation error;
Ribeiro et al. Automatica 121:109158) partitions the horizon and imposes continuity through a penalty term,
smoothing the cost and reducing local minima (SEARCH-LEVEL). Phase 1 established our "multiple shooting
failed" evidence is confounded by the pre-D-101 lr bug, so this is **untested, not refuted**, and it marks
C1-C5 clean. It is the supervisor's own named position-based alternative (priority list item 8).

---

## 6. Q5: pole-1-preserving grey-box families

### 6.1 What definitively does NOT preserve pole 1 (do not revisit)
Contraction-based parametrizations (RENs arXiv:2104.05942, ci-RNN, LBEN, Lipschitz-bounded networks) enforce
a contraction rate strictly below 1, which excludes spectral-radius-1 modes by construction; stable
port-Hamiltonian networks with `R > 0` give global asymptotic stability, hence poles strictly inside
(arXiv:2502.02480, "ensures global asymptotic stability of the identified dynamics ... by constraining the
Hamiltonian to be a convex, positive definite Lyapunov function", SEARCH-LEVEL). The supervisors' own
constraint-free contracting-LFR augmentation (arXiv:2604.11421) fails us for the same reason. Our v6b
measurement independently killed the magnitude-cap route empirically.

### 6.2 Conservative port-Hamiltonian, `R = 0`
Setting `R = 0` removes dissipation, giving Lyapunov-stable-but-not-asymptotic behaviour with bounded
trajectories and no attractive equilibrium. Marks C1 yes, but the theorem assumes a strict positive-definite
Hessian / unique energy minimum, so applying it to an exact `K = 0` continuum is an extrapolation beyond the
stated conditions (this caveat is the project's own, from `diagnostics-literature.md`, and I did not find a
source that removes it). C2/C3 fail as a deliverable (class restriction). Reference arm at best.

### 6.3 Negative Imaginary with free body motion: the theory EXISTS, the learned realization does not
This is the most important correction the search offers to the project's older framing. Nonlinear NI theory
has been extended to systems with FREE BODY MOTION, i.e. a pure integrator / pole at the origin, using a
positive SEMIdefinite storage function (SEARCH-LEVEL: "extending the definition of nonlinear NI systems to
allow for systems with free body motion", arXiv:2011.14610; nonlinear NI via dissipativity,
arXiv:2201.00144; and on the linear side a generalized NI lemma valid with poles on the imaginary axis
including the origin, arXiv:1107.4255). Conditions are established "under which a nonlinear system in a
cascade connection with an integrator would be nonlinear negative imaginary".
**Caveat (honesty guard)**: my attempt to primary-read arXiv:2201.00144 FAILED (PDF not decoded). These are
search-level statements. The project's D-104/D-106 position (the NI theory exists; the gap is the LEARNED,
FORWARD, LPV realization) is CONFIRMED at search level and should not be re-litigated, but the exact
definitions must be primary-read before any thesis claim.
Marks: C1 yes (this family is the only structural one that natively keeps the marginal pole), C3 yes in
principle (NI permits dissipative forces), **C2 no as a deliverable** (still a class restriction).
Build cost: months (the project's own D-117/§5j assessment). Best positioned as the parallel theory
contribution, not the drift fix.

### 6.4 Cyclo-dissipativity / indefinite storage
Cyclo-dissipativity requires only SOME storage function, not a non-negative one, and "is not uncommon in
physical systems modeling, especially in the nonlinear case" (SEARCH-LEVEL, arXiv:2003.10143). Neural
learning of storage and supply-rate functions is established (arXiv:2506.06564, arXiv:2309.16032), and the
identification literature notes explicitly that "system identification, even in linear settings, does not
automatically preserve dissipativity or passivity without explicit constraints". Relevant to us as the
formalism that ALLOWS a flat storage direction (the marginal mode) rather than forbidding it. Same C2 verdict
as NI: reference/theory arm.

### 6.5 Do-no-harm weighted regularization (W-PGNN) and orthogonal projection (the project's own line)
W-PGNN adds "a weighted regularization term to the cost function to penalize the difference between the state
and output function of the baseline physics-based and final identified model", so that "the estimated model
follows the baseline physics model functions in regions where the data has low information content"
(ABSTRACT-VERIFIED, arXiv:2405.10429, Liu, Toth, Schoukens, SYSID 2024). It does not force stability, so C1
holds; it is soft, so C2 holds; the weight is a data-informativity measure, so C5 holds.
**But d16 is a direct warning**: an informativity weight computed over state-input data DENSITY (not over the
DC direction of a free mode) will not target our drift, and an information-based weight computed over the DC
direction points the WRONG WAY on a K=0 axis. Re-pointing the weight is exactly the project's Layer-2 task.

Gyorok orthogonal projection (arXiv:2501.05842) remains the in-framework soft steering mechanism; the
orthogonal-by-construction successor (arXiv:2511.01321) is proven statistically consistent with recovery of
the true physical parameters and gives "a clear separation between the physics-based and learning
components", **for input-output models** (ABSTRACT-VERIFIED). The state-space/LFR extension is genuinely
open, which is where the thesis contribution sits. A newer independent line, OrthoReg (arXiv:2606.19145),
"directly penalizes overlap between the symbolic and neural components" for hybrid symbolic-neural systems
and reports improved symbolic recovery and out-of-distribution behaviour versus plain `L2`
(ABSTRACT-VERIFIED). It is worth citing as convergent evidence that the orthogonality idea is being
independently reinvented, and worth reading for its penalty formulation.
**Scope honesty**: none of these is a drift fix. They buy interpretability and non-negation. The project
already scoped it this way and should keep doing so.

---

## 7. Q6: integrator factoring, and whether it can admit a net impulse

Tustin-Net hardcodes the known position-velocity relation in the forward pass (position dynamics discretized
by the trapezoidal/Tustin rule, velocity dynamics by a learned feedforward net), which keeps a position-domain
loss and makes hidden states directly estimable (SEARCH-LEVEL: arXiv:1911.01310, IFAC 2020; practical
accounts arXiv:2408.12266).

**The key answer, and it is negative by algebra, not by experiment.** Our bounded-integral / telescoping
block gets its no-drift guarantee from `sum F = phi(z_N) - phi(z_0)`, which is bounded BECAUSE `phi` is a
single-valued bounded state function. That same property forces the net impulse around any closed state-space
loop to be ZERO, which forbids exactly the hysteretic/path-dependent impulse Coulomb friction produces
(`integral sign(v) dt != 0` even on a closed path). So the mechanism that buys C4-style no-drift is the same
one that breaks C3. **There is no version of "output = derivative of a bounded state function" that admits a
net impulse.** The only escapes are the ones the project already listed: put friction in `f_base` (grey-box,
Route A, conditional on the parametric friction model being adequate), or allow `phi` to depend on internal
STORAGE states so it can be path-dependent in the observed coordinates while the total stays bounded (the
port-Hamiltonian storage layer), which is a strictly weaker guarantee.
Marks for Tustin/telescoping as deliverable: C1 yes, C4 yes, C5 yes, **C2 no, C3 no**. Keep as the
verification reference. Note Phase 1 also corrected the headline number: 1070x on X, 117x on Y, and on Y it
is slightly worse than plain mean removal.

---

## 8. Q7 and Q8, briefly

**Q7 (exponential versus marginal, and marginal carve-outs).** The rollout-stability literature gives the
discriminator our `test_efolding` already implements (contractivity constant `L_h` versus 1; polynomial
smoothness growth at `L_h = 1`, exponential above; Ribeiro et al. Automatica 121:109158). What does NOT exist
in any source found here or in the project's prior passes is a stability-by-construction parametrization with
an EXACT marginal-mode carve-out, that is, contraction on the strictly-stable subspace with rate exactly 1
preserved on a designated integrator subspace. That gap is real and is a candidate contribution.
Adjacent and worth a look: no-regret prediction in marginally stable systems (Ghai et al., PMLR v125), and
prefiltered least squares with guarantees that "apply even to marginally stable systems" (SEARCH-LEVEL) as a
precedent for prefiltering as a marginal-mode conditioning tool.

**Q8 (encoder as a FLOOR problem).** SUBNET reconstructs the current state from past input-output data via a
learned encoder and trains a truncated prediction loss (ABSTRACT-VERIFIED, arXiv:2210.14816; initialization
approach arXiv:2304.02119; continuous-time variant arXiv:2204.09405). Nothing found addresses encoder bias on
an INTEGRATOR mode specifically, and the framework literature assumes a stable baseline. Since Phase 1
demoted the encoder from "DC cause" to "floor", the correct scope is: reduce the 1.7e-5 m floor (longer or
nonlinear velocity map, logged velocities, measured initial conditions) so that per-axis drift can be judged
against a lower floor. It is a measurement-quality fix, not a drift fix.

---

## 9. Candidate table (every candidate against all five constraints)

Order: constraint-clean and cheap first.

| # | Candidate | C1 pole=1 | C2 expressivity | C3 admits friction DC | C4 estimator-side | C5 knowledge-free | Build cost | Verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | **Newton/curvature-aware step restricted to the measured output-DC subspace** (Hessian already computed exactly, PD, 6x6) | yes | yes | yes | yes | yes | tiny | **Top pick for the DC component.** Directly attacks the measured mechanism (parked at ~lr instead of `b* ~ 1e-11`). Untried. |
| 2 | **Decoupled/proximal application of the projection penalty** (AdamW mechanism) | yes | yes | yes | yes | yes | tiny | **Top pick to unblock Layer 2.** Removes the measured beta-saturation ceiling. Untried. Needs a defensible weight-space back-map. |
| 3 | **Update-space projection** (OGD / null-space projection instead of a loss penalty) | yes | yes | conditional | yes | yes | small | Cannot saturate, but does NOT solve the rank-1 dodge. Pair with a better subspace. |
| 4 | **ARTBP (poly-tail) + DECAYING lr schedule** | yes | yes | yes | yes | yes | small (built) | Theoretically indicated pairing that has never been run. Unbiasedness needs a vanishing step to buy convergence. |
| 5 | **Multiple shooting + continuity penalty, post-D-101 lr** | yes | yes | yes | yes | yes | medium | Untested, not refuted (Phase 1 correction). The supervisor's own named position-based alternative. |
| 6 | **Soft power-sign (`F . v`) steering term** | yes | yes | **yes, rewards friction** | yes | yes | small | **Best-founded untried SEPARATOR.** The only criterion that distinguishes drift from friction without knowing the dynamics. Soft, so no class restriction. |
| 7 | **Free-run-consequence / rollout-consistency penalties** (time-increment consistency, commutativity/non-normality) | yes | yes | yes | yes | yes | medium | Prices the accumulation rather than the force. Inherits long-horizon conditioning problems; pair with 4 or 5. |
| 8 | **Re-aimed orthogonal projection (Gyorok), as currently specified** | yes | yes | conditional | yes | yes | built | Negative on its first real test (rank-1 dodged, beta saturates). Revive ONLY with 2 and a better subspace. Remains the thesis contribution for interpretability. |
| 9 | **W-PGNN weighted do-no-harm regularization, re-pointed** | yes | yes | conditional | yes | yes | medium | In-framework and supervisor-authored. Its density weight must be re-pointed; d16 warns a naive informativity weight points the wrong way here. |
| 10 | **Hard output-space projection layer** (HardNet / KKT-hPINN style) | yes | conditional | conditional | yes | yes | medium | Only as good as the annihilated subspace. Full velocity-DC annihilation = mean penalty = C3 fail. |
| 11 | **Adaptive-control robustification** (sigma-mod, projection operator, dead-zone) | yes | soft | sigma-mod NO / projection conditional | yes | weakens (needs a bound) | small | Classical home of the exact failure. Sigma-mod is the mean penalty renamed. The PROJECTION operator is the interesting one: it bounds rather than shrinks. |
| 12 | **Frequency-selective near-DC prior** (kernel / frequency-domain side information) | yes | soft | **NO alone** | yes | yes | medium | Friction is near-DC too. Viable only combined with 6. |
| 13 | **Telescoping / bounded-impulse output (Route A)** | yes | **NO** | **NO (algebraic, section 7)** | yes | yes | built | Verification REFERENCE only, as already decided. Re-run and log `dC`. |
| 14 | **Nonlinear NI with free body motion** | **yes (only structural family that is)** | NO | yes | structural | yes | months | Parallel theory contribution. Theory exists for free-body/semidefinite storage; the learned/forward/LPV realization is the gap. |
| 15 | **Conservative pH (`R = 0`)** | yes with caveat | NO | yes | structural | yes | large | Theorem assumes a strict energy minimum; extrapolation to an exact `K=0` continuum is unproven. Reference arm. |
| 16 | **Contraction / REN / Lipschitz / stable pHNN (`R > 0`)** | **NO** | NO | yes | structural | yes | n/a | **Ruled out** by theory and by our v6b measurement. Do not revisit. |
| 17 | **Encoder floor reduction** | yes | yes | yes | yes | yes | small | Not a drift fix. Reduces the 1.7e-5 m floor so drift can be judged per axis against something lower. |

---

## 10. Recommended composition, and the order to test it

The constraint set admits no single method (this is the project's own D-108 conclusion and nothing found here
overturns it). The composition that respects ALL FIVE constraints and targets each measured source is:

- **For the DC displacement component (X-dominant, ~80% of null drift)**: candidate 1 (curvature-aware step
  on the measured DC subspace) OR candidate 4 (ARTBP with a decaying lr). These attack the two independently
  measured mechanisms, truncation bias and step-size displacement, and they are complementary rather than
  alternative.
- **For the Y anti-damping feedback component**: candidate 6 (soft power-sign steering) plus candidate 7
  (rollout-consistency), because a wrong-sign velocity feedback is exactly an `F . v > 0` object and is
  exactly a long-horizon consequence. Neither is in the current plan.
- **For interpretability and non-negation (the thesis contribution)**: candidate 8/9, with candidate 2 as the
  fix that makes the penalty bite at all.
- **As the verification reference**: candidate 13, with `dC` re-run and logged.
- **As the honest ceiling**: R4 stays EMPIRICAL. Full expressivity and a for-all-weights no-drift guarantee
  are provably incompatible, and nothing in this search contradicts that.

**Test order (cheapest decisive first).**
1. Does a decoupled/proximal penalty escape beta saturation on the same null testbed step 4 used? One run,
   one code change. If no, the whole soft-penalty family is dead under Adam and the plan must move to
   update-space projection or a per-group optimizer.
2. Does a Newton step restricted to the 6-D output-DC subspace land the DC at `b*` and remove the drift,
   WITHOUT the SGD inaction failure (test on the injected-Coulomb rig, so R2 is exercised)?
3. Does a soft `F . v` term reduce the Y anti-damping Jacobian entry (`dW_dY/ddY`, currently +1.43e-8) while
   the injected friction is still learned?
4. ARTBP gate-2 with a decaying lr schedule, per-axis drift.
5. Re-run and log `dC` on the injected-Coulomb rig for the reference row.

---

## 11. What the literature does NOT provide (the contribution surface, updated)

1. **No stability-by-construction parametrization with an exact marginal-mode carve-out** (contraction on the
   strictly-stable subspace, rate exactly 1 on a designated integrator subspace). Confirmed absent again here.
2. **Orthogonal-by-construction augmentation exists only for input-output models** (arXiv:2511.01321,
   ABSTRACT-VERIFIED); the state-space / LFR / LPV extension is open, and that is where this thesis sits.
3. **No source reports the combination we measured**: truncated-BPTT bias plus a sign-like optimizer parking
   a stiff constant on a pole-1 mode inside a simulation-error loss. The pieces are all supported; the
   assembly is ours.
4. **No source addresses making a subspace regularizer effective under a scale-invariant optimizer for a
   dynamics model.** The AdamW decoupling precedent exists for weight decay and the continual-learning
   projection toolkit exists for task subspaces, but the transfer to a physics-augmentation null direction is
   unreported. This is a small, concrete, publishable methodological contribution and it is one code change
   away from being tested.
5. ~~**No source separates a spurious near-DC bias from a genuine friction impulse on a free integrator.** The
   power-sign criterion is the obvious candidate and appears in the literature only in HARD (class-restricting)
   form; its use as a soft SEPARATOR in an expressivity-preserving augmentation is, as far as this search
   goes, unreported.~~
   **RETRACTED 2026-07-25 by a dedicated refutation search (`scripts/gantry/drift-fix-trials/research/
   thread-EF-novelty-and-primary-reads.md`).** The soft one-sided power penalty IS published, twice:
   (a) DiLaR-PINN (Long, Solak, Ajoudani, arXiv:2604.18277, IFAC 2026) uses exactly
   `lambda * sum ReLU(grad_v V . r_phi)`, which for a kinetic `V` is `relu(F . v)` on a residual added to a
   first-principles model, and their stated motivation is our problem verbatim (residual MLPs "may
   inadvertently inject artificial energy"); (b) the same mechanism is long established in
   thermodynamics-informed ML (Jones, Frankel, Johnson, arXiv:2111.14714), where the Macauley bracket "is
   identical in form to the ReLU activation function". The earlier negative was a VOCABULARY artifact: the
   project had only ever searched control-theoretic terms.
   **What survives**: the separate "guaranteed position boundedness on a marginal mode" gap is untouched by
   this retraction; no soft penalty guarantees anything, and conflating the two would either overclaim
   novelty or wrongly abandon a still-valid gap. Also new and genuinely useful: DiLaR reports the SOFT
   variant FAILING against its hard variant (test RMSE 0.4726 versus 0.0504), which is a published negative
   result on exactly the soft-versus-hard axis and is the honest baseline our T3b must be compared to.

---

## 12. Sources

Verification level in brackets. `[A]` = abstract verified this session, `[S]` = search-level snippets,
`[P]` = primary-read on disk by the project.

**Optimizer geometry and implicit bias**
- Loshchilov, Hutter, "Decoupled Weight Decay Regularization", [arXiv:1711.05101](https://arxiv.org/pdf/1711.05101) (ICLR 2019) `[S, mechanism corroborated by multiple sources]`
- Zhang, Zou, "The Implicit Bias of Adam on Separable Data", [arXiv:2406.10650](https://arxiv.org/pdf/2406.10650) `[S]`
- "The Rich and the Simple: On the Implicit Bias of Adam and SGD", [arXiv:2505.24022](https://arxiv.org/pdf/2505.24022) `[S]`
- "The Implicit Bias of Adam and Muon on Smooth Homogeneous Neural Networks", [arXiv:2602.16340](https://arxiv.org/html/2602.16340) `[S]`
- Barrett, Dherin, "Implicit Gradient Regularization", [arXiv:2009.11162](https://arxiv.org/abs/2009.11162) `[S, via project report]`
- "Scaling Limits of Constant-Stepsize SGD at Flat Minima", arXiv:2607.16384; Barakat, Bianchi, [arXiv:1810.02263](https://arxiv.org/abs/1810.02263) `[S, via project report]`

**Subspace-restricted updates and constrained learning**
- Farajtabar et al., "Orthogonal Gradient Descent for Continual Learning", [arXiv:1910.07104](https://arxiv.org/pdf/1910.07104) `[S]`
- "GNSP: Gradient Null Space Projection", [arXiv:2507.19839](https://arxiv.org/abs/2507.19839) `[A]`
- "Restricted Orthogonal Gradient Projection for Continual Learning", [arXiv:2301.12131](https://arxiv.org/pdf/2301.12131) `[S]`
- "ONG: Orthogonal Natural Gradient Descent", [arXiv:2508.17169](https://arxiv.org/html/2508.17169v1) `[S]`
- HardNet, "Hard-Constrained Neural Networks with Universal Approximation Guarantees", [MIT/azizan](https://azizan.mit.edu/papers/HardNet.pdf) `[S]`
- "Physics-Informed Neural Networks with Hard Linear Equality Constraints" (KKT-hPINN), [arXiv:2402.07251](https://arxiv.org/pdf/2402.07251) `[S]`
- "ENFORCE: Nonlinear Constrained Learning with Adaptive-depth Neural Projection", [arXiv:2502.06774](https://arxiv.org/html/2502.06774) `[S]`
- "LMI-Net: LMI-Constrained Neural Networks via Differentiable Projection Layers", [arXiv:2604.05374](https://arxiv.org/pdf/2604.05374) `[S]`

**Truncated BPTT, multiple shooting, rollout stability**
- Tallec, Ollivier, "Unbiasing Truncated Backpropagation Through Time", [arXiv:1705.08209](https://arxiv.org/abs/1705.08209) `[P on disk; the lr-independence of TBPTT divergence and the decreasing-lr schedule confirmed at search level this session]`
- "Adaptively Truncating Backpropagation Through Time to Control Gradient Bias", [arXiv:1905.07473](https://arxiv.org/pdf/1905.07473) `[S]`
- Ribeiro et al., "On the smoothness of nonlinear system identification", [arXiv:1905.00820](https://arxiv.org/pdf/1905.00820), Automatica 121:109158 `[P via project]`
- Turan, Jaschke, multiple shooting for Neural ODEs, [arXiv:2109.06786](https://arxiv.org/abs/2109.06786) `[S]`
- "Controlling Transient Amplification Improves Long-horizon Rollouts", [arXiv:2605.08856](https://arxiv.org/html/2605.08856) `[S]`
- "Towards Scalable One-Step Generative Modeling for Autoregressive Dynamical System Forecasting" (Time Increment Consistency), [arXiv:2605.05540](https://arxiv.org/html/2605.05540) `[S]`
- Sertbas, Kumbasar, state-consistency regularizer, [arXiv:2510.24757](https://arxiv.org/abs/2510.24757) `[P via project; Schur claim refuted by the project]`

**Augmentation, orthogonality, physics-guided**
- Gyorok et al., "Orthogonal projection-based regularization for efficient model augmentation", [arXiv:2501.05842](https://arxiv.org/abs/2501.05842) (L4DC 2025) `[P on disk]`
- Gyorok et al., "Orthogonal-by-construction augmentation of physics-based input-output models", [arXiv:2511.01321](https://arxiv.org/abs/2511.01321) `[A: input-output only, consistency + parameter recovery under mild conditions]`
- "OrthoReg: Orthogonal Regularization for Hybrid Symbolic-Neural Dynamical Systems", [arXiv:2606.19145](https://arxiv.org/abs/2606.19145) `[A]`
- Liu, Toth, Schoukens, W-PGNN, [arXiv:2405.10429](https://arxiv.org/abs/2405.10429) (SYSID 2024) `[A]`
- "Data-driven augmentation of first-principles models under constraint-free well-posedness and stability guarantees", [arXiv:2604.11421](https://arxiv.org/html/2604.11421) `[A; contraction-based, so pole-1 incompatible]`
- Hoekstra et al., "Learning-based model augmentation with LFRs", European Journal of Control (ScienceDirect S0947358025001335) `[P via project]`

**Stability, passivity, negative imaginary, dissipativity**
- Revay, Wang, Manchester, RENs, [arXiv:2104.05942](https://arxiv.org/abs/2104.05942) `[S; strict contraction, excludes pole 1]`
- Neumeier et al., "Stable Port-Hamiltonian Neural Networks", [arXiv:2502.02480](https://arxiv.org/abs/2502.02480) (NeurIPS 2025) `[S; R>0 gives asymptotic stability, R=0 is the conservative knob]`
- "Negative Imaginary Neural ODEs", [arXiv:2504.19497](https://arxiv.org/pdf/2504.19497) `[S; a controller, with a strict DC-gain assumption]`
- Ghallab, Petersen, "Negative Imaginary Systems Theory for Nonlinear Systems: A Dissipativity Approach", [arXiv:2201.00144](https://arxiv.org/pdf/2201.00144) `[S ONLY: my PDF fetch failed, do not cite without primary read]`
- "Output Feedback Consensus for Networked Heterogeneous Nonlinear Negative-Imaginary Systems with Free Body Motion", [arXiv:2011.14610](https://arxiv.org/pdf/2011.14610) `[S; extends nonlinear NI to free body motion]`
- "A New Stability Result for the Feedback Interconnection of Negative Imaginary Systems with a Pole at the Origin", [arXiv:1107.4255](https://arxiv.org/pdf/1107.4255) `[S]`
- "Cyclo-dissipativity revisited", [arXiv:2003.10143](https://arxiv.org/pdf/2003.10143) `[S; storage need not be non-negative]`
- "Learning Neural Controllers with Optimality and Stability Guarantees Using Input-Output Dissipativity", [arXiv:2506.06564](https://arxiv.org/abs/2506.06564) `[S]`
- "Safe Physics-Informed Machine Learning for Dynamics and Control", [arXiv:2504.12952](https://arxiv.org/html/2504.12952v2) (ACC 2025 tutorial) `[S; explicit soft-versus-hard constraint discussion]`
- "Physics-Informed Machine Learning for Modeling and Control", [MERL TR2023-052](https://www.merl.com/publications/docs/TR2023-052.pdf) `[S]`
- Sivaranjani et al., "Control-Oriented System Identification: Classical, Learning, and Physics-Informed Approaches", [arXiv:2512.06315](https://arxiv.org/html/2512.06315v1) `[P via project, D-108 gap-confirming survey]`

**Integrator factoring and marginal-mode identification**
- "Tustin neural networks: a class of recurrent nets for adaptive MPC of mechanical systems", [arXiv:1911.01310](https://arxiv.org/pdf/1911.01310) `[S]`
- "Accounts of using the Tustin-Net architecture on a rotary inverted pendulum", [arXiv:2408.12266](https://arxiv.org/abs/2408.12266) `[S]`
- Ghai et al., "No-Regret Prediction in Marginally Stable Systems", [PMLR v125](https://proceedings.mlr.press/v125/ghai20a/ghai20a.pdf) `[S]`
- "Structure-preserving model reduction for marginally stable LTI systems", [arXiv:1704.04009](https://arxiv.org/pdf/1704.04009) `[S]`

**Regularized and kernel-based identification**
- Pillonetto et al., "Regularization and Bayesian Learning in Dynamical Systems", [arXiv:1511.01543](https://arxiv.org/pdf/1511.01543) `[S]`
- "Kernel-Based Identification with Frequency Domain Side-Information", [arXiv:2111.00410](https://arxiv.org/pdf/2111.00410) `[S]`

**SUBNET / encoder**
- Beintema, Toth, Schoukens, "Deep Subspace Encoders for Nonlinear System Identification", [arXiv:2210.14816](https://arxiv.org/abs/2210.14816), Automatica `[A]`
- "Initialization Approach for Nonlinear State-Space Identification via the Subspace Encoder Approach", [arXiv:2304.02119](https://arxiv.org/pdf/2304.02119) `[S]`
- "Continuous-time identification of dynamic state-space models by deep subspace encoding", [arXiv:2204.09405](https://arxiv.org/pdf/2204.09405) `[S]`
