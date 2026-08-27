# Problem statement after the D1 to D4 diagnostics: a research brief

**Written 2026-07-25**, immediately after the `scripts/gantry/drift-diagnostics/` campaign.
**Purpose: this is the input document for a fresh deep-research session.** It supersedes
`docs/drift-problem-statement.md` on the four questions the diagnostics settled, and it leaves that
document standing everywhere else. Read both; where they disagree, this one is the measurement and the
older one is the argument it replaced.

**Do not re-run the previous sweep.** `docs/drift-literature-sweep-2026-07-25.md` already swept the
*pre-diagnostic* statement with seven agents. Its §10 queue and its §6 corrections are partly obsolete
now, and §5 of this document says exactly which parts. Repeating it would burn the budget on questions
that are now closed by measurement.

**Conclusions, with every number traced to its artifact: `docs/drift-conclusions-2026-07-25.md`.**
That document is the one to read first if the question is "what do we now know and what is
ruled out"; this one is the input for a literature session.

## 0. Evidence grades (unchanged from the parent document)

| grade | meaning |
|---|---|
| **ROBUST** | current frozen rig, 3 seeds, reproduced under 2 independent training protocols |
| **SOLID** | current frozen rig, 3 seeds, one protocol |
| **SINGLE** | current rig, 1 seed or 1 step count |
| **OTHER-RIG** | measured on a previous rig or routing. Not a valid comparator |
| **INFERRED** | argued from theory or a correlate, never isolated by intervention |
| **DERIVED-HERE** | our algebra, stated by no paper. Re-derive before it enters the thesis |

## 1. The system, in one paragraph

A physics-based LPV-LFR baseline of a dual-gantry high-precision motion system (Y-scheduled inertia) is
augmented with a learned parallel dynamic component: a static ANN block whose output is routed as a
**force onto the velocity rows** `(dX, dTheta, dY)` of the physical state, through a `deepSI`
interconnection with a frozen SUBNET encoder. Training is windowed simulation-error BPTT over
`nf = 400` samples (0.1 s at 4 kHz). The deliverable is scored by **2 s free-run simulation** on
held-out data. The two translational axes are free integrators: `K = 0`, one-step poles exactly at
`|lambda| = 1` (ROBUST), which is a physical property of a free-floating stage and must be preserved.

A constant force error `f` on such a row produces position error growing as `f t^2 / 2`, so the 0.1 s
training window prices weakly exactly what the 2 s deliverable is scored on. That gap is the problem.

Rig for every current number: `scripts/gantry/drift-fix-trials/rig.py`, hash `e1b0511a4c`, perfect-match
null (the recorded baseline output IS the target, so the correct ANN output is identically zero and any
drift is manufactured by the estimator), true-state init, encoder frozen, Adam `lr = 1e-7`,
deterministic full-batch over a fixed 256-window bank, 84 steps (past the step-50 transient).
Measured ANN-off floors: **X 1.084e-07 m, Y 1.121e-06 m**. No oracle appears in any threshold.

## 2. What the diagnostics settled, and what that removes from the search

Full write-ups: `scripts/gantry/drift-diagnostics/results/`. One line each on how it was measured, so a
reader can judge the grade.

### 2.1 The optimization-versus-identifiability fork is CLOSED, toward OPTIMIZATION (SOLID)

Measured at the zero-output init, which for the perfect-match null is the correct optimum, on the 2-D
`(dX, dY)` routed-constant subspace at routing (3, 4, 5):

* Hessian positive definite, eigenvalues 200.1 and 3153.7, cond 15.8, stable to 0.44% across two decades
  of the finite-difference step (so not precision-limited; float64 was not needed).
* Minimiser `b* = 6.90e-10`, i.e. **51x below** the parked constant `3.5e-08`.
* Frye's gradient-flatness index `r = 3.9e-16` against a `r > 0.9` flat cutoff
  (Frye et al., *Neural Computation* 33(6), 2021, DOI `10.1162/neco_a_01388`, App. A.4).
* Two-sided bound `||g||/lambda_max = 6.69e-10 <= ||b - b*|| <= ||g||/lambda_min = 1.05e-08`; the parked
  value is **3.3x above the upper bound**, and outside the measured profile interval (`+-1e-08`) at 7.8x
  the threshold. The threshold is the per-window standard error of the loss over the bank, `1.27e-13`,
  because the seed spread and the float32 repeatability are both **exactly zero** (the evaluation is
  bit-deterministic at that point).

Independent corroboration from a different direction (SOLID): changing the training bank from four
static Y positions to continuous Y traversal, a **3030x** increase in within-window Y motion at
comparable static coverage, leaves the parked constant in the same `3e-08` to `5e-08` band and only
flips its sign.

**Consequence for search: the identifiability and optimal-input-design family is no longer the remedy
space.** Data informativity, persistency of excitation, Fisher-information design for integrator modes,
and the whole "is the direction identifiable" branch are now **background**, not candidates. The
previous sweep ranked Gevers et al. 2009 (`10.1109/tac.2009.2034199`) as its top unfetched item on those
grounds; that ranking is void.

### 2.2 I3 is the published curse of memory, measured along the trained direction (SOLID)

Curvature `d' H d` by Pearlmutter HVP along `d = grad_theta(dW_dY/ddY)` measured at the trained point,
`d` held fixed across horizons: `kappa = 30.4 / 417 / 5598` (seed 0), `37 / 522 / 6146` (seed 1),
`44 / 635 / 8525` (seed 2) at `nf_probe = 400 / 800 / 1600`, strictly monotone, giving
**`p = 3.762 / 3.685 / 3.798`, mean 3.749, spread 0.113**.

That sits in the `H^3` to `H^4` band our mapping of Zucchet and Orvieto's `(1 - lambda^2)^-3` law
predicts (NeurIPS 2024, `arXiv:2405.21064`, Eqs. 5 and 6), and it matches the earlier **synthetic**
canonical-gain `H^3.7` to within the seed spread, which **closes the faithfulness caveat**: the
synthetic direction was faithful. The `[3, 4]` band itself is **DERIVED-HERE** and must be re-derived
before it enters the thesis.

The reproduction gate passed first: `jac_self` reproduced the problem statement's full-batch I3 row to
three digits on all three seeds.

### 2.3 The "cure it with a better gradient estimator" reading is WEAKER than it looks (SOLID, new)

This is the finding that should drive the next sweep, and it is not in any existing document.

Along that same trained anti-damping direction, the **gradient component** grows 201x to 925x from
`nf = 400` to `1600` (exponent 3.8 to 4.9). But the **whole gradient** grows nearly as fast (`|g|`
exponent 3.8 to 4.4), so the direction's **share** of the gradient does not consistently improve: the
cosine goes 0.0274 to 0.0619 (seed 0), 0.0398 to **0.0179** (seed 1), 0.0061 to 0.0201 (seed 2), staying
in a 0.6 to 6.2% band throughout.

Adam is **scale-free per coordinate** (Zhuang et al., TMLR 2022, `arXiv:2202.00089`), so a roughly
uniform amplification of the gradient is largely cancelled by the preconditioner rather than converted
into a larger step along `d`. A perfect long-effective-horizon gradient estimator, fed to Adam, is
therefore not obviously the lever.

And the stochastic branch of that family is independently blocked on this mode: Tallec and Ollivier's
ARTBP variance control requires geometrically decaying memory, which is absent at `|lambda| = 1`;
Beatson and Adams (ICML 2019, `arXiv:1905.07006`, Thm 4.1) prove no sampling distribution gives finite
variance for a non-decaying residual, and measure that the Russian-roulette variant this project chose
is the worse of the two; ARTBP empirically made the perfect-match null worse, which Tallec and
Ollivier's §6.1 predicts for a deterministic problem.

**What survives of the estimator route is only its deterministic form** (exact forward-mode
sensitivities: RTRL and its structured approximations), where cost, not correctness, is the obstacle.

**And D5 closes the other half of that fork: the METRIC does not reach the direction either.** In Adam's
own metric the DC and anti-damping directions have comparable Rayleigh quotients (`2.4` to `2.9e7`
against `1.7` to `3.7e7`) despite raw curvatures differing by 50 to 70x, so the preconditioner very
nearly equalises them. What separates them is gradient alignment: 22 to 68% for the DC direction against
0.6 to 4.0% for the anti-damping one. Neither a better estimator nor the existing preconditioner reaches
it, which is a negative result that prunes both obvious remedy families.

### 2.4 The real Telica residual, three measured numbers (SOLID on real data)

Against the fitted (run 71447) LPV-LFR baseline **without** its Coulomb term, train split only, 22 logs,
212,364 one-step residual samples, in logical force units:

1. **Spectrum.** Content below 10 Hz exceeds the 130 to 180 Hz peak by **991x (X)** and **1377x (Y)**,
   and exceeds the measured noise floor by 967x and 127x. The OTHER-RIG figure this replaces had DC
   sitting 60 to 1700x *below* the band peak. Caveat: the Telica logs are closed-loop ILC point-to-point
   moves with no designed content in that band, so this is not a like-for-like refutation.
2. **Velocity odd/even split: NOT SEPARABLE on this data.** Exactly **0.00% of 85,358 gross-sliding
   samples travel backwards**; every log is a single forward stroke. Measured, not assumed. What is
   measurable instead is the constant-like part: mean residual `-157.5 N` (X) and `-83.7 N` (Y) at 315
   and 344 sigma, and `+177.8 N` / `+63.6 N` at rest. A constant and `Fc sign(v)` are exactly collinear
   here, so its size is certain and its attribution is impossible on these logs.
3. **`dF/dv` is NEGATIVE on both axes**: `-173.3` (X) and `-18.8 N/(m/s)` (Y), negative on **22 of 22**
   and **21 of 22** logs, by a joint `[1, v, a]` regression that separates a damping error from a mass
   error (`corr(v, a) = 0.000`, so the binned and joint estimates agree to four digits).

### 2.5 The drift is the residue of a near-cancellation, not a component (SOLID, D6)

The frozen-state decomposition the parent document calls "the single highest-value remaining
measurement" has now been run at 84 steps. Per axis, on the position trajectory:
`dc = xp_frozen - xp_off`, `fb = xp_full - xp_frozen`, which telescope exactly to `xp_full - xp_off`
(measured residual `0.00e+00`).

On **5 of 6** axis-seed pairs the two components are anti-correlated and **each is larger than their
sum**. Seed 1 on X: contributions at **54.3x and 56.0x the ANN-off floor cancelling to 1.7x**. Seed 0 on
X: 60.1x and 24.3x cancelling to 35.8x.

So "which component carries the drift" is the wrong question: neither does, the imbalance does. This
supplies the mechanism behind **I8**, the per-axis trade the parent document calls its most searchable
open problem. T1b crushed the constant, Y reached the floor on 3/3 seeds and X degraded on 2 of 3;
removing one side of a cancelling pair unveils the other at its full 24 to 56x-floor magnitude. The
barrier is not evasion (I4 refuted that) and not the rank-1 support as such: **the intervention breaks a
balance.** Any candidate that suppresses one component alone must be assumed harmful until measured on
both components and both axes.

Caveat: on a `K = 0` axis every constant-force contribution gives the same `t^2` position shape, so the
`-1.000` correlations are near-automatic and carry no mechanistic information. The magnitude ratio is
the finding. The one exception is informative: seed 2 on Y does not cancel (correlation `+0.716`, DC 4%,
feedback 96%), the first isolated measurement of the anti-damping feedback carrying the drift alone.

### 2.6 Neither the estimator nor the metric reaches the anti-damping direction (SOLID, D5)

D5 formed the preconditioned Hessian from Adam's own state at the trained point. In Adam's metric the DC
and anti-damping directions have comparable Rayleigh quotients (`2.4` to `2.9e7` against `1.7` to
`3.7e7`) despite raw curvatures differing 50 to 70x: the preconditioner nearly equalises them. What
separates them is gradient alignment, 22 to 68% versus 0.6 to 4.0%, and the preconditioner does not fix
that. Combined with §2.3, **both obvious remedy families are pruned**, and what is left is a change to
the objective or to the parameterisation rather than to the step rule or the gradient estimate.

D5 also re-measured `b*` at the trained point: `1.81e-9 / 5.49e-10 / 2.00e-9` against parked
`4.61e-8 / 9.57e-9 / 3.02e-8`, a 15 to 25x mis-seating, so §2.1's verdict survives the move off the init
(Cohen et al.'s non-portability warning changes the factor, not the finding). And the retrain reproduced
D2's checkpoints **bitwise on all three seeds**, so the deterministic protocol is verified rather than
assumed.

## 3. The current problem, stated for a searcher

> Windowed simulation-error training of a learned force on the marginally stable velocity rows of an
> LPV-LFR gantry model leaves two measured artefacts: a persistent constant force, and a positive
> (anti-damping) velocity self-feedback. The loss **does** determine the constant, with a
> well-conditioned strict minimum three orders below where the optimizer parks, so this is an
> optimizer-placement problem, not an identifiability problem. The anti-damping component's curvature
> grows as `H^3.75` with the BPTT horizon, matching the published curse-of-memory law, but its share of
> the gradient does not grow with horizon, and the optimizer is scale-free per coordinate, so amplifying
> the gradient does not obviously reach it. **The open question is therefore what change to the step
> geometry, the metric, or the parameterisation moves an adaptive optimizer onto the minimum the loss
> already defines, on a system whose poles must stay exactly at `|lambda| = 1`.**

## 4. The five questions for the new sweep, in priority order

Each states what would count as a good answer, and which vocabularies to try. **No novelty claim should
be made until at least two non-control vocabularies have been tried and named** (this project has three
times found an idea already published under another field's terms).

### Q1. Where does an adaptive optimizer park on a marginal direction, and what moves it?

**ANSWERED IN PART, 2026-07-25, by D5 plus one literature agent. Both named attractors are refuted and
the question has changed shape. Read this before searching further.**

* **Bock and Weiss's limit cycle is not active here, and the "8.9x prefactor gap" was an error of ours.**
  An earlier version of this section, and `docs/drift-literature-sweep-2026-07-25.md` §1 Leg 1, stated
  that they derive without bias correction at `eps = 0`. **Bias correction is that paper's stated
  contribution** (Theorem 2: the bias-corrected trajectories reach the same 2-cycles), and it uses
  nonzero `eps` by design, inside the square root, which is not PyTorch's convention. The `eps = 0` step
  is one tractability step inside the derivation of their eq. (4). More decisively, their bifurcation
  inequality (6), `alpha lambda_max / sqrt(eps) (1 - beta1) < 2 beta1 + 2`, evaluates on our rig to
  `0.543 < 3.8`: we sit on the **stable** side by a factor 7, so the formula never applied and there is
  no gap to explain. (Our evaluation is DERIVED-HERE and their `eps` convention differs from ours; the
  IJCNN 2019 paper that proves inequality (6) is unread.)
* **The edge-of-stability equilibrium is not active either.** D5 measures the preconditioned
  `lambda_max` at `0.23 / 0.66 / 0.27` of `38/eta` at step 84, below threshold on all three seeds and
  still rising, while raw `lambda_max` is static to five digits. This reproduces Cohen et al.'s
  **Appendix D "Corner case"** (extremely small learning rates: sharpness flatlines or rises slowly
  below threshold), except that their mechanism needs the preconditioner to grow and ours shrinks.
* **What is live instead**: the iterate has not arrived. The measured per-step move along the constant
  is `0.005` to `0.013 x lr` (median over 9 runs), so closing the residual gap needs tens of consistent
  steps and the protocol stops at 84. This also **voids Leg 2** of the earlier sweep, which dismissed
  the not-converged reading on the premise that Adam's step is of order `lr`.

*What is still open, and worth searching*: a predictive model for the pre-edge-of-stability transient of
Adam with momentum on a full-batch problem. The nearest hit is the "central flows" stable flow
(Cohen, Damian, Talwalkar, Kolter, Lee, ICLR 2025, `arXiv:2410.24206`, eq. 25 and footnote 41, which
names this regime "gradient flow with a learning rate warmup"), valid for full-batch but for RMSProp
rather than Adam with `beta1 = 0.9`. The extension to Adam is `arXiv:2605.06821` (2026), abstract-only,
and is the single highest-value unread item for this question.

*Vocabularies*: optimisation theory (limit cycles and non-convergence of adaptive methods, implicit bias
of adaptive gradient methods, edge of stability, sign descent, scale-freeness); numerical analysis
(fixed points of the iteration map, stability of the discrete flow); dynamical systems (attracting
2-cycles); econometrics is **not** relevant here, unlike the previous sweep.

### Q2. Reaching a direction the preconditioner suppresses, without touching the model class

We need the step geometry to move along a direction with high curvature and 0.6 to 6.2% gradient
alignment, on a system where the poles must stay exactly marginal.

*A good answer looks like*: a preconditioner, metric or proximal step whose active subspace is selected
from measured curvature rather than from a norm bound, with an argument that it cannot damp a
representable dynamic; or a published account of adaptive methods failing on low-alignment,
high-curvature directions.

*Vocabularies*: optimisation (natural gradient, K-FAC and Gauss-Newton for recurrent models, decoupled
and proximal adaptive methods, AdamW and ProxGen, variable-metric proximal operators, trust region and
curvature-aware steps); numerical linear algebra (preconditioning for ill-aligned spectra); ML systems
(second-order methods at scale). Note the project already owns the AdamW-flavoured result: an in-loss
penalty saturates because it passes through the preconditioner, while an exact prox applied after the
step restores monotone control (SOLID, 5.1x to 27.6x). The question is whether that generalises from a
penalty to a **step**.

### Q3. Exact or structured long-horizon sensitivities at spectral radius exactly 1

Given that the stochastic unbiased family is blocked (§2.3), the live question is deterministic
forward-mode sensitivity and its structured approximations, and specifically their behaviour as the
spectral radius approaches and reaches 1.

*A good answer looks like*: a method with a stated cost and a variance or bias statement that does
**not** assume geometric forgetting, plus evidence at or near `|lambda| = 1`.

*Vocabularies*: ML (RTRL and its approximations UORO, KF-RTRL, SnAp, OK; online learning of RNNs);
control and system identification (sensitivity equations, adjoint versus forward sensitivity, multiple
shooting, quasi-linearisation); scientific computing (tangent linear models, checkpointing schemes,
differentiable simulation of stiff or marginal systems); numerical weather prediction (4D-Var adjoint
practice on marginally stable dynamics, a field that has fought this exact fight for decades and is not
in our citation graph).

### Q4. Parameterisations that stay valid AT the marginal point

Zucchet and Orvieto's own mitigation is a reparametrisation (`gamma(lambda) = sqrt(1 - lambda^2)` plus
`lambda = exp(-exp(nu))`), and **both pieces degenerate exactly at `|lambda| = 1`**, which is where we
live. That is an author-stated hole.

*A good answer looks like*: a parameterisation or normalisation of a recurrent or state-space model that
remains well conditioned at unit spectral radius, and does not impose strict stability.

*Vocabularies*: ML (structured state-space models S4, S5, LRU and their eigenvalue parameterisations
near the unit circle; orthogonal and unitary RNNs; antisymmetric and Hamiltonian RNNs);
numerical analysis and geometric integration (symplectic and variational integrators, structure
preservation for marginally stable systems, why symplectic schemes avoid artificial damping);
control (marginally stable realisation and balanced truncation near the imaginary axis).
**Already refuted, do not chase**: that I3's positive sign is an explicit-Euler artefact. That rests on
AntisymmetricRNN's Proposition 2, which concerns purely imaginary eigenvalues at `s = +-i w`; our
marginal modes are free integrators at `s = 0`, which forward Euler maps to `z = 1` exactly.

### Q5. Two smaller questions with concrete hooks

* **Estimator-induced anti-damping under parameter-varying excitation.** D3 found the anti-damping
  self-feedback is produced by the Y-traversing records: standstill-only training gives
  `dW_dY/ddY = +2.9e-08 / -2.8e-09 / -1.2e-09`, i.e. it vanishes or turns negative on 2 of 3 seeds,
  while ysweep-only gives the full `+2.5e-08` to `+3.0e-08`. Is there any literature on a learned
  component acquiring destabilising feedback specifically from scheduling motion? Try LPV
  identification, gain-scheduling identification, and adaptive control (parameter drift under
  non-persistent excitation, bursting).
* **Excitation design for friction and bias separation.** D4's parity route is blocked because the logs
  are unidirectional. What reversal, dwell and pre-sliding protocols does the friction-identification
  literature prescribe? This would **specify** the missing experiment rather than leaving us to invent
  it. Try tribology and mechatronics (friction identification protocols, pre-sliding and the Generalized
  Maxwell-Slip model), robotics (dynamic parameter identification excitation trajectories), and
  precision motion control.

## 5. What the new sweep must NOT redo

| already settled, do not re-search | why |
|---|---|
| whether the drift direction is identifiable from this excitation | measured: it is (§2.1). The whole informativity, persistency-of-excitation and optimal-input-design branch is background now |
| whether `H^3.7` was a synthetic-direction artefact | measured: it was not (§2.2). Sweep correction #6 is confirmed |
| whether Y-modulation confers identifiability of the constant | measured: it does not, at 3030x modulation depth (§2.1). Sweep correction #7's reasoning is refuted, though its conclusion survives on other grounds |
| whether parity separates bias from friction on the current real data | measured: not separable, 0.00% reverse travel (§2.4). It needs a new experiment, not a new method |
| zero-mean or window-mean priors on the velocity rows | closed before, and now measured dead: the real residual's mean is 315 to 344 sigma from zero |
| longer fixed BPTT windows as a **fix**, lr tuning, Adam-to-SGD swap, Lipschitz or spectral caps, contraction, RENs, strictly-stable port-Hamiltonian, Theta-only routing, velocity-only ANN input, velocity or acceleration-domain loss, re-aiming the orthogonal projection as a drift cure, disturbance or extended-state observers as a separator, Negative Imaginary in open loop | all closed in `docs/drift-problem-statement.md` §7 and unaffected by these measurements. `nf` is a probe here, never a remedy |

**One live correction to the earlier documents**, which the new session should carry: §6 constraint 4's
over-damped-baseline argument is **not supported** by the real data. It reasons that the fitted viscous
parameters sit far above datasheet, so the residual needs a positive `dF/dv`, which is the class that
sign and passivity constraints forbid. The measurement says the residual needs **more** damping, not
less, even against a baseline whose fitted `cg1 = 290` already sits 2.1x above the datasheet 136. Under
constraint 4's own empirical test, a positive-`dF/dv` prohibition would now **pass**. Caveats before
anyone builds on it: one travel direction, one speed range, closed-loop ILC logs, and a residual measured
against a fitted model, so a damping-parameter error and a genuine velocity-dependent residual are the
same object at that level of analysis.

## 6. Hard constraints any candidate must still satisfy

Unchanged from `docs/drift-problem-statement.md` §6 except where §5 above corrects them.

1. **Expressivity, the overriding one.** The deliverable is real nonlinear data with an unknown residual.
   Any candidate that makes a representable dynamic unlearnable is pruned regardless of its drift score.
   Measured as windowed nf-RMS percent-learned on the Coulomb rig, never free-run.
2. **Marginal-preserving.** X and Y one-step poles stay at `|lambda| = 1`: not below (artificial damping),
   not above (anti-damping).
3. **Knowledge-free.** The mechanism may use data properties or the known baseline structure, never an
   assumption about the unknown residual.
4. **No class-restricting hard constraint unless that class is demonstrably absent from the real data.**
   See the correction in §5: the evidence on this test has moved.
5. **Real-data viability**: measurement noise, closed-loop logs, no oracle in any threshold.

## 7. Pointers

| what | where |
|---|---|
| the four diagnostics, their numbers, verdicts and caveats | `scripts/gantry/drift-diagnostics/results/` (`D1` to `D4`, `SUMMARY.md`) |
| every judgement call made while measuring | `scripts/gantry/drift-diagnostics/results/DECISIONS.md` |
| the pre-diagnostic problem statement (still the source for I4 to I8) | `docs/drift-problem-statement.md` |
| the previous seven-agent sweep (partly superseded; see §5) | `docs/drift-literature-sweep-2026-07-25.md` |
| the frozen rig, hash `e1b0511a4c` | `scripts/gantry/drift-fix-trials/rig.py` |
| trained checkpoints, Adam states, unit JSONs | `scripts/gantry/drift-diagnostics/data/` |
| run table, with the hypothesis and outcome of every training run | `docs/gantry-augmentation-problem-log.md` §12 |
