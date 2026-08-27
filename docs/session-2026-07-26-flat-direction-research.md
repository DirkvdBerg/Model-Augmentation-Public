# Session record: flat-direction research, 2026-07-26

**Request.** Read `docs/flat-direction-problem-2026-07-26.md` and invoke the deep-research skill.

**Companion document.** `docs/flat-direction-literature-sweep-2026-07-26.md` holds the full
bibliography, access status, evidence quality, novelty position and merged research log. This
document records what was done, what it means, and what to do next. Where the two overlap, the
sweep is authoritative on citations and this one on conclusions.

---

## 1. What was done

| Step | Outcome |
|-|-|
| Step 0 FRAME (mandatory for document-driven research) | Ran the local-holdings check across the WHOLE repo, not just `literature/` and `docs/`. Found that four of the six vocabularies the input document lists as "none yet searched" are already searched and held |
| Frame presented to the user before querying | 4 sub-questions, disqualification filter, anti-scope, ten vocabularies |
| TU/e browser preflight (parent-owned, per the skill) | **AVAILABLE**, verified end to end on a closed IEEE item |
| 4-agent fan-out | SQ1 underspecification and metric mismatch; SQ2 drift versus diffusion; SQ3 parameterisation; SQ4 cross-field translation |
| Parent follow-up after the fan-out | Chased the ranked `needs-browser-route` queue, re-ran the arXiv counts the agents lost to rate limiting, and measured a correction to the skill's novelty instrument |

**Cost and reach.** ~200 queries. OpenAlex 39 of 48 budget with no rate-limit hit. dblp 6 of 8.
Crossref carried the metadata load. Five papers read in full, eight in part.

---

## 2. Conclusions

### 2.1 The input document's mechanism table needs two corrections, and both weaken it

**Link 4 does not do the work the document assigns it.** The document treats
"offset after one step is exactly `3.48 x lr`, slope 1" and "drift is proportional to `lr`" as
evidence that Adam grants flat directions full-size steps and therefore accumulates a systematic
offset. Under the published finite-step law in the noise-dominated regime (Malladi et al.,
NeurIPS 2022, Section 4.1), the parameter after `k` steps is distributed as
`N((k*eta/sigma)*g_bar, k*eta^2*I)`. The **drift term and the diffusion term are both exactly
proportional to `lr`**. Link 4 therefore cannot discriminate between a systematic gradient and a
pure random walk. It is a real measurement that proves less than claimed.

**Link 3 is doubly unsafe.** The document already flags it as other-rig. Two further problems:

1. The diagnostic `d12` uses (mean signed short-window tendency error, pooled) is NWP's initial
   tendency method (Rodwell and Palmer 2007). That field runs it at `O(10^3)` initialisations
   because it is known to be underpowered below that. `n = 120` with `Delta/SE = +0.71` is a null
   result at an underpowered sample size, not a demonstration of neutrality.
2. It contradicts a production-path measurement already in the repo.
   `docs/gantry-augmentation-problem-log.md:310` records the windowed loss as **stiffest**, not
   flattest, on the integrator output DC (curvature `7.08e4` on X, autograd equal to finite
   difference, positive-definite Hessian) and concludes the loss OVER-constrains the DC. The input
   document does not address this.

So the document's own honesty note ("strong on WHAT, weaker on WHY") is correct, and the WHY leg
is weaker than it says: links 3 and 4 are not merely other-rig, they are non-probative as stated.

### 2.2 The paradox in the document's opening dissolves without a new mechanism

The document's stated gap is that indifference does not create a bias, so something must move the
iterate. The answer is that the statement is true instantaneously and false cumulatively. A
gradient too small to detect in 120 samples is not too small to move the iterate over 5200 steps,
because **drift accumulates as `k` while noise accumulates as `sqrt(k)`**. There is a computable
crossover.

Applying the law to this project's own numbers (`g_bar/sigma = 0.71/sqrt(120) = 0.0648`):

| Quantity | Value | Regime |
|-|-|-|
| Crossover `k* = (sigma/g_bar)^2` | ~238 steps | |
| SNR at the 130-batch checkpoint | 0.74 | **diffusion-dominated** |
| SNR at the 5200-batch checkpoint | 4.7 | **drift-dominated** |
| Predicted 130 to 5200 offset ratio | 24.3 | (pure walk 6.3, pure drift 40) |

MS5's measured collapse ratio of 13.4 sits between the pure regimes, where the crossover model
requires it. This arithmetic is derived here from a cited equation applied to measured
quantities. It is a prediction, not a published result, and it is directly testable on
checkpoints already held.

**The practical consequence:** both regimes are present across the measured checkpoints, so a
single pooled verdict from all ten `f07` checkpoints would be the wrong test.

### 2.3 What the literature actually says about the primary question

Adam's stationary law for a quadratic (Compagnoni et al., ICLR 2025, Lemma C.52) has **mean zero**
and covariance proportional to `eta * H^(-1)`, which **diverges as curvature goes to zero**. So
along a genuinely flat direction Adam has no stationary distribution and performs an unbounded
random walk. Adam's spread scales with `Sigma^(1/2)` where SGD's scales with `Sigma`, so on a
low-signal high-noise direction Adam's excursion is inflated relative to SGD's. That is the
document's link 4 effect, correctly stated: **diffusive, not directional**.

Verified scope disclaimer, and it is a genuine gap: every lemma in that paper assumes strong
convexity or PL with `mu > 0` and inverts `H`. The exactly-flat case yields a divergence, not a
theorem. **Nobody has proved the zero-curvature case for Adam.**

A second, independent mechanism exists that needs no mean gradient at all: the second moment of
the gradient noise produces systematic motion along a direction where the mean gradient is zero,
at rate proportional to `eta/B` (Ziyin, Li and Ueda, Phys. Rev. E 111, 065303, 2025). Their
Figure 1 is the drift-versus-diffusion contrast in one picture.

### 2.4 Where the fix belongs: the objective, and the instrument already exists

The document's secondary question asks whether the fix belongs in the objective, the optimiser or
the parameterisation. The sweep's answer is **the objective**, and the instrument is 25 years old.

In goal-oriented (dual-weighted-residual) error estimation, the error in a quantity of interest
equals the residual weighted by the **adjoint solution of that quantity's functional**. The
adjoint weight IS the document's factor of 14400: for a double integrator with a
terminal-position quantity of interest over horizon `T`, the adjoint weight of a constant force
is `T^2/2`, which reproduces `72/0.005` analytically rather than empirically. Foundations are
Becker and Rannacher (Acta Numerica 10:1-102, 2001) and Oden and Prudhomme (Comput. Math. Appl.
41(5-6):735-756, 2001).

**Why this is the right shape, and why it is the only candidate that is:**

| Constraint | Adjoint weighting |
|-|-|
| No hard model-class restriction | PASS: changes the weighting, not the hypothesis class |
| No oracle states at deployment | PASS: the adjoint is solved on the model |
| No new excitation or hardware | PASS |
| Must not suppress DC | **PASS, and it inverts the usual failure.** Under an adjoint weight a GENUINE friction DC becomes the most strongly LEARNED component, not the most suppressed |

Constraint 4 is where every zero-mean and window-mean prior died. Across ten vocabularies this is
the only instrument found that makes the DC direction non-flat by making it **expensive to get
wrong**, rather than by making it cheap to set to zero. That is exactly the document's Section 3
request: make the direction non-flat without restricting the model class.

The neural branch of goal-oriented estimation is tiny (six papers) and all of it uses the adjoint
for mesh, sampling or architecture adaptivity. **None weights the training loss of a learned
component inside a dynamical model.**

### 2.5 The phenomenon has a published name, and this project has been using it uncited

**Objective mismatch** (Lambert, Amos, Yadan and Calandra, L4DC 2020, PMLR 120:761-770): for a
learned dynamics model the training objective is decorrelated from the deployment metric, with
models that predict better controlling worse. Wei et al. (TMLR, arXiv:2310.06253) give a taxonomy
of solution categories organised along exactly the objective/optimiser/parameterisation axis the
document's secondary question asks about. The repo already uses the phrase "objective mismatch"
descriptively in `docs/drift-conclusions-2026-07-25.md:255` and in `scripts/gantry/baseline-null/`
with no citation. It should be cited.

### 2.6 The parameterisation leg is nearly empty, and the one thing that fits is from statistics

Every gauge-fixing and quotient-manifold body of work reachable treats **symmetry-induced**
degeneracy, where the FUNCTION is invariant along the direction. This project's direction is
**data-silent**: the function changes and the windowed loss does not. These are not the same
problem, and reporting a gauge-fixing paper as a solution would be a false find. Symmetry
teleportation (Zhao et al., NeurIPS 2022) is the anti-fix: it deliberately travels far along flat
directions on the assumption that doing so is harmless.

Classical identifiable reparameterisation (AutoRepar, Massonis, Banga and Villaverde) is
**structural-only** by its own abstract: the reparameterised model has the exact same dynamics and
input-output mapping, so it removes only function-preserving redundancy.

**The one mechanism that fits** is the generalized additive model identifiability constraint,
which has solved this structural problem for thirty years: a flexible term and an intercept are
confounded, so the flexible term carries a sum-to-zero constraint absorbed into its basis and the
constant moves into an explicitly estimated intercept. Expressivity is untouched except in the one
confounded dimension, and the DC stays fully representable. Stringer (Can. J. Statistics 52(2),
2023) adds the caveat that the choice of gauge is not neutral.

**And a theorem that justifies the whole leg.** Poirier (Econometric Theory 14(4), 1998): in a
non-identified model there exist quantities whose marginal prior and posterior are identical, i.e.
the data are uninformative about them. If the DC is such a quantity on this training distribution,
then **no objective term and no optimiser can set it**; it can only come from a gauge choice or an
external anchor.

**Synthesis proposed** (an extension of a held result, not a new mechanism): parameterise the
learned block as `g(z) = g0(z) + c`, with `g0` sum-to-zero constrained in its basis and `c` an
explicit DC parameter. Because `c` enters linearly it is exactly what VarPro (Golub and Pereyra,
already held) eliminates in closed form. GAM says why `c` must leave `g0`, VarPro says how to
profile it out, Poirier says that once out it must be set by something other than this objective.
On this testbed that gauge is `c = 0`; on real data it becomes a physically parameterised Coulomb
term whose `sign(v)` modulation makes it genuinely identified, which is where `thread-CD` already
arrived from the navigation side.

### 2.7 Cross-field: five things other fields already know

1. **Numerical weather prediction** invented `d12` and runs it at `O(10^3)` initialisations
   (Section 2.1).
2. **Climate modelling** has the closest constraint-compatible deliverable: **flux correction**
   (Sausen, Barthel and Hasselmann, Climate Dynamics 2(3), 1988), a constant correction field that
   removes drift explicitly without affecting the dynamical response. MS12 already proves it is
   well-posed here, since eight constants reproduce 112.8% of the 12 s error. The field's own
   caveat is that flux adjustment was later criticised as unphysical and abandoned.
3. **ML-in-climate** has published this project's exact setup: Brenowitz et al. (NeurIPS 2020
   workshop) train a learned block inside a physics model with Adam at lr 1e-3 on a normalised MSE
   over short windows, deploy it in a long free run, and find the network wins every offline metric
   while being **more biased in the mean**, which they attribute to non-Gaussian outliers
   distorting an MSE objective. Their proposed remedy, a **robust loss (Huber or MAE)**, violates
   none of the four constraints and is cheap to test.
4. **Hydrology** calls it **equifinality** and reports, over two decades, that more and more
   distributed data does not resolve it and that it is endemic. Independent cross-field
   corroboration of this project's "longer windows and more data are dead" negative, which is
   currently supported by one rig.
5. **Econometrics** supplies an impossibility result: under weak identification a valid confidence
   set must be infinite with positive probability, so correct coverage cannot be obtained by
   adjusting finite standard errors (Andrews, Stock and Sun, Annu. Rev. Econ. 11, 2019). Any
   inference of the form "the windowed metric improved, therefore the free run is fine to within
   something" is formally forbidden. Muller and Watson (Rev. Econ. Studies 83(4), 2016) supply the
   constructive counterpart: project onto the first `q` low-frequency cosine transforms and build
   the prediction set on those alone. Applied here that gives one scalar per row on exactly the
   band the `z = 1` poles amplify.

Plus, from **systems biology**, the cleanest methodological transfer found: the **prediction
profile likelihood** (Kreutz, Raue and Timmer, BMC Syst. Biol. 6:120, 2012). Profile the
deployment metric, not the parameter, re-optimising all other weights subject to the training
likelihood staying within a chi-squared threshold. This fixes the flaw the repo's own
`D1-dc-curvature` notes record, namely that profiling a parameter without letting the rest relax
overstates curvature. The repo holds Raue 2009 (parameter profile likelihood); this is a different
method on a different object.

### 2.8 The underspecification framing came back negative

All 430 forward citers of D'Amour et al. (JMLR 2022) enumerated and filtered. The cone is
overwhelmingly vision, NLP and clinical-ML fairness, and nothing in it treats dynamical systems,
marginal stability, free-run simulation, or a parameter-direction sensitivity gap. The nearest
neighbours (Rashomon effect, predictive multiplicity) have the wrong geometry: they concern
multiplicity in predictions on the training distribution, whereas here predictions agree and a
longer-horizon functional diverges.

---

## 3. Recommended next actions

**One recommendation, and it needs no training run.**

**Run the per-checkpoint z-test on the existing `f07` checkpoints.** Project the parameter
displacement onto the DC direction `u`. Under the pure-random-walk null (`g_bar = 0`):

```
z = u' (theta_k - theta_0) / (eta * sqrt(k))   ~   N(0, 1)
```

Reject at `|z| > 1.96` for systematic drift. It needs only checkpoints already held plus `lr` and
the step count, and it decides whether the document's central mechanism is drift or diffusion.
**Run it separately at the early and late checkpoints**, because Section 2.2 predicts
diffusion-dominated below `k* ~ 238` and drift-dominated above.

Then, in order:

2. **Before planning the document's Section 5 sign falsifier, note the power problem.** For `S`
   seeds the majority-sign count is `Binomial(S, 1/2)` under the random-walk null. With **3 seeds,
   all-same-sign gives `p = 0.25` and is not significant**. `S >= 6` is needed for `p = 0.031`
   two-sided. This project's floor is 3 seeds, so the falsifier as framed cannot produce a
   significant result.
3. **Re-pool `d12` at larger `n`**, and reconcile it against the contradicting curvature
   measurement at `docs/gantry-augmentation-problem-log.md:310`.
4. **Fit the MSD exponent** across the ten `f07` checkpoints over a fixed window, never from two
   points. `c ~ 1` diffusion, `c ~ 2` drift.
5. **Prototype the adjoint-weighted objective** (Section 2.4). This is the constructive deliverable
   the sweep recommends.
6. **Swap the normalised MSE for a Huber loss and re-measure** (Section 2.7, item 3). Nearly free.
7. **Adopt the prediction profile likelihood** as the reporting standard for the deployment metric,
   replacing parameter-level curvature arguments.

Items 5 to 7 all need the injected-friction simulation
(`scripts/gantry/datasilent-friction-sim/`, built to step 2, steps 3a and 3b never built) before
they can be validated, for the reason the input document's Section 6 gives: on the current testbed
the correct DC is genuinely zero, so any method that suppresses DC will look successful here while
being exactly wrong on real data.

---

## 4. Honest limits of this session

- **Google Scholar was hard-blocked for the entire run.** It returned empty for every query in all
  four agents including deliberate control queries, and a direct request returned HTTP 429. Scholar
  is the skill's only full-text-indexed route and therefore the only one that can reach an in-body
  scope disclaimer, which is where "nobody has done this" is actually written. **Every novelty
  claim from this session is capped at MODERATE and none should be upgraded to STRONG until
  Scholar is re-run.**
- **The arXiv API rate-limited the shared IP.** One agent lost the route for its whole session.
  The parent re-ran the lost counts after the fan-out finished.
- **A measured correction to the novelty instrument.** arXiv's `abs:` search over-matches:
  `abs:"identifiability"` alone returns 231,892, and a two-phrase AND returned 168 against
  OpenAlex's exact-phrase 6. A nonzero count is therefore an **upper bound, not a phrase count**.
  A **zero is stronger than the skill claims**, because even an over-matching engine found no
  co-occurrence. All the zeros in the sweep stand and are strengthened.
- **The highest-value follow-up was not done:** the forward citation cones of Spantini 2017 and
  Becker and Rannacher 2001 were not traversed into the ML literature. That is exactly where a
  prior instance of an adjoint-weighted objective for a learned dynamics block would live, so the
  novelty claim in Section 2.4 is provisional until it is run.
- **No venue-year enumeration** was run in any agent, and CDC/ECC/ACC 2022-2026 is effectively
  unswept.
- **TU/e browser access was AVAILABLE** and verified, so no item is permanently unreachable. The
  ranked outstanding queue is in the sweep document, Section 5.

---

## 5. Proposed skill revisions, not yet applied

Six concrete fixes to `.claude/skills/deep-research/SKILL.md`, all measured this run, are listed in
the sweep document Section 8. The two most important: record that arXiv `abs:` counts are upper
bounds with the zero/nonzero asymmetry spelled out, and require a control query before grading any
Google Scholar negative. **These are proposals; SKILL.md has not been edited.**
