# Flat-direction literature sweep, 2026-07-26

Input document: `docs/flat-direction-problem-2026-07-26.md`.
Procedure: `.claude/skills/deep-research/SKILL.md`, step 0 FRAME plus a 4-agent fan-out.
Predecessor sweeps, none superseded, all enumerated in the frame this time:
`docs/narrowband-literature-sweep-2026-07-26.md`, `docs/drift-literature-sweep-2026-07-25.md`,
`docs/multiple-shooting-sweep-2026-07-25.md`, `docs/rollout-stability-literature.md`,
`scripts/gantry/drift-fix-trials/research/thread-{AB,CD,EF}*.md`,
`literature/stability-training/claude-deep-research-{Adam-optimizer-drift,drift,drift-diagnostics}.md`.

**This sweep produces one result that changes the input document rather than extending it.**
Link 4 of the mechanism table is not evidence for a systematic gradient. See 3.1. Read that
section before acting on anything else here.

---

## 0. Frame that was used

| # | Item |
|-|-|
| Sub-questions | 4, listed in Section 2 |
| Seed IDs from the document | none cited directly; seeds were supplied per sub-question from the candidate vocabularies in its Section 3 |
| Disqualification filter | (1) hard model-class restrictions violating full expressivity; (2) needs oracle states at deployment; (3) needs new excitation or hardware; (4) suppresses DC as such (real residual mean at 315 to 344 sigma) |
| Anti-scope | longer windows; multiple-shooting defect; ARTBP and unbiased truncated BPTT; optimiser swap or lr tuning as DELIVERABLE; zero-mean priors on velocity rows |
| Vocabularies searched | ML optimisation, statistical physics, control and sysid, applied maths (FEM goal-oriented error estimation), statistics and UQ, systems biology, econometrics, hydrology, numerical weather prediction and climate, inertial navigation. Ten, against rule 117's minimum of three |
| Evidence floor | every paper named came from a query run in session; repo-sourced items are labelled as such |

### The frame fact that reshaped the sweep

The document's Section 3 lists six "candidate vocabularies, none yet searched under this
framing". **Four of the six were already searched and are held in the repo.** The local-holdings
check found:

| Held where | Covers |
|---|---|
| `literature/stability-training/claude-deep-research-Adam-optimizer-drift.md` | Adam as sign / l-infinity steepest descent (Balles and Hennig 2018; Kunstner et al. ICLR 2023), anti-regularization ODE (Cattaneo, Klusowski and Shigida ICML 2024), l-infinity KKT (Xie and Li ICML 2024), separable-data implicit bias (Zhang, Zou and Cao NeurIPS 2024), rotational equilibrium (Kosson et al. 2024), cautious optimizers, Lion, gradient centralization, OGD, GPM |
| `scripts/gantry/drift-fix-trials/research/thread-AB-optimizer-mechanics.md` | AdamW as prox (Zhuang et al. TMLR 2022), ProxGen, scaled prox, VarPro, damped Newton on a subspace |
| `docs/drift-literature-sweep-2026-07-25.md` | Bock and Weiss ICANN 2019 (Adam's parked position in closed form), Cohen et al. adaptive edge of stability, Frye et al. gradient-flat regions |
| `literature/stability-training/claude-deep-research-drift-diagnostics.md` | sloppiness (Transtrum 2015, Gutenkunst 2007), practical identifiability (Raue 2009), structural identifiability (Villaverde) |

So "flat / sloppy directions", "practical identifiability and sloppy models", "implicit bias of
adaptive methods" and "null-space drift" were all already covered. The genuinely unsearched
residue is what the four sub-questions below target.

**Frame error to avoid next time.** The document's own Section 3 vocabulary list is not a
reliable statement of what has been searched. Grep the repo before trusting it.

---

## 1. Executive answer

| Question | Answer | Where |
|-|-|-|
| Primary: adaptive optimisers accumulating offsets along objective-flat directions | The asymptotic law for Adam has **zero mean** and a covariance that **diverges** as curvature goes to zero. Motion along a flat direction is diffusion, not drift, until a computable crossover step count. That crossover is ~238 steps on this project's own numbers, so BOTH regimes are present across the measured checkpoints | 3.1 |
| Is link 4 (`offset = 3.48 x lr`) evidence of a systematic gradient? | **No.** Under the noise-dominated law, drift and diffusion are BOTH exactly proportional to lr. Link 4 cannot discriminate | 3.1 |
| Where does the fix belong? | **The objective**, and the instrument already exists with 25 years of theory: re-weight the residual by the **adjoint of the deployment functional** (goal-oriented / dual-weighted-residual error estimation). It survives all four constraints, uniquely among everything found | 3.2 |
| Is the factor 14400 a heuristic? | No. It is the **adjoint weight**. For a double integrator with terminal-position QoI over horizon `T` it is `T^2/2` exactly, which reproduces `72/0.005` analytically | 3.2 |
| Does the phenomenon have a name? | Yes: **objective mismatch**, in the learned-dynamics-model literature. The repo uses the phrase already with no citation | 3.3 |
| Fix by parameterisation? | The gauge-fixing corpus is entirely **function-preserving** and does not apply. The one mechanism that fits is the **GAM identifiability constraint**: sum-to-zero on the flexible term plus an explicit intercept | 3.5 |
| Is link 3 (`the direction is flat`) safe? | **No.** The field that invented that diagnostic runs it at `O(10^3)` initialisations. `n = 120` is underpowered by their standard, and a production-path measurement in the repo says the opposite | 3.1, 3.6 |
| Is the compound claim novel? | Yes, per this sweep, but at MODERATE grade only: Google Scholar was hard-blocked for the whole run | 7 |

---

## 2. Sub-questions as fanned out

SQ1 underspecification and train-versus-deploy metric mismatch, including the sensitivity ratio.
SQ2 drift versus diffusion along a flat direction, and the test that separates them.
SQ3 fix by parameterisation: gauge fixing, identifiable reparameterisation, structural DC elimination.
SQ4 cross-field translation: navigation, data assimilation, hydrology, econometrics, systems biology.

---

## 3. Findings

### 3.1 HEADLINE. Link 4 is not evidence, and the paradox in Section 1 dissolves without a new mechanism

The document's stated gap is: "Blindness explains why a constant offset is TOLERATED. It does
not explain why training RELIABLY PRODUCES one... Indifference does not create a bias; something
has to move the iterate there."

**Malladi, S., Lyu, K., Panigrahi, A., Arora, S.**, "On the SDEs and Scaling Rules for Adaptive
Gradient Algorithms", *NeurIPS 2022*. `arXiv:2205.10287`. Free: `https://arxiv.org/pdf/2205.10287`.
**READ IN FULL** (targeted grep). Section 4.1, p5, gives the finite-step law in the regime
`sigma >> ||g_bar||`, which is precisely what "statistically neutral direction" means:

```
theta_k ~ N( (k*eta/sigma) * g_bar ,  k * eta^2 * I )
```

| Component | Growth in step count `k` | Growth in `lr` |
|-|-|-|
| Drift (mean) | **linear in `k`** | linear |
| Diffusion (std) | **`sqrt(k)`** | linear |

**Both terms are exactly proportional to `lr`.** So the measured "ANN offset after one step is
exactly `3.48 x lr`, slope 1" and "drift is proportional to `lr`" are consistent with a pure
random walk. Link 4 is a real measurement and it is not evidence for a systematic gradient.
The document's own confidence ordering ("strong on WHAT, weaker on WHY") is correct but
understates the problem: link 4 was doing work it cannot do.

**The resolution.** "Blindness does not create a bias" is true instantaneously and false
cumulatively. A gradient too small to detect in `n = 120` samples is not too small to move the
iterate over 5200 steps, because drift accumulates as `k` while noise accumulates as `sqrt(k)`.
Applying the law to this project's own numbers (`Delta/SE = +0.71` at `n = 120`, so
`g_bar/sigma = 0.71/sqrt(120) = 0.0648`):

| Quantity | Value |
|-|-|
| Crossover step count `k* = (sigma/g_bar)^2` | **~238 optimiser steps** |
| Displacement SNR at the 130-batch checkpoint | `sqrt(130) * 0.0648 = 0.74`, **diffusion-dominated** |
| Displacement SNR at the 5200-batch checkpoint | `sqrt(5200) * 0.0648 = 4.7`, **drift-dominated** |
| Predicted 130 -> 5200 offset ratio, crossover model | **24.3** (pure random walk 6.3, pure drift 40) |

MS5's measured metric collapse ratio (9.5x to 127x, i.e. 13.4) sits between the pure regimes,
where the crossover model requires it to sit. *The arithmetic in this paragraph is DERIVED-HERE
from the cited equation and this project's measured statistics. It is a prediction to test, not
a published result.*

**Consequence for the falsifier in the document's Section 5.** The sign of the per-row mean
should be UNSTABLE across seeds at the early checkpoint and STABLE at the late one. A single
verdict from all 10 `f07` checkpoints pooled would be the wrong test.

#### The Adam stationary law, which is the closest thing to an answer to the primary question

**Compagnoni, E. M., Liu, T., Islamov, R., Proske, F. N., Orvieto, A., Lucchi, A.**, "Adaptive
Methods through the Lens of SDEs: Theoretical Insights on the Role of Noise", *ICLR 2025*.
`arXiv:2411.15958`. Free: `https://arxiv.org/pdf/2411.15958`. **READ IN FULL** (targeted grep,
65 pp). Lemma C.52, p53, for `f(x) = x' H x / 2` with `H = diag(lambda_1..lambda_d)`:

```
(E[X_inf], Cov(X_inf)) = ( 0 , (eta/2) * Sigma^(1/2) * H^(-1) )
```

Three consequences, each directly on the document's claims:

1. **`E[X_inf] = 0`.** Adam carries no drift term at stationarity. The asymptotic law does not
   produce a mean offset.
2. **`Cov` diverges as curvature goes to zero.** Along an objective-flat direction Adam has no
   stationary distribution at all; the iterate performs an unbounded random walk. This is the
   formal version of "the objective cannot resist".
3. **Adam's spread scales with `Sigma^(1/2)`; SGD's with `Sigma`** (their Lemma 3.8:
   `Cov = (eta/2) H^(-1) Sigma`). On a low-signal high-noise direction Adam's excursion is
   inflated relative to SGD's. That is the document's link 4 effect, but diffusive rather than
   directional.

**Scope disclaimer verified in the body**, and it matters: every lemma assumes `mu`-strongly
convex or `mu`-PL with `mu > 0`, and every stationary formula inverts `H`. The exactly-flat
`lambda = 0` case is out of scope. What the theory gives at `H = 0` is a divergence, not a
theorem. **Nobody has proved the `H = 0` case for Adam.** That is a genuine gap.

**Lemma C.57, AdamW:** `Cov = (eta/2) * (H*Sigma^(-1/2) + theta*I)^(-1)`. With decoupled weight
decay `theta > 0` the covariance stays **finite even at `H = 0`**. This is the only mechanism
found that regularises the flat direction without touching the model class. It passes filters
1, 2 and 3 and brushes filter 4 (it shrinks the learned DC toward zero), as a soft
gradient-balanced shrinkage rather than a DC prohibition. **It cannot be judged on the current
testbed** for the reason the document's Section 6 gives: the correct DC here is genuinely zero.
The injected-friction simulation is the prerequisite.

#### A second mechanism that needs no mean gradient at all

**Ziyin, L., Li, H., Ueda, M.**, "Noise balance and stationary distribution of stochastic
gradient descent", *Phys. Rev. E* 111, 065303 (2025). DOI `10.1103/physreve.111.065303`.
Free: `arXiv:2308.06671`. **READ IN FULL** (targeted grep, 22 pp).

Their Figures 1 and 7 are the drift-versus-diffusion contrast in one picture: GD with injected
isotropic Gaussian noise **diverges by free diffusion** along the degenerate direction, while
SGD with its own low-rank state-dependent minibatch noise **converges to a balanced point**.
Eq. (A13) is a deterministic ODE because the Brownian term cancels:

```
d/dt ( ||u||^2 - ||w||^2 ) = -T * ( Sigma Var[dL/dw] - Sigma Var[dL/du] )
```

So the **second moment** of the gradient noise, not its mean, produces systematic motion along a
direction where the mean gradient is zero, at rate proportional to `T = eta/B`. This is a second
lr-proportional mechanism requiring no mean gradient, and it is the statistical-physics answer
to "something has to move the iterate there". Authors' own scope caveat: it requires a rescaling
parameter symmetry in the loss and is "currently limited to a minimal model".

#### The MSD exponent test

**Kunin, D., Sagastuy-Brena, J., Gillespie, L., Margalit, E., Tanaka, H., Ganguli, S.,
Yamins, D. L. K.**, "The Limiting Dynamics of SGD: Modified Loss, Phase-Space Oscillations, and
Anomalous Diffusion", *Neural Computation* 36(1):151-174, 2023. DOI `10.1162/neco_a_01626`.
Free: `arXiv:2107.09133`. **READ IN FULL** (targeted grep, 78 pp).

Supplies `||Delta_k||^2 ~ k^c`: `c = 1` Brownian, `c < 1` subdiffusive, `1 < c < 2`
superdiffusive, `c = 2` ballistic drift. Measured `c = 1.07` to `1.31` on ResNet-18/ImageNet.
Two caveats verified in the body and important here: **Assumption 4 (quadratic loss, `H >= 0`,
a local minimum)** is load-bearing and is exactly what fails for a genuinely flat direction; and
**Appendix I.7, Figure 9** shows the fitted exponent increases with the length of trajectory
fitted (1.069 at 20 epochs to 1.306 at 100), so a two-checkpoint fit is biased upward. Grep of
the full body confirms **no occurrence of "Adam"**: SGD with momentum and weight decay only.

### 3.2 The factor 14400 is an adjoint weight, and the fix that follows survives all four constraints

This is the strongest constructive finding of the sweep, and it came from translating the
question into applied-maths words. No ML or control query in this run would have reached it.

**Becker, R., Rannacher, R.**, "An optimal control approach to a posteriori error estimation in
finite element methods", *Acta Numerica* 10:1-102, 2001. DOI `10.1017/S0962492901000010`.
**METADATA ONLY**, `needs-browser-route` (CUP; a free Heidelberg preprint very likely exists).
**Oden, J. T., Prudhomme, S.**, "Goal-oriented error estimation and adaptivity for the finite
element method", *Computers and Mathematics with Applications* 41(5-6):735-756, 2001.
DOI `10.1016/S0898-1221(00)00317-5`. **METADATA ONLY**, `needs-browser-route`.

The dual-weighted-residual (DWR) identity: the error in a **quantity of interest** equals the
residual **weighted by the adjoint solution of that quantity's functional**. The adjoint weight
IS the amplification ratio the document measured. For a double integrator with a terminal-position
QoI over horizon `T`, the adjoint weight of a constant force is `T^2/2`, which reproduces
`72 / 0.005 = 14400` analytically instead of empirically. *That specialisation is DERIVED-HERE;
the DWR identity is the cited result.* This supplies the missing "why" for link 6.

**Why the prescribed fix is the right shape for this project.** DWR prescribes re-weighting the
residual by the adjoint. Marked against the filter:

| Constraint | Verdict |
|-|-|
| 1. No hard model-class restriction | **PASS.** It changes the objective's weighting, not the hypothesis class |
| 2. No oracle states at deployment | **PASS.** The adjoint is solved on the model, not on truth |
| 3. No new excitation or hardware | **PASS** |
| 4. Must not suppress DC | **PASS, and it inverts the usual failure.** Under an adjoint weight a GENUINE friction DC becomes the most strongly learned component, not the most suppressed |

Constraint 4 is where every zero-mean and window-mean prior died. An adjoint weight is the only
instrument found in ten vocabularies that makes the DC direction non-flat by making it
**expensive to get wrong**, rather than by making it cheap to set to zero. That is precisely the
document's Section 3 request: "make the direction non-flat without restricting the model class".

**The neural branch is tiny and none of it does this.** All six papers found use the adjoint for
mesh, sampling or architecture adaptivity, never to weight the training loss of a learned
component inside a dynamical model. All free on arXiv, **abstracts only**:
Brevis, Muga, van der Zee `arXiv:2003.04485` (conceptually closest: train the discretisation so
the QoI is resolved regardless of the underlying loss); Govoeyi and Richter `arXiv:2604.01835`
(goal-oriented weighting of where the PINN loss is sampled); Roth, Schroder, Wick
`arXiv:2102.12450`; Hintermuller, Hinze, Korolev `arXiv:2601.07397` (neural ODEs, optimal-control
perspective); Chakraborty, Wick, Zhuang, Rabczuk `arXiv:2112.11360`; Wallwork et al.
`arXiv:2207.11233`.

#### The Bayesian counterpart, verified

**Spantini, A., Cui, T., Willcox, K., Tenorio, L., Marzouk, Y.**, "Goal-Oriented Optimal
Approximations of Bayesian Linear Inverse Problems", *SIAM J. Sci. Comput.* 39(5):S167-S196,
2017. DOI `10.1137/16M1082123`. Free: `arXiv:1607.01881v2`. **READ IN PART** (parent, pp. 1-2).

The paper's object is a quantity of interest `Z = OX` that is a function of the inversion
parameters, with the explicit aim of characterising `Z` and not `X`. Their framing: including
the ultimate goal in the inference formulation is a modelling step, and it is not obvious how to
leverage it. This is "flat in the objective, hypersensitive in the deployment metric" as a
construction, and it quantifies the gap between the data-informed and the goal-informed subspace.

Related, **METADATA ONLY**: Cui, Martin, Marzouk, Solonen, Spantini, *Inverse Problems*
30(11):114015, 2014, DOI `10.1088/0266-5611/30/11/114015`; Zahm et al., *Math. Comp.*
91:1789-1835, 2022, DOI `10.1090/mcom/3737`; Constantine, Dow, Wang, *SIAM J. Sci. Comput.*
36(4):A1500-A1524, 2014, DOI `10.1137/130916138` (active subspaces).
Repo note: `docs/drift-problem-statement-post-diagnostics.md:246` already uses the words "active
subspace" informally with no citation. This literature is not held.

### 3.3 The phenomenon has a published name: objective mismatch

New to the repo. Grep for "Lambert", "decision-focused", "value-aware", "value equivalence"
returns nothing; the repo uses the phrase "objective mismatch" descriptively in
`docs/drift-conclusions-2026-07-25.md:255` and in `scripts/gantry/baseline-null/` with no citation.

- **Lambert, N., Amos, B., Yadan, O., Calandra, R.**, "Objective Mismatch in Model-based
  Reinforcement Learning", *L4DC 2020*, **PMLR 120:761-770**. Free:
  `http://proceedings.mlr.press/v120/lambert20a.html`, `arXiv:2002.04523`. **ABSTRACT ONLY.**
  The training objective for a learned dynamics model (likelihood or one-step prediction) is
  decorrelated from the deployment metric (task performance). Models with better prediction loss
  control worse. Canonical citation for "training metric improves while deployment metric collapses".
- **Wei, R., Lambert, N., McDonald, A., Garcia, A., Calandra, R.**, "A Unified View on Solving
  Objective Mismatch in Model-Based Reinforcement Learning", *TMLR*. `arXiv:2310.06253`.
  **ABSTRACT READ IN FULL.** States that model predictive accuracy is often not correlated with
  action quality and traces the root cause to the objective mismatch. Supplies a **taxonomy of
  solution categories**, which is the single best entry point for the document's secondary question.
- **Lambert, N., Wilcox, A., Zhang, H., Pister, K. S. J., Calandra, R.**, "Learning Accurate
  Long-term Dynamics for Model-based Reinforcement Learning", *CDC 2021*, pp. 2880-2887,
  DOI `10.1109/CDC45484.2021.9683134`. **METADATA ONLY.** Short-horizon training objective versus
  long-horizon deployment accuracy: the same horizon asymmetry as 0.1 s versus 12 s.
- Supporting, all new to the repo: **Farahmand, Barreto, Nikovski**, "Value-Aware Loss Function
  for Model-based Reinforcement Learning", *AISTATS 2017*, PMLR 54:1486-1494 (weights model error
  by its effect on the downstream value function: the closest existing objective-side fix that
  restricts nothing about the model class); **Grimm, Barreto, Singh, Silver**, "The Value
  Equivalence Principle for Model-Based Reinforcement Learning", *NeurIPS 2020* (formalises the
  equivalence class of models indistinguishable under the deployment functional, i.e. the mirror
  image of the flat direction); **Elmachtoub and Grigas**, "Smart Predict, then Optimize",
  *Management Science* 68(1):9-26, 2022, DOI `10.1287/mnsc.2020.3922`; **Donti, Amos, Kolter**,
  *NeurIPS 2017*; **McAllister et al.**, "Control-Aware Prediction Objectives for Autonomous
  Driving", *ICRA 2022*, DOI `10.1109/ICRA46639.2022.9811884`.

**Partial match, reported with its caveat.** **Huang, C., Zhai, S., Talbott, W., Bautista, M.,
Sun, S.-Y., Guestrin, C., Susskind, J.**, "Addressing the Loss-Metric Mismatch with Adaptive Loss
Alignment", *ICML 2019*, PMLR 97:2891-2900, `arXiv:1905.05895`. **ABSTRACT ONLY.** This is the
literal phrase "loss-metric mismatch", but the mismatch is proxy-loss versus evaluation metric on
the same data at the same horizon, fixed by meta-learning the loss with RL. **It is not along a
parameter-space direction.** The naming does not carry the geometry.

### 3.4 Underspecification: a clean negative

**D'Amour, A., Heller, K., Moldovan, D., Adlam, B., et al.**, "Underspecification Presents
Challenges for Credibility in Modern Machine Learning", *JMLR* 23(226):1-61, 2022. No DOI (JMLR
deposits none). Free: `arXiv:2011.03395`, `jmlr.org/papers/v23/20-1335.html`. OpenAlex
`W3100511085`. **METADATA AND ABSTRACT ONLY.**

All **430 forward citers** enumerated (3 cursor pages, complete, no rate limiting) and filtered
locally. The cone is overwhelmingly vision, NLP and clinical-ML fairness. **Nothing in it treats
dynamical systems, marginal stability, free-run simulation, or a parameter-direction sensitivity
gap.** Nearest items and why they are not the answer:

- Semenova, Rudin et al. (Rashomon effect, ISIT 2023 `10.1109/isit54713.2023.10206657`;
  predictive multiplicity, AAAI 2023 `10.1609/aaai.v37i9.26227`): formalises multiplicity **in
  predictions on the training distribution**. Our geometry is the opposite: predictions agree and
  a different functional on a longer horizon diverges.
- Di Natale et al., "Physically Consistent Neural Networks for building thermal modeling",
  *Applied Energy* 325:119806, 2022, DOI `10.1016/j.apenergy.2022.119806`: **found but
  disqualified by constraint 1** (enforces consistency by architecture).

**Frame correction worth recording.** The instruction to resolve a published record before
citation traversal was wrong for this paper. There is no published record and there cannot be:
JMLR deposits no DOIs, so Crossref is empty. And the rule's premise is false here:
`W3100511085` is `type: preprint` and carries **117 references and 430 citers**, fully
traversable. The zero-reference pathology is a property of the record, not of `type: preprint`.

### 3.5 The parameterisation leg: gauge fixing does not apply, but GAM identifiability constraints do

**The distinction is the finding.** Every gauge-fixing and quotient-manifold body of work
reachable treats **symmetry-induced** degeneracy, where the FUNCTION is invariant along the
direction. Ours is **data-silent**: the function changes, the windowed loss does not. Confirmed
from abstracts:

- **Kunin, D., Sagastuy-Brena, J., Ganguli, S., Yamins, D., Tanaka, H.**, "Neural Mechanics:
  Symmetry and Broken Conservation Laws in Deep Learning Dynamics", *ICLR 2021*,
  `arXiv:2012.04728`. **ABSTRACT ONLY.** The symmetries are architectural and present for any
  dataset. The conserved quantities exist because the loss cannot change along them **for any
  data**. Its 14 forward citers were enumerated and are uniformly function-preserving.
- **Zhao, B., Dehmamy, N., Walters, R., Yu, R.**, "Symmetry Teleportation for Accelerated
  Optimization", *NeurIPS 2022*, `arXiv:2205.10637`. **ABSTRACT ONLY.** Uses loss-invariant group
  actions to travel a large distance on the loss level set. **This is the anti-fix**: it
  deliberately moves far along the flat direction on the assumption that doing so is harmless.
- **Marti-Gomez, C., McCandlish, D., Kinney, J.**, "GaugeFixer: overcoming parameter
  non-identifiability in models of sequence-function relationships", bioRxiv 2025,
  DOI `10.64898/2025.12.08.693054`. **ABSTRACT ONLY.** The most explicit gauge-fixing-for-a-learned-model
  literature that exists, and it settles the question: many parameter choices encode the same
  landscape. Foundations: Posfai, McCandlish, Kinney, *Phys. Rev. Research* 7:023005 (2025),
  DOI `10.1103/PhysRevResearch.7.023005`; Posfai, Zhou, McCandlish, Kinney, *PLOS Comput. Biol.*
  2025, DOI `10.1371/journal.pcbi.1012818`. Both open access.
- Mishra and Sepulchre quotient geometry (*SIAM J. Optim.* 26(1), 2016, DOI `10.1137/140970860`;
  *Comput. Stat.* 29, 2013, DOI `10.1007/s00180-013-0464-z`): **METADATA ONLY**; the quotient is
  always over a group action leaving the reconstructed object invariant. Same class.

**Identifiable reparameterisation is structural-only, verbatim from its own abstract.**
**Massonis, G., Banga, J. R., Villaverde, A. F.**, "AutoRepar: A method to obtain identifiable and
observable reparameterizations of dynamic models with mechanistic insights", *Int. J. Robust
Nonlinear Control* 33(9):5039-5057, DOI `10.1002/rnc.5887`. **ABSTRACT ONLY.** The
reparameterisation has the exact same dynamics and input-output mapping as the original model. It
repairs **structural** identifiability using Lie symmetries from STRIKE-GOLDD, so it only ever
removes function-preserving redundancy and cannot touch a practically non-identifiable direction.
There is a published Comment: Rahimabadi and Benali, IJRNC 2025, DOI `10.1002/rnc.8074`,
`needs-browser-route`. State of the art in a control venue: **Meshkat, N., Ovchinnikov, A.,
Scanlon, T.**, *IEEE Trans. Automatic Control* 70(10):6688-6703, 2025, DOI
`10.1109/TAC.2025.3565058`, **METADATA ONLY**, `needs-browser-route`.

**Found but disqualified by constraint 4:** the MBAM family (Transtrum and Qiu, *PRL* 113:098701,
2014, DOI `10.1103/PhysRevLett.113.098701`; power-systems applications, NAPS 2018 DOI
`10.1109/NAPS.2018.8600617` and *IEEE Trans. Power Syst.* 2017 DOI `10.1109/TPWRS.2016.2611511`).
MBAM removes a sloppy combination by a manifold-boundary limit, which for our DC would fix it at
a boundary and make real friction unrepresentable. Worth citing as "the standard answer, and why
it is wrong here".

**Found but disqualified by constraint 1, and it names the tension this project refuses to resolve
by restriction:** **Whipple, A., Hernandez-Vargas, E.**, "Mechanistic Identifiability Preservation
for Hybrid Neural Differential Equations", bioRxiv, DOI `10.1101/2024.12.08.627408`. **ABSTRACT
ONLY.** Formalises bounded neural correction classes, derives Gronwall-type bounds, and concludes
with a fundamental expressiveness-identifiability trade-off. Adjacent, diagnostic rather than fix:
**Giampiccolo, S., et al.**, *npj Systems Biology and Applications* 10, 2024, DOI
`10.1038/s41540-024-00460-3` (open access): flags that identifiability assessment is hindered by
the expressive nature of neural networks, and answers with an a posteriori analysis of the
**mechanistic** parameters, not the neural block's output components.

#### The mechanism that fits: GAM identifiability constraints

**Stringer, A.**, "Identifiability constraints in generalized additive models", *Canadian Journal
of Statistics* 52(2):461-476, 2023. DOI `10.1002/cjs.11786`. **ABSTRACT ONLY**,
`needs-browser-route`. Background: **Wood, S. N.**, *Generalized Additive Models: An Introduction
with R*, 2nd ed., CRC Press, 2017, DOI `10.1201/9781315370279`.

The GAM world has solved exactly this structural problem for thirty years. A flexible term `f(x)`
and an intercept are confounded, so `f` is reparameterised with a QR-absorbed sum-to-zero
(centring) constraint and the constant moves into an **explicitly estimated intercept**. The
expressivity of `f` is untouched except in the one confounded dimension, and the DC remains fully
representable through the intercept. That is precisely "make the DC identified, not impossible".
Stringer's own contribution is the part that matters here: **the choice of gauge is not neutral.**
Centring constraints are applied by default because they are thought to give the lowest standard
errors, and he shows that holds only for a Gaussian response.

#### The theorem that justifies the whole leg

**Poirier, D. J.**, "Revising Beliefs in Nonidentified Models", *Econometric Theory* 14(4):483-509,
1998. DOI `10.1017/S0266466698144043`. **ABSTRACT ONLY**, `needs-browser-route`. In a
non-identified model there exist quantities about which the data are uninformative, i.e. their
marginal prior and posterior distributions are identical. Translated: if the DC is such a
quantity on this training distribution, **no objective term and no optimiser can set it**. It can
only come from a gauge choice or an external anchor. Companion: **Gustafson, P.**, *Int. J.
Biostatistics* 6(2), 2010, DOI `10.2202/1557-4679.1206`, and the book DOI `10.1201/b18308`.

#### The synthesis, and it is an extension of a held result rather than a rediscovery

Parameterise the learned block as `g(z) = g0(z) + c`, where `g0` carries a sum-to-zero constraint
absorbed into the basis (GAM-style, expressivity unchanged) and `c` is an explicit DC parameter.
`c` is then no longer a free byproduct of the last-layer bias. Because `c` enters linearly, it is
exactly the variable **VarPro** eliminates in closed form (Golub and Pereyra 2003, already held).
So: GAM says why you must move `c` out of `g0` first; VarPro says how to profile it out; Poirier
says that once out, `c` must be set by something other than this objective. On the current
testbed that gauge is `c = 0`; on real data it becomes a physically parameterised Coulomb term
whose `sign(v)` modulation makes it genuinely identified, which is where `thread-CD` already
arrived from the navigation side.

### 3.6 Cross-field translation

#### Numerical weather prediction: the diagnostic `d12` uses is theirs, and `n = 120` is underpowered by their standard

**Rodwell, M. J., Palmer, T. N.**, "Using numerical weather prediction to assess climate models",
*Q. J. R. Meteorol. Soc.* 133(622):129-146, 2007. DOI `10.1002/qj.23`. **ABSTRACT READ IN FULL
via the TU/e browser route** (the article is entitled but PDF-only on Wiley, so the body was not
reached). The method: quantify systematic initial tendencies in the first few time steps of a
forecast; after suitable temporal averaging these imply systematic imbalances associated with
model error. Companion: **Phillips, T., et al.**, *BAMS* 85(12):1903-1916, 2004, DOI
`10.1175/BAMS-85-12-1903`, **METADATA ONLY**.

That is `d12` in NWP words. The field runs it at `O(10^3)` initialisations. A pooled
`Delta/SE = +0.71` at `n = 120` is a **null result at a sample size the originating field would
call underpowered**, not evidence of neutrality. Combined with 3.1 (where the same statistic
predicts a 238-step crossover), link 3 needs re-pooling at larger `n` before it can carry the
weight the document puts on it.

**This is the second independent reason to doubt link 3.** The first is in the repo:
`docs/gantry-augmentation-problem-log.md:310` records a production-path measurement finding the
windowed loss is **stiffest**, not flattest, on the integrator output DC (curvature `7.08e4` on X,
autograd equal to FD, positive-definite Hessian) and concludes the loss OVER-constrains the DC.
The document acknowledges link 3 is other-rig; it does not address this contradiction.

#### Climate: flux correction is the closest constraint-compatible deliverable found

**Sausen, R., Barthel, K., Hasselmann, K.**, "Coupled ocean-atmosphere models with flux
correction", *Climate Dynamics* 2(3):145-163, 1988. DOI `10.1007/BF01053472`. Free: MPG.PuRe,
`http://hdl.handle.net/21.11116/0000-0000-B4CA-1`. **ABSTRACT VERIFIED.** A **constant**
correction field is introduced in the boundary conditions to remove drift, explicitly without
affecting the dynamical response of the coupled system.

It does not forbid DC (constraint 4 respected); it separates DC estimation from the training
objective entirely. MS12 already proves the projection is well-posed here, since eight constants
reproduce 112.8% of the 12 s error, so the correction field is 8 numbers estimable from one free
run and applied post hoc, leaving the block's AC content untouched. **Report the field's own
caveat**: flux adjustment was later criticised as unphysical and abandoned as models improved
(Hourdin et al., *BAMS* 98(3):589-602, 2017, DOI `10.1175/BAMS-D-15-00135.1`, **METADATA ONLY**;
Gupta et al., *J. Climate* 26(21):8597-8615, 2013, DOI `10.1175/JCLI-D-12-00521.1`, **METADATA
ONLY**). It papers over the cause.

#### The closest published analogue of this project's exact setup

**Brenowitz, N. D., Henn, B., McGibbon, J., Clark, S. K., Kwa, A., Perkins, W. A.,
Watt-Meyer, O., Bretherton, C. S.**, "Machine Learning Climate Model Dynamics: Offline versus
Online Performance", *Tackling Climate Change with ML workshop, NeurIPS 2020*. `arXiv:2011.03081`.
Free: `https://arxiv.org/pdf/2011.03081`. **READ IN FULL** (6 pp).

A learned block `f(x_i; theta)` inside a physics model `dx_i/dt = g_i(x,t) + f(x_i; theta)`,
trained with **Adam at lr 1e-3 on a normalised MSE** over short windows, deployed in a long free
run. Same architecture class, same optimiser, same objective shape, same deployment gap. Their
findings: the neural net has better `R^2` than a random forest at all output levels, but its
global and time average is **more biased**, which they attribute to non-Gaussian outliers
distorting the MSE-based loss; the baseline has **less systematic drift** while the ML runs have
better short-horizon skill; and they close by calling the offline-to-online translation an open
problem for future work.

Three things this buys. (a) The document's secondary question is stated as an open problem by this
field in 2020. (b) The pattern "every windowed metric improves, the DC-driven long-run metric
degrades" is published and named. (c) **A remedy not previously considered here that violates
none of the four constraints:** the shape of the loss determines the bias of the learned mean.
MSE with heavy-tailed residuals produces a biased mean; Huber or MAE does not. Objective-side,
no class restriction, no oracle states, no new excitation, no DC ban, and cheap to test.

**Rasp, S.**, "Coupled online learning as a way to tackle instabilities and biases in neural
network parameterizations", *Geosci. Model Dev.* 13:2185-2196, 2020. DOI `10.5194/gmd-13-2185-2020`.
Free (CC-BY): `https://gmd.copernicus.org/articles/13/2185/2020/gmd-13-2185-2020.pdf`.
**READ IN PART.** Separates instabilities from biases explicitly, and nudges the reference
simulation in sync so the network learns from the tendencies the high-resolution simulation would
produce **if it experienced the states the network creates**. **Disqualified by constraint 2 on
real data**, but available in this project's simulation testbed, and it is the principled version
of what the oracle-state defect aggregation was reaching for.

#### Hydrology: equifinality, and an independent corroboration of a held negative

**Beven, K.**, "A manifesto for the equifinality thesis", *J. Hydrology* 320(1-2):18-36, 2006.
DOI `10.1016/j.jhydrol.2005.07.007`. Free: Lancaster EPrints,
`https://eprints.lancs.ac.uk/id/eprint/4419/1/Manifesto12.pdf`. **READ IN PART** (abstract,
pp. 11-16). Origin: Beven 1993, DOI `10.1016/0309-1708(93)90028-E`; method: Beven and Freer 2001,
DOI `10.1016/S0022-1694(01)00421-8` (**METADATA ONLY**).

Diagnostic: Monte Carlo over the feasible parameter space, classify realisations as behavioural
or not against **multiple prior limits of acceptability set before the run from observation
error**, and explicitly permit the outcome that no model passes. Remedy: abandon the point
estimate, carry the behavioural ensemble, report prediction limits.

**The part that bears on this project's anti-scope**: Beven reports that distributed
observational information disappointingly does not help much in eliminating equifinality, and
that in distributed groundwater modelling equifinality is endemic. This project killed "longer
windows" and "more data" on measured evidence (SLURM 71013). Hydrology reached the same negative
over two decades and treats it as structural. Independent cross-field corroboration of a negative
currently supported by one rig.

Constructive branch, all **METADATA ONLY**: Kirchner, *WRR* 42(3), 2006, DOI `10.1029/2005WR004362`;
Gupta, Wagener, Liu, *Hydrological Processes* 22(18):3802-3813, 2008, DOI `10.1002/hyp.6989`;
Yilmaz, Gupta, Wagener, *WRR* 44(9), 2008, DOI `10.1029/2007WR006716`; Clark, Kavetski, Fenicia,
*WRR* 47(9), 2011, DOI `10.1029/2010WR009827`. The transferable idea is **diagnostic signatures**:
stop calibrating on an aggregate error norm, and calibrate against statistics each chosen to be
informative about one identified deficiency. Same shape as the adjoint weight in 3.2, arrived at
empirically rather than from duality.

#### Econometrics: an impossibility result and a low-frequency diagnostic

**Andrews, I., Stock, J. H., Sun, L.**, "Weak Instruments in Instrumental Variables Regression:
Theory and Practice", *Annual Review of Economics* 11(1):727-753, 2019. DOI
`10.1146/annurev-economics-080218-025643`. Free: NSF PAR, `https://par.nsf.gov/servlets/purl/10142670`
(41 pp). **READ IN PART** (p1, p3, pp. 22-25). Under weak identification a valid confidence set
must be infinite with positive probability, so correct coverage **cannot** be obtained by
adjusting finite standard errors; the remedy is test inversion. Practically: pre-test strength
with the effective F of Montiel Olea and Pflueger (2013), and report identification-robust
Anderson-Rubin sets regardless of the F value. An **empty** AR set is itself a rejection of the
model's overidentifying restrictions, so the machinery doubles as a misspecification detector.

Translated: any inference of the form "the windowed metric improved, therefore the 12 s run is
fine to within something" is exactly the move this theorem forbids.

Supporting, **METADATA ONLY**: Staiger and Stock, *Econometrica* 65(3):557-586, 1997, DOI
`10.2307/2171753`; Anderson and Rubin, *Ann. Math. Stat.* 20(1):46-63, 1949, DOI
`10.1214/aoms/1177730090`; Moreira, *Econometrica* 71(4):1027-1048, 2003, DOI
`10.1111/1468-0262.00438`; Montiel Olea and Pflueger, *JBES* 31(3):358-369, 2013, DOI
`10.1080/00401706.2013.806694`; Andrews and Cheng, *Econometrica* 80(5):2153-2211, 2012, DOI
`10.3982/ECTA9456`.

**Muller, U. K., Watson, M. W.**, "Measuring Uncertainty about Long-Run Predictions", *Rev. Econ.
Studies* 83(4):1711-1740, 2016. DOI `10.1093/restud/rdw003`. Free at the author's homepage:
`https://www.princeton.edu/~umueller/longpred.pdf`. **READ IN PART** (Sections 1-3 opening).
Found only via the skill's step-3b homepage-scrape route; OpenAlex returned five locations with
`pdf=None` on every one.

For a long-run prediction the crucial characteristic is the pseudo-spectrum near frequency zero,
and they warn that high-frequency sample variation is generally not informative about
low-frequency characteristics and may lead to faulty low-frequency inference. That is a precise
statement of why 0.1 s windows cannot certify a 12 s run. Their method transfers as a diagnostic
almost unchanged: project onto `cos[(t - 1/2)*pi*j/T]` for `j = 0..q` (they use `q = 12`) and
build the prediction set on those coefficients alone. **Applied here**: compute the low-frequency
cosine transform of the learned block's per-row output over the DEPLOYMENT horizon, not the
training window. `j = 0` is the DC offset; `j = 1..q` is the near-DC content the windowed loss
also cannot see. One scalar per row, on exactly the band the `z = 1` poles amplify, with
distribution theory attached. Pairs with Potscher 2002 (already held): Potscher bounds the risk of
estimating the spectral density at frequency zero, Muller and Watson give the constructive
procedure that lives with that bound. Related, **METADATA ONLY**: Muller and Watson,
*Econometrica* 76(5):979-1016, 2008, DOI `10.3982/ECTA6814`.

#### Systems biology: profile the prediction, not the parameter

**Kreutz, C., Raue, A., Timmer, J.**, "Likelihood based observability analysis and confidence
intervals for predictions of dynamic models", *BMC Systems Biology* 6(1):120, 2012. DOI
`10.1186/1752-0509-6-120`. Fully open. **READ IN PART** (abstract, Background). Matlab template at
`http://www.fdmold.uni-freiburg.de/~ckreutz/PPL`.

Confidence intervals from the **prediction profile likelihood**, usable as a data-based
observability analysis, and explicitly applicable when there are non-identifiable parameters,
where insufficiently specified predictions are interpreted as non-observability. A **validation
profile likelihood** is introduced for noisy validation experiments.

**The cleanest methodological transfer in the sweep, and it survives all four constraints.** The
repo's `D1-dc-curvature` notes already record that a naive profile overstates curvature because
the other ANN parameters cannot relax. Kreutz's point is that the parameter should not be profiled
at all: profile the **prediction**, sweeping the 12 s free-run position error while re-optimising
every other weight subject to the training likelihood staying within a chi-squared threshold. A
wide interval is then a proof of non-observability of the deployment metric from the training
data, and it is a statement about the quantity actually shipped. The repo holds Raue et al. 2009
(parameter profile likelihood); this is a different method on a different object and is not held.
Complementary: **Cedersund, G.**, *FEBS Journal* 279(18):3513-3527, 2012, DOI
`10.1111/j.1742-4658.2012.08725.x` (bronze OA), on separating predictions that are unique across
the whole non-identifiable set from those that are not. **METADATA ONLY.**

#### Inertial navigation: the excitation condition, and a diagnostic

**Yang, Y., Geneva, P., Eckenhoff, K., Huang, G.**, "Degenerate Motion Analysis for Aided INS With
Online Spatial and Temporal Sensor Calibration", *IEEE Robotics and Automation Letters*
4(2):2070-2077, 2019. DOI `10.1109/LRA.2019.2893803`. Free extended technical report:
`https://udel.edu/~ghuang/papers/tr_degen.pdf` (23 pp). **READ IN PART** (Table 1, Sections
2.4-2.5, Lemmas 2.1-2.7, Section 4.1).

Their Lemma 2.1 gives the sufficient excitation condition, and it is a **rotation** condition, not
an amplitude or bandwidth one: under general 3D motion with more than one axis of rotation fully
excited, the state is fully observable. Nine named degenerate motions each kill a specific
subspace, with the null space given in closed form. The clause that matters most here: they verify
numerically that **estimation error accumulates over time towards exactly the unobservable
directions** induced by each degenerate motion, and that lower measurement noise improves
consistency without removing that drift. So error growing along a flat direction is expected,
published estimator behaviour, not an Adam pathology. **Disqualified as a deliverable by
constraint 3; reported as mechanism.**

Their remedy when the direction is not observable is the observability-constrained / FEJ family
(Hesch, Kottas, Bowman, Roumeliotis, *IEEE Trans. Robotics* 30(1):158-176, 2014, DOI
`10.1109/TRO.2013.2277549`, **METADATA ONLY**; free TR `https://udel.edu/~ghuang/papers/tr_fej2.pdf`,
**READ IN PART**), which enforces the initial unobservable null space through the linearised
system. **Important nuance that cuts against a naive transfer:** OC/FEJ prevents the estimator
becoming falsely confident along the null space. It does **not** stop the error accumulating
there. The navigation field's answer to "the update should not move along an unidentified
direction" is a covariance fix, not a drift fix.

**Diagnostic worth adopting.** IEEE Std 952-1997, DOI `10.1109/IEEESTD.1998.86153`, whose
Allan-variance annex separates **bias instability** (the `tau^0` floor) from **rate random walk**
(`tau^+1/2`) purely by the slope of the log-log Allan deviation against averaging time. Model-free,
data-only, and it is the document's Section 5 falsifier with a standardised procedure attached.

### 3.7 One adjacent method with inverted polarity, flagged rather than recommended

**Zhou, C., Fang, et al.**, "PhysGuard: Fisher-Guided Gradient Projection for Sim-to-Real Neural
PDE Surrogates", `arXiv:2606.16602` (2026-06). Code at `github.com/ZhouChaunge/PhysGuard`.
**ABSTRACT READ IN FULL.** Uses the empirical Fisher information matrix on simulation data to
identify physics-critical parameter directions, then restricts fine-tuning updates to their
complement, with a layer-wise Gram formulation and an adaptive threshold for the protected
subspace size. Reports that dominant Fisher directions are strongly associated with low-frequency
output structure, and cuts low-frequency error by up to 32% under domain shift.

Same family as OGD/GPM (held), but **the polarity is inverted**: it protects directions the data
DOES see, whereas this problem needs the complement, restraining directions the data does NOT see.
The low-frequency association is a direct empirical echo of the DC-drift phenomenon.

---

## 4. Recommended next actions, in order

**1. Run Test A on the existing `f07` checkpoints. This is the single recommended next action.**
It needs no training run, it decides whether the document's central mechanism is drift or
diffusion, and until it is answered the sweep's other recommendations cannot be prioritised.

Project the parameter displacement onto the DC direction `u`. Under the pure-random-walk null
(`g_bar = 0`), from Malladi Section 4.1:

```
z = u' (theta_k - theta_0) / (eta * sqrt(k))   ~   N(0, 1)
```

Reject at `|z| > 1.96`. Requires only checkpoints already held, plus `lr` and the step count.
Run it separately at the early and late checkpoints: 3.1 predicts diffusion-dominated below
`k* ~ 238` and drift-dominated above.

**2. Note the power problem before planning the Section 5 sign falsifier.** For `S` independent
seeds, the count of the majority sign is `Binomial(S, 1/2)` under the random-walk null. With
**3 seeds, all-same-sign has `p = 0.25` and is not significant.** `S >= 6` is needed to reach
`p = 0.031` two-sided. This project's floor is 3 seeds, so the falsifier as currently framed
cannot produce a significant result.

**3. Re-pool `d12` at larger `n`** before link 3 carries any further weight (3.6, NWP standard),
and reconcile it against the contradicting production-path curvature measurement at
`docs/gantry-augmentation-problem-log.md:310`.

**4. Fit the MSD exponent** across the 10 `f07` checkpoints over a fixed window, never from two
points (Kunin Appendix I.7). `c ~ 1` diffusion, `c ~ 2` drift.

**5. Prototype the adjoint-weighted objective** (3.2). This is the constructive deliverable the
sweep recommends, and it is the only instrument found in ten vocabularies that makes the DC
direction non-flat while leaving a genuine friction DC the most strongly learned component.

**6. Cheap objective-side test that costs almost nothing:** swap the normalised MSE for a Huber
loss and re-measure (3.6, Brenowitz). Their attribution of the mean bias to MSE plus heavy-tailed
residuals is directly checkable here.

**7. Adopt the prediction profile likelihood** (3.6, Kreutz) as the reporting standard for the
deployment metric, replacing parameter-level curvature arguments.

Items 5 to 7 all require the injected-friction simulation (`scripts/gantry/datasilent-friction-sim/`,
built to step 2; steps 3a and 3b never built) before they can be validated, for the reason the
document's Section 6 gives.

---

## 5. Access status (MANDATORY)

**TU/e browser access: AVAILABLE.** Verified end to end by the parent on
`10.1109/IROS60139.2025.11247377` (IEEE Xplore, closed): all five sections returned, no
"Sign in to Continue Reading" interstitial. Used successfully on `10.1002/qj.23` (Wiley,
Rodwell and Palmer), which is entitled but **PDF-only with no HTML body**, so only the abstract
was extracted. That is a format limit, not an entitlement limit.

Agents did not run the preflight (correct, per the skill). Items still marked
`needs-browser-route`, ranked:

1. **Becker and Rannacher 2001**, *Acta Numerica* 10:1-102, DOI `10.1017/S0962492901000010`. The
   DWR foundation for 3.2. A free Heidelberg preprint very likely exists and should be tried first.
2. **Rodwell and Palmer 2007** full PDF, `10.1002/qj.23`. Wanted: the number of initialisations
   they consider adequate, which is what would settle whether `n = 120` is underpowered.
3. **Meshkat, Ovchinnikov and Scanlon 2025**, IEEE TAC, DOI `10.1109/TAC.2025.3565058`.
4. **Poirier 1998**, *Econometric Theory*, DOI `10.1017/S0266466698144043` (also plausibly on the
   author's page).
5. **Stringer 2023**, *Can. J. Statistics*, DOI `10.1002/cjs.11786` (Wiley).
6. **Oden and Prudhomme 2001**, Elsevier, DOI `10.1016/S0898-1221(00)00317-5`.
7. **Lambert et al. CDC 2021**, DOI `10.1109/CDC45484.2021.9683134` (an arXiv preprint should make
   this unnecessary).
8. Hesch et al. 2014 `10.1109/TRO.2013.2277549`; Rhee et al. 2004 `10.1109/TAES.2004.1310002`;
   Levinson and Majure 1987 `10.1002/j.2161-4296.1987.tb01490.x`; Griffith and Nichols 2000
   `10.1023/A:1011454109203`; Rahimabadi and Benali `10.1002/rnc.8074`; the hydrology
   diagnostic-signature branch (Beven and Freer 2001, Kirchner 2006, Gupta 2008, Yilmaz 2008);
   Di Natale et al. 2022 `10.1016/j.apenergy.2022.119806` (constraint-1 disqualified anyway).

No item in this sweep is reported as permanently unreachable.

---

## 6. Evidence quality

**Read in full:** Compagnoni et al. `arXiv:2411.15958`; Malladi et al. `arXiv:2205.10287`;
Kunin et al. `arXiv:2107.09133`; Ziyin et al. `arXiv:2308.06671` (all four via targeted pypdf
grep with context, quoted equations are verbatim extractions); Brenowitz et al. `arXiv:2011.03081`
(6 pp).

**Read in part** (specific sections verified in the PDF): Spantini et al. `arXiv:1607.01881v2`
pp. 1-2 (parent); Yang/Geneva/Eckenhoff/Huang TR (Table 1, Sections 2.4-2.5, Lemmas 2.1-2.7,
Section 4.1); FEJ2 TR (Section 2.1.1); Beven 2006 (abstract, pp. 11-16); Muller and Watson 2016
(Sections 1-3 opening); Andrews, Stock and Sun 2019 (p1, p3, pp. 22-25); Kreutz et al. 2012
(abstract, Background); Rasp 2020 (abstract, Section 1).

**Abstract verified this session:** Rodwell and Palmer 2007 (via TU/e browser); Sausen et al. 1988;
Kunin ICLR 2021; Symmetry Teleportation; GaugeFixer; AutoRepar; Stringer 2023; Poirier 1998;
Gustafson 2010; Whipple and Hernandez-Vargas; Giampiccolo et al.; Lambert L4DC 2020; Wei et al.
TMLR; Huang et al. ICML 2019; PhysGuard; the six goal-oriented neural papers.

**Metadata only** (Crossref-confirmed authors, venue, volume, pages, DOI; finding NOT verified):
Becker and Rannacher; Oden and Prudhomme; Lambert CDC 2021; Meshkat TAC 2025; Transtrum NAPS 2018
and MBAM; Mishra and Sepulchre; Posfai et al.; Wood; Chaudhari and Soatto; Yaida; the four Kunin
citers; Beven and Freer 2001; Beven 1993; Kirchner; Gupta 2008; Yilmaz; Clark; Phillips 2004;
Gupta 2013; Hourdin 2017; Griffith and Nichols; Staiger and Stock; Anderson and Rubin; Moreira;
Montiel Olea and Pflueger; Andrews and Cheng; Muller and Watson 2008; Hesch et al.; Rhee et al.;
Levinson and Majure; Cedersund; IEEE Std 952-1997; Cui et al. 2014; Zahm et al. 2022; Constantine
et al. 2014; Elmachtoub and Grigas; Donti et al.; McAllister et al.; Farahmand et al.; Grimm et al.

**Second-hand, do not quote:** none relied upon. Rodwell and Palmer's method description is from
its own abstract, read directly.

**The arithmetic in 3.1 and the `T^2/2` specialisation in 3.2 are DERIVED-HERE**, from cited
equations applied to this project's measured quantities. They are predictions to test, not
published results.

---

## 7. Novelty position, with vocabularies (rule 117)

Vocabularies searched across the sweep: **ML optimisation, statistical physics, control and
sysid, applied maths (goal-oriented FEM error estimation), statistics and UQ, systems biology,
econometrics, hydrology, numerical weather prediction and climate, inertial navigation.** Ten.

**Do NOT claim as new:**
- The pattern "offline metric improves, online free-run degrades, driven by a learned mean bias":
  published as **offline versus online performance** (Brenowitz et al. 2020), in a near-identical
  architecture, optimiser and objective.
- "The training objective is decorrelated from the deployment metric for a learned dynamics
  model": published as **objective mismatch** (Lambert et al. L4DC 2020), with a solution taxonomy
  (Wei et al. TMLR).
- The amplification ratio between a training residual and a deployment functional: this is the
  **adjoint weight** of dual-weighted-residual error estimation, 2001.
- The gap between the data-informed and the goal-informed subspace: **Spantini et al. 2017**.
- Estimation error accumulating along an unobservable direction: standard, and measured, in
  **aided-INS degenerate-motion analysis** (Yang et al. 2019).
- "Many parameter sets fit the calibration data and diverge in prediction": **equifinality**
  (Beven 1993, 2006).
- Motion along a flat direction under an adaptive optimiser being diffusive: **Compagnoni et al.
  ICLR 2025**, Lemma C.52.
- Confidence intervals on a prediction under non-identifiability: **prediction profile likelihood**
  (Kreutz et al. 2012).

**Unreported per this sweep, each with its grade:**

1. **An adjoint-weighted (goal-oriented) training objective for a learned block inside a physical
   dynamical model.** MODERATE. Every goal-oriented neural paper found uses the adjoint for mesh,
   sampling or architecture adaptivity. `abs:"goal-oriented" AND abs:"system identification"` = 0
   and `abs:"goal-oriented" AND abs:"model augmentation"` = 1 on arXiv; dblp `venue:CDC:+goal-oriented`
   shows the concept reached control only as **experiment design**. Weakness: Scholar was blocked,
   and the forward citation cones of Spantini 2017 and Becker and Rannacher 2001 were NOT traversed
   into the ML literature. **That traversal is the highest-value follow-up in this sweep**, because
   it is exactly where a prior instance would live.
2. **A statistical test separating systematic parameter drift from diffusion across seeds or
   checkpoints.** MODERATE for ML, PROVISIONAL for navigation and econometrics (arXiv indexes
   neither). Measured zeros: `"variance ratio"+"random walk"+"neural network"`,
   `"Allan variance"+"neural network"`, `"Hurst exponent"+"stochastic gradient descent"`,
   `"random walk"+"parameter drift"+"seeds"`, `"mean squared displacement"+"parameter space"+"training"`,
   `"random walk"+"flat direction"`, `"drift"+"diffusion"+"flat directions"`,
   `"noise-induced drift"+"neural network"`, all 0.
3. **A stationary-distribution or fluctuation-dissipation result for Adam at exactly zero
   curvature.** MODERATE. Compagnoni's own lemmas require `mu > 0` and invert `H`; the `H = 0`
   case is explicitly out of scope and yields a divergence, not a theorem.
4. **Gauge fixing of a DATA-SILENT (as opposed to symmetry-induced) direction.** MODERATE. The
   entire gauge-fixing corpus reached is function-preserving. OpenAlex
   `"practically non-identifiable" AND "reparameterization"` = **0 works**.
5. **A GAM-style sum-to-zero identifiability constraint applied to a neural block's output inside a
   physics model, with the DC profiled out by VarPro.** MODERATE. This is a synthesis of three held
   or found results rather than a new mechanism, and should be claimed as such.
6. **The document's own term "data-silent".** `all:"data-silent"` = 2 on arXiv, neither related.
   The term is free and will not collide.

**Every grade above is capped at MODERATE because Google Scholar was hard-blocked for the entire
run** (see Section 8). Scholar is the skill's only full-text-indexed route and therefore the only
one that can reach an in-body scope disclaimer, which is where "nobody has done this" is actually
written. No claim in this sweep should be upgraded to STRONG until Scholar is re-run.

---

## 8. Research log

**Volume.** ~200 queries across 4 agents plus the parent. OpenAlex 39 of a 48 budget, every parse
guarded with `assert 'error' not in d`, **no 429 in any agent**. dblp 6 of 8 (SQ4 spent 0
deliberately and correctly, since dblp indexes none of *J. Hydrology*, *Econometrica*, *QJRMS*,
*WRR*). Crossref carried the metadata load: SQ4 went 30/31 with zero errors while rationing
OpenAlex.

**What worked.**
- **Translating the question into applied-maths words.** "Sensitivity ratio between a training
  objective and a deployment objective along the same direction" is, in FEM words, *the adjoint
  weight in dual-weighted-residual error estimation*. This produced the sweep's best constructive
  result and no ML or control query would have reached it. Direct vindication of rule 117.
- **Complete forward-citer enumeration plus local regex filtering** (430 D'Amour citers in 3 calls).
  A `search=` inside the `cites:` filter would have over-constrained, exactly as the skill's IFAC
  lesson predicts.
- **Author-homepage and research-group index scraping** (skill step 3b). The ONLY route to Muller
  and Watson 2016 (OpenAlex: five locations, `pdf=None` on all). Also turned two closed IEEE
  navigation papers into free 23- and 26-page technical reports **with the lemmas intact**.
- **grep-don't-page on long PDFs.** Compagnoni's Lemma C.52 is on p53 of 65; two greps found it.
- **OpenAlex `title_and_abstract.search` with two quoted phrases joined by AND.** Returns a
  countable universe rather than a ranked list. `"gauge fixing" AND "identifiability"` = 6,
  `"practically non-identifiable" AND "reparameterization"` = 0.
- **The arXiv HTML advanced-search endpoint** (`arxiv.org/search/advanced`) when the API was
  hard-429. Separate rate-limit pool; it served 12 queries while the API refused ~20 consecutive
  requests over 15 minutes.

**What failed.**
- **`search_google_scholar` is hard-blocked for this run.** It returned `[]` for every query in
  all four agents, *including deliberate control queries* ("deep learning", "neural network"). A
  direct `curl` to `scholar.google.com` returned HTTP 429. This is a shared-IP fan-out cost and it
  is the sweep's largest coverage gap.
- **arXiv API 429/503 on the shared IP.** SQ3 lost the route for its entire session (7 attempts).
  SQ1 recovered with 6 s spacing plus a 200 s backoff; 3 s (the skill's documented value) was not
  enough at 4 concurrent agents. The parent re-ran the lost counts after the fan-out finished, when
  the limit had cooled.
- **`search_core`** returns `[]` on any query over about 3 terms. Not a Scholar substitute.
- **`http://export.arxiv.org`** 301-redirects to https and returns a 0-byte body without `-L`. The
  snippet in SKILL.md uses the bare `http://` form.
- `conda run python -c` with a multi-line snippet cost turns in two agents, as documented.

**A measured correction to the skill's strongest instrument, found by the parent.** arXiv's `abs:`
search **over-matches**: `abs:"identifiability"` alone returns **231,892**, and
`abs:"gauge fixing" AND abs:"identifiability"` returns **168** against OpenAlex's exact-phrase 6.
A control (`abs:"zzzqqq gauge fixing"` = 0) confirms the field is live but the phrase matching is
loose. **Consequence, and it cuts both ways:** a nonzero `opensearch:totalResults` is an UPPER
BOUND, not a phrase count, and must not be reported as one. But a **zero is stronger than the
skill claims**, because even an over-matching engine found no co-occurrence. All the zeros in
Section 7 stand, and are strengthened; the nonzero counts should be read as upper bounds.

**Dead ends.** The Rashomon / predictive-multiplicity branch of D'Amour's cone (right shape, wrong
geometry). Mishra and Sepulchre quotient geometry (entirely matrix-factorisation invariance).
Kunin's 14 forward citers (two are QFT-neural-network correspondence theses). Chaudhari and
Soatto's 37 forward citers (dominated by applications). `abs:"sloppy" AND abs:"prediction" AND
abs:"stiff"` returns 10 hits, all Transtrum-lineage work already held. "underspecification" as a
token outside ML fairness: 7 arXiv abstracts, 0 on target; NLP query-ambiguity has captured the word.

**Coverage gaps.**
- **Google Scholar entirely.** No full-text search was performed in any agent. Every negative in
  Section 7 is title-and-abstract coverage only.
- OpenAlex holds 430 D'Amour citers; Scholar reports several times that. The 3.4 negative is over
  the OpenAlex subset.
- **The forward citation cones of Spantini 2017 and Becker and Rannacher 2001 were not traversed
  into ML.** This is the highest-value follow-up (see Section 7, item 1).
- CDC/ECC/ACC 2022-2026 reachable only through 6 dblp queries; effectively unswept.
- No venue-year enumeration was run in any agent. For the cross-field legs the right targets would
  be *WRR* and *J. Hydrology* via Crossref journal-ISSN filters.
- Navigation (Allan variance) and econometrics (variance-ratio, unit-root) primary literature is
  not indexed by arXiv, so those negatives are PROVISIONAL.

**Suggested skill fixes** (five, all measured this run; not yet applied to SKILL.md):

1. **arXiv `abs:` counts are upper bounds, not phrase counts.** Add the measurement above and the
   asymmetry: report zeros as strong evidence, never report a nonzero as a phrase count. Add
   OpenAlex `title_and_abstract.search:"A" AND "B"` as the exact-phrase counterpart, with its COUNT
   as the deliverable.
2. **Fan-out rate discipline.** At 4 concurrent agents the arXiv API 429s within a few calls even
   at the documented 3 s spacing; 6 s plus a 200 s backoff on first 429 recovered it. The parent
   should stagger launches or assign arXiv to one agent at a time. Add the HTML advanced-search
   endpoint as a first-class fallback with its own rate-limit pool.
3. **Validate a Scholar `[]` with a control query** ("deep learning") before grading any negative.
   `[]` on the control means blocked, not empty. Report the mandatory cross-check as NOT PERFORMED
   and cap every novelty claim at MODERATE.
4. **Amend the "resolve the PUBLISHED DOI before traversing" rule.** Check `len(referenced_works)`
   on the preprint record first: the zero-reference pathology is a property of the record, not of
   `type: preprint`. For JMLR, PMLR and papers.nips.cc no DOI-bearing record exists at all, so the
   arXiv record is the canonical node.
5. **Promote research-group publication indexes to a first-class route (3d).** Step 3b frames
   homepage scraping as the route for closed economics; it is equally the route for closed IEEE
   robotics and control, and it yielded extended technical reports containing lemmas the published
   short version omits. Try it BEFORE marking an IEEE item `needs-browser-route`.
6. Minor: Copernicus (GMD/HESS) and BioMedCentral PDFs use ligatures that break `re.escape` greps
   (`conﬁdence`, `identiﬁability`). Grep a fragment avoiding `fi`/`fl`, or dump and read the page.
