# Multiple shooting versus windowed BPTT on a marginally stable plant

**Literature sweep, 2026-07-25.** Run under `.claude/skills/deep-research/SKILL.md`.
**Decision this unblocks:** replace or augment windowed BPTT with multiple shooting for training the
learned block inside the LPV-LFR baseline, or not.
**Context read:** `docs/drift-conclusions-2026-07-25.md` sections 3 and 4 (only).

---

## 0. The one-line answer

**Our pipeline is already a multiple-shooting method, with the shooting nodes eliminated by the SUBNET
encoder and the continuity defect omitted entirely.** The literature's multiple shooting is therefore not
a replacement for what we do; it is the *addition of the term we left out*. That term (the defect
`d_i = x_i0 - x_{i-1}[m_i]`) is exactly what makes short segments compose into the long-horizon objective
the deliverable is scored on, and it is the only construct found in this sweep that changes what the loss
measures without restricting the learned block or touching the poles.

Whether it *helps* is not settled by the literature and the sweep will not settle it. What the literature
does settle is that the construct is well-posed at `|lambda| = 1`, that the equivalence is exact, and that
the segment length is a free design parameter rather than a thing to be tuned against the drift.

---

## 1. Findings

### 1.1 (a) The recipe, and the property it buys

Three papers carry the whole answer. All three were read in full text.

**[R1] Ribeiro, A. H., Tiels, K., Umenberger, J., Schön, T. B., Aguirre, L. A.**, "On the smoothness of
nonlinear system identification", *Automatica* 121:109158, 2020. DOI `10.1016/j.automatica.2020.109158`.
Free: `arXiv:1905.00820`. Code: `github.com/antonior92/MultipleShootingPEM.jl`. **FULL-READ.**

This is the paper that states the property rather than the slogan.

*Recipe.* Split the record of length `N` into `M` intervals with breakpoints `m_i`. Each interval `i`
carries its **own initial state `x_i0` as a decision variable**. The cost is the length-weighted sum
`V_M = sum_i (dm_i / N) V_i`. Continuity is imposed as **hard equality constraints**
`x_{i-1}[m_i] = x_i0`, solved by a trust-region SQP with a merit function `phi(theta; mu) = V(theta) +
mu ||c(theta)||` and a monotonically increasing penalty parameter `mu` (their Appendix G, Algorithm 2).

*Property 1, the equivalence.* Their **Theorem 2**: if the constraints hold, `V = V_M` exactly. Their
**Corollary 3**: `(theta*, x0*)` is a global solution of the single-shooting problem **iff** there exist
node states making the multiple-shooting problem globally solved. Stated in the body:

> "Multiple shooting is equivalent, in the sense of Theorem 2 and Corollary 3, to solving the original
> (single shooting) problem **regardless of the choice of simulation interval** `dm_max`."

This is the load-bearing sentence for our decision. It says the segment length is free: a short segment
does **not** buy a short-horizon objective. It buys the *full-record* objective computed over short
gradient paths. That is precisely the trade the drift problem needs, and it is why multiple shooting is
categorically different from "a longer fixed BPTT window" (already refuted on our rig).

*Property 2, the conditioning.* Their **Theorem 1**, with `L_h` the Lipschitz constant of the state map:

| | `L_h > 1` | `L_h = 1` | `L_h < 1` |
|---|---|---|---|
| Lipschitz constant of `V` | `O(L_h^{2N})` | **`O(N)`** | `O(1)` |
| beta-smoothness (Lipschitz constant of `grad V`) | `O(L_h^{3N})` | **`O(N^3)`** | `O(1)` |

*Property 3, the mechanism, in their own words:*

> "the mechanism used here is not to take the system outside of the chaotic regime, but rather avoid
> simulating the system for too long."

That sentence is the disqualification-filter clearance: the method does not stabilise, damp, cap or
otherwise touch the model's spectrum. It changes the *optimisation problem*, not the *model class*.

**[R2] Turan, E. M., Jäschke, J.**, "Multiple Shooting for Training Neural Differential Equations on
Time Series", *IEEE Control Systems Letters* 6:1897-1902, 2022. DOI `10.1109/LCSYS.2021.3135835`.
Free: `arXiv:2109.06786`. **FULL-READ.** (Already held as METADATA in
`docs/drift-literature-sweep-2026-07-25.md` C5; now read.)

The soft-constraint recipe, which is the one implementable in a PyTorch/Adam pipeline. Partition
`[t0, tf]` by a grid `t0 = tau_0 < ... < tau_Ns = tf`; the states at the grid points are additional
decision variables ("shooting variables"); each interval is an independent IVP; the resulting trajectory
is discontinuous early in training and becomes continuous as the constraints are enforced. Constraint
handling, two options and an explicit warning about each:

- **Penalty:** `phi = C(z) + rho Q(h(z))`. Quadratic `Q` is the common choice, but the `l1`, `l2`
  (not squared) and `l_inf` norms are **exact penalty functions**: under standard assumptions a single
  minimisation at some finite `rho*` yields the constrained solution. "a too large `rho` can result in
  numerical issues, while a too small `rho` may result in constraint violation".
- **Augmented Lagrangian:** `phi = C(z) + sum h_i^T v_i + rho sum h_i^T h_i`, with `v` and `rho`
  updated in an outer loop (their Algorithm 1). This is what they use in both examples; they report the
  plain penalty method is also feasible but that "the penalty parameter strongly influences the fit".

Their motivating failure is directly ours in a different coordinate: single shooting on **oscillatory**
data produces a "flattened out" or low-frequency trajectory that does not describe the data, and they
attribute it to spectral bias plus local minima. An undamped oscillation is a marginal mode. They fix it
with multiple shooting on a cascaded-tanks benchmark and a synthetic spiral.

**[R3] Prabhu, S., Rangarajan, S., Kothare, M.**, "A condensing approach to multiple shooting neural
ordinary differential equation", `arXiv:2506.00724`, 2025. Free on arXiv. **FULL-READ (skim + targeted
grep).**

The third handling of the node states: **condensing** (Bock & Plitt 1984; Albersmeyer & Diehl 2010),
which *eliminates* the node states rather than carrying them. Their contribution is doing it under
**first-order optimisers such as Adam**: the update to the continuity variables `dx` and the parameters
`dp` is constrained to satisfy the first-order Taylor expansion of the equality constraints at each
iteration. They state plainly why this was missing: prior constrained-NN work handles only linear or
linearly separable equality constraints, which shooting constraints are not. **Honest negative in their
own conclusion:** on several of their systems MS-NODE "accurately captures the training data but fails to
generalize to unseen testing data".

**Summary of the three node treatments** (the taxonomy is stated in [R4] below):

| Treatment | Node states are | Cost | Source |
|---|---|---|---|
| Free variables + hard constraints | decision variables, SQP | model complexity scales with `M`; constrained solver | Bock 1981; [R1] |
| Free variables + penalty / aug. Lagrangian | decision variables, unconstrained solver | one extra hyperparameter (`rho`) or an outer loop | [R2] |
| Condensed | eliminated, first-order-consistent | needs the sensitivity of the defect | [R3] |
| **Encoder** | **eliminated by a learned map of past I/O** | **no extra variables at all** | **[R4] — this is us** |

### 1.2 The finding that changes the decision: we are already doing multiple shooting

**[R4] Beintema, G., Tóth, R., Schoukens, M.**, "Nonlinear state-space identification using deep encoder
networks", *PMLR* 144 (L4DC 2021), pp. 241-250. Free: `arXiv:2012.07697`. **FULL-READ (targeted grep).**
Journal version: *Automatica* 156:111210, 2023 ("Deep subspace encoders for nonlinear system
identification").

Its own keyword list ends with "**Multiple Shooting**". From the body:

> "(ii) the multiple shooting method (Bock, 1981) which splits the time series into multiple sections
> where each section has its own independent loss function. This splitting operation has recently been
> shown to have a smoothing effect on the loss function and its gradient (Ribeiro et al., 2020) ... How
> to estimate the initial state at the start of each section remains one of the main issues in
> successfully applying the multiple shooting method. Two approaches are commonly used: (i) Setup the
> initial states as parameters of the optimization (Bock, 1981), this however scales the model complexity
> with the number of sections, and (ii) estimate the initial state by using equality constrains to the
> final state of the previous section (Ribeiro et al., 2020), this constraint optimization is
> considerably more involved. This paper proposes a new approach to the initialization problem by using
> an encoder function."

And, disambiguating it from the estimator-side family this project already closed:

> "Our approach and multiple shooting works on the level of the **cost function** whereas TBTT is a
> **gradient computation method**."

**Consequences for the decision, stated flatly.**

1. "Replace windowed BPTT with multiple shooting" is not the available move. Our `nf = 400` windows with
   encoder re-initialisation **are** the shooting segments; the encoder **is** the node elimination. The
   only thing missing relative to [R1]/[R2] is the **continuity constraint between consecutive segments**.
2. Our segments are currently **fully decoupled**: each window is independently re-anchored to measured
   data, so nothing in the objective ever prices what happens across a window boundary. This is the
   mechanism `docs/rollout-stability-literature.md` (already in the repo) describes as the encoder
   "re-cleaning" the state every 0.1 s. Read through [R1] Theorem 2, it is the exact statement that our
   objective is **not** equivalent to the 2 s simulation-error problem, because the equivalence is
   conditional on the defects being zero and we never form them.
3. The actionable delta is therefore small and specific: **compute the defect and price it**. Take
   consecutive (or overlapping) windows, propagate the end state of window `i-1` forward, and add
   `||x_i0^encoder - x_{i-1}[m_i]||^2_W` to the loss. Under [R2] this is the penalty form of multiple
   shooting; under [R4] the encoder supplies `x_i0` for free, so no new decision variables are
   introduced. No stability constraint, no class restriction, no oracle.
4. It also reframes the repo's own record. `tasks/lessons.md` (`prove-overconstraint-dont-multiply-methods`)
   asserts multiple shooting "was [tried], and failed on the augmentation: Optuna 69399 best = epoch 0".
   `docs/drift-critical-analysis.md` §3 row 4 already corrects this: 69399 is confounded by the D-101 lr
   bug and "multiple shooting proper was never cleanly run post-D-101". This sweep supports the
   correction and sharpens it: **69399 was a long-horizon single-shooting sweep, not multiple shooting at
   all**, since without a defect term there is nothing multiple-shooting about it.

### 1.3 (b) Does any of it address a marginally stable plant?

**Short answer: yes in two of the three vocabularies, and one of them treats `|lambda| = 1` as a named
first-class case. Nobody treats a free integrator specifically.**

**Control / sysid vocabulary — the boundary case is in the theorem, not in an experiment.**
[R1] Theorem 1's middle column *is* the marginal case: `L_h = 1`. It gives `L_V = O(N)` and
`L'_V = O(N^3)`, and the body says the constants "may ... blow up exponentially (**or polynomially for
some limit cases**) with the maximum simulation length". So the literature does state what happens at the
boundary, and the statement is favourable: at `|lambda| = 1` the ill-conditioning that motivates multiple
shooting is **polynomial, not exponential**. Their experiments, however, are on chaotic and unstable
systems (`L_h > 1`), never on a marginal one. No paper found in this sweep runs multiple shooting on a
plant with poles exactly at 1.

**Data assimilation vocabulary — this is the field that names the case.**
The zero-Lyapunov-exponent direction is not an edge case there; it has a standard name, the
**neutral** subspace, and the central object of the field's theory is the **unstable-neutral subspace**.

**[R5] Grudzien, C., Carrassi, A., Bocquet, M.**, "Asymptotic forecast uncertainty and the unstable
subspace in the presence of additive model error", *SIAM/ASA J. Uncertainty Quantification* 6(4), 2018.
Free: `arXiv:1707.08334`. **FULL-READ (targeted grep).** From the abstract and introduction:

> "for filters and smoothers in perfect, linear, Gaussian models, the distribution of forecast errors
> asymptotically conforms to the **unstable-neutral subspace**. Specifically, the column span of the
> forecast and posterior error covariances asymptotically align with the span of backward Lyapunov
> vectors with **non-negative** exponents."

and it cites the chain of proofs of that statement (Gurumoorthy et al.; **[R6]** Bocquet, Gurumoorthy,
Apte, Carrassi, "Degenerate Kalman filter error covariances and their convergence onto the unstable
subspace", *SIAM/ASA JUQ* 5(1):304-333, 2017, `arXiv:1604.02578`; Bocquet & Carrassi), with a necessary
and sufficient criterion phrased "in terms of the **detectability of the unstable-neutral subspace**".

Translated into our terms: the estimation error of this class of problem lives in the span of the
non-negative-exponent modes, and a free integrator (`lambda = 0` exponent, `|z| = 1`) is exactly such a
mode. The effective dimension of the state-estimation problem equals the number of non-negative
exponents. Our two free axes contribute 4 such modes (double integrator per axis). This is the
strongest available statement that the *shooting-node freedom must at least span the marginal
directions* — which is automatic when the nodes are full state vectors, and is a real question when they
are produced by an encoder that may not resolve those directions.

**[R7] Trevisan, A., D'Isidoro, M., Talagrand, O.**, "Four-dimensional variational assimilation in the
unstable subspace and the optimal subspace dimension", *Quarterly J. Royal Meteorological Society*
136(647):487-496, 2010. DOI `10.1002/qj.571`. Free: `arXiv:0902.2714`. **ABSTRACT + SNIPPET ONLY.**
Confines the 4D-Var control variable to the unstable-neutral subspace; the optimal subspace dimension is
tied to the number of non-negative exponents. Relevant as the design rule; not read in full.

**[R8] Palatella, L., Carrassi, A., Trevisan, A.**, "Lyapunov vectors and assimilation in the unstable
subspace: theory and applications", *J. Physics A* 46(25):254020, 2013.
DOI `10.1088/1751-8113/46/25/254020`. **SNIPPET ONLY** — the Google Scholar snippet reads "...system has
two **null** Lyapunov exponents whose degeneracy...", i.e. the degenerate-neutral case is discussed
explicitly. **Needs full text before citing** (`needs-browser-route`, IOP).

**The structural identity nobody in our citation graph has pointed out.**
**[R9] Fisher, M., Trémolet, Y., Auvinen, H., Tan, D., Poli, P.**, "Weak-constraint and long-window
4D-Var", *ECMWF Technical Memorandum / Seminar on Data Assimilation*, 2011. Free:
`https://www.ecmwf.int/sites/default/files/elibrary/2011/9414-weak-constraint-and-long-window-4dvar.pdf`
**FULL-READ (targeted grep).** Their formulation, verbatim:

> "Weak constraint 4D-Var estimates a four-dimensional state x, defined as a collection of
> three-dimensional states `x_k` (`k = 0, ..., N-1`), each of which is valid at the start of a time
> interval `[t_k, t_{k+1})`. We refer to these intervals as '**sub-windows**'."

with the states coupled by `x_k = M_k(x_{k-1}) + q_k` and `q_k` penalised through a **model-error
covariance**. That is multiple shooting, node-for-node, with two differences that matter to us:

- The continuity penalty weight is **not a hyperparameter**. It is `Q^{-1}`, the inverse model-error
  covariance, which has a statistical meaning and a prescribed estimation procedure. Compare [R2], where
  `rho` is admitted to "strongly influence the fit" and is chosen by Bayesian optimisation. If we
  implement a defect penalty, this is the principled route to its weight, and it is a *measurable*
  quantity on our rig (the residual's own covariance), not an oracle.
- They also note the parallelism asymmetry that condensing ([R3]) fights: computing the defects from the
  node states parallelises across sub-windows; recovering the states from the defects is inherently
  sequential.

### 1.4 (c) What segment length does the literature prescribe?

**There is a principled rule, it comes from the third vocabulary, and at `|lambda| = 1` it is vacuous.
The sysid and ML literatures offer no principled rule at all.**

**What is NOT prescribed.**
- [R1] proves the opposite of a rule: with hard constraints the solution is invariant to `dm_max`. Their
  empirical observation is only that convergence *time* is non-monotonic in it (single shooting converges
  in a few iterations to a bad nearby local solution; `dm_max = 10` was their slowest; shorter was faster
  again). Segment length is a **conditioning/compute** knob, not an **accuracy** knob.
- **[R10] Iakovlev, V., Yildiz, C., Heinonen, M., Lähdesmäki, H.**, "Latent Neural ODEs with Sparse
  Bayesian Multiple Shooting", *ICLR 2023*. Free: `arXiv:2210.03466`. **FULL-READ (targeted grep).** The
  only paper found with an actual block-length ablation (their §4.2, Fig. 9). Verdict: "the optimal block
  size is much smaller than the length of the observed trajectory (51 in our case), and ... in some cases
  the model benefits from increasing the block size, but only up to some point after which the
  performance starts to drop." An interior optimum, found by sweeping. **A heuristic, not a rule.**
- **[R11] Peifer, M., Timmer, J.**, "Parameter estimation in ordinary differential equations for
  biochemical processes using the method of multiple shooting", *IET Systems Biology* 1(2):78-88, 2007
  (and their 2005 review, read here). **FULL-READ (2005 review, targeted grep).** The only stated
  constraint is a lower bound: subdivide "such that each interval contains **at least one measurement**".
- arXiv-wide, `abs:"multiple shooting" AND abs:"segment length"` returns **0** results. There is no
  literature on prescribing it by name.
- **[R12] Mattheij, R. M. M., Staarink, G. W. M.**, "On optimal shooting intervals", *Mathematics of
  Computation* 42(165):25-40, 1984. This is the one paper whose title promises the principled rule, and
  it selects intervals from the growth of the fundamental solution (a dichotomy/amplification criterion).
  **METADATA ONLY — could not be retrieved**; the AMS PDF endpoint returned a 4.5 kB HTML shell on both
  URL patterns. Marked `needs-browser-route`. **If any single item in this report is worth chasing with
  institutional access, it is this one**, because its criterion is a spectral growth bound and our growth
  factor over a segment is exactly 1.

**What IS prescribed, in data assimilation.** The field's window-length rule is
`window length <~ error-doubling time of the leading Lyapunov exponent` ([R9] and the Fisher/Trevisan
cluster; "error growth ... occurs at a rate determined by the leading Lyapunov exponent" — snippet-level
evidence from Scholar, not verified in the full text I read). **Weak-constraint / long-window 4D-Var
exists precisely to break that limit**: introducing the sub-window states as free variables with a
penalised defect is what makes windows longer than the predictability time tractable.

**Applied to us, and this is the point.** Our leading exponent is **zero**. The doubling time is
**infinite**. The classical rule does not bind at all, and the binding constraint falls back to [R1]'s
polynomial one: beta-smoothness `O(N^3)` in the *within-segment* length `N`. Two consequences:

- There is **no dynamics-derived reason** to keep `nf = 400`. At `L_h = 1` the conditioning penalty of a
  longer segment is cubic, not exponential; going from 0.1 s to 0.5 s costs a factor ~125 in the
  smoothness constant, which is a real but finite and *measurable* cost, not a wall.
- More importantly, under [R1] Theorem 2 **the segment length is the wrong knob for the drift**. The
  0.1 s blind spot documented in `drift-conclusions` §3 item 1 is a property of an objective with no
  defect term. Adding the defect couples segments into the 2 s objective at *any* segment length; growing
  the segment without the defect term does not, which is why the `nf` sweeps were refuted.

*Provisional grading:* the claim "no principled segment-length rule exists in the sysid/ML MS literature"
rests on a 0-result whole-of-arXiv abstract search, a 6-hit dblp sweep, one Scholar query, and a read of
the five method papers. OpenAlex forward-citation traversal from [R1] (53 citers) was **blocked by a 429
daily cap** and never ran. That traversal is the single most likely place a prescriptive rule would be
hiding. Treat the negative as **provisional**.

### 1.5 (d) The ML side: exposure bias and scheduled sampling

**Verdict: the repo already holds this literature, and it does not offer anything multiple shooting does
not. It offers something *different*, and the two are complementary rather than competing.**

`docs/rollout-stability-literature.md` (2026-07-11, in-repo, PRIMARY-READ status) already covers:
Brandstetter et al. pushforward (`arXiv:2202.03376`, ICLR 2022), Sanchez-Gonzalez et al. GNS random-walk
noise (`arXiv:2002.09405`, ICML 2020), Pervez & Locatello transient amplification (`arXiv:2605.08856`),
Ebers/Steele/Kutz discrepancy modelling (`arXiv:2203.05164`), and Bengio scheduled sampling
(search-verified). **This sweep does not rediscover them.** One item that document does not hold:

**[R13] Vlachas, P. R., Koumoutsakos, P.**, "Learning from Predictions: Fusing Training and
Autoregressive Inference for Long-Term Spatiotemporal Forecasts", `arXiv:2302.11101`, 2023.
**ABSTRACT-LEVEL ONLY.** One of only **2** arXiv abstracts matching `"exposure bias"` AND
`"dynamical systems"`. Directly on the train/test horizon mismatch for long-term forecasting of
spatiotemporal dynamics. Worth a read; not read here.

**The structural distinction, which is the answer to "does it offer anything MS does not".**
[R4] states it in one sentence: multiple shooting acts **on the cost function**; the exposure-bias family
acts on the **training distribution**. They fix different halves of the same gap:

| | fixes | at `|lambda| = 1` |
|---|---|---|
| Multiple shooting + defect | the objective no longer stops at the segment boundary; equivalence to the long-horizon problem ([R1] Thm 2) | polynomial conditioning cost ([R1] Thm 1, `L_h = 1`) |
| Pushforward / GNS noise / scheduled sampling | the model sees its own drifted states during training | injected noise on a free integrator is a random walk, which is the correct perturbation model (already argued in `rollout-stability-literature.md`) |

Neither subsumes the other. Multiple shooting makes the loss *measure* the long-horizon consequence; the
exposure-bias family makes the model *robust* to arriving in a drifted state. Given
`drift-conclusions` §4's conclusion that only "change what the loss measures" survives, **multiple
shooting is the family that matches the stated criterion and the exposure-bias family is not** (it
changes the data, not the measurement). That is a reason to order them, not to drop one.

---

## 2. Found but disqualified by constraint

| Item | Constraint | Why |
|---|---|---|
| Contraction/stability-based smoothing of the loss ([R1] `L_h < 1` branch: make the model contractive and all constants become `O(1)`) | **1** | Requires the poles inside the unit circle. Our `|lambda| = 1` must be preserved. Note [R1] does not *recommend* this; it is the favourable branch of their theorem, and their own stated mechanism is explicitly not to move the system out of the bad regime. |
| Assimilation in the Unstable Subspace as an *algorithm* ([R7]) — confining the control variable to the non-negative-exponent subspace | **2** (partially) | As an estimation-dimension-reduction method it restricts what the correction may represent. As a *diagnostic* for how many directions the shooting nodes must span, it passes and is used as such above. |
| Latent-space multiple shooting with a learned latent state ([R10]) | **2** | The shooting nodes live in a learned latent space, not the physical state; the defect then constrains a latent, which does not price the physical position error the deliverable scores. Their machinery (sparse Bayesian nodes, transformer encoder) is not transferable to a grey-box LFR with physically meaningful states. |
| Model-error-covariance-weighted defect ([R9]) if `Q` were taken from the baseline model | **3** | Passes only if `Q` is estimated from the measured residual (which D4 already provides: `docs/drift-conclusions` §2 C7 gives real Telica residual statistics). Taking `Q` from the truth model or the baseline matrices would be an oracle. Flagged because the temptation is real. |
| Iterative-training / trajectory-splitting heuristics ([R10] cites Yildiz 2019, Kochkov 2021, Lienen & Günnemann 2022 as "cumbersome heuristics") | — | Not disqualified, but strictly dominated: they are the ad-hoc versions of what [R1] formalises. |

**Nothing in the multiple-shooting family itself is disqualified.** It touches neither the model class
(constraint 2) nor the spectrum (constraint 1), and no threshold in it needs an oracle (constraint 3).

---

## 3. What the repo already said, and where it was wrong

Checked before searching, per the skill's local-holdings rule.

| Location | What it says | Status after this sweep |
|---|---|---|
| `docs/prioirity-list-meeting-07-07.md` item 8 | Supervisor: "Not convinced can't pull off with just positions, with multiple shooting"; multiple shooting is "the position-based alternative to try before switching the output" | **Stands, and is now supported.** [R1] Theorem 2 is the technical content of the supervisor's intuition: the position-domain objective can be made long-horizon without a long gradient path. |
| `tasks/lessons.md` `velocity-loss-is-last-resort` | Names MS as the position-based alternative to try before velocity | Stands |
| `tasks/lessons.md` `unknown-system-no-class-restriction` | "No-drift must come from the ESTIMATOR: training conditioning (multiple shooting + continuity) + regularizing only the unexcited/null direction" | **Stands and is now the best-supported line in the file.** "multiple shooting + continuity" is exactly [R1]/[R2]. |
| `tasks/lessons.md` `prove-overconstraint-dont-multiply-methods` | "multiple shooting was [tried], and failed on the augmentation: Optuna 69399 best = epoch 0" | **Wrong twice.** 69399 is lr-bug confounded (already flagged in `docs/drift-critical-analysis.md` §2.9 item 4 and §3 row 4), *and* it swept `nf` under single shooting with no continuity term, so it was not multiple shooting. Recommend amending the lesson. |
| `docs/drift-literature-sweep-2026-07-25.md` §C5 | Turan & Jäschke at METADATA level, "unchanged verdict" | **Upgraded to FULL-READ** here. The structural-fit sentence in that entry ("shooting variables are exactly the freedom to absorb it") is correct. |
| `docs/loss-function-design.md` line 18 and §refs | The **baseline parameter-recovery** pipeline already trains by "batched multiple shooting: at each epoch, 8 short segments are sampled ... rolled out with RK4 from a **cached initial state**"; cites Houska, Logist, Diehl & Van Impe (2011) tutorial | **Same gap as the augmentation pipeline.** A cached/true initial state per segment with no defect term is batched *single* shooting, not multiple shooting. The Houska et al. tutorial is a legitimate local holding for the recipe. |
| `docs/rollout-stability-literature.md` | Full exposure-bias treatment, primary-read | Complete; not rediscovered. |
| `literature/optimization/Gradient/stochastic-multiple-shooting-gpt-deep-research.md` | Despite the filename, is about **checkpoint selection** under stochastic segment sampling, not the MS method | Not relevant to this question. |

---

## 4. What this implies for the next action

Not a recommendation to build; a statement of what the literature licenses.

1. The construct that survives every filter is **a continuity/defect term between consecutive segments**,
   with the encoder supplying the node state ([R4]) so no new decision variables appear. Penalty form
   ([R2]) is the implementable one; the exact-penalty norms (`l1`/`l2`-not-squared/`l_inf`) are the
   ones with a finite-`rho` equivalence guarantee, and the quadratic default does not have it.
2. **The defect weight should come from the measured residual covariance**, not a swept `rho` ([R9]'s
   `Q^{-1}`). D4 already measured the quantities this needs.
3. **Do not sweep segment length as the intervention.** [R1] Theorem 2 says the solution is invariant to
   it under hard constraints; the repo has already refuted `nf` sweeps empirically. Segment length is a
   compute/conditioning knob and its cost at `|lambda| = 1` is `O(N^3)`, which is measurable.
4. This does not override `docs/drift-conclusions-2026-07-25.md` §6, which says to train to 400-800 steps
   and watch the constant **before** designing an objective change. Nothing here is a reason to skip that.

---

## Access status (MANDATORY)

**TU/e browser access: NOT ATTEMPTED.** Per the task instructions, no browser preflight was run in this
agent (the skill assigns one preflight to the parent in a fan-out). Items that would need it are marked
`needs-browser-route`, not `unreachable`:

- **[R12] Mattheij & Staarink 1984**, *Math. Comp.* 42(165):25-40 — AMS PDF endpoints returned a 4.5 kB
  HTML shell on two URL patterns. **Highest-value unread item in this sweep** (it is the one paper whose
  subject is literally how to choose the shooting interval, by a growth criterion).
- **[R8] Palatella, Carrassi & Trevisan 2013**, *J. Phys. A* — IOP, snippet only, contains the explicit
  null-Lyapunov-exponent discussion.
- **[R7] Trevisan, D'Isidoro & Talagrand 2010**, QJRMS — Wiley; an arXiv preprint (`0902.2714`) exists and
  was not fetched.
- **[R11] Peifer & Timmer 2007**, IET Syst. Biol. — the 2005 review was read instead; the journal version
  was not fetched.

Every other item was reached free (arXiv, ECMWF, a university mirror). No publisher URL was fetched.

## Evidence quality

| Item | Level |
|---|---|
| [R1] Ribeiro et al. | **FULL-READ**, theorems and body quotes verified by pypdf text extraction on the arXiv PDF |
| [R2] Turan & Jäschke | **FULL-READ**, 6 pp, method and discussion verified |
| [R3] Prabhu et al. | **FULL-READ** by targeted grep (abstract, method, conclusion) |
| [R4] Beintema et al. L4DC | **FULL-READ** by targeted grep; the two quoted passages verified verbatim |
| [R5] Grudzien et al. | **FULL-READ** by targeted grep on "neutral"; abstract and intro verified |
| [R6] Bocquet et al. 2017 | metadata + cited-in-[R5]; **not read** |
| [R9] Fisher et al. | **FULL-READ** by targeted grep; formulation quote verified. The Lyapunov/doubling-time window rule attributed to this cluster is **snippet-level only** and was NOT found in the text I extracted |
| [R10] Iakovlev et al. | **FULL-READ** by targeted grep; §4.2 block-size result verified verbatim |
| [R11] Peifer & Timmer 2005 review | **FULL-READ** by targeted grep |
| [R7], [R8], [R12], [R13] | **metadata / snippet only** |

Citation metadata for [R1] came from Crossref; [R7] from Crossref; the rest from arXiv/dblp/publisher
pages. Crossref 429'd partway, so page numbers for [R2] are carried from the repo's earlier sweep rather
than re-verified here.

## BibTeX for the load-bearing items

```bibtex
@article{ribeiro2020smoothness,
  author  = {Ribeiro, Ant\^{o}nio H. and Tiels, Koen and Umenberger, Jack and
             Sch\"{o}n, Thomas B. and Aguirre, Luis A.},
  title   = {On the smoothness of nonlinear system identification},
  journal = {Automatica}, volume = {121}, pages = {109158}, year = {2020},
  doi     = {10.1016/j.automatica.2020.109158}, note = {arXiv:1905.00820}
}

@article{turan2022multiple,
  author  = {Turan, Evren Mert and J\"{a}schke, Johannes},
  title   = {Multiple Shooting for Training Neural Differential Equations on Time Series},
  journal = {IEEE Control Systems Letters}, volume = {6}, pages = {1897--1902}, year = {2022},
  doi     = {10.1109/LCSYS.2021.3135835}, note = {arXiv:2109.06786}
}

@inproceedings{beintema2021encoder,
  author    = {Beintema, Gerben and T\'{o}th, Roland and Schoukens, Maarten},
  title     = {Nonlinear state-space identification using deep encoder networks},
  booktitle = {Proceedings of the 3rd Conference on Learning for Dynamics and Control (L4DC)},
  series    = {PMLR}, volume = {144}, pages = {241--250}, year = {2021},
  note      = {arXiv:2012.07697}
}

@article{grudzien2018asymptotic,
  author  = {Grudzien, Colin and Carrassi, Alberto and Bocquet, Marc},
  title   = {Asymptotic Forecast Uncertainty and the Unstable Subspace in the
             Presence of Additive Model Error},
  journal = {SIAM/ASA Journal on Uncertainty Quantification}, volume = {6}, number = {4},
  year    = {2018}, note = {arXiv:1707.08334}
}

@inproceedings{fisher2011weak,
  author    = {Fisher, Mike and Tr\'{e}molet, Yannick and Auvinen, Harri and Tan, David and Poli, Paul},
  title     = {Weak-constraint and long-window 4D-Var},
  booktitle = {ECMWF Technical Memorandum / Seminar on Data Assimilation}, year = {2011},
  note      = {www.ecmwf.int/sites/default/files/elibrary/2011/9414-weak-constraint-and-long-window-4dvar.pdf}
}

@inproceedings{iakovlev2023sparse,
  author    = {Iakovlev, Valerii and Yildiz, Cagatay and Heinonen, Markus and L\"{a}hdesm\"{a}ki, Harri},
  title     = {Latent Neural {ODE}s with Sparse Bayesian Multiple Shooting},
  booktitle = {International Conference on Learning Representations (ICLR)}, year = {2023},
  note      = {arXiv:2210.03466}
}
```

---

## Research Log

**Queries run** (37 total across 6 sources).

*Local (before any query, per skill step 0):* repo-wide grep for `shooting`/`multi-shooting` — 42 files;
targeted reads of `tasks/lessons.md` (3 rules), `docs/prioirity-list-meeting-07-07.md` item 8,
`docs/drift-critical-analysis.md` §2.9/§3, `docs/drift-literature-sweep-2026-07-25.md` §C5,
`docs/rollout-stability-literature.md`, `docs/loss-function-design.md`,
`literature/optimization/Gradient/stochastic-multiple-shooting-gpt-deep-research.md`, and a grep for
`69399`.

*arXiv API (10 queries, 3-4 s apart), with `opensearch:totalResults` recorded:*

| Query | Total | On-target |
|---|---|---|
| `abs:"multiple shooting" AND abs:"neural"` | 17 | 8/17 — the whole ML-side MS field in one query |
| `abs:"multiple shooting" AND abs:"identification"` | 8 | 5/8 |
| `abs:"multiple shooting" AND abs:"unstable"` | 5 | 1/5 |
| `abs:"unstable subspace" AND abs:"assimilation"` | 9 | 6/9 |
| `abs:"weak constraint" AND abs:"4D-Var"` | 9 | 4/9 |
| `abs:"neutral" AND abs:"unstable subspace"` | 5 | 3/5 |
| `abs:"assimilation window" AND abs:"Lyapunov"` | 1 | 1/1 |
| `abs:"exposure bias" AND abs:"dynamical systems"` | **2** | 1/2 |
| `abs:"shooting" AND abs:"marginally stable"` | **1** | 0/1 (holographic superconductors) |
| `abs:"multiple shooting" AND abs:"segment length"` | **0** | — |

*dblp (2 of a 5 budget, no block encountered):* `multiple shooting identification` (2 hits),
`multiple shooting neural` (6 hits). Both confirm the field is small; no new items.

*OpenAlex (2 calls):* `works/doi:10.1016/j.automatica.2020.109158` resolved `W2943507526`,
`cited_by_count = 53`. The follow-up `filter=cites:W2943507526` returned **HTTP 429** (shared daily spend
cap, exactly the failure the skill documents). **The forward-citation traversal never ran.**

*Crossref (3 calls, 1 succeeded then 429):* metadata for [R1] and [R7].

*Google Scholar (3 queries, mandatory cross-check):* the sentence-shaped query
"length of the assimilation window is limited by the doubling time of the leading Lyapunov exponent
4D-Var multiple minima" was the **sole source** of [R7], [R8] and [R9] — the entire third vocabulary's
window-length rule. The query "how to choose the length of the shooting intervals ... marginally stable
integrator" was the **sole source** of [R11] and [R12]. A third query on marginal stability + shooting
constraints returned 12/12 astrodynamics noise.

**What worked.**
- Two-term arXiv abstract queries. Precision was near-perfect and the totals were themselves the
  deliverable: `"shooting" AND "marginally stable"` = 1 (off-target) and
  `"multiple shooting" AND "segment length"` = 0 are the strongest negative evidence in this report.
- **Reading the local holdings before querying.** The SUBNET/multiple-shooting identity ([R4]) is the
  finding that changes the decision, and it was reachable only because the frame knew our pipeline is
  SUBNET. A pure topical search would have returned Beintema as "a neural sysid paper", not as "us".
- **Grepping the assumption vocabulary** in [R1] (`contractive`, `we assume`, `Theorem 2`) landed on the
  `L_h = 1` row of Theorem 1 and on the equivalence sentence, which are the two load-bearing facts.
  Paging the 18-page PDF would have cost more turns and might have missed the middle table column.
- **Google Scholar written as a sentence.** Both productive Scholar queries were sentences; the one
  written as keywords ("marginally stable ... eigenvalues on the unit circle") returned pure noise.

**What failed.**
- OpenAlex forward citations, HTTP 429 on the shared daily cap. This is the sweep's largest hole: 53
  citers of [R1] were never enumerated.
- AMS Math. Comp. PDF ([R12]), two URL patterns, both 4.5 kB HTML. Same class as the publisher bot walls
  the skill already documents; **AMS should be added to the "never fetch a publisher URL" list**.
- Crossref 429'd after 3 calls in quick succession, faster than the skill's text implies.
- Scholar query 3 (keyword-shaped, on marginal stability) — 12/12 off-target, all astrodynamics
  ("marginally stable" is owned by the periodic-orbit/Floquet community).

**Dead ends.**
- `literature/optimization/Gradient/stochastic-multiple-shooting-gpt-deep-research.md` — the filename
  promises the method, the content is checkpoint selection under EMA/SWA. Cost one read.
- Massaroli et al., "Differentiable Multiple Shooting Layers" (`arXiv:2106.03885`) — surfaced twice but
  is a parallel-in-time *inference* layer, not an identification objective. Not pursued.
- Kuntz & Rawlings, `arXiv:2406.03760`, "Maximum Likelihood Identification of Linear Models with
  Integrating Disturbances for Offset-Free Control" — the only hit in this sweep on identification *with*
  unit-root modes. Not multiple shooting, MPC-side, **unread**. Plausible follow-up for the `|lambda| = 1`
  identifiability question, but out of this question's scope.

**Coverage gaps (so the negatives can be graded).**
1. **No forward-citation traversal at all** (OpenAlex 429). The claim "no principled segment-length rule
   in the sysid/ML MS literature" and the claim "nobody runs MS on a plant with poles exactly at 1" are
   both **provisional** on that.
2. **No venue-year enumeration.** IFAC-PapersOnLine, CDC, ECC, ACC 2023-2026 were not swept. dblp budget
   was spent on named-artefact queries instead, per the skill's rule that dblp cannot match a property.
3. **[R12] unread** — the single paper most likely to contain the principled rule this question asks for.
4. The data-assimilation window-length rule is **snippet-level**. The `doubling time` string does not
   appear in the [R9] text I extracted, so the rule as stated is attributed to the cluster, not verified
   in one document. Verify before it enters the thesis.
5. Books not covered: Bock (1981) and Bock & Plitt (1984) are the primary sources for the method and for
   condensing; both are known only through citations here.

**Suggested skill fix (two, both concrete).**

1. **Add AMS (`ams.org`) to the never-fetch publisher list** in §4, and add a row to the Failure Modes
   table: `AMS Math. Comp. PDF endpoint returns a ~4.5 kB HTML shell on HTTP 200` — indistinguishable
   from a real download by size alone unless checked. The existing rule covers Elsevier/IEEE; a
   mathematics-society press was not anticipated and cost 2 turns.
2. **Add a step-0 sub-rule: "check whether the project's own method already IS the thing being
   researched."** The decisive finding here is that our pipeline is a multiple-shooting method under
   another name, which reframed the question from "adopt X?" to "we run a degenerate X; add the missing
   term". The current local-holdings check looks for *papers* the repo holds; it does not prompt a check
   of what the repo's *code and architecture* already implement. Proposed wording: *"Before framing a
   method as new, name the project's current method in the target literature's vocabulary. If the entry
   point paper for our own architecture lists the researched method in its keywords, the question is
   about a missing component, not an adoption."* Cost of not having it: this sweep would have returned a
   correct but useless "multiple shooting looks promising, here are 6 papers".
