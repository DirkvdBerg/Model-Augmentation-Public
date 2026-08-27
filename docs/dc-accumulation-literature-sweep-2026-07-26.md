# DC-accumulation literature sweep, 2026-07-26

**Input document:** `docs/dc-accumulation-research-brief-2026-07-26.md`.
**Method:** `.claude/skills/deep-research/SKILL.md`, step 0 FRAME in the parent, then a 5-agent fan-out,
one agent per sub-question, merged here.
**Supersedes in framing (not in bibliography):** `docs/narrowband-literature-sweep-2026-07-26.md`,
`docs/flat-direction-literature-sweep-2026-07-26.md`.

**Read this caveat before any claim below is used.** `search_google_scholar` returned `{"result":[]}` to
**all five agents, 19 attempts, every one of them including a control query that cannot legitimately return
zero** (for example `PCGrad gradient surgery multi-task learning`, and `implicit bias of Adam`). Scholar is
this skill's only full-text-indexed route, and therefore the only route to an in-body scope disclaimer,
which is where "nobody has done this" is actually written. **Every novelty and gap claim in this document
rests on titles and abstracts only and is graded provisional for that reason.** One serial Scholar pass,
run later from a single IP, would upgrade or overturn several of them.

---

## 0. Frame that was used

Stated before any query ran, as the skill requires.

| Frame element | Content |
|---|---|
| Sub-questions | 5, listed in section 2 |
| Seed IDs from the document | Reddi, Kale, Kumar, ICLR 2018 (AMSGrad). Named in the brief, never searched by either prior sweep |
| Entry points | ML: PMLR/NeurIPS venue enumeration, OpenAlex topic `T11206`. Control: IFAC source `S2898405271`, dblp `venue:CDC:`. Never a distilled keyword query |
| Disqualifies | Hard model-class restrictions (passivity, contraction, RENs, Lipschitz or spectral caps, bounded impulse, port-Hamiltonian with R>0); optimizer swap or `lr` tuning as the deliverable; zero-mean or window-mean priors on the residual |
| Anti-scope | The six mechanisms measured false in brief section 2; longer windows; ARTBP; multiple-shooting continuity; adjoint reweighting of a long-horizon **position** functional |
| Vocabularies | ML optimization; stochastic approximation and applied probability; control and system identification; econometrics and statistics; numerical analysis (backward error); statistical physics |

### The frame fact that reshaped this sweep

The previous sweep's headline rested on the offset direction being **flat**, and treated Compagnoni et al.
(ICLR 2025, `arXiv:2411.15958`) Lemma C.52 as out of scope because every lemma there assumes `H > 0`.

The brief reports curvature measured at `d2L/db2 = 7.06e+04` on two independent rigs, agreeing to 0.3%.
**The direction is not flat, so that theory is now in scope, and in scope it predicts `E[X_inf] = 0`.**
The anomaly is therefore no longer "no theory covers this case". It is **"the theory that covers this case
is contradicted by measurement"**, which is a far more specific thing to search against. Everything in
section 3.2 follows from taking that seriously.

### Local holdings checked before querying

Per skill rule, the check covered `literature/`, `docs/`, and `scripts/**/research/`, not just the first
two. Already held and therefore excluded from every agent's novelty budget: Balles and Hennig
`1705.07774`; Zhuang et al. `2202.00089`; Xie and Li `2404.04454`; Cattaneo, Klusowski and Shigida
ICML 2024; Kunstner et al. ICLR 2023; Zhang, Zou and Cao NeurIPS 2024; Kosson et al. `2305.17212`;
Liang et al. `2411.16085`; Chen et al. `2302.06675`; Compagnoni et al. `2411.15958`; Malladi et al.
`2205.10287`; Ziyin et al. `2308.06671`; Kunin et al. `2107.09133`; ProxGen (Yun, Lozano, Yang,
NeurIPS 2021); Sophia `2305.14342`; Clarke and Hernandez-Lobato `2310.14963`; Golub and Pereyra
(Inverse Problems 19 (2003) R1); Gan et al. `2511.01234`; the Thread D econometrics set (Phillips 1987,
Stock 1991, Mikusheva 2007/2012, Kendall, Marriott and Pope, Andrews 1993); Lambert et al. objective
mismatch.

**Two open verification debts the repo handed this sweep**, both worth more than a new query:
`arXiv:2006.06650` (Thread AB item A8, SEARCH-LEVEL, flagged "highest-value follow-up", never read) and
`arXiv:2202.00089` Section 3 (Thread AB negative 6, PARTIAL-FETCH). The first was paid, see 3.2.1. The
second is **still open**; no agent was pointed at it.

---

## 1. Executive answer

**Primary question (why does Adam accumulate an offset its own objective penalises).** No published result
matches the full signature. Three mechanism families each match a proper subset, and each now carries a
discriminating measurement (section 3.3). One whole family, stationary-distribution bias, is **excluded**
by two independent theorems plus the brief's own dose-response (section 3.2.2).

**Secondary question (does the marginal spectrum participate or only amplify).** **It participates.** There
is a theorem, in panel econometrics, in exactly this structure, stating that at a local unit root the
estimator distortion **loses its dependence on the sample length entirely** and instead grows with the
number of independent short samples (section 3.1). This is the strongest single finding of the sweep, and
it **retrodicts a measurement already sitting in the brief's anti-scope**: the `nf` 800 to 3200 sweep that
found the offset present at every horizon.

**Third question (is there a construction that makes an optimiser take an available descent direction).**
**Yes, and it is mature.** It is called **level-set teleportation**, the project holds none of it, and the
substitution needed to point it at this problem is one line (section 3.4).

**Genuine gap, defensible as a contribution.** The Ito versus Stratonovich spurious-drift argument for
Adam's `m/sqrt(v)` update has never been written down, and the one paper that could have closed it
removes the multiplicative noise by modelling fiat and then observes the correction vanishes (section 3.6).

---

## 2. Sub-questions as fanned out

| # | Sub-question | Agent outcome |
|---|---|---|
| SQ1 | Adam stationary or persistent bias under **positive** curvature; the Reddi construction; constant-step stochastic approximation bias; the `2006.06650` debt | Debt paid (refuted), family excluded, Reddi converted into a measurement |
| SQ2 | `beta2` memory versus a 5200-step run; bias-correction transient; `epsilon` as a systematic bias source | Three persistence results, one of them three months old |
| SQ3 | Persistent (not asymptotic) implicit bias, plus cross-field translation | The Ito/Stratonovich gap, established with a positive control |
| SQ4 | Does `z = 1` participate in accumulation or only amplify | Answered: participates, via panel econometrics |
| SQ5 | Constructions that force an available descent direction; novelty check | Teleportation; two standing novelty claims re-graded |

---

## 3. Findings

### 3.1 HEADLINE. The marginal pole participates, and the theorem is in panel econometrics

**Liao, Chengwang; Mei, Ziwei; Shi, Zhentao.** "Nickell Meets Stambaugh: A Tale of Two Biases in Panel
Predictive Regressions". `arXiv:2410.09825v2`, econ.EM, v2 dated 24 May 2026, 150 pp including online
appendices. Peer-review acknowledgements present in the PDF, journal not yet stamped. Free:
`https://arxiv.org/pdf/2410.09825`.
**Verification: TARGETED FULL READ** of Sections 1, 2.2, 2.3, 3.1, Propositions 1, 3 and 4, Corollary 2,
Remark 9, Conclusion.

Their structure is this project's structure: `n` independent short samples, each carrying its own fitted
nuisance constant, one shared parameter estimated across all of them. `gamma` indexes persistence, with
`gamma = 0` stationary and `gamma = 1` a local unit root.

Verbatim, p3:

> "In panel data, the Stambaugh bias will be carried over and fused with the Nickell bias in WG, resulting
> in a **composite Nickell-Stambaugh bias** in the t-statistic with an order substantially enlarged from
> `1/sqrt(T^{1-gamma})` to `sqrt(n/T^{1-gamma})`."

Reading the two ends of that expression:

| Regime | Distortion order | Behaviour |
|---|---|---|
| `gamma = 0`, stable plant | `sqrt(n/T)` | decays with longer windows |
| `gamma = 1`, marginal pole | `sqrt(n)` | **no `T` dependence at all**, grows with the number of independent short windows |

**Why this is the finding of the sweep.** The brief's anti-scope records that `nf` was swept from 800 to
3200 and the offset was present at every horizon, so "longer windows" was retired as a fix. Every
optimiser-side mechanism in section 3.3 predicts *some* dependence on `T`. **Liao et al. predict exactly
none, and only at `gamma = 1`.** No other candidate found in three sweeps explains why that measurement
came back flat. A mechanism that retrodicts an existing measurement it was not fitted to is worth more
than a mechanism that merely fails to contradict one.

Three further load-bearing items from the same paper:

- **Proposition 3** gives the compensation structure explicitly:
  `sqrt(n T^{1+gamma}) [ beta_WG - beta* + omega*_ev . b_WG(rho*) ] -> N(0, Sigma)` with
  `b_WG(rho*) = O_p(1/T)`. The other coefficient absorbs the autoregressive-root bias, scaled by
  `omega*_ev`, the correlation between the two innovation processes. This is "the fitted constant grows to
  compensate a downward-biased root" as a theorem rather than an analogy.
- **Corollary 2**: `rho_WG - rho* = O_p(1/sqrt(n T^{1+gamma}) + 1/T)`, and the `1/T` term "arises from the
  Nickell-Stambaugh bias in panel AR". **It does not shrink with `n`.**
- **Remark 9 kills the obvious fix.** Verbatim: bias correction by plugging in a consistent estimator
  `rho_hat` "is infeasible under `gamma = 1`, because correcting this excessively large bias demands an
  impossibly fast rate of convergence of `rho_hat`". Their own remedy is an instrument (IVX/DIVX), that is,
  an external construction, not a correction term.

#### Corroboration that the offset and the root interact inside the estimator

**Moon, Hyungsik R.; Phillips, Peter C. B.** "Estimation of Autoregressive Roots near Unity using Panel
Data". *Econometric Theory* 16(6):927-997, December 2000. DOI `10.1017/S026646660016606X`. Free copy read:
Cowles Foundation Discussion Paper 1224, `https://cowles.yale.edu/sites/default/files/2022-08/d1224.pdf`,
64 pp, January 1999 pre-publication version.
**Verification: PRIMARY READ** of abstract, Sections 1 to 3, and the Figure 1 and 2 discussion.
**Citation caveat: any quote must be attributed to CFDP 1224, not to the *Econometric Theory* pagination,
because the published version was not compared.**

Their model (1) is `z_it = mu_i + beta_i' g_t + y_it`, `y_it = a y_{i,t-1} + eps_it`, `a = exp(c/T)`. Each
unit carries its own nuisance intercept or trend; one shared near-unit-root parameter `c`.

- Pooling **helps** when there is only the near-integrated stochastic trend: "a simple pooled
  least-squares estimator does produce a consistent estimator for the local to unity parameter".
- Fitting the per-unit deterministic part **destroys** that: "the simple data-pooling heuristic does not
  hold in situations where there are **both** deterministic and near-integrated stochastic trends in the
  model. In such cases, it is shown that the pooled least-squares estimator of the localizing coefficient
  `c` generates an inconsistency that depends upon the true unknown localizing parameter."
- p12, eq. (16) to (18): `plim c_hat = F(c) = c + Omega_a(c)/Omega_b(c)`, and "the main reason for the
  inconsistency ... is that the detrending procedure produces a correlation between the lagged filtered
  regressor and the equation error". **The bias does not vanish as `n -> inf`.**
- **The intercept-only case (`g_t = 1`) already suffers it.** The trend case additionally loses
  one-to-oneness of `F(c)` over a region, so `c` is not even identified there.
- Constructive counterpart, their Section 4: a "distancing parameter" for **distant initial conditions**
  *is* consistently estimable from panel variation **when the distancing is common across units**. Shared
  structure across windows is what buys back the nuisance-initial-condition information.

**The reading for this project.** Offset alone is fine. Near-unit root alone is fine. Together they
interact inside the estimator and produce a non-vanishing, `c`-dependent inconsistency.

#### The stationary baseline

**Nickell, Stephen.** "Biases in Dynamic Models with Fixed Effects". *Econometrica* 49(6):1417-1426,
November 1981. DOI `10.2307/1911408`. 8,652 citations.
**Verification: METADATA ONLY**, content verified second-hand through Liao et al., which formalises
"Nickell bias" as `O_p(1/T)` and non-vanishing in `n`. No free copy located. **`needs-browser-route`.**

This is the incidental-parameters result for the stationary case: estimating `n` per-unit nuisance
constants jointly with a shared dynamic parameter biases the shared parameter by `O(1/T)`, vanishing in `T`
but **never in `n`**. So the "harmless on a stable plant" half of the Secondary question is answered
"not harmless, merely horizon-curable".

**Stambaugh, Robert F.** "Predictive regressions". *Journal of Financial Economics* 54(3):375-421,
December 1999. DOI `10.1016/S0304-405X(99)00041-0`. **METADATA ONLY**, mechanism verified through
Liao et al. Proposition 3. **`needs-browser-route`.** This is the named result for "a biased autoregressive
root forces a compensating bias in the coefficient sharing its innovations", and it is worse the more
persistent the regressor. The project holds the root-bias half (Kendall, Marriott and Pope) but not this
compensation half.

#### Honest limits on this whole subsection

1. **The mapping is structural analogy, not a theorem about our estimator.** `n` = number of training
   windows, `T` = `nf`, per-window encoder output = the incidental parameter. The analogy is strong
   (the encoder produces exactly `n` growing nuisance initial conditions) but **nobody has proved it for a
   learned encoder**, and Liao et al. treat linear panel predictive regressions.
2. Repo grep for `Nickell`, `incidental parameter`, `Neyman-Scott`, `dynamic panel`, `fixed effect`,
   `Stambaugh` returned **0 hits**. The entire route is new to this project, which also means none of it
   has been checked by anyone here.
3. Thread CD negative 4 already warned that all Thread D results are linear AR with an intercept and that
   transfer is by analogy. That warning applies here too and is not discharged.

#### The one control or sysid statement that short horizons bias

**Galioto, Nicholas; Gorodetsky, Alex A.** "Likelihood-based generalization of Markov parameter estimation
and multiple shooting objectives in system identification". *Physica D: Nonlinear Phenomena* 462:134146,
June 2024. DOI `10.1016/j.physd.2024.134146` (OpenAlex reports `closed`, no OA location). Free:
`arXiv:2212.13902`, 20 pp.
**Verification: FULL READ of Section 3.2 plus an exhaustive grep for `bias` and `unstable` over all 20 pp.**

p9, and this is the **only** occurrence of the word "bias" in the paper:

> "The differences in the performance of these objectives are primarily caused by the length of
> un-interrupted simulation, i.e., the value `j-i`. Longer simulation lengths lead to greater error
> accumulation, but **shorter simulation lengths can introduce bias when the data are noisy.**"

It is asserted, not proven, and not attributed to a pole. The same page names this project's exact
construction: multiple shooting "requires the estimation of the set of subtrajectory initial conditions
`Z_L := {x_l_i}`, which can be done by adding the initial conditions as parameters, **training an
encoder** [49], or ... simply using the data", and their reference [7] is **Beintema, Toth and Schoukens,
L4DC 2021 (SUBNET)**, benchmarked at horizon `T = 80`. Their diagnosis of why short horizons help is
objective smoothness; their remedy is a Bayesian marginal likelihood. **Grep for "unstable" and "biased"
over the whole paper returned zero: they never analyse a marginally stable or integrating plant.** Their
hard cases are chaotic (Duffing, logistic map).

So the closest thing in system identification to "short-horizon shooting is biased" is a single uncited
sentence, tied to measurement noise, in a paper that explicitly benchmarks this project's estimator class.

#### What short-horizon bias, as published, is NOT

**Wu, Yuhuai; Ren, Mengye; Liao, Renjie; Grosse, Roger B.** "Understanding Short-Horizon Bias in Stochastic
Meta-Optimization". *ICLR 2018*. `arXiv:1803.02021`, 17 pp. Free: `https://arxiv.org/pdf/1803.02021`.
**Verification: FULL READ**, every occurrence of the term plus the conclusion.

- The bias is in **hyperparameters** (learning rate, momentum), not a model parameter.
- Mechanism is **stochasticity plus ill-conditioning, both necessary**. p7: "This result illustrates that
  **stochasticity is necessary** for short-horizon bias to manifest". p11: "when the problem is either
  deterministic or spherical, the greedy learning rate schedule is **globally optimal**".
- No plant, no pole, no state-space model anywhere in the paper.

**Grading the transfer: it does not transfer as stated.** It produces a step-size bias, not a parameter
offset, so it cannot be cited for this failure mode. What does transfer is their Figure 1 picture: greedy
short-horizon optimisation makes no progress along the **low-curvature direction** of the loss, and at
`nf = 400` the DC direction is exactly that direction.

**Negative, graded STRONG:** the term has never been adopted in control or system identification. dblp
title search `short-horizon bias`: total 2 records, both the same paper. All 45 OpenAlex forward citers
enumerated: 100% meta-learning, learned-optimizer or large-batch work (Lookahead, McCandlish, Lorraine,
Metz), **zero** control, sysid or state-space citers. arXiv `ti:"short-horizon"` total 21, none in this
sense.

### 3.2 Mechanism families now EXCLUDED, with citations rather than assertion

#### 3.2.1 The `arXiv:2006.06650` claim is false. Repo debt paid.

**Alacaoglu, Ahmet; Malitsky, Yura; Cevher, Volkan.** "Convergence of adaptive algorithms for constrained
weakly convex optimization", `arXiv:2006.06650`, 19 pp. Note the published version swaps the word order
relative to the preprint title. **Crossref holds no deposit and OpenAlex holds only preprint records, so
there is no DOI; cite the arXiv ID.** Free: `https://arxiv.org/pdf/2006.06650`.
**Verification: READ IN FULL**, grep-and-context over all 19 pages plus whole-document term counts.

Thread AB item A8 recorded, at SEARCH-LEVEL, that this paper shows "the presence of projection induces a
stochastic bias (independent of iteration number) for constrained nonconvex optimization", and Thread AB
negative 7(a) named it the highest-value follow-up read in the whole campaign. **It says no such thing.**

- Whole-document term counts: `bias` **0**, `biased` **0**, `unbiased` **0**, `asymptotic` **0**,
  `neighborhood` **0**, `does not converge` **0**, `constant step` **0**.
- What it proves, abstract p1: "We analyze the adaptive first order algorithm AMSGrad, for solving a
  constrained stochastic optimization problem with a weakly convex objective. We prove the `O~(t^-1/4)`
  rate of convergence for the norm of the gradient of Moreau envelope, which is the standard stationarity
  measure for this class of problems. It **matches the known rates** that adaptive algorithms enjoy for the
  specific case of unconstrained smooth stochastic optimization."
- All 9 occurrences of `projection` are machinery: the weighted projection
  `P^v_X(x) = argmin_{y in X} ||y-x||^2_v` and repeated appeals to its **nonexpansiveness**, always used to
  make a bound smaller, never to introduce an error floor.
- p8 makes the unconstrained case a strict special case with no extra term, so nothing transfers to
  unconstrained adaptive optimisation either.

**Where the false snippet came from.** The abstract says "constant first and second order moment
parameters", meaning `beta1, beta2` held constant. The algorithm's step size is **decaying**,
`alpha_t = alpha/sqrt(t)` (Algorithm 1, p3). A search engine compressed "constant ... parameters" into
"constant stepsize" and paired it with "projection".

Applied to the repo: see section 4.

#### 3.2.2 Stationary-distribution bias cannot be the mechanism

**Dieuleveut, Aymeric; Durmus, Alain; Bach, Francis.** "Bridging the gap between constant step size
stochastic gradient descent and Markov chains". *The Annals of Statistics* 48(3), 2020.
DOI `10.1214/19-AOS1850`. Free: `https://arxiv.org/pdf/1707.06386`, 49 pp.
**Verification: READ IN FULL**, Theorem 4 and the p3, p7, p13 discussions plus Lemma 18 p31.

Main result, eq. (9) and Theorem 4 eq. (15):

```
theta_bar_gamma = INT vartheta pi_gamma(d vartheta) = theta* + gamma*Delta + r_gamma,   ||r_gamma|| <= C gamma^2
E[theta_bar_k^(gamma) - theta*] = A(theta0,gamma)/k + gamma*Delta + r_gamma
theta_bar_gamma - theta* = gamma f''(theta*)^-1 f'''(theta*) A C(theta*) + O(gamma^2)
```

Three consequences, each decisive here:

1. **The bias does not depend on step count.** The `k`-dependent part `A(theta0,gamma)/k` **decays**; the
   surviving term `gamma*Delta` is constant in `k`. This is structurally incompatible with the brief's
   measured dose-response (130 batches to 9.5x, 5200 batches to 127x). **The mechanism is excluded by the
   project's own measurement, now with a citation.**
2. **It is exactly ZERO for a quadratic.** p3: "For quadratic functions, it turns out that the
   deterministic part vanishes, that is, `theta_bar_gamma = theta*` ... However, it is not true for general
   objective functions where we can only show that `theta_bar_gamma - theta* = O(gamma)`". The p13 proof is
   one line: `INT f'(vartheta) d pi_gamma(vartheta) = 0` always, so for linear gradients
   `f'(theta_bar_gamma) = 0`. **The bias is driven by `f'''`, not by curvature.** Since the rig measured
   `d2L/db2 = 7.06e+04` on the direction, a locally quadratic `L` there predicts **zero** offset, which is
   the same verdict as Compagnoni Lemma C.52 reached by a completely independent route.
3. **No adaptive-method version exists.** Whole-of-arXiv abstract searches:
   `abs:"adaptive" AND abs:"constant stepsize" AND abs:"bias"` = **0**;
   `abs:"Adam" AND abs:"asymptotic bias"` = **0**; `abs:"Adam" AND abs:"drifts away"` = **0**.

**Two independent theories saying zero is a strong statement that the mechanism is not stationary bias.**

#### 3.2.3 The refinement that does apply, and what it can and cannot explain

**Huo, Dongyan; Chen, Yudong; Xie, Qiaomin.** "Bias and Extrapolation in Markovian Linear Stochastic
Approximation with Constant Step Sizes". *Mathematics of Operations Research*, 2026.
DOI `10.1287/moor.2024.0471`. Earlier: ACM SIGMETRICS 2023, DOI `10.1145/3578338.3593526`. Free:
`https://arxiv.org/abs/2210.00953`. **Verification: ABSTRACT-VERIFIED.**

From the abstract: "the bias vector of this limit admits an infinite series expansion with respect to the
stepsize. Consequently, the bias is proportional to the stepsize up to higher order terms. **This result
stands in contrast with LSA under i.i.d. data, for which the bias vanishes.** ... establishing that [the
bias and the mixing time of the Markovian data] are roughly proportional to each other. **While
Polyak-Ruppert tail-averaging reduces the variance of the LSA iterates, it does not affect the bias.**"

**Why this matters here.** A nonzero constant-stepsize bias appears **precisely when the data process is
correlated rather than i.i.d.**, and its size scales with the **mixing time** of that correlation. This
project's windows are contiguous segments of a small number of trajectories, so the batch process is
strongly correlated, not i.i.d. **Shuffle granularity and bank correlation time is a concrete, unmeasured
knob that is neither an optimizer swap nor an `lr` change**, so it is inside the anti-scope.
**Limit: still iteration-independent.** It explains presence, not growth.

**Allmeier, Sebastian; Gast, Nicolas.** "Computing the Bias of Constant-step Stochastic Approximation with
Markovian Noise". *NeurIPS 2024* (venue confirmed via dblp `conf/nips/AllmeierG24`).
arXiv `2405.14285`; proceedings DOI `10.52202/079017-4379`. **Verification: READ IN FULL** by one agent,
ABSTRACT-VERIFIED by another; treat the quoted items as read.

`E[theta_bar_n] ~ theta* + V*alpha + O(alpha^2)`, with `V` characterised by a **Lyapunov equation**
(their Lemma 11: `V = Dh(theta*)(H1)^-1 (S + H2 . W)` with `W` solving `H1 W + W H1^T + O = 0`). Their
noise `X_n` may depend on `theta_n`, which is the setting closest to ours. Three items in the body matter
more than the theorem:

1. **Assumption (A4)** requires the mean ODE `theta_dot = f_bar(theta)` to have a globally attracting fixed
   point with **Hurwitz** Jacobian, and of Theorem 1 they write: *"Without the latter assumption, one would
   naturally obtain a constant `C` that grows exponentially with `n`."*
2. p6: **the `O(alpha)` expansion is false for un-averaged iterates.** "One may wonder if Theorem 2 would
   be true for the (non-averaged) iterates `theta_n`. The answer is no and a counter-example is provided in
   Appendix A. This example illustrates that when the Markovian component `X_n` can be periodic, the
   `O(alpha)` term of `E[theta_n]` does not necessarily stabilize to a constant `V` but can be periodic as
   well." **This project observes un-averaged iterates.**
3. p4: if the system parameters are not differentiable, the bias is `O(sqrt(alpha))` rather than
   `O(alpha)`. Adam's `sqrt(v)` with a friction-like residual is a candidate.

**CORRECTION, applied here to a subagent's reading.** One agent connected (A4) failing to this project's
plant poles at `z = 1`. **That is a category error and must not propagate.** (A4) concerns the Jacobian of
the stochastic-approximation mean ODE in **parameter** space, roughly the loss Hessian, not the plant's
state matrix. Along the direction in question that Hessian was measured at `+7.06e+04`, so (A4) looks
**satisfied** there. The usable route into this paper is item 2, the un-averaged-iterate counterexample,
not the marginal pole. Anyone wanting the growth clause must first show the mean-ODE Jacobian is
non-Hurwitz for some independent reason.

Companion family, **ABSTRACT-VERIFIED or METADATA-ONLY**, all giving bias `= alpha*V + O(alpha^2)`, all
constant in `n`, all SGD or linear-SA, **none adaptive**: Sheshukova, Belomestny, Durmus et al.
`arXiv:2410.05106` (non-asymptotic, closest to the pre-stationary question); Zhang, Huo, Chen
`arXiv:2404.06023` (nonsmooth contractive); Zhang and Xie `arXiv:2401.13884` (Q-learning); Levin, Naumov,
Samsonov `arXiv:2508.05570` (higher-order); Huo, Chen, Xie `arXiv:2312.10894` (inference). Also
**Lauand, Caio Kalil; Meyn, Sean**, "Bias in Stochastic Approximation Cannot Be Eliminated With
Averaging", Allerton 2022, DOI `10.1109/allerton49937.2022.9929369`, pp. 1-4, **METADATA ONLY**,
**`needs-browser-route`** (closed IEEE, no preprint located).

**Richardson-Romberg extrapolation** (run at `alpha` and `2*alpha`, extrapolate) is the standard debiasing
device in this family and reduces bias to `O(alpha^2)`. **It has never been applied to an adaptive
method**; no arXiv abstract pairs it with one.

#### 3.2.4 Gradient surgery cannot be the fix

**Li, Zeman; Deng, Yuan; Zhong, Peilin; Razaviyayn, Meisam; Mirrokni, Vahab.** "PiKE: Adaptive Data Mixing
for Large-Scale Multi-Task Learning Under Low Gradient Conflicts". *NeurIPS 2025*. `arXiv:2502.06244`.
**Verification: ABSTRACT-VERIFIED.**

Large-scale pretraining "often exhibit[s] little to no gradient conflict", and they build a method that
instead **exploits positive gradient interaction**. Combined with the definition of PCGrad, this is the
citation establishing that **gradient surgery is a no-op by construction when gradients do not conflict**.
The brief's measurement (a direction reduces the offset 0.892x while the loss also falls 0.936x) says the
two objectives here provably do **not** conflict. **Do not spend a queue slot on PCGrad, CAGrad or MGDA.**

Related and worth holding: **Zhou, Shiji; Zhang, Wenpeng; Jiang, Jiyan; Zhong, Wenliang; Gu, Jinjie;
Zhu, Wenwu.** "On the Convergence of Stochastic Multi-Objective Gradient Manipulation and Beyond".
*NeurIPS 2022*. No arXiv version; free at `papers.nips.cc`. **ABSTRACT-VERIFIED.** MGDA, PCGrad and CAGrad
**fail to converge to Pareto-optimal solutions in the stochastic mini-batch setting**, because the
composite weights are computed from instantaneous stochastic gradients; exponential averaging of historical
composite weights restores convergence. That is a published instance of "a descent direction exists in
expectation and the stochastic algorithm does not take it", which is this project's shape in the
two-objective setting.

### 3.3 Surviving optimiser-side candidates, each with a discriminating test

#### 3.3.1 Reddi et al., the closest published construction

**Reddi, Sashank J.; Kale, Satyen; Kumar, Sanjiv.** "On the Convergence of Adam and Beyond". *ICLR 2018*.
No DOI (ICLR/OpenReview; OpenAlex holds only preprint `W2785523195`, 1618 citers). Free:
`https://arxiv.org/pdf/1904.09237`, 23 pp, the camera-ready ("Published as a conference paper at ICLR
2018"). **Verification: READ IN FULL, including Appendices B and C.**

**Does the iterate move to the worst point monotonically in step count? Yes, in two senses.**

1. **Deterministic online version (Theorems 1 and 2).** A *cyclic* linear sequence on `F = [-1,1]`, with
   `f_t(x) = Cx` for `t mod 3 = 1` and `-x` otherwise. p4: "The above examples of non-convergence are
   catastrophic insofar that ADAM and RMSPROP converge to a point that is **worst amongst all points in the
   set [-1,1]**. Note that **above example also holds for constant step size `alpha_t = alpha`**." The
   Theorem 2 proof (p14) establishes `x_{t+C} >= min{1, x_t + lambda/sqrt(t)}` with `lambda >= 0` and
   "**observe that `lambda` is independent of `t`**", then "If `x_t = 1`, then `x_{t+C} = 1` for all
   `t >= T'`". The iterate increases by a fixed positive increment every `C` steps until it saturates
   against the domain boundary. **In an unconstrained problem there is no boundary to saturate against.**
2. **Stochastic version (Theorem 3, p14).** `f_t(x) = Cx` with probability `p = (1+delta)/(C+1)`, `-x`
   otherwise; `F(x) = delta x`, optimum `x* = -1`. "We now show that for a large enough constant `C`,
   **`E[Delta_t] >= 0`**, which implies that the **ADAM's steps keep drifting away from the optimal
   solution `x* = -1`**." A per-step positive-mean displacement away from the optimum.

**The mechanism, in the authors' words (p4):** "The algorithm obtains the large gradient `C` once every 3
steps, and while the other 2 steps it observes the gradient `-1`, which moves the algorithm in the wrong
direction. The large gradient `C` is unable to counteract this effect since it is **scaled down by a factor
of almost `C`** for the given value of `beta2`". Lemma 1 (p15) quantifies it:
`T1 >= -1/sqrt(1-beta2)` for the rare large gradient against
`E[T3] >= (1-beta1)/sqrt(beta2(1+delta)C^2 + (1-beta2))`.

**Three properties that matter for this rig:**

- **`epsilon` does not save it.** p4: "for **any constant `eps > 0`**, there exists an online optimization
  setting where, again, ADAM has non-zero average regret asymptotically (see Theorem 6 in Section F)."
  Theorem 6's proof (p22) rebuilds the cycle with `beta2 = 2/((1+C^2)C^2)`. **This closes the brief's
  "`epsilon` in the denominator as a systematic bias source" starting point: `epsilon` is not a fix.**
- Holds for **any** `beta1, beta2` with `beta1 < sqrt(beta2)` (Theorem 2). Production `(0.9, 0.999)`
  satisfies `0.9 < 0.9995`, so it is **inside** the failure region, and p4 notes "large `beta2` is
  advisable", meaning `0.999` is the good end, not the bad end.
- **No stochastic sampling is required.** Theorem 1 is a deterministic cycle. A cyclic mini-batch order
  over a fixed window bank reproduces exactly that structure.

**The structural mismatch, stated honestly.** Every `f_t` is **linear on a bounded box**: no positive
curvature, no interior optimum, and the "worst point" is a boundary vertex. **Reddi et al. do not prove the
Primary question's statement.**

**What transfers is a discriminating measurement neither prior sweep proposed.** The offset direction
should be one where, across the window bank, the **restoring** gradient is **rare and large** (present only
in windows where the offset bites) while the **offset-increasing** gradient is **frequent and small**. This
is computable from the per-window gradient projections already produced. Two further predictions: the
effect should **strengthen** as `beta2 -> 1`, and be **independent of `epsilon`**. And `amsgrad=True` is a
one-flag mechanism probe: if the mechanism is Reddi's, it should reduce the offset; if it does not, the
mechanism is excluded. (Probe, not deliverable; an optimizer swap remains out of scope as a fix.)

#### 3.3.2 Memory implicitly modifies the loss, and the coefficient is 990

**Cattaneo, Matias D.; Shigida, Boris.** "How Memory in Optimization Algorithms Implicitly Modifies the
Loss". *NeurIPS 2025*. `https://arxiv.org/abs/2502.02132`, 31 pp. **NOT held locally**; the repo holds only
the same authors' ICML 2024 paper. **Verification: READ IN FULL on the AdamW section.**

In the `eps -> 0` limit, full-batch AdamW is preconditioned GD on a modified loss:

```
L~(theta) = [1 + lambda(beta2/(1-beta2) - beta1/(1-beta1)) h] L(theta)
            - h (beta2/(1-beta2) - beta1/(1-beta1)) [ ||grad L(theta)||_1 + lambda grad L(theta)^T theta ]
```

Their reading, p7: "**Assuming `beta2 > beta1`, we see that `(*)` is implicitly anti-penalized.** ... so the
main effect of memory is anti-penalizing the one-norm of the gradient. Thus, **if weight decay is
sufficiently small, memory anti-regularizes (large-batch) AdamW**."

**At this project's `(beta1, beta2) = (0.9, 0.999)`:** `beta2/(1-beta2) = 999`, `beta1/(1-beta1) = 9`, so
the anti-regularisation coefficient is **`990 * h`**. Adam's effective objective **rewards** a large
gradient one-norm, by a margin scaling with `lr` (matching the measured lr-proportionality of the drift)
and with `beta2/(1-beta2)` (making the brief's "`beta2` timescale versus run length" starting point
testable).

**Honest limit from the same page:** "the correction term is zero if and only if the point is stationary".
In the **full-batch** case the anti-regularisation biases the *trajectory* and which minimum is selected;
it does not relocate a fixed point. Since this run is 5200 steps and demonstrably not at `grad L = 0`, it
sits in the regime where the correction is active throughout, which is consistent with an offset that
keeps growing rather than settling. In the **mini-batch** case their p9 equation adds a term that does
**not** vanish at `grad L = 0`:

```
L~(theta) = L(theta) + [h beta/(2(1-beta)^2)] ||grad L(theta)||^2
            + [h beta/(2(1-beta)(1+beta))] E|| grad L^(pi(1))(theta) - grad L(theta) ||^2
```

**Their Section 6 (Limitations) states the mini-batch extension for AdamW specifically was "out of scope of
this article and is a work in progress".** That is the scope disclaimer confirming this exact case was open
as of NeurIPS 2025.

**Cattaneo, Matias D.; Shigida, Boris.** "The Effect of Mini-Batch Noise on the Implicit Bias of Adam".
`arXiv:2602.01642`, 2026, 46 pp, CC-BY. DOI `10.48550/arXiv.2602.01642`. dblp
`journals/corr/abs-2602-01642`, no published venue yet. **Verification: READ IN FULL by one agent,
ABSTRACT-VERIFIED by another.** This is that work in progress, delivered. The averaged correction has six
terms, `FB_j + MBN_{1..5,j}`; the mini-batch-noise terms scale as `b^-1 * B_simple` (inverse batch size
times the simple noise scale `trSigma/||g||^2`) and **compete** with the full-batch term, so the total
coefficient

```
C_total(beta1, beta2, lambda) = beta1/(1-beta1) - beta2/(1-beta2) + {C1 + C2}*lambda
```

**changes sign with batch size**: small batch and high noise regularizes; large batch and low noise means
larger `beta2` strengthens anti-regularization. Abstract: "in the case of large batch sizes, higher `beta2`
increases the magnitude of anti-regularization ... but as the batch size becomes smaller, the dependence of
(anti-)regularization on `beta2` is reversed. A similar monotonicity shift (in the opposite direction)
happens in `beta1`." **This is the paper that predicts, for this project's batch size and `beta2`, whether
Adam's memory is regularising or anti-regularising and by how much.**

Note for honesty: `docs/flat-direction-literature-sweep-2026-07-26.md` already mentioned "mini-batch-noise
implicit bias" as an unread 2025-2026 preprint, so the topic was glimpsed. The arXiv IDs, the theorem and
the modified-loss formula were not. Also note the displacement here is `O(eta)` and constant in step count,
a shifted fixed point rather than a growing offset, and the direction is a sharpness proxy rather than an
arbitrary positively-curved direction. **It does not by itself explain monotone growth over 5200 steps.**

#### 3.3.3 Bias correction is an implicit learning-rate schedule

**Laing, Sam; Orvieto, Antonio.** "Simplifying Adam: Bias Correction Debunked". `arXiv:2511.20516`,
November 2025. ELLIS Institute Tuebingen / MPI-IS. Free: `https://arxiv.org/pdf/2511.20516`.
**Verification: READ IN FULL**, pp. 1 to 3 including eq. (3). **This is three months old and is the direct
answer to the brief's bias-correction question.**

Their eq. (3): the bias-corrected step factorises as
`m_hat/sqrt(v_hat) = rho(t; beta1, beta2) * m_t/sqrt(v_t)` with
`rho(t; beta1, beta2) := sqrt(1-beta2^t)/(1-beta1^t)`, and "this term modulates the effective learning rate
`rho(t;beta1,beta2) * eta_t` over time in a manner which depends heavily on the values of `beta1, beta2`".

Verbatim, p2: "The inclusion of bias correction induces an implicit learning rate schedule by altering the
effective learning rate"; "for the `beta1 = beta2` setting (LLM-optimal), bias correction provides no
benefit and can even degrade performance unless appropriate learning rate scheduling is implemented"; "for
default parameters where performance is suboptimal, its removal is detrimental, explaining the source of
conventional wisdom."

**Critically for this rig, p3 and Fig. 1: the effect is largest at a FIXED learning rate and largest at the
torch default `(0.9, 0.999)`, which is this project's exact configuration.** `rho(t)` at that default is
the other side of the 12.4x inflation at step 84 the project already measured. They also state the
assumption behind bias correction plainly (p2): `E[g_t] ~ E[g_i]` for `i < t` "usually does not hold in
practice", which is exactly false in a windowed simulation-error loss whose surface changes step to step.

Supporting, **ABSTRACT-ONLY**: **St John**, "AdamD: Improved bias-correction in Adam", `arXiv:2110.10828`
(not peer reviewed), independently states "With the default bias-correction, Adam may actually make
**larger than requested** gradient updates early in training". **Ellis, Jackson, Lupu, Goldie, Fellows,
Whiteson, Foerster**, "Adam on Local Time" (Adam-Rel), NeurIPS 2024, `arXiv:2412.17113`, treats the
bias-correction timestep `t` as an intervention on the path by resetting it after each target change.
**Choi, Shallue, Nado, Lee, Maddison, Dahl**, `arXiv:1910.05446` (READ IN FULL, p4), are forced to the same
reading: to make Adam approximate Momentum "one needs to choose a learning rate schedule that accounts for
Adam's bias correction". Cross-vocabulary (differential privacy): **Tang, Shpilevskiy, Lecuyer**,
"DP-AdamBC", AAAI 2024, `arXiv:2312.14334`, a systematic bias in the second-moment estimate "leads to a
**different scaling for low variance parameter updates**", worth up to 3.5% final accuracy. That is the
closest published statement that a biased `v` selectively distorts exactly the low-gradient coordinates.

#### 3.3.4 The `v_hat` transient determines the final parameter state

**Liu, Liyuan; Jiang, Haoming; He, Pengcheng; Chen, Weizhu; Liu, Xiaodong; Gao, Jianfeng; Han, Jiawei.**
"On the Variance of the Adaptive Learning Rate and Beyond" (RAdam). *ICLR 2020*. `arXiv:1908.03265`, no
DOI. Free: `https://arxiv.org/pdf/1908.03265`. **Verification: READ IN FULL.**

**This project previously dismissed RAdam in one line as "only addresses early-training variance". That
dismissal came from the abstract and is wrong.** RAdam's claim is a **persistence** claim:

- p2: "without applying warmup, the gradient distribution is distorted to have a mass center in relatively
  small values within 10 updates. Such gradient distortion means that the vanilla Adam is **trapped in
  bad/suspicious local optima after the first few updates**."
- **Appendix p14, the decisive experiment.** Running RAdam but switching **only the first 4 updates** to
  vanilla Adam makes the model permanently fail to reach the good solution (training loss plateaus at 10
  instead of 3 on IWSLT'14 De-En); switching updates **5 to 8** instead is "less deleterious". **Fewer than
  ten steps inside the `v_hat` transient determine the final parameter state.** This is the strongest
  published version of the Primary question available.
- **Adam-2k**, their control (freeze `theta` and `m`, update only `v` for the first 2000 iterations, then
  start) removes the effect entirely. **This is a free `beta2`-transient ablation for this rig.**
- p4, their `Adam-eps` ablation (`eps = 1e-4` against default `1e-8`): it removes the divergence "However
  ... it produces a much worse performance comparing to Adam-2k and Adam-warmup. **We conjecture that this
  is because large `eps` induces a large bias into the adaptive learning rate**". Marked **conjecture** by
  the authors; do not cite as proven. The lesson stands: raising `eps` trades one bias for another.

**Ma, Jerry; Yarats, Denis.** "On the Adequacy of Untuned Warmup for Adaptive Optimization". *AAAI* 2021,
35(10):8828-8836. DOI `10.1609/aaai.v35i10.17069`. Free preprint `arXiv:1910.04209`.
**Verification: READ IN FULL.** **This refutes RAdam's variance mechanism while keeping the phenomenon**,
p3: at `t = 1` "it is guaranteed that `|m_t| = sqrt(v_t)` for all parameters, making all Adam parameter
updates either `-alpha` or `alpha` (assuming `eps = 0`). Thus, even though `Var[(1-beta2^t)/v_t]` is
divergent, the magnitude of the parameter updates themselves are constant. **Ironically, it is precisely
when the adaptive learning rate's variance is 'divergent' that the actual parameter update magnitudes have
zero variance.**"

**So the real early pathology is full-size `+-alpha` steps taken blind to gradient magnitude**, which is
the offset-accumulation mechanism localised to the `beta2` transient. Their prescription sizes the problem:
linear warmup over **`2/(1-beta2)`** iterations, which at `beta2 = 0.999` is **2000 steps, that is 38% of a
5200-step run**.

Third independent intervention on the `v` transient, **ABSTRACT-ONLY**: **Kalra, Barkeshli**, "Why Warmup
the Learning Rate? Underlying Mechanisms and Improvements", NeurIPS 2024, `arXiv:2406.09405`, which
attributes warmup's benefit to conditioning rather than variance and "suggest[s] an initialization for the
variance in Adam which provides benefits similar to warmup".

#### 3.3.5 `epsilon` in the intermediate regime, measured

**Wortsman, Mitchell; Liu, Peter J.; Xiao, Lechao; Everett, Katie; Alemi, Alex; Adlam, Ben; Co-Reyes,
John D.; Gur, Izzeddin; Kumar, Abhishek; Novak, Roman; Pennington, Jeffrey; Sohl-Dickstein, Jascha; Xu,
Kelvin; Lee, Jaehoon; Gilmer, Justin; Kornblith, Simon.** "Small-scale proxies for large-scale Transformer
training instabilities". *ICLR 2024*. `arXiv:2309.14322`. Free: `https://arxiv.org/pdf/2309.14322`.
**Verification: READ IN FULL, Section 3.4, pp. 9-11, Figures 11-13.**

p9: "At the largest scale and learning rate we test, grad RMS is around the default AdamW `eps`
hyperparameter. ... **If the grad RMS is on the same order as `eps`, then `Delta` will decrease in
magnitude ... and parameters will not receive learning signals as intended.**" And: "Decreasing `eps` to
1e-15 improves loss and mitigates a collapse in grad RMS. ... increasing `eps` to 1e-6 results in an
instability." Figure 13 shows update RMS collapsing exactly when grad RMS crosses `eps`.

**The bias is one-sided and persistent:** coordinates in the `|g| ~ eps` band are systematically
**under**-updated and therefore **retain whatever value the earlier large-gradient sign-descent phase left
them at**. That is a measured mechanism by which an early transient offset becomes unremovable.

Supporting: **Choi et al.** `arXiv:1910.05446` (READ IN FULL, pp. 1, 4) establish that "**ADAM with the
default `eps` is 'different' from ADAM with tuned `eps`**", treating them as different optimizers, which
makes the intermediate regime a distinct operating point rather than a smooth irrelevance.
**Lead only, unreviewed, do not cite as support**: Wang, Cao, Song, Bi, Yu, `arXiv:2607.06013`, which names
the exact gap ("a fixed positive numerical stability constant eventually changes the update geometry
again. This paper studies the rate-controlled middle case") and claims the limit point depends on the `eps`
schedule.

**Vocabulary sweep for `eps` under other names**, graded **STRONG negative**:
`abs:"Adam" AND abs:"epsilon" AND abs:"damping"` = **0** whole-of-arXiv;
`abs:"AdaGrad" AND abs:"epsilon" AND abs:"convergence"` = **0**. The "damping" vocabulary lives entirely in
the second-order and K-FAC literature, whose bridge paper (`arXiv:2310.14963`) the project already holds.

#### 3.3.6 The convergence worth acting on, and the tension worth naming

**Three independent routes point at one configuration change, `beta1 = beta2`:**

| Route | Why `beta1 = beta2` matters |
|---|---|
| 3.3.2, Cattaneo and Shigida | The anti-regularisation coefficient `beta2/(1-beta2) - beta1/(1-beta1)` vanishes **identically** |
| 3.3.3, Laing and Orvieto | Bias correction is **inert** at `beta1 = beta2` and maximal at `(0.9, 0.999)` |
| **Orvieto, Antonio; Gower, Robert.** "In Search of Adam's Secret Sauce", NeurIPS 2025, `arXiv:2505.21829`, ABSTRACT-VERIFIED, 1500 language models | `beta1 = beta2` **preserves near-optimal performance**, so the test costs nothing in fit quality, and admits a clean statistical reading (online mean/variance estimation, mean-field Gaussian VI) |

It is not an optimizer swap and not an `lr` sweep, so it stays inside the anti-scope.

**The tension, stated rather than smoothed over.** The 3.3.4 and 3.3.5 mechanisms together predict
**plant-early-then-freeze**, which **saturates**. The brief reports growth (9.5x at 130 batches, 127x at
5200). Those are **degradation ratios, not offset magnitudes**. The discriminating read is what fraction of
the **final offset magnitude** already exists at step 130, computable from checkpoints already on disk. If
the offset magnitude saturates while the metric keeps growing, the transient story survives and the brief's
"grows with optimiser steps" framing needs restating.

**A second measurement gap in the brief itself.** Section 2 establishes the offset appears in 10 of 10
checkpoints, but not that its **sign** is consistent across seeds. A random walk also produces 10 of 10
nonzero magnitudes. The prior sweep predicted the sign should be unstable at the early checkpoint and
stable at the late one, and warned that a single pooled verdict over all 10 `f07` checkpoints would be the
wrong test. **If the sign is random across the 10, the drift-versus-diffusion question reopens regardless
of what the literature says.** One script, existing data.

#### 3.3.7 One further relevant theoretical result

**Li, Wen, Lyu.** "Adam Reduces a Unique Form of Sharpness: Theoretical Insights Near the Minimizer
Manifold". *NeurIPS 2025*. `arXiv:2511.02773`. **ABSTRACT-VERIFIED.** The closest result to "Adam ends up
somewhere systematically different and keeps moving after the loss has converged": "when the training loss
is small, **Adam wanders around the manifold of minimizers and takes semi-gradients to minimize this
sharpness measure**", via an SDE. Where SGD minimises `tr(H)`, Adam provably minimises
`tr(Diag(H)^{1/2})`. Extends to RMSProp, Adam-mini, Adalayer, Shampoo. **Caveat: their setting is a
minimiser manifold, that is flat directions, whereas this direction has `d2L/db2 = 7.06e+04 > 0`. Applies
by analogy only.**

### 3.4 The constructive answer to the Third question: level-set teleportation

This is a published family that moves parameters along or inside the loss's own level set in order to
optimise a **second, non-loss quantity**. **The project holds none of it. None of it restricts the model
class.**

**Mishkin, Aaron; Bietti, Alberto; Gower, Robert M.** "Level Set Teleportation: An Optimization
Perspective". *AISTATS 2025*, PMLR 258:5059-5067. OpenReview `L9sU4lx63Y`. Free PDF, verified 2.80 MB,
41 pp: `https://raw.githubusercontent.com/mlresearch/v258/main/assets/mishkin25a/mishkin25a.pdf`.
**Verification: READ IN FULL via targeted pypdf grep** (scope disclaimers, assumptions, Theorem 2.10,
Proposition 2.6, Proposition B.2).

They solve the **sub-level-set** teleportation operator

```
w+ = argmax  0.5 ||grad f(w)||^2   s.t.   f(w) <= f(w_k)
```

with a projected-gradient algorithm alternating ascent on the teleportation objective with projections onto
the linearisation `{w : f(x_t) + <grad f(x_t), w - x_t> = f(w_k)}`. They prove a combined
sub-linear/linear rate under L-smoothness plus Hessian stability, and prove teleportation is exactly the
identity (no help, no harm) in the standard strongly convex setting. Empirically the oracle "can even
outperform approximate Newton methods"; pp. 7 and 9 flag that full iteration complexity for *approximate*
teleportation is open.

**The scope disclaimer that decides usability here**, p3, verbatim: "We show in Proposition B.2 that
teleportation is **ill-posed for non-coercive neural network problems**", which is precisely why they use
the sub-level-set form rather than the equality form.

**Why this is the direct answer.** Substituting the offset functional for the gradient norm gives

```
min |mean(y_hat(w))|   s.t.   L(w) <= L(w_k)
```

a one-line substitution into a construction that already has a solver, a convergence theory and a stated
ill-posedness caveat. **The brief's measurement (mean falls 0.892x while the loss also falls 0.936x) is
exactly a feasible point of that sub-level-set problem**, which is the strongest possible evidence the
operator is non-trivial in this setting.

**The lineage, all ABSTRACT-VERIFIED, all UPDATE-only (no model-class restriction):**

| Item | Citation | Actual finding |
|---|---|---|
| Origin | **Zhao, Bo; Dehmamy, Nima; Walters, Robin; Yu, Rose.** "Symmetry Teleportation for Accelerated Optimization", *NeurIPS 2022*, `arXiv:2205.10637` | Gradient methods "update parameters locally"; teleportation lets parameters "travel a large distance on the loss level set" using loss-invariant group actions, exactly preserving the loss. Proves a necessary condition for a rate improvement; closely related to second-order methods |
| Auxiliary objective precedent | **Zhao, Bo; Gower, Robert M.; Walters, Robin; Yu, Rose.** "Improving Convergence and Generalization Using Parameter Symmetries", *ICLR 2024*, `arXiv:2305.13404` | The teleportation objective need **not** be gradient norm; they teleport to minima with **different curvature** and improve generalization |
| Mechanical bridge to held OGD/GPM work | **Wu, Zihao; Dong, Juncheng; Aloui, Ahmed; Tarokh, Vahid.** "Teleportation With Null Space Gradient Projection for Optimization Acceleration", `arXiv:2502.11362`, 17 Feb 2025, no DOI | Projects **the gradient of the teleportation objective** onto the input null space, keeping the step inside the loss-invariant level set and cutting cost. **The projection is applied to the auxiliary gradient, not the task gradient** |
| Keeps Adam state | **Zhou, Zhipeng; Meng, Ziqiao; Wu, Pengcheng; Zhao, Peilin; Miao, Chunyan.** "Continual Optimization with Symmetry Teleportation for Multi-Task Learning" (COST), *NeurIPS 2025*, `arXiv:2503.04046` | On a conflict, "seek an alternative loss-equivalent point on the loss landscape"; LoRA adapter with "convergent, loss-invariant objectives" plus historical-trajectory reuse **so an advanced optimizer (Adam) keeps its state**. Plug-and-play. Sibling: same group, ACM MM 2025, DOI `10.1145/3746027.3755153` |

**The gap inside this family, and it is this project's opening.** Every published instance uses a
**parameter-space** teleportation objective: gradient norm, curvature or sharpness, inter-task conflict.
**Nobody uses an output functional**, that is a property of the simulated response.
`abs:"level set" AND abs:"constraint" AND abs:"auxiliary objective" AND abs:"neural network"` = **0**
whole-of-arXiv. Related zero: `abs:"projection" AND abs:"output functional" AND abs:"gradient descent"`
= **0**.

### 3.5 The elimination route, and the single question that gates it

**Newman, Elizabeth; Chung, Julianne; Chung, Matthias; Ruthotto, Lars.** "slimTrain: A Stochastic
Approximation Method for Training Separable Deep Neural Networks". *SIAM J. Sci. Comput.* 44(4), 2022.
DOI `10.1137/21m1452512`. OpenAlex reports `closed`, no OA location. Free preprint: `arXiv:2109.14002`,
26 pp. **Verification: READ IN FULL by the parent, pp. 2 and 8, from the preprint. The SIAM Version of
Record was NOT needed and was NOT fetched.**

**This is the closest published construction to both of the project's Thread B leads at once.** p8,
verbatim, on the sampled-Tikhonov (sTik) update for the linear block:

> "(3.3) `w_k(Lambda) = w_{k-1} - B_k(Lambda) g_k(w_{k-1}, Lambda)`, with
> `g_k(w_{k-1}, Lambda) = A_k^T (A_k w_{k-1} - b_k) + Lambda w_{k-1}` containing gradient information for
> the current mini-batch and `B_k(Lambda) = ((Lambda + sum_{i=1}^{k-1} Lambda_i) I + sum_{i=1}^{k} A_i^T
> A_i)^{-1}` containing **global curvature information** of the least-squares problem. Note that contrary
> to standard SA methods, (3.3) **does not require a learning rate nor a line search parameter. The
> learning rate can be interpreted as one, which is optimal for Newton's method.**"

It also states the regularization parameter "has been replaced with a new parameter estimate `Lambda` which
can be chosen **adaptively at each iteration**".

So: an exact, learning-rate-free, curvature-accumulating solve on the small linear block, **per mini-batch,
inside the same loop** as first-order training of the nonlinear complement.

**THE GATE, p2, verbatim:** "We assume that the network, `G`, is parameterized by two blocks of weights,
`W` and `theta`, and is of the form (1.1) `G(., W, theta) = W F(., theta)`, where `F`, also referred to as
a feature extractor, is a parameterized, nonlinear function. The important observation here is that the DNN
is nonlinear in `theta` and, crucially, is **linear in `W`**."

**This is the same separability condition as VarPro (Thread B item B5), and it is the single question that
decides this entire branch:**

> **Do the 8 output constants enter the simulated output affinely, or does the learned residual feed them
> back through the recurrence?**

In a linear state-space realisation over a short window they do enter affinely. The moment the residual
feeds back through the recurrence, the map from those constants to the free-run output is no longer affine,
the closed form is gone, and VarPro/slimTrain degrade to an inner nonlinear solve. **Settle this from the
code before building anything on this branch.** The PDE-VarPro literature states the boundary explicitly:
for nonlinear problems the least-squares problem "is not separable, which precludes the variable projection
strategy".

Surrounding work: **Newman, Ruthotto, Hart, van Bloemen Waanders**, "Train Like a (Var)Pro", *SIAM J. Math.
Data Sci.* 3(4):1041-1066, 2021, DOI `10.1137/20m1359511`, free `arXiv:2007.13171` (the parent of
slimTrain, already the known VarPro-in-NN anchor). **Dus, Mathias**, "Grassmannian Geometry and Global
Convergence of Variable Projection for Neural Networks", `arXiv:2601.22897`, January 2026, METADATA and
title only, the current theory frontier. VarPro in dynamics but not neural, title/abstract level:
**Aravkin et al.** `arXiv:1905.09169` (hybrid-system state estimation), **Askham and Kutz**
`arXiv:1704.02343` (optimized DMD), and `arXiv:2606.23077` (non-intrusive nonlinear ROM).

**Searched hard, all zero:** `abs:"variable projection" AND abs:"neural ordinary differential"` = **0**;
`abs:"linear parameters" AND abs:"eliminated" AND abs:"neural network" AND abs:"dynamics"` = **0**;
`abs:"variable projection" AND abs:"recurrent neural network"` = **1** (a 2018 van der Pol / LSTM paper,
off-target); NeurIPS 2024 plus 2025, 10,316 titles enumerated, **zero** VarPro or separable-elimination
papers.

Second-order-on-a-part precedent: **Petersen, Felix; Borgelt, Christian; Sutter, Tobias; Kuehne, Hilde;
Deussen, Oliver; Ermon, Stefano.** "Newton Losses: Using Curvature Information for Learning with
Differentiable Algorithms". *NeurIPS 2024*. `arXiv:2410.19055`. **ABSTRACT-VERIFIED**, verbatim: "Instead
of training the neural network with second-order techniques, we only utilize the loss function's
second-order information to replace it by a Newton Loss, while training the network with gradient descent."
The identified component is the **loss**, not a parameter subspace, but the update architecture is
identical. **This is the paper a reviewer will cite at any "nobody does simultaneous second-order on a
subspace" claim.**

Lower priority, title/abstract only: `arXiv:2406.17954`, "Why Line Search when you can Plane Search?",
exact multi-dimensional sub-search alongside a first-order method.

### 3.6 The genuine gap: Ito versus Stratonovich for `m/sqrt(v)`

**Nystrom, Kaj.** "Fokker-Planck Analysis and Invariant Laws for a Continuous-Time Stochastic Model of
Adam-Type Dynamics". arXiv preprint, 2026. DOI `10.48550/arXiv.2604.00840`. Free:
`https://arxiv.org/pdf/2604.00840`, 58 pp, CC-BY. **Verification: READ IN FULL** on the abstract,
Section 4, Remark 5.7, the closure on pp. 6-7, and the open-problems section p47-48.

He builds the coupled SDE limit of **bias-corrected** Adam under the scaling `eta = gamma*h`,
`alpha = 1-a*h`, `beta = 1-b*h`, `xi_k = sigma/sqrt(h) * zeta_k`, proves hypoellipticity, a Lyapunov
function, and existence and uniqueness of an invariant measure with exponential Harris-type convergence.

**Remark 5.7, p19, verbatim:** "Note that the Ito and Stratonovich formulations of (4.1)-(4.3) coincide,
since the diffusion coefficient is state-independent and the Ito-Stratonovich correction term vanishes."

**That reads as a refutation until you find why the diffusion is state-independent. p6:**

> "Similarly, the cross term `2 d_xi f(x_k) xi^i_k` is centered and would formally give rise to an
> **additional multiplicative noise term** in the limiting equation for `y_t`. However, this term
> introduces **state-dependent fluctuations** and does not contribute to the drift. Moreover, the resulting
> diffusion coefficient does not vanish at `y^i_t = 0`, so the limiting process would not preserve
> positivity, and `y^i_t` could become negative with positive probability. This is undesirable..."

He then adopts an "effective closure" (eq. 3.11),
`(d_xi f(x_k) + xi^i_k)^2 -> (d_xi f(x_k))^2 + sigma^2`, described p7 as "an averaging or moment-closure
procedure, in which the second-moment dynamics retain the mean-square effect of the noise while
**discarding fast and state-dependent fluctuations**."

**Verdict: the multiplicative noise Adam's second-moment recursion genuinely generates is deleted for
tractability and positivity, and only then do Ito and Stratonovich agree. The spurious-drift mechanism is
assumed away, not shown absent.** His invariant measure is proved to exist and be unique; **its mean is
never computed and never claimed to sit at the minimiser.** His diffusion enters only the momentum
coordinate (constant coefficient `a^2 sigma^2`); the `x` marginal is driftful and diffusion-free.

**Two further statements from the same paper bound how far Compagnoni Lemma C.52 can be pushed:**

- p48: "In contrast to the Langevin models, **`pi_inf` does not admit a closed-form Gibbs
  representation.** Instead, it reflects anisotropic couplings and nontrivial correlations between
  `(x,z,y)` induced by adaptivity and bias correction. To approximate `pi_inf`, it may therefore be natural
  to develop a conditional-Gaussian (Hermite-Galerkin) ansatz in the fast variable `z`..." **Characterising
  Adam's stationary mean is stated as an open problem in 2026.** So `E[X_inf] = 0` is a quadratic-model
  result; nothing published contradicts a nonzero stationary mean under positive curvature, and nothing
  establishes one either.
- The noise matrix is `A(x) = Diag(grad f(x)) H_f(x)`, and "hypoellipticity may fail at the critical points
  of `f`" because `{x : d_{x_j} f(x) = 0 for some j}` lies in the degeneracy set. **Adam's diffusion
  degenerates coordinate-wise wherever a coordinate gradient vanishes**, which is exactly the situation on
  a direction sitting at a near-zero optimum. A named, published structural pathology located exactly where
  this offset lives.

**Coverage behind the gap claim.** 16 whole-of-arXiv abstract counts, four vocabularies (ML, statistical
physics / stochastic thermodynamics, stochastic approximation, numerical analysis):

| Query | `totalResults` |
|---|---|
| `abs:"Adam" AND abs:"spurious drift"` | **0** |
| `abs:"Stratonovich" AND abs:"Adam"` | **0** |
| `abs:"state-dependent noise" AND abs:"adaptive optimizer"` | **0** |
| `abs:"noise-induced drift" AND abs:"stochastic gradient"` | **0** |
| `abs:"noise-induced drift" AND abs:"neural network"` | **0** |
| `abs:"ratchet" AND abs:"stochastic gradient descent"` | **0** |
| `abs:"Brownian ratchet" AND abs:"learning"` | **0** |
| `abs:"rectification" AND abs:"gradient noise"` | **0** |
| `abs:"nonequilibrium steady state" AND abs:"stochastic gradient descent"` | **0** |
| `abs:"modified equation" AND abs:"Adam"` | **0** |
| `abs:"modified loss" AND abs:"Adam"` | **0** |
| `abs:"nonzero mean" AND abs:"stationary distribution" AND abs:"SGD"` | **0** |
| `abs:"shadow" AND abs:"Hamiltonian" AND abs:"stochastic gradient descent"` | **0** |
| `abs:"bias" AND abs:"grows" AND abs:"number of iterations" AND abs:"Adam"` | **0** |
| `abs:"Stratonovich" AND abs:"stochastic gradient descent"` | 1, off-target (Ising MCMC) |
| **`abs:"Ito" AND abs:"spurious drift"` (POSITIVE CONTROL)** | **2, both pure physics** (curved-space path integrals; Brownian motion near a soft surface) |

**The positive control is what makes these zeros evidence rather than silence.** The spurious-drift concept
is alive and being actively developed in physics in 2025-2026 with **zero contact** with optimiser theory.
Confirmed by direct token count: `2602.01642` (Cattaneo and Shigida) and `2509.21614` (Callisti, Romito,
Triggiano, "Effective continuous equations for adaptive SGD: a stochastic analysis view", 42 pp, Pisa/SNS,
the diffusion-approximation route to adaptive methods) each contain **zero** occurrences of "Stratonovich",
"spurious", "multiplicative noise", "state-dependent", "noise-induced", "ratchet", "probability current" or
"nonequilibrium".

**Backward-error / modified-equation anchors worth holding** (METADATA-VERIFIED): **Barrett and Dherin**,
"Implicit Gradient Regularization", *ICLR 2021*, `arXiv:2009.11162`, the origin of backward-error analysis
in ML (GD implicitly descends `L + (eta/4)||grad L||^2`), and the template Cattaneo and Shigida generalise;
**Dherin**, `arXiv:2311.00235`.

### 3.7 Explicitly disqualified, reported so they are not re-found

Every 2024-2026 "implicit bias of Adam" paper outside the Cattaneo line is **asymptotic and directional**,
which the frame excluded: Zhang, Zou and Cao `2406.10650`; Fan, Schmidt and Thrampoulidis `2502.04664`
(SignGD/Adam, multiclass separable); Gronich and Vardi `2602.16340` (Adam and Muon, homogeneous nets);
Baek, Song and Yun `2510.26303` (per-sample Adam, separable); Tsilivis, Gronich and Kempe `2410.22069`;
Vasudeva, Lee, Sharan and Soltanolkotabi, *NeurIPS 2025*, `2505.24022`.

Disqualified as **deliverables** but retained above for mechanism only: AMSGrad, ADOPT (Taniguchi et al.,
NeurIPS 2024, `arXiv:2411.02853`, framing: vanilla Adam converges only if `beta2` is chosen
problem-dependently), AdamD, and any `beta1`/`beta2` change presented as a fix rather than a probe.

Dead ends worth recording so they are not re-run: `abs:"integrator" AND "neural network" AND "long-term
prediction"` (25 hits, all structure-preserving/Hamiltonian architectures, that is the disqualified
model-class-restriction category); `abs:"offset-free" AND "learning" AND "disturbance"` (8 hits, all MPC
*control* constructions adding an integrating disturbance state at deployment, none analysing whether the
*identification* acquired the offset; the project already holds the closest, `arXiv:2406.03760`);
`abs:"rollout" AND "compounding" AND "dynamics model"` (16 hits, all model-based RL compounding error,
variance and distribution shift, none about bias of a parameter); `abs:"bias observability"` (6 hits,
0 on-target: the inertial-navigation term does not cross into the neural-network corpus).

---

## 4. Repo corrections applied on 2026-07-26

All five were applied as part of this sweep. Recorded here so the change is auditable.

| # | File | Change |
|---|---|---|
| 1 | `scripts/gantry/drift-fix-trials/research/thread-AB-optimizer-mechanics.md`, item **A8** | Marked **REFUTED (READ IN FULL)** with the term counts, the paper's actual result, and the diagnosis of the bad snippet. R4 no longer flagged "at risk" from this source. QUEUE IMPACT rewritten: the residual-DC instrumentation is kept but rejustified on step4's **measured** rank-1 dodge, and annotated with the T1b result that settled it (`c_perp` prox mean 4.18e-09 vs control 4.36e-09, 3/3 seeds). No longer a follow-up read |
| 2 | same file, item **A7** | `arXiv:2603.16573` **located and citation narrowed**. It is Chen, Jian and Yang, Xinmin, "Preconditioned Proximal Gradient Methods with Conjugate Momentum: A Subspace Perspective", 17 March 2026: a **two-dimensional** subspace proximal-Newton framework whose "Hessian-induced orthogonalization" reduces the coupled nonsmooth subproblem to **two decoupled one-dimensional problems**, with global convergence and Q-linear rate under strong convexity. **Deterministic convex composite optimisation: no Adam, no stochasticity, no rank-r penalty.** It does **not** support "diagonal-plus-rank-r Adam prox". The Woodbury note is our own derivation and does not depend on it |
| 3 | same file, **negative 1** (non-separable prox under Adam) | **UPHELD**, with the added coverage in section 8 below, and the single reason it stays provisional |
| 4 | same file, **negative 2** (simultaneous subspace Newton with Adam) | **DOWNGRADED**, with slimTrain and Newton Losses written up as prior art implementing the same update architecture on a different split, plus slimTrain's linearity gate. Marked "do not put in the thesis in this form"; what remains novel is which subspace and the setting, not the mechanism |
| 5 | `literature/stability-training/claude-deep-research-Adam-optimizer-drift.md`, recommendation 4 | The AdaBelief half kept; the **RAdam half corrected** with the p2 quote, the appendix p14 experiment, Adam-2k, and Ma and Yarats refuting the variance mechanism while keeping the phenomenon |

Three downstream files that cited the refuted claim as a live risk were corrected **without touching the
science**, because the rank-1 dodge is this project's own measurement and is real:
`scripts/gantry/drift-fix-trials/t1_penalty_bite.py` (docstring; file re-parsed clean),
`scripts/gantry/drift-fix-trials/results/T1-penalty-bite.md`, and
`scripts/gantry/drift-fix-trials/results/T1b-prox-84-steps.md`.

**Still open, not addressed by this sweep:** Thread AB **negative 6**, the `arXiv:2202.00089` Section 3
algebra (AdamW-as-prox, PARTIAL-FETCH, "re-derive before citing"), and Thread AB **negative 7(b)**, the
inverse-Hessian noise-amplification attribution (`arXiv:2208.00441`, unread). With 7(a) closed, **7(b) is
now the highest-priority unpaid read in Thread A/B**.

---

## 5. Recommended next actions, in order

**1. The distinct-windows-versus-steps experiment. This is the single recommendation.**

Hold optimiser steps fixed and vary the number of **distinct** training windows: a small bank cycled many
times, against a large bank seen once.

- Every optimiser-side mechanism in section 3.3 predicts the offset tracks **step count**.
- The incidental-parameters mechanism in section 3.1 predicts it tracks the **number of distinct nuisance
  initial conditions** `n`, specifically as `sqrt(n)` with **no `T` dependence** at `gamma = 1`.

These two are **confounded in every run this project has**, which is why five sweeps have not separated
them. Separating them says which **family** the answer is in, and everything else is downstream of that.
Note the design must control for the correlation structure too, because section 3.2.3 predicts bias
scaling with the **mixing time** of the batch process, which cycling a small bank also changes; report
shuffle granularity alongside `n`.

**2. Two free reads off existing checkpoints, before any new training.**

- What fraction of the **final offset magnitude** already exists at step 130 (section 3.3.6). Distinguishes
  plant-early-then-freeze from genuine growth, and may require restating the brief's section 3.
- Whether the offset **sign** is consistent across the 10 checkpoints, tested per-checkpoint rather than
  pooled (section 3.3.6).

**3. If the answer lands on the optimiser side: `beta1 = beta2`.** One flag, kills the section 3.3.2 and
3.3.3 mechanisms simultaneously, costs nothing in fit quality per Orvieto and Gower.

**4. Settle the affinity question in the code** (section 3.5). One reading of the model decides whether the
entire elimination branch (slimTrain, VarPro, exact block solve) is available or degrades to an inner
nonlinear solve.

**5. Build the teleportation operator with the offset functional** (section 3.4), using the sub-level-set
form and respecting Proposition B.2. This is the only candidate found in three sweeps that attacks the
"reachable but not taken" fact directly, restricts the update rather than the model class, and is a
publishable extension (nobody has used an output functional as the teleportation objective).

**6. Pay the two remaining Thread A/B debts** (`2202.00089` Section 3, `2208.00441`) and run one serial
Google Scholar pass to upgrade the provisional negatives.

---

## 6. Access status (MANDATORY)

**TU/e browser access: AVAILABLE.** Verified end to end by the parent before the agents reported. Layer 1
(extension bridge) up: `list_connected_browsers` returned one device. Layer 2 (institutional entitlement)
up: navigating to `https://doi.org/10.1109/IROS60139.2025.11247377` (IEEE Xplore, closed) returned all five
sections plus LaTeX-rendered equations, with **no** "Sign in to Continue Reading" string.

**No finding in this document required it.** Every paper read was reached free through arXiv, PMLR,
`papers.nips.cc`, or the Cowles Foundation. Items marked `needs-browser-route` and **not** chased, in
priority order:

1. **Nickell 1981**, *Econometrica*, DOI `10.2307/1911408`. Currently second-hand through Liao et al.
2. **Stambaugh 1999**, *JFE*, DOI `10.1016/S0304-405X(99)00041-0`. Currently second-hand.
3. **Moon, Perron and Phillips 2007**, *J. Econometrics* 141(1):416-459, DOI
   `10.1016/j.jeconom.2006.10.003`. Metadata only.
4. **Lauand and Meyn**, Allerton 2022, DOI `10.1109/allerton49937.2022.9929369`. Metadata only, low
   priority (corroborated by Huo et al.).
5. **Moon and Phillips 2000**, published *Econometric Theory* version, **only if journal page numbers are
   needed for a quote**. The CFDP 1224 read is sufficient for the substance.
6. **Zhou et al. NeurIPS 2022** (multi-objective), only if the theorem statement is wanted verbatim;
   OpenReview served a browser-verification wall.
7. **slimTrain**, SIAM SISC. **NOT needed**: the load-bearing Section 3 detail was read from the preprint
   by the parent and is quoted in section 3.5.

---

## 7. Evidence quality

**READ IN FULL** (targeted pypdf grep with context, or full-text fetch; quoted strings are verbatim
extractions): Alacaoglu, Malitsky and Cevher `2006.06650`; Reddi, Kale and Kumar `1904.09237` incl.
Appendices B and C; Dieuleveut, Durmus and Bach `1707.06386`; Cattaneo and Shigida `2502.02132`;
Cattaneo and Shigida `2602.01642`; Nystrom `2604.00840`; Allmeier and Gast `2405.14285`; Liu et al. RAdam
`1908.03265`; Ma and Yarats `1910.04209`; Laing and Orvieto `2511.20516`; Wortsman et al. `2309.14322`
Section 3.4; Choi et al. `1910.05446`; Mishkin, Bietti and Gower (PMLR v258, 41 pp); Liao, Mei and Shi
`2410.09825`; Moon and Phillips (CFDP 1224); Galioto and Gorodetsky `2212.13902`; Wu, Ren, Liao and Grosse
`1803.02021`; slimTrain `2109.14002` pp. 2 and 8 (parent).

**ABSTRACT-VERIFIED**: Huo, Chen and Xie `2210.00953`; Zhao et al. `2205.10637` and `2305.13404`; Wu et al.
`2502.11362`; Zhou et al. `2503.04046`; Zhou et al. NeurIPS 2022; PiKE `2502.06244`; Newton Losses
`2410.19055`; Chen and Yang `2603.16573`; Li, Wen and Lyu `2511.02773`; Orvieto and Gower `2505.21829`;
ADOPT `2411.02853`; Adam-Rel `2412.17113`; AdamD `2110.10828`; Kalra and Barkeshli `2406.09405`; DP-AdamBC
`2312.14334`; Wang and Aitchison `2405.13698`; Shulgin et al. `2603.15958`.

**SCANNED / TOKEN-COUNTED, not read in argument**: Callisti et al. `2509.21614`; Liu, Ziyin and Ueda
`2012.03636`.

**METADATA or TITLE ONLY, do not quote**: Nickell 1981; Stambaugh 1999; Moon, Perron and Phillips 2007;
Lauand and Meyn; the five companion constant-step SA papers (`2410.05106`, `2404.06023`, `2401.13884`,
`2508.05570`, `2312.10894`); Barrett and Dherin `2009.11162`; Dherin `2311.00235`; Dus `2601.22897`;
Aravkin et al. `1905.09169`; Askham and Kutz `1704.02343`; `2606.23077`; `2406.17954`; Metz et al.
`2111.05803`; Klosin `2410.16112`; Lin et al. ICML 2024; `2506.04805`.

**SEARCH-SNIPPET ONLY, do not cite without fetching**: the restricted-OGD limitation (AI Open 2023,
`sciencedirect.com/science/article/pii/S2666651023000128`).

**UNREVIEWED PREPRINT, treated as a lead not as support**: Wang, Cao, Song, Bi and Yu `2607.06013`.

**Nothing in this document was recalled.** Every paper named came from a query run in the session. Where a
statement came from repo files rather than a query, it is labelled as such.

---

## 8. Novelty position, with vocabularies (skill rule 117)

**Claim tested:** "a constant output offset in a learned residual inside a marginally stable state-space
model, penalised by the training objective yet accumulated by Adam."

| Vocabulary | Query | `totalResults` |
|---|---|---|
| Machine learning | `abs:"constant offset" AND abs:"state-space model" AND abs:"neural"` | **0** |
| Machine learning | `abs:"auxiliary objective" AND abs:"does not decrease" AND abs:"Adam"` | **0** |
| Control | `abs:"bias" AND abs:"marginally stable" AND abs:"neural network"` | **0** |
| Control | `abs:"steady-state error" AND abs:"learned" AND abs:"residual" AND abs:"physics"` | **0** |
| System identification | `abs:"hybrid model" AND abs:"bias" AND abs:"simulation error" AND abs:"neural"` | **0** |
| System identification | `abs:"residual model" AND abs:"offset" AND abs:"drift"` | **0** |
| Navigation / estimation | `abs:"bias observability" AND abs:"neural network"` | 6, **all off-target** |

Four vocabularies, six whole-of-arXiv zeros, one 6-hit query with zero on-target results. **The combination
is not published in any of them.** The construction that should be applied to it (teleportation) is
published and mature but has never been pointed at an output functional or at a dynamics model.

**Graded negatives, with the coverage behind each:**

| Claim | Grade | Basis |
|---|---|---|
| The Ito/Stratonovich spurious-drift argument has not been written for Adam | **STRONG** | Four vocabularies, 16 arXiv zeros, a **positive control** proving the query shape works and the concept is live in physics, plus the confirming in-body scope disclaimer in the one paper that could have contained it |
| The `eps`-as-damping vocabulary bridge is unexploited | **STRONG** | Two whole-of-arXiv abstract searches returned 0 |
| "Short-horizon bias" has never been adopted in control or sysid | **STRONG** | dblp title search total 2, both the same paper; all 45 forward citers enumerated, zero control/sysid; `ti:"short-horizon"` = 21, none in this sense |
| No decoupled/proximal step with a non-separable penalty under Adam (thread-AB negative 1) | **HOLDS, provisional** | 3 arXiv zeros plus NeurIPS 2024+2025 enumeration (10,316 titles); nearest prior art `2603.16573` now cited correctly. Weakened only by no full-text route |
| The incidental-parameters framing has not been applied to dynamical-system ID with learned initial-condition encoders | **MODERATE-STRONG** | `abs:"incidental parameters" AND abs:"dynamical"` = 12, all panel econometrics; `abs:"incidental parameters" AND abs:"neural"` = 1 (discrete choice); repo grep = 0 hits |
| No paper analyses multiple-shooting or short-window bias for a marginally stable or integrating plant | **MODERATE** | `abs:"multiple shooting" AND abs:"bias" AND abs:"initial"` = 0; `abs:"marginally stable" AND abs:"simulation error"` = 0; Galioto and Gorodetsky has zero occurrences of "unstable". Weakened by no PMLR/L4DC/NeurIPS enumeration on this sub-question and no dblp venue sweep |
| No published result gives a nonzero stationary mean for Adam under positive curvature | **MODERATE** | 23/23 Compagnoni citers title-screened via Semantic Scholar, 3 arXiv zeros, Nystrom explicitly listing `pi_inf` as open. **Not** based on any full-text query |
| No adaptive-method analogue of constant-stepsize SA bias | **MODERATE** | Three arXiv zeros in the probability vocabulary; no dblp venue sweep (blocked); no Scholar |
| No exact Newton step on a small subspace simultaneously with Adam (thread-AB negative 2) | **DOWNGRADED** | Literally unrefuted (5 arXiv zeros, 10,316 NeurIPS titles, 211 optimiser-adjacent read) **but** slimTrain and Newton Losses implement the same architecture on a different split |
| No result where a downward-biased AR **root** forces a compensating **intercept** specifically | **WEAK-MODERATE** | The adjacent and arguably stronger Stambaugh structure was found instead (the **slope** on a persistent regressor absorbs the root bias). The pure intercept version was not located, but this literature is pre-arXiv and journal-bound so the zero carries little weight |

---

## 9. Research Log

**Queries run.** Approximately 110 arXiv raw-API queries in batched groups (about 35 of them
`totalResults` probes reported above as deliverables); about 30 OpenAlex, every parse guarded with
`assert 'error' not in d`, one 429 encountered; 8 dblp, of which 2 were blocked (one HTTP 500 with a 30 kB
non-JSON body, one empty body on HTTP 200, both correctly recorded as transients rather than zeros);
about 12 Crossref (bibliographic sweeps plus DOI metadata and BibTeX pulls); 1 Semantic Scholar graph API
call; 19 `search_google_scholar` calls, all empty; venue enumeration of NeurIPS 2024 (4,493 titles),
NeurIPS 2025 (5,823), ICML v235 (2,610), ICML v267 (3,330) and AISTATS v258; about 20 PDFs downloaded and
grepped; roughly 25 repo greps for local holdings.

**What worked.**

1. **Vocabulary translation over keyword search, decisively.** `abs:"Nickell bias"` returned 4 hits and one
   of them answers the Secondary question. Reaching it required first recognising the structure ("`n` short
   samples, per-window nuisance constants, one shared parameter") as a **dynamic panel**. No keyword from
   the problem statement could have found it.
2. **Guessing the field's own word.** Guessing that ML calls the section 3.4 construction "teleportation"
   (not "level-set move", not "loss-invariant reparameterisation") converted a dead lead into a five-paper
   lineage in one query.
3. **Venue title enumeration beat every keyword query.** Newton Losses, COST, PiKE, Adam-Rel, Kalra and
   Barkeshli, Li/Wen/Lyu and Orvieto and Gower were surfaced **only** by regex over raw title dumps.
4. **Positive controls turn a zero into evidence.** See section 3.6. One extra query upgraded the verdict
   from "we found nothing" to "the concept exists, is active in physics, and has zero contact with
   optimiser theory".
5. **Token-count a PDF before context-grepping it.** Counting "Stratonovich / spurious / multiplicative
   noise / ratchet" across three PDFs in one call proved two literatures do not touch, and identified the
   single PDF worth an expensive context grep.
6. **Grep the assumption vocabulary, not the topic.** "we assume", "(A4)", "counter-example", "beyond the
   scope", "does not admit" produced the highest-value sentences in this sweep: Nystrom p48, Allmeier and
   Gast (A4), Mishkin Proposition B.2, slimTrain's linearity gate. None is visible in a title or abstract.
7. **Read the appendix.** RAdam's decisive result (p14) is invisible in the abstract, the introduction and
   every citation of it, and is why this project's prior one-line dismissal was wrong.
8. **Whole-document term counts as a refutation instrument.** `bias: 0, biased: 0, unbiased: 0` over 19
   pages refutes a claim faster and more convincingly than any amount of reading.

**What failed.**

- **`search_google_scholar`: 19 of 19 empty, across all five agents, including trivially-hitting control
  queries.** Same silent-`[]`-masking-a-block failure the skill documents for `search_semantic`. This
  removed the only full-text route and is the single largest weakness of this sweep.
- **arXiv API 429 storms under 5-way fan-out.** 4 to 5 second sleeps were insufficient; recovery needed
  20 to 25 second spacing and, in one case, a 12-minute gap. Several batches had to be re-run.
- **dblp blocked twice** on a shared IP, in two different shapes (HTTP 500 with a large non-JSON body, and
  an empty body on HTTP 200), neither matching the shapes the skill currently documents.
- **OpenAlex forward traversal is dead at the frontier**, as the skill predicts: `cited_by_count` of 0 or 1
  for Compagnoni, Ziyin (Phys. Rev. E version), Cattaneo and Allmeier and Gast.
- **dblp `venue:ICLR:+Adam` AND-matched "Adam" in author names**, returning 60 hits with about 3 relevant.
- OpenAlex `title.search` returned **count 0** for "Train Like a (Var)Pro", a paper it holds: punctuation
  in the title defeats it. Crossref found it first try.
- Crossref `query.bibliographic` on descriptive control phrases returned mass spectrometry, ADC metrology
  and gender-bias-in-ML on 3 of 4 attempts.
- OpenAlex returned a **duplicate ghost record** for Ziyin's Phys. Rev. E paper with author "Anonymous" and
  a different DOI alongside the correct one. Relevant when deduplicating by DOI.

**Near-miss worth recording.** An agent guessed an `elischolar.library.yale.edu/cgi/viewcontent.cgi?
article=<n>` id and received **HTTP 200 with a valid PDF of an entirely different paper** ("Information
Acquisition in Committees"). Caught only by checking page 1. This is the same class as the fabricated-PMLR
path rule, in a new venue, and it fails **silently and well-formed** rather than 404ing.

**Coverage gaps.**

1. No full-text search engine was available at any point. Every negative rests on titles and abstracts.
2. ICLR, TMLR and JMLR were never enumerated; OpenReview was never queried and is the obvious hole, since
   ICLR is where a bias-correction or `eps` paper would most plausibly sit.
3. No IFAC, CDC, ECC or ACC sweep was run: the frame treated this as an ML and statistics question. If the
   control community has its own name for teleportation or for short-window shooting bias, it was missed.
4. Forward citations on all 2025-2026 items are structurally unavailable (indexing lag), so the newest
   descendants of the teleportation and Cattaneo lines are unreachable.
5. The constant-step SA companion family is metadata-only; one of those six may contain a non-Hurwitz or
   growing-bias case in an appendix.
6. Nystrom is 2026 with no forward citations, so whether anyone has objected to his closure is unknown.
7. `abs:"beta2"` and `abs:"beta_2"` both return **0** on all of arXiv: a symbol-valued hyperparameter is
   unreachable by keyword and must be reached by enumeration.

**Suggested skill fixes** (five, all measured this run):

1. **Add the Semantic Scholar graph API as the forward-citation route when OpenAlex reports 0 on a frontier
   preprint.** Measured: OpenAlex reported 0 citers for `arXiv:2411.15958` (ICLR 2025) while
   `curl -s "https://api.semanticscholar.org/graph/v1/paper/arXiv:<ID>/citations?fields=title,year,venue,externalIds&limit=100"`
   returned **23**, and three novel findings came off that list. Note the `/citations` endpoint answered on
   an anonymous IP where the sibling `/paper/<id>` detail endpoint returned 429 in the same minute, so
   treat the two as independently rate-limited: get the citer list first, then resolve titles through arXiv
   `id_list=` (comma-separated, full abstracts in one call).
2. **`search_google_scholar` shares `search_semantic`'s silent-`[]`-on-block failure.** Fire a
   trivially-hitting control query first. If it returns `[]`, declare the mandatory Scholar cross-check
   **not performed** and downgrade every novelty claim in the run to provisional. Never let `[]` stand in
   for "I ran the check". Consider having the parent run one serial Scholar pass per sub-question after the
   fan-out completes, the same way the parent owns the browser preflight.
3. **Run a positive control before reporting a zero as absence.** Pair every cross-field zero with one
   query that keeps the **foreign** term and drops the **home-field** term. Cost: one query. Benefit: the
   verdict upgrades from "we found nothing" to "the concept exists in field X and has zero contact with
   field Y".
4. **Raise the arXiv inter-call sleep to 20 s and retry backoff to 25 s when N >= 4 agents**, and add
   `search_google_scholar` and the arXiv API to the shared-IP hazard list alongside dblp and OpenAlex.
5. **Never construct a working-paper repository id.** Add to route 3b: `elischolar` and similar
   `?article=<n>` endpoints return **HTTP 200 and a valid PDF for the wrong paper** when `n` is wrong, a
   silent well-formed wrong retrieval unlike a 404. Always print page 1 and confirm the title before
   quoting. Also add to section 5: some older PDFs (Scientific Word, early-1990s Type-1 fonts) extract as
   **Caesar-shifted** text ("Hvwlpdwlrq" for "Estimation"), so a pypdf grep returns 0 for every real term
   and reads as a genuine miss. If page 1 extracts as gibberish, try an ASCII shift of -3 before concluding
   the PDF is unsearchable.
