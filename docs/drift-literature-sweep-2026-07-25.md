# Drift literature sweep, 2026-07-25

**Input**: `docs/drift-problem-statement.md` (written 2026-07-25, after the `drift-fix-trials` T0/T1 campaign).
**Method**: `deep-research` skill, step 0 FRAME then a seven-way subagent fan-out, one agent per
sub-question. Frame shown to the user and confirmed before any query was run.
**Distinct from** `docs/drift-research-report.md` (2026-07-24, analysis session #1 Phase 2), which swept the
earlier brief. This sweep answers the *re-graded* problem statement and corrects several of its claims.

## 0. Verification levels used here

| level | meaning |
|---|---|
| **FULL-READ** | body read, equations or theorems verified against the PDF |
| **PART-READ** | some sections read in full text; the rest not |
| **ABSTRACT** | abstract only. An abstract can invert a verdict, so no conclusion rests on one |
| **METADATA** | authors/venue/DOI confirmed via Crossref or OpenAlex; finding NOT verified |
| **DERIVED-HERE** | our algebra, stated by no paper. Re-derive before it enters the thesis |
| **REPO** | from a project file, not from a query in this session |

**Access status for the whole sweep: TU/e browser route AVAILABLE**, both layers verified 2026-07-25
(IEEE Xplore header "Access provided by: Eindhoven University of Technology", active PDF button).
No agent used it, by instruction, so every finding below came through arXiv, PMLR, TU/e Pure, HAL,
MDPI, IOP, CORE, author homepages or Federal Reserve working papers. The unfetched items are listed
in §6 and remain `needs-browser-route`, not `unreachable`.

---

## 1. The §2 fork resolves toward OPTIMIZATION, on three independent legs

### Leg 1: Adam's parked position is predicted in closed form, and contains no curvature term

**Bock, S., Weiss, M.**, "Non-convergence and Limit Cycles in the Adam Optimizer", *ICANN 2019*, LNCS,
pp. 232-243. DOI `10.1007/978-3-030-30484-3_20`. Free: `arXiv:2210.02070`. **FULL-READ** (pp. 3-6).

For a scalar quadratic under full-batch Adam without bias correction at `eps = 0`, they solve the 2-limit-cycle
fixed point symbolically. The cycle position is

```
|w| = alpha * (1 - beta1) / (2 * (1 + beta1))
```

with **no curvature term**, while the second moment absorbs the coefficient as `c^2` and the first as `c`.
They also state the cycle's eigenvalues do not depend on `alpha` or `c`, and that a stable cycle is attracting.

So a coefficient-independent, lr-proportional, seed-independent plateau is a *predicted consequence* of Adam's
fixed-point structure. This upgrades I5: thread A4 in
`scripts/gantry/drift-fix-trials/research/thread-AB-optimizer-mechanics.md` currently states that
"neither paper states the beta-saturation result itself, that inference is ours". That caveat is now false,
in our favour. Bock and Weiss appears nowhere in that thread.

**Quantitative check, and the open gap.** `beta1 = 0.9` is confirmed on the rig: no `betas` argument is passed
to any `torch.optim.Adam` call in `scripts/gantry/drift-fix-trials/`, so the PyTorch default holds. Prediction
`0.0263*lr = 2.63e-09` against measured `0.235*lr = 2.35e-08`. Functional form matches on all three counts;
prefactor is 8.9x off. Two candidate reconciliations, both checkable without a training run:
1. they derive without bias correction at `eps = 0`, the rig runs with both;
2. the routed constant may be a scaled version of the penalized parameter rather than the parameter itself.

The `beta <-> c` identification that lets their result speak to a penalty coefficient is **DERIVED-HERE**,
valid in the regime where the penalty gradient dominates the data gradient.

### Leg 2: Adam cannot be mid-relaxation

A third diagnosis not named in the problem statement is that the direction is stiff, the optimizer is heading
for `b*`, and 84 steps is simply not enough: for a locally quadratic direction the gradient-descent time
constant is `1/(eta*lambda) = 141` steps on the document's own `H_XX = 7.08e4`, `lr = 1e-7`.

**That arithmetic is right for gradient descent and wrong for Adam** (DERIVED-HERE). Adam's per-step
displacement is bounded by roughly `alpha = lr = 1e-7`, and the gap from the parked `3.5e-08` to `b* ~ 1e-11`
is `3.5e-08`, about **one third of a single step**. Adam could close it immediately and does not. The
not-yet-converged leg is therefore refuted by the same arithmetic that makes leg (b) plausible.

### Leg 3: the excitation is less impoverished than §5.5 assumes

See §3. The Y-scheduling already supplies the modulation that the identifiability reading assumes is absent.

### The binding caveat is ours, not the literature's

`b*` and `H_XX` are OTHER-RIG (routing 0..5); `3.5e-08` is current-rig. Our own rule forbids that comparator,
so the fork is **argued, not settled**, until the 6-D output-DC curvature is re-measured at routing (3,4,5).

**Cohen, J. M., Ghorbani, B., Krishnan, S., Agarwal, N., Medapati, S., Badura, M., Suo, D., Cardoze, D.,
Nado, Z., Dahl, G. E., Gilmer, J.**, "Adaptive Gradient Methods at the Edge of Stability", 2022/2024.
Free: `arXiv:2207.14484`. **PART-READ** (abstract, §1, Fig. 1).

Two results. First, in full-batch training the maximum eigenvalue of the *preconditioned* Hessian equilibrates
at `(2 + 2*beta1) / ((1 - beta1)*eta)`, which is `38/eta` at `beta1 = 0.9`, i.e. **3.8e8** at our `lr`. Measuring
the preconditioned curvature along the output-DC direction against that number either names the mechanism as an
adaptive-edge-of-stability equilibrium (whose parked offset must then scale proportionally to `eta`, matching the
"offset proportional to lr" observations) or refutes it. Second, and a warning for every single-checkpoint
Hessian number in this project: adaptive methods at the edge of stability keep advancing into higher raw
curvature while the preconditioner compensates, so `H` along the DC direction is **not a fixed property of the
rig** and a `b*` from one step count is not portable to another. That is an independent reason not to reuse the
OTHER-RIG curvature, separate from the routing change.

### The decision statistic exists, and we have half-computed it

**Frye, C. G., Simon, J., Wadia, N. S., Ligeralde, A., DeWeese, M. R., Bouchard, K. E.**, "Critical
Point-Finding Methods Reveal Gradient-Flat Regions of Deep Network Losses", *Neural Computation*
33(6):1469-1497, 2021. DOI `10.1162/neco_a_01388`. Free: `arXiv:2003.10397`. **FULL-READ** (incl. App. A.4).

A *gradient-flat* point is one where the gradient lies in the kernel of the Hessian, so the loss is locally
linear along the gradient "whether or not the gradient is itself small". Their index is the relative residual
of the Newton system `Hp + g = 0`:

```
r(p) = ||Hp + g|| / (||H||_F * ||p|| + ||g||)          cutoff r > 0.9 = approximately gradient-flat
```

Restricted to the 6-D output-DC subspace: `r ~ 0` with finite `||p|| = b*` refutes identifiability and says the
parked point is a stationary point of the *optimizer*, not of the loss, so the remedy space is step geometry.
`r -> 1` means `g` is effectively in `ker H`, no finite `b*` exists, and the loss genuinely does not determine
the constant. The cutoff is dimensionless and data-derived, so it satisfies the no-oracle rule. Reporting a
*measured* `b* = -g/H` is already an implicit claim that `r ~ 0`; naming `r` turns that into a test.

Companion analytic certificate, textbook: on a locally quadratic direction
`||g||/lambda_max <= ||theta - theta*|| <= ||g||/lambda_min`. Reporting that two-sided bound next to the
observed `3.5e-08` is the cleanest available statement of the fork.

**Schneider, F., Dangel, F., Hennig, P.**, "Cockpit: A Practical Debugging Tool for the Training of Deep
Neural Networks", *NeurIPS 2021*. Free: `arXiv:2102.06604`. **PART-READ** (§2 verbatim).
Their `alpha` instrument builds a noise-informed univariate quadratic along the step direction and reports
where on that parabola the optimizer landed (`0` = valley floor, `-1` = start). It is the only instrument here
designed for the noisy real-data case. Caveat, reported-not-verified: their §3.3 finds it is generally *not*
good practice to tune so `alpha ~ 0`, so `alpha != 0` alone is not evidence of pathology. Use `alpha` for the
mis-seating, and Frye's `r` for whether a minimum exists to be seated at.

### The named expressivity-versus-optimization protocol

**Krishnapriyan, A. S., Gholami, A., Zhe, S., Kirby, R. M., Mahoney, M. W.**, "Characterizing possible failure
modes in physics-informed neural networks", *NeurIPS 2021*. Free: `arXiv:2109.01050`. **FULL-READ**.
Their §5 is titled "Expressivity versus optimization difficulty" and concludes PINN failures are not a lack of
expressivity but a loss landscape that is hard to optimize. The three-part protocol transfers directly:
(1) an **achievability construction** exhibiting a parameter setting in the same architecture with much lower
error; (2) landscape slices along the **top Hessian eigenvectors** rather than random directions; (3) a
**condition-number** analysis of the operator the soft penalty introduces.

Two consequences. Their point (3) is that a soft physics penalty *raises* ill-conditioning, which is an
independent precedent for our in-loss plateau and a reason the proximal application is structurally right
rather than lucky. And we already own the strongest possible version of point (1): in the perfect-match null
the correct answer is `b = 0` at loss `~1e-11`, and at step 30 the windowed loss is *worse* than ANN-off on
2 of 3 seeds. Under their own logic that is a completed achievability construction pointing at optimization.

---

## 2. Two measured "anomalies" are documented expected behaviour, in ARTBP's own paper

**Tallec, C., Ollivier, Y.**, "Unbiasing Truncated Backpropagation Through Time", `arXiv:1705.08209`.
**REPO** (`literature/stability-training/`) and **FULL-READ** this session.

- **§4, the heavy tail.** Their compensation factors grow polynomially like `L^(alpha-1)`, and stay controlled
  only "if the dynamical system has geometrically decaying memory", because `(1-eps)^L * L^alpha` stays bounded.
  At `|lambda| = 1` there is no `(1-eps)^L` and nothing cancels the growth. **Our heavy tail on a pole-1 mode
  is the method's own variance argument running out, not a tuning failure.**
- **§6.1, the perfect-match null.** On a deterministic problem a deterministic gradient scheme converges
  geometrically, whereas ARTBP, randomised, "will not converge faster than `O(t^(-1/2))`", and the difference
  "would disappear, for instance, with noisy targets or a noisy system". **Our null is that deterministic
  problem, so ARTBP degrading it is predicted, and the effect should shrink on real Telica data.**

**DERIVED-HERE** (follows from their Eq. 14 plus §4, stated by no paper): with length density `p(L) ~ L^-alpha`
and compensation `w_L ~ L^(alpha-1)`, the second moment is `sum_L L^(alpha-2) * g_L^2`. Convergence needs `g_L`
to decay faster than `L^((1-alpha)/2)`; with no forgetting `g_L` does not decay, so the variance diverges for
every `alpha > 1`. That is why raising `alpha` bought only 2-5x: `alpha` trades tail weight against
compensation weight and both carry the same `L^(alpha-2)`.

### CORRECTION to I3: the reported variance ratio is not a defined quantity

The problem statement reports "a measured poly-tail variance gain over geometric of only 2-5x". If the variance
is infinite in this regime that ratio is undefined. **Wang, H., Gürbüzbalaban, M., Zhu, L., Şimşekli, U.,
Erdogdu, M. A.**, "Convergence Rates of SGD under Infinite Noise Variance", `arXiv:2102.10346` (**ABSTRACT**)
further show that properly scaled Polyak-Ruppert averaging converges to a multivariate alpha-stable law rather
than a Gaussian, so sample means and standard errors across ARTBP draws are not valid summaries. Restate the
2-5x as a quantile or truncated-moment comparison.

### The impossibility is proved, and declared out of scope by the field's reference treatment

**Beatson, A., Adams, R. P.**, "Efficient Optimization of Loops and Limits with Randomized Telescoping Sums",
*ICML 2019*, PMLR v97. Free: `arXiv:1905.07006`. **FULL-READ** (§3-4, §8-9).
For residual `psi_n <= c/n^p`, choosing `q(N) ~ 1/N^(p + 1/2)` gives horizon-agnostic variance and compute
**iff `p > 3/2`** (Thm 4.1); for geometric decay it is free for all rates (Thm 4.2); regret independent of `H`
follows (Thm 4.3). A **non-decaying** direction admits **no** sampling distribution with finite variance.
Secondary and directly actionable: they measure that single-sample randomized telescoping (RT-SS) beats fixed
truncation while the Russian-roulette variant (RT-RR), which is ARTBP's scheme, "suffer[s] from very poor
convergence". **We picked the worse of the two variants.**
Ancestry: Rhee, C.-H., Glynn, P. W., *Operations Research* 63(5):1026-1043, 2015, DOI
`10.1287/opre.2015.1404` (**METADATA**, needs-browser-route); McLeish, D., DOI `10.1515/mcma.2011.013`
(free `arXiv:1005.2228`).

**Massé, P.-Y., Ollivier, Y.**, "Convergence of Online Adaptive and Recurrent Optimization Algorithms",
`arXiv:2005.05645`, 126 pp. **PART-READ** (pp. 11, 24, 42-44).
Assumption 2.13 requires first-order stability at `theta*`, justified as: "If this assumption is not satisfied,
then even running the system with fixed parameter `theta*` is numerically unstable, so there is little interest
in trying to learn `theta*`." And p24: allowing infinite memory "would require allowing the spectral radius of
the sequence to tend to 1 over time, but this is beyond the scope of the present work." **This is the cleanest
available "nobody has treated this case", stated by the authors rather than inferred.**

Their §3.2 / Def. 3.13 is nonetheless the citation we need for horizon scheduling: with truncation intervals
`T_(k+1) - T_k = T_k^A` and exponents tied to the learning-rate schedule `eta_t ~ t^-b`, non-overlapping TBPTT
**converges locally** to `theta*`. It inherits Assumption 2.13, so the proof does not literally apply at
`|lambda| = 1`, but it identifies the correct knob (`A` versus `b`) and says the coupling of horizon growth to
step-size decay is the load-bearing part. Their §6.5 "Spectral Radius Close to theta*" and §6.6 "Stable Tubes"
(pp. 83-90) were **not read** and are the most likely place for a quantitative near-unit-root statement.

**Aicher, C., Foti, N. J., Fox, E. B.**, "Adaptively Truncating Backpropagation Through Time to Control
Gradient Bias", *UAI 2019*, PMLR v115. **REPO** and **FULL-READ**. Not unbiased: it bounds bias under
Assumption (A-1), geometric decay in expectation. Their §5.4 is our exact failure: without regularisation
"our estimates `beta_hat` were often close to or greater than 1 ... a few dimensions did not [decay] and these
dimensions cause the overall norm to decay slowly". Their proposed Mahalanobis-norm remedy is left as future
work. **A marginal mode among decaying modes is a stated open problem in the adaptive-truncation literature.**

---

## 3. The Y-scheduling is a modulation channel, and the training set already uses it

**Wu, Y., Zhang, H., Wu, M., Hu, X., Hu, D.**, "Observability of Strapdown INS Alignment: A Global
Perspective", *IEEE Trans. Aerospace and Electronic Systems* 48(1), 2012. DOI `10.1109/taes.2012.6129622`.
Free: `arXiv:1112.5282`. **FULL-READ**.

- **Theorem 1**: "For static alignment, the SAOP is unobservable. The number of unobservable states is
  infinite." Constant biases are not observable at standstill.
- **Remark 2**: on an unobservable problem an estimator "is supposed to converge to one of the unobservable
  states depending on the estimator settings, e.g., the selection of initial value". **That is our parked
  constant, described in this field as the expected signature of non-identifiability.**
- Alignment becomes completely observable under rotation about two axes, nearly observable about one, and
  "it is not the static positions but the rotating motion that matters for observability".
- **Remark 3**: two still positions with different headings do not give unique bias solutions; at least four
  still positions are needed, and the rotating motion *between* them is what matters. A warning against
  "just add more standstill operating points".

Supporting: **Rhee, I., Abdel-Hafez, M. F., Speyer, J. L.**, *IEEE TAES* 40(2), 2004, DOI
`10.1109/taes.2004.1310002` (**ABSTRACT**, needs-browser-route): constant acceleration without rotation
"does not change the number of observable modes but rather the structure of the observable space", while a
manoeuvre increases instantaneously observable modes by at least 2. **A constant excitation buys structure,
not rank; rank comes from time variation.** Modulation mechanisations: **Prikhodko, I. P., Zotov, S. A.,
Trusov, A. A., Shkel, A. M.**, *J. MEMS* 22(6):1257, 2013, DOI `10.1109/jmems.2013.2282936`
(free at eScholarship, **PART-READ**), carouseling versus maytagging, with azimuth uncertainty falling as
`sqrt(N)` turns. Caveat: **Du, S., Sun, W., Gao, Y.**, *Sensors* 16(12):2017, 2016, DOI `10.3390/s16122017`
(**ABSTRACT**), rotation modulation "induces additional sensor errors" needing their own calibration, so
modulation is not free.

### The transfer, and why it changes §5.5

There is no group action on a linear stage, so the literal rotary cure does not transfer. Two partial
transfers do, both satisfying constraints 1, 2 and 3:

1. **Maytagging maps onto direction reversal.** A velocity-odd residual flips sign with travel direction;
   a constant force bias does not. Matched forward/reverse segments separate them by parity, using only a
   symmetry of the KNOWN baseline. Unlike a zero-mean penalty this does not suppress a genuine friction impulse.
2. **Carouseling maps onto Y traverse, through the LPV scheduling.** `M(Y)` is a known, motion-dependent gain
   from force to acceleration, so traversing Y modulates a constant ANN force into a Y-varying acceleration
   signature, with `M(Y)` playing the rotation matrix's role.

**And the training set already does this.** `rig.py:49-52`: four standstill records at Y = -30, -15, 0, +15;
then `T6_ysweep_slow`, `T7_ysweep_fast`, `T8_ysweep_xmix`; then four APRBS records including `T12_aprbs_yaw`;
held-out at Y = +30. The ANN constant is one shared parameter across all of them, so modulation depth is not
zero.

**Consequence.** Zero input power at 0 Hz does **not** imply the constant is unidentifiable on an LPV system,
because the scheduling supplies time variation independently of the input spectrum. §5.5's "practically
non-identifiable from this data" is weaker than the document assumes, and this is leg 3 of §1.

**The decisive experiment is now cheap and needs no new hardware**: fit the constant on the three ysweep
records alone, then on the four standstill records alone, and compare where it parks. If Y-modulation confers
identifiability, the ysweep-only fit should pin it markedly closer to zero and the standstill-only fit
should not.

### Two hard disqualifications on the input-design remedy (both DERIVED-HERE)

- **Literal DC force is physically inadmissible.** A force sinusoid of amplitude `a` at frequency `f` on a free
  mass gives excursion `a / (m*(2*pi*f)^2)`, so under stroke bound `S` the admissible amplitude is
  `a_max = S*m*(2*pi*f)^2` and admissible force *power* falls as `f^4`, reaching exactly zero at DC for any
  finite stroke. **Disqualified by constraint 4.**
- **Reference design cannot deliver DC force in closed loop.** With a stabilising `K` on a free-integrator
  axis, `KS = K/(1+GK) -> 1/G` at low frequency, and `|G| -> infinity` as `omega -> 0` for a double integrator,
  so `KS(0) -> 0`. Physically: holding a constant position offset on a frictionless free stage costs zero
  steady force. On already-recorded closed-loop logs there is **no design freedom at all**.

**The useful reformulation**: "constant over a window of length `T`" is indistinguishable from anything below
`1/T`. At `nf = 400` and 4 kHz that is about 10 Hz, a decade below the existing 130-180 Hz content, and the
`f^4` penalty is far milder there than at 0 Hz.

### The decisive measurement has a standard name and a data-derived threshold

**Raue, A., Kreutz, C., Maiwald, T., Bachmann, J., Schilling, M., Klingmüller, U., Timmer, J.**, "Structural
and practical identifiability analysis of partially observed dynamical models by exploiting the profile
likelihood", *Bioinformatics* 25(15), 2009. DOI `10.1093/bioinformatics/btp358`. **ABSTRACT**.
Profile likelihood separates *structural* from *practical* non-identifiability and yields confidence intervals.
A flat profile whose interval covers the admissible range means the loss does not determine the constant; a
tight interval around a well-determined `b*` means it does and the optimizer sits elsewhere. We already compute
`H` and `b* = -g/H`; the only missing piece is the **threshold**, and this literature supplies one derived from
the measured noise floor rather than from a model.

### Informativity framework (the control-side version of the same question)

**Colin, K., Bombois, X., Bako, L., Morelli, F.**, "Data informativity for the open-loop identification of
MIMO systems in the prediction error framework", *Automatica* 120:109000, 2020. DOI
`10.1016/j.automatica.2020.109000`. Free: HAL `hal-02305057`. **FULL-READ**.
Informativity "ensures the consistency of the prediction error estimate"; an input is sufficiently rich of
order `eta` iff "its power spectrum is nonzero in at least `eta` frequencies"; and the usual sufficient
condition (spectrum strictly positive definite almost everywhere) "is too restrictive (e.g. a multisine input
vector will never respect this condition)". So informativity is a property of *where the spectrum is nonzero*,
and multisines are the case the theory treats specially.

**Gevers, M., Bazanella, A. S., Bombois, X., Mišković, L.**, "Identification and the Information Matrix: How to
Get Just Sufficiently Rich?", *IEEE TAC* 54(12):2828-2840, 2009. DOI `10.1109/tac.2009.2034199`.
**METADATA**, needs-browser-route. The citation for "the information matrix is singular in this direction",
and the highest-priority unfetched item in this sweep.
**Bazanella, A. S., Bombois, X., Gevers, M.**, "Necessary and sufficient conditions for uniqueness of the
minimum in Prediction Error Identification", *Automatica* 48(8):1621-1630, 2012. DOI
`10.1016/j.automatica.2012.06.018`. **METADATA**, needs-browser-route. **This is the control literature's own
version of our §2 fork**: non-uniqueness of the PEM minimum is "the loss does not determine the constant";
uniqueness plus mis-seating is the optimizer reading. The fork is a named, settled question there.

**DERIVED-HERE**: in `M ~ integral F_theta(w) Phi_u(w) F_theta*(w) dw`, a direction whose sensitivity is a pure
DC gain contributes `delta' M delta ~ Phi_u(0) * |F_delta(0)|^2`. With `Phi_u(0) = 0` exactly, that direction
lies in `null(M)`: zero Fisher information, no consistency, estimate fixed by initialisation and optimizer
geometry. Treat as an instantiation of Gevers et al. 2009, not an independent citation. Note this is the
*standstill* case; §3's Y-modulation argument is what breaks it.

**On disk and unread**: `literature/experiment-design/Papers/Optimal identification experiment design for LPV
systems using the local approach.pdf` = **Ghosh, D., Bombois, X., Huillery, J., Scorletti, G., Mercère, G.**,
*Automatica*, 2017, DOI `10.1016/j.automatica.2017.10.013`. The OED tool for a scheduled system. **REPO**.

---

## 4. I3's `H^3.7` is theory-matched, which closes the faithfulness caveat

**Zucchet, N., Orvieto, A.**, "Recurrent neural networks: vanishing and exploding gradients are not the end of
the story", *NeurIPS 2024*. DOI `10.52202/079017-4425`, pp. 139402-139443. Free: `arXiv:2405.21064`.
**FULL-READ** (pp. 1-5, Eqs. 5-6).

They name and quantify the **curse of memory**: as memory lengthens, output sensitivity to parameters explodes
**even when the dynamics stay stable and gradients do not explode**. For `h_(t+1) = lambda*h_t + x_(t+1)` with
wide-sense-stationary input of autocorrelation `R_x`:

```
E[h_t^2]           = 1/(1-lambda^2)     * [R_x(0) + 2*sum_(d>=1) lambda^d R_x(d)]
E[(dh_t/dlambda)^2] = (1+lambda^2)/(1-lambda^2)^3 * [ ... ] + 2/(1-lambda^2)^2 * [sum d*lambda^d*R_x(d)]
```

so sensitivity to the **self-feedback gain** diverges like `(1-lambda^2)^-3`, faster than the state's
`(1-lambda^2)^-1`, and faster still as input autocorrelation rises toward 1.

**DERIVED-HERE**: truncation at horizon `H` caps `1/(1-lambda^2)` at about `H`, predicting a curvature exponent
in the `H^3` to `H^4` band. **Our measured 3.7 sits inside it**, and our multisine at 130-180 Hz with zero DC
power is exactly the strongly-autocorrelated case the law says is worst. The `H^3.7` figure therefore no longer
needs defending as a synthetic-canonical-gain artefact: it reproduces a published closed-form law on the right
parameter class.

Two further consequences:
- Their explosion is specifically for the derivative w.r.t. the **recurrent** parameter, which they state
  dominates over input and readout weights. **That is I3, not I1**, and it is independent theoretical support
  for why I3 is the campaign's most reproducible quantity while the DC behaves differently.
- **Marginal-preserving does not buy conditioning.** They note that forcing the recurrent matrix orthogonal
  fixes vanishing and exploding gradients yet the weights "may remain sensitive to learn because of the curse
  of memory". For a fully connected matrix the blowup is "distributed across all entries", partly explaining
  why dense linear RNNs are hard to train.

Their mitigation is a **reparametrisation, not a constraint** (input normalisation `gamma(lambda) =
sqrt(1-lambda^2)` plus `lambda = exp(-exp(nu))`), so it would pass constraint 1, **but both pieces degenerate
exactly at `|lambda| = 1`**. A lead for the step-geometry fork, not a drop-in.

### Standard practice for curvature along a *learned* direction

A single Hessian-vector product gives `d' H d` exactly for any measured `d`: **Pearlmutter, B. A.**, "Fast Exact
Multiplication by the Hessian", *Neural Computation* 6(1):147-160, 1994, DOI `10.1162/neco.1994.6.1.147`
(**METADATA**, needs-browser-route, but the method is textbook and in every AD framework). Spectrum and
density: **Yao, Z., Gholami, A., Keutzer, K., Mahoney, M. W.**, "PyHessian", *IEEE BigData 2020*, DOI
`10.1109/bigdata50022.2020.9378171`, free `arXiv:1912.07145`; **Ghorbani, B., Krishnan, S., Xiao, Y.**,
*ICML 2019*, `arXiv:1901.10159`. Whether the learned direction is the relevant one: **Gur-Ari, G.,
Roberts, D. A., Dyer, E.**, `arXiv:1812.04754`, the gradient collapses into the top-Hessian subspace, so the
defensible probe is `d' H d` along the measured direction **plus** the overlap of `d` with that subspace.

---

## 5. Candidates that survive the frame

Marked against the problem statement's §6 constraints.

### C1. Prox on the orthogonal complement of the marginal subspace (recommended)

**Dinev, D., Liu, T., Kavan, L.**, "Stabilizing Integrators for Real-Time Physics", *ACM Trans. Graphics*
37(1), Article 9, pp. 1-19, 2018. DOI `10.1145/3153420`. Free:
`users.cs.utah.edu/~ladislav/dinev18stabilizing/dinev18stabilizing.pdf`. **FULL-READ**.

They state our constraint-2 dilemma verbatim in numerical-analysis words: "The commonly used backward Euler
method is stable but introduces artificial damping. Methods such as implicit midpoint do not suffer from
artificial damping but are unstable in many common simulation scenarios."

The transferable part is their damping model, which **exempts the free modes by construction**: compute the
difference between each vertex velocity and its best-fit rigid-body velocity, and "damp out only these
non-rigid velocity components". That is exactly "regularise on the orthogonal complement of the marginal
subspace, leave the marginal mode untouched". Cited implementation: **Müller, M., Heidelberger, B., Hennix, M.,
Ratcliff, J.**, "Position based dynamics", DOI `10.1016/j.jvcir.2007.01.005` (**METADATA**).

**Constraint marks**: 1 pass (restricts no dynamic class, only where the penalty acts). 2 pass by construction.
3 pass, because the subspace is the **known** rigid-body null space of the baseline, not a measured drift
direction, so this is *not* the closed "re-aim the projection at the measured drift direction" item.
4 pass. **We already own both pieces**: an orthogonal projection, plus I5's verified exact prox.
It reframes the drift work from *whether* to regularise to *on which subspace*.

### C2. Spectral regulariser on the tendency, not the state

**Guan, H., Arcomano, T., Chattopadhyay, A., Maulik, R.**, "LUCIE: A Lightweight Uncoupled Climate Emulator
With Long-Term Stability and Physical Consistency", *JAMES*, 2025. DOI `10.1029/2025MS005152`. Free:
`arXiv:2405.16297`. **PART-READ** (§3.2, §3.3; body grepped for `dissipat|damp|relax|nudg`, none found).

100-year stable rollouts from as little as 2 years of data, via two mechanisms and no damping, nudging or
relaxation anywhere. Mechanism 1, "Euler integration as a hard constraint", predicting the **tendency** and
integrating it: **we already have this**, since the ANN is routed as a force. Mechanism 2, a **spectral
regulariser on the tendency**: we do not.

**Constraint marks**: 1 pass. 2 pass structurally, since a penalty on the ANN's own output cannot move baseline
poles. 3 pass, and note this is **not** the closed zero-mean item: the penalty matches predicted against
reference tendency spectra, i.e. it uses a data property, rather than imposing a prior on the residual's mean.
4 pass.

### C3. PES instead of ARTBP

**Vicol, P., Metz, L., Sohl-Dickstein, J.**, "Unbiased Gradient Estimation in Unrolled Computation Graphs with
Persistent Evolution Strategies", *ICML 2021*, PMLR v139; extended abstract *IJCAI 2022*, DOI
`10.24963/ijcai.2022/750`. Free: `arXiv:2112.13835`. **FULL-READ** (pp. 1-7, Statement 4.1, Table 2, Fig. 3).

Accumulates ES perturbations across truncated unrolls instead of resetting them. Their Table 2 gives total
variance by the structure of the gradient matrix; a free integrator with no forgetting is the
**upper-triangular, near-identical-gradient** case, giving variance **`O(PT)`: linear in horizon and parameter
count, not heavy-tailed**. Two properties that matter here: variance is proportional to `||g||`, so a
perfect-match null with near-zero gradient does **not** inflate it, the exact opposite of ARTBP's behaviour on
our null; and it matched *exact RTRL* on the influence-balancing task where UORO needed ~30k iterations and
short-truncation TBPTT diverged. PES+Analytic (their Eq. 9) cuts variance 1-2 orders.

**Constraint marks**: 1 pass trivially (estimator only). 2 pass. 3 pass. 4 pass.
**Costs**: `N*K*F` per step with `N` particles; unbiased for the *smoothed* objective; variance carries `P`,
fine for a small ANN block.

### C4. Exact forward-mode sensitivity, worth a feasibility check before anything stochastic

**Zucchet, N., Meier, R., Schug, S., Mujika, A., Sacramento, J.**, "Online learning of long-range
dependencies", *NeurIPS 2023*. DOI `10.52202/075280-0460`, pp. 10477-10493. Free: `arXiv:2305.15947`.
**PART-READ** (p. 1).
Exact forward-mode (RTRL) credit assignment at **twice** the memory and compute of one inference pass, by
exploiting independent (element-wise) recurrent modules. **Zero estimator variance**, memory independent of
sequence length. Normally the element-wise precondition is an architecture restriction, but our baseline is a
known LPV-LFR state matrix with the ANN routing force onto three velocity rows, so maintaining
`d s_t / d theta` exactly **for the marginal modes specifically** may be tractable. Approximate family for
completeness: UORO `arXiv:1702.05043`, KF-RTRL `arXiv:1805.10842`, OK-RTRL `arXiv:1902.03993`,
SnAp `arXiv:2006.07232` (SnAp-1 exact for element-wise), unified comparison `arXiv:1907.02649`.

### C5. Multiple shooting (control-side, unchanged verdict)

**Turan, E. M., Jäschke, J.**, "Multiple Shooting for Training Neural Differential Equations on Time Series",
*IEEE Control Systems Letters* 6:1897-1902, 2022. DOI `10.1109/lcsys.2021.3135835`. Free: `arXiv:2109.06786`.
**METADATA**. Short segments with independent initial states and continuity re-imposed by penalty or augmented
Lagrangian, decoupling effective horizon from gradient path length with no variance term. Already in
`literature/stability-training/claude-deep-research-inwindow-accumulation.md` Gap 4 (**REPO**). Structural fit:
our drift is a *state* error accumulating over the window, and shooting variables are exactly the freedom to
absorb it without asking the gradient to reach further back.

### C6. Parity as a DIAGNOSTIC only (see §7 for why not as a regulariser)

**Rijlaarsdam, D., van Loon, B., Nuij, P., Steinbuch, M.**, "Nonlinearities in Industrial Motion Stages:
Detection and Classification", *ACC 2010*, pp. 6644-6649. DOI `10.1109/ACC.2010.5531368`. Free at TU/e Pure:
`pure.tue.nl/ws/files/3176591/583911629801061.pdf`. **FULL-READ**.
On an industrial high-precision stage: excite with a **random odd multisine**, then compare power at
non-excited **odd** versus **even** lines. Repeated periods of one realisation give the **noise** variance;
repeated realisations give the **nonlinear** variance, so the floor is data-derived and there is no oracle.
Measured: odd contribution ~20 dB above even, noise ~30 dB below the nonlinear variance, and the conclusion
drawn from the parity signature alone is that "the main source of nonlinear behaviour is friction in this case".
**Constraint marks**: all four pass as a measurement.
**Structural caveat**: this parity is in the input-output *frequency* domain (odd versus even harmonics), which
coincides with velocity-domain parity only insofar as the residual is a memoryless function of the excited
velocity. §7 reopens exactly that gap.

Complementary, for the spectrum leg from routine closed-loop logs: **Choudhury, M. A. A. S., Shah, S. L.,
Thornhill, N. F.**, "Diagnosis of poor control-loop performance using higher-order statistics", *Automatica*
40(10):1719-1728, 2004. DOI `10.1016/j.automatica.2004.03.022`. Free via CORE. **PART-READ**. Bicoherence-based
non-Gaussianity and nonlinearity indices from routine operating data, distinguishing an actuator nonlinearity
(stiction, backlash, deadzone) from an external linear disturbance or poor tuning. It detects and localises;
it does **not** report parity, so it complements C6 rather than replacing it.

---

## 6. Corrections to `docs/drift-problem-statement.md`

Proposed, not applied. Each is a text change to the statement, not a change of direction.

| # | Where | Correction |
|---|---|---|
| 1 | §4 I4 | The near-unit-root sentence has its logic backwards. Dou and Müller derive non-estimability of `c` **from** non-discriminability, not the reverse. |
| 2 | §4 I4 | "Any criterion whose only input is the existing data cannot distinguish ..." is too strong. Müller 2008 proves the impossibility is **one-sided**: a consistent unit-root test exists; no consistent scale-invariant *stationarity* test does. Certifying presence of the unit root is possible; certifying **absence of the drift** is what is impossible. We are on the hard side, so the claim survives, but state the asymmetry. |
| 3 | §4 I4 | The "known pole plus structural criterion" escape is **supported** by the same papers: every impossibility result in this family is stated only over the **unrestricted** parameter class. |
| 4 | §2 | The branch-(a) remedy "or a prior on a provably non-identifiable direction" does not identify anything. Faust 1996: for any prior there is another with identical observable implications disagreeing arbitrarily about the long-run effect. A prior is a declared **choice**. |
| 5 | §4 I3 | "poly-tail variance gain over geometric of only 2-5x" is undefined if the variance is infinite. Restate as a quantile or truncated moment. |
| 6 | §4 I3 | The `H^3.7` faithfulness caveat can be closed: it matches Zucchet and Orvieto's `(1-lambda^2)^-3` law, band `H^3` to `H^4`. |
| 7 | §5.5 | "Practically non-identifiable from this data" is weaker than stated: the Y-sweep records already supply LPV modulation of the constant, independent of input spectrum. |
| 8 | §6 constraint 4 | Add the two physical obstructions to the input-design remedy: `f^4` admissible-power law under a stroke bound, and `KS(0) -> 0` in closed loop. |
| 9 | §5.2 / §6 | Parity separation is established in five vocabularies but is **disqualified as a regulariser** by constraint 3 (oddness is an assumption about the unknown residual) and by constraint 1 (pre-sliding friction has no velocity parity). It passes as a **measurement**. See §7. |
| 10 | thread A3/A4 | `scripts/gantry/drift-fix-trials/research/thread-AB-optimizer-mechanics.md`: A3 upgrades PARTIAL-FETCH to PRIMARY-READ (Zhuang Eqs. 3 and 5 verified from PDF). A4's caveat "neither paper states the beta-saturation result itself" is now **false**: Bock and Weiss give the closed form. |

### Claims of the statement that this sweep CONFIRMS

- **§6 constraint 4's impossibility stands.** One agent argued that AntisymmetricRNN and Euler State Networks
  refute "full expressivity and a for-all-weights no-drift guarantee are logically incompatible", because they
  impose marginality for all weights structurally. That conflates two guarantees: preserving `|lambda| = 1`
  constrains the **homogeneous** dynamics, while drift comes from the **forced** response, and a marginal
  system under a constant force drifts exactly as before. The claim is unrefuted; no-drift still comes from
  the estimator.
- **I5's proximal result is mechanism-published but measurement-novel.** See §8.

### A mechanism hypothesis that does NOT transfer

One agent proposed that I3's positive self-feedback is an explicit-Euler artefact, from AntisymmetricRNN's
Proposition 2 that `|1 + eps*lambda| > 1` always when the Jacobian's eigenvalues are purely imaginary. That
holds for **oscillatory** marginal modes at `s = ±i*w`. Our marginal modes are free integrators at `s = 0`,
which forward Euler maps to `z = 1` exactly. The mechanism predicts nothing here, and the measured positive
`dF/dv` remains a real destabilising term in continuous time. Do not chase it.

---

## 7. Parity: established, and why it is a measurement rather than a penalty

Established as a mechanism-attribution discriminator in five vocabularies (control and nonlinear sysid via odd
multisines; tribology and friction identification; robot dynamic parameter identification; navigation via
rotation modulation; signal processing via higher-order statistics).

- **Ku, X., Li, Z., Zhu, Y.**, *Machines* 14(7):783, 2026. DOI `10.3390/machines14070783`. **FULL-READ**.
  Their framing is ours verbatim: least squares "cannot decouple strong nonlinear friction residuals from
  inertial identification **bias**". Their protocol: at constant velocity, acceleration terms vanish and
  Coriolis satisfies `C(q,qd)qd = C(q,-qd)(-qd)`, so under an **assumed** odd friction the reversal isolates
  pure friction. Parity is an explicit assumption, not a derived property.
- **Righettini, P., Legnani, G., Cortinovis, F., Tabaldi, F., Santinelli, J.**, *Robotics* 14(4):36, 2025.
  DOI `10.3390/robotics14040036`. **FULL-READ**. Parity as a structural prior on the friction regressor,
  `tau_F = sign(v)*[1, |v|, |v|^2, ...]*pi_F`, with the authors' own note that an asymmetric model "could also
  ... be easily derived at the cost of a doubled number of parameters". **That doubling is the constraint-1
  price, stated by the authors.**
- **Wu, Y., Wang, Z., Ren, X., Zhou, C.**, *Symmetry* 17(7):1012, 2025. DOI `10.3390/sym17071012`.
  **FULL-READ**. Stribeck friction "exhibits an **approximately** odd-function characteristic". Note the
  qualifier.

### The decisive objection: hysteresis destroys pointwise parity

**Ruderman, M., Rachinskii, D.**, "Use of Prandtl-Ishlinskii hysteresis operators for Coulomb friction modeling
with presliding", *J. Physics: Conf. Series* 811:012013, 2017 (MURPHYS 2016). DOI
`10.1088/1742-6596/811/1/012013`. Free at IOP. **FULL-READ**.

Their Eq. (10) is `F(y, yd) = c1*sign(yd) + integral_0^R f(y,r) dmu(r)`, with `f` the state of a play-type
Prandtl-Ishlinskii operator. **Exactly one term is the memoryless odd `sign(yd)`; every other term is a
function of displacement with memory.** And explicitly: the discontinuous Coulomb force "constitutes the limit
case of presliding hysteresis curves at motion reversals", with `F -> F_c*sign(yd)` only as all elementary
stiffnesses tend to infinity. **So the pointwise-odd-in-velocity friction the parity criterion assumes is the
memoryless limit of real pre-sliding friction. In pre-sliding, `v -> F` is not a function at all, so it has no
parity.**

What survives is operator-level (trajectory-level) oddness. Corroborating evidence: the "generalized" and
"asymmetric" Prandtl-Ishlinskii families exist precisely because the classical operator produces only symmetric
loops (Al Janaideh, Su, Rakheja, *AIM 2010*, DOI `10.1109/AIM.2010.5695767`; Li et al., *ACC 2012*, DOI
`10.1109/ACC.2012.6315018`; Zhou et al., *Sensors* 22(22):8763, DOI `10.3390/s22228763`; all **METADATA**,
but the pattern is itself the finding).

### Constraint verdict

| constraint | verdict |
|---|---|
| 1 expressivity | **Hard odd-only constraint DISQUALIFIED.** Asymmetric friction needs the doubled parameterisation, and pre-sliding friction is not in the odd-in-`v` class at all. Diagnostic, or a soft/prox penalty on the even part only, passes. |
| 2 marginal poles | Pass. A penalty on the parity decomposition of the ANN output moves no baseline pole. |
| 3 knowledge-free | **As a regulariser DISQUALIFIED**: oddness is knowledge about the unknown residual, asserted as an assumption by Ku et al., hedged as "approximately" by Wu et al., and false in pre-sliding per Ruderman and Rachinskii. **As a measurement of the real Telica residual, pass**, which is what §6 constraint 4's "what would change this" asks for. |
| 4 real-data viability | Pass. Rijlaarsdam et al. and Choudhury et al. both run on real noisy data with data-derived floors. |

### The protocol this recommends

On the real Telica logs, at trajectory level rather than pointwise: extract matched constant-velocity forward
and reverse segments at the same operating point (Ku et al.'s protocol), form the half-difference (odd, genuine
friction) and half-sum (even, bias plus position-dependent effects), and take the noise floor from repeated
realisations rather than from the model (Rijlaarsdam et al.'s two-variance split). Report the odd/even ratio in
dB against that floor. If the even part sits at the noise floor while the odd part is well above it, §6
constraint 4 converts from argument into evidence, without training anything. This touches real data and is
therefore an ASK gate.

**Gap in existing holdings this fills**: `literature/augmentation/Literatuuronderzoek voor Dahl-frictiestaten
in een H-type dual-drive gantry stage.pdf` (**REPO**, read in full by the agent) covers Dahl/LuGre/GMS model
structures, `F_c`/`F_s` parameter orders, position-dependent friction and pre-sliding excitation design, but
covers **neither** velocity parity **nor** bias-versus-friction separation anywhere.

---

## 8. Novelty position, with vocabularies stated (rule 117)

### Published, do NOT claim as new

- **The I5 saturation mechanism.** Already logged in
  `scripts/gantry/drift-fix-trials/research/thread-AB-optimizer-mechanics.md` A3/A4 (**REPO**).
  **Zhuang, Z., Liu, M., Cutkosky, A., Orabona, F.**, "Understanding AdamW through Proximal Methods and
  Scale-Freeness", *Transactions on Machine Learning Research*, 2022, OpenReview `IKhEPWGdwK`, free
  `arXiv:2202.00089`. **FULL-READ** pp. 1-6 this session, which closes A3's open verification request.
  Their Eq. (3) is the variable-metric prox `(I + lambda*M_t)^-1 (x_(t-1) - M_t p_t)`; with `M_t = eta_t I` it
  becomes Eq. (5), **character for character our `c/(1 + lr*beta)` after the Adam step**, and AdamW is its
  first-order Taylor expansion with remainder `O(lambda*eta^2)`. Scale-freeness: "once Adam-l2 adopts a
  non-zero lambda, it loses the scale-freeness property; in contrast, AdamW enjoys this property for arbitrary
  lambda", extended to "any AdaGrad-type and Adam-type algorithm that incorporates the squared l2 regularizer
  by simply adding the gradient". They also name our null's pathology: from `x_(t-1) = 0` with zero gradient the
  in-loss update "may actually *increase* the weights by causing `x_t` to *overshoot* the origin. In contrast,
  the proximal update will never demonstrate such pathological behavior."
  Primary ancestor for the effective-strength argument, and not previously cited in the thread:
  **Kingma, D. P., Ba, J.**, *ICLR 2015*, `arXiv:1412.6980`, §2.1 (**FULL-READ**), which proves the update is
  invariant to gradient rescaling and bounds the effective step by `alpha` as a trust region. Also
  **Loshchilov, I., Hutter, F.**, *ICLR 2019*, `arXiv:1711.05101`, whose Proposition 3 makes the
  `1/sqrt(s)` effective-strength rescaling exact.
- **Enforcing a physics condition outside the optimizer step.** Established in safe neural control as a hard
  SDP projection after each update: **Junnarkar, N., Arcak, M., Seiler, P.**, `arXiv:2404.07373v2`
  (**PART-READ** pp. 1-2). Disqualified as a deliverable by constraints 1 and 2, but it makes "enforce outside
  the step" a documented practice, and our finite-beta prox is precisely the **interpolation** between it and
  DiLaR-Soft's in-loss penalty.
- **ARTBP's two failure modes** (§2 above), in ARTBP's own paper.

### Genuinely unreported, per this sweep

Each with the vocabularies searched.

1. **A controlled measurement, at matched coefficient and multiple seeds, that an in-loss penalty plateaus
   coefficient-independently while the exact prox is monotone, for a non-weight-decay penalty.** AdamO
   (**Rosseau, A., Muller, R., Nowe, A.**, *ICML 2026*, `arXiv:2606.09762`) asserts the mechanism for an
   isometry penalty ("mixing them inside the adaptive preconditioner can undesirably rescale task updates or
   make the effective regularization strength highly parameter- and time-dependent") but never runs the
   in-loss versus decoupled comparison at matched coefficient, and its step is additive rather than an exact
   prox. Zhuang et al. run the comparison but measure update-magnitude concentration and generalization, not
   the penalized quantity. *Vocabularies: ML optimization, convex optimization and operator splitting,
   numerical analysis, statistics/shrinkage, PINNs.* **This is a claim about measurement, not mechanism.**
2. **A one-sided power or dissipativity-violation penalty applied proximally, or outside the adaptive metric.**
   Measured zeros: `"one-sided" AND "penalty" AND "dissipative"` = 0; `"energy injection" AND "neural network"
   AND "penalty"` = 0; `"decoupled weight decay" AND "proximal"` = 0; only **2** arXiv abstracts contain both
   "proximal operator" and "physics-informed". *Vocabularies: control, ML, thermodynamics-informed ML, convex
   optimization, numerical analysis.* Note the thermodynamics line mirrors DiLaR exactly, soft penalty
   (Hernández et al., *J. Comput. Phys.* 426:109950, 2021, **FULL-READ** pp. 4-7) then hard structural
   parameterisation (Gruber, Lee, Lim, `arXiv:2405.16305`); the prox third option is never taken in either
   field.
3. **Variance theory for unbiased truncated-BPTT estimators at spectral radius exactly 1.** Declared out of
   scope by the field's reference treatment (§2). *Vocabularies: ML, control, stochastic approximation,
   numerical analysis.*
4. **Excitation design so a learned residual becomes identifiable.** PROVISIONAL: OpenAlex rate-limited that
   sweep partway and IFAC coverage reached ~30% of one year. *Vocabularies: control/sysid, ML, statistics.*
5. **Lossless (`R = 0`) port-Hamiltonian neural networks as a marginal-preserving device**, and **learned
   residuals constrained to preserve a known null space.** Both measured zeros on arXiv abstract search. The
   prior sweep's hope that passivity admits the lossless mode natively is right in principle and unimplemented.
6. **Velocity parity applied to a learned residual, at training time, or on a marginally stable axis where the
   even part integrates twice.** Established in five vocabularies but never in that combination (§7).

### Formal expressivity cost of exact marginality

**Emami, M., Sahraee-Ardakan, M., Pandit, P., Rangan, S., Fletcher, A. K.**, "Input-Output Equivalence of
Unitary and Contractive RNNs", *NeurIPS 2019*. Free: `arXiv:1910.13672`. **FULL-READ** (Thms 3.1, 3.2, 4.1).
For **ReLU**, any contractive RNN with `n` states has an exactly-unitary equivalent with `2n` states (Thm 3.1),
and the factor 2 is **tight** (Thm 3.2). For **smooth** activations, there exists a contractive RNN with no
equivalent controllable-and-observable unitary network **at any state dimension** (Thm 4.1), because smooth
positive-slope activations let you linearise and make the transition eigenvalues visible in the input-output
map. **The cost of exact marginality is activation-dependent, and a tanh ANN sits in the restrictive regime.**
Scope limit to state plainly: unitary means **all** eigenvalues on the unit circle, while we need only two, so
this bounds the extreme class rather than our actual constraint.

Adjacent, for completeness: **Chang, B., Chen, M., Haber, E., Chi, E. H.**, AntisymmetricRNN, *ICLR 2019*,
`arXiv:1902.09689` (**FULL-READ**), whose "critical criterion" `Re(lambda) ~ 0` is constraint 2 independently
derived in ML words, with the explicit argument that strictly-negative real parts are a *defect*.
**Gallicchio, C.**, Euler State Networks, *Neurocomputing* 566:127411, 2024, DOI
`10.1016/j.neucom.2024.127411`, `arXiv:2203.09382` (**FULL-READ**), marginality "irrespectively of the
particular choice of [the weight] values". **Karuvally, A., et al.**, AUSSM, *NeurIPS 2025*,
`arXiv:2507.05238` (**ABSTRACT**), skew-symmetric *input-dependent* recurrence, i.e. the nearest structural
analogue to a self-scheduled LPV marginal parameterisation; its expressivity claims are formal-language, not
dynamical-approximation, so they do not transfer directly. The practical device for exact `|lambda| = 1` is the
**Cayley transform**, which is the same machinery already used in this framework for well-posedness, applied
to a different matrix.
Noether route: **Müller, E. H.**, "Exact conservation laws for neural network integrators of dynamical
systems", *J. Comput. Phys.* 488:112234, 2023, DOI `10.1016/j.jcp.2023.112234`, `arXiv:2209.11661`
(**PART-READ**), exact linear-momentum conservation **is** preservation of the free translational mode. Forced
variants are the ones that fit our setting: Variational Integrator Networks `arXiv:1910.09349` and Forced VINs
`arXiv:2106.02973` (**METADATA**).
The classical no-go: fixed-timestep methods cannot have symplecticity, energy conservation and momentum
conservation together (**Ge Zhong, Marsden**, *Phys. Lett. A* 133(3):134-139, 1988, DOI
`10.1016/0375-9601(88)90773-6`); adaptive timestepping escapes it (**Kane, Marsden, Ortiz**, *J. Math. Phys.*
40(7):3353-3371, 1999, DOI `10.1063/1.532892`); the with-symmetry case is **Gonzalez, Simo**, *CMAME*
134(3):197-222, 1996, DOI `10.1016/0045-7825(96)01009-2`. All **METADATA**, read through Dinev et al.'s
bibliography, so second-hand until verified.

### Near-unit-root theory: the citations I4 was missing

- **Pötscher, B. M.**, "Lower Risk Bounds and Properties of Confidence Sets for Ill-Posed Estimation Problems
  with Applications to Spectral Density and Persistence Estimation, Unit Roots, and Estimation of Long Memory
  Parameters", *Econometrica* 70(3):1035-1065, 2002. DOI `10.1111/1468-0262.00318`. Free: author homepage
  (Sept 2000 working-paper version). **PART-READ**.
  **The strongest single citation in this sweep.** The minimax risk for estimating the spectral density **at
  frequency zero** is bounded away from zero for every estimator at every sample size, and often infinite,
  absent restrictive a priori assumptions on the feasible data-generating processes; and the phenomenon "will
  occur regardless of how large sample size is". Our drift-carrying direction **is** the zero-frequency content
  of the residual force, so this is a theorem about our exact quantity, not an analogy. It is also the right
  backing for constraint 3: the reason a residual-mean assumption cannot be dodged and then recovered from data
  is a lower risk bound, not a modelling preference. **Quote with care: the free copy is the working paper, so
  quotes carry no journal page numbers.**
- **Müller, U. K.**, "The Impossibility of Consistent Discrimination between I(0) and I(1) Processes",
  *Econometric Theory* 24(3):616-630, 2008. DOI `10.1017/s0266466608080250`. Free: author homepage,
  journal-typeset. **FULL-READ**. Thm 1: a consistent unit-root test with correct asymptotic size exists.
  Thm 2: no consistent scale-invariant *stationarity* test does. Also characterises the earlier line: the
  problem is ill-posed "**without further restrictions on the parameter space**".
- **Dou, L., Müller, U. K.**, "Generalized Local-to-Unity Models", *Econometrica* 89(4):1825-1854, 2021.
  DOI `10.3982/ecta17944`. Free: author homepage, journal-typeset. **FULL-READ**. "since the LTU model cannot
  be perfectly discriminated from the unit root model, the parameter `c` in (1) cannot be consistently
  estimated." Note the direction of implication.
- **Faust, J.**, "Near Observational Equivalence and Theoretical Size Problems with Unit Root Tests",
  *Econometric Theory* 12(4):724-731, 1996. DOI `10.1017/s0266466600007003`. Free: Fed IFDP 447 (1993
  working paper). **PART-READ**. "we cannot distinguish any values for the long-run effect of shocks", plus
  the Bayesian result behind correction 4 in §6.
- The LTU parameterisation `rho_T = 1 - c/T` traces to **Phillips, P. C. B.**, *Biometrika* 74(3):535-547,
  1987, DOI `10.1093/biomet/74.3.535`. **METADATA**, no free copy located, so the original statement rests on
  Dou and Müller's restatement.

### What is estimable near a unit root, and it inverts the intuition

**Simchowitz, M., Mania, H., Tu, S., Jordan, M. I., Recht, B.**, "Learning Without Mixing", *COLT 2018*,
`arXiv:1802.08334` (**PART-READ**): OLS is nearly minimax optimal from a single trajectory, and "**more
unstable linear systems are easier to estimate**", qualitatively opposite to mixing-time intuition.
**Sarkar, T., Rakhlin, A.**, *ICML 2019*, `arXiv:1812.01251` (**PART-READ**): sharp finite-time bounds in three
separate regimes, **stable, marginally stable, explosive**, matching the lower bound up to log factors.
Reframing for the thesis: near `|lambda| = 1` the **pole** is the easy object; the hard objects are the local
deviation and the zero-frequency offset. Our problem is not that the marginal mode is hard to identify, it is
that **the constant on the marginal mode is the one quantity in the model with a lower risk bound bounded away
from zero.**

---

## 9. Recommended next step: four cheap diagnostics (three of them training-free)

All four reuse machinery already in the repo and none needs a training *campaign*, but they are not all
training-free, and the distinction matters for budgeting. Corrected accounting, after checking the repo for
reusable checkpoints on 2026-07-25:

- **D1 and D4 train nothing.** D1 is evaluated at the zero-output init, which for the perfect-match null IS the
  correct optimum and is the same point the OTHER-RIG `b*` was measured at. D4 is analysis of real logs.
- **D2 and D3 train.** `scripts/gantry/drift-fix-trials/data/` holds no ANN checkpoints (45 files, all unit
  JSONs plus the Coulomb truth npz), so D2 must produce its own trained point: one 84-step full-batch run per
  seed, about 550 s each. D3 is two short fits per seed.
- **The three `.pt` checkpoints that do exist**, all under `scripts/gantry/ARTBP/data/`, are **OTHER-RIG and
  disqualified** as D2's trained point: `train_artbp.py` takes routing from the production `CFG` and never sets
  `nx_ann=0` or `ann_route_ix=(3,4,5)`, uses stride 40 against the rig's 100, uses an ARTBP estimator rather
  than plain full-batch Adam, and exists for seed 0 only. They are permitted only as separately-headed
  cross-rig corroboration. The six-point provenance gate is in
  `scripts/gantry/drift-diagnostics/README.md` §4.1.

Execution instructions for all four, explicit enough to run without supervision, are in
**`scripts/gantry/drift-diagnostics/README.md`**, with a paste-in starter in `SESSION-PROMPT.md` beside it.

1. **Profile-likelihood interval plus Frye's `r(p)`** on the 6-D output-DC subspace at routing (3,4,5), with
   the threshold from the measured noise floor. Settles the §2 fork with no oracle. (§1)
2. **`d' H d` by Pearlmutter HVP along the *measured* trained self-feedback direction** at `H = 400, 800,
   1600`, checked against the `H^3` to `H^4` band, plus the overlap of that direction with the top-Hessian
   subspace. Replaces the synthetic canonical gain and tests the curse-of-memory mechanism. (§4)
3. **ysweep-only versus standstill-only fit** of the constant. **NOT training-free**: two short fits, 84 steps
   and 3 seeds each, on records already on disk (`T6/T7/T8` versus `T1-T4`). No new data and no new hardware.
   Direct test of leg (a) versus leg (b): if the Y-scheduling confers identifiability, the ysweep-only fit
   should pin the constant markedly closer to zero while the standstill-only fit should not. (§3)
4. **Trajectory-level parity test on real Telica logs**, half-sum versus half-difference of matched
   forward/reverse constant-velocity segments, floor from repeated realisations. Converts §6 constraint 4 from
   argument into evidence. **ASK gate: touches real data.** (§7)

Additionally, before quoting the Bock and Weiss prefactor: resolve the 8.9x gap by checking bias correction and
`eps`, and whether the routed constant equals the penalized parameter. (§1)

---

## 10. `needs-browser-route` queue, ranked

TU/e access is AVAILABLE and verified; these were simply not fetched in this sweep.

1. **Gevers et al. 2009**, *IEEE TAC* 54(12):2828-2840, DOI `10.1109/tac.2009.2034199`. The information-matrix
   citation; the only load-bearing item whose precise theorem statement is unread.
2. **Park, Choi 2018**, *Int. J. Automotive Technology* 19(3):443-453, DOI `10.1007/s12239-018-0043-y`. Its
   indexed snippet is literally the S7 mechanism ("set position as an even function and velocity as an odd
   function in order to separate them"); snippet only, CORE deposit returns `BlobNotFound`.
3. **Bazanella, Bombois, Gevers 2012**, *Automatica* 48(8):1621-1630, DOI `10.1016/j.automatica.2012.06.018`.
   The control literature's own version of our §2 fork.
4. **Al-Bender, Lampaert, Swevers 2004**, *Chaos* 14(2):446-460, DOI `10.1063/1.1741752`. The `dF/dv` sign leg
   rests on its Crossref abstract.
5. **Rijlaarsdam et al. 2011**, *MSSP* 25(8), DOI `10.1016/j.ymssp.2011.08.008`; and 2017 *Mechatronics*
   survey, DOI `10.1016/j.mechatronics.2016.12.008`.
6. Then: Ge Zhong and Marsden 1988; Gonzalez and Simo 1996; Müller et al. 2007; Rhee et al. 2004;
   Goshen-Meskin and Bar-Itzhack 1992 (DOI `10.1109/7.165368`); Vanhoenacker, Dobrowiecki, Schoukens 2001
   (DOI `10.1109/19.963166`); the asymmetric-PI cluster; Rhee and Glynn 2015; Pearlmutter 1994;
   Bazanella et al. 2010 EJC; Narasimhan and Bombois 2012; Banerjee et al. 2026;
   UEPI (*NeurIPS 2025*, no arXiv record); Hairer 2006 and Hairer/Lubich/Wanner 2006 (books, absent from the
   citation graph entirely).

---

## 11. Merged Research Log

Seven agents, deduplicated by DOI. Skill fixes applied to
`.claude/skills/deep-research/SKILL.md` on 2026-07-25.

**What worked.** Author-ID enumeration with **local** regex filtering was the highest-recall move in four of
seven runs, including regexing Johan Schoukens's 1004-work corpus for parity vocabulary with no topical
keyword. Reading the seeds in full before querying was decisive for the gradient-estimator question, where both
negative results were already in ARTBP's own text. Crossref `query.bibliographic` resolved econometrics at rank
1 on six of six attempts where OpenAlex `title.search` returned zero. Grepping long PDFs beat paging them: two
`pypdf` greps found the three load-bearing passages in a 126-page monograph.

**The mandatory Google Scholar cross-check earned its slot four times out of seven**, each time from a
community with no link to the seeds: Frye et al. and Cockpit; Massé and Ollivier's in-body scope disclaimer;
the Rivera integrating-processes paper; Dinev et al. from computer graphics; Park and Choi plus the robot
reversal cluster. Two distinct reasons, and the second is new: recall breadth, and retrieval of **in-body scope
disclaimers**, which is where "nobody has done this" is actually written and which no title or abstract index
can reach.

**What failed.** `mcp__paper-search__search_arxiv` is broken for multi-concept queries, condemned
independently by four agents (it matched the token "Deep" in author names, returned particle physics for an
exact title, and returned air-pollution LSTMs and symplectic 4-manifolds for a marginal-stability query). dblp
was near-worthless here, 0 useful from 9 queries, because it AND-matches **titles** and cannot find a
**property** such as variance, bias or conditioning. **OpenAlex hit a shared per-IP daily spend cap**, and its
429 returns valid JSON with no `results` key, which every snippet in the skill rendered as zero hits: three
apparent misses were refusals, and one agent ran **no citation-graph queries at all**.

**The worst failure was silent and wrong rather than an error.** The shared scratchpad root caused two sibling
collisions: one agent's `enum.py` shadowed the stdlib and broke every `python` call, and worse, a sibling
overwrote `cr.py` mid-run so a batch of six Crossref lookups returned confident, correctly-formatted metadata
for **entirely unrelated papers**, which was nearly reported as findings.

**Dead ends.** Weak-identification econometrics (weak instruments is rank deficiency, i.e. leg (a) restated,
with no statistic for "stiff but mis-seated"). Systems-biology profile-likelihood follow-ons, several of which
explicitly *conflate* non-identifiability with optimizer non-convergence. Passivity shortage and excess of
passivity (system-theoretic indices, not training penalties, though useful as measurement vocabulary for our
`+2e-8` to `+3e-8`). `secular growth` is 89 of 89 cosmology and QFT, so that translation does not exist even
though the content does under "long-time energy conservation" and "energy drift". ORP-EIS. Google Scholar's
"proximal" space is owned by Proximal Policy Optimization, 13 of 15 hits.

**Coverage gaps.** CDC, ECC and ACC 2023-2026 are effectively unreached across all seven sub-questions, because
dblp was the only route and it was spent or blocked; this is the largest hole. IFAC-PapersOnLine was enumerated
to roughly 30% of one year. ICLR was never enumerated. Forward citation traversal was unusable at the frontier
and unavailable once OpenAlex capped. DiLaR's citation graph is **structurally** unreachable: both OpenAlex
records are preprints with zero references and Crossref has no IFAC version yet, so the usual "a zero means
wrong record" rule does not apply. Massé and Ollivier §6.5-6.6 unread. Phillips 1987 unread.
