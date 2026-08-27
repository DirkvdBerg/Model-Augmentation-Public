# Narrowband-objective literature sweep, 2026-07-26

Input document: `docs/narrowband-objective-problem-2026-07-26.md`.
Procedure: `.claude/skills/deep-research/SKILL.md`, step 0 FRAME plus a 5-agent fan-out.
Predecessor sweeps (read these first, they are NOT superseded):
`docs/drift-literature-sweep-2026-07-25.md`, `docs/multiple-shooting-sweep-2026-07-25.md`,
`scripts/gantry/drift-fix-trials/research/thread-{AB,CD,EF}*.md`,
`docs/rollout-stability-literature.md`.

**Two findings in this sweep were CORRECTED by a full-text read after the agents reported them
second-hand. Both corrections are in Section 3. Do not cite the agent-level version.**

## 0. Frame that was used

| # | Item |
|-|-|
| Sub-questions | 5, listed in Section 2 |
| Seed IDs from the document | `arXiv:2406.03760`; Ribeiro Thm 2; Beatson and Adams ICML 2019; Fisher et al. ECMWF 2011 |
| Disqualification filter | (1) hard class restrictions violating R2; (2) new excitation/hardware; (3) needs oracle states at deployment; (4) Theta-only routing |
| Anti-scope | longer windows; ARTBP and unbiased truncated BPTT; optimiser swaps and lr tuning; zero-mean priors on velocity rows; hard class restrictions |
| Vocabularies searched | control/sysid, machine learning, data assimilation and geophysics, navigation and estimation, econometrics, signal processing, numerical analysis, statistics |
| Evidence floor | every paper named came from a query run in session; repo-sourced items are labelled as such |

**Frame error, recorded so it is not repeated.** The do-not-rediscover list omitted
`docs/multiple-shooting-sweep-2026-07-25.md`, which already holds the DA neutral-subspace cluster
(Grudzien/Carrassi/Bocquet 2018, Bocquet et al. 2017, Fisher et al. ECMWF 2011 full-read including
the `Q^-1` argument, Trevisan et al. 2010 = the `10.1002/qj.571` the frame could not identify).
One agent caught it; the other four returns were deduplicated against that document by hand.
**Rule for next time: the local-holdings list must enumerate EVERY prior sweep doc by filename.**

## 1. Executive answer to the document's Section 8

| Q | Answer | Where |
|-|-|-|
| 1. Constant force or slowly diverging mode? | **Separable, by construction, not by a statistic on the drifted signal.** Orrell et al. 2001 decomposition plus a two-run identical-twin experiment. Runnable on an existing checkpoint. | 3.1, 4 |
| 2. Loss/selector normalisation mismatch | Not a literature question, but Ljung 1999 Section 5 names it ("simulation focus" vs "prediction focus") with measured evidence that the two disagree. Still the cheapest open item. | 3.5 |
| 3. Estimator bias vs model bias | **Separable only under a structural condition.** Laloyaux et al. 2020 give it: the two sources must occupy different scales. M3's evenness in `Y` supplies one. | 3.4 |
| 4. Frequency-weighted objectives | Classical for linear PEM; extended to nonlinear ID by one 3-paper thread; **absent for nonlinear rollout objectives**. One published near-test of the idea came out negative in isolation. | 3.5 |
| 5. Data assimilation, spectral content of model error | Yes, treated. Bonavita 2021: the assumed model-error correlation time silently decides which components the `Q^-1`-weighted defect can see. | 3.6 |
| 6. `arXiv:2406.03760` | **CLOSE IT.** Read in full by two agents. Contains no DC-vs-diverging-mode discriminator. Published as IEEE TAC 70(9), 2025. | 3.3 |

## 2. Sub-questions as fanned out

SQ1 coherence of shooting defects / DA innovations as a model-error statistic.
SQ2 low-frequency-selective objectives for nonlinear rollout training.
SQ3 separating estimator bias from model bias with an operating-point-dependent estimator.
SQ4 constant force vs slowly diverging mode.
SQ5 primary read of `arXiv:2406.03760` plus offset-free detectability conditions.

## 3. Findings

### 3.1 HEADLINE. The decomposition that dissolves the M6/M7 ambiguity

**Orrell, D., Smith, L., Barkmeijer, J., Palmer, T. N.**, "Model error in weather forecasting",
*Nonlinear Processes in Geophysics* 8(6):357-371, 2001. DOI `10.5194/npg-8-357-2001`.
Free: `https://npg.copernicus.org/articles/8/357/2001/npg-8-357-2001.pdf` (diamond OA).
**FULL-READ.**

Their Eq. (5): `e(tau) = M(tau) e(0) + d(tau)`. `M` is the linear propagator acting on
**displacement error** (the document's reading 6.2); `d` is the **drift**, the accumulated
**tendency error** (the document's reading 5).

1. **Short-time shape test.** Non-zero tendency error makes solutions diverge at an initial
   *linear* rate; displacement errors "may have an initial zero or negative growth rate".
   Two integrations on a K=0 axis give `t^2`. `verify_ms_method.py` already measured exponent
   1.993 for that law.
2. **The coherent defect IS Orrell's drift.** They compute it by integrating short forecasts
   started along the target orbit and vector-summing them. Where the nodes sit on the target
   orbit the displacement error is zero, so the coherent statistic is the model-error component
   **by construction**. M7 is therefore not ambiguous between the two readings, provided the
   nodes are clean.
3. **Spin-up trap, and it reinterprets our own M7.** They warn that each short forecast carries a
   spin-up error from initial mismatch, whose signature is step-size dependence of the drift, and
   their guard is scale-invariance in segment length. Our encoder-node column is FLAT
   (1.06 / 1.21 / 1.50x at `n_seg` = 4/12/30) where the true-node column climbs
   (2.34 / 4.34 / 13.80x). Under their criterion that flatness reads as **spin-up contamination**,
   not merely weaker signal. The deployable 1.50x figure is therefore suspect, which is a harsher
   reading than the source document's.
4. **Their Eq. (16)** turns M7's coherence into a falsifiable numeric prediction:
   `d(t) = d_m * sqrt((t/24)(1+2 c_m) - 2 c_m)`, `c_m` = mean cosine between consecutive
   short-forecast error vectors. `c_m -> 0` gives `sqrt(t)`; `c_m -> 1` gives linear in `t`.
   Our measured degraded-model coherence is 0.83. **NOT YET EVALUATED on our numbers.**
5. **Their Eq. (17)** parameterises the cross term (displacement error *generated by* the drift)
   as a separate curve, so "bias seeds an instability" is a third shape, not an unmodelled
   alternative.

### 3.2 The algebraic gate, reached independently by two agents

**Rawlings, J. B., Mayne, D. Q., Diehl, M. M.**, *Model Predictive Control: Theory, Computation,
and Design*, 2nd ed., Nob Hill. **Lemma 1.8** and **Corollary 1.9**. Free PDF at
`sites.engineering.ucsb.edu/~jbraw/mpc/`. **FULL-READ by two agents independently.**
Journal original: **Pannocchia, G., Rawlings, J. B.**, *AIChE Journal* 49(2):426-437, 2003,
DOI `10.1002/aic.690490213`.

> The augmented system is detectable **iff** `(A,C)` is detectable and
> `rank [[I-A, -B_d],[C, C_d]] = n + n_d`. Maximal `n_d` is the number of measurements, `n_d <= p`.

That matrix is the **PBH observability test at `z = 1`**, valid with repeated eigenvalues and
Jordan blocks, so it holds when the plant contributes its own `z=1` modes.

- **The book carries our exact counterexample** (Example 1.11b): a plant whose own level is an
  integrator, where an **output** disturbance is undetectable because the state cannot be
  distinguished from the disturbance added to it. Part (c) fixes it with an **input** disturbance.
  This inverts Kuntz and Rawlings's footnote-1 claim that input vs output is a mere interpretability
  choice: the equivalence holds only among pairs that keep the augmented system observable.
- **`n_d <= p` bites.** With 3 measured positions there are **at most 3** integrating-disturbance
  states. An 8-row ANN correction cannot be represented as an identifiable integrating-disturbance
  model; at most a 3-D projection of it can.
- **Conditioning goes as `1/Ts`.** On placeholder gantry matrices the passing case had
  `cond = 1.13e6` at 4 kHz. Use an explicit scale-aware SVD tolerance, never
  `np.linalg.matrix_rank` defaults.

**CITATION HAZARD: the two agents quote different page numbers (p50 vs p100) because they read
different printings. Pin the printing before citing a page.**

**The numbers above came from PLACEHOLDER gantry matrices, not the project's real ones.**
Re-run before quoting. Scripts were in the session scratchpad and are not preserved.

### 3.3 Corrections from full-text reads (parent, TU/e browser route)

#### 3.3.1 CORRECTION 1: Bageshwar and Borrelli does NOT apply to our case

**Bageshwar, V. L., Borrelli, F.**, "On a Property of a Class of Offset-Free Model Predictive
Controllers", *IEEE TAC* 54(3):663-669, 2009. DOI `10.1109/TAC.2009.2012998`.
**FULL-READ via TU/e browser route, this session.**

SQ5 reported it second-hand from Kuntz and Rawlings's one-sentence characterisation as
"the largest real filter eigenvalue is bounded below by the largest real open-loop eigenvalue,
which for our K=0 axes is exactly 1" and called it the sweep's most important negative transfer.
**That reading is wrong.** The paper's own standing assumptions exclude our plant:

- **Assumption 1**: they consider ONLY the output integrator disturbance model, `n_d = p`,
  `C_d = I_p`, **`B_d = 0`**.
- **Remark 3**, verbatim: "To satisfy condition (9) of Proposition 1, the nominal system cannot
  have integrating modes."
- Their `lambda_max,r` is by construction the largest **stable** real eigenvalue (max over
  `|lambda| < 1` of the real eigenvalues of `A` and their inverses). The theorems place a
  closed-loop estimator pole between that and `z = 1`. A plant with poles AT 1 violates the
  observability condition the framework assumes.

**Their conclusion names the escape, and it is our configuration**, verbatim: "in [19], it was
shown that the closed loop performance of a nominal plant model augmented with an output integrator
disturbance model with correlated process noise vectors was equivalent to the performance of a
nominal plant model augmented with a standard input integrator disturbance model. Therefore, by
relaxing the assumption on uncorrelated state and disturbance process noise vectors, one might ease
or eliminate the restrictions." They also note an H-infinity observer design (their ref [16]) for
which the limitation was not reported.

Consistent with 3.2: on an integrating plant, output disturbances lose detectability and input
disturbances restore it. **Net effect: the offset-free route is LESS blocked than SQ5 reported.**

#### 3.3.2 CORRECTION 2: data prefiltering and error filtering are NOT equivalent for a nonlinear model

**Spinelli, W., Piroddi, L., Lovera, M.**, "On the role of prefiltering in nonlinear system
identification", *IEEE TAC* 50(10):1597-1602, 2005. DOI `10.1109/TAC.2005.856655`.
**FULL-READ via TU/e browser route, this session.** SQ2 had it at snippet level only.

Ljung's linear equivalence (filter the data == filter the prediction error) **breaks** in the
nonlinear case. Filtering the DATA leaves a residual weighting factor on the order-`n` Volterra
kernel, `L(e^{j(w1+..+wn)}) / (L(e^{jw1})...L(e^{jwn}))`, which for any rational low-pass or
band-pass filter with positive relative degree is **high-pass type**, "taking abnormally high
values outside the filter bandwidth for higher order kernels". Damage is not confined outside the
band: the optimiser "will be enticed to consider as numerically significant only the portion of the
identification data with frequency content outside the filter bandwidth, thus producing an unwanted
bias also inside the filter bandwidth". Measured: data-prefiltered LS gave "abnormally high values
for the parameters associated with the quadratic regressors" and recovered only the first-order
GFRF; error-filtered LS stayed consistent.

**Two hard implementation constraints for any DC-emphasis proposal:**
1. Apply the weight to the **residual**, never to the data. On a nonlinear augmented model this is
   the difference between a consistent and an inconsistent estimator.
2. Regressors containing simulated-output terms "must be filtered at each algorithm iteration after
   `y_hat` has been recalculated given the current parameterization". A free-run rollout model is
   exactly that case, so **the filter must live inside the training loop**, reapplied every step.
   It cannot be precomputed on the data.

Transfer caveat: their analysis is one-step prediction error under a Volterra representation, not a
multi-step free-run objective. The data-vs-error distinction transfers cleanly; the frequency
algebra does not automatically.

### 3.4 Estimator bias vs model bias (document Q3)

**Laloyaux, P., Bonavita, M., Chrust, M., Gurol, S.**, "Exploring the potential and limitations of
weak-constraint 4D-Var", *QJRMS* 146(732):4067-4082, 2020. DOI `10.1002/qj.3891`.
**ABSTRACT-VERIFIED only** (Crossref publisher-deposited abstract). Green OA via HAL,
`pdf_url = None`. Verbatim: weak-constraint 4D-Var estimates model errors and the initial state
"only when background and model errors have different spatial scales and when the observations are
unbiased and spatially homogeneous."

**This is the citable justification for building the separator on the Y-dependence.** M3 gives a
structural difference: the encoder defect is monotone and EVEN in `|Y|`
(`corr(|mean Y|, |e_enc|) = +0.995`, slope `0.249/m`, 24x span, symmetric in sign); a constant
residual force is neither.

**Trémolet, Y.**, "Model-error estimation in 4D-Var", *QJRMS* 133(626):1267-1280, 2007,
DOI `10.1002/qj.94`. **ABSTRACT-VERIFIED.** Our failure mode, published: the model-error term
"captures part of the observation bias" and "varies rapidly, and cannot be used to correct medium-
or long-range forecasts". Closest published analogue to "windowed metric improves 14% while the
12 s free run collapses 117x".

**Trémolet, Y.** 2006, *QJRMS* 132(621):2483-2504, DOI `10.1256/qj.05.224`. **ABSTRACT-VERIFIED.**
Adding a model-error control variable makes the problem "essentially an initial-value problem",
i.e. it does not create an independent second degree of freedom. The 4D-state control variable is
the one that behaves differently, and our multiple-shooting defect with encoder nodes IS that
formulation.

**Dee, D. P.** 2005, *QJRMS* 131(613):3323-3343, DOI `10.1256/qj.05.137`. **ABSTRACT-VERIFIED.**
Bias-aware methods "require attribution of a bias to a particular source, and its characterization
in terms of some well-defined set of parameters". A precondition, not a method. Fifth-vocabulary
corroboration of thread-CD's negative.

**Dee, D. P., da Silva, A. M.** 1998, *QJRMS* 124(545):269-295, DOI `10.1002/qj.49712454512`.
**ABSTRACT-VERIFIED.** Estimates forecast bias "based on an unbiased subset of the observing
system", i.e. an external anchor. Derives its scheme as the two-stage separate-bias estimator of
Friedland (1969, 1978) and Ignagni (1981, 1990), which is the bridge to 3.4.1.

**Nóvoa, A., Racca, A., Magri, L.**, "Inferring unknown unknowns: Regularized bias-aware ensemble
Kalman filter", *CMAME* 418:116502, 2024. DOI `10.1016/j.cma.2023.116502`. Free `arXiv:2306.04315`.
**FULL-READ.** The only formulation found in any vocabulary combining bias-aware separation with a
**learned** (echo-state-network) bias estimator. Abstract names our problem: bias-aware methods
"can infer model biases that are not unique for the same model and data". Unregularised, the
framework "may indistinguishably recover solutions with (b) unchanged, (c) increased, or (d)
reduced bias norm". Their penalty `gamma * ||b||^2_{C_bb^-1}` sets `C_bb = C_dd`, so the weight is
**measurable rather than a free hyperparameter** (same principled-weight argument as the `Q^-1`
weighting already held in the MS sweep). Useful window `1 <~ gamma <~ 5`.
Passes the disqualification filter: regularises the bias ESTIMATE, not the model class.
Transfer caveat: sequential ensemble filter with observations at assimilation time; what transfers
to offline batch training is the cost-function form and the `C_bb = C_dd` rule, not the algorithm.

#### 3.4.1 Friedland two-stage chain, and why it is not a knowledge-free separator

Friedland 1969 `10.1109/tac.1969.1099223`; Ignagni 1981 `10.1109/tac.1981.1102697`;
Ignagni 1990 `10.1109/9.50352`; **Alouani et al. 1993 `10.1109/9.233168`** (where the exactness
condition is actually written); Hsieh and Chen 1999 `10.1109/9.739135`;
Keller and Darouach 1999 `10.1016/s0005-1098(98)00194-0`; Ignagni 2000 `10.1109/9.847741`.
All **METADATA-VERIFIED + SEARCH-LEVEL on content**. The exactness conditions are algebraic
constraints on a **specified** bias-coupling matrix, so the family cannot serve as the
knowledge-free separator. Estimator-side only, so no disqualification applies.
**The exactness condition itself is UNVERIFIED. Alouani 1993 is in the browser queue.**

#### 3.4.2 M3 is a documented property of our own encoder, with a named in-framework fix

**ALREADY ON DISK, and unread on this point until now:**
`literature/augmentation/Encoder initialisation methods in the model augmentation setting.pdf`
= **Hoekstra, J. H., Györök, B., Tóth, R., Schoukens, M.**, `arXiv:2602.13108`, submitted IFAC WC
2026. **PRIMARY-READ (targeted grep, 7 pp).**

p4 verbatim: the model-based state reconstruction "will only be accurate around the equilibrium
point `(x*_b, u*)`". Footnote, same page: "Alternatively, the bias terms in (8) may be initialised
with `x*_b`. Utilising these bias terms **also enables linearisation around non-equilibrium
points**, though this requires minor adjustments to the reconstructability map (27) to account for
the resulting affine terms."

Our encoder is initialised from the baseline linearised at `Y_op = 0` only, because
`gantry_linearize_and_discretize` raises `NotImplementedError` for any other value. **M3 is the
predicted consequence of the single-equilibrium initialisation, not an anomaly**, and the footnote
names an in-framework remedy that costs no expressivity and touches nothing else.

Negative, MODERATE confidence: a complete OpenAlex enumeration of the Beintema + Verhoek corpus
(65 works, 3 author IDs unioned) found **no** paper characterising how encoder/initial-state error
scales with operating point in an LPV setting. `arXiv:2602.13108` is the closest thing that exists
and it is qualitative.

### 3.5 Frequency-weighted objectives (document Q4)

**Ljung, L.**, "Estimation focus in system identification: prefiltering, noise models, and
prediction", *38th IEEE CDC*, vol. 3, pp. 2810-2815, 1999. DOI `10.1109/CDC.1999.831359`.
Free: LiU DiVA. **FULL-READ.**
- Prefiltering by `L(q)` makes the asymptotic estimate minimise
  `int |G(e^{iw},theta) - G0(e^{iw})|^2 Phi_u(w) |L(e^{iw})|^2 dw`. The prefilter IS the frequency
  weight on the bias distribution. Our `Phi_u` is dominated by the 150 Hz absorber content, so the
  unweighted objective is already implicitly weighted AWAY from DC. This is the formal backing for
  the source document's Section 5 argument.
- Prefilter/noise-model equivalence and its trap: with an adjustable noise model, `theta` fights
  the prefilter and the realised weighting is "far from the desired `L`". His PEM(f) construction
  is the fix.
- **Section 5 names our Q2 in classical terms**: "simulation focus" vs "prediction focus", with
  measured glass-furnace evidence that models fit under one are worse under the other, and
  "A (much) better fit on estimation data is no guarantee for the performance on validation data."
- **LAST RESORT flag**: his Section 2 reason 2 for prefiltering is differencing / high-pass for a
  drifting output level. That is the forbidden transform (`velocity-loss-is-last-resort` in
  `tasks/lessons.md`). Report, do not adopt. Note it is one sentence in a paper whose framework is
  direction-neutral.

**Spinelli/Piroddi/Lovera 2005**: see the correction in 3.3.2. FULL-READ this session.

**Piroddi, L., Lovera, M.**, "NARX model identification with error filtering", *IFAC Proceedings
Volumes* 41(2):2726-2731, 2008. DOI `10.3182/20080706-5-KR-1001.00459`. **ABSTRACT ONLY**,
`needs-browser-route`. Abstract states the mechanism in our words: accuracy problems can be
"circumvented by focusing the identification process on the obtainment of an accurate local model
over a specific frequency range". **If their filtering acts on the SIMULATION error rather than the
one-step error, this is a direct precedent for the proposal. That is the open question.**

**Lovera, Piroddi, Spinelli 2006**, `10.3182/20060329-3-AU-2901.00161`. METADATA ONLY.

**Forgione, M., Piga, D.**, `arXiv:1911.13034`. **FULL-READ.** The canonical enumeration of fitting
criteria for neural state-space ID (one-step, full simulation, and a regularised multi-step
criterion with jointly estimated initial conditions = our defect term). A full-text grep for
`filter` and `frequency` returns **ZERO** occurrences in 17 pp. Frequency weighting is not in that
literature's design space. This is the deliverable negative for SQ2.

**Chattopadhyay, A., et al.** (FouRKS), `arXiv:2304.07029`. **FULL-READ of load-bearing sections.**
**The closest published test of the source document's Section 5 falsification condition, and it
came out NEGATIVE for the band term in isolation.** Their loss adds a wavenumber-band-restricted
Fourier-coefficient mismatch (`lambda = 0.8`, `k_T = 30/40`, "after significant trial and error").
Their own ablation, Fig. 6(b): "the spectral regularizer does not diminish the error growth at all."
And: "the regularizer only assists in capturing the small scales in a single time step of
prediction. The effect of this regularizer on long-term autoregressive error propagation has not
been studied." They needed a differentiable RK4 layer plus an inference-time spectrum correction.
**Not decisive for us**: their band is spatial wavenumber, their target band is the opposite end,
and the regulariser acts on a single step.

**Chakraborty, D., Mohan, A., Maulik, R.**, "Binned spectral power loss for improved prediction of
chaotic systems", *J. Comput. Phys.* 558:114866, 2026. DOI `10.1016/j.jcp.2026.114866`.
Free `arXiv:2502.00472`. **READ (targeted grep).** The transferable construct: a per-band **ratio**
loss `(1 - (E_F^bin + eps)/(E_G^bin + eps))^2`, evaluated at each autoregressive rollout step.
Their gradient analysis: the ratio "leads to equal importance to all ranges of the energy spectrum",
whereas plain MSE follows "the components that have higher values in the Fourier series
representation". **That is the source document's Section 5 argument with a published remedy:
normalise per band so magnitude weighting disappears.** Direction-agnostic, so it transfers even
though our low-power band is the opposite one. Limits: spatial frequency axis, chaotic PDEs,
periodic domain.

Adjacent, ABSTRACT ONLY: Sen and Maulik `arXiv:2607.19387` (graph-Laplacian frequency bands, so a
band projector built from the system's own operator works without an FFT-able domain);
Hu et al. `arXiv:2407.01598` (spectral bias as the primary instability contributor; remedy is
architectural, classification (c)); Luo `arXiv:2506.10711` ("release the constraint that each
frequency weighs the same").

### 3.6 Defect and innovation coherence as a model-error statistic (SQ1)

**Hooker, G., Ellner, S. P.**, "Goodness of fit in nonlinear dynamics: Misspecified rates or
misspecified states?", *Ann. Appl. Stat.* 9(2):754-776, 2015. DOI `10.1214/15-AOAS828`.
Free `arXiv:1312.0294`. **FULL-READ.**
They add a time-varying forcing `g(t)` to an ODE (the continuous-time analogue of our defect) and
classify its STRUCTURE, never its size, into three cases: unmodelled random disturbance;
misspecified rate function `f`; **misspecified state vector**.
- Case 1 vs 2: `H0: E(g(t)|x(t)) = 0` by block permutation with a residual bootstrap.
  **M7's `|mean d| / rms d` is a scalar special case of this null**, testing only the unconditional
  mean.
- Case 2 vs 3, the missing-state test: an omitted state implies `dg/dt = l(x,g)`; testing `dg/dt`
  is unstable, so they "test for dependence of `g(t)` on `g(t - delta)`... if `g(t)` is just a
  function of `x(t)`, past values of `g` provide no additional information about its present value."
- **Our unmodelled truth is an MSD absorber, i.e. literally their case 3.** Needs node states, not
  oracle states, so it survives disqualification 3 where M7's 13.80x does not.
- Their stated limitation: if `g` is too low dimensional, a case-2 misspecification can appear as
  case 3.
- **Complete 15-citer forward sweep: every citer is biostatistics, ecology or mathematical
  statistics. Zero control, zero sysid, zero ML.**

Predecessor, **METADATA ONLY, `needs-browser-route`**: Hooker 2009, *Biometrics* 65(3):928-936,
DOI `10.1111/j.1541-0420.2008.01172.x`. The 2015 hierarchy assumes this test has already fired.

**Bonavita, M.** 2021, *QJRMS*, DOI `10.1002/qj.4137`, free `arXiv:2105.09776`. **FULL-READ.**
Introduces the Lagged Analysis Increment Covariance diagnostic. Key transfer: the cycled
weak-constraint `Q` "induces a decorrelation time of about one to two weeks in the model error
estimates ... which filters out the diurnal variability". **The assumed model-error correlation
time silently determines which components of model error the `Q^-1`-weighted defect penalty can
see.** He fixes it by changing the cycling, not the magnitude. Also reports the free forecast
"quickly revert[s] back to the model (biased) attractor".

**Mojgani, R., Chattopadhyay, A., Hassanzadeh, P.** 2024, *JAMES* 16(3), DOI `10.1029/2023MS004033`,
free `arXiv:2309.13211`. **FULL-READ.** M6 measured independently in another field:
networks at `R^2 ~ 0.99` and 0.25% relative RMSE produced total structural-discovery failure; and
"short-horizon DA increments may not be able to capture some structural errors, although such
errors may have significant long-term effects".

Innovation-diagnostics lineage, **ABSTRACT or METADATA ONLY**: Daley 1992 (lagged innovation
covariance as a suboptimality diagnostic independent of innovation variance,
`10.1175/1520-0493(1992)120<0178:TLICAP>2.0.CO;2`); Desroziers et al. 2005 `10.1256/qj.05.108`
(includes a spectral interpretation and a scale-separation argument between background and
observation errors, which is the DA form of our Q3); Dee 1995; Mehra 1970 `10.1109/TAC.1970.1099422`
(the only item where innovation correlation is an **estimation objective**, matching `Q` and `R` to
the innovation autocorrelation sequence); Billings and Voon 1986 `10.1080/00207178608933593`
(higher-order correlation tests detecting unmodelled nonlinearity that plain whiteness passes; but
one-step residuals, not rollout).

Farchi, Bocquet, Laloyaux, Bonavita, Chrust cluster (`arXiv:2107.11114`, `2210.13817`, `2403.03702`),
ABSTRACT ONLY: the operational-scale version of "learn a network from the defects of a physics
model", online inside incremental 4D-Var. Worth a separate look independent of this sweep.

### 3.7 Constant force vs diverging mode, remaining routes (SQ4)

**Econometrics door is closed, with a reason.** The right-tailed unit-root family (Phillips, Wu and
Yu 2011 `10.1111/j.1468-2354.2010.00625.x`; Phillips, Shi and Yu 2014 `10.1111/obes.12026` and 2015
`10.1111/iere.12132`) does test drift against mild explosiveness, but its usable limit distribution
requires the drift to be **assumed asymptotically negligible** via a localizing coefficient `eta`
(`y_t = d T^{-eta} + theta y_{t-1} + e_t`). On a K=0 axis with `f t^2 / 2` it is not.
Specification verified in primary text via **Caspi, I.**, *J. Stat. Software* 81(CS1), 2017,
DOI `10.18637/jss.v081.c01` (**FULL-READ**, diamond OA); PSY 2014 itself is SNIPPET ONLY.
This refines rather than duplicates thread-CD's Thread D.

**Near-zero Lyapunov exponent is the documented hard case.** Parlitz 1992
`10.1142/S0218127492000148` (snippet: spectra "overlap near zero"); Zeng, Eykholt and Pielke 1991
`10.1103/PhysRevLett.66.3229`; Lu and Smith 1997 (bias and variance of local-Lyapunov estimators,
no DOI resolved). All SNIPPET or ABSTRACT level.

**The perturbation test has a name, two in fact**: identical-twin / perfect-model experiment
(Simmons and Hollingsworth 2002 `10.1256/003590002321042135`), twin OSSE used explicitly to
apportion model vs initial-condition error (Privé and Errico 2013 `10.3402/tellusa.v65i0.21740`),
and breeding / bred vectors (Toth and Kalnay 1997). All SNIPPET level, all `needs-browser-route`.

**Nearest quantitative modern version, ABSTRACT ONLY, high value if the full text supports it**:
Li, Feng, Ding, Li 2021 `10.1007/s00376-021-0434-2` and Li, Ding, Li 2020
`10.1016/j.chaos.2020.110094`, quantifying the RELATIVE contribution of initial-condition error vs
model error to finite-time error growth via nonlinear local Lyapunov exponents.

**ML side is a gap.** No ML paper found that attributes rollout divergence between bias and an
unstable Jacobian. Instability DETECTORS exist (eigenvalue-based Jacobian diagnostics). Confidence
WEAK: 3 arXiv boolean queries at 0 totals plus 1 Scholar query, no PMLR/NeurIPS enumeration.

### 3.8 Parity and soft symmetry

**Finzi, M., Benton, G., Wilson, A. G.**, "Residual Pathway Priors for Soft Equivariance
Constraints", *NeurIPS 2021*, `arXiv:2112.01388`. **ABSTRACT-VERIFIED.**
Converts hard architectural constraints into soft priors with a free residual pathway, and is
"resilient to approximate or misspecified symmetries", "as effective as fully constrained models
even when symmetries are exact". **This resolves the R2 objection recorded in
`literature/stability-training/claude-deep-research-drift.md` line 57**, which identified
translation-equivariance as the theoretically correct structural cure but rejected it as a genuine
class restriction. Finzi is in the repo only as LieConv (`arXiv:2002.12880`).

Negative, MODERATE: no work found using a parity or symmetry test to **attribute** a coherent
residual to the estimator rather than the model. Given M3's measured evenness in the sign of `Y`,
an even/odd-in-`Y` split of the **defect** (not of the residual force) appears unclaimed.
This closes thread-CD's flagged novelty search item 2, provisionally.

## 4. Recommended next actions, in order

1. **Run the Lemma 1.8 rank test on the real matrices.** Build
   `M = [[I - A_d, -B_d],[C, C_d]]` from `gantry_linearize_and_discretize`, `B_d` = the ANN's
   routing columns into the velocity rows, `C_d = 0`. SVD; require `rank = n + n_d` AND report
   `sigma_min/sigma_max` with an explicit scale-aware tolerance. Training-free. Gates everything
   downstream: a failure is a theorem-strength "not distinguishable from this data".
2. **Run the Orrell two-run discriminator** on `gantry_drift_71167_last.pth`.
   (a) Free-run from TRUE `x0` so `e(0) = 0`, fit `log||e||` vs `log t`, expect exponent 2 on the
   K=0 rows if reading 5 holds. (b) Free-run twice from `x0` and `x0 + alpha*delta` with identical
   inputs, which cancels `d(tau)` exactly; a positive slope of `log||Delta(t)||` is a finite-time
   Lyapunov exponent and confirms reading 6.2. (c) Repeat at `alpha/2`, `2*alpha` to bound the
   linear regime. (d) **Orrell scale-invariance guard**: recompute the drift over a fixed span at
   two different segment lengths; disagreement means the encoder re-anchoring contaminates the
   number and no defect statistic at that segment length is interpretable.
3. **Evaluate Orrell Eq. (16)** with our measured `c_m = 0.83` and check it against the observed
   accumulation. Cheap, and it either supports or kills the coherent-defect reading.
4. **Fix the encoder initialisation** per `arXiv:2602.13108` p4 footnote: carry `x*_b` in the
   encoder bias terms so the reconstructability map is valid away from `Y = 0`. In-framework, no
   expressivity cost. Directly targets M3.
5. **Implement the Hooker and Ellner lagged-dependence test** on the defect sequence. Strictly
   stronger than M7's coherence and it does not need oracle states.
6. Only then consider any frequency-weighted objective, and if so: weight the RESIDUAL not the
   data, inside the training loop, per 3.3.2; and expect the FouRKS negative (3.5) to apply until
   shown otherwise.

## 5. Access status

**TU/e browser route: AVAILABLE and VERIFIED** in the parent session on
`10.1109/tac.2009.2034199` (closed IEEE TAC; all six sections returned, no
"Sign in to Continue Reading" tell). Two further items were fetched through it this session
(3.3.1, 3.3.2), and both CORRECTED an agent-level claim.

Sub-agents were instructed not to preflight (skill rule: one preflight in the parent), so every
item below is `unreachable - browser route not attempted`, never a bare "unreachable".

**`needs-browser-route` queue, ranked, NOT yet fetched:**

1. **Laloyaux 2020** `10.1002/qj.3891` (Wiley) - the twin-experiment section behind the
   identifiability condition that action 1 and the Q3 separator rest on. Currently abstract-only.
2. **Piroddi and Lovera 2008** `10.3182/20080706-5-KR-1001.00459` (Elsevier) - decides whether
   their filtering acts on the SIMULATION error, i.e. whether it is a direct precedent.
3. **Hooker 2009** `10.1111/j.1541-0420.2008.01172.x` (Biometrics) - prerequisite test for 3.6.
4. **Alouani et al. 1993** `10.1109/9.233168` - where the two-stage exactness condition is written.
5. **Li, Ding, Li 2020** `10.1016/j.chaos.2020.110094` and **Li et al. 2021**
   `10.1007/s00376-021-0434-2` - quantitative version of the Orrell attribution.
6. Then: Trémolet 2006 and 2007, Dee 2005, Dee and da Silva 1998, Desroziers 2005, Daley 1992,
   Billings and Voon 1986, Pannocchia/Gabiccini/Artoni 2015 `10.1016/j.ifacol.2015.11.304`
   (independently treats a plant with two integrators), PSY 2014, Toth and Kalnay 1997,
   Bageshwar and Borrelli ref [19] (the correlated-noise equivalence named in 3.3.1).

## 6. Evidence quality

**FULL-READ, quotes verified against extracted or rendered text:** Orrell 2001; Rawlings/Mayne/Diehl
Lemma 1.8 and Cor. 1.9 (two independent reads); Kuntz and Rawlings `arXiv:2406.03760` (two
independent reads, 46 pp); Bageshwar and Borrelli 2009 (parent, browser); Spinelli et al. 2005
(parent, browser); Hooker and Ellner 2015; Bonavita 2021; Mojgani 2024; Nóvoa 2024;
Hoekstra `arXiv:2602.13108`; Ljung CDC 1999; Forgione and Piga 2020; Chattopadhyay FouRKS;
Chakraborty BSP; Caspi 2017.

**ABSTRACT-VERIFIED only** (Crossref publisher-deposited abstract, quoted verbatim, body unread):
Laloyaux 2020; Trémolet 2006 and 2007; Dee 2005; Dee and da Silva 1998; Finzi RPP;
Desroziers 2005; Daley 1992.

**SNIPPET or METADATA ONLY, do not cite internals:** Piroddi and Lovera 2008; Lovera et al. 2006;
the Friedland/Ignagni/Alouani/Hsieh/Keller chain; Zeng 1991; Parlitz 1992; Lu and Smith 1997;
PWY 2011, PSY 2014 and 2015; Toth and Kalnay 1997; Privé and Errico 2013; Simmons and
Hollingsworth 2002; both Li/Ding/Li papers; Beintema CT-SUBNET; Billings and Voon 1986;
Mehra 1970; Dee 1995; Hooker 2009; Homm and Breitung 2011.

**DERIVED BY AGENTS, not from any source, flagged as such:** the mapping of Orrell's drift onto M7;
the spin-up reading of M7's encoder-node column; the application of Lemma 1.8 to our K=0 axes; the
step-by-step procedure in Section 4; the rank table, which used **placeholder gantry matrices, not
the project's real ones**.

## 7. Novelty position, with vocabularies (lessons rule 117)

Vocabularies searched across the sweep: control/sysid, machine learning, data assimilation and
geophysics, navigation and estimation, econometrics, signal processing, numerical analysis,
statistics.

**Do NOT claim as new:**
- Soft equivariance as an escape from a hard class restriction: published as RPP (3.8).
- Bias-aware separation with a learned bias model: published as r-EnKF (3.4).
- The multiple-shooting defect as a constraint, and offset-free detectability (3.2).

**Unreported per this sweep, each with its grade:**
1. **Frequency-weighted objectives for nonlinear ROLLOUT identification.** MEDIUM-HIGH.
   11 zero-total arXiv abstract searches; a full-text zero on the canonical fitting-criteria paper;
   all 29 forward citers of Spinelli 2005 enumerated; a dblp title zero. Weakness: arXiv covers
   control badly, and CDC/ECC/ACC 2023-2026 were not reached at all.
2. **A DC / zero-frequency-selective training objective.** MEDIUM. Four zero-total arXiv searches
   plus one Scholar sentence query at 0/12. Three Scholar calls returned `[]` from rate limiting
   and Scholar is the only full-text route, so treat as provisional.
3. **The lagged-dependence-of-the-residual test carried into control or sysid.**
   MODERATE-TO-STRONG. Complete 15-citer forward sweep, all outside control.
4. **Parity or symmetry used to ATTRIBUTE a coherent residual to the estimator vs the model.**
   MODERATE.
5. **Variance-ratio machinery applied to shooting defects or DA innovations.** The `1/sqrt(n)`
   vs flat-coherence signature is a random-walk-vs-drift statistic;
   `abs:"variance ratio" AND abs:"innovations"` = 1 on arXiv, and it is a finance paper. The null
   distribution and power theory (Lo and MacKinlay 1988, `10.1093/rfs/1.1.41`) is unused here.
   WEAK as a negative: arXiv indexes neither *Rev. Financial Studies* nor *QJRMS* nor *MWR*.
6. **Multiple-shooting defect statistics as model VALIDATION.** `"multiple shooting" AND "defect"
   AND "model error"` = 0 and `"multiple shooting" AND "model validation"` = 0 on arXiv.
   PROVISIONAL: that tradition publishes in SIAM/IFAC venues arXiv does not cover, and no
   IFAC venue-year enumeration was run.

## 8. Research log

**Volume.** ~110 queries across 5 agents plus the parent. arXiv raw API ~70 (primary route);
OpenAlex 19 of a 40 budget, every parse guarded with `assert 'error' not in d`, **no 429 in any
agent**; dblp 2 of 10 (four agents declined it deliberately and correctly, because their questions
were analytical PROPERTIES, which dblp cannot match); Crossref ~45 at 100% resolution;
Google Scholar ~20.

**What worked.**
- **Google Scholar written as the SENTENCE the paper would contain** was the sole source of the
  headline find in three of five sub-questions: Orrell 2001, the Milan prefiltering thread, and
  Hooker and Ellner at rank 1. No enumeration route reached any of them.
- **Crossref `works/<DOI>` for the publisher-deposited ABSTRACT** turned five closed Wiley/QJRMS
  papers into verbatim-quotable findings with zero paywall interaction, 5 of 5.
- **Forward-citation traversal on OLD seeds** as a NOVELTY instrument: 29 citers of Spinelli 2005,
  15 of Hooker and Ellner, both complete.
- **Grepping a downloaded PDF for a WORD CLASS** (`filter`, `frequency`) to produce a verified
  negative about a paper's design space.
- **The local grep before any novelty claim** converted what would have been a rediscovery
  (`arXiv:2602.13108`, already on disk) into the sweep's most actionable item.

**What failed, and it cost real turns.**
- The skill's arXiv snippet uses `http://export.arxiv.org`, which returns **HTTP 301 with a
  zero-byte body**, indistinguishable from the rate transient the skill says to sleep-and-retry on.
  Three agents hit it independently. `https` plus `-L` fixes it.
- `download_with_fallback` returned a correctly-named PDF of an **entirely unrelated biomedical
  paper** for an IEEE TAC DOI.
- A guessed tech-report path returned HTTP 200 and a valid 168 kB PDF that was a paper about
  GNU Octave.
- AMS (`journals.ametsoc.org`) bot-walls curl (202) and WebFetch (403). Cowles serves 1.14 MB of
  HTML that looks like a successful PDF fetch by status and size alone.
- arXiv 429s have a 0-byte body; back off 60 s, not 3 s.
- `abs:"<single word>"` on arXiv is stemmed, not a phrase match: `abs:"whiteness"` returned
  white-box adversarial attacks and white-matter MRI, 92 hits, 0 on target.

**Coverage gaps.**
- **CDC/ECC/ACC 2023-2026 not reached at all.** The most likely hiding place for a recent
  control-side hit, and the venue class dblp is the only route to.
- IFAC-PapersOnLine 2025 reached ~15% of one year.
- Errors-in-variables (SQ3 sub-part 4) is effectively unsearched; an arXiv query that should have
  hit returned an implausible 0 and was not re-run.
- **Aeroelastic flutter-margin prediction was identified as the right fourth vocabulary for SQ4 and
  never searched.** It is the engineering discipline built entirely around "is this lightly damped
  mode about to cross zero, judged from short subcritical records". Highest-value unexplored
  vocabulary in this sweep.
- No forward-citation traversal on any 2024-2026 seed (frontier indexing lag).

**Suggested skill fixes**, all measured this run, to be folded into
`.claude/skills/deep-research/SKILL.md`:
1. Change the arXiv endpoint to `https://export.arxiv.org/api/query` with `-L`, and add the 301
   row to the failure table.
2. Promote Crossref from "metadata fallback" to **the primary route to a closed paper's
   load-bearing sentence**, tried BEFORE marking anything `needs-browser-route`.
3. After downloading any PDF from a guessed or constructed URL, extract page 1 and confirm the
   title before using it. HTTP 200 plus a valid `%PDF` header plus a plausible size is not evidence
   you got the right document.
4. Add AMS to the bot-wall list; add Copernicus (`*.copernicus.org`) to the resolution order as
   first-class diamond OA (it delivered this sweep's headline finding first try).
5. `-w "\nHTTP %{http_code}"` cannot be combined with a pipe into `json.load`.
6. dblp HTTP 500 is a third failure mode and it still consumes the budget.
7. Note that `abs:"<single word>"` is stemmed on arXiv, not a phrase match.
8. **The local-holdings step must enumerate EVERY prior sweep document by filename**, not just
   `literature/` and `docs/`. See the frame error in Section 0.
