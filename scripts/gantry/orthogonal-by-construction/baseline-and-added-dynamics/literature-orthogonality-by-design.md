# Literature: enforcing orthogonality between the baseline and the addition

Two `deep-research` agents, 2026-09-18, on the question this folder's derivation raised: the
parameter bias under orthogonal-by-construction augmentation is `J^+ Delta*`, so recovery needs
`J^T Delta* = 0`, and that is a property of the plant and the data rather than of the estimator.
Who has made it **hold**, rather than measured it or penalised it?

Companion artefacts: `baseline-and-added-dynamics.md` (the derivation and the measurement),
`GPT-RESEARCH-PROMPT-orthogonality-by-design.md` (an independent sweep on a different stack).
Cite keys and access routes for everything below are in `docs/references.md`.

## 1. The headline: our derivation is the seed paper's theorem, and its open problem is ours

`gyorok2026obc` (Györök, Schoukens, Péni, Tóth, *IFAC J. Systems and Control* 35:100376, 2026,
arXiv:2511.01321, held) already contains what this folder derived independently:

| ours | theirs |
|-|-|
| `J^T Delta* = 0` | Condition 4, Eq. (17), the empirical form; Eq. (16) is its domain-integral parent |
| `dtheta = J^+ Delta*` | Eq. (21) |
| recovery iff the condition holds | Theorem 7 |
| (not derived here) | Theorem 16 consistency, Theorem 18 zero covariance |

Their setting is linear-in-parameters input-output; ours is nonlinear-in-parameters LPV
state-space with an encoder. **Their Conclusion names ours as the open case:** "extending the
approach for augmenting baseline models in state-space form, where the general assumption of no
available full-state measurement complicates the orthogonal projection of the learning component,
thus requiring careful investigation." Author-corpus enumeration confirms it is still open in
their own group: the two 2026 state-space and LFR follow-ups do well-posedness, contraction-based
stability and group-lasso order selection, and **neither does orthogonality or parameter
recovery**.

## 2. The most important finding: the seed's own construction breaks on dynamic discrepancies

`gyorok2026obc` Sect. 5.1 gets exact orthogonality on a static example by appending the input
multiplied by `-1`, so even and odd nonlinearities separate. Sect. 5.2 applies the same trick to a
mass-spring-damper and then **concedes, in the body, that it fails**:

> As `y_{k-2}` and `y_{k-1}` also affect `y_k`, this experiment design does not fully guarantee
> orthogonality between the unmodeled dynamics and the baseline model. To achieve that, a more
> concise data acquisition process would be required. However, the deviation from orthogonality is
> minimal...

So the sign-flip construction **breaks as soon as the regressor contains past outputs**, which is
every dynamic system including ours, and the authors say the fix is unknown. This is the single
most valuable clause found in the sweep, it cost zero queries, and it means the question is the
seed paper's own stated gap rather than a gap we invented.

## 3. Route (a), excitation design: the mechanism is classical, the connection is not

`schoukens2016bla` (Schoukens, Vaes, Pintelon, *IEEE Control Systems Magazine* 36(3):38-69, 2016,
arXiv:1804.09587, **already held**) is the canonical statement of the mechanism Györök's Sect. 5.1
uses without attribution:

- p11: a multisine exciting only **odd** lines makes **even** nonlinearities appear only at even
  lines and odd nonlinearities only at odd lines. Unexcited odd lines become **detection lines**.
- p30: "it is still possible to partially eliminate their impact by using an **odd excitation**.
  This can be either a random noise source with a **symmetric amplitude distribution** (for
  example, zero-mean Gaussian noise) or a well-designed multisine."
- Measured payoff on a hot-air setup: about 20 dB standard-deviation reduction against random
  excitation, a factor 100 in measurement time.

**The connection worth making, and it is ours to make.** Györök's `u` against `-u` append is the
time-domain, N-point special case of "symmetric amplitude distribution kills even nonlinearity".
The odd-multisine version is the exact, per-realisation, frequency-domain version, and by Parseval
it makes `J^T Delta*` vanish **line by line rather than only in the N to infinity average**.
Crucially the odd-multisine argument is a statement about the **spectral support of the
distortion**, not about the regressor containing only `u`, so it does not obviously suffer the
Sect. 5.2 breakdown above. Nobody in the located literature has made that link.

`liu2025spacefilling` (Liu, Kiss, Tóth, M. Schoukens, *IEEE L-CSS* 9:1868-1873, 2025, newly
fetched) is the negative control: the seed group's own state of the art in nonlinear input design
optimises **coverage**, not orthogonality. No discrepancy term, no baseline regressor, no inner
product. Its own framing, "no systematic approach exists for nonlinear systems", is the citable
evidence that orthogonality-targeted input design has not been done by the people who posed the
condition.

`dreef2022excitation` (held) is the closest control-side "design the experiment given that part of
the model is known" result, and it is disqualified: the fixed module is a known LTI transfer
function and the criterion is generic rank, not a vanishing inner product with an unknown
nonlinear discrepancy.

## 4. Route (b), structural design: the precedent exists, one field over

`plumlee2018orthogonalgp` (Plumlee and Joseph, *Statistica Sinica* 28(2):601-619) is route (b)
verbatim in **statistics** vocabulary: the *discrepancy model class itself* is built so its
realisations cannot overlap the physics basis, rather than the estimate being projected or
penalised. Both agents found it independently. Two things follow.

**It is on disk and the record was wrong.** `docs/references.md` had it as "No local PDF",
abstract-only. Corrected 2026-09-18.

**Its p6 criticism is a 2018 statement of our own risk.** It faults the restricted-spatial-
regression line (Reich 2006, Hodges and Reich 2010, Hughes and Haran 2013, Hanks 2015, a
vocabulary that appears nowhere else in this repo) for achieving orthogonality "only at the
observed locations", which "induces two negatives: (i) the stochastic model has a dependency on
the observation locations and (ii) outside of the observed locations there is no orthogonality".
That is exactly the Eq. (16) against Eq. (17) gap, and exactly our exposure to a projector built
on a fixed reference set. D3 applies: the mean model is linear-in-parameters and orthogonality is
imposed through the covariance kernel, which has no obvious ANN analogue.

`manna2026orthkernel` (held) is the realisation of the **domain-integral** version, via the
Plumlee-Joseph kernel on a SINDy dictionary. `plumlee2017inexact` (held) is the prior-side
version, and the existing OBC review cites it only for its Theorem 1, the estimator reading, not
for the construct-the-discrepancy-class reading.

`kon2026unconstrained` (Kon, Tóth, van de Wijdeven, Heertjes, Oomen, *IEEE TAC* 71(3):1660-1675,
2026, newly fetched from TU/e Pure) is the methodological template for the plant side: a
**property** of the learned block, quadratic stability or dissipativity, is turned into an
**unconstrained parametrisation** through a Cayley transform, "enabling the use of neural network
coefficient functions". It does not do orthogonality (the word does not occur in the paper), and
its only lossless content is a citation to Peeters, Hanzon and Olivi 2007 on canonical lossless
state-space systems. `martinelli2023dissipative` (arXiv:2304.02976, metadata only, NOT read) is
the continuous-time neural analogue and the natural place to look for whether a **lossless**,
zero-dissipation restriction is parametrisable.

## 5. Route (a) past the Taylor fallback, and what the statistics field actually proves

`kuppa2026embedded` (Kuppa, Sargsyan, Panesi, Najm, arXiv:2602.17923, newly fetched) is the only
paper found that goes past the Taylor-around-`theta_bar` fallback. Its **ROGP** constraint, Eq.
(31), is derived with no linearisation as the exact stationarity condition of an L2-projection
definition of the parameter, and the authors state the constraints are "non-linear in `delta_w`".
Its **LOGP** constraint, Eqs. (27-28), is the linearised version but weighted by the sensitivity of
the output to the embedded discrepancy, which is our routing factor appearing explicitly inside the
orthogonality condition. Disqualified as a construction by D2 (penalty-enforced with a trade-off)
and D4 (static PDE discrepancy, no rollout, no encoder).

**And the framing point that matters most for the thesis.** In statistics nobody proves "recovery
iff orthogonal", because they **define the target by the orthogonality**. `tuowu2015calibration`
Eq. (2.2) states outright that the Kennedy-O'Hagan "true" calibration parameter is unidentifiable
and redefines it as `theta* := argmin ||zeta - y_s(., theta)||_{L2}`; the orthogonality is then a
first-order condition holding **automatically** at `theta*`, for a fully nonlinear-in-theta
simulator. `xiexu2020projected` gives the same closed form, which is Eq. (21) in the domain inner
product instead of the empirical one.

So the whole calibration corpus proves consistency for a **redefined estimand**. Györök's
Condition 4 is the strictly stronger object, because `theta*` is the parameter of a real
data-generating system and orthogonality therefore has to be **engineered**.
`brynjarsdottir2014discrepancy` is the negative counterpart: `theta` and `delta` are not separately
identifiable even with infinite data, so *some* restriction on the discrepancy is necessary, which
is what makes Condition 4 a condition rather than a technicality.

## 6. The multi-step and closed-loop objective: named in print, with the right object

Our caveat 1 in `baseline-and-added-dynamics.md` was that the training objective is a multi-step
closed-loop rollout while the projection orthogonalises in a one-step inner product. Two findings.

**`shan2026semiparametric` (Shan and Liu, SSRN 6275318, 2026, already on disk, never quoted)** says
it outright: "Due to the highly nonlinear coupling induced by the ODE solution operator,
**standard semiparametric orthogonalization fails**", because "simple orthogonalization methods
typical of semiparametric regression fail because they do not account for how perturbations in `g`
propagate through the integrated trajectory to confound `alpha`". Their efficient-information
object is the Gauss-Newton Schur complement `I_eff = I_aa - I_ag I_gg^-1 I_ga`, built from
trajectory Jacobians through a **fixed-step Runge-Kutta** integrator. **That is the multi-step
analogue of `J^T J`, computable with our exact discretisation**, and the right replacement for a
hand-waved caveat. They also get identifiability from **multi-trajectory design** with dispersed
initial conditions, of which our Y-sweep is the analogue. Their own future work leaves a
Neyman-orthogonal moment for the ODE setting open, needing second-order ODE sensitivity operators.

**`donati2025offwhite` does not say what the seed's Remark 9 implies.** Read for that claim: their
Theorem 2 bounds the parameter error by the residual the black model fails to compensate, and
Remark 8 says the bound **tightens as the black box compensates better**. That is the opposite
incentive to Eq. (21). The two are consistent only because their black model is a sparse dictionary
that cannot span the regressor; nothing in the theorem enforces it. Their Appendix D, which would
say whether `M_Delta` contains the rollout sensitivity Jacobian, was **not read**.

## 7. The lossless hypothesis is unstated, and the caveat on it is real

The claim derived in this folder, that a dissipative addition always has a strictly positive
time-averaged correlation with the damping-parameter sensitivities so exact orthogonality demands a
lossless addition, returned **nothing** in any vocabulary. Genuine zeros, non-zero-byte bodies:

| arXiv abstract search | total |
|-|-|
| `"lossless" AND "physics-based model" AND "learning"` | 0 |
| `"input design" AND "unmodeled dynamics" AND "orthogonal"` | 0 |
| `"excitation" AND "orthogonal" AND "grey-box"` | 0 |
| `"Neyman orthogonality" AND "dynamical system"` | 0 |
| `"port-Hamiltonian" AND "model augmentation"` | 1, off-target |
| OpenAlex `"model augmentation" "experiment design"` | 0 |

**The caveat to apply before building on it.** `J^T Delta* = 0` is an inner product over a finite
sampled trajectory, whereas "dissipative" is a statement about a supply-rate integral. The two
coincide only when the baseline sensitivity for a damping parameter is proportional to the velocity
signal the supply rate pairs with. For the gantry's damping parameters that is plausibly true; for
the mass and stiffness sensitivities it is not. So **lossless is at best necessary for the damping
block of `J`, not sufficient for all of it**, which is exactly what the parity table in
`baseline-and-added-dynamics.md` Sect. 8 already implies and what the resonance sign-change
argument is there to handle.

## 8. Coverage gaps that bound every negative claim above

Both agents spent **zero dblp queries**, so **CDC, ECC, ACC and IFAC-SYSID 2023-2026 are entirely
unreached**. OpenAlex holds no source records for those venues, so they are invisible to every
other route used. Optimal input design publishes heavily there, and it is the most likely place for
a counterexample to Sect. 3. Every novelty claim here is therefore **provisional against the recent
conference literature**, graded medium rather than strong.

Also unreached: forward citations of the seed and of `kon2026unconstrained` (frontier, structurally
zero); an IFAC-PapersOnLine 2026 venue-year enumeration; Donati's Appendix D; and the **design of
experiments** literature on orthogonal designs proper. That last one is a live lead: Györök's
`u` against `-u` construction is literally a **foldover design** under a sixty-year-old name, and
nobody chased that thread. It is the first query to run next.

Google Scholar returned 0 on-target across 7 sentence-form queries in both agents, against the
skill's recorded expectation that it is often the sole source. The concept has no settled
cross-field name, so full-text snippet matching had nothing to latch onto.
