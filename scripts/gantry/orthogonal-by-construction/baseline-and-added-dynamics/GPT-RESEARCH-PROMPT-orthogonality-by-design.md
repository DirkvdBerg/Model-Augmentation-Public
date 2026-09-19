# Research prompt for GPT web search

Paste everything below the line. Written 2026-09-18. Companion to the two Claude deep-research
agents run the same day; this one is for an independent sweep with a different retrieval stack,
so overlap is expected and useful as a cross-check.

Construction rules applied (from the 2026-09-03 verification of the last prompt): entry points
are author names, seed arXiv IDs and venue-years rather than keyword strings; one sub-question
per gap ID and one output-table row per gap ID; every quoted search phrase is written in the
target field's own words; free arXiv routes are named instead of publisher routes; the
already-held list is given so rediscoveries are not reported as findings.

---

You are a literature-search assistant for a control-engineering master's thesis. Use web search
aggressively. Do not answer from memory: every paper you name must come from a search you ran,
and you must give its venue, year, DOI and a working free link.

## 1. The exact setting

A physics-based baseline model is augmented additively with a learned component:

```
y_k  =  f_base(theta, x_k)  +  f_ANN(x_k)
```

The data-generating truth is `f_truth = f_base(theta*, x) + delta(x)`, where `theta*` is the
physically true parameter vector and `delta` is the unmodelled dynamics.

Let `Phi` be the stacked sensitivity of the baseline to its parameters over the training set
(the regressor matrix when the baseline is linear in its parameters, `d f_base / d theta`
otherwise), and `Delta` the stacked unmodelled term. Constraining the learned component to be
orthogonal to the sensitivity span, `Phi^T f_ANN = 0`, makes the baseline/augmentation split
unique, and the resulting parameter error is exactly

```
theta_hat - theta*  =  (Phi^T Phi)^-1 Phi^T Delta  =  Phi^+ Delta
```

so the true parameters are recovered **if and only if**

```
Phi^T Delta = 0,     i.e.   SUM_i  phi^T(x_i) delta(x_i) = 0.
```

This condition is a property of the PLANT and the DATA, not of the estimator. The estimator side
is settled. **The open question is how to make the condition HOLD.**

## 2. What is already known to us. Do NOT report these as findings

Report them only if you find a clause in them that bears on a gap below, and say which clause.

| paper | what we already take from it |
|-|-|
| Györök, Schoukens, Péni, Tóth, "Orthogonal-by-construction augmentation of physics-based input-output models", arXiv:2511.01321 | the seed. Condition 4 (Eq. 17) is the orthogonality condition, Eq. 21 the parameter error, Theorem 7 the recovery result, Theorem 16 consistency, Theorem 18 zero covariance. Remark 5 suggests designing the input from knowledge of `delta`'s structure. Section 5.1 builds a sign-symmetric dataset so even and odd nonlinearities are orthogonal |
| Györök et al., L4DC 2025, arXiv:2501.05842 | the soft projection-based regulariser, and the Taylor fallback for baselines nonlinear in their parameters |
| Kon, Bruijnen, Van de Wijdeven, Heertjes, Oomen, ACC 2022, arXiv:2201.03308 | origin of the orthogonal-projection idea for physics-guided feedforward |
| Bolderman, Lazar, Butler, ECC 2022 and CEP 2024, arXiv:2301.08568 | the parameter-anchor regulariser, a different mechanism |
| Hoekstra, Verhoek, Tóth, Schoukens, EJC 2025, arXiv:2404.01901 | LFR augmentation, names the negation problem, solves it with the anchor not with projection |
| Donati, Mammarella, Dabbene, Novara, Lagoa, Automatica 2025, arXiv:2405.18186 | multi-step off-white/sparse-black identification with a similar error expression |
| Tuo and Wu AnnStat 2015; Tuo SIAM-ASA-JUQ 2019; Wang 2022; Xie and Xu JASA 2020; Plumlee JASA 2017 and "Orthogonal Gaussian process models"; Brynjarsdóttir and O'Hagan, Inverse Problems 2014 | the computer-model calibration corpus and its orthogonality/identifiability geometry |
| Nekipelov et al., Econometrics J. 2022, arXiv:1806.04823 | regularised orthogonal ML for nonlinear semiparametric models |
| Gevers, Bazanella, Bombois, Mišković, IEEE TAC 2009 | the information matrix and "sufficiently rich" conditions |

## 3. The gaps. One row per gap ID in your output table

**G1. Designing the EXCITATION so the orthogonality condition holds.**
The seed's Remark 5 asserts this is possible when the structure of `delta` is known, and gives
quadratic aerodynamic drag as the example, but proves nothing and cites no design method. Who has
actually formulated input design whose objective or constraint is the decorrelation of an
unmodelled term from a baseline regressor? Adjacent named areas to check: optimal input design for
grey-box identification, least-costly and application-oriented experiment design, informative
experiments, excitation allocation for identifiability in dynamic networks.

**G2. Parity and symmetry designs.**
The seed's Section 5.1 gets exact orthogonality by appending the input multiplied by `-1`, so that
even and odd nonlinearities are orthogonal. In nonlinear system identification this is the odd
random-phase multisine with detection lines (Pintelon and Schoukens), where odd and even
nonlinear contributions are separated by which FFT lines they land on. Has anyone connected the
multisine parity machinery to PARAMETER RECOVERY in a grey-box or augmented model, as opposed to
using it for nonlinearity DETECTION and best-linear-approximation distortion analysis? This
connection is the one we most want and the one we least expect to exist.

**G3. Structural design of the ADDED DYNAMICS itself.**
Rather than designing the input, design the plant's extra dynamics so its contribution is
orthogonal to the parameter sensitivities for any input. Two specific claims we have derived and
want checked against the literature:
  (i) pointwise orthogonality is impossible whenever the number of parameters exceeds the number
      of generalised coordinates, since the sensitivity vectors then span the coordinate space, so
      orthogonality must come from cancellation over the record;
  (ii) a DISSIPATIVE addition always has a strictly positive time-averaged correlation with the
      damping-parameter sensitivities, so exact orthogonality to damping parameters requires a
      LOSSLESS addition.
Has either been stated anywhere? Look in structural dynamics and vibration as well as control: the
objects are apparent mass, tuned mass dampers, and the sign change of a resonator's reactive force
through its resonance.

**G4. Nonlinear-in-parameters and state-space baselines.**
The seed's own Conclusion names this as open: "extending the approach for augmenting baseline
models in state-space form, where the general assumption of no available full-state measurement
complicates the orthogonal projection of the learning component". Our baseline is a linear
parameter-varying linear-fractional representation with a state-dependent mass matrix, nonlinear
in its parameters, discretised by Runge-Kutta, with the state reconstructed by an encoder rather
than measured. Who has orthogonality or non-overlap conditions in that setting? Check separable
nonlinear least squares and VARIABLE PROJECTION (Golub and Pereyra and successors), where the same
normal-equation orthogonality reappears as the variable-projection functional.

**G5. Multi-step and closed-loop objectives.**
The recovery theorem assumes the ONE-STEP least-squares normal equations. Our training objective
is a multi-step CLOSED-LOOP simulation error, so the inner product the optimiser actually works in
is not the one the projection orthogonalises in. Does anyone bound, characterise or remove that
mismatch? Relevant neighbourhoods: simulation-error versus prediction-error identification,
multiple shooting for system identification, closed-loop identification bias.

## 4. Search the same ideas in at least three fields' own words

State next to each finding which vocabulary produced it. Do not reuse our words; each field names
this differently.

| field | phrase to search |
|-|-|
| control, experiment design | "least costly experiment design", "application oriented experiment design", "informative experiments", "persistency of excitation" |
| nonlinear system identification | "odd random phase multisine", "detection lines", "best linear approximation", "even and odd nonlinear distortions" |
| statistics and UQ | "computer model calibration", "model discrepancy", "identifiability of the calibration parameter", "orthogonal Gaussian process" |
| econometrics | "Neyman orthogonality", "orthogonal moment function", "debiased machine learning", "nuisance tangent space" |
| numerical linear algebra | "variable projection", "separable nonlinear least squares" |
| structural dynamics | "apparent mass", "tuned mass damper", "reactive force", "antiresonance" |

Write each search as a SENTENCE you expect the target paper to contain, and quote at most one
phrase per search. Two or more quoted phrases in one query returns nothing on most engines, which
is indistinguishable from a real zero.

## 5. Disqualifiers. Report as "found but disqualified by Dn", not as a hit

- **D1** requires full-state measurement. We reconstruct the state with an encoder.
- **D2** a soft penalty with a trade-off weight and nothing more. That is the ancestry we already
  have; it is not an answer to "by construction".
- **D3** linear-in-parameters only, with no stated route to a baseline nonlinear in its parameters.
- **D4** a static discrepancy model with no dynamics. Still report it if the GEOMETRY is on point,
  but label it.

## 6. Out of scope. Do not return work on these

How to build, refresh or factorise the projector once the condition is granted. Parameter-anchor
or nominal-value regularisation. Encoder and state-estimator architecture.

## 7. Access

Prefer arXiv, PMLR and institutional repositories. Every paper in section 2 above is free on arXiv
at the ID given, so do not route those through IEEE or Elsevier. IFAC-PapersOnLine is fully open
access. For anything closed, say so and give the DOI rather than guessing a link.

## 8. Output format

```
## Gap table
| gap | best paper | venue, year | DOI | free link | what it actually establishes | confidence |
| G1 | ... |
| G2 | ... |
| G3 | ... |
| G4 | ... |
| G5 | ... |
(one row per gap ID, and a row saying "nothing found" if that is the honest answer)

## Findings in detail
<per paper: full citation, the specific result or clause, and whether you read the full text,
 the abstract only, or metadata only. Be explicit; an abstract can invert a verdict.>

## Novelty assessment
<for G2 and G3, which we believe may be genuinely unreported: say which vocabularies you searched
 before concluding, and grade the claim. "Not found in N searches across M vocabularies" is an
 honest answer; "nobody has done this" is not.>

## Search log
<query, engine, number of results, on-target or not. Include the searches that returned nothing:
 a genuine zero over a well-chosen phrase is evidence, and we want to see which phrase produced it.>
```
