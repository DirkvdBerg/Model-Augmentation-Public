# ML-for-Control Literature Sweep (6 directions, one at a time)

**Date**: 2026-07-11. **Why this doc**: the exposure-bias find (`docs/rollout-stability-literature.md`)
showed our drift problem lives under DIFFERENT vocabularies in different communities. This sweep translates
our problem into each community's native terms and searches them one at a time. Rule: **quote directly from
the primary PDF** (downloaded + text-extracted); NEVER quote from a search snippet. Each direction ends with
a decision link. Main doc: `docs/drift-diagnosis-status.md` (§0 index). Direction: D-107. Quotes are
transcribed from on-disk PDF text layers; re-verify character-exact before thesis use.

The six directions:
1. Rollout stability frontier (beyond noise injection).
2. Hybrid-model identifiability / negation (interpretability meets drift).
3. Learning on manifolds / symmetries (the free mode as translation symmetry).
4. **[DONE]** Robust/consistent ID for marginal + closed-loop data (bias correction, IV).
5. LPV + ML (our model class).
6. Diagnosis: what actually causes rollout drift (Jacobian spectrum).

---

## Direction 4 -- Bias-corrected / instrumental-variable ID (marginal + closed-loop) [PRIMARY-READ]

**Our problem in this community's terms:** our real Telica data is CLOSED-LOOP; the deliverable is the
OPEN-LOOP model; the controller correlates the input with the noise, which BIASES least-squares estimates.
The spurious DC that drifts is (partly) an estimation BIAS. This community removes exactly that bias.

### 4.1 Closed-loop LPV bias correction -- OUR EXACT SETTING  [PRIMARY-READ]
**Reference.** M. Mejari, D. Piga, A. Bemporad, "A Bias-Correction Method for Closed-Loop Identification of
Linear Parameter-Varying Systems", Automatica 87 (2018) 128-141. (Preprint ipg.idsia.ch/preprints/piga2018c.)
- **The setting = ours (Abstract, verbatim):** "we present a bias-correction scheme for closed-loop
  identification of Linear Parameter-Varying Input-Output (LPV-IO) models, which aims at correcting the bias
  caused by the correlation between the input signal exciting the process and output noise. The proposed
  identification algorithm provides a consistent estimate of the open-loop model parameters when both the
  output signal and the scheduling variable are corrupted by measurement noise."
- **Why closed-loop biases LS (Intro, verbatim):** "one of the main issues which makes identification from
  closed-loop experiments more challenging than in the open-loop setting is due to the correlation between
  the plant input and output noise."
- **The mechanism (Intro, verbatim):** "The idea underlying bias-correction methods is to eliminate the bias
  from ordinary Least Squares (LS) to obtain a consistent estimate of the model parameters." The paper also
  notes Instrumental-Variable / Refined-IV (RIV) as related routes for the same goal.
- **Match to us:** closed-loop data (yes), LPV-IO model (our `M(Y)` is LPV), recover the OPEN-LOOP model
  (our deliverable), scheduling variable Y is measured/noisy (yes). This is a 1:1 structural match for the
  REAL-DATA baseline/scheduling fit.
- **CRITICAL CAVEAT (be honest):** it is LINEAR-parameter-varying, linear-in-the-model. It bias-corrects the
  LPV BASELINE fit; it is NOT a method for the nonlinear ANN residual. Its role for us is: get an UNBIASED
  LPV baseline from closed-loop data so the ANN does not inherit a biased baseline (and so a controller-
  induced bias is not mistaken for a residual the ANN must learn). It does not by itself fix the ANN drift.

### 4.2 Instrumental variables synthesized from data, nonlinear, finite-sample  [PRIMARY-READ]
**Reference.** S. Kuang, X. Lin, "Instrumental variables system identification with Lp consistency", PMLR
vol 331 (2026).
- **Claim (Abstract, verbatim):** "Instrumental variables (IV) eliminate the bias that afflicts least-squares
  identification of dynamical systems through noisy data, yet traditionally relies on external instruments
  that are seldom available for nonlinear time series data. We propose an IV estimator that synthesizes
  instruments from the data. We establish finite-sample Lp consistency for all p >= 1 ... On a forced Lorenz
  system our estimator reduces parameter bias by 200x (continuous-time) and 500x (discrete-time) relative to
  least squares and reduces RMSE by up to tenfold."
- **CRITICAL CAVEAT (their own words, Abstract + Intro, verbatim):** "Because the method only assumes that
  the model is linear in the unknown parameters, it is broadly applicable to modern sparsity-promoting
  dynamics learning models." And: "assuming a parametric form in which the prediction is nonlinear in the
  inputs ... but linear in the parameter vector." So it is **linear-in-parameters (SINDy/library style), NOT
  a general nonlinear ANN.** The "nonlinear" refers to nonlinearity in the inputs, not a neural residual.
- **Match to us:** a modern, finite-sample-consistent bias remover -- but only for linear-in-parameters
  models. Usable for a linear-in-parameters friction/basis residual, NOT for Jan's ANN. Strong as a
  BASELINE/grey-box-friction estimator, not the ANN.

### 4.3 Integrating-disturbance ML-ID (marginally-stable mode from data)  [PRIMARY-READ, earlier]
**Reference.** S.J. Kuntz, J.B. Rawlings, "Maximum Likelihood Identification of Linear Models with
Integrating Disturbances for Offset-Free Control", IEEE TAC 70(9):5675-5689, 2025 (arXiv:2406.03760).
- **Verbatim (Abstract):** "linear time-invariant models are augmented with (fictitious) uncontrollable
  integrating modes, called integrating disturbances ... We implement eigenvalue constraints to protect
  against undesirable filter behavior (unstable or marginally stable modes ...)". Code released.
- **Match/caveat:** estimates the integrating mode DIRECTLY from data with eigenvalue-LMI constraints --
  linear, and it CONSTRAINS the marginal mode (eigenvalue constraints) rather than preserving it freely; a
  parametrization idea for the marginal mode, not a nonlinear-residual method.

### Direction-4 verdict
- **What it gives:** a control-native, citable, SOLVE-not-hide bias remover for EXACTLY our closed-loop -> 
  open-loop LPV situation (Piga/Mejari/Bemporad), plus a modern finite-sample nonlinear-in-inputs IV
  (Kuang-Lin, 200-500x bias reduction).
- **Honest limit (critical):** ALL of these are LINEAR / LINEAR-IN-PARAMETERS. None fits Jan's nonlinear ANN
  residual. Their real role is the **REAL-DATA BASELINE/scheduling fit**: remove the closed-loop bias so the
  ANN does not inherit or have to "explain" a controller-induced bias (which would otherwise look like a
  residual to learn, feeding the drift). This is genuinely useful and was NOT on our radar.
- **Requirements:** 1 KF yes; 2 expressivity -- NO for the ANN (linear-in-parameters only), so these are for
  the baseline, not the expressive residual; 3 marginal -- Piga yes (does not damp), Kuntz constrains it; 4
  drift -- addresses the BIAS component of drift, not the training-pathology component.
- **Decision link:** Piga-Bemporad (4.1) is a strong candidate for the `real-data-verification` pipeline
  (closed-loop Telica -> unbiased open-loop LPV baseline). Primary-read done; flagged for that pipeline. It
  does NOT change the sim-phase drift work (D-107), which is about the ANN, not baseline bias.

### Provenance / primary-read status (Direction 4)
- PRIMARY-READ: Mejari-Piga-Bemporad (Automatica 2018 preprint), Kuang-Lin (PMLR 331, 2026), Kuntz-Rawlings
  (arXiv:2406.03760). Quotes above transcribed from the PDF text layers; re-verify character-exact before
  thesis use.
- Directions 1, 2, 3, 6: pending (this sweep, one at a time).

---

## Direction 5 -- LPV + ML (our actual model class) [PRIMARY-READ]

**Our problem in this community's terms:** we run an LPV-LFR baseline + SUBNET encoder. What does the
deep-LPV-identification literature (esp. our supervisors' Toth/Schoukens group) say about drift, stability,
and long-horizon prediction for exactly this model class?

### 5.1 The framework we use, with CONSISTENCY guarantees  [PRIMARY-READ] -- in-framework anchor
**Reference.** C. Verhoek, G.I. Beintema, S. Haesaert, M. Schoukens, R. Toth, "Deep-Learning-Based
Identification of LPV Models for Nonlinear Systems", arXiv:2204.04060 (CDC 2022). (Schoukens + Toth = our
supervisors.)
- **What it is (Abstract, verbatim):** "This paper presents a deep-learning-based approach to provide joint
  estimation of a scheduling map and an LPV state-space model of a NL system from input-output data, and has
  consistency guarantees under general innovation-type noise conditions." Built on the SUBNET / Sub-Space
  Encoder Network (their words: "identification in terms of the Sub-Space Encoder Network").
- **Match to us:** this IS our pipeline's method (LPV + encoder). It provides CONSISTENCY (a statistical
  guarantee the estimate converges to truth under innovation noise) -- an ESTIMATION guarantee, distinct from
  a for-all-weights structural no-drift guarantee. It does NOT claim to bound free-run position on a free
  integrator; consistency is about parameter convergence, not marginal-mode rollout boundedness.
- **Role:** the in-framework anchor. Our drift problem is NOT addressed by it (consistency != no-drift on a
  free integrator), which reconfirms the gap is specific to the marginal/free-integrator rollout.

### 5.2 Stable-by-design LPV NN-SS -- SEPARABLE: reject the Schur part, KEEP the state-consistency regularizer  [PRIMARY-READ]
**Reference.** A.E. Sertbas, T. Kumbasar, "Stable-by-Design Neural Network-Based LPV State-Space Models for
System Identification", arXiv:2510.24757 (2025).
- **It treats drift as a STABILITY problem (Intro, verbatim):** unconstrained NN-SS models "lack guarantees
  of stability [9], which may lead to drift, error accumulation, and reduced reliability in long-horizon"
  prediction. Fix 1 = Schur-stable A: "The state-transition matrix ... is guaranteed to be stable through a
  Schur-based parameterization", "Schur stable, with all eigenvalues strictly inside the disk".
- **WHY FIX 1 IS WRONG FOR US (critical):** Schur = all eigenvalues STRICTLY inside the unit disk = strictly
  stable = it DAMPS the free-integrator pole (which sits ON the disk at z=1). This is the SAME failure as
  contraction/RENs (`dissipativity-limits.md` B3): it fails requirement 3 (marginal-preservation) by
  construction. So the paper's headline mechanism is exactly the one we must NOT use.
- **BUT fix 2 is SEPARABLE and useful -- the state-consistency regularizer (Eq 13, verbatim def):**
  "an auxiliary regularization term Lstate ... penalizes discrepancies between the propagated states and
  those inferred by the encoder, excluding the initial state ... where x_k denotes the state propagated
  through the model dynamics and x_k^ENC the encoder-derived estimate at the same step. This regularization
  enforces consistency between the two, reducing drift and improving robustness over long horizons and noisy
  data." Total loss (Eq 11): `Ltotal = Lresponse + lambda * Lstate`. Inspired by SIME [9].
- **Why fix 2 fits our constraints:** it is a SOFT regularizer on the training loss (not a hard structural
  constraint), so it PRESERVES expressivity (req 2) and does NOT damp the marginal mode (req 3) -- it only
  penalizes the gap between the propagated state and the encoder's re-inferred state at each step. That is a
  MULTIPLE-SHOOTING / continuity-style term (our §5 SUPPORTING, D-107 conditioning) in LPV-NN-SS form, and
  it is claimed to REDUCE DRIFT. Requirements: 1 KF yes; 2 yes; 3 yes; 4 empirical (drift reduced, not
  structurally forbidden).
- **CAVEAT:** their benchmarks are stable nonlinear systems, NOT a free integrator; "reduces drift" is
  empirical on stable systems. And in THEIR paper the consistency term is paired WITH Schur -- we would take
  the consistency term WITHOUT the Schur damping. Separating them is our adaptation, not their result.

### Direction-5 verdict
- **In-framework anchor (5.1):** our own method (Verhoek et al. 2204.04060) has CONSISTENCY (estimation
  convergence), not marginal-mode no-drift -- reconfirms the gap is the free-integrator rollout, not a hole
  in the framework.
- **The reusable piece (5.2):** the STATE-CONSISTENCY regularizer (encoder-vs-propagated-state discrepancy,
  Eq 13) is a soft, expressivity-preserving, marginal-preserving, in-LPV-NN-SS drift-reducer -- essentially
  the multiple-shooting/continuity conditioning of D-107, already formulated for a neural LPV state-space
  model. **KEEP it; REJECT the Schur stabilization it is bundled with (Schur damps our free pole = fails
  req 3).**
- **Requirements (5.2 state-consistency only):** 1 yes, 2 yes, 3 yes, 4 empirical. Solve-not-hide (it fixes
  the model's state evolution). Directly feeds the D-107 first step alongside GNS noise injection /
  pushforward (`rollout-stability-literature.md`).
- **Decision link:** add the state-consistency (encoder-propagated-state) regularizer to the D-107 clean
  re-run as the LPV-NN-SS form of continuity conditioning; do NOT adopt Schur/stable-by-design (damps the
  marginal mode). The consistency term composes with our encoder (we already have `x^ENC` and propagated `x`).

### Provenance / primary-read status (Direction 5)
- PRIMARY-READ: Verhoek-Beintema-Haesaert-Schoukens-Toth (arXiv:2204.04060), Sertbas-Kumbasar
  (arXiv:2510.24757). Quotes transcribed from the PDF text layers; re-verify character-exact before thesis
  use. Note 5.2's `Lstate` is "inspired by SIME [9]" -- trace that reference before thesis citation.
- Directions 1, 2, 3: pending.

---

## Direction 6 -- Diagnosis: what actually CAUSES rollout drift (Jacobian spectrum) [PRIMARY-READ]

**Our problem in this community's terms:** we DIAGNOSED our drift as a near-pure DC offset on the K=0 rows
(d6: `|mean|/rms = 1.00`), ramped by the free integrator. Does the rollout-diagnosis literature CONFIRM that,
or reveal an ADDITIONAL mechanism (e.g. non-normal transient growth) we assumed away? This direction is
DIAGNOSTIC, not a new fix.

### 6.1 Non-normal transient amplification -- the one transferable diagnostic  [PRIMARY-READ, also rollout-lit C]
**Reference.** A. Pervez, F. Locatello, "Controlling Transient Amplification Improves Long-horizon Rollouts",
arXiv:2605.08856. (Full read in `rollout-stability-literature.md` C.)
- **Mechanism (verbatim):** "when the Jacobians along an autoregressive trajectory are non-normal and
  non-commuting, the model amplifies errors transiently, resulting in model rollout drift even when the
  overall system is asymptotically stable."
- **Diagnostic value for us:** this is a GENERAL, low-dimensional-applicable check -- compute the trained
  model's step Jacobians along the rollout and test (a) normality defect and (b) cross-step commutator. If
  non-negligible, our drift has a transient-amplification COMPONENT beyond the measured DC; if negligible,
  it CONFIRMS the DC-only diagnosis (d6). Cheap to run on our trained checkpoint (Jacobian-vector products).
- **Honest note:** d6 already measured near-pure DC, so we EXPECT the non-normal component to be small -- but
  we never checked it. This is a genuine gap-closing diagnostic, not a new solution.

### 6.2 Semigroup consistency -- a model-agnostic rollout diagnostic (weak)  [PRIMARY-READ]
**Reference.** L.J. Shikhman, "Semigroup Consistency as a Diagnostic for Learned Physics Simulators",
arXiv:2605.26324 (2026).
- **Idea (verbatim):** "exact solution maps satisfy a semigroup law: direct evolution over s + t should agree
  with evolution over s followed by t. We propose normalized semigroup error as a post hoc, model-agnostic
  diagnostic comparing these direct and composed learned predictions."
- **HONEST LIMIT (their own conclusion, verbatim):** "semigroup error is positively associated with rollout
  degradation, with trajectory-level Spearman correlation rho = 0.635"; and "Semigroup regularization has
  mixed effects, supporting semigroup consistency primarily as an evaluation diagnostic rather than a
  universally beneficial training objective." So it is a WEAK diagnostic (rho=0.635) and NOT a useful
  training objective. Low value for us; do not adopt as a fix.

### 6.3 Autoregressive neural-operator instability -- PDE-specific, does NOT transfer  [PRIMARY-READ]
**Reference.** M. McCabe, P. Harrington, S. Subramanian, J. Brown, "Towards Stability of Autoregressive
Neural Operators", TMLR 11/2023 (arXiv:2306.10619).
- **Mechanism (verbatim):** "autoregressive spatiotemporal models show signs of aliasing and numerical
  instability, contributing to" error growth; they "draw parallels between instability [and numerical
  analysis]"; "uncontrolled error growth occurs in high frequencies ... consistent with the accumulation of
  aliasing error."
- **Why it does NOT transfer (critical):** their mechanism is ALIASING / high-frequency growth in
  HIGH-DIMENSIONAL SPATIAL PDE fields (weather, shallow water). Our system is LOW-DIMENSIONAL mechanical with
  a free integrator; our drift is DC x integrator (d6), not spatial aliasing. So this paper's specific
  mechanism and fixes (spectral filtering, dealiasing) do NOT apply. Included to record the negative: the
  PDE-simulator drift literature is partly about a mechanism we do not have.

### Direction-6 verdict
- **Diagnosis, not solution.** Direction 6 gives DIAGNOSTIC tools, not a new fix. Our DC diagnosis (d6) is
  sound; the one gap-closing check we have NOT run is the **Jacobian non-normality / commutator test (6.1)**
  -- to confirm the drift is DC-only vs partly transient-amplification. Cheap, worth running on the trained
  checkpoint.
- **What does NOT apply:** semigroup consistency (weak, rho=0.635, 6.2) and PDE aliasing (6.3, wrong
  mechanism for a low-dim mechanical integrator). Recording these as negatives is useful: the high-dim
  PDE-simulator drift literature is only PARTIALLY about our problem.
- **Requirements:** N/A (diagnostics, not augmentation methods). 6.1's commutativity regularization, IF the
  non-normal component turns out significant, is an expressivity-preserving regularizer (1,2,3 yes, 4
  empirical) -- but only pursue it if the diagnostic finds non-normality.
- **Decision link:** add the Jacobian non-normality/commutator check (6.1) to the diagnostics on the trained
  checkpoint, to confirm/refine the d6 DC-only finding. Do NOT adopt semigroup or PDE-dealiasing.

### Provenance / primary-read status (Direction 6)
- PRIMARY-READ: Pervez-Locatello (2605.08856, also rollout-lit C), Shikhman (2605.26324), McCabe et al.
  (2306.10619). Quotes transcribed from the PDF text layers; re-verify character-exact before thesis use.
- Directions 1, 2, 3: pending.

---

## Direction 1 -- Rollout-stability frontier BEYOND noise injection [PRIMARY-READ]

**Our problem in this community's terms:** we have pushforward + GNS noise injection (rollout-lit A/B). What
does the frontier say about HOW to do the unrolled/multi-step training (which our D-107 conditioning IS), and
is noise injection actually the right tool?

### 1.1 Unrolled training, disentangled -- directly informs D-107, and studies OUR setup  [PRIMARY-READ]
**Reference.** B. List, L.-W. Chen, K. Bali, N. Thuerey, "Differentiability in Unrolled Training of Neural
Physics Simulators on Transient Dynamics", TMLR / arXiv:2402.12971 (2024).
- **It studies OUR exact structure -- "correction setups" (Abstract, verbatim):** "In prediction setups, we
  rely solely on neural networks to compute a trajectory. In contrast, CORRECTION setups include a numerical
  solver that is supported by a neural network." Our grey-box = baseline solver (numerical) + ANN correction
  = a correction setup. (Their words, p.9: "Correction setups are also concerned with time-evolving a
  discretized PDE but additionally include a numerical [solver].")
- **It disentangles the TWO effects of unrolling (Abstract, verbatim):** comparing one-step, fully-
  differentiable unrolling, and unrolling-WITHOUT-temporal-gradients "disentangles the two dominant effects
  of unrolling, TRAINING DISTRIBUTION SHIFT and LONG-TERM GRADIENTS." This is exactly our question: how much
  of the D-107 conditioning benefit is distribution-shift correction (what noise injection targets) vs
  long-horizon gradients (what full BPTT gives).
- **Main finding (Abstract, verbatim):** "Non-differentiable but unrolled training with a numerical solver in
  a correction setup can yield substantial improvements over a fully differentiable prediction setup" and
  "Differentiable ones perform best ... [but] the accuracy of non-differentiable unrolling comes close."
  Also: unrolling without gradients "has a stabilizing effect [but] it also limits the effective unrolling
  horizon."
- **CRITICAL for us -- their CRITIQUE of noise injection (p.6, verbatim):** "A common data-augmentation
  approach is to perturb the ground truth states ... with noise, in the hope that the perturbed states fit
  A_gθ [the model's own attractor] more closely. While this approach was reported to be successful, it still
  DOES NOT GUARANTEE that the observed states are from A_gθ, and thus, those augmented states are NOT
  NECESSARILY CLOSER to states observed during inference." -> noise injection is a PROXY for the model's own
  drifted distribution; it may not match it. This is an honest caveat on our GNS recommendation
  (`rollout-stability-literature.md` B): noise injection helps but is not guaranteed correct; TRUE unrolling
  (feed the model its own output) is the more faithful conditioning.
- **Requirements:** 1/2/3 yes (training-setup choices, no class restriction); 4 empirical. SOLVES.
- **Decision link (strong):** our setup IS a correction setup; this paper says (a) unrolled/multi-step
  training substantially helps correction setups, (b) TRUE unrolling (feed own output) is more faithful than
  noise injection (which is a proxy), and (c) non-differentiable unrolling (unroll without BPTT through the
  baseline solver) is a viable, stabilizing option that "comes close" to full differentiable -- relevant
  because our baseline is a differentiable LFR, but the pushforward "don't backprop the first step" is
  exactly this family. This directly shapes the D-107 first step: prefer TRUE short unrolling (pushforward /
  multi-step) over pure noise injection, and consider partial-gradient variants.

### 1.2 Scheduled sampling / DAgger -- the exposure-bias curriculum  [SEARCH-LEVEL, not primary-read]
**References (foundational, NOT primary-read this session -- cite carefully):** S. Bengio et al., "Scheduled
Sampling for Sequence Prediction with RNNs", NeurIPS 2015; S. Ross, G. Gordon, D. Bagnell, "DAgger", AISTATS
2011; H. Daume et al. 2009.
- **Idea (search-level):** gradually transition training from ground-truth inputs to the model's OWN
  predictions (scheduled sampling), or aggregate training data on the model's own visited states (DAgger) --
  the imitation-learning framing of exposure bias. The dynamics-model version is the unrolled/blend-forcing
  training of 1.1. "Blend-forcing" (2026 frontier, search-level) = gradually interpolate ground-truth and
  generated states = scheduled sampling for dynamics.
- **Status:** foundational and directly the exposure-bias curriculum, but these are NLP/imitation origins;
  the DYNAMICS-specific, rigorous version is 1.1 (unrolled training). Primary-read 1.1, not these. Verify at
  source before thesis citation.

### 1.3 2026 frontier (mostly high-dim PDE/generative -- limited transfer)  [SEARCH-LEVEL]
Memory-conditioned flow matching (2602.06689), multi-token prediction, conservative-discrete-structure
rollouts (2606.01366), Thermalizer (2503.18731). These target HIGH-DIM spatiotemporal chaos / generative
rollouts; the mechanisms (memory bottleneck, dealiasing, conservative discretization) are mostly PDE/spatial,
limited transfer to our low-dim mechanical integrator. Recorded as frontier context, not adopted.

### Direction-1 verdict
- **The on-target find (1.1):** the unrolled-training study is DIRECTLY about our setup ("correction setups"
  = grey-box baseline+ANN) and disentangles distribution-shift vs long-term-gradients. Key takeaways for
  D-107: (a) unrolled/multi-step training substantially helps correction setups; (b) TRUE unrolling (feed the
  model its own output; pushforward/multi-step) is MORE FAITHFUL than noise injection, which is only a PROXY
  for the model's own drifted distribution and "not necessarily closer" to inference states; (c)
  non-differentiable / partial-gradient unrolling is a viable stabilizer.
- **CRITICAL CAVEAT -- do NOT claim rollout stability "solves" OUR drift (added 2026-07-11, user challenge).**
  Two reasons it may NOT fix our specific drift:
  1. **Timescale mismatch.** Our drift appears at ~0.5 s (d7), but training windows are 0.1 s, Optuna 69399
     unrolled only to nf=1600=0.4 s (below onset), and a 2-step pushforward is 0.2 s -- all BELOW where the
     DC becomes visible. Short unrolling does NOT reach our drift regime, so "unrolling-first" only helps if
     unrolled PAST ~0.5 s (the expensive/CHyLL-divergence region). This is likely part of why nf up to 1600
     already failed (aside from the lr confound).
  2. **Two-component drift.** Our drift = (a) a distribution-shift component (model visits drifted states it
     never trained on -- rollout methods help this) PLUS (b) an IDENTIFIABILITY component (the DC direction
     is RMS-invisible over the window, so the loss cannot penalize it -- §5m / Direction 2). Rollout methods
     address (a); they do NOT make a genuinely-UNEXCITED DC identifiable (b). If (b) dominates (d6/d7 suggest
     it does), rollout methods help only via reaching the drift timescale (long unroll) or must be paired
     with excitation / data-silent regularization.
- **Refines our earlier recommendation, WITH the caveat:** GNS random-walk NOISE INJECTION is the one tool
  well-matched WITHOUT long unrolling (a random walk of the right magnitude simulates the free-integrator's
  long-horizon drifted state at short cost). BUT it carries a REQ-3 RISK (see rollout-stability-literature.md
  GNS-fit section): "correct-back-from-drift" training can induce an effective RESTORING/DAMPING action that
  DAMPS the marginal pole -> must eigen-check that the X/Y pole stays at the origin. So neither "unrolling-
  first" nor "noise-injection" is a safe silver bullet; both are falsifiable experiments for D-107, not
  proven fixes.
- **Requirements:** 1/2 yes; 3 -- AT RISK for GNS/correct-back (may damp the marginal mode; eigen-check
  required); 4 empirical, and ONLY if the exposure reaches the drift timescale. SOLVES the distribution-shift
  half; the identifiability half needs excitation / data-silent. Feeds D-107 as EXPERIMENTS, not guarantees.

### Provenance / primary-read status (Direction 1)
- PRIMARY-READ: List-Chen-Bali-Thuerey (arXiv:2402.12971). Quotes transcribed from the PDF text layer;
  re-verify character-exact before thesis use.
- SEARCH-LEVEL (verify before citing): scheduled sampling (Bengio 2015), DAgger (Ross 2011), blend-forcing /
  memory-conditioned / conservative-rollout frontier (2602.06689, 2606.01366, 2503.18731).
- Directions 2, 3: pending.

---

## Direction 2 -- Hybrid-model identifiability / NEGATION (interpretability meets drift) [PRIMARY-READ]

**Our problem in this community's terms:** our thesis contribution is INTERPRETABLE augmentation -- keep the
physical parameters identifiable while a NN learns the residual, and stop the NN from ABSORBING/NEGATING the
baseline physics (CLAUDE.md control-stance point 6; our orthogonal-projection layer, Gyorok). This is the
"hybrid/UDE identifiability" problem, and it is a large, active field we had NOT searched. This is the most
CONTRIBUTION-relevant direction.

### 2.1 Functional vs parametric identifiability of UDEs -- the framing for our contribution  [PRIMARY-READ]
**Reference.** T.E. Loman, R.E. Baker, "Functional and parametric identifiability for universal differential
equations applied to chemical reaction networks", arXiv:2510.14140 (2025).
- **The split we should adopt (Abstract, verbatim):** "UDE identifiability, i.e. our ability to identify true
  system properties, can be split into PARAMETRIC and FUNCTIONAL identifiability (assessing identifiability
  for the mechanistic and data-driven model parts, respectively)." -> exactly our two concerns: PARAMETRIC =
  the physical parameters stay identifiable (interpretability), FUNCTIONAL = the ANN residual is identifiable.
- **Encouraging result (Abstract, verbatim):** "across a wide range of models, the generalisation of a fully
  mechanistic model to a UDE has LITTLE IMPACT on the mechanistic components' parametric identifiability."
  And (p.5): "the generalisation of functions from their known (parameterised) forms to neural networks
  typically has little effect on parametric identifiability."
- **Relevance:** gives us the precise VOCABULARY (parametric vs functional identifiability) and a measurement
  toolkit (profile likelihood for parametric; "ensemble plots of fitted functions" for functional). CAVEAT:
  their result is for chemical-reaction-network ODEs, NOT an LPV free-integrator with a drift; "little impact
  on parametric identifiability" is their domain finding, do not assume it transfers -- but the FRAMING does.

### 2.2 Gray-box identifiability: interpretability preserved via regularization, even when non-identifiable  [PRIMARY-READ]
**Reference.** M. Hotvedt, B. Grimstad, L. Imsland, "Identifiability and physical interpretability of hybrid,
gray-box models - a case study", IFAC 2021, arXiv:2010.13416.
- **The core message (Abstract, verbatim):** "For gray-box, hybrid models, model identifiability is rarely
  obtainable due to a high number of parameters. We illustrate ... that physical interpretability may be
  PRESERVED even for NON-IDENTIFIABLE models with adequate parameter regularization in the estimation
  problem."
- **Relevance:** directly supports our approach -- a hybrid model need not be fully identifiable to keep
  physical interpretability, IF the estimation is regularized to keep the parameters near-physical. This is
  the same role as our `param_loss`/Lambda (bound parameter deviation) AND the orthogonal projection (keep
  the ANN out of the physical subspace). Real-industrial validation (petroleum choke valve).

### 2.3 The KEY contribution-validating finding: other fields independently reinvented ORTHOGONAL PROJECTION  [SEARCH-LEVEL -- primary source paywalled]
**Reference (NOT primary-read: Nature/npj paywalled, could not verify verbatim).** "Robust parameter
estimation and identifiability analysis with hybrid neural ODEs", npj Systems Biology and Applications 2024
(s41540-024-00460-3).
- **Search-level finding (MUST verify at source before any thesis use):** the hybrid-NODE literature enforces
  mechanistic-parameter identifiability by regularizers that (paraphrase, per search summary) "minimize the
  impact of the neural network on the model" OR "ensure the outputs of the neural network and mechanistic
  part are UNCORRELATED", and that generic weight decay ALONE is INSUFFICIENT to recover the mechanistic
  parameters.
- **Why this matters (if verified):** "ensure NN and mechanistic outputs are UNCORRELATED/decorrelated" is
  STRUCTURALLY the SAME IDEA as Gyorok orthogonal projection (penalize the ANN component in the FP-model
  output subspace). If independent communities (systems biology) arrived at the same decorrelation principle,
  that VALIDATES the orthogonal-projection approach as a general hybrid-ID tool -- AND sharpens what is NOVEL
  in our work: the LPV/MIMO/LFR extension and the tie to marginal-mode drift, not the decorrelation idea
  itself. **This is a double-edged sword for the thesis: it strengthens the method's grounding but means the
  bare "decorrelate NN from physics" idea is NOT solely ours -- Gyorok's specific FP-Jacobian-SVD projection
  and our LPV extension are.** DO NOT quote the npj paper until the verbatim is verified (paywalled here).

### Direction-2 verdict
- **Framing win (2.1):** adopt PARAMETRIC vs FUNCTIONAL identifiability as the vocabulary for the
  interpretability contribution (Loman-Baker). Parametric = physical params stay identifiable; functional =
  ANN residual identifiable. Measurement tools: profile likelihood + function ensembles.
- **Support (2.2):** interpretability can be preserved for a non-identifiable hybrid model via parameter
  regularization (Hotvedt et al.) -- backs our `param_loss` + projection.
- **Contribution sharpening (2.3, UNVERIFIED):** the "decorrelate NN from mechanistic output" principle
  appears independently in systems biology -> validates orthogonal projection as a general tool, and localizes
  our novelty to the LPV/MIMO/LFR extension + the drift/marginal-mode connection. MUST verify the npj quote
  at a non-paywalled source before citing.
- **Requirements:** these are about INTERPRETABILITY (criterion adjacent), not the drift-4 directly. But 2.3
  ties interpretability to our drift work: orthogonal projection (interpretability) and data-silent
  regularization (drift) are the SAME machinery on different target subspaces (§5k0, D-107) -- Direction 2
  confirms the interpretability half is a recognized, validated idea.
- **Decision link:** (a) frame the contribution with parametric/functional-identifiability language;
  (b) VERIFY the npj decorrelation quote at source -- if it holds, cite it as independent support for
  orthogonal projection and be careful to claim only the LPV/LFR extension as novel; (c) no change to the
  drift experiment, but it strengthens the "interpretability + no-drift = one regularization layer" story.

### Provenance / primary-read status (Direction 2)
- PRIMARY-READ: Loman-Baker (arXiv:2510.14140), Hotvedt-Grimstad-Imsland (arXiv:2010.13416). Quotes
  transcribed from PDF text layers; re-verify character-exact before thesis use.
- SEARCH-LEVEL, PRIMARY SOURCE PAYWALLED (do NOT quote until verified): npj Syst Biol Appl 2024 HNODE
  decorrelation result (s41540-024-00460-3); UDE origin (Rackauckas et al. arXiv:2001.04385).
- Direction 3: pending.

---

## Direction 3 -- Learning on manifolds / symmetries (the free mode as translation symmetry) [PRIMARY-READ]

**Our problem in this community's terms:** the free-integrator X/Y mode is a translation symmetry (Noether ->
conserved momentum). Does symmetry-preserving / equivariant learning give a NATIVE way to keep the marginal
mode without damping it? **Verdict: elegant but WRONG for us, for two concrete reasons.**

### 3.1 Exact conservation-law NN integrators -- for CONSERVATIVE systems; ours is not  [PRIMARY-READ]
**Reference.** E.H. Muller, "Exact conservation laws for neural network integrators of dynamical systems",
J. Comp. Phys. / arXiv:2209.11661 (2023).
- **Method (Abstract, verbatim):** "we present an alternative approach which uses Noether's Theorem to
  inherently incorporate conservation laws into the architecture of the neural network" -> energy / (angular)
  momentum conserved "by construction". Validated on a particle in a Newtonian potential, Schwarzschild
  geodesics, two interacting particles.
- **WHY IT DOES NOT APPLY (critical):** these are CONSERVATIVE, CLOSED, autonomous Lagrangian systems (no
  external forcing, no dissipation). Our X/Y axes are FORCED (external control input u) and DISSIPATIVE
  (damping, finite tau) -- NOT a conservative Lagrangian system. Exact energy/momentum conservation is the
  wrong invariant: our system deliberately exchanges energy/momentum with the actuator and the damper. So
  "conserve momentum by construction" cannot be imposed -- it would contradict the physics.

### 3.2 Momentum-conserving GNN (Dynami-CAL) -- external channel unconstrained  [PRIMARY-READ, also §5m]
**Reference.** Dynami-CAL GraphNet, arXiv:2501.07373 (full read in `drift-diagnosis-status.md` §5m).
- **Already established:** internal pairwise forces are equal-and-opposite (momentum conserved for all
  weights), BUT "If external forces are present, the changes in velocity ... are decoded directly from the
  node scalar embeddings" -- the EXTERNAL-force channel is UNCONSTRAINED. Our drift is a net EXTERNAL force on
  X/Y, so it lives in the unconstrained channel -> momentum conservation does NOT forbid it. (Fails crit 4.)

### 3.3 Translation-equivariance -- would FORBID the position-dependence we REQUIRE  [reasoned from the equivariance principle]
- **The principle (search-level, well-established):** enforcing translation-equivariance means the learned
  dynamics do NOT depend on ABSOLUTE position (invariant under shifting position) -> by Noether, conserves
  linear momentum. LieConv / equivariant nets achieve this "to machine epsilon".
- **WHY IT IS WRONG FOR US (critical, decisive):** our augmentation MUST depend on absolute position:
  (a) COGGING and position-dependent friction are genuine position-dependent residuals we need to learn;
  (b) our whole model is LPV with Y-SCHEDULING -- the dynamics depend on Y-position BY DESIGN, which
  EXPLICITLY BREAKS translation symmetry in Y. So enforcing translation-invariance is a CLASS RESTRICTION
  that kills LPV scheduling and cogging -> fails requirement 2 (full expressivity) AND contradicts the model
  structure. This is the same "structural constraint restricts the class" failure as dissipativity
  (dissipativity-limits B5), in symmetry clothing.

### Direction-3 verdict
- **Elegant but wrong, closed with reasons (valuable negative):** symmetry/equivariance/conservation-law
  learning does NOT fit our problem. (1) Exact conservation (3.1) is for CONSERVATIVE systems; ours is
  forced+dissipative. (2) Momentum conservation (3.2) constrains the INTERNAL channel; our drift is EXTERNAL.
  (3) Translation-equivariance (3.3) forbids ABSOLUTE-position dependence, but we REQUIRE it (cogging,
  Y-scheduling) -> class restriction, fails expressivity, contradicts LPV.
- **Requirements:** 3.1/3.3 fail req 2 (class restriction) or are physically inapplicable; 3.2 fails req 4
  (external channel free). None is admissible.
- **Decision link:** DO NOT pursue equivariant/conservation-law architectures. Record as a closed door: the
  free mode being a "symmetry" does NOT mean symmetry-enforcing learning helps, because (a) the system is not
  conservative and (b) we need to break the very symmetry (position-dependence / Y-scheduling) that
  equivariance would enforce.

### Provenance / primary-read status (Direction 3)
- PRIMARY-READ: Muller (arXiv:2209.11661); Dynami-CAL (arXiv:2501.07373, also §5m). 3.3 reasoned from the
  standard equivariance/Noether principle (search-level), applied to our LPV/cogging structure. Quotes
  transcribed from PDF text layers; re-verify character-exact before thesis use.

---

## Direction 7 (R5-driven follow-up, 2026-07-11) -- the scheduling variable is a corrupted/drifting state

**Trigger:** R5 (drift-diagnosis-status.md §5 crit 5) confirmed the pipeline SELF-SCHEDULES off the drifting
Y-state (`M(Y=x[2])`), and Y-drift DETUNES `M(Y)`. Searched this specific insight. **Finding: R5 has a NAME
in the LPV literature -- but the named problem is NOISE, and ours is DRIFT; the distinction is decisive.**

### 7.1 R5 = "corrupted-scheduling" / errors-in-variables LPV identification (a recognized problem)
- **The named problem (multi-source search convergence, TU/e Toth-group lineage):** most LPV identification
  assumes the scheduling signal is noise-free; in practice it is a measured signal and is noise-corrupted.
  Neglecting scheduling noise is an ERRORS-IN-VARIABLES (EIV) problem that BIASES the estimate, and it is
  "more difficult in the LPV case than in the LTI case, since the stochastic noise affecting scheduling
  observations is distorted by nonlinear functions."
- **References [SEARCH-LEVEL -- flagship PAYWALLED, do NOT quote until verified]:**
  - Piga, Cox, Toth, Laurain (?), "LPV system identification under noise corrupted scheduling and output
    signal observations", Automatica 61 (2015) (S0005109815000199, TU/e). THE flagship; ScienceDirect,
    could not primary-read here.
  - IV scheme for closed-loop LPV identification (Automatica 2012); set-membership LPV (Automatica 2011);
    "LPV model identification with an unknown scheduling variable ... robust global approach" (IET 2018);
    observer design for qLPV with UNMEASURABLE scheduling (norm-L2). All SEARCH-LEVEL.
  - Connects DIRECTLY to Direction 4 (Piga-Bemporad closed-loop LPV bias correction, same group/family).
- **qLPV terminology [PRIMARY-READ, arXiv:2505.07287, Mulagaleti-Bemporad 2025, L28]:** "In qLPV systems,
  a.k.a. self-scheduled LPV systems, the dynamics are described by linear [models whose scheduling is a
  function of the state]." Confirms our pipeline is qLPV/self-scheduled. (That paper is control-invariant-set
  oriented, not R5; only the terminology is used here.)

### 7.2 THE CRITICAL DISTINCTION -- noise (their problem) vs drift (ours). Do not conflate.
The corrupted-scheduling literature treats the scheduling error as **stochastic NOISE on the OBSERVED
scheduling during IDENTIFICATION** (`p_measured = p_true + noise`), and removes the resulting bias
(bias-correction / IV, assuming noise statistics). **Our R5 failure is DIFFERENT on two axes:**
1. **Systematic DRIFT, not stochastic noise.** Our scheduling error is the model's slow, accumulating
   free-integrator DRIFT -- deterministic and growing, not i.i.d. noise. Noise-statistics-based
   bias-correction/IV do not obviously apply to a deterministic accumulating corruption.
2. **INFERENCE-time self-scheduling, not identification-time observation noise.** Their problem: the
   scheduling DATA is noisy while you IDENTIFY. Our problem: at FREE-RUN, the model's OWN propagated Y-state
   drifts, so its OWN `M(Y=x[2])` self-scheduling is evaluated at a wrong Y -- a closed-loop
   drift->detune->wrong-dynamics feedback at inference, not a data-corruption-at-fit issue.
So the named literature gives R5 a HOME and shows it is a recognized LPV difficulty, but it does NOT directly
solve our version. Our R5 (inference-time self-scheduling drift of a free-integrator scheduling state) is
arguably a NOVEL/harder case: the corruption IS the drift we are trying to fix, and it is deterministic.

### 7.3 Where it IS useful (honest)
- **Real-data baseline fit:** the real Telica Y is a measured (noisy) scheduling signal -> the
  corrupted-scheduling / IV methods (7.1, and Direction 4 Piga-Bemporad) ARE the right tools to identify the
  LPV baseline consistently from real data despite noisy Y. This is a real-data-pipeline win, same slot as
  Direction 4.
- **Framing:** lets us STATE R5 in standard terms ("self-scheduled LPV with a drifting scheduling state /
  a deterministic errors-in-scheduling problem at inference"), and cite the recognized EIV-LPV difficulty as
  precedent -- while claiming the drift/inference-time version as the harder, less-covered case.
- **NOT a solution to the sim self-scheduling drift.** Do not present corrupted-scheduling ID as fixing our
  drift; it addresses a different (identification-time, stochastic) failure mode.

### Direction-7 verdict
- R5 = a recognized LPV problem (corrupted scheduling / EIV-LPV), TU/e-group lineage -> gives framing + real-
  data-baseline tools (ties to Direction 4). BUT their problem is stochastic-noise-at-identification; ours is
  deterministic-drift-at-inference-self-scheduling -> NOT directly solved; our version is the harder,
  less-covered case. Flagship (Automatica 2015) is PAYWALLED -- verify + quote before any thesis use.
- **Decision link:** (a) adopt the "errors-in-scheduling / corrupted-scheduling" framing for R5;
  (b) use corrupted-scheduling IV/bias-correction for the REAL-DATA baseline fit (with Direction 4);
  (c) do NOT expect it to fix the sim self-scheduling drift -- that stays the open R5 sub-problem;
  (d) PRIMARY-READ the Automatica 2015 flagship when a non-paywalled copy is available.

### Provenance / primary-read status (Direction 7)
- PRIMARY-READ: Mulagaleti-Bemporad qLPV-CIS (arXiv:2505.07287, terminology only).
- SEARCH-LEVEL, PAYWALLED, DO NOT QUOTE until verified: Piga-Cox-Toth corrupted-scheduling LPV (Automatica
  2015); IV closed-loop LPV (Automatica 2012); set-membership LPV (2011); robust-global unknown-scheduling
  (IET 2018); qLPV unmeasurable-scheduling observer.

---

## Direction 8 (full-requirement search + AUTHORITATIVE survey, 2026-07-11) -- the gap is confirmed by a 2025 survey

**Trigger:** user wants the LITERATURE route to find an ML-for-control method matching ALL FIVE requirements
(incl. R5). Searched the full-requirement intersection + the closest physical analog (free-floating base:
pose drifts AND configures dynamics = R5 analog), then PRIMARY-READ a comprehensive 2025 survey as the
authoritative check.

### 8.1 Full-requirement / free-floating-base search -- NO match
- The free-floating-base analog (base pose drifts AND configures inertia = R5) is real, but the methods are
  CONTROL (PINN-MPC for space robots) or momentum-based ESTIMATION (momentum conservation -- which we already
  found leaves the external channel free, §5m Dynami-CAL) or inverse-dynamics CONTROL -- NOT a marginal-
  preserving FORWARD model with friction + no-drift. No match.
- Physically-consistent GP / Lagrangian-GP (2405.17199 dissipative-GP friction; 2406.03224 projector control)
  and structure-preserving / Lie-group variational integrators (Duruisseaux; 2403.10070) -- all CONSERVATIVE
  or CONTROL, same walls as Direction 3. No match.

### 8.2 AUTHORITATIVE survey -- confirms the gap  [PRIMARY-READ]
**Reference.** S. Sivaranjani, Y. Shi, N. Atanasov, T. Duong, J. Feng, T. Martin, Y. Xu, V. Gupta,
F. Allgower, "Control-Oriented System Identification: Classical, Learning, and Physics-Informed Approaches",
2025 (arXiv:2512.06315). Comprehensive survey by leading control groups (UCSD, Purdue, Stuttgart).
- **The core limitation it names (Abstract, verbatim):** machine-learning system identification's "utility
  in control applications is limited by their ability to provide provable guarantees on control-relevant
  properties." It surveys EXACTLY the property families we studied: "dissipativity, monotonicity, energy
  conservation, and symmetry-preserving structures."
- **The open challenge (verbatim, §4.1):** "some control-relevant properties such as stability and passivity
  can be directly embedded through parameterization ... However, capturing more complex physics-informed or
  control-relevant properties through identifiable parameterizations REMAINS AN OPEN CHALLENGE."
- **Time-varying / LPV physics-preserving ID = explicit FUTURE WORK (verbatim, §7.3):** "control-informed
  system identification for [switched and time-varying] systems is an important direction for future work."
  The survey cites our supervisors' group (Verhoek et al. LPV) as the LPV state of the art -- i.e. the LPV
  physics-preserving ID line is active and OPEN, not closed.
- **No method in the survey matches our 5-requirement combination** (marginal free-integrator + self-
  scheduling LPV + friction-permitting + no-drift + full expressivity). The survey's property toolkit is the
  same one we exhaustively triaged (dissipativity/passivity/monotonicity/symmetry/energy) -- none native to a
  free-integrator + drifting-self-scheduling forward model.

### Direction-8 verdict -- the search route has reached an AUTHORITATIVE endpoint
- **The 5-requirement combination is genuinely absent from the literature, now CONFIRMED by a comprehensive
  2025 survey**, not just by our own searches. The survey explicitly frames property-preserving time-varying/
  LPV identification as OPEN future work, and states that embedding complex control-relevant properties via
  identifiable parameterizations "remains an open challenge."
- **This is a THESIS-POSITIVE negative result:** the gap is real and now CITABLE to an authoritative survey.
  The contribution (marginal-preserving + self-scheduling-robust + friction-permitting + no-drift LEARNED LPV
  forward augmentation) sits in an explicitly-open area.
- **Search saturation reached (multiple independent angles + an authoritative survey converge on the same
  negative).** Continuing keyword search has diminishing returns; the value now is in BUILDING the empirical
  solution (D-107 layers) and FRAMING the contribution against this survey. Any further reading should be
  TARGETED (verify the paywalled corrupted-scheduling flagship; primary-read Verhoek LPV consistency), not
  broad.

### Provenance / primary-read status (Direction 8)
- PRIMARY-READ: Sivaranjani et al. control-oriented-SysID survey (arXiv:2512.06315) -- quotes transcribed
  from the PDF text layer; re-verify character-exact before thesis use.
- SEARCH-LEVEL: free-floating-base / space-robot PINN-MPC, dissipative-GP friction (2405.17199), structure-
  preserving/Lie-group integrators -- triaged, no match, not primary-read.

---

## Direction 9 (LPV cost function / SUBNET training, 2026-07-11) -- what loss does the LPV literature use?

**Trigger:** user -- "we currently just have MSE on the output; look into LPV cost functions / SUBNET training."
PRIMARY-READ the cost function of our own framework's LPV paper + the LPV-training cluster.

### 9.1 Our current loss IS the framework-native LPV cost (truncated multi-step), NOT naive single-step MSE
**Reference [PRIMARY-READ]:** Verhoek, Beintema, Haesaert, Schoukens, Toth, arXiv:2204.04060 (our framework).
- **The cost (Eq 7a, verbatim):** `V_trun = (1/(T(N-T+1))) sum_t sum_{k=0}^{T-1} ||yhat_{t+k|t} - y_{t+k}||^2`
  -- a TRUNCATED prediction loss: multiple truncated subsections of length T, iterate the model T steps,
  average the OUTPUT error. "If the truncation length is set to T=N, then [full simulation error] is
  recovered." So the LPV-SUBNET cost is the TRUNCATED MULTI-STEP output MSE = exactly our nf-window loss
  (T=nf). **Clarification for the user: we are NOT on naive single-step MSE; the nf-window SUBNET loss IS
  the standard LPV-SUBNET truncated multi-step cost.** The literature has no fundamentally different BASE cost.
- **Innovation/predictor form (Eq 7b-e):** the model carries `K_theta(phat) ehat` with `ehat = y - yhat`
  (measured-output correction each step) -- a PREDICTOR, not a pure simulator. This BOUNDS drift DURING
  TRAINING via measured y, but our deliverable is FREE-RUN SIMULATION (no measured y) -> train(predictor)/
  inference(simulator) mismatch = EXPOSURE BIAS (Direction 1). So the prediction-vs-simulation cost choice IS
  the drift-relevant axis.
- **Scheduling reads measured y (verbatim, Eq 7e):** `phat_{t+k|t} = phi_eta(xhat, u, y)` -- the scheduling
  map takes the MEASURED OUTPUT as an input, not purely the propagated state. This is a literature-grounded
  version of exogenous/de-drifted Y-scheduling (Layer 3, R5).

### 9.2 What the literature ADDS beyond the truncated-MSE (the actual answer to "beyond MSE")
- **State-consistency regularization** [Sertbas-Kumbasar 2510.24757 Eq 13, Direction 5]: penalize
  encoder-vs-propagated-state discrepancy -> reduces drift. KEEP (reject its Schur). = Layer 1 add-on.
- **Innovation/predictor vs simulation-error** [Verhoek Eq 7b-e]: the cost's predictor-vs-simulator choice is
  THE drift lever (predictor hides drift via measured y; simulation exposes it; our deliverable is simulation).
- **Exogenous / measured-y scheduling** [Verhoek Eq 7e phi(x,u,y); Olucha-Preda-Das-Toth "Learning Surrogate
  LPV SS with UQ", arXiv:2603.29532: "self-scheduled LPV models generate the scheduling [internally] ... [vs]
  exogenous scheduling variables"]. = literature-grounded Layer 3 (R5): schedule off measured/de-drifted Y.
- **Velocity-form LPV embedding** [SEARCH-LEVEL, S2405896323004846 / autoencoder LPV embedding; velocity-form
  for global stability]: models state-DIFFERENCES -> removes the integrator -> stability guarantees. BUT this
  is VELOCITY-DOMAIN-adjacent -> gated by the supervisor LAST-RESORT constraint (top of drift-diagnosis-
  status.md); do NOT adopt as the loss without go-ahead. Flag only.
- **Data-silent projection + orthogonal projection** = OUR additions (not in the LPV-cost literature).

### 9.3 The gap in LPV COST functions (still open)
No LPV cost function in the literature has a **scheduling-aware DRIFT term** for a self-scheduling variable
that drifts (R5). The truncated-MSE + innovation + state-consistency are all generic; none targets the
drifting-self-scheduling-Y. So a **scheduling-integrity cost term** (penalize `M(Y)` corruption / keep the
Y-pole) would be novel -- part of the R5 contribution.

### Direction-9 verdict
- **We are already on the correct BASE cost** (truncated multi-step output MSE = LPV-SUBNET, Verhoek Eq 7a).
  The user's "just MSE" = the standard truncated-multistep cost; not a deficiency, it is the framework cost.
- **Actionable cost upgrades, all supervisor-group-grounded:** (a) add STATE-CONSISTENCY regularization
  (Sertbas-Kumbasar) = Layer 1; (b) use EXOGENOUS / measured-y SCHEDULING (Verhoek Eq 7e; Olucha) = Layer 3
  (R5); (c) the prediction-vs-simulation cost choice is the drift lever. These map DIRECTLY onto the
  construction spec (`all-five-construction-spec.md`) Layers 1 and 3.
- **Do NOT adopt velocity-form** (LAST-RESORT-gated). **Novel piece:** a scheduling-integrity cost term (R5).

### Provenance / primary-read status (Direction 9)
- PRIMARY-READ: Verhoek et al. LPV-SUBNET cost (arXiv:2204.04060, Eq 7); Olucha-Preda-Das-Toth surrogate-LPV-UQ
  (arXiv:2603.29532, self-vs-exogenous scheduling). Quotes from PDF text layers; re-verify before thesis use.
- SEARCH-LEVEL: velocity-form / autoencoder LPV embedding (S2405896323004846); meta-learning physically-
  constrained sysid (2501.06167); "do we need a state estimator" (2206.12928). Triaged, not primary-read.

---

## Direction 10 (post-sweep targeted follow-up, 2026-07-11): the qLPV estimated/unmeasurable-scheduling sub-community (the R5 thread)

**Trigger:** the one thread the completed sweep left open for R5: primary-read the corrupted-scheduling
flagship (previously SEARCH-LEVEL/paywalled) and its surrounding sub-community where the model runs with a
WRONG scheduling value (unmeasurable premise variables / estimated parameters). Goal: a scheduling-mismatch
propagation bound, and precedent for the Layer-3 de-drifted/exogenous scheduling lever. All PDFs below are in
`literature/corrupted-scheduling/` (plus the LPV-SUBNET PDF in the session scratchpad).

### 10.1 The flagship, now PRIMARY-READ: Piga, Cox, Toth, Laurain, Automatica 53 (2015) 329-338
- **Source:** author PDF from rolandtoth.eu (`AUT2015b.pdf`), "DRAFT. Article published in Automatica, Vol. 53,
  2015". Saved as `piga2015_noise-corrupted-scheduling-automatica.pdf`. Quotes transcribed from the PDF text layer.
- **Setting (verbatim, Section 2.2):** the scheduling observation is `p(k) = po(k) + eta_o(k)` where eta_o "is a
  Gaussian distributed white noise process with zero-mean and finite variance", independent of the output noise.
  Method: a bias-corrected IV estimate; the correction matrices Psi_k are built from the noise MOMENTS
  (`po^2 = E{p^2} - sigma_eta^2`, `po^3 = E{p^3 - 3 p sigma_eta^2}`, ...), with sigma_eta^2 known or estimated
  by gridding a bilinear equation set (Section 5).
- **The transferable object, the STRUCTURAL BIAS decomposition (Eq. 13-14, 16):** the IV estimate splits into
  the true parameter plus noise terms plus the term (14d)
  `B_Delta = Gamma_N^{-1} sum (1/N) z(k) [chi(k) (x) (po(k)-p(k))]^T theta_o`, "referred in the sequel as
  structural bias". This isolates EXACTLY how a scheduling error (for them noise, for us drift) biases the
  estimate. A deterministic-drift analog of B_Delta is a clean template for writing the R5 training-bias term.
- **Why the fix does NOT transfer to R5 (now verified at primary level, upgrading the Direction-7 verdict from
  search-level to primary-read):** every correction step uses that the scheduling error is ZERO-MEAN, WHITE,
  with known/estimable moments, at IDENTIFICATION time. Our R5 corruption is a deterministic, growing,
  inference-time drift of the model's own propagated Y; none of the moment machinery applies. Remark 4 adds
  that the multi-dim scheduling extension is exponential in n_p.
- **How big the effect is (their sim, Table 1-2):** at SNR_p = 21 dB, ignoring scheduling noise gives "a bias
  which, in some cases, has the same magnitude as the true value of the parameters"; validation BFR 45% (IV)
  vs 96% (bias-corrected). Scheduling corruption is a first-order failure mode, not a refinement: supports
  taking R5 seriously in the thesis motivation.
- **Real-data slot confirmed:** for the Telica baseline fit with measured (noisy) Y this method IS applicable
  (same slot as Direction 4, Piga-Bemporad closed-loop bias correction; same author family).

### 10.2 TS/qLPV observers with unmeasurable premise variables: Ichalal, Marx, Ragot, Maquin (MED 2012) [PRIMARY-READ, HAL hal-00684701]
- **This is the formal home of "the model is evaluated at the WRONG scheduling".** System self-schedules on the
  state (`mu_i(x)`), the observer/model schedules on the ESTIMATE (`mu_i(x_hat)`). The error dynamics become
  (their Eq. 7-10): `e_dot = Phi e + delta(x, x_hat, u)` with `delta = (A_mu - A_muhat) x + (B_mu - B_muhat) u`.
  That delta IS the detune term of R5 (our `M(Y_drifted)` vs `M(Y_true)`), given a name and a treatment.
- **Two bounding regimes:** (1) Lipschitz: `delta^T delta <= eta^2 e^T e` yields asymptotic convergence via LMIs
  (their Theorem 1); (2) ISS: bounded mismatch gives a bounded error with EXPLICIT gain,
  `||e(t)|| <= sqrt(alpha2/alpha1) ||e(0)|| exp(-alpha t/2) + sqrt(c/(alpha alpha1)) ||delta||_inf`
  (their Eq. 34), "the proposed observer can take into account uncertainties on the premise variables" (Rem. 4).
- **The structural caveat for us (the load-bearing point):** every bound is built on error dynamics made
  exponentially stable by OUTPUT INJECTION (`P^{-1} L (y - y_hat)`). In a pure free-run there is no measured y
  to inject, and exponentially stabilizing the Y-error would damp the marginal pole (violates R3). So this
  machinery quantifies detune PROPAGATION (drift of size epsilon corrupts the dynamics by a bounded, computable
  amount) but does NOT bound free-run drift itself. It is analysis language + a bound template, not a fix.
- Related journal versions on disk: `ts_state_est.pdf` (IET CTA 2010), `ts_L2.pdf` (IFAC WC 2008), same lineage.

### 10.3 LPV with estimated parameters: Millerioux, Rosier, Bloch, Daafouz, IEEE TAC 49(8) 2004 [primary-OBTAINED, HAL hal-00121005; text layer CORRUPTED, do NOT quote verbatim]
- The LPV-native (polytopic, discrete-time) version of 10.2: state reconstruction when the scheduling parameter
  is known only "with a finite accuracy" (`|rho - rho_hat| <= delta`). Error dynamics driven by
  `d_k = (A(rho_k) - A(rho_hat_k)) x_k`; boundedness is NOT automatic (a bounded disturbance can drive a
  nonlinear system unbounded); an explicit bound is derived via ISS with a poly-quadratic Lyapunov function.
- Same caveat as 10.2: requires the (observer) error dynamics to be exponentially stable. For us its value is
  the FRAMING: a scheduling error acts exactly like a bounded unknown exogenous input, and boundedness of the
  state under it is a property that must be PROVEN, never assumed. This formalizes why Y-drift detuning M(Y)
  is dangerous rather than benign.
- The PDF's embedded fonts produce mojibake in the text layer (PyMuPDF and pypdf both). Structure verified by
  reading equations/numbering; for thesis quotes use the IEEE published version (10.1109/TAC.2004.832669).

### 10.4 LPV-SUBNET self- vs EXTERNAL scheduling: Verhoek, Beintema, Haesaert, Schoukens, Toth (arXiv:2204.04060) [PRIMARY-READ, on disk]
- **The Layer-3 decision has an in-framework donor.** The supervisor-group paper defines BOTH variants:
  - Self-scheduling (their Fig. 1): the p-net "uses the previously calculated x_hat_{t+k|t} to compute the
    scheduling for the next state update. This enforces that the LPV-SS model state can be used to determine
    the scheduling, which is often called self-scheduling in the LPV literature." (= our current pipeline,
    Y_op=None.)
  - External scheduling (their Fig. 2): "Alternatively, the encoder psi_xi can be used to estimate the possibly
    required x_hat_k in each time-step, separating the scheduling map calculation from the forward propagation
    of the model. This formulation considers the scheduling as an external signal determined by a filter
    operation directly from the data-generating system, which is in line with the intended use of the model
    for analysis and control purposes."
- So exogenous/measured scheduling is not a hack around R5; it is one of the two standard formulations in the
  framework's own paper, and the one the authors tie to "the intended use of the model". Quotable support for
  the supervisor discussion. CAVEAT to state honestly: with external scheduling, validation is no longer a pure
  free-run from u alone (the scheduling channel is fed from data); that is standard LPV simulation practice
  (scheduling is part of the data tuple), but it changes what the free-run metric means on the sim.
- **The gap stands:** the paper gives consistency (inherited from SUBNET, assuming PE + global optimum) and NO
  analysis of self-scheduled free-run robustness to scheduling corruption. Nothing here bounds our drift.

### Direction-10 verdict
- **R5 gains three concrete pieces:** (a) the structural-bias decomposition (10.1) as the template for writing
  the deterministic drift-induced bias term; (b) the detune-propagation bound machinery (10.2/10.3): IF the
  augmentation bounds Y-drift to epsilon, THEN the scheduling corruption acts as a bounded disturbance with a
  computable effect, i.e. a quantitative R4-implies-R5-integrity argument for the thesis; (c) the external-
  scheduling precedent (10.4) as the supervisor-group-grounded donor for Layer 3.
- **What the thread does NOT give (consistent with D-108):** no mechanism bounds free-run drift on a marginal
  pole. All bounds in this community hinge on exponentially stable error dynamics obtained via output
  injection, which a pure free-run does not have and R3 forbids creating on the Y pole. The construction
  spec's Layers 1-2 remain the R4 mechanism; this thread strengthens the R5 layer AROUND them.
- **Single next action from this direction:** bring the Y-scheduling decision (self- vs external) to the
  supervisor WITH the 10.4 quotes and the 10.2 caveat; it is the keystone of Layer 3 and now fully briefed
  from primary sources.

### Provenance / primary-read status (Direction 10)
- PRIMARY-READ, quotes from PDF text layer: Piga-Cox-Toth-Laurain (Automatica 2015, author PDF, rolandtoth.eu);
  Ichalal-Marx-Ragot-Maquin (MED 2012, HAL hal-00684701); Verhoek et al. LPV-SUBNET (arXiv:2204.04060).
- primary-OBTAINED but NOT quotable (corrupted text layer): Millerioux-Rosier-Bloch-Daafouz (TAC 2004, HAL
  hal-00121005); structure verified, quote only from the IEEE version after independent check.
- On disk, NOT yet read: `ts_state_est.pdf` (IET CTA 2010 journal version), `ts_L2.pdf` (IFAC 2008).
- SEARCH-LEVEL leads deliberately not pursued (saturation): IET 2018 unknown-scheduling robust-global; Heemels-
  Daafouz-Millerioux TAC 2010 (observer-based CONTROL, not ID); adaptive sliding-mode FDI with uncertain
  scheduling.

---

## Direction 11 (broadened search, 2026-07-11): aerospace / NASA qLPV + the gain-scheduling "hidden coupling" vocabulary

**Trigger:** user -- do not limit the search to the TU/e lineage; aerospace/NASA should have examples. New
search-matrix cells: gain-scheduling classic ("hidden coupling terms"), aerospace qLPV, NASA NTRS (fully
open-access). PDFs in `literature/aerospace-qlpv/` and `literature/theses-lpv-lineage/`.

### 11.1 Schuet, Malpica, Aires (NASA Ames), "A Gaussian Process Enhancement to LPV Models", AIAA Aviation 2021 [PRIMARY-READ, NTRS 20210017417]
- **What it is:** stitched full-envelope qLPV models (anchor-point linearizations) where the GP REPLACES the
  linear interpolation of `A(nu), B(nu), x_t(nu), u_t(nu)` -- the learned part is the SCHEDULING-DEPENDENCE
  of the model matrices, not a parallel dynamic residual. Self-scheduled ("In the quasi-LPV (qLPV) case the
  look-up parameter vector nu has elements that depend on x"). Demo: NASA electric quad-rotor air-taxi,
  nu = airspeed, LQR + mu-analysis (robstab) robustness vs GP uncertainty at frozen airspeeds.
- **The R5-relevant find (verbatim, Section III.B):** for the qLPV linearization they derive the chain-rule
  term `Jf_x = A(nu) + Jf_nu Jnu_x - Jg_xt Jxt_nu Jnu_x` and state: "With the GP approach, one may also
  intentionally neglect the parasitic term ... and treat it as a contributor to the uncertainty in A(nu)."
  -> independent, quotable precedent for TREATING THE SELF-SCHEDULING (hidden-coupling/detune) TERM AS
  BOUNDED MODEL UNCERTAINTY with a data-derived magnitude, the same analysis slot as Direction 10's
  drift->detune bound, from a non-TU/e community.
- **Layer-2 resonance (verbatim, Section II.G):** the GP predictive variance "depends only on where the data
  is observed, not what data is observed" -- an independent formalization of "data-silent direction =
  location-of-data property", the exact premise of `data-silent-regularization-concept.md` (GP prior =
  Bayesian sibling of the projection; consistent with Rogers-Friis in concept §8).
- **What it is NOT:** no long-horizon free-run analysis, no drift treatment, closed-loop frozen-point
  robustness only; scheduling is airspeed (damped velocity states), NOT a marginal/integrator state; not a
  parallel augmentation. **Does not solve any of the five; gives R5 analysis language + Layer-2 precedent.**

### 11.2 The "hidden coupling terms" vocabulary [SEARCH-LEVEL -- named vein, not yet primary-read]
- Classic gain scheduling names R5's mechanism: endogenous (state) scheduling creates "hidden coupling
  terms" / parasitic feedback that frozen-scheduling analysis misses -- "often omitted, which might result in
  performance degradations or even destabilization" (search-level). Lineage: Shamma-Athans; Rugh-Shamma
  survey (Automatica 2000); Lhachemi et al. pitch-axis missile (JGCD/TCST ~2016-17).
- Their slow-variation stability bounds (stability if the scheduling state varies slowly enough) may fit the
  free-run drift regime (drift IS slow) -- untested transfer, flagged for a targeted read if R5 must be
  solved under self-scheduling.
- Also archived, unread: Shin NASA CR-2007-213926 (qLPV valid over NON-TRIM regions -- relevance: what
  happens to the embedding along drifted trajectories); Hanema tube-MPC x2 (scheduling tubes = future
  scheduling in a set sequence, open-loop-over-horizon propagation WITHOUT output injection); Cox thesis
  §6.2.5 (PEM under scheduling noise, beyond the 2015 IV paper); Verhoek thesis 2025 (guarantee chapters).

### 11.3 Cox PhD thesis (TU/e 2018, pure.tue.nl open access) -- the scheduling-noise sections [PRIMARY-READ, targeted]
- **§6.2.5 ("PEM under noise-corrupted scheduling measurement") adds NO new mechanism beyond the
  Automatica 2015 paper**: same bias-corrected-IV construction (Psi_t moment matrices, augmented-instrument
  bilinear equations, sigma^2-grid), same simulation system, in thesis form. The Direction-10 verdict on the
  flagship transfers unchanged.
- **THE quotable boundary statement (§11.3.1, verbatim):** "In this thesis, the proposed bias correction
  scheme can only handle white Gaussian noise on the scheduling signal in a static, polynomial IO model,
  where the noise on the scheduling signal is independent of the additive output noise... To tackle the EIV
  problem under general noise conditions, these concepts should be extended to be able to: 1) handle other
  scheduling dependency structures, 2) to handle correlation between the scheduling noise signal and the
  output noise sequences, and 3) to handle a colored noise process on the scheduling signal." Also future-work
  item 1: "Investigate the joint EIV problem w.r.t. colored input and scheduling noise sequences."
  -> Our R5 corruption is the EXTREME of the declared-open cases: deterministic+growing (beyond "colored"),
  perfectly correlated with the model's own state error (beyond "correlated"), at INFERENCE (beyond
  identification). The group's own thesis places it beyond the 2018 state of the art -- primary-read support
  for the D-108 gap on the R5 axis, from inside the supervisors' lineage.
- **Secondary asset (Ch. 3, Subgoal-6 summary):** robust quadratic stability / performance LMI tests for
  DT LPV systems "that exhibit bounded rates of variation on their scheduling signals" (affine
  scheduling-dependent Lyapunov, partial-convexity extension). Same family as the Hanema scheduling tubes:
  in-lineage analysis machinery for "slowly varying scheduling" -- and free-run drift IS a slowly varying
  scheduling error. Candidate tool for the R4->R5 detune-propagation argument; NOT a fix (analysis only,
  and asymptotic stability is its premise -- the marginal pole sits outside; check applicability at read
  depth before any use).

### 11.4 Hanema-Toth-Lazar scheduling tubes [PRIMARY-READ: CDC 2016 full; Automatica 2017 targeted scan]
- **The make-or-break question (pre-declared): does the tube stay bounded without control? ANSWER: NO.**
  The tube (CDC16 Def. 1) is a CONSTRAINT-invariant tube: cross-sections X_{i|k} together with CONTROL LAWS
  Pi_{i|k} such that `A(theta)x + B Pi(x,theta) in X_{i+1|k}` for all theta in the scheduling tube -- the
  boundedness is synthesized by choosing vertex control actions, and recursive feasibility/stability rest on
  a CONTROLLED rho-contractive terminal set (Def. 2; Automatica 2017: "controlled periodically contractive").
  With no input to choose (our open-loop free-run) and a marginal A, the guarantee machinery does not apply.
  The third R5 leg is therefore PARTIAL: description language transfers, the guarantee does not -- consistent
  with D-107 (closed-loop/control is exactly what hides or fixes the drift; remove it and no bound remains).
- **What DOES transfer (the asset):**
  (a) **The scheduling-tube FORMALISM** (CDC16 Assumption 2 + Fig. 1): future scheduling confined to a set
  sequence around a nominal trajectory -- "uncertainty around a nominal parameter trajectory or known bounds
  on the rate of variation" -- is the right description language for R5's drifting Y (true scheduling in a
  tube around the model's propagated Y, or vice versa). The LPV-C / LPV-A / LPV-O taxonomy (unknown /
  tube-bounded / exactly-known future scheduling) cleanly frames the Layer-3 exogenous-vs-self-scheduled
  discussion.
  (b) **Finite-horizon set-valued propagation without control is still computable** (the per-step map
  `X_{i+1} = closure of A(Theta_i) X_i` needs no controller): over the FINITE 12 s validation window this
  gives a computable, non-asymptotic detune-envelope bound. No infinite-horizon claim is possible on the
  marginal pole -- state it as a finite-horizon argument only.
  (c) **Quotable motivation match (CDC16 Introduction, verbatim):** "motion systems, where the relevant
  scheduling variables often correspond to a position which approximately tracks a pre-defined reference
  trajectory" -- the community's own motivating case for scheduling tubes IS a position-scheduled motion
  system, i.e. our gantry Y.
- Files: `literature/theses-lpv-lineage/hanema2016_anticipative-tube-mpc-cdc.pdf`, `hanema2017_stabilizing-tube-mpc-automatica.pdf`.

### 11.5 NASA machine-learning-for-flight families [SEARCH-LEVEL addendum, 2026-07-12]
- **Neural adaptive flight control (IFCS Gen-II, NF-15B; RESTORE/X-36):** a FLOWN physics+NN augmentation --
  sigma-pi NN adapting on the model-following error of a dynamic-inversion controller, flight-tested through
  simulated failures. Closest real-world instantiation of "physics baseline + learned residual" found
  anywhere; but it is DIRECT-ADAPTIVE and CLOSED-LOOP: the NN corrects error inside a feedback loop, weight
  robustness comes from the adaptive-control mods (sigma/e-mod/limiters = the §5m Family-A toolbox), and the
  restricted sigma-pi class + limiters trade away R2. No open-loop free-run exists in their setting.
- **Learn-to-Fly (Langley, Morelli):** real-time onboard global aerodynamic model ID (frequency-domain
  derivatives + global polynomial models) fused with adaptive control -- continuous re-identification from
  fresh measurements is the ultimate re-seeding; drift is corrected by data inflow by construction.
- **Verdict:** NASA's ML-for-flight corpus repeats the universal pattern: every fielded system obtains its
  guarantee from FEEDBACK or CONTINUOUS RE-MEASUREMENT (D-107's mechanism), which our free-run deliverable
  deliberately forbids. No five-requirement method. Assets: a flown precedent for the augmentation CONCEPT
  (motivation), and independent confirmation that the drift problem is universally solved by closing a loop
  -- making the open-loop gap sharper, not narrower.

### Direction-11 verdict
- The broadened (non-TU/e) search CONFIRMS the D-108 pattern rather than overturning it: independent
  communities name and analyze R5's mechanism (hidden coupling; parasitic-term-as-uncertainty; scheduling
  tubes) and provide precedent for LPV+learned+UQ (NASA GP-qLPV), but none bounds a free-run on a marginal
  self-scheduled state and none is an augmentation method. Two NEW quotable assets: the Schuet
  parasitic-term-as-uncertainty move (R5 analysis) and the GP-variance data-location line (Layer-2 premise).
- Primary-read queue remaining (all on disk): Cox §6.2.5 -> Hanema tubes -> Shin non-trim -> Verhoek
  guarantee chapters; plus targeted Rugh-Shamma/Lhachemi if self-scheduling must be kept.

---

## Direction 12 (post-diagnosis ML round, 2026-07-12): the d8-d12 diagnosis translated into ML vocabularies -- ALL ABSTRACT/SEARCH-LEVEL, none primary-read yet

**Trigger:** user -- re-run the ML search with the fields/insights found this session. The d8-d12 diagnosis
(DC is loss-NEUTRAL; the ANN compensated a training-geometry artifact; drift = exploitation of a data-silent
direction) yields three ML translations that Direction 1 (rollout stability) could not have searched, because
the diagnosis did not exist yet. Everything below is abstract-level [verify at source before any use].

### 12.1 Model-based RL: "model exploitation" + uncertainty-penalized rollouts -- the strongest new cell
- **Their known failure IS our measured failure:** MBRL rollouts compound model errors, and optimization
  EXPLOITS model errors in low-data regions ("model exploitation"). Our d12 finding (training wanders into a
  loss-neutral DC; free-run integrates it) is this phenomenon with the policy replaced by the fit.
- **Their standard mitigation IS our training geometry:** MBPO (Janner et al., arXiv:1906.08253) uses SHORT
  rollouts branched from REAL states specifically to avoid compounding -- structurally identical to our
  windowed encoder-re-init training. MBRL adopted our geometry deliberately AND still needed a second
  mechanism, which independently supports "conditioning alone is not enough" (consistent with d8).
- **Their second mechanism is a Layer-2 sibling with a THEOREM: MOPO** (Yu et al., NeurIPS 2020,
  arXiv:2005.13239): penalize the reward by ensemble-disagreement model uncertainty; the penalized (pessimistic)
  MDP's return LOWER-BOUNDS the true return. I.e., "penalize where the data is silent" with a guarantee shape.
  [search-level; primary-read before citing the bound.]
- **NEW Layer-2 IMPLEMENTATION OPTION (flag for the build):** ENSEMBLE DISAGREEMENT as the data-silence
  estimator. The limits doc (C1-C4) flags the SVD-of-Fisher construction of `Pi_low` as the unsolved design
  step for a nonlinear residual; ensemble disagreement (k small ANN heads, disagreement = epistemic
  uncertainty = data-silence) is a scale-proven, ANN-native, data-derived alternative estimator of the same
  quantity. Candidate resolution of concept-note §7 / limits C4. ALSO connects to Schuet §II.G (GP variance
  = data-location) -- three formalisms (GP variance / ensemble disagreement / Fisher SVD) of ONE quantity.
- Also relevant: "Investigating Compounding Prediction Errors in Learned Dynamics Models" (arXiv:2203.09637)
  -- an empirical anatomy of compounding errors [unread]; M2AC (uncertainty-masked rollouts).

### 12.2 Implicit bias of GD: the null model for a loss-neutral direction
- Theory (linear/overparametrized settings): GD converges to the minimum-norm / closest-to-initialization
  solution in underdetermined directions. **Null prediction for us: zero-init ANN + truly flat DC direction
  => the DC STAYS ~0.** We measured it GROWING (d6/d12) => the direction is not autonomously flat; it is
  DRIVEN by shared-weight coupling from the informed rows (the d12 "shared-net byproduct" reading now has a
  theory-side counterpart). Consequence: implicit regularization alone provably will not hold the DC at zero
  under coupling -- an EXPLICIT mechanism (Layer 2) is required. [Abstract-level synthesis; the cited theory
  is for linear/idealized settings -- do not over-claim transfer.]

### 12.3 Shortcut learning / Clever Hans: the frame for the d9 finding
- The encoder-ramp compensation is a textbook SHORTCUT: a highly-available, low-validity feature (the
  recurring window-init artifact) exploited to reduce training loss, failing off-distribution (free-run).
  The Clever-Hans-correction literature (ClArC, "Class Artifact Compensation"; Anders et al., Inf. Fusion
  2022) corrects models trained on data artifacts -- same problem class, classification-domain methods.
  Value = thesis framing + the term "artifact compensation" for the d9 mechanism; methods likely do not
  transfer directly. [search-level]

### Direction-12 verdict
- The diagnosis-informed translation worked: MBRL gives (a) independent confirmation that short-window
  re-init training is the standard mitigation AND insufficient alone, (b) a guarantee-shaped precedent for
  data-silence penalties (MOPO), and (c) a concrete, scale-proven implementation candidate for `Pi_low`
  (ensemble disagreement) that may resolve the limits-doc C4 blocker. 12.2 upgrades "the DC is a shared-net
  byproduct" from hypothesis language to a theory-backed expectation violation. 12.3 names the d9 mechanism.
- **Primary-read shortlist for AFTER the supervisor checkpoint** (all arXiv-open): MOPO 2005.13239 (the
  bound + penalty form), MBPO 1906.08253 (rollout-length theory), compounding-errors 2203.09637. The
  ensemble-disagreement Layer-2 option goes on the §7 design-choice list EXPLICITLY as an alternative to the
  Fisher-SVD construction.

---

## SWEEP COMPLETE -- consolidated takeaways (all 6 directions)
- **Direction 1 (rollout stability):** the most on-target FRAMING (short-window training vs long free-run),
  and the most-supported UNTRIED direction -- but NOT a proven fix for our drift. Two caveats: (1) TIMESCALE
  -- our drift appears ~0.5 s, above the tested/short exposure horizons, so short unrolling does not reach it;
  (2) TWO-COMPONENT -- rollout methods address distribution-shift, NOT the identifiability half (unexcited DC).
  GNS random-walk noise is the best-matched single tool (simulates long-horizon drift cheaply) BUT risks
  DAMPING the marginal pole (req-3; eigen-check required). -> D-107 as falsifiable EXPERIMENTS, not "unrolling
  solves it".
- **Direction 2 (hybrid identifiability):** adopt PARAMETRIC vs FUNCTIONAL identifiability language; orthogonal
  projection is independently validated (systems biology, UNVERIFIED quote) -> our novelty = the LPV/LFR
  extension + drift tie. -> contribution framing.
- **Direction 4 (bias/IV ID):** closed-loop LPV bias correction (Piga-Bemporad) for the REAL-DATA baseline
  fit; linear-only, not the ANN. -> real-data pipeline.
- **Direction 5 (LPV+ML):** our framework has CONSISTENCY not no-drift (Verhoek/Toth); KEEP the
  state-consistency regularizer, REJECT Schur (damps marginal). -> D-107 conditioning.
- **Direction 6 (drift diagnosis):** run the Jacobian non-normality/commutator check to confirm/refine the d6
  DC-only finding; PDE-aliasing does not transfer. -> a diagnostic to run.
- **Direction 3 (symmetry):** closed door -- forced/dissipative (not conservative) + we need position-
  dependence (cogging/Y-scheduling) that equivariance forbids.
- **Net:** the strongest ACTIONABLE outputs are Directions 1 + 5 (unrolling-first + state-consistency
  regularizer for the D-107 clean re-run) and Direction 2 (contribution framing + verify the npj quote).
  4 is a real-data-pipeline tool; 6 is a diagnostic; 3 is closed. No direction overturns the impossibility
  (structural req 4 vs full expressivity); all actionable fixes give req 4 EMPIRICALLY (solve-not-hide),
  which is the honest ceiling (open-loop-solution-decision.md / D-107).
