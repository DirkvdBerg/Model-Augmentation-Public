# Windowed-BPTT Rollout Drift and DC-Offset Birth: a Cited Synthesis

**Created**: 2026-07-17, from a web literature search (background research agent) requested to
diagnose WHY the augmentation ANN acquires a constant (DC) output during training and how to
document/monitor it per update step. Companion to `claude-deep-research-drift-diagnostics.md`
(identifiability / loss-blindness / committee-facing mechanism demos): this file COMPLEMENTS that
one and does not repeat it. Focus: (1) windowed/truncated BPTT and rollout-horizon vs error
accumulation, (2) early-training emergence of a bias/mean, (3) train-vs-deployment mismatch
monitoring, (4) fixes compatible with "zero at equilibrium + full expressivity", (5) update-step /
gradient-level diagnosis.

**Verification status**: every arXiv ID resolves to a real paper; entries marked (verified) were
independently checked against their primary arXiv/proceedings page by the search agent. Two 2026
Jacobian-regularization preprints (2602.04608, 2603.05538) were surfaced but NOT individually
verified: treat as unvetted until checked. Our setting is NOISELESS, DETERMINISTIC simulation, so
noise-model dependence is flagged throughout.

The pathology in one sentence: a short windowed simulation loss cannot price the free-run
consequence of a constant force on a marginally stable (z=1) integrator, and narrowband excitation
carries no DC information, so the constant-force direction of the learned residual is unidentifiable
and gets parked at a non-zero mean; free-run integration turns that into unbounded position drift.

---

## Gap 1: truncated/windowed BPTT, rollout horizon, and error accumulation

- **Ross, Gordon & Bagnell (2011), "A Reduction of Imitation Learning and Structured Prediction to
  No-Regret Online Learning," AISTATS, PMLR v15.** The theoretical backbone. A policy with per-step
  error epsilon on the visited-state distribution incurs cost growing like **O(epsilon * T^2)** over
  horizon T under naive supervised imitation; DAgger reduces it to O(epsilon * T). A purely
  dynamical, noise-free argument: maps directly onto our **T^2 in-window drift growth** and explains
  why a per-step error invisible at 0.1 s is ~100x worse at 12 s. Our most defensible "why
  superlinear in horizon" citation.
- **Tallec & Ollivier (2017), "Unbiasing Truncated Backpropagation Through Time," arXiv:1705.08209.**
  Truncating BPTT to a fixed window gives a **systematically biased gradient** that favors
  short-term dependencies and underweights long-range ones; ARTBP restores an unbiased estimator via
  randomized truncation with reweighting. Directly relevant: our nf=400 window systematically
  under-prices the slow/DC direction. Note this is *gradient-truncation* bias, distinct from
  input-distribution exposure bias: cite it precisely as such.
- **Bengio, Vinyals, Jaitly & Shazeer (2015), "Scheduled Sampling for Sequence Prediction with
  RNNs," NeurIPS, arXiv:1506.03099.** Names the train/deploy mismatch: trained on ground-truth
  prefixes, at inference the model conditions on its own outputs, so errors accumulate along the
  generated sequence. CAUTION: stochastic framing (token sampling); the mechanism transfers but the
  fix (annealed sampling) is designed for stochastic generation. Also **Ranzato et al. (2016),
  "Sequence Level Training with RNNs," ICLR, arXiv:1511.06732**, which coined "exposure bias".
- **Vicol, Metz & Sohl-Dickstein (2021), "Unbiased Gradient Estimation in Unrolled Computation
  Graphs with Persistent Evolution Strategies," ICML, arXiv:2112.13835.** Formalizes the
  **truncation-bias vs variance/cost tradeoff** in unrolled graphs: full unrolling is unbiased but
  high-variance/expensive; truncated BPTT is cheap but biased. Domain-agnostic; the general theory
  behind "how many unroll steps trades bias for stability," applies to our windowing regardless of
  noise.
- *Adversarial note:* He et al. (2021, EMNLP, arXiv:1905.10617) empirically dispute universal
  compounding ("self-recovery"), but that dissent is explicitly about stochastic open-ended text
  generation and does not transfer to a deterministic integrator with an unexcited DC direction: our
  drift cannot self-recover because there is no restoring force (K=0).

What window length does to a DC term (the honest composite): (a) Ross gives the T^2 horizon
amplification of any per-step bias; (b) Tallec-Ollivier gives *why* a short window under-weights the
slow direction so the bias is not penalized; (c) Vicol gives the bias-vs-cost tradeoff of
lengthening the window. Together they justify "a short window is structurally blind to a
DC/integrating error".

---

## Gap 2: early-training emergence of a bias / systematic mean

- **Rubruck, Bauer, Saxe & Summerfield (2024), "Early learning of the optimal constant solution in
  neural networks and humans," arXiv:2406.17467 (verified).** The strongest direct hit. Empirically
  and theoretically (deep linear nets, CNNs) shows that early in training, model outputs mirror the
  distribution of the target labels while ignoring the input, i.e. the network learns the optimal
  constant (mean/DC) solution first. Closest published support for "the correction is born with a DC
  component before it learns input-dependent structure".
- **Rahaman et al. (2019), "On the Spectral Bias of Neural Networks," ICML, arXiv:1806.08734** and
  the **F-Principle (Xu et al. 2019, arXiv:1901.06523).** Gradient descent fits low-frequency
  components first; the DC/constant is the zero-frequency limit, so the offset is learned earliest.
  Frames it as frequency ordering, not as an operating-point constant: supportive but not on-the-nose.
- **Glorot & Bengio (2010, AISTATS, PMLR v9)** and **He et al. (2015, ICCV, arXiv:1502.01852).**
  Xavier/He init are variance-preservation arguments assuming approximately zero-mean activations;
  they reason about signal variance, not about a nonzero operating-point offset the network must
  produce. Initial outputs are centered near zero, so a required DC level must be *learned in*.
- **Ioffe & Szegedy (2015), "Batch Normalization," ICML, arXiv:1502.03167.** BN subtracts the batch
  mean by construction and only a learned shift beta can restore an offset, so any needed constant is
  actively suppressed until beta learns it. Relevant if any normalization sits in the correction path.

Honest gap flag: there is abundant literature that networks fit the mean/lowest-frequency/simplest
component first, and clear treatment of zero-mean init/BN assumptions, but NO primary source treats
our exact mechanism (a learned parallel dynamics correction acquiring a spurious DC offset early
because the target carries an operating-point constant and the loss does not price it). Treat
DC-birth as a genuine gap we contribute to; assemble it from Rubruck + spectral bias + the init/BN
zero-mean assumption rather than citing it wholesale.

---

## Gap 3: train-vs-deployment mismatch, what to monitor during training

- **Lamb, Goyal et al. (2016), "Professor Forcing," NeurIPS, arXiv:1610.09038.** Canonical "teacher
  forcing hides instability": a model can look correct one-step-ahead yet diverge in free-running
  mode. Transferable diagnostic (independent of their GAN fix): free-run trajectory statistics should
  match the teacher-forced ones; divergence between the two regimes is the instability signal.
  Noise-independent; well-defined in our deterministic sim.
- **Piroddi & Spinelli (2003), Int. J. Control**; **Aguirre, Barbosa & Braga (2010), MSSP**;
  **Farina & Piroddi (2011), Int. J. Adaptive Control & Signal Proc.** The system-identification
  statement of the same theme: one-step-ahead prediction error is not a good indicator of dynamic
  fidelity; simulation (free-run) error is the criterion that matters. Farina-Piroddi motivate
  multi-step prediction as the bridge, so horizon length is a validation knob. Aguirre contrasts the
  *parameter* estimates from PEM vs SEM (relevant to our interpretability goal; partly
  noise-motivated, but the "SEM gives better dynamics" point holds noiselessly).
- **Beintema, Toth & Schoukens (2021, L4DC, PMLR 144)** and **(2023, "Deep Subspace Encoders,"
  Automatica, arXiv:2210.14816).** Our own SUBNET framework: the encoder-initialized truncated
  simulation-error over short subsections IS a short-window approximation of full simulation error.
  Names the exact mismatch we are diagnosing (training = short window, deployment = full rollout).
- **Ribeiro & Aguirre (2020), "On the smoothness of nonlinear system identification," Automatica,
  arXiv:1905.00820** (and Ribeiro et al. 2020, AISTATS, "Beyond exploding and vanishing gradients").
  Explains *why* simulation/multi-step loss surfaces are less smooth than one-step, motivating
  horizon curricula rather than a hard stability class.
- **Pervez & Locatello (2026), "Controlling Transient Amplification Improves Long-horizon Rollouts,"
  arXiv:2605.08856 (verified).** Load-bearing caveat: non-normal Jacobians amplify errors transiently
  even when the system is asymptotically stable, so a spectral-radius monitor alone can miss drift;
  also check Jacobian norm / non-normality. Our K=0 poles at z=1 are exactly the marginal case where
  spectral radius = 1 tells you nothing; transient/DC growth is the real risk.

---

## Gap 4: fixes compatible with "zero at equilibrium + full expressivity"

- **Turan & Jaschke (2022), "Multiple shooting for training neural differential equations on time
  series," IEEE L-CSS, arXiv:2109.06786 (verified).** Split the long trajectory into short segments
  with independent initial states, fit in parallel, re-impose continuity via penalty /
  augmented-Lagrangian gap-closing. Cures the "flattened-out / over-smoothed" minimum that
  single-shooting BPTT collapses into, without imposing any stability class. Tradeoff: extra
  per-segment state variables and a penalty schedule. **Massaroli, Poli et al. (2021), "Differentiable
  Multiple Shooting Layers," NeurIPS, arXiv:2106.03885** casts this as a differentiable root-finding
  layer.
- **List, Chen, Bali & Thuerey (2024), "Differentiability in Unrolled Training of Neural Physics
  Simulators on Transient Dynamics," arXiv:2402.12971 (verified).** Best source on unroll/window
  length vs bias-stability. Disentangles the two effects of unrolling: reducing training distribution
  shift vs providing long-term gradients; finds non-differentiable-but-unrolled (solver-in-loop)
  training recovers most of the benefit. Implication for us: longer windows help mainly by fixing
  distribution shift (noise-independent), and an intermediate window is typically optimal. Our
  citation for horizon curriculum / scheduled unrolling.
- **Brandstetter, Worrall & Welling (2022), "Message Passing Neural PDE Solvers," ICLR,
  arXiv:2202.03376** (pushforward trick) and **Um et al. (2020), "Solver-in-the-Loop," NeurIPS,
  arXiv:2007.00016.** Train on the input distribution the *rollout itself* produces. CONTRAST with
  **Sanchez-Gonzalez et al. (2020), "Learning to Simulate," ICML, arXiv:2002.09405** and
  MeshGraphNets (Pfaff 2021), which inject training noise for the same end: their "noise" is a proxy
  for self-generated rollout error, not sensor noise, so it is still usable noiselessly, but the
  pushforward/solver-in-loop approaches address the exact mechanism without any noise process and are
  the more principled fit for our deterministic setting.
- Equilibrium-preserving, full-expressivity architectures:
  - **Pacifico et al. (2026), "Exact Fixed-Point Constraints in Neural-ODEs with Provable
    Universality," arXiv:2605.10613 (verified).** Directly on our **f(x_eq)=0** requirement: imposes
    exact equilibrium constraints while proving the constrained class stays universal. The correction
    can be forced to vanish at equilibrium yet remain fully expressive.
  - **White et al. (2024), "Projected Neural Differential Equations for Learning Constrained
    Dynamics," arXiv:2410.23667 (verified).** Enforce invariants/equilibrium by projecting the vector
    field onto the constraint tangent space (hard, not penalty). Template for "constrain only the
    direction you must, leave the rest free".
  - **Manek & Kolter (2019), "Learning Stable Deep Dynamics Models," NeurIPS, arXiv:2001.06116
    (verified).** Jointly learns dynamics + Lyapunov function, stable by construction over the whole
    state space. Include as the cautionary counter-example: exactly the restrictive stability class
    our angle warns against; it can preclude a fully expressive correction and forces an attractor
    structure our marginally-stable plant does not have.
- Regularizing only the unexcited direction (our method):
  - **Gyorok, Hoekstra, Kon, Peni, Schoukens & Toth (2025), "Orthogonal projection-based
    regularization for efficient model augmentation," L4DC, PMLR 283:166-178, arXiv:2501.05842
    (verified).** Soft orthogonal-projection regularizer constraining the ML component only in the
    subspace that would negate the physics part. Tradeoff: soft reg bounds but does not hard-guarantee
    non-negation.
  - **Gyorok et al. (2025), "Orthogonal-by-construction augmentation," arXiv:2511.01321.** Follow-up
    with exact by-construction orthogonality (hard null-space projection) for input-output models; the
    guarantee-carrying version, at the cost of restricting the realizable subspace.

---

## Gap 5: update-step / gradient-level diagnosis (is the bias systematic or stochastic?)

Deterministic-safe tools:
- **Gur-Ari, Roberts & Dyer (2018), "Gradient Descent Happens in a Tiny Subspace,"
  arXiv:1812.04754.** The gradient collapses into a low-dim subspace spanned by top-Hessian
  eigenvectors. Geometric, not sample-noise-based; lets you ask "does the loss geometry systematically
  push along the DC/constant-force direction?" via overlap of the gradient with that direction.
- **Goodfellow, Vinyals & Saxe (2015, ICLR), "Qualitatively Characterizing Neural Network
  Optimization Problems," arXiv:1412.6544.** 1-D loss slices along a chosen direction; projecting the
  gradient onto a fixed direction d (compute g^T d) is the first-order online analogue of a
  profile-likelihood slice. This is the theoretical home for the `dLoss/d(bias)` probe in
  `v3_dc_birth_monitor.py`: our test for whether the loss consistently pushes along the unexcited DC
  direction.
- **Li et al. (2018, NeurIPS), "Visualizing the Loss Landscape of Neural Nets," arXiv:1712.09913.**
  Filter-normalized loss slices + PCA of the optimization trajectory; deterministic-compatible, shows
  whether the path bends toward a bias/constant-output basin.
- **Pascanu, Mikolov & Bengio (2013, ICML), "On the Difficulty of Training RNNs," arXiv:1211.5063.**
  Canonical reference for logging per-layer gradient-norm / update-norm curves.

CAUTION, stochastic-only (do NOT use as-is): the gradient noise scale (McCandlish et al. 2018,
arXiv:1812.06162) and GSNR (Liu et al. 2020, ICLR, arXiv:2001.07384) are defined from per-example
gradient *variance*; in a noiseless full-batch run that variance goes to 0 and the ratio is
degenerate. To recover a "systematic vs noise" test you must manufacture the stochasticity (multiple
seeds/inits or data subsampling) and measure agreement across those, not across minibatches. The
numerator of GSNR (squared mean gradient along a chosen direction) remains the meaningful "systematic
push" quantity. This is exactly why `v3_dc_birth_monitor.py` loops over unfixed seeds and reads
sign agreement.

---

## What to actually monitor per training step (distilled)

1. **Multi-horizon validation gap**: validation free-run error at increasing rollout lengths (e.g.
   0.1 s, 1 s, 12 s). A *growing* short-vs-long gap is the earliest, cheapest warning (SUBNET
   truncated loss; Farina-Piroddi multi-step; List 2024).
2. **Mean/DC of the learned correction's output** over each window: directly tracks the birth of the
   offset (operationalizes Rubruck's optimal-constant-solution finding for our case).
3. **Free-run vs teacher-forced trajectory divergence**: noise-independent, well-defined
   deterministically (Professor Forcing; Piroddi SEM-vs-PEM).
4. **Gradient projection g^T d onto the unexcited DC/constant-force direction**, tracked over
   training: is the loss geometry systematically pushing there? (Goodfellow 2015 slice; Gur-Ari 2018
   subspace.)
5. **Jacobian norm / non-normality of the learned one-step map, not just spectral radius**: since our
   poles sit at z=1, spectral radius ~ 1 is uninformative and transient (non-normal) growth is the
   real drift driver (Pervez & Locatello 2026).
6. **State-norm / position drift over a long held-out rollout** as a model-agnostic backstop.

Items 2 and 4 are implemented per update step in `scripts/gantry/gantry-zero-mean/v3_dc_birth_monitor.py`;
item 1 is its post-run snapshot.

---

## Caveats for a noiseless deterministic setting

- Exposure-bias / scheduled-sampling literature is stochastic-framed (token sampling, distribution
  matching). The mechanism (train/deploy input mismatch leads to compounding) transfers; the fixes
  designed around sampling and the self-recovery counter-evidence (He 2021) do NOT transfer cleanly.
  The two setting-agnostic anchors are DAgger (O(epsilon * T^2)) and Tallec-Ollivier (truncation
  bias).
- Training-noise-injection stabilizers (Sanchez-Gonzalez, Pfaff) use noise as a proxy for
  self-generated rollout error, so they remain usable, but the pushforward trick / solver-in-the-loop
  / temporal unrolling family (Brandstetter, Um, List) hits the exact mechanism without any noise
  process and is the more principled fit.
- Any variance-over-samples diagnostic (gradient noise scale, GSNR, gradient-variance estimators) is
  degenerate under full-batch deterministic gradients. Use geometric probes (gradient projection,
  Hessian subspace, loss-landscape slices) instead, or deliberately inject seed/subsampling
  stochasticity to make "consistency across draws" measurable.
- DC-birth itself is a literature gap. No primary source treats "a learned parallel correction
  acquiring an operating-point DC offset because the windowed loss does not price it." We assemble it
  from mean-first learning (Rubruck; spectral bias), zero-mean init/BN assumptions
  (Glorot/He/Ioffe), and horizon amplification (Ross; Tallec-Ollivier), which positions it as a
  genuine contribution rather than a restatement.

## Unverified (check before citing)
- 2026 Jacobian-regularization preprints arXiv:2602.04608 and arXiv:2603.05538 (surfaced by the
  diagnostics search, not individually verified).
