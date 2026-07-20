# Diagnosing Within-Rollout, Per-Step Error Accumulation: a Cited Synthesis

**Created**: 2026-07-17, from a web literature search (background research agent) requested to
design the in-window per-step accumulation diagnostic (Jan's "eerste 100 stappen" / Theme D / V4).
Companion to `claude-deep-research-drift-diagnostics.md` and
`claude-deep-research-inwindow-accumulation.md`. This one is scoped to: for a FIXED trained model,
how the prediction error accumulates WITHIN one rollout window as a function of rollout step index,
where it takes off, whether it explodes or ramps, whether init error fades or persists, and how a
constant (DC) output reduces the late-window error.

**Setting**: noiseless deterministic simulation; learned parallel correction on a physics baseline;
windowed TBPTT over nf=400 steps; plant has K=0 integrator states (poles at z=1, no restoring
force); the trained model develops a DC offset that integrates into unbounded drift. Every method
below is graded for noise-model dependence. Verification was done in-agent (source existence
confirmed via arXiv/DOI; unfetched/thin items flagged).

---

## Gap 1: decomposing per-step rollout error growth and fitting a growth law

- **Parthipan, Anand, Christensen, Hosking, Wischik, "Defining error accumulation in ML atmospheric
  simulators," NeurIPS 2024** (arxiv.org/abs/2405.14714). The key method: subtract a
  ROLLOUT-FREE reference model from the autoregressive error so the term attributable to
  iteration/dynamics is isolated from error the model would incur anyway. For us: compare free-run
  against a one-step teacher-forced baseline to isolate the accumulation term the K=0 states
  amplify. Noise caveat: their delta(t) is a KL between conditional distributions (probabilistic);
  keep the subtract-a-reference concept, replace the KL with plain per-step MSE.
- **Lorenz error-growth lineage: Bednar et al. (PMC4539497); "Revisiting Lorenz's Error Growth
  Models," MDPI Encyclopedia 4(3):73, 2024** (mdpi.com/2673-8392/4/3/73). The canonical empirical
  growth-law fitting method: fit competing forms (quadratic small-error, power-law,
  logarithmic/saturating) to error-vs-time curves and read the regime. Our template for fitting
  linear vs quadratic vs exponential. Caveat: their growth source is intrinsic chaos, a DIFFERENT
  source than our offset drift; borrow the fitting method, not the interpretation.
- **Schaefer et al., "Imagined Rollouts are Kinematic, Not Dynamic: A Diagnosis of Long-Horizon
  World-Model Failure," Robotic World Models Workshop @ RSS 2026** (arxiv.org/abs/2607.05966).
  A per-step diagnostic that uses the STRUCTURE of the error to diagnose its source (departure from
  a closed-form kinematic null). Analogous to reading our error shape: linear position error implies
  constant velocity offset, quadratic implies constant force. Deterministic-friendly.
- **Janner, Fu, Zhang, Levine, "When to Trust Your Model (MBPO)," NeurIPS 2019**. The standard
  citation for one-step error compounding into a horizon-dependent bound. Caveat: stochastic-MDP
  framing (TV-distance); in noiseless deterministic systems the TV terms collapse to a per-step
  prediction error but the compounding structure survives. The exact "quadratic-in-k" power is from
  secondary summaries, NOT confirmed from the primary.

Honest gap: no single paper maps growth-law exponent to physical source (constant velocity ->
linear, constant force -> quadratic, instability -> exponential). That mapping is our own
contribution, assembled from the Lorenz fitting method and the kinematic-vs-dynamic diagnosis idea.

---

## Gap 2: ramp/drift vs exponential instability (strongest-sourced area)

Clean split: spectral radius / Lyapunov exponents govern asymptotic exponential growth;
non-normality / pseudospectra of the Jacobian product govern finite-time transient (ramp-like)
amplification even when the spectrum is stable. This is the "sits beside vs explodes" test, and it
is a purely deterministic linear-algebra fact.

- **Trefethen & Embree, "Spectra and Pseudospectra: The Behavior of Nonnormal Matrices and
  Operators," Princeton Univ. Press, 2005**. The citable classic for finite-time / non-normal
  transient growth: a matrix with an entirely stable spectrum can still produce large transient
  growth when non-normal (pseudospectra, departure-from-normality). The rigorous basis for a bounded
  ramp beside truth despite a stable spectrum. Use this as the classical anchor (with Pervez &
  Locatello 2026 as the on-point modern version).
- **Storm, Linander, Bec, Gustavsson, Mehlig, "Finite-Time Lyapunov Exponents of Deep Neural
  Networks," Phys. Rev. Lett. 132, 057301 (2024)** (arxiv.org/abs/2306.12548). Defines/computes
  finite-time Lyapunov exponents (FTLE) for a network map: positive = local exponential expansion,
  negative = contraction. Transfers to a fixed-length rollout Jacobian product; our nf=400 window is
  exactly a finite horizon.
- **Engelken, Wolf, Abbott, "Lyapunov spectra of chaotic recurrent neural networks," Phys. Rev.
  Research 5, 043044 (2023)** (arxiv.org/abs/2006.02427). The full Benettin/QR machinery to compute
  the Lyapunov spectrum from per-step Jacobians; ties the leading exponent's sign to chaotic
  (exponential) vs stable propagation. Notes fluctuating input REDUCES entropy rate, so the noiseless
  case is the sharper regime (favorable to us).
- **Vogt, Puelma Touzel, Shlizerman, Lajoie, "On Lyapunov Exponents for RNNs," Front. Appl. Math.
  Stat. 8:818799 (2022)** (arxiv.org/abs/2006.14123). A practical, efficient LE-spectrum estimator
  (products of per-step Jacobians) distinguishing contractive from expansive regimes. A ready recipe.

Additional leads NOT deeply verified (treat as leads only): arXiv:2605.24868 (comparative rollout
stability), arXiv:2603.08191 ("Non-Normal Route to Chaos").

---

## Gap 3: separating initial-state / encoder error from per-step model error

- **Hoekstra, Gyorok, Toth, Schoukens, "Encoder initialisation methods in the model augmentation
  setting," arXiv:2602.13108 (2026)**. Directly our framework and authors: uses the baseline model
  to predict the initial state from past I/O, isolating the encoder/initial-state problem from the
  augmented model's rollout error. Closest source to our setup. Caveat: partly noise-motivated but
  applies in the deterministic limit (demonstrated on a mass-spring-damper).
- **Forgione, Mejari, Piga, "Learning neural state-space models: do we need a state estimator?,"
  arXiv:2206.12928 (2022)** (code: github.com/forgi86/sysid-neural-estimator). Isolates initial-state
  assignment as its own error source in multi-step neural SSM training; shows encoder quality matters
  only for certain dynamics ("for asymptotically stable ones, zero/random init is competitive"). The
  key point for us: our MARGINALLY-stable plant is exactly the regime where init error does NOT wash
  out.
- **Hatfield et al., "Building Tangent-Linear and Adjoint Models for Data Assimilation With Neural
  Networks," JAMES, 2021** (DOI:10.1029/2021MS002521). The formal x0-sensitivity machinery:
  tangent-linear and adjoint models propagate sensitivity of later states back to the initial
  condition. The adjoint route to attribute rollout error to x0. Deterministic linearization.
- **Magnusson, "Dependence on initial conditions versus model formulations ...," QJRMS
  145(722):2085-2100, 2019** (DOI:10.1002/qj.3545). The clean experimental-differencing recipe: run
  pairs holding model fixed / swapping x0 (and vice versa), then difference trajectories to attribute
  error to init vs model. The "roll from true x0 vs estimated x0 and difference" logic. Ensemble-
  framed, but the differencing DESIGN transfers verbatim to deterministic sim.
- **Tool: Chen, Rubanova, Bettencourt, Duvenaud, "Neural ODEs," NeurIPS 2018**
  (arxiv.org/abs/1806.07366). The adjoint state a(t)=dL/dx(t) integrated backward is the mechanism
  for rollout sensitivity to x0. Deterministic by construction.

Honest gap: no single paper does "roll from true x0 vs encoded x0 and difference to isolate init
from per-step model error in a noiseless deterministic neural-SSM." 3.1/3.2 closest in-framework;
3.3/3.4 supply the transferable method.

---

## Gap 4: marginal stability / integrators / lack of contraction (why a DC offset ramps)

Argument chain: contraction (Lohmiller-Slotine) bounds rollout error -> fading memory (Boyd-Chua) /
echo-state property is that condition for input-driven systems -> at z=1 (no restoring force) the
condition fails -> a bounded/DC input integrates into an unbounded ramp instead of washing out. All
deterministic; none require noise.

- **Lohmiller & Slotine, "On Contraction Analysis for Nonlinear Systems," Automatica 34(6):683-696,
  1998**. Contraction (Jacobian symmetric part uniformly negative-definite, rate lambda<0) implies
  neighboring trajectories converge exponentially and bounded perturbations give bounded, decaying
  error: precisely the property that bounds rollout error. Our integrator has lambda=0 (pole z=1):
  the bound degenerates and offsets accumulate. Load-bearing classic.
- **Boyd & Chua, "Fading Memory and the Problem of Approximating Nonlinear Operators with Volterra
  Series," IEEE TCS 32(11):1150-1161, 1985** (DOI:10.1109/TCS.1985.1085649). Canonical definition of
  fading memory (remote-past influence must decay). An integrator (pole z=1) has infinite DC gain and
  does NOT forget: a constant offset is the pathological case where fading memory fails. Load-bearing
  classic.
- **Singh, Sankaranarayanan, Raman, "Contraction, Criticality, and Capacity: A Dynamical-Systems
  Perspective on Echo-State Networks," arXiv:2507.18467 (2024/25)**. The cleanest modern statement
  tying ESP -> FMP -> contraction; when contraction fails "two trajectories typically do not
  converge" and the reservoir "lacks FMP." Their lambda_max -> 0^- criticality is the analog of our
  z=1 pole. Caveat: preprint, not peer-reviewed; use as synthesis/illustration.
- **Gonon & Ortega (Grigoryeva-Ortega school), "Fading memory echo state networks are universal,"
  Neural Networks 138:10-13, 2021** (arxiv.org/abs/2010.12047). Establishes rigorously that ESP +
  fading memory are what a recurrent map must have to be a well-defined universal I/O operator: the
  formal reason a non-contractive (integrating) per-step model has no bounded rollout operator.
- **Angeli, "A Lyapunov Approach to Incremental Stability Properties," IEEE TAC 47(3), 2002**
  (DOI:10.1109/TAC.2002.800648); modern learned-dynamics treatment: **"Formally Verified Neural
  Lyapunov Function for Incremental ISS of Unknown Systems," arXiv:2501.05778 (2025)**. delta-ISS =
  small input perturbations give proportionally small, bounded state deviations (the rollout-error
  bound). Requires a strictly negative decay term; a pure integrator (z=1) has zero decay, is not
  delta-ISS, so an input bias yields an unbounded ramp.

Honest note: the bare fact "DC input to a z=1 pole -> linear ramp -> BIBO-unstable at marginal
stability" is textbook (cite an Oppenheim / Astrom text, not a paper). No single source states the
composite "learned per-step model with DC offset on an integrator gives unbounded rollout"; we
assemble it from 4.1 + 4.2 + the BIBO fact.

---

## Gap 5: visualizing and reporting rollout error accumulation (plot recipes)

- **Lam et al., "GraphCast," Science 2023** (arxiv.org/pdf/2212.12794). The canonical RMSE-vs-lead-
  time curve: x = rollout step index, y = error per variable, one line per model, averaged over
  held-out initializations. Direct analog of "error vs step-index within window." No noise.
- **Vonich & Hakim, "Testing the Limit of Atmospheric Predictability with a ML Weather Model,"
  Science 2025** (arxiv.org/html/2504.20238v1). Error-growth curve on log-scale y to expose an
  exponential law: identify phases, fit a doubling time from the log-linear segment. On our curve a
  straight line on LINEAR axes = ramp, straight line on LOG axes = exponential. No noise.
- **Parthipan et al., NeurIPS 2024** (same as Gap 1). Reporting recipe: attribute per-step growth
  into fixable model-deficiency vs irreducible floor. Caveat: the intrinsic branch leans on chaos +
  ensemble spread/skill; in noiseless deterministic sim the irreducible branch collapses, so only the
  model-deficiency decomposition transfers.
- **"Long Roll-outs of Auto-regressive Neural Operators for Compressible Navier-Stokes ...,"
  arXiv:2601.22541 (2026)** (arxiv.org/html/2601.22541). A full single-deterministic-rollout
  diagnostic panel: (a) L2 error vs rollout timestep; (b) final-snapshot field with correlation;
  (c) log-log energy spectrum truth vs prediction; (d) frequency-vs-time heatmap of spectral drift.
  "Which part of the state drifts, and when." No noise.
- **Watt-Meyer et al., "ACE: A fast, skillful learned global atmospheric model," arXiv:2310.02074
  (2023); ACE2, npj Clim. Atmos. Sci. 2025**. Conserved-quantity-drift-over-rollout plots: track a
  conserved budget as a time series over a long rollout, look for a secular slope away from zero. For
  us: plot integrated position/momentum vs step and look for the DC-offset-driven linear slope.

Honest gap on "per-step contribution to loss/gradient": the closest verified hit is step-resolved
data attribution / Step-Decomposed Influence for looped transformers (arXiv:2602.10097, 2026), but
that is influence-per-step, not gradient-norm-per-unroll-step, and targets looped transformers, not
BPTT-through-a-simulator. A clean citable source that plots gradient-norm-per-unroll-step for
autoregressive simulators does not appear to exist: a genuinely thin spot (novelty).

---

## Distilled checklist: what to plot and compute (within-window per-step diagnostic)

For a FIXED trained model, over rollout step k = 1..400:

1. **Per-step error curve e(k)** = norm(xhat_k - x_k), and per state component (positions vs
   velocities). Plot on BOTH linear and log y. Read the shape: straight on linear = ramp (constant
   velocity offset -> linear position error); upward-curving = quadratic (constant force); straight on
   log = exponential (instability).
2. **Growth-law fit**: regress e(k) against k and k^2 and log e(k) against k; report which form wins
   per component and, if log-linear, the doubling time / growth rate.
3. **Rollout-free reference subtraction**: subtract the one-step / teacher-forced error from the
   free-run error to isolate the accumulation term (deterministic MSE version, not KL).
4. **Init-vs-model error separation**: roll from true x0 and from encoded x0, difference the two
   trajectories (Magnusson design); attribute residual to per-step model error. Optionally compute
   the adjoint sensitivity de(k)/dx0.
5. **Ramp-vs-blowup test (deterministic)**: form the ordered Jacobian product Phi_k =
   J_{k-1}...J_0; compute (a) leading finite-time Lyapunov exponent via Benettin-QR (sign decides
   exponential blow-up vs bounded); (b) spectral norm ||Phi_k|| vs spectral radius rho(Phi_k) and a
   departure-from-normality / pseudospectral check (large norm with rho~1 = non-normal transient
   ramp, NOT instability).
6. **DC-offset probe**: estimate the model's mean per-step output offset over the window; verify the
   integrated offset (offset x k on the z=1 states) matches the observed position ramp. Ties the
   growth law to its source and explains why the constant reduces late-window error (the trained bias
   is the least-squares-optimal constant that minimizes summed window error on a ramp).
7. **Conserved-/integrated-quantity drift plot**: track integrated position/momentum vs k, look for a
   secular slope.
8. **Per-step contribution to loss/gradient**: plot per-step loss e(k)^2 and per-step gradient-norm
   across the unroll to see which steps dominate the TBPTT objective (literature thin: novel-
   diagnostic opportunity).

Items 1, 3, 6 and the DC-on-vs-muted comparison are the CORE (approved for the first script);
5 and 4 (FTLE/Jacobian, init-vs-model) are the rigorous follow-on.

---

## Noiseless-deterministic caveats

- Fully transferable (no noise dependence): Trefethen-Embree pseudospectra; all Lyapunov/FTLE tools
  (Engelken et al. note the noiseless case is the sharper regime); contraction / fading-memory /
  delta-ISS theory; adjoint/TLM x0-sensitivity; the RMSE-vs-lead-time, log-scale doubling-time,
  energy-spectrum-drift, and conservation-drift plot recipes; the hold-one-fixed differencing design.
- Partially transferable (strip the stochastic half): Parthipan (KL metric and spread/skill floor are
  probabilistic; keep the subtract-a-reference concept and model-deficiency decomposition); MBPO
  (TV-distance collapses to per-step error deterministically); Magnusson (only the differencing idea
  transfers).
- Do NOT use: any ensemble-spread / spread-skill diagnostic as a primary error-growth measure; these
  are inherently stochastic and have no meaning on a single deterministic trajectory.

Genuine novelty (thin literature): (a) mapping growth-law exponent to physical error source, (b) the
composite "DC-offset-on-an-integrator -> unbounded rollout" statement, (c) gradient-norm-per-unroll-
step attribution for BPTT-through-a-simulator.

## Lower-confidence / unverified (check before citing)
- Ruzmaikin et al. 2022 (Meteorology MDPI) orthogonal init/model decomposition: author list
  unverified, paywalled.
- Exact "quadratic-in-k" power in MBPO: secondary-source only.
- Unfetched leads: arXiv:2605.24868, arXiv:2603.08191, arXiv:2602.10097 (Step-Decomposed Influence).
