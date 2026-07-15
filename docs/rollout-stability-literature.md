# Rollout-Stability / Exposure-Bias Literature (the drift, under other names)

**Date**: 2026-07-11. **Why this doc**: the open-loop-rollout drift has a large literature under keywords we
had NOT searched (exposure bias, compounding error, rollout stability, discrepancy modeling). It names our
exact problem and offers PROVEN, expressivity-preserving, open-loop, position-domain, solve-not-hide training
fixes that have NOT been tried in this project. This is the strongest new direction from the 2026-07-11
search. Main doc: `docs/drift-diagnosis-status.md`; direction: `docs/open-loop-solution-decision.md` (D-107);
verdict/table: `docs/augmentation-literature-verdict.md`. Quotes transcribed from on-disk PDF text layers
(primary-read); re-verify character-exact before thesis use.

## The core reframe: our drift IS exposure bias / distribution shift
Training on short windows (nf=0.1 s, encoder re-init, ground-truth-initialized) means the model only sees the
ground-truth state distribution; at open-loop free-run it sees its OWN (drifted) output distribution, which
differs because errors survive training. That train-test mismatch is "exposure bias" / "distribution shift",
and it drives long-horizon rollout drift. This is the ML-for-dynamics framing of our nf-window-vs-full-traj
problem (drift-diagnosis §3), and it comes with proven fixes.

## A. Pushforward trick -- train on the model's OWN error  [PRIMARY-READ]
**Reference.** J. Brandstetter, D. Worrall, M. Welling, "Message Passing Neural PDE Solvers", ICLR 2022,
arXiv:2202.03376.
- **Diagnosis (p.3):** one-step training overfits the one-step distribution; "small errors in A accumulate
  over rollouts ... and lead to divergence"; the solver at test time sees `A#p_k` not `p_{k+1}` -> the
  "distribution shift problem ... a domain adaptation problem".
- **Fix (Eq 7):** add a stability loss with the input perturbed by the model's OWN one-step error,
  `(u_k + eps) = A(u_{k-1})`; implemented by unrolling 2 steps and BACKPROPAGATING ONLY THE LAST STEP. Trains
  the model to stay correct from its own drifted states without full BPTT.
- **Requirements:** 1 KF yes; 2 expressive yes (loss change, no class restriction); 3 marginal yes (no model
  constraint); 4 empirical (stabilizes long rollouts). SOLVES (fixes the model), does not hide.

## B. Noise injection -- corrupt the input with random-walk noise  [PRIMARY-READ]
**Reference.** A. Sanchez-Gonzalez et al., "Learning to Simulate Complex Physics with Graph Networks" (GNS),
ICML 2020, arXiv:2002.09405. (Related: Pfaff et al. MeshGraphNets; Godwin et al. "Noisy Nodes".)
- **Fix (p.5):** "Because we train our models on ground-truth one-step data, they are never presented with
  input data corrupted by ... accumulated noise ... at training we corrupt the input VELOCITIES of the model
  with RANDOM-WALK noise." -> robust to accumulated error -> reduced rollout error.
- **Why especially well-matched to us:** a FREE INTEGRATOR's accumulated error IS a random walk, so
  random-walk noise on the velocity input directly simulates the drifted states our model must become robust
  to. This is the best-matched single tool for our free-integrator drift.
- **Requirements:** 1/2/3 yes; 4 empirical. SOLVES, does not hide. Distinct from measurement noise (that is
  per-sample; this is accumulated-rollout-error simulation) and from the nf-sweep (that just lengthened
  windows, at the wrong lr -- see D-107 / Optuna 69399 confound).

## C. Transient amplification -- a DIFFERENT drift mechanism  [PRIMARY-READ]
**Reference.** A. Pervez, F. Locatello, "Controlling Transient Amplification Improves Long-horizon Rollouts",
arXiv:2605.08856.
- **Mechanism:** "when the Jacobians along an autoregressive trajectory are NON-NORMAL and NON-COMMUTING, the
  model amplifies errors transiently, resulting in model rollout drift EVEN WHEN THE OVERALL SYSTEM IS
  ASYMPTOTICALLY STABLE." Fix: "commutativity regularization" (penalize normality defect + cross-step
  commutator via Jacobian-vector products, no inference cost); propagator bound; thousands of stable steps.
- **Relevance:** expressivity-preserving regularizer that kills rollout drift (satisfies 1/2/3, 4 empirical),
  BUT targets a DIFFERENT mechanism than ours -- d6 measured our drift as a near-pure DC offset (DC x free
  integrator), not transient amplification of perturbations around a stable trajectory. COMPLEMENTARY tool,
  not the primary fix for our DC-drift.

## D. Discrepancy modeling -- validates our grey-box choice  [PRIMARY-READ]
**Reference.** M.R. Ebers, K.M. Steele, J.N. Kutz, "Discrepancy Modeling Framework", SIAM J. Appl. Dyn. Sys.
2024, arXiv:2203.05164.
- **Rule (abstract):** "if the true dynamics are unknown (imperfect model), one should learn a discrepancy
  model of the missing physics in the DYNAMICAL space" (= our dynamics augmentation); disambiguates
  missing-physics vs measurement error and deterministic vs random effects. Supportive framework, not a drift
  fix.

## E. Precision mechatronics (Oomen / TU-e domain) -- the field works CLOSED-LOOP  [PRIMARY-READ]
**Reference.** e.g. Chou, Duan, Okwudire, "A physics-guided data-driven feedforward tracking controller ...",
arXiv:2206.11960; Oomen "Advanced Motion Control for Precision Mechatronics"; GP position-dependent
feedforward (2201.07511, 2202.00257); ILC/feedforward-learning cluster.
- **Finding:** the precision-stage community handles unmodeled stage dynamics + friction via FEEDFORWARD /
  ILC control (hybrid physics + data-driven, filtered-basis-function CONTROLLER for TRACKING) -- i.e.
  CLOSED-LOOP, inverse-model, grey-box friction compensation. There is NO open-loop forward-model solution;
  the domain sidesteps it by working in the loop. Consistent with "closed-loop hides" (D-107): the field uses
  the loop for control, not for open-loop forward-model fidelity.

## GNS noise injection -- concrete fit to OUR pipeline (and a req-3 risk)
Added 2026-07-11 (user: "how does GNS random-walk noise work and fit here"). Precise mechanism + fit +
critical caveat.

### Mechanism (verbatim reminder + how the random walk works)
GNS (2002.09405, p.5): "at training we corrupt the input velocities of the model with random-walk noise",
because "we train our models on ground-truth one-step data, [so] they are never presented with input data
corrupted by ... accumulated noise". Random-walk (not i.i.d.): `n_k = sum_{i<=k} eps_i`, `eps_i ~ N(0,
sigma^2)` -- accumulated like integration error; sigma tuned so the noise on the latest input ~ the model's
own one-step error over the window. TARGET stays CLEAN -> the model learns (drifted input) -> (clean next
state) = to correct back toward the data instead of compounding.

### Fit to our pipeline -- IMPORTANT CORRECTION: our state is ENCODER-initialized, not ground-truth
GNS assumes ground-truth one-step initialization. **Our SUBNET pipeline is different: the state is
ENCODER-reconstructed from a window of past (clean) I/O, and the encoder RE-INITIALIZES the state at every
window start.** Consequences:
- The "clean initial state" is the ENCODER ESTIMATE (d3 showed it is near-clean: encoder x0 dX~7e-6), not
  literally ground truth -- but still near-clean, so the exposure-bias gap remains.
- **The per-window encoder re-init is EXACTLY the mechanism that HIDES the drift from training:** every 0.1 s
  the state is re-cleaned from real positions, so the model is NEVER launched from its OWN accumulated drift.
  Combined with the short window (0.1 s, below the ~0.5 s drift onset), the model structurally never
  experiences the drifted regime during training. This is why the drift is training-INVISIBLE (§3, d7), and
  why GNS-style injection is needed: to ARTIFICIALLY create the drifted state the encoder would otherwise
  re-clean away.
- **Where to inject (for us):** on the K=0 axis velocities (Ẋ, Ẏ) of the PROPAGATED state during the rollout
  (and/or as a random-walk perturbation of the encoder's x0 output on those rows) -- NOT on the encoder's
  position-window input, because velocities are encoder-reconstructed by differentiating positions, so
  position-window noise is AMPLIFIED into velocity error ([[trace-state-reconstruction]]). Inject on the
  velocity state directly.
- **Magnitude:** sigma data-derived (measured noise floor / accumulated-error scale), NOT arbitrary (else
  heuristic, flag per CLAUDE.md).

### CRITICAL req-3 risk: "correct-back" training may DAMP the marginal pole
Training the model to map (drifted X/Y state) -> (clean target) teaches it to REDUCE the position error. On a
free integrator, a learned position-error-reducing action is an effective RESTORING force = added
stiffness/damping on the pole we are REQUIRED to keep at the origin (req 3, marginal preservation). So GNS
noise could trade DRIFT for DAMPING -- the opposite failure, and the same wall dissipativity hits.
- **Subtlety (why it is not obviously fatal):** the "clean" target is not a fixed point; it is the
  u-determined trajectory, so correcting toward it MAY be legitimate tracking of the correct response to u,
  not absolute stiffness. Undecided a priori.
- **MANDATORY GATE:** eigen-check the trained model (linearize; confirm the X/Y pole stays AT the origin, not
  pushed inside). GNS noise is admissible ONLY if this passes. This makes it a FALSIFIABLE experiment, not a
  safe default.

### What it does and does not address
- Addresses the DISTRIBUTION-SHIFT half of the drift (model learns to not compound from drifted states),
  cheaply, WITHOUT needing a 0.5 s unroll (the random walk simulates the long-horizon drifted state).
- Does NOT make a genuinely-UNEXCITED DC identifiable (the §5m / Direction-2 identifiability half) -- that
  still needs excitation or the data-silent route.
- Net: a well-matched, cheap, FALSIFIABLE experiment for D-107 -- expected to either help (learns not to
  compound) or damp the pole (fails req-3 eigen-check). Not a proven fix.

## Do these extend to the REAL nonlinear data? YES, with three caveats; they PERMIT friction
- **Why they extend:** knowledge-free, self-supervised (use the model's own rollout, not ground truth or a
  dynamics assumption); PROVEN on real data in the source community (ERA5 weather, real fluids);
  expressivity-preserving (can learn real nonlinear friction).
- **They permit real friction (do NOT suppress it):** the criterion is "match the real data over the
  rollout". Real friction is IN the data and is self-limiting (opposes motion, no runaway), so the model
  learns it; only a SPURIOUS runaway drift is penalized. Unlike the net-impulse block, these penalize
  drift-BEHAVIOUR, not DC-carrying forces -- friction survives.
- **Caveat 1 -- noise calibration.** The injected random-walk noise magnitude must be data-derived (relative
  to the measured noise floor), never arbitrary. It is DISTINCT from measurement noise (per-sample) -- it
  simulates accumulated rollout error, so it is still needed on real data. [[measurement-noise-post-hoc]]
- **Caveat 2 -- closed-loop data, open-loop deliverable.** The Telica data is closed-loop, but the deliverable
  is open-loop free-run BFR of the FITTED model -- which is exactly where exposure bias bites, so the fix
  applies to the right quantity.
- **Caveat 3 -- no ground truth.** Fine: these are self-supervised (own rollout); validate by held-out
  free-run BFR (no oracle).
- **Net:** extends, and is arguably BETTER-suited to our low-dim mechanical/integrator case than to chaos
  (random-walk velocity noise matches free-integrator drift structure), while keeping friction.

## Verdict and recommendation
- These are the most on-target research found for our drift: they name it (exposure bias / distribution
  shift), and give PROVEN, expressivity-preserving, open-loop, position-domain, SOLVE-not-hide fixes that are
  UNTRIED in this project (distinct from the confounded nf-sweep).
- **All give requirement 4 EMPIRICALLY, not structurally** -- consistent with the impossibility (structural
  req 4 vs full expressivity, `augmentation-literature-verdict.md`). This is the honest ceiling, now
  much better supported than "our own regularizer".
- **Recommendation:** the D-107 first step (clean position-domain re-run at correct lr) should ADD
  (a) GNS-style random-walk velocity-input noise (best-matched single tool) and/or (b) the pushforward loss
  (2-step unroll, backprop last), alongside multiple shooting. Try these BEFORE any bespoke construction.
  Commutativity regularization (C) is a complementary add-on if non-normal amplification is also present.

## Provenance / primary-read status
- PRIMARY-READ this thread: Brandstetter pushforward (2202.03376), Sanchez-Gonzalez GNS (2002.09405),
  Pervez-Locatello transient amplification (2605.08856), Ebers-Steele-Kutz discrepancy (2203.05164),
  Chou-Duan-Okwudire physics-guided feedforward (2206.11960).
- SEARCH-VERIFIED (not primary-read): Pfaff MeshGraphNets, Godwin Noisy Nodes, Bengio scheduled sampling,
  the broader Oomen ILC/GP-feedforward cluster.
- All quotes: re-verify character-exact against the PDF before thesis use.
