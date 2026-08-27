# Problem statement: training-induced free-run drift in an augmented LPV-LFR gantry model

**Written 2026-07-25**, after the `drift-fix-trials` T0/T1 measurement campaign. Purpose: give a fresh
deep-research session a precise, evidence-graded problem statement. Supersedes the working six-issue list
where marked. Every number below carries its provenance, because this project has twice drawn wrong
conclusions from numbers measured on a different rig or at a different step count.

> **STATUS 2026-07-25: several claims below are superseded by measurement.** The D1 to D6
> campaign settled the §2 fork (toward optimization), refuted §5.5's non-identifiability framing,
> closed I3's faithfulness caveat, supplied the mechanism behind I8, and removed the empirical
> support for §6 constraint 4's over-damped-baseline argument. Read
> **`docs/drift-conclusions-2026-07-25.md`** before acting on §2, §4/I2, §4/I3, §4/I8, §5 item 1,
> §5 item 5 or §6 constraints 3 and 4. This document remains the source for the issue definitions
> and for I4 to I8.

## 0. Evidence grades used here

| grade | meaning |
|---|---|
| **ROBUST** | measured on the current frozen rig, 3 seeds, and reproduced under 2 independent training protocols |
| **SOLID** | measured on the current frozen rig, 3 seeds, one protocol |
| **SINGLE** | measured on the current rig but 1 seed, or 1 step count |
| **OTHER-RIG** | measured on a previous rig/routing. **Not a valid comparator** for current numbers |
| **INFERRED** | a mechanism argued from theory or from a correlate, never isolated by intervention |
| **RETRACTED** | previously asserted in this project, now contradicted by measurement |

## 1. The system and the deliverable

A physics-based LPV-LFR baseline of a dual-gantry high-precision motion system (Y-scheduled inertia) is
augmented with a learned parallel dynamic component (ANN block, SUBNET encoder, `deepSI` interconnect). The
ANN's output is routed as a FORCE onto the velocity rows of the physical state. Training is windowed
simulation-error minimisation (BPTT over `nf` samples). The deliverable is validated by **free-run
simulation** on held-out data.

The two translational axes (X and Y) are **free integrators**: `K = 0`, so the one-step map has poles exactly
at `|lambda| = 1` (ROBUST, `pole_check.py` on the baseline: max `|lambda| = 1.000000` over the full Y range,
two eigenvalues exactly at 1). This is a physical property of a free-floating stage, not a modelling artifact,
and it must be preserved: pulling those poles strictly inside is artificial damping and is a failure.

**Why that matters.** A constant force error `f` on a velocity row of a free integrator produces position
error growing like `f*t^2/2`, without bound. So an arbitrarily small persistent force error becomes an
arbitrarily large free-run position error at long horizon. The windowed training loss, over `nf = 400`
samples = 0.1 s, barely sees it; the 2 s free-run evaluation sees it amplified ~1000x. **This gap between
what the loss penalises and what the deliverable is scored on is the core problem.**

## 2. The core problem, stated precisely

> Windowed simulation-error training of the augmentation reliably leaves a residual force field with a
> persistent low-frequency component, which the free integrators convert into free-run position drift far
> above the model's own floor. The task is to suppress that component **without** restricting the class of
> dynamics the ANN can represent, because the real residual is unknown and nonlinear.

**Do not write "the loss cannot constrain it" (corrected 2026-07-25).** An earlier version of this statement
did, and it is the one ungraded claim in this document; the available evidence points the other way.
Measured curvature (OTHER-RIG, routing 0..5) makes the windowed loss **stiffest** on the integrator
output-DC, with a positive-definite Hessian and a strict minimum at zero, and with the constant's minimiser
at `b* = -g/H ~ 1e-11`, roughly four orders BELOW the ~1e-07 the optimizer actually parks at. On that
reading the loss constrains the direction strongly and the optimizer simply does not sit where the loss puts
the minimum.

The distinction is load-bearing, because the two readings have disjoint solution spaces:
- **"the loss cannot see it"** is an IDENTIFIABILITY problem. Remedies: excitation and input design
  (§5 item 5), or a prior on a provably non-identifiable direction.
- **"the optimizer does not sit at the constrained minimum"** is an OPTIMIZATION problem. Remedies: the step
  geometry (curvature-aware or preconditioned steps) and the conditioning of the gradient.

**Neither is established on the current rig.** The curvature has not been re-measured at routing (3,4,5),
and the excitation evidence is OTHER-RIG. Settling which holds here is a precondition for choosing between
the two families, and it is cheap: re-measure the 6-D output-DC curvature and `b*` on this rig, then compare
`b*` against the observed ~3.5e-08. If `b*` is orders below the parked value, it is an optimization problem;
if `b*` is comparable to it, the loss genuinely does not determine the constant and the honest framing is
identifiability.

The sharpest available demonstration is the **perfect-match null**: train the augmentation on data generated
by the baseline itself, with true-state initialisation, noiseless, so the correct ANN output is exactly zero
and the windowed loss starts at ~1e-11. Training still moves the ANN off zero and still manufactures free-run
drift. There is nothing to learn, so every bit of the drift is estimator-induced (SOLID). At step 30 the
windowed loss is *worse* than at ANN-off for 2 of 3 seeds (1.71e-07 and 1.45e-07 vs 1.303e-07), i.e. the block
is being **displaced from a minimum, not fitting** (SOLID).

## 3. The frozen rig the current numbers refer to

`scripts/gantry/drift-fix-trials/rig.py`, hash `e1b0511a4c`. 6-state pure baseline (`nx_ann=0`), ANN routed
to velocity rows `(3,4,5)` = (dX, dTheta, dY) — deliverable-consistent, X and Y kept in the routing. Encoder
frozen, true-state init throughout, `orth_beta=0`, `orth_observe=False`. 11 training records + 1 held out
(T5), full 48000-sample (12 s) records, `nf=400`, `stride=100`, 5236 windows, free-run horizon 8000 (2.0 s).
Adam `lr=1e-7`. Two protocols: minibatch (batch 256, 21 batches/epoch, 4 epochs = 84 steps) and
**deterministic full-batch** (fixed 256-window bank, no shuffling, 84 steps).

Measured floors (ANN-off, true-init; SOLID): **X 1.084e-07 m, Y 1.121e-06 m**. Every drift number below is a
multiple of these. There is no oracle in any threshold.

**Expressivity rig:** the same setup with an injected dissipative Coulomb friction `F = -0.5*tanh(v/0.2)` on
the X/Y velocity rows, giving an ANN-off windowed residual of 8.337e-03 = **532x** its perfect-match floor
(SOLID), i.e. well-conditioned signal to learn.

## 4. The six issues, re-graded

### I1 — a persistent low-frequency (DC) force on the K=0 velocity rows
**Status: SOLID, and its character is now pinned.** Under the deterministic full-batch protocol the ANN's
constant routed output (which for this architecture IS the final-layer bias, exactly) is **sustained and
sign-consistent across all 3 seeds**: dX −3.49/−1.58/−3.82e-08, dY +3.56/+1.33/+3.16e-08, unit direction
~(−0.75, +0.66), 0–2 sign flips in 15 tail steps. Under **minibatch** the same quantity **flips sign across
seeds** and is ~7x smaller (~5e-09).

A coherent reading, offered as INFERRED not isolated: minibatch gradient noise partially *cancels* a
systematic component by flipping its sign step to step, while full-batch exposes it cleanly. If so the
systematic bias is the real object and minibatch noise merely masks it.

*Correction to the working list:* "new data weakens I1" is only true of the minibatch protocol. On the
deterministic protocol I1 is a clean, reproducible, direction-stable object. What genuinely weakened is the
claim that I1 is the *dominant carrier of the drift* (see §5, open question 1).

**I1 IS NOW CURABLE, AND WE KNOW WHAT CURING IT BUYS (SOLID, T1b, 84 steps, 3 seeds).** An exact proximal
penalty on the measured direction annihilates the constant: `|c|` falls to 5.25 / 5.33 / 7.44e-10 from controls
spanning 1.27-4.89e-08, a 24-92x reduction, landing on a strikingly consistent endpoint despite the controls
differing 4x. **What it buys is Y and only Y**: Y drift goes to 0.7 / 0.8 / 0.9x the ANN-off floor on 3/3
seeds, i.e. at or below the no-ANN reference. **What it costs is X**: X drift goes 34.8 -> 25.2 (better),
0.7 -> 12.7 (18x worse), 2.1 -> 34.3 (16x worse) -- worse on 2 of 3 seeds, mean 12.5 -> 24.1x. See I8: this
per-axis trade is a finding in its own right and it is what makes I1 "solved but not usable".

*Cross-rig corroboration, worth one line:* the measured unit direction ~(−0.75, +0.66) is close to the
(−0.70, +0.71) measured at a different routing on a previous rig (step4). Two independent rigs landing on
nearly the same direction is weak but real evidence that the drift direction is a property of the system
and the loss geometry rather than of one configuration. Graded OTHER-RIG on the earlier number, so this is
corroboration, not a measurement.

### I2 — optimizer step geometry (curvature-blind Adam parks far from the minimiser)
**Status: mechanism SOLID in general form, but every specific number is OTHER-RIG.** The quoted
"offset = exactly 3.48·lr, drift proportional to lr, floor at lr=0", and the curvature figures
`H_XX = 7.08e4` with minimiser `b* = -g/H ~ 1e-11`, come from `baseline-null/` at routing **0..5**, not from
this rig. They have *not* been re-measured at routing (3,4,5) and per this project's own rule are not valid
comparators. For scale: the current rig's full-batch DC is ~3.5e-08 at `lr=1e-7`, i.e. ~0.35·lr, same order
but not the same coefficient.

What IS newly SOLID here is the *consequence* of curvature-blindness for regularisation, which is I5.

### I3 — wrong-sign (anti-damping) self-feedback on the velocity channels
**Status: ROBUST. This is the best-measured quantity in the campaign.** The trained ANN's self-feedback
`dW_(->s)/dx_s` is **positive** (destabilising) on both routed velocity channels, on 3 seeds, under **both**
protocols at matched step count (84):

| protocol | dW_dX/ddX (s0/s1/s2) | dW_dY/ddY (s0/s1/s2) |
|---|---|---|
| minibatch, 84 steps | +3.49 / +4.07 / +3.53e-08 | +2.06 / +2.59 / +2.27e-08 |
| full-batch, 84 steps | +1.98 / +3.34 / +3.53e-08 | +2.61 / +2.75 / +2.86e-08 |

Six independent measurements of dW_dY/ddY all land in **2.06–2.86e-08, a 1.4x total spread**, versus the DC's
2.7x and the drift's 3–4.6x. Within-seed sampling sd is ~1e-10, two orders below the mean.

**Important measurement caveat, and the source of a wrong claim I made earlier:** this quantity flips sign
between probes at steps 10–30 on every seed and only settles positive by step 50. A 30-step reading is inside
a transient. Reading it there produced a spurious "I3 is a minibatch artifact" conclusion, now RETRACTED.

On real-data checkpoints (OTHER-RIG, with-MSD, longer horizon) the Y free-run envelope climbs 5 → 37 → 70 →
100x and never saturates, while a zero-init baseline saturates at ~11x. That is the reason I3 is believed to
matter at deliverable scale, but it is not measured on the current rig.

**The candidate the working queue omits, and the only one aimed at I3 that restricts nothing (OTHER-RIG).**
The loss does price this feedback: weakly at our horizon, sharply just beyond it. `feedback_instrument.py`
measured the curvature along a canonical Y anti-damping gain growing about `H^3.7`, from `kappa_g = 315` at
`H = 400` to `4.66e4` at `H = 1600`, with no saturation, so the destabilising response turns on INSIDE the
existing ARTBP cap. That makes I3 plausibly a **conditioning** problem, curable by an unbiased
long-effective-horizon gradient rather than by any constraint on the block. Two reasons to grade this up:
it changes the gradient estimator only, so it restricts no class and satisfies constraint 1 trivially,
whatever turns out to be in the real data; and **the "longer fixed windows are refuted" verdict in §7 does
not transfer to it**, because that refutation was measured on the DC, which falls as `1/nf`, whereas this
curvature RISES as `H^3.7`. Opposite scaling, opposite conclusion; the project previously lumped both under
one verdict, and that was wrong. Caveats: the measurement used a SYNTHETIC canonical gain injected on the
baseline, not the trained ANN's own direction, so faithfulness is unproven; and ARTBP is heavy-tailed on a
pole-1 mode, with a measured poly-tail variance gain over geometric of only 2-5x, and it made the
perfect-match null WORSE, where its variance dominates because there is no signal-driven bias to remove.

### I4 — the spurious force and a genuine friction impulse are not separable
**Status: mixed, and weaker than the working list states.** Two sub-claims:
- *Not separable by output direction.* Established for a **hard rank-1 pin** (OTHER-RIG, step4: pin the
  measured direction and the optimizer displaces into the orthogonal DC direction). **REFUTED for the proximal
  application on the current rig (SOLID, T1b, 84 steps, 3 seeds): there is no dodge.** The orthogonal
  component is statistically unchanged by the intervention -- control mean 4.36e-09 vs prox mean 4.18e-09,
  per-seed 2.43 -> 2.49, 7.12 -> 3.92, 3.53 -> 6.14e-09 (one up, two down). At 30 steps a single seed appeared
  to dodge and that reading was retracted. So the dodge is a property of step4's HARD pin, not of
  direction-pinning as such, and the arXiv:2006.06650 "projection induces a compensating stochastic bias" risk
  did NOT materialise here. **This is a genuinely useful negative: it means the barrier to direction-based
  methods is not evasion by the optimizer.** The barrier is I8 instead.
- *Not separable by information content* (on a K=0 axis a constant is the most loss-informed direction per
  unit amplitude): INFERRED, OTHER-RIG.

A structural point that survives regardless: any criterion whose only input is the existing data cannot
distinguish "pole exactly 1 plus a genuine drift" from "pole slightly stable plus an offset" (near-unit-root
theory: the localising parameter is identified but not consistently estimable). That kills *inferring* the
pole and the drift from the same data. It does **not** kill using the KNOWN-pole model and splitting the
residual by a structural criterion such as velocity parity.

### I5 — soft penalties saturate in beta under Adam
**Status: RESTATED. This is the single biggest correction in this document.** The working claim ("every soft
penalty saturates, bit-identical 1e3 → 1e12, so the whole soft-regulariser family is capped at the step
scale") is **too broad**. Saturation is a property of *where* the penalty is applied, not of soft penalties.

Measured on the current rig, 3 seeds, 4 betas, deterministic protocol (SOLID):
- **In-loss** (penalty inside the loss, so it passes through Adam's preconditioner): converges to a
  **beta-independent and nearly seed-independent plateau** — 2.350 / 2.378 / 2.376e-08 at beta=1e7 from three
  different inits, already within 4% of that at beta=1e6, i.e. flat over the top two decades. A quantity that
  lands on the same value from three inits and across two decades of the knob is pinned by something other
  than the knob. This reproduces the original saturation finding.
- **Proximal** (exact/implicit prox applied after the optimizer step, `c <- c/(1+lr*beta)`): **monotone on
  every seed**, 5.1x / 27.6x / 11.6x reduction, reaching 1.9–4.6e-09, ~6.6x below the in-loss plateau. The
  implementation was verified against closed form to three digits.

So: **a penalty saturates when applied inside Adam's preconditioner; moving it outside restores beta's
authority.** The soft-regulariser family is not dead. A useful structural consequence: the exact prox of a
quadratic is the shrinkage `1/(1+lr*beta)`, whose beta → infinity limit is a *hard* projection onto the
orthogonal complement — so "soft vs hard" is a continuous dial here, not a dichotomy.

Also refuted by the same data: prox does NOT get dodged (see I4), so the two objections that would have killed
direction-based regularisation are both retired.

**PRIOR ART — do not claim this mechanism as novel, and use it as a search lead.** This is, in substance, the
AdamW observation (Loshchilov and Hutter, "Decoupled Weight Decay Regularization", arXiv:1711.05101): an L2
penalty placed *in the loss* interacts with the adaptive preconditioner so that its effective strength is
distorted and coupled to the learning rate, while *decoupling* it from the gradient restores direct control.
Our contribution is not the mechanism but its transfer: applying it to a **projection/direction penalty on a
routed physical output** rather than to weight decay, and showing that the previously-reported saturation of
this project's orthogonal-projection penalty (step4, "bit-identical beta = 1e3 to 1e12") is an instance of it
rather than a property of the penalty. The exact prox is the ProxGen family (Yun, Lozano, Yang, NeurIPS 2021,
which additionally takes the prox in Adam's own metric rather than the Euclidean one — an untried refinement
here). **Search consequence: the literature on decoupled and proximal adaptive optimisation is the right place
to look for how to apply ANY of our candidate penalties, and it is a different literature from the one this
project has been reading.**

### I6 — encoder-init floor
**Status: OTHER-RIG, and correctly excluded as a target.** The ~1.7e-5 m encoder-init floor (~40x the model's
own 3.79e-7 m true-x0 floor) was measured at routing 0..5. The current campaign runs true-init only, so the
encoder is not in its init path and this floor does not enter its numbers. It is a reference for the
deliverable, not a drift to fix. Not re-measured here.

### I7 (NEW) — the drift metric is itself ill-posed at short training length
**Status: SOLID for the instability, SINGLE for the settling.** Not in the working list, but it has already
caused two wrong readings this session and it should shape any future experimental design.

Per-axis free-run drift on this rig is **non-monotonic in step count** and spans 3–4.6x across seeds. Seed 0's
control goes 1149 → 407 → 5.3 → 62 → 144 → 186x floor over steps 5–30, while its DC is sign-stable
throughout. So drift and DC are not in one-to-one correspondence, and any verdict resting on a single-step-
count drift number is confounded.

**Now CONFIRMED across all 3 seeds (upgraded SINGLE -> SOLID, T1b).** Unconstrained control drift, same seeds
and protocol, 30 steps -> 84 steps:

| seed | X: 30 -> 84 | Y: 30 -> 84 |
|---|---|---|
| 0 | 185.7 -> **34.8** | 56.3 -> **4.8** |
| 1 | 227.0 -> **0.7** | 17.4 -> **1.2** |
| 2 | 75.8 -> **2.1** | 37.9 -> **2.0** |

So **the control is already at or near the floor on 2 of 3 seeds at 84 steps**, and its DC decays with training
on those seeds (s1: 3.50e-08 -> 8.15e-09) while seed 0's plateaus (~4.9e-08) and stays 35x floor. A substantial
part of what this project has called "the drift" is an **early-training transient**, and how much survives is
strongly **init-dependent**: one init in three does not settle.

**This is a caveat on §2 and §8, and it partly invalidates the testbed.** "Free-run position drift far above
the model's own floor" is overstated for the converged estimator in this regime. Worse, on Y the 84-step
control drift is 4.8 / 1.2 / 2.0x floor = 5.4e-06 / 1.3e-06 / 2.2e-06 m, i.e. **all three below the ~2.2e-5 m
absorber signal the augmentation exists to learn**. A null whose drift sits below the signal cannot
discriminate candidates on that axis — which is exactly what T1b then showed: prox's Y "improvement" to
0.7-0.9x floor is an improvement on a quantity that was already negligible. **Any future candidate comparison
needs a testbed where the drift exceeds the signal at convergence**: excited records rather than standstill, a
longer free-run horizon, or the Coulomb rig. This is now the campaign's binding methodological constraint, not
a footnote.

**Protocol consequence, actionable immediately.** The transient runs to roughly step 50 (I3's sign settles
there). A 30-step unit therefore measures inside it. Any candidate comparison whose metric is DRIFT must use
a unit longer than the transient, or it compares transients rather than estimators. The DC is less exposed,
being sign-stable throughout, but the Jacobian is not: per I3 it flips between steps 10 and 30 and only
settles positive by 50. So the safe rule is that **every metric in this campaign requires units past ~50
steps**, and the 30-step protocol should be treated as diagnostic only.

### I8 (NEW) — a rank-1 constraint on the routed output COUPLES the two axes, trading one for the other
**Status: SOLID (T1b, 84 steps, 3 seeds). This is the finding that replaced "the pin gets dodged" as the real
barrier to direction-based methods, and it is the most searchable open problem in this document.**

The ANN writes a force to both the dX and dY velocity rows. The measured drift direction is a single unit
vector in that 2-D output space, `v ~ (-0.75, +0.66)`. Pinning `c = v . f_xy` therefore does not pin "the DC" —
it imposes a **linear relation between the X and Y force components**. The optimizer can then only serve X
through the orthogonal direction, and empirically that is worse for X than being unconstrained: Y reaches the
floor on 3/3 seeds while X degrades on 2/3 (0.7 -> 12.7x, 2.1 -> 34.3x), with the mean X drift roughly doubling.

Three properties make this the crux rather than a detail:
1. **It is not an evasion effect.** The orthogonal component does not grow (I4), so this is not the optimizer
   escaping the constraint; it is the constraint being genuinely wrong for one axis.
2. **The obvious fix is already refuted.** Pinning the two rows *separately* (rank-2, per-row) removes the
   coupling — but rank-2 on the velocity rows IS the mean / zero-mean penalty, which is closed in §7 because it
   suppresses real friction (a genuine friction impulse under an asymmetric duty cycle has nonzero time mean).
   So the campaign is caught between a rank-1 pin that couples the axes and a rank-2 pin that kills the
   residual it must learn. **Any proposed resolution has to escape both horns.**
3. **An aggregate metric hides it entirely.** Y's gain masks X's loss in any pooled sim-RMS. This is a concrete
   instance of the project's standing rule to report per axis, and it is why this was nearly logged as a
   success.

What would resolve it: a constraint whose *support* is chosen so that the constrained subspace does not
intersect the subspace the residual needs, with that choice made from data rather than assumed — or a
formulation that constrains an accumulated quantity (see §7's note on T6 / rollout-consequence penalties) so
that no output direction is named at all.

## 5. What is genuinely NOT known — the targets for deep research

**Read §9 for how to search each of these; it lists the field vocabularies, because several of the terms this
project uses internally are not the terms the relevant literatures use.**

1. **Which component carries the drift? PARTLY ANSWERED (T1b), and the answer is per-axis.** Annihilating the
   constant (24-92x) takes **Y** to the floor on 3/3 seeds, so on Y the constant WAS the carrier. It does not
   help **X** (worse on 2/3), and it leaves the anti-damping Jacobian untouched (X +1.98 -> +3.08, +3.34 ->
   +3.33, +3.53 -> +3.76e-08). So X's residual drift is carried by something other than the pinned constant,
   with I3 the leading candidate but NOT yet isolated. **The measurement that would isolate it is the
   frozen-state decomposition** (`rig.free_run('frozen')`: pin the ANN's state input at x0 so it becomes a pure
   function of u, and difference against the live-state rollout — the DC part is `frozen − off`, the
   state-feedback part is `full − frozen`). That is supported in the rig and has NOT been run at 84 steps. It is
   cheap and it is the single highest-value remaining measurement.
   *Caveat that weakens all of this:* per I7 the Y drift being "fixed" is a fix on a quantity already below the
   absorber signal, so the Y result is real but may be inconsequential.
2. **Does the anti-damping feedback (I3) have a cure that is not class-restricting?** I3 is the best-measured
   issue, it is now the leading suspect for X's residual drift, and **it survives the best available I1
   treatment untouched** — so it is the campaign's real gap. Two live angles, neither tested:
   (a) the *conditioning* angle (see I3's own section): the curvature along a Y anti-damping gain rises as
   `H^3.7`, opposite to the DC's `1/nf` fall, so an unbiased long-effective-horizon gradient (ARTBP) may cure it
   without constraining anything — and the "longer windows are refuted" verdict in §7 provably does not transfer,
   because it was measured on the oppositely-scaling quantity;
   (b) the *soft power-sign* angle: `relu(F·v)` is published as a baseline that FAILED (DiLaR-Soft,
   arXiv:2604.18277, test RMSE 0.4726 vs 0.0504 for their hard variant), but our I5 result now gives a specific,
   measured reason why their soft variant might have been handicapped (in-loss application under an adaptive
   optimizer), so applying it proximally would be *explaining a published negative* rather than reproducing it.
   Angle (a) is the stronger bet on constraint grounds; angle (b) is the stronger bet on novelty grounds.
3. **R2 has never been measured under the current protocol.** No candidate has an expressivity number on the
   deterministic rig. Per the campaign's own rule a candidate without an R2 number is not a fix. This is the
   largest open cell.
4. **Whether any of this transfers to real data with measurement noise**, longer horizons, the with-MSD
   absorber, and the deliverable's latent-augmented-state routing. Everything above is a noiseless
   perfect-match or injected-friction null at one routing.
5. **Whether the marginal mode can be made identifiable by excitation instead of regularisation.** The
   training data is standstill multisine at 130–180 Hz plus sweeps and APRBS; near-DC content on X/Y is
   plausibly absent, in which case the drift-carrying direction is *practically non-identifiable from this
   data* and the honest framing is optimal input design, not regularisation.
   **Correction: this is not unchecked, it is OTHER-RIG.** The training input was measured to carry exactly
   zero power at 0 Hz, and the one-step baseline residual's DC sits 60 to 1700x below its 130-180 Hz band
   peak. Both already lean toward "near-DC content absent". So the status is "measured elsewhere, needs
   confirming on this rig and this routing", not "unknown". It remains the cheapest potentially decisive
   question in the list, and it is the same measurement that settles the §2 fork between the identifiability
   and optimization readings.

## 6. Hard constraints any candidate must satisfy

1. **Expressivity (the overriding one).** The deliverable is real nonlinear data with an unknown residual. A
   candidate that makes any representable dynamic unlearnable is pruned regardless of its drift score.
   Measured as windowed nf-RMS %learned on the Coulomb rig, never free-run (free-run conflates fit and drift).
2. **Marginal-preserving.** X/Y one-step poles must stay at `|lambda| = 1`: not `< 1` (artificial damping),
   not `> 1` (anti-damping). This eliminates contraction, RENs, strictly-stable port-Hamiltonian with `R > 0`,
   and any spectral/Lipschitz cap tight enough to matter.
3. **Knowledge-free.** The mechanism may use data properties (loss curvature, power sign, a measured
   direction) or the KNOWN baseline structure, never an assumption about the unknown residual. In particular
   the residual's *mean* is not knowable, which is why time-domain zero-mean priors were dropped.
4. **No class-restricting hard constraint as the deliverable — now an EMPIRICAL test, not an axiom**
   (user, 2026-07-25: "if it's not in the nonlinear real noisy data it is fine"). The rule is no longer
   "never forbid a class because it might be the residual"; it is **"forbidding a class is acceptable only if
   that class is demonstrably absent from the real data"**. That is a stronger position, because it is
   checkable, and it shifts the burden onto whoever proposes a structural constraint.
   **Applying the test to every structural candidate we hold, the answer is currently NO for all of them:**
   - *self-channel sign constraints* forbid a positive `dF/dv`, but the residual is truth minus baseline, so
     an over-damped baseline gives residual `(c_base - c_true)*v` with exactly that sign; the real-data fit
     puts viscous `cg1/cg2/cy` at 6-7x the Telica datasheet, so the baseline IS over-damped and the forbidden
     class is the one the residual most likely needs. (The inflation is measured; the "it is faking dry
     friction" explanation is a logged hypothesis whose run never completed.)
   - *telescoping / bounded net impulse* forbids a net-impulse force, i.e. Coulomb friction; the datasheet
     quotes static friction 43/43/49 N.
   - *DiLaR-hard* (`r` proportional to `grad V`, hence vanishing at `v = 0`) forbids a force at rest, i.e.
     stiction, same datasheet evidence.
   - *passivity* forbids energy injection, which is the same positive `dF/dv` as the first item.
   So the conclusion is unchanged but its basis is now evidence rather than caution. Note the coupling: if
   Coulomb moves into the baseline (D-116) the viscous inflation should relax and the residual's damping-error
   sign may change, so any structural claim must name the baseline it is stated against.
   Full expressivity and a for-all-weights no-drift guarantee remain logically incompatible; no-drift must
   therefore come from the ESTIMATOR and is training-conditional.
   **What would change this**: a direct characterisation of the real Telica one-step residual (its spectrum,
   its velocity-even versus velocity-odd split, and the sign of its self-channel Jacobian). Those three
   numbers convert this entire constraint from argument into measurement. It is a no-training analysis, but it
   touches real data and is therefore an ASK gate.
5. **Real-data viability**: measurement noise, closed-loop logs, no oracle available for any threshold.

## 7. Closed — do not re-open

Longer fixed BPTT windows as a fix (DC present at every `nf` 800–3200, ~1/nf, every free-run worse than
epoch 0). lr tuning (drift is lr-proportional by construction; cannot train at lr → 0). Adam↔SGD swap (the
null pass was *inaction* at ~0 gradient; with a real residual SGD learned +0% while still drifting 83x floor,
so R2 and R4 are a single-knob trade-off there). Lipschitz/spectral caps. Contraction/RENs/strictly-stable
pHNN. Zero-mean or window-mean penalties on the velocity rows (suppress real friction: a genuine friction
impulse under an asymmetric duty cycle has nonzero time mean). Theta-only routing. Velocity-only ANN input,
and velocity/acceleration-domain training loss (explicit last resort, supervisor constraint).
Re-aiming the orthogonal projection at the measured drift direction *as a drift cure*. Disturbance/extended
state observers as a *separator* (they estimate the lumped disturbance and cannot discriminate components).
Negative Imaginary as a drift cure in open-loop evaluation (its position boundedness comes from
interconnection with an output-strictly-NI partner, which we do not have).

**Scope note.** The Gyorok orthogonal projection is the thesis's scientific contribution and serves **joint
estimation** — keeping physical parameters identifiable while theta and the ANN are estimated together, i.e.
stopping the ANN absorbing or negating baseline dynamics. It is **not** a drift cure and no-drift is not one
of its jobs. Drift and the joint-estimation projection are independent workstreams.

## 8. One-paragraph version, for a research prompt

*(Kept as written for reuse, but note it predates T1b: it does not yet carry the per-axis trade (I8), the
no-dodge result (I4), or the confirmation that the control reaches the floor on 2 of 3 seeds (I7). For a
research prompt, pair it with §9.)*

> An LPV-LFR physics model of a dual-gantry stage is augmented with a learned force on its velocity rows and
> trained by windowed (0.1 s) simulation-error BPTT, but scored by 2 s free-run simulation. The two
> translational axes are free integrators with poles exactly at 1, so any persistent force error integrates
> twice into unbounded position error that the short training window prices far more weakly than the
> deliverable does. (Whether the window genuinely cannot SEE that component, or sees it but the optimizer
> does not sit at the minimum the loss defines, is itself unsettled on this rig and forks the solution space
> between input design and step geometry.) Even in a perfect-match
> null — baseline-generated data, true init, noiseless, correct ANN output identically zero — training
> displaces the block off zero and manufactures free-run drift 5–200x the model's own floor, while making the
> windowed loss worse. That drift range is measured across steps 5–30, which is inside an early-training
> transient: at 84 steps it falls to 34.8x (X) and 4.8x (Y) on the one seed measured so far, and 4.8x on Y is
> below the absorber signal the augmentation exists to learn, so how much of it survives convergence is open. Two components are measurably present in the trained block: a sustained constant force
> (direction stable across seeds, ~3e-8 normalised) and a positive, i.e. anti-damping, velocity self-feedback
> (~2–3e-8, the most reproducible quantity measured: 1.4x spread over 3 seeds and 2 protocols). Soft
> regularisation of the constant does not saturate as previously believed — that was an artifact of applying
> the penalty inside Adam's adaptive preconditioner; an exact proximal application restores monotone control
> and cuts the constant 5–28x. What is not known is which component carries the residual drift once training
> passes an early transient (~50 steps), whether the anti-damping feedback admits a cure that is not
> class-restricting, and whether the drift-carrying direction is practically identifiable from this excitation
> at all. Required: preserve the marginal poles, do not make any representable dynamic unlearnable (the real
> residual is unknown and nonlinear), and use no oracle in any threshold.

## 9. Search plan for the literature session

**Why this section exists.** This project has three times found that an idea it believed unreported was
published under another field's vocabulary (the one-sided power penalty as DiLaR-Soft *and* as the Macauley
bracket in thermodynamics-informed ML; the drift-versus-friction separator as "bias observability" in inertial
navigation). Keyword search in one field's terms measurably fails here. **Every question below therefore lists
the vocabularies to search, and no novelty claim should be made until at least two non-control vocabularies
have been tried and named.**

Note also that our own I5 mechanism turned out to be the AdamW insight (see I5). Assume by default that a
mechanism we "discover" has a name in optimisation, thermodynamics, navigation, econometrics, or statistics.

### Q1 — Is this an optimization problem or an identifiability problem? (decides the whole solution family)
The §2 fork. Measured curvature says the windowed loss is *stiffest* on the output-DC with minimiser `b* ~
1e-11`, four orders below where the optimizer parks (~1e-07 to 3.5e-08), which points at optimization; the
excitation evidence (zero input power at 0 Hz) points at identifiability. Both are OTHER-RIG.
- **Optimisation / ML theory**: implicit bias (implicit regularization) of adaptive methods; sign descent;
  "stationary point versus iterate displacement"; preconditioner-induced bias; edge of stability;
  Adam without convergence; decoupled and proximal adaptive methods (AdamW, ProxGen).
- **System identification**: consistency and bias of simulation-error (multi-step, free-run) estimators versus
  prediction-error; truncated-BPTT gradient bias; multiple shooting.
- **Econometrics**: local-to-unity / near-unit-root asymptotics; weak identification; flat directions of the
  concentrated likelihood. (This is where "identified but not consistently estimable" is standard.)
- **Experiment design**: optimal input design for identifiability; persistency of excitation; Fisher-information
  and D-optimal design for integrator or marginally stable modes.
- *A good answer looks like*: a criterion, computable from our data, that distinguishes "the loss determines
  this direction and the optimizer fails to reach it" from "the loss does not determine it".

### Q2 — Enforcing dissipativity on the residual WITHOUT restricting the model class (I3, the real gap)
We need the anti-damping self-feedback suppressed where the data says it is spurious, while leaving positive
`dF/dv` representable where the real residual needs it (see §6 item 4: our baseline is measurably over-damped,
so the forbidden class is plausibly the one the residual needs).
- **Control**: dissipativity- and passivity-preserving identification; sector-bounded or IQC-constrained
  learning; port-Hamiltonian and GENERIC learning with *semidefinite* dissipation; Lyapunov- or
  certificate-constrained residual learning; negative-imaginary systems (note §7's scope limit).
- **Thermodynamics-informed ML**: GENERIC / metriplectic structure; Macauley bracket; entropy-production
  penalties. (This is where the one-sided penalty already lives.)
- **Numerical analysis / geometric integration**: structure-preserving and energy-conserving discretisation for
  *marginally stable* systems; why symplectic schemes avoid artificial damping.
- **Inertial navigation, astrodynamics, robotics**: bias observability; IMU bias estimation; empirical
  accelerations; odometry drift. (These fields routinely separate a slowly-varying bias from a physical force
  on an integrated state — our exact problem, under different names.)
- *Key twist to search for explicitly*: **conditional or data-supported constraints** — a constraint active
  only on the subspace the data does not excite, rather than globally. Terms: constraint activation on the
  unexcited subspace; regularisation restricted to the null space of the Fisher information; "do no harm"
  regularisation.

### Q3 — Multi-output constraints that do not trade one channel for another (I8, most searchable new problem)
A rank-1 constraint on a 2-channel routed output imposes a relation between channels and degrades one. Rank-2
per-channel is refuted (it becomes the mean penalty). Escape both horns.
- **Multi-task / multi-objective ML**: negative transfer; gradient conflict and gradient surgery (PCGrad);
  Pareto multi-task optimisation; per-task versus shared regularisation.
- **Statistics**: per-coordinate versus joint shrinkage; James-Stein and its coordinate-wise variants; group
  versus isotropic penalties; structured sparsity.
- **Control / MIMO**: per-channel versus joint constraints in MIMO identification; decoupling and
  channel-wise weighting; directional forgetting in recursive estimation (this last one is a close analogue:
  forget only in the excited direction).
- *A good answer looks like*: a penalty whose active subspace is selected per channel from data, with a
  guarantee or an argument that it cannot remove a physically required component from either channel.

### Q4 — A testbed whose drift exceeds the signal at convergence (I7, blocking methodology)
Currently the null's converged Y drift (1.3-5.4e-06 m) is *below* the absorber signal (~2.2e-5 m), so it cannot
discriminate candidates. This is mostly an internal design question, but two literatures help:
- **System identification**: validation-signal-to-error ratio; benchmark discriminating power; how
  free-run/rollout benchmarks are designed for marginally stable plants.
- **ML**: exposure bias and compounding error in autoregressive rollout; scheduled sampling; train/test horizon
  mismatch. (This is the ML name for our windowed-train / long-rollout-evaluate gap and it is a large
  literature this project has not read.)

### What to bring back
For each question: the closest in-framework papers first (check `literature/` before searching), read end to
end, and extract the actual recipe into a table — not an abstract-level summary, since an abstract inverted a
verdict for us once already (arXiv:2111.14714 reads as a hard constraint in the abstract and is a soft penalty
in the body). Mark every candidate against the five constraints in §6, and state which of them it fails; a
candidate that fails §6 item 1 (expressivity) is not worth queueing however good its drift story.
