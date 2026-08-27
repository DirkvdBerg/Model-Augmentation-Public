# Drift: conclusions after the D1 to D6 diagnostic campaign

**Written 2026-07-25** at the end of the `scripts/gantry/drift-diagnostics/` campaign (D1 to D6) plus
one literature agent on the optimizer question. **Purpose: a future session should be able to act on
this document without re-deriving anything, and should be able to check every number against the
artifact that produced it.**

Every claim below carries (a) an evidence grade, (b) the artifact that holds the number, and (c) what
the claim rules out. Where this document contradicts an earlier one, the contradiction is stated
explicitly in §7 rather than left for the reader to find.

## 0. How to read this

Evidence grades, unchanged from `docs/drift-problem-statement.md` §0:

| grade | meaning |
|---|---|
| **ROBUST** | current frozen rig, 3 seeds, reproduced under 2 independent training protocols |
| **SOLID** | current frozen rig, 3 seeds, one protocol |
| **SINGLE** | current rig, 1 seed or 1 step count |
| **OTHER-RIG** | measured on a previous rig or routing. Not a valid comparator |
| **INFERRED** | argued from theory or a correlate, never isolated by intervention |
| **DERIVED-HERE** | our algebra, stated by no paper. Re-derive before it enters the thesis |

All rig numbers are on **rig hash `e1b0511a4c`**: `scripts/gantry/drift-fix-trials/rig.py`, perfect-match
null, routing `(3, 4, 5)` = force on `(dX, dTheta, dY)`, true-state init, encoder frozen, deterministic
full-batch Adam at `lr = 1e-7`, 84 steps, fixed 256-window bank, `nf = 400`, float32.
Measured ANN-off floors: **X `1.084e-07 m`, Y `1.121e-06 m`**. No oracle enters any threshold.

## 1. Provenance map

| Artifact | What it holds |
|---|---|
| `scripts/gantry/drift-diagnostics/results/D1-dc-curvature.md` | loss curvature, `b*`, Frye index, profile interval |
| `scripts/gantry/drift-diagnostics/results/D2-i3-curvature.md` | curvature exponent along the trained anti-damping direction |
| `scripts/gantry/drift-diagnostics/results/D3-ysweep-vs-standstill.md` | ysweep versus standstill fit of the constant |
| `scripts/gantry/drift-diagnostics/results/D4-telica-residual.md` | real Telica one-step residual, three numbers |
| `scripts/gantry/drift-diagnostics/results/D5-preconditioned-sharpness.md` | preconditioned sharpness, `b*` at the trained point |
| `scripts/gantry/drift-diagnostics/results/D6-frozen-decomposition.md` | frozen-state decomposition of the drift |
| `scripts/gantry/drift-diagnostics/results/DECISIONS.md` | every judgement call made while measuring |
| `scripts/gantry/drift-diagnostics/results/SUMMARY.md` | the one-page version of D1 to D4 |
| `scripts/gantry/drift-diagnostics/data/` | unit JSONs, 3 checkpoints, 3 Adam states, the D4 payload |
| `scripts/gantry/drift-diagnostics/logs/` | the raw `.output` of every run |
| `docs/gantry-augmentation-problem-log.md` §12 | run-table rows with hypothesis and outcome for D2, D3, D5 |
| `docs/drift-problem-statement-post-diagnostics.md` | the research brief for a fresh literature session |
| `docs/drift-problem-statement.md` | the pre-diagnostic statement; still the source for I4 to I8 |

Reproduction: every script is `scripts/gantry/drift-diagnostics/d{1..6}_*.py` and each results doc
carries the exact command in its §2. D6 and D4 train nothing; D1 trains nothing; D2, D3 and D5 train.

## 2. The conclusions

### C1. The windowed loss DOES determine the drift-carrying constant. Identifiability is refuted.

**Grade: SOLID.** Evidence: `D1-dc-curvature.md` §3.1 to §3.5, units `data/D1_zeroinit_2d_seed*.json`.

At the zero-output init, which for the perfect-match null is the correct optimum, on the 2-D `(dX, dY)`
routed-constant subspace:

* `H_2x2 = [[3153.66, 0.2296], [0.2296, 200.12]]`, eigenvalues `200.12` and `3153.66`, **cond 15.8**,
  positive definite. Accepted after an `h` sweep over five decades; the pair `1e-09 / 1e-08` agrees to
  **0.44%**, well inside the 20% acceptance rule, so not precision-limited and no float64 needed.
* `g = (+2.1088e-06, -1.7414e-09, -3.4180e-08)` on `(dX, dTheta, dY)`. The X constant carries 62x the Y
  gradient: the loss pulls hardest on the axis the drift is worst on.
* `b* = -H^-1 g` has norm **`6.90e-10`**, i.e. **51x below** the parked `3.5e-08`.
* **Frye's gradient-flatness index `r = 3.9e-16`** against a `r > 0.9` flat cutoff
  (Frye et al., *Neural Computation* 33(6), 2021, DOI `10.1162/neco_a_01388`, App. A.4).
* Two-sided bound `6.688e-10 <= ||b - b*|| <= 1.054e-08`; the parked value is **3.3x above the upper
  bound**, and outside the measured profile interval `+-1e-08` at **7.8x** the threshold
  `delta_L = 1.266e-13`.

**Rules out:** the identifiability branch of `docs/drift-problem-statement.md` §2, and with it the whole
excitation and optimal-input-design remedy family (data informativity, persistency of excitation,
Fisher-information design for integrator modes).

**Caveat that must travel with C1:** at this evaluation point the ANN emits exactly zero, so `L`, `g` and
`H` do not depend on the trained weights and the three seeds return **bit-identical** numbers
(`L(0) = 8.847964678634912e-13` on all three). That is a deterministic identity, not reproducibility
evidence. See `DECISIONS.md` D1-1.

### C2. Y-modulation does not make the constant identifiable, and the "modulation was too weak" escape is closed.

**Grade: SOLID.** Evidence: `D3-ysweep-vs-standstill.md` §3.1 to §3.2, units
`data/D3_{ysweep,standstill}_fb84_seed*.json`, bank statistics `data/D3_bank_stats.json`.

* The intervention is real: the ysweep bank carries **`1.194e-02 m`** of within-window Y traversal
  (std) against **`3.941e-06 m`** for standstill, a factor **3030**, at comparable static coverage
  (pooled Y range `0.600` versus `0.450 m`).
* Tail `|c|` is **not smaller** on ysweep: `4.266 / 5.481 / 3.872e-08` (mean `4.540e-08`) against
  standstill `3.747 / 3.656 / 2.886e-08` (mean `3.429e-08`), i.e. **1.32x larger**, pairwise on every
  seed. That is inside the measured within-arm seed spread (1.42x ysweep, 1.30x standstill), so the arms
  are comparable at this resolution.
* The "ysweep simply has more to fit" explanation is excluded: the null's residual is identically zero
  and the ysweep arm reaches a **3.1x lower** tail loss (`1.94e-13` against `5.96e-13`).

**Rules out:** the sweep-report correction that §5.5's "practically non-identifiable" should be softened
*because of LPV modulation*. The conclusion survives on C1's grounds; the reasoning does not.

**Two by-products, both 3/3 seeds, both unexplained:**
* the ysweep-trained model drifts **6.5x less on X and 4.7x less on Y** on the held-out **standstill**
  record than the standstill-trained model does (`4.3x / 5.1x` floor against `27.9x / 23.8x`);
* **I3 comes from the ysweep records**: `dW_dY/ddY` is `+2.469` to `+3.017e-08` on all three ysweep
  fits, but `+2.934e-08 / -2.753e-09 / -1.178e-09` on the standstill fits, vanishing or turning negative
  on 2 of 3 seeds.

### C3. The anti-damping self-feedback (I3) is the published curse of memory, along the ANN's own trained direction.

**Grade: SOLID.** Evidence: `D2-i3-curvature.md` §3.1 to §3.5, units
`data/D2_trained-dir_nf{400,800,1600}_seed*.json`.

* Reproduction gate first: `jac_self` reproduced the problem statement's full-batch I3 row **to three
  digits** on all three seeds (`dW_dX/ddX = +1.984 / +3.342 / +3.525e-08`,
  `dW_dY/ddY = +2.608 / +2.754 / +2.858e-08`).
* `kappa = d' H d` by Pearlmutter HVP along `d = grad_theta(dW_dY/ddY)`, `d` fixed across horizons:
  `30.4 / 417.5 / 5598.1` (seed 0), `37.1 / 521.9 / 6145.6` (seed 1), `44.0 / 634.6 / 8525.2` (seed 2)
  at `nf_probe = 400 / 800 / 1600`, strictly monotone.
* Exponent **`p = 3.762 / 3.685 / 3.798`, mean 3.749, spread 0.113**, inside the `H^3` to `H^4` band and
  matching the earlier **synthetic** canonical-gain `H^3.7` within the seed spread.

**Closes:** the faithfulness caveat in `docs/drift-problem-statement.md` §4/I3. The synthetic direction
was faithful.

**Caveat:** the `[3, 4]` band is **DERIVED-HERE**, our mapping of Zucchet and Orvieto's
`(1 - lambda^2)^-3` law (NeurIPS 2024, `arXiv:2405.21064`, Eqs. 5 and 6) onto a truncation horizon. It
must be re-derived before it enters the thesis. Independent of that derivation: the measured exponent
and its agreement with the earlier measurement. Robustness: dividing `kappa` by the probe loss, which
itself grows 25x, still leaves a superlinear exponent `1.37` to `1.52`.

### C4. Neither a better gradient estimator nor Adam's metric reaches that direction.

**Grade: SOLID.** Evidence: `D2-i3-curvature.md` §3.4, `D5-preconditioned-sharpness.md` §3.2.

* **Estimator side.** The gradient component along `d` grows 201x to 925x from `nf = 400` to `1600`, but
  the whole gradient grows nearly as fast (`|g|` exponent 3.8 to 4.4), so the direction's **share** does
  not consistently improve: cosine `0.0274 -> 0.0619` (seed 0), `0.0398 -> 0.0179` (seed 1),
  `0.0061 -> 0.0201` (seed 2), staying in a 0.6 to 6.2% band. Adam is scale-free per coordinate
  (Zhuang et al., TMLR 2022, `arXiv:2202.00089`), so uniform amplification is cancelled by the
  preconditioner.
* **Metric side.** In Adam's own metric the DC and anti-damping directions have comparable Rayleigh
  quotients (`2.437` to `2.857e7` against `1.690` to `3.699e7`) despite raw curvatures differing 50 to
  70x. What separates them is gradient alignment: DC `0.449` to `0.681`, anti-damping `0.006` to
  `0.040`. The preconditioner does not touch alignment.
* **The stochastic long-horizon branch is independently blocked** at `|lambda| = 1`: Tallec and
  Ollivier's variance control needs geometrically decaying memory; Beatson and Adams (ICML 2019,
  `arXiv:1905.07006`, Thm 4.1) prove no sampling distribution gives finite variance for a non-decaying
  residual, and measure that the Russian-roulette variant this project chose is the worse of the two.

**Rules out:** ARTBP and the stochastic unbiased-truncation family as the cure for I3; and
preconditioning or curvature-aware steps as the cure for I3. What survives of the estimator route is
only its deterministic form (exact forward-mode sensitivities, RTRL and structured approximations),
where cost rather than correctness is the obstacle.

### C5. The parked constant is not an attractor of either named kind. The live explanation is that the iterate has not arrived.

**Grade: SOLID for the refutations, INFERRED for the not-converged reading.** Evidence:
`D5-preconditioned-sharpness.md` §3.1 and §4; the step-size measurement below; the literature agent's
full-text read of Bock and Weiss and Cohen et al.

* **Edge of stability refuted.** Preconditioned `lambda_max` at step 84 is
  `8.713e7 / 2.514e8 / 1.039e8` = **`0.229 / 0.662 / 0.273`** of Cohen et al.'s `38/eta = 3.8e8`, below
  threshold on every seed and still **rising** from step 42 by a uniform `1.42 / 1.39 / 1.42x`. Raw
  `lambda_max` is **identical to five significant digits** across those 42 steps
  (`5.4316e3 / 5.2008e3 / 4.9701e3`), so the entire rise comes from the preconditioner **shrinking**
  (`d'Pd` falls about 30%), not from progressive sharpening. This reproduces Cohen et al.'s
  **Appendix D "Corner case"** for extremely small learning rates (their smallest tested `eta` is
  `1e-6`; ours is `1e-7`), though their stated mechanism needs `P` growing and ours shrinks.
* **Adam limit cycle refuted.** Bock and Weiss's bifurcation inequality (6),
  `alpha lambda_max / sqrt(eps) (1 - beta1) < 2 beta1 + 2`, evaluates to **`0.543 < 3.8`**: the rig sits
  on the **stable** side by a factor 7, so their 2-cycle formula never applied.
  **DERIVED-HERE**, and note their `eps` sits inside the square root, which is not PyTorch's convention.
* **Measured step size, 9 runs** (T1b control and D3 both arms, 3 seeds each, from the logged
  `c_trajectory` in each unit JSON): median per-step `|dc|` in the tail is **`0.005` to `0.013 x lr`**,
  with the increment reversing sign on 9 to 22 of 41 tail steps. Closing the residual gap therefore
  needs tens of consistent steps, and the protocol stops at 84. Note also `beta2 = 0.999` gives the
  second moment a 1000-step time constant, so bias correction is applying a 12.4x inflation at step 84.
* **`b*` at the trained point** is `1.813e-9 / 5.486e-10 / 2.002e-9` against parked
  `4.611e-8 / 9.575e-9 / 3.016e-8`, a **15 to 25x** mis-seating (51x at the init). C1's verdict survives
  the move off the init; the factor is not portable, exactly as Cohen et al. warn. `b*` is strongly
  non-monotonic in step count (4 to 12x larger at step 42, which is inside the transient).

**Rules out:** both published mechanisms as explanations of the parked position on this rig.
**Re-opens:** the not-yet-converged explanation, which the earlier sweep dismissed on a false premise
(see §7, correction 2).

### C6. The free-run drift is the RESIDUE of a near-cancellation between two much larger contributions.

**Grade: SOLID. This is the most consequential single result of the campaign.** Evidence:
`D6-frozen-decomposition.md` §3, units `data/D6_frozen-decomp_fb84_seed*.json`.

Decomposition on the position trajectory, `dc = xp_frozen - xp_off`, `fb = xp_full - xp_frozen`, which
telescope exactly to `xp_full - xp_off` (measured residual `0.00e+00` on every axis and seed):

| seed | axis | total above floor | DC alone | feedback alone |
|---|---|---|---|---|
| 0 | X | 3.882e-6 (**35.8x**) | 6.518e-6 (60.1x) | 2.637e-6 (24.3x) |
| 1 | X | 1.821e-7 (**1.7x**) | 5.892e-6 (**54.3x**) | 6.075e-6 (**56.0x**) |
| 2 | X | 3.353e-7 (3.1x) | 7.206e-7 (6.6x) | 3.854e-7 (3.6x) |
| 0 | Y | 6.175e-6 (5.5x) | 1.257e-5 (11.2x) | 6.407e-6 (5.7x) |
| 1 | Y | 1.983e-6 (1.8x) | 2.311e-6 (2.1x) | 4.293e-6 (3.8x) |
| 2 | Y | 3.063e-6 (2.7x) | 1.350e-7 (**0.1x**) | 2.928e-6 (2.6x) |

On **5 of 6** axis-seed pairs the two components are anti-correlated and each is larger than their sum.

**What it explains.** This is the mechanism behind **I8**, the per-axis trade
`docs/drift-problem-statement.md` calls "the most searchable open problem in this document". T1b crushed
the constant, Y reached the floor on 3/3 seeds and X degraded on 2 of 3. Removing one side of a
cancelling pair unveils the other at its full 24 to 56x-floor magnitude. The barrier was never evasion
by the optimizer (I4 refuted that) and it is not the rank-1 support as such: **the intervention breaks a
balance.**

**Caveat that bounds the correlation column:** on a `K = 0` axis every constant-force contribution gives
the same `t^2` position shape, so the `-1.000` correlations are near-automatic and carry no mechanistic
information. The magnitude ratio is the finding.

**The exception is informative:** seed 2 on Y does not cancel (correlation `+0.716`, DC 4%, feedback
96%, DC component `0.1x` the floor). That is the first isolated measurement of the anti-damping feedback
carrying the drift on its own.

### C7. On the real Telica data: the residual is low-frequency dominated, its mean is not knowable as zero, and it needs MORE damping, not less.

**Grade: SOLID on real data, with the caveats in `D4-telica-residual.md` §5.** Evidence:
`D4-telica-residual.md`, payload `data/D4_telica_residual.json`. Train split only (11 operating points
x `iter0`/`iter8` = 22 logs, 212,364 residual samples), against the fitted run-71447 LPV-LFR baseline
**without** its Coulomb term. Logical frame: X translation [N], Y [N].

1. **Spectrum.** Content below 10 Hz exceeds the 130 to 180 Hz peak by **991x (X)** and **1377x (Y)**,
   and exceeds the measured noise floor by 967x and 127x. The OTHER-RIG figure this replaces had DC
   sitting 60 to 1700x *below* the band peak. Not a like-for-like refutation: the Telica logs are
   closed-loop ILC point-to-point moves with no designed content in that band.
2. **Odd/even split: NOT SEPARABLE on this data.** Exactly **0.00% of 85,358 gross-sliding samples
   travel backwards**; every log is a single forward stroke. Measured, not assumed. What is measurable
   is the constant-like part: mean residual `-157.5 N` (X) and `-83.7 N` (Y) at **315 and 344 sigma**,
   and `+177.8 N` / `+63.6 N` at rest against a noise floor of `172.5 / 43.0 N` std.
3. **`dF/dv` is NEGATIVE**: `-173.3` (X) and `-18.8 N/(m/s)` (Y), negative on **22 of 22** and **21 of
   22** logs, by a joint `[1, v, a]` regression that separates damping error from mass error
   (`corr(v, a) = 0.000`).

**Changes to `docs/drift-problem-statement.md` §6:**
* **Constraint 3 is reinforced with measurement.** The residual's mean is demonstrably not zero and not
  knowable as zero, so zero-mean and window-mean priors on the velocity rows stay dead, now on evidence.
* **Constraint 4's over-damped-baseline argument is NOT supported.** It reasons that the fitted viscous
  parameters sit far above datasheet so the residual needs a **positive** `dF/dv`, the class that sign
  constraints and passivity forbid. The measurement says the opposite, even against a baseline whose
  fitted `cg1 = 290.5` already sits 2.1x above the datasheet 136. Under constraint 4's own empirical
  test, a positive-`dF/dv` prohibition would now **pass** in this operating regime. Before anything is
  built on that: one travel direction, one speed range, closed-loop ILC logs, and a residual measured
  against a fitted model, so a damping-parameter error and a genuine velocity-dependent residual are
  the same object at this level of analysis.
* **Parity is unavailable, not disqualified.** The protocol is right; the data lacks the reverse stroke.

## 3. The synthesis: what the training issue actually is

Not one thing. Three, stacked, and the third is what makes the first two hard to see.

1. **An objective mismatch that no optimizer can fix.** The loss is 0.1 s windowed simulation error; the
   deliverable is a 2 s free run, and on a `K = 0` axis a constant force error grows as `f t^2 / 2`. D1's
   profile interval puts a number on the blind spot: the loss cannot distinguish `|c| <= 1e-08` from
   zero at its own resolution. The optimizer parks at `3.5` to `4.6e-08`, i.e. it **overshoots the
   objective's indifference band by about 4x**. Perfect optimisation of this loss buys roughly that
   factor and then stops. (The band is protocol-dependent, since `delta_L` is a sampling resolution.)
2. **An optimizer that stops short**, by 15 to 25x of the local minimiser at step 84 (C5), for reasons
   that are now most likely mundane (not converged) rather than exotic (both attractors refuted).
3. **A cancellation that makes the observable drift a small difference of large opposing terms** (C6).
   This is why the drift metric has been non-monotonic in step count, 3 to 4.6x variable across seeds,
   and why per-axis conclusions kept inverting.

Plus two contributors that are not about the optimizer at all: the data determines whether I3 appears
(C2), and a substantial part of what the campaign historically called "the drift" is an early-training
transient (I7, and re-confirmed by D5's step-42 versus step-84 numbers).

## 4. What is ruled out, and why

| Candidate | Status | Why, with reference |
|---|---|---|
| Excitation / optimal input design as the remedy | **Ruled out** | C1: the loss determines the constant. C2: 3030x more modulation moves nothing |
| Pin or prox on the measured direction (rank-1) | **Ruled out in its current form** | C6: it removes one side of a cancelling pair and unveils the other. This is the measured explanation of T1b's X degradation |
| Rank-2 per-row pin | Ruled out earlier, unchanged | it is the mean penalty, which suppresses real friction; and C7 now shows the real residual's mean is far from zero |
| ARTBP / stochastic unbiased truncation | **Ruled out** | C4, plus Beatson and Adams Thm 4.1 at `|lambda| = 1` |
| Preconditioning / curvature-aware step as the I3 cure | **Ruled out** | C4: Adam's metric already nearly equalises the directions; alignment is the problem |
| Zero-mean or window-mean priors | **Ruled out** | C7 item 2, at 315 to 344 sigma |
| Longer fixed BPTT windows as a fix | Ruled out earlier, unchanged | the DC falls only as `1/nf`; `nf` is a probe here, never a remedy |
| lr tuning, Adam-to-SGD swap, Lipschitz/spectral caps, contraction, RENs, strictly-stable pHNN, Theta-only routing, velocity-only input, velocity-domain loss, disturbance observers as separator, NI in open loop | Ruled out earlier, unchanged | `docs/drift-problem-statement.md` §7 |

**What that leaves.** Change **what the loss measures**, not how it is optimised or where the step goes.
The shape that survives C6 is a term that prices the accumulated rollout consequence rather than naming
an output direction, because it acts on the residue instead of on either component and therefore cannot
break the balance. This is a shape, not a candidate: nothing here has been designed, let alone tested.

## 5. What is still open, with the measurement that would close each

| Open question | Measurement that closes it | Cost |
|---|---|---|
| **Is the parked constant an attractor or an unconverged iterate?** | continue training to 400 or 800 steps and watch `c` against `b*` | about 45 min per seed at 400 steps; trains, so it needs a run-table row |
| Why do the two components cancel (C6)? | an intervention, not an observation. Nothing here isolates it | unknown |
| Does the cancellation survive a real residual? | repeat D6 on the Coulomb rig, where the DC has real work to do | 3 free runs per seed after a Coulomb training run |
| Is the raw Hessian really static in the weights (D5 §6)? | evaluate `H` at a deliberately perturbed weight vector | minutes, no training |
| Does I3 come from Y traversal causally, or only correlationally (C2)? | a third D3 arm, and a bank that varies traversal depth continuously | 3 to 6 fits |
| Does `dF/dv < 0` hold on reverse travel (C7)? | one reverse-stroke Telica log per operating point | new data collection; the analysis then runs in 21 s |
| Where does an adaptive optimizer park on a marginal mode? | literature: `arXiv:2605.06821` (rod flow for Adam) is the highest-value unread item | one agent |

## 6. The single next action

**Train to 400 or 800 steps on the frozen rig and watch the constant.** It decides whether C5's
"stops short" is a real phenomenon or an unfinished run, it is the cheapest experiment remaining, and
its outcome determines which of the surviving families is worth designing. If `c` walks to `b*`, a
substantial part of what this project has called "the drift" is a convergence artefact and the campaign's
baseline numbers need restating. If it plateaus near `3e-08`, we have an attractor that no current
theory names, which is a thesis-grade finding in its own right.

Do not design an objective change before that number exists.

## 7. Corrections this session made to earlier documents

A future session must not re-inherit these.

1. **Bock and Weiss do NOT derive without bias correction.** `docs/drift-literature-sweep-2026-07-25.md`
   §1 Leg 1 states they do, and an earlier version of
   `docs/drift-problem-statement-post-diagnostics.md` §4 Q1 repeated it as an unexplained "8.9x prefactor
   gap". Bias correction is that paper's stated contribution (their Theorem 2: the bias-corrected
   trajectories reach the same 2-cycles), and it uses nonzero `eps` by design. The `eps = 0` step is one
   tractability step inside the derivation of eq. (4). More decisively, their own bifurcation inequality
   puts our rig on the stable side by 7x, so the formula never applied and there is no gap (C5).
2. **Leg 2 of that sweep is void.** It dismissed the not-yet-converged explanation on the premise that
   Adam's per-step displacement is of order `lr`, making the gap "one third of a single step". Measured,
   the step is `0.005` to `0.013 x lr`, so the gap is 35 to 40 consistent steps (C5). The not-converged
   reading is live again.
3. **Sweep correction #6 is confirmed by measurement** (C3); **#7's reasoning is refuted while its
   conclusion survives on other grounds** (C2); **#9 is refined**: parity passes as a measurement in
   principle but is unavailable on the current train split (C7).
4. **D1's Cohen comparison is now measured, not indicative** (C5), and D1's stated non-portability
   caveat is closed: the factor moves from 51x to 15 to 25x, the verdict does not move.
5. **`arXiv:2006.06650` does not support the claim `thread-AB` item A8 attributes to it.** Full-text
   grep: the word "bias" does not occur in the body; it is a positive convergence result for AMSGrad on
   constrained weakly convex problems. The predicted "projection induces a compensating stochastic bias"
   failure mode of the proximal fix therefore has no source. `arXiv:2208.00441` (item B3) is confirmed
   but disqualified for this rig, since its mechanism is stochastic gradient noise and our gradient is
   exact.

Four rules were added to `tasks/lessons.md` from this session's own errors: measure an optimizer's
realized step from the trajectory rather than reasoning from `lr`; verify what a frame claims a paper
contains; treat a deterministic evaluation point's seeds as an identity rather than a sample; and use
projection shares for telescoping decompositions.

## 8. What this campaign verified about the rig itself

Worth recording, because it makes everything above reusable:

* **The deterministic full-batch protocol is verified end to end.** D5 retrained from scratch and
  reproduced D2's checkpoints **bitwise on all three seeds**
  (`data/ckpt_null_seed{0,1,2}_step84.pt`).
* **The rig reproduces its best-measured quantity exactly.** `jac_self` matched the problem statement's
  full-batch I3 row to three digits in D2, D5 and D6 independently.
* **`drift(full)` reproduces T1b's control exactly** in D6 (X `34.8 / 0.7 / 2.1x` floor,
  Y `4.8 / 1.2 / 2.0x`).
* Three ANN checkpoints and three Adam optimizer states are now on disk, so any curvature, sharpness or
  decomposition measurement at the 84-step point is cheap and needs no training.
