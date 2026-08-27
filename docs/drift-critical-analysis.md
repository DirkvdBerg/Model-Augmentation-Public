# Free-run drift: independent critical analysis of the whole project's evidence

**Date**: 2026-07-24. **Author**: analysis session #1, Phase 1.
**Scope**: an INDEPENDENT re-assessment of the entire gantry-augmentation drift evidence base, built from
the primary artifacts (run table, subfolder scripts, saved `.npz`), not from any single session's summary.
`docs/drift-problem-research-brief.md` was treated as a pointer and is CORRECTED here (section 7).
Companion deliverable: `docs/drift-research-report.md` (Phase 2 literature).

**Method**: read the §12 run table in full, the diagnostic READMEs and result docs of every gantry
subfolder, `drift-diagnosis-status.md`, `decisions.md` (D-100..D-120), the two spec docs, and re-opened
several saved result files directly (`dB_boundedintegral_projection_V1.npz`,
`dC_boundedintegral_train_V1.npz`) rather than trusting their prose summaries. Where two sessions disagree,
both claims are stated with their evidence and a verdict on which survives.

---

## 1. What the problem actually is (defended from primary evidence)

The one-line framings in circulation ("the ANN learns a DC", "it is an estimator artifact", "Adam wanders a
flat direction") are each true of one slice and false as a whole-problem statement. The evidence supports a
**three-source, two-regime** description.

### 1.1 The structural setting (solid)
X and Y are zero-stiffness axes: velocity is damped, position is a free integrator (discrete pole exactly 1).
`pole_check.py` measured the baseline one-step Jacobian at 9 points across the T5 Y range: `max|lambda| =
1.000000` everywhere, two eigenvalues exactly at 1.0, none above 1. Theta is sprung and parks. Consequence
(measured, not argued): a persistent force on a velocity row ramps position without bound, while the same
force on Theta parks at a bounded offset (d17, v4, f01).

### 1.2 Source A: a persistent DC on the K=0 rows, driven by truncated-BPTT bias
- The trained ANN carries a near-constant output on the K=0 rows, dominantly `dY`: `-4.21/-3.62/-3.55e-6`
  across 3 independent inits, born by ~step 13, sign-reproducing (v3b). Present on 9/9 nf-sweep checkpoints
  plus 71167, magnitude ~`1/nf` across `nf` in {400 .. 3200} (SLURM 71013 + the 400->800 point).
- The DRIVE is the truncated-BPTT bias, established **by intervention**: v12 replaced the fixed-window
  gradient with an unbiased random-horizon (ARTBP) gradient at the SAME mean horizon; the DC collapsed
  20-90x into the clean-noise band AND lost its reproducible sign (-,+,+ vs the control's locked negative).
  The 5-seed Phase-D grid reproduced this (fixed `-4.12e-6`, frac<0 = 1.00; poly6 `-1.21e-7`, frac<0 = 0.60).
  This is the single cleanest causal result in the project.
- The AMPLIFIER in parameter space is the optimizer's step geometry: at matched lr and matched loss, SGD
  built `+1.98e-9` where Adam built `-3.98e-6` (~2000x, v3x0/Experiment 6). See section 2.5 for why this
  does NOT make "swap the optimizer" a fix.

### 1.3 Source B: a wrong-sign (anti-damping) state feedback on the Y velocity channel
This is a genuinely DIFFERENT object from the DC and is under-weighted in most summaries.
- In the perfect-match null, after attributing drift by intervention (freeze the ANN's state input), the
  displaced DC carries 75-88% and a **systematic** feedback carries 20-27% on Y/dY. The Jacobian has the
  destabilizing sign on the velocity self-channels: `dW_dY/ddY = +1.43e-8`, `dW_dX/ddX = +5.74e-9`, sd ~1e-11
  (`gain_vs_dc.py`). Small here, but real and not noise.
- On a real with-MSD checkpoint it dominates: `test_efolding.py` finds a slow GROWING Y oscillation (~0.085
  Hz), envelope ratio climbing monotonically 5 -> 37 -> 70 -> 100x with horizon and never saturating, versus
  a zero-init baseline that SATURATES at ~11x. The input has 0.00% power below 1 Hz, so the growth is
  unforced. `test_self_scheduling.py` rules out the LPV self-scheduling loop (all 5 conditions = 1.00x).
- Nulling the DC fixes X but makes Y worse (~1.79x, v5, second-hand from the lessons/handoff record). The two
  sources sit on different axes and a fix for one is not a fix for the other.

### 1.4 Source C: the encoder-init offset floor (bounded, not drift, but it sets every absolute number)
`floor_horizon.py`: true-x0 free-run floor 3.79e-7 m (2 s) rising to 4.81e-6 m (12 s); encoder-x0 floor
1.49e-5 -> 2.16e-5 m. The gap ~1.7e-5 m is roughly horizon-INDEPENDENT, so it is a bounded offset, not a
drift. d17 shows the same at baseline level (dominant term 1.45e-3 m encoder IC on X, everything settles).
This is not the drift, but it is 40x the model's own floor and it contaminates any metric that is not
per-axis and floor-referenced.

### 1.5 The honest problem statement
> The deliverable metric is long free-run position on two marginal (pole-1) axes. The short-horizon
> simulation-error estimator (a) is biased by truncation on exactly the non-decaying mode, which tilts the
> gradient along a persistent-force direction, and (b) is optimized by a curvature-blind step that parks the
> parameters a fixed lr-scale away from the direction's minimizer. The result is a persistent velocity-row
> force (X-dominant) plus a small wrong-sign velocity feedback (Y-dominant, growing with horizon), both of
> which the marginal mode integrates. On top of this sits a bounded encoder-init offset that sets the floor.

This is an **optimization/conditioning** failure on a stiff-but-badly-approached direction, NOT (as the
project asserted for several weeks) an identifiability failure in a flat direction. That distinction is
load-bearing and is defended in 2.3 and 2.4.

---

## 2. What is solidly established, what is over-claimed, what is contradictory

### 2.1 Solidly established (multiple independent lines, at least one interventional)
| Claim | Evidence |
|---|---|
| The physics-based model is correct | d2 (1.2e-7 m from true x0), `floor_horizon` true-x0 3.79e-7 m |
| The physics carries no DC the baseline lacks | v1f (both DC mechanisms, largest ~1e-7, 5 orders below the ANN DC); v1b; v1d |
| No code or data-path offset manufactures the DC | v8 self + matlab arms (mixed sign, <=3.4e-7), with a size-matched positive control v8-inj recovering 86-95% at >=1e-6 and a measured detection limit ~3e-7 |
| K=0 is the amplifier; sprung axes park | d17, v4 (X/Y linear ramps R2 0.99, Theta no growth law) |
| Truncation bias drives the sign-locked DC | v12 ARTBP intervention + Phase-D 5-seed grid + the 1/nf law |
| Longer FIXED windows are NOT a fix | 71013 (nf 800..3200, all worse than epoch 0, DC present on all 9), 70903 curriculum, d8 sign argument |
| Lipschitz / contraction caps are the wrong knob | v6b: measured trained ANN Lipschitz ~5.1e-4, orders BELOW every planned cap, Y destabilizes 114-270x anyway |
| Velocity-only routing does not escape the problem | `gain_vs_dc GV_ROUTE=3,4,5`: mechanism unchanged, Y drift 2.4x WORSE (double integration of a velocity DC) |
| A Fisher/low-information cutoff cannot be the pin constructor | d16: on a K=0 axis DC probes are the MOST loss-informed (Gram 0.44-290 vs band 2.6e-3..1e-9) |
| Telescoping bounded-impulse output bounds the drift by construction and does not damp the pole | B1 unit test (`sup|sum F| = 0.015` over 8000 steps at random weights), 1.0x floor over 6 epochs, `pole_check` shift 3.2e-7 |
| M(Y) scheduling detune is second-order on this machine | d13 (worst measured drift gives 0.03x absorber-RMS output deviation; even dY = 0.5 m stays below absorber level) and `test_self_scheduling` (all 1.00x) |

### 2.2 OVER-CLAIM 1 (the biggest): "the drift is an estimator artifact, decoupled from the signal"
The null tests do NOT support this at the magnitude that matters.
- baseline-null arm B manufactures a dY DC of only `-1.09e-7`, about **40x below** the real-data `-4.5e-6`.
  The run's own UPDATE explicitly walks the claim back.
- v8 (system == model exactly, full pipeline, ANN on) produced NO sign-locked DC at all: last-50 means
  `-2.2e-8 / +1.4e-7 / +1.2e-7`, mixed sign, max |dY| <= 3.4e-7, and the per-step `dLoss/d(bias)` sign-flips
  step to step (versus sign-consistent in the real-residual run).
**Correct statement**: there is a small signal-INDEPENDENT displacement floor (~1e-7), and a much larger
signal-DEPENDENT component (~4.5e-6) that is the estimator's BIASED RESPONSE to a genuine residual. Any cure
validated only on the null is validated against 2% of the phenomenon. This is exactly why the null-based
"SGD passes R4" conclusion later collapsed (2.5).

### 2.3 OVER-CLAIM 2: "the encoder-init compensation is the dominant DC mechanism"
This is the §3b/d8/d9 causal chain, and it is the frame of the research brief's section 2A. **It was refuted
twice by intervention inside this project, and the brief resurrects it.**
- **SLURM 70558** (na=27): a pre-flight measurement showed the encoder's mean dY init bias collapses from
  `+2.675e-4` to `+9.4e-5 m/s` (0.41 SE, statistically zero). Pre-declared criterion 2 was "val sim-RMS does
  not degrade monotonically". Outcome: val nf-RMS improved (the encoder fix worked as an encoder fix) but val
  sim-RMS rose 7.46e-5 -> 8.13e-4 after ONE epoch, identical drift signature. HYPOTHESIS FALSIFIED.
- **v3x0** (true-init training, not just true-init evaluation, so it is a genuine training-time
  intervention): the DC still grows to `-3.36e-6`, about 85% of the encoder-init control's `-3.98e-6`.
- **d11**: the TRAINING set never had a mean dY encoder bias (pooled `+5.1e-5 m/s`, 0.69 SE); d8's
  "the loss prefers the DC" was a val-set artifact.
- **d12**: on TRAINING windows the DC is loss-NEUTRAL (pooled Delta/SE `+0.71`, per-trajectory mixed signs).
**What survives**: encoder-init dominates the UNTRAINED free-run error and sets the ~1.5e-5 m floor
(d17, `floor_horizon`), and removing it drops the NULL drift ~40x. It is a real and separate problem. It is
NOT what the DC compensates. Chasing the encoder as the root cause of the DC is chasing a refuted chain.

### 2.4 OVER-CLAIM 3: "the windowed loss is flat/blind in the DC direction"
Refuted for the position channels, and the refutation is unusually solid: `curvature_sensitivity.py` measured
`d2L/db2` = X 7.084e4, Y 3.542e4, dX 3.671e3, **dY 3.285e2**, versus Theta 3.967e-1, dTheta 8.913e-2. It is
eps-invariant (the map bias -> output -> loss is affine so L(b) is exactly quadratic), autograd matches FD to
4 significant figures, and the 6x6 Hessian is positive definite, so `b=0` is a STRICT local minimum.
**Correct statement**: the loss is stiff on X/Y position-DC and comparatively soft on dY (the blindness ratio
`|dDrift/dLoss|` is 0.03 (X), 0.78 (Y), **3.9 (dY)**, the only channel above 1). The zero-mean folder's "flat
direction" language is imprecise, but its specific finding (the walked DC lives on dY) is independently
vindicated by this test. The consequence is important: **the minimizer is tiny (`b* = -g/H ~ 1e-11`) and the
trained model sits ~1e-7 away**, so this is a DISPLACEMENT problem, not a "the data does not determine it"
problem. Adding curvature (a soft penalty) attacks the wrong quantity if the operating point is set by step
size rather than by curvature; step-4's beta-saturation result (2.6) is the direct confirmation.

### 2.5 OVER-CLAIM 4 (already self-corrected, but still present in older docs): "SGD is the fix"
Sequence: gantry-zero-mean measured 2000x less DC under SGD at matched loss; TASK 1 measured SGD at 1.0x
floor in the null and PROMOTED the estimator route; `r2_fit_probe` then showed the null pass was
**inaction** (perfect match gives ~0 gradient, SGD's `lr*g` step is ~0, so it never moves: no drift AND no
learning), and with a genuine injected residual SGD learned +0% while still drifting 83x floor, versus Adam
+18% learned and 1034x drift. The lr sweep meant to settle it was confounded (anti-damping injection,
underpowered windows, non-robust Adam reference flipping +18% -> -13% on a MAXLEN change).
**Verdict**: the optimizer-only slice is correctly eliminated, but it was eliminated on theory plus TASK 5,
not on the confounded sweep. The surviving mechanistic content is real and useful: at a stiff direction a
curvature-aware step converges to `b*` while a sign-like step parks ~lr away.

### 2.6 UNDER-WEIGHTED FINDING: a soft penalty saturates in beta under Adam
`step4_orth_projection_null.py` is negative on its headline but its two mechanisms are the most
decision-relevant results of the recent work:
1. **beta saturates**: results are bit-identical from beta = 1e3 to 1e12, because at high beta Adam's update
   tends to `lr * sign(grad V)`, so the penalty's WEIGHT stops mattering and the achievable displacement caps
   at ~lr. This is a structural limit on the ENTIRE soft-regularizer family under a scale-invariant
   optimizer, and it is not mentioned in any plan document.
2. **a rank-1 direction pin is dodged**: pinning the measured joint-DC direction `(dX,dY) = (-0.70,+0.71)`
   moved the displacement into the orthogonal DC direction `(-0.71,-0.71)`, which drifts equally. Catching
   all velocity-row DC requires a rank-2 pin, which is the mean penalty, which suppresses friction.
**Caveat honored**: that testbed was noise dominated (drift bouncing 3-5x per epoch, DC sign flipping), so
it is not a clean pass/fail on efficacy. The two MECHANISMS above are structural, not statistical, and stand.

### 2.7 OVER-CLAIM 5: the bounded-integral numbers, and an orphan result nobody logged
Re-read directly from `simulations/gantry_subnet/diagnostics/dB_boundedintegral_projection_V1.npz`:

| treatment | X1 tail drift | X2 | Y | Y band (130-180 Hz) |
|---|---|---|---|---|
| raw (trained ANN) | 2.190e-3 | 2.191e-3 | 2.591e-2 | 2.106e-6 |
| mean removed | 1.838e-4 | 1.841e-4 | **1.952e-4** | 2.107e-6 |
| high-pass (bounded-integral proxy, fc=30 Hz) | **2.047e-6** | 2.047e-6 | 2.223e-4 | 2.112e-6 |

So the "1100x" is an **X-axis** number (2.19e-3 -> 2.05e-6 = 1070x). On the DOMINANT Y axis the high-pass
gives 117x and is slightly WORSE than plain mean removal (2.22e-4 vs 1.95e-4). The absorber band is genuinely
preserved (Y band unchanged to 0.3%). Quoting 1100x as the method's characteristic gain overstates it.

**And: `dC` is NOT unbuilt.** `scripts/gantry/diagnostics-drift/dC_boundedintegral_train.py` exists (written
2026-07-09 23:50) and `dC_boundedintegral_train_V1.npz` exists (23:52) containing a trained-with-the-
constraint result: tail drift `[1.33e-7, 1.32e-7, 2.19e-4]`, slope `[2.6e-8, 2.6e-8, 6.8e-8] m/s` (flat, so
bounded), band `[3.05e-8, 3.02e-8, 2.10e-6]`, `bestfit = 7.788e-5`. This appears in NO doc, NO decision entry
and NO run-table row. **Provenance is questionable**: the saved npz lacks the `sim_rms_full` field the
current script writes, so the file came from a different script revision. Treat it as a STRONG PRIOR that
Route A trains without drift, and as an action item to re-run and log, not as a result to cite. Note also
that its Y tail (2.19e-4) is the encoder-IC bounded offset, not ANN drift, and its band error equals the
unconstrained checkpoint's, i.e. **it did not demonstrably learn the absorber either**.

### 2.8 OVER-CLAIM 6: "ARTBP reduces the free-run drift"
ARTBP is established as a **DC-collapse** mechanism at 1 epoch, lr=1e-7, with the fit gate explicitly weak
(held-out nf-RMS ~1.2e-3 = baseline level). Gate-2 (converged 20-epoch fit + free-run drift eval) is BUILT
and PENDING; it has never been run. Meanwhile in the baseline-null ARTBP made things WORSE (dY-DC -1.09e-7 ->
-2.82e-7, free-run 1.24e-4 -> 3.0e-4 with a 1.3e-3 spike), because with no signal-driven bias to remove, its
unbounded variance on the z=1 mode dominates. The paper's variance bound requires geometrically decaying
memory, which a pole-1 mode does not have; the poly-tail is a mitigation, not a guarantee (measured variance
reduction versus geometric is 2-5x, after the 1-seed probe's 24-47x was corrected by the 5-seed grid).
**Verdict**: ARTBP is a demonstrated DC intervention and an untested drift fix, with a live variance risk.

### 2.9 Direct contradictions still open in the record
1. **Problem 2: exponential or marginal?** `test_efolding` (2026-07-23): EXPONENTIAL wins, pole/step
   1.00006-1.00009 > 1, r stabilizes, tau 2.9-4.2 s. The handoff START-HERE block (2026-07-24) says D4/D8
   REFINE it to MARGINAL/QUADRATIC (pole ~1) and calls the 1.00016 frozen pole a defective-eigenvalue
   artifact. Both are in `tasks/handoff.md`, one day apart, one checkpoint, and they imply DIFFERENT cures
   (a stability constraint versus conditioning). **Unresolved. This is the highest-value cheap test left.**
2. **Loss stance toward the DC**: PREFERS (d8, val windows, -2.0 SE) / NEUTRAL (d12, training windows, +0.71
   SE) / COSTS 2% to remove (d14, +2.0 SE) / STRICT MINIMUM at zero with huge curvature (curvature test).
   These are reconcilable (val-vs-train set, shallow valley, position-vs-dY channel) but no document
   reconciles them, and the reconciliation changes which cure is indicated.
3. **DC share of the drift**: 133x collapse on the old rough checkpoint (d6), **2.6x on the cleanest
   purpose-built checkpoint 71167** (f05), ~80% in the null (`gain_vs_dc`). "The DC IS the drift" is
   checkpoint-dependent; on the best-trained model available a state-dependent component dominates Y.
4. **"Multiple shooting was tried and failed"** rests on Optuna 69399 (best = epoch 0), which the run table
   itself flags as **confounded by the D-101 lr bug** (before D-101 every gantry run silently trained at Adam
   default 1e-3 instead of the configured lr). It is repeated as settled in `tasks/lessons.md`. It is not a
   clean refutation.
5. **Velocity time constants disagree**: `gantry-zero-mean/README.md` states tau_X = 21.05 s, tau_Y = 0.049 s;
   `drift-diagnosis-status.md` and `dissipative-block-spec.md` state tau_X = 1.55 s, tau_Y = 1.01 s. The
   bounded-impulse no-drift proof is load-bearing on `c = m/tau > 0` and quotes the second pair. One is wrong.

### 2.10 A scope caveat that affects several "established" numbers
The clean `baseline-null` harness runs a 6-state pure baseline with `nx_ann = 0` and the ANN routed to
physical states 0..5, i.e. **directly onto the POSITION rows**, which the deliverable architecture does not
do (it writes force to velocity rows and carries 2 latent aug states). The curvature ranking (X/Y position
stiffest) is partly a property of that routing. This was partially de-risked: `GV_ROUTE=3,4,5` reproduced the
same mechanism (DC displacement dominant, same anti-damping Jacobian signs), so the conclusion transfers.
But quantitative curvature numbers should not be carried into the deliverable configuration unchecked.

---

## 3. Every fix tried, and the real verdict

| # | Fix | Where | Real verdict |
|---|---|---|---|
| 1 | Longer fixed BPTT window (nf) | 71013, 70903, d7/d8, v3nf800 | **REFUTED.** DC ~ 1/nf, nonzero at every finite nf up to 3200; no nf beat the epoch-0 free-run. Not "insufficient", actively refuted. |
| 2 | Lower lr / lr tuning | `lr_sweep`, Optuna 69399, run 71167 | **NOT A LEVER.** Displacement is proportional to lr, and lr -> 0 means no training. 71167 (cleanest monotone descent) still drifted. |
| 3 | SGD instead of Adam | v3x0 Exp 6, TASK 1, TASK 5, TASK 6 | **ELIMINATED as a standalone fix.** 2000x less DC, but the null pass is inaction, and with a real residual SGD learned +0% while drifting 83x. Single-knob R2/R4 tradeoff. |
| 4 | Naive long-horizon / multiple shooting | Optuna 69399, step 3a (NF 900) | **NEGATIVE, one arm confounded.** NF 900 was WORSE than NF 300 at every lr tried (diverged at 1e-3; +19% then -5% at 3e-4 with Y drift growing). 69399 is lr-bug confounded. Long horizon through a marginal mode is ill-conditioned; this is consistent, not conclusive, and multiple shooting proper was never cleanly run post-D-101. |
| 5 | Lipschitz / spectral cap (D-118) | v6, v6b | **REFUTED as a cure.** Trained ANN's natural Lipschitz ~5e-4; no cap that leaves the correction alive can bind; Y destabilizes 114-270x regardless. Magnitude is the wrong axis; sign/structure is the problem. |
| 6 | Strict contraction / REN / passivity | `passive-augmentation`, `diagnostics-literature.md`, §5j | **RULED OUT for pole-1.** Contraction pulls the marginal pole strictly inside. Separately measured: passivity bounds VELOCITY not POSITION (`p1_drift_probe`: passive block seeded with stored energy, env ratio 1.62, position grows O(sqrt(T)) by Cauchy-Schwarz). |
| 7 | Zero-mean / DC mean penalty | d6, `zeromean_pin.py`, v7 | **WORKS IN SIM, DEMOTED.** Removing the DC cuts drift 133x on the old checkpoint (2.6x on 71167). Fails knowledge-free: real friction carries a net impulse, so the penalty suppresses it (measured later: +41% -> +4% learned). |
| 8 | Encoder fix (na = 27, one absorber period) | SLURM 70558, d10, d11 | **FALSIFIED as a DC fix** (criterion 2 failed). Succeeded as an ENCODER fix (val nf-RMS floor halved). Two different problems. |
| 9 | Broadband (1-200 Hz) excitation | v3joint | **REFUTED as the cause-fix.** DC identical to narrowband (-4.18/-3.72/-3.82e-6 vs -4.21/-3.62/-3.55e-6). Identifiability-by-excitation does not remove it. |
| 10 | Pole perturbation (artificial stiffness) | v3pole1k | **INCONCLUSIVE, invalid knob.** Decay is damping-limited, so stiffness cannot move the within-window pole; it only injected a fit-wrecking 300 N force (sim-RMS 650x worse). |
| 11 | ARTBP (unbiased truncated BPTT) | v12, Phase B/B0/D | **PROVEN as a DC intervention; UNTESTED as a drift fix.** DC collapses 20-90x, sign scatters, poly6 best (variance 2-5x below geometric). Gate-2 pending. Made the NULL worse (variance on z=1). |
| 12 | Telescoping bounded-impulse output (Route A) | `GV_TELESCOPE=1`, dB, dC | **PROVEN no-drift by construction** (1.0x floor, B1 unit test, pole shift 3.2e-7 so marginal-preserving). **Sacrifices expressivity**: forbids ANY non-conservative net impulse, i.e. Coulomb friction and preload, not merely "DC". Expressivity NEVER tested (the null has nothing to learn). Orphan `dC` result exists and is unlogged. |
| 13 | Orthogonal projection re-aimed at the measured DC direction (Route B, the thesis contribution) | orth-projection step0-8b, d14/d15/d16, step 4 | **BUILT and formula-validated; NEGATIVE on its first real test.** Rank-1 pin is dodged into the orthogonal DC direction; soft beta saturates under Adam from 1e3 to 1e12. Testbed was noisy, so efficacy is not settled, but both failure MECHANISMS are structural. |
| 14 | Velocity-row-only routing | `GV_ROUTE=3,4,5`, D-066, 68458 | **WORSE on Y** (2.4x) because a velocity DC double-integrates. Physically appealing, structurally useless here. |
| 15 | Theta-only routing | D-068 / D-103 | **FORBIDDEN as deliverable** (coupling uncapturable). Diagnostic baseline only. |
| 16 | Velocity/acceleration-domain loss | standing constraint | **LAST RESORT, supervisor-gated.** Not tried, deliberately. |
| 17 | State-consistency regularizer (Sertbas-Kumbasar Eq 13 without Schur) | sweep Direction 5, all-five spec Layer 1 | **NEVER BUILT.** Named in the plan repeatedly, never implemented or tested. |
| 18 | Y-scheduling off a de-drifted / exogenous Y (Layer 3) | d13, `test_self_scheduling` | **NECESSITY REFUTED on this machine.** Detune is second-order; teacher-forcing true Y changes the Y drift by nothing. R5 largely collapses into R4 here. |

**Reading the table**: exactly two things have ever demonstrably stopped drift on this system: removing the
DC post hoc (a cheat that suppresses friction), and the telescoping output parametrization (a class
restriction). Everything estimator-side is either refuted, confounded, or untested at the horizon that
matters.

---

## 4. The sharpest open question

Two candidates; the first is the decision-critical one.

### 4.1 Primary: can a soft, subspace-selective regularizer bite at all under a scale-invariant optimizer, and if it can, what subspace does it need to be?
The step-4 result decomposes into two obstacles that together kill the plan as currently written:
- **Optimizer obstacle**: with Adam, a penalty's effect saturates in beta (bit-identical 1e3 -> 1e12), because
  the update tends to `lr * sign(grad)`. The reachable displacement floor is set by lr, not by beta. So the
  Layer-2 plan ("add `beta ||Q^T f||^2`") has a ceiling that no tuning can lift. Candidate escapes:
  decoupled/proximal application of the penalty (AdamW-style, applied outside the adaptive preconditioner),
  projecting the UPDATE rather than penalizing the LOSS, a hard constraint or Riemannian/manifold step, or a
  separate non-adaptive optimizer on the parameter group that carries the constrained direction.
- **Subspace obstacle**: the drift direction is not a fixed rank-1 object. A rank-1 pin is dodged; a full
  velocity-DC pin equals the mean penalty and kills friction. And d16 rules out choosing the subspace by
  information content, because on a K=0 axis a DC is the MOST informed direction per unit amplitude.
So the open question is: **what data-derived object separates a spurious persistent force from a genuine
dissipative one, given that direction and information both fail?** The only candidates left on the table are
(i) frequency-band structure relative to the residual's own spectrum, (ii) free-run CONSEQUENCE (a penalty
priced on accumulated position rather than on the force), and (iii) the POWER SIGN `F . v` (the one criterion
that provably distinguishes drift from friction, and which unfortunately is a class restriction if imposed
hard, though it may be usable as a soft/steering term).

### 4.2 Secondary but cheap and blocking: is the Y destabilization a genuine pole above 1, or marginal forced growth?
The record contains both verdicts, from the same week, on one checkpoint. This determines whether the Y half
needs a stability-type structural constraint (D-117 route) or is reachable by conditioning (ARTBP already
suppresses the synthetic analogue at the existing cap, `kappa_g(H)` growing ~H^3.7 identically to the DC's).
Resolving it costs one repeat of `test_efolding` on the second checkpoint (`data/72659/...ep8_h1600...pt`).

---

## 5. Hard constraints the solution space must respect

These are non-negotiable and any candidate that violates one is out, regardless of drift performance.

1. **Preserve the marginal pole |lambda| = 1 on X/Y.** No artificial damping. Rules out contraction, RENs,
   Lipschitz-by-construction stability, strict-passivity (R > 0) pHNN, and any Schur-stabilizing regularizer.
   Test: `pole_check.py` on the TRAINED map, X/Y `|lambda|` within measurement of 1, none above 1.
2. **Full expressivity (the most important requirement per the user).** No hard, for-all-weights,
   class-restricting constraint as the DELIVERABLE, because the true residual is unknown and unverifiable.
   The bounded-impulse/telescoping block is a REFERENCE arm, not the deliverable.
3. **Must NOT forbid a DC-carrying friction.** Coulomb friction carries a net impulse and therefore a
   velocity-row DC. Any mechanism whose no-drift guarantee comes from forbidding net impulse (telescoping,
   mean penalty, high-pass factoring, zero-net-impulse parametrizations) suppresses exactly the physics the
   real-data deliverable must learn. Corollary: no-drift must be TRAINING-CONDITIONAL (estimator-side), which
   the project has already proven is the logical price of expressivity.
4. **The fix acts on the training/estimator, not on routing or input.** X and Y stay in the ANN routing
   (D-103); the ANN keeps the full `[x, u]` input (never buy a property by amputating inputs); Theta-only is
   a diagnostic only.
5. **Position-domain loss.** Velocity/acceleration-domain training loss and velocity-only input are
   supervisor-gated LAST RESORT. A telescoping FORCE on velocity rows that keeps the position loss is NOT
   this, but is still gated by constraint 2.
6. **Knowledge-free target.** Any pinned subspace must be derived from the DATA or from the KNOWN baseline
   structure, never from an assumption about the unknown residual. It must move when the excitation moves.
7. **Judgment discipline.** Per-axis (X and Y separately), against the MEASURED ANN-off / noise floor, never
   an oracle threshold; free-run judged by position ENVELOPE ratio, not slope; identification quality judged
   by WINDOWED nf-RMS, never by free-run (which conflates fit with drift).
8. **The Gyorok orthogonal projection is the thesis's scientific contribution** and must remain the central
   mechanism being extended, not be replaced by a hand-rolled proxy. `kamtin-fp-model/` and
   `model_augmentation/` are read-only.

---

## 6. What I would test next (ranked, none of it done in this session)

1. Re-run `test_efolding` on the second checkpoint to settle exponential-vs-marginal for Y (4.2). Cheapest
   test with the biggest branch consequence.
2. Re-run and LOG `dC` (train with the telescoping parametrization) on the injected-Coulomb rig, measuring
   R2 (windowed nf-RMS on the friction) as well as R4. This converts the orphan result into either a
   verification reference or a refutation, and it is the missing expressivity test for Route A.
3. Test whether ANY penalty formulation escapes beta-saturation under Adam (decoupled/proximal update,
   update-space projection, hard constraint, per-group optimizer). If none does, the entire Layer-2 plan
   needs re-architecting before any more efficacy runs.
4. Run ARTBP gate-2 (converged 20 epochs) with the per-axis free-run drift eval. It is built. Until it runs,
   "ARTBP reduces drift" is unsupported.
5. Reconcile the loss-stance contradiction (2.9 item 2) with one paired test that reports the DC's loss cost
   on TRAIN and VAL windows, per channel, at the deliverable routing.

---

## 7. Corrections to `docs/drift-problem-research-brief.md`

The brief is a useful map but it centers one session's framing and carries errors that would misdirect the
literature search. Corrections, in order of severity:

1. **Section 2A is refuted.** "Encoder-init-compensation DC (dominant on real data), the closed causal chain"
   is the d8/d9 story that SLURM 70558, v3x0, d11 and d12 all falsified. Encoder-init sets the bounded FLOOR;
   it is not the DC's cause. Literature question 7 ("is the correct fix to constrain the ENCODER?") should be
   demoted to a floor-reduction question, not a drift-cause question.
2. **Section 1's "it happens even in a perfect-model NULL" is 40x smaller than the real effect**, and v8
   found no sign-locked DC at all in an exact system == model test. The problem is mostly signal-DRIVEN
   estimator bias, not a signal-independent artifact.
3. **The "1100x" bounded-integral figure is X-only.** On Y it is 117x and slightly worse than plain mean
   removal. Corrected table in 2.7.
4. **`dC` is not unbuilt.** A script and a result exist, undocumented, with a promising but
   provenance-questionable outcome (2.7).
5. **Missing from the brief entirely**: the beta-saturation-under-Adam mechanism (2.6), which is arguably the
   most consequential recent finding, and the d13 / `test_self_scheduling` result that largely dissolves R5
   on this machine.
6. **The DC-is-the-drift number is checkpoint-dependent** (133x old checkpoint, 2.6x on 71167). The brief
   quotes 133x as established.
7. **"Multiple shooting was tried and failed" is confounded** by the pre-D-101 lr bug and should be treated
   as untested, not refuted, when reading the literature on multiple shooting.
8. The brief's framing that the drift lives in a "loss-flat" direction is superseded: the loss is stiff on
   X/Y and soft only on dY; the operating point is displaced, not undetermined.

## 8. Refined questions for the Phase 2 literature search

Derived from this analysis, replacing the brief's section 4 where they differ:

- **Q1** Implicit bias and step geometry of adaptive (sign-like) optimizers on STIFF but badly approached
  directions of a simulation-error loss, and on near-unit-root / marginal modes specifically. Not "flat
  direction wandering" (that framing is refuted here) but "curvature-blind parking at ~lr from a stiff
  minimizer".
- **Q2** How to make a subspace-selective regularizer BITE under a scale-invariant optimizer: decoupled /
  proximal penalties, projection of the update rather than the gradient, constrained or Riemannian
  optimization, per-parameter-group optimizers, hard equality constraints on a learned block's output.
- **Q3** Separating a spurious persistent force from a genuine dissipative one on a marginal mode when
  direction fails (rank-1 pins are dodged) and information fails (d16). Specifically: free-run-consequence /
  multi-step accumulation penalties; frequency-selective residual priors; power-sign (F . v) as a SOFT
  steering term rather than a hard class restriction; cyclo-dissipativity and indefinite storage.
- **Q4** Truncated-BPTT bias and its unbiased corrections (ARTBP and successors) on modes with NO geometric
  memory decay: variance behavior at rho = 1, whether unbiasedness at constant step size helps at all, and
  practical variance control.
- **Q5** Grey-box / physics-augmented learning that PRESERVES a marginal pole while keeping full
  expressivity: conservative port-Hamiltonian (R = 0), do-no-harm / W-PGNN, Gyorok orthogonal projection and
  the orthogonal-by-construction line, Negative-Imaginary with semidefinite storage. Which of these keep
  pole = 1 AND admit a DC-carrying dissipative residual?
- **Q6** Integrator factoring (Tustin-Net and relatives): can a bounded-integral / high-pass output
  factoring be modified to still admit a genuine net impulse, or is the exclusion structural?
- **Q7** Exponential-versus-marginal discrimination for a learned rollout mode, and stability constraints
  that act only on a designated non-marginal subspace (the "exact marginal-mode carve-out" gap).
- **Q8** Bias and consistency of learned state estimators / encoders (SUBNET, deep state-space) on
  integrator modes, scoped as a FLOOR problem, not a drift-cause problem.
