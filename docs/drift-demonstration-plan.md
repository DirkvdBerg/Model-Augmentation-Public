# Drift Demonstration Plan (supervisor meeting)

**Purpose**: demonstrate to the supervisor (Jan) WHERE the augmentation drift comes from, using a small
set of CLEAN controlled comparisons, and confirm the fix direction (non-energy). Clean scripts live in
`scripts/gantry/drift-demo/` (new folder, purpose-built; distinct from the exploratory `diagnostics-drift/`
tree). Verified physics (`diagnostics-drift/drift_common.py`) and the pipeline (`gantry_dynamic/`) are
REUSED, not re-derived; only the demonstration drivers are new and clean.

Companion (full diagnosis, do not duplicate): `docs/drift-diagnosis-status.md` (§3b encoder mechanism,
§3c d17 decomposition, §10 diagnostic index).

## 1. The issue in one paragraph (and reconciliation with Jan)

The X/Y translational axes are stiffness-free (K=0) free integrators. On such an axis the constant-force /
rigid-body (DC) direction is left UNCONSTRAINED by the identification, because (a) the zero-mean narrowband
excitation carries no information about it and (b) the 0.1 s windowed simulation loss cannot see its
free-run consequence. With no restoring stiffness, any persistent term left in that direction integrates
into position drift. It is an IDENTIFIABILITY problem, not an energy/dissipation problem.

Jan's message reconciled: he is right that it is NOT an energy drift and the force SHOULD be zero-mean, and
that an energy constraint is possible but not necessary. The missed piece is that zero-mean is EXPECTED but
never ENFORCED on the unconstrained K=0 direction, and that translational X/Y have no stiffness (unlike
every benchmark his framework was validated on, MSD / Bouc-Wen / cascaded tanks, which all have a spring),
so the same small errors that are harmless there become drift here.

## 1b. The causal chain, link by link (each link = one measured fact + its evidence)

The demonstration is this chain; every figure proves one or two links. If any link were false the
chain breaks, and each link states what would falsify it.

| # | Link | Evidence | Falsified if |
|---|---|---|---|
| L0 | STRUCTURE: X/Y rows have zero stiffness -> discrete poles at `z=1` (free double integrators); theta is sprung | model matrices (`_K4` X/Y rows all-zero); C1 eigen-check | X/Y poles strictly inside (drift impossible) or outside (discretization instability instead) |
| L1 | DATA: excitation is zero-mean narrowband 130-180 Hz -> carries NO information about a constant force -> the DC direction on K=0 rows is unexcited/unidentifiable | excitation spectra (demo2); d12 neutrality is the training-level consequence | `u` had informative DC/low-freq content that penalized a spurious DC |
| L2 | LOSS: a DC force moves position only `0.5*(F/M)*t^2` inside a 0.1 s window -> its FREE-RUN cost is invisible to the windowed loss; in-window the DC is neutral (train, d12) to slightly preferred (val, d8) | d7 (drift enters RMS past ~0.5 s); d8; d12 | window RMS at 0.1 s already reflected the drift |
| L3 | ENCODER: velocities are reconstructed from the position window (linear reconstructability map); under this narrowband data the map yields biased initial velocities on the K=0 axes -> bounded `tau*dv` offsets that DOMINATE the untrained free-run error | d17 (X: 1.45e-3 m, matches `tau_X*dvX`); d10-P2. NOT the naive 150 Hz line-aliasing story (d10-P3 refuted it, R^2=0.14) | `enc_IC` did not match `tau*dv`, or true-x0 run carried the same offset |
| L4 | ANN: the shared zero-init net, trained on windows, faces a TINY gradient for the real signal (weak-signal, log Sect. 7) and a FREE direction in the DC (L1+L2) -> the optimizer wanders into a nonzero DC as a byproduct | d12 (loss-neutral, n=120); reproduced on a fresh init (70905 nf=800: monotone sim-RMS degradation) | training left the DC at ~0, or the windowed loss penalized it |
| L5 | FREE-RUN: the DC applies EVERY step -> terminal velocity `F/c` -> position ramps without bound (damping bounds velocity, never position; K=0 has no restoring force) | d6 (Y -> -2.9 cm; removing the DC collapses drift 133x); C7 velocity-vs-position panel | removing the DC did not collapse the drift |
| L6 | METRIC: full-trajectory sim-RMS is drift-dominated -> ANY trained checkpoint scores worse than the zero-init epoch 0 -> the selector always reverts to epoch 0 | 70903 + 70905: best = epoch 0 in EVERY trial, warm and cold, nf 400-3200 | some trained checkpoint beat 8.015e-5 |

One sentence: *the data does not constrain the DC (L1), the loss cannot price its free-run damage
(L2), so training parks a nonzero DC on it (L4), and the stiffness-free integrator amplifies it into
unbounded drift (L0+L5), which dominates the deployment metric (L6); independently, the encoder's
velocity reconstruction seeds the dominant BOUNDED offset of the untrained model (L3).*

## 2. What we must show (2 error sources + 2 framing facts)

Two ERROR SOURCES leave a persistent term in the unconstrained direction:
- **Prong 1, encoder IC**: the encoder reconstructs velocity by DIFFERENTIATING the position window,
  amplifying the 150 Hz absorber ripple into a biased initial velocity on the K=0 axes -> a BOUNDED
  offset (`tau*dv`). Dominant untrained free-run error (d17: 1.45e-3 m on X). Fix is IN-FRAMEWORK encoder
  conditioning / measured ICs (keep the encoder; it is integral to SUBNET).
- **Prong 2, ANN DC**: the trained ANN parks a near-constant DC force on the K=0 rows; it is loss-neutral
  on the training distribution (d12), so training wanders into it; in free-run it integrates to UNBOUNDED
  drift (d6: Y -> -2.9 cm; removing the DC collapses it 133x). Fix is the estimator-side direction pin
  (orthogonal-projection re-aim).

Two FRAMING FACTS make the two sources legible (without them they look like bugs):
- **K=0 structure**: X/Y have no stiffness (unlike the benchmarks); theta has a spring. Same IC error ->
  theta rings down to zero, X/Y park at a permanent offset. This is THE "something we missed."
- **Loss blindness**: the 0.1 s window integrates a DC to a negligible position change; the drift only
  enters the RMS past ~0.5 s. This is why the DC is unconstrained and zero-mean is never enforced.

One open question the demonstration settles: do the two prongs interact (does the encoder IC SEED the
trained DC, d9 val, or is the DC an independent loss-neutral byproduct, d11/d12 train)? -> demo5.

## 2b. Why the ANN only makes training worse (the epoch-0 paradox, stated explicitly)

The observed paradox: EVERY training run, warm or cold, nf 400-3200, ends with "best = epoch 0",
i.e. the best augmented model is the one where the ANN does nothing. Explicitly:

1. At epoch 0 the ANN output is EXACTLY zero (zero-init final layer, `torch_nets.py`), so the
   augmented model IS the baseline: sim-RMS = 8.015e-5 (the encoder-IC + absorber floor).
2. Training must move the weights to fit anything. In weight space there are two relevant
   directions: the REAL residual (absorber band) with a tiny in-window gradient (weak-signal), and
   the DC on the K=0 rows, which the window prices at ~zero (L2) and the data leaves free (L1).
3. A shared net cannot hold the K=0 rows at exactly zero DC while fitting the other rows, and
   nothing in the objective asks it to -> every reachable trained configuration carries some DC.
4. The free integrator converts that DC into free-run drift that DWARFS any in-window gain:
   the window improves by ~1e-6 m (rung 0: 4.34e-5 -> 4.07e-5) while sim-RMS worsens by ~1.8e-3 m
   (8.0e-5 -> 1.9e-3, 24x). The exchange rate is catastrophic and the training objective cannot
   see it (L2), so it keeps paying it.
5. Hence: sim-RMS(any trained checkpoint) > sim-RMS(epoch 0) -> the selector reverts to epoch 0 ->
   "the ANN only makes it worse."

MEASURED support: 70903 (warm): window flat-improving, sim-RMS 8.0e-5 -> 1.9e-3 at nf=400, recovers
through 8.3e-4 (nf=800) to 4.6e-4 (nf=1600) but never below 8.0e-5. 70905 (fresh init per nf):
nf=800 degrades monotonically 3.8e-4 -> 1.26e-3 (training walks INTO the DC); nf=2400/3200 hover
1-2e-4 (longer window prices the DC partially -> slower acquisition) but train nf-RMS stays flat
(no signal learned) and best = epoch 0 in all four trials. Longer horizon slows the drift
acquisition; it never converts into net improvement (consistent with d8: in-window benefit vs t^2
free-run cost cross over only above ~1 s, beyond any feasible BPTT window).

So "the ANN makes it worse" is NOT "the ANN is broken" and NOT "training diverges": it is the
windowed objective buying negligible window gain at an enormous, invisible free-run price, plus a
selector that correctly refuses the trade. The ANN is doing exactly what the loss asks.

## 3. The demonstration ladder (controlled comparisons, one variable each)

Each row isolates ONE variable and answers one yes/no. Comparisons are against BASELINE (ANN off) and
TRUE initial states, as requested.

| # | Comparison (one variable) | Question it answers | Clean script | Prior art (reuse) |
|---|---|---|---|---|
| 1 | baseline free-run: encoder x0 vs true x0 (stage + logical) | Is it the encoder IC? Is it the K=0 structure? | `demo1_baseline_encoder_ic.py` | d17 |
| 2 | `u_w` vs `u_n` + spectra + `delta_a` | Is the data informative / where is the ripple? | `demo2_excitation.py` | d17 excitation |
| 3 | trained: ANN on vs off; DC removed vs kept | Is it the ANN, and specifically its DC? | `demo3_ann_dc_drift.py` | d6 |
| 4 | windowed loss vs free-run horizon | Why can training not see/fix it? | `demo4_loss_horizon.py` | d7 |
| 5 | trained ANN: encoder x0 vs true x0 | Do measured ICs fix the TRAINED drift? (prong 1 vs 2) | `demo5_trained_true_vs_encoder_ic.py` | NEW (extends d6) |

Rows 1-2 pin prong 1 (encoder) + the K=0 structure; rows 3-4 pin prong 2 (ANN DC) + loss blindness; row 5
separates the two in the trained model and answers "do measured ICs solve it."

## 4. The clean scripts (`scripts/gantry/drift-demo/`)

Shared, minimal, one config surface, live per-epoch prints where relevant, sci-notation + explicit legends
on all plots, figures to `simulations/gantry_subnet/diagnostics/drift-demo/`.

- **`demo_common.py`** (shared): SLIM load (train trajectories for the encoder norm + one T-file
  with/without MSD, NOT all val/test/noise, per the speed note), build the pipeline/encoder once, baseline
  free-run (reuse `drift_common.simulate_baseline`), encoder x0 and true x0, amplitude spectra. One place
  for the trajectory choice and horizon.
- **`demo1_baseline_encoder_ic.py`**: baseline free-run from encoder x0 vs true x0, STAGE and LOGICAL,
  `E = R + enc_IC`. Shows prong 1 dominant + the K=0/stiffness contrast (theta rings down, X/Y offset).
- **`demo2_excitation.py`**: `u_w` vs `u_n` time + spectra, `delta_a` spectrum. Shows the absorber breaks
  the loop cancellation (45 N in 130-180 Hz), data informative, `u_w != u_n`.
- **`demo3_ann_dc_drift.py`**: load a trained checkpoint, free-run, measure the ANN mean force on the K=0
  rows, counterfactual (subtract the DC, re-run). Shows the DC IS the drift (collapse factor).
- **`demo4_loss_horizon.py`**: windowed error vs free-run horizon; drift enters the RMS past ~0.5 s while
  the training window is 0.1 s. Shows the loss blindness.
- **`demo5_trained_true_vs_encoder_ic.py`** (NEW, decisive): free-run the TRAINED model from true x0 vs
  encoder x0. Persists from true x0 -> prong 2 independent of the encoder (needs the pin, confirms d12);
  collapses -> the encoder seeded it (confirms d9). NOTE: requires free-running the AUGMENTED model from a
  SPECIFIED x0 (bypassing the encoder); the injection path (`apply_experiment` state seeding vs manual
  interconnect rollout) needs a short investigation before writing. DESIGN WRINKLE (state explicitly): the
  "true x0" exists only for the 6 physical states; the NX_ANN latent states have NO ground truth (arbitrary
  learned basis), so demo5 is a PARTIAL IC substitution (true physical + encoder-or-zero latent) and the
  choice must be stated on the figure.
- **`demo6_objective_split.py`** (C8): parse the 70903/70905 logs (or the histories inside the `_last`
  checkpoints) into the objective-vs-deployment split figure + the cross-nf panel. No simulation.

## 5. Numbers to quote (measured)

| Quantity | Value | Meaning |
|---|---|---|
| absorber signal (target) | 2.2e-5 m RMS | what the ANN must learn |
| encoder-IC offset (prong 1, untrained) | 1.45e-3 m (X) | ~60x the signal, dominant, BOUNDED |
| ANN DC drift (prong 2, trained 12 s) | Y -> -2.9 cm | ~1300x the signal, UNBOUNDED |
| DC removed (counterfactual) | drift / 133 | the DC is the drift |
| injected force (context) | 45 N RMS @130-180 Hz | absorber breaks the loop cancellation |
| loss horizon vs drift onset | 0.1 s vs ~0.5 s | why the loss is blind |

Scale headline: the dominant free-run error (encoder, 1.45e-3) is ~60x the signal we want to learn, so a
chunk of "bad augmentation" is an init/observability artifact, not the model.

## 6. Meeting narrative arc (HYPOTHESIS WALK -- supervisor feedback 2026-07-14: lead with
"what do we expect / what do we see / what explains it", not with conclusions)

1. **Expectation (shared with Jan):** a correct baseline + a zero-mean residual + a servo-verified
   input should give a free-run error at the absorber scale (2.2e-5 m), bounded, and the force the
   ANN learns should be zero-mean. (This IS Jan's stated expectation -- agree first.)
2. **Observation 1 (untrained):** the free-run error is 1.45e-3 m on X -- 60x the expectation.
   *Candidate explanations:* model class wrong / input wrong / initial condition wrong / numerics.
   *Discriminator = F1's four lines:* R (true x0) kills model-class-on-X; F (oracle) kills numerics
   -on-X; enc_IC landing ON the independently predicted tau*dv confirms the IC. -> the ENCODER IC.
   [demo1; demo2 as the input check]
3. **Observation 2 (structure):* the SAME IC error dies on Theta but parks permanently on X/Y.
   *Explanation:* Theta has a spring, X/Y have none (K=0) -- the benchmarks all had restoring
   dynamics, so this case is new to the framework. [demo1 logical panel; F4 poles]
4. **Observation 3 (trained):** every trained checkpoint is WORSE in free-run than epoch 0, at
   every nf 400-3200, warm and cold. *Candidate explanations:* lr wrong / too few epochs / signal
   unlearnable / the objective itself misprices something. *Discriminators:* F3 (window improves
   while free-run worsens => the objective), F2 (the mispriced object, measured: a near-constant
   K=0 force whose removal collapses the drift 133x), C6/d12 (WHY it is unpriced: loss-neutral).
   [demo3, demo6, demo4]
5. **Observation 4 (not energy):** the drifting axis's VELOCITY stays bounded (energy is fine --
   Jan's zero-mean intuition is correct in-window); only POSITION ramps. *Explanation:* a marginal
   pole's forced response, not instability, not energy injection. [F2 panel 2]
6. **The remaining open split** (encoder-seeded vs loss-neutral wandering) and its discriminator
   [demo5], stated as OPEN, with both outcomes acceptable.
7. **Fix direction that follows** (non-energy, as Jan expects): condition the encoder velocity
   step (keep the encoder) + pin the unexcited near-DC direction (orthogonal-projection re-aim).
   Both preserve expressivity, add no stiffness. Then the ask (§11).

## 7. Build order + reuse

1. `demo_common.py` (shared infra, slim load) -- build + smoke first.
2. `demo1_baseline_encoder_ic.py` -- the backbone figure (clean d17).
3. `demo5_trained_true_vs_encoder_ic.py` -- the decisive NEW run (after the augmented-from-x0 injection is
   confirmed).
4. `demo2` / `demo3` / `demo4` -- clean ports of existing diagnostics.

Reuse (verified, do NOT re-derive): `drift_common.py` (truth EOM, `simulate_baseline`, `tau_X/tau_Y`,
P-transform), `gantry_dynamic/` (loader, normalization, `build_model`, `encoder_init_state`). New code is
only the clean demonstration drivers + plots.

## 8. Demonstration rigor: exactly what to print/plot to prove the cause WITH CERTAINTY

Goal: define the cause defensibly (`this IS the problem`) even without a solution. Each claim is a
FALSIFIABLE test, not an illustration. Every demonstration combines as many of these as possible:
**independent measurement** of the cause; **prediction match** (predict the effect from the measured cause
via known physics); **counterfactual** (remove the cause -> effect vanishes); a stated **falsification
condition**. Plot titles POSE the question, never assert the conclusion; the reference/prediction line is
drawn so the viewer judges the match. Save data (`.npz`) + figure to
`scripts/gantry/drift-demo/figures/` (user 2026-07-14).

### The seven decisive tests

**C1 -- X/Y are marginal (K=0), theta is not (the structural enabler).**
- PRINT: discrete eigenvalues of the baseline state-transition `A_d`; X/Y translational modes at
  `|lambda| ~ 1` (repeated `z=1`), theta mode `|lambda| < 1`. Also the continuous poles (0 for X/Y).
- PLOT: pole-zero map on the unit circle -- X/Y ON the circle, theta inside.
- CERTAINTY: direct structural fact from the model matrices (not inferred). Also answers the supervisor's
  "op de rand van stabiliteit": show the poles are AT `z=1` by construction, not pushed OUTSIDE by
  discretization. FALSIFIED IF X/Y poles were strictly inside (then drift is impossible) or outside (then
  it is a discretization instability, a different problem).

**C2 -- the baseline physics is clean (it is NOT the cause).**
- PRINT: envelope-growth ratio `RMS|q|(last third) / RMS|q|(prior third)` for the baseline free-run from
  TRUE x0; expect `~1` (bounded) on X and Y. Contrast with the trained-ANN ratio (C4, `>1.2`).
- PLOT: baseline position from true x0 vs time -- flat / bounded.
- CERTAINTY: counterfactual -- the SAME physics from the correct IC does not drift, so the physics is
  innocent. FALSIFIED IF the baseline drifted from true x0.

**C3 -- the encoder velocity bias is the dominant untrained offset (PRONG 1).** Three-part proof:
- PRINT (measure the cause): `x0_enc - x0_true` per state, especially the velocity rows dX, dtheta, dY.
- PRINT (predict the effect): `tau_X*dvX`, `tau_Y*dvY` (settled offset of a velocity error on a damped K=0
  axis).
- PLOT (prediction match + counterfactual): `E` (encoder x0), `R` (true x0), `enc_IC = E - R`, stage and
  logical, with a HORIZONTAL reference line at the predicted `tau_X*dvX` on the X panel.
- CERTAINTY: measured cause (dv) + predicted effect (`tau*dv`) MATCHES observed (`enc_IC` plateaus at the
  line) + counterfactual (`R` from true x0 is `~0` on X). Lead on X (clean, direct); state that Y is
  cross-coupled (`enc_IC(Y) != tau_Y*dvY`, dominated by dX/dtheta coupling, d17/§3c). FALSIFIED IF
  `enc_IC(X)` did not match `tau_X*dvX`, or `R` also carried the offset.

**C4 -- the trained ANN DC causes the UNBOUNDED drift (PRONG 2).**
- PRINT (measure the cause): ANN output time-mean, rms, and `|mean|/rms` per K=0 row (`|mean|/rms ~ 1` =
  pure DC).
- PRINT + PLOT (counterfactual): free-run with the ANN vs with the measured DC SUBTRACTED; collapse factor
  `drift_with / drift_without` (~133x). PLOT both position traces on one axis.
- PRINT (unboundedness): envelope-growth ratio `> 1.2` (drifting) for the ANN vs `~1` for DC-removed.
- CERTAINTY: measured DC + counterfactual collapse + unboundedness. FALSIFIED IF removing the DC did not
  collapse the drift (then it is not the DC).

**C5 -- the TRAINABLE window is BLIND to the drift, but a LONGER horizon CAN see it (blindness is
horizon-limited, not fundamental).** The proof of "blind" REQUIRES a positive control: a horizon that DOES
see the drift. Without it, "blind" could be misread as "the drift is fundamentally undetectable."
- PLOT: free-run error RMS vs horizon length `T` (log-x), with THREE annotations: (i) the training window
  `T = 0.1 s` marked; (ii) a horizontal DETECTABILITY floor at the absorber RMS `2.2e-5 m` (the physical
  signal we model; on real data this is the measurement noise floor); (iii) the horizon `T*` where the
  drift RMS crosses ABOVE the floor.
- PRINT: drift RMS at `T = 0.1 s` (BELOW the floor = blind), at `T*` (the crossover = first visible), and
  at `T = 12 s` (far above). The positive control is `T*`: a horizon that CAN see the drift exists.
- CERTAINTY: the drift is below detectability at the training window yet rises above it at `T > T*`, so a
  horizon that sees the drift EXISTS -> the 0.1 s window's failure is BLINDNESS (a horizon limit), not
  fundamental invisibility. FALSIFIED IF the drift never exceeded the floor at ANY horizon (then it would
  be undetectable, a different claim, and "blindness" would be the wrong word).
- CRUCIAL CAVEAT (pre-empts "then just use a longer nf") -- **CORRECTED BY MEASUREMENT (2026-07-14,
  f5): T* = 0.235 s, which lies BELOW the ~0.5 s BPTT wall, not beyond it.** The honest two-regime
  statement: below T* (e.g. the 0.1 s window: drift contribution 0.11x floor) the window is genuinely
  BLIND; above T* the drift IS visible in the window, yet the windowed re-init loss still PREFERS /
  tolerates the DC (d8 through nf=2000, d12 neutrality) -- so the failure switches from blindness to
  mispricing, and NO trainable horizon fixes it either way (measured: no nf 400-3200 beats ANN-off,
  F3-B). The estimator-side pin is required. State the two regimes explicitly with C5.

**C6 -- the DC is loss-NEUTRAL, so training parks it (why it is unconstrained).**
- PRINT: paired windowed loss (DC-carrying vs DC-removed) on TRAINING windows: pooled `Delta/SE ~ 0`.
- PLOT: per-window paired `Delta` histogram with 0 marked (d12 style).
- CERTAINTY: if the loss cannot distinguish DC from no-DC, training cannot remove it (it wanders in).
  FALSIFIED IF the windowed loss penalized the DC (`Delta/SE` clearly positive).

**C7 -- it is NOT an energy problem (damping bounds velocity, not position) -- the Jan-facing clincher.**
- PRINT: settled terminal velocity and the position ramp rate on the drifting K=0 axis.
- PLOT: on one time axis, the K=0 VELOCITY (settles to a constant terminal value, BOUNDED) and the
  POSITION (ramps linearly, UNBOUNDED).
- CERTAINTY: shows the velocity / kinetic energy IS bounded (Jan's zero-mean intuition is correct) while
  position drifts, so the mechanism is the position INTEGRATOR, not energy injection. This is why an energy
  constraint is not the fix. FALSIFIED IF the velocity also grew unbounded (then it would be an
  instability / energy problem).

**C8 -- the training objective and the deployment metric DIVERGE (the split; why training cannot
fix it).** The most direct proof that the windowed loss misleads: minimizing it makes the free-run
WORSE.
- PLOT: per training run/rung, twin-axis vs epoch: windowed train nf-RMS (left) and full sim-RMS
  (right, log). Annotate the two arrows: objective down / deployment up. Data: 70903 rung 0
  (window 4.34e-5 -> 4.07e-5 while sim 8.0e-5 -> 1.9e-3) and 70905 nf=800 (sim 3.8e-4 -> 1.26e-3,
  monotone). Add the ANN-off line at 8.015e-5.
- PLOT (companion, cross-nf): best-achieved sim-RMS per nf, warm (70903: 1.9e-3 / 8.3e-4 / 4.6e-4)
  and cold (70905: all >= ~1e-4, bouncing), vs the 8.015e-5 ANN-off line. CAVEAT stated on the
  figure: 8 epochs each, so cross-nf levels are lower bounds on achievable; the claim is only
  "no nf produced net improvement", not "these are converged optima".
- PRINT: per run: window delta, sim-RMS delta, and their RATIO (the exchange rate, ~1:1000 against).
- CERTAINTY: if the window were a faithful proxy, the two would move together; they SPLIT (the
  pre-declared curriculum question, answered). FALSIFIED IF reducing the window reduced sim-RMS.
- NOTE: windowed nf-RMS is NOT comparable across different nf (longer windows accumulate more
  error trivially); compare shapes within a rung, use sim-RMS as the only cross-nf line.

### Why this set defines the cause defensibly
- C1 + C2: the stage (structure permits drift; physics is innocent).
- C3 + C4: the two sources, each proven by measure + predict + remove (see the cause, predict its size,
  delete it and watch the effect vanish).
- C5 + C6: why it persists (invisible to, and unconstrained by, the loss).
- C7: not energy (velocity/energy bounded, position drifts) -- confirms Jan and pins the mechanism.

### Mapping to scripts
- C1 + C2 + C3 -> `demo1_baseline_encoder_ic.py` (add the pole-zero print/plot and the `tau*dv` line).
- C4 + C7 -> `demo3_ann_dc_drift.py` (add the velocity-vs-position panel and the collapse factor).
- C5 + C6 -> `demo4_loss_horizon.py`.
- C8 -> `demo6_objective_split.py` (log/checkpoint-history parsing only, zero simulation).
- `demo2_excitation.py` is context (data informativity), not one of the eight proofs.
- `demo5_trained_true_vs_encoder_ic.py` is the INTERACTION test (does true x0 fix the TRAINED drift):
  strengthens C3 vs C4 by showing which prong survives measured ICs.

## 9. Claims discipline -- exact wording (attackable claims, fixed BEFORE the meeting)

These are the phrasings that survive an adversarial supervisor; do not use the looser versions.

1. **"Blind" means blind to the free-run COST, not to the DC itself.** d8 showed the window CAN
   distinguish the DC-carrying model (and slightly prefers it on val); d12 showed neutrality on
   train. Say: "the windowed loss cannot price the free-run consequence of the DC; in-window the DC
   is neutral-to-slightly-preferred." Never say "the loss cannot see the DC."
2. **Encoder mechanism wording (d10-P3-safe).** Say: "the velocity bias is a measured property of
   the linear reconstruction map under this fully-narrowband excitation (d10-P2)." Do NOT say "it
   mis-reads the 150 Hz line / aliases the ripple" -- that clean attribution was tested and FAILED
   (d10-P3: R^2=0.14, predicted mean wrong sign).
3. **Reconcile "A2 rejected" (d3) with "encoder dominant" (d17) -- state BOTH, they answer
   different questions.** d3/A2 rejected the encoder IC as the cause of the TRAINED cm-scale drift
   (a bounded `tau*dv` offset cannot reach -2.9 cm). d17 shows the encoder IC dominates the
   UNTRAINED mm-scale free-run error. No contradiction; say both sentences together.
4. **Benchmarks claim, softened and verifiable.** Say: "the published benchmarks of the framework
   (MSD, Bouc-Wen, cascaded tanks) all have restoring dynamics, so an unconstrained DC was always a
   harmless bounded offset (`F/k`)." Do NOT claim "the first stiffness-free system the framework
   has ever met" (unverified; wafer-stage work exists in the lineage).
5. **The DC evidence must be multi-checkpoint.** Report the ANN mean-force measurement on >= 3
   independently-trained checkpoints (70905 trials nf=800..3200, 70903 rungs; fetch `_last` files).
   Expected and acceptable: sign/magnitude VARIES across runs -- that is the wandering story, state
   it as such. A single-checkpoint DC claim is attackable as "one broken run."
6. **C6 is a null result -- quote its power.** With n=120 paired windows, state the minimum
   detectable |Delta|/SE alongside "neutral" (and note d14's small window-set-dependent +-2 SE
   effects, <= ~2% relative). The claim is "any in-window preference is orders below the 133x
   free-run damage," not "exactly zero."
7. **The R-offset is NOT evidence for a legitimate persistent DC.** d17's R carries a bounded
   ~+3.8e-4 near-DC offset on Y (open-loop replay effect + rigid-vs-sprung mass handling). A critic
   may say "so the ANN's DC is partly legitimate." Answer: a BOUNDED offset requires a transient,
   settling force; a PERSISTENT constant force produces an UNBOUNDED ramp, which the truth does not
   show. The separator is the envelope: truth bounded, ANN-DC free-run unbounded.
8. **Do not call it instability.** The poles are AT z=1 (marginal), not outside; the drift is the
   forced response of a marginal pole to a persistent input (linear/quadratic growth, not
   exponential). C1 shows this and simultaneously answers "discretisatie op de rand."

## 10. Anticipated questions -- prepared answers

- **"Just train longer?"** More steps descend the same loss, and the loss is neutral-to-preferring
  toward the DC (d8 by SIGN, d12); 70905 nf=800 shows more epochs = monotonically MORE drift. The
  failure is in the objective, not the step count.
- **"Just increase nf?"** Measured, both directions: warm 400-2000 (70903) and cold 800-3200
  (70905), every trial best = epoch 0. The in-window benefit vs t^2 free-run cost cross over only
  above ~1 s (d8), beyond the memory wall (nf=4000 = 566 MB), and the measured cost is 6.4 -> 47
  s/batch for 400 -> 3200. Longer horizon slows drift acquisition (70905), never reverses it.
- **"Add low-frequency excitation so the DC is informed?"** A spurious DC's in-window displacement
  is `0.5*(F/M)*t^2` REGARDLESS of `u` -- richer excitation does not change what the window
  integrates, so it does not fix the blindness. (It does improve identifiability of REAL low-freq
  residuals; flag as complementary, not the fix. Family C, status doc §5m.)
- **"Train closed-loop?"** Hides a biased model (the servo supplies the missing stiffness);
  rejected by D-107 and by Jan ("niet het doel van dit project").
- **"Energy/dissipativity constraint?"** C7: the velocity is already bounded (damping works, energy
  is fine, Jan's zero-mean intuition is CORRECT in-window); the problem is the position integrator.
  Jan: "mogelijk maar niet nodig" -- agreed.
- **"Wrong lr?"** lr=1e-5 overshoots this routing (train loss rises); lr=1e-7 trains theta cleanly
  (learnability control, F8). And d8/d12 are forward-only measurements on the loss surface,
  lr-independent.
- **"Is the baseline/model wrong?"** C2: the baseline from true x0 is clean; d17's E = R + enc_IC
  accounts for the full untrained error with no unexplained remainder.
- **"Only 8 epochs?"** The within-run trajectories are monotone (degrading) or flat; d8's sign
  says more steps move TOWARD more DC. Extrapolating more epochs predicts more drift, not less.
  (Stated as caveat on the cross-nf panel: levels are lower bounds, claim is "no net improvement".)

## 11. Figure budget + the meeting ask

**Lead trio (shown by default, everything else is backup):**
1. **F1** = demo1 (E = R + enc_IC, stage + logical; tau*dv line; theta rings down vs X/Y park).
   Caption-claim: "the untrained free-run error is the encoder IC, bounded, and only the
   stiffness-free axes keep it."
2. **F2** = demo3 (DC counterfactual collapse + velocity-bounded/position-unbounded panel).
   Caption-claim: "the trained ANN emits a near-constant force; removing it collapses the drift
   133x; energy/velocity stays bounded, so it is the integrator, not energy."
3. **F3** = demo6/C8 (objective vs deployment split + cross-nf panel with the ANN-off line).
   Caption-claim: "minimizing the training window worsens the free-run at every nf, so no amount
   of window training fixes it."

**Backups (in reserve):** C1 poles, C5 horizon curve with T*, C6 neutrality histogram, demo2
excitation, F8 learnability control, demo5 IC-swap.

**The ask (end the meeting with decisions, not impressions):**
1. Does Jan ACCEPT the diagnosis (two sources: encoder-IC bounded offset + loss-unconstrained ANN
   DC on stiffness-free axes)?
2. Does he endorse the NON-ENERGY fix direction: (a) condition/replace the encoder initial
   condition (measured ICs on real data; window/na conditioning in sim), (b) pin the unexcited
   near-DC direction estimator-side (the orthogonal-projection re-aim, our contribution)?
3. Practical: HPC limits (his offer), and whether empirical (demonstrated, not guaranteed)
   non-drift is acceptable as the R4 deliverable.

## 12. Figure-by-figure build specs (agreed 2026-07-14; scripts in `scripts/gantry/drift-demo/`)

**Three global rules (decide several designs below):**
- **R-a: every number states its evaluation set.** Three quantities circulate: cropped-val sim-RMS
  (8000-sample search validation; the 8.015e-5 ANN-off line lives HERE), full-V1 12 s free-run
  traces, and windowed nf-RMS. Every figure names its set in the axis label or caption.
- **R-b: checkpoint provenance on every figure** (job id, nf, lr, epochs, warm/cold) as a small
  annotation. Available checkpoints: `trial_ckpts_71013/trial{0..3}_nf{2400,3200,1600,800}` (cold),
  `curriculum_70903/rung{0..}_nf{400,800,...}` (warm), `gantry_drift_last` (dissected, d6-d15).
- **R-c: no twin axes** (invites "you scaled it to look dramatic"); stacked subpanels, shared x.

### F1 (demo1) -- encoder-IC decomposition [untrained baseline, T1, 12 s]
**E/O/E/D block (present in this order):**
- EXPECT: correct baseline + correct x0 + verified input => free-run error at absorber scale
  (~2.2e-5 m), bounded.
- OBSERVE: 1.45e-3 m settled offset on X (60x), 1.5e-4 on Y; Theta rings down.
- EXPLAIN (candidates): model class / input / initial condition / numerics.
- DISCRIMINATE: R (true x0) ~ 3e-5 on X kills model-class-on-X; F (oracle) = 1.8e-6 on X kills
  numerics-on-X; enc_IC plateaus AT the independently predicted tau_X*dvX = +1.28e-3 => the
  ENCODER INITIAL CONDITION. (MEASURED 2026-07-14, full 12 s run.)
- Two figures: STAGE (X1/X2/Y) and LOGICAL (X/Theta/Y), 3 stacked panels each.
- Lines: `E = free-run error @ encoder x0 (sim - y, with-MSD)` (red); `R = residual @ true x0
  (= absorber + replay, the ANN target)` (green); `enc_IC = E - R (pure encoder-IC effect)` (blue).
- References: dotted +-sigma(delta_a)=2.2e-5 m ("absorber displacement RMS = residual to learn");
  on the X panel ONLY: horizontal dashed line at predicted `tau_X*dvX = +1.28e-3 m`, legend
  "predicted settled offset tau*dv (from measured encoder velocity error)". Print measured dv on-figure.
- Certainty logic: R is the one-variable counterfactual; enc_IC plateauing AT the independently
  predicted line = prediction match; measure+predict+remove.
- Attacks + closures: "one trajectory" -> stated; V1 companion as BACKUP (same mechanism, different
  axis: dX on T1, dY on V1 -- strengthens: mechanism is the map, axis follows the data). "Y doesn't
  match its prediction" -> pre-empted in caption (Y is cross-coupling dominated, 9x single-axis
  prediction; X is the clean case, hence the line on X only). "Is it the MSD?" -> Theta panel rings
  down; R carries the ripple; F7 is the deeper backup.
- Optional add-on: enc_IC at na=17 vs na=27 (the conditioning lever; measured for dY/V1 in d10,
  NOT yet measured for dX/T1 -- label accordingly). na=17 stays the demonstrated pipeline.
- FOURTH LINE (added 2026-07-14): `F = oracle (FP + true MSD) @ true x0 - y` (dark grey), the
  DISCRETIZATION floor (oracle.py, D-097-verified, pipeline ts/up_sample per the fairness rule).
  Value: (i) converts the §3c `[INFERRED]` R-offset attribution into a measurement (F ~ 0 => R's
  offset is model-class, not numerics); (ii) pre-empts "is R just integration error?"; (iii)
  answers the supervisors' discretization hypothesis quantitatively at pipeline conditions.
  Label on-figure: "sim-only reference"; NEVER an acceptance bar (thresholds stay data-derived).
  NOTE: the oracle ceiling was CONSIDERED for F3-B and REJECTED -- the F3-B conclusion ("no
  trained model beats ANN-off") is unchanged by it, and an oracle line on a performance plot
  reads as an acceptance target (the exact confusion the lessons forbid).
  **MEASURED (2026-07-14, full 12 s): F(X1/X2) = 1.8e-6; the initially observed F(Y) = -1.1e-4
  was OUR OWN backward-FD vdelta_a seed (OE-1, CLOSED, §13): with the central-difference seed
  F(Y) = +5.5e-6. Honest per-axis numerics floor: ~1.8e-6 (X) / ~5.5e-6 (Y), both below the
  absorber RMS -- the attribution is airtight on BOTH axes.** demo1 uses the central-diff seed.

### F2 (demo3) -- trained-ANN DC counterfactual + not-energy panel [V1 full, 12 s]
**E/O/E/D block (present in this order):**
- EXPECT: a well-trained residual ANN's force is zero-mean (Jan's expectation, and the true
  residual IS zero-mean); the free-run should not degrade with training.
- OBSERVE: the trained free-run drifts unbounded (Y -> cm scale); the ANN's dY-row output is
  ~pure DC (|mean|/rms = 0.997).
- EXPLAIN (candidates): broken training / energy injection / a persistent unpriced force.
- DISCRIMINATE: subtracting ONE constant per row collapses the drift ~133x (=> the DC IS the
  drift); the velocity panel stays bounded (=> not energy; marginal-pole forced response); the
  cross-checkpoint panel shows the DC on every independently trained checkpoint (=> not one
  broken run); C6/d12 show the loss cannot price it (=> why training parks it).
- Headline: ONE checkpoint = `gantry_drift_last` (fully dissected provenance), 3 stacked panels:
  1. Y position error: `trained ANN (full)` (red) vs `trained ANN, measured DC subtracted` (blue)
     vs `baseline / ANN off (same encoder x0)` (grey). Legend states the collapse factor (133x).
  2. Y VELOCITY error, same run: red settles to a bounded terminal value. Caption: "velocity
     bounded = energy bounded (damping works); the drift is position-only" (the Jan panel, C7).
     Claim BOUNDED velocity, not "terminal" (tau_Y is seconds-scale; may still vary within 12 s).
  3. ANN force on the dY row vs time: near-horizontal, annotated `|mean|/rms = 0.997 (pure DC)`.
- COMPANION PANEL (the A2 closure, mandatory): bar chart of the K=0-row DC (mean force, rms
  whiskers) for ALL checkpoints (71013 trials 0-3 + 70903 rungs + gantry_drift_last). Expected:
  nonzero DC everywhere, sign/magnitude varying = "training generically parks a DC."
  HONEST RISK (pre-declared): if some checkpoints show ~zero DC, show it anyway and soften.
  **MEASURED (2026-07-14, all 9 checkpoints, 2 s V1 free-run each): the dY-DC is present in
  EVERY checkpoint (risk did not materialize), ALL NEGATIVE, and its magnitude DECREASES
  MONOTONICALLY with nf in BOTH families (cold: -2.4e-6 @800 -> -4.5e-7 @3200; warm: -3.4e-6
  @400 -> -6.3e-7 @2000) -- the mechanistic counterpart of F3-B (longer window prices the DC
  more, so the optimizer parks less of it). Headline 12 s run reproduces d6 exactly: collapse
  133x, envelope 1.78 -> 0.95, velocity bounded (max 9.7e-3 m/s). The SIGN consistency across
  independent inits is a NEW systematic finding -> OE-2 (§13).**
- Attacks + closures: "cherry-picked broken run" -> companion panel. "DC measured on the run you
  subtract it from = circular" -> d15 stationarity (-0.7% over 12 s) + DC-removed lands on the
  ANN-off level (one constant/row explains a 260x effect). "grey line has its own offset" ->
  deliberate: layers prong 1 (grey, bounded ~1e-4) under prong 2 (red, unbounded ~1e-2); say so.

### F3 (demo6) -- objective vs deployment split + the Roland cross-nf figure [logs/histories only]
**E/O/E/D block (present in this order):**
- EXPECT: if the windowed loss is a faithful proxy, minimizing it should improve (or at worst not
  hurt) the free-run; and per the supervisors' suggestion, a longer window should progressively
  fix any horizon blindness.
- OBSERVE: the window improves while the free-run worsens 24x (rung 0); across nf 400-3200, warm
  and cold, no trained model ever beats ANN-off; longer nf slows the degradation but never
  reverses it.
- EXPLAIN (candidates): lr / epochs / unlearnable signal / the objective misprices the drift.
- DISCRIMINATE: the SPLIT itself (objective down, deployment up) rules out lr/epochs as the
  driver; d8's forward-only sign result (the window PREFERS the DC through every feasible nf)
  rules out "just train longer/longer nf"; F8 (the machinery learns what it can see) rules out
  "nothing is learnable". => the objective cannot price the free-run cost of the DC.
- Figure A (the split), 2x2 stacked (NO twin axes): columns = warm (70903 rung 0, nf=400) and cold
  (71013 trial 3, nf=800). Top: train nf-RMS vs epoch (linear). Bottom: val sim-RMS vs epoch (log)
  with the ANN-off line labeled "ANN off (epoch 0), cropped-val sim-RMS". Annotate per column the
  exchange rate ("objective -6% / deployment x24").
- Figure B (Roland: "increase the window"), cross-nf summary: x = nf (log2, 400-3200), y = val
  sim-RMS (log). Per nf: start / best / end markers. WARM = circles, CONNECTED (one continuous
  model); COLD = squares, independent. ANN-off line across. Two annotations: "warm: longer window
  pulls an acquired DC down (1.9e-3 -> 8.3e-4 -> 4.6e-4 -> rung-3)" and "no point ever crosses
  below ANN-off". Caption caveat: 8-epoch budgets, levels are lower bounds; claim is "no net
  improvement", not "converged optima".
- Attacks + closures: "warm and cold on one axis" -> different marker families, connected vs not,
  caption states warm = recovery dynamics, cold = acquisition dynamics; their AGREEMENT from
  opposite directions is the strength. "window nf-RMS not comparable across nf" -> never plotted
  across nf (only within a run, Fig A top). "lr/epochs?" -> prepared answers (§10) + F8.

### Backups (build cheap, show only if probed)
- **F4 poles (C1):** unit-circle map, discrete baseline A at T1's Y_op; "X/Y: repeated pole at z=1
  (Jordan) = free integrator BY CONSTRUCTION, not discretization"; Theta pole inside. Answers "op
  de rand van stabiliteit" in one glance.
- **F5 horizon curve (C5):** cumulative free-run RMS vs horizon T (log-x) from F2's red trace;
  floor at sigma(delta_a); marks at 0.1 s (training window, below floor) and T* (crossover).
  Caption: T* is beyond the feasible BPTT window; within the trainable range the window PREFERS
  the DC (d8).
- **F6 neutrality histogram (C6):** d12 per-window paired Delta histogram, zero marked, plus the
  minimum-detectable-effect sentence (null-result power).
- **F7 excitation:** the existing d17 figure (already to spec).
- **F8 learnability control:** in-hand version = 70903 rung 0 val nf-RMS improving 3.18e-5 ->
  2.90e-5 ("the machinery optimizes what it can see") -- labeled exactly as that. UPGRADE if the
  old Theta-only run's artifacts are located (clean monotone val improvement at lr=1e-7).
- **F9 (demo5):** NOT brought to the meeting unless built and clean; the latent-state design
  decision (true physical + encoder-or-zero latent x0) comes first.

### Build order
`demo_common.py` -> F1 (demo1) -> F2 (demo3, incl. the cross-checkpoint DC companion) -> F3
(demo6, log parse) -> backups F4/F5 (minutes each, reuse F2's trace) -> F8 -> (decide) F9.

## 13. Open expectations (hypothesis discipline, live list)

Working rule (supervisor feedback 2026-07-14): state the expectation BEFORE looking; log every
deviation here with its candidate explanations and the discriminating test; close items by
measurement, never by assumption.

- **OE-1 (CLOSED 2026-07-14, verdict = (b) the vdelta_a SEED): the oracle Y floor was our own
  differentiated-velocity seed.** EXPECTED F ~ 0 on all axes (model class = truth). OBSERVED
  (demo1, 12 s): F(X) = 1.8e-6 (as expected) but F(Y) = -1.1e-4 = 5x absorber RMS. CANDIDATES:
  (a) RK4@up_sample=2 under-resolves the 150 Hz resonance; (b) the backward-FD vdelta_a seed;
  (c) replay/resampling mismatch. DISCRIMINATORS (measured, scratch `oe1_oracle_upsample_check`):
  up_sample=8 gives F(Y) IDENTICAL to up_sample=2 (ratio 1.000) -> (a) REFUTED; zeroing the vda
  seed swings F(Y) to +3.8e-4 -> seed-dominated; a CENTRAL-difference (O(Ts^2)) seed collapses
  F(Y) to +5.5e-6 -> (b) CONFIRMED. FIX: demo1's oracle now seeds vdelta_a by central difference;
  the honest per-axis numerics floor is ~1.8e-6 (X) / ~5.5e-6 (Y). NOTE the echo: velocity from
  differentiated position, biased, converted by the K=0 coupling into a settled offset -- the
  SAME mechanism class as the encoder prong, caught in our own diagnostic seed by stating the
  expectation first. (Side observation, unexplained: with the vda seed zeroed the oracle's Y
  offset +3.800e-4 numerically matches R(Y); plausibly the missing initial absorber momentum;
  not load-bearing, left open.)

- **OE-2 (OPEN, 2026-07-14): the dY-DC SIGN is consistent across independent inits.** EXPECTED
  (loss-neutral wandering, d12): random sign across seeds. OBSERVED (f2 companion panel): all 9
  checkpoints NEGATIVE on the dY row (5 independent inits: 4 cold seeds + drift_last from a
  different config; the 4 warm rungs share one init), magnitude ~1/nf-monotone in both families.
  A directionally SYSTEMATIC bias, not pure random walk. CANDIDATES: (i) the d9 encoder-window
  re-init geometry rewarding a negative dY-DC beyond the first moment (d11 refuted only the MEAN
  bias on train windows); (ii) shared-weight gradient geometry coupling the theta/absorber fit
  into a preferred dY-DC sign; (iii) an asymmetry of the training distribution. NOT LOAD-BEARING:
  the drift cause (the DC) and the fix (the pin) are unchanged either way -- but "loss-neutral
  WANDERING" should be stated as "loss-neutral, directionally biased" until this is isolated.
  DISCRIMINATOR (cheap, when wanted): d12-style paired test with a SIGN-FLIPPED DC injection, or
  multi-seed short runs at fixed nf checking sign statistics.

## 14. FIGURE REDESIGN (2026-07-14) -- the critique and the final figure specs

### 14.1 The critique (user + supervisors, collected; drives every spec in 14.3)

1. **First-thought test** (user, on f1_encoder_ic_logical): "my first thought would be if I look at
   the Y-axis, the input or the model is not going good, because even for the real states it drifts
   off. so then we should show (better in a different plot) that the model actually matches the
   input output for these channels." -- a line that raises a question the figure does not answer is
   a liability; answer it in the deck (context figure) or move the line to backup.
2. **Legend/line overload** (user): "i find the legend too chaotic ... unclear what each line is,
   the amount of lines makes it less informative/overwhelming." -- <= 3 curves/panel, 2-4-word
   labels, explanations in captions.
3. **Errors without context** (user): "are we looking at the error? might help if we also show the
   trajectory that is being used to get an understanding." -- absolute-signal orientation before
   error plots; every caption names its trajectory.
4. **Hypothesis-first** (supervisors + user): "what do I expect ... what do I actually see ... what
   can explain this"; "redesign every plot with a clear hypothesis and make it interpretable for
   humans." -- one hypothesis per figure, as a question in the title; caption = expect / observe /
   falsified-if.
5. **f3a contents** (user): "we should show train-nf rms, val-nf rms and validation sim-rms" --
   the train/val nf-RMS pair reads GENERALIZATION (kills the overfitting alternative on-figure);
   sim-RMS reads deployment.
6. **f3b clarity** (user): "it's not clear at all what you're trying to show" -- collapse to the
   ratio bar chart with a break-even line at 1.0.
7. **De-overlay techniques** (agreed): plot the DIFFERENCE when the message is a gap; SMALL
   MULTIPLES (one condition per mini-panel, shared scale, panel titles replace legends); INSET
   ZOOM when lines differ by orders of magnitude (f2's bounded blue/grey are invisible at the red
   scale -- the evidence must be visible, not asserted); ENDPOINT ANNOTATION instead of legends.
8. **Human units**: no "normalized state increment" on figures -- convert the ANN output/DC to a
   physical equivalent (mN / m/s^2) so it can be weighed against the 45 N excitation.
9. **No jargon on figures**: "constant force" not DC, "no stiffness" not K=0, "training window"
   not nf, "error caused by the encoder start" not enc_IC. Internal names live in the docs only.
10. **One name per concept** across the whole deck: "no ANN" (never also "ANN off"/"baseline"/
    "epoch 0"); "encoder start" / "true start".
11. **Opening summary figure**: one figure that states the whole problem in 30 s before the
    evidence deck.
12. **Sign/magnitude hygiene**: plot |.| where sign carries no message; where it does (9/9 same
    sign) say it explicitly on the figure.
13. **Grayscale/projector robustness**: differentiate by linestyle + width, not color alone.
14. **Prints match plots**: console tables use the same plain-language names as the figures.

Global color/style semantics (all figures): red solid = suspect cause PRESENT; blue dashed = cause
REMOVED; grey = "no ANN" reference; black dashed = INDEPENDENT PREDICTION; dotted grey band =
absorber scale sigma(delta_a). Every figure keeps the provenance footer + evaluation-set label.
Old multi-line versions move to `figures/backup/` (they remain the full-decomposition evidence).

### 14.2 Meeting subset
Open with G0; walk G1 -> G2 -> G4 -> G5; hold G3/G6/G7/G8 as probed backups; G9 when run.

### 14.3 The figures, explicitly

**G0 -- opening summary (NEW).** Title: "What goes wrong when we augment the gantry model?"
- Two panels side by side, SAME y-scale (Y position error [m], 12 s):
  (L) "without the ANN": grey curve, bounded, settling; endpoint annotation "bounded, but 60x
      the absorber signal (encoder start)".
  (R) "with the trained ANN": red curve ramping to -2.9 cm; endpoint annotation "unbounded:
      constant learned force x no stiffness".
- Dotted absorber band on both; no legend (annotations only).
- Caption: the two problems in two sentences + "evidence in the following figures".

**G1 -- context/trajectory (was Fig 0).** Title: "What does the system do, and does the model
track it?"
- (a) measured trajectory, ABSOLUTE: Y at -0.30 m and X1 over 12 s (T1); inset zoom showing the
  ~2e-5 m ripple. One line per channel, endpoint-annotated.
- (b) measured vs model-from-true-start OVERLAID at trajectory scale (2 lines, visually
  indistinguishable); caption: "max deviation 0.13% -- the model matches the I/O; everything
  below lives 3-4 orders under the signal".

**G2 -- the encoder start (redesign of F1).** Title: "Does the free-run error come from the
encoder's estimated initial state?"
- Three stacked panels (logical X, Theta, Y), each TWO lines: red "encoder start", blue dashed
  "true start". Below them ONE difference panel: "error caused by the encoder start" (red minus
  blue, one line) with the black-dashed independent prediction tau*dv on X and the dotted
  absorber band.
- Endpoint annotations (values in mm); no legend box (line-end labels).
- Caption: expect absorber-scale error; observe 1.45 mm on X = predicted tau*dv (measured
  velocity error x known time constant); theta dies (has a spring), X/Y park (no stiffness);
  falsified if the difference had not matched the prediction. Trajectory: T1. Full 4-line
  decomposition + oracle floor: backup (values quoted: oracle floor 1.8e-6 X / 5.5e-6 Y).

**G3 -- constant-force universality (slim F2-companion).** Title: "Is the constant force a
one-run accident?"
- ONE panel: bar chart, dY-row mean force per checkpoint (9 bars: 4 cold nf, 4 warm rungs,
  drift_last), PHYSICAL units (equivalent mN), rms whiskers; bars grouped cold|warm, nf on the
  tick labels.
- On-figure annotation: "9/9 push the SAME direction; larger training window -> smaller force,
  never zero."
- Caption: expected random signs if pure wandering -> observed systematic sign (OE-2, §13).
  Other rows (10-100x smaller, mixed sign): backup.

**G4 -- the split (redesign of f3a).** Title: "Does improving the training objective improve the
deployed model?"
- Two columns (WARM 70903 nf=400 | COLD 71013 nf=800), two stacked panels each, shared x = epoch:
  (top) train nf-RMS + val nf-RMS, two lines, SAME units/window -> reads generalization
        ("both healthy, no overfit");
  (bottom) val sim-RMS, log y, + grey dashed "no ANN" level -> reads deployment (worsens 24x).
- <= 2 lines/panel; short labels ("train window fit", "val window fit", "free-run error").
- Caption: expect both to fall together if the objective is a faithful proxy; observe the split;
  kills lr/epochs/overfitting as explanations. nf-RMS never compared across columns.

**G5 -- "increase the window" (redesign of f3b).** Title: "Does a longer training window make the
augmented model better than NO model?"
- ONE panel: ratio bar chart. x = training window (400..3200 samples, with seconds in the tick
  labels); y = end-of-training free-run error / "no ANN" error, log scale. One bar per (nf,
  family); warm bars orange "continued training", cold bars blue "fresh start". Horizontal line
  at 1.0 labeled "break-even: as good as no ANN". Each bar annotated with its ratio (23.9x ...
  1.3x).
- Caption: all bars above 1.0, shrinking, SATURATING at 1.3x (2400 vs 3200 identical); 8-epoch
  budgets -> bars are lower bounds, the claim is only "never below 1.0"; measured cost 47 s/batch
  at nf=3200. Start/best trajectories: backup.

**G6 -- the counterfactual (redesign of F2 headline).** Title: "Is the learned constant force the
drift?"
- (a) SMALL MULTIPLES, three mini-panels, same y-scale (+-3e-2): "trained ANN" (red) | "constant
  force removed" (blue) | "no ANN" (grey), one line each, panel titles as the only labels;
  endpoint annotations (-2.9 cm | 0.02 cm | 0.02 cm). INSET in the middle/right panels zoomed to
  +-4e-4 showing the bounded oscillation (the evidence, visible).
- (b) velocity of the red run, one line, + black-dashed INDEPENDENT PREDICTION "terminal velocity
  F/c = -2.7 mm/s" -> bounded velocity = not energy (the Jan panel).
- (c) the dY force vs time, PHYSICAL units (mN), one line + its mean (black dash); annotation
  "never crosses zero: |mean|/rms = 0.997".
- Caption: removing ONE constant per row collapses the drift 133x onto the no-ANN level;
  falsified if the drift had survived the removal. Trajectory: V1.

**G7 -- horizon (slim F5).** Title: "At which horizon can training SEE the drift?"
- ONE curve: the drift contribution vs evaluation horizon (log-log), + dotted floor
  (sigma(delta_a)), + three vertical marks: training window 0.1 s, T* = 0.24 s, BPTT wall 0.5 s.
  Annotations at the marks ("0.11x floor: blind", "crosses the floor", "memory wall").
- Caption: two regimes -- genuinely blind below T*; visible but PREFERRED/tolerated above (d8,
  d12) -> no trainable horizon fixes it. Full/DC-removed cumulative curves: backup.

**G8 -- poles + excitation (backups, relabeled only).** f4_poles with the three marker families
and plain-language labels ("position modes: exactly ON the circle = free integrator; velocity
modes: inside, = exp(-ts/tau) to 8 decimals; yaw pair: inside (sprung)"); d17 excitation figure
with shortened labels.

**G9 -- the intervention (future, demo7).** Title: "If we enforce the zero-mean force Jan expects,
does the drift ever form?"
- (a) free-run Y error: red "trained, unconstrained" vs blue "trained with zero-mean pin" + grey
  "no ANN"; (b) window fit: both, overlapping (cost <= 2%).
- Caption: prevention (train-time) vs cure (post-hoc subtraction, G6): the interventional close of
  the causal chain. Sim-only demonstrator; the deliverable pin is the frequency-selective version
  (d16), one sentence on why.

