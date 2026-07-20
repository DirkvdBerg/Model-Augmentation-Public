# DC / Drift Diagnosis: Results and Conclusions (2026-07-17)

Scope: this session ran experiments to answer WHERE the non-zero-mean ("DC") correction the
augmentation ANN learns comes from, and why it drifts. It closes gate G-A on the physics side (v1f),
confirms gate G-C is systematic (v3/v3b), characterises the in-window error accumulation (v4), and
tests the encoder-init hypothesis by intervention (v3x0). Results are written out explicitly below;
each block ends with the conclusion drawn from it.

> **CORRECTION (2026-07-18):** an earlier version of this doc concluded the DC is caused by the
> ENCODER-INITIALISATION error (inferred from v4's shape-match). **Experiment 4 (v3x0) refuted that:**
> training from the TRUE initial state still produces the DC (~85% of the encoder-init magnitude), so
> encoder-init is NOT the cause. v4's within-window ramp is a valid observation but is decoupled from
> the DC. A second intervention (v3joint, broadband 1-200 Hz excitation) then ALSO refuted the
> identifiability candidate (DC unchanged). So BOTH leading causes are eliminated: the DC is intrinsic
> to the training dynamics (see Integrated conclusion). Sections below are corrected in place.

Gates (README section 7): G-A = does the physics demand a DC; G-B = estimator zero-mean assumptions;
G-C = training dynamics born/accumulated the DC.

State layout (Python augmentation model, model.py:31): `[X, Theta, Y, dX, dTheta, dY, delta_a,
vdelta_a]` = idx 0..7. K=0 (spring-less) axes: X (pos 0), Y (pos 2), dX (vel 3), dY (vel 5). Theta
is sprung. Absorber floor: `std(delta_a) = 2.2e-5 m`.

---

## Experiment 1: v1f, open-loop DC + AC excitation (does the physics carry a DC?)

Script `v1f_dc_excitation_openloop.m`. Sustained offset + a sustained 150 Hz tone (the MSD resonance
`fa`), open loop, SAME input to both plants: truth = `gantrySystemExtended` (8-state, hidden MSD)
vs baseline = `gantrySystem` (6-state). One axis driven at a time, from four initial Y
(-0.30/0/+0.10/+0.30 m). DC read by harmonic least squares (constant + first two harmonics of `fa`)
on all three logical channels, plus `delta_a` tail RMS. Two DC mechanisms tested: (A) static-gain
DC (offset), (B) second-order rectification DC (tone excites `delta_a`; the truth-only
`ma*(Y+L0+delta_a)^2` term rectifies into a mean).

### Results (explicit)
- **Mechanism A (static-gain DC) = 0.** `baseDC` and `truthDC` on the driven axis identical to
  ~1e-7 relative on every axis and Y0. (Both sit below the analytic 3e-2 m/s / 1e-3 rad only from
  incomplete K=0 settling at 5 s, identical for both plants: X `2.715e-2` vs analytic `3.000e-2`;
  Y `2.914e-2` vs `3.000e-2`; Theta `1.000e-3` = analytic.)
- **delta_a excitation proof (tail RMS):** Y-drive `1.671e-5`; X-drive `~3e-11`; Theta-drive
  `~2e-9`. Only the Y drive shakes the MSD.
- **Mechanism B (rectification DC), read on Theta (its home channel) for the Y drive** (the only run
  that excites `delta_a`): `B[Theta]` = `+3.22e-10 / +3.51e-10 / +3.23e-10 / +2.41e-10 rad` across
  Y0. Average `~3.1e-10`.
- **Quantitative confirmation of the mechanism:** `<delta_a^2> = (1.671e-5)^2 = 2.79e-10`, vs
  measured `B[Theta] ~ 3.1e-10`, matching to ~10%. Confirms amplitude-squared (second-order Volterra)
  scaling.
- **Largest DC anywhere in the table = the static mass-matrix asymmetry, also negligible:**
  Theta-drive `B[dX] = -1.1e-7 m/s`; X-drive `B[Theta] = 4.7e-8 rad`. These appear on drives that do
  NOT excite `delta_a` and are steady across Y0, so they are the L0 / (`mh -> mh_rigid + ma`)
  mass-split asymmetry (present at `delta_a ~ 0`), not resonance rectification. Bounded `~1e-7`
  (relative `~1e-5`).

### Conclusion
The physics carries no meaningful DC the baseline lacks. Across every drive x every channel,
including the case where the MSD is genuinely excited and read on its home channel, the largest
`truth - baseline` DC is `~1e-7` (relative `~1e-5`), 5+ orders below the ANN's learned DC. The
`delta_a^2` rectification is real, amplitude-squared, and negligible (`3e-10`). **G-A refuted on
DC-exciting data: the DC the ANN learns is not justified by the system; it originates in the
estimator/training (G-B/G-C).**

---

## Experiment 2: v3 / v3b, per-update-step DC-birth monitor (is the DC systematic or diffusion?)

Script `v3_dc_birth_monitor.py`. Instrument a short training run: patch `torch.optim.Adam.step` to
log, per optimizer step, (A) the ANN's per-row output mean/std on the fixed probe points `Z_pts`,
and (B) `dLoss/d(bias)` = the loss gradient along a constant per-row correction (via a zero bias
added in `ann.forward`; the interconnect calls `block.forward` directly, so a forward HOOK is
bypassed, requiring the forward patch). lr = 1e-7, 1 epoch (260 update steps), 3 UNFIXED seeds, full
X+Theta+Y routing, nf = 400. Two runs: v3 (hook, B came back NaN, the bug) then v3b (forward-patch,
B works; A identical, so v3b is canonical).

### Results (explicit)

A: ANN output per-row MEAN at end of epoch (the DC), 3 seeds:

| row | seed 0 | seed 1 | seed 2 | onset |
|---|---|---|---|---|
| **dY** | **-4.207e-6** | **-3.623e-6** | **-3.548e-6** | ~step 13 |
| Y | +4.635e-7 | +3.715e-7 | +4.087e-7 | step 1 |
| dX | +1.123e-7 | +3.944e-8 | +5.139e-8 | step 1 |
| X | -3.515e-8 | -1.801e-10 | -8.276e-9 | step 1 |

- Loss stayed flat (no overshoot): step 1 `1.738e-6` -> last `1.709e-6 / 1.711e-6 / 1.706e-6`.

B: mean `dLoss/d(bias)` over the 260 steps (the force on the DC direction), 3 seeds:

| row | seed 0 | seed 1 | seed 2 | pushes DC | matches observed DC? |
|---|---|---|---|---|---|
| dY | +2.154e-5 | +2.465e-5 | +2.698e-5 | negative | YES (dY DC is negative) |
| Y | -2.059e-4 | -1.476e-4 | -1.820e-4 | positive | YES (Y DC is positive) |
| dX | +3.814e-5 | +7.494e-6 | +3.933e-5 | negative | NO (dX DC positive, but tiny) |

- Per-seed the gradient is minibatch-noisy (|t| < 1); the systematic signal is the reproduction
  across seeds.

### Conclusion
The DC is SYSTEMATIC, not diffusion. It is born in the first ~13 update steps and reproduces in sign
across three independent inits (A), and the loss gradient along the DC direction reproduces in sign
and magnitude across seeds and points the right way on the dominant rows (B). Same-sign across
independent seeds is the systematic signature; diffusion would scatter. The per-seed t is small
because the loss is nearly flat in this direction (minibatch-noisy), so cross-seed agreement is the
correct instrument (variance-based per-seed tests degenerate at near-deterministic full-batch;
GSNR/gradient-noise-scale). **The windowed loss is nearly neutral to the DC but carries a small
consistent bias that the K=0 integrators accumulate: a systematic loss-geometry cause (G-C), not
random wander.**

---

## Experiment 3: v4, in-window per-step error accumulation (where/how does it accumulate?)

Script `v4_inwindow_accumulation.py`. Roll the FIXED drifted checkpoint `gantry_drift_71167_last`
(D-114; NOT `_best` = epoch-0 zero-ANN, guarded) forward within a window from encoder-init, per-step
predicted vs true output. Three passes per window via the `freerun` shadow (copied from
`generate_data.py`): FULL, DC-MUTED (subtract the model's measured mean output), ANN-OFF (zero it).
Non-overlapping 800-step windows (60 total, across 3 standstill records at Y = -0.30/0/+0.30),
error transformed to logical coords (X, Y = K=0; Theta = sprung), averaged over windows. Axis =
rollout step index within one window (different from v3, which was training steps).

### Results (explicit)
- **ANN per-row |mean|/rms on the checkpoint:** X=0.69, Theta=0.79, Y=0.93, dX=0.59, dTheta=0.45,
  dY=0.91, delta_a=0.96, vdelta_a=0.95. DC on the K=0 velocity rows: `dX = +7.957e-7`,
  `dY = -8.816e-6` (normalized ANN output units).
- **Growth-law fit of the full pass** (e ~ a*k linear / a*k^2 quad / log-linear exp):

| channel | type | best fit | R2 lin | R2 quad | R2 exp | slope a_lin |
|---|---|---|---|---|---|---|
| X | K=0 | **linear** | 0.995 | 0.820 | 0.339 | 7.055e-8 |
| Y | K=0 | **linear** | 0.994 | 0.815 | 0.311 | 1.824e-7 |
| Theta | sprung | (none) | -1.383 | -2.563 | 0.062 | 3.611e-8 |

  X and Y are linear ramps (constant velocity offset); Theta has no growth law (all R2 <= 0.06;
  negative for linear/quad means a growth model fits worse than a constant): it oscillates and parks.
- **Envelope ratio (late/early RMS), full / muted / off:** X `7.59 / 7.34 / 7.28`;
  Theta `1.01 / 0.97 / 1.00`; Y `6.95 / 6.03 / 5.82`. K=0 axes grow ~7x; Theta stays ~1.
- **Init-error dominance within the window:** the three passes (off / muted / full) nearly overlap
  on X and Y, and the ramp is fully present with the ANN OFF. On log-y the encoder handoff is at
  ~step 17 (error jumps ~1e-11 -> ~1e-6), after which the init error propagates as the ramp. The
  init-error ramp reaches `~1.5e-4` on Y, vs the absorber signal `2.2e-5` (roughly 7x larger). The
  ANN's own contribution (full-minus-off) ramps to `~4.5e-5` on Y, a small, same-shaped linear ramp;
  the net error magnitude barely moves between passes.

### Conclusion
1. The K=0 error grows LINEARLY (constant velocity offset), not quadratically (force) or
   exponentially (instability). This answers the "explodeert het na de eerste 100 stappen" question:
   NO, it is a steady linear ramp, a constant sitting beside the truth.
2. No fading memory is K=0-specific: X, Y ramp (~7x), Theta parks (~1x). "Geen fading memory want
   geen damper," isolated to the spring-less axes.
3. Within one window the K=0 error is DOMINATED by the encoder-INITIALISATION error (present with
   the ANN off), roughly an order of magnitude above the absorber signal the ANN is meant to learn.
   The ANN's DC adds only a small, same-shaped ramp on top.
4. It EXPLAINS v3's small gradient: the windowed loss is dominated by the (DC-independent)
   encoder-init ramp, so it is nearly flat in the DC direction. (A tempting further inference, that
   the DC is the loss-optimal compensation of the same-shaped init ramp, was REFUTED by Experiment 4.)

Correction (2026-07-18): the causal step "the ANN parks a DC BECAUSE of the init ramp" was flagged
here as a strong-but-unproven inference. Experiment 4 (v3x0) intervened and FALSIFIED it: training
from the true initial state still produces the DC. So the encoder-init error, while it does dominate
the within-window baseline ramp (the observation above stands), is NOT what the DC compensates; the
ramp and the DC are decoupled.

---

## Experiment 4: v3x0, true-init probe (does the encoder-init error cause the DC?)

Script `v3x0_true_init_probe.py`. Custom training loop that bypasses deepSI's encoder-based init: each
window starts from x0 = [TRUE physical 6 `(x_logical[p]-x_mean)/std_x`; aug 2 = 0], free-runs nf=400
steps (`yhat,x = fit_sys.hfn(x, u_norm)`, mean per-step MSE), trains the ANN. Else identical to v3
(lr=1e-7, augmentation band). A control mode (`INIT=encoder`) re-inits from the encoder and MUST
reproduce v3b before the true-init result is trusted. The 2 aug states are arbitrary latents
(`W^a` random-init, no fixed physical scale, hence `R2_linmap`), so a true aug init is undefined;
they are tiny and start ~equilibrium, hence 0. Interface validated: true-init step-0 MSE = 2.2e-18.

### Results (explicit), 1 seed, all 14 records, 1 epoch (~250 steps)
- **Control (INIT=encoder) VALIDATES the loop:** dY DC = -3.98e-6 at step 250, matching v3b
  (-3.5..-4.2e-6), same growth shape. The custom loop faithfully reproduces deepSI's encoder-init
  training.
- **True-init (encoder bypassed):** dY DC still grows NEGATIVE: -8.1e-7 (step 50) -> -1.80e-6 (100)
  -> -2.45e-6 (150) -> -2.87e-6 (200) -> **-3.36e-6 (250)**. That is ~85% of the encoder-init DC,
  same sign, same shape.

### Conclusion
**Encoder-init is NOT the cause.** Feeding the true initial state does not remove the DC (it still
reaches ~-3.4e-6, vs -4.0e-6 with the encoder). The DC forms almost identically regardless of the
init. This falsifies the v4-based encoder-init hypothesis (per `causal-claim-needs-intervention-not-
observation`): v4's within-window init ramp is real but the ANN's DC is NOT compensating it (else
true init would have removed it); the ramp and the DC are decoupled. The DC is INHERENT to the
training on this data. Remaining candidate cause: identifiability (the DC-free excitation) and/or the
free-direction training dynamics; the broadband [1,200] Hz run (Experiment 5) tested the
identifiability branch and ALSO refuted it, leaving the training dynamics themselves.

---

## Experiment 5: v3joint, broadband excitation (does the DC-free excitation cause the DC?)

Same as the v3b augmentation run but `mode='joint'` -> trains on the 1-200 Hz broadband data instead
of the 130-180 Hz narrowband (the narrowband has no low-frequency content, so the constant-output
direction is unconstrained; the broadband adds low-freq lines that should constrain the near-DC gain).

### Results (explicit), 3 seeds, 1 epoch, step 250
- **dY DC = -4.18e-6 / -3.72e-6 / -3.82e-6**, essentially IDENTICAL to v3b narrowband
  (-4.21/-3.62/-3.55e-6). Low-frequency excitation does NOT reduce the DC.

### Conclusion
**Identifiability (the DC-free excitation) is NOT the cause.** Adding low-frequency content leaves the
DC unchanged. Combined with v3x0 (encoder-init refuted), both leading candidate causes are eliminated
by intervention: the DC forms the same regardless of init and excitation band. See the Integrated
conclusion.

---

## Experiment 6: SGD vs Adam (the mechanism)

Same custom-loop probe as v3x0 (INIT=encoder, validated vs v3b), lr=1e-7 matched, ONLY the optimizer
changed Adam -> SGD. Tests whether the DC is Adam's implicit bias: Adam's per-step normalization
(~sign-descent) would amplify a tiny consistent gradient in the loss-flat DC direction into a steady
~lr walk; SGD (step ~lr*grad) takes a vanishing step in a near-flat direction.

### Results (explicit), seed 0, step 250
| optimizer | dY DC | loss |
|---|---|---|
| **Adam** (validated control) | -3.98e-6 | ~1.5e-6 |
| **SGD** | +1.98e-9 | ~2.0e-6 |

SGD builds ~2000x LESS DC than Adam, at the SAME loss (SGD trains to the same level, it is not stalled).

### Conclusion
**Adam is the amplifier: the DC is an OPTIMIZER artifact.** SGD reaches the same loss without the
drift-causing constant, so the DC is not needed for the fit; only Adam's normalization deposits it.
This CLOSES the mechanism: the loss is flat in the DC direction, and Adam (via ~sign-descent) walks
the ANN off the zero-mean center into that flat valley, systematically -- SGD does not. Fix options,
all consistent: a soft DC pin (any optimizer), a non-adaptive optimizer (SGD avoided it here at no
loss cost), or an Adam variant without flat-direction creep.

### Seeding caveat (2026-07-18)
The Adam-vs-SGD comparison is a valid CONTROLLED contrast (identical init, only the optimizer differs
-> 2000x). But the intended "3-seed SGD confirm" was NOT genuinely multi-seed: v3x0 set the global
np/torch seed, yet `demo_common.build_pipeline` reseeds from `cfg.seed` (demo_common.py:61,78) and
v3x0 passed the shared `CFG`, so every "seed" rebuilt the identical model -- the three SGD runs are
byte-identical (seed0==seed1: same step-0 loss 1.4305e-6, same DC path). FIXED: v3x0 now builds a
per-seed `dataclasses.replace(CFG, seed=seed)` and passes it to `build_pipeline`. Seed-robustness of
the DC ITSELF is independently established by the v3 monitor (which seeds correctly via
`RunConfig(seed=seed)` -- all 3 seeds show the DC with sign agreement). A genuine multi-seed v3x0
SGD-vs-Adam contrast is re-running under the fix.

---

## Conceptual clarifications settled this session (conclusions)

- **The DC is a constant in the ANN's OUTPUT, not a force in the system.** The augmented model is
  `x[k+1] = baseline_step(x[k], u[k]) + w[k]`; `w[k]` is the ANN's per-step state correction. The DC
  is the time-mean of `w[k]` on the K=0 velocity rows: a small constant added to the velocity
  correction every step. On a spring-less axis a sustained constant input produces a ramp, which is
  why it drifts. It behaves like a phantom constant force but is not a force term in the plant.
- **Three distinct zero-mean questions, do not conflate:** (1) the system signals are zero-mean;
  (2) the correction the ANN SHOULD learn (`w_perfect` = truth minus baseline, one step) is zero-mean
  (f04b: `|mean|/rms = 0.000`, largest row-mean `1.1e-8` on dY); (3) the correction the ANN ACTUALLY
  learns is NON-zero-mean (dY DC `~5.8e-7` on run 71167, `50x` above the required mean). The bug is
  precisely (3) != (2).
- **"Training setup" = the choices that leave the DC direction free / fail to pin it**, none of which
  is "the physics has a DC": a free-run windowed loss (not a direct output match); a short window
  (drift ~T^2, invisible at 0.1 s, f09); DC-free excitation (U0=0 multisine, f10, leaves the DC
  direction unidentified); with the K=0 integrator as the amplifier. (An encoder-estimated start was a
  candidate ingredient, but v3x0 ELIMINATED it: true init still produces the DC.)

---

## Integrated conclusion

- **v1f:** the physics is zero-mean (no DC the baseline lacks; largest `~1e-7`).
- **v3/v3b:** the DC is systematically born during training (consistent sign across seeds), in a
  direction the windowed loss barely prices (small consistent gradient).
- **v4:** within one window the K=0 error is a LINEAR ramp dominated by the encoder-init error
  (`~1.5e-4` on Y, ~7x the absorber), not an explosion; Theta parks. (Observation valid; the causal
  reading was refuted by v3x0.)
- **v3x0 (2026-07-18):** training from the TRUE initial state does NOT remove the DC (`dY` -3.36e-6
  vs the encoder control -3.98e-6) -> **encoder-init is NOT the cause**; the ramp and the DC are
  decoupled.
- **f05 (prior):** over the unreset 12 s deployment the persistent DC dominates the drift.

Corrected chain: the DC is systematic (v3) and forms the same REGARDLESS OF THE INITIAL STATE
(v3x0: true init still gives dY -3.36e-6) AND REGARDLESS OF THE EXCITATION BAND (v3joint broadband
1-200 Hz gives dY -4.18/-3.72/-3.82e-6, identical to narrowband). Both leading candidate causes are
therefore ELIMINATED by intervention: it is neither the encoder-init error nor the DC-free
excitation. The DC is intrinsic to the TRAINING DYNAMICS / loss geometry on this architecture and
data: a small, systematic, loss-nearly-flat gradient that parks a constant on the K=0 rows in the
first ~13 steps, amplified by the K=0 integrator. It is not physics and not a model deficiency. The
remaining SOURCE question (why the loss geometry carries that consistent gradient) is untested:
candidates are the estimator's zero-mean normalization/init assumptions (G-B / V2, unrun) and a
systematic baseline-discretization bias. **The fix does not depend on the source:** since neither
better init nor input design removes the DC, the robust fix is a direct SOFT PIN on the DC direction
(the zero-at-equilibrium direction). Multiple shooting was already tried and failed (Optuna 69399,
best at epoch 0).

---

## Experiment 7: truncation-length sweep (source test -- POSITIVE)

Tests whether the DC comes from truncated-BPTT bias: a longer rollout window captures more of the
non-decaying lambda->1 integrator sensitivity, so the biased gradient (and the DC it feeds) should
shrink. Same v3-monitor script, only nf changed (V3_NF knob), seed 0, lr=1e-7.

| window | tail-20 dY DC | tail-20 dX DC | loss |
|---|---|---|---|
| nf=400 (baseline) | -4.21e-6 | +1.17e-7 | 1.34e-6 |
| nf=800 (2x)       | -2.26e-6 | +4.27e-8 | 2.90e-6 |

dY DC fell 46% (ratio 0.536), dX 64%, on doubling the window. Caveat: the nf=800 loss is ~2x higher
(longer free-run windows are harder to minimize -- Beintema's multiple-shooting motivation), so this
is not a perfectly loss-matched comparison; but the DC drop is far larger than the loss change and the
direction is unambiguous.

**RECONCILE with the prior nf-sweep (SLURM 71013, §12 run log) -- this is corroboration, not new news.**
That independent sweep already ran nf={800,1600,2400,3200} and found the dY-DC present on ALL 9
checkpoints with magnitude **~1/nf**, and EVERY free-run worse than the epoch-0 baseline (8.0e-5) --
i.e. the drift was NOT fixed at any nf. Our nf=400->800 point (-4.21e-6 -> -2.26e-6, ~half on doubling)
sits exactly on that 1/nf law (DC ~ 1.7e-3 / nf across nf in {400..3200}).

**Conclusion (corrected).** The 1/nf scaling confirms the DC is truncation-window-DEPENDENT (consistent
with truncated-BPTT bias as a source), but it simultaneously PROVES longer fixed windows are NOT a fix:
1/nf is nonzero at every finite nf, and any residual DC integrates to unbounded drift (71013 saw exactly
this up to nf=3200). So this experiment is DIAGNOSTIC ONLY; window length is a REFUTED fix, not a
complementary one. The only untried window-related intervention is ARTBP (an UNBIASED gradient at fixed
nf -- different from a longer fixed window), lower priority than the direct DC-direction fix. Net: you
cannot out-window the drift; the robust fix is the DC-direction intervention (zero-mean constraint in
sim / soft DC penalty on real data). Command: `V3_NF=800 V3_SEEDS=0 V3_PREFIX=v3nf800 ... v3_dc_birth_monitor.py`.

---

## Experiment 8: pole-perturbation (source test -- INCONCLUSIVE, confounded)

Intended test: add a light stiffness to the K=0 X/Y axes (pole z=1 -> z=1-delta, lambda<1) and check
whether the DC vanishes (env-gated GANTRY_KX_ART/KY_ART in gantry_ss.py, default 0 = exact ground
truth). Ran kx=ky=1000 N/m, seed 0.

Result: the fit was WRECKED (val sim-RMS 1.6e-4 -> 0.106, 650x worse) and the DC did NOT shrink
(dY -4.9e-6, dX +8.5e-6 at step 250 -- comparable to / larger than baseline). The comparison is
therefore meaningless: it is a different, broken operating regime, not "same fit, poles moved".

Why the knob is invalid for this system (pre-run pole analysis, confirmed): the position-mode decay
rate is -c/(2m) ~ -0.323 rad/s, set by damping/mass and ~INDEPENDENT of stiffness (the modes are
heavily underdamped). So stiffness barely moves the within-window decay (~3% over 0.1 s; natural
timescale ~3 s >> window) but injects a large spurious restoring force K*q ~ 1000 * 0.3 m ~ 300 N that
the truth lacks -> the baseline free-run diverges. No stiffness value threads the needle: large k
wrecks the fit, small k does nothing measurable. **The stiffness pole-perturbation cannot test
lambda->1 for this system.** The truncation sweep (Exp 7) is the clean source test and settled it.

---

## Mechanism (2026-07-18): confirmed amplifier + literature-grounded source

A targeted web deep-research synthesis (`literature/stability-training/claude-research-optimizer-SGD-vs-ADAM.md`)
plus Experiment 6 give the full mechanism as a THREE-STAGE amplification, each stage independently
supported:

1. **Source (parameter-gradient bias) -- truncated-BPTT bias on the non-decaying integrator.** The
   truncated / free-run BPTT gradient is systematically biased (Tallec & Ollivier 2017); the bias
   decays geometrically only for a contractive mode (factor lambda<1, Aicher et al. 2019). For the
   K=0 pole at z=1, lambda->1, so the bias does NOT decay for any finite window -> a tiny,
   deterministic, seed-consistent gradient along the integrator (DC) direction. LITERATURE-GROUNDED
   HYPOTHESIS, not yet tested by us.
2. **Amplifier (parameter space) -- Adam ~ sign descent.** Adam moves ~lr*sign(grad), independent of
   gradient magnitude, so a tiny consistent gradient in a flat direction accumulates into the DC;
   SGD (step ~lr*grad) stalls. CONFIRMED by Experiment 6: SGD DC = 0.05% of the Adam DC at matched
   loss (the report's confirmation threshold is <1-5%). (Balles-Hennig 2018; Kunstner et al. ICLR
   2023; Cattaneo et al. "Implicit Bias of Adam" 2024.)
3. **Amplifier (state space) -- K=0 integration.** The constant integrates over the long free-run
   into unbounded drift. Established (f01/f05, v4).

Framework match: this is the SUBNET / deep-subspace-encoder augmentation setting (Beintema/Toth/
Schoukens); the Retzler industrial-robot augmentation (integrator positions, ANN at acceleration) is
the canonical structural case. No single paper reports the exact chain (truncated-BPTT + Adam -> DC on
an integrator) -> a genuine contribution.

Stage 1 (source) -- two decisive tests recommended by the synthesis, both run 2026-07-18:
- (i) **Truncation-length sweep** (Experiment 7): **DIAGNOSTIC POSITIVE, but NOT a fix.** Doubling the
  window nf=400->800 cut the dY DC 46% (-4.21e-6 -> -2.26e-6), dX 64%; this sits on the **~1/nf law**
  already found by the prior nf-sweep (SLURM 71013, nf={800..3200}: DC present on all, ~1/nf, drift NOT
  fixed at any nf). The 1/nf scaling confirms truncated-BPTT bias as a source, but PROVES longer fixed
  windows cannot fix it (nonzero DC at every finite nf -> unbounded drift). Window length is a REFUTED
  fix; only ARTBP (unbiased gradient) is untried. **Source confirmed; not out-windowable.**
- (ii) **Pole-perturbation** (Experiment 8): **INCONCLUSIVE (confounded).** kx=ky=1000 wrecked the fit
  (val sim-RMS 1.6e-4 -> 0.106, 650x) and did not shrink the DC. For this system the pole decay is
  damping-limited (-c/2m ~ -0.323, ~independent of stiffness), so stiffness cannot produce meaningful
  within-window decay -- it only injects a fit-wrecking spurious force K*q ~ 300 N. The stiffness knob
  cannot test lambda->1 here; the truncation sweep is the clean test and it settled the question.

## Provenance

Scripts (all in `scripts/gantry/gantry-zero-mean/`):
- `v1f_dc_excitation_openloop.m` -> `data/v1f_results.mat`, `data/v1f_console.txt`,
  `figures/v1f_<axis>_dcac_response.png`.
- `v3_dc_birth_monitor.py` -> `data/v3b_perstep_seed{0,1,2}.npz` (canonical; v3 identical A, B=NaN),
  `figures/v3b_perstep_seed*.png`, `figures/v3b_multiseed_dc.png`. D-090 row in
  `docs/gantry-augmentation-problem-log.md` section 12.
- `v4_inwindow_accumulation.py` -> `data/v4_results.npz`, `figures/v4_perstep_error.png`,
  `v4_growth_law.png`, `v4_reference_subtraction.png`. Rolls `gantry_drift_71167_last` (D-114).
- `v3x0_true_init_probe.py` (encoder-init test) -> `data/v3x0{true,ctrl}_*seed*.npz`; control (INIT=
  encoder) reproduces v3b (-3.98e-6), true-init (encoder bypassed) still gives -3.36e-6. D-090 row.
- `v3_dc_birth_monitor.py` `V3_MODE=joint` (broadband test) -> `data/v3joint_perstep_seed*.npz`;
  dY DC -4.18/-3.72/-3.82e-6, identical to narrowband. D-090 row.

Literature (in `literature/stability-training/`):
- `claude-deep-research-drift-diagnostics.md` (identifiability / mechanism-demo catalog),
- `claude-deep-research-inwindow-accumulation.md` (BPTT accumulation, DC-birth, update-step
  diagnosis),
- `claude-deep-research-perstep-rollout-diagnostics.md` (within-rollout per-step growth,
  ramp-vs-explosion, init-vs-model, plot recipes).

Lessons added this session (`tasks/lessons.md`): `verify-nonlinear-mechanism-fully`,
`instrument-deepSI-at-the-called-method`, plus the `test-zero-mean-properly` clause (5) on mirror
blindness.
