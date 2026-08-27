# An optimiser accumulating an offset its own objective penalises: research brief

**Written 2026-07-26 (late).** **This supersedes the framings in
`docs/narrowband-objective-problem-2026-07-26.md` §5 and
`docs/flat-direction-problem-2026-07-26.md` §2-3.** Both were searched
(`narrowband-literature-sweep-2026-07-26.md`, `flat-direction-literature-sweep-2026-07-26.md`)
and both rested on claims that have since been measured false. The candidates those sweeps
returned failed because the question was wrong.

**Why this brief should work where those did not:** the phenomenon is now defined mostly by
what it is NOT. Five mechanisms are measured false, which makes the surviving question
narrow and specific rather than descriptive.

---

## 0. THE MINIMAL REPRODUCTION (read this before section 1)

**The failure reproduces in a system with almost nothing in it**, and that is the most
useful fact for a literature search. `scripts/gantry/pysynth-data/k_sweep_minimal.py`:

* a single damped mass, `m = 1`, `c = 0.65` (`tau = 1.54 s`), **no spring** (`K = 0`);
* an unmodelled residual: one 150 Hz sinusoid, amplitude `2e-2`;
* a 600-parameter MLP writing force to the velocity row, zero-output init;
* training: windowed simulation error, `nf = 400` (0.1 s), Adam `lr = 1e-4`, 1500 steps;
* **windows initialised from the TRUE state** -- no encoder anywhere.

Result at `K = 0`: windowed loss **0.943x** ANN-off (the network helps on the objective it
minimises) while the 4 s free run is **1.713x** ANN-off (it is worse than emitting nothing).
That is the failure, qualitatively, with none of this project's machinery present.

**Measured NOT NECESSARY for the failure:**
the SUBNET encoder (the testbed uses true-state init); LPV scheduling; MIMO / the
P-transform; the coupled 8-state MSD absorber; and the two augmented latent states -- the
testbed has none, and on the real model they DECAY from `-3.80e-01` / `+3.54e-01` to
`-9.2e-05` / `+1.0e-04` over 12 s (an encoder-init transient, not an integrator).

**Measured NECESSARY:** the integrator. Sweeping the spring, the free-run harm ratio goes
`1.713` (`K=0`) -> `1.059` (`wn = 1 Hz`) -> `0.998` (`wn = 10 Hz`), i.e. it vanishes. But the
DC does NOT: at `wn = 1 Hz` the network's output is `|mean|/rms = 0.998`, essentially a pure
constant, while doing almost no harm. **So marginality is an AMPLIFIER of the offset, not
its cause.** The offset is acquired either way.

**The question, in its smallest form:** *why does a static network, trained to minimise
short-window simulation error inside a model of an integrating plant, converge to emitting a
constant offset that makes a long free run worse than emitting nothing -- when that offset
also costs it on the very objective being minimised?*

Caveat: the minimal testbed reproduces the failure's SHAPE but at `1.7x` against the real
pipeline's `58.9x`, so the two are not magnitude-comparable and no absolute number should be
carried between them. It is also 1 seed. Its value is as a fast screen (minutes per arm
versus hours on the real pipeline) and as the smallest system that exhibits the phenomenon.

---

## 1. The phenomenon, stated in field-neutral terms

A learned component (600-parameter MLP) is trained **inside** a physics-based state-space
model by minimising simulation error over short windows (0.1 s, 400 steps), each
re-initialised from measured data by a learned encoder. Deployment is a 12 s free run. The
plant has poles at `z = 1` (double integrators on two axes).

The learned component acquires a **constant output offset** on the velocity rows
(`17 mN` and `23 mN` equivalent force). That offset:

* reproduces **112.8%** of the entire deployment failure when it replaces the network
  (eight constants substituted for the whole trained net);
* **grows with optimiser steps**: 130 batches -> 9.5x degradation, 5200 batches -> 127x;
* appears in **10 of 10** independent checkpoints, across `nf` from 400 to 3200;
* is **penalised by the training objective** it is produced under.

That last point is the anomaly. **The optimiser persistently and progressively accumulates
a parameter-space offset that its own objective assigns a positive cost to.**

---

## 2. What is measured FALSE (this is the useful part)

| candidate explanation | refuted by | number |
|---|---|---|
| the objective is blind / flat in that direction | curvature probe, **two independent rigs** | `d2L/db2 = 7.06e+04` (production path) and `7.08e+04` (null rig), agreeing to 0.3% |
| the objective rewards the offset | windowed loss per arm | removing the offset makes the TRAINING loss **3% better**; the offset alone is worse than no learned component at all |
| the offset is cheap because paired with a compensating term | same measurement | the pair is NOT the cheapest arm |
| exposure bias / off-distribution extrapolation | offset measured on encoder-anchored training windows vs the free run | ratio **1.00 / 0.98 / 1.00** -- identical where trained and where deployed |
| the offset is entangled with the useful fit | gradient descent on the output mean in weight space | mean falls **0.892x** while the windowed loss also falls **0.936x** -- weight space CAN express "same shape, less offset" |
| non-convergence (a transient that more steps would remove) | the dose-response above | it **grows** with steps, monotonically, over 5200 steps and 10 runs |

Also refuted earlier, on measurement: longer training windows (`nf` 800-3200, offset present
at every horizon); unbiased truncated BPTT / better gradient estimators; preconditioning;
`lr` tuning and Adam-to-SGD swap as deliverables; and a multiple-shooting continuity term
(built, verified 5/5 as the method, then measured to discriminate a 65x-worse model at
**1.01x** -- blind).

---

## 3. The question

**Primary.** What is known about a stochastic optimiser — specifically Adam — persistently
accumulating a parameter offset along a direction where its own objective has **positive
curvature and a near-zero optimum**, with the offset **growing monotonically in step count**
rather than decaying, when the direction is neither flat, nor rewarded, nor entangled with
the fitted signal, nor an artefact of distribution shift?

**Secondary.** The plant has poles at `z = 1`, so the offset integrates: it is negligible
over the training horizon and dominant over the deployment horizon. Does the marginal
spectrum play any role in the accumulation itself, or is it purely an amplifier of an
optimiser-side effect that would be harmless on a stable plant?

**Third.** Given the offset is penalised by the objective and reachable in weight space, is
there a published construction that makes an optimiser actually take a descent direction it
demonstrably has available?

### Named starting points not yet searched under this framing

* **Adam's non-convergence results.** Reddi, Kale, Kumar, "On the Convergence of Adam and
  Beyond" (ICLR 2018) construct convex problems where Adam converges to the **worst** point,
  driven by the exponential-moving-average second moment forgetting rare large gradients.
  AMSGrad is their fix. This is the closest published match to "the optimiser moves against
  its own objective" and neither prior sweep raised it.
* **`beta2` timescale versus run length.** `beta2 = 0.999` gives the second moment a
  ~1000-step memory; production runs 5200 steps, so the estimator's transient covers 20% of
  training. `drift-conclusions` C5 already noted bias correction applying a 12.4x inflation
  at step 84. Whether the accumulated offset tracks the `v_hat` transient is measurable and
  unmeasured.
* **Implicit bias / implicit regularisation of adaptive methods**, specifically any result
  on a *persistent* rather than asymptotic offset.
* **`epsilon` in the denominator as a systematic bias source** on directions whose gradient
  is comparable to `sqrt(v_hat)`.

### Anti-scope (measured, do not re-propose)

> **CORRECTION 2026-07-26 night. ARTBP was in this list and does not belong here.** It was
> excluded on a theoretical argument (Beatson and Adams Thm 4.1, unbounded variance at
> `|lambda| = 1`) without anyone opening the artifacts. **Six** converged ARTBP runs on the
> production with-MSD data, written 2026-07-23, **pre-date every document that rules it out**
> (`drift-critical-analysis` 07-24, `drift-conclusions` §4 07-25, `flat-direction-problem` §4
> and this brief 07-26). Recomputed from the `.npz` in `dc-accumulation/results/step3_artbp_recompute.json`:
> at `H_max = 1600` ARTBP cuts the first-epoch collapse from `2.9047e-02` to `2.7523e-03` m
> (12 s val free run), the best trained-epoch 12 s error from `2.1920e-02` to `1.7588e-03` m,
> and the free-run drift ratio (0.5 s tail of a 2 s run) from `83.1x` to `13.5-22.1x`, at a
> **2 percent** cost in windowed 0.1 s fit. The variance blow-up the theory predicts is real
> and **is** measured, but only at `H_max >= 3200` (`dcgrad_var` `2.2e+00` at 1600 against
> `1.5e+03` at 3200 and `6.6e+07` at 6400).
>
> **The correct statement is: ARTBP is a measured 4-6x mitigation with a hard `H_max = 1600`
> ceiling, not a fix and not ruled out.** It still fails G1 (do not degrade the init) on every
> arm. It is the **benchmark any new candidate must beat**, so a literature sweep should look
> for work that *extends or replaces* it, not avoid it. Two sweeps were already steered away
> from it; that is the error this note exists to stop repeating.
>
> Grade SINGLE (one seed per arm), and see the run-table row for four structural defects in
> how these runs were reported (no epoch-0 row on three arms; a borrowed and non-invariant
> epoch-0 reference; `best_val_sim` and `drift_ratio` describing different models).

Longer windows; hard class restrictions (passivity,
contraction, RENs, bounded impulse, spectral caps -- these violate the project's
full-expressivity requirement AND are separately measured to fail); zero-mean or window-mean
priors (the real residual's mean is `-157.5 N` / `-83.7 N` at 315-344 sigma); optimiser
swap or `lr` tuning as a deliverable (drift is proportional to `lr`; SGD learns `+0%` on a
real residual); adjoint / dual-weighted-residual re-weighting of a long-horizon **position**
functional (weight ratio DC to 150 Hz measured at `2.27e12`, so it would suppress the signal
to be learned by twelve orders of magnitude -- a different quantity of interest may survive).

---

## 4. Evidence grade, stated up front

**Everything above is 1 seed and mostly 1 record**, below this project's 3-seed floor, with
two exceptions: the curvature figure is confirmed on two independent rigs, and the
10-checkpoint count spans several runs and horizons.

Two of the refutations rest on **small samples**: the "objective penalises the offset" result
uses 40 windows of one training record, and the "not entangled" result uses 12 windows of one
validation record, both against a 6664-window training bank. They agree with each other and
with the curvature measurement, but neither is a full-distribution statement.

A continuation run (`b0_continue_training.py`) testing the non-convergence branch directly
was running when this was written. Its pre-declared readings are in the script.

---

## 5. Provenance

Numbers: `docs/results-log-2026-07-26.md`, and `docs/gantry-augmentation-problem-log.md` §12
rows MS1-MS12 and A1-A9. Scripts and unit JSONs: `scripts/gantry/pysynth-data/` and its
`results/`. Prior sweeps, superseded in framing but not in bibliography:
`docs/narrowband-literature-sweep-2026-07-26.md`,
`docs/flat-direction-literature-sweep-2026-07-26.md`,
`docs/multiple-shooting-sweep-2026-07-25.md`, `docs/rollout-stability-literature.md`.
