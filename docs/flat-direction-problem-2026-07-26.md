# Drift along an objective-flat direction: problem statement for literature research

**Written 2026-07-26, after MS11 and MS12.** This supersedes the framing in
`docs/narrowband-objective-problem-2026-07-26.md` §5, which asked the wrong question.
That document's measurements (§1-4) and its list of unexcluded alternatives (§6) still
stand and should be read alongside this.

**What changed.** The earlier framing said "the loss is blind to the DC component".
Blindness explains why a constant offset is TOLERATED. It does not explain why training
RELIABLY PRODUCES one, growing with optimiser step count. Indifference does not create a
bias; something has to move the iterate there. The mechanism below closes that gap and is
the reason this document exists.

---

## 1. The problem in one paragraph

A learned block is trained inside a physics model by minimising simulation error over
short (0.1 s) windows. The deployment metric is a 12 s free run. On a plant with poles at
`z = 1` (double integrators), a **constant** output offset from the learned block is
essentially invisible to the windowed objective and catastrophic in the free run: measured,
`17 mN` on one axis and `23 mN` on another produce `0.219 m` of position error in 12 s and
a 127x collapse of the deployment metric in the FIRST epoch, while every windowed metric
IMPROVES monotonically. The direction carrying that offset is measured to be **flat**
(statistically neutral) in the training objective, and the optimiser is **Adam**, which
gives a flat direction a full `~lr`-sized step regardless of how small its gradient is.
So the learned block accumulates an offset along a direction the objective cannot resist
and the deliverable cannot tolerate.

---

## 2. The mechanism, with the measurement behind each link

| # | Link | Measurement |
|---|---|---|
| 1 | The failure is an accumulated **force/tendency error**, not a diverging mode | MS11. Twin-perturbation (which cancels the tendency term exactly) grows **polynomially and sub-linearly**, exponent `+0.42`, R^2_poly `0.96` vs R^2_exp `0.69`, and is alpha-invariant. Clean-start error is a power law of exponent **`1.484` at R^2 `0.997`** -- the signature of a constant force on a lightly damped axis |
| 2 | That force IS the learned block's **mean output**, and it accounts for the whole failure | MS12. Replacing the entire trained network with **eight constants** gives **112.8%** of the full 12 s error (`8.708e-03` vs `7.719e-03` m, floor `1.311e-04`). Removing the mean cuts the error 2.7x |
| 3 | The offending direction is **flat in the training objective** | `d12`: on training windows the DC direction is **NEUTRAL** (pooled `Delta/SE = +0.71`, n=120) -- "a data-silent direction on the training distribution". (`d8` had reported a preference; `d11`/`d12` showed that was a validation-set artifact) |
| 4 | The optimiser gives flat directions **full-size steps** | `lr_sweep`: ANN offset after one step is **exactly `3.48 x lr`**, slope 1, and drift is proportional to `lr`. Adam divides by `sqrt(v_hat)`, so a direction with a vanishing gradient still receives an `O(lr)` step. Independently: on a perfect-match null, Adam's first move is `~1.0 x lr` per coordinate at **every** lr tested |
| 5 | The damage therefore scales with **optimiser steps**, not epochs or data | MS5, two checkpoints: **130 batches -> 9.5x**, **5200 batches -> 127x** |
| 6 | The deployment metric is **hypersensitive** in exactly that direction | `K = 0` double integrator: a constant force gives `f t^2 / 2`, so `0.005 f` at 0.1 s against `72 f` at 12 s -- a factor **14400**. Measured on this plant: injected-constant-force defect grows as `nf_seg^2`, exponent **`1.993`** |

Links 1, 2, 5 and 6 are measured on the production path this session. **Links 3 and 4 are
measured on a different rig** (perfect-match null, routing `(3,4,5)`) and are the weakest
part of the chain (see §5).

---

## 3. The statement to search on

> A training objective is provably flat (statistically neutral, "data-silent") in a
> parameter direction to which the deployment metric is hypersensitive. The objective is
> minimised by a scale-free adaptive optimiser (Adam), which grants flat directions steps
> of order `lr` regardless of gradient magnitude. The learned component therefore
> accumulates a systematic offset along that direction at a rate set by the learning rate,
> and the offset is amplified by the plant's marginally stable modes into a failure of the
> deployment metric, while every quantity the objective can see improves.

**Primary question.** What is known about adaptive optimisers accumulating offsets along
objective-flat directions when a downstream metric is not flat in those directions?

**Secondary question.** Where does the fix belong: the **objective** (make the direction
non-flat without restricting the model class), the **optimiser** (do not grant full steps
to unidentified directions), or the **parameterisation** (remove the degree of freedom
from the model rather than penalising its use)?

**Candidate vocabularies, none yet searched under this framing:** flat / degenerate /
sloppy directions in loss landscapes; practical identifiability and sloppy models;
implicit bias and implicit regularisation of adaptive methods; parameter drift in
overparameterised models; null-space or unidentifiable-subspace drift; preconditioner
behaviour on near-singular Hessian directions; continual-learning drift along
low-curvature directions.

---

## 4. What is already ruled out (do not re-derive)

Each on measured evidence, with the reference:

* **Longer training windows.** SLURM 71013 (`nf` 800 to 3200, DC present at every horizon);
  NF=900 diverged; `O(N^3)` conditioning at `|lambda| = 1`.
* **The multiple-shooting continuity/defect term.** BUILT and verified as the method (5/5:
  zero defect on an exact model `8.6e-07`, Ribeiro Thm 2 equivalence `1.19e-06`, linear in
  injected force exponent `1.000`, `nf_seg^2` exponent `1.993`, minimisable). Then MEASURED
  DEAD: a model 65x worse on the free run produces a defect **1.01x** larger, flat across
  `n_seg` 4/12/30. A coherent (mean) aggregation of the same defects reaches 13.80x but
  ONLY with oracle states; with the real encoder it reaches 1.50x.
* **ARTBP / unbiased truncated BPTT / better gradient estimators.** `drift-conclusions` C4;
  Beatson and Adams (ICML 2019) Thm 4.1 at `|lambda| = 1`.
* **Hard class restrictions** (passivity, contraction, RENs, bounded impulse, spectral
  caps). Violate full expressivity, which is this project's non-negotiable requirement;
  separately measured to fail (passivity bounds velocity, not position).
* **Zero-mean / window-mean priors on the velocity rows.** They target exactly the measured
  mechanism and are still ruled out: the real residual's mean is `-157.5 N` / `-83.7 N` at
  315 to 344 sigma, so suppressing DC forbids learning real friction.
* **Optimiser swap or lr tuning as the fix.** Drift is proportional to `lr`; SGD learns
  `+0%` on a real residual. Note this rules them out as a DELIVERABLE, not as evidence
  about the mechanism -- the `lr` proportionality is link 4 above.
* **"Multiple shooting was tried and failed (Optuna 69399)".** That run was pre-D-101
  (silent Adam default `lr = 1e-3`) and was an `nf` sweep under SINGLE shooting with no
  defect term.

---

## 5. Honest weaknesses of this framing

**Links 3 and 4 are the weak ones.** Both were measured on the perfect-match null rig with
routing `(3,4,5)`, not on the production path whose failure this document describes.
Neither has been re-measured where it matters. The chain is therefore: strong on WHAT
(links 1, 2, 5, 6, production path, this session), weaker on WHY (links 3, 4, other rig).

**A cheap falsifier exists and has not been run.** If the offset is a random walk along a
flat direction, the SIGN of the per-row mean should vary across checkpoints and seeds; if
it is a systematic gradient, it should not. `drift-visual/figures/f07` already holds 10
checkpoints x 8 rows of per-row means. This needs no new run.

**This may not be distinct from exposure bias.** "The model never sees its own drifted
states during training" and "the DC direction is flat in the training loss" may be two
descriptions of one thing. `docs/rollout-stability-literature.md` already holds that
literature (pushforward, GNS noise, scheduled sampling). Nothing measured separates them.

**Two cheap open items remain unchecked.** (a) The training loss is a normalised,
dimensionless MSE while the selector is raw metres -- channel weighting may determine which
rows acquire the offset. (b) The two augmented latent states have collapsed to almost pure
DC (`|mean|/rms` = `0.956` and `0.886`), which is a separate phenomenon from the routed
force rows and has not been investigated at all.

**All results from this session are 1 seed and mostly 1 record**, below this project's
3-seed floor. MS11 and MS12 included.

---

## 6. A constraint on validation, which applies to any candidate fix

On the current simulation the unmodelled truth is a mass-spring-damper absorber, whose
residual is oscillatory and zero-mean (`dA`: dominant coupling `|mean|/rms ~ 1e-4`). So
**the correct DC on this testbed is genuinely zero.** Any method that suppresses DC will
therefore look successful here while being exactly wrong on real data, where friction
contributes a mean at 315 to 344 sigma. The testbed cannot distinguish "suppress spurious
DC" from "suppress all DC".

**Consequence: the injected-friction simulation is a prerequisite for testing anything in
this family, not an optional extra.** `scripts/gantry/datasilent-friction-sim/` was built
to step 2; steps 3a and 3b were never built.

---

## 7. Provenance

Measurements: `docs/results-log-2026-07-26.md` (numbers only, no interpretation) and
`docs/gantry-augmentation-problem-log.md` §12 rows MS1-MS12. Scripts and unit JSONs in
`scripts/gantry/pysynth-data/` and its `results/`. Prior framing and the five unexcluded
readings: `docs/narrowband-objective-problem-2026-07-26.md`. Prior sweeps, none superseded:
`docs/narrowband-literature-sweep-2026-07-26.md`, `docs/multiple-shooting-sweep-2026-07-25.md`,
`docs/drift-literature-sweep-2026-07-25.md`, `docs/rollout-stability-literature.md`.
