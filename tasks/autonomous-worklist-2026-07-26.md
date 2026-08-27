# Autonomous worklist, 2026-07-26 evening

## STATE AFTER A1-A8: four explanations refuted, one paradox, one candidate

**The DC hurts BOTH objectives.** A7: removing it makes the WINDOWED TRAINING LOSS 3%
better (`6.889e-07` vs `7.102e-07`), and the DC alone is worse than no ANN at all. MS12:
it causes 112.8% of the 12 s failure. It is not a trade-off.

**Four candidate explanations are now measured false:**

| explanation | refuted by |
|---|---|
| the loss is blind to the DC | A6 -- curvature `7.06e4`, confirmed on two rigs |
| the loss rewards the DC | A7 -- it penalises it by 3% |
| the DC is cheap because paired with a compensator | A7 -- the pair is not the cheapest arm |
| the DC is an exposure-bias artefact | A8 -- `\|mean_free\|/\|mean_train\|` = 1.00 |

**The paradox, well posed.** The objective strictly prefers the mean-removed network. The
loss's optimum in that direction is `b* ~ 6.9e-10`; the ANN sits at `~3.5e-08`, 51x beyond.
The parameter change needed is ~`1e-6` against an Adam budget of ~`5.2e-4`. Training had
ample room to remove a DC that hurts the objective it minimises, and did not.

**The one surviving candidate** is that the DC is **not independently adjustable**: the same
weights producing the useful AC fit produce the harmful DC. The network's output is mostly
AC (purity `0.16-0.28` on the velocity rows), so the DC is an uncontrolled RESIDUE of
fitting the absorber, not something the network is trying to emit. That is exactly what the
GAM split `g(z) = g0(z) + c` addresses.

**Next test, cheap, no training:** is there a weight perturbation that reduces the output
mean WITHOUT degrading the windowed AC fit? Entangled -> GAM is indicated. Not entangled ->
this is plain non-convergence and the fix is a step-count question.

---

## EARLIER SUMMARY (A1-A4, A6)

**A6 REFUTED the A1/A4 link I proposed an hour earlier. Do not act on it.** I claimed the
loss under-weights Y by 34x (`ystd` Y `1.901e-01` vs X `3.233e-02`) and that this explains
why Y is the only axis with a systematic DC. Measured, the loss's DC resistance is
X `7.064e+04` vs Y `3.387e+04`, **ratio 2.086, not 34.589**. The reason is in numbers A4
itself printed: `std_x` equals `ystd` per channel, because `compute_normalization` derives
`x_all` from `y`, so the state and output normalisations **cancel exactly** and there is no
differential axis weighting. The proposed "train in the selector's units" fix would not
address A1.

**A1's Y-specific systematic sign therefore still has no explanation.**

**What survived, and it is worth more than the retracted link:** A6's production-path
curvatures reproduce `curvature_sensitivity`'s null-rig values to 0.3% (X `7.064e4` vs
`7.084e4`) and 4% (Y `3.387e4` vs `3.542e4`) -- different rig, different script, different
data, encoder live rather than absent. The windowed loss is **robustly stiff on a pure DC**,
so the flat-direction hypothesis is refuted on two independent rigs, not one.

**Two of my own claims are now dead:**
- **Link 3 of `flat-direction-problem-2026-07-26.md` is wrong as stated** (A2). The loss is
  STIFF on a pure DC (`7.084e4`); what is flat is the COMBINED direction, DC plus its
  compensator. `d12` and `curvature_sensitivity` do not contradict -- they measure
  different objects at different points on different rigs.
- **The sweep's headline adjoint recommendation is disqualified for the functional as
  stated** (A3). Adjoint weight ratio DC/150 Hz = `2.27e12`, so it would suppress the
  absorber by twelve orders of magnitude. Any adjoint route needs a different quantity of
  interest.

**A1 is suggestive, not significant.** 2 of 8 rows at `p = 0.0215`, and neither survives
Bonferroni or Benjamini-Hochberg over 8 tests. What lifts it above noise is structure, not
p: the two rows are the same axis with opposite signs. Also across configurations, not
seeds.


Ordered by cost and by what gates what. Phase A needs no training and should complete.
Phase B needs training runs and may not finish; this machine does ~300 optimiser steps per
23 minutes against the ~5200 the failure needs.

Each item states its **pre-declared reading** before it runs, so the result cannot be
fitted afterwards. Status is updated in place.

---

## Phase A — no training

### A1. Drift or diffusion? Sign test across all available checkpoints
**Status: RUNNING**

**Why.** The sweep showed link 4 is non-probative: under `N((k*eta/sigma)*g_bar, k*eta^2*I)`
drift and diffusion are BOTH proportional to `lr`, so the `3.48 x lr` measurement cannot
discriminate. The discriminator is the SIGN pattern: a random walk gives random signs, a
systematic gradient gives consistent ones.

**Why it is possible now.** My §5 falsifier assumed 3 seeds, where all-same-sign gives
`p = 0.25` and cannot reject. There are **ten checkpoints with real weights** on disk:
`71167_last`, `curriculum_70903` rungs 0-3 (warm-started sequentially, so a genuine
parameter trajectory), `trial_ckpts_71013` trials 0-3, and `diagnostics/gantry_drift_last`.
Nine same-sign gives `p = 0.004`.

**Method.** Extract each checkpoint's per-row ANN output mean along a fixed trajectory
(same method as MS12). Binomial sign test per row. Separately, fit the MSD exponent
`||theta_k - theta_0||^2 ~ k^c` across the curriculum rungs, which ARE ordered in `k`:
`c ~ 1` diffusion, `c ~ 2` drift.

**Pre-declared reading.**
- consistent signs (`p < 0.05`) -> systematic drift; the flat-direction framing is wrong
  and something is actively pushing the DC
- random signs -> diffusion; Adam random-walks an under-determined direction, and the fix
  belongs to the parameterisation
- mixed by row -> report per row, do not pool

**Caveat to carry:** these are different RUNS with different `nf`, not seeds of one run. A
sign test across configurations is weaker evidence than across seeds and must be labelled
as such.

### A2. Reconcile the flat-vs-stiff contradiction
**Status: PENDING**

`d12` says the DC direction is NEUTRAL on training windows (`Delta/SE = +0.71`, n=120).
`curvature_sensitivity.py` (problem-log line 310) says the windowed loss is **STIFFEST** on
the integrator output DC (`d2L/db2 = 7.084e4` on X), and explicitly REFUTED the flatness
hypothesis on 2026-07-24. These cannot both describe the same object.

**Method.** Read both artifacts and determine what each actually probed: `d12` uses a mean
signed short-window tendency error on a training bank; `curvature_sensitivity` probes a
constant bias on the ANN output column at `b = 0`, 6-state, route 0..5, true-init, no
encoder. Establish whether they differ in coordinate, in rig, or in the quantity.

**Pre-declared reading.** If they probe the same direction, one is wrong and the framing in
`flat-direction-problem-2026-07-26.md` collapses. If they probe different objects, both can
stand and the document must say which one link 3 rests on.

**Note.** D1 measured the loss optimum at `b* ~ 6.90e-10`, nonzero, while the ANN parks at
`3.5e-08` — **51x beyond it**. That gap is unexplained by either "flat" or "the loss wants
this DC", and is the real open quantity.

### A3. Adjoint weight profile versus frequency
**Status: PENDING**

Tests my objection to the sweep's headline recommendation. For a long-horizon position
functional on a double integrator the adjoint weight is `~(T-t)`, so DC is amplified by
`T^2/2` while a 150 Hz oscillation should get a near-zero weight — i.e. adjoint weighting
would suppress the drift and simultaneously de-emphasise the absorber we want learned.

**Method.** Compute the adjoint weight of the ACTUAL deployment functional (12 s trajectory
sim-RMS, not terminal position) on the linearised model, as a function of input frequency.
Report the weight at DC and at 150 Hz.

**Pre-declared reading.** ratio > 100 -> the objection stands and adjoint weighting trades
G2 for G1. ratio ~ 1 -> my objection is wrong and the trajectory functional does not have
the terminal-position pathology.

### A4. Loss/selector normalisation mismatch
**Status: PENDING**

The training loss is a normalised dimensionless MSE; the selector is raw metres. Never
checked. Report `ystd` and `std_x` per channel and whether the drifting axes are weighted
down relative to the metric that judges them.

### A5. Why the augmented states collapsed to DC
**Status: PENDING**

MS12 found `aug0` and `aug1` at `|mean|/rms` = `0.956` and `0.886` — nearly pure constants.
These are the free latent states that exist to represent unmodelled dynamics. Nobody has
looked at why they degenerate. Cheap: inspect their trajectories and their coupling into
the physical rows.

---

## Phase B — needs training, may not finish

### B1. GAM reparameterisation prototype
**Status: PENDING**

The only candidate that targets LEARNING rather than drift. Parameterise the block as
`g(z) = g0(z) + c` with `g0` sum-to-zero constrained in its basis and `c` explicit.
Rationale: MS12 shows the network's state-dependent part currently contributes essentially
nothing (eight constants give 112.8% of the failure), so removing the DC degree of freedom
should force its capacity onto the shape. Expressivity is untouched — the DC remains
representable, by `c` instead of by `g0` — which is what kills every zero-mean prior and
not this.

**Readings:** `R2_linmap` for the augmented states against `delta_a` / `vdelta_a` (currently
`0.270` / `0.066` on the degraded checkpoint) should RISE, and the 12 s free run should stop
collapsing. Both come from one run.

**Honest cost:** reproducing the failure needs ~5200 optimiser steps; this machine does
~300 per 23 min. A meaningful arm is hours, and background jobs here get killed.

### B2. Huber loss swap
**Status: PENDING**

Brenowitz et al. report this project's exact setup — learned block inside a physics model,
Adam, normalised MSE on short windows, long free-run deployment, network wins every offline
metric while being more biased in the mean — and attribute it to non-Gaussian outliers
distorting MSE. Remedy is Huber/MAE. Violates none of the four constraints. Cheap change,
same training cost.

---

## Not doing, and why

- **MS10 rerun** (finite-difference probe at smaller alpha). Superseded: it was measuring
  which loss points where, and A1/A2 are upstream of that question.
- **Any zero-mean penalty.** Ruled out on R2; real residual mean is `-157.5 N` / `-83.7 N`
  at 315-344 sigma.
- **The Y-scheduled encoder.** MS11/MS12 show the failure is a force error, not an
  initial-condition error, and a residual-based objective is far less encoder-sensitive than
  the defect was. Withdrawn as a proposal.
