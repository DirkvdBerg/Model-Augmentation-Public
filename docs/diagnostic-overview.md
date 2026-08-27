# Diagnostic overview: what is actually established, and how the diagnosis moved

**Written 2026-07-26 by an independent review pass.** Purpose: give a new person the
shortest path to trusting something here. Nothing in this document inherits a framing.
Every claim carried forward was re-derived from a stored artifact, or is marked as not
having one.

**Method.** I read the run table (`docs/gantry-augmentation-problem-log.md` §12), the
measurement log, all seven framing documents, and then opened the artifacts. I loaded both
production checkpoints and read their per-epoch histories directly; I opened the D1, D2,
D3, D4 unit JSONs, all twelve MS unit JSONs, the twenty ARTBP training `.npz` files, and
`drift-visual/data/f07.npz`. Two parallel sweeps enumerated the script folders for retained
output. Where a document and a stored number disagree below, the number is quoted and the
document is named.

**Grades** (project convention, `docs/drift-conclusions-2026-07-25.md` §0):
ROBUST = 3 seeds, 2 protocols. SOLID = 3 seeds, 1 protocol. SINGLE = 1 seed or 1 record.
ORACLE = uses information unavailable on real data. VOID = the run's own control failed.
ASSERTED = repeated in prose, no artifact found. I use ASSERTED heavily and it is earned.

**Horizon rule.** No error number in this document appears without its horizon. This is not
pedantry: the repo contains at least two wrong conclusions traceable to comparing across
horizons, and the warning sentence the project itself wrote to guard against this
(`results-log-2026-07-26.md` line 26, `7.86e-05` at 2 s) is itself unsourced. See §8.

---

## 1. The problem, in one paragraph

A static ANN, zero-output at initialisation, is trained inside an LPV baseline by minimising
simulation error over 0.1 s windows, and is selected on a 12 s free-run RMS. On the
production run the 12 s free-run error goes from `1.661e-04 m` at epoch 0 to `2.109e-02 m`
after one epoch, a factor of 127, and never returns below epoch 0 in twenty epochs, while
over the same run the **validation** windowed error falls 14 percent and the training
windowed error falls 12 percent. Same model, same validation record, same data: the
windowed view improves monotonically while the deployed view collapses. The mechanism at
the trained checkpoint is not an instability. Replacing the entire 600-parameter network
with eight constants (its own per-row time-average along the free run) reproduces 113
percent of the 12 s error, and a twin-perturbation test shows the error propagator grows
sub-linearly and polynomially, not exponentially. The eight constants amount to about
17 mN on X and 23 mN on Y, and X and Y are pure double integrators, so a constant force
integrates as `f t^2 / 2` and is amplified by roughly 14400 between the 0.1 s objective and
the 12 s deliverable. The failure is therefore a small constant force error on marginally
stable axes, which the training objective prices at a factor 14400 discount relative to the
metric that judges it. That much is measured and survives scrutiny. What is *not*
established, despite being asserted in the newest framing, is that the objective is *flat*
in that direction; the project's own curvature measurements say the opposite.

---

## 2. The verified core

Ordered by how much rests on each. Everything in this section I opened and checked myself
unless marked otherwise.

### C-1. The failure, read off the production checkpoint. VERIFIED, grade SINGLE.

| quantity (all from one file) | epoch 0 | epoch 1 | epoch 20 |
|---|---|---|---|
| val sim-RMS, **12 s free run**, V1 to V4, metres | `1.661376e-04` | `2.109e-02` (**126.9x**) | `1.950576e-02` |
| val nf-window RMS, **0.1 s**, metres | `4.385691e-05` | `4.16157e-05` | `3.771019e-05` (**-14.0%**) |
| train nf-window RMS, **0.1 s**, metres | `3.807315e-05` | `3.5771e-05` | `3.334428e-05` (-12.4%) |
| train loss, normalised, dimensionless | *not recorded* | `1.36860e-06` | `1.103392e-06` (-19.4%) |

Artifact: `simulations/gantry_subnet/augmentation_linear_map/71167/gantry_drift_71167_last.pth`.
Produced by `scripts/gantry/gantry_interconnect_dynamic.py`, SLURM 71167, MATLAB
`augmentation` data, routing `(0..7)`, `nx_ann=2`, `na=nb=17`, `lr=1e-7`, `nf=400`,
`up_sample=1`, seed 42, 20 epochs / 5200 batches. Load with `scripts/gantry` on `sys.path`.

`argmin(Loss_val) = 0` and the stored `bestfit` field is `1.66137600899674e-04`, i.e. the
epoch-0 value, which independently confirms that `gantry_ckpt_71167.pt` **is** the
initialisation and can never serve as a degraded model.

**Two corrections to how this is reported elsewhere.** (a) `Loss_train[0]` is `NaN` in the
file, so the "-19%" is measured from epoch 1 to epoch 20, not from epoch 0; `results-log`
§2 and `ann-worse-than-init` §2 both present `1.369e-06` as the starting value without
saying it is epoch 1. (b) `Loss_val` is not monotone after epoch 1: it falls to
`1.315e-02` at epoch 9 (79x) before rising again to `1.951e-02`. "Never returns" is true
against epoch 0 and is the claim that matters; "monotone" is not stated anywhere but is
easy to infer from the way the two endpoints are quoted.

The load-bearing number is the **validation** windowed improvement. It excludes train/val
overfitting as the explanation, and it is the single most important measurement in the
repository.

**Third independent copy.** The same 21-element `loss_val` array is stored again in
`71167/gantry_results_71167.npz` (written by `gantry_dynamic/evaluation.py`), and
`gantry_ckpt_71167.npz` records `bestfit = 0.000166` with `done_epochs = 20`. Three
artifacts from the same run agree. C-1 is the best-provenanced claim here.

### C-1b. The augmentation never learned the absorber. VERIFIED, grade SINGLE, under-reported.

From `71167/gantry_state_recovery_71167.npz` (2087 windows): the six physical states are
recovered well (`r2_lin` `0.9998 / 0.99996 / 0.941 / 0.9999 / 0.99998 / 0.941`) but the two
augmented latent states give `r2_aug_raw = [-1.210, -1.354]` and
`r2_aug_lin = [0.344, 0.350]`. The latent states did **not** learn `delta_a`. The same
figure appears in `gantry_ckpt_71167.npz` (`diag_r2_linmap`) and in the earlier SLURM 68676
log (`+0.0016 / +0.0107`, worse).

Put next to MS12's finding that those same two rows have `|mean|/rms` of `0.956` and
`0.886`, i.e. they are almost pure DC, the picture is that the two free latent states became
constant carriers rather than absorber models. `flat-direction-problem` §5 lists this as
"a separate phenomenon that has not been investigated at all"; it has in fact been measured
three times and the measurement is unambiguous.

Also in that folder: `gantry_grad_norms_71167.npz` records `encoder.0 = 1.159e-04`,
`encoder.1 = 2.253e-03`, and `encoder.2` through `encoder.8` **exactly `0.0`**. Consistent
with `diag9`'s `t5_grad_aug_nf1 = t5_grad_aug_nf400 = 0.0`. Large parts of the model receive
no gradient at all. Neither fact is in any framing document.

### C-2. Dose response in optimiser steps. VERIFIED, grade SINGLE.

`simulations/gantry_subnet/diagnostics/checkpoints/gantry_drift_last.pth`, 5 epochs,
130 batches. `Loss_val` (12 s free run, metres): `8.060659e-05`, then
`1.83x / 4.69x / 6.06x / 7.16x / 9.47x`, monotone. Its windowed metrics move by under
1.5 percent over the same run (`Loss_val_nf` `3.1824e-05` to `3.1403e-05`).

So 130 updates costs 9.5x and 5200 updates costs 127x. The damage scales with **optimiser
steps**, not epochs and not data. This is what makes short reproduction runs useless (§5,
MS3) and is the correct sizing rule for any future arm.

### C-3. The ANN's DC output is sufficient for the 12 s failure. VERIFIED twice, grade SINGLE seed.

Four 12 s free runs on `gantry_drift_71167_last.pth`, record V1, `H = 47979` steps:

| arm | 12 s free-run RMS, metres | relative |
|---|---|---|
| ANN-OFF | `1.3113e-04` | floor |
| FULL | `7.7194e-03` | 58.9x floor |
| **MEAN-ONLY** (network replaced by its own 8 per-row means) | **`8.7084e-03`** | **112.8% of FULL** |
| MEAN-REMOVED | `2.8711e-03` | 37.2% of FULL, still 22x floor |

Artifact `scripts/gantry/pysynth-data/results/MS12_ann_dc_force.json`, script
`measure_ann_dc_force.py`. Equivalent forces recomputed by me from the stored
`equiv_accel` and the stored `M_X = 53.8`, `M_Y = 10.1`: dX `+1.673e-02 N`, dY
`+2.332e-02 N`. Confirms the "17 mN and 23 mN" figure.

**Independent cross-check, not noted anywhere.** `scripts/gantry/drift-visual/data/f07.npz`
holds per-row means for ten checkpoints, computed by a different script eleven days earlier
(2026-07-15). Its row for `gantry_drift_71167_last` agrees with MS12 on all eight rows to
three significant figures (dY `+1.20730e-06` vs `+1.20962e-06`; aug0 `-8.548e-05` vs
`-8.542e-05`; dTheta `-1.42961e-04` vs `-1.43646e-04`). Two independent measurements of the
same object on different dates. This is the only claim in the recent campaign with that
property.

### C-4. The error is an accumulated constant force, not a diverging mode. VERIFIED, corroborated on a second checkpoint by two further methods.

MS11 (`orrell_two_run.py`, `results/MS11_orrell_two_run.json`, 12 s = 48000 steps, V1,
seed 42). Twin-perturbation test, which cancels the tendency term exactly and isolates the
propagator: polynomial fit wins in all four arms (`R^2_poly` 0.874 to 0.961 against
`R^2_exp` 0.476 to 0.693), exponent is **sub-linear** (INIT `+0.342 / +0.336`, DEGRADED
`+0.422 / +0.415`) and **alpha-invariant** across `1e-6` and `1e-5`, so the linear-regime
guard passes. Clean-start test: DEGRADED X power law exponent `1.4842` at `R^2 = 0.9972`,
Y `1.6765` at `R^2 = 0.9312`, against INIT `0.4791` at `R^2 = 0.8094`.

**Corroboration the record does not connect.** Two ARTBP artifacts from 2026-07-24, on a
*different* checkpoint (`ckpt_poly6_seed0_ep10_h3200_b256.pt`), reach the same verdict by
different routes:
* `ARTBP/data/growth_aug_vs_base_ckpt_*.npz`: `aug_best = ['quadratic','quadratic','quadratic']`
  against `base_best = ['exponential','exponential','exponential']` on records T1/T3/T5, with
  a stored verdict string naming constant-force displacement rather than exponential
  instability.
* `ARTBP/data/pole_horizon_diag.npz`: `maxabs_base` and `maxabs_aug` agree to the sixth
  decimal at all five Y operating points (`1.00016 / 1.00017 / 0.99999 / 1.00016 / 1.00016`).
  The augmentation does not create an unstable pole. Note the *baseline* already sits at
  `1.00016`, marginally above 1, which is a separate item nobody has logged.

Three methods, two checkpoints, one conclusion. This is the second-best-evidenced claim
here after C-1 and it is the one that settles which fix family is relevant.

### C-5. ARTBP collapses the DC. VERIFIED, grade SOLID (5 seeds, the best replication in the repository).

`ARTBP/data/train_{mode}_seed{0..4}.npz`, 1 epoch, `nf = 400` mean horizon, `H_max = 1600`,
Adam `lr=1e-7`, batch 256, real with-MSD data, routing `(0..7)`. Recomputed by me across all
20 files:

| mode | endpoint DC dY, 5-seed mean | fraction negative | dcgrad variance | held-out 0.1 s nf-RMS |
|---|---|---|---|---|
| `fixed` (biased control) | `-4.122e-06` | **1.00 (sign-locked)** | `2.05e-07` | `1.180e-03` |
| `geom` | `-4.031e-07` | 0.80 | 10.24 | `1.207e-03` (+2.3%) |
| `poly4` | `-2.577e-07` | 0.60 | 4.80 | `1.205e-03` (+2.1%) |
| `poly6` | `-1.214e-07` | 0.60 | 2.22 | `1.212e-03` (+2.7%) |

The control is sign-locked on 5/5 seeds and every ARTBP arm scatters. The variance ordering
`poly6 < poly4 < geom` holds, at 2 to 5x rather than the 24 to 47x a single-seed probe
suggested (the run table records this correction honestly).

### C-6. ARTBP gate-2 at 20 epochs exists on disk and has never been reported. VERIFIED, grade SINGLE.

**This is the most consequential unreported material in the repository.** Five converged
training runs on the production with-MSD data, written 2026-07-23 03:55 to 06:27, each
carrying per-epoch `val_sim_traj` and a free-run drift evaluation against a matched ANN-off
baseline. They appear in no run-table row, no results log, and no framing document. On
2026-07-24 `drift-critical-analysis` §2.8 states gate-2 "has never been run"; on 2026-07-25
`drift-conclusions` §4 lists ARTBP as ruled out; on 2026-07-26 `flat-direction-problem` §4
repeats that. All three post-date these files.

Common reference on this rig: `val_sim_traj[0] = 1.8461e-04 m` (12 s val free run,
epoch 0). Drift ratio is full-ANN over ANN-off, tail-RMS over the last quarter of an
`8000`-step (2 s) free run, per `train_artbp.py:60,275`, on T1/T3/T5.

| arm (seed 0, 20 ep unless noted) | val sim-RMS **12 s**, epoch 1 | best over ep 1-20 | mean Y drift ratio, **2 s tail** | held-out **0.1 s** nf-RMS |
|---|---|---|---|---|
| `fixed`, `H_max`=1600 | `2.905e-02` (157x ep0) | `2.192e-02` (119x) | **83.1x** | `5.197e-05` |
| `geom`, `H_max`=1600 | `2.75e-03` (14.9x) | `1.759e-03` (**9.5x**) | **22.1x** | `5.326e-05` |
| `poly6`, `H_max`=1600 | `7.61e-03` | `2.103e-03` (11.4x) | **13.5x** | `5.309e-05` |
| `poly6`, `H_max`=3200, 10 ep | `6.99e-03` | best = epoch 0 | 47.9x | `6.917e-05` |
| `poly6`, `H_max`=6400, batch 128 | `1.682e-02` | best = epoch 0 | 252.7x | `1.325e-04` |

Read plainly: at `H_max = 1600`, ARTBP cuts the first-epoch collapse from 157x to 15x, the
best-achieved 12 s error by about 10x, and the free-run Y drift ratio by 4 to 6x, at a
2 percent cost in windowed fit. It does **not** fix the problem: every arm's best is still
9 to 11x the epoch-0 initialisation, so G1 (do not degrade the init) fails everywhere.
Raising `H_max` past 1600 makes it strictly worse, and at 6400 the DC-direction gradient
variance reaches `6.59e7` and the fit degrades 2.5x. Separately, the `H_max = 3200` run
lands its endpoint DC at `+1.479e-07`, inside ARTBP's own pre-registered `±3e-7` band, while
its drift is still 47.9x, which is a clean demonstration that DC collapse and drift
reduction are not the same axis.

**Grade SINGLE (one seed per arm), so this does not overturn anything by itself.** But it is
a 4 to 6x drift reduction on the production path with a matched control, and it is the only
intervention in the repository that has ever been measured against the actual deliverable
metric at production step count. It should not be sitting under a "ruled out" heading.

### C-7. The real Telica residual has a large nonzero mean. VERIFIED, grade SOLID on real data, with the significance over-stated.

`scripts/gantry/drift-diagnostics/data/D4_telica_residual.json`, train split only, 11
operating points x `iter0`/`iter8` = 22 logs, 212364 residual samples, against the fitted
run-71447 LPV-LFR baseline **without** its Coulomb term, logical frame, newtons.

Verified: `mean_N` X `-157.4576`, Y `-83.6812`; `mean_over_sem` `314.57` and `344.14`.

**Three qualifications the citing documents drop.**
1. The artifact's own `caveat` field says the SEM assumes iid samples and the residual is
   autocorrelated, "so the true resolution is coarser". The "315 to 344 sigma" figure is
   therefore an upper bound on significance, not a measurement of it.
2. `noise_std_at_rest_N` is `172.5 N` on X, *larger* than the `157.5 N` mean it is being
   contrasted with, and `std_N` is `230.7 N`, so the residual's own scatter is 1.5x its
   mean. The mean of 212k samples is still well resolved; the signal is not large relative
   to itself.
3. `mean_after_fitted_coulomb_N` is `-106.4` (X) and `-36.8` (Y). Between 32 and 56 percent
   of the "legitimate nonzero mean" is a Coulomb term the **baseline's own parameter set**
   can already carry. The argument "the ANN must be allowed a nonzero mean, therefore
   zero-mean priors are dead" is weaker than presented, because part of that mean does not
   belong to the ANN.

4. **The force scale on this data is under an open RED verdict, and C-7 is measured against
   a model fitted to it.** `simulations/gantry_subnet/diagnostics/linear_id/summary.json`
   concludes verbatim: *"RED: best physical fit needs ~half the real mass
   (m_total x0.49). Data NOT directly fittable by the physical baseline, force-input
   scale/units (D-077) must be resolved before parameter recovery."*
   `diagnostics/residual_force/summary.json` independently returns
   `inertial_scale_s` of `0.505 / 0.478 / 0.656` on the three axes. Two separate analyses
   say the force-to-acceleration scale is off by about a factor two. A residual **mean in
   newtons**, measured against a baseline fitted under that unresolved scale, inherits the
   question. This does not overturn "the residual mean is nonzero"; it does mean the
   magnitude `-157.5 N` should not be treated as calibrated.

**And the companion `dF/dv` claim does not survive its own artifact.** The headline is
`slope_aggregate = -173.32 N/(m/s)` on X, negative on 22 of 22 logs. But the same JSON
block records `frac_bins_positive: 0.5` and `frac_bins_agree_with_aggregate: 0.5` for the
velocity-binned decomposition, with bin slopes running `+28738, -9912, -765, -586, +216,
+435, -887, +82, -31, +342, +150, -2673`. The aggregate sign is a real fact about the
regression; the physical reading ("the residual needs more damping, not less") is not
supported at bin level and is dominated by the two extreme-velocity bins.

### C-8. The windowed loss is not flat in the DC direction. VERIFIED, **grade upgraded 2026-07-26 late: now confirmed on TWO independent rigs.**

> **ADDENDUM (later session, 2026-07-26 late).** A6
> (`scripts/gantry/pysynth-data/a6_dc_resistance_per_axis.py`,
> `results/A6_dc_resistance.json`) re-measured this on the **production path** -- 8 states,
> routing `(0..7)`, MSD data, **encoder live** rather than absent -- and obtained
> `d2L/db2` = X **`7.064e+04`**, Y `3.387e+04`, against this entry's null-rig X
> `7.084e+04`, Y `3.542e+04`. Agreement **0.3%** on X and 4% on Y, across different rig,
> script, data and encoder condition. A6 used an h-sweep with a 20% adjacent-agreement
> acceptance rule and reported Theta / dTheta as UNRESOLVED rather than quoting noise
> (its v1, at a single `h`, returned a NEGATIVE curvature on the Theta control and is
> void). This is now one of the better-replicated claims in the repository.

`scripts/gantry/drift-diagnostics/data/D1_zeroinit_2d_seed{0,1,2}.json`, frozen rig
`e1b0511a4c`, perfect-match null, routing `(3,4,5)`, `nf = 400`:
`H_2x2` eigenvalues `200.12` and `3153.66`, condition number `15.76`, positive definite;
`g = (+2.1088e-06, -1.7414e-09, -3.4180e-08)`; `b* = -H^-1 g` norm `6.9037e-10`, i.e. 51x
below the parked `3.5e-08`; Frye flatness index `3.948e-16` against a `> 0.9` flat cutoff.
Corroborated on a second rig by `baseline-null/curvature_sensitivity.py`, which measured
`d^2L/db^2` of `7.084e4` (X) and `3.542e4` (Y) against `3.967e-1` (Theta), eps-invariant and
matching autograd to four figures.

**Grade caveat, stated in the source and worth repeating.** At the zero-output evaluation
point the ANN emits exactly zero, so `L`, `g` and `H` do not depend on the weights and all
three seeds return **bit-identical** values (`L0 = 8.847964678634912e-13`). "3 seeds" here
is an identity, not a replication. C1 is graded SOLID in `drift-conclusions`; on the
project's own definition it is SINGLE.

**A contradiction inside the artifact that no document reconciles.** The 3-D companion file
carries the auto-verdict string `"DC-NOT-THE-CARRIER (b* comparable to the parked value;
re-read I7)"` with `b_star_norm = 3.4912e-08`, essentially equal to the parked constant.
`drift-conclusions` C1 reports only the 2-D number. I resolved this: the 3-D
`b*` is `[dX -6.755e-10, dTheta +3.4905e-08, dY +2.002e-10]`, so the entire norm sits on
**dTheta**, the softest eigen-direction (eigenvalue `0.0625`, 3-D condition number `5.0e4`),
and on the two drifting axes the 2-D and 3-D answers agree to within 1 percent. **C1's
verdict stands.** But the selection of the 2-D subspace is not disclosed anywhere, and a
reader who opens the 3-D file gets the opposite headline.

### C-9. The encoder initial condition settles; it does not drift. VERIFIED at 12 s only.

`results/MS2_encoder_drift.json`, `pysynth_baseline` (model equals data exactly), ANN off,
3 records, 12 s free run. Recomputed by me from the stored per-record arrays:
pooled RMS across records, X: true-x0 `1.07e-06 m`, encoder-x0 `2.500e-04 m`, a **234x
bounded offset**; mean late slope, encoder arm, X: `5.876e-07 m/s`. On `pysynth`
(absorber on): pooled X RMS true-x0 `1.246e-04` against encoder-x0 `2.308e-04`, mean late
slopes `1.595e-05` and `2.112e-05 m/s`. All five figures reproduce the document exactly.

**The 2 s arm is not on disk.** `measure_encoder_drift.py` takes `--steps` with default
48000 and writes a single report with no horizon field; the file was overwritten by the
12 s run. So the "92x slope collapse from 2 s to 12 s", which is MS2's headline and the
source of the project's own horizon warning, has a verified 12 s endpoint and an
**ASSERTED** 2 s endpoint.

Consequence that does survive: with a real residual present, both initialisations drift and
a perfect encoder would not remove it. Consistent with rig `e1b0511a4c`, which used
true-state init with the encoder frozen and drifted anyway.

### C-10. The continuity-defect term, as formulated, cannot see the failure. VERIFIED, grade SINGLE.

`results/MS6_defect_sees_failure.json`: free-run ratio INIT to DEGRADED `65.04x`
(`1.1964e-04` to `7.7810e-03 m`, 12 s, V1); defect RMS ratio with encoder nodes
**`1.0143x`**. I recomputed the true-node figure from the stored per-row arrays over the six
physical rows: `4.892e-03` to `8.719e-03` = `1.782x`. `MS6b_nseg_sweep.json` gives
`1.076 / 1.019 / 1.042x` at `n_seg = 4 / 12 / 30` (0.4 / 1.2 / 3.0 s coupled), flat, so
under-coverage is excluded. A pre-registered hard falsifier that fired. Method correctness
is separate and is not in question here; see §5 for its evidential status.

### C-11. The D9 question was answered on 2026-07-26, in a folder nothing cites. Grade SINGLE, and under-stepped.

`scripts/gantry/msd_transfer_diagnostics/data/diag_msd_summary.json`, written 12:24 on
2026-07-26 by `diag_msd_failure_cause.py`. This is the redo of the void D9 run (§5): the
pre-registered "does the Adam-damages-a-good-init mechanism transfer to MSD data" test from
`ann-worse-than-init` §5b. It ran, it wrote an artifact, and no document mentions it.

Config: `mode = augmentation`, `nf = 400`, batch 256, `STRIDE = 100`, 2 epochs, seed 0, five
arms. Result: **every arm's windowed train and val RMS improve**, while
`Loss_val_simNRMS` starts at `16.2766` and moves to `15.58` (adam 1e-7), `12.07` (adam 1e-6)
or `16.272` (sgd 1e-5). By the pre-committed decision table this is **H2, horizon
mismatch**, not H1a: the optimiser does not damage a not-yet-optimal init here, and SGD
simply does nothing (`16.2766` to `16.2720` over three probes).

**Two limits that stop it settling anything.** `STRIDE = 100` and 2 epochs give roughly 52
optimiser updates against production's 5200, which is the same step-count trap that voided
MS3 and which C-2 shows is exactly the axis the failure lives on. And two of the five arms
(`adam_1e-05`, `sgd_1e-07`) recorded only one epoch, so they are truncated. The artifact is
real and readable; the conclusion it supports is "not reproducible at 52 updates", which is
already known.

---

---

## 2b. ADDENDUM, later session 2026-07-26 (evening): five further verified entries

Added after this document was written. Same grading rules. **All are 1 seed.**

### C-12. The DC hurts BOTH objectives. VERIFIED, grade SINGLE. *The sharpest new fact.*

`pysynth-data/a7_pairing_test.py`, `results/A7_pairing_test.json`. Windowed **training** loss
(the objective actually minimised, never previously measured for these arms), `nf = 400`,
40 windows, training record T1, checkpoint `gantry_drift_71167_last.pth`:

| arm | windowed loss (0.1 s) | vs FULL |
|---|---|---|
| FULL | `7.1019e-07` | -- |
| MEAN-ONLY | `7.9367e-07` | 1.1175x |
| **MEAN-REMOVED** | **`6.8889e-07`** | **0.9700x** |
| ANN-OFF | `7.4722e-07` | 1.0521x |

**Removing the DC makes the TRAINING objective 3% better**, and the DC alone is worse than
no ANN at all. Combined with C-3, the DC costs 3% on the windowed loss AND 2.7x on the 12 s
free run: **it is bad for both, so it is not an optimisation trade-off.** The pre-declared
prediction (that the DC is cheap because paired with a compensating term) was REFUTED.

### C-13. It is not exposure bias. VERIFIED, grade SINGLE.

`results/A8_exposure_bias.json`. Per-row ANN output mean along the FREE RUN versus on
ENCODER-ANCHORED training windows: ratio Y `1.00`, dX `1.00`, dY `0.98`, aug0 `1.00`,
aug1 `0.99`. **The DC is identical where the model is trained and where it is deployed**, so
off-distribution extrapolation is excluded.

### C-14. The DC is not entangled with the useful fit. VERIFIED, grade SINGLE.

`results/A9_entanglement.json`. Twelve small steps descending `sum_r (mean_r)^2` in WEIGHT
space (not output-space surgery): `||mean||` `1.7874e-04 -> 1.5941e-04` (0.892x) while the
windowed loss also fell, `2.0921e-07 -> 1.9582e-07` (0.936x). **Weight space can express
"same shape, less mean", and doing so improves the objective.** Caveat: measured on V1 with
12 windows, so part of the 6.4% may be generalisation rather than descent; C-12's 3% is on
training data and points the same way.

### C-15. A MINIMAL REPRODUCTION EXISTS. VERIFIED, grade SINGLE. *Practically the most useful entry.*

`pysynth-data/k_sweep_minimal.py`, `results/K_sweep_minimal.json`. One damped mass
(`m=1`, `c=0.65`, `tau=1.54 s`), one 150 Hz sinusoid as the unmodelled residual, a
600-parameter MLP on the velocity row, **windows initialised from the TRUE state so there is
no encoder**, `nf=400`, Adam `lr=1e-4`, 1500 steps. At `K = 0`: windowed loss **`0.943x`**
ANN-off while the 4 s free run is **`1.713x`** ANN-off.

**Measured NOT necessary for the failure:** the SUBNET encoder, LPV scheduling, MIMO /
P-transform, the coupled 8-state absorber, and the augmented latent states.
**Measured necessary:** the integrator. Free-run harm ratio `1.713` (K=0) -> `1.059`
(`wn=1 Hz`) -> `0.998` (`wn=10 Hz`).

**Marginality participates in the acquisition, it is not only an amplifier:** the ANN output
mean runs `1.15e-05` (K=0) to `6.87e-09` (`wn=10 Hz`), three orders down. An earlier claim
in this session that marginality is "an amplifier, not a cause" was over-read from the
`wn=1 Hz` arm alone and is withdrawn.

**Honest limit:** reproduces the SHAPE at `1.713x` against the production path's `58.9x`, a
~35x magnitude gap, so **no absolute number transfers between testbed and pipeline**. Its
value is as a screen at MINUTES per arm against HOURS on the real pipeline.

### C-16. The augmented latent states decay; they are not a second integrator. VERIFIED, grade SINGLE.

`results/A5_aug_states.json`, 12 s free run of `gantry_drift_71167_last.pth`. `aug0`
`-3.802e-01 -> -9.221e-05`, `aug1` `+3.539e-01 -> +1.038e-04`, slopes `+4.3e-06` and
`-4.9e-06` per s, both bounded and ending 3 orders below the dY state span
(`-2.474e-02 .. +3.556e-02`). They start large because the encoder's `W^a` rows are random
kaiming (`diag18`: `x_a rms 1.76/1.66`) and decay away. Their high **output** purity
(`0.956`/`0.886`, C-3) is a near-constant output into states that do not accumulate it.

---

## 3. The progression: how the diagnosis moved

Nine framings in seventeen days. Each is listed with what forced the change.

**Phase 1, to 2026-07-13. Horizon mismatch and routing.**
Original failure mode (`problem-log` §3): train on 400 steps, validate on ~8000, best
checkpoint = epoch 0. Blamed on the 20x ratio. Killed by the nf sweep: SLURM 71013 ran
`nf = {800, 1600, 2400, 3200}` cold, and every trial's best was still epoch 0, with the
DC present on 9/9 checkpoints at magnitude `~1/nf`. Longer windows saturate at 1.3x and
never beat the `8.0e-05` initialisation. **Longer windows are dead from this point and stay
dead.**

**Phase 2, 2026-07-17 to 07-18. Where does the DC come from?**
Systematic-vs-diffusion settled by seed agreement (v3b: dY `-4.21 / -3.62 / -3.55e-6`,
sign-consistent across 3 unfixed seeds, born by step 13). Then three candidate sources were
each killed by intervention: physics (v1f, largest physics DC `~1e-7`, five orders below),
encoder initialisation (v3x0, true-init still gives 85 percent of the DC), excitation band
(v3joint broadband 1-200 Hz gives `-4.18 / -3.72 / -3.82e-6`, identical). What replaced
them: truncated-BPTT bias, amplified by Adam.

**Phase 3, 2026-07-18 to 07-22. Adam as the amplifier, then the estimator route.**
SGD at matched lr gave `+1.98e-9` against Adam's `-3.98e-6`, roughly 2000x less, at the same
loss. This became the mechanism story. It is materially weakened by something the sweep
found: the three `v3x0sgd_encoder_seed{0,1,2}.npz` files are **numerically identical** in
every array, so the "3-seed SGD" arm is n = 1. Meanwhile v8 (system equals model, ANN on)
plus its injection-recovery control established that no code or data offset manufactures
the DC, to a measured detection limit of `~3e-7`. That chain is genuinely airtight and is
the best-designed piece of work in the record. ARTBP then closed the mechanism by
intervention (v12, 3 seeds) and Phase D replicated it at 5 seeds.

**Phase 4, 2026-07-23 to 07-24. The null campaign, and two self-corrections.**
`baseline-null` established the model is not buggy (true-x0 12 s floor `4.81e-6 m` against a
`2.16e-5 m` encoder-x0 floor), then walked back its own "pure estimator artifact, decoupled
from signal" claim when the no-signal DC came out 40x below the real-data DC. Then
`curvature_sensitivity.py` **refuted the flat-direction framing** the whole zero-mean thread
rested on, and `gain_vs_dc.py` refuted the "state-dependent gain" over-claim that the
curvature finding had itself produced. Two corrections in two days, both logged in place.
`drift-critical-analysis` (07-24) is the honest audit of this state and remains the best
single document in the repository.

**Phase 5, 2026-07-25. The frozen-rig campaign.**
Protocol discipline arrives: a hashed rig, deterministic full-batch, pre-registered
branches. Its real contributions are methodological. T0.1b found the minibatch drift metric
unusable; T0.1c found that everything read before step 50 is a transient, which
retroactively invalidated T1's secondary numbers; T1b found that the drift the null campaign
had been measuring is **substantially an early-training transient** (control X goes 227x to
0.7x floor between steps 30 and 84 on 2 of 3 seeds). That sentence deserves more weight than
it got: it means most short-null drift numbers quoted anywhere in this repository are
transient readings.

**Phase 6, 2026-07-26. The pysynth / multiple-shooting session.**
D-126 rebuilt the data self-consistently. MS1 found the encoder error scales with distance
from the `Y_op = 0` linearisation point (`corr = +0.995`). MS2 settled encoder-as-offset.
MS3 was invalid (its control passed). MS5 read the failure straight off the production
checkpoints and is the definitive statement. MS6 killed the defect term. MS11 and MS12 then
identified the failure as a constant force and quantified it.

**Phase 7, the same evening. The flat-direction framing.**
`docs/flat-direction-problem-2026-07-26.md` supersedes the narrowband framing written four
hours earlier, on the correct observation that indifference cannot *create* a bias. Its
links 1, 2, 5, 6 are solid and are C-1 to C-4 above. **Its link 3 is the problem.** It
states "the offending direction is measured to be flat (statistically neutral) in the
training objective", citing `d12` (2026-07-12, other rig, other routing). This is the same
claim that `docs/ISSUE-OVERVIEW-FOR-INDEPENDENT-REVIEW.md` §5 lists in its
already-falsified table, and that `drift-critical-analysis` §2.4 documents at length as
"OVER-CLAIM 3", refuted by a measurement that is newer, better graded, and asks exactly that
question. The newest document acknowledges link 3 is weak only on the grounds that it was
measured on a different rig. It does not mention that the project has a direct refutation.

**What the reconciliation actually is**, and it is not in any document: the DC direction has
large positive curvature (C-8) *and* a negligible relative loss cost, because on real data
the windowed loss floor is set by absorber ripple the ANN cannot reduce. `v11_loss_landscape`
measures `L(0)/L_min = 1.0001`, a 0.01 percent effect, on real data, while the same direction
on the perfect-match null is the stiffest in the problem. Both numbers are right. The
failure is a **relative sensitivity ratio** between two objectives, not absolute flatness,
and saying "flat" points the fix at the wrong quantity.

---

## 4. Falsified claims

Claims this project believed and then disproved. The last column is the one that matters.

| claim | what refuted it | still cited as though it stood? |
|---|---|---|
| The train/validate horizon mismatch (400 vs 8000) is the failure | SLURM 71013: all `nf` up to 3200 still best = epoch 0, DC at every horizon | no |
| The drift is practically non-identifiable from this excitation | D3: 3030x more within-window Y traversal (`std 1.194e-02` vs `3.941e-06 m`) leaves tail `|c|` 1.32x *larger* | no |
| "The windowed loss cannot constrain the drift direction" / the DC direction is flat | D1 (positive definite `H`, Frye `3.9e-16`) and `curvature_sensitivity.py` (X curvature `7.08e4`) | **YES.** `flat-direction-problem-2026-07-26.md` §2 link 3, written after both |
| The encoder init bias is the DC's cause | SLURM 70558 (`na=27` collapsed the bias to statistically zero; sim-RMS still rose 11x in one epoch) and v3x0 (true init keeps 85% of the DC) | no |
| The drift is a pure estimator artifact, decoupled from the signal | `baseline_null_artbp.npz`: no-signal DC `-1.09e-7` against real-data `-4.5e-6`, 40x | no, walked back in place |
| The drift carrier is a state-dependent gain | `gain_vs_dc.py`: freezing the ANN's state input reproduces 87% of the drift | no, corrected in the run table |
| "Adam's step is of order lr, so it cannot be mid-relaxation" | measured median per-step move `0.005` to `0.013 x lr` over 9 runs | no |
| Bock and Weiss derive without bias correction | full-text read; their bifurcation inequality also puts this rig on the stable side by 7x | no |
| `arXiv:2006.06650` supports "projection induces a compensating stochastic bias" | full-text grep: the word "bias" does not occur in the body | no |
| The over-damped-baseline argument (residual must have positive `dF/dv`) | D4: `dF/dv` aggregate negative on 22/22 X logs | no. **But see C-7:** the refutation is itself only 50/50 at bin level |
| SGD is the fix (null R4 pass) | `r2_fit_probe.py`: the null pass was inaction; with a real residual SGD learns `+0%` and still drifts 83x floor | no |
| Velocity-only routing fixes the drift | `GV_ROUTE=3,4,5` on the clean null: Y drift 2.4x *worse*, because a velocity-row DC integrates twice | no |
| A Lipschitz cap bounds the drift | v6b: the trained ANN's measured Lipschitz is `~5.1e-4`, orders below every planned cap; Y still destabilises 114x to 270x | no |
| "The DC is not the sole carrier of the drift" (T1) | T0.1c: T1's drift numbers were read at step 30, inside the transient; the claim was marked UNSETTLED | no, retracted in place |
| Anti-damping self-feedback (I3) is what the deliverable failure needs fixing | MS11: growth is polynomial and sub-linear, no exponential mode; ANN adds 1.29x of propagator amplification | no, closed 07-26 |
| "Multiple shooting was tried and failed (Optuna 69399)" | that run was pre-D-101 (silent Adam `lr=1e-3`) and was an `nf` sweep under single shooting with no defect term | no, corrected in `lessons.md` |
| Gate-2 for ARTBP "has never been run" (`drift-critical-analysis` §2.8, 07-24) | five 20-epoch runs dated 2026-07-23, with free-run drift evaluation, sitting in `ARTBP/data/` | **YES.** Repeated as "ruled out" on 07-25 and 07-26 |

**Two claims where a later document re-asserts something the project had already refuted**
are marked YES. Both are in the two newest framing documents. That is the pattern the
`ISSUE-OVERVIEW` §0 warning was written about, and it recurred after the warning was written.

---

### 4b. ADDENDUM (later session, 2026-07-26 evening): four more, three of them that session's own

| claim | what refuted it | notes |
|---|---|---|
| "the DC direction is FLAT in the training loss" (`flat-direction-problem-2026-07-26.md` link 3) | C-8 + A6: curvature `7.06e+04` on two independent rigs | The refuting run (`curvature_sensitivity.py`, 2026-07-24) PRE-REGISTERED this exact hypothesis and refuted it. It was quoted correctly earlier the same day, then contradicted hours later in the same session |
| "the DC is cheap because PAIRED with a compensating term" | C-12: the pair is not the cheapest arm; MEAN-REMOVED is | Pre-declared prediction, refuted by its own test |
| "the loss under-weights Y by 34x, which explains the Y-specific DC" | A6: ratio `2.086`, not `34.589`. `std_x` EQUALS `ystd` per channel (`compute_normalization` derives `x_all` from `y`), so the two normalisations cancel and there is NO differential axis weighting | Proposed and refuted within one hour |
| "adjoint / dual-weighted-residual re-weighting is the fix" (`narrowband-literature-sweep` §2.4) | A3: `\|G\|^2` ratio DC to 150 Hz = **`2.27e+12`** | Disqualified for a long-horizon POSITION functional -- it would suppress the 150 Hz absorber by twelve orders. A different quantity of interest may survive |

### 4c. A propagation error worth recording

`docs/drift-conclusions-2026-07-25.md` C4 ruled ARTBP out on theory (Beatson and Adams at
`|lambda| = 1`). The five converged 20-epoch ARTBP runs in C-6 are dated **2026-07-23** and
therefore PRE-DATE every document that rules ARTBP out. On 2026-07-26 that ruling was
carried, unchecked, into the anti-scope of **two** further research briefs
(`flat-direction-problem-2026-07-26.md` §4 and `dc-accumulation-research-brief-2026-07-26.md`
§3), and two literature sweeps were run against it. **So the only intervention ever measured
against the real deliverable at production step count, showing a 4 to 6x drift reduction,
was actively excluded from the search twice in one day.** The lesson is the one this
document's method already states: a kill is only as good as the arm that produced it, and
the arm must be looked at.

---

## 5. Void runs

Attempted measurements whose own control failed, or which produced no usable number. Their
numbers must not be quoted. Keeping them listed is what stops them being re-run.

| id | what was attempted | why it is unusable |
|---|---|---|
| **MS3** | 3-arm defect on/off A/B on `pysynth` | The control **passed** both gates on `pysynth` (G1 1.000, G2 0.595) and on the original MATLAB data (G1 1.000, G2 0.857). Sizing to a foreground call cut the step count ~250x (~21 Adam steps against production's ~5250). A 10x-steps rerun showed the model had not moved at all: `first = best = last` bit-identical on three metrics, 0 non-finite. Arms B and C were never launched |
| **MS4** | resume from a degraded checkpoint | Loaded `gantry_ckpt_71167.pt`, which is the *best* checkpoint, i.e. epoch 0. The `9.02e-01` reading was a normalisation-frame mismatch from a 4-train-record trim, not a degraded model |
| **MS8** | gradient alignment against `theta_deg - theta_init` | Cosines `+0.0001 / +0.0057 / +0.0054`, all below chance for 600 parameters (`1/sqrt(600) = 0.041`), **including the control**. The reference is an Adam-accumulated displacement, not a gradient |
| **MS9** | gradient alignment against the free-run gradient at `H = 4000` (1 s) | Pre-registered control required `cos(g_fit, g_free) <= ~0`; measured `+0.4588`. A DC error is 144x weaker at 1 s than at 12 s, so the reference sat inside the blind spot being tested |
| **MS10** | finite-difference probe along each candidate gradient | Two separate problems. (a) `results-log` §9 says the run is INCOMPLETE pending the random control; the artifact (written 16:47, before the log at 18:01) **already contains it**: `RANDOM (C2)` at `alpha=1e-5` costs `+2335x` against `fit +59331x`, `rms +9968x`, `coh +7608x`. (b) A `1e-5` step raising the error 2335x even for a random direction is far outside any linear regime; MS11's own row calls MS10 "measured in a saturated regime". The ranking is not a statement about gradients. Also, `E0 = 6.939e-06` is not on the same scale as any other 12 s number in the campaign and the script's metric definition is not recorded |
| **D7** | 400-step null run, named by `drift-conclusions` §6 as **"the single next action"** | **Crashed.** `data/D7_summary.json` is `{"0": {"error": "IndexError: index 83 is out of bounds for axis 0 with size 3"}}`. The log reaches step 100 of 400 on seed 0. `manifest_d7.json` nonetheless records `"status": "ok"`. `ISSUE-OVERVIEW` §6 says it "was stopped at ~step 350" and that its partial result "suggests the constant does converge"; no artifact holds that partial result. The one probe that did land (step 84) reproduces T1b and adds nothing |
| **D9** | the pre-registered MSD version of the "worse than init" test (`ann-worse-than-init` §5b) | **Two independent failures.** (a) `data/D9_summary.json` has `NaN` in every probe field with a stored `TypeError`. (b) Worse: the SGD control **never switched optimiser**. The log shows `Model already initilized (init_model_done=True), skipping ... the creation of the optimizer`, and the SGD arm reproduces the Adam arm bit-for-bit (`It 26` sim-NRMS `513.4`, `It 52` `443.1` in both). `logs/d9_sgd.output` is 0 bytes. Also the run used routing `(1,4,6,7)`, not the `(0..7)` of the failure it was testing. `ISSUE-OVERVIEW` §6 quotes its "16.3 to 513 to 443" numbers as live evidence |
| **diag_xy_routing_blowup** | Theta-only vs Theta+X+Y routing | Invalid control: Theta-only also went 23x above init and never beat epoch 0, on 8 gradient steps. Cannot attribute the X/Y divergence to K=0 structure |
| **Pole perturbation (`GANTRY_KX_ART=1000`)** | move the position poles off `z=1` | Confounded: the stiffness wrecked the fit 650x and the DC did not shrink. The decay rate is damping-limited, so the knob cannot move the target |
| **T0.1 minibatch control** | frozen-rig reference numbers | Control failed its stability precondition: X drift spans 4.6x across seeds, per-row DC sign flips on all three routed rows. Superseded by the full-batch protocol |
| **R2 SGD lr sweep (TASK 6)** | is there an SGD lr that learns without drifting | Reported confounded by its author: non-robust Adam reference (`+18%` to `-13%` on a window-sample change), underpowered windows, and an anti-damping injection that destabilises the truth |
| **v6b binding Lipschitz sweep** | binding caps `L in {0.1, 0.01, 0.001}` | Both shards stopped mid-second-config; the completed configs showed no cap can bind (natural Lipschitz `~5.1e-4`). Verdict stands on the first configs; the sweep itself is partial. `v6b_shardA/B.log` exist with no `.npz` |
| **v7 optimizer 4-way** | Adam / SGD / AdamW / DC-pin at matched lr | `v7_optimizer_comparison.npz` does not exist. `v7_run.log` ends mid-run at step 150 of the `adamw` arm. Only a figure survives |

---

### 5b. ADDENDUM (later session, 2026-07-26 evening): four more void runs

| run | what was attempted | why it is void |
|---|---|---|
| **MS8** | gradient alignment against `theta_deg - theta_init` | cosines `+0.0001 / +0.0057 / +0.0054`, **all below chance** (`1/sqrt(600) = 0.041`) INCLUDING the control. The reference is an Adam-ACCUMULATED DISPLACEMENT, not a gradient |
| **MS9** | gradient alignment against the free-run gradient at `H = 4000` (1 s) | pre-registered control required `cos(g_fit, g_free) <= ~0` (C-1 shows the windowed loss makes the free run 127x worse); measured **`+0.4588`**. A DC error is 144x weaker at 1 s than at 12 s, so **the diagnostic sat inside the same horizon blind spot it was built to measure** |
| **MS10** | finite-difference probe, one step along each candidate's gradient | run at `alpha` far outside the linear regime: a single step gave `+59331x` on the 12 s error where production's 5200 steps give 127x. The RANDOM control also hurt (`+2335x`), so the ordering carries no information |
| **A6 v1** | per-axis DC resistance at a single `h = 1e-6` | returned a **NEGATIVE** curvature on the Theta control, impossible at a minimum: noise-limited. Superseded by the h-swept v2 in C-8 |

Three of these four were caught by their **own pre-registered controls** rather than by
inspection. That is the mechanism working, not four wasted runs -- but note MS9's lesson
specifically: **a diagnostic must not share the blind spot it is built to measure.**

---

## 6. What is genuinely open, and what is merely unfinished

**Genuinely open** (nobody knows, and a well-posed measurement would change the answer):

1. **Does anything reduce the 12 s free-run error below the epoch-0 initialisation?** After
   seventeen days and every intervention tried, no run in this repository has ever produced
   a trained model that beats its own encoder initialisation on the deployment metric. The
   best is ARTBP `geom` at 9.5x worse (C-6). This is the actual deliverable question and it
   has never been answered affirmatively.
2. **Why do the DC and feedback components anti-correlate (C6)?** Nothing isolates it. And
   see the caveat in §8: the D6 shares are not separately identified.
3. **Is the offset a systematic gradient or a random walk?** The newest framing says a cheap
   falsifier exists in `f07.npz` and has not been run. **I ran it.** Across the ten stored
   checkpoints the dY row mean is negative on 9 and positive on 1, and the Y row mean is
   positive on 9 and negative on 1. **The single dissenting checkpoint is 71167, the one
   production run that exhibits the 127x failure**, which flips sign on both rows. The dX
   and X rows scatter 5/5 and 6/4. Two things follow: the sign is not seed-random, and the
   deployed failure does not share the sign of the nine checkpoints the "universality"
   figure is built from. Caveat: those ten differ in `nf` and warm-start, not in seed, so
   this is not the seed test the framing asked for.
4. **Whether the failure is distinguishable from exposure bias.** Stated as an open item in
   the newest framing; nothing measured separates them.
5. **Does the cancellation survive a real residual?** Requires repeating D6 on the Coulomb
   rig.
6. **Why does the baseline's own `max|lambda|` sit at `1.00016`?** Surfaced by
   `pole_horizon_diag.npz`; logged nowhere.
7. **The orthogonal-projection route, which is the thesis's stated scientific contribution,
   has three failed gates that were never followed up.** In
   `simulations/gantry_subnet/diagnostics/orth_projection/`:
   `step7b_result.json` has `pass_all: false` with a maximum principal angle of
   **56.66 degrees against a 5 degree tolerance** (the penalty basis is not stable to a
   10 percent parameter detune); `step8b_result.json` has `pass_all: false` at
   **38.19 degrees**; `harness_check.json` has `pass_all: false`; `step7c_result.json`
   returns verdict `"MIXED"`. Steps 0 to 6 and step 8 all pass cleanly. Separately,
   `standalone_checkpoint/standalone_...json` records `nrms_hat` **bit-identical to
   `nrms_init`** on both val and test, i.e. that run did not move the model at all. None of
   this appears in any framing document, in `decisions.md`, or in the run table. If the
   deliverable is the projection, the basis-stability failure at 56.66 degrees is the first
   thing that needs an answer.
8. **Is the MSD-augmented system observable?** `diagnostics/system_dynamics.json` records
   `baseline.obs_rank = 6` of 6 but `msd.obs_rank = 3` of 8. Taken at face value that is a
   structural identifiability problem sitting underneath the entire augmentation programme.
   It may be a numerical-rank artefact of the MATLAB `obsv` call at the 421 Hz absorber pole
   and I did not verify it. Either way it is unexamined and cheap to check.

**Merely unfinished** (the design exists and the answer is probably reachable):

* D7 to 400 steps: crashed, needs a rerun with the indexing bug fixed.
* D9 on MSD: needs the optimizer-creation bug fixed and the routing corrected to `(0..7)`.
* ARTBP gate-2 at 3 or 5 seeds: 1 seed each exists, the harness works, the runs cost about
  6 hours apiece.
* `datasilent-friction-sim` steps 3a and 3b: never built. Multiple documents correctly
  identify the injected-friction sim as a prerequisite for validating anything in the
  DC-handling family, because on the current testbed the correct DC is genuinely zero.
* `t3a_parity.py`: written, never run, blank scorecard row.
* MS7 rerun: the script exists, the artifact is corrupt (§8).
* The loss-versus-selector normalisation mismatch (normalised MSE against raw metres):
  flagged as the cheapest open item on 07-26 and still unchecked.

---

### 6b. ADDENDUM (later session, 2026-07-26 evening): the tension that outranks the rest

**Is the DC the failure, or only correlated with it?** C-3 (MS12) finds eight constants
reproduce **112.8%** of the 12 s error, i.e. the DC is sufficient. C-6's `H_max = 3200` arm
lands its endpoint DC **inside** ARTBP's pre-registered `+-3e-7` band while its free-run
drift is still **47.9x**, i.e. the DC was removed and the drift was not. Different models,
so not yet a contradiction -- but **until this is resolved, "the DC is the failure" is not
safe**, and a large part of the 2026-07-26 reasoning rests on it. This is the cheapest
high-value item on the list: both artifacts are on disk.

**Why the DC is acquired at all is now unexplained.** Six candidate mechanisms are measured
false: the loss is blind to it (C-8, A6); the loss rewards it (C-12); it is paired with a
compensator (C-12); it is exposure bias (C-13); it is entangled with the useful fit (C-14);
it is a transient more steps would remove (C-2's dose response -- it GROWS with steps).
Meanwhile the objective strictly prefers the mean-removed network, the loss optimum in that
direction is `b* ~ 6.9e-10` against a parked `~3.5e-08` (51x, C-8), and the required
parameter change is ~`1e-6` against an Adam budget of ~`5.2e-4`.

**A continuation run was left running.** `pysynth-data/b0_continue_training.py` resumes
`gantry_drift_71167_last.pth` for ~1000 further steps, tracking `||ANN mean||` per epoch and
the 12 s free run every 5. Its readings are pre-declared in the script. Note the session that
launched it expected it to REFUTE its own non-convergence hypothesis, because C-2's dose
response has the failure growing with steps.

---

### 6c. Document status, so the next reader does not re-inherit a dead framing

| document | status |
|---|---|
| `docs/results-log-2026-07-26.md` | **STANDS.** Numbers only, no interpretation, by design |
| `docs/dc-accumulation-research-brief-2026-07-26.md` | **STANDS in framing**, but its section 3 anti-scope wrongly excludes ARTBP -- see 4c |
| `docs/narrowband-objective-problem-2026-07-26.md` section 5 | **VOID.** Superseded the same day |
| `docs/flat-direction-problem-2026-07-26.md` sections 2-3 | **VOID.** Links 3 and 4 both measured non-probative or false |
| `docs/multiple-shooting-sweep-2026-07-25.md` | stands; its own critique ("derive before writing code") was correct and was ignored |

**Literature status.** Three sweeps exist:
`narrowband-literature-sweep`, `flat-direction-literature-sweep`, and
`dc-accumulation-literature-sweep` (the newest, and the only one framed on the refutation
table). The newest contributes two constructive items the project does not hold:
**level-set teleportation** (Mishkin, Bietti, Gower, AISTATS 2025) which directly targets
C-14's "a descent direction exists that Adam does not take", and a **panel-econometrics
theorem** (Liao, Mei, Shi, `arXiv:2410.09825`) predicting that at a local unit root the
estimator distortion loses its horizon dependence entirely -- which retrodicts the `nf`
800-3200 sweep finding the offset at every horizon.

**Caveat capping all three:** Google Scholar returned empty to every agent in two of the
sweeps, including control queries. Scholar is the only full-text-indexed route, so **every
novelty and gap claim in all three is titles-and-abstracts only and is provisional.**
One open check: the Liao mapping requires a per-window fitted nuisance constant, and the
minimal reproduction (C-15) has none -- true-state init -- yet still fails.

---

## 7. What is ruled out, with the evidence and its grade

Do not re-propose these.

| candidate | grade of the killing evidence | evidence |
|---|---|---|
| Longer fixed BPTT windows | **SOLID** | SLURM 71013, 4 window lengths cold, 9/9 checkpoints carry the DC at `~1/nf`, every free run worse than the `8.0e-05` init, saturating at 1.3x. Independently, NF=900 on the Coulomb rig diverged |
| Excitation / optimal input design | **SOLID** | D3, 3 seeds: 3030x more Y traversal leaves the parked constant 1.32x *larger*. D1: the loss determines the constant |
| Better initialisation of the encoder | **SOLID** | SLURM 70558 (bias killed, drift signature identical) plus v3x0 (true init keeps 85% of the DC), two independent interventions |
| Input band / broadband excitation | **SOLID** | v3joint, 3 seeds, DC identical to narrowband to two figures |
| A code or data offset as the source | **SOLID** | v8 self and matlab arms clean, plus a size-matched injection-recovery control proving 86 to 95 percent detection efficiency at the disputed size and a `~3e-7` detection limit. The best-designed negative result in the repository |
| Lipschitz / spectral caps | **SINGLE** | v6/v6b: no proposed cap binds, and the destabilisation happens at `~1e-5` gain. Verdict follows structurally, but from one seed |
| Velocity-only routing | **SINGLE** | clean null, `GV_ROUTE=3,4,5`: Y drift 2.4x worse; a velocity-row DC integrates twice |
| Optimiser swap (Adam to SGD) as the deliverable | **SINGLE, and weaker than presented** | `r2_fit_probe.py` TASK 5 (SGD `+0%` learned, still 83x floor drift) is **print-only, no artifact**. The supporting 3-seed SGD DC measurement is n=1 (identical files). The theory argument is sound; the measurement backing is thin |
| Zero-mean / window-mean priors on the velocity rows | **SOLID, with C-7's qualifications** | D4 real-data residual mean, 212k samples. Note up to 56 percent of that mean is attributable to the baseline's own Coulomb term |
| Rank-1 pin or prox on the measured DC direction | **SOLID** | T1b, 3 seeds: crushing `|c|` 24 to 92x fixes Y (0.7 to 0.9x floor) and degrades X on 2 of 3 seeds (0.7 to 12.7x, 2.1 to 34.3x). A per-axis trade, not a win. An aggregate metric would have recorded this as a success |
| In-loss soft penalties under Adam | **SOLID** | T1, 24 units, 3 seeds: `in_loss` converges to a beta-independent plateau over two decades of beta; `prox` is monotone. I5 correctly restated as application-mode-specific |
| The continuity-defect term as an RMS norm | **SINGLE, but a clean pre-registered falsifier** | MS6/MS6b: 1.01x against a 65x failure, flat across `n_seg` |
| Hard class restrictions (passivity, contraction, RENs, bounded impulse) | **SINGLE** | `p1_drift_probe.json` records `"PASS": false`: the strictly-passive arm holds (`env_ratio` 1.008 / 1.000) but the stored-energy arm drifts (`env_ratio` 1.62 / 1.48, `slope_q4 -2.4e-5`). Also excluded by R2 on principle. The artifact has no horizon field and lives outside the folder that "owns" it |
| Anti-damping / stabilisation as the *fix family* for the deliverable failure | **SINGLE, three methods** | MS11 plus `growth_aug_vs_base` plus `pole_horizon_diag`. I3 is real and well measured; it is not what the 12 s failure needs |
| ARTBP | **NOT RULED OUT.** Listed as ruled out in three documents; the evidence contradicts them | See C-5 and C-6. The theoretical argument (Beatson and Adams Thm 4.1 at `\|lambda\|=1`) is about unbounded variance, and the variance blow-up **is** measured, at `H_max >= 3200`. At `H_max = 1600` the measured result is a 4 to 6x drift reduction with a matched control. The correct statement is "ARTBP is a measured 5 to 10x mitigation with a hard `H_max` ceiling, not a fix", not "ruled out" |

---

## 8. Assessment of the evidence base

**How much is single-seed.** Almost all of the recent, load-bearing work. Every MS result
(MS1 to MS12) is 1 seed and mostly 1 record. Every `baseline-null` result is seed 0. The
ARTBP gate-2 runs are 1 seed each. The only genuinely replicated results are the frozen-rig
campaign (D1 to D6, T0.1b to T1b: 3 seeds) and ARTBP Phase D (5 seeds). Two nominal 3-seed
results collapse on inspection: **D1's three seeds are bit-identical** because the
evaluation point is deterministic (the file says so; the doc still grades it SOLID), and
the **three SGD seed files in `gantry-zero-mean/data/` are numerically identical in every
array**, so the "2000x less DC under SGD" result is n = 1.

**How much is oracle-dependent.** Less than feared, and it is well flagged. MS7's headline
`13.80x` uses true states; its deployable figure is `1.50x` and the documents say so. MS1's
`e_roll` scale is measured at initialisation with true states. The `v9` oracle-floor
attribution uses the matched 8-state oracle by design. No threshold anywhere is
oracle-derived, which is the thing that would have been fatal.

**Load-bearing claims I could not verify.**

1. **MS7, the "route that survives MS6", has no readable artifact.**
   `pysynth-data/results/MS7_coherent_defect.json` is 184 bytes and truncates mid-value at
   `"rms_init":`. It does not parse. No log, no figure, no rerun. Every number in the MS7
   row of the run table, in `results-log` §7, and in `ann-worse-than-init` M7 (the
   `2.34 / 4.34 / 13.80x` coherent ratios, the `0.858 / 0.831 / 0.834` coherence column,
   the `1.06 / 1.21 / 1.50x` encoder-node figures) is **ASSERTED**. This matters because
   MS7 is the basis of the "two-part programme" that the problem log names as the surviving
   route out of the multiple-shooting work, and of the claim that fixing the encoder buys
   `9x` more discriminative power. Rerunning `check_coherent_defect.py` costs minutes.
2. **The `7.86e-05 at 2 s` figure has no source.** It appears exactly once in the entire
   repository, in `results-log-2026-07-26.md` line 26, as the illustration of the horizon
   trap. Its 12 s partner (`1.66e-04`) I verified. The 2 s value is ASSERTED, as is MS2's
   2 s arm (C-9), which is the other half of the same warning.
3. **The R2/R4 single-knob tradeoff has no artifact.** `baseline-null/r2_fit_probe.py`
   (TASK 5) and `pole_check.py` (TASK 3, the R3 `|lambda| = 1` gate) contain no `savez`, no
   `savefig` and no `json.dump`. Both are print-only. The R3 pass is quoted as established
   fact in another campaign's scorecard; the R2/R4 tradeoff is the stated reason the
   estimator route was deprioritised and the structural split was declared mandatory. Both
   are ASSERTED.
4. **The entire Route B evidence base is prose.** All three scripts in
   `scripts/gantry/datasilent-friction-sim/` write nothing at all. The step-1 numbers quoted
   in the run table (`+41%` learned, `532x` residual, Y deviation `1.55` to `1.47`,
   `env_ratio 1.00`) and the step-2 subspace numbers (`2502x`, error `1e-15`) exist only in
   `PROGRESS.md`. Whether `step4_orth_projection_null.py` was ever run is unrecoverable.
5. **`scripts/gantry/encoder_initialisation/` is a total artifact void**: 11 scripts, one
   surviving output anywhere in the tree.
6. **The `scripts/gantry/verification/` folder retains almost nothing.** 23 scripts, of
   which 14 write no output at all. Every interconnect, normalisation, LFR-residual,
   one-step and decimation verification result exists only in lost console output. The one
   numeric survivor is `memory_diagnostic_data.json`. This is the layer the whole Python
   pipeline's correctness rests on.
7. **The Coulomb parameter-recovery run was never launched.** `COULOMB_HANDOFF.md`
   (2026-07-19) documents a complete, verified build (LFR versus direct EOM agreeing to
   `3.6e-15`, MATLAB parity to `2.8e-18 m`) and states the recovery run is pending. Its
   output directory `simulations/pr_telica_split/<run_id>/` does not exist. The three
   verification numbers quoted are themselves print-only. Note that D4, and therefore C-7,
   uses a `71447` parameter set from that pipeline, so the run does exist somewhere; the
   folder that would let anyone reproduce it does not.
8. **D6's attribution shares are not separately identified.** `D6_summary.json` records
   component correlations of `-0.9999996` and shares of `+1.679 / -0.679` on seed 0 and
   `-32.351 / +33.351` on seed 1. Two components that sum to 1 by construction while each
   ranges over ±30 do not carry an attribution. The `drift-conclusions` C6 caveat notes the
   correlation is near-automatic on a `K=0` axis; the *magnitude ratio* it offers instead is
   drawn from the same degenerate decomposition. C6 is graded "SOLID, the most consequential
   single result of the campaign". I would grade it INFERRED.

**Two comparisons that are silently not like-for-like.** `ARTBP/data/self_scheduling_zeroinit.npz`
stores `H_free = 800` while the checkpoint file it is compared against stores `H_free =
47000`, under the same key name. `efolding_zeroinit.npz` uses `hsweep = [8000..32000]`
against the checkpoint's `[12000..47000]`. The `test_efolding` "exponential wins, pole
`1.00006` to `1.00009`" result that `drift-critical-analysis` §2.9 lists as an open
contradiction rests on that pair. It is not a contradiction; it is a horizon mismatch, and
MS11 plus `growth_aug_vs_base` plus `pole_horizon_diag` all say polynomial.

**What the record does exceptionally well.** Pre-registration is real here, not decorative:
hypotheses are written before launch, falsifiers are named, and several pre-registered
falsifiers actually fired and were honoured (MS6, SLURM 70558, D3, v8). Self-corrections are
made in place rather than buried, and at least six are logged against the author's own prior
claims. `D4_telica_residual.json` carries an explicit frame string with units and a
provenance note; `drift-visual/data/manifest.json` records source and shapes per file;
`drift-visual/data/f09.npz` stores its horizons as named fields. Those three are the model
the rest should follow.

**What it does badly, and the single change that would help most.** Numbers reach documents
before they reach disk. Of the twelve MS results, one artifact is corrupt, one holds a
different horizon than the claim built on it, and one contains a completed control that the
document written afterwards says is still pending. Whole campaigns (Route B, the R2/R4
tradeoff, the R3 pole gate) rest on `print`. **Every script that produces a number a
document will cite must write that number, with its horizon and its units, to a file.** The
project already knows this: `drift-visual` was built as a rewrite of `drift-demo` for
exactly this reason, and its manifest is the best artifact here. The practice did not
propagate.

**My overall read.** The *what* is settled and well evidenced: a constant force error of
17 to 23 mN on double-integrator axes, sufficient on its own to reproduce the 12 s failure,
invisible to a 0.1 s objective at a 14400x sensitivity discount. C-1 through C-4 would
survive an adversarial review. The *why* is weaker than the newest document presents: its
two mechanism links are on a different rig, and one of them re-asserts a claim the project's
own better-graded measurement refutes. The *what to do* is the weakest part, and it has one
concrete blocker that is not a research problem: the best-measured intervention in the
repository is filed under "ruled out" on the strength of a theoretical argument, while five
converged runs with a matched control sit unreported on disk. Reading those five files
should precede designing anything new.

**The uncomfortable one.** Seventeen days of work have gone into diagnosing why the
augmentation degrades the initialisation. Over the same period, three artifacts recorded
that the augmentation's two latent states never learned the absorber at all
(`r2_aug_lin` about `0.35`, C-1b), that most of the encoder receives exactly zero gradient,
and that the projection route which is the thesis's actual contribution has an unaddressed
basis-stability failure at 56.66 degrees. The drift is real and worth understanding. But it
is possible that the reason nothing beats epoch 0 is not only that training pushes the model
the wrong way, but that at this routing and step budget the model was never learning the
thing it exists to learn. Nothing in the record tests those two against each other, and it
would be cheap to: the numbers are already on disk.

---

### 8b. ADDENDUM (later session, 2026-07-26 evening)

**Everything added in 2b, 4b, 5b and 6b is 1 seed**, below this project's 3-seed floor,
with one exception: C-8's curvature is now confirmed on two independent rigs (0.3% on X).

Two entries carry weaker samples than their prominence suggests. C-12's "the objective
penalises the DC" uses **40 windows of one training record**, and C-14's "not entangled"
uses **12 windows of one validation record**, both against a 6664-window training bank.
They agree with each other and with the two-rig curvature, but neither is a
full-distribution statement.

C-15's minimal reproduction is the most useful new asset and the most easily over-read: it
reproduces the failure's SHAPE at `1.713x` against the production path's `58.9x`. **No
absolute number transfers between them.** Its value is speed -- minutes per arm against
hours -- which makes it a screen, not evidence.

Finally, the session that produced these additions killed **three of its own hypotheses**
(flat direction, pairing, the normalisation link) and had **four runs voided by their own
controls**. That is the intended behaviour of pre-registration rather than a failure rate,
but it should calibrate how much weight any single-session conclusion here carries.

---

## 9. Provenance

Artifacts I opened and checked directly for this document:
`gantry_drift_71167_last.pth`, `gantry_drift_last.pth`;
`pysynth-data/results/{MS1,MS2,MS3*,MS6,MS6b,MS7,MS10,MS11,MS12}*.json`;
`drift-diagnostics/data/{D1_summary,D1_zeroinit_2d_seed*,D1_zeroinit_3d_seed*,D2_summary,D3_summary,D4_telica_residual,D7_summary}.json`;
`drift-diagnostics/logs/{d7,d9}.output`;
`ARTBP/data/train_{fixed,geom,poly4,poly6}_seed{0..4}.npz` and all five `*_ep20*` / `*_ep10*` variants;
`drift-visual/data/f07.npz`; `ARTBP/train_artbp.py` (for the drift-eval horizon).

Documents read in full: the run table §12, `results-log-2026-07-26.md`,
`flat-direction-problem-2026-07-26.md`, `narrowband-objective-problem-2026-07-26.md`,
`drift-conclusions-2026-07-25.md`, `ann-worse-than-init-diagnosis.md`,
`ISSUE-OVERVIEW-FOR-INDEPENDENT-REVIEW.md`, `ms3-decision-table.md`,
`all-five-construction-spec.md`, and `drift-critical-analysis.md` §2.

Folder enumeration for retained output was done by three parallel read-only sweeps over
nineteen `scripts/gantry/` sub-folders, `simulations/gantry_subnet/diagnostics/` and the
71167 run folder. Their findings are reflected in C-1b, C-7, §5, §6, §7 and §8, and were
spot-checked by me for the claims I use them for (`passive-augmentation`,
`datasilent-friction-sim`, `baseline-null/r2_fit_probe.py`,
`gantry-zero-mean/data/v3x0sgd_*`, `orth_projection/step7b`, `71167/*.npz`,
`msd_transfer_diagnostics/data/diag_msd_summary.json`).

**Everything here is read-only.** No training was run and no file outside this document was
modified.

---

## 10. ADDENDUM, 2026-07-26 night: the dc-accumulation campaign, steps 0-3

Appended, not merged. Nothing above this line was edited. Same grading rules.

### C-17. The C-6 ARTBP table reproduces; the framing around it does not. VERIFIED, grade SINGLE.

`dc-accumulation/step3_recompute_artbp.py`, read-only over
`scripts/gantry/ARTBP/data/*.npz`. Full table and horizons in
`results-log-2026-07-26.md` §11 and `results/step3_artbp_recompute.json`.

The magnitudes C-6 quotes are right. Four things around them are not, and all four are
readable off the same arrays C-6 was built from.

1. **A sixth converged run exists**, `poly6 h1600 b256 ep8` in `ARTBP/data/72659/`, in no
   document including C-6. It is the **only** arm carrying both a `_best.pt` and a `_last.pt`
   checkpoint, which is what made step 0 possible at all. Best trained epoch 12 s
   `1.9716e-03`, drift `12.7x`, held-out 0.1 s nf-RMS `5.9475e-05`.
2. **Three arms store no epoch-0 row.** `len(val_sim_traj) == epochs` for `fixed/geom/poly6
   h1600 ep20`, against `epochs + 1` for the rest; they pre-date `train_artbp.py:199-200`.
   Every "`x` epoch-0" ratio C-6 quotes for those three arms is **borrowed from a different
   run**, which C-6 does not say.
3. **The borrowed reference is not arm-invariant.** `h3200` and `h6400` store
   `1.846056547947228e-04` bit-identical, but the `ep8 h1600` arm stores
   `2.239589812234044e-04`, because it ran `val_n = 2` against the others' 4. That is a
   different validation set, so the two are different metrics rather than a disagreement —
   but it is a **21% swing** on every G1 ratio, decided by a config field the three older
   files do not record at all. Step 0's control resolves which applies (below).
4. **`best_val_sim` and `drift_ratio` describe different models, in all six arms.** The drift
   eval runs after the training loop on the live final weights (`train_artbp.py:251-291`);
   `best_val_sim` is a per-epoch minimum. Best trained epoch vs epoch the drift was measured
   on: `19/20, 5/20, 7/20, 7/8, 6/10, 14/20`. C-6 and `step3_artbp_benchmark.md` place both
   in one row as though one model produced them. **`geom`'s headline pairing — "best 12 s
   `1.7588e-03`, drift `22.1x`" — is a 5th-epoch model quoted next to a 20th-epoch model.**

Also: the ANN-off drift denominator is not a fixed reference (`_roll(..., off=True)` keeps
each arm's own trained encoder), though it empirically takes only 2 distinct values, 2.8%
apart. And the per-record spread inside the drift ratio is large — `ep8`'s Y ratios are
`1.584 / 32.980 / 3.509` on T1/T3/T5 — so a 3-record mean of a 20x-spread quantity is
carrying more weight in these tables than it can bear.

**The bar to beat, restated on a single model.** Taking the final-model triple, which is the
only internally consistent one (`dc_endpoint`, `val_sim_last` and `drift_ratio` are all the
endpoint): `geom h1600` = 12 s `5.8958e-03`, drift `22.1x`; `poly6 h1600 ep20` = 12 s
`3.9420e-03`, drift `13.5x`. On the best-epoch basis `geom` is `1.7588e-03`. The two bases
disagree by 3.4x on `geom`, so **which basis a future candidate is scored on must be fixed
before it is run, not after.**

None of this overturns C-6's conclusion. ARTBP at `H_max = 1600` really does cut the
first-epoch collapse by ~10x and the drift by 4-6x at ~2% windowed cost, and it really is
still 9-11x epoch 0. It remains the benchmark and it remains not a fix. The anti-scope in
`dc-accumulation-research-brief-2026-07-26.md` §3 has been corrected in place.

### C-18. The minimal testbed ports faithfully. Control only, grade SINGLE.

`dc-accumulation/step1_testbed.py --configs C0 --seeds 0` reproduces C-15 exactly: free-run
harm `1.713x`, windowed `0.943`, ANN output mean `+1.1500e-05`. 0.0% deviation.

Recorded because the first port did **not**: it returned `4.871x`. `make_data` ends with
`u = u - u.mean()`, so lengthening the data array to make room for encoder history shifts
the input by a constant, and on a `K = 0` plant that is not a small perturbation. This is
the same class of harness confound that voided MS4, and it was caught only because the
control was pinned to a published number rather than to a plausible one.

### C-19. STEP 0 GATE ANSWERED: the DC is the failure, not a correlate. VERIFIED, grade SINGLE.

`dc-accumulation/step0_dc_sufficiency.py`, `results/step0_dc_sufficiency.json`. Numbers in
`results-log-2026-07-26.md` §13. MS12's four-arm decomposition, re-run on three ARTBP
checkpoints on the rig that trained them, one record (V1), one horizon (12 s).

**Control passed both legs** (0.064% and 0.133%), so the rows are readable, and it
incidentally settles C-17 point 3: the two stored epoch-0 values are two different metrics
(4 val records vs 2), not a disagreement.

On all three ARTBP arms the per-row mean alone reproduces **98.7-99.5%** of the 12 s error,
and removing it collapses the model to **0.868-1.264x** of its own ANN-OFF floor. This is
**cleaner than on 71167**, where MEAN-ONLY was 112.8% and MEAN-REMOVED still 37.2% and 22x
floor. C-3 generalises. The 6b tension is closed and step 4's DC-targeting candidates are
correctly aimed.

**My pre-declared discriminator was refuted by its own test, and I am recording that rather
than quietly substituting the analysis that worked.** Reading (c) predicted the ARTBP
`Z_pts` dY operator and the MS12 along-trajectory operator would ORDER the checkpoints
differently. They do not: both rank `h3200 ep6 < h1600 ep7 < h1600 ep8`.

**What does resolve C-6's low-DC/high-drift arm is a magnitude decomposition, computed after
the fact and therefore graded as post-hoc, not pre-registered.** The `dY` row carries only
**0.77% / 0.95% / 1.76%** of `||traj mean||` on the three arms; the DC lives on `aug0` and
`aug1` (`h3200 ep6`: `aug0 +2.408e-05`, `aug1 +1.292e-05` against `dY -4.883e-07`). On 71167
the same figure is 0.63%. ARTBP's `dc` (`train_artbp.py:226`) and its pre-registered `±3e-7`
band (`ARTBP/README.md:136`) watch **one row of eight, carrying 1-2% of the DC norm.** So
C-6's "the DC was removed and the drift was not" is not supported by its own artifact: only
the `dY` component was ever measured, and the other 98% was never in the band's scope. The
two claims were never in contradiction.

Consequence for any future DC work: **`dc_endpoint` in the ARTBP `.npz` files is not a
measure of "the DC"**, and no arm should be scored on it alone.

### C-20. A trained model that beats its own initialisation, once its DC is removed. VERIFIED, grade SINGLE, and NOT yet deployable.

Incidental to C-19 and it outranks the gate question. §6 item 1 states that after seventeen
days "no run in this repository has ever produced a trained model that beats its own encoder
initialisation on the deployment metric."

`poly6 h1600 ep8 LAST` (epoch 8), MEAN-REMOVED, record V1, **12 s**: `1.038413e-04 m`
against an ANN-OFF floor of `1.196380e-04 m` = **`0.868x`**, i.e. 13 percent better than the
initialisation. The same model at FULL is `4.190942e-03` = `35.03x` floor.

So the ARTBP-trained network **has** learned something useful on the deployment metric. The
entire net-destructive effect is the constant. That is a much stronger form of C-14
("weight space can express same-shape-less-mean") and of C-12 ("the DC hurts both
objectives"): here the useful part is not merely present, it is worth 13 percent against the
init.

**Three limits, none of them small.**
1. The removed constant was computed **along the same V1 free run it is then scored on**. This
   is output-space surgery with a constant fitted on the evaluation record, so it is not a
   deployable model. C-13 (`A8`) measured the free-run and training-window DC agreeing to
   0.98-1.00, which suggests a train-computed constant would land close, but that is an
   inference, not this measurement.
2. 1 seed, 1 record, one checkpoint. Below the project floor.
3. It is `_last`, not `_best`; the `_best` (epoch 7) arm gives `1.264x`, i.e. still above the
   floor. The 0.868x is one epoch of one arm.

**The cheap run that would settle it:** recompute the per-row mean on the TRAINING windows,
subtract that fixed constant, and re-score the 12 s free run on V1-V4. If it holds, the
deliverable question that §6 item 1 calls "never answered affirmatively" has a first
affirmative answer, and the fix family narrows to "estimate and subtract the constant" rather
than "prevent it forming". If it does not hold, the 0.868x was fitted-on-the-test-record and
must be withdrawn.
