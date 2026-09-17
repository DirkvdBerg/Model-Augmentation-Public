# OBC state-space implementation: running narrative

Implementation of Gyorok's orthogonal-by-construction (OBC) augmentation for the state-space
gantry, per `tasks/handoffs/2026-09-15-obc-state-space-implementation.md`. One entry per stage:
what it measured, whether the number met its criterion, what changed in the plan as a result.

Settings common to every stage, matching the production pipeline
(`scripts/gantry/gantry_dynamic/config.py`): `fs_new = 4000` so `Ts = 2.5e-4 s`,
`up_sample = 2`, `Y_op = None` (LPV self-scheduling), `nx_ann = 2`, routing `(0..7)`,
`nf = 400`. Identity state/input normalisation (`std_x = 1`, `x_mean = 0`) so every number is
in physical units. float64, CPU throughout.

## Stage 0 -- per-timestep parameter Jacobian cost -- 00-cost/

**Two clarifications from the user, received during this stage.** Both were already the choice
made here, so nothing was switched: "representative excitation" is the designed point set
(`design_points(n, seed)`, copied verbatim into `implementation/common.py` from
`investigation-20260915/obc_common.py:196`), `n = 256` headline with a `32 / 64 / 256 / 1024`
sweep, no dataset loaded anywhere in stages 0 to 5; and stage 1 adds a new subclass BESIDE
`Parameterized_Gantry_State_Block` rather than reparameterising it in place.

**Measured.** All at the pipeline's settings, float64, CPU, torch 2.5.1, 6 threads.

| | batch 256 (production) | batch 512 (handoff's ask) |
|-|-|-|
| plain block step | 0.004873 s | 0.005625 s |
| `vmap(jacfwd(f_single))`, 10 reduced dirs | - | 0.122 s, **28.2x** |
| `vmap(jacfwd(f_single))`, 14 raw dirs | - | 0.114 s, **26.2x** |
| `jacfwd(f_batched)`, 10 reduced dirs | 0.0742 s, 15.2x | 0.0803 s, 14.3x |
| 10 separate `jvp(f_batched)` calls | - | 0.463 s, 88.5x |
| `jvp(f_batched)`, ONE direction | 0.0440 s, 9.0x | 0.0477 s, 8.5x |
| plain step forward + backward | 0.01775 s | 0.01437 s |
| one-direction jvp with grad-carrying tangent, + backward | 0.0566 s, **3.19x** | 0.0602 s, 4.19x |

Value checks before any timing was believed: the gauge section round-trips exactly
(`max abs (combos(section(v)) - v) = 0.000e+00`); the reduced free coordinate at zero reproduces
the raw step to `5.4e-20` against a state-increment scale of `9.2e-03`; `jacfwd(f_batched)` and
ten `jvp` calls reproduce `vmap(jacfwd(f_single))` to `2.1e-16` relative; and
`jvp(f, v0, a) == J @ a` to `4.6e-16` relative.

**Criterion.** Under roughly 2x take candidate (a); at 10x the analytic Jacobian (b) becomes a
prerequisite. The literal quantity the handoff named, `vmap(jacfwd)` over ten directions at batch
512, came in at **28.2x**: NOT MET, by an order of magnitude.

**What the number actually says, and the plan change it forces.** Three findings, in the order
they were established.

1. *Ten directions cost the same as fourteen* (28.2x against 26.2x). A cost proportional to the
   tangent count cannot do that, so the cost is a fixed per-call charge. Confirmed by the batch
   sweep: the jvp-to-step ratio FALLS from 13.1x at batch 64 to 8.8x at batch 2048, which is the
   signature of per-OPERATION overhead rather than per-element work. The block rebuilds the whole
   `M(Y)` rational structure on every call, some fifty tiny operations on 3x3 blocks, and then
   runs eight `deriv` evaluations; under a `torch.func` transform every one of those is wrapped.
2. *The per-sample `vmap` composition is the wrong shape inside a rollout.* Keeping the batch
   intact (`jacfwd` over the batched step) halves the cost, 28.2x to 14.3x, for a matrix that
   agrees to `2.1e-16` relative. `gantry_orth_matrix` is right to use the per-sample form, because
   it builds the regressor ONCE on a fixed point set where 28x of a cheap thing is irrelevant.
3. **The handoff's candidate (a) prices the wrong object, and so did stages 0 and 0b.** The
   rollout does not need `Phi(z_k)`. Read the correction term: `g~(z_k) = g(z_k) - [J(z_k) |
   c(z_k)] a` with `a` fixed for the pass, so what is applied is
   `Delta(z_k) = J(z_k) a[0:10] + c(z_k) a[10]`. The first term is a single Jacobian-vector
   product in the direction `a[0:10]`: ONE tangent, not ten. The second, in the reduced/log
   coordinate, is `f_vartheta_bar(z_k)`, the nominal baseline transition, which the rollout has
   ALREADY computed at that step, so it is free. The ten-column Jacobian is needed only on the
   construction evaluations, where it is built once per epoch under `no_grad` and costs 0.040 s
   at `M = 256`. This is an exact algebraic identity, verified to `4.6e-16` relative, not an
   approximation.

**Verdict, on the quantity that governs training.** At the production batch of 256 a training step
(forward AND backward) costs **3.19x** more with the subtraction than without: 7.10 s to 22.65 s
for one `nf = 400` window. That is inside the handoff's 10x abort line, so candidate (a) proceeds.
The forward-only ratio is 9.0x, but a training step is not forward-only and the backward pass is
where the plain baseline spends most of its time, which is why the two ratios differ by 3x.

**Plan changes.**
- The rollout applies a DIRECTIONAL derivative, `jvp` in the direction `a[0:10]`, not the
  ten-column Jacobian. Stage 4's wrapper is written that way. The full Jacobian appears only in
  the once-per-epoch basis build.
- The basis build uses `jacfwd` over the BATCHED step, not `vmap(jacfwd(f_single))`. It is the
  same matrix to `2.1e-16` relative at half the cost. `gyorok_orth_matrix` is left untouched and
  is still the right implementation for its own fixed-point-set use; this is a different call
  shape for a different use, not a second Jacobian implementation.
- The analytic forward-sensitivity route (candidate (b)) is NOT built. It is recorded as the
  identified optimisation: it attacks exactly the per-operation dispatch overhead that the batch
  sweep isolated, and would propagate one sensitivity direction through the RK4 stages alongside
  the state at an estimated cost near 2x. Estimated, NOT measured; nothing downstream depends on
  it.

## Stage 1 -- reduced ten-combination block -- 01-reduced/

**Built.** `Reduced_Gantry_State_Block` in `model_augmentation/fit_systems/blocks.py`, `@added`,
a subclass BESIDE `Parameterized_Gantry_State_Block` per the user's clarification. Its trainable
parameter is a ten-vector `free_params`; the parent's fourteen-dimensional `log_params` is deleted
rather than shadowed, so no checkpoint carries fourteen untrained tensors and a
`functional_call(blk, {'log_params': ...})` cannot silently do nothing. Nine combinations are in
log, `m_diff` in a relative linear coordinate because it is signed. One `# CHANGED:` marker in the
parent: `RMSE_baseline` is now kept as a plain float, because the only previous record of it was
the already-divided `Lambda` buffer and the reduced block needs to rebuild its own against
`combo_init`.

**Measured.** float64, designed points, `Y` swept over `[-0.30, 0.30]`.

| test | number |
|-|-|
| T1 tensor combination map vs `_combos_from_raw` | `0.000e+00` |
| T2 gauge section is a section, `combos(section(v)) == v` | `0.000e+00` nominal, `2.2e-15` relative when displaced |
| **T3 reduced vs raw transition, 256 points (THE CRITERION)** | **`0.000e+00`**, bit-identical |
| T3 per-`Y`, all 24 designed sweep points | `0.000e+00` at every one |
| T4 at displaced combinations, 1 / 5 / 20 / 50 percent | `6.9e-15`, `0.0`, `0.0`, `1.0e-15` relative |
| T5 gauge invariance, four alternative gauges | `0.000e+00` at every one |
| T6 admissibility, nominal and plus/minus 20 percent | `min eig M(Y)` `3.7363` / `4.5292` / `3.0748` kg, `min abs d(Y)` `2.06e+03` / `3.76e+03` / `1.13e+03`, admissible throughout |

**Criterion.** Relative difference below `1e-12` over at least 24 designed points spanning the
scheduling range. Measured `0.000e+00`. **MET**, and by more than the criterion asks: the two
transitions are bit-identical, not merely agreeing to round-off.

T5 is the test that earns the gauge its name. Four alternative gauges (`mb` at 5.0, at 40.0, the
inertia split at 1:1 and at 1:9 with `mb = 1.0`) give the same ten combinations to `0.0` and the
same transition to `0.0`. The arbitrary split does not enter the model. `min eig M(Y) = 3.7363 kg`
reproduces the investigation's `3.736`.

**One defect found and fixed, which is why the first run failed at `4.1e-09` instead of `1e-12`.**
`combo_init` was formed at float32 and cast to float64 afterwards, because the blocks were
constructed at the gantry_ss default dtype and cast with `.to(F64)` only after `__init__` had run.
`m_total` and `J_eff` are sums of same-order quantities, so forming them at float32 costs about
`1e-7` relative on `m1`, `m2`, `Jb` and `Jh` after the gauge round-trip, which reached the velocity
rows as a `2.3e-11` absolute difference: a thousand times the criterion, and entirely invisible in
the positions (`2.3e-15`). Two changes: the library now forms the combination map in float64
regardless of the block dtype, and `common.make_raw_block` / `make_reduced_block` build from a
float64 `params_init` so the block is in float64 from construction rather than cast into it. The
diagnosis is worth keeping because it is generic: any block whose derived quantities are computed
in `__init__` is silently pinned to the construction dtype, and `.to()` afterwards cannot recover
the bits.

**Plan change:** none to the ladder. `common.gauge_section` is now a thin alias for the library's
`Reduced_Gantry_State_Block.gauge_section`, so there is exactly one implementation; stage 0 kept
its name and its numbers are unaffected (timings do not depend on parameter values).

## Stage 2 -- empirical rank of the reduced stacked Jacobian -- 02-rank/

**Built.** `model_augmentation/fit_systems/obc_projection.py`, `__project_origin__ = "added"`:
`stacked_basis` (the `[J | c]` build via `jacfwd` over the batched step), `directional` (the one
jvp the rollout applies), `rank_report`, `principal_angles`, `RankLossAbort` and the `OBCBasis`
frozen-per-epoch object with its column-scaled pseudo-inverse, `coefficient`, `rho`,
`residual_ratio` and `spectrum`. The wrapper block comes in stage 4.

**Measured.** Designed point set, `Y` swept over `[-0.30, 0.30]`, `M = 256` headline, float64.

| | |
|-|-|
| `rank(J)`, 10 columns, at rtol `1e-4` / `1e-6` / `1e-8` / `1e-10` / `1e-12` / `1e-14` | **10 at every one** |
| condition number of the column-scaled `J` | **4.50** |
| smallest singular value of `J` | `2.221e-01` of `sigma_0`, i.e. `1.01e+15` x float64 epsilon |
| `rank([J abs c])`, 11 columns, same rtol band | **11 at every one** |
| condition number of the column-scaled `[J abs c]` | 6.59 |
| fraction of the offset column outside `span(J)` | **0.584** |
| rank vs point count `M = 32 / 64 / 256 / 1024` | `rank(J) = 10` and `rank([J abs c]) = 11` at every `M`; `cond(J)` 4.94 / 4.75 / 4.50 / 4.57 |
| rank under column scaling (unit-column, all x1e6, one column x1e6) | 10 in every case, condition number unchanged at 4.50 |
| additional-state rows of the basis | `0.000e+00` exactly |

**Criterion.** `rank(J) == 10` at every rtol from `1e-4` to `1e-14`, gap above `1e6`;
`rank([J abs c])` reported separately, expected 11. Rank: **MET**, 10 everywhere and 11 for the
extended basis. The gap clause needs an honest reading rather than a tick: `J` has ten columns and
rank ten, so NOTHING is discarded and there is no internal gap to measure; the reported value is
`inf` because the set of discarded singular values is empty. The number that carries the meaning
the clause was after is the margin to the noise floor, `sigma_min / sigma_0 = 2.22e-01`, which is
`1.0e+15` times float64 epsilon and nine orders above the `1e6` the criterion asks for. Stated
this way rather than claiming a gap that does not exist.

**The offset hazard, reproduced on the reduced basis and quantified.** The offset column's norm is
`141x` the largest parameter column here, against `1030x` on the investigation's raw physical
basis; the reduced/log coordinates shrink the disparity but do not remove it. A cutoff stated
against `sigma_0` of the UNSCALED matrix still collapses the rank: 11 to 8 at `rcond = 1e-4` and 11
to 1 at `1e-2`. Column-scaled, the rank is 11 at every cutoff from `1e-2` to `1e-14`. This is the
evidence for the library's choice to scale columns before the pseudo-inverse rather than to state a
cutoff against `sigma_0` of the raw matrix; the applied correction is unchanged by the scaling,
since `F D (D^-1 a) = F a`.

**The rank-loss policy, implemented and exercised.** Silent truncation is forbidden, so `OBCBasis`
takes an `expected_rank` and an `on_rank_loss` of `report` or `abort`. Forcing a drop by
duplicating a column: rank 11 falls to 10, the discarded singular value is logged
(`6.40e-17` relative to `sigma_0`), the principal angles to the previous epoch's span are logged
(max `2.107e-08 rad`, min `0.0`, i.e. the span is unchanged as a duplicated column should leave
it), the rank loss is named in the log with what it means for the method, and `abort` raises
`RankLossAbort` instead of continuing.

**Plan change:** none.

## Stage 3 -- Taylor order of the protected basis -- 03-taylor/

**Measured.** Displacement in the reduced free coordinate, so `s` reads as "every combination
moved by `s` relative". 256 designed points, float64.

| | |
|-|-|
| slope over 5 random displacement directions | 1.9978, 1.9945, 1.9956, 1.9991, 2.0012 |
| slope over the 10 coordinate directions | 1.9940 to 2.0078 |
| **all 15 slopes** | min 1.9940, max 2.0078, mean 2.0009, **max deviation from 2.0 = 0.0078** |
| round-off floor, measured not assumed | `1.11e-16`; the fit window `1e-3 .. 1e-1` predicts a smallest error near `2.3e-10`, which is `2.1e+06` x the floor |
| neglected term at 1 percent displacement | `2.34e-08`, i.e. `2.9e-06` of the state increment and `2.3e-03` of the linear term |
| neglected term at 10 percent | `2.32e-06`, `2.9e-04` of the increment, `2.2e-02` of the linear term |

**Criterion.** Log-log slope `2.0 +/- 0.1` over a decade. Measured worst deviation `0.0078`.
**MET**, with a factor 13 of margin, in every direction tested including each coordinate
separately.

The fit window was chosen from the MEASURED round-off floor rather than assumed: below `s = 1e-5`
the error stops falling because it has reached float64 round-off, and the window sits six orders
above that. T4 records what the expansion would cost if anyone ever substituted it into the
prediction: at a 5 percent displacement the affine surrogate and the exact nonlinear transition
differ by `5.8e-07`, `7.2e-05` of the state increment, at EVERY step. The rollout keeps the exact
transition; the expansion builds the basis only.

**Plan change:** none.

## Stage 4 -- the projection -- 04-projection/

**Built.** `OBC_Projected_ANN_Block` in `obc_projection.py`: wraps a `Static_ANN_Block`, is
registered in the `Interconnect` in its place, refits the coefficient from the current ANN output
on every forward and subtracts `[J(z) | c(z)] a` pointwise through one directional derivative.
`rebuild_basis(Z_ref)` is the epoch-boundary call; it takes the ANN inputs at the construction
evaluations and carves the physical points out of them, so the two cannot drift apart. The
physical block is held in a plain list rather than as a submodule, so its keys are not duplicated
in every `state_dict` (`parameters()` dedupes by identity, `state_dict` does not).

**Measured.** `M = 256`, production routing `(0..7)`, `n_a = 2`, nonzero ANN weights (the pipeline
initialises the last layer to zero, and a zero write would make every residual trivially zero, so
a projection that removes nothing would not be evidence of anything).

| test | number |
|-|-|
| **T1 scale-free residual, full rank (THE CRITERION)** | **`9.9099e-17`** (prototype: `1.96e-14`) |
| T2a genuine rank deficiency: duplicated column, two duplicated, zeroed column | `7.2e-17`, `1.2e-16`, `6.1e-18` |
| T2b artificial truncation of NONZERO singular values, rcond 0.3 / 0.6 | `2.0e-04` / `1.3e-02`, exactness LOST |
| T3 `g~ = (I - P) g` against an independent QR projector | `3.1e-16` relative |
| T4 additional-state ANN columns `[6, 7]` | changed by `0.000e+00` exactly |
| T4 physical ANN columns `[0..5]` | changed by `1.123e+00`, so the projection is doing something |
| T5 per-sample product, median / min / max | `3.22e-01` / `3.54e-03` / `1.59e+00` |
| T5 stacked sum | `2.93e-14`, i.e. `1.1e+13` times smaller than the median per-sample |
| T6 `rho` on the construction evaluations, before then after | `0.2305` then `0.0000` |
| T6 `rho` on a foreign designed set | `0.3295` then `0.1434`; frozen-coefficient residual `3.6e-02` |
| T6 `rho` on a foreign set at 4x latent scale | `0.3319` then `0.1456`; residual `3.7e-02` |
| T6 `rho` on a narrow-`Y` set | `0.2269` then `0.1009`; residual `4.2e-03` |
| T7 wrapper against the stacked algebra by hand | coefficient `0.000e+00`, output `1.06e-16` relative |
| T7 flag OFF | `torch.equal` TRUE, the wrapped tensor itself |

**Criterion.** Residual at or below `1e-12` at full rank and at deliberately reduced rank. Full
rank `9.9e-17`, genuine reduced rank `7.2e-17`. **MET**, two orders better than the investigation
prototype.

**One correction to the test, which changes what the criterion means.** The first version read
"deliberately reduced rank" as "truncate the pseudo-inverse harder", and at `rcond = 0.3` the
residual rose to `2.0e-04`, apparently failing. That reading is wrong, and the failure is real but
is not the construction's. Truncating a NONZERO singular value leaves exactly that direction in
the residual, so exactness is genuinely lost; the any-rank proposition concerns a GENUINELY
rank-deficient basis, one carrying a zero singular value, where only the uniqueness of the
coefficient is at stake. Tested that way, with a duplicated column, two duplicated columns and a
zeroed column, the residual stays between `6e-18` and `1.2e-16`. Both readings are now reported,
because the second is an independent argument for the column scaling of stage 2: scaling is what
keeps a `sigma_0`-relative cutoff from discarding genuine directions and switching the guarantee
off without saying so.

**The honest off-reference picture.** On the frozen set the construction is exact and `rho` goes to
`0.0000`. Away from it, `rho` falls only from `0.33` to `0.14` and the frozen-coefficient residual
is `3.6e-02`, not `1e-16`. The construction removes part, not all, off its own point set. That is
generalisation failure of a coefficient fitted to a write which is not in the span, not a defect of
the projector: the investigation established that a write genuinely lying in the baseline span is
removed at ANY point set, to `4.9e-14`.

**T5 is the wording gate.** Median per-sample product `3.2e-01` against a stacked sum of
`2.9e-14`, a ratio of `1.1e+13`. Nothing we write may be worded pointwise.

**Plan change:** the hollow interconnect stub in this stage was deleted rather than left asserting
nothing; the wrapper inside a full `Interconnect` is stage 6's business.

## Stage 3 -- Taylor order of the protected basis -- 03-taylor/

**Measured.** Displacement in the reduced free coordinate, so `s` reads as "every combination
moved by `s` relative". 256 designed points, float64.

| | |
|-|-|
| slope over 5 random displacement directions | 1.9978, 1.9945, 1.9956, 1.9991, 2.0012 |
| slope over the 10 coordinate directions | 1.9940 to 2.0078 |
| **all 15 slopes** | min 1.9940, max 2.0078, mean 2.0009, **max deviation from 2.0 = 0.0078** |
| round-off floor, measured not assumed | `1.11e-16`; the fit window `1e-3 .. 1e-1` predicts a smallest error near `2.3e-10`, which is `2.1e+06` x the floor |
| neglected term at 1 percent displacement | `2.34e-08`, i.e. `2.9e-06` of the state increment and `2.3e-03` of the linear term |
| neglected term at 10 percent | `2.32e-06`, `2.9e-04` of the increment, `2.2e-02` of the linear term |

**Criterion.** Log-log slope `2.0 +/- 0.1` over a decade. Measured worst deviation `0.0078`.
**MET**, with a factor 13 of margin, in every direction tested including each coordinate
separately.

The fit window was chosen from the MEASURED round-off floor rather than assumed: below `s = 1e-5`
the error stops falling because it has reached float64 round-off, and the window sits six orders
above that. T4 records what the expansion would cost if anyone ever substituted it into the
prediction: at a 5 percent displacement the affine surrogate and the exact nonlinear transition
differ by `5.8e-07`, `7.2e-05` of the state increment, at EVERY step. The rollout keeps the exact
transition; the expansion builds the basis only.

**Plan change:** none.

## Stage 4 -- the projection -- 04-projection/

**Built.** `OBC_Projected_ANN_Block` in `obc_projection.py`: wraps a `Static_ANN_Block`, is
registered in the `Interconnect` in its place, refits the coefficient from the current ANN output
on every forward and subtracts `[J(z) | c(z)] a` pointwise through one directional derivative.
`rebuild_basis(Z_ref)` is the epoch-boundary call; it takes the ANN inputs at the construction
evaluations and carves the physical points out of them, so the two cannot drift apart. The
physical block is held in a plain list rather than as a submodule, so its keys are not duplicated
in every `state_dict` (`parameters()` dedupes by identity, `state_dict` does not).

**Measured.** `M = 256`, production routing `(0..7)`, `n_a = 2`, nonzero ANN weights (the pipeline
initialises the last layer to zero, and a zero write would make every residual trivially zero, so
a projection that removes nothing would not be evidence of anything).

| test | number |
|-|-|
| **T1 scale-free residual, full rank (THE CRITERION)** | **`9.9099e-17`** (prototype: `1.96e-14`) |
| T2a genuine rank deficiency: duplicated column, two duplicated, zeroed column | `7.2e-17`, `1.2e-16`, `6.1e-18` |
| T2b artificial truncation of NONZERO singular values, rcond 0.3 / 0.6 | `2.0e-04` / `1.3e-02`, exactness LOST |
| T3 `g~ = (I - P) g` against an independent QR projector | `3.1e-16` relative |
| T4 additional-state ANN columns `[6, 7]` | changed by `0.000e+00` exactly |
| T4 physical ANN columns `[0..5]` | changed by `1.123e+00`, so the projection is doing something |
| T5 per-sample product, median / min / max | `3.22e-01` / `3.54e-03` / `1.59e+00` |
| T5 stacked sum | `2.93e-14`, i.e. `1.1e+13` times smaller than the median per-sample |
| T6 `rho` on the construction evaluations, before then after | `0.2305` then `0.0000` |
| T6 `rho` on a foreign designed set | `0.3295` then `0.1434`; frozen-coefficient residual `3.6e-02` |
| T6 `rho` on a foreign set at 4x latent scale | `0.3319` then `0.1456`; residual `3.7e-02` |
| T6 `rho` on a narrow-`Y` set | `0.2269` then `0.1009`; residual `4.2e-03` |
| T7 wrapper against the stacked algebra by hand | coefficient `0.000e+00`, output `1.06e-16` relative |
| T7 flag OFF | `torch.equal` TRUE, the wrapped tensor itself |

**Criterion.** Residual at or below `1e-12` at full rank and at deliberately reduced rank. Full
rank `9.9e-17`, genuine reduced rank `7.2e-17`. **MET**, two orders better than the investigation
prototype.

**One correction to the test, which changes what the criterion means.** The first version read
"deliberately reduced rank" as "truncate the pseudo-inverse harder", and at `rcond = 0.3` the
residual rose to `2.0e-04`, apparently failing. That reading is wrong, and the failure is real but
is not the construction's. Truncating a NONZERO singular value leaves exactly that direction in
the residual, so exactness is genuinely lost; the any-rank proposition concerns a GENUINELY
rank-deficient basis, one carrying a zero singular value, where only the uniqueness of the
coefficient is at stake. Tested that way, with a duplicated column, two duplicated columns and a
zeroed column, the residual stays between `6e-18` and `1.2e-16`. Both readings are now reported,
because the second is an independent argument for the column scaling of stage 2: scaling is what
keeps a `sigma_0`-relative cutoff from discarding genuine directions and switching the guarantee
off without saying so.

**The honest off-reference picture.** On the frozen set the construction is exact and `rho` goes to
`0.0000`. Away from it, `rho` falls only from `0.33` to `0.14` and the frozen-coefficient residual
is `3.6e-02`, not `1e-16`. The construction removes part, not all, off its own point set. That is
generalisation failure of a coefficient fitted to a write which is not in the span, not a defect of
the projector: the investigation established that a write genuinely lying in the baseline span is
removed at ANY point set, to `4.9e-14`.

**T5 is the wording gate.** Median per-sample product `3.2e-01` against a stacked sum of
`2.9e-14`, a ratio of `1.1e+13`. Nothing we write may be worded pointwise.

**Plan change:** the hollow interconnect stub in this stage was deleted rather than left asserting
nothing; the wrapper inside a full `Interconnect` is stage 6's business.

## Stage 5 -- the gradient through the subtraction -- 05-gradient/

**Measured.** `M = 64` construction evaluations, 32 evaluation points, a deliberately small ANN
(128 parameters, so an independent central difference is affordable), float64.

| test | number |
|-|-|
| **T1 autograd vs central difference, relative 2-norm (THE CRITERION)** | **`1.8122e-09`** |
| T1 worst per-parameter relative difference | `3.88e-04` (on components whose gradient is near zero; the norm is the criterion) |
| T2 additional-state columns, subtraction | `0.0e+00`, exactly |
| **T3 `norm(F' dG~/deta)`, coefficient differentiated through** | **`6.71e-14`** |
| **T3 the same, coefficient DETACHED (the control)** | **`5.8173e+01`** |
| T3 the same with no projection at all | `5.8173e+01` |
| T3 ratio detached to unprojected | **`1.000000`** |
| T3 ratio differentiated-through to unprojected | `1.15e-15` |
| T4 `dG~/deta` vs `(I - P) dG/deta`, four random parameter directions | `1.05e-10`, `2.80e-10`, `6.52e-11`, `6.42e-11` |
| T5 `d loss / d free_params` | `None`, no gradient reaches the physical parameters through the basis |

**Criterion.** Central-difference agreement at or below `1e-6`; latent rows exactly `0.0`; the
detached control roughly `1e-2`, not `1e-13`. Measured `1.81e-09`, `0.0`, and a control that
reproduces the unprojected value to ratio **1.000000**. **MET** on all three.

The detached control is the sharpest number in the ladder so far. Detaching the coefficient leaves
the forward output bit-identical and returns the gradient orthogonality to EXACTLY its unprojected
value, ratio 1.000000 rather than merely "close". Differentiating through the coefficient is not
an optimisation; it is the method.

**T6 qualifies a sentence in the handoff, and the qualification is worth carrying forward.**
Handoff Sect. 8 says candidate (a) "dissolves the section-6 latent-blindness problem without using
either repair the user declined: the evaluation points come from the real rollout, so they carry
nonzero `x_a` already." Measured, half of that holds and half does not.

- The coefficient VALUE does depend on the latent content of the reference points:
  `norm(a from a rollout with live x_a - a with x_a forced to zero) / norm(a) = 1.57e-01`. The
  rollout-drawn points are not cosmetic; they change what gets subtracted.
- `d a / d eta_lat` is nonetheless **exactly `0.000000e+00`** under the shipped design, for EVERY
  reference set tried, the rollout-drawn ones included.
- The cause is the FREEZE, not the choice of points, and the diagnostic separates them cleanly.
  Leaving the reference points ATTACHED to the ANN parameters gives `5.49e+05` at one rollout step
  and `7.36e+04` at two; forcing `x_a` to zero gives `0.0` even attached. So both mechanisms are
  visible and distinguishable: zero latent content kills it, and so does detaching.
- And the freeze is not negotiable. The derivation rules out fitting the coefficient along the
  live trajectory precisely because the correction at step `k` would then depend on the learned
  writes at later steps, i.e. on the future.

**What this does and does not mean, stated carefully, because an earlier draft of this entry
overstated it.** Two things must not be run together.

1. *The latent WRITE `g_a` is deliberately not orthogonalized, and that is correct.* The baseline
   defines no update equation for the additional states, so the basis has zero rows there and
   there is nothing for a learned contribution to be orthogonal TO. Stage 4 measured those columns
   changing by exactly `0.0`. This is the supervisors' settled definition working as intended, not
   a gap.
2. *The physical write's DEPENDENCE on the additional states is a separate question, and the
   projection does handle it.* The subtraction is applied at the actual rollout point, `x_a`
   included, so whatever part of the physical write the latent states drive is projected out along
   with the rest. Nothing about that is blind.

What is exactly zero is narrower than either: only `d a / d eta_lat`, the sensitivity of the
FITTED COEFFICIENT to the latent-path weights. The coefficient is fitted once per pass from a
frozen snapshot, so it does not know how it would shift if those weights moved. The projection is
still applied in full at every point.

So the limitation is real but small, and the earlier phrasing here -- "the construction does not
charge the latent path" -- was too strong for what was measured and has been replaced. The
accurate statement is: the coefficient fit is insensitive to the latent-path weights, while the
subtraction itself is not. The latent parameters are trained by the loss through the rollout. What
still may NOT be claimed is that the projection constrains the augmentation's use of the
additional states, which is what Sect. 7 of the derivation says in words; but that is a statement
about the latent WRITE, item 1 above, and it was never something this construction set out to do.

**Plan change:** none to the code. One claim narrowed, recorded above and in D-190's constraints.

## Stage 6 -- integration -- 06-integration/

**Caps observed.** Batch 8, `nf` 50, 20 optimizer steps, CPU, float64. This is a SEAM test;
nothing in it is a result about learning.

**One library change the stage forced.** `OBC_Projected_ANN_Block` subclasses `Block`, not
`nn.Module`. `Interconnect.determine_signal_ix` resolves a block to its signal index by
`isinstance(signal, Block)`, so a bare `nn.Module` cannot be wired at all. Caught by trying it.

**Measured.** Wired exactly as `scripts/gantry/gantry_dynamic/model.py` wires the pipeline.

| test | number |
|-|-|
| **T1 flag-off bit-identity over 50 recursive steps, output sequence** | **`torch.equal` TRUE**, max abs difference `0.0e+00` |
| **T1 the same, final state** | **`torch.equal` TRUE**, `0.0e+00` |
| T2 physical-block keys in the wrapper's `state_dict` | none; keys are `ann.net.*` only |
| T2 interconnect parameter count | 610, exactly ANN + 10 reduced physical |
| T3 checkpoint save and load after a `jacfwd` over the physical block | succeeded, 0.023 s, 34 keys, 18108 bytes |
| T3 reloaded model, flag off, reproduces the flag-off rollout | `torch.equal` TRUE |
| T4 flag-on rollout | finite, `max abs y = 2.61`, differs from flag-off by `1.53` |
| T4 wall cost of the seam, forward only, batch 8 | 12.50x (2.42 s against 0.194 s) |
| T5 20 optimizer steps with the epoch-boundary rebuild every 10 | ran, loss `8.65e-02` to `9.59e-03`, all finite, 4.57 s per step |

**Criterion.** Flag off bit-identical, exact equality on the output tensor. **MET**, and on the
final state as well as the output sequence, over 50 recursive steps rather than one.

T3 is the `_cur` hazard closed: the basis build runs a `jacfwd` over the physical block, which
leaves functorch `TensorWrapper` duals in the per-forward cache, and without the `__getstate__`
drop already in HEAD `torch.save` raises `NotImplementedError: Cannot access storage of
TensorWrapper`. The round-trip is clean with it.

T5 is not evidence about learning and is not offered as any: 20 steps against a synthetic target
at batch 8. What it shows is that the seam survives backward, the optimizer and the mid-run basis
rebuild, and that the physical parameters move (`free_params` reaches `1.4e-03`) rather than
sitting frozen.

**The cost, and how to read it.** 12.5x forward-only at batch 8 against the 9.0x stage 0 measured
at batch 256, and 4.57 s per training step at batch 8 with `nf = 50`. Both are consistent with the
per-operation overhead stage 0 isolated: the smaller the batch, the larger the overhead's share.
The stage-0 figure at the production batch, 3.19x on a training step, remains the number the
design is judged on; the stage-6 figures are an upper bound taken at a batch 32 times smaller.
Worth recording separately: the coefficient is refitted at EVERY timestep, so the graph holds one
ANN-over-`M` subgraph per step, `nf * M = 3200` ANN evaluations per rollout here. That is cheap in
time (the ANN is tiny beside the physical step) but it does grow the graph, and at `nf = 400` with
`M = 256` it would be 102400. If memory ever becomes the binding constraint, caching the
coefficient across the timesteps of one pass is exact -- the parameters do not change within a
pass -- and was left out only because it needs pass-boundary bookkeeping that the interconnect
gives no hook for.

**Plan change:** none.

## Stage 7 -- Condition 4, the scientific gate -- 07-condition4/

**Measured.** 64 designed points, three seeds, static learning component, baseline started 15
percent detuned on the ten combinations, LBFGS, float64. Two truths identical in every respect
except whether the residual lies inside the baseline tangent span.

| truth | `rho(Phi; delta)` | arm | mean | max | seed sd | fit |
|-|-|-|-|-|-|-|
| friction `-c_f tanh(qdot/v0)` | **0.8888** | no training | 0.1080 | 0.1500 | - | - |
| | | A no projection | **0.0872** | 0.1424 | 1.2e-03 | 3.73e-07 |
| | | C 2026 subtraction | **0.0974** | **0.2914** | 3.9e-03 | 6.78e-07 |
| the same residual, projected out | **3.07e-15** | no training | 0.1080 | 0.1500 | - | - |
| | | A no projection | 0.0818 | 0.1396 | 3.2e-03 | 1.12e-07 |
| | | C 2026 subtraction | **0.0773** | 0.1397 | **7.6e-04** | 1.09e-07 |

**Criterion.** At low `rho` the projected arm beats the unprojected; at high `rho` it is expected
to be worse, and reporting that honestly is the pass condition. Measured: at
`rho = 3.1e-15` projected `0.0773` against unprojected `0.0818`, and at `rho = 0.8888` projected
`0.0974` against unprojected `0.0872`. **The sign of the effect flips with Condition 4. MET.**

Two secondary numbers carry the same message. Inside the hypothesis the projected arm is **four
times more reproducible across seeds** (`7.6e-04` against `3.2e-03`) and fits marginally better
(`1.09e-07` against `1.12e-07`), which is the investigation's "a factor 10 more reproducible, and
it fits the data as well" at a smaller scale. Outside it, the worst single combination is **twice
as bad** (`0.2914` against `0.1424`) while the mean is only 12 percent worse, so the damage is
concentrated rather than spread, exactly as the mechanism predicts.

**The effect is milder here than in the investigation, and the reason is stateable.** There, arm C
went to 0.4175 against 0.0891, a factor 4.7; here 0.0974 against 0.0872, a factor 1.12. Three
differences, none of them the implementation: the residual is smaller relative to the detuning
(a `1.5e-03` velocity increment against a `7.6e-03` state increment and a 15 percent parameter
displacement), the budget is 20 LBFGS calls on 64 points rather than a converged fit on 256, and
`rho` is 0.8888 rather than 0.9465, so an eighth of the residual has somewhere legitimate to go.
The DIRECTION is what this stage tests and the direction reproduces; the magnitude is not claimed
to transfer.

**One difference from the investigation worth flagging rather than smoothing over.** There, the
high-`rho` damage fell on the damping combinations (`cb_sum` 1.79, `cy` 1.27, `cg1` 0.61). Here the
standout is `mh`, which doubles from 0.0740 unprojected to 0.1526 projected, while `cb_sum` moves
only 0.1424 to 0.1440 and `cy` and `cg1` actually improve. The residual is injected differently:
here it is an additive per-step increment on the velocity ROWS, there it entered the
continuous-time force balance through `f_extra` and was therefore filtered by `M(Y)^{-1}` before
reaching the state. Those project onto different baseline directions, so a different combination
absorbs the damage. Which combination is hit is a property of the residual, not of the method, and
no claim should name a particular combination as the vulnerable one.

**Deviation from the caps, stated not worked around.** This script takes 228 s, not the 120 s
handoff Sect. 12 asks for. Everything else is inside the caps: 64 points, 20 `opt.step()` calls,
CPU, float64, no rollout, peak memory a `384 x 11` basis. The budget itself was chosen from a
measurement rather than picked: at one `opt.step()` the arms are indistinguishable (0.1078 against
0.1080 untrained), at five the high-`rho` effect has the WRONG sign, at twenty it has settled.
Reporting a flip from a five-call budget would have been reporting noise.

**Plan change:** none. One defect in the stage's own first version is recorded in the script: it
re-initialised the ANN to `N(0, 0.1)`, which put the learned write three orders above the residual,
sent the starting loss to 6.27 instead of 3.2e-04, and spent the entire optimiser budget shrinking
noise so that no arm moved its parameters at all. The pipeline's own zero-initialised last layer is
what the experiment needs.

---

# Summary: all eight stages

| stage | criterion | measured | verdict |
|-|-|-|-|
| 0 cost | `vmap(jacfwd)` under 2x a block step | 28.2x; but the rollout needs ONE directional derivative, and a training step costs **3.19x** at the production batch | criterion NOT met as written, design revised and proceeds |
| 1 reduced block | relative difference below `1e-12` | **`0.000e+00`**, bit-identical | MET |
| 2 rank | `rank(J) = 10` at every rtol, `rank([J abs c]) = 11` | 10 and 11 at every rtol and every point count; `cond = 4.50` | MET |
| 3 Taylor | log-log slope `2.0 +/- 0.1` | `[1.9940, 2.0078]`, worst deviation `0.0078` | MET |
| 4 projection | residual at or below `1e-12` | **`9.9e-17`** full rank, `7.2e-17` rank-deficient | MET |
| 5 gradient | central difference `1e-6`, latent rows `0.0`, control fails | `1.8e-09`, `0.0`, control at ratio **1.000000** of unprojected | MET |
| 6 integration | flag off bit-identical | `torch.equal` TRUE over 50 recursive steps, output AND final state | MET |
| 7 Condition 4 | helps at low `rho`, honest report at high | `0.0773` vs `0.0818` at low, `0.0974` vs `0.0872` at high: **the sign flips** | MET |

**What was built.** `Reduced_Gantry_State_Block` in `model_augmentation/fit_systems/blocks.py`
(`@added`) and `model_augmentation/fit_systems/obc_projection.py`
(`__project_origin__ = "added"`), holding `stacked_basis`, `directional`, `rank_report`,
`principal_angles`, `RankLossAbort`, `OBCBasis` and `OBC_Projected_ANN_Block`. One `# CHANGED:`
marker in the parent block, keeping `RMSE_baseline` as a float. Nothing else in
`model_augmentation/` was touched.

**The three things a reader should carry away.**

1. **The handoff priced the wrong object and the design is better for it.** The rollout never needs
   the ten-column Jacobian, only a directional derivative in the coefficient's own direction, plus
   the nominal transition it has already computed. That is an exact identity, verified to
   `4.6e-16`, and it is the difference between 15.2x and 9.0x on the forward and between an
   unaffordable design and a 3.19x one on a training step.
2. **The construction is exact where it is defined and partial where it is not.** On the frozen
   construction evaluations the residual is `9.9e-17` and `rho` goes to `0.0000`. On a foreign
   point set `rho` falls only from 0.33 to 0.14. Both belong in any description of the method.
   The same "exact here, partial there" shape governs the latent question: the subtraction IS
   applied to the whole physical write including its latent-driven part, and only the fitting of
   the coefficient is insensitive to the latent-path weights (`d a / d eta_lat = 0.0`).
3. **Condition 4 decides the sign, and it is a property of the true residual, which on real data is
   unknown.** Every recovery result must be reported with an estimate of `rho(Phi; delta)` beside
   it. This is the finding that governs whether the method can be claimed to preserve physical
   interpretability on this plant at all.

**What is NOT established by any of this.** Everything remains noiseless, float64, one-step, and
on designed points; no dataset was loaded anywhere in the ladder. Nothing has been trained. No
claim is made at horizon level, under free-run multiple shooting, under noise, or on real Telica
data, and no orthogonality claim may be worded pointwise. The analytic forward-sensitivity route
(handoff candidate (b)) is not built and its estimated 2x is an estimate, not a measurement.

**The one open item a next session should take first.** The coefficient is refitted at every
timestep, which is exact and cheap in time but grows the autograd graph by one ANN-over-`M`
subgraph per step; at `nf = 400` and `M = 256` that is 102400 ANN evaluations per rollout. Caching
the coefficient across the timesteps of one pass is exact, because the parameters do not change
within a pass, and was left out only because the interconnect offers no pass-boundary hook. If
memory binds before anything else does, that is the change to make.


---

# Stage 8 -- one-step OBC on the production pipeline (D-192) -- 08-one-step/

Handoff `tasks/handoffs/2026-09-16-implement-one-step-obc-fable.md`. Unlike stages 0 to 7 this
stage runs on the PRODUCTION model (`gantry_dynamic.model.build_model`, closed loop, the
entry point's config switched to reduced joint estimation) and on the ACTIVE dataset
`augmentation_ma50_b140-230_a6_z03`, float32 rollout, float64 construction. The remnant
`obc_projection.py` of stages 2 to 6 was not used.

**Built.** `model_augmentation/fit_systems/obc.py` (`__project_origin__ = "added"`): `OBCBasis`
(economy SVD, declared rank tolerance, differentiable coefficient, `r_perp`, `r_overlap`,
`relative_change`), `OBCLifecycle` (once-per-objective coefficient, epoch refresh, rank-loss
retention, `OBCRankLossAbort`), `build_stacked_sensitivity` (chunked float64 `jacfwd` over the
batched step), `directional_correction` (the `jvp` reference used by equivalence tests). Gantry side in
`scripts/gantry/gantry_dynamic/obc_gantry.py` (reference set, `GantryOBCCorrection`,
`attach_obc`) and `obc_diagnostics.py` (qualification, absorber truth plant with parameters as
arguments, `rho`). Seams: `Reduced_Gantry_State_Block.transition_from_free(v, z)` (pure, no
`_cur` write; `_rk4`, `_deriv_with`, `_mats_from_raw` factored out with the same operations),
`Interconnect.forward(x, u, obc_pass=None)` plus the `obc_correction` submodule,
`SSE_Interconnect.simulate`, `ClosedLoopSimulator.__call__`, `closed_loop_rollout`,
`_rollout_segment` (all thread `obc_pass`), `SSE_Interconnect_Composed.loss` (refresh at the
epoch boundary, coefficient once), D-198 validation synchronization of `theta_frozen`, checkpoint exclusion
of the lifecycle. Config: `obc`, `obc_refresh`, `obc_ref_chunk`, `obc_rank_rtol` (defaults off,
recorded in config.json). Three arm configs in `08-one-step/experiment_configs.py`
(`joint_noproj`, `joint_soft2025`, `joint_obc`); none launched.

**Compile probe** (`probe_compile.py`, Quadro P2000 sm_61, torch 2.5.1). Rung 1 holds under
Dynamo plus AOTAutograd: one graph, 0 graph breaks, 825 ops; with `error_on_recompile=True` no
recompile across two coefficient values, a rewritten `vbar` buffer, a grad-carrying coefficient
and ten objectives through the production simulator seam including one mid-run basis refresh;
compiled output and `d x_plus / d theta` equal eager to `0.0`. Rung 2 (the argument contract)
was found by the probe itself, three times: the coefficient, the state and the fed-back input
must carry `requires_grad=True`, and rank and batch must match the rollout's, which is how
`fit()` calls the step; each mismatch was a probe-only call shape. Inductor codegen could not
run here (`Triton only supports devices of CUDA Capability >= 7.0, but your device is of CUDA
capability 6.1`); that measurement needs the cluster. Rung 4 recorded for reference only:
central difference against the exact JVP, relative error `6.6e-4` float32, `7.2e-10` float64.

**Behavioral tests** (`test_behavior.py`, CPU float32, nf 50, batch 8, reference stride 200).

| | number | verdict |
|-|-|-|
| T1 coefficient once per objective | +1 objective eval, 1 reference-set ANN pass, 50 rollout passes | MET |
| T2 gradient through the coefficient | norm of `d theta.sum() / d ANN weights` = 3.77e+02 | MET |
| T3 detaching changes the ANN gradient | loss identical, relative gradient change 4.3e-03, cos 0.999991 | MET |
| T4 stacked orthogonality at the floor | see the floor paragraph below | MET |
| T5 after one Adam step | fresh coefficient at the floor, stale coefficient 2.8e-03 | MET |
| T6 additional-state rows | `torch.equal` on rows 6..13 and on `y`; physical rows differ by 2.6e-04 | MET |
| T7 disabled path vs commit a52dd55 | block, `transition_from_free == forward`, `Interconnect.forward`, 50-step closed-loop rollout all `torch.equal` (float32; block also float64) | MET |
| T8 ten physical gradients | all nonzero, 1e-07 to 9e-05 | MET |
| T9 save / reload | no functorch wrapper in `_cur`; 7 MB checkpoint without lifecycle or reference set; restore on the live system; reload reproduces; objective runs against the reloaded adapter | MET |
| T10 no recompilation | `results/2026-09-16/probe_compile.log`: 10 objectives, one refresh, no recompile (aot_eager) | MET |
| T11 synthetic lifecycle | rank 9 retains the previous basis, second consecutive aborts, first-build rank loss aborts | MET |

**Dataset qualification** (`run_qualification.py`, every valid sample: 671,944 tuples, 4,031,664
rows, 14 x 47,996). Both points rank 10 at the numpy tolerance (rtol 8.95e-10) and at every
rtol from 1e-4 to 1e-14.

| | nominal | detuned 10 percent (training start) |
|-|-|-|
| singular values | 2.15e+02, 2.36e+01, 1.94e+01, 1.15e+01, 3.69, 3.16, 2.37, 9.2e-01, 7.9e-01, 2.8e-01 | 2.38e+02, 2.37e+01, 2.35e+01, 1.24e+01, 4.49, 3.41, 2.46, 1.06, 9.2e-01, 2.7e-01 |
| condition number | 7.66e+02 | 8.93e+02 |
| column-normalized condition number | 10.5 | 12.5 |
| worst column correlation | `cg1 ~ cg2 = -0.977` | `cg1 ~ cg2 = -0.982` |
| leave-one-record-out | rank 10 for every record; `sigma_min` 0.232 (drop T13) to 0.281 | rank 10 for every record; `sigma_min` 0.221 (drop T13) to 0.267 |

Relative Frobenius distance between the two bases 0.110. Column contributions: `kb_sum` comes
almost entirely from the two yaw records (T12 0.42, T14 0.58); the damping columns `cg1`, `cg2`
from T11, T13, T10 (0.25 / 0.32 / 0.16); `cy` from T7, T14, T11; the mass columns from every
record. No record is indispensable for rank, and no column is carried by a single record. The
one poorly separated pair is `cg1` against `cg2` (correlation -0.98; the tenth singular value
is that difference direction, at 1.3e-3 of the largest). Y coverage spans [-0.300, +0.300]
with mass at the five standstill points and under the sweeps.

**Verdict on the dataset.** Numerically usable rank ten at both points and at every
leave-one-out; condition number below 1e3 in the block's coordinates. NOT a claim of practical
joint identifiability: `cg1 - cg2` is separated by one part in 770, and this preflight measured
structure and conditioning, not estimation error. The training configuration is marked
"rank-qualified, cg1/cg2 weakly separated".

**Truth plant check and the rho table.** One-step prediction of the recorded next state from the
reference tuples: D-188 parameters (`ma_frac 0.50, zeta_a 0.03`) relative RMS 1.74e-4;
`zeta 0.05` 4.4e-4; the `oracle.py` values (`0.10, 0.05`) 7.3e-3. D-188 confirmed and used.
`rho(B; delta) = ||P_B delta|| / ||delta||`, routed physical rows, normalized coordinates:

| delta | `J` nominal | `[J | Gamma]` nominal | `J` detuned | `[J | Gamma]` detuned |
|-|-|-|-|-|
| truth step, recorded absorber state | 0.2003 | 0.2003 | 0.8545 | 0.8545 |
| truth step, zero absorber state | 0.3723 | 0.3723 | 0.9413 | 0.9413 |
| recorded next state minus baseline (data) | 0.2067 | 0.2068 | 0.8559 | 0.8559 |

`Gamma` is 0.90 outside `span(J)` and `rank([J | Gamma]) = 11`, yet adding it changes `rho` in
the fourth decimal: on this data the affine offset does not overlap the discrepancy. At the
NOMINAL point a fifth of the discrepancy is in the baseline tangent (Condition 4 unfavourable
but not dominant); at the DETUNED start 85 percent of the discrepancy is in span, because there
the discrepancy is dominated by the parameter error itself, which is exactly what joint
estimation is supposed to remove and the projection must leave to the physical block. No
training conclusion is drawn from this (handoff sect. 5.5).

**Decimation** (diagnostic only; the primary set is complete). Stride 10 / 100 / 1000: rank 10,
cond 888 / 897 / 1016, scaled-singular-value error 6.8e-3 / 6.5e-3 / 9.7e-2, coefficient of
the ANN field relative to the full solve 2.0e-3 / 2.3e-2 / 1.9e-1; for random fields the
decimated coefficient is wrong by factors of 1 to 30. Not adopted.

**Runtime and memory** (full set, CUDA P2000, eager, batch 512): per step forward off 13.0 ms,
on 72.8 ms (5.6x); forward plus backward off 48.9 ms, on 131.2 ms (2.7x); once-per-objective
reference pass plus solve 0.020 s forward, 0.090 s with backward; basis refresh 11.3 s (build
10.6 s, SVD 0.5 s); one update at nf 20 off 0.75 s, on 2.82 s (3.8x); a second run of the same script measured 3.3x per step forward plus backward and 4.0x per update, so the eager ratio on this card is 2.7x to 3.3x per step and 3.8x to 4.0x per update. Memory: reference tuples
70 MB (172 MB with the diagnostic arrays), basis factors 322.5 MB float64 on the device, retained
ANN graph after the reference pass 194 MB, peak 1.38 GB. The per-step ratio is the eager
dispatch-bound figure on a card that cannot run inductor; stage 0 measured 3.19x for a training
step at the production batch on the CPU, consistent with the 2.7x here.

**Orthogonality floor.** Measured, not assumed, through the ROLLOUT path (float32 ANN write minus the float32 JVP correction with the float64-solved coefficient cast to float32), on random fields with the ANN field's norm and in-span fraction (`r_overlap` 0.179 on the full set). Acceptance = 3 x the worst draw (HEURISTIC c = 3). Decimated set (stride 200, T4): floor 1.5e-9 to 6.0e-8 over ten draws, ANN residual `r_perp` 9.3e-9, float64 stored-factor construction 4.5e-15; after one Adam step the fresh coefficient gives 1.1e-9 and the stale one 2.8e-3 (T5). Full set (671,944 tuples): floor max 4.3e-8 over five draws, ANN residual 5.8e-8, float64 construction 5.6e-17; `kappa * eps` for reference 1.1e-4 float32, 2.0e-13 float64. Within the declared threshold at both scales: the stacked condition holds to float32 rounding on the path training actually runs, and to float64 rounding in the construction.

**Plan changes.** Two, both from the probe: the coefficient argument contract (rung 2) is now
written into D-192 and the probe; and the basis is built on the device of the live parameters
with the drift comparison device-safe, after the mid-run refresh in the probe crossed CPU and
CUDA. One diagnostic correction: the random-field floor must match the ANN field's in-span
fraction, not only its norm, or it is optimistic by the ratio of the two in-span fractions
(first run, white fields: max 1.2e-9 against an ANN-field residual of 5.8e-8 on the full set;
`run_qualification_run1_whitefloor.log`).

**Not established.** No training was run; nothing here says whether the projection helps
recovery on this plant. Inductor codegen and CUDA graphs with the JVP are untested (host
limitation). Every orthogonality claim is one-step, stacked over the reference set, local at
`vbar`, on the slice `x_a = 0`.
