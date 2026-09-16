# Decision report

One recommendation per priority, the evidence behind it, and what it does not establish.
Every number is from a log in `results/` produced by a script in this folder. Nothing is
quoted from a stored artifact of an earlier session except where explicitly named as a
cross-check.

Environment: `conda run -n GraduationProject python`, torch 2.5.1, numpy 2.0.1,
scipy 1.17.0, sympy 1.14.0, float64 throughout. MATLAB R2025a and R2021b are installed;
no MATLAB job was run here, see `continuation.md` item 4.

---

## The single most important finding

The whole method turns on one hypothesis, and it is not one of the ones the existing
derivation flags.

> **Györök's Condition 4 (Eq. 17), that the true unmodeled term is orthogonal to the
> baseline regressor on the data, is the hypothesis that decides whether the
> orthogonal-by-construction parameterization helps or actively harms physical parameter
> recovery on this plant.** It is not a technicality that weakens a bound. Measured on
> the same rig, with everything else held fixed, the sign of the effect flips.

Two truths, identical in every other respect, differing only in whether the residual lies
inside the baseline tangent span. Learning component static, baseline started from a
detuned parameter vector, LBFGS, three seeds, the same 256 designed points.

| truth | `rho(Phi; delta)` | arm | combination error, mean | max | seed sd |
|---|---|---|---|---|---|
| friction `-c_f tanh(qdot/v0)` | **0.9465** | no training | 0.1122 | 0.3000 | - |
| | | A no projection | 0.0891 | 0.4565 | 4.6e-03 |
| | | B 2025 penalty, beta 1e2 | 0.0886 | 0.3654 | 8.9e-04 |
| | | C 2026 subtraction | **0.4175** | **2.1666** | 1.6e-01 |
| the same residual, projected out of the span | **2.7e-15** | no training | 0.1122 | 0.3000 | - |
| | | A no projection | 0.0825 | 0.2732 | 9.9e-03 |
| | | B 2025 penalty, beta 1e2 | 0.0978 | 0.3013 | 1.0e-02 |
| | | C 2026 subtraction | **0.0206** | **0.1684** | **1.0e-03** |

(`results/s3c_competition.log`, `results/s3d_refset_identity.log`,
`results/s3e_condition4_control.log`. `rho(Phi; G) = ‖P_Phi G‖ / ‖G‖` is the fraction of
a write that the baseline tangent set can absorb.)

Inside the hypothesis, arm C is a clean win: a factor 4 better than no projection on the
mean combination error, better on the worst combination, a factor 10 more reproducible
across seeds, and it fits the data as well (1.403e-04 against 1.428e-04). Outside the
hypothesis it is a factor 4 **worse than doing nothing**, and the damage is entirely on
the damping combinations: `cb_sum` 1.79, `cy` 1.27, `cg1` 0.61, while `kb_sum`, `mh`,
`m_total`, `J_eff`, `d` are all recovered to three or four digits.

The mechanism is not a defect of the implementation. A friction law
`-c_f tanh(qdot/v0)` is, at small velocity, `-(c_f/v0) qdot`, which is exactly extra
viscous damping, and viscous damping is a baseline parameter direction. The construction
forbids the learning component from representing anything in the baseline span, so the
94.6 percent of the friction that lies in the span has nowhere to go except into the
damping parameters. This is precisely the paper's own error expression, Eq. (21),
`‖theta* - theta_hat‖ = ‖(Phi^T Phi)^{-1} Phi^T Delta‖`, made visible.

**The obvious alternative explanation was tested and refuted.** Stage 3d re-ran arm C with
the reference set made identical to the estimation set, so that orthogonality holds
exactly on the data the loss sees (measured residual ratio 3.0e-13, `rho(G) = 0.0000`).
The combination error was 0.4175, no better than with a foreign reference set (0.4070).
A foreign reference set is not the problem; Condition 4 is.

**Consequence for the thesis.** Whether this method can be claimed to preserve physical
interpretability on the gantry depends on a property of the *true residual*, which on
real data is unknown. The claim must be conditioned on it, an estimate of
`rho(Phi; delta)` must be reported alongside any recovery result, and the synthetic
validation residual must be chosen and justified on this basis, not for convenience. The
hidden-MSD absorber used elsewhere in this project is a dynamic mode the baseline has no
state for, so it is a far better test object than friction; that should be stated as a
reason, not assumed.

---

## Priority 1 - baseline parameterization and Y scheduling

**Recommendation.** Keep the explicit, self-scheduled RK4 transition exactly as the
pipeline has it, and take the protected object to be the affine tangent set
`[Phi_p | c]` at a declared expansion point. Build the reference point set with
**endogenous** scheduling spanning the full documented range `Y in [-0.30, +0.30] m`.
Retain the implicit force balance as a consistency check only.

**Evidence.**

- No exact linear-in-parameters form of the explicit map exists. `det M(Y)` depends
  symbolically on `Jb, Jh, Lb, m1, m2, mb, mh`, so every velocity row is a ratio whose
  denominator carries the unknowns (`s12`, S1a). An LFR factorization does not change
  this; it rewrites the inverse as a feedback loop, it does not remove the parameters
  from the denominator.
- The **implicit** force balance `r = M(Y) qddot + C qdot + K q - P u` **is** exactly
  affine in 11 lumped coefficients, verified symbolically to `[0, 0, 0]` with every
  second derivative zero (`s12`, S1b). It is rejected as the estimator for three named
  reasons: it needs `qddot`, which is not measured; it is redundant, 11 coefficients for
  10 degrees of freedom, with the admissibility constraint `MD^2 = MH * MD2` that must be
  imposed or `(mh, d)` become unrecoverable; and it weights the residual by `M(Y)`, so it
  is a different estimator, not a rearrangement of the same one.
- Frozen scheduling is a diagnostic. At `Y` frozen to 0.00, 0.15 or 0.30 the transition
  differs from the self-scheduled one by 9 to 13 percent of the state increment, against
  a float64 round-off floor of exactly `0.00e+00` (`s12`, S1c).
- **The decisive number:** only `0.108` of a frozen-scheduling transition error lies in
  the span of the 14 parameter tangents. So 89 percent of it is outside the protected
  set, which means a reference set built with frozen scheduling would leave the learning
  component free to absorb the scheduling convention itself, and the projection would not
  charge it. This is why the reference set must use endogenous scheduling.

**Not established.** Whether an exogenous measured `Y` is preferable on real data, where
`Y` is measured and the intra-sample re-read (`1.33e-08`, relative `7.6e-06`) is far
below the sensor noise floor. On that data the two conventions are likely
indistinguishable, but that is an argument, not a measurement.

## Priority 2 - identifiable parameter combinations

**Recommendation.** Keep training the 14 raw scalars in **log coordinates** and keep
reporting only the 10 combinations, which is what the code already does. Reparameterize
to the 10 only if the flat directions start to cause trouble in practice; the structural
case for it is made but the practical case is not.

**Evidence.**

- The map from the 14 raw scalars to `(M0, M1, M2, C, K)` factors exactly through the 10
  quantities already coded in `_combos_from_raw`. The stacked parameter Jacobian of the
  RK4 transition has rank 10 and nullity 4, stable at every relative tolerance from
  `1e-4` to `1e-14`, with a gap factor of `7.9e12` between the last kept and first
  discarded singular value (`s12`, S2). Reproduced at three admissible parameter sets and
  two reference grids.
- The four null directions are `kb1-kb2`, `cb1-cb2`, and the two-dimensional family
  `(-m1/2, -m2/2, +mb, +(Lb^2/4) J)` for `J` either inertia, whose difference is
  `Jb-Jh`. Maximum principal angle to the numerically computed null space: `2.58e-08 rad`, and
  below `4e-08 rad` at three parameter sets and two grids.
  Exact invariance confirmed directly, `max |dx_next| = 0.00e+00` at displacements up to
  `1e-1`.
- Log and normalized coordinates improve the condition number on the identifiable range
  by a factor 52 over physical coordinates (180 against 9360), which is an argument for
  the existing log parameterization independent of positivity.

**Two corrections to project prose, reported not edited.**

1. The closing remark of `state-level-obc-derivation.tex` Sect. 9 asks for a
   reconciliation that is not needed: it states that "the project's documented set of ten
   identifiable quantities lists the inertia sum on its own", but `blocks.py:1056` codes
   `J_eff = Jb + Jh + (m1+m2) Lb^2 / 4`, which is exactly the combination the MATLAB
   Euler probe found. The two agree. The remark's premise is what is wrong.
2. `CLAUDE.md`, Control Engineering Stance item 3, says "only `kb1+kb2`, `cb1+cb2`,
   `Jb+Jh` are identifiable". `Jb+Jh` alone is **not** identifiable: it trades against
   `m1+m2` at fixed `m_total`. Only `J_eff` is invariant, and there are four null
   directions, not three.

**Not established.** Multi-start recovery behaviour in reduced versus raw coordinates,
and the empirical rank of the stacked measured-output sensitivity over production-length
records.

## Priority 3 - protected basis, linearization point and reference policy

**Recommendation.** Use the affine tangent-plus-offset basis `[Phi_p | c]` in log
coordinates at a declared expansion point, with the coefficient solved by a
Moore-Penrose pseudo-inverse and **differentiated through**. Report `rho` and never the
raw inner product. Scale the regressor columns before the pseudo-inverse, or state the
cutoff against the parameter columns rather than against `sigma_0`.

**Evidence.**

- The construction is exact on the reference set at any rank: residual ratio `1.96e-14`
  (`s3` T1). Rank deficiency costs only the uniqueness of the coefficient: moving the
  coefficient by `3.70` along a null direction moved the projected write by `1.19e-15`
  (`s3` T2).
- **Differentiating through the coefficient is not optional.** `‖Phi^T dG̃/d eta‖` is
  `1.29e-13` when the coefficient is differentiated through and `4.49e-02` when it is
  detached, which is exactly the unprojected value. The identity
  `dG̃/d eta = (I - P) dG/d eta` holds to `3.84e-17` (`s3` T3).
- The offset column is real and strictly enlarges the protected set: rank 10 to 11, with
  57 percent of the offset column outside the parameter span (`s3` T7).
- **The offset column is also a numerical hazard.** Its singular value is about 1030
  times the largest parameter column, because it is the nominal transition itself, whose
  position rows are order 0.3 m, while every parameter column is a derivative of order
  1e-3 (`s3b` A). A pseudo-inverse cutoff stated relative to `sigma_0` is therefore a
  cutoff relative to the offset: at `rcond = 1e-4` the rank drops from 11 to 6, silently
  discarding five genuine parameter directions. At `rcond = 1e-2` it drops to 1.
- **The raw residual metric inverts the conclusion.** `‖Phi^T G̃‖` off the reference set
  reported a 5x degradation (`s3` T1) while the scale-free `rho` reported a genuine
  improvement from 0.4295 to 0.2843 (`s3b` B). The raw metric is dominated by the offset
  direction. Report `rho`.
- The protected span is **invariant to the parameter coordinate choice**: maximum
  principal angle between the physical, normalized and log spans is `2.98e-08 rad`
  (`s3` T6). Only the conditioning of the coefficient moves. In log coordinates the
  offset is `f(theta_bar)` itself, not `f - Phi theta_bar`; both give the same span.
- Expansion-point sensitivity is mild and roughly linear: maximum principal angle to the
  nominal span is `1.5e-03 rad` at a 1 percent parameter displacement, `7.3e-03` at 5
  percent, `2.9e-02` at 20 percent. So a reference refresh is not urgent, and a refresh
  policy can be bounded and reported rather than run every epoch.
- Reference-set size and scheduling coverage both help monotonically off the reference
  set: `rho` after projection falls 0.383, 0.398, 0.306, 0.290 at M = 32, 64, 256, 1024,
  and 0.306, 0.335, 0.353, 0.356 as the reference `Y` range narrows from `[-0.3, 0.3]` to
  frozen (`s3b` B, C). Control D, reference set equal to evaluation set, gives
  `7.3e-14`.
- A write that genuinely lies in the baseline span is removed at **any** point set, to
  `4.9e-14` (`s3b` E). So the off-reference gap is a generalization failure of a
  coefficient fitted to a write that is not in the span, not a defect of the projector.

**Correction to the source, established here.** G26's Eq. (40) displays the block
structure per data point, "for all data points in `D_N`". Measured on our prototype the
per-sample `‖phi_k^T J_f,k‖` has median `1.88e-03` while the stacked sum is `1.29e-13`
(`s3` T4). Only the sum vanishes. Theorem 18's conclusion is unaffected because only the
sum enters Eq. (38), but no claim of ours may be worded pointwise.

**Not established.** Whether the construction or the penalty wins under free-run
multiple-shooting training, which is the production estimator. Everything here is
one-step.

## Priority 4 - additional states and encoder

**Recommendation.** Adopt the lagged augmented rollout (R2) or the designed latent set
(R1) as the reference policy, never the baseline-only rollout. Never constrain the latent
rows. Evaluate any encoder-versus-parameter claim on a window of at least one yaw period.

**Evidence.**

- Proposition 7.1 of the derivation is reproduced on the real gantry regressor, not just
  on the surrogate. With a baseline-only reference set, `‖d eta_aux / d eta_lat‖` is
  **exactly** `0.000000e+00` across all 66 latent-path parameters, while
  `‖d eta_aux / d eta‖` is `4.00e+02`. Under R1 it is `3.83e+01` and under R2 it is
  `1.02e+02` (`s4` T1). Both repairs work; R2 gives the larger sensitivity because its
  latent values are larger (RMS 1.39 against 0.59).
- The blindness is a defect, not a harmless invariance: the same parameters move the
  learned **physical** write by `3.20e-02` wherever `x_a != 0`, and by exactly zero where
  `x_a = 0` (`s4` T2).
- Latent gauge invariance is exact: under `x_a -> V x_a` absorbed into the block weights,
  the physical write changes by `1.08e-18` and the latent write transforms as `V` demands
  to `3.55e-15` (`s4` T3). A condition constraining the latent rows would depend on an
  arbitrary coordinate choice.
- **The encoder result in `s4` T4 is superseded by `s4b` and must not be quoted.** It ran
  on a 200-sample window, 10 ms, and found 95 to 99.9 percent overlap between the
  initial-state span and every parameter tangent. The plant's yaw period, computed from
  the governing matrices, is **197 ms**, and its two free-integrator axes have viscous
  timescales of **1546 ms** and **1010 ms**. A 10 ms window contains none of them, so the
  overlap is a property of the window.
- On the corrected sweep the overlap falls with window length, and very unevenly
  (`s4b`):

  | window | median overlap with the initial-state span | min | the four most confounded |
  |---|---|---|---|
  | 200 samples, 10 ms | 0.9894 | 0.9546 | `m1` 0.999, `Jh` 0.998, `Jb` 0.998, `mb` 0.997 |
  | 1000 samples, 50 ms | 0.9704 | 0.5934 | `mh` 0.995, `mb` 0.993, `m1` 0.990, `cg1` 0.980 |
  | 4000 samples, 200 ms | **0.7525** | 0.6442 | `mh` 0.993, `cy` 0.982, `cg1` 0.979, `mb` 0.964 |

  So most parameters do separate from the initial state once the window reaches one yaw
  period, and the median falls from 0.99 to 0.75. But `mh`, `cy` and `cg1` stay above
  0.97 even at 200 ms. Those three are the ones an encoder can most nearly stand in for,
  and they are exactly the directions to watch in a joint encoder-plus-parameter run.
- **There is no exact dependency between the parameter span and the initial-state span.**
  At every window length `rank(dy/dtheta) = 10`, `rank(dy/dx0) = 6`, and
  `rank(joint) = 16 = 10 + 6`, so the two spans intersect only trivially. `s4` appeared
  to show a dependency (joint rank 15) purely because it applied a `1e-10` relative
  tolerance to a matrix with condition number `1.9e9`.

**Not established.** Whether a learnable encoder trained jointly actually exploits the
overlap, which needs a training run, not a sensitivity calculation. A nonzero sensitivity
is evidence of influence, not of failure. Also: the window sweep stops at one yaw period;
the two integrator axes have viscous timescales of 1546 ms and 1010 ms and were not
reached.

## Priority 5 - Györök's theorem assumptions for our training setting

**Recommendation.** Claim exactly three things and nothing more:
one-step non-overlap of the learned physical write with the affine baseline tangent set
over a frozen reference set, at any rank; exact block structure of the joint parameter
Jacobian over that same set, provided the coefficient is differentiated through; and
removal of the flat direction of the source's Example 2 at the level of the one-step
augmented map. Claim no consistency, no asymptotics, no exact parameter recovery, and
nothing at horizon level.

The full assumption-by-assumption matrix, with source anchors and a status per row, is
`gyorok-assumption-audit.md`. Three source corrections are established there: the missing
transpose in the prose after Eq. (8), the per-sample display in Eq. (40), and the
unnumbered "input independent of the noise" hypothesis in Theorem 12, which is the one
that closed-loop data violate and which is easy to miss when auditing the numbered
Conditions 10, 11 and 14 alone.

On marginal stability, the audit separates two things the project has previously
conflated. The open-loop plant has poles at `z = 1` because `K[0,0] = K[2,2] = 0`, which
is physical and must not be altered to make a theorem apply. The records are generated
under a stabilizing feedback loop, so Condition 10 may well hold for the data-generating
system *as operated* while failing for the isolated plant. Which object a condition
applies to has to be stated. Condition 11, the stable predictor, still fails, because our
predictor is a free-run rollout of the marginally stable augmented model. That kills the
asymptotic results and leaves the finite-horizon projection identities untouched, since
those never referred to a limit.

---

## Next implementation step, one action

Implement the construction of `state-level-obc-derivation.tex` Eq. (construction) in
`model_augmentation/fit_systems/` behind a flag, with the R2 lagged-rollout reference set,
the log-coordinate tangent-plus-offset basis, a column-scaled pseudo-inverse, the
coefficient differentiated through, and `rho` on both the reference set and the current
trajectory reported every validation. Validate it first on a synthetic residual whose
`rho(Phi; delta)` is measured and small, because the evidence above says that is the
condition under which the method does what it exists to do.

## What is not established anywhere in this investigation

- Anything under noise. Every number is noiseless float64.
- Anything at horizon level, or under free-run multiple-shooting training.
- Any statistical consistency claim. None is made, and none could be made from these
  tests.
- Any statement about real Telica data. No data were read.
- An independent MATLAB implementation of the projection prototype. The structural
  identifiability results do have two independent implementations behind them, Python
  here and the existing MATLAB Euler probe, agreeing to below `4e-08 rad`; the projection
  prototype has only one.
