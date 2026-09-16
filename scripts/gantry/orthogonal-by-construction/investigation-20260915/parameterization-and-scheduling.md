# Baseline parameterization, scheduling and identifiable combinations

Everything here is produced by `s12_parameterization_and_rank.py` and reproduced in
`results/s12_parameterization_and_rank.log` and `results/s12_tables.json`. The model is
read from source, not assumed: `model_augmentation/systems/gantry_ss.py` lines 70 to 128
for the matrices, `model_augmentation/fit_systems/blocks.py` lines 692 to 1082 for the
discrete transition and the parameterization.

## 1. The governing map

In logical coordinates `q = (q1, q2, q3)`, state `x = (q, qdot)`, stage input `u`:

    M(Y) qddot  +  C qdot  +  K q  =  P u,        Y = q3 = x[2]
    M(Y) = M0 + M1 Y + M2 Y^2
    y    = [ P^T   0 ] x

with, writing `alpha = m1+m2+mb+mh`, `beta = (m1-m2) Lb/2`, `gamma = Jb+Jh+(m1+m2) Lb^2/4`,

    M0 = [ alpha            beta                   0      ]
         [ beta             gamma + mh d^2        -mh d   ]
         [ 0               -mh d                   mh     ]

    M1 = [ 0     -mh    0 ]        M2 = [ 0    0     0 ]
         [-mh     0     0 ]             [ 0    mh    0 ]
         [ 0      0     0 ]             [ 0    0     0 ]

    C  = [ cg1+cg2              (cg1-cg2) Lb/2                     0   ]
         [ (cg1-cg2) Lb/2       cb1+cb2 + (cg1+cg2) Lb^2/4         0   ]
         [ 0                     0                                 cy  ]

    K  = [ 0    0          0 ]      P = [ 1        1        0 ]
         [ 0    kb1+kb2    0 ]          [ Lb/2    -Lb/2     0 ]
         [ 0    0          0 ]          [ 0        0        1 ]

Three objects must be kept apart and are kept apart throughout this investigation.

1. **The CT dynamics**, the display above.
2. **The discretization**: RK4 with `up_sample = 10` substeps of `Ts/10`,
   `Ts = 1/20000 s`. The scheduling variable is re-read from the current state at every
   RK4 stage and every substep (`blocks.py:778-784`, `blocks.py:825`), so the discrete
   map is not the RK4 integral of any frozen-`Y` vector field.
3. **The measured-output predictor**: `y = P^T q`, positions only, 3 of the 6 states.

The standalone float64 reimplementation used for every experiment agrees with the
production `Gantry_State_Block` to a relative 2-norm of `1.75e-07`, which is the float32
precision of the production block. `M(Y)` is positive definite over the whole documented
scheduling range, minimum eigenvalue `3.736 kg` at `Y = +0.006 m`, so the LFR
interconnection is well posed there.

The documented scheduling range is `Y in [-0.30, +0.30] m`, read from `RECORD_Y_OP` in
`scripts/gantry/gantry_dynamic/controller.py:151-161`, not invented.

## 2. Is there an exact linear-in-parameters representation?

### The explicit transition: no

`det M(Y)` was expanded symbolically and depends on `Jb, Jh, Lb, m1, m2, mb, mh`
(S1a). Every velocity row of the explicit map is therefore a ratio of polynomials whose
**denominator carries the unknowns**. An LFR factorization rewrites `M(Y)^{-1}` as a
feedback interconnection; it does not remove the parameters from the denominator. So:

> There is no exact representation of `x(k+1) = f_theta(x(k), u(k))` that is linear in a
> finite set of unknown parameter combinations. Any linear-in-parameters surrogate for
> the explicit map is a local expansion, and the Györök construction can only be applied
> against a tangent set at a declared expansion point.

Cross-check performed in the same run: `det M(0) - mh (alpha gamma - beta^2) = 0`
exactly, confirming that the `gamma` convention of `build_poly_constants` (which excludes
`mh d^2`) reproduces `det M0`.

### The implicit force balance: yes, with a named redundancy

The residual

    r = M(Y) qddot + C qdot + K q - P u

is **exactly affine** in the 11 lumped coefficients

    A = alpha,  B = beta,  G = gamma,  MH = mh,  MD = mh d,  MD2 = mh d^2,
    CG1 = cg1,  CG2 = cg2,  CBs = cb1+cb2,  CY = cy,  KBs = kb1+kb2

Verified symbolically: rebuilding `r` from these symbols and substituting back gives
`[0, 0, 0]` identically, and every second derivative with respect to every lumped
coefficient is zero (S1b).

The embedding is exact but **redundant**: 11 coefficients carry 10 degrees of freedom.
The admissibility constraints are

    MD^2 = MH * MD2                     (because MD = mh d and MD2 = mh d^2)
    MH is the SAME scalar in M0[2,2], M1[0,1], M1[1,0] and M2[1,1]
    A > 0, G > 0, MH > 0, and M(Y) > 0 over the scheduling range

Estimating `MD` and `MD2` as free coefficients breaks the first constraint and leaves the
physical `(mh, d)` unrecoverable. That is the price of the exact linear form, and it must
be stated rather than discovered later.

**The estimator changes, and not only the parameterization.** `r` is `M(Y)` times the
acceleration error, so least squares on `r` weights the three logical channels by `M(Y)`,
which depends on the unknowns and on the scheduling variable. It also needs `qddot`,
which the pipeline does not measure; it would have to be differenced from position, which
is an errors-in-variables problem with a noise amplification of `1/Ts^2`. Clearing the
denominator is not an equivalent estimator, and the estimand is not output prediction.

## 3. Scheduling conventions

Three conventions are genuinely different objects, measured on 24 designed points
covering the documented range (S1c), against a reference scale of
`max |x(k+1) - x(k)| = 1.742e-03`:

| convention | `max |x_next(endo) - x_next(this)|` | relative to the state increment |
|---|---|---|
| frozen `Y = 0.00` | 2.295e-04 | 0.132 |
| frozen `Y = 0.15` | 1.557e-04 | 0.089 |
| frozen `Y = 0.30` | 2.081e-04 | 0.120 |
| exogenous `Y` held at the initial `x[2]` | 1.326e-08 | 7.6e-06 |

The float64 round-off floor of the same computation repeated is exactly `0.00e+00`, so
none of these is numerical noise. The exogenous row is small but not zero because the
endogenous convention re-reads `Y` inside every RK4 stage while a measured `Y` is held
across the sample.

**Frozen scheduling is a diagnostic, not a substitute for the plant.** A frozen-`Y` model
differs from the self-scheduled one by more than a tenth of the state increment.

### Can a scheduling mismatch mimic a parameter mismatch?

No, and this is the more useful half of the result. The frozen-`Y` transition error was
projected onto the span of the 14 parameter tangents at the same points:

    fraction explained by the parameter tangent span  =  0.108
    residual 2-norm / total                           =  0.944

So roughly 89 percent of a frozen-scheduling error lies **outside** everything the
physical parameters can produce. Two consequences.

- A scheduling-convention error cannot be absorbed into the physical parameters, so it
  will not corrupt them; it will appear as model error.
- Because it lies outside the protected span, **the learned component is free to absorb
  it**. If the reference set is built with frozen scheduling, the augmentation can learn
  the scheduling convention rather than the physics, and the projection will not charge
  it for doing so. This is a concrete argument for building the reference set with the
  same endogenous scheduling the plant uses, and for spanning the operating range.

## 4. Identifiable parameter combinations

### The result

The map `theta (14 raw scalars) -> (M0, M1, M2, C, K)` factors **exactly** through 10
quantities, which are precisely the ones already coded in
`Parameterized_Gantry_State_Block._combos_from_raw` (`blocks.py:1045-1058`):

    kb_sum = kb1 + kb2
    cg1, cg2, cy
    cb_sum = cb1 + cb2
    mh
    m_total = m1 + m2 + mb
    m_diff  = m1 - m2                                     (signed)
    J_eff   = Jb + Jh + (m1 + m2) Lb^2 / 4
    d

The stacked parameter Jacobian of the **RK4** transition over 24 designed points has
rank **10** and nullity **4**. The rank is stable at every relative tolerance from `1e-4`
to `1e-14`; the gap between `sigma_9` and `sigma_10` is a factor `7.9e12`, so this is a
structural statement, not a threshold artifact.

    sigma = 5.07e-03  1.27e-03  1.22e-04  5.91e-05  1.08e-05  8.92e-06  6.79e-06
            2.09e-06  1.69e-06  5.42e-07 | 6.84e-20  2.46e-20  7.09e-22  1.07e-37

The four null directions are, in raw coordinates,

    kb1 - kb2
    cb1 - cb2
    -m1/2 - m2/2 + mb + (Lb^2/4) Jb
    -m1/2 - m2/2 + mb + (Lb^2/4) Jh

whose difference is `Jb - Jh`. The maximum principal angle between the predicted and the
numerically computed null space is `2.58e-08 rad`, and it stays below `4e-08 rad` at
three admissible nominal parameter sets and two independent reference grids.

Exact invariance was tested directly, not inferred from an SVD: perturbing along each
null direction at displacements up to `1e-1` leaves `M0, M1, M2, C, K`, the full RK4
transition at 16 independent points, and all 10 combinations unchanged to
`0.00e+00` or `4.44e-16`. The converse holds too: expressed in the 10-combination
coordinates the one-step map has full rank 10 with condition number `1.33e4`, so every
combination produces an independent response.

Independent central differences agree with the autodiff Jacobian to `1.0e-08` relative at
a relative step of `1e-4`, degrading as expected for smaller steps.

### The discrepancy flagged in the derivation is a misreading, and is closed

`state-level-obc-derivation.tex` Sect. 9 closes with "An identifiability finding, to be
reconciled", stating that the null space has four directions rather than three and that
"the project's documented set of ten identifiable quantities lists the inertia sum on its
own", so that the two statements "should be reconciled before either is quoted".

There is nothing to reconcile. The project's coded quantity is

    "J_eff": p["Jb"] + p["Jh"] + (p["m1"] + p["m2"]) * Lb ** 2 / 4

(`blocks.py:1056`), which is exactly the combination the MATLAB Euler probe found. The
two agree, both give rank 10, and the four null directions of the Euler probe
(`code/logs/gantry_euler_regressor_structure.log`) are reproduced here at full RK4
fidelity to below `4e-08 rad`. The premise of the remark, that the project lists the bare
inertia sum, is false; it is the prose of the remark that needs correcting, not either
result.

### What IS wrong in the project's prose

`CLAUDE.md`, "Control Engineering Stance", item 3, states that "only `kb1+kb2`,
`cb1+cb2`, `Jb+Jh` are identifiable". The third of those is **not** identifiable on its
own: `Jb + Jh` can be traded against `m1 + m2` at fixed `m_total` along the fourth null
direction, and only `J_eff = Jb + Jh + (m1+m2) Lb^2/4` is invariant. The docstring of
`Parameterized_Gantry_State_Block` is correct; the CLAUDE.md sentence is not. Also, the
sentence lists three sums where there are four null directions, which is the trap the
handoff warned about. This is reported, not edited, because the file was not in scope.

### Coordinates

Rank is invariant, conditioning is not:

| coordinates | rank | condition number on the range | `sigma_0` | `sigma_9` |
|---|---|---|---|---|
| physical | 10 | 9.36e+03 | 5.07e-03 | 5.42e-07 |
| normalized `theta/theta_init` | 10 | 1.80e+02 | 2.80e-03 | 1.56e-05 |
| log `theta = theta_init exp(l)` | 10 | 1.80e+02 | 2.80e-03 | 1.56e-05 |

Log and normalized coordinates coincide exactly at the zero-initialized point, because
`d(theta_init e^l)/dl = theta_init` there, and both improve the conditioning by a factor
52 over physical coordinates. This supports the pipeline's existing log parameterization
(D-035) on conditioning grounds and not only on positivity grounds.

**Caveat on the signed difference.** `m_diff = m1 - m2` is signed and must never be log
transformed. In the current code it is not: `m1` and `m2` are each logged separately and
`m_diff` is a derived readout. That is the correct arrangement and should be kept.

## 5. Candidate table

| candidate | exact model match | linear in parameters | physical interpretation | measurements needed | estimator change | conditioning | why rejected or kept |
|---|---|---|---|---|---|---|---|
| Raw 14 scalars, explicit RK4 map (current) | exact | no, rational | direct, but 4 directions are pure gauge | `u, y`, encoder for `x0` | none | cond 1.80e+02 in log coords on the identifiable range; 4 exactly flat directions | **kept for training**, with reporting in the 10 combinations. The flat directions are held by the `param_loss` anchor, not identified. |
| Reduced 10 combinations, explicit RK4 map | exact, same map | no, rational | every coordinate is identifiable by construction | same | reparameterize the block only | cond 1.33e4 in combination coordinates, full rank 10 | **recommended** where the 4 flat directions matter. Costs a rewrite of `_build_KC` / `build_poly_constants` to take combinations; `mh` and `d` stay separate so `M1, M2` are unaffected. |
| 11 lumped coefficients, implicit force balance | exact | **yes** | loses `(mh, d)` individually unless `MD^2 = MH MD2` is imposed | `u`, `q`, and `qddot` | different loss, `M(Y)`-weighted, errors in variables | not measured here | **rejected as the primary estimator**: needs accelerations, changes the estimand away from output prediction, and reweights the residual by a parameter-dependent matrix. Retained as a **consistency check** and as the only route to an exactly linear regressor. |
| Frozen-`Y` LTI per operating point | not exact, 0.13 of the state increment | no | fine per point | same | none | same | **diagnostic only**. Differs from the plant by more than a tenth of the state increment, and 89 percent of that difference is outside the parameter span, so the augmentation would absorb it. |
| Exogenous / measured `Y` scheduling | near exact, 7.6e-06 relative | no | same | needs a measured `Y` | none | same | admissible where `Y` is measured; the residual gap is the intra-sample re-read only. |

## 6. What is not established here

- The **empirical** rank of the stacked measured-output sensitivity over long records,
  which is a different object from the one-step rank and is what training actually sees.
  Partially addressed in `s4b_encoder_window.py`; see `decision-report.md`.
- Multi-start parameter recovery in reduced versus raw coordinates. Not run; see
  `continuation.md`.
- Anything about noise. Every number above is noiseless float64.
