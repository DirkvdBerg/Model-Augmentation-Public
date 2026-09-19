# When does orthogonal-by-construction recover the TRUE parameters?

Maarten Schoukens, 2026-09-18, raised the claim this folder answers: orthogonal projection makes
the split between baseline and augmentation **unique**, but it recovers the **true** physical
parameters only if the true missing dynamics are themselves orthogonal to the baseline's parameter
sensitivities. The claim is correct, it has a closed form, and the gantry data says how far from
orthogonal the hidden absorber actually is.

Two artefacts produce what follows, both in this folder:

| file | what it does |
|-|-|
| `derive_condition.py` / `.out` | the symbolic derivation, five results, every step machine-checked |
| `measure_delta_orthogonality.py` / `.out` / `.json` | the measurement on `augmentation_ma50_b140-230_a6_z03` |

This document states the result and connects it to
`documentation/obc-io-additional-state-chapter.tex` and `documentation/state-level-obc-derivation.tex`.

## 1. Notation

Stacked over the OBC reference set, sample-major, restricted to the protected rows:

| symbol | meaning |
|-|-|
| `f_base(theta)` | the discrete one-step baseline map, a vector in `R^n`, `n = N * n_protected` |
| `theta*` | the true physical parameter vector, `R^p` |
| `Delta*` | the true missing dynamics, defined by `f_truth = f_base(theta*) + Delta*` |
| `J` | `d f_base / d theta` at the expansion point, `R^{n x p}` |
| `P = J J^+` | the orthogonal projector onto `range(J)` |
| `f_ann` | the learning component on the same reference set |
| `dtheta` | `theta_hat - theta*`, the parameter bias |

The OBC constraint is `J^T f_ann = 0`. This is the same `J` the production path builds, through
`build_stacked_sensitivity` in `model_augmentation/fit_systems/obc.py`, and the same inner product
`OBCBasis` projects in.

## 2. The condition

**R1, the claim itself.** With `f_base` affine in `theta` and the model interpolating the truth,

```
J dtheta + f_ann = Delta*,    J^T f_ann = 0
  =>  (J^T J) dtheta = J^T Delta*
  =>  dtheta = J^+ Delta*     and     f_ann = (I - P) Delta*
```

With `J` of full column rank the pair is unique, so **projection does deliver uniqueness**. And

```
theta_hat = theta*    <==>    J^T Delta* = 0    <==>    Delta* PERPENDICULAR range(J)
```

Uniqueness is not correctness. The split is always well defined; it is correct only under the
orthogonality condition. The natural scalar summary is

```
rho* = ||P Delta*|| / ||Delta*||  in [0, 1],     rho* = 0  <==>  no parameter bias.
```

`rho*` alone does not fix the size of the bias, because
`rho* ||Delta*|| / sigma_max(J) <= ||dtheta|| <= rho* ||Delta*|| / sigma_min(J)`: the conditioning
of `J` enters too. That is why the measurement reports `J^+ Delta*` itself and not only `rho*`.

**R2, and this is the result that makes the prediction usable.** R1 assumed exact interpolation,
which a three-layer, 24-node network cannot deliver. Under **least squares** with `f_ann`
restricted to any subspace `A` of `range(J)^perp`, the objective splits by Pythagoras,

```
|| J dtheta + f_ann - Delta* ||^2  =  || J dtheta - P Delta* ||^2  +  || f_ann - (I - P) Delta* ||^2
```

and the two terms share no variable. So `dtheta = J^+ Delta*` **exactly**, whatever the network can
represent. Finite capacity degrades the fit and leaves the parameter bias untouched. `J^+ Delta*`
is therefore a **prediction**, not a lower bound, which corrects the assumption carried in the
handoff.

**R3, curvature.** With `f_base` nonlinear in `theta`,
`dtheta = J^+ Delta* - 1/2 J^+ H[J^+ Delta*, J^+ Delta*] + O(||Delta*||^3)`. Nine of the ten gantry
coordinates are logarithmic, where the second derivative equals the first, so the relative
remainder is about `||dtheta||/2`: at the run's 9.5 percent detuning that is a few percent **of**
the bias, not a few percentage points.

**R4, the expansion point.** In training the basis is refreshed at the current parameter point
`theta_0`, not at the unknown `theta*`. The projection becomes oblique and
`dtheta = (J_0^T J)^-1 J_0^T Delta*`. The vanishing condition becomes `J_0^T Delta* = 0`, a
condition on the basis actually used.

**R5, rank.** With `rank(J) < p` the bias is fixed only modulo `ker(J)`. Not active here: the
ten-combination reduction (D-190) gives full rank 10.

Every one of these is machine-checked in `derive_condition.py`; the printed output is
`derive_condition.out`.

## 3. How `Delta*` is built, and why you can believe it

`Delta*` is **recorded, not simulated**. The reference tuples already carry the truth's one-step
map: `Z_step` is the normalised block input at sample `k`, and the record carries the positions at
`k+1`. So

```
Delta* = (x_next_phys - x_mean) / std_x  -  step_base(theta*, Z_step)     on rows 0..5
```

Three choices decide whether this number means anything, and each is settled rather than assumed.

**Units (D-202).** `transition_from_free` consumes the block's normalised input and returns the
**normalised** next state, so `J` has normalised rows. `Delta*` is formed in the same frame. The
six protected rows span five decades of `std_x`, so mixing physical and normalised would silently
rescale every row of the inner product and make `rho*` meaningless. Normalising the truth side,
rather than denormalising the model, is what makes the measured projection the projection training
actually enforces.

**Stencil (D-203).** `x_phys` in the reference set gets its velocities from a fourth-order central
difference; `x_logical` in the record gets its velocities from a MATLAB `gradient()` of
single-precision positions (D-197). Differencing the two puts the **difference of two stencils**
into the velocity rows, which are exactly the rows the absorber acts on. The primary `Delta*`
therefore builds the successor as `[q[k+1]; D4 q[k+1]]`, one stencil on both sides, still purely
data-derived. The handoff's `recorded` variant is computed and reported beside it. They agree to
four digits in norm, and the one place they disagree is instructive: see section 5.

**The null test.** The baseline block is rolled forward over 56 windows of 400 steps from recorded
states, its exact velocities are thrown away, they are rebuilt with the same stencil from its own
positions, and the same one-step residual is formed. A trajectory that **is** a baseline trajectory
must give a residual at the reconstruction floor. Measured floor, normalised RMS:

```
8.7494e-04  (positions quantised to float32, the realistic case)
8.7455e-04  (float64 positions)
```

against a per-element RMS of `Delta*` of `8.45e-03`: a factor 9.7. More usefully, the **bias the
floor alone would predict** is `0.74 %` combo-err, against the `4.96 %` predicted from `Delta*`.
The answer clears its own floor by a factor near seven. That the two `Delta*` variants agree in the
position rows also retires the handoff's off-by-one worry: a one-sample misalignment would have
corrupted the position rows first, and it did not.

## 4. The measurement

Reference set: 671944 tuples, 14 records, stride 1, the set the runs used.
`J` is `4031664 x 10`, rank 10 of 10, condition `7.664e+02` at `theta*` and `8.925e+02` at the
detuned start (the latter reproduces the epoch-0 number logged by run 84033, `8.9250e+02`, exactly).

### rho*

| `J` at | `Delta*` variant | `rho*` |
|-|-|-|
| `theta*` | stencil | **0.2046** |
| `theta*` | recorded | 0.2067 |
| `theta_0` (detuned) | stencil | 0.2047 |
| `theta_0` (detuned) | recorded | 0.2067 |

**The hidden absorber is NOT orthogonal to the baseline's parameter sensitivities.** About 20
percent of the norm of the true added dynamics lies inside the ten-dimensional span the physical
parameters can reach. It is also far from fully aligned: 80 percent of it is genuinely new
dynamics that no parameter setting can imitate, which is why the augmentation has something to do.

`rho*` is insensitive to the expansion point (`0.2046` against `0.2047`, a 10 percent detuning) and
to the `Delta*` variant, and it is stable under subsampling of the reference set (0.199 at stride
97, 0.209 at stride 997). It is the robust half of the answer.

### The predicted bias

| prediction | combo-err |
|-|-|
| `J` at `theta*`, `Delta*` stencil (**primary**) | **4.96 %** |
| `J` at `theta_0`, `Delta*` stencil | 4.76 % |
| oblique, `(J_0^T J)^-1 J_0^T Delta*` (R4) | 4.92 % |
| `J` at `theta*`, `Delta*` recorded | 13.52 % |
| numerical floor alone | 0.74 % |

against the observed `obc` arm: start **9.49 %**, minimum **6.98 %**, final **7.69 %**.

The expansion point barely matters (4.76 to 4.96), so R4's oblique correction is not where the
uncertainty lives. The `recorded` variant's 13.52 percent is a **stencil artefact**, not a rival
answer: it is carried almost entirely by one parameter, `cb_sum`, which moves from `+40.5` to
`+7.9` percent when the stencil is made consistent while no other parameter moves by more than a
point. `cb_sum` is the yaw damping, and the yaw velocity row is where the two stencils differ most.

### Where the bias comes from: one direction, and a structural reason for it

Decomposing `dtheta = sum_i (u_i^T Delta* / sigma_i) v_i` over the singular directions of `J`:

| i | `sigma_i` | `\|u_i^T Delta*\|` | share of `\|\|dtheta\|\|^2` | dominant column |
|-|-|-|-|-|
| 0 | 2.15e+02 | 2.20e+00 | 0.0 % | J_eff |
| 3 | 1.15e+01 | 8.82e-01 | 0.5 % | m_total |
| **6** | **2.37e+00** | **2.51e+00** | **98.8 %** | **m_diff** |
| 7 | 9.23e-01 | 5.91e-02 | 0.4 % | cb_sum |
| 9 | 2.81e-01 | 1.56e-02 | 0.3 % | cg1 |

(the six omitted rows are all below 0.1 percent; the full table is in `delta_orthogonality.out`).

**The bias is 98.8 percent one direction, and it is not the ill-conditioned one.** Direction 6 sits
at `sigma = 2.37`, eight times the smallest singular value, and it also carries the single largest
overlap with `Delta*` of any direction. So this is not a small perturbation amplified by bad
conditioning; it is a large, genuine alignment between the missing dynamics and one parameter
direction. In free coordinates that direction is almost purely `m_diff` (component `+1.055`, next
largest `mh` at `+0.081`).

There is an exact structural reason, visible in the extended plant's mass matrix
(`scripts/gantry/transient/code/truth.py:104-106`, verified):

```python
    def M(self, Y, da):
        ma, mhr, L0 = self.ma, self.mh_rigid, self.L0
        off = (m1 - m2) * Lb / 2 - (mhr + ma) * Y - ma * L0 - ma * da
```

The absorber's equilibrium offset contributes `- ma * L0` to the **same** mass-matrix entry that
`(m1 - m2) * Lb / 2` occupies. A constant offset there is **structurally indistinguishable from a
mass imbalance**: at `ma = 5.05 kg` and `L0 = 0.10 m` it is worth
`-2 ma L0 / Lb = -1.393 kg` of `m_diff`. The projection cannot see the difference, so it books part
of the absorber as a rail-mass imbalance. The fit does not take the whole static offset (the
absorber's dynamic part pulls the other way): the predicted `m_diff` is `-1.080 kg` against a truth
of `-0.500 kg`, between the true value and the `-1.893 kg` the static offset alone would imply.

**This is the concrete answer to the supervisor's question.** The condition `J^T Delta* = 0` fails
here for an identifiable structural reason, not by accident of the data: a hidden mass mounted at a
fixed offset enters the baseline's own inertia parameterisation.

**A caution about how this is reported.** In the `combo-err` meter `m_diff` is scaled by the mean
actuator mass, `10.45 kg`, rather than by its own nominal `0.5 kg` (D-034 convention, `training.py`).
So a 116 percent error on `m_diff` reads as `-5.55 %`. The aggregate `4.96 %` therefore
**understates** what the condition does to the one parameter it actually corrupts. Any thesis claim
about parameter interpretability under OBC should quote `m_diff` on its own scale.

Per parameter, primary prediction against the observed final iterate:

| param | predicted % | observed % | sign |
|-|-|-|-|
| kb_sum | -0.07 | +15.52 | no |
| cg1 | -3.36 | -2.43 | yes |
| cg2 | -2.48 | -5.26 | yes |
| cy | +1.66 | +2.33 | yes |
| cb_sum | +7.86 | -9.63 | no |
| mh | +8.39 | -2.35 | no |
| m_total | +7.85 | +0.99 | yes |
| m_diff | -5.55 | -1.46 | yes |
| J_eff | +0.60 | -4.05 | no |
| d | -0.61 | +13.90 | no |

## 5. What this says about the completed run

Against the acceptance criterion the handoff set, the result is **inconclusive by the letter and
informative in a specific direction**. The predicted `4.96 %` is neither inside the `[6.98, 7.69]`
band that would have said "the arm converged where theory says it must", nor below the `2 %` that
would have ruled the explanation out. Three things follow, in decreasing order of confidence.

**1. The interpretability claim needs the condition attached to it.** `rho* = 0.20` is a measured,
a-priori, data-derived number that says this augmentation problem is **not** one where OBC can
return the true parameters. The projection buys uniqueness unconditionally and correctness only
under a condition this system does not satisfy. That is a testable criterion the thesis did not
previously have, and it is computable before any training run: it needs only the reference set, the
baseline sensitivity and a one-step residual.

**2. The predicted bias accounts for a large part of the residual error, but not the shape of it.**
`4.96` against an observed `6.98` minimum is `(4.96/6.98)^2 = 50 %` of the squared error. But the
per-parameter signs agree on only 5 of 10, and the two biggest observed errors (`kb_sum +15.5`,
`d +13.9`) are predicted at essentially zero, while the parameter the prediction says is wrecked,
`m_diff`, is one the run barely moved. So the aggregate magnitude is in the right range and
the direction is not. An explanation that got the direction right would be much stronger evidence
than one that got only the RMS right.

**3. The run is not at the OBC optimum, so the observed error should not be read as the bias.**
The RMS distance from the predicted optimum went from `11.48 %` at the detuned start to `9.85 %`
at the final iterate: the parameters closed **14 percent** of that distance, against 19 percent of
the distance to the truth. Both are small, and they are small for the same reason, which is the
timescale separation already on record: at epoch
150 the network had completed 100 percent of its fit improvement while the parameters had removed
24 percent of their initial error. The observed `7.69 %` is therefore dominated by **unremoved
initial detuning**, not by the projection bias. This is the reading the `lr_theta` experiment is
meant to test, and this analysis sharpens the prediction it should be judged against: with the
parameters allowed to converge, an OBC arm should go to roughly `5 %`, not to zero.

## 6. Caveats, in order of size

1. **The objective is not the one the identity assumes.** R2's separation argument needs the
   least-squares objective to be the one-step residual on the reference set in the normalised
   inner product. The runs minimise a multi-step **closed-loop** rollout, weighted per channel by
   `1 / ystd^2`. The two weight samples and rows differently, so the effective inner product is not
   the one OBC orthogonalises in, and the equality in R2 becomes an approximation of unquantified
   size. This is the largest gap between the prediction and the experiment.
2. **`Delta*` is a trajectory-realised signal, not a function of `(x, u)`.** It depends on
   `delta_a`, which is not part of the reference tuple. So "orthogonal" here is a statement about
   the realised missing dynamics on this reference set, not about an abstract function. This is the
   subject of `documentation/obc-io-additional-state-chapter.tex`, and the derivation agrees with
   it: the additional state is exactly why `Delta*` cannot be written as a static field on `z`, and
   why the augmentation needs its own dynamics rather than a static map.
3. **`f_ann` in the theory is the ANN's static contribution to the protected rows at `x_a = 0`.**
   In the model the augmentation is dynamic, carries eight states, and routes to all fourteen rows.
   The OBC constraint applies to the reference-set slice only. The identity therefore describes the
   constrained slice, not the whole augmentation.
4. **`J^+ Delta*` is sensitive to subsampling of the reference set, `rho*` is not.** At stride 97
   the predicted bias is `9.40 %` and at stride 997 it is `28.2 %`, while `rho*` stays at 0.199 to
   0.209. The swing is carried by the `cg1`/`cg2` pair, whose sensitivity columns are 97.7 percent
   correlated, and it is a SAMPLING effect rather than a numerical one: `u_i^T Delta*` is a sample
   average over the reference set, and a 0.1 percent subsample estimates it badly on nearly
   degenerate directions. The stride-1 solve itself is well conditioned (`cond(J) = 766`, and the
   direction carrying 98.8 percent of the bias sits at eight times `sigma_min`), so the stride-1
   number is robust. Stride 1 is also the set the training constraint actually used, which is what
   makes it the number that applies. Do not quote a strided number as the answer.
5. **Curvature** (R3) is a few percent of the bias, not of the parameter, and cannot reconcile
   `4.96` with `6.98`.

## 7. Reproduce

```
cd scripts/gantry/orthogonal-by-construction/baseline-and-added-dynamics
conda run -n GraduationProject python -u derive_condition.py                 # seconds
PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 OBC_REF_STRIDE=1 \
  conda run --no-capture-output -n GraduationProject python -u measure_delta_orthogonality.py
```

The second takes about three minutes on one GPU and writes `delta_orthogonality.out` and
`delta_orthogonality.json`. `OBC_REF_STRIDE` subsamples the reference set for a quick check; see
caveat 4 before reading a strided number as the answer. The script asserts its own calibration
against `figures/OBC/obc_data.json` iteration 0 before computing anything, so a convention drift in
the percentage meter fails loudly rather than silently.

## 8. PROPOSED: an experiment design that enforces `Jᵀ Δ* = 0`

**Status: proposed, not executed.** Nothing below has been run. It is recorded here because it is
the constructive answer to the measurement in sections 4 and 5, and because the supervisor asked
(2026-09-18) for a realisation in which the condition holds by construction rather than by luck.

Sections 4 and 5 measured `rho* = 0.205` on the existing dataset and concluded the condition fails.
The purpose here is the opposite direction: build a dataset in which it holds, so that the method
can be validated where theory says it must succeed. A failure on such a dataset is then a failure
of the implementation or the optimiser, never of the plant.

### 8.1 The lever is the excitation, not the plant

Keep the hidden MSD exactly as it is and design the multisine that drives it. On a standstill
record `Y` is fixed, `M(Y)` is constant, and the baseline is exactly LTI. Both sides of the inner
product are then linear filters of the same input, `R_j(w) U(w)` for the sensitivity column and
`F_a(w) U(w)` for the added dynamics, with `R_j` and `F_a` available in closed form from the
baseline and absorber models. The multisine sits on exact FFT bins (`gtd_make_multisine` already
guarantees this), so distinct lines are exactly orthogonal over the record and

```
<J_j , Delta*>  =  SUM over lines  c_j(w) p(w),      c_j(w) = Re[ R_j(w)^H W F_a(w) ],   W = M^-T M^-1
```

with `p(w) = |U(w)|^2` the power spectrum. **The inner product is linear in the power spectrum and
independent of the phases.** So `Jᵀ Δ* = 0` is ten linear equations in a few hundred nonnegative
line amplitudes: a wide feasibility problem, not an overdetermined one.

### 8.2 The design problem

Choose `p(w) >= 0` per line and per logical channel such that

| | constraint | why |
|-|-|-|
| 1 | `SUM c_j(w) p(w) = 0`, `j = 1..10` | the orthogonality condition itself |
| 2 | `\|\|Delta*\|\|` above a floor | the augmentation must still have something to learn |
| 3 | `cond(J)` below a ceiling | the parameters must stay identifiable |
| 4 | force RMS and peak within `cfg.lim` | the existing hardware-limit gate |

Ten equalities plus inequalities on the nonnegative orthant: a linear program. Its solution is a
designed amplitude spectrum. Everything else about the dataset generation is unchanged, which is
what keeps the realisation physically meaningful: the same Simulink model, the same hidden MSD,
the same `P` transform and controller, only a different multisine amplitude design.

### 8.3 Prerequisite: `L0 = 0`

Do this first, independently of the LP. Section 5 identified `ma * L0` as a constant sitting in the
same mass-matrix entry as `(m1 - m2) * Lb / 2`, carrying 98.8 percent of the predicted bias. Setting
`L0 = 0` deletes it. Mounting the absorber at the payload's rotation centre is physically natural,
it is one line in `gtd_config.m`, and it removes the largest structural coupling before the LP has
to work around it. What remains in that entry is `-ma * da`, which is zero-mean once `L0 = 0`.

### 8.4 What the construction does not give

1. **Exactness needs frozen `Y`.** Build the orthogonal dataset from standstill records at several
   `Y` values, one LTI system each, with the amplitudes of each record as separate design
   variables. Y-sweep and Lissajous records can stay in the set as a held-out family, outside the
   guarantee, which is arguably the right place for them anyway.
2. **The `ma * da` term is a genuine weak nonlinearity**, so the construction is exact only to that
   order. The residual is directly measurable with `measure_delta_orthogonality.py`.
3. **`c_j` must be computed over the EXACT reference-set window**, the trimmed hold-plus-active
   record the tuples are actually built from, not over idealised whole periods. Otherwise the hold
   segments leak and exactness is lost.
4. **The operative condition during training is `J_0ᵀ Δ* = 0` at the refresh point**, not at
   `theta*` (R4). The construction makes `theta*` a fixed point of the epoch-refreshed iteration
   and the scheme locally consistent; it does not zero the bias from a detuned start. Measured on
   the current dataset the two spans are close (4.96 against 4.76 percent), so this is a small
   correction rather than a blocker.

### 8.5 The cheap decisive step, before any regeneration

`design_orthogonal_excitation.py`: compute `c_j(w)` on the exact reference-set window from the
closed-form baseline and absorber models at `L0 = 0`, solve the LP for the line amplitudes, and
report the **predicted `rho*` and `J⁺ Δ*` for the designed spectrum**. If the predicted `rho*`
lands at the numerical floor established in section 3, the design justifies regenerating the
dataset in MATLAB. If it does not, that is learned for the cost of one script and the constraint
set can be adjusted. This reuses the measurement machinery already in this folder and needs no new
data.
