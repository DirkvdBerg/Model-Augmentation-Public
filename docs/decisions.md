# Design Decisions

Decisions are logged here before implementation. Each entry states what was decided, why, what was ruled out, and what it constrains going forward.

---

## Decision Template

```
### [D-NNN] Title
**Date**: YYYY-MM-DD
**What**: What was decided.
**Why**: The reason — constraint, evidence, or trade-off that drove the choice.
**Ruled out**: Alternatives considered and why they were rejected.
**Constrains**: What future decisions or implementations this locks in.
```

---

## Decisions

### [D-166] The 2 kHz and 1 kHz downsampling arms are rejected on the loop gate; the P2a table at four rates
**Date**: 2026-08-27

**What**: 1 kHz and 2 kHz are rejected as closed-loop training rates. 4 kHz (D-141) stands. The
decision rests on the phase-margin gate of `p2_rate_compare.py`, run for the first time with all
four arms (`CL_RATES=20000,4000,2000,1000`). No data-side test was needed, because the loop gate
already disqualifies both candidates.

**The P2a table** (frozen design loop, `f_bw = 100` Hz, reference 20 kHz). `sigma_max(So)`:

| `Y_op` | rate | 1 Hz | 10 Hz | 50 Hz | 100 Hz | 150 Hz | 180 Hz | 500 Hz |
|-|-|-|-|-|-|-|-|-|
| 0.10 | 20 kHz | 0.0004 | 0.0214 | 1.0544 | 1.6695 | 1.7983 | 1.8043 | 1.1438 |
| 0.10 | 4 kHz | 0.0004 | 0.0214 | 1.0816 | 1.8362 | 2.0738 | 2.0506 | 1.1230 |
| 0.10 | 2 kHz | 0.0004 | 0.0214 | 1.1153 | 2.1066 | 2.5867 | 2.4289 | 1.0603 |
| 0.10 | 1 kHz | 0.0004 | 0.0214 | 1.1793 | 2.9965 | 4.9401 | 2.9735 | n/a |
| 0.00 | 20 kHz | 0.0004 | 0.0217 | 1.0807 | 1.6569 | 1.7937 | 1.8076 | 1.1450 |
| 0.00 | 4 kHz | 0.0004 | 0.0217 | 1.1098 | 1.8277 | 2.0711 | 2.0590 | 1.1239 |
| 0.00 | 2 kHz | 0.0004 | 0.0217 | 1.1457 | 2.0990 | 2.5907 | 2.4484 | 1.0606 |
| 0.00 | 1 kHz | 0.0004 | 0.0217 | 1.2144 | 2.9865 | 5.0467 | 3.0175 | n/a |

The 500 Hz entry is `n/a` at 1 kHz because it is Nyquist. Phase margin and crossover, identical to
two decimals at both operating points:

| rate | PM [deg] (X1, X2, Y) | fc [Hz] | PM shift vs 20 kHz | gate (tol 5 deg) | `So` at 150 Hz vs 20 kHz | gate (tol 10 %) |
|-|-|-|-|-|-|-|
| 20 kHz | 37.43 / 37.42 / 37.16 | 100.00 | ref | ref | ref | ref |
| 4 kHz | 33.83 / 33.82 / 33.57 | 100.04 | 3.59 | ok | +15.3 % / +15.5 % | FLAG |
| 2 kHz | 29.35 / 29.34 / 29.08 | 100.17 | 8.08 | FLAG | +43.8 % / +44.4 % | FLAG |
| 1 kHz | 20.31 / 20.30 / 20.05 | 100.70 | 17.12 | FLAG | +174.7 % / +181.4 % | FLAG |

**Why the phase-margin gate is the decisive one and the `So` gate is not**: D-141 accepted 4 kHz
*while it flagged* at +15.3 %, explicitly as "a known, stated bias" rather than a hidden one, and
D-142 then measured the consequence and found it working in the augmentation's favour (the 77-79x
closed-loop headroom exists *because* `sigma_max(So) = 2.07` amplifies the absorber mismatch at
130-180 Hz). So the 10 % `So` tolerance has never functioned as an admissibility gate in this
project; it reports the size of a bias that is then priced separately. The 5 degree phase-margin
tolerance is the gate 4 kHz actually passed, and it is the one 2 kHz and 1 kHz fail, by 1.6x and
3.4x respectively.

**Why the degradation is exactly what it should be, which is what makes the rejection safe**: the
measured PM shifts of 3.59, 8.08 and 17.12 degrees are reproduced to two decimals by the ZOH
half-sample lag `180 * fc / fs` differenced against the 20 kHz arm: `180*100/4000 - 180*100/20000
= 3.6`, `9.0 - 0.9 = 8.1`, `18.0 - 0.9 = 17.1`. The loss is therefore pure sample-delay phase lag,
smooth and predictable, not a Tustin-warping pathology at the `10*w_b = 1000` Hz roll-off pole.
That pole's Tustin image walks `+0.729 -> +0.120 -> -0.222 -> -0.517` across the four rates, so at
2 kHz it sits on Nyquist and at 1 kHz above it, but that is not what breaks the loop. Reading the
rejection as a delay result rather than a discretisation-artefact result means no anti-alias or
alternative-discretisation fix recovers these rates: at `fc = 100` Hz, a 5 degree budget caps the
rate from below at roughly 3.5 kHz regardless of method.

**Why the `So` peak at 150 Hz is the second, independent reason**: 150 Hz is `cfg.fa`, the hidden
MSD frequency (`gtd_config.m:50`), i.e. exactly the mode the augmentation exists to learn. Going
from 1.80 (20 kHz) to 4.94 (1 kHz) changes by 2.75x how heavily the training objective weights
that mode relative to the loop that generated the data. Combined with D-142's finding that the
headroom is a waterbed effect at this frequency, changing the rate does not merely add numerical
error, it rescales the quantity being selected on.

**Ruled out**: (1) **Keeping `Cfb` at 20 kHz around a 1-2 kHz model.** `ClosedLoopSimulator` steps
`xc` once per model step, so this needs a multirate inner loop plus interpolated `y_model` and
`y_data` at 20 kHz inside every window, a real change to `closed_loop.py`. It is also aimed at the
wrong target: in the residual form `u_plant = u_data + Cfb*(y_data - y_model)` the recorded
`u_data` already carries the true 20 kHz feedback, so `Cfb` at the training rate is a correction
operator shaping the objective, not a reproduction of the data-generating controller. Matching it
exactly was never the requirement; matching its shaping over 130-180 Hz was, and that is what the
table measures. (2) **Re-tuning the controller at the lower rate** to recover phase margin. That
makes the training loop a different loop by design rather than by discretisation, and every
closed-loop number on record (D-139, D-140, D-142) is referenced to the frozen `ruleOfThumb`
design. (3) **The data-side aliasing and discretisation-floor tests**, not because they are wrong
but because they are now moot; they were the expensive gate and the cheap gate already closed the
question. Recorded here in case the rate question reopens: the current decimation is deliberately
naive (`y[::D]` point sampling, per-hold block mean on `u`, `gantry_dynamic/data.py:78-104`), and
its justification is a `frac_above` measured at the **2 kHz** Nyquist (4e-14 on `u`, 2.5e-8 on
`y`). That justification does not transfer to a 500 Hz Nyquist and the truth sim has Coulomb
friction, which is broadband, so any future sub-4 kHz proposal must re-measure it
(`scripts/gantry/augmentation-error/diag_downsample_spectra.py`) before point sampling is trusted.

**Constrains**: `cfg.fs_new = 4000` stays. Any future proposal to change the closed-loop training
rate runs `p2_rate_compare.py` with that rate in `CL_RATES` and clears the 5 degree phase-margin
tolerance *before* any training job is launched, and states its `sigma_max(So)` bias at 150 Hz as
D-141 did rather than treating the FLAG as a pass or a fail. The `4x` wall-clock saving that
motivated the question (`nf = nf_seconds / ts_new`, so 400 steps at 4 kHz against 100 at 1 kHz)
must be found somewhere other than the sample rate. One thing the table does NOT say: it is a
frozen-design-loop calculation on the baseline plant, so it prices the loop, not the data, and it
is silent on aliasing and on the 150 Hz mode's own discretisation. Those stay open questions for
any rate below 4 kHz, listed under "ruled out" above.

### [D-165] PLAN phase 5 gate 1 compares two different metrics; the `1e-6` tolerance was never achievable
**Date**: 2026-08-25

**What**: the D-072 reference `2.186601103417735e-06` is **deepSI's sim-RMS on
`data.val_ckpt_data`**, printed as `Initial Validation sim-RMS` at the start of `fit`. The number
every arm reports as its untrained score is `validation_metrics`' POOLED free run over the four
full V1-V4 records, `2.1865622e-06`. Those are different objects, and PLAN phase 5 gate 1 asks for
them to agree to `1e-6` relative. They differ by `1.78e-05`.

**Why it matters**: the gate is the attribution gate. Its job is to prove the transplant is a no-op
at initialisation, which it is: the offset is reproduced bit-for-bit on two independent launches
and is IDENTICAL at `nx_aug = 2` and `nx_aug = 8`, so it is a property of the metric pair and not
of the augmented block. Enforced literally at `1e-6` the gate refuses every correct run, which is
what a first launch of `run_bla_arm.py` did before the tolerance was corrected.

**What was done**: tolerance set to `1e-4`, with the deviation against BOTH references recorded in
every run summary (`d072_rel_dev` and `rel_dev_vs_arms_i_ii_iii`). `1e-4` is two orders above the
measured cross-metric offset and four orders below the effect the gate exists to catch, since a
readout that does not start at zero moves this metric by tens of percent.

**Ruled out**: changing the reference to a pooled value. It is the correct fix and it was not made,
because `2.186601103417735e-06` is quoted across `DISCUSSION-POINTS.md`, the run table and the
handoffs, and silently redefining it would invalidate the comparability those numbers exist for.

**Constrains**: anyone tightening this gate must first replace the reference with a pooled one and
say so. Quoting `2.186601103417735e-06` next to a pooled RMS without the note above is comparing
two metrics.

### [D-163] PLAN acceptance gate 3 is withdrawn as stated and replaced by cross-record agreement
**Date**: 2026-08-25

**What**: The handoff's gate 3, "the zero of the plant FRF must not move between the direct and
indirect estimates by more than one bin", is not a valid gate and is not implemented as one. What
is implemented instead: the dominant in-band pair is identified INDEPENDENTLY on five training
records with five different phase realisations and five different `Y`, and on one held-out record,
and the estimator passes only if they agree to within their own scatter.

**Why**: the gate watches a feature theory does not protect. In MIMO
`(P S)_{22} = sum_j P_{2j} S_{j2}`, which is not `P_{22} S_{22}`, so the zero of a single ELEMENT
is not feedback-invariant and the `149.0833 -> 147.0833 Hz` move that failed the gate is the
expected behaviour, not an estimator defect. The invariant that IS protected is the transmission
zero, `det(P S) = det(P) det(S)` with `det(S) = 1/det(I + P C)` having no zeros; but measured on
all six records, `|det G|` of the indirect estimate has its minimum at the band edge, i.e. the
baseline has no transmission zero inside `130-180 Hz` and the protected feature does not exist
here. A gate on a feature that either is not invariant or is not present cannot fail informatively.

**Measured on the replacement**: `157.9045 Hz` mean over the five training records, standard
deviation `0.0054 Hz`; `zeta` mean `0.052762`, standard deviation `0.000035`; held-out
`V1_standstill_Yp10` at `157.9035 Hz`, `0.0009 Hz` from the training mean, i.e. `0.2` training
standard deviations. Both quantities from the withdrawn gate are still reported in the artefact.

**Ruled out**: (1) Keeping the element-notch gate and tuning the estimator until it passes: it
would have driven the estimator away from correctness towards a MIMO artefact. (2) Gating on
`|det G|`: implemented first, then withdrawn when the minimum turned out to sit at the band edge on
every record.

**Constrains**: the estimator's validity claim now rests on reproducibility across records, so any
future run that uses fewer than two independent records cannot make it.

### [D-161] The pole identification is consistent ONLY because the simulation is noiseless; what breaks on Telica, and the remedies
**Date**: 2026-08-23
**Status**: FORWARD CONSTRAINT. Recorded before the noisy case is attempted, so the simulation result is not later mistaken for evidence of transfer.

**What**: `gantry_dynamic/pole_init.py` fits the closed-loop residual `r = y - y_baseline` by DIRECT
least squares on the logged `u`. That is consistent here and it is **not** consistent on real data.
Every claim about the identification made on the augmentation simulation is therefore a claim about
the noiseless case only, and none of it transfers without one of the remedies below.

**Why it is consistent here.** The direct method is biased in closed loop when the noise is
correlated with the input THROUGH the loop. The augmentation data is noiseless by default
(`snr=None`; output noise is opt-in per D-078), so `r` is deterministic model mismatch rather than
a noise realisation and that mechanism is absent. Measured 2026-08-23 on `T3_standstill_Y000`, top
modal pairs of the shared-denominator fit at every order `na` in `{8, 12, 16, 20, 24, 28}`:
**`157.89-157.90 Hz`, `zeta 0.0527-0.0528`**, against the plant's `158.1139 Hz` and `zeta_a = 0.05`
from `gtd_config.m`. That is `0.014 %` in frequency and `+5 %` in damping, at every order.

**A correction that is NOT needed here and was measured to hurt.** Regressing on the effective
closed-loop drive `u + C_fb(r)` instead of the logged `u` moves the estimate to **`163.55 Hz`,
`zeta 0.100`**, i.e. `3.4 %` off in frequency and twice the true damping. The "obvious" closed-loop
correction is actively harmful in the noiseless case. It was asserted as necessary twice in the
session that wrote this entry, both times without measurement, and both times wrongly.

**What breaks once noise is present**, each with what it costs and what it is evidenced by:

| # | failure | evidence |
|-|-|-|
| 1 | **Closed-loop bias of the direct method.** With measurement noise the input carries a component correlated with the residual through `C_fb`, and least squares becomes inconsistent. This is the classical result, not a conjecture | Ljung; Forssell and Ljung, "Closed-loop Identification Revisited", `literature/closed-loop-id/forssell1998_cl_revisited_liu2021.pdf` (report LiTH-ISY-R-2021; **mangled text layer, do not quote from it**) |
| 2 | **The residual stops being pure model mismatch.** On real data the baseline PARAMETERS are estimates from the recovery work, so `r` contains parameter mismatch as well as omitted dynamics, and the identified pole is a mixture | `cl_residual_spectrum.py` `residual_for` docstring states this as a known seam |
| 3 | **`x0` is not available** as a true state; positions are measured and velocities come from numerical differentiation. **MEASURED 2026-08-23 and it does NOT break: see the resolution below** | same docstring |
| 4 | **The fit may refuse outright.** `fit_reduce.py` already hits this on its noisy condition: out-of-sample free-run VAF `-0.0136`, i.e. worse than the zero predictor, reported as "the residual is not a linear object on this data" | D-159 |
| 5 | **Damping estimates degrade first, and damping is the memory ingredient.** Even noiseless, peak picking on the power spectrum is already `+16.6 %` on `zeta` because a half-power width carries the input spectrum's shape; noise widens peaks further and biases the same way | measured 2026-08-23, `cl_residual_spectrum.json` `0.05829` against `0.05` |
| 6 | **A BLA is not available as a fallback.** `pintelon2020_bla-feedback-process-noise` extends BLA theory to feedback AND process noise, so feedback is not the obstacle, but the technique rests on "specially designed periodic excitation signals called random phase multisines and periodic noise". Telica runs a jerk-limited point-to-point move of 40 mm X and 80 mm Y with `Y` sweeping inside every record, so the plant is not LTI over it, and the ILC iterations "differ only in feedforward" and are not realisations | `docs/kamtin-telica-schema.md`; abstract verified `MATCH OK` 2026-08-23 |
| 7 | **Excitation may not reach the modes at all.** A jerk-limited move concentrates its power below a few tens of Hz. If the unmodelled dynamics sit in the hundreds of Hz, no estimator recovers them because nothing excited them. This is an experiment-design problem and it bites before the estimator choice does | inference from the schema, NOT measured |

**Resolution of item 3, measured 2026-08-23, and it is a POSITIVE result.** `residual_for` seeds the
baseline rollout with `x_log[K0]`, the true logical state from the `.mat` file. That was the only
oracle dependency anywhere in the pole-placement chain. Measured on `T3_standstill_Y000`, identified
mode nearest the absorber, `n_pairs = 2`:

| `x0` construction | `f_d` [Hz] | `zeta` | `rho` | free-run VAF |
|-|-|-|-|-|
| oracle `x_log[K0]` (what runs today) | 157.8937 | 0.05276 | 0.986982 | 99.907 |
| **measured positions + central difference** | **157.8937** | **0.05276** | **0.986982** | 99.902 |
| measured positions + 5-point difference | 157.8937 | 0.05276 | 0.986982 | 99.902 |
| perturbed `1.1 * x_log[K0]` | 157.8939 | 0.05275 | 0.986983 | 99.906 |
| zero | 57.3640 | 0.45014 | 0.955594 | 99.889 |
| positions kept, velocities ZEROED | 136.5785 | 0.48511 | 0.887794 | 99.884 |

Identical to seven digits under the real-data construction, even though the differentiated velocity
is off by `76 %` on one channel. **So the chain is portable: it needs logged `u`, logged `y`, the
baseline model and the known controller, and no true state.**

Two things this also corrects. The docstring's argument, that an `x0` error "decays into the first
transient" so dropping `K0` samples suffices, is **false**: the `zero` and `velocities-zeroed` rows
are catastrophically wrong at `K0 = 17`, plausibly because the plant carries `K = 0` double
integrators and the controller has integrator poles at `|z| = 1` by design, so the correction
transient far outlasts 17 samples and the fit models the transient instead of the absorber. What is
true is the weaker and sufficient statement that a *reasonable* `x0` suffices; a `10 %` error is
harmless and differentiation is well inside that.

Remaining caveat, NOT measured: differentiation amplifies noise, so on real data the velocity
estimate degrades with the measurement noise this entry is about. The structural portability is
settled; the noisy case is not.

**Remedies, in the order they should be tried**:
(a) **Instrumental variables.** Already implemented as
`scripts/gantry/BLA-Augmentation/probe_d8_residual_fit.py::fit_shared_denominator_iv`, so this is a
wiring job rather than new code. IV is the standard first answer to correlated regressors.
(b) **Two-stage closed-loop identification.** `vandenhof-schrama-1993-automatica-twostage.pdf`,
Automatica 29(6):1523-1527, verified on disk: an indirect method that estimates the plant
consistently from closed-loop data "even in the situation where the model of the noise disturbance
on the data is not accurate". That last clause is the one that matters here, because we do not have
a noise model for Telica.
(c) **Dedicated experiments.** If (a) and (b) both refuse, the honest conclusion is that the data
does not support the identification, and the ask is machine time from ASMPT for periodic multisine
excitation at frozen operating points. That is an experiment-design request, not a modelling fix,
and it should be raised early rather than after (a) and (b) have failed.

**Ruled out**: (1) using `u + C_fb(r)` as the regressor, measured worse above; (2) treating the
noiseless result as evidence of transfer, which is the specific error this entry exists to prevent;
(3) switching to peak picking, which is worse on damping and gets worse under noise.

**Constrains**: (a) Any Telica arm must state which of (a)/(b)/(c) it used, and a Telica result
produced by the direct method is void. (b) D-160's estimator comparison table is labelled as
noiseless and must not be quoted for the real system. (c) The noise gate already exists as
`CL_RS_NOISE_SIGMA` in `cl_residual_spectrum.py`, so the cheapest next measurement is to re-run the
identification with data-derived noise injected and record where it breaks; that is a simulation
experiment and it should precede any Telica attempt.

### [D-160] The augmented-state initialisation follows Schoukens ECC 2021: an explicit live linear part, initialised from a linear approximation of the residual
**Date**: 2026-08-23
**Status**: SPECIFICATION. No arm below has run. Written before implementation per the standing rule.
**Supersedes**: the first version of D-160, written the same day, which framed this as "a BLA of the residual". That framing was wrong twice. What the repo computes is a parametric ARX/IV fit, not a BLA in the Pintelon-Schoukens sense; and the governing citation is not the BLA literature but `schoukens2021_improved-init-state-space-ANN_ECC` (`arXiv:2103.14516`), which specifies this exact construction.

**What**: The minimal addition to Jan's framework is one block whose STRUCTURE and INITIALISATION both come from Schoukens ECC 2021, the generalised residual state-space neural network (gR-SS-NN) of its Eq. (5):

    Schoukens Eq. (5):  x(k+1) = [A B] [x(k); u(k)] + W~x sigma(W~fx x + W~fu u + b~f) + b~x
    ours:               x_a[k+1] = A_aa x_a[k] + B_u z[k] + F(z[k])

Every element carries a quote verified `MATCH OK` by `verify_pdf_quote.py <pdf> any <quotefile>` on 2026-08-23:

| element | Schoukens ECC 2021, verified |
|-|-|
| the scheme as a whole | "Some of the neural network weights are initialized starting from a linear approximation of the nonlinear system, while others are initialized using random values or zeros." |
| **explicit live linear part** (`B_u`, our step 1) | "It is illustrated in this paper that, together with an improved initialization, the inclusion of the explicit linear part improves the estimation of the SS-NN model significantly." |
| **poles from an identification** (our step 3) | "The linear state-space matrices are directly used to initialize the A, B, C, D matrices in eq. (5)." |
| **zero output projection over random inner layers**, i.e. the harness default, and NO gate | "This paper proposes to do this the other way around, random weights in the nonlinear layer and zero weights in the linear layers, works better in the benchmark examples"; and the reason, "random weights and biases in the nonlinear layer generates a pool of nonlinearly transformed outputs which the estimator can pick from using the linear weights during optimization" |
| **`B_u` scaled to unit `x_a` std, measured** | "The state-space matrices of the linear approximate model are normalized such that each of the states has a standard deviation equal to 1" |

**Amendment 2026-08-23, same day, after implementation: `gamma` and `alpha` are removed.** The first
version of this entry carried Orvieto's `Gamma = sqrt(1 - rho^2)` on the input map and a Bachlechner
ReZero scalar gate. Both are redundant against the citation above and both are gone from the default
path.
* **`alpha`**: Schoukens Sec. IV.2 specifies a ZERO OUTPUT PROJECTION over random inner layers,
  which is exactly `zero_init_feed_forward_nn` and needs no gate. ReZero is strictly weaker here:
  it makes ONE scalar live at step one where the zero projection makes the whole of `W_out` live,
  and `dL/dW_out = <dL/dw, sigma(z)>` with `sigma(z)` containing `x_a`, so the readout starts
  learning to USE the augmented states at the first update and only because they are driven.
  ReZero was introduced for the D-130 `W^a` dead zone and does not fix that either: `W^a` is
  gradient-free at step one under both schemes. `GATE_ZERO` is retained only as a comparison arm.
  This also moves us TOWARD arm 2, which used the harness's zero projection.
* **`gamma`**: exact for a WHITE unit-variance input and our `z` is not white, being dominated by
  the `[0, 40] Hz` motion island plus the `[130, 180] Hz` excitation island (`cl_band_split.py` off
  `gtd_config.m`). It is also nearly vacuous once the poles come from an identification, since every
  pair then carries the same damping. Replaced by `empirical_input_scale`, which measures
  `std(x_a)` under the real drive and rescales `B`; exact in one shot because the recurrence is
  linear in `B`. Measured effect at build time: the raw `U(-1,1)` draw put `x_a` at **`33.98x`**
  unit variance, which saturates `tanh` on those input columns. This also closes the confound
  `augmented-states/README.md` section 7 records as open, that driven-state RMS differed by up to
  `10x` across arms so "right pole" and "louder pole" were not separated.
  **Risk carried forward:** arm 2 used a TUNED constant `AUG_LRU_B = 0.377` (the class D-158 bans).
  Unit `x_a` std is principled and `0.377` is not, but the two have not been shown to land in a
  similar place. If a trained result disappoints, check this first; the build prints the pre-scale
  std so the comparison is recoverable.

**Elements beyond the three steps, stated so the count is honest.** `empirical_input_scale` is a
fourth element, cited but not one of the three; it is argued as part of step 1 because a live input
path that puts `x_a` at `34x` unit variance is not usable. The `rho` clamp to `[eps, 1-eps]` has NO
citation: `exp(-exp(nu_log))` returns exactly `1.0` below `nu_log ~ -45` and exactly `0.0` above
`+45` in float64, so "stable by construction" was true in exact arithmetic and false as implemented.
It is numerical hygiene, not method, and must not be presented as method.

The paper's own benchmarks are Bouc-Wen and Wiener-Hammerstein, the same two carried in `scripts/`. It also reads the explicit linear part as "a generalized form of a so-called residual network or resNet", which is the same condition Hoekstra states as "the learning functions are ResNets" (D-151).

This replaces three separate ad-hoc justifications with one paper by a supervisor of this project. Orvieto (`|lambda|` near 1, stable exponential parameterisation), Bachlechner (ReZero) and Hoekstra (p10, Eq. 31) remain as the parameterisation-level citations; Schoukens 2021 is the structural and procedural one.

**Which residual is identified.** `r_k = y_k - y_baseline,k`, formed in the loop the machine actually runs, which `cl_residual_spectrum.residual_for` already computes from logged `u`, logged `y` and the baseline alone, with no oracle, and whose header states the same path runs on real data. NOT the open-loop `rho = y - P0 u` that `fit_reduce.py` uses. Measured this session with `loop_sensitivity.py`: `smax(So)` is `3.5e-04` at 1 Hz, `2.1e-02` at 10 Hz and **`1.81` at 157.89 Hz**, so the loop suppresses the setpoint-motion band by `47x` to `2800x` and AMPLIFIES the absorber band. The open-loop residual is `1.224e-04` against the closed-loop `2.187e-06`, a factor 56, and that factor is almost entirely content the objective never sees. Identifying the open-loop residual points the fit at the wrong part of the spectrum.

**Estimator: parametric (ARX/IV), not a BLA, and this is a transfer decision.** Telica cannot supply a BLA. `pintelon2020_bla-feedback-process-noise` confirms feedback is not the obstacle (it extends BLA to feedback and process noise) but states the technique rests on "specially designed periodic excitation signals called random phase multisines and periodic noise". Per `docs/kamtin-telica-schema.md` the Telica excitation is a point-to-point move of 40 mm X and 80 mm Y, non-periodic and non-stationary, and `Y` (the scheduling variable) sweeps within every record so the plant is not LTI over it. The ILC iterations `iter0..iter8` are not realisations either: they "differ only in feedforward", which changes deliberately. A parametric fit needs none of that and runs unchanged on both simulation and machine.

Measured comparison of estimators on `T3_standstill_Y000` (a TRAINING record; validation records are not used to choose an initialisation):

| estimate | `f_n` [Hz] | err | `zeta` | err |
|-|-|-|-|-|
| truth (`fa = 150` coupled to `mh_rigid`) | 158.1139 | | 0.05000 | |
| peak picking (`cl_residual_spectrum.json`) | 158.2031 | `+0.056 %` | 0.05829 | `+16.6 %` |
| **parametric ARX, modal (D-159)** | 157.9884 | `-0.079 %` | 0.05247 | `+4.9 %` |
| FRF ratio on the excited lines | 156.8176 | `-0.820 %` | 0.04336 | `-13.3 %` |

The FRF was expected to beat peak picking on damping and did not, because it used the logged `u` in a closed-loop residual without instrumenting, used Levy's biased criterion with no Sanathanan-Koerner iteration, and fitted order 2/2 across 578 lines. The parametric route wins on both axes and already exists. No further estimator work is warranted in simulation: the errors above are all far inside what already works, since arm 2's own poles span `-36 %` to `+40 %` in damping and up to `6.1 Hz` in frequency.

**Deliberate departure from Schoukens 2021.** He initialises `A, B, C, D` from the linear model; we take only `A_aa` and keep `B_u` random per Hoekstra p10. Reason: D-158 refuses the fitted input map as `NONCAUSAL_IDENTIFICATION_COORDINATES`, and the `[u, x_b]` regressor split is unidentifiable in open loop. The departure is one-directional (we use less of the identification than he does) and must be stated, not hidden.

**Arms.** Seeds 42/43/44, `--route-all --stride 10 --epochs 2` (522 updates, matched to the overnight arms' 520), `na_nb = 17` pinned, physical baseline frozen. Reported per arm and seed: pooled closed-loop free-run RMS over complete V1-V4, `rms_per_channel_m`, and D-157's `F` on both ablation surfaces with the ratio beside it.

* **B1**, `nx_aug` = the sizing rule's output, poles exactly as identified. This is D-159's NOT-DECIDED (a), "whether one exactly placed pair matches the four-pole random bank", which no arm has ever run. It is a live candidate rather than a formality: the pole we would install is more accurate than any of arm 2's four.
* **B2**, four distinct pairs: the identified pair plus three spread across `2 zeta_hat f_hat`. **The spreading rule is unsourced** and D-159 records `zeta_hat * f_hat` as a conjecture with zero literature hits. It runs as an experimental factor and is never quoted as method.
* **C**, control, no identification: four pairs evenly spaced on `[130, 180] Hz` at `r = 0.99`, deterministic. Not a candidate method (see Ruled out 2), only a saturation check.

**Sizing.** `nx_aug` from the identification, not from arm 2. D-156's tolerance `eps` (split-half `H-infinity` disagreement) survives; its `2 sum_{k>r} sigma_k <= eps` bound does not, because that is a balanced-truncation bound and D-159 went modal. Repair: smallest even order whose MEASURED modal-truncation `H2` error is `<= eps`. Treat the result as a FLOOR: the absorber is one pair, so the rule plausibly returns `nx_aug = 2`, which is arm 1's known failure (`F = 0.03`). B1 against B2 is what decides whether the floor suffices.

**Why any of this is needed at all.** The objective freezes the poles, so they must be placed rather than learned: with the true mode planted C6 measures `dL/d(nu_log) < 0` on 7 of 8 batches, T3 proves the damping term is strictly positive under every non-negative weighting, and both trained arms moved their poles under `0.15 Hz` in 520 updates. And plain Jan cannot start at all: nothing writes rows `6..13`, so `x_a = 0` for every `k`, the ANN's read weights on `x_a` have zero gradient and its write path has no downstream effect. Measured `1.0002x`, `F = 0.0007` (A0, `BLA-Augmentation/RESULTS.md:298`).

Measured this session, the problem is also two to four decades harder than the benchmark Jan's zero-init design was demonstrated on. Unmodelled fraction `RMS(baseline error)/RMS(y)`: ECC MSD (3-DOF cubic against its linear BLA) **`20.5 %`**; gantry open loop **`0.121 %`**; gantry closed loop, our scoring surface, **`0.00217 %`**. That is `170x` like-for-like and `9450x` on the metric.

**Ruled out**:
(1) **The full-disk draw** (`RANDOM_LRU`, `capacity_runs.sh` idx 3-8). Computed at the exact seeds it would use: closest of four poles to the mode is `930.8 / 120.9 / 1009.9 Hz` at seeds 42/43/44, max radius `0.735 / 0.920 / 0.856` against arm 2's `0.985-0.992`. Neither memory nor span. `P(one of four pairs has r > 0.98 within 10 Hz of the mode) = 1.6e-03`. About 27 CPU-hours to confirm arithmetic. Its premise that `[0,1]` is "Orvieto's own default" is also wrong: Lemma 3.2 is stated for a ring `[r_min, r_max]`, `[0,1]` is the Glorot-equivalence baseline, and §3.3 tunes away from it ("increasing `r_min` closer to 1", `r_max` up to `0.99`) with Table 2 listing `+ Ring Init` as a separate improving row.
(2) **Orvieto Ring Init as the method.** Fixes the radius, leaves placement: with `r in [0.98, 0.999]` and unbounded phase, `P(one of four pairs within 8 Hz of the mode) = 3.2 %`. Cited and insufficient.
(3) **Placing poles across the excitation band as the method.** Kept as arm C only. `gtd_config.m` confines the injected multisine to `[130, 180] Hz`, a band chosen around `fa = 150 Hz`, so it does not avoid using knowledge of the absorber location; it moves that knowledge from a stated identification into the experiment design, where it is implicit and harder to defend, and it does not transfer to Telica where the band must be found first.
(4) **Residual weighting of the objective** (T3).
(5) **`EXACT_REPLICATED`** and any construction repeating one pole.
(6) **`na_nb != 17`** (C1).

**Constrains**: (a) `B_u` is mandatory whenever the block is on, consistent with D-159's `F2_no_Ba`. (b) Every arm reports `F` per D-157; the `2.0x` ablation threshold in `tasks/handoffs/2026-08-23-minimal-augmented-state-implementation.md` section 10 is superseded and must not be used. (c) The `2 zeta_hat f_hat` spreading rule stays an experimental factor until B1 against B2 gives it a result; if B1 matches B2, drop the rule rather than defend it. (d) If arm C matches B2, that is evidence that placement inside the band is saturated (consistent with C7) and it strengthens the identification result rather than replacing it, because C does not transfer. (e) Nothing lands in `model_augmentation/` until an arm has exercised it, and then only with the `@added` / `__project_origin__` / `# CHANGED` marker and a citation per element. (f) The word "BLA" must not be used for the parametric fit anywhere in the write-up or in `scripts/gantry/BLA-Augmentation/`; the directory name is now a misnomer and should be flagged wherever it is cited. (g) `literature/identification/ljung1999_mem.pdf` is Ljung, "Model Validation and Model Error Modeling", LiTH-ISY-R-2125, NOT *System Identification: Theory for the User*, which is not on disk; `forssell1998_cl_revisited_liu2021.pdf` has a mangled text layer and its `2021` is report number LiTH-ISY-R-2021, not a year.

### [D-159] Replace balanced reduction with modal selection in the BLA route; pole geometry is a validated surrogate, the band width is not yet a rule
**Date**: 2026-08-23
**Supersedes**: the first version of D-159, written the same day, whose central premise (that BSP at order 4 would have retained the 158 Hz mode) was falsified by measurement within hours. That version also carried a rate-conversion conjecture that is false and a magnitude argument that is withdrawn. Both are recorded in the amendments to sections 5.9, 9.8 and 9.11 of `docs/augmented-state-attribution-2026-08-23.md`.
**What**: One decision, plus two things explicitly NOT decided.
DECIDED: **`fit_reduce.py` selects the retained pair(s) modally from the unreduced fit and no longer balance-reduces.** `balanced_sp` is kept and reachable via `FIT_REDUCE_REDUCTION=bsp` so the recorded artefact reproduces, but the default path is modal selection. The order still comes from the existing Hankel-tail rule; the reported error becomes the measured relative `H2` error of the modal truncation, because the `2*sum(sigma)` bound is a balanced-truncation bound and does not apply to a modal one.
NOT DECIDED (a): whether one exactly placed pair matches the four-pole random bank. No arm has ever run it. The modal-selection arm IS that experiment.
NOT DECIDED (b): whether `zeta_hat * f_hat` is a usable band rule on Telica. It gives a width, not a warrant: a poorly identified friction or parameter-mismatch pole has a mathematically defined damping width without representing omitted dynamics worth modelling.
**Why**: Balanced singular perturbation returns `Ar = A11 + A12 (I - A22)^-1 A21`, a Schur complement, so its poles are not a subset of the unreduced spectrum. Measured on a read-only reconstruction of the normalised differenced clean ARX-28 fit: unreduced nearest pole `157.9884 Hz` (`zeta 0.05247`), BSP order 2 `5.0405 Hz`, order 4 two pairs at `~5.0400 Hz`, order 6 those plus `496.79 Hz`. **BSP retains the absorber mode at no tested order.** The mode is in every fit and in no reduced model, so the identification is not what failed and a larger order does not fix it. Modal selection retains the identified pair by construction, at `d ~ 0.008` from truth against `d = 0.047` for the best of the 24 poles ever drawn randomly.
Supporting, and independent of the reduction question: pole geometry is causal at fixed drive. Over the F4 family, where `B_a` is bit-identical and only pole frequency (F4a) or only pole radius (F4b) changes against arm-2 seed 0, `|B(lambda*)|` of `0.0062 / 0.6949 / 0.9242 / 0.9896` orders the ablation ratios `5.2081x / 1.9381x / 1.0255x / 1.0139x`, four of four with zero free parameters. On `band_draw_probe.json`, eight in-band and eight full-circle single-pair draws with no training and a linear readout are completely separated (exact permutation `p = 7.8e-5`), with `Pearson(d, 1-R^2) = +0.960` over all sixteen. And `F2_no_Ba` keeps the good poles with every pair inert (`1.0150, 1.0114, 1.0096, 1.0098`), which is the precondition in CONFIRMED `EVIDENCE.md` claim 4 that the states feeding a zero read-out must be excited, so `B_a` stays mandatory whenever `AUG_LRU` is on.
**Status of the mechanism**: `|B(lambda*)|` is a **surrogate**, not a proven achievable error or ceiling for this architecture. The identity is exact for scalar Hardy-`H2` approximation with free residues over an infinite horizon; the trained block has real conjugate pairs, fixed random vector `B_a`, finite records, a nonlinear jointly trained readout and closed-loop feedback. The literature agrees only to that strength: Toth 2010 eq. (2.61) PDF p65 (`MATCH OK`) names the quantity the **Kolmogorov measure**, and Toth/Heuberger/Van den Hof, Automatica 45(6):1359-1370, 2009 p4 (`MATCH OK`) states a bound and a proportionality, never an equality for one target pole. The `zeta*f0` band half-width has **no source at all** (zero arXiv and OpenAlex hits) and is a conjecture with a cheap synthetic falsifier.
**Ruled out**: (a) Hypothesis 7.1(a), Y-dependence of the target: `(M^-1)_44 = (mh+ma)/(mh*ma)` identically in `gantrySystemExtended.m`, so `f_n = fa*sqrt(mh_total/mh_rigid) = 158.113883 Hz` at every `Y` and every absorber deflection. Dead for this simulated plant. (b) "Use the fit for the band, keep the spanning draw" (5.10): it keeps the insurance and discards the information. (c) A larger BSP order as the fix, falsified above. (d) A resonant-gain account of why `W^a` cannot matter: `gamma = sqrt(1-r^2)` cancels the resonant gain by design, and the encoder-to-input state-RMS ratio is measured at `1.56x`, not the `178x` derived. (e) An Adam-drift account of `W^a`'s motion, superseded by 7.2's horizon derivation and `wa_freerun_probe.json`.
**Constrains**: `fit_reduce`'s reported reduction error must be the measured modal-truncation error, not the Hankel tail, since the two are different quantities. Any claim that a band width generalises to Telica must first establish that the residual contains a target worth hitting; `fit_reduce`'s own noisy-condition refusal (out-of-sample VAF `-0.0136`, "the residual is not a linear object on this data") is current evidence that it may not. Open and not decided: what sets the 2.5x spread across seeds once every draw already represents the mode to `|B| <= 0.0066` (9.6), and why a random `W^a` costs `2.1x` in training when post-hoc substitution costs `1.27%` (9.8). Both are located negatives, both appear only during training and in no post-hoc probe, and neither is explained here.

### [D-158] Phase B wiring: new-code augmented block, live linear bypass, `B_xb` zero-init trainable, ReZero on the shared net
**Date**: 2026-08-23
**What**: The BLA-Augmentation Phase B block is NEW code in `scripts/gantry/BLA-Augmentation/aug_block.py`, applied by the arm runner after `build_model`; one env-gated hook (`BLA_ARM_SPEC`) in `cl_train.py` (an experiment script); the only production file touched is the approved restore of `model_augmentation/fit_systems/closed_loop.py`. Block equation `x_a' = A_aa x_a + B_u u~ + B_xb x_b~ + gated NL`, with `A_aa` in the stable exponential parameterisation, trainable. A1: Orvieto Lemma 3.2 full-disk default (`[0,1]`, phase `2 pi u`; EVIDENCE claim 29), `B` ~ `U(-1,1)` (claim 9) with Orvieto `gamma`. A2: reduced fitted realisation, ZOH rate-converted, no `gamma` on the fitted `B_u`. `W^a` Xavier both arms (D-155). Full reasoning and Telica-portability audit: `scripts/gantry/BLA-Augmentation/DESIGN.md` D9.
**Why**: D7's amendment showed `C_r = 0` plus a zero-initialised shared final layer makes `dL/dx_a` exactly zero; claim 27 (linear component live, NL zeroed) plus claim 28 (ReZero) restore trainability while D-072 holds bit-exactly. The `B_r`-on-`u`-alone regression is resolved by derivation: open loop, `x_b = F(q) u` exactly, so the `[u, x_b]` regressor split is unidentifiable; the `x_b -> x_a` path is kept structurally, zero-initialised, trainable - no random scale constant returns.
**Ruled out**: (1) restoring the snapshot `model.py` gate collection (ten env gates in the production build path); (2) a random `B_xb` with a tuned scale (`AUG_LRU_B = 0.377` class, banned); (3) fitting the residual with `x_b` as a measured input (collinear regressors, non-unique split); (4) Jan's composite-condition baseline equality in place of the gate (does not give bit-exact D-072).
**Constrains**: arms A1/A2 share `nx_aug` (D5's output) and every training hyperparameter (lr `1e-5`, Adam eps `1e-16`, stride 10, `na_nb = 17` pinned, serial validation, 4 epochs max); nothing here goes into `model_augmentation/` beyond the approved restore.
**Addendum 2026-08-22 (D1 reopened, user correction)**: A2's `(A_r, B_u)` provenance is the contested open-loop residual construction. A2 runs (the pole gate gives it something installable) but is a simulation result only and NOT evidence of transfer; no Telica-portability argument is written on top of it, and the audit in DESIGN.md D9 is a checklist, not such an argument. On the real system the ordering is fit the baseline, then augment. The same contested-provenance label applies to D-156's `eps` as computed on fits of the open-loop residual: the RULE (split-half H-infinity disagreement) is construction-agnostic, the NUMBER produced tonight is not.

### [D-157] The ablation threshold is replaced by the improvement-fraction criterion `F`, with a noise-draw significance floor
**Date**: 2026-08-23
**What**: The `2.0x` (C8) and `1.02` (`probe_arm_ablation.py`) ablation thresholds are retired. Verdicts use `F = (RMS_ablated - RMS_trained) / (RMS_untrained - RMS_trained)`, the fraction of the arm's own improvement undone by removing `x_a`. `F > 1/2` = load-bearing (the majority of the learning went through `x_a` - the semantic boundary of the claim under test, not a tuned constant); `F` indistinguishable from 0 = dead. Under noise, significance is data-derived: `RMS_ablated` must lie outside the observed range of `RMS_trained` over `K = 3` independent validation-noise re-draws (`K` reported).
**Why**: the user ruled out heuristics; the handoff mandates replacing the threshold by a measured spread before use. Retrospective calibration: arm 2 `F = 0.88`, arm 1 `F = 0.03`, planted oracle `F = 0.93` - the criterion reproduces every recorded verdict without carrying the constants.
**Ruled out**: keeping either constant; per-record spread as the floor (conflates record heterogeneity with measurement noise).
**Constrains**: every Phase B arm reports `F` (both ablation surfaces) next to its RMS; a good RMS with `F` near 0 is a negative.

### [D-156] D5's `eps` is the split-half `H-infinity` disagreement of the residual fit; the VAF coupling is withdrawn
**Date**: 2026-08-23
**What**: `eps := max_w sigma_max(G^(1)(e^jw) - G^(2)(e^jw))` over the record's FFT grid, where `G^(1)`, `G^(2)` are the same estimator at the same settings on the two disjoint record halves D8 defined. `nx_aug = min{even r : 2 sum_{k>r} sigma_k <= eps}`; if no such `r`, the code refuses.
**Why**: the previous coupling (an `H-infinity` tolerance tied to a VAF difference) was flagged as underived in DESIGN.md D5 and no identity connects the two norms. The split-half disagreement is a measured realisation of the identification uncertainty in exactly the bound's norm, is constant-free (grid = the record's own FFT bins), and is computable on Telica. Deep-research (2026-08-23) confirmed no published rule ties `eps` to identification uncertainty (EVIDENCE claim 31; Forgione p9 budgets a measured test degradation with an underived "1%"), so this is recorded as our own derivation in EVIDENCE.md's Derivations table, never cited as literature.
**Ruled out**: Forgione's 1% fit-index budget (their constant, underived); Gavish-Donoho `2.858 * y_med` (i.i.d.-noise data matrix theory, does not transfer to Hankel singular values); picking a tolerance.
**Constrains**: `nx_aug` is an output of the resolved rule, identical in A1 and A2; a result using an unresolved `eps` is void per the handoff's acceptance criterion.

### [D-155] `W^a` is random by Xavier: the D-152 refutation does not survive contact with the whole page
**Date**: 2026-08-22
**What**: `W^a` is initialised **random, by Xavier**, per `hoekstra2026lfrfp` Eq. (31). `ENC_WA_ZERO`
is retired as a design question. Full reasoning: `scripts/gantry/BLA-Augmentation/DESIGN.md` D7;
verified quotes: `EVIDENCE.md` claim 9.
**Why**: D-152 correctly found that Eq. (31) specifies Xavier, then argued we should still zero
`W^a` "by refutation of the source on its own terms", using the encoder paper's Eq. (7)
(`x_bar = E[x | u_hist, y_hist]`, which is zero when the readout is zero). **That argument reads
Eq. (7) as an initialisation rule. It is not one.** Eq. (7) states what the encoder *approximates*,
i.e. what training drives it toward. The initialisation rule is Eq. (31), and the paper says of the
loss that fits the baseline encoder: *"The loss function (30) is no longer considered after
initialisation."* Three independent sources now say the same thing: `schoukens2020lfr` eq. (8)
(input path random, output path zero, on the same page as eq. (7)'s zeros); `schoukens2021ssnn_init`
Table II gR-SS-NN column (hidden weights `U(-1,1)/sqrt(n)`, biases `U(-1,1)`, only the linear output
layers zero); and `hoekstra2026lfrfp` p10 (*"All matrices not required to set the baseline model
behaviour at initialisation (29) have all elementsmof the matrix initialised randomly ...
m∼U(−1,1)"*). **`ENC_WA_ZERO` is a departure from Hoekstra, not an instance of it**, and it is the
departure that produces the D-130 dead zone: `schoukens2021ssnn_init` Sect. IV-B.2 (D-152's own
claim 4, CONFIRMED) identifies **both** layers zero as the untrainable case, which is exactly what
`4cdb7c1` had.
**Ruled out**: (1) Zeroing `W^a` (D-152's conclusion): superseded above. (2) Deriving `W^a` from the
residual model's observability, the way `W^b` comes from the baseline's (`hoekstra2026encoderinit`
Eqs. 16-17): **undefined, not merely unsupported.** That construction needs
`O_n^r = [C_r; C_r A_r; ...]^{-1}`, and the design discards `C_r`, so `O_n^r` is identically zero and
has no inverse. A design cannot both discard `C_r` and derive `W^a` from it. (3) Keeping
`kaiming_uniform_`: Eq. (31) says Xavier, a different initialiser at a different scale.
**Constrains**: (a) D-152's *citation corrections* stand in full and are unaffected; only its
`W^a = 0` conclusion is superseded. (b) The comment that ships must say Xavier per Eq. (31), with no
refutation clause. (c) **D-072 must be re-verified bit-identically after this change and before any
training**: a random `W^a` gives a non-zero `x_a(0)`, and baseline equality then rests entirely on
`C_r = 0` plus the zero-initialised ANN final layer.

### [D-154] The rigid-body integrators must be excluded from the residual model, and this is an identifiability condition
**Date**: 2026-08-22
**What**: In the additive split `P = P0 + Delta`, the residual model `Delta` is constrained
**strictly stable with no pole at the origin**. The rigid-body poles are NOT put into `Delta`
(which is what the wafer-stage literature does for a *full-plant* fit) and low-frequency lines are
NOT discarded as the primary remedy. Full reasoning: `BLA-Augmentation/DESIGN.md` D4.
**Why**: `vanderhulst2025additive` (IEEE L-CSS 9:547-552, 2025) p2, verified at the PDF, states of
an additive model structure: *"where at most one submodel may include li > 0 poles at the origin"*
and *"where the Ai(p) polynomials are stable, i.e., all roots lie in the left-half plane"*. `P0`
owns the `1/s^2`, so `Delta` may have none. **This is a uniqueness condition**: if both terms may
carry integrators, rigid-body content can be traded between baseline and augmentation at zero cost
in fit, which is the negation failure mode this project already tracks, arriving as an
identifiability statement rather than as a training pathology. `voorhoeve2021positiondep` p6
(*"n0 = 2 nrb poles are located at s = 0, by factoring out the rigid body dynamics"*) points the
**opposite** way, but it constrains a full plant, not an additive residual; a design reading only
that paper would do the wrong thing.
**Measured, not argued**: the unconstrained shared-denominator fit of the open-loop residual returns
`rho(A_r) = 1.00003` at every order from `na = 8` upward, i.e. a near-integrator reproducing the
drift. The Hankel Gramians then do not exist, so the balanced-truncation order rule is unevaluable
at every order that fits. At 1x Telica sigma it is worse: `rms(rho)` rises from `1e-06..1e-03` m to
`2.4e-03..3.3e-02` m - **8.5 nm of sensor noise becomes 3-33 mm of residual** - because
`-C_fb(v)` is injected as a force into an open-loop double integrator. Order then buys nothing:
out-of-sample VAF is `0.94` at `na = 2` and `0.94` at `na = 28`.
**Ruled out**: (1) Factoring `s^2` INTO the residual model (`voorhoeve2021positiondep`): wrong
object. (2) Discarding lines below 20 Hz as the rationale (`vanderhulst2025waferstage` p4): the
practice is real but its stated reason is closed-loop FRF measurement quality, *"as the rigid-body
behavior is poorly captured at lower frequencies in the measurement"*, which is not our reason and
must not be attributed to that paper. Available as a fallback. (3) Leaving it to the
orthogonal-projection penalty: the penalty currently has `dV_orth/dp = 0` for every augmented
parameter, and an exact degeneracy should be removed at the estimator anyway.
**Constrains**: `nx_aug` cannot be determined until this is applied - the balanced-truncation rule
needs a stable realisation. The recommended route is to fit `(1 - q^-1)^2 rho`, which keeps the
least-squares structure; it is specified and **not yet run**.

### [D-153] The residual the prototype fitted was the open-loop residual filtered by the baseline's own sensitivity
**Date**: 2026-08-22
**What**: Every residual-BLA number recorded before today - `157.8946 Hz`, `zeta = 0.05257`,
`n_A = 28`, the `-12.0 Hz/m` Y locus, the noise sweep, and the band recipe's
`[149.90234, 164.06250] Hz` - was computed on `S rho`, not on the residual. They are marked as
predating this check and are not carried into any later decision without recomputation. The one
exception, argued below, is the mode **location**.
**Why**: `cl_residual_spectrum.residual_for` calls `cl_headroom.closed_loop_run`, which at every
sample forms `e = y_data[k] - y_model[k]`, runs the controller on it, and applies
`u = u_data[k] + C_fb e`. **The baseline is inside an auxiliary tracking loop driven to follow the
recorded output.** Frozen-Y LTI algebra: `y_m = P0(u_d + C(y_d - y_m))` gives
`r = y_d - y_m = (I + P0 C)^{-1} (y_d - P0 u_d) = S rho`. Measured on three records by
`BLA-Augmentation/probe_d1_residual_identity.py`, which realises `S rho` without ever forming `S`:
`|r - S rho| / |r|` is `0.24 %` to `52 %` (the residual IS the LPV departure), and the suppression
`rms(rho)/rms(r)` spans **`10.1` to `642.0`**. Decisively, `rms(rho)` varies over three orders of
magnitude across records while `rms(r)` is nearly constant at `~3.7e-07` (X) and `~3.7e-06` (Y) in
all nine record-channels: **a tracking loop drives its error to its own floor regardless of the
disturbance, so the prototype was measuring the auxiliary loop's floor.** This violates the explicit
side condition of the project's own sympy-verified ratio derivation ("only with the baseline
simulated open loop on the recorded input") and the handoff's own do-not list.
**What survives**: the mode **location**. `S` multiplies `rho`, and multiplication cannot move a
pole of `rho`, so `157.8946 Hz` remains evidence that a lightly damped mode near `157.89 Hz` is in
the residual. The 28th-order **realisation** does not survive: `r`'s poles are
`poles(rho) ∪ poles(S)`, and `poles(S)` are the closed-loop poles of `(P0, C_fb)`, so taking `A_r`
wholesale would plant auxiliary-loop poles into the augmented block.
**Also measured, separately**: the controller replay check D1 was written for
(`verify_cfb_against_records.py`) comes back **clean** on standstill and y-sweep records - at or
below the float32 storage floor once the `z = 1` integrator ramp is removed - and **`6x` to `18x`
over the floor on the APRBS records**, at `~10^-3` relative to `rms(u_fb)`. Per `sugie2020dualyoula`
Remark 2, `K != K_hat` puts `(K - K_hat) y` into the regressor and correlates it with the noise;
at `10^-3` that is a real but second-order contaminant. **It is not what invalidates the numbers.**
**Ruled out**: dividing the recorded `r` by `S` after the fact. The suppression is record-dependent
by a factor of 64 and `S` is itself Y-scheduled, so there is no fixed factor; and the point of the
open-loop arm is that it costs one extra call with a zeroed controller, through the same integrator.
**Constrains**: (a) all Phase A fitting is of `rho`; (b) D-154 exists because `rho` drifts and `S`
was hiding it - `S` has zeros at `P0`'s poles, i.e. at the double integrators; (c) anything in
`docs/gantry-augmentation-problem-log.md` or the handoff's "established and verified" table that
cites a residual-BLA number is established about `S rho`.

### [D-152] The `W^a` random initialisation DOES have a literature source; we depart from it by refutation, not by absence
**Date**: 2026-08-22
**What**: A citation correction, RECORDED here and not applied to the code this session (the user's instruction: `model_augmentation/` stays unmodified while the attribution runs are in flight). `model_augmentation/fit_systems/pre_encoder.py:422` reads `# HEURISTIC, with no literature source: kaiming_uniform_ on both blocks.` **That is false.** Hoekstra, Gyorok, Verhoek, Toth, Schoukens, arXiv:2602.17297 (`literature/closed-loop-id/hoekstra2026_lfr-augmentation-fp-models.pdf`), **p.9 Sec. 5.4.2 Eq. (31)**, verified at the PDF, states of the augmented-state encoder block: *"where the weights and biases of psi_aug are initialised by the Xavier approach"*, and **p.10 Sec. 5.4.3** adds *"All matrices not required to set the baseline behaviour at initialisation (29) have all elements m of the matrix initialised randomly ... m ~ U(-1,1)."* A random `W^a` is therefore Hoekstra's stated convention. A **second** error, found in the same reading and not previously recorded: the source specifies **Xavier** (Glorot) while our code calls `nn.init.kaiming_uniform_`, which is a different initialiser at a different scale. So the comment denies a source that exists AND the code does not implement the convention that source specifies.
**Why we still zero `W^a`**: by refutation of the source on its own terms, not by absence of one. Hoekstra's encoder paper, arXiv:2602.13108 (`literature/augmentation/Encoder initialisation methods in the model augmentation setting.pdf`), **p.3 Eq. (7)**, verified at the PDF, defines the encoder as approximating `x_bar_k = E_e[x_k | u^{k-1}_{k-n}, y^{k-1}_{k-n}]` and calls it "an unbiased estimator of `x_k`". Under D-072 the augmented readout is **exactly zero**, so at initialisation `x_a` cannot influence `y` and the window carries no information about it; Eq. (7)'s conditional expectation collapses to the unconditional one, and the only value consistent with baseline equality is `0`. A random `W^a` is not an unbiased estimator of anything at initialisation, it is an arbitrary O(1) functional of the window injected into a state the output cannot see.
**Ruled out**: (1) Leaving the comment as written. It is the kind of error that survives into a thesis, and "no literature source" invites a reviewer who knows Eq. (31) to conclude we did not read the paper. (2) Silently switching `kaiming_uniform_` to Xavier to match the source: the whole point of `ENC_WA_ZERO` is that the initial value should be zero, so matching the random convention more faithfully is the wrong repair. (3) Applying the fix now: `pre_encoder.py` is one of the two files the attribution runs are running against, and editing it mid-factorial would put a different source file behind the later arms.
**Constrains**: (a) Wherever the `W^a` block ships in a clean `model_augmentation/` implementation, the comment must read "Hoekstra's stated convention (arXiv:2602.17297 p.9 Sec. 5.4.2 Eq. (31)), refuted here by his own arXiv:2602.13108 p.3 Eq. (7) under D-072", never "no literature source". (b) `scripts/gantry/gantry_dynamic/model.py`'s `ENC_WA_ZERO` comment block repeats the same wrong claim and takes the same correction. (c) `docs/references.md` line 47 asserts "Our `W^a` init is therefore an assumption with no literature source"; that sentence is wrong and is superseded by this entry. (d) The F3 arm that wins in the ablation determines only the VALUE, not this provenance text; the correction applies whichever arm wins.
**Related**: a second, smaller provenance correction found in the same pass, in `gantry_dynamic/model.py` rather than in `model_augmentation/`: the comment `# THEORY: Orvieto et al. ICML 2023 Sec. 3.3 -- nu_log = log(-log r), theta_log = log th.` is right about `nu_log` and wrong about `theta_log`. Orvieto Sec. 3.3 p.8 gives `lambda_j = exp(-exp(nu_j^log) + i theta_j)`: the magnitude is exponentiated, the phase is learned directly. The extra exponential on the phase is ours (it keeps the frequency strictly positive so a conjugate pair cannot collapse onto the real axis mid-training) and must be labelled `# HEURISTIC:`. Full provenance table, including the confirmed Lemma 3.2 radius draw and the Eq. (7) / footnote-9 `gamma`, in `tasks/ablation-2026-08-22-what-earned-its-place.md` step 4.

### [D-151] Restore the input path into the augmented states: a non-zero `B_a` injection, readout still exactly zero
**Date**: 2026-08-20
**What**: An env-gated (`AUG_LRU_B=<scale>`) extension of the D-150 `AugLRUBypass` in `scripts/gantry/gantry_dynamic/model.py`. The augmented rows become `x_a,k+1 = A_aa x_a,k + gamma * (B_a z + NL(z))` with `z = [x, u]` the ANN input, `B_a` drawn i.i.d. `N(0, 1/nz)` from a seeded generator and scaled by the env value. The columns of `B_a` on the augmented states themselves are forced to zero, so `B_a` cannot feed `x_a` back into `x_a` and the pole stays exactly the band-initialised `A_aa`. Rows 0-5 of the ANN output are untouched and still exactly zero at initialisation. Default OFF, so every existing run reproduces.
**Why**: measured 2026-08-20 (`transient-investigation/calibrate_lambda_defect.py`, `probe_input_injection.py`). At initialisation `gamma * NL = 0`, so `x_a` is an autonomous ringing that nothing drives, and rows 0-5 are exactly zero, so nothing observes it. Consequences, all measured: `||grad||` from `L_settled` is EXACTLY `0.0000e+00` on both `W^a` and `nu_log`/`theta_log`, i.e. neither the augmented encoder block nor the pole can be trained by the output loss; and the multiple-shooting defect, the only term that does reach them, is degenerate on the augmented rows, because with `x_a` undriven `d_a,j = enc_a(psi_j) - A_aa^nf_seg enc_a(psi_{j-1})` is minimised by `enc_a == 0`. The shrinkage is confirmed rather than argued: `<grad_{W^a} L_defect, W^a> = +1.983` (positive, so descent reduces `||W^a||`), and under defect-only Adam at the `lr_enc = 1e-4` of the epoch-1 run both `L_defect` and `RMS(enc_a)` fall about 2.3 % per 15 updates on matched batches, i.e. roughly halving within one epoch's 416 updates. This is the mechanism behind the epoch-1 result (validation 2.8x worse) and behind Arm F's `rho` "holding" at 0.9920, which held because it was frozen, not because training endorsed it.
**Why this is a restoration, not an invention**: Hoekstra `arXiv:2602.17297` Section 5.4 initialises the augmentation with the NONLINEAR component zeroed and the LINEAR component live (`phi_aug(z_a) = 0 + W_a z_a`, Condition (b) "the learning functions are ResNets"), and randomises every LFR matrix not needed for baseline equality (`m ~ U(-1,1)`); his augmented encoder block is a Xavier draw with no target (Eq. 31), which our `kaiming_uniform_` `W^a` already matches. Jan's own code carries both variants: `identity_init_simple_res_net` (`torch_nets.py:40`, `net_lin` identity-initialised plus a zero MLP) and `zero_init_feed_forward_nn` (`torch_nets.py:97`, zero final layer, no linear skip). The gantry build passes the latter, which zeroes the input path into `x_a` as a side effect of enforcing baseline equality. `B_a` restores the property Jan's initialisation has, by the one route that does not disturb D-150's pole or D-072.
**D-072 status**: preserved, verified bit-exactly. `y` reads `x_a` only through ANN rows 0-5, which stay exactly zero, so the injection cannot move the output at initialisation. Measured in one process with the gate off then on: validation free-run RMS `2.186601103417735e-06` both times, bit-identical and per-record identical, matching the D-072 reference.
**Scale**: `AUG_LRU_B` is the multiplier on the `N(0, 1/nz)` draw. # HEURISTIC: the reference value 0.377 was measured on this dataset (seed 0, `nz = 11`) as the scale putting `RMS(x_a)` equal to `RMS(x_phys)` in normalised coordinates, an engineering choice that the augmented state be neither negligible nor dominant, with no literature source. It is data-derived per dataset and must be re-measured, never carried across datasets as a constant.
**Ruled out**: (1) switching the block to `identity_init_simple_res_net`, whose `nn.init.eye_` linear part would write `x_6, x_7` straight through (`A_aa = I`, unstable, discarding the D-150 band) and put non-zero entries on rows 0-5, breaking D-072; (2) letting `B_a` act on the augmented columns of `z`, which would add a feedback term to the pole and void the LRU stability guarantee; (3) weakening D-072 with an epsilon readout, which attacks observability directly but is the user's decision and not taken here; (4) keeping the multiple-shooting defect as the route to `W^a`, whose descent direction is measured shrinkage in the undriven case and whose gradient explodes by 21x to 1056x per group once the injection is live, pushing the balancing weight to about `1e-10`.
**Constrains**: the gate must stay OFF by default. `AUG_LRU_B` requires `AUG_LRU`. Checkpoints from injected runs carry `B_a` in the ANN block and reload only into a build with the same gate and scale. `rho(A_aa)` is the pole of the augmented block in isolation and stops being a pole of the model once `B_a` and the readout are both live, so the band claim must be phrased accordingly. The secondary acceptance criterion "`rho(A_aa)` above 0.5" is superseded: a model with `x_a == 0` passes it. Replace with `RMS(x_a)` non-negligible against the physical states AND the readout's augmented columns non-zero.

### [D-150] Split `f_aug` from `g_aug`: a stable linear bypass on the augmented rows, ring-initialised from a data-derived frequency BAND
**Date**: 2026-08-19
**What**: An env-gated (`AUG_LRU=1`) change to the ANN parameterisation in `scripts/gantry/gantry_dynamic/model.py`. The ANN's `zero_init_feed_forward_nn` is wrapped so that its output rows split into the two functions Hoekstra's S-DP structure keeps separate (`arXiv:2602.17297` Table 1): rows 0-5 (`f_aug`, the correction into the physical states) stay exactly zero at initialisation, and rows 6-7 (`g_aug`, the augmented states' own update) become `x_a,k+1 = A_aa x_a,k + gamma * NL(x,u)[6:8]` with `NL` zero at initialisation and `A_aa` live. `A_aa` is a trainable 2x2 rotation-scaling block held in the LRU stable exponential parameterisation `lambda = exp(-exp(nu) + j exp(theta_log))` with input normalisation `gamma = sqrt(1 - abs(lambda)^2)` (Orvieto et al., ICML 2023). Its eigenvalues are ring-initialised over a BAND read at build time from `runs/cl_residual_spectrum.json`: theta spans the frequency range of the dominant strong (over 10 dB above floor, `zeta_ok`) residual peaks across all records and channels, and the annulus radius spans the per-peak `rho = exp(-zeta*wn*Ts)` range from the same set. No frequency, damping or radius constant is written into code.
**Why**: measured 2026-08-19 (`cl_aug_spectrum.py`, `cl_latent_init_test.py`): `rho(A_aa)` is exactly 0 at initialisation and 2.9e-10 after training against 0.976 for the planted model that closes 82 % of the headroom, because one shared zero-initialised output layer produces both `f_aug` and `g_aug`, so zeroing it for D-072 baseline equality destroys the augmented dynamics as a side effect. Restoring a live `A_aa` removes the 34.83x loss barrier entirely (1.00x, monotone descent) and makes the readout gradient carry `x_a` information (cos with `x_a` blanked drops from +1.000000 to +0.464718). A single MLP output row cannot realise a target `A_aa` (asked 0.98 at 159 Hz, got 0.649 at 38.5 Hz), which is the measured argument for a linear bypass where `A_aa = A` exactly.
**Why a BAND and not a mode**: the real Telica campaign provably cannot supply an identified resonance (`telica_plant_frf.py`: identifiable band under 83 Hz on X, under 55 Hz on Y, no plant resonance supported in 10-8000 Hz), so a mode-based initialisation would work in simulation and have nothing to run on at the machine. The LRU result is precisely that the eigenvalue DISTRIBUTION at initialisation decides learnability, so the recipe is a distribution over a band; where a mode is identifiable (simulation) the band collapses toward it and the point estimate is recovered as a special case.
**D-072 status**: preserved structurally. Rows 0-5 of the ANN output are exactly zero at initialisation and `y = C x[0:6] + D u` never reads rows 6-7, so the augmented model IS the baseline at t=0. The known cost: `dL/dA_aa` is zero at step 1 and unlocks from step 2, forced by exact baseline equality. The epsilon-readout variant that trains from step 1 weakens D-072 and is the user's decision, not taken here.
**Ruled out**: (1) solving `W_out[6:8,:]` for a target Jacobian, measured to miss by 2.5x in rho and 4x in frequency; (2) `ANN_INIT_SCALE`, cannot move `A_aa` off zero; (3) hard-coding 158-159 Hz or rho 0.9856, oracle-adjacent in simulation and undefined on Telica; (4) a frozen (non-trainable) `A_aa`, unrecoverable if the band is wrong; (5) an unconstrained trainable `A_aa` matrix, loses the guaranteed `abs(lambda) < 1` during training that the exponential parameterisation gives by construction.
**Constrains**: the recorded LRU limitation stands: `lambda = exp(-exp(nu))` maps onto the OPEN unit disk, acceptable for a parallel augmentation (the integrators live in the baseline) and not for a black-box arm. The gate must stay OFF by default so every existing run reproduces. Runs made with `AUG_LRU=1` checkpoint extra parameters (`nu_log`, `theta_log`) inside the ANN block and reload only into a pipeline built with the same gate. The band derivation reads a residual-spectrum artefact; on datasets where no strong peak exists the build refuses with instructions to supply an explicit band from loop-bandwidth and sample-rate requirements rather than silently defaulting.

### [D-149] New data track `joint_lowf`: the multisine band starts at the record fundamental, not 1 Hz
**Date**: 2026-08-19
**What**: A new TRACK in `Matlab-scripts/Augmentation/data/gtd_config.m`, `joint_lowf`, identical to `joint` in every respect except `cfg.f_low = 1/cfg.t_record` (0.0833 Hz) instead of 1 Hz. `cfg.f_high` stays 200 Hz, `t_record` stays 12 s, record table, seeds and amplitudes unchanged. Because `cfg.out_dir` is keyed on TRACK and `cfg.fig_dir` derives from it, the new track writes to `data/gantry/matlab/trajectory/joint_lowf/` and its own `figures/` subfolder, so **nothing in `joint/` is overwritten**. All 22 records to be generated. An `otherwise` branch was added to the same switch, because an unrecognised TRACK previously left `f_low`/`f_high` undefined and failed later inside the multisine synthesis.
**Why**: the baseline model has two REAL poles, first-order corners rather than resonances, at the coast-down time constants `tau_X = (m1+m2+mb+mh)/(cg1+cg2) = 1.546 s` and `tau_Y = mh/cy = 1.010 s`, i.e. corners near 0.103 and 0.158 Hz. These are the same constants D-087 already cites for the open-loop offset problem. They sit BELOW the `joint` band, and measurement shows the band cannot see them: deleting both poles entirely changes the FRF over 1-200 Hz by a **median of 0.56 %**, against the ~1.5 % accuracy achieved by `lpm_frf.py`. Consequences: `frf_to_ss.py` cannot fit them (order 8 returns an unstable spurious pole, so order 7 was selected), the black-box initialisation gets pure integrators where the truth settles with a 1.5 s constant, and `cg1+cg2` and `cy` are weakly identifiable as physical parameters, which matters directly once joint estimation estimates them.
**Why 1/t_record specifically, and why not lower**: `gtd_make_multisine.m:53-55` places lines on exact FFT bins (`bins = round(freqs/df)+1`), and at `t_record = 12 s`, `fs = 20 kHz` the spacing is `df = 0.08333 Hz`. The fundamental is therefore the lowest bin that exists. Setting `f_low` to 0.05 or 0.07 snaps to the same bin set and gains nothing. The lever for more margin is `t_record`, which the user has deferred. Sensitivity at the bins this buys, measured as deviation from a pure double integrator: 0.0833 Hz gives -4.02 dB and **+51.0 deg**, 0.1667 Hz +31.7 deg, 0.25 Hz +22.4 deg. Large signatures, so identification should work, but there is exactly **one line below each corner** and it is the fundamental, which has a single period in the record and no period-to-period averaging. Marginal rather than comfortable; if these two poles later fit badly, revisit `t_record` (20 s gives `df = 0.05` exactly and three lines below the Y corner) before touching the estimator.
**Band justified from the BASELINE, not the truth**: `plant.py:53` sets `_C3, _K3 = _C4[:3,:3]`, so the 6-state baseline carries the same `cg1`, `cg2`, `cy` as the truth and predicts these corners itself. The truth pole values (0.1029, 0.1576 Hz) are used only as a post-hoc diagnostic. Designing the band from the truth would tune the experiment to the answer and would not transfer to hardware.
**Ruled out**: (1) Modifying `joint` in place: destroys a working dataset and every result measured on it. (2) Lengthening `t_record` to 60-120 s: my first proposal, withdrawn. It was derived from a cycles-counting heuristic appropriate to a resonance, not to a first-order lag, and the sensitivity numbers above show 12 s already carries a large signature. (3) Amplitude shaping from the start: deferred, see below. (4) Standstill records only: cheaper, but a half-populated track invites later mixing of records across folders, and joint estimation wants the full set.
**Open risk, flat spectrum**: the spectrum stays flat in step one. Open-loop worst-case estimates put the lowest Y line at 103 mm rms and total Y usage at 85.7 % of stroke. These records are closed loop and the controller suppresses low frequency hard, so the true figure should be far lower, and `gtd_enforce_limits.m` checks the actual simulated signals. **If it trips on Y, the fix is displacement shaping in `gtd_make_multisine.m:synth()`** (weight each line so predicted displacement is capped; a 10 mm cap gives 0.234 N on X and 0.059 N on Y at the lowest line against the flat 0.816 and 0.612), and the track then differs from `joint` by two things rather than one, which must be recorded here.
**Constrains**: `joint` and `joint_lowf` are **siblings, not nested**. Changing `f_low` changes which lines exist, so the multisine realisations differ throughout and the datasets are not comparable record-by-record. Everything measured on `joint` remains valid for `joint`; a model trained on one must not be evaluated on the other without saying so. `lpm_frf.py`'s `TRACK` constant selects the folder and must be set deliberately.

### [D-148] BLA estimation moves from time-domain N4SID to a frequency-domain, per-line-weighted chain
**Date**: 2026-08-19
**What**: `bla_init.py`'s `fit_bla` (deepSI `SS_linear`, i.e. N4SID on the normalised time record, `SS_f` swept and selected on `Vn`) is superseded as the BLA estimator. The replacement chain, in order: (1) decimate `u` with the SAME zero-phase FIR used for `y`, or fit at the 4 kHz native rate; (2) estimate the FRF nonparametrically by the local polynomial method, which models the transient term alongside the FRF; (3) fit the parametric model on the FRF with a **per-frequency-line** weight, factoring the rigid-body poles out instead of estimating them, and discarding the lowest frequency lines; (4) for the Y-dependence, fit per-Y-band BLAs in zero-pole-gain form and parameterise each pole, zero and gain as a polynomial in Y. Full literature basis and citations: `scripts/gantry/ann-blackbox/BLA-LITERATURE.md`.
**Why**: the current estimator misses both resonant pairs. Measured (`server-results/paired_bla76176.out:32`): `|eig(A)| = [0.999986, 0.999977, 0.999213, 0.998775, 0.896084, 0.896084, 0.291552, 0.07701]` against a frozen truth of `[1, 1, 0.999192, 0.998763, 0.996625 (5.12 Hz pair), 0.936582 (157.9 Hz pair)]`. `frf_diagnostic.py` (run 2026-08-19, `results/frf_diagnostic/summary.json`) attributes the miss: T10 carries **-17.1 to -22.6 dB** of INPUT power at 157.9 Hz relative to 5.1 Hz, so the absorber is well driven, but only **-73.1 to -85.1 dB** of OUTPUT power, so its share of the output energy is order 1e-8. N4SID truncates on Hankel singular values, which is an energy ranking, so the mode is discarded by construction. The cure in the literature is not a tuning change but a different estimator: the frequency-domain cost is normalised per frequency line rather than per time sample (Schoukens, Vaes, Pintelon, IEEE CSM 36(3):38-69, 2016, arXiv:1804.09587, Eq. 11), and Pintelon and Schoukens 2012, Sec. 13.11.2 states that the frequency domain has no problem with plants whose poles lie on or outside the unit circle. **CORRECTION, 2026-08-19, after reading the sources in full:** an earlier version of this entry said Bauer and Ljung (Automatica 38(5):763-773, 2002) show the subspace row weighting acts as a frequency weighting, "which is the theory statement of the observed failure". That claim came from metadata and did not survive reading the paper. Bauer and Ljung prove that the CCA weighting minimises the ASYMPTOTIC VARIANCE under noise; they mention a frequency-weighted choice only in passing, citing Bauer's 1998 thesis. More importantly, both that paper and Gustafsson (Automatica 38(3):433-443, 2002) assume in their Section 2 / Assumption A1 that **all eigenvalues of A lie strictly inside the unit circle**, and additionally assume a white or quasi-stationary input and a noise process. Our plant has two poles exactly ON the unit circle, our input is a clock-held APRBS, and our simulation is noiseless. **The subspace-weighting theory therefore does not reach our problem on three independent grounds, and that is itself the argument for leaving the time-domain estimator rather than tuning it.** Gustafsson's pre-filtering is also of the INSTRUMENT vector, not of u and y, and his Section 7.2 states it has basically no effect when past outputs are included in the instruments, which is deepSI's `SS_linear` case. Marginal stability is not an obstacle in this domain: Pintelon and Schoukens 2012, Sec. 13.11.2 states that the frequency domain has no problem with unstable plants because the transfer function is evaluated only on a grid, with coincident lines dropped or regularized; our poles at z=1 are exactly that case. Rigid-body handling is taken from the motion-control literature (Voorhoeve et al., IEEE TCST 29(1):194-206, 2021: factor the rigid-body dynamics out, fix `2*n_rb` poles at s=0, inverse-magnitude clipped weighting; van der Hulst et al., IFAC 59(17):67-72, 2025: discard lines below the band where rigid-body behaviour is poorly captured).
**Why the weight is a HEURISTIC, not THEORY**: the literature's per-line weight is the total variance (noise plus stochastic nonlinear distortion), estimated from period-to-period and realisation-to-realisation scatter. Our data is a **noiseless** simulation and, at frozen Y, linear, so both variances are identically zero and the weight degenerates to 0/0. The substitute is a relative-error weight (divide by the squared magnitude of the measured FRF). No paper states that substitution for the noiseless case, so it carries `# HEURISTIC:` per the repo's signal-processing labelling rule.
**On the `u`/`y` decimation mismatch, TESTED AND NOT THE CAUSE (added 2026-08-19 after the fact)**: `data.py:29-30` decimates `u` by a block mean and `y` by a zero-phase FIR, which costs 0.54 dB and 28 degrees of uncompensated u-versus-y phase at 157.9 Hz. This was initially recorded here as a defect to fix first. `bla_decimation_test.py` measured four conventions and **falsified that**: the absorber is missed by essentially the same distance in every one, so `data.py` is NOT changed and D-087 stands.

| variant | fs | u | y | dist to absorber | dist to 5.12 Hz pair |
|-|-|-|-|-|-|
| A, current | 800 | block mean | zero-phase FIR | 9.19e-01 | 4.02e-02 |
| B, matched filters | 800 | zero-phase FIR | zero-phase FIR | no stable BLA in the SS_f grid | - |
| C, D-087 as written | 800 | block mean | point-sampled | 8.87e-01 | 4.02e-02 |
| D, native, no 2nd decimation | 4000 | - | - | no stable BLA in the SS_f grid | - |

**What the test found instead, and it is more useful than the hypothesis it killed.** The BLA's spare states go to near-Nyquist artefacts, not to the physical resonances. Variant A's pole list is four good slow modes plus a pair at **341.5 Hz** and a single pole at **400.0 Hz, exactly Nyquist**; variant C moves that pair to 303 Hz. The zero-phase FIR applied to `y` is non-causal and has no causal pole-zero realisation, so a causal BLA cannot represent it and produces near-Nyquist junk while trying. Second finding: **the 5.12 Hz pair is missed too**, by 4.02e-02, which is the full distance from the real axis, so the nearest BLA pole to it is real. The BLA has no complex pole at either physical resonance. Third: variant C recovers the integrators 20x more accurately than A (9.93e-07 versus 1.76e-05 from z=1). Fourth: at 4 kHz the whole `SS_F_GRID` is rejected by deepSI's own explosion guard, because the poles crowd z=1, so a 4 kHz arm needs a different grid before it can be used as a control.
**Constrains (added)**: `data.py` stays as it is and D-087 stands unmodified. Any future 4 kHz arm must first widen or replace `SS_F_GRID` in `bla_init.py`. The near-Nyquist artefact is now the leading secondary hypothesis for where the two spare states go, and the frequency-domain chain should be evaluated on whether it stops spending states there.
**Ruled out**: (1) keeping N4SID and only changing the selection criterion from `Vn` to a free-run score: `Vn` is not the root cause, the energy ranking is, and no selection rule recovers a mode the truncation removed. (2) Blaming the excitation and regenerating data first: step 0 measured the input spectrum and the absorber is driven at -17 to -23 dB, so an APRBS-to-multisine swap alone would not have fixed it. The sinc-squared clock-period argument (5SMB0 Lecture 9 p16-19) is real but not binding here. (3) Blaming the LPV scheduling: BLTI theory predicts a single global BLA is a record-weighted average of the frozen responses that lands inside the Y-locus and drops no mode, and only the 5 Hz pair moves with Y (4.83 to 5.12 Hz) while the absorber is Y-invariant to six digits. (4) Forcing `SS_A_stability`: already ruled out in `bla_init.py`, it drags the poles to 0.90-0.975 and a stable BLA is the wrong model for a plant with free integrators.
**ADDENDUM 2026-08-19, naming, and it changes what the thesis may claim.** The model this chain produces is fitted on STANDSTILL records at one frozen Y, while the network trains on APRBS records whose trajectories sweep Y. The BLA is defined as the mean-square-optimal LTI approximation **for a given input class**, so that model is **not a BLA** and must not be called one. Both were estimated (`bla_vs_frozen.py`, `results/bla_vs_frozen/summary.json`), z-plane distance to the frozen truth pole at 800 Hz:

| arm | absorber | slow pair | unstable |
|-|-|-|-|
| BLA of the training input class (APRBS + lissajous, Y -0.30 to +0.30 m) | 1.86e-02 | 4.03e-02, missing | 1, at \|z\|=1.286 |
| frozen point (T3 standstill, Y=0, periodic) | 9.08e-06 | 6.75e-04 | 0 |
| time-domain N4SID BLA, for scale | 9.19e-01 | 4.02e-02 | n/a |

Two separable effects: changing only the ESTIMATOR (time domain to frequency domain, same input class) buys ~50x on the absorber; changing the INPUT CLASS as well buys another ~2000x plus the slow mode plus stability. **Decision: report the training-class fit as the BLA, and call the frozen-point model a frequency-domain linear model at a frozen operating point**, which is the same kind of object Hoekstra et al. 2026 initialise from (local linearisation of the baseline at an equilibrium). The defensible claim is that the BLA of the training data cannot represent the plant's resonant structure while a frozen-point model can.
**Also corrected here**: an earlier version of this entry, and of `BLA-LITERATURE.md` and `references.md`, said Marconato et al. 2014 shows linear initialisation is inadequate for MLP-type nets "regardless of quality". The paper compares two initialisation SCHEMES at a fixed linear model and never varies the linear model's quality, so it cannot support that. Whether a better linear model helps is open, and is what the paired arms measure.
**Constrains**: `bla_init.py`'s `apply_bla_init` keeps its interface (the bypass write and the encoder map are unchanged); only the source of `A, B, C` changes. Any new numerical weight or threshold in the new estimator must carry `# THEORY:` or `# HEURISTIC:` before it is written. The paired retraining arms must report convergence speed and run-to-run spread, not only best validation, because that is what the initialisation literature consistently measures (Schoukens, ECC 2021; Hoekstra et al. 2026) and therefore what the thesis claim has to be. Per-Y-band BLA work is gated on the global BLA first reproducing both resonant pairs.

### [D-148] The ceiling is not the optimiser: three optimisers land in one basin, a 4x better point exists in the same function class, and the objective barely tells them apart
**Date**: 2026-08-19
**What**: Four training runs, a capability test and a band split, all on the D-147 harness. The conclusion is negative in a useful way: **every optimiser-side defect found today was real, was fixed, and changed nothing about where training stops.** Work moves to the objective. Three defects are recorded here because they are genuine and must not be re-found; two results are recorded because they bound what is achievable.

**1. Adam's `eps` is a floor on a closed-loop objective, and it silently froze 77 % of the ANN.** Adam's step is `lr*|g|/(|g|+eps)`, so `eps` is a threshold below which it stops normalising. The closed-loop loss sits at `2.2e-10` because the loop already makes the normalised residual small, and the gradients that come with it are `1e-11` to `1e-14` on the hidden layers against `1e-5` to `1e-7` on the output layers. With the PyTorch default `eps = 1e-8` the interior of both nets therefore moved at `lr/1000` while the readout moved at `lr`. MEASURED after 3 epochs: **139 of 600 ANN parameters moved, and the 139 are exactly the output layer (16x8+8 = 136)**; the hidden layers moved `1.0e-08` on weights of `0.30`, below float32 resolution (`1.5e-08`). The encoder is the same: `Wa_psi_y` and `Wa_psi_u` moved **0 of 108** entries. So the augmentation was a 136-parameter linear readout of frozen random features, and every earlier statement about "the ANN plateaus" was a statement about that, not about the augmentation. Nobody chose this: `torch.optim.Adam` defaults to `eps=1e-8`, deepSI's `init_optimizer` forwards kwargs unchanged (`fit_system.py:119-125`), Jan's `interconnect.py:523` passes only `lr`, and our `build_model` likewise. It is specific to a well-conditioned plant loop producing an ill-conditioned optimisation.
**The gradients are signal, not float32 noise**, which is the objection that had to be cleared before touching `eps`: per-tensor cosine between DISJOINT batches is **0.9988 to 0.9996 on the hidden layers**, the same as the output layer's 0.9994, and `Wa_psi_y` at `6e-15` still gives 0.98. `cos(full batch, mean of its two halves)` is **1.0000** everywhere, so rounding contributes nothing. `eps = 1e-16` (`CL_ADAM_EPS`) then makes 600/600 and 2908/3130 parameters train.
**Effect on the result: none.** `+36.19 %` against `+36.13 %` at one epoch.

**2. Raising the learning rate is not the constraint either, and the `1e-3` NaN needs re-reading.** At `eps = 1e-16`, `lr = 1e-5` ran a full epoch with no NaN, 46x more parameter travel, and a slightly WORSE result (`+35.82 %`). Note what this does to D-101/D-102: the `1e-3` NaN that pinned this pipeline at `1e-7` happened under the `eps` imbalance, where the readout ran at full `lr` into the K = 0 rows while the network behind it was frozen. **Every learning-rate conclusion in the run table predates the fix and is not safe to carry forward.**

**3. THE BASIN, which is the actual finding.** At 260 updates from the same initialisation, the fixed-batch loss lands at `1.0246e-10` (`eps 1e-8`, `lr 1e-7`), `1.0137e-10` (`eps 1e-16`, `lr 1e-7`) and `1.0251e-10` (`eps 1e-16`, `lr 1e-5`): **within 1.1 % of each other across 100x in learning rate and 4.3x in the number of live parameters.** Three genuinely different optimisers converge to one point in one epoch and then go sideways. That is a property of the objective and the parameterisation, not of the optimiser, and it is why the remaining optimiser levers were dropped.

**4. The function class is NOT the limit, but its output parameterisation is.** `cl_capability.py` regresses the SAME architecture onto the exactly computable target `phi_true(x,u) - phi_base(x,u)` (RK4 of `plant.deriv8` minus the model's own `Gantry_State_Block`, same `Ts` and `up_sample`, so discretisation cancels). Two arms differing ONLY in output parameterisation: as-parameterised today fails the two latent rows outright (`1-R^2` **0.98** and **0.91**, i.e. no better than the mean) while per-row output scaling fits them to **`9.6e-05`** and **`1.5e-04`**, and also fixes `dTh` (0.17 -> `1.7e-04`) and `dX` (0.30 -> `6.1e-04`). Cause, measured: **the eight required corrections span NINE DECADES** in normalised state units, from `3.9e-08` on X to `1.03` on the absorber latents, out of one shared output layer. Rows 6-7 are O(1) because no block but the ANN writes them, so the ANN carries the absorber's entire state evolution. X is unfittable in both arms and that is an artefact, not a finding: its target is below the float32 rounding of the baseline step it is a difference of.

**5. THE CEILING, and it beats training by 3.3x.** The regressed weights planted into the model score **`4.177e-07` m free run** (encoder-init; `4.122e-07` from the true `x0`), against untrained `2.1866e-06`, trained `1.3934e-06` and the oracle `2.81e-08`. That is **82 % of the free-run headroom closed against training's 36.7 %**, in the same function class, on the same data. Diagnostic ceiling ONLY: the target is built from `x_aug`, so this must never become a deliverable initialisation.

**6. The objective barely ranks it.** On the free run the planted model is 3.3x better than the trained one. On the training WINDOW, the quantity the loss minimises, it is **1.26x** better, or 1.99x with `W^a = 0`. The free run pays a bad initial state once in 48000 samples; the window pays it at 1666 window starts. Even with the TRUE latent state at every start the planted model scores `7.160e-07` on windows against `4.122e-07` on the free run, so 41 % of the window penalty is the latent initialisation and the remaining 1.7x is horizon.

**7. Combining everything makes it worse, so far.** `eps 1e-16` + per-row ReZero gates + `W^a = 0` + `lr 1e-5`, 2 epochs: loss `2.1788e-10 -> 1.7039e-10 -> 1.5392e-10`, free run `+19.56 %`. It is the ONLY configuration still descending at the end of its budget, and it is still 50 % above the basin the others reach in one epoch. Its own decay rate has halved between epochs, so reaching them would take 5 to 10 more epochs (2 to 3.5 h) for a best case of matching rather than beating. Not pursued.

**8. N1, THE HORIZON SWEEP: the objective is not short of optimisation, it is short of SIGHT.** `cl_nf_sweep.py`, no training, both models scored on the SAME init policy (the encoder as it actually is) over the same window grid:

| nf | seconds | planted (correct) | trained | discrimination |
|-|-|-|-|-|
| 400 | 0.1 | `1.2068e-06` | `1.5072e-06` | **1.25x** |
| 800 | 0.2 | `8.7262e-07` | `1.4392e-06` | 1.65x |
| 1600 | 0.4 | `6.4894e-07` | `1.4154e-06` | 2.18x |
| 3200 | 0.8 | `5.5332e-07` | `1.4071e-06` | **2.54x** |
| free run | 12 | `4.177e-07` | `1.3934e-06` | 3.34x |

**Lengthening the objective from 0.1 s to 0.8 s DOUBLES its ability to distinguish a correct augmentation from the one training finds**, heading toward the free run's 3.34x. The mechanism is in the two columns separately: the planted model's window error falls **54 %** with length because its error is TRANSIENT-dominated and amortises, while the trained model's is **flat at 6.6 %** because its error is PERSISTENT. At `nf = 400` both are dominated by the same transient, so the loss sees them as nearly equal and has almost no gradient reason to prefer the correct one. That is why every optimiser in finding 3 lands in the same basin.
**A statistic that misleads here, recorded so it is not repeated**: with the latent init held perfect, the planted model's ratio to the per-window FLOOR gets WORSE with nf (10.20 at 400 to 13.12 at 3200), which reads as an argument against horizon. It is not: the floor amortises the same way the planted model does (`xc = 0` at every window start is its entire error), so the floor comparison hides precisely the effect under test. Model-against-model is the correct statistic for a discrimination question.
**CORRECTED BY FINDING 10**: this finding's own conclusion, that the fix is "effective length via multiple shooting", is WRONG. Multiple shooting re-encodes at every segment start, so it preserves the transient DENSITY and cannot buy what the `nf = 3200` column shows. The horizon reading of the table stands; the proposed remedy does not.

**9. THE FIX: do not SCORE the startup transient. `K = 100` at `nf = 400` recovers the FULL free-run discrimination, and beats an 8x longer window.** Finding 8 said the excess mean-square scales as `1/n`, which is a fixed startup transient being diluted rather than an error that persists. Decomposed at `nf = 400`, using each model's own settled level: **88 % of the planted (correct) model's window loss is transient, against 15 % of the trained one's.** The objective grades a good model almost entirely on its initial state and a bad one mostly on its dynamics. The transient is an INITIALISATION error, not a model error: the planted model has **3.9x more startup energy** (`5.13e-10` against `1.32e-10` m^2 samples) precisely because it USES its latent states and the encoder cannot initialise them, so the current loss **penalises a model for using its augmented states**, 1666 times per epoch.
`cl_burnin_sweep.py`, no training, four rollouts re-reduced over the grid, objective `V = w_burn*||e over [0,K)||^2 + ||e over [K,nf)||^2`:

| `W^a` | K | `w_burn` | planted | trained | discrimination |
|-|-|-|-|-|-|
| random | 0 | 0 | `1.2068e-06` | `1.5072e-06` | **1.249x** (today) |
| zero | 0 | 0 | `7.6156e-07` | `1.5041e-06` | 1.975x |
| random | 100 | 0 | `4.2060e-07` | `1.3932e-06` | **3.312x** |
| zero | 100 | 0 | `4.0972e-07` | `1.3932e-06` | **3.400x** |
| zero | 200 | 0 | `4.0764e-07` | `1.3909e-06` | 3.412x |
| zero | 100 | 0.1 | `4.7015e-07` | `1.4080e-06` | 2.995x |

against `nf = 3200` (0.8 s) at 2.54x and the 12 s free run at 3.34x. **Discarding the first 100 samples of each window recovers the full free-run discrimination AT `nf = 400`, and beats an eight-times-longer window**, because length dilutes the transient while burn-in removes it. `K = 100` is 25 ms, which lands on the ~20 ms settling implied by the loop's 100 Hz crossover and 33.8 degree phase margin, i.e. two independent routes to the same number.
**Three consequences.** (a) **The `xc`-at-segment-boundary decision is no longer blocking**: multiple shooting was the route to 0.8 s of objective, and burn-in gets more than that at `nf = 400` with no extra gradient depth, no rate change and no new semantics. It stays available and stays unvalidated (it has never improved a production result and has only ever run open loop). (b) **`W^a` stops mattering**, exactly as predicted from the overlap: 1.975x against 1.249x at `K = 0`, but 3.400x against 3.312x at `K = 100`. Adopt zero anyway, since it is free, better at every `K`, and makes the live encoder agree with `HybridGantryEncoder` and `LinearInitEncoderWrapper`. (c) **An explicit initialisation term costs real discrimination**: `w_burn = 0.1` drops 3.400x to 2.995x with `W^a = 0` and to 2.408x with the random one. So keeping the encoder an explicitly-trained object is a stated trade with numbers on both sides, and zeroing `W^a` recovers most of what it costs.
**What this does NOT prove**: discrimination is a proxy. It says the loss can now RANK the right answer, not that training will FIND it, and with `w_burn = 0` the encoder loses its explicit criterion, which costs nothing in discrimination but says nothing about whether it degrades over a run. Both need one training run, judged on the training-window RMS, on whether the free run passes `1.39e-06`, and on the encoder's parameter delta.

**Ruled out**: (a) more epochs, more learning rate, or a different optimiser, by finding 3; (b) capacity or architecture change, by finding 4, which shows the class can represent the correction once the outputs are scaled; (c) **weighting as the FIRST move**, by finding 8: the discrimination problem is horizon, so band- or record-normalised weighting is a second-order fix and not the one to try first. Note this SUPERSEDES the reading in D-147 and in finding 8's own premise that "horizon work is premature": D-147 correctly ruled out horizon as the explanation for the ERROR LEVEL (train window ~ free run, no train/val gap), and finding 8 shows horizon IS the explanation for the objective's DISCRIMINATION. Those are different questions and both answers stand.
**10. CORRECTION: multiple shooting could never have delivered the horizon benefit, because it RE-ENCODES at every segment start. And its defect term is the ENCODER fix, which burn-in is not.** Findings 8 and 9 first framed multiple shooting as the route to a 0.8 s objective without BPTT depth. Reading `multiple_shooting.py` refutes that:

```
x_node = self.encoder(ufuture[:, s-nb : s+nb_right], yfuture[:, s-na : s+na_right])
defects.append(x_node - x)      # gradient into the encoder AND back through segment j-1
x = x_node                      # the segment starts from the ENCODER, not from the rollout
```

Every segment is re-initialised by the encoder, so `n_seg = 8` x `nf_seg = 400` produces 3200 samples containing EIGHT startup transients, i.e. **one per 400 samples, exactly the density we have today**. The 2.54x that finding 8 measured at `nf = 3200` came from ONE transient per 3200 samples. Multiple shooting holds the density fixed and would return roughly today's 1.25x. **This is consistent with the user's observation that it has never improved anything, and it retires it as a horizon device.**
**What it IS, and this matters more**: `defects.append(x_node - x)` penalises the disagreement between the encoder's estimate at time `s` and what the dynamics rolled forward to `s`, with gradient into both. That is a direct encoder criterion, and it has the property that makes it usable here: **it needs no ground truth for the latent coordinate.** It does not require the latent to be `delta_a`; it requires the estimate to match whatever coordinate the ANN has chosen, which dissolves the gauge problem that made the planted latent test a diagnostic only. It also resolves the initialisation chicken-and-egg: with the ANN output at zero the latent rows have no dynamics, so the rolled-forward latent is zero and the defect drives the encoder's latent output toward zero, which is the `W^a = 0` that finding 9 measured as better, discovered automatically and then tracked as the ANN's coordinate develops.

**THE TWO-PART FRAMING, which supersedes "burn-in is the fix":**

| | fixes | mechanism | evidence |
|-|-|-|-|
| **burn-in** | the DYNAMICS criterion | stop grading the ANN on an initialisation error it cannot fix | finding 9: discrimination 1.249x -> 3.400x at `K = 100` |
| **defect term** | the ENCODER criterion | grade the encoder on agreeing with the dynamics, no ground truth needed | not yet measured on this rig |

**Neither substitutes for the other.** Burn-in alone leaves the encoder untrained, which is a real objection and it stands: it is a workaround for the encoder, not a fix for it. The defect alone leaves the dynamics criterion still 88 % transient for a correct model. The `xc`-at-segment-boundary question therefore returns, but for a better reason than horizon, and with a leaning: if a segment exists to carry a defect term and starts from a re-encoded state, `xc = 0` is the consistent choice, because the segment is being treated as a fresh short experiment exactly as a window start is (D-142).

**11. WHAT JAN ACTUALLY DOES, read from his code, and why it does not transfer.** The closest example in the repo is `scripts/ecc_2025/msd_ndof_interconnect_dynamic.py`, and it has OUR STRUCTURE: true system 3-DOF, FP baseline 2-DOF, dynamic parallel augmentation, so `nxd = 6` of which **4 physical and 2 AUGMENTED**, ANN writing all rows with a zero-init net. His training call is

```
fit_sys = SSE_Interconnect(interconnect, na=nxd*2+1, nb=nxd*2+1, e_net_kwargs={"n_nodes_per_layer":16})
fit_sys.fit(..., batch_size=2000, epochs=2, loss_kwargs={'nf':200}, validation_measure="sim-RMS")
```

i.e. **default RANDOM encoder, joint training, nf = 200, two epochs, and nothing whatsoever for the augmented-state initialisation.** No burn-in, no defect, no multiple shooting, no pre-encoder.
**The separation idea IS his**, in `msd_ndof_pre_encoder.py`: `SS_pre_encoder` fitted at `nf = 1` for 100 epochs with `validation_measure="1-step-RMS"` against `System_data_with_x(x=train_data.x/std_x, ...)`, i.e. SUPERVISED ON THE TRUE STATE, then transplanted (`fit_sys.encoder = encoder_sys.encoder`) and trained jointly at `nf = 200` for 500 epochs. **But that script sets `sys_dof = FP_dof = 2`, so `nx_aug = 0`.** The pre-encoder was never applied to the augmented-state case, exactly as `hoekstra2026encoder`'s own experiment is a static augmentation with `nx_aug = 0` (D-148 finding relating to `W^a`, and the `references.md` row). **Two independent artefacts, the code and the paper, leave the same case uncovered.**
**Why his setup tolerates what ours does not, and it is NOT the window length.** Measured from `data/mass_spring_damper/msd_2dof.mat`: `Ts = 0.02 s`, so `nf = 200` is a **4.0 s** window; discrete eigenvalues `0.9723, 0.9723, 0.9956, 0.9956`, slowest time constant **4.5 s**, so a 4-tau settling is 18 s, i.e. **449 % of his window**. His initial-state error does not wash out inside a window either; on that axis he is worse off than we are (our 25 ms settling in a 100 ms window is 25 %). The difference is the SIZE of the initial error, not its decay rate: his states are observable from the output history and his encoder is trained on all of them from step one, while our latent rows come from a frozen random map with exactly zero gradient at initialisation (D-130) producing an arbitrary O(1) value. Our controller HELPS here: it supplies 20 ms of settling where the gantry open loop has poles at exactly 1 and never settles.
**Consequence for the design.** His answer to "the encoder needs help" is a supervised stage against the true state. We cannot copy it: the latent coordinate has no ground truth and no fixed gauge, and using `x_aug` would be oracle information that a "the method learns the physics" claim cannot rest on. **The defect term is the unsupervised version of the same idea**, tying the encoder to whatever coordinate the ANN has chosen instead of to an external target we do not have. Same intent as his pre-encoder, no oracle.

**Next**: implement burn-in and run it, because it is three lines and tests a measured hypothesis. `cfg.burn_in` defaults to 0 and is an exact no-op, since the loss currently reduces over the whole window and burn-in is `mse_loss(yfuture[:, K:], y_pred[:, K:])`. Configuration: `K = 100`, `w_burn = 0`, `W^a = 0`, `eps = 1e-16`, `lr = 1e-7`, 2 epochs. Then the defect term as the encoder criterion, which needs the `xc` decision and a gate against the `n_seg = 1` no-op. Do NOT describe burn-in alone as the fix.
**Constrains**: `CL_ADAM_EPS` sets `eps` on ALL param groups AFTER `build_model`, so it silently overrides the per-group `eps_theta` that P1-e sets on the `log_params` group. Harmless today (both `1e-16`, and `lr_theta` defaults to None so the theta group does not exist in these runs) but it must be guarded before the first joint closed-loop run. The per-row gate must use the `as_module` form: `torch.nn.utils.parametrize` cannot be pickled and `checkpoint_save_system` writes the whole `__dict__` at every validation, which is why `ANN_REZERO_GATE` had never survived a real run.

### [D-147] The training-window diagnostic is measured IN the closed loop, not with the D-095 nf-probe, and it forces `concurrent_val = False`
**Date**: 2026-08-18
**What**: The handoff's next action was "attach the D-095 `_NfProbe` to `cl_train.py`" to record the training-window error. That probe cannot answer the question it was attached for, so a closed-loop window probe is added instead and the D-095 probe is kept alongside it as a second, contrasting number. Three consequences, all decided here: (1) `ClosedLoopNfProbe` (`cl_validation.py`) measures the nf-window RMS through `closed_loop_window_rms` (`model_augmentation/fit_systems/closed_loop.py`), which re-uses `closed_loop_rollout` and deepSI's own window grid; (2) attaching any probe forces `concurrent_val = False` unless explicitly overridden; (3) the probe keeps its OWN history, and `cl_train.py` reads it from the probe object rather than from `fit_sys`.
**Why (1)**: `_NfProbe._nf_rms` calls `fit_sys.n_step_error`, which is deepSI's (`systems/system.py:311`) and drives the model through `measure_act_multi`, i.e. `hfn(x, u)` with the recorded input and no `y_data`. That is the OPEN loop. The training loss is a CLOSED-loop rollout, so the D-095 number is not the training-window error at all; reading it as one would repeat variant B exactly, an objective optimised in one loop and judged in another. The new probe uses the same rollout, the same encoder re-init per window, the same `xc = 0` and the same window starts as the loss, so "does the model fit its own training window" is asked in the loop the window is trained in. It reduces to a physical RMS over all windows, samples and channels (`sqrt(mean(e**2))`), not to the mean over per-step RMS values that `_NfProbe` reports, because the first is the quantity the oracle floor is computed in and the second is not comparable to anything.
**Why (2)**: with `concurrent_val = True` deepSI runs `cal_validation_error` in a subprocess on a `deepcopy` and the child returns only `(Loss_val, Loss_train, batch_id, time, epoch_id, bestfit)` (`fit_system.py:617-636`). Every probe side effect is written in the child and thrown away, so the diagnostic would silently record nothing. Measured cost of giving up the overlap: run 76573 did 12 epochs with in-process validation in **1h43** on the cluster, so the serial path is affordable and the handoff's "order of a day" estimate is a local-machine number.
**Why (3)**: `fit()` ends with `checkpoint_load_system('_best')`, which does `self.__dict__ = torch.load(file)` (`fit_system.py:501`). Everything a probe wrote onto `fit_sys` is replaced by the BEST checkpoint's snapshot. This is not hypothetical: it is why `step6_result_76573.json` records a three-point validation series for a run that completed all twelve epochs and printed twelve `[cl-val]` lines. A probe that only writes to `fit_sys` loses every validation after the best one.
**What the run-76573 log already settles**: the series is `2.1866, 1.3966, 1.3934, 1.3939, 1.3936, 1.3941, 1.3952, 1.3943, 1.3950, 1.3958, 1.3955, 1.3966, 1.3948` (x1e-06 m, untrained first). The run FINISHED, the best is validation 2 of 12, and the remaining ten oscillate inside +/-0.2 %. **The plateau is real and was already on the record**; section 5's "assumed but not verified" is closed by reading the `.out`, not by a new run. The training loss moves with it: sqrt-loss `1.383e-05` after epoch 1 to `1.310e-05` after epoch 12, i.e. **5.3 % over eleven epochs**, so the optimiser is stalling on its OWN objective, not only on the free-run selector.
**Ruled out**: (a) the one-line `_NfProbe` attachment from the handoff, for the reason above; it is still attached, but as the open-loop contrast, and it is labelled as such in the output. (b) Making the probe write through the concurrent path by returning extra arrays: that means changing deepSI's `_worker` protocol, i.e. editing the installed package, for a diagnostic. (c) Computing the per-window floor in a separate script: the floor must be computed on exactly the window grid the probe uses, and `window_starts` is the single definition of that grid, so it stays in the same run.
**Constrains**: `closed_loop_window_rms` and `closed_loop_free_run_rms` must stay the only two scoring paths, both on `closed_loop_rollout`; a third would be the defect D-144 removed. Any future probe must keep its own history or state that it only survives to the best checkpoint. `window_starts` is now the one derivation of deepSI's window grid and `window_controller_index` counts the same range; changing either without the other silently misaligns the controller assignment.

### [D-146] `concurrent_val = True` becomes the default for closed-loop training
**Date**: 2026-08-18
**What**: `cl_train.py` runs deepSI's concurrent validation by default (`CL_CONCURRENT=0` disables). Validation of each epoch happens in a subprocess while training continues, instead of halting the training loop.
**Why**: this was plan open item 3 and `cl_sanity.py` had recorded that `concurrent_val` MUST be False for the closed-loop path. That finding was about the MONKEY PATCH, not the loop: the concurrent branch pickles the fit system into a subprocess, a patched `cal_validation_error` does not survive that, and the child fell back to deepSI's default measure, i.e. the OPEN-loop free run. Selection then optimised one objective and chose on another, silently, which is the same class of failure the whole D-144 migration removes. With `simulator` a declared attribute holding an importable class, the mechanism is gone. MEASURED rather than argued: with `concurrent_val=True` on full-length validation records the subprocess returned **2.1866011034e-06 m** against the recorded untrained closed-loop scalar **2.1866026634e-06 m**, rel **7.13e-07**, and selection picked a trained checkpoint through `remote_recv`. An open-loop score would not agree with a closed-loop reference to seven significant figures.
**What it costs and saves**: the saving is the overlap, so it is bounded by whichever of training and validation is shorter, NOT a halving. At this machine's numbers, roughly 18 min of training and 8.5 min of validation per epoch, serial 26.5 min against concurrent 18 min: about a THIRD. It would approach a half only with more frequent validation. The child holds its own copy of the model and data, so peak memory roughly doubles, which is the thing to watch on a cluster node.
**Risk accepted, and it is stated rather than resolved**: the verification ran on Windows, whose multiprocessing start method is SPAWN; the cluster is Linux, which FORKS. Those are different code paths (spawn re-imports the module, which is how a missing `__main__` guard in `cl_train.py` surfaced and was fixed; fork copies memory and has its own hazards with threads). Child-crash behaviour is also untested: whether `fit()` hangs on `remote_recv`, records a nan, or propagates is unknown. The recommendation was to run one cheap `CL_CONCURRENT=1 CL_ITS=2` job on the cluster before the 12-epoch run; the user decided to enable it permanently, with the caveats logged here.
**How a silent failure would be caught**: two guards, because the failure this replaces was silent by construction. (1) The inline pre-fit validation is compared against `UNTRAINED_SEL`. (2) `Loss_val[0]`, which under `concurrent_val` is computed IN THE SUBPROCESS, is compared against the same number after `fit()` returns. The second is the one that actually watches the child; an open-loop free run on these records is orders away from the closed-loop value, not a few ulp, so the check cannot be passed by accident.
**Ruled out**: (1) keeping it False, which costs about a third of wall clock on every run for a failure mode that has been measured away. (2) A `__main__` guard alone without the scalar checks: the guard fixes the crash, not the silence, and the crash was never the dangerous part.
**Constrains**: `cl_train.py` must stay inside `main()` under a `__main__` guard, or spawn-based platforms re-execute the module on every validation. Any future change to how the simulator is attached must keep it picklable and importable by name; that is the property the whole thing rests on.

### [D-144] The closed loop moves into `model_augmentation/` behind four seams, and the monkey patches go
**Date**: 2026-08-17
**What**: Executed `scripts/gantry/closed-loop-controller/PLAN-move-to-model-augmentation.md`. The closed-loop training path is now `model_augmentation/fit_systems/closed_loop.py` (`ControllerBank`, `closed_loop_rollout`, `ClosedLoopSimulator`), reached through four declared seams on `SSE_Interconnect`: `simulate()` delegating to a `simulator` attribute, `make_training_data` and `cal_validation_error` overrides that call `super()` then delegate, and a `validation_probes` tuple. `Interconnect` gained `output_only()`, which evaluates the output signal's dependency cone. `SSE_Interconnect_MultipleShooting` routes its segment rollout through `self.simulate()`. Deleted: `cl_fitsys.py` (a fit-system class created at runtime by `type()`), `ClosedLoopValidator.install()`, `loss_variants`' B and C mixins and `attach()`, `cl_controller`'s duplicate rollout and bank, and `_NfProbe.__reduce__`. `_install_nf_val_probe` is now a `validation_probes` entry.
**Why**: The old path grafted a class onto the fit-system instance at runtime and patched `cal_validation_error` twice, so `param_loss` and the orthogonality penalty were re-implemented by hand in three files, checkpoints pickled a class that only existed after `attach()`, and whichever patch was installed last decided checkpoint selection. All four failure modes are structural, not accidental, and all four are removed by construction: a probe cannot change the selection value because the seam ignores its return; the penalties are never mentioned by closed-loop code so they cannot be dropped; the simulator holds no model handles so `checkpoint_load_system` cannot leave it pointing at stale modules.
**Verified against three independent references** (`scripts/gantry/closed-loop-controller/references/`): (A) MATLAB, all eight levels of `test_controller_exact.py` pass including the new L5 at the 4 kHz TRAINING rate, which did not exist before; (C) R1, the production loss with no simulator, BIT-IDENTICAL including gradients through the seam extraction; (B) R2, the framework closed loop against the implementation it replaces, on a frozen pre-migration recording that cannot be regenerated: loss rel 1.4e-05, gradient `1 - cos` 5.3e-09, trajectory 2.8e-07, units 7.1e-08, selection scalar 1.56e-12 m against a 1e-10 m tolerance, and 66626 of 66626 training windows carrying the controller their record actually had.
**Runtime, measured before optimising anything** (`cl_step0_profile.py`): the regime is dispatch-bound, not FLOP-bound. A `(32,8)@(8,8)` matmul runs at about 60 MFLOP/s, so operation COUNT is the cost and FLOPs are free. The controller is 8.1 % of the forward pass with a 7.6 % ceiling on any optimisation of it; `bank.gather` is 0.002 % and the encoder 0.012 %. The largest single item in a training step is `blocks.py`'s per-timestep `deriv` at 22 %, which this migration does not touch.
**Ruled out**: (1) `SSE_Interconnect_ClosedLoop(SSE_Interconnect_OrthLoss)`, a fourth link in the loss chain: couples two unrelated concerns and makes the closed loop depend on the orthogonality penalty for no reason. (2) The block-diagonal `Cfb` storage: cuts FLOPs 3x and the operation count by zero, so it buys nothing in a dispatch-bound regime, and it conflicts with the stacking that does cut the count. REVERSES the plan's earlier recommendation, which predated the profile. (3) A checkpoint compatibility shim: it would keep the runtime-class machinery alive purely to read files produced by the structure being removed. (4) Deriving `Cfb` from a window's measured `Y`: the machine froze the controller at each record's NOMINAL `Y_op`, so a per-window derivation applies a different operator and is silently wrong on exactly the `ysweep` records.
**Constrains**: `simulate()` returns `(y_pred, x_final)` and every `loss()` in the chain takes `*sim_args`, because deepSI's `fit()` passes simulator arrays POSITIONALLY. `fit()` must be called with `validation_measure='sim-RMS'`: the closed-loop validator computes a full free run scored in metres and now RAISES on anything else, including deepSI's own default of `sim-NRMS`. The first version of that guard tested `startswith('sim')` and would have returned an RMS under the NRMS name, which is the same silent-substitution failure this decision exists to remove, found by re-reading the code rather than by any gate. The scoring path is `closed_loop.closed_loop_free_run_rms`, one implementation: unifying the rollout while leaving two copies of the scoring would have been the identical defect one level up. `n_seg > 1` with a simulator attached RAISES: whether the driver resets its own state at a segment boundary is an open modelling question, and the `xc = 0` argument (D-142) is about a window start, not a segment start. Existing `FitSys_ClosedLoop_*.pth` no longer load, by decision; the retrain is a cluster job (`cl_train.py`).

### [D-145] The closed-loop gradient is not a usable regression quantity at large ANN perturbation, and one pre-existing defect blocked step 4's gate
**Date**: 2026-08-17
**What**: Two measurement findings from executing D-144, both of which changed what gets checked rather than what gets built. (1) The perturbed-ANN arm of the closed-loop reference uses `sigma = 1e-4`, not the `1e-2` used on the open-loop arms. (2) `SSE_Interconnect_MultipleShooting.loss` could not run at `n_seg > 1` with ANY encoder in this codebase, and was fixed at the caller with `.contiguous()`.
**Why (1)**: At `sigma = 1e-2` the closed-loop rollout is chaotic and the gradient is not well defined at any precision. The SAME implementation gives loss `4.81e-02` in float32 and `2.29e-03` in float64, a factor 20. Two implementations that agree to `1 - cos = 8.9e-16` at `1e-4` give `1 - cos = 1.87` at `1e-2` IN FLOAT64. Sweep, float64, old against new: `1e-2` -> 1.87, `1e-3` -> 4.6e-05, `1e-4` -> 8.9e-16. A quantity two precisions of one code disagree on by 20x is a coin flip, not a regression net. At `1e-4` every ANN parameter is still perturbed and the closed-loop loss is `4.57e-06` against `2.54e-10` at zero, four orders above the zero-ANN arm.
**Why (2)**: the method builds each interior node with `self.encoder(ufuture[:, s-nb : s+nb_right], ...)`, and a time-axis slice of a contiguous `(batch, nf, nu)` tensor keeps dim-0 stride `nf*nu`, i.e. it is not contiguous, while both encoders reshape with `.view`: `RuntimeError` at `pre_encoder.py:450` for `encoder_init='linear_map'` (the production encoder) and at `interconnect.py:384` for `'default'`. deepSI's `to_hist_future_data` hands the encoder contiguous windows, so the `n_seg = 1` production path never touched it and nothing noticed. Consequence worth carrying: the defect diagnostics that justify keeping `multiple_shooting.py` cannot have been produced through this method.
**Ruled out**: widening the gradient tolerance to accommodate the `1e-2` disagreement, which would have hidden a real regression of that size behind a number chosen to make a bad test point pass; and leaving the contiguity fault in place, which would have left migration step 4 with no numeric gate at all.
**Constrains**: any future closed-loop regression check states its perturbation amplitude and stays at or below `1e-4`. The reference checker (`cl_step1_reference.py --check`) refuses to compare across a different build fingerprint or thread count, carries a per-key comparison class rather than one global tolerance, and reports keys present in the recording that a run stopped computing, so a check that silently stops running cannot read as a check that passed.

### [D-143] State reconstruction for the orthogonal-projection basis: decide by measurement, one construction for all SNR arms
**Date**: 2026-08-17
**Status**: PRE-REGISTERED, outcome pending. Criteria stated before the run (D-090).
**What**: Choose, by measurement rather than argument, the single construction that produces the states `X` feeding `Phi(X,U)` in the penalty basis. The chosen construction is used for **every** SNR arm including `snr=None`. Candidates:
  (A) FP rollout at `theta_bar` (the paper's own fallback for the no-full-state case, GYOROK end of Sect. 3): never touches `y`, so noise-immune, but drifts off the data manifold on the K=0 axes.
  (B) Static inversion + forward difference (current D-111): `q = P^-T y`, `qdot = diff(q)*fs`.
  (C) Central difference: same, without the half-sample delay of (B).
  (D) Savitzky-Golay differentiator: polynomial least squares over a window.
  (E) Kinematic RTS smoother: constant-acceleration model per logical channel, `theta`-independent.
**Why one construction for all arms**: the SNR arms exist to measure what noise does to the method. Changing the state construction between arms confounds noise with construction and destroys the comparison. It is also what [GYOROK] does (`x_meas=True` across 0/30/25 dB).
**Why this needs deciding now**: (B) is the only candidate containing a differentiator, and at `fs = 4 kHz` it amplifies output noise by `sqrt(2)*fs = 5657`. Measured consequence (data.py): `sigma_v = 0.79 m/s` at SNR 60 against true velocity stds 0.008-0.48 m/s. D-111 selected (B) over (A) on measured leakage (0.017 vs 0.164) but that measurement was taken at `snr=None` only, so the D-111 ranking does not automatically carry to the noisy arms. `build_orth_penalty` now raises on `cfg.snr is not None` rather than silently building a meaningless basis; this decision is what lifts that guard.
**The `P`-transform asymmetry (why `dTheta` is the hard channel)**: `y_stage = P^T q` with `Lb = 0.725` gives `q1 = (X1+X2)/2` (sum: noise *reduced* by `1/sqrt(2)`), `q2 = (X1-X2)/Lb` (difference: noise *amplified* by `sqrt(2)/Lb = 1.95`), `q3 = Y` (unchanged). `dTheta` also has the smallest true signal (0.008 m/s vs ~0.2/0.4 for `dX`/`dY`) because the beam is stiff by design. Chain check: `1.4e-4 * 1.95 * sqrt(2) * 4000 / 0.008 = 193x`, reproducing data.py's measured `dTheta` inflation at SNR 60 exactly. The information is lost in the measurement geometry, before any processing, so no `y`-only estimator can recover what the encoder difference destroyed.
**Test design (two stages, pre-stated)**:
  Stage 1 (no Jacobians, seconds): per-channel velocity RMSE against the stored ground-truth `x_logical[:, 3:6]`, normalised by each channel's true std, for candidates (B)-(E) at `snr` in {None, 60, 55, 50}. Smoother/window hyperparameters are swept and the BEST is reported, making this an **upper bound** on achievable accuracy: if even the oracle-tuned estimator fails a channel, that channel is unrecoverable by that construction, which is a valid negative result. Oracle tuning is used only to establish feasibility here, never to select a production setting.
  Stage 2 (survivors only, ~1 min per build at coarse stride): build the basis from each surviving construction, measure principal angles against a reference basis built from the ground-truth states, plus the step7c-style leakage floor. Same metric as D-111 so the numbers are directly comparable to 0.017 / 0.164.
**Pre-stated selection rule**: choose the construction with the best WORST-CASE normalised velocity RMSE across the four SNR settings, subject to its noiseless-arm leakage floor not being materially worse than D-111's 0.017. Ties broken toward the simpler construction.
**Pre-stated prediction**: `dX` and `dY` are recoverable by (E) at all SNRs; `dTheta` is not recoverable by any `y`-only candidate at SNR 60, on the bandwidth argument above (recovery needs a filter bandwidth under ~7 Hz while the `dTheta` content sits at 130-180 Hz). If that holds, the choice narrows to (A) or to a model-based observer, which would make the states `theta`-dependent and reintroduce the coupling [GYOROK] deliberately avoids (its `Pi` is parameter-independent, Sect. 3 after Eq. 11).
**Ruled out already**: low-pass then difference, as a standalone fix. data.py measured that it buys only `sqrt(D)` (0.79 -> 0.354 m/s) and does not recover `dTheta`, whose content lies inside the band where differentiated noise dominates.
**Constrains**: the outcome sets the state source for every future orth run and for the Telica arm, where neither pre-noise signals nor a trustworthy rollout exist. It is one of four changes landing in the same basis rebuild (D-103 `[0..5]` routing, Y-stratified point set, new states, re-derived `beta_center`), so all Stage A numbers are regenerated once against the chosen construction.

### [D-142] `xc = 0` at each training-window start IS Kessels' Remark 5.4, the closed-loop metric has 77x headroom, and step 3's two gate failures were defects in the gates
**Date**: 2026-08-16
**What**: Three findings that together green-light the closed-loop training loss. (1) **The controller-state initialisation question is settled: `xc = 0` per training window is not a shortcut, it is exactly Kessels' Remark 5.4.** No warm-up, no encoder extension, no cached rollout. (2) **The closed-loop metric has a 77-79x headroom** between the baseline and the FP+MSD oracle, against 2.35x open loop, so it is worth selecting on. (3) Step 3's G7 and G10 failures were both defects in the gates, not in the code; both are rewritten.
**Why `xc = 0` is Remark 5.4, verified verbatim against thesis p157**: (5.13d) is `ê_{τ+k|τ} = r̄_{τ+k} - ŷ_{τ+k|τ}`, where the hat on `ŷ` is the EA MODEL output, so Kessels' own `x̂^FB` is model-dependent exactly as ours is. Remark 5.4 then says the true machine states `x̄^FB` "may be reconstructed using the output measurement `ȳ`, the known reference profile `r̄_k`, and the controller ... (hereby assuming `x̄^FB_{k=1} = 0`)", and that "This information can be used to initialize the feedback states for each TW". So he initialises the MODEL's controller state with the MACHINE's, reconstructed from measured output. Working the error through: `x̄^FB(τ) - x̂^FB_true(τ)` is the state of `Cfb` filtering `(ŷ - ȳ)`, i.e. `-xc_A(τ)` in our notation. In the residual form `u = u_data + Cfb*(y_data - y_model)`, the recorded `u_data` already carries the machine's controller history implicitly, so setting `xc_A = 0` makes the model's effective controller state exactly `x̄^FB(τ)`. **Same object, exactly.** Supporting: §5.3.2.1 (p171) confirms the industrial case extracts `x^FB` from measurement data so the encoder initialises only `x^(2)`; and the reconstruction runs from `k = 1` of the record rather than per window, which is what `u_data` supplies for free.
**Measured cost of that choice (step 5, `cl_step5_reset_cost.py`, `runs/cl_step5.log`)**: two measurements on T1, T10, V1, V2 with the ANN forced to zero. M1 isolates the controller state by driving it through a FIXED residual sequence from one continuous rollout, `xc` carried against `xc` reset every `nf`: difference is 7.2-22.0 % of `rms(u_fb)`, with the DC share splitting sharply by record type, 26-29 % on standstill against 70-82 % on APRBS motion. M2 uses the actual training geometry, 40 windows, both arms encoder-initialised, only `xc0` differing: the reset perturbs the window trajectory by **8.5-13.4 % of the model error the loss is fitting**. Median recovery is 6 samples of 400. **This 13 % is the cost of KESSELS' method on our system, not of a shortcut of ours**, and it is set against the 77x headroom below, so it is not the binding constraint.
**Headroom, measured (`cl_headroom.py`, `runs/cl_headroom.log`)**: closed-loop free run, `plant.deriv6` (no absorber) against `plant.deriv8` (true absorber), SAME numpy harness, same integrator, same rate, same `up_sample`, both seeded from the true state at `K0` (D-097 fairness, D-087 interior seed). Baseline agg `2.185e-06` to `2.198e-06`; oracle agg `2.780e-08` to `2.842e-08`. **Factor 76.9 to 79.0 aggregate, 107.4 to 108.4 on Y, 15.5 to 16.3 on X.** Open loop the same prize was 2.35x on Y (handoff s6). The loop AMPLIFIES the absorber mismatch because V1's content sits at 130-180 Hz where `sigma_max(So) = 2.07`, i.e. D-139's waterbed working in the augmentation's favour. Consequence: **the `2.19e-06` plateau seen on every record is the absorber mismatch, a plant property, not a discretisation floor**, which is why it is trajectory-independent. D-141's 4 kHz decision stands.
**Metric sensitivity (`cl_diag_step3.py` D2)**: perturbing the ANN final layer moves the closed-loop score `1e-4 -> +0.7 %`, `1e-3 -> +18 %`, `1e-2 -> 5.70x`, monotonically. Combined with the headroom this is the opposite of the open-loop situation that produced `VERDICT: ANN inactive` and best-checkpoint-equals-epoch-0, where drift moved the metric 36x and swamped any ANN effect.
**Step 3's gate defects, both mine**: (1) **G7 was too strict.** It demanded the closed-loop MODEL score equal the closed-loop BASELINE score to `1e-9` at init. `cl_diag_step3.py` D1 located the cause: with BIT-IDENTICAL encoder parameters (`0.000e+00`) and identical inputs, the live encoder and a `deepcopy` of it produce `x0` differing by `1e-6` to `1e-5`, because `deepcopy` reallocates the parameter tensors and changes BLAS reduction order in the encoder's ~100-wide matmuls. Decomposed into G7a (ANN live-but-zero against ANN patched off, everything else fixed, MUST be exactly 0) and G7b (live encoder against its deepcopy, gated at the MEASURED float32 floor). G7a returns exactly `0.0`, so there is no wiring bug. (2) **G10 asserted a falsehood.** It demanded the D-139 waterbed (closed/open > 1) on the encoder-init free run. That signature only appears when the open-loop run is already near-perfect, as in the step-2 true-`x0` replay where it IS correctly gated by G5/G6. From a realistic encoder init the loop's suppression of the initial-state error dominates, so closed/open < 1 is correct. Replaced by an initialisation-insensitivity gate, which measures what step 3 actually established: closed-loop spread between encoder-init and true-`x0` is **0.003-0.005 %**, a reduction of **393x to 371,000x** against open loop. (3) A third defect fixed in passing: the true-`x0` baseline was seeded with the analytic rest state of sample 0 while the run starts at `K0`, which is not the true state there and injects a velocity error the `K = 0` axes integrate (the D-139 artefact). Now seeds `x_logical[K0]` per D-072/D-087.
**Ruled out**: (1) **Warm-up lead-in with a swept `n_w`.** Rejected on principle before the Kessels check, and the reasoning is worth keeping: `xc = 0` asserts the model tracked the data over the pre-window history, which is false by exactly the model error, so the discrepancy is structurally coupled to the quantity being learned and **the correct `n_w` would shrink as the ANN improves**. A constant swept against the untrained model is wrong at every later epoch. (2) **Switching to the lumped-`r` form to enable Remark 5.4.** This was proposed mid-session and is WRONG: with `xc_A = xc_D - xc_B`, the residual form setting `xc_A = 0` and the lumped form setting `xc_B = xc_D` are the SAME assumption. The claim that the lumped form's error is "second order on a large quantity" against the residual form's "first order on a small quantity" was an error and is retracted. (3) **Adding `xc` to the encoder output** (`nx` 8 -> 17). Not needed once `xc = 0` is recognised as Remark 5.4, and it would reopen the D-066/D-067 routing question for no measured benefit. Still available if a run stalls in a way that points at initialisation bias. (4) **Cached continuous rollout refreshed per epoch.** Same: correct but unnecessary, and it adds a staleness parameter. (5) **Making validation match training by scoring with per-window resets.** Kessels does the opposite: (5.14) evaluates on a SINGLE non-truncated window over the whole record with `τ = n_o`, and he justifies the train/evaluate asymmetry explicitly on cost (p159-160). Matching training would also stop the metric being the deployment metric.
**Constrains**: the closed-loop training loss uses `xc = 0` at every window start, and any future deviation must justify itself against Remark 5.4 rather than against convenience. The 13 % M2 figure is the pre-registered number to compare against if a run stalls: a stall that is NOT accompanied by evidence of initialisation bias should not be attributed to `xc`. Gate thresholds asserting bit-exactness across a `deepcopy` boundary are invalid in float32 and must either run in float64 or calibrate against a measured floor. No gate may assert the waterbed signature on a free run started from an encoder estimate.

### [D-141] The closed-loop path runs at 4 kHz, lives in `scripts/gantry/` as a subclass first, and carries the per-record controller through a fifth training-data array
**Date**: 2026-08-16
**What**: Implementation decisions for putting `Cfb` in the model (D-140), taken before writing code. (1) **Rate: 4 kHz**, the existing pipeline rate, with `Cfb` re-discretised there. (2) **Location: `scripts/gantry/closed-loop-controller/`**, as a mixin grafted onto the fit-system instance, following the pattern `loss_variants.py` already uses. `model_augmentation/` is not modified. The user's stated intent is to test it there and migrate it into `model_augmentation/fit_systems/` natively once it works, so the controller subsystem and the rollout are written framework-agnostic and liftable rather than entangled with the gantry pipeline. (3) **Per-record `Cfb` is carried by a fifth array.** `make_training_data` is overridden to return `[uhist, yhist, ufuture, yfuture, rec_ix]`; `fit_system.py:393` calls `self.loss(*train_batch)` and `My_Simple_DataLoader` slices every array in the list by the same shuffled ids, so the record index arrives in `loss` correctly shuffled and batched.
**Why 4 kHz**: measured in `p2_rate_compare.py`, re-discretising at 4 kHz moves `sigma_max(So)` at 150 Hz by **+15.3 %** (that script's own 10 % tolerance flags it) and phase margin by 3.6 degrees (inside its 5 degree tolerance). So the training loop is close to, but not identical to, the loop that made the data, and the discrepancy sits in the absorber band the augmentation must learn. That is accepted as a **known, stated bias** rather than a hidden one, against the alternative of running the closed-loop path at 20 kHz, which multiplies the pipeline by 5 in `nf`, memory and wall clock and collides with D-137's finding that the horizon must be set in seconds. The zero-ANN replay gate (D-140 step 10) is run at both rates so the cost is read off the actual signals rather than from `sigma_max(So)` alone.
**Why `scripts/gantry/` first**: keeps Jan's framework clean and mergeable, needs no `@added` or `# CHANGED` markers, and is reversible if the closed-loop objective turns out not to help. The cost is that the rollout is overridden rather than native, so a future framework change could drift from it; the migration into `model_augmentation/fit_systems/` is the planned end state, not an afterthought, and the module split is chosen to make it a move rather than a rewrite.
**Why the residual form needs no new loaded signal**: `u_plant = u_data + Cfb*(y_data - y_model)` uses only `u_total` and `y`, which `load_traj` already returns. `r_sim` and `f_sim` are NOT needed by the training path at all; they were needed only for the D-140 verification. The one thing that must be added to the data path is the record's `Y_op`.
**Ruled out**: (1) **Inferring `Y_op` from `yfuture`**: the Y channel of `y` is the Y position, which equals `Y_op` only on standstill records; T6-T14, V2-V4 sweep `Y`, so it is not recoverable from the window. (2) **Appending `Y_op` as a fourth input channel**: changes `nu`, feeds the ANN a new input, and perturbs normalisation. (3) **Recovering record identity from window ordering alone**: `System_data_list.to_hist_future_data` does concatenate per-record blocks in `sdl` order, so it is recoverable in principle, but it depends on an exact window-count formula that would silently break on a `stride`, `na`, `nb` or `nf` change. The explicit index array cannot drift.
**Constrains**: every number produced by the closed-loop path is at 4 kHz and carries the +15.3 % `sigma_max(So)` bias in the absorber band; that must be stated wherever those numbers are compared against the 20 kHz records. The fifth-array convention makes the loss signature `loss(uhist, yhist, ufuture, yfuture, rec_ix)`, which is incompatible with any sibling mixin that overrides `loss` with the four-argument signature: `loss_variants.py` variants B and C must not be attached at the same time. The migration into `model_augmentation/` is a separate, later decision and is not authorised by this entry.

### [D-140] Cfb is verified exact against the records, its acceptance criterion was wrong, and it stays a separate subsystem rather than joining the interconnect state vector
**Date**: 2026-08-16
**What**: Two things, in that order. (1) `scripts/gantry/closed-loop-controller/verify_cfb_against_records.py` (read-only, new) measures, on all **22** records in `data/gantry/matlab/trajectory/augmentation/` and per channel, the identity `u_total - (u_fb + f_sim)` and the residual between the stored `u_fb` and the `u_fb` recomputed by rebuilding `Cfb` at that record's `Y_op` via `loss_variants.controller_ss` and simulating it on `r_sim - y` from rest, both over the full record and with the first 0.5 s discarded. Log: `scripts/gantry/closed-loop-controller/runs/vcfb2.log`. E1-E4 are included and labelled `test`: this is a signal-identity check on the generator, not a model evaluation, so no test information is consumed. (2) The section-8 placement question is settled: **`Cfb` becomes a separate subsystem stepped alongside the model in the rollout, not a block whose states join the interconnect state vector.** Nothing implemented; `model_augmentation/` untouched this session.
**Measured, the controller structure**: per-channel order `[3 3 3]`, **`n_FB = 9`**, identical at every operating point. `D` is nonzero and large, diag `[2.844e+06 2.914e+06 1.509e+06]` N/m at `Y_op = 0`, i.e. `Cfb` has direct feedthrough. Nine DISTINCT controllers across the 22 records: `kappa` moves 1.5x on X1 and X2 between `Y_op = +0.30` and `-0.30` (`2.019e7` to `2.959e7` on X1) and 0.4 % on Y (`1.155e7` to `1.160e7`).
**Result 1, additivity**: `u_total = u_fb + f_sim` holds on every record and every channel. `max abs` is exactly one float32 ulp of `u_total` in each case (e.g. `1.144e-05` N against a step of `1.526e-05` N); relative to `rms(u_total)` it is `3.3e-08` to `3.9e-08` throughout. E1's X channels and all of E4 are identically zero (no injected excitation). Five orders inside the `1e-3` criterion.
**Result 2, the controller residual, and why the acceptance criterion was the thing that was wrong**: the criterion `rms(u_fb_recomputed - u_fb)/rms(u_fb) <= 1e-3` is NOT met on the Y channel of T1, T2, T4, T5, V1, V3, E2 (up to `4.66e-01` on T1) nor on all three channels of T9-T12, V2, E3, E4. **It is not achievable, and the controller is not at fault.** The criterion assumed the float32 storage floor behaves like broadband noise. It does not: `Cnorm` has a pole at `s = 0`, so the tustin controller has a pole at `z = 1` and integrates any DC in `e = r_sim - y` into a RAMP. On the standstill records that DC is the float32 rounding error of the CONSTANT Y setpoint, which does not average out while the moving `y`'s rounding does. Measured, `mean(e_Y)` against `float32(Y_op) - Y_op`: `-0.30` gives `-1.2243e-08` against `-1.1921e-08` (ratio 1.027), `-0.15` gives `-6.0298e-09` against `-5.9605e-09` (1.012), `+0.15` (0.951), `+0.22` (1.021), `+0.30` (0.968), `+0.10` (0.858). `T3` at `Y_op = 0.00` is exactly representable and has no ramp (`3.80e-08` relative, exact). The fitted ramp explains 99.92-99.997 % of the Y residual on those records and its slope matches `kappa_j*w/54*mean(e_j)`, the integral gain times the measured bias, to 1.3-5.2 % (T1 `-1.6076` against `-1.6525` N/s; T5 `+1.6114` against `+1.5579`). Remove the ramp and the Y residual lands at **0.77-0.80x the measured storage floor** on every one of them. V3 is the partial case: it is a Y-SWEEP at `Y_op = +0.10`, so the bias is time-varying and a straight line removes only 41 %.
**How the remaining regime was closed**: on the records where X moves (T9-T12, V2, E3, E4) the residual is `6e-04` to `1.4e-02` relative and 6-33x the storage floor the script measures. That floor is an underestimate there BY CONSTRUCTION, because the script perturbs only `y`, while on those records `r_sim` is also a large moving signal whose float32 rounding is correlated in time and therefore concentrated where `|Cfb|` is 1/f large. The confound was removed with the instrument that already existed, `test_controller_exact.py`, which re-runs MATLAB's own `lsim` on exactly the bits Python forms: on `T10_aprbs_60` (worst of that regime) the agreement is `[3.97e-10 4.66e-10 2.17e-10]` with identical input bits against `[1.28e-03 1.21e-03 3.03e-03]` against the STORED `u_fb`, and on `V1_standstill_Yp10` `[5.89e-11 8.50e-11 4.32e-11]` against `[7.55e-08 6.89e-08 2.76e-02]`. L1 (coefficients, no simulation) `9.59e-12`, L2 `1.46e-16`, L3 `1.14e-09`, L4 `4.66e-10`, all PASS.
**Verdict**: `Cfb` as rebuilt in Python IS the controller that generated the data. **D-139's closed-loop numbers do not need revisiting.** Coverage stated honestly: the bit-exact gate covers 2 of 22 records, one from each otherwise-unexplained regime, and L1 checks the coefficients at `Y_op = 0.10` only. The `Y_op` dependence enters solely through `kappa_j`, and the run tests it on X1 and X2 at all nine operating points, where the residual is `3.1e-08` throughout.
**Why the controller stays a separate subsystem (Kessels 2025 Chapter 5, read pp144-183 including 5.3.2)**: (1) **Kessels keeps the constraints apart.** (5.13b) is the EA model, (5.13c) the control input, (5.13d) the FB controller's own dynamics with its own state `x^FB`; the encoder (5.13a) returns only `x^(1,p), x^(1,v), x^(2,p), x^(2,v)`. The sentence after (5.13d), p157: since `f_FB` and `g_FB` are known, the FB controller may be regarded as part of the known closed-loop system. Figures 5.2 and 5.4 draw FB/FF in one box and the EA model in another. (2) **The encoder must not be handed states we can compute.** Remark 5.4 reconstructs `x^FB` from `y`, `r` and the known controller assuming `x^FB = 0` at `k = 1`; §5.3.2.1 (p171) then states the consequence on the industrial case: the FB states are extracted from data and are NOT initialised by the encoder, so the encoder initialises only `x^(2)`. Remark 5.3 pushes the same way via `h^-1`. Joining the interconnect takes `nxd` from 8 to **17** and hands the encoder 9 exactly-known states, the opposite of what Kessels does; `n_FB = 9` makes that material rather than marginal. (3) **`Cfb` is not one object**, measured above: nine instances, `kappa` varying 1.5x with `Y_op`. A block inside the state vector would change the model's state dimension and dynamics with the data batch. Kessels' §5.3.2.3 generalization test is the converse and makes the point: ONE trained EA model (`n_ext = 14`) is re-evaluated under a different FB controller C2, gains +-20 % against the training controller C1 (Tables 5.6 and 5.7, rows R2/C2 and R3/C2), no retraining. That test is only expressible if the controller is outside the model, and this project needs the same freedom because evaluating at a held-out `Y` already means a different `Cfb`. (4) **Orthogonality.** The projection basis is `d(FP)/dtheta`. With `Cfb` states inside the interconnect that sensitivity is taken through the closed loop and emerges premultiplied by `So`, which is exactly the objection already used to exclude merging `Cfb` into the FP block. Remark 5.2 names this thesis's contribution as an open problem, so there is no guidance to borrow and the argument stands on its own. (5) **Feedthrough and well-posedness.** `D ~ 3e6` N/m, measured. Remark 5.1 is precisely this concern: `l^(3)_w3` is forbidden from depending on `u_hat` because `u_hat` depends on `u^FB` which depends on `y_hat`, and the relation would be non-causal. Our loop is well posed only because the plant has `D_d = 0`. Outside, the step ordering is explicit in the rollout; inside, the interconnect graph's acyclicity becomes contingent on a property of a different block and any later output augmentation breaks it silently.
**Ruled out**: (1) **A block whose states join the interconnect state vector** (the user's initial proposal), per the five reasons above. Its one genuine advantage is real and is answered rather than dismissed: the closed loop becomes a property of the model so training, evaluation, diagnostics and selection cannot silently disagree, which is the exact failure that invalidated variant B. But that failure's cause was that the loop lived in the LOSS, which only the training path executes. The section-8 framing of "by construction against by discipline" overstates the gap: a single closed-loop `simulate()` that training, validation and checkpoint selection all call is construction too. What must hold is that there is exactly ONE rollout entry point and no path can reach the plant without it, and that is a smaller and more reversible commitment than +9 states in the encoder's output. (2) **Merging `Cfb` into `Gantry_State_Block`**, already excluded: parameter sensitivities would come out premultiplied by `So` and it breaks the standalone-baseline negation test. (3) **The controller in the loss** (`ClosedLoopLossMixin`), which is the implementation being replaced. (4) **Kessels' Remark 5.6 as an argument for the closed-loop form here.** He reports that open-loop identification of the wire bonder was attempted and failed, attributed to noise-induced bias from correlation between input and output measurement noise together with the FB integrator. Our records are noiseless simulation (`snr=None`), so that specific argument does NOT transfer and must not be quoted as if it did.
**Constrains**: the acceptance criterion `rms(residual)/rms(u_fb) <= 1e-3` on a raw record-level controller residual is retired and must not be re-used; any future controller check either compares coefficients (L1), or runs on identical input bits (L4), or compares against a measured storage floor that perturbs BOTH `r_sim` and `y`. The `residual / storage floor` column in `verify_cfb_against_records.py` perturbs only `y` and is therefore an underestimate wherever the reference moves; it is a diagnostic, not a gate. The implementation session inherits: one shared closed-loop rollout entry point used by training, validation and selection alike; the residual form `u_model = u_data + Cfb*(y_data - y_model)`, which makes `xc = 0` exact at every window start and avoids Remark 5.4 reconstruction; `Cfb` rebuilt per record at that record's `Y_op`; and `ann_route_ix` at all eight states, which is decided and not reopened here.

### [D-139] The open-loop replay offset is an initialisation artefact, not physics: the only genuine DC mismatch is on X, and the absorber is a resonance
**Date**: 2026-08-15
**What**: The baseline's open-loop replay error against the closed-loop records is fixed at, per channel `[X1, X2, Y]`, `V1_standstill_Yp10` rms `2.992e-08 / 2.995e-08 / 2.260e-06 m` with mean `-1.819e-08 / -1.819e-08 / -4.441e-07 m`, and `V2_aprbs_Ylow` rms `2.154e-04 / 2.416e-04 / 4.597e-05 m` with mean `+8.204e-05 / +8.214e-05 / -6.767e-08 m`. Two independent implementations agree: `Gantry_State_Block` standalone (6 states, no ANN object) against `fs.hfn` (8 states, ANN forced to zero) differ by at most `1.5e-11 / 1.5e-11 / 6.5e-09 m` on V1 and `5.0e-09 / 6.9e-09 / 2.0e-08 m` on V2, all below the `3.9e-08 m` numerics floor. The previously reported `~1e-1 m` V1 Y offset is not an offset: `Y_op = 0.10 m`, so that panel plots absolute output rather than error. Comparison script: `scripts/gantry/dc-accumulation/compare_annoff_replay.py`; figures `annoff_replay_compare_<record>.png` there and `closed-loop-controller/figures/baseline_drift_replay_<record>.png`.
**Why the offsets were artefacts, with the mechanism**: writing `np.zeros` into a NORMALISED state vector means physical `norm.x_mean`, which on the velocity states is `dX +8.403e-05`, `dTh -3.167e-06`, `dY +1.455e-04 m/s`. X and Y have no restoring stiffness (`K = 0`), so that velocity error integrates to a permanent position offset `v0 * tau` with `tau_X = 1.546 s` and `tau_Y = 1.010 s`. Measured contribution `+1.13e-04 / +1.13e-04 / +1.36e-04 m` on V1 and `+1.13e-04 / +1.13e-04 / +1.36e-04 m` on V2. **It is RECORD-INDEPENDENT, which is the signature that distinguishes an initialisation artefact from a model error**, whereas the true error changes by four orders of magnitude between records. On V1 the artefact is 6400x the real X error and 300x the real Y error.
**Why the "circa 6 second slow dynamics" is also this artefact**: the approach to the asymptote is `v0 * tau * (1 - exp(-t/tau))`, and 3 to 4 time constants is `4 * 1.546 = 6.2 s` on X and `4 * 1.010 = 4.0 s` on Y. That is the observed 6 s, and it is the settling of a wrong `x0` on the `K = 0` axes, not a plant or residual timescale. **Consequence: the hypothesis that the ANN fails because the residual has slow dynamics the `nf = 0.100 s` window cannot see is FALSIFIED.** The residual is the absorber at 150 Hz, `tau_msd` about 20 ms, so the window spans 15 periods and 5 time constants. The window was never the constraint.
**What the true mismatch is**: on Y there is no DC component (`-4.4e-07` and `-6.8e-08 m`, zero to the resampling floor), because a mass-spring-damper returning to rest transfers no net momentum and exerts no net force at DC; the absorber is a RESONANT mismatch at 150 Hz, consistent with the oracle detuning sweep having its minimum exactly there. The only genuine DC mismatch is on X, `+8.2e-05 m`, from `mh = mhr + ma` shifting the centre of mass when the absorber is removed. Even that is not a ramp: `ramp_fraction` is 5.7 %, and the error tracks the motion profile envelope and decays back toward zero, so it is excitation-driven, not integrating.
**Open loop against closed loop, measured**: the controller removes the DC part completely on every record (mean from `+8.2e-05 m` to `~1e-12 m`, ramp `0.00 %`, forced by the `z = 1` pole) and only attenuates the oscillatory part below about 50 Hz. Closed/open rms ratio is `1.56e-03 / 1.69e-03 / 8.20e-02` on V2 (content near 10 Hz, `sigma_max(So) = 0.021`) but `13.9 / 12.6 / 1.66` on V1, i.e. the loop makes it an order of magnitude WORSE, because V1 sits at 130-180 Hz where `sigma_max(So) = 2.07`. This is the waterbed, not a defect. **For the augmentation it is favourable: the residual is larger in the closed-loop data in exactly the band where the absorber lives.**
**Ruled out**: (1) **That the 8-state `fs.hfn` path differs from the 6-state `Gantry_State_Block`** (the leading suspect in the handoff): they agree to `1.5e-11 m`, so the two ANN states and the `ann_route_ix=(3,4,5,7)` routing change nothing when the ANN output is zero. (2) **`K0 = 17` mid-record start as the cause of the ANN-off dataset offset**: `gen_annoff_data.py` unpacks `K0` and never uses it, and `dm.load_T` returns the full record, so that script starts at sample 0. `K0` was a separate artefact in `baseline_drift_replay.py`, since fixed, worth `7.84e-04 m` on V1 Y via inherited absorber momentum `0.101 * vda(t0)`. (3) **Absorber-momentum seeding** (`overlay_records.py:44-52`): predicts exactly zero from rest and its documented maximum is `4e-03 m`, 25x below the reported figure. (4) **Numerics**: pipeline against `deriv6` float64 is `3.9e-08 m` closed loop, and float32 against float64 in the controller is `5.2e-08 m` bounded over 48000 samples.
**Live bug fixed in the same change**: `scripts/gantry/dc-accumulation/gen_annoff_data.py` set `x0 = np.zeros((1, nx))` and corrected only `[:3]` via an identified affine map, leaving the velocities at normalised zero. Its self-check could not catch this because it verifies the first OUTPUT, which depends on positions only and passes for any velocity. `x0[0, 3:6]` is now set to physical rest, `(-x_mean[3:] / std_x[3:])`, and the implied physical velocity is printed. **Any `.mat` written by that script before 2026-08-15 carries a non-physical `1.3e-04 m` component and must be regenerated before use as a training target.**
**Constrains**: no replay starts at `K0` or any mid-record sample; use sample 0 with `closed_loop.x0_for`. No script writes zeros into a normalised state vector to mean "at rest"; build physical, then normalise. `ramp_fraction` is never quoted as evidence about an offset, since it is variance-based and scores `0.00 %` on a pure DC shift. No general conclusion is drawn from V1 alone: it is the record where the real effect is smallest and every artefact dominates. The `2.2e-04 / 2.4e-04 / 4.6e-05 m` V2 figure is the reference the augmentation is judged against, and the drift framing in `docs/writeup/` and the drift-demo figures rests on the falsified premise and needs revisiting.

### [D-138] The Coulomb data generator uses Garcia's hard `sign`, not the `tanh` of D-116, and the model copy is forced to fixed-step `ode4`
**Date**: 2026-07-31
**What**: New folder `Matlab-scripts/Augmentation-coulomb/`, mirroring `Augmentation-kxy/` file for file, adding dry friction to the 8-state truth used for trajectory-data generation. Friction is `u_eff = u - P*(cc .* sign(P.'*qdot))` built in STAGE coordinates and projected to logical with the pipeline's own `P`. Values are Garcia's identified `cc1 = 16.8 N`, `cc2 = 18.35 N`, `ccy = 11.6 N`, used as given: no sweep, no tuning, no threshold-hunting. Data goes to `data/gantry/matlab/trajectory/augmentation_coulomb/` with `fig_dir` derived from it. The purpose is to settle whether the settled position offset documented in `scripts/gantry/msd-offset/` is a property of the machine or an artefact of modelling the stages as frictionless.
**Why hard `sign` and not the `tanh` of D-116**: D-116 chose `tanh(v/v0)` for `coulomb_lfr.py` because that code is differentiated through and BPTT needs `dF/dv` at `v = 0`; note that `dF/dcc = sign(v)` is nonzero either way, so the smoothing buys the STATE sensitivity, not the parameter gradient. That justification is absent here, because a data generator is never differentiated through. Three further reasons, in increasing order of weight. (1) Garcia 2013 Fig. 2 writes `cc1*sign(dX1)`, `cc2*sign(dX2)`, `ccy*sign(dY)` and nothing else; `tanh` is our numerical addition and has no support in the source. (2) For `|v| << v0`, `tanh(v/v0) -> v/v0`, so smoothed Coulomb is not friction at all in that regime but viscous damping of coefficient `cc/v0`: at `v0 = 1e-3` that is `3.5e4 Ns/m` on the X row against a real `cg1+cg2` of `34.8`, a factor of 1000. Neither X nor Y has stiffness (`K11 = K33 = 0`), so under a residual force `F` the stage creeps at `F*v0/cc` with nothing to stop it: about `1.4e-4 m` over a 5 s hold at `F ~ 1 N`, against the `4.63e-5 m` X offset being measured. The regularization error would be several times the signal. Keeping the creep under the `1e-7 m` floor needs `v0 < ~3e-7 m/s`, three decades below the default, at which point the ODE is as stiff as `sign` and the smoothing has bought nothing. (3) The asymmetry that decides it: `sign` is exactly right everywhere the stage slides and wrong only on the measure-zero set `v = 0`, whereas `tanh` is wrong on an interval whose width we chose, and the settled offset lives entirely inside that interval.
**Why fixed-step `ode4`**: the original `gantry_additional_state_2025a` runs variable-step `ode45` at `RelTol = 1e-4` (measured, not assumed). `sign` is discontinuous and the Stateflow chart declares no zero-crossing signal, so `ode45` would either crawl at the crossings or step through them with uncontrolled error, and the output sample grid would depend on where the crossings landed. Fixed-step `ode4` at `h = 5e-5 s` makes the step a knob we control (which is what makes the step-halving diagnostic meaningful), keeps the output grid uniform so `gtd_run_simulation`'s resample interpolation never fires, and matches the RK4 used on the Python side. Verified to run with the Simscape subsystem present.
**Known artefacts of bare `sign`, and the diagnostic that bounds them**: with no stick state the force flips `+cc`/`-cc` at a zero crossing, giving chatter of order `(cc_row/m)*h^2` in position (`1.6e-9 m` on X, `2.9e-9 m` on Y at `h = 5e-5`, both below the `1e-7 m` floor) and, more dangerously, ratcheting: the two half-cycles decelerate at `(cc-F)/m` and `(cc+F)/m`, so each cycle nets a displacement toward `F` and accumulates over the hold. Ratcheting looks exactly like "the offset decayed" and would be read as physics. Both artefacts scale with `h` and a physical offset does not, so `check_step_halving.m` runs the record at `h` and `h/2` and compares the settled value, with a `cc = 0` pair as the control that separates ordinary RK4 truncation error from the discontinuity specifically. PASS means bare `sign` is adequate at this step; FAIL means adding a Karnopp stick state (hold when `|v| ~ 0` and `|F_applied| <= cc`), which is Coulomb's law at `v = 0` rather than the `sign(0) = 0` placeholder, and is therefore not a deviation from Garcia.
**Pre-registered, before the run (D-090)**: Garcia identifies `cc` by displacing each axis at CONSTANT VELOCITY, i.e. a pure sliding experiment, so the values are slip forces and the paper measures no breakaway. Prediction: friction is a matched force present in both truth and baseline, so it largely cancels in their difference while sliding and bites only once motion stops; the standstill record should therefore change more than a sweeping one. Second prediction, from the paper's own parameters rather than from ours: `cc1 /= cc2`, so the Theta row carries `Lb/2*(cc1-cc2) = -0.562 Nm` under common-mode X motion, and Theta being the only sprung axis (`kb1+kb2 = 3975 Nm/rad`) it deflects about `1.4e-4 rad` with a sign that flips with the direction of travel. The `msd-offset` brief's assumption that Theta shows no offset therefore stops holding once friction is on, though it should largely cancel in the truth-minus-baseline difference.
**Ruled out**: (1) **`tanh` smoothing**, per the three reasons above; D-116 is unchanged and still governs `coulomb_lfr.py`, which is a different thread (Telica real-data recovery) with a real differentiability requirement. (2) **Sweeping `cc` over decades to find a collapse threshold**, as `scripts/gantry/coulomb-offset/SESSION-PROMPT.md` proposed. The parameters are identified, not free, and fitting a physical constant to the outcome it is supposed to explain is not a measurement. A sensitivity check around the identified values remains available if the answer looks knife-edge. (3) **Telica datasheet friction (X 2x43 N, Y 49 N)**, which belongs to the real machine; the synthetic gantry is a Garcia machine and must use Garcia's numbers. (4) **Enabling the Simscape Coulomb blocks**, which `Matlab-scripts/coulomb-friction/PLAN.md` already established are orphaned and not wired into the loaded model; irrelevant anyway, since the record-generating truth is the "Extended ODE" chart (`gantrySystemExtended.m`), not the Simscape subsystem. (5) **Closed-loop replay**, which the offset investigation does not do: the records are generated closed loop, the replay through the baseline is open loop.
**Python side: the baseline is the LFR block, and friction must enter at BOTH `u` sites** (added after building it, 2026-07-31). The baseline for the replay is `Gantry_State_Block` (`model_augmentation/fit_systems/blocks.py:642`), the LFR rational form that `gantry_interconnect_dynamic.py` actually trains against, NOT the collapsed 3-DOF `deriv6` in `scripts/gantry/msd-offset/plant.py`. Those are different realizations of the same physics, and an offset measured against the collapsed form would be an offset in a lookalike. Friction is added by SUBCLASSING (`scripts/gantry/coulomb-offset/plant_coulomb.py`), so Jan's framework is not modified. The load-bearing detail: `u_log` enters `deriv()` at two sites, the net force (`blocks.py:799-801`) and the direct `Bu*u_log` term through `G` (`blocks.py:829-830`). At `Y = 0` the LFR latent `w` vanishes, so the direct term is the ONLY route to acceleration and an `fnet`-only insertion would make friction silently disappear there. `u_eff` therefore replaces `u_log` at both, `G` is untouched (it is built from `N0, d0, M1, M2, K, C` and `u` never enters it), and friction is built from the PHYSICAL velocity after denormalisation, never from the normalised state. Note the input-convention difference from the MATLAB EOM: the block's `u` is in STAGE forces with `u_log = P @ u`, while `gantrySystemExtendedCoulomb.m` takes `u` already logical; the friction term is identical in both. The law is a switch defaulting to `sign` so the replay matches the truth exactly, with `tanh` retained for a future training path where D-116's differentiability argument becomes live again. Gates measured: `cc = 0` bit-identical on the frozen-`Y = 0.10`, frozen-`Y = 0`, and LPV branches (`0.000e+00` each); `cc = Garcia` changes `xdot` (`3.5` to `3.7`); friction never does positive work; the wrong-frame trap is detectable (`1.525e+01`); the `fnet`-only negative control diverges at `Y = 0` by `3.661 m/s^2` on both branches, confirming the second site is load-bearing; and `sign` versus `tanh` agree exactly at `1 m/s` while differing by `34.8 N`, essentially the full X-row Coulomb force, at `1e-5 m/s`, which is the quantitative form of the argument against `tanh` for a settled-offset measurement.
**AMENDED 2026-08-01: hard `sign` is REPLACED by the Karnopp stick state. The reasoning above was right on the evidence it had and is kept for that reason; what it could not see is below.** The MATLAB step-halving gate passed, so the truth was self-consistent at generation time. That says nothing about whether the trajectory can be REPRODUCED, and it cannot: `sign(0) = 0` makes the vector field discontinuous, so crossing `v = 0` flips the force by `2*cc ~ 35 N`. Measured (`scripts/gantry/coulomb-offset/diag_sign_floor.py`), perturbing `dX` by `1e-12 m/s` and integrating 3 s: perturbation gain `X 1.07e+06`, `Theta 2.38e+06`, `Y 4.59e+06`, against `1.43` for the frictionless system. That million-fold amplification of round-off put a `~1e-6 m` floor under every open-loop replay measurement, which was LARGER than the offsets being measured. Step refinement confirmed the diagnosis and ruled out a fix on our side: at the matching step the Python replay reproduced the record to `1e-09 m`, and halving the step made it 677x WORSE, because the MATLAB record is not a sample of the exact solution either (fixed-step `ode4`, hard `sign`, no zero-crossing detection) and we only matched it by making the identical error. **The replacement introduces no new physical parameter.** Classical Coulomb friction has one coefficient per contact and is already set-valued at rest (`|F| <= cc`); `sign(0) = 0` picked a value outside the admissible set. Formally the hard-`sign` equation is a differential inclusion whose solution on the switching surface is defined in the Filippov sense, and the Filippov solution IS the stick solution, so Karnopp is the correct solution of the law Garcia wrote down rather than a departure from it. Verified end to end: Python-only perturbation gain falls to `X 1.47`, `Theta 1.44e-03`, `Y 1.53e-03`, matching the frictionless `1.43`, and moving only 4% when `V_EPS` is swept over two decades (`diag_karnopp.py`); the regenerated record replays open loop at `X 1.44e-09 m`, `Theta 6.86e-11 rad`, `Y 8.75e-09 m`, i.e. 1.5 to 2.7x the frictionless floor instead of 1000x (`verify_karnopp_floor.py`). Data goes to `augmentation_coulomb_karnopp/`; the hard-`sign` `augmentation_coulomb/` stays on disk so the two laws can be compared and nothing quoted against it shifts underneath. **Every Coulomb number produced before this amendment rests on the superseded dataset and does not carry over**, including the stick fractions, the `123x` Y collapse and the X trend result. **The one genuine limitation to state in the write-up**: Karnopp has infinitely stiff stick, so it omits presliding displacement (elastic microslip before breakaway, `1e-06` to `1e-05 m` on real bearings, the same length scale as the offsets) and it has no stiction above kinetic. Both need parameters Garcia never measured, since every friction experiment in that paper is constant-velocity sliding. **Two implementation traps, both found the hard way**: on breakaway the slip force must use `cc*sign(F_required)`, NOT `cc*sign(v)`, because a stuck rail has `|v| < V_EPS` so that sign is arbitrary and `sign(0) = 0` removes the friction at the exact moment it should saturate; and a gate asking only "does it break away" passes on that bug, because zero friction accelerates the stage just as well as saturated friction does, which is why gate A7 now compares against the frictionless acceleration (`0.667 = 2/3` expected for `3*cc` applied against a saturated `cc`).
**Constrains**: `Matlab-scripts/Augmentation/`, `model_augmentation/` and `scripts/gantry/msd-offset/` stay untouched, so every existing number remains valid. `kamtin-fp-model/` is copied from, never written. The two gates are load-bearing and must both pass before any offset number from this dataset is quoted: `check_coulomb_noop` (`cc = 0` reproduces `gantrySystemExtended` bit-identically) and `check_coulomb_reaches_plant` (`cc` actually reach the integrator, `cc = 0` bit-identical THROUGH the model against the original run at the same fixed step). The Python baseline used for the replay must carry the SAME friction law with the same `P` projection, or the replay difference measures a friction-law mismatch instead of the model mismatch it is meant to isolate. This dataset is NOT interchangeable with the frictionless one for anything but the friction question, because `Cfb` was designed on a frictionless plant and can hunt or sit in a deadband with Coulomb in the loop. The stick-fraction print is part of the deliverable, not decoration: if a standstill record is stuck for most of its length the record carries little information, and "the offset went away" would then be a statement about an uninformative dataset rather than about physics.

### [D-137] The rate sweep is dropped: sampling rate is information-neutral, the horizon is the binding constraint, and the standalone ANN runs once at 800 Hz with nf=3700
**Date**: 2026-07-31
**What**: Cancel the four-arm decimation sweep planned in D-136. Run the standalone full ANN (no baseline, no interconnect, no projection) at a single configuration: `fs = 800 Hz`, `nf = 3700`, `na = nb = 17`, `nx = 8`, batch 256, `lr = 1e-3` Adam, train `T10_aprbs_60`, validate `V2_aprbs_Ylow`. Diagnostics in `scripts/gantry/ann-blackbox/pretrain_diagnostic.py`, results in `results/pretrain_diagnostic.json`.
**Why the sweep is dropped**: it was designed to test an information hypothesis that two measurements falsify. (1) **Achievable ceiling is rate-insensitive**: a free run of the EXACT discretised truth over the validation record gives `4.76e-04 m` at 4 kHz and `5.06e-04 m` at 800 Hz, so decimating 5x costs 6%. (2) **State-to-window conditioning depends only on window DURATION, not rate**: 0.5 s of data gives `1.52e5` whether that is `nf=2000` at 4 kHz or `nf=400` at 800 Hz, agreeing to three digits. Rate therefore does not change what is learnable. What rate does change is the loss landscape: the trivial-predictor basin measured in `oversampling_diagnostic.py` sits at `0.162` (normalised, 400-step CVEL) at 4 kHz and `1.574` at 800 Hz. **Decimation is a loss-landscape fix and a way to afford the horizon, not an information fix.** Stating it the old way would have been wrong.
**Why nf=3700**: Beintema, Schoukens, Toth, *Automatica* 156:111210 (2023), DOI `10.1016/j.automatica.2023.111210`, Section 3.4: "Choose T to be a few times the largest characteristic time scale for stable data-generating systems." Measured from the truth: `tau_max = 1.5459 s`, and the damped output transient reaches 1% at 6.73 s (identical in seconds at every rate, as physics requires). At 3x`tau_max` the rule gives `nf = 18551` at 4 kHz and `nf = 3711` at 800 Hz. **`nf = 400` was 46x short of the rule at 4 kHz.** 800 Hz is what makes the rule affordable: `185 x 400 = 74k` step-batches per epoch before, `23 x 3711 = 85k` after, i.e. the correctly sized run costs about what the wrong one cost.
**Scope caveat, stated because the rule is being used outside its own conditions**: the same paper's Condition 1 requires incremental exponential output stability with `lambda < 1`. Our plant has two poles exactly at `z = 1`, so Condition 1 FAILS and the consistency theorems do not cover this system. `tau_max` is therefore the largest FINITE time constant, over the damped subspace only, with the integrator directions projected out. This is a judgement call and must be reported as one.
**Target revised**: a free run with PERFECT dynamics and a near-perfect initial state reaches only `4.76e-04 m`, three times worse than the `1.6e-4 m` bar. That number is the ceiling for a frozen-`Y` LINEAR model, and the gap exists because `Y` swings +-0.14 m on `V2` while `M` was frozen at its mean. So `1.6e-4 m` provably **cannot** be reached by a linear black box: beating it requires the ANN to capture the `Y`-dependence of the inertia. Gates become: beat own epoch-0 with best checkpoint off epoch 0; then `4.8e-4 m` (linear-equivalent); then `1.6e-4 m` (requires learned `Y`-dependence).
**Established, and NOT what was expected**: (1) **float32 is not a limitation.** The exact truth run in float32 against itself in float64 over the full 12 s record differs by `5.93e-06 m` at 4 kHz, 27x BELOW the target. An earlier claim in this project that free-run accumulation over 48000 steps would exceed float32 resolution is **withdrawn**; it assumed coherent error accumulation, and the damped modes bound the walk. (2) **The encoder is not the bottleneck.** Least-squares recovery of the exact 8-state from 17 past samples has relative residual `1.7e-3` at 4 kHz, worst single state `6.9e-3` (the two absorber states). `na = nb = 17` sits inside the empirically good band `4 < n <= 20` reported in the same paper's Section 5.6, whose own finding is that the theoretical minimum `n = nx - 1` is NOT optimal. Note this test is one-sided: it proves an encoder CAN recover the state, not that failure would prove it cannot, since a black box may use any state coordinates.
**Ruled out**: (1) The four-arm sweep, per the falsification above. (2) The 400 Hz arm, already excluded in D-136 (Nyquist 200 Hz against a 180 Hz band top). (3) Fisher-information analysis on physical parameters, since the standalone black box has none; that belongs to the baseline recovery thread. (4) False-nearest-neighbours / mutual-information embedding selection, which is built for unknown systems from a scalar series and is strictly weaker than the direct observability and recoverability tests available when the truth is known. (5) TBPTT gradient-bias truncation sweeps, which require gradients and so are not pre-training diagnostics.
**Constrains**: any future horizon choice in this project is set in SECONDS against `tau_max = 1.5459 s`, not in samples. Any claim that the black box "cannot learn this" must now be measured against the `4.76e-04 m` linear ceiling rather than against `1.6e-4 m`.

### [D-136] The standalone black box is rebuilt as a minimal transcription of Jan's 37-line reference, and the decimation sweep floors at 800 Hz
**Date**: 2026-07-31
**What**: New folder `scripts/gantry/ann-blackbox/` holding `ann_blackbox.py` (one file, no imports from `scripts/gantry/full-blackbox/`) plus `CORRESPONDENCE.md`, a line-by-line map onto `scripts/ecc_2025/msd_ndof_deepSI_encoder.py`. Every departure from Jan's 37 lines carries a `DEV` comment on that line naming what forced it. Structural parameters follow Jan's formulas at `dof = 4` (the 8-state truth is X + Theta + Y + absorber): `nx = dof*2 = 8`, `na = nb = dof*4+1 = 17`, f/h nets `2x8`, encoder `2x16`, `auto_fit_norm=True`, `validation_measure="sim-RMS"`. Train `T10_aprbs_60`, validate `V2_aprbs_Ylow`. The planned decimation sweep runs **4000 / 2000 / 1000 / 800 Hz**, not the 400 Hz named in the session brief.
**Why the 800 Hz floor**: the excitation is band-limited to 130-180 Hz by the multisine (`gtd_config.m:106-107`) and the absorber mode sits at 158.114 Hz. At `fs = 400` Hz, Nyquist is 200 Hz: no anti-alias filter passes 180 Hz and stops by 200 Hz, so that arm deletes the signal the model must learn and would fail for a reason unrelated to the hypothesis under test. At 800 Hz the band top sits at 0.45 Nyquist, which is clean.
**Why the premise of the sweep survives anyway**: measured PSD of `u_total` (not inferred) on `T10_aprbs_60` gives per-channel energy fractions of 0.905 / 0.915 / 0.778 in **1-20 Hz** against 0.079 / 0.070 / 0.184 in 130-180 Hz; `V2_aprbs_Ylow` is 0.913 / 0.899 / 0.813 against 0.055 / 0.071 / 0.176. The data is closed-loop, so the APRBS reference dominates `u_total` at low frequency even though the multisine is narrowband. The 5.1 Hz sprung-Theta content therefore IS excited on the APRBS records, and the "400 samples at 4 kHz is half a period of the dominant content" argument holds. On the standstill records it does not: `V1_standstill_Yp10` is 0.997 / 1.000 / 0.997 inside 130-180 Hz with nothing below 20 Hz.
**Why `V2_aprbs_Ylow` and not `V1_standstill_Yp10` as validation**: `V1` has `y` std `3e-6 m` against `T10`'s `5.7e-2 m`, four orders down. A model trained on `T10` cannot be scored on it.
**Established while building**: `y` in the records is **stage frame** `[x1, x2, Y]`, all three channels in metres (`gtd_save_record.m:28`, `S.y = q` with `S.Y_trajectory = q(:,3)`). So pooled `sim-RMS` mixes no units and is comparable to the `1.6e-4 m` bar, but yaw lives in the `x1 - x2` difference of two channels three orders larger and is invisible to it; the script writes per-channel RMS alongside.
**Blocking defect found and fixed**: `plant.load_record` returns Fortran-ordered arrays (MATLAB `.mat` layout). deepSI's `default_encoder_net.forward` calls `.view` (`encoders.py:122`), which raises on the resulting non-C-contiguous tensor, so `fit()` dies on its **first validation** before a single gradient step. Fixed with `np.ascontiguousarray` in the loader wrapper. Any script driving deepSI from these records hits this.
**Ruled out**: (1) switching to the broadband `joint` track ([1,200] Hz, full record set exists on disk) — it would give the black box an easier problem than the augmentation arm faces, making the comparison unfair in the black box's favour. (2) Chunked `fit()` calls for incremental persistence — `fit` reloads the `_best` checkpoint on return (`fit_system.py:486`), so chunking rewinds the weights each chunk and changes the training trajectory; `timeout=` is used instead, which makes `fit` return normally before a wall clock kills it. (3) Jan's `batch_size = 2000` — at `nf = 400` it gives 19 updates per epoch on a 48 k record.
**Constrains**: measured cost at 4 kHz is 185 updates/epoch at ~1.75 it/s, i.e. ~105 s/epoch, so 500 epochs is ~14.6 h for that arm alone. Equal-compute comparison across arms requires fixing the number of gradient updates (`n_its`), not `epochs`, since `N_batch_updates_per_epoch` scales with the decimation factor.

### [D-135] The next augmentation experiment swaps the hidden MSD absorber for a truth-only cubic spring on Y
**Date**: 2026-07-30
**What**: Regenerate the truth data with the hidden absorber ABSENT (`USE_MSD = false`, the existing baseline path, so the truth is the 6-state gantry) and a cubic spring to ground added on the logical Y coordinate of the TRUTH ONLY, `F_spring = -k3 * Y^3` with `k3 > 0`. The baseline model is unchanged and stays unsprung, so the spring is the ONLY discrepancy and it is purely STATIC. **The supervisor's 2026-07-30 notes name three things, but two of them are one idea, leaving TWO SEPARATE EXPERIMENTS.** Jan's EJC paper uses `tanh` as the activation of the learning components (`hoekstra2025_lfr-augmentation-ejc.pdf` p5: tanh for all models except the linear dynamic parallel one, which uses identity), so "relu op een tanh opgegooid in jan zijn paper" is not a plant nonlinearity taken from that paper but a proposal to build a relu-on-tanh element into the plant; and relu(tanh(.)) is a bounded kinked limiter, which is why he wrote "**of** saturation" in the same breath. **Experiment 1: the cubic hardening spring** on X and Y, this entry. Smooth, growing with amplitude, physical, close to in-class (`k3*Y^2` is a Y-scheduled stiffness the LPV-LFR structure could nearly express), and it is the Silverbox nonlinearity so published SUBNET numbers exist. **Experiment 2: the saturation / relu-on-tanh limiter** on the input, its own D-entry when specified. Kinked, bounded, so the mismatch force and hence the free-run degradation are bounded too; input-side (Hammerstein) rather than state-side; not physical and out-of-class. Its one prerequisite is that saturation is identifiable only if the recorded `u` actually reaches the clip level (Bai, *Automatica* 38(5), 2002), which must be checked against the recorded `u_total` distribution first. **The two are run separately, each with its own dataset, runs and result.** They are not rungs of a single campaign and not alternatives to choose between; experiment 1 goes first only because it has no prerequisite check. The contrast is itself informative: one adds force as motion grows and the other removes it, and one is nearly expressible by the physical model class while the other is not at all. Separately, his notes name three ANALYSES that apply to each experiment: does the network actually learn it, how it interacts in closed loop, and a Bode reading obtained by replacing the nonlinear component with its maximum and linearising around the extreme operating points, since the plant is nonlinear and a single FRF does not exist. Build precedent is T4's `Matlab-scripts/Augmentation-kxy/`; the truth EOM to COPY (never modify, it is under the read-only tree) is `kamtin-fp-model/03 Simulink gantry/functions/gantrySystem.m`, and the model to copy is `kamtin-fp-model/03 Simulink gantry/gantry_2025a.slx`.
**CORRECTION HISTORY, both on 2026-07-30, recorded because the second reverses the first.** (i) The entry as first drafted specified `ma_frac = 0` on the 8-state extended ODE. That specific route is indeed impossible: row 4 of `M` in `gantrySystemExtended.m` is `[0, -ma*d, ma, ma]`, identically zero at `ma = 0`, so `M \ ...` fails. (ii) The first correction then concluded that removing the absorber was infeasible altogether, on the ground that the no-MSD path is a Simscape Multibody plant with no editable m-file ODE. **That was wrong, and it was inferred from a code comment rather than verified.** Inspection of `gantry_2025a.slx` establishes: the model runs THREE plants in parallel and logs them separately, a Simscape Multibody `Single H-gantry` subsystem to `q`, a MATLAB Function chart `gantrySystemMFile` calling `gantrySystem` to `q1`, and `gantrySystemCoriolisCentripetalMFile` to `q2`; block connectivity in `system_root.xml` traces `q1` back through Gain4 and Selector1 and the integrator to the `MATLAB Function1` subsystem, whose chart (`stateflow/chart_42.xml`) is the plain `gantrySystem` wrapper. Since `gtd_run_simulation.m` reads `q1` on the non-MSD branch, **the baseline truth already IS the m-file `gantrySystem.m`**, a 6-state ODE with a nonsingular 3x3 mass matrix and the same `dxdt = A*x + B*u` shape as the extended one. The absorber therefore does not need to be zeroed out; the no-MSD path simply has no absorber, and it is an existing, exercised code path. Rung A is restored to the purely static form.
**Why the static-only form is worth the correction**: at `k3 = 0` the baseline model IS the truth exactly, so epoch 0 must win and the ANN must learn nothing, which is a free null control that the absorber-retained version cannot offer. At `k3 > 0` there is exactly one thing to learn, static and visible inside the training window. That is the cleanest isolating form available, and it is the one that mirrors Retzler's Case 2 against Case 1.
**Why**: Under the current target the ANN has nothing to win. The absorber contributes about `1e-9` of the squared sim-RMS metric while the untrained FP baseline already scores `1.6e-4 m`, so the objective can only do damage; every completed run in the drift-isolation programme selected epoch 0 (`scripts/gantry/drift-isolation/CONCLUSIONS.md`, cross-cutting finding 1). A cubic spring on Y deflects the rigid-body trajectory itself, at low frequency, on the large-motion records, which is 100% of the metric. It is therefore the first target for which epoch 0 is beatable at all. The discrepancy is also STATIC: no hidden states (`nx_ann = 0`), fully visible inside a 400-sample BPTT window, and exactly representable by the additive static ANN block. Published precedent for the contrast this creates: Retzler, Toth, Schoukens, Beintema et al., "Learning-based augmentation of physics-based models: an industrial robot use case", *Data-Centric Engineering* 5:e12, 2024, DOI `10.1017/dce.2024.8` (gold OA), where augmentation beats ANN-only when the discrepancy is a static sliding-friction map (150-step NRMS 0.0434 against 0.0786) and FAILS to beat it on data containing hysteretic presliding friction, i.e. a hidden-state discrepancy. The cubic hardening spring is also the field's canonical nonlinearity: the Silverbox benchmark is an electronic Duffing oscillator, "a 2nd order LTI system with a 3rd degree polynomial static nonlinearity around it in feedback" (Wigren & Schoukens, ECC 2013, DOI `10.23919/ECC.2013.6669201`), and our own model class has published numbers on it (SUBNET encoder 0.32 mV test RMS excluding amplitude extrapolation, Beintema, Toth, Schoukens, L4DC 2021, PMLR v144).
**Design choices, pinned**: (1) **Absolute (ground-referenced) Y in the logical frame, not a relative coordinate.** A relative-coordinate spring needs a second body, which does not exist once the absorber is removed. Note the consequence, and it is the reason this is a diagnostic and not a fix: the local stiffness `3*k3*Y^2` is exactly ZERO at `Y = 0`, so the standstill poles remain at `z = 1` and the marginal-pole mechanism of D-134 and T4 is NOT removed, only softened at large amplitude. (2) **Y axis only.** Single-axis keeps attribution clean. The ANN still routes to X and Y per D-103, so the X rows must learn to output zero; whether they do is a free measurement of negation/absorption on an axis with no discrepancy, and it is informative rather than a confound. (3) **Truth-only.** This is the one difference from T4, which put a LINEAR spring in both truth and model as a pole diagnostic; T4's negative result therefore does not predict this one, because they answer different questions. (4) **`k3` is a hyperparameter with no default and needs a `# HEURISTIC:` label**, since no literature sets it for this rig.
**Campaign folder**: `scripts/gantry/discrepancy-ladder/`, with the full design in its `PLAN.md` (rung structure, model arms, gates, sizing derivation, open decisions, inherited traps, literature anchors). This entry records the decision; that file records how it is built.
**Implementation route**: a NEW folder `Matlab-scripts/Augmentation-cubic/`, mirroring `Matlab-scripts/Augmentation-kxy/` file for file. **Nothing under `Matlab-scripts/Augmentation/` is modified** (user instruction, 2026-07-30; same constraint T4 held to). `kamtin-fp-model/` is READ ONLY and is only ever COPIED FROM, exactly as `make_kxy_model.m` does it (`load_system`, `save_system` to the new path, `close_system(src, 0)` to discard). Contents: `gantrySystemCubic.m` (copy of `gantrySystem.m` plus the spring term), `check_cubic_noop.m` (Class A gate: `k3 = 0` reproduces the original derivative bit-identically, `k3 > 0` changes it), `check_cubic_reaches_plant.m` (frozen-controller gate through Simulink), `make_cubic_model.m` (scripted `.slx` copy of `gantry_2025a`), `generate_trajectory_data_cubic.m` (production generator writing to `data/gantry/matlab/trajectory/augmentation_cubic/`), and a `README.md`. **The ODE edit is NOT a stiffness-matrix entry.** T4 could put `k_xy` straight into `K4` because a linear spring is linear in the state, but the file computes `dxdt = A*x + B*u` with `A = [0 I; -M\K, -M\C]`, and a cubic term cannot be represented there. It enters as an added nonlinear generalised force: `f_nl = [0; 0; -k3*Y^3]` and `dxdt = A*x + B*u + [zeros(3,1); pinv(M)*f_nl]`, using `pinv` to match how the file already builds `B = [zeros(3); pinv(M)]`. One added line plus one modified line, but structural rather than a matrix entry, so the Class A no-op gate is load-bearing rather than a formality. Three properties of the T4 machinery carry over and are why this is cheap: `make_kxy_model.m` builds the model copy entirely through the Stateflow API with NO GUI edit, so `k3` is added as one more chart PARAMETER resolving from the base workspace and the diagram is structurally untouched (`make_cubic_model.m` must assert the resulting data inventory exactly as `make_kxy_model.m` does, rather than assume the count); the controller is frozen for free, because `cfg.K` cannot express a cubic term at all and so `gtd_build_plant` produces the identical `Cfb`, `G`, reference and limit scaling; and the chart edit targets the same `*MFile` wrapper shape. Inherit T4's two traps verbatim: override `cfg.fig_dir` as well as `cfg.out_dir` (`gtd_config` bakes `fig_dir` from `out_dir` at config time, and missing this overwrote baseline figures once), and keep the per-record force-peak print, which is the diagnostic that would have caught T4's 300 N sizing error on record 1.
**Four consequences of taking the non-MSD branch, each needing a decision or a check before launch**: (1) `gtd_config` switches the multisine band to **1 to 7 Hz** when `USE_MSD` is false, against 130 to 180 Hz for the augmentation track. For a `Y^3` discrepancy that is an improvement, since the cubic term needs amplitude and low frequency delivers it, but it is a deliberate choice and must be stated rather than inherited. (2) `gtd_run_simulation` reads `q1` rather than `q_aug`, skips the second without-multisine run, and does not swap `mh` to `mh_rigid`; all three are existing behaviour on an exercised path. (3) The saved records carry no meaningful `delta_a` / `vdelta_a`, so the Python loader's expectations must be checked before the first training run. (4) The augmentation model must be configured with `nx_ann = 0` for a static discrepancy, which needs confirming as a supported setting in `gantry_dynamic/`.
**Sizing rule: `k3` is set by the FREE-RUN DEGRADATION it induces, not by excitation preservation.** The first draft of this entry proposed T4's P3 force-fraction rule and landed on 30 to 110 N/m^3. **That is wrong by one to two orders of magnitude**, and the reason is structural rather than arithmetic: T4 put its spring in BOTH truth and model, so no mismatch force ever existed and excitation was the only thing at risk. Here the spring is truth-only, so the ENTIRE spring force is a mismatch force acting on a `K = 0` integrating axis, and the metric's sensitivity to it dominates by three orders of magnitude. The correct lever, derived from the plant: on the Y axis `gantrySystem.m` has `M(3,3) = mh = 10.1 kg`, `C(3,3) = cy = 10 Ns/m`, `K(3,3) = 0`, so a force the model lacks obeys `mh*ddy + cy*dy = dF` in the open-loop free run. Terminal velocity is `dF/cy` and the time constant is `mh/cy = 1.01 s` (T5 measured about 1 s on the real model, which is what makes this trustworthy rather than notional). Position error therefore ramps LINEARLY at `dF/cy` after about a second, reaching about `1.1*dF` metres by 12 s, and the RMS of a ramp is its endpoint over `sqrt(3)`, giving a **mismatch-force-to-sim-RMS gain of about 0.64 m per newton**. Worked sizing: to put the smallest-`|Y|` validation record (V1 sits at `|Y| = 0.1 m`) at 10x the `1.66e-4 m` untrained baseline, i.e. `1.6e-3 m`, needs `dF = 2.5e-3 N`, hence `k3 = dF/|Y|^3 = 2.5 N/m^3`. At that `k3` a record parked at `|Y| = 0.30 m` sees `dF = 0.068 N` and about `4.3e-2 m` of free-run error, 260x the baseline, still well inside both the `0.4 m` Y limit and the training Y range. **Candidate range 1 to 5 N/m^3**, fixed by a `derive_k3.py` script before any MATLAB batch. Two consequences: T4's P3 excitation criterion is satisfied automatically by roughly a factor of 400 (`0.068 N` against a `30 N` multisine) and is not binding; and because the degradation goes as `|Y_op|^3` while the metric gain is linear, **the sizing must be tabulated PER VALIDATION RECORD against its own Y operating point**, since a val set concentrated at small `|Y|` would leave no headroom no matter what `k3` is. The `0.64 m/N` gain is derived from `mh` and `cy`; the choice of "10x the baseline on the weakest val record" is a `# HEURISTIC:` design target and must be labelled as one.
**Pre-registered predictions (D-090), stated before launch**: (i) **`k3` is identified by Y OPERATING-POINT coverage across records, not by motion amplitude within a record, and this is a correction to the first draft of this entry.** The spring force is set by the parked Y offset, not by the motion about it: at `k3 = 2.5 N/m^3` a record parked at `|Y| = 0.30 m` carries a constant `6.8e-2 N` whether or not it moves, one at `|Y| = 0.20 m` carries `2.0e-2 N`, and one at `Y = 0` carries exactly zero. The standstill records T1 to T5 are therefore NOT information-free under this discrepancy, unlike under the absorber, because they sample the static curve at distinct Y offsets; the earlier draft wrongly used the per-record `y std` (`4.4e-6 m`) in place of the Y offset and concluded they were dead. What remains open and must be MEASURED rather than assumed is whether the global-`ystd` normaliser then actually delivers gradient to them, since its 0.00% share for T1 to T5 (`objective_rescale_diag.py`) was measured on a model that had no DC error to make. Consequence for the design: the record set's Y coverage is the identifiability condition, so it must be tabulated before launch. (ii) Amplitude coverage must be designed, not inherited: Beintema et al. report the encoder's Silverbox error "increases significantly" under amplitude extrapolation, and a `y^3` discrepancy is maximally amplitude-sensitive, so train/val/test `|Y|` ranges must overlap and be stated. (iii) The failure mode, if it appears, moves rather than vanishes: the ANN should fit the static nonlinearity inside the window and beat epoch 0 on the windowed loss, and the open question becomes whether a residual DC force bias on the `K = 0` axes still ramps the 12 s free run, which the marginal poles at `Y = 0` permit unchanged.
**Acceptance criterion**: PRIMARY, any epoch beats epoch 0 on val sim-RMS by more than 2x, which breaks the epoch-0 selection that every completed run in the programme has shown. STRETCH, the trained model recovers more than half of the degradation the spring caused (`epoch0_sprung` back toward `1.6e-4 m`). Both are computed from the run's own records with no oracle quantity involved, per the standing thresholds rule.
**Verification before the run is trusted** (2026-07-18 lesson `verify-knob-moves-the-target-before-running`, which T4 executed and which caught a 300 N sizing error): zero records bit-identical to the unsprung set; force change correlating with `mean|Y|^3` and matching `k3 * |Y|^3` at three amplitudes; force change exactly zero on the records at `mean|Y| = 0`; and the augmented run's training loss compared against an unsprung control, since a large train-loss shift means the spring perturbed the excitation and the run is confounded (T4 cleared this at 0.09% and 0.3%).
**What this does NOT resolve, stated so it is not over-claimed**: the plant still integrates and the metric is still a 48,000 step free run, so the accumulation mechanism survives (Ribeiro, Tiels, Umenberger, Schon, Aguirre, *Automatica* 121:109158, 2020, `arXiv:1905.00820`, Thm 1: at Lipschitz constant 1 the loss Lipschitz constant is `O(N)` and the gradient's is `O(N^3)`, with the Appendix B.2 inequality showing windowed parameter sensitivity growing only linearly in window length; Asadi, Misra, Littman, ICML 2018, `arXiv:1804.07193`, Thm 1, tight for deterministic linear transitions, giving `N * Delta` accumulation at spectral radius 1). It does not touch the normaliser defect, the closed-loop baseline decision left open by T6, or D-134's standalone black-box negative, which holds for any data generated on this plant because the cubic spring adds no local stiffness at `Y = 0`.
**Ruled out**: (1) **Input saturation first.** Also static and with a strong benchmark (the Wiener-Hammerstein benchmark contains a saturation nonlinearity and our model class holds the best published RMS on it, 0.241 mV, Beintema, Schoukens, Toth, *Automatica* 156:111210, 2023), but it is input-side, excites only when `u` actually reaches the saturation level (Bai, *Automatica* 38(5), 2002, DOI `10.1016/S0005-1098(01)00281-3`: identifiability requires driving into saturation), and its most interesting behaviour is closed-loop, which is blocked on T6's unresolved baseline decision. Deferred to rung B. (2) **Spring in both truth and model.** That is T4, a plant modification and a pole diagnostic, not a learning target. (3) **Retaining the absorber alongside the spring.** Briefly adopted on 2026-07-30 and reversed the same day once `gantry_2025a.slx` was actually inspected (see the correction history). It would have made the discrepancy static plus hidden-dynamic simultaneously and forfeited the `k3 = 0` null control, for no saving in implementation effort. Kept on the roadmap only as an optional later rung if the static case succeeds and a static-plus-dynamic ladder is wanted. (4) **A linear spring as the target.** It is absorbable into the baseline's own `K` matrix, so it tests parameter drift rather than the ANN's nonlinear capacity, and it would collide with the orthogonal-projection thread.
**Constrains**: rung A must complete and be understood before the rung B saturation variant is submitted (`dont-advance-past-a-failing-isolating-test`). `Matlab-scripts/Augmentation/` stays untouched and `kamtin-fp-model/` is copied from but never written, and the new dataset goes to its own `augmentation_cubic` directory, so the existing datasets and every result resting on them remain valid. The `k3 = 0` arm must be generated and checked as the null control before the `k3 > 0` batch, since it is the one configuration where the baseline model equals the truth exactly. `k3` gets a `# HEURISTIC:` label with the P3 derivation beside it, never a `# THEORY:` label, since no source sets it. The regenerated dataset is a NEW mode alongside `augmentation` and `augmentation_kxy` and inherits D-131's guard pattern: the run must refuse to start unless the mode flag and the `k3` flag are both set or neither is, because a sprung model on unsprung data still trains and still prints a plausible number that answers no question. Whatever rung A returns is a thesis result: success localises the failure to hidden-state discrepancies and gives the first positive augmentation number, failure with a visible, static, in-class target is a far stronger negative than anything currently on record.

### [D-134] The standalone black box is closed as a NEGATIVE RESULT, not carried forward as a bug to fix
**Date**: 2026-07-30
**What**: Stop trying to make `blackbox_standalone.py` reach the FP baseline's 1.6e-4 m val sim-RMS. The campaign's question ("can a full ANN learn this system at all") is answered: not with a short-window training objective and a 12 s free-run metric, for a structural reason. Write it up as a thesis result and, if a black-box comparison column is still wanted, produce it with the CT SUBNET (`SS_encoder_deriv_general`, `integrator_euler`) reported honestly rather than tuned toward 1.6e-4. Full evidence chain: `docs/blackbox-standalone-audit-2026-07-30.md` Part 3. Scripts: `ref_subnet_v2_example.py` (tiers 1-2), `msd_stability_contrast.py` (tier 3), `objective_rescale_diag.py`, `objective_train_diag.py`.
**Why**: X and Y are K=0 free masses, so the true discrete state matrix has eigenvalues exactly 1 and all 8 sit at or just inside the unit circle. Measured `|eig(df/dx)|`: 0.655 max at initialisation with 0 of 48 above 0.99; after 130k updates (run 73940) 1.0003 max with 12 of 48 above 0.99, i.e. two eigenvalues reached 1 and overshot, which is the mechanism behind that run's 3.87e14 excursions, and the rest stayed at 0.78 and below. Holding position for 12 s to 0.1% requires `|lambda-1| < 2e-8`; over the 400-step training window `lambda=1` and `lambda=0.9999` differ by 4%, far below the loss's other error terms. The objective is close to blind to the quantity that sets the metric. SUBNET's short-window design is sound for systems that FORGET (errors decay, so short-horizon accuracy implies long-horizon accuracy) and that implication fails for a system that INTEGRATES. Controlled proof on Jan's own MSD generator, one variable changed (`k[0]`, `c[0]`, the only terms acting on an absolute coordinate): true spectral radius 0.996232 / 0.999503 / 1.000000 gives trained free-run NRMS 0.144 / 0.349 / 3.795 and learned max `|eig|` 1.0000 / 1.0034 / 0.8558. Only the arm whose truth is exactly 1 fails to reach the unit circle, and it ends worse than holding the last sample constant.
**Why this is not an implementation fault**: the same code path fits Beintema's own v2 example plant to NRMS 0.024 (SISO) and a 3x3 coupled version to 0.090, and the training path and simulation path agree bit-exactly (0.000e+00 over 200 steps, 3 channels) on real gantry data. The full audit of `blackbox_standalone.py` against Jan's script and deepSI v2 found no defect that changes the model class or the numerics.
**Ruled out**: (1) More updates or a larger `nf`: 73940 had 130k updates and its best free run was still worse than holding the last sample; 74045 at nf=800 never reached the trivial baseline on its own objective and rose from iteration 11700. Closing a 120x horizon gap by 2x cannot matter, and it pays the O(N^3) within-segment smoothness cost (D-127 note, Ribeiro et al. 2020). (2) Learning rate: all six arms of both lr screens degrade V1 within one epoch while train loss falls, so the objective and the metric move in opposite directions from the start; the blow-ups are an eigenvalue above 1, not a step size. (3) Re-scaling the loss normaliser: real (five of fourteen records receive 0.00% of the gradient under the current global `ystd`) but second order. A paired 900-update comparison left `current` at 0.0743, `per_window` at 0.0644 and `per_record` at 0.1338, none near the 0.047 hold-last-sample floor. (4) Logical-frame reweighting for Theta: Theta is ~1e-9 of the metric and free-run NRMS is ~1 on EVERY mode of every record, so Theta is not the binding constraint. (5) Identity-initialising `default_state_net`: this was proposed and withdrawn. Neither Jan's script nor deepSI v2's discrete `SUBNET` does it (v2's `MLP_res_net` linear branch is randomly initialised), so it is a deviation from both references, and the framework already ships the principled version as a first-class model class.
**Constrains**: The black-box baseline column in the thesis results table is a NEGATIVE result reported with its mechanism, not a number to be improved. Any future black-box run uses the CT SUBNET; `tau` is a genuine hyperparameter with no default (small `tau` keeps the net's O(1) outputs representing the true `dx/dt`, large `tau` tightens the init spectrum; `tau=2.5e-2` is the first value where all 48 measured eigenvalues clear 0.99) and needs a `# THEORY:` label tying it to the 130-180 Hz band. The CT SUBNET fixes conditioning but NOT stability: `max|lambda| > 1` at every `tau` tested, so epoch-0 free runs may be non-finite and the G1/G2 gates, which anchor to epoch-0, will have nothing to anchor to. G0 (data-derived hold-last-sample floor, added to `blackbox_standalone.py` 2026-07-30) is unaffected. This decision does NOT touch the augmented pipeline; if anything it is the quantitative argument for it.

### [D-133] Session handoff gets a protocol document, and the handoff it produces is written as an Opus 5 prompt
**Date**: 2026-07-30
**What**: New file `docs/session-handoff-protocol.md`: the instructions Claude reads when the current session must hand its work to a fresh session (context pressure, degraded performance, or the user asking for a handoff). It contains the trigger conditions, the rules for writing the handoff as a prompt for Claude Opus 5, a fill-in template with fourteen required sections, an anti-pattern list, and one acceptance criterion (the cold-read test). Generated handoffs are written to `tasks/handoffs/YYYY-MM-DD-<slug>.md`, one file per handoff, and are exempt from the "No new files unless asked" standing rule because the protocol is the standing authorisation. `CLAUDE.md` gains a Key File Map row, a Workflow bullet, and a pointer inside the **Respect session boundaries** standing rule. `tasks/handoff.md` is NOT reused and NOT read.
**Why**: The user asked for it on 2026-07-30, and specified that the document must give instructions for writing an optimal Opus 5 prompt. That constraint is what makes it a protocol rather than a template: the handoff's reader is a model whose documented failure modes (`https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-5`, fetched 2026-07-30) are scope expansion, over-verification when told to verify, eager delegation, and literal obedience to hedging instructions such as "be conservative". A handoff that reads as a diary triggers all four; a handoff that reads as a complete task specification with an explicit anti-scope triggers none, which is the same page's positive finding that the model "performs best when given the complete task specification up front and left to run". Two project-specific sections are carried into the template beyond what generic prompt guidance would suggest: **What was tried and failed, with the mechanism**, because this project's cost of a repeated dead end is a multi-hour training run, and a **verified vs assumed** split, because the Control Engineering Stance forbids oracle-based thresholds and the code-quote rule forbids unverified line-number claims, so a handoff that blurs the two launders an assumption into the next session as a fact. A separate output directory rather than `tasks/handoff.md` because that file means something else (open blockers plus cross-agent proposals) and carries an archive-on-read rule that would restructure it as a side effect of every handoff.
**Ruled out**: (a) A skill at `.claude/skills/handoff/SKILL.md` giving a `/handoff` command and description-based auto-triggering. Genuinely attractive, and the better answer if invocation friction proves to be the failure mode, but rejected for now on two grounds: skill triggering depends on a description matching the user's phrasing, whereas `CLAUDE.md` is auto-loaded into every session, so a pointer there fires more reliably at exactly the moment context is scarce; and `.claude/skills/` is not in Claude's ownership list under Multi-Agent Ownership, while `docs/` is. A ten-line skill wrapper pointing at this document remains available later at no cost. (b) Appending to `tasks/handoff.md`: see above. (c) Overwriting one `tasks/session-handoff.md` each time: loses the trail, and two handoffs on the same problem are the evidence that a thread is looping. (d) Including a "verify your handoff is complete" step: exactly the over-verification the Opus 5 page tells you to delete. The cold-read test is stated as an acceptance criterion instead, which the document explains explicitly so the distinction is learnable rather than arbitrary.
**Correction applied 2026-07-30, same day**: the first version made the protocol self-triggering (context pressure, degraded performance, and a launching long run were listed as triggers alongside the user's request, and `CLAUDE.md`'s Workflow bullet repeated them). The user's decision, given directly: the handoff is written ONLY on explicit request, and Claude never decides by itself to hand off, end a session, or start a new one. Fixed in three places: the protocol's trigger section now states the single trigger and instructs that context pressure or looping be reported in one sentence rather than acted on; the `CLAUDE.md` Workflow bullet is marked user-triggered only with the same prohibition; and the pointer added to the **Respect session boundaries** standing rule was reverted to that rule's original wording, since "work happens in another session" is not the same request as "write me a handoff" and the pointer blurred them. The protocol now says so explicitly and tells Claude to ask in one line if the two are unclear.
**Constrains**: The protocol is now the single place where handoff format lives; a handoff that omits a required section is out of compliance with it. Anything generated under `tasks/handoffs/` is a prompt, not an archive: the length ceiling is roughly 400 lines, and detail belongs behind a `file:line` pointer rather than pasted in. Because the successor session auto-loads `CLAUDE.md`, a handoff must not restate it; duplicating standing rules into a handoff is a documented anti-pattern in the file. Any future change to how Anthropic recommends prompting the current model should be applied to this document, since it hardcodes 2026-07-30 guidance for Opus 5 (effort tiers, thinking-on default, delegation caps) and would otherwise silently age.

### [D-132] `CLAUDE.md` restructured against Anthropic's Opus 5 prompting guidance
**Date**: 2026-07-30
**What**: Eight changes to `CLAUDE.md`, no change to any other file. (1) Every em-dash removed from the file itself, including markdown table separators (`|---|` becomes `|-|`), so the file no longer demonstrates the punctuation its own rule forbids. (2) New standing rule **No new files unless asked**, with document length calibration. (3) New standing rule **Concise by default**. (4) The code-quote procedure now writes its quote file to the session scratchpad instead of the hardcoded `/tmp/quote.txt`. (5) The two conflicting definitions of "don't touch" (Hard Constraints vs Standing Rules) consolidated into one standing rule covering both the per-file and the whole-codebase form. (6) Subagent ceiling added to Workflow: one Explore agent by default, never a subagent to verify own work. (7) Section order changed to Project Identity, Standing Rules, Hard Constraints, Control Engineering Stance, then reference material; the code-quote procedure moved out of the top slot. (8) **Answer before code** softened to permit read-only lookups needed for an accurate answer, while still forbidding edits, runs and new files before direction is confirmed.
**Why**: The user asked for a review of `CLAUDE.md` against `https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-5` (fetched 2026-07-30), then approved all eight items. Three of the changes counter behaviours that page documents as specific to this model: default responses and written deliverables are longer than on Opus 4.8 (items 2 and 3), and the model expands scope and delegates more readily than prior models (item 6). Item 1 is the highest-leverage change and the reason is stated on that page directly: positive examples of the wanted style outperform instructions about what not to do, so 9 lines of em-dashes inside the file that bans them were working against the ban. Item 4 is a platform bug rather than a style choice: the environment is win32 with a session scratchpad, and `/tmp/quote.txt` resolves only by accident through Git Bash. Items 5 and 7 remove ambiguity and put the always-active constraints where they are read rather than two thirds down the file. All 19 file pointers in the Key File Map were checked to still resolve before the rewrite; none were stale, so none were touched.
**Ruled out**: (a) Deleting the **Verification** rule ("never mark a task complete without proving it works"), which the Opus 5 page explicitly recommends removing as a cause of over-verification. Rejected because in this project proving it works means running the simulation and reading the BFR, which is the deliverable itself, not a redundant self-check. The page's target is instructions like "double-check your answer before responding", and no such instruction exists in the file. (b) Purging em-dashes from `CODEX.md` in the same pass: Multi-Agent Ownership gives Claude read-and-propose rights only on that file, so it is left to Codex and flagged in text instead. (c) Editing `tasks/handoff.md` to record the `CODEX.md` proposal: reading that file triggers the archival rule, which would restructure it as an unrequested side effect. (d) Deleting **Answer before code** outright, per the standing rule that weakening a user-authored rule needs justification; its intent (stopping premature implementation before redirection is possible) still holds, so it was narrowed rather than dropped. Noted at the time: item 8 loosens a rule that constrains Claude's own behaviour, which is a conflict of interest, and the user was told so before approving.
**Constrains**: New markdown tables added to `CLAUDE.md` must use single-hyphen separators or item 1 regresses silently; a `grep "—" CLAUDE.md` returning anything means the file is out of compliance. Any future addition to Standing Rules belongs in that one section, not duplicated into Hard Constraints, which is now reserved for the read-only and logging constraints. Because **No new files unless asked** is now active, work that previously produced a fresh `docs/*.md` per investigation must append to `docs/decisions.md`, `docs/gantry-augmentation-problem-log.md` or `tasks/todo.md` instead, or ask first. The 22 untracked `docs/*.md` files on the `Augmentation` branch predate this rule and are not retroactively affected.

### [D-131] T4's X/Y spring enters the model as a subclass and a class swap, not as an edit to `gantry_ss.py`
**Date**: 2026-07-29
**What**: The Python half of T4 (`scripts/gantry/drift-isolation/t4_xy_stiffness/`). `KXY_Gantry_State_Block` subclasses `Gantry_State_Block` and rebuilds the two quantities that carry K (`K_mat`, and `A_combined` whose `Ax` block is `-M0inv @ K`) from one modified K via the parent's own `build_G_matrix_entries`; `linearize_kxy` does the same for the encoder's reconstructability map. `kxy_physics(k_xy)` installs them, for the duration of a run, at the four sites that construct the K=0 plant in their own namespace: `gantry_dynamic.model.Gantry_State_Block`, `gantry_dynamic.model.gantry_linearize_and_discretize`, `gantry_dynamic.baselines.Gantry_State_Block`, `common.rollout.Gantry_State_Block`. `--k_xy` (default 0.0) and `--mode augmentation_kxy` are added to the drift-isolation CLI, and `run_training.py` refuses to start unless both are set or neither is. `model_augmentation/systems/gantry_ss.py` is untouched. Gated by `check_t4_noop.py`, 12/12 passing.
**Why**: The user's decision, given directly on 2026-07-29, over the alternative recorded in `HANDOFF.md` section 8a. `gantry_ss.py` is shared by every run in the project, so a T4 knob there sits in the path of T1, T2, T3 and production even behind a `K_XY = 0.0` default; the subclass keeps the whole intervention inside the test folder, matching what T3 already does with `blocks_t3.py` and the standing constraint that nothing outside `drift-isolation/` and `Matlab-scripts/Augmentation-kxy/` is modified. A class swap rather than a copy of `build_model` because T4 changes no structure at all (same routing, loss, encoder dimensions, optimiser), so copying 150 lines to change nothing in them would only create a second copy that can drift. Four sites and not one because each module imported the name into its own namespace; the `baselines` one is the easy one to miss and the worst to miss, since T4's truth is sprung and an unsprung FP baseline would flatter the augmentation on every rung for a reason unrelated to the augmentation.
**Ruled out**: (a) Editing `gantry_ss.py` with a `# CHANGED:` marker and a `K_XY = 0.0` default: an exact no-op numerically, but it puts the knob in the shared path, which is what the user declined. (b) The existing `GANTRY_KX_ART`/`GANTRY_KY_ART` env knob (`gantry_ss.py:102-113`), which writes the same two entries at import time. **This is a closer call than it first looks and the initial write-up oversold the case against it.** It needs no new code, and because every construction site derives its K from `gantry_ss.K`, one env var covers all four sites plus the encoder map with no enumeration at all, which is strictly more robust than (c) against a fifth site being added later. Two objections that were raised do NOT hold: "it is implicit and would affect other scripts in the process" is weak when a Slurm job runs one script, and "it cannot be proven inert by construction" is simply wrong, since the `if _kx_art:` branch is skipped when unset. The objection that DOES hold is timing: the knob is read at IMPORT of `gantry_ss`, which `run_training.py` has already triggered through `gantry_dynamic.data` before argv is parsed, so it can never be driven by a CLI flag and must be exported in the shell (the file's own comment says as much). That costs three things a diagnostic cannot afford: the plant would not be recorded in the run's own `config.json`/`results.json`; the data/model pairing guard could not be enforced in one place, since one half would live in the environment and the other in argv; and being per-process global it cannot give a sprung model and an unsprung reference in the same process, which T4's analysis plausibly wants. It is also labelled a HEURISTIC knob belonging to the gantry-zero-mean pole-perturbation study, so sharing it would couple two experiments to one control. `blocks_t4.py` therefore refuses to run while either variable is set, because two mechanisms writing the same two matrix entries is a trap. (c) Copying `build_model` the way `build_t3.py` does: justified there by three structural changes, unjustified here by zero. (d) Patching `gantry_ss.K` in memory for the duration of the build: one mechanism covering all four sites and the encoder map at once, but it mutates a shared module global, which is the same implicitness objection as (b).
**Correction applied 2026-07-29, same day**: the first version of the patch bound k_xy by creating a subclass inside a function. Pickle stores classes by name, and deepSI checkpoints with `torch.save(self.__dict__, file)` on every validation improvement (`fit_system.py:496`), so that class could not be looked up on reload and T4 would have died at its first checkpoint hours into the job (`PicklingError: attribute lookup KXY_Gantry_State_Block_k1000 on blocks_t4 failed`, measured). Fixed by binding with `functools.partial` over the module-level class, so instances pickle by name exactly as T3's `blocks_t3` instances do. Gate C4 now saves and reloads a fully built sprung model and checks `K_mat` survives.
**Constrains**: T4 must run at `orth_beta = 0` and `joint_estimation = False`, and `kxy_physics` raises otherwise: both paths construct `Parameterized_Gantry_State_Block`, which builds its own `K(theta)` internally and therefore does NOT pick up the spring. At `orth_observe = True` with `beta = 0` (the reference configuration) that block is a meter only, so training is unaffected, but its `orth-frac` must not be read as a sprung-plant quantity. The matched control for T4 is T1 rung 0 (`t1_nf400`: nf=400, stride 100, 20 epochs, seed 42), so `run_t4.sh` uses exactly those settings and changing them destroys the only comparison T4 has. T4 remains blocked on the MATLAB dataset (`Matlab-scripts/Augmentation-kxy/generate_trajectory_data_kxy.m`, k_xy = 10 N/m, output `data/gantry/matlab/trajectory/augmentation_kxy/`), which the CLI's `--mode augmentation_kxy` resolves with no further project change.

### [D-130] Encoder `W^a` dead zone is structural; run 71167 is a horizon result, not a failed run
**Date**: 2026-07-29
**What**: Two findings recorded, both measured rather than inferred, with the consequences they impose on how existing results are read. (1) **`W^a` dead zone.** The encoder rows reconstructing the augmented states (`Wa_psi_y`, `Wa_psi_u`, `pre_encoder.py:404-405`) receive exactly zero gradient at initialisation (measured `0.000000e+00` via `compute_gradient_norms`, against `1.15e-4` and `2.25e-3` for the `W^b` rows), and moved 250x less than every other trainable weight across run 71167's 20 epochs (max|Δ| 2.0e-6 vs 4e-4 to 6e-4). Full write-up and mechanism in `docs/gantry-augmentation-problem-log.md` §13. (2) **Run 71167 re-read.** Its best checkpoint is epoch 0, so `gantry_ckpt_71167.pt` contains the *untrained* weights and any diff against a fresh init measures nothing; the epoch-20 state survives only in `gantry_drift_71167_last.pth`. From that file: the ANN did train (final layer 0 → ‖W‖=1.94e-3), train and val nf-RMS both improved at the 400-step horizon (3.81e-5 → 3.33e-5 and 4.39e-5 → 3.77e-5 m), while val sim-RMS over 12 s degraded 127x on the first epoch. The run is therefore a clean horizon-gap measurement plus a negation-tendency baseline (`orth-frac` 0.226 → 0.468, monotone, still climbing at beta=0), not a failed training run.
**Why**: The dead zone follows from the wiring, not from a hyperparameter: the augmented rows reach the loss through one path only (they are the ANN's input; `h_base` and `f_base` are both wired with `selection_matrix(PHY_IX, nxd)`, `model.py:123` and `:127`), and the ANN's final layer is zero-initialised (`torch_nets.py:113-114`), so its input-Jacobian is exactly zero. It is D-065/D-066 in reverse: there a near-zero `C_aug` on the output side choked the ANN's gradient, here a zero ANN on the input side chokes the encoder's. The 71167 re-read matters because train and val nf-RMS improve *together* at the trained horizon, which rules out overfitting and generalisation failure as explanations and isolates the 120x train/select horizon ratio (400 samples optimised, 48000 selected) as the sole cause. The damage arithmetic closes: 2.1e-2 m over 48000 steps is 4.4e-7 m/step on the K=0 position rows, and the same per-step term inside a 400-step window contributes 1.8e-4 m, discounted exactly 120x by the objective.
**Ruled out**: (a) "The encoder is inert because its gradient is weak" — refuted: `W^b` and the ANN moved at 0.84 to 1.19x the Adam ceiling (5200 steps x lr), i.e. as far as the learning rate permits with nearly every step aligned. What is small is the movement *relative to weight scale* (‖Wb_psi_y‖=1413, so Δ is 1e-6 relative), which is a learning-rate-to-scale mismatch, not gradient starvation. (b) "Nothing activated in 71167" — refuted by the `_last` state; only the *saved best* checkpoint is inert. (c) Fixing the dead zone by raising the encoder learning rate alone: the gate is the ANN leaving zero, not the encoder's step size. (d) Fixing it by initialising the ANN final layer small-nonzero: opens the path immediately but breaks "the model starts exactly at the baseline", which D-072/D-089 rely on for the same-init baseline comparison.
**Constrains**: Any `R2_linmap` or aug-state quality number reported while the ANN is still near zero describes a random projection and must not be read as a learned estimate; this applies retroactively to 71167 and to the earlier 5-epoch runs. Diffs against `gantry_ckpt_*.pt` are only meaningful once it is confirmed which epoch won; otherwise use the `_last` artefact. The next augmentation run must address the horizon, not the learning rate: multiple shooting (`n_seg > 1`, `defect_weight > 0`, D-127) is the implemented mechanism that lengthens the objective without lengthening the gradient path, and it is also the only term in the codebase that gives the encoder a short-path gradient (`multiple_shooting.py:132-135`). Per D-090 that run needs its row written before launch. **Open dependency**: the Ribeiro et al. (Automatica 2020) theorems cited in `multiple_shooting.py`'s docstring, which are the argument for segments over a longer `nf`, remain unverified; the paper is not in `literature/`, and it must be checked before the claim reaches the thesis.

### [D-129] Training objective gets its own figure; the block scheme stays a structure figure
**Date**: 2026-07-28
**What**: Two artefacts. (1) `docs/writeup/figure-style.md` codifies the house figure style already implicit in `jan-blockscheme-v4.tex` (standalone TikZ, `>=Latex`, the `block`/`rblock`/`small`/`sum`/`jn` node vocabulary, `\footnotesize` base with `\scriptsize` annotations, black plus `black!55` and `black!12` only, no in-figure legend, 85 mm legibility floor, scale-honesty rule). (2) `docs/writeup/training-objective-v1.tex` is a NEW figure covering input, output, loss and validation on a time axis, in three panels: (a) one 12 s validation record with the selection horizon and the training horizon drawn to true scale, (b) the interior of one training window (encoder window, open-loop rollout, residual sticks, stride), (c) the state-to-output map showing which state rows the loss can reach. `jan-blockscheme-v4.tex` is NOT extended to cover this.
**Why**: The supervisor asked for input, output, loss function and validation. v4 answers input and output structurally but has no time axis, so `nf`, the encoder window, the stride and the two different validation horizons cannot be drawn on it without destroying its readability. Those four quantities are properties of the data flow through time, which is a different object from the block topology. The specific number the figure has to carry is the horizon ratio: records are 12 s at 4 kHz (48000 samples, `data.py` + the 12 s convention used throughout the drift docs) while `nf = nf_seconds/ts = 0.100*4000 = 400` samples (`config.py:127-136`), so the training horizon is exactly 1/120 of the `sim-RMS` selection horizon (`training.py:211-223`). Drawn to true scale, that ratio is the figure's whole argument, and it sets up the multiple-shooting work (D-127) directly: the same panel redrawn at `n_seg > 1` shows the objective spanning more of the selection bar while the gradient path stays one segment. Panel (c) exists because the velocity states are the one thing a reader cannot infer: they are states, they are scaled from finite-differenced position (`data.py:205-207`), they seed the encoder init normalisation (D-119), and they are never compared to anything because the loss is output error on the three positions only (`gantry_ss.py:135-138`, `Cd = [P^T | 0]`).
**Ruled out**: (a) Adding the training/validation content to v4 (no time axis; overloads a figure that is currently readable). (b) One panel with a broken or log time axis (a broken axis is exactly the device that hides the ratio the figure exists to show). (c) Drawing the training bar oversized "so it can be seen" (discards the argument; the scale-honesty rule in `figure-style.md` section 5 forbids it, and the mitigation is an external label with a leader). (d) Real data traces in panel (a): at `mode='augmentation'` the excitation is narrowband 130 to 180 Hz, so a 12 s trace renders as a solid band and carries no information at that scale. Panel (a) therefore shows the axis and the horizon bars only, and panel (b) traces are declared schematic in the caption.
**Constrains**: New figures follow `docs/writeup/figure-style.md`; deviations are logged, not improvised. v4 remains the structure figure and is superseded only for the two additions noted separately (normalisation boundary, split encoder-output annotation), which belong on a v5 and are NOT part of this decision. Any later claim about the horizon gap in prose must use the same numbers as panel (a) (400 samples, 48000 samples, 1/120) so the figure and the text cannot drift apart.

**AMENDMENT 2026-07-28 (normalisation gets its own figure; the no-legend rule is withdrawn):** two changes after review. (1) The normalisation content is now `docs/writeup/coordinates-normalisation-v1.tex`, a third figure, rather than the dashed boundary on a v5 block scheme that this entry originally anticipated. Reason: the question "how do we get the velocities" and the question "how is the model normalised" have the same answer, because the finite-differenced velocity exists nowhere in the training path except inside `x_all` (`data.py:200-211`), which is also what normalises the encoder-init matrices (`model.py:170-178`, D-119). A boundary annotation on v4 can show that normalised and SI coordinates exist; it cannot show where the constants come from or that one shared `std_x` feeds all three consumers. The figure carries three callouts: the FD velocity fixing the encoder's state frame; `x_bar` having no data-derived scale at all (`x_mean`/`std_x` are 6-vectors, `W^a` is Kaiming random, and `pre_encoder.py:462-465` applies the offset to the physical rows only, so `delta_a` is dimensionless and only comparable after a best affine map); and every constant being estimated on training records and applied unchanged to val/test. It also fixes a notation clash the first draft introduced: normalised matrices are `A_n, B_n, C_n, D_n`, not `A_bar`, because v4 already uses the bar for the augmented partition. (2) `figure-style.md` section 7 no longer forbids in-figure legends. That rule was over-generalised from a single comment in `jan-blockscheme-v4.tex`, where dropping the legend was correct because every object is labelled in place. The rule now reads "prefer direct labelling, use a legend when direct labelling is not possible" and lists the cases where a legend is the right tool (overlapping or dense series, an encoding repeated across panels, more than about four series, figures read out of context). **Constrains additionally**: the recommendation to replace the legend block in `training-objective-v1` panel (b) still stands, but on the merits of that panel (two curves that visibly separate at the right edge, so inline labels are placeable), not by appeal to a blanket rule.

**AMENDMENT 2026-07-28b (notation pass applied; the legend recommendation was wrong; validation is not covered):** three things. (1) The notation pass on `training-objective-v1` is applied: `g_aug` added to the chain label (the previous label named only `f_base + f_aug` and silently dropped the half of the router output that drives `x_bar`), `x_k_hat` at the encoder handoff becomes `x_{k|k}` to match the filtered-not-predicted conditioning that `na_right = nb_right = 1` produces, panel (c) now carries BOTH partitions (an overbrace for the three rows `h_base` reads, underbraces for v4's `x_tilde`/`x_bar`), `psi` is declared trained jointly, `C` is marked as the SI matrix against the coordinates figure's `C_n`, and the signal units and channel names are stated (`u = [F_X1, F_X2, F_Y]` [N], `y` [m], stage frame), which the figure previously never did. (2) **The loss expression was missing the `|k` conditioning** and now reads `y_{k+t} - yhat_{k+t|k}`; without it the objective reads as a plain simulation error and the defining property of the method, that every window re-estimates its own initial state, is invisible. The batch average stays implicit and is stated in the annotation. (3) **The previous amendment's legend recommendation is refuted.** Measuring the drawn traces, the residual envelope is about 0.098 cm, so the two curves are at most ~1 mm apart and oscillate at 157 Hz; there is nowhere on the curve to place an inline label. That is exactly the "series too dense to label inline" case in the rewritten `figure-style.md` section 7, so the legend is RETAINED. What was actually wrong was the word "ticks", which named the ink instead of the quantity; the residual is now direct-labelled via one leader to one tick, because a tick has a location even when a curve does not. **Constrains additionally**: `training-objective-v1` covers input, output and loss, and covers checkpoint SELECTION only. It does not cover the validation strategy (held-out test records E1 to E4, held-out Y positions, the encoder-init and true-x0 baselines, the FP+MSD oracle, NRMS). Panel (a)'s bar is labelled "checkpoint selection" and must not be relabelled "validation". Covering validation properly needs a fourth figure, which is NOT built.

### [D-128] Model-correctness verification on no-MSD data: step refinement is the verdict, duration sweep is the symptom
**Date**: 2026-07-27
**What**: `scripts/gantry/verification/verify_model_duration_sweep.py` verifies the Python plant model against the MATLAB records in `data/gantry/matlab/trajectory/augmentation/baseline/`, where `USE_MSD=false` makes the 6-state `Gantry_State_Block` structurally identical to the data-generating plant, so every residual is numerical rather than structural. Six conventions are fixed. (1) The object under test is `Gantry_State_Block` itself, driven exactly as `baselines.py::compute_baseline_fp_nrms` drives it, not a re-derived 3-DOF EOM. (2) `x0` is the TRUE initial state: positions from `x_logical[0]`, velocities set to EXACTLY zero because the records start from standstill. **Corrected 2026-07-27** (see the amendment below): this originally read "the stored true `x_logical[0]`", which was wrong. (3) Normalisation is identity (physical units), so `Cd_norm = Cd` and `y0 = 0`. (4) Two independent axes are swept: duration `T` in {0.5, 1, 2, 4, 6, 8, 12} s at fixed step, and `up_sample` in {1, 2, 4, 8} at fixed `T = 12` s. (5) Both `float64` and `float32` are run at every `up_sample`. (6) Errors are reported in stage coords (X1, X2, Y, the channels the loss sees) **and** in logical coords (X, Theta, Y) via `P^-T`, because K=0 is a statement about the logical axes and is unreadable in the stage mixture.
**Why**: A duration sweep alone cannot separate integration error from a structural or force-scale mismatch: both grow with `T`, and on the K=0 rows (logical X and Y are pure double integrators, `gantry_ss.py` `Cd = [P^T | 0]`) a constant force error and an accumulating truncation error both land near a `T^2` envelope. Step refinement decides it, because only truncation responds to `h`: RK4 is globally 4th order, so each doubling of `up_sample` must drop the error by roughly 16x, and a plateau under refinement is by definition not truncation. This is the same instrument as `parameter-diagnostics/rk4_substep_sweep.json` (cited in D-126 to prove the shipped-data mismatch was solver-and-rate, not substep), applied here to the model instead of the data. The `float32` arm is not decoration: eps is 1.2e-7, and against stage outputs of order 1e-2 to 3e-1 m the reported "e-7 on one state" sits within a few ulp, so the precision floor is a live explanation that must be excluded before any structural claim is made. Choices (2) and (3) each remove a known confound of the size being measured: `analytical_x0` injects an O(dt) velocity error at sample 0 which the K=0 rows then integrate (the D-087 artifact), and the training normalisation is a conditioning choice whose float32 rounding would otherwise be charged to the model.
**Ruled out**: (a) Duration slope as the sole verdict (cannot separate truncation from structure, per above). (b) The 8-state `gantry_dynamic/oracle.py` simulator (that is the MSD plant; setting `MA=0` makes its 4x4 mass matrix singular rather than reducing it to the baseline). (c) Hand-deriving the 3-DOF EOM to compare against (would verify a fresh derivation, not the model the pipeline integrates, and `kamtin-fp-model/` is read-only). (d) `V1_standstill_Yp10` as the primary record (Y is constant, `M(Y)` is frozen, so the LPV scheduling path is never exercised and cannot fail); it is retained as the clean-integration control. (e) The pipeline normalisation (see above). (f) Running only at `d=1` or only at the pipeline rate: the gap between them is the D-087 block-mean resampling error, which is a separate error source and is only visible if both are run.
**Constrains**: A "the model is correct" claim from this script requires the refinement arm to show approximately 4th-order decay down to the float64 floor, not merely a small number at one duration. Any residual that survives refinement in float64 is structural and blocks use of the baseline track until explained. The script reads only `augmentation/baseline/`; the MSD track cannot be verified this way because there `model != system` by construction, which is the whole point of the augmentation.

**AMENDMENT 2026-07-27 (first run refuted convention 2, and convention 6's second arm):** the first run measured a step- and precision-independent residual of RMS X `3.314e-05` / Y `1.326e-04` m on `T6_ysweep_slow` and I initially read it as a possible structural model defect. It was neither structural nor a model property: it was the seeding. `gtd_save_record.m:19-22` builds the stored velocities with MATLAB `gradient`, which is **one-sided at sample 0**, so on a ramp from rest it returns `(q1-q0)/ts = a*ts/2` where the true `v(0)` is zero. The data confirms the ramp: stored `qdot[1]` is exactly 2x stored `qdot[0]`, the signature of `q = a*t^2/2`. The unsprung X and Y axes integrate that spurious velocity into a **permanent** position offset `tau*v0`, `tau = m/c`. Predicted X `1.546 s * 2.372e-05 = 3.667e-05` m and Y `1.010 s * 1.447e-04 = 1.461e-04` m, against measured max|e| X `3.687e-05` and Y `1.419e-04` m: agreement to 0.5% and 3%. That accounts for the entire residual and explains all three of its properties (step-independent, precision-independent, saturating rather than growing, because the velocity error decays over `tau` and freezes the offset in). Convention 2 is therefore **inverted**: `x_logical[0]` is itself a finite difference, so calling it "the true x0, never the finite difference `analytical_x0`" was self-contradictory. The true initial state is analytic here: recorded positions with velocities identically zero, which is exactly what `validate_lfr.py:216-217` uses and why the param-recovery path reaches e-7 while this one did not. `--x0 stored` is retained solely to reproduce the artifact. Convention 6's `pysynth` arm is separately void as a verification: `drift_common.build_baseline_block` constructs the SAME `Gantry_State_Block` under test, so that arm is circular and can only serve as a harness self-check. **Constrains additionally**: no run of this script may seed from a differentiated state and call it ground truth; D-087 and `baselines.py:51` already said to use an interior sample K0, and where the true state is known analytically (rest) that is better still than K0.

### [D-127] Continuity defect term: SUBNET windows become true multiple shooting by pricing the inter-segment gap
**Date**: 2026-07-25
**What**: The training loss gains an optional continuity (defect) penalty. A training sample now spans `n_seg` contiguous segments of `nf_seg` steps each. Each segment is independently encoder-initialised (as today), and at every internal boundary the loss adds `w * ||x_j0_encoder - x_{j-1}[end]||` over the normalised state, using a **non-squared** norm. Weight `w` is set from the measured residual covariance, not swept. `n_seg=1` reproduces the current loss bit-for-bit, so the feature is an exact no-op when off.
**Why**: `docs/multiple-shooting-sweep-2026-07-25.md` established that the pipeline already *is* multiple shooting (Beintema et al., L4DC 2021, list "Multiple Shooting" in their own keywords; the encoder is the node-elimination step) with the continuity constraint omitted. Ribeiro et al. (Automatica 121:109158, 2020) Theorem 2 gives `V = V_M` exactly **only when the defects vanish**; with fully decoupled windows nothing in the objective prices what crosses a boundary, which is the formal statement of the 120x train/select horizon gap (`drift-conclusions-2026-07-25.md` §3 item 1). Their Theorem 1 at `L_h = 1` gives beta-smoothness `O(N^3)` in the *within-segment* length, so the defect buys the long-horizon objective over short gradient paths. That is why it is categorically different from longer `nf`, which was refuted empirically (SLURM 71013) and diverged at NF=900 exactly as `O(N^3)` predicts. The construct touches neither the model class (R2 intact) nor the spectrum (R3 intact) and needs no oracle.
**Ruled out**: (a) Squared `||d||^2` (Turan and Jäschke, LCSS 6:1897-1902, 2022): only exact as the weight goes to infinity; the `l1`/`l2`-not-squared/`l_inf` norms are exact penalties at finite weight. (b) Hard equality constraints via SQP (Ribeiro's own recipe): not implementable under Adam in this pipeline. (c) Free node states as decision variables (Bock 1981): the encoder already supplies them, so this would add parameters for nothing. (d) Sweeping `rho`: Fisher et al. (ECMWF weak-constraint 4D-Var, 2011) give the weight a statistical meaning as `Q^-1`, the inverse model-error covariance, which D4 measured. (e) Sweeping segment length as the intervention: Ribeiro Theorem 2 makes the solution invariant to it.
**Constrains**: "Multiple shooting was tried and failed (Optuna 69399)" is void and must not be re-cited: that run was pre-D-101 (silent Adam default lr 1e-3) **and** was an `nf` sweep under single shooting with no defect term. `tasks/lessons.md` `prove-overconstraint-dont-multiply-methods` and `docs/gantry-augmentation-problem-log.md` §12 carry the wrong claim and are corrected with this entry.

### [D-126] Self-consistent Python-generated training data (`pysynth`) so `model == data` up to the absorber
**Date**: 2026-07-25
**What**: A second data track generated by `scripts/gantry/pysynth-data/generate_pysynth_data.py`: the 22 records' MATLAB `u_total` is block-meaned to 4 kHz once, then the trajectory is **re-simulated in Python** by `drift_common.simulate_truth` (the verified 8-state RK4 truth) at the model's own `Ts = 2.5e-4` and `up_sample = 1`, in float64. Written as `.mat` with the same keys as the MATLAB records into `data/gantry/matlab/trajectory/pysynth/` (absorber on) and `pysynth_baseline/` (absorber off, `simulate_baseline`). Training on it sets `fs_orig = fs_new = 4000` so `d = 1` and neither `_resample_u` nor `y[::D]` does anything.
**Why**: The shipped data is ode45 variable-step (`RelTol 1e-4`) on a 20 kHz grid cast to `single` at save time (`gtd_save_record.m`), while the model is fixed-step RK4 at 4 kHz fed a block-mean input. The measured size of that mismatch is X `1.54e-6 m`, Y `5.17e-6 m` RMS with a nonzero tail-mean (`drift-demo/figures/f1_encoder_ic.npz`, `log_F` channel, float64 oracle at true x0) against an absorber signal of `2.196e-5 m` RMS: **24% of the target on Y, and systematic rather than noise**, so it enters the K=0 rows as a DC force. Re-simulating removes it by construction and makes the residual 100% absorber. The excitation design is preserved exactly because only the integration changes, not the input.
**Ruled out**: (a) Regenerating in MATLAB with a fixed-step solver (`kamtin-fp-model/` and the Simulink model are read-only; and it would still not be the model's own integrator). (b) Designing new inputs (would confound the change with a different excitation, losing the Y coverage and band design). (c) Raising `up_sample` to shrink the mismatch (`parameter-diagnostics/rk4_substep_sweep.json`: up_sample 1 vs 20 differs by only `4e-7` to `1.2e-5`, so the mismatch is the *solver and rate* difference, not RK4 substep error). (d) Treating the float64 toggle alone as sufficient: `use_f64=True` is currently a no-op end-to-end (deepSI hard-casts at `fit_system.py:684` and `encoders.py:288-289`, and `pre_encoder.py:389/393` builds the encoder map in float32), so the flag must be fixed before it means anything.
**Constrains**: `pysynth` is a diagnostic testbed, not the deliverable; every number measured on it must say so, and the real-data conclusions (C7, D4) remain the reference for what the true residual looks like. The absorber-off arm is the first test of whether the ANN stays at zero **with the encoder live in the production path**, which no rig has done (D1-D8 used true-state init with the encoder frozen).

### [D-125] D4 real-residual conventions: no-Coulomb baseline as primary, gross-velocity separability verdict, at-rest floor split into static hold and noise
**Date**: 2026-07-25
**What**: Four conventions fixed for the real-Telica residual characterisation (`scripts/gantry/drift-diagnostics/d4_telica_residual.py`). (1) The one-step residual is formed against the LPV-LFR baseline **without** the Coulomb term, using the physical parameters recovered by `run_telica_param_recovery.py` run 71447, with the fitted-Coulomb correction reported alongside every mean rather than folded in. (2) `train_param_recovery._build_sim_params` (lines 212-228) is mirrored in the script with the source cited, instead of importing `run_telica_param_recovery`, whose import-time side effects rewrite `precompute._load_trajectory`, `_TRUE_PARAMS` and `tr.simulate`. (3) The velocity-reversal separability verdict rests on the **gross-sliding** statistic (10 ms moving-average velocity, `|v| >` 5x the measured at-rest velocity noise std), with T0.5's raw sample-rate statistic reported beside it. (4) The 40 ms pre-motion at-rest segment is split into its **mean** (the static holding force the baseline does not model: X +177.8 N, Y +63.6 N) and its **std** (the noise floor of the construction: X 172.5 N, Y 43.0 N); PSD floors remove the mean so a real static offset is never counted as noise.
**Why**: (1) The augmentation is added to the no-Coulomb baseline, so that is the residual it must learn; but 71447 fitted its viscous parameters jointly with a trainable `cc`, so the two must be reported separately, exactly as `docs/drift-problem-statement.md` §6 constraint 4's coupling note requires ("any structural claim must name the baseline it is stated against"). (2) Depending on import-order side effects is the provenance hazard `rig.py` was frozen to prevent. (3) The raw statistic says the Telica logs are well-balanced (mean|mean sign| 0.24 to 0.37), which is an artifact of 1 um position quantisation dithering the differentiated velocity near standstill; on the gross statistic **0.00% of 85,358 gross-sliding samples travel backwards**. Taking the raw number at face value would have produced a parity result out of quantisation noise, the exact failure T0.5 exists to prevent. (4) With `dv = 0` at rest the residual is minus the net modelled force, so the at-rest mean is a physical quantity, not noise; calling it a floor would have inflated the floor by 4x on X and hidden the finding.
**Ruled out**: (a) Importing the recovery script and undoing its patches (fragile, order-dependent). (b) Using the fitted-Coulomb model as the primary baseline (measures the residual of a model the augmentation is not added to). (c) Rijlaarsdam et al.'s two-variance floor as specified: its noise leg needs repeated periods of one realisation and each Telica log is a single stroke, so it is only half-available; the iter0-versus-iter8 leg is reported as an upper bound because the feedforward differs. (d) Sample-matched windows (400 samples = 20 ms at 20 kHz), which would move the "DC" band boundary from 10 Hz to 50 Hz; the window is time-matched instead.
**Constrains**: Any future real-residual number must state which baseline it is against and must use the gross-sliding cut for anything velocity-signed. The parity split stays closed until a reverse stroke exists in the data (one log per operating point travelling -40 mm X / -80 mm Y at matched speed); everything else that analysis needs already runs in 21 s.

### [D-124] Profile-interval threshold `delta_L` = per-window SEM when the null is bit-deterministic
**Date**: 2026-07-25
**What**: D1's profile-likelihood threshold uses a three-step measured fallback chain: (1) spread of the ANN-off full-batch loss across seeds; (2) if that is exactly zero, the float32 repeatability of `L(0)` over ten evaluations; (3) if that is exactly zero too, the **per-window standard error of the mean** of `L(0)` over the 256-window bank. On the perfect-match null steps 1 and 2 are both exactly zero, so step 3 is what was used: `delta_L = 1.266e-13`. The payload records `delta_L_source` verbatim and the results doc names the fallback.
**Why**: At the zero-output init the ANN emits exactly zero, so the loss and its bias-derivatives do not depend on the ANN weights at all: `L(0) = 8.847964678634912e-13` on all three seeds, to the last digit, and ten repeat evaluations agree bitwise. The README's two thresholds are therefore both identically zero, not small. The per-window SEM is the sampling resolution of the loss statistic given this data, computed from the measured per-window losses, with no truth-model quantity and no hand-picked number in it, and it is the finite-data analogue of the noise-derived threshold in Raue et al.'s profile-likelihood method (DOI 10.1093/bioinformatics/btp358).
**Ruled out**: (a) Declaring the interval unmeasurable (the interval is D1's cheapest decisive statement and a measured threshold exists). (b) Any fixed relative threshold such as "1% of L(0)" (a hand-picked number, banned by the no-oracle rule). (c) Reporting the zero seed spread as reproducibility evidence (it is a deterministic identity, not an empirical result).
**Constrains**: Every quantity measured at the zero-output init on this rig is deterministic given the rig, so a seed count is not a sample size there and must not be reported as one. Any future profile interval on this rig should use the same chain and record which step it landed on.

### [D-123] Diagnostic units are written into the diagnostic's own folder via a runtime redirect of `rig.DATA_DIR`
**Date**: 2026-07-25
**What**: The `drift-diagnostics` scripts (D1, D2, D3) keep the frozen rig's unit format by calling `rig.write_unit` unchanged, but wrap it in a local `write_unit_local` that temporarily points `rig.DATA_DIR` and `rig.MANIFEST` at `scripts/gantry/drift-diagnostics/data/` and restores them afterwards. `rig.py` is never edited and nothing is written into `drift-fix-trials/`. Reads of existing units (`R.read_unit('T0.1b', ...)` for the measured DC direction) still resolve against the trials folder. D3, which runs concurrently with D2, writes its own `manifest_d3.json`.
**Why**: The campaign README requires both "use `R.write_unit`" (so every unit carries the `rig_hash` and git stamps that make cross-run comparison auditable) and "never write into `drift-fix-trials/`". `rig.write_unit` writes into `rig.DATA_DIR`, which is the trials folder, so the two cannot both be followed literally. Redirecting the module attribute at runtime satisfies both. The separate manifest avoids a lost-update race, since `write_unit` does a read-modify-write on one JSON file and two training jobs were run concurrently to fit the wall clock.
**Ruled out**: (a) Editing `rig.py` to parameterise the data directory: forbidden, and it would change `rig_hash`'s provenance story. (b) Hand-rolling a JSON schema in the new folder: drops the `rig_hash` and `git` stamps, which are the point of the unit format. (c) Serialising D2 and D3 to share one manifest: about 45 minutes of wall clock for no measurement gain.
**Constrains**: Any future folder that reuses the frozen rig should follow the same pattern (redirect, never edit) and should take its own manifest if it can run concurrently with another rig job.

### [D-122] `tasks/lessons.md` no longer auto-read at session start; 8 cross-cutting rules promoted into CLAUDE.md
**Date**: 2026-07-25
**What**: The `## Step 0 — Every Session` section of `CLAUDE.md` ("Read `tasks/lessons.md` before any work") is replaced by `## Standing Rules (always active)`: eight distilled one-liners (answer-before-code, modify-only-what-was-asked, use-users-exact-term, respect-session-boundaries, commit-after-direction-given, one-recommendation, no-em-dashes, removing-a-rule-needs-justification). `tasks/lessons.md` is retained unchanged as the full 117-rule ruleset, read **on demand**: after a user correction (the `LESSON CHECK` UserPromptSubmit hook in `~/.claude/settings.json` still fires every prompt) or before working a thread it covers.
**Why**: User-observed, and this is the decisive evidence: **performance measurably improved in sessions where the lessons context was split/lighter**. Mechanism is attention dilution, and the measured cost supports it — `lessons.md` is 64,772 chars (~16,200 tokens), **6.3x the whole of `CLAUDE.md`** (10,345 chars), loaded every session, with most rules irrelevant to any given task. Corroborating incident the same session: I violated `respect-explicit-session-handoff-boundaries` (line 33) and used em-dashes against `no-em-dashes` (line 110) roughly fifty times, having never opened the file — an eager-load that is skipped in practice is worth less than ten lines actually resident in context. The promotion keeps the highest-generality rules (the ones that apply to EVERY task regardless of thread) at ~250 tokens instead of 16,200.
**Ruled out**: (1) Deleting `lessons.md` or its `CLAUDE.md` references entirely — the hook would still fire and the thread-specific rules (drift/ARTBP, zero-mean, augmentation design, figure discipline) retain real value on their own threads; this is a load-strategy change, not a deletion. (2) Two-tier trim only (move the six 2-3.5k-char inline narratives to the existing `archive/lessons-incidents.md`, ~27% lighter) — proposed first, but 27% off 16k tokens still eager-loads ~12k of mostly-irrelevant rules; it does not address the user's observed effect. Still worth doing later as hygiene, since those narratives violate the file's own declared two-tier structure. (3) Lazy-load with no promotion — rejected because rules would then only arrive AFTER a mistake, which is precisely how the two violations above happened.
**Constrains**: New rules still go to `tasks/lessons.md` via the 3-criteria gate; only rules proven cross-cutting get promoted to the CLAUDE.md standing list, which must stay short (~8 items) or it recreates the problem. **Conflict of interest disclosed**: this decision reduces the constraints loaded on the assistant by default, and was assessed by the assistant; it was adopted on the user's own cross-session performance observation, over the assistant's initial recommendation to keep the auto-read.

### [D-121] Literature search routed through a `deep-research` skill run in subagents, replacing ad-hoc web search
**Date**: 2026-07-25
**What**: New project skill `.claude/skills/deep-research/SKILL.md` encoding a five-stage procedure (SEED via dblp -> EXPAND via OpenAlex author-ID/citation-graph enumeration -> FILTER to control venues -> ACCESS via `locations[]` -> READ full text), plus a mandatory **Research Log** output section. Evaluation harness in `ACCESS_PROBE.md`: a six-probe **capability probe** (broad topic, renamed concept, venue-specific, direct fetch, citation graph, unreachable boundary) returning an access map, full-text reachability counts, and ranked optimizations. `CLAUDE.md` Workflow updated to route web/literature deep research through this skill in subagents. Supporting install (outside the repo): conda env `papersearch` with `paper-search-mcp 0.1.4`, registered as user-scope MCP server `paper-search`; `~/.config/paper-search-mcp/.env` holds the Unpaywall email.
**Why**: Measured 2026-07-25 on this thesis's own topic. Keyword search fails here: OpenAlex `search=` for "orthogonal projection regularization model augmentation" returned brain-tumor classification, Quantum ESPRESSO and climate projections; the same query shape on three sources never surfaced Györök et al. 2026 (`10.1016/j.ifacsc.2026.100376`). An OpenAlex **author-ID enumeration** (`filter=author.id:A5088619613,from_publication_date:2024-01-01`) found it in one call. Root cause: control publishes in IFAC/CDC/ECC/ACC + Elsevier/IEEE, poorly covered by arXiv and poorly ranked by keyword indices, and authors rename concepts between papers ("projection-based regularization" 2025 -> "orthogonal-by-construction" 2026), so a query written from the old vocabulary structurally cannot match the new paper. Enumeration beats matching. Second measured lesson: query OpenAlex `locations[]`, not `best_oa_location` — for the 2026 paper the latter is `pdf_url=None` (looks paywalled) while the former lists both an arXiv PDF and the published Elsevier PDF on TU/e Pure.
**Ruled out**: (1) Relying on the `paper-search-mcp` MCP server alone — its search tools are keyword-based and reproduce the failure above; kept as a secondary convenience/download path. (2) Adding PubMed/bioRxiv/medRxiv/Europe PMC/DOAJ coverage — biomedical, zero yield for control topics; the skill explicitly skips them. (3) A single sequential research agent — subagent cost is only repaid by fan-out, so the skill spawns one agent per independent seed/sub-question. (4) Keeping `CLAUDE.md` line 118 as-is (subagents restricted to codebase fan-out) — it contradicts routing literature work to agents, so that clause is rewritten rather than appended to. (5) Sci-Hub retrieval — the OA chain (arXiv -> TU/e Pure -> repository -> Unpaywall -> TU/e subscription) resolved every paper needed in this session, including the "hybrid, pdf=None" case that first looked closed.
**Constrains**: Any literature/state-of-the-art request should invoke `deep-research` rather than ad-hoc `WebSearch`. Every run must emit the Research Log, whose **Suggested skill fix** line is the iteration hook for revising `SKILL.md`. Skill effectiveness is measured with `ACCESS_PROBE.md`, re-run unchanged after each `SKILL.md` revision so successive versions stay comparable. **Ruled out during design: a recall exam with a hidden answer key.** It was built first and discarded — the key named the target DOIs, and `CLAUDE.md`'s Key File Map, the skill directory listing, and this very decision entry all pointed at them, so a test agent could read the answers instead of searching. Any evaluation whose target is documented in this repo is leaky by construction; the probe avoids this by measuring coverage and failure modes, which have no answer to leak. The Györök 2026 by-construction result is now a live input to the thesis framing (cf. D-117/D-118, which already use the Györök LFR-contraction rate as the model for by-construction stability) — whether it already covers LPV/LFR/MIMO is unverified and is the first real test case.

### [D-120] ARTBP Phase B reframed: loss-curvature kappa(H) as the mechanism instrument; the trained DC is Adam-driven, not a loss optimum
**Date**: 2026-07-22
**What**: The ARTBP verification plan (`scripts/gantry/ARTBP/README.md`) is reframed. The original Phase B target ("build a convergent `true_grad(T)` ground-truth gradient in the DC direction and show a ~1/nf truncation bias, then verify ARTBP unbiasedness in Phase D") is DROPPED as the wrong target. Replacements: (1) the **loss-landscape curvature `kappa(H)`** (and the loss-optimal constant `c*(H)`) is adopted as the bounded, validated MECHANISM instrument; (2) verification of the FIX moves to a **training-dynamics** instrument (the ANN's DC trajectory under {fixed, ARTBP-geometric, ARTBP-poly-tail}, extending v12), not a static-gradient unbiasedness test.
**Why**: Two diagnostics decided it. `ground_truth.py`: the raw DC-direction gradient does NOT converge on the z=1 (marginal) dY axis; its variance explodes ~H^3 while its mean is unresolvable (SE ~= mean at every H, |t|<1.3) -> there is no convergent `true_grad(T)` to serve as ground truth. `instrument_select.py` (Phase B0, 256 windows, ann-route injection matched to training) then ran three pre-registered tests: **Test 1 (better)** kappa rel-SE <0.05% at every H vs the raw gradient's ~42% (~1000x better conditioned); the gradient blows up only because `g = -kappa*c*` (identity held to 3 sig figs) with kappa~H^3.8 and c*->0. **Test 2 (positive control)** planted offsets +4e-6/+1e-6/+3e-7 recovered to 99.8-100.1% (SE ~1e-9), more sensitive than the v8-inj training readout (64% at 3e-7). **Test 3 (what we want)** the loss-optimal constant is SMALL and convention-dependent (`c*(400)` = +1.7e-6 faithful ann-route, -2.4e-6 v11 state-row), heads to ~0 with horizon, and never equals the trained -4.5e-6 DC. Conclusion: the DC is Adam walking a near-flat, weakly-constrained direction (small kappa at the training horizon), NOT a loss-geometry optimum; kappa(H) growing is the restoring curvature ARTBP supplies via rare long rollouts (reconciles v3b "systematic gradient" with v11/SGD "flat direction / Adam artifact", and explains v12's ARTBP DC collapse). Data: `scripts/gantry/ARTBP/data/{b_bias_gap,b0_instrument_select}.npz`, figures `figures/b0_{landscapes,instrument}.png`.
**Ruled out**: (1) `true_grad(T)` as ground truth -- non-convergent on the marginal mode (variance ~H^3, mean unresolvable). (2) The raw autograd gradient as the instrument -- it is the exploding product `-kappa*c*`; kappa/c* are the bounded, ~1000x-better-resolved quantities. (3) "ARTBP unbiasedness toward the loss optimum" as the Phase D test -- the loss optimum is ~0 and the DC is Adam-driven, so an unbiasedness test misses the mechanism. (4) v11's raw dY-state-row injection as canonical -- it is a different physical direction from the ANN's actual output route (sign flips vs the faithful ann-route injection); the ann-route injection (patched `ann.forward`, matching how the trained DC enters) is canonical.
**Constrains**: Phase B deliverable is the kappa(H)/c*(H) mechanism curve (in hand). Phase D/E verify the fix by the trained DC trajectory (fixed vs ARTBP-geometric vs ARTBP-poly-tail Eq. 14, matched average cost), extending v12, with kappa(H) as the mechanistic backbone. Any such training run gets a D-090 row. The paper's variance bound holds only for geometrically-decaying memory (Sec. 4); the z=1 axis has none, so the poly-tail (Eq. 14) is necessary not optional, and variance is a live risk to test, not assume.

### [D-119] Encoder normalization scaled from norm.x_all unconditionally (baseline_states.npz source removed)
**Date**: 2026-07-20
**What**: `build_model` (`scripts/gantry/gantry_dynamic/model.py`) no longer loads `data/gantry/baseline_simulations/<mode>_LPV/baseline_states.npz` to supply state trajectories to `normalize_linear_ss_matrices`; it uses `norm.x_all` (the finite-difference logical states that also define `x_mean`/`std_x`) unconditionally.
**Why**: Frame consistency. The D-055 encoder convention fix subtracts `x_off = x_mean/std_x` (`pre_encoder.py:465`) and `Gantry_State_Block` denormalizes with the same `norm.std_x`/`x_mean` (`blocks.py:787`), so the state scaling baked into the encoder matrices must be that same `std_x`. Scaling with npz-derived stds is a latent frame divergence (state-proportional encoder x0 error). The 2026-07-20 normalization audit verified the `norm.x_all` path round-trips an exact LTI with no offset (dY mean error -8.3e-12 m/s over 1297 windows) and that the npz never existed for the `augmentation`/`joint` modes: every logged gantry run printed the fallback WARNING, so this change is behavior-preserving for all logged runs.
**Ruled out**: (1) Keep the npz branch and assert its stds match `norm.std_x`: the npz stds can never legitimately differ from the frame stds, so the assert is either trivially true or blocks a run over a file that should not be consulted; dropping the source is simpler and equivalent. (2) Threading `std_x` into `normalize_linear_ss_matrices` explicitly: requires changing Jan's util signature for no gain (the internal np.std vs +1e-8 mismatch is at most 1.7e-5 relative, measured negligible in the audit).
**Constrains**: `baseline_states.npz` is no longer a pipeline input. Standalone diagnostic scripts that mirror the old branch (`_local_test_stride.py`, `augmentation-error/diag*`, `encoder-augmentation/diag*`) are historical snapshots and are left untouched. If a future mode wants baseline-simulation-derived normalization, it must change `compute_normalization` (the single frame source) so the whole pipeline moves together, never the encoder path alone.

### [D-118] Stability-preserving prototype: by-construction Lipschitz cap on the augmentation ANN
**Date**: 2026-07-18
**What**: Implement and test a by-construction Lipschitz cap on the static augmentation ANN as the first stability-preserving prototype (D-117 route). `scripts/gantry/gantry_dynamic/lipschitz.py`: `SpectralCap` = a SOFT per-layer spectral-norm cap (`W -> W·cap/max(σ(W),cap)`, σ by warm-started power iteration) registered as a weight-parametrization on each ANN Linear; `apply_lipschitz_cap(net, L)` caps each of the n Linear layers at `L**(1/n)` so the overall Lipschitz <= L (tanh is 1-Lipschitz). Wired into `model.py:build_model` via env `ANN_LIPSCHITZ=<L>` (default off = exact current behaviour). Test harness `v6_lipschitz_sweep.py` sweeps L, trains the full X+Θ+Y augmentation identically per L, and measures long-horizon free-run drift (full-ANN vs ANN-off) + windowed nf-RMS fit. The cap L is the **static-ANN analog of the Györök LFR-contraction rate `ᾱ`** — a magnitude/Jacobian bound (our augmentation is a static ANN with no learnable `{A,B_w,C_z}`, so the faithful stability-by-construction tool is a Lipschitz bound: Revay Lipschitz-bounded networks, cited by Drenth/Györök).
**Why**: v5 showed the dominant long-horizon drift is the ANN DESTABILIZING the free-run (dynamic, state-dependent output), not a DC. Bounding the ANN's Lipschitz constant by construction bounds its contribution to the augmented state-transition Jacobian, so the augmentation cannot inject unbounded gain. Drenth already gives WELL-POSEDNESS (`D_zw=e^{-N}`); this adds the missing STABILITY half at the learned block. Soft spectral CAP (not torch `spectral_norm`, which forces σ=1) preserves the zero-init gentle start the encoder-init story relies on.
**Ruled out**: (1) torch `spectral_norm` — forces σ=1, destroys the zero-init final layer (jump to full strength after one step). (2) A soft contraction/Lipschitz PENALTY — not by-construction (the point is a structural guarantee). (3) Editing Jan's `Static_ANN_Block` — instead wrap `ann.net` via a `torch.nn.utils.parametrize` registration from our code (`blocks.py` untouched). (4) Full Revay LBEN parameterization now — heavier for the same first signal; revisit if the cap route is promising.
**Constrains**: The cap is a MAGNITUDE bound, not a sign/dissipativity constraint, so it is conservative for the genuine z=1 integrator (D-117): it limits destabilization but does not structurally protect the marginal mode. Decision rule: an L that brings full-ANN drift to ~the ANN-off baseline WITHOUT raising the windowed fit (and without a leaky-integrator low-freq error) confirms stability-by-construction; if drift only dies by degrading the fit / leaking the integrator, contraction is too blunt → the passivity route (pHNN, D-117) is indicated. Any sweep run gets a D-090 row (done). Mechanism verified pre-launch (per-layer σ<=L^(1/n), zero-init preserved, pipeline builds).

### [D-117] Pivot: from DC-artifact fix to stability-preserving augmentation BY CONSTRUCTION
**Date**: 2026-07-18
**What**: Reframe the drift problem and the fix strategy. The v5 DC-null counterfactual (`scripts/gantry/gantry-zero-mean/v5_dc_null_counterfactual.py`, on the 71167 drifted checkpoint) showed the dominant long-horizon drift is the LEARNED augmentation DESTABILIZING the free-run on the marginally-stable (K=0, z=1) axis: the NN's STATE-DEPENDENT output makes Y ~50x worse than the physics baseline alone over 2 s (ANN-off Y drift 1.3e-4, full-ANN 6.7e-3), while the short-window loss looks fine. Nulling the K=0 DC fixes X (back to baseline) but makes Y WORSE (1.79x) at a ~7% windowed-loss cost -- so the DC is only the minor X-axis story and is weakly load-bearing, NOT the cause of the dominant Y drift. **Decision: pursue stability-preserving augmentation BY CONSTRUCTION** -- parameterize the learned LFR block so the AUGMENTED model is guaranteed stable -- instead of post-hoc symptom fixes. Adopt the group's existing machinery: **Györök, Drenth, Verhoek, Schoukens, Tóth, Péni 2026 (arXiv 2604.11421)** = constraint-free well-posedness (Cayley `D_zw`) + contraction of the full LFR augmentation `{A,B_w,C_z}` (baseline need only be Lipschitz); and/or **Moradi, Beintema, Jaensson, Tóth, Schoukens 2025 (arXiv 2502.14432)** = passivity-by-construction pHNN + output-error noise + SUBNET for real noisy data. First-pass scan + deep-research prompt in `literature/stability-training/`.
**Why**: v5 evidence + the first-pass literature scan. Post-hoc fixes are refuted or partial and do not transfer to the real nonlinear (Telica) data, where a genuine bounded nonzero-mean correction may be physically required (Coulomb friction). The clean, real-data-transferable, thesis-worthy path is stability-by-construction, and the group already owns the framework (same LFR the pipeline uses). Ties [[project_passivity_gap_scope]].
**Ruled out**: (1) DC/zero-mean pin (the ZeroMeanPin, MODE='zeromean'): fixes only the minor X axis, worsens the dominant Y drift, is weakly load-bearing (~7% fit cost), and on real data a genuine DC may be physical. (2) Longer training windows: refuted by the nf-sweep (DC ~ 1/nf, drift persists to nf=3200). (3) SGD as the deployed optimizer: block-heterogeneity -> underfits (v2 optimizer report). (4) Generic pole/spectral regularization: corrupts the genuine z=1 integrator physics.
**Constrains**: The open research question is PRESERVING the marginally-stable (z=1 integrator) modes: strict contraction (Györök `ᾱ<1`) / Schur / spectral reg would pull the true integrator inside the unit circle and corrupt the physics; the resolution is marginal contraction (`ᾱ→1`) or a passivity / lossless-mode carve-out (port-Hamiltonian). Any candidate must (a) preserve the z=1 physics, (b) transfer to real noisy data (do NOT assume the correction is zero-mean on real data; gate on a HAC zero-mean residual test), and (c) be validated on free-run BFR / long-horizon drift on held-out operating points and unseen profiles, >=5 seeds -- never matched short-window training loss. Supersedes the DC-centric framing of the recent gantry-zero-mean investigation (which stands as the diagnosis that led here).

### [D-116] Coulomb friction added to the LPV-LFR baseline as u_eff = u - F_c at both u-sites; cc1/cc2/ccy trainable; implemented as override modules in real-data-verification (baseline core untouched)
**Date**: 2026-07-16
**What**: Add Coulomb (dry) friction to the dual-gantry LPV-LFR baseline as a generalized force, injected as an effective input `u_eff = u - F_c`, with `F_c = P * (cc .* sign(P' * qdot))` mapping per-actuator stage-frame Coulomb forces to logical coordinates (same P transforms the baseline uses for input/output). `cc = [cc1, cc2, ccy]` become trainable physical parameters recovered from data. Implemented as NEW override modules in `scripts/gantry/real-data-verification/` (`coulomb_lfr.py`: Coulomb-aware forward/RK4/`simulate`; `lfr_param_block_coulomb.py`: a `ParameterizedLFRBlock` subclass adding cc), wired into `run_telica_param_recovery.py` via its existing patch mechanism. `lpv_lfr_baseline/` core is NOT edited. `sign()` is smoothed as `tanh(v/v0)` for BPTT (v0 is HEURISTIC, a differentiable-friction standard per Makkar & Dixon 2005; hard sign as a reference). Verification is the LPV-LFR-vs-direct equivalence in Python: the `lfr` forward (`u_eff` through `G`) must equal the direct collapsed form `A_c(Y) x + B_c(Y) (u - F_c)` across Y including Y=0, and cc=0 must reproduce the baseline bit-for-bit. Full plan: `Matlab-scripts/coulomb-friction/PLAN.md`. CORRECTION (2026-07-16): an earlier plan to verify against enabled Simscape Coulomb blocks was dropped -- those blocks are orphaned XML in `gantry_2025a.slx`, not present in the loaded model (`find_system` returns none), so no Simscape Coulomb oracle exists; MATLAB is off the critical path.
**Why**: Job 70821 recovery plateaus at 40-68% open-loop NRMSE with the viscous friction pinned at 6-7x its datasheet maximum (cg1 136 to 841, cy 98 to 664), the signature of a viscous-only structure faking a rate-independent (Coulomb) force it cannot represent; the open-loop residual is a low-frequency hold-phase drift, i.e. a force/DC-level error. Garcia 2013 (the FP model's source) includes Coulomb with identified values cc1 = 16.8 N, cc2 = 18.35 N, ccy = 11.6 N; the FP and Simscape models contain the terms but commented/disabled. Adding Coulomb should absorb the hold-phase drift and let cg/cy relax toward the datasheet.
**Format detail (the Y=0 trap)**: in this LFR realization `u` appears twice, inside `fnet` (which flows through the M(Y)^-1 loop into w) and directly in the `[x, w, u] @ G` concat. At Y=0, `w=0`, so the acceleration comes only from the direct `Bu*u` term; `u_eff` MUST replace `u` at BOTH sites or friction silently vanishes at Y=0. G is unchanged: it is built from N0, d0, M1, M2, K, C, and `u` never enters it, so cc never touches it.
**Ruled out**: (1) A standalone MATLAB ode45 wrapper `base(u - F_c, x, ...)` as the delivery form, wrong format; the pipeline needs it inside the LPV-LFR `simulate` that `train_param_recovery.py` uses. MATLAB/Simscape is a cross-check only. (2) Editing the `lpv_lfr_baseline/` core, the Telica work overrides the baseline from `real-data-verification/` (like the existing loader and windowed-validation patches), keeping the baseline as the clean no-Coulomb reference. (3) Folding Coulomb into the `C` matrix, it is nonlinear (sign(v)) and does not fit the LTI matrices. (4) Subtracting `F_c` only inside `fnet`, it vanishes at Y=0. (5) Hard `sign()` during training, zero gradient almost everywhere; `tanh(v/v0)` surrogate instead.
**Constrains**: `cc = 0` must reproduce the baseline `simulate` bit-for-bit (this is both the format gate and a fidelity check on the re-implemented forward, since the baseline core is not edited). Acceptance for the real-data payoff is data-derived (open-loop NRMSE lower than 70821 on held-out operating points, hold-phase drift reduced, recovered cc positive and order tens of N, viscous cg/cy relaxing toward the datasheet range), never an oracle/model threshold. Any Telica retrain gets a D-090 run-table row before launch.
**Note (2026-07-16) -- cc init source:** the trainable Coulomb init `CC_INIT` uses the **Telica datasheet** (literature/gantry/telica-xyz-0750-0800-data.pdf), FORCE CAPABILITIES row "Static friction (maximal value)": X 2x43 N, Y 49 N, i.e. `CC_INIT = (43, 43, 49)` -- the machine's own values (the same datasheet's dynamic-friction row 2x136 / 98 is the viscous cg/cy init, D-112). Supersedes Garcia 2013's 16.8/18.35/11.6 N (a different gantry). Caveat: the datasheet value is STATIC/breakaway friction (>= kinetic Coulomb) and a maximal spec value, so it is an upper-bound init for the (trainable) kinetic cc, not an exact kinetic value.

### [D-115] drift-visual f03 becomes the oracle / baseline / encoder decomposition
**Date**: 2026-07-15
**What**: f03 gains a third curve: the oracle (FP + true MSD, true x0, `oracle_open_loop` with the
central-difference vdelta_a seed of OE-1) at the pipeline's exact ts/up_sample (fairness rule,
D-097). Three curves, one change per pair: oracle@true-x0 (discretization floor, Y rms 9.2e-6),
baseline@true-x0 (floor + absorber effect = the ANN target, Y rms 7.3e-4), baseline@untrained-
encoder-x0 (deployment init, Y rms 2.1e-4 -- the encoder absorbs part of the residual into its
state estimate). Data added to the existing `f03.npz` by a one-off deterministic patch script
(scratchpad, mirrors the permanent `gen_real()` block; manifest records the patch) to avoid
rerunning the ~10 min full generation for one 2-min sim. `gen_fake`/`gen_reuse` carry the same
`err_oracle_*` keys (reuse maps demo1's `stage_F`/`log_F`).
**Why**: The user's proposal: the old f03 (true-x0 vs encoder-x0 only) raised the first-thought
question "is the model or input bad?" at the +0.8 mm parked Y offset. The oracle answers it in
the figure: same model + absorber = floor, so the gap IS the absorber effect.
**Ruled out**: full regeneration (identical outputs except f03; wastes the baseline + free-run
sims); separate fourth figure (3 curves fit the panel budget and keep one decomposition in one
place).
**Constrains**: The oracle is a sim-only diagnostic reference, never an acceptance threshold
(standing rule). Deck oracle numbers differ from the run log's ref[1] (9.0e-5): the deck uses
the OE-1 central-diff seed; state this when comparing.

### [D-114] drift-visual deck regenerates from run 71167's rescued `_last` checkpoint
**Date**: 2026-07-15
**What**: Implemented `SOURCE=real` in `scripts/gantry/drift-visual/generate_data.py` and pointed
`config.py`'s `CKPT` at `simulations/gantry_subnet/augmentation_linear_map/71167/gantry_drift_71167_last.pth`
(`CKPT_TAG = gantry_drift_71167_last`). That file is run 71167's end-of-training (drifted) model,
rescued from the cluster's `~/.deepSI/checkpoints/SSE_Interconnect_OrthLoss_5i7INg_last.pth` and
identified by mtime (Jul 15 01:17 = fit end) plus content fingerprint (`bestfit` =
0.000166137600899674 = 71167's initial validation sim-RMS; 20 epochs). `gen_real()` reuses the
proven drift-demo machinery (demo3 shadow free-run, demo1 baseline decomposition, f5 horizon
computation) and regenerates all deck npz from this one checkpoint; the f07 universality bank is
the reused 9-checkpoint bank with 71167 appended as a measured 10th member. `SOURCE=reuse` keeps
its own tag (`gantry_drift_last`) so old-checkpoint arrays are never mislabeled with the new tag.
**Why**: The prior deck checkpoint (`gantry_drift_last.pth`, Optuna 69399 trial 3: 5 epochs,
lr 1.49e-8, stride 100) was a deliberately rough rescue. Run 71167 is the purpose-built drift-deck
run (20 epochs, lr 1e-7, stride 10, nf=400) with monotone window-loss descent (train nf-RMS
3.81e-5 -> 3.33e-5 m). The run folder's own artifacts (`gantry_ckpt_71167.pt`, `gantry_71167`) are
useless for the deck: fit reloads `_best` at the end, and `_best` = epoch 0 = zero-output ANN
(run-log VERDICT: "ANN inactive", +0.0% vs baseline).
**Ruled out**: (1) Using `gantry_ckpt_71167.pt` — epoch-0 weights, no drift to show. (2) Retraining
locally to recreate `_last` — hours of compute for a file that already existed on the cluster.
**Constrains**: `gen_real()` refuses to run (hard exit) if the loaded ANN's captured output is
identically zero, so an epoch-0/`_best`-type checkpoint can never silently produce a "no drift"
deck. Every manifest entry and provenance footer cites run 71167.

### [D-113] Independent nf window-length sweep (grid) instead of extending the warm-started curriculum
**Date**: 2026-07-13
**What**: Reconfigured `gantry_optuna.py` to `MODE='nf_sweep'` — a GridSampler over `nf ∈ {800, 1600, 2400, 3200}` at FIXED `lr=1e-7`, full X+Θ+Y routing `[0..7]`, free ANN (`joint=False`, nominal θ, no orth). Each nf is a self-contained trial built FRESH from the encoder init (no warm-start), 8 epochs each; the best nf is retrained 8 epochs on the full val set with the state-recovery diagnostic. Reuses the existing `run_search_main`/`objective` grid path (the previous orth-smoke config in the `MODE != 'curriculum'` branch was repointed to this regime).
**Why**: The warm-started curriculum (70903) showed full-sim val RMS improving monotonically with nf across rungs (1.9e-3 → 8.3e-4 → 4.6e-4) but never beating the epoch-0 encoder-init baseline (8.0e-5), and rung 0 (nf=400) actively *degraded* the init before longer rungs clawed back. Warm-starting confounds "longer window helps" with "recovering from rung-0 self-degradation." Independent per-nf trials from the same init isolate the window-length effect and directly test whether ANY window lets a trained ANN beat the 8.0e-5 init. Grid search is safe from the cross-rung `cal_validation_error` bug (below) because `objective` restores the real selector in `finally` each chunk and never reloads a checkpoint between trials.
**Also fixed**: the curriculum path crashed nondeterministically at rung ≥1 with `TypeError: '>=' not supported between float and NoneType`. Cause: the D-095 nf-probe pickles into `_last.pth` as `_noop_cve` (returns None); `checkpoint_load_system` replaces `__dict__` wholesale, so after the inter-rung reload the next rung's probe wrapped a None-returning selector. Fix: `fit_sys.__dict__.pop('cal_validation_error', None)` after `checkpoint_load_system('_last')` restores the class method. (deepSI's `cal_validation_error` masks this by constructing but never raising its final `NotImplementedError`, so an unmatched measure returns None silently.)
**Ruled out**: (1) Extending the curriculum ladder to longer nf — faster to a long window but keeps the recovery confound (user chose independent). (2) Including nf=400 — 70903 already shows it degrades. (3) Warm-starting each grid trial from the previous — reintroduces the confound.
**Constrains**: The sweep answers window-length capability only at lr=1e-7 / full routing / free ANN. If no nf beats 8.0e-5, the conclusion is that window length alone does not make the ANN improve on the encoder init (points to the drift/negation problem needing a separate mechanism, not a bigger window). Runs on CPU (cluster); nf=3200 training arrays ~0.5 GB.

### [D-112] Telica pipeline anchored to the machine's own parameters: datasheet masses pinned, per-axis force scale on Kt, loader repointed to Telica 1.mat
**Date**: 2026-07-12
**What**: Three coupled changes to the real-data parameter-recovery pipeline.
(1) `telica_loader.py` reads Kt and fs from `kamtin-data/Telica 1.mat` (the user-designated
machine-parameter export; values identical to the previous source: Kt_X 109, Kt_Y 77.6 N/Arms,
fs 20 kHz). `kamtin-data/Telica.mat` is off-limits per user instruction (2026-07-12).
(2) Force conversion gains a per-axis scale: `_A_TO_N = [Kt_X, Kt_X, Kt_Y] * _S_FORCE` with
`_S_FORCE = [3.469, 3.469, 3.202]` (HEURISTIC). Derivation: reconciling the linear-ID mass
lumps (diag_linear_identification.py, 2026-07-05: m_total 26.229 kg, mh 5.933 kg) against the
ETEL TELICA datasheet (ASME-YGNN-08-0750-0800W3 v1.0) moving masses (X-moving 91.0 kg,
Y-moving 19 kg) gives s_X = 91.0/26.229 = 3.469, s_Y = 19.0/5.933 = 3.202 -- near-identical
across axes despite different motors, the signature of a global conversion factor. A factor 2
of it is machine-config documented: `Telica 1.mat` `Motor.SubAxes` lists a forcer PAIR per
logged channel (X1-L/X1-R, X2-L/X2-R, Y-L/Y-R), so one logged ampere drives two forcers. The
residual ~1.73/1.60 (candidate: sqrt(3) three-phase current convention; 2*sqrt(3) = 3.464
matches s_X within 0.2%) is an open question posed to Kamtin: "is logged MF30 per sub-axis,
and in what convention relative to Kt = 109 N/Arms?"
(3) `run_telica_param_recovery.py` initializes at a datasheet-anchored parameter set instead
of Kamtin nominals: mh = 19.0 (THEORY: datasheet Y-moving mass, incl. 1.7 kg Z stage, excl.
payload), m1/m2/mb = 16.81/17.63/37.56 (THEORY: sum = 91-19 = 72 kg; HEURISTIC: split by
Kamtin proportions, split not on datasheet), cg1 = cg2 = 136, cy = 98 N/(m/s) (HEURISTIC:
spec maxima, "dynamic friction maximal value", upper bounds, trainable). kb/cb/J/d/Lb keep
Kamtin values (not on the datasheet). `_TRUE_PARAMS` is patched to the same dict so the
param_table % columns read "deviation from datasheet init". Also `checkpoint_interval=10`
is passed to `tr.train()` (the old default 100 with EPOCHS=40 wrote zero checkpoints, so a
post-training crash lost the parameters).
**Why**: The datasheet ruled out the "machine is lighter" branch of the D-077 degeneracy (the
real machine is 1.7x HEAVIER than Kamtin nominals), and made the mass deficit near-identical
on both axes, flipping the leading hypothesis to a global force underscale. Adopting datasheet
masses without the force scale would be self-defeating: training would drag the masses back
down the degenerate m/F direction toward the 26 kg the data demands.
**Ruled out**: Keeping Kamtin masses (contradicted by datasheet); treating 26 kg as the real
mass (contradicted by datasheet); oracle/model-based scale derivation (scale comes from data
+ manufacturer spec only); changing `lpv_lfr_baseline/` itself (all patches live in the
wrapper/loader, simulation pipeline untouched).
**Constrains**: Masses are now PINNED constants (manufacturer spec), not recovered -- report
them as such; the recoverable content shifts to damping/stiffness/inertia in correct units.
`_S_FORCE` changes force units in ALL consumers of `telica_loader` (training, closed-loop
eval, diagnostics); results before/after D-112 are not comparable. Run hypothesis for the
next launch: with datasheet-anchored init + force scale, the windowed validation floor drops
and recovered parameters stay physical (masses no longer trainable targets of the scale
error). If Kamtin's unit answer contradicts `_S_FORCE`, replace the HEURISTIC with the
documented constant and rerun.

### [D-111] Orthogonal-projection penalty states are DATA-DERIVED (P^-T y + FD), not FP-simulated; the paper's simulated-states fallback is rejected on measurement
**Date**: 2026-07-12
**What**: The penalty point set for the Gyorok orthogonal-projection regularizer
(`gantry_dynamic/orth_penalty.py`) evaluates the regressor at states reconstructed
from measured outputs (q = P^-T y exact static inversion, qdot by finite
differencing -- the data.py construction), NOT at states from an FP rollout at
theta_bar. Cache key carries states='data' so pre-revision rollout caches can
never load.
**Why (measured)**: The paper's fallback for "no full-state measurement" (GYOROK
p. 7: forward-simulate the FP model) silently assumes the rollout stays near the
data manifold. On the gantry's marginally stable K=0 axes over 48k-sample records
at a 10%-detuned theta_bar this fails: step7b measured worst-case
negation-signature leakage 0.164 and subspace rotation 56.7 deg. The ablation
step7c (same detuned linearization point, truth-manifold states) isolates the
contributors: state drift dominates; with correct states the theta_bar-only
leakage is 3.8e-3..1.7e-2 (matching the pre-stated curvature prediction band)
and rotation 11.1 deg. Data-derived states eliminate the dominant contributor
while staying truth-free, and realize the paper's PRIMARY setting (Sect. 3
full-state measurement; their code x_meas=True builds the basis from measured
states) -- our y is not x, but x is a known static kinematic function of y,
a case the paper's binary framing does not name.
**Constrains**: Remark-2-style per-epoch basis recompute stays a documented
escalation only (trigger: Stage C shows theta drifting far enough that the
measured ~1.7e-2 leakage grows); precompute-once economics retained (one ~6 min
Jacobian pass, cached). Real-data phase must rerun the step7c-style check at the
actual noise level (FD velocities amplify noise) and may need central
differences/smoothing. Worst-case caveat on record: a drift aligned exactly with
the worst principal direction would leak ~sin(11 deg) ~ 0.19; random-direction
floors say such alignment is thin.

### [D-110] Extended R5 literature rounds (Directions 10-11) CONCLUDED: the gap holds at primary-read depth; the R5 evidence file is complete; next gate is the supervisor checkpoint, not more search
**Date**: 2026-07-12
**What**: Two post-D-108 targeted search rounds are concluded and documented in `docs/ml-for-control-search-sweep.md`
(Direction 10: corrupted-scheduling/unmeasurable-premise/estimated-parameters; Direction 11: broadened
non-TU/e search -- dissertations, NASA/aerospace qLPV, gain-scheduling "hidden coupling", scheduling tubes).
Primary-read this round: Piga-Cox-Toth-Laurain Automatica 2015 (flagship, author PDF); Ichalal et al. MED 2012;
Verhoek LPV-SUBNET self- vs external scheduling; Schuet et al. NASA AIAA 2021 GP-qLPV; Cox PhD thesis
(scheduling-noise sections + future work); Hanema CDC 2016 (full) + Automatica 2017 (targeted). PDFs archived:
`literature/corrupted-scheduling/`, `literature/theses-lpv-lineage/`, `literature/aerospace-qlpv/`.
**Outcome (the R5 evidence file)**:
1. **Gap CONFIRMED at primary-read depth, now from inside the supervisors' lineage**: Cox thesis §11.3.1
   delimits the corrupted-scheduling machinery to white/independent/identification-time scheduling noise and
   declares colored/correlated cases OPEN; our R5 corruption (deterministic, growing, self-generated, at
   inference) is the extreme of the declared-open cases.
2. **Three independent communities converge on the same analysis pattern** -- "bounded scheduling deviation =>
   bounded detune": Cox Ch. 3 bounded-rate-of-variation LMIs; NASA parasitic-term-as-uncertainty (Schuet
   §III.B); Hanema scheduling tubes. **All three obtain the bound from a stability/control premise the
   open-loop free-run lacks** (asymptotic stability / frozen-point closed loop / controlled contractivity).
   Consequence: the R4->R5 detune argument is writable, but ONLY as a FINITE-HORIZON (12 s validation window)
   set-propagation bound; no asymptotic claim is possible on the marginal pole. This mirrors D-107 in the
   literature's own structure: the bounds exist because a controller exists.
3. **Layer-3 vocabulary and precedent secured**: Verhoek LPV-SUBNET defines self- AND external scheduling as
   standard formulations (exogenous Y is not a hack); Hanema's LPV-C/A/O taxonomy frames the decision; the
   tube community's own motivating case is a position-scheduled motion system (quotable).
4. **Layer-2 premise independently supported**: Schuet §II.G ("uncertainty ... depends only on where the data
   is observed, not what data is observed") + the d12 measurement (DC direction loss-neutral on the training
   distribution).
**Why concluded**: every remaining read (Verhoek thesis chapters, Shin non-trim, Rugh-Shamma/Lhachemi hidden
coupling) is depth reserve, gated on decisions only the supervisor can make: (a) is empirical R4 acceptable
as the deliverable; (b) exogenous/de-drifted vs self-scheduled Y (the R5 keystone). Searching further before
those decisions repeats the menu-multiplication failure mode (lessons: interrogate the requirement set).
**Constrains**: no new search rounds until the supervisor checkpoint; hidden-coupling/Lhachemi read fires
ONLY if the decision is "self-scheduling must stay". The next artifact is the supervisor-meeting brief.

### [D-109] DC-visibility probe (d8): forward-only loss-visibility curve on the trained drifted checkpoint; ANN mean measured on the training distribution (windowed passes), not the drifted free-run
**Date**: 2026-07-11
**What**: New diagnostic `scripts/gantry/diagnostics-drift/d8_dc_visibility_horizon.py`. Question it answers:
at what evaluation horizon nf does the drift-driving slow ANN force become VISIBLE to the windowed RMS loss?
Method (forward simulation only, no BPTT, so the nf=4000 566 MB training wall does not apply): (1) run the
trained checkpoint over non-overlapping encoder-re-initialized windows (d7 S1a pattern) at nf=400, capturing
the ANN output (d6 shadow pattern) and computing its per-routed-row time-mean; (2) re-run the same windows at
nf in {400, 1000, 2000, 4000} twice, model as-is vs model with that fixed mean vector subtracted from the ANN
output; (3) the visibility curve is Delta-RMS(nf) = windowed RMS(full) - windowed RMS(debiased), judged
against the across-window standard error. Object: `gantry_drift_last.pth` (made 2026-07-09, AFTER the D-101
lr fix of 2026-07-08; config lr=1.49e-8, nf=1400, 20 epochs, cropped val, X+Theta+Y routing). It is a genuine
trained drifting instance but NOT the 07-11 de-confound config (lr=1e-7, nf=400), whose checkpoint was never
saved (its _best reverted to epoch 0); rerunning `make_drift_checkpoint.py` with LR=1e-7 NF=400 can regenerate
that object if the result proves config-sensitive.
**Why the mean is measured on the WINDOWED (near-truth) passes, not the drifted free-run (d6's choice)**: the
probe asks what the TRAINING loss can see and correct; the loss only ever sees the model on encoder-re-seeded
windows near the true trajectory, so the bias expressed there is the one training could act on. The free-run
mean (d6) additionally contains detune/drift-fed-back contributions that no windowed loss could ever see.
**Ruled out**: (1) answering the visibility question by TRAINING at nf=2000+ (566 MB BPTT wall, hours);
(2) a velocity/acceleration-domain metric (LAST RESORT, gated); (3) judging visibility on the full free-run
RMS (that is the 12 s deliverable metric, already known to degrade; the question is about the WINDOWED loss).
**Constrains**: if Delta-RMS stays below ~2x standard error up to nf=4000, brute-force nf is dead on this
hardware and Layer 2 (data-silent projection) is the primary route; if Delta-RMS is significant at nf<=1000,
a moderate-nf run becomes a live option (cost to be checked with the user before any launch).
**OUTCOME (2026-07-11, same day)**: a THIRD case occurred — Delta is significantly NEGATIVE (the DC-carrying
model fits the windows BETTER: paired -2.0/-2.2 SE at nf=400, same sign through nf=2000). The windowed loss
REWARDS the drift driver at every feasible horizon, so moderate-nf training is refuted by SIGN, not cost.
Follow-up `d9_dc_compensation_shape.py` identified WHAT the DC compensates: the encoder's dY init bias
(+2.7e-4 m/s, present in the UNTRAINED encoder = init-scheme property), re-created at every window by the
re-init training geometry. Full mechanism + caveats: `docs/drift-diagnosis-status.md` §3b. Direction (no
decision entry yet): fix the encoder dY init at the source; Layer 2 stays as insurance.

### [D-108] Literature search CONCLUDED: no published method meets the 5 requirements; gap confirmed by a 2025 authoritative survey
**Date**: 2026-07-11
**What**: Concluded the exhaustive ML-for-control literature search (docs/ml-for-control-search-sweep.md
Directions 1-8; docs/literature-search-conclusion.md). No published method satisfies all FIVE requirements
(knowledge-free, friction-permitting/expressive, marginal-preserving, non-drifting, scheduling-integrity=R5).
This is CONFIRMED not only by our multi-community search (dissipativity/passivity/NI, rollout-stability/
exposure-bias, bias/IV estimation, LPV+ML, hybrid identifiability, symmetry, corrupted-scheduling) but by a
comprehensive 2025 survey: Sivaranjani, Shi, Atanasov, Gupta, Allgower et al., "Control-Oriented System
Identification" (arXiv:2512.06315), which states embedding complex control-relevant properties via
identifiable parameterizations "remains an open challenge" (§4.1) and names property-preserving time-varying/
LPV identification as future work (§7.3), citing our supervisors' group (Verhoek LPV) as the state of the art.
**Why**: The user pursued the literature route to find a matching method. It paid off as a NEGATIVE result:
the gap is genuine and now CITABLE to an authoritative survey -> the contribution (learned, LPV self-scheduled,
marginal-preserving, friction-permitting, non-drifting-EMPIRICAL, R5-scheduling-integrity forward augmentation)
sits in an explicitly-open area. Search saturation reached (multi-angle + survey convergence).
**Ruled out**: (1) Continuing BROAD keyword search — diminishing returns, authoritative endpoint reached.
(2) Claiming any found method solves it — none meets all 5 (structural methods sacrifice R2/R3; expressive
methods give no structural guarantee = the impossibility; none native to free-integrator + drifting-self-
scheduling R5). (3) Presenting the negative as "proven nonexistent" — it is "confirmed open per the 2025
survey", the honest framing for a negative.
**Constrains**: Remaining literature work is TARGETED, not broad: (a) verify+quote the paywalled corrupted-
scheduling flagship (Automatica 2015) for R5; (b) primary-read Verhoek LPV consistency (2204.04060) to
localize the contribution. Then value shifts to BUILDING (D-107 empirical layers: long-horizon conditioning +
data-silent projection + de-drifted/exogenous Y-scheduling) and FRAMING against the survey. The contribution
is the ASSEMBLY + the R4 (empirical no-drift) and R5 (scheduling-integrity) handling, not a new structural
guarantee (impossibility). Full record: docs/literature-search-conclusion.md.

### [D-107] Stay OPEN-LOOP; reject closed-loop (it hides a bad model); the drift must be SOLVED not hidden
**Date**: 2026-07-11
**What**: Direction decision for the X/Y augmentation-drift problem, captured in full in
`docs/open-loop-solution-decision.md`. (1) Closed-loop evaluation/deployment is REJECTED as the solution: the
servo bounds position for any model, so it HIDES a spurious model DC / bad fit (certifies the loop, not the
model). (2) The OPEN-LOOP free-run metric is KEPT because it EXPOSES drift and bad fits (a feature); this is
why velocity-domain and closed-loop (both change/remove the metric so drift stops showing) are demoted. (3)
The drift must be SOLVED (remove the spurious DC from the model), verified by the open-loop metric, not hidden.
(4) Under open-loop + position-domain + full expressivity, requirement 4 (non-drift) can only be EMPIRICAL
(the expressivity-XOR-structural-guarantee impossibility). (5) The sole admissible open-loop path is the
ESTIMATION route: data-silent regularization (= Gyorok orthogonal projection re-aimed at the unexcited
subspace, so in-framework and = the thesis contribution) + horizon conditioning (multiple shooting +
continuity) + re-excitation + grey-box friction. (6) FIRST step = a CLEAN position-domain re-run at the
correct post-D-101 lr with conditioning, because the main "conditioning fails" evidence (Optuna 69399) is
confounded by the lr bug (all trials ran at lr=1e-3).
**Why**: User: closed-loop "will just hide a bad model"; Jan's framework is open-loop; supervisors named
velocity-domain a last resort and their stance on closed-loop is unknown, so closed-loop is not assumed
acceptable. Keeping the open-loop metric is the honest judge of solve-vs-hide.
**Ruled out**: (1) Closed-loop as the solution (hides defects). (2) Velocity/acceleration-domain loss
(LAST RESORT, changes the metric; needs explicit go-ahead). (3) Structural constraints
(dissipativity/net-impulse/contraction/NI) — restrict the class, reject friction or the marginal mode, or
need a closed-loop partner (see `docs/dissipativity-limits.md`, `docs/augmentation-literature-verdict.md`).
**Constrains**: All further drift work stays open-loop, position-domain, in Jan's framework, and must SOLVE
(not hide). Req 4 is accepted as empirical, not structural. The next experiment is the clean lr-corrected
position-domain re-run with conditioning; do not conclude "conditioning fails" from 69399. If that fails, add
the data-silent projection. Full detail + document index: `docs/open-loop-solution-decision.md` and
`docs/drift-diagnosis-status.md` §0.

### [D-106] Marginal-native dissipativity theory EXISTS (cyclo-passivity, EIP, Casimirs) — corrects "semidefinite-storage unworked", but none bounds POSITION; contribution narrows to the learned/forward/LPV realization + criterion-4 coupling
**Date**: 2026-07-10
**What**: Primary-read of two classical marginal-dissipativity papers (user challenge: "there is more theory
on the dissipative method"): **van der Schaft "Cyclo-dissipativity revisited" (arXiv:2003.10143)** and
**Simpson-Porco / Hines-Arcak-Packard Equilibrium-Independent (dissipativity/passivity) (arXiv:1709.06986;
Automatica 2011)**, plus PH-Casimir and shifted/Krasovskii passivity as leads. Documented in
`docs/passivity-augmentation-literature.md` **§H** and `docs/drift-diagnosis-status.md` **§5m
(marginal-native dissipativity subsection + scorecard row D3)**. Findings: (a) **cyclo-passivity is the
INDEFINITE-storage relaxation** — verbatim p.6 "we do not yet require S to be nonnegative or bounded from
below"; Def 3.1 `∮ s dt ≥ 0` for `x(T)=x(0)`; **but Remark 3.4: "only INSTABILITY results can be inferred"**
(indefinite storage → no boundedness/Lyapunov conclusion). (b) **EID/EIP characterizes the continuum-of-
equilibria case** — Def 3.2 requires a nonnegative storage per equilibrium for EVERY `x̄∈EΣ`, with `EΣ=X`
when m=n (every state an equilibrium = free integrator); but stability rests on an incremental condition and
does NOT bound the free coordinate.
**Why**: This CORRECTS a second over-claim (after the nonlinear-NI one, D-104): the docs repeatedly called
the "semidefinite/marginal-preserving dissipativity notion" unworked-out (§5b, §5e, §5j). It is NOT —
cyclo-passivity/EID/Casimirs are mature, classical, citable theory for the marginal/continuum-equilibrium
case. The over-claim was an artifact of keeping the "learned/neural" keyword in prior searches.
**Ruled out**: (1) Continuing to present marginal-storage dissipativity as an unsolved theory gap — false.
(2) Treating cyclo-passivity/EID as a solution to the drift — they PERMIT/CHARACTERIZE the marginal mode
(criterion 3) but explicitly do NOT bound POSITION (criterion 4); cyclo gives "only instability results".
This is the same §5j fact from the classical side: passivity of any flavour bounds velocity/kinetic-energy,
not position on a free integrator.
**Constrains**: The four-requirements verdict is UNCHANGED — still no single published method meets all four;
D remains the only family that can (with the extra criterion-4 layer). The contribution narrows AGAIN and is
better-founded: the marginal-preserving STORAGE RELAXATION (crit 3) = REUSE (EID/cyclo/Casimir, classical) +
the NI free-body theory (D-104); the LEARNED realization + FORWARD augmentation + LPV + POSITION-bound layer
(crit 4, via net-impulse Route B or NI) = the genuine invention. Thesis proofs 2-3 (§5e) should CITE
EID/cyclo for the marginal-storage language rather than claim to invent it. New leads (Casimir, shifted/
Krasovskii, Neural Energy-Casimir 2112.03339, PHAST 2602.17998) are `[verify-at-source]`, not primary-read.

### [D-105] Re-frame the X/Y drift as an IDENTIFIABILITY (unexcited null-direction) problem, not primarily a stability problem — four solution families, not one
**Date**: 2026-07-10
**What**: Re-stated the X/Y free-integrator drift solution-neutrally (docs/drift-diagnosis-status.md **§5m**):
the rigid-body (DC/net-impulse) direction of the learned residual is a NULL/UNEXCITED direction of the
training objective (narrowband zero-mean multisine → no DC information; 0.1 s window → no slow-mode
information), while the free integrator makes the deliverable unboundedly sensitive to exactly that
direction. This is an ill-posed-inverse / "parameter drift along an unexcited direction" problem (the
founding problem of robust adaptive control), NOT fundamentally a dissipativity problem. Decomposed the
solution space into FOUR families: (A) pin the null direction with a prior [Tikhonov-in-null-space,
adaptive-control drift mods σ/e-mod/projection, Bayesian stable-kernel, **our own Györök orthogonal
projection = a null-space regularizer**, Lavretsky-Gibson projection ON DISK]; (A-phys) physically-structured
residual-force prior [latent restoring-force / GP latent-force, Rogers-Friis 2109.10681; switching-GP for
friction 2303.03858]; (B) remove the direction from the hypothesis space [integrator factoring/Tustin-net,
our validated bounded-impulse block, Kuntz-Rawlings integrating-mode-LMI arXiv:2406.03760]; (D)
knowledge-free structural guarantee [dissipativity/passivity/NI — the ONLY family deeply searched in §5-§5L].
**Why**: The prior §5 search was anchored on family D (passivity/NI) and concluded "no method exists" — an
artifact of the narrow keyword. Re-derived from the diagnosis, sim-drift is a Problem-1 (ESTIMATION)
pathology best attacked with estimation tools (A/B/C) FIRST; passivity/NI is Problem-2 (knowledge-free
real-data) INSURANCE, not the primary sim fix. This INVERTS the §5 PRIMARY/SUPPORTING ranking for the sim
phase. Crucially, families A/B are mature literatures with boundedness/consistency PROOFS (σ-mod, projection,
stable-kernel, integrating-LMI, bounded-impulse) — they meet the user's "no random heuristic" bar, which
family-D-only searching had made look unreachable. We also already OWN two family-A tools (Györök projection
+ Lavretsky-Gibson projection) that were mis-filed as interpretability/steering rather than as the
null-direction pin they structurally are.
**Ruled out**: (1) Continuing to treat passivity/NI as the PRIMARY sim-drift fix — mis-scoped; it is
real-data insurance (D-104 sharpened the gap; this scopes it). (2) Concluding "no principled method exists" —
false, it was a keyword artifact. (3) Any tuned-constant heuristic (ε-damping floor, soft mean-penalty) —
already rejected (§5j) and unnecessary given the proven family-A/B options.
**Constrains**: The novel contribution shrinks to the real-data case where the unexcited direction ALSO
carries genuine friction DC (family-A pinning would suppress signal → needs grey-box friction in `f_base` or
a family-D permit-dissipative-DC guarantee) — smaller and better-supported than "invent neural NI." All new
cites (Pillonetto-Ljung PNAS 2023, Rogers-Friis MSSP 2022, Kuntz-Rawlings IEEE TAC 2025, switching-GP
2303.03858, adaptive drift mods) are `[verify-at-source]` leads, not yet primary-read. Does not change the
buildable plan (§5g) or D-103/D-104; it re-scopes which family leads for SIM vs REAL data and surfaces two
already-owned tools.

### [D-104] Passivity-augmentation literature: verified at primary source; the theory-gap is the LEARNED realization, not the NI theory
**Date**: 2026-07-10
**What**: An independent adversarial verification pass (brief `docs/fable-review-brief.md`) opened every
on-disk PDF in `literature/passivity-augmentation/`, extracted full text, and checked every
`[extract-verify]` / `[online-*]` claim in `docs/passivity-augmentation-literature.md` and the gap synthesis
in `docs/drift-diagnosis-status.md` §5e/§5j/§5L. Results recorded as a new section
**`passivity-augmentation-literature.md` §G** (per-item CONFIRMED/CORRECTED/REFUTED with exact page/prop
locations) and a **§5L VERIFICATION ADDENDUM** in the drift doc. The load-bearing claims are CONFIRMED:
DiLaR-PINN's stability theorem (**Prop 3, p.4**) requires an ISS baseline (excludes our non-ISS free
integrator); RENs enforces contraction w.r.t. a **strictly** PD metric `P≻0` with incremental passivity
JOINT with contraction and NO marginal variant; Mabrok 2014 is LTI-only free-body NI; NINODE is a
controller re-imposing a strict DC gain; §5j's O(√T) / bounded-impulse math is correct.
**One over-claim CORRECTED**: "NI theory is LTI-only / nonlinear-NI semidefinite-storage is unworked-out"
is REFUTED — nonlinear NI with positive-**semidefinite** storage and poles at the origin exists analytically
(Shi-Petersen-Vladimirov **arXiv:2011.14610 Def 1**; Ghallab-Petersen **arXiv:2201.00144**). The central gap
claim (no **learned** dissipative **forward** augmentation preserves a pole at the origin with bounded
position) **HOLDS** after on-disk + web red-team, but its rationale is re-scoped.
**Why**: The thesis contribution must be stated as **"bring the existing nonlinear-NI free-body
semidefinite-storage theory into a LEARNED parallel forward augmentation in the LPV-LFR/SUBNET framework"**,
NOT "invent nonlinear-NI free-body theory." This is narrower, defensible, and gives a citable classical
foundation (Shi-Petersen-Vladimirov Def 1) to build the learned block on — de-risking the Phase-T theory
work. It also fixes a real risk: presenting an existing analytical result as our own invention would not
survive a supervisor/examiner check.
**Ruled out**: (1) Leaving the "NI is LTI-only" framing — falsified by the on-disk PDF, would misrepresent
the contribution. (2) Treating the gap as refuted — no learned falsifier found on disk or by web search
(2404.12554 "Learning Stable and Passive NODEs" uses PD storage → attractor; 2309.16032 likewise; nonlinear
NI papers are analytical controllers). (3) Citing the LuGre-PINN "not passive by construction" as a paper
quote — it is an inference ("passive"/"dissipative" appear 0× in 2504.12441).
**Constrains**: All future thesis text and slides must use the re-scoped contribution statement (learned +
forward + LPV realization of an existing nonlinear-NI free-body theory). DiLaR-PINN must be cited as
Long-Solak-Ajoudani (IIT, IFAC 2026). The `[disk]` §A framework quotes (Hoekstra/Drenth/Gyorok) were NOT
re-verified in this pass and remain re-verify-before-thesis. Does not change the buildable plan (§5g Phases
0–6) or D-103; it sharpens the Phase-T theory framing only.

### [D-103] HARD CONSTRAINT: the ANN must route to X and Y (not Theta-only) — the augmented system's coupling cannot be captured without them
**Date**: 2026-07-08
**What**: The augmentation routing MUST include the X and Y axes (full routing
`ann_route_ix=(0,1,2,3,4,5,6,7)` or at least X/Y rows in addition to Theta+absorber).
**Theta-only routing `(1,4,6,7)` (D-068) is NOT an acceptable end state**, even though it trains
without drift. Supervisor/user directive (2026-07-08): "only theta routing is not acceptable the
augmented system has coupling we cant capture it without X and Y."
**Why**: The truth system's added dynamics couple into the X and Y responses; an augmentation that can
only write corrections to Theta and the absorber states cannot represent that coupled X/Y behaviour, so
Theta-only is structurally insufficient regardless of how cleanly it trains. This **overrides** the
earlier reasoning (mine) that "the MSD coupling into X/Y already flows through the baseline M(Y), so the
ANN need not route to X/Y" — that argument is rejected as the design basis: the LEARNED component itself
must have X/Y authority.
**Ruled out**: Reverting to Theta+absorber routing to avoid the K=0 free-integrator drift. It removes the
drift but also removes the ability to capture the coupled X/Y dynamics, which is the whole point of the
augmentation. Trading the requirement away to make training easy is not allowed.
**Constrains**: The K=0 X/Y drift (val sim-RMS diverges on long free-run, best=epoch 0; see the 69374
run) must be solved **with X/Y kept in the routing**, never by dropping them. Admissible directions:
velocity-/acceleration-fit loss (take the integrator out of the error path), nf-curriculum with
per-stage lr (let the loss see and penalise the drift), DC-free/open-loop excitation + a drift guardrail
on the free-integrator channels. lr tuning alone is NOT a fix (D-101/D-102 runs: no lr gives
learning-without-drift on free integrators). Do not propose or default to Theta-only in any plan,
diagnostic, or search. Supersedes the D-068 exclusion of X/Y as a permanent choice: D-068's routing
remains only a controlled diagnostic baseline, not the deliverable.

### [D-102] nf-window probe reports train AND val (physical meters), prints per epoch, and is checkpoint-safe (extends D-095)
**Date**: 2026-07-08
**What**: Rewrote the D-095 nf-window val probe in `gantry_dynamic/training.py` to (1) also compute a
**train** nf-window RMS (one extra `n_step_error(mode='RMS')` per epoch on `data.train_list[0]`),
(2) **print** `    [nf-probe] train nf-RMS=… val nf-RMS=… [m] (@nf=…)` each epoch, gated by a new
`RunConfig.nf_probe_print` toggle (default True, visible in `gantry_interconnect_dynamic.py`'s CFG),
and (3) fix a latent pickle crash by making the probe a module-level `_NfProbe` class with
`__reduce__ -> _restore_noop_cve` instead of a local closure. `train_model_with_diagnostics` prints a
train nf-RMS summary line and returns `loss_train_nf` alongside `loss_val_nf`.
**Why**: The user wanted the per-epoch train/val nf-RMS (physical meters, same nf horizon as training)
visible during real gantry runs, matching what the diagnostics print — so generalization gap (train
low / val high) vs long-rollout drift (both bounded, sim-RMS grows) is readable live, not only in the
end-of-run figure. The train quantity is NOT redundant with deepSI's `sqrt loss`: that is normalized,
per-batch, sqrt-of-mean-MSE, whereas the probe is physical meters, full non-overlapping pass, directly
comparable to the val nf-RMS and the sim-RMS selector. The pickle fix is mandatory in production: the
old D-095 closure would crash `checkpoint_save_system` (`torch.save(self.__dict__)` -> "Can't pickle
local object '_install_nf_val_probe.<locals>.wrapped'") the moment a checkpoint saves on val improvement
— a latent bug that only stayed hidden because no probed real run happened to improve+checkpoint. A
module-level class with `__reduce__` returning `(_restore_noop_cve, ())` serialises the transient probe
back to the `_noop_cve` callable (verified: pickles OK, restores a callable not None); the probe is
re-installed each fit and restored in the `finally`, so the no-op placeholder is never actually called.
**Ruled out**: (a) Print val nf-RMS only, read `sqrt loss` as "train" — rejected: normalized/per-batch,
not comparable to val meters (user explicitly asked for train nf-RMS on the same footing). (b) Keep the
closure and no-op `checkpoint_save_system` like the diagnostics do — the real pipeline needs
checkpointing for its val-sim-RMS model selection, so disabling it is not acceptable. (c)
`functools.partial`/bound-method probe — would pickle the train/val trajectories into every checkpoint,
bloating them; the `__reduce__` no-op keeps checkpoints clean. (d) Compute train nf-RMS over the full
`train_list` — costlier per epoch; `train_list[0]` is a representative single-pass, consistent with the
val probe cost.
**Constrains**: (1) `_install_nf_val_probe` signature is now `(fit_sys, hp, cfg, train_sd, val_sd)`;
callers (`training.py`, `diag_theta_lr_sweep.py`) updated. The two diagnostics that define their own
local `_install_dual_nf_probe` are unaffected. (2) `nf_probe_print` is a runtime-only field — NOT in
`cfg.hp`, so it does not touch the checkpoint/npz hp contract. (3) The probe adds ~one extra windowed
train pass per epoch; negligible vs the full-sim validation (~75% of epoch time), but note it if epoch
timing is analysed.

### [D-101] Pass hp['lr'] into init_model in build_model — the configured learning rate was silently ignored (every gantry run trained at Adam default 1e-3)
**Date**: 2026-07-08
**What**: One-line change in `scripts/gantry/gantry_dynamic/model.py:build_model`:
`fit_sys.init_model(sys_data=..., auto_fit_norm=False)` becomes
`fit_sys.init_model(sys_data=..., auto_fit_norm=False, optimizer_kwargs={'lr': hp['lr']})`.
**Why**: The learning rate knob was disconnected. `build_model` calls `init_model`, which creates
the optimizer (`interconnect.py:425` -> `init_optimizer` -> `Adam(parameters)` with no `lr` ->
**Adam default 1e-3**) and sets `init_model_done=True`. Later `train_model` (model.py:185) calls
`fit(..., optimizer_kwargs={'lr': hp['lr']})`, but `fit` only consumes `optimizer_kwargs` inside the
`if init_model_done==False` branch (interconnect.py:548); the `else` branch just runs
`_check_and_refresh_optimizer_if_needed()` (a CUDA-graph health check, fit_system.py:520) which never
touches lr. So `hp['lr']=1e-4` (config `cfg.lr`) was silently dropped and **every gantry augmentation
run — real pipeline and all historical run-table entries — trained at 1e-3, 10x the intended rate.**
Discovered when a Theta-only lr sweep (`diag_theta_lr_sweep.py`, lr in {1e-5,1e-6,1e-7} then
{1e-10,1e-12,1e-13}) produced **bit-identical** loss curves across all lrs (val sim-RMS
0.006071/0.001105/0.002977 at It 1/2/3 regardless of lr) — proof the lr never reached the optimizer.
Strong candidate cause for the "even Theta blows up after init" instability: the effective step was
10x too large, matching the supervisor's "learning step too high can blow up NN, last 0".
**Ruled out**: (a) `param_groups` lr override after `build_model` in each caller — works but must be
repeated in every entry point and diagnostic, and hides the real fix. (b) Editing Jan's `fit()` to
honor `optimizer_kwargs` when `init_model_done` — touches shared framework code affecting MSD/Bouc-Wen/
all systems, higher blast radius; the wrong assumption actually lives in OUR `build_model` (it creates
the optimizer early, then relies on `fit` to set lr). (c) Not calling `init_model` in `build_model` and
deferring to `fit` — `build_model` must init the nets so the post-build encoder-init x0 capture and
baseline sims work; deferring breaks those. Chosen fix passes lr where the optimizer is actually
created, in our own code, minimal diff.
**Constrains**: (1) All callers (`gantry_interconnect_dynamic.py`, `diag_theta_lr_sweep.py`,
`diag_xy_routing_blowup.py`) set `hp['lr']`/`cfg.lr` before `build_model`, so this single fix repairs
all of them. (2) `hp['lr']` MUST be set before `build_model`; setting it only via `fit`'s
`optimizer_kwargs` remains dead — do not rely on it. (3) **All prior gantry run-table results were at
lr=1e-3, not their stated lr; re-interpret accordingly and re-run any lr-sensitive conclusion.**
(4) The sibling pipelines (`lpv_lfr_baseline/`, `scripts/gantry/real-data-verification/`) likely share
the `init_model`-before-`fit` pattern and the same stranded-lr bug — audit separately before trusting
their lr settings.

### [D-100] Unified config: all parameters in one RunConfig; hp is a derived view (supersedes the D-092 split)
**Date**: 2026-07-08
**What**: `RunConfig` now holds EVERY user-tunable parameter, including the model/training
hyperparameters that D-092 had left in a separate `default_hp(cfg)` dict (`nx_ann`,
`n_nodes_per_layer`, `n_hidden_layers`, `up_sample`, `batch_size`, `lr`, `epochs`,
`nf_seconds`). The entry file `gantry_interconnect_dynamic.py` constructs one object with all
fields visible. `nf` and `na_nb` are derived properties (from `nf_seconds`/`ts_new` and Jan's
`(nx_phys+nx_ann)*2+1` rule) with optional direct overrides `nf_override` / `na_nb_override`
(None = derive). `cfg.hp` is a read-only property returning the legacy dict (exact keys/order),
so the ~67 `hp['...']` call sites, the checkpoint `.npz` meta, the results-npz `config`/`hp`
JSON, and resume are all unchanged. `default_hp(cfg)` remains as a one-line backward-compat
accessor returning `cfg.hp` (used by `diag_xy_routing_blowup.py`); the entry file and
`gantry_optuna.py` now use `cfg.hp` directly.
**Why**: The D-092 split left the setting surface in two places; the user could not set all
parameters from the entry file and found the separate dict messy ("why is this still a separate
dict compared to all the parameters?"). One object, one place, one source of truth.
**Verification**: Stage A re-run bit-exact vs the unchanged legacy copy after the change
(`cfg.hp` byte-identical to old `default_hp` incl. JSON; all model tensors, both RNG streams,
66626 training windows, and first-batch loss hex `0x1.5ddeac0p-21` identical). Behavior-preserving.
**Ruled out**: (1) Plain dict of all params in the entry file (no dataclass) -- loses the frozen
guarantee and derived properties. (2) Keeping two objects both edited in the entry file -- still
two things to reconcile. (3) Making `nf`/`na_nb` plain settable numbers -- loses the physical
`5*tau` default and Jan's-rule default; the override fields cover the "set directly" need.
**Constrains**: `cfg.hp` key set/order remains the frozen checkpoint/npz contract. New tunables
go on `RunConfig` as fields; if they belong in the persisted hp dict, add them to the `hp`
property in the same key position.

### [D-099] Anti-aliasing is a non-issue for the simulated dataset; keep asymmetric resampling (block-mean u, point-sample y/states)
**Date**: 2026-07-08
**What**: Empirically scoped the supervisor's anti-aliasing concern (07-07) with
`scripts/gantry/augmentation-error/diag_downsample_spectra.py` — 20 kHz Welch PSDs of
`y`, `x_logical` (6 ch), `delta_a`, and `u_total` over the worst-case records
(E1_resonance_sweep, E3_aprbs_above, E4_multisine_off, T11_aprbs_100, V1) in BOTH modes.
Metric: fraction of power above the new Nyquist (2 kHz). Result: **every signal is band-limited
far below 2 kHz.** Worst-case `frac_above`: `y`/states = 2.5e-8, and — critically —
`u_total` = ~4e-14 (machine floor). All PSDs roll off steeply and hit a flat numerical/solver
noise floor by ~500 Hz, ~15 decades of headroom below Nyquist. Decision: **do NOT add
`resample_poly`/`decimate` to the simulation pipeline; keep point-sampling `y`/states (exact
here) and keep the block-mean for `u` unchanged.**
**Why**: Point sampling folds only the energy above 2 kHz — which is absent — so "point sampling
is exact" (data.py:101) is now verified to ≤2.5e-8, not assumed. The block-mean `u` fix (D-087)
is retained but its justification is corrected: `u_total` has NO HF content either, so its benefit
is NOT anti-aliasing — it is a DC/area-consistency (impulse-equivalent ZOH reduction) effect that
matters only because the K=0 axes are open-loop integrators that accumulate any systematic
force-mean offset. This also **falsifies the handoff premise** that "the 20 kHz ZOH controller
puts step-harmonic energy far above 2 kHz in u": the controller force is smooth (band-limited
excitation), not a sample-rate square wave.
**Ruled out**: (a) Switching everything to one `resample_poly` — adds filter transients, edge
effects and u/y group-delay bookkeeping for zero measured benefit, and risks reintroducing a u/y
phase mismatch. (b) Replacing block-mean-u with plain `[::D]` — would reintroduce the D-087 open-loop
drift (Y −3.5e-4 m). (c) FIR-decimating the FD-derived velocity states — perturbs the fragile
boundary velocities used for interior-K0 seeding for no benefit.
**Constrains**: The anti-alias machinery belongs to the **real-data (Telica) pipeline**, not the
simulation pipeline. Real logs carry measurement noise, quantization, and true HF resonances that
WILL alias under `[::D]` and MUST be anti-alias filtered (`scipy.signal.resample_poly`, same
zero/linear-phase filter on u and y) before decimation. This conclusion is data-specific to the
noiseless simulation; re-scope with the same diagnostic if the sim excitation band or FS_ORIG changes.

**Measurement noise does NOT change this conclusion (supervisor 2026-07-08).** The noise model is
**measurement noise only, added post-hoc to the output, NOT injected through the closed loop** —
identical to Jan's ECC SNR convention (`msd_ndof_interconnect_dynamic.py:46-47`,
`train_data.y += np.random.normal(0, sigma_n, ...)`). Implementation:
`data.py:150-151` adds `sd.y = sd.y + N(0, sigma_n)` **after decimation, at the 4 kHz working rate,
on the measured output `y` only** (`sigma_n = rms(y)·10^(-SNR/20)`, the acceptance floor, D-078).
Direct supervisor instruction: *"only measurement noise. DONT ADD IN THE CLOSED-LOOP. SHOULD NOT GO
THROUGH THE CLOSED-LOOP. Same as how jan does it with his SNR."*

Consequence for the anti-alias filter (the reason this is recorded here): because the noise is
generated at the 4 kHz working rate and never passes through the loop or the 20 kHz plant, it is
**white only up to the 4 kHz Nyquist by construction — it has no energy above 2 kHz to fold.** So
turning SNR on does NOT reintroduce an aliasing problem and does NOT require `resample_poly` in the
simulation pipeline. **D-099 holds unchanged with measurement noise on.** The ONLY scenario that
would reactivate the anti-alias requirement is noise injected *before* decimation (at 20 kHz, or
in-loop) — which the supervisor has explicitly ruled out for this simulation and which remains a
real-data (Telica) concern only. Future sessions: do not add an anti-alias filter to the sim
pipeline "because we added noise"; the noise is post-decimation output noise and is band-limited by
construction.

### [D-098] Wire the oracle into evaluation tables/error-trace + per-record coverage; cache deferred
**Date**: 2026-07-07
**What**: `evaluation.py` now runs the FP+MSD oracle (D-097) on the val and test records (true-x0,
interior-K0 seed, `hp['up_sample']`, pipeline rate) and shows it as a labeled reference column in
the A-tables and a dotted line on the error-trace plot; oracle NRMS/RMS/trajectory added to the
results npz (conditional keys). `_print_same_init_comparison` generalized to take a list of
reference columns (true-x0 baseline + oracle). Entry `main()` prints per-record augmented NRMS
over BOTH val and test (was test only). Step 6 of `docs/eval-restructure-plan.md`.
**Why**: The oracle bounds the achievable error (best-case augmentation) and is shown on the same
rate/up_sample footing as baseline/augmented (fairness), but as a true-x0 REFERENCE, not a same-init
"+%" target (the encoder cannot observe the absorber). Per-record coverage surfaces where
augmentation helps/hurts across the operating range, not just on V1/E1.
**Ruled out / DEFERRED -- the reference cache**: The plan's shared trajectory cache (fingerprint-keyed,
append-only) for the training-independent references is DEFERRED. Rationale: D-089 moved all baseline
sims to AFTER training, so there is no longer a pre-training wait; the true-x0 baseline + oracle sims
total ~10-30 s and run once post-training. Caching would save that only on repeat runs of an identical
config, at the cost of fingerprint-correctness / stale-cache risk (the exact class flagged as
dangerous for a fair comparison). Low value now, non-trivial risk -> not implemented; revisit only if
per-run eval time becomes a real bottleneck.
**Constrains**: Oracle failure is caught and reported (never breaks eval). npz gains optional
`nrms_oracle`, `rms_oracle`, `y_hat_oracle`, `nrms_oracle_test`. The error-trace baseline/oracle
lines are now present; the NRMS-summary bar figure remains optional/unimplemented (per-record numbers
print to the log). This completes the eval-restructure plan (Steps 1-6 = D-093..D-098).

### [D-097] Python 8-state FP+MSD oracle model (gantry_dynamic/oracle.py)
**Date**: 2026-07-07
**What**: New `gantry_dynamic/oracle.py`: the FP baseline plus the true hidden absorber, an RK4
port of `Matlab-scripts/Augmentation/gantrySystemExtended.m` (state `[X,Th,Y,da, dX,dTh,dY,vda]`,
nonlinear M(Y,da), logical-coordinate force). Simulates a record open-loop from the true interior
state and returns the stage-coordinate output + delta_a. MSD params from ma_frac=0.10
(project_gantry_msd_params; the mat does not store them): ma=1.01, mh_rigid=9.09, fa=150,
ka=ma*(2pi*fa)^2, ca=2*0.05*sqrt(ka*ma), L0=0.10. Step 5 of `docs/eval-restructure-plan.md`.
**Why**: A best-case "augmentation target" reference: how well the FP + true absorber reproduces
the data. Makes "augmented sitting on baseline" read as "ANN did nothing" and bounds the achievable
error. Verified before wiring in (`scripts/gantry/augmentation-error/diag_oracle_vs_data.py`):
native 20 kHz isolates model correctness (delta_a ratio 3.4e-5, X 0.02, Y 0.19 -> model is exact);
pipeline-matched 4 kHz/up_sample=2/block-mean-u confirms fairness at run conditions (delta_a 0.5%,
Y RMSE 2e-6 m vs baseline ~2e-4 m, ~100x below baseline). Two D-087-consistent facts baked in:
seed from an interior sample (sample-0 qdot is a one-sided gradient() artifact); up_sample=1 is
already converged at 20 kHz (up_sample=4 identical), residual is the ZOH-force replay limit.
**Ruled out**: Reading the MATLAB plant at run time (no MATLAB dependency in the Python pipeline);
adding ma_frac to RunConfig (kept as an oracle-module constant, documented, single source);
finer up_sample/native rate in the pipeline oracle (fairness: it MUST match cfg.up_sample and
cfg.fs_new like baseline/augmented; only the standalone diagnostic goes finer -- lessons.md).
**Constrains**: Oracle uses `hp['up_sample']` and `cfg` rate, block-mean u, interior-K0 seeding
-- identical footing to the same-init comparison (D-094). Wiring into the tables/error-trace and
the reference cache is D-098 (Step 6). As a true-x0 reference it is a labeled row, not a same-init
"+%" target (the encoder cannot observe the absorber states).

### [D-096] Diagnostic plots: dotted nf-RMS on the loss plot, error-trace, error-spectrum, plots/ subtree
**Date**: 2026-07-07
**What**: `evaluation.py:_make_plots` now (1) routes all figures into a per-run `plots/` subtree
(`plots/val/` for record-specific ones); (2) adds the **val nf-window RMS as a dotted line** on the
loss convergence plot next to the solid sim-RMS selector and dashed train loss (y-axis relabeled
RMS [m]; the two val curves are the same deepSI physical-meter unit, D-095); (3) adds an
**error-vs-time** plot (residual `y_model - y_data` per axis, augmented encoder-init and true-x0
init) that reveals sub-mm drift/absorber structure the overlay hides; (4) adds a **Y error
spectrum** marking the 130-180 Hz absorber band and ~157 Hz resonance. Step 4 of
`docs/eval-restructure-plan.md`.
**Why**: The existing overlay hides a 4e-4 m residual on a 0.24 m axis; the error trace makes it
visible (ramp=drift, oscillation=absorber). The Y spectrum is direct absorber evidence: if
augmentation removes the ~157 Hz peak, the ANN learned it (with the ANN at zero it is fully
present). The dotted nf-RMS answers "good on the training horizon while full-traj rises?"
**Ruled out**: Separate metric-over-epochs figure (folded into the loss plot); baseline/oracle
lines on the error trace and the val+test NRMS-summary bars (deferred to Step 6 - they need the
cached baseline trajectory and per-record coverage sims not yet plumbed into `_make_plots`).
**Constrains**: nf-RMS plotting aligns to the tail of `epoch_id_full` (resume-safe). Existing PNG
filenames are unchanged, only relocated to `plots/`. Step 6 adds the baseline/oracle error-trace
lines and the coverage summary.

### [D-095] Per-epoch nf-window RMS diagnostic alongside the sim-RMS selector
**Date**: 2026-07-07
**What**: `training.py` records a second validation curve during training: the nf-window RMS (same
nf as training, encoder re-init per window), alongside the framework's full-traj sim-RMS. deepSI
validates once per epoch via `self.cal_validation_error` (concurrent_val=False), so a temporary
instance wrapper (`_install_nf_val_probe`) piggybacks the extra metric into `fit_sys.Loss_val_nf`
and returns the selector value untouched. Restored after training. Returned via the diag dict for
plotting (Step 4). Step 3 of `docs/eval-restructure-plan.md`.
**Why**: The sim-RMS selector currently picks epoch 0 (training makes full-traj worse). The
nf-window curve measures what training actually optimizes (its 0.1 s horizon), distinguishing
"wrong selector / horizon" (nf-RMS improves while sim-RMS rises) from "not learning" (both rise).
Both metrics are deepSI physical-meter RMS (`'sim-RMS'` -> `System_data.RMS`; nf via
`n_step_error(mode='RMS')`), so they are directly comparable. **Selection and `bestfit` are
untouched** (the wrapper returns the original selector value); this is diagnostic only.
**Ruled out**: Changing the selector to windowed now (deferred until the curves are seen);
epoch-by-epoch fit loop (invasive, risks framework state); probe `stride=cfg.stride`
(~40x sim cost). Chose non-overlapping windows `stride=nf` (~1 sim-pass; the average windowed RMS
is near-invariant to stride, more windows only reduce estimator variance).
**Constrains**: Adds ~one sim-pass to per-epoch validation time (acceptable; diagnostic). Valid
for `concurrent_val=False` (our config); the wrapper would not propagate to concurrent-val remote
workers. On resume, `Loss_val_nf` covers only this call's epochs (tail of `Loss_val`); Step-4
plotting aligns to the tail. `n_step_error` runs under `torch.no_grad()`; failures record NaN and
never break training.

### [D-094] Same-init augmented-vs-baseline reporting + RMS/NRMS + verdict + grouped output
**Date**: 2026-07-07
**What**: `evaluation.py:evaluate_and_save` now compares the augmented model (encoder-init) against
the **encoder-init** baseline (`baseline_encinit_nrms`), not the true-x0 baseline; the true-x0
baseline is kept as a labeled reference column. Every metric prints both **RMS [m]** and
**NRMS [-]**. A **verdict** line is printed first (ANN active? via aug-state RMS; same-init
improvement %). Output is grouped under section headers (A. Model / B. Encoder / C. Augmentation /
D. Training health; B and D headers added in `main()` before `state_recovery_diagnostic` and the
grad-norm block). Step 2 of `docs/eval-restructure-plan.md`.
**Why**: The prior table paired augmented (encoder-init) against the true-x0 baseline (different
init), so its "+77%" was an initialization artifact, not the ANN — provably 100% artifact when the
ANN is at zero (augmented == encoder-init baseline exactly). Same-init pairing isolates the ANN's
actual contribution (currently +0.0%, honest). RMS[m] is the physical/defensible quantity
(compares to the noise floor sigma_n); NRMS enables cross-channel comparison. Reporting-only: no
sims added (both baselines already computed in `main()`), plots and the results npz unchanged
(still receive the true-x0 `baseline_nrms`).
**Ruled out**: Dropping the true-x0 baseline (keeps value as an oracle-init reference); computing a
single "+%" across mixed inits (the artifact being fixed).
**Constrains**: When the ANN starts learning, the headline % reflects the ANN alone; the oracle
column (D-097) and per-record coverage (D-098) extend this same table. Falls back to the true-x0
baseline for the comparison when `baseline_encinit_nrms` is absent (non-linear_map encoder).

### [D-093] Per-run output subfolder + config.json snapshot
**Date**: 2026-07-07
**What**: Entry `main()` writes all run artifacts to `save_dir(cfg)/<run_id>/` instead of
`save_dir(cfg)/`. `save_dir(cfg)` stays the run FAMILY dir (reserved as the shared reference-cache
home, D-098). A `config.json` (`config_json_dict(cfg)` + `hp` + `run_id`) is written at the run
folder root. Step 1 of the eval-restructure plan (`docs/eval-restructure-plan.md`).
**Why**: Runs currently drop model/npz/plots/checkpoint into one shared folder with `run_id` baked
into every filename — hard to browse, archive, or delete a single run. `sdir` already threads into
training (checkpoint_dir), `evaluate_and_save`, `state_recovery_diagnostic`, and the grad-norm save,
so the subfolder is a one-line change; nothing else moves. `RESUME_CHECKPOINT` is a full path, so
resume is unaffected. config.json makes each run self-documenting at a glance.
**Ruled out**: Per-run folder inside filenames only (status quo — cluttered); writing config.json to
the family dir (would be overwritten per run).
**Constrains**: Downstream steps write into the run folder; the shared reference cache (D-098) lives
in the family dir `save_dir(cfg)`, not the run folder. A run that crashes still creates its folder.

### [D-092] Behavior-preserving restructure of gantry_interconnect_dynamic.py into a package
**Date**: 2026-07-07
**What**: `scripts/gantry/gantry_interconnect_dynamic.py` (1231 lines) is restructured into a
package `scripts/gantry/gantry_dynamic/` (config, data, model, baselines, diagnostics,
evaluation, training) plus a thin entry file at the unchanged path holding the run knobs and
`main()` under a `__main__` guard. Config boundary: a frozen `RunConfig` dataclass carries
experiment identity (MODE, SNR, STRIDE, FS_NEW, ENCODER_INIT, ...; serialized to the npz
`config` JSON); `hp` stays a plain dict with exactly the current keys (incl. `up_sample`)
because it is JSON-round-tripped in checkpoints and results npz, and resume of existing
checkpoints must keep working. Module-level globals (~20) become two explicit objects
(`DataBundle`, `Norm`) passed as parameters. Duplications factored: shared encoder-window
builder, shared stepwise open-loop rollout, shared affine-map R2. `evaluate_and_save` splits
into metrics / plots / npz-save internals with identical orchestration order. Checkpoint I/O
extracted from `train_model_with_diagnostics`; formats frozen. Importers
`gantry_optuna.py` and `diag_nf100_fullrouting.py` updated to the new API. The restructure is
strictly behavior-preserving: numerics, RNG consumption order, D-087 data conditioning,
the training call, prints, plot files, and all npz/checkpoint keys are unchanged.
**Why**: The monolith made nothing importable or testable (importing it triggered a full
training run at import time; D-091's preflight had to duplicate `build_model` for exactly
this reason), config was split over ~15 module constants plus DEFAULT_HP with an unclear
boundary, and a ~330-line `evaluate_and_save` mixed four concerns. The user explicitly chose
the restructure and accepts losing diff-comparability with Jan's ECC reference script; this
supersedes the lessons.md "preserve the reference-script skeleton" rule for this file only.
**Verification**: Stage A (mandatory): harness monkeypatches the deepSI `fit` entry to
capture, at the training call, the fit kwargs, normalization constants, full hfn+encoder
state_dicts, np/torch RNG states, and a deterministic first-batch loss; old vs new must match
bit-exactly (`np.array_equal`, no tolerance). Stage B (recommended): end-to-end 1-epoch CPU
run of both versions, comparing all output npz files key-by-key. Harness lives in the session
scratchpad, not the repo.
**Outcome (verified 2026-07-08)**: Stage A passed bit-exactly (all fit kwargs, 4 norm
constants, 27 hfn tensors, 13 encoder tensors, numpy+torch RNG states, 66626 training windows,
first-batch loss to identical float hex). Full-config confirmation on the cluster: job 69124
(old code) vs 69125 (refactored), both 10 epochs / nf=400 on the same node. Every printed
training loss and Val sim-RMS (It 260 -> 2600), bestfit=0.00017, R2_linmap
(delta_a=+0.0060, vdelta_a=+0.1640), and all downstream NRMS/RMS/baseline/state-recovery/
gradient-norm tables were identical. Only differences: job-id in filenames, wall-clock seconds
(20232 vs 20369 s), time-profile percentages (measurement noise), one tqdm/print interleaving
artifact, and a cosmetic path string (old `scripts/gantry/../../data`, new abspath-collapsed
`data` -- same resolved location).
**Ruled out**: (1) Single-file restructure with `main()`: fixes side effects but keeps a
1200-line file (user chose the package). (2) Converting `hp` to a dataclass: breaks the
JSON/npz/checkpoint contract and resume of existing checkpoints. (3) Moving `up_sample` out
of `hp`: same contract reason. (4) Keeping the old file as a legacy sibling in the repo:
git history + scratchpad snapshot suffice.
**Constrains**: The pipeline is now multi-file; cluster syncs must include the whole
`scripts/gantry/gantry_dynamic/` directory. Diagnostic scripts that previously copied
config/normalization blocks "verbatim from gantry_interconnect_dynamic.py" should import
from `gantry_dynamic` instead going forward. npz keys, checkpoint `.pt`/`.npz` layout, and
the `hp` dict keys remain a frozen contract for any future edit. The D-088 pipeline-table
rule applies: entry-point path is unchanged, so CLAUDE.md needs no edit.

### [D-091] WITHDRAWN — Pre-flight gate script for augmentation training runs
**Date**: 2026-07-07
**Status**: Withdrawn same day. The script was written but never run; the user rejected it
on review ("I'm not sure about this preflight script" -> remove). `scripts/gantry/preflight.py`
deleted; the CLAUDE.md run-discipline rule now references D-090 only. The entry below is kept
as the design record in case the idea returns (e.g. before the Aspect 3 beta sweeps).
**What**: New standalone diagnostic `scripts/gantry/preflight.py`, run before committing a
cluster training job. Four checks with PASS/WARN/FAIL verdicts, results printed and saved as
JSON to `simulations/gantry_subnet/diagnostics/`: (1) measurability: baseline FP residual on
V1 (true-x0 open-loop sim) vs the D-078 noise floor sigma_n; (2) gradient routing: one
forward+backward at epoch 0 on a small batch, per-group gradient norms (encoder / hfn),
calibrated against the documented dead-zone incident (ANN grad 1.04e-2 dead vs 2.85e-1
healthy, diag_gradient_routing); (3) encoder-init quality at FS_NEW vs the 20 kHz native
reference (per-channel state NRMS of the untrained reconstructability map); (4) absorber
excitation: delta_a std per training record (delta_a ~ 0 means nothing to learn). Checks 1
and 2 survive the hardware transition (data-derived); checks 3 and 4 are marked
simulation-only (they need ground-truth states / delta_a).
**Why**: Three documented incidents wasted cluster runs on conditions checkable before
launch (C_aug dead zone, 200 Hz encoder-init trap, job 68458). Consolidating the fragments
into one pre-launch gate converts prose checklist items (CLAUDE.md stance,
control-reasoning Section 7) into an executable that a session cannot forget to apply.
**Ruled out**: (1) Importing `build_model` from `gantry_interconnect_dynamic.py`: the
training script executes at module level (data loading + training), so importing it runs it;
a __main__ guard refactor would restructure the experiment file (rejected per the
no-scaffolding lesson). The preflight duplicates the minimal build per the diagnostic
independence lesson (construct the component from scratch). (2) Hard thresholds from
invented numbers: verdicts are calibrated on documented incident values and labeled
HEURISTIC, or expressed relative to the native-rate reference (encoder check).
**Constrains**: Config constants (MODE, FS_NEW, SNR, hp) are duplicated from the training
script header and must be kept in sync manually; the script prints the values it used so a
mismatch is visible. Preflight is advisory: a FAIL does not block anything mechanically.

### [D-090] Hypothesis-per-run discipline for training runs
**Date**: 2026-07-07
**What**: Every training run with a new hypothesis or new config gets a row in the run table
(`docs/gantry-augmentation-problem-log.md`, Section 12) BEFORE launch, stating the hypothesis
the run tests; the outcome is added to the same row after the run. Trivial re-runs (same
hypothesis, same config) do not get rows. Enforced via a one-line Workflow rule in CLAUDE.md
and a convention note at the top of the run table.
**Why**: The run table is the registry of dead hypotheses; when it is stale, sessions
re-derive and re-test failure hypotheses that are already answered. Writing the hypothesis
before launch forces every run to be a falsifiable experiment, and the maintained table
becomes the experimental narrative for the thesis (writing phase W21-23). Near-zero cost.
**Ruled out**: A separate run-log file: the problem log Section 12 table already exists and
is referenced; a second location would split the history.
**Constrains**: Launching a run without a hypothesis row is a process violation; sessions
asked to launch runs must add the row first.

### [D-089] Baseline FP sims moved post-training; untrained-encoder x0 captured pre-training
**Date**: 2026-07-07
**What**: In `gantry_interconnect_dynamic.py`, the four full-record baseline simulations
(`compute_baseline_fp_nrms`: val/test x true-x0/encoder-init) move from before
`train_model_with_diagnostics` to directly after it. Pre-training, only the untrained-encoder
initial-state estimates are captured (`_encoder_init_state`, one no-grad forward per record);
the encoder-init baseline sims consume those captured vectors post-training.
**Why**: The four sims are ~2 min each (~8-10 min before the first epoch), delaying visible
training start on the cluster; nothing in training consumes their results (they feed only
`evaluate_and_save` and the convergence plot). Correctness: `compute_baseline_fp_nrms` builds
its own fresh `Gantry_State_Block` and never touches `fit_sys`; the sims draw no randomness, and
the encoder capture stays at the same pre-training point in the RNG stream — training and all
reported numbers are bit-identical to the previous ordering.
**Ruled out**: Skip-flag / env hook (operational scaffolding in an experiment script,
lessons.md); disk cache with config fingerprint (deferred — only pays off on repeat configs and
the encoder-init cache key is fragile); batching the four sims (optimization, separate concern).
**Constrains**: Log order becomes training -> baselines -> test NRMS -> evaluation. A run that
crashes during training leaves no baseline numbers in its log.

### [D-088] Context system: control-reasoning reference doc + CLAUDE.md identity/stance sections
**Date**: 2026-07-07
**What**: (1) New reference doc `docs/control-reasoning.md`: project identity, three-pipeline
map with signal chains, plan-vs-code status table, expanded 8-item control reasoning checklist,
Lambda-vs-Pi interpretability section (standalone-baseline negation test, the three thesis
extensions to the Gyorok method), identifiable-combination table, diagnosis-order pointer.
(2) CLAUDE.md gains two always-on sections placed after Hard Constraints: "Project Identity"
(thesis one-liner + three-pipeline table) and "Control Engineering Stance" (8-item checklist as
one-liners, closing pointer to the doc). Compressions elsewhere (quote-verification 10 -> 3
lines, ownership table -> 2 lines, workflow subagent-trigger block dropped) keep net size
roughly flat. (3) Key File Map extended with the two key training scripts, the new doc, the
research plan PDF, and the literature folders.
**Why**: CLAUDE.md contained process rules but no domain identity and no control-engineering
stance; every session re-derived the project framing from scattered docs and tended to answer
from generic ML knowledge instead of control/system-identification reasoning. Checklists
transfer to future sessions (and to smaller models) better than prose. Keeping depth in one
on-demand doc (~2.5k tokens when read) instead of always-on context avoids instruction
saturation.
**Ruled out**: (1) Expanded reasoning content directly in CLAUDE.md: saturates always-on
context and degrades compliance with all other rules. (2) Duplicating the problem log's
failure detail in the new doc: drift liability; the log stays the single owner, the doc only
points. (3) Hook-based stance enforcement: stronger mechanism, deferred until file-based
guidance proves insufficient.
**Constrains**: Rule ownership split: behavioral rules and incident history live in
`tasks/lessons.md`; the domain checklist lives in CLAUDE.md as one-liners; expansions live
only in `docs/control-reasoning.md`. Project Identity holds slow-changing facts only; phase
state stays in `tasks/todo.md`. Restructuring a pipeline now requires updating CLAUDE.md's
pipeline table and Section 2 of the doc.

### [D-087] ZOH-consistent input resampling (block mean) + interior-sample true-x0 init
**Date**: 2026-07-07
**What**: Two data-conditioning fixes in `gantry_interconnect_dynamic.py`. (1) Downsampling of
the plant force 20 kHz -> FS_NEW uses the per-interval block mean (`u[:n*D].reshape(n, D, nu).mean(axis=1)`)
instead of point sampling `u[::D]`; outputs and states stay point-sampled (`y[::D]`, exact for
states). (2) All "true x0" open-loop simulations (the x_logical-init model sim in
`evaluate_and_save` and the true-x0 baselines in the main block) start from the interior sample
K0 = cheat_n with state `x_logical[K0]`, instead of sample 0.
**Why**: The slide-21 "open-loop problem" (meeting 07-07) decomposes exactly into these two
artifacts, amplified by the K=0 axes (any low-frequency input/init error integrates into a
permanent offset tau*dv, tau = m/c: X 1.55 s, Y 1.01 s; verified to 3 digits by dv injection).
(a) ~75%: `u_total` is ZOH at 20 kHz (discrete controller), so the exact FS_NEW input is the
mean force per hold interval (impulse equivalence); `u[::D]` leaves a nonzero-mean force error.
V1 baseline-only open-loop offset: Y -3.47e-4 m / X +6.1e-5 m with `[::D]`, -2.8e-9 / -3.0e-8 m
with block mean (and -2.7e-9 / -2.7e-7 m at native D=1). (b) ~25%: `gtd_save_record.m` computes
`qdot_logical` with `gradient()`, one-sided at sample 0; V1 starts at rest yet stored v0 is
[9.5e-6, -6.2e-5, -1.05e-4], contributing tau*dv = -1.06e-4 m on Y. Interior samples carry
central differences at 20 kHz (accurate on noiseless positions). Evidence:
`scripts/gantry/augmentation-error/diag_openloop_x0.py`, `diag_onestep_residual.py`; artifacts
in `simulations/gantry_subnet/diagnostics/` (openloop_x0_V1, onestep_residual_V1,
openloop_x0_V1_20kHz, openloop_x0_V1_4kHz_uavg). Sum of the two contributions matches the
observed offsets within 1% (Y: -1.01e-4 + -3.47e-4 = -4.48e-4 vs -4.46e-4 observed).
**Comparison to Jan's ECC MSD method** (`scripts/ecc_2025/msd_ndof_data_generation_dynamic.py`):
Jan has no resampling step at all — the discrete truth system is simulated at the model rate
(dt=0.02 both), so data and model share one ZOH convention by construction; and he discards the
first multisine period (`train[Ntrain:]`), so no simulation ever starts on a cold-start sample.
This decision restores those two invariants for the gantry pipeline, where the truth is a
continuous Simulink plant logged at 20 kHz.
**Ruled out**: (1) `scipy.signal.decimate` on u — an IIR anti-alias filter distorts an
already-ZOH signal; block mean is exact, not an approximation. (2) Filtering y — states are
point-sampled exactly; filtering would inject phase error. (3) Zeroing v0 at sample 0 —
assumes at-rest records, fails for sweep records; interior-sample init needs no assumption.
(4) Regenerating data with logged Simulink states — valid long-term fix for
`gtd_save_record.m`, but not needed once sims start at K0; defer to next data regeneration.
**Constrains**: All baseline and sim-RMS numbers change (improve); results from runs before
this decision are not comparable. The x_logical-init sim now starts at cheat_n (same instant
as encoder-init — cleaner comparison; its saved `y_hat_xlog` is NaN before cheat_n). The
encoder-init velocity error remains a separate open item (needs the cluster run npz:
`x_enc_phys[cheat_n]` vs mat `x_logical[cheat_n]`).

### [D-086] E1 sinesweep tapered with a fade envelope; delta_a panel added to the plot
**Date**: 2026-07-06
**What**: (1) `make_sinesweep` (E1) now applies a 0.5 s half-cosine fade-in/out to the chirp
amplitude over the active window, instead of switching the 34 N / 130 Hz force on and off
abruptly. (2) `gtd_plot_record` gains a 5th full-width row showing the hidden MSD displacement
delta_a (in micrometres), present whenever the MSD is simulated.
**Why**: The un-tapered chirp slammed the resting system on and off, kicking all modes and
producing large onset/offset transients ("peaks at the start and end of the envelopes") that
buried the actual swept response. The steady-state Y-position response to a ~150 Hz force is only
~micrometres (F/(m*omega^2)), correct physics but uninterpretable in the raw position plot, and
the resonance signature lives in delta_a, which was not plotted. With the taper the response
shows a clean resonance bulge where the sweep crosses ~150-157 Hz; the delta_a panel makes that
bulge directly visible. Sweep rate (5 Hz/s) is slow enough for the Q~10 mode (settling ~0.02 s
vs ~3 s in the resonance band).
**Ruled out**: Leaving the chirp un-tapered (transients dominate, not presentable). Hardcoding a
resonance-crossing marker at 150/157 Hz in the plot (the coupled peak is uncertain; left to the
viewer's eye on the delta_a bulge).
**Constrains**: The fade slightly lowers the active-window RMS below the nominal amp_frac*A_Y
(faded ends carry less energy); pack() still reports the nominal RMS. E1's multisine-row RMS on
the plot reads low anyway because RMS is over the full 12 s record while the sweep fills only the
10 s active window.

### [D-085] Save full 8-state augmented ground truth; vdelta_a by differentiation; 4x3 force plot
**Date**: 2026-07-06
**What**: (1) `gtd_save_record` now saves the full augmented state for encoder pre-training:
`x_logical` (6 baseline states, logical coords) plus `x_aug = [delta_a, vdelta_a]` (the 2 MSD
states), where `vdelta_a = gradient(delta_a, ts)`. Previously only `delta_a` (7 of 8 states).
(2) `gtd_plot_record` splits the forces into three separately-scaled rows (total / feedback /
multisine) in a 4x3 grid (positions + 3 force types), so the ~30 N multisine is visible instead
of buried under the ~300 N feedback.
**Why**: The hidden MSD is second-order, so the true augmented state has 8 components; encoder
pre-training (Donor A/B save true states) needs both delta_a and its velocity. Velocities are
obtained by differentiation to match the reference generators, which pull only positions
(q_aug, delta_a) from the model To Workspace and differentiate all velocities; noiseless 20 kHz
data makes gradient exact. Overlaying total/feedback/multisine on one axis hid the small
multisine on moving records; per-type rows fix it.
**Ruled out**: Routing vdelta_a to a new To Workspace block (the model is ours, in
Matlab-scripts/Augmentation, not read-only, so it is possible, but it would be the only
velocity in the schema coming from the model rather than differentiation, an inconsistency for
no accuracy gain in noiseless sim). Adding a force PSD panel (deferred by user: revisit once the
multisine is not visually noisy).
**Constrains**: Baseline states saved in LOGICAL coordinates (project convention: reference
generators save x_logical + stage-coord y); if the training pipeline expects stage-coord states,
both old and new data would need transforming. amp_rms has mixed units [N, N*m, N] (A_anti is a
torque, D-080), documented in the save header.

### [D-084] A_anti sized as a modest fixed torque capped by the yaw budget, not sized to fill it
**Date**: 2026-07-06
**What**: `gtd_size_anti_amp` now returns `A_anti = min(cfg.A_anti, budget_cap)`, where
`cfg.A_anti = 0.5*A_sym*Lb` (a fixed torque chosen so the anti channel contributes the same
per-rail force RMS as the symmetric channel) and `budget_cap = yaw_budget / yaw_peak_per_unit`.
The 2 mm yaw budget is a CEILING that can only scale A_anti down, never a target to fill.
Reverses the original D-081 implementation, which set `A_anti = yaw_budget / yaw_peak` (fill).
**Why**: Filling the 2 mm budget in the augmentation band (130-180 Hz) demanded kilonewtons of
anti force, because moving real mass 2 mm at ~150 Hz costs force ~ omega^2 (order of magnitude:
2 mm yaw at 150 Hz => ~9000 N*m => ~thousands of N per rail). Observed on T9: FX1/FX2 > 1000 N
while FY ~ 150 N (the mirror-image X-only signature of a dominant anti channel). Worse, the
anti/yaw channel excites the theta mode, not the Y-axis hidden MSD, so the force was both huge
and irrelevant to the augmentation target. The forces still passed `gtd_enforce_limits` because
1-2 kN is within the TELICA hardware ceiling [2000,2000,1420] N; the limit check is "won't break
the machine", not "sensible excitation", which is why activation-based sizing (GATE-2) is needed.
**Ruled out**: Filling the budget (original). Driving anti to zero (kept a modest level for MIMO
identifiability, but it is now cheap and could be zeroed for the augmentation track later).
**Constrains**: A_sym=40, A_Y=30 N remain GATE-2 defaults, still unvalidated by a delta_a
activation diagnostic. The cap essentially never binds at 130-180 Hz (modest torque produces
microns of yaw), so anti force is now comparable to sym force rather than kilonewtons.

### [D-083] Phase 4 Simulink integration: base-workspace contact contained in `gtd_run_simulation`
**Date**: 2026-07-06
**What**: The generator's Simulink call runs through `gtd_run_simulation`, which pushes every
model input to the BASE workspace via `assignin` and launches the run with
`evalin('base','sim(...)')`, then fetches `q_aug`/`delta_a` (MSD) or `q1` (baseline). It runs the
model twice for the MSD case (with and without multisine, informativeness baseline), swapping `mh`
to `mh_rigid` during each run. `gtd_enforce_limits` does the hard limit check + proportional
scale-down on the LINEAR closed loop (lsim), before Simulink; the scaled force is what gets
simulated. `gtd_save_record` writes the spec-1.12 schema. The driver `generate_trajectory_data.m`
is a thin loop; validation records run in the same loop (distinct seeds already give independent
realizations).
**Why**: Simulink resolves block variables from the base workspace, not a calling function's
locals, so a pure function with local inputs would be invisible to `sim()`. Containing all base
contact in one function keeps the rest pure and the driver thin, instead of making the whole
driver a base-workspace monolith.
**Ruled out**: `Simulink.SimulationInput`/`setVariable` (needs every model variable enumerated and
the output-return config confirmed; not verifiable here). Inlining the sim in the driver script
(reproduces the monolith). A pure `gtd_run_simulation` with local variables (invisible to `sim`).
**Constrains**: The base variables the `gantry_additional_state_2025a` model references are
inferred from the working `generate_oscillatory_multisine_data.m`; `push_params` sends a superset.
If `sim` errors "Undefined variable X", add X to `push_params`. Assumed model outputs: `q_aug`,
`delta_a` (MSD); `q1` (baseline). `gtd_check_sim` smoke-tests one record to surface a missing name
before the full run.

### [D-082] Section 3 of Jan writeup: ANN presented as `phi_aug`, name-only (no `W_1/W_2`)
**Date**: 2026-07-06
**What**: Rebuilding Section 3 of `docs/writeup/jan-augmentation-writeup.tex` around the finished
`jan-blockscheme-v2.pdf` figure (components table 3a / figure interconnection 3b /
dynamic-parallel model 3c). Two notation choices: (1) the learning component is written
`phi_aug` to match the figure, not `N_theta` from the outline; (2) its internals are named
only, as a `tanh` feedforward network with output `w in R^4`, zero-initialised so
`phi_aug ~ 0` at start, with NO explicit `W_1/W_2` layer matrices.
**Why**: (1) The figure is locked/done and is the section centrepiece; text must agree with
it, so `phi_aug` wins over `N_theta`. (2) `Static_ANN_Block` builds a `zero_init_feed_forward_nn`
with 2 hidden layers x 64 nodes; the outline's single-hidden-layer sketch
`w = W_2 tanh(W_1[.]+b_1)+b_2` would misstate the depth, which Jan could catch. Lowercase `w`
is kept because it is Jan's genuine LFR interconnection-channel signal (Eq. 4), used verbatim
in the framework code (`blocks.py` `forward(z)->w`, `nz/nw`); capital `W_1,W_2` are not paper
notation and were dropped.
**Ruled out**: `N_theta` symbol (would force a `phi_aug == N_theta` aside or a figure edit);
the `W_1/W_2` single-layer formula (wrong depth, introduces non-paper symbols).
**Constrains**: The top Notation table now declares `phi_aug`, `w`, `psi`. Any future change to
the ANN architecture must keep the "name-only, paper-altitude" presentation unless Jan asks for
internals. Writeup compiles to 4 pages.

### [D-081] Multisine layer: purpose-built `gtd_make_multisine` with IFFT synthesis and yaw-budget A_anti sizing
**Date**: 2026-07-06
**What**: `gtd_make_multisine` generates the injected stage force per record. Design points:
(1) It is self-contained, NOT a refactor of the shared `generate_cached_multisine` (reversing
the Phase-3 outline), because the per-channel constrained crest-factor scoring does not fit that
helper's joint-selection contract, and the old script still depends on it unchanged.
(2) Synthesis is by IFFT (`ifft(X,'symmetric')` with unit-magnitude random-phase in-band bins),
not the explicit cosine sum: at period = record the grid is df = fs/N = 1/12 Hz with ~2388 lines,
so cosine-sum is O(N*F) ~ 5e8/signal (minutes); IFFT is O(N log N).
(3) Crest-factor selection keeps the best of `cfg.n_ms_candidates` (=30) random draws per logical
channel: f_sym/f_Y scored on their own signal (stage force is a uniform scaling), f_anti scored
on the closed-loop yaw response via the SISO transfer `H_yaw = [1 -1 0]*sys_cl*([1;-1;0]/Lb)`
(the same P^{-1} anti column verified in D-080).
(4) `gtd_size_anti_amp` sizes A_anti (a torque, N*m) so the anti-driven peak |X1-X2| equals the
2 mm yaw budget exactly (linear loop). Sym/Y coupling into yaw (M_op off-diagonals ~5%) is left
to the 2 mm margin of the 6 mm budget (spec 1.8) and the hard 6 mm enforced downstream in Phase 4.
(5) Realizations cached per record keyed by seed/band/period/Y_op/n_cand.
**Why**: Period = record and fine df make synthesis the cost bottleneck; IFFT removes it. Scoring
f_anti on the yaw response (not raw CF) is what the spec's "CF on the constrained coordinate"
requires, since the anti channel is yaw-budget-limited. A torque-unit A_anti follows directly
from D-080.
**Ruled out**: Refactoring `generate_cached_multisine` (contract mismatch, shared-helper risk).
Cosine-sum synthesis (too slow at period=record). Sizing A_anti on total yaw including sym/Y
coupling (unnecessary; the budget margin covers it, and the hard limit is enforced in Phase 4).
**Constrains**: A_sym=40, A_Y=30 N remain GATE-2 defaults (unvalidated). Phase 4 `gtd_enforce_limits`
must still check the full stage force and total 6 mm yaw and scale down if needed. The multisine
spans the full 12 s record (including holds); only the E1 sinesweep is confined to the active window.

### [D-080] Logical->stage force transform is P^{-1} (f_anti is a yaw torque), not the naive f_sym +/- f_anti
**Date**: 2026-07-06
**What**: The multisine is designed in logical (generalized) force channels [f_sym, f_anti, f_Y]
and injected into the plant as stage rail forces via `gtd_logical_to_stage`, which applies
**F_stage = P^{-1} f_logical**: F_X1 = 0.5*f_sym + f_anti/Lb, F_X2 = 0.5*f_sym - f_anti/Lb,
F_Y = f_Y. Derived from the plant convention (sys = P'*G*P, q_stage = P'*q_logical) by
virtual-work invariance (force map is the dual of the position map). Verified 5 independent
ways in `gtd_check_transform`: P^{-1} vs analytic inverse; f_sym -> equal rails; f_anti ->
opposite rails scaled 1/Lb; F_stage.q_stage = F_logical.q_logical; and DC consistency with the
actual built plant (injecting through the transform into the stage plant equals injecting into
the logical plant).
**Why**: The spec placeholder (F_X1 = f_sym + f_anti) is wrong two ways: it over-scales the
symmetric force by 2x, and it adds a torque to a force. Logical coordinate 2 is the tilt angle
theta ~ (X1-X2)/Lb, so its conjugate force f_anti is a yaw TORQUE [N*m]; dividing by Lb is what
makes it a rail force [N]. Getting this wrong silently corrupts the entire yaw budget and the
anti-symmetric channel, and a shape-only check would not catch it (lessons rule).
**Ruled out**: The naive shape-based map f_sym +/- f_anti (mis-normalized, dimensionally
invalid). Assuming P' or P instead of P^{-1} for forces (P' is the position map, not the force
map).
**Constrains**: `gtd_size_anti_amp` sizes A_anti in torque units [N*m] and must apply the same
P^{-1} before checking the 2 mm |X1-X2| budget. Any amplitude specified "per logical channel"
(spec 1.7: A_sym, A_Y in N; A_anti in N*m) is pre-transform; force-limit and yaw checks are
post-transform on stage forces.

### [D-079] Trajectory-data generator rewritten as modular `gtd_*` functions with three reference shapes
**Date**: 2026-07-06
**What**: The new gantry trajectory-data generator (spec `docs/trajectory-generation-spec-draft.md`)
is built as a set of single-responsibility functions in `Matlab-scripts/Augmentation/data/`
(`gtd_config`, `gtd_build_records`, `gtd_build_plant`, `gtd_make_reference`, `gtd_validate_ref`,
plus later `gtd_make_multisine`, `gtd_run_simulation`, `gtd_enforce_limits`, `gtd_save_record`,
and a thin `generate_trajectory_data.m` driver), replacing the 830-line monolith
`generate_oscillatory_multisine_data.m`. The 22 records (T1-14, V1-4, E1-4) collapse to THREE
reference shapes: `standstill`, `oscillatory`, `aprbs`. Ladder limits are derived from `cfg.lim`
(training top T11 = 75% of the enforced limits, test E3 = 90%). Each mode writes to its own
top-level folder: `data/gantry/matlab/trajectory/<joint|augmentation>/<m50|baseline>/`.
**Why**: The spec is a redesign (fixed-absolute amplitudes, period=record multisine,
logical-coordinate transform, 22-record table), not a delta on the old script. Separating
concerns makes each piece independently verifiable in MATLAB (the P-transform gate and the
Simulink integration become isolated checkpoints), which matters because the assistant cannot
run MATLAB. Y-sweep and lissajous are the same sinusoidal-sum builder with different parameters,
and E1's sinesweep is a standstill motion with a swept excitation, so six spec "classes" reduce
to three motion shapes with no loss.
**Ruled out**: (1) Minimal edit of the existing monolith, rejected because the amplitude
strategy inverts and the trajectory table/timing are rewritten, so a diff-only adaptation would
be more error-prone than a clean decomposition. (2) One reference builder per spec class (six),
rejected as redundant. (3) Hardcoding the ladder numbers, rejected in favour of deriving them
from `cfg.lim` so the enforced limits are the single source of truth.
**Constrains**: Downstream modules read `cfg` and the `records` struct array; the P force
transform stays a derive-and-verify step (D-... / spec 7.5) before `gtd_make_multisine`.
Interpretations pending user confirmation: APRBS X_anti is active only for T12/T14 (off for
T9-T11); APRBS `Y_op` = midpoint of the record's Y range; V2 uses the T10 (60%) jerkTime.

### [D-077] Residual-force diagnostic: dominant model mismatch is inertial-scale (~2x), not friction
**Date**: 2026-07-05
**What**: New real-data diagnostic `scripts/gantry/real-data-verification/diag_residual_force.py`
computes the generalized force the FP model cannot account for from measured motion and
measured applied force: `f_missing = u_applied - [M(Y) qdd + C qd + K q]`, using the model's
own physics matrices (physics.py) and Savitzky-Golay smoothing differentiation for qd, qdd.
Least-squares decomposition of f_missing over moving samples onto [M qdd, qd, sign(qd)] gives,
consistently across 3 operating points (R^2 = 0.997-0.999):
  inertial-scale s = 1 + a:  X ~ 0.48-0.51,  Y ~ 0.65   (dominant term)
  viscous b:  ~60-100 N/(m/s) on X1 and Y, ~10 on X2
  Coulomb c:  ~27-66 N  (present but secondary; below the 136/98 N static-friction spec)
So the FP model's inertial force M(Y) qdd is ~2x (X) / ~1.5x (Y) larger than the applied
MF30*Kt force needed to produce the observed acceleration. Coordinate frame is validated
(corr between model force and applied force 0.93-0.99).
**Why (interpretation, HYPOTHESIS not yet committed)**: M qdd = force is degenerate between
"applied force under-scaled" and "model inertia too large". A pure global current-unit error
(RMS/peak sqrt(2), or x2) would give the SAME scale on all axes; the axis-dependent s (X~0.5,
Y~0.65) instead points at the MF30->N conversion (Kt = MotorForceConst = 109 N/Arms X, 77.6 Y)
possibly being wrong per axis / per motor topology (X rails have multiple sub-motors). This
would corrupt all open-loop training (model driven by mis-scaled force) and explains why
parameter recovery kept trying to halve the masses (compensating the scale error) and why the
optimum is horizon-dependent. Overturns the prior "dominant residual is friction -> go to
augmentation" framing (superseded pending verification).
**Ruled out**: Friction as the primary gap (it is real but secondary, ~30-65 N vs the ~100%
inertial-scale residual). Accepting the e-1 m open-loop error as purely structural before
checking the force-input units.
**Constrains**: Before any further parameter-recovery training, verify the MF30 -> N force
conversion (Kt units: Arms vs peak vs per-motor; number of sub-motors per axis; any amplifier
factor). If the force scale is wrong, fixing it may remove most of the open-loop error without
friction augmentation. Open question for Kamtin: exact definition/units of MF30 and Kt.
**Update (2026-07-05) -- CONFIRMED by linear identification** (`diag_linear_identification.py`,
globally-optimal linear-in-parameters inverse-dynamics fit on all 11 training ops, no training):
the data determines the identifiable mass lumps at ~HALF nominal, rock-solid consistent across
every operating point independently: m_total = 26.2 kg (nom 53.8, ratio 0.49; per-op 26.1-26.4),
mh = 5.9 kg (nom 10.1, ratio 0.59). Adding Coulomb columns changes the residual 24.1% -> 24.0%
and assigns only 2-4 N (vs 98/136 N spec), so FRICTION IS RULED OUT as the primary gap (this
refines/overturns the residual-diagnostic's earlier friction reading, which came from allowing
only a single scalar on the nominal inertia per axis). Since M*qdd = force, half-mass <=> half-
force: the applied force u = MF30*Kt is ~2x too small. The clean factor of ~2, consistent across
all ops, plus Telica.mat showing each X rail split into L+R sub-axes, points to force being
under-counted by the sub-motor count (true force ~ N_submotors * MF30 * Kt). Verdict: the data
is clean and useful but NOT directly fittable by the physical baseline until the force scale is
fixed. Cheap confirmation available: scale applied force by ~2 (or per-axis sub-motor count) and
re-run the linear ID -- m_total should jump to ~53.8 and the residual drop. Other params
(damping/stiffness) are unidentifiable from this data (cond ~2.5e15; rotation mode barely
excited, X1=X2 commanded together) -- only the mass lumps are determined, and they show the 2x.
**Correction (2026-07-05, user)**: the earlier "force is ~2x too small" framing OVER-COMMITTED.
F = m*a is degenerate: the data determining m_total ~ 26 kg is equally consistent with (a) the
real Telica moving mass genuinely being ~26 kg -- the kamtin-fp-model masses come from main.m,
a simulation of possibly a DIFFERENT/earlier gantry, so they need not match the real hardware
(user: "different system") -- or (b) a ~2x force-input scale/units error. The fit cannot
distinguish them. If (a), the data IS directly fittable and parameter recovery is WORKING (it
found the real mass); the "half" is the correct value, not a bug. Resolve with an EXTERNAL
reference: the real Telica moving-mass spec / mechanical drawing vs the main.m assumed masses,
and the MF30/Kt units definition. Do not treat "recovered != model nominal" as automatically a
data error.

### [D-076] Telica validation selector rewired to windowed loss (same measure as training)
**Date**: 2026-07-05
**What**: `run_telica_param_recovery.py` monkeypatches `tr._full_traj_eval` with
`_windowed_val_eval`: sigma-normalized MSE on teacher-forced windows (length SEGMENT_LEN) of
the held-out VAL_SPECS trajectories, returning (normalized-RMS, entries) with the same call
signature so the scheduler, best-checkpoint selection, and both sync/async eval paths are
untouched. Global sigma computed once from the validation q1.
**Why**: The user specified validation should use the same method as training; the framework
default (full-trajectory OPEN-LOOP RMSE) was kept instead (deviation noted only in D-075). On
real data that OL metric is dominated by friction/force-scale drift on the K=0 double-integrator
axes, which no parameter can reduce, so in run 68775 it rose monotonically from epoch 0,
collapsed the ReduceLROnPlateau LR to 1e-5 by epoch 360, and selected epoch 0 (nominal) as
"best". The windowed metric measures the same short-horizon fit training optimizes, on held-out
operating points, so scheduler and selection now track a quantity that can actually improve.
**Ruled out**: Full-trajectory OL as selector (structural drift makes it monotone-degrading);
metres RMS without sigma normalization (Y channel dominates, ignores X fit, differs from
training's per-channel weighting). Closed-loop metric as selector (deployment-relevant but adds
the controller to every eval; kept for FINAL assessment only).
**Constrains**: Validation numbers are now a dimensionless normalized RMS, not metres (printed
`eval_rmse` column relabeled in intent). Final assessment still reports full-trajectory OL and
closed-loop separately. A windowed-val improvement does NOT imply small full-trajectory or
closed-loop error while the D-077 force-scale/friction gap remains.

### [D-001] Target system is the ASMPT dual-gantry (García-Herreros et al.)
**Date**: 2026-03-16
**What**: The sole target system for this project is the ASMPT dual-gantry stage modeled by García-Herreros et al.
**Why**: This is the industrial use case for the graduation project. All other benchmarks (MSD, Bouc-Wen, Cascaded Tanks) are reference implementations of the augmentation framework only.
**Ruled out**: Using MSD or other benchmarks as the target system.
**Constrains**: All new code, data pipelines, and model structures must be built around the gantry system.

---

### [D-002] MATLAB files in `kamtin-fp-model/` are immutable
**Date**: 2026-03-16
**What**: The MATLAB model files defining the FP model structure are the ground truth and must never be modified.
**Why**: They represent the validated physical model from García-Herreros et al. and are the hard constraint that the Python implementation must conform to.
**Ruled out**: Adapting the MATLAB model to fit the Python code. The direction of adaptation is always MATLAB → Python, never the reverse.
**Constrains**: Any Python state-space implementation must reproduce the structure defined in the MATLAB files exactly.

---

### [D-003] Augmentation structure is parallel dynamic LFR
**Date**: 2026-03-16
**What**: The augmentation architecture is a parallel dynamic structure within the LFR framework.
**Why**: Parallel structure is required for orthogonal projection-based regularization (Gyorok et al.), which prevents the learned component from capturing dynamics already described by the baseline. Dynamic (not static) augmentation is needed because cross-coupling and position-dependent flexible dynamics require additional learned states beyond the baseline.
**Ruled out**: Series interconnection (incompatible with orthogonal projection regularization); static augmentation (cannot capture dynamics requiring additional states).
**Constrains**: The LFR interconnection must be realized as a parallel structure. Regularization implementation follows Gyorok et al.

---

### [D-004] Scheduling variable is payload position Y
**Date**: 2026-03-16
**What**: The LPV scheduling variable is the payload position Y.
**Why**: Y enters the inertia matrix algebraically in the García-Herreros model, making it the natural scheduling variable. Since Y is a system state (not an exogenous signal), the formulation is quasi-LPV. Y is directly available from the physical model and does not need to be identified from data.
**Ruled out**: Data-driven scheduling variable identification (not needed here since Y follows from the physics).
**Constrains**: The LPV discretization must handle Y as a state-dependent scheduling variable. Invertibility of the position-dependent inertia matrix must be verified across the full operational range.

---

### [D-006] Python implementation uses stage coordinates
**Date**: 2026-03-16
**What**: The Python discrete-time state-space model is implemented in stage coordinates: states q = [X1, X2, Y, dX1, dX2, dY], inputs u = [F_X1, F_X2, F_Y], outputs y = [X1, X2, Y].
**Why**: Real experimental gantry data is measured in stage coordinates (X1, X2, Y from encoders; F_X1, F_X2, F_Y from amplifiers). The model must match the data — the model is coordinate-independent, so the data determines the choice. The MATLAB model also discretizes in stage coordinates (`c2d(StageCoordinatesSystem, ts, 'zoh')`), providing a direct reference.
**Ruled out**: Logical coordinates [X, Θ, Y] — the augmentation framework trains on measured data, which is in stage coordinates. Working in logical coordinates would require transforming every data sample and adds no benefit.
**Constrains**: The A, B, C, D matrices passed to the augmentation blocks must be in stage coordinates. Normalization statistics (T_x, T_u, T_y) must also be computed from stage-coordinate data.

---

### [D-009] One file per responsibility — scripts import from gantry_ss.py, not duplicate it
**Date**: 2026-03-17
**What**: Each script in `scripts/gantry/` has a single responsibility. `gantry_ss.py` is the sole definition of the model (physics → discrete A, B, C, D). All other scripts (simulation, validation, augmentation wiring) import `gantry_discrete_ss()` from it rather than redefining the matrices.
**Why**: Avoids parameter duplication — if a physical parameter changes, it changes in one place only. Makes the boundary between "model definition" and "model use" explicit.
**Ruled out**: Copying A, B, C, D into each script — creates silent inconsistencies if parameters are updated.
**Constrains**: Any script that needs the discrete model must import from `gantry_ss.py`. Extensions (LPV variant, different Y) are added as new functions in `gantry_ss.py`, not in the calling scripts.

---

### [D-008] Fixed SISO-only bug in modified_encoder_net; kept local copy over deepSI default
**Date**: 2026-03-16
**What**: Uncommented line 361 in `model_augmentation/fit_systems/interconnect.py` so `self.ny` is set from the `ny` argument instead of hardcoded to `tuple()`.
**Why**: The original code forced `np.prod(self.ny) = 1` regardless of actual ny, making the encoder input `nb·nu + na·1` even for MIMO systems. For the gantry (ny=3) this would silently drop output channels 2 and 3 from encoder history, giving input size 40 instead of the correct 60.
**Verified**: Unit test confirmed SISO (ny=1) input unchanged at 20; MIMO (ny=3) input now 60 (was 40).
**Ruled out**: Replacing `modified_encoder_net` with deepSI's `default_encoder_net` — kept local copy to allow gantry-specific encoder extensions later. The two are now functionally identical.
**Constrains**: Nothing locked in — local copy can still be extended independently of deepSI upstream.

---

### [D-007] Implement fixed baseline first, add trainability in a second step
**Date**: 2026-03-16
**What**: The Python FP model is first implemented as a fixed (non-trainable) baseline using `Linear_State_Block` and `Linear_Output_Block`. Trainability (`Parameterized_Linear_State_Block` / `Parameterized_Linear_Output_Block`) is added only after the fixed baseline is validated end-to-end in the augmentation interconnect.
**Why**: Stepwise approach reduces the number of failure modes at each stage. A fixed baseline is easier to verify (output is deterministic and can be compared directly against the MATLAB `G` matrices). Trainability introduces regularization and gradient flow, which should only be debugged once the structural wiring is confirmed correct.
**Ruled out**: Going straight to parameterized blocks — adds trainable parameters and param_loss complexity before the block shapes, wiring, and normalization are validated.
**Constrains**: Validation milestone required before promoting to parameterized blocks: simulated output from the Python baseline must match the MATLAB `c2d` matrices to numerical tolerance.

---

### [D-005] LFR structure confirmed for the LPV augmentation
**Date**: 2026-03-16 (updated 2026-03-20)
**What**: The augmentation framework will use an LFR structure for the LPV scheduling. This was initially deferred but was confirmed as the right approach by the supervisor in the meeting of 2026-03-20.
**Why**: The supervisor stated: "LFR gives more flexibility. Can always compute a state-space representation if we want to remap. Suggestion: start with LFR structure for scheduling/LPV." The LFR parameterization allows the learned correction to vary with Y in a principled way through the delta-p block (see D-017). Rank of the M matrix across different trajectories should be computed to confirm no rank drop occurs (expected to be fine, but must be verified).
**Ruled out**: Pure state-space augmentation without LFR structure. Deferring LFR indefinitely (supervisor explicitly suggested it as the starting point for the LPV scheduling).
**Constrains**: Step 3 implementation targets the LFR structure for LPV scheduling. A paper on discretizing LFRs must be found and reviewed before implementation (supervisor action item from 2026-03-20 meeting). The CT conversion must be written up first before the LFR structure is implemented (see D-018).

---

### [D-010] LPV baseline and LPV augmentation are separate concerns
**Date**: 2026-03-17
**What**: The LPV extension has two distinct parts that must not be conflated:
  1. **LPV baseline** — the FP model with A(Y[k]), B(Y[k]) recomputed each step from physics. This is what Step 2 builds and validates.
  2. **LPV augmentation** — a data-driven network on top of the baseline that also varies with Y. This is a Step 3+ concern.
**Why**: Jan's original augmentation framework has no LPV support. The `Parameterized_LPV_Affine_Linear_State_Block` found in the codebase is a user-added augmentation component, not a baseline block. Treating it as the LPV baseline would conflate two separate responsibilities.
**Ruled out**: Using `Parameterized_LPV_Affine_Linear_State_Block` as the LPV baseline block — it is trainable, augmentation-side, and uses an affine-in-Y² approximation that does not represent the full physics.
**Constrains**: Step 2 validates the LPV baseline purely in Python (no framework). Step 3 requires a new `LPV_Linear_State_Block` (see D-011).

---

### [D-011] Framework integration of LPV baseline requires a new block type
**Date**: 2026-03-17 (updated 2026-03-22)
**What**: Wiring the LPV baseline into the augmentation interconnect requires a new block, `CT_RK4_State_Block`, that reads Y from the current state at each forward call and integrates the CT ODE using one RK4 step.
**Why**: The existing `Linear_State_Block` stores A and B as fixed attributes set at init, so it cannot update them per step. The LPV baseline needs physics that change every timestep as Y evolves. No existing block in the framework supports this.
**Ruled out**: Reusing `Linear_State_Block` with a single frozen operating point (that is the frozen LTI). Reusing `Parameterized_LPV_Affine_Linear_State_Block` (wrong structure: affine-in-Y², trainable, augmentation-side).
**Constrains**: The block computes A_c(Y), B_c(Y) from physics at each step and applies RK4 with dt=ts (see D-018). The baseline should also be expressed in LFR form for compatibility with Drenth's augmentation procedure (see D-005, updated 2026-03-22). Y is read from state index 2 in stage coordinates (self-scheduled).

**Update 2026-03-22**: Changed from `LPV_Linear_State_Block` calling `gantry_discrete_ss(Y)` (pre-discretized DT) to `CT_RK4_State_Block` integrating the CT ODE with RK4 (per D-018). Additionally, the baseline should be expressed in LFR form per supervisor confirmation (D-005).

---

### [D-012] LPV discretization: frozen ZOH for validation, exact ZOH via matrix_exp for training
**Date**: 2026-03-17 (updated 2026-03-18)
**What**: Two discretization approaches are used, one per use case:
  1. **Validation (Step 2)**: frozen-at-sampling-instant — call `cont2discrete(A_c(Y[k]), ts)` at each step.
  2. **Training loop (Step 3)**: exact ZOH via `torch.linalg.matrix_exp(A_c(Y) * ts)`. Fully torch-differentiable (confirmed by test).

**Theoretical status — quasi-LPV caveat (important)**:
  Tóth (2010) states the ZOH setting is *"only reasonable for the discretization of LPV-SS
  representation with static dependence as dynamic dependence requires a higher-order hold
  approach"* (Section I, page 2).
  Our system is **quasi-LPV with dynamic dependence**: Y = x(3) is a system state, not an
  exogenous signal. Within each sampling interval, Y evolves continuously as the state
  integrates — it is not truly held constant by ZOH. Consequently:
  - The "errorless" property (Tóth Section IV-A: *"The complete method theoretically provides
    errorless discretization in terms of the ZOH setting"*) applies strictly to static
    dependence only.
  - For our system there is a **small but nonzero residual intra-sample error** from the
    within-interval variation of Y.

**Formal requirements from Tóth (Assumptions 1 and 2, page 5–6)**:
  - Assumption 1 (ZOH setting): *"We are given a CT-LPV system S, with CT input signal uc,
    scheduling signal pc, and output signal yc, where uc and pc are generated by an ideal ZOH
    device and yc is sampled in a perfectly synchronized manner with Td > 0 as the sampling
    period or discretization time-step."*
    Satisfied: our 20 kHz discrete control loop holds u_c and p_c (=Y) constant within each
    50 µs sample interval ✓
  - Assumption 2 (Switching effects): *"The switching behavior of the ZOH actuation has no
    effect on the CT plant, i.e. the switching of the signals is assumed to take place smoothly."*
    Tóth notes: *"this assumption is automatically satisfied in most numerical simulations of
    LPV systems, like in the implemented numerical approaches of SIMULINK in MATLAB."*
    Satisfied: Y changes continuously — no discontinuous jumps; our Python numerical simulation
    mirrors the SIMULINK approach Tóth explicitly endorses ✓
  Note: Tóth provides no quantitative bound on dp/dt. The qualitative remark on page 20
  (*"p_c changes smoothly and relatively slowly with respect to the actual dynamics of the
  plant"*) is motivating prose, not a formal condition.
  Closed-loop applicability: *"The presented ZOH setting is also applicable for closed-loop
  controllers in the structure given in Figure 2"* — our closed-loop Python simulation is
  within the scope Tóth explicitly covers.

**Self-scheduling vs external scheduling**:
  Tóth's Assumption 1 requires p[k] to be held by an ideal ZOH device -- it must be
  *measurable* (externally available) at each step k, not predicted from internal state.
  This implies external scheduling: Y[k] is read from the encoder at step k and held for
  that interval.

  Using Y[k] = x_predicted[k][2] from the model's own state (self-scheduling) introduces
  a further approximation on top of the dynamic dependence caveat already accepted above:
  - Dynamic dependence caveat: Y is a state, not an exogenous signal -- ZOH is approximate.
  - Self-scheduling: Y[k] itself is approximate (from predicted state, not measured). If the
    open-loop state drifts, the scheduling variable is wrong, compounding the error.

  External scheduling (Y[k] from measurement) is more consistent with Tóth and is used
  wherever measurements are available:
  - Training loop: Y[k] = x_measured[k][2] from real data (external, consistent with Tóth).
  - Validation against q1: Y[k] = Y_trajectory[k] from the MATLAB reference (external).
  Self-scheduling is reserved for autonomous simulation with no external measurements and
  carries the additional compounding approximation noted above.

**A_c invertibility (Tóth footnote 2)**:
  Tóth writes the complete discretization formula assuming A_c invertible *"for convenience"*
  but footnote 2 states: *"To compute the resulting matrix functions of this discretization
  approach, Ac(p) is not required to be invertible, but if it is, we can write the resulting
  DT description of the state-evolution conveniently as (9a)."*
  Our A_c is singular (rigid body modes → top-left 3×3 block is zero). The naive formula
  B_d = A_c⁻¹(A_d − I)B_c is therefore undefined. The augmented matrix exponential (D-015)
  is the correct general form — directly supported by Tóth's own footnote.

**Practical justification for small residual error**:
  The intra-sample Y variation is bounded by ΔY ≤ 0.100 mm/sample
  (= v_max × ts = 2 m/s × 50 µs; v_max from ETEL datasheet and main.m vmax=2).
  Physical timescale argument: Y traverses its full 700 mm operational range (ETEL datasheet,
  5% margin from 800 mm stroke) at maximum speed in ≥ 350 ms = 5600 samples, while the
  plant's fastest relevant dynamics act on the closed-loop bandwidth timescale
  ~1/(2π×100 Hz) ≈ 1.6 ms (fbw=100, main.m) — a ~220:1 timescale separation. This makes
  the intra-sample Y variation negligible in practice. Rigorous numerical confirmation:
  ‖A(Y+ΔY) − A(Y)‖/‖A(Y)‖ at ΔY = 0.125 mm is verified in Task 2.5.

**RESOLVED: sample rate set to 20 kHz (matching PLTI spec)**:
  AccurET-Oper&Soft-VerV.pdf confirms PLTI = 50 µs (20 kHz), matching the position control
  loop rate. main.m, export_lpv_sim.m, export_lpv_matrices.m, and physics.py all updated
  to fs = 20e3 (T_d = 50 µs). ΔY_max = 2 × 50e-6 = 0.1 mm — strengthens the slowly-varying
  argument relative to the old 0.125 mm/sample at 16 kHz.

**Why**:
  - Validation: `cont2discrete` is exact for the frozen ODE and fast enough for a one-off
    simulation. The residual quasi-LPV error is accepted as small (see above).
  - Training: `torch.linalg.matrix_exp` is a native PyTorch op — autograd traces through it,
    gradients flow back to Y[k]. The rectangular approximation (Option D, O(ts) error) is a
    valid fallback but is strictly inferior — there is no reason to accept approximation error
    when the matrix exponential is differentiable.
**Ruled out**:
  - Polynomial expansion (Option A): A_c(Y) is rational (from M(Y)⁻¹), so no exact polynomial A_d(Y) exists.
  - Linear-affine approximation (Option B): drops dominant Y² term in M[1,1].
  - Grid interpolation (Option C): not natively torch-differentiable.
  - Rectangular approximation (Option D): O(ts) error — valid fallback only. Superseded by Option E.
  - scipy `cont2discrete` in training loop: not inside autograd graph.
**Constrains**: `LPV_Linear_State_Block.forward()` must compute A_c(Y) analytically from M(Y)⁻¹ using tensor ops, then apply `torch.linalg.matrix_exp(A_c(Y) * ts)`. See `docs/lpv-discretization.md` for full rationale and option comparison table.

**Update 2026-03-20 (supervisor meeting)**: For the augmentation training loop (Step 3+), the discretization approach shifts from pre-discretized ZOH to CT model with RK4 integration. The ZOH approach remains valid for Step 2 validation (completed). See D-018, which supersedes the "training loop" part of this decision. Read D-012 as: Steps 1-2 validation used ZOH (done); Step 3+ training loop uses RK4 on the CT model (see D-018).

---

### [D-013] LPV baseline uses LFR form with CT+RK4 integration
**Date**: 2026-03-17 (updated 2026-03-22)
**What**: The LPV baseline must be *available* in LFR form {M^b, Δ^b(Y)} and integrated using RK4 inside a custom `CT_RK4_State_Block`. The `SSE_Interconnect` wiring machinery is used unchanged. Internally, the forward simulation may collapse to evaluating an equivalent CT vector field A_c(Y)x + B_c(Y)u (as Drenth Ch. 2 eq. 2.29 confirms), but the baseline must remain representable in LPV-LFR form for compatibility with Drenth's augmentation framework (Ch. 5 eq. 5.1-5.2).
**Why**: Supervisor confirmed (2026-03-22) that the baseline itself should use the LFR structure. Drenth Ch. 5 eq. 5.1 assumes the baseline is available in LPV-LFR form. Self-scheduled quasi-LPV (Y from state) is supported. The LFR representation of the baseline requires converting A_c(Y) with its rational M(Y)^{-1} entries into LFR form using standard LFT realization methods (Zhou, Doyle & Glover, 1996).
**Ruled out**: Computing A_c(Y) directly without LFR form (originally chosen, but revised per supervisor guidance). New `SSE_Interconnect` subclass (existing class is sufficient).
**Constrains**: The baseline LFR must be realized from the known physics. Normalization is handled by Drenth eq. 5.5: T_x, T_u, T_y scaling applies to all LFR submatrices. The conversion requires choosing η (repetition count in Δ) and verifying LFR well-posedness. One implementation detail remains open: whether runtime code evaluates the explicit LFR loop or the equivalent collapsed CT vector field. See `docs/lpv-lfr-interconnect.md` for the original assessment (partially superseded by this update).

**Update 2026-03-22**: Major revision. Original decision said LFR is NOT required for the baseline. Supervisor confirmed the opposite: use LFR structure for the baseline. Also changed from pre-discretized A_d(Y), B_d(Y) to CT+RK4 (per D-018). Normalization question is answered by Drenth eq. 5.5.

---

### [D-014] gantry_discrete_ss stays numpy; torch version lives in a separate file
**Date**: 2026-03-17
**What**: `gantry_ss.py` / `gantry_discrete_ss()` is not modified to support PyTorch. A separate file `scripts/gantry/gantry_lpv_torch.py` holds a torch-native implementation that mirrors `gantry_discrete_ss` in structure but uses tensor ops and `torch.linalg.matrix_exp` throughout.
**Why**: Two entirely different use cases with different dependencies and contracts:
  - `gantry_discrete_ss`: numpy in, numpy out, scipy `cont2discrete`, validation and MATLAB comparison only. Pure, simple, zero framework dependency.
  - torch version: torch tensor in, torch tensor out, differentiable, lives inside the training loop. Must stay inside the autograd graph.
  Adding a `use_torch=True` flag to `gantry_discrete_ss` would mix two concerns, add a conditional dependency on torch in a validation-only file, and violate D-009 (one file per responsibility).
**Ruled out**: Modifying `gantry_discrete_ss` to support a torch mode via flag — mixes validation and training concerns in one function.
**Constrains**: `gantry_lpv_torch.py` is a full torch reimplementation — NOT a wrapper around `gantry_discrete_ss`. Every value (physical parameters, M(Y), A_c, B_c, P transform, A_d, B_d) is defined as a `torch.tensor` from the start. No numpy intermediates, no conversion. This ensures gradients flow through the entire computation and physical parameters can optionally be made trainable later without refactoring. The only structural change from `gantry_ss.py` is replacing `cont2discrete` with `torch.linalg.matrix_exp` on the 9×9 augmented matrix (see D-015).

---

### [D-015] B_d(Y) must use augmented matrix exponential — naive formula fails
**Date**: 2026-03-17
**What**: Computing B_d(Y) via the naive formula `B_d = A_c⁻¹ · (A_d − I) · B_c` is forbidden. The correct formula uses the augmented matrix exponential:
```
M_aug = [[A_c(Y),  B_c(Y)],    # (n+m) × (n+m) = 9×9 for gantry
         [  0,        0   ]]

[A_d, B_d] = expm(M_aug · ts)[:n, :], split at column n
```

**Mathematical background**:
  The general ZOH formula for B_d (Tóth complete method, always valid) is:

    B_d = [∫₀^{T_d} exp(A_c · τ) dτ] · B_c

  This integral has no simple closed form when A_c is singular.
  When A_c is invertible, the integral simplifies algebraically to:

    ∫₀^{T_d} exp(A_c · τ) dτ  =  A_c⁻¹ · (exp(A_c · T_d) − I)  =  A_c⁻¹ · (A_d − I)

  giving the convenient form:  B_d = A_c⁻¹ · (A_d − I) · B_c   [Tóth eq. 9a]

  Tóth footnote 2: *"To compute the resulting matrix functions of this discretization
  approach, Ac(p) is not required to be invertible, but if it is, we can write the
  resulting DT description of the state-evolution conveniently as (9a)."*

  The augmented matrix exponential (Van Loan 1978) computes the integral numerically
  without any inversion:

    exp([[A_c, B_c], [0, 0]] · T_d)  =  [[A_d, B_d], [0, I]]

  B_d drops out of the top-right block directly. No A_c⁻¹ anywhere.
  This is what scipy cont2discrete(method='zoh') uses internally.

**Why A_c is singular for our system**:
  The gantry A_c has block structure:

    A_c = [[  0,    I  ],
           [-M⁻¹K, -M⁻¹C]]

  The top-left 3×3 block is identically zero. The K matrix has zero rows for X and Y
  (rigid body modes — no spring restoring force in those directions), so det(K) = 0,
  which propagates to det(A_c) = 0. A_c⁻¹ does not exist.
  Note: B_c itself is not the problem — it is well-defined as [0; M⁻¹].
  The singularity is entirely in A_c, and only affects the shortcut for B_d.

**Complexity increase vs invertible case**:
  - Invertible A_c: compute A_d = expm(A_c · ts) [6×6], then B_d algebraically — two steps.
  - Singular A_c: must form 9×9 augmented matrix and compute one expm — A_d and B_d
    obtained together. Cannot be separated. Computationally more expensive but exact.

**Why**: The gantry A_c(Y) is singular — the top-left 3×3 block is all zeros (position states
  have no velocity-independent dynamics; rigid body modes give zero eigenvalues). `A_c⁻¹`
  does not exist, so the naive formula is undefined. The augmented exponential sidesteps the
  singularity and is mathematically identical to what scipy `cont2discrete(method='zoh')`
  does internally. Both scipy and the torch version must use this formula — any discrepancy
  between them is a numerical precision issue only.
**Ruled out**: `B_d = A_c⁻¹ · (A_d − I) · B_c` — undefined for singular A_c. `B_d = ts · B_c` (rectangular fallback) — O(ts) error, only valid as Option D fallback.
**Constrains**: Both `gantry_lpv_torch.py` and any future `LPV_Linear_State_Block` must form the 9×9 augmented matrix before calling `torch.linalg.matrix_exp`. See `docs/lpv-discretization.md` for the full derivation.

---

### [D-016] Step 2 validation is matrix comparison, not trajectory simulation
**Date**: 2026-03-17
**What**: Step 2 validation compares discrete A(Y), B(Y) matrices directly against MATLAB output at 5 operating points (Y = 0.1, 0.2, 0.3, 0.4, 0.5 m). It does not require simulating a full trajectory.
**Why**: A(Y), B(Y) already match MATLAB to 1e-19 at Y=0.3 (Task 1.2). The LPV question is whether the same holds at other Y values. If the matrices match at every Y, the physics is correct — no trajectory needed to confirm that. Trajectory simulation would add complexity (need input data, initial conditions, etc.) without providing additional information about the correctness of the physics parameterization.
**Ruled out**: Running a full closed-loop trajectory simulation at each Y — unnecessary for validating the LPV matrix computation. The trajectory simulation in Step 1 already validated the dynamics at Y=0.3.
**Constrains**: Requires a new MATLAB script `Matlab-scripts/export_lpv_matrices.m` (does not modify immutable files — calls existing functions) that evaluates G at each Y and saves A, B, C, D per operating point to `Matlab-output/lpv_matrices.mat`. Python comparison script `gantry_lpv_validate.py` checks max absolute error < 1e-10 per matrix per Y. Validation sweep: Y = linspace(0.05, 0.75, 50) — confirmed from ETEL Telica datasheet (total Y stroke = 800 mm, 5% margin from hard limits). 5 points is insufficient: M(Y)⁻¹ is rational in Y and could have non-monotone error behaviour between sparse samples. Dense 50-point sweep allows plotting error vs Y to confirm uniformity across the full operational range.

**Important distinction — what matrix comparison proves vs simulation comparison**:
Matrix comparison (Task 2.4) proves implementation correctness only: Python A(Y), B(Y) match
the same physics as MATLAB G(Y). It does NOT prove that the LPV simulation is a better baseline
than the frozen LTI. That requires Export 2 (Task 2.2) on a varying-Y trajectory.

**Correct simulation comparison target: q1, not q (Simscape).**
q1 (gantrySystem.m in Simulink) is a continuous-time quasi-LPV simulation — M(Y) is
re-evaluated each integration step as Y evolves. It uses identical physics to the LPV model
(same M(Y), C, K; no Coriolis, no Coulomb). Comparing LPV vs frozen LTI both against q1
isolates the Y-varying inertia effect cleanly, without Coriolis/Coulomb interference.
q (Simscape) is the secondary target: q1 vs q quantifies the augmentation gap (Coriolis +
Coulomb). The model is quasi-LPV: captures Y-dependent inertia only — Coriolis, centripetal,
and velocity-dependent friction are dropped and must be learned by the augmentation.

---

### [D-017] Both baseline and augmentation use LFR Δ(Y) structure
**Date**: 2026-03-19 (updated 2026-03-22)
**What**: Both the FP LPV baseline and the learned augmentation use the LFR Δ(Y) structure, as required by Drenth Ch. 5 eq. 5.1-5.2. The baseline has its own Δ^b(Y) block derived from the known physics (M(Y)^{-1}). The augmentation has a separate Δ^a(Y) block with trainable parameters. The two Δ blocks are block-diagonal (no cross-coupling in Δ), but the interconnection between baseline and augmentation happens through the combined M matrix (Drenth eq. 5.2, the `ab` and `ba` submatrices).
**Why**: Supervisor confirmed (2026-03-22) that the baseline should use LFR structure. Drenth Ch. 5 eq. 5.1 explicitly assumes the baseline is in LPV-LFR form. The baseline's Δ^b(Y) is fixed (derived from physics, not trained). The augmentation's Δ^a(Y) has trainable parameters. Well-posedness of the combined LFR is guaranteed by Drenth's direct parameterization (D_zw = exp(-N), Theorem 2.5).
**Open questions**:
- Whether parameter refinement of the FP baseline (making mb, mh, etc. trainable) changes the baseline's Δ^b structure during training. To be confirmed with supervisor at April 9 meeting.
- Whether the baseline implementation should live internally in logical coordinates or be similarity-transformed to stage coordinates before coding, given D-006.
- Whether the current latent-variable realization is accepted as the project baseline or treated as an intermediate realization pending a canonical/minimal LFT realization.
**Ruled out**: Original decision that the baseline does not need LFR (revised per supervisor guidance 2026-03-22).
**Constrains**: The baseline LFR realization must be derived from M(Y)^{-1}. A latent-variable realization now exists and is acceptable as a valid candidate baseline unless a stronger canonical/minimal realization requirement is imposed. This determines the baseline's Δ^b structure and the practical η (repetition count). The combined well-posedness (baseline + augmentation) must be ensured.

**Update 2026-03-22**: Major revision. Original decision said baseline does NOT need Δ(Y). Supervisor confirmed the opposite. Both baseline and augmentation now use LFR structure, per Drenth Ch. 5.

---

### [D-018] CT model kept in continuous time; RK4 used for integration at fixed step
**Date**: 2026-03-20
**What**: The gantry FP model is implemented and maintained as a continuous-time (CT) ODE. Simulation and augmentation training both integrate the CT equations using RK4 with a fixed time step equal to the sampling period (ts = 1/fs). The model is not pre-discretized before the integration step in the training loop.
**Why**: Supervisor confirmed in meeting (2026-03-20), quoting directly from notes: "write up the ct conversion. dont do discretization first will get messy." and "use rk4 not euler discretization. better to not precompute." Key reasoning:
  - RK4 with fixed step always takes the same dt, so it responds correctly to the sampling period and is compatible with the discrete control loop.
  - RK4 is a sum of 4 terms (4 evaluations with weighting), strictly more accurate than Euler (1st order) at the same step size.
  - ODE45 uses variable step sizes (cannot enforce a consistent sampling period by default). The ode4 variant forces a fixed step, but that is equivalent to RK4 directly.
  - ZOH pre-discretization is kept only for Steps 1-2 validation (already completed) where exact MATLAB matrix comparison was the goal. It is not used in the augmentation training loop.
  - When using system identification with a CT baseline, the same RK4 approach applies: keep the model in CT, apply RK4 alongside it.
  - ZOH (zero-order hold) holds the input constant within each interval but says nothing about how the ODE is integrated inside the interval. RK4 is the integration method used inside that interval.
**Ruled out**:
  - Euler discretization: O(h) truncation error, inferior accuracy for the same step size. Supervisor confirmed: "use rk4 not euler."
  - ODE45 with variable step: incompatible with a fixed sampling period in a discrete control loop. Acceptable only as the ode4 variant (fixed step), but RK4 achieves the same result directly.
  - Pre-discretizing with ZOH for the training loop: supervisor explicitly said not to pre-compute. Write up CT first, apply RK4 at runtime.
**Constrains**:
  - The CT model equations must be written up in full before integration is applied. This means: coordinate transforms, all physical quantities with dimensions and units, the full state-space ODE in logical and stage coordinates. This write-up is a prerequisite for Step 3.
  - A paper on discretizing LFRs must be found and reviewed (supervisor action item from 2026-03-20). The LFR structure also operates on the CT equations; understanding how LFRs are discretized informs the Step 3 implementation.
  - The torch training loop integrates the CT ODE using RK4 with dt=ts. The `LPV_Linear_State_Block` planned in D-011 is revised: instead of computing and storing A_d(Y), B_d(Y), it computes A_c(Y), B_c(Y) and applies one RK4 step.
  - The LFR structure for LPV augmentation (D-005, confirmed 2026-03-20) also builds on the CT formulation.
  - Rank of the M matrix should be computed across different trajectories to confirm no rank drop occurs across the operational range.

---

### [D-020] Two methods for rational LPV dependency; Method 2 (state-space form) chosen
**Date**: 2026-03-29 (resolved 2026-03-31, Roland Tóth meeting)
**What**: Two methods exist for handling the rational LPV dependency introduced by M(Y)⁻¹. Method 2 is chosen.

**Method 1 — Online resolve (what Roel implemented):**
Keep the full LFR structure live at runtime. G and Δ(Y) remain as separate blocks. During training, the backward pass propagates through the matrix inverse, implemented either by differentiating through the explicit inverse or via fixed-point iteration. Benefits: stays in true LFR form; LTI and parameter-varying blocks remain separated (useful for control design); potentially faster inference. Disadvantage: must deal with the rational symbolic form of M(Y)⁻¹ explicitly; more complex to implement.

**Method 2 — State-space form (chosen):**
Take M(Y)⁻¹ analytically and absorb it into Ac(Y), Bc(Y). Runtime evaluates `ẋ = Ac(Y)x + Bc(Y)u` directly via RK4. Rational dependency on Y is retained (do NOT rewrite to affine). LFR is used for derivation and structural analysis only, not as a live runtime loop. The augmentation block operates on the same collapsed signals; its black box component can remain affine.

**Why Method 2**: Roland confirmed in 2026-03-31 meeting that this is acceptable. The "algebraic loop" concern was a misapplication of the definition: M(Y) being invertible means the system is well-posed and no true algebraic loop exists. Need to stick to the original parameter structure of M(Y) (augmentation can be added on top without changing the baseline structure). Simpler to implement.
**Ruled out**: Method 1 for now. Not blocked, but not needed: the simpler SS form suffices and Method 1 can be revisited if control design or faster inference become priorities.
**Note — third option not pursued (delay)**: ASMPT mentioned a third approach: introduce a unit delay into the scheduling loop to break the algebraic dependency, rather than collapsing it analytically (Method 2) or resolving it online during training (Method 1). Not chosen because Method 2 is simpler and sufficient, but recorded here for completeness.
**Constrains**: Implement `CT_RK4_State_Block` using Ac(Y), Bc(Y) with rational-in-Y entries (from M(Y)⁻¹). Do not rewrite to affine. Verify M(Y) invertibility numerically: compute singular values of M(Y) across the full Y operational range and confirm they remain bounded away from zero. Check that maximum signal values in M(Y) are below 1 (or 1/0.75) to bound remaining concern.

---

### [D-021] Verify M(Y) invertibility numerically across the Y operational range
**Date**: 2026-03-31
**What**: Before relying on M(Y)⁻¹ in the runtime implementation, numerically verify that M(Y) remains invertible across the full operational Y range. Compute singular values of M(Y) for Y swept across [0, 0.7] m. Confirm all singular values stay bounded away from zero. Also check that maximum signal values in M(Y) are below 1 (or 1/0.75) to bound any remaining well-posedness concern.
**Why**: Roland noted this as a concrete verification step. Y range is also relevant for centering the scheduling variable: centering Y (e.g., Y_c = Y - Y_mean) improves numerical conditioning and avoids potential singularities near the boundary of the operational range.
**Ruled out**: Assuming invertibility without verification.
**Constrains**: This is a prerequisite check before implementing `CT_RK4_State_Block`. Script can be a short standalone MATLAB or Python check. Results should confirm M(Y) is positive definite (physical mass matrix) throughout the range.

---

### [D-022] Non-baseline physics go in augmentation, not in baseline
**Date**: 2026-03-31
**What**: Physical effects not present in the García-Herreros first-principles equations must not be added to the baseline model. They belong in the augmentation component and can be parametrized there.
**Why**: Confirmed by Roland in the 2026-03-31 meeting, specifically in response to the ASMPT-raised question about hysteresis. The concrete example: using sign(dY/dt) as an additional scheduling variable to capture hysteresis direction is a good idea, but it goes in the augmentation, not the baseline. Hysteresis is the motivating example that established this rule. The baseline must remain the exact FP model as derived. Adding extra physics to the baseline would conflate the known physics with the learned correction, making it harder to isolate what the augmentation is doing.
**Ruled out**: Extending the baseline state-space equations with additional physical terms (hysteresis, Coriolis, resonance, etc.).
**Constrains**: The baseline is frozen at the García-Herreros equations. Additional dynamics, forces, and scheduling variables (including sign(dY/dt) for hysteresis) are added in the augmentation block only.

---

### [D-023] Training roadmap: validate parameter estimation on synthetic MATLAB data before adding augmentation
**Date**: 2026-03-31
**What**: The training proceeds in two phases before full augmentation:
  1. Generate synthetic data from MATLAB for various Y values and parameter volumes.
  2. Train the baseline model with free (trainable) physical parameters only — no augmentation black box (Jan's parameter update method). Initialize parameters close to the true values. Show that the parameter estimation recovers the correct parameters from MATLAB-generated data.
  Only after this is demonstrated does augmentation (extra states, Coriolis, etc.) get added.
**Why**: Roland specified this phasing in the 2026-03-31 meeting. Validating the parameter update step in isolation (no black box) proves the baseline training pipeline works before adding augmentation complexity. This mirrors Jan's original method.
**Ruled out**: Jumping straight to augmentation training without first showing the baseline parameter estimation works on synthetic data.
**Constrains**: Synthetic data must cover a representative range of Y and other parameter volumes. The parameter initialization must be close enough to the true values for convergence. The "show it works" milestone (baseline parameters converge to MATLAB ground truth) is required before Step 4 (augmentation) begins.

---

### [D-024] Augmentation ordering: resonance first, Coriolis second
**Date**: 2026-03-31 (ASMPT meeting)
**What**: The augmentation is built up in two steps: first catch resonance dynamics, then add Coriolis as a second step. Coriolis is the more complex effect and should not be targeted before resonance is demonstrated to work.
**Why**: ASMPT guidance from the 2026-03-31 meeting. Resonance is the simpler and more immediate correction; Coriolis requires additional states and is a larger modelling step.
**Ruled out**: Adding Coriolis in the first augmentation step.
**Constrains**: The augmentation milestones in D-023 (training roadmap) follow this ordering.

---

### [D-025] Hysteresis: significant effect, sign(dY/dt) scheduling variable in augmentation
**Date**: 2026-03-31 (ASMPT meeting)
**What**: Hysteresis is a significant unmodelled effect in the gantry. The current scheduling structure (Y-only) cannot capture hysteresis direction because that requires the sign of velocity. Proposed approach: add sign(dY/dt) as an additional scheduling variable, or add a simple explicit hysteresis sub-model. Both approaches belong in the augmentation, not the baseline (see D-022). If hysteresis is not addressed at all, the network will absorb it through black-box fitting, which may reduce interpretability.
**Why**: Raised by ASMPT in the 2026-03-31 meeting. Confirmed by Roland as a good idea for the augmentation side.
**Open**: Whether to apply cost function weighting for hysteresis-dominated regions. Whether a dedicated simple hysteresis sub-model is better than the scheduling variable approach.
**Ruled out**: Adding hysteresis handling to the baseline model.
**Constrains**: When designing the augmentation scheduling structure, include sign(dY/dt) as a candidate scheduling variable. Revisit after resonance augmentation is validated (D-024 ordering).

---

### [D-026] Remove G from lfr_forward — replace G-matrix steps with direct physics expressions
**Date**: 2026-04-02

#### What was decided

The `G` argument is removed from `lfr_forward`. Steps 6 and 7 of the forward pass are replaced with direct physics expressions:

**Before (removed):**
```python
def lfr_forward(x, u, Y, G, M0, M1, M2, K, C):
    ...
    # Step 6: state derivative via G matrix  →  (batch, 6)
    xdot = x @ G.Ax.T + w @ G.Bw.T + u @ G.Bu.T

    # Step 7: output  →  (batch, 3)
    y = x @ G.Cy.T
```

**After (implemented):**
```python
def lfr_forward(x, u, Y, M0, M1, M2, K, C):
    ...
    # Step 6: state derivative — direct from physics (no G needed)
    xdot = torch.cat([x[:, 3:], v], dim=-1)   # (batch, 6)

    # Step 7: output — positions in logical coordinates
    y = x[:, :3]   # (batch, 3)
```

The `G` argument is also removed from `rk4_step` in `lfr_simulate.py`, from the `simulate` function signature, and from `LFRBaselineBlock` in `lfr_block.py` (the `self._G` attribute is removed; `rk4_step` no longer needs it).

---

#### Why this is a valid change — mathematical justification

**The physical state equations.** The gantry equation of motion in logical coordinates is:

```
M(Y) q̈ = -K q - C q̇ + u
```

The state is `x = [q; q̇] ∈ R⁶`, with `x[0:3] = q` (positions) and `x[3:6] = q̇` (velocities). The continuous-time state derivative is therefore:

```
ẋ = [q̇; q̈] = [x[3:6];  M(Y)⁻¹ fnet]       (equation 1)
```

where `fnet = -K x[0:3] - C x[3:6] + u` is the net generalized force. After **step 3** of `lfr_forward`, the quantity `v = M(Y)⁻¹ fnet` is already computed via `torch.linalg.solve(M_Y, fnet)`. Equation (1) then gives directly:

```
xdot = cat([x[:, 3:], v], dim=-1)            (equation 2)
```

This is always exactly correct, for any value of Y, because it is derived directly from the physical equations of motion.

**What G.Ax/Bw/Bu encode.** The LFR G-matrix representation expresses the same state equation as:

```
xdot = G.Ax @ x + G.Bw @ w + G.Bu @ u
```

where `w = [v₁; v₂] = [Y·v; Y²·v]` are the LFR latent signals (already computed in steps 4–5). The entries G.Ax, G.Bw, G.Bu are **constant matrices**, constructed by `build_G_matrix()` using `M₀⁻¹` (the mass matrix at a nominal point). The Y-dependence is captured through the latent signals `w`, not through G directly.

This G-matrix expression is algebraically identical to equation (2) — the LFR G matrices were derived precisely to encode the physical state equations in the LFR framework. The identity holds because the LFR structure is exact: the LFR is not a linearization or approximation; it is an exact rewriting of the rational-in-Y equations using the Δ(Y) = Y·I₆ block (verified in `lfr_forward.py` Check 2 against the collapsed form A_c(Y)@x + B_c(Y)@u).

**Why the G-matrix expression is inferior to the direct expression.** Even though the two forms are algebraically equivalent, the G-matrix form has a hidden dependency: it is only correct when G.Ax/Bw/Bu are consistent with the current values of M0/M1/M2/K/C. G is precomputed at import time in `lfr_matrices.py` by calling `build_G_matrix(M0, M1, M2, K, C)`. If M0/M1/M2/K/C are updated during parameter estimation, but G is not rebuilt, the G-matrix expression silently produces incorrect gradients and incorrect dynamics. The direct expression (equation 2) has no such dependency: it is always correct for whatever M0/M1/M2/K/C are passed to `lfr_forward` at that call.

**Why G.Cy = [I₃ | 0₃] is also removed.** The output `y = x @ G.Cy.T` selects the first 3 state components (logical positions). G.Cy is always `[I₃ | 0_{3×3}]` by the gantry output definition (output = position in logical coordinates). This is directly `x[:, :3]`. Unlike G.Ax/Bw/Bu, G.Cy would not become stale during parameter estimation (it does not depend on M0). However, replacing it with `x[:, :3]` is simpler, removes the G dependency entirely, and is more readable.

**Autograd implications.** The gradient path for physical parameters (M0, M1, M2) flows through `torch.linalg.solve(M_Y, fnet)` → `v` → `xdot`. This path exists in both the old and new implementation. The G-matrix form additionally has gradient paths through G.Ax/Bw/Bu entries when G is built dynamically from M0 inside the forward context. These extra paths disappear with the G removal. However, the physically correct gradient path (through the solve) is the one that was always present and is the one required for parameter estimation. The extra G-entry gradient paths in the old implementation were an artifact of redundant parameterization, not a feature.

---

#### What was ruled out

**Option A: Keep G in the signature but always rebuild it inside forward.**
`G = build_G_matrix(M0, M1, M2, K, C)` at each forward call, then use `G.Ax/Bw/Bu`. This adds unnecessary matrix computation at every forward step (linalg.solve inside build_G_matrix) and computes the same result as equation (2) through a much more expensive path. Rejected: unnecessary overhead, no benefit over the direct expression.

**Option B: Keep G and require the caller to always pass a freshly built G.**
Documented as a constraint ("caller must keep G consistent"). This is error-prone: the interface has two representations of the same physics, and nothing prevents them from diverging silently. Rejected: fragile by design, no benefit over the direct expression.

**Option C: Keep G only for documentation/clarity.**
G was never purely documentary — it participates directly in computation and autograd. Keeping a live computational dependency on G for readability reasons is not justified. Rejected.

---

#### What this constrains

- **lfr_forward signature** is now `(x, u, Y, M0, M1, M2, K, C)`. Any call site must be updated.
- **rk4_step and simulate** no longer accept or pass G. All call sites updated accordingly.
- **LFRBaselineBlock** does not store `self._G`. `build_G_matrix` is not called inside `forward()`.
- **G and build_G_matrix** remain in `lfr_matrices.py` — they are still useful for numerical analysis, LFR structure inspection, and offline verification. They are not deleted.
- **SVD-reduced forward pass** (`svd/lfr_svd_forward.py`) must NOT apply this shortcut. In the reduced realization the state and latent vectors are rotated by the SVD transformation matrices; the physical structure (positions first, velocities last) no longer holds, so `cat([x[:,3:], v])` is incorrect for the reduced system. The SVD-reduced forward must retain its G_reduced.Ax/Bw/Bu parameterization.
- **Check F in test_jan_compat.py** (trainable physical parameter gradient test) is simplified: only the solve-path gradient path exists. The distinction between "static G" and "dynamic G" is removed. The updated check verifies that M0.grad is non-None after backward — which is guaranteed by the linalg.solve gradient — and reports the gradient norm.

---

### [D-027] Fix y-output coordinate mismatch in the Interconnect connection matrix
**Date**: 2026-04-02
**What**: The `S_y` selection matrix in `build_baseline_interconnect` and `build_augmented_interconnect` (in `test_jan_compat.py`) was:
```python
S_y = selection_matrix(np.arange(3), 18)    # (3, 18) — selects logical positions
```
This routes `x_next[0:3]` (logical positions [X, Θ, Y]) directly as the Interconnect output `y`. The reference and training data use stage coordinates [X1, X2, Y]. The fix embeds the logical→stage transform into the connection matrix:
```python
S_y = P.numpy() @ selection_matrix(np.arange(3), 18)    # (3, 18) — logical → stage
```
In row-vector convention used throughout the Python code, `y_stage = y_logical @ P` (see `simulate()`: `Y_list.append(y_k @ P)`). For the Interconnect where the connection matrix acts as `y = S_y @ w_block` (column-vector convention), the correct transform is `S_y = P.numpy().T @ selection_matrix(np.arange(3), 18)`.

Wait — the Interconnect uses column-vector convention (w_block is (batch, nw, 1)), so `y = S_y @ w_block` computes (3, 18) @ (18, 1) = (3, 1). To obtain `y_stage = P.T @ y_logical` (column-vector form), `S_y = P.numpy().T @ selection_matrix(np.arange(3), 18)`.

**Why**: The MATLAB reference data (`q3`, simulation outputs) are in stage coordinates [X1, X2, Y]. The `lfr_forward` output `y = x[:, :3]` is in logical coordinates [X, Θ, Y]. The two coordinate systems differ in the X1/X2 vs X/Θ representation — they are related by `y_stage = P.T @ y_logical` (column-vector form). Without the P-transform in S_y, the Interconnect would output logical positions as training targets, causing incorrect loss computation when compared against stage-coordinate reference data.
**Ruled out**: Embedding the P-transform in `lfr_block.py` (adding y-routing logic to the block output, changing nw). The connection matrix is the correct place for coordinate transforms in Jan's framework — the block output format is fixed by the nw=18 contract.
**Constrains**: `build_baseline_interconnect` and `build_augmented_interconnect` in `test_jan_compat.py` apply this fix. Any future Interconnect wiring for the gantry baseline must use `P.numpy().T @ selection_matrix(np.arange(3), 18)` for the y connection matrix, not a plain selection matrix.

---

### [D-028] Add BPTT mode toggle to simulate()
**Date**: 2026-04-03
**What**: `simulate()` in `lfr_simulate.py` gains a `bptt_mode` parameter with three options: `"full"` (default, unchanged behaviour — retains entire graph), `"truncated"` (detach state every `segment_len` steps), and `"checkpoint"` (use `torch.utils.checkpoint` for exact gradients at O(sqrt(N)) memory). `simulate_frozen()` moved from `validate_lfr.py` to `lfr_simulate.py`.
**Why**: The full computation graph across N RK4 steps is O(N) in memory. For realistic training horizons (N > 1000), this becomes impractical. Jan's framework handles this implicitly via `nf`-bounded windows (typical nf=200), but our standalone `simulate()` had no such bound. The three modes give callers explicit control: `"truncated"` matches Jan's nf pattern (cheap, biased gradients); `"checkpoint"` gives exact gradients at ~1.3x compute; `"full"` remains the default for backward compatibility and short horizons.
**Ruled out**: Adjoint method (torchdiffeq) — exact O(1) memory but numerically unstable for stiff systems and adds an external dependency. Hardcoding a single BPTT strategy — different training scenarios benefit from different trade-offs.
**Constrains**: Training scripts should choose `bptt_mode` explicitly based on horizon length and gradient quality requirements. `segment_len` for truncated mode should cover the system's settling time (~200-1000 steps at 20 kHz).

### [D-029] LPV-LFR baseline code cleanup: performance and CUDA readiness
**Date**: 2026-04-05
**What**: Cleaned up the lpv_lfr_baseline package based on a line-by-line code review. Changes: (1) Pre-transform u_seq from stage to logical coords once before the simulate() loop instead of N times inside it. (2) Pre-allocate output tensors in simulate() and simulate_frozen() instead of list+stack. (3) Removed `_rk4_step_for_checkpoint` wrapper (identical to `rk4_step`; checkpoint calls `rk4_step` directly now). (4) Added `Y_override` parameter to `rk4_step` so `simulate_frozen` reuses the same RK4 logic instead of duplicating it. (5) Made lfr_block.py dtype cast conditional (skip when already float64). (6) Fixed CUDA device bug in simulate_frozen (`torch.full` was missing `device=x0.device`). (7) Pre-allocated tensors use `x0.new_empty()` to inherit device and dtype. (8) Trimmed module docstrings in lfr_forward.py and lfr_simulate.py. (9) Fixed test_jan_compat.py S_y construction to avoid unnecessary numpy round-trip.
**Why**: Preparing for GPU training. The original code had N redundant P.T matmuls per trajectory, N+1 tensor object allocations in Python lists, and a device bug that would crash on CUDA.
**Ruled out**: Deleting lfr_matrices.py (still used by svd/). Switching from torch.linalg.solve to Cholesky (negligible difference for 3x3 matrices).
**Constrains**: `rk4_step` now has an optional `Y_override` keyword argument. Callers using positional args are unaffected. `simulate_frozen` is now a thin wrapper around `simulate`-style logic with `Y_override`.

### [D-030] Trainable physical parameter set for ParameterizedLFRBlock
**Date**: 2026-04-06
**What**: 10 trainable scalars, 2 fixed scalars, in `ParameterizedLFRBlock`. Trainable: `kb_sum` (=kb1+kb2), `cg1`, `cg2`, `cy`, `cb_sum` (=cb1+cb2), `mh`, `m1`, `m2`, `mb`, `J_sum` (=Jb+Jh). Fixed buffers: `Lb`, `d`.
**Why**: Identifiability analysis on the matrix structure of M(Y), C, K:
- `kb1`, `kb2` appear only as their sum in K[1,1] → not individually identifiable; train sum.
- `cg1`, `cg2` appear as both sum and difference in C → individually identifiable.
- `cy` appears isolated in C[2,2] → directly identifiable.
- `cb1`, `cb2` appear only as sum in C[1,1] → train sum.
- `mh` is the sole LPV parameter (enters M0, M1, M2) → strongest signal, must train.
- `m1`, `m2` appear individually via M0[0,1]=(m1-m2)*Lb/2 → identifiable.
- `mb` appears only in M0[0,0] sum with m1+m2+mh → weakest signal; train with tight Lambda.
- `Jb`, `Jh` appear only as sum in M0[1,1] → train sum.
- `Lb` appears in M0, C, and the P coordinate transform; changing P corrupts stage↔logical mapping during training → fixed.
- `d` appears only in products mh*d and mh*d² alongside trainable mh → not separately identifiable; fixed.
All 10 trainable scalars are simultaneously trained from the start (same pattern as `Parameterized_MSD_State_Block`). Lambda regularization weights handle the varying identifiability — tighter for `mb` (2% detuning), standard for others (5–10% detuning).
**Ruled out**: Training `Lb` (corrupts P transform), training `d` (unidentifiable alongside mh), training `Jb`/`Jh` individually (only sum is identifiable), phased training (Jan trains all params at once; regularization handles weak identifiability).
**Constrains**: `_build_matrices()` in `lfr_param_block.py` must reconstruct M0, M1, M2, K, C from these 10 scalars plus fixed `Lb`, `d`. Detuning amounts: kb_sum −5%, cg1/cg2/cy/cb_sum −10%, mh/m1/m2/J_sum −5%, mb −2%.

---

### [D-031] Implement ParameterizedLFRBlock in a separate file lfr_param_block.py
**Date**: 2026-04-06
**What**: The trainable-parameter LFR block lives in `lpv_lfr_baseline/lfr_param_block.py`, not in `lfr_block.py`.
**Why**: `lfr_block.py` has a single well-tested responsibility (stateless frozen-parameter wrapper). The parameterized variant adds substantial new logic: scalar parameter management, `_build_matrices()` differentiable reconstruction, and `param_loss()` regularization. Mixing these two concerns would make both files harder to read and test independently. The existing module follows a one-concern-per-file pattern.
**Ruled out**: Extending `lfr_block.py` with a subclass (same file becomes bloated); creating a generic `parameterized_block.py` (too abstract for one use case).
**Constrains**: `lfr_block.py` stays untouched as the frozen baseline reference. `lfr_param_block.py` imports `rk4_step` from `lfr_simulate.py` and scalar constants from `physics.py` as initial values only.

---

### [D-032] Subclass SSE_Interconnect to handle ParameterizedLFRBlock.param_loss()
**Date**: 2026-04-06
**What**: A thin subclass of `SSE_Interconnect` (living in `lpv_lfr_baseline/`) overrides `loss()` to add a generic `hasattr(m, 'param_loss')` sweep over connected blocks. Jan's `model_augmentation/` code is not modified.
**Why**: `SSE_Interconnect.loss()` calls `param_loss()` only on hard-coded `isinstance` checks for its own block types. `model_augmentation/` is read-only (CLAUDE.md). A subclass override is the minimal, non-invasive extension.
**Ruled out**: Editing Jan's `interconnect.py` (violates read-only constraint); monkey-patching at runtime (fragile).
**Constrains**: The subclass must call `super().loss()` minus the block-type sweep, then add its own generic sweep — or replicate the loss structure with the generic check. It lives in `lpv_lfr_baseline/` and is the entry point for all training scripts in this project.

---

### [D-033] Data strategy: Option A (MATLAB) for first experiment, Option B (Python simulate) future
**Date**: 2026-04-06
**What**: The first parameter-recovery experiment uses the existing `Matlab-output/lpv_sim_varying_y.mat` as training data (Option A). Option B — generating fresh synthetic data via Python `simulate()` with a multisine input, controlled noise (SNR), and explicit train/val/test splits — is deferred to a future experiment.
**Why**: The MATLAB trajectory was generated with the true physical parameters and provides the ground-truth output we need to train against. It exercises varying Y (0.3→0.1 m), which is exactly the range where M(Y) variation is observable. Option B is more rigorous and mirrors Jan's experimental design exactly, but requires additional scripting (input design, noise model, data splits) that is not needed to prove the concept.
**Ruled out**: Using frozen-Y data (LPV parameter mh not identifiable without Y variation); skipping Option B entirely (it is the right long-term approach for a rigorous benchmark).
**Constrains**: The training script must load and convert `lpv_sim_varying_y.mat` to deepSI format. When Option B is implemented, the training script should be parameterizable to switch data sources without changing the model structure.

---

### [D-019] Use Drenth thesis for CT LPV-LFR citations; treat IFAC paper as DT companion
**Date**: 2026-03-24
**What**: For any continuous-time LPV-LFR definition, notation, or generic interconnection equations used in the gantry write-up, the primary source is Drenth's thesis (`literature/books/drenth2025_lpv-lfr-thesis.pdf`). The IFAC paper (`literature/lpv-lfr/drenth2025_lpv-lfr-rational.pdf`) is treated as the discrete-time companion paper and cited as such.
**Why**: The two local Drenth sources are not interchangeable. The thesis explicitly gives the LPV-LFR pair `(G, Delta(p))` in continuous time with `x_dot(t)`, `z(t)`, `w(t)`, `y(t)` and the equivalent rational LPV-SS form. The IFAC paper defines the LPV-LFR pair `{M, Delta(p)}` in discrete time. Citing the IFAC paper as if it were the primary CT definition overstates the DT-to-CT adaptation and obscures the notation difference between the two sources.
**Ruled out**: Treating the thesis and IFAC paper as equivalent sources for Section 2-style CT LPV-LFR definitions. Citing IFAC eq. 6-9 as if it were the primary CT source.
**Constrains**: `docs/references.md`, `docs/lfr-structure.md`, and future LaTeX source notes should cite the thesis for CT LPV-LFR definitions. The IFAC paper remains useful for DT LPV-LFR context, rational-dependency motivation, and well-posedness discussion, but should be labeled as the DT companion when referenced.

---

### [D-034] RMSE_baseline for Lambda regularization computed from detuned baseline on MATLAB data
**Date**: 2026-04-06 (updated 2026-04-20)
**What**: Before training begins, compute the per-trajectory RMSE of `ParameterizedLFRBlock` with `params = params_init` (detuned values) on the active MATLAB trajectories. Two quantities are derived from this:

1. `rmse_baseline` — group-balanced RMSE **in metres** (physical units). Used only for reporting and to instantiate the block when the loss is in physical units (not the current training setup).
2. `rmse_baseline_normalized` — the same RMSE expressed **in sigma-normalized units** (dimensionless), computed via `_aggregate_normalized_rmse_baseline()`. This is what is actually passed to `ParameterizedLFRBlock.__init__()` as `RMSE_baseline`.

The distinction matters because the training loss is normalized by sigma (see D-042):
```
mse_loss = mean(((Y_pred - q1) / sigma)²)    # dimensionless, O(1)
```
Lambda must be calibrated in the same unit system as `mse_loss`. Passing the metre-space value would make Lambda ~450× too small, effectively disabling regularization.

Inside the block, Lambda is computed as:
```python
Lambda[i] = RMSE_baseline_normalized / params_init[i]
```
This ensures the regularization cost is comparable to the simulation MSE when parameters have moved enough to reduce the (normalized) prediction error by one `RMSE_baseline_normalized` unit.

**Why**: RMSE_baseline_normalized scales the regularization relative to the simulation loss in the same unit system. Computing it from the actual detuned baseline on actual data gives principled, automatic calibration. Jan's fixed constant (0.2) is only valid because his data is already normalized to O(1) — our raw data is in metres and sigma-normalization must be applied first.
**Ruled out**: Passing `rmse_baseline` (metres) to the block — Lambda would be ~450× too small and regularization would be ineffective. Manual constant without sigma normalization — arbitrary and unit-dependent.
**Constrains**: `train_param_recovery.py` must compute both `rmse_baseline` (for logging) and `rmse_baseline_normalized` (for the block). The block always receives the sigma-normalized value. Both values should be logged in the saved `.pt` file for reproducibility. See D-042 for the sigma normalization itself.

---

### [D-036] OPEN — Augmentation training: state initialisation and mini-batch strategy
**Date**: 2026-04-08
**Status**: Deferred — decide when implementing augmentation training.
**What**: Two coupled design choices must be made when extending from parameter recovery to augmentation training:

**Choice A — State initialisation for segment start states:**

Option 1 (data-derived, current): positions from measured q1, velocities from central finite differences. Cached as `state_traj_n{N}.pt`. Works for parameter recovery because all states are observable (q, q̇ from positions). **Will not generalise to augmentation**: the augmentation block introduces latent states (e.g. hidden flexible modes) that cannot be read from measured positions or computed by finite differences.

Option 2 (encoder, Jan's approach — `model_augmentation/fit_systems/interconnect.py` line 417): `x = self.encoder(uhist, yhist)`. A learned neural network maps a window of past inputs and outputs to the full augmented state. The encoder is trained jointly with the physics parameters. This is the only correct approach when latent states exist.

**Recommendation**: Keep data-derived states for parameter recovery (current code). Switch to an encoder when augmentation is added. The encoder architecture Jan used is `modified_encoder_net` in `interconnect.py` — a `simple_res_net` mapping `[uhist, yhist]` → `x0`.

**Choice B — Segmentation strategy (overlapping vs non-overlapping):**

Current (parameter recovery): non-overlapping segments, stride = segment_len. Batch = n_seg = N // segment_len (e.g. 70). One gradient update per epoch = full-batch GD.

Jan's approach (augmentation): overlapping sliding windows, stride controlled by deepSI data loader (typically stride=1 or small). Many more gradient updates per epoch — effectively mini-batch SGD. More diverse gradient signal; helps generalisation and can escape local minima.

Trade-off: overlapping windows require the encoder to re-estimate state at every window start (batch × encoder forward pass per epoch). Non-overlapping is cheaper but less diverse. For noisy real data with a learned augmentation, mini-batch SGD over overlapping windows is the standard choice (confirmed by Jan's code).

**Recommendation**: For augmentation training, adopt Jan's overlapping strategy with encoder-based state init. The precomputed `state_traj` cache is still useful for the physical (observable) state components as a warm-start or validation reference.

**Ruled out at this stage**: None — decision deferred until augmentation implementation begins.
**Constrains**: Augmentation training script design. Encoder architecture and hyperparameters (nb, na window lengths) must be chosen at that time.

---

### [D-035] Physical parameter positivity enforced via log/exp reparameterization
**Date**: 2026-04-06
**What**: Physical scalars in `ParameterizedLFRBlock` are stored as `self.log_params = nn.Parameter(torch.log(params_init))`. Physical values are recovered as `params = torch.exp(self.log_params).clamp(min=1e-6)` inside `forward()` and `param_loss()`. The clamp is a numerical crash guard only, not an optimization mechanism.
**Why**: If any physical parameter goes zero or negative during training, `M(Y) = M0 + M1*Y + M2*Y²` becomes singular and `torch.linalg.solve` crashes or produces garbage. L2 regularization alone provides no hard guarantee. Log/exp reparameterization maps the unconstrained real line to `(0, ∞)` — the optimizer trains `log_params` freely in ℝ and positivity is guaranteed by construction. Literature survey (GPyTorch, Stan, neural ODE grey-box models, PINN parameter ID papers) confirms log/exp is the dominant choice for positive scalar physical parameters. Initialisation is trivial: `log(params_init)` exactly inverts the exp transform, so training starts at the correct physical values.
**Ruled out**:
- *Softplus*: `params = log(1 + exp(raw))`. Functionally equivalent to log/exp at our parameter magnitudes (all ≥ 1.05 kg) — softplus saturates to identity for large inputs so the two are numerically indistinguishable. Softplus is GPyTorch's default because it prevents overflow during large hyperparameter searches; this concern does not apply here since L2 regularization keeps params near init. Rejected in favour of log/exp for simplicity (no `softplus_inverse` needed at init) and because it is the more standard choice in the system identification literature.
- *Projected gradient / clamping as training strategy*: `params.clamp_(min=1e-6)` after each optimizer step. Creates a discontinuous gradient at the boundary — the optimizer sees a flat landscape and cannot recover. Parameters cluster at the clip value. Widely considered an antipattern (cf. WGAN weight clipping critique). Retained only as a numerical safety net after exp, not as a constraint mechanism.
- *Log-barrier term*: Add `-λ · Σ log(params)` to the loss. Requires scheduling λ toward 0 (interior point method) to be principled; in stochastic gradient training with Adam this scheduling is difficult to get right. Adds a hyperparameter with no clear benefit when L2 regularization already anchors parameters near positive initial values.
- *Unconstrained training relying on regularization alone*: L2 regularization provides a soft pull toward positive init values but no hard guarantee. For a small detuning (5-10%) and well-calibrated Lambda this would likely work in practice, but provides no protection against edge cases (aggressive learning rates, long training, poor RMSE_baseline calibration).

---

### [D-036] OPEN: LFR structure vs. state-space-only for LPV baseline and augmentation
**Date**: 2026-04-09 (raised in supervisor meeting, not yet decided)
**What**: Decide whether to express the LPV baseline as a true LFR (with M(Y) invertibility as a
rational/symbolic expression) or remain in state-space form (current: `torch.linalg.solve` at
every step).
**Why this matters**:
- Current `linalg.solve` approach is numerically correct but gives zero LFR structural benefit.
- LFR structure is almost essential for control design (H-inf, mu-synthesis) — a primary interest
  of ASMPT even when a black-box augmentation is added on top.
- Expressing M(Y)^{-1} symbolically as a rational function (MATLAB can do this) means no per-step
  matrix inversion; the forward pass becomes matrix-vector products only — computationally cheaper
  and structurally a proper LFR.
- Jan's interconnect framework supports state-space directly (no LFR required), but this trades
  away the control-design benefit.
**Open sub-questions**:
1. Does the parallel augmentation (D-003) still provide the orthogonality regularization benefit
   if the baseline is in state-space form rather than LFR? (I.e., what exactly is traded away?)
2. SVD on the LFR channels: reduces latent signals (good for control), but how does it affect
   interpretability of the learned augmentation states?
3. Identifiability / uniqueness of parameter updating: which parameter combinations only appear
   as sums in M(Y)? Can trajectory excitation separate them, or is norm regularization needed?
**Decision path**:
- If project scope includes control design deliverable → invest in symbolic M(Y)^{-1} (MATLAB)
  to recover LFR structure before augmentation.
- If scope is simulation/prediction only → state-space form is acceptable; note the limitation
  explicitly in the thesis.
**Ruled out**: Nothing ruled out yet — decision deferred pending scope clarification with supervisors.
**Constrains**: LPV model implementation (`lpv_lfr_baseline/`), augmentation interconnect structure,
and any control design work downstream.

---

### [D-037] IMPLEMENTED: Split regularization on degenerate parameter pairs
**Date**: 2026-04-09 (raised in supervisor meeting); **Implemented**: 2026-04-22
**What**: kb1/kb2, cb1/cb2, and Jb/Jh each appear only as sums in the physics equations (K[1,1]=kb1+kb2, C[1,1]=cb1+cb2, M[1,1] contains Jb+Jh). This creates a flat ridge in loss: any split summing to the correct value gives identical RMSE. A scale-invariant "split loss" breaks the degeneracy.
**Why**: The standard RMSE loss has zero gradient in the split direction for these pairs. Without a tiebreaker the optimizer stagnates on a line rather than converging to the true split.
**Implementation** (`SPLIT_REG_WEIGHT = 1e-2`):
```python
# lfr_param_block.py -- ParameterizedLFRBlock.split_loss()
def split_loss(self) -> Tensor:
    p = self._recover_params()
    kb1, kb2 = p[0], p[1]
    cb1, cb2 = p[5], p[6]
    return (
        ((kb1 - kb2) / (kb1 + kb2)).pow(2)   # symmetric pairs -- prefers equal split
        + ((cb1 - cb2) / (cb1 + cb2)).pow(2)
        + (self.log_params[11] - self.log_params[12]).pow(2)  # Jb/Jh -- log-space (true values differ)
    )
```
- kb/cb pairs: normalised squared difference `((a-b)/(a+b))^2` — dimensionless, scale-invariant, zero at a=b. Correct because true values are equal by design (kb1=kb2=1987.5, cb1=cb2=9.0).
- Jb/Jh: log-space squared difference — prefers proportional fractional detuning rather than equal split. Correct because true values differ (Jb=1.0, Jh=0.05); forcing equal split would be physically wrong.
- Weight `1e-2` is small enough that it does not meaningfully distort the RMSE landscape when the sum is already near its correct value; it only resolves the flat direction.
**Compute cost**: Three tensor ops per backward pass — negligible.
**Constrains**: `train_param_recovery.py` (`SPLIT_REG_WEIGHT`, `train()` signature, loss assembly, hist_entry, save dict); `lfr_param_block.py` (`split_loss()` method).
**Old notes (pre-implementation)**:
- Roland's suggestion: centre and normalize log-parameters around ~1 before gradient step. Not implemented — log/exp reparameterization (D-035) already handles scale.
- Alternative to log: `p^2` reparameterization. Not needed; log/exp stable in practice.

---

### [D-038] Simulation study extra state: Y-position-dependent Dahl friction states [z₁, z₂]
**Date**: 2026-04-10
**What**: The 8-state data-generating model for the augmentation simulation study adds two Dahl friction states [z₁, z₂] — bristle deflections on the X₁ and X₂ guides — to the 6-state LPV baseline. The baseline remains unmodified (6 states, constant C and K). The augmentation must discover the extra states and their coupling.

Data-generating model dynamics (extra states):
```
ż₁ = Ẋ₁ − (|Ẋ₁|/g) · z₁     where Ẋ₁ = Ẋ + (Lb/2)·Θ̇
ż₂ = Ẋ₂ − (|Ẋ₂|/g) · z₂     where Ẋ₂ = Ẋ − (Lb/2)·Θ̇

Y-dependent Coulomb amplitudes:
  Fc₁(Y) = Fc · (Lb/2 − Y) / Lb
  Fc₂(Y) = Fc · (Lb/2 + Y) / Lb

Modified force equations in data generator:
  F_X_friction = Fc₁(Y)·z₁ + cg1·Ẋ₁ + Fc₂(Y)·z₂ + cg2·Ẋ₂
  τ_Θ_friction = (Fc₁(Y)·z₁ − Fc₂(Y)·z₂) · Lb/2 + (cg1·Ẋ₁ − cg2·Ẋ₂) · Lb/2
```

**Why**: Five candidates were evaluated; the friction states were the only choice satisfying all criteria simultaneously:
1. Genuine dynamic states (own ODE, memory — not computable from current [X,Θ,Y,Ẋ,Θ̇,Ẏ])
2. Creates coupling: asymmetric Fc₁(Y) ≠ Fc₂(Y) when Y ≠ 0 generates Y-dependent torque on Θ from X motion
3. Position-dependent: coupling amplitude varies with Y, enriching the LPV structure (C(Y) alongside M(Y))
4. Direction-sensitive: z₁, z₂ carry history through direction reversals (pre-sliding transient)
5. Physically motivated: load distribution N₁(Y), N₂(Y) on X-guides changes with payload Y — documented in gantry literature
6. Directly connects to D-025 (supervisor's hysteresis observation) as the proper dynamic formulation of sign(Ẏ) scheduling
7. Exact Jan-analogy: extra states in data generator (absent from baseline), augmentation must rediscover them

**Ruled out**:
- *Support structure resonance [x_b, ẋ_b]*: Garcia's 37.7 Hz die-cast base resonance is specific to his rig; Telica uses granite/polymer-concrete frame with first resonance >100 Hz, above control bandwidth. No Y-dependence — does not enrich LPV structure.
- *Cross-arm bending mode [δ, δ̇]*: Garcia explicitly calls cross-arm vibration "negligible in comparison to the coupling between actuators." Building the simulation study on a phenomenon the original paper dismisses is a weak foundation.
- *Coriolis coupling (Ẏ·Θ̇ terms)*: Not a state — a static nonlinear function of existing states. A non-dynamic augmentation could capture it without extra states. Reserved for second augmentation step (D-024).
- *sign(Ẏ)*: Not a state — a static (memoryless) nonlinearity. Already approximately modelled as Coulomb friction in the baseline. The friction states [z₁, z₂] are the correct dynamic version that captures the hysteresis memory sign(Ẏ) approximates.

**Constrains**:
- Data generator implementation extends `rk4_step` / `lfr_simulate.py` to an 8-state variant; the 6-state baseline code is NOT modified.
- Augmentation interconnect uses `nxd=2` extra states (analogous to Jan's `nxd=2` for m₃ in MSD).
- Verification: true z₁(t), z₂(t) from the data generator are saved and compared against the augmentation's learned states.
- Key metric: Θ prediction error as a function of Y-position and motion direction.
- Parameter g (Dahl stiffness) and Fc (nominal Coulomb amplitude) must be chosen to produce a physically plausible but clearly observable effect — suggested range: g ≈ 1–5 μm (pre-sliding displacement), Fc ≈ 10–30 N.
- Cross-references: D-022 (extra states in augmentation, not baseline), D-023 (validate parameter recovery before augmentation), D-024 (friction study is the first augmentation demonstration), D-025 (friction states are the dynamic formulation of hysteresis scheduling).

---

### [D-039] Feedback controller operating point per trajectory: Y_initial
**Date**: 2026-04-17
**What**: In `export_lpv_multi_traj.m`, the feedback controller `Cfb` and frozen LTI `G`
are designed at `Y_op = sp.Y_initial` for each trajectory — the Y position at the start
of the main motion. This replaces the previous single frozen choice of `Y_op = 0.3` for
all trajectories.

| Trajectory | Y_initial | Cfb designed at |
|---|---|---|
| T1 | 0.3 | Y = 0.3 |
| T2 | 0.3 | Y = 0.3 |
| T3 | 0.0 | Y = 0.0 |
| T4 | 0.2 | Y = 0.2 |
| T5 | 0.2 | Y = 0.2 |
| T6 | 0.3 | Y = 0.3 |

**Why**: Designing at `Y_op = 0.3` for all trajectories is unnecessarily wrong for T3
(Y=0.0), T4 (Y=0.2), T5 (starts at Y=0.2). Using `Y_initial` gives each trajectory a
controller optimally matched to its operating condition without requiring any Simulink
changes — `Cfb` and `G` are still plain workspace variables.

**Ruled out**:
- *Single Y=0.3 for all*: unnecessarily off-design for T3/T4/T5.
- *Gain-scheduled LPV controller Cfb(Y)*: the correct solution for trajectories where
  Y varies during motion (T1, T5, T6). Requires replacing the fixed LTI `Cfb` block in
  Simulink with an online-scheduled controller (S-function or MATLAB function block).
  Not implemented because it requires modifying the Simulink model, which is out of
  scope for the current parameter recovery phase.

**Constrains**:
- For T1, T5, T6 where Y actively sweeps during the main motion, `Cfb` at `Y_initial`
  is still an approximation — the controller is off-design-point as Y moves. This is
  accepted for now; the recorded `(u_q1, q1)` pair remains a valid input-output dataset
  for parameter recovery regardless of controller quality, since both signals are saved
  exactly as simulated.
- If gain-scheduled control is added later, `Cfb` computation must move inside the
  trajectory loop and be evaluated online using the current Y state.

---

### [D-040] torch.compile on rk4_step deferred — hardware constraint
**Date**: 2026-04-18
**What**: `@torch.compile(fullgraph=True, dynamic=False)` was added to `rk4_step` as
Phase 2 of the Step 3c training speed optimization. It has been removed and deferred.
**Why**:
- Training GPU is a Quadro P2000 (CUDA Capability 6.1). Triton requires CC ≥ 7.0 (Volta+).
  `backend='inductor'` fails with: *"Found Quadro P2000 which is too old to be supported
  by the triton GPU compiler"*.
- CPU path also blocked: MSVC `cl.exe` is not installed on this Windows machine; TorchInductor
  cannot compile C++ kernels for the CPU fallback.
- `backend='aot_eager'` works on both but provides no kernel fusion — only Python dispatch
  overhead reduction, which is negligible on a GPU-bound workload.
**What WAS completed (kept)**: Phase 1 (GMatrix → (15,15) tensor refactor) is complete
and stays. It reduces the buffer count from 7 to 1 in `lfr_block.py`, simplifies the API,
and is the necessary prerequisite for Triton kernel fusion once hardware is upgraded.
**Ruled out**: `aot_eager` as a permanent solution — it provides ~0% speedup on CUDA.
**Re-enable when**: Training moves to a Volta/Turing/Ampere GPU (CC ≥ 7.0). The code
comment in `lfr_simulate.py` contains the exact decorator to uncomment.
**Known issue to fix on re-enable**: `rk4_step` is called in both gradient (training loop)
and no-grad (eval pass) contexts. With the default `cache_size_limit=8`, this triggers
`GLOBAL_STATE changed: grad_mode` recompilations that eventually raise `CacheLimitExceeded`.
Fix: use `options={"cache_size_limit": 4}` in the decorator — allows grad/no-grad × dtype
specializations without restructuring the call sites. No logic change needed.
**Constrains**: `lfr_simulate.py` — the commented-out decorator block must not be removed;
it documents the intended optimization for future hardware.

---

### [D-041] Physics computation kept in float64 — float32 not precise enough
**Date**: 2026-04-18
**What**: All physics in `rk4_step` and `lfr_forward` (the polynomial loop solve, RK4
integration, matrix products) is computed in float64. The Jan framework uses float32
throughout; explicit casts are applied at the block boundary in `lfr_block.py` and
`lfr_param_block.py` (float32 → float64 on entry, float64 → float32 on exit).
**Why**:
- The polynomial loop solve `N(Y)/d(Y)` uses Horner evaluation of the adjugate matrix
  (N0, N1, N2) and the scalar determinant polynomial d(Y). These involve subtraction
  of near-equal terms and division by a scalar that can be small near the limits of the
  Y operational range. float32 provides only ~7 decimal digits of precision — insufficient
  to guarantee numerical accuracy of the solve across the full Y range and over long
  trajectories (4000 RK4 steps per segment).
- RK4 integration accumulates truncation error per step; float32 rounding adds a second
  error source on top. Over 4000 steps at ts = 1/16 kHz the accumulated float32 error
  has not been validated against the required parameter recovery accuracy.
- Physical parameters (masses ~10–25 kg, stiffnesses ~2000 N/m) span two orders of
  magnitude. float32 relative error (~1e-7) translates to absolute errors that may not
  be negligible for gradient-based parameter recovery where small parameter deltas matter.
**Ruled out**: float32 physics — not validated, risk of gradient degradation during
parameter recovery training. The Quadro P2000 has 1/32 fp64-to-fp32 throughput ratio
(Pascal), so float32 would be significantly faster, but correctness must come first.
**Future investigation**: If training speed becomes a bottleneck after moving to better
hardware (or if float64 remains slow), run a controlled experiment:
1. Train with float64 (reference), record `param_table()` and val RMSE per epoch.
2. Remove the two cast lines in `lfr_block.py` to run entirely in float32.
3. Compare `param_table()` — if parameters agree to within ~0.1% and RMSE curves match,
   float32 is acceptable and the cast lines can be removed permanently.
The comment in `lfr_block.py` marks the exact two lines to change.
**Constrains**: `lfr_block.py` and `lfr_param_block.py` — the float32↔float64 cast lines
must not be removed without the above validation. `lfr_forward.py` and `lfr_simulate.py`
need no changes; they operate on whatever dtype the caller passes.

---

### [D-042] Training loss normalized by per-channel output standard deviation (sigma)
**Date**: 2026-04-20
**What**: The MSE training loss in `train_param_recovery.py` is computed in sigma-normalized space:
```python
sigma = std of q1 across all 6 TRAJ_SPECS trajectories, per channel  # (3,) float64 tensor
err   = (Y_pred - q1_seg) / sigma
mse_loss = err.pow(2).mean()                                           # dimensionless
```
`sigma` is computed over the **full trajectory set** (all TRAJ_SPECS, not just ACTIVE_TRAJ_IDS) and cached to disk. It does not change when the active trajectory subset is changed.

**Why**: The three output channels [X1, X2, Y] are in metres but have different signal amplitudes. Without normalization the Y channel (largest excursion) dominates the loss, pulling parameter gradients toward Y-related parameters (mh, cy) at the expense of X-related ones (m1, m2, cg1, cg2). Dividing by sigma gives each channel unit variance, so MSE contribution is proportional to relative prediction error, not absolute channel amplitude.

Using the full TRAJ_SPECS for sigma (not the active subset) means:
- Sigma is stable regardless of which trajectories are active — no cache invalidation when ACTIVE_TRAJ_IDS changes.
- Sigma represents the full operating envelope of the system, not just the subset being trained on.

**Connection to D-034 (RMSE_baseline_normalized)**: Because the loss is dimensionless, the RMSE_baseline passed to `ParameterizedLFRBlock` must also be in sigma-normalized units. `rmse_baseline_normalized` is computed by `_aggregate_normalized_rmse_baseline()`, which applies the same per-channel sigma division to the per-trajectory RMSE before aggregating. This is the value passed to the block — not the metre-space `rmse_baseline`. See D-034 for the full Lambda calibration rationale.

**Ruled out**:
- *No normalization*: Y channel dominates; X1/X2 parameter gradients are suppressed.
- *Global scalar normalization*: a single scalar (e.g. overall std) does not correct the per-channel imbalance.
- *Normalizing by active-subset sigma*: sigma would shift when ACTIVE_TRAJ_IDS changes, making Lambda (which is fixed at block construction) inconsistent across runs.

**Constrains**: `train_param_recovery.py` — `sigma` must always be computed from the full TRAJ_SPECS, not the active subset. The `SIGMA_CACHE_VERSION` constant must be incremented if TRAJ_SPECS itself changes. Any future training script for this system must apply the same sigma normalization and pass `rmse_baseline_normalized` (not metres) to the block.

---

### [D-043] Checkpoint/epoch selection strategy for parameter recovery training
**Date**: 2026-04-20
**What**: Three decisions about which parameter vector to save and how to track convergence:

1. **Current phase (clean MATLAB data):** Use **Polyak-Ruppert tail averaging** over the plateau phase. Start averaging on the first LR reduction event from `ReduceLROnPlateau` — this trigger is automatic and requires no additional hyperparameter. The averaged `log_params` are saved alongside the final-epoch `log_params` in the `.pt` file. This is not yet implemented.

2. **Convergence tracking:** Run a full-trajectory eval (same as step 5) every `PARAM_LOG_INTERVAL` epochs. Save the result in `history`. This gives a clean convergence curve comparable to the final step 5 result, and provides the signal for best-epoch tracking if needed. This is not yet implemented.

3. **Future phase (measurement noise):** Polyak averaging over the late plateau becomes harmful — the late iterates are corrupted by semi-convergence (the optimizer fits noise after exhausting the clean signal). Switch to early stopping:
   - Known noise variance → **Morozov Discrepancy Principle**: halt when the smoothed training residual hits the noise floor `τ·δ²`.
   - Unknown noise variance → **L-curve method**: log `(residual norm, solution norm)` at each epoch; find the corner post-training. This requires logging `‖log_params‖` (or deviation from init) alongside the loss in `history`. The `log_params_snapshot` already saved at `PARAM_LOG_INTERVAL` supports this.

**Why:**
- **Saving last epoch is not principled.** The last epoch may not be optimal: the stochastic 8-segment train loss has high variance, and `ReduceLROnPlateau` does not guarantee the last iterate is the best. Last ≈ best only if LR has fully decayed to `min_lr` — which may not happen within 2000 epochs.
- **Best-epoch on stochastic train loss is actively wrong.** It rewards lucky random batches, not genuine parameter improvement. Confirmed by both the subagent research and Gemini Deep Research.
- **Polyak tail averaging is theoretically optimal for clean data.** For a 13-parameter, physics-constrained, locally convex problem, iterate averaging achieves the Cramér-Rao lower bound. It cancels the zero-mean batch noise algebraically without any additional computation beyond a running sum of 13 scalars.
- **Full-trajectory eval every PARAM_LOG_INTERVAL solves two problems at once:** the convergence plot becomes directly comparable to the step 5 final result, and it provides a stable signal for best-epoch tracking that is immune to batch sampling noise.
- **Semi-convergence is a real risk when noise is added.** With only 13 parameters, structural overfitting cannot occur. But the optimizer will eventually start fitting measurement noise rather than physics — "clean priority learning" means accuracy peaks mid-training, not at the end. Polyak averaging the corrupted plateau would amplify this effect.

**Ruled out:**
- *Best-epoch on stochastic train loss:* rewards sampling variance; statistically invalid for epoch selection.
- *Best-epoch on fixed held-out segment set:* computationally wasteful per epoch; vulnerable to trajectory divergence and the same noise issue as the train set (just with a fixed random seed instead of a varying one). Less principled than full-trajectory eval.
- *Schedule-Free optimizer (Defazio 2024):* eliminates epoch selection entirely by unifying momentum and iterate averaging — promising but not implemented. Would remove `ReduceLROnPlateau` and its associated patience/factor hyperparameters. Deferred as a future experiment.
- *Stochastic Weight Averaging (SWA) with cyclical LR:* correct in principle but requires replacing `ReduceLROnPlateau` with a cyclical schedule. More disruptive to the current setup than Polyak tail averaging which re-uses the existing scheduler trigger.

**Constrains:**
- `train_param_recovery.py`: add `averaging_active` flag, `AveragedModel` from `torch.optim.swa_utils`, triggered by first LR reduction. Save `averaged_log_params` in the `.pt` file.
- `train_param_recovery.py`: add full-trajectory eval loop inside the `PARAM_LOG_INTERVAL` block. Save per-trajectory RMSE snapshots in `history`.
- When noise is added: `history` must log solution norm `‖log_params − log(params_init)‖` per epoch to support L-curve analysis post-training. The `log_params_snapshot` at `PARAM_LOG_INTERVAL` already provides this at coarser resolution.
- Both `params_learned` (last epoch) and `params_learned_avg` (Polyak average) must appear in the final `.pt` save so results can be compared.

---

### [D-044] Multi-trajectory loss function: binary masking + per-trajectory per-channel sigma
**Date**: 2026-04-21
**What**: Replace the current global-sigma unweighted MSE loss with a loss that applies
binary channel masks per trajectory group, normalizes by per-trajectory per-channel signal
std, and averages per segment before averaging over the batch.

**The six problems with the current implementation (global sigma, no masking):**

1. **Dormant channels included in the loss.** On T1/T6 (Y-only), X1 and X2 are actively
   suppressed by the feedback controller but contribute equally to the MSE. The optimizer
   receives gradient signal from controller suppression dynamics rather than plant physics,
   pulling physical parameters away from their true values.

2. **Global sigma dilutes Y, inflates X.** sigma[Y] is computed from all 6 trajectories
   including T2/T3/T4 where Y is constant → sigma[Y] is artificially small → Y is
   over-weighted. sigma[X1] is computed across all 6 trajectories including T1/T6 where
   X1 ≈ 0 → sigma[X1] is artificially large → X1 is under-weighted on trajectories where
   it is actually active. Both biases compound simultaneously.

3. **Within-trajectory amplitude imbalance.** On T5 (X + Y both active), if Y sweeps much
   more than X1/X2, Y dominates the loss. Parameters primarily identified by X motion
   (m1, m2, cg1, cg2) are undertrained relative to Y-related parameters (mh, cy).

4. **Cross-trajectory amplitude imbalance.** Trajectories with the same active channels
   can have very different amplitudes (T1 conservative vs T6 aggressive Y sweep). A single
   global sigma[Y] does not capture this: T6 segments always dominate T1 segments in the
   loss, even though both are Y-only trajectories contributing equal information about Y.

5. **Denominator is inconsistent across segments.** Different segments have different numbers
   of active channels (T1: 1 active, T2/T3/T4: 2 active, T5: 3 active). A fixed global
   denominator gives unequal weight per active channel-step across trajectory groups. No
   single global denominator is correct for all segments simultaneously.

6. **Adam sees inconsistent loss scale across batches.** With 8 segments sampled from
   different trajectory groups per batch, the loss magnitude depends on which groups appear.
   Without per-segment normalization, Adam's second moment estimate v_t cannot stabilize,
   making its adaptive learning rate unreliable.

**Why**: Problems 1–6 compound. Problems 1 and 2 corrupt the gradient direction. Problems
3 and 4 create systematic undertraining of specific parameter subsets. Problems 5 and 6
make Adam's adaptation unreliable across epochs. The combination means the optimizer is
simultaneously given wrong gradient directions AND wrong step sizes.

**Chosen solution:**
```
For each segment in the batch:
  1. Binary mask:  zero out dormant channels for this trajectory group
  2. Normalize:    divide residual by sigma[traj_id][channel]
                   (sigma computed from that trajectory individually, active channel only)
  3. Per-segment loss = masked_normalized_err².mean() over (active_channels × T)
Average segment losses over the batch.
```

Formally:
```
loss = (1/B) Σ_i [ (1 / (n_active_i · T)) Σ_c Σ_t  m_{g,c} · ((ŷ_c - y_c) / σ_{traj,c})² ]
```

where m_{g,c} ∈ {0,1} is the binary mask for channel c in trajectory group g,
and σ_{traj,c} is the std of channel c computed from that trajectory only.

**Why per-trajectory sigma solves problems 3 and 4:** Each trajectory's sigma reflects
its own excitation amplitude. T6's sigma[Y] ≈ 300 mm; T1's sigma[Y] ≈ 50 mm. After
normalization, a 30 mm residual on T6 contributes (30/300)² = 0.01 — equal to a 5 mm
residual on T1 contributing (5/50)² = 0.01. Equal relative contribution regardless of
absolute excitation amplitude.

**Why per-segment averaging solves problems 5 and 6:** Each segment contributes O(1) to
the loss regardless of how many active channels it has. Adam sees a consistent loss
magnitude across all batches regardless of trajectory group composition. The second
moment estimate v_t stabilizes correctly.

**Forward compatibility (future hardware data):** When moving to real measurements with
additive noise, per-trajectory sigma transitions directly to the principled Λ⁻¹ weighting
(Ljung 1999 §7.4, Gautier, Janot & Vandanjon 2013). At high SNR (gantry encoders:
signal mm–cm, noise µm), signal std ≈ noise-floor-independent scale → per-trajectory
sigma is the high-SNR approximation of Λ⁻¹ weighting. No architectural change required
at the transition to hardware data; only the interpretation of sigma changes.

**Literature support:**

*Problem 1 — Dormant channel masking in gradient-based SysID (verified by direct quote):*
- **Werling et al., "Trajectory-based actuator identification via differentiable
  simulation"** (PDF p. 5, Eq. 2 and p. 12, Appendix B): loss `L = (1/MN) Σ ‖W(s'−s)‖²`
  with `W = diag(w_q, w_qdot)`; set to `diag(1, 0)` so velocity remains in the rollout
  but *"velocity residuals are not penalized because the measured velocity signal is
  noticeably noisier than position."* Directly confirms: mask in the loss, keep in the
  dynamics. Optimizer: Adam (Appendix B).
- **Gautier & Khalil (1990)** — dormant joints produce structural zeros in the regressor
  (classical least-squares analog). Forssell & Ljung (1999) additionally applies when
  measurement noise is present (closed-loop bias-pull mechanism).

*Problems 2 & 3 — Amplitude normalization across channels in gradient-based SysID (verified):*
- **Lutter et al., "Dynamic Modeling of Robotic Manipulator via an Augmented Deep
  Lagrangian Network"** (PDF p. 4, Eq. 8): Mahalanobis norm with diagonal covariance
  matrix W_τ; explicit justification: *"It is necessary to normalize the loss function
  using covariance matrix since the torque magnitude may vary greatly from joint to joint."*
- **Lutter et al., "Combining Physics and Deep Learning to learn Continuous-Time Dynamics
  Models" (Deep Lagrangian Networks, IJRR)** (PDF p. 7, Eq. 12): same Mahalanobis norm
  with diagonal W_τ; *"It is beneficial to normalize the loss using the covariance matrix
  because magnitude of the residual might vary between different joints."*
- **"Constrained Gray-Box Identification of Electromechanical Systems Under Unfiltered
  Step-Response Data"** (PDF pp. 6–7, Eq. 3): normalized composite residual dividing
  trajectory errors by `RMS(signal)` per channel; *"naturally balances the relative
  contribution of current and velocity; thus α_ω = α_i = 1 is sufficient and avoids
  additional manual scaling."*

*Problems 5 & 6 — Segmented minibatch objective for Adam consistency (verified):*
- **Werling et al. (above)**, Eq. 2: loss averaged over M segments and N timesteps as
  `(1/MN) Σ_j Σ_i ‖W(s'_{i,j} − s_{i,j})‖²` — each segment normalized independently
  before batch average. Adam confirmed as optimizer (Appendix B).

*Problem 4 — Cross-trajectory amplitude imbalance:*
- **No exact citable method found** that matches all of: multiple trajectories + same
  active channels + different amplitudes + joint gradient-based physical parameter ID +
  trajectory-specific normalization in the training loss.
- **Citable principle — experiment-balanced weighting:** adjacent inverse-identification
  literature explicitly supports the broader principle that multiple experiments should
  contribute in a balanced or uncertainty-weighted way to the cost function, rather than
  in proportion to raw residual magnitude:
  - **Zhang et al., Int. J. Solids Struct. (2023), doi:10.1016/j.ijsolstr.2023.112534**:
    explicitly states that good inverse-identification results depend on *"maintaining
    equal contribution of the strain states from each experiment to the cost function"*
    — the clearest paper-level support for equal cross-experiment contribution.
  - **Neggers et al., Mech. Mater. (2019), doi:10.1016/j.mechmat.2019.03.001**:
    when combining multiple experiments and data sources, weighting should follow
    measurement uncertainty derived from a Bayesian formulation — citable basis for
    experiment-wise balancing rather than raw aggregation.
- **Framing for thesis:** per-trajectory sigma normalization is an engineering
  realization of experiment-balanced weighting — supported in adjacent inverse-
  identification literature as a principle, but not a canonical standard method in
  robot gradient-based SysID. It is not "uncited" but it is also not "established."

*Supporting context — gradient-based physical SysID as established paradigm (verified):*
- **Muratore et al., "Differentiable Simulation for Physical System Identification"
  (RA-L 2021)** (PDF p. 6, Sec. IV-B): friction and mass estimated by backpropagating
  MSE loss through differentiable simulator via PyTorch AD; Adam optimizer.
- **Saveriano et al., "Physics-informed online learning of gray-box models by moving
  horizon estimation" (EJC 2023, 100861)** (PDF pp. 3–4): physical submodel + neural
  network trained via BPTT; arrival cost covariance *"can be seen as an adaptive
  learning-rate."*
- **Ljung (1999) §7.4 eq. (7.27)** — Λ⁻¹ weighting of multi-output prediction errors
  (classical PEM; per-trajectory sigma is the high-SNR approximation of this).
- **Gautier, Janot & Vandanjon (2013), IEEE TCST** — per-joint inverse-std normalization
  *"normalises the errors"* in closed-loop robot ID (regressor analog).

**Ruled out:**
- *Global sigma (D-042):* contaminated by inactive-channel samples for every channel
  (Problems 1–4). Documented as the identified flaw in D-042.
- *Per-channel-global sigma (no per-trajectory split):* solves Problems 1–2 partially
  but not Problems 3–4. T6 still dominates T1 after normalization.
- *Per-segment sigma (normalize each segment by its own std):* independently normalizes
  each segment but breaks Adam — momentum estimates are built from segments with
  incompatible normalization bases, corrupting gradient direction across batches.
- *GradNorm (Chen et al. 2018):* correct in principle but requires computing ‖∂L_i/∂θ‖
  through the RK4 graph at every step — expensive and unverified on physical grey-box
  sensitivity Jacobians.

**Constrains:**
- `train_param_recovery.py`: precompute `sigma[traj_id][channel]` from each trajectory's
  active samples before training. Pass trajectory ID with each segment in the batch.
- Loss function must use per-segment averaging (Option B), not global averaging (Option A).
- When hardware data is available: replace sigma computation with noise std estimated from
  static measurements; loss architecture unchanged.

**Implemented**: 2026-04-21 in `lpv_lfr_baseline/scripts/train_param_recovery.py`.
Changes: `CHANNEL_MASKS` dict (6 changes), `_get_or_compute_sigma` rewritten to return
`{traj_id: (3,) tensor}` (SIGMA_CACHE_VERSION bumped to 2), sigma display table updated,
`sample_plan` captured in training loop, per-segment loss loop replacing 2-line MSE,
`_aggregate_normalized_rmse_baseline` updated to mask + per-trajectory sigma.
Verified: sigma table output correct (dormant channels = 1.0 m, active channels physically
meaningful); exit code 0; loss value is O(1) per segment.

---

### [D-046] Multi-mode crest factor not fixed for simulation; fix specified for hardware
**Date**: 2026-04-30
**File**: `Matlab-scripts/export_param_recovery_inject_ref.m`, function `generate_ref_multisine`

**What**: The multisine generator designs each spatial mode (common, diff, y) as an
independent Schroeder-phase odd-harmonic signal. When two modes are combined on the same
actuator channel, the combined signal is no longer guaranteed to be Schroeder-optimal.
The only affected trajectory is T8 (`ms_modes = {'common', 'diff', 'y'}`), where:

```
X1_ms = common_sig + diff_sig   (two modes, overlapping frequency bands at 1-20 Hz)
X2_ms = common_sig - diff_sig
Y_ms  = y_sig                   (single mode, no issue)
```

T1-T7 each assign at most one mode per actuator channel, so T8 is the only case.

**Why the gap exists**: Schroeder phases minimize crest factor for a single multisine
signal. When two Schroeder signals with different seeds are summed, the combined CF
is not guaranteed to be ~1.58. The seed-based phase offset in the script provides partial
decorrelation between modes:

```matlab
phi = phi + 2*pi*freqs*(seed - 1)/(7*f_high);
```

This is a linear phase ramp (time shift) that decorrelates modes but does not produce
a Schroeder-optimal combined signal.

**Why we are not fixing it for simulation**: The kinematic pre-check (`check_ref_total`)
evaluates the actual position, velocity, and acceleration of `r_total = r_traj + r_ms`
before each simulation. Any elevated peak caused by non-optimal CF is caught there and
stops the amplitude sweep. For noise-free simulation data, crest factor is a hardware-safety
metric, not a parameter identifiability metric. The sweep already enforces the binding
constraint (kinematics), so the CF gap has no practical consequence in the current pipeline.

**Ruled out for simulation**: Interleaved frequency grids and per-channel numerical
phase optimization. Both add complexity with no measurable benefit when `check_ref_total`
already catches kinematic violations.

**Fix for hardware experiments**: Use interleaved odd harmonics to eliminate frequency
overlap between modes on the same channel. For T8 with two X-modes:

```
common mode: odd harmonics 1, 5,  9, 13, 17 Hz  (every other odd)
diff   mode: odd harmonics 3, 7, 11, 15, 19 Hz  (interleaved)
```

Combined on X1: harmonics at 1, 3, 5, 7, 9, 11, 13, 15, 17, 19 Hz with no overlap.
Each mode's Schroeder phases apply to non-overlapping lines, so the combined CF is
still bounded by the per-mode Schroeder construction.

Cost: each mode gets half the lines in the shared band (1-20 Hz). For diff mode this
gives 5 lines instead of 10 in 1-20 Hz, which remains above the F >= 7 guard only
if the full common mode band (1-100 Hz) is counted. If the F >= 7 guard is applied
per-mode, the diff band would need to be widened or the grid adapted. Verify the
guard on the actual interleaved line count before implementing.

**Constrains**: For all hardware experiments on T8 involving simultaneous common and diff
modes, switch to interleaved odd harmonics in `generate_ref_multisine`. For simulation,
no change required.

---

### [D-045] param_loss disabled (PARAM_LOSS_WEIGHT = 0.0) for parameter recovery training
**Date**: 2026-04-22
**What**: `PARAM_LOSS_WEIGHT = 0.0` in `train_param_recovery.py` — `param_loss()` is not
added to the training loss. The method exists on `ParameterizedLFRBlock` but is bypassed.

**Why**: `param_loss` is a Lambda-weighted L2 pull toward `params_init` (the detuned
initial values). It was designed for the noisy-data regime, where the MSE landscape is
rough and the optimizer needs an anchor to stay in a physically plausible region.

In the parameter recovery setting:
- Training data is noise-free MATLAB simulation output.
- The Python model reproduces MATLAB exactly at the true parameter values (verified:
  full-trajectory RMSE on T1 = 0.000 mm at `_TRUE_PARAMS`).
- The MSE landscape therefore has an unambiguous global minimum at the true parameters.

Under these conditions, `param_loss` provides no benefit and actively harms convergence:
it adds a competing gradient pull toward `params_init` (the detuned values, ±10% from
true), which is the wrong target. The stronger the regularization weight, the further the
optimizer is biased away from the true parameter values.

**Ruled out**: Enabling `param_loss` at any non-zero weight for noise-free parameter
recovery — it anchors toward detuned init, not toward truth, and slows or prevents
convergence to the true parameters.

**When to revisit**: If training data gains additive measurement noise (encoder noise,
etc.) and the MSE landscape becomes rough or ill-conditioned, a small `param_loss` weight
anchored toward physically plausible values may help stability. At that point, `params_init`
should ideally be updated to the best currently known parameter estimate rather than the
detuned starting values, to avoid the wrong-anchor problem documented here.

---

### [D-047] Parameter sensitivity diagnostic removed from experiment_diagnostics.py
**Date**: 2026-05-03
**What**: `_diag_param_sensitivity` was implemented and then removed. The final
`experiment_diagnostics.py` contains three diagnostics only: FFT, step response,
and observability. Segment length is determined from the step response oscillatory
frequency alone.

**Why it was built**: An attempt to determine the minimum segment length rigorously —
by computing `∂y/∂log(θᵢ)` for each of the 14 parameters over time (via finite
differences through `simulate_frozen`), finding the time `t_95` at which 95% of
cumulative sensitivity energy is captured, and setting `segment_len = t_95_max`.

**Why it was removed**:
1. **Not supervisor-suggested.** Supervisors explicitly recommended FFT + step response.
   Parameter sensitivity was an independent addition from research reasoning, not
   requested or validated by supervisors. Their guidance: keep it simple, don't solve
   problems you are not facing.
2. **Slow.** 14 parameters × 8 trajectories × 2 forward passes = 224 `simulate_frozen`
   calls per diagnostic run. On CPU eager mode this takes several minutes.
3. **Result was unusable.** `t_95` for all parameters hit the T_test cap of 2.0 s
   (the full decimated trajectory length), meaning sensitivity never converged within
   the available data. The diagnostic returned `segment_len ≈ 39420 samples at 20000 Hz`
   — essentially the full trajectory — giving only 1 segment per trajectory and no
   meaningful segment pool.
4. **Wrong reference timescale.** An earlier version used `segment_len = max(10×tau_max,
   t_95_max)`, which produced 314436 samples (15.7 s) — longer than the trajectories
   entirely. Even after removing the 10× multiplier, the result was still impractical.

**What replaced it**: Segment length is derived from the oscillatory poles in the step
response. The slowest oscillatory frequency `f_osc_min` is extracted from the complex
eigenvalues of `A_c` at each frozen Y operating point. Segment length is then:

    segment_len_s = N_PERIODS / f_osc_min

with `N_PERIODS = 3` (configurable). At `f_osc_min ≈ 4.94 Hz` (Y=0.30 m):
`segment_len_s ≈ 0.61 s → 610 samples at 1000 Hz`. This gives multiple segments per
2 s trajectory and is consistent with the supervisor-recommended approach.

**Ruled out**: Re-enabling sensitivity in any form unless supervisors specifically request
it and longer trajectories are available (so t_95 can actually converge).

**Constrains**: `recommend_segment_len` now only calls `_diag_step_response` — it no
longer requires trajectory data as input (only `fs` and `dtype`). The function signature
changes accordingly.

**Constrains**: `PARAM_LOSS_WEIGHT = 0.0` must be kept for all clean-data parameter
recovery runs. If re-enabled, the anchor target (`params_init`) and weight must be
revisited together.

---

### [D-049] experiment_diagnostics.py: fs_new derived from system physics, not signal content
**Date**: 2026-05-08
**What**: Restructured `experiment_diagnostics.py` in five concrete ways:

1. `fs_new` is now determined from `f_osc_min` (pole analysis, Diagnostic 2) using
   `_FS_RULE_FACTOR = 10`: first candidate in `_FS_CANDIDATES` satisfying
   `fs_new >= 10 * f_osc_min`. Previously, `fs_new` was set from `f_99` (Welch PSD,
   Diagnostic 1) with `_FS_RULE_FACTOR = 8`.

2. `_FS_RULE_FACTOR` changed from 8 to 10 to match the lecture lower bound.

3. `segment_len` is now the maximum of three rules:
   ```python
   segment_len = max(
       ceil(N_PERIODS / f_osc_min * fs_new),   # period rule
       ceil(10 * tau_max * fs_new),              # 10x time constant rule
       10 * n_params,                            # 10x parameter count rule
   )
   ```
   Previously only the period rule was applied (yielding ~608 samples at 1000 Hz).
   With the 10x tau_max rule the correct lower bound is ~15720 samples at 1000 Hz.

4. `f_99` demoted to a warning-only check: if `f_99 > 10 * f_osc_min`, a warning is
   printed that excitation energy is above the model band. `f_99` no longer drives
   any design variable.

5. `[::D]` stride in `_diag_gradient_convergence` replaced with
   `scipy.signal.decimate`, which applies a Chebyshev Type I anti-aliasing filter
   before striding.

**Why**:

*For change 1 and 2:*
- Source: Lecture 9, slides 10-12 (5SMB0): "10 * omega_b <= omega_s <= 30 * omega_b"
  where omega_b is the system bandwidth — a physics quantity, not a signal quantity.
- `f_99` is the 99% energy frequency of the excitation. It measures where the
  injected signal has power, not where the system has dynamics. Setting `fs_new` from
  `f_99` ties the sampling rate to the excitation design rather than the model band.
  This is the wrong causal direction: the sampling rate should be set first (from
  physics), and then the multisine frequency range should be designed to stay within
  the model band.
- Factor 8 is below the lecture-stated lower bound of 10. Factor 10 is used.
- Source: Ljung (1999) — setting fs too high causes all discrete-time poles to cluster
  near unity, degrading numerical conditioning.
- Source: Pintelon & Schoukens (2001/2012) — set fs from the model band, not the
  excitation band.

*For change 3:*
- Source: Lecture 9, slide 9 (5SMB0): "N >= 10 * tau_set,95" and "N >= 10 * n_theta".
- Source: Lecture 3, periodic measurement material (5SMB0) — integer periods required.
- N_PERIODS = 3 is a HEURISTIC (covers the slowest mode with margin; lecture uses 10
  for FRF quality, which is more conservative than needed for BPTT training).
- The 10x tau_max rule dominates at the current parameters: tau_max = 1.572 s,
  giving 15720 samples at 1000 Hz — 25x larger than the period rule alone.
  This may be overly conservative for BPTT (the rule is derived for stationary FRF
  estimation). The discrepancy is now reported in the diagnostics output and should
  be discussed with the supervisor before shortening trajectories.

*For change 4:*
- Source: Gonzalez, van Haren, Oomen, Rojas (arXiv:2410.19629 / IEEE TAC 2024):
  parametric estimator consistency survives aliasing of out-of-band input content,
  provided in-band frequencies are correctly resolved. Therefore `f_99` above the
  model band is not a problem for parameter recovery, only a warning.

*For change 5:*
- Source: Lecture 9 (5SMB0) pre-processing steps: "Apply anti-aliasing filter before
  any downsampling."
- Source: lecture_digital-filters.pdf (4CM00), slides 30-35: filter must provide
  >= 40 dB attenuation at the new Nyquist frequency.
- `scipy.signal.decimate` applies Chebyshev Type I filter automatically.

**Ruled out**:
- `_F99_PHYSICAL_CAP_FACTOR`: applying the 10x rule to `f_99` to cap it at
  `10 * f_osc_min`. Documented in `docs/multisine-diagnostics-interface.md` —
  this applies the 10x rule to the wrong variable and conflates two separate
  design choices.

**Constrains**:
- `experiment_diagnostics.py`: `run_all_diagnostics` now computes `f_osc_min`,
  `fs_new`, and `D` before calling `_diag_fft`. `_diag_fft` accepts `fs_new` and
  `f_osc_min` as keyword parameters.
- `recommend_segment_len`: now returns the max-of-three segment_len, which is larger
  than before. Any caller that relies on the old (period-only) segment length will get
  longer segments and fewer segments per trajectory. This is the correct direction.
- Note: the 10x tau_max rule may produce segments longer than available trajectory
  data (tau_max = 1.572 s => 15720 samples at 1000 Hz; trajectories are approximately
  40000 samples at 20000 Hz = 2000 samples at 1000 Hz). This is a trajectory design
  issue, not a code issue — the diagnostic now correctly reports it.

---

### [D-048] `ref_injection` dataset is incompatible with open-loop parameter recovery training
**Date**: 2026-05-04
**What**: The `ref_injection` dataset (multisine injected into the reference `r`) is
fundamentally incompatible with the open-loop simulation objective used in
`train_param_recovery.py`. The `multisine` dataset (force injection via `f_sim`) is
the correct choice for parameter recovery.

**Why**: The training minimises `||simulate(x0, u_recorded, params) - q1_recorded||²`
open-loop. In `ref_injection`, within the controller bandwidth (≤ 100 Hz):

    u_ms = C * S * r_ms ≈ 0          (sensitivity S ≈ 0 kills the force)
    q1_ms = T * r_ms ≈ r_ms          (position closely tracks reference)

The open-loop model receives a near-zero multisine force but must predict a full-amplitude
multisine position. The residual `q1_ms - simulate(u_ms) ≈ r_ms` is large and almost
independent of plant parameters. This uninformative residual dominates the MSE, masks the
parameter-sensitive gradient from trajectory dynamics, and drives the optimizer into bad
local minima. Observed: `ref_injection` stalls at loss `2.8e-3` vs `base` converging to
`3.2e-7`; recovered parameters off by up to +1083% for `cy`.

With force injection (`multisine`), `f_sim` is generated independently of the plant and
added as a direct input. The open-loop model receives the full multisine force and must
produce the matching oscillations at the correct frequency/amplitude — a parameter-sensitive
residual that gives informative gradients.

**Ruled out**: Continuing to use `ref_injection` for open-loop training. The
S-attenuation argument ("ref injection reaches plant via T≈1") is correct for
closed-loop identification on real hardware; it is irrelevant for the open-loop
simulator in `train_param_recovery.py`.

**Constrains**:
- Use `DATASET = 'multisine'` for parameter recovery training runs.
- `ref_injection` data can still be used for: (a) closed-loop identification frameworks,
  (b) training with the `r_ms` component subtracted from `q1` targets (see D-048 options
  in `docs/ref-injection-openloop-incompatibility.md`).
- T7 and T8 provide genuine observability benefit (all 13 parameters excited simultaneously)
  but only when the multisine injection method is compatible with the training objective.
  They should be included in the `multisine` dataset runs.

---

### [D-050] Resonance/bandwidth-weighted broadband multisine as active experiment design strategy
**Date**: 2026-05-10 (updated same day)
**What**: Active multisine design strategy: all odd harmonics from f_low to f_high, with
amplitude biased toward resonances and system bandwidth. Replaces FIM-driven scan-score
band selection. Declared HEURISTIC — variance motivation is PEM/noise-based, not
BPTT-specific; declared as such to supervisors.

**Design**:
- All odd harmonics from f_low to f_high (full band coverage)
- Amplitude concentrated toward resonances and system bandwidth (Lecture 9 slide 13, 27)
- Schroeder phases: φ_k = -k(k-1)π/F (Schroeder 1970, IEEE Trans. IT)
- Odd harmonics only: enables nonlinearity detection via even output lines (P&S Ch.4 §4.3.2)
- Force injection after controller (D-048): keeps excitation in u_recorded for BPTT replay
- PE condition: F ≥ 7 positive sinusoids (2F ≥ 14 = n_params; Lecture 6 slides 17–20,
  Lecture 9 slide 22: "PE(u) = 2 × harmonics")
- f_low, f_high from system physics (f_osc_min ≈ 4.9 Hz from eigenvalues; f_high ≈ 100 Hz)

**Why resonance-weighted over flat uniform**:
5SMB0 Lecture 9 slide 13 explicitly supports concentrating input power at resonances and
bandwidth. This is the lecture-backed middle ground: stronger motivation than flat uniform
(Ljung §13 §number unconfirmed for our claim), weaker than FIM-optimal but without
FIM's source gaps. Qualitatively compensates for |S| attenuation of force injection
inside the controller bandwidth without requiring the unjustifiable A_k ∝ 1/|S| formula.

**Why broadband over FIM-driven**:
FIM-driven requires ∂G/∂θ at each operating point and has unresolved source gaps for
deterministic BPTT (Gap G1). When NN augmentation is added, FIM-optimal for the 14
known params under-excites model-error frequencies. Broadband with resonance weighting
covers both needs without redesign. FIM-optimal deferred to G12.

**Constrains**:
- Drop scan-score band selection from `export_param_recovery_multisine.m`.
- Replace with all odd harmonics from f_low to f_high, resonance-weighted amplitudes.
- F ≥ 7 bins is the PE lower bound; more is better up to available trajectory length.
- Amplitude weighting shape must be declared as HEURISTIC in thesis and to supervisors.

---

### [D-051] Step 0 preanalysis uses simulation-based empirical Ŝ(jω), not analytical S(jω)
**Date**: 2026-05-10
**What**: The Step 0 survival profile is estimated empirically by injecting a flat broadband
probe into the closed-loop simulation and computing `Ŝ(jω) = FFT(u_total) / FFT(f_sim)`,
rather than computing S(jω) analytically from A_c, B_c, C_c, and the controller.

**Why**: In the current parametric model both methods give identical results. However,
when the model becomes incomplete (NN augmentation added) or moves to hardware, the
analytical S(jω) from the nominal model diverges from the true survival profile. The
simulation-based approach uses the actual closed-loop response at every stage, so the
same code path applies to:
- Current parametric simulation: Ŝ = S (equivalent)
- Augmented simulation: Ŝ reflects changed dynamics automatically
- Hardware: replace simulation run with real measurements — same formula

**Ruled out**: Purely analytical S(jω) from state-space matrices. Correct now but
requires explicit code change at every model update; simulation-based is forward-compatible
at no additional cost.

**Constrains**:
- Step 0 requires a short simulation run before Step 1 can proceed.
- Probe signal: flat broadband multisine (all harmonics, equal amplitude, force injection).
- f_low threshold from `|Ŝ|²` has no universal source — must be declared as engineering choice.

---

### [D-052] FRF pretest uses stage coordinates directly -- no input/output transform
**Date**: 2026-05-19
**What**: The frozen-Y MIMO FRF pretest uses raw stage coordinates throughout:
- Inputs: `[F1, F2, FY]` (physical actuator forces)
- Outputs: `[X1, X2, Y]` (physical position sensors)
No `output_to_modal` or `input_to_modal` transform is applied.

**Why**: Orthogonality of the input matrix comes entirely from the excitation design
(`f_vec = [1,1,0]`, `[1,-1,0]`, `[0,0,1]`), not from transforming the measured signals.
At each frequency line k the U_all columns are orthogonal in stage coordinates by
construction (the [1,1;1,-1] X-block is the Hadamard structure from Lecture 9; Y is
independent). The pretest purpose is frequency range selection -- resonance peak
locations are invariant to coordinate transforms. Stage coordinates are the simplest
valid choice.

**Ruled out**: Kamtin logical coordinates (P matrix transform) -- would enable a direct
oracle-test overlay against the analytical model, but adds scaling decisions with no
benefit for frequency range selection. Ad-hoc symmetric transform `(X1+/-X2)/2`,
`(F1+/-F2)/2` -- neither stage nor logical, has no clear benefit and mismatches kamtin
by constant factors anyway.

**Constrains**:
- FRF is 3x3 in stage coordinates. Plot axis labels are X1/X2/Y for both inputs and outputs.
- All 3 excitation modes (common X, diff X, Y) are retained -- Y is a physical DOF, not
  only a scheduling variable.
- A post-hoc coordinate transform would be needed to directly compare this FRF against
  kamtin's `StageCoordinatesSystem` (which is in logical coordinates).

---

### [D-053] State recovery diagnostic appended to gantry_interconnect_dynamic.py (not standalone)
**Date**: 2026-06-10
**What**: A `state_recovery_diagnostic()` function is added at the end of
`scripts/gantry/gantry_interconnect_dynamic.py` and called after `evaluate_and_save` in
both main paths. It compares encoder state estimates x_hat(k) on the validation set against
physical states reconstructed from measurements (q = inv(P^T) y, velocities via backward FD),
reporting per channel: R2_raw (x_hat[:, :6] read directly as normalized physical states),
R2_linmap (best OLS linear map x_true ~ x_hat @ W + b), and R2_raw_lag1 (against x_true(k-1)).

**Why**: The 2026-06-10 code review verified physics, normalization, wiring, and data loading
as correct, leaving two candidate explanations for poor theta/velocity recovery:
(a) basis rotation -- the dynamic-parallel ANN corrects all 8 derivative channels, so the
output-only loss does not pin states 3:6 to physical velocities; (b) information genuinely
absent (observability / training config). R2_linmap ~ 1 with low R2_raw proves (a);
low R2_linmap proves (b). R2_raw_lag1 > R2_raw exposes the separately-found hybrid encoder
one-sample misalignment (deepSI na_right=0: ypast ends at y[k-1] while the encoder
initializes x(k)).

**Ruled out**: Standalone script in `scripts/gantry/verification/` (preferred per the
self-contained-diagnostic rule) -- rejected by user because no trained checkpoint exists;
the diagnostic must piggyback on the next training run. Window construction follows the
deepSI hist convention exactly (ypast = y[k-na:k], na_right=0) so the diagnostic sees what
training saw.

**Constrains**: Diagnostic runs on the validation trajectory only; windows are subsampled
(~2000) to bound memory. "True" velocities are backward-FD reconstructions at fs=4000 Hz,
exact only for noise-free data (currently the case).

---

## D-054: Encoder initialization via reconstructability map (Hoekstra 2026)

**Date**: 2026-06-11

**Decision**: Replace detached `HybridGantryEncoder` with `linear_encoder_init`-based encoder
from Hoekstra 2026 ("Encoder initialisation methods in the model augmentation setting").

**Why**: The `HybridGantryEncoder` computes physical states analytically with `.detach()`,
freezing them. The FP model's positions/velocities don't exactly match the real system, and
the optimizer cannot correct this mismatch. The `linear_encoder_init` approach initializes
encoder weights from the baseline model's reconstructability map (Eq. 16-17) while keeping
all weights as trainable `nn.Parameter`. This gives a good starting point that the optimizer
can then refine.

**Implementation**:
- Linearize CT gantry model at Y_op=0 and discretize (ZOH at TS_NEW=1/4000)
  → `model_augmentation/systems/gantry_linearization.py`
- Normalize (Ad, Bd, Cd, Dd) with `normalize_linear_ss_matrices()` using training data stats
- Create `linear_encoder_init(A_bar, B_bar, C_bar, D_bar, nx=6, na=25, nb=25)`
- Wrap with `LinearInitEncoderWrapper` (physical encoder + zero-init ANN for augmented states)
- Inject with `na_right=1, nb_right=1` (encoder window includes y(k), required by
  reconstructability map)
- `na = nb = 4*NX_PHYS + 1 = 25` (Jan's rule of thumb)
- Observability rank verified = 6 (full), ZOH vs RK4 error < 1e-11

**Ruled out**:
- Data-based encoder init (SS_pre_encoder, Eq. 35): deferred, not ruled out. Will use if
  model-based struggles with LPV nonlinearity.
- Keeping HybridGantryEncoder: `.detach()` prevents learning of physical state corrections.

**Constrains**: Requires `na_right=1, nb_right=1` in SSE_Interconnect. Baseline simulation
states must exist at `data/gantry/baseline_simulations/multisine_LPV/baseline_states.npz` for
the normalization of the DT matrices.

---

### [D-055] D-017 convention fix migrated into linear_encoder_init_aug
**Date**: 2026-06-23
**What**: The normalization convention fix (D-017) is moved from `LinearInitEncoderWrapper`
(torch_nets.py) into `linear_encoder_init_aug` itself (pre_encoder.py). Six optional
keyword arguments are added: `u_mean, std_u, y0, ystd, x_mean, std_x`. The fix is
implicit: it is enabled if and only if all six are provided; omitting any one disables it
(backward-compatible, collapse property diag1 unaffected).

**Why**: `LinearInitEncoderWrapper` had a dead-code ANN bug (augmented states were not
wired into the optimizer). That bug was the reason `linear_encoder_init_aug` was created.
Putting the convention fix back in a wrapper would recreate the same structural problem.
Embedding it in the class directly keeps the encoder self-contained and eliminates the need
for the wrapper entirely for the augmented case.

**Implementation** (`model_augmentation/fit_systems/pre_encoder.py`):
- `__init__`: if fix_enabled, register three non-learnable buffers:
  - `u_off` (nu*(nb+1), 1): tile(u_mean/std_u, nb+1)
  - `y_off` (ny*(na+1), 1): tile(y0/ystd, na+1)
  - `x_off` (nx, 1): x_mean/std_x
- `forward`: if fix_enabled, add u_off/y_off to uhist_mod/yhist_mod before W^b/W^a;
  subtract x_off from x_b (physical states) after. x_a (augmented) untouched.
  ANN receives original pipeline-convention inputs.

**Verified by**: diag6 (5/6 checks pass; S1/S2/S3 confirm 28-285x NRMS improvement
at init; T1 failure is expected for exact linear system due to self-cancellation).

**Constrains**: Call sites of `linear_encoder_init_aug` that want the fix must pass
all 6 constants. `gantry_interconnect_dynamic.py` must be updated to use
`linear_encoder_init_aug` directly (replacing `linear_encoder_init` + `LinearInitEncoderWrapper`).

---

### [D-056] Narrowband multisine amplitude uses 5% of trajectory RMS, not 40%
**Date**: 2026-06-23
**What**: When `MULTISINE_BAND == 'narrowband'` (130–180 Hz), `force_cap_frac` is 0.05 instead of 0.40.
**Why**: The 40% heuristic was calibrated for broadband excitation where the multisine overlaps with trajectory frequency content (1–7 Hz or 1–200 Hz). For narrowband at 130–180 Hz, the trajectory has zero spectral content, so 40% of trajectory RMS forces is applied entirely in a band where it has no competition — causing the multisine to dominate the total force. The MSD resonance provides Q=10 amplification, so 5% (~10–30 N RMS) still yields 2–10 µm of delta_a, which is measurable. Using 40% produced 100–200 N RMS of narrowband force, far exceeding what is needed.
**Ruled out**: Absolute cap (Option B) — depends on knowing force levels per experiment in advance. Target delta_a SNR (Option C) — requires noise floor characterisation not yet done.
**Constrains**: If `force_cap_frac` is ever made a parameter, narrowband must remain at 5% unless delta_a SNR is verified to be sufficient at lower amplitudes.

---

### [D-057] Narrowband MIMO floor = force_cap_frac × max(traj_rms)
**Date**: 2026-06-23
**What**: After the per-channel 5% rule and inactive_frac, apply `amp_ch = max(amp_ch, force_cap_frac * max(traj_rms))` in narrowband mode only.
**Why**: The per-channel 5% rule undersizes weak channels when one channel dominates (e.g. T1 Y-only: X channels get 0.1 N; T3 X-only: Y channel gets 1.1 N). The floor referenced to the dominant channel's RMS keeps all channels proportionate to the experiment's overall intensity without arbitrary constants. Acknowledged: T1/T5 still give small amplitudes (~0.6–0.95 N) and negligible MSD excitation (~20–30 nm delta_a), accepted because those experiments cover scheduling range, not MSD identification.
**Ruled out**: Absolute floor (5 N) — arbitrary constant with no physical grounding. Per-channel skip based on symmetric-mode activity — excluded anti-symmetric excitation incorrectly.
**Constrains**: T3, T4, T7, T10 carry the MSD identification burden. T1, T5 contribute scheduling and coupling data only.

---

---

### [D-058] Telica real-data verification reads .log directly, no .mat conversion
**Date**: 2026-06-23
**What**: `telica_loader.py` reads Telica `iter*.log` files directly and returns
`(u, q1, fs)` matching `precompute._load_trajectory`'s contract. `run_telica_param_recovery.py`
monkey-patches `precompute._load_trajectory` and `compute_rmse_baseline_metrics` before
calling `train_param_recovery.train()` without modifying either original file.
**Why**: Converting to intermediate `.mat` files adds a redundant step with no benefit — Python
can read the `.log` files directly. Monkey-patching keeps both original scripts untouched.
**Ruled out**: (1) Intermediate `.mat` conversion — unnecessary overhead. (2) Adding a Telica
entry to `_DATASETS` in `train_param_recovery.py` — modifies a shared file for a single use case.
**Constrains**: The loader must always return `(u, q1, fs)` with shapes `(1, T, 3)`, `(T, 3)`,
`float`. If `precompute._load_trajectory` signature changes, `telica_loader.py` must match it.

### [D-059] Telica force input is MF30 kept in raw ci units; I_max unknown without Telica.mat
**Date**: 2026-06-23 (corrected 2026-06-23)
**What**: `u = MF30 × 1.0` (raw ci) is used as the plant input. `_CI_TO_AMP = 1.0` — no
conversion to Amperes. MF30 is the total current command (feedback + feedforward + cogging,
after KF60 saturation).
**Why**: `I_max = M82/100` (AccurET §23.2) is stored in `Telica.mat`, which is not in the repo.
Without I_max the formula `I[A] = MF30 × I_max/32768` cannot be evaluated. The factor
`1/481.882` (earlier logged) comes from the **old 5-column** Telica log format (commented out
at MATLAB line 438); the active MATLAB code does **not** convert MF30 at all. The ci scale
folds uniformly into all recovered mass/stiffness/damping parameters; NRMSE is position-based
and is unaffected by the force-unit scale.
**Ruled out**: (1) `1/481.882` (old 5-column format, not applicable to current logs).
(2) Estimating I_max from drive specs (AccurET 400 15/40A → I_max≈40A gives ~49 A from raw
MF30 values, which is physically impossible, confirming the estimate is wrong without Telica.mat).
(3) Using `MF30 - MF230` as feedforward-only — valid only when cogging is off and saturation
confirmed absent, which cannot be verified from the log files alone.
**Constrains**: Recovered parameter values are in `[unit] × I_max/32768` — not in SI.
Physical values recoverable once `Telica.mat` provides I_max.

### [D-060] Structural validation criterion: NRMSE with 15%/30% thresholds from SEM literature
**Date**: 2026-06-23
**What**: Post-training evaluation computes NRMSE = RMSE / std(q1_measured) × 100% per channel.
Decision rule: < 15% = structure compatible; > 30% = structural mismatch or force-signal problem;
15-30% = ambiguous, inspect trajectory plot.
**Why**: NRMSE is the scale-independent metric recommended for simulation error method (SEM)
structural validation. Thresholds from Schoukens & Ljung (2011) and Paduart et al. (2018).
Absolute RMSE [m] alone is not interpretable without knowing trajectory amplitude.
**Ruled out**: Absolute RMSE threshold — depends on motion amplitude which varies per operating point.
**Constrains**: NRMSE is computed against the full-trajectory simulation, not per-segment training loss.

### [D-061] Telica native sampling rate = 10 kHz from AccurET PLTI; timestamps discarded
**SUPERSEDED by D-073 (2026-07-03)**: the iter logs are 20 kHz native; the controller-notch
fingerprint contradicts the 1/(2*PLTI) formula for these files. Kept for the still-valid
part: raw .log timestamps are host-side reception artefacts and must never be used.
**Date**: 2026-06-23
**What**: `_NATIVE_FS = 10_000.0` Hz (fixed constant). Raw `.log` timestamps are discarded.
Synthetic time axis is built from sample index: `t = arange(N) / _NATIVE_FS`, exactly matching
MATLAB `runFDILCAllHostSwLog.m` line 92. Upsampling to `_FS_TARGET = 20_000.0` Hz is done by
linear interpolation, matching MATLAB `interp1` default.
**Why**: AccurET manual §1 (page 18): position-loop PLTI = 50 µs → FsHz = 1/(2×PLTI) = 10 kHz.
Raw `.log` timestamps are host-side reception times (non-uniform artefact — burst at ~66 kHz
during the first 125 ms, then ~200 Hz). Inferring fs from these timestamps gives ~411 Hz,
which is wrong by a factor of 24. MATLAB explicitly discards them (line 92).
**Ruled out**: Inferring fs from median timestamp difference — produces wildly wrong rate due
to non-uniform host logging. Reading `_NATIVE_FS` from a header field — no such field exists.
**Constrains**: Any code that processes Telica `.log` files must use `_NATIVE_FS = 10_000` Hz
and build synthetic timestamps from sample index. Raw timestamps must never be used for resampling.

### [D-062] Motion detection threshold 1e-9 µm; ILC data has no meaningful standstill to trim
**Date**: 2026-06-23
**What**: `_find_motion_start` uses threshold `> 1e-9` (post µm→m conversion, so effectively
1e-15 m). In the ILC experiment, M0 has quantization noise ±0.012 µm from sample 1, triggering
detection at sample index 1. Pre-motion samples to keep: `max(0, 1 - 500) = 0` → `trim_start = 0`
→ all data is kept (T = 32 856 samples at 20 kHz = 1.64 s).
**Why**: MATLAB `runFDILCAllHostSwLog.m` line 100 uses threshold `> 0` (any deviation from M0[0]).
Python `> 1e-9` is equivalent: both trigger at sample 1 due to quantization noise. MATLAB then
computes `startIdx = 1 - 500 - 1 = -499` (1-indexed); `(1:-499)=[]` is an empty range → nothing
deleted. Python replicates: `trim_start = max(0, 1 - 500) = 0`. The ILC experiment parks the
gantry at a fixed absolute setpoint; the relative M0 signal is near-zero throughout with only
quantization noise — there is no true standstill period to discard.
**Ruled out**: Threshold 0.5 µm (old version) — triggers at sample ~8218 (mid-ramp), discarding
valid ILC data. Threshold 0 (exact MATLAB match) — identical outcome in practice because float
quantization noise is always > 0.
**Constrains**: If a different Telica log has a genuine standstill (non-ILC trajectory), the
threshold still works: quantization noise will again trigger at sample ~1, and the 500-sample
pre-motion window will be preserved. The logic is therefore general.

### [D-017] Convention fix in LinearInitEncoderWrapper

**Date**: 2026-06-12

**Decision**: Add normalization convention conversion inside `LinearInitEncoderWrapper` to
bridge the mismatch between `normalize_linear_ss_matrices` (pure scaling) and the pipeline's
mean-subtracted data.

**Why**: `normalize_linear_ss_matrices` produces (Ad_bar, Bd_bar, Cd_bar, Dd_bar) in the
pure-scaled convention: x_scaled = x/std_x, u_scaled = u/std_u, y_scaled = y/std_y. The
Wb_psi_y and Wb_psi_u matrices are derived from these normalized matrices. But the training
pipeline normalizes with mean subtraction: u_norm = (u - u_mean)/std_u, y_norm = (y - y0)/ystd.
Diagnostic results showed this mismatch caused up to 97% velocity NRMS (dq3), while pure-scaled
reconstruction achieved ~10% (limited by LTI model accuracy and O_n conditioning at 818).

**Implementation**: `LinearInitEncoderWrapper` now accepts optional normalization constants
(u_mean, std_u, y0, ystd, x_mean, std_x). In `forward()`:
1. Add u_mean/std_u and y0/ystd to input (undo mean subtraction → pure-scaled)
2. Wb_psi_y @ y_scaled + Wb_psi_u @ u_scaled (reconstruction in pure-scaled space)
3. Subtract x_mean/std_x from output (pure-scaled → pipeline convention)
Constants are stored as registered buffers (no gradients, move with `.to(device)`).
ANN branch still receives original pipeline-convention data.

**Ruled out**: Adding bias correction to encoder output (CHECK 7 in diagnostic showed this
doesn't work because it assumes perfect pure-scaled reconstruction, which doesn't hold due
to LTI model error). Modifying `normalize_linear_ss_matrices` itself (would break other users).

**Constrains**: All call sites of `LinearInitEncoderWrapper` must pass the 6 normalization
constants. Old call sites (without constants) still work -- convention fix is skipped when
constants are None (backward compatible).

---

### [D-063] Epoch-0 diagnostic thresholds for augmented encoder (diag8)

**Date**: 2026-06-23

**Decision**: `diag8_aug_encoder_init.py` uses absolute NRMS thresholds for physical channels,
and reports augmented channels (delta_a, vdelta_a) without a pass/fail check.

Revised checks:
- C1: all physical NRMS < 1.0 (encoder in-signal range)
- C2: all velocity NRMS < 0.5 (W^b gives reasonable velocity estimates)
- C3: output is finite (no NaN/Inf)
- C4: all position NRMS < 0.2 (position tracking with ANN perturbation)

**Why**: The original checks compared against `1.1 * max(analytical)` and `1.5 * analytical_pos`.
The analytical P_inv baseline is kinematically exact -- positions are computed directly from y
via P_inv, giving NRMS near machine zero. Any absolute threshold relative to that value is
trivially violated by the ANN random-weight perturbation on W^b. Specifically, q3 analytical
NRMS = 0.0 exactly, making `1.5 * 0.0 = 0` an impossible pass criterion.

For the augmented channels (delta_a, vdelta_a): W^a is randomly initialized. delta_a signal
std = 84 µm. Any random output will give NRMS >> 1. Checking NRMS < 1.0 at epoch 0 is
testing a property that only emerges after training, not a property of initialization.

**Ruled out**: Relative thresholds vs analytical; augmented-channel NRMS checks at epoch 0.

**Constrains**: When comparing across training iterations, delta_a NRMS should decrease
below 1.0. If it does not after training, it indicates W^a failed to learn the MSD state.
The diag8 results (.npz file) can serve as the epoch-0 baseline for this comparison.

---

### [D-064] Encoder history na_nb = nxd*2+1 (Jan's standard formula)
**Date**: 2026-06-23
**What**: `na_nb` in `DEFAULT_HP` set to `(NX_PHYS + NX_ANN) * 2 + 1 = 17` samples (4.25 ms at 4 kHz), replacing the previous time-based `NANB_SECONDS = 0.025` (100 samples, 25 ms).
**Why**: Jan's reference implementations (`msd_ndof_interconnect_dynamic.py`, `msd_ndof_interconnect_fit.py`) both use `na = nb = nxd * 2 + 1` as the principled minimum. The factor of 2 provides a margin over the observability lower bound (nxd outputs needed to reconstruct nxd states). Using a physically-motivated time window was longer than necessary and inconsistent with Jan's pipeline.
**Ruled out**: `NANB_SECONDS = 0.025 s` (100 samples) — not principled; more than 6× Jan's formula without justification. Longer windows are not needed because W^b initialization already gives a good physical state estimate from short history.
**Constrains**: If `NX_ANN` changes in `DEFAULT_HP`, `na_nb` updates automatically via the formula. The Optuna search range for `na_nb` should also be anchored around this formula, not a fixed sample count.

---

### [D-065] Output augmentation: y = Cd@x_phys + C_aug@x_aug with trainable C_aug
**Date**: 2026-06-25
**What**: Changed output equation from `y = Cd@x_phys` to `y = Cd@x_phys + C_aug@x_aug`. Replaced `Linear_Output_Block(C=Cd_norm)` with `Parameterized_Linear_Output_Block(C=[Cd_norm|C_aug_init], flag_loss_reg=False)` and changed the state selection from `PHY_IX` to `np.arange(nxd)` (all states).
**Why**: Two constraints blocked training. Constraint 1: gantry has 2 DT poles exactly at |z|=1 (K[q1]=K[q3]=0, rigid-body integrators). ANN routed to any physical state row amplifies ~400x over nf=400 BPTT rollout, producing 800-1634x blowup in 1 gradient step (diag13). Constraint 2: with ANN routed to x_aug only and y=Cd@x_phys, the ANN output is unobservable -- (A_aug, C_aug=0) forms an unobservable pair, ANN gradient is identically zero (diag11 T1). Output augmentation resolves both: the gradient path loss->y->C_aug->x_aug->ANN never passes through A_phys, so Constraint 1 is bypassed. C_aug nonzero gives ANN a gradient path, resolving Constraint 2. Verified by diag15: T1 ANN grad = 3.5e-4 (vs 0 before), T2 val ratio = 1.03x at nf=400 (vs 800-1634x before).
**Why Jan's approach (ANN->all states) does not apply**: Jan's MSD has min(1-|z|) = 4.4e-3 (all springs nonzero), amplification = 4.4x at nf=400. Gantry min(1-|z|) = 0 exactly, amplification = 400x. Jan's default is architecturally safe for MSD and architecturally unsafe for the gantry (diag14).
**C_aug initialization**: `C_aug_init[2,0] = 1e-2` (Y channel receives delta_a weakly). Absorber is coupled to Y axis. Scale 1e-2 keeps the init ANN contribution sub-percent of normalized output. C_aug is trainable (`nn.Parameter` via `Parameterized_Linear_Output_Block`) so it grows during training.
**Ruled out**: (1) ANN->velocity rows [3,4,5]+x_aug: velocities also near-unit-circle; T_vel test showed 836x blowup (diag13). (2) Fixed C_aug (register_buffer): ANN signal stays at 3.5e-4 permanently; C_aug must be trainable. (3) Gradient clipping: clip was inactive at nf=400 (grad_norm 0.26 < max_norm 1.0), so clipping cannot prevent the eigenvalue-amplification blowup (diag13 T_clip: 1634x with clip).
**Constrains**: The 5-step stability test (diag15 T3) showed +14% val degradation when training on 1 trajectory. This is encoder overfitting to a single trajectory, not architectural instability -- the ANN gradient (3.5e-4) is 10000x smaller than the encoder gradient (3.67). Full training on all 8 trajectories is required to assess real convergence. Monitor C_aug magnitude during training: if it stays near 1e-2 after many epochs, the ANN/encoder may not be learning the absorber dynamics.

---

### [D-078] Noise-floor acceptance criterion for the augmentation benchmark (Jan's SNR method, pinned to the baseline residual)
**Date**: 2026-07-05
**What**: Define "good enough" for the augmentation on the multisine pipeline via Jan's output-noise convention (`msd_ndof_interconnect_dynamic.py`: `sigma_n = rms(y) * 10^(-SNR/20)`, noise added to `y`, floor = `sigma_n` plotted as a horizontal line the val RMS descends to). Added measurement noise per output channel; success = augmented val sim-RMS reaches the noise floor `sigma_n`. Chosen level: **SNR = 50 dB** primary, sweep **55 and 60 dB** to locate the plateau. Signal levels backed out of run 68676's baseline table (`std(y) = RMS_error / NRMS`): X1 0.060 m, X2 0.065 m, Y 0.230 m. Resulting per-channel floor at 50 dB: `sigma_n` = X1 1.9e-4, X2 2.0e-4, Y 7.3e-4 m; aggregate val sim-RMS floor = **4.5e-4 m** (aggregate confirmed to be `sqrt(mean of per-channel MS)`, matches the printed 5.175e-4). Runs are **JOINT_ESTIMATION off** and use the **linear** augmentation (D-071), so parameter fitting cannot soak up the residual and a memoryless shortcut cannot fake the absorber's dynamic contribution. Noise is injected in **Python at the output** (reproducible, sweepable, exactly measurement noise), NOT in the MATLAB generator.
**Why**: In noiseless simulation there is no acceptance line: error can crawl toward 0 indefinitely (Jan: numerically sensitive, slow), and any value hit has a smaller one below it, so there is no pass/fail. Random noise cannot be predicted by any model and does not average out of the error, and neither does uncaptured deterministic content, so the error bottoms at `sqrt(unmodeled_deterministic^2 + noise^2)`: reaching `sigma_n` certifies all learnable signal (incl. the absorber) was captured to below the noise; plateauing above it quantifies the uncaptured content (the D-068 closed-Y-row routing ceiling) in absolute NRMS. The SNR is pinned to the measured baseline residual (NRMS ~0.0031-0.0037 val -> crossover 49.4 dB) so the floor sits BELOW the unmodeled content: at SNR 50 the baseline lands at 1.53x the floor (cannot reach without improving), the current augmentation at 1.19x (nearly there), giving a clean, falsifiable, achievable separation. The noise does not make learning easier; it converts an open-ended optimization into a bounded one with a known optimum, which is the entire value.
**Ruled out**: (1) Oracle-model floor (baseline-vs-oracle NRMS as the target): numerically valid but model-dependent, simulation-only, and not defensible (no oracle exists on hardware) - user rejected it explicitly; a threshold must be model-free and data-derived (lessons.md). (2) Jan's low SNRs 20/30 (floors 10%/3.2%): far ABOVE our baseline residual (~0.3-0.4%), so the untrained baseline already sits on the floor - reaching it is trivial and proves nothing. His grid is calibrated to his deliberately under-modeled 2-DOF-approx-3-DOF baseline; our near-perfect FP baseline needs SNR ~50. (3) 40 dB (Jan's loose "40 dB ofzoiets"): floor NRMS 0.01, still above the val residual, uninformative for our system; it is at best a top anchor. (4) A dedicated SNR helper function: the level is a two-line inline op (per-channel `sigma_n`), wrapping it is scaffolding (lessons.md: no operational scaffolding in the experiment script). (5) Estimating `sigma_n` from the data via the standard nonparametric methods (period variance, non-excited multisine lines; Pintelon & Schoukens): correct for REAL measured data where noise exists and is unknown, but the current phase is a NOISELESS simulation where there is nothing to estimate - those estimators belong to the future real-gantry phase, and the simulator's role there is to validate the estimator (inject known `sigma_n`, repeated periods, confirm recovery). (6) Injecting noise in the MATLAB generator: process/closed-loop noise is not what the floor criterion needs (open-loop pipeline) and baking one realization into the dataset loses reproducibility/sweepability.
**Constrains**: Output-floor is necessary but NOT sufficient for the state-interpretability claim (the augmented states could carry the absorber in a rotated basis, or a nonlinear correction could fake part of it), so the full acceptance test stays TWO-AXIS: output val sim-RMS reaching `sigma_n` AND augmented-state R2_linmap vs the oracle absorber (>~0.9), the latter measured NOISELESS (noise only lowers the achievable R2 ceiling). The linear augmentation couples the two axes on Y (a memoryless linear map cannot add the absorber's pole pair), so a run that reaches the Y floor should also raise R2_linmap; if it reaches the floor with R2 still ~0, the routing (Option A/B) is binding. Per-channel `sigma_n` is mandatory (X1/X2/Y amplitudes differ ~4x, MEET-05); a single global SNR would give three different effective SNRs. Numbers derive from run 68676, which is a JE-ON pilot: the baseline table is JE-independent (nominal params) so the floor is solid, but the augmented clean error (2.94e-4, the 1.19x) must be re-confirmed on a JE-OFF linear run before claiming the separation. Margin note: at SNR 50 the floor is only ~1.15x below the baseline content (tight); SNR 55 (aggregate floor ~2.5e-4 m) gives cleaner headroom and is the safer primary if the 50 dB separation proves borderline in practice.

### [D-077] Joint estimation v2: all 14 raw physical parameters trainable (train raw, trust combinations)
**Date**: 2026-07-05
**What**: `Parameterized_Gantry_State_Block` extended from the 5 damping/stiffness sums (D-076 v1) to ALL 14 raw physical scalars, mirroring `lpv_lfr_baseline/train_param_recovery.py` exactly. `PARAM_NAMES = [kb1, kb2, cg1, cg2, cy, cb1, cb2, mh, m1, m2, mb, Jb, Jh, d]`, each log-reparameterized (D-035) and detuned individually; only `Lb` stays frozen (it defines the coordinate frame via P, not the M(Y) rational structure). Because masses are now trainable, `nonlinear_function()` rebuilds the ENTIRE M(Y) structure per timestep from the parameters: `gantry_ss.build_poly_constants` -> alpha/beta/gamma/N0/N1/N2, `d0 = mh(alpha*gamma - beta^2)`, M1/M2 from mh, K/C from the stiffness/damping (using only the identifiable sums kb1+kb2, cb1+cb2), then `build_G_matrix_entries` -> A_combined. The parent `_mats()` hook is widened from `(K, C, A_combined)` to also carry `(mh, alpha, beta, gamma_, N0, N1, N2)`, and `deriv()`'s LPV branch reads all of them from the hook instead of as buffers (behavior-neutral for the fixed block, A0-guarded). Reporting (`identifiable_combinations()`, `param_table()`) exposes only the 10 data-identifiable quantities [kb_sum, cg1, cg2, cy, cb_sum, mh, m_total, m_diff, J_eff, d]; `m_diff = m1 - m2` is a SIGNED derived readout of the individually-logged (positive) m1, m2 — never itself a parameter.
**Why**: v1 froze masses to keep M(Y) constant and dodge the invertibility question. The Task 3.1 proof (D-077 companion) shows M(Y) is positive-definite for ALL Y provided every mass/inertia/geometry scalar > 0, which the log-parameterization guarantees by construction — so all 14 can be trained safely. Training the raw params does not "mess up" training even though only combinations are trusted: the non-identifiable splits (kb1 vs kb2, cb1 vs cb2, mass flat direction) are FLAT (zero data-gradient) and rest at their `param_loss` anchor; only the identifiable combinations receive gradient signal. "Train raw, trust combinations" is exactly the train_param_recovery design; the combination view is a reporting choice applied after training, not a parameterization constraint.
**Ruled out**: parameterizing the combinations directly (m_diff is signed — cannot be logged; unnecessary because it is derived from logged m1, m2); keeping a 5-vs-14 selector flag (speculative flexibility against minimal — freezing a subset can be a future flag if ever needed); a 5-param and 14-param class coexisting (v1 superseded, its results preserved in D-076); mutating parent buffers in the child to inject trainable matrices (breaks autograd/state_dict — the `_mats()` hook is the clean seam); rebuilding once per forward instead of per timestep (needs a forward-boundary hook Jan's step-by-step block calling does not cleanly provide; the redundancy is only the ~10% overhead and A1/gate measure it).
**Constrains**: Runtime ~+10% over v1 (est. ~+20% over the fixed block; the M-gate/D-timing measure the exact figure before any long run). Gate updated: A1 now also validates the full nominal M(Y) rebuild; B covers all 14 log_params; NEW check M samples the positive parameter orthant and asserts `M(Y) @ N(Y)/d(Y) = I` (inverse-consistency, off-nominal transcription guard) AND `eig(M(Y)) > 0` (PD, proof realization). Check D judges recovery on the 10 identifiable combinations: the well-conditioned subset {kb_sum, cg1, cg2, cy, cb_sum} is gated <= 0.5 (regression vs v1), mass combos are reported and only guarded against divergence (their identifiability from short data is the open run-design question, not a correctness gate). Same scientific-scope caveat as D-076: on this benchmark FP params are true by construction, so v2 JE is machinery validation + bias-demonstration, not a fix for the absorber output-reachability issue.
**Gate results (2026-07-05)**: Gate 1 PASS — A0 and A1 bitwise 0.0 (the full nominal M(Y) rebuild reproduces the fixed block exactly); B PASS on all 14 params under the gradcheck-style tolerance `|auto-fd| <= atol(1e-9) + rtol(1e-5)|fd|` (pure-relative would fail only on Jh, gradient ~2.7e-7, where autograd and central-difference agree to 4 sig figs — a roundoff floor, not an error; A1 + the M-check independently verify the Jb/Jh path); NEW check M PASS — `max|M·N/d - I| = 6.66e-16`, `min eig(M) = 2.97 > 0` over 200 positive-orthant samples x 7 Y (inverse-consistency + PD verified off-nominal). Gate 2 PASS — C loss integration rel 1.16e-9; D recovery on the state-readout config (C=I, flag_loss_reg=False) recovered ALL 10 identifiable combinations, not just the gated v1 subset: kb_sum 0.008, cg1 0.141, cg2 0.052, cy 0.019, cb_sum 0.033 (gate <= 0.5), and the reported masses mh 0.001, m_total 0.000, m_diff 0.002 (detuned +1.59 -> learned -0.4961, truth -0.5, signed), J_eff 0.028, d 0.000 gap-ratio. **Measured overhead**: parameterized-block forward +15.9% vs the fixed block (v1 was +11.7%, so ~+4% forward vs v1); fwd+bwd 1.49 s/batch vs v1's 1.09 s (+37% on the training step in isolation, because backprop through the per-timestep mass rebuild is costlier than forward alone — smaller net effect in the real pipeline where the ANN/encoder/longer windows dilute the physics-block share). Script edits (14-nominal `build_model` derivation from `gantry_ss`, `param_table()` report) validated via a no-train `build_model` rehearsal: params_init exact, combination table correct. Flag-off anchor (JE=False initial validation sim-RMS 6.4948e-4) is guaranteed unchanged by A0 (fixed-block deriv bitwise identical) and by the JE=False path not touching any new code; not re-run. Artifacts: `simulations/gantry_subnet/diagnostics/joint_estimation/` (gate1/gate2 JSON, gate_v2*.log).

### [D-076] Joint estimation in the multisine pipeline: Parameterized_Gantry_State_Block + generic param_loss trainer
**Date**: 2026-07-04
**What**: Three additions enabling joint estimation of physical parameters in `gantry_interconnect_dynamic.py`:
(1) `Parameterized_Gantry_State_Block` in `model_augmentation/fit_systems/blocks.py`, placed directly below `Gantry_State_Block` (mirroring Jan's fixed/parameterized adjacency, marked `@added`). Subclass with trainable vector `[kb_sum, cg1, cg2, cy, cb_sum]` stored as `log_params = nn.Parameter(zeros)` meaning log(theta/params_init) (D-035 positivity, MEET-02 centering: all params start at 1 in normalized space); regularization `param_loss()` with `Lambda = RMSE_baseline / params_init` toward `params_init` in physical space (D-034); `params_init` constructor override exists for detuned recovery tests. Per timestep the block rebuilds K and C (`torch.stack` construction, pattern copied from `lpv_lfr_baseline/blocks/lfr_param_block.py`, NOT imported) and `A_combined` via `gantry_ss.build_G_matrix_entries` — the functions kept autograd-safe for exactly this call. The parent `Gantry_State_Block` gains a ~3-line `_mats()` hook returning `(K_mat, C_mat, A_combined)` read by `deriv()`, so the child overrides matrices without duplicating the deriv kernel. New child buffers: `d0`, `M1`, `M2`.
(2) `SSE_Interconnect_ParamLoss` in `model_augmentation/fit_systems/interconnect.py` (`@added`): delegating `loss()` = `super().loss(...) + sum(m.param_loss())` over blocks exposing `param_loss` (hasattr sweep). Reimplementation of the D-032 idea from `lpv_lfr_baseline/blocks/lfr_fit_system.py`; deliberately NOT imported from there (no cross-pipeline dependency — user decision 2026-07-04). Used unconditionally in the script: exact no-op when no block exposes `param_loss`. Documented caveat: would double-count Jan's `Parameterized_Linear_*` blocks if ever combined (never used in this pipeline).
(3) `JOINT_ESTIMATION` flag in `gantry_interconnect_dynamic.py` gating ONLY the block class; `PARAM_RMSE_BASELINE = 0.01` constant (HEURISTIC: measured initial sqrt-loss of jobs 68675/68676); flag-guarded learned-vs-nominal parameter printout plus `params_init`/`params_learned` fields in the results npz; clear error when `RESUME_CHECKPOINT` points at a pre-JE checkpoint while the flag is on.
**Why**: Joint estimation machinery is the prerequisite for the gray-box absorber path (Option A applies the same log_params/param_loss pattern to a 3-scalar absorber block) and for real hardware where nominal parameters are uncertain. v1 trains damping+stiffness only: none of these enter M(Y), so every M(Y)-derived parent buffer (N0/N1/N2, Horner d(Y) constants, M0inv, Bw, Bu) stays constant and valid, and the Task 3.1 M(Y)-invertibility proof is untouched. Sums (kb_sum, cb_sum) are parameterized directly because only the sums are identifiable (flat-ridge analysis in lfr_param_block).
**Ruled out**: importing `LFRFitSystem` from `lpv_lfr_baseline` (wrong dependency direction); a new module or experiment script inside `model_augmentation/` (experiments live in `scripts/`, blocks belong beside their fixed siblings); duplicating `deriv` in the child (~35-line maintenance hazard, replaced by the `_mats()` hook); hand-assembled Ax/Bw/Bu in the child (second copy of the G-matrix math, replaced by reusing `build_G_matrix_entries`); in-script Lambda auto-calibration (`set_RMSE_baseline` machinery is operational scaffolding — a measured constant with provenance suffices; log-space centering makes Lambda scale non-critical); flag-gating the trainer class (delegation makes the subclass a provable no-op when unused).
**Constrains**: Verification gates precede any full run; diagnostic results go to `simulations/gantry_subnet/diagnostics/joint_estimation/`. Gate 1 (after block): A0 = refactored fixed block reproduces a reference batch captured BEFORE the refactor; A1 = parameterized block at `log_params=0` matches the fixed block (~1e-6 float32); B = finite-difference vs autograd gradients for all 5 params (float64). Gate 2 (after trainer): C = one `loss()` call equals MSE + param_loss on a minimal no-ANN interconnect; D = mini recovery on self-generated absorber-free data (fixed block, nominal params, real multisine u), `params_init` detuned ±10%, pass = gap to nominal shrinks ≥50% for all 5 params; it/s from D doubles as the runtime-overhead measurement. Then the manual 1-epoch rehearsal (D-071 procedure) with the flag on, and the flag-off anchor check (initial validation sim-RMS must remain exactly 6.4948e-4). Scientific scope note: in THIS benchmark the FP parameters are true by construction, so JE of FP params serves machinery validation and bias-demonstration ablation only — it cannot address the absorber state-learning issue (output reachability unchanged) and is expected to bias parameters if trained on absorber-containing data. JE runs start from fresh checkpoints (old .pt files lack `log_params`).
**Gate results (2026-07-05)**: Gate 1 PASS (A0 and A1 bitwise 0.0; B max FD-vs-autograd rel err 2.5e-7). Gate 2 PASS with a redesigned check D: two recorded failed attempts showed that the ORIGINAL check-D configuration could not isolate the machinery — (attempt 1, random default encoder + sim-RMS validation) the K=0 horizon-mismatch checkpoint trap restored epoch 0 and all 15 epochs went into encoder learning; (attempt 2, Hoekstra-style encoder + windowed validation) the co-trained encoder absorbed the loss (down 1000x with parameters frozen at init; nominal and detuned params gave IDENTICAL initial loss 2.0921, proving the loss was encoder-driven). Final check D therefore isolates parameter learning: synthetic output = full state (C = I), exact parameter-free readout encoder, flag_loss_reg=False (check C separately proves the regularization path, exact to rel 4e-10). Result: gap ratios kb_sum 0.000, cg1 0.211, cg2 0.189, cy 0.001, cb_sum 0.055 (pass <= 0.5); parameterized-block forward overhead +11.7% (fwd+bwd 1.09 s/batch, nf=100, batch 128). **Run-design findings for real JE (input to Phase 3/4 and the Jan discussion)**: (1) with position-only outputs, short-window BPTT and a co-trained encoder, physical parameters are practically unidentifiable — the encoder compensates; (2) param_loss anchored to the init values actively pins parameters there once the MSE landscape flattens; (3) windowed validation is mandatory on this plant (sim-RMS checkpoint selection reproduces the documented horizon-mismatch trap even without an ANN). Artifacts: `simulations/gantry_subnet/diagnostics/joint_estimation/` (gate1/gate2 JSON, gate_run*.log including the failed attempts).

### [D-075] Telica train/validation/test wiring: supervisor's split, iter0+iter8, SEGMENT_LEN 650 confirmed
**Date**: 2026-07-04
**What**: `run_telica_param_recovery.py` switched from single-trajectory to the supervisor's
split (folders under `06 40 mm XL 80 mm YL/`, split by operating point): TRAIN = 11 OPs x
{iter0, iter8} = 22 trajectories; VALIDATION = 2 OPs x {iter0, iter8} = 4; TEST = 2 OPs x
{iter0, iter8, iterTEST} = 6, final evaluation only. IDs T1a..T11b / V1a..V2b / E1a..E2T
(a = iter0, b = iter8, T = iterTEST); `tr._traj_set_tag` is monkeypatched to '22traj' to
keep the checkpoint filename inside the Windows 260-char path limit. EPOCHS = 40,
VALIDATION_INTERVAL = 5, SEGMENT_LEN = 650 (re-picked consciously per D-073: 32.5 ms at
the true 20 kHz spans 6+ periods of the ~200 Hz servo band and 27 periods of the 845 Hz
notch resonance), NORM_MODE 'global', FULL_COVERAGE, no overlap. Final evaluation loops
open-loop AND closed-loop (`_post_eval` + `_post_eval_cl`) over a train sample (T1a/T1b),
all validation and all test trajectories.
**Why**: iterations within an OP share the same reference and differ only in feedforward;
iter0 (feedback-dominated, transient-rich) and iter8 (converged ILC, smooth) are the two
extreme input spectra; the 7 in between are near-duplicates (5x cost, little information).
Operating-point diversity, not iteration count, drives LPV identifiability. iter6_1.log
(redo artifact) excluded.
**Deviation from the stated plan**: the framework's built-in validation
(`_full_traj_eval`) is a FULL-TRAJECTORY OPEN-LOOP RMSE on the validation trajectories,
not the windowed training measure. It is controller-free and it is the metric we
ultimately care about for OL quality, so checkpoint selection and LR scheduling use it
as-is rather than adding a windowed-validation code path.
**Ruled out**: all ~110 iterations (redundant, ~5x runtime); iter0-only (single input
character); windowed validation implementation (extra code path in the training script
for marginal benefit).
**Constrains**: Runtime estimate revised: FULL_COVERAGE gives ~13 gradient steps/epoch at
batch 22 (not one batch of 294), so an epoch is ~30-40 s CPU; 40 epochs + 9 validation
passes = roughly 30-45 min training, plus ~30-60 min for the 12 OL + 12 CL final
evaluations. TEST trajectories are also evaluated open-loop by tr.train()'s own Step 4
at the very end (after best-checkpoint restore); they never influence training.

### [D-074] Closed-loop validation added to run_telica_param_recovery.py; training stays open loop
**Date**: 2026-07-03
**What**: Three additions, no change to the training path: (1) `telica_loader.load_telica_log_cl`
returns r [m], q1 [m], u_ff [N] ((MF30-MF230)*Kt) and logged i_fb [A] (MF230) on the same
grid/trim as the training loader. (2) New `telica_controller.py`: `TelicaFeedbackController`,
per-sample direct-form-II-transposed stepper for the LX1/LX2/LY controllers from
`dFeedbackControllersTelica_ba.mat` (exported from the supervisor's zpk file); self-test
verifies bit-exactness vs scipy lfilter and replays iter0 (corr 0.96-0.97 against logged
MF230; known amplitude offsets: time-domain LS 1.16/1.33/3.55, coherent-band 0.74/1.08/1.23
per D-073, attributed to an unmodeled rail decoupling transform). (3) `_post_eval_cl` +
`_run_closed_loop` appended to the runner: initial condition only, then full trajectory with
u = u_ff + Kt * K(r - y_model) stepped through rk4_step under no_grad; reports the same
RMS/NRMSE table as the open-loop eval plus a feedback-current plot with a built-in wiring
check (controller fed with measured error vs logged MF230). Called for baseline and best in
`__main__`, next to the open-loop evals.
**Why**: Supervisor decision: train open loop (windowed BPTT parallelizes; a controller in
the training loop forces one long sequential rollout), validate closed loop (the
control-relevant metric; a model is good if it behaves like the plant inside the same loop).
Timing convention: controller acts on the same-sample error, no computational delay; the
output has no feedthrough (y = Cy x), so the loop is well-posed.
**Ruled out**: CLOE training (controller inside the gradient path): runtime and complexity
not justified while open-loop windowed training suffices; everything learned here (loader,
controller module, conventions) is reusable if it becomes necessary. Closed-loop-only
validation: CL suppresses model error inside the bandwidth, so it must always be read next
to the open-loop numbers.
**Constrains**: CL numbers are only comparable between models evaluated with the same
controller file. Smoke test (untrained Kamtin-parameter block on iter0): the loop diverges
(~3 m RMS), which is the expected verdict for a plant-mismatched model under the real
controller, not a wiring bug (the replay check inside the same run is clean).
**Verified (diag_cl_correctness.py, 2026-07-03)**: (1) controller bit-exact vs MATLAB
filter(tfdata) on the original zpk (0.0 deviation; the 2.15e-5 deviation vs lsim is
MATLAB-internal tf-vs-ss conditioning of the unit-circle integrator poles); (2) perfect-model
test: same-x0 repeat exact, rebuilt-x0 NRMSE 0.006-0.03% with a gain-scaled (stable)
controller; (3) OL replay of the CL force reproduces CL positions exactly; (4) timing: the
logged MF230 lags K(logged M2) by a BROAD ~2.5 ms correlation maximum; a 2.5 ms in-loop delay
is physically impossible (the ~300 Hz-crossover loop would be unstable), so it is a
logging-path artifact and the zero-delay loop is correct; (5) all six controllers marginally
stable by design (integrator poles at |z|=1 within zpk rounding 6e-8), logged currents max
~6.1 A vs 27.9 A peak limit, so no saturation modeling needed; controller tuning consistent
across iterations and operating points (replay corr/scale identical on train and validation
positions).

### [D-073] Telica iter logs are 20 kHz native; loader upsampling removed (supersedes D-061)
**Date**: 2026-07-03
**What**: `telica_loader.py` now uses `fs_native = 1/SamplingTime = 20000 Hz` and performs
no resampling (a guard raises if native rate and pipeline rate ever diverge). The previous
chain (assume 10 kHz native, linearly upsample 2x to 20 kHz, D-061) stretched the time axis
by a factor 2: every Telica training and evaluation before this date fitted a plant with
2x slowed dynamics and is not comparable to later runs.
**Why**: Controller-notch fingerprint (`diag_controller_fingerprint.py`). The real Telica
controllers (dFeedbackControllersTelica.mat from the supervisor, 6x6 diagonal zpk at
Ts = 5e-5 s; axis order confirmed by supervisor: LX1, LX2, LY, RX1, RX2, RY) have notches
at fixed normalized frequencies of the 20 kHz DSP. In the iter0 empirical FRF
(M2 -> MF230, exact controller I/O pair since feedforward = 0), the X1 notch appears at
normalized frequency 0.10 of the LOG, exactly where LX1 has it under the 20 kHz
interpretation; under the 10 kHz interpretation it would appear at 0.20, where the data
shows nothing. Shape residuals: X1 3.2 dB (20 kHz) vs 5.5 dB (10 kHz); gain scales at
20 kHz: X1 0.74, X2 1.08, Y 1.23. Telica.mat SamplingFrequency description ("The number
of samples logged per second" = 20000) agrees. Figures:
`simulations/gantry_subnet/diagnostics/controller_fingerprint/`.
**Ruled out**: FsHz = 1/(2*TsSec) = 10 kHz from runFDILCAllHostSwLog.m line 30 (basis of
D-061): that formula belongs to a different logging configuration; the data itself
contradicts it for the iter*.log files.
**Constrains**: All pre-2026-07-03 Telica training results are invalidated for comparison.
`SEGMENT_LEN = 650` samples now means 32.5 ms instead of 65 ms: re-pick consciously before
the next training run. Diagnostic scripts hardcoding `_FS_NATIVE = 10_000`
(diag_cloe_signals.py) predate this finding. Remaining gain offsets (0.74/1.08/1.23) and
corr ~0.97 are consistent with a decoupling transform around the SISO controllers
(rail cross-coupling), acceptable for closed-loop validation. Related earlier decisions
D-069 (diagnostics) and the AeroPro finding (Telica.mat MachineType = "AeroProCoC").

### [D-072] Baseline comparison matrix: oracle-x0 and encoder-init baselines, revived oracle model sim, aligned averaging windows
**Date**: 2026-07-03
**What**: Four changes to `gantry_interconnect_dynamic.py` making every baseline/model comparison a well-posed cell of a matrix {baseline FP, augmented model} x {true x0, encoder init}: (1) `compute_baseline_fp_nrms` gains `x0_norm`, `start_ix`, `avg_from` parameters. (2) NEW encoder-init baseline: the baseline FP is seeded with the state estimated by the UNTRAINED `linear_encoder_init_aug` (Hoekstra reconstructability map W^b, built from the baseline's own linearization) from the first measured I/O window, simulating from sample k0=max(na,nb); computed for val and E1, printed with explicit labels distinguishing 'true x0' from 'encoder-init', stored in the results npz. Using the untrained map is deliberate: before training it is purely baseline-derived (no co-training with the augmented dynamics), so 'baseline + linear init' is well-defined; the trained encoder would not be. (3) The x_logical-initialised model simulation (augmented model from true x0) is revived: it was dead code because it checked `val_data.x` which `load_traj` never sets; it now seeds from `val_x_logical[0]` (always loaded). Its NRMS is now averaged over `[cheat_n:]` like the encoder-init model metric. (4) All baseline averaging windows aligned to k0 (`avg_from=k0` for oracle baselines, `start_ix=k0` for encoder-init), removing the ~0.2% asymmetry where the baseline was averaged over all N samples but the model over N-cheat_n.
**Why**: The pre-existing comparison was baseline(true x0, full window) vs model(encoder init, cheat_n window) — biased in the baseline's favor. Conservative for improvement claims, but on the K=0 axes initial-state errors do not decay (they integrate into position drift over the full horizon), so the bias understates the model-vs-baseline gap by an encoder-quality-dependent amount rather than a negligible one. The completed matrix separates model quality (true-x0 vs true-x0) from encoder contribution (model true-x0 vs model encoder-init) from realistic end-to-end performance (encoder-init vs encoder-init).
**Constrains**: Oracle baseline NRMS values shift slightly vs earlier logs (averaging now starts at k0). k0=max(na,nb) may differ from deepSI's cheat_n by one sample — negligible and documented here rather than plumbed through.

### [D-071] Linear parallel augmentation experiment (Jan's ECC config) + E1 generalization evaluation + smoke-test hook
**Date**: 2026-07-02
**What**: (1) `ANN_ACTIVATION` default switched from 'tanh' to 'linear' (Identity activation, Jan's `linear_parallel` ECC configuration) for the next training run. (2) `evaluate_and_save` extended with a test-set (E1) simulation: per-channel NRMS plus baseline-FP comparison on the unseen excitation; `compute_baseline_fp_nrms` generalized with `(data, x0_phys, label)` arguments so the baseline can be computed on E1 as well. (3) ~~`SMOKE_TEST=1` environment hook~~ — implemented, then REMOVED at user request (no operational scaffolding in the experiment script). The rehearsal is now a manual procedure: before a long submission, temporarily set epochs=1/nf=10, run the script end-to-end once (fresh + resume), revert, submit. Both rehearsals were executed on 2026-07-03 and passed (exit 0), validating the D-070 checkpoint save, the resume load, and the E1 evaluation.
**Why**: Run 68597 (tanh ANN, D-068 routing) improved aggregate val sim-RMS by ~43% vs the baseline FP but R2_linmap(delta_a) stayed at 0: the ANN compensates memorylessly from the instantaneous state instead of learning the absorber. A tanh static correction can imitate much of the absorber effect; a LINEAR static correction cannot add the missing pole pair, so any error reduction at the absorber resonance must flow through the two augmented states. The hidden absorber is itself LTI (Y-scheduling enters only via the fixed LPV FP block that propagates the corrections downstream), so a linear augmentation is the correct residual class, not a restriction. The E1 evaluation separates compensation from captured dynamics: a memoryless compensator tuned to training-excitation correlations degrades on unseen excitation, a learned oscillator transfers.
**Ruled out (for now, staged behind this run)**: (1) Gray-box absorber via `Parameterized_Linear_State_Block`: guarantees state meaning by construction but changes the model class and touches the D-068 routing question (reopening the Y row); escalation if the linear run keeps R2 near 0, after consulting Jan. (2) LPV-linear augmentation (correction linear in [x,u] with coefficients affine in Y): restores scheduling without adding dynamics; refinement if states learn but an error gap vs the tanh run remains. (3) Supervised x_aug loss on saved delta_a ground truth: simulation-only scaffold, does not transfer to real data.
**Constrains**: The outcome is diagnostic in both directions: R2 rises means the memoryless shortcut was the blocker; R2 stays near 0 means the closed Y injection row (D-068) is binding and the gray-box path becomes necessary rather than optional. Full capture of the MSD effect is structurally impossible while the Y row is closed (the absorber force reaches Y only via the Theta inertial coupling M0[1,2]=-mh*d, with collateral X1/X2 cost). The augmented loop has exactly zero gradient at init (zero-init final layer), so use more epochs than 30 for this run.

### [D-070] Weights-only training checkpoints via component state_dicts, saved before diagnostics
**Date**: 2026-07-02
**What**: Four changes to `train_model`/`train_model_with_diagnostics` in `gantry_interconnect_dynamic.py`: (1) Checkpoint save changed from `torch.save(fit_sys.state_dict(), ...)` to a dict of component state_dicts `{'hfn': ..., 'encoder': ..., 'optimizer': ...}`. `SSE_Interconnect` inherits from deepSI `System` (not `nn.Module`) and has no `state_dict`; only `fit_sys.hfn` (Interconnect) and `fit_sys.encoder` are torch modules, and together they hold all trainable parameters. (2) Resume path loads these component dicts into the model built by `build_model(hp)`; optimizer state included so Adam moments continue instead of restarting. (3) The `.pt` checkpoint is written immediately after `fit()` returns, BEFORE `aug_state_r2`, and the diagnostic is wrapped in try/except (NaN placeholders on failure), so no post-training step can lose the weights. (4) ~~`fit()` called with `verbose=1` under SLURM~~ — implemented, then REVERTED on 2026-07-03 at user request: the tqdm progress bar is how long cluster runs are monitored (ETA, it/s); log length is not a defect.
**Why**: SLURM job 68597 (30 epochs, 11 h, first successful D-068 run) crashed at the old save line with `AttributeError: 'SSE_Interconnect' object has no attribute 'state_dict'` after training completed. Because the save ran before `evaluate_and_save`, all artifacts (model save, results npz, plots, state recovery diagnostic) were lost; the best weights survived only in deepSI's internal `~/.deepSI/checkpoints/SSE_Interconnect_<code>_best.pth`. The resume path (`fit_sys.load_state_dict`) had the same bug and would have failed on first use.
**Ruled out**: `fit_sys.save_system()` whole-object pickle (`torch.save(self, file)`): works and is already used in `evaluate_and_save` for the final model, but deepSI's own docstring warns it is "quite unstable for long term storage or switching between versions", and it would replace the build-then-restore resume design rather than fit into it.
**Constrains**: Checkpoint `.pt` format is now the component dict; resume requires `build_model` with the same hp (already guaranteed: hp is read from the checkpoint `.npz` meta). No backward compatibility needed: the old save line never executed successfully, so no old-format checkpoints exist.

### [D-069] Controller reconstruction gain mismatch: three diagnostics before CLOE
**Date**: 2026-07-02
**What**: Three diagnostic scripts in `scripts/gantry/real-data-verification/` to resolve the
11-22x amplitude mismatch between the documented controller chain (M2[um] x 1024 cnt/um ->
Filter1 -> Filter2 -> x AmplifierGain 0.002075 A/DAC) and the logged feedback current MF230:
(1) `diag_log_rate.py`: resolves the log-rate ambiguity (10 kHz per D-061 vs 20 kHz per
Telica.mat SamplingFrequency description "The number of samples logged per second") by
locating known filter features (notches, integrator slope) in the empirical M2 -> MF230 FRF
on the normalized frequency axis. A filter feature at normalized frequency nu (designed at
20 kHz) appears at nu in the log FRF if the log is at 20 kHz, at 2*nu if decimated to 10 kHz.
No timestamps used (D-061 forbids them).
(2) `diag_frf_controller.py`: extracts the controller actually active during the FRF campaign
via K_eff = G^-1 (S^-1 - I) from frfPlant [cnt/dac] and frfSensitivity in Telica.mat
(THEORY: S = (I+GK)^-1, Skogestad & Postlethwaite 2005 Ch. 2). Compares K_eff against
Filter1*Filter2 per frequency. This is excitation-based and free of the closed-loop
correlation concern; a flat ratio identifies the missing per-channel gain and its value.
(3) `diag_iteretel_decode.py`: dumps the full column schema of iterETEL.log and iter0.log
(iter0 has 25 columns, only 13 identified so far; DatalogListVarMapping lists 25 ETEL
channels including X_HIGS_INPUT/X_HIGS_OUTPUT, X_FB_OUTPUT, X_ENC_POS, X_DAC) and runs
conditional analyses: HIGS input/output scatter (gain-mode slope) and raw-unit chain checks.
**Why**: Telica.mat is now fully read (MATLAB batch confirmed a single top-level variable);
no additional scale parameter exists in it. The mismatch is real (reproduced independently in
MATLAB with native filter()). The DatalogListVarMapping names HIGS blocks in the servo loop:
a HIGS (hybrid integrator-gain system) between error and filters acts approximately as a
per-channel constant gain, which fits every observation (corr near 1, constant ratio per
axis, different ratio per axis). A static-gain workaround was rejected (lessons.md): the
missing element must be identified, not approximated away.
**Ruled out**: LS scale from iter0 time-domain fit as the final answer: it conflates the
missing gain with rate misapplication effects (11x at 10 kHz native vs 22x at 20 kHz
upsampled shows the estimate is method-dependent). Asking Kamtin first: these diagnostics
use data already on disk and can fully resolve the question; ask only if they fail.
**Constrains**: CLOE implementation is gated on the missing gain being explained and
reproduced (reconstruction matching MF230 in iter0 within a few percent). Results go to
`simulations/gantry_subnet/diagnostics/`.

### [D-068] Route ANN only to states with spring stiffness (Jan's state_augment_specific_states)
**Date**: 2026-07-01
**What**: Change `build_model()` in `gantry_interconnect_dynamic.py` to route ANN corrections only to state rows with K > 0, instead of all `nxd` rows. For the gantry: `STIFF_IX = [1, 4, 6, 7]` (Theta position, Theta velocity, delta_a, vdelta_a). ANN output width changes from `nxd=8` to `len(STIFF_IX)=4`. Implementation uses `expansion_matrix(STIFF_IX, nxd)` per Jan's `state_augment_specific_states` API.
**Why**: Jan confirmed K=0 gantry axes (X, Y) cause ANN correction accumulation and suggested this fix. X (index 0,3) and Y (index 2,5) have K=0 — additive corrections accumulate without restoring force (O(N) drift). Theta (index 1,4) has kb1+kb2 stiffness; absorber states (index 6,7) have absorber spring. Routing only to K>0 states eliminates drift at the source. This is the physically motivated fix within Jan's framework.
**Ruled out**: (1) Full-state routing (D-067): K=0 axes accumulate drift — existing failure mode. (2) Velocity-only routing (D-066): velocity corrections still integrate to position drift under K=0. (3) Aug-only routing (D-065): C_aug gradient dead zone.
**Constrains**: ANN output dim becomes 4. Absorber-to-X/Y coupling is not directly captured (X/Y rows excluded). First test should use single-stage sim-RMS to confirm K=0 blowup is eliminated before revisiting curriculum design.

### [D-067] Revert to Jan's full-state routing + curriculum nf training
**Date**: 2026-07-01
**What**: Reverted `gantry_interconnect_dynamic.py` from Model B (velocity+aug rows) back to Jan's full-state routing (`connect_block_signals(ann_block, ["x","u"], ["xp"])`), matching `msd_ndof_interconnect_dynamic.py:91`. ANN `nw` reverted from 5 back to `nxd=8`. `VEL_AUG_IX` removed. `DIAG_INTERVAL` replaced by `NF_CURRICULUM`: a 6-stage curriculum schedule `(nf, epochs, validation_measure)` progressing 25→50→100→200→400 (windowed) → 400 (sim-RMS). `train_model` signature extended with optional `nf` and `validation_measure` overrides. `train_model_with_diagnostics` iterates `NF_CURRICULUM`, logging R2_linmap after each stage.
**Why**: Model B training failed: best checkpoint = epoch 0 (untrained) on all 20 epochs. Root cause is the K=0 + training/validation horizon mismatch: ANN learns velocity corrections that reduce nf=400 training loss but cause O(N_val/nf)=20× larger position drift on full 8000-sample validation. Excluding position rows from routing does not fix this — velocity corrections still integrate to unbounded position drift under K=0. Full-state routing is correct (same as Jan's working MSD implementation); the K=0 instability is addressed via curriculum nf. Literature precedent: CHyLL (arXiv:2512.10117) shows direct training at long nf diverges while curriculum converges; Farina & Piroddi (2011) establishes sub-sequence length as critical hyperparameter; Uy et al. (arXiv:2212.01418) demonstrates rollout training suppresses drift on marginally stable systems.
**Why curriculum fixes K=0**: At small nf, position drift per window O(nf·ε·Ts) is small — ANN learns absorber oscillation. Absorber displacement is physically zero-mean, so correctly learned corrections are also zero-mean and don't cause net long-horizon drift. Increasing nf progressively forces the ANN to maintain zero-mean corrections. Final sim-RMS stage continues training using dynamics already learned at nf=400, not just evaluates — the model adapts to the full-trajectory regime.
**Ruled out**: (1) nf=1000+ windowed validation — more expensive than sim-RMS (7M vs 8k steps). (2) Stay with Model B — doesn't fix K=0 drift, only fixes C_aug dead zone. (3) Increase nf to 8000 directly — computationally infeasible, equivalent to CHyLL failure mode. (4) Series augmentation — excluded per project scope.
**Constrains**: NF_CURRICULUM controls all training. Optuna objective unaffected (uses `train_model` directly). Model B (D-066) is superseded.

### [D-066] Model B routing: ANN → velocity rows [3,4,5] + aug rows [6,7]; C_aug removed
**Date**: 2026-06-30
**What**: Replaced D-065 C_aug routing with Model B routing in `gantry_interconnect_dynamic.py`. Changes: (1) `Parameterized_Linear_Output_Block` and C_aug removed; output is `Linear_Output_Block(Cd_norm)` only. (2) `VEL_AUG_IX = [3,4,5,6,7]` defined. (3) ANN `nw` changed from `NX_ANN=2` to `len(VEL_AUG_IX)=5`. (4) `expansion_matrix(AUG_IX, nxd)` → `expansion_matrix(VEL_AUG_IX, nxd)`. ANN input unchanged (sees full state + u).
**Why**: D-065 C_aug routing has a gradient dead zone by construction. C_aug is initialized near-zero (Frobenius norm = 1e-2). The gradient chain Loss→y→C_aug@x_aug→x_aug→ANN scales with ‖C_aug‖_F ≈ 0, so the ANN receives no learning signal. Confirmed by `diag_gradient_routing.py` on real gantry data: Model A (C_aug) ANN grad = 1.04e-2, Model B (vel routing) ANN grad = 2.85e-1, ratio 27x. This is the root cause of R²≈0 from diag12.
**Why velocity rows are safe (contra D-065 ruling)**: D-065 ruled out velocity routing citing "836x blowup (diag13)". That test used a non-zero-initialized ANN at long nf=400 rollout. Model B is safe at epoch 0 because ANN is zero-initialized: correction starts at 0, so initial position drift is zero. Velocity states have stable eigenvalue z=1-C*Ts/m < 1 (C>0 for gantry). Gradient of position loss w.r.t. velocity correction converges to Ts·m/C as T→∞ — a finite bound, not O(T²) as for position-row routing. `diag_spring_stiffness.py` confirms K=0 position routing gives O(T²) gradient growth; velocity routing is structurally bounded.
**Why position rows remain excluded**: Gantry X/Y/bridge axes have K=0 (no spring stiffness). DT position eigenvalue z=1 exactly. Additive correction to position accumulates without restoring force: gradient O(T) unbounded (confirmed `diag_spring_stiffness.py`). No spring stiffness can be added — this is the physical system.
**Literature**: Tustin-Net (Pozzoli et al. 2019/2020), van Esch et al. 2024 — multiple independent groups converged on ANN injection at force/velocity level, never at position level, for systems with integrating modes.
**Ruled out**: (1) C_aug routing (D-065): gradient dead zone, ANN learns nothing. (2) Full-state parallel (Jan's default): position rows give unbounded gradient at K=0. (3) CT force injection via LFR deriv(): gradient 7.89e-4, ~500x weaker than Model B (two CT integrations vs one DT). (4) Series-in (identity init): 177x stronger gradient but comes from position modification path — same drift risk.
**Constrains**: Output is now solely through fixed Cd_norm (no trainable C_aug). ANN must cause sufficient position change via the velocity→position integration for the loss signal to drive learning. The aug states [6,7] still exist and receive ANN correction — they are free latent states for any unmodeled dynamics the ANN discovers. First training run needed to confirm R² improvement.
