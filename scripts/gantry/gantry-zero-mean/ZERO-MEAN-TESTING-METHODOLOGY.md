# How to test "is it zero-mean?" properly (methodology reference)

**Created**: 2026-07-16, from a session course-correction. The user observed that the whole
gantry-zero-mean investigation had been *asserting* zero-mean (via `|mean|/std` ratios and by
eyeballing "climb-and-park vs ramp" in traces) rather than *testing* it with the established
statistical machinery. This file is the durable write-up of the proper methodology, its formulas,
the tools, and how each piece maps onto our specific problem. Read it before making any further
zero-mean / DC / offset claim in this project.

Cross-reference: the one-line rule is `tasks/lessons.md` -> `test-zero-mean-properly`
(under "Verification & testing"). This file is its full backstory and recipe.

---

## 0. Scope: what question are we even asking?

The single word "zero-mean" has been standing in for at least FOUR statistically distinct
questions. Separate them before testing anything:

| id | question | the right instrument (section) |
|----|----------|-------------------------------|
| Q-a | Is *this recorded signal's* time-mean zero? | stationarity classification (§3) then HAC mean t-test (§2) |
| Q-b | Does the *system / baseline model* carry a static (DC) offset it should model? | residual analysis: zero-mean + whiteness (§4) |
| Q-c | Is the *required correction* (baseline one-step residual) zero-mean? | same as Q-b: it literally IS a model residual (§4) |
| Q-d | Is the observed drift a deterministic constant, a deterministic trend, or a stochastic random walk? | unit-root vs stationarity tests: ADF + KPSS (§3) |

Two facts sit above all four and must be stated alongside any verdict:
- **Identifiability (§5)**: our excitation has no power at 0 Hz, so the data is partly blind to DC.
- **Nonlinear rectification (§6)**: a nonlinear / LPV plant turns a zero-mean input into a genuine
  nonzero-mean output, so "input is zero-mean" never implies "output is zero-mean".

---

## 1. What we were doing wrong (so we do not repeat it)

Three bad practices, all used in this folder's prior work (v1b, v1d, the V1 audit, figure f04b):

1. **`|mean|/std` (or `|mean|/rms`) ratios.** Reported as if "1e-4 is small, therefore zero-mean".
   This is not a hypothesis test. It has no null distribution, no p-value, and the denominator
   (`std` or `rms`) is not the standard error of the mean. It cannot tell you whether a mean is
   *significantly* nonzero.

2. **The standard error was wrong even where an SE was used.** The naive SE of a sample mean,
   `std / sqrt(N)`, assumes independent samples. Our signals are strongly autocorrelated (a 150 Hz
   ripple sitting on a slow trend), so the effective number of independent samples is far below N.
   The naive SE is therefore wildly optimistic. The correct sample-mean variance is the **long-run
   variance** (§2), which can be orders of magnitude larger.

3. **Eyeballing traces.** Deciding "park (bounded) vs ramp (drift)" by looking. This is subjective
   AND it conflates two statistically different objects: a *deterministic* bounded transient and a
   *stochastic* random walk can look identical over a finite record. The distinction is exactly
   what the unit-root tests in §3 are built to make.

4. **Testing the mean of a nonstationary series.** On the K=0 (X/Y) free-integrator axes the
   position is an integrated (nonstationary) process. The sample mean of a nonstationary series
   does not estimate any fixed population mean, so `mean(Y)` is ill-posed. Always test the
   stationary quantity instead: the velocity, or the one-step model residual, not the integrated
   position. (This is why the position-mean arguments always felt slippery.)

---

## 2. Q-a: is a stationary signal's mean zero? -> HAC / Newey-West long-run variance

Model the (stationary) signal as
```
X_t = mu + eps_t ,   eps_t  = zero-mean, weakly stationary, autocorrelated
```
The estimator of mu is the sample mean `Xbar`. Its variance is NOT `sigma^2 / N`. It is the
**long-run variance** divided by N:
```
Var(Xbar) = (1/N) * LRV ,   LRV = sum_{k=-inf}^{+inf} gamma_k = 2*pi*S(0)
```
where `gamma_k` are the autocovariances and `S(0)` is the power spectral density at zero
frequency. The DC significance of a signal IS the value of its spectrum at f = 0.

**HAC / Newey-West estimator** of the long-run variance, with bandwidth (max lag) L and a Bartlett
(triangular) window:
```
LRV_hat = gamma_0 + 2 * sum_{k=1}^{L} ( 1 - k/(L+1) ) * gamma_k
se(Xbar) = sqrt( LRV_hat / N )
t = Xbar / se(Xbar)
```
Compare `t` to a normal (or Student-t / fixed-b) critical value. Bandwidth L is chosen by a rule
(Newey-West 1994 automatic, or Andrews 1991 optimal). This is the standard replacement for the
`|mean|/std` ratio.

- **Tool**: `statsmodels`. Either compute the Newey-West LRV directly, or regress the signal on a
  constant with `cov_type='HAC'` (`cov_kwds={'maxlags': L}`) and read the intercept's t-stat and
  p-value. `statsmodels.stats.sandwich_covariance` and `OLS(...).fit(cov_type='HAC', ...)`.
- **Maps to us**: any per-channel mean claim (delta_a, u, velocities, the correction rows). Report
  `Xbar`, HAC `se`, `t`, `p`, and the chosen L. Never report `|mean|/std` again.

---

## 3. Q-d: constant vs trend vs random walk -> ADF + KPSS (run BOTH)

Before testing a mean on a K=0 axis you must classify the series, because "is there a constant"
only makes sense for a stationary or trend-stationary series. Two complementary tests:

- **ADF (Augmented Dickey-Fuller)**. Null hypothesis: the series HAS a unit root (nonstationary,
  stochastic drift / random walk). Reject -> no unit root -> stationary. Include a drift and a
  deterministic trend term in the regression when testing for trend-stationarity.
- **KPSS (Kwiatkowski-Phillips-Schmidt-Shin)**. Null hypothesis: the series IS (trend-)stationary.
  Reject -> nonstationary. Has a "level" version (null = stationary around a constant) and a
  "trend" version (null = stationary around a deterministic trend).

They are complementary because their nulls are swapped; run both and cross-read:

| ADF | KPSS (level) | conclusion |
|-----|-------------|------------|
| rejects (stationary) | fails to reject (stationary) | **stationary** -> a mean is well-defined; test it (§2) |
| fails to reject | rejects | **unit root / stochastic drift** -> mean undefined; this is a random walk |
| rejects | rejects | **trend-stationary** -> deterministic trend; detrend, then test the residual mean |
| fails to reject | fails to reject | inconclusive / too little data |

**This is the rigorous version of "park vs ramp".** A deterministic, mean-reverting series that
settles is a *park* (bounded transient, shocks decay). A unit-root series whose shocks are
permanent is a *drift / ramp*. The ANN's runaway is the unit-root case; the baseline error settling
to a fixed offset is the trend-stationary / level-stationary case. Use the tests, not the eye.
The **Mann-Kendall** nonparametric trend test is a lightweight companion for "is there a monotonic
slope" when you only want the trend question.

- **Tool**: `statsmodels.tsa.stattools.adfuller`, `statsmodels.tsa.stattools.kpss`. (Mann-Kendall:
  `pymannkendall` or a hand-rolled S-statistic.)
- **Maps to us**: run on the free-run position AND on velocity for every K=0 row, and on the
  required-correction rows, to classify each as park (trend/level-stationary) vs drift (unit root).

---

## 4. Q-b / Q-c: does the baseline model demand a DC? -> residual analysis

This is the key reframe. The **required correction**
```
w_perfect[k] = x_true[k+1] - baseline_step( x_true[k], u[k] )
```
is LITERALLY the baseline model's one-step-ahead residual. "Is the correction zero-mean?" is
therefore the textbook **model-validation residual test**, and system identification already owns
the exact procedure. An adequate model's residuals must be BOTH:

1. **zero-mean** -> test with the HAC t-stat from §2 (a significantly nonzero residual mean = model
   bias = a real missing DC that the model should absorb). This is the direct, defensible version
   of figure f04b's "|mean|/rms = 0.000".
2. **white** (uncorrelated across lags) -> test with the **Ljung-Box** portmanteau statistic:
```
Q = N (N+2) * sum_{k=1}^{h} rho_hat_k^2 / (N - k)
```
where `rho_hat_k` is the sample autocorrelation of the residual at lag k, and h is the number of
lags tested. Under the white-noise null, Q is chi-squared with (h - p) degrees of freedom (p =
number of fitted parameters; 0 for a raw residual). A significant Q (small p-value) = residual
autocorrelation remains = the model is misspecified (missing structure, possibly a DC or a slow
mode). Box-Pierce is the uncorrected precursor; Ljung-Box has the small-sample correction, use it.

- **Tool**: `statsmodels.stats.diagnostic.acorr_ljungbox(resid, lags=[...], return_df=True)` for
  whiteness; the §2 HAC t-test for the zero-mean part.
- **Maps to us**: this is the *correct* form of the central question "does the with-MSD system
  demand a non-zero-mean correction". Run it on `w_perfect` per row. Zero-mean + white on the K=0
  rows = the baseline needs no DC there (G-A refuted, properly). Nonzero-mean or colored = a real
  deficiency the augmentation legitimately must carry.

---

## 5. Identifiability caveat: our excitation is blind to DC

A multisine excitation is DEFINED with its zero-frequency component set to zero: `U_0 = 0`. There
is **no power at 0 Hz by design** (our gantry input is a multisine at 130-180 Hz, exactly zero at
DC, see figure f10). Consequences that MUST accompany any zero-mean verdict from this data:

- The data carries **zero information about the system's static / DC gain**. You cannot identify a
  DC offset from DC-free excitation. A "the correction is zero-mean" finding is therefore *partly
  guaranteed by the experiment*, not discovered in the physics.
- Any honest conclusion states: "a DC / static deficiency is **not identifiable** from this
  excitation; to test it one must excite at or near 0 Hz (a step, a slow ramp, an offset, or a
  low-frequency multisine line)".
- This is also why the eventual fix must stay a SOFT, data-conditional pin and never a structural
  zero-mean constraint: on the real machine the baseline may have a genuine static deficiency that
  this excitation simply cannot see.

Reference: Pintelon & Schoukens, *System Identification: A Frequency Domain Approach*
(multisine design; `U_0 = 0` convention; DC gain not identifiable without DC excitation).

---

## 6. Nonlinear rectification: zero-mean in does NOT imply zero-mean out

A nonlinear or LPV plant maps a zero-mean input to a **nonzero-mean output** through its even-order
(second-order) Volterra kernel. In a Volterra series the output mean is the zeroth kernel `h_0`
plus the rectified contribution of the second-order kernel:
```
E[y] = h_0 + integral integral  h_2(tau_1, tau_2) * R_uu(tau_1 - tau_2)  d tau_1 d tau_2  +  ...
```
where `h_2` is the second-order Volterra kernel and `R_uu` is the input autocorrelation. So:

- A quadratic / bilinear / LPV nonlinearity **rectifies** zero-mean oscillation into a real DC.
  This is the exact, computable formalization of the informal "sin^2 -> 1/2" argument used
  throughout this investigation.
- In our system the Y-dependence of the mass matrix `M(Y)` (and the `ma*(Y+L0+delta_a)^2` inertia
  term) is exactly such a nonlinearity, so a genuine second-order DC exists. It is small (measured
  at |mean|/std ~ 1e-4, the "e-5" level in v1d) but nonzero and structured, not noise.
- Crucially for us: this rectification is present in BOTH the truth and the baseline (both carry
  `M(Y)`), so it largely CANCELS in the residual `w_perfect` = truth - baseline. What survives is
  only the MSD's extra Y-dependent term. That is why the LPV rectification, though real, is not the
  source of the ANN's DC via the "the system demands it" pathway.

References: Rugh, *Nonlinear System Theory: The Volterra/Wiener Approach*; Schetzen, *The Volterra
and Wiener Theories of Nonlinear Systems*.

---

## 7. The problem, restated correctly

Replace the loose "is the correction zero-mean?" with the precise statement:

> Are the baseline model's one-step residuals `w_perfect`, evaluated on the STATIONARY quantity
> (velocity or the residual itself, never the integrated K=0 position), (i) classified as
> level/trend-stationary by ADF + KPSS, (ii) not significantly different from zero-mean under a
> HAC / Newey-West standard error, and (iii) white under Ljung-Box, bearing in mind that (iv) a DC
> deficiency is not identifiable from our U_0 = 0 excitation, and (v) the LPV plant contributes a
> real but common-mode second-order rectification DC that cancels in the residual?

Not "is `|mean|/std` small".

---

## 8. Recommended test harness (spec, not yet built)

A Python-only analysis on data we already have. Proposed as a `gantry-zero-mean` companion script
(`zeromean_residual_test.py`), producing a table and a short figure per row:

1. Build `w_perfect[k] = x_true[k+1] - baseline_step(x_true[k], u[k])` per record (reuse the
   verified physics in `drift-demo/demo_common.py` / `diagnostics-drift/drift_common.py`; do NOT
   re-derive).
2. For each of the 8 rows (and separately for position vs velocity where relevant):
   - **Stationarity**: `adfuller` + `kpss` (level and trend). Report both statistics, both
     p-values, and the classification (stationary / trend-stationary / unit-root).
   - **Zero-mean**: HAC / Newey-West mean t-test (OLS on a constant, `cov_type='HAC'`,
     `maxlags` from Newey-West 1994). Report `mean`, HAC `se`, `t`, `p`, and L.
   - **Whiteness**: `acorr_ljungbox` at a set of lags spanning the ripple and slow scales. Report
     Q and p.
3. Print the identifiability caveat (U_0 = 0) as a header line on every output so no verdict is
   ever read without it.
4. Repeat across records at DIFFERENT Y operating points (and a jerk-limited Y-sweep such as
   `T7_ysweep_fast`) so any operating-point / LPV dependence of the residual DC is visible.
5. Optional confirmation of the rectification magnitude: compare the residual DC against the
   second-order Volterra prediction, or re-run with the pure multisine `f_ms` instead of the
   reconstructed `u_total` to remove the input-reconstruction confound (the v1d discriminator).

Standing constraints (from the folder README): thresholds data-derived only, noiseless-sim phase,
scripts live in `scripts/gantry/gantry-zero-mean/`, figures in its `figures/`.

---

## 9. Quick tool reference (all in the GraduationProject env; confirm before use)

| test | question | statsmodels entry point |
|------|----------|-------------------------|
| HAC / Newey-West mean t-test | is a stationary signal's mean zero? | `OLS(x, ones).fit(cov_type='HAC', cov_kwds={'maxlags': L})` |
| ADF | unit root (stochastic drift)? | `statsmodels.tsa.stattools.adfuller` |
| KPSS | (trend-)stationarity? | `statsmodels.tsa.stattools.kpss` |
| Ljung-Box | residual whiteness? | `statsmodels.stats.diagnostic.acorr_ljungbox` |
| Mann-Kendall | monotonic trend? | `pymannkendall` (or hand-rolled S-stat) |

---

## 10. Sources (verified this session, 2026-07-16)

- Muller, *HAC Corrections for Strongly Autocorrelated Time Series* (long-run variance, S(0)):
  https://www.princeton.edu/~umueller/HACtest.pdf
- Newey-West estimator (formula, bandwidth): https://metricgate.com/docs/newey-west-estimator/
- ADF vs KPSS (complementary nulls, deterministic vs stochastic trend):
  https://medium.com/@tannyasharma21/comparision-study-of-adf-vs-kpss-test-c9d8dec4f62a
- KPSS test for stationarity: https://machinelearningplus.com/time-series/kpss-test-for-stationarity/
- Ljung-Box test (portmanteau whiteness, small-sample correction):
  https://www.numberanalytics.com/blog/ultimate-ljung-box-test-guide
- Whiteness test in model identification (ScienceDirect):
  https://www.sciencedirect.com/science/article/abs/pii/S000510982200019X
- Pintelon & Schoukens, *System Identification: A Frequency Domain Approach* (multisine U_0 = 0,
  DC gain identifiability): https://books.google.com/books/about/System_Identification.html?id=KhonXGwETWsC
- Rugh, *Nonlinear System Theory: The Volterra/Wiener Approach* (Volterra kernels, DC from even
  order): https://rfic.eecs.berkeley.edu/courses/ee242/pdf/volterra_book.pdf
- Volterra model overview (zeroth kernel = system average; second-order rectification):
  https://www.sciencedirect.com/topics/engineering/volterra-model

Textbook homes to cite in the thesis (not re-fetched this session, standard references):
- L. Ljung, *System Identification: Theory for the User* (residual/correlation tests, detrending,
  handling offsets and drift in preprocessing).
- T. Soderstrom & P. Stoica, *System Identification* (model validation, residual analysis).
- D. Percival & A. Walden, *Spectral Analysis for Physical Applications* (spectrum at f = 0).
</content>
</invoke>
