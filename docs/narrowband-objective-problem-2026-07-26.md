# Training destroys the initialisation: measurements of 2026-07-26, for independent diagnosis

**Purpose: give a research session everything measured, so it can diagnose the problem
FOR ITSELF.** Sections 1-4 are measurements with no interpretation. Section 5 is my
reading, clearly separated and scrutinizable. Section 6 lists readings I could NOT
exclude. Section 7 is the anti-scope, which rests on evidence rather than on framing.

**Do not inherit section 5.** Several long-standing framings in this repo have been
falsified by later measurement, including two of mine today. Where a document and the raw
numbers disagree, the numbers win; every script and unit file is named in section 9.

---

## 1. The system

* Dual-gantry motion system. Logical states `[X, Theta, Y, dX, dTheta, dY]`. **X and Y are
  pure double integrators** (`K = 0`); Theta has a spring (`kb1 + kb2`).
* Baseline: LPV-LFR, `Y`-scheduled inertia `M(Y)`, RK4, `Ts = 2.5e-4 s` (4 kHz),
  `up_sample = 1`.
* Augmentation: static ANN (16 nodes, 2 hidden layers, tanh, **zero-output init**, 600
  parameters) writing additive corrections to all 8 rows (6 physical + 2 latent),
  `nx_ann = 2`.
* Encoder: Hoekstra reconstructability map, `na = nb = 17`, `na_right = nb_right = 1`,
  **trainable**, initialised from the baseline linearised at `Y_op = 0`
  (`gantry_linearize_and_discretize` raises `NotImplementedError` for any other value).
* **Training loss:** mean per-step MSE on **normalised** outputs over `nf = 400` steps
  (0.1 s), each window independently encoder-initialised. Adam, `lr = 1e-7`,
  `batch = 256`, `stride = 10`, 20 epochs (~5200 updates).
* **Selection metric:** `sim-RMS` = `apply_experiment(val).RMS(val)`, in **metres**, over
  the **full 12 s** record from a single encoder init. Note the loss is normalised and
  dimensionless; the selector is raw metres. They are not the same functional.
* Unmodelled truth: a mass-spring-damper absorber on the head, `ma_frac = 0.10`,
  `fa = 150 Hz`, `zeta_a = 0.05`. Position signature `2.186e-05 m` std, velocity
  `2.160e-02 m/s`.

---

## 2. The failure (from the production checkpoints; no re-training)

`gantry_drift_71167_last.pth`, 20 epochs, 5200 batches. The full per-epoch history is
stored inside the file.

| quantity | epoch 0 | epoch 1 | epoch 20 |
|---|---|---|---|
| val sim-RMS, **12 s free run** | `1.661e-04` | `2.109e-02` (**127x**) | `1.951e-02` (117x) |
| val nf-window RMS, **0.1 s** | `4.386e-05` | | `3.771e-05` (**-14%**) |
| train nf-window RMS, 0.1 s | `3.807e-05` | | `3.324e-05` (-13%) |
| train loss (normalised) | | `1.369e-06` | `1.103e-06` (-19%) |

`argmin(val sim-RMS) = epoch 0`, all 20 epochs.

Note the **validation** windowed metric improves while the **validation** free run
collapses — same record, same model, same data.

**Dose-response.** `gantry_drift_last.pth`, 130 batches, 5 epochs:
`1.83x / 4.69x / 6.06x / 7.16x / 9.47x`. So 130 updates gives 9.5x and 5200 gives 127x.

---

## 3. Measurements on the initialisation and the encoder

**M1. The encoder initialisation settles; it does not ramp.** ANN off, `pysynth_baseline`
(model == data exactly). Encoder-x0 free run: mean late slope `5.39e-05 m/s` at 2 s versus
`5.88e-07 m/s` at 12 s (**92x collapse**) while the level grows only 1.7x
(`1.46e-04 -> 2.50e-04 m` on X). True-x0 control sits at `1.07e-06 m`. At 12 s = 7.8
`tau_X`. **Caution:** at 2 s the same data reads as "growing" (`last/first = 2.08`) purely
because the exponential is 73% settled; horizon must be stated with any such number.

**M2. With the absorber present, both initialisations drift comparably.** `pysynth`,
ANN off: pooled X RMS true-x0 `1.246e-04` vs encoder-x0 `2.308e-04`; late slopes
`1.60e-05` vs `2.11e-05 m/s`, about 30x the absorber-free arm on **both**.

**M3. The encoder's state error scales with the operating point.** `corr(|mean Y|,
|e_enc|) = +0.995`, slope `0.249 / m`. `|e_enc|` runs `3.28e-03` at `Y = 0` to
`8.08e-02` at `Y = -0.300` and `7.62e-02` at `Y = +0.300` — **24x**, symmetric in sign,
monotone in `|Y|`.

**M4. Per-row, the model error dominates the encoder error on the position rows.**
Ratio `e_enc / e_roll`: X `0.01`, Theta `0.16`, Y `0.02`, dX `0.90`, dTheta `1.34`,
dY `1.18`. `dTheta` is ~100x every other row (`e_enc = 1.220e-01`,
`e_roll = 9.133e-02`), matching `diag19`'s pseudo-inverse noise amplification of `115.2`
on that channel. Pooled `e_enc/e_roll = 1.33`, but that aggregate is almost entirely
`dTheta`.

**M5. Two error sources partially cancel.** On records near `Y = 0`, `|d|` is *smaller*
than either of its two components. Consistent with C6 in
`drift-conclusions-2026-07-25.md`, now observed at a segment boundary.

---

## 4. Measurements on the continuity (defect) term

A multiple-shooting defect term was implemented and **verified as the method** (5/5):
zero defect on an exact model `8.6e-07`; Ribeiro Thm 2 equivalence `1.19e-06`; defect
linear in an injected constant force, exponent `1.000`; position defect `~nf_seg^2`,
exponent `1.993`; minimisable, `0.68x` in 30 steps.

**M6. The defect RMS does not distinguish a healthy model from a 65x-degraded one.**
Free-run ratio `1.196e-04 -> 7.781e-03` = **65.0x**. Defect RMS ratio: **1.01x** with
encoder nodes, 1.78x with true nodes. Flat as coverage grows: `n_seg = 4 / 12 / 30`
(0.4 / 1.2 / 3.0 s) gives `1.08x / 1.02x / 1.04x`.

**M7. The coherent (summed / mean) defect does distinguish them, with ideal nodes.**

| `n_seg` | RMS ratio | coherent ratio | coherence, healthy | coherence, degraded |
|---|---|---|---|---|
| 4 | 1.57x | 2.34x | 0.576 | 0.858 |
| 12 | 1.69x | 4.34x | 0.323 | 0.831 |
| 30 | 1.79x | **13.80x** | 0.108 | 0.834 |

(true nodes; coherence = `|mean d| / rms d`). The healthy model's coherence tracks
`1/sqrt(n)` (predicted `0.500 / 0.289 / 0.183`); the degraded model's stays flat at
`~0.83`. **With encoder nodes the coherent ratio reaches only `1.06 / 1.21 / 1.50x`.**

**M8 is VOID.** An attempt to measure the gradient each term supplies compared gradients
against `theta_deg - theta_init`, which is an Adam-accumulated displacement, not a
gradient. It returned **below-chance** cosines (`< 1/sqrt(600) = 0.041`) for all three
terms **including the control**. When the control fails the test is wrong. Nothing is
known about the gradients.

---

## 5. My reading — ONE candidate, not a conclusion

The failure is a **near-DC component of the residual force**, and every objective and
diagnostic tried so far is a **broadband average** that suppresses it.

Supporting: on a `K = 0` axis a constant force gives `f t^2 / 2`, so `0.005 f` at 0.1 s
against `72 f` at 12 s — a factor 14400, which is why a 0.1 s objective cannot see it
(and the `nf_seg^2` exponent `1.993` in the method tests is that same law, measured). The
absorber residual is oscillatory, zero-mean and present at full amplitude in every window,
so magnitude-weighted averages favour it over the harmful component. M7's coherence
column is the most direct support: the degraded model's defect is coherent (`0.83`, flat),
the healthy model's is not (`1/sqrt(n)`).

If true, it also explains why `drift-conclusions` §4's ruled-out list failed *as a set*:
longer `nf`, ARTBP, preconditioning, `lr` tuning and Adam->SGD are all broadband
interventions.

**What would falsify it:** a coherent/low-frequency-selective objective that still fails
to change the 12 s outcome; or evidence that the degraded model's DC is a *consequence* of
something else (see 6.2) rather than the primary error.

**Its weakest point:** it does not account for M5 (the cancellation), and it does not
explain why the very first epoch already costs 127x.

---

## 6. Readings I could NOT exclude

**6.1 Exposure bias / training distribution.** The model never sees its own drifted states
during training, because every window re-anchors to measured data. That is a statement
about the *training distribution*, not the horizon, and it is not the same claim as mine.
`docs/rollout-stability-literature.md` already holds this literature (pushforward, GNS
noise, scheduled sampling). Nothing measured today separates it from 5.

**6.2 Anti-damping self-feedback on the marginal mode. — CLOSED 2026-07-26, REFUTED.**
Originally: `drift-conclusions` C3 measured a curvature exponent `p = 3.749` along the
ANN's own trained direction and identified a wrong-sign velocity feedback
(`dW_dY/ddY` positive); a slow instability would *also* produce an apparently coherent
defect, so M7 could not distinguish it from a DC force error. **This was flagged here as
the most serious unexcluded alternative.**

**MS11 tested it and refuted it** (`orrell_two_run.py`, Orrell et al. 2001 Eq. 5). The
twin-perturbation test cancels the drift term exactly and isolates the propagator: growth
is **polynomial, not exponential**, in every arm (R^2_poly `0.874-0.961` vs R^2_exp
`0.476-0.693`) and the exponent is **sub-linear** (INIT `+0.34`, DEGRADED `+0.42`),
alpha-invariant across `1e-6` and `1e-5`. The degraded model adds only **1.29x** of
propagator amplification over a control whose ANN output is identically zero. Meanwhile
the clean-start shape test gives the degraded model a power law of exponent **`1.484`
at R^2 `0.997`** on X (and `1.677` on Y), which is the signature of a constant force on a
lightly damped axis (`t^2/2` early, toward `t*f*tau` late, `tau_X = 1.546 s`).

**So reading 5 is confirmed and 6.2 is refuted. The fix family is DC-selective /
bias-aware, not stabilisation.** Anyone re-opening this needs to beat MS11's numbers, not
re-argue from C3.

**6.3 Normalisation mismatch between loss and selector.** The training loss is MSE on
outputs normalised by `ystd`; the selector is raw metres. If `ystd` differs substantially
across channels, the loss is implicitly weighting axes differently from the metric that
judges it, and the drifting axes may simply be under-weighted in training. **Cheap to
check and not checked.**

**6.4 Under-determination.** 600 ANN parameters, ~5200 updates, `lr = 1e-7`. The drift
direction may be weakly determined by this data, in which case almost any regularisation
would help and none of the above is the "cause".

**6.5 The absorber may not be learnable at this routing at all.** `r2_fit_probe.py`
recovered only `+18%` of a target that was *exactly representable by construction*. If the
estimator can only ever recover a fraction, the free-run failure may be a symptom of an
under-fit residual rather than of a harmful component.

---

## 7. Anti-scope (evidence-based, not framing)

* **Longer training windows.** Refuted: SLURM 71013 (`nf` 800 to 3200, DC at every
  horizon); NF=900 diverged; `O(N^3)` conditioning at `|lambda| = 1`.
* **ARTBP / unbiased truncated BPTT / better gradient estimators.** `drift-conclusions`
  C4; Beatson & Adams (ICML 2019) Thm 4.1 at `|lambda| = 1`.
* **Hard class restrictions** (passivity, contraction, RENs, bounded impulse, spectral
  caps). Violate R2 (full expressivity, the project's non-negotiable), and separately
  measured to fail — passivity bounds velocity, not position (`p1_drift_probe.py`).
* **Zero-mean / window-mean priors on velocity rows.** Real Telica residual mean is
  `-157.5 N` (X), `-83.7 N` (Y) at **315 to 344 sigma**.
* **Optimiser swaps or `lr` tuning.** Drift is proportional to `lr`; SGD learns `+0%` on a
  real residual.
* **"Multiple shooting was tried and failed (Optuna 69399)".** That run was pre-D-101
  (silent Adam default `lr = 1e-3`) *and* was an `nf` sweep under single shooting with no
  defect term.

---

## 8. Open questions, ordered by what they would unblock

1. Is the degraded model's error a **constant force** or a **slowly diverging mode**?
   (6.2). Distinguishing these changes everything downstream and nothing measured today
   separates them.
2. Does the loss/selector **normalisation mismatch** (6.3) contribute? Cheapest open item.
3. Is there a formulation that separates **estimator bias from model bias**? Any coherent
   statistic that detects the model's DC also detects the encoder's `Y`-dependent bias
   (M3, M7). Bias-aware filtering, Schmidt-Kalman, augmented-state bias estimation.
4. **Frequency-weighted objectives** in system identification — filtered PEM and whether
   any of it transfers to a nonlinear rollout objective.
5. **Data assimilation**: weak-constraint 4D-Var uses `Q^-1`-weighted defects (Fisher et
   al., ECMWF 2011). Does that field treat the *spectral content* of model error, and the
   neutral / zero-Lyapunov-exponent subspace where our integrators live?
6. `arXiv:2406.03760` (Kuntz & Rawlings, MLE with integrating disturbances) — flagged
   unread in the earlier sweep; the closest named hit on unit-root modes.

---

## 9. Reproduction, and traps

Scripts in `scripts/gantry/pysynth-data/`, units in its `results/`:
`generate_pysynth_data.py` (self-consistent data, D-126), `verify_defect_term.py` (7/7
plumbing), `verify_ms_method.py` (5/5 method), `measure_encoder_drift.py` (M1/M2),
`measure_defect_split.py` (M3/M4/M5), `check_defect_sees_failure.py` (M6),
`check_coherent_defect.py` (M7), `check_defect_gradient.py` (M8, **void**).

**Traps that invalidated runs today.** `RunConfig` defaults `up_sample = 2` while the
entry file and every checkpoint use `1`. Trimming `TRAIN_FILES` changes
`compute_normalization`, which changes the encoder built from `norm.x_all` (D-119), which
changes every number downstream — a 4-record trim moved epoch-0 from `1.66e-04` to
`1.13e-01`. Filtering non-finite values out of a metric series makes divergence look
identical to a flat pass. `gantry_ckpt_71167.pt` is the **best** checkpoint and the
failure is that best = epoch 0, so that file **is** the initialisation; use
`gantry_drift_71167_last.pth`. The `.pth` files pickle `gantry_dynamic` as a top-level
module (put `scripts/gantry` on `sys.path`) and carry their own `norm` — take weights and
`norm` together, never mixed.

**Evidence grade: everything measured today is 1 seed and mostly 1 record, below the
project's 3-seed floor. M7's headline number uses ORACLE true states, which do not exist
on real data; the deployable figure there is 1.50x, not 13.80x.**
