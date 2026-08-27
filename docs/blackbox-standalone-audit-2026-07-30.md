# Audit of `blackbox_standalone.py` and diagnosis of runs 73940 / 74045

Written 2026-07-30. Part 1 (audit) was done before any result was read; Part 2 after.

---

# Part 1. Is it a correct full-ANN SUBNET?

**Yes.** `blackbox_standalone.py` builds the same model as Jan's `msd_ndof_deepSI_encoder.py:30`
and the same model as Gerben's `dsi.SUBNET` (`deepSI-master/examples/docs/basic-example.py:24`,
"Creates encoder, f and h as MLP"): `SS_encoder_general_hf` with a learned `simple_res_net`
encoder over the flattened `[u_past, y_past]` window, a learned state net, a learned output net,
no feedthrough, `validation_measure='sim-RMS'`. Nothing physical enters the model or the training
path. No deviation among the twelve changes the model class, and none corrupts the numerics.

The defects found are in the **diagnostics and reporting layer** appended to the reference
skeleton, not in the model. Two of them (11a and 12) are severe, because they are why two failed
runs read as partial successes.

## Verdicts on the twelve

| # | Deviation | Verdict | Confidence | Severity |
|---|---|---|---|---|
| 1 | Gantry `.mat` data via `load_datasets` | correct as code; carries every consequence in Part 2 | high | see Part 2 |
| 2 | `auto_fit_norm=False` + pinned `u0/ustd/y0/ystd` | sound, and numerically a no-op vs the default | high | none |
| 3 | `init_model` before `fit` to pass `lr` | correct and necessary | high | none |
| 4 | `.reshape` for `.view` in encoder/state nets | genuine no-op; the fix is required | high | none |
| 5 | `nx=8`, `na=nb=17` | not a deviation: `17 = 2*8+1` is Jan's own rule | high | none |
| 6 | all three nets 2x64 | deepSI's own default | high | none |
| 7 | `stride` in `loss_kwargs` | correct | high | none |
| 8 | `its_per_val=1300` | correct; the comment beside it is stale | high | low |
| 9 | explicit `lr=1e-3` | equals Adam's default anyway | high | none |
| 10 | read the series from `_last`, save `_best` | correct, and a real deepSI trap correctly avoided | high | none |
| 11 | `_NfProbe` | works, but the train number is measured on one record | high | **high** |
| 12 | gates / npz / `config.json` | npz and config fine; **the gates cannot fail the way these runs failed** | high | **high** |

### Detail on the ones that needed checking

**2.** deepSI's `System_data_norm.fit` (`system_data.py:843-849`) is exactly per-channel
`mean`/`std` along axis 0 over the concatenated train records, which is exactly what
`compute_normalization` computes (`gantry_dynamic/data.py:234-237`). The only difference is the
guard epsilon, `1e-8` here against deepSI's `1e-15`, irrelevant at `ystd ~ 3e-2`. The claim that
this matches the augmented run also holds: `gantry_dynamic/model.py:159-162` pins the same four
fields the same way.

**3.** Confirmed at `fit_system.py:311-322`: with `init_model_done=True`, `fit` skips `init_model`
entirely and only calls `_check_and_refresh_optimizer_if_needed`, which preserves
`optimizer.defaults`. So `fit`'s `optimizer_kwargs` and `scheduler_kwargs` are genuinely dead, and
`init_model` is the only place `lr` and any scheduler can be set. The D-101 reasoning is right.

**4.** Root cause confirmed and it is not masking anything. `to_hist_future_data` has two branches:
`stride==1` uses `sliding_window_view` plus a transpose (`system_data.py:305-315`) and returns a
**non-contiguous** array; `stride!=1` builds a list and calls `np.array` (`:316-330`) and returns a
contiguous one. Training uses `stride=10`; `apply_experiment` and `n_step_error` use `stride=1`.
Hence the failure appears only at validation, exactly as documented. Verified empirically at the
gantry's shapes that both branches produce the identical logical window (sample 0 equals
`u[0:nb]`, `ufuture[0]` equals `u[nb:nb+nf]`) and that `reshape` equals `view` wherever `view` is
legal. Same net, same numbers.

**10.** Confirmed at `fit_system.py:483-490` and `497-507`. `fit` writes `_last`, then reloads
`_best` by replacing `self.__dict__`, which replaces `Loss_val`/`Loss_train`/`batch_id` with the
history as of the best point. Reading the series after `fit` returns would silently truncate it at
the best point. The workaround is correct, and it is also correctly ordered: `bestfit` is captured
first, `_last` is loaded for the series, `_best` is reloaded before `save_system`.

### 11. `_NfProbe`: five findings

**(a) Severity high, confidence high.** The "train nf-RMS" is computed on `data.train_list[0]`
alone (`blackbox_standalone.py:285`), i.e. **T1_standstill_Ym30 only**, not the 14-record train
set. T1 is a record whose entire dynamic content is `y std = [2.2e-6, 3.0e-6, 4.4e-6] m` sitting on
a `Y = -0.30 m` offset. Its RMS about the global `y0` is `1.756e-01 m`. So `train nf-RMS` on T1
does not measure fit quality; it measures how close the model's DC level is to `-0.30 m`. This is
the number that was read as "fitted the 0.1 s window well".

**(b) Severity medium.** Four quantities are printed side by side on four different scales and are
not comparable: deepSI's `train` (normalised MSE over all 14 records, printed as its square root),
`train nf-RMS` (metres, T1 only, mean over steps 1..nf), `val nf-RMS` (metres, windows pooled over
V1-V4), and `Val sim-RMS` (metres, weighted mean of per-record RMS over V1-V4, `system_data.py:712`).

**(c) Severity low.** The probe is installed as an instance attribute, so `checkpoint_save_system`
(`torch.save(self.__dict__)`) pickles the probe object, and with it T1 and V1-V4 (about 6 MB of
data) plus a self-reference, into every `_best` and `_last` write.

**(d) Severity low.** `except Exception: return nan` hides any probe failure.

**(e) Severity trivial.** The epoch-0 sim-RMS is computed twice: once at `blackbox_standalone.py:267`
and again by `fit`'s own initial validation (`fit_system.py:367`). One extra 4-record 12 s free run.

### 12. The gates: severity high, confidence high

G1 and G2 are scored only against epoch-0. Measured this session, epoch-0 is numerically the
**predict-the-global-mean** predictor:

| predictor, on the exact metric (`sim-RMS`, metres, V1-V4) | value |
|---|---|
| output the global `y0` constant | `9.549e-02` |
| 74045 epoch-0 | `9.577e-02` |
| 73940 epoch-0 | `1.033e-01` |

That is the weakest possible reference. A trivial "hold `y[k0]` for the whole 12 s" predictor
scores `4.740e-02 m`, twice as good as epoch-0, and neither run ever reached it. So G2 ("some epoch
beats init") can pass while the model is worse than doing nothing, which is what happened in both
runs. **Fix: print the trivial-predictor references (hold-last-sample; per-record DC) and the FP
baseline alongside epoch-0 in the gate block.**

## Findings outside the twelve

| | Finding | Confidence | Severity |
|---|---|---|---|
| E1 | The stage-0 control is **SISO**: the MSD npz has `u` shape `(20000,)`, `y` shape `(20000,)`, `nu=None`, `ny=None`. It therefore never touches the multi-channel window path, never needs the `.reshape` fix and does not use it. It also runs different nets (2x8 / 2x16), `auto_fit_norm=True`, `batch=2000`, `nf=200`, no norm pinning. It validates "deepSI plus the appended scaffolding runs", not "this script's model and data path work". | high | medium |
| E2 | The runner's deployed-copy grep covers `EPOCHS`, `LR`, `LR_MIN`, `ITS_PER_VAL` only. `nf_seconds`, `n_nodes_per_layer`, `stride`, `batch_size` are not in it, and `nf` and width are precisely what changed between 73940 and 74045. Nothing was lost (the script's own Configuration block prints them), but the grep does not do the job its comment claims. | high | low |
| E3 | `NF = CFG.nf   # 400` (line 122) is stale; it was 800 for 74045. Same for the `its_per_val` comment claiming "100 points on the curve", true only at 500 epochs; 74045 printed 21. | high | low |
| E4 | Dead arguments in the `fit` call: `auto_fit_norm=False` and `optimizer_kwargs={'lr': LR}` are both ignored once `init_model_done=True`. Harmless, but they read as active. | high | trivial |
| E5 | `_n_windows` is off by one per record versus deepSI (66052 against 66066). Rounds to the same steps/epoch. | high | trivial |
| E6 | At `stride=10`, `nf=800`, consecutive training windows overlap by 98.75%. The 66k windows per epoch are about 840 independent 0.2 s segments seen ~79 times each. Not wrong (standard SUBNET practice), but "epoch" and "update count" overstate the data seen. | high | informational |
| E7 | The materialised training tensor is about 1.3 GB at `nf=800` and scales linearly in `nf`; `nf=1600` would be ~2.6 GB. Fine at `--mem=24gb`. | high | informational |
| E8 | Free-run validation steps one sample at a time in Python (`system.py:159-162`): 4 x 47,983 single-sample forward passes per validation. That is the measured 27% per-epoch cost, and it is inherent to deepSI 0.3.29, not a defect here. | high | informational |

---

# Part 2. What went wrong in the runs

## The reference table the runs were missing

All computed this session on the exact metrics the script reports.

**12 s free run, `sim-RMS` in metres over V1-V4 (the campaign objective):**

| predictor | sim-RMS |
|---|---|
| predict the global `y0` (what a random net does) | `9.549e-02` |
| **hold `y[k0]` for the whole 12 s** | **`4.740e-02`** |
| oracle: this record's own mean | `4.698e-02` |
| 73940, best of 101 points | `7.236e-02` |
| 73940, last-quarter median | `9.76e-02` |
| 74045, best logged | `6.199e-02` |
| FP baseline, untrained | `1.6e-04` |

**Both runs are worse than a model that outputs a constant for twelve seconds**, and about 400x
away from the target. 73940's last-quarter median sits on the predict-the-mean value; its
`7.236e-02` best is a best-of-101 draw from a series that spans `7.2e-02` to `3.9e+14`.

**Normalised nf-window objective (deepSI's `Loss_train`):**

| | nf=400 | nf=800 |
|---|---|---|
| hold-last-sample baseline | `0.102` (sqrt `0.319`) | `0.346` (sqrt `0.588`) |
| run best | 73940: `0.0214` (sqrt `0.146`) | 74045: `0.499` (sqrt `0.707`) |

73940 did genuinely beat the trivial baseline on the 0.1 s objective, by 4.8x. **74045 never
reached it**: its sqrt-loss bottomed at `0.7066` against a trivial `0.588`, then rose monotonically
from iteration 11700 to `0.8890` while val stuck at `0.0946`. It was worse than doing nothing for
its entire logged life.

## Where the failure is

Not the implementation. Not, decisively, the training setup. It is the **objective and the metric
against this data and this model class**, and it has four parts.

### 1. The metric is ~100% DC and gross trajectory; the dynamics of interest are ~1e-9 of it

Measured band split of the validation outputs:

- **V1**: 100% of AC energy in 120-200 Hz, amplitude `3e-6 m`, on a `Y = +0.1 m` DC offset.
  Per-channel `RMS(y - y0) = [0.000, 0.000, 0.096]`. The val sim-RMS on V1 is *entirely* "did you
  output `Y = +0.1 m`".
- **V3**: X channels 99.98% in 120-200 Hz at `3e-6 m`; Y 99.9% below 1 Hz at `0.096 m`.
- **V2, V4**: more than 99% of energy below 20 Hz.

So `sim-RMS` in metres over V1-V4 measures the rigid-body trajectory and essentially nothing else.
The 130-180 Hz content the augmentation exists for contributes about `(3e-6/0.1)^2 ~ 1e-9` of the
squared metric.

The training loss has the same shape. A single global elementwise normalisation is applied across
14 records whose per-record `y std` spans `4.4e-6 m` (T1-T5) to `0.19 m` (T6, T7). After dividing
by the global `ystd = [0.032, 0.032, 0.190]`, records T1-T5 become constants at normalised levels
up to 1.6 with AC content of order `1e-4`. **Five of the fourteen training records carry no
gradient except a DC level.**

### 2. The metric asks for a 48,000-step free run of a plant with free-integrator axes

X and Y have `K=0`. `apply_experiment` encodes once at `k=17` and then runs 47,983 steps open loop.
Holding a `0.1 m` offset across that requires the learned `df/dx` to have unit-modulus eigenvalues
in the rigid-body directions to within about `1/48000`; representing five different standstill
levels requires a continuum of fixed points, i.e. an eigenvalue at exactly 1. Nothing in a 0.2 s
BPTT objective enforces this, and float32 barely resolves it. This is why the FP baseline reaches
`1.6e-04` with no training at all (it has the integrators by construction) and why an unconstrained
tanh MLP cannot. It is a structural problem, not a fitting-capacity one.

73940's val series is the signature: it is **bimodal**, a floor at `0.08-0.10` (contracted to a
fixed point, so it predicts a constant) and 13 excursions above 10x epoch-0 including `3.87e+14`
(spectral radius above 1, so 48,000 steps blow up). Almost nothing in between. That is what
"the critical eigenvalue is on the wrong side of 1" looks like.

### 3. 74045 specifically: `nf=800` made optimisation strictly worse and could not have closed the gap

The premise of the change was a 0.1 s to 12 s horizon gap of 120x; doubling to 0.2 s halves it to
60x, which cannot matter. Against that it paid the documented `O(N^3)` within-segment smoothness
cost (the project's own note in `gantry_dynamic/config.py:75-82`, Ribeiro et al. 2020 Thm 1) and
lost 78% of the update budget. The wider net (16 to 64) did not compensate. Net effect: a run that
never reached the trivial baseline on its own training objective.

The premise itself came from finding 11a: "train nf-RMS `1.28e-01 -> 1.70e-02`, 7.6x" on T1, whose
own dynamic content is `4.4e-6 m`. The model went from a wrong DC level to a DC level accurate to
about 5% of `0.30 m`. It was never a good 0.1 s fit on that record.

### 4. The two lr screens are not in conflict; they were read on the wrong axis

In **all six arms of both screens**, val sim-RMS on V1 rises from its initial value within one
epoch, while train loss falls in all six. The "ranking disagreement" is noise on top of a unanimous
signal: from the first epoch the training objective and the free-run metric move in opposite
directions. The correct reading of the probe was not "which lr" but "the objective is not aligned
with the metric", and it was available on day one for 260 updates per arm. For scale, V1's own
hold-the-DC error is `3.45e-06 m`; every arm's *initial* value of `0.027-0.031` is already ~8000x
worse than that.

## The one-line diagnosis

The data set was built for the **augmentation** experiment, where the FP baseline supplies the
rigid-body trajectory and the ANN supplies only the 130-180 Hz correction. Handed unchanged to a
standalone black box, with a metric in absolute metres over a 12 s free run, it asks the black box
to reproduce the rigid-body trajectory, which is exactly the part the FP model was there to
provide, and which is 100% of the score.

## Cheapest next measurement

**One call, no training, minutes, on the saved 73940 model** (`results/73940/blackbox_standalone_73940`;
74045's `_best` survives in the deepSI checkpoint directory and can be added free):

```
n_step_error(V1..V4, nf=48000, stride=48000, mode='RMS', mean_channels=False)
```

That returns per-channel free-run error as a function of step, 1 to 48,000, in metres, from a
single call. Overlay the same curve for the hold-last-sample predictor and for the FP baseline.

It settles the question in one plot:

- error flat and small out to a few hundred steps, then growing without bound -> the
  marginal-stability / drift mechanism of Part 2.2 is confirmed, and "train longer" or "raise `nf`"
  are both dead;
- error already at `1e-2 m` by step 1 -> the model never fit anything, and the culprit is the
  objective weighting of Part 2.1, not the horizon;
- the step at which the curve crosses the hold-last-sample line is the honest horizon over which
  this model is worth anything at all.

For two more lines, add the eigenvalues of the autograd Jacobian `df/dx` at the encoder state on
V1. It reads off directly how far the critical eigenvalue sits from 1, and whether the `0.08-0.10`
floor and the `3.87e+14` excursions are the two sides of that one number.

## Note on the campaign's success criterion

Whatever that measurement returns, "approach the FP baseline's `1.6e-4 m`" is a criterion that
cannot see the dynamics the thesis is about: on this data and this metric, the 130-180 Hz band is
`1e-9` of the score. That is worth settling before the next long run is sized.

---

# Part 3. The mechanism, and a controlled proof of it

Written later the same day, after the Part 2 measurement was actually run. **Part 2's conclusion
that the objective normaliser is the primary defect is superseded.** The normaliser is real but
second order. The primary defect is below it, and it is now measured rather than argued.

## 3.1 What the horizon measurement returned

Free-run error against horizon for 73940's saved `_best`, on V1-V4, against the hold-last-sample
predictor:

| horizon | time | model | hold `y[k0]` |
|---|---|---|---|
| 1 | 0.00 s | **7.80e-03** | 8.06e-07 |
| 400 | 0.10 s | 4.22e-03 | 5.43e-06 |
| 2000 | 0.50 s | 1.09e-02 | 5.58e-06 |
| 4000 | 1.0 s | 2.64e-02 | 2.64e-02 |
| 47983 | 12.0 s | 7.24e-02 | 4.74e-02 |

The error is flat out to 0.5 s and then diverges to the predict-the-mean level by 3 s. The model
is 7.8 mm wrong on the **first** predicted sample, with `y[16]` inside its own encoder window.

Per-mode NRMS in the logical frame, on the two yaw records that are inside 73940's training set:

| record | X | Theta | Y |
|---|---|---|---|
| T12_aprbs_yaw | 1.63 | 1.23 | 1.03 |
| T14_lissajous_yaw | 1.29 | 1.05 | 1.30 |

NRMS is about 1 on **every mode of every record**: the model is at "predict the mean" everywhere in
free run. Theta was not learned, but neither was anything else, so the Theta weighting identified
in Part 2 is not the binding constraint.

## 3.2 The mechanism: the learned state map cannot reach the unit circle

X and Y are `K=0` free masses, so the true discrete state matrix has eigenvalues **exactly 1**, and
the 130-180 Hz modes at `Ts=2.5e-4` sit above 0.97. All 8 true eigenvalues are at or just inside
the unit circle. Measured `|eig(df/dx)|` at six operating points:

| | max | spectrum at one point | count > 0.99 |
|---|---|---|---|
| at initialisation | 0.655 | 0.65, 0.65, 0.53, 0.27, 0.27, 0.23, 0.23, 0.06 | **0 of 48** |
| 73940, 130k updates | 1.0003 | 1.0002, 0.9998, 0.78, 0.78, 0.36, 0.13, 0.12, 0.001 | 12 of 48 |

- At init the model is nearly memoryless: `|lambda| = 0.65` forgets the state in about 2 samples.
- Training reached 1.000 on two eigenvalues and **overshot**. `1.0003^48000 = e^14.4`, which is the
  mechanism behind the `3.87e14` excursions. Those were never a learning-rate problem.
- The rest remain at 0.78 and below, decaying 600x per period of a 150 Hz oscillation, so the
  flexible dynamics are gone within 50 ms regardless of the input.

**Why the objective cannot fix this.** Holding position for 12 s to 0.1% needs `|lambda - 1| < 2e-8`.
Over the 400-step training window, `lambda = 1` and `lambda = 0.9999` differ by 4%, which is far
below the other error terms in the loss. The objective is close to blind to the quantity that
determines the metric. This is a property of short-window training on an integrating system, not of
this implementation.

## 3.3 The implementation is correct (tiers 1 and 2)

Same code path (`SS_encoder_general_hf` on 0.3.29 plus our `contiguous_*` nets), on Beintema's own
example plant from `deepSI-master/examples/docs/basic-example.py`:

| test | plant | nu/ny | untrained | hold last | **trained** |
|---|---|---|---|---|---|
| tier 1 | the v2 example | 1/1 | 1.03 | 1.25 | **0.024** |
| tier 2 | 3 coupled copies | 3/3 | 1.09 | 1.48 | **0.090** |

Script: `scripts/gantry/full-blackbox/ref_subnet_v2_example.py`. Tier 2 closes the gap noted as E1:
the stage-0 MSD control is SISO, so it never exercised the multi-channel window path. Together with
the train-path vs sim-path check (bit-exact, `0.000e+00` over 200 steps and 3 channels), the MIMO
adaptation is cleared by demonstration.

## 3.4 The controlled proof (tier 3)

`scripts/gantry/full-blackbox/msd_stability_contrast.py`. Jan's own MSD generator, three arms,
identical masses, cubic nonlinearity, `dt`, input signal, network, seed and budget. `Msd_ndof.deriv`
shows `k[0]` and `c[0]` are the only terms acting on an absolute coordinate, so zeroing both leaves
the internal flexible modes untouched and adds one free rigid-body mode.

| arm | `k[0]`, `c[0]` | true max abs(eig) | trained NRMS | **learned** max abs(eig) | > 0.99 |
|---|---|---|---|---|---|
| stable | 100, 0.50 | 0.996232 | **0.144** | 1.0000 | 10/30 |
| weak | 1, 0.05 | 0.999503 | **0.349** | 1.0034 | 4/30 |
| free | 0, 0.00 | **1.000000** | **3.795** | **0.8558** | **0/30** |

**As the plant's spectral radius moves from 0.9962 to exactly 1.0000, the trained free-run error
goes from 0.144 to 3.795.** The free arm ends worse than holding the last sample (1.436). Stable and
weak both learn a state map that reaches the unit circle; the free arm, the only one whose truth is
exactly 1, is the only one that fails to, at 0.8558 with none of its 30 eigenvalues above 0.99.
That is the gantry's signature on Jan's own benchmark, with one variable changed.

Caveats to carry with it: 6000 updates is a short budget and the free arm's learned spectrum was
still climbing (0.466 at init to 0.856), so this run alone does not exclude under-training; what
excludes it is the gantry, where 130k updates still left only 2 of 8 eigenvalues near 1. The free
arm's NRMS is also inflated by a scale effect (`y std` 161/60/68 across the contiguous split, so it
trains and tests at different amplitudes) and its epoch-0 NRMS is 7.94 against about 1.00 for the
others. The learned-spectrum column is scale-free and is the column to quote. One seed per arm; the
stable-to-free gap is far larger than noise, but nothing should be argued from `weak` alone.

## 3.5 A data-only diagnostic, with the qualifier that makes it work

Hold-last-sample NRMS **at the training horizon**, no model involved:

| | nf-window | full record | true max abs(eig) |
|---|---|---|---|
| stable | 1.433 | 1.452 | 0.9962 |
| weak | 1.061 | 1.230 | 0.9995 |
| free | **0.069** | 1.436 | 1.0000 |
| gantry | **~0.21** | ~1.01 | 1 by construction |

Monotone in the true spectral radius, and it places the gantry with the free arm. **Stated without
a horizon it is wrong**: over a full record the free arm loses to the mean despite never forgetting,
because a free chain drifts far beyond its own std. An earlier version of this analysis proposed the
full-record form as a general diagnostic; that is retracted, the horizon-qualified form stands.

## 3.6 Corrections to earlier parts of this document

- **Part 2's primary conclusion is superseded.** The objective normaliser is a real defect (five of
  fourteen records receive 0.00% of the gradient, confirmed) but it is second order. A paired
  training comparison at 900 updates
  (`scripts/gantry/full-blackbox/objective_train_diag.py`, results in
  `results/objective_train_diag/u900_*.json`) left all arms between 0.073 and 0.115 sim-RMS, none
  near the 0.047 floor.
- **The Theta thread is downgraded.** Theta is about `1e-9` of the sim-RMS metric and NRMS is about 1
  on every mode, so Theta weighting is not what blocks the model. The rank-2 conditioning finding
  itself stands (`std(X1-X2)/std(X1)` is 1.1e-4 on the large-motion records) and matters for any
  future metric that does try to score it.
- **"No validation record excites yaw" was wrong.** V1 and V3 carry Theta at the 5e-6 m level with
  `sigma3/sigma1 = 0.35`, i.e. genuinely rank 3. It is invisible in absolute metres, not absent.

## 3.7 What this means

The campaign asked whether a full ANN can learn this system. The answer, with evidence:
**not with a short-window objective and a 12 s free-run metric, and the reason is structural rather
than a defect in the setup.** SUBNET trains on short windows by design, which is sound for systems
that forget, because errors decay and short-horizon accuracy implies long-horizon accuracy. The
gantry integrates, so errors accumulate and that implication fails.

This is a result for the thesis rather than a failure of it. "An unconstrained black box cannot
free-run a marginally stable mechanical system, because the short-window objective it must be
trained on is blind to the long-horizon behaviour it is scored on" is the quantitative form of the
argument for augmenting a physical baseline. The FP model reaches `1.6e-4` untrained precisely
because it does not learn the integrator, it is one.

## 3.8 If a black-box column is still wanted for the results table

`SS_encoder_deriv_general` (the CT SUBNET, `encoders.py:392`, Beintema et al. arXiv 2204.09405) is
the in-framework model class whose state map is an integrator, so eigenvalues start at 1 rather than
0.65. Measured at initialisation, sweeping `tau`, Euler and RK4 identical to 4 digits:

| tau [s] | dt/tau | max abs(lam) | min abs(lam) | > 0.99 |
|---|---|---|---|---|
| 2.5e-4 | 1.0 | 1.5602 | 0.6931 | 18/48 |
| 2.5e-2 | 0.01 | **1.0056** | **0.9955** | **48/48** |
| 1.0 | 2.5e-4 | 1.0001 | 0.9999 | 48/48 |

It fixes the conditioning and **not** the stability: `max abs(lam) > 1` at every `tau` tested, so it
starts marginally unstable. Use `integrator_euler`; RK4 costs 4x the compute for an identical init
spectrum. `tau` is a genuine trade with no default (small `tau` keeps the net's `O(1)` outputs
representing the true `dx/dt`, large `tau` tightens the spectrum) and needs a labelled choice.
Run it to report a fair black-box baseline, not expecting `1.6e-4`.
