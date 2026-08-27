# Overnight 2026-08-21: which rung of the chain stops the augmented states from learning

Appended after each unit, never composed at the end. If the session died before the last row, every
row above it is still correct.

```
VERDICT:   LEAF B2a by the pre-registered gate, and CAPACITY by the arms. Both, and they are one
           mechanism: the objective FREEZES the augmented pole (it damps a correct mode, and the
           poles move under 0.15 Hz over 520 updates in both arms), and a frozen ONE-pole basis
           cannot span the mode. Give it four frozen poles spanning the band and the augmented
           states become load-bearing without any pole ever adapting.
             arm 1, nx_aug = 2:  free run 1.379891e-06, ablation 1.0183x  -> decoration
             arm 2, nx_aug = 8:  free run 3.795974e-07, ablation 5.2081x  -> load-bearing
           at MATCHED 520 updates, same lag (na_nb = 17), same everything else.
           The actionable rung is capacity, because T3 proved the objective cannot be fixed by any
           residual weighting while capacity can, and capacity routes around the frozen pole.
EVIDENCE:  C6, with the true absorber pole (r = 0.986982 at 157.8937 Hz, computed from the plant,
           no oracle constant) planted and the readout warmed off zero:
             condition 1  dL/d(nu_log) < 0 on 7 of 8 disjoint batches   (boundary >= 7 of 8)
             condition 2  nu_log monotonically increasing, 100 % of recorded steps
             over 150 steps r falls 0.986980 -> 0.986967 and f drifts 157.9120 -> 157.8178 Hz
           artefact scripts/gantry/closed-loop-controller/transient-investigation/runs/
           objective_sign_probe.json
           Upstream leaves eliminated first, in the plan's causal order:
             LEAF A  representation  ruled out by C2, planted free run 4.176627e-07 against a
                     1.869e-06 boundary, and a 6.010x ablation cost
             LEAF B1 estimation      ruled out by C8, encoder gap 0.997x against a 2.0x boundary
FIX:       None of the four candidate weightings can work, and that is a derived result, not a
           failure to find one. T3 proved the batch-consistent damping term is strictly positive
           under EVERY non-negative weighting, so no per-row, per-frequency or combined residual
           weighting can flip the sign; and for a narrowband mode a prefilter multiplies the whole
           loss change by |L(theta)|^2 > 0, leaving the sign exactly invariant. The one lever with
           real leverage is a time mask over the initial-condition transient, and it needs
           K_burn = 520 against nf = 400, so it does not fit inside a training window.
           Predicted effect of the runnable reduced form: none, 1.30e-06 to 1.39e-06, sign stays
           8 of 8 negative, ablation under 3 %.
IMPLEMENTED: AUG_LRU_NA_NB (scripts/gantry/gantry_dynamic/model.py, get_encoder_dims), env-gated,
           OFF by default, explicit cfg.na_nb_override still wins.
           D-072 line with the gate unset: 17  2  2.186601103417735e-06  rel dev 0.000e+00  PASS
           (runs/d072_noop_check.json), and no [aug-lag] line printed, so the default path is a
           certified no-op. This is not the B2a fix; it is what C1 made a prerequisite for any
           capacity arm, and it is the part of the night's fix set that is inside this session's
           scope. The two changes the diagnosis actually calls for both land in files the handoff
           puts out of scope, and are written out for one-step application instead of being
           applied: T3's per-row weighting (model_augmentation/fit_systems/interconnect.py:573)
           and the orthogonality extension (gantry_dynamic/orth_penalty.py:152-156).
TESTED?    Arm 3 NOT RUNNABLE (the B2a row's weighting needs burn-in at K_burn = 520 against
           nf = 400, and its reduced form needs an edit inside model_augmentation/, out of scope;
           the row states no fallback, so the handoff's rule is to say so and stop, not to
           substitute). Arms 1 and 2 both RAN, both truncated at 520 of 1300 updates by the host
           killing background jobs, both with the ablation measured on the best checkpoint. The
           capacity result is therefore tested; the objective fix is derived but untested.
CEILING:   9.13e-07 was never a floor, and arm 2 is below it. C9 decomposed it: a static
           correction removes 97.4 % of that out-of-band power (to 2.090e-07), while the two
           suspected sources are each about 1e-08, two decades down (encoder startup transient
           9.946e-09; baseline model error with no absorber present 8.146e-09). Defining the
           ceiling as "remove only the in-band part" overstates the irreducible error by more than
           a decade. Arm 2 reached 3.795974e-07. The genuine floor implied is ~1e-08, consistent
           with the data-derived 2.81e-08.
```

## What the headline metric actually is, stated because "free run" is misleading

Every RMS in this file, including `2.1866011034e-06` untrained, the `1.3933793e-06` plateau, the
`1.215e-06` target, the planted `4.176627e-07` and both arms, is the SAME quantity:
`closed_loop_free_run_rms` on the four validation records, aggregated as the quadratic mean.

| record | N | fs | duration | scored from `k0 = 17` |
|-|-|-|-|-|
| V1_standstill_Yp10 | 48000 | 4 kHz | 12.000 s | 11.996 s |
| V2_aprbs_Ylow | 48000 | 4 kHz | 12.000 s | 11.996 s |
| V3_ysweep_Yp10 | 48000 | 4 kHz | 12.000 s | 11.996 s |
| V4_lissajous_Ym10 | 48000 | 4 kHz | 12.000 s | 11.996 s |

**It is NOT an open-loop simulation.** The encoder sets `x0` once and the model state is never reset
for 11.996 s, but the model is driven by `u_data + C_fb(y_data - y_model)`, the stabilized-PEM
rollout, so the controller continuously corrects the input using the MEASURED output. "Free run"
here means "no state reset and no teacher forcing", not "no feedback". The comparison across all the
numbers above is apples-to-apples because they all use it, and it does discriminate (untrained
scores `2.19e-06` under the same loop). **But no measurement in this file reports open-loop 12 s
simulation performance, and arm 2's result must not be read as one.**

Per-record values for the two arms, so the aggregates can be checked and so it is visible that no
single trajectory carries them:

| arm | V1 | V2 | V3 | V4 | aggregate |
|-|-|-|-|-|-|
| arm 2 intact | `2.7824e-07` | `5.2879e-07` | `2.7950e-07` | `3.7579e-07` | `3.795974e-07` |
| arm 2 blind to `x_a` | `1.9700e-06` | `2.0006e-06` | `1.9697e-06` | `1.9675e-06` | `1.976996e-06` |
| arm 1 intact | `1.3411e-06` | `1.4645e-06` | `1.3499e-06` | `1.3605e-06` | `1.379891e-06` |
| arm 1 blind to `x_a` | `1.3709e-06` | `1.4902e-06` | `1.3809e-06` | `1.3751e-06` | `1.405174e-06` |

## What the two ablation surfaces mean

`y = Cd_norm x + Dd u` has **zero columns on the augmented states**, so `x_a` reaches the output ONLY
through ANN output rows 0-5. Literally "zero the readout's augmented columns" is a no-op here and
would report a false negative. The two surfaces that actually cut the route, both applied post-hoc
to the TRAINED model with no retraining:

* **A, blind to `x_a`**: zero the ANN INPUT columns holding `x_a` (columns 6..13 of `z = [x, u]` at
  `nx_aug = 8`). The ANN can no longer READ the augmented states; they still exist and still evolve,
  but nothing reads them into the physical rows.
* **B, `x_a` driven to zero**: zero the ANN OUTPUT columns that WRITE the augmented rows, so the
  recurrence loses its drive and `x_a` decays.

A and B agreeing to four digits on both arms is a consistency check on the ablation itself.

**What the ablation does NOT establish.** It shows that THIS trained model depends on `x_a`. The
model was trained with `x_a` available, so its learned function is built around it and removing an
input post-hoc is a severe perturbation regardless. It cannot tell you whether a same-sized ANN
TRAINED WITHOUT the extra augmented states would have reached the same place by another route. That
is what the width-matched control below is for.

## Units

| unit | hypothesis it tests | what ran | artefact | number | verdict | what it eliminated |
|-|-|-|-|-|-|-|
| W | the five missing probes and the C5 change exist and compile | wrote `probe_d072_matrix.py` (C1), `probe_representation_ceiling.py` (C2), `probe_wa_freerun.py` (C3), `probe_objective_sign.py` (C6), `probe_encoder_isolation.py` (C7+C8), `probe_out_of_band.py` (C9); env-gated `PROBE_EB_CKPT`/`PROBE_EB_OUT` on `probe_error_budget.py` (C4); env-gated `CL_NOISE_CONSISTENT` in `cl_train.py` (C5) | the six files above | `py_compile` ALL COMPILE OK on all seven scripts plus `cl_train.py` | DONE | nothing yet; this is the enabling block |
| C1 | does D-072 hold bit-identically at every `na_nb` x `nx_aug` an arm might use (open question section 7 row 4) | `probe_d072_matrix.py`, `na_nb` in {17, 32, 64, **103**} x `nx_aug` in {2, 8, 14}, all at `AUG_LRU=1 AUG_LRU_B=0.377`, JSON dumped after every cell | `runs/d072_matrix_probe.json` | (17, 2) `2.186601103417735e-06`, rel dev `0.000e+00`, **PASS**. (32, 2) `2.186893202561451e-06`, rel dev `1.336e-04`, **FAIL**. (64, 2) `2.1872077792682523e-06`, rel dev `2.775e-04`, **FAIL**. (103, 2) `2.1874819650488627e-06`, rel dev `4.028e-04`, **FAIL**. Deviation is **monotone in `n`**: `0 -> 1.34e-04 -> 2.78e-04 -> 4.03e-04`. **(17, 8) `2.186601103417735e-06`, rel dev `0.000e+00`, PASS.** **INCOMPLETE: 5 of the 9 pre-registered cells ran, plus the extra `na_nb = 103` column. Not run: (32, 8), (64, 8), (17, 14), (32, 14), (64, 14)**, the first host kill ending the sweep and a targeted re-run of the `nx_aug = 14` cells being killed before its first cell finished. The decision-relevant cells did run: `(17, 2)` gates arm 1 and `(17, 8)` gates arm 2 | **D-072 does NOT survive `na_nb != 17`**, and this CANCELS arm configurations | Answers section 7 row 4, which was open, with a NO. `W^b = A^n O_n^{-1}` is exact in exact arithmetic for any `n` above the observability index, but the pipeline is float32 and the conditioning of the observability inverse degrades with `n`: the deviation grows monotonically, `1.34e-04` at `n = 32` and `2.78e-04` at `n = 64`. That is the same mechanism T2 derived independently from the OLS slope-variance law and the same one `encoder_conditioning.json` measures, arriving here as a loss of baseline equality rather than as noise gain. **Consequence: T2's `na_nb = 103` arm is cancelled, and so is the B1 fallback `na_nb = 32`.** The gate is exact equality by D-090, not "close" |
| C2 | can the augmented route carry useful information at all, or is representation blocked (LEAF A) | `probe_representation_ceiling.py`, the `cl_capability.py` planted ANN loaded into an ungated build, free-run RMS on V1-V4, plus two ablations | `runs/representation_ceiling.json` | A0 untrained `2.186601103417735e-06` (D-072 exact). A1 planted encoder-init **`4.176627e-07`**, reproducing `cl_capability.json`'s `4.1766265955483893e-07` to every digit. A2 planted true latent `x0` `4.121625e-07`. A3 ANN blind to `x_a` `2.509986e-06`. A4 `x_a` driven to zero `2.509978e-06`. **Ablation `A3/A1 = A4/A1 = 6.010x`** | **representation CONFIRMED**, `-80.90 %` against untrained, `138.9 %` of the stated headroom recovered | **Eliminates LEAF A.** The pre-registered boundary needed `A1 <= 1.869e-06` and got `4.18e-07`, clearing it by more than a factor of four. The ablation is the gauge-free criterion of section 6.7 and it is emphatic: taking `x_a` away from a model that has been handed the physics costs `6.010x` and lands it **worse than untrained**. In the planted model the augmented states are the opposite of decoration. Recovering more than 100 % of "headroom" is not an error and not a ceiling violation: the `9.1327e-07` ceiling is the untrained error with only the in-band part removed, whereas the planted ANN also corrects out-of-band content |
| C3 | does the `W^a` zeroing claim transfer from the window metric to the free-run metric, and what should `ENC_WA_ZERO` be for arm 1 | `probe_wa_freerun.py`, three arms on the planted model, both metrics in one process, plus an untrained null | `runs/wa_freerun_probe.json` | **untrained**: `W^a` random and `W^a = 0` both `2.186601103417735e-06`, bit-identical, both match D-072. **planted, window**: random `1.206786e-06`, zero `7.615642e-07`, true `x0` `7.160347e-07`, all reproducing the recorded values to `1.0000x`. **planted, free run**: random `4.176627e-07`, zero `4.124427e-07`, true `x0` `4.121625e-07` | **the claim does NOT transfer.** `ENC_WA_ZERO = 1` for arm 1 | Eliminates "zeroing `W^a` is worth 1.59x". It is worth **1.585x on the window and 1.013x on the free run**. The `model.py:463` comment justifies the switch on the window number, which is the metric the loss minimises but not the one that decides; on the headline currency it is a **1.3 %** effect. The setting for arm 1 is unchanged, but its expected value is two decades smaller than recorded. The untrained null also states P1 cleanly: with the readout exactly zero, `W^a` cannot move `y` by even one float |
| C4 | how much in-band energy is LEFT at the plateau for the augmented states to claim (open question section 7 row 10) | `probe_error_budget.py` with `PROBE_EB_CKPT` on the Arm F best checkpoint under `AUG_LRU=1`, output redirected so the untrained artefact is preserved | `runs/error_budget_plateau.json` | free run `1.386786e-06` (reproduces the recorded Arm F `1.3841e-06`); **band share of error POWER falls 0.826 to 0.614**; in-band component `1.9891e-06` to `1.0866e-06`; error with the in-band part removed `8.616381e-07` | ANSWERED | Eliminates "the static ANN left the band untouched". It did not: it already removed **45 % of the in-band amplitude, i.e. 70 % of the in-band power**. But `1.0866e-06` of in-band energy remains, so the augmented states still have something substantial to claim, and claiming all of it would give `8.62e-07`, below the `1.215e-06` target |
| C6 | does the training loss damp a mode that is CORRECT, or only a randomly drawn one (P3) | `probe_objective_sign.py`: true mode computed from the plant, planted into `nu_log`/`theta_log`, D-072 re-checked, 40-step warm-up replicating `probe_consistency.py` exactly, then the sign test on 8 disjoint batches under BOTH the real training loss and the `consistency_probe` split loss, plus a discriminating arm with `x_a,0` forced to zero | `runs/objective_sign_probe.json` | D-072 bit-identical before and after planting, `2.186601103417735e-06`. Pristine init: `dL/d(nu_log)` **all exactly `0.000000e+00`** (that is P1, not P3). Warm-up preserved the plant: `r` `0.986982 -> 0.986980`, `f` `157.8937 -> 157.9120 Hz`. **Real loss: negative on 7 of 8.** Split loss: 6 of 8, against the recorded **8 of 8 at the DRAWN mode**. Discriminating arm, `x_a,0 = 0`: **5 of 8**. Run level, 150 steps on the LRU parameters only: `nu_log` `-4.334724 -> -4.333753`, **monotonically increasing on 100 % of recorded steps**, `r` `0.986980 -> 0.986967`, `f` `157.9120 -> 157.8178 Hz` | **P3 CONFIRMED, leaf B2a.** Condition 1 met (7 of 8), condition 2 met (monotone) | **Eliminates leaf B2b and settles the tree.** The objective damps the mode **even when the mode is exactly correct**, which is what P3 needed and what a single snapshot at a lucky draw could not establish. Planting the correct pole weakened the damping (8/8 to 6/8 on the matched loss) without reversing it, and over 150 steps descent walks `r` steadily DOWN. This is now a run-level property, not a one-snapshot artefact, which was the open item in handoff section 5 |
| C7 | does the data-derived band draw earn its keep against a Jan-faithful full-circle draw (section 8.0) | `probe_encoder_isolation.py PROBE_C7=1`, 8 draws per setting, each scored by the residual of a least-squares readout from `x_a` to the measured in-band residual | `runs/band_draw_probe.json` | band draw median `1-R^2` **0.98819** (best 0.98680, worst 0.99039); full circle median **0.99889** (best 0.99858, worst 0.99999); true mode planted **0.98842**. **The two sets do not overlap**: the worst band draw beats the best full-circle draw | band draw **WINS**, and this is its first direct evidence | Eliminates "the band recipe is not contributing". But the margin is small in absolute terms and the band draw is already AT the true-mode value (0.98819 vs 0.98842), which says pole placement is saturated and is not what is limiting |
| C8 | is the encoder the binding constraint: fit the augmented pathway against the measured in-band residual with the true `x_a` handed in and with the encoder's | `probe_encoder_isolation.py`, one model, one window set, one target, `A_aa` planted at the true mode, only the augmented initial condition differing; plus a TRUE-TRAJ ceiling arm added after the first run showed both arms equally uninformative | `runs/encoder_isolation_probe.json` | injection ON: TRUE `1-R^2 = 0.99511`, ENC `0.98842`, **gap = 0.997x** against the pre-registered 2.0x boundary. Injection OFF: gap 0.997x. **Ceiling arm TRUE-TRAJ: `1-R^2 = 0.35254`** | **gap SMALL, the encoder is NOT the binding constraint**; the tree goes to C6's outcome | Eliminates leaf **B1** (estimation). And the ceiling arm turns a null into a positive finding: with the true absorber state supplied at every sample a static 2-to-3 readout explains **65 % of the in-band residual variance**, so the two-state pathway is amply capable. Supplying it only at the window start is worth essentially nothing (0.995 against 0.353), because at `r = 0.987` the initial condition decays to `1/e` in 76 samples of a 400-sample window |

| C9 | what is the `9.13e-07` out-of-band remainder MADE OF, by source | `probe_out_of_band.py`, five arms scored identically and split into in-band and out-of-band error power: baseline encoder-init, baseline true `x0`, planted encoder-init, planted true `x0`, and the baseline on the NO-ABSORBER record from `trajectory/augmentation/baseline/` | `runs/out_of_band_probe.json` | A baseline out-of-band `9.126169e-07`, reproducing `error_budget.json`'s `9.132650e-07` to 0.07 %. **Encoder startup transient `9.946e-09` (0.0 % of A power). Baseline model error alone, no absorber present, `8.146e-09` (0.0 %). Left after the best STATIC correction `2.090168e-07` (5.2 % of A power).** Static-ANN capacity term `3.409e-08`. Time split shows the first 400 samples are NOT worse than the rest (ratios 0.558 to 1.186) | **The `9.13e-07` is not a floor. It is mostly still the absorber** | **Reframes the ceiling, which was the point of this gate.** The out-of-band remainder was uncharacterised and was being treated as irreducible. It is not: a static correction removes **97.4 % of its power**, taking it to `2.09e-07`. The two sources that were suspected, the encoder startup transient (section 4.3, the `7.76e+03` velocity rows) and baseline physical-parameter error, are each about **1e-08, two decades below**, and the time split independently kills the startup story by showing the first 400 samples are not the worst. So the absorber's contribution is not confined to `[140, 175] Hz`; defining the ceiling as "remove only the in-band part" overstates what is irreducible by more than a decade. The genuine floor implied here is `~1e-08`, consistent with the data-derived `2.81e-08` |
| arm 1 | do the augmented states help AT ALL, with a live pole AND a live input path | `cl_train.py`, `AUG_LRU=1 AUG_LRU_B=0.377 ENC_WA_ZERO=1`, `nx_aug=2`, `na_nb=17`, nf 400, stride 10, `lr=1e-5`, Adam `eps=1e-16`, no burn-in, no consistency term, no defect. **TRUNCATED at 520 of 1300 updates** by the host's background-job kill. Ablation measured on the best checkpoint by `probe_arm_ablation.py` | `runs/arm_ablation_arm1_520upd.json`, checkpoint `SSE_Interconnect_MultipleShooting_jBLNYQ_best.pth` | free run: untrained `2.1866011034e-06` -> **260 updates `1.383192160424035e-06`** -> **520 updates `1.379891240402659e-06`** (`-36.89 %` vs untrained, `-0.97 %` vs the plateau). **ABLATION: ANN blind to `x_a` `1.405174e-06` (1.0183x); `x_a` driven to zero `1.405157e-06` (1.0183x)** | **NEGATIVE on the primary criterion.** Stop condition not triggered; RMS secondary criterion beaten; but the augmented states are **decoration** | **Answers section 7 row 1, the handoff's headline question, with a NO.** The augmented states carry `1.83 %` of the free-run RMS against the `6.010x` that gate C2 measures for a model which genuinely uses them, i.e. about 0.3 % of what the same pathway is capable of. And the `-36.89 %` is NOT evidence for them: it is the static augmentation again, reproducing the known `-36.3 %` result. Two independent surfaces agree to four digits (1.0183x both), which is what you expect when the augmented route is carrying almost nothing |

| C5 | is the D-150 "survives Telica-level noise" claim resting on a non-physical gate | code change only, not a training run: `CL_NOISE_CONSISTENT=1` added to `cl_train.py`, perturbing `y -> y+v` **and** `u -> u - C_fb(v)` together, using a controller bank built at the training rate, noise in normalised units and the correction returned to physical units | `cl_train.py`, the `CL_NOISE_SIGMA` block | implemented and compiling; the existing `y`-only path is untouched and still the default | **IMPLEMENTED, NOT RUN** | Nothing yet, and that is the honest state. It removes the objection rather than answering it: the default gate injects a spurious `+C_fb(v)` that a real machine would have cancelled, so it tests a harder non-physical problem, and D-150's noise claim rests on it. Running the pair is a next-session job. `alpha_cancellation.json` already shows the cancellation is exact whole-record and degrades to `2.0x` at `nf=400` |
| arm 2 | is `nx_aug = 2` the binding constraint, tested WITHOUT the encoder-lag confound for the first time | `cl_train.py` with `CL_NX_AUG=8 AUG_LRU_NA_NB=17`, serial validation forced, matched to arm 1 at **520 updates**; `CL_NX_AUG` added to `cl_train.py` and the lag pin to `model.py` to make this runnable at all. Ablation on the best checkpoint by `probe_arm_ablation.py` | `runs/arm_ablation_arm2_nx8_520upd.json`, `runs/cl_train_arm2_nx8_overnight.json`, checkpoint `SSE_Interconnect_MultipleShooting_9cyquw_best.pth` | D-072 at init `2.1866011034e-06`. 260 updates `3.8220142105593476e-07`; **520 updates `3.795973722364048e-07`** (`-82.64 %` vs untrained, `-72.76 %` vs the plateau), reproduced in-process by the ablation probe to 7 digits. **ABLATION: blind to `x_a` `1.976996e-06` (5.2081x); `x_a` zeroed `1.977064e-06` (5.2083x)** | **POSITIVE on the primary criterion. The augmented states ARE load-bearing** | **The night's largest result, and it reverses the reading arm 1 alone would have supported.** At matched updates, `nx_aug` 2 to 8 takes the free run from `1.3799e-06` to `3.7960e-07`, a factor **3.63**, and takes the ablation from `1.02x` (decoration) to **`5.21x`**, which is **87 % of the planted model's `6.010x`**. It clears the `1.215e-06` target and sits below the `9.13e-07` figure that had been treated as a ceiling, which C9 independently showed was never a floor. It is also below the planted model's own `4.177e-07`, which is not a contradiction: the planted ANN was regressed onto the exact ONE-STEP correction, while this arm optimises the ROLLOUT objective, and those have different minima |
| arm 3 | the leaf's fix, trained | **NOT RUNNABLE**, and per handoff section 2 that is recorded rather than substituted | none | none | **NOT RUNNABLE** | The B2a row calls for "the weighting T3 derives". T3 derived it and then proved it cannot work: the batch-consistent damping term is strictly positive under EVERY non-negative weighting, so no residual weighting can flip the sign, and a narrowband prefilter multiplies the whole loss change by `\|L(theta)\|^2 > 0`. The only lever with leverage is a time mask needing `K_burn = 520` against `nf = 400`, i.e. longer than the window, and burn-in is on the do-not-re-run list. The reduced form that would fit at `nf = 400` needs a consumer edit at `model_augmentation/fit_systems/interconnect.py:573`, which is out of scope for this session. **The B2a row states no fallback**, so the handoff's rule is to write "not runnable" and stop rather than substitute |

## The three trained models, side by side, with the spectrum (2026-08-21 morning)

All at 520 updates, `na_nb = 17`, serial validation, same lr and eps. The only differences are the
two named columns.

| | ANN params | `nx_aug` | free run | in-band | out-of-band | **ablation** | in-band ratio | out-of-band ratio | in/out preference |
|-|-|-|-|-|-|-|-|-|-|
| arm 1 | 600 | 2 | `1.379891e-06` | `1.057645e-06` | `8.862399e-07` | **`1.0183x`** | 1.017x | 1.020x | **0.997** |
| width control | **828** | 2 | `1.384274e-06` | `1.101039e-06`* | `8.769091e-07`* | **`1.0169x`** | 1.009x | 1.029x | **0.980** |
| **arm 2** | 798 | **8** | **`3.795974e-07`** | `2.858956e-07` | `2.497579e-07` | **`5.2081x`** | **6.241x** | 3.409x | **1.830** |

`*` the control's in-band and out-of-band columns are its BLINDED values, since its intact split was
not printed; its ratios are exact.

Reference: untrained in-band `1.989087e-06` (share 0.826); Arm F plateau in-band `1.086626e-06`
(share 0.614).

**Three readings, and the third is the one that matters.**

1. **ANN size is not the lever.** The control carries the most parameters of the three and lands
   0.32 % WORSE than arm 1. Its augmented states are decoration (`1.0169x`), exactly as arm 1's are.
2. **Arm 1 and the control are spectrally flat nulls.** Their ablations move in-band and
   out-of-band by the same 1-3 %, preferences `0.997` and `0.980`. A single detuned pole is not
   weakly helping the mode; it is uniformly irrelevant.
3. **Arm 2 acts preferentially IN BAND.** `6.241x` in band against `3.409x` out, preference
   `1.830`. It takes the in-band residual from the untrained `1.989e-06` to `2.859e-07`, a **7.0x
   reduction in the mode's own band** and 3.8x below the plateau. Blinding it returns the in-band
   error to `1.784e-06`, i.e. nearly all the way back to untrained, so in arm 2 the labour has
   divided: the augmented route does the in-band work the static ANN does not.

The effect is not purely in band, and that is expected rather than contradictory: C9 established
that the absorber's contribution is not confined to `[140, 175] Hz`.

## The readout Jacobian, and a prediction of mine that it FALSIFIED

`d w[0:6] / d x_a` is the single gate through which the augmented states reach the loss, since
`Cd_norm` has zero columns on them and the only path is `x_a -> ANN rows 0-5 -> x_phys -> y`, one
step late. Measured by `probe_readout_jacobian.py` at 400 operating points harvested from the
model's own closed-loop rollout, not at the origin. Artefacts `runs/readout_jacobian_*.json`.

| model | `\|\|J_aug\|\|_F` | `\|\|J_phys\|\|_F` | ratio aug/phys | sum of per-pair products |
|-|-|-|-|-|
| untrained | **`0.000000e+00`** | `0.000000e+00` | n/a | `0.000000e+00` |
| arm 1, `nx_aug=2` | `1.046098e-03` | `1.332348e-03` | **0.785** | `7.132750e-04` |
| arm 2, `nx_aug=8` | `1.083590e-03` | `8.073450e-04` | **1.342** | `1.208191e-03` |

Per pair in arm 2: `3.017e-04`, `5.813e-04`, `5.906e-04`, `6.268e-04`, with `RMS(x_a)` per pair
`0.485`, `0.495`, `0.346`, `0.909`.

**The untrained row is the clean confirmation of P1 at the gate itself**: `J` is exactly zero, to
the last bit, which is why `dL/dx_a` and `dL/d(nu_log)` are exactly zero and why nothing in the
augmented block can learn until the readout leaves zero.

**I predicted "the gate wide open in arm 2 and nearly shut in arm 1". That is wrong.** The total
gate opening is essentially the SAME in both, `1.046e-03` against `1.084e-03`, 3.6 % apart. Arm 2's
four pairs are individually SMALLER than arm 1's single pair; the total is merely spread over four.
Even the sum of `\|\|J_pair\|\| * RMS(x_a,pair)` differs by only **1.69x**, against an ablation
difference of **5.21x versus 1.02x**.

**So the ablation gap is NOT explained by how strongly the ANN reads `x_a`.** It has to be explained
by what `x_a` CONTAINS, i.e. whether the injected correction is useful, not whether it is read. The
one statistic here that does move in the right direction is the ratio: in arm 1 the ANN's
sensitivity to the physical inputs dominates its sensitivity to `x_a` (0.785), while in arm 2 the
augmented route dominates (1.342). That is a real 1.71x shift in where the ANN gets its information,
and it is consistent with the ablation, but it is not of the size of the ablation effect.

**The most likely mechanism, stated as a hypothesis and NOT as a measurement.** Arm 1's single pole
sits `3.35 Hz` from the true mode. Over a 12 s record that is about 40 cycles of relative phase
drift, so whatever it injects goes in and out of phase with the residual and averages away: read
strongly, useless anyway. Four poles spanning `151.995` to `162.854 Hz` can be combined by the
readout into something that stays phase-coherent with the mode for far longer. That is the "span"
story sharpened from gain to phase coherence. **The direct test is the in-band residual spectrum of
the two trained arms, which has not been run.**

## THE SYNTHESIS: why C6 and arm 2 are both right

C6 says the objective damps a correct mode. Arm 2 says more modes fix the problem. Those look
opposed and are not, and the pole tables are what reconcile them.

**Trained pole sets, read from the checkpoints** (`scratchpad/poles.py`, true mode `r = 0.986982`
at `157.8937 Hz`):

| arm | drawn at init | after 520 updates | moved |
|-|-|-|-|
| arm 1, `nx_aug=2` | `154.52 Hz` | `r 0.992038` at `154.543 Hz` | `0.02 Hz` |
| arm 2, `nx_aug=8` | `159.26 / 153.34 / 162.94 / 151.94 Hz` | `0.992032 @ 159.350`, `0.986527 @ 162.854`, `0.985932 @ 151.995`, `0.984708 @ 153.475` | all under `0.15 Hz`; **all four still live**, none decayed toward `r -> 0` |

**The poles do not move, in either arm.** That is C6's finding seen from the other side: the
gradient into the pole is damped, so the resonator cannot walk to the mode. It also rules out T4's
"coverage" branch of its own diagnostic, which required one pair to migrate onto the mode while the
others decayed. Nothing migrated and nothing decayed.

So the augmented block is not behaving as an ADAPTIVE resonator. It is behaving as a **fixed
random-feature basis, whose SPAN is what matters**, with the ANN readout learning the combination.
And that reconciles everything measured tonight:

* the objective damps the pole, so the pole cannot adapt (C6, 7 of 8 and monotone);
* with ONE fixed pole `3.35 Hz` off the mode and no ability to adapt, the span misses and the route
  is dead (arm 1, ablation `1.02x`);
* with FOUR fixed poles spanning `151.995` to `162.854 Hz`, the span brackets the mode, no
  adaptation is needed, and the readout can synthesise it (arm 2, ablation `5.21x`).

**T4's half-power argument was the wrong criterion, and this is the cleanest thing the night
falsified.** T4 reasoned that because the mode's half-power width (`18.44 Hz`) exceeds the band
width (`14.16 Hz`), any single in-band draw is "close enough", and predicted a near-null capacity
arm: free run in `[1.30e-06, 1.42e-06]`, falsified below `1.25e-06`, ablation degradation `<= 1.2 %`,
falsified above `5 %`. Measured: `3.80e-07` and `421 %`. **Both falsifiers fired, by a factor of 3.3
and a factor of 84.** The half-power width bounds how much a single resonator's GAIN is attenuated
off-peak; it says nothing about whether a one-dimensional span can represent the mode's contribution
once the pole is frozen. To T4's credit the falsifiers were pre-registered, numeric, and it flagged
these very numbers as its least grounded.

**The corrected reading of leaf B2a.** C6's pre-registered conditions were met and the verdict
stands as measured: the objective does damp a correct mode. But the actionable fault is capacity,
because the objective cannot be fixed (T3 proved no residual weighting can flip the sign) while
capacity can, and adding capacity routes around the frozen pole entirely. The chain is: **objective
freezes the pole, and a frozen one-pole basis cannot span the mode.** Arm 1 and arm 2 differ by
exactly that.

**The confound is now CLOSED. The width-matched control was run on 2026-08-21 and the capacity
claim stands.**

Arm 2 changed `nx_aug` 2 to 8, which also widens the ANN itself (8 output rows to 14, `nz` 11 to
17, so 600 ANN parameters to 798). The encoder lag was never a confound, both arms being at
`na_nb = 17`, which is what the pin bought. But ANN width was, and the ablation alone could not
separate them: it shows the augmented ROUTE is used, not that the gain came from capacity rather
than from network size.

The control is `nx_aug = 2` with `n_nodes_per_layer` raised 16 to 20, giving **828** ANN parameters,
i.e. slightly MORE than arm 2, with only two augmented states. Everything else identical, 520
updates matched. Decision rule fixed before launch: below `6.0e-07` withdraws the capacity claim,
above `1.1e-06` supports it.

| at 520 updates | ANN params | `nx_aug` | free-run RMS |
|-|-|-|-|
| arm 1 | 600 | 2 | `1.379891e-06` |
| **width-matched control** | **828** | **2** | **`1.384274e-06`** |
| arm 2 | 798 | 8 | `3.795974e-07` |

**The control lands on top of arm 1 and 3.6x away from arm 2, and is in fact 0.32 % WORSE than arm 1
despite carrying the most ANN parameters of the three.** More static capacity than arm 2 buys
nothing. Arm 2's 3.63x is attributable to the augmented states, not to the wider network.

Residual caveats, stated so this is not over-read: `CL_NODES` also widens the ENCODER net (same
`n_nodes_per_layer` feeds `e_net_kwargs` and `linear_encoder_init_aug`), so the match is on ANN
parameter count and not on every parameter in the pipeline; and the frozen-pole variant, which would
test the fixed-basis mechanism directly rather than the capacity attribution, has still not been
run.

## Side finding that validates the band recipe, and it is thesis-relevant

C6 and C8 both need "the correct mode" and neither is allowed an oracle constant, so both compute it
from the plant itself: linearise `plant.deriv8` (truth) and `plant.deriv6` (baseline) numerically at
`Y = 0`, discretise with `expm(A*Ts)` at `Ts = 2.5e-4`, and take the oscillatory eigenvalue with no
counterpart in the 6-state. The result, printed with both full pole tables so it can be audited:

```
deriv6 poles: |1.000000| @ 0.00 Hz  x2   |0.999254| @ 5.13 Hz x2   |0.999752| @ 0.00   |0.999838| @ 0.00
deriv8 poles: the same six, plus  |0.986982| @ 157.89 Hz  (conjugate pair)
selected: r = 0.986982,  f = 157.8937 Hz    (distance to nearest deriv6 pole 2.38e-01,
                                             against 9.87e-06 for the only other candidate)
```

**The true absorber mode is `r = 0.98698` at `157.89 Hz`. The data-derived band recipe returns
`[149.90, 164.06] Hz` with `rho in [0.9794, 0.9956]`, and the artefact's own summary fit is
`rho = 0.98560` at `158.203 Hz`.** The recipe, which uses only `u`, `y` and the baseline and never
sees the truth, lands `0.31 Hz` and `0.0014` in radius from the true mode. That is a direct
validation of `lru_band_from_artifact` against ground truth, and as far as the records show it has
not been made before.

**Method note, recorded because it bit once.** The first version of this extraction matched each
8-state eigenvalue to its nearest 6-state one greedily, removing as it went, and returned a spurious
`f = 0.00 Hz` mode. X and Y are `K = 0` double integrators contributing repeated eigenvalues at
`z = 1`, which the greedy pass mis-assigns. The C8 run made with that bug is superseded; the fix
scores every oscillatory candidate against the nearest deriv6 pole with no removal, and prints both
tables. Any future use of this extraction should keep the printed tables.

## Derivations T1-T6

| unit | cross-check artefact it had to reproduce | pass/fail | deliverable |
|-|-|-|-|
| T1 `W^a` | planted-model window RMS ordering `1.2068e-06` (random) > `7.6156e-07` (zero) > `7.1603e-07` (true `x0`), `cl_capability.json` | **PASS** | `W^a = 0` is not a convention, it is Hoekstra arXiv:2602.13108 Eq. (7) (the encoder approximates `E[x_a | psi]`) evaluated at readout gain zero, where `x_a` is independent of the window so the conditional mean IS zero. General form `W^a(eps) = Sigma_a O_a^T [O_a Sigma_a O_a^T + Sigma_v]^{-1}`, whose `eps -> 0` limit is 0 and whose `Sigma_v -> 0` limit is Hoekstra Eq. (17). Bayes identity `MSE(W) = MSE(0) + E||W psi||^2` makes any non-zero draw strictly worse, which is the measured ordering with no free parameter. Noise gain bounded by `sigma_a / (2 sigma_v)` independently of `cond(O_a)`, so it does not inherit the `7.76e+03` problem. `W^a = 0` is the unique fixed point of the gauge group, so it is the only value that commits to no gauge |
| T2 lag rule | velocity-row amplification `1.000 / 0.396 / 0.137 / 0.043` at `n = 17/32/64/128`, `encoder_conditioning.json` | **PASS** | Derived `sqrt(12/(N(N^2-1)))/Ts` with `N = na_nb + 1` (the `pre_encoder.py:376` stacking convention), i.e. the OLS slope-variance law, matching the two `K = 0` axes to better than 1.5 % across the sweep (fitted exponents dX `-1.5052`, dY `-1.5078` against theory `-1.5000`). Recommended `na_nb = 103` (`N = 104` = exactly 4.00 periods of the 154 Hz mode) |
| T3 objective | 8 of 8 negative `dL/d(nu_log)` batches AND the drawn pole `r = 0.99204002` at `154.51750 Hz`, `consistency_probe.json`, both re-derived independently | **PASS** (both) | **A negative deliverable, and it is the sharpest result of the six.** Decomposing the mode's contribution into a driven part and the free response of the encoder-set initial condition gives `d\|\|s_ic\|\|_v^2/dr = sum_k v_k c_k 2k r^{2k-1} > 0` strictly, for **any** non-negative weight sequence `v_k`; with `dr/d(nu_log) = r ln r < 0` that term alone forces `dL/d(nu_log) < 0` on every batch. **Corollary: no weighting of the residual, per-row, per-frequency or any combination, can flip the sign.** Theorem (W1): for a narrowband mode a prefilter multiplies the whole loss change by `\|L(theta)\|^2 > 0`, so the sign and the help/hurt condition are exactly invariant to it. Recommendation: **do not run a weighting arm** |
| T4 band coverage | Arm F: pole 154.52 Hz, free run `-0.665 %`, `rho` 0.9920 to 0.9920, `f` 154.52 to 154.56 Hz | **PASS**, and it **refutes the hypothesis T4 was assigned to defend** | Recomputed from `cl_residual_spectrum.json` by the artefact producer's own rule: 54 peaks, band width `W = 14.16 Hz`, median `zeta = 0.0583` at `f_n = 158.20 Hz`, so the mode's half-power width is `Delta_f = 2 zeta f_n = 18.44 Hz`. **The band is NARROWER than the mode it is supposed to cover**, so `n_pairs = 1` is sufficient and every in-band draw lands inside the 3 dB skirt. The drawn pole is `0.40` of a half-power half-width off peak, i.e. `-0.64 dB`. 50 of 54 estimates (93.5 %) give `Delta_f >= W`. Predicts `nx_aug` 2 to 8 lands in `[1.30e-06, 1.42e-06]` and the ablation degrades by `<= 1.2 %`, i.e. decoration |
| T5 gauge | burn-in decay `0.97510 / 0.97686 / 0.97917` (global `0.97758`) against the independently measured planted `rho_aa_median = 0.9755435` | **PASS**, agreeing to 0.2 % | **The most consequential derivation of the six, and it is about the thesis contribution rather than about tonight's plateau.** Gauge audit: of the four continuous parameters of the `2x2` similarity group, the rotation-scaling parameterisation `r*[[cos w, -sin w],[sin w, cos w]]` already fixes two, and **two remain**, carried by `B_a`, `W^a`, the encoder net's augmented output rows and the ANN readout rows. Recommends Route A (balanced SCALE step, which removes a standing `22.9x` imbalance) plus Route C (project the gradient off the 2-dimensional gauge tangent space, `# THEORY: McKelvey and Helmersson, Proc. 36th IEEE CDC 1997, pp. 2986-2987, Sec. 2.1-2.2 Eq. (7)`). **Phase must be gated, not shipped**: Glover Lemma 4.1 (Int. J. Control 39(6), 1984, pp. 1129-1130) makes a balanced realisation unique only up to an orthogonal `T` commuting with `Sigma`, and our pair's Hankel singular values differ by only **1.21 % to 3.32 %**, so the residual group is nearly the full `SO(2)` and the phase is not fixable by balancing at all |
| T6 windowed closed loop | exact cancellation whole-record vs saturating `2.0x` at `nf=400`, `alpha_cancellation.json`, all five rows | **PASS** (the `sigma`-independence is derived exactly; the constant is bracketed 1.7 to 2.2, not pinned) | Recommends **keep `xc = 0`** (option a), and rejects the other three. `xc = 0` in the residual form is proved IDENTICAL to Kessels Eq. (5.13d) plus Remark 5.4 (verified numerically on the real 9-state `Cfb`, max relative difference `3.951e-14`), so it is the published method rather than a shortcut. Under Sugie Remark 2 a windowed reset is a `Khat != K` perturbation, whose documented cost is **variance, not bias** |

### T1 corrections to the record (both are corrections to this project's own comments)

1. **`W^a` random DOES have a Hoekstra source, and the code says it does not.**
   `pre_encoder.py` labels the kaiming draw `HEURISTIC, with no literature source`, and
   `model.py:456-468` repeats it. Hoekstra, Gyorok, Verhoek, Toth, Schoukens, arXiv:2602.17297
   (the EJC extended version, local at `literature/closed-loop-id/hoekstra2026_lfr-augmentation-fp-models.pdf`),
   p. 9 Sec. 5.4.2 Eq. (31): *"the weights and biases of psi_aug are initialised by the Xavier
   approach"*, and p. 10: all matrices not fixed by baseline equality are drawn `m ~ U(-1,1)`. So
   the random draw is Hoekstra's stated convention. It is refuted here by the Eq. (7)
   conditional-mean argument, not by absence of a source, and the comment must be corrected.
2. **The brief's attribution of the Xavier draw to Eq. (31) of arXiv:2602.13108 was wrong.** Eq. (31)
   of the encoder paper is the Jacobian linearisation. Two Hoekstra papers renumber; the equation
   number must be verified against the specific PDF.
3. **The same group's newest paper abandons the augmented encoder.** Hoekstra et al.
   arXiv:2604.11421 (submitted to Automatica, 2026-04-14) p. 4 Eq. (12) treats `x_hat(0)` as a free
   optimisation variable instead. That is the authors themselves regarding the augmented encoder
   block as unsettled.

### T1 operational finding that constrains arm 1

`ENC_WA_ZERO=1` must NOT be run alone. With `W^a = 0` and no input path, `x_a` is identically zero
for the whole window, `dL/d(readout) = dL/dy * x_a = 0`, and the augmented subsystem is a dead fixed
point that nothing ever leaves. Random `W^a` has exactly one virtue, that it makes `x_a` non-zero so
the readout receives gradient at all, and it pays for it with an unobservable O(1) bias at every
window start. The right move is to keep the excitation and move it from the initial condition to the
input path: `ENC_WA_ZERO=1` **together with** `AUG_LRU=1` and `AUG_LRU_B`. That is exactly the arm-1
configuration, so arm 1 is safe; a future `ENC_WA_ZERO=1` run without `AUG_LRU_B` would not be.

### T2 operational finding that constrains arm 3 (leaf B1)

`na_nb = 103` needs no code change to run: `RunConfig.na_nb_override = 103` already exists and takes
precedence. `K0` moves 17 to 103, encoder parameters go 12,456 to 49,608, and the orthogonal
projection basis must be rebuilt for the new window length. T2's own prediction for the arm is
**no material change on this data**, because the simulation is noiseless (`cfg.snr = None`) and the
derivation is a variance law: predicted plateau `1.32e-06` to `1.46e-06`, falsified below
`1.20e-06`. Its cheap decisive pre-test (untrained free-run RMS at `na_nb = 103`, predicted
`1.90e-06` to `2.20e-06`, falsified below `1.60e-06`) is folded into C1 as a fourth `na_nb` column,
so it costs nothing extra.

### THE FINDING OF THE NIGHT, and it is about the thesis contribution, not about the plateau

**The orthogonal-projection penalty cannot see the dynamic augmentation. Measured, not argued.**

T5 derived it by reading the code; it was then verified directly by `probe_orth_gauge.py`, which is
one backward pass. Artefact `runs/orth_gauge_probe.json`.

| group | `\|\|dV_orth/dp\|\|` | exactly zero |
|-|-|-|
| `W^a_psi_y` | `0.000000e+00` | yes |
| `W^a_psi_u` | `0.000000e+00` | yes |
| encoder net (all) | `0.000000e+00` | yes |
| `B_a` (D-151) | `0.000000e+00` | yes |
| `nu_log` | `0.000000e+00` | yes |
| `theta_log` | `0.000000e+00` | yes |
| ANN layer-1 columns on `x_a` | `0.000000e+00` | yes |
| **ANN layer-1 columns on `(x_phys, u)`** (control) | **`1.393244e+00`** | no |
| **ANN final layer weight** (control) | **`2.765908e+01`** | no |

The mechanism, read off the built penalty object rather than off the source:
`Z_pts` has shape `(6718, 11, 1)` with `max |Z_pts[:, x_aug]| = 0.0` **exactly**
(`orth_penalty.py:152-156` allocates zeros and writes only the physical-state and input slots), and
`route_cols = [0, 1, 2, 3, 4, 5]`, which **excludes** the augmented output columns `[6, 7]`
(`orth_penalty.py:192`). So the penalty evaluates the ANN at `x_a = 0` and reads only physical
output rows.

Gauge invariance was also confirmed directly: applying `T = 2.3 R(0.7)` to the augmented block, a
similarity on `A_aa` that leaves the input-output behaviour unchanged, leaves `V_orth` **bit-identical**
at `0.37444624304771423`.

**A necessary control, and the reason the first attempt proved nothing.** At initialisation the ANN
output layer is exactly zero, so the stacked field is zero, `V_orth = 0.0`, and its gradient
`2 beta Q Q^T f` is zero for EVERY parameter including the static ones. The first run returned nine
zeros, which is uninformative. The numbers above are taken after perturbing the ANN final layer by
`N(0, 1e-2)`, which takes the field off zero and makes the static group non-zero. The dynamic group
staying at exactly `0.000000e+00` through that perturbation is what makes the claim STRUCTURAL
rather than an artefact of a stationary point.

**What this means, in one sentence.** The regulariser that is this thesis's scientific contribution
currently constrains only the STATIC augmentation, and the dynamic path
`(x_phys, u) -> x_a -> ANN rows 0-5 -> y` is an entirely unpenalised negation channel.

**And C2 prices that channel.** Blinding the ANN to `x_a` costs the planted model **6.010x** on
free-run RMS. So the unpenalised path is not a corner case; it is worth six times the planted
model's performance. The invariance is invariance by blindness.

This does not invalidate any result obtained so far, because no run to date has had a materially
non-zero augmented block (the trained ungated arm sits nine decades from minimality). It does mean
the contribution's claim must be stated as covering the static augmentation until the penalty is
extended. The smallest honest fix is T5's Route B: penalise the augmentation's **Markov parameters**
`C_a A_aa^k B_a`, which are similarity invariants and therefore need no canonicalisation, against
`d/d theta` of the baseline's. T5's Route A (canonicalisation) is the alternative and needs a gauge
convention; both are written out in its report.

**Recommended as a permanent regression guard** (T5's suggestion, and it is cheap): a Step-8 parity
assertion that applies a random `T = s R(phi)` and asserts `V_orth` is bit-identical. If it ever
fails, someone has made the penalty `x_a`-aware without noticing they also made it gauge-dependent.

**T5's own self-refutation item 5 is now closed.** Its consolidated report states: *"The central
claim of Section 1 is analytic, not measured. I read the code path and reasoned; I ran no gradient
probe. The five-line parity check in Section 6 is what would convert it from a reading to a
measurement, and it should be run before this goes in the thesis."* That check was run tonight, in
the form above, and it confirms the claim with a working control. The finding is a measurement, not
a reading, and it is safe to take into the write-up on that basis.

**One caveat T5 raises that is NOT closed**, and it should gate any canonicalisation work: the
`22.9x` gauge imbalance is an upper bound, because `ann_out_rms_phys` is the ANN's total
physical-row output including its direct `(x_p, u)` dependence, not only its `x_a` dependence. The
missing measurement is `d w[0:6] / d x_a` directly. That should precede any `AUG_CANON` run.

### T3 operational finding that cancels an arm, and the probe it replaced it with

The one weighting with any leverage on the damping term is a TIME mask over the transient, and its
required length is `K_burn = ceil(ln(eps) / (2 ln r))`. At the top of the radius band the data
itself supports (`r_max = 0.9955750`) that is **`K_burn = 520` against `nf = 400`**: the initial
condition transient never dies inside one training window, so the only lever with leverage is
inapplicable at the available window length. It is blocked twice over, by the do-not-re-run list and
by the window geometry. T3's recommendation is therefore recorded as the result: **objective defect
located and characterised, predicted fix is a no-op at `nf = 400`, not worth an arm.**

In its place T3 named a five-minute measurement that decides the leaf, and it has been folded into
C6 as a third arm: recompute `dL/d(nu_log)` with `x_a,0` forced to zero (zero the encoder `W^a`
rows) while KEEPING the input path `AUG_LRU_B`. A fall toward 4 of 8 confirms that the damping is
the encoder initial condition wearing an objective disguise, which is leaf **B1**; staying at 8 of 8
refutes the mechanism and leaves the objective as the wall, which is leaf **B2a**.

**Result: 7 of 8 with the random `W^a`, falling to 5 of 8 with `x_a,0 = 0`. The direction matches
T3's prediction, and the test is UNDERPOWERED to confirm it.** Eight batches cannot separate these:
under the coin-flip null `p = 0.5`, `P(>= 5 of 8) = 0.3633` and `P(>= 7 of 8) = 0.0352`. So 7 of 8
is unlikely to be chance while 5 of 8 is entirely consistent with chance, which is the shape T3
predicted, but the two-batch difference itself carries almost no evidential weight at this sample
size. Recorded as **consistent with T3's initial-condition mechanism, not confirmation of it**. The
cheap fix if anyone wants this settled is more batches, not a different measurement: 32 disjoint
batches would separate `p = 0.5` from `p = 0.875` cleanly.

This also sits well with C8, which found by a completely different route that the initial condition
is worth almost nothing once the mode is right (gap `0.997x`). Both say `x_a,0` is a minor term. The
difference is that C8 measured it on fit quality and C6 on the gradient sign.

### T3 and T4 corrections to the record

4. **"Landau Eq. 32" is not a frequency-weighted criterion.** In
   `literature/closed-loop-id/landau-karimi-2002-ejc-duality.pdf` (a 20-page preprint with its own
   numbering) Eq. 32 is `nu(t+1) = -(R/S) eps_CL(t+1)`, the closed-loop input error. The
   frequency-weighted result in that file is Eqs. 11 to 13 (p. 5) and Eq. 42 (p. 13). Any document
   citing Landau Eq. 32 for a weighted criterion is wrong.
5. **Use Wahlberg and Ljung 1986 for the bias-distribution formula, not Ljung's textbook.** IEEE TAC
   31(2):134-144, p. 138 Eqs. (4.9b) and (4.12): the design variables enter only through
   `Q = Phi_u |L|^2 / |H|^2`. Ljung's (14.19) and (8.71) are unreadable in the available scan and
   remain UNVERIFIED as printed formulas.
6. **A prefilter in closed loop is not free.** Pintelon and Schoukens 2nd ed. p. 524 Remark (i):
   prefiltering changes the noise model, and in practice "a bias can even appear in closed loop
   identification". Our setting is closed loop, so the prefilter arm is not merely predicted useless
   but mildly risky.
7. **The `# THEORY: Orvieto Lemma 3.2` label on the one-pair draw over-reaches.** The radius formula
   is correctly attributed. The phase-arc restriction is NOT in Lemma 3.2 (it is Sec. 3.4, p. 11,
   empirical, for PathX). And the implicit claim that one pair is a legitimate instance is supported
   by nothing in the paper: everything offered as a reason to ring-initialise is population level
   (Thm 3.1 is `N -> infinity`, Prop 3.3 is an ensemble expectation verified at `N = 500`, and every
   reported experiment uses `N` in the hundreds). One pair is adequate HERE for a different reason,
   which T4 measured: `Delta_f > W`. T4 supplied exact replacement label text.
8. **Kessels' `n_ext` result is weaker than "settled on 14".** Table 5.5, p. 173: the whole jump is
   `2 -> 6` (`1.019e-6 -> 2.726e-7`); from 6 to 34 the spread is a factor 1.36 and 14 wins by a
   hair. His stated reason for needing more states is MULTIPLE unmodelled modes (base frame plus
   flexible dynamics). Our residual has one dominant mode at 65 to 168 dB over floor with the next
   peaks at 4 to 19 dB, so his premise does not transfer.
9. **The burn-in citation now has a correct replacement.** Schiller, Heinrich, Lopez, Mueller,
   "Tuning the burn-in phase in training recurrent neural networks improves their performance",
   ICLR 2026, p. 4 Eq. (5) and p. 5 Eq. (7), is the real source for overlap-plus-discarded-warm-in,
   and it names Beintema et al. 2021 as a prior informal user. So the project's burn-in is
   legitimately "SUBNET practice, first analysed by Schiller et al. 2026" rather than uncited. But
   its own Theorem 2 (p. 8) bounds the benefit by `E_1((S-1) lambda^{2 o_min} + S lambda^m)/(T-m)`
   under `lambda < 1`, and **our controller has `lambda = 1` exactly** (three poles at `z = 1`,
   measured), at which the bound collapses to something independent of both `m` and `o_min`. The
   theorem is vacuous for the controller state and predicts no benefit from any warm-in length,
   which is what T6 measured independently (4.177 at `m = 0` against 4.674 at `m = 1600`).
10. **`docs/references.md` line 271 mis-parses a report number as an author-year.**
    `forssell1998_cl_revisited_liu2021.pdf` is Linkoping technical report **LiTH-ISY-R-2021**,
    Forssell and Ljung, 1998-04-01, 55 pp., the preprint of Automatica 35(7):1215-1241. The `2021`
    is the report number, not a year and not an author. Its text layer is OCR-damaged and no
    verbatim quote should be taken from it.
11. **There is no initial-condition or controller-state prescription in the classical closed-loop
    literature at all.** T6 grepped five papers for `nitia`, `transient` and `steady state` with
    character counts to prove the extraction worked: Forssell and Ljung 1998 (124,034 chars, 0
    hits), Van den Hof and Schrama 1993 (21,580, 0), Van den Hof and Schrama 1995 survey (92,237,
    0), Landau and Karimi 2001 ECC (26,357, 0), Landau and Karimi 2002 EJC (38,422, 0). The reason
    is structural: these papers work in transfer functions with asymptotic-in-`N` results, so the
    controller has no state and the question is never posed. Hansen, Franklin and Kosut 1989 remains
    unreachable and unread.
12. **The initial-condition consistency condition Sugie omits does exist in print.** Boroujeni et
    al., "Neural Identification of Feedback-Stabilized Nonlinear Systems", Appendix p. 8, proof of
    T.2, Eq. (9): for a model interconnected with a copy of the controller, equivalence to the
    intended model requires the initial condition to be matched THROUGH the controller copy. Sugie's
    transfer-function proof cannot state it because it has no states. Their paper does not window,
    so it does not say what to do at a window boundary, but it establishes that the condition is
    real.

## The twelve open questions of `augmentation-training-status.md` section 7, updated in place

| # | question | status after tonight |
|-|-|-|
| 1 | Do the augmented states help at all? | **ANSWERED, and the answer depends on capacity.** At `nx_aug = 2` they are decoration: arm 1's ablation costs `1.0183x`. At `nx_aug = 8`, at the same 520 updates, they are load-bearing: arm 2's ablation costs **`5.2081x`**, which is 87 % of the `6.010x` a planted model shows, and the free run reaches `3.795974e-07`. So the honest answer is **yes, but only once the fixed pole basis spans the mode**. Arm 1 alone would have said no.
| 2 | Is `na_nb = 17` wrong? | **Answered, and the answer is the opposite of the suspicion.** 17 is the ONLY value at which D-072 holds bit-identically (C1). T2 derived `na_nb = 103` from the OLS slope-variance law and cross-checked it to under 1.5 %, but C1 **cancels** that arm. The lag is not free to move in this pipeline, and the reason is float32 conditioning of `W^b = A^n O_n^{-1}`, not the rule that chose 17 |
| 3 | What does the `xc = 0` reset cost in ACCURACY, not just noise rejection? | **Deferred, with the experiment now designed rather than merely named.** T6 specifies the matched pair, the ordered non-shuffled sampler both arms must share, matched update counts, and the metrics, and then recommends NOT running it, because validation already carries `xc` continuously so the metric cannot reward matching training to it, and because 97 % of the reset damage is below 1 Hz while 80 % of the error is at 140 to 175 Hz. It proposes a gradient-cosine check first: cosine `> 0.99` guarantees the pair is null |
| 4 | Does D-072 survive `na_nb != 17`? | **Answered: NO.** `0.000e+00` at 17, `1.336e-04` at 32, `2.775e-04` at 64, `4.028e-04` at 103, monotone. And `(17, 8)` passes at `0.000e+00`, so the dependence is on `na_nb`, not on `nx_aug` (C1) |
| 5 | Is the objective's sensitivity weighting the real ceiling? | **C6 decides.** T3 proved analytically that NO weighting of the residual can flip `sign(dL/d(nu_log))`, because the initial-condition term is positive under every non-negative weight, and that a prefilter cancels exactly for a narrowband mode. So even if the objective is the wall, a weighting is not the lever |
| 6 | Is `nx_aug = 2` enough? | **ANSWERED: NO, and this is the night's largest result.** At matched updates `nx_aug` 2 to 8 moves the free run `1.379891e-06 -> 3.795974e-07` (3.63x) and the ablation `1.0183x -> 5.2081x`. T4 predicted a near-null arm and both of its pre-registered falsifiers fired (RMS below `1.25e-06`, ablation above `5 %`), by factors of 3.3 and 84. The mechanism is NOT pole adaptation: the poles moved under `0.15 Hz` in both arms. It is the SPAN of a frozen basis. Confound not separated: `nx_aug = 8` also widens the ANN (8 to 14 output rows, `nz` 11 to 17), so a width-matched control is still owed.
| 7 | Is the encoder lag under-set for VARIANCE (Beintema section 3.2)? | **Derived and cross-checked, then cancelled by D-072.** T2 reproduced the measured `n^-3/2` law to under 1.5 % on the two `K = 0` axes and recommended 103. It also predicted, before any run, that this changes nothing on noiseless simulation data because the argument is a variance argument and `cfg.snr = None`. It becomes mandatory the moment real Telica data enter |
| 8 | ~~Is the reference channel empty?~~ | Already answered in section 1.1; untouched tonight |
| 9 | ~~Does `cl_band_split.py` contradict P7?~~ | Already answered in section 1.1; untouched tonight |
| 10 | How much in-band energy is left at the PLATEAU, not untrained? | **Answered (C4).** Band share of error power falls `0.826 -> 0.614`; the in-band component falls `1.9891e-06 -> 1.0866e-06`, so the static ANN already took 70 % of the in-band power, which nothing had measured. `1.0866e-06` remains to be claimed, and claiming all of it would give `8.616e-07` |
| 11 | Do the state rows need per-row LOSS weighting (nine decades)? | **Deferred, and now with a derived candidate and a prediction against it.** T3 derived `W = diag(0.28634263, 0.28639728, 1.68404)` from `ystd`, which aligns the objective with the reported metre-domain metric, and measured the misalignment at only `1.72x` on the channel carrying the mode. Predicted not to move a plateau. Blocked from implementation tonight because the consumer is in `model_augmentation/` |
| 12 | Does any of this survive real noise? | **The gate is fixed but not yet exercised.** `CL_NOISE_CONSISTENT=1` is implemented in `cl_train.py` (C5), perturbing `y -> y+v` and `u -> u - C_fb(v)` together, which is the physical scenario; the existing `CL_NOISE_SIGMA` path is untouched and still the default. D-150's "survives Telica-level noise" rests on the non-physical `y`-only gate and should be re-read accordingly until a run is made through the new one |

## The fix: what was implemented, and what is blocked by scope

### Implemented tonight, env-gated, OFF by default

**`AUG_LRU_NA_NB=<int>`** in `scripts/gantry/gantry_dynamic/model.py`, `get_encoder_dims`, with the
active-state line printed by `build_model`. An explicit `cfg.na_nb_override` still wins, so every
existing sweep and every `dataclasses.replace(..., na_nb_override=...)` call site is untouched.

**Why C1 makes this a prerequisite rather than a convenience.** Jan's rule ties the encoder lag to
the augmented-state count, so `nx_aug` 2, 8, 14 forces `na_nb` 17, 29, 41 and a capacity arm sweeps
two variables at once. C1 then showed the lag is not free to move: D-072 holds bit-identically
**only** at `na_nb = 17`. So an `nx_aug > 2` arm under Jan's rule starts from a model that is not
the baseline, and is uninterpretable twice over. Pinning the lag is what makes a capacity arm legal.

**And C1 shows the pin works.** Cell `(na_nb = 17, nx_aug = 8)` reproduces
`2.186601103417735e-06` with rel dev `0.000e+00`. **Baseline equality depends on `na_nb`, not on
`nx_aug`.** That is a clean separation of section 7 row 2 (the lag question) from P6 (the capacity
question), which the handoff expected to remain confounded on every leaf except B1.

**D-072 line for the gate, as the handoff requires.** With `AUG_LRU_NA_NB` UNSET, on the edited
`model.py`, at `AUG_LRU=1 AUG_LRU_B=0.377`, `na_nb = 17`, `nx_aug = 2`:

```
17       2        2.186601103417735e-06      0.000e+00      PASS
1 of 1 cells hold D-072 bit-identically
```

and no `[aug-lag]` line is printed, confirming the gate is inactive by default. Artefact
`runs/d072_noop_check.json`. The change is a certified no-op when OFF.

**A bug I introduced and caught.** The first version of this edit deleted the
`ic = Interconnect(nxd, nu, ny, debugging=False)` line, so every build raised
`NameError: name 'ic' is not defined`. It was caught within minutes by the gate's own verification
run, which is the reason that run existed. `py_compile` passed throughout, because the fault is a
runtime NameError and not a syntax error; a compile check is not a substitute for executing the
build. The in-flight jobs (C1, C6, arm 1) were launched before the edit and had already imported the
module, so none of them was affected. Line restored and re-verified.

### Derived tonight but NOT implementable within this session's scope

Both are written out precisely enough to be applied in one step, and neither was applied, because
the handoff puts the files out of scope and "don't touch" is absolute.

| fix | where it must go | why it is blocked |
|-|-|-|
| T3's per-row objective weighting `W = diag(0.28634263, 0.28639728, 1.68403815)` on the normalised residual, gate `CL_OBJ_ROWW` | producer in `cl_train.py` (in scope), **consumer at `model_augmentation/fit_systems/interconnect.py:573`** | `model_augmentation/` is out of scope for this session. Note T3 also predicts this is a no-op on the sign and worth only `1.72x` of objective-versus-metric alignment, so it is a hygiene change, not the fix |
| The orthogonality extension: add non-zero `x_aug` points to `Z_pts` so the penalty can see the dynamic augmentation | **`scripts/gantry/gantry_dynamic/orth_penalty.py:152-156`** | explicitly out of scope ("the block is cleared, but they are still not part of this task"). This is the change the night's biggest finding calls for, and it is the first thing the next session should do |

## Incident: the concurrent validation path returns a STALE value

Found by accident and worth a decision entry, because D-146 records this path as verified fixed and
because it silently invalidates any run that uses it.

`cl_train.py:130` reads `CONCURRENT = bool(int(os.environ.get('CL_CONCURRENT', 0 if PROBE else 1)))`.
So **setting `CL_PROBE=0`, which looks like a pure diagnostics switch, also turns concurrent
subprocess validation ON.** I set it for speed after the host started killing jobs, and got:

| updates | concurrent path | serial path (`CL_CONCURRENT=0`) |
|-|-|-|
| 0 | `2.186601103417735e-06` | `2.1866011034e-06` |
| 260 | **`2.186601103417735e-06`**, bit-identical to untrained at 16 digits | **`1.383192160424035e-06`** |
| 520 | not reached | `1.379891240402659e-06` |

Same configuration, same seed, same everything else. A genuine 260-update change moves the value
somewhere in sixteen digits; exact equality means the trained parameters are not reaching the scored
model. **The concurrent child is scoring a stale model.** This is the same class of failure D-146
was written about (selection optimising one objective and scoring another), and the guards described
in `cl_train.py`'s docstring did not catch it here.

Consequence for tonight: nothing. Every gate result came from probes that compute the free run
directly and in-process, never through deepSI's validation path, and the arm-1 numbers reported here
are all from the serial path. Consequence for the next session: **do not run an arm with `CL_PROBE=0`
unless `CL_CONCURRENT=0` is also set**, and re-open D-146.

## Deferred (written out, not started, per the handoff's out-of-scope list)

| deferral | the experiment, stated so it can be run later |
|-|-|
| the `xc = 0` **accuracy** cost (section 4.2 / section 7 row 3) | `alpha_cancellation.json` measured the NOISE cost of resetting the controller state every 400 samples: an exact cancellation degrades to a saturating `2.0x`. The accuracy cost on noiseless data is a different question and is unmeasured. It needs two training arms that differ in nothing but the reset: (a) windowed `nf = 400` with `xc = 0`, the current configuration, and (b) whole-record training with the controller state carried, at matched update counts, both scored on free-run validation RMS. T6 is deriving what the literature prescribes for the controller state, and its answer should choose the second arm's exact form (encoder for `xc`, overlap warm start, or whole-record) before the pair is run |
| per-row **loss** weighting for the nine-decade state rows (section 4.5 / section 7 row 11) | The eight ANN correction rows span nine decades in normalised units, and regressing this architecture onto the exact target fits the absorber rows to `1-R^2 = 9.6e-05` with a per-row scale and fails completely (`0.98`) without one. The per-row ReZero gate (`ANN_REZERO_GATE=row`) already applies a per-row scale to the OUTPUT. Whether the LOSS needs one is untested. The experiment is a matched pair at fixed update count: `ANN_REZERO_GATE=row` alone against `ANN_REZERO_GATE=row` plus a per-row weighting of the state-space defect term, scored on free-run RMS and on the ablation. Note the loss here is on `y`, not on state rows, so this only becomes well-posed once a state-space term exists in the objective, which is itself a design decision that has not been taken |

## Recommended next action

**Run the width-matched control for arm 2, and nothing else, before anything is written up.**

Two runs, both at 520 updates with the ablation, both exactly arm 2's configuration except for the
one thing under test.

**Run 1, the width-matched static control. This is the one that matters.**
`nx_aug = 2`, `na_nb = 17`, and `n_nodes_per_layer` raised **16 to 20**. Arm 2's ANN is a third
larger than arm 1's purely because `nx_aug` widens both ends of it:

| | `nx_aug` | `nz` | `nw` | ANN params |
|-|-|-|-|-|
| arm 1 | 2 | 11 | 8 | **600** |
| arm 2 | 8 | 17 | 14 | **798** |
| control | 2 | 11 | 8 | **828** at `n_nodes_per_layer = 20` |

So the control has slightly MORE capacity than arm 2 and only two augmented states. If it reaches
about `3.8e-07`, the gain was ANN size and the capacity story is wrong. If it stays near arm 1's
`1.38e-06`, the gain is genuinely the augmented states. Launch:
`CL_NX_AUG` unset, `AUG_LRU=1 AUG_LRU_B=0.377 ENC_WA_ZERO=1 CL_CONCURRENT=0`, and
`n_nodes_per_layer = 20` via `dataclasses.replace` (it is a `RunConfig` field, so it needs either a
new env gate in `cl_train.py` or a one-line edit there; `config.py` is out of scope).

**Run 2, the frozen-pole variant.** `nx_aug = 8` exactly as arm 2 but with `nu_log` and
`theta_log` set `requires_grad = False`. The pole tables say the poles moved under `0.15 Hz` anyway,
so this should change nothing; if it does change something, the pole tables were read wrong and the
fixed-basis synthesis is wrong with them.

Why this one and not the more exciting options. Arm 2 is the night's biggest result and its
headline reading, that capacity was the binding constraint, rests on a comparison that moved three
things at once: the augmented-state count, the ANN's output width (8 to 14 rows) and its input
width (`nz` 11 to 17). The lag is controlled, which is what the `AUG_LRU_NA_NB` pin bought, but ANN
width is not. Until that control exists, "capacity was the constraint" and "a wider ANN was the
constraint" both fit `1.379891e-06 -> 3.795974e-07`, and only the first is interesting. The frozen
-pole variant tests the span mechanism directly: if freezing the poles changes nothing, the
augmented block really is a fixed basis and the whole synthesis holds; if it collapses, the poles
were adapting after all and the pole tables were read wrong.

It is also the cheapest decisive thing available: two runs of about 40 minutes plus two ablations
of about 25, all inside the window the host's kill cycle allows, and it needs no new code beyond
setting `requires_grad = False` on two tensors.

Everything else should wait behind it, including the two changes I could not apply tonight (the
orthogonality extension at `orth_penalty.py:152-156`, and T3's per-row weighting at
`interconnect.py:573`). The orthogonality hole is the more important finding for the thesis, but it
is a claim about what the regulariser fails to constrain, and it will be argued far better once it
is known whether the dynamic augmentation can be made load-bearing at all. Arm 2 says it can. The
control says whether that is for the reason claimed.
