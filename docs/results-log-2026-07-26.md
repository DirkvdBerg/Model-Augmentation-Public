# Results log, 2026-07-26

**Measurements only.** Each entry states the question, the method, the numbers, the
artifact holding them, and a validity status. **No interpretation** — it is deliberately
kept out of here so these numbers can be re-read against a different reading later.

**Interpretation lives elsewhere, and most of it has since been superseded.** For status,
read `docs/diagnostic-overview.md` (authoritative, with grades and artifact paths) and
`docs/dc-accumulation-research-brief-2026-07-26.md` (the current framing). **Do NOT use the
pointer this file originally carried:** `narrowband-objective-problem-2026-07-26.md` §5 and
`flat-direction-problem-2026-07-26.md` §2-3 are **VOID framings**, superseded the same day
they were written. The numbers below are unaffected; only the readings attached to them
were wrong.

Void entries are kept, not deleted: knowing which measurements were attempted and failed
is what stops them being re-run.

**Grades** (project convention, `docs/drift-conclusions-2026-07-25.md` §0): ROBUST =
3 seeds, 2 protocols. SOLID = 3 seeds, 1 protocol. SINGLE = 1 seed or 1 record.
ORACLE = uses information unavailable on real data. VOID = the run's own control failed.

---

## 0. Common setup

Unless stated: MATLAB `augmentation` data (20 kHz, `d = 5`), 14 train records,
`up_sample = 1`, `seed = 42`, `nx_ann = 2`, routing `(0..7)`, `na = nb = 17`,
`nf_seg = 400`, `Ts = 2.5e-4 s`. Encoder frozen where stated. All free-run errors are
val record V1 unless noted.

`sim-RMS` = `apply_experiment(val).RMS(val)` = `mean((y-yhat)^2)^0.5` on **raw metres**,
over the horizon stated. **Always state the horizon**: the same ANN-off model measures
`7.86e-05` at 2 s and `1.66e-04` at 12 s.

---

## 1. Data generation (D-126)

| | |
|---|---|
| **Question** | How far apart are the shipped MATLAB records and the Python model that trains on them? |
| **Method** | Re-simulate all 22 records in Python from the verified 8-state truth (`drift_common.simulate_truth`) using the model's own RK4, `Ts` and block-mean input, float64. Compare to the shipped `y`. |
| **Result** | Pooled `\|y_mat - y_py\|` RMS over 22 records: **X `2.414e-05`, Theta `2.414e-05`, Y `5.198e-05` m**, against an absorber signal of `2.196e-05 m` RMS. Per-record means are the same order as the RMS, i.e. the difference is a systematic offset, not noise. Two records (E1, E4) sit at `3.2e-08` / `8.5e-08`. |
| **Sensitivity** | `up_sample` 1 / 2 / 4 give results identical to 4 significant digits, so this is the solver-and-rate difference (ode45 variable-step at 20 kHz, saved as `single`, vs fixed-step RK4 at 4 kHz on a block-mean input), not RK4 substep error. |
| **Artifact** | `scripts/gantry/pysynth-data/generate_pysynth_data.py`; output `data/gantry/matlab/trajectory/pysynth/` and `pysynth_baseline/`, 22 records each, integrity-checked (all finite, correct shapes, `dt = 2.5e-4`; `delta_a std = 2.196e-05` on the absorber arm, exactly `0` on the baseline arm) |
| **Grade** | SOLID (22 records) |

---

## 2. MS5 — the failure, read off the production checkpoints

| | |
|---|---|
| **Question** | What exactly happens to each metric during the production run? |
| **Method** | `torch.load` on the stored deepSI system saves. No training. Requires `scripts/gantry` on `sys.path` (they pickle `gantry_dynamic` as a top-level module). |
| **Result, `gantry_drift_71167_last.pth` (20 epochs, 5200 batches)** | val sim-RMS **12 s**: `1.661e-04` (ep0) -> `2.109e-02` (ep1, **127x**) -> `1.951e-02` (ep20). `argmin = epoch 0`. Over the same run: val nf-window RMS **0.1 s** `4.386e-05 -> 3.771e-05` (**-14%**); train nf-window `3.807e-05 -> 3.324e-05` (-13%); train loss `1.369e-06 -> 1.103e-06` (-19%). |
| **Result, `gantry_drift_last.pth` (5 epochs, 130 batches)** | `8.060e-05` (ep0) then `1.83x / 4.69x / 6.06x / 7.16x / 9.47x`. |
| **Artifact** | the two `.pth` files; both carry full per-epoch `Loss_val` / `Loss_train` / `Loss_val_nf` / `Loss_train_nf` |
| **Grade** | SINGLE (1 run each) |
| **Note** | `gantry_ckpt_71167.pt` is the **best** checkpoint; since best = epoch 0, that file IS the initialisation and cannot serve as a degraded model. |

---

## 3. MS2 — encoder initialisation: ramp or settle?

| | |
|---|---|
| **Question** | Does the encoder initial condition drift, or settle to a bounded offset? |
| **Method** | ANN off, free run from true-x0 vs encoder-x0, on `pysynth_baseline` (model == data exactly) and `pysynth`. Verdict read from the SHAPE: late slope over the last half, 2 s vs 12 s. `tau_X = 1.546 s`, so 12 s = 7.8 tau. |
| **Result, `pysynth_baseline`** | encoder-x0 mean late slope **`5.39e-05 m/s` at 2 s -> `5.88e-07 m/s` at 12 s (92x collapse)** while the level grows only 1.7x (`1.46e-04 -> 2.50e-04 m`, X). True-x0 control `1.07e-06 m`. |
| **Result, `pysynth`** | pooled X RMS true-x0 `1.246e-04` vs encoder-x0 `2.308e-04`; late slopes `1.60e-05` vs `2.11e-05 m/s`, ~30x the absorber-free arm on **both** arms. |
| **Artifact** | `measure_encoder_drift.py`, `results/MS2_encoder_drift.json` |
| **Grade** | SINGLE (3 records, 1 seed) |
| **Caution** | At 2 s the same data reads as "growing" (`last/first = 2.08`) because the exponential is only 73% settled. Any such number is meaningless without its horizon. |

---

## 4. MS1 — where the segment-boundary defect comes from

| | |
|---|---|
| **Question** | Split the boundary defect `d = x_node - x_rollout` into encoder error `e_enc` and model error `e_roll`, using exact true states. |
| **Method** | `pysynth`, untrained model, 6 records, stride 100, 15 windows/record, `n_seg = 5 x 400`. |
| **Result, per row (`e_enc / e_roll`)** | X `0.01`, Theta `0.16`, Y `0.02`, dX `0.90`, dTheta `1.34`, dY `1.18`. Pooled `1.33`, but dominated by dTheta, which is ~100x every other row (`e_enc = 1.220e-01`, `e_roll = 9.133e-02`). |
| **Result, `e_roll` per row** | X `8.501e-04`, Theta `6.929e-03`, Y `6.088e-04`, dX `1.298e-03`, dTheta `9.133e-02`, dY `5.075e-03` (normalised state units) |
| **Result, operating point** | `corr(\|mean Y\|, \|e_enc\|) = **+0.995**`, slope `0.249 / m`. `\|e_enc\|` runs `3.28e-03` at `Y = 0` to `8.08e-02` at `Y = -0.300` and `7.62e-02` at `Y = +0.300` — **24x**, symmetric, monotone in `\|Y\|`. |
| **Result, cancellation** | On the two near-`Y = 0` records, `\|d\|` is SMALLER than either component (`enc_share` 1.76 and 1.46). |
| **Artifact** | `measure_defect_split.py`, `results/MS1_defect_split_encoder_nodes.json` |
| **Grade** | SINGLE (6 records, 1 seed) |
| **Context** | `gantry_linearize_and_discretize` raises `NotImplementedError` for `Y_op != 0`, so the encoder map is LTI at `Y = 0` by construction. |

---

## 5. Defect term — implementation verification

| test | result | status |
|---|---|---|
| no-op contract (`n_seg=1`, `w=0`) vs parent | bitwise identical, `2.22840785824018894e-08` | PASS |
| guard (`n_seg>1`, `w=0`) | bitwise identical | PASS |
| segment equivalence (segmented fit term == mean of N single-window losses) | rel `3.62e-08` vs float32 tol `1.2e-05` | PASS |
| zero defect on an exact model (true nodes, `model == data`) | max `8.64e-07` in training-std units; augmented rows exactly `0` | PASS |
| Ribeiro Thm 2 (re-anchoring is a no-op vs one long roll) | rel `1.19e-06` | PASS |
| defect linear in an injected constant force | exponent **`1.000`** (both position and velocity rows) | PASS |
| position defect grows as `nf_seg^2` | exponent **`1.993`** over `nf_seg` 100/200/400 | PASS |
| minimisable by gradient descent | `8.70e-03 -> 5.93e-03` (0.68x), 30 steps at `lr = 1e-6` | PASS |

Artifacts: `verify_defect_term.py` (7/7), `verify_ms_method.py` (5/5). Grade SOLID
(analytic references, not seed-dependent).

**Two test-design errors found and fixed:** dividing the defect by a standstill record's
own state std measured that record's excitation rather than the code; and running the
minimisability test at `lr = 1e-3` blew the defect up 150x on the FIRST Adam step
(Adam's first step is ~1.0 x lr per coordinate regardless of gradient).

---

## 6. MS6 / MS6b — does the defect see the failure?

| | |
|---|---|
| **Question** | Does a model 65x worse on the free run produce a larger defect? |
| **Method** | INIT vs `gantry_drift_71167_last.pth`, identical windows, encoder nodes (real training condition) and true nodes (pure model error). MS6b sweeps `n_seg`. |
| **Reference** | free-run 12 s: `1.196e-04 -> 7.781e-03` = **65.04x** |
| **Result** | defect RMS ratio **1.01x** (encoder nodes, `2.7026e-02 -> 2.7414e-02`); **1.78x** (true nodes, `4.892e-03 -> 8.719e-03`) |
| **Result, coverage sweep** | `n_seg` 4 / 12 / 30 = 0.4 / 1.2 / 3.0 s coupled: **`1.08x / 1.02x / 1.04x`**. Flat. (`n_seg = 120` skipped: record too short.) |
| **Artifact** | `check_defect_sees_failure.py`, `results/MS6_defect_sees_failure.json`, `MS6b_nseg_sweep.json` |
| **Grade** | SINGLE (1 record, 1 seed) |

---

## 7. MS7 — coherent vs broadband aggregation of the same defects

| `n_seg` | RMS ratio | `\|MEAN\|` ratio | coherence, healthy | coherence, degraded |
|---|---|---|---|---|
| 4 | 1.57x | 2.34x | 0.576 | 0.858 |
| 12 | 1.69x | 4.34x | 0.323 | 0.831 |
| 30 | 1.79x | **13.80x** | 0.108 | 0.834 |

True nodes. Coherence = `|mean d| / rms d`. The healthy model's coherence tracks
`1/sqrt(n)` (predicted `0.500 / 0.289 / 0.183`); the degraded model's is flat at `~0.83`.

**With encoder nodes the `|MEAN|` ratio reaches only `1.06 / 1.21 / 1.50x`.**

Absolute magnitudes at `n_seg = 30`, true nodes: `|mean|` degraded `6.137e-03`, init
`4.446e-04`. The **sum** is `n` times these (`1.78e-01` and `1.29e-02`); ratios are
identical because `n` cancels.

Artifact: `check_coherent_defect.py`, `results/MS7_coherent_defect.json`.
Grade: SINGLE + **ORACLE** for the true-node column (true states do not exist on real
data; the deployable figure is the encoder-node `1.50x`).

---

## 8. VOID results — attempted, not usable

| id | what was attempted | why it is void |
|---|---|---|
| **MS3** | 3-arm defect on/off A/B on `pysynth` | The **control never failed**. Arm A passed both gates on `pysynth` (G1 1.000, G2 0.595) AND on the original MATLAB data (G1 1.000, G2 0.857). Cause: sizing the run to a foreground budget cut the optimiser count to **~21 Adam steps** against production's ~5250. A 10x-steps rerun (~296 steps) showed the model had not moved at all — val sim-RMS, val nf-window and train nf-window all bit-identical `first = best = last`, `0` non-finite. |
| **MS4** | resume from a degraded checkpoint | Loaded `gantry_ckpt_71167.pt`, which is the **best** checkpoint and therefore epoch 0, i.e. the initialisation. An earlier reading of `9.02e-01` (8.02x) as "the failure reproduced" was a normalisation-frame mismatch from a 4-train-record trim, not a degraded model. |
| **MS8** | gradient alignment vs `theta_deg - theta_init` | Cosines `+0.0001` / `+0.0057` / `+0.0054` — **all below chance** (`1/sqrt(600) = 0.041`), including the control. `theta_deg - theta_init` is an Adam-accumulated displacement, not a gradient. |
| **MS9** | gradient alignment vs the free-run gradient at `H = 4000` (1 s) | Pre-registered control required `cos(g_fit, g_free) <= ~0` (MS5 shows the windowed loss makes the free run 127x worse); measured **`+0.4588`**. A DC error is 144x weaker at 1 s than at 12 s, so the reference sat inside the same blind spot as the thing being tested. |

---

## 9. MS10 — finite-difference probe (IN PROGRESS)

| | |
|---|---|
| **Question** | Does one normalised step along each candidate's gradient reduce the ACTUAL 12 s free-run error? |
| **Method** | `theta' = theta - alpha * g/\|\|g\|\|`, then forward-only free run at the full 48000 steps. Encoder frozen. Controls: C1 reproducibility (`E(theta)` twice, must be bit-identical) and C2 a random unit direction at the same `alpha`. |
| **C1** | PASSED — `E` reproduces exactly |
| **Partial result** | baseline `E0 = 6.938921181e-06`. `fit` @ `alpha=1e-5`: `+4.1170e-01` (**+59331x**); `fit` @ `1e-4`: `+3.3492e+01`; `rms` @ `1e-5`: `+6.9170e-02` (+9968x); `rms` @ `1e-4`: `+6.5057e+00`; `coh` @ `1e-5`: `+5.2791e-02` (+7608x). All HURT. |
| **Status** | **INCOMPLETE — `coh` @ `1e-4` and the RANDOM control (C2) have not returned. No ranking may be read until C2 lands: if a random direction also costs ~`+5e4`x, then every direction hurts and the ordering carries no information.** |
| **Scale note** | Adam at `lr = 1e-7` over 600 parameters moves ~`2.4e-6` in L2, so `alpha = 1e-5` is about four Adam steps. |
| **Artifact** | `probe_step_vs_freerun.py`, `results/MS10_probe_init.json` |

---

## 9b. MS11 — Orrell two-run discriminator (constant force vs diverging mode)

| | |
|---|---|
| **Question** | Is the degraded model's error accumulated tendency (force) error, or a displacement error amplified by a diverging mode? |
| **Method** | Orrell et al. 2001 Eq. (5), `e(tau) = M(tau) e(0) + d(tau)`. **Test B**: two runs from `x0` and `x0 + alpha*delta` with identical inputs; `d(tau)` cancels exactly, so the difference isolates the propagator. **Test A**: free run from the recorded true state (`e(0)` printed as `0.000e+00`), fit `log\|e\|` vs `log t`. Arms: INIT (zero-output ANN = no ANN dynamics, the control) and `gantry_drift_71167_last.pth`. 48000 steps, alphas `1e-6` and `1e-5`. |
| **Test B result** | Polynomial in all four cases: R^2_poly `0.874-0.961` vs R^2_exp `0.476-0.693`. Exponent SUB-LINEAR — INIT `+0.342 / +0.336`, DEGRADED `+0.422 / +0.415`. Growth INIT `4.85x / 4.32x`, DEGRADED `6.28x / 5.92x`. **Exponent is alpha-invariant, so the linear-regime guard passes.** |
| **Test A result** | INIT X final `9.944e-05`, poly-exp `+0.479` (R^2 0.809). DEGRADED X final **`2.192e-01`**, poly-exp **`+1.484` (R^2 0.997)**; DEGRADED Y `1.295e-01`, poly-exp `+1.677` (R^2 0.931). |
| **Artifact** | `orrell_two_run.py`, `results/MS11_orrell_two_run.json` |
| **Grade** | SINGLE (1 record, 1 seed) |
| **Script defect** | The auto-verdict line printed "mixed / inconclusive" because it used an absolute threshold (`growth < 2x`) rather than comparing arms. Read the R^2 columns and the INIT control instead. Numbers unaffected. |
| **Caveat** | Test A's reference is MATLAB's recorded truth, not the Python model's own state, so `e(0) = 0` holds against the record rather than against a self-consistent trajectory. |

---

## 9c. MS12 — is the ANN's DC output sufficient and necessary for the failure?

| | |
|---|---|
| **Question** | MS11 gave the shape (accumulated force error). Does the trained ANN's per-row MEAN account for it, and how big is it? |
| **Method** | Four 12 s free runs on `gantry_drift_71167_last.pth`: FULL, MEAN-ONLY (ANN replaced by its own per-row mean along the free-run trajectory), MEAN-REMOVED, ANN-OFF. |
| **Result** | ANN-OFF `1.311e-04` (floor); FULL `7.719e-03` (58.9x floor); **MEAN-ONLY `8.708e-03` = 112.8% of FULL**; MEAN-REMOVED `2.871e-03` = 37.2% of FULL (still 22x floor) |
| **Per row** | dX `mean = +3.46e-07`, purity 0.278, **`+1.67e-02 N`**; dY `+1.21e-06`, 0.233, **`+2.33e-02 N`**; dTheta `-1.44e-04`, 0.569; X `+3.14e-07`, 0.485; Y `-5.38e-07`, 0.708; **aug0 `-8.54e-05`, purity 0.956; aug1 `+9.34e-05`, purity 0.886** |
| **Artifact** | `measure_ann_dc_force.py`, `results/MS12_ann_dc_force.json` |
| **Grade** | SINGLE (1 record, 1 seed) |
| **Note** | MEAN-ONLY exceeding FULL means the fluctuating part partially cancels the DC (same signature as C6 and MS1). |

---

## 9d. Constraint on the validation strategy (follows from MS12 + C7 + dA)

Not a measurement; a consequence of three that already exist, recorded here because it
gates what any future test can prove.

* The failure is a spurious ANN DC of `17-23 mN` (MS12).
* A zero-mean penalty targets exactly that, and is RULED OUT on R2: the real Telica
  residual mean is `-157.5 N` (X) / `-83.7 N` (Y) at 315-344 sigma (C7), so suppressing DC
  forbids learning real friction.
* On THIS simulation the true residual is the absorber, which is zero-mean
  (`dA`: dominant coupling `|mean|/rms ~ 1e-4`).

**Therefore the correct DC on the current testbed is genuinely zero, and a zero-mean
constraint would look successful here while being exactly wrong on real data. The sim
cannot distinguish "suppress spurious DC" from "suppress all DC", so it cannot validate
ANY method in this family.** The injected-friction sim (`datasilent-friction-sim/`, built
to step 2; steps 3a/3b never built) is a prerequisite for testing anything here.

**Speculative, NOT yet an argument:** `17-23 mN` spurious against `157.5 N` legitimate is
~4 orders apart and would permit a scale-separated rule with a threshold derived from the
windowed loss's own resolution rather than hand-picked. But those two numbers come from
DIFFERENT SYSTEMS (sim vs real Telica), so the ratio is not a valid comparison.

---

## 10. Config traps that invalidated runs today

Each of these silently produced wrong numbers before being caught:

1. **`RunConfig` defaults `up_sample = 2`**; the entry file and every checkpoint use `1`.
   Numerically small here (0.3%) but wrong.
2. **Trimming `TRAIN_FILES` changes `compute_normalization`**, which changes the encoder
   built from `norm.x_all` (D-119). A 4-record trim moved epoch-0 from `1.66e-04` to
   `1.13e-01`. Validation trims are safe; training trims are not.
3. **Filtering non-finite values** out of a metric series makes divergence
   indistinguishable from a flat pass.
4. **`gantry_ckpt_71167.pt` is the best checkpoint**, which for this failure is epoch 0.
5. **The `.pth` files carry their own `norm`** — take weights and `norm` together; mixing
   a checkpoint's weights with locally computed normalisation produces a spurious result.
6. **Piping a long run through `grep`** block-buffers stdout, so the log stays empty until
   exit; and background jobs are killed unpredictably (`--start`-style resumability or
   foreground chunks are required).

---

## 11. Step 3 — ARTBP gate-2 benchmark, recomputed from the stored `.npz`

Appended 2026-07-26 night. Numbers only.

| | |
|---|---|
| **Question** | Do the five (in fact six) converged ARTBP runs of C-6 say what C-6 transcribes? |
| **Method** | Read-only pass over `scripts/gantry/ARTBP/data/*.npz`, written 2026-07-23 by `train_artbp.py`. No training, no model load. `dc-accumulation/step3_recompute_artbp.py`. |
| **Grade** | SINGLE (one seed per arm) |

**Horizons.** `val sim-RMS` = V1-V4 pooled free run, **12 s**. `nf-RMS` = windowed, **0.1 s**.
`drift ratio` = full-ANN / ANN-off RMS over the **last quarter of an 8000-step (2 s) run**,
i.e. a **0.5 s tail at `t` in `[1.5, 2.0] s`**, records T1/T3/T5 (`train_artbp.py:60,275`).
`DC` = `ann(Z_pts)[...,0].mean(0)[5]`, the **dY row only**, averaged over the orthogonality
probe set `Z_pts`, not along any trajectory (`train_artbp.py:226`); `dc_endpoint` is its mean
over the last 50 optimiser steps (`train_artbp.py:300`).

| arm | ep-1 12 s | best 12 s | best trained ep | last 12 s | Y drift, 0.5 s tail | held-out 0.1 s nf-RMS | `dc_endpoint` | `dcgrad_var` |
|---|---|---|---|---|---|---|---|---|
| `fixed h1600 b256 ep20` | `2.9047e-02` | `2.1920e-02` | 19 | `2.2193e-02` | `83.1x` | `5.1971e-05` | `-3.4616e-06` | `2.3982e-07` |
| `geom h1600 b256 ep20` | `2.7523e-03` | `1.7588e-03` | 5 | `5.8958e-03` | `22.1x` | `5.3255e-05` | `-1.4113e-06` | `8.4016e+00` |
| `poly6 h1600 b256 ep20` | `7.6065e-03` | `2.1032e-03` | 7 | `3.9420e-03` | `13.5x` | `5.3091e-05` | `-1.1086e-06` | `6.5986e+00` |
| `poly6 h1600 b256 ep8` | `7.3888e-03` | `1.9716e-03` | 7 | `3.5542e-03` | `12.7x` | `5.9475e-05` | `-6.6055e-07` | `4.5753e-01` |
| `poly6 h3200 b256 ep10` | `6.9918e-03` | `2.9319e-03` | 6 | `8.1952e-03` | `47.9x` | `6.9175e-05` | `+1.4791e-07` | `1.5357e+03` |
| `poly6 h6400 b128 ep20` | `1.6816e-02` | `1.1807e-03` | 14 | `6.4226e-02` | `252.7x` | `1.3246e-04` | `+4.8628e-06` | `6.5867e+07` |

All six arms: seed 0, `nf = 400`, `lr = 1e-7`, `stride = 10`, routing `(0..7)`, real with-MSD
MATLAB `augmentation` data. Steps per epoch: 254 (`h1600`), 245 (`h3200`), 455 (`h6400`).

**A sixth arm exists.** `poly6 h1600 b256 ep8`, in `ARTBP/data/72659/`, appears in no
document. It carries **both** `_best.pt` and `_last.pt` checkpoints; the other five carry
none except `h3200` (whose single file is the **best** epoch, 6, not the endpoint).

**Four structural facts about how these runs store their numbers.**

1. `len(val_sim_traj)` is `epochs` for the three `h1600 ep20` arms and `epochs + 1` for the
   other three. The epoch-0 row (`train_artbp.py:199-200`) is **absent** on the first three.
2. Stored epoch-0 values: `1.846056547947228e-04` (`h3200`) and `1.846056547947228e-04`
   (`h6400`), bit-identical; `2.239589812234044e-04` (`ep8 h1600`). The last differs because
   that run stored `val_n = 2` against the others' 4 — a different val set, hence a different
   metric, not a disagreement. The three `h1600 ep20` arms store no `val_n` field at all.
3. `best_val_sim` is a per-epoch minimum; `drift_ratio` is computed after the training loop
   on the live final weights (`train_artbp.py:251-291`). Best trained epoch vs epoch the
   drift was measured on: `19/20`, `5/20`, `7/20`, `7/8`, `6/10`, `14/20`. **Different models
   in all six arms.**
4. `drift_off` (the ANN-off denominator) takes 2 distinct values across the 6 arms:
   T1 Y `6.2941e-04` (the three `h1600 ep20`) and `6.4723e-04` (the other three), 2.8% apart.

Per-record drift ratio spread, `ep8 h1600` (T1/T3/T5, Y channel): `1.584 / 32.980 / 3.509`.

Artifact: `scripts/gantry/dc-accumulation/results/step3_artbp_recompute.json`.

---

## 12. Step 1 — minimal-testbed port control

Appended 2026-07-26 night.

| | |
|---|---|
| **Question** | Does the C-15 testbed, copied into `dc-accumulation/`, reproduce its source? |
| **Method** | `dc-accumulation/step1_testbed.py --configs C0 --seeds 0`, `K = 0`, 1500 steps, 4 s (16000-step) free run. |
| **Grade** | SINGLE (control only) |

| quantity | measured | C-15 published |
|---|---|---|
| free-run harm ratio, ANN / ANN-off, **4 s** | `1.713x` | `1.713x` |
| windowed loss ratio, **0.1 s** | `0.943` | `0.943` |
| ANN output mean along the free run | `+1.1500e-05` | `1.15e-05` |

PASS at 0.0% deviation.

**A first port returned `4.871x` and was wrong.** `make_data` ends with `u = u - u.mean()`,
so lengthening the data array (to make room for encoder history) shifts the input by a
constant, and on a `K = 0` plant a constant input offset is not a small perturbation. The
array length is now pinned to C-15's `HFREE + NF + 10` for every arm and only the start
**index** varies.

Artifact: `scripts/gantry/dc-accumulation/results/step1_control_check.json`.

---

## 13. Step 0 (GATE) — DC sufficiency on the ARTBP checkpoints

Appended 2026-07-26 night. Numbers only.

| | |
|---|---|
| **Question** | Is the DC the failure, or only correlated with it? (C-3 sufficiency vs C-6's low-DC / high-drift arm) |
| **Method** | MS12's four-arm decomposition re-run on the ARTBP checkpoints. Rig = `drift-demo/demo_common.build_pipeline(dataclasses.replace(CFG, seed=0))`, i.e. the rig `train_artbp.py` uses. The checkpoints store `ann` + `encoder` only, no `norm`. `dc-accumulation/step0_dc_sufficiency.py`. NO TRAINING. |
| **Grade** | SINGLE (1 seed, 1 record) |

**Control.** Zero-ANN epoch-0 val sim-RMS, ARTBP val protocol:

| `VAL_N` | measured | stored | dev |
|---|---|---|---|
| 4 | `1.844927e-04` | `1.846056547947228e-04` | 0.064% |
| 2 | `2.242571e-04` | `2.239589812234044e-04` | 0.133% |

Both PASS. This also settles §11 point 2: the two stored epoch-0 values are two different
metrics (4 val records vs 2), not a disagreement.

**Free runs.** Record V1, `k0 = 17`, `H = 47982` (**12.00 s**). ANN-OFF floor
`1.196380e-04 m` (identical for all arms to 7 figures).

| model | FULL 12 s | xOFF | MEAN-ONLY | MO/F | MEAN-REMOVED | MR/F | MR/OFF |
|---|---|---|---|---|---|---|---|
| ZERO-ANN (epoch 0) | `1.196380e-04` | `1.00x` | `1.196380e-04` | 100.0% | `1.196380e-04` | 100.0% | `1.000x` |
| `poly6 h1600 ep8` BEST (ep 7) | `2.343532e-03` | `19.59x` | `2.320671e-03` | 99.0% | `1.512223e-04` | 6.5% | `1.264x` |
| `poly6 h1600 ep8` LAST (ep 8) | `4.190942e-03` | `35.03x` | `4.136484e-03` | 98.7% | `1.038413e-04` | 2.5% | **`0.868x`** |
| `poly6 h3200 ep10` BEST (ep 6) | `3.511288e-03` | `29.35x` | `3.492215e-03` | 99.5% | `1.231194e-04` | 3.5% | `1.029x` |

For reference, MS12 on `gantry_drift_71167_last.pth` (`H = 47979`, V1): FULL `7.719388e-03`
(`58.87x`), MEAN-ONLY `8.708423e-03` (112.8%), MEAN-REMOVED `2.871076e-03` (37.2%).

**DC composition.** `||traj mean||` = norm over all 8 routed rows of the per-row time-average
along the free run. `dY` is the single row ARTBP's `dc` and its `±3e-7` band watch.

| model | `\|\|traj mean\|\|` | `\|dY\|` traj | dY as % of norm | `\|\|Z_pts mean\|\|` | `\|dY\|` Z_pts | dY as % of norm | 3 largest rows |
|---|---|---|---|---|---|---|---|
| `h1600 ep8` BEST | `7.5914e-05` | `5.8217e-07` | 0.77% | `7.0475e-05` | `5.4719e-07` | 0.78% | `aug0 5.555e-05`, `aug1 4.900e-05`, `dTheta 1.658e-05` |
| `h1600 ep8` LAST | `7.9774e-05` | `7.5408e-07` | 0.95% | `7.3969e-05` | `7.0166e-07` | 0.95% | `aug0 5.730e-05`, `aug1 5.187e-05`, `dTheta 1.973e-05` |
| `h3200 ep10` BEST | `2.7691e-05` | `4.8831e-07` | 1.76% | `2.5732e-05` | `4.2269e-07` | 1.64% | `aug0 2.408e-05`, `aug1 1.292e-05`, `Theta 3.637e-06` |

71167 for comparison: `||traj mean|| = 1.917204e-04`, rows `dTheta -1.4365e-04`,
`aug1 +9.3351e-05`, `aug0 -8.5421e-05`, `dY +1.2096e-06` (0.63% of the norm).

**Operator ordering.** Both operators rank the three checkpoints identically
(`h3200 ep6` < `h1600 ep7` < `h1600 ep8`). The pre-declared ordering discriminator did
**not** fire.

**Drift ratio on this rig** (0.5 s tail of a 2 s run, V1, FULL/OFF): `1.60x` / `4.51x` for
`h1600 ep8` BEST and `h3200` BEST. Not comparable to C-6's `13.5x` / `47.9x`, which are
3-record standstill T1/T3/T5 means on the epoch-final models.

Artifact: `scripts/gantry/dc-accumulation/results/step0_dc_sufficiency.json`.
