# Reference Injection vs. Force Injection: Why `ref_injection` Fails Open-Loop Parameter Recovery

**Date**: 2026-05-04
**Status**: Diagnostic — explains observed failure; recommends dataset strategy

---

## 1. Context

`train_param_recovery.py` supports three dataset modes:

| `DATASET` | Trajectories | Multisine injection point |
|-----------|-------------|--------------------------|
| `base` | T1–T6 | None |
| `multisine` | T1–T8 | Force: `f_sim` added directly to plant input |
| `ref_injection` | T1–T8 | Reference: `r_ms` added to the reference signal `r` |

The `ref_injection` approach was motivated by the attenuation argument: with post-controller force injection the excitation reaches the plant attenuated by `S = 1/(1+GC) ≪ 1` within bandwidth, whereas injecting into the reference delivers excitation via `T = GC/(1+GC) ≈ 1` — the full reference perturbation reaches the plant position output.

This reasoning is correct for **closed-loop identification**. It fails for **open-loop parameter recovery training** as implemented in this project. This document explains the mechanism in detail and records the observed experimental evidence.

---

## 2. What the Training Loop Does

`train_param_recovery.py` minimises, over all trainable parameters, the open-loop prediction error:

```
min_{params}  ||simulate(x0, u_recorded, params) - q1_recorded||²
```

Key points:
- `u_recorded` is loaded from the `.mat` file as-is.
- `q1_recorded` is the true closed-loop plant output.
- `simulate(...)` runs the LFR model **open-loop**, i.e., without any feedback.
- BPTT windows of `W=50` samples detach the state between windows but do not change the open-loop structure.

The implicit assumption for this to work is:

> **`u_recorded` must be a physically meaningful, parameter-independent excitation.** If it is not, then the residual `simulate(x0, u, params) - q1` is driven by the input structure rather than parameter errors, and gradients are uninformative.

---

## 3. Force Injection (`multisine` dataset) — Compatible

In `export_param_recovery.m` with `USE_MULTISINE = true`, the multisine force `f_sim` is generated independently of the plant:

```
f_sim  = generate_multisine(...)     % open-loop, no plant dependency
u_q1   = Cfb * (r_traj - q1)         % feedback force from reference tracking
u      = u_q1 + f_sim                 % total plant input
```

Saved to `.mat`: `u_q1`, `f_sim`, `q1`.  
Loaded in `precompute.py`: `u = u_q1 + f_sim`.

**Why this is compatible with open-loop training:**  
`f_sim` is independent of plant parameters — it is a fixed Schroeder-phase multisine signal. When the open-loop model is driven by `u = u_q1 + f_sim`, the multisine component forces the model to produce oscillations at the excited frequencies. If the model parameters are wrong, the oscillation amplitude and phase are wrong → large, **parameter-sensitive** residual → informative gradient.

---

## 4. Reference Injection (`ref_injection` dataset) — Incompatible

In `export_param_recovery_inject_ref.m`, the multisine is added to the reference:

```
r_total = r_traj + r_ms              % total reference (trajectory + position multisine)
f       = 0                          % no feedforward force ever
u_q1    = Cfb * (r_total - q1)       % total feedback force
```

Saved to `.mat`: `u_q1`, `q1`.  
Loaded in `precompute.py`: `u = u_q1` (since `f_sim` is absent or zero).

### 4.1 Closed-Loop Signal Decomposition

Consider the multisine component in isolation. In the continuous-time closed-loop:

```
u_ms  = C * (r_ms - q1_ms)       [force from tracking r_ms]
q1_ms = G * u_ms                  [plant response]
```

Solving:

```
u_ms  = C * S * r_ms    where S = 1 / (1 + GC)   [sensitivity]
q1_ms = T * r_ms        where T = GC / (1 + GC)   [complementary sensitivity]
```

Within the controller bandwidth (≤ 100 Hz in this system):

```
S ≈ 0    →    u_ms ≈ 0          (tiny force from reference multisine)
T ≈ 1    →    q1_ms ≈ r_ms      (position closely tracks the reference)
```

### 4.2 The Open-Loop Mismatch

At training time, the open-loop simulator receives `u_recorded = C*S*r_ms` (small) and must predict `q1_recorded = T*r_ms ≈ r_ms` (full amplitude). For the detuned initial model:

```
simulate(x0, u_recorded, params_detuned)
  ≈ open_loop_response(C * S * r_ms, params_detuned)   ← small force → small output
```

Residual:

```
q1_recorded - simulate(x0, u_recorded, params_detuned)
  ≈ T * r_ms - open_loop_response(C * S * r_ms, params_detuned)
  ≈ r_ms  - small                                        ← LARGE, ≈ r_ms
```

This residual is dominated by `r_ms` itself — the position reference multisine — which is **independent of plant parameters**. Gradients computed from this residual carry almost no information about which parameters need to change.

### 4.3 Effect on Training

The large uninformative residual from the within-bandwidth multisine component:
1. Dominates the MSE loss, masking the informative (parameter-sensitive) gradient from the trajectory dynamics.
2. Pushes the optimizer into bad local minima where parameters are adjusted to minimise the multisine mismatch rather than recover the true physical values.
3. Persists for all trajectories that contain Y or X multisine modes because the controller bandwidth is ~100 Hz for all axes.

---

## 5. Experimental Evidence

### 5.1 Observed losses

| Dataset | FULL_COVERAGE | Best epoch loss | Final RMSE (train) |
|---------|--------------|-----------------|-------------------|
| `base` | True | `3.2e-07` | ~0 (well converged) |
| `ref_injection` | True | `2.8e-03` | `1.63e-01` m |
| `ref_injection` | False | `2.3e-01` | `1.63e-01` m |

`base` converges 4 orders of magnitude lower. The `ref_injection` MSE plateau of `~2-3e-3` corresponds to the irreducible mismatch from the within-bandwidth multisine.

### 5.2 Parameter recovery failure for `ref_injection`

```
Parameter        True    Detuned    Learned      delta
-------------------------------------------------------
cy            10.0000    11.0000   118.3866  +1083.87%
cb_sum        18.0000    16.2000   123.1818   +584.34%
d              0.1000     0.1100     0.7183   +618.28%
cg2           20.3000    18.2700    45.1009   +122.17%
mh            10.1000     9.0900     1.6732    -83.43%
m1            10.2000    11.2200     5.0947    -50.05%
```

The parameters governing the Y-axis dynamics (`cy`, `mh`, `d`) and rotational damping (`cb_sum`) are the most severely wrong — exactly the parameters targeted by the trajectories with Y-axis multisine (T1, T5, T6, T7, T8). This is consistent with the analysis: the Y-axis multisine within bandwidth creates the largest uninformative residual, and the gradient pulls these parameters furthest from truth.

### 5.3 Per-trajectory RMSE pattern

Trajectories with Y multisine have catastrophic RMSE; trajectories with X-only motion are much closer:

| Trajectory | Mode(s) | RMSE [m] | Dominant excitation |
|------------|---------|----------|---------------------|
| T2 | `common` | 0.029 | X symmetric |
| T3 | `common` | 0.029 | X symmetric |
| T4 | `diff` | 0.004 | X anti-symmetric |
| T1 | `y` | 0.217 | Y multisine |
| T5 | `common`, `y` | 0.159 | Y + X |
| T6 | `y` | 0.242 | Y multisine |
| T7 | `diff`, `y` | 0.235 | Y + X anti |
| T8 | `common`, `diff`, `y` | 0.160 | Y + X both |

T2/T3/T4 use only X-axis multisine. The X axes are stiffer and the multisine amplitude is smaller relative to the trajectory amplitude — the within-bandwidth mismatch is present but smaller, so it does not dominate as severely.

---

## 6. Why T7 and T8 Do Not Help (And May Hurt)

T7 and T8 were added to `ref_injection` to improve observability — they excite all 13 parameters simultaneously via combined X anti-symmetric + Y sweep motion. The observability argument is sound.

**However**, T7 and T8 also both include Y multisine (`ms_modes = {'diff', 'y'}` and `{'common', 'diff', 'y'}`). This makes them subject to the same within-bandwidth mismatch as T1, T5, T6. Since T7 and T8 are the two trajectories added relative to `base`, adding them to an incompatible dataset introduces more uninformative residual without providing the intended benefit.

Additionally, T7 uses `Y_disp = 0.6 m` (full sweep, Y from 0.3 → −0.3 m) while the controller was frozen at `Y_initial = 0.3`. The closed-loop sensitivity `S` degrades as Y drifts from the design point, making the approximation `u_ms ≈ C*S*r_ms` increasingly parameter-dependent in the wrong direction (sensitivity grows, more force, but still not the right signal for open-loop ID).

**Conclusion**: T7 and T8 add value only if the training objective is compatible with closed-loop identification or if the multisine component is excluded from the training loss (see Section 8, Option 3).

---

## 7. Segment Length Is Not the Issue

A candidate explanation was that `ref_injection` uses shorter segments (~600 samples after decimation) than `base`. This is ruled out because `base` also uses ~600 samples after decimation and converges to `3.2e-07`. The segment length is the same for both datasets; the difference is entirely in the multisine structure.

---

## 8. Recommended Options

### Option 1 — Use `multisine` dataset (recommended)
Switch `DATASET = 'multisine'` in `train_param_recovery.py`. The feedforward multisine `f_sim` is a parameter-independent excitation, compatible with the open-loop training objective. T7/T8 provide the observability benefit and the multisine provides informative gradients.

**Caveat**: the old `multisine` dataset uses force injection, which IS attenuated by `S` in the real closed-loop sense. For simulation-based training this is irrelevant — the model is open-loop and `f_sim` drives it directly. The S-attenuation argument only matters when using real hardware measurements.

### Option 2 — Strip the multisine from `ref_injection` training targets
Save `r_ms` to the `.mat` file (it is already saved: `r_ms` variable in `export_param_recovery_inject_ref.m`). In `precompute.py`, subtract `r_ms` from `q1` before training:

```python
q1_corrected = q1 - r_ms_loaded   # remove the T≈1 component
```

Train on `q1_corrected`. This removes the within-bandwidth `T*r_ms ≈ r_ms` component, leaving only the parameter-sensitive part. The model then fits the residual motion that depends on plant dynamics.

**Caveat**: This requires loading `r_ms` from the `.mat` files and accounting for decimation — a non-trivial pipeline change.

### Option 3 — Frequency-band masking in the loss
Mask out the multisine frequency bands from the loss computation (e.g., apply a low-pass filter to both `simulate(...)` and `q1` before computing MSE). This suppresses the uninformative high-frequency residual while preserving the informative low-frequency trajectory mismatch.

**Caveat**: Loses the intended benefit of multisine excitation for parameter recovery — equivalent to training without multisine.

### Option 4 — Closed-loop identification framework
Replace the open-loop `simulate()` objective with a closed-loop predictor (simulation error method or prediction error method with the controller in the loop). This is structurally compatible with reference injection. This is a significant redesign of the training loop.

---

## 9. Summary

| Question | Answer |
|----------|--------|
| Why does `base` converge (loss `3.2e-07`)? | `u` is smooth, large, parameter-independent trajectory force; open-loop model matches easily |
| Why does `ref_injection` stall (loss `2.8e-03`)? | `u_ms = C*S*r_ms ≈ 0` within bandwidth; `q1_ms ≈ r_ms`; residual ≈ `r_ms` — large and parameter-insensitive |
| Why are Y-axis parameters most wrong? | Y multisine has the most within-bandwidth energy; `cy, mh, d` gradients are most corrupted |
| Why don't T7/T8 help? | They add Y multisine on top of an already incompatible objective |
| What is the correct approach? | Force injection (`multisine` dataset) — `f_sim` is parameter-independent and drives the open-loop model directly |
