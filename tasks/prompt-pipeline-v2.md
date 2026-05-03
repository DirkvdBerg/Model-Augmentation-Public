# Implementation Prompt — Parameter Recovery Data Pipeline v2

**For**: New Claude Code session  
**Date written**: 2026-05-03  
**Design doc**: `docs/param-recovery-data-pipeline-v2.md`  
**Decision log**: `docs/decisions.md` → D-047

---

## Context

We are implementing the v2 data pipeline for LPV-LFR parameter recovery training on the
dual-gantry system. The goal is to determine and apply the correct sampling rate and
segment length from data diagnostics, reducing training time by ~66×.

The three scripts involved are:
- `lpv_lfr_baseline/scripts/experiment_diagnostics.py`
- `lpv_lfr_baseline/scripts/precompute.py`
- `lpv_lfr_baseline/scripts/train_param_recovery.py`

Read all three files in full before making any changes.

---

## What has already been done (do NOT redo)

- `_diag_fft` in `experiment_diagnostics.py` is complete and working. It returns
  `f99_overall`, `fs_new`, `decimation_factor` (D). Do not change it.
- `_diag_observability` is complete and working. Do not change it.
- `_build_segment_pools` and `_sample_batch` are in `precompute.py` and
  `train_param_recovery.py` respectively and are correct. Do not change them.
- The cache fingerprint + fast-path (overlap_fraction-only change) in `precompute.py`
  is already implemented. Do not change that logic.
- `simulate()` and `rk4_step()` already accept `ts` as an explicit tensor argument.
  Do NOT modify `lfr_simulate.py` — it is read-only for this task.

---

## What needs to be implemented

### Step 1 — Restructure `_diag_step_response` in `experiment_diagnostics.py`

Currently it returns `tau_max` and `poles`. It must also return `segment_len` and
`f_osc_min`, computed as follows:

1. After computing eigenvalues at each Y operating point, extract the **oscillatory poles**
   — those with `|Im(λ)| > 1.0` rad/s (complex conjugate pairs).
2. Compute `f_osc_min = min(|Im(λ)|) / (2π)` across all Y points and all oscillatory poles.
   This is the slowest oscillatory frequency in rad/s converted to Hz.
3. `segment_len_s = N_PERIODS / f_osc_min`  where `N_PERIODS = 3` (add as module constant).
4. `segment_len = math.ceil(segment_len_s * fs_new)` where `fs_new` is passed in as a new
   argument.
5. Add to the print output: `f_osc_min`, `segment_len_s`, `segment_len`.
6. Return dict gains: `f_osc_min` [Hz], `segment_len` [samples at fs_new],
   `segment_len_s` [seconds].

Updated signature:
```python
def _diag_step_response(fs, fs_new, save_dir, dtype=torch.float64):
```

### Step 2 — Remove `_diag_param_sensitivity` from `experiment_diagnostics.py`

Delete the entire function `_diag_param_sensitivity` and its plot helper
`_plot_sensitivity`. Also remove these now-unused module constants:
- `_ENERGY_THRESHOLD`
- `_SENS_T_TEST_MAX`
- `_SENS_T_TEST_FACTOR`
- `_PARAM_CATEGORIES`
- `_CATEGORY_COLORS`

Update the module docstring at the top of the file to remove all mentions of
"Parameter sensitivity" from the diagnostic list and public API description.

### Step 3 — Update `recommend_segment_len` in `experiment_diagnostics.py`

Currently calls both `_diag_step_response` and `_diag_param_sensitivity`.

New version: calls `_diag_step_response` only (no `trajs` needed).

New signature:
```python
def recommend_segment_len(fs, fs_new, save_dir, dtype=torch.float64):
    r_step = _diag_step_response(fs, fs_new, save_dir=None, dtype=dtype)
    return r_step['segment_len']
```

### Step 4 — Update `run_all_diagnostics` in `experiment_diagnostics.py`

Currently calls all 4 diagnostics. New version calls 3:
1. `r_fft  = _diag_fft(trajs, save_dir)`
2. `r_step = _diag_step_response(fs, r_fft['fs_new'], save_dir, dtype=torch.float64)`
3. `r_obs  = _diag_observability(fs, save_dir, dtype=torch.float64)`

Update the summary block accordingly. Remove all sensitivity references.

### Step 5 — Apply decimation in `precompute._compute`

After loading each trajectory (the `for spec in traj_specs` loop), apply decimation
using D from the FFT result. The FFT must be called first, before the trajectory loop
returns from `_compute`.

Restructure `_compute` as follows:

```
1. Load all trajectories into trajs list (same as now)
2. Call _diag_fft(trajs, save_dir) → get D, fs_new
3. Decimate each traj in-place:
       traj['u']          = traj['u'][:, ::D, :]          # (1, T//D, 3)
       traj['q1']         = traj['q1'][::D, :]             # (T//D, 3)
       traj['state_traj'] = traj['state_traj'][::D, :]     # (T//D, 6)
       traj['N']          = traj['u'].shape[1]
       traj['fs']         = fs_new
4. Compute ts_eff = float(_ts) * D  (import _ts from lpv_lfr_baseline.core.physics)
5. Call recommend_segment_len(fs_orig, fs_new, save_dir, dtype) to get segment_len
   (fs_orig = native fs before decimation, fs_new = after)
6. Build pools using the decimated traj['N'] and new segment_len
7. Store D, fs_new, ts_eff in metadata and in the returned dict
```

The returned dict from `_compute` must include:
- `'D'`: int decimation factor
- `'fs_new'`: float new sampling rate
- `'ts_eff'`: float effective timestep = `_ts * D`

### Step 6 — Update `precompute()` fingerprint

Add `'D'` to the fingerprint dict in `_fingerprint()`. D is derived from the data
(signal bandwidth), so a change in D should invalidate the full cache. It does NOT
belong in the fast-path (overlap_fraction only).

Note: D is deterministic given the data (computed by `_diag_fft`), so you do NOT need
to pass D as an argument to `precompute()`. It is computed internally.

### Step 7 — Update `train_param_recovery.py`

1. Load `ts_eff` from `pre['ts_eff']` after the precompute call.
2. Everywhere `block._ts` is used as the timestep in `simulate()` calls, replace with
   `torch.tensor(ts_eff, dtype=DTYPE, device=device)`.

   Specifically:
   - In `_run_no_grad`: pass `torch.tensor(ts_eff, ...)` instead of `block._ts`
   - In `train()` epoch loop: pass `torch.tensor(ts_eff, ...)` to `simulate()`
   
   Note: `ts_eff` is a Python float loaded from the cache. Convert to tensor once
   outside the epoch loop:
   ```python
   ts_tensor = torch.tensor(ts_eff, dtype=DTYPE, device=device)
   ```
   Then use `ts_tensor` in all `simulate()` calls.

3. Update the logging line to show `fs_new` and `D`:
   ```
   segment_len=XXX, W=50, n_windows=XX, batch=8 trajs/epoch
   fs_new=1000 Hz (D=20), overlap=0%, stride=XXX
   ```

4. `_run_no_grad` currently receives `block._ts` implicitly via `simulate`. Since
   `_run_no_grad` is a module-level function, pass `ts_eff` as an explicit argument:
   ```python
   def _run_no_grad(block, x0, u, ts_eff):
   ```
   Update all call sites.

---

## Key invariants to preserve

- `lfr_simulate.py` and `lfr_forward.py` — READ ONLY. Do not touch.
- `kamtin-fp-model/` — READ ONLY. Do not touch.
- The fast-path in `precompute()` (overlap_fraction-only cache update) must still work.
- `_build_segment_pools` and `_sample_batch` in their respective files — do not change.
- `_diag_observability` — do not change.
- `_diag_fft` — do not change.

---

## Verification

After implementation, run:

```
conda run -n GraduationProject python -m lpv_lfr_baseline.scripts.experiment_diagnostics
```

Expected output:
- FFT: f_99 ≈ 86 Hz, fs_new = 1000 Hz, D = 20
- Step response: tau_max ≈ 1.57 s, f_osc_min ≈ 4.94 Hz,
  segment_len ≈ 607 samples at 1000 Hz
- Observability: horizon = 2
- Summary shows segment_len ≈ 607 samples

Then run a short training smoke test (5 epochs):
```
conda run -n GraduationProject python -c "
from lpv_lfr_baseline.scripts.train_param_recovery import train
train(epochs=5)
"
```

Expected: precompute runs diagnostics, decimates data to 1000 Hz, builds pools with
~3 segments per trajectory, training loop completes 5 epochs without error.

---

## Expected result summary

| Quantity | Before | After |
|---|---|---|
| Training fs | 20000 Hz | 1000 Hz |
| ts_eff | 5×10⁻⁵ s | 1×10⁻³ s |
| segment_len | ~39420 samples | ~607 samples |
| Segments per trajectory | 1 | ~3 |
| Simulation steps/epoch | ~315000 | ~4856 |
