# Parameter Recovery Data Pipeline — v2 Design

**Date**: 2026-05-03  
**Status**: Planned — not yet implemented  
**Replaces**: `docs/param-recovery-pipeline-redesign.md` (v1, sensitivity-based)

---

## Problem statement

The v1 pipeline trained at native 20 kHz with a fixed segment length (4000 samples = 0.2 s).
This was too short to capture the dominant oscillatory dynamics (~5 Hz, period 0.2 s),
causing poor gradient signal for slow parameters (cg1, cg2, cy). Training took ~10 hours.

The v2 pipeline uses FFT + step response diagnostics to determine the correct sampling rate
and minimum segment length from the data, then applies both in precompute before training.

---

## Design overview

```
Raw MATLAB data (20 kHz)
        │
        ▼
  [Diagnostic 1: FFT]
  ─ Find f_99 (99% energy frequency)
  ─ Compute D = round(fs_orig / fs_new)
    where fs_new = smallest candidate ≥ 8 × f_99
  ─ Result: D = 20, fs_new = 1000 Hz
        │
        ▼
  [precompute: decimate]
  ─ u          → u[:, ::D, :]       (1, T//D, 3)
  ─ q1         → q1[::D, :]         (T//D, 3)
  ─ state_traj → state_traj[::D, :] (T//D, 6)
  ─ ts_eff     = ts_native × D      (1e-3 s at 1000 Hz)
  ─ Cache stored at 1000 Hz
        │
        ▼
  [Diagnostic 2: Step response]
  ─ Build A_c at Y ∈ {0.00, 0.20, 0.30} m from detuned params
  ─ Extract oscillatory poles (complex eigenvalues)
  ─ f_osc_min = min |Im(λ)| / (2π)  across all Y
  ─ segment_len_s = N_PERIODS / f_osc_min   (N_PERIODS = 3)
  ─ segment_len   = ceil(segment_len_s × fs_new)
  ─ Result: f_osc_min ≈ 4.94 Hz → segment_len ≈ 607 samples at 1000 Hz
        │
        ▼
  [precompute: build pools]
  ─ stride = segment_len × (1 − OVERLAP_FRACTION)
  ─ pools[traj_id] = list of valid start indices
  ─ At 1000 Hz, T = 2000 samples, segment_len ≈ 607
  ─ Result: ~3 non-overlapping segments per trajectory
        │
        ▼
  [Diagnostic 3: Observability]  (fast, no data needed)
  ─ Build O_h at each Y, track rank growth
  ─ Expected: full rank at h = 2
        │
        ▼
  [Training — train_param_recovery.py]
  ─ Loads cached decimated data + pools from precomputed.pt
  ─ Simulates at ts_eff = ts_native × D
  ─ Each epoch: sample 1 segment per trajectory → batch of 8
  ─ BPTT with W = 50 windows over 607-sample segment
  ─ ~20× faster than native 20 kHz
```

---

## Key numbers (expected after implementation)

| Quantity | v1 (20 kHz, no decimation) | v2 (1000 Hz, D=20) |
|---|---|---|
| Training fs | 20000 Hz | 1000 Hz |
| ts_eff | 5×10⁻⁵ s | 1×10⁻³ s |
| Trajectory length | 40000 samples | 2000 samples |
| segment_len | 4000 samples (0.2 s, fixed) | ~607 samples (0.61 s, diagnostic) |
| Segments per trajectory | ~10 (but too short) | ~3 (physically motivated) |
| Simulation steps/epoch | ~320000 | ~4856 |
| Relative training speed | 1× | ~66× faster |

---

## Files to change

### `lpv_lfr_baseline/scripts/experiment_diagnostics.py`

1. **Remove** `_diag_param_sensitivity` and `_plot_sensitivity` entirely (see D-047)
2. **Remove** constants: `_ENERGY_THRESHOLD`, `_SENS_T_TEST_MAX`, `_SENS_T_TEST_FACTOR`,
   `_PARAM_CATEGORIES`, `_CATEGORY_COLORS`
3. **Modify** `_diag_step_response`:
   - Extract complex (oscillatory) poles from eigenvalues
   - Compute `f_osc_min = min(|Im(λ)|) / (2π)` across all Y operating points
   - Add `N_PERIODS = 3` constant
   - Return `segment_len` (at fs_new) and `f_osc_min` in addition to `tau_max`
4. **Modify** `_diag_fft`:
   - Return `D` and `fs_new` as before (no change)
5. **Modify** `recommend_segment_len`:
   - Signature: `recommend_segment_len(fs, fs_new, dtype)` — no longer needs `trajs`
   - Calls `_diag_step_response` only, returns `(segment_len, D)` or separate values
6. **Modify** `run_all_diagnostics`:
   - Calls FFT → step response (passes `fs_new` from FFT) → observability
   - No sensitivity call
7. **Update** module docstring

### `lpv_lfr_baseline/scripts/precompute.py`

1. **Modify** `_compute`:
   - After loading trajectories: apply decimation using D from `_diag_fft`
   - Store `ts_eff = float(_ts) * D` in the cache
   - Call `recommend_segment_len(fs, fs_new, dtype)` — passes `fs_new` from FFT result
   - All cached tensors are at `fs_new`, not native fs
2. **Modify** `precompute()`:
   - Add `ts_eff` to returned dict
   - Fingerprint includes `D` (tied to data content, not a hyperparameter)
3. **Add** `D` and `fs_new` and `ts_eff` to `metadata`

### `lpv_lfr_baseline/scripts/train_param_recovery.py`

1. **Modify** `train()`:
   - Load `ts_eff` from `pre['ts_eff']`  (replaces hardcoded `block._ts`)
   - Pass `ts_eff` to `simulate()` calls (both in `_build_sim_params` path and `_run_no_grad`)
   - Update logging to show `fs_new` and `D`

---

## Design rationale

**Why decimate in precompute, not on-the-fly in training?**
Precompute is the right place — it is a one-time transformation of the data that does not
depend on trainable parameters. Caching the decimated data avoids repeating the transform
every run and keeps the training loop simple.

**Why D from FFT, not hardcoded?**
D is a property of the data (signal bandwidth), not a training hyperparameter. If the
dataset changes (different excitation, different system), D should update automatically.

**Why N_PERIODS = 3?**
Three complete oscillation cycles give enough data to observe the resonance and its decay.
The subagent research (2026-05-03) confirms 3–5 time constants is standard practice.
N_PERIODS = 3 is conservative and results in ~3 segments per 2 s trajectory at 1000 Hz.
Can be increased if training is unstable.

**Why not use tau_max for segment_len?**
tau_max = 1.57 s is the slowest REAL pole (rigid-body / damping drift). Using 3×tau_max
gives 4.7 s — longer than the trajectories. The oscillatory poles are the physically
relevant dynamics for identification: they are excited by the input and visible in the
output. The real slow poles represent drift that is not identifiable from short segments
regardless of segment length.

**Why remove sensitivity?**
See D-047. Not supervisor-suggested, slow (224 simulate calls), and produced an unusable
result (t_95 hit the trajectory cap, giving segment_len = full trajectory). The step
response oscillatory criterion gives a simpler, faster, and more practically useful answer.

---

## Open questions before implementation

1. Does `simulate()` accept an arbitrary `ts` argument, or is it read from `block._ts`?
   If the latter, `block._ts` must be overridden or a separate argument added.
2. Should `ts_eff` invalidate the fingerprint (yes — it is derived from D which is
   data-dependent and already in the fingerprint via `fs_new`).
3. Should `N_PERIODS` be user-configurable (CLI arg to `experiment_diagnostics.py`)?
   Default: no — keep it simple unless a problem arises.
