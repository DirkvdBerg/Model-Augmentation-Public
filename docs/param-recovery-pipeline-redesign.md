# Parameter Recovery Pipeline Redesign

**Date**: 2026-04-30 (updated 2026-05-01)  
**Session**: Supervisor meeting follow-up — diagnostics and training data pipeline  
**Scope**: `experiment_diagnostics.py` (replaces `segment_diag.py`), trajectory subdivision,
mini-batch structure  
**Status**: Design complete. Q2 and Q3 in §3.3 still open — resolve before implementing §3.

---

## 1. Context and motivation

Two things are being redesigned together because they share a dependency:

1. The segment length used to subdivide trajectories is determined by diagnostics.
2. The training mini-batch structure is defined by how those subdivisions are laid out.

### Why the current approach is insufficient

**`segment_diag.py`**: scored candidate window lengths by checking whether true parameters
beat detuned alternatives on short segments. This is an indirect proxy — it measures output
distinguishability, not gradient informativeness. It does not account for parameter-specific
timescales and can select a window too short for slow parameters (stiffness, low-frequency
damping) to receive any meaningful gradient.

**Training loop**: 8 segments sampled randomly per epoch from any trajectory. No balance
guarantee: one trajectory may appear multiple times while another is skipped entirely.
Start indices are resampled every epoch even though the valid range is fixed.

---

## 2. `experiment_diagnostics.py` — replaces `segment_diag.py`

### 2.1 File location and role

```
lpv_lfr_baseline/scripts/experiment_diagnostics.py
```

- Called by `precompute._compute()` for the `segment_len` recommendation (replaces
  `run_segment_diag`).
- Can be run standalone via `__main__` to produce all plots and printed summaries.
- Does **not** own trajectory loading — it receives the precompute output dict so
  trajectories are not loaded twice.

### 2.2 Diagnostic pipeline — sequential, each feeds the next

The four diagnostics run in order. The output of each informs the inputs of the next.

```
[1] FFT analysis
      ↓ recommended fs (decimation factor)
[2] Step response  (run at decimated fs)
      ↓ τ_max, dominant poles per Y operating point
[3] Parameter sensitivity  (run at decimated fs)
      ↓ minimum window per parameter → global segment_len
[4] Observability analysis  (run at decimated fs)
      ↓ sanity check: confirms horizon is short (expected: 2 samples)
```

---

### 2.3 Diagnostic 1 — FFT / frequency content

**Purpose**: determine whether 20 kHz is excessive and recommend a decimation factor.

**What to compute**:
- For each trajectory, compute the one-sided FFT of each output channel (X1, X2, Y)
  using `torch.fft.rfft` on the full signal.
- Compute the power spectral density (PSD): `|FFT|² / N`.
- Find the frequency `f_99` at which cumulative PSD energy reaches 99% of total.
- Find the frequency `f_99` across all trajectories and channels; take the maximum.
- Recommended `f_keep = f_99`. Recommended `fs_new` = smallest value in
  {1000, 2000, 4000, 8000} Hz that satisfies `fs_new >= 8 * f_keep`.
- Decimation factor `D = round(fs_original / fs_new)`.

**What to print**:
```
FFT analysis
  fs_original = 20000 Hz
  f_99 per channel (max across trajectories):
    X1: XXX Hz   X2: XXX Hz   Y: XXX Hz
  f_99 overall: XXX Hz
  Recommended fs_new: XXXX Hz  (decimation factor D=X)
```

**What to plot**:
- One figure with 3 subplots (one per channel). Each subplot: overlaid PSD curves
  for all trajectories (log-log scale). Vertical dashed line at `f_99`. Title states
  recommended decimation factor.
- Save as `fft_analysis.png` in `save_dir`.

**Ground**: supervisor recommendation (Jan Hoekstra, verbal). Supported by standard
sampled-data rule: `fs >= 10 × bandwidth`. For 100 Hz closed-loop bandwidth,
4 kHz = 40× is conservative and safe; 20 kHz is likely overkill. A 5× reduction
in `fs` for the same physical window length reduces sample count 5× and directly
reduces training cost.

---

### 2.4 Diagnostic 2 — Step response / dominant time constant

**Purpose**: determine `τ_max` (dominant time constant) which gives a lower bound on
segment length for the fast (inertial) parameters.

**What to compute**:
- Freeze the LPV model at three representative Y operating points:
  `Y ∈ {0.00, 0.20, 0.30}` m (covers the range of T1–T6 trajectories).
- At each Y, build the frozen discrete-time A matrix using `_build_matrices` and
  `build_poly_constants` with true parameters. Compute eigenvalues of A.
- Convert discrete-time poles `λ_i` to continuous-time time constants:
  `τ_i = -T_s / ln(|λ_i|)`. Take `τ_max = max(τ_i)` across all poles and Y values.
- Simulate a step input (unit step on each force channel separately, or all at once)
  using `simulate()` with `x0 = zeros(1, 6)` and true parameters.
  Run for `t_sim = 5 * τ_max` seconds (= `ceil(5 * τ_max / T_s)` samples).
- Measure the 2%–5% settling time from the step response output.

**What to print**:
```
Step response analysis
  True parameters, T_s = X.XXXe-XX s
  Operating point Y=0.00 m:  poles (top 3 by τ): ...,  τ_max = X.XXX s
  Operating point Y=0.20 m:  ...
  Operating point Y=0.30 m:  ...
  Global τ_max = X.XXX s  →  min segment (3τ) = XXX samples at fs_new
                             min segment (5τ) = XXX samples at fs_new
```

**What to plot**:
- One figure: step response of all 3 output channels to a combined step input,
  at each Y operating point (overlaid or separate subplots). Vertical dashed lines
  at 3τ_max and 5τ_max. Save as `step_response.png`.

**Ground**: explicit recommendation by supervisor Jan Hoekstra. Not in Hoekstra (2025)
paper — that paper does not perform parameter recovery. The step response gives the
lower bound for inertial parameters (mh, m1, m2, mb, Jb, Jh) which dominate at ω².

---

### 2.5 Diagnostic 3 — Parameter sensitivity trajectories

**Purpose**: determine the minimum segment length driven by the slowest parameters.
The step response captures fast dynamics; this captures slow ones (stiffness, damping).

**Background**: The minimum segment length for reliable parameter identification is set
by the slowest-informing parameter, not by `τ_max` alone. From the frozen system
response `Q(jω) = (−ω²M + jωC + K)⁻¹`:
- **Inertial** (mh, m1, m2, mb, Jb, Jh): dominate at ω², visible on short timescales.
- **Damping** (cg1, cg2, cb1, cb2): dominate at ω, intermediate timescales.
- **Stiffness** (kb1, kb2, cy): quasi-static term, only visible over multiple oscillation
  cycles, potentially much longer than `τ_max`.

Grounded in Fisher Information Matrix theory: the FIM for output-error identification is
`F = Sᵀ R⁻¹ S` where `S` contains output sensitivities `∂y/∂θᵢ`. A segment where
`∂y/∂θᵢ ≈ 0` contributes nothing to the FIM for parameter `θᵢ`. Reference: Ljung
(2010), cited in Hoekstra (2025).

**What to compute**:
- Use a representative trajectory (e.g. T5, which excites all channels). If running
  standalone, use the first trajectory in `trajs` that has Y variation.
- Run `simulate()` with true parameters (from `_TRUE_PARAMS`) on a segment of length
  `T_sens = ceil(10 * τ_max / T_s_new)` samples (10 time constants at decimated rate).
- Compute `∂Y/∂log_params` via `torch.autograd.grad` with `retain_graph=True` for each
  output timestep. Concretely: loop over time steps or use `torch.func.jacrev` on the
  simulation output w.r.t. `block.log_params`.
- For each parameter `θᵢ` (index into `_PARAM_NAMES`), compute the sensitivity norm:
  `s_i(t) = ||∂Y(t) / ∂log_params_i||` (norm over output channels).
- Compute the cumulative energy: `E_i(t) = cumsum(s_i(t)²) / sum(s_i(t)²)`.
- Find `t_95_i`: first time step where `E_i(t) >= 0.95` (95% of gradient energy).
- `segment_len_samples = max(t_95_i for all i)` at decimated `fs_new`.

**What to print**:
```
Parameter sensitivity analysis  (fs = XXXX Hz)
  param       t_95 [s]   t_95 [samples]   category
  mh          X.XXX      XXXX             inertial
  m1          X.XXX      XXXX             inertial
  ...
  kb1         X.XXX      XXXX             stiffness
  cy          X.XXX      XXXX             stiffness
  ...
  Slowest parameter: XXX  (t_95 = X.XXX s = XXXX samples)
  Recommended segment_len = XXXX samples  (= X.XXX s at fs_new)
```

**What to plot**:
- One figure: subplot per parameter (or grouped by category). Plot `s_i(t)` over time
  and the cumulative `E_i(t)`. Vertical dashed line at `t_95_i`. Save as
  `param_sensitivity.png`.

---

### 2.6 Diagnostic 4 — Observability analysis

**Purpose**: sanity check — confirm the observability horizon for our specific system
structure. Expected result is 2 samples (not a bottleneck).

**Background**: For a system with state `x = [q; q̇]` and output `y = q` (positions only),
`C = [I, 0]`. Then `O_1 = C = [I 0]` and `O_2 = [C; CA]`. Since `CA` contains the
velocity rows, `O_2` typically has full column rank. The theoretical bound is `n_state = 6`
steps, but for our C=[I,0] mechanical system it is expected to be 2.

**What to compute**:
- At each of the three Y operating points from §2.4, build the frozen `A` matrix.
- Build `O_t = [C; CA; CA²; ...; CA^(t-1)]` for `t = 1, 2, ..., 6`.
- At each `t`, compute `rank(O_t)` using `torch.linalg.matrix_rank`.
- Record the first `t` where `rank(O_t) == n_state` — this is the observability horizon.

**What to print**:
```
Observability analysis
  Y=0.00 m:  rank(O_1)=X  rank(O_2)=X  rank(O_3)=X  ...  horizon=X steps
  Y=0.20 m:  ...
  Y=0.30 m:  ...
  Conclusion: observability horizon = X samples (NOT the bottleneck for segment_len)
```

No plot needed — table is sufficient.

---

### 2.7 Public interface

```python
def recommend_segment_len(trajs, fs, save_dir, dtype=torch.float64) -> int:
    """
    Run step response and parameter sensitivity diagnostics to determine
    the minimum segment length in samples at the given fs.

    Called by precompute._compute() — replaces run_segment_diag().
    Saves plots to save_dir. Returns segment_len in samples at fs.

    Parameters
    ----------
    trajs    : list of traj dicts from precompute (id, u, q1, state_traj, N, fs)
    fs       : sampling frequency to use for the analysis (after decimation)
    save_dir : directory to save diagnostic plots and printed output
    dtype    : torch dtype (default float64)

    Returns
    -------
    int — segment_len in samples at the given fs
    """

def run_all_diagnostics(pre, save_dir) -> None:
    """
    Run all four diagnostics using the precompute output dict.
    Saves all plots and prints all summaries.
    Called by __main__ for standalone inspection.

    Parameters
    ----------
    pre      : dict returned by precompute() — keys: trajs, sigma, segment_len, metadata
    save_dir : directory to save plots
    """
```

**`__main__` block** (standalone use):
```python
if __name__ == '__main__':
    # 1. Load _DATASETS from train_param_recovery (benign module-level import)
    # 2. Parse --multisine flag from sys.argv to select dataset
    # 3. Call precompute() to get/load trajectory cache
    # 4. Call run_all_diagnostics(pre, save_dir)
    # 5. All plots saved to save_dir; summaries printed to stdout
```

Run as:
```
conda run -n GraduationProject python -m lpv_lfr_baseline.scripts.experiment_diagnostics
conda run -n GraduationProject python -m lpv_lfr_baseline.scripts.experiment_diagnostics --multisine
```

### 2.8 Integration with `precompute.py`

Replace lines 221–223 in `precompute._compute()`:

```python
# OLD
from lpv_lfr_baseline.scripts.segment_diag import run_segment_diag
segment_len = run_segment_diag(traj_specs, traj_dir, save_dir)

# NEW
from lpv_lfr_baseline.scripts.experiment_diagnostics import recommend_segment_len
fs_new = <decimation-recommended fs from FFT, or pass fs from trajs>
segment_len = recommend_segment_len(trajs, fs_new, save_dir, dtype=dtype)
```

The precompute cache format does not change — `segment_len` is still stored as an `int`.
The cache fingerprint (`_COMPUTE_HASH`) will naturally invalidate because `_compute` changes.

### 2.9 Helpers migrated out of `segment_diag.py`

| Helper | Destination |
|--------|-------------|
| `_attach_valid_start_idx` | `train_param_recovery.py` (training-only) |
| `_sample_balanced_segments` | **Removed** — replaced by pre-division in §3 |
| `_traj_set_tag` | `precompute.py` (used for cache key naming) |
| `_load_trajs` | **Removed** — `experiment_diagnostics` uses `pre['trajs']` directly |

---

## 3. Training data pipeline — trajectory subdivision and mini-batch structure

### 3.1 Resolved design decisions

**Q1 (segment length consistency): RESOLVED — single global `segment_len` for all trajectories.**

Reason: `segment_len` is driven by the slowest-informing parameter, which is a property
of the physical system, not of any individual trajectory. All trajectories are from the
same dual-gantry system. Using different lengths per trajectory would produce different
loss scales and make gradients harder to combine. Uniform length also enables pre-allocated
batch tensors (no padding required).

**Batch size = number of active trajectories.**  
Each trajectory contributes exactly one segment per batch. This guarantees all trajectories
are represented in every gradient update. For the base dataset (T1–T6): batch size = 6.
For multisine dataset (T1–T8): batch size = 8.

### 3.2 Pre-division procedure (done once at training setup, not per epoch)

```python
def _build_segment_pool(trajs, segment_len, stride=None):
    """
    Pre-divide all trajectories into fixed segments.
    
    stride=None defaults to segment_len (non-overlapping).
    stride=segment_len//2 gives 50% overlap.
    
    Returns a dict: traj_id -> list of (x0, u_seg, q1_seg) tensors,
    where each tensor is already sliced and CPU-resident.
    All segments have exactly shape (segment_len, ...) — final short
    segment of each trajectory is discarded.
    """
    pool = {}
    for traj in trajs:
        s = stride if stride is not None else segment_len
        starts = range(0, traj['N'] - segment_len + 1, s)
        pool[traj['id']] = [
            {
                'x0':  traj['state_traj'][start],           # (6,)
                'u':   traj['u'][0, start:start+segment_len, :],  # (segment_len, 3)
                'q1':  traj['q1'][start:start+segment_len, :],    # (segment_len, 3)
            }
            for start in starts
        ]
    return pool
```

Called once after `precompute()` returns, before the training loop begins.

### 3.3 Mini-batch construction per epoch step

```python
def _sample_batch(pool, trajs, device, dtype):
    """
    Sample one segment per trajectory. Returns stacked batch tensors.
    
    Returns
    -------
    x0_batch  : (B, 6)              — initial states
    u_batch   : (B, segment_len, 3) — inputs
    q1_batch  : (B, segment_len, 3) — reference outputs
    traj_ids  : list of str, length B — for sigma lookup
    """
    x0_list, u_list, q1_list, ids = [], [], [], []
    for traj in trajs:
        seg = random.choice(pool[traj['id']])
        x0_list.append(seg['x0'])
        u_list.append(seg['u'])
        q1_list.append(seg['q1'])
        ids.append(traj['id'])
    return (
        torch.stack(x0_list).to(device=device, dtype=dtype),
        torch.stack(u_list).to(device=device, dtype=dtype),
        torch.stack(q1_list).to(device=device, dtype=dtype),
        ids,
    )
```

This replaces `_sample_balanced_segments` in the training loop.

### 3.4 Epoch structure

```
for epoch in range(epochs):
    # Shuffle pool order once per epoch (optional, for variety)
    batch_x0, batch_u, batch_q1, traj_ids = _sample_batch(pool, trajs, device, DTYPE)
    sigma_batch = torch.stack([sigma_device[tid] for tid in traj_ids])  # (B, 3)
    # ... BPTT windowed loss, exactly as before ...
```

The inner BPTT window loop (`W` steps per window, `n_windows = ceil(segment_len / W)`)
is **unchanged**. The only change is how the batch is constructed.

### 3.5 Open questions (resolve before implementing §3)

**Q2: Full trajectory vs. mini-batch per trajectory**

Options:
- **Mini-batch** (described in §3.2–§3.4): one segment per trajectory per step.
  Cheap per step, requires many epochs. Current plan.
- **Full trajectory**: roll out the complete trajectory each step, accumulate loss
  over all timesteps with BPTT windows. Maximum gradient signal per step. Cost scales
  with trajectory length (T1 ≈ 60,000 samples at 20 kHz, ≈ 12,000 at 4 kHz).

Decision pending: after FFT diagnostic determines `fs_new` and after parameter
sensitivity determines `segment_len`, check whether a full trajectory at `fs_new`
is computationally feasible per step.

**Q3: Overlap factor for segment pool**

Options:
- **Non-overlapping** (`stride = segment_len`): cleanest, minimal correlation between
  consecutive batches. May give small pool for short trajectories.
- **50% overlap** (`stride = segment_len // 2`): doubles pool size, acceptable
  correlation.
- **No overlap needed if using mini-batch**: if Q2 resolves to mini-batch, the pool
  size matters for variety across epochs. If Q2 resolves to full trajectory, no pool
  is needed.

Decision pending Q2.

### 3.6 What does not change

- The BPTT window `W` (inner loop, detaches state between windows).
- The `simulate()` call and its signature.
- The loss computation: `(result.Y - q1_win) / sigma_batch.unsqueeze(1)`.
- `precompute.py` owns trajectory loading and `segment_len` storage.
- The async eval worker and full-trajectory eval logic in `train_param_recovery.py`.

---

## 4. Normalization — change from per-trajectory to global

**Resolved**: switch `NORM_MODE` default from `'per_traj'` to `'global'`.

Supervisor feedback: "normalization should be global — the system does not change from
one experiment to another; per-trajectory normalization is conceptually wrong."
Confirmed by Hoekstra (2025) §3.5: normalization uses statistics from the full
training set, not per-trajectory statistics.

**Implementation**: `precompute.py` already supports `norm_mode='global'` (see
`_compute_sigma`). Change the default in `train_param_recovery.py`:
```python
NORM_MODE = 'global'   # was 'per_traj'
```
This invalidates the precompute cache (fingerprint includes `norm_mode`) and triggers
a recompute on next run. No other code changes needed.

---

## 5. Summary of all file changes

| File | Change |
|------|--------|
| `segment_diag.py` | **Delete entirely** |
| `experiment_diagnostics.py` | **Create** — 4 diagnostics (§2.3–2.6), public interface (§2.7) |
| `precompute.py` | Replace `run_segment_diag` call with `recommend_segment_len` (§2.8); absorb `_traj_set_tag` |
| `train_param_recovery.py` | Change `NORM_MODE='global'`; absorb `_attach_valid_start_idx`; replace `_sample_balanced_segments` with `_build_segment_pool` + `_sample_batch` (§3.2–3.4, pending Q2/Q3) |

---

## 6. Implementation order

1. **First**: implement `experiment_diagnostics.py` (independent of training changes).
   Run it on the existing data to get `fs_new` and `segment_len` recommendations.
2. **Then**: update `precompute.py` to call `recommend_segment_len` and add
   `_traj_set_tag`. Delete `segment_diag.py`.
3. **Then**: resolve Q2/Q3 using diagnostic results, then rewrite the training loop in
   `train_param_recovery.py` (§3.2–3.4).
4. **Finally**: switch `NORM_MODE='global'`, rerun precompute, validate training.

---

## 7. Open items

- [ ] **Q2**: Full trajectory vs. mini-batch — decide after seeing `fs_new` and
  `segment_len` from diagnostics (determines feasibility of full-trajectory per step)
- [ ] **Q3**: Overlap factor — decide after Q2
- [ ] Confirm step input for step response: apply to each force channel separately or
  combined (e.g. equal unit force on all three)
- [ ] Confirm `T_sens = 10 * τ_max` is long enough for slowest parameter — adjust if
  sensitivity curves have not decayed to 5% at that horizon
- [ ] Confirm FFT plot: overlaid per-trajectory curves, or mean ± std band
- [ ] Confirm whether diagnostic plots are shown interactively or only saved to disk
  (recommendation: save only, for server compatibility)
