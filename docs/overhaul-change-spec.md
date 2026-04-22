# Overhaul Change Specification

Companion to `docs/pytorch-optimization-guidelines.md`. That document states governing
principles; this document states exactly what changes in each file.

---

## `lpv_lfr_baseline/core/lfr_matrices.py`

### Problem
`GMatrix` is a Python `@dataclass`. `torch.jit.script` uses `inspect.getsource` on
`__init__`, which fails for dataclass-generated methods. This blocks `jit.script` on
any function that receives a `GMatrix` argument, including `lfr_forward` and `rk4_step`.

### Change
Replace the `@dataclass` with a `NamedTuple` subclass. All seven fields, all shapes,
and all `float64` semantics stay identical. `build_G_matrix` return type annotation
changes from `GMatrix` (dataclass) to `GMatrix` (NamedTuple) — no change in the call
sites.

```python
# REMOVE
from dataclasses import dataclass

@dataclass
class GMatrix:
    Ax:  torch.Tensor
    Bw:  torch.Tensor
    Bu:  torch.Tensor
    Cz:  torch.Tensor
    Dzw: torch.Tensor
    Dzu: torch.Tensor
    Cy:  torch.Tensor

# ADD
from typing import NamedTuple

class GMatrix(NamedTuple):
    Ax:  torch.Tensor   # (6, 6)
    Bw:  torch.Tensor   # (6, 6)
    Bu:  torch.Tensor   # (6, 3)
    A_combined: torch.Tensor # (6, 15)  [Ax, Bw, Bu] concatenated for fused update
    Cz:  torch.Tensor   # (6, 6)
    Dzw: torch.Tensor   # (6, 6)
    Dzu: torch.Tensor   # (6, 3)
    Cy:  torch.Tensor   # (3, 6)
```

The `build_G_matrix` function must be updated to concatenate `Ax`, `Bw`, `Bu` into `A_combined = torch.cat([Ax, Bw, Bu], dim=1)` before returning the `GMatrix`. The module-level `G` singleton and `__main__` verification block are otherwise unchanged.

---

## `lpv_lfr_baseline/core/lfr_forward.py`

### Problems
1. No `lfr_xdot` fast path. RK4 substeps 2, 3, 4 only need `xdot` but call the full
   `lfr_forward` which computes `z`, `w`, and `y` — three wasted tensor constructions
   per substep per time step.
2. No `@torch.compile` decoration. The function runs eagerly, launching each elementwise
   op as a separate CUDA kernel. On Windows without MSVC, `inductor` is unavailable but
   `cudagraphs` works and eliminates kernel-launch overhead.

### Changes

**Add `lfr_xdot`** — identical physics to `lfr_forward` but returns only `xdot`.
Stops computation after step 5; skips `z`, `w`, and `y` assembly.

```python
def lfr_xdot(
    x, u, Y, G, K, C, mh, alpha, beta, gamma, N0, N1, N2
) -> torch.Tensor:
    """Fast path: returns only xdot. Used for RK4 substeps 2, 3, 4."""
    fnet = -(x[:, :3] @ K.T) - (x[:, 3:] @ C.T) + u
    dY   = mh * (alpha * gamma - beta ** 2
                 + 2 * beta * mh * Y
                 + mh * (alpha - mh) * Y ** 2)
    Y_r  = Y.unsqueeze(0)
    a    = (N0 @ fnet.T + Y_r * (N1 @ fnet.T + Y_r * (N2 @ fnet.T))).T / dY[:, None]
    z    = torch.cat([a, Y[:, None] * a], dim=-1)
    w    = Y[:, None] * z
    
    # Fused state update
    combined_input = torch.cat([x, w, u], dim=-1)
    return combined_input @ G.A_combined.T
```

**Update `lfr_forward` to use the fused update** (but do NOT compile it here):

```python
def lfr_forward(x, u, Y, G, K, C, mh, alpha, beta, gamma, N0, N1, N2):
    # ... existing body up to Step 5 ...
    
    # Fused state update replacing (x @ Ax.T) + (w @ Bw.T) + ...
    combined_input = torch.cat([x, w, u], dim=-1)
    xdot = combined_input @ G.A_combined.T
    
    y = x @ G.Cy.T
    return xdot, z, w, y
```

**Important Note on Compilation:**
Do NOT apply `@torch.compile` to `lfr_forward` or `lfr_xdot` in this file. They must remain pure Python functions. We will compile the entire `rk4_step` loop at once in `lfr_simulate.py` instead. This reduces the 4 CUDA graph launches per time step down to just 1.

The `__main__` verification block calls `_lfr_forward_impl` directly so it remains
unaffected by compile.

---

## `lpv_lfr_baseline/core/lfr_simulate.py`

### Problems
1. `_COMPILE_LFR_FORWARD` flag with conditional `torch.compile` call. Violates the
   "compile unconditionally" rule. Dead code path when flag is False.
2. `_fwd` closure created inside `rk4_step` on every call. Allocates a new function
   object each invocation.
3. `rk4_step` substeps 2, 3, 4 call `lfr_forward` and discard `z`, `w`, `y`.
4. `simulate()` contains broken truncated BPTT: it detaches state inside the loop of a
   single `simulate()` call, so only the last segment contributes to backward. Windowed
   BPTT must be implemented as an outer loop in the training script, not inside
   `simulate()`.
5. Closure inside `if bptt_mode == "checkpoint"` block: `_step` is created on every loop
   iteration inside `simulate()` when checkpoint mode is active.

### Changes

**Remove entirely:**
- `_COMPILE_LFR_FORWARD` constant (line 31)
- `if _COMPILE_LFR_FORWARD: ... else: ...` block (lines 33-36)
- `_fwd` closure inside `rk4_step` (lines 80-81)
- Truncated BPTT from `simulate()`: remove the `bptt_mode == "truncated"` branch and
  `segment_len` parameter entirely
- `_step` closure inside the checkpoint branch of `simulate()` loop (lines 148-154)

**Update imports:** change
```python
from lpv_lfr_baseline.core.lfr_forward import lfr_forward as _lfr_forward_raw
```
to
```python
from lpv_lfr_baseline.core.lfr_forward import lfr_forward, lfr_xdot
```

**`rk4_step` after changes (compiled as a whole):**

We move `COMPILE_BACKEND` here to `lfr_simulate.py` and compile the entire RK4 sequence.

```python
COMPILE_BACKEND = 'cudagraphs'   # change to 'inductor' on server when MSVC is available

@torch.compile(backend=COMPILE_BACKEND, mode='reduce-overhead', fullgraph=True)
def rk4_step(x, u_logical, G, K, C, mh, alpha, beta, gamma, N0, N1, N2, ts,
             Y_override=None):
    Y = x[:, 2] if Y_override is None else Y_override

    k1, z, w, y = lfr_forward(x, u_logical, Y, G, K, C, mh, alpha, beta, gamma, N0, N1, N2)

    x2 = x + (ts / 2) * k1
    k2 = lfr_xdot(x2, u_logical, x2[:, 2] if Y_override is None else Y_override,
                  G, K, C, mh, alpha, beta, gamma, N0, N1, N2)

    x3 = x + (ts / 2) * k2
    k3 = lfr_xdot(x3, u_logical, x3[:, 2] if Y_override is None else Y_override,
                  G, K, C, mh, alpha, beta, gamma, N0, N1, N2)

    x4 = x + ts * k3
    k4 = lfr_xdot(x4, u_logical, x4[:, 2] if Y_override is None else Y_override,
                  G, K, C, mh, alpha, beta, gamma, N0, N1, N2)

    x_next = x + (ts / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_next, z, w, y
```

**`simulate()` after changes** — `bptt_mode` becomes `Literal["full", "checkpoint"]`
only. Remove `segment_len` parameter. For checkpoint mode, define `_rk4_step` as a
module-level helper (not a per-loop closure):

```python
# module level, defined once
def _rk4_checkpoint(x, u_logical, G, K, C, mh, alpha, beta, gamma, N0, N1, N2, ts):
    return rk4_step(x, u_logical, G, K, C, mh, alpha, beta, gamma, N0, N1, N2, ts)
```

Inside `simulate()` the checkpoint branch becomes:
```python
x_next, z_k, w_k, y_k = grad_checkpoint(
    _rk4_checkpoint, x, u_logical,
    G, K, C, mh, alpha, beta, gamma, N0, N1, N2, ts,
    use_reentrant=False,
)
```

No closure is created per iteration.

**`simulate_frozen`** — no changes required (already correct structure, no compile flags).

---

## `lpv_lfr_baseline/scripts/precompute.py` (NEW FILE)

### Purpose
All computation that does not depend on trainable parameters and is expensive to rerun:
sigma normalization constants, RMSE baseline, and trajectory tensors. Writes a single
`.pt` cache file. If the cache exists and is valid, returns it immediately.

### What it computes

| Item | Shape | Description |
|------|-------|-------------|
| `trajs` | list of dicts | Per-trajectory: `u` `(1,T,3)`, `q1` `(T,3)`, `state_traj` `(T,6)`, metadata |
| `sigma` | dict `traj_id -> (3,)` | Per-trajectory, per-channel output std for normalization |
| `rmse_baseline` | dict `traj_id -> float` | Per-trajectory RMSE of zero-model, meters |
| `rmse_baseline_normalized` | `float` | Group-balanced normalized RMSE baseline (training reference) |
| `segment_len` | `int` | Chosen window length from segment diagnostic |

`state_traj` is built from stage-position data: convert to logical coordinates via `P`,
then estimate velocities by finite differences (central differences interior, forward/
backward at endpoints). Shape `(T, 6)` = `[q_logical; qdot_logical]`.

`sigma` is computed per trajectory, per channel: `q1[:, c].std()` over the full
trajectory. Channels with near-zero variance (controller-suppressed) are clamped:
`sigma[c] = max(std, 1e-4)`. No masking -- all three channels contribute to the loss,
just with their own normalization.

### Caching pattern

```python
CACHE_PATH = Path("simulations/param_recovery/precomputed.pt")

def precompute(force: bool = False) -> dict:
    if not force and CACHE_PATH.exists():
        return torch.load(CACHE_PATH, weights_only=True)
    data = _compute()
    torch.save(data, CACHE_PATH)
    return data
```

`force=False` is the default — calling `precompute()` from the training loop is free
after the first run.

### What it does NOT compute
- `G`, `alpha`, `beta`, `gamma`, `N0`, `N1`, `N2`: depend on trainable parameters.
  The training loop rebuilds these each forward pass from current `log_params`.
- Anything depending on `device` or GPU state: precompute on CPU, move in training loop.

---

## `lpv_lfr_baseline/scripts/train_param_recovery.py` (REWRITE)

### Remove entirely
- `_SimWrapper` and `DataParallel` wrapper
- `PROFILE`, `PROFILE_WAIT`, `PROFILE_WARMUP`, `PROFILE_ACTIVE` flags
- `_COMPILE_LFR_FORWARD` flag
- `CHANNEL_MASKS` and all mask-based loss weighting (`batch_weights`, `n_active_per_traj`)
- Segment length diagnostic (moved to `scripts/segment_diag.py`)
- All `bptt_mode` config mixing -- training always uses windowed BPTT

### Required functionality

The implementation details below are not prescriptive -- use whatever structure produces
the cleanest code. What must be preserved is the behavior.

**Data and configuration:**
- Trajectory set is defined by a list of specs (id, group, file). Active set is
  configurable at the top of the file. Groups determine how segments are balanced.
- Module-level constants for training hyperparameters: segment window size, epochs, LR,
  eval interval, log interval, checkpoint interval, split regularization weight, seed.

**Training loop:**
- Each epoch draws a balanced batch of segments across trajectory groups (equal group
  representation regardless of how many trajectories are in each group). Sampling is
  seeded and reproducible.
- Windowed BPTT: each segment is simulated in windows of W steps. Backward is called
  per window; state is detached between windows. Optimizer steps once per segment batch.
- Loss: sigma-normalized MSE across all three output channels plus split regularization.
- LR scheduler reduces on plateau, driven by full-trajectory eval RMSE.
- Periodic checkpointing: saves enough state to resume training.
- Per-epoch console output: train loss, grad norm, wall time, latest eval RMSE.

**Evaluation:**
- Full-trajectory eval (no grad) runs periodically: simulate each full trajectory from
  its true initial state, compute per-trajectory and group-balanced RMSE in meters.
- If a second GPU is available, eval runs asynchronously on that GPU while training
  continues on the first. If not, it runs synchronously.
- Best-epoch tracking: the parameter snapshot with the lowest full-trajectory eval RMSE
  is retained throughout training.

**Post-training (required, must not be dropped):**
- Restore best-epoch parameters before final evaluation.
- Full per-trajectory eval: report per-channel RMSE (X1, X2, Y) for each trajectory
  and an overall group-balanced RMSE.
- Parameter recovery table: learned vs. true values for all physical parameters.
- Save a result file containing: learned parameters, true parameters, initial parameters,
  best-epoch snapshot, sigma, RMSE baseline, eval results, training history, run config.

### Loss function
No channel masks. Loss is sigma-normalized MSE averaged over all three output channels,
all time steps, all segments in the batch:

```python
sigma_traj = precomputed['sigma'][traj_id]          # (3,) per-trajectory, per-channel
err = (Y_pred - q1_ref) / sigma_traj.unsqueeze(0)   # (B, T, 3) -- sigma broadcast over T
mse_loss = err.pow(2).mean()
split_reg = block.split_loss() * SPLIT_REG_WEIGHT
loss = mse_loss + split_reg
```

`sigma` for zero-variance channels is set to `1.0` in `precompute.py` (not masked out,
just normalized to unit variance -- their contribution to loss is naturally small).

### Structure

```
precompute()          # load or compute and cache all fixed data

ParameterizedLFRBlock # nn.Module with log_params as nn.Parameter
                      # build_params() rebuilds K, C, G, poly constants each call

training loop:        # lean outer loop
    for epoch in range(N_epochs):
        torch.compiler.cudagraph_mark_step_begin()
        model.train()

        for seg_idx, (x0_seg, u_seg, q1_seg) in enumerate(segments):
            # windowed BPTT over this segment
            x = x0_seg.to(device).detach()
            seg_loss = 0.0

            for t in range(0, T, W):
                params = model.build_params()
                G = build_G_matrix(...)
                result = simulate(
                    x, u_seg[:, t:t+W].to(device),
                    G, ..., bptt_mode='full'
                )
                loss = normalized_rmse(result.Y[..., 0], q1_seg[:, t:t+W])
                loss.backward()
                seg_loss += loss.detach()
                x = result.X[:, -1, :].detach()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # post-epoch: log, checkpoint, trigger async eval
```

### Async evaluation
One background thread owns the eval GPU (GPU 1 if available, else same as train GPU).
The training loop posts a snapshot (detached parameter dict) to `snap_queue` at the end
of each epoch. The eval thread pops the snapshot, runs `simulate()` on validation
trajectories with `torch.no_grad()`, computes RMSE, and posts the result to
`result_queue`. The training loop drains `result_queue` at the start of the next epoch
log step — non-blocking, with a `get(timeout=0)` that skips if no result is ready.

The eval thread never touches the training graph. It builds its own `G` and poly
constants from the snapshot parameters.

### Window size W
A module-level constant, not a command-line argument. Start with `W = 50`. Reduce if
GPU OOM during backward; increase if gradients are too noisy (visible as oscillating
loss). Determined empirically, not by tuning during a run.

### Float precision toggle
One module-level constant controls dtype throughout:

```python
DTYPE = torch.float64   # switch to torch.float32 to test; float64 is the default
```

Every tensor creation, `.to(device)` call, and `precompute()` load uses `DTYPE`.
The constant lives at the top of `train_param_recovery.py` and is passed into
`precompute()` so cached tensors are recast if the dtype changes.

Keep `float64` by default. The gantry ODE is stiff and float32's ~7 significant digits
corrupt gradients. Only switch to float32 for profiling comparisons or if a future
profiling pass proves float ops (not launch overhead) are the bottleneck. See
`docs/pytorch-optimization-guidelines.md` -- Float precision section.

### What is NOT in this file
- Segment length search -- moved to `scripts/segment_diag.py`
- Profiler integration -- moved to `scripts/profile_run.py`
- Any conditional compile path
- DataParallel or model.to(device) wrappers that scatter inputs

---

## `lpv_lfr_baseline/scripts/segment_diag.py` (NEW FILE)

Extracted from the current `train_param_recovery.py` segment-length diagnostic block.

**Purpose:** Run once to choose the optimal window length `W` for windowed BPTT.
Tests candidate lengths `[0.1, 0.2, 0.4, 0.6]` seconds of data, scores each by whether
the true parameters produce lower loss than detuned alternatives on held-out segments.
Saves result to `simulations/param_recovery/segment_diag_<traj_tag>_v<N>.pt`.

**What moves here from train:**
- `SEGMENT_DIAG_CANDIDATES_S`, `SEGMENT_DIAG_SEGMENTS_PER_GROUP`, `SEGMENT_DIAG_VERSION`
- `_get_or_run_segment_length_diagnostic`
- `_choose_segment_len_from_diag`
- `_print_segment_diag_summary`

**Interface:** `run_segment_diag(traj_specs, save_dir) -> int` returns chosen segment
length. Loads from cache if already run for this trajectory set.

`precompute.py` calls this and stores the result as `segment_len` in the cache.

---

## `lpv_lfr_baseline/scripts/profile_run.py` (NEW FILE)

**Purpose:** Run a short profiled training pass (warmup + active epochs) and export a
Chrome trace. Completely separate from the training script -- no `PROFILE` flag, no
`n_steps` cap baked into train.

**What it does:**
```python
with torch.profiler.profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=2, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(save_dir),
    record_shapes=True,
    with_stack=True,
) as prof:
    for epoch in range(wait + warmup + active):
        _run_one_epoch(model, data, optimizer)
        prof.step()

prof.export_chrome_trace(os.path.join(save_dir, 'profile_trace.json'))
```

`_run_one_epoch` is the same windowed BPTT loop as in `train_param_recovery.py`,
extracted as a shared function. No step cap -- profile runs on the real data size so
timing is representative.

---

---

## Cross-cutting constraints

These apply to every file touched in this overhaul. See
`docs/pytorch-optimization-guidelines.md` for full rationale.

### No list accumulation

Never collect tensors in a Python list and call `torch.stack` or `torch.cat` at the
end. Pre-allocate the output tensor once and write into it by index:

```python
# WRONG
ys = []
for t in range(T):
    ys.append(step(x))
Y = torch.stack(ys)   # T allocations + 1 final

# RIGHT
Y = x.new_empty(T, batch, 3)
for t in range(T):
    Y[t] = step(x)    # single pre-allocated buffer
```

`simulate()` already does this correctly. Do not regress it.

### For loops: keep only what has sequential dependency

| Loop | Status | Reason |
|------|--------|--------|
| Time loop in `simulate()` | Keep -- unavoidable | Each step depends on previous state; cannot vectorize over time |
| RK4 substeps (k1..k4) | Keep -- unavoidable | Sequential dependency chain |
| Segment loop in training | Keep -- unavoidable | Windowed BPTT requires sequential state carry |
| Epoch loop | Keep -- unavoidable | Sequential optimization |
| Any loop over batch elements | Remove | Replace with `vmap` or existing batch dimension |

No new Python loops over batch elements are permitted. If a function processes one
trajectory and the caller needs N trajectories, use `torch.func.vmap`, not a for loop.

### `new_empty` / `new_zeros`, not `torch.empty` / `torch.zeros`

Inside compiled regions and `vmap`-transformed functions, the free-function forms do
not inherit the batched tensor context. Always use the method form:

```python
buf = x.new_empty(T, batch, 6)   # inherits device, dtype from x
```

---

## Files that do NOT change

| File | Reason |
|------|--------|
| `core/physics.py` | Physical constants only; no training logic |
| `core/lfr_matrices.py` `build_G_matrix()` | Function body unchanged; only `GMatrix` class definition changes |
| All `__main__` verification blocks | Tests run against `_impl` functions directly; compile wrappers are transparent |
