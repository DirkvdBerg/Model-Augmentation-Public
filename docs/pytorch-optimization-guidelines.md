# PyTorch Optimization Guidelines

Reference for all new code in this codebase. The two governing principles are:

1. **Minimal code.** Write the smallest correct implementation. No flags, no fallbacks, no conditional compilation paths.
2. **Fast functional GPU code.** Every hot-path function must be written to compile and fuse cleanly. Correctness first, then prove it compiles.

---

## Minimal code rule

Write the smallest correct implementation. Cut code that does not serve the current
requirement -- not functions, but unnecessary constructs:

- **No feature flags.** `_COMPILE_LFR_FORWARD = True/False`, `PROFILE = True`, `bptt_mode` config switches -- these add dead code paths. Pick one correct behavior and implement only that.
- **No fallback paths** -- with one explicit exception: the `torch.compile` backend. `inductor` requires MSVC (unavailable on the development PC); `cudagraphs` works everywhere. One module-level constant selects the backend:
  ```python
  COMPILE_BACKEND = 'cudagraphs'   # 'inductor' on server when MSVC is available
  lfr_forward = torch.compile(_lfr_forward_impl, backend=COMPILE_BACKEND, fullgraph=True)
  ```
  This is the only conditional in the codebase that routes between two implementations. All other fallbacks are removed.
- **No conditional compilation.** `if _COMPILE: fn = torch.compile(fn)` is two implementations of the same thing. Write one.
- **No backwards-compatibility shims.** Do not keep old code commented out or behind a flag "just in case".
- **No redundant wrappers.** `_SimWrapper` exists only to satisfy `DataParallel`'s interface. Remove both.

Functions are good: they name concepts, hide complexity, and enable reuse. The problem
in the old code is not functions -- it is flags and conditional branches that multiply
the number of active code paths without adding capability.

---

## File structure

Concerns must be separated into distinct files. The new structure:

| File | Responsibility |
|---|---|
| `core/physics.py` | Physical constants: P, ts, M1, M2, K, C. Truly constant, no nn.Parameters. |
| `core/lfr_matrices.py` | `GMatrix` (NamedTuple), `build_G_matrix()`. Rebuild inside `forward()` when params are trainable. |
| `core/lfr_forward.py` | `lfr_forward` and `lfr_xdot` -- compiled at definition time with `@torch.compile`. |
| `core/lfr_simulate.py` | `rk4_step`, `simulate`. No compile flags. Clean Python loop. |
| `scripts/precompute.py` | All expensive one-time setup: sigma, RMSE baseline, trajectory tensors, segment sampling setup. Returns cached result if already on disk. |
| `scripts/train_param_recovery.py` | Lean training loop only: load precomputed data, build model, iterate, log, checkpoint. |

**Nothing is computed twice.** If `precompute.py` has already run and written a cache to disk, subsequent calls return the cache immediately without recomputation.

---

## torch.compile

### What it does

`torch.compile` is a two-layer system:

- **Dynamo** (frontend): hooks into CPython's frame evaluator, traces bytecode symbolically, extracts FX graphs of PyTorch ops, and attaches runtime guards (shape/dtype/device checks). On a cache hit the guard check costs microseconds; on a miss it recompiles.
- **Inductor** (backend): takes the FX graph and generates fused Triton kernels (CUDA) or C++ with OpenMP (CPU). The primary optimization is **operator fusion**: adjacent pointwise ops are merged into one kernel -- one memory read, one write, instead of N reads and N writes.

### Compile the step body, not the loop

This is the most important rule for sequential simulation code.

`torch.compile` applied to a function containing a Python `for` loop **unrolls the loop** into one FX graph. For a 4000-step simulation, this produces a graph with 4000 copies of every operation inside the loop -- compile time scales as O(T) and becomes prohibitive.

```python
# WRONG: compiles the T-step loop, Dynamo unrolls it
simulate = torch.compile(_simulate_impl)

# RIGHT: compile the per-step body; the Python loop retains its structure
@torch.compile(fullgraph=True, mode='reduce-overhead')
def lfr_forward(...): ...

# The Python loop calls the compiled body T times
for t in range(T):
    x = rk4_step(x, u[t], ...)  # compiled body, fast
```

### No conditional compile flags

Never write:

```python
_COMPILE = True
if _COMPILE:
    fn = torch.compile(_fn_impl)
else:
    fn = _fn_impl
```

Compile unconditionally at module definition time. The guard check on a compiled function cache hit is essentially free (microseconds). There is no meaningful overhead to using the compiled path when it is already warm.

```python
# RIGHT: compile once at definition
@torch.compile(fullgraph=True, mode='reduce-overhead')
def lfr_forward(...):
    ...
```

If you need the raw implementation for unit tests, define the implementation function with a private name and compile the public name:

```python
def _lfr_forward_impl(...):
    ...

lfr_forward = torch.compile(_lfr_forward_impl, fullgraph=True, mode='reduce-overhead')
```

### Backends

| Backend | When to use |
|---|---|
| `inductor` (default) | Always. Generates Triton/C++ with kernel fusion. Requires MSVC on Windows (cl.exe). |
| `cudagraphs` | When inductor is unavailable and the computation is static (fixed shapes). No kernel fusion; eliminates CPU dispatch overhead by replaying a recorded GPU op sequence. No MSVC required. |
| `aot_eager` | Debugging only. Traces the graph but executes eagerly -- confirms graph correctness without Inductor's transformations. |

On Windows without MSVC, `backend='cudagraphs'` is the best available option. Enable with:

```python
@torch.compile(backend='cudagraphs', fullgraph=True)
def lfr_forward(...): ...
```

### Modes

| Mode | Use case |
|---|---|
| `'reduce-overhead'` | Sequential simulation with many small kernels. Enables CUDA Graphs internally. Best when the bottleneck is CPU kernel-launch overhead, not GPU compute. |
| `'default'` | General training. Balanced compile time vs runtime. |
| `'max-autotune'` | Long-running inference deployments only. Very slow compile, best runtime. |

For this codebase: use `mode='reduce-overhead'` on `lfr_forward` and `rk4_step`. The simulation loop dispatches many small kernels; CUDA Graphs will eliminate the dispatch overhead.

### fullgraph=True

Always set `fullgraph=True` on functions where you intend to compile the entire body. This raises an exception (instead of silently inserting a graph break) if Dynamo cannot capture the function as a single graph. Use it during development to enforce compilability.

Graph breaks prevent operator fusion across the break boundary. A function with one graph break becomes two compiled sub-graphs with Python code running between them.

### Dynamic shapes

Dynamo specializes on shapes at first compile and recompiles on shape changes. For functions called with varying batch sizes or sequence lengths, declare the varying dimension:

```python
torch._dynamo.mark_dynamic(x, dim=0)  # batch dimension is dynamic
```

Or compile with `dynamic=True`:

```python
torch.compile(fn, dynamic=True)
```

### CUDA Graphs and training

`mode='reduce-overhead'` uses CUDAGraph Trees internally. When multiple compiled callables are invoked in a training loop, call this at the start of each iteration:

```python
torch.compiler.cudagraph_mark_step_begin()
```

This signals that previous iteration tensors can be freed and new buffers prepared. Without it, CUDA graphs may reference freed memory.

---

## Memory and tensor patterns

### Pre-allocate; never accumulate in lists

Python lists inside a loop that later become tensors are always wrong:

```python
# WRONG
results = []
for t in range(T):
    results.append(step(x))
output = torch.stack(results)  # allocates T times + one final alloc

# RIGHT
output = x.new_empty(T, *x.shape)
for t in range(T):
    output[t] = step(x)
```

`torch.stack` and `torch.cat` inside loops allocate new memory on every call. Pre-allocate the output tensor once and write into it with index assignment.

### Use new_empty / new_zeros, not torch.empty / torch.zeros

Inside `vmap`-transformed functions and inside compiled regions, `torch.zeros(shape)` does not inherit the batched tensor context. Always use the method form:

```python
# Wrong inside vmap or compiled functions
buf = torch.zeros(batch, 6, device=device, dtype=dtype)

# Right
buf = x.new_zeros(batch, 6)   # inherits device, dtype, layout from x
buf = x.new_empty(batch, 6)   # same, uninitialized
```

### One large tensor over multiple smaller tensors

Prefer one `(T, B, N)` tensor over T tensors of shape `(B, N)`. A single large tensor:
- Can be sliced with zero-copy views
- Is stored contiguously in memory for efficient sequential access
- Enables bulk operations without Python overhead

### Contiguous memory

Slicing, transposing, and permuting return non-contiguous views. Non-contiguous tensors silently insert a copy when passed to kernels that require contiguous input. If a tensor will be passed to many ops, make it contiguous once:

```python
u_seq_logical = (u_seq_stage @ P.T).permute(1, 0, 2).contiguous()  # one copy here
# ...
u_t = u_seq_logical[t]  # contiguous slice, zero-copy
```

For sequential simulation: time-first layout `(T, B, N)` gives contiguous reads `data[t]` in the time loop.

### In-place ops and autograd

In-place ops (`add_`, `mul_`, `copy_`) avoid allocating a new tensor but have autograd constraints:

- Modifying a tensor in-place that is saved for backward by another op will raise a runtime error.
- Safe use: updating pre-allocated output buffers (`X_t[t] = x_next`), zeroing optimizer state.
- `param.grad = None` is faster than `param.grad.zero_()` -- the former frees memory, the latter zeroes in place.

### .detach() vs .clone()

```python
x.detach()          # zero-cost: same storage, removed from autograd graph
x.clone()           # allocates new memory and copies data
x.detach().clone()  # canonical: independent copy, not tracked by autograd
```

Never use `.clone()` just to detach from autograd. Use `.detach()` for that.

---

## Hot-path code rules

### No closures in loops

Closures created inside a loop allocate a new function object on every iteration:

```python
# WRONG: _fwd closure is created on every rk4_step call
def rk4_step(x, u, G, ...):
    def _fwd(s):
        return lfr_forward(s, u, s[:, 2], G, ...)
    k1 = _fwd(x)
    ...

# RIGHT: inline directly or pass as a top-level function
def rk4_step(x, u, G, ...):
    k1, z, w, y = lfr_forward(x, u, x[:, 2], G, ...)
    ...
```

### Skip unused outputs

`lfr_forward` returns `(xdot, z, w, y)`. RK4 substeps 2, 3, and 4 only use `xdot`. Computing z, w, and y for those substeps is pure waste -- three unnecessary tensor constructions per time step.

Write a separate `lfr_xdot` function that returns only `xdot`. Use it for the intermediate RK4 substeps:

```python
# substep 1: need z, w, y for output recording
xdot, z, w, y = lfr_forward(x, u, x[:, 2], ...)

# substeps 2, 3, 4: need only xdot
xdot2 = lfr_xdot(x + ts_2 * xdot, u, ...)
xdot3 = lfr_xdot(x + ts_2 * xdot2, u, ...)
xdot4 = lfr_xdot(x + ts * xdot3, u, ...)
```

### No .item(), .cpu(), .numpy() in hot paths

These force a CPU-GPU synchronization, terminate the current CUDA kernel stream, and prevent operator fusion across the call site. They also break CUDA graph capture.

```python
# WRONG: .item() inside training loop
if loss.item() < threshold: ...

# RIGHT: keep as tensor; compare in GPU
converged = (loss < threshold)   # tensor bool, stays on GPU
```

For logging: call `.item()` once per log interval, outside the loss computation.

### No Python conditionals on tensor values

```python
# WRONG: Dynamo must execute this in Python, causes graph break
if grad_norm.item() > 1.0:
    grads = grads / grad_norm

# RIGHT: stays in graph, no break
clip_coef = torch.clamp(1.0 / grad_norm, max=1.0)
grads = grads * clip_coef
```

Use `torch.where`, `torch.clamp`, `torch.cond` instead of Python `if` on tensor values.

### Return tensors, not Python objects, from compiled functions

NamedTuples of tensors are fine. Python dataclasses with mixed Python/tensor fields may cause graph breaks. `dict` of tensors is fine.

```python
# WRONG: dataclass causes jit.script failure and may cause torch.compile graph breaks
@dataclass
class GMatrix:
    Ax: torch.Tensor
    ...

# RIGHT: NamedTuple is fully compatible with both jit.script and torch.compile
class GMatrix(NamedTuple):
    Ax: torch.Tensor
    ...
```

---

## Precomputation

Everything that does not depend on trainable parameters should be computed once and reused. During parameter recovery training, the trainable parameters change every step, so `G`, `alpha`, `beta`, `gamma`, `N0`, `N1`, `N2` must be rebuilt every forward pass. But they can be built efficiently from the current parameter vector without redundant work.

Constants that are truly fixed (not trainable):
- `P`, `ts`: coordinate transform and timestep
- `M1`, `M2`: mass matrix coefficients (if masses are fixed)

Constants that depend on trainable parameters (must rebuild in forward):
- `K`, `C`: depend on stiffness and damping params
- `G`: depends on K, C, M1, M2 via `build_G_matrix`
- `alpha`, `beta`, `gamma`, `N0`, `N1`, `N2`: depend on mass params via `build_poly_constants`

For training data preprocessing, sigma normalization, RMSE baseline, and trajectory tensors: all are expensive to compute and fixed across training. These go in `precompute.py`, which writes a cache to disk and loads from cache on subsequent runs.

---

## Backpropagation through time (BPTT)

### The broken pattern (do not use)

```python
# WRONG: simulate all T steps, one backward
Y_pred = simulate(x0, u_all_T_steps, ...)   # T=4000 step graph
loss = mse(Y_pred, q1)
loss.backward()   # traverses 4000-node graph; slow and memory-heavy
```

This unrolls the full T-step computation graph before backprop. The backward pass cost scales linearly with T.

### The correct pattern: windowed BPTT

```python
# RIGHT: simulate W steps at a time, backward through each window
x = x0.detach()
total_loss = 0.0

for t in range(0, T, W):
    u_win  = u[:, t:t+W]
    q1_win = q1[:, t:t+W]

    result = simulate(x, u_win, ..., bptt_mode='full')   # W steps only
    loss   = mse(result.Y, q1_win)
    loss.backward()                                        # traverses W-node graph only
    total_loss += loss.detach()

    x = result.X[:, -1, :].detach()   # carry state forward, detach from graph

optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

Window size W is a tradeoff: smaller W means shorter backward graphs (faster) but less gradient information per step (higher variance). With good initial parameters from analytical identification, W=20-50 is sufficient.

---

## vmap and functional transforms

`torch.func.vmap` vectorizes a function over a batch dimension without a Python loop:

```python
# Instead of looping over trajectories
for i in range(N_trajs):
    Y_i = simulate_one(params, x0[i], u[i])

# vmap compiles this into one parallel kernel
simulate_batch = torch.func.vmap(simulate_one, in_dims=(None, 0, 0))
Y_batch = simulate_batch(params, x0, u)   # (N_trajs, T, 3)
```

`vmap` constraints:
- No `.item()`, `.numpy()`, `.cpu()` inside the vmapped function
- No Python conditionals on tensor values
- No `torch.zeros(...)` -- use `tensor.new_zeros(...)` instead
- No `nn.Module` directly -- use `torch.func.functional_call` with a parameter dict

For sensitivity analysis (Jacobian of outputs w.r.t. parameters):

```python
dY_dparams = torch.func.jacrev(simulate_one, argnums=0)(params, x0, u)
```

---

## Optimizer settings

```python
# zero_grad with set_to_none=True frees gradient memory instead of zeroing it
optimizer.zero_grad(set_to_none=True)
```

---

## Float precision

Keep `float64` throughout. The gantry simulation involves stiff ODEs where float32's ~7 significant digits introduce numerical errors that corrupt gradients. Do not introduce mixed precision without first confirming via profiling that floating-point ops (not loop structure or kernel launch overhead) are the bottleneck.

TF32 (Ampere) does not apply to float64 ops. `torch.backends.cuda.matmul.allow_tf32` has no effect on this codebase.

---

## Profiling checklist

Before optimizing, profile to identify the bottleneck type:

| Symptom | Bottleneck | Fix |
|---|---|---|
| Many small CUDA kernel launches, each taking microseconds | CPU launch overhead | CUDA Graphs (`mode='reduce-overhead'`) |
| High GPU utilization, low FLOP utilization | Memory bandwidth | Operator fusion (`torch.compile` with inductor) |
| High GPU utilization, high FLOP utilization | Compute | `max-autotune` mode, lower precision (not applicable here) |
| GPU idle, CPU busy | Python / host overhead | `torch.jit.script`, eliminate `.item()` calls |

Use `torch.utils.benchmark.Timer` for microbenchmarks, not `time.time()`. It handles CUDA synchronization and warmup automatically.
