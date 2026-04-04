# LPV-LFR Baseline — Known Gaps and Open Issues

Seven issues identified during code review that are not bugs in the current
validation-only use, but will matter before training begins.

Priority levels:
- **CRITICAL** — will crash or silently produce wrong results in the training loop
- **WARN** — risk of silent wrong results in edge cases or future changes
- **NOTE** — missing test, documentation gap, maintenance trap

---

## Gap 1 — GPU training will crash on the first forward call [CRITICAL — FIXED 2026-04-04]

**What**: Physical parameters (`M0`, `M1`, `M2`, `K`, `C`, `P`, `ts`) are stored as plain
Python attributes in `LFRBaselineBlock.__init__`:

```python
self._M0 = M0
self._M1 = M1
# etc.
```

**Why it breaks**: PyTorch's `.to(device)` and `.cuda()` methods only move
`nn.Parameter`s and tensors registered via `self.register_buffer()`. Plain attributes
are ignored. So when the training framework puts the model on GPU:

```python
block.cuda()
# self._M0 is still on CPU
# z_in arrives on GPU
```

The very first forward call crashes at `lfr_forward.py` Step 1:

```python
Y_e = Y[:, None, None]                   # GPU tensor (from z_in)
M_Y = M0.unsqueeze(0) + ...* Y_e         # CPU + GPU → RuntimeError
```

Error: `RuntimeError: Expected all tensors to be on the same device`.

**What the fix looks like** (two-line change per parameter in `lfr_block.py`):

```python
# Before (current):
self._M0 = M0

# After:
self.register_buffer('_M0', M0)
```

`register_buffer` tells PyTorch this tensor is part of the module state and should
move with `.to(device)`. No other file needs changing.

**Why it hasn't been hit yet**: all validation runs on CPU only. The first GPU
training attempt will hit this immediately.

---

## Gap 2 — Augmentation can only add a correction, not modify the scheduling loop [WARN]

**What**: The resolve-and-retain forward pass collapses `M(Y)^{-1}` analytically before
the LFR signals `z` and `w` are produced. From Jan's Interconnect's point of view, the
block is a black box: input `(x, u)` → output `(x_next, z_lfr, w_lfr)`. The scheduling
relationship `w = Y * I6 * z` is enforced internally, not as a live wired loop in the
Interconnect.

**Consequence**: The augmentation network can only attach an *additive correction* to
`x_next` (i.e. `x_next_corrected = x_next_baseline + delta`). It cannot intercept the
signal between `z` and `w` to modify how `M(Y)` is applied. This is called
"Architecture 1" in `lfr-dzw-zero-alternative.md`.

Architecture 2 (where the augmentation sits inside the Δ(Y) loop and can correct the
inertia itself) would require `Dzw = 0` — the alternative LFR realization using
`ρ = [1/det(M), Y/det(M), Y²/det(M)]` as scheduling variables.

**Where this is documented**: `lpv_lfr_baseline/lfr-dzw-zero-alternative.md` contains
a full derivation of the alternative, but the question "which architecture does the
supervisor intend?" has not been answered. Roland Tóth should be asked explicitly.

**Why it hasn't mattered yet**: training has not started. The distinction only matters
when designing how the augmentation network connects to the baseline.

---

## Gap 3 — Row-vector convention silently depends on K and C being symmetric [WARN]

**What**: `lfr_forward.py` uses batched row-vector matrix multiplication:

```python
fnet = -(x[:, :3] @ K) - (x[:, 3:] @ C) + u
```

In column-vector notation, `M(Y) q_ddot = -K q - C q_dot + u` means
`fnet = -K q - C q_dot + u`. In row-vector notation: `fnet = -q @ K.T - qdot @ C.T + u`.

This is only equivalent to `-q @ K - qdot @ C + u` (what the code computes) when
`K = K.T` and `C = C.T` — i.e. when both matrices are symmetric.

Currently both are symmetric:
- `K` is diagonal (only `K[1,1] != 0`): obviously symmetric.
- `C` has `C[0,1] = C[1,0] = (cg1-cg2)*Lb/2`: symmetric.

**The risk**: if a non-symmetric term is ever added to `C` (e.g. gyroscopic or
velocity-dependent asymmetric damping from the augmentation, or a parameter
correction), the formula silently computes the wrong `fnet`. There is no assertion
or comment that warns about this.

**Fix**: add a comment at the definition of `fnet` stating explicitly that this is
only correct for symmetric `K` and `C`, and add a one-line assertion in the
`__main__` block: `assert torch.allclose(K, K.T) and torch.allclose(C, C.T)`.

---

## Gap 4 — `simulate_frozen` is defined in `validate_lfr.py`, not in `lfr_simulate.py` [NOTE]

**What**: `simulate_frozen` lives in `validate_lfr.py` (line 70), not alongside
`simulate()` in `lfr_simulate.py`. Any code outside `validate_lfr.py` that needs
the frozen baseline must either copy the function or import it from the wrong module.

**Why it matters**: `lfr_block.py` and `test_jan_compat.py` both import from
`lfr_simulate`. If a future test wants to run frozen vs LPV inside the Interconnect,
the function is not available from the natural import path.

**Fix**: move `simulate_frozen` to `lfr_simulate.py` alongside `simulate()`. The only
change to `validate_lfr.py` is changing its import to match.

---

## Gap 5 — Y index correctness is never verified [NOTE]

**What**: The block receives state in *logical* coordinates `x = [X, Θ, Y, dX, dΘ, dY]`.
`rk4_step` extracts the scheduling variable as `x[:, 2]`, which is index 2 — `Y`.

This happens to be correct in stage coordinates too: `q_stage = [X1, X2, Y]` and
`q_logical = [X, Θ, Y]`. Index 2 is `Y` in both, because the P-transform's third row
is `[0, 0, 1]` — Y is unchanged by the coordinate transform.

**The risk**: if the state ordering ever changes (e.g. `x = [Y, X, Θ, ...]` for
numerical reasons), `x[:, 2]` picks the wrong variable silently. More practically:
if someone reads the block docstring ("z_in[:, :6, :] = x (logical state)") and
expects index 2 to be Θ (angle), they would be confused. The coincidence that
logical index 2 == stage index 2 == Y should be explicitly stated.

**Fix**: add a one-line comment in `rk4_step` and `lfr_block.py`: `# x[:, 2] = Y in
both logical [X, Θ, Y] and stage [X1, X2, Y] coordinates — P's third row is [0,0,1]`.

---

## Gap 6 — No batch-correctness test [NOTE]

**What**: `linalg.solve` on a batched `(batch, 3, 3)` matrix applies the solver
independently per batch item. The code assumes this is equivalent to calling
`linalg.solve` N times with batch size 1. This is true by PyTorch's contract, but
it has never been tested explicitly.

**Why it matters**: Jan's training loop runs with large batch sizes. If there were
a bug in the batched solve (e.g. the RHS shape was wrong, or the matrices were
accidentally shared across the batch), the error would not be caught by any current
test because all existing checks use known-good inputs.

**Fix**: add one check to `lfr_forward.py` or `test_jan_compat.py`:

```python
# Run batch=N with identical inputs; compare against N runs of batch=1.
# Any broadcasting bug shows up as a non-zero diff.
```

---

## Gap 7 — Multi-step BPTT gradient flow tested for only 3 steps [NOTE]

**What**: `test_jan_compat.py` Check E verifies that gradients flow back through 3
consecutive RK4 steps (`x0 → xp1 → xp2 → xp3`). This catches gradient graph
truncation at step boundaries, but 3 steps is too short to catch gradient vanishing
over a realistic training window.

**Why it matters**: in training, backpropagation through time (BPTT) is typically
truncated at 32–256 steps. If `∂loss/∂x0` has already effectively vanished at step
10 (due to repeated multiplication of the Jacobian), the baseline will not be
trainable in practice even though the gradient is technically non-zero.

**What to check**: run the 3-step test with 50 or 100 steps and print the norm of
`x0.grad`. If it vanishes (e.g. < 1e-30), the physics Jacobian is contracting too
fast. This is a property of the system (stable eigenvalues) and cannot be fixed by
code changes — but it would explain why parameter estimation fails to converge, and
should be known before training starts.

---

## Summary table

| # | Issue | File | Priority | Blocks training? |
|---|-------|------|----------|-----------------|
| 1 | GPU crash: plain attributes not moved by `.cuda()` | `lfr_block.py` | ~~CRITICAL~~ **FIXED** | Fixed 2026-04-04 |
| 2 | Augmentation architecture: additive only, can't modify Δ(Y) loop | `lfr_block.py`, `lfr-dzw-zero-alternative.md` | **WARN** | Design question |
| 3 | Row-vector convention silently assumes symmetric K, C | `lfr_forward.py` | **WARN** | No — but maintenance trap |
| 4 | `simulate_frozen` in wrong file | `validate_lfr.py` | **NOTE** | No |
| 5 | Y index coincidence never documented or tested | `lfr_forward.py`, `lfr_block.py` | **NOTE** | No |
| 6 | No batch-correctness test | `lfr_forward.py`, `test_jan_compat.py` | **NOTE** | No |
| 7 | BPTT gradient vanishing not tested beyond 3 steps | `test_jan_compat.py` | **NOTE** | Possibly |

**Before training**: fix Gap 1 (2-line change per parameter in `lfr_block.py`).
Confirm Gap 2 with supervisor. Run Gap 7 test to understand gradient magnitude.
