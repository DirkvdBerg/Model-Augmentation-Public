# LPV Discretization — Specification and Open Questions

This document defines exactly what the LPV discretization must deliver, how it fits into the
augmentation framework (replacing the MSD reference), and what specific questions Tóth (2010)
must answer before implementation begins.

---

## What we need to replace

In Jan's MSD reference (`scripts/ecc_2025/msd_ndof_interconnect_dynamic.py`), the baseline
plant is a custom `Parameterized_MSD_State_Block` that hard-codes the nonlinear physics
equations in its `forward()` call. This block is the MSD equivalent of our gantry baseline.

For the gantry, we need an equivalent block — `LPV_Linear_State_Block` (see D-011) — that
computes the state update using Y-dependent matrices:

```
x[k+1] = A_d(Y[k]) @ x[k] + B_d(Y[k]) @ u[k]
y[k]   = C @ x[k]
```

where Y[k] = x[k, 2] (the Y-position state in stage coordinates).

---

## Two separate use cases with different requirements

### Use case 1 — Validation simulation (Step 2)

**What**: Simulate the LPV model in Python over a recorded trajectory to verify the
Y-dependent dynamics are correctly captured.

**Requirement**: Compute A_d(Y[k]), B_d(Y[k]) at each timestep k.

**Approach**: Call `gantry_discrete_ss(Y[k], fs=20e3)` at each step — this runs `cont2discrete`
with the current Y value. This is the frozen-at-sampling-instant approach.

**Open question for Tóth**: Is this approach sufficiently accurate at fs = 20 kHz given
ΔY ≤ 0.125 mm/sample? What is the order of the approximation error in ts?

---

### Use case 2 — Training loop integration (Step 3)

**What**: `LPV_Linear_State_Block.forward()` is called at every step of every training
iteration inside a PyTorch autograd graph. Speed and differentiability are critical.

**Requirement**: A_d(Y) and B_d(Y) must be computable as a closed-form, analytically
differentiable function of Y — NOT via `cont2discrete` (matrix exponentiation is not
torch-differentiable and is too slow at training scale).

**What this function must look like** — one of:

| Option | Form | Notes |
|--------|------|-------|
| A | `A_d(Y) = A0_d + Y·A1_d + Y²·A2_d` | Polynomial; requires Tóth or Taylor expansion |
| B | `A_d(Y) = A0_d + Y·A1_d` | Linear-affine approximation; simpler, may lose Y² term |
| C | `A_d(Y[k])` interpolated from a pre-computed grid | Table lookup; not differentiable w.r.t. Y |
| D | Hardcode A_d(Y) symbolically from physics | Like MSD block — most accurate, most work |

Option A is preferred if Tóth provides the polynomial expansion.
Option D matches the MSD reference pattern and is always valid.

---

## Structure of A_c(Y) from the physics

The continuous-time state matrix is:
```
A_c(Y) = [ 0      |    I   ]
          [-M(Y)⁻¹K | -M(Y)⁻¹C]
```

M(Y) is polynomial in Y:
- M[0,1] = M[1,0] = (m1-m2)·Lb/2 − mh·Y         ← linear in Y
- M[1,1] = Jb + Jh + (m1+m2)·Lb²/4 + mh·d² + mh·Y²  ← quadratic in Y
- All other entries of M are constant

Therefore:
- M(Y) = M0 + Y·M1 + Y²·M2  (polynomial in Y, known analytically)
- M(Y)⁻¹ is a rational function of Y (not polynomial)
- A_c(Y) = f(M(Y)⁻¹) is rational in Y — NOT affine

Similarly B_c(Y) = [0; M(Y)⁻¹] — rational in Y, C and D are constant.

---

## Questions Tóth (2010) must answer

1. **Frozen-at-sampling-instant accuracy**: What is the order of the approximation error
   in ts when Y is frozen at Y[k] during [k·ts, (k+1)·ts]? Is it O(ts²) or better?
   Does it depend on the rate of change of Y?

2. **Polynomial expansion of A_d(Y)**: Does the paper provide a method to compute
   `A_d(Y) = A0_d + Y·A1_d + Y²·A2_d + ...` analytically from A_c(Y)? If so, how many
   terms are needed for accuracy at ts = 50 µs?

3. **Non-affine handling**: A_c(Y) is rational (not polynomial) in Y. Does Tóth's method
   apply directly, or does it assume an affine (A0 + ρ·A1) structure?

4. **B_d(Y) treatment**: Does B_d(Y) require the same polynomial expansion, or is the
   frozen-at-sampling-instant approach sufficient for B even if A needs a correction?

5. **Quasi-LPV considerations**: Y is a state, not an exogenous signal. Does Tóth address
   quasi-LPV systems, and does it change the discretization approach?

---

## Decision gate

After reading Tóth (2010), update D-012 with:
- Whether frozen-at-sampling-instant is justified for validation (Use case 1)
- Which option (A/B/C/D above) to use for the training loop (Use case 2)
- Whether `gantry_discrete_ss` needs modification or a new `gantry_lpv_analytic.py`

---

## Method selection — findings from Tóth (2010)

*Assessed via `assess-paper` skill against `literature/toth2010_zoh-discretization-lpv.pdf`.*

### Use case 1 (validation): frozen-at-sampling-instant ✅ justified

Tóth Section III-B shows the complete frozen-at-sampling-instant method has **zero local
truncation error** within the ZOH assumption. The only real error is from the slowly-varying
scheduling during `[k·ts, (k+1)·ts]`, which at 20 kHz and ΔY ≤ 0.100 mm/sample is
negligible. Calling `cont2discrete(A_c(Y[k]), ts)` at each step is theoretically exact.

### Use case 2 (training loop): exact ZOH via `torch.linalg.matrix_exp` ✅ chosen

**Fact-checked 2026-03-17**: `torch.linalg.matrix_exp` fully supports PyTorch autograd.
Gradients flow through the matrix exponential and back to scalar inputs (e.g. Y).
Test confirmed: `d/dY [sum(expm(A_c(Y)·ts))]` returns a valid gradient.

This makes exact ZOH available inside the training loop — no approximation needed.

The five options and why each was accepted or rejected:

| Option | Form | Torch-differentiable? | Verdict |
|--------|------|-----------------------|---------|
| A | `A_d(Y) = A0_d + Y·A1_d + Y²·A2_d` | ✅ Yes | ❌ Not achievable: A_c(Y) is rational, so no exact polynomial A_d(Y) exists. |
| B | `A_d(Y) = A0_d + Y·A1_d` | ✅ Yes | ❌ Drops the Y² term in M[1,1] — dominant Y-dependence lost. |
| C | Interpolate A_d(Y) from a pre-computed grid | ❌ Not natively | ❌ scipy/numpy lookup outside autograd graph. |
| D | `A_d(Y) = I + ts·A_c(Y)`, `B_d(Y) = ts·B_c(Y)` | ✅ Yes — all tensor ops | ⚠️ First-order (rectangular), error **O(ts)**. Still valid fallback if matrix_exp is too slow at scale. |
| **E** | `A_d(Y) = expm(A_c(Y)·ts)` via `torch.linalg.matrix_exp` | ✅ **Yes — confirmed** | ✅ **Chosen.** Zero local truncation error (exact ZOH within frozen-at-sampling-instant). No approximation. Mirrors Use case 1 exactly but with torch ops instead of scipy. |

### Why Option E is torch-differentiable

#### The singularity problem — and why naive B_d fails

The standard ZOH formula for B_d is:
```
B_d = A_c⁻¹ · (A_d - I) · B_c
```
This requires A_c to be invertible. **The gantry A_c is singular** — the top-left 3×3 block
is all zeros (position states have no velocity-independent dynamics — rigid body modes). So
`A_c⁻¹` does not exist and this formula cannot be used.

#### Correct formula — augmented matrix exponential

The correct ZOH method (used internally by scipy `cont2discrete`) avoids the singularity by
augmenting the system matrix before taking the exponential:

```
n = 6  (number of states)
m = 3  (number of inputs)

M_aug = [ A_c(Y)   B_c(Y) ]   ← (n+m) × (n+m) = 9×9 matrix
        [   0         0   ]

exp(M_aug · ts) = [ A_d(Y)   B_d(Y) ]
                  [   0         I   ]

→ A_d = exp(M_aug · ts)[:n, :n]     # top-left  6×6
→ B_d = exp(M_aug · ts)[:n, n:]     # top-right 6×3
```

This works even when A_c is singular. It is equivalent to the exact ZOH formula and is what
scipy uses internally. For the torch implementation, `torch.linalg.matrix_exp(M_aug * ts)`
is a single differentiable call that produces both A_d and B_d.

#### Full runtime computation for Option E

```
M(Y)   = M0 + Y·M1 + Y²·M2           # 3×3 polynomial in Y (tensor ops)
M_inv  = torch.linalg.inv(M(Y))       # 3×3 inverse (tensor op)
A_c(Y) = [[0,      I    ],            # 6×6 continuous-time state matrix
           [-M_inv@K, -M_inv@C]]
B_c(Y) = [[zeros(3,3)],               # 6×3 continuous-time input matrix
           [M_inv      ]]

M_aug  = [[A_c(Y), B_c(Y)],           # 9×9 augmented matrix
           [zeros(3,6), zeros(3,3)]]

EM     = torch.linalg.matrix_exp(M_aug * ts)   # 9×9, differentiable
A_d(Y) = EM[:6, :6]                   # exact discrete A
B_d(Y) = EM[:6, 6:]                   # exact discrete B
```

`torch.linalg.matrix_exp` is a native PyTorch op — autograd traces through it.
Gradients propagate from the loss back through `A_d(Y[k])` and `B_d(Y[k])` to `Y[k]`.

`cont2discrete` (scipy) remains non-differentiable and is only used for validation (Use case 1).
It uses the same augmented matrix exponential internally — so the two approaches are
mathematically identical. Any discrepancy between them would be a numerical precision issue,
not a formula difference.

### Approximation error orders (from Tóth Table I)

| Method | Local truncation error | Torch-differentiable |
|--------|----------------------|---------------------|
| Exact ZOH via scipy `cont2discrete` | Zero | ❌ No |
| Exact ZOH via `torch.linalg.matrix_exp` (Option E) | **Zero** | ✅ Yes |
| Rectangular (Option D) | O(ts) — first-order | ✅ Yes |
| Trapezoidal | O(ts²) — second-order | ❌ Not natively |

Option E dominates Option D on accuracy with no cost in differentiability.
Option D is kept as a documented fallback (simpler, faster per step).

### Evaluation strategy — Step 2 (pure Python, no framework)

Step 2 is validation only. Neither Option D nor E is used here — scipy `cont2discrete` is exact and fast. The sequence is:

1. Implement `gantry_lpv_ss.py` using exact ZOH: call `gantry_discrete_ss(Y[k])` at each step.
2. At Y=0.3 (constant): verify output matches frozen LTI exactly (same call → identical matrices).
3. At Y = 0.1, 0.2, 0.3, 0.4, 0.5 m: export MATLAB G at each Y, compare Python vs MATLAB to < 1e-10.
4. After Step 2 is validated, quantify rectangular approximation error (Option D) numerically:
   compare `A_d = I + ts·A_c(Y)` vs `expm(A_c(Y)·ts)` at each Y value.
   This establishes the O(ts) error bound — confirms Option E is preferred over D for training.

### Open question — LPV baseline in the LFR interconnect (Step 3)

Before implementing the training block, Drenth's thesis (`literature/drenth2025_lpv-lfr-thesis.pdf`)
must be consulted to understand how the LPV baseline enters the LFR interconnect structure.

Full question list: **`docs/lpv-lfr-interconnect.md`** (spec file for the Drenth assessment).

Use `assess-paper` with spec `docs/lpv-lfr-interconnect.md` on Drenth's thesis before
starting Step 3 implementation.

### Output files

- Full Q-by-Q assessment: `assess-paper-workspace/iteration-1/toth-assessment/with_skill/outputs/assessment.md`
- Decision updated in: `docs/decisions.md` D-012

---

## Reference

Tóth, R. (2010). *ZOH Discretization of LPV Systems.*
File: `literature/toth2010_zoh-discretization-lpv.pdf`
Cited in: `Research-Plan/research-methods.md` as `toth2010discretization`
