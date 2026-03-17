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

**Approach**: Call `gantry_discrete_ss(Y[k], fs=16e3)` at each step — this runs `cont2discrete`
with the current Y value. This is the frozen-at-sampling-instant approach.

**Open question for Tóth**: Is this approach sufficiently accurate at fs = 16 kHz given
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
   terms are needed for accuracy at ts = 62.5 µs?

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

## Reference

Tóth, R. (2010). *Modeling and Identification of Linear Parameter-Varying Systems.*
Lecture Notes in Control and Information Sciences, Vol. 403. Springer.
Cited in: `Research-Plan/research-methods.md`
