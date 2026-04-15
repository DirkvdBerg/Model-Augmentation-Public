# LPV-LFR Conversion Guide
## Rewriting `lfr_forward.py` from LPV-SS to True LFR-First

**Reference documents:**
- Theoretical spec: `literature/lpv-lfr/Additional notes/LPV-LFR-Implementation-Spec.md`
- Polynomial derivation: `literature/lpv-lfr/Additional notes/LPV-LFR-Rational-rewrite.md`
- Rational MATLAB verification: `Verification/LPV-LFR-Rational/LPV_LFR_rational_verification.m`

---

## Situation at the Start of This Conversion

### What already exists and is correct — do not touch

**`lpv_lfr_baseline/core/lfr_matrices.py`**
- Contains `GMatrix` dataclass with all G submatrices: `Ax, Bw, Bu, Cz, Dzw, Dzu, Cy`
- `build_G_matrix(M0, M1, M2, K, C)` constructs them correctly
- Module-level singleton `G = build_G_matrix(M0, M1, M2, K, C)` is already available
- All constant matrices verified — no changes needed

**`lpv_lfr_baseline/core/physics.py`**
- All physical scalar parameters: `m1, m2, mb, mh, Jb, Jh, Lb, d, ...`
- Mass matrix decomposition `M0, M1, M2` and constant `K, C`
- Verified correct against `kamtin-fp-model/` — no changes to existing content

### What is wrong — the only file that needs changing

**`lpv_lfr_baseline/core/lfr_forward.py`** — current docstring says:
```
Steps: M(Y) -> fnet -> v=solve(M,fnet) -> z=[v;Yv] -> w=[Yv;Y^2 v] -> xdot=[qdot;v].
```

Line 42 is the smoking gun:
```python
xdot = torch.cat([x[:, 3:], v], dim=-1)   # WRONG: xdot driven directly from v, not from G
```

`w` is reconstructed after `xdot` is already computed. `G` is never used. This is LPV-SS, not LFR.

---

## Step 1 — Do NOT Add Polynomial Constants as Module-Level Values

> **Critical architectural constraint: physical parameters are trainable.**
>
> The augmentation framework makes physical parameters (`m1, m2, mb, mh, Jb, Jh, Lb, d`, ...)
> trainable `nn.Parameter` objects during the parameter recovery step.
>
> `alpha`, `beta`, `gamma`, `N0`, `N1`, `N2` are all explicit functions of these parameters.
> If they are precomputed as module-level constants in `physics.py`, they will be frozen at
> their initial values. **Gradients will not flow back through them to the physical parameters.**
> This silently breaks parameter recovery.
>
> The same issue already exists for the `GMatrix` singleton in `lfr_matrices.py` (see comment
> there: "If physical parameters become trainable in future, call build_G_matrix() inside
> forward() instead of using this singleton."). The polynomial constants have exactly the same
> dependency structure and must follow the same pattern.

**The correct pattern:** provide a `build_poly_constants()` function that computes
`alpha, beta, gamma, N0, N1, N2` from the current parameter values. Call it **inside
`forward()`** so that autograd tracks the full dependency chain:

```
physical parameters (nn.Parameter)
  → alpha, beta, gamma, N0, N1, N2  (via build_poly_constants)
  → a = N(Y) @ fnet / dY            (analytical loop solution)
  → z, w
  → xdot = Ax @ x + Bw @ w + Bu @ u
  → loss
```

Similarly, `G` must be rebuilt inside `forward()` via `build_G_matrix()` rather than using
the module-level singleton, whenever physical parameters are trainable.

**Add to `physics.py` — the builder function only, no module-level constants:**

```python
# ----------------------------------------------------------------------
# Polynomial constant builder for the analytical LPV-LFR loop solution
# Source: literature/lpv-lfr/Additional notes/LPV-LFR-Rational-rewrite.md
#
# IMPORTANT: physical parameters are trainable (nn.Parameter) in the
# augmentation parameter recovery step. Call build_poly_constants()
# INSIDE forward() — do not cache the result as a module-level constant.
#
# Scalar shorthands:
#   alpha = m1 + m2 + mb + mh
#   beta  = (m1 - m2) * Lb / 2
#   gamma = Jb + Jh + (m1 + m2) * Lb**2 / 4   (WITHOUT mh*d^2)
#
# Denominator polynomial in Y (always positive when M(Y) is PD):
#   dY = mh * (alpha*gamma - beta**2 + 2*beta*mh*Y + mh*(alpha-mh)*Y**2)
#
# Adjugate coefficient matrices:  adj(M(Y)) = N0 + Y*N1 + Y^2*N2
# ----------------------------------------------------------------------
def build_poly_constants(m1, m2, mb, mh, Jb, Jh, Lb, d):
    """
    Compute polynomial coefficient matrices for M(Y)^{-1} from physical parameters.

    Returns (alpha, beta, gamma, N0, N1, N2) where N0, N1, N2 are (3,3) tensors
    and alpha, beta, gamma are scalars.

    All outputs are differentiable with respect to the inputs.
    Call inside forward() when physical parameters are trainable.
    """
    alpha = m1 + m2 + mb + mh
    beta  = (m1 - m2) * Lb / 2
    gamma = Jb + Jh + (m1 + m2) * Lb**2 / 4    # WITHOUT mh*d^2

    dtype = m1.dtype if hasattr(m1, 'dtype') else torch.float64
    z = torch.zeros(1, dtype=dtype).squeeze()    # differentiable zero

    N0 = torch.stack([
        torch.stack([mh * gamma,                -beta * mh,                    -beta * d * mh              ]),
        torch.stack([-beta * mh,                 alpha * mh,                    alpha * d * mh             ]),
        torch.stack([-beta * d * mh,             alpha * d * mh,                alpha * (gamma + mh*d**2) - beta**2]),
    ])

    N1 = torch.stack([
        torch.stack([z,         mh**2,       d * mh**2  ]),
        torch.stack([mh**2,     z,           z          ]),
        torch.stack([d*mh**2,   z,           2*beta*mh  ]),
    ])

    N2 = torch.stack([
        torch.stack([mh**2,     z,           z                      ]),
        torch.stack([z,         z,           z                      ]),
        torch.stack([z,         z,           alpha*mh - mh**2       ]),
    ])

    return alpha, beta, gamma, N0, N1, N2
```

**Naming note:** `d` in `physics.py` is the geometric arm-length parameter (0.1 m).
In the forward pass, use a different name (e.g. `dY`) for the scalar denominator value.

---

## Step 2 — Rewrite `lfr_forward.py`

> **Because physical parameters are trainable, `lfr_forward` must receive all
> parameter-dependent objects as arguments — it cannot import module-level constants
> from `physics.py` or use the `lfr_matrices.G` singleton.**
>
> The caller (e.g. `lfr_block.py` or `lfr_fit_system.py`) is responsible for calling
> `build_G_matrix(...)` and `build_poly_constants(...)` from the current (possibly updated)
> parameter values and passing the results in. This keeps the full gradient path intact:
>
> `nn.Parameter → build_G_matrix/build_poly_constants → G, alpha, beta, gamma, N0/N1/N2 → lfr_forward → loss`

**Mandatory signal ordering (from `LPV-LFR-Implementation-Spec.md` Section 2):**
```
x, u, Y  →  Delta(Y)  →  rhs = Cz@x + Dzu@u  →  z (loop solve)  →  w = Delta@z  →  xdot = Ax@x + Bw@w + Bu@u
```

**New `lfr_forward.py`:**

```python
"""
lfr_forward.py
--------------
True LPV-LFR forward pass for the gantry baseline.

Signal flow (LFR-first, NOT collapsed LPV-SS):
    x, u, Y
    -> Delta(Y) = Y * I6                          [explicit scheduling block]
    -> rhs = Cz @ x + Dzu @ u                    [RHS of loop equation, from G]
    -> z = L(Y)^{-1} rhs   (analytical)          [solve loop]
    -> w = Delta(Y) @ z                           [apply scheduling block]
    -> xdot = Ax @ x + Bw @ w + Bu @ u           [state equation, through G]
    -> y = Cy @ x                                  [output equation, from G]

The analytical loop solution uses the polynomial form of M(Y)^{-1}:
    M(Y)^{-1} = (N0 + Y*N1 + Y^2*N2) / dY
    dY = mh * (alpha*gamma - beta^2 + 2*beta*mh*Y + mh*(alpha-mh)*Y^2)

Derived in: literature/lpv-lfr/Additional notes/LPV-LFR-Rational-rewrite.md
Spec:       literature/lpv-lfr/Additional notes/LPV-LFR-Implementation-Spec.md

IMPORTANT — trainable parameters:
    Physical parameters (m1, m2, mb, mh, ...) are nn.Parameter in the augmentation
    parameter recovery step. G, N0, N1, N2, alpha, beta, gamma all depend on them.
    This function therefore receives ALL parameter-dependent quantities as arguments.
    Do NOT import them as module-level constants. The caller must rebuild G and the
    polynomial constants from current parameter values at each forward() call.

The key audit criterion: xdot is computed via Ax, Bw, Bu (through w),
NOT directly from the acceleration a. This is what makes it LFR-first.
"""

import torch
from lpv_lfr_baseline.core.lfr_matrices import GMatrix


def lfr_forward(
    x:     torch.Tensor,   # (batch, 6)   state in logical coordinates
    u:     torch.Tensor,   # (batch, 3)   input in logical coordinates
    Y:     torch.Tensor,   # (batch,)     scheduling variable  x[:, 2] in caller
    G:     GMatrix,        # G submatrices — rebuilt from current params by caller
    K:     torch.Tensor,   # (3, 3)       stiffness — current params
    C:     torch.Tensor,   # (3, 3)       damping   — current params
    mh:    torch.Tensor,   # scalar       payload mass — current params
    alpha: torch.Tensor,   # scalar       from build_poly_constants
    beta:  torch.Tensor,   # scalar       from build_poly_constants
    gamma: torch.Tensor,   # scalar       from build_poly_constants (WITHOUT mh*d^2)
    N0:    torch.Tensor,   # (3, 3)       from build_poly_constants
    N1:    torch.Tensor,   # (3, 3)       from build_poly_constants
    N2:    torch.Tensor,   # (3, 3)       from build_poly_constants
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    True LPV-LFR forward pass. Returns (xdot, z, w, y).

    xdot is computed through G.Bw and w — never directly from the
    acceleration vector a. z and w are causally upstream of xdot.

    All parameter-dependent inputs (G, K, C, mh, alpha, beta, gamma, N0, N1, N2)
    must be built by the caller from the current nn.Parameter values so that
    gradients flow back to the physical parameters.
    """
    # ------------------------------------------------------------------
    # Step 1: Delta(Y) = Y * I6   [explicit scheduling block]
    # Applied as scalar multiplication throughout; conceptually it is the
    # 6x6 block that forms the right branch of the LFR interconnection.
    # ------------------------------------------------------------------
    # (applied in Steps 3 and 4 below)

    # ------------------------------------------------------------------
    # Step 2: RHS of loop equation  rhs = Cz @ x + Dzu @ u
    # Equivalently: rhs = [M0inv @ f_net; 0]  (spec Section 3.2)
    # K and C passed explicitly so gradient flows to physical parameters.
    # ------------------------------------------------------------------
    fnet = -(x[:, :3] @ K.T) - (x[:, 3:] @ C.T) + u          # (batch, 3)
    rhs  = (x @ G.Cz.T) + (u @ G.Dzu.T)                       # (batch, 6)

    # ------------------------------------------------------------------
    # Step 3: Solve loop analytically:  z = L(Y)^{-1} rhs
    #
    # From LPV-LFR-Rational-rewrite.md Final Result:
    #   z = [M(Y)^{-1} fnet;  Y * M(Y)^{-1} fnet]
    #
    # M(Y)^{-1} = N(Y) / dY  where:
    #   dY  = mh * (alpha*gamma - beta^2 + 2*beta*mh*Y + mh*(alpha-mh)*Y^2)
    #   N(Y)= N0 + Y*N1 + Y^2*N2
    #
    # alpha, beta, gamma, N0, N1, N2 passed by caller — differentiable
    # w.r.t. physical parameters.
    # ------------------------------------------------------------------
    dY  = mh * (alpha * gamma - beta**2
                + 2 * beta * mh * Y
                + mh * (alpha - mh) * Y**2)                    # (batch,)

    Ye  = Y[:, None, None]                                      # (batch, 1, 1)
    NY  = N0.unsqueeze(0) + Ye * N1.unsqueeze(0) \
              + Ye**2 * N2.unsqueeze(0)                         # (batch, 3, 3)

    a   = (NY @ fnet.unsqueeze(-1)).squeeze(-1) / dY[:, None]  # (batch, 3)
    # a = M(Y)^{-1} fnet  — upper half of z

    z   = torch.cat([a, Y[:, None] * a], dim=-1)               # (batch, 6)

    # ------------------------------------------------------------------
    # Step 4: w = Delta(Y) @ z = Y * z
    # ------------------------------------------------------------------
    w   = Y[:, None] * z                                        # (batch, 6)

    # ------------------------------------------------------------------
    # Step 5: xdot = Ax @ x + Bw @ w + Bu @ u    [THROUGH G, not from a]
    #
    # AUDIT: this line is what makes this LFR-first.
    # w is causally upstream of xdot via G.Bw.
    # The collapsed (wrong) version would be:
    #   xdot = torch.cat([x[:, 3:], a], dim=-1)    <- DO NOT USE
    # ------------------------------------------------------------------
    xdot = (x @ G.Ax.T) + (w @ G.Bw.T) + (u @ G.Bu.T)        # (batch, 6)

    # ------------------------------------------------------------------
    # Step 6: y = Cy @ x
    # ------------------------------------------------------------------
    y    = x @ G.Cy.T                                           # (batch, 3)

    return xdot, z, w, y
```

---

## Step 3 — Update All Call Sites

The function signature changes from:
```python
lfr_forward(x, u, Y, M0, M1, M2, K, C)
```
to:
```python
lfr_forward(x, u, Y, G, K, C, mh, alpha, beta, gamma, N0, N1, N2)
```

**Because physical parameters are trainable, the caller must rebuild `G` and the polynomial
constants from the current parameter values at each forward pass** — not use cached
module-level singletons. The pattern at each call site is:

```python
from lpv_lfr_baseline.core.lfr_matrices import build_G_matrix
from lpv_lfr_baseline.core.physics import build_poly_constants

# Inside the model's forward() method, where self.m1, self.m2, ... are nn.Parameter:
G = build_G_matrix(M0, M1, M2, self.K, self.C)   # rebuilt from current params
alpha, beta, gamma, N0, N1, N2 = build_poly_constants(
    self.m1, self.m2, self.mb, self.mh, self.Jb, self.Jh, self.Lb, self.d
)

xdot, z, w, y = lfr_forward(x, u, Y, G, self.K, self.C,
                              self.mh, alpha, beta, gamma, N0, N1, N2)
```

> **Do not** pass the module-level `G` singleton from `lfr_matrices.py` or the frozen
> scalar constants from `physics.py` — those do not update with the trainable parameters.

Search for all callers to update:
- `lpv_lfr_baseline/core/lfr_simulate.py`
- `lpv_lfr_baseline/blocks/lfr_block.py`
- `lpv_lfr_baseline/blocks/lfr_fit_system.py`
- Any test files in `lpv_lfr_baseline/tests/`

---

## Step 4 — Verification

### 4a. Numerical equivalence with old implementation

The new LFR-first pass must produce numerically identical results to the old collapsed pass.
Write a verification script that runs both implementations on the same inputs and asserts:
```python
assert torch.allclose(xdot_new, xdot_old, atol=1e-10)
assert torch.allclose(z_new,    z_old,    atol=1e-10)
assert torch.allclose(w_new,    w_old,    atol=1e-10)
```
If these fail, the polynomial constants (N0, N1, N2) or the denominator formula are wrong.

### 4b. Structural (LFR-first) audit

Verify the decisive criterion: `w` must be causally upstream of `xdot`.

```python
# Inject a perturbation into w and check that xdot changes via Bw
# (this is automatic if xdot = Ax@x + Bw@w + Bu@u)
import torch
from lpv_lfr_baseline.core.lfr_matrices import G

w_perturbed = w + torch.randn_like(w) * 0.01
xdot_perturbed = (x @ G.Ax.T) + (w_perturbed @ G.Bw.T) + (u @ G.Bu.T)
assert not torch.allclose(xdot_perturbed, xdot)   # xdot must depend on w
```

Alternatively, use autograd:
```python
w_test = w.detach().requires_grad_(True)
xdot_test = (x @ G.Ax.T) + (w_test @ G.Bw.T) + (u @ G.Bu.T)
xdot_test.sum().backward()
assert w_test.grad is not None            # gradient must flow from xdot through w
assert w_test.grad.abs().max() > 0
```

### 4c. Check that the collapsed pattern is gone

Grep the new `lfr_forward.py` to confirm the collapsed pattern does not appear:
```bash
grep "torch.cat.*x\[:, 3:\].*v\|x\[:, 3:\].*a" lpv_lfr_baseline/core/lfr_forward.py
```
This must return nothing. The line `xdot = torch.cat([x[:, 3:], v], dim=-1)` must not exist.

---

## Summary of Changes

| File | Action |
|------|--------|
| `lpv_lfr_baseline/core/physics.py` | Add `build_poly_constants()` function — **no module-level constants** |
| `lpv_lfr_baseline/core/lfr_matrices.py` | No changes to `GMatrix` or `build_G_matrix()`. **Do not use the `G` singleton in training code** — rebuild inside `forward()` |
| `lpv_lfr_baseline/core/lfr_forward.py` | Full rewrite — LFR-first signal flow; all param-dependent quantities received as arguments |
| `lpv_lfr_baseline/core/lfr_simulate.py` | Update call site: rebuild `G` and polynomial constants from current params before calling |
| `lpv_lfr_baseline/blocks/lfr_block.py` | Update call site: same as above |
| `lpv_lfr_baseline/blocks/lfr_fit_system.py` | Update call site: same as above; this is where `nn.Parameter` physical params live |
| `lpv_lfr_baseline/tests/*` | Update call sites; add structural audit tests |

> **Root constraint:** physical parameters are `nn.Parameter` in the augmentation parameter
> recovery step. Any quantity derived from them (G submatrices, N0/N1/N2, alpha/beta/gamma)
> must be computed inside `forward()` to keep the gradient path intact. Using module-level
> singletons silently freezes those quantities and breaks parameter recovery.

## What This Unlocks

Once `lfr_forward.py` uses `G` as the primary computational object:
- `G` is available as an explicit model object for synthesis tools
- Augmentation can be done by extending `GMatrix` with cross-coupling blocks and extending `Delta`
  without touching the baseline loop — baseline LFR structure is preserved
- The repeated-$Y$ coupling structure is explicit in `Delta(Y) = Y * I6`
