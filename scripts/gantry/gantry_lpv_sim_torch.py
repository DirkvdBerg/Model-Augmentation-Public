"""
gantry_lpv_sim_torch.py
-----------------------
PyTorch LPV simulation of the dual-gantry FP model.

Implements the discrete-time LPV state-space model derived in LPV/LPV-derivation.tex:

    x[k+1] = A_d(Y[k]) @ x[k] + B_d(Y[k]) @ u[k]
    y[k]   = C_d          @ x[k]

where Y[k] = x[k][2] (self-scheduling: payload Y-position is the third state).

The scheduling variable is extracted as a tensor slice (x_k[2]), never as a Python
scalar via .item(), so the full autograd graph is preserved for BPTT.

Architecture
------------
GantryLPVSimulator(nn.Module):
    forward(x0, u)  -- full simulation, BPTT-compatible (gradients tracked)
    simulate(x0, u) -- inference-only (torch.no_grad wrapper)

- C_d is constant (no Y-dependence). It is computed once in __init__ and
  registered as a buffer so it moves with the module (to(device), dtype).
- State trajectory is accumulated in a Python list and stacked once at the
  end with torch.stack. No in-place tensor writes, preserving the autograd graph.

Wiring note
-----------
The self-scheduling loop p[k] = x[k][2] is wired inside forward(). The caller
does NOT pass the scheduling variable. The caller must ensure x0[2] contains
the correct initial Y-position [m] in the same coordinate frame as gantry_lpv_torch.py.
This wiring does not exist anywhere outside this file yet. For the training loop,
whoever calls forward() is responsible for supplying the correct initial state x0.

Validation (__main__)
---------------------
Test 1: Constant Y free-response
    u = 0, Y held near-constant at 0.3 m (zero initial dY, zero F_Y).
    Compare against scipy reference with A_d, B_d frozen at Y=0.3.
    Expected: max absolute error < 1e-8 (Y drift is negligible at 100 steps).

Test 2: BPTT gradient test
    requires_grad=True on x0. Run forward, compute loss, call backward.
    Verify gradient flows through the LPV loop back to x0.

Reference: LPV/LPV-derivation.tex (self-scheduling, Steps 1-6)
           scripts/gantry/gantry_lpv_torch.py (A_d, B_d computation)
           docs/decisions.md D-015 (augmented matrix exponential)
"""

import sys
import os

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from gantry_lpv_torch import gantry_lpv_matrices_torch, _DEFAULT_FS


class GantryLPVSimulator(nn.Module):
    """
    Discrete-time LPV simulation of the gantry FP model.

    The scheduling variable p[k] = Y[k] = x[k][2] is extracted from the
    state at each step (self-scheduling). Gradients flow through this
    extraction and through the matrix exponential in gantry_lpv_matrices_torch,
    making the full simulation differentiable with respect to x0 and u.

    Parameters
    ----------
    fs : torch.Tensor, optional
        Sample frequency [Hz], scalar float64. Default: 16000 Hz.
    """

    def __init__(self, fs: torch.Tensor = None):
        super().__init__()
        if fs is None:
            fs = _DEFAULT_FS
        self.register_buffer('fs', fs.to(dtype=torch.float64))

        # C_d does not depend on Y. Compute once with an arbitrary Y value.
        # Any Y gives the same C_d; Y=0 is convenient.
        with torch.no_grad():
            _, _, C_d, D_d = gantry_lpv_matrices_torch(
                torch.tensor(0.0, dtype=torch.float64), self.fs
            )
        self.register_buffer('C_d', C_d)  # (3, 6)
        self.register_buffer('D_d', D_d)  # (3, 3), zero

    def forward(self, x0: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Run LPV simulation with BPTT support.

        Implements the self-scheduling loop from LPV/LPV-derivation.tex:

            p[k]   = Y[k] = x[k][2]               (self-scheduling)
            A_d(k), B_d(k) = matrix_exp(...)       (frozen-at-sampling-instant ZOH)
            x[k+1] = A_d(k) @ x[k] + B_d(k) @ u[k]
            y[k]   = C_d @ x[k]                   (D_d = 0)

        The scheduling variable is extracted as x_k[2] (tensor slice), never via
        .item(). This preserves the autograd graph through the self-scheduling
        path and through the matrix exponential in gantry_lpv_matrices_torch.

        IMPORTANT: the wiring p[k] = x[k][2] is implemented here. The caller
        does NOT supply p[k]. The caller must set x0[2] to the correct initial
        Y-position [m]. This is the only point in the codebase where the
        scheduling variable is extracted from the state.

        Parameters
        ----------
        x0 : (6,) torch.Tensor
            Initial state [X, Theta, Y, dX, dTheta, dY] in logical coordinates.
            x0[2] must be the initial Y-position [m].
        u  : (N, 3) torch.Tensor
            Input sequence [F_X1, F_X2, F_Y] in stage coordinates [N].

        Returns
        -------
        y : (N, 3) torch.Tensor
            Output sequence [X1, X2, Y] in stage coordinates [m].
            y[k] = C_d @ x[k], so y[0] corresponds to the response at x0.
        """
        x_k = x0.to(dtype=torch.float64)
        u = u.to(dtype=torch.float64)
        N = u.shape[0]

        states = []

        for k in range(N):
            # Self-scheduling: Y is the third state (0-indexed).
            # x_k[2] is a 0-d tensor slice. Grad flows through here on the
            # backward pass, linking state evolution to scheduling.
            Y_k = x_k[2]

            # LPV matrices at the current scheduling variable.
            # Grad flows from A_d(Y_k) and B_d(Y_k) back to Y_k via matrix_exp.
            A_d, B_d, _, _ = gantry_lpv_matrices_torch(Y_k, self.fs)

            # Collect x[k] before the update so that y[k] = C_d @ x[k].
            states.append(x_k)

            # Out-of-place state update. In-place operations (x_k += ...) break
            # the autograd graph. This produces a new tensor at each step.
            x_k = A_d @ x_k + B_d @ u[k]

        # Stack accumulated states into (N, 6).
        # torch.stack preserves all grad_fns from the loop.
        X = torch.stack(states, dim=0)  # (N, 6)

        # Output equation: y[k] = C_d @ x[k].
        # Equivalent batch form: y = X @ C_d.T  gives (N, 3).
        y = X @ self.C_d.T  # (N, 3)

        return y

    @torch.no_grad()
    def simulate(self, x0: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Inference-only simulation (no gradient tracking).

        Identical to forward() but wrapped in torch.no_grad() for efficiency.
        Use for validation, comparison, and any context where BPTT is not needed.
        """
        return self.forward(x0, u)


# ---------------------------------------------------------------------------
# Validation
# Run from repo root:
#   conda run -n GraduationProject python scripts/gantry/gantry_lpv_sim_torch.py
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import numpy as np
    from gantry_ss import gantry_discrete_ss

    dtype = torch.float64

    # -----------------------------------------------------------------------
    # Test 1: Constant Y free-response vs scipy frozen reference
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Test 1: Constant Y free-response vs scipy reference")
    print("=" * 60)

    # Initial state: Y = 0.3 m, small X displacement, all velocities zero.
    # With u = 0 and dY = 0 initially, Y stays near 0.3 m throughout.
    # The LPV simulator (self-scheduling) and the frozen scipy reference
    # (A_d, B_d fixed at Y=0.3) should therefore agree to near-numerical
    # precision. Any residual error comes from Y drifting slightly from 0.3.
    Y_val = 0.3
    N_steps = 100

    x0 = torch.zeros(6, dtype=dtype)
    x0[0] = 1e-3   # X = 1 mm (small excitation to observe dynamics)
    x0[2] = Y_val  # Y = 0.3 m

    u_zero = torch.zeros(N_steps, 3, dtype=dtype)

    # LPV simulation (self-scheduling loop)
    sim = GantryLPVSimulator()
    y_lpv = sim.simulate(x0, u_zero)  # (N_steps, 3)

    # Scipy frozen reference: A_d, B_d fixed at Y=0.3
    A_ref, B_ref, C_ref, D_ref = gantry_discrete_ss(Y=Y_val)
    A_ref_t = torch.tensor(A_ref, dtype=dtype)
    B_ref_t = torch.tensor(B_ref, dtype=dtype)
    C_ref_t = torch.tensor(C_ref, dtype=dtype)

    x_sc = x0.clone()
    states_ref = []
    for k in range(N_steps):
        states_ref.append(x_sc)
        x_sc = A_ref_t @ x_sc + B_ref_t @ u_zero[k]
    X_ref = torch.stack(states_ref, dim=0)   # (N_steps, 6)
    y_ref = X_ref @ C_ref_t.T               # (N_steps, 3)

    max_err = (y_lpv - y_ref).abs().max().item()
    tol = 1e-8
    status = 'PASS' if max_err < tol else 'FAIL'
    print(f"  Max absolute error (LPV vs scipy frozen at Y={Y_val}): {max_err:.2e}")
    print(f"  Tolerance: {tol:.0e}  ->  {status}")
    print(f"  (Residual reflects Y drift from {Y_val} m over {N_steps} steps)")

    # -----------------------------------------------------------------------
    # Test 2: BPTT gradient test
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Test 2: BPTT gradient test")
    print("=" * 60)
    print("  Checks that grad flows from loss through the LPV loop back to x0.")
    print("  Two grad paths at each step:")
    print("    (A) direct:          A_d @ x_k, B_d @ u[k]")
    print("    (B) self-scheduling: A_d(Y_k=x_k[2]) via matrix_exp")

    x0_grad = x0.clone().detach().requires_grad_(True)
    u_small = torch.zeros(N_steps, 3, dtype=dtype)

    sim_grad = GantryLPVSimulator()
    y_out = sim_grad.forward(x0_grad, u_small)
    loss = y_out.sum()
    loss.backward()

    grad_ok = x0_grad.grad is not None
    print(f"\n  Backward pass succeeded: {grad_ok}")
    if grad_ok:
        print(f"  grad norm (x0):  {x0_grad.grad.norm().item():.6e}")
        print(f"  grad (x0):       {x0_grad.grad.tolist()}")

    print(f"\nGradient test: {'PASS' if grad_ok else 'FAIL'}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    all_pass = (status == 'PASS') and grad_ok
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 60)
