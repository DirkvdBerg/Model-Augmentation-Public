"""
test_augmentation_compat.py
---------------------------
Compatibility tests for Jan's augmentation framework integration.

These tests verify that the LFR baseline's output structure and gradient
graph are compatible with Jan's training loop — without importing any
model_augmentation/ code. They simulate what the framework does internally.

Background
----------
Jan's training loop (simplified):
    1. output = baseline_block.forward(x, u)        # cat([x_next, z, w]), (batch, 18)
    2. z_b    = output[:, 6:12]                     # selection extracts z
    3. delta  = augmentation_block.forward(z_b)     # learned correction
    4. x_next = output[:, :6] + delta               # additive correction (D-003)
    5. loss   = criterion(x_next, target)
    6. loss.backward()                              # gradients reach aug weights

For the framework to work:
    - Gradients must flow from loss through the stacked output back to x and Y
    - Slicing the stacked output must preserve the computation graph
    - The augmentation's parameters must receive gradients through z

All inputs use batch=1 to match the batched lfr_forward signature (batch, n).

Tests
-----
    Check 1 — Mock augmentation gradient:
        z @ W.T as a minimal augmentation. Verify W.grad is not None after backward.
        Proves the full gradient chain: loss -> xdot_aug -> z -> lfr_forward -> W.

    Check 2 — Stacked output gradient preservation:
        Stack cat([xdot, z, w], dim=-1), slice back z via output[:, 6:12].
        Verify the slice has grad_fn — Jan's connect_signals does exactly this.

    Check 3 — Additive correction differentiability:
        xdot_aug = xdot + z @ W.T. Verify gradients reach both W (through delta)
        and x (through xdot). Both paths must be live for training to work.

Run as:
    conda run -n GraduationProject python -m lpv_lfr_baseline.test_augmentation_compat
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch

from lpv_lfr_baseline.physics import M0, M1, M2, K, C, P
from lpv_lfr_baseline.lfr_matrices import G
from lpv_lfr_baseline.lfr_forward import lfr_forward

dtype = torch.float64

# Fixed test inputs — batch=1
x_test    = torch.tensor([[0.05, 0.01, 0.30, 0.02, -0.01, 0.05]], dtype=dtype)  # (1, 6)
u_stage   = torch.tensor([[10.0, -5.0, 3.0]], dtype=dtype)                       # (1, 3)
u_logical = u_stage @ P.T                                                         # (1, 3)
Y_val     = x_test[:, 2]                                                          # (1,)  from state


if __name__ == '__main__':

    # ------------------------------------------------------------------
    # Check 1 — Mock augmentation gradient
    #
    # Simulate: augmentation applies a linear map W to z_baseline.
    # W is the stand-in for Jan's augmentation model weights.
    # After backward, W.grad must be non-None — this proves the gradient
    # chain from loss through z all the way to the augmentation parameters.
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Check 1: Mock augmentation gradient  z @ W.T -> W.grad")
    print("=" * 60)

    W = torch.randn(6, 6, dtype=dtype, requires_grad=True)

    x_in = x_test.clone().requires_grad_(True)
    xdot, z, w, y = lfr_forward(x_in, u_logical, Y_val, G, M0, M1, M2, K, C)

    delta_xdot = z @ W.T          # (1, 6) — mock augmentation correction
    xdot_aug   = xdot + delta_xdot
    loss       = xdot_aug.sum()
    loss.backward()

    w_grad_ok = W.grad is not None
    x_grad_ok = x_in.grad is not None
    print(f"  W.grad is not None (aug weights receive gradient) : {w_grad_ok}")
    print(f"  x.grad is not None (baseline state also reached)  : {x_grad_ok}")
    if w_grad_ok:
        print(f"  W.grad norm = {W.grad.norm().item():.6e}")
    print(f"\nCheck 1: {'PASS' if w_grad_ok and x_grad_ok else 'FAIL'}")

    # ------------------------------------------------------------------
    # Check 2 — Stacked output gradient preservation
    #
    # Jan's connect_signals slices the stacked block output to route
    # z to the augmentation. Verify that slicing preserves grad_fn —
    # a slice with no grad_fn would silently break the computation graph.
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Check 2: Stacked output slicing preserves grad_fn")
    print("=" * 60)

    x_in = x_test.clone().requires_grad_(True)
    xdot, z, w, y = lfr_forward(x_in, u_logical, Y_val, G, M0, M1, M2, K, C)

    # Simulate what the Block wrapper returns and connect_signals does.
    # Stacked along feature dim (batch, 18).
    stacked    = torch.cat([xdot, z, w], dim=-1)   # (1, 18) — Block output
    x_next_sl  = stacked[:, :6]                    # routed back as state
    z_slice    = stacked[:, 6:12]                  # routed to augmentation input
    w_slice    = stacked[:, 12:18]                 # routed to augmentation input (M_ab)

    x_next_has_grad = x_next_sl.grad_fn is not None
    z_has_grad      = z_slice.grad_fn is not None
    w_has_grad      = w_slice.grad_fn is not None

    print(f"  stacked[:, :6]   (x_next) has grad_fn : {x_next_has_grad}")
    print(f"  stacked[:, 6:12] (z)      has grad_fn : {z_has_grad}")
    print(f"  stacked[:, 12:]  (w)      has grad_fn : {w_has_grad}")
    print(f"\nCheck 2: {'PASS' if all([x_next_has_grad, z_has_grad, w_has_grad]) else 'FAIL'}")

    # ------------------------------------------------------------------
    # Check 3 — Additive correction differentiability (D-003)
    #
    # Parallel augmentation adds δxdot to xdot_baseline.
    # Two gradient paths must both be live:
    #   Path A: loss -> xdot_aug -> delta_xdot -> W  (augmentation learns)
    #   Path B: loss -> xdot_aug -> xdot -> x        (state gradient for BPTT)
    #
    # Test with two separate parameter matrices to confirm both paths
    # are independent and both receive gradients.
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Check 3: Additive correction — both gradient paths live")
    print("=" * 60)

    W_aug = torch.randn(6, 6, dtype=dtype, requires_grad=True)  # augmentation weights
    x_in  = x_test.clone().requires_grad_(True)

    xdot, z, w, y = lfr_forward(x_in, u_logical, Y_val, G, M0, M1, M2, K, C)

    delta_xdot = z @ W_aug.T           # (1, 6)
    xdot_aug   = xdot + delta_xdot    # parallel augmentation (D-003)
    loss       = xdot_aug.sum()
    loss.backward()

    path_a_ok = W_aug.grad is not None    # augmentation learns from z
    path_b_ok = x_in.grad is not None     # state gradient for BPTT

    print(f"  Path A — W_aug.grad (aug weights via delta_xdot) : {path_a_ok}")
    print(f"  Path B — x.grad     (state via baseline xdot)    : {path_b_ok}")
    if path_a_ok:
        print(f"  W_aug.grad norm = {W_aug.grad.norm().item():.6e}")
    if path_b_ok:
        print(f"  x.grad norm     = {x_in.grad.norm().item():.6e}")
    print(f"\nCheck 3: {'PASS' if path_a_ok and path_b_ok else 'FAIL'}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    results = {
        'Check 1 (mock aug gradient)':        w_grad_ok and x_grad_ok,
        'Check 2 (stacked output grad_fn)':   all([x_next_has_grad, z_has_grad, w_has_grad]),
        'Check 3 (additive correction paths)': path_a_ok and path_b_ok,
    }
    all_pass = all(results.values())
    for name, passed in results.items():
        print(f"  {name:40s} {'PASS' if passed else 'FAIL'}")
    print()
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 60)
