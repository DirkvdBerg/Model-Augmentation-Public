"""
test_augmentation_compat.py
---------------------------
Compatibility tests for Jan's augmentation framework integration.

Verifies gradient flow through the stacked block output without importing
model_augmentation/. Tests simulate what Jan's training loop does internally.

Checks:
    1. Additive augmentation (z @ W.T + xdot): both gradient paths live
       (Path A: W receives grad through z, Path B: x receives grad through xdot).
    2. Stacked output slicing preserves grad_fn (required for connect_signals).

Run as:
    conda run -n GraduationProject python -m lpv_lfr_baseline.test_augmentation_compat
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch

from lpv_lfr_baseline.core.physics import (
    M1, M2, K, C, P, build_poly_constants,
    mh as _mh, m1 as _m1, m2 as _m2, mb as _mb, Jb as _Jb, Jh as _Jh,
    Lb as _Lb, d as _d,
)
from lpv_lfr_baseline.core.lfr_matrices import build_G_matrix
from lpv_lfr_baseline.core.lfr_forward import lfr_forward

dtype = torch.float64

# Fixed test inputs — batch=1
x_test    = torch.tensor([[0.05, 0.01, 0.30, 0.02, -0.01, 0.05]], dtype=dtype)  # (1, 6)
u_stage   = torch.tensor([[10.0, -5.0, 3.0]], dtype=dtype)                       # (1, 3)
u_logical = u_stage @ P.T                                                         # (1, 3)
Y_val     = x_test[:, 2]                                                          # (1,)  from state

# G and polynomial constants from fixed physics params
_alpha, _beta, _gamma, _N0, _N1, _N2 = build_poly_constants(
    _m1, _m2, _mb, _mh, _Jb, _Jh, _Lb, _d
)
_d0 = _mh * (_alpha * _gamma - _beta ** 2)
_G  = build_G_matrix(_N0, _d0, M1, M2, K, C)


if __name__ == '__main__':

    # ------------------------------------------------------------------
    # Check 1 -- Additive augmentation: both gradient paths live
    #
    # Parallel augmentation adds delta_xdot = z @ W.T to xdot_baseline.
    # Two gradient paths must both be live:
    #   Path A: loss -> xdot_aug -> delta_xdot -> W  (augmentation learns)
    #   Path B: loss -> xdot_aug -> xdot -> x        (state gradient for BPTT)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Check 1: Additive augmentation gradient paths (A: W, B: x)")
    print("=" * 60)

    W_aug = torch.randn(6, 6, dtype=dtype, requires_grad=True)
    x_in  = x_test.clone().requires_grad_(True)

    xdot, z, w, y = lfr_forward(x_in, u_logical, Y_val, _G, K, C, _mh, _alpha, _beta, _gamma, _N0, _N1, _N2)

    delta_xdot = z @ W_aug.T
    xdot_aug   = xdot + delta_xdot
    loss       = xdot_aug.sum()
    loss.backward()

    path_a_ok = W_aug.grad is not None
    path_b_ok = x_in.grad is not None

    print(f"  Path A: W_aug.grad (aug weights via z) : {path_a_ok}")
    print(f"  Path B: x.grad     (state via xdot)    : {path_b_ok}")
    if path_a_ok:
        print(f"  W_aug.grad norm = {W_aug.grad.norm().item():.6e}")
    if path_b_ok:
        print(f"  x.grad norm     = {x_in.grad.norm().item():.6e}")
    print(f"\nCheck 1: {'PASS' if path_a_ok and path_b_ok else 'FAIL'}")

    # ------------------------------------------------------------------
    # Check 2 -- Stacked output gradient preservation
    #
    # Jan's connect_signals slices the stacked block output to route
    # z to the augmentation. Verify that slicing preserves grad_fn.
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Check 2: Stacked output slicing preserves grad_fn")
    print("=" * 60)

    x_in = x_test.clone().requires_grad_(True)
    xdot, z, w, y = lfr_forward(x_in, u_logical, Y_val, _G, K, C, _mh, _alpha, _beta, _gamma, _N0, _N1, _N2)

    stacked    = torch.cat([xdot, z, w], dim=-1)   # (1, 18)
    x_next_sl  = stacked[:, :6]
    z_slice    = stacked[:, 6:12]
    w_slice    = stacked[:, 12:18]

    x_next_has_grad = x_next_sl.grad_fn is not None
    z_has_grad      = z_slice.grad_fn is not None
    w_has_grad      = w_slice.grad_fn is not None

    print(f"  stacked[:, :6]   (x_next) has grad_fn : {x_next_has_grad}")
    print(f"  stacked[:, 6:12] (z)      has grad_fn : {z_has_grad}")
    print(f"  stacked[:, 12:]  (w)      has grad_fn : {w_has_grad}")
    all_grad = x_next_has_grad and z_has_grad and w_has_grad
    print(f"\nCheck 2: {'PASS' if all_grad else 'FAIL'}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    results = {
        'Check 1 (aug gradient paths A+B)':  path_a_ok and path_b_ok,
        'Check 2 (stacked output grad_fn)':  all_grad,
    }
    all_pass = all(results.values())
    for name, passed in results.items():
        print(f"  {name:40s} {'PASS' if passed else 'FAIL'}")
    print()
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 60)
