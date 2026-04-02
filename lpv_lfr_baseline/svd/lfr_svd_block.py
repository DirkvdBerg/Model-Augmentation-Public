"""
lfr_svd_block.py
----------------
Jan-compatible Block wrapper for the reduced dual-gantry LPV-LFR realization.

This is the fully reduced interface for the SVD path:

    nz = 9   (x:6 + u_stage:3)
    nw = 14  (x_next:6 + z_tilde:4 + w_tilde:4)

The reduced latent signals are abstract 4-channel coordinates of the exact
SVD-reduced LPV-LFR, not the original helper signals [v; Y*v] and [Y*v; Y^2*v].
"""

import torch
from torch import Tensor

from lpv_lfr_baseline.physics import P, ts
from lpv_lfr_baseline.svd.lfr_svd_reduction import G_reduced
from lpv_lfr_baseline.svd.lfr_svd_simulate import rk4_step

try:
    from model_augmentation.fit_systems.blocks import Block as _JanBlock

    _BASE = _JanBlock
except ImportError:
    import torch.nn as nn

    _BASE = nn.Module


class LFRReducedBlock(_BASE):
    """
    Stateless reduced LPV-LFR block compatible with Jan's Interconnect.

    nz = 9  : input  = cat([x (6), u_stage (3)], dim=1)
    nw = 14 : output = cat([x_next (6), z_tilde (4), w_tilde (4)], dim=1)
    """

    def __init__(self, *args, **kwargs):
        if _BASE.__name__ == "Block":
            super().__init__(nz=9, nw=14, *args, **kwargs)
        else:
            super().__init__(*args, **kwargs)
            self.nz = 9
            self.nw = 14

        self._G = G_reduced
        self._P = P
        self._ts = ts

    def forward(self, z_in: Tensor) -> Tensor:
        """
        One RK4 step through the reduced LPV-LFR block.

        Parameters
        ----------
        z_in : (batch, 9, 1) float32
            cat([x_logical (6), u_stage (3)], dim=1)

        Returns
        -------
        w_out : (batch, 14, 1) float32
            cat([x_next (6), z_tilde (4), w_tilde (4)], dim=1)
        """
        z_f64 = z_in.squeeze(-1).double()
        x = z_f64[:, :6]
        u_stage = z_f64[:, 6:]

        u_logical = u_stage @ self._P.T

        x_next, z_lfr, w_lfr, _ = rk4_step(x, u_logical, self._G, self._ts)

        w_f64 = torch.cat([x_next, z_lfr, w_lfr], dim=-1)
        return w_f64.float().unsqueeze(-1)


# Backward-compatible local alias for callers that still expect the old class
# name from the SVD subpackage.
LFRBaselineBlock = LFRReducedBlock


# ----------------------------------------------------------------------
# Verification
# Run as: python -m lpv_lfr_baseline.svd.lfr_svd_block
# ----------------------------------------------------------------------
if __name__ == "__main__":
    dtype = torch.float64

    block = LFRReducedBlock()

    x_test = torch.tensor([[0.05, 0.01, 0.30, 0.02, -0.01, 0.05]], dtype=torch.float32)
    u_test = torch.tensor([[10.0, -5.0, 3.0]], dtype=torch.float32)
    z_in = torch.cat([x_test, u_test], dim=1).unsqueeze(-1)

    print("=" * 60)
    print("Check 1: Output shape and dtype")
    print("=" * 60)

    with torch.no_grad():
        w_out = block.forward(z_in)

    shape_ok = w_out.shape == (1, 14, 1)
    dtype_ok = w_out.dtype == torch.float32
    print(f"  Output shape (1, 14, 1) : {shape_ok}  got {tuple(w_out.shape)}")
    print(f"  Output dtype float32    : {dtype_ok}  got {w_out.dtype}")
    print(f"\nCheck 1: {'PASS' if shape_ok and dtype_ok else 'FAIL'}")

    print()
    print("=" * 60)
    print("Check 2: Physical consistency - block vs direct rk4_step")
    print("=" * 60)

    x_f64 = x_test.double()
    u_stage_f64 = u_test.double()
    u_logical_f64 = u_stage_f64 @ P.T

    with torch.no_grad():
        x_next_ref, z_lfr_ref, w_lfr_ref, _ = rk4_step(
            x_f64, u_logical_f64, G_reduced, ts
        )
        w_out = block.forward(z_in)

    x_next_block = w_out[0, :6, 0].double()

    err = (x_next_block - x_next_ref[0]).abs().max().item()
    tol = 1e-6
    status = "PASS" if err < tol else "FAIL"
    print(f"  Max |x_next error| (block vs rk4_step) : {err:.2e}   {status}")
    print(f"  (Expected: float32 rounding ~1e-7, tolerance {tol:.0e})")
    print(f"\nCheck 2: {status}")

    print()
    print("=" * 60)
    print("Check 3: Autograd - gradient flows through block")
    print("=" * 60)

    z_grad = z_in.clone().float().requires_grad_(True)
    w_out = block.forward(z_grad)
    w_out.sum().backward()

    grad_ok = z_grad.grad is not None
    print(f"  Backward pass succeeded   : {grad_ok}")
    if grad_ok:
        print(f"  z_in.grad norm            : {z_grad.grad.norm().item():.6e}")
    print(f"\nCheck 3: {'PASS' if grad_ok else 'FAIL'}")

    print()
    print("=" * 60)
    print("Check 4: Stateless - repeated calls with same input are identical")
    print("=" * 60)

    with torch.no_grad():
        out_a = block.forward(z_in)
        out_b = block.forward(z_in)

    identical = torch.equal(out_a, out_b)
    print(f"  forward(z) == forward(z) on second call : {identical}")
    print(f"\nCheck 4: {'PASS' if identical else 'FAIL'}")

    print()
    print("=" * 60)
    print("Check 5: Output slot contract  [x_next | z_tilde | w_tilde]")
    print("=" * 60)

    with torch.no_grad():
        w_out = block.forward(z_in)

    x_next_out = w_out[0, :6, 0]
    z_lfr_out = w_out[0, 6:10, 0]
    w_lfr_out = w_out[0, 10:, 0]

    nonzero_z = z_lfr_out.abs().max().item() > 0
    nonzero_w = w_lfr_out.abs().max().item() > 0

    print(f"  x_next slot non-trivial   : {x_next_out.abs().max().item():.4e}")
    print(f"  z_tilde slot non-zero     : {nonzero_z}  (max {z_lfr_out.abs().max().item():.4e})")
    print(f"  w_tilde slot non-zero     : {nonzero_w}  (max {w_lfr_out.abs().max().item():.4e})")
    status = "PASS" if nonzero_z and nonzero_w else "FAIL"
    print(f"\nCheck 5: {status}")

    print()
    print("=" * 60)
    print(f"nz={block.nz}, nw={block.nw}  |  Base class: {_BASE.__name__}")
    print("=" * 60)
