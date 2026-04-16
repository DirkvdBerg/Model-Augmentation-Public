"""
lfr_block.py
------------
Jan-compatible Block wrapper for the dual-gantry LPV-LFR baseline.

Wraps rk4_step as a stateless Block subclass that speaks Jan's (batch, n, 1)
interface. This is the sole boundary between the physics implementation and
the model_augmentation framework.

Block interface (from model_augmentation/fit_systems/blocks.py):
    forward(z: Tensor) -> Tensor
    z shape: (batch, nz, 1)  ->  w shape: (batch, nw, 1)

For LFRBaselineBlock:
    nz = 9   (nx=6 state + nu=3 stage-coord input)
    nw = 18  (x_next=6 + z_lfr=6 + w_lfr=6)

    z_in[:, :6, :]   = x        — current state in logical coordinates
    z_in[:, 6:, :]   = u_stage  — input in stage coordinates

    w_out[:, :6, :]  = x_next   — next state (logical coords)
    w_out[:, 6:12,:] = z_lfr    — LFR latent z (routed to augmentation)
    w_out[:, 12:, :] = w_lfr    — LFR latent w (routed to augmentation)

Design notes:
    STATELESS — state is managed externally by Jan's Interconnect. The
    Interconnect routes x_next back into z_in at each step. Do NOT store
    self.x or update it inside forward().

    dtype boundary — Jan's framework uses float32; physics requires float64.
    float32 -> float64 at entry, float64 -> float32 at exit. All physics
    computation is in float64. To run the entire pipeline in float64, remove
    the two cast lines (marked below) — no other file needs changing.

    Naming — Jan's API uses 'z' for block input and 'w' for block output.
    Our LFR latent signals are renamed z_lfr / w_lfr inside this file to
    avoid confusion.

    Y scheduling — Y = x[:, 2] is extracted inside rk4_step from the state.
    Y is never a named external signal — this keeps the Interconnect free of
    algebraic loops (see README: Interconnect pitfalls).

    Physical parameters — G submatrices and polynomial constants are precomputed
    at construction from the fixed physics.py values and stored as buffers.
    They move automatically with .to(device) / .cuda(). G is not a trainable
    parameter here — see lfr_param_block.py for the trainable version.
"""

import torch
from torch import Tensor

from lpv_lfr_baseline.core.physics import (
    M1, M2, K, C, P, ts,
    mh as _mh, m1 as _m1, m2 as _m2, mb as _mb, Jb as _Jb, Jh as _Jh,
    Lb as _Lb, d as _d,
    build_poly_constants,
)
from lpv_lfr_baseline.core.lfr_matrices import GMatrix, build_G_matrix
from lpv_lfr_baseline.core.lfr_simulate import rk4_step

try:
    from model_augmentation.fit_systems.blocks import Block as _JanBlock
    _MODEL_AUG_AVAILABLE = True
    _BASE = _JanBlock
except ImportError:
    _MODEL_AUG_AVAILABLE = False
    _BASE = torch.nn.Module


class LFRBaselineBlock(_BASE):
    """
    Stateless LPV-LFR baseline block compatible with Jan's Interconnect.

    nz = 9  : input  = cat([x (6), u_stage (3)], dim=1)
    nw = 18 : output = cat([x_next (6), z_lfr (6), w_lfr (6)], dim=1)
    """

    def __init__(self, **kwargs):
        if _MODEL_AUG_AVAILABLE:
            super().__init__(nz=9, nw=18, **kwargs)
        else:
            super().__init__(**kwargs)
            self.nz = 9
            self.nw = 18

        # Build polynomial constants and G from fixed physics params.
        # Store as individual buffers so .to(device) / .cuda() moves them.
        # G uses N0/d0 = adj(M0)/det(M0) — no linalg.solve.
        alpha, beta, gamma, N0, N1, N2 = build_poly_constants(
            _m1, _m2, _mb, _mh, _Jb, _Jh, _Lb, _d
        )
        d0  = _mh * (alpha * gamma - beta ** 2)
        G   = build_G_matrix(N0, d0, M1, M2, K, C)

        # G submatrices
        self.register_buffer('_G_Ax',  G.Ax)
        self.register_buffer('_G_Bw',  G.Bw)
        self.register_buffer('_G_Bu',  G.Bu)
        self.register_buffer('_G_Cz',  G.Cz)
        self.register_buffer('_G_Dzw', G.Dzw)
        self.register_buffer('_G_Dzu', G.Dzu)
        self.register_buffer('_G_Cy',  G.Cy)

        # Polynomial constants for loop solve
        self.register_buffer('_mh',    _mh.clone())
        self.register_buffer('_alpha', alpha if isinstance(alpha, Tensor) else torch.tensor(alpha, dtype=torch.float64))
        self.register_buffer('_beta',  beta  if isinstance(beta,  Tensor) else torch.tensor(beta,  dtype=torch.float64))
        self.register_buffer('_gamma', gamma if isinstance(gamma, Tensor) else torch.tensor(gamma, dtype=torch.float64))
        self.register_buffer('_N0',    N0)
        self.register_buffer('_N1',    N1)
        self.register_buffer('_N2',    N2)

        # K, C needed by rk4_step for f_net computation
        self.register_buffer('_K',  K)
        self.register_buffer('_C',  C)

        # Coordinate transform and sample period
        self.register_buffer('_P',  P)
        self.register_buffer('_ts', ts)

    def _get_G(self) -> GMatrix:
        """Reconstruct GMatrix from stored buffers."""
        return GMatrix(
            Ax=self._G_Ax, Bw=self._G_Bw, Bu=self._G_Bu,
            Cz=self._G_Cz, Dzw=self._G_Dzw, Dzu=self._G_Dzu,
            Cy=self._G_Cy,
        )

    def forward(self, z_in: Tensor) -> Tensor:
        """One RK4 step. (batch, 9, 1) -> (batch, 18, 1)."""
        in_dtype = z_in.dtype
        z_flat   = z_in.squeeze(-1)                                # (batch, 9)

        # Cast to float64 only if needed (Jan's framework uses float32; physics needs float64)
        if z_flat.dtype != torch.float64:
            z_flat = z_flat.double()

        x       = z_flat[:, :6]
        u_stage = z_flat[:, 6:]

        u_logical = u_stage @ self._P.T

        G = self._get_G()
        x_next, z_lfr, w_lfr, _ = rk4_step(
            x, u_logical,
            G, self._K, self._C,
            self._mh, self._alpha, self._beta, self._gamma,
            self._N0, self._N1, self._N2,
            self._ts,
        )

        w_f64 = torch.cat([x_next, z_lfr, w_lfr], dim=-1)        # (batch, 18)

        # Cast back to input dtype only if we changed it
        out = w_f64 if in_dtype == torch.float64 else w_f64.to(in_dtype)
        return out.unsqueeze(-1)


# ----------------------------------------------------------------------
# Verification  (run as: conda run -n GraduationProject python -m lpv_lfr_baseline.blocks.lfr_block)
# ----------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

    dtype = torch.float64

    block = LFRBaselineBlock()

    # Fixed test input: x at Y=0.3 m, small non-zero state; modest stage forces
    x_test   = torch.tensor([[0.05, 0.01, 0.30, 0.02, -0.01, 0.05]], dtype=torch.float32)  # (1, 6)
    u_test   = torch.tensor([[10.0, -5.0, 3.0]], dtype=torch.float32)                       # (1, 3)
    z_in     = torch.cat([x_test, u_test], dim=1).unsqueeze(-1)                             # (1, 9, 1)

    # ------------------------------------------------------------------
    # Check 1 — Output shape and dtype
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Check 1: Output shape and dtype")
    print("=" * 60)

    with torch.no_grad():
        w_out = block.forward(z_in)

    shape_ok = w_out.shape == (1, 18, 1)
    dtype_ok = w_out.dtype == torch.float32
    print(f"  Output shape (1, 18, 1) : {shape_ok}  got {tuple(w_out.shape)}")
    print(f"  Output dtype float32    : {dtype_ok}  got {w_out.dtype}")
    print(f"\nCheck 1: {'PASS' if shape_ok and dtype_ok else 'FAIL'}")

    # ------------------------------------------------------------------
    # Check 2 — Physical consistency
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Check 2: Physical consistency — block vs direct rk4_step")
    print("=" * 60)

    from lpv_lfr_baseline.core.physics import (
        M1, M2, K, C, P, ts, build_poly_constants,
        mh as _mh, m1 as _m1, m2 as _m2, mb as _mb,
        Jb as _Jb, Jh as _Jh, Lb as _Lb, d as _d,
    )
    from lpv_lfr_baseline.core.lfr_matrices import build_G_matrix
    from lpv_lfr_baseline.core.lfr_simulate import rk4_step as rk4_ref

    alpha_r, beta_r, gamma_r, N0_r, N1_r, N2_r = build_poly_constants(
        _m1, _m2, _mb, _mh, _Jb, _Jh, _Lb, _d
    )
    d0_r  = _mh * (alpha_r * gamma_r - beta_r ** 2)
    G_ref = build_G_matrix(N0_r, d0_r, M1, M2, K, C)

    x_f64        = x_test.double()
    u_stage_f64  = u_test.double()
    u_logical_f64 = u_stage_f64 @ P.T

    with torch.no_grad():
        x_next_ref, z_lfr_ref, w_lfr_ref, _ = rk4_ref(
            x_f64, u_logical_f64,
            G_ref, K, C, _mh, alpha_r, beta_r, gamma_r, N0_r, N1_r, N2_r, ts
        )
        w_out = block.forward(z_in)

    x_next_block = w_out[0, :6, 0].double()
    err = (x_next_block - x_next_ref[0]).abs().max().item()
    tol = 1e-6
    status = 'PASS' if err < tol else 'FAIL'
    print(f"  Max |x_next error| (block vs rk4_step) : {err:.2e}   {status}")
    print(f"  (Expected: float32 rounding ~1e-7, tolerance {tol:.0e})")
    print(f"\nCheck 2: {status}")

    # ------------------------------------------------------------------
    # Check 3 — Autograd
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Check 3: Autograd — gradient flows through block")
    print("=" * 60)

    z_grad = z_in.clone().float().requires_grad_(True)
    w_out  = block.forward(z_grad)
    w_out.sum().backward()

    grad_ok = z_grad.grad is not None
    print(f"  Backward pass succeeded   : {grad_ok}")
    if grad_ok:
        print(f"  z_in.grad norm            : {z_grad.grad.norm().item():.6e}")
    print(f"\nCheck 3: {'PASS' if grad_ok else 'FAIL'}")

    # ------------------------------------------------------------------
    # Check 4 — Stateless
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Check 4: Stateless — repeated calls with same input are identical")
    print("=" * 60)

    with torch.no_grad():
        out_a = block.forward(z_in)
        out_b = block.forward(z_in)

    identical = torch.equal(out_a, out_b)
    print(f"  forward(z) == forward(z) on second call : {identical}")
    print(f"\nCheck 4: {'PASS' if identical else 'FAIL'}")

    # ------------------------------------------------------------------
    # Check 5 — Output slot contract
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Check 5: Output slot contract  [x_next | z_lfr | w_lfr]")
    print("=" * 60)

    with torch.no_grad():
        w_out = block.forward(z_in)

    x_next_out = w_out[0, :6,  0]
    z_lfr_out  = w_out[0, 6:12, 0]
    w_lfr_out  = w_out[0, 12:, 0]

    nonzero_z = z_lfr_out.abs().max().item() > 0
    nonzero_w = w_lfr_out.abs().max().item() > 0

    print(f"  x_next slot non-trivial   : {x_next_out.abs().max().item():.4e}")
    print(f"  z_lfr  slot non-zero      : {nonzero_z}  (max {z_lfr_out.abs().max().item():.4e})")
    print(f"  w_lfr  slot non-zero      : {nonzero_w}  (max {w_lfr_out.abs().max().item():.4e})")
    status = 'PASS' if nonzero_z and nonzero_w else 'FAIL'
    print(f"\nCheck 5: {status}")

    print()
    print("=" * 60)
    print(f"nz={block.nz}, nw={block.nw}  |  Base class: {_BASE.__name__}")
    print("=" * 60)
