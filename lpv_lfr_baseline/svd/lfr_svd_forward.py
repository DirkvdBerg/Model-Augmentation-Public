"""
lfr_svd_forward.py
------------------
Reduced forward pass for the dual-gantry LPV-LFR realization.

This module implements the exact 4-channel latent-loop realization produced by
the SVD reduction in lfr_svd_reduction.py. Unlike the original 6-channel LFR,
the reduced implementation does not construct the helper signals

    v, Y*v, Y^2*v

explicitly. Instead, it works directly with the reduced latent coordinates

    z_tilde in R^4,   w_tilde = Y * z_tilde in R^4

and resolves the algebraic loop

    z_tilde = Cz*x + Dzw*w_tilde + Dzu*u
    w_tilde = Y * z_tilde

as the batched linear solve

    (I4 - Y*Dzw) z_tilde = Cz*x + Dzu*u .

Provides:
    lfr_forward(x, u, Y, G) -> (xdot, z_tilde, w_tilde, y)

All tensors are expected in logical coordinates and float64.
"""

import torch

from lpv_lfr_baseline.svd.lfr_svd_reduction import GMatrixReduced


def lfr_forward(
    x: torch.Tensor,   # (batch, 6) state in logical coordinates
    u: torch.Tensor,   # (batch, 3) input in logical coordinates
    Y: torch.Tensor,   # (batch,)   scheduling variable, typically x[:, 2]
    G: GMatrixReduced,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reduced LPV-LFR forward pass. Returns (xdot, z_tilde, w_tilde, y).

    Inputs carry a leading batch dimension. The reduced latent dimension is 4.
    """
    dtype = x.dtype
    device = x.device

    Ax = G.Ax.to(dtype=dtype, device=device)
    Bw = G.Bw.to(dtype=dtype, device=device)
    Bu = G.Bu.to(dtype=dtype, device=device)
    Cz = G.Cz.to(dtype=dtype, device=device)
    Dzw = G.Dzw.to(dtype=dtype, device=device)
    Dzu = G.Dzu.to(dtype=dtype, device=device)
    Cy = G.Cy.to(dtype=dtype, device=device)

    rhs = x @ Cz.T + u @ Dzu.T  # (batch, 4)

    latent_dim = Dzw.shape[0]
    eye_latent = torch.eye(latent_dim, dtype=dtype, device=device)
    lhs = eye_latent.unsqueeze(0) - Y[:, None, None] * Dzw.unsqueeze(0)  # (batch, 4, 4)

    z_tilde = torch.linalg.solve(lhs, rhs.unsqueeze(-1)).squeeze(-1)  # (batch, 4)
    w_tilde = Y[:, None] * z_tilde                                     # (batch, 4)

    xdot = x @ Ax.T + w_tilde @ Bw.T + u @ Bu.T  # (batch, 6)
    y = x @ Cy.T                                  # (batch, 3)

    return xdot, z_tilde, w_tilde, y


# ----------------------------------------------------------------------
# Verification
# Run as: python -m lpv_lfr_baseline.svd.lfr_svd_forward
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from lpv_lfr_baseline.physics import K, C, P, build_M
    from lpv_lfr_baseline.svd.lfr_svd_reduction import G_reduced

    dtype = torch.float64

    torch.manual_seed(0)
    x_test = torch.tensor([0.05, 0.01, 0.30, 0.02, -0.01, 0.05], dtype=dtype)
    u_stage = torch.tensor([10.0, -5.0, 3.0], dtype=dtype)
    u_logical = P @ u_stage

    test_Y_vals = [0.0, 0.1, 0.3, -0.2, 0.35]
    nb = len(test_Y_vals)

    Y_batch = torch.tensor(test_Y_vals, dtype=dtype)
    x_batch = x_test.unsqueeze(0).expand(nb, -1).clone()
    u_batch = u_logical.unsqueeze(0).expand(nb, -1).clone()

    print("=" * 60)
    print("Check 1: Reduced loop residual  (I - Y*Dzw) z - (Cz*x + Dzu*u)")
    print("=" * 60)
    xdot_b, z_b, w_b, y_b = lfr_forward(x_batch, u_batch, Y_batch, G_reduced)

    eye4 = torch.eye(4, dtype=dtype)
    rhs_b = x_batch @ G_reduced.Cz.T + u_batch @ G_reduced.Dzu.T
    all_pass = True
    for i, y_val in enumerate(test_Y_vals):
        lhs_i = eye4 - Y_batch[i] * G_reduced.Dzw
        residual = (lhs_i @ z_b[i] - rhs_b[i]).abs().max().item()
        status = "PASS" if residual < 1e-12 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  Y = {y_val:+.2f} m   residual = {residual:.2e}   {status}")
    print(f"\nCheck 1: {'ALL PASS' if all_pass else 'SOME FAILED'}\n")

    print("=" * 60)
    print("Check 2: xdot vs collapsed A_c(Y)@x + B_c(Y)@u  (batch=5)")
    print("=" * 60)
    all_pass = True
    eye3 = torch.eye(3, dtype=dtype)
    z33 = torch.zeros(3, 3, dtype=dtype)
    for i, y_val in enumerate(test_Y_vals):
        M_Y_ref = build_M(torch.tensor(y_val, dtype=dtype))
        MYinvK = torch.linalg.solve(M_Y_ref, K)
        MYinvC = torch.linalg.solve(M_Y_ref, C)
        MYinv = torch.linalg.solve(M_Y_ref, eye3)

        A_c = torch.cat([
            torch.cat([z33, eye3], dim=1),
            torch.cat([-MYinvK, -MYinvC], dim=1),
        ], dim=0)
        B_c = torch.cat([z33, MYinv], dim=0)

        xdot_ref = A_c @ x_test + B_c @ u_logical
        err = (xdot_b[i] - xdot_ref).abs().max().item()
        status = "PASS" if err < 1e-10 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  Y = {y_val:+.2f} m   max|xdot error| = {err:.2e}   {status}")
    print(f"\nCheck 2: {'ALL PASS' if all_pass else 'SOME FAILED'}\n")

    print("=" * 60)
    print("Check 3: Autograd - gradient flows through reduced solve to Y")
    print("=" * 60)
    Y_grad = torch.tensor([0.3], dtype=dtype, requires_grad=True)
    x_b1 = x_test.unsqueeze(0)
    u_b1 = u_logical.unsqueeze(0)
    xdot_g, _, _, _ = lfr_forward(x_b1, u_b1, Y_grad, G_reduced)
    xdot_g.sum().backward()

    grad_ok = Y_grad.grad is not None
    print(f"  Backward pass succeeded : {grad_ok}")
    if grad_ok:
        print(f"  dL/dY = {Y_grad.grad[0].item():.6e}")
    print(f"\nCheck 3: {'PASS' if grad_ok else 'FAIL'}")

    print()
    print("=" * 60)
    print("Check 4: Autograd - gradient flows through xdot to x")
    print("=" * 60)
    x_grad = x_test.unsqueeze(0).clone().requires_grad_(True)
    Y_b1 = torch.tensor([0.3], dtype=dtype)
    xdot_g, _, _, _ = lfr_forward(x_grad, u_b1, Y_b1, G_reduced)
    xdot_g.sum().backward()

    grad_ok = x_grad.grad is not None
    print(f"  Backward pass succeeded : {grad_ok}")
    if grad_ok:
        print(f"  dx/dx (norm) = {x_grad.grad.norm().item():.6e}")
    print(f"\nCheck 4: {'PASS' if grad_ok else 'FAIL'}")

    print()
    print("=" * 60)
    print("Check 5: Reduced signal structure  w = Y*z  (batch=5)")
    print("=" * 60)
    all_pass = True
    for i, y_val in enumerate(test_Y_vals):
        err = (w_b[i] - Y_batch[i] * z_b[i]).abs().max().item()
        status = "PASS" if err == 0.0 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  Y = {y_val:+.2f} m   max|w - Y*z| = {err:.2e}   {status}")
    print(f"\nCheck 5: {'ALL PASS' if all_pass else 'SOME FAILED'}")
