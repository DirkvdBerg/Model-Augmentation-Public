"""
lfr_forward.py
--------------
Resolve-and-retain forward pass for the dual-gantry LPV-LFR baseline.

Steps: M(Y) -> fnet -> v=solve(M,fnet) -> z=[v;Yv] -> w=[Yv;Y^2 v] -> xdot=[qdot;v].
All inputs/outputs have a leading batch dim, dtype=float64, logical coordinates.
Caller applies P transform for stage coords (see lfr_simulate.py).
"""

import torch


def lfr_forward(
    x:  torch.Tensor,   # (batch, 6)   state in logical coordinates
    u:  torch.Tensor,   # (batch, 3)   input in logical coordinates
    Y:  torch.Tensor,   # (batch,)     scheduling variable — x[:, 2] in caller
    M0: torch.Tensor,   # (3,3)
    M1: torch.Tensor,   # (3,3)
    M2: torch.Tensor,   # (3,3)
    K:  torch.Tensor,   # (3,3)
    C:  torch.Tensor,   # (3,3)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resolve-and-retain forward pass. Returns (xdot, z, w, y)."""
    # Step 1: M(Y) for each item in the batch  ->  (batch, 3, 3)
    Y_e = Y[:, None, None]
    M_Y = M0.unsqueeze(0) + M1.unsqueeze(0) * Y_e + M2.unsqueeze(0) * Y_e ** 2

    # Step 2: net force  ->  (batch, 3)
    fnet = -(x[:, :3] @ K.T) - (x[:, 3:] @ C.T) + u

    # Step 3: v = M(Y)^{-1} fnet  (batched solve)
    v = torch.linalg.solve(M_Y, fnet.unsqueeze(-1)).squeeze(-1)

    # Steps 4-5: LFR latent signals, Delta(Y) = Y*I6
    v1 = Y[:, None] * v
    v2 = Y[:, None] * v1
    z = torch.cat([v,  v1], dim=-1)    # (batch, 6)
    w = torch.cat([v1, v2], dim=-1)    # (batch, 6)

    # Step 6: xdot = [qdot; qddot] direct from physics (D-026)
    xdot = torch.cat([x[:, 3:], v], dim=-1)

    # Step 7: output = logical positions
    y = x[:, :3]

    return xdot, z, w, y


# ----------------------------------------------------------------------
# Verification  (run as: conda run -n GraduationProject python -m lpv_lfr_baseline.lfr_forward)
# ----------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    from lpv_lfr_baseline.physics import M0, M1, M2, K, C, P, build_M

    dtype = torch.float64

    # Fixed test inputs — chosen to exercise all non-zero entries.
    # Batched: 5 Y values tested simultaneously to exercise batch dimension.
    torch.manual_seed(0)
    x_test    = torch.tensor([0.05, 0.01, 0.30, 0.02, -0.01, 0.05], dtype=dtype)
    u_stage   = torch.tensor([10.0, -5.0, 3.0], dtype=dtype)
    u_logical = P @ u_stage

    test_Y_vals = [0.0, 0.1, 0.3, -0.2, 0.35]
    nb          = len(test_Y_vals)

    # Batch inputs: same x and u for all Y values
    Y_batch = torch.tensor(test_Y_vals, dtype=dtype)               # (5,)
    x_batch = x_test.unsqueeze(0).expand(nb, -1).clone()           # (5, 6)
    u_batch = u_logical.unsqueeze(0).expand(nb, -1).clone()        # (5, 3)

    # ------------------------------------------------------------------
    # Check 1 — Loop resolution residual: M(Y) @ v - fnet < 1e-12
    #
    # Calls lfr_forward once with batch=5 (all Y values simultaneously).
    # Verifies each item's residual independently.
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Check 1: Loop resolution residual  M(Y)@v - fnet  (batch=5)")
    print("=" * 60)
    xdot_b, z_b, w_b, y_b = lfr_forward(x_batch, u_batch, Y_batch, M0, M1, M2, K, C)

    all_pass = True
    fnet_ref = -K @ x_test[:3] - C @ x_test[3:] + u_logical   # same for all Y
    for i, y_val in enumerate(test_Y_vals):
        M_Y_ref  = build_M(torch.tensor(y_val, dtype=dtype))
        v_i      = z_b[i, :3]   # first half of z[i] is v
        residual = (M_Y_ref @ v_i - fnet_ref).abs().max().item()
        status   = 'PASS' if residual < 1e-12 else 'FAIL'
        if status == 'FAIL':
            all_pass = False
        print(f"  Y = {y_val:+.2f} m   residual = {residual:.2e}   {status}")
    print(f"\nCheck 1: {'ALL PASS' if all_pass else 'SOME FAILED'}\n")

    # ------------------------------------------------------------------
    # Check 2 — CT vector field: xdot vs collapsed A_c(Y)@x + B_c(Y)@u
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Check 2: xdot vs collapsed A_c(Y)@x + B_c(Y)@u  (batch=5)")
    print("=" * 60)
    all_pass = True
    eye3 = torch.eye(3, dtype=dtype)
    z33  = torch.zeros(3, 3, dtype=dtype)
    for i, y_val in enumerate(test_Y_vals):
        M_Y_ref = build_M(torch.tensor(y_val, dtype=dtype))
        MYinvK  = torch.linalg.solve(M_Y_ref, K)
        MYinvC  = torch.linalg.solve(M_Y_ref, C)
        MYinv   = torch.linalg.solve(M_Y_ref, eye3)

        A_c = torch.cat([
            torch.cat([z33,      eye3    ], dim=1),
            torch.cat([-MYinvK, -MYinvC  ], dim=1),
        ], dim=0)
        B_c = torch.cat([z33, MYinv], dim=0)

        xdot_ref = A_c @ x_test + B_c @ u_logical
        err      = (xdot_b[i] - xdot_ref).abs().max().item()
        status   = 'PASS' if err < 1e-12 else 'FAIL'
        if status == 'FAIL':
            all_pass = False
        print(f"  Y = {y_val:+.2f} m   max|xdot error| = {err:.2e}   {status}")
    print(f"\nCheck 2: {'ALL PASS' if all_pass else 'SOME FAILED'}\n")

    # ------------------------------------------------------------------
    # Check 3 — Autograd: gradient flows through M(Y)^{-1} back to Y
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Check 3: Autograd — gradient flows through solve to Y  (batch=1)")
    print("=" * 60)
    Y_grad    = torch.tensor([0.3], dtype=dtype, requires_grad=True)   # (1,)
    x_b1      = x_test.unsqueeze(0)                                     # (1, 6)
    u_b1      = u_logical.unsqueeze(0)                                  # (1, 3)
    xdot_g, _, _, _ = lfr_forward(x_b1, u_b1, Y_grad, M0, M1, M2, K, C)
    xdot_g.sum().backward()

    grad_ok = Y_grad.grad is not None
    print(f"  Backward pass succeeded : {grad_ok}")
    if grad_ok:
        print(f"  dL/dY = {Y_grad.grad[0].item():.6e}")
    print(f"\nCheck 3: {'PASS' if grad_ok else 'FAIL'}")

    # ------------------------------------------------------------------
    # Check 4 — Autograd: gradient flows through xdot back to x
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Check 4: Autograd — gradient flows through xdot to x  (batch=1)")
    print("=" * 60)
    x_grad = x_test.unsqueeze(0).clone().requires_grad_(True)   # (1, 6)
    Y_b1   = torch.tensor([0.3], dtype=dtype)                   # (1,)
    xdot_g, _, _, _ = lfr_forward(x_grad, u_b1, Y_b1, M0, M1, M2, K, C)
    xdot_g.sum().backward()

    grad_ok = x_grad.grad is not None
    print(f"  Backward pass succeeded : {grad_ok}")
    if grad_ok:
        print(f"  dx/dx (norm) = {x_grad.grad.norm().item():.6e}")
    print(f"\nCheck 4: {'PASS' if grad_ok else 'FAIL'}")

    # ------------------------------------------------------------------
    # Check 5 — LFR signal structure: w = Y * I6 * z  (Δ(Y) = Y·I6)
    #
    # Verified entry-wise across the full batch.
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Check 5: LFR signal structure  w = Y·I6·z  (batch=5)")
    print("=" * 60)
    _, z_b5, w_b5, _ = lfr_forward(x_batch, u_batch, Y_batch, M0, M1, M2, K, C)

    all_pass = True
    for i, y_val in enumerate(test_Y_vals):
        err    = (w_b5[i] - Y_batch[i] * z_b5[i]).abs().max().item()
        status = 'PASS' if err == 0.0 else 'FAIL'
        if status == 'FAIL':
            all_pass = False
        print(f"  Y = {y_val:+.2f} m   max|w - Y*z| = {err:.2e}   {status}")
    print(f"\nCheck 5: {'ALL PASS' if all_pass else 'SOME FAILED'}")
