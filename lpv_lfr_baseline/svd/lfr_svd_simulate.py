"""
lfr_svd_simulate.py
-------------------
RK4 single-step function and standalone simulation loop for the reduced
dual-gantry LPV-LFR realization.

This module is the SVD-reduced counterpart of lfr_simulate.py. The external
state and output dimensions stay the same, but the internal latent interface is
reduced from 6 channels to 4 channels:

    z_tilde in R^4,   w_tilde in R^4

Provides two functions:

    rk4_step(x, u_logical, G, ts) -> (x_next, z_tilde, w_tilde, y)
    simulate(x0, u_seq_stage, G, P, ts) -> SimResult

All tensors carry a leading batch dimension.
"""

from dataclasses import dataclass

import torch

from lpv_lfr_baseline.svd.lfr_svd_forward import lfr_forward
from lpv_lfr_baseline.svd.lfr_svd_reduction import GMatrixReduced


@dataclass
class SimResult:
    """
    Trajectory output from simulate().

    X : (batch, N+1, 6) state trajectory in logical coordinates
    Y : (batch, N,   3) output trajectory in stage coordinates
    Z : (batch, N,   4) reduced latent z recorded at start of each step
    W : (batch, N,   4) reduced latent w recorded at start of each step
    """

    X: torch.Tensor
    Y: torch.Tensor
    Z: torch.Tensor
    W: torch.Tensor


def rk4_step(
    x: torch.Tensor,         # (batch, 6)
    u_logical: torch.Tensor, # (batch, 3)
    G: GMatrixReduced,
    ts: torch.Tensor,        # ()
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Single RK4 step. Returns (x_next, z_tilde, w_tilde, y_logical).

    Y is extracted from the state at each sub-step. The reduced latent signals
    z_tilde and w_tilde are recorded at the start of the step.
    """
    k1, z, w, y = lfr_forward(x, u_logical, x[:, 2], G)

    x2 = x + (ts / 2) * k1
    k2, _, _, _ = lfr_forward(x2, u_logical, x2[:, 2], G)

    x3 = x + (ts / 2) * k2
    k3, _, _, _ = lfr_forward(x3, u_logical, x3[:, 2], G)

    x4 = x + ts * k3
    k4, _, _, _ = lfr_forward(x4, u_logical, x4[:, 2], G)

    x_next = x + (ts / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return x_next, z, w, y


def simulate(
    x0: torch.Tensor,          # (batch, 6)
    u_seq_stage: torch.Tensor, # (batch, N, 3)
    G: GMatrixReduced,
    P: torch.Tensor,           # (3, 3)
    ts: torch.Tensor,
) -> SimResult:
    """
    Simulate N steps using RK4. Returns SimResult.

    u_seq_stage is in stage coordinates; P transforms it into logical
    coordinates at each step. Output Y is converted back to stage coordinates.
    """
    N = u_seq_stage.shape[1]

    X_list = [x0]
    Y_list = []
    Z_list = []
    W_list = []

    x = x0
    for k in range(N):
        u_logical = u_seq_stage[:, k, :] @ P.T
        x_next, z_k, w_k, y_k = rk4_step(x, u_logical, G, ts)

        Y_list.append(y_k @ P)
        Z_list.append(z_k)
        W_list.append(w_k)
        X_list.append(x_next)
        x = x_next

    return SimResult(
        X=torch.stack(X_list, dim=1),
        Y=torch.stack(Y_list, dim=1),
        Z=torch.stack(Z_list, dim=1),
        W=torch.stack(W_list, dim=1),
    )


# ----------------------------------------------------------------------
# Verification
# Run as: python -m lpv_lfr_baseline.svd.lfr_svd_simulate
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import os

    from scipy.io import loadmat

    from lpv_lfr_baseline.physics import P, ts
    from lpv_lfr_baseline.svd.lfr_svd_reduction import G_reduced

    dtype = torch.float64
    mat_base = os.path.join(os.path.dirname(__file__), "..", "..", "Matlab-output")

    print("=" * 60)
    print("Check 1: rk4_step shapes and basic sanity  (batch=1)")
    print("=" * 60)

    x0_test = torch.zeros(1, 6, dtype=dtype)
    u_test = torch.tensor([[1.0, -0.5, 0.2]], dtype=dtype)
    u_logical = u_test @ P.T

    x_next, z_k, w_k, y_k = rk4_step(x0_test, u_logical, G_reduced, ts)

    shape_ok = (
        x_next.shape == (1, 6)
        and z_k.shape == (1, 4)
        and w_k.shape == (1, 4)
        and y_k.shape == (1, 3)
    )
    evolves = not torch.allclose(x_next, x0_test)
    print(f"  Output shapes correct : {shape_ok}")
    print(f"  State evolves (x_next != x0) : {evolves}")
    print(f"  x_next[0] = {x_next[0].detach().numpy()}")
    print(f"\nCheck 1: {'PASS' if shape_ok and evolves else 'FAIL'}")

    print()
    print("=" * 60)
    print("Check 2: rk4_step autograd  (batch=1)")
    print("=" * 60)

    x_grad = x0_test.clone().requires_grad_(True)
    x_next_g, _, _, _ = rk4_step(x_grad, u_logical, G_reduced, ts)
    x_next_g.sum().backward()

    grad_ok = x_grad.grad is not None
    print(f"  Backward pass succeeded : {grad_ok}")
    if grad_ok:
        print(f"  x.grad norm = {x_grad.grad.norm().item():.6e}")
    print(f"\nCheck 2: {'PASS' if grad_ok else 'FAIL'}")

    print()
    print("=" * 60)
    print("Check 3: Trajectory vs MATLAB lsim (gantry_q3_lsim.mat)")
    print("=" * 60)

    lsim_path = os.path.join(mat_base, "gantry_q3_lsim.mat")
    input_path = os.path.join(mat_base, "gantry_input.mat")

    if os.path.exists(lsim_path) and os.path.exists(input_path):
        mat_lsim = loadmat(lsim_path)
        mat_input = loadmat(input_path)

        q3_ref = torch.tensor(mat_lsim["q3"], dtype=dtype)
        u_matlab = torch.tensor(mat_input["u"], dtype=dtype)
        N_sim = u_matlab.shape[0]

        x0 = torch.tensor([[0.0, 0.0, 0.3, 0.0, 0.0, 0.0]], dtype=dtype)
        with torch.no_grad():
            result = simulate(x0, u_matlab.unsqueeze(0), G_reduced, P, ts)

        Y_ours = result.Y[0]

        err_max = (Y_ours - q3_ref).abs().max().item()
        err_mean = (Y_ours - q3_ref).abs().mean().item()
        print(f"  N steps simulated     : {N_sim}")
        print(f"  Max  |error| [m]      : {err_max:.4e}")
        print(f"  Mean |error| [m]      : {err_mean:.4e}")
        print("  (Differences expected: RK4 vs ZOH, Y self-scheduled vs frozen at 0.3)")
        print("\nCheck 3: INFORMATIONAL (LPV vs frozen LTI - differences expected)")
    else:
        print("  SKIPPED - MATLAB reference files not found.")
        print(f"  Expected: {lsim_path}")

    print()
    print("=" * 60)
    print("Check 4: Varying-Y trajectory vs MATLAB CT quasi-LPV (lpv_sim_varying_y.mat)")
    print("=" * 60)

    lpv_path = os.path.join(mat_base, "lpv_sim_varying_y.mat")

    if os.path.exists(lpv_path):
        mat_lpv = loadmat(lpv_path)
        q1_ref = torch.tensor(mat_lpv["q1"], dtype=dtype)
        u_lpv = torch.tensor(mat_lpv["u_q1"], dtype=dtype)
        N_lpv = u_lpv.shape[0]

        x0_lpv = torch.tensor([[0.0, 0.0, 0.3, 0.0, 0.0, 0.0]], dtype=dtype)

        with torch.no_grad():
            result_lpv = simulate(x0_lpv, u_lpv.unsqueeze(0), G_reduced, P, ts)

        Y_ours_lpv = result_lpv.Y[0]

        err_max = (Y_ours_lpv - q1_ref).abs().max().item()
        err_mean = (Y_ours_lpv - q1_ref).abs().mean().item()
        print(f"  N steps simulated     : {N_lpv}")
        print(f"  Max  |error| [m]      : {err_max:.4e}")
        print(f"  Mean |error| [m]      : {err_mean:.4e}")
        print(
            f"  Y range in sim [m]    : "
            f"{result_lpv.X[0, :, 2].min().item():.3f} to "
            f"{result_lpv.X[0, :, 2].max().item():.3f}"
        )
        status = "PASS" if err_max < 1e-10 else "FAIL (larger than expected - check x0 or input)"
        print(f"\nCheck 4: {status}")
    else:
        print("  SKIPPED - MATLAB reference file not found.")
        print(f"  Expected: {lpv_path}")
