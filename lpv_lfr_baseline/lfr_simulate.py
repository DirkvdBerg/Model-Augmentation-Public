"""
lfr_simulate.py
---------------
RK4 single-step function and standalone simulation loop for the dual-gantry LPV-LFR baseline.

Discretization: RK4 with fixed step ts = 1/fs. Consistent with D-018.

Provides two functions:

    rk4_step(x, u_logical, G, M0, M1, M2, K, C, ts) -> (x_next, z, w, y)
        Single RK4 step in logical coordinates.
        Used by both simulate() and lfr_block.py (Jan's Block wrapper).
        Y is extracted from the state at each sub-step — self-scheduled.
        z and w are from the START of the step (x[k], not sub-steps).
        y is in logical coordinates — caller applies @ P for stage output.

    simulate(x0, u_seq_stage, G, M0, M1, M2, K, C, P, ts) -> SimResult
        Full trajectory simulation over N steps.
        x0           : (batch, 6)    initial state in logical coordinates
        u_seq_stage  : (batch, N, 3) input sequence in stage coordinates
        Returns SimResult with fields:
            X  : (batch, N+1, 6)  state trajectory (logical coordinates)
            Y  : (batch, N, 3)    output trajectory (stage coordinates)
            Z  : (batch, N, 6)    latent z at start of each step
            W  : (batch, N, 6)    latent w at start of each step

All inputs and outputs carry a leading batch dimension.
For single-trajectory use, add/remove the batch dim with unsqueeze(0)/squeeze(0).

RK4 sub-step schedule:
    k1, z, w, y = lfr_forward(x,             u_logical, x[:, 2],           ...)
    k2, _, _, _ = lfr_forward(x + ts/2 * k1, u_logical, (x+ts/2*k1)[:,2], ...)
    k3, _, _, _ = lfr_forward(x + ts/2 * k2, u_logical, (x+ts/2*k2)[:,2], ...)
    k4, _, _, _ = lfr_forward(x + ts   * k3, u_logical, (x+ts*k3)[:,2],   ...)
    x_next = x + ts/6 * (k1 + 2*k2 + 2*k3 + k4)

Note on RK4 vs ZOH:
    Validation against MATLAB (ZOH) will show small bounded differences at 16kHz.
    This is expected — RK4 and ZOH are different discretization methods.
    Do not switch to ZOH to match MATLAB exactly — RK4 is the chosen method (D-018).
"""

from dataclasses import dataclass

import torch

from lpv_lfr_baseline.lfr_forward import lfr_forward
from lpv_lfr_baseline.lfr_matrices import GMatrix


@dataclass
class SimResult:
    """
    Trajectory output from simulate().

    X  : (batch, N+1, 6)  state trajectory in logical coordinates
    Y  : (batch, N, 3)    output trajectory in stage coordinates
    Z  : (batch, N, 6)    latent z recorded at start of each step
    W  : (batch, N, 6)    latent w recorded at start of each step
    """
    X: torch.Tensor
    Y: torch.Tensor
    Z: torch.Tensor
    W: torch.Tensor


def rk4_step(
    x:         torch.Tensor,   # (batch, 6)  state in logical coordinates
    u_logical: torch.Tensor,   # (batch, 3)  input in logical coordinates
    G:         GMatrix,
    M0:        torch.Tensor,   # (3,3)
    M1:        torch.Tensor,   # (3,3)
    M2:        torch.Tensor,   # (3,3)
    K:         torch.Tensor,   # (3,3)
    C:         torch.Tensor,   # (3,3)
    ts:        torch.Tensor,   # ()    sample period
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Single RK4 step. Returns (x_next, z, w, y_logical).

    Y is extracted from the state at each sub-step — quasi-LPV self-scheduling.
    z and w are from the start of the step (x[k]).
    y is returned in logical coordinates — apply @ P for stage output.
    """
    # k1 — also records z, w, y at x[k] (start of step)
    k1, z, w, y = lfr_forward(x,                   u_logical, x[:, 2],                   G, M0, M1, M2, K, C)

    # k2 — Y from intermediate state
    x2 = x + (ts / 2) * k1
    k2, _, _, _  = lfr_forward(x2,                  u_logical, x2[:, 2],                  G, M0, M1, M2, K, C)

    # k3 — Y from intermediate state
    x3 = x + (ts / 2) * k2
    k3, _, _, _  = lfr_forward(x3,                  u_logical, x3[:, 2],                  G, M0, M1, M2, K, C)

    # k4 — Y from end-of-step state
    x4 = x + ts * k3
    k4, _, _, _  = lfr_forward(x4,                  u_logical, x4[:, 2],                  G, M0, M1, M2, K, C)

    x_next = x + (ts / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return x_next, z, w, y


def simulate(
    x0:           torch.Tensor,   # (batch, 6)     initial state in logical coordinates
    u_seq_stage:  torch.Tensor,   # (batch, N, 3)  input sequence in stage coordinates
    G:            GMatrix,
    M0:           torch.Tensor,
    M1:           torch.Tensor,
    M2:           torch.Tensor,
    K:            torch.Tensor,
    C:            torch.Tensor,
    P:            torch.Tensor,   # (3,3)  stage <-> logical transform
    ts:           torch.Tensor,
) -> SimResult:
    """
    Simulate N steps using RK4. Returns SimResult.

    u_seq_stage is in stage coordinates — P transform applied at each step.
    Output Y is converted to stage coordinates.
    All tensors float64.
    """
    N = u_seq_stage.shape[1]

    X_list = [x0]
    Y_list = []
    Z_list = []
    W_list = []

    x = x0
    for k in range(N):
        # Stage -> logical: u_logical[n] = P @ u_stage[n]  =>  u_logical = u_stage @ P.T
        u_logical = u_seq_stage[:, k, :] @ P.T                    # (batch, 3)
        x_next, z_k, w_k, y_k = rk4_step(x, u_logical, G, M0, M1, M2, K, C, ts)

        # Logical -> stage: y_stage[n] = P.T @ y_logical[n]  =>  y_stage = y_logical @ P
        Y_list.append(y_k @ P)
        Z_list.append(z_k)
        W_list.append(w_k)
        X_list.append(x_next)
        x = x_next

    return SimResult(
        X=torch.stack(X_list, dim=1),   # (batch, N+1, 6)
        Y=torch.stack(Y_list, dim=1),   # (batch, N, 3)
        Z=torch.stack(Z_list, dim=1),   # (batch, N, 6)
        W=torch.stack(W_list, dim=1),   # (batch, N, 6)
    )


# ----------------------------------------------------------------------
# Verification  (run as: conda run -n GraduationProject python -m lpv_lfr_baseline.lfr_simulate)
# ----------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    import os
    import numpy as np
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    from scipy.io import loadmat

    from lpv_lfr_baseline.physics import M0, M1, M2, K, C, P, ts
    from lpv_lfr_baseline.lfr_matrices import G

    dtype    = torch.float64
    mat_base = os.path.join(os.path.dirname(__file__), '..', 'Matlab-output')

    # ------------------------------------------------------------------
    # Check 1 — rk4_step: single step from rest with zero input
    # Verify output shapes and that x_next != x (system evolves).
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Check 1: rk4_step shapes and basic sanity  (batch=1)")
    print("=" * 60)

    x0_test   = torch.zeros(1, 6, dtype=dtype)                              # (1, 6)
    u_test    = torch.tensor([[1.0, -0.5, 0.2]], dtype=dtype)               # (1, 3)  stage
    u_logical = u_test @ P.T                                                 # (1, 3)  logical

    x_next, z_k, w_k, y_k = rk4_step(x0_test, u_logical, G, M0, M1, M2, K, C, ts)

    shape_ok = (
        x_next.shape == (1, 6) and
        z_k.shape    == (1, 6) and
        w_k.shape    == (1, 6) and
        y_k.shape    == (1, 3)
    )
    evolves  = not torch.allclose(x_next, x0_test)
    print(f"  Output shapes correct : {shape_ok}")
    print(f"  State evolves (x_next != x0) : {evolves}")
    print(f"  x_next[0] = {x_next[0].detach().numpy()}")
    print(f"\nCheck 1: {'PASS' if shape_ok and evolves else 'FAIL'}")

    # ------------------------------------------------------------------
    # Check 2 — rk4_step autograd: gradients flow through a full step
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Check 2: rk4_step autograd  (batch=1)")
    print("=" * 60)

    x_grad = x0_test.clone().requires_grad_(True)
    x_next_g, _, _, _ = rk4_step(x_grad, u_logical, G, M0, M1, M2, K, C, ts)
    x_next_g.sum().backward()

    grad_ok = x_grad.grad is not None
    print(f"  Backward pass succeeded : {grad_ok}")
    if grad_ok:
        print(f"  x.grad norm = {x_grad.grad.norm().item():.6e}")
    print(f"\nCheck 2: {'PASS' if grad_ok else 'FAIL'}")

    # ------------------------------------------------------------------
    # Check 3 — Trajectory vs MATLAB lsim (fixed Y ≈ 0, ZOH reference)
    #
    # Reference: gantry_q3_lsim.mat (q3) + gantry_input.mat (u)
    # x0 = [0, 0, 0.3, 0, 0, 0] — matching MATLAB initial output q3[0]=[0,0,0.3].
    # Expected: small bounded differences (RK4 vs ZOH, Y self-scheduled vs frozen).
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Check 3: Trajectory vs MATLAB lsim (gantry_q3_lsim.mat)")
    print("=" * 60)

    lsim_path  = os.path.join(mat_base, 'gantry_q3_lsim.mat')
    input_path = os.path.join(mat_base, 'gantry_input.mat')

    if os.path.exists(lsim_path) and os.path.exists(input_path):
        mat_lsim  = loadmat(lsim_path)
        mat_input = loadmat(input_path)

        q3_ref   = torch.tensor(mat_lsim['q3'], dtype=dtype)   # (N, 3)
        u_matlab = torch.tensor(mat_input['u'],  dtype=dtype)   # (N, 3)
        N_sim    = u_matlab.shape[0]

        # Batch dim: unsqueeze(0) → (1, N, 3) and (1, 6)
        x0 = torch.tensor([[0.0, 0.0, 0.3, 0.0, 0.0, 0.0]], dtype=dtype)   # (1, 6)
        with torch.no_grad():
            result = simulate(x0, u_matlab.unsqueeze(0), G, M0, M1, M2, K, C, P, ts)

        Y_ours = result.Y[0]   # (N, 3) — squeeze batch dim

        err_max  = (Y_ours - q3_ref).abs().max().item()
        err_mean = (Y_ours - q3_ref).abs().mean().item()
        print(f"  N steps simulated     : {N_sim}")
        print(f"  Max  |error| [m]      : {err_max:.4e}")
        print(f"  Mean |error| [m]      : {err_mean:.4e}")
        print(f"  (Differences expected: RK4 vs ZOH, Y self-scheduled vs frozen at 0.3)")
        # Informational only: LPV (self-scheduled Y) vs frozen LTI at Y=0.3.
        # Errors grow as Y deviates from 0.3 m. See README for expected magnitudes.
        print(f"\nCheck 3: INFORMATIONAL (LPV vs frozen LTI — differences expected)")
    else:
        print("  SKIPPED — MATLAB reference files not found.")
        print(f"  Expected: {lsim_path}")

    # ------------------------------------------------------------------
    # Check 4 — Varying-Y trajectory vs MATLAB CT quasi-LPV (gantrySystem.m)
    #
    # Reference: lpv_sim_varying_y.mat (q1, u_q1, Y_trajectory)
    # Y moves 0.3 -> 0.1 m. x0 = [0, 0, 0.3, 0, 0, 0] (logical).
    #
    # q1 is produced by Simulink running gantrySystem.m — which uses
    # Y = x(3) self-scheduled and the full varying M(Y), i.e. the SAME
    # CT quasi-LPV ODE as our Python. The error reflects integration method
    # mismatch (RK4 fixed-step vs MATLAB ode45 adaptive), NOT machine
    # precision. Expect ~1e-13 to 1e-14 for smooth dynamics at 16 kHz.
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Check 4: Varying-Y trajectory vs MATLAB CT quasi-LPV (lpv_sim_varying_y.mat)")
    print("=" * 60)

    lpv_path = os.path.join(mat_base, 'lpv_sim_varying_y.mat')

    if os.path.exists(lpv_path):
        mat_lpv = loadmat(lpv_path)
        q1_ref  = torch.tensor(mat_lpv['q1'],   dtype=dtype)   # (N, 3)
        u_lpv   = torch.tensor(mat_lpv['u_q1'], dtype=dtype)   # (N, 3)
        N_lpv   = u_lpv.shape[0]

        # Initial state: Y=0.3 m, X1=X2=0, all velocities=0
        x0_lpv = torch.tensor([[0.0, 0.0, 0.3, 0.0, 0.0, 0.0]], dtype=dtype)   # (1, 6)

        with torch.no_grad():
            result_lpv = simulate(x0_lpv, u_lpv.unsqueeze(0), G, M0, M1, M2, K, C, P, ts)

        Y_ours_lpv = result_lpv.Y[0]   # (N, 3) — squeeze batch dim

        err_max  = (Y_ours_lpv - q1_ref).abs().max().item()
        err_mean = (Y_ours_lpv - q1_ref).abs().mean().item()
        print(f"  N steps simulated     : {N_lpv}")
        print(f"  Max  |error| [m]      : {err_max:.4e}")
        print(f"  Mean |error| [m]      : {err_mean:.4e}")
        print(f"  Y range in sim [m]    : {result_lpv.X[0, :, 2].min().item():.3f} to {result_lpv.X[0, :, 2].max().item():.3f}")
        # Threshold: 1e-10 is conservative — both sims integrate the same CT ODE.
        # Errors above this indicate a physics or initial-condition mismatch.
        status = 'PASS' if err_max < 1e-10 else 'FAIL (larger than expected — check x0 or input)'
        print(f"\nCheck 4: {status}")
    else:
        print("  SKIPPED — MATLAB reference file not found.")
        print(f"  Expected: {lpv_path}")
