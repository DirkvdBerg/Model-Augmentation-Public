"""
lfr_simulate.py
---------------
RK4 single-step and trajectory simulation for the dual-gantry LPV-LFR baseline.

Provides:
    rk4_step()        - single RK4 step, self-scheduled Y from state
    simulate()        - N-step trajectory with BPTT mode control
    simulate_frozen() - same but M(Y) frozen at a fixed Y (LTI comparison)

All functions take G (GMatrix) and polynomial constants (mh, alpha, beta, gamma,
N0, N1, N2) as explicit arguments. Callers must build these from current physical
parameters using build_G_matrix() and build_poly_constants() before calling.
If physical parameters are trainable (nn.Parameter), rebuild inside forward().
"""

from dataclasses import dataclass
from typing import Literal

import torch
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from lpv_lfr_baseline.core.lfr_forward import lfr_forward
from lpv_lfr_baseline.core.lfr_matrices import GMatrix


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
    x:          torch.Tensor,           # (batch, 6)  state in logical coordinates
    u_logical:  torch.Tensor,           # (batch, 3)  input in logical coordinates
    G:          GMatrix,                # constant interconnection matrix
    K:          torch.Tensor,           # (3, 3)  stiffness
    C:          torch.Tensor,           # (3, 3)  damping
    mh:         torch.Tensor,           # ()      payload mass
    alpha:      torch.Tensor,           # ()      m1+m2+mb+mh
    beta:       torch.Tensor,           # ()      (m1-m2)*Lb/2
    gamma:      torch.Tensor,           # ()      Jb+Jh+(m1+m2)*Lb^2/4
    N0:         torch.Tensor,           # (3, 3)  adjugate at Y^0
    N1:         torch.Tensor,           # (3, 3)  adjugate at Y^1
    N2:         torch.Tensor,           # (3, 3)  adjugate at Y^2
    ts:         torch.Tensor,           # ()      sample period
    Y_override: torch.Tensor | None = None,  # (batch,) freeze Y at this value
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Single RK4 step. Returns (x_next, z, w, y_logical).

    Y_override=None: self-scheduled, Y re-extracted from state at each sub-step.
    Y_override=tensor: frozen LTI, Y held constant across all sub-steps.
    """
    def _Y(state: torch.Tensor) -> torch.Tensor:
        return state[:, 2] if Y_override is None else Y_override

    def _fwd(s):
        return lfr_forward(s, u_logical, _Y(s), G, K, C, mh, alpha, beta, gamma, N0, N1, N2)

    k1, z, w, y = _fwd(x)

    x2 = x + (ts / 2) * k1
    k2, _, _, _ = _fwd(x2)

    x3 = x + (ts / 2) * k2
    k3, _, _, _ = _fwd(x3)

    x4 = x + ts * k3
    k4, _, _, _ = _fwd(x4)

    x_next = x + (ts / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_next, z, w, y


def simulate(
    x0:           torch.Tensor,   # (batch, 6)     initial state in logical coordinates
    u_seq_stage:  torch.Tensor,   # (batch, N, 3)  input sequence in stage coordinates
    G:            GMatrix,        # constant interconnection matrix
    K:            torch.Tensor,   # (3, 3)
    C:            torch.Tensor,   # (3, 3)
    mh:           torch.Tensor,   # ()
    alpha:        torch.Tensor,   # ()
    beta:         torch.Tensor,   # ()
    gamma:        torch.Tensor,   # ()
    N0:           torch.Tensor,   # (3, 3)
    N1:           torch.Tensor,   # (3, 3)
    N2:           torch.Tensor,   # (3, 3)
    P:            torch.Tensor,   # (3, 3)  stage <-> logical transform
    ts:           torch.Tensor,
    bptt_mode:    Literal["full", "truncated", "checkpoint"] = "full",
    segment_len:  int = 200,
) -> SimResult:
    """
    Simulate N steps using RK4. Returns SimResult.

    bptt_mode: "full" (exact, O(N) memory), "truncated" (detach every segment_len),
               "checkpoint" (exact, O(sqrt(N)) memory, ~1.3x compute).
    """
    batch = x0.shape[0]
    N = u_seq_stage.shape[1]

    # Pre-transform entire input sequence: stage -> logical (once, not N times)
    u_seq_logical = u_seq_stage @ P.T                              # (batch, N, 3)

    # Pre-allocate output tensors
    X = x0.new_empty(batch, N + 1, 6)
    Y = x0.new_empty(batch, N, 3)
    Z = x0.new_empty(batch, N, 6)
    W = x0.new_empty(batch, N, 6)
    X[:, 0, :] = x0

    x = x0
    for k in range(N):
        u_logical = u_seq_logical[:, k, :]

        if bptt_mode == "checkpoint":
            # Capture G and poly constants in closure; only pass x and u as tensor args.
            # use_reentrant=False handles tensors captured in the closure correctly.
            def _step(x_in, u_in):
                return rk4_step(
                    x_in, u_in, G, K, C, mh, alpha, beta, gamma, N0, N1, N2, ts
                )
            x_next, z_k, w_k, y_k = grad_checkpoint(
                _step, x, u_logical, use_reentrant=False,
            )
        else:
            x_next, z_k, w_k, y_k = rk4_step(
                x, u_logical, G, K, C, mh, alpha, beta, gamma, N0, N1, N2, ts
            )

        X[:, k + 1, :] = x_next
        Y[:, k, :]     = y_k @ P          # logical -> stage
        Z[:, k, :]     = z_k
        W[:, k, :]     = w_k

        # Truncated BPTT: detach state at segment boundaries
        if bptt_mode == "truncated" and (k + 1) % segment_len == 0:
            x = x_next.detach().requires_grad_(x_next.requires_grad)
        else:
            x = x_next

    return SimResult(X=X, Y=Y, Z=Z, W=W)


def simulate_frozen(
    x0:          torch.Tensor,   # (batch, 6)    initial state in logical coordinates
    u_seq_stage: torch.Tensor,   # (batch, N, 3) input sequence in stage coordinates
    G:           GMatrix,        # constant interconnection matrix
    K:           torch.Tensor,   # (3, 3)
    C:           torch.Tensor,   # (3, 3)
    mh:          torch.Tensor,   # ()
    alpha:       torch.Tensor,   # ()
    beta:        torch.Tensor,   # ()
    gamma:       torch.Tensor,   # ()
    N0:          torch.Tensor,   # (3, 3)
    N1:          torch.Tensor,   # (3, 3)
    N2:          torch.Tensor,   # (3, 3)
    P:           torch.Tensor,   # (3, 3)  stage <-> logical transform
    ts:          torch.Tensor,
    Y_freeze:    float = 0.3,
) -> SimResult:
    """
    Simulate N steps with M(Y) frozen at Y_freeze (LTI baseline for comparison).
    Reuses rk4_step with Y_override to avoid duplicating RK4 logic.
    """
    batch = x0.shape[0]
    N     = u_seq_stage.shape[1]
    Y_c   = torch.full((batch,), Y_freeze, dtype=x0.dtype, device=x0.device)

    u_seq_logical = u_seq_stage @ P.T                  # pre-transform once

    X = x0.new_empty(batch, N + 1, 6)
    Y = x0.new_empty(batch, N, 3)
    Z = x0.new_empty(batch, N, 6)
    W = x0.new_empty(batch, N, 6)
    X[:, 0, :] = x0

    x = x0
    for k in range(N):
        x_next, z_k, w_k, y_k = rk4_step(
            x, u_seq_logical[:, k, :], G, K, C, mh, alpha, beta, gamma, N0, N1, N2, ts,
            Y_override=Y_c,
        )
        X[:, k + 1, :] = x_next
        Y[:, k, :]     = y_k @ P
        Z[:, k, :]     = z_k
        W[:, k, :]     = w_k
        x = x_next

    return SimResult(X=X, Y=Y, Z=Z, W=W)


# ----------------------------------------------------------------------
# Verification  (run as: conda run -n GraduationProject python -m lpv_lfr_baseline.core.lfr_simulate)
# ----------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    import os
    import numpy as np
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

    from scipy.io import loadmat

    from lpv_lfr_baseline.core.physics import (
        M1, M2, K, C, P, ts, build_poly_constants,
        mh as _mh, m1 as _m1, m2 as _m2, mb as _mb, Jb as _Jb, Jh as _Jh,
        Lb as _Lb, d as _d,
    )
    from lpv_lfr_baseline.core.lfr_matrices import build_G_matrix

    dtype    = torch.float64
    mat_base = os.path.join(os.path.dirname(__file__), '..', 'Matlab-output')

    # Build G and poly constants from true physics params
    alpha, beta, gamma, N0, N1, N2 = build_poly_constants(_m1, _m2, _mb, _mh, _Jb, _Jh, _Lb, _d)
    d0_true = _mh * (alpha * gamma - beta ** 2)
    G_true  = build_G_matrix(N0, d0_true, M1, M2, K, C)

    # ------------------------------------------------------------------
    # Check 1 — rk4_step: single step from rest with zero input
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Check 1: rk4_step shapes and basic sanity  (batch=1)")
    print("=" * 60)

    x0_test   = torch.zeros(1, 6, dtype=dtype)
    u_test    = torch.tensor([[1.0, -0.5, 0.2]], dtype=dtype)
    u_logical = u_test @ P.T

    x_next, z_k, w_k, y_k = rk4_step(
        x0_test, u_logical, G_true, K, C, _mh, alpha, beta, gamma, N0, N1, N2, ts
    )

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
    # Check 2 — rk4_step autograd
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Check 2: rk4_step autograd  (batch=1)")
    print("=" * 60)

    x_grad = x0_test.clone().requires_grad_(True)
    x_next_g, _, _, _ = rk4_step(
        x_grad, u_logical, G_true, K, C, _mh, alpha, beta, gamma, N0, N1, N2, ts
    )
    x_next_g.sum().backward()

    grad_ok = x_grad.grad is not None
    print(f"  Backward pass succeeded : {grad_ok}")
    if grad_ok:
        print(f"  x.grad norm = {x_grad.grad.norm().item():.6e}")
    print(f"\nCheck 2: {'PASS' if grad_ok else 'FAIL'}")

    # ------------------------------------------------------------------
    # Check 3 — Trajectory vs MATLAB lsim
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

        q3_ref   = torch.tensor(mat_lsim['q3'], dtype=dtype)
        u_matlab = torch.tensor(mat_input['u'],  dtype=dtype)
        N_sim    = u_matlab.shape[0]

        x0 = torch.tensor([[0.0, 0.0, 0.3, 0.0, 0.0, 0.0]], dtype=dtype)
        with torch.no_grad():
            result = simulate(
                x0, u_matlab.unsqueeze(0),
                G_true, K, C, _mh, alpha, beta, gamma, N0, N1, N2, P, ts
            )

        Y_ours = result.Y[0]
        err_max  = (Y_ours - q3_ref).abs().max().item()
        err_mean = (Y_ours - q3_ref).abs().mean().item()
        print(f"  N steps simulated     : {N_sim}")
        print(f"  Max  |error| [m]      : {err_max:.4e}")
        print(f"  Mean |error| [m]      : {err_mean:.4e}")
        print(f"\nCheck 3: INFORMATIONAL (LPV vs frozen LTI — differences expected)")
    else:
        print("  SKIPPED — MATLAB reference files not found.")

    # ------------------------------------------------------------------
    # Check 4 — Varying-Y trajectory vs MATLAB CT quasi-LPV
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Check 4: Varying-Y trajectory vs MATLAB CT quasi-LPV (lpv_sim_varying_y.mat)")
    print("=" * 60)

    lpv_path = os.path.join(mat_base, 'lpv_sim_varying_y.mat')

    if os.path.exists(lpv_path):
        mat_lpv = loadmat(lpv_path)
        q1_ref  = torch.tensor(mat_lpv['q1'],   dtype=dtype)
        u_lpv   = torch.tensor(mat_lpv['u_q1'], dtype=dtype)
        N_lpv   = u_lpv.shape[0]

        x0_lpv = torch.tensor([[0.0, 0.0, 0.3, 0.0, 0.0, 0.0]], dtype=dtype)

        with torch.no_grad():
            result_lpv = simulate(
                x0_lpv, u_lpv.unsqueeze(0),
                G_true, K, C, _mh, alpha, beta, gamma, N0, N1, N2, P, ts
            )

        Y_ours_lpv = result_lpv.Y[0]
        err_max  = (Y_ours_lpv - q1_ref).abs().max().item()
        err_mean = (Y_ours_lpv - q1_ref).abs().mean().item()
        print(f"  N steps simulated     : {N_lpv}")
        print(f"  Max  |error| [m]      : {err_max:.4e}")
        print(f"  Mean |error| [m]      : {err_mean:.4e}")
        print(f"  Y range in sim [m]    : {result_lpv.X[0, :, 2].min().item():.3f} to {result_lpv.X[0, :, 2].max().item():.3f}")
        status = 'PASS' if err_max < 1e-10 else 'FAIL (larger than expected — check x0 or input)'
        print(f"\nCheck 4: {status}")
    else:
        print("  SKIPPED — MATLAB reference file not found.")

    # ------------------------------------------------------------------
    # Check 5 — BPTT modes
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Check 5: BPTT modes — trajectory match and gradient behaviour")
    print("=" * 60)

    N_bptt  = 50
    x0_bptt = torch.tensor([[0.05, 0.01, 0.30, 0.02, -0.01, 0.05]], dtype=dtype)
    u_bptt  = torch.randn(1, N_bptt, 3, dtype=dtype) * 5.0

    with torch.no_grad():
        ref = simulate(
            x0_bptt, u_bptt, G_true, K, C, _mh, alpha, beta, gamma, N0, N1, N2, P, ts,
            bptt_mode="full"
        )

    with torch.no_grad():
        trunc = simulate(
            x0_bptt, u_bptt, G_true, K, C, _mh, alpha, beta, gamma, N0, N1, N2, P, ts,
            bptt_mode="truncated", segment_len=20
        )
    err_trunc = (trunc.X - ref.X).abs().max().item()
    trunc_traj_ok = err_trunc == 0.0
    print(f"  5a  truncated trajectory matches full : {trunc_traj_ok}  (max|diff| = {err_trunc:.2e})")

    with torch.no_grad():
        ckpt = simulate(
            x0_bptt, u_bptt, G_true, K, C, _mh, alpha, beta, gamma, N0, N1, N2, P, ts,
            bptt_mode="checkpoint"
        )
    err_ckpt = (ckpt.X - ref.X).abs().max().item()
    ckpt_traj_ok = err_ckpt == 0.0
    print(f"  5b  checkpoint trajectory matches full: {ckpt_traj_ok}  (max|diff| = {err_ckpt:.2e})")

    x0_t = x0_bptt.clone().requires_grad_(True)
    res_t = simulate(
        x0_t, u_bptt, G_true, K, C, _mh, alpha, beta, gamma, N0, N1, N2, P, ts,
        bptt_mode="truncated", segment_len=20
    )
    res_t.X[:, -1, :].sum().backward()
    trunc_grad_blocked = x0_t.grad is None or x0_t.grad.norm().item() == 0.0
    print(f"  5c  truncated: last-step grad blocked at x0: {trunc_grad_blocked}")

    x0_t2 = x0_bptt.clone().requires_grad_(True)
    res_t2 = simulate(
        x0_t2, u_bptt, G_true, K, C, _mh, alpha, beta, gamma, N0, N1, N2, P, ts,
        bptt_mode="truncated", segment_len=20
    )
    res_t2.X[:, 10, :].sum().backward()
    trunc_grad_flows = x0_t2.grad is not None and x0_t2.grad.norm().item() > 0.0
    print(f"  5c  truncated: within-segment grad flows to x0: {trunc_grad_flows}")

    x0_full = x0_bptt.clone().requires_grad_(True)
    res_full = simulate(
        x0_full, u_bptt, G_true, K, C, _mh, alpha, beta, gamma, N0, N1, N2, P, ts,
        bptt_mode="full"
    )
    res_full.X[:, -1, :].sum().backward()

    x0_ckpt = x0_bptt.clone().requires_grad_(True)
    res_ckpt = simulate(
        x0_ckpt, u_bptt, G_true, K, C, _mh, alpha, beta, gamma, N0, N1, N2, P, ts,
        bptt_mode="checkpoint"
    )
    res_ckpt.X[:, -1, :].sum().backward()

    grad_err = (x0_full.grad - x0_ckpt.grad).abs().max().item()
    ckpt_grad_ok = grad_err < 1e-10
    print(f"  5d  checkpoint grad matches full: {ckpt_grad_ok}  (max|diff| = {grad_err:.2e})")

    all_bptt = trunc_traj_ok and ckpt_traj_ok and trunc_grad_blocked and trunc_grad_flows and ckpt_grad_ok
    print(f"\nCheck 5: {'PASS' if all_bptt else 'FAIL'}")
