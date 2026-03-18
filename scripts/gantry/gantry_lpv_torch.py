"""
gantry_lpv_torch.py
-------------------
Torch reimplementation of the ASMPT dual-gantry FP model discrete-time state-space
matrices. Mirrors gantry_ss.py exactly in physics and structure, but every value is
a torch tensor from the start — physical parameters, M(Y), A_c, B_c, P, A_d, B_d.

This is NOT a wrapper around gantry_ss.py. The only structural difference is the
numerical backend: scipy cont2discrete is replaced by torch.linalg.matrix_exp on the
9x9 augmented matrix (required for differentiability and to handle singular A_c —
rigid body modes cause A_c to be singular so the naive B_d = A_c^{-1}(A_d - I)B_c
formula is undefined; the augmented exponential sidesteps this).

Use cases
---------
- LPV baseline block (Step 3): Y is a torch scalar from the state vector, requires_grad
  flows through A_d(Y) and B_d(Y) back to Y and any trainable physics parameters.
- Validation (this file's __main__): compare .detach().numpy() against gantry_ss.py
  output at Y=0.3 to verify the torch implementation matches scipy to < 1e-10.

Coordinate system
-----------------
States  : logical [X, Theta, Y, dX, dTheta, dY]
Inputs  : stage  [F_X1, F_X2, F_Y]
Outputs : stage  [X1, X2, Y]
Dimensions: nx=6, nu=3, ny=3

Reference: kamtin-fp-model/03 Simulink gantry/main.m (immutable ground truth)
           docs/lpv-discretization.md  (method rationale, D-012, D-015)
           docs/decisions.md           (D-014: separate file; D-015: augmented expm)
"""

import torch


def gantry_lpv_matrices_torch(Y: torch.Tensor, fs: float = 16e3):
    """
    Compute discrete-time LPV state-space matrices for the gantry FP model.

    All intermediate values are torch tensors — gradients flow from A_d, B_d
    back through Y and all physical parameters.

    Parameters
    ----------
    Y  : torch.Tensor, scalar — payload Y-position [m]. May have requires_grad=True.
    fs : float — sample frequency [Hz]. Default: 16000 (from main.m line 164).

    Returns
    -------
    A_d : (6, 6) torch.Tensor — discrete-time state matrix
    B_d : (6, 3) torch.Tensor — discrete-time input matrix
    C_d : (3, 6) torch.Tensor — output matrix (constant, no Y-dependence)
    D_d : (3, 3) torch.Tensor — feedthrough matrix (zero)

    All outputs are dtype=torch.float64.
    """
    dtype = torch.float64
    ts = torch.tensor(1.0 / fs, dtype=dtype)

    # ------------------------------------------------------------------
    # Step 1: Physical parameters (from main.m lines 12-36)
    # All defined as torch scalars so gradients can flow through them
    # if requires_grad is set later.
    # ------------------------------------------------------------------
    mb  = torch.tensor(22.8,   dtype=dtype)  # Mass of moving cross-arm       [kg]
    mh  = torch.tensor(10.1,   dtype=dtype)  # Mass of payload (Y-axis)       [kg]
    m1  = torch.tensor(10.2,   dtype=dtype)  # Mass of actuator X1            [kg]
    m2  = torch.tensor(10.7,   dtype=dtype)  # Mass of actuator X2            [kg]

    Jb  = torch.tensor(1.0,    dtype=dtype)  # Rotary inertia of cross-arm    [kg.m^2]
    Jh  = torch.tensor(0.05,   dtype=dtype)  # Rotary inertia of payload      [kg.m^2]

    cg1 = torch.tensor(14.5,   dtype=dtype)  # Viscous friction X1            [N/(m/s)]
    cg2 = torch.tensor(20.3,   dtype=dtype)  # Viscous friction X2            [N/(m/s)]
    cy  = torch.tensor(10.0,   dtype=dtype)  # Viscous friction Y             [N/(m/s)]

    cb1 = torch.tensor(9.0,    dtype=dtype)  # Viscous friction joint 1       [Nm/(rad/s)]
    cb2 = torch.tensor(9.0,    dtype=dtype)  # Viscous friction joint 2       [Nm/(rad/s)]

    kb1 = torch.tensor(1987.5, dtype=dtype)  # Stiffness joint 1              [N.m/rad]
    kb2 = torch.tensor(1987.5, dtype=dtype)  # Stiffness joint 2              [N.m/rad]

    Lb  = torch.tensor(0.725,  dtype=dtype)  # Length of moving cross-arm     [m]
    d   = torch.tensor(0.1,    dtype=dtype)  # Distance cross-arm to payload  [m]

    # Ensure Y is float64
    Y = Y.to(dtype=dtype)

    # ------------------------------------------------------------------
    # Step 2: Mass matrix M(Y)  (from main.m lines 52-54)
    # M[0,1] = M[1,0] : linear in Y
    # M[1,1]           : quadratic in Y
    # All other entries: constant
    # ------------------------------------------------------------------
    M_00 = m1 + m2 + mb + mh
    M_01 = (m1 - m2) * Lb / 2 - mh * Y
    M_11 = Jb + Jh + (m1 + m2) * Lb**2 / 4 + mh * d**2 + mh * Y**2
    M_12 = -mh * d
    M_22 = mh

    M = torch.stack([
        torch.stack([M_00, M_01, torch.zeros(1, dtype=dtype).squeeze()]),
        torch.stack([M_01, M_11, M_12]),
        torch.stack([torch.zeros(1, dtype=dtype).squeeze(), M_12, M_22]),
    ])  # (3, 3)

    # ------------------------------------------------------------------
    # Step 3: Viscous damping matrix C_damp  (from main.m lines 57-59)
    # Named C_damp to avoid collision with output matrix C_d.
    # Constant — no Y-dependence.
    # torch.stack used (not torch.tensor) because elements are torch tensors.
    # ------------------------------------------------------------------
    z = torch.zeros((), dtype=dtype)
    C_00 = cg1 + cg2
    C_01 = (cg1 - cg2) * Lb / 2
    C_11 = cb1 + cb2 + (cg1 + cg2) * Lb**2 / 4
    C_damp = torch.stack([
        torch.stack([C_00, C_01, z   ]),
        torch.stack([C_01, C_11, z   ]),
        torch.stack([z,    z,    cy  ]),
    ])  # (3, 3)

    # ------------------------------------------------------------------
    # Step 4: Stiffness matrix K  (from main.m lines 62-64)
    # Constant — no Y-dependence.
    # ------------------------------------------------------------------
    K_11 = kb1 + kb2
    K = torch.stack([
        torch.stack([z,    z,    z   ]),
        torch.stack([z,    K_11, z   ]),
        torch.stack([z,    z,    z   ]),
    ])  # (3, 3)

    # ------------------------------------------------------------------
    # Step 5: Continuous-time SS in logical coordinates  (from getss.m)
    #
    #   A_c = [     0      |    I    ]
    #         [ -M^{-1}K   | -M^{-1}C]
    #
    #   B_c = [    0    ]
    #         [ M^{-1}  ]
    #
    # Use torch.linalg.solve(M, X) = M^{-1} X — more stable than explicit inv.
    # M^{-1} itself is needed as a block in B_c; compute via solve with identity.
    # ------------------------------------------------------------------
    eye3 = torch.eye(3, dtype=dtype)

    MiK = torch.linalg.solve(M, K)           # (3, 3)  M^{-1} K
    MiC = torch.linalg.solve(M, C_damp)      # (3, 3)  M^{-1} C
    Mi  = torch.linalg.solve(M, eye3)        # (3, 3)  M^{-1}

    zeros_33 = torch.zeros(3, 3, dtype=dtype)
    A_c = torch.cat([
        torch.cat([zeros_33, eye3 ], dim=1),
        torch.cat([-MiK,    -MiC  ], dim=1),
    ], dim=0)  # (6, 6)

    B_c = torch.cat([zeros_33, Mi], dim=0)  # (6, 3)

    # ------------------------------------------------------------------
    # Step 6: Transform I/O to stage coordinates  (from main.m lines 98-103)
    #
    #   P maps logical forces to stage forces: f_logical = P * f_stage
    #   Stage positions from logical:          q_stage   = P.T * q_logical
    #
    #   B_stage = B_c @ P
    #   C_stage = P.T @ C_c   where C_c = [I_3 | 0_3]
    #   A unchanged (internal states stay in logical coordinates)
    # ------------------------------------------------------------------
    one = torch.ones((), dtype=dtype)
    P = torch.stack([
        torch.stack([one,      one,      z  ]),
        torch.stack([Lb / 2,  -Lb / 2,  z  ]),
        torch.stack([z,        z,        one]),
    ])  # (3, 3)

    B_c_stage = B_c @ P                                               # (6, 3)
    C_c       = torch.cat([eye3, torch.zeros(3, 3, dtype=dtype)], dim=1)  # (3, 6)
    C_c_stage = P.T @ C_c                                             # (3, 6)
    D_c_stage = torch.zeros(3, 3, dtype=dtype)

    # ------------------------------------------------------------------
    # Step 7: Exact ZOH discretization via augmented matrix exponential
    #
    # A_c is singular (rigid body modes → top-left 3x3 block is zero),
    # so B_d = A_c^{-1}(A_d - I)B_c is undefined. Instead use:
    #
    #   M_aug = [[A_c,  B_c_stage],    (9x9)
    #            [ 0,       0     ]]
    #
    #   expm(M_aug * ts) = [[A_d,  B_d],
    #                       [ 0,    I  ]]
    #
    # Extract A_d = EM[:6, :6],  B_d = EM[:6, 6:]
    # This is identical to what scipy cont2discrete(method='zoh') computes.
    # torch.linalg.matrix_exp is differentiable — gradients flow through A_d, B_d
    # back to Y and all physical parameters.  (D-012, D-015)
    # ------------------------------------------------------------------
    n, m = 6, 3
    M_aug = torch.cat([
        torch.cat([A_c,                          B_c_stage                   ], dim=1),
        torch.cat([torch.zeros(m, n, dtype=dtype), torch.zeros(m, m, dtype=dtype)], dim=1),
    ], dim=0)  # (9, 9)

    EM  = torch.linalg.matrix_exp(M_aug * ts)

    A_d = EM[:n, :n]   # (6, 6)
    B_d = EM[:n, n:]   # (6, 3)
    C_d = C_c_stage    # (3, 6) — constant
    D_d = D_c_stage    # (3, 3) — zero

    return A_d, B_d, C_d, D_d


# ----------------------------------------------------------------------
# Validation: compare against gantry_discrete_ss(Y=0.3) from gantry_ss.py
# Run from repo root:
#   conda run -n GraduationProject python scripts/gantry/gantry_lpv_torch.py
# ----------------------------------------------------------------------
if __name__ == '__main__':
    import numpy as np
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from gantry_ss import gantry_discrete_ss

    print("=" * 60)
    print("Validation: torch vs scipy at Y = 0.3 m")
    print("=" * 60)

    Y_val = 0.3

    # --- scipy reference ---
    A_ref, B_ref, C_ref, D_ref = gantry_discrete_ss(Y=Y_val)

    # --- torch version (no grad needed for comparison) ---
    Y_t = torch.tensor(Y_val, dtype=torch.float64)
    A_t, B_t, C_t, D_t = gantry_lpv_matrices_torch(Y_t)

    A_np = A_t.detach().numpy()
    B_np = B_t.detach().numpy()
    C_np = C_t.detach().numpy()
    D_np = D_t.detach().numpy()

    tol = 1e-10
    results = {
        'A': np.max(np.abs(A_np - A_ref)),
        'B': np.max(np.abs(B_np - B_ref)),
        'C': np.max(np.abs(C_np - C_ref)),
        'D': np.max(np.abs(D_np - D_ref)),
    }

    print(f"\nMax absolute error (torch vs scipy), tolerance = {tol:.0e}:")
    all_pass = True
    for name, err in results.items():
        status = 'PASS' if err < tol else 'FAIL'
        if status == 'FAIL':
            all_pass = False
        print(f"  {name}: {err:.2e}  ->  {status}")

    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")

    # --- Gradient test ---
    print("\n" + "=" * 60)
    print("Gradient test: d(sum(A_d)) / dY at Y = 0.3")
    print("=" * 60)

    Y_grad = torch.tensor(Y_val, dtype=torch.float64, requires_grad=True)
    A_g, B_g, _, _ = gantry_lpv_matrices_torch(Y_grad)
    loss = A_g.sum() + B_g.sum()
    loss.backward()

    grad_ok = Y_grad.grad is not None
    print(f"  Backward pass succeeded: {grad_ok}")
    if grad_ok:
        print(f"  dL/dY = {Y_grad.grad.item():.6e}")
    print(f"\nGradient test: {'PASS' if grad_ok else 'FAIL'}")
