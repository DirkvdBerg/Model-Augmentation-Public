"""
physics.py
----------
Physical constants for the dual-gantry FP model as torch tensors.

Source: kamtin-fp-model/03 Simulink gantry/main.m (immutable ground truth).
All values must match main.m exactly. Do not modify without checking main.m first.

Provides:
    - Scalar physical parameters (masses, inertias, damping, stiffness, geometry)
    - Mass matrix decomposition: M(Y) = M0 + M1*Y + M2*Y^2
    - Constant damping matrix C
    - Constant stiffness matrix K
    - Coordinate transform P  (logical -> stage)
    - Sampling constants: fs, ts

All tensors are dtype=torch.float64.
"""

import torch

_D = torch.float64  # shorthand — all tensors in this module use float64

# ----------------------------------------------------------------------
# Scalar physical parameters  (from main.m lines 12-36)
# ----------------------------------------------------------------------
mb  = torch.tensor(22.8,   dtype=_D)  # Mass of moving cross-arm       [kg]
mh  = torch.tensor(10.1,   dtype=_D)  # Mass of payload (Y-axis)       [kg]
m1  = torch.tensor(10.2,   dtype=_D)  # Mass of actuator X1            [kg]
m2  = torch.tensor(10.7,   dtype=_D)  # Mass of actuator X2            [kg]

Jb  = torch.tensor(1.0,    dtype=_D)  # Rotary inertia of cross-arm    [kg.m^2]
Jh  = torch.tensor(0.05,   dtype=_D)  # Rotary inertia of payload      [kg.m^2]

cg1 = torch.tensor(14.5,   dtype=_D)  # Viscous friction X1            [N/(m/s)]
cg2 = torch.tensor(20.3,   dtype=_D)  # Viscous friction X2            [N/(m/s)]
cy  = torch.tensor(10.0,   dtype=_D)  # Viscous friction Y             [N/(m/s)]

cb1 = torch.tensor(9.0,    dtype=_D)  # Viscous friction joint 1       [Nm/(rad/s)]
cb2 = torch.tensor(9.0,    dtype=_D)  # Viscous friction joint 2       [Nm/(rad/s)]

kb1 = torch.tensor(1987.5, dtype=_D)  # Stiffness joint 1              [N.m/rad]
kb2 = torch.tensor(1987.5, dtype=_D)  # Stiffness joint 2              [N.m/rad]

Lb  = torch.tensor(0.725,  dtype=_D)  # Length of moving cross-arm     [m]
d   = torch.tensor(0.1,    dtype=_D)  # Distance cross-arm to payload  [m]

# ----------------------------------------------------------------------
# Mass matrix decomposition  M(Y) = M0 + M1*Y + M2*Y^2
# (from main.m lines 52-54)
#
# M(Y) full form:
#   [  m1+m2+mb+mh,               (m1-m2)*Lb/2 - mh*Y,                     0  ]
#   [  (m1-m2)*Lb/2 - mh*Y,   Jb+Jh+(m1+m2)*Lb^2/4+mh*d^2 + mh*Y^2,   -mh*d  ]
#   [  0,                                         -mh*d,                    mh  ]
#
# Decomposition by polynomial degree in Y:
#   M0 : constant part  (Y=0 substitution)
#   M1 : coefficient of Y    — only M1[0,1] = M1[1,0] = -mh
#   M2 : coefficient of Y^2  — only M2[1,1] = mh
# ----------------------------------------------------------------------
z3 = torch.zeros(3, 3, dtype=_D)

M0 = z3.clone()
M0[0, 0] = m1 + m2 + mb + mh
M0[0, 1] = (m1 - m2) * Lb / 2
M0[1, 0] = (m1 - m2) * Lb / 2
M0[1, 1] = Jb + Jh + (m1 + m2) * Lb**2 / 4 + mh * d**2
M0[1, 2] = -mh * d
M0[2, 1] = -mh * d
M0[2, 2] = mh

M1 = z3.clone()
M1[0, 1] = -mh
M1[1, 0] = -mh

M2 = z3.clone()
M2[1, 1] = mh

# ----------------------------------------------------------------------
# Constant damping matrix C  (from main.m lines 57-59)
# No Y-dependence.
# ----------------------------------------------------------------------
C = z3.clone()
C[0, 0] = cg1 + cg2
C[0, 1] = (cg1 - cg2) * Lb / 2
C[1, 0] = (cg1 - cg2) * Lb / 2
C[1, 1] = cb1 + cb2 + (cg1 + cg2) * Lb**2 / 4
C[2, 2] = cy

# ----------------------------------------------------------------------
# Constant stiffness matrix K  (from main.m lines 62-64)
# No Y-dependence.
# ----------------------------------------------------------------------
K = z3.clone()
K[1, 1] = kb1 + kb2

# ----------------------------------------------------------------------
# Coordinate transform P  (from main.m lines 98-100)
# Logical forces  = P @ stage forces
# Stage positions = P.T @ logical positions
# ----------------------------------------------------------------------
P = z3.clone()
P[0, 0] = 1.0
P[0, 1] = 1.0
P[1, 0] = Lb / 2
P[1, 1] = -Lb / 2
P[2, 2] = 1.0

# ----------------------------------------------------------------------
# Sampling constants  (from main.m line 164)
# ----------------------------------------------------------------------
fs = torch.tensor(20e3,     dtype=_D)   # sample frequency  [Hz]
ts = torch.tensor(1 / 20e3, dtype=_D)  # sample period     [s]


def build_poly_constants(
    m1:  torch.Tensor,
    m2:  torch.Tensor,
    mb:  torch.Tensor,
    mh:  torch.Tensor,
    Jb:  torch.Tensor,
    Jh:  torch.Tensor,
    Lb:  torch.Tensor,
    d:   torch.Tensor,
) -> tuple:
    """
    Build polynomial constants for the analytical LPV-LFR loop solution.

    Returns (alpha, beta, gamma, N0, N1, N2) where:
        alpha, beta, gamma : scalar tensors (polynomial denominator shorthand)
        N0, N1, N2         : (3, 3) adjugate coefficient matrices

    gamma does NOT include mh*d^2 — see LPV-LFR-Rational-rewrite.md.

    All outputs are differentiable w.r.t. inputs.
    Call inside forward() when any input is a trainable nn.Parameter.

    Source: LPV-LFR-Implementation-Spec.md Section 4, verified against
    Verification/LPV-LFR-Rational/LPV_LFR_rational_verification.m.
    """
    alpha = m1 + m2 + mb + mh
    beta  = (m1 - m2) * Lb / 2
    gamma = Jb + Jh + (m1 + m2) * Lb ** 2 / 4    # WITHOUT mh*d^2

    z = torch.zeros((), dtype=m1.dtype, device=m1.device)

    N0 = torch.stack([
        torch.stack([mh * gamma,        -beta * mh,                    -beta * d * mh              ]),
        torch.stack([-beta * mh,         alpha * mh,                    alpha * d * mh             ]),
        torch.stack([-beta * d * mh,     alpha * d * mh,   alpha * (gamma + mh * d ** 2) - beta ** 2]),
    ])
    N1 = torch.stack([
        torch.stack([z,           mh ** 2,       d * mh ** 2  ]),
        torch.stack([mh ** 2,     z,             z            ]),
        torch.stack([d * mh ** 2, z,             2 * beta * mh]),
    ])
    N2 = torch.stack([
        torch.stack([mh ** 2,  z,  z                    ]),
        torch.stack([z,        z,  z                    ]),
        torch.stack([z,        z,  alpha * mh - mh ** 2 ]),
    ])
    return alpha, beta, gamma, N0, N1, N2


def build_M(Y: torch.Tensor) -> torch.Tensor:
    """
    Compute M(Y) = M0 + M1*Y + M2*Y^2 for a scalar Y tensor.

    Parameters
    ----------
    Y : torch.Tensor, scalar (0D) — payload Y-position [m]

    Returns
    -------
    M_Y : (3, 3) torch.Tensor, dtype=float64
    """
    return M0 + M1 * Y + M2 * (Y ** 2)


# ----------------------------------------------------------------------
# Verification  (run as: conda run -n GraduationProject python lpv_lfr_baseline/physics.py)
# ----------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    import os
    import numpy as np

    # ------------------------------------------------------------------
    # Check 1: M(Y) decomposition matches gantry_lpv_torch.py at sample Y values
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Check 1: M(Y) decomposition vs gantry_lpv_torch.py")
    print("=" * 60)

    test_Y = [0.0, 0.1, 0.3, -0.2, 0.35]
    all_pass = True
    for y_val in test_Y:
        Y_t = torch.tensor(y_val, dtype=_D)

        # Our decomposition
        M_ours = build_M(Y_t)

        # Reference: extract M from gantry_lpv_torch internals by recomputing
        # M(Y) using the same formula (copied from gantry_lpv_torch.py)
        M_00 = m1 + m2 + mb + mh
        M_01 = (m1 - m2) * Lb / 2 - mh * Y_t
        M_11 = Jb + Jh + (m1 + m2) * Lb**2 / 4 + mh * d**2 + mh * Y_t**2
        M_12 = -mh * d
        M_22 = mh
        z    = torch.zeros((), dtype=_D)
        M_ref = torch.stack([
            torch.stack([M_00, M_01, z   ]),
            torch.stack([M_01, M_11, M_12]),
            torch.stack([z,    M_12, M_22]),
        ])

        err = (M_ours - M_ref).abs().max().item()
        status = 'PASS' if err == 0.0 else 'FAIL'
        if status == 'FAIL':
            all_pass = False
        print(f"  Y = {y_val:+.2f} m   max|error| = {err:.2e}   {status}")

    print(f"\nCheck 1: {'ALL PASS' if all_pass else 'SOME FAILED'}")

    # ------------------------------------------------------------------
    # Check 2: det(M(Y)) matches det_M from lpv_matrices.mat (50 Y values)
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Check 2: det(M(Y)) vs MATLAB det_M  (50 Y values)")
    print("=" * 60)

    mat_path = os.path.join(
        os.path.dirname(__file__), '..', 'Matlab-output', 'lpv_matrices.mat'
    )
    try:
        from scipy.io import loadmat
        mat      = loadmat(mat_path)
        Y_vals   = mat['Y_values'].squeeze()   # (50,)
        det_M_ml = mat['det_M'].squeeze()      # (50,)

        tol = 1e-8
        errs = []
        for i, y_val in enumerate(Y_vals):
            Y_t   = torch.tensor(float(y_val), dtype=_D)
            M_Y   = build_M(Y_t)
            det_py = torch.linalg.det(M_Y).item()
            errs.append(abs(det_py - float(det_M_ml[i])))

        max_err  = max(errs)
        status   = 'PASS' if max_err < tol else 'FAIL'
        print(f"  Max |det error| over 50 Y values: {max_err:.2e}   (tol={tol:.0e})   {status}")
        print(f"\nCheck 2: {status}")

    except FileNotFoundError:
        print(f"  SKIPPED — file not found: {mat_path}")
        print("  Run Matlab-scripts/export_lpv_matrices.m first.")
