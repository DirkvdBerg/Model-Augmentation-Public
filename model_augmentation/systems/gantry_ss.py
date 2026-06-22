__project_origin__ = "added"  # entire file is project-specific, not Jan's original framework

"""
gantry_ss.py
------------
Physical constants and LFR matrices for the dual-gantry system.
Single source of truth for the SubNet augmentation pipeline.

Source: kamtin-fp-model/03 Simulink gantry/main.m (immutable ground truth).
All values must match main.m exactly.

System:
  Inputs  u : 3  (stage forces F_X1, F_X2, F_Y)
  Outputs y : 3  (stage positions X1, X2, Y)
  States  x : 6  (logical: q1, q2, q3, q1_dot, q2_dot, q3_dot)

Provides:
  - Scalar physical parameters
  - M(Y) = M0 + M1*Y + M2*Y^2 decomposition
  - Damping C, stiffness K, coordinate transform P, output Cd/Dd
  - build_poly_constants() — alpha, beta, gamma, N0, N1, N2
  - build_G_matrix_entries() — Ax, Bw, Bu, A_combined for LFR signal flow
  - Module-level precomputed: alpha, beta, gamma, N0, N1, N2, d0, Ax, Bw, Bu, A_combined

All tensors are dtype=torch.float32.
"""

import torch

_D = torch.float32

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
# Sampling constants  (from main.m line 164)
# ----------------------------------------------------------------------
fs = torch.tensor(20e3,     dtype=_D)  # sample frequency  [Hz]
ts = torch.tensor(1 / 20e3, dtype=_D)  # sample period     [s]

# ----------------------------------------------------------------------
# Mass matrix decomposition  M(Y) = M0 + M1*Y + M2*Y^2
# (from main.m lines 52-54)
# ----------------------------------------------------------------------
_z = torch.zeros(3, 3, dtype=_D)

M0 = _z.clone()
M0[0, 0] = m1 + m2 + mb + mh
M0[0, 1] = (m1 - m2) * Lb / 2
M0[1, 0] = (m1 - m2) * Lb / 2
M0[1, 1] = Jb + Jh + (m1 + m2) * Lb**2 / 4 + mh * d**2
M0[1, 2] = -mh * d
M0[2, 1] = -mh * d
M0[2, 2] = mh

M1 = _z.clone()
M1[0, 1] = -mh
M1[1, 0] = -mh

M2 = _z.clone()
M2[1, 1] = mh

# ----------------------------------------------------------------------
# Constant damping matrix C  (from main.m lines 57-59)
# ----------------------------------------------------------------------
C = _z.clone()
C[0, 0] = cg1 + cg2
C[0, 1] = (cg1 - cg2) * Lb / 2
C[1, 0] = (cg1 - cg2) * Lb / 2
C[1, 1] = cb1 + cb2 + (cg1 + cg2) * Lb**2 / 4
C[2, 2] = cy

# ----------------------------------------------------------------------
# Constant stiffness matrix K  (from main.m lines 62-64)
# ----------------------------------------------------------------------
K = _z.clone()
K[1, 1] = kb1 + kb2

# ----------------------------------------------------------------------
# Coordinate transform P  (from main.m lines 98-100)
# Logical forces  = P @ stage forces
# Stage positions = P.T @ logical positions
# ----------------------------------------------------------------------
P = _z.clone()
P[0, 0] = 1.0
P[0, 1] = 1.0
P[1, 0] = Lb / 2
P[1, 1] = -Lb / 2
P[2, 2] = 1.0

# ----------------------------------------------------------------------
# Output matrix  Cd: state x=[q_logical; qdot_logical] -> stage positions
# y = [X1, X2, Y]_stage = P^T @ q_logical
# X1 = q1_log + (Lb/2)*q2_log
# X2 = q1_log - (Lb/2)*q2_log
# Y  = q3_log  (identical in both coordinate systems)
# Cd = [P^T | 0_{3x3}]  (velocities do not appear in the output)
# ----------------------------------------------------------------------
Cd = torch.zeros(3, 6, dtype=_D)
Cd[:, :3] = P.T

Dd = torch.zeros(3, 3, dtype=_D)

# ----------------------------------------------------------------------
# Polynomial constants for the LFR rational form
# M(Y)^{-1} = N(Y)/d(Y)  where N(Y) = N0 + N1*Y + N2*Y^2
# d(Y) = mh*(alpha*gamma - beta^2 + 2*beta*mh*Y + mh*(alpha-mh)*Y^2)
#
# Source: LPV-LFR-derivation (supervisor), verified in lpv_lfr_baseline.
# Mirror of build_poly_constants() in lpv_lfr_baseline/core/physics.py —
# kept here as a function so it can be called with nn.Parameter inputs
# when joint estimation is added (Phase future).
# ----------------------------------------------------------------------
def build_poly_constants(m1_, m2_, mb_, mh_, Jb_, Jh_, Lb_, d_):
    """
    Polynomial constants for LFR rational inverse of M(Y).

    Returns (alpha, beta, gamma, N0, N1, N2).
    All inputs and outputs share the same dtype/device — safe for autograd
    when called with nn.Parameter inputs.

    gamma does NOT include mh*d^2 (see LPV-LFR-Rational-rewrite.md).
    """
    alpha = m1_ + m2_ + mb_ + mh_
    beta  = (m1_ - m2_) * Lb_ / 2
    gamma = Jb_ + Jh_ + (m1_ + m2_) * Lb_**2 / 4  # WITHOUT mh*d^2

    z = torch.zeros((), dtype=m1_.dtype, device=m1_.device)

    N0 = torch.stack([
        torch.stack([mh_ * gamma,      -beta * mh_,                  -beta * d_ * mh_             ]),
        torch.stack([-beta * mh_,       alpha * mh_,                  alpha * d_ * mh_             ]),
        torch.stack([-beta * d_ * mh_,  alpha * d_ * mh_,  alpha * (gamma + mh_ * d_**2) - beta**2]),
    ])
    N1 = torch.stack([
        torch.stack([z,           mh_**2,       d_ * mh_**2  ]),
        torch.stack([mh_**2,      z,             z            ]),
        torch.stack([d_ * mh_**2, z,             2 * beta * mh_]),
    ])
    N2 = torch.stack([
        torch.stack([mh_**2,  z,  z                       ]),
        torch.stack([z,        z,  z                       ]),
        torch.stack([z,        z,  alpha * mh_ - mh_**2   ]),
    ])
    return alpha, beta, gamma, N0, N1, N2


def build_G_matrix_entries(N0_, d0_, M1_, M2_, K_, C_):
    """
    Build constant G matrix entries for the LFR realisation.

    G is built from M0_inv = N0/d0 (Y=0 adjugate/det — purely polynomial,
    no linalg.solve). M1, M2, K, C are passed explicitly so this function
    can be called with trainable parameters for joint estimation.

    Returns (Ax, Bw, Bu, A_combined) — the entries needed for
    xdot = Ax@x + Bw@w + Bu@u_logical  (fused via A_combined).

    Shapes:
        Ax          : (6, 6)
        Bw          : (6, 6)
        Bu          : (6, 3)
        A_combined  : (6, 15)  — [Ax, Bw, Bu] for fused matmul

    Source: lfr_matrices.py in lpv_lfr_baseline (reference only — not imported).
    """
    dtype  = N0_.dtype
    device = N0_.device
    eye3   = torch.eye(3, dtype=dtype, device=device)
    z33    = torch.zeros(3, 3, dtype=dtype, device=device)

    M0inv   = N0_ / d0_       # (3,3)  M0^{-1} = adj(M0)/det(M0) — polynomial, no solve
    M0invK  = M0inv @ K_
    M0invC  = M0inv @ C_
    M0invM1 = M0inv @ M1_
    M0invM2 = M0inv @ M2_

    # Ax = [  0,       I3    ]
    #      [ -M0invK, -M0invC]
    Ax = torch.cat([
        torch.cat([z33,      eye3    ], dim=1),
        torch.cat([-M0invK, -M0invC  ], dim=1),
    ], dim=0)

    # Bw = [  0,        0       ]
    #      [ -M0invM1, -M0invM2 ]
    Bw = torch.cat([
        torch.cat([z33,       z33      ], dim=1),
        torch.cat([-M0invM1, -M0invM2  ], dim=1),
    ], dim=0)

    # Bu = [  0    ]
    #      [ M0inv ]
    Bu = torch.cat([z33, M0inv], dim=0)

    A_combined = torch.cat([Ax, Bw, Bu], dim=1)  # (6, 15)

    return Ax, Bw, Bu, A_combined


