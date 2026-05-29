"""
gantry_ss.py
------------
Physical constants for the dual-gantry system — single source of truth
for the SubNet augmentation pipeline.

Source: kamtin-fp-model/03 Simulink gantry/main.m (immutable ground truth).
All values must match main.m exactly.

System:
  Inputs  u : 3  (stage forces F_X1, F_X2, F_Y)
  Outputs y : 3  (stage positions X1, X2, Y)
  States  x : 6  (logical: q1, q2, q3, q1_dot, q2_dot, q3_dot)

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
# Output matrix  Cd: state x=[q;qdot] -> logical positions q
# Stage positions = P.T @ q, handled in Linear_Output_Block(Cd, Dd)
# ----------------------------------------------------------------------
Cd = torch.zeros(3, 6, dtype=_D)
Cd[0, 0] = 1.0
Cd[1, 1] = 1.0
Cd[2, 2] = 1.0

Dd = torch.zeros(3, 3, dtype=_D)
