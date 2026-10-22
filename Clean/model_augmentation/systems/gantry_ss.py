"""Physical constants and LFR matrices of the dual-gantry system in float32, from kamtin-fp-model/03 Simulink
gantry/main.m (the immutable ground truth)."""
__project_origin__ = "added"

import torch

_D = torch.float32

mb  = torch.tensor(22.8,   dtype=_D)
mh  = torch.tensor(10.1,   dtype=_D)
m1  = torch.tensor(10.2,   dtype=_D)
m2  = torch.tensor(10.7,   dtype=_D)

Jb  = torch.tensor(1.0,    dtype=_D)
Jh  = torch.tensor(0.05,   dtype=_D)

cg1 = torch.tensor(14.5,   dtype=_D)
cg2 = torch.tensor(20.3,   dtype=_D)
cy  = torch.tensor(10.0,   dtype=_D)

cb1 = torch.tensor(9.0,    dtype=_D)
cb2 = torch.tensor(9.0,    dtype=_D)

kb1 = torch.tensor(1987.5, dtype=_D)
kb2 = torch.tensor(1987.5, dtype=_D)

Lb  = torch.tensor(0.725,  dtype=_D)
d   = torch.tensor(0.1,    dtype=_D)

_z = torch.zeros(3, 3, dtype=_D)

# M(Y) = M0 + Y M1 + Y^2 M2
M1 = _z.clone()
M1[0, 1] = -mh
M1[1, 0] = -mh

M2 = _z.clone()
M2[1, 1] = mh

# damping C and stiffness K of the logical coordinates q
C = _z.clone()
C[0, 0] = cg1 + cg2
C[0, 1] = (cg1 - cg2) * Lb / 2
C[1, 0] = (cg1 - cg2) * Lb / 2
C[1, 1] = cb1 + cb2 + (cg1 + cg2) * Lb**2 / 4
C[2, 2] = cy

K = _z.clone()
K[1, 1] = kb1 + kb2

# stage-to-logical transform: q = P^-T y, y = P^T q
P = _z.clone()
P[0, 0] = 1.0
P[0, 1] = 1.0
P[1, 0] = Lb / 2
P[1, 1] = -Lb / 2
P[2, 2] = 1.0

# output map y = Cd x = P^T q
Cd = torch.zeros(3, 6, dtype=_D)
Cd[:, :3] = P.T

Dd = torch.zeros(3, 3, dtype=_D)

def build_poly_constants(m1_, m2_, mb_, mh_, Jb_, Jh_, Lb_, d_):
    """alpha, beta, gamma and N0, N1, N2 of the rational inverse M(Y)^-1 = (N0 + Y N1 + Y^2 N2) / d(Y)."""
    alpha = m1_ + m2_ + mb_ + mh_
    beta  = (m1_ - m2_) * Lb_ / 2
    gamma = Jb_ + Jh_ + (m1_ + m2_) * Lb_**2 / 4

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
    """Ax, Bw, Bu and A_combined = [Ax Bw Bu] of the LFR, with M0^-1 = N0 / d0."""
    dtype  = N0_.dtype
    device = N0_.device
    eye3   = torch.eye(3, dtype=dtype, device=device)
    z33    = torch.zeros(3, 3, dtype=dtype, device=device)

    M0inv   = N0_ / d0_
    M0invK  = M0inv @ K_
    M0invC  = M0inv @ C_
    M0invM1 = M0inv @ M1_
    M0invM2 = M0inv @ M2_

    Ax = torch.cat([
        torch.cat([z33,      eye3    ], dim=1),
        torch.cat([-M0invK, -M0invC  ], dim=1),
    ], dim=0)

    Bw = torch.cat([
        torch.cat([z33,       z33      ], dim=1),
        torch.cat([-M0invM1, -M0invM2  ], dim=1),
    ], dim=0)

    Bu = torch.cat([z33, M0inv], dim=0)

    A_combined = torch.cat([Ax, Bw, Bu], dim=1)

    return Ax, Bw, Bu, A_combined
