"""Nominal parameters theta^0 (Garcia-Herreros et al., Table I) and the baseline matrices of the gantry, in float32,
from kamtin-fp-model/03 Simulink gantry/main.m (the immutable ground truth). Code names: alpha = m_Sigma + m_h,
beta = (L_b/2) m_Delta, gamma = J_eff."""
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
# M_1 = -m_h (e1 e2^T + e2 e1^T), M_2 = m_h e2 e2^T
M1[0, 1] = -mh
M1[1, 0] = -mh

M2 = _z.clone()
M2[1, 1] = mh

# damping C and stiffness K of the generalised coordinates q
C = _z.clone()
# C = [[c_g1 + c_g2, (L_b/2)(c_g1 - c_g2), 0], [., c_b1 + c_b2 + (L_b^2/4)(c_g1 + c_g2), 0], [0, 0, c_y]]
C[0, 0] = cg1 + cg2
C[0, 1] = (cg1 - cg2) * Lb / 2
C[1, 0] = (cg1 - cg2) * Lb / 2
C[1, 1] = cb1 + cb2 + (cg1 + cg2) * Lb**2 / 4
C[2, 2] = cy

K = _z.clone()
# K = diag(0, k_b1 + k_b2, 0)
K[1, 1] = kb1 + kb2

# coordinate and force transformation: q = P^-T q_act, u = P u_act
P = _z.clone()
P[0, 0] = 1.0
P[0, 1] = 1.0
P[1, 0] = Lb / 2
P[1, 1] = -Lb / 2
P[2, 2] = 1.0

# output map y_hat = Cd x_tilde = P^T q, D = 0
Cd = torch.zeros(3, 6, dtype=_D)
Cd[:, :3] = P.T

Dd = torch.zeros(3, 3, dtype=_D)

def build_poly_constants(m1_, m2_, mb_, mh_, Jb_, Jh_, Lb_, d_):
    """Rational inverse of the inertia matrix.

    M(Y)^-1 = (N0 + Y N1 + Y^2 N2) / d(Y), with N0 + Y N1 + Y^2 N2 = adj M(Y) and
    d(Y) = m_h (alpha gamma - beta^2 + 2 beta m_h Y + m_h (alpha - m_h) Y^2) formed by the caller.

    Args:
        m1_, m2_, mb_, mh_: m_1, m_2, m_b, m_h, kg (scalar tensors).
        Jb_, Jh_: J_b, J_h, kg m^2.
        Lb_, d_: L_b, d, metres.
    Returns:
        alpha = m_Sigma + m_h, beta = (L_b/2) m_Delta, gamma = J_eff (scalars) and N0, N1, N2 (3, 3), physical.
    """
    # M(Y) = [[alpha, beta - m_h Y, 0], [beta - m_h Y, gamma + m_h (d^2 + Y^2), -m_h d], [0, -m_h d, m_h]].
    # alpha = m_Sigma + m_h, beta = (L_b/2) m_Delta, gamma = J_eff.
    alpha = m1_ + m2_ + mb_ + mh_
    beta  = (m1_ - m2_) * Lb_ / 2
    gamma = Jb_ + Jh_ + (m1_ + m2_) * Lb_**2 / 4

    z = torch.zeros((), dtype=m1_.dtype, device=m1_.device)

    # Adjugate coefficients of Y^0, Y^1, Y^2.
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
    """State row of the LFR G.

    A_x = [0 I; -R K -R C], B_w = [0 0; -R M_1 -R M_2], B_u = [0; R], R = M_0^-1 = N0 / d0; the default vector field
    evaluates xdot = A_combined (x, w, u) with A_combined = [A_x B_w B_u].

    Args:
        N0_: (3, 3) adj M_0.
        d0_: scalar det M_0.
        M1_, M2_: (3, 3) Y-dependent inertia parts.
        K_, C_: (3, 3) stiffness and damping.
    Returns:
        A_x (6, 6), B_w (6, 6), B_u (6, 3), A_combined (6, 15): the physical state row of G.
    """
    dtype  = N0_.dtype
    device = N0_.device
    eye3   = torch.eye(3, dtype=dtype, device=device)
    z33    = torch.zeros(3, 3, dtype=dtype, device=device)

    # R = M_0^-1 = N0 / d0.
    M0inv   = N0_ / d0_
    M0invK  = M0inv @ K_
    M0invC  = M0inv @ C_
    M0invM1 = M0inv @ M1_
    M0invM2 = M0inv @ M2_

    # A_x = [0 I; -R K -R C], B_w = [0 0; -R M_1 -R M_2], B_u = [0; R].
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
