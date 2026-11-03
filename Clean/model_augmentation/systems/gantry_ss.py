"""Nominal parameters theta^0 (Garcia-Herreros et al., Table I) and the baseline matrices of the gantry, in float64.
Code names: alpha = m_Sigma + m_h, beta = (L_b/2) m_Delta, gamma = J_eff."""
__project_origin__ = "added"

import numpy as np
import torch

# float64, so theta^0 equals the values of the data generator (data_generation/tdg_config.m) exactly
_D = torch.float64

# nominal parameters theta^0 (THEORY: Garcia-Herreros et al. Table I)
mb  = torch.tensor(22.8,   dtype=_D)  # [kg] cross-arm mass
mh  = torch.tensor(10.1,   dtype=_D)  # [kg] payload mass
m1  = torch.tensor(10.2,   dtype=_D)  # [kg] X1 carriage mass
m2  = torch.tensor(10.7,   dtype=_D)  # [kg] X2 carriage mass

Jb  = torch.tensor(1.0,    dtype=_D)  # [kg m^2] cross-arm inertia about the Theta axis
Jh  = torch.tensor(0.05,   dtype=_D)  # [kg m^2] payload inertia about the Theta axis

cg1 = torch.tensor(14.5,   dtype=_D)  # [N s/m] guide damping of the X1 carriage
cg2 = torch.tensor(20.3,   dtype=_D)  # [N s/m] guide damping of the X2 carriage
cy  = torch.tensor(10.0,   dtype=_D)  # [N s/m] guide damping of the payload along Y

# the two cross-arm-carriage joints; only the sums enter the model
cb1 = torch.tensor(9.0,    dtype=_D)  # [N m s/rad] rotational damping, joint 1
cb2 = torch.tensor(9.0,    dtype=_D)  # [N m s/rad] rotational damping, joint 2

kb1 = torch.tensor(1987.5, dtype=_D)  # [N m/rad] torsional stiffness, joint 1
kb2 = torch.tensor(1987.5, dtype=_D)  # [N m/rad] torsional stiffness, joint 2

Lb  = torch.tensor(0.725,  dtype=_D)  # [m] distance between the X1 and X2 actuators
d   = torch.tensor(0.1,    dtype=_D)  # [m] offset of the payload from the cross-arm axis

# q = P^-T q_act and u = P u_act; q_act = (X1, X2, Y), u_act = (F_X1, F_X2, F_Y), u = (F_X, torque on Theta, F_Y)
P = torch.zeros(3, 3, dtype=_D)
P[0, 0] = 1.0
P[0, 1] = 1.0
P[1, 0] = Lb / 2
P[1, 1] = -Lb / 2
P[2, 2] = 1.0
# P^-T written out: X = (X1 + X2)/2, Theta = (X1 - X2)/L_b, Y = Y
P_INV_T = np.array([[0.5, 0.5, 0.0], [1 / float(Lb), -1 / float(Lb), 0.0], [0.0, 0.0, 1.0]])

# output map y_hat = Cd x_tilde = P^T q, D = 0
Cd = torch.zeros(3, 6, dtype=_D)
Cd[:, :3] = P.T

Dd = torch.zeros(3, 3, dtype=_D)


def damping_and_stiffness(kb1, kb2, cg1, cg2, cy, cb1, cb2, Lb):
    """Stiffness K and damping C in generalised coordinates (X, Theta, Y); the joint pairs enter only through their
    sums.

    Args:
        kb1, kb2: torsional stiffness of the joints [N m/rad] (scalar tensors).
        cg1, cg2, cy: guide damping of the X1 and X2 carriages and of the payload along Y [N s/m].
        cb1, cb2: rotational damping of the joints [N m s/rad].
        Lb: distance between the X1 and X2 actuators [m].
    Returns:
        K and C, each (3, 3).
    """
    z = kb1.new_zeros(())
    kb_sum, cb_sum = kb1 + kb2, cb1 + cb2
    K = torch.stack([
        torch.stack([z, z,      z]),
        torch.stack([z, kb_sum, z]),
        torch.stack([z, z,      z]),
    ])
    C = torch.stack([
        torch.stack([cg1 + cg2,            (cg1 - cg2) * Lb / 2,                z]),
        torch.stack([(cg1 - cg2) * Lb / 2, cb_sum + (cg1 + cg2) * Lb ** 2 / 4,  z]),
        torch.stack([z,                    z,                                   cy]),
    ])
    return K, C


def inertia_constants(m1, m2, mb, mh, Jb, Jh, Lb):
    """The three mass combinations of the inertia matrix: alpha = m_Sigma + m_h, beta = (L_b/2) m_Delta and
    gamma = J_eff.

    Args:
        m1, m2, mb, mh: masses of the X1 and X2 carriages, the cross-arm and the payload [kg] (scalar tensors).
        Jb, Jh: inertia of the cross-arm and the payload about the Theta axis [kg m^2].
        Lb: distance between the X1 and X2 actuators [m].
    Returns:
        alpha [kg], beta [kg m], gamma [kg m^2].
    """
    alpha = m1 + m2 + mb + mh
    beta = (m1 - m2) * Lb / 2
    gamma = Jb + Jh + (m1 + m2) * Lb**2 / 4
    return alpha, beta, gamma


def inertia_parts(mh):
    """The parts of the inertia matrix that change with the payload position, M(Y) = M_0 + Y M_1 + Y^2 M_2.

    Args:
        mh: payload mass m_h [kg] (scalar tensor).
    Returns:
        M_1 and M_2, each (3, 3).
    """
    z = mh.new_zeros(())
    M1 = torch.stack([
        torch.stack([z,   -mh,  z]),
        torch.stack([-mh,  z,   z]),
        torch.stack([z,    z,   z]),
    ])
    M2 = torch.stack([
        torch.stack([z,   z,   z]),
        torch.stack([z,   mh,  z]),
        torch.stack([z,   z,   z]),
    ])
    return M1, M2


def inertia_matrix(Y, m1, m2, mb, mh, Jb, Jh, Lb, d):
    """Inertia matrix M(Y) = M_0 + Y M_1 + Y^2 M_2 in generalised coordinates (X, Theta, Y), with the payload at Y.

    Args:
        Y: payload position along the cross-arm [m].
        m1, m2, mb, mh, Jb, Jh, Lb: as in inertia_constants (scalar tensors).
        d: offset of the payload from the cross-arm axis [m].
    Returns:
        (3, 3) M(Y).
    """
    alpha, beta, gamma = inertia_constants(m1, m2, mb, mh, Jb, Jh, Lb)
    z = mh.new_zeros(())
    M0 = torch.stack([
        torch.stack([alpha, beta,               z]),
        torch.stack([beta,  gamma + mh * d**2,  -mh * d]),
        torch.stack([z,     -mh * d,            mh]),
    ])
    M1, M2 = inertia_parts(mh)
    return M0 + Y * M1 + Y**2 * M2


def build_poly_constants(m1_, m2_, mb_, mh_, Jb_, Jh_, Lb_, d_):
    """Rational inverse of the inertia matrix.

    M(Y)^-1 = (N0 + Y N1 + Y^2 N2) / d(Y), with N0 + Y N1 + Y^2 N2 = adj M(Y) and
    d(Y) = m_h (alpha gamma - beta^2 + 2 beta m_h Y + m_h (alpha - m_h) Y^2) formed by the caller.

    Args:
        m1_, m2_, mb_, mh_: m_1, m_2, m_b, m_h [kg] (scalar tensors).
        Jb_, Jh_: J_b, J_h [kg m^2].
        Lb_, d_: L_b, d [m].
    Returns:
        alpha = m_Sigma + m_h, beta = (L_b/2) m_Delta, gamma = J_eff (scalars) and N0, N1, N2 (3, 3), physical.
    """
    # the adjugate of M(Y) as built in inertia_matrix
    alpha, beta, gamma = inertia_constants(m1_, m2_, mb_, mh_, Jb_, Jh_, Lb_)

    z = torch.zeros((), dtype=m1_.dtype, device=m1_.device)

    # adjugate coefficients of Y^0, Y^1, Y^2
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

    A_x = [0 I; -R K -R C], B_w = [0 0; -R M_1 -R M_2], B_u = [0; R], R = M_0^-1 = N0 / d0;
    xdot = A_combined (x, w, u), A_combined = [A_x B_w B_u].

    Args:
        N0_: (3, 3) adj M_0.
        d0_: scalar det M_0.
        M1_, M2_: (3, 3) Y-dependent inertia parts, M(Y) = M_0 + Y M_1 + Y^2 M_2.
        K_, C_: (3, 3) stiffness and damping.
    Returns:
        A_x (6, 6), B_w (6, 6), B_u (6, 3), A_combined (6, 15), continuous time, xdot = (qdot, qddot);
        B_u acts on u = P u_act.
    """
    dtype  = N0_.dtype
    device = N0_.device
    eye3   = torch.eye(3, dtype=dtype, device=device)
    z33    = torch.zeros(3, 3, dtype=dtype, device=device)

    # R = M_0^-1 = N0 / d0
    M0inv   = N0_ / d0_
    M0invK  = M0inv @ K_
    M0invC  = M0inv @ C_
    M0invM1 = M0inv @ M1_
    M0invM2 = M0inv @ M2_

    # A_x = [0 I; -R K -R C], B_w = [0 0; -R M_1 -R M_2], B_u = [0; R]
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
