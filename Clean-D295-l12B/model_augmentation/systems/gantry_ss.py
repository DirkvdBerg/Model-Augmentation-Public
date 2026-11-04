"""Nominal parameters theta^0 (Garcia-Herreros et al., Table I) and the baseline matrices of the gantry, in float64.
Code names: alpha = m_Sigma + m_h, beta = (L_b/2) m_Delta, gamma = J_eff."""
__project_origin__ = "added"

import numpy as np
import torch

# float64, so theta^0 equals the values of the data generator (data_generation/tdg_config.m) exactly
_D = torch.float64

# nominal parameters theta^0
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
P = torch.tensor([[1.0, 1.0, 0.0], [float(Lb) / 2, -float(Lb) / 2, 0.0], [0.0, 0.0, 1.0]], dtype=_D)
# P^-T written out: X = (X1 + X2)/2, Theta = (X1 - X2)/L_b, Y = Y
P_INV_T = np.linalg.inv(P.detach().cpu().numpy().astype(np.float32).astype(np.float64).T)

# output map y_hat = Cd x_tilde = P^T q, D = 0
Cd = np.hstack([P.numpy().T, np.zeros((3, 3), P.numpy().dtype)])
Dd = np.zeros((3, 3), P.numpy().dtype)


def stiffness_and_damping(kb1, kb2, cg1, cg2, cy, cb1, cb2, Lb):
    """Stiffness K and damping C in generalised coordinates (X, Theta, Y); the joints enter only through their sums.

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
    """The three mass terms of the inertia matrix: alpha the total mass, beta the carriage mass difference times
    L_b/2 and gamma the effective inertia about the Theta axis (J_eff).

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


def inertia_adjugate(alpha, beta, gamma, mh, d):
    """Adjugate of the inertia matrix in parts, adj M(Y) = N_0 + Y N_1 + Y^2 N_2, so M(Y)^-1 = adj M(Y) / det M(Y);
    the entries are the cofactors of inertia_matrix.

    Args:
        alpha, beta, gamma: as returned by inertia_constants.
        mh: payload mass m_h [kg] (scalar tensor).
        d: offset of the payload from the cross-arm axis [m].
    Returns:
        N_0, N_1 and N_2, each (3, 3).
    """
    z = mh.new_zeros(())
    N0 = torch.stack([
        torch.stack([mh * gamma,      -beta * mh,      -beta * d * mh]),
        torch.stack([-beta * mh,       alpha * mh,      alpha * d * mh]),
        torch.stack([-beta * d * mh,   alpha * d * mh,  alpha * (gamma + mh * d**2) - beta**2]),
    ])
    N1 = torch.stack([
        torch.stack([z,           mh**2,  d * mh**2]),
        torch.stack([mh**2,       z,      z]),
        torch.stack([d * mh**2,   z,      2 * beta * mh]),
    ])
    N2 = torch.stack([
        torch.stack([mh**2,  z,  z]),
        torch.stack([z,      z,  z]),
        torch.stack([z,      z,  alpha * mh - mh**2]),
    ])
    return N0, N1, N2


def inertia_determinant(Y, alpha, beta, gamma, mh):
    """Determinant of the inertia matrix, det M(Y) = m_h (alpha gamma - beta^2 + 2 beta m_h Y + m_h (alpha - m_h) Y^2).

    Args:
        Y: payload position along the cross-arm [m].
        alpha, beta, gamma: as returned by inertia_constants.
        mh: payload mass m_h [kg].
    Returns:
        det M(Y), with the shape of Y.
    """
    return mh * (alpha * gamma - beta ** 2
                 + 2 * beta * mh * Y
                 + mh * (alpha - mh) * Y ** 2)


def lfr_state_matrix(M0_inv, M1, M2, K, C):
    """State matrix of the baseline LFR, xdot = A_combined (x, w, u), with x = (q, qdot), xdot = (qdot, qddot) and
    the latent signal w = (Y qddot, Y^2 qddot). It is M(Y) qddot = u - C qdot - K q with the Y-dependent parts moved
    to the right-hand side: M_0 qddot = u - C qdot - K q - M_1 w_1 - M_2 w_2.

    Args:
        M0_inv: (3, 3) inverse of M_0, the part of the inertia matrix that does not change with Y.
        M1, M2: (3, 3) as returned by inertia_parts.
        K, C: (3, 3) stiffness and damping.
    Returns:
        (6, 15) A_combined = [A_x B_w B_u], continuous time; B_u acts on u = P u_act.
    """
    eye3 = torch.eye(3, dtype=M0_inv.dtype, device=M0_inv.device)
    z33 = torch.zeros(3, 3, dtype=M0_inv.dtype, device=M0_inv.device)
    M0invK = M0_inv @ K
    M0invC = M0_inv @ C
    M0invM1 = M0_inv @ M1
    M0invM2 = M0_inv @ M2

    Ax = torch.cat([
        torch.cat([z33,      eye3    ], dim=1),
        torch.cat([-M0invK, -M0invC  ], dim=1),
    ], dim=0)

    Bw = torch.cat([
        torch.cat([z33,       z33      ], dim=1),
        torch.cat([-M0invM1, -M0invM2  ], dim=1),
    ], dim=0)

    Bu = torch.cat([z33, M0_inv], dim=0)
    return torch.cat([Ax, Bw, Bu], dim=1)
