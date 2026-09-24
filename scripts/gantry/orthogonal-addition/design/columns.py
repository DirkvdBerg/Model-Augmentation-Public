"""Candidate-addition column library (OA-006): which generalised forces the addition may contain.

LEGITIMACY RULE (OA-006). A stateless term may only occupy a STRUCTURAL ZERO of the baseline:
an (entry, Y-power) slot of M(Y), C or K that is identically zero for EVERY parameter value.
Anything else is a parameter change in disguise (the truth's physical parameter would then be
the shifted one, and "exact recovery" would be a relabelling). Baseline slots, from
gantrySystem.m:
    M11: Y^0          M12: Y^0, Y^1     M22: Y^0, Y^2      M23: Y^0     M33: Y^0     M13: none
    C11, C12, C22, C33: Y^0                                  C13, C23: none
    K22: Y^0                                                  all other K: none
Physical terms are named after the mechanism; the rest of the library is the "worked example"
extension used only by route B.
"""
__project_origin__ = "added"

import numpy as np

# slot -> (matrix, i, j, ypow); the unit column is f = -(E_ij v) Y^ypow, E_ij symmetric
BASELINE_SLOTS = {('M', 0, 0, 0), ('M', 0, 1, 0), ('M', 0, 1, 1), ('M', 1, 1, 0), ('M', 1, 1, 2),
                  ('M', 1, 2, 0), ('M', 2, 2, 0),
                  ('C', 0, 0, 0), ('C', 0, 1, 0), ('C', 1, 1, 0), ('C', 2, 2, 0),
                  ('K', 1, 1, 0)}


def is_legit(slot):
    return slot not in BASELINE_SLOTS


def stateless_library(max_ypow=2, include_damping=False):
    """Every legitimate symmetric stateless slot up to Y^max_ypow (K and M; C optional).

    The truth EOM (gantrySystemOA.m) carries K_a and M_a up to Y^2 (OA-018) and C_a at Y^0, so the
    library is capped to what the MATLAB plant can realise.
    """
    out = []
    mats = ['K', 'M'] + (['C'] if include_damping else [])
    cap = {'K': max_ypow, 'M': max_ypow, 'C': 0}          # OA-018: M up to Y^2 (oa_Ma2)
    for m in mats:
        for i in range(3):
            for j in range(i, 3):
                for p in range(cap[m] + 1):
                    s = (m, i, j, p)
                    if is_legit(s):
                        out.append(s)
    return out


def slot_name(s):
    m, i, j, p = s
    return '%s%d%d%s' % (m, i + 1, j + 1, '' if p == 0 else ('Y' if p == 1 else 'Y%d' % p))


def physical_mount(pb):
    """The physically named stateless terms, as force fields on the tuples (unit coefficient).

    k_x   X spring to ground at the beam centre (Theta lever 0):        K11 = k
    k_y   Y spring to ground at the payload's rotation-neutral point:   K33 = k
    k_xp  X spring to ground at the payload, b = [1, -Y, 0]:            K = k b b^T
    gamma skew of the Y guide by gamma rad (sympy check D-skew):       M13 = mh gamma,
                                                                        M22 += -2 mh d gamma Y
    """
    from model_augmentation.systems import gantry_ss as gss
    mh, d = float(gss.mh), float(gss.d)
    q, qdd, Y = pb.q, pb.qdd, pb.Y
    f = {}
    z = np.zeros_like(q)
    fx = z.copy(); fx[:, 0] = -q[:, 0]
    f['k_x'] = fx
    fy = z.copy(); fy[:, 2] = -q[:, 2]
    f['k_y'] = fy
    bq = q[:, 0] - Y * q[:, 1]                      # b^T q, b = [1, -Y, 0]
    fxp = z.copy(); fxp[:, 0] = -bq; fxp[:, 1] = Y * bq
    f['k_xp'] = fxp
    fg = z.copy()
    fg[:, 0] = -mh * qdd[:, 2]                      # -(M13) Ydd on X
    fg[:, 2] = -mh * qdd[:, 0]                      # -(M13) Xdd on Y
    fg[:, 1] = 2.0 * mh * d * Y * qdd[:, 1]         # -(M22 += -2 mh d Y) Thetadd on Theta
    f['gamma'] = fg
    return f


def absorber_b(alpha, lever, mount='payload'):
    """Direction of a point absorber in logical coordinates, b(Y) = b0 + Y b1.

    payload: moves along in-plane direction alpha; its X velocity carries -(Y + e_y) Thetadot and
             its Y velocity (e_x - d) Thetadot, so b = [cos a, lever - Y cos a, sin a], with the
             constant offsets absorbed in `lever`.
    beam:    on the beam, X direction only: b = [1, lever, 0].
    """
    if mount == 'beam':
        return np.array([1.0, lever, 0.0]), np.zeros(3)
    return (np.array([np.cos(alpha), lever, np.sin(alpha)]),
            np.array([0.0, -np.cos(alpha), 0.0]))
