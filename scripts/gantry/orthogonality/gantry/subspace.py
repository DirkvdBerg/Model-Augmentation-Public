"""Subspace comparison WITHOUT ever forming a dense N-by-N projector.

The specification's storage table lists a dense `N x N` weight or projector as PROHIBITED, and
the reason is on record: an earlier version of the geometry materialised one at `N = 7200` and
crashed the machine (gaps.md Sect. 7h). The first version of the verification suite reintroduced
exactly that, in the robustness check, by writing `Q1 @ Q1.T - Q2 @ Q2.T`; at `N = 4800` that is
184 MB per projector plus an equal temporary, six times over. This module exists so the
comparison has one implementation and it is the cheap one.

THE IDENTITY USED. For orthonormal `Q1 (N x k1)` and `Q2 (N x k2)` with projectors `P_i = Q_i
Q_i^T`,

    ||P1 - P2||_2 = max( ||(I - P2) Q1||_2 , ||(I - P1) Q2||_2 )

and `(I - P2) Q1 = Q1 - Q2 (Q2^T Q1)` costs `O(N k1 k2)`. At equal ranks this is the sine of the
largest principal angle; at different ranks it is 1, which is the correct answer and is why the
rank is always reported next to it rather than the angle alone.
"""
__project_origin__ = "added"

import numpy as np


def projector_distance(Q1, Q2):
    """||Q1 Q1^T - Q2 Q2^T||_2, computed in the factors."""
    Q1 = np.asarray(Q1, dtype=np.float64)
    Q2 = np.asarray(Q2, dtype=np.float64)
    if Q1.shape[1] == 0 and Q2.shape[1] == 0:
        return 0.0
    if Q1.shape[1] == 0 or Q2.shape[1] == 0:
        return 1.0
    r1 = np.linalg.norm(Q1 - Q2 @ (Q2.T @ Q1), 2)
    r2 = np.linalg.norm(Q2 - Q1 @ (Q1.T @ Q2), 2)
    return float(max(r1, r2))


def principal_angles_deg(Q1, Q2):
    """Principal angles in degrees, for equal-rank reporting alongside the distance."""
    Q1 = np.asarray(Q1, dtype=np.float64)
    Q2 = np.asarray(Q2, dtype=np.float64)
    if Q1.shape[1] == 0 or Q2.shape[1] == 0:
        return []
    c = np.clip(np.linalg.svd(Q1.T @ Q2, compute_uv=False), -1.0, 1.0)
    return np.degrees(np.arccos(c)).tolist()


def residual_outside(Q_sub, Q_big):
    """||(I - P_big) Q_sub||_2: how far `range(Q_sub)` sticks out of `range(Q_big)`.

    Zero means containment. Used for the `range(E) subseteq range(blockdiag Gamma_w)` check,
    which is a one-sided statement and must not be tested with a symmetric distance.
    """
    Q_sub = np.asarray(Q_sub, dtype=np.float64)
    Q_big = np.asarray(Q_big, dtype=np.float64)
    if Q_sub.shape[1] == 0:
        return 0.0
    if Q_big.shape[1] == 0:
        return float(np.linalg.norm(Q_sub, 2))
    return float(np.linalg.norm(Q_sub - Q_big @ (Q_big.T @ Q_sub), 2))
