"""Discretised truth helpers, shared by pretrain_diagnostic.py and bars.py.

Separate module for the same reason as data.py: two files must use one definition of the
discretised truth, and a duplicated copy would let them drift apart silently.
"""
__project_origin__ = "added"

import os
import sys
import numpy as np
from scipy.linalg import expm
from scipy.signal import decimate

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'scripts', 'gantry', 'msd-offset'))
import plant

FS_NATIVE = 4000.0


def truth_ct(Y):
    """Continuous-time truth linearised at frozen Y, absorber at rest. x = [q; qdot], q logical."""
    Minv = np.linalg.inv(plant.M8(Y, 0.0, freeze=True))
    A = np.block([[np.zeros((4, 4)), np.eye(4)],
                  [-Minv @ plant._K4, -Minv @ plant._C4]])
    B = np.vstack([np.zeros((4, 3)), Minv @ plant._E43])
    C = np.hstack([plant.P_np.T @ np.hstack([np.eye(3), np.zeros((3, 1))]), np.zeros((3, 4))])
    return A, B, C                                 # y = C x is STAGE position [x1, x2, Y], metres


def discretise(A, B, ts):
    """Exact ZOH via the block-matrix exponential."""
    n, m = A.shape[0], B.shape[1]
    E = expm(np.block([[A, B], [np.zeros((m, n + m))]]) * ts)
    return E[:n, :n], E[:n, n:]


def free_run(Ad, Bd, C, x0, u, dtype):
    Ad, Bd, C = Ad.astype(dtype), Bd.astype(dtype), C.astype(dtype)
    x = np.asarray(x0, dtype)
    out = np.empty((len(u), C.shape[0]), dtype)
    u = u.astype(dtype)
    for k in range(len(u)):
        out[k] = C @ x
        x = Ad @ x + Bd @ u[k]
    return out.astype(np.float64)


def true_states(name, fs):
    """The exact 8-state truth from the record, decimated like the outputs.

    x = [X, Th, Y, da, dX, dTh, dY, vda] (plant.py docstring). The record stores x_logical as
    [q_logical, qdot_logical] plus delta_a and vdelta_a separately, so it is reassembled here.
    """
    rec = plant.load_record(name, fs_new=int(FS_NATIVE))
    xl, da, vda = rec['x_logical'], rec['delta_a'], rec['vdelta_a']
    x = np.column_stack([xl[:, 0:3], da, xl[:, 3:6], vda])
    q = int(round(FS_NATIVE / fs))
    if q > 1:
        x = decimate(x, q, ftype='fir', zero_phase=True, axis=0)
    return np.ascontiguousarray(x, dtype=float)
