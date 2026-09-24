"""Reconstructability maps for the encoder-transient study (ET-003).

One builder, several models. The map is Hoekstra 2026's linear reconstructability map, built with
the SAME formulas and the SAME normalised I/O frame as `linear_encoder_init_aug`
(`vendor/model_augmentation/fit_systems/pre_encoder.py`), but in float64 and for any (A, B, C, D):

    W_y = A^n O_n^+                         # THEORY: Hoekstra 2026 Eq. 17
    W_u = -A^n O_n^+ T_n + r_n              # THEORY: Hoekstra 2026 Eq. 16 (T_n Eq. 10, r_n Eq. 13-14)

x(k) = W_u u[k-n..k] + W_y y[k-n..k], oldest sample first, in the pure-scaled frame
(u / std_u, y / ystd, x / std_x), exactly the frame the pipeline encoder's W^b lives in (D-017,
D-055). `physical_map` folds the scalings in, so a map is applied to PHYSICAL windows and returns a
PHYSICAL state; scoring then never depends on a normalisation convention.

Models:
  baseline_dt(dt, Y)  the pipeline's frozen-Y baseline linearisation (6 states, D-167 closed form)
  planted_dt(dt, Y)   the planted 8-state plant (`truth.Plant8`, dataset absorber) linearised at
                      q = [0, 0, Y, 0], qdot = 0, u = 0, in MODEL order [X,Th,Y,dX,dTh,dY,da,vda]
"""
__project_origin__ = "added"

import os
import sys

import numpy as np
import scipy.signal

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import et_paths                                                  # noqa: E402,F401

from model_augmentation.systems import gantry_ss as gss                          # noqa: E402
from model_augmentation.systems.gantry_linearization import (                    # noqa: E402
    gantry_linearize_and_discretize)
import truth                                                     # noqa: E402
from core import recon_map                                       # noqa: E402,F401  (ET-006: moved)

P_np = gss.P.numpy().astype(np.float64)
C_POS = gss.Cd.numpy().astype(np.float64)[:, :3]                # y_stage = P^T q  (3x3)
# truth order [X,Th,Y,da,dX,dTh,dY,vda] -> model order [X,Th,Y,dX,dTh,dY,da,vda]
T2M = np.array([0, 1, 2, 4, 5, 6, 3, 7])
S_T2M = np.eye(8)[T2M]                                           # x_model = S x_truth


def baseline_dt(dt, Y=0.0):
    """(Ad, Bd, Cd, Dd), 6 states, stage input: the vendored production linearisation."""
    return gantry_linearize_and_discretize(dt=dt, Y_op=float(Y))


def planted_ct(Y=0.0, plant=None):
    """Continuous-time 8-state linearisation of the planted plant, MODEL order, stage input.

    # THEORY: at q = [0,0,Y,0], qdot = 0, u = 0 the EOM M(Y,da) qdd = E u_log - K q - C qdot has
    # zero right-hand side, so d(M^-1)/dq drops out and the Jacobian is the frozen-Y form
    # [[0, I], [-M^-1 K, -M^-1 C]], [[0], [M^-1 E P]]. Same reasoning as the 6-state D-167 map.
    """
    p = plant or truth.plant_for(et_paths.MODE)
    Minv = np.linalg.inv(p.M(float(Y), 0.0))
    A = np.zeros((8, 8))
    A[:4, 4:] = np.eye(4)
    A[4:, :4] = -Minv @ p._K4
    A[4:, 4:] = -Minv @ p._C4
    B = np.zeros((8, 3))
    B[4:, :] = Minv @ p._E43 @ P_np                              # stage -> logical force
    C = np.zeros((3, 8))
    C[:, :3] = C_POS
    return S_T2M @ A @ S_T2M.T, S_T2M @ B, C @ S_T2M.T, np.zeros((3, 3))


def planted_dt(dt, Y=0.0, plant=None):
    A, B, C, D = planted_ct(Y, plant)
    Ad, Bd, Cd, Dd, _ = scipy.signal.cont2discrete((A, B, C, D), dt, method='zoh')
    return Ad, Bd, Cd, Dd


def physical_map(Ad, Bd, Cd, Dd, n, sx, su, sy):
    """Map from PHYSICAL windows to the PHYSICAL state, built in the pipeline's normalised frame.

    sx (nx,), su (nu,), sy (ny,): the stds `normalize_linear_ss_matrices` uses (plain np.std of the
    training arrays). Returns (Wy_phys (nx, (n+1)ny), Wu_phys (nx, (n+1)nu)).
    """
    Tx, Tix = np.diag(1 / sx), np.diag(sx)
    Tu, Tiu = np.diag(1 / su), np.diag(su)
    Ty, Tiy = np.diag(1 / sy), np.diag(sy)
    Wy, Wu = recon_map(Tx @ Ad @ Tix, Tx @ Bd @ Tiu, Ty @ Cd @ Tix, Ty @ Dd @ Tiu, n)
    return Tix @ Wy @ np.diag(np.tile(1 / sy, n + 1)), Tix @ Wu @ np.diag(np.tile(1 / su, n + 1))


def apply_map(Wy, Wu, ywin, uwin):
    """ywin (N, n+1, ny), uwin (N, n+1, nu), physical, oldest first -> x (N, nx) physical."""
    N = len(ywin)
    return ywin.reshape(N, -1) @ Wy.T + uwin.reshape(N, -1) @ Wu.T


class ScheduledMap:
    """Exact frozen-Y maps on a grid, linearly interpolated at a per-window Y (ET-003 variant A)."""

    def __init__(self, builder, dt, n, sx, su, sy, lo=-0.40, hi=0.40, step=0.005):
        # HEURISTIC: 0.005 m grid over [-0.40, 0.40] covers the measured range [-0.36, 0.36]
        # with 161 nodes; interpolation error is checked against an exact map mid-node.
        self.nodes = np.round(np.arange(lo, hi + step / 2, step), 10)
        maps = [physical_map(*builder(dt, Y), n, sx, su, sy) for Y in self.nodes]
        self.Wy = np.stack([m[0] for m in maps])
        self.Wu = np.stack([m[1] for m in maps])
        self.lo, self.step = lo, step

    def apply(self, ywin, uwin, Y):
        Y = np.clip(np.asarray(Y, float), self.nodes[0], self.nodes[-1] - 1e-12)
        pos = (Y - self.lo) / self.step
        i = np.floor(pos).astype(int)
        t = (pos - i)[:, None]
        N = len(ywin)
        yf, uf = ywin.reshape(N, -1), uwin.reshape(N, -1)
        x = np.zeros((N, self.Wy.shape[1]))
        for node in np.unique(i):
            m = i == node
            x0 = yf[m] @ self.Wy[node].T + uf[m] @ self.Wu[node].T
            x1 = yf[m] @ self.Wy[node + 1].T + uf[m] @ self.Wu[node + 1].T
            x[m] = (1 - t[m]) * x0 + t[m] * x1
        return x
