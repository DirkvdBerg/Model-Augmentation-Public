"""The Telica loop as linear discrete-time models at 20 kHz (CN-005).

    e = r - (q + v),  u = K e,  S = (I + G K)^-1,  stage axes [X1, X2, Y].

G is the Jacobian of the identified attempt-2 block's RK4 step (the exact map `cl_sim.py`
iterates) at standstill; three readings of the standstill plant: 'sliding' (friction off),
'tanh' (the identified tanh law linearised at v = 0) and 'stuck' (G = 0). K is the identified
controller (telica-real G3), error [m] -> stage force [N]; `ctrl_ss(amps=True)` gives [A].
"""
import json
import os

import numpy as np
import torch

from params import telica_params as tp
from baseline.cl_sim import make_block, P as P_STAGE
from controller.telica_bank import physical_ss

D = torch.float64
HERE_TR = os.path.dirname(os.path.dirname(os.path.abspath(__import__('baseline').__file__)))
A2_FILE = os.path.join(HERE_TR, 'baseline', 'recovered_params_a2.json')
READINGS = ('sliding', 'tanh', 'stuck')
TS = tp.TS


def a2_params():
    R = json.load(open(A2_FILE))
    return R['raw14'], tuple(R['cc'])


_BLOCKS = {}


def block(reading):
    if reading not in _BLOCKS:
        raw, cc = a2_params()
        if reading == 'sliding':
            _BLOCKS[reading] = make_block(raw=raw, cc=(0.0, 0.0, 0.0), mode='none')
        elif reading == 'tanh':
            _BLOCKS[reading] = make_block(raw=raw, cc=cc, mode='tanh')
        else:
            raise ValueError(reading)
    return _BLOCKS[reading]


def stage_to_logical(y_stage):
    return np.linalg.solve(P_STAGE.T, np.asarray(y_stage, float))


def plant_ss(y_stage, reading):
    """(A 6x6, B 6x3, C 3x6) discrete plant at standstill at stage position y_stage [m]."""
    q0 = stage_to_logical(y_stage)
    C = np.hstack([P_STAGE.T, np.zeros((3, 3))])
    if reading == 'stuck':
        return np.eye(6), np.zeros((6, 3)), C
    blk = block(reading)
    z0 = torch.tensor(np.concatenate([q0, np.zeros(3), np.zeros(3)]), dtype=D).reshape(1, 9, 1)
    J = torch.autograd.functional.jacobian(lambda z: blk.nonlinear_function(z), z0)
    J = J.reshape(6, 9).numpy()
    return J[:, :6], J[:, 6:], C


def ctrl_ss(amps=False):
    A, B, C, Dm = physical_ss('identified')
    if amps:
        g = np.asarray(tp.a_to_n())
        C = C / g[:, None]
        Dm = Dm / g[:, None]
    return A, B, C, Dm


def frf(A, B, C, Dm, f, ts=TS):
    """Frequency response (F, p, m) of a discrete state space at frequencies f [Hz].

    The DC bin is evaluated at 0.01 Hz: plant (rigid body) and controller (integrator) both have
    poles at z = 1, so the response at exactly DC does not exist; S there is ~0 either way.
    """
    f = np.where(np.asarray(f, float) == 0.0, 0.01, np.asarray(f, float))
    z = np.exp(2j * np.pi * f * ts)
    n = A.shape[0]
    out = np.empty((len(z), C.shape[0], B.shape[1]), complex)
    I = np.eye(n)
    for k, zk in enumerate(z):
        out[k] = C @ np.linalg.solve(zk * I - A, B) + Dm
    return out


def loop_frf(y_stage, reading, f):
    """dict G [m/N], K [N/m], S, T, SG at f, all (F, 3, 3)."""
    Ap, Bp, Cp = plant_ss(y_stage, reading)
    G = frf(Ap, Bp, Cp, np.zeros((3, 3)), f)
    K = frf(*ctrl_ss(), f)
    I = np.eye(3)[None]
    S = np.linalg.inv(I + G @ K)
    return dict(G=G, K=K, S=S, T=I - S, SG=S @ G)


def closed_loop(y_stage, reading):
    """Closed-loop matrices, inputs [v (3); d (3) force at plant input], state [x_p; x_c].

    Outputs: e = -(C x_p + v)... returned as (Acl, Bcl, Ccl, Dcl) with outputs [e (3), q (3), u (3)].
    """
    Ap, Bp, Cp = plant_ss(y_stage, reading)
    Ac, Bc, Cc, Dc = ctrl_ss()
    n, m = Ap.shape[0], Ac.shape[0]
    # e = -(Cp xp + v); u = Cc xc + Dc e; xp+ = Ap xp + Bp (u + d); xc+ = Ac xc + Bc e
    Acl = np.block([[Ap - Bp @ Dc @ Cp, Bp @ Cc], [-Bc @ Cp, Ac]])
    Bcl = np.block([[-Bp @ Dc, Bp], [-Bc, np.zeros((m, 3))]])
    Ce = np.hstack([-Cp, np.zeros((3, m))])
    Cq = np.hstack([Cp, np.zeros((3, m))])
    Cu = np.hstack([-Dc @ Cp, Cc])
    Ccl = np.vstack([Ce, Cq, Cu])
    Dcl = np.vstack([np.hstack([-np.eye(3), np.zeros((3, 3))]), np.zeros((3, 6)),
                     np.hstack([-Dc, np.zeros((3, 3))])])
    return Acl, Bcl, Ccl, Dcl


def least_damped(Acl, ts=TS, fmin=1.0):
    """(f [Hz], zeta, |z|) of the least-damped closed-loop mode above fmin."""
    lam = np.linalg.eigvals(Acl)
    s = np.log(lam.astype(complex)) / ts
    fn = np.abs(s) / (2 * np.pi)
    zeta = -s.real / np.maximum(np.abs(s), 1e-30)
    ok = fn > fmin
    i = np.argmin(np.where(ok, zeta, np.inf))
    return float(fn[i]), float(zeta[i]), float(np.abs(lam).max())


def lsim(A, B, C, Dm, u, x0=None):
    """Discrete simulation, u (N, m) -> y (N, p)."""
    x = np.zeros(A.shape[0]) if x0 is None else x0.copy()
    y = np.empty((len(u), C.shape[0]))
    for k in range(len(u)):
        y[k] = C @ x + Dm @ u[k]
        x = A @ x + B @ u[k]
    return y
