"""Shared plant models and record loading for the msd-offset figures.

Two models, both open loop, both driven by the recorded stage force u_total:

  TRUTH    8-state, gantrySystemExtended.m: x = [X, Th, Y, da, dX, dTh, dY, vda]
  BASELINE 6-state, gantrySystem.m (full mh): x = [X, Th, Y, dX, dTh, dY]

Both freeze M at the current state (no Coriolis/centrifugal), matching the
generator. Parameters ma_frac=0.10, fa=150 Hz, L0=0.10 are confirmed from the data
by a fit, not assumed: see docs/msd-offset-mechanism-2026-07-29.md section 2.
"""
__project_origin__ = "added"

import os
import numpy as np
from scipy.io import loadmat

# VENDOR-PATCH (BB-002): five levels up from vendor/msd_offset; data read from the repo.
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
TRAJ = os.path.join(REPO, 'data', 'gantry', 'matlab', 'trajectory', 'augmentation')
OUT_DIAG = os.path.join(REPO, 'simulations', 'gantry_subnet', 'diagnostics')

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # VENDOR-PATCH (BB-002): vendored model_augmentation
from model_augmentation.systems.gantry_ss import (
    m1, m2, mb, mh, Jb, Jh, cg1, cg2, cy, cb1, cb2, kb1, kb2, Lb, d, P)

m1, m2, mb, mh = float(m1), float(m2), float(mb), float(mh)
Jb, Jh, cg1, cg2, cy = float(Jb), float(Jh), float(cg1), float(cg2), float(cy)
cb1, cb2, kb1, kb2, Lb, d = float(cb1), float(cb2), float(kb1), float(kb2), float(Lb), float(d)
P_np = P.numpy().astype(np.float64)

# --- absorber (fitted from the data, section 2 of the doc) -------------------
MA_FRAC = 0.10
MA = MA_FRAC * mh                  # 1.01 kg
MHR = mh - MA                      # 9.09 kg
L0 = 0.10                          # m
FA = 150.0                         # Hz  THEORY: MSD natural freq (gtd_config)
ZETA_A = 0.05
KA = MA * (2 * np.pi * FA) ** 2     # THEORY: k = m*(2*pi*f)^2
CA = 2 * ZETA_A * np.sqrt(KA * MA)  # THEORY: c = 2*zeta*sqrt(k*m)

TAU_Y = mh / cy                    # 1.010 s   free-integrator settling on Y
TAU_X = (m1 + m2 + mb + mh) / (cg1 + cg2)   # 1.546 s
OFFSET_GAIN = MA / cy              # 0.101 s   dY(inf) = OFFSET_GAIN * vda(t0)

_C4 = np.array([[cg1 + cg2, (cg1 - cg2) * Lb / 2, 0., 0.],
                [(cg1 - cg2) * Lb / 2, cb1 + cb2 + (cg1 + cg2) * Lb ** 2 / 4, 0., 0.],
                [0., 0., cy, 0.],
                [0., 0., 0., CA]])
_K4 = np.array([[0., 0., 0., 0.], [0., kb1 + kb2, 0., 0.],
                [0., 0., 0., 0.], [0., 0., 0., KA]])
_E43 = np.vstack([np.eye(3), np.zeros((1, 3))])
_C3, _K3 = _C4[:3, :3].copy(), _K4[:3, :3].copy()

_A11 = m1 + m2 + mb + mh
_B12 = (m1 - m2) * Lb / 2
_G22 = Jb + Jh + (m1 + m2) * Lb ** 2 / 4 + mh * d ** 2
_MHD = mh * d


def M8(Y, da, l0=L0, freeze=False):
    """Truth 4x4 mass matrix. freeze=True evaluates the da-dependence at da=0."""
    e = 0.0 if freeze else da
    off = _B12 - (MHR + MA) * Y - MA * l0 - MA * e
    return np.array([
        [m1 + m2 + mb + MHR + MA, off, 0., 0.],
        [off, Jb + Jh + (m1 + m2) * Lb ** 2 / 4 + (MHR + MA) * d ** 2
              + MHR * Y ** 2 + MA * (Y + l0 + e) ** 2, -(MHR + MA) * d, -MA * d],
        [0., -(MHR + MA) * d, MHR + MA, MA],
        [0., -MA * d, MA, MA]])


def deriv8(x, u, l0=L0, freeze=False):
    q, qd = x[:4], x[4:]
    qdd = np.linalg.solve(M8(x[2], x[3], l0, freeze), _E43 @ u - _K4 @ q - _C4 @ qd)
    return np.concatenate([qd, qdd])


def deriv6(x, u):
    """Baseline 6-state; hand-coded 3x3 solve via cofactors (M is symmetric)."""
    Y = x[2]
    b = _B12 - mh * Y
    a11, a12, a22, a23, a33 = _A11, b, _G22 + mh * Y * Y, -_MHD, mh
    r = u - _C3 @ x[3:] - _K3 @ x[:3]
    c11 = a22 * a33 - a23 * a23
    c12 = -(a12 * a33)
    c13 = a12 * a23
    c22 = a11 * a33
    c23 = -(a11 * a23)
    c33 = a11 * a22 - a12 * a12
    det = a11 * c11 + a12 * c12
    qdd = np.array([(c11 * r[0] + c12 * r[1] + c13 * r[2]) / det,
                    (c12 * r[0] + c22 * r[1] + c23 * r[2]) / det,
                    (c13 * r[0] + c23 * r[1] + c33 * r[2]) / det])
    return np.concatenate([x[3:], qdd])


def rollout(fn, x0, u_log, ts, n_out=3):
    """RK4, u held over each step. Returns the first n_out states per sample."""
    N = len(u_log)
    out = np.empty((N, n_out))
    x = np.asarray(x0, float).copy()
    for k in range(N):
        out[k] = x[:n_out]
        uk = u_log[k]
        k1 = fn(x, uk); k2 = fn(x + .5 * ts * k1, uk)
        k3 = fn(x + .5 * ts * k2, uk); k4 = fn(x + ts * k3, uk)
        x = x + (ts / 6.) * (k1 + 2 * k2 + 2 * k3 + k4)
    return out


def to_stage(q_logical):
    return (P_np.T @ np.asarray(q_logical).T).T


def load_record(name, fs_new=4000):
    """Load one record at fs_new. u block-mean (D-087), y/states point-sampled."""
    dm = loadmat(os.path.join(TRAJ, name + '.mat'), squeeze_me=True)
    ts0 = float(dm['dt'])
    D = int(round((1.0 / ts0) / fs_new))
    u0 = np.asarray(dm['u_total'], float)
    n = len(u0) // D
    rec = dict(
        name=name, ts=ts0 * D, D=D, fs=fs_new,
        u=u0[:n * D].reshape(n, D, 3).mean(axis=1),          # block mean (D-087)
        y=np.asarray(dm['y'], float)[::D][:n],
        x_logical=np.asarray(dm['x_logical'], float)[::D][:n],
        delta_a=np.asarray(dm['delta_a'], float)[::D][:n],
        vdelta_a=np.asarray(dm['vdelta_a'], float)[::D][:n],
        f_ms=np.asarray(dm['f_sim'], float)[::D][:n],
        r=np.asarray(dm['r_sim'], float)[::D][:n],
    )
    rec['u_log'] = (P_np @ rec['u'].T).T                      # stage -> logical force
    rec['Y_op'] = float(rec['y'][0, 2])                       # y(0) = [0, 0, Y_op]
    rec['t'] = np.arange(n) * rec['ts']
    return rec


def exact_ic(rec):
    """The Simulink integrator initial condition, [0;0;Y;0;0;0;0;0]."""
    Yop = rec['Y_op']
    return (np.array([0., 0., Yop, 0., 0., 0.]),
            np.array([0., 0., Yop, 0., 0., 0., 0., 0.]))


def true_ic(rec, k):
    """True state at sample k: 6-state from x_logical, 8-state adding the absorber."""
    xl, da, vda = rec['x_logical'][k], rec['delta_a'][k], rec['vdelta_a'][k]
    return (xl.copy(),
            np.array([xl[0], xl[1], xl[2], da, xl[3], xl[4], xl[5], vda]))
