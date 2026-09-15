"""FP + true hidden-MSD oracle model (best-case augmentation reference).

RK4 port of `Matlab-scripts/Augmentation/gantrySystemExtended.m`: the 8-state
extended plant used to generate the augmentation data. State (logical coords)
  x = [X, Theta, Y, delta_a, dX, dTheta, dY, vdelta_a]
force u = [F_X, F_Theta, F_Y] (logical). Mass matrix M(Y, delta_a) is nonlinear.

Verified against the Simulink data before use (D-097,
`scripts/gantry/augmentation-error/diag_oracle_vs_data.py`): reproduces y to
discretization error and delta_a to <1% at pipeline conditions.

Fairness (D-097 / lessons.md): the pipeline oracle runs at the SAME rate and
`up_sample` as the baseline/augmented model (never native/finer); callers pass
`hp['up_sample']` and `cfg.ts_new`. Seed from an interior sample (K0): the
sample-0 stored qdot is a one-sided gradient() artifact (D-087).
"""
__project_origin__ = "added"

import numpy as np

from model_augmentation.systems.gantry_ss import (
    m1 as _m1, m2 as _m2, mb as _mb, mh as _mh, Jb as _Jb, Jh as _Jh,
    cg1 as _cg1, cg2 as _cg2, cy as _cy, cb1 as _cb1, cb2 as _cb2,
    kb1 as _kb1, kb2 as _kb2, Lb as _Lb, d as _d, P as _P)

# Fixed FP scalars (single source: gantry_ss / kamtin-fp-model)
m1, m2, mb, mh = float(_m1), float(_m2), float(_mb), float(_mh)
Jb, Jh = float(_Jb), float(_Jh)
cg1, cg2, cy = float(_cg1), float(_cg2), float(_cy)
cb1, cb2, kb1, kb2 = float(_cb1), float(_cb2), float(_kb1), float(_kb2)
Lb, d = float(_Lb), float(_d)
P_np = _P.numpy().astype(np.float64)

# Hidden-MSD absorber params of the augmentation training data (ma_frac=0.10;
# project_gantry_msd_params -- the mat files do not store these).
MA_FRAC  = 0.10
MA       = MA_FRAC * mh          # 1.01 kg  hidden absorber mass
MH_RIGID = mh - MA               # 9.09 kg  rigid payload (the 'mh' EOM argument)
L0       = 0.10                  # equilibrium offset of ma in +Y [m]
FA       = 150.0                 # THEORY: MSD natural freq [Hz] (gtd_config)
KA       = MA * (2 * np.pi * FA) ** 2          # THEORY: k = m*(2*pi*f)^2
ZETA_A   = 0.05
CA       = 2 * ZETA_A * np.sqrt(KA * MA)       # THEORY: c = 2*zeta*sqrt(k*m)

# Constant (Y/da-independent) damping/stiffness blocks and the force selector.
_C4 = np.array([
    [cg1 + cg2,             (cg1 - cg2) * Lb / 2,                       0.0, 0.0],
    [(cg1 - cg2) * Lb / 2,  cb1 + cb2 + (cg1 + cg2) * Lb ** 2 / 4,      0.0, 0.0],
    [0.0, 0.0, cy,  0.0],
    [0.0, 0.0, 0.0, CA],
], dtype=np.float64)
_K4 = np.array([
    [0.0, 0.0,       0.0, 0.0],
    [0.0, kb1 + kb2, 0.0, 0.0],
    [0.0, 0.0,       0.0, 0.0],
    [0.0, 0.0,       0.0, KA],
], dtype=np.float64)
_E43 = np.vstack([np.eye(3), np.zeros((1, 3))])   # force enters first 3 gen. coords


def _M(Y, da):
    """State-dependent 4x4 mass matrix (gantrySystemExtended.m)."""
    off = (m1 - m2) * Lb / 2 - (MH_RIGID + MA) * Y - MA * L0 - MA * da
    return np.array([
        [m1 + m2 + mb + MH_RIGID + MA, off, 0.0, 0.0],
        [off, Jb + Jh + (m1 + m2) * Lb ** 2 / 4 + (MH_RIGID + MA) * d ** 2
              + MH_RIGID * Y ** 2 + MA * (Y + L0 + da) ** 2, -(MH_RIGID + MA) * d, -MA * d],
        [0.0, -(MH_RIGID + MA) * d, MH_RIGID + MA, MA],
        [0.0, -MA * d,               MA,            MA],
    ], dtype=np.float64)


def _deriv(x, u_log):
    """dxdt = A(x) x + B(x) u for the 8-state extended plant."""
    q, qd = x[:4], x[4:]
    qdd = np.linalg.solve(_M(x[2], x[3]), _E43 @ u_log - _K4 @ q - _C4 @ qd)
    dx = np.empty(8)
    dx[:4] = qd
    dx[4:] = qdd
    return dx


def simulate(x0_8, u_log, ts, up_sample):
    """RK4 open-loop rollout; u held (ZOH) over each output step. Returns (N,4) [X,Th,Y,da]."""
    N = len(u_log)
    out = np.empty((N, 4))
    x = np.asarray(x0_8, dtype=np.float64).copy()
    h = ts / up_sample
    for k in range(N):
        out[k] = x[:4]
        uk = u_log[k]
        for _ in range(up_sample):
            k1 = _deriv(x, uk)
            k2 = _deriv(x + 0.5 * h * k1, uk)
            k3 = _deriv(x + 0.5 * h * k2, uk)
            k4 = _deriv(x + h * k3, uk)
            x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return out


def oracle_open_loop(u_stage, x_logical, x_aug, cfg, up_sample, start_ix):
    """Best-case FP+MSD open-loop sim, seeded from the true interior state at start_ix.

    Parameters
    ----------
    u_stage    : (N, 3) resampled plant force in STAGE coords (block-mean, D-087).
    x_logical  : (N, 6) true logical state [q, qd] (point-sampled at cfg rate).
    x_aug      : (N, 2) true absorber state [delta_a, vdelta_a].
    up_sample  : RK4 substeps -- pass hp['up_sample'] (fairness: match the pipeline).
    start_ix   : interior seed sample K0 (avoids the sample-0 gradient() artifact).

    Returns
    -------
    y_hat  : (N, 3) STAGE output, NaN before start_ix (aligns with y_hat_enc).
    da_hat : (N,)   simulated delta_a, NaN before start_ix.
    """
    ts = cfg.ts_new
    dtype = cfg.dtype_np
    N = len(u_stage)

    # State x0 = [X,Th,Y, da, dX,dTh,dY, vda] from the true state at start_ix.
    xl0, a0 = x_logical[start_ix], x_aug[start_ix]
    x0 = np.array([xl0[0], xl0[1], xl0[2], a0[0], xl0[3], xl0[4], xl0[5], a0[1]], dtype=np.float64)

    # Force to logical coords: u_logical = P @ u_stage.
    u_log = (P_np @ np.asarray(u_stage[start_ix:], dtype=np.float64).T).T

    out = simulate(x0, u_log, ts, up_sample)          # (N-start_ix, 4)
    q_stage = (P_np.T @ out[:, :3].T).T               # logical -> stage

    y_hat  = np.full((N, 3), np.nan, dtype=dtype)
    da_hat = np.full(N, np.nan, dtype=dtype)
    y_hat[start_ix:]  = q_stage
    da_hat[start_ix:] = out[:, 3]
    return y_hat, da_hat
