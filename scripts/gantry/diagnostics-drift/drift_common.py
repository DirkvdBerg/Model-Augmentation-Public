"""
drift_common.py -- shared building blocks for the X+Theta+Y augmentation drift
diagnostics (session 2026-07-09).

Purpose: give every d1..d5 script one verified data loader, one full-truth
(baseline + MSD absorber) 8-state simulator, one baseline-only 6-state open-loop
simulator, and the P-transform / time-constant helpers -- so the diagnostics are
assembly, not re-derivation.

Provenance (do NOT re-derive):
  * Full-truth 8-state EOM (_M / _deriv / simulate_truth): direct port of the
    verified RK4 oracle in scripts/gantry/augmentation-error/diag_oracle_vs_data.py
    (itself a port of Matlab-scripts/Augmentation/gantrySystemExtended.m).
    State x = [X, Th, Y, da, dX, dTh, dY, vda] in LOGICAL coords; force u_log
    = [F_X, F_Th, F_Y] logical.  M(Y, da) is nonlinear.
  * Baseline-only 6-state (Gantry_State_Block, identity normalisation -> physical
    units): same call pattern as diag_openloop_x0.py.
  * Data conventions (gtd_* / generate_gantry_lti_augmented.m):
        u_total, y are STAGE coords ; x_logical is 6D LOGICAL.
        u_logical = P @ u_stage ;  q_stage = P^T @ q_logical.
  * Resampling FS_ORIG->FS_NEW: block-mean u (D-087 impulse-equivalent ZOH),
    point-sampled y / x_logical / delta_a (D-099, band-limited << Nyquist).

MSD absorber params: ma_frac=0.10, fa=150 Hz, L0=0.10, zeta_a=0.05
(project_gantry_msd_params; the mat files do NOT store them).
"""
__project_origin__ = "added"

import os
from dataclasses import dataclass

import numpy as np
import torch
from scipy.io import loadmat

from model_augmentation.systems.gantry_ss import (
    Cd, Dd, P,
    m1, m2, mb, mh, Jb, Jh, cg1, cg2, cy, cb1, cb2, kb1, kb2, Lb, d,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DATA_DIR = os.path.join(REPO, 'data', 'gantry', 'matlab', 'trajectory', 'augmentation')
OUT_DIR  = os.path.join(REPO, 'simulations', 'gantry_subnet', 'diagnostics')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Pipeline-matched discretisation ──────────────────────────────────────────
FS_ORIG        = 20000        # native Simulink rate [Hz]
DEFAULT_FS_NEW = 4000         # training pipeline rate (D=5)
UP_SAMPLE      = 2            # RK4 substeps per output step (matches the pipeline)
DTYPE          = np.float64

# ── Fixed physical scalars (float) ───────────────────────────────────────────
m1f, m2f, mbf, mhf = float(m1), float(m2), float(mb), float(mh)
Jbf, Jhf           = float(Jb), float(Jh)
cg1f, cg2f, cyf    = float(cg1), float(cg2), float(cy)
cb1f, cb2f         = float(cb1), float(cb2)
kb1f, kb2f         = float(kb1), float(kb2)
Lbf, df            = float(Lb), float(d)
Pnp   = P.numpy().astype(DTYPE)
Cdnp  = Cd.numpy().astype(DTYPE)
Ddnp  = Dd.numpy().astype(DTYPE)

# ── Time constants of the K=0 (double-integrator) axes ───────────────────────
# THEORY: for a mass-damper axis with no spring, an initial velocity error dv
# leaves a settled position offset x_inf = tau*dv (integral of v0*exp(-t/tau)).
# Theta has a spring (kb1+kb2) so its velocity error leaves no offset.
tau_X = float((m1f + m2f + mbf + mhf) / (cg1f + cg2f))
tau_Y = float(mhf / cyf)

# ── MSD absorber (hidden truth dynamics) ─────────────────────────────────────
MA_FRAC  = 0.10
ma       = MA_FRAC * mhf              # absorber mass
mh_rigid = mhf - ma                   # rigid head mass passed to the EOM
L0       = 0.10                       # equilibrium offset of ma in +Y
fa       = 150.0                      # absorber natural frequency [Hz]
ka       = ma * (2 * np.pi * fa) ** 2
zeta_a   = 0.05
ca       = 2 * zeta_a * np.sqrt(ka * ma)

# Constant sub-blocks of the 4-DOF (X, Th, Y, da) truth EOM (Y/da-independent).
_C4 = np.array([
    [cg1f + cg2f,          (cg1f - cg2f) * Lbf / 2,                          0.0, 0.0],
    [(cg1f - cg2f) * Lbf / 2, cb1f + cb2f + (cg1f + cg2f) * Lbf ** 2 / 4,    0.0, 0.0],
    [0.0, 0.0, cyf, 0.0],
    [0.0, 0.0, 0.0, ca],
], dtype=DTYPE)
_K4 = np.array([
    [0.0, 0.0,           0.0, 0.0],
    [0.0, kb1f + kb2f,   0.0, 0.0],
    [0.0, 0.0,           0.0, 0.0],
    [0.0, 0.0,           0.0, ka],
], dtype=DTYPE)
_E43 = np.vstack([np.eye(3), np.zeros((1, 3))])   # force enters first 3 gen. coords


def _M(Y, da):
    """State-dependent 4x4 mass matrix (gantrySystemExtended.m)."""
    off = (m1f - m2f) * Lbf / 2 - (mh_rigid + ma) * Y - ma * L0 - ma * da
    return np.array([
        [m1f + m2f + mbf + mh_rigid + ma, off, 0.0, 0.0],
        [off, Jbf + Jhf + (m1f + m2f) * Lbf ** 2 / 4 + (mh_rigid + ma) * df ** 2
              + mh_rigid * Y ** 2 + ma * (Y + L0 + da) ** 2, -(mh_rigid + ma) * df, -ma * df],
        [0.0, -(mh_rigid + ma) * df, mh_rigid + ma, ma],
        [0.0, -ma * df,              ma,            ma],
    ], dtype=DTYPE)


def _deriv(x, u_log):
    """dx/dt for the 8-state truth (nonlinear via M(Y, da))."""
    Y, da = x[2], x[3]
    M = _M(Y, da)
    q, qd = x[:4], x[4:]
    qdd = np.linalg.solve(M, _E43 @ u_log - _K4 @ q - _C4 @ qd)
    dx = np.empty(8)
    dx[:4] = qd
    dx[4:] = qdd
    return dx


def simulate_truth(x0_8, u_log, ts, up_sample=UP_SAMPLE, return_state=False):
    """RK4 open-loop rollout of the 8-state truth. u_log held (ZOH) per output step.

    Returns q_log (N,4) = [X, Th, Y, da] pre-update per step, and, if
    return_state, the full 8-state history (N,8).
    """
    N = len(u_log)
    q_log  = np.empty((N, 4))
    x_hist = np.empty((N, 8)) if return_state else None
    x = x0_8.astype(DTYPE).copy()
    h = ts / up_sample
    for k in range(N):
        q_log[k] = x[:4]
        if return_state:
            x_hist[k] = x
        uk = u_log[k]
        for _ in range(up_sample):
            k1 = _deriv(x, uk)
            k2 = _deriv(x + 0.5 * h * k1, uk)
            k3 = _deriv(x + 0.5 * h * k2, uk)
            k4 = _deriv(x + h * k3, uk)
            x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return (q_log, x_hist) if return_state else q_log


# ── Baseline-only 6-state open-loop (Gantry_State_Block, physical units) ──────
def build_baseline_block(ts, up_sample=UP_SAMPLE):
    """Identity-normalised baseline block -> states/inputs/outputs in physical units."""
    from model_augmentation.fit_systems.blocks import Gantry_State_Block
    blk = Gantry_State_Block(
        Y_op=None,                       # LPV self-scheduling (Y = x[2])
        std_x=np.ones((6, 1)), std_u=np.ones((3, 1)),
        x_mean=np.zeros((6, 1)), u_mean=np.zeros((3, 1)),
        Ts=ts, up_sample=up_sample,
    ).to(torch.float64)
    blk.eval()
    return blk


def simulate_baseline(x0_6, u_stage, ts, up_sample=UP_SAMPLE, return_state=False):
    """Baseline-only (no absorber, no ANN) open-loop rollout from x0_6.

    x0_6, states are LOGICAL; u_stage is STAGE force (the block applies P
    internally). Returns y_stage (N,3) and optionally state history (N,6).
    y[k] = Cd@x[k] + Dd@u[k], then x[k+1] = f(x[k], u[k]).
    """
    blk = build_baseline_block(ts, up_sample)
    N = len(u_stage)
    y_stage = np.empty((N, 3))
    x_hist  = np.empty((N, 6)) if return_state else None
    x = torch.tensor(x0_6, dtype=torch.float64)
    with torch.no_grad():
        for k in range(N):
            u_t = u_stage[k]
            xn = x.numpy()
            y_stage[k] = Cdnp @ xn + Ddnp @ u_t
            if return_state:
                x_hist[k] = xn
            z = torch.cat([x.view(1, 6, 1),
                           torch.tensor(u_t, dtype=torch.float64).view(1, 3, 1)], dim=1)
            x = blk(z).view(6)
    return (y_stage, x_hist) if return_state else y_stage


# ── P-transform helpers ──────────────────────────────────────────────────────
def stage_force_to_logical(u_stage):
    """u_logical = P @ u_stage  (per sample, (N,3) -> (N,3))."""
    return (Pnp @ u_stage.T).T


def logical_pos_to_stage(q_log):
    """q_stage = P^T @ q_logical."""
    return (Pnp.T @ q_log.T).T


def stage_pos_to_logical(q_stage):
    """q_logical = P^{-T} @ q_stage (inverse of q_stage = P^T q_logical)."""
    return (np.linalg.inv(Pnp.T) @ q_stage.T).T


# ── Data loading ─────────────────────────────────────────────────────────────
@dataclass
class Record:
    name: str
    u_stage: np.ndarray     # (N,3) block-mean stage force
    y_stage: np.ndarray     # (N,3) stage position (noiseless)
    x_logical: np.ndarray   # (N,6) logical states [q, qd]
    delta_a: np.ndarray     # (N,)  absorber displacement
    vdelta_a: np.ndarray    # (N,)  absorber velocity
    ts: float               # decimated sample period [s]
    D: int                  # decimation factor


def load_record(filename, fs_new=DEFAULT_FS_NEW):
    """Load one augmentation mat record at fs_new with pipeline-matched resampling.

    Block-mean u (D-087), point-sampled y / x_logical / delta_a / vdelta_a (D-099).
    """
    path = filename if os.path.isabs(filename) else os.path.join(DATA_DIR, filename)
    m = loadmat(path, squeeze_me=True)
    ts0 = float(m['dt'])
    D   = round((1.0 / ts0) / fs_new)
    ts  = ts0 * D

    u0   = m['u_total'].astype(DTYPE)
    y0   = m['y'].astype(DTYPE)
    xl0  = m['x_logical'].astype(DTYPE)
    da0  = m['delta_a'].astype(DTYPE)
    vda0 = m['vdelta_a'].astype(DTYPE)

    if D > 1:
        n = u0.shape[0] // D
        u_stage = u0[:n * D].reshape(n, D, 3).mean(axis=1)   # block mean (D-087)
        y_stage, x_logical = y0[::D][:n], xl0[::D][:n]        # point sample (D-099)
        delta_a, vdelta_a  = da0[::D][:n], vda0[::D][:n]
    else:
        u_stage, y_stage, x_logical, delta_a, vdelta_a = u0, y0, xl0, da0, vda0

    return Record(os.path.basename(path), u_stage, y_stage, x_logical,
                  delta_a, vdelta_a, ts, D)


def true_x0_8(rec: Record, k=0):
    """Assemble the true 8-state initial condition at sample k from the record.

    x = [X, Th, Y, da, dX, dTh, dY, vda]  (logical).
    """
    xl = rec.x_logical
    return np.array([xl[k, 0], xl[k, 1], xl[k, 2], rec.delta_a[k],
                     xl[k, 3], xl[k, 4], xl[k, 5], rec.vdelta_a[k]], dtype=DTYPE)
