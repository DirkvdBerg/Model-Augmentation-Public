"""
diag_residual_force.py
----------------------
Real-data force-balance diagnostic (no training). Answers "why does the
frictionless FP model leave an e-2..e-1 m open-loop error?" by isolating the
generalized force the model cannot account for.

Method (real data only; the FP matrices are the hypothesis under test):
    Real equation of motion of the machine (per axis, logical coords):
        M(Y) qdd + C qd + K q + f_missing(qd, ...) = u_applied
    The FP model omits f_missing (Coulomb/static friction is not in the SS
    model, per docs). So from the MEASURED motion and the MEASURED applied
    force we recover it directly:
        f_missing = u_applied - [ M(Y) qdd + C qd + K q ]
    If f_missing is friction it has an unmistakable fingerprint: a roughly
    constant-magnitude force that flips sign with velocity (a rectangular
    loop in the f_missing-vs-velocity plane), of order the static-friction
    spec (Telica 1.mat: X 136 N, Y 98 N per axis).

Coordinates (physics.py convention):
    q_stage    = q_logical @ P         (stage positions)
    u_logical  = u_stage   @ P.T       (logical forces)
    => f in logical maps to per-motor stage force by  f_stage = f_logical @ inv(P).T
    Stage axes = physical motors [X1, X2, Y].

Derivatives from the measured position use Savitzky-Golay smoothing
differentiation (robust to encoder quantization at 20 kHz).

Run:
    conda run -n GraduationProject python scripts/gantry/real-data-verification/diag_residual_force.py
"""

__project_origin__ = "added"

import os
import sys
import json

import numpy as np
import torch
from scipy.signal import savgol_filter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from telica_loader import load_telica_log
from lpv_lfr_baseline.core.physics import (
    M0, M1, M2, C, K, P, ts as _TS,
)

_SAVE_DIR = os.path.join(_ROOT, 'simulations', 'gantry_subnet',
                         'diagnostics', 'residual_force')
_DATASET_ROOT = os.path.join(_ROOT, 'kamtin-data', 'Data Telica',
                             '06 40 mm XL 80 mm YL')

# Motion-rich iter0 (pure feedback) at operating points spanning the Y grid.
_TRAJ = (
    ('xpos_-60_ypos-40',  'iter0.log', 'x=-60 y=-40'),
    ('xpos_-60_ypos-200', 'iter0.log', 'x=-60 y=-200'),
    ('xpos_-210_ypos120', 'iter0.log', 'x=-210 y=+120'),
)

# Static-friction reference (Telica 1.mat ForceCapabilities, per axis) [N]
_FC_SPEC = {'X': 136.0, 'Y': 98.0}

# HEURISTIC: Savitzky-Golay window 61 samples = 3.05 ms at 20 kHz, polyorder 3.
# Long enough to suppress encoder-quantization noise in the 2nd derivative,
# short enough to preserve the move dynamics (<~330 Hz).
_SG_WIN  = 61
_SG_POLY = 3
_EDGE    = _SG_WIN            # trim edge transients of the SG filter
_AXES    = ('X1', 'X2', 'Y')
_AXIS_FC = ('X', 'X', 'Y')   # which spec each motor is compared against


def _derivatives(q_logical, dt):
    """Smoothed velocity and acceleration in logical coords via Savitzky-Golay.
    q_logical: (T,3) numpy. Returns qd, qdd (T,3)."""
    # THEORY: Savitzky-Golay smoothing differentiation
    #         (Savitzky & Golay, Anal. Chem. 36(8), 1964). delta=dt gives the
    #         derivative directly in physical units (1/s, 1/s^2).
    qd  = savgol_filter(q_logical, _SG_WIN, _SG_POLY, deriv=1, delta=dt, axis=0)
    qdd = savgol_filter(q_logical, _SG_WIN, _SG_POLY, deriv=2, delta=dt, axis=0)
    return qd, qdd


def analyse(op_folder, fname, label):
    path = os.path.join(_DATASET_ROOT, 'train', op_folder, fname)
    u_t, q1_t, fs = load_telica_log(path, dtype=torch.float64)
    u_stage  = u_t[0].numpy()      # (T,3) total applied force [N], stage
    q1_stage = q1_t.numpy()        # (T,3) measured position [m], stage
    dt = 1.0 / fs

    Pn    = P.numpy()
    Pinv  = np.linalg.inv(Pn)
    M0n, M1n, M2n = M0.numpy(), M1.numpy(), M2.numpy()
    Cn, Kn = C.numpy(), K.numpy()

    # stage -> logical positions:  q_logical = q_stage @ inv(P)
    q_log = q1_stage @ Pinv
    qd_log, qdd_log = _derivatives(q_log, dt)
    Y = q_log[:, 2]                                    # payload position [m]

    # M(Y) qdd  (inertial part alone) and full model generalized force
    MY = M0n[None] + M1n[None] * Y[:, None, None] + M2n[None] * (Y[:, None, None] ** 2)
    inertial_log  = np.einsum('tij,tj->ti', MY, qdd_log)          # (T,3)
    gen_model_log = inertial_log + qd_log @ Cn.T + q_log @ Kn.T   # (T,3)

    # applied force in logical:  u_logical = u_stage @ P.T
    u_log = u_stage @ Pn.T

    # force the model is MISSING (real applied minus model-accounted)
    f_missing_log   = u_log - gen_model_log
    f_missing_stage = f_missing_log @ Pinv.T           # per motor [X1,X2,Y] [N]
    u_model_stage   = gen_model_log @ Pinv.T           # model force per motor
    inertial_stage  = inertial_log @ Pinv.T            # model inertial force per motor
    qd_stage        = qd_log @ Pn                       # stage velocity [m/s]

    # trim SG edge transients
    sl = slice(_EDGE, -_EDGE)
    f_missing_stage = f_missing_stage[sl]
    u_model_stage   = u_model_stage[sl]
    inertial_stage  = inertial_stage[sl]
    u_stage_tr      = u_stage[sl]
    qd_stage        = qd_stage[sl]
    t = np.arange(f_missing_stage.shape[0]) / fs

    # per-axis stats + coordinate sanity (model force vs applied force corr)
    stats = {}
    print(f'\n{"=" * 66}\n{op_folder}  ({label})   T={len(t)} samples\n{"=" * 66}')
    print(f'  {"axis":>4} {"|f_miss| rms":>12} {"|f_miss| p95":>12} '
          f'{"|u_appl| rms":>12} {"miss/appl":>9} {"corr(umod,u)":>12} {"Fc spec":>8}')
    for i, ax in enumerate(_AXES):
        fm, ua, um = f_missing_stage[:, i], u_stage_tr[:, i], u_model_stage[:, i]
        rms   = float(np.sqrt(np.mean(fm ** 2)))
        p95   = float(np.percentile(np.abs(fm), 95))
        u_rms = float(np.sqrt(np.mean(ua ** 2)))
        ratio = rms / u_rms if u_rms > 0 else float('nan')
        corr  = float(np.corrcoef(um, ua)[0, 1])
        fc    = _FC_SPEC[_AXIS_FC[i]]
        print(f'  {ax:>4} {rms:>12.2f} {p95:>12.2f} {u_rms:>12.2f} '
              f'{ratio:>9.2f} {corr:>12.4f} {fc:>8.0f}')
        stats[ax] = dict(f_missing_rms_N=rms, f_missing_p95_N=p95,
                         u_applied_rms_N=u_rms, ratio=ratio,
                         corr_umodel_uapplied=corr, fc_spec_N=fc)

    # ---- decomposition: is the missing force inertial-scale / viscous / Coulomb? ----
    # LS fit over MOVING samples:  f_missing ~= a*(inertial) + b*qd + c*sign(qd)
    #   a  -> effective inertia scale error: applied ~= (1+a)*(M qdd)+..., so s=1+a
    #   b  -> extra viscous damping [N/(m/s)]
    #   c  -> Coulomb (dry-friction) amplitude [N]
    # HEURISTIC: "moving" = |stage velocity| > 20 mm/s (excludes standstill/reversal)
    print(f'  decomposition (moving samples):  f_missing ~ a*Mqdd + b*qd + c*sign(qd)')
    print(f'  {"axis":>4} {"a (inert.err)":>13} {"s=1+a":>7} '
          f'{"b [N/(m/s)]":>12} {"c (Coulomb) [N]":>15} {"R2":>6}')
    for i, ax in enumerate(_AXES):
        v = qd_stage[:, i]
        mv = np.abs(v) > 20e-3
        if mv.sum() < 50:
            print(f'  {ax:>4}   (insufficient motion)')
            continue
        X = np.stack([inertial_stage[mv, i], v[mv], np.sign(v[mv])], axis=1)
        y = f_missing_stage[mv, i]
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        a_c, b_c, c_c = coef
        resid = y - X @ coef
        r2 = 1.0 - np.var(resid) / np.var(y) if np.var(y) > 0 else float('nan')
        print(f'  {ax:>4} {a_c:>13.3f} {1+a_c:>7.3f} {b_c:>12.1f} '
              f'{c_c:>15.1f} {r2:>6.3f}')
        stats[ax].update(inertial_err_a=float(a_c), inertial_scale_s=float(1 + a_c),
                         viscous_b_Npm_s=float(b_c), coulomb_c_N=float(c_c),
                         decomp_r2=float(r2))

    # ---- figure: time series (left) + friction loop (right), per axis ----
    fig, axs = plt.subplots(3, 2, figsize=(14, 10))
    for i, ax in enumerate(_AXES):
        fc = _FC_SPEC[_AXIS_FC[i]]
        axs[i, 0].plot(t, u_stage_tr[:, i], color='0.7', lw=0.7, label='applied force')
        axs[i, 0].plot(t, u_model_stage[:, i], 'C0', lw=0.7, alpha=0.8,
                       label='FP model force (Mqdd+Cqd+Kq)')
        axs[i, 0].plot(t, f_missing_stage[:, i], 'C3', lw=0.7,
                       label='missing force (applied - model)')
        axs[i, 0].set_ylabel(f'{ax}  force [N]')
        if i == 0:
            axs[i, 0].legend(fontsize=7, loc='upper right')
        axs[i, 0].grid(alpha=0.3)

        axs[i, 1].plot(qd_stage[:, i] * 1e3, f_missing_stage[:, i], '.',
                       ms=1.5, alpha=0.3)
        axs[i, 1].axhline(0, color='k', lw=0.5)
        axs[i, 1].axhline(fc, color='C2', ls='--', lw=1, label=f'+/- Fc spec = {fc:.0f} N')
        axs[i, 1].axhline(-fc, color='C2', ls='--', lw=1)
        axs[i, 1].axvline(0, color='k', lw=0.5)
        axs[i, 1].set_ylabel(f'{ax}  missing force [N]')
        if i == 0:
            axs[i, 1].legend(fontsize=7, loc='upper right')
        axs[i, 1].grid(alpha=0.3)
    axs[-1, 0].set_xlabel('time [s]')
    axs[-1, 1].set_xlabel('stage velocity [mm/s]')
    fig.suptitle(f'Residual force diagnostic  |  {op_folder} ({label})\n'
                 f'Left: does the FP model force match applied?  '
                 f'Right: is the missing force a velocity-sign friction loop?')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fp = os.path.join(_SAVE_DIR, f'residual_force_{op_folder}.png')
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    print(f'  figure -> {os.path.relpath(fp, _ROOT)}')
    return stats


def main():
    os.makedirs(_SAVE_DIR, exist_ok=True)
    print('=' * 66)
    print('RESIDUAL FORCE DIAGNOSTIC  (real data, no training)')
    print('  f_missing = u_applied - [ M(Y) qdd + C qd + K q ]')
    print('  friction fingerprint: constant |f| that flips sign with velocity')
    print('=' * 66)
    summary = {}
    for op, fname, label in _TRAJ:
        summary[op] = analyse(op, fname, label)
    with open(os.path.join(_SAVE_DIR, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)
    print('\nsummary -> ' + os.path.relpath(
        os.path.join(_SAVE_DIR, 'summary.json'), _ROOT))


if __name__ == '__main__':
    main()
