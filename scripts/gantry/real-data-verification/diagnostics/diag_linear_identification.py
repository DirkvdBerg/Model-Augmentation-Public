"""
diag_linear_identification.py
-----------------------------
Go/no-go on "is this real data useful for fitting the physical baseline?"

The baseline equation of motion is linear (and homogeneous) in the physical
parameters:
    M(Y) qdd + C qd + K q = u        (logical coords)
so it can be fit by a single linear least-squares solve on the inverse-dynamics
regressor, with NO gradient training. This is the globally-optimal linear-in-
parameters fit -> an upper bound on how well ANY fitter (incl. BPTT) can match
the physical baseline to this data.

THEORY: inverse-dynamics (linear-in-parameters) identification of rigid-body
dynamics -- Atkeson, An & Hollerbach, Int. J. Robotics Res. 5(3), 1986;
Siciliano et al., Robotics: Modelling, Planning and Control, sec. 7.3.

Regressor construction is exact and reuses the model's own matrix builder:
because f(theta) = sum_j theta_j * f(e_j), column j is obtained by evaluating
the model force with a UNIT parameter vector e_j (via _build_matrices).

Coordinates (physics.py): q_stage = q_logical @ P ; u_logical = u_stage @ P.T.
Fit is done per physical motor (stage) so Coulomb friction is per-motor.

Run:
    conda run -n GraduationProject python scripts/gantry/real-data-verification/diag_linear_identification.py
    (--fast : 4 operating points ; default : all 11 training ops, iter0)
"""

__project_origin__ = "added"

import os
import sys
import json
import argparse

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
from lpv_lfr_baseline.core.physics import P as _P, Lb as _Lb, d as _d
from lpv_lfr_baseline.blocks.lfr_param_block import _build_matrices

_SAVE_DIR = os.path.join(_ROOT, 'simulations', 'gantry_subnet',
                         'diagnostics', 'linear_id')
_DATASET_ROOT = os.path.join(_ROOT, 'kamtin-data', 'Data Telica',
                             '06 40 mm XL 80 mm YL')

# 10 identifiable-ish physical params in _build_matrices order, + nominal values
_PNAMES = ('kb_sum', 'cg1', 'cg2', 'cy', 'cb_sum', 'mh', 'm1', 'm2', 'mb', 'J_sum')
_NOMINAL = np.array([3975.0, 14.5, 20.3, 10.0, 18.0, 10.1, 10.2, 10.7, 22.8, 1.05])

_TRAIN_OPS = (
    'xpos_-60_ypos-40',   'xpos_-60_ypos120',  'xpos_-60_ypos-120',
    'xpos_-60_ypos-200',  'xpos_-135_ypos40',  'xpos_-135_ypos-40',
    'xpos_-135_ypos-200', 'xpos_-210_ypos40',  'xpos_-210_ypos120',
    'xpos_-210_ypos-120', 'xpos_-210_ypos-200',
)
_COULOMB = ('X1', 'X2', 'Y')
_FC_SPEC = (136.0, 136.0, 98.0)     # static-friction spec per motor [N]

# HEURISTIC: Savitzky-Golay 61 samples (3 ms) polyorder 3 -- same as the
# residual diagnostic; robust 2nd derivative at 20 kHz.
_SG_WIN, _SG_POLY, _EDGE = 61, 3, 61


def _derivatives(q_log, dt):
    # THEORY: Savitzky-Golay smoothing differentiation (Savitzky & Golay 1964)
    qd  = savgol_filter(q_log, _SG_WIN, _SG_POLY, deriv=1, delta=dt, axis=0)
    qdd = savgol_filter(q_log, _SG_WIN, _SG_POLY, deriv=2, delta=dt, axis=0)
    return qd, qdd


def _param_matrices(j):
    """M0,M1,M2,K,C (numpy) for the j-th UNIT parameter vector e_j."""
    e = torch.zeros(10, dtype=torch.float64)
    e[j] = 1.0
    M0, M1, M2, K, C = _build_matrices(e, _Lb, _d)
    return (M0.numpy(), M1.numpy(), M2.numpy(), K.numpy(), C.numpy())


def build_regressor(op_folder, fname='iter0.log'):
    """Return (W_base (3T,10), W_coul (3T,3), b (3T,), meta) for one trajectory,
    all in stage (per-motor) force units [N]."""
    path = os.path.join(_DATASET_ROOT, 'train', op_folder, fname)
    u_t, q1_t, fs = load_telica_log(path, dtype=torch.float64)
    u_stage  = u_t[0].numpy()
    q1_stage = q1_t.numpy()
    dt = 1.0 / fs

    Pn   = _P.numpy()
    Pinv = np.linalg.inv(Pn)
    q_log = q1_stage @ Pinv
    qd_log, qdd_log = _derivatives(q_log, dt)
    Y = q_log[:, 2]

    # baseline regressor columns: f_j_stage(t) for each unit param
    cols = []
    for j in range(10):
        M0j, M1j, M2j, Kj, Cj = _param_matrices(j)
        MY = M0j[None] + M1j[None] * Y[:, None, None] + M2j[None] * (Y[:, None, None] ** 2)
        f_log = (np.einsum('tij,tj->ti', MY, qdd_log)
                 + qd_log @ Cj.T + q_log @ Kj.T)          # (T,3) logical
        cols.append((f_log @ Pinv.T))                     # (T,3) stage
    W_base = np.stack(cols, axis=2)                       # (T,3,10)

    # Coulomb columns: sign(stage velocity) per motor, placed on that motor row
    qd_stage = qd_log @ Pn                                # (T,3)
    W_coul = np.zeros((len(Y), 3, 3))
    for k in range(3):
        W_coul[:, k, k] = np.sign(qd_stage[:, k])

    sl = slice(_EDGE, -_EDGE)
    W_base = W_base[sl].reshape(-1, 10)                   # (3T,10)
    W_coul = W_coul[sl].reshape(-1, 3)                    # (3T,3)
    b      = u_stage[sl].reshape(-1)                      # (3T,)
    return W_base, W_coul, b


def solve(W, b):
    """Column-preconditioned least squares. Returns theta, resid stats, cond, sv."""
    # THEORY: column scaling to unit norm before LS -- standard preconditioning
    # for inverse-dynamics identification (columns span kg..N/rad regimes).
    scale = np.linalg.norm(W, axis=0)
    scale[scale == 0] = 1.0
    Ws = W / scale
    theta_s, _, _, sv = np.linalg.lstsq(Ws, b, rcond=None)
    theta = theta_s / scale
    resid = W @ theta - b
    rms   = float(np.sqrt(np.mean(resid ** 2)))
    frac  = rms / float(np.sqrt(np.mean(b ** 2)))
    cond  = float(sv.max() / sv.min()) if sv.min() > 0 else float('inf')
    return theta, rms, frac, cond, sv


def _lumps(theta):
    """Identifiable physical lumps from the 10-vector."""
    p = {n: float(theta[i]) for i, n in enumerate(_PNAMES)}
    return {
        'm_total': p['m1'] + p['m2'] + p['mb'] + p['mh'],
        'm_diff':  p['m1'] - p['m2'],
        'mh':      p['mh'],
        'J_sum':   p['J_sum'],
        'cg1':     p['cg1'], 'cg2': p['cg2'], 'cy': p['cy'],
        'cb_sum':  p['cb_sum'], 'kb_sum': p['kb_sum'],
    }


_LUMP_NOM = {'m_total': 53.8, 'm_diff': -0.5, 'mh': 10.1, 'J_sum': 1.05,
             'cg1': 14.5, 'cg2': 20.3, 'cy': 10.0, 'cb_sum': 18.0, 'kb_sum': 3975.0}


def _report_fit(tag, theta, rms, frac, cond, fc=None):
    print(f'\n  --- {tag} ---')
    print(f'  residual: {rms:8.2f} N   ({frac*100:5.1f}% of |u|)   '
          f'cond(W) = {cond:.2e}')
    print(f'  {"lump":>8} {"nominal":>10} {"identified":>11} {"ratio":>7}  plaus?')
    plausible = True
    lm = _lumps(theta)
    for name, val in lm.items():
        nom = _LUMP_NOM[name]
        ratio = val / nom if nom != 0 else float('nan')
        ok = bool((val * np.sign(nom) > 0) and (0.5 < abs(ratio) < 2.0))
        plausible = plausible and ok
        print(f'  {name:>8} {nom:>10.3f} {val:>11.3f} {ratio:>7.2f}  '
              f'{"ok" if ok else "FLAG"}')
    if fc is not None:
        print(f'  {"Coulomb":>8} {"spec":>10} {"identified":>11}')
        for k, ax in enumerate(_COULOMB):
            print(f'  {ax:>8} {_FC_SPEC[k]:>10.0f} {fc[k]:>11.1f} N')
    return plausible, lm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fast', action='store_true')
    args = ap.parse_args()
    ops = _TRAIN_OPS[:4] if args.fast else _TRAIN_OPS
    os.makedirs(_SAVE_DIR, exist_ok=True)

    print('=' * 70)
    print('LINEAR BASELINE IDENTIFICATION  (real data, no training)')
    print('  fit  M(Y)qdd + C qd + K q  (+ Coulomb)  = u  by least squares')
    print('=' * 70)

    per_op, Wb_all, Wc_all, b_all = {}, [], [], []
    for op in ops:
        Wb, Wc, b = build_regressor(op)
        Wb_all.append(Wb); Wc_all.append(Wc); b_all.append(b)
        th, rms, frac, cond, _ = solve(Wb, b)
        per_op[op] = dict(resid_frac=float(frac),
                          m_total=float(_lumps(th)['m_total']), cond=float(cond))
        print(f'  {op:<22} resid {frac*100:5.1f}%   '
              f'm_total {_lumps(th)["m_total"]:6.1f} (nom 53.8)   cond {cond:.1e}')

    # pooled global identification
    Wb = np.vstack(Wb_all); Wc = np.vstack(Wc_all); b = np.concatenate(b_all)
    print(f'\n{"=" * 70}\nPOOLED over {len(ops)} operating points ({Wb.shape[0]} rows)\n{"=" * 70}')

    th_b, rms_b, frac_b, cond_b, sv_b = solve(Wb, b)
    plaus_b, lump_b = _report_fit('baseline only', th_b, rms_b, frac_b, cond_b)

    Wfull = np.hstack([Wb, Wc])
    th_f, rms_f, frac_f, cond_f, _ = solve(Wfull, b)
    plaus_f, lump_f = _report_fit('baseline + Coulomb', th_f[:10], rms_f, frac_f,
                                  cond_f, fc=th_f[10:])

    # verdict
    print(f'\n{"=" * 70}\nVERDICT\n{"=" * 70}')
    mt = lump_f['m_total'] / 53.8
    if not plaus_f:
        if 0.35 < mt < 0.75:
            v = ('RED: best physical fit needs ~half the real mass '
                 f'(m_total x{mt:.2f}). Data NOT directly fittable by the '
                 'physical baseline -> force-input scale/units (D-077) must be '
                 'resolved before parameter recovery.')
        else:
            v = ('AMBER: identified parameters are non-physical (see FLAGs). '
                 'Inspect conditioning and force units.')
    elif frac_f < 0.25:
        v = ('GREEN: a physically-plausible baseline+Coulomb fits the data '
             f'({frac_f*100:.0f}% residual). Data IS useful for baseline '
             'fitting; friction is the remaining gap (augmentation target).')
    else:
        v = (f'AMBER: parameters physical but residual still {frac_f*100:.0f}%; '
             'structural mismatch beyond friction.')
    print('  ' + v)
    print(f'  baseline-only residual {frac_b*100:.0f}%  ->  '
          f'+Coulomb residual {frac_f*100:.0f}%   '
          f'(Coulomb per motor: '
          f'{", ".join(f"{_COULOMB[k]} {th_f[10+k]:.0f}N" for k in range(3))})')

    # figure: reconstruction on one op + identified-vs-nominal lumps
    Wb0, Wc0, b0 = build_regressor(ops[0])
    T = b0.shape[0] // 3
    u_m  = b0.reshape(T, 3)
    u_b  = (Wb0 @ th_b).reshape(T, 3)
    u_bc = (np.hstack([Wb0, Wc0]) @ th_f).reshape(T, 3)
    fs = 20000.0
    t = np.arange(T) / fs
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for i, ax in enumerate(_COULOMB):
        axs[i].plot(t, u_m[:, i], color='0.7', lw=0.8, label='measured u')
        axs[i].plot(t, u_b[:, i], 'C0', lw=0.8, alpha=0.8, label='baseline fit')
        axs[i].plot(t, u_bc[:, i], 'C3', lw=0.8, alpha=0.8, label='baseline+Coulomb fit')
        axs[i].set_ylabel(f'{ax} force [N]')
        axs[i].grid(alpha=0.3)
        if i == 0:
            axs[i].legend(fontsize=8, loc='upper right')
    axs[-1].set_xlabel('time [s]')
    fig.suptitle(f'Linear baseline identification (pooled theta on {ops[0]})\n'
                 f'{v}', fontsize=9)
    fig.tight_layout()
    fp = os.path.join(_SAVE_DIR, 'linear_id_reconstruction.png')
    fig.savefig(fp, dpi=150)
    plt.close(fig)
    print(f'\n  figure -> {os.path.relpath(fp, _ROOT)}')

    with open(os.path.join(_SAVE_DIR, 'summary.json'), 'w') as fh:
        json.dump({'per_op': per_op,
                   'pooled_baseline': dict(resid_frac=float(frac_b), cond=float(cond_b),
                                           lumps=lump_b, plausible=bool(plaus_b)),
                   'pooled_coulomb': dict(resid_frac=float(frac_f), cond=float(cond_f),
                                          lumps=lump_f, plausible=bool(plaus_f),
                                          fc={_COULOMB[k]: float(th_f[10 + k])
                                              for k in range(3)}),
                   'verdict': v}, fh, indent=2)
    print('  summary -> ' + os.path.relpath(
        os.path.join(_SAVE_DIR, 'summary.json'), _ROOT))


if __name__ == '__main__':
    main()
