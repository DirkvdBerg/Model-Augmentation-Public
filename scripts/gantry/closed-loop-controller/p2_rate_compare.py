"""P2: the sample-rate fork (PLAN-controller-in-the-loop.md).

Cfb was designed and discretised at ts = 5e-5 s (20 kHz). The training pipeline decimates to
4 kHz (msd-offset/plant.py, load_record(fs_new=4000)). At 4 kHz the 100 Hz bandwidth has only
16 samples per period and the Tustin images of the poles move, so Cfb_4k is a DIFFERENT
controller. This script quantifies how different, so the choice can be made on numbers.

The normalisation gains kappa_j are unchanged by the rate: eq. (6) of controller-in-derivation.tex
evaluates the CONTINUOUS plant and the continuous Cnorm at the bandwidth, and only the final
discretisation step sees ts.

Criteria:
  P2a  the table exists and is logged
  P2c  flag if sigma_max(So) at 150 Hz differs by more than 10 % between the rates, or if any
       diagonal loop's phase margin moves by more than 5 degrees.
       HEURISTIC thresholds, chosen to catch a qualitatively different loop rather than to
       certify one that is not.
"""
__project_origin__ = "added"

import numpy as np
from scipy.signal import cont2discrete

from verify_controller import (M_op, sys_stage_frf, cnorm_at, cnorm_coeffs,
                               P, C_DAMP, K_STIFF, FBW, W)

RATES = [(20e3, '20 kHz'), (4e3, '4 kHz')]
FREQS = [1.0, 10.0, 50.0, 100.0, 150.0, 180.0, 500.0]
Y_OPS = [0.10, 0.00]
TOL_SO_REL = 0.10          # HEURISTIC
TOL_PM_DEG = 5.0           # HEURISTIC


def build_cfb_at(Y_op, ts):
    """Same formula as verify_controller.build_cfb, with ts a parameter."""
    sysw = sys_stage_frf(Y_op, 1j * W)
    cw = cnorm_at(1j * W)
    num, den = cnorm_coeffs()
    out, gains = [], []
    for j in range(3):
        kj = 1.0 / abs(sysw[j, j] * cw)
        b, a, _ = cont2discrete((kj * num, den), ts, method='bilinear')
        out.append((np.asarray(b).ravel(), np.asarray(a).ravel()))
        gains.append(kj)
    return out, np.array(gains)


def G_discrete_frf(Y_op, f, ts):
    M = M_op(Y_op)
    Minv = np.linalg.inv(M)
    A = np.block([[np.zeros((3, 3)), np.eye(3)], [-Minv @ K_STIFF, -Minv @ C_DAMP]])
    B = np.vstack([np.zeros((3, 3)), Minv])
    Cm = np.hstack([np.eye(3), np.zeros((3, 3))])
    Ad, Bd, Cd, Dd, _ = cont2discrete((A, B @ P, P.T @ Cm, np.zeros((3, 3))), ts, method='zoh')
    z = np.exp(1j * 2 * np.pi * f * ts)
    return Cd @ np.linalg.solve(z * np.eye(6) - Ad, Bd) + Dd


def Cfb_frf(cfb, f, ts):
    z = np.exp(1j * 2 * np.pi * f * ts)
    return np.diag([np.polyval(b, z) / np.polyval(a, z) for b, a in cfb])


def phase_margin(Y_op, cfb, ts, j):
    """Phase margin of diagonal loop j: phase of L_jj at the frequency where |L_jj| = 1."""
    fg = np.logspace(0, np.log10(0.5 / ts) - 1e-6, 4000)
    Lm = np.empty(len(fg))
    Lp = np.empty(len(fg))
    for i, f in enumerate(fg):
        L = G_discrete_frf(Y_op, f, ts)[j, j] * Cfb_frf(cfb, f, ts)[j, j]
        Lm[i] = abs(L)
        Lp[i] = np.angle(L, deg=True)
    idx = np.where(np.diff(np.sign(Lm - 1.0)))[0]
    if len(idx) == 0:
        return np.nan, np.nan
    i0 = idx[0]
    w = (1.0 - Lm[i0]) / (Lm[i0 + 1] - Lm[i0])
    fc = fg[i0] + w * (fg[i0 + 1] - fg[i0])
    ph = Lp[i0] + w * (Lp[i0 + 1] - Lp[i0])
    return fc, 180.0 + ph


# CHANGED: the report body is guarded so it fires only when this file is RUN, not when it is
# imported. `loss_variants` imports build_cfb_at from here, and cl_controller imports that, so
# every importer was paying the full phase_margin sweep (4000 freqs x 3 channels x 2 rates x 2
# operating points, about 20 s) and printing the P2 table, including inside training jobs.
# Running `python p2_rate_compare.py` still prints the identical table.
if __name__ == '__main__':
    print('P2  sample-rate comparison of the frozen design loop, f_bw = %g Hz\n' % FBW)
    so150 = {}
    for Y_op in Y_OPS:
        print('Y_op = %.2f m' % Y_op)
        print('  %-8s %-10s %-10s %-10s %-10s %-10s %-10s %-10s' % ('rate', *['%g Hz' % f for f in FREQS]))
        for fs, lab in RATES:
            ts = 1.0 / fs
            cfb, _ = build_cfb_at(Y_op, ts)
            row = []
            for f in FREQS:
                if f >= 0.5 / ts:
                    row.append(np.nan)
                    continue
                G = G_discrete_frf(Y_op, f, ts)
                C = Cfb_frf(cfb, f, ts)
                So = np.linalg.inv(np.eye(3) + G @ C)
                row.append(np.linalg.svd(So, compute_uv=False)[0])
            so150[(Y_op, lab)] = row[FREQS.index(150.0)]
            print('  %-8s %s' % (lab, ' '.join('%-10.4f' % v for v in row)))

        print('  phase margin [deg] and crossover [Hz]:')
        for fs, lab in RATES:
            ts = 1.0 / fs
            cfb, _ = build_cfb_at(Y_op, ts)
            pms, fcs = [], []
            for j in range(3):
                fc, pm = phase_margin(Y_op, cfb, ts, j)
                fcs.append(fc); pms.append(pm)
            print('  %-8s PM [%6.2f %6.2f %6.2f]   fc [%7.2f %7.2f %7.2f]'
                  % (lab, *pms, *fcs))
            if lab == '20 kHz':
                pm_ref = np.array(pms)
            else:
                dpm = np.abs(np.array(pms) - pm_ref)
                flag = 'FLAG' if np.nanmax(dpm) > TOL_PM_DEG else 'ok'
                print('           phase-margin shift [%5.2f %5.2f %5.2f] deg   tol %.0f   %s'
                      % (*dpm, TOL_PM_DEG, flag))
        d = abs(so150[(Y_op, '4 kHz')] - so150[(Y_op, '20 kHz')]) / so150[(Y_op, '20 kHz')]
        print('  sigma_max(So) at 150 Hz: %.4f -> %.4f, relative change %.2f %%   tol %.0f %%   %s\n'
              % (so150[(Y_op, '20 kHz')], so150[(Y_op, '4 kHz')], 100 * d,
                 100 * TOL_SO_REL, 'FLAG' if d > TOL_SO_REL else 'ok'))

    print('P2a table produced. Record the rate decision in docs/decisions.md before implementing')
    print('A, B or C at a rate other than 20 kHz.')
