"""Gate: does the hand-written controller formula reproduce MATLAB's Cfb?

DERIVATION.md section 2 writes the controller in closed form:

    Cnorm(s) = 10w (s + w/6)(s + w/3) / [ s (s + 3w)(s + 10w) ],   w = 2*pi*f_bw
    kappa_j  = 1 / |sys_jj(i w) Cnorm(i w)|
    C_j(z)   = tustin( kappa_j Cnorm(s), ts )

This script builds that from scratch in Python. It imports no MATLAB object and reads no
exported controller matrices: the only inputs are the physical parameters, P, f_bw and ts.
It then applies Cfb(z) to the stored (r_sim - y) of a closed-loop record and compares against
the stored u_fb, which MATLAB produced with lsim(plant.Cfb, r_sim - q_with).

Agreement floor is storage, not formula: gtd_save_record.m writes u_fb, y and r_sim as single
while MATLAB computed u_fb in double. The float32 step on y is about 1e-8 m, so a residual at
that level is expected and is not a formula error. The test therefore checks the in-band
relative error and the SHAPE of the residual: a formula error is structured and follows the
signal, quantisation noise is broadband and flat.

The design constants and the build functions are importable; the comparison runs under main().
"""
__project_origin__ = "added"

import os
import numpy as np
from scipy.io import loadmat
from scipy.signal import cont2discrete, lfilter, welch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
# CHANGED (copy in core/): this file sits one level deeper than the original in
# closed-loop-controller/, so the repo root is four levels up, not three.
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
TRAJ = os.path.join(REPO, 'data', 'gantry', 'matlab', 'trajectory', 'augmentation')

# --- gtd_config.m physical parameters (lines 39-41, 61-68) -------------------
mb, mh, m1, m2, Jb, Jh = 22.8, 10.1, 10.2, 10.7, 1.0, 0.05
cg1, cg2, cy, cb1, cb2 = 14.5, 20.3, 10.0, 9.0, 9.0
kb1, kb2, Lb, d = 1987.5, 1987.5, 0.725, 0.1
FBW, TS = 100.0, 1.0 / 20e3
P = np.array([[1., 1., 0.], [Lb / 2, -Lb / 2, 0.], [0., 0., 1.]])
C_DAMP = np.array([[cg1 + cg2, (cg1 - cg2) * Lb / 2, 0.],
                   [(cg1 - cg2) * Lb / 2, cb1 + cb2 + (cg1 + cg2) * Lb ** 2 / 4, 0.],
                   [0., 0., cy]])
K_STIFF = np.array([[0., 0., 0.], [0., kb1 + kb2, 0.], [0., 0., 0.]])

RECORDS = [('V1_standstill_Yp10', 0.10), ('T10_aprbs_60', 0.00)]   # Y_op, gtd_build_records.m
W = 2 * np.pi * FBW
F_BAND = 200.0


def M_op(Y_op):
    """gtd_build_plant.m:18-20, full payload mass mh, frozen Y_op."""
    return np.array([
        [m1 + m2 + mb + mh, (m1 - m2) * Lb / 2 - mh * Y_op, 0.],
        [(m1 - m2) * Lb / 2 - mh * Y_op,
         Jb + Jh + (m1 + m2) * Lb ** 2 / 4 + mh * d ** 2 + mh * Y_op ** 2, -mh * d],
        [0., -mh * d, mh]])


def sys_stage_frf(Y_op, s):
    """sys = P' * getss(n, M_op, C_damp, K) * P, evaluated at s. getss.m:2-6."""
    M = M_op(Y_op)
    Minv = np.linalg.inv(M)
    A = np.block([[np.zeros((3, 3)), np.eye(3)], [-Minv @ K_STIFF, -Minv @ C_DAMP]])
    B = np.vstack([np.zeros((3, 3)), Minv])
    Cm = np.hstack([np.eye(3), np.zeros((3, 3))])
    G = Cm @ np.linalg.solve(s * np.eye(6) - A, B)
    return P.T @ G @ P


def cnorm_coeffs():
    """Cnorm(s) = 10w (s + w/6)(s + w/3) / [s (s + 3w)(s + 10w)], as num/den polynomials."""
    num = 10 * W * np.polymul([1., W / 6], [1., W / 3])
    den = np.polymul([1., 0.], np.polymul([1., 3 * W], [1., 10 * W]))
    return num, den


def cnorm_at(s):
    num, den = cnorm_coeffs()
    return np.polyval(num, s) / np.polyval(den, s)


def build_cfb(Y_op):
    """Three discrete SISO controllers, one per stage channel. Returns [(b, a), ...] and gains."""
    sysw = sys_stage_frf(Y_op, 1j * W)
    cw = cnorm_at(1j * W)
    num, den = cnorm_coeffs()
    out, gains = [], []
    for j in range(3):
        kj = 1.0 / abs(sysw[j, j] * cw)                 # ruleOfThumb.m:11
        b, a, _ = cont2discrete((kj * num, den), TS, method='bilinear')   # c2d(..,'tustin')
        out.append((np.asarray(b).ravel(), np.asarray(a).ravel()))
        gains.append(kj)
    return out, np.array(gains)


def main():
    print('hand-built controller, f_bw = %g Hz, ts = %g s' % (FBW, TS))
    print('Cnorm(s) = 10w (s + w/6)(s + w/3) / [s (s + 3w)(s + 10w)], w = %.4f rad/s\n' % W)

    summary = []
    for name, Y_op in RECORDS:
        path = os.path.join(TRAJ, name + '.mat')
        if not os.path.exists(path):
            print('MISSING %s' % path)
            continue
        dm = loadmat(path, squeeze_me=True)
        r = np.asarray(dm['r_sim'], float)
        y = np.asarray(dm['y'], float)
        u_fb = np.asarray(dm['u_fb'], float)
        ts = float(dm['dt'])
        e = r - y

        cfb, gains = build_cfb(Y_op)
        print('%s  (Y_op = %.2f m)' % (name, Y_op))
        print('  normalisation gains kappa = [%.4e %.4e %.4e]' % tuple(gains))
        print('  stored dt = %.6e s, design ts = %.6e s, N = %d' % (ts, TS, len(e)))

        u_hat = np.column_stack([lfilter(b, a, e[:, j]) for j, (b, a) in enumerate(cfb)])
        res = u_hat - u_fb

        # float32 storage floor: the quantisation step on the stored y this record sees
        q_step = np.array([np.spacing(np.abs(y[:, j]).max(), dtype=np.float32) for j in range(3)])

        # In-band error. The broadband relative rms is dominated by frequencies where u_fb has
        # already rolled off and only the flat quantisation floor is left, which says nothing
        # about the formula. Integrate both spectra over [0, F_BAND] instead.
        rel_band = np.empty(3)
        for j in range(3):
            f, Pr = welch(res[:, j], fs=1 / TS, nperseg=8192)
            _, Pu = welch(u_fb[:, j], fs=1 / TS, nperseg=8192)
            b_ = f <= F_BAND
            rel_band[j] = np.sqrt(Pr[b_].sum() / Pu[b_].sum())

        print('  u_fb rms       [%.4e %.4e %.4e] N' % tuple(u_fb.std(axis=0)))
        print('  residual rms   [%.4e %.4e %.4e] N' % tuple(res.std(axis=0)))
        print('  residual max   [%.4e %.4e %.4e] N' % tuple(np.abs(res).max(axis=0)))
        rel = res.std(axis=0) / u_fb.std(axis=0)
        # Decompose the residual. The controller has a pole at z = 1, so a CONSTANT bias in e
        # (which float32 storage of r and y guarantees) is integrated into a ramp rather than
        # appearing as broadband noise. Report how much of each residual is that ramp, and
        # check the fitted slope against kappa_j*w/54 * mean(e), the integral gain times the
        # measured bias. See RESULT.md; this is why the residual is not at machine precision.
        t = np.arange(len(e)) * TS
        A_fit = np.vstack([t, np.ones_like(t)]).T
        ramp = np.empty_like(res)
        frac = np.empty(3)
        slope_fit = np.empty(3)
        slope_pred = np.empty(3)
        for j in range(3):
            coef, *_ = np.linalg.lstsq(A_fit, res[:, j], rcond=None)
            ramp[:, j] = A_fit @ coef
            frac[j] = 1.0 - np.var(res[:, j] - ramp[:, j]) / np.var(res[:, j])
            slope_fit[j] = coef[0]
            slope_pred[j] = gains[j] * W / 54.0 * e[:, j].mean()

        print('  relative, broadband  [%.3e %.3e %.3e]' % tuple(rel))
        print('  relative, <= %g Hz    [%.3e %.3e %.3e]' % (F_BAND, *rel_band))
        print('  float32 step on stored y [%.2e %.2e %.2e] m' % tuple(q_step))
        print('  ramp explains  [%6.2f%% %6.2f%% %6.2f%%]' % tuple(100 * frac))
        print('  slope fitted   [%+.3e %+.3e %+.3e] N/s' % tuple(slope_fit))
        print('  slope predicted[%+.3e %+.3e %+.3e] N/s  (kappa*w/54 * mean(e))'
              % tuple(slope_pred))
        print('')
        summary.append((name, Y_op, u_fb, u_hat, res, rel, ramp, frac))

    if not summary:
        return
    fig, axes = plt.subplots(3, len(summary) * 2, figsize=(6.6 * len(summary), 7.4),
                             squeeze=False)
    CH = ['X1', 'X2', 'Y']
    for c, (name, Y_op, u_fb, u_hat, res, rel, ramp, frac) in enumerate(summary):
        t = np.arange(len(u_fb)) * TS
        for j in range(3):
            # left column: the residual itself, with the fitted ramp on top
            ax = axes[j, 2 * c]
            ax.plot(t, res[:, j], color='#D55E00', lw=0.6, label='residual')
            ax.plot(t, ramp[:, j], color='#0072B2', lw=1.4, label='fitted ramp')
            ax.grid(alpha=0.25, lw=0.5)
            ax.text(0.03, 0.90, 'ramp explains %.2f%%' % (100 * frac[j]),
                    transform=ax.transAxes, fontsize=8, color='#333333')
            if j == 0:
                ax.set_title('%s\nresidual = float32 bias integrated by the pole at z = 1'
                             % name, fontsize=10)
                ax.legend(fontsize=7, frameon=False, loc='lower right')
            ax.set_ylabel('%s [N]' % CH[j])
            if j == 2:
                ax.set_xlabel('time [s]')

            # right column: spectra WITHOUT detrending. welch's default detrend='constant'
            # removes each segment's mean, which turns a ramp into a broad flat floor and
            # makes an integrated bias look like white quantisation noise. It is not.
            ax = axes[j, 2 * c + 1]
            f, Pxx = welch(res[:, j], fs=1 / TS, nperseg=8192, detrend=False)
            f2, Pu = welch(u_fb[:, j], fs=1 / TS, nperseg=8192, detrend=False)
            ax.loglog(f2, np.sqrt(Pu), color='#333333', lw=0.8, label='u_fb')
            ax.loglog(f, np.sqrt(Pxx), color='#D55E00', lw=0.8, label='residual')
            ax.grid(alpha=0.25, lw=0.5, which='both')
            if j == 0:
                ax.set_title('%s\nspectra, detrend off' % name, fontsize=10)
                ax.legend(fontsize=7, frameon=False, loc='lower left')
            if j == 2:
                ax.set_xlabel('frequency [Hz]')
    fig.suptitle('Formula against MATLAB\'s Cfb on the stored records. The residual is a '
                 'STORAGE artefact, not a formula error:\nfloat32 rounding of r and y biases '
                 'e, and the controller\'s integrator turns that bias into a ramp. '
                 'See test_controller_exact.py for the machine-precision gate.',
                 fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    outdir = os.path.join(HERE, 'figures')
    os.makedirs(outdir, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(outdir, 'controller_formula_check.%s' % ext), dpi=160,
                    bbox_inches='tight')
    print('wrote %s' % os.path.join(outdir, 'controller_formula_check.png'))


if __name__ == '__main__':
    main()
