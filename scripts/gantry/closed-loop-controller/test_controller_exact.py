"""Exactness test: is the closed-form controller bit-for-bit MATLAB's Cfb?

verify_controller.py already checks the formula against the generated records, but those store
u_fb, y and r_sim as single (gtd_save_record.m), which caps the agreement at about 1e-9. This
test removes that cap by comparing against export_controller.m's double-precision export of the
controller object itself.

Three levels, increasing in strictness and decreasing in what they can blame:

  L1  coefficients   our num/den of C_j(z) against MATLAB's tfdata, plus poles, zeros and the
                     design scalars kappa_j and sys_jj(i wb). No simulation is involved, so a
                     pass here IS the formula being right. This is the sharpest test.
  L2  realisation    MATLAB's exported (A,B,C,D) run in Python on e_test, against MATLAB's
                     u_test. Both sides are then the same realisation, so this measures
                     arithmetic only and calibrates the floor for L3.
  L3  end to end     our num/den run in Python on e_test, against MATLAB's u_test. This is the
                     claim the LaTeX note makes.

L3 is expected to be looser than L2. gtd_build_plant.m:28 does Cfb = ss(Cfb), so MATLAB
simulates a state-space realisation of the transfer function; with kappa ~ 2e7 and a pole at
z = 1, the tf-to-ss conversion is ill conditioned. A gap between L2 and L3 is that conversion,
not a wrong formula, and L1 is what proves which of the two it is.

Run export_controller.m first (MATLAB, seconds, no Simulink).
"""
__project_origin__ = "added"

import os
import sys
import numpy as np
from scipy.io import loadmat
from scipy.signal import lfilter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from verify_controller import build_cfb, sys_stage_frf, cnorm_at, W, TS, FBW

HERE = os.path.dirname(os.path.abspath(__file__))
EXPORT = os.path.join(HERE, 'controller_export.mat')
REPO_TRAJ = os.path.abspath(os.path.join(HERE, '..', '..', '..', 'data', 'gantry', 'matlab',
                                         'trajectory', 'augmentation'))

# Pass thresholds.
#   L1  Set by conditioning, not by machine epsilon. The numerator coefficients are of order
#       kappa*10*w ~ 1e6 and are formed by polynomial products followed by a bilinear
#       transform, so cancellation puts the achievable relative agreement near 1e-11, which
#       is where it actually lands. 1e-10 keeps a decade of margin; tightening to 1e-11 makes
#       the test fail on arithmetic reordering rather than on a wrong formula.
#   L2  Same realisation on both sides, so this is pure round-off and lands at 1e-16.
#   L3  Carries the tf-to-ss conversion inside ss(Cfb), which is ill conditioned with a pole
#       at z = 1; 1e-7 is loose enough to pass that and tight enough to catch a real error.
TOL_L1, TOL_L2, TOL_L3 = 1e-10, 1e-11, 1e-7
CH = ['X1', 'X2', 'Y']


def relerr(a, b):
    """Relative error, scaled by the larger magnitude so a near-zero reference cannot inflate it.

    Complex-safe: poles and zeros are complex in general, and casting them to float would
    silently compare real parts only. This loop's roots happen to be real, but a different
    f_bw or design plant makes them complex and the test must not go blind to that.
    """
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    scale = max(np.abs(a).max(), np.abs(b).max())
    return np.abs(a - b).max() / scale if scale > 0 else 0.0


def matlab_tf(dm, j):
    """(num, den) of MATLAB's C_j(z) out of the exported cell array, whatever its orientation."""
    cn, cd = dm['num'], dm['den']
    nm = cn[j, 0] if cn.shape[1] == 1 else cn[0, j]
    dn = cd[j, 0] if cd.shape[1] == 1 else cd[0, j]
    return np.asarray(nm, float).ravel(), np.asarray(dn, float).ravel()


def ss_rollout(A, B, C, D, u):
    """x_{k+1} = A x_k + B u_k, y_k = C x_k + D u_k, from rest. Matches lsim on a discrete ss."""
    n = A.shape[0]
    x = np.zeros(n)
    y = np.empty((len(u), C.shape[0]))
    for k in range(len(u)):
        y[k] = C @ x + D @ u[k]
        x = A @ x + B @ u[k]
    return y


if not os.path.exists(EXPORT):
    sys.exit('MISSING %s\nRun export_controller.m in MATLAB first.' % EXPORT)

dm = loadmat(EXPORT, squeeze_me=False)
ts_m = float(dm['ts'].ravel()[0])
fbw_m = float(dm['fbw'].ravel()[0])
Y_op = float(dm['Y_op'].ravel()[0])
print('export: ts = %.12g s, fbw = %g Hz, Y_op = %.2f m' % (ts_m, fbw_m, Y_op))
print('python: ts = %.12g s, fbw = %g Hz\n' % (TS, FBW))
assert abs(ts_m - TS) < 1e-18 and abs(fbw_m - FBW) < 1e-12, 'design constants disagree'

cfb, kappa_py = build_cfb(Y_op)
results = {}

# ---------------------------------------------------------------- L1
print('L1  COEFFICIENTS  (no simulation; a pass here is the formula being correct)')
kappa_m = np.asarray(dm['kappa'], float).ravel()
sysjj_m = np.asarray(dm['sysjj']).ravel()
sysjj_py = np.array([sys_stage_frf(Y_op, 1j * W)[j, j] for j in range(3)])
kappa_chk = np.array([1.0 / abs(sysjj_py[j] * cnorm_at(1j * W)) for j in range(3)])

e_sys = max(relerr(np.real(sysjj_m), np.real(sysjj_py)),
            relerr(np.imag(sysjj_m), np.imag(sysjj_py)))
e_kap = relerr(kappa_m, kappa_chk)
print('  sys_jj(i wb)  MATLAB [%s]' % ' '.join('%.10e' % abs(v) for v in sysjj_m))
print('                python [%s]' % ' '.join('%.10e' % abs(v) for v in sysjj_py))
print('                rel err %.3e' % e_sys)
print('  kappa         MATLAB [%s]' % ' '.join('%.10e' % v for v in kappa_m))
print('                python [%s]' % ' '.join('%.10e' % v for v in kappa_py))
print('                rel err %.3e' % e_kap)

e_num = e_den = e_pol = e_zer = 0.0
for j in range(3):
    nm, dn = matlab_tf(dm, j)
    b, a = cfb[j]
    # normalise both to a monic denominator so a common scaling cannot mask a difference
    nm_n, dn_n = nm / dn[0], dn / dn[0]
    b_n, a_n = b / a[0], a / a[0]
    e_num = max(e_num, relerr(nm_n, b_n))
    e_den = max(e_den, relerr(dn_n, a_n))
    e_pol = max(e_pol, relerr(np.sort_complex(np.roots(dn)), np.sort_complex(np.roots(a))))
    e_zer = max(e_zer, relerr(np.sort_complex(np.roots(nm)), np.sort_complex(np.roots(b))))
    if j == 0:
        print('  C_1(z) numerator  MATLAB %s' % np.array2string(nm_n, precision=12))
        print('                    python %s' % np.array2string(b_n, precision=12))
        print('  C_1(z) denominator MATLAB %s' % np.array2string(dn_n, precision=12))
        print('                    python %s' % np.array2string(a_n, precision=12))
print('  max rel err   numerator %.3e   denominator %.3e' % (e_num, e_den))
print('                poles     %.3e   zeros       %.3e' % (e_pol, e_zer))
l1 = max(e_sys, e_kap, e_num, e_den, e_pol, e_zer)
results['L1'] = (l1, TOL_L1)
print('  L1 worst %.3e   %s\n' % (l1, 'PASS' if l1 < TOL_L1 else 'FAIL'))

# ---------------------------------------------------------------- L2 and L3
e_test = np.asarray(dm['e_test'], float)
u_test = np.asarray(dm['u_test'], float)
print('test signal: %d samples, %.2f s, rms [%.3e %.3e %.3e] m'
      % (len(e_test), len(e_test) * TS, *e_test.std(axis=0)))
print('MATLAB u_test rms [%.6e %.6e %.6e] N\n' % tuple(u_test.std(axis=0)))

A = np.asarray(dm['A'], float)
B = np.asarray(dm['B'], float)
C = np.asarray(dm['C'], float)
D = np.asarray(dm['D'], float)
print('L2  REALISATION  (MATLAB\'s own (A,B,C,D) run in Python; measures arithmetic only)')
u_l2 = ss_rollout(A, B, C, D, e_test)
e2 = np.array([relerr(u_l2[:, j], u_test[:, j]) for j in range(3)])
for j in range(3):
    print('  %-3s rel err %.3e' % (CH[j], e2[j]))
results['L2'] = (e2.max(), TOL_L2)
print('  L2 worst %.3e   %s\n' % (e2.max(), 'PASS' if e2.max() < TOL_L2 else 'FAIL'))

print('L3  END TO END  (our num/den from the formulas, against MATLAB\'s u_test)')
u_l3 = np.column_stack([lfilter(b, a, e_test[:, j]) for j, (b, a) in enumerate(cfb)])
e3 = np.array([relerr(u_l3[:, j], u_test[:, j]) for j in range(3)])
for j in range(3):
    print('  %-3s rel err %.3e' % (CH[j], e3[j]))
results['L3'] = (e3.max(), TOL_L3)
print('  L3 worst %.3e   %s\n' % (e3.max(), 'PASS' if e3.max() < TOL_L3 else 'FAIL'))

# ---------------------------------------------------------------- L4
# Record level, at machine precision. verify_controller.py compares against the STORED u_fb,
# which MATLAB computed from a double q_with, while Python can only supply the single-precision
# y. That mismatch biases e by about one float32 step and the pole at z = 1 integrates it into
# a ramp, so that check cannot reach machine precision no matter how right the formula is.
# export_record_reference.m removes the mismatch by re-running MATLAB's own lsim on exactly the
# signal Python forms, double(r_sim) - double(y). Identical input bits on both sides.
import glob
REFS = sorted(glob.glob(os.path.join(HERE, 'record_reference_*.mat')))
if REFS:
    print('L4  RECORD LEVEL, SAME INPUT BITS  (machine-precision version of the record gate)')
    l4_tf, l4_ss = [], []
    for ref in REFS:
        dr = loadmat(ref, squeeze_me=True)
        nm = str(dr['name'])
        Y_op_i = float(dr['Y_op'])
        u_ref = np.asarray(dr['u_ref'], float)

        dmr = loadmat(os.path.join(REPO_TRAJ, nm + '.mat'), squeeze_me=True)
        e_rec = np.asarray(dmr['r_sim'], float) - np.asarray(dmr['y'], float)
        u_fb_stored = np.asarray(dmr['u_fb'], float)

        cfb_i, _ = build_cfb(Y_op_i)
        u_tf = np.column_stack([lfilter(b, a, e_rec[:, j]) for j, (b, a) in enumerate(cfb_i)])
        u_ss = ss_rollout(np.asarray(dr['A'], float), np.asarray(dr['B'], float),
                          np.asarray(dr['C'], float), np.asarray(dr['D'], float), e_rec)

        etf = np.array([relerr(u_tf[:, j], u_ref[:, j]) for j in range(3)])
        ess = np.array([relerr(u_ss[:, j], u_ref[:, j]) for j in range(3)])
        # For contrast: the SAME comparison against the stored u_fb, which MATLAB computed
        # from the double q_with. The gap between these two lines is the storage artefact.
        eold = np.array([relerr(u_tf[:, j], u_fb_stored[:, j]) for j in range(3)])
        l4_tf.append(etf.max()); l4_ss.append(ess.max())
        print('  %-22s Y_op %.2f' % (nm, Y_op_i))
        print('      our num/den vs MATLAB lsim, same input   [%.3e %.3e %.3e]' % tuple(etf))
        print('      MATLAB (A,B,C,D) in Python, same input   [%.3e %.3e %.3e]' % tuple(ess))
        print('      for contrast, vs the STORED u_fb         [%.3e %.3e %.3e]' % tuple(eold))
    results['L4'] = (max(l4_tf), TOL_L3)
    results['L4ss'] = (max(l4_ss), TOL_L2)
    print('  L4 worst (num/den) %.3e   (same realisation) %.3e\n' % (max(l4_tf), max(l4_ss)))
else:
    print('L4  SKIPPED: run export_record_reference.m in MATLAB to enable it.\n')

# ---------------------------------------------------------------- L5
# THE TRAINING RATE. Everything above runs at 20 kHz because that is the rate MATLAB generated
# the records at. The training loop steps Cfb at cfg.ts_new = 1/4000 s (D-141) and builds it by
# re-discretising the same continuous design in Python, so before this section the object the
# loop actually steps had never been compared against MATLAB at any level. L1 is the test that
# proves a formula rather than a simulation, so it is the one repeated here.
#
# Comparing the two implementations to EACH OTHER at 4 kHz gives 2.7e-10, i.e. worse than the
# 9.6e-12 the same comparison gives at 20 kHz, and a two-way comparison cannot say which side
# moved. So a third reference is used: the bilinear transform applied to the same double-valued
# continuous coefficients in EXACT rational arithmetic (fractions.Fraction, no rounding at all).
# Measured against it: MATLAB lands at 7e-14 at BOTH rates, while scipy's cont2discrete lands at
# 5e-12 at 20 kHz and 2.1e-10 at 4 kHz. The formula is therefore right and the whole gap is
# scipy's numerator arithmetic, which degrades as 2/ts approaches the design frequencies (at
# 20 kHz 2/ts = 40000 rad/s dominates every other term in the expansion; at 4 kHz it is 8000,
# comparable to 10w = 6283, so the alternating-sign sums cancel far more).
#
# Three things are checked, and they fail differently:
#   L5   THE FORMULA. MATLAB's 4 kHz coefficients against exact rational arithmetic. This is the
#        claim plan section 10 makes ("L1 coefficients <= 1e-11, both rates") and it is the only
#        one of the three that a wrong formula could fail.
#   L5py THE IMPLEMENTATION THE LOOP USES. scipy's cont2discrete against the same exact
#        reference, with its error attributed: if |scipy - MATLAB| tracks |scipy - exact| then
#        the two-way gap IS scipy's rounding and nothing else. Tolerance is one decade above
#        the measured floor, not a number chosen to make it pass.
#   L5ss THE REALISATION. loss_variants.controller_ss, i.e. tf2ss per channel assembled
#        block-diagonally, driven against MATLAB's own exported (A,B,C,D) at 4 kHz on the same
#        input. Different state bases are expected and allowed; the input-output response is
#        what must agree, and D must agree exactly since it is basis independent.
#
# 2.1e-10 relative on the controller coefficients is far below anything the training path can
# resolve: it runs in float32, whose epsilon is 1.2e-07, i.e. three decades coarser.
TOL_L5, TOL_L5PY = 1e-11, 1e-9


def _bilinear_exact(num, den, ts):
    """(b, a) of num(s)/den(s) under s = (2/ts)(z-1)/(z+1), in exact rational arithmetic.

    Every input is a double and Fraction(float) is exact, so the only approximation left is the
    inputs themselves. That makes this the reference both implementations are measured against.
    """
    from fractions import Fraction as Fr

    def pmul(p, q):
        r = [Fr(0)] * (len(p) + len(q) - 1)
        for i, a_ in enumerate(p):
            for j_, b_ in enumerate(q):
                r[i + j_] += a_ * b_
        return r

    def padd(p, q):
        n = max(len(p), len(q))
        p = [Fr(0)] * (n - len(p)) + list(p)
        q = [Fr(0)] * (n - len(q)) + list(q)
        return [a_ + b_ for a_, b_ in zip(p, q)]

    def ppow(p, k):
        r = [Fr(1)]
        for _ in range(k):
            r = pmul(r, p)
        return r

    c = Fr(2) / Fr(ts)
    m = max(len(num), len(den)) - 1

    def subst(poly):                       # descending coefficients, as np.poly1d
        deg = len(poly) - 1
        out = [Fr(0)]
        for i, coef in enumerate(poly):
            k = deg - i
            out = padd(out, pmul([Fr(coef)],
                                 pmul(ppow(pmul([c], [Fr(1), Fr(-1)]), k),
                                      ppow([Fr(1), Fr(1)], m - k))))
        return out
    b_, a_ = subst(num), subst(den)
    return ([float(v / a_[0]) for v in b_], [float(v / a_[0]) for v in a_])


if 'num4k' in dm:
    print('L5  TRAINING RATE, 1/4000 s  (the rate the loop steps Cfb at, D-141)')
    ts_tr = float(dm['ts_train'].ravel()[0])
    from p2_rate_compare import build_cfb_at
    from verify_controller import cnorm_coeffs
    cfb4, kappa4 = build_cfb_at(Y_op, ts_tr)
    cnum, cden = cnorm_coeffs()

    e5m = e5py = e5two = e5d = e5p = e5z = 0.0
    for j in range(3):
        cn, cd = dm['num4k'], dm['den4k']
        nm = np.asarray(cn[j, 0] if cn.shape[1] == 1 else cn[0, j], float).ravel()
        dn = np.asarray(cd[j, 0] if cd.shape[1] == 1 else cd[0, j], float).ravel()
        b, a = cfb4[j]
        nm_n, dn_n = nm / dn[0], dn / dn[0]
        b_n, a_n = b / a[0], a / a[0]
        b_ex, a_ex = _bilinear_exact([kappa4[j] * v for v in cnum], list(cden), ts_tr)
        e5m = max(e5m, relerr(nm_n, b_ex))                   # MATLAB vs exact: the formula
        e5py = max(e5py, relerr(b_n, b_ex))                  # scipy vs exact: the implementation
        e5two = max(e5two, relerr(nm_n, b_n))                # the two-way gap being explained
        e5d = max(e5d, relerr(dn_n, a_ex))
        e5p = max(e5p, relerr(np.sort_complex(np.roots(dn)), np.sort_complex(np.roots(a))))
        e5z = max(e5z, relerr(np.sort_complex(np.roots(nm)), np.sort_complex(np.roots(b))))
    print('  ts = %.12g s, %d Hz   (kappa is a frequency-response scalar: rate independent)'
          % (ts_tr, round(1 / ts_tr)))
    print('  MATLAB vs exact rational   numerator %.3e   denominator %.3e' % (e5m, e5d))
    print('  scipy  vs exact rational   numerator %.3e' % e5py)
    print('  the two-way gap            numerator %.3e   zeros %.3e   poles %.3e'
          % (e5two, e5z, e5p))
    attributed = abs(e5two - e5py) <= 0.5 * max(e5py, 1e-300)
    print('  attribution: two-way %.3e vs scipy-vs-exact %.3e -> %s'
          % (e5two, e5py,
             'the gap IS scipy\'s rounding' if attributed else
             'UNEXPLAINED: the two-way gap is not scipy\'s distance from exact'))
    results['L5'] = (max(e5m, e5d), TOL_L5)
    results['L5py'] = (e5py if attributed else max(e5py, e5two), TOL_L5PY)
    print('  L5   (formula, MATLAB vs exact) %.3e   %s'
          % (max(e5m, e5d), 'PASS' if max(e5m, e5d) < TOL_L5 else 'FAIL'))
    print('  L5py (scipy implementation)     %.3e   %s'
          % (results['L5py'][0], 'PASS' if results['L5py'][0] < TOL_L5PY else 'FAIL'))

    # the realisation the training path uses, against MATLAB's, on one deterministic signal
    from loss_variants import controller_ss
    A4 = np.asarray(dm['A4k'], float); B4 = np.asarray(dm['B4k'], float)
    C4 = np.asarray(dm['C4k'], float); D4 = np.asarray(dm['D4k'], float)
    Ap, Bp, Cp_, Dp = controller_ss(Y_op, ts_tr)

    rng = np.random.default_rng(0)
    N5 = 4000
    t5 = np.arange(N5) * ts_tr
    e5 = np.column_stack([
        1e-4 * (t5 < 0.1)
        + 1e-5 * np.sin(2 * np.pi * 150 * t5 + c)
        + 1e-6 * rng.standard_normal(N5) for c in range(3)])
    u_m = ss_rollout(A4, B4, C4, D4, e5)
    u_p = ss_rollout(Ap, Bp, Cp_, Dp, e5)
    e5r = np.array([relerr(u_p[:, j], u_m[:, j]) for j in range(3)])
    e5D = relerr(np.diag(Dp), np.diag(D4))
    print('  diag(D) MATLAB [%s] N/m' % ' '.join('%.6e' % v for v in np.diag(D4)))
    print('          python [%s] N/m' % ' '.join('%.6e' % v for v in np.diag(Dp)))
    print('          rel err %.3e   (D is basis independent, so this is exact or wrong)' % e5D)
    print('  response, our tf2ss realisation vs MATLAB ss  [%.3e %.3e %.3e]' % tuple(e5r))
    results['L5ss'] = (max(e5r.max(), e5D), TOL_L3)
    print('  L5ss worst %.3e   %s\n' % (max(e5r.max(), e5D),
                                        'PASS' if max(e5r.max(), e5D) < TOL_L3 else 'FAIL'))
else:
    print('L5  SKIPPED: controller_export.mat predates the 4 kHz export; re-run '
          'export_controller.m.\n')

# ---------------------------------------------------------------- figure
# Four rows, one confirmation per row, so each claim can be checked on its own:
#   1  frequency response of C_j(z), MATLAB against the formula, overlaid
#   2  relative difference of those two responses against the L1 tolerance (this is L1)
#   3  poles and zeros of both, with the unit circle (shows the z = -1 zero)
#   4  u_test against the formula's response on a window where the multisine is active (L3)
C_INK, C_FORM, C_TOL = '#333333', '#D55E00', '#0072B2'
fgrid = np.logspace(0, np.log10(0.5 / TS), 2000)
zg = np.exp(1j * 2 * np.pi * fgrid * TS)
fig, axes = plt.subplots(4, 3, figsize=(13.5, 13.0))
for j in range(3):
    nm, dn = matlab_tf(dm, j)
    b, a = cfb[j]
    Cm = np.polyval(nm, zg) / np.polyval(dn, zg)
    Cp = np.polyval(b, zg) / np.polyval(a, zg)

    ax = axes[0, j]
    ax.loglog(fgrid, np.abs(Cm), color=C_INK, lw=1.6, label='MATLAB $C_j(z)$')
    ax.loglog(fgrid, np.abs(Cp), color=C_FORM, lw=1.0, ls='--', label='formula, eq. (5) to (7)')
    ax.axvline(FBW, color=C_TOL, lw=0.8, ls=':')
    ax.text(FBW * 1.1, np.abs(Cm).min() * 3, '$f_b$', color=C_TOL, fontsize=8)
    ax.grid(alpha=0.25, lw=0.5, which='both')
    ax.set_title('%s' % CH[j], fontsize=11)
    if j == 0:
        ax.set_ylabel('$|C_j|$  [N/m]')
        ax.legend(fontsize=8, frameon=False, loc='lower left')

    # The rise at both ends is evaluation conditioning, not disagreement: C_j has a pole at
    # exactly z = 1 and a zero at exactly z = -1, so evaluating den(z) near DC and num(z) near
    # Nyquist cancels to zero and the RELATIVE difference is dominated by that cancellation.
    # The coefficient comparison (L1) is immune to this and is the number that counts.
    ax = axes[1, j]
    ax.loglog(fgrid, np.abs(Cm - Cp) / np.abs(Cm), color=C_FORM, lw=1.0)
    ax.axhline(l1, color=C_TOL, lw=1.0, ls='--')
    ax.text(fgrid[5], l1 * 1.6, 'L1 coefficient agreement %.1e' % l1, color=C_TOL, fontsize=8)
    ax.axvspan(fgrid[0], 100.0, color='#999999', alpha=0.15)
    ax.axvspan(6000.0, fgrid[-1], color='#999999', alpha=0.15)
    ax.set_ylim(1e-18, 1e-4)
    ax.grid(alpha=0.25, lw=0.5, which='both')
    if j == 0:
        ax.set_ylabel('relative difference\nof the two responses')
        ax.text(1.3, 3e-16, 'shaded: evaluation limited by\nthe pole at $z=1$ (DC) and\n'
                            'the zero at $z=-1$ (Nyquist)', fontsize=7, color='#555555')

    ax = axes[2, j]
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color='#999999', lw=0.8)
    pm, zm = np.roots(dn), np.roots(nm)
    pp, zp = np.roots(a), np.roots(b)
    ax.plot(np.real(pm), np.imag(pm), 'x', color=C_INK, ms=11, mew=2, label='MATLAB poles')
    ax.plot(np.real(zm), np.imag(zm), 'o', color=C_INK, ms=10, mfc='none', mew=1.6,
            label='MATLAB zeros')
    ax.plot(np.real(pp), np.imag(pp), '+', color=C_FORM, ms=11, mew=1.6, label='formula poles')
    ax.plot(np.real(zp), np.imag(zp), '.', color=C_FORM, ms=7, label='formula zeros')
    ax.set_aspect('equal')
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-0.45, 0.45)
    ax.grid(alpha=0.25, lw=0.5)
    if j == 0:
        ax.set_ylabel('imag')
        ax.legend(fontsize=7, frameon=False, loc='upper left', ncol=2)

    ax = axes[3, j]
    i0, i1 = int(0.80 / TS), int(0.83 / TS)          # inside the multisine segment
    tt = np.arange(i0, i1) * TS
    ax.plot(tt, u_test[i0:i1, j], color=C_INK, lw=1.6, label='MATLAB $u_{test}$')
    ax.plot(tt, u_l3[i0:i1, j], color=C_FORM, lw=1.0, ls='--', label='formula')
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_xlabel('time [s]')
    ax.text(0.97, 0.06, 'max rel err %.2e' % e3[j], transform=ax.transAxes, ha='right',
            fontsize=9, color=C_INK)
    if j == 0:
        ax.set_ylabel('$u$ [N]')
        ax.legend(fontsize=8, frameon=False, loc='upper left')

fig.suptitle('Exactness of the closed-form controller against MATLAB, double precision.\n'
             'The claim is the coefficient agreement (L1, dashed line in row 2); the rise at '
             'both ends of row 2 is polynomial-evaluation cancellation, not disagreement.',
             fontsize=11.5, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.955])
outdir = os.path.join(HERE, 'figures')
os.makedirs(outdir, exist_ok=True)
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(outdir, 'controller_exactness.%s' % ext), dpi=160,
                bbox_inches='tight')
print('wrote %s\n' % os.path.join(outdir, 'controller_exactness.png'))

# ---------------------------------------------------------------- verdict
print('=' * 74)
LABEL = {'L1': 'coefficients', 'L2': 'realisation', 'L3': 'end to end',
         'L4': 'record, num/den', 'L4ss': 'record, same real.',
         'L5': '4 kHz formula', 'L5py': '4 kHz scipy c2d', 'L5ss': '4 kHz realisation'}
ok = True
for name in ('L1', 'L2', 'L3', 'L4', 'L4ss', 'L5', 'L5py', 'L5ss'):
    if name not in results:
        continue
    val, tol = results[name]
    good = val < tol
    ok &= good
    print('%-5s %-20s worst %.3e   tol %.1e   %s'
          % (name, LABEL[name], val, tol, 'PASS' if good else 'FAIL'))
print('=' * 74)
if results['L3'][0] > 10 * max(results['L2'][0], 1e-16):
    print('L3 is looser than L2: the gap is the tf-to-ss conversion in ss(Cfb), not the formula.')
    print('L1 is the statement about the formula, and it is independent of both.')
sys.exit(0 if ok else 1)
