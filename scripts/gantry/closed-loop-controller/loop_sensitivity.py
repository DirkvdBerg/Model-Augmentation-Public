"""Sensitivity of the frozen design loop, for the numbers quoted in the LaTeX note.

Builds the same objects the generator builds: the frozen stage-frame design plant
G = c2d(P' getss(M_op, C, K) P, ts, 'zoh') and the hand-built Cfb(z) of verify_controller.py,
then reports sigma_max of

    So = (I + G Cfb)^-1     output sensitivity
    Si = (I + Cfb G)^-1     input sensitivity
    T  = So G Cfb           complementary sensitivity

at a few frequencies. The one that matters is f_a = 150 Hz, the absorber: if |So| there is
close to 1 the loop leaves the model discrepancy of interest untouched, which is the argument
for exciting in [130, 180] Hz.
"""
__project_origin__ = "added"

import numpy as np
from scipy.signal import cont2discrete
from verify_controller import (M_op, cnorm_coeffs, build_cfb, P, C_DAMP, K_STIFF, TS, FBW)

FREQS = [1.0, 10.0, 50.0, 100.0, 150.0, 180.0, 500.0]
Y_OPS = [0.10, 0.00]


def G_discrete_frf(Y_op, f):
    """c2d(P' * getss(n, M_op, C_damp, K) * P, ts, 'zoh') evaluated at e^{i 2 pi f ts}."""
    M = M_op(Y_op)
    Minv = np.linalg.inv(M)
    A = np.block([[np.zeros((3, 3)), np.eye(3)], [-Minv @ K_STIFF, -Minv @ C_DAMP]])
    B = np.vstack([np.zeros((3, 3)), Minv])
    Cm = np.hstack([np.eye(3), np.zeros((3, 3))])
    # stage in / stage out, then ZOH discretise the whole thing (as gtd_build_plant.m does)
    Ad, Bd, Cd, Dd, _ = cont2discrete((A, B @ P, P.T @ Cm, np.zeros((3, 3))), TS, method='zoh')
    z = np.exp(1j * 2 * np.pi * f * TS)
    return Cd @ np.linalg.solve(z * np.eye(6) - Ad, Bd) + Dd


def Cfb_frf(cfb, f):
    z = np.exp(1j * 2 * np.pi * f * TS)
    d = [np.polyval(b, z) / np.polyval(a, z) for b, a in cfb]
    return np.diag(d)


print('frozen design loop, f_bw = %g Hz\n' % FBW)
for Y_op in Y_OPS:
    cfb, gains = build_cfb(Y_op)
    print('Y_op = %.2f m' % Y_op)
    print('  %8s %12s %12s %12s' % ('f [Hz]', 'smax(So)', 'smax(Si)', 'smax(T)'))
    for f in FREQS:
        G = G_discrete_frf(Y_op, f)
        C = Cfb_frf(cfb, f)
        I = np.eye(3)
        So = np.linalg.inv(I + G @ C)
        Si = np.linalg.inv(I + C @ G)
        T = So @ G @ C
        s = lambda X: np.linalg.svd(X, compute_uv=False)[0]
        print('  %8.1f %12.4f %12.4e %12.4f' % (f, s(So), s(Si), s(T)))
    print('')
