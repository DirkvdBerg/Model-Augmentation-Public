"""The loop itself: plant, controller, reference. Shared by P1, A and the learnability sweep.

Wiring, per gtd_run_simulation.m and the Simulink model:

    e_k    = r_k - y_k                     stage frame, y = P' q
    u_fb,k = Cc xc_k + Dc e_k              discrete controller, 9 states, ZOH between samples
    u_k    = u_fb,k + f_ms,k               injected multisine adds at the plant input
    x_k+1  = RK4(f, x_k, P u_k)            continuous plant, force held across the step
    xc_k+1 = Ac xc_k + Bc e_k

No algebraic loop: the plant feedthrough is zero (D_d = 0 in the baseline write-up eq. 10), so
y_k depends on x_k only and e_k is computable before u_k. The controller IS biproper (Tustin
gives Dc != 0), which is why the order above matters.

The controller is used in its exported state-space form, not as num/den. test_controller_exact.py
level L4 measures 1.9e-16 for the state-space realisation against 4.7e-10 for the transfer
function, and inside a loop that difference is fed back rather than merely observed.
"""
__project_origin__ = "added"

import os
import sys
import glob
import numpy as np
from scipy.io import loadmat

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'scripts', 'gantry', 'msd-offset'))
import plant as PL                                    # deriv6, deriv8, P_np, to_stage

TRAJ = os.path.join(REPO, 'data', 'gantry', 'matlab', 'trajectory', 'augmentation')
TS = 1.0 / 20e3


def load_controller(record_name):
    """MATLAB's ss(Cfb) for this record's Y_op, from export_record_reference.m."""
    path = os.path.join(HERE, 'record_reference_%s.mat' % record_name)
    if not os.path.exists(path):
        raise SystemExit('MISSING %s\nRun export_record_reference.m in MATLAB first.' % path)
    d = loadmat(path, squeeze_me=True)
    return (np.asarray(d['A'], float), np.asarray(d['B'], float),
            np.asarray(d['C'], float), np.asarray(d['D'], float), float(d['Y_op']))


def load_record(record_name):
    d = loadmat(os.path.join(TRAJ, record_name + '.mat'), squeeze_me=True)
    return dict(r=np.asarray(d['r_sim'], float),
                f_ms=np.asarray(d['f_sim'], float),
                y=np.asarray(d['y'], float),
                u_total=np.asarray(d['u_total'], float),
                u_fb=np.asarray(d['u_fb'], float),
                ts=float(d['dt']))


def simulate(deriv, x0, r, f_ms, ctrl, ts=TS, n_out=3, force_clip=None):
    """Closed-loop rollout. Returns (y_stage, u_stage, xc_final).

    deriv      f(x, u_logical) -> xdot, e.g. PL.deriv8 or PL.deriv6
    x0         plant initial state, logical coordinates
    r, f_ms    (N,3) stage-frame reference and injected force
    ctrl       (Ac, Bc, Cc, Dc)
    force_clip if set, a (3,) peak force limit; the run aborts when exceeded (divergence guard)
    """
    Ac, Bc, Cc, Dc = ctrl
    N = len(r)
    nc = Ac.shape[0]
    x = np.asarray(x0, float).copy()
    xc = np.zeros(nc)
    y_out = np.empty((N, n_out))
    u_out = np.empty((N, 3))
    Pt = PL.P_np.T
    for k in range(N):
        q = x[:3]
        y = Pt @ q                                   # stage positions
        e = r[k] - y
        u_fb = Cc @ xc + Dc @ e
        u = u_fb + f_ms[k]
        y_out[k] = y
        u_out[k] = u
        if force_clip is not None and np.any(np.abs(u) > force_clip):
            return y_out[:k + 1], u_out[:k + 1], xc          # diverged, caller checks length
        ul = PL.P_np @ u                             # stage force -> logical force
        k1 = deriv(x, ul)
        k2 = deriv(x + .5 * ts * k1, ul)
        k3 = deriv(x + .5 * ts * k2, ul)
        k4 = deriv(x + ts * k3, ul)
        x = x + (ts / 6.) * (k1 + 2 * k2 + 2 * k3 + k4)
        xc = Ac @ xc + Bc @ e
    return y_out, u_out, xc


def x0_for(model, Y_op):
    """Simulink's integrator initial condition, [0, 0, Y_op, ...]."""
    if model == 'truth':
        return np.array([0., 0., Y_op, 0., 0., 0., 0., 0.])
    return np.array([0., 0., Y_op, 0., 0., 0.])


def ramp_fraction(res, ts=TS):
    """Share of the variance explained by a straight line, per column. See RESULT.md: a ramp
    means a constant bias is being integrated by the controller's pole at z = 1."""
    res = np.atleast_2d(res.T).T
    t = np.arange(len(res)) * ts
    A = np.vstack([t, np.ones_like(t)]).T
    out = np.empty(res.shape[1])
    for j in range(res.shape[1]):
        coef, *_ = np.linalg.lstsq(A, res[:, j], rcond=None)
        out[j] = 1.0 - np.var(res[:, j] - A @ coef) / np.var(res[:, j])
    return out


def available_records():
    return sorted(os.path.basename(p)[len('record_reference_'):-len('.mat')]
                  for p in glob.glob(os.path.join(HERE, 'record_reference_*.mat')))
