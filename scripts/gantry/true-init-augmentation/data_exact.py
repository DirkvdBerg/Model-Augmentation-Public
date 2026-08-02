"""Record loading and the EXACT truth state, with analytic velocities.

WHY THIS FILE EXISTS
--------------------
Every initial condition used in this project so far came from `x_logical` in the
`.mat` records, whose velocity rows are `gradient()` finite differences
(`Matlab-scripts/Augmentation/data/gtd_save_record.m:22`, and the same for
`vdelta_a`). On the K = 0 axes (X and Y have no restoring force) a seed velocity
error never decays: it displaces the trajectory by `dv * tau` permanently. That
alone was measured at ~1e-06 m on X, an order above the project's own e-8/e-9
standard (`scripts/gantry/coulomb-offset/IMPLEMENTATION-LOG.md`, trap T4).

The fix used there was `[0, 0, Y_op, 0, 0, 0]` at `t = 0`, which is exact only
because the machine starts at rest. This experiment needs an exact initial
condition at ARBITRARY window starts, so the rest IC is not enough.

WHAT "ANALYTICALLY EXACT" MEANS HERE
------------------------------------
The truth's velocities are STATES of the 8-state extended plant, not derived
quantities. So we integrate that plant ourselves, in Python, at the rate it was
generated at (20 kHz, ZOH input, RK4), starting from the one initial condition
that is exactly known (rest). Every velocity we then read is an integrator state,
never a difference quotient. Decimating that trajectory by D gives an exact state
at every sample of the training grid.

This is validated, not assumed: `gate_replay()` compares the replayed POSITIONS
against the record's own positions, which are independent of our integration.
The established figure for this dataset is 5.37e-10 m on X (coulomb-offset log,
trap T4). Anything of that order means the replay is the record.

The 8-state EOM is imported from `scripts/gantry/gantry_dynamic/oracle.py`, the
pipeline's own oracle, verified against the Simulink data under D-097. It is not
restated here: a second copy is a second thing to be wrong.

STATE ORDER. `oracle` uses  x8 = [X, Th, Y, da, dX, dTh, dY, vda]  while the
project's logical 6-state is  x6 = [X, Th, Y, dX, dTh, dY]. `x6_from_x8` is the
only place that mapping is written.
"""
__project_origin__ = "added"

import os
import sys

import numpy as np
from scipy.io import loadmat

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts', 'gantry'))

from gantry_dynamic.oracle import _deriv as _deriv8, P_np   # noqa: E402

TRAJ = os.path.join(REPO, 'data', 'gantry', 'matlab', 'trajectory', 'augmentation')
FS_ORIG = 20000
FS_NEW = 4000


def load_record(name, fs_new=FS_NEW, traj_dir=TRAJ):
    """One record at fs_new, using the TRAINING pipeline's own conventions.

    u is a per-hold-interval block mean and y/x are point-sampled, exactly as
    `scripts/gantry/gantry_dynamic/data.py::_resample_u` / `load_mat_aug` do
    (D-087). Deviating here would make the target we check different from the
    target training sees, which is the whole point of the check.
    """
    dm = loadmat(os.path.join(traj_dir, name + '.mat'), squeeze_me=True)
    ts0 = float(dm['dt'])
    D = int(round((1.0 / ts0) / fs_new))
    u0 = np.asarray(dm['u_total'], float)
    n = len(u0) // D
    rec = dict(
        name=name, ts=ts0 * D, D=D, fs=fs_new, traj_dir=traj_dir,
        u=u0[:n * D].reshape(n, D, 3).mean(axis=1),        # block mean (D-087)
        y=np.asarray(dm['y'], float)[::D][:n],             # STAGE positions
        x_logical=np.asarray(dm['x_logical'], float)[::D][:n],   # FD velocities!
        r=np.asarray(dm['r_sim'], float)[::D][:n],
    )
    for opt in ('delta_a', 'vdelta_a'):
        if opt in dm:
            rec[opt] = np.asarray(dm[opt], float)[::D][:n]
    rec['u_log'] = (P_np @ rec['u'].T).T                   # stage -> logical force
    rec['Y_op'] = float(rec['y'][0, 2])
    rec['t'] = np.arange(n) * rec['ts']
    return rec


def rollout8(x0_8, u_log, ts, up_sample=1):
    """RK4 on the 8-state truth, u held (ZOH) per output step. Returns (N, 8)."""
    N = len(u_log)
    out = np.empty((N, 8))
    x = np.asarray(x0_8, dtype=np.float64).copy()
    h = ts / up_sample
    for k in range(N):
        out[k] = x
        uk = u_log[k]
        for _ in range(up_sample):
            k1 = _deriv8(x, uk)
            k2 = _deriv8(x + 0.5 * h * k1, uk)
            k3 = _deriv8(x + 0.5 * h * k2, uk)
            k4 = _deriv8(x + h * k3, uk)
            x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return out


def x6_from_x8(x8):
    """[X,Th,Y,da,dX,dTh,dY,vda] -> [X,Th,Y,dX,dTh,dY]  (the model's 6 states)."""
    x8 = np.atleast_2d(x8)
    return np.concatenate([x8[:, 0:3], x8[:, 4:7]], axis=1)


def exact_truth(name, fs_new=FS_NEW, traj_dir=TRAJ, cache=True):
    """Exact 8-state truth on the fs_new grid, integrated at FS_ORIG from rest.

    Returns a dict with
      x8      (N, 8) exact truth state at fs_new  [X,Th,Y,da,dX,dTh,dY,vda]
      x6      (N, 6) the model's physical partition of it
      rec     the fs_new record (u block-mean, y/x_logical point-sampled)
      gate    replay-vs-record position residual, per axis
    """
    tag = f'_exact_{name}_{fs_new}.npz'
    path = os.path.join(HERE, 'figures', tag)
    rec = load_record(name, fs_new=fs_new, traj_dir=traj_dir)
    if cache and os.path.exists(path):
        z = np.load(path)
        x8 = z['x8']
    else:
        hi = load_record(name, fs_new=FS_ORIG, traj_dir=traj_dir)
        # The one exactly-known initial condition: the Simulink integrators start
        # at zero and the Y integrator at Y_op, everything at rest.
        x0 = np.array([0., 0., hi['Y_op'], 0., 0., 0., 0., 0.])
        full = rollout8(x0, hi['u_log'], hi['ts'], up_sample=1)
        D = FS_ORIG // fs_new
        x8 = full[::D][:len(rec['u'])]
        if cache:
            np.savez_compressed(path, x8=x8)
    gate = np.abs(x6_from_x8(x8)[:, :3] - rec['x_logical'][:, :3]).max(axis=0)
    return dict(name=name, x8=x8, x6=x6_from_x8(x8), rec=rec, gate=gate,
                ts=rec['ts'], fs=fs_new)


def gate_replay(res, tol_x=5e-09):
    """Does the replay reproduce the record's own positions? Prints and returns."""
    ok = bool(res['gate'][0] <= tol_x)
    print(f"  replay-vs-record max |dq| [m,rad,m]: "
          f"X {res['gate'][0]:.4e}  Theta {res['gate'][1]:.4e}  Y {res['gate'][2]:.4e}"
          f"   -> {'PASS' if ok else 'CHECK'} (X tol {tol_x:.1e})")
    return ok


def fd_velocity_error(res):
    """How wrong ARE the record's finite-difference velocities? (m/s, rad/s)

    This is the quantity the handoff calls "the velocity fix". Reported as an
    absolute max and as an RMS ratio against the true velocity RMS.
    """
    v_true = res['x6'][:, 3:]
    v_fd = res['rec']['x_logical'][:, 3:]
    e = v_fd - v_true
    return dict(max_abs=np.abs(e).max(axis=0), rms=e.std(axis=0),
                rms_true=v_true.std(axis=0),
                rel=e.std(axis=0) / np.maximum(v_true.std(axis=0), 1e-30))
