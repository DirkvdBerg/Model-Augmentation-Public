"""Exact 8-state truth for ANY augmentation dataset, with analytic velocities.

WHY THIS FILE EXISTS (D-197)
----------------------------
Every velocity stored in the `.mat` records is a MATLAB `gradient()` of the
single-precision positions (`Matlab-scripts/Augmentation/data/gtd_save_record.m`:
`qdot_logical(:,j) = gradient(q_logical(:,j), ts)` and `vda = gradient(da, ts)`).
`gradient` is ONE-SIDED at the endpoints, so at `k = 0`, where the machine is
provably at rest and the true velocity is exactly zero, the stored value is not.
Scoring an encoder against those numbers charges the encoder for an error in the
target. This module produces the target that has no such error.

WHAT "EXACT" MEANS HERE
-----------------------
The velocities are STATES of the 8-state extended plant, never difference
quotients: the plant is integrated here, in Python, at the rate it was generated
at (20 kHz, ZOH input, RK4), from the one initial condition that is exactly known
(rest, with the Y integrator at `Y_op`). Decimating gives an exact state on the
4 kHz training grid.

This is checked, not asserted, by `selfcheck()`:
  1. the parameterised EOM must reproduce `scripts/gantry/common/oracle.py::_deriv`
     to machine precision at THAT module's own absorber parameters, and
  2. `exact_truth()` gates the replayed POSITIONS against the record's own
     positions, which are independent of this integration.

`scripts/gantry/true-init-augmentation/data_exact.py` does the same thing for one
dataset. It cannot be reused: its absorber parameters are `ma_frac = 0.10`,
`zeta_a = 0.05` (the `augmentation` folder), the production run trains on
`ma_frac = 0.50`, `zeta_a = 0.03`, and it imports `gantry_dynamic/oracle.py`,
which was deleted in ead9181.

STATE ORDER.  x8 = [X, Th, Y, da, dX, dTh, dY, vda];  the model's physical
partition is x6 = [X, Th, Y, dX, dTh, dY].
"""
__project_origin__ = "added"

import os
import sys

import numpy as np
from scipy.io import loadmat

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts', 'gantry'))

from model_augmentation.systems.gantry_ss import (          # noqa: E402
    m1 as _m1, m2 as _m2, mb as _mb, mh as _mh, Jb as _Jb, Jh as _Jh,
    cg1 as _cg1, cg2 as _cg2, cy as _cy, cb1 as _cb1, cb2 as _cb2,
    kb1 as _kb1, kb2 as _kb2, Lb as _Lb, d as _d, P as _P)

m1, m2, mb, mh = float(_m1), float(_m2), float(_mb), float(_mh)
Jb, Jh = float(_Jb), float(_Jh)
cg1, cg2, cy = float(_cg1), float(_cg2), float(_cy)
cb1, cb2, kb1, kb2 = float(_cb1), float(_cb2), float(_kb1), float(_kb2)
Lb, dd = float(_Lb), float(_d)
P_np = _P.numpy().astype(np.float64)

CACHE = os.path.join(HERE, '..', 'cache')
TRAJ_ROOT = os.path.join(REPO, 'data', 'gantry', 'matlab', 'trajectory')
FS_ORIG = 20000
FS_NEW = 4000

# Hidden-MSD absorber knobs per dataset folder. The .mat files do NOT store them, so they
# are transcribed from the generator that produced each folder and then GATED by the replay
# (a wrong ma_frac or zeta_a moves the positions far above the 1e-9 m gate).
#   gtd_config.m:   fa = 150 Hz, L0 = 0.10 m, ka = ma*(2*pi*fa)^2, ca = 2*zeta*sqrt(ka*ma)
#   generate_trajectory_data.m: MA_FRAC = 0.50, ZETA_A_OVERRIDE = 0.03 for b140-230_a6_z03
MODE_ABSORBER = {
    'augmentation':                       dict(ma_frac=0.10, fa=150.0, zeta_a=0.05, L0=0.10),
    'augmentation_ma50':                  dict(ma_frac=0.50, fa=150.0, zeta_a=0.05, L0=0.10),
    'augmentation_ma50_a5':               dict(ma_frac=0.50, fa=150.0, zeta_a=0.05, L0=0.10),
    'augmentation_ma50_b165-260_a5':      dict(ma_frac=0.50, fa=150.0, zeta_a=0.05, L0=0.10),
    'augmentation_ma50_b140-230_a10':     dict(ma_frac=0.50, fa=150.0, zeta_a=0.05, L0=0.10),
    'augmentation_ma50_b140-230_a6_z03':  dict(ma_frac=0.50, fa=150.0, zeta_a=0.03, L0=0.10),
}


class Plant8:
    """The 8-state extended plant of `gantrySystemExtended.m`, absorber parameterised."""

    def __init__(self, ma_frac, fa, zeta_a, L0):
        self.ma_frac, self.fa, self.zeta_a, self.L0 = ma_frac, fa, zeta_a, L0
        self.ma = ma_frac * mh
        self.mh_rigid = mh - self.ma
        self.ka = self.ma * (2 * np.pi * fa) ** 2          # THEORY: k = m*(2*pi*f)^2 (gtd_config.m)
        self.ca = 2 * zeta_a * np.sqrt(self.ka * self.ma)  # THEORY: c = 2*zeta*sqrt(k*m) (gtd_config.m)
        self._C4 = np.array([
            [cg1 + cg2,            (cg1 - cg2) * Lb / 2,                     0.0, 0.0],
            [(cg1 - cg2) * Lb / 2, cb1 + cb2 + (cg1 + cg2) * Lb ** 2 / 4,    0.0, 0.0],
            [0.0, 0.0, cy,  0.0],
            [0.0, 0.0, 0.0, self.ca],
        ], dtype=np.float64)
        self._K4 = np.array([
            [0.0, 0.0,       0.0, 0.0],
            [0.0, kb1 + kb2, 0.0, 0.0],
            [0.0, 0.0,       0.0, 0.0],
            [0.0, 0.0,       0.0, self.ka],
        ], dtype=np.float64)
        self._E43 = np.vstack([np.eye(3), np.zeros((1, 3))])

    def M(self, Y, da):
        ma, mhr, L0 = self.ma, self.mh_rigid, self.L0
        off = (m1 - m2) * Lb / 2 - (mhr + ma) * Y - ma * L0 - ma * da
        return np.array([
            [m1 + m2 + mb + mhr + ma, off, 0.0, 0.0],
            [off, Jb + Jh + (m1 + m2) * Lb ** 2 / 4 + (mhr + ma) * dd ** 2
                  + mhr * Y ** 2 + ma * (Y + L0 + da) ** 2, -(mhr + ma) * dd, -ma * dd],
            [0.0, -(mhr + ma) * dd, mhr + ma, ma],
            [0.0, -ma * dd,         ma,       ma],
        ], dtype=np.float64)

    def deriv(self, x, u_log):
        q, qd = x[:4], x[4:]
        qdd = np.linalg.solve(self.M(x[2], x[3]),
                              self._E43 @ u_log - self._K4 @ q - self._C4 @ qd)
        dx = np.empty(8)
        dx[:4] = qd
        dx[4:] = qdd
        return dx

    def rollout(self, x0_8, u_log, ts, up_sample=1):
        """RK4, u held (ZOH) per output step. Returns (N, 8)."""
        N = len(u_log)
        out = np.empty((N, 8))
        x = np.asarray(x0_8, dtype=np.float64).copy()
        h = ts / up_sample
        deriv = self.deriv
        for k in range(N):
            out[k] = x
            uk = u_log[k]
            for _ in range(up_sample):
                k1 = deriv(x, uk)
                k2 = deriv(x + 0.5 * h * k1, uk)
                k3 = deriv(x + 0.5 * h * k2, uk)
                k4 = deriv(x + h * k3, uk)
                x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return out


def plant_for(mode):
    if mode not in MODE_ABSORBER:
        raise KeyError(
            f'No absorber parameters transcribed for dataset folder {mode!r}. Add them to '
            'MODE_ABSORBER from the generator that produced it; do not guess. The replay gate '
            'catches a wrong value, but only after the integration has run.')
    return Plant8(**MODE_ABSORBER[mode])


def x6_from_x8(x8):
    """[X,Th,Y,da,dX,dTh,dY,vda] -> [X,Th,Y,dX,dTh,dY] (the model's six states)."""
    x8 = np.atleast_2d(x8)
    return np.concatenate([x8[:, 0:3], x8[:, 4:7]], axis=1)


def load_record(name, mode, fs_new=FS_NEW):
    """One record at fs_new using the TRAINING pipeline's own conventions.

    u is a per-hold-interval block mean and y/x_logical are point-sampled, exactly as
    `gantry_dynamic/data.py::_resample_u` / `load_mat_aug` do (D-087). Deviating here would
    make the target differ from what training sees, which is the point of the check.
    """
    dm = loadmat(os.path.join(TRAJ_ROOT, mode, name + '.mat'), squeeze_me=True)
    ts0 = float(dm['dt'])
    D = int(round((1.0 / ts0) / fs_new))
    u0 = np.asarray(dm['u_total'], float)
    n = len(u0) // D
    rec = dict(
        name=name, mode=mode, ts=ts0 * D, D=D, fs=fs_new,
        u=u0[:n * D].reshape(n, D, 3).mean(axis=1),              # block mean (D-087)
        y=np.asarray(dm['y'], float)[::D][:n],                   # STAGE positions
        x_logical=np.asarray(dm['x_logical'], float)[::D][:n],   # gradient() velocities!
    )
    for opt in ('delta_a', 'vdelta_a'):
        if opt in dm:
            rec[opt] = np.asarray(dm[opt], float)[::D][:n]
    rec['u_log'] = (P_np @ rec['u'].T).T                         # stage -> logical force
    rec['Y_op'] = float(rec['y'][0, 2])
    rec['t'] = np.arange(n) * rec['ts']
    return rec


def cache_path(name, mode, fs_new=FS_NEW):
    return os.path.join(CACHE, f'exact_{mode}_{name}_{fs_new}.npz')


def exact_truth(name, mode, fs_new=FS_NEW, cache=True, verbose=False):
    """Exact 8-state truth on the fs_new grid, integrated at FS_ORIG from rest.

    Returns dict with x8 (N,8), x6 (N,6), rec, and `gate` = max |replayed position minus the
    record's own position| per axis.
    """
    rec = load_record(name, mode, fs_new=fs_new)
    path = cache_path(name, mode, fs_new)
    if cache and os.path.exists(path):
        x8 = np.load(path)['x8']
    else:
        hi = load_record(name, mode, fs_new=FS_ORIG)
        # The one exactly-known initial condition: the Simulink integrators start at zero,
        # the Y integrator at Y_op, everything at rest.
        x0 = np.array([0., 0., hi['Y_op'], 0., 0., 0., 0., 0.])
        full = plant_for(mode).rollout(x0, hi['u_log'], hi['ts'], up_sample=1)
        x8 = full[::(FS_ORIG // fs_new)][:len(rec['u'])]
        if cache:
            os.makedirs(CACHE, exist_ok=True)
            np.savez_compressed(path, x8=x8)
    gate = np.abs(x6_from_x8(x8)[:, :3] - rec['x_logical'][:, :3]).max(axis=0)
    if verbose:
        print(f"  {name:<24} replay-vs-record max |dq|: X {gate[0]:.3e} m  "
              f"Th {gate[1]:.3e} rad  Y {gate[2]:.3e} m")
    return dict(name=name, mode=mode, x8=x8, x6=x6_from_x8(x8), rec=rec,
                gate=gate, ts=rec['ts'], fs=fs_new)


# Replay-gate tolerance: an absolute float32 storage floor plus a relative term.
#
# The gate asks ONE question: is this the plant that produced the record (right structure, right
# absorber parameters)? It is not a claim about the accuracy of the velocities; that is bounded
# separately and is three orders tighter (see below).
#
# (1) STORAGE, absolute. `y` and `x_logical` are saved as float32 (`gtd_save_record.m`), so the
#     record's own positions carry up to half an ulp of their own magnitude. Axis-dependent, so
#     it is computed per record rather than tabulated.
# (2) INPUT QUANTISATION, relative. `u_total` is float32 too, and the K = 0 axes integrate its
#     quantisation into a slow position drift no integrator can avoid. MEASURED on
#     V1_standstill_Yp10 by replaying with `u` dithered by +/- half an ulp32: positions moved by
#     [3.1e-09, 1.6e-10, 1.2e-08] m/rad, i.e. a relative 6e-04 of the record's own position std
#     on the worst axis. It is charged as a RELATIVE term because it grows with the excitation:
#     E3_aprbs_above, whose X force is 2.3x V1's, drifts 14x further on Theta.
#
# HEURISTIC 3e-03: above the measured 6e-04 floor, and far below anything the gate must catch.
# Measured on V1 by replaying with a deliberately wrong absorber, as a relative position error:
#     zeta_a 0.05 instead of 0.03   X 4.6e-03  Theta 3.2e-03  Y 1.27
#     ma_frac 0.10 instead of 0.50  X 4.61     Theta 1.2e-01  Y 5.64
#     fa 151 Hz instead of 150      X 3.7e-03  Theta 1.6e-03  Y 3.9e-01
# The correct parameters give Y 9.8e-04, so the Y channel separates right from wrong by a factor
# of 400 even for a 1 Hz error in the absorber frequency.
#
# THE TRUTH ITSELF is bounded by the same dither experiment, not by this gate: the VELOCITIES
# moved by a relative 2.4e-07 under the input dither and 2.5e-06 under a halved RK4 step,
# against the 8.1e-04 relative error of the record's own finite-difference velocities. The truth
# is three orders better than the target it replaces, which is the whole point of D-197.
GATE_REL = 3e-03


def gate_tolerance(res):
    """Per-axis position tolerance: float32 storage floor plus GATE_REL of the record's own std."""
    q = res['rec']['x_logical'][:, :3]
    storage = 4.0 * np.spacing(np.abs(q).max(axis=0).astype(np.float32)).astype(np.float64)
    return storage + GATE_REL * q.std(axis=0)


def gate_replay(res, verbose=True):
    """Does the replay reproduce the record's own positions to the quantisation floor?"""
    tol = gate_tolerance(res)
    ok = bool(np.all(res['gate'] <= tol))
    if verbose:
        print(f"  {res['name']:<24} |dq| {res['gate'][0]:.2e} {res['gate'][1]:.2e} "
              f"{res['gate'][2]:.2e}  tol {tol[0]:.2e} {tol[1]:.2e} {tol[2]:.2e}  "
              f"{'PASS' if ok else 'CHECK'}")
    return ok


def fd_velocity_error(res):
    """How wrong ARE the record's finite-difference velocities? (m/s, rad/s)"""
    v_true = res['x6'][:, 3:]
    v_fd = res['rec']['x_logical'][:, 3:]
    e = v_fd - v_true
    return dict(max_abs=np.abs(e).max(axis=0), rms=e.std(axis=0),
                rms_true=v_true.std(axis=0),
                rel=e.std(axis=0) / np.maximum(v_true.std(axis=0), 1e-30),
                at_k0=v_fd[0] - v_true[0], v_true_k0=v_true[0], v_fd_k0=v_fd[0])


def selfcheck(seed=0, n=64):
    """Check 1 of D-197: reproduce common/oracle._deriv at ITS parameters, to machine precision."""
    from common import oracle                                   # noqa: E402
    p = Plant8(ma_frac=oracle.MA_FRAC, fa=oracle.FA, zeta_a=oracle.ZETA_A, L0=oracle.L0)
    for nm, a, b in (('ma', p.ma, oracle.MA), ('mh_rigid', p.mh_rigid, oracle.MH_RIGID),
                     ('ka', p.ka, oracle.KA), ('ca', p.ca, oracle.CA)):
        assert a == b, f'{nm}: {a!r} != {b!r}'
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n):
        x = rng.normal(size=8) * np.array([0.3, 0.05, 0.4, 1e-6, 1., 1., 1., 1e-3])
        u = rng.normal(size=3) * np.array([500., 300., 400.])
        a, b = p.deriv(x, u), oracle._deriv(x, u)
        worst = max(worst, float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30)))
    print(f'selfcheck vs common/oracle._deriv (ma_frac={oracle.MA_FRAC}, zeta_a={oracle.ZETA_A}): '
          f'worst relative difference over {n} random states = {worst:.3e}')
    assert worst == 0.0, 'parameterised EOM does not reproduce the verified oracle exactly'
    return worst


if __name__ == '__main__':
    import argparse
    import time

    ap = argparse.ArgumentParser(description='Cache the exact 8-state truth for a dataset.')
    ap.add_argument('--mode', default='augmentation_ma50_b140-230_a6_z03')
    ap.add_argument('--records', nargs='*', default=None,
                    help='record stems; default = every .mat in the folder')
    ap.add_argument('--fs-new', type=int, default=FS_NEW)
    args = ap.parse_args()

    selfcheck()
    names = args.records
    if not names:
        names = sorted(f[:-4] for f in os.listdir(os.path.join(TRAJ_ROOT, args.mode))
                       if f.endswith('.mat'))
    print(f'\nCaching the exact 8-state truth for {len(names)} records of {args.mode} '
          f'({FS_ORIG // 1000} kHz RK4 from the rest IC, decimated to '
          f'{args.fs_new // 1000} kHz)\n')
    fails = []
    for nm in names:
        t0 = time.time()
        cached = os.path.exists(cache_path(nm, args.mode, args.fs_new))
        r = exact_truth(nm, args.mode, fs_new=args.fs_new)
        fd = fd_velocity_error(r)
        ok = gate_replay(r, verbose=False)
        tol = gate_tolerance(r)
        fails += [] if ok else [nm]
        print(f"  {nm:<24} |dq| {r['gate'][0]:.2e} {r['gate'][1]:.2e} {r['gate'][2]:.2e}  "
              f"tol {tol[0]:.2e} {tol[1]:.2e} {tol[2]:.2e}  {'PASS' if ok else 'CHECK'}  "
              f"FD vel rel {fd['rel'][0]:.1e}/{fd['rel'][1]:.1e}/{fd['rel'][2]:.1e}  "
              f"k0 {np.abs(fd['at_k0']).max():.1e}  "
              f"[{'cache' if cached else '%.0f s' % (time.time() - t0)}]")
    print('\nAll records passed the replay gate.' if not fails
          else f'\nGATE FAILED for: {fails}')
