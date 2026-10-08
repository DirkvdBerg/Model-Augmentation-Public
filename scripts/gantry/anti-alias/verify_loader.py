"""Verify the D-232 loader change (data.py::_record_at_fs_new) against independent formulas.

V1 legacy 'augmentation' mode: bitwise as before (old formulas inline), all three loaders.
V2 noise-free thesis mode: bitwise the old formulas plus the 5 ms trim at both ends, all three loaders, with and
   without the smoke cut.
V3 noisy thesis mode: bitwise the tested rule (measure_aliasing.py 'butter_err') plus the trim; the three loaders
   give the same samples; x_logical lines up with y (a one-sample shift would show).
V4 the twins share one sample grid (float64, as the thesis runs load): noisy y minus noise-free y at the same
   loader index is noise-sized (nm); a one-sample offset would be um during motion.
V5 config.json records Y_DECIMATION and RECORD_TRIM per mode; the config coverage check still passes at import.
"""
__project_origin__ = "added"

import os
import sys
from dataclasses import replace

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'scripts', 'gantry'))
sys.path.insert(0, REPO)
from gantry_dynamic.config import RunConfig, config_json_dict                   # noqa: E402  (runs the coverage check)
from gantry_dynamic import data as G                                            # noqa: E402
from gantry_dynamic.obc_gantry import _load_record_float64                      # noqa: E402

fails = []
K = 20                                                          # 5 ms at 4 kHz, written out independently
P = np.array([[1.0, 1.0, 0.0], [0.725 / 2, -0.725 / 2, 0.0], [0.0, 0.0, 1.0]])
SOS = butter(8, 1600.0, fs=20000.0, output='sos')


def check(name, ok):
    print('  %-78s %s' % (name, 'PASS' if ok else 'FAIL'))
    if not ok:
        fails.append(name)


def old(d, cfg, k, fields=(), dtype=None):
    """The pre-D-232 formulas, then the trim k at both ends, then the smoke cut."""
    D = cfg.d
    u = G._resample_u(d['u_total'], cfg)
    N = len(u)
    arrs = [u, d['y'][::D][:N]] + [np.asarray(d[f])[::D][:N] for f in fields]
    arrs = [a[k:N - k] if k else a for a in arrs]
    if cfg.truncate_s is not None:
        arrs = [a[:int(cfg.truncate_s * cfg.fs_new_hz)] for a in arrs]
    return [a.astype(dtype) if dtype is not None else a for a in arrs]


def old_aug(d, cfg, k):
    u, y, xl, da = old(d, cfg, k, ('x_logical', 'delta_a'), cfg.dtype_np)
    vda = np.zeros_like(da)
    vda[1:] = (da[1:] - da[:-1]) * cfg.fs_new_hz
    vda[0] = vda[1]
    return u, y, xl, np.stack([da, vda], axis=1)


def old_obc(d, cfg, k):
    u, y, xl, da, vda = old(d, cfg, k, ('x_logical', 'delta_a', 'vdelta_a'), np.float64)
    return u, y, xl, np.stack([da, vda], 1)


def same(a, b):
    return all(x.shape == y.shape and x.dtype == y.dtype and np.array_equal(x, y) for x, y in zip(a, b))


def formulas(mode, files, k, truncate=None, n_full=4):
    cfg = replace(RunConfig(), mode=mode, truncate_s=truncate)
    print('\n%s mode %s, trim %d, truncate_s %s, %d records' % ('V1' if k == 0 else 'V2', mode, k, truncate, len(files)))
    for i, f in enumerate(files):
        d = loadmat(os.path.join(G.traj_dir(cfg), f), squeeze_me=True)
        sd = G.load_traj(f, cfg)
        ok = same(old(d, cfg, k, dtype=cfg.dtype_np), (np.asarray(sd.u), np.asarray(sd.y)))
        if i < n_full:
            ok = ok and same(old_aug(d, cfg, k), G.load_mat_aug(f, cfg))
            if 'vdelta_a' in d:
                ok = ok and same(old_obc(d, cfg, k), _load_record_float64(f, cfg))
        check('%s load_traj%s' % (f, ' / load_mat_aug / obc' if i < n_full else ''), ok)


# V1 legacy, V2 noise-free thesis (every rollout record; the smoke cut on two)
formulas('augmentation', G.TRAIN_FILES[:3] + G.VAL_FILES[:1], 0)
cfg_nf = replace(RunConfig(), mode='thesis_taf_noisefree')
tr, va, te = G.record_files(cfg_nf)
formulas('thesis_taf_noisefree', tr + va + te, K)
formulas('thesis_taf_noisefree', tr[:2], K, truncate=1.0, n_full=2)

# V3 noisy: exact rule, three loaders, alignment; V4 twins on one grid
cfg = replace(RunConfig(), mode='thesis_taf_noisy')
tr, va, te = G.record_files(cfg)
print('\nV3 exact rule + alignment, V4 twin grid; mode thesis_taf_noisy, %d records' % len(tr + va + te))
worst, worst_twin = 0.0, 0.0
for f in tr + va + te:
    d = loadmat(os.path.join(G.traj_dir(cfg), f), squeeze_me=True)
    r = d['r_sim']
    N = d['u_total'].shape[0] // 5
    y_ref = (r[::5] + sosfiltfilt(SOS, d['y'] - r, axis=0)[::5])[:N][K:N - K].astype(np.float32)
    u_ref = d['u_total'][:N * 5].reshape(N, 5, 3).mean(axis=1)[K:N - K].astype(np.float32)
    sd = G.load_traj(f, cfg)
    ok3 = np.array_equal(np.asarray(sd.y), y_ref) and np.array_equal(np.asarray(sd.u), u_ref) \
        and len(sd.u) == N - 2 * K
    u2, y2, xl, xa = G.load_mat_aug(f, cfg)
    u3, y3, xl3, xa3 = _load_record_float64(f, cfg)
    ok3 = ok3 and (np.array_equal(u2, np.asarray(sd.u)) and np.array_equal(y2, np.asarray(sd.y))
                   and np.array_equal(u3.astype(np.float32), np.asarray(sd.u))
                   and np.array_equal(y3.astype(np.float32), np.asarray(sd.y)) and len(xl3) == len(sd.u))
    yt = d['y_true'][::5][:N][K:N - K]
    q_stage = xl3[:, :3] @ P                                    # y = P^T q (oracle.py), row form
    e0 = np.max(np.abs(q_stage - yt))
    e1 = np.max(np.abs(q_stage[1:] - yt[:-1]))
    worst = max(worst, e0)
    check('%s V3 exact rule, same samples in 3 loaders, aligned %.0e m (shift: %.0e m)' % (f, e0, e1),
          ok3 and e0 < 1e-12 and (e1 > 1e3 * e0 or e1 < 1e-9))
    # float64 as the thesis runs load (use_f64=True): float32 quantises Y = 0.3 m to 30 nm steps
    y_nf = np.asarray(G.load_traj(f, replace(cfg_nf, use_f64=True)).y)
    ynz = np.asarray(G.load_traj(f, replace(cfg, use_f64=True)).y)
    dn = np.sqrt(np.mean((ynz - y_nf) ** 2, axis=0))
    ds = np.sqrt(np.mean((ynz[1:] - y_nf[:-1]) ** 2, axis=0))
    worst_twin = max(worst_twin, float(dn.max()))
    check('%s V4 twins: noisy - noise-free rms %s nm (shifted by one: %s nm)'
          % (f, np.array2string(1e9 * dn, precision=1), np.array2string(1e9 * ds, precision=0)),
          len(y_nf) == len(ynz) and dn.max() < 1e-7)

print('\nV5 config.json keys')
for mode, rule, trim in (('thesis_taf_noisy', 'r + zero-phase Butterworth 8 at 1600 Hz on y - r, then [::5]', K),
                         ('thesis_taf_noisefree', 'point sampling [::5]', K),
                         ('augmentation', 'point sampling [::5]', 0)):
    j = config_json_dict(replace(RunConfig(), mode=mode))
    check('%s: Y_DECIMATION %r, RECORD_TRIM %r' % (mode, j.get('Y_DECIMATION'), j.get('RECORD_TRIM')),
          j.get('Y_DECIMATION') == rule and j.get('RECORD_TRIM') == trim)

print('\n%s (%d failures); worst x_logical misalignment %.1e m; worst twin difference %.1f nm'
      % ('ALL PASS' if not fails else 'FAILURES', len(fails), worst, 1e9 * worst_twin))
for f in fails:
    print('  FAIL', f)
