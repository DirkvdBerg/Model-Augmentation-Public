"""Data loading (the manifest split of one noise entry), resampling and normalization."""
__project_origin__ = "added"

import csv
import os
from dataclasses import dataclass

import numpy as np
import deepSI
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt

from model_augmentation.systems.gantry_ss import Cd, Dd, P

from .config import (RunConfig, AA_ORDER, AA_FC_HZ,
                     record_trim, y_decimation)

# D-221: the thesis data sets below cfg.data_root, one folder per noise entry
DATASETS = {'lowpass': 'Coulomb-tanh-and-MSD-lowpass-noise', 'noisefree': 'Coulomb-tanh-and-MSD-noise-free'}
_THESIS_SPLIT = {'training': 'train', 'validation': 'val', 'test-I': 'test', 'test-E': 'test'}


def record_files(cfg: RunConfig):
    """(train, val, test) file names for cfg.mode, relative to traj_dir(cfg), in manifest order."""
    out = {'train': [], 'val': [], 'test': []}
    with open(os.path.join(traj_dir(cfg), 'MANIFEST.csv'), newline='') as fh:
        for row in csv.DictReader(fh):
            split = _THESIS_SPLIT.get(row['block'])
            # D-274: only K1 records (thesis: every record is generated with K1); drops E5 (K2-g+),
            # which was loaded but never scored (D-227).
            if split is not None and row['controller'] == 'K1':
                out[split].append('%s/%s.mat' % (row['folder'], row['file']))
    return out['train'], out['val'], out['test']


def _register_controller(d, filename):
    """Hand the record's stored controller (meta.controller, D-221) to the closed-loop bank."""
    from .controller import register_stored_controller
    ctl = d['meta']['controller'].item()
    get = lambda k: ctl[k].item() if hasattr(ctl[k], 'item') else ctl[k]
    register_stored_controller(filename, str(get('name')), float(get('design_Y')),
                               float(get('gain')), float(get('fbw_Hz')))


@dataclass
class DataBundle:
    train_list: list
    val_list: list
    test_list: list
    train_data: object
    val_ckpt_data: object
    train_files: list
    val_files: list
    test_files: list
    e_train: list
    e_val: list
    e_test: list


@dataclass
class Norm:
    x_mean: object
    std_x: object
    std_u: object
    u_mean: object
    ystd: object
    y0: object
    Cd_norm: object
    Dd_np: object


def traj_dir(cfg: RunConfig) -> str:
    return os.path.join(cfg.data_root, DATASETS[cfg.noise])


def _load_u(d):
    """Return plant input u_total [F_X1, F_X2, F_Y] from the generated mat file."""
    return d['u_total']


def _resample_u(u, cfg: RunConfig):
    """Downsample the plant force FS_ORIG -> FS_NEW by per-hold-interval block mean.

    # THEORY: u_total is ZOH at FS_ORIG (discrete 20 kHz controller), so the exact
    # input for a ZOH model at FS_NEW is the mean force over each TS_NEW hold
    # interval (impulse equivalence). Point sampling u[::D] leaves a nonzero-mean
    # force error that the K=0 axes integrate into a permanent offset tau*dv
    # (tau=m/c): V1 open-loop settled at Y -3.5e-4 m / X +6e-5 m; block mean
    # collapses this to ~3e-9 / 3e-8 m (diag_openloop_x0.py, D-087).
    # NB (D-099): this is a DC/area-consistency effect, NOT anti-aliasing --
    # u_total is band-limited well below the 2 kHz Nyquist (frac_above ~4e-14,
    # diag_downsample_spectra.py), so there is no HF energy to fold.
    """
    D = cfg.d
    n = u.shape[0] // D
    return u[:n * D].astype(np.float64).reshape(n, D, u.shape[1]).mean(axis=1)


def _decimate_y(d, cfg: RunConfig):
    """y at FS_NEW: the D-232 anti-alias rule on the lowpass entry, point sampling otherwise."""
    D = cfg.d
    if cfg.noise != 'lowpass':
        return d['y'][::D]
    r = d['r_sim']
    sos = butter(AA_ORDER, AA_FC_HZ, fs=cfg.fs_orig, output='sos')
    return r[::D] + sosfiltfilt(sos, d['y'] - r, axis=0)[::D]


def _record_at_fs_new(d, cfg: RunConfig, fields=()):
    """[u, y, *fields] of one record at FS_NEW, all on the same samples."""
    D = cfg.d
    u = _resample_u(_load_u(d), cfg)
    N = len(u)
    arrs = [u, _decimate_y(d, cfg)[:N]] + [np.asarray(d[k])[::D][:N] for k in fields]
    k = record_trim(cfg)
    arrs = [a[k:N - k] for a in arrs]
    if cfg.truncate_s is not None:
        arrs = [a[:int(cfg.truncate_s * cfg.fs_new)] for a in arrs]
    return arrs


def _load_record(filename, cfg: RunConfig, folder=None):
    """(System_data, e) of one record from ONE read of its file; e = y - r at FS_NEW (float64),
    on the same samples as y. `folder` overrides traj_dir(cfg), for the evaluation-only twins (D-258)."""
    d = loadmat(os.path.join(folder or traj_dir(cfg), filename), squeeze_me=True)
    _register_controller(d, filename)
    u, y, r = _record_at_fs_new(d, cfg, ('r_sim',))
    e = np.asarray(y, np.float64) - np.asarray(r, np.float64)
    sd = deepSI.System_data(u=u.astype(cfg.dtype_np), y=y.astype(cfg.dtype_np), dt=cfg.ts_new)
    return sd, e


def servo_sigma(e_list, k0):
    """sigma_e per record and axis (D-262): population std of the servo error e = y - r over the
    scored samples k0..N-1, shape (n_records, ny). THE one definition, used by the loss weights,
    the selection, the black box and the evaluation."""
    # THEORY: Beintema 2023 Eq. (43) normalises by the sample standard deviation of the output;
    # here the output is the servo error (thesis eq:setup_rms, D-256)
    s = np.stack([np.std(np.asarray(e, np.float64)[k0:], axis=0) for e in e_list])
    if not (np.all(np.isfinite(s)) and np.all(s > 0)):
        raise ValueError('servo error standard deviation must be finite and positive, got %s' % s)
    return s


def load_datasets(cfg: RunConfig) -> DataBundle:
    """Load the train/val/test records of the manifest split."""
    print(f'Data dir ({cfg.mode}): {traj_dir(cfg)}')
    print(f'y decimation (D-232): {y_decimation(cfg)}; {record_trim(cfg)} samples dropped at both '
          f'record ends (u, y and every field)')
    train_files, val_files, test_files = record_files(cfg)

    def _split(files):
        pairs = [_load_record(f, cfg) for f in files]
        return [p[0] for p in pairs], [p[1] for p in pairs]

    (train_list, e_train), (val_list, e_val), (test_list, e_test) = (
        _split(train_files), _split(val_files), _split(test_files))

    train_data    = deepSI.System_data_list(train_list)
    val_ckpt_data = deepSI.System_data_list(val_list)

    print(f'Loaded {len(train_list)} train, {len(val_list)} val, {len(test_list)} test trajectories')
    for i, (f, t) in enumerate(zip(train_files, train_list)):
        print(f'  T{i+1}: {t.u.shape[0]} samples  ({f})')

    return DataBundle(
        train_list=train_list, val_list=val_list, test_list=test_list,
        train_data=train_data, val_ckpt_data=val_ckpt_data,
        train_files=train_files, val_files=val_files, test_files=test_files,
        e_train=e_train, e_val=e_val, e_test=e_test,
    )


def compute_normalization(cfg: RunConfig, data: DataBundle) -> Norm:
    """All NX_PHYS-dimensional normalization constants (independent of NX_ANN)."""
    u_all = np.concatenate([t.u for t in data.train_list])
    y_all = np.concatenate([t.y for t in data.train_list])

    fs = 1.0 / data.train_list[0].dt
    P_inv_T = np.linalg.inv(P.numpy().T).astype(cfg.dtype_np)
    x_logical_list = []
    for t in data.train_list:
        pos_logical = (P_inv_T @ t.y.T).T
        # THEORY: var of a first difference of white noise = 2*sigma_n^2, so sigma_v = sigma_n*sqrt(2)*fs
        vel_logical = np.diff(pos_logical, axis=0) * fs
        vel_logical = np.vstack([vel_logical[:1], vel_logical])
        x_logical_list.append(np.hstack([pos_logical, vel_logical]))
    x_all = np.concatenate(x_logical_list)

    x_mean = x_all.mean(axis=0).reshape(cfg.nx_phys, 1)
    std_x  = x_all.std(axis=0).reshape(cfg.nx_phys, 1) + 1e-8
    std_u  = u_all.std(axis=0).reshape(cfg.nu, 1) + 1e-8
    u_mean = u_all.mean(axis=0).reshape(cfg.nu, 1)
    ystd   = y_all.std(axis=0) + 1e-8
    y0     = y_all.mean(axis=0)

    Cd_norm = Cd.numpy() * std_x.flatten()[None, :] / ystd[:, None]
    Dd_np   = Dd.numpy()

    return Norm(
        x_mean=x_mean, std_x=std_x, std_u=std_u, u_mean=u_mean,
        ystd=ystd, y0=y0, Cd_norm=Cd_norm, Dd_np=Dd_np,
    )
