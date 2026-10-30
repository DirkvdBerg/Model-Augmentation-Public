"""Data loading (the manifest split of one data set), resampling to the training rate and normalisation.

Conventions: samples are rows, so q = P^-T y is written y @ P^-T.T; superscript n = normalised; x_mean, std_x,
u_mean, std_u, y0, ystd are mu_x, sigma_x, mu_u, sigma_u, mu_y, sigma_y.
"""
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

# the thesis data sets below cfg.data_root, one folder per noise setting
DATASETS = {'lowpass': 'Coulomb-tanh-and-MSD-lowpass-noise', 'noisefree': 'Coulomb-tanh-and-MSD-noise-free'}
# manifest block -> split
_THESIS_SPLIT = {'training': 'train', 'validation': 'val', 'test-I': 'test', 'test-E': 'test'}


def record_files(cfg: RunConfig):
    """(train, val, test) file names relative to traj_dir(cfg), in manifest order."""
    out = {'train': [], 'val': [], 'test': []}
    with open(os.path.join(traj_dir(cfg), 'MANIFEST.csv'), newline='') as fh:
        for row in csv.DictReader(fh):
            split = _THESIS_SPLIT.get(row['block'])
            # K1 records only: every thesis record is generated with the controller K1
            if split is not None and row['controller'] == 'K1':
                out[split].append('%s/%s.mat' % (row['folder'], row['file']))
    return out['train'], out['val'], out['test']


def _register_controller(d, filename):
    """Register the record's stored controller (meta.controller) for the closed-loop bank."""
    from .controller import register_stored_controller
    ctl = d['meta']['controller'].item()
    get = lambda k: ctl[k].item() if hasattr(ctl[k], 'item') else ctl[k]
    register_stored_controller(filename, str(get('name')), float(get('design_Y')),
                               float(get('gain')), float(get('fbw_Hz')))


@dataclass
class DataBundle:
    """The records of the three splits: System_data at FS_NEW, file names and servo errors e = y - r."""
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
    """Normalisation constants of the physical states, inputs and outputs, and the output map C^n."""
    x_mean: object
    std_x: object
    std_u: object
    u_mean: object
    ystd: object
    y0: object
    Cd_norm: object
    Dd_np: object


def traj_dir(cfg: RunConfig) -> str:
    """Folder of the data set of cfg.noise."""
    return os.path.join(cfg.data_root, DATASETS[cfg.noise])


def _load_u(d):
    """The plant input u_total = [F_X1, F_X2, F_Y] of a record."""
    return d['u_total']


def _resample_u(u, cfg: RunConfig):
    """Actuator force at the model rate.

    u_act,k = mean of the D samples of u_act in [k T_s, (k + 1) T_s), D = fs_orig / fs_new: the force held over
    each sample.

    Args:
        u: (N_sim, 3) u_act at f_sim, newtons.
        cfg: run configuration; cfg.d = D.
    Returns:
        (N, 3) u_act at 1 / T_s, newtons.
    """
    # THEORY: the block mean preserves the force integral over each model interval
    D = cfg.d
    n = u.shape[0] // D
    return u[:n * D].astype(np.float64).reshape(n, D, u.shape[1]).mean(axis=1)


def _decimate_y(d, cfg: RunConfig):
    """Measured positions at the model rate.

    y_k = r(k T_s) + B(y - r)(k T_s), B = forward-backward Butterworth (order 8, 1.6 kHz), on the lowpass data set;
    point sampling y_k = y(k T_s) otherwise. Only the servo error is filtered.

    Args:
        d: the record; cfg: run configuration.
    Returns:
        (N, 3) y at 1 / T_s, physical.
    """
    D = cfg.d
    if cfg.noise != 'lowpass':
        return d['y'][::D]
    r = d['r_sim']
    sos = butter(AA_ORDER, AA_FC_HZ, fs=cfg.fs_orig, output='sos')
    # Error-only filtering: y_k = r_k + [B(y - r)]_{k D}; r is unfiltered.
    return r[::D] + sosfiltfilt(sos, d['y'] - r, axis=0)[::D]


def _record_at_fs_new(d, cfg: RunConfig, fields=()):
    """[u, y, *fields] of one record at FS_NEW on the same samples, trimmed at both ends."""
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
    """Load a record once and return (System_data, servo error), using folder when supplied."""
    d = loadmat(os.path.join(folder or traj_dir(cfg), filename), squeeze_me=True)
    _register_controller(d, filename)
    u, y, r = _record_at_fs_new(d, cfg, ('r_sim',))
    e = np.asarray(y, np.float64) - np.asarray(r, np.float64)
    sd = deepSI.System_data(u=u.astype(cfg.dtype_np), y=y.astype(cfg.dtype_np), dt=cfg.ts_new)
    return sd, e


def servo_sigma(e_list, k0):
    """Servo-error scale sigma_e.

    sigma_e,j = population standard deviation of e_j = y_j - r_j over k = k0 .. N - 1; it normalises the loss
    (W_r) and the validation NRMS_e.

    Args:
        e_list: per record (N, 3) servo errors, metres; k0: first scored sample (n + 1).
    Returns:
        (n_records, 3) sigma_e.
    """
    # THEORY: the NRMS normalisation by a standard deviation (Beintema et al.), here that of the servo error
    # sigma_e,j = (mean over k >= k0 of (e_j,k - mean e_j)^2)^(1/2).
    s = np.stack([np.std(np.asarray(e, np.float64)[k0:], axis=0) for e in e_list])
    if not (np.all(np.isfinite(s)) and np.all(s > 0)):
        raise ValueError('servo error standard deviation must be finite and positive, got %s; check the '
                         'records for a constant or missing servo error' % s)
    return s


def load_datasets(cfg: RunConfig) -> DataBundle:
    """Load the train, val and test records of the manifest split."""
    print(f'Data dir ({cfg.mode}): {traj_dir(cfg)}')
    print(f'y decimation: {y_decimation(cfg)}; {record_trim(cfg)} samples dropped at both '
          f'record ends (u, y and every field)')
    train_files, val_files, test_files = record_files(cfg)

    def _split(files):
        """(System_data, servo error) lists of the records of one split."""
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
    """Normalisation of the augmented model.

    mu_i, sigma_i of x_tilde = (q, qdot), u_act and y over the training records, and C^n = T_y P^T [I 0] T_x^-1; the
    position means equal P^-T mu_y, so the output map has no offset.

    Args:
        cfg: run configuration.
        data: the loaded records.
    Returns:
        Norm: mu_x, sigma_x (6, 1) of the physical q, qdot; mu_u, sigma_u (3, 1) of the actuator force, newtons;
        mu_y, sigma_y (3,) of the measured positions, metres; C^n (3, 6) and D (3, 3), maps between normalised
        coordinates.
    """
    # 1. Physical states of the training records.
    u_all = np.concatenate([t.u for t in data.train_list])
    y_all = np.concatenate([t.y for t in data.train_list])

    fs = 1.0 / data.train_list[0].dt
    P_inv_T = np.linalg.inv(P.numpy().astype(cfg.dtype_np).T)
    x_logical_list = []
    for t in data.train_list:
        # Positions: q = P^-T y (so the position means are P^-T mu_y).
        pos_logical = (P_inv_T @ t.y.T).T
        # Velocities: qdot_k = (q_k - q_{k-1}) / T_s for k >= 1, qdot_0 = qdot_1.
        # THEORY: var of a first difference of white noise = 2*sigma_n^2, so sigma_v = sigma_n*sqrt(2)*fs
        vel_logical = np.diff(pos_logical, axis=0) * fs
        vel_logical = np.vstack([vel_logical[:1], vel_logical])
        x_logical_list.append(np.hstack([pos_logical, vel_logical]))
    x_all = np.concatenate(x_logical_list)

    # 2. Statistics.
    # mu_i = mean(i), sigma_i = (mean (i - mu_i)^2)^(1/2) + 1e-8 (keeps T_i finite), i in {x_tilde, u_act, y}.
    x_mean = x_all.mean(axis=0).reshape(cfg.nx_phys, 1)
    std_x  = x_all.std(axis=0).reshape(cfg.nx_phys, 1) + 1e-8
    std_u  = u_all.std(axis=0).reshape(cfg.nu, 1) + 1e-8
    u_mean = u_all.mean(axis=0).reshape(cfg.nu, 1)
    ystd   = y_all.std(axis=0) + 1e-8
    y0     = y_all.mean(axis=0)

    # 3. Output map: C^n = T_y P^T [I 0] T_x^-1, with no offset since P^T mu_q = mu_y.
    Cd_norm = Cd.numpy() * std_x.flatten()[None, :] / ystd[:, None]
    Dd_np   = Dd.numpy()

    return Norm(
        x_mean=x_mean, std_x=std_x, std_u=std_u, u_mean=u_mean,
        ystd=ystd, y0=y0, Cd_norm=Cd_norm, Dd_np=Dd_np,
    )
