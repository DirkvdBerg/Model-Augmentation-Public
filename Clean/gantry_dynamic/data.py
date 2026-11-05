"""Data loading, resampling to the training rate and normalisation.

y = (X1, X2, Y): measured positions of the two X actuators and of the payload along the cross-arm [m].
q = P^-T y = (X, Theta, Y): generalised coordinates, the cross-arm position (X1 + X2)/2 [m], its yaw angle
(X1 - X2)/L_b [rad] and Y [m]. Samples are rows, so q = y @ P^-T.T.
Normalisation per channel: x^n = (x - mu) / sigma. x_mean and std_x hold the means and standard deviations of the
physical states (q, qdot). u_mean and std_u hold those of the actuator forces; y0 and ystd hold those of the measured
positions.
"""
__project_origin__ = "added"

import csv
import os
from dataclasses import dataclass

import numpy as np
import deepSI
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt

from model_augmentation.systems.gantry_ss import Cd, Dd, P_INV_T

from .config import RunConfig

# data set folder per noise setting; the true system is the baseline plus tanh Coulomb friction and a hidden absorber
DATASETS = {'noisy': 'Coulomb-tanh-and-MSD-lowpass-noise', 'noisefree': 'Coulomb-tanh-and-MSD-noise-free'}
# the record list of the data set (MANIFEST.csv) gives each record a block; this maps the block to its split.
# test-I and test-E (interpolation, extrapolation) both go to test
_SPLIT_OF_BLOCK ={'training': 'train', 'validation': 'val', 'test-I': 'test', 'test-E': 'test'}


def record_files(cfg: RunConfig):
    """(train, val, test) record files as paths inside the data set folder, e.g. 'Training/TR-S1_r1.mat', in the
    order of MANIFEST.csv."""
    out = {'train': [], 'val': [], 'test': []}
    with open(os.path.join(dataset_dir(cfg), 'MANIFEST.csv'), newline='') as fh:
        for row in csv.DictReader(fh):
            split = _SPLIT_OF_BLOCK.get(row['block'])
            # K1 records only; the data set also has a record under a changed controller (K2), which the thesis does
            # not use
            if split is not None and row['controller'] == 'K1':
                out[split].append('%s/%s.mat' % (row['folder'], row['file']))
    return out['train'], out['val'], out['test']


def record_name(path):
    """Name of a record without folder and realisation, e.g. 'Training/TR-S1_r1.mat' gives 'TR-S1'."""
    return os.path.splitext(os.path.basename(path))[0].split('_')[0]


def _controller_spec(record):
    """The record's stored controller (meta.controller): (name, design Y [m], gain, bandwidth [Hz])."""
    controller = record['meta']['controller'].item()
    # loadmat wraps each MATLAB value in a numpy array; .item() takes the plain value out
    name, design_y, gain, fbw_hz = (np.asarray(controller[k]).item() for k in ('name', 'design_Y', 'gain', 'fbw_Hz'))
    return str(name), float(design_y), float(gain), float(fbw_hz)


@dataclass
class Datasets:
    """Result of load_datasets: the training, validation and test records at fs_new. Each record is a deepSI
    System_data with its forces u and positions y. train_files, val_files and test_files hold the record file names;
    e_train, e_val and e_test hold the servo errors e = y - r, (N, 3) [m], in the same order. controllers maps each
    record file name to its stored controller, to build the closed-loop controller bank."""
    train_list: list
    val_list: list
    test_list: list
    train_files: list
    val_files: list
    test_files: list
    e_train: list
    e_val: list
    e_test: list
    controllers: dict


@dataclass
class Norm:
    """Result of compute_normalization: per channel mean and standard deviation of the training records."""
    x_mean: object   # physical states (q, qdot), (6, 1)
    std_x: object    # physical states (q, qdot), (6, 1)
    std_u: object    # actuator forces, (3, 1)
    u_mean: object   # actuator forces, (3, 1)
    ystd: object     # measured positions, (3,)
    y0: object       # measured positions, (3,)
    Cd_norm: object  # output map from normalised physical states to normalised measured positions
    Dd_np: object    # zero matrix: the force has no direct path to the measured positions


def dataset_dir(cfg: RunConfig) -> str:
    """Folder of the run's data set: data_root plus DATASETS[cfg.noise], the noisy or the noise-free records. A noisy
    run also needs the noise-free records for its noise floor; the grey-box evaluation finds each record's noise-free
    twin through the pair_file column of MANIFEST.csv."""
    return os.path.join(cfg.data_root, DATASETS[cfg.noise])


def _force_at_fs_new(u, cfg: RunConfig):
    """Actuator forces at fs_new: each sample is the average of the decimation samples at fs_orig in one model step,
    so the model holds the average force over that step.

    Args:
        u: (N_sim, 3) force at fs_orig [N]: the actuator force u_total, or the multisine force u^ms for the black box.
        cfg: run configuration.
    Returns:
        (N, 3) the same force at fs_new [N].
    """
    decimation = cfg.decimation
    n_blocks = u.shape[0] // decimation
    return u[:n_blocks * decimation].astype(cfg.dtype_np).reshape(n_blocks, decimation, u.shape[1]).mean(axis=1)


def _positions_at_fs_new(rec, cfg: RunConfig):
    """Measured positions at fs_new: every decimation-th sample of y. For the noisy data set, the servo error y - r is
    first filtered with the zero-phase anti-aliasing filter and the reference r is added back.

    Args:
        rec: record loaded from its .mat file.
        cfg: run configuration.
    Returns:
        (N, 3) y at fs_new [m].
    """
    decimation = cfg.decimation
    # the anti-alias filter is for the measurement noise, a third of whose variance lies above 2.5 kHz; noise-free data
    # are only decimated
    if cfg.noise != 'noisy':
        return rec['y'][::decimation]
    r = rec['r_sim']
    # Butterworth low-pass; sosfiltfilt runs it forward and backward, so the filtered error has no delay
    anti_alias_filter = butter(cfg.aa_order, cfg.aa_fc_hz, fs=cfg.fs_orig, output='sos')
    return r[::decimation] + sosfiltfilt(anti_alias_filter, rec['y'] - r, axis=0)[::decimation]


def record_at_fs_new(rec, cfg: RunConfig, reference=False):
    """The record at fs_new: forces u and positions y, plus the reference r when reference=True, on the same samples.
    record_trim samples (aa_trim_s at fs_new) are dropped at both ends, and the record is cut to truncate_s seconds
    if set."""
    # u_total = [F_X1, F_X2, F_Y], the plant input
    u = _force_at_fs_new(rec['u_total'], cfg)
    n_samples = len(u)
    # the block mean drops an incomplete last block, so y and r are cut to the length of u
    signals = [u, _positions_at_fs_new(rec, cfg)[:n_samples]]
    if reference:
        signals.append(np.asarray(rec['r_sim'])[::cfg.decimation][:n_samples])
    # drop the anti-aliasing filter's edge transient at both ends
    trim_samples = cfg.record_trim
    signals = [a[trim_samples:n_samples - trim_samples] for a in signals]
    # truncate_s keeps only the start of each record
    if cfg.truncate_s is not None:
        signals = [a[:round(cfg.truncate_s * cfg.fs_new)] for a in signals]
    return signals


def load_record(filename, cfg: RunConfig, folder=None):
    """Load one record file at fs_new. Returns its deepSI System_data (forces u, positions y), its servo error
    e = y - r and its stored controller. filename is relative to folder, by default the run's data set folder;
    evaluation passes data_root to load a record's noise-free twin."""
    record = loadmat(os.path.join(folder or dataset_dir(cfg), filename), squeeze_me=True)
    u, y, r = record_at_fs_new(record, cfg, reference=True)
    e = np.asarray(y, np.float64) - np.asarray(r, np.float64)
    system_data = deepSI.System_data(u=u.astype(cfg.dtype_np), y=y.astype(cfg.dtype_np), dt=cfg.ts_new)
    return system_data, e, _controller_spec(record)


def servo_sigma(e_list, k0):
    """Standard deviation of each record's servo error, per output channel, from sample k0 onward; the first k0
    samples are the encoder's input.

    This scale, sigma_e, normalises the training loss and NRMS_e in validation and evaluation.

    Args:
        e_list: per record (N, 3) servo errors [m].
        k0: first scored sample, the first with a full encoder window.
    Returns:
        (n_records, 3) sigma_e [m].
    """
    sigma_e = np.stack([np.std(np.asarray(e, np.float64)[k0:], axis=0) for e in e_list])
    if not (np.all(np.isfinite(sigma_e)) and np.all(sigma_e > 0)):
        raise ValueError('sigma_e divides the loss and NRMS_e, so it must be finite and positive, got %s' % sigma_e)
    return sigma_e


def load_datasets(cfg: RunConfig) -> Datasets:
    """Load the training, validation and test records listed in MANIFEST.csv."""
    print(f'Data dir ({cfg.noise}): {dataset_dir(cfg)}')
    print(f'{cfg.record_trim} samples dropped at both record ends (u, y and r)')
    train_files, val_files, test_files = record_files(cfg)
    splits = (train_files, val_files, test_files)
    # record file -> (System_data, servo error e, stored controller)
    loaded = {f: load_record(f, cfg) for files in splits for f in files}
    train_list, val_list, test_list = ([loaded[f][0] for f in files] for files in splits)
    e_train, e_val, e_test = ([loaded[f][1] for f in files] for files in splits)
    controllers = {f: spec for f, (_, _, spec) in loaded.items()}

    print(f'Loaded {len(train_list)} train, {len(val_list)} val, {len(test_list)} test records')
    for i, (f, t) in enumerate(zip(train_files, train_list)):
        print(f'  T{i+1}: {t.u.shape[0]} samples  ({f})')

    return Datasets(
        train_list=train_list, val_list=val_list, test_list=test_list,
        train_files=train_files, val_files=val_files, test_files=test_files,
        e_train=e_train, e_val=e_val, e_test=e_test, controllers=controllers,
    )


def compute_normalization(cfg: RunConfig, data: Datasets) -> Norm:
    """Means and standard deviations of the training records, returned as Norm.

    Args:
        cfg: run configuration.
        data: the loaded records.
    """
    # 1. Training forces, measured positions and physical states (q, qdot).
    u_all = np.concatenate([t.u for t in data.train_list])
    y_all = np.concatenate([t.y for t in data.train_list])

    x_per_record = []
    for record in data.train_list:
        # measured positions to generalised coordinates
        q = record.y @ P_INV_T.T
        # a first difference suffices for this scale: its standard deviation differs by under 0.05 % between the noisy
        # and noise-free data
        # velocity at sample k = (q_k - q_(k-1)) * fs_new; sample 0 has no earlier sample, so it takes the velocity
        # of sample 1
        qdot = np.diff(q, axis=0) * cfg.fs_new
        qdot = np.vstack([qdot[:1], qdot])
        x_per_record.append(np.hstack([q, qdot]))
    x_all = np.concatenate(x_per_record)

    # 2. Training means and standard deviations.
    # + 1e-8: a constant channel has standard deviation 0, and normalising divides by it
    x_mean = x_all.mean(axis=0).reshape(cfg.nx_phys, 1)
    std_x = x_all.std(axis=0).reshape(cfg.nx_phys, 1) + 1e-8
    u_mean = u_all.mean(axis=0).reshape(cfg.nu, 1)
    std_u = u_all.std(axis=0).reshape(cfg.nu, 1) + 1e-8
    y0 = y_all.mean(axis=0)
    ystd = y_all.std(axis=0) + 1e-8

    # 3. Output map in normalised units: each column of Cd is multiplied by its state's standard deviation and each
    # row divided by its position's standard deviation. No bias term is needed: y = P^T q holds for every sample, so
    # the mean of y is P^T times the mean of q.
    Cd_norm = Cd * std_x.flatten()[None, :] / ystd[:, None]
    Dd_np = Dd

    return Norm(
        x_mean=x_mean, std_x=std_x, u_mean=u_mean, std_u=std_u,
        y0=y0, ystd=ystd, Cd_norm=Cd_norm, Dd_np=Dd_np,
    )
