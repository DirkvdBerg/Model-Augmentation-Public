"""Data loading, resampling, output noise, and normalization.

All numerics are verbatim moves from the pre-refactor entry file; the only
change is that module globals (cfg constants) are now read from a RunConfig and
the loaded state is returned in two explicit containers (DataBundle, Norm)
instead of being written to module-level names.
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

from .config import (RunConfig, REPO_ROOT, ANTI_ALIAS_MODES, AA_ORDER, AA_FC_HZ, TRIM_MODES,
                     record_trim, y_decimation)

# D-206: the TP*/VP*/EP* records are Telica-derived operational motion profiles with NO
# multisine (cycloidal setpoint measured from the real machine; the reference trajectory is
# the only excitation). They exist because every T*/V*/E* record carries a multisine, so a
# model trained on them alone extrapolates badly on multisine-free evaluation data.
#
# APPENDED, NEVER PREPENDED. VAL_FILES[0] and TEST_FILES[0] are the PRIMARY diagnostic
# records (load_mat_aug, baselines, GT plots); prepending would silently move every
# diagnostic off V1/E1.
#
# These names only resolve in a dataset folder that HOLDS them. The Telica records are
# generated into 'augmentation_ma50_z03_telica' and copied alongside the production records
# into 'augmentation_ma50_z03_b140-230_a6_telica'; cfg.mode must point at the merged folder
# or load_traj raises FileNotFoundError on the first TP* entry.
TRAIN_FILES = [
    'T1_standstill_Ym30.mat', 'T2_standstill_Ym15.mat', 'T3_standstill_Y000.mat',
    'T4_standstill_Yp15.mat', 'T5_standstill_Yp30.mat',
    'T6_ysweep_slow.mat', 'T7_ysweep_fast.mat', 'T8_ysweep_xmix.mat',
    'T9_aprbs_30.mat', 'T10_aprbs_60.mat', 'T11_aprbs_100.mat', 'T12_aprbs_yaw.mat',
    'T13_lissajous.mat', 'T14_lissajous_yaw.mat',
    'TP1_telica_y000.mat', 'TP2_telica_y000_rev.mat',
    'TP3_telica_yp06.mat', 'TP4_telica_ym06.mat',
]
VAL_FILES = [
    'V1_standstill_Yp10.mat', 'V2_aprbs_Ylow.mat',
    'V3_ysweep_Yp10.mat', 'V4_lissajous_Ym10.mat',
    'VP1_telica_ylow.mat', 'VP2_telica_yp12.mat',
]
TEST_FILES = [
    'E1_resonance_sweep.mat', 'E2_multisine_Yp22.mat',
    'E3_aprbs_above.mat', 'E4_multisine_off.mat',
    'EP1_telica_test.mat',
]

# D-221: the thesis data sets (Thesis-writeup/Data), one folder per noise version. The record
# lists are NOT constants: they come from each folder's MANIFEST.csv, so the split is exactly the
# one DATA-DESIGN.md section 7 generated. The constants above stay for the legacy modes, which
# many experiment scripts import.
THESIS_DATASETS = {
    # D-230: the tanh-friction / encoder-noise set (Thesis-writeup/Code/Data/tdg_config.m name_*);
    # the Karnopp / force-noise set of 2026-09-27 stays in 'Coulomb-and-MSD[-noise-free]'
    'thesis_taf_noisy': 'Coulomb-tanh-and-MSD',
    'thesis_taf_lowpass_noise': 'Coulomb-tanh-and-MSD-lowpass-noise',
    'thesis_taf_noisefree': 'Coulomb-tanh-and-MSD-noise-free',
}
# MANIFEST.csv `block` -> split. The TF-* FRF references (block `frf`) are not rollout records.
_THESIS_SPLIT = {'training': 'train', 'validation': 'val', 'test-I': 'test', 'test-E': 'test'}
# D-232: every thesis mode decides its end trim (config.TRIM_MODES), so a new data set cannot
# silently load on a different sample grid than its twin.
assert set(TRIM_MODES) == set(THESIS_DATASETS), 'D-232: TRIM_MODES must list every thesis mode'


def is_thesis(cfg: RunConfig) -> bool:
    return cfg.mode in THESIS_DATASETS


def record_files(cfg: RunConfig):
    """(train, val, test) file names for cfg.mode, relative to traj_dir(cfg), in manifest order."""
    if not is_thesis(cfg):
        return list(TRAIN_FILES), list(VAL_FILES), list(TEST_FILES)
    out = {'train': [], 'val': [], 'test': []}
    with open(os.path.join(traj_dir(cfg), 'MANIFEST.csv'), newline='') as fh:
        for row in csv.DictReader(fh):
            split = _THESIS_SPLIT.get(row['block'])
            if split is not None:
                out[split].append('%s/%s.mat' % (row['folder'], row['file']))
    return out['train'], out['val'], out['test']


def _register_controller(d, filename):
    """Hand the record's stored controller (meta.controller, D-221) to the closed-loop bank."""
    from .controller import register_stored_controller
    ctl = d['meta']['controller'].item()
    get = lambda k: ctl[k].item() if hasattr(ctl[k], 'item') else ctl[k]   # noqa: E731
    register_stored_controller(filename, str(get('name')), float(get('design_Y')),
                               float(get('gain')), float(get('fbw_Hz')))


@dataclass
class DataBundle:
    train_list: list
    val_list: list
    test_list: list
    train_data: object
    val_ckpt_data: object
    val_data: object
    test_data: object
    val_x_logical: object
    val_x_aug: object
    test_x_logical: object
    test_x_aug: object
    sigma_n: object
    # D-221: the file names the lists above were loaded from (same order). Defaults keep older
    # callers that build a DataBundle themselves working.
    train_files: list = None
    val_files: list = None
    test_files: list = None


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
    u_all: object
    y_all: object
    x_all: object


def traj_dir(cfg: RunConfig) -> str:
    if cfg.mode in THESIS_DATASETS:
        return os.path.join(REPO_ROOT, 'Thesis-writeup', 'Data', THESIS_DATASETS[cfg.mode])
    return os.path.join(REPO_ROOT, 'data', 'gantry', 'matlab', 'trajectory', cfg.mode)


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
    if D == 1:
        return u
    n = u.shape[0] // D
    return u[:n * D].astype(np.float64).reshape(n, D, u.shape[1]).mean(axis=1)


def _decimate_y(d, cfg: RunConfig):
    """y at FS_NEW: the D-232 anti-alias rule in ANTI_ALIAS_MODES, point sampling otherwise."""
    D = cfg.d
    if cfg.mode not in ANTI_ALIAS_MODES:
        return d['y'][::D]   # point sampling exact: y band-limited << 2 kHz (D-087; verified D-099, frac_above 2.5e-8)
    if (cfg.fs_orig, cfg.fs_new_hz) != (20000, 4000):
        raise ValueError('the D-232 anti-alias rule is validated for 20 -> 4 kHz only, got %d -> %d Hz'
                         % (cfg.fs_orig, cfg.fs_new_hz))
    r = d['r_sim']
    sos = butter(AA_ORDER, AA_FC_HZ, fs=cfg.fs_orig, output='sos')
    return r[::D] + sosfiltfilt(sos, d['y'] - r, axis=0)[::D]


def _record_at_fs_new(d, cfg: RunConfig, fields=()):
    """[u, y, *fields] of one record at FS_NEW, all on the same samples.

    u is the block mean (D-087), y follows _decimate_y, each named field is point-sampled. All are
    then cut by config.record_trim at both ends (D-232, the thesis modes); smoke runs cut last (D-222).
    """
    D = cfg.d
    u = _resample_u(_load_u(d), cfg)
    N = len(u)
    arrs = [u, _decimate_y(d, cfg)[:N]] + [np.asarray(d[k])[::D][:N] for k in fields]
    k = record_trim(cfg)
    if k:
        arrs = [a[k:N - k] for a in arrs]
    if cfg.truncate_s is not None:                        # smoke tests only (D-222)
        arrs = [a[:int(cfg.truncate_s * cfg.fs_new_hz)] for a in arrs]
    return arrs


def load_traj(filename, cfg: RunConfig):
    d = loadmat(os.path.join(traj_dir(cfg), filename), squeeze_me=True)
    if is_thesis(cfg):
        _register_controller(d, filename)
    u, y = _record_at_fs_new(d, cfg)
    return deepSI.System_data(u=u.astype(cfg.dtype_np), y=y.astype(cfg.dtype_np), dt=cfg.ts_new)


def load_mat_aug(filename, cfg: RunConfig):
    """Load u, y, x_logical, delta_a, vdelta_a from augmented simulation mat file.

    Used for diagnostics only (not for training). Returns:
      u         (N, nu)   plant input [N]
      y         (N, ny)   plant output [m]
      x_logical (N, 6)    physical states from augmented sim [m, m/s]
      x_aug     (N, 2)    [delta_a, vdelta_a] -- vdelta_a via backward FD
    """
    d = loadmat(os.path.join(traj_dir(cfg), filename), squeeze_me=True)
    # the samples load_traj gives the same record (D-232 rule and trim, D-222 smoke cut)
    u, y, x_logical, delta_a = (a.astype(cfg.dtype_np) for a in
                                _record_at_fs_new(d, cfg, ('x_logical', 'delta_a')))   # (N,3) (N,3) (N,6) (N,)
    # HEURISTIC: backward FD for velocity -- O(Ts) accurate, consistent with x_logical convention
    vdelta_a      = np.zeros_like(delta_a)
    vdelta_a[1:]  = (delta_a[1:] - delta_a[:-1]) * cfg.fs_new_hz
    vdelta_a[0]   = vdelta_a[1]
    x_aug = np.stack([delta_a, vdelta_a], axis=1)       # (N, 2)
    return u, y, x_logical, x_aug


def load_datasets(cfg: RunConfig) -> DataBundle:
    """Load train/val/test trajectories, add output noise, build the diagnostic GT.

    Reproduces the pre-refactor module-level sequence exactly (including print
    order and the noise-loop iteration order). The caller must seed numpy/torch
    before calling this, as the original did at the top of the data section.
    """
    print(f'Data dir ({cfg.mode}): {traj_dir(cfg)}')
    if cfg.mode in TRIM_MODES:
        print(f'y decimation (D-232): {y_decimation(cfg)}; {record_trim(cfg)} samples dropped at both '
              f'record ends (u, y and every field)')
    SNR = cfg.snr
    if SNR is not None and is_thesis(cfg):
        raise ValueError('cfg.snr=%r would add output noise on top of %s, whose noise is already in '
                         'the data; use thesis_taf_noisefree for the '
                         'noise-free twin instead (D-221)' % (SNR, cfg.mode))
    train_files, val_files, test_files = record_files(cfg)

    train_list = [load_traj(f, cfg) for f in train_files]
    val_list   = [load_traj(f, cfg) for f in val_files]
    test_list  = [load_traj(f, cfg) for f in test_files]

    ## ------------- Add output noise (Jan's ECC convention, D-078) -------------
    if SNR is not None:
        if   SNR == 50: sigma_n = 4.5e-4
        elif SNR == 55: sigma_n = 2.5e-4
        elif SNR == 60: sigma_n = 1.4e-4
        else: raise ValueError('SNR must be 50, 55, 60 or None')
        for sd in train_list + val_list + test_list:
            sd.y = sd.y + np.random.normal(0, sigma_n, sd.y.shape).astype(cfg.dtype_np)
        print(f'Added output noise: SNR={SNR} dB, sigma_n={sigma_n:.2e} m (noise floor)')
    else:
        sigma_n = None

    train_data    = deepSI.System_data_list(train_list)   # build after noising
    val_ckpt_data = deepSI.System_data_list(val_list)     # checkpoint selection over all V1-V4
    val_data  = val_list[0]     # primary val record (V1) for GT-based diagnostics/plots
    test_data = test_list[0]    # primary test record (E1) for GT-based diagnostics/baselines

    # Augmented ground-truth fields for the PRIMARY val/test records (diagnostics only)
    _, _, val_x_logical, val_x_aug = load_mat_aug(val_files[0], cfg)
    print(f'Loaded val augmented GT: x_logical={val_x_logical.shape}  '
          f'delta_a std={val_x_aug[:,0].std():.3e} m  '
          f'vdelta_a std={val_x_aug[:,1].std():.3e} m/s')

    test_x_aug = None
    try:
        _, _, test_x_logical, test_x_aug = load_mat_aug(test_files[0], cfg)
    except KeyError as e:
        print(f'WARNING: test file lacks augmented GT fields ({e}); skipping test baseline')
        test_x_logical = None

    print(f'Loaded {len(train_list)} train, {len(val_list)} val, {len(test_list)} test trajectories')
    for i, (f, t) in enumerate(zip(train_files, train_list)):
        print(f'  T{i+1}: {t.u.shape[0]} samples  ({f})')

    return DataBundle(
        train_list=train_list, val_list=val_list, test_list=test_list,
        train_data=train_data, val_ckpt_data=val_ckpt_data,
        val_data=val_data, test_data=test_data,
        val_x_logical=val_x_logical, val_x_aug=val_x_aug,
        test_x_logical=test_x_logical, test_x_aug=test_x_aug,
        sigma_n=sigma_n,
        train_files=train_files, val_files=val_files, test_files=test_files,
    )


def compute_normalization(cfg: RunConfig, data: DataBundle) -> Norm:
    """All NX_PHYS-dimensional normalization constants (independent of NX_ANN)."""
    NX_PHYS, nu = cfg.nx_phys, cfg.nu
    DTYPE_NP = cfg.dtype_np
    train_list = data.train_list

    u_all = np.concatenate([t.u for t in train_list])
    y_all = np.concatenate([t.y for t in train_list])

    fs = 1.0 / train_list[0].dt
    P_inv_T = np.linalg.inv(P.numpy().T).astype(DTYPE_NP)  # stage -> logical
    x_logical_list = []
    for t in train_list:
        pos_logical = (P_inv_T @ t.y.T).T        # (N, 3) stage -> logical
        # Velocity scale from a first difference of the decimated positions.
        # NOISELESS: exact. Measured against the stored 20 kHz central-difference
        # x_logical, the resulting std_x agrees to 3 decimals on every channel, because
        # a std is dominated by low-frequency content where a first difference is
        # accurate. Nothing to fix while cfg.snr is None.
        # WITH OUTPUT NOISE: unusable as-is. Differencing white position noise gives
        # # THEORY: var of a first difference of white noise = 2*sigma_n^2, so
        # sigma_v = sigma_n*sqrt(2)*fs = 0.79 m/s at sigma_n=1.4e-4 (SNR 60), against
        # true velocity stds of 0.008 to 0.48. Measured std_x inflation vs the 20 kHz
        # reference: SNR 60 -> dX 2.7x, dTheta 193x, dY 1.9x; SNR 55 -> 4.6x/344x/3.1x;
        # SNR 50 -> 8.1x/619x/5.4x. dTheta is worst because P^-T amplifies that channel
        # by 1.95 while its true std is only 0.008. std_x scales both the encoder
        # matrices and the Gantry_State_Block denormalisation (D-119), so this silently
        # re-frames the encoder per noise level and makes SNR arms incomparable.
        # Before any run with cfg.snr set, do one of:
        #   (a) derive these statistics from the PRE-NOISE signals (SNR-independent, and
        #       bit-identical to today when snr is None), or freeze them to a stored
        #       calibration record for the real-data case. Preferred.
        #   (b) low-pass the POSITIONS before differencing. Buys only sqrt(D) (0.79 ->
        #       0.354 m/s) and does NOT recover dTheta, whose true velocity content sits
        #       inside the 130-180 Hz band where the differentiated noise dominates.
        #       Partial mitigation, not a fix.
        vel_logical = np.diff(pos_logical, axis=0) * fs  # (N-1, 3)
        vel_logical = np.vstack([vel_logical[:1], vel_logical])  # (N, 3)
        x_logical_list.append(np.hstack([pos_logical, vel_logical]))  # (N, 6)
    x_all = np.concatenate(x_logical_list)

    x_mean = x_all.mean(axis=0).reshape(NX_PHYS, 1).astype(DTYPE_NP)
    std_x  = x_all.std(axis=0).reshape(NX_PHYS, 1).astype(DTYPE_NP) + 1e-8
    std_u  = u_all.std(axis=0).reshape(nu, 1).astype(DTYPE_NP) + 1e-8
    u_mean = u_all.mean(axis=0).reshape(nu, 1).astype(DTYPE_NP)
    ystd   = y_all.std(axis=0).astype(DTYPE_NP) + 1e-8
    y0     = y_all.mean(axis=0).astype(DTYPE_NP)

    # Cd_norm[i,j] = Cd[i,j] * std_x[j] / ystd[i]
    Cd_norm = Cd.numpy() * std_x.flatten()[None, :] / ystd[:, None]  # (3, 6)
    Dd_np   = Dd.numpy()                                               # (3, 3)

    return Norm(
        x_mean=x_mean, std_x=std_x, std_u=std_u, u_mean=u_mean,
        ystd=ystd, y0=y0, Cd_norm=Cd_norm, Dd_np=Dd_np,
        u_all=u_all, y_all=y_all, x_all=x_all,
    )
