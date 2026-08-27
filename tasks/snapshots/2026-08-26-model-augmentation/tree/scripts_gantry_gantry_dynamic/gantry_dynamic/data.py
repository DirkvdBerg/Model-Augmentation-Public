"""Data loading, resampling, output noise, and normalization.

All numerics are verbatim moves from the pre-refactor entry file; the only
change is that module globals (cfg constants) are now read from a RunConfig and
the loaded state is returned in two explicit containers (DataBundle, Norm)
instead of being written to module-level names.
"""
__project_origin__ = "added"

import os
from dataclasses import dataclass

import numpy as np
import deepSI
from scipy.io import loadmat

from model_augmentation.systems.gantry_ss import Cd, Dd, P

from .config import RunConfig, REPO_ROOT

TRAIN_FILES = [
    'T1_standstill_Ym30.mat', 'T2_standstill_Ym15.mat', 'T3_standstill_Y000.mat',
    'T4_standstill_Yp15.mat', 'T5_standstill_Yp30.mat',
    'T6_ysweep_slow.mat', 'T7_ysweep_fast.mat', 'T8_ysweep_xmix.mat',
    'T9_aprbs_30.mat', 'T10_aprbs_60.mat', 'T11_aprbs_100.mat', 'T12_aprbs_yaw.mat',
    'T13_lissajous.mat', 'T14_lissajous_yaw.mat',
]
VAL_FILES = [
    'V1_standstill_Yp10.mat', 'V2_aprbs_Ylow.mat',
    'V3_ysweep_Yp10.mat', 'V4_lissajous_Ym10.mat',
]
TEST_FILES = [
    'E1_resonance_sweep.mat', 'E2_multisine_Yp22.mat',
    'E3_aprbs_above.mat', 'E4_multisine_off.mat',
]


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


def load_traj(filename, cfg: RunConfig):
    D = cfg.d
    d = loadmat(os.path.join(traj_dir(cfg), filename), squeeze_me=True)
    u = _resample_u(_load_u(d), cfg).astype(cfg.dtype_np)
    return deepSI.System_data(
        u=u,
        y=d['y'][::D][:len(u)].astype(cfg.dtype_np),   # point sampling exact: y band-limited << 2 kHz (D-087; verified D-099, frac_above 2.5e-8)
        dt=cfg.ts_new,
    )


def load_mat_aug(filename, cfg: RunConfig):
    """Load u, y, x_logical, delta_a, vdelta_a from augmented simulation mat file.

    Used for diagnostics only (not for training). Returns:
      u         (N, nu)   plant input [N]
      y         (N, ny)   plant output [m]
      x_logical (N, 6)    physical states from augmented sim [m, m/s]
      x_aug     (N, 2)    [delta_a, vdelta_a] -- vdelta_a via backward FD
    """
    D = cfg.d
    d = loadmat(os.path.join(traj_dir(cfg), filename), squeeze_me=True)
    u         = _resample_u(_load_u(d), cfg).astype(cfg.dtype_np)     # block mean (D-087)
    N         = len(u)
    y         = d['y'][::D][:N].astype(cfg.dtype_np)
    x_logical = d['x_logical'][::D][:N].astype(cfg.dtype_np)     # (N, 6)
    delta_a   = d['delta_a'][::D][:N].astype(cfg.dtype_np)       # (N,)
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

    train_list = [load_traj(f, cfg) for f in TRAIN_FILES]
    val_list   = [load_traj(f, cfg) for f in VAL_FILES]
    test_list  = [load_traj(f, cfg) for f in TEST_FILES]

    ## ------------- Add output noise (Jan's ECC convention, D-078) -------------
    SNR = cfg.snr
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
    _, _, val_x_logical, val_x_aug = load_mat_aug(VAL_FILES[0], cfg)
    print(f'Loaded val augmented GT: x_logical={val_x_logical.shape}  '
          f'delta_a std={val_x_aug[:,0].std():.3e} m  '
          f'vdelta_a std={val_x_aug[:,1].std():.3e} m/s')

    test_x_aug = None
    try:
        _, _, test_x_logical, test_x_aug = load_mat_aug(TEST_FILES[0], cfg)
    except KeyError as e:
        print(f'WARNING: test file lacks augmented GT fields ({e}); skipping test baseline')
        test_x_logical = None

    print(f'Loaded {len(train_list)} train, {len(val_list)} val, {len(test_list)} test trajectories')
    for i, (f, t) in enumerate(zip(TRAIN_FILES, train_list)):
        print(f'  T{i+1}: {t.u.shape[0]} samples  ({f})')

    return DataBundle(
        train_list=train_list, val_list=val_list, test_list=test_list,
        train_data=train_data, val_ckpt_data=val_ckpt_data,
        val_data=val_data, test_data=test_data,
        val_x_logical=val_x_logical, val_x_aug=val_x_aug,
        test_x_logical=test_x_logical, test_x_aug=test_x_aug,
        sigma_n=sigma_n,
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
