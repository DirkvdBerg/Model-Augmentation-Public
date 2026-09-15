"""In-memory, matched-filter rate views for the isolated horizon premise test.

No dataset is generated or written. Every view is derived from the existing
20 kHz MATLAB master record. Input and output are concatenated before one
polyphase FIR call, which makes filter coefficients, padding, phase treatment,
and sample alignment identical across channels.
"""

from __future__ import annotations

import os

import deepSI
import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly

from gantry_dynamic.config import RunConfig
from gantry_dynamic.data import (DataBundle, TRAIN_FILES, VAL_FILES, TEST_FILES, compute_normalization,
                   traj_dir)


FILTER_WINDOW = ('kaiser', 8.6)
FILTER_PADTYPE = 'line'


def matched_decimate(u, y, down: int):
    """Filter and decimate ``u`` and ``y`` identically, returning aligned arrays."""
    u = np.asarray(u)
    y = np.asarray(y)
    if u.shape[0] != y.shape[0]:
        raise ValueError('u and y master records must have identical lengths')
    if down < 1:
        raise ValueError('down must be positive')
    joined = np.concatenate((u, y), axis=1).astype(np.float64, copy=False)
    if down == 1:
        out = joined.copy()
    else:
        out = resample_poly(joined, up=1, down=int(down), axis=0,
                            window=FILTER_WINDOW, padtype=FILTER_PADTYPE)
    return out[:, :u.shape[1]], out[:, u.shape[1]:]


def _load_record(filename: str, cfg: RunConfig):
    if cfg.fs_orig % cfg.fs_new_hz:
        raise ValueError('premise rates must integer-divide the master rate')
    raw = loadmat(os.path.join(traj_dir(cfg), filename), squeeze_me=True)
    u, y = matched_decimate(raw['u_total'], raw['y'], cfg.d)
    return deepSI.System_data(u=u.astype(cfg.dtype_np), y=y.astype(cfg.dtype_np),
                              dt=cfg.ts_new)


def load_matched_datasets(cfg: RunConfig) -> DataBundle:
    """Load one filtered rate view without changing or writing the master data."""
    records = {f: _load_record(f, cfg)
               for f in TRAIN_FILES + VAL_FILES + TEST_FILES}
    train_list = [records[f] for f in TRAIN_FILES]
    val_list = [records[f] for f in VAL_FILES]
    test_list = [records[f] for f in TEST_FILES]
    return DataBundle(
        train_list=train_list, val_list=val_list, test_list=test_list,
        train_data=deepSI.System_data_list(train_list),
        val_ckpt_data=deepSI.System_data_list(val_list),
        val_data=val_list[0], test_data=test_list[0],
        # These fields are not training inputs and are intentionally omitted from
        # this premise experiment; no rate-dependent finite-difference GT is made.
        val_x_logical=None, val_x_aug=None,
        test_x_logical=None, test_x_aug=None, sigma_n=None)


def load_premise_data(cfg: RunConfig, normalization_rate_hz: int = 4000):
    """Return arm-rate data plus normalization frozen from the 4 kHz view."""
    if cfg.snr is not None:
        raise ValueError('the premise test is preregistered noiseless')
    ref_cfg = RunConfig(**{**cfg.__dict__, 'fs_new': int(normalization_rate_hz)})
    reference = load_matched_datasets(ref_cfg)
    norm = compute_normalization(ref_cfg, reference)
    data = reference if cfg.fs_new_hz == normalization_rate_hz else load_matched_datasets(cfg)
    return data, norm, reference
