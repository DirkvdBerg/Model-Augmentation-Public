"""Real Telica data for the vendored augmentation pipeline (G6; replaces gantry_dynamic/data.py).

Section 4 items this module settles (handoff list):
  3. normalisation: FROZEN, computed once from the train split with LOW-PASSED velocities
     (never a difference of raw positions) and stored in `norm_frozen.json`; val, test and the
     server all read that file (TR-016);
  4. decimation: 20 kHz as logged (TR-008) or 5 kHz by the TR-022 rule (anti-aliased positions,
     block-mean forces, `rate.py`); any other rate is refused, and `y[::D]` point sampling of
     unfiltered data never touches real data;
  6. file lists / format: the split comes from data_index (operating points), records from the
     vendored loader; each record is cut to [motion start - 100 ms, end].
Data are read through the loader only; nothing is cached to disk.
"""
import json
import os
from dataclasses import dataclass

import numpy as np
import deepSI
from scipy.signal import butter, sosfiltfilt

from params import telica_params as tp
from baseline.records import iter_records
import rate

HERE = os.path.dirname(os.path.abspath(__file__))
NORM_FILE = os.path.join(HERE, 'norm_frozen.json')
PRE = 2000                          # samples kept before motion start (100 ms)
LP = butter(4, 200.0, fs=tp.FS, output='sos')      # same HEURISTIC 200 Hz as G4 (TR-014)
P = np.array([[1.0, 1.0, 0.0], [tp.Lb / 2, -tp.Lb / 2, 0.0], [0.0, 0.0, 1.0]])


@dataclass
class RealData:
    train_list: list
    val_list: list
    test_list: list
    train_names: list
    val_names: list
    test_names: list
    train_data: object
    val_ckpt_data: object
    val_data: object
    test_data: object
    # the fields below exist for the pipeline's DataBundle interface; real data has no GT
    val_x_logical: object = None
    val_x_aug: object = None
    test_x_logical: object = None
    test_x_aug: object = None
    sigma_n: object = None


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


def _cut(d, pre=PRE):
    ks = max(0, d['motion_idx'] - pre)
    return d['u'][ks:], d['q1'][ks:]


def _at_rate(d, cfg):
    """The record at the pipeline rate: 20 kHz as logged, or 5 kHz by the TR-022 rule."""
    if cfg.fs_new is None:
        return d, PRE
    if int(cfg.fs_new) == int(rate.FS5):
        return rate.decimate_record(d), PRE // rate.D
    raise ValueError(f'fs_new={cfg.fs_new}: real data runs at 20 kHz (TR-008) or 5 kHz (TR-022) '
                     f'only, the rates at which an identified Telica controller exists')


def load_split(cfg, splits_iters, dtype_np, log=print, max_records=None, max_len=None):
    """[(name, System_data)] for the requested (split, iters) pairs."""
    out = []
    for split, iters in splits_iters:
        for s, op, it, d in iter_records(splits=(split,), iters=iters, optional=True, log=log):
            d, pre = _at_rate(d, cfg)
            u, y = _cut(d, pre)
            if max_len:
                u, y = u[:max_len], y[:max_len]
            out.append((f'{s}/{op}/{it}',
                        deepSI.System_data(u=u.astype(dtype_np), y=y.astype(dtype_np),
                                           dt=cfg.ts_new)))
            if max_records and len(out) >= max_records:
                return out
    return out


def load_datasets_real(cfg, train_iters=None, val_iters=('iter0', 'iterETEL'),
                       test_iters=('iter0', 'iterETEL', 'iterTEST'), max_records=None,
                       max_len=None, log=print):
    if cfg.fs_new is not None and int(cfg.fs_new) != int(rate.FS5):
        raise ValueError(f'fs_new={cfg.fs_new}: real data runs at 20 kHz (TR-008) or 5 kHz '
                         f'(TR-022) only')
    dt = cfg.dtype_np
    tr = load_split(cfg, [('train', train_iters)], dt, log, max_records, max_len)
    va = load_split(cfg, [('validation', val_iters)], dt, log, max_records, max_len)
    te = load_split(cfg, [('test', test_iters)], dt, log, max_records, max_len)
    log(f'[real_data] train {len(tr)}, val {len(va)}, test {len(te)} records at {1 / cfg.ts_new:.0f} Hz')
    tl = [sd for _, sd in tr]; vl = [sd for _, sd in va]; el = [sd for _, sd in te]
    return RealData(train_list=tl, val_list=vl, test_list=el,
                    train_names=[n for n, _ in tr], val_names=[n for n, _ in va],
                    test_names=[n for n, _ in te],
                    train_data=deepSI.System_data_list(tl), val_ckpt_data=deepSI.System_data_list(vl),
                    val_data=vl[0], test_data=el[0])


def build_frozen_norm(log=print):
    """Compute the normalisation from the train split (filtered velocities) and store it."""
    xs, us, ys = [], [], []
    Pinv_T = np.linalg.inv(P.T)
    for s, op, it, d in iter_records(splits=('train',), log=log):
        u, y = _cut(d)
        yf = sosfiltfilt(LP, y, axis=0)
        v = np.gradient(yf, 1 / tp.FS, axis=0)
        q = y @ Pinv_T.T; qd = v @ Pinv_T.T
        xs.append(np.hstack([q, qd])[::10]); us.append(u[::10]); ys.append(y[::10])
    x = np.concatenate(xs); u = np.concatenate(us); y = np.concatenate(ys)
    N = dict(x_mean=x.mean(0).tolist(), std_x=(x.std(0) + 1e-12).tolist(),
             u_mean=u.mean(0).tolist(), std_u=(u.std(0) + 1e-12).tolist(),
             y0=y.mean(0).tolist(), ystd=(y.std(0) + 1e-12).tolist(),
             source='train split, [motion-100 ms, end], 200 Hz zero-phase LP velocities, every 10th '
                    'sample', n_samples=int(len(x)))
    json.dump(N, open(NORM_FILE, 'w'), indent=1)
    log(f'[real_data] frozen normalisation written: {NORM_FILE}')
    return N


def frozen_norm(cfg):
    """The pipeline's Norm from norm_frozen.json (no statistics from the data being trained)."""
    from model_augmentation.systems.gantry_ss import Cd, Dd
    N = json.load(open(NORM_FILE))
    dt = cfg.dtype_np
    std_x = np.asarray(N['std_x'], dt).reshape(6, 1)
    ystd = np.asarray(N['ystd'], dt)
    Cd_norm = Cd.double().numpy() * std_x.flatten()[None, :] / ystd[:, None]
    return Norm(x_mean=np.asarray(N['x_mean'], dt).reshape(6, 1), std_x=std_x,
                std_u=np.asarray(N['std_u'], dt).reshape(3, 1),
                u_mean=np.asarray(N['u_mean'], dt).reshape(3, 1),
                ystd=ystd, y0=np.asarray(N['y0'], dt), Cd_norm=Cd_norm.astype(dt),
                Dd_np=Dd.double().numpy().astype(dt), u_all=None, y_all=None, x_all=None)
