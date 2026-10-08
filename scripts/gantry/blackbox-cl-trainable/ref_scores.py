"""Section-10 references for option B (handoff 2026-10-03), on the 6 validation records, per stage
axis X1 / X2 / Y, metres, scored from k0 = 29:

  1 trivial      y_hat = r                       RMS(r - y), NRMS_e = 1 by definition
  2 grey box     epoch 0 (physics at the thesis start, learned block zero, linear-map encoder),
                 scored by the pipeline's own `closed_loop_table` (rows 'baseline, start' and, as
                 context, 'baseline, true'); no trained thesis grey box exists (stored thesis runs
                 are 2 s smoke tests, TRUNCATE_S = 2.0)
  3 floor        RMS(y_noisy - y_noisefree)

The thesis CFG comes from the entry file through its declared THESIS_* interface (arm u, 2 added
states, detuned start, seed 1, CPU). Data are loaded with the pipeline's own `load_traj`, but only
the training and validation records: the test records are never loaded.

Local:  python -u ref_scores.py
"""
__project_origin__ = "added"

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
for p in (os.path.join(REPO, 'scripts', 'gantry'), REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                                                     # noqa: E402
from scipy.io import loadmat                                           # noqa: E402

K0 = 29
OUT = os.path.join(HERE, 'outputs', 'ref')


def thesis_cfg(noise):
    os.environ.update(THESIS_ARM='u', THESIS_NOISE=noise, THESIS_START='detuned', THESIS_NA='2',
                      THESIS_SEED='1', THESIS_DEVICE='cpu')
    import importlib
    import gantry_interconnect_dynamic as entry
    entry = importlib.reload(entry)                                    # re-read CFG for this noise
    return entry.CFG


def load_rfy(cfg, files):
    from gantry_dynamic.data import traj_dir, _record_at_fs_new
    out = []
    for fn in files:
        d = loadmat(os.path.join(traj_dir(cfg), fn), squeeze_me=True)
        _, y, r = _record_at_fs_new(d, cfg, ('r_sim',))
        out.append((np.asarray(r, float), np.asarray(y, float)))
    return out


def grey_box_epoch0(cfg):
    import torch
    from gantry_dynamic.data import (DataBundle, compute_normalization, load_traj, record_files,
                                     load_mat_aug)
    import deepSI
    from gantry_dynamic.model import build_model
    from gantry_dynamic.controller import build_closed_loop
    from gantry_dynamic.closed_loop_report import closed_loop_table
    tr_f, va_f, _ = record_files(cfg)                                  # test names not loaded
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    tr = [load_traj(f, cfg) for f in tr_f]
    va = [load_traj(f, cfg) for f in va_f]
    _, _, vxl, vxa = load_mat_aug(va_f[0], cfg)
    data = DataBundle(train_list=tr, val_list=va, test_list=[],
                      train_data=deepSI.System_data_list(tr), val_ckpt_data=deepSI.System_data_list(va),
                      val_data=va[0], test_data=None, val_x_logical=vxl, val_x_aug=vxa,
                      test_x_logical=None, test_x_aug=None, sigma_n=None,
                      train_files=tr_f, val_files=va_f, test_files=[])
    norm = compute_normalization(cfg, data)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    fit_sys = build_model(cfg.hp, cfg, data, norm)
    fit_sys.simulator = build_closed_loop(fit_sys, norm, cfg, train_files=tr_f, val_files=va_f,
                                          val_data=data.val_ckpt_data)
    tab = closed_loop_table(fit_sys, cfg, data, norm, cfg.hp['up_sample'], 'val')
    return va_f, tab


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    res = {}
    rfy = {}
    for noise in ('noisy', 'noisefree'):
        cfg = thesis_cfg(noise)
        assert cfg.truncate_s is None and cfg.mode.endswith(noise)
        from gantry_dynamic.data import record_files
        _, va_f, _ = record_files(cfg)
        rfy[noise] = load_rfy(cfg, va_f)
        triv = np.array([np.sqrt(np.mean((r[K0:] - y[K0:]) ** 2, 0)) for r, y in rfy[noise]])
        files, tab = grey_box_epoch0(cfg)
        assert tab['k0'] == K0, tab['k0']
        rms = np.asarray(tab['rms'])
        res[noise] = dict(files=list(map(str, files)), trivial_per_record=triv.tolist(),
                          trivial_mean=triv.mean(0).tolist(),
                          gb_start_per_record=rms[2].tolist(), gb_start_mean=rms[2].mean(0).tolist(),
                          gb_true_per_record=rms[3].tolist(), gb_true_mean=rms[3].mean(0).tolist(),
                          oracle_mean=np.nanmean(rms[4], 0).tolist(),
                          gb_start_nrms_e=(rms[2].mean(0) / triv.mean(0)).tolist())
        print('[ref] %s trivial %s | gb start %s | gb true %s m' % (
            noise, np.array2string(triv.mean(0), precision=3), np.array2string(rms[2].mean(0), precision=3),
            np.array2string(rms[3].mean(0), precision=3)), flush=True)
    floor = np.array([np.sqrt(np.mean((a[1][K0:] - b[1][K0:]) ** 2, 0))
                      for a, b in zip(rfy['noisy'], rfy['noisefree'])])
    res['floor_per_record'] = floor.tolist()
    res['floor_mean'] = floor.mean(0).tolist()
    res['k0'] = K0
    res['wall_s'] = time.time() - t0
    print('[ref] floor %s m (%.0f s)' % (np.array2string(floor.mean(0), precision=3), res['wall_s']),
          flush=True)
    with open(os.path.join(OUT, 'references.json'), 'w') as fh:
        json.dump(res, fh, indent=1)


if __name__ == '__main__':
    main()
