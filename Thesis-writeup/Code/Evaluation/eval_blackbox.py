"""Test-set predictions of the selected black box (R8, 88757 index 3, nx16_w08) and the floors.

Black box: deepSI `SS_encoder_general_hf` on the closed-loop map (r, f) -> e = y - r, loaded from
its `_best.pth` (the checkpoint selected on validation), fed exactly as in its training script
(`train_bb_clmap.load`: --target e --in64 --f-blockmean), y_hat = r + e_hat, scored from k0 = 29.
E5 is left out: the black box models the K1 loop, so a K2 record is outside its definition.
Check: the validation RMS must reproduce `result.json` `per_record`.

Floor per test record: the noisy record minus its noise-free twin, both through the same
lowpass-mode loader (D-232 decimation and trim), from k0 (THESIS-RESULTS section 2).
Also stored per record: RMS(e) = RMS(y - r) for NRMS_e.

Usage: python Thesis-writeup/Code/Evaluation/eval_blackbox.py
"""
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.io import loadmat

from eval_common import (REPO, TW, BB_CKPT, BB_CODE, CACHE, DATA_NF, K0, record_name)

sys.path.insert(0, str(REPO / 'scripts' / 'gantry' / 'blackbox-cl-trainable' / 'bbcl'))


def load_records(cfg, files, folder=None):
    """(u, target e, r, y) per record, as train_bb_clmap.load with --target e --in64 --f-blockmean."""
    from gantry_dynamic.data import traj_dir, _record_at_fs_new, _load_u, _resample_u
    from gantry_dynamic.config import record_trim
    out = []
    for fn in files:
        d = loadmat(os.path.join(folder or traj_dir(cfg), fn), squeeze_me=True)
        _, y, r, f = _record_at_fs_new(d, cfg, ('r_sim', 'f_sim'))
        r, y = np.asarray(np.broadcast_to(r, y.shape), float), np.asarray(y, float)
        n = len(_resample_u(_load_u(d), cfg))
        f = _resample_u(np.asarray(d['f_sim'], float), cfg)[:n]
        k = record_trim(cfg)
        f = f[k:n - k] if k else f
        assert f.shape == y.shape, (fn, f.shape, y.shape)
        out.append((np.hstack([r, f]), y - r, r, y))
    return out


def main():
    import deepSI
    from deepSI.fit_systems.encoders import SS_encoder_general_hf
    from setup import cfg_for
    from gantry_dynamic.data import record_files
    t0 = time.time()
    cfg = cfg_for('lowpass', 1, local=True)
    _, va_f, te_f = record_files(cfg)
    te_f = [f for f in te_f if record_name(f) != 'E5']

    model = SS_encoder_general_hf.__new__(SS_encoder_general_hf)
    model.__dict__ = torch.load(BB_CKPT / ('SS_encoder_general_hf_%s_best.pth' % BB_CODE),
                                map_location='cpu', weights_only=False)
    print('[BB] nx %d, loaded in %.0f s' % (model.nx, time.time() - t0), flush=True)

    def predict(recs):
        preds = []
        for u, e, r, y in recs:
            sd = deepSI.System_data(u=np.ascontiguousarray(u), y=np.ascontiguousarray(e), dt=cfg.ts_new)
            with np.errstate(all='ignore'):
                preds.append(r + np.asarray(model.apply_experiment(sd).y, float))
        return preds

    # check: validation per-record RMS of result.json
    res = json.load(open(TW / 'Results' / '11-10-2026' / 'Black-box' / 'thesis_r8_grid' /
                         '3_nx16_w08_n20000_88757' / 'result.json'))
    va = load_records(cfg, va_f)
    pv = predict(va)
    mine = np.array([np.sqrt(np.mean((p[K0:] - y[K0:]) ** 2, 0)) for p, (_, _, _, y) in zip(pv, va)])
    rel = np.max(np.abs(mine - np.array(res['per_record'])) / np.array(res['per_record']))
    print('[BB] check validation per_record vs result.json: max rel. difference %.1e' % rel, flush=True)
    assert rel < 1e-4, 'black box does not reproduce its validation score'

    te = load_records(cfg, te_f)
    pt = predict(te)
    nf = load_records(cfg, te_f, folder=str(DATA_NF))
    out = dict(records=np.array([record_name(f) for f in te_f]),
               y=np.stack([y[K0:] for _, _, _, y in te]),
               r=np.stack([r[K0:] for _, _, r, _ in te]),
               y_nf=np.stack([y[K0:] for _, _, _, y in nf]),
               bb=np.stack([p[K0:] for p in pt]))
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE / 'preds_BB.npz', code=BB_CODE, **out)
    rms = np.sqrt(np.mean((out['bb'] - out['y']) ** 2, axis=1))
    flo = np.sqrt(np.mean((out['y'] - out['y_nf']) ** 2, axis=1))
    for n, a, b in zip(out['records'], rms, flo):
        print('  %-3s BB %s | floor %s' % (n, ' '.join('%.3e' % v for v in a), ' '.join('%.3e' % v for v in b)))
    print('[BB] done in %.0f s' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    main()
