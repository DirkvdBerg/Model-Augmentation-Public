"""Predictions y_hat = r + e_hat of a trained black box on every split and the multisine twins, with the
noise-free output y_nf of each record (same loader, for the noise floor) and sigma_e, all from k0 = n + 1.
Writes preds_BB_<split>.npz, and preds_BB.npz for test (the name the figures read).

Usage: python -u Clean/blackbox/eval_bb.py <run folder> <output folder>
"""
__project_origin__ = "added"

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import deepSI
from deepSI.fit_systems.encoders import SS_encoder_general_hf

from bb import RunConfig, DATA_ROOT, data_cfg, load_records
from gantry_dynamic.data import DATASETS, record_files, servo_sigma

TWINS = os.path.join(DATA_ROOT, 'Multisine-test-twins')       # evaluation-only multisine twins


def predict(model, recs):
    """Free-run prediction of the black box.

    y_hat = r + e_hat, simulated from the state x^bb_0 = psi(u_past, e_past) over the whole record.

    Args:
        model: the black box.
        recs: (u, r, y) per record from `load_records`, physical.
    Returns:
        List of (N, 3) predictions y_hat, physical.
    """
    with np.errstate(all='ignore'):
        return [r + np.asarray(model.apply_experiment(deepSI.System_data(
                    u=np.ascontiguousarray(u), y=np.ascontiguousarray(y - r), dt=RunConfig.ts_new)).y, float)
                for u, r, y in recs]


def main():
    """Writes the predictions of one run folder."""
    t0 = time.time()
    run, out = Path(sys.argv[1]), Path(sys.argv[2])
    res = json.load(open(run / 'result.json'))
    k0, name = int(res['k0']), res['checkpoint']
    # 1. Best checkpoint: the run's own folder on Windows; deepSI's default folder on Linux, where the server
    # grid wrote it.
    ck = run / 'localappdata' / 'deepSI' / 'checkpoints' / name
    ck = ck if ck.is_file() else Path.home() / '.deepSI' / 'checkpoints' / name
    model = SS_encoder_general_hf.__new__(SS_encoder_general_hf)
    model.__dict__ = torch.load(ck, map_location='cpu', weights_only=False)
    print('[BB] %s, nx %d' % (ck, model.nx), flush=True)

    # 2. Splits: each with its noise-free twin folder.
    cfg = data_cfg()
    nf_dir = os.path.join(DATA_ROOT, DATASETS['noisefree'])
    with open(os.path.join(TWINS, DATASETS['lowpass'], 'MANIFEST.csv'), newline='') as fh:
        twins = ['%s/%s.mat' % (row['folder'], row['file']) for row in csv.DictReader(fh)]
    tr, va, te = record_files(cfg)
    splits = {'train': (tr, None, nf_dir), 'val': (va, None, nf_dir), 'test': (te, None, nf_dir),
              'twins': (twins, os.path.join(TWINS, DATASETS['lowpass']), os.path.join(TWINS, DATASETS['noisefree']))}
    # 3. Predict, check, write.
    out.mkdir(parents=True, exist_ok=True)
    for split, (files, folder, folder_nf) in splits.items():
        recs = load_records(files, cfg, folder)
        pr = predict(model, recs)
        if split == 'val':
            # Check: validation RMS_j equals result.json `per_record` (relative 1e-4).
            rms = np.array([np.sqrt(np.mean((p[k0:] - y[k0:]) ** 2, 0)) for p, (_, _, y) in zip(pr, recs)])
            rel = np.max(np.abs(rms - np.array(res['per_record'])) / np.array(res['per_record']))
            print('[BB] check validation per_record vs result.json: max rel. difference %.1e' % rel, flush=True)
            assert rel < 1e-4, 'black box does not reproduce its validation score'
        names = np.array([os.path.basename(f).split('_')[0] for f in files])
        o = dict(records=names, k0=k0, split=split, y=np.stack([y[k0:] for _, _, y in recs]),
                 r=np.stack([r[k0:] for _, r, _ in recs]), sigma_e=servo_sigma([y - r for _, r, y in recs], k0),
                 bb=np.stack([p[k0:] for p in pr]),
                 y_nf=np.stack([y[k0:] for _, _, y in load_records(files, cfg, folder_nf)]))
        np.savez_compressed(out / ('preds_BB.npz' if split == 'test' else 'preds_BB_%s.npz' % split),
                            code=name.rsplit('_', 2)[-2], **o)
        e = np.sqrt(np.mean((o['bb'] - o['y']) ** 2, axis=1))
        for n, a, sg in zip(names, e, o['sigma_e']):
            print('  %-5s %-4s BB %s | NRMS_e %s' % (split, n, ' '.join('%.3e' % v for v in a),
                                                     ' '.join('%.3f' % v for v in a / sg)))
    print('[BB] done in %.0f s' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    main()
