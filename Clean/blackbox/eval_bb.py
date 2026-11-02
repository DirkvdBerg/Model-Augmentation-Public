"""Predictions y_hat = r + e_hat of a trained black box on every split and the multisine twins, with the noise-free
output of each record for the noise floor; written as preds_BB_<split>.npz, and preds_BB.npz for test."""
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

from bb import data_cfg, load_records
from gantry_dynamic.config import RunConfig, DATA_ROOT
from gantry_dynamic.data import DATASETS, record_files, servo_sigma

RUN_DIR = None     # run folder with result.json; the first command-line argument overrides it
OUT_DIR = None     # output folder; None: <RUN_DIR>/evaluation; the second argument overrides it
N_THREADS = 4      # CPU threads, the server's cores per task
TWINS_DIR = os.path.join(DATA_ROOT, 'Multisine-test-twins')   # multisine twins, used only in the evaluation


def predict(model, recs):
    """Free-run prediction y_hat = r + e_hat over the whole record, from the encoder state x^bb_0 = psi(u_past, e_past).

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


def main(run, out):
    """Writes the predictions of one run folder.

    Args:
        run: run folder with result.json and the best checkpoint.
        out: output folder.
    """
    t0 = time.time()
    with open(run / 'result.json') as fh:
        res = json.load(fh)
    k0, name = int(res['k0']), res['checkpoint']
    # 1. Load the best checkpoint: in the run's own folder on Windows, in deepSI's default folder on Linux, where the
    #    server grid wrote it.
    ckpt = run / 'localappdata' / 'deepSI' / 'checkpoints' / name
    ckpt = ckpt if ckpt.is_file() else Path.home() / '.deepSI' / 'checkpoints' / name
    model = SS_encoder_general_hf.__new__(SS_encoder_general_hf)
    model.__dict__ = torch.load(ckpt, map_location='cpu', weights_only=False)
    print('[BB] %s, nx %d' % (ckpt, model.nx), flush=True)

    # 2. Splits, each with the folder of its noise-free twins.
    cfg = data_cfg()
    nf_dir = os.path.join(DATA_ROOT, DATASETS['noisefree'])
    with open(os.path.join(TWINS_DIR, DATASETS['lowpass'], 'MANIFEST.csv'), newline='') as fh:
        twins = ['%s/%s.mat' % (row['folder'], row['file']) for row in csv.DictReader(fh)]
    train_files, val_files, test_files = record_files(cfg)
    splits = {'train': (train_files, None, nf_dir), 'val': (val_files, None, nf_dir),
              'test': (test_files, None, nf_dir),
              'twins': (twins, os.path.join(TWINS_DIR, DATASETS['lowpass']),
                        os.path.join(TWINS_DIR, DATASETS['noisefree']))}
    # 3. Predict each split and write it from k0 on, with the noise-free output for the floor.
    out.mkdir(parents=True, exist_ok=True)
    for split, (files, folder, folder_nf) in splits.items():
        recs = load_records(files, cfg, folder)
        preds = predict(model, recs)
        names = np.array([os.path.basename(f).split('_')[0] for f in files])
        saved = dict(records=names, k0=k0, y=np.stack([y[k0:] for _, _, y in recs]),
                     r=np.stack([r[k0:] for _, r, _ in recs]), sigma_e=servo_sigma([y - r for _, r, y in recs], k0),
                     bb=np.stack([p[k0:] for p in preds]),
                     y_nf=np.stack([y[k0:] for _, _, y in load_records(files, cfg, folder_nf)]))
        # code: deepSI's checkpoint code, so a figure can tell which run it shows
        np.savez_compressed(out / ('preds_BB.npz' if split == 'test' else 'preds_BB_%s.npz' % split),
                            code=name.rsplit('_', 2)[-2], **saved)
        rms = np.sqrt(np.mean((saved['bb'] - saved['y']) ** 2, axis=1))
        for record, rms_r, sigma_r in zip(names, rms, saved['sigma_e']):
            print('  %-5s %-4s BB %s | NRMS_e %s' % (split, record, ' '.join('%.3e' % v for v in rms_r),
                                                     ' '.join('%.3f' % v for v in rms_r / sigma_r)))
    print('[BB] done in %.0f s' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(N_THREADS)
    run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else RUN_DIR)
    out_dir = sys.argv[2] if len(sys.argv) > 2 else OUT_DIR
    main(run_dir, Path(out_dir) if out_dir else run_dir / 'evaluation')
