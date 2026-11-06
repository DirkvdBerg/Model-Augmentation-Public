"""Predictions y_hat = r + e_hat of a trained black box on every split and the multisine repeats, with the noise-free
output of each record for the noise floor (the error of the measurement noise alone, noisy minus noise-free y);
written as preds_BB_<split>.npz, and preds_BB.npz for test, the name the thesis figures read."""
__project_origin__ = "added"

import csv
import argparse
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from deepSI.fit_systems.encoders import SS_encoder_general_hf

from blackbox.bb import N_THREADS, data_config, load_records, predict
from gantry_dynamic.data import DATASETS, record_files, record_name, servo_sigma

RUN_DIR = None     # run folder with result.json; the first command-line argument overrides it
OUT_DIR = None     # output folder; None: <RUN_DIR>/evaluation; the second argument overrides it


def main(run, out, data_root=None):
    """Writes the predictions of one run folder.

    Args:
        run: run folder with result.json and the best checkpoint.
        out: folder of the .npz files.
    """
    t0 = time.time()
    with open(run / 'result.json') as fh:
        res = json.load(fh)
    cfg = data_config(res)
    if data_root is not None:
        cfg = replace(cfg, data_root=str(data_root))
    k0, name = int(res['k0']), res['checkpoint']
    # New runs hold the checkpoint directly; the other locations support existing Windows and server runs.
    candidates = (run / name, run / 'localappdata' / 'deepSI' / 'checkpoints' / name,
                  Path.home() / '.deepSI' / 'checkpoints' / name)
    ckpt = next((path for path in candidates if path.is_file()), None)
    if ckpt is None:
        raise FileNotFoundError('checkpoint not found; checked: %s' % ', '.join(map(str, candidates)))
    # the checkpoint holds the model's attributes; a bare SS_encoder_general_hf receives them, since prediction needs
    # only deepSI's methods, not the training class ServoWeightedEncoder
    model = SS_encoder_general_hf.__new__(SS_encoder_general_hf)
    model.__dict__ = torch.load(ckpt, map_location='cpu', weights_only=False)
    print('[BB] %s, nx %d' % (ckpt, model.nx), flush=True)

    # splits, each with its record folder (None: the noisy data set) and the folder of its noise-free versions
    nf_dir = os.path.join(cfg.data_root, DATASETS['noisefree'])
    # multisine repeats: the motion-only test records repeated with the multisine force added (block test-M);
    # evaluation only
    with open(os.path.join(cfg.data_root, DATASETS['noisy'], 'MANIFEST.csv'), newline='') as fh:
        repeats = ['%s/%s.mat' % (row['folder'], row['file']) for row in csv.DictReader(fh) if row['block'] == 'test-M']
    if 'records' in res:
        train_files, val_files, test_files = (res['records'][key] for key in ('train', 'val', 'test'))
    else:
        train_files, val_files, test_files = record_files(cfg)
    splits = {'train': (train_files, None, nf_dir), 'val': (val_files, None, nf_dir),
              'test': (test_files, None, nf_dir), 'multisine': (repeats, None, nf_dir)}
    # predictions from k0 on, the first sample after the encoder input, with the noise-free output for the floor
    out.mkdir(parents=True, exist_ok=True)
    for split, (files, folder, folder_nf) in splits.items():
        recs = load_records(files, folder, cfg)
        preds = predict(model, recs, cfg)
        names = np.array([record_name(f) for f in files])
        saved = dict(records=names, k0=k0, y=np.stack([y[k0:] for _, _, y in recs]),
                     r=np.stack([r[k0:] for _, r, _ in recs]), sigma_e=servo_sigma([y - r for _, r, y in recs], k0),
                     bb=np.stack([p[k0:] for p in preds]),
                     y_nf=np.stack([y[k0:] for _, _, y in load_records(files, folder_nf, cfg)]))
        # code: deepSI's checkpoint code, so a figure can tell which run it shows
        np.savez_compressed(out / ('preds_BB.npz' if split == 'test' else 'preds_BB_%s.npz' % split),
                            code=name.rsplit('_', 2)[-2], **saved)
        rms = np.sqrt(np.mean((saved['bb'] - saved['y']) ** 2, axis=1))
        for record, rms_r, sigma_r in zip(names, rms, saved['sigma_e']):
            print('  %-5s %-4s BB %s | NRMS_e %s' % (split, record, ' '.join('%.3e' % v for v in rms_r),
                                                     ' '.join('%.3f' % v for v in rms_r / sigma_r)))
    print('[BB] done in %.0f s' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Evaluate a black-box run using its saved data settings.')
    p.add_argument('run', nargs='?', type=Path, default=RUN_DIR)
    p.add_argument('out', nargs='?', type=Path, default=OUT_DIR)
    p.add_argument('--data-root', type=Path, help='data folder on this machine; preprocessing stays as saved')
    args = p.parse_args()
    if args.run is None:
        p.error('provide a run folder or set RUN_DIR at the top of this file')
    torch.set_num_threads(N_THREADS)
    main(args.run, args.out or args.run / 'evaluation', args.data_root)
