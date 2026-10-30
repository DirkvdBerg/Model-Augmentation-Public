"""Build the layer-4 staging folder of design 6.5 from the server's final run folders (mainline side).

    python -u l4_stage.py --src <repo>/simulations/gantry_subnet/thesis_taf_lowpass_noise_data_lsq \
                          --dst <repo>/simulations/clean-golden/L4-staging [--middle 221,222]

Copies, never moves: the original server folders stay untouched. Every row whose final model D-274
does not change is copied as is (config.json, both pairs, val_hist.npz, gantry_results_<job>.npz).
A row given in --middle (221 or 222 when its [lbfgs] line shows ACCEPTED with val_after at or above
the checkpoint's bestfit but below the rebuilt score) is staged as the D-274 final model:
gantry_ckpt_<job>.pt = the rebuilt `_pre_lbfgs` weights; gantry_ckpt_<job>.npz = the original with
bestfit = that model's validation sim-NRMSe measured here by the mainline evaluation code
(eval_greybox build + scoring) and lbfgs_outcome marked rejected by D-274; no gantry_results file
(it scores the polished model). meta.json records what was staged and why.
"""
import argparse
import json
import os
import shutil
import sys

import hcommon as H

JOBS = {221: 89792, 222: 89796, 223: 89797, 224: 89793, 225: 89798, 226: 89799, 227: 89800,
        229: 89794, 230: 89795}

ap = argparse.ArgumentParser()
ap.add_argument('--src', required=True)
ap.add_argument('--dst', required=True)
ap.add_argument('--middle', default='', help='comma list of rows (221, 222) staged as the rebuilt model')
args = ap.parse_args()
middle = [int(r) for r in args.middle.split(',') if r]
assert set(middle) <= {221, 222}, middle
dst = os.path.abspath(args.dst)
if os.path.basename(dst) == os.path.basename(os.path.abspath(args.src)):
    sys.exit('the staging folder must not carry the server family name (eval cache name, 6.5)')
if os.path.exists(dst):
    sys.exit('%s exists' % dst)
os.makedirs(dst)
meta = {'src': os.path.abspath(args.src), 'rows': {}}
for row, job in JOBS.items():
    s, d = os.path.join(args.src, str(job)), os.path.join(dst, str(job))
    os.makedirs(d)
    names = [f for f in os.listdir(s) if os.path.isfile(os.path.join(s, f))
             and (f == 'config.json' or f.startswith(('gantry_ckpt_%d' % job, 'val_hist', 'gantry_results_%d' % job)))]
    for f in names:
        shutil.copy2(os.path.join(s, f), os.path.join(d, f))
    meta['rows'][row] = {'job': job, 'files': sorted(names), 'staged_as': 'copy'}
    if row in middle:
        os.remove(os.path.join(d, 'gantry_ckpt_%d.pt' % job))
        shutil.copy2(os.path.join(s, 'gantry_ckpt_%d_pre_lbfgs.pt' % job), os.path.join(d, 'gantry_ckpt_%d.pt' % job))
        if os.path.exists(os.path.join(d, 'gantry_results_%d.npz' % job)):
            os.remove(os.path.join(d, 'gantry_results_%d.npz' % job))
        meta['rows'][row]['staged_as'] = 'D-274 rebuilt model (intended deviation)'

if middle:
    # the rebuilt model's validation score, by the mainline evaluation code
    H.select_side('mainline')
    import numpy as np
    import torch
    os.environ['THESIS_EVAL'] = 'final'
    os.environ['THESIS_RUNS_DIR'] = dst
    sys.path.insert(0, os.path.join(H.REPO, 'Thesis-writeup', 'Code', 'Evaluation'))
    from pathlib import Path
    from eval_greybox import run_config, build
    from gantry_dynamic.training import load_checkpoint
    from gantry_dynamic.controller import build_controller_bank
    from gantry_dynamic.data import servo_sigma
    from model_augmentation.fit_systems.closed_loop import closed_loop_free_run_rms_batch
    for row in middle:
        job = JOBS[row]
        d = os.path.join(dst, str(job))
        cfg = run_config(Path(d))
        data, norm, fs = build(cfg)
        base = os.path.join(d, 'gantry_ckpt_%d' % job)
        orig = dict(np.load(os.path.join(args.src, str(job), 'gantry_ckpt_%d.npz' % job), allow_pickle=True))
        np.savez(base + '.npz', **orig)                       # pair complete before loading
        load_checkpoint(fs, base, joint_estimation=True, start_phase='lbfgs')
        bank, rows, _ = build_controller_bank(data.val_files, cfg.ts_new, ystd=norm.ystd, std_u=norm.std_u,
                                              dtype=cfg.dtype_pt, use_gain=cfg.controller_gain)
        k0 = max(fs.na, fs.nb)
        with torch.no_grad():
            _, _, preds = closed_loop_free_run_rms_batch(fs, data.val_list, bank, rows, return_pred=True)
        y = np.stack([np.asarray(sd.y, float)[k0:] for sd in data.val_list])
        pred = np.stack([np.asarray(p, float) for p in preds])
        score = float(np.mean(np.sqrt(np.mean((pred - y) ** 2, axis=1)) / servo_sigma(data.e_val, k0)))
        out = json.loads(str(orig['lbfgs_outcome']))
        out['accepted'] = False
        out['reason'] = '%s | D-274: rejected (val_after %.6e not below the checkpoint bestfit %.6e)' % (
            out.get('reason', ''), out.get('val_after', float('nan')), float(orig['bestfit_selected']))
        orig.update(bestfit=np.array(score), lbfgs_outcome=np.array(json.dumps(out)),
                    optimizer_state_dropped=np.array(False))
        np.savez(base + '.npz', **orig)
        meta['rows'][row].update(rebuilt_score=score, original_bestfit=float(np.load(
            os.path.join(args.src, str(job), 'gantry_ckpt_%d.npz' % job))['bestfit']))
        print('[stage] %d (job %d): rebuilt model staged, bestfit %.6e' % (row, job, score), flush=True)
with open(os.path.join(dst, 'meta.json'), 'w') as fh:
    json.dump(meta, fh, indent=1)
print('L4_STAGE_DONE %s middle=%s' % (dst, middle or 'none'))
