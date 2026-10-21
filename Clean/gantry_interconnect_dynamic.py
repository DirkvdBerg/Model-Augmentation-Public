"""Gantry augmentation training entry point (clean minimal code of the thesis runs).

    python -u Clean/gantry_interconnect_dynamic.py [preset] [noise] [--cpu] [--smoke]
        [--resume RUN_DIR [--from {best,last}] | --polish-only RUN_DIR] [--data-root P] [--out-root P]

No arguments = R1 lowpass on cuda with reduce-overhead compilation (the production run); without
CUDA use --cpu. Presets R1 to R7, R13, R14 are runs.tsv rows 221 to 230 (gantry_dynamic/config.py).
"""
import argparse
import json
import os
import shutil
from dataclasses import replace
from datetime import datetime

import numpy as np
import torch
from deepSI.datasets import get_work_dirs

from gantry_dynamic.config import (RunConfig, PRESETS, DATA_ROOT, OUT_ROOT, save_dir, config_json_dict,
                                   git_provenance)
from gantry_dynamic.data import load_datasets, compute_normalization, traj_dir
from gantry_dynamic.model import build_model, prefit_physical_encoder
from gantry_dynamic.controller import build_closed_loop
from gantry_dynamic.training import train, resolve_resume_hp, load_checkpoint


def _dumps(run_id):
    """The deepSI dumps `_best.pth`, `_last.pth` of a run id."""
    return [os.path.join(get_work_dirs()['checkpoints'], 'SSE_Interconnect_Composed_%s%s.pth' % (run_id, s))
            for s in ('_best', '_last')]


def _rerun_source(kind, run_dir, source):
    """D-275 D6: a rerun continues inside its run folder under the folder's run id. Returns
    (run_id, the source checkpoint in place); nothing moves here."""
    with open(os.path.join(run_dir, 'config.json')) as fh:
        run_id = str(json.load(fh)['run_id'])
    if kind == 'polish-only':
        src = os.path.join(run_dir, 'gantry_ckpt_%s_pre_lbfgs' % run_id)
        need = [src + '.pt', src + '.npz']
    else:
        src = _dumps(run_id)[0 if source == 'best' else 1]
        need = [src]
    for f in need:
        if not os.path.exists(f):
            raise FileNotFoundError('rerun source checkpoint not found: %s' % f)
    return run_id, src


def _archive_rerun(kind, run_dir, run_id, src):
    """D-275 D6: the previous outcome (and for a resume both deepSI dumps) moves to
    previous/<rerun_id>/, config.json gets a RESUMES entry. Called only once the source has loaded
    (D-279 F6). Returns the source's path in the archive."""
    cj_path = os.path.join(run_dir, 'config.json')
    with open(cj_path) as fh:
        meta = json.load(fh)
    rerun_id = os.environ.get('SLURM_JOB_ID') or datetime.now().strftime('%Y%m%d_%H%M%S')
    prev = os.path.join(run_dir, 'previous', rerun_id)
    os.makedirs(prev)
    for f in sorted(os.listdir(run_dir)):
        if f not in ('config.json', 'previous'):
            shutil.move(os.path.join(run_dir, f), os.path.join(prev, f))
    if kind == 'resume':
        for f in _dumps(run_id):
            if os.path.exists(f):
                shutil.move(f, os.path.join(prev, os.path.basename(f)))
    resume_ckpt = os.path.join(prev, os.path.basename(src))
    meta.setdefault('RESUMES', []).append(dict(
        time=datetime.now().isoformat(timespec='seconds'), type=kind,
        source=os.path.relpath(resume_ckpt, run_dir), code='clean', git=git_provenance(),
        rerun_id=rerun_id))
    with open(cj_path, 'w') as fh:
        json.dump(meta, fh, indent=2)
    return resume_ckpt


def main(cfg, run_id=None, rerun=None):
    """One run: data, model, closed loop, encoder start, training, polish, checkpoints.
    `rerun` = (kind, run_dir, source): ('resume', run_dir, 'best' | 'last') or ('polish-only', run_dir, None)."""
    if rerun is None:
        run_id = run_id or os.environ.get('SLURM_JOB_ID') or datetime.now().strftime('%Y%m%d_%H%M%S')
        sdir = os.path.join(save_dir(cfg), run_id)
        if os.path.exists(sdir):
            # D-279 (F1): a fresh run never writes into an existing run folder (D-275 D6)
            raise FileExistsError(
                'run folder %s exists. A fresh run never writes into an existing run: continue it with '
                '--resume %s, polish it with --polish-only %s, or launch under another run id.'
                % (sdir, sdir, sdir))
        src = None
    else:
        kind, sdir, source = rerun
        run_id, src = _rerun_source(kind, sdir, source)
        if kind == 'polish-only':
            cfg = replace(cfg, start_phase='lbfgs')
    print('[thesis run] arm=%s noise=%s start=%s n_a=%d seed=%d nf=%.3f s detune=%s%s'
          % (cfg.arm, cfg.noise, cfg.start, cfg.n_a, cfg.seed, cfg.nf_seconds, cfg.combo_init_detune,
             '  SMOKE' if cfg.ps2_smoke else '')
          + ' g_aug=mlp/live' + ('  PS2' if cfg.ps2 else '  plain')
          + ('  %s from %s' % (rerun[0], rerun[1]) if rerun else ''), flush=True)

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    data = load_datasets(cfg)
    norm = compute_normalization(cfg, data)

    hp = resolve_resume_hp(cfg, cfg.hp, src)

    if rerun is None:
        os.makedirs(sdir)
        with open(os.path.join(sdir, 'config.json'), 'w') as _f:
            json.dump({**config_json_dict(cfg, git=git_provenance()),
                       'hp': hp, 'run_id': run_id, 'data_dir': traj_dir(cfg)}, _f, indent=2)

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    fit_sys = build_model(hp, cfg, data, norm)

    fit_sys.simulator = build_closed_loop(
        fit_sys, norm, cfg,
        train_files=data.train_files, val_files=data.val_files, val_data=data.val_ckpt_data,
        e_train=data.e_train, e_val=data.e_val)
    print('\nLoss rollout: CLOSED loop (known controller wrapped around the model)')
    # D-271: the data-based encoder start needs the controller bank; a resume restores the
    # fitted encoder instead.
    resume = None
    if src is None:
        prefit_physical_encoder(fit_sys, cfg, data, norm)
    else:
        # D-279 (F6): data, device and the whole checkpoint load (format, dtype, architecture, PS2 state,
        # Adam groups) succeed BEFORE anything in the run folder moves
        meta = load_checkpoint(fit_sys, src, cfg.joint_estimation, start_phase=cfg.start_phase)
        resume = (_archive_rerun(rerun[0], sdir, run_id, src), meta)

    bestfit = train(fit_sys, hp, cfg, data, resume=resume, checkpoint_dir=sdir, run_id=run_id)
    print(f"\nTraining complete. Best validation {cfg.validation_measure}: {bestfit:.6g}")
    return bestfit


def cli():
    ap = argparse.ArgumentParser(description='no arguments = R1 lowpass on cuda with reduce-overhead '
                                             'compilation (the production run); without CUDA use --cpu')
    ap.add_argument('preset', nargs='?', default=None, choices=sorted(PRESETS))
    ap.add_argument('noise', nargs='?', default=None, choices=('lowpass', 'noisefree'))
    ap.add_argument('--cpu', action='store_true', help='device cpu, eager')
    ap.add_argument('--smoke', action='store_true', help='launch test: 2 s records, 4 updates, 2-iteration polish')
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument('--resume', metavar='RUN_DIR', help='continue the Adam phase of this run folder')
    grp.add_argument('--polish-only', metavar='RUN_DIR', help='polish the pre_lbfgs pair of this run folder')
    ap.add_argument('--from', dest='source', choices=('best', 'last'), default=None,
                    help='deepSI dump a --resume starts from (default best)')
    ap.add_argument('--data-root', default=DATA_ROOT)
    ap.add_argument('--out-root', default=OUT_ROOT)
    a = ap.parse_args()
    run_dir = a.resume or a.polish_only
    if run_dir is None:
        if a.source is not None:
            ap.error('--from applies to --resume only')
        cfg = RunConfig(**PRESETS[a.preset or 'R1'], noise=a.noise or 'lowpass',
                        data_root=a.data_root, out_root=a.out_root)
        rerun = None
    else:
        if a.preset is not None or a.noise is not None:
            ap.error('a run folder fixes the configuration; give no preset or noise with it')
        if a.polish_only and a.source is not None:
            ap.error('--from applies to --resume only')
        with open(os.path.join(run_dir, 'config.json')) as fh:
            cfg = RunConfig.from_config_json(json.load(fh), data_root=a.data_root, out_root=a.out_root)
        rerun = ('resume', run_dir, a.source or 'best') if a.resume else ('polish-only', run_dir, None)
    if a.cpu:
        cfg = replace(cfg, device='cpu', compile_mode=None)
    if a.smoke:
        cfg = replace(cfg, truncate_s=2.0, compile_mode=None, n_its=4, its_per_val=4,
                      lbfgs_inner_iter=1, lbfgs_max_iter=2, lbfgs_windows=256, ps2_smoke=True)
    main(cfg, rerun=rerun)


if __name__ == '__main__':
    cli()
