"""Launch a fresh thesis run, resume, or polish-only rerun, then evaluate it; defaults to R1 lowpass on CUDA."""
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
from gantry_dynamic.training import train, load_checkpoint
from gantry_dynamic.evaluation import evaluate
from model_augmentation.fit_systems.closed_loop import SCORE_NAME

EVALUATE = True        # evaluate the run after training
EVAL_RUN_DIR = None    # a run folder: evaluate only that run, no training


def _dumps(run_id):
    """Paths of the deepSI dumps _best.pth and _last.pth of a run; deepSI names them after the model class and keeps
    them in its checkpoints folder (get_work_dirs), outside the run folder; a resume moves them into previous/."""
    return [os.path.join(get_work_dirs()['checkpoints'], 'AugmentedModel_%s%s.pth' % (run_id, s))
            for s in ('_best', '_last')]


def _rerun_source(kind, run_dir, source):
    """(config.json, run id, source checkpoint) of a rerun, which continues in its run folder under the
    folder's run id."""
    with open(os.path.join(run_dir, 'config.json')) as fh:
        cj = json.load(fh)
    run_id = str(cj['run_id'])
    if kind == 'polish-only':
        # the pre-polish pair: the Adam-selected weights with their Adam moments (training.train, step 3)
        src = os.path.join(run_dir, 'gantry_ckpt_%s_pre_lbfgs' % run_id)
        need = [src + '.pt', src + '.npz']
    else:
        src = _dumps(run_id)[0 if source == 'best' else 1]
        need = [src]
    for f in need:
        if not os.path.exists(f):
            raise FileNotFoundError('rerun source %s not found; check RUN_DIR, or choose the other dump of a '
                                    '--resume with --from' % f)
    return cj, run_id, src


def _archive_rerun(kind, run_dir, cj, run_id, src):
    """Move the previous outcome, and for a resume both deepSI dumps, to previous/<rerun_id>/ and add a
    RESUMES entry to config.json; returns the source's path in the archive."""
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
    cj.setdefault('RESUMES', []).append(dict(
        time=datetime.now().isoformat(timespec='seconds'), type=kind,
        source=os.path.relpath(resume_ckpt, run_dir), code='clean', git=git_provenance(),
        rerun_id=rerun_id))
    with open(os.path.join(run_dir, 'config.json'), 'w') as fh:
        json.dump(cj, fh, indent=2)
    return resume_ckpt


def main(cfg, run_id=None, rerun=None, evaluate_run=EVALUATE):
    """Build and train a fresh run or the requested resume/polish-only rerun, evaluate it when evaluate_run; return
    its final validation score."""
    # 1. Run folder.
    # rerun = ('resume', run_dir, 'best' | 'last') or ('polish-only', run_dir, None); None for a fresh run
    if rerun is None:
        run_id = run_id or os.environ.get('SLURM_JOB_ID') or datetime.now().strftime('%Y%m%d_%H%M%S')
        sdir = os.path.join(save_dir(cfg), run_id)
        if os.path.exists(sdir):
            raise FileExistsError(
                'run folder %s exists; continue it with --resume %s, polish it with --polish-only %s, or '
                'launch under another run id' % (sdir, sdir, sdir))
        src = None
    else:
        kind, sdir, source = rerun
        cj, run_id, src = _rerun_source(kind, sdir, source)
        if kind == 'polish-only':
            cfg = replace(cfg, start_phase='lbfgs')
    print('[thesis run] arm=%s noise=%s start=%s n_a=%d seed=%d nf=%.3f s detune=%s'
          % (cfg.arm, cfg.noise, cfg.start, cfg.n_a, cfg.seed, cfg.nf_seconds, cfg.combo_init_detune)
          + ('  PS2' if cfg.ps2 else '  plain')
          + ('  %s from %s' % (rerun[0], rerun[1]) if rerun else ''), flush=True)

    # 2. Data and normalisation.
    data = load_datasets(cfg)
    norm = compute_normalization(cfg, data)

    hp = cfg.hp

    if rerun is None:
        os.makedirs(sdir)
        with open(os.path.join(sdir, 'config.json'), 'w') as fh:
            json.dump({**config_json_dict(cfg, git=git_provenance()),
                       'run_id': run_id, 'data_dir': traj_dir(cfg)}, fh, indent=2)

    # 3. Model and closed loop.
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    fit_sys = build_model(hp, cfg, data, norm)

    fit_sys.simulator = build_closed_loop(
        fit_sys, norm, cfg,
        train_files=data.train_files, val_files=data.val_files, e_train=data.e_train, e_val=data.e_val)
    print('\nLoss rollout: closed loop (the known controller around the model)')

    # 4. Encoder start, or the restored checkpoint.
    # the encoder prefit needs the controller bank to simulate the baseline in closed loop
    resume = None
    if src is None:
        prefit_physical_encoder(fit_sys, cfg, data, norm)
    else:
        # the run folder is archived only once data, device and the whole checkpoint have loaded
        meta = load_checkpoint(fit_sys, src, start_phase=cfg.start_phase)
        resume = (_archive_rerun(rerun[0], sdir, cj, run_id, src), meta)

    # 5. Training.
    bestfit = train(fit_sys, hp, cfg, data, resume=resume, checkpoint_dir=sdir, run_id=run_id)
    print(f"\nTraining complete. Best validation {SCORE_NAME}: {bestfit:.6g}")

    # 6. Evaluation of the final checkpoint gantry_ckpt_<run_id>: the accepted polish, else the Adam-selected model.
    if evaluate_run:
        evaluate(sdir, data_root=cfg.data_root)
    return bestfit


def cli():
    """Parse the command line into a RunConfig and a rerun, then start main."""
    ap = argparse.ArgumentParser(description='no arguments = R1 lowpass on cuda with reduce-overhead '
                                             'compilation (the production run); without CUDA use --cpu')
    ap.add_argument('preset', nargs='?', default=None, choices=sorted(PRESETS))
    ap.add_argument('noise', nargs='?', default=None, choices=('lowpass', 'noisefree'))
    ap.add_argument('--cpu', action='store_true', help='device cpu, eager')
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument('--resume', metavar='RUN_DIR', help='continue the Adam phase of this run folder')
    grp.add_argument('--polish-only', metavar='RUN_DIR', help='polish the pre_lbfgs pair of this run folder')
    grp.add_argument('--evaluate', metavar='RUN_DIR', default=EVAL_RUN_DIR,
                     help='evaluate the final checkpoint of this run folder, no training')
    ap.add_argument('--no-evaluate', action='store_true', help='no evaluation after training')
    # best: the dump of the lowest validation score; last: the dump of the latest validation, to continue the latest
    # weights; both carry their Adam moments, neither the batch stream nor an unfinished PS2 opening
    ap.add_argument('--from', dest='source', choices=('best', 'last'), default=None,
                    help='deepSI dump a --resume starts from (default best)')
    # skips the LFR output equation (A_combined, w = Y z); the acceleration a is the same
    ap.add_argument('--closed-form', action='store_true',
                    help='baseline derivative as (qdot, a) directly instead of through the LFR output equation; '
                         'results differ from the thesis runs by rounding')
    ap.add_argument('--data-root', default=DATA_ROOT)
    ap.add_argument('--out-root', default=OUT_ROOT)
    a = ap.parse_args()
    if a.evaluate is not None:
        if a.preset is not None or a.noise is not None:
            ap.error('a run folder fixes the configuration; give no preset or noise with --evaluate')
        evaluate(a.evaluate, data_root=a.data_root)
        return
    if a.source is not None and not a.resume:
        ap.error('--from applies to --resume only')
    run_dir = a.resume or a.polish_only
    if run_dir is None:
        cfg = RunConfig(**PRESETS[a.preset or 'R1'], noise=a.noise or 'lowpass',
                        baseline_form='closed' if a.closed_form else 'lfr',
                        data_root=a.data_root, out_root=a.out_root)
        rerun = None
    else:
        if a.preset is not None or a.noise is not None or a.closed_form:
            ap.error('a run folder fixes the configuration; give no preset, noise or --closed-form with it')
        with open(os.path.join(run_dir, 'config.json')) as fh:
            cfg = RunConfig.from_config_json(json.load(fh), data_root=a.data_root, out_root=a.out_root)
        rerun = ('resume', run_dir, a.source or 'best') if a.resume else ('polish-only', run_dir, None)
    if a.cpu:
        cfg = replace(cfg, device='cpu', compile_mode=None)
    main(cfg, rerun=rerun, evaluate_run=EVALUATE and not a.no_evaluate)


if __name__ == '__main__':
    cli()
