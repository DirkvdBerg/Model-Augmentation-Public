"""Launch a fresh thesis run, resume, or polish-only rerun; defaults to R1 lowpass on CUDA."""
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


def _dumps(run_id):
    """Paths of the deepSI dumps _best.pth and _last.pth of a run; deepSI names them after the model class."""
    return [os.path.join(get_work_dirs()['checkpoints'], 'AugmentedModel_%s%s.pth' % (run_id, s))
            for s in ('_best', '_last')]


def _rerun_source(kind, run_dir, source):
    """(config.json, run id, source checkpoint) of a rerun, which continues in its run folder under the
    folder's run id; nothing moves here."""
    with open(os.path.join(run_dir, 'config.json')) as fh:
        cj = json.load(fh)
    run_id = str(cj['run_id'])
    if kind == 'polish-only':
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


def main(cfg, run_id=None, rerun=None):
    """Build and train a fresh run or the requested resume/polish-only rerun; return its final validation score."""
    # 1. Run folder.
    # rerun = ('resume', run_dir, 'best' | 'last') or ('polish-only', run_dir, None); None for a fresh run
    # run folder: a fresh run never writes into an existing one
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
    print('[thesis run] arm=%s noise=%s start=%s n_a=%d seed=%d nf=%.3f s detune=%s%s'
          % (cfg.arm, cfg.noise, cfg.start, cfg.n_a, cfg.seed, cfg.nf_seconds, cfg.combo_init_detune,
             '  SMOKE' if cfg.ps2_smoke else '')
          + ' g_aug=mlp/live' + ('  PS2' if cfg.ps2 else '  plain')
          + ('  %s from %s' % (rerun[0], rerun[1]) if rerun else ''), flush=True)

    # 2. Data and normalisation.
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    data = load_datasets(cfg)
    norm = compute_normalization(cfg, data)

    hp = cfg.hp

    if rerun is None:
        os.makedirs(sdir)
        with open(os.path.join(sdir, 'config.json'), 'w') as fh:
            json.dump({**config_json_dict(cfg, git=git_provenance()),
                       'hp': hp, 'run_id': run_id, 'data_dir': traj_dir(cfg)}, fh, indent=2)

    # 3. Model and closed loop.
    # reseed before construction, so loading the data cannot alter the model's initial draws
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    fit_sys = build_model(hp, cfg, data, norm)

    fit_sys.simulator = build_closed_loop(
        fit_sys, norm, cfg,
        train_files=data.train_files, val_files=data.val_files, val_data=data.val_ckpt_data,
        e_train=data.e_train, e_val=data.e_val)
    print('\nLoss rollout: closed loop (the known controller around the model)')

    # 4. Encoder start, or the restored checkpoint.
    # the encoder prefit needs the controller bank to simulate the baseline in closed loop
    resume = None
    if src is None:
        prefit_physical_encoder(fit_sys, cfg, data, norm)
    else:
        # the run folder is archived only once data, device and the whole checkpoint have loaded
        meta = load_checkpoint(fit_sys, src, cfg.joint_estimation, start_phase=cfg.start_phase)
        resume = (_archive_rerun(rerun[0], sdir, cj, run_id, src), meta)

    # 5. Training.
    bestfit = train(fit_sys, hp, cfg, data, resume=resume, checkpoint_dir=sdir, run_id=run_id)
    print(f"\nTraining complete. Best validation {cfg.validation_measure}: {bestfit:.6g}")
    return bestfit


def cli():
    """Parse the command line into a RunConfig and a rerun, then start main."""
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
    ap.add_argument('--closed-form', action='store_true',
                    help='baseline derivative as (qdot, a) directly instead of through the LFR output equation; '
                         'results differ from the thesis runs by rounding')
    ap.add_argument('--data-root', default=DATA_ROOT)
    ap.add_argument('--out-root', default=OUT_ROOT)
    a = ap.parse_args()
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
    if a.smoke:
        cfg = replace(cfg, truncate_s=2.0, compile_mode=None, n_its=4, its_per_val=4,
                      lbfgs_inner_iter=1, lbfgs_max_iter=2, lbfgs_windows=256, ps2_smoke=True)
    main(cfg, rerun=rerun)


if __name__ == '__main__':
    cli()
