"""Train the augmented gantry model with the settings below; or, on an existing run folder, continue its Adam
training from a saved checkpoint up to the epoch budget, rerun its L-BFGS polish, or only evaluate its final
checkpoint. The default is the thesis model M_OBC (OBC arm, detuned start, two added states, PS opening, lowpass
data) on CUDA."""
import argparse
import json
import os
import shutil
from dataclasses import replace
from datetime import datetime

import numpy as np
import torch
from deepSI.datasets import get_work_dirs

from gantry_dynamic.config import RunConfig, DATA_ROOT, OUT_ROOT, save_dir, config_json_dict, git_provenance
from gantry_dynamic.data import load_datasets, compute_normalization, traj_dir
from gantry_dynamic.model import build_model, prefit_physical_encoder
from gantry_dynamic.controller import build_closed_loop
from gantry_dynamic.training import train, load_checkpoint
from gantry_dynamic.evaluation import evaluate
from model_augmentation.fit_systems.closed_loop import SCORE_NAME

# What to run; --resume, --polish-only and --evaluate RUN_DIR override MODE and RUN_DIR
MODE = 'fresh'         # 'fresh': train a new run with SETTINGS; 'resume': continue the Adam training of RUN_DIR;
                       # 'polish-only': polish the pre-polish checkpoint of RUN_DIR again; 'evaluate': evaluate RUN_DIR
RUN_DIR = None         # run folder of resume, polish-only and evaluate; its config.json fixes the settings
RESUME_FROM = 'best'   # resume: the deepSI checkpoint at the lowest validation score ('best') or the latest ('last')
EVALUATE = True        # evaluate the final checkpoint after training

# Run settings of a fresh run, recorded in its config.json; with the values of the thesis run M_OBC
SETTINGS = dict(
    # model
    arm='obc',          # 'obc': OBC projection of the network; 'u': unconstrained
    start='detuned',    # baseline parameters at the start: 'correct' (nominal) or 'detuned' (tests their recovery)
    n_a=2,              # number of added states; the thesis runs 0, 2 and 8. The baseline has 2 missing states.
    ps2=True,           # PS initialization of the added states (S1, then S2)
    noise='lowpass',    # data set: 'lowpass' (measurement noise) or 'noisefree'
    # sampling and windows
    fs_new=4000,        # [Hz] training rate; fast enough for the controller, 5x cheaper than the 20 kHz data
    # zero-phase anti-aliasing before decimation to fs_new, so y stays aligned with r and u
    aa_order=8,         # order of the Butterworth low-pass on y - r
    aa_fc_hz=1600.0,    # [Hz] its cutoff
    aa_trim_s=0.005,    # [s] cut from both record ends: the filter has no samples beyond a record's ends, so its
                        # output is wrong in the first and last 4.6 ms
    nf_seconds=0.100,   # [s] training window; twice the 50 ms settling time of the closed loop
    stride=10,          # samples between window starts
    na_nb=30,           # past samples of y and u in the encoder input; Hoekstra's 2 n_x + 1 for n_a = 8, plus one
                        # because the window ends at tau - 1; Kept at 30 for n_a = 2.
    seed=1,             # seed of weights, detune signs, polish batch, PS opening
    detune=0.10,        # relative detuning of each baseline parameter
    # networks: 2 hidden layers of 16 nodes, as Hoekstra et al. (2026)
    n_nodes_per_layer=16,  # hidden layer width
    n_hidden_layers=2,     # hidden tanh layers
    up_sample=1,           # RK4 steps per sample
    # Adam
    batch_size=512,     # windows per update
    lr=1e-5,            # learning rate; compared different orders of magnitude
    adam_eps=1e-16,     # Adam epsilon; with the 1e-8 default the hidden layers do not train, gradient issue
    epochs=200,         # epochs; limited by compute
    n_its=None,         # budget in updates instead of epochs; None: epochs
    its_per_val=1300,   # updates between validations; full record evaluated keep val minimal
    # L-BFGS polish
    lbfgs_max_iter=100,   # iterations in total
    lbfgs_windows=16384,  # training windows the polish fits, drawn once; lower it if the polish runs out of memory
    # PS initialization: 
    # S1 trains the encoder so the added states predict the model's output error, scored by nu, the share
    # of that error left unexplained on held-out records
    # S2 then fits the network's added-state update to those states, scored by its cost on held-out pairs
    ps2_opening=dict(
        aux_batch=64, lr=1e-3,                        # S1: training windows per step, learning rate of its own Adam
        eval_every=25, s1_min=100, s1_cap=350,        # S1: updates between nu values, earliest stop, latest stop
        s1_tol=0.02, s1_window=4,                     # S1 stops when none of its last s1_window nu improves the prior
                                                      # best by at least s1_tol (relative)
        screen_at=150, screen_max=0.90, n_cal=128,    # S2 is skipped if nu > screen_max at update screen_at or at the
                                                      # end of S1; held-out windows nu is computed on
        n_snap=8192, n_snap_cal=2048, s2_batch=512,   # S2: pairs (state at k, added states at k + 1) fitted, held out,
                                                      # per step
        s2_check=100, s2_min=500, s2_cap=8000,        # S2: steps between held-out costs, earliest stop, latest stop
        s2_tol=0.01, s2_window=5,                     # S2 stops when none of its last s2_window costs improves the
                                                      # prior best by at least s2_tol (relative)
    ),
    # environment
    device='cuda',                   # 'cuda' or 'cpu'
    compile_mode='reduce-overhead',  # torch.compile mode on CUDA; None: eager; 6 to 7x faster than eager
    truncate_s=None,                 # [s] keep only the start of each record; None: whole records
    baseline_form='lfr',             # baseline derivative: 'lfr' evaluates the LFR output equation through its latent
                                     # signals, as in the thesis runs; 'closed' returns (qdot, a) directly, same up to
                                     # rounding and 3 to 8 % faster
    obc_ref_chunk=16384,             # reference tuples per Jacobian call in the OBC basis build; lower it if that
                                     # build runs out of memory, the basis stays the same
    data_root=DATA_ROOT,             # data set folder
    out_root=OUT_ROOT,               # run folder root
)


def _deepsi_checkpoint_paths(run_id):
    """Paths of the checkpoints deepSI saves during Adam, _best.pth and _last.pth, in deepSI's checkpoints folder."""
    return [os.path.join(get_work_dirs()['checkpoints'], 'AugmentedModel_%s%s.pth' % (run_id, s))
            for s in ('_best', '_last')]


def _rerun_checkpoint(kind, run_dir, source):
    """Find the checkpoint a resume or polish-only run starts from and check that its files exist.

    Args:
        kind: 'resume' (a deepSI checkpoint saved during Adam) or 'polish-only' (the pre-polish pair).
        run_dir: the existing run folder.
        source: 'best' or 'last' deepSI checkpoint for a resume; unused for polish-only.
    Returns:
        (config.json contents, run id, checkpoint path).
    """
    with open(os.path.join(run_dir, 'config.json')) as fh:
        cj = json.load(fh)
    run_id = str(cj['run_id'])
    if kind == 'polish-only':
        # the pre-polish pair: the Adam-selected weights and their Adam moments
        src = os.path.join(run_dir, 'gantry_ckpt_%s_pre_lbfgs' % run_id)
        need = [src + '.pt', src + '.npz']
    else:
        # the deepSI checkpoint at the lowest validation score ('best') or the latest one ('last')
        src = _deepsi_checkpoint_paths(run_id)[0 if source == 'best' else 1]
        need = [src]
    for f in need:
        if not os.path.exists(f):
            raise FileNotFoundError('checkpoint %s not found; check RUN_DIR, or resume from the other deepSI '
                                    'checkpoint with --from' % f)
    return cj, run_id, src


def _archive_previous_outcome(kind, run_dir, cj, run_id, src):
    """Move a run folder's previous outcome to previous/<rerun id>/ before a rerun and log the rerun in config.json.

    Args:
        kind: 'resume' or 'polish-only'.
        run_dir: the run folder.
        cj: its config.json contents; the rerun is appended to RESUMES.
        run_id: run id in the checkpoint names.
        src: the checkpoint the rerun starts from (from _rerun_checkpoint).
    Returns:
        the path of that checkpoint after the move.
    """
    # 1. Archive folder, named after the SLURM job id or the time.
    rerun_id = os.environ.get('SLURM_JOB_ID') or datetime.now().strftime('%Y%m%d_%H%M%S')
    prev = os.path.join(run_dir, 'previous', rerun_id)
    os.makedirs(prev)
    # 2. Move every output except config.json, so the rerun writes into an empty folder.
    for f in sorted(os.listdir(run_dir)):
        if f not in ('config.json', 'previous'):
            shutil.move(os.path.join(run_dir, f), os.path.join(prev, f))
    # 3. A resume also writes new deepSI checkpoints, so the old ones move as well.
    if kind == 'resume':
        for f in _deepsi_checkpoint_paths(run_id):
            if os.path.exists(f):
                shutil.move(f, os.path.join(prev, os.path.basename(f)))
    # 4. Log the rerun in config.json: time, kind, start checkpoint, rerun id and the git commit it ran with.
    resume_ckpt = os.path.join(prev, os.path.basename(src))
    cj.setdefault('RESUMES', []).append(dict(
        time=datetime.now().isoformat(timespec='seconds'), type=kind,
        source=os.path.relpath(resume_ckpt, run_dir), code='clean', git=git_provenance(),
        rerun_id=rerun_id))
    with open(os.path.join(run_dir, 'config.json'), 'w') as fh:
        json.dump(cj, fh, indent=2)
    return resume_ckpt


def main(cfg, run_id=None, rerun=None, evaluate_run=EVALUATE):
    """Train one run, fresh or a rerun, then evaluate it.

    Args:
        cfg: RunConfig of the run.
        run_id: name of a fresh run folder; None takes the SLURM job id or the time.
        rerun: ('resume', run_dir, 'best' | 'last'), ('polish-only', run_dir, None), or None for a fresh run.
        evaluate_run: evaluate the final checkpoint after training.
    Returns:
        the best validation score.
    """
    # 1. Run folder.
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
        cj, run_id, src = _rerun_checkpoint(kind, sdir, source)
        if kind == 'polish-only':
            cfg = replace(cfg, start_phase='lbfgs')
    print('[thesis run] arm=%s noise=%s start=%s n_a=%d seed=%d nf=%.3f s detune=%s'
          % (cfg.arm, cfg.noise, cfg.start, cfg.n_a, cfg.seed, cfg.nf_seconds, cfg.combo_init_detune)
          + ('  PS2' if cfg.ps2 else '  plain')
          + ('  %s from %s' % (rerun[0], rerun[1]) if rerun else ''), flush=True)

    # 2. Data and normalisation.
    data = load_datasets(cfg)
    norm = compute_normalization(cfg, data)

    if rerun is None:
        os.makedirs(sdir)
        with open(os.path.join(sdir, 'config.json'), 'w') as fh:
            json.dump({**config_json_dict(cfg, git=git_provenance()),
                       'run_id': run_id, 'data_dir': traj_dir(cfg)}, fh, indent=2)

    # 3. Model and closed loop.
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    fit_sys = build_model(cfg, data, norm)

    fit_sys.simulator = build_closed_loop(
        fit_sys, norm, cfg,
        train_files=data.train_files, val_files=data.val_files, e_train=data.e_train, e_val=data.e_val)
    print('\nTraining loss: each window simulated in closed loop with the known controller')

    # 4. Encoder start, or the restored checkpoint.
    # after build_closed_loop: the encoder start simulates the start model in closed loop through its simulator
    if src is None:
        prefit_physical_encoder(fit_sys, cfg, data, norm)
        resume = None               # fresh run: train starts from the encoder start above
    else:
        # load first: if the checkpoint does not fit, load_checkpoint raises an error 
        # and the outputs are never moved to previous/
        meta = load_checkpoint(fit_sys, src, start_phase=cfg.start_phase)
        resume = (_archive_previous_outcome(rerun[0], sdir, cj, run_id, src), meta)

    # 5. Training.
    bestfit = train(fit_sys, cfg, data, resume=resume, checkpoint_dir=sdir, run_id=run_id)
    print(f"\nTraining complete. Best validation {SCORE_NAME}: {bestfit:.6g}")

    # 6. Evaluation of the final checkpoint: the accepted polish, else the Adam-selected model.
    if evaluate_run:
        evaluate(sdir, data_root=cfg.data_root)
    return bestfit


def cli():
    """Read MODE, RUN_DIR and SETTINGS with the command-line overrides into a RunConfig and a rerun, then start main."""
    ap = argparse.ArgumentParser(description='without flags: MODE, RUN_DIR and SETTINGS at the top of this file (the '
                                             'thesis model M_OBC on CUDA); without CUDA add --cpu')
    # model choices of a fresh run; a rerun reads them from the run folder's config.json
    ap.add_argument('--arm', choices=('obc', 'u'))
    ap.add_argument('--start', choices=('detuned', 'correct'))
    ap.add_argument('--n-a', type=int)
    ap.add_argument('--ps2', action=argparse.BooleanOptionalAction,
                    help='PS opening of the added states during the first Adam updates; --no-ps2 runs without')
    ap.add_argument('--noise', choices=('lowpass', 'noisefree'))
    ap.add_argument('--closed-form', action='store_true',
                    help='baseline derivative in closed form: same up to rounding, 3 to 8 %% faster')
    ap.add_argument('--cpu', action='store_true', help='run on the CPU, without torch.compile')
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument('--resume', metavar='RUN_DIR', help='MODE resume: continue the Adam training of this run folder')
    grp.add_argument('--polish-only', metavar='RUN_DIR',
                     help='MODE polish-only: rerun the L-BFGS polish of this run folder from its saved '
                          'pre-L-BFGS weights')
    grp.add_argument('--evaluate', metavar='RUN_DIR', help='MODE evaluate: evaluate the final checkpoint only')
    ap.add_argument('--from', dest='source', choices=('best', 'last'),
                    help='deepSI checkpoint a resume starts from (default RESUME_FROM)')
    ap.add_argument('--no-evaluate', action='store_true', help='skip the evaluation after training')
    ap.add_argument('--data-root', default=SETTINGS['data_root'], help='data set folder')
    ap.add_argument('--out-root', default=SETTINGS['out_root'], help='root of the run folders')
    a = ap.parse_args()

    # 1. Mode and run folder: a mode flag overrides MODE and RUN_DIR.
    mode, run_dir = MODE, RUN_DIR
    for flag in ('resume', 'polish_only', 'evaluate'):
        if getattr(a, flag) is not None:
            mode, run_dir = flag.replace('_', '-'), getattr(a, flag)
    if mode not in ('fresh', 'resume', 'polish-only', 'evaluate'):
        ap.error("MODE must be 'fresh', 'resume', 'polish-only' or 'evaluate', got %r" % mode)
    if (run_dir is None) != (mode == 'fresh'):
        ap.error('MODE %r %s RUN_DIR' % (mode, 'takes no' if mode == 'fresh' else 'needs a'))
    if a.source is not None and mode != 'resume':
        ap.error('--from applies to MODE resume only')
    if RESUME_FROM not in ('best', 'last'):
        ap.error("RESUME_FROM must be 'best' or 'last', got %r" % RESUME_FROM)
    axes = {k: v for k, v in dict(arm=a.arm, start=a.start, n_a=a.n_a, ps2=a.ps2, noise=a.noise).items()
            if v is not None}
    if a.closed_form:
        axes['baseline_form'] = 'closed'
    if axes and mode != 'fresh':
        ap.error("a rerun reads its settings from the run folder's config.json; drop %s" % ', '.join(sorted(axes)))

    # 2. Configuration: SETTINGS for a fresh run, the folder's config.json otherwise.
    if mode == 'evaluate':
        evaluate(run_dir, data_root=a.data_root)
        return
    if mode == 'fresh':
        cfg = RunConfig(**dict(SETTINGS, **axes, data_root=a.data_root, out_root=a.out_root))
        rerun = None
    else:
        with open(os.path.join(run_dir, 'config.json')) as fh:
            cfg = RunConfig.from_config_json(json.load(fh), data_root=a.data_root, out_root=a.out_root)
        rerun = ('resume', run_dir, a.source or RESUME_FROM) if mode == 'resume' else ('polish-only', run_dir, None)
    if a.cpu:
        cfg = replace(cfg, device='cpu', compile_mode=None)
    main(cfg, rerun=rerun, evaluate_run=EVALUATE and not a.no_evaluate)


if __name__ == '__main__':
    cli()
