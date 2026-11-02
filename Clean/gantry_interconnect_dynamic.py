"""Train the augmented gantry model with the settings below, or continue, polish or evaluate a run folder; the default
is the thesis model M_OBC (OBC arm, detuned start, two added states, PS opening, lowpass data) on CUDA."""
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
MODE = 'fresh'         # 'fresh': train a new run with SETTINGS; 'resume': continue the Adam phase of RUN_DIR;
                       # 'polish-only': polish the pre-polish checkpoint of RUN_DIR again; 'evaluate': evaluate RUN_DIR
RUN_DIR = None         # run folder of resume, polish-only and evaluate; its config.json fixes the settings
RESUME_FROM = 'best'   # resume: the deepSI dump at the lowest validation score ('best') or the latest ('last')
EVALUATE = True        # evaluate the final checkpoint after training

# Run settings of a fresh run, recorded in its config.json; the values are those of the thesis runs
SETTINGS = dict(
    # model: the five axes; the command-line flags of the same name override them
    arm='obc',          # 'obc': the network cannot mimic a change of the baseline parameters; 'u': unconstrained
    start='detuned',    # baseline parameters at the start: 'correct' (nominal) or 'detuned' (each of the ten +- detune)
    n_a=2,              # number of added states, 0, 2 or 8; 0: the network only corrects the physical states
    ps2=True,           # True: the PS opening (ps2_opening.py) first fits the added states; meant for n_a > 0
    noise='lowpass',    # data set: 'lowpass' (with measurement noise) or 'noisefree'
    # sampling and windows
    fs_new=4000,        # [Hz] training rate; must divide the 20 kHz of the data sets (HEURISTIC: thesis value)
    aa_order=8,         # order of the zero-phase anti-aliasing Butterworth on y - r of the lowpass data, before decimation
    # [Hz] its cutoff; change it with fs_new (HEURISTIC: 0.8 x the 2 kHz Nyquist frequency of fs_new = 4 kHz)
    # THEORY: forward-backward pass squares the order-8 magnitude, |H| = 1 / (1 + x^16), x = tan(pi f/fs) / tan(pi fc/fs)
    aa_fc_hz=1600.0,
    # [s] cut from both ends of every record, on both data sets, to drop the filter's edge transients
    # HEURISTIC: the filter's edge transient exceeds 0.01 of the in-band noise for at most 4.6 ms on the 36 records
    aa_trim_s=0.005,
    nf_seconds=0.100,   # [s] simulated length of each training window (HEURISTIC: thesis value)
    stride=10,          # samples between window starts; 1 gives 10x more windows per epoch (HEURISTIC: thesis value)
    na_nb=30,           # past samples of y and u the encoder sees (HEURISTIC: thesis value)
    seed=1,             # initial weights, detune signs, polish batch, PS opening (Adam batch order: seed 0)
    detune=0.10,        # relative detuning of each baseline parameter on the detuned start (HEURISTIC: 10 %)
    # networks: augmentation phi_aug and encoder psi_tilde (HEURISTIC: thesis values)
    n_nodes_per_layer=16,  # width of each hidden tanh layer
    n_hidden_layers=2,     # number of hidden layers
    up_sample=1,           # RK4 steps per sample; more is more accurate and slower
    # Adam (HEURISTIC: thesis values)
    batch_size=512,     # windows per update
    lr=1e-5,            # learning rate, all parameters
    adam_eps=1e-16,     # below the 1e-8 default, for float64
    epochs=200,         # budget in epochs (passes over the training windows)
    n_its=None,         # budget in updates instead; validates at start and end only
    its_per_val=1300,   # updates between validations (checkpoint selection)
    # L-BFGS polish (HEURISTIC: thesis values)
    lbfgs_max_iter=100,       # iterations in total
    lbfgs_inner_iter=20,      # iterations per step() call; a call that stops early ends the polish
    lbfgs_windows=16384,      # windows in its fixed batch; lower if out of memory
    lbfgs_history=50,         # curvature pairs kept
    lbfgs_tol_grad=2.6e-5,    # stop when the largest loss gradient is below this
    lbfgs_tol_change=2.6e-7,  # stop when the loss or the step changes less than this
    # PS opening; counts in Adam updates (eval_every, s1_*, screen_at), S2 steps (s2_*) or windows and pairs (n_*)
    ps2_opening=dict(
        aux_batch=64,       # windows per S1 step, separate from the training batch (HEURISTIC)
        lr=1e-3,            # learning rate of the opening's own Adam (THEORY: the Adam default, Kingma and Ba)
        # S1: decoder fit (HEURISTIC: thesis values)
        eval_every=25,      # updates between calibration checks
        s1_min=100,         # no S1 end before this many updates
        s1_cap=350,         # S1 ends here at the latest
        s1_tol=0.02,        # plateau: the best of the last s1_window checks improved by less than this, relative
        s1_window=4,        # checks in the plateau window
        screen_at=150,      # update of the early screen; a multiple of eval_every
        screen_max=0.90,    # S2 runs only if the decoder explains at least 10 % of the residual
        n_cal=128,          # left-out windows of the S1 check
        # S2: added-state transition fit (HEURISTIC: thesis values)
        n_snap=8192,        # fit pairs
        n_snap_cal=2048,    # left-out pairs
        s2_batch=512,       # pairs per step
        s2_check=100,       # steps between checks
        s2_window=5,        # checks in the plateau window
        s2_min=500,         # no S2 end before this many steps
        s2_cap=8000,        # S2 ends here at the latest
        s2_tol=0.01,        # plateau tolerance, as s1_tol
    ),
    # environment
    device='cuda',                   # 'cuda' or 'cpu' (the --cpu flag also sets compile_mode None)
    compile_mode='reduce-overhead',  # CUDA graphs for the training rollouts, faster; None on the CPU
    truncate_s=None,                 # [s] use only the start of each record, for trial runs; None: whole records
    baseline_form='lfr',             # baseline derivative through the LFR output equation, or 'closed' (--closed-form):
                                     # (qdot, a) directly, which differs from the thesis runs by rounding
    obc_ref_chunk=16384,             # OBC basis build: points per batch; memory only, same result
    data_root=DATA_ROOT,             # folder of the data sets (--data-root)
    out_root=OUT_ROOT,               # folder of the run folders <out_root>/<noise>/<run_id> (--out-root)
)


def _dumps(run_id):
    """Paths of the deepSI dumps _best.pth and _last.pth of a run, in deepSI's checkpoints folder."""
    return [os.path.join(get_work_dirs()['checkpoints'], 'AugmentedModel_%s%s.pth' % (run_id, s))
            for s in ('_best', '_last')]


def _rerun_source(kind, run_dir, source):
    """(config.json, run id, source checkpoint) of a rerun in an existing run folder."""
    with open(os.path.join(run_dir, 'config.json')) as fh:
        cj = json.load(fh)
    run_id = str(cj['run_id'])
    if kind == 'polish-only':
        # the pre-polish pair: the Adam-selected weights and their Adam moments
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
    """Move the previous outcome to previous/<rerun_id>/, log the rerun in config.json, return the source's new path."""
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
    print('\nLoss rollout: closed loop (the known controller around the model)')

    # 4. Encoder start, or the restored checkpoint.
    # the encoder prefit simulates the baseline in closed loop, so it needs the controllers built above
    resume = None
    if src is None:
        prefit_physical_encoder(fit_sys, cfg, data, norm)
    else:
        # archive the run folder only after the checkpoint has loaded
        meta = load_checkpoint(fit_sys, src, start_phase=cfg.start_phase)
        resume = (_archive_rerun(rerun[0], sdir, cj, run_id, src), meta)

    # 5. Training.
    bestfit = train(fit_sys, cfg, data, resume=resume, checkpoint_dir=sdir, run_id=run_id)
    print(f"\nTraining complete. Best validation {SCORE_NAME}: {bestfit:.6g}")

    # 6. Evaluation of the final checkpoint: the accepted polish, else the Adam-selected model.
    if evaluate_run:
        evaluate(sdir, data_root=cfg.data_root)
    return bestfit


def cli():
    """Read MODE, RUN_DIR and SETTINGS with the command-line overrides into a RunConfig and a rerun, then start main."""
    ap = argparse.ArgumentParser(description='no arguments = MODE and SETTINGS at the top of this file (the thesis '
                                             'model M_OBC on CUDA); without CUDA use --cpu')
    # model axes of a fresh run; a run folder fixes them
    ap.add_argument('--arm', choices=('obc', 'u'))
    ap.add_argument('--start', choices=('detuned', 'correct'))
    ap.add_argument('--n-a', type=int, choices=(0, 2, 8))
    ap.add_argument('--ps2', action=argparse.BooleanOptionalAction, help='PS opening of the added states')
    ap.add_argument('--noise', choices=('lowpass', 'noisefree'))
    ap.add_argument('--closed-form', action='store_true', help="baseline_form 'closed'")
    ap.add_argument('--cpu', action='store_true', help='device cpu, eager')
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument('--resume', metavar='RUN_DIR', help='MODE resume: continue the Adam phase of this run folder')
    grp.add_argument('--polish-only', metavar='RUN_DIR', help='MODE polish-only: polish the pre_lbfgs pair again')
    grp.add_argument('--evaluate', metavar='RUN_DIR', help='MODE evaluate: evaluate the final checkpoint only')
    ap.add_argument('--from', dest='source', choices=('best', 'last'),
                    help='deepSI dump a resume starts from (default RESUME_FROM)')
    ap.add_argument('--no-evaluate', action='store_true', help='no evaluation after training')
    ap.add_argument('--data-root', default=SETTINGS['data_root'])
    ap.add_argument('--out-root', default=SETTINGS['out_root'])
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
        ap.error('a run folder fixes the configuration; give no %s with it' % ', '.join(sorted(axes)))

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
