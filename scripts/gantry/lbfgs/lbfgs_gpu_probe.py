"""GPU probe for the L-BFGS polish phase (D-171). Measures, decides nothing.

Answers the three questions `tasks/lbfgs-known-issues.md` leaves open, none of which can be
answered on the CPU:

  Q1  Is the closure bit-deterministic on CUDA, eager and compiled?
      The strong-Wolfe line search compares f at several points, so this is the precondition for
      the whole phase. It has only ever been checked on the CPU. If the COMPILED rollout is
      self-consistent, the phase keeps the ~6.5x it is otherwise giving up.

  Q2  What does one closure cost, and how much memory does it hold, over
      (n_windows) x (chunk) x (float32, float64)?
      The closure holds the whole batch's activation graph, and
      docs/gpu-nf12000-feasibility-2026-08-31.md:155 puts batch 4096 at the float32 ceiling of an
      8 GB card. float64 doubles it. OOM is a RESULT here, not a failure: the table is the point.

  Q3  Does the batch build now scale with the request rather than the dataset?
      Reports windows built vs windows used vs the full set, per point.

Runs no optimisation and changes no weights. Output is one table per dtype plus a verdict block.

Usage (see runners/lbfgs_gpu_probe.sh):
    python -u scripts/gantry/lbfgs/lbfgs_gpu_probe.py [--ckpt PATH] [--nf 400]
"""
__project_origin__ = "added"

import argparse
import os
import sys
import time
from dataclasses import replace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'gantry'))

from model_augmentation.fit_systems.lbfgs_polish import (  # noqa: E402
    FixedBatch, _named_params, _quiet_probes, warm_up_lazy_init, window_count,
)

DEFAULT_CKPT = os.path.join(
    ROOT, 'scripts', 'gantry', 'meeting', 'meeting-07-09-2026', 'server', 'checkpoints',
    'SSE_Interconnect_Composed_p2LPDA_best.pth')


def build(use_f64, compile_mode, ckpt, nf_override):
    """The checkpoint's system on the GPU, at the requested dtype and compile mode.

    The ARCHITECTURE comes from the checkpoint, not from `CFG`. Job 81461 failed exactly there:
    the cluster's entry file carried a 16x2 `nx_ann=2` configuration while the checkpoint is
    24x3 `nx_ann=8`, so the build and the weights disagreed. A probe whose model depends on which
    revision of the entry file happens to be checked out is measuring the wrong thing anyway.

    Everything else still comes from `CFG`: dataset, nf, stride, batch, closed loop. Those are
    properties of the experiment rather than of the weights, and they are what the cost table is
    supposed to be about.
    """
    import gantry_interconnect_dynamic as G
    from gantry_dynamic.data import load_datasets, compute_normalization, VAL_FILES, TRAIN_FILES
    from gantry_dynamic.model import build_model
    from gantry_dynamic.controller import build_closed_loop
    from gantry_dynamic.training import config_for_checkpoint, resolve_checkpoint

    hfn_sd, enc_sd, _opt, _meta, _fmt = resolve_checkpoint(ckpt)
    # `start_phase` pinned: the probe never resumes, and CFG may carry 'lbfgs' from a
    # smoke-test configuration, which would make this build depend on the entry file's mood.
    cfg = replace(G.CFG, device='cuda', compile_mode=compile_mode, save_flag=False,
                  use_f64=use_f64, lbfgs=True, start_phase='adam',
                  **({'nf_override': nf_override} if nf_override else {}))
    cfg = config_for_checkpoint(cfg, hfn_sd)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    data = load_datasets(cfg)
    norm = compute_normalization(cfg, data)
    hp = cfg.hp
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    fit_sys = build_model(hp, cfg, data, norm)
    fit_sys.simulator = build_closed_loop(fit_sys, norm, cfg, train_files=TRAIN_FILES,
                                          val_files=VAL_FILES, val_data=data.val_ckpt_data)
    # The probe DELIBERATELY casts, and it is the one place that may. It measures COST and
    # DETERMINISM, for which the exact weights are irrelevant as long as they are the right
    # scale, and it has to report both dtypes from one checkpoint. Everything on the training
    # path refuses the cast instead (training.py::_assert_same_dtype): precision must stay
    # coherent across a resume, or the phase's before/after comparison spans a dtype change.
    fit_sys.hfn.load_state_dict({k: v.to(cfg.dtype_pt) for k, v in hfn_sd.items()})
    fit_sys.encoder.load_state_dict({k: v.to(cfg.dtype_pt) for k, v in enc_sd.items()})
    fit_sys.cuda()
    return fit_sys, data, cfg, {'nf': hp['nf'], 'stride': cfg.stride}


def one_closure(fit_sys, batch, params):
    """Loss and gradient over the whole (possibly chunked) batch. Returns the loss value."""
    for p in params:
        p.grad = None
    total = 0.0
    for uh, yh, uf, yf, kw, w in batch:
        loss = fit_sys.loss(uh, yh, uf, yf, **kw) * w
        loss.backward()
        total += float(loss)
    return total


def probe_point(fit_sys, data, lk, n_windows, chunk, params):
    """(built, used, MB_batch, MB_peak, s_build, s_closure, loss) or an 'OOM' row."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        with _quiet_probes(fit_sys):
            batch = FixedBatch(fit_sys, data.train_data, lk, n_windows, 42,
                               chunk=chunk, verbose=False)
    except torch.cuda.OutOfMemoryError:
        return None, 'OOM building the batch'
    t_build = time.time() - t0
    built = window_count(fit_sys, data.train_data, lk['nf'], batch.effective_stride)
    warm_up_lazy_init(fit_sys, batch)      # lazy init outside the compiled region (job 81462)

    try:
        one_closure(fit_sys, batch, params)                      # warm-up / compile
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        loss = one_closure(fit_sys, batch, params)
        torch.cuda.synchronize()
        t_closure = time.time() - t1
    except torch.cuda.OutOfMemoryError:
        return None, 'OOM in the closure (batch fits, the graph does not)'

    return dict(built=built, used=batch.n_used, total=batch.n_total,
                stride=batch.effective_stride,
                mb_batch=batch.bytes_on_device / 1024 ** 2,
                mb_peak=torch.cuda.max_memory_allocated() / 1024 ** 2,
                s_build=t_build, s_closure=t_closure, loss=loss), None


def determinism(fit_sys, data, lk, params, label):
    with _quiet_probes(fit_sys):
        batch = FixedBatch(fit_sys, data.train_data, lk, 256, 42, verbose=False)
    # `Interconnect.output_only` initialises itself lazily on the first call, and that build is
    # untraceable by torch.compile ("Dynamic control flow is not supported", job 81462). A
    # training run never sees it because fit() validates eagerly first; a probe that goes
    # straight to the compiled rollout does. One eager forward first.
    warm_up_lazy_init(fit_sys, batch)
    a = one_closure(fit_sys, batch, params)
    b = one_closure(fit_sys, batch, params)
    ga = torch.cat([p.grad.reshape(-1) for p in params if p.grad is not None]).clone()
    c = one_closure(fit_sys, batch, params)
    gb = torch.cat([p.grad.reshape(-1) for p in params if p.grad is not None])
    loss_ok, grad_ok = (a == b == c), bool(torch.equal(ga, gb))
    print('  %-9s loss %.17e / %.17e / %.17e -> %s | gradient bit-identical -> %s'
          % (label, a, b, c, 'DETERMINISTIC' if loss_ok else 'NOT DETERMINISTIC',
             'yes' if grad_ok else 'NO'))
    return loss_ok, grad_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=DEFAULT_CKPT)
    ap.add_argument('--nf', type=int, default=None, help='nf_override; default = the config nf')
    ap.add_argument('--windows', type=int, nargs='+', default=[512, 1024, 2048, 4096, 8192])
    ap.add_argument('--chunks', type=int, nargs='+', default=[0, 512],
                    help='0 = a single chunk')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('no CUDA device visible; this probe is GPU-only')
    print('device:', torch.cuda.get_device_name(0))
    print('checkpoint:', args.ckpt)

    # State the architecture the probe will build, and where it came from, before spending an
    # hour measuring it. A silent mismatch here is what killed job 81461.
    from gantry_dynamic.training import infer_arch_from_hfn, resolve_checkpoint
    _hfn_sd = resolve_checkpoint(args.ckpt)[0]
    _nx_ann, _nodes, _hidden = infer_arch_from_hfn(_hfn_sd)
    print('architecture inferred FROM THE CHECKPOINT: nx_ann=%d, %dx%d'
          % (_nx_ann, _nodes, _hidden))

    verdict = {}
    for use_f64 in (False, True):
        tag = 'float64' if use_f64 else 'float32'
        print('\n' + '=' * 100)
        print('%s' % tag)
        print('=' * 100)

        # ---- Q1, determinism, compiled first because that is the version we want to keep -----
        print('\n[Q1] determinism, 256 windows')
        for mode in ('reduce-overhead', None):
            label = 'compiled' if mode else 'eager'
            fit_sys, data, cfg, lk = build(use_f64, mode, args.ckpt, args.nf)
            params = [p for _, p in _named_params(fit_sys) if p.requires_grad]
            try:
                verdict[(tag, label)] = determinism(fit_sys, data, lk, params, label)
            except torch.cuda.OutOfMemoryError:
                print('  %-9s OOM' % label)
                verdict[(tag, label)] = (None, None)
            if mode is None:
                keep = (fit_sys, data, lk, params)               # reuse the eager build below
            del fit_sys
            torch.cuda.empty_cache()

        # ---- Q2 + Q3, cost and memory, on the EAGER build (a compiled sweep would recompile
        #      per shape and measure compilation, not the rollout) ------------------------------
        fit_sys, data, lk, params = keep
        print('\n[Q2/Q3] cost and memory per closure (eager; multiply by ~1/6.5 if compiled)')
        print('  %8s %7s %8s %8s %9s %9s %9s %9s' %
              ('windows', 'chunk', 'built', 'stride', 'MB batch', 'MB peak', 's build', 's clos'))
        for n_windows in args.windows:
            for chunk in args.chunks:
                row, err = probe_point(fit_sys, data, lk, n_windows, chunk or None, params)
                if row is None:
                    print('  %8d %7s   %s' % (n_windows, chunk or '-', err))
                    torch.cuda.empty_cache()
                    continue
                print('  %8d %7s %8d %8d %9.1f %9.1f %9.2f %9.2f'
                      % (n_windows, chunk or '-', row['built'], row['stride'],
                         row['mb_batch'], row['mb_peak'], row['s_build'], row['s_closure']))
        print('  full window set at the config stride: %d'
              % window_count(fit_sys, data.train_data, lk['nf'], lk['stride']))
        del fit_sys
        torch.cuda.empty_cache()

    print('\n' + '=' * 100)
    print('VERDICT')
    print('=' * 100)
    for (tag, label), (loss_ok, grad_ok) in sorted(verdict.items()):
        print('  %-8s %-9s loss deterministic=%s  gradient bit-identical=%s'
              % (tag, label, loss_ok, grad_ok))
    print("""
How to read it:
  * If 'compiled' is loss-deterministic, leave force_eager=False: the phase keeps the ~6.5x.
    The GRADIENT column is informative but not required; L-BFGS's line search compares LOSSES,
    and a non-reproducible gradient only perturbs the curvature pair.
  * If 'compiled' is not deterministic, the phase already falls back to eager on its own and
    prints a line saying so. Nothing to change; expect eager cost.
  * Pick lbfgs_windows as the largest that fits with margin at the dtype you will run, and set
    lbfgs_chunk if the single-chunk row OOMs while the chunked one does not. Chunking is exact:
    it changes memory and not the result.""")


if __name__ == '__main__':
    main()
