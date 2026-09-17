"""Compile probe for the one-step OBC seam: walk the fallback ladder (D-192 sect. 9.7).

ONE MODE PER PROCESS, selected with `--mode`. That is not tidiness: job 83796 died with
`CUDA error: an illegal memory access was encountered` inside `cudagraph_trees.py`, and an illegal
access poisons the CUDA context, so everything after it in the same process is meaningless. The
runner invokes this script once per rung and lets each one fail independently.

The ladder, in the order the runner walks it:

    --mode reduce-overhead   inductor + kernel fusion + CUDA graphs. The production setting, and
                             the rung that failed in job 83796.
    --mode default           inductor + kernel fusion, NO CUDA graphs. Removes the code path the
                             crash lives in. Measured at ~2.6x against eager in config.py, versus
                             ~6.5x for reduce-overhead.
    --mode max-autotune      fusion plus autotuned kernels, still with CUDA graphs.
    --mode aot_eager         tracing contract only (Dynamo + AOTAutograd, no codegen). This is
                             what the development PC could run, since Triton refuses its Quadro
                             P2000; it proves the graph and the guards, not the kernels.
    --mode eager             no compilation, the reference numbers.

Each run: builds the production model with the FULL reference set, attaches OBC, compiles the way
`build_closed_loop` does in a real run, then drives `fs.loss` through the production simulator
seam for a few objectives with `error_on_recompile` armed, and finishes with one update at the
PRODUCTION shape reporting peak device memory. Results land in
`results/<date>/probe_compile_<mode>.json`; a failure is recorded there and the exit code is 1.
"""
__project_origin__ = "added"

import argparse
import json
import os
import sys
import time
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rig import (RESULTS_DIR, Tee, obc_config, build_system, perturb_output_layer,   # noqa: E402
                 make_batch, cuda_sync)

SHORT_NF = 20      # horizon for the recompile checks: the seam is PER STEP, so nf is irrelevant
BATCH = 512        # the production batch, so Dynamo's shape guards match a real run
STRIDE = 200       # window decimation; must leave at least BATCH windows at the production nf
INDUCTOR_MODES = ('default', 'reduce-overhead', 'max-autotune')
# Rung 3 of the ladder. `build_closed_loop` hardcodes fullgraph=True, so the functorch JVP cannot
# leave the graph; job 83913 then died with an illegal memory access in EVERY inductor mode,
# including `default`, which has no CUDA graphs. Relaxing fullgraph lets the transform break out
# and run eager while every other operation in the step still compiles. These modes therefore
# compile the pair here rather than through build_closed_loop.
NOFULLGRAPH_MODES = tuple('%s-nofullgraph' % m for m in INDUCTOR_MODES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='reduce-overhead',
                    choices=list(INDUCTOR_MODES) + list(NOFULLGRAPH_MODES)
                    + ['aot_eager', 'eager'])
    ap.add_argument('--objectives', type=int, default=6,
                    help='objective evaluations after the first; any recompile among them fails')
    ap.add_argument('--prod-updates', type=int, default=3,
                    help='TIMED updates at the production shape, after one discarded warm-up')
    ap.add_argument('--skip-memory', action='store_true')
    # THE DIAGNOSTIC. `reduce-overhead` costs 525 s then 1472 s per update at nf 400 and degrades,
    # which says cudagraph trees is re-recording rather than replaying. The hypothesis is that
    # threading freshly allocated per-pass tensors into the compiled step is what breaks replay:
    # `pass_mats` is ten tensors and `obc_pass` twenty more, rebuilt every objective, and CUDA
    # graphs want inputs at stable addresses. `--no-thread` reproduces the PRE-D-195 configuration
    # (no hoist, no correction), which job 83795 runs at 0.77 s under `reduce-overhead`. If this
    # is fast, threading is the cause and static buffers are the fix. If it is still slow, the
    # cause is elsewhere and the static-buffer work would be wasted.
    ap.add_argument('--no-thread', action='store_true',
                    help='disable the M(Y) hoist and the OBC correction: nothing is threaded into '
                         'the compiled step, reproducing the configuration of job 83795')
    # D-199. Route the threaded tensors through fixed-address buffers while testing cudagraph
    # replay. Verified to leave loss and both gradients bit-identical and measured neutral in the
    # paired throughput probe; no replay or speed-up claim is made.
    ap.add_argument('--static-pass', action='store_true',
                    help='fixed-address buffers for the per-pass tensors (D-199)')
    ap.add_argument('--space', choices=('tangent', 'affine'), default='tangent',
                    help='protected OBC space; affine exercises the D-200 three-item pass')
    args = ap.parse_args()
    mode = args.mode

    tag = mode if args.space == 'tangent' else '%s_affine' % mode
    sys.stdout = Tee(os.path.join(RESULTS_DIR, 'probe_compile_%s.log' % tag))
    out = {'mode': mode, 'space': args.space, 'torch': torch.__version__, 'ok': False}
    if not torch.cuda.is_available():
        print('no CUDA device: this probe needs one'); return 1
    dev = torch.device('cuda')
    props = torch.cuda.get_device_properties(0)
    out['device'] = '%s sm_%d%d %.1f GB' % (props.name, props.major, props.minor,
                                            props.total_memory / 1e9)
    print('=' * 78)
    print('MODE %s | device %s | torch %s' % (mode, out['device'], torch.__version__))
    print('=' * 78)

    # Compile exactly the way a real run does: `build_closed_loop` reads cfg.compile_mode and
    # compiles both `hfn` and `hfn.output_only`. Driving that path, rather than hand-rolling a
    # torch.compile here, is what makes this probe evidence about the production run.
    cfg_mode = mode if mode in INDUCTOR_MODES else None
    cfg = obc_config(device='cuda', compile_mode=cfg_mode, stride=STRIDE,
                     static_pass_buffers=args.static_pass, obc_space=args.space)
    out['static_pass'] = bool(args.static_pass)
    print('static-address buffers for the per-pass tensors: %s' % bool(args.static_pass))
    t0 = time.perf_counter()
    rig = build_system(cfg, ref_stride=1, verbose=True)
    fs, life, corr = rig.fs, rig.life, rig.corr
    fs.cuda()
    perturb_output_layer(rig.ann, 1e-3)
    batch, kw = make_batch(rig, B=BATCH, seed=0, device=dev)
    short = [batch[0], batch[1], batch[2][:, :SHORT_NF], batch[3][:, :SHORT_NF], batch[4]]

    # EAGER WARM-UP, and it is required rather than cosmetic. `Interconnect.init_forward` is LAZY:
    # the first `forward` or `output_only` call builds the connection matrices, walks the signal
    # graph in numpy and asserts there is no algebraic loop. Inside a compiled region Dynamo sees
    # that as data-dependent control flow and refuses under fullgraph=True:
    #     UserError: Dynamic control flow is not supported at the moment
    # A real run never hits this because deepSI's fit() performs an EAGER initial validation
    # before the first training update, which sets `initialized_forward_function`. Job 83907
    # failed here on every inductor rung purely because the probe compiled first and called
    # eagerly never. This reproduces fit()'s ordering.
    with torch.no_grad():
        _xw = torch.zeros(2, fs.hfn.nx, device=dev, dtype=cfg.dtype_pt)
        _uw = torch.zeros(2, fs.hfn.nu, device=dev, dtype=cfg.dtype_pt)
        fs.hfn.output_only(_xw, _uw)
        fs.hfn(_xw, _uw)
    print('eager warm-up done: init_forward complete (initialized_forward_function=%s)'
          % fs.hfn.initialized_forward_function)
    if args.no_thread:
        # Nothing reaches the compiled step but (x, u): no correction submodule, no hoisted
        # matrices. This is what job 83795 compiles, and it is the control for the hypothesis.
        fs.hfn.obc_correction = None
        fs.obc = None
        fs._mats_ix = None
        out['no_thread'] = True
        print('DIAGNOSTIC --no-thread: correction detached, M(Y) hoist disabled. The compiled '
              'step takes (x, u) only, as in job 83795.')
    print('build %.1f s | reference tuples %d | windows in batch %d | production nf %d'
          % (time.perf_counter() - t0, rig.ref.n, batch[0].shape[0], batch[2].shape[1]))
    out['reference_tuples'] = rig.ref.n
    out['batch'] = int(batch[0].shape[0])
    out['production_nf'] = int(batch[2].shape[1])

    if mode == 'aot_eager':
        # Tracing contract only: one graph, no graph breaks, and the guards hold. No codegen, so
        # this says nothing about kernels; it is the rung the development PC can reach.
        print('\n[trace] backend=aot_eager fullgraph=True')
        torch._dynamo.reset()
        try:
            x = batch[0].new_zeros(BATCH, fs.hfn.nx).normal_(0, 0.1).requires_grad_(True)
            u = batch[0].new_zeros(BATCH, 3).normal_(0, 0.1).requires_grad_(True)
            # The coefficient needs a basis; the lifecycle builds one at the epoch boundary, which
            # in a real run happens inside the first loss() call. Do it explicitly here.
            life.maybe_refresh(0, corr.expansion_point)
            th = life.coefficient(rig.ann).detach().clone().requires_grad_(True)
            exp = torch._dynamo.explain(fs.hfn)(x, u, th)
            out['graph_count'] = exp.graph_count
            out['graph_break_count'] = exp.graph_break_count
            out['op_count'] = exp.op_count
            print('  graphs %d | graph breaks %d | ops %d'
                  % (exp.graph_count, exp.graph_break_count, exp.op_count))
            for r in exp.break_reasons:
                print('    break:', str(r)[:200])
            fs.simulator._compiled = (
                torch.compile(fs.hfn, backend='aot_eager', fullgraph=True),
                torch.compile(fs.hfn.output_only, backend='aot_eager', fullgraph=True))
        except Exception as e:  # noqa: BLE001
            out['error'] = ''.join(traceback.format_exception_only(type(e), e))[:800]
            print('  FAILED: ' + out['error'])
            _write(out, mode); return 1
    elif mode in NOFULLGRAPH_MODES:
        base = mode[:-len('-nofullgraph')]
        print('\n[compile] backend=inductor mode=%s fullgraph=False (the JVP may graph-break out '
              'and run eager; everything else still compiles)' % base)
        torch._dynamo.reset()
        try:
            exp = torch._dynamo.explain(fs.hfn)(
                batch[0].new_zeros(BATCH, fs.hfn.nx).normal_(0, 0.1),
                batch[0].new_zeros(BATCH, fs.hfn.nu).normal_(0, 0.1),
                life_theta(life, corr, rig))
            out['graph_count'] = exp.graph_count
            out['graph_break_count'] = exp.graph_break_count
            out['break_reasons'] = [str(r)[:200] for r in exp.break_reasons]
            print('  graphs %d | graph breaks %d | ops %d'
                  % (exp.graph_count, exp.graph_break_count, exp.op_count))
            for r in exp.break_reasons[:6]:
                print('    break:', str(r)[:200])
        except Exception as e:  # noqa: BLE001
            print('  explain failed (not fatal): %s'
                  % ''.join(traceback.format_exception_only(type(e), e))[:200])
        torch._dynamo.reset()
        fs.simulator._compiled = (
            torch.compile(fs.hfn, backend='inductor', mode=base, fullgraph=False),
            torch.compile(fs.hfn.output_only, backend='inductor', mode=base, fullgraph=False))
    elif mode in INDUCTOR_MODES:
        comp = getattr(fs.simulator, '_compiled', None)
        print('\n[compile] build_closed_loop compiled the pair: %s (backend inductor, mode %s, '
              'fullgraph=True)' % (comp is not None, mode))
        if comp is None:
            out['error'] = 'build_closed_loop did not compile; cfg.compile_mode was %r' % cfg_mode
            print('  FAILED: ' + out['error']); _write(out, mode); return 1
    else:
        fs.simulator._compiled = None
        print('\n[eager] no compilation')

    # ---- objectives through the PRODUCTION seam --------------------------------------------
    # The first one pays compilation and builds the basis. From the third onward any recompile is
    # an error: a per-objective recompile would cost more than the projection saves.
    print('\n[objectives] %d evaluations at nf %d, batch %d, through ClosedLoopSimulator'
          % (args.objectives + 1, SHORT_NF, BATCH))
    times, armed = [], False
    try:
        for i in range(args.objectives + 1):
            # Graph BREAKS are expected and legal on the nofullgraph rungs; a RECOMPILE is not,
            # on any rung, so the guard is armed either way once the shapes have stabilised.
            if i == 2:
                torch._dynamo.config.error_on_recompile = True
                armed = True
            if i == args.objectives:
                fs.epoch_counter = 1.0        # force a basis refresh mid-sequence
            t0 = time.perf_counter()
            fs.optimizer.zero_grad()
            L = fs.loss(*short[:4], **kw)
            L.backward()
            fs.optimizer.step()
            cuda_sync()
            dt = time.perf_counter() - t0
            times.append(dt)
            print('  %2d  %8.2f s  loss %.6e%s' % (i, dt, float(L),
                                                   '  (basis refresh)' if i == args.objectives else ''))
        out['objective_times_s'] = times
        out['refreshes'] = life.n_refresh
        out['objectives_no_recompile'] = args.objectives - 1
        out['steady_state_s'] = sorted(times[2:])[len(times[2:]) // 2] if len(times) > 3 else None
        print('  first call %.1f s (compile + basis), steady state %.2f s, refreshes %d'
              % (times[0], out['steady_state_s'] or float('nan'), life.n_refresh))
    except Exception as e:  # noqa: BLE001
        out['error'] = ''.join(traceback.format_exception_only(type(e), e))[:800]
        out['failed_at_objective'] = len(times)
        out['objective_times_s'] = times
        print('  FAILED at objective %d: %s' % (len(times), out['error']))
        if 'illegal memory access' in out['error']:
            print('  (an illegal memory access poisons the CUDA context; this process cannot '
                  'continue, which is why each rung runs in its own process)')
        _write(out, mode); return 1
    finally:
        if armed:
            torch._dynamo.config.error_on_recompile = False

    # ---- production-shape update and peak memory --------------------------------------------
    # ABORT THE SECTION IF THE SHORT OBJECTIVES WERE ALREADY SLOW. The production shape is 20x
    # the horizon and runs four updates in each of two states, so a rung that costs seconds at
    # nf 20 costs hours here: job 83947 spent 937 s on a SINGLE update and hit the wall limit,
    # having learned nothing the nf-20 times had not already told us. A healthy rung should be
    # well under a second per objective at nf 20.
    SHORT_OBJ_LIMIT_S = 3.0
    PROD_UPDATE_LIMIT_S = 60.0
    steady = out.get('steady_state_s') or 0.0
    if not args.skip_memory and steady > SHORT_OBJ_LIMIT_S:
        print('\n[memory] SKIPPED. Steady state at nf %d is %.2f s, above the %.1f s guard, so a '
              'production-shape update would take roughly %.0f s and four of them in each state '
              'would exceed the wall limit. Fix the per-step cost first; the nf %d timings above '
              'are the diagnosis.'
              % (SHORT_NF, steady, SHORT_OBJ_LIMIT_S, steady * out['production_nf'] / SHORT_NF,
                 SHORT_NF))
        out['production_memory'] = {'skipped': True, 'steady_state_s': steady,
                                    'guard_s': SHORT_OBJ_LIMIT_S}
    elif not args.skip_memory:
        print('\n[memory] one update at the PRODUCTION shape: nf %d, batch %d, checkpoint_chunk %d'
              % (batch[2].shape[1], batch[0].shape[0], cfg.checkpoint_chunk))
        mem = {'device_total_bytes': int(props.total_memory)}
        saved = fs.hfn.obc_correction
        states = (('off', False),) if args.no_thread else (('off', False), ('on', True))
        for label, on in states:
            fs.hfn.obc_correction = saved if on else None
            fs.obc = life if on else None
            try:
                # ONE DISCARDED WARM-UP, then timed repeats. The first update at this shape pays
                # shape-specific codegen on a compiled rung, and the objectives above ran at
                # nf 20, so the production shape is new to the guards. Job 83913 reported 7.22 s
                # and 30.47 s from single samples that included exactly that cost, which is fine
                # for eager and would overstate every compiled rung.
                times = []
                for j in range(args.prod_updates + 1):
                    fs.optimizer.zero_grad(set_to_none=True)
                    if j == 1:      # measure memory over the TIMED updates, not the warm-up
                        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                    t0 = time.perf_counter()
                    L = fs.loss(*batch[:4], **kw); L.backward(); cuda_sync()
                    times.append(time.perf_counter() - t0)
                    # JUDGE THE SECOND UPDATE, NOT THE FIRST. The first at a new shape pays
                    # codegen, and under `reduce-overhead` it also pays the CUDA-graph recording
                    # over 400 sequential invocations; `gantry_interconnect_dynamic_gpu.sh`
                    # records that as "~500 s of ONE-OFF compile on the first two updates", and
                    # job 83795 then runs at 0.77 s. Job 83948's 623.5 s was that one-off, and an
                    # earlier version of this guard would have condemned the rung for it. The
                    # `default` rung shows the same shape harmlessly: 14.82 s first, 0.97 s steady.
                    if len(times) > 1 and times[1] > PROD_UPDATE_LIMIT_S:
                        print('  %-3s SECOND update took %.1f s, above the %.0f s guard, so this '
                              'is a per-update cost rather than a one-off. Recording and skipping '
                              'the rest.' % (label, times[1], PROD_UPDATE_LIMIT_S))
                        break
                warm = sorted(times[1:]) or [times[0]]      # guard may have stopped after one
                mem[label] = {'ok': True, 'loss': float(L), 'first_s': times[0],
                              'seconds': warm[len(warm) // 2], 'all_s': times,
                              'timed_repeats': len(times) - 1,
                              'peak_bytes': int(torch.cuda.max_memory_allocated())}
                print('  %-3s loss %.4e   first %.2f s   steady %.2f s (median of %d)   '
                      'peak %.2f GB of %.2f GB'
                      % (label, float(L), times[0], mem[label]['seconds'], len(warm),
                         mem[label]['peak_bytes'] / 1e9, props.total_memory / 1e9))
            except Exception as e:  # noqa: BLE001
                mem[label] = {'ok': False, 'error': ''.join(
                    traceback.format_exception_only(type(e), e))[:300]}
                print('  %-3s FAILED: %s' % (label, mem[label]['error']))
                torch.cuda.empty_cache()
            fs.optimizer.zero_grad(set_to_none=True)
        fs.hfn.obc_correction = saved; fs.obc = life
        if mem.get('on', {}).get('ok') and mem.get('off', {}).get('ok'):
            mem['peak_ratio'] = mem['on']['peak_bytes'] / mem['off']['peak_bytes']
            mem['update_ratio'] = mem['on']['seconds'] / mem['off']['seconds']
            mem['headroom_on'] = 1.0 - mem['on']['peak_bytes'] / props.total_memory
            # THE NUMBER THE EXPERIMENT IS BUDGETED ON. 130 updates per epoch at nf 400.
            mem['hours_per_100_epochs_on'] = mem['on']['seconds'] * 130 * 100 / 3600.0
            print('  peak on/off %.2fx | update on/off %.2fx | headroom with the correction on '
                  '%.0f%%' % (mem['peak_ratio'], mem['update_ratio'], 100 * mem['headroom_on']))
            print('  BUDGET: %.2f s/update with the correction on is %.1f h per 100 epochs '
                  '(130 updates/epoch); the compiled control arm runs at 0.77 s/update'
                  % (mem['on']['seconds'], mem['hours_per_100_epochs_on']))
            if mem['headroom_on'] < 0.15:
                print('  WARNING: under 15 percent headroom. The validation free run and the '
                      'L-BFGS phase allocate on top of this; raise checkpoint_chunk.')
        out['production_memory'] = mem

    out['ok'] = True
    print('\nMODE %s: OK' % mode)
    _write(out, mode)
    return 0


def life_theta(life, corr, rig):
    """A representative coefficient for tracing: the basis is built if it does not exist yet."""
    life.maybe_refresh(0, corr.expansion_point)
    return life.coefficient(rig.ann).detach().clone()


def _write(out, mode):
    tag = mode if out.get('space', 'tangent') == 'tangent' else '%s_affine' % mode
    p = os.path.join(RESULTS_DIR, 'probe_compile_%s.json' % tag)
    with open(p, 'w') as f:
        json.dump(out, f, indent=2)
    print('written %s' % p)


if __name__ == '__main__':
    sys.exit(main())
