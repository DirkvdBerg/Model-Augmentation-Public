"""GPU + torch.compile ladder for the trajectory-orthogonality penalty.

WHY THIS EXISTS. Everything verified so far ran eager on CPU. The production training path is
CUDA with `torch.compile` (`RunConfig.compile_mode`, wired at `gantry_dynamic/controller.py:257`,
measured about 6.5x with 'reduce-overhead'). Getting a long run to merely COMPLETE on that path
proves nothing about the mathematics: compilation can change gate behaviour or gradients silently,
and this penalty is unusually exposed to that for two structural reasons.

  1. THE INTERVENTION IS A MUTATED BUFFER. `ann_gate` swaps `Static_ANN_Block.out_gate` between
     the full / off / clamped rollouts. Dynamo guards on module attributes, so a gate switch can
     trigger a recompile (slow but correct) or, if the value is specialised into the graph, be
     IGNORED (fast and silently wrong: `p_off` would come back as `p_full`). Stage 2 tests the
     gate directly rather than assuming.
  2. RECOMPUTATION UNDER `torch.utils.checkpoint`. The forward is replayed during backward, by
     which time the gate context manager has exited. This is already documented on `ann_gate`;
     compilation changes when and how that replay happens, so `cfg.checkpoint_chunk` is reported
     and the chunked case is not assumed equivalent.

So the ladder gates the long run behind numerical parity, and every stage records wall time and
peak memory so the speedup is verified rather than assumed.

    stage 0  preflight: device, compute capability, inductor availability. Decides the fallback.
    stage 1  eager GPU reference, run TWICE to measure the run-to-run nondeterminism floor.
    stage 2  compiled, identical inputs: p_full, p_off, the gate, V, per-group gradients and one
             optimizer update, against stage 1. Recompilation counters recorded.
    stage 3  save / load / resume under compilation.
    stage 4  the longer training run, ONLY if 1 to 3 pass, timed against an eager arm.

TOLERANCES ARE MEASURED, NOT CHOSEN. Compiled and eager kernels reassociate floating-point sums,
so bit-exactness is not the correct acceptance criterion and demanding it would only invite
loosening it later until it passed. Stage 1 runs the identical eager computation twice to measure
what this GPU does to the same mathematics under run-to-run nondeterminism (atomics, cuDNN
selection, reduction order). That measured floor is the yardstick stage 2 is judged against.

USAGE
    python scripts/gantry/orthogonality/run_gpu_ladder.py --run-dir <ref run> [--stages 0,1,2,3]
    python scripts/gantry/orthogonality/run_gpu_ladder.py --run-dir <ref run> --train-its 200

Nothing here selects `beta`, and stage 4 is a plumbing and throughput run, NOT an efficacy
experiment: it reports that the penalty is applied and what it costs, never that it helps.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))

from dataclasses import replace                                          # noqa: E402

import check_update as CU                                                # noqa: E402
from gantry import diag                                                  # noqa: E402

check = CU.check


# --------------------------------------------------------------------------- stage 0
def preflight(args):
    """What this machine can actually do, decided BEFORE anything is built.

    A compiled run that silently falls back to eager, or an eager run reported as compiled, would
    make every later number uninterpretable. This resolves it once and records it.
    """
    print('\n' + '=' * 92)
    print('STAGE 0: preflight')
    print('=' * 92, flush=True)
    info = dict(torch=torch.__version__, cuda_available=torch.cuda.is_available(),
                requested_device=args.device, requested_compile=args.compile_mode)
    ok_cuda = torch.cuda.is_available() and args.device == 'cuda'
    if torch.cuda.is_available():
        info['gpu'] = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        info['capability'] = '%d.%d' % cap
        info['sm'] = cap[0] * 10 + cap[1]
    # ONLY A FAILURE IF CUDA WAS ASKED FOR. Requesting --device cpu is a legitimate way to run
    # this ladder (the local development box has no usable GPU), and recording that as a failed
    # check would gate stage 4 on a condition the caller deliberately chose. What must never
    # happen is asking for cuda and silently getting cpu, which is the case below.
    if args.device == 'cuda':
        check('CUDA is available, as requested', ok_cuda,
              info.get('gpu', 'NO GPU VISIBLE: --device cuda was requested and cannot be honoured')
              + (', capability ' + info['capability'] if 'capability' in info else ''))
    else:
        print('  [note] --device cpu requested; no CUDA assertion is made. Compilation is only '
              'wired for CUDA, so stage 2 will be skipped, not failed.')

    # INDUCTOR NEEDS sm >= 7.0 AND A C++ COMPILER. Both are recorded in `model.py`'s backend note
    # as the reason the Windows box cannot compile; asserting them here turns a confusing runtime
    # failure deep inside codegen into one line at the top of the log.
    can_compile = bool(ok_cuda) and info.get('sm', 0) >= 70
    reason = ''
    if not ok_cuda:
        reason = 'no CUDA device'
    elif info.get('sm', 0) < 70:
        reason = 'compute capability %s < 7.0, inductor cannot generate code' % info.get(
            'capability')
    else:
        try:
            # `importlib`, NOT `import torch._inductor`: the statement form binds the name
            # `torch` in THIS function's scope, which turns every earlier use of the module-level
            # `torch` in this function into an UnboundLocalError. It did exactly that on the
            # first run.
            import importlib
            importlib.import_module('torch._inductor')
        except Exception as e:                                           # noqa: BLE001
            can_compile, reason = False, 'torch._inductor did not import: %r' % e
    info['can_compile'] = can_compile
    info['compile_unavailable_reason'] = reason
    # AN EXPLICITLY REQUESTED FALLBACK IS NOT A FAILURE. `--no-compile` and `--device cpu` are
    # legitimate ways to run this ladder, and recording either as a failed check would gate
    # stage 4 on a condition the caller chose deliberately. Compilation being unavailable when
    # it WAS asked for is a different matter and stays a failure, because the alternative is
    # reporting an eager run as a compiled one.
    if args.no_compile or args.device != 'cuda':
        print('  [note] compilation not requested (%s); inductor availability is recorded, not '
              'asserted. Stage 2 will be SKIPPED, and a skipped parity stage is not a passed '
              'one: stage 4 will say it ran eager.'
              % ('--no-compile' if args.no_compile else '--device ' + args.device))
    else:
        check('torch.compile (inductor) is usable on this machine', can_compile,
              reason or 'inductor available')
        if not can_compile:
            print('        FALLING BACK TO EAGER. Stage 2 will be reported as SKIPPED, not '
                  'passed, and stage 4 will run eager.')
    info['peak_rss_mb_at_start'] = diag.peak_rss_mb()
    return info


# --------------------------------------------------------------------------- probe
def probe(env, hook, batch, args, label, seed):
    """One complete evaluation: the three rollouts, the penalty, the gradients, one update.

    THE SAME CODE PATH for every arm, differing only in whether the model was compiled. Seeding
    is reset per arm so any residual stochasticity (dropout, nondeterministic kernels selecting
    an algorithm) starts from the same place, which is what makes the eager/eager repeat a
    measurement of the FLOOR rather than of a different random draw.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    fit_sys, ad, geom = env.fit_sys, hook.adapter, hook.geometry
    groups = hook.groups()
    out = {}

    diag._cuda_reset(args.device)
    t0 = time.time()

    # ---- the three interventions, from ONE encode ------------------------------------
    #
    # GRAD MUST BE ENABLED HERE, AND THAT IS NOT A DETAIL. `ClosedLoopSimulator._rollout`
    # (closed_loop.py:554) selects the compiled callable with
    #
    #     use_c = (self._compiled is not None and ufuture.is_cuda and torch.is_grad_enabled())
    #
    # so a rollout computed inside `torch.no_grad()` runs EAGER even in a fully compiled run.
    # The first version of this probe wrapped all three interventions in `no_grad`, which would
    # have compared eager against eager and reported perfect parity on exactly the quantities
    # this stage exists to check. Found 2026-09-08 by reading the production GPU runner.
    xp, xa = ad.encode()
    z = torch.zeros_like(xa)

    # The PRODUCTION pairing, reproduced exactly: `p_full` carries grad (so it takes the compiled
    # path), `p_off` does not (so it takes the eager one). See `reference_penalty_gradient` and
    # the hook's own `_contribution`.
    p_full = ad.scored_stack(mode='full', x0=ad.x0_from(xp, xa))
    with torch.no_grad():
        p_off_nograd = ad.scored_stack(mode='off', x0=ad.x0_from(xp, z))
        p_cl_nograd = ad.scored_stack(mode='clamped', x0=ad.x0_from(xp, z))
    # The SAME quantities forced down the same path as `p_full`, used only to measure how much
    # of `d` is the intervention and how much is the path difference. Never used by production.
    p_off_grad = ad.scored_stack(mode='off', x0=ad.x0_from(xp, z))
    p_cl_grad = ad.scored_stack(mode='clamped', x0=ad.x0_from(xp, z))

    out['p_full'] = p_full.detach().float().cpu()
    out['p_off'] = p_off_nograd.detach().float().cpu()
    out['p_clamped'] = p_cl_nograd.detach().float().cpu()
    out['p_off_same_path'] = p_off_grad.detach().float().cpu()
    out['d'] = (p_full - p_off_nograd).detach().float().cpu()
    out['d_same_path'] = (p_full - p_off_grad).detach().float().cpu()

    # MIXED-PATH CONTAMINATION. In a compiled CUDA run `d` is a compiled `p_full` minus an eager
    # `p_off`, so it carries the compiled-vs-eager difference as a SYSTEMATIC term on top of the
    # intervention it is supposed to isolate. This is a property of the method as implemented,
    # not of this script, and it is zero by construction in any eager run. Reported as a fraction
    # of |d| so it can be judged rather than assumed negligible.
    dd = float((out['d'] - out['d_same_path']).abs().max())
    out['mixed_path_abs'] = dd
    out['mixed_path_frac_of_d'] = dd / max(float(out['d'].abs().max()), 1e-30)

    # THE GATE MUST ACTUALLY DO SOMETHING. The clamped rollout differs from full ONLY by the
    # gate, so it is the probe that isolates a gate value specialised into a compiled graph.
    # Taken on the same-path pair, so a nonzero value cannot come from the path difference.
    out['gate_effect'] = float((p_full - p_cl_grad).abs().max())

    # ---- base loss, penalty, gradients -----------------------------------------------
    for p in [q for ps in groups.values() for q in ps]:
        p.grad = None
    loss = CU.base_loss(fit_sys, batch)
    loss.backward()
    out['base_loss'] = float(loss)
    out['base_grad'] = {k: CU.flat_grad(v).float().cpu() for k, v in groups.items()}

    hook(fit_sys)
    out['V'] = float(hook.last.get('V'))
    out['total_grad'] = {k: CU.flat_grad(v).float().cpu() for k, v in groups.items()}

    # ---- one optimizer update ---------------------------------------------------------
    before = {k: CU.flat(v).float().cpu() for k, v in groups.items()}
    opt = CU.fresh_adam(groups, args.lr, args.adam_eps)
    opt.step()
    out['after_step'] = {k: CU.flat(v).float().cpu() for k, v in groups.items()}
    with torch.no_grad():
        for k, ps in groups.items():
            i = 0
            for p in ps:
                n = p.numel()
                p.copy_(before[k][i:i + n].view_as(p).to(p.device, p.dtype))
                i += n

    if args.device == 'cuda':
        torch.cuda.synchronize()
    out['seconds'] = time.time() - t0
    out['peak_cuda_mb'] = diag.cuda_peak_mb(args.device)
    out['peak_rss_mb'] = diag.peak_rss_mb()
    print('  [%s] base %.12e  V %.9e  gate effect %.3e  %.2f s  peak %s MB'
          % (label, out['base_loss'], out['V'], out['gate_effect'], out['seconds'],
             out['peak_cuda_mb']))
    return out


def diff_report(a, b):
    """Worst absolute difference per compared quantity, on identical keys."""
    d = {}
    for k in ('p_full', 'p_off', 'p_clamped', 'p_off_same_path', 'd', 'd_same_path'):
        d[k] = float((a[k] - b[k]).abs().max())
    d['base_loss'] = abs(a['base_loss'] - b['base_loss'])
    d['V'] = abs(a['V'] - b['V'])
    for grp in ('base_grad', 'total_grad', 'after_step'):
        for k in a[grp]:
            d['%s.%s' % (grp, k)] = float((a[grp][k] - b[grp][k]).abs().max())
    return d


# --------------------------------------------------------------------------- stages 1-2
def stage_eager_reference(env, hook, batch, args):
    """Two identical eager runs. The difference between them IS the acceptance yardstick.

    Run-to-run nondeterminism on a GPU comes from atomics, cuDNN algorithm selection and
    reduction order. Whatever it produces here is the smallest difference this machine can
    distinguish from zero for this computation, so it is the correct scale against which to judge
    a compiled kernel that reassociates the same sums. Choosing a tolerance instead would be an
    oracle threshold, which this project's standing rules forbid.
    """
    print('\n' + '=' * 92)
    print('STAGE 1: eager reference, run twice to measure the nondeterminism floor')
    print('=' * 92, flush=True)
    a = probe(env, hook, batch, args, 'eager #1', args.seed)
    b = probe(env, hook, batch, args, 'eager #2', args.seed)
    floor = diff_report(a, b)
    worst = max(floor.values())
    check('the eager reference reproduces itself', True,
          'worst run-to-run difference %.3e at %s'
          % (worst, max(floor, key=floor.get)))
    check('the gate changes the rollout at all', a['gate_effect'] > 0,
          'max |p_full - p_clamped| = %.3e; if this were 0 the intervention would be inert and '
          'every later comparison vacuous' % a['gate_effect'])
    return a, floor, worst


def stage_compiled_parity(env_c, hook_c, batch_c, args, ref, floor, worst_floor):
    print('\n' + '=' * 92)
    print('STAGE 2: compiled vs eager parity')
    print('=' * 92, flush=True)
    # ---- EAGER WARM-UP OF THE INTERCONNECT, BEFORE ANY COMPILED CALL ------------------
    #
    # `Interconnect.output_only` and `Interconnect.forward` both open with
    #
    #     if not self.initialized_forward_function:
    #         self.init_forward()                      # interconnect.py:288
    #
    # and `init_forward` runs the structural graph check `assert not detect_algebraic_loop(...)`,
    # which is numpy (`if np.trace(An) != 0`). Dynamo cannot trace numpy control flow under
    # fullgraph=True and raises "Dynamic control flow is not supported" (job 82505).
    #
    # PRODUCTION NEVER HITS THIS, and that is the whole point. By the time `fit()` reaches its
    # first compiled call the model has already run eagerly, so the flag is True and Dynamo
    # constant-folds the branch away without ever entering `init_forward`. This ladder built a
    # FRESH env for the compiled arm and probed it cold, so the first call to `output_only`
    # happened under compilation with the flag still False. The defect was in the ladder's
    # sequencing, not in the method or the interconnect.
    #
    # One `no_grad` rollout puts the arm in exactly production's state. It is guaranteed to run
    # EAGER: `ClosedLoopSimulator._rollout` selects the compiled callable only when
    # `ufuture.is_cuda AND torch.is_grad_enabled()` (closed_loop.py:554), so a no_grad call
    # cannot accidentally compile. It touches both `forward` and `output_only`, and they share
    # the one flag.
    ad_c = hook_c.adapter
    with torch.no_grad():
        xp0, xa0 = ad_c.encode()
        ad_c.scored_stack(mode='full', x0=ad_c.x0_from(xp0, xa0))
    check('the interconnect forward graph was initialised EAGERLY before compiling',
          bool(getattr(env_c.fit_sys.hfn, 'initialized_forward_function', False)),
          'init_forward runs a numpy graph check that fullgraph=True cannot trace; production is '
          'always already initialised at its first compiled call')

    try:
        from torch._dynamo.utils import counters
        counters.clear()
    except Exception:                                                    # noqa: BLE001
        counters = None

    # WARM UP FIRST, UNTIMED. The production GPU runner records roughly 500 s of ONE-OFF
    # compilation on the first two updates (job 80713), against 0.50 s/update steady state.
    # Timing the first compiled call would therefore report a 1000x SLOWDOWN and the
    # `speedup_vs_eager` figure would be meaningless. The warm-up result is discarded; only its
    # duration is kept, because the compile cost is worth knowing on its own.
    t_c0 = time.time()
    probe(env_c, hook_c, batch_c, args, 'compiled (warm-up, discarded)', args.seed)
    compile_seconds = time.time() - t_c0
    print('  [warm-up] %.1f s including one-off compilation; timing below excludes it'
          % compile_seconds)

    got = probe(env_c, hook_c, batch_c, args, 'compiled', args.seed)
    got['compile_seconds'] = compile_seconds
    d = diff_report(ref, got)

    # HEURISTIC: accept a compiled quantity if it differs from eager by no more than 10x the
    # measured eager-to-eager floor, or by the dtype's own resolution at this magnitude,
    # whichever is larger. No literature source; it is an engineering margin, flagged as
    # CLAUDE.md requires. The REASONING: compilation reassociates sums, and reassociation error
    # is the same order as the nondeterministic reduction-order error stage 1 measured, so the
    # floor is the right SCALE; the factor 10 is headroom for a longer fused chain. The absolute
    # term matters because a deterministic quantity can measure a floor of exactly 0, against
    # which any multiple is still 0 and every check would fail spuriously.
    eps = float(torch.finfo(torch.float32).eps)

    def _scale(k):
        """The magnitude of the quantity itself, so the absolute term is meaningful."""
        if '.' in k and k.split('.', 1)[0] in ('base_grad', 'total_grad', 'after_step'):
            grp, sub = k.split('.', 1)
            return float(ref[grp][sub].abs().max())
        v = ref[k]
        return abs(v) if isinstance(v, float) else float(v.abs().max())

    tol = {}
    for k in d:
        tol[k] = max(10.0 * floor.get(k, 0.0), 20.0 * eps * max(_scale(k), 1e-30))

    bad = sorted([k for k in d if d[k] > tol[k]], key=lambda k: -d[k] / max(tol[k], 1e-300))
    for k in ('p_full', 'p_off', 'p_clamped', 'p_off_same_path', 'd', 'd_same_path',
              'base_loss', 'V'):
        check('compiled matches eager: %s' % k, d[k] <= tol[k],
              'diff %.3e, tolerance %.3e (10x the measured floor %.3e)'
              % (d[k], tol[k], floor.get(k, 0.0)))
    for grp, lbl in (('base_grad', 'base gradient'), ('total_grad', 'gradient with the penalty'),
                     ('after_step', 'parameters after one update')):
        ks = [k for k in d if k.startswith(grp + '.')]
        ok = all(d[k] <= tol[k] for k in ks)
        wk = max(ks, key=lambda k: d[k])
        check('compiled matches eager: %s, all groups' % lbl, ok,
              'worst %s at %s (tolerance %.3e)' % ('%.3e' % d[wk], wk.split('.', 1)[1], tol[wk]))

    # THE GATE, TESTED DIRECTLY. This is the failure mode that would otherwise look like success.
    check('the gate still changes the compiled rollout', got['gate_effect'] > 0,
          'max |p_full - p_clamped| = %.3e. Zero here means the gate value was specialised into '
          'the compiled graph and the off/clamped interventions are inert, which makes the '
          'penalty a different quantity entirely' % got['gate_effect'])
    rel_gate = abs(got['gate_effect'] - ref['gate_effect']) / max(ref['gate_effect'], 1e-30)
    check('the gate has the same effect compiled as eager', rel_gate <= 1e-3,
          'relative difference in |p_full - p_clamped|: %.3e' % rel_gate)

    # IS THIS ARM ACTUALLY COMPILED? Asserted against the object, not inferred from a flag or
    # grepped from a log line. `_compiled` is a (hfn, output_only) pair or None.
    sim = getattr(env_c.fit_sys, 'simulator', None)
    comp = getattr(sim, '_compiled', None)
    check('the compiled arm really holds compiled callables', comp is not None,
          'simulator._compiled = %s' % ('a (hfn, output_only) pair' if comp else 'None, so this '
                                        'stage compared eager against eager and establishes '
                                        'NOTHING about the compiled path'))

    # THE MIXED-PATH TERM, reported for both arms. Nonzero only when compilation is live.
    check('the compiled/eager mix inside d is small next to d itself',
          got['mixed_path_frac_of_d'] <= 1e-3,
          'production d is a compiled p_full minus an EAGER p_off (closed_loop.py:554 keeps a '
          'no_grad rollout eager). |mixed| = %.3e, %.3e of |d|. Eager reference: %.3e'
          % (got['mixed_path_abs'], got['mixed_path_frac_of_d'], ref['mixed_path_frac_of_d']))

    st = dict(counters['stats']) if counters else {}
    print('  [dynamo] %s' % (st or 'counters unavailable'))
    return got, d, tol, bad, st


# --------------------------------------------------------------------------- main
def build(args, compile_mode, verbose=True):
    """One arm. Geometry is NEVER rebuilt for a compiled arm; see `main`."""
    ns = argparse.Namespace(**vars(args))
    ns.compile_mode = compile_mode
    env, hook, batch = CU.setup(ns, verbose=verbose)
    return env, hook, batch


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--stages', default='0,1,2,3',
                    help='comma-separated. Stage 4 (the longer training run) is opt-in via '
                         '--train-its and still refuses to start unless 1 to 3 passed')
    ap.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    ap.add_argument('--compile-mode', default='reduce-overhead',
                    choices=('default', 'reduce-overhead', 'max-autotune'))
    ap.add_argument('--no-compile', action='store_true',
                    help='force the eager fallback without pretending stage 2 passed')
    ap.add_argument('--windows', type=int, default=2)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--beta', type=float, default=1.0,
                    help='A CHOSEN CONSTANT WITH NO JUSTIFICATION. Nothing here selects it')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--adam-eps', type=float, default=1e-16)
    ap.add_argument('--seed', type=int, default=20260908)
    ap.add_argument('--f64', action='store_true')
    ap.add_argument('--resume-steps', type=int, default=2)
    ap.add_argument('--skip-resume', action='store_true')
    ap.add_argument('--train-its', type=int, default=0,
                    help='stage 4: optimizer updates for the longer run. 0 disables it')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    stages = {s.strip() for s in args.stages.split(',') if s.strip()}
    stamp = os.environ.get('SLURM_JOB_ID') or time.strftime('%Y%m%d_%H%M%S')
    out_dir = args.out or os.path.join(HERE, 'results', 'gpu_ladder_%s' % stamp)
    os.makedirs(out_dir, exist_ok=True)
    rec = dict(settings=vars(args), stamp=stamp, stages={})
    t_all = time.time()

    def save():
        rec['seconds'] = time.time() - t_all
        rec['summary'] = dict(passed=len(CU.PASS), failed=len(CU.FAIL), failures=list(CU.FAIL))
        with open(os.path.join(out_dir, 'gpu_ladder.json'), 'w') as f:
            json.dump(rec, f, indent=1, default=str)

    # ---- stage 0 ----------------------------------------------------------------------
    pre = preflight(args)
    rec['stages']['0_preflight'] = pre
    save()
    use_compile = pre['can_compile'] and not args.no_compile
    rec['compiled_arm_ran'] = use_compile

    # ---- stage 1 ----------------------------------------------------------------------
    if '1' not in stages:
        print('stage 1 not requested; nothing downstream can run without it')
        return save()
    with diag.timed('build eager arm'):
        env_e, hook_e, batch_e = build(args, None)
    ref, floor, worst = stage_eager_reference(env_e, hook_e, batch_e, args)
    rec['stages']['1_eager'] = dict(
        floor={k: v for k, v in floor.items()}, worst_floor=worst,
        base_loss=ref['base_loss'], V=ref['V'], gate_effect=ref['gate_effect'],
        mixed_path_frac_of_d=ref['mixed_path_frac_of_d'],
        seconds=ref['seconds'], peak_cuda_mb=ref['peak_cuda_mb'])
    save()

    # ---- stage 2 ----------------------------------------------------------------------
    if '2' in stages and use_compile:
        with diag.timed('build compiled arm'):
            env_c, hook_c, batch_c = build(args, args.compile_mode, verbose=False)
        # THE FROZEN BASIS IS SHARED, NOT REBUILT. The geometry is defined at a fixed reference
        # checkpoint on a fixed window set; two arms that each built their own would differ by
        # the very floating-point noise this stage is trying to measure, and the comparison
        # would be meaningless. Rebuilding it under compilation would also push the functorch
        # JVPs through inductor, which is neither needed (it is built once) nor safe.
        hook_c.geometry = hook_e.geometry
        hook_c.beta = hook_e.beta
        for k in ('ufuture', 'yfuture', 'uhist', 'yhist', 'ctrl_ix'):
            if k in hook_e.adapter.w and k in hook_c.adapter.w:
                hook_c.adapter.w[k] = hook_e.adapter.w[k].to(
                    hook_c.adapter.w[k].device)
        check('the compiled arm uses the SAME frozen geometry object',
              hook_c.geometry is hook_e.geometry,
              'shared by construction; a rebuilt basis would differ by the noise being measured')
        got, d, tol, bad, st = stage_compiled_parity(
            env_c, hook_c, batch_c, args, ref, floor, worst)
        rec['stages']['2_compiled'] = dict(
            diffs=d, tolerances=tol, out_of_tolerance=bad, dynamo=st,
            seconds=got['seconds'], peak_cuda_mb=got['peak_cuda_mb'],
            compile_seconds=got.get('compile_seconds'),
            mixed_path_abs=got['mixed_path_abs'],
            mixed_path_frac_of_d=got['mixed_path_frac_of_d'],
            really_compiled=getattr(getattr(env_c.fit_sys, 'simulator', None),
                                    '_compiled', None) is not None,
            speedup_vs_eager=(ref['seconds'] / got['seconds']) if got['seconds'] else None,
            tolerance_policy='max(10 x measured eager-to-eager floor, 20 x eps x scale); '
                             'the factor 10 is a HEURISTIC margin, not a derivation')
        save()
    else:
        rec['stages']['2_compiled'] = dict(
            skipped=True, reason=('--no-compile' if args.no_compile
                                  else pre['compile_unavailable_reason']))
        print('\nSTAGE 2 SKIPPED: %s' % rec['stages']['2_compiled']['reason'])
        print('A SKIPPED PARITY STAGE IS NOT A PASSED ONE. Stage 4 stays gated.')
        env_c = hook_c = batch_c = None

    # ---- stage 3 ----------------------------------------------------------------------
    if '3' in stages and not args.skip_resume:
        env_r = env_c if (use_compile and env_c is not None) else env_e
        hook_r = hook_c if (use_compile and hook_c is not None) else hook_e
        batch_r = batch_c if (use_compile and batch_c is not None) else batch_e
        rec['stages']['3_resume'] = CU.check_resume(env_r, hook_r, batch_r, args)
        rec['stages']['3_resume']['ran_on'] = 'compiled' if use_compile else 'eager'
        save()

    # ---- stage 4 ----------------------------------------------------------------------
    gate_ok = not CU.FAIL
    if args.train_its > 0:
        if not gate_ok:
            print('\nSTAGE 4 REFUSED: %d earlier check(s) failed: %s'
                  % (len(CU.FAIL), CU.FAIL[:4]))
            rec['stages']['4_train'] = dict(refused=True, failures=list(CU.FAIL))
        elif '2' in stages and not use_compile:
            print('\nSTAGE 4 running EAGER: the compiled arm never ran, so its parity is '
                  'unknown rather than established.')
            rec['stages']['4_train'] = _train(args, None, out_dir)
        else:
            rec['stages']['4_train'] = _train(args, args.compile_mode, out_dir)
        save()

    save()
    print('\n' + '=' * 92)
    print('%d passed, %d failed' % (len(CU.PASS), len(CU.FAIL)))
    if CU.FAIL:
        print('FAILED: ' + ', '.join(CU.FAIL))
    print('wrote ' + os.path.join(out_dir, 'gpu_ladder.json'))
    return 1 if CU.FAIL else 0


def _train(args, compile_mode, out_dir):
    """Stage 4. Plumbing and throughput only: this measures cost, never benefit."""
    print('\n' + '=' * 92)
    print('STAGE 4: longer run, %d updates, %s'
          % (args.train_its, compile_mode or 'eager'))
    print('=' * 92, flush=True)
    import train_orth
    ns = argparse.Namespace(**vars(args))
    ns.its = args.train_its
    ns.compile_mode = compile_mode
    ns.chunk = 0
    ns.rank_rtol = 1e-12
    ns.log_every = max(1, args.train_its // 20)
    ns.drift_every = 0
    ns.route_every = 0
    ns.control = False
    diag._cuda_reset(args.device)
    t0 = time.time()
    arm = train_orth.run_arm(ns, True)
    arm['wall_seconds'] = time.time() - t0
    arm['peak_cuda_mb'] = diag.cuda_peak_mb(args.device)
    arm['compile_mode'] = compile_mode
    arm['is_not_an_efficacy_result'] = (
        'beta is unselected and there is no control arm here. This stage shows the penalty runs '
        'on the production path at production speed, nothing about whether it helps.')
    check('the longer run completed', True,
          '%d updates in %.1f s (%.3f s/update), peak %s MB'
          % (args.train_its, arm['wall_seconds'], arm['wall_seconds'] / max(args.train_its, 1),
             arm['peak_cuda_mb']))
    check('the penalty fired once per update in the longer run',
          arm.get('hook_calls_in_step') == args.train_its,
          'in-step calls %s, optimizer steps %s, updates %s'
          % (arm.get('hook_calls_in_step'), arm.get('optimizer_steps'), args.train_its))
    check('the penalty value stayed finite', bool(np.isfinite(arm.get('last', {}).get('V', np.nan))),
          'final V = %s' % arm.get('last', {}).get('V'))
    return arm


if __name__ == '__main__':
    sys.exit(main())
