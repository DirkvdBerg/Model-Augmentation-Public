"""THE verification entry point for fixed-reference trajectory orthogonality.

Two suites, run separately or together:

    small    the mathematical reference cases (SymPy; the MATLAB twin is a separate script)
    gantry   the production ladder P0/V0-V5/V7 against a real checkpoint

Neither suite is a training run. There is no beta sweep, no parameter-recovery study and no
joint optimisation here, by design: the specification's own rule is that no long gantry
experiment should be launched to discover an already-testable simulator or gradient contract
error.

Usage
-----
    conda run -n GraduationProject python scripts/gantry/orthogonality/verify.py --suite small
    conda run -n GraduationProject python scripts/gantry/orthogonality/verify.py --suite gantry \
        --run-dir scripts/gantry/meeting/meeting-07-09-2026/server/augmentation_ma50_b140-230_a6_z03_linear_map/81757 \
        --windows 4
    conda run -n GraduationProject python scripts/gantry/orthogonality/verify.py --suite gantry \
        --perturbed --noise-sigma 1.4e-4

Results land in `scripts/gantry/orthogonality/results/<stamp>/`, one JSON per suite, with
values, residuals, tolerances, ranks, failures, configuration, seeds, timing and peak memory.
"""
__project_origin__ = "added"

import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
for p in (REPO, os.path.join(REPO, 'scripts', 'gantry'), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

RESULTS = os.path.join(HERE, 'results')
DEFAULT_RUN = os.path.join(
    REPO, 'scripts', 'gantry', 'meeting', 'meeting-07-09-2026', 'server',
    'augmentation_ma50_b140-230_a6_z03_linear_map', '81757')


def _peak_memory_mb():
    """Peak RSS in MB. Windows has no `resource`, so psutil is used where available."""
    try:
        import psutil
        return float(psutil.Process().memory_info().peak_wset) / 1e6
    except Exception:
        try:
            import resource
            return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1e3
        except Exception:
            return float('nan')


def _provenance():
    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO,
                                      stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        sha = 'unavailable'
    import torch
    return dict(git=sha, python=platform.python_version(), platform=platform.platform(),
                torch=torch.__version__, numpy=np.__version__)


# ------------------------------------------------------------------------------- small
def run_small():
    """Delegate to the SymPy reference script and return its JSON."""
    import runpy
    path = os.path.join(HERE, 'code', 'trajectory_ownership_geometry.py')
    print(f'running {path}')
    rc = subprocess.call([sys.executable, '-u', path])
    out = os.path.join(HERE, 'code', 'trajectory_ownership_geometry',
                       'trajectory_ownership_geometry_sympy.json')
    with open(out) as f:
        return dict(returncode=rc, results=json.load(f), path=out)


# ------------------------------------------------------------------------------ gantry
def run_gantry(run_dir, n_windows, records, seed, beta, rank_rtol, use_f64,
               perturb=None, noise_sigma=None, tag='', stages=('V3', 'V4', 'V6', 'V7'),
               dump_path=None, device='cpu', op_timeout=None, dump_dir=None,
               heartbeat=None):
    import torch
    # THREAD COUNT IS A RUN PARAMETER, and it is recorded rather than assumed. On this Windows
    # box `torch.set_num_threads(1)` was measured BLOCKING: the process sat at 9 s of CPU over
    # ten minutes of wall clock inside an ordinary no-grad rollout that otherwise costs under
    # two seconds. Default threading is therefore what runs, and the number that ran is saved.
    n_threads = int(os.environ.get('ORTH_VERIFY_THREADS', 0)) or torch.get_num_threads()
    torch.set_num_threads(n_threads)
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(
            'device="cuda" was requested but torch.cuda.is_available() is False. Falling back to '
            'the CPU is deliberately NOT done: the run would be far slower while the manifest '
            'recorded cuda, and the two would be compared as if they were the same measurement. '
            'On the cluster this means --gres=gpu:1 was not requested, or CUDA_VISIBLE_DEVICES '
            'was cleared by a copied CPU runner.')
    from gantry import diag
    diag.enable_fault_handler()
    from gantry.env import load_env, production_windows, to_tensors
    from gantry.adapter import GantryTrajectoryAdapter
    from gantry.encoder_split import split_encoder
    from gantry import checks as C

    t_start = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)

    print('=' * 92)
    print(f'GANTRY VERIFICATION LADDER{(" [" + tag + "]") if tag else ""}')
    print('Category: data-computable throughout. No simulation truth is read anywhere in this '
          'suite.')
    print('=' * 92)

    hb = diag.heartbeat(heartbeat,
                        os.path.join(dump_dir, 'stacks.log') if dump_dir else None)
    hb.__enter__()
    t0 = time.time()
    env = load_env(run_dir, use_f64=use_f64, joint_estimation=True, device=device)
    t_env = time.time() - t0
    man = env.manifest()
    print(json.dumps({k: man[k] for k in
                      ('nx_phys', 'nx_ann', 'nu', 'ny', 'nf', 'burn_in', 'na', 'stride',
                       'ann_n_outputs', 'route_cols_physical', 'route_cols_latent',
                       'physical_block', 'controller_n_states', 'controller_n_rows',
                       'dtype')}, indent=1))

    L = C.Ledger()
    per_rec = max(1, n_windows // len(records))
    win = production_windows(env, records, per_rec, seed=seed)
    keep = np.arange(min(n_windows, win['n_windows']))
    win_t = to_tensors(win, env.dtype, device=device)
    win_t = {k: (v[keep] if hasattr(v, 'shape') and v.shape[0] == win['n_windows'] else v)
             for k, v in win_t.items()}
    win_t['provenance'] = win['provenance']

    # ---- D. perturbed / noisy inputs, FROZEN before any paired evaluation --------------
    if perturb:
        g = torch.Generator().manual_seed(seed + 991)
        if noise_sigma:
            # THE NOISE IS GENERATED ONCE AND FROZEN. Re-sampling between a plus and a minus
            # perturbation would make every finite difference a difference of two DIFFERENT
            # objectives, and the resulting error would be noise, not a derivative check.
            for k in ('yhist', 'yfuture'):
                # Seeded on the CPU, then moved: see `checks.seeded_like` for why.
                z = torch.randn(win_t[k].shape, generator=g, dtype=torch.float64)
                win_t[k] = win_t[k] + noise_sigma * z.to(device=win_t[k].device,
                                                         dtype=win_t[k].dtype)
        print(f'[perturbed] frozen additive output noise sigma={noise_sigma} on yhist and '
              f'yfuture; u and the controller assignment are unchanged')

    # ---- P0/V0: conversions ------------------------------------------------------------
    print('\n[P0] physical-block and encoder conversions')
    p0 = _check_physical_conversion(L, env, win_t, records, seed, use_f64)
    orig_enc = env.fit_sys.encoder
    env.fit_sys.encoder = split_encoder(orig_enc).to(env.dtype)
    env.fit_sys.encoder.eval()

    print('\n[V0] ownership')
    ad = C.stage_V0(L, env, GantryTrajectoryAdapter, win_t, orig_enc)

    print('\n[V1] production simulator parity and interventions')
    contrib = C.stage_V1(L, env, ad)

    print('\n[V2] derivatives against directional finite differences')
    t0 = time.time()
    v2 = C.stage_V2(L, env, ad)
    t_v2 = time.time() - t0

    # PER-OPERATION LEDGER for the stage that has never completed. Every operation inside V3 is
    # timed and watchdogged; `ops.incomplete` names the one that never returned, which is the
    # single fact four earlier attempts could not produce.
    ops = diag.OpLedger('V3')

    v3 = v45 = v7 = None
    geom = None
    t_v3 = t_v45 = t_v6 = 0.0
    # REQUESTED versus COMPLETED. Added 2026-09-08 after review: the saved record said
    # "21 passed, 0 skipped" while three requested stages had never run, and null result fields
    # were the only trace. A stage that was asked for and did not finish is INCOMPLETE, which is
    # neither a pass nor a skip, and the summary has to say so in its own words.
    stages_requested = list(stages)
    stages_completed = []

    def assemble():
        return dict(
            tag=tag, provenance=_provenance(), manifest=man, physical_conversion=p0,
            seeds=dict(master=seed, window_selection=seed, noise=seed + 991),
            settings=dict(n_windows=int(len(keep)), records=list(records), beta=beta,
                          rank_rtol=rank_rtol, use_f64=use_f64, perturb=bool(perturb),
                          noise_sigma=noise_sigma, stages=list(stages),
                          torch_num_threads=n_threads, device=device,
                          op_timeout_s=op_timeout,
                          penalty_convention='V = (beta/N) ||Q^T d||^2, plain MSE (spec Eq. 11)'),
            stages_requested=stages_requested, stages_completed=list(stages_completed),
            stages_incomplete=[x for x in stages_requested if x not in stages_completed],
            windows=win['provenance'], V2=v2, V3=v3, V4_V5=v45, V7=v7,
            V3_operations=ops.summary(),
            timing=dict(env_seconds=t_env, V2_seconds=t_v2, V3_seconds=t_v3,
                        V45_seconds=t_v45, V6_seconds=t_v6,
                        total_seconds=time.time() - t_start),
            peak_memory_mb=_peak_memory_mb(),
            summary=dict(L.summary(),
                         stages_requested=stages_requested,
                         stages_completed=list(stages_completed),
                         stages_incomplete=[x for x in stages_requested
                                            if x not in stages_completed],
                         complete=not [x for x in stages_requested
                                       if x not in stages_completed]),
            checks=L.rows)

    def snapshot():
        """Write what is known SO FAR.

        A stage that hangs or exhausts memory must not cost the stages already completed.
        Everything finished is on disk before the next stage starts, and a stage that never
        returned is simply ABSENT from the JSON rather than reported as passing.
        """
        if dump_path:
            with open(dump_path, 'w') as f:
                json.dump(assemble(), f, indent=1, default=str)

    snapshot()

    if 'V3' in stages:
        print('\n[V3] geometry, ranks and robustness'
              + (f'   [watchdog {op_timeout}s per operation]' if op_timeout else ''))
        t0 = time.time()
        try:
            geom, S, Q_E, v3 = C.stage_V3(L, env, ad, rank_rtol=rank_rtol, ops=ops,
                                          op_timeout=op_timeout, dump_dir=dump_dir)
            stages_completed.append('V3')
        except diag.OperationTimeout as e:
            L.add('V3', 'stage_completed', False,
                  'every operation in the geometry build returned within its budget',
                  'diag.timed watchdog', 'completion', str(e), category='data-computable',
                  detail=str(e),
                  not_establishes='anything about the geometry; the stage did not finish')
            print(f'  [V3] ABORTED: {e}', flush=True)
        t_v3 = time.time() - t0
        snapshot()

    if 'V4' in stages and geom is not None:
        print('\n[V4/V5] gradient ownership and exact chunking')
        t0 = time.time()
        v45 = C.stage_V4_V5(L, env, ad, geom, beta=beta)
        t_v45 = time.time() - t0
        stages_completed.append('V4')
        snapshot()

    if 'V6' in stages and geom is not None:
        print('\n[V6] execution safety')
        from gantry import checks_v6 as C6
        t0 = time.time()
        C6.check_feature_disabled_is_a_no_op(L, env, ad)
        C6.check_functional_call_is_clean(L, env, ad)
        C6.check_gate_checkpoint_replay(L, env, ad)
        C6.check_accumulation_hook_survives_zero_grad(L, env, ad, geom, beta=beta)
        t_v6 = time.time() - t0
        stages_completed.append('V6')
        snapshot()

    if 'V7' in stages and geom is not None:
        print('\n[V7] interpretation')
        v7 = C.stage_V7(L, env, ad, geom, contrib)
        stages_completed.append('V7')
        snapshot()

    summary = L.summary()
    print('\n' + '=' * 92)
    print(f"{summary['passed']} passed, {summary['failed']} failed, {summary['skipped']} skipped")
    if summary['failures']:
        print('FAILED: ' + ', '.join(summary['failures']))

    hb.__exit__(None, None, None)
    snapshot()
    return assemble()


def _check_physical_conversion(L, env, win_t, records, seed, use_f64):
    """The parameterised physical block must reproduce the non-parameterised one at lambda = 0.

    This is what makes `S` meaningful for a checkpoint trained with the physical parameters
    fixed: the geometry is the sensitivity of the SAME predictor, differentiated in coordinates
    that the trained run held constant, not of a different model.
    """
    import torch
    from gantry.env import load_env, to_tensors
    from gantry.adapter import GantryTrajectoryAdapter

    ref = load_env(env.run_dir, use_f64=use_f64, joint_estimation=False, verbose=False,
                   data_norm=(env.data, env.norm), device=str(env.device))
    with torch.no_grad():
        x = ref.fit_sys.encoder(win_t['uhist'], win_t['yhist'])
        kw = dict(ctrl_ix=win_t['ctrl_ix']) if 'ctrl_ix' in win_t else {}
        y_ref, _ = ref.fit_sys.simulate(x, win_t['ufuture'], win_t['yfuture'], **kw)
        x2 = env.fit_sys.encoder(win_t['uhist'], win_t['yhist'])
        y_new, _ = env.fit_sys.simulate(x2, win_t['ufuture'], win_t['yfuture'], **kw)
    e = float((y_ref - y_new).abs().max())
    scale = max(float(y_ref.abs().max()), 1e-300)
    lp = env.phys_block.log_params
    theta = env.phys_block._recover_params().detach()
    init = env.phys_block.params_init.detach()
    clamped = int((theta <= 1e-6 * (1 + 1e-12)).sum())
    dist = float((theta / init).min()), float((theta / init).max())
    L.add('P0', 'physical_block_conversion_parity', e <= 1e-10 * scale,
          'the Parameterized_Gantry_State_Block at log_params = 0 reproduces the trained '
          'Gantry_State_Block predictions exactly',
          'blocks.py _recover_params: theta = params_init * exp(0) = params_init',
          'relative difference at round-off', e / scale, tol=1e-10, category='data-computable',
          detail=f'max |dy| / |y| = {e/scale:.3e}',
          establishes='S is the physical sensitivity of the predictor that was actually trained',
          not_establishes='that the run optimised those coordinates; it did not, and the anchor '
                          'is off in this suite by construction')
    L.add('P0', 'no_active_parameter_clamp', clamped == 0,
          'no physical parameter sits on the 1e-6 clamp boundary, so the smooth-argument '
          'requirement for the sensitivity holds',
          'spec Eq. (2) and its warrant', 'zero active clamps', clamped, tol=0,
          category='data-computable',
          detail=f'theta/theta_init in [{dist[0]:.6f}, {dist[1]:.6f}], '
                 f'min theta = {float(theta.min()):.4e} against the 1e-6 floor')
    # RELEASE THE REFERENCE MODEL. It is a second complete fit system, built only to check the
    # physical-block conversion, and on a 16 GB box that is enough to push the forward-mode
    # rollouts into paging: the first attempt at this suite spent 35 minutes accumulating 42 s
    # of CPU inside one JVP, which is a memory symptom, not a slow one.
    import gc
    del ref
    gc.collect()
    return dict(max_rel=e / scale, n_clamped=clamped,
                theta=theta.cpu().numpy().tolist(),
                param_names=list(env.phys_block.PARAM_NAMES))


# --------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--suite', choices=('small', 'gantry', 'all'), default='all')
    ap.add_argument('--run-dir', default=DEFAULT_RUN)
    ap.add_argument('--windows', type=int, default=4)
    ap.add_argument('--records', type=int, nargs='*', default=[0, 5, 8, 12],
                    help='train-record indices; the default spans standstill, y-sweep, aprbs '
                         'and lissajous excitation')
    ap.add_argument('--seed', type=int, default=20260908)
    ap.add_argument('--beta', type=float, default=1.0,
                    help='a CHOSEN CONSTANT with no justification (RULES.md 2026-09-07): this '
                         'suite verifies identities, and no value of beta is selected here')
    ap.add_argument('--rank-rtol', type=float, default=1e-12)
    ap.add_argument('--f32', action='store_true', help='run at the checkpoint dtype instead of '
                                                       'the float64 geometry reference')
    ap.add_argument('--device', choices=('cpu', 'cuda'), default='cpu',
                    help='where the rollouts run. cuda requires --gres=gpu:1 on the cluster and '
                         'a runner that does NOT clear CUDA_VISIBLE_DEVICES')
    ap.add_argument('--heartbeat', type=float, default=None,
                    help='seconds between whole-run stack dumps. Covers every stage, not just '
                         'V3: across four stalls the blocked call appeared in V1, V2 and V3, so '
                         'a per-stage watchdog alone would have missed two of them')
    ap.add_argument('--op-timeout', type=float, default=None,
                    help='seconds before an operation inside V3 dumps every thread stack. The '
                         'stage then aborts with a recorded failure instead of hanging until '
                         'SLURM kills the job. Omit to disable the watchdog')
    ap.add_argument('--perturbed', action='store_true')
    ap.add_argument('--stages', nargs='*', default=['V3', 'V4', 'V6', 'V7'],
                    help='stages beyond the always-run P0/V0/V1/V2')
    ap.add_argument('--perturbed-stages', nargs='*', default=[],
                    help='stages beyond P0/V0/V1/V2 for the perturbed arm; empty by design')
    ap.add_argument('--noise-sigma', type=float, default=1.4e-4,
                    help='frozen additive output noise, in NORMALISED units of y')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    stamp = time.strftime('%Y%m%d_%H%M%S')
    out_dir = a.out or os.path.join(RESULTS, stamp)
    os.makedirs(out_dir, exist_ok=True)
    wrote = []

    if a.suite in ('small', 'all'):
        r = run_small()
        p = os.path.join(out_dir, 'small_reference.json')
        with open(p, 'w') as f:
            json.dump(r, f, indent=1, default=str)
        wrote.append(p)

    if a.suite in ('gantry', 'all'):
        p = os.path.join(out_dir, 'gantry_nominal.json')
        run_gantry(a.run_dir, a.windows, a.records, a.seed, a.beta, a.rank_rtol,
                   use_f64=not a.f32, tag='nominal', stages=tuple(a.stages), dump_path=p,
                   device=a.device, op_timeout=a.op_timeout, dump_dir=out_dir,
                   heartbeat=a.heartbeat)
        wrote.append(p)
        if a.perturbed:
            # Part D repeats SELECTED identity and derivative checks under changed numerical
            # inputs. It is deliberately NOT a second full ladder: a rank measured under one
            # frozen noise draw is not a statistical statement about noise, so repeating the
            # geometry stages there would invite exactly the reading the specification forbids.
            p2 = os.path.join(out_dir, 'gantry_perturbed.json')
            run_gantry(a.run_dir, a.windows, a.records, a.seed + 1, a.beta, a.rank_rtol,
                       use_f64=not a.f32, perturb=True, noise_sigma=a.noise_sigma,
                       tag='perturbed-noisy', stages=tuple(a.perturbed_stages), dump_path=p2,
                       device=a.device, op_timeout=a.op_timeout, dump_dir=out_dir,
                       heartbeat=a.heartbeat)
            wrote.append(p2)

    print('\nwrote:')
    for p in wrote:
        print('  ' + p)

    # EXIT STATUS. Added 2026-09-08 after review: this returned 0 unconditionally, so a run with
    # recorded failures, or with requested stages that never finished, was indistinguishable
    # from a clean one to anything downstream of it.
    bad = []
    for path in wrote:
        try:
            with open(path) as f:
                rec = json.load(f)
        except Exception as e:
            bad.append(f'{os.path.basename(path)}: unreadable ({e})')
            continue
        res = rec.get('results', rec)
        summ = res.get('summary') or {}
        if summ.get('failed'):
            bad.append(f"{os.path.basename(path)}: {summ['failed']} failed "
                       f"({', '.join(summ.get('failures', [])[:4])})")
        inc = summ.get('stages_incomplete')
        if inc:
            bad.append(f"{os.path.basename(path)}: requested stages did not complete: "
                       f"{', '.join(inc)}")
        if res.get('returncode'):
            bad.append(f"{os.path.basename(path)}: sub-run exited {res['returncode']}")
    if bad:
        print('\nNOT CLEAN:')
        for b in bad:
            print('  ' + b)
        return 1
    print('\nall requested stages completed with no recorded failures')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
