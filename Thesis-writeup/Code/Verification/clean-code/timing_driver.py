"""Compiled timing of one training update, per code arm (D-284 step 5, item 2; server_timing.sbatch).

    python -u timing_driver.py --arm mainline|pre-round|final --out DIR [--device cuda] [--compile reduce-overhead]
                               [--warmup 20] [--timed 100]
    python -u timing_driver.py --summarise OUT       median and range per arm over the repeats, pairwise outcome,
                                                     compile seconds from each arm's cold-cache pass

One pass = one process = one arm. Arms: mainline (scripts/gantry, model_augmentation), and the `pre-round` and
`final` snapshots of simulations/clean-golden/src-snapshots/, each put on sys.path as Clean/ is (the snapshot's
files are checked against its MANIFEST.sha256 first). Row 221 (R1: OBC and PS2) on lowpass with the gpu_smoke.py
budget, so every arm runs the same seed, data, windows and DtypeDataLoader batch order. Validation runs only once,
before the first update (its_per_val above n_its), so no validation or checkpoint falls inside the timed window.
A training update is fit()'s loop body: from the start of one Adam optimizer.step to the start of the next (the
closure, the step, the PS2 after_step and the counters). A step pre-hook on fs.optimizer only (not the PS2 S2 or
L-BFGS optimizers) calls torch.cuda.synchronize() and reads the clock; after warmup + timed updates the pass
writes DIR/result.json and stops, before the polish. Recorded beside the times: the PS2 phase and the OBC
last_refresh_epoch after every update, the first three training losses (a check that the arms run the same
workload), the Dynamo compile times and counters, and the peak GPU memory over the whole pass and over the timed
window. Nothing in the code roots is modified; the hooks live in THIS process only.
"""
import argparse
import json
import os
import statistics
import sys

ap = argparse.ArgumentParser()
ap.add_argument('--arm', choices=('mainline', 'pre-round', 'final'))
ap.add_argument('--out')
ap.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
ap.add_argument('--compile', default='reduce-overhead', choices=('reduce-overhead', 'none'))
ap.add_argument('--warmup', type=int, default=20)      # HEURISTIC: step-5 protocol, compile and CUDA-graph recording fall in it
ap.add_argument('--timed', type=int, default=100)      # HEURISTIC: step-5 protocol
ap.add_argument('--summarise', default=None, help='the folder server_timing.sbatch wrote')
args = ap.parse_args()

ARMS = ('mainline', 'pre-round', 'final')


def summarise(out):
    """Per arm: median s/update of each repeat, their median and range; per pair the outcome of step 5 item 3."""
    res = {}
    for name in sorted(os.listdir(out)):
        p = os.path.join(out, name, 'result.json')
        if os.path.exists(p):
            r = json.load(open(p))
            res.setdefault(r['arm'], {}).setdefault('trace' if r['traced'] else 'timing', []).append((name, r))
    rep = {'arms': {}, 'pairs': {}}
    for arm in ARMS:
        runs = res.get(arm, {})
        t = [r for _, r in runs.get('timing', []) if r['complete']]
        a = dict(passes=[n for n, _ in runs.get('timing', [])], complete=len(t))
        if t:
            med = [r['median_s'] for r in t]
            cold = [r for r in t if r.get('cache') == 'cold']
            # compile and CUDA-graph seconds from the cold-cache pass; the warm passes list theirs beside it
            a.update(median_s_per_repeat=med, median_s=statistics.median(med), range_s=[min(med), max(med)],
                     cold=dict(first_update_s=cold[0]['first_update_s'], compile_s=cold[0]['compile_s'],
                               warmup_excess_s=cold[0]['warmup_excess_s']) if cold else None,
                     cache=[r.get('cache') for r in t],
                     first_update_s=[r['first_update_s'] for r in t],
                     compile_s=[r['compile_s'] for r in t],
                     warmup_excess_s=[r['warmup_excess_s'] for r in t],
                     peak_mem_gb=[r['peak_mem_gb'] for r in t], peak_mem_timed_gb=[r['peak_mem_timed_gb'] for r in t],
                     first_losses=[r['first_losses'] for r in t])
        for n, r in runs.get('trace', []):
            a['trace'] = dict(pass_=n, graph_break=r['counters'].get('graph_break', {}),
                              recompiles=r['counters'].get('recompiles', {}),
                              unique_graphs=r['counters'].get('stats', {}).get('unique_graphs'),
                              cudagraph_skips=r['counters'].get('inductor', {}).get('cudagraph_skips'))
        rep['arms'][arm] = a
    # step 5 item 3: faster or slower only when the repeat ranges do not overlap, otherwise indistinguishable
    for x, y in (('final', 'pre-round'), ('final', 'mainline'), ('pre-round', 'mainline')):
        rx, ry = rep['arms'][x].get('range_s'), rep['arms'][y].get('range_s')
        if rx is None or ry is None:
            v = 'missing'
        elif rx[1] < ry[0]:
            v = '%s faster' % x
        elif rx[0] > ry[1]:
            v = '%s slower' % x
        else:
            v = 'indistinguishable'
        rep['pairs']['%s vs %s' % (x, y)] = v
    print(json.dumps(rep, indent=1, default=str))
    with open(os.path.join(out, 'timing-summary.json'), 'w') as fh:
        json.dump(rep, fh, indent=1, default=str)
    print('TIMING_SUMMARY %s' % json.dumps(rep['pairs']))


if args.summarise:
    summarise(os.path.abspath(args.summarise))
    sys.exit(0)
if not (args.arm and args.out):
    ap.error('--arm and --out are required for a pass')

OUT = os.path.abspath(args.out)
if os.path.exists(OUT):
    sys.exit('%s exists' % OUT)
os.makedirs(OUT)
os.environ['LOCALAPPDATA'] = os.path.join(OUT, 'localappdata')     # deepSI dumps stay in the output
os.makedirs(os.environ['LOCALAPPDATA'])

import hcommon as H

# SHA256 of each snapshot's MANIFEST.sha256 (snapshot.json manifest_sha256), so a stale copy on the server is refused
MANIFEST_SHA256 = {'pre-round': '45bdf761403f8a7f6c507018e871423f3764d635069c5721b089706d05c3fe52',
                   'final': '98d23ae244c2fa4cd0774f7fc2a6b48e1e8997129d473fa76f541e26ceae1f99'}
SNAP = None
if args.arm != 'mainline':
    SNAP = os.path.join(H.GOLDEN, 'src-snapshots', args.arm)
    if H.sha256_file(os.path.join(SNAP, 'MANIFEST.sha256')) != MANIFEST_SHA256[args.arm]:
        sys.exit('the %s snapshot is not the one this driver times (MANIFEST.sha256 hash differs)' % args.arm)
    H.CLEAN = os.path.join(SNAP, 'Clean')       # the snapshot is the clean root: sys.path, H2 hashes, assert_roots
    listed = {}
    for line in open(os.path.join(SNAP, 'MANIFEST.sha256')):
        h, p = line.split(None, 1)
        listed[p.strip().lstrip('*')] = h
    if H.clean_tree_hashes() != listed:
        sys.exit('the %s snapshot does not match its MANIFEST.sha256' % args.arm)
H.select_side('mainline' if args.arm == 'mainline' else 'clean')

import socket
import subprocess
import time

import torch
from torch._dynamo.utils import counters, compile_times
from torch.optim.optimizer import register_optimizer_step_pre_hook

N_ITS = args.warmup + args.timed + 1
BUDGET = dict(truncate_s=2.0, n_its=N_ITS, its_per_val=10 ** 9, lbfgs_inner_iter=1, lbfgs_max_iter=2,
              lbfgs_windows=256, device=args.device,
              compile_mode=None if args.compile == 'none' else args.compile)
run_id = 'timing-221-%s' % args.arm
extra = {} if SNAP is None else dict(data_root=os.path.join(H.REPO, 'Thesis-writeup', 'Data'), out_root=OUT)
cfg, EP = H.make_cfg(221, 'lowpass', run_id=run_id, budget=BUDGET, **extra)
from model_augmentation.fit_systems import interconnect as _IC
_MODEL = _IC.AugmentedModel if SNAP else _IC.SSE_Interconnect_Composed
H.assert_roots()
CUDA = args.device == 'cuda'


class _Stop(BaseException):
    """Ends the pass after the timed window; a BaseException, so no `except Exception` of the code roots holds it."""


ST = dict(fs=None, t=[], state=[], losses=[], t0=time.perf_counter(), peak_pre=0.0, peak_timed=None, stop='')
_fit, _loss = _MODEL.fit, _MODEL.loss


def _fit_rec(self, *a, **k):
    ST['fs'] = self
    _fit(self, *a, **k)
    ST['stop'] = 'fit returned after %d updates' % len(ST['t'])
    raise _Stop


def _loss_rec(self, *a, **k):
    L = _loss(self, *a, **k)
    if torch.is_grad_enabled() and len(ST['losses']) < 3 and len(ST['t']) <= 3:
        ST['losses'].append(float(L.detach()))
    return L


def _state(fs):
    tp, obc = getattr(fs, 'training_phase', None), getattr(fs, 'obc', None)
    return dict(ps2=None if tp is None else [tp.phase, tp.it],
                obc_refresh_epoch=None if obc is None else getattr(obc, 'last_refresh_epoch', None))


def _pre_step(opt, a, k):
    fs = ST['fs']
    if fs is None or opt is not fs.optimizer:
        return
    if CUDA:
        torch.cuda.synchronize()
    ST['t'].append(time.perf_counter())
    n = len(ST['t'])                 # update n starts now; the state is the one after update n - 1
    if n > 1:
        ST['state'].append(_state(fs))
    if n == args.warmup + 1 and CUDA:
        ST['peak_pre'] = torch.cuda.max_memory_allocated() / 1e9
        torch.cuda.reset_peak_memory_stats()
    if n == N_ITS:
        if CUDA:
            ST['peak_timed'] = torch.cuda.max_memory_allocated() / 1e9
        ST['stop'] = 'timed window complete'
        raise _Stop


_MODEL.fit, _MODEL.loss = _fit_rec, _loss_rec
register_optimizer_step_pre_hook(_pre_step)


def _gpu_info():
    try:
        return subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
                              capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


print('[timing] arm %s device %s compile %s warmup %d timed %d out %s'
      % (args.arm, args.device, args.compile, args.warmup, args.timed, OUT), flush=True)
try:
    if SNAP is None:
        EP.save_dir = lambda cfg: OUT
        EP.evaluate_and_save = lambda *a, **k: None
        EP.state_recovery_diagnostic = lambda *a, **k: None
        EP.compute_gradient_norms = lambda *a, **k: ({}, {})
        EP.main()
    else:
        EP.main(cfg, run_id=run_id)
    ST['stop'] = ST['stop'] or 'main returned'
except _Stop:
    pass

t = ST['t']
d = [b - a for a, b in zip(t, t[1:])]                   # d[i] = update i + 1
warm, timed = d[:args.warmup], d[args.warmup:args.warmup + args.timed]
complete = len(timed) == args.timed
med = statistics.median(timed) if timed else None
try:
    names, secs = compile_times(repr='csv', aggregate=True)
    ctimes = {n: float(s) for n, s in zip(names, secs)}
except Exception as e:                                # the record is diagnostic; a format change must not lose the pass
    ctimes = {'error': repr(e)}
# Dynamo compile (forward graphs and their backend) plus the backward graphs, which compile at the first backward;
# the other entries are nested inside these two
compile_s = (sum(v for n, v in ctimes.items() if n == '_compile.compile_inner' or n.endswith('bw_compiler'))
             if 'error' not in ctimes else None)
res = dict(
    arm=args.arm, snapshot=None if SNAP is None else json.load(open(os.path.join(SNAP, 'snapshot.json'))),
    sources=H.source_hashes(), complete=complete, stop=ST['stop'],
    traced=bool(os.environ.get('TORCH_TRACE')), torch_logs=os.environ.get('TORCH_LOGS'),
    device=args.device, compile_mode=BUDGET['compile_mode'], budget=BUDGET, row=221, noise='lowpass',
    warmup=args.warmup, timed=args.timed,
    host=socket.gethostname(), gpu=_gpu_info() if CUDA else None,
    cuda=torch.version.cuda, torch=torch.__version__, libraries=H.lib_versions(), threads=H.thread_settings(),
    torch_threads=torch.get_num_threads(),
    cache=os.environ.get('TIMING_CACHE'),               # cold or warm, set by server_timing.sbatch
    cache_dirs={k: os.environ.get(k) for k in ('TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR')},
    setup_s=(t[0] - ST['t0']) if t else None,          # build, encoder pre-fit and the initial validation
    first_update_s=d[0] if d else None,                # includes Dynamo + Inductor compilation and CUDA-graph recording
    compile_s=compile_s, compile_times=ctimes,
    warmup_excess_s=(sum(warm) - len(warm) * med) if med is not None else None,
    median_s=med, timed_range_s=[min(timed), max(timed)] if timed else None,
    warmup_s=warm, timed_s=timed,
    state_after_update=ST['state'], first_losses=ST['losses'],
    # the peak is reset at the window start: the whole-pass peak is the larger of the readings before and after it
    peak_mem_gb=max(ST.get('peak_pre', 0.0), torch.cuda.max_memory_allocated() / 1e9) if CUDA else None,
    peak_mem_timed_gb=ST['peak_timed'],
    counters={k: dict(v) for k, v in counters.items()})
with open(os.path.join(OUT, 'result.json'), 'w') as fh:
    json.dump(res, fh, indent=1, default=str)
print('TIMING_DONE %s' % json.dumps(dict(arm=args.arm, complete=complete, stop=ST['stop'], median_s=med,
                                         first_update_s=res['first_update_s'], compile_s=res['compile_s'],
                                         peak_mem_timed_gb=res['peak_mem_timed_gb'])), flush=True)
sys.exit(0 if complete else 1)
