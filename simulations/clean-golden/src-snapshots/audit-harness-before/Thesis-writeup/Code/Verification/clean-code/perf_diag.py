"""Step-1 baseline diagnostics of Clean/ (tasks/handoffs/2026-10-26-clean-code-opt/01-baseline-claude.md, items 2, 3).

    python -u perf_diag.py --out DIR              both rows, one child process each (fresh Dynamo state)
    python -u perf_diag.py --out DIR --row 226    one row

Builds the model as main() does (seed, data, normalisation, seed, build_model, build_closed_loop, encoder
pre-fit) with the gpu_smoke.py budget on device cpu, compile_mode None, then runs, in this order:
    validation   one eager validation (fit's first step, completes the lazy init; timed apart)
    2a           torch._dynamo.explain on fs.hfn and fs.hfn.output_only with the first rollout step's inputs
    2b           aot_eager fullgraph closed-loop step (D-284 M17) in closed_loop_rollout: loss and every gradient
                 against eager, from identical state and batch, no optimizer step
    3            CPU eager: 2 warm-up updates, 10 timed, 1 profiled; then the same on CUDA if available,
                 with the CUDA sync debug mode over one more update
    2c           recompiles of the aot_eager step over 8 consecutive training updates
    2d           optional: backend inductor on CPU
    2e           one whole closed-loop time step under aot_eager fullgraph
A training update is fit()'s loop body: optimizer.step(closure).item(), the PS2 after_step, the counters.
Nothing in Clean/ is modified; the instrumentation patches module globals in THIS process only.
Writes DIR/<row>/result.json, the profiler tables and the recompile log.
"""
import argparse
import os
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--row', type=int, choices=(221, 226), default=None)
args = ap.parse_args()
OUT = os.path.abspath(args.out)

if args.row is None:
    os.makedirs(OUT, exist_ok=True)
    for row in (226, 221):
        print('=== perf_diag row %d ===' % row, flush=True)
        rc = subprocess.call([sys.executable, '-u', os.path.abspath(__file__), '--out', OUT, '--row', str(row)])
        print('=== row %d exit %d ===' % (row, rc), flush=True)
    sys.exit(0)

import copy
import json
import logging
import platform
import statistics
import time
import traceback
import warnings

import hcommon as H

ROW_OUT = os.path.join(OUT, str(args.row))
if os.path.exists(os.path.join(ROW_OUT, 'result.json')):
    sys.exit('%s has a result.json' % ROW_OUT)
os.makedirs(ROW_OUT, exist_ok=True)
os.environ['LOCALAPPDATA'] = os.path.join(ROW_OUT, 'localappdata')     # deepSI dumps stay in the output
os.makedirs(os.environ['LOCALAPPDATA'], exist_ok=True)
BUDGET = dict(truncate_s=2.0, n_its=4, its_per_val=4, lbfgs_inner_iter=1, lbfgs_max_iter=2, lbfgs_windows=256)

H.select_side('clean')
import numpy as np
import torch
import torch._dynamo
from torch._dynamo.utils import counters

cfg, EP = H.make_cfg(args.row, 'lowpass', run_id='perfdiag-%d' % args.row, budget=BUDGET)
import gantry_dynamic.data as GDD
import gantry_dynamic.model as GDM
from gantry_dynamic.controller import build_closed_loop
from gantry_dynamic.training import PS2Opening, PS2Spec
import model_augmentation.fit_systems.closed_loop as CL
from model_augmentation.fit_systems.interconnect import DtypeDataLoader
from deepSI.fit_systems.fit_system import loop_dataset
H.assert_roots()

R = dict(row=args.row, case='%d-lowpass' % args.row, budget=BUDGET, device='cpu', compile_mode=None,
         preset=H.ROWS[args.row][0], arm=cfg.arm, n_a=cfg.n_a, ps2=bool(cfg.ps2), obc=bool(cfg.obc),
         libraries=H.lib_versions(), threads=H.thread_settings(),
         hardware=dict(platform=platform.platform(), processor=platform.processor(), cpu_count=os.cpu_count(),
                       cuda=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                       cuda_version=torch.version.cuda),
         clean_tree=H.clean_tree_hashes())


def save():
    with open(os.path.join(ROW_OUT, 'result.json'), 'w') as fh:
        json.dump(R, fh, indent=1, default=str)


def fail(e):
    return dict(ok=False, error='%s: %s' % (type(e).__name__, str(e).splitlines()[0] if str(e) else ''),
                traceback=traceback.format_exc()[-3000:])


# ---- build, as main() does
T0 = time.time()
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
data = GDD.load_datasets(cfg)
norm = GDD.compute_normalization(cfg, data)
hp = cfg.hp
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
fs = GDM.build_model(hp, cfg, data, norm)
fs.simulator = build_closed_loop(fs, norm, cfg, train_files=data.train_files, val_files=data.val_files,
                                 val_data=data.val_ckpt_data, e_train=data.e_train, e_val=data.e_val)
GDM.prefit_physical_encoder(fs, cfg, data, norm)
sim = fs.simulator
R['t_build_s'] = time.time() - T0

# ---- train()'s setup and fit()'s, without fit()
fs.unique_code = 'perfdiag-%d' % args.row
opening = None
if cfg.ps2:
    spec = PS2Spec(seed=cfg.seed, eval_every=2, s1_min=0, s1_cap=4, screen_at=-1, screen_max=float('inf'),
                   n_cal=32, n_snap=256, n_snap_cal=64, s2_batch=64, s2_check=50, s2_min=50, s2_cap=100)
    opening = PS2Opening(fs, data, n_xb=cfg.nx_phys, n_a=cfg.nx_ann, width=hp['n_nodes_per_layer'], spec=spec,
                         route_ix=cfg.ann_route_ix)
    fs.training_phase = opening
fs.train()
fs.epoch_counter, fs.batch_counter = 0, 0
loss_kwargs = {'nf': hp['nf'], 'stride': cfg.stride}
data_train = fs.make_training_data(fs.norm.transform(data.train_data), **loss_kwargs)
N = len(data_train[0])
bs = min(hp['batch_size'], N)
n_per_epoch = N // bs
dtype = next(fs.hfn.parameters()).dtype
loader = DtypeDataLoader(data_train, batch_size=bs, dtype=dtype)
batches = loop_dataset(loader)
R['workload'] = dict(dtype=str(dtype), batch_size=bs, n_windows=N, updates_per_epoch=n_per_epoch,
                     window_length_nf=hp['nf'], na=fs.na, nb=fs.nb, nx=fs.hfn.nx, seed=cfg.seed,
                     n_controllers=sim.bank.n_controllers, nc=sim.bank.nc,
                     batch_order='DtypeDataLoader default_rng(0), as fit()')
print('[perf_diag] row %d workload %s' % (args.row, R['workload']), flush=True)

# ---- instrumentation: module-global patch of the rollout (the simulator and the validation call it by name)
_rollout = CL.closed_loop_rollout
MODE = dict(compiled=None, labels=False, first=None)
rf = torch.profiler.record_function


def _label(name, fn):
    def w(*a, **k):
        with rf(name):
            return fn(*a, **k)
    return w


class _LabelledHfn:
    """fs.hfn and its output_only under profiler labels, for a closed_loop_step (D-284 M17)."""

    def __init__(self, hfn):
        self.hfn = hfn
        self.output_only = _label('diag.output_only', hfn.output_only)

    def __call__(self, *a):
        with rf('diag.hfn'):
            return self.hfn(*a)


def patched_rollout(step, u_data, y_data, x0, bank, ctrl_ix, obc_pass=None, pass_mats=None):
    if MODE['first'] is None and torch.is_grad_enabled():
        # the inputs of the first step of a gradient-enabled rollout, for 2a and 2e
        MODE['first'] = dict(x=x0.detach().clone(), u0=u_data[:, 0].detach().clone(),
                             y0=y_data[:, 0].detach().clone(), ctrl_ix=ctrl_ix.clone(),
                             obc_pass=obc_pass, pass_mats=pass_mats)
    if MODE['compiled'] is not None and torch.is_grad_enabled():
        step = MODE['compiled']
    if MODE['labels']:
        bank_step = bank.step
        bank.step = _label('diag.bank_step', bank_step)
        try:
            return _rollout(CL.closed_loop_step(_LabelledHfn(fs.hfn), bank), u_data, y_data, x0, bank, ctrl_ix,
                            obc_pass, pass_mats)
        finally:
            del bank.step
    return _rollout(step, u_data, y_data, x0, bank, ctrl_ix, obc_pass, pass_mats)


CL.closed_loop_rollout = patched_rollout


def to_dev(batch, dev):
    return [b.to(dev) for b in batch]


def update(batch, labels=False):
    """fit()'s loop body for one batch; returns the training loss."""
    kw = dict(loss_kwargs, **dict(zip(sim.extra_array_names, batch[4:])))
    ctx = rf if labels else (lambda name: _Null())

    def closure():
        with ctx('diag.loss'):
            L = fs.loss(*batch[:4], **kw)
        with ctx('diag.zero_grad'):
            fs.optimizer.zero_grad()
        with ctx('diag.backward'):
            L.backward()
        return L

    with ctx('diag.opt_step'):
        loss = fs.optimizer.step(closure).item()
    if fs.training_phase is not None:
        with ctx('diag.ps2'):
            fs.training_phase.after_step(fs)
            fs.ps2_opening = (fs.training_phase.phase, fs.training_phase.it)
    fs.batch_counter += 1
    fs.epoch_counter += 1 / n_per_epoch
    return loss


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def sync(dev):
    if dev == 'cuda':
        torch.cuda.synchronize()


# ---- validation, eager (fit's first step)
fs.eval()
fs._sync_obc_coefficient(refresh_basis=False)
t = time.perf_counter()
val0 = fs.cal_validation_error(data.val_ckpt_data, cfg.validation_measure)
R['validation_eager'] = dict(seconds=time.perf_counter() - t, score=float(val0),
                             n_records=len(data.val_ckpt_data.sdl),
                             lengths=sorted({len(sd.u) for sd in data.val_ckpt_data.sdl}))
fs.train()
print('[perf_diag] validation %.3f s score %.6g' % (R['validation_eager']['seconds'], val0), flush=True)
save()

# one fixed batch for 2a/2b (the first of the loader stream would be consumed; draw it from the windows instead)
ix = np.sort(np.random.default_rng(0).choice(N, bs, replace=False))
fixed = [torch.as_tensor(np.asarray(a)[ix]) for a in data_train]
fixed = [b.to(dtype) if b.is_floating_point() else b for b in fixed]


def named_grads():
    out = {}
    for gname in fs.parameters_with_names:
        mod = getattr(fs, gname)
        named = list(mod.named_parameters()) if isinstance(mod, torch.nn.Module) else [('', mod)]
        for pname, p in named:
            out['%s.%s' % (gname, pname)] = None if p.grad is None else p.grad.detach().clone()
    return out


def loss_and_grads(batch):
    kw = dict(loss_kwargs, **dict(zip(sim.extra_array_names, batch[4:])))
    fs.optimizer.zero_grad(set_to_none=True)
    L = fs.loss(*batch[:4], **kw)
    L.backward()
    return L.detach().clone(), named_grads()


state0 = dict(hfn=copy.deepcopy(fs.hfn.state_dict()), enc=copy.deepcopy(fs.encoder.state_dict()),
              rng=torch.get_rng_state(), nprng=np.random.get_state())


def restore():
    fs.hfn.load_state_dict(state0['hfn'])
    fs.encoder.load_state_dict(state0['enc'])
    torch.set_rng_state(state0['rng'])
    np.random.set_state(state0['nprng'])


# eager reference loss and gradients (also captures the first-step inputs)
restore()
basis_before = None if fs.obc is None else (id(fs.obc.basis), fs.obc.last_refresh_epoch)
L_eager, G_eager = loss_and_grads(fixed)
first = MODE['first']

# ---- 2a: explain
R['2a_explain'] = {}
for name, fn, a in (('hfn', fs.hfn, (first['x'], first['u0'], first['obc_pass'], first['pass_mats'])),
                    ('output_only', fs.hfn.output_only, (first['x'],))):
    try:
        torch._dynamo.reset()
        ex = torch._dynamo.explain(fn)(*a)
        R['2a_explain'][name] = dict(ok=True, graph_count=ex.graph_count, graph_break_count=ex.graph_break_count,
                                     op_count=ex.op_count,
                                     break_reasons=[dict(reason=str(b.reason),
                                                         where=[str(f) for f in (b.user_stack or [])][-3:])
                                                    for b in ex.break_reasons])
    except Exception as e:
        R['2a_explain'][name] = fail(e)
    print('[perf_diag] 2a %s %s' % (name, {k: v for k, v in R['2a_explain'][name].items() if k != 'traceback'}),
          flush=True)
torch._dynamo.reset()
save()

# ---- 2b: aot_eager fullgraph pair, paired with eager
try:
    # D-284 M17: the compiled unit is the closed-loop step
    pair = torch.compile(CL.closed_loop_step(fs.hfn, sim.bank), backend='aot_eager', fullgraph=True)
    restore()
    MODE['compiled'] = pair
    t = time.perf_counter()
    L_comp, G_comp = loss_and_grads(fixed)
    t_first = time.perf_counter() - t
    MODE['compiled'] = None
    basis_after = None if fs.obc is None else (id(fs.obc.basis), fs.obc.last_refresh_epoch)
    per = {}
    for k in G_eager:
        a, b = G_eager[k], G_comp.get(k)
        if a is None or b is None:
            per[k] = dict(eager_none=a is None, compiled_none=b is None, equal=(a is None and b is None))
        else:
            d = (a - b).abs().max().item() if a.numel() else 0.0
            per[k] = dict(equal=bool(torch.equal(a, b)), max_abs=d,
                          max_rel=d / max(a.abs().max().item(), 1e-300) if a.numel() else 0.0)
    R['2b_paired'] = dict(ok=True, loss_eager=float(L_eager), loss_compiled=float(L_comp),
                          loss_equal=bool(torch.equal(L_eager, L_comp)),
                          loss_rel=abs(float(L_eager - L_comp)) / abs(float(L_eager)),
                          n_grads=len(per), n_equal=sum(v['equal'] for v in per.values()),
                          worst_rel=max((v.get('max_rel', 0.0) for v in per.values()), default=0.0),
                          grads=per, first_call_seconds_incl_compile=t_first,
                          obc_basis_unchanged=basis_before == basis_after,
                          unique_graphs=int(counters['stats']['unique_graphs']))
except Exception as e:
    MODE['compiled'] = None
    pair = None
    R['2b_paired'] = fail(e)
print('[perf_diag] 2b %s' % {k: v for k, v in R['2b_paired'].items() if k not in ('grads', 'traceback')},
      flush=True)
restore()
fs.optimizer.zero_grad(set_to_none=True)
save()


# ---- 3: where eager time goes
def time_device(dev):
    out = dict(device=dev)
    if dev == 'cuda':
        fs.cuda()
    try:
        for _ in range(2):
            update(to_dev(next(batches), dev))
        sync(dev)
        ts, losses = [], []
        for _ in range(10):
            b = to_dev(next(batches), dev)
            sync(dev)
            t = time.perf_counter()
            losses.append(update(b))
            sync(dev)
            ts.append(time.perf_counter() - t)
        out.update(timed_s=ts, median_s=statistics.median(ts), min_s=min(ts), max_s=max(ts), losses=losses,
                   ps2_phase_after=None if opening is None else (opening.phase, opening.it))
        # one profiled update with the exclusive split
        b = to_dev(next(batches), dev)
        acts = [torch.profiler.ProfilerActivity.CPU] + ([torch.profiler.ProfilerActivity.CUDA] if dev == 'cuda' else [])
        MODE['labels'] = True
        sync(dev)
        with torch.profiler.profile(activities=acts) as prof:
            t = time.perf_counter()
            update(b, labels=True)
            sync(dev)
            wall = time.perf_counter() - t
        MODE['labels'] = False
        ka = prof.key_averages()
        tot = {e.key: e.cpu_time_total / 1e6 for e in ka if e.key.startswith('diag.')}
        cnt = {e.key: e.count for e in ka if e.key.startswith('diag.')}
        g = lambda k: tot.get(k, 0.0)
        split = {'hfn (forward)': g('diag.hfn'), 'output_only': g('diag.output_only'),
                 'bank.step': g('diag.bank_step'),
                 'loss other (encoder, pass_mats, OBC, stack, objective)':
                     g('diag.loss') - g('diag.hfn') - g('diag.output_only') - g('diag.bank_step'),
                 'backward': g('diag.backward'), 'zero_grad': g('diag.zero_grad'),
                 'optimizer step (Adam own)': g('diag.opt_step') - g('diag.loss') - g('diag.zero_grad')
                 - g('diag.backward'),
                 'PS2 after_step': g('diag.ps2')}
        split['Python and other overhead (unlabelled)'] = wall - g('diag.opt_step') - g('diag.ps2')
        out['profiled'] = dict(wall_s=wall, split_s=split, split_frac={k: v / wall for k, v in split.items()},
                               range_counts=cnt)
        sort = 'self_cuda_time_total' if dev == 'cuda' else 'self_cpu_time_total'
        tab = ka.table(sort_by='self_cpu_time_total', row_limit=25)
        with open(os.path.join(ROW_OUT, 'profile_%s_top25_selfcpu.txt' % dev), 'w', encoding='utf-8') as fh:
            fh.write(tab)
        if dev == 'cuda':
            with open(os.path.join(ROW_OUT, 'profile_cuda_top25_selfcuda.txt'), 'w', encoding='utf-8') as fh:
                fh.write(ka.table(sort_by=sort, row_limit=25))
        out['top25_selfcpu'] = [dict(op=e.key, self_cpu_s=e.self_cpu_time_total / 1e6, count=e.count)
                                for e in sorted([e for e in ka if not e.key.startswith('diag.')],
                                                key=lambda e: -e.self_cpu_time_total)[:25]]
        if dev == 'cuda':
            # sync sites over one update, attributed to the innermost Clean/ frame of the warning's stack
            sites = {}
            old_show = warnings.showwarning

            def show(message, category, filename, lineno, file=None, line=None):
                if 'synchroniz' not in str(message).lower():
                    return old_show(message, category, filename, lineno, file, line)
                st = traceback.extract_stack()[:-1]
                cl = [f for f in st if os.path.abspath(f.filename).startswith(H.CLEAN + os.sep)]
                key = ('%s:%d' % (os.path.relpath(cl[-1].filename, H.REPO).replace(os.sep, '/'), cl[-1].lineno)
                       if cl else '%s:%d (outside Clean)' % (os.path.basename(filename), lineno))
                s = sites.setdefault(key, dict(count=0, message=str(message).splitlines()[0][:160],
                                               code=(cl[-1].line if cl else line)))
                s['count'] += 1

            b = to_dev(next(batches), dev)
            with warnings.catch_warnings():
                warnings.simplefilter('always')
                warnings.showwarning = show
                torch.cuda.set_sync_debug_mode('warn')
                try:
                    update(b)
                finally:
                    torch.cuda.set_sync_debug_mode(0)
                    warnings.showwarning = old_show
            out['sync_sites'] = dict(sorted(sites.items(), key=lambda kv: -kv[1]['count']))
        out['ok'] = True
    except Exception as e:
        out.update(fail(e))
    finally:
        MODE['labels'] = False
        if dev == 'cuda':
            fs.cpu()
    return out


R['3_timing'] = {}
for dev in ('cpu',) + (('cuda',) if torch.cuda.is_available() else ()):
    R['3_timing'][dev] = time_device(dev)
    r = R['3_timing'][dev]
    print('[perf_diag] 3 %s median %.4f s range [%.4f, %.4f] split %s' % (
        dev, r.get('median_s', float('nan')), r.get('min_s', float('nan')), r.get('max_s', float('nan')),
        {k: round(v, 3) for k, v in r.get('profiled', {}).get('split_frac', {}).items()}), flush=True)
    save()

# ---- 2c: recompiles of the compiled pair over 8 consecutive training updates
rec_log = []


class _H(logging.Handler):
    def emit(self, record):
        rec_log.append(dict(logger=record.name, msg=record.getMessage()[:2000]))


try:
    torch._logging.set_logs(recompiles=True, graph_breaks=True)
    for n in list(logging.root.manager.loggerDict):
        if 'recompiles' in n or 'graph_breaks' in n:
            logging.getLogger(n).addHandler(_H())
    if pair is None:
        raise RuntimeError('no compiled pair from 2b')
    torch._dynamo.reset()
    counters.clear()
    # D-284 M17: the compiled unit is the closed-loop step
    pair = torch.compile(CL.closed_loop_step(fs.hfn, sim.bank), backend='aot_eager', fullgraph=True)
    MODE['compiled'] = pair
    per_update = []
    for i in range(8):
        t = time.perf_counter()
        L = update(next(batches))
        per_update.append(dict(update=i, seconds=time.perf_counter() - t, loss=L,
                               unique_graphs=int(counters['stats']['unique_graphs']),
                               frames_ok=int(counters['frames']['ok']),
                               recompile_msgs=len(rec_log)))
    MODE['compiled'] = None
    R['2c_recompiles'] = dict(ok=True, per_update=per_update, log=rec_log,
                              unique_graphs=int(counters['stats']['unique_graphs']),
                              graph_breaks=dict(counters['graph_break']),
                              ps2_phase_after=None if opening is None else (opening.phase, opening.it))
except Exception as e:
    MODE['compiled'] = None
    R['2c_recompiles'] = fail(e)
    R['2c_recompiles']['log'] = rec_log
with open(os.path.join(ROW_OUT, 'recompiles.log'), 'w', encoding='utf-8') as fh:
    for r in rec_log:
        fh.write('[%s] %s\n' % (r['logger'], r['msg']))
print('[perf_diag] 2c %s' % {k: v for k, v in R['2c_recompiles'].items() if k not in ('log', 'traceback', 'per_update')},
      flush=True)
torch._logging.set_logs()
save()

# ---- 2d: inductor on CPU (optional)
try:
    torch._dynamo.reset()
    c = torch.compile(fs.hfn, backend='inductor', fullgraph=True)
    with torch.no_grad():
        y_c, x_c = c(first['x'], first['u0'], first['obc_pass'], first['pass_mats'])
        y_e, x_e = fs.hfn(first['x'], first['u0'], first['obc_pass'], first['pass_mats'])
    R['2d_inductor_cpu'] = dict(ok=True, max_abs_y=(y_c - y_e).abs().max().item(),
                                max_abs_x=(x_c - x_e).abs().max().item())
except Exception as e:
    R['2d_inductor_cpu'] = fail(e)
print('[perf_diag] 2d %s' % {k: v for k, v in R['2d_inductor_cpu'].items() if k != 'traceback'}, flush=True)
save()

# ---- 2e: one whole closed-loop time step (output, controller step, state update)
bank = sim.bank
ctrl = (bank.M_state[first['ctrl_ix']], bank.M_error[first['ctrl_ix']])


def closed_loop_step(x, xc, u_t, y_t, M_state, M_error, obc_pass, pass_mats):
    y_model = fs.hfn.output_only(x)
    u_fb, xc = bank.step(xc, y_t - y_model, (M_state, M_error))
    _, x = fs.hfn(x, u_t + u_fb, obc_pass, pass_mats)
    return y_model, x, xc


xc0 = torch.zeros(first['x'].shape[0], bank.nc, dtype=first['x'].dtype)
step_args = (first['x'], xc0, first['u0'], first['y0'], ctrl[0], ctrl[1], first['obc_pass'], first['pass_mats'])
try:
    torch._dynamo.reset()
    ex = torch._dynamo.explain(closed_loop_step)(*step_args)
    e_out = dict(graph_count=ex.graph_count, graph_break_count=ex.graph_break_count, op_count=ex.op_count,
                 break_reasons=[dict(reason=str(b.reason), where=[str(f) for f in (b.user_stack or [])][-3:])
                                for b in ex.break_reasons])
    torch._dynamo.reset()
    cstep = torch.compile(closed_loop_step, backend='aot_eager', fullgraph=True)
    with torch.no_grad():
        oc = cstep(*step_args)
        oe = closed_loop_step(*step_args)
    e_out.update(ok=True, fullgraph_compiled=True,
                 bitwise_equal_no_grad=[bool(torch.equal(a, b)) for a, b in zip(oc, oe)],
                 max_abs=[(a - b).abs().max().item() for a, b in zip(oc, oe)])
    R['2e_step'] = e_out
except Exception as e:
    R['2e_step'] = dict(fail(e), **(e_out if 'e_out' in dir() else {}))
print('[perf_diag] 2e %s' % {k: v for k, v in R['2e_step'].items() if k != 'traceback'}, flush=True)

CL.closed_loop_rollout = _rollout
H.assert_roots()
R['clean_tree_unchanged'] = H.clean_tree_hashes() == R['clean_tree']
R['seconds_total'] = time.time() - T0
save()
print('PERF_DIAG_DONE row %d %.1f s' % (args.row, R['seconds_total']), flush=True)
