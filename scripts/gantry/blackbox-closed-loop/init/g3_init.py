"""G3: black box on the production data at initialisation, one arm per process (BB-008, BB-011).

    python init/g3_init.py --arm {random,n4sid,frf} [--model {bb,gb}]

Measures (1) the closed-loop free run on the 6 validation records through the production
simulator (`cal_validation_error`, the selection number) plus per-record stability; (2) the same
with a zero controller (open-loop context, same harness); (3) three batches of cfg.batch_size
training windows, forward + backward through `fit_sys.loss(..., ctrl_ix=...)`, NO optimizer step.
`--model gb` runs step (3) only on the grey box, for the BB-009 per-update cost ratio.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'adapter'))
import bbcl                                                                # noqa: E402

cfg, CFG, entry = bbcl.load_cfg(local=True)
import numpy as np                                                         # noqa: E402
import torch                                                               # noqa: E402
from gantry_dynamic.data import load_datasets, compute_normalization, TRAIN_FILES, VAL_FILES  # noqa: E402
from gantry_dynamic.controller import build_closed_loop                    # noqa: E402
from model_augmentation.fit_systems.closed_loop import ControllerBank, window_controller_index  # noqa: E402
import bb_model                                                            # noqa: E402
import cl_score                                                            # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--arm', choices=bb_model.ARMS, default='random')
p.add_argument('--model', choices=('bb', 'gb'), default='bb')
p.add_argument('--n-batches', type=int, default=3)
args = p.parse_args()
tag = args.arm if args.model == 'bb' else 'greybox'
OUT = os.path.join(bbcl.OUT, 'g3')
os.makedirs(OUT, exist_ok=True)
print(bbcl.describe_cfg(cfg, CFG))

np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
data = load_datasets(cfg)
norm = compute_normalization(cfg, data)
if args.model == 'bb':
    fs = bb_model.build_blackbox(cfg, data, norm, arm=args.arm, attach=True)
else:
    from gantry_dynamic.model import build_model
    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    fs = build_model(cfg.hp, cfg, data, norm)
    fs.simulator = build_closed_loop(fs, norm, cfg, train_files=TRAIN_FILES, val_files=VAL_FILES,
                                     val_data=data.val_ckpt_data)
sim = fs.simulator
res = dict(arm=tag, model=args.model)

# ------------------------------------------------ (1)+(2) closed-loop validation at init
if args.model == 'bb':
    fs.eval()
    t0 = time.time()
    v = float(fs.cal_validation_error(data.val_ckpt_data, validation_measure='sim-RMS'))
    res['val_seconds'] = time.time() - t0
    per = [float(x) for x in sim.last_per_record]
    rows = [r for _, _, r in sim.val_records]
    tr = cl_score.closed_loop_traces(fs, data.val_list, sim.bank, rows, per_record_ref=per)
    b = sim.bank
    z = np.zeros
    zb = ControllerBank(z((1, b.nc, b.nc)), z((1, b.nc, b.ny)), z((1, b.nu, b.nc)),
                        z((1, b.nu, b.ny)), ystd=norm.ystd, std_u=norm.std_u, dtype=cfg.dtype_pt)
    names = [f[:-4] for f in VAL_FILES]
    ol, ol_per, _, _, _ = cl_score.closed_loop_score(fs, names, data.val_list, cfg.ts_new,
                                                     bank=zb, rows=[0] * len(names))
    tr_ol = cl_score.closed_loop_traces(fs, data.val_list, zb, [0] * len(names), per_record_ref=ol_per)
    res.update(val_cl=v, val_ol=float(ol), per_record={})
    for n, r, t, ro, to in zip(VAL_FILES, per, tr, ol_per, tr_ol):
        res['per_record'][n] = dict(cl=r, cl_stable=t['stable'], cl_finite=t['finite'],
                                    cl_max_abs_err=t['max_abs_err'], max_abs_y=t['max_abs_y'],
                                    ol=float(ro), ol_stable=to['stable'])
        print('   %-24s closed %.4e (%s, max|e| %.3e vs max|y| %.3e)   open %.4e (%s)'
              % (n, r, 'stable' if t['stable'] else 'UNSTABLE', t['max_abs_err'], t['max_abs_y'],
                 ro, 'stable' if to['stable'] else 'UNSTABLE'))
    res['n_stable'] = int(sum(t['stable'] for t in tr))
    print('[G3] %s: initial closed-loop val sim-RMS %.4e m (%d/%d records stable), open-loop %.4e m,'
          ' validation %.1f s' % (tag, v, res['n_stable'], len(tr), ol, res['val_seconds']))
    fs.train()

# ------------------------------------------------ (3) gradients, no optimizer step (BB-011 sampler)
tr_sdl = data.train_data.sdl
k0 = max(fs.na, fs.nb)
_, counts = window_controller_index(data.train_data, sim.train_ctrl_rows, fs.na, fs.nb, cfg.nf,
                                    fs.na_right, fs.nb_right, cfg.stride)
N = int(sum(counts))
rng = np.random.default_rng(cfg.seed)
B = int(cfg.batch_size)
picks = [np.sort(rng.choice(N, size=B, replace=False)) for _ in range(args.n_batches)]
offs = np.concatenate([[0], np.cumsum(counts)])
batches = [[[], [], [], [], []] for _ in picks]
for r, sd in enumerate(tr_sdl):
    arr = fs.norm.transform(sd).to_hist_future_data(na=fs.na, nb=fs.nb, nf=cfg.nf,
                                                    na_right=fs.na_right, nb_right=fs.nb_right,
                                                    stride=cfg.stride)
    assert len(arr[0]) == counts[r], (r, len(arr[0]), counts[r])
    for bi, pk in enumerate(picks):
        loc = pk[(pk >= offs[r]) & (pk < offs[r + 1])] - offs[r]
        for a in range(4):
            batches[bi][a].append(arr[a][loc])
        batches[bi][4].append(np.full(len(loc), sim.train_ctrl_rows[r], dtype=np.int64))
    del arr
res.update(n_windows=N, windows_per_record=[int(c) for c in counts], batch_size=B, grads=[])
print('[G3] %d training windows (%s per record); %d batches of %d' % (N, counts[0], len(picks), B))
params =[q for m in (fs.encoder, fs.hfn) for q in m.parameters() if q.requires_grad]
for bi, bt in enumerate(batches):
    uh, yh, uf, yf = (torch.as_tensor(np.concatenate(bt[a]), dtype=cfg.dtype_pt) for a in range(4))
    cix = torch.as_tensor(np.concatenate(bt[4]))
    for q in params:
        q.grad = None
    t0 = time.time()
    L = fs.loss(uh, yh, uf, yf, ctrl_ix=cix)
    t_f = time.time() - t0
    L.backward()
    t_fb = time.time() - t0
    g = [q.grad for q in params if q.grad is not None]
    finite = bool(all(torch.isfinite(x).all() for x in g))
    gn = float(torch.sqrt(sum((x.double() ** 2).sum() for x in g))) if g else 0.0
    d = dict(loss=float(L), sqrt_loss=float(L) ** 0.5 if float(L) >= 0 else float('nan'),
             grad_norm=gn, grad_finite=finite, n_grad_tensors=len(g), n_params=len(params),
             t_forward=t_f, t_forward_backward=t_fb)
    res['grads'].append(d)
    print('   batch %d: loss %.4e (sqrt %.3e)  |grad| %.4e  finite %s  (%d/%d tensors)  fwd %.1f s, '
          'fwd+bwd %.1f s' % (bi, d['loss'], d['sqrt_loss'], gn, finite, len(g), len(params), t_f, t_fb))
for q in params:
    q.grad = None
gok = all(np.isfinite(d['loss']) and d['grad_finite'] and np.isfinite(d['grad_norm'])
          and d['grad_norm'] > 1e-12 for d in res['grads'])
res['grad_ok'] = bool(gok)
if args.model == 'bb':
    res['arm_pass'] = bool(gok and res['n_stable'] == len(VAL_FILES))
    print('[G3] %s: gradients %s; arm %s' % (tag, 'OK' if gok else 'NOT OK',
                                            'PASS' if res['arm_pass'] else 'FAIL'))
json.dump(res, open(os.path.join(OUT, tag + '.json'), 'w'), indent=1)
