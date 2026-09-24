"""BB-013: the FRF-init closed loop's margin to random-sign parameter perturbations. No optimizer.

For each delta and sign draw: theta + delta * s (s random in {-1, +1}, never a gradient sign),
then loss on the three G3 batches (forward only, no_grad) and the closed-loop validation free run
on the 6 records with the BB-007 stability rule. The model is restored between copies.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'adapter'))
import bbcl                                                                # noqa: E402

cfg, CFG, entry = bbcl.load_cfg(local=True)
import numpy as np                                                         # noqa: E402
import torch                                                               # noqa: E402
from gantry_dynamic.data import load_datasets, compute_normalization, VAL_FILES  # noqa: E402
from model_augmentation.fit_systems.closed_loop import window_controller_index   # noqa: E402
import bb_model                                                            # noqa: E402
import cl_score                                                            # noqa: E402

DELTAS = (1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6)
DRAWS = 3
np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
data = load_datasets(cfg)
norm = compute_normalization(cfg, data)
fs = bb_model.build_blackbox(cfg, data, norm, arm='frf', attach=True)
sim = fs.simulator

# The three G3 batches, same sampler and seed as init/g3_init.py (BB-011).
_, counts = window_controller_index(data.train_data, sim.train_ctrl_rows, fs.na, fs.nb, cfg.nf,
                                    fs.na_right, fs.nb_right, cfg.stride)
N, B = int(sum(counts)), int(cfg.batch_size)
rng = np.random.default_rng(cfg.seed)
picks = [np.sort(rng.choice(N, size=B, replace=False)) for _ in range(3)]
offs = np.concatenate([[0], np.cumsum(counts)])
batches = [[[], [], [], [], []] for _ in picks]
for r, sd in enumerate(data.train_data.sdl):
    arr = fs.norm.transform(sd).to_hist_future_data(na=fs.na, nb=fs.nb, nf=cfg.nf, na_right=fs.na_right,
                                                    nb_right=fs.nb_right, stride=cfg.stride)
    assert len(arr[0]) == counts[r]
    for bi, pk in enumerate(picks):
        loc = pk[(pk >= offs[r]) & (pk < offs[r + 1])] - offs[r]
        for a in range(4):
            batches[bi][a].append(arr[a][loc])
        batches[bi][4].append(np.full(len(loc), sim.train_ctrl_rows[r], dtype=np.int64))
    del arr
batches = [[torch.as_tensor(np.concatenate(bt[a]), dtype=cfg.dtype_pt) for a in range(4)]
           + [torch.as_tensor(np.concatenate(bt[4]))] for bt in batches]

params = [q for m in (fs.encoder, fs.hfn) for q in m.parameters()]
theta0 = [q.detach().clone() for q in params]
rows = [r for _, _, r in sim.val_records]
w = np.array([len(sd.u) for sd in data.val_list], float)


def evaluate():
    with torch.no_grad():
        losses = [float(fs.loss(*b[:4], ctrl_ix=b[4])) for b in batches]
        tr = cl_score.closed_loop_traces(fs, data.val_list, sim.bank, rows)
    rms = np.array([t['rms'] for t in tr])
    return dict(losses=losses, val=float(np.average(rms, weights=w)) if np.all(np.isfinite(rms)) else float('nan'),
                n_stable=int(sum(t['stable'] for t in tr)),
                ok=bool(all(np.isfinite(losses)) and all(t['stable'] for t in tr)))


res = dict(deltas=list(DELTAS), draws=DRAWS, runs=[])
t0 = time.time()
base = evaluate()
res['unperturbed'] = base
print('[margin] unperturbed: losses %s  val %.4e  stable %d/6  (%.0f s)'
      % (np.array2string(np.array(base['losses']), precision=3), base['val'], base['n_stable'], time.time() - t0))
for d in DELTAS:
    for k in range(DRAWS):
        g = np.random.default_rng(cfg.seed + k)
        with torch.no_grad():
            for q, q0 in zip(params, theta0):
                s = torch.as_tensor(g.choice([-1.0, 1.0], size=tuple(q.shape)), dtype=q.dtype)
                q.copy_(q0 + d * s)
        r = evaluate()
        r.update(delta=d, draw=k)
        res['runs'].append(r)
        print('[margin] delta %.0e draw %d: losses %s  val %.4e  stable %d/6  -> %s'
              % (d, k, np.array2string(np.array(r['losses']), precision=3), r['val'], r['n_stable'],
                 'ok' if r['ok'] else 'UNSTABLE'))
with torch.no_grad():
    for q, q0 in zip(params, theta0):
        q.copy_(q0)
ok_by_d = {d: all(r['ok'] for r in res['runs'] if r['delta'] == d) for d in DELTAS}
star = None
for d in sorted(DELTAS, reverse=True):
    if all(ok_by_d[e] for e in DELTAS if e <= d):
        star = d
        break
res['ok_by_delta'] = {str(d): v for d, v in ok_by_d.items()}
res['delta_star'] = star
res['lr_server'] = None if star is None else star / 10
print('[margin] all-draws-stable by delta: %s' % ok_by_d)
print('[margin] delta* = %s  ->  server lr = %s (BB-013: delta*/10)' % (star, res['lr_server']))
os.makedirs(os.path.join(bbcl.OUT, 'g4'), exist_ok=True)
json.dump(res, open(os.path.join(bbcl.OUT, 'g4', 'lr_margin.json'), 'w'), indent=1)
