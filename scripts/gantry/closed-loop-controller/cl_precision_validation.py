"""Does float32 move the VALIDATION score, and can it flip checkpoint selection?

WHY THIS IS A DIFFERENT QUESTION FROM cl_precision_gradient.py
--------------------------------------------------------------
That script asked whether float32 corrupts the training GRADIENT over an nf = 400 window. It does
not: the float32/float64 gradient disagreement is 1.8e-06 of the batch-to-batch scatter.

This asks about the VALIDATION metric, which is a different object:

  * it is a FULL-RECORD free run, ~48000 steps rather than 400, so anything that amplifies
    rounding has two orders of magnitude more steps to do it in;
  * it is used for CHECKPOINT SELECTION, and the gaps it has to resolve are small. The step-6 run
    moved the score 2.187e-06 -> 1.393e-06 m over 12 epochs, so consecutive checkpoints differ by
    far less than that;
  * a score is a reported number, and reporting the float64 rollout of float32-trained weights is
    reporting a different object than the one that was optimised. Defensible, but only if the two
    are known to agree.

THE DECISION THIS FEEDS
-----------------------
  scores agree well inside the checkpoint-to-checkpoint gaps  -> keep validation in float32, i.e.
                                                                 the same arithmetic as training,
                                                                 and say so once
  scores differ, or the RANKING of two checkpoints flips      -> the validation metric is partly
                                                                 numerical noise; that is a finding
                                                                 for the problem log, not something
                                                                 to fix by choosing float64

WHAT IS MEASURED
----------------
Both available checkpoints (`_best` and `_last` of the step-6 run) scored on all four validation
records, closed loop, full record from k0, in float32 and in float64. Same weights, same data,
same rollout function; only the arithmetic differs. The selection scalar is the quadratic mean
over records, matching deepSI's `System_data.RMS` aggregation (see `cl_validation.rms_phys`).

The `_best` vs `_last` gap is the natural yardstick: it is a real difference between two models
that selection is expected to resolve. If the precision difference is a meaningful fraction of it,
selection is partly random.

Usage
-----
  PYTHONUNBUFFERED=1 python -u cl_precision_validation.py
"""
__project_origin__ = "added"

import dataclasses
import glob
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
GANTRY = os.path.join(REPO, 'scripts', 'gantry')
for p in (REPO, GANTRY, HERE, os.path.join(GANTRY, 'drift-demo'),
          os.path.join(GANTRY, 'msd-offset')):
    if p not in sys.path:
        sys.path.insert(0, p)

import demo_common as dm                                                  # noqa: E402
from demo_common import CFG                                               # noqa: E402
from gantry_dynamic.data import load_traj, VAL_FILES                      # noqa: E402
import cl_plant as PLANT                                                  # noqa: E402
import cl_validation as CV                                                # noqa: E402
from cl_controller import ControllerBank                                  # noqa: E402
import cl_fitsys as CLF                                                   # noqa: E402

SRV = os.path.join(HERE, 'server-results')
CH = ['X1', 'X2', 'Y']
t0 = time.time()


def build(use_f64, lr):
    cfg = dataclasses.replace(CFG, seed=0, ann_route_ix=tuple(range(8)), lr=lr,
                              use_f64=use_f64)
    fs, norm, K0, na, nb, na_r, nb_r = dm.build_pipeline(cfg=cfg, verbose=False)
    nx = cfg.nx_phys + cfg.nx_ann
    C_out, b_out = PLANT.identify_output_map(fs.hfn, nx, cfg.nu, dtype=cfg.dtype_pt)
    step_fn, out_fn = PLANT.make_fns(fs, C_out, b_out)
    vnames = [f[:-4] for f in VAL_FILES]
    bank = ControllerBank(vnames, cfg.ts_new, dtype=cfg.dtype_pt,
                          ystd=norm.ystd, std_u=norm.std_u)
    CLF.attach(fs, bank, step_fn, out_fn)      # BEFORE torch.load, see cl_plot_step6.py
    return dict(cfg=cfg, fs=fs, K0=K0, dims=(na, nb, na_r, nb_r), bank=bank,
                step_fn=step_fn, out_fn=out_fn, vnames=vnames)


def load_state(fs, ckpt):
    ck = torch.load(ckpt, weights_only=False)
    fs.hfn.load_state_dict(ck['hfn'].state_dict())        # copy_ casts into this build's dtype
    fs.encoder.load_state_dict(ck['encoder'].state_dict())


def score(env, name, slot):
    """Closed-loop full-record free run. Returns (per-channel rms [m], aggregate [m])."""
    cfg, fs = env['cfg'], env['fs']
    na, nb, na_r, nb_r = env['dims']
    rv = lambda a: np.asarray(a).ravel()                                  # noqa: E731
    sd = load_traj(name + '.mat', cfg)
    un = ((sd.u - rv(fs.norm.u0)) / rv(fs.norm.ustd)).astype(cfg.dtype_np)
    yn = ((sd.y - rv(fs.norm.y0)) / rv(fs.norm.ystd)).astype(cfg.dtype_np)
    x0 = CV.encoder_x0(fs.encoder, un, yn, env['K0'], na, nb, na_r, nb_r, cfg.dtype_pt)
    ctrl = env['bank'].gather(torch.tensor([slot], dtype=torch.long))
    y_pred = CV.free_run(env['step_fn'], env['out_fn'], un, yn, x0,
                         env['bank'], ctrl, k0=env['K0'], closed=True)
    return CV.rms_phys(y_pred, sd.y[env['K0']:], rv(fs.norm.ystd), rv(fs.norm.y0))


print('=' * 100)
print('PRECISION vs the VALIDATION SCORE and checkpoint selection')
print('=' * 100)

ckpts = sorted(glob.glob(os.path.join(SRV, '**', '*.pth'), recursive=True))
if not ckpts:
    raise SystemExit('no .pth under %s' % SRV)
res_path = sorted(glob.glob(os.path.join(SRV, 'step6_result_*.json')))
res = json.load(open(res_path[0])) if res_path else {}
lr = res.get('lr', 1e-7)
print('checkpoints: %s' % ', '.join(os.path.basename(c) for c in ckpts))
print('run 76573 reference: val %.4e -> %.4e m over %s epochs'
      % (res.get('base', float('nan')), res.get('final', float('nan')), res.get('epochs', '?')))

print('\nbuilding float32 ...', flush=True)
e32 = build(False, lr)
print('building float64 ...', flush=True)
e64 = build(True, lr)
vnames = e32['vnames']
print('records: %s' % ', '.join(vnames))
print('k0 = %d, full record from there, closed loop\n' % e32['K0'])

agg = {}
for ck in ckpts:
    tag = os.path.basename(ck).replace('FitSys_ClosedLoop_', '').replace('.pth', '')
    load_state(e32['fs'], ck)
    load_state(e64['fs'], ck)
    print('=' * 100)
    print('%s' % tag, flush=True)
    print('  %-20s %-14s %-14s %-12s %-12s' % ('record', 'rms f32 [m]', 'rms f64 [m]',
                                               '|diff| [m]', 'rel diff'))
    per_rec = {}
    for slot, nm in enumerate(vnames):
        _, a32 = score(e32, nm, slot)
        _, a64 = score(e64, nm, slot)
        d = abs(a32 - a64)
        print('  %-20s %-14.6e %-14.6e %-12.3e %-12.3e'
              % (nm, a32, a64, d, d / a64), flush=True)
        per_rec[nm] = (a32, a64)
    q32 = float(np.sqrt(np.mean([v[0] ** 2 for v in per_rec.values()])))
    q64 = float(np.sqrt(np.mean([v[1] ** 2 for v in per_rec.values()])))
    print('  %-20s %-14.6e %-14.6e %-12.3e %-12.3e   <- selection scalar'
          % ('QUADRATIC MEAN', q32, q64, abs(q32 - q64), abs(q32 - q64) / q64))
    agg[tag] = (q32, q64, per_rec)
    print('  [%.0fs]' % (time.time() - t0), flush=True)

print('\n' + '=' * 100)
print('VERDICT')
print('=' * 100)
tags = list(agg)
if len(tags) >= 2:
    a, b = tags[0], tags[1]
    gap32 = agg[a][0] - agg[b][0]
    gap64 = agg[a][1] - agg[b][1]
    prec = max(abs(agg[t][0] - agg[t][1]) for t in tags)
    print('checkpoint gap (%s minus %s):  float32 %+.4e   float64 %+.4e' % (a, b, gap32, gap64))
    print('worst precision shift on a score:              %.4e' % prec)
    print('precision shift / checkpoint gap:              %.3g'
          % (prec / max(abs(gap64), 1e-30)))
    flip = (gap32 > 0) != (gap64 > 0)
    print('ranking flips between precisions:              %s' % ('YES' if flip else 'no'))
    print('')
    if flip:
        print('-> Selection is precision-dependent. The validation metric is partly numerical')
        print('   noise at this checkpoint spacing. Log it; do not paper over it with float64.')
    elif prec < 0.1 * abs(gap64):
        print('-> The precision shift is well inside the gap selection has to resolve. Keep')
        print('   validation in float32, matching training, and state that once.')
    else:
        print('-> The precision shift is a material fraction of the checkpoint gap. Selection is')
        print('   not obviously safe at this spacing; worth a closer look before reporting.')
else:
    print('only one checkpoint found, ranking not testable')
    for t in tags:
        print('%s: f32 %.6e  f64 %.6e  rel %.3e'
              % (t, agg[t][0], agg[t][1], abs(agg[t][0] - agg[t][1]) / agg[t][1]))
print('[%.0fs]' % (time.time() - t0))
