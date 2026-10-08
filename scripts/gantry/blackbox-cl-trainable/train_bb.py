"""Trainability run (DECISIONS.md BB2-013, BB2-014): Jan's plain SUBNET, closed loop through K1,
window curriculum, one seed, one data version. No monkey patching: each stage is one ordinary
`train_model(..., nf=stage)` call; per-axis errors come from the declared `validation_probes`
hook calling the pipeline's scorer; losses from fit's own Loss_train / Loss_val arrays.

Local:  python -u train_bb.py --version noisy --seed 1 --its-per-stage 100 --out outputs/d/noisy_s1
Server: python -u train_bb.py --server ... (stride 10, batch 512, the entry file's device)
"""
__project_origin__ = "added"

import argparse
import json
import os
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--version', choices=('noisy', 'noisefree'), required=True)
p.add_argument('--seed', type=int, required=True)
p.add_argument('--its-per-stage', type=int, required=True)
p.add_argument('--stages', default='10,25,50,100,200,400,400')     # BB2-015
p.add_argument('--nx', type=int, default=16)
p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--out', required=True)
p.add_argument('--server', action='store_true')
args = p.parse_args()

OUT = os.path.abspath(args.out)
os.makedirs(OUT, exist_ok=True)
# deepSI's fit() writes _best/_last checkpoints under LOCALAPPDATA on Windows: keep them in OUT
os.environ['LOCALAPPDATA'] = os.path.join(OUT, 'localappdata')
os.makedirs(os.path.join(OUT, 'localappdata', 'deepSI', 'checkpoints'), exist_ok=True)

import numpy as np                                                     # noqa: E402
import torch                                                           # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bbcl'))
from setup import cfg_for, load                                        # noqa: E402
import jan_model as J                                                  # noqa: E402
from model_augmentation.fit_systems.closed_loop import closed_loop_free_run_rms_batch  # noqa: E402
from gantry_dynamic.model import train_model                          # noqa: E402

if not args.server:
    torch.set_num_threads(4)

# The probe's state lives here, at module level: deepSI pickles the whole model at every
# checkpoint and reloads `_best` at the end of each fit(), so a probe object must be picklable
# (a module-level class, no local closure) and its log must not live inside the pickled object.
CTX = dict(sdl=None, rows=None, bank=None, ymax=None, t0=None, nf=None)
VALS = []


class Probe:
    """validation_probes hook: per-axis closed-loop free-run RMS through the pipeline's scorer."""

    def __call__(self, fs, val, value):
        with np.errstate(all='ignore'):
            pc, _ = closed_loop_free_run_rms_batch(fs, CTX['sdl'], CTX['bank'], CTX['rows'])
        pc = np.asarray(pc, float)
        stable = bool(np.all(np.isfinite(pc)) and all(np.all(pc[i] <= 10 * CTX['ymax'][i])
                                                      for i in range(len(CTX['sdl']))))
        VALS.append(dict(nf=CTX['nf'], t_s=time.time() - CTX['t0'], value=float(value),
                         per_axis=np.mean(pc, 0).tolist(), stable=stable))
        print('[val] nf %s  agg %.4e m  per-axis %s m  stable %s  (%.0f s)'
              % (CTX['nf'], value, np.array2string(np.mean(pc, 0), precision=4), stable,
                 time.time() - CTX['t0']), flush=True)


def main():
    t0 = time.time()
    stages = [int(s) for s in args.stages.split(',')]
    cfg = cfg_for(args.version, args.seed, local=not args.server, lr=args.lr,
                  n_its=args.its_per_stage)
    data, norm = load(cfg)
    fit_sys = J.build(cfg, data, norm, args.seed, nx=args.nx)
    sim = fit_sys.simulator
    sdl = data.val_ckpt_data.sdl
    rows = [r for (_, _, r) in sim.val_records]
    k0 = max(fit_sys.na, fit_sys.nb)
    ymax = [float(np.max(np.abs(np.asarray(sd.y)[k0:]))) for sd in sdl]
    mean_rms = np.mean([np.sqrt(np.mean((np.asarray(sd.y)[k0:] - np.ravel(norm.y0)) ** 2, 0))
                        for sd in sdl], 0)
    print('[run] mean-model per-axis RMS %s m (BB2-014 reference)'
          % np.array2string(mean_rms, precision=4), flush=True)
    CTX.update(sdl=sdl, rows=rows, bank=sim.bank, ymax=ymax, t0=t0)
    fit_sys.validation_probes = (Probe(),)

    STAGES = []
    for nf in stages:
        CTX['nf'] = nf
        n_tr0 = len(fit_sys.Loss_train)
        ts = time.time()
        err = None
        try:
            train_model(fit_sys, cfg.hp, cfg, data, nf=nf)
        except Exception as e:                                         # a failure is a result
            err = '%s: %s' % (type(e).__name__, e)
            print('[stage] nf %d raised %s' % (nf, err), flush=True)
        lt = np.asarray(fit_sys.Loss_train, float)[n_tr0:].tolist()
        STAGES.append(dict(nf=nf, loss_train=lt, error=err, wall_s=time.time() - ts))
        print('[stage] nf %d done: Loss_train %s (%.0f s)' % (nf, np.array2string(np.asarray(lt),
              precision=4), time.time() - ts), flush=True)

    last = [v for v in VALS if v['nf'] == stages[-1]]
    fin_st = [st for st in STAGES if st['nf'] == stages[-1]]
    fin = dict(error=next((st['error'] for st in fin_st if st['error']), None),
               loss_train=[x for st in fin_st for x in st['loss_train']])
    lt_fin = [x for x in fin['loss_train'] if np.isfinite(x)]
    ok_last = [v for v in last if v['stable']]
    final = min(ok_last, key=lambda v: v['value']) if ok_last else None   # fit keeps its best (BB2-015)
    ratio = (np.asarray(final['per_axis']) / mean_rms).tolist() if final else None
    crit = dict(
        # fit() writes NaN as the train loss of its initial validation; a NaN-loss break leaves a
        # call without any finite entry, which is what this checks
        a_final_stage_ok=bool(fin['error'] is None
                              and all(any(np.isfinite(x) for x in st['loss_train']) for st in fin_st)
                              and any(v['stable'] for v in last)),
        b_each_axis_below_third_of_mean_model=bool(final is not None and final['stable']
                                                   and max(ratio) <= 1 / 3),
        c_final_stage_loss_decreases=bool(len(lt_fin) >= 2 and lt_fin[-1] < lt_fin[0]))
    crit['PASS'] = all(crit.values())
    res = dict(args=vars(args), stride=cfg.stride, batch=cfg.batch_size, lr=cfg.lr,
               mean_model_rms=mean_rms.tolist(), vals=VALS, stages=STAGES,
               ratio_final_over_mean_model=ratio, criteria=crit, wall_s=time.time() - t0,
               n_params=sum(q.numel() for m in (fit_sys.encoder, fit_sys.hfn)
                            for q in m.parameters()))
    with open(os.path.join(OUT, 'result.json'), 'w') as fh:
        json.dump(res, fh, indent=1)
    print('[run] criteria %s  final/mean-model per axis %s  (%.0f s)'
          % (json.dumps(crit), ratio, res['wall_s']), flush=True)


if __name__ == '__main__':
    main()
