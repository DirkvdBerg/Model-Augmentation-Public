"""Is the error zero-mean, on which states, and does the CoG fix change that?

`diag_window_target.py` reported bias and a HAC t only for the CoG-ON arms, so
two questions it raised were left unanswered:

  Q1  on the 12 s free run, is only Y zero-mean, or are the others too?
  Q2  is the X / dX free-run bias caused by the centre-of-gravity correction, or
      was it there before it?

Both are settled by running the identical statistic on both baselines. Three
seeding arms x two baselines x six states, on two records.

WHY THE DISTINCTION MATTERS. Zero-mean across windows is what kills a mean
penalty (coulomb-offset F4, `zeromean_pin.py`): a penalty on a quantity whose
mean is already zero can only add variance. The per-window scatter is a separate
quantity and it is the one that corrupts the target. A state can be perfectly
zero-mean and still have a scatter three decades above the floor, and on this
rig most of them do.

Run:
  PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 conda run --no-capture-output \\
      -n GraduationProject python -u \\
      scripts/gantry/true-init-augmentation/diag_zeromean_cog.py
"""
__project_origin__ = "added"

import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))

from plant_cog import make_block, rollout_batch                 # noqa: E402
from data_exact import exact_truth                              # noqa: E402
from diag_window_target import newey_west_t, NF, STRIDE, K0, STATES   # noqa: E402

OUT = os.path.join(REPO, 'simulations', 'gantry_subnet', 'diagnostics')
RECORDS = ('V1_standstill_Yp10', 'V3_ysweep_Yp10')


def arms(name, cog):
    tr = exact_truth(name)
    rec, x6 = tr['rec'], tr['x6']
    N = len(rec['u'])
    starts = np.arange(K0, N - NF + 1, STRIDE)
    u_win = np.stack([rec['u'][s:s + NF] for s in starts])
    ref = np.stack([x6[s:s + NF] for s in starts])
    blk = make_block(Y_op=None, cog=cog, ts=rec['ts'], up_sample=1,
                     dtype=torch.float64)
    out = {}
    for tag, x0 in (('record', rec['x_logical'][starts]), ('exact', x6[starts])):
        out[tag] = (rollout_batch(blk, x0, u_win, n_out=6) - ref).mean(axis=1)
    free = rollout_batch(blk, np.array([[0., 0., rec['Y_op'], 0., 0., 0.]]),
                         rec['u'][None, :N], n_out=6)[0] - x6
    out['freerun'] = np.stack([free[s:s + NF].mean(axis=0) for s in starts])
    return out, len(starts)


def main():
    print('Zero-mean check, both baselines, all six states\n')
    print('  per-window means over 0.100 s windows; t is Newey-West HAC, lag n^(1/3)')
    print('  |t| < 2 is consistent with zero mean at the usual 5 % level\n')
    res = {}
    for name in RECORDS:
        res[name] = {}
        for cog in (True, False):
            A, nw = arms(name, cog)
            tag = 'CoG ON' if cog else 'CoG off'
            print(f'=== {name}  {tag}  ({nw} windows) ===')
            print(f'  {"state":<8}' + ''.join(f'{h:>16}' for h in
                                              ('record t', 'exact t', 'free t',
                                               'free bias', 'free scatter')))
            row = {}
            for c in range(6):
                ts_ = [newey_west_t(A[k][:, c]) for k in ('record', 'exact', 'freerun')]
                fb = float(A['freerun'][:, c].mean())
                fs = float(A['freerun'][:, c].std())
                row[STATES[c]] = dict(t_record=ts_[0], t_exact=ts_[1], t_free=ts_[2],
                                      bias_free=fb, scatter_free=fs,
                                      bias_exact=float(A['exact'][:, c].mean()),
                                      scatter_exact=float(A['exact'][:, c].std()))
                print(f'  {STATES[c]:<8}{ts_[0]:>16.2f}{ts_[1]:>16.2f}{ts_[2]:>16.2f}'
                      f'{fb:>16.4e}{fs:>16.4e}')
            res[name][tag] = row
            print()

    print('=== READING ===')
    for name in RECORDS:
        on, off = res[name]['CoG ON'], res[name]['CoG off']
        print(f'  {name}')
        for c in range(6):
            s = STATES[c]
            print(f'    {s:<8} free t  CoG on {on[s]["t_free"]:>8.2f}   '
                  f'CoG off {off[s]["t_free"]:>8.2f}      '
                  f'exact-seed t  on {on[s]["t_exact"]:>6.2f}  off {off[s]["t_exact"]:>6.2f}')
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, 'true_init_zeromean_cog.json')
    with open(p, 'w') as f:
        json.dump(res, f, indent=2)
    print(f'\n  wrote {p}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
