"""The 12 s free-run arm: does the trained ANN help over a whole record?

The training arm scores a `nf = 400` free run from the exact IC, because that is
the object it optimises. This scores the OTHER horizon: one continuous free run
from the exact rest IC over the whole 12 s record, ANN off against ANN on. The
project's 120x train/select horizon gap (D-129, D-130) lives exactly between
these two numbers, and reporting only the short one has been a documented way to
miss a degrading model.

The initial condition is the exact rest state, not an encoder estimate, so this
is a free run of the augmented model with no initialisation error at all. Any
difference from ANN-off is the ANN.

Run:
  ... python -u scripts/gantry/true-init-augmentation/eval_freerun.py \\
        --ckpt simulations/gantry_subnet/true_init_augmentation/ann_<tag>_best.pt
"""
__project_origin__ = "added"

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts', 'gantry'))

from gantry_dynamic.data import (                                   # noqa: E402
    load_datasets, compute_normalization, VAL_FILES)
from true_init_train import CFG, build_interconnect                 # noqa: E402
from data_exact import exact_truth                                  # noqa: E402

OUT = os.path.join(REPO, 'simulations', 'gantry_subnet', 'diagnostics')
NF_WIN = 400


@torch.no_grad()
def freerun(ic, rec, x6, norm, dtype, nmax=None):
    """One continuous rollout from the exact rest IC. Returns y error in metres."""
    N = len(rec['u']) if nmax is None else min(nmax, len(rec['u']))
    u = torch.as_tensor((rec['u'][:N] - norm.u_mean.flatten()) / norm.std_u.flatten(),
                        dtype=dtype)
    y = (rec['y'][:N] - norm.y0) / norm.ystd
    x = torch.zeros(1, CFG.nx_phys + CFG.nx_ann, dtype=dtype)
    x0 = np.zeros(6)
    x0[2] = rec['Y_op']                       # the Simulink integrators start at rest
    x[0, :6] = torch.as_tensor((x0 - norm.x_mean.flatten()) / norm.std_x.flatten(),
                               dtype=dtype)
    err = np.empty((N, 3))
    for k in range(N):
        yhat, x = ic(x, u[k:k + 1])
        err[k] = (yhat[0].numpy() - y[k]) * norm.ystd
    return err


def summarise(err, ts):
    n_win = len(err) // NF_WIN
    wm = err[:n_win * NF_WIN].reshape(n_win, NF_WIN, 3).mean(axis=1)
    return dict(rms=float(np.sqrt((err ** 2).mean())),
                rms_ch=np.sqrt((err ** 2).mean(axis=0)).tolist(),
                final_ch=err[-1].tolist(),
                settled_ch=err[-int(0.5 / ts):].mean(axis=0).tolist(),
                win_dc_scatter=wm.std(axis=0).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--no-cog', action='store_true', dest='no_cog')
    a = ap.parse_args()

    np.random.seed(CFG.seed)
    torch.manual_seed(CFG.seed)
    data = load_datasets(CFG)
    norm = compute_normalization(CFG, data)
    dtype = torch.float32

    np.random.seed(CFG.seed)
    torch.manual_seed(CFG.seed)
    ic, ann, _ = build_interconnect(CFG, norm, cog=not a.no_cog, dtype=dtype)

    res = {}
    print(f'12 s free run from the exact rest IC, ANN off vs ANN on ({a.ckpt})\n')
    print(f'  {"record":<22}{"arm":<6}{"RMS [m]":>13}{"X1":>13}{"X2":>13}{"Y":>13}'
          f'{"win DC Y":>13}')
    init_state = {k: v.clone() for k, v in ann.state_dict().items()}
    state = torch.load(a.ckpt, map_location='cpu')
    for f in VAL_FILES:
        tr = exact_truth(f[:-4])
        rec, x6 = tr['rec'], tr['x6']
        row = {}
        for arm in ('off', 'on'):
            # The off arm restores the UNTRAINED weights, whose final layer is
            # exactly zero (`zero_init_feed_forward_nn`), so the ANN output is
            # exactly 0 and the off arm is the baseline by construction rather
            # than by approximation.
            ann.load_state_dict(state if arm == 'on' else init_state)
            e = freerun(ic, rec, x6, norm, dtype)
            row[arm] = summarise(e, rec['ts'])
            print(f'  {f[:-4]:<22}{arm:<6}{row[arm]["rms"]:>13.4e}'
                  + ''.join(f'{v:>13.4e}' for v in row[arm]['rms_ch'])
                  + f'{row[arm]["win_dc_scatter"][2]:>13.4e}')
        r = row['on']['rms'] / max(row['off']['rms'], 1e-300)
        print(f'  {"":<22}{"ratio":<6}{r:>13.4f}   '
              f'{"IMPROVES" if r < 0.99 else "no change" if r < 1.01 else "DEGRADES"}')
        res[f[:-4]] = row

    os.makedirs(OUT, exist_ok=True)
    tag = os.path.basename(a.ckpt).replace('.pt', '')
    p = os.path.join(OUT, f'true_init_freerun_{tag}.json')
    with open(p, 'w') as fh:
        json.dump(res, fh, indent=2, default=float)
    print(f'\n  wrote {p}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
