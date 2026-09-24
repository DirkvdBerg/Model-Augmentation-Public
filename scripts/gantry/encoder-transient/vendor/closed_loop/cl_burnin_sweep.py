"""Can the objective be fixed AT nf = 400 by not scoring the startup transient?

THE FINDING THIS TESTS
----------------------
N1 (`cl_nf_sweep.py`, D-148 finding 8) showed the loss ranks a correct augmentation only 1.25x
better than the one training finds at nf = 400, rising to 2.54x at nf = 3200. The excess
mean-square over each model's own settled level scales as 1/n, which is the signature of a FIXED
startup transient being diluted rather than of an error that persists. Decomposed at nf = 400:

    planted (correct)  window MS 1.456e-12, settled 1.745e-13  ->  88 % of the loss is TRANSIENT
    trained            window MS 2.272e-12, settled 1.942e-12  ->  15 %

So the objective grades a good model almost entirely on its initial state and a bad one mostly on
its dynamics, and it charges that 1666 times per epoch. The transient is an INITIALISATION error,
not a model error: the planted model, which was handed the physics, has 3.9x MORE startup energy
than the trained one, because it USES its latent states and the encoder cannot initialise them.

If that reading is right, the fix is to stop scoring the transient rather than to dilute it with a
longer window, which would give the horizon benefit at nf = 400 with no change to gradient depth,
no segment-boundary semantics, and no rate change.

THE OBJECTIVE UNDER TEST
------------------------
    V = w_burn * ||e over [0, K)||^2  +  ||e over [K, nf)||^2

w_burn = 0 is hard exclusion. w_burn > 0 keeps an explicit initialisation criterion so the encoder
still has a reason to produce a good x0, which matters because the encoder is part of the model,
not a nuisance parameter: `x_a` is only reachable through it. The two estimation problems, initial
state and dynamics, then carry separate and visible weights instead of the accidental 88/12 ratio.

W^a IS THE THIRD FACTOR, because it overlaps with burn-in. Its measured benefit (window RMS
1.207e-06 -> 7.616e-07, D-148 finding 6) comes entirely from reducing the startup transient, and
burn-in stops scoring that term, so most of it should evaporate. Testing them jointly is the only
way to see whether W^a still earns its place under the new objective.

WHAT THIS MEASURES, AND WHAT IT DOES NOT
----------------------------------------
It measures DISCRIMINATION: the ratio between the trained model's criterion and the planted one's,
on the same windows, same init policy, same reduction. A loss that cannot rank the answer cannot
descend toward it, so this is the right proxy. It is still a proxy: the confirmation is a training
run. No gradients are taken here.

Cost: four rollouts (2 models x 2 W^a settings). K and w_burn are re-reductions of the same stored
per-sample error, so the grid is free.

Usage: python -u cl_burnin_sweep.py
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

import deepSI                                                              # noqa: E402
import demo_common as dm                                                   # noqa: E402
from demo_common import CFG                                                # noqa: E402
from gantry_dynamic.data import load_traj, VAL_FILES                       # noqa: E402
from cl_pipeline import build_closed_loop                                  # noqa: E402
from model_augmentation.fit_systems.blocks import Static_ANN_Block         # noqa: E402
from model_augmentation.fit_systems.closed_loop import (                   # noqa: E402
    closed_loop_rollout, window_starts, make_window_tensors)
from deepSI.fit_systems.fit_system import get_work_dirs                    # noqa: E402

NF = int(os.environ.get('CL_BI_NF', 400))
K_LIST = [int(v) for v in os.environ.get('CL_BI_K', '0,50,100,200').split(',')]
W_LIST = [float(v) for v in os.environ.get('CL_BI_W', '0,0.1').split(',')]
PLANTED = os.path.join(HERE, 'runs', 'cl_capability_planted_ann.pt')
TRAINED = os.environ.get('CL_BI_CKPT', '')
OUT = os.path.join(HERE, 'runs', 'cl_burnin_sweep.json')
t0 = time.time()


def rollout_errors(fs, bank, vl_files, vl, rows, nf):
    """Per-sample closed-loop error for every window of every record. (list of (W, nf, ny) [m])."""
    rv = lambda a: np.asarray(a).ravel()                                    # noqa: E731
    out = []
    for fname, sd, ctrl_row in zip(vl_files, vl, rows):
        st = window_starts(len(sd.u), fs.na, fs.nb, nf,
                           getattr(fs, 'na_right', 0), getattr(fs, 'nb_right', 0), nf)
        uh, yh, uf, yf = make_window_tensors(fs, sd, nf, st)
        with torch.no_grad():
            x0 = fs.encoder(uh, yh)
            y_pred, _, _ = closed_loop_rollout(
                fs.hfn, fs.hfn.output_only, uf, yf, x0, bank,
                torch.full((len(st),), int(ctrl_row), dtype=torch.long))
            out.append((y_pred - yf).numpy() * rv(fs.norm.ystd))
    return out


def criterion(errs, K, w_burn):
    """sqrt of the WEIGHTED mean square, so the number stays in metres and comparable to the
    unweighted window RMS at K = 0. The weights are applied per sample, then normalised by the
    total weight, which is what makes w_burn = 0 exactly 'score only [K, nf)'."""
    sse, wsum = 0.0, 0.0
    for e in errs:
        nf = e.shape[1]
        w = np.ones(nf)
        w[:K] = w_burn
        sse += float(np.sum((e ** 2) * w[None, :, None]))
        wsum += float(w.sum()) * e.shape[0] * e.shape[2]
    return float(np.sqrt(sse / wsum))


def main():
    cfg = dataclasses.replace(CFG, seed=0, ann_route_ix=tuple(range(8)), lr=1e-7)
    fs, norm, K0, na, nb, na_r, nb_r = dm.build_pipeline(cfg=cfg, verbose=False)
    vl_files = list(VAL_FILES)
    vl = [load_traj(f, cfg) for f in vl_files]
    fs.simulator = build_closed_loop(fs, norm, cfg, train_files=[], val_files=vl_files,
                                     val_data=deepSI.System_data_list(vl), verbose=False)
    bank = fs.simulator.bank
    rows = [r for _, _, r in fs.simulator.val_records]
    ck_path = TRAINED or sorted(glob.glob(os.path.join(get_work_dirs()['checkpoints'],
                                                       '*XadbYQ_best.pth')))[0]
    ck = torch.load(ck_path, weights_only=False)
    ann_planted = torch.load(PLANTED, weights_only=False)
    hfn_trained, enc_trained = ck['hfn'].state_dict(), ck['encoder'].state_dict()
    hfn_init = {k: v.detach().clone() for k, v in fs.hfn.state_dict().items()}
    enc_init = {k: v.detach().clone() for k, v in fs.encoder.state_dict().items()}

    print('=' * 100)
    print('BURN-IN SWEEP at nf = %d: does not scoring the startup transient fix the objective?' % NF)
    print('=' * 100)
    print('trained checkpoint: %s' % os.path.basename(ck_path))

    errs = {}
    for model in ('planted', 'trained'):
        for wa in ('random', 'zero'):
            fs.hfn.load_state_dict(hfn_init)
            fs.encoder.load_state_dict(enc_init)
            if model == 'trained':
                fs.hfn.load_state_dict(hfn_trained)
                fs.encoder.load_state_dict(enc_trained)
            else:
                ann = next(m for m in fs.hfn.connected_blocks
                           if isinstance(m, Static_ANN_Block))
                ann.load_state_dict(ann_planted)
            if wa == 'zero':
                with torch.no_grad():
                    fs.encoder.Wa_psi_y.zero_()
                    fs.encoder.Wa_psi_u.zero_()
            errs[(model, wa)] = rollout_errors(fs, bank, vl_files, vl, rows, NF)
            print('  rolled out %-8s W^a %-7s [%.0fs]' % (model, wa, time.time() - t0), flush=True)

    res = {}
    print('\n%-9s %-7s %-8s %-13s %-13s %s'
          % ('W^a', 'K', 'w_burn', 'planted [m]', 'trained [m]', 'DISCRIMINATION'))
    for wa in ('random', 'zero'):
        for K in K_LIST:
            for w in W_LIST:
                if K == 0 and w != W_LIST[0]:
                    continue                      # w_burn is meaningless with no burn-in region
                p = criterion(errs[('planted', wa)], K, w)
                t = criterion(errs[('trained', wa)], K, w)
                res['%s_K%d_w%g' % (wa, K, w)] = dict(planted=p, trained=t, ratio=t / p)
                mark = ''
                if wa == 'random' and K == 0:
                    mark = '<- the objective as it is today'
                print('%-9s %-7d %-8g %-13.4e %-13.4e %.3fx  %s' % (wa, K, w, p, t, t / p, mark))
    base = res['random_K0_w%g' % W_LIST[0]]['ratio']
    best_k = max(res, key=lambda k: res[k]['ratio'])
    print('\ntoday %.3fx  ->  best %.3fx at %s' % (base, res[best_k]['ratio'], best_k))
    print('reference: nf = 3200 (0.8 s) gives 2.54x, the free run gives 3.34x')
    print('\nverdict: %s' % (
        'burn-in recovers most of what a 8x longer window buys, AT nf = 400. The fix is a change '
        'of reduction, not of horizon: no segment-boundary decision, no rate change, no extra '
        'gradient depth.' if res[best_k]['ratio'] > 2.0 else
        'burn-in helps but does not reach the 2.54x that nf = 3200 buys, so the transient is only '
        'part of it and effective length is still needed.' if res[best_k]['ratio'] > 1.5 else
        'burn-in does NOT recover the discrimination, so the 1/n reading is wrong and horizon '
        'work (multiple shooting) is back to being the primary route.'))
    json.dump(res, open(OUT, 'w'), indent=2)
    print('\nwrote %s   [%.0fs]' % (OUT, time.time() - t0))


if __name__ == '__main__':
    main()
