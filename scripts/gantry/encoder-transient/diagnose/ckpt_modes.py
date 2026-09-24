"""Did the trained model learn the absorber? Poles and error spectrum of a checkpoint (no training).

1. Poles: the trained model's own step linearised at rest at several Y (encoder/model_map.linearise),
   discrete poles -> s = log(z)/Ts -> frequency and damping, next to the TRUE plant's continuous
   poles (recon.planted_ct: baseline + the dataset's absorber). The eigenvector condition number is
   reported with every set (D-168: a bare eigenvalue is not evidence).
2. Error spectrum: the closed-loop nf = 400 windows of V1-V4 (as R031), window error per channel,
   Hann-windowed periodogram averaged over windows, power per band, for the trained model (its own
   encoder), the untrained model (production encoder) and the correct model (exact state, the floor).

    CKPT=<base> CKPT_CONFIG=<config.json> python -u diagnose/ckpt_modes.py
Output: outputs/ckpt_modes/<tag>.json
"""
__project_origin__ = "added"

import copy
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
import et_paths                                                  # noqa: E402
sys.path.insert(0, os.path.join(et_paths.ET, 'encoder'))
sys.path.insert(0, os.path.join(et_paths.ET, 'runners'))
sys.path.insert(0, HERE)
import cl_common as cc                                           # noqa: E402
import model_map as mm                                           # noqa: E402
import recon                                                     # noqa: E402
from g1_transient import setup, prod_encoder                     # noqa: E402
from r2_ckpt import ckpt_overrides                               # noqa: E402

Y_LIST = [-0.30, -0.15, 0.0, 0.10, 0.15, 0.30]
BANDS = [(0, 20), (20, 100), (100, 140), (140, 230), (230, 500), (500, 2000)]
CH = ['X1', 'X2', 'Y']


def modes_discrete(A, ts):
    """Discrete poles -> (f [Hz], zeta, |z|) for the oscillatory and real poles away from z = 0."""
    z, V = np.linalg.eig(A)
    keep = np.abs(z) > 1e-6
    s = np.log(z[keep].astype(complex)) / ts
    f = np.abs(s.imag) / (2 * np.pi)
    zeta = -s.real / np.maximum(np.abs(s), 1e-30)
    rows = sorted({(round(float(fi), 2), round(float(zi), 4), round(float(ai), 6))
                   for fi, zi, ai in zip(f, zeta, np.abs(z[keep]))})
    return rows, float(np.linalg.cond(V))


def modes_ct(A):
    s, V = np.linalg.eig(A)
    f = np.abs(s.imag) / (2 * np.pi)
    zeta = -s.real / np.maximum(np.abs(s), 1e-30)
    return sorted({(round(float(fi), 2), round(float(zi), 4)) for fi, zi in zip(f, zeta)}), float(np.linalg.cond(V))


def band_power(errs, ts):
    """errs: list of (W, nf, 3) [m]. Returns per band the mean-square error per channel [m^2]
    (Parseval on the Hann-windowed window, normalised so the bands sum to the windowed MS)."""
    e = np.concatenate(errs, axis=0)
    nf = e.shape[1]
    w = np.hanning(nf)
    E = np.fft.rfft(e * w[None, :, None], axis=1)
    P = (np.abs(E) ** 2).mean(axis=0)                          # (nfreq, 3)
    P[1:-1] *= 2                                               # one-sided
    P /= nf * (w ** 2).sum()
    f = np.fft.rfftfreq(nf, ts)
    return {f'{a}-{b}': P[(f >= a) & (f < b)].sum(axis=0) for a, b in BANDS}


def main():
    t0 = time.time()
    cfg, norm, fit_sys, hfn, enc_prod, bank, rowmap, planted, sa, dims = setup(
        ckpt_overrides(os.environ.get('CKPT_CONFIG')))
    na, ts = dims[0], cfg.ts_new
    hfn_u, enc_u = copy.deepcopy(hfn), copy.deepcopy(enc_prod)   # untrained, before loading
    from gantry_dynamic.training import resolve_checkpoint
    hfn_sd, enc_sd = resolve_checkpoint(os.environ['CKPT'])[:2]
    hfn.load_state_dict(hfn_sd if isinstance(hfn_sd, dict) else hfn_sd.state_dict())
    enc_prod.load_state_dict(enc_sd if isinstance(enc_sd, dict) else enc_sd.state_dict())
    hfn.to(cc.DT); enc_prod.to(cc.DT)
    nx = cfg.nx_phys + cfg.nx_ann
    tag = os.environ.get('R2_TAG', 'ckpt')
    res = dict(ckpt=os.environ['CKPT'], poles={}, spectrum={})

    # ==== 1. poles ====
    print(f'  poles (f [Hz], zeta, |z|), model step at rest, Ts {ts}')
    for Y in Y_LIST:
        A, _, _, off = mm.linearise(hfn, Y, norm, nx)
        tr, ctr = modes_discrete(A, ts)
        Au, _, _, _ = mm.linearise(hfn_u, Y, norm, nx)
        un, cun = modes_discrete(Au, ts)
        At, _, _, _ = recon.planted_ct(Y)
        tt, ctt = modes_ct(At)
        res['poles'][str(Y)] = dict(trained=tr, trained_cond=ctr, untrained=un, untrained_cond=cun,
                                    true_ct=tt, true_cond=ctt, affine_offset=float(np.abs(off).max()))
        osc = lambda rows: [(f, z) for f, z, *_ in rows if f > 5]          # noqa: E731
        print(f'\n  Y = {Y:+.2f}  (affine offset at rest {np.abs(off).max():.2e})')
        print(f'    TRUE plant (ct)   cond {ctt:.1e}: ' + '  '.join(f'{f:.1f}Hz/z{z:.3f}' for f, z in osc(tt)))
        print(f'    untrained model   cond {cun:.1e}: ' + '  '.join(f'{f:.1f}Hz/z{z:.3f}' for f, z in osc(un)))
        print(f'    TRAINED model     cond {ctr:.1e}: ' + '  '.join(f'{f:.1f}Hz/z{z:.3f}' for f, z in osc(tr)))
        slow = [(f, z, a) for f, z, a in tr if f <= 5]
        print(f'    trained non-oscillatory / slow poles (f<=5 Hz): '
              + '  '.join(f'|z|{a:.5f}' for f, z, a in slow))

    # ==== 2. error spectrum on the closed-loop windows ====
    errs = {k: [] for k in ('trained', 'untrained', 'correct_floor')}
    for nm in cc.VAL:
        w = cc.windows_cl(nm, cfg, norm, na, 400)
        errs['trained'].append(cc.rollout_cl(hfn, prod_encoder(enc_prod, w, norm), w, bank, rowmap[nm], norm.ystd))
        errs['untrained'].append(cc.rollout_cl(hfn_u, prod_encoder(enc_u, w, norm), w, bank, rowmap[nm], norm.ystd))
        x8 = (w['x8'] - planted.xm.numpy()) / planted.sx.numpy()
        errs['correct_floor'].append(cc.rollout_cl(planted, x8, w, bank, rowmap[nm], norm.ystd))
        print(f'  {nm} windows done [{time.time() - t0:.0f} s]')
    bp = {k: band_power(v, ts) for k, v in errs.items()}
    res['spectrum'] = {k: {b: v.tolist() for b, v in d.items()} for k, d in bp.items()}
    print('\n  window-error RMS per band [m], channels X1 / X2 / Y')
    print(f'    {"band [Hz]":<11}' + ''.join(f'{k:>36}' for k in errs))
    for b in bp['trained']:
        print(f'    {b:<11}' + ''.join(
            f'{"  ".join(f"{np.sqrt(x):.2e}" for x in bp[k][b]):>36}' for k in errs))
    tot = {k: sum(bp[k].values()) for k in bp}
    print('\n  share of the trained model\'s window MS per band (X1 / X2 / Y), and trained / untrained RMS')
    for b in bp['trained']:
        sh = bp['trained'][b] / tot['trained']
        ratio = np.sqrt(bp['trained'][b] / np.maximum(bp['untrained'][b], 1e-40))
        print(f'    {b:<11} share ' + ' '.join(f'{x:6.1%}' for x in sh)
              + '   trained/untrained ' + ' '.join(f'{x:.3f}' for x in ratio))
    od = os.path.join(et_paths.OUT, 'ckpt_modes')
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, f'{tag}.json'), 'w') as f:
        json.dump(res, f, indent=1, default=float)
    print(f'  saved outputs/ckpt_modes/{tag}.json  [{time.time() - t0:.0f} s]')


if __name__ == '__main__':
    main()
