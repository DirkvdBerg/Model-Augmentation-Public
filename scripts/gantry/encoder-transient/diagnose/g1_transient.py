"""G1 part 2: today's startup transient on the exact planted model, closed loop, nf = 400 (ET-003).

No gradients. Arms (V1-V4, non-overlapping windows):
  U|P0          untrained pipeline model (zero ANN), production encoder (all 14 rows)
  PL|P0|random  planted model, production W^b physical rows, first two rows of the pipeline W^a
  PL|P0|zero    planted model, production W^b physical rows, latent rows zero
  PL|exact      planted model seeded with the EXACT 8-state truth: the floor of the planted test
  U|exact       untrained model seeded with the exact physical state (context: R2 in closed loop)
Output: outputs/g1/g1_transient.json.
"""
__project_origin__ = "added"

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import et_paths                                                  # noqa: E402
import cl_common as cc                                           # noqa: E402
import enc_common as ec                                          # noqa: E402
import truth                                                     # noqa: E402

NF, K = 400, 100
OUT = os.path.join(et_paths.OUT, 'g1')


def setup(overrides=None):
    """Pipeline (float64 for the rollout), controller bank, planted model, absorber std."""
    from gantry_dynamic.controller import build_controller_bank
    from gantry_dynamic.data import TRAIN_FILES
    et_paths.check()
    cfg, data, norm, fit_sys, hp, dims = ec.build(overrides)
    fit_sys.eval()
    hfn = fit_sys.hfn.to(cc.DT)
    enc = fit_sys.encoder.to(cc.DT)
    bank, rows, yops = build_controller_bank(cc.VAL, cfg.ts_new, ystd=norm.ystd,
                                             std_u=norm.std_u, dtype=cc.DT)
    xa = np.concatenate([truth.exact_truth(f[:-4], cfg.mode)['x8'][:, [3, 7]] for f in TRAIN_FILES])
    sa = xa.std(axis=0)
    planted = cc.PlantedHfn(norm, sa, cfg.ts_new, cfg.up_sample)
    return cfg, norm, fit_sys, hfn, enc, bank, dict(zip(cc.VAL, rows)), planted, sa, dims


def prod_encoder(enc, w, norm):
    """The pipeline encoder module on the window, in its own normalised frame (W, 14)."""
    un = (w['uw'] - np.asarray(norm.u_mean).ravel()) / np.asarray(norm.std_u).ravel()
    yn = (w['yw'] - np.asarray(norm.y0).ravel()) / np.asarray(norm.ystd).ravel()
    with torch.no_grad():
        return enc(torch.as_tensor(np.ascontiguousarray(un), dtype=cc.DT).contiguous(),
                   torch.as_tensor(np.ascontiguousarray(yn), dtype=cc.DT).contiguous()).numpy()


def main():
    t0 = time.time()
    cfg, norm, fit_sys, hfn, enc, bank, rowmap, planted, sa, dims = setup()
    na = dims[0]
    chk = cc.selfcheck_planted(planted)
    print(f'  planted deriv vs truth.Plant8.deriv: worst rel diff {chk:.2e}')
    print(f'  nf={NF} K={K} na={na} up_sample={cfg.up_sample} ts={cfg.ts_new} rows {rowmap}')
    errs = {a: [] for a in ('U|P0', 'PL|P0|random', 'PL|P0|zero', 'PL|exact', 'U|exact')}
    for nm in cc.VAL:
        w = cc.windows_cl(nm, cfg, norm, na, NF)
        xe = prod_encoder(enc, w, norm)
        x_ex6 = cc.x0_norm(w['x8'][:, :6], norm)
        x_ex8 = ((w['x8'] - planted.xm.numpy()) / planted.sx.numpy())
        nx_u = xe.shape[1]
        xu_ex = np.zeros((len(xe), nx_u)); xu_ex[:, :6] = x_ex6
        arms = {
            'U|P0': (hfn, xe),
            'PL|P0|random': (planted, np.concatenate([xe[:, :6], xe[:, 6:8]], 1)),
            'PL|P0|zero': (planted, np.concatenate([xe[:, :6], np.zeros((len(xe), 2))], 1)),
            'PL|exact': (planted, x_ex8),
            'U|exact': (hfn, xu_ex),
        }
        for a, (m, x0) in arms.items():
            errs[a].append(cc.rollout_cl(m, x0, w, bank, rowmap[nm], norm.ystd))
        print(f'  {nm}: {len(w["k"])} windows  [{time.time() - t0:.0f} s]')

    res = {a: cc.metrics(errs[a], K) for a in errs}
    for lat in ('random', 'zero'):
        p = res[f'PL|P0|{lat}']
        res[f'disc|{lat}'] = dict(K0=res['U|P0']['rms0'] / p['rms0'], K100=res['U|P0']['rmsK'] / p['rmsK'])
    res['floor_over_planted_rms0'] = {lat: res['PL|exact']['rms0'] / res[f'PL|P0|{lat}']['rms0']
                                      for lat in ('random', 'zero')}
    res['selfcheck_planted'] = chk
    res['sa'] = sa.tolist()

    print(f'\n  {"arm":<15}{"RMS[0:400]":>12}{"RMS[100:]":>12}{"share":>8}{"grow":>8}   chan RMS[0:400] X1/X2/Y')
    for a in errs:
        r = res[a]
        print(f'  {a:<15}{r["rms0"]:>12.4e}{r["rmsK"]:>12.4e}{r["share"]:>8.1%}{r["grow"]:>8.2f}   '
              + ' '.join(f'{v:.3e}' for v in r['chan0']))
    for lat in ('random', 'zero'):
        d = res[f'disc|{lat}']
        print(f'  discrimination (W^a {lat}): K=0 {d["K0"]:.3f}x   K=100 {d["K100"]:.3f}x')
    print(f'  planted floor / planted K=0 RMS: {res["floor_over_planted_rms0"]}')

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'g1_transient.json'), 'w') as f:
        json.dump(res, f, indent=1)
    print(f'  saved outputs/g1/g1_transient.json  [{time.time() - t0:.0f} s]')


if __name__ == '__main__':
    main()
