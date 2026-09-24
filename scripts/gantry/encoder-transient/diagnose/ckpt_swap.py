"""Which part of x0 causes the X-channel transient of a trained checkpoint? Row-swap test (no training).

Training records T1-T14, closed loop nf = 400 (as ckpt_train_transient.py). x0 is assembled per arm from
the checkpoint's trained encoder and the exact physical state (truth.py), row group by row group:
model state = [X, Th, Y, dX, dTh, dY, latent 0..7]. Scored per channel and in the OBJECTIVE's weighting
(each channel / ystd^2): window MS, MS after step 100, transient share.

    CKPT=<base> CKPT_CONFIG=<config.json> R2_TAG=<tag> python -u diagnose/ckpt_swap.py
"""
__project_origin__ = "added"

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
import et_paths                                                  # noqa: E402
sys.path.insert(0, os.path.join(et_paths.ET, 'encoder'))
sys.path.insert(0, os.path.join(et_paths.ET, 'runners'))
sys.path.insert(0, HERE)
import cl_common as cc                                           # noqa: E402
from g1_transient import setup, prod_encoder                     # noqa: E402
from r2_ckpt import ckpt_overrides                               # noqa: E402

NF, K = 400, 100
XTH = [0, 1, 3, 4]          # X, Theta and their velocities
YR = [2, 5]                 # Y, dY
POS, VEL = [0, 1, 2], [3, 4, 5]
# arm: (rows taken from the exact state, latent policy)
ARMS = {
    'ENC':            ([], 'enc'),
    'EXACT-PHYS':     (list(range(6)), 'enc'),
    'ENC,LAT0':       ([], 'zero'),
    'EXACT-PHYS,LAT0': (list(range(6)), 'zero'),
    'EXACT-XTH':      (XTH, 'enc'),
    'EXACT-Y':        (YR, 'enc'),
    'EXACT-POS':      (POS, 'enc'),
    'EXACT-VEL':      (VEL, 'enc'),
}


def main():
    t0 = time.time()
    cfg, norm, fit_sys, hfn, enc_prod, _, _, planted, sa, dims = setup(
        ckpt_overrides(os.environ.get('CKPT_CONFIG')))
    from gantry_dynamic.training import resolve_checkpoint
    from gantry_dynamic.data import TRAIN_FILES
    from gantry_dynamic.controller import build_controller_bank
    hfn_sd, enc_sd = resolve_checkpoint(os.environ['CKPT'])[:2]
    hfn.load_state_dict(hfn_sd if isinstance(hfn_sd, dict) else hfn_sd.state_dict())
    enc_prod.load_state_dict(enc_sd if isinstance(enc_sd, dict) else enc_sd.state_dict())
    hfn.to(cc.DT); enc_prod.to(cc.DT)
    ystd = np.asarray(norm.ystd, float).ravel()
    names = [f[:-4] for f in TRAIN_FILES]
    bank, rows, _ = build_controller_bank(names, cfg.ts_new, ystd=norm.ystd, std_u=norm.std_u, dtype=cc.DT)
    na = dims[0]
    acc = {a: dict(ms0=np.zeros(3), msK=np.zeros(3), n=0) for a in ARMS}
    for nm, row in zip(names, rows):
        w = cc.windows_cl(nm, cfg, norm, na, NF)
        xe = prod_encoder(enc_prod, w, norm)
        xt = cc.x0_norm(w['x8'][:, :6], norm)
        for a, (ex_rows, lat) in ARMS.items():
            x0 = xe.copy()
            if ex_rows:
                x0[:, ex_rows] = xt[:, ex_rows]
            if lat == 'zero':
                x0[:, 6:] = 0.0
            e = cc.rollout_cl(hfn, x0, w, bank, row, norm.ystd)        # (W, nf, 3) metres
            acc[a]['ms0'] += (e ** 2).sum(axis=(0, 1)) / NF
            acc[a]['msK'] += (e[:, K:] ** 2).sum(axis=(0, 1)) / (NF - K)
            acc[a]['n'] += len(e)
        print(f'  {nm} done [{time.time() - t0:.0f} s]')
    wobj = 1.0 / ystd ** 2
    res = {}
    ref = None
    print(f'\n  {"arm":<17}{"obj MS":>11}{"obj share":>11}{"X1 RMS":>11}{"X1 share":>10}{"X2 share":>10}'
          f'{"Y RMS":>11}{"Y share":>9}{"X1 RMS[100:]":>14}')
    for a in ARMS:
        ms0 = acc[a]['ms0'] / acc[a]['n']
        msK = acc[a]['msK'] / acc[a]['n']
        obj0, objK = float((ms0 * wobj).sum()), float((msK * wobj).sum())
        per = (ms0 - msK) / ms0
        res[a] = dict(obj_ms=obj0, obj_share=(obj0 - objK) / obj0, rms0=np.sqrt(ms0).tolist(),
                      rmsK=np.sqrt(msK).tolist(), share=per.tolist())
        ref = ref or obj0
        print(f'  {a:<17}{obj0 / ref:>11.3f}{(obj0 - objK) / obj0:>11.1%}{np.sqrt(ms0[0]):>11.2e}'
              f'{per[0]:>10.1%}{per[1]:>10.1%}{np.sqrt(ms0[2]):>11.2e}{per[2]:>9.1%}{np.sqrt(msK[0]):>14.2e}')
    print('  (obj MS relative to ENC, the checkpoint as trained)')
    od = os.path.join(et_paths.OUT, 'ckpt_modes')
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, f'{os.environ.get("R2_TAG", "ckpt")}_swap.json'), 'w') as f:
        json.dump(res, f, indent=1)
    print(f'  [{time.time() - t0:.0f} s]')


if __name__ == '__main__':
    main()
