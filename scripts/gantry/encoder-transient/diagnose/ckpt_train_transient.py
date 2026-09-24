"""Transient share of a trained checkpoint on the TRAINING records T1-T14 (no training).

Same measurement as runners/r2_ckpt.py (closed loop, residual form, xc = 0, nf = 400,
share = (MS[0:400] - MS[100:400]) / MS[0:400], grow = RMS(step 399)/RMS(step 0)), but on the records
the training loss is built from. Windows are non-overlapping from k = na + 1 (training uses stride 10;
the records and samples are the same, the overlap only changes the weighting).
x0 from the checkpoint's own trained encoder, and from the exact physical state for context.

    CKPT=<base> CKPT_CONFIG=<config.json> R2_TAG=<tag> python -u diagnose/ckpt_train_transient.py
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
    names = [f[:-4] for f in TRAIN_FILES]
    bank, rows, _ = build_controller_bank(names, cfg.ts_new, ystd=norm.ystd, std_u=norm.std_u, dtype=cc.DT)
    na = dims[0]
    errs = {'trained_encoder': [], 'exact_state': []}
    per = {}
    for nm, row in zip(names, rows):
        w = cc.windows_cl(nm, cfg, norm, na, NF)
        xe = prod_encoder(enc_prod, w, norm)
        xx = xe.copy(); xx[:, :6] = cc.x0_norm(w['x8'][:, :6], norm)
        e_t = cc.rollout_cl(hfn, xe, w, bank, row, norm.ystd)
        e_x = cc.rollout_cl(hfn, xx, w, bank, row, norm.ystd)
        errs['trained_encoder'].append(e_t); errs['exact_state'].append(e_x)
        m = cc.metrics([e_t], K)
        per[nm] = dict(n_win=len(w['k']), rms0=m['rms0'], share=m['share'], grow=m['grow'])
        print(f'  {nm:<22} {len(w["k"]):>4} win  trained encoder: RMS {m["rms0"]:.3e}  '
              f'share {m["share"]:6.1%}  grow {m["grow"]:.2f}   [{time.time() - t0:.0f} s]')
    res = {a: cc.metrics(errs[a], K) for a in errs}
    for a in errs:
        r = res[a]
        print(f'\n  POOLED T1-T14, x0 = {a:<16} RMS {r["rms0"]:.4e}  RMS[100:] {r["rmsK"]:.4e}  '
              f'share {r["share"]:.1%}  grow {r["grow"]:.2f}  ({r["n_win"]} windows)')
    od = os.path.join(et_paths.OUT, 'ckpt_modes')
    os.makedirs(od, exist_ok=True)
    tag = os.environ.get('R2_TAG', 'ckpt')
    with open(os.path.join(od, f'{tag}_train_transient.json'), 'w') as f:
        json.dump(dict(pooled=res, per_record=per), f, indent=1, default=float)
    print(f'  saved outputs/ckpt_modes/{tag}_train_transient.json  [{time.time() - t0:.0f} s]')


if __name__ == '__main__':
    main()
