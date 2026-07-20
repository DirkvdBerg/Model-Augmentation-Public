"""
d11_trainset_encoder_bias.py -- post-70558 falsification follow-up: what is the
encoder's dY init error on the TRAINING-set windows, at na=17 vs na=27?

Run 70558 (na_nb=27) falsified the mean-bias hypothesis: the V1 mean dY init bias
collapsed (pre-flight +9.4e-5, 0.4 SE) yet the drift returned unchanged. Two
readings remain; this measurement separates them:
  (a) TRANSFER failure: the TRAINING-set mean bias (14 trajectories, different
      excitation classes; never measured -- the pre-flight used V1 only) is still
      large at na=27 -> the mean-bias story survives with a distribution
      correction; kill the mean on the TRAIN distribution.
  (b) PER-WINDOW compensation: train-set mean ALSO ~ zero at na=27 -> the ANN's
      reward is compensating the per-window init error (std ~2.5e-3 m/s, 10x the
      mean, untouched by na=27); the DC is merely its average. Fix must make the
      ANN unable to act as an encoder-error compensator (Layer 2 / consistency
      term) -- design discussion before building.

Uses data.train_list EXACTLY as training does (same loader, same noise convention,
same normalization); ground truth from load_mat_aug per trajectory. Encoder called
directly on normalized windows (d10 pattern; encoder_init_state per-call transform
is too slow for ~1700 windows x 2 na).

Run:
  conda run -n GraduationProject python scripts/gantry/diagnostics-drift/d11_trainset_encoder_bias.py
Env: STRIDE (window start spacing, default 400 = non-overlapping nf windows).
Outputs -> simulations/gantry_subnet/diagnostics/ (npz; table to stdout)
"""
import os
import sys
import dataclasses

import numpy as np
import torch

REPO   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
GANTRY = os.path.join(REPO, 'scripts', 'gantry')
sys.path.insert(0, GANTRY)
sys.path.insert(0, os.path.dirname(__file__))

import drift_common as dc
from gantry_dynamic.data import load_datasets, compute_normalization, load_mat_aug, TRAIN_FILES
from gantry_dynamic.model import build_model, get_encoder_dims
from gantry_interconnect_dynamic import CFG as cfg_base

STRIDE = int(os.environ.get('STRIDE', '400'))
NA_VARIANTS = [None, 27]          # None -> default na=17; 27 -> the 70558 treatment

results = {}
for na_override in NA_VARIANTS:
    cfg = dataclasses.replace(cfg_base, na_nb_override=na_override)
    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    data = load_datasets(cfg)
    norm = compute_normalization(cfg, data)
    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    fit_sys = build_model(cfg.hp, cfg, data, norm)
    fit_sys.hfn.eval()
    na, nb, na_right, nb_right = get_encoder_dims(cfg.hp, cfg)
    warm = max(na, nb)
    std_x = norm.std_x.flatten(); xm = norm.x_mean.flatten()
    ystd = np.asarray(norm.y0 * 0 + norm.ystd).flatten() if hasattr(norm, 'ystd') else None
    y0v  = np.asarray(norm.y0).flatten()
    ystd = np.asarray(norm.ystd).flatten()
    u0v  = np.asarray(fit_sys.norm.u0).flatten()
    ustd = np.asarray(fit_sys.norm.ustd).flatten()

    per_traj = {}
    pooled = []
    for fi, fname in enumerate(TRAIN_FILES):
        tr = data.train_list[fi]                      # EXACTLY what training consumes
        u_n = (np.asarray(tr.u, dtype=np.float64) - u0v) / ustd
        y_n = (np.asarray(tr.y, dtype=np.float64) - y0v) / ystd
        _, _, x_log, _ = load_mat_aug(fname, cfg)     # noiseless ground truth
        N = len(u_n)
        starts = list(range(warm, N - 400, STRIDE))
        errs = np.empty((len(starts), 2))             # [Y err, dY err]
        for i, s in enumerate(starts):
            uw = np.ascontiguousarray(u_n[s - nb: s + nb_right], dtype=cfg.dtype_np)
            yw = np.ascontiguousarray(y_n[s - na: s + na_right], dtype=cfg.dtype_np)
            with torch.no_grad():
                x0 = fit_sys.encoder(torch.tensor(uw[None], dtype=cfg.dtype_pt),
                                     torch.tensor(yw[None], dtype=cfg.dtype_pt)).numpy()[0]
            x0 = x0[:6] * std_x + xm
            errs[i] = [x0[2] - x_log[s, 2], x0[5] - x_log[s, 5]]
        per_traj[fname] = errs
        pooled.append(errs)
    pooled = np.concatenate(pooled)
    results[na] = dict(per_traj=per_traj, pooled=pooled)

    nwin = len(pooled)
    m = pooled.mean(axis=0); se = pooled.std(axis=0, ddof=1) / np.sqrt(nwin)
    print(f'\n=== TRAIN set, na={na} (window {(na+1)*cfg.ts_new*1e3:.2f} ms, {nwin} windows, stride {STRIDE}) ===')
    print(f'  pooled Y  bias {m[0]:+.3e} m   ({m[0]/se[0]:+.2f} SE)')
    print(f'  pooled dY bias {m[1]:+.3e} m/s ({m[1]/se[1]:+.2f} SE)   per-window std {pooled[:,1].std():.3e}')
    print(f'  {"trajectory":28s} {"mean dY err":>12s} {"/SE":>7s} {"std":>10s}')
    for fname, errs in per_traj.items():
        md = errs[:, 1].mean(); sd = errs[:, 1].std(ddof=1) / np.sqrt(len(errs))
        print(f'  {fname:28s} {md:>+12.3e} {md/sd:>7.2f} {errs[:,1].std():>10.3e}')

# paired comparison across na on the pooled set (same starts per trajectory)
na_a, na_b = sorted(results)
pa, pb = results[na_a]['pooled'], results[na_b]['pooled']
n = min(len(pa), len(pb))
d = pa[:n, 1] - pb[:n, 1]
print(f'\npaired dY improvement na={na_a} -> na={na_b}: {d.mean():+.3e} m/s '
      f'({d.mean()/(d.std(ddof=1)/np.sqrt(n)):+.2f} SE, n={n})')

stem = os.path.join(dc.OUT_DIR, 'd11_trainset_encoder_bias')
np.savez(stem + '.npz',
         **{f'pooled_na{k}': v['pooled'] for k, v in results.items()},
         stride=STRIDE,
         **{f'traj_{k}_{fn.split(".")[0]}': v['per_traj'][fn]
            for k, v in results.items() for fn in v['per_traj']})
print(f'\nSaved: {stem}.npz')
