"""Critical look at step 3: why did G7 fail, and can the closed-loop metric see model quality?

D1  G7 root cause.  The two paths must be bit-identical when the ANN output is zero. Locate the
    difference instead of guessing: check the ANN output, the encoder parameters, x0 per record,
    then the trajectories sample by sample.

D2  IS THE METRIC BLIND?  Every val record and both initialisations returned 2.18e-06 +- 1 %.
    A metric that cannot distinguish four records or two initialisations may not be able to
    distinguish a good model from a bad one either. Perturb the model deliberately and see whether
    the closed-loop score moves. If a large perturbation leaves the score at 2.18e-06, closed-loop
    sim-RMS is blind and cannot be used for selection at 4 kHz, which reopens D-141.

Usage: python -u cl_diag_step3.py
"""
__project_origin__ = "added"

import dataclasses
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
from model_augmentation.fit_systems.blocks import Static_ANN_Block        # noqa: E402

import cl_plant as PLANT                                                  # noqa: E402
import cl_validation as CV                                                # noqa: E402
from cl_controller import ControllerBank                                  # noqa: E402

t0 = time.time()
cfg = dataclasses.replace(CFG, seed=0)
print('building ...', flush=True)
fs, norm, K0, na, nb, na_r, nb_r = dm.build_pipeline(cfg=cfg, verbose=False)
nx = cfg.nx_phys + cfg.nx_ann
C_out, b_out = PLANT.identify_output_map(fs.hfn, nx, cfg.nu, dtype=cfg.dtype_pt)
step_fn, out_fn = PLANT.make_fns(fs, C_out, b_out)
names = [f[:-4] for f in VAL_FILES]
bank = ControllerBank(names, cfg.ts_new, dtype=cfg.dtype_pt, ystd=norm.ystd, std_u=norm.std_u)
val = [load_traj(f, cfg) for f in VAL_FILES]
print('built [%.0fs]\n' % (time.time() - t0), flush=True)


def prep(sd):
    un = ((sd.u - fs.norm.u0) / fs.norm.ustd).astype(cfg.dtype_np)
    yn = ((sd.y - fs.norm.y0) / fs.norm.ystd).astype(cfg.dtype_np)
    return un, yn


print('=' * 88)
print('D1  G7 ROOT CAUSE')
print('=' * 88)

ann = next(m for m in fs.hfn.connected_blocks if isinstance(m, Static_ANN_Block))
g = torch.Generator().manual_seed(0)
z = torch.randn(64, nx + cfg.nu, 1, generator=g, dtype=cfg.dtype_pt)
with torch.no_grad():
    w = ann(z)
print('ANN output on random inputs: max abs = %.3e  (exactly 0 => paths must agree)'
      % float(w.abs().max()))

enc_frozen = CV.snapshot_encoder(fs)
pdiff = max(float((p - q).abs().max())
            for p, q in zip(fs.encoder.parameters(), enc_frozen.parameters()))
print('encoder params live vs frozen: max abs diff = %.3e' % pdiff)

for i, (name, sd) in enumerate(zip(names, val)):
    un, yn = prep(sd)
    x0_live = CV.encoder_x0(fs.encoder, un, yn, K0, na, nb, na_r, nb_r, cfg.dtype_pt)
    x0_froz = CV.encoder_x0(enc_frozen, un, yn, K0, na, nb, na_r, nb_r, cfg.dtype_pt)
    dx0 = float((x0_live - x0_froz).abs().max())
    ctrl = bank.gather(torch.tensor([i], dtype=torch.long))

    y_live = CV.free_run(step_fn, out_fn, un, yn, x0_live, bank, ctrl, k0=K0, closed=True)
    r = PLANT.zero_the_ann(fs)
    y_off = CV.free_run(step_fn, out_fn, un, yn, x0_froz, bank, ctrl, k0=K0, closed=True)
    r()
    d = np.abs(y_live - y_off)
    first = int(np.argmax(d.max(axis=1) > 0)) if (d.max(axis=1) > 0).any() else -1
    print('  %-22s dx0 %.3e   traj max|diff| %.3e   first differing sample %d'
          % (name, dx0, d.max(), first))

print('\n' + '=' * 88)
print('D2  IS THE CLOSED-LOOP METRIC BLIND?')
print('=' * 88)
print('Perturb the ANN final layer by increasing amounts and watch the closed-loop score.')
print('A metric that cannot see a deliberately broken model cannot select a good one.\n')

validator = CV.ClosedLoopValidator(fs, bank, step_fn, out_fn, names, K0,
                                   (na, nb, na_r, nb_r), dtype=cfg.dtype_pt, verbose=False)
import deepSI                                                             # noqa: E402
val_ckpt = deepSI.System_data_list(val)

last = list(ann.net.parameters())[-2]        # final layer weight
saved = last.detach().clone()
print('%-14s %-16s %-16s' % ('perturbation', 'closed-loop score', 'ratio to baseline'))
base = None
for s in [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
    with torch.no_grad():
        last.copy_(saved)
        if s > 0:
            gg = torch.Generator().manual_seed(1)
            last.add_(torch.randn(last.shape, generator=gg, dtype=last.dtype) * s)
    try:
        sc = validator(val_ckpt)
    except Exception as e:
        print('%-14.0e FAILED (%s)' % (s, type(e).__name__))
        continue
    if base is None:
        base = sc
    print('%-14.0e %-16.6e %-16.4f' % (s, sc, sc / base))
with torch.no_grad():
    last.copy_(saved)
print('\nrestored. [%.0fs]' % (time.time() - t0))
