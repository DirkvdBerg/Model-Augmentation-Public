"""Is the 0.35 % improvement limited by learning rate, by budget, or by something structural?

The sanity run improved the closed-loop validation score from 2.3138e-06 to 2.3057e-06 in 300
iterations at lr = 1e-3. That is real (the metric is deterministic) but tiny against the 77x
headroom, and three explanations are conflated:

  lr          Adam normalises by the second moment, so step size is ~lr regardless of the loss
              being 4e-10. lr = 1e-3 should move parameters by ~1e-3 per iteration.
  budget      300 iterations is 0.4 EPOCHS (748 batches/epoch). Probably the dominant limit.
  structural  zero-init gives ANN layers 0 and 2 EXACTLY zero gradient (G13, problem log s13).
              They stay frozen until the final layer leaves zero, which costs iterations at any lr.

Reasoning cannot separate these, so this sweeps lr at fixed budget and reports, for each arm:

  * the closed-loop validation score against the untrained baseline
  * the ANN final-layer weight norm (is the network moving at all, or crawling?)
  * the ANN FIRST-layer weight norm (has the dead zone opened, i.e. did layers 0/2 start moving?)

and then runs the ATTRIBUTION test on each trained model: force the ANN output to zero and
re-score. If the score returns to the untrained value, the improvement was the ANN and not the
encoder. G10 measured that the closed-loop score moves only 0.003-0.005 % between an untrained
encoder estimate and the true state, so the encoder cannot explain a 0.35 % change, but that is an
argument and this is the measurement.

Reading it:
  improvement scales with lr        -> rate/budget limited, raise both for step 6
  improvement saturates across lr   -> structural, and the dead zone or the horizon is the target
  first-layer norm stays 0          -> the dead zone never opened within this budget

Usage: python -u cl_lr_probe.py
"""
__project_origin__ = "added"

import copy
import dataclasses
import io
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

import deepSI                                                             # noqa: E402
import demo_common as dm                                                  # noqa: E402
from demo_common import CFG                                               # noqa: E402
from gantry_dynamic.data import load_traj, TRAIN_FILES, VAL_FILES         # noqa: E402
from model_augmentation.fit_systems.blocks import Static_ANN_Block        # noqa: E402

import cl_plant as PLANT                                                  # noqa: E402
import cl_fitsys as CLF                                                   # noqa: E402
import cl_validation as CV                                                # noqa: E402
from cl_controller import ControllerBank                                  # noqa: E402

NF, N_TRAIN, VAL_LEN = 100, 4, 6000
# (lr, n_its). Rate AND budget, because 300 its is 0.40 epochs and budget is the more likely
# limiter. The third arm is the same rate as the first at 5x the budget, so the two effects
# separate instead of being confounded.
ARMS = [(1e-3, 300), (1e-2, 300), (1e-3, 1500)]
t0 = time.time()

print('=' * 96)
print('LR PROBE: is the improvement rate-limited, budget-limited, or structural?')
print('=' * 96)
print('arms (lr, n_its): %s   nf %d, %d train records' % (ARMS, NF, N_TRAIN))

cfg = dataclasses.replace(CFG, seed=0)
train_names = [f[:-4] for f in TRAIN_FILES[:N_TRAIN]]
val_names = [VAL_FILES[0][:-4]]
train_list = [load_traj(f, cfg) for f in TRAIN_FILES[:N_TRAIN]]
train_data = deepSI.System_data_list(train_list)
sdv = load_traj(VAL_FILES[0], cfg)
val_data = deepSI.System_data_list(
    [deepSI.System_data(u=sdv.u[:VAL_LEN], y=sdv.y[:VAL_LEN], dt=sdv.dt)])

rows = []
for lr, N_ITS in ARMS:
    print('\n' + '=' * 96)
    print('lr = %.0e   n_its = %d  (%.2f epochs)' % (lr, N_ITS, N_ITS / 748.0), flush=True)
    # Fresh model per arm: same seed, so arms differ ONLY by (lr, n_its).
    fs, norm, K0, na, nb, na_r, nb_r = dm.build_pipeline(cfg=cfg, verbose=False)
    nx = cfg.nx_phys + cfg.nx_ann
    C_out, b_out = PLANT.identify_output_map(fs.hfn, nx, cfg.nu, dtype=cfg.dtype_pt)
    step_fn, out_fn = PLANT.make_fns(fs, C_out, b_out)
    bank_tr = ControllerBank(train_names, cfg.ts_new, dtype=cfg.dtype_pt,
                             ystd=norm.ystd, std_u=norm.std_u)
    bank_va = ControllerBank(val_names, cfg.ts_new, dtype=cfg.dtype_pt,
                             ystd=norm.ystd, std_u=norm.std_u)
    CLF.attach(fs, bank_tr, step_fn, out_fn)
    validator = CV.ClosedLoopValidator(fs, bank_va, step_fn, out_fn, val_names, K0,
                                       (na, nb, na_r, nb_r), dtype=cfg.dtype_pt, verbose=False)
    CV.install(fs, validator)

    ann = next(m for m in fs.hfn.connected_blocks if isinstance(m, Static_ANN_Block))
    pars = list(ann.net.parameters())
    n_first = float(pars[0].detach().norm())
    n_last = float(pars[-2].detach().norm())
    base = validator(val_data)
    print('  untrained: val %.10e   ANN |W_first| %.4e  |W_last| %.4e'
          % (base, n_first, n_last), flush=True)

    torch.save(fs.__dict__, io.BytesIO())          # pre-flight, see cl_sanity.py
    fs.fit(train_data, val_data, n_its=N_ITS, batch_size=256,
           loss_kwargs=dict(nf=NF), auto_fit_norm=False, validation_measure='sim-RMS',
           optimizer_kwargs=dict(optimizer=torch.optim.Adam, lr=lr),
           its_per_val=N_ITS, concurrent_val=False, verbose=0)

    # ---- REBUILD THE EVALUATION STACK FROM THE CURRENT fs -------------------------------
    # deepSI's fit ends with checkpoint_load_system('_best') (interconnect.py:716), which does
    # `self.__dict__ = torch.load(file)` (fit_system.py:501). That REPLACES fs.hfn with a freshly
    # deserialised object. Any step_fn, validator or ANN handle captured before fit now points at
    # the OLD module, so evaluating through them measures a different model than the one you then
    # patch. The first version of this probe did exactly that and reported ANN-off identical to
    # ANN-on to 11 digits, which looked like "the ANN contributes nothing" and was really
    # "the ANN I zeroed is not the ANN being evaluated".
    C_out, b_out = PLANT.identify_output_map(fs.hfn, nx, cfg.nu, dtype=cfg.dtype_pt)
    step_fn, out_fn = PLANT.make_fns(fs, C_out, b_out)
    validator = CV.ClosedLoopValidator(fs, bank_va, step_fn, out_fn, val_names, K0,
                                       (na, nb, na_r, nb_r), dtype=cfg.dtype_pt, verbose=False)
    ann = next(m for m in fs.hfn.connected_blocks if isinstance(m, Static_ANN_Block))
    pars = list(ann.net.parameters())
    # IDENTITY GUARD: the ANN we are about to zero must be the one the rollout actually reaches.
    reachable = next(m for m in step_fn.hfn.connected_blocks if isinstance(m, Static_ANN_Block))
    assert reachable is ann, ('the ANN being zeroed is not the ANN in the evaluation path; '
                             'the attribution test would be meaningless')

    final = validator(val_data)
    n_first_t = float(pars[0].detach().norm())
    n_last_t = float(pars[-2].detach().norm())

    # ---- ATTRIBUTION: force the ANN output to zero on the TRAINED model and re-score ----
    restore = PLANT.zero_the_ann(fs)
    annoff = validator(val_data)
    restore()
    # sanity: zeroing must actually change SOMETHING once the ANN is nonzero
    if n_last_t > 0 and annoff == final:
        print('  WARNING: ANN weights are nonzero (%.3e) yet zeroing the ANN changed the score by '
              'exactly 0. Suspect the patch is not in the evaluation path.' % n_last_t)

    imp = 100.0 * (base - final) / base
    imp_annoff = 100.0 * (base - annoff) / base
    print('  trained  : val %.10e   improvement %+.4f %%' % (final, imp))
    print('  ANN off  : val %.10e   improvement %+.4f %%  <- residual is NOT the ANN'
          % (annoff, imp_annoff))
    print('  ANN |W_first| %.4e -> %.4e   |W_last| %.4e -> %.4e'
          % (n_first, n_first_t, n_last, n_last_t))
    tr = np.asarray(fs.Loss_train, dtype=float)
    print('  train loss %.4e -> %.4e' % (tr[0], np.nanmin(tr)) if len(tr) else '  train loss n/a')
    rows.append((lr, N_ITS, base, final, annoff, imp, imp_annoff, n_first_t, n_last_t))

print('\n' + '=' * 96)
print('SUMMARY')
print('=' * 96)
print('%-8s %-7s %-14s %-14s %-11s %-11s %-12s'
      % ('lr', 'n_its', 'val trained', 'val ANN-off', 'improv %', 'ANN-off %', '|W_last|'))
for lr, nit, base, final, annoff, imp, imp_a, nf_, nl_ in rows:
    print('%-8.0e %-7d %-14.6e %-14.6e %-11.4f %-11.4f %-12.4e'
          % (lr, nit, final, annoff, imp, imp_a, nl_))
print('\nANN CONTRIBUTION = improv %% minus ANN-off %%  (what the ANN itself bought):')
for lr, nit, base, final, annoff, imp, imp_a, nf_, nl_ in rows:
    print('  lr %.0e n_its %-5d  total %+.4f %%   encoder-only %+.4f %%   ANN %+.4f %%'
          % (lr, nit, imp, imp_a, imp - imp_a))
ann_share = np.array([r[5] - r[6] for r in rows])
print('\nANN-only improvement across arms: %s' % ' '.join('%+.4f %%' % v for v in ann_share))
print('\nREADING IT')
print('  ANN column ~0 everywhere        -> the ANN is still inactive; the loop fixed the METRIC')
print('                                     but not the learning. D-067 / variant A persisting.')
print('  ANN column grows with lr        -> rate limited, raise lr for step 6')
print('  ANN column grows with n_its     -> budget limited, the 0.4-epoch smoke test was too short')
print('  ANN grows with neither          -> structural: dead zone, horizon, or model class')
print('[%.0fs]' % (time.time() - t0))
