"""SANITY RUN: does closed-loop training actually run end to end, through deepSI's own fit()?

NOT an experiment. This asks four yes/no questions and nothing else:

  S1  the training loss decreases
  S2  nothing goes NaN or infinite
  S3  the closed-loop validation score is finite and MOVES between validations
  S4  the best checkpoint is not the untrained model

S4 is the specific failure that wasted variants A and B: with the open-loop metric, drift moved
the score 36x per epoch and selection returned epoch 0 on every run. If it happens again here it
means the closed-loop metric has not fixed it, and step 6 should not be launched.

It goes through `fit_sys.fit(...)` rather than a bespoke loop, so the loss, the batching, the
fifth-array plumbing, and the replaced selection hook are all exercised on the real path.

DELIBERATELY SMALL, and none of these numbers are the ones to quote for anything:
  4 train records, nf = 100, 300 iterations, validation on ONE truncated val record. A full-length
  closed-loop free run over four val records costs about 12 minutes per validation (step 3 took
  3022 s for 16 of them), which is fine once per epoch in a real run and useless in a smoke test.

concurrent_val MUST be False: deepSI's concurrent path validates in a separate process, which
would pickle the fit system and lose the monkeypatched cal_validation_error, silently reverting
selection to the open-loop measure. That is exactly the variant B failure, so it is pinned here.

Usage: python -u cl_sanity.py
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

import deepSI                                                             # noqa: E402
import demo_common as dm                                                  # noqa: E402
from demo_common import CFG                                               # noqa: E402
from gantry_dynamic.data import load_traj, TRAIN_FILES, VAL_FILES         # noqa: E402

import cl_plant as PLANT                                                  # noqa: E402
import cl_fitsys as CLF                                                   # noqa: E402
import cl_validation as CV                                                # noqa: E402
from cl_controller import ControllerBank                                  # noqa: E402

NF = 100
N_TRAIN = 4
N_ITS = 300
ITS_PER_VAL = 100
VAL_LEN = 6000            # truncated val record, smoke test only
LR = 1e-3
t0 = time.time()

print('=' * 92)
print('SANITY RUN: closed-loop training through deepSI fit()')
print('=' * 92)
cfg = dataclasses.replace(CFG, seed=0)
fs, norm, K0, na, nb, na_r, nb_r = dm.build_pipeline(cfg=cfg, verbose=False)
nx = cfg.nx_phys + cfg.nx_ann
C_out, b_out = PLANT.identify_output_map(fs.hfn, nx, cfg.nu, dtype=cfg.dtype_pt)
step_fn, out_fn = PLANT.make_fns(fs, C_out, b_out)

train_names = [f[:-4] for f in TRAIN_FILES[:N_TRAIN]]
val_names = [VAL_FILES[0][:-4]]
# SEPARATE banks: rec_ix in the loss indexes TRAIN records, rec_ix in the validator indexes VAL
# records. One shared bank would silently attach a training record's controller to a val record.
bank_train = ControllerBank(train_names, cfg.ts_new, dtype=cfg.dtype_pt,
                            ystd=norm.ystd, std_u=norm.std_u)
bank_val = ControllerBank(val_names, cfg.ts_new, dtype=cfg.dtype_pt,
                          ystd=norm.ystd, std_u=norm.std_u)
print('train %s' % train_names)
print('val   %s (truncated to %d samples)' % (val_names, VAL_LEN))

train_list = [load_traj(f, cfg) for f in TRAIN_FILES[:N_TRAIN]]
train_data = deepSI.System_data_list(train_list)
sdv = load_traj(VAL_FILES[0], cfg)
val_trunc = deepSI.System_data(u=sdv.u[:VAL_LEN], y=sdv.y[:VAL_LEN], dt=sdv.dt)
val_data = deepSI.System_data_list([val_trunc])

CLF.attach(fs, bank_train, step_fn, out_fn)
validator = CV.ClosedLoopValidator(fs, bank_val, step_fn, out_fn, val_names, K0,
                                   (na, nb, na_r, nb_r), dtype=cfg.dtype_pt, verbose=True)
CV.install(fs, validator)
print('loss = closed loop (xc = 0 per window, Remark 5.4), selection = closed-loop sim-RMS')
print('nf %d, n_its %d, its_per_val %d, lr %.0e   [%.0fs]'
      % (NF, N_ITS, ITS_PER_VAL, LR, time.time() - t0), flush=True)

# PRE-FLIGHT: deepSI's checkpoint_save_system does torch.save(self.__dict__) at EVERY validation
# (fit_system.py:496), so anything unpicklable grafted onto the instance kills the run at the first
# validation rather than at the start. Variant B lost its verdict block to this after 3.2 h. Check
# it here, where it costs a second.
import io                                                                 # noqa: E402
try:
    torch.save(fs.__dict__, io.BytesIO())
    print('pre-flight: fit_sys.__dict__ is picklable')
except Exception as e:
    print('pre-flight FAILED: %s: %s' % (type(e).__name__, e))
    print('checkpointing would die at the first validation. Fix before running.')
    sys.exit(2)

print('\n' + '-' * 92, flush=True)
fs.fit(train_data, val_data, n_its=N_ITS, batch_size=256,
       loss_kwargs=dict(nf=NF), auto_fit_norm=False, validation_measure='sim-RMS',
       optimizer_kwargs=dict(optimizer=torch.optim.Adam, lr=LR),
       its_per_val=ITS_PER_VAL, concurrent_val=False, verbose=2)
print('-' * 92, flush=True)

tr = np.asarray(fs.Loss_train, dtype=float)
vl = np.asarray(fs.Loss_val, dtype=float)
print('\ntrain loss series: %s' % ' '.join('%.4e' % v for v in tr))
print('val   loss series: %s' % ' '.join('%.4e' % v for v in vl))
print('validator history: %s' % ' '.join('%.4e' % s for s, _ in validator.history))

s1 = len(tr) > 1 and np.nanmin(tr) < tr[0]
s2 = bool(np.all(np.isfinite(tr)) and np.all(np.isfinite(vl)))
s3 = len(vl) > 1 and np.all(np.isfinite(vl)) and (np.nanmax(vl) - np.nanmin(vl)) > 0
s4 = len(vl) > 1 and int(np.nanargmin(vl)) != 0

print('\n' + '=' * 92)
print('S1  train loss decreases          %s   (first %.4e -> best %.4e)'
      % ('PASS' if s1 else 'FAIL', tr[0] if len(tr) else float('nan'),
         np.nanmin(tr) if len(tr) else float('nan')))
print('S2  everything finite             %s' % ('PASS' if s2 else 'FAIL'))
print('S3  val score finite and moving   %s   (spread %.3e)'
      % ('PASS' if s3 else 'FAIL', (np.nanmax(vl) - np.nanmin(vl)) if len(vl) else float('nan')))
print('S4  best checkpoint is not the untrained model  %s   (argmin at validation %d of %d)'
      % ('PASS' if s4 else 'FAIL', int(np.nanargmin(vl)) if len(vl) else -1, len(vl)))
print('=' * 92)
ok = s1 and s2 and s3
print('SANITY %s   [%.0fs]' % ('PASSED' if ok else 'HAS FAILURES', time.time() - t0))
if not s4:
    print('NOTE: S4 failed. On a 300-iteration smoke test that is not conclusive by itself, but')
    print('      it is the exact signature that invalidated variants A and B, so it must be')
    print('      understood before step 6 rather than assumed to be the short horizon.')
sys.exit(0 if ok else 1)
