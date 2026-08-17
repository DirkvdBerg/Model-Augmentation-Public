"""STEP 6: the first real closed-loop run. Controller in the model, ANN routed to all eight states.

Everything this rests on is gated: G1-G6 (`cl_gate_replay.py`), G7-G10 (`cl_gate_validation.py`),
G11-G14 (`cl_gate_loss.py`), plus `cl_sanity.py` and `cl_lr_probe.py`. Decisions: D-140 placement,
D-141 rate and location, D-142 initialisation and headroom. Run-table row written before launch
(D-090), problem log section 12.

CONFIG, and why each value
--------------------------
  ann_route_ix = (0..7)   the user's decision. Reopens the D-066/D-067 position-row question: X and
                          Y have K = 0, so additive corrections there integrate without a restoring
                          force OPEN loop. The loop is expected to supply it (D-139 measured DC
                          drift +8.2e-05 m open against ~1e-12 m closed), but that is the thing
                          being tested, not an assumption.
  nf = 400                0.1 s, five absorber time constants (tau_msd ~ 20 ms), fifteen periods of
                          the 150 Hz mode. D-139.
  stride = 10             windows at stride 1 overlap 99.75 %; stride 10 cuts iterations per epoch
                          10x for almost no loss of information. Precedented in this repo's
                          black-box runs.
  lr on param_groups      `fit_system.py:311` only builds the optimizer when `init_model_done` is
                          False, so `optimizer_kwargs` passed to `fit()` on a pre-built model is
                          SILENTLY IGNORED. The lr probe's first two arms were identical to 11
                          digits because of this. Set it on the existing optimizer instead.
  full-length validation  the lr probe used a 6000-sample truncated record, where the init transient
                          occupies ~8x more of the scored window than on the full 48000. That
                          flatters the encoder. Selection here runs the full V1-V4.
  concurrent_val = False  the concurrent path validates in a subprocess, which pickles the fit
                          system and loses the monkeypatched cal_validation_error, silently
                          reverting selection to the OPEN-loop measure. That is the variant B
                          failure exactly.

WHAT WOULD FALSIFY THE HYPOTHESIS
---------------------------------
Pre-registered: the ANN contribution was +0.355 % at 300 iterations and +1.671 % at 1500, linear
with no saturation. If it plateaus far below the 77x headroom while iterations keep rising, the
reading changes from "budget limited" to "structurally limited" and the target becomes the W^a dead
zone (problem log s13), the horizon, or the model class. If best-checkpoint returns to iteration 0,
the closed-loop metric has not fixed what killed D-067 and variant A.

Usage:
  CL_ITS=20000 CL_LR=1e-3 python -u cl_step6_run.py
"""
__project_origin__ = "added"

import dataclasses
import io
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

import deepSI                                                             # noqa: E402
import demo_common as dm                                                  # noqa: E402
from demo_common import CFG                                               # noqa: E402
from gantry_dynamic.data import load_traj, TRAIN_FILES, VAL_FILES         # noqa: E402
from model_augmentation.fit_systems.blocks import Static_ANN_Block        # noqa: E402

import cl_plant as PLANT                                                  # noqa: E402
import cl_fitsys as CLF                                                   # noqa: E402
import cl_validation as CV                                                # noqa: E402
from cl_controller import ControllerBank                                  # noqa: E402

# EPOCHS is the bound the user asked for. Note deepSI cannot honour both: with `timeout` set it
# runs `itertools.count()` and IGNORES n_its/epochs entirely (interconnect.py:604). So the two are
# mutually exclusive, and choosing epochs means the SLURM wall clock is the only backstop. Size -t
# with margin: on a hard kill the _best/_last checkpoints survive but the result JSON and the
# per-record attribution do NOT, because they run after fit() returns.
EPOCHS = int(os.environ.get('CL_EPOCHS', 12))
N_ITS = int(os.environ.get('CL_ITS', 0)) or None
# DEFAULT 1e-7, NOT 1e-3. config.py:62 states that routing to the K = 0 rows (X/Y: 0,2,3,5) needs
# "a much smaller lr (~1e-7)" per D-101/D-102, and ann_route_ix = (0..7) includes exactly those
# rows. The first launch of this script used 1e-3 and hit a NaN in the training loss at iteration
# 81. Note also that D-101 is the reason `build_model` passes lr into `init_model`: fit()'s
# optimizer_kwargs are ignored once init_model_done is True, and the accidental Adam default that
# every earlier gantry run trained at was 1e-3, i.e. exactly the value that just diverged.
LR = float(os.environ.get('CL_LR', 1e-7))
# 'epoch' is deepSI's own default and resolves to N_batch_updates_per_epoch, giving one validation
# point per epoch, i.e. 12 points over this run. Each is a full closed-loop free run over all four
# validation records, measured at ~8 min, so this is ~1.6 h of the budget. Set an integer here to
# validate less often if the node turns out to be slow.
_ipv = os.environ.get('CL_ITS_PER_VAL', 'epoch')
ITS_PER_VAL = _ipv if _ipv == 'epoch' else int(_ipv)
STRIDE = int(os.environ.get('CL_STRIDE', 10))
TIMEOUT = float(os.environ.get('CL_TIMEOUT', 0)) or None
ROUTE = tuple(range(8))
# Tag the output so a server job cannot overwrite a local one, or a rerun a previous run.
TAG = os.environ.get('CL_TAG') or os.environ.get('SLURM_JOB_ID') or 'local'
OUT = os.path.join(HERE, 'runs', 'step6_result_%s.json' % TAG)
# Create it NOW, not at the end: the JSON is written after fit() returns, so a missing directory
# would raise only after the full run and lose the result of hours of training.
os.makedirs(os.path.dirname(OUT), exist_ok=True)
t0 = time.time()

# NOTE on CL_TIMEOUT: deepSI IGNORES n_its when timeout is set
# (`rang = range(n_its) if timeout is None else itertools.count(0)`, interconnect.py:604), so the
# timeout is the ONLY bound in that case. Size it BELOW the job's wall clock: the untrained
# validation, the two final evaluations and the JSON all run AFTER fit() returns, and on a hard
# SLURM kill deepSI's _best/_last checkpoints survive but the result JSON and the printed
# attribution do NOT (the same lesson as run 74045, problem log).

print('=' * 96)
print('STEP 6: closed-loop training, controller in the model, ANN -> all eight states')
print('=' * 96)
# lr goes through cfg so build_model creates the optimizer with it (D-101, model.py:227). Setting
# it afterwards on param_groups would OVERRIDE that existing fix, which is what caused the NaN.
cfg = dataclasses.replace(CFG, seed=0, ann_route_ix=ROUTE, lr=LR)
print('ann_route_ix %s   nf %d   stride %d   lr %.0e   epochs %s   n_its %s   its_per_val %s'
      % (str(ROUTE), cfg.nf, STRIDE, LR, EPOCHS, N_ITS, ITS_PER_VAL))
if TIMEOUT:
    print('WARNING: CL_TIMEOUT is set, so deepSI IGNORES epochs and n_its and runs on the clock')

fs, norm, K0, na, nb, na_r, nb_r = dm.build_pipeline(cfg=cfg, verbose=True)
nx = cfg.nx_phys + cfg.nx_ann
ann = next(m for m in fs.hfn.connected_blocks if isinstance(m, Static_ANN_Block))
print('ANN output width %d (one per routed row)' % ann.net.net[-1].out_features
      if hasattr(ann.net, 'net') else 'ANN built')

C_out, b_out = PLANT.identify_output_map(fs.hfn, nx, cfg.nu, dtype=cfg.dtype_pt)
step_fn, out_fn = PLANT.make_fns(fs, C_out, b_out)

train_names = [f[:-4] for f in TRAIN_FILES]
val_names = [f[:-4] for f in VAL_FILES]
bank_tr = ControllerBank(train_names, cfg.ts_new, dtype=cfg.dtype_pt,
                         ystd=norm.ystd, std_u=norm.std_u)
bank_va = ControllerBank(val_names, cfg.ts_new, dtype=cfg.dtype_pt,
                         ystd=norm.ystd, std_u=norm.std_u)
print('controller bank: n_FB %d, distinct Y_op train %s / val %s'
      % (bank_tr.nc, bank_tr.y_ops_unique, bank_va.y_ops_unique))

train_data = deepSI.System_data_list([load_traj(f, cfg) for f in TRAIN_FILES])
val_data = deepSI.System_data_list([load_traj(f, cfg) for f in VAL_FILES])   # FULL length

CLF.attach(fs, bank_tr, step_fn, out_fn)
validator = CV.ClosedLoopValidator(fs, bank_va, step_fn, out_fn, val_names, K0,
                                   (na, nb, na_r, nb_r), dtype=cfg.dtype_pt, verbose=True)
CV.install(fs, validator)

# CONSISTENCY CHECK, not an override. build_model already passed cfg.lr into init_model (D-101),
# so the optimizer should already carry it. Overriding here is what set 1e-3 on top of the correct
# value and produced the NaN, so this only verifies and reports.
got = [g['lr'] for g in fs.optimizer.param_groups]
print('optimizer lr from build_model: %s   (cfg.lr = %.0e)' % (got, LR))
assert all(abs(v - LR) < 1e-15 for v in got), (
    'optimizer lr %s does not match cfg.lr %g; build_model did not pass it through and an '
    'override here would repeat the D-101 mistake' % (got, LR))

torch.save(fs.__dict__, io.BytesIO())                 # pre-flight, cl_sanity.py
print('pre-flight: picklable   [%.0fs]' % (time.time() - t0), flush=True)

base = validator(val_data)
base_pr = validator.history[-1][1]           # per record, see the per-record block at the end
print('\nUNTRAINED closed-loop sim-RMS: %.10e m' % base)
print('oracle floor (cl_headroom.py): ~2.81e-08 m, i.e. %.1fx headroom' % (base / 2.81e-08))
print('\n' + '-' * 96, flush=True)

fs.fit(train_data, val_data, epochs=EPOCHS, n_its=N_ITS, batch_size=256,
       loss_kwargs=dict(nf=cfg.nf, stride=STRIDE), auto_fit_norm=False,
       validation_measure='sim-RMS', its_per_val=ITS_PER_VAL,
       concurrent_val=False, verbose=2, timeout=TIMEOUT)
print('-' * 96, flush=True)

# ---- rebuild the eval stack: fit() replaced fs.__dict__ via checkpoint_load_system ----
C_out, b_out = PLANT.identify_output_map(fs.hfn, nx, cfg.nu, dtype=cfg.dtype_pt)
step_fn, out_fn = PLANT.make_fns(fs, C_out, b_out)
validator = CV.ClosedLoopValidator(fs, bank_va, step_fn, out_fn, val_names, K0,
                                   (na, nb, na_r, nb_r), dtype=cfg.dtype_pt, verbose=False)
ann = next(m for m in fs.hfn.connected_blocks if isinstance(m, Static_ANN_Block))
assert next(m for m in step_fn.hfn.connected_blocks
            if isinstance(m, Static_ANN_Block)) is ann, 'eval-path ANN identity mismatch'

final = validator(val_data)
final_pr = validator.history[-1][1]
restore = PLANT.zero_the_ann(fs)
annoff = validator(val_data)
annoff_pr = validator.history[-1][1]
restore()
imp = 100.0 * (base - final) / base
imp_a = 100.0 * (base - annoff) / base
vl = np.asarray(fs.Loss_val, dtype=float)

print('\n' + '=' * 96)
print('RESULT')
print('=' * 96)
print('untrained            %.10e m' % base)
print('trained              %.10e m   %+.4f %%' % (final, imp))
print('trained, ANN forced 0 %.10e m   %+.4f %%' % (annoff, imp_a))
print('ANN contribution     %+.4f %%   (total minus encoder-only)' % (imp - imp_a))
print('oracle floor         ~2.81e-08 m; fraction of headroom closed %.2f %%'
      % (100.0 * (base - final) / (base - 2.81e-08)))
print('val series (%d points): %s' % (len(vl), ' '.join('%.4e' % v for v in vl)))

# ---- PER RECORD. Selection uses the quadratic-mean AGGREGATE, so a model that improves three
# records and degrades one can still be selected and would read as progress. The four records are
# different regimes with DIFFERENT controllers (V1 standstill Y=+0.10 at 130-180 Hz, V2 APRBS
# motion Y=-0.22 near 10 Hz, V3 Y-sweep Y=+0.10, V4 lissajous Y=-0.10), and with the ANN routed to
# the K = 0 rows there is no reason for them to move together. An aggregate improvement is not
# believable until no record has moved the wrong way.
print('\n%-22s %-13s %-13s %-13s %-10s %-10s' %
      ('record', 'untrained', 'trained', 'ANN off', 'improv %', 'ANN %'))
worst = None
for nm, b_, f_, a_ in zip(val_names, base_pr, final_pr, annoff_pr):
    i_tot = 100.0 * (b_ - f_) / b_
    i_ann = i_tot - 100.0 * (b_ - a_) / b_
    print('%-22s %-13.6e %-13.6e %-13.6e %-+10.4f %-+10.4f' % (nm, b_, f_, a_, i_tot, i_ann))
    if worst is None or i_tot < worst[1]:
        worst = (nm, i_tot)
print('worst record: %s at %+.4f %%   %s'
      % (worst[0], worst[1],
         'ALL RECORDS IMPROVED' if worst[1] > 0 else
         'A RECORD GOT WORSE: the aggregate is hiding a regression, do not read it as progress'))
if len(vl) > 1:
    print('best at validation %d of %d   %s'
          % (int(np.nanargmin(vl)), len(vl),
             'FALSIFIER FIRED: selection returned iteration 0' if int(np.nanargmin(vl)) == 0
             else 'selection picked a trained checkpoint'))
json.dump(dict(base=base, final=final, annoff=annoff, imp=imp, imp_annoff=imp_a, epochs=EPOCHS,
               per_record=dict(names=val_names, base=list(map(float, base_pr)),
                               final=list(map(float, final_pr)),
                               annoff=list(map(float, annoff_pr))),
               val=vl.tolist(), n_its=N_ITS, lr=LR, stride=STRIDE, nf=cfg.nf,
               route=list(ROUTE), seconds=time.time() - t0),
          open(OUT, 'w'), indent=2)
print('\nwrote %s   [%.0fs]' % (OUT, time.time() - t0))
