"""The server job: the full black box trained and selected in the grey box's closed loop (BB-005, BB-010).

    BB_ARM=<random|n4sid|frf> python -u scripts/gantry/blackbox-closed-loop/runners/train_bb.py
    BB_ARM=... python -u .../train_bb.py --print-config        # configuration snapshot, no run
    BB_ARM=... python -u .../train_bb.py --smoke-records 4 --n-its 2   # G4 local smoke (BB-011)

Everything except the black box itself is the grey box's, read at run time from its entry file's
CFG: data folder, split, rate, horizon, stride, burn-in, batch, epochs, its_per_val, selection,
the Adam call (`gantry_dynamic.model.train_model`) and the L-BFGS polish
(`gantry_dynamic.training.run_lbfgs_polish`). The black-box config is `replace(CFG, lr, adam_eps)`
and nothing else (BB-005). On the server `device` and `compile_mode` are CFG's own.
Outputs: scripts/gantry/blackbox-closed-loop/outputs/server/<arm>_<run_id>/ (or smoke_<...>).
"""
import argparse
import json
import os
import sys
import time
from dataclasses import replace, fields

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'adapter'))
import bbcl                                                                # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--smoke-records', type=int, default=0,
               help='G4 only: first K training records, CPU, eager, no polish (BB-011)')
p.add_argument('--n-its', type=int, default=None, help='G4 only: optimizer-update cap')
p.add_argument('--print-config', action='store_true')
args = p.parse_args()
smoke = args.smoke_records > 0
if smoke and args.n_its is None:
    raise SystemExit('--smoke-records needs --n-its (the session budget is 2 updates)')

cfg0, CFG, entry = bbcl.load_cfg(local=smoke)
import numpy as np                                                         # noqa: E402
import torch                                                               # noqa: E402
import deepSI                                                              # noqa: E402
import bb_model                                                            # noqa: E402
import cl_score                                                            # noqa: E402

ARM = os.environ.get('BB_ARM')
if ARM not in bb_model.ARMS:
    raise SystemExit('BB_ARM must be one of %s, got %r' % (bb_model.ARMS, ARM))
bb_cfg = replace(cfg0, lr=bb_model.BB_LR, adam_eps=bb_model.BB_EPS)
if smoke:
    bb_cfg = replace(bb_cfg, n_its=args.n_its, lbfgs=False)
elif args.n_its is not None:
    raise SystemExit('--n-its is a smoke option; the server run takes n_its from CFG')

print('BB_ARM=%s  smoke=%s' % (ARM, smoke))
print('=== EFFECTIVE configuration (grey box CFG; differences listed) ===')
for f in sorted(fields(bb_cfg), key=lambda f: f.name):
    v, V = getattr(bb_cfg, f.name), getattr(CFG, f.name)
    print('  %-26s %r%s' % (f.name, v, '' if v == V else '   <- CFG %r' % (V,)))
for k in ('nf', 'na_nb', 'ts_new'):
    print('  %-26s %r' % ('DERIVED ' + k, getattr(bb_cfg, k)))
BB_SET = dict(arm=ARM, nx=bb_model.NX, f=bb_model.F_KW, h=bb_model.H_KW, encoder=bb_model.E_KW,
              lr=bb_model.BB_LR, eps=bb_model.BB_EPS, na=bb_cfg.na_nb, na_right=bb_model.NA_RIGHT)
print('  black box: %s' % BB_SET)
if args.print_config:
    raise SystemExit(0)

from gantry_dynamic.config import config_json_dict, git_provenance        # noqa: E402
from gantry_dynamic.data import (load_datasets, compute_normalization,     # noqa: E402
                                 TRAIN_FILES, VAL_FILES, TEST_FILES)
from gantry_dynamic.model import train_model                               # noqa: E402
from gantry_dynamic.training import run_lbfgs_polish                       # noqa: E402

run_id = os.environ.get('SLURM_JOB_ID') or time.strftime('%Y%m%d_%H%M%S')
sdir = os.path.join(bbcl.OUT, 'server' if not smoke else 'g4', ('smoke_' if smoke else '')
                    + '%s_%s' % (ARM, run_id))
os.makedirs(sdir, exist_ok=True)
print('run_id=%s  out=%s' % (run_id, sdir))
if smoke:
    # BB-012: deepSI writes fit()'s _best/_last checkpoints under %LOCALAPPDATA%/deepSI on Windows
    # (datasets/dataset_utils.py get_work_dirs). Locally that is outside this folder, so the
    # process's LOCALAPPDATA is pointed inside the run folder. The server run keeps ~/.deepSI.
    os.environ['LOCALAPPDATA'] = os.path.join(sdir, 'localappdata')
    os.makedirs(os.environ['LOCALAPPDATA'], exist_ok=True)

np.random.seed(bb_cfg.seed); torch.manual_seed(bb_cfg.seed)
data = load_datasets(bb_cfg)
norm = compute_normalization(bb_cfg, data)          # from all 18 training records, as the grey box
train_files = list(TRAIN_FILES)
if smoke:
    train_files = train_files[:args.smoke_records]
    data = replace(data, train_list=data.train_list[:args.smoke_records],
                   train_data=deepSI.System_data_list(data.train_list[:args.smoke_records]))
    print('[smoke] training records: %s' % train_files)
hp = bb_cfg.hp
fs = bb_model.build_blackbox(bb_cfg, data, norm, arm=ARM, attach=True, train_files=train_files)
fs.unique_code = 'bbcl_%s_%s' % (ARM, run_id)       # deepSI checkpoint names, as training.py does

json.dump({**config_json_dict(bb_cfg, git=git_provenance()), 'hp': hp, 'run_id': run_id,
           'black_box': BB_SET, 'smoke': smoke, 'train_files': train_files},
          open(os.path.join(sdir, 'config.json'), 'w'), indent=2, default=str)

# Wall time per optimizer update: the closure (loss, backward) runs INSIDE optimizer.step, so
# timing step times the whole update. Instance attribute; fit() keeps this optimizer
# (init_model_done is already True).
_t_upd = []
_step = fs.optimizer.step


def _timed_step(*a, **k):
    t0 = time.perf_counter()
    out = _step(*a, **k)
    _t_upd.append(time.perf_counter() - t0)
    return out


fs.optimizer.step = _timed_step
_t_val = []
_cve = fs.cal_validation_error


def _timed_val(*a, **k):
    t0 = time.perf_counter()
    out = _cve(*a, **k)
    _t_val.append(time.perf_counter() - t0)
    return out


fs.cal_validation_error = _timed_val

t0 = time.time()
bestfit = train_model(fs, hp, bb_cfg, data, nf=hp['nf'], validation_measure='sim-RMS')
t_adam = time.time() - t0
print('\n[bb] Adam phase done in %.1f s: bestfit %.6e m, %d updates, %d validations'
      % (t_adam, bestfit, len(_t_upd), len(_t_val)))
# Remove the timing wrappers. fit() ends with `self.__dict__ = torch.load(best)`, which brings back
# the instance attributes as they were pickled (by reference to this module's functions), so
# they are popped from whatever objects the system holds NOW rather than reassigned.
fs.__dict__.pop('cal_validation_error', None)
getattr(fs.optimizer, '__dict__', {}).pop('step', None)
res = dict(arm=ARM, run_id=run_id, smoke=smoke, bestfit_adam=float(bestfit), t_adam=t_adam,
           t_updates=_t_upd, t_validations=_t_val,
           Loss_val=[float(v) for v in np.asarray(fs.Loss_val)],
           Loss_train=[float(v) for v in np.asarray(fs.Loss_train)],
           epoch_id=[float(v) for v in np.asarray(fs.epoch_id)])

polish = run_lbfgs_polish(fs, hp, bb_cfg, data, bestfit=bestfit)
if polish is not None:
    res['lbfgs'] = {k: v for k, v in polish.items() if isinstance(v, (int, float, str, bool))}
    if polish.get('accepted'):
        bestfit = polish['val_after']
res['bestfit'] = float(bestfit)

torch.save({'encoder': fs.encoder.state_dict(), 'hfn': fs.hfn.state_dict(),
            'norm': dict(u0=fs.norm.u0, ustd=fs.norm.ustd, y0=fs.norm.y0, ystd=fs.norm.ystd),
            'black_box': BB_SET}, os.path.join(sdir, 'weights.pt'))

# Final scores, same harness: validation (selection) and test (reporting), per record.
fs.cpu()
for split, files, sdl in (('val', VAL_FILES, data.val_list), ('test', TEST_FILES, data.test_list)):
    names = [f[:-4] for f in files]
    t0 = time.time()
    s, per, perch, _, _ = cl_score.closed_loop_score(fs, names, sdl, bb_cfg.ts_new,
                                                     dtype=bb_cfg.dtype_pt)
    res[split] = dict(score=float(s), per_record=dict(zip(files, [float(v) for v in per])),
                      per_channel={f: [float(v) for v in c] for f, c in zip(files, perch)},
                      seconds=time.time() - t0)
    print('[bb] %s closed-loop sim-RMS %.6e m  (%s)' % (split, s, ', '.join(
        '%s %.3e' % (f[:-4], v) for f, v in zip(files, per))))
json.dump(res, open(os.path.join(sdir, 'result.json'), 'w'), indent=1, default=str)
print('[bb] done: %s' % os.path.join(sdir, 'result.json'))
