"""G0 (JH-002): build the production model from the vendored entry file and reproduce its epoch-0
closed-loop validation number (what fit() stores as Loss_val[0]).

    --source vendor|repo   vendor = this folder's copies (default); repo = the original packages,
                           imported in place in this separate process (criterion a', read only)
    --arm production|noproj14
                           production = the entry file's CFG as is (criterion a);
                           noproj14   = OBC_ARM=noproj on the 14-record dataset of run 84032 (b)

Never calls main(): nothing is written outside outputs/. Only device='cpu' and compile_mode=None
are overridden (JH-002).
"""
import argparse
import importlib.util
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('--source', default='vendor', choices=['vendor', 'repo'])
ap.add_argument('--arm', default='production', choices=['production', 'noproj14'])
ap.add_argument('--out', default=None)
args = ap.parse_args()

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import jh_env  # noqa: E402

if args.arm == 'noproj14':
    if args.source == 'repo':
        raise SystemExit('noproj14 needs the JH_FILESET switch, which only the vendored data.py has')
    os.environ['JH_FILESET'] = 'obc14'
    os.environ['OBC_ARM'] = 'noproj'

if args.source == 'vendor':
    entry_path = os.path.join(jh_env.VENDOR, 'gantry_interconnect_dynamic.py')
else:
    # Original packages: take vendor/ OFF the path and resolve from the repo, as the entry does.
    while jh_env.VENDOR in sys.path:
        sys.path.remove(jh_env.VENDOR)
    sys.path.insert(0, jh_env.REPO)
    entry_path = os.path.join(jh_env.REPO, 'scripts', 'gantry', 'gantry_interconnect_dynamic.py')

spec = importlib.util.spec_from_file_location('gantry_interconnect_dynamic', entry_path)
entry = importlib.util.module_from_spec(spec)
sys.modules['gantry_interconnect_dynamic'] = entry
spec.loader.exec_module(entry)

from dataclasses import replace  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from gantry_dynamic.data import (load_datasets, compute_normalization,  # noqa: E402
                                 TRAIN_FILES, VAL_FILES)
from gantry_dynamic.model import build_model  # noqa: E402
from gantry_dynamic.controller import build_closed_loop  # noqa: E402
from gantry_dynamic.training import resolve_resume_hp  # noqa: E402

import gantry_dynamic, model_augmentation  # noqa: E401,E402
print('[g0] entry file      :', entry.__file__)
print('[g0] gantry_dynamic  :', gantry_dynamic.__file__)
print('[g0] model_augmentation:', model_augmentation.__file__)
if args.source == 'vendor':
    jh_env.check_no_leak()
else:
    for n, m in list(sys.modules.items()):
        if n.split('.')[0] in ('model_augmentation', 'gantry_dynamic') and getattr(m, '__file__', None):
            assert not os.path.abspath(m.__file__).startswith(jh_env.VENDOR), (n, m.__file__)
    print('[g0] repo mode: no module from vendor/')

cfg = entry.CFG
if args.arm == 'noproj14':
    cfg = replace(cfg, mode='augmentation_ma50_b140-230_a6_z03', epochs=150)
cfg = replace(cfg, device='cpu', compile_mode=None)
print('[g0] cfg: mode=%s joint=%s param=%s detune=%s prior=%s obc=%s nf=%d burn_in=%d dtype=%s'
      % (cfg.mode, cfg.joint_estimation, cfg.physics_parameterization, cfg.combo_init_detune,
         cfg.param_prior, cfg.obc, cfg.nf, cfg.burn_in, cfg.dtype_pt))
print('[g0] files: %d train, %d val' % (len(TRAIN_FILES), len(VAL_FILES)))

t0 = time.time()
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
data = load_datasets(cfg)
norm = compute_normalization(cfg, data)
hp = resolve_resume_hp(cfg, cfg.hp, None)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
fit_sys = build_model(hp, cfg, data, norm)
fit_sys.simulator = build_closed_loop(fit_sys, norm, cfg, train_files=TRAIN_FILES,
                                      val_files=VAL_FILES, val_data=data.val_ckpt_data)
blk = fit_sys.hfn.connected_blocks[0]
print('[g0] physical block:', type(blk).__name__)
if hasattr(blk, 'identifiable_combinations'):
    print('[g0] start combos:', {k: round(v, 6) for k, v in blk.identifiable_combinations().items()})
print('[g0] build %.1f s; validating (closed-loop free run over %d records) ...'
      % (time.time() - t0, len(VAL_FILES)))
t1 = time.time()
fit_sys.eval()
with torch.no_grad():
    v = fit_sys.cal_validation_error(data.val_ckpt_data, validation_measure='sim-RMS')
per = [float(x) for x in fit_sys.simulator.last_per_record]
print('[g0] validation %.1f s' % (time.time() - t1))
print('[g0] per record [m]:', ' '.join('%s %.10e' % (f[:-4], x) for f, x in zip(VAL_FILES, per)))
print('G0 EPOCH0 source=%s arm=%s Loss_val0=%.12e' % (args.source, args.arm, v))
if args.source == 'vendor':
    jh_env.check_no_leak()
out = args.out or jh_env.out_dir('g0_%s_%s' % (args.source, args.arm))
os.makedirs(out, exist_ok=True)
json.dump(dict(source=args.source, arm=args.arm, loss_val0=v, per_record=per,
               val_files=list(VAL_FILES), mode=cfg.mode), open(os.path.join(out, 'g0.json'), 'w'),
          indent=1)
