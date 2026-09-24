"""G3 smoke (JH-006/JH-007): ONE optimizer update of one arm through the vendored production path.

    JH_ARM=control|staged python -u runners/smoke.py

Sets JH_FILESET=obc14, JH_SMOKE=1 (CPU, eager, batch 8, n_its=1, no L-BFGS), JH_REF_STRIDE=97 and a
window stride of 400 (memory only) before importing the vendored entry, then does what main() does
up to and including training. Checks: exactly one optimizer update; the staged arm's ten combinations
bit-identical after it and the control's moved (connectivity, D-095); the ANN moved in both.
Writes outputs/<run>/smoke_<arm>.json and the arm's checkpoints under the same folder.
"""
import json
import os
import sys
import time

ARM = os.environ.get('JH_ARM')
assert ARM in ('control', 'staged'), ARM
os.environ['JH_FILESET'] = 'obc14'
os.environ['JH_SMOKE'] = '1'
os.environ.setdefault('JH_REF_STRIDE', '97')
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('JH_STAGE1_JSON', os.path.join(HERE, 'runners', 'stage1.json'))
# deepSI's fit() writes its _best/_last checkpoints under %LOCALAPPDATA%/deepSI/checkpoints on
# Windows (deepSI/datasets/dataset_utils.py::get_work_dirs). Point it inside this run's folder so
# the smoke writes nothing outside scripts/gantry/joint-estimation-horizon/ (handoff section 0).
_LAD = os.path.join(HERE, 'outputs', 'g3_smoke_%s' % ARM, 'localappdata')
os.makedirs(_LAD, exist_ok=True)
os.environ['LOCALAPPDATA'] = _LAD
sys.path.insert(0, HERE)
import jh_env  # noqa: E402
import jh_model as jm  # noqa: E402
from dataclasses import replace  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

entry = jm.load_entry()
from gantry_dynamic.data import load_datasets, compute_normalization, TRAIN_FILES, VAL_FILES  # noqa: E402
from gantry_dynamic.model import build_model  # noqa: E402
from gantry_dynamic.controller import build_closed_loop  # noqa: E402
from gantry_dynamic.training import train_model_with_diagnostics, resolve_resume_hp  # noqa: E402

OUT = jh_env.out_dir('g3_smoke_%s' % ARM)
cfg = replace(entry.CFG, stride=400)          # memory only (JH-007 smoke)
print('[smoke] JH_ARM=%s mode=%s joint=%s obc=%s detune=%s n_its=%s batch=%d device=%s lbfgs=%s '
      'stride=%d files %d/%d' % (ARM, cfg.mode, cfg.joint_estimation, cfg.obc, cfg.combo_init_detune,
                                 cfg.n_its, cfg.batch_size, cfg.device, cfg.lbfgs, cfg.stride,
                                 len(TRAIN_FILES), len(VAL_FILES)))
assert cfg.n_its == 1 and not cfg.lbfgs and cfg.device == 'cpu' and cfg.obc
t0 = time.time()
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
data = load_datasets(cfg)
norm = compute_normalization(cfg, data)
hp = resolve_resume_hp(cfg, cfg.hp, None)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
fit_sys = build_model(hp, cfg, data, norm)
frozen = entry.jh_apply_arm(fit_sys)
fit_sys.simulator = build_closed_loop(fit_sys, norm, cfg, train_files=TRAIN_FILES,
                                      val_files=VAL_FILES, val_data=data.val_ckpt_data)
blk = fit_sys.hfn.connected_blocks[0]
ann = next(b for b in fit_sys.hfn.connected_blocks if type(b).__name__ == 'Static_ANN_Block')
free_before = blk.free_params.detach().clone()
ann_before = torch.cat([p.detach().flatten() for p in ann.parameters()]).clone()
combos_before = blk.identifiable_combinations()
print('[smoke] built in %.0f s; frozen=%s requires_grad=%s; combos %s'
      % (time.time() - t0, frozen, blk.free_params.requires_grad, combos_before))
jh_env.check_no_leak()
bestfit, _ = train_model_with_diagnostics(fit_sys, hp, cfg, data, norm, resume_ckpt=None,
                                          checkpoint_dir=OUT, run_id='smoke_%s' % ARM)
blk = fit_sys.hfn.connected_blocks[0]
ann = next(b for b in fit_sys.hfn.connected_blocks if type(b).__name__ == 'Static_ANN_Block')
d_free = (blk.free_params.detach() - free_before).abs()
d_ann = (torch.cat([p.detach().flatten() for p in ann.parameters()]) - ann_before).abs()
n_upd = int(getattr(fit_sys, 'batch_counter', -1))
ok_free = bool(d_free.max() == 0) if ARM == 'staged' else bool((d_free > 0).any())
res = dict(arm=ARM, updates=n_upd, bestfit=float(bestfit), d_free_max=float(d_free.max()),
           d_free=d_free.tolist(), d_ann_max=float(d_ann.max()), free_check=ok_free,
           ann_moved=bool(d_ann.max() > 0), combos_before=combos_before,
           combos_after=blk.identifiable_combinations(), seconds=time.time() - t0,
           peak_rss_mb=jm.peak_rss_mb())
print('[smoke] updates=%d  |d free|max=%.3e (%s)  |d ann|max=%.3e  bestfit %.6e  %.0f s'
      % (n_upd, d_free.max(), 'PASS' if ok_free else 'FAIL', d_ann.max(), bestfit, time.time() - t0))
json.dump(res, open(os.path.join(OUT, 'smoke_%s.json' % ARM), 'w'), indent=1)
jh_env.check_no_leak()
