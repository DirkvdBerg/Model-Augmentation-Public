"""G1(c) number 3 (BB-006): the grey box's own closed-loop number from the ORIGINAL repo packages.

Imports nothing from vendor/: the repo's `gantry_interconnect_dynamic.py`, `gantry_dynamic`,
`model_augmentation` and `common` exactly as the grey box's own run imports them. Read-only use
(PYTHONDONTWRITEBYTECODE=1 via tools/run.sh). Local overrides as BB-001: device cpu, eager.
"""
import json
import os
import sys
import time
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
BB = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(BB, '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'scripts', 'gantry'))
sys.path.insert(0, REPO)

import numpy as np                                                         # noqa: E402
import torch                                                               # noqa: E402
torch.set_num_threads(4)
import gantry_interconnect_dynamic as entry                                # noqa: E402
from gantry_dynamic.data import load_datasets, compute_normalization, TRAIN_FILES, VAL_FILES  # noqa: E402
from gantry_dynamic.model import build_model                               # noqa: E402
from gantry_dynamic.controller import build_closed_loop                    # noqa: E402

mods = {n: getattr(sys.modules[n], '__file__', '') for n in ('gantry_dynamic.model',
        'model_augmentation.fit_systems.closed_loop', 'common.oracle')}
assert not any('blackbox-closed-loop' in f for f in mods.values()), mods
print('[g1c-orig] modules:', json.dumps(mods, indent=1))

cfg = replace(entry.CFG, device='cpu', compile_mode=None)
np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
data = load_datasets(cfg)
norm = compute_normalization(cfg, data)
np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
fs = build_model(cfg.hp, cfg, data, norm)
fs.simulator = build_closed_loop(fs, norm, cfg, train_files=TRAIN_FILES, val_files=VAL_FILES,
                                 val_data=data.val_ckpt_data)
t0 = time.time()
v = fs.cal_validation_error(data.val_ckpt_data, validation_measure='sim-RMS')
dt = time.time() - t0
per = [float(x) for x in fs.simulator.last_per_record]
print('[g1c-orig] grey box (untrained, CFG) closed-loop val sim-RMS = %.10e m  (%.1f s)' % (v, dt))
for n, r in zip(VAL_FILES, per):
    print('   %-24s %.6e' % (n, r))
os.makedirs(os.path.join(BB, 'outputs', 'g1'), exist_ok=True)
json.dump(dict(value=float(v), per_record=per, val_files=VAL_FILES, seconds=dt, modules=mods),
          open(os.path.join(BB, 'outputs', 'g1', 'g1c_original.json'), 'w'), indent=1)
