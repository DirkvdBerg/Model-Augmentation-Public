"""G5 table rows: the grey box at epoch 0 (FP baseline, zero ANN, linear_map encoder) on the
validation AND test records, through this folder's scorer (proved equal to the grey box's own
selection path in G1(c)). No training.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'adapter'))
import bbcl                                                                # noqa: E402

cfg, CFG, entry = bbcl.load_cfg(local=True)
import numpy as np                                                         # noqa: E402
import torch                                                               # noqa: E402
from gantry_dynamic.data import load_datasets, compute_normalization, VAL_FILES, TEST_FILES  # noqa: E402
from gantry_dynamic.model import build_model                               # noqa: E402
import cl_score                                                            # noqa: E402

np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
data = load_datasets(cfg)
norm = compute_normalization(cfg, data)
np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
gb = build_model(cfg.hp, cfg, data, norm)
res = {}
for split, files, sdl in (('val', VAL_FILES, data.val_list), ('test', TEST_FILES, data.test_list)):
    t0 = time.time()
    s, per, perch, _, _ = cl_score.closed_loop_score(gb, [f[:-4] for f in files], sdl, cfg.ts_new,
                                                     dtype=cfg.dtype_pt)
    res[split] = dict(score=float(s), per_record=dict(zip(files, [float(v) for v in per])),
                      per_channel={f: [float(v) for v in c] for f, c in zip(files, perch)},
                      seconds=time.time() - t0)
    print('[G5] FP baseline (grey box epoch 0) %s closed-loop sim-RMS %.6e m  (%s)  %.0f s'
          % (split, s, ', '.join('%s %.3e' % (f[:-4], v) for f, v in zip(files, per)), time.time() - t0))
os.makedirs(os.path.join(bbcl.OUT, 'g5'), exist_ok=True)
json.dump(res, open(os.path.join(bbcl.OUT, 'g5', 'baseline.json'), 'w'), indent=1)
