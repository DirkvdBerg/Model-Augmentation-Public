"""D-227 oracle check on the noise-free pilot records (stated in docs/decisions.md before the run).

The closed-loop oracle (common/oracle.py `oracle_closed_loop`, truth parameters from each record's
meta.truth) is run on the pipeline-rate record (u block-mean, y point-sampled, as gantry_dynamic
data.py) from the true state at k0, K1 at the training rate, residual form. Pass: rms(y_oracle - y)
< 1 um per axis on every noise-free pilot record.
"""
__project_origin__ = "added"

import os
import sys

import numpy as np
from scipy.io import loadmat

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..')))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', '..')))
from gantry_dynamic.config import RunConfig                 # noqa: E402
from gantry_dynamic.data import _resample_u                 # noqa: E402
from gantry_dynamic.controller import controller_ss         # noqa: E402
from common import oracle                                   # noqa: E402

PILOT = os.path.join(HERE, 'outputs', 'pilot_data', 'Coulomb-and-MSD-noise-free')
RECS = [('Training', 'TR-T1_r1'), ('Training', 'TR-S3_r1'), ('Validation', 'VA-P1_r1')]
cfg = RunConfig()
up = cfg.hp['up_sample']
k0 = max(cfg.hp['na'], cfg.hp['nb']) if 'na' in cfg.hp else 30
D = cfg.d
ok = True
for split, stem in RECS:
    m = loadmat(os.path.join(PILOT, split, stem + '.mat'), squeeze_me=True, struct_as_record=False)
    u = _resample_u(m['u_total'], cfg)
    N = len(u)
    y = m['y'][::D][:N]
    xl = m['x_logical'][::D][:N]
    da = m['delta_a'][::D][:N]
    vda = np.r_[0.0, np.diff(da)] * cfg.fs_new_hz
    vda[0] = vda[1]
    x0 = np.array([xl[k0, 0], xl[k0, 1], xl[k0, 2], da[k0], xl[k0, 3], xl[k0, 4], xl[k0, 5], vda[k0]])
    tp = oracle.truth_from_meta(m['meta'].truth)
    ctrl = controller_ss(float(m['meta'].controller.design_Y), cfg.ts_new, float(m['meta'].controller.gain))
    y_or = oracle.oracle_closed_loop(u[k0:], y[k0:], x0, tp, ctrl, cfg.ts_new, up)
    rms = np.sqrt(np.mean((y_or - y[k0:]) ** 2, axis=0))
    sig = np.sqrt(np.mean((y[k0:] - y[k0:].mean(0)) ** 2, axis=0))
    p = bool(np.all(rms < 1e-6))
    ok = ok and p
    print('%-10s rms(y_oracle - y) %s m   (y spread %s m, up_sample %d, k0 %d)  %s'
          % (stem, np.array2string(rms, precision=3), np.array2string(sig, precision=3), up, k0,
             'PASS' if p else 'FAIL'))
print('oracle check:', 'PASS' if ok else 'FAIL')
