"""How much can the ANN win, measured in the CLOSED-LOOP metric it will be selected on?

THE QUESTION
------------
Step 3 puts the closed-loop baseline score at 2.19e-06 m on every validation record. D2 showed
the metric is not blind: perturbing the ANN moves it (1e-3 -> +18 %, 1e-2 -> 5.7x). But that only
shows the metric can see the model get WORSE. It says nothing about whether there is room to get
BETTER, and that is what decides whether closed-loop training is worth building.

The prize available to the augmentation is the gap between the baseline (no MSD) and the ORACLE
(FP + the true hidden MSD), both scored the same way. If the gap is small, no training scheme will
show anything and the problem is the metric or the rate, not the ANN. Open loop the equivalent gap
was a factor 2.35 on Y (encoder-init baseline 2.121e-04 against oracle 9.023e-05, handoff s6).
Closed loop it is unmeasured.

FAIRNESS (D-097, lessons.md)
----------------------------
Both arms run in the SAME numpy harness, same integrator, same rate, same up_sample, same start
sample, same controller, same metric. `plant.deriv6` (baseline, no absorber) and `plant.deriv8`
(truth, with absorber) are a matched pair from one module, so the only difference between the arms
is the absorber. The oracle is NOT given a finer rate or a better integrator than the baseline;
that is the specific unfairness D-097 warns about.

Both are seeded from the TRUE state at K0 (D-087: sample 0's stored qdot is a gradient() artefact).
This is a headroom measurement, not a model comparison, so both get the best possible x0 and the
encoder is out of the picture entirely.

The loop is the residual form used everywhere else:
    u = u_data + Cfb * (y_data - y_model)

Usage: python -u cl_headroom.py
"""
__project_origin__ = "added"

import dataclasses
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
GANTRY = os.path.join(REPO, 'scripts', 'gantry')
for p in (REPO, GANTRY, HERE, os.path.join(GANTRY, 'drift-demo'),
          os.path.join(GANTRY, 'msd-offset')):
    if p not in sys.path:
        sys.path.insert(0, p)

from demo_common import CFG                                               # noqa: E402
from gantry_dynamic.data import load_traj, load_mat_aug, VAL_FILES        # noqa: E402
import plant as PL                                                        # noqa: E402
from loss_variants import controller_ss                                   # noqa: E402
from cl_controller import y_op_for                                        # noqa: E402

CH = ['X1', 'X2', 'Y']
cfg = dataclasses.replace(CFG, seed=0)
TS = cfg.ts_new
UP = cfg.hp['up_sample']
K0 = 17
Pt = PL.P_np.T
t0 = time.time()


def closed_loop_run(deriv, x0, u_data, y_data, ctrl, nx):
    """Residual-form closed loop in physical units. Returns y_model (N,3) [m]."""
    Ac, Bc, Cc, Dc = ctrl
    N = len(u_data)
    x = np.asarray(x0, float).copy()
    xc = np.zeros(Ac.shape[0])
    y_out = np.empty((N, 3))
    h = TS / UP
    for k in range(N):
        y = Pt @ x[:3]                                  # stage positions
        y_out[k] = y
        e = y_data[k] - y                               # residual [m]
        u_fb = Cc @ xc + Dc @ e                         # [N]
        u = u_data[k] + u_fb                            # stage force
        xc = Ac @ xc + Bc @ e
        ul = PL.P_np @ u                                # stage -> logical force
        for _ in range(UP):
            k1 = deriv(x, ul)
            k2 = deriv(x + 0.5 * h * k1, ul)
            k3 = deriv(x + 0.5 * h * k2, ul)
            k4 = deriv(x + h * k3, ul)
            x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y_out


print('=' * 92)
print('CLOSED-LOOP HEADROOM: baseline (no MSD) against oracle (FP + true MSD)')
print('=' * 92)
print('rate %d Hz, up_sample %d, start K0 = %d, same integrator both arms'
      % (cfg.fs_new_hz, UP, K0))

rows = []
for f in VAL_FILES:
    name = f[:-4]
    Y_op = y_op_for(name)
    sd = load_traj(f, cfg)
    _, _, x_log, x_aug = load_mat_aug(f, cfg)
    u_data = sd.u[K0:].astype(np.float64)
    y_data = sd.y[K0:].astype(np.float64)
    ctrl = controller_ss(Y_op, TS)

    # true seeds at K0; deriv6 order [X,Th,Y,dX,dTh,dY], deriv8 [X,Th,Y,da,dX,dTh,dY,vda]
    x0_6 = np.asarray(x_log[K0], float)
    x0_8 = np.array([x_log[K0, 0], x_log[K0, 1], x_log[K0, 2], x_aug[K0, 0],
                     x_log[K0, 3], x_log[K0, 4], x_log[K0, 5], x_aug[K0, 1]], float)

    print('\n%s  Y_op %+.2f  N = %d' % (name, Y_op, len(u_data)), flush=True)
    y_b = closed_loop_run(PL.deriv6, x0_6, u_data, y_data, ctrl, 6)
    y_o = closed_loop_run(PL.deriv8, x0_8, u_data, y_data, ctrl, 8)
    e_b = np.sqrt(np.mean((y_b - y_data) ** 2, axis=0))
    e_o = np.sqrt(np.mean((y_o - y_data) ** 2, axis=0))
    agg_b = float(np.sqrt(np.mean(e_b ** 2)))
    agg_o = float(np.sqrt(np.mean(e_o ** 2)))
    print('    baseline (no MSD) rms [m]  [%.4e %.4e %.4e]   agg %.4e' % (*e_b, agg_b))
    print('    oracle (FP + MSD) rms [m]  [%.4e %.4e %.4e]   agg %.4e' % (*e_o, agg_o))
    print('    headroom factor per ch     [%9.3f %9.3f %9.3f]   agg %9.3f'
          % (*(e_b / np.maximum(e_o, 1e-30)), agg_b / max(agg_o, 1e-30)))
    rows.append((name, e_b, e_o, agg_b, agg_o))

print('\n' + '=' * 92)
print('SUMMARY   headroom = baseline / oracle, closed loop, same harness')
print('=' * 92)
print('%-22s %-12s %-12s %-10s %s' % ('record', 'baseline agg', 'oracle agg', 'factor',
                                      'per-channel factor X1 X2 Y'))
for name, e_b, e_o, agg_b, agg_o in rows:
    print('%-22s %-12.4e %-12.4e %-10.3f %s'
          % (name, agg_b, agg_o, agg_b / max(agg_o, 1e-30),
             ' '.join('%8.3f' % v for v in e_b / np.maximum(e_o, 1e-30))))
best = max(agg_b / max(agg_o, 1e-30) for _, _, _, agg_b, agg_o in rows)
worst = min(agg_b / max(agg_o, 1e-30) for _, _, _, agg_b, agg_o in rows)
print('\nheadroom factor across records: %.3f to %.3f' % (worst, best))
print('Open-loop reference for scale: 2.35x on Y (handoff s6).')
print('[%.0fs]' % (time.time() - t0))
