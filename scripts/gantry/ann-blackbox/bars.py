"""Re-derive the bars the standalone ANN is measured against, in this folder (D-136 required this).

All are free runs over the validation record driven only by the recorded force, i.e. exactly the
quantity `sim-RMS` measures. None involves training.

  FLOOR      8-state truth, oracle initial state          nothing can go below this
  BASELINE   6-state FP model, oracle initial state       the bar the user wants beaten. This is
                                                          the "encoder init" case: perfect state,
                                                          no learning, so it isolates what the
                                                          physics alone buys
  FROZEN-LTI 8-state truth linearised at mean Y           what any LINEAR black box tops out at
  MEAN       predict the training-set mean                a model with no dynamics at all

The gap between BASELINE and FROZEN-LTI is the value of Y-scheduling, since deriv6 rebuilds M at
the current Y every step while the frozen model does not.
"""
__project_origin__ = "added"

import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, 'scripts', 'gantry', 'msd-offset'))
import plant
from truthmodel import truth_ct, discretise, free_run

# The augmentation run selects checkpoints on ALL FOUR validation records
# (gantry_dynamic/data.py:160 builds System_data_list(val_list), model.py:221 passes it to fit).
# System_data_list.RMS is a SAMPLE-WEIGHTED MEAN of the per-record RMS (system_data.py:709-713),
# and all four records are the same length, so the reported number is their plain average.
VAL = ['V1_standstill_Yp10', 'V2_aprbs_Ylow', 'V3_ysweep_Yp10', 'V4_lissajous_Ym10']
TRAIN = ['T1_standstill_Ym30', 'T2_standstill_Ym15', 'T3_standstill_Y000', 'T4_standstill_Yp15',
         'T5_standstill_Yp30', 'T6_ysweep_slow', 'T7_ysweep_fast', 'T8_ysweep_xmix',
         'T9_aprbs_30', 'T10_aprbs_60', 'T11_aprbs_100', 'T12_aprbs_yaw',
         'T13_lissajous', 'T14_lissajous_yaw']
FS = 4000


def rms(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


P = plant.P_np
train_y = np.vstack([plant.load_record(n, fs_new=FS)['y'] for n in TRAIN])
train_mean = np.mean(train_y, axis=0)

per_record, N = {}, []
for name in VAL:
    rec = plant.load_record(name, fs_new=FS)
    y, ts, u_log = rec['y'], rec['ts'], rec['u_log']
    N.append(len(y))
    x8 = np.concatenate([rec['x_logical'][0, 0:3], [rec['delta_a'][0]],
                         rec['x_logical'][0, 3:6], [rec['vdelta_a'][0]]])
    x6 = np.concatenate([rec['x_logical'][0, 0:3], rec['x_logical'][0, 3:6]])

    y8 = (P.T @ plant.rollout(plant.deriv8, x8, u_log, ts, n_out=3).T).T   # FLOOR
    y6 = (P.T @ plant.rollout(plant.deriv6, x6, u_log, ts, n_out=3).T).T   # BASELINE, Y-scheduled
    A, B, C = truth_ct(float(np.mean(y[:, 2])))                            # FROZEN-LTI at mean Y
    Ad, Bd = discretise(A, B, ts)
    yl = free_run(Ad, Bd, C, x8, u_log, np.float64)

    per_record[name] = dict(
        floor=rms(y8, y), baseline=rms(y6, y), frozen=rms(yl, y),
        mean_pred=rms(np.broadcast_to(train_mean, y.shape), y),
        baseline_per_channel=[float(v) for v in np.sqrt(np.mean((y6 - y) ** 2, axis=0))],
        y_std=[float(v) for v in np.std(y, axis=0)])
    r = per_record[name]
    print(f"  {name:22s} floor {r['floor']:.3e}   BASELINE {r['baseline']:.3e}"
          f"   frozen-LTI {r['frozen']:.3e}   mean-pred {r['mean_pred']:.3e}")

w = np.array(N, float)
agg = {k: float(np.average([per_record[n][k] for n in VAL], weights=w))
       for k in ('floor', 'baseline', 'frozen', 'mean_pred')}
res = dict(val=VAL, train_records=len(TRAIN), fs=FS, per_record=per_record, aggregate=agg,
           y_scheduling_gain=agg['frozen'] / agg['baseline'])

print(f"\n  sample-weighted mean over V1-V4, i.e. the number deepSI reports as sim-RMS:\n")
print(f"    FLOOR      8-state truth      {agg['floor']:.4e} m")
print(f"    BASELINE   6-state FP model   {agg['baseline']:.4e} m   <- THE BAR")
print(f"    FROZEN-LTI 8-state at mean Y  {agg['frozen']:.4e} m")
print(f"    MEAN       no dynamics        {agg['mean_pred']:.4e} m")
print(f"\n  Y-scheduling is worth {agg['frozen']/agg['baseline']:.1f}x. The frozen-LTI row bounds a")
print(f"  LINEAR model only; the nonlinear ANN is bounded by the floor, {agg['floor']:.3e} m.")

out = os.path.join(HERE, 'results')
os.makedirs(out, exist_ok=True)
json.dump(res, open(os.path.join(out, 'bars.json'), 'w'), indent=1)
print(f"\nwritten: {os.path.join(out, 'bars.json')}")
