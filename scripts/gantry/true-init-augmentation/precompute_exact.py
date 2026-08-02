"""Cache the exact 8-state truth for every record the training arm touches.

Each record is a 20 kHz RK4 replay of the 8-state plant from the rest IC and takes
a couple of minutes, so it is done once, up front, into
`figures/_exact_<record>_4000.npz`. The replay gate (positions vs the record's own
positions) is printed per record: it is the only thing that certifies the cached
velocities, since nothing in the .mat files can be compared against them directly.

Run:
  PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 conda run --no-capture-output \\
      -n GraduationProject python -u \\
      scripts/gantry/true-init-augmentation/precompute_exact.py
"""
__project_origin__ = "added"

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'scripts', 'gantry'))

from data_exact import exact_truth, fd_velocity_error            # noqa: E402
from gantry_dynamic.data import TRAIN_FILES, VAL_FILES, TEST_FILES  # noqa: E402

OUT = os.path.join(REPO, 'simulations', 'gantry_subnet', 'diagnostics')
NAMES = [f[:-4] for f in TRAIN_FILES + VAL_FILES + TEST_FILES]


def main():
    print(f'Caching the exact 8-state truth for {len(NAMES)} records '
          f'(20 kHz RK4 from the rest IC, decimated to 4 kHz)\n')
    print(f'  {"record":<24}{"N":>8}{"X [m]":>13}{"Theta [rad]":>14}{"Y [m]":>13}'
          f'{"dX rel":>10}{"s":>7}')
    rows = {}
    for n in NAMES:
        t0 = time.time()
        r = exact_truth(n)
        fd = fd_velocity_error(r)
        dt = time.time() - t0
        print(f'  {n:<24}{len(r["x6"]):>8}{r["gate"][0]:>13.4e}{r["gate"][1]:>14.4e}'
              f'{r["gate"][2]:>13.4e}{fd["rel"][0]:>10.2e}{dt:>7.1f}')
        rows[n] = dict(N=int(len(r['x6'])),
                       replay_gate=[float(v) for v in r['gate']],
                       fd_vel_rel=[float(v) for v in fd['rel']],
                       fd_vel_rms=[float(v) for v in fd['rms']])
    worst = max(v['replay_gate'][0] for v in rows.values())
    print(f'\n  worst X replay residual across records: {worst:.4e} m  '
          f'(established figure for this dataset: 5.37e-10 m)')
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, 'true_init_exact_cache.json')
    with open(p, 'w') as f:
        json.dump(dict(records=rows, worst_x=float(worst)), f, indent=2)
    print(f'  wrote {p}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
