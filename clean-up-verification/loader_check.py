"""Clean's loader against the thesis time trace (D-284 amendment 9): record I1, Y axis, the loaded y, r = y - e and
e = y - r of Clean/gantry_dynamic/data.py against r_Y, y_Y and y_Y - r_Y of
Thesis-writeup/Results/24-10-2026/CSV/res_time_trace.csv (written by make_results.time_trace with '%.6e', from
k0 = 30, t = k0 / 4 kHz on).

The bound is the CSV's printed precision, derived per sample: a value printed with 7 significant digits is within
half a unit of its 7th digit, u(v) = 0.5 * 10^(floor(log10|v|) - 6), so |y - y_Y| <= u(y_Y), |r - r_Y| <= u(r_Y)
and |e - (y_Y - r_Y)| <= u(y_Y) + u(r_Y). Also reported: the same comparison shifted by one sample either way
(alignment), the largest excess at the record edges and in the interior, and the noise-free twin's e (noise
version).

    python -u clean-up-verification/loader_check.py
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, 'Clean'))

from gantry_dynamic.config import RunConfig, PRESETS             # noqa: E402
from gantry_dynamic.data import _load_record                      # noqa: E402

CSV = os.path.join(REPO, 'Thesis-writeup', 'Results', '24-10-2026', 'CSV', 'res_time_trace.csv')
RECORD = 'Test/I1_r1.mat'
K0 = 30


def half_unit(v):
    """Half a unit of the 7th significant digit of each printed value ('%.6e'); 0 where the value is 0."""
    v = np.abs(v)
    out = np.zeros_like(v)
    nz = v > 0
    out[nz] = 0.5 * 10.0 ** (np.floor(np.log10(v[nz])) - 6)
    return out


with open(CSV, newline='') as fh:
    rows = list(csv.DictReader(fh))
t_csv = np.array([float(r['t_s']) for r in rows])
r_csv = np.array([float(r['r_Y']) for r in rows])
y_csv = np.array([float(r['y_Y']) for r in rows])
assert rows[0]['t_s'] == '%.6f' % (K0 / RunConfig.fs_new), 'the trace must start at k0 = %d' % K0

cfg = RunConfig(**PRESETS['R1'], noise='lowpass', device='cpu', compile_mode=None)
sd, e = _load_record(RECORD, cfg)
y = np.asarray(sd.y)[:, 2]
e = e[:, 2]
r = y - e
_, e_nf = _load_record('Coulomb-tanh-and-MSD-noise-free/' + RECORD, cfg, folder=cfg.data_root)
e_nf = e_nf[:, 2]
n = len(rows)
print('record %s: %d samples loaded, %d in the trace (from k0 = %d); loader ends at %d' % (RECORD, len(y), n, K0,
                                                                                           K0 + n))
bound = dict(y=half_unit(y_csv), r=half_unit(r_csv))
bound['e'] = bound['y'] + bound['r']
ok_all = True
for shift in (0, -1, 1):
    sl = slice(K0 + shift, K0 + shift + n)
    if sl.start < 0 or sl.stop > len(y):
        continue
    d = dict(y=y[sl] - y_csv, r=r[sl] - r_csv, e=e[sl] - (y_csv - r_csv))
    excess = {k: np.abs(v) - bound[k] for k, v in d.items()}
    within = {k: bool((x <= 0).all()) for k, x in excess.items()}
    print('shift %+d: max |diff| y %.3e r %.3e e %.3e m; within the printed precision: %s'
          % (shift, *(np.abs(d[k]).max() for k in ('y', 'r', 'e')), within))
    if shift == 0:
        ok_all = all(within.values())
        k = int(np.argmax(np.abs(d['e'])))
        print('  largest |e diff| at trace row %d (t = %.6f s, |y_Y| = %.3e m, bound there %.3e m)'
              % (k, t_csv[k], abs(y_csv[k]), bound['e'][k]))
        edge = np.r_[np.arange(100), np.arange(n - 100, n)]
        print('  largest excess over the bound: first/last 100 samples %.3e m, interior %.3e m'
              % (excess['e'][edge].max(), np.delete(excess['e'], edge).max()))
d_nf = np.abs(e_nf[K0:K0 + n] - (y_csv - r_csv))
print('noise-free twin e against the trace: max |diff| %.3e m (the noisy record: %.3e m)'
      % (d_nf.max(), np.abs(e[K0:K0 + n] - (y_csv - r_csv)).max()))
print('LOADER_CHECK %s' % ('PASS' if ok_all else 'FAIL'))
sys.exit(0 if ok_all else 1)
