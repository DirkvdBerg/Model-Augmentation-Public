"""D-224 band re-check: the band rule of the excitation study, rerun on the tanh truth.

Runs `scripts/gantry/excitation-closed-loop/j2_bla.py` and `j7_band.py` unchanged, with
`common.OUT` and `common.HERE` redirected to outputs/band_tanh/, so they read the B6 to B10 cases
written by bla_tanh.m and nothing in the excitation folder is written. Then compares with the
Karnopp result `excitation-closed-loop/outputs/j7_band.json` (read only).

Pass (stated in D-224 before the run): (1) the plant features stay inside B_tr = 106 to 297 Hz
(dip and X1<-F_Y notch from j7, closed-loop peak and upper -3 dB edge per Y from j2's print), and
(2) the nominal 90 % band edges move by at most 5 Hz from the Karnopp result 106.5 / 297.0 Hz
(HEURISTIC: 10 lines of the 0.5 Hz grid, under 3 % of the band). (1) without (2) is reported, and a
band change is then the user's decision.

Run: conda run -n GraduationProject python band_tanh.py <gain>   (reads and writes outputs/band_tanh_g<gain>/)
"""
__project_origin__ = "added"

import json
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
EX = os.path.join(REPO, 'scripts', 'gantry', 'excitation-closed-loop')
GAIN = sys.argv[1] if len(sys.argv) > 1 else '1000'   # the friction gain whose BLA cases to analyse
OUT = os.path.join(HERE, 'outputs', 'band_tanh_g' + GAIN)
os.makedirs(os.path.join(OUT, 'figures'), exist_ok=True)

sys.path.insert(0, EX)
import common as C  # noqa: E402

C.OUT = OUT
C.HERE = OUT
runpy.run_path(os.path.join(EX, 'j2_bla.py'), run_name='__main__')
runpy.run_path(os.path.join(EX, 'j7_band.py'), run_name='__main__')

with open(os.path.join(EX, 'outputs', 'j7_band.json')) as fh:
    old = json.load(fh)
with open(os.path.join(OUT, 'j7_band.json')) as fh:
    new = json.load(fh)
print('\n==== Karnopp (stored) vs tanh (D-224) ====')
print('f_c: %.1f -> %.1f Hz' % (old['f_c'], new['f_c']))
for k in new['comparisons']:
    for s in new['comparisons'][k]['bands']:
        print('%-8s %s band: %s -> %s Hz' % (k, s, old['comparisons'][k]['bands'][s],
                                              new['comparisons'][k]['bands'][s]))
    for key in new['comparisons'][k]:
        if key.startswith('feature') or key in ('dip', 'notch', 'peak', 'features'):
            print('%-8s %s: %s -> %s' % (k, key, old['comparisons'][k].get(key), new['comparisons'][k][key]))
