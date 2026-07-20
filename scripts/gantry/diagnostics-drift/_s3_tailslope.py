"""Quick check: is the S3 open-loop 'drift' a bounded IC settle or a genuine ramp?
Compares whole-trajectory slope vs tail-only slope for true/baseline/model."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import drift_common as dc

d = np.load(os.path.join(dc.OUT_DIR, 's3_openloop_multisine_V1_standstill_Yp10.npz'))
t = d['t']; ts = float(d['ts'])
y_ol = d['y_ol']; y_base = d['y_base']
y_model = d['y_model'] if d['y_model'].size else None

def slopes(y, name):
    N = len(y)
    for frac, lbl in [(1.0, 'whole 0-12s'), (0.5, 'last 50%'), (0.2, 'last 20%'), (0.1, 'last 10%')]:
        s0 = int((1 - frac) * N)
        tt, yy = t[s0:], y[s0:]
        sl = np.array([np.polyfit(tt, yy[:, c], 1)[0] for c in range(3)])
        print(f'  {name:14s} {lbl:12s} slope [m/s]  X1={sl[0]:+.2e} X2={sl[1]:+.2e} Y={sl[2]:+.2e}')
    print()

print(f'tau_X={dc.tau_X:.2f}s  tau_Y={dc.tau_Y:.2f}s   (12s = {12/dc.tau_X:.1f} tau_X, {12/dc.tau_Y:.1f} tau_Y)\n')
slopes(y_ol,   'full-truth')
slopes(y_base, 'baseline')
if y_model is not None:
    # align model length to t tail
    M = len(y_model); tm = t[-M:]
    for frac, lbl in [(0.2, 'last 20%'), (0.1, 'last 10%')]:
        s0 = int((1 - frac) * M)
        sl = np.array([np.polyfit(tm[s0:], y_model[s0:, c], 1)[0] for c in range(3)])
        print(f'  {"aug-model":14s} {lbl:12s} slope [m/s]  X1={sl[0]:+.2e} X2={sl[1]:+.2e} Y={sl[2]:+.2e}')

print('\nReading: if tail slope -> 0 (<< whole slope), the "drift" is the bounded IC settle (benign).')
print('If tail slope stays nonzero, it is a genuine ongoing ramp (real drift).')
