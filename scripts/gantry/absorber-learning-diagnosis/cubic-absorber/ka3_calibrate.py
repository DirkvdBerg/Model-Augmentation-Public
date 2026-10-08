"""Truth-side calibration of the cubic absorber spring (data generation only; never used in training or selection).
# HEURISTIC: ka3 such that RMS(ka3 * delta^3) = 0.5 RMS(ka * delta) on the LINEAR truth's absorber stroke delta over the
# 9 training records of the test set (TR-S1..S3, Y1..Y3, P1..P3), the same rule as Gate D. ka from tdg_config.m.
"""
import os
import numpy as np
from scipy.io import loadmat
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'Thesis-writeup', 'Data',
                 'Coulomb-tanh-and-MSD', 'Training')
d0 = loadmat(os.path.join(D, 'TR-S1_r1.mat'), mat_dtype=True)
print([k for k in d0 if not k.startswith('__')])
import re
txt = open(os.path.join(D, '..', '..', '..', 'Code', 'Data', 'tdg_config.m'), encoding='utf-8').read()
mh = float(re.search(r'cfg\.mh\s*=\s*([0-9.eE+-]+)', txt).group(1))
ma = 0.5 * mh; ka = ma * (2 * np.pi * 150) ** 2
recs = ['TR-%s%d_r1' % (c, i) for c in 'SYP' for i in (1, 2, 3)]
dl = np.concatenate([np.ravel(loadmat(os.path.join(D, r + '.mat'), mat_dtype=True)['delta_a']) for r in recs])
ka3 = 0.5 * ka * np.sqrt(np.mean(dl ** 2)) / np.sqrt(np.mean(dl ** 6))
print('mh %.4g ma %.4g ka %.6g | delta_a rms %.4e max %.4e | ka3 %.6g N/m^3 | cubic/linear force at max stroke %.2f' % (
    mh, ma, ka, np.sqrt(np.mean(dl ** 2)), np.abs(dl).max(), ka3, ka3 * np.abs(dl).max() ** 2 / ka))
