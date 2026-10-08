"""Truth-side sanity check of the cubic-absorber test set (data generation only): the absorber stroke differs from the
linear set's same record, the cubic force share on it, and the measured output difference."""
import os
import numpy as np
from scipy.io import loadmat
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'Thesis-writeup', 'Data')
ka = 5.05 * (2 * np.pi * 150) ** 2; ka3 = 2.41021e14
for f in ('Training/TR-S1_r1', 'Training/TR-Y1_r1', 'Training/TR-P1_r1', 'Validation/VA-P1_r1'):
    try:
        c = loadmat(os.path.join(R, 'Cubic-absorber-test', 'Coulomb-tanh-and-MSD', f + '.mat'), mat_dtype=True)
    except FileNotFoundError:
        continue
    l = loadmat(os.path.join(R, 'Coulomb-tanh-and-MSD', f + '.mat'), mat_dtype=True)
    dc, dl = np.ravel(c['delta_a']), np.ravel(l['delta_a'])
    yc, yl = np.asarray(c['y_true']), np.asarray(l['y_true'])
    print('%-22s stroke rms lin %.3e cubic %.3e | cubic/linear force rms %.2f (peak %.2f) | y_true diff rms %.2e (y rms %.2e)' % (
        f, np.sqrt(np.mean(dl ** 2)), np.sqrt(np.mean(dc ** 2)), ka3 * np.sqrt(np.mean(dc ** 6)) / (ka * np.sqrt(np.mean(dc ** 2))),
        ka3 * np.abs(dc).max() ** 2 / ka, np.sqrt(np.mean((yc - yl) ** 2)), np.sqrt(np.mean((yl - yl.mean(0)) ** 2))))
