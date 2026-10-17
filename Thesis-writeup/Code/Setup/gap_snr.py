"""Excitation between the motion profiles and the multisine band (Section V-B; D-219 amendment, T3 stand-in).

Question: is the servo error between the motion band and the multisine band (24 to 105 Hz) above the
noise? The motion profiles put 99 % of their force below about 24 Hz (band_edge.py, SetupMotionFNN); the
multisine starts at the highest K1 crossover, 106 Hz (SetupCrossMax). Uses the known noise-free twin
(simulation only, D-259 item 1): signal e0 = y0 - r, noise n = y - y0.
Per training record and stage axis:
  P_sig = power of e0 in [F_LO, F_HI] (Welch PSD, integrated)
  P_noi = power of n  in [F_LO, F_HI]
  SNR   = 10 log10(P_sig / P_noi) [dB]
Input: training records and their noise-free twins. Output: Results/setup_gap_snr.csv; macros
SetupGapSnrMin{Motion,Ms}, SetupGapSnrMed{Motion,Ms} [dB] for the records without and with multisine.
"""
import os

import numpy as np
from scipy.signal import welch

import _setup_common as sc

FS = 20000
F_LO, F_HI = 24.0, 105.0              # SetupMotionFNN; just below SetupCrossMax so the 106 Hz multisine line is excluded
NPERSEG = 2 ** 14                     # HEURISTIC: 1.2 Hz resolution, about 29 averaged segments per 12 s record
TRIM = int(round(0.005 * FS))         # 5 ms at each end, as the reduction (filter_edge.py)


def band_power(x):
    f, p = welch(x, fs=FS, nperseg=NPERSEG, axis=0)
    m = (f >= F_LO) & (f <= F_HI)
    # THEORY: power in a band = integral of the one-sided PSD over the band
    return np.trapezoid(p[m], f[m], axis=0)


def main():
    ds = sc.dataset_arg()
    rows, mot, ms = [], [], []
    for row in sc.manifest(ds, ('training',)):
        d = sc.load(sc.record_path(ds, row), ['y', 'r_sim'])
        d0 = sc.load(sc.record_path(sc.NOISE_FREE, row), ['y'])
        e0 = (d0['y'] - d['r_sim'])[TRIM:-TRIM]
        n = (d['y'] - d0['y'])[TRIM:-TRIM]
        snr = 10 * np.log10(band_power(e0) / band_power(n))
        (ms if row['excitation'] == 'multisine' else mot).append(snr)
        for j, a in enumerate(sc.AXES):
            rows.append([row['file'], row['excitation'], a, snr[j]])
        print('%-9s %-10s SNR 24-105 Hz [dB] %s' % (row['file'], row['excitation'], np.array2string(snr, precision=1)))
    mot, ms = np.array(mot), np.array(ms)
    sc.write_csv('gap_snr', ['record', 'excitation', 'axis', 'snr_db'], rows)
    sc.set_macros(os.path.splitext(os.path.basename(__file__))[0], {
        'SetupGapSnrMinMotion': ('%.0f' % mot.min(), 'min SNR 24-105 Hz, training records without multisine [dB]'),
        'SetupGapSnrMedMotion': ('%.0f' % np.median(mot), 'median SNR 24-105 Hz, training records without multisine [dB]'),
        'SetupGapSnrMinMs': ('%.0f' % ms.min(), 'min SNR 24-105 Hz, training records with multisine [dB]'),
        'SetupGapSnrMedMs': ('%.0f' % np.median(ms), 'median SNR 24-105 Hz, training records with multisine [dB]'),
    })


if __name__ == '__main__':
    main()
