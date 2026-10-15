"""Anti-alias filter of the y decimation on the lowpass-noise records (Section V-B, eq:setup_decim).

Justifies: "The filter acts on the servo error only, so the noise above the Nyquist frequency of T_s
does not alias, and the noise-free reference passes unchanged" (D-232; FINAL-RUN-FIXES.md check 3: the
D-232 measurement was made on the earlier noisy set, not on these records).
Per training record and axis, with n = y - y0 the noise in the measured output (noisy minus noise-free):
  alias_raw  = RMS of n above 2 kHz (what point sampling [::5] would fold into the band)
  alias_filt = RMS of B*n above 2 kHz (what the D-232 rule still folds)
  distortion = RMS( rule(y0) - y0[::5] ): what the rule changes on the noise-free output
Input: training records and their noise-free twins. Output: Results/setup_filter_leak.csv; macros
SetupAliasRaw, SetupAliasFilt, SetupFiltDist [m, largest over records and axes].
"""
import os

import numpy as np
from scipy.signal import butter, sosfiltfilt

import _setup_common as sc

FS, D = 20000, 5
# HEURISTIC (D-232, GD/config.py AA_ORDER, AA_FC_HZ): order 8 at 1.6 kHz, zero-phase
SOS = butter(8, 1600.0, fs=FS, output='sos')
TRIM = int(round(0.005 * FS))          # 5 ms at each end, as GD/config.py AA_TRIM_S


def above(x, f0=FS / (2 * D)):
    """RMS of the part of x above f0 (ideal FFT high-pass), per column."""
    X = np.fft.rfft(x, axis=0)
    f = np.fft.rfftfreq(x.shape[0], 1.0 / FS)
    X[f <= f0] = 0.0
    return sc.rms(np.fft.irfft(X, n=x.shape[0], axis=0))


def rule(y, r):
    # THEORY: D-232 rule, GD/data.py::_decimate_y: y_dec = r[::D] + BW(y - r)[::D]
    return r[::D] + sosfiltfilt(SOS, y - r, axis=0)[::D]


def main():
    ds = sc.dataset_arg()
    rows, A_raw, A_filt, dist = [], [], [], []
    for row in sc.manifest(ds, ('training',)):
        d = sc.load(sc.record_path(ds, row), ['y', 'r_sim'])
        d0 = sc.load(sc.record_path(sc.NOISE_FREE, row), ['y', 'r_sim'])
        n = (d['y'] - d0['y'])[TRIM:-TRIM]
        a_raw = above(n)
        a_filt = above(sosfiltfilt(SOS, d['y'] - d['r_sim'], axis=0)[TRIM:-TRIM]
                       - sosfiltfilt(SOS, d0['y'] - d0['r_sim'], axis=0)[TRIM:-TRIM])
        k = TRIM // D
        ds0 = (rule(d0['y'], d0['r_sim']) - d0['y'][::D])[k:-k]
        di = sc.rms(ds0)
        A_raw.append(a_raw); A_filt.append(a_filt); dist.append(di)
        for j, a in enumerate(sc.AXES):
            rows.append([row['file'], a, a_raw[j], a_filt[j], di[j]])
        print('%-8s alias raw %s filt %s dist %s' % (row['file'], np.array2string(a_raw, precision=2),
              np.array2string(a_filt, precision=2), np.array2string(di, precision=2)))
    sc.write_csv('filter_leak', ['record', 'axis', 'alias_raw_m', 'alias_filtered_m', 'distortion_noisefree_m'], rows)
    sc.set_macros(os.path.splitext(os.path.basename(__file__))[0], {
        'SetupAliasRaw': (sc.sci(np.max(A_raw)), 'largest RMS of the noise above 2 kHz, unfiltered [m]'),
        'SetupAliasFilt': (sc.sci(np.max(A_filt)), 'largest RMS of the noise above 2 kHz after the filter [m]'),
        'SetupFiltDist': (sc.sci(np.max(dist)), 'largest RMS change of the noise-free output by the rule [m]'),
    })


if __name__ == '__main__':
    main()
