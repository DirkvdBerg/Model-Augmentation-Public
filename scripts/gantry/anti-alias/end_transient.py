"""D-232 end trim: how long is butter_err's zero-phase edge transient? Truncation test on every noisy record.

For each record, e = y - r (noisy, 20 kHz) is filtered in full (F) and cut short at several instants c; the cut
record's output is compared with F near the cut, where F has the true future. The distance from the cut beyond
which |difference| stays below 0.01 x Q1 (per axis; Q1 = rms of the noise part below 2 kHz, from the noise-free
twin) is the transient length. End cuts: t = 2, 4, 6, 8, 10, 11.8 s (inside the multisine and moves, as the real
record end). Start cuts: t = 0.25 s (inside the start hold, as the real record start) and the same instants.
Rule (docs/decisions.md, D-232 end trim): trim = the largest such distance, rounded up to whole ms.

Run: conda run -n GraduationProject python end_transient.py
"""
__project_origin__ = "added"

import glob
import os

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
DATA = os.path.join(REPO, 'Thesis-writeup', 'Data')
NOISY, FREE = 'Coulomb-tanh-and-MSD', 'Coulomb-tanh-and-MSD-noise-free'
FS = 20000.0
HOLD = int(0.5 * FS)
AA = butter(8, 1600.0, fs=FS, output='sos')            # butter_err (D-232 amendment)
L = int(0.1 * FS)                                      # distances examined: 0.1 s from the cut
CUTS = [int(t * FS) for t in (2, 4, 6, 8, 10, 11.8)]
CUT_HOLD = int(0.25 * FS)
REL = 0.01                                             # HEURISTIC: 0.01 x Q1, see the docstring


def lowpass_fft(x, fc, fs=FS):
    X = np.fft.rfft(x, axis=0)
    f = np.fft.rfftfreq(x.shape[0], 1.0 / fs)
    X[f >= fc] = 0.0
    return np.fft.irfft(X, n=x.shape[0], axis=0)


def length(dev, thr):
    """Samples from the cut beyond which dev (L, 3), ordered by distance, stays below thr (3,)."""
    above = np.any(dev > thr, axis=1)
    return 0 if not above.any() else int(np.nonzero(above)[0][-1]) + 1


def load(folder, path):
    return loadmat(path.replace(os.sep + NOISY + os.sep, os.sep + folder + os.sep), squeeze_me=True)


files = sorted(f for f in glob.glob(os.path.join(DATA, NOISY, '*', '*.mat'))
               if not os.path.basename(f).startswith('TF-'))
worst_end, worst_start, worst_hold, peak = (0, ''), (0, ''), (0, ''), 0.0
for path in files:
    A, B = load(NOISY, path), load(FREE, path)
    r = A['r_sim']
    thr = REL * np.sqrt(np.mean(lowpass_fft(A['y'] - B['y'], 2000.0)[HOLD:] ** 2, axis=0))
    e = A['y'] - r
    F = sosfiltfilt(AA, e, axis=0)
    n_end, n_start = 0, 0
    for c in CUTS:
        dev = np.abs(sosfiltfilt(AA, e[:c], axis=0)[c - L:][::-1] - F[c - L:c][::-1])   # distance 0 first
        n_end = max(n_end, length(dev, thr))
        peak = max(peak, float(dev.max()))
        dev = np.abs(sosfiltfilt(AA, e[c:], axis=0)[:L] - F[c:c + L])
        n_start = max(n_start, length(dev, thr))
    dev = np.abs(sosfiltfilt(AA, e[CUT_HOLD:], axis=0)[:L] - F[CUT_HOLD:CUT_HOLD + L])
    n_hold = length(dev, thr)
    stem = os.path.basename(path)[:-4]
    print('%-12s thr %s nm   end %5.2f ms   start (moving) %5.2f ms   start (hold) %5.2f ms'
          % (stem, np.array2string(1e9 * thr, precision=3), 1e3 * n_end / FS, 1e3 * n_start / FS,
             1e3 * n_hold / FS))
    worst_end = max(worst_end, (n_end, stem))
    worst_start = max(worst_start, (n_start, stem))
    worst_hold = max(worst_hold, (n_hold, stem))

print('\n== %d records; largest transient length (|difference| > %.2f x Q1) ==' % (len(files), REL))
print('  end cuts            %.2f ms (%s)   peak difference at a cut %.1f nm' % (1e3 * worst_end[0] / FS, worst_end[1], 1e9 * peak))
print('  start cuts, moving  %.2f ms (%s)' % (1e3 * worst_start[0] / FS, worst_start[1]))
print('  start cut, hold     %.2f ms (%s)' % (1e3 * worst_hold[0] / FS, worst_hold[1]))
print('  -> end trim %d ms' % int(np.ceil(1e3 * worst_end[0] / FS)))
