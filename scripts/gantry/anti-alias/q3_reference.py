"""D-232 amendment: is the (ii) excess of butter_err the filter, or aliasing in the point-sampled reference?

Criterion (ii) scores each decimation against the POINT-SAMPLED noise-free y, which folds any signal content
above 2 kHz into the band. Here the same Q3 (rms of the difference below 1 kHz, against Q1) is scored against an
ALIAS-FREE reference instead: the noise-free y - r band-limited below 2 kHz (FFT brick wall at 20 kHz), sampled
[::5], plus r[::5], i.e. an ideal anti-alias filter on the servo error. If point sampling is the rule that is
off against it and butter_err is not, the (ii) excess is the reference.

The FFT brick wall is exact only on a periodic stretch: over the full record the multisine runs to the last
sample and the wrap jump (2 to 78 um) rings through everything (first attempt, invalid). So both references are
scored on the last 4 multisine periods (2 s each, D-219 0.5 Hz grid, t = 4 to 12 s), where a steady periodic
response has no wrap. The window's periodicity is printed (max |e over its first period - e over its last
period|) and a record counts for the alias-free reference only when it is below 10 nm (HEURISTIC: a wrap jump J
rings about J / (pi sqrt(2 fc T)) = 0.002 J rms above fc = 2 kHz over T = 8 s, so 10 nm gives 0.02 nm).
Also printed: the rms of the noise-free y - r above 2 kHz on the window (the content point sampling folds).

Run: conda run -n GraduationProject python q3_reference.py
"""
__project_origin__ = "added"

import os

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
DATA = os.path.join(REPO, 'Thesis-writeup', 'Data')
NOISY, FREE = 'Coulomb-tanh-and-MSD', 'Coulomb-tanh-and-MSD-noise-free'
RECS = ['TR-S1_r1', 'TR-S3_r1', 'TR-P1_r1', 'TR-T1_r1', 'TR-Y1_r1', 'TR-L1_r1']   # as measure_aliasing.py
FS, D = 20000.0, 5
HOLD = int(0.5 * FS)
PER = int(2.0 * FS)                                    # one multisine period (0.5 Hz line grid, D-219)
NREC = 240000                                          # 12 s at 20 kHz
W0 = NREC - 4 * PER                                    # the window: the last 4 periods, t = 4 to 12 s
PERIODIC_TOL = 10e-9                                   # HEURISTIC, see the docstring
TRIM = int(0.1 * FS / D)                               # HEURISTIC: 0.1 s, far longer than BW8's impulse response
END = int(0.02 * FS / D)                               # the last 20 ms, where the record-end transient sits
AA = butter(8, 1600.0, fs=FS, output='sos')            # as measure_aliasing.py (D-232 amendment)
AA18 = butter(8, 1800.0, fs=FS, output='sos')

DECIM = {
    'point':         lambda y, r: y[::D],
    'butter_err':    lambda y, r: r[::D] + sosfiltfilt(AA, y - r, axis=0)[::D],
    'butter_err_18': lambda y, r: r[::D] + sosfiltfilt(AA18, y - r, axis=0)[::D],
}


def lowpass_fft(x, fc, fs=FS):
    X = np.fft.rfft(x, axis=0)
    f = np.fft.rfftfreq(x.shape[0], 1.0 / fs)
    X[f >= fc] = 0.0
    return np.fft.irfft(X, n=x.shape[0], axis=0)


def rms(x):
    return np.sqrt(np.mean(np.asarray(x) ** 2, axis=0))


def load(folder, stem):
    return loadmat(os.path.join(DATA, folder, 'Training', stem + '.mat'), squeeze_me=True, struct_as_record=False)


def fmt(x):
    return np.array2string(np.asarray(x), precision=3)


worst = {name: [0.0, 0.0] for name in DECIM}
for stem in RECS:
    A, B = load(NOISY, stem), load(FREE, stem)
    r = B['r_sim']
    assert len(r) == NREC and W0 % D == 0
    q1 = rms(lowpass_fft(A['y'] - B['y'], 2000.0)[HOLD:])
    ew = (B['y'] - r)[W0:]
    per = np.max(np.abs(ew[:PER] - ew[-PER:]), axis=0)
    ok = bool(np.all(per < PERIODIC_TOL))
    hi = rms(ew - lowpass_fft(ew, 2000.0))
    ideal = r[W0::D] + lowpass_fft(ew, 2000.0)[::D]
    ref_point = DECIM['point'](B['y'], r)[W0 // D:]
    print('\n%s  Q1 %s nm   window periodicity %s nm (%s)   y - r above 2 kHz %s nm'
          % (stem, fmt(1e9 * q1), fmt(1e9 * per), 'periodic' if ok else 'NOT periodic: alias-free reference invalid',
             fmt(1e9 * hi)))
    for name, dec in DECIM.items():
        yd = dec(B['y'], r)[W0 // D:]
        # interior: the last 0.1 s dropped (the zero-phase filters' record-end transient, reported apart)
        q3_pt = rms(lowpass_fft((yd - ref_point)[:-TRIM], 1000.0, FS / D))
        q3_id = rms(lowpass_fft((yd - ideal)[:-TRIM], 1000.0, FS / D))
        end = np.max(np.abs((yd - ideal)[-END:]), axis=0)
        worst[name][0] = max(worst[name][0], float(np.max(q3_pt / q1)))
        if ok:
            worst[name][1] = max(worst[name][1], float(np.max(q3_id / q1)))
        print('  %-13s interior: vs point-sampled %s nm (%s of Q1)   vs alias-free %s nm (%s of Q1)'
              '   last 20 ms max |vs alias-free| %s nm'
              % (name, fmt(1e9 * q3_pt), fmt(q3_pt / q1), fmt(1e9 * q3_id), fmt(q3_id / q1), fmt(1e9 * end)))

print('\n== (ii) max Q3 / Q1 on the window (criterion 0.10; alias-free: periodic records only) ==')
for name, (a, b) in worst.items():
    print('  %-13s point-sampled reference %.3f   alias-free reference %.3f' % (name, a, b))
