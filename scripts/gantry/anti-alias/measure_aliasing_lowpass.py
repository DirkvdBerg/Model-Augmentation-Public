"""D-232: how much encoder noise the 20 kHz -> 4 kHz decimation of y folds into the training band.

Per record and axis, with the noise part n = y_noisy - y_noisefree (D-230 twins, identical except the
noise):
  Q1  the true in-band noise: rms of n band-limited below 2 kHz (FFT mask at 20 kHz)
  Q2  rms of n after each decimation to 4 kHz (the aliasing excess is Q2 above Q1)
  Q3  what each filter changes in the SIGNAL: rms of (decimation minus point sampling) of the noise-free
      y, low-passed below 1 kHz, against the noise floor
  Q4  the closed-loop objective floor: the D-227 oracle (the truth at 4 kHz, residual form, true x0 at k0)
      on the noisy record with y_data from each decimation, rms(y_oracle - y_data)
Decimations: 'point' y[::5] (the loader today), 'block' mean of 5 (the loader's rule for u, D-087),
'fir' zero-phase FIR low-pass at 1.6 kHz then [::5], 'cheby' scipy.signal.decimate (order-8 Chebyshev I,
zero phase). u is always the block mean, as in the loader. Criteria in docs/decisions.md D-232.
D-232 amendment (LITERATURE.md): the entries take (y, r); 'butter_err' r[::5] + BW8(y - r)[::5] (order-8
Butterworth, zero phase, 1.6 kHz), 'butter' the same on the full position (control), 'butter_err_18' the
1.8 kHz fallback. Q5 (diagnostic iv) the controller-branch leak: the oracle on the noisy record minus the
oracle on the same record cleaned of its encoder noise (y_true for y, u_total + K1_20kHz v_enc for u).

Run: conda run -n GraduationProject python measure_aliasing.py
"""
__project_origin__ = "added"

import os
import sys

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, decimate, filtfilt, firwin, lfilter, sosfiltfilt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'scripts', 'gantry'))
sys.path.insert(0, REPO)
from gantry_dynamic.config import RunConfig          # noqa: E402
from gantry_dynamic.data import _resample_u          # noqa: E402
from gantry_dynamic.controller import build_cfb_at, controller_ss  # noqa: E402
from common import oracle                            # noqa: E402

DATA = os.path.join(REPO, 'Thesis-writeup', 'Data')
NOISY, FREE = 'Coulomb-tanh-and-MSD-lowpass-noise', 'Coulomb-tanh-and-MSD-noise-free'  # CHANGED: copy of measure_aliasing.py on the D-238 lowpass noise (only change)
RECS = ['TR-S1_r1', 'TR-S3_r1', 'TR-P1_r1', 'TR-T1_r1', 'TR-Y1_r1', 'TR-L1_r1']
FS, D = 20000.0, 5
K0 = 30                                               # the thesis encoder window (lag 29), as check_oracle_pilot
HOLD = int(0.5 * FS)                                  # the start hold: excluded from every rms

cfg = RunConfig()
UP = cfg.hp['up_sample']
FIR = firwin(101, 1600.0, fs=FS)                      # HEURISTIC: 101 taps, Hamming, 0.8 x the 2 kHz Nyquist
# HEURISTIC: order 8, fc = 0.8 x the 2 kHz Nyquist (TR-022's ratio; scipy decimate's 0.8/q)
# THEORY: Butterworth forward-backward magnitude 1/(1 + x^16), x = tan(pi f/fs)/tan(pi fc/fs): DC gain
#   exactly 1, error 1.7e-12 at 300 Hz, 4.4e-4 at 1 kHz (LITERATURE.md Q3)
AA = butter(8, 1600.0, fs=FS, output='sos')
AA18 = butter(8, 1800.0, fs=FS, output='sos')          # the pre-stated fallback (D-232 amendment)


def lowpass_fft(x, fc, fs=FS):
    X = np.fft.rfft(x, axis=0)
    f = np.fft.rfftfreq(x.shape[0], 1.0 / fs)
    X[f >= fc] = 0.0
    return np.fft.irfft(X, n=x.shape[0], axis=0)


DECIM = {
    'point': lambda y, r: y[::D],
    'block': lambda y, r: y[:len(y) // D * D].reshape(-1, D, y.shape[1]).mean(axis=1),
    'fir':   lambda y, r: filtfilt(FIR, [1.0], y, axis=0)[::D],
    'cheby': lambda y, r: decimate(y, D, n=8, ftype='iir', axis=0, zero_phase=True),
    # THEORY: Gustafsson 1996 Sects. 5-6 (offsets and trends drive forward-backward edge transients), so the
    # filter acts on the servo error y - r (Heertjes 2024 Def. 2.1)
    'butter_err':    lambda y, r: r[::D] + sosfiltfilt(AA, y - r, axis=0)[::D],
    'butter':        lambda y, r: sosfiltfilt(AA, y, axis=0)[::D],
    'butter_err_18': lambda y, r: r[::D] + sosfiltfilt(AA18, y - r, axis=0)[::D],
}
LEAK = ('point', 'butter_err', 'butter_err_18')        # (iv); linear in v, so 'butter' equals 'butter_err'


def k1_20khz(e, meta):
    """The data's K1 at 20 kHz (Tustin, zero initial state, as lsim in gtd_run_simulation.m) on e (N, 3)."""
    cfb, _ = build_cfb_at(float(meta.controller.design_Y), 1.0 / FS)
    return float(meta.controller.gain) * np.column_stack([lfilter(b, a, e[:, j]) for j, (b, a) in enumerate(cfb)])


def rms(x):
    return np.sqrt(np.mean(np.asarray(x) ** 2, axis=0))


def load(folder, stem):
    split = 'Training'
    return loadmat(os.path.join(DATA, folder, split, stem + '.mat'), squeeze_me=True, struct_as_record=False)


rows = []
for stem in RECS:
    A, B = load(NOISY, stem), load(FREE, stem)
    r = A['r_sim']
    assert np.array_equal(r, B['r_sim']) and r.shape == A['y'].shape    # the twins differ only in v
    n = A['y'] - B['y']
    q1 = rms(lowpass_fft(n, 2000.0)[HOLD:])
    u = _resample_u(A['u_total'], cfg)
    N = len(u)
    xl = A['x_logical'][::D][:N]
    da = A['delta_a'][::D][:N]
    vda = np.r_[0.0, np.diff(da)] * cfg.fs_new_hz
    vda[0] = vda[1]
    x0 = np.array([xl[K0, 0], xl[K0, 1], xl[K0, 2], da[K0], xl[K0, 3], xl[K0, 4], xl[K0, 5], vda[K0]])
    tp = oracle.truth_from_meta(A['meta'].truth)
    ctrl = controller_ss(float(A['meta'].controller.design_Y), cfg.ts_new, float(A['meta'].controller.gain))
    ref_free = DECIM['point'](B['y'], r)[:N]
    u_clean = _resample_u(A['u_total'] + k1_20khz(A['v_enc'], A['meta']), cfg)   # the force had K1 seen y_true
    floor_ref = None
    print('\n%s   Q1 true in-band noise (< 2 kHz) %s nm' % (stem, np.array2string(1e9 * q1, precision=2)))
    for name, dec in DECIM.items():
        q2 = rms((dec(A['y'], r) - dec(B['y'], r))[:N][HOLD // D:])
        yf = dec(B['y'], r)[:N]
        q3 = rms(lowpass_fft(yf - ref_free, 1000.0, FS / D)[HOLD // D:])   # signal change below 1 kHz
        y_data = dec(A['y'], r)[:N]
        y_or = oracle.oracle_closed_loop(u[K0:], y_data[K0:], x0, tp, ctrl, cfg.ts_new, UP)
        q4 = rms((y_or - y_data[K0:])[HOLD // D:])
        if name == 'point':
            floor_ref = q4
        q5 = np.full(3, np.nan)
        if name in LEAK:
            y_cl = dec(A['y_true'], r)[:N]
            y_or_cl = oracle.oracle_closed_loop(u_clean[K0:], y_cl[K0:], x0, tp, ctrl, cfg.ts_new, UP)
            q5 = rms((y_or - y_or_cl)[HOLD // D:])
        rows.append((stem, name, q1, q2, q3, q4, floor_ref, q5))
        print('  %-13s Q2 %s nm (x%s of Q1)  Q3 signal change %s nm  Q4 oracle floor %s nm (x%s of point)'
              '  Q5 leak %s nm'
              % (name, np.array2string(1e9 * q2, precision=2), np.array2string(q2 / q1, precision=2),
                 np.array2string(1e9 * q3, precision=2), np.array2string(1e9 * q4, precision=2),
                 np.array2string(q4 / floor_ref, precision=2), np.array2string(1e9 * q5, precision=2)))

print('\n== D-232 criteria per decimation (all records, all axes) ==')
for name in DECIM:
    r = [x for x in rows if x[1] == name]
    c1 = max(float(np.max(np.abs(x[3] / x[2] - 1))) for x in r)
    c2 = max(float(np.max(x[4] / x[2])) for x in r)       # signal change against the true in-band noise Q1
    c3 = max(float(np.max(x[5] / x[6] - 1)) for x in r)
    c4 = max(float(np.max(x[7] / x[2])) for x in r)       # (iv) leak against Q1 (nan if not computed)
    ok = c1 <= 0.10 and c2 <= 0.10 and c3 <= 0.10
    print('  %-13s (i) max |Q2/Q1 - 1| %.3f  (ii) max Q3 / Q1 %.3f  (iii) max Q4/Q4_point - 1 %+.3f'
          '  (iv) max Q5 / Q1 %.3f  -> %s'
          % (name, c1, c2, c3, c4, 'QUALIFIES (i-iii)' if ok else 'no'))
