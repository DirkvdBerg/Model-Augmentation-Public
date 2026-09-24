"""Decimation 20 kHz -> 5 kHz (TR-022), one definition used by controller, baseline and pipeline.

Positions and the reference: zero-phase Butterworth order 8 at 2 kHz, then every D-th sample.
Forces and logged currents: block mean over each D-sample hold interval (ZOH-consistent, D-087).
Sample k of the output is sample D*k of the 20 kHz record.
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt

from params import telica_params as tp

D = 4
FS5 = tp.FS / D
TS5 = D * tp.TS
AA = butter(8, 2000.0, fs=tp.FS, output='sos')     # TR-022 anti-alias for position-like signals


def dec_pos(x):
    """(T, n) position-like -> (T // D, n): anti-aliased, then point-sampled."""
    n = (len(x) // D) * D
    return sosfiltfilt(AA, np.asarray(x, np.float64), axis=0)[:n:D]


def dec_force(u):
    """(T, n) force/current (ZOH at 20 kHz) -> (T // D, n) block mean per hold interval."""
    u = np.asarray(u, np.float64)
    n = len(u) // D
    return u[:n * D].reshape(n, D, *u.shape[1:]).mean(axis=1)


def decimate_record(d):
    """A load_telica_log_full dict (+ 'u', 'u_fb') -> the same keys at 5 kHz."""
    out = dict(d)
    r, q = dec_pos(d['r']), dec_pos(d['q1'])
    out['r'], out['q1'] = r, q
    out['e'] = r - q
    for k in ('i_fb', 'i_tot', 'u', 'u_fb'):
        if k in d:
            out[k] = dec_force(d[k])
    out['motion_idx'] = int(d['motion_idx'] // D)
    out['fs'] = FS5
    return out
