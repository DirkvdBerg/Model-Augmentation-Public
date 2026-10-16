"""Edge transient of the anti-alias filter against the 5 ms trim (Section V-B, eq:setup_decim).

Justifies: "5 ms is removed at both record ends, where the filter has its transient" (D-232, D-259 item 1:
a property of the known filter, independent of the noise realisation; the earlier measurement,
scripts/gantry/anti-alias/end_transient.py, was made on the older noisy set).
A zero-phase filter run forward and backward has an edge error that lasts as long as its impulse response.
For the filter B of filter_leak.py (8th-order Butterworth at 1.6 kHz, 20 kHz):
  tau_slow = slowest pole time constant, 1 / min |Re s_p|
  t_1pct   = last time at which |h(t)| of the forward impulse response exceeds 1 % of its peak
Input: none (the filter only). Output: Results/setup_filter_edge.csv; macros SetupFiltTau [ms],
SetupFiltDecay [ms], SetupTrim [ms].
"""
import os

import numpy as np
from scipy.signal import butter, sosfilt

import _setup_common as sc

FS = 20000
# HEURISTIC (D-232, GD/config.py AA_ORDER, AA_FC_HZ): order 8 at 1.6 kHz, as filter_leak.py
SOS = butter(8, 1600.0, fs=FS, output='sos')
TRIM_S = 0.005                          # GD/config.py AA_TRIM_S
# HEURISTIC: 1 % of the peak as "decayed", the threshold of loop_memory.py
DECAY_FRAC = 0.01


def main():
    _, p, _ = butter(8, 2 * np.pi * 1600.0, analog=True, output='zpk')
    tau = 1.0 / np.min(np.abs(p.real))
    imp = np.zeros(int(0.05 * FS)); imp[0] = 1.0
    h = np.abs(sosfilt(SOS, imp))
    t = np.arange(len(h)) / FS
    td = t[np.nonzero(h > DECAY_FRAC * h.max())[0][-1]]
    print('slowest pole tau %.2f ms, 1%% decay %.2f ms, trim %.1f ms' % (1e3 * tau, 1e3 * td, 1e3 * TRIM_S))
    sc.write_csv('filter_edge', ['tau_slow_s', 't_decay_1pct_s', 'trim_s'], [[tau, td, TRIM_S]])
    sc.set_macros(os.path.splitext(os.path.basename(__file__))[0], {
        'SetupFiltTau': ('%.1f' % (1e3 * tau), 'slowest pole time constant of the anti-alias filter [ms]'),
        'SetupFiltDecay': ('%.1f' % (1e3 * td), '1 % decay time of the anti-alias filter impulse response [ms]'),
        'SetupTrim': ('%.0f' % (1e3 * TRIM_S), 'samples removed at each record end [ms]'),
    })


if __name__ == '__main__':
    main()
