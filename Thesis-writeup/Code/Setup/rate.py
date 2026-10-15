"""Phase margin of the loop with K1 discretised at 20, 4 and 2 kHz (Section V-B, model rate).

Justifies: "4 kHz is the lowest tested rate at which the loop with K1 re-discretised at T_s loses less
than 5 degrees of phase margin" (D-166 gate, recomputed here on the simulated system).
Method: per stage axis j, the discrete loop gain L_j = G_jj(zoh at T) K1_j(tustin at T) on the
simulated system (absorber on, friction off) at the five training positions; phase margin at the
first crossover. The zero-order hold of the plant is applied as exp(-s T/2) (half-sample delay) on the
continuous frequency response, which is exact in phase for a ZOH sampled loop below Nyquist only up to
the aliasing of G; G rolls off as 1/f^2, so the error is small below the crossover.
Input: meta.truth. Output: Results/setup_rate.csv; macros SetupPmLossFour, SetupPmLossTwo [deg],
SetupPmGate [deg].
"""
import os

import numpy as np
from scipy.signal import cont2discrete, freqz

import _setup_common as sc

TRAIN_Y = (-0.30, -0.15, 0.0, 0.15, 0.30)
RATES = (20000, 4000, 2000)
GATE_DEG = 5.0          # HEURISTIC: D-166 phase-margin gate (decision, not literature)


def margin(T, Y, fs, kappa, num, den):
    f = np.linspace(20.0, 400.0, 19001)
    G = sc.frf_stage(T, Y, f, truth=True)
    # THEORY: ZOH frequency response exp(-j w T/2) sinc(w T/2) (textbook, e.g. Franklin, Powell, Workman,
    # Digital Control; section not checked); the sinc magnitude factor is kept as well
    w = 2 * np.pi * f
    zoh = np.exp(-1j * w / (2 * fs)) * np.sinc(f / fs)
    pm = []
    for j in range(3):
        b, a, _ = cont2discrete((kappa[j] * num, den), 1.0 / fs, method='bilinear')
        _, Kd = freqz(np.ravel(b), np.ravel(a), worN=f, fs=fs)
        L = G[:, j, j] * zoh * Kd
        i = np.argmax(np.abs(L) < 1.0)
        pm.append(180.0 + np.degrees(np.angle(L[i])))
    return np.array(pm)


def main():
    ds = sc.dataset_arg()
    T = sc.truth_params(ds)
    num, den, kappa = sc.k1_continuous(T)
    rows, pm = [], {}
    for fs in RATES:
        pm[fs] = np.array([margin(T, Y, fs, kappa, num, den) for Y in TRAIN_Y])   # (5, 3)
        for iy, Y in enumerate(TRAIN_Y):
            for j, a in enumerate(sc.AXES):
                rows.append([fs, Y, a, pm[fs][iy, j]])
        print('%5d Hz  PM [deg] min %.1f max %.1f' % (fs, pm[fs].min(), pm[fs].max()))
    loss = {fs: (pm[20000] - pm[fs]).max() for fs in RATES}
    print('largest PM loss vs 20 kHz [deg]:', {k: round(v, 2) for k, v in loss.items()})
    sc.write_csv('rate', ['fs_Hz', 'Y', 'axis', 'phase_margin_deg'], rows)
    sc.set_macros(os.path.splitext(os.path.basename(__file__))[0], {
        'SetupPmLossFour': ('%.1f' % loss[4000], 'largest phase-margin loss at 4 kHz vs 20 kHz [deg], 5 training Y, 3 axes'),
        'SetupPmLossTwo': ('%.1f' % loss[2000], 'largest phase-margin loss at 2 kHz vs 20 kHz [deg]'),
        'SetupPmGate': ('%.0f' % GATE_DEG, 'phase-margin gate [deg], D-166 HEURISTIC'),
    })


if __name__ == '__main__':
    main()
