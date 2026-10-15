"""Lower edge of the multisine band (Section V-B).

Justifies: "The lower edge of the band is the crossover of the loop with K1. Below it ... the motion
profiles excite this range."
(a) Crossover of each axis's loop gain |G_jj K1_j| with the simulated system (absorber on, friction
    off, tdg_truth_lin.m with g = 0) at the five training positions. Uses the simulated system:
    simulation-only design information, stated as such in the text.
(b) From the data: the frequency below which 99 % of the actuator-force power of the training
    records without multisine lies (noise-free twins, so the number is the motion's, not the noise's).
Input: meta.truth and the noise-free training records of class 'telica' (excitation none).
Output: Results/setup_band_edge.csv; macros SetupCrossMin, SetupCrossMax [Hz], SetupMotionFNN [Hz].
"""
import os

import numpy as np

import _setup_common as sc

TRAIN_Y = (-0.30, -0.15, 0.0, 0.15, 0.30)       # the five standstill training positions (tdg_manifest.m)
# HEURISTIC: 99 % of the power as "the band the motion excites"; reported, not used as a threshold
POWER_SHARE = 0.99


def crossover(T, Y, kappa, num, den):
    f = np.linspace(20.0, 400.0, 38001)                          # 0.01 Hz grid
    G = sc.frf_stage(T, Y, f, truth=True)
    s = 2j * np.pi * f
    Kc = np.polyval(num, s) / np.polyval(den, s)
    fc = []
    for j in range(3):
        L = np.abs(G[:, j, j] * kappa[j] * Kc)
        i = np.argmax(L < 1.0)                                   # first crossing from above
        fc.append(f[i])
    return np.array(fc)


def main():
    ds = sc.dataset_arg()
    T = sc.truth_params(ds)
    num, den, kappa = sc.k1_continuous(T)
    rows, allfc = [], []
    for Y in TRAIN_Y:
        fc = crossover(T, Y, kappa, num, den)
        allfc.append(fc)
        for j, a in enumerate(sc.AXES):
            rows.append(['crossover', Y, a, fc[j]])
        print('Y %+.2f  crossover [Hz] %s' % (Y, np.round(fc, 1)))
    allfc = np.array(allfc)

    f99 = []
    for row in sc.manifest(ds, ('training',)):
        if row['excitation'] != 'none':
            continue
        u = sc.load(sc.record_path(sc.NOISE_FREE, row), ['u_total'])['u_total']
        u = u - u.mean(axis=0)
        # THEORY: periodogram of the whole record, |DFT|^2 per axis; cumulative share over frequency
        U = np.abs(np.fft.rfft(u, axis=0)) ** 2
        f = np.fft.rfftfreq(u.shape[0], 1.0 / 20e3)
        cum = np.cumsum(U, axis=0) / U.sum(axis=0)
        fr = [f[np.argmax(cum[:, j] >= POWER_SHARE)] for j in range(3)]
        f99.append(fr)
        for j, a in enumerate(sc.AXES):
            rows.append(['motion_f99', row['file'], a, fr[j]])
        print('%-8s f99 of u [Hz] %s' % (row['file'], np.round(fr, 1)))
    f99 = np.array(f99)
    sc.write_csv('band_edge', ['quantity', 'record_or_Y', 'axis', 'Hz'], rows)
    sc.set_macros(os.path.splitext(os.path.basename(__file__))[0], {
        'SetupCrossMin': ('%.0f' % allfc.min(), 'min K1 crossover over axes and training Y [Hz], simulated system'),
        'SetupCrossMax': ('%.0f' % allfc.max(), 'max K1 crossover [Hz]'),
        'SetupMotionFNN': ('%.0f' % f99.max(), 'max over motion-only training records and axes of the 99 % '
                           'power frequency of u [Hz]'),
    })


if __name__ == '__main__':
    main()
