"""D-225 figure for the supervisor: simulated standstill servo-error spectrum against the measured one.

Reads outputs/noise_shape_Y0.mat (check_encoder_noise.m part B: standstill at Y 0, friction on,
per-axis encoder noise shape-corrected above the corner) and draws one PNG, one panel per stage axis:
the Welch spectrum of the simulated measured error (Hann 2048, 50 %, 20 kHz, as the Telica spectrum
G1) over the measured Telica standstill spectrum, as amplitude spectral density. The region below
the correction corner is shaded (the stated limit).

Run: conda run -n GraduationProject python plot_noise_shape.py
"""
__project_origin__ = "added"

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.signal import welch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'outputs')
MEAS = '#000000'
SIM = '#2a78d6'


def main():
    m = loadmat(os.path.join(OUT, 'noise_shape_Y0.mat'), squeeze_me=True)
    ek, fs, fm, pm, fc = m['ek'], float(m['fs']), m['f_meas'], m['phi_meas'], float(m['fc'])
    f, P = welch(ek, fs=fs, window='hann', nperseg=2048, noverlap=1024, axis=0)
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 8.2), sharex=True)
    for j, (ax, name) in enumerate(zip(axes, ['X1', 'X2', 'Y'])):
        ax.axvspan(9.0, fc, color='#ececec', lw=0, zorder=0)
        ax.loglog(fm, 1e9 * np.sqrt(pm[:, j]), color=MEAS, lw=1.4, label='measured (Telica standstill)', zorder=2)
        ax.loglog(f[1:], 1e9 * np.sqrt(P[1:, j]), color=SIM, lw=1.0, label='simulated (encoder noise, friction on)',
                  zorder=3)
        ax.set_xlim(9.0, 1e4)
        ax.set_ylabel(f'{name}  [nm/sqrt(Hz)]')
        ax.grid(True, which='major', color='#d9d9d9', lw=0.5)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc='upper center', bbox_to_anchor=(0.5, 0.955), ncol=2, frameon=False, fontsize=9)
    axes[-1].set_xlabel(f'frequency [Hz]   (grey: below the {fc:g} Hz correction corner)')
    fig.suptitle('Standstill servo error at Y 0: simulated vs measured', fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    p = os.path.join(OUT, 'noise_shape_Y0.png')
    fig.savefig(p, dpi=150)
    print('saved', p)


if __name__ == '__main__':
    main()
