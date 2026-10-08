"""F1: simulated standstill servo error vs measured Telica (D-225 check B data), amplitude spectrum per axis."""
__project_origin__ = "added"

import os

import numpy as np
from scipy.io import loadmat
from scipy.signal import welch

from style import plt, BLUE

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..', '..', '..'))
SRC = os.path.join(REPO, 'scripts', 'gantry', 'coulomb-tanh-gain', 'outputs', 'noise_shape_Y0.mat')
OUT = os.path.join(HERE, '..', 'figures')


def main():
    os.makedirs(OUT, exist_ok=True)
    m = loadmat(SRC, squeeze_me=True)
    ek, fs, fm, pm, fc = m['ek'], float(m['fs']), m['f_meas'], m['phi_meas'], float(m['fc'])
    f, P = welch(ek, fs=fs, window='hann', nperseg=2048, noverlap=1024, axis=0)
    fig, axes = plt.subplots(3, 1, figsize=(5.4, 4.9), sharex=True)
    for j, (ax, name) in enumerate(zip(axes, ['X1', 'X2', 'Y'])):
        ax.axvspan(9.0, fc, color='#ececec', lw=0, zorder=0)
        ax.loglog(fm, np.sqrt(pm[:, j]), color='black', lw=2.0, label='measured', zorder=2)
        ax.loglog(f[1:], np.sqrt(P[1:, j]), color=BLUE, lw=1.3, label='simulated', zorder=3)
        ax.set_xlim(9.0, 1e4)
        ax.text(0.99, 0.9, name, transform=ax.transAxes, ha='right', va='top', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('servo error [m/$\\sqrt{\\mathrm{Hz}}$]')
    axes[0].legend(loc='lower left', ncol=2, frameon=False)
    axes[0].text(20, axes[0].get_ylim()[1], '< 50 Hz', ha='center', va='top', fontsize=12, color='#606060')
    axes[-1].set_xlabel('frequency [Hz]')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'F1_noise_shape.png'), dpi=200)


if __name__ == '__main__':
    main()
