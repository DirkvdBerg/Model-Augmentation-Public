"""Figure for check C2 (NOISE-INJECTION.md check 2, supervisor's frequency-domain check).

Measured Telica standstill servo-error spectrum (noise/noise_target.mat, from g1.npz) against the
simulated one (friction off, K1, standstill, noise on; mean over the Y_op values of the manifest,
checks/noise_calibration_spectra.mat written by check_noise_calibration.m), as amplitude spectral
density per stage axis, with the calibration band 19.53 to 292.97 Hz marked. Writes
checks/fig_noise_calibration.png.
Run from the repo root:
  conda run -n GraduationProject python Thesis-writeup/Code/Data/checks/plot_noise_calibration.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat

here = os.path.dirname(os.path.abspath(__file__))
repo = os.path.abspath(os.path.join(here, '..', '..', '..', '..'))
T = loadmat(os.path.join(repo, 'Thesis-writeup', 'Code', 'Data', 'noise', 'noise_target.mat'))
S = loadmat(os.path.join(here, 'noise_calibration_spectra.mat'))
f_m, P_m = T['f'].ravel(), T['P']
f_s, P_s = S['fsim'].ravel(), S['Psim']
lo, hi = float(T['band_lo'].squeeze()), float(T['band_hi'].squeeze())
labels = ['X1', 'X2', 'Y']
fig, ax = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True)
for i in range(3):
    ax[i].loglog(f_m[1:], np.sqrt(np.real(P_m[1:, i, i])), 'k', lw=1.2, label='measured (Telica, 145 standstill logs)')
    ax[i].loglog(f_s[1:], np.sqrt(P_s[1:, i]), color='tab:red', lw=1.0, label='simulated (friction off, K1, mean over Y_op)')
    ax[i].axvspan(lo, hi, color='0.9', zorder=0)
    ax[i].set_xlim(5, 2000)
    ax[i].set_ylim(1e-12, 2e-9)
    ax[i].grid(True, which='both', lw=0.3)
    ax[i].set_title(f'{labels[i]} servo error')
    ax[i].set_xlabel('frequency [Hz]')
ax[0].set_ylabel(r'ASD [m/$\sqrt{\mathrm{Hz}}$]')
ax[0].legend(loc='lower left', fontsize=7)
fig.suptitle('Force noise calibrated to the measured error below 300 Hz (grey: calibration band)', fontsize=10)
fig.tight_layout()
out = os.path.join(here, 'fig_noise_calibration.png')
fig.savefig(out, dpi=120)
print('saved', out)
