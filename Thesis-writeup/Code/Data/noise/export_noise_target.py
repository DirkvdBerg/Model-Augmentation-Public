"""Export the measured Telica standstill error spectrum to MATLAB (NOISE-INJECTION.md section 2).

Source: scripts/gantry/closed-loop-noise/outputs/g1_spectra/g1.npz (145 standstill windows,
pwelch Hann 2048, 50 %, 20 kHz, one-sided spectral matrix P[f, i, j] = E[X_i X_j^*] in m^2/Hz,
stage axes [X1, X2, Y]). Writes noise/noise_target.mat with f (1025), P (1025 x 3 x 3 complex),
the source path and its SHA-256, and the rms of the calibration band as a record.
Run from the repo root:  conda run -n GraduationProject python Thesis-writeup/Code/Data/noise/export_noise_target.py
"""
import hashlib
import os
import numpy as np
from scipy.io import savemat

here = os.path.dirname(os.path.abspath(__file__))
repo = os.path.abspath(os.path.join(here, '..', '..', '..', '..'))
src = os.path.join(repo, 'scripts', 'gantry', 'closed-loop-noise', 'outputs', 'g1_spectra', 'g1.npz')
sha = hashlib.sha256(open(src, 'rb').read()).hexdigest()
d = np.load(src)
f = d['f'].astype(float)
P = d['Ppool'].astype(complex)
df = f[1] - f[0]
diag = np.real(np.einsum('fii->fi', P))
band = (f >= 19.5) & (f <= 300.0)          # NOISE-INJECTION section 3: 19.5 to 300 Hz
rms_band = np.sqrt(diag[band].sum(0) * df)
rms_all = np.sqrt(diag.sum(0) * df)
print('bins', f[band][0], '...', f[band][-1], 'n', band.sum())
print('band rms nm', np.round(rms_band * 1e9, 3), ' total rms nm', np.round(rms_all * 1e9, 3))
savemat(os.path.join(here, 'noise_target.mat'),
        {'f': f[:, None], 'P': P, 'df': df, 'rms_band_m': rms_band[None, :], 'rms_total_m': rms_all[None, :],
         'band_lo': float(f[band][0]), 'band_hi': float(f[band][-1]),
         'source': 'scripts/gantry/closed-loop-noise/outputs/g1_spectra/g1.npz', 'source_sha256': sha},
        do_compression=True)
print('sha256', sha)
