"""SNR of the pilot record TR-T1_r1 relative to the servo error: rms of the noise-free servo error (r - y)
against what the noise adds to it (noisy minus noise-free y). Full band and training band (< 1.6 kHz,
the loader's anti-alias filter, D-232). New noise (D-238) and old noise (D-225)."""
import os
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt
R = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
get = lambda p: loadmat(p, variable_names=['y', 'r_sim'])
F = get(R + '/Thesis-writeup/Data/Coulomb-tanh-and-MSD-noise-free/Training/TR-T1_r1.mat')
sets = {'new (D-238)': R + '/temp/noise-lowfit/gen_test/Coulomb-tanh-and-MSD-lowpass-noise/Training/TR-T1_r1.mat',
        'old (D-225)': R + '/Thesis-writeup/Data/Coulomb-tanh-and-MSD/Training/TR-T1_r1.mat'}
sos = butter(8, 1600.0, fs=20000.0, output='sos')
rms = lambda x: np.sqrt(np.mean(x ** 2, axis=0))
e = F['r_sim'] - F['y']
for name, p in sets.items():
    n = get(p)['y'] - F['y']
    for band, fil in (('full band', lambda x: x), ('< 1.6 kHz', lambda x: sosfiltfilt(sos, x, axis=0))):
        s, v = rms(fil(e)), rms(fil(n))
        print(f'{name:12s} {band:9s} | servo error {np.array2string(s, precision=2)} m | noise '
              f'{np.array2string(v, precision=2)} m | SNR X1 X2 Y {np.array2string(20 * np.log10(s / v), precision=1)} dB')
