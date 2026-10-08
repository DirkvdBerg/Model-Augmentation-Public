"""Audit A3, EXCITATION-VALIDATION section 19 item (3): noise-driven FRF error against the friction distortion.

Section 19 claims "the noise-driven FRF error in the band is about 100x smaller than the friction distortion sigma"
(estimated from the superseded white-force noise). Recomputed here from the thesis FRF records (TF-1 to TF-3,
Y 0.15 / 0.225 / 0.35 m, 4 phase realisations each, both versions), per excited line of B_tr (106 to 297 Hz,
0.5 Hz grid; periodic in 2 s, 6 periods per record, the first dropped as transient):
- friction distortion: noise-free records; per line, Y_r = G F_r + Ys_r over the 4 realisations; least squares
  G = Y F^+ (3 inputs f_sim, 3 outputs y), residual E = Y - G F; sigma_s^2 = sum_r |E_r|^2 / (4 - 3) per output
  (THEORY: residual of a least-squares fit with 3 regressors on 4 samples has 4 - 3 degrees of freedom;
  Pintelon and Schoukens, robust BLA method, the realisation-to-realisation spread is the stochastic nonlinear
  distortion)
- noise-driven: per realisation the period-averaged output of the noisy record minus that of its noise-free twin,
  at the same lines; sigma_n^2 = mean_r |dY_r|^2 per output (this includes the noise-induced change of friction
  switching, which the noisy data carry)
Both are output errors at the same lines under the same inputs, so their ratio is the ratio of the FRF errors
they cause. Also reported: the period-to-period spread of the noise-free records (non-periodic friction behaviour).

Stated before the run: the section 19 claim holds when the median of sigma_s / sigma_n over the lines is at least
10 per output and Y (HEURISTIC: an order of magnitude, the claim says about 100); below 3 anywhere means the noise,
not friction, would set R3's threshold there. Reads only; prints the result.
"""
import os

import numpy as np
from scipy.io import loadmat

ROOT = r"C:\Users\20203253\OneDrive - TU Eindhoven\Graduation Project\Baseline FP model\Baseline-LPV-Augmentation"
DN = os.path.join(ROOT, 'Thesis-writeup', 'Data', 'Coulomb-tanh-and-MSD', 'Test')
DF = os.path.join(ROOT, 'Thesis-writeup', 'Data', 'Coulomb-tanh-and-MSD-noise-free', 'Test')
FS, TP, NPER = 20000.0, 2.0, 6
NP = int(FS * TP)
K = np.round(np.arange(106.0, 297.0 + 1e-9, 0.5) * TP).astype(int)   # excited bins of one 2 s period
OUT = ['X1', 'X2', 'Y']


def spectra(p):
    d = loadmat(p, mat_dtype=True, squeeze_me=True, variable_names=['y', 'f_sim'])
    y, f = np.asarray(d['y'], float), np.asarray(d['f_sim'], float)
    assert y.shape[0] == NP * NPER
    Yp = np.fft.fft(y.reshape(NPER, NP, 3), axis=1)[1:, K, :]         # periods 2..6, excited lines, (5, L, 3)
    Fp = np.fft.fft(f.reshape(NPER, NP, 3), axis=1)[1:, K, :]
    return Yp, Fp


for tf, yop in (('TF-1', 0.15), ('TF-2', 0.225), ('TF-3', 0.35)):
    Yf, Yn, F, per_spread = [], [], [], []
    for r in range(1, 5):
        fn = '%s_r%d.mat' % (tf, r)
        yf, ff = spectra(os.path.join(DF, fn))
        yn, fn_ = spectra(os.path.join(DN, fn))
        assert np.array_equal(ff, fn_), 'multisine differs between versions'
        Yf.append(yf.mean(0)); Yn.append(yn.mean(0)); F.append(ff.mean(0))
        per_spread.append(np.abs(yf - yf.mean(0)).max(0) / np.abs(yf.mean(0)))   # relative period spread
    Yf, Yn, F = np.array(Yf), np.array(Yn), np.array(F)                # (4, L, 3)
    L = len(K)
    sig_s2 = np.zeros((L, 3))
    for k in range(L):
        Fk, Yk = F[:, k, :].T, Yf[:, k, :].T                           # (3 in, 4 real), (3 out, 4 real)
        G = Yk @ np.linalg.pinv(Fk)
        E = Yk - G @ Fk
        sig_s2[k] = (np.abs(E) ** 2).sum(1) / (4 - 3)
    sig_n2 = (np.abs(Yn - Yf) ** 2).mean(0)                            # (L, 3)
    ratio = np.sqrt(sig_s2 / sig_n2)
    med, p10 = np.median(ratio, 0), np.percentile(ratio, 10, axis=0)
    ok = med >= 10.0
    print('%s (Y %.3f m): sigma_s / sigma_n median %s, 10th pct %s, min %s -> %s; noise-free period spread max %.2e' % (
        tf, yop, np.round(med, 1), np.round(p10, 1), np.round(ratio.min(0), 2),
        ', '.join('%s %s' % (c, 'holds' if o else 'DOES NOT HOLD') for c, o in zip(OUT, ok)),
        max(float(np.max(s)) for s in per_spread)))
    for lo, hi in ((106, 180), (180, 240), (240, 297.1)):
        b = (K / TP >= lo) & (K / TP < hi)
        print('    %3d to %3d Hz: median ratio %s' % (lo, hi, np.round(np.median(ratio[b], 0), 1)))
