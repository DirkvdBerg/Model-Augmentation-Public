"""Audit A3, EXCITATION-VALIDATION section 19 items (1) and (2) on the D-230 data (encoder noise, D-225).

The noise is no longer a force, so its force is taken from the data: w_n = u_total(noisy) - u_total(noise-free),
the force the controller applies in reaction to the encoder noise (the twins are identical except the noise;
this replaces the removed noise-force field d_in of the D-218 data). 18 training records of 12 s, per logical
channel, one-sided energy spectral densities summed over the records, averaged per 1.5 Hz cell:
  (2) below 106 Hz: the reference force K1 r against w_n. Method as `scripts/gantry/excitation-closed-loop/
      j9_refcov.py` (the section 17 evidence): the reference acts at the plant input as K1 r.
  (1) in B_tr (106 to 297 Hz): the multisine force f_sim against w_n (force side), and, because the noise now sits
      on the output, the noise-free response y - r against the output noise y_noisy - y_free (output side;
      logical positions q = P^-T y).
Stated before the run (docs/decisions.md, section 19 on the D-230 data): section 19 needs (1) and (2) at least about
20 dB; this script's own heuristic for (2) is a median of at least 30 dB per band (10 dB below the first claim of
40 dB). Cells below 10 dB are listed. Reads only; prints the result.
"""
import os
import sys

import numpy as np
from scipy.io import loadmat

ROOT = r"C:\Users\20203253\OneDrive - TU Eindhoven\Graduation Project\Baseline FP model\Baseline-LPV-Augmentation"
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'gantry', 'excitation-closed-loop'))
import common as C  # noqa: E402  (K1 FRF and P as used for the section 17 evidence)

NOISY = os.path.join(ROOT, 'Thesis-writeup', 'Data', 'Coulomb-tanh-and-MSD', 'Training')
FREE = os.path.join(ROOT, 'Thesis-writeup', 'Data', 'Coulomb-tanh-and-MSD-noise-free', 'Training')
FC = 106.0                     # K1 crossover, lower band edge (EXCITATION-VALIDATION section 16, D-219)
FB = 297.0                     # upper band edge (D-219)
CELL = 1.5                     # FRF line cell, as j9_refcov.py
BANDS_REF = [(1.0, 20.0), (20.0, 50.0), (50.0, FC)]
CH_F, CH_Q = ['sym', 'anti', 'Y'], ['X', 'Theta', 'Y']
PinvT = np.linalg.inv(np.asarray(C.P, float).T)

files = sorted(f for f in os.listdir(NOISY) if f.endswith('.mat'))
assert len(files) == 18, files
esd = {}
for fn in files:
    A = loadmat(os.path.join(NOISY, fn), mat_dtype=True, squeeze_me=True,
                variable_names=['r_sim', 'u_total', 'y', 'f_sim', 'dt'])
    B = loadmat(os.path.join(FREE, fn), mat_dtype=True, squeeze_me=True, variable_names=['r_sim', 'u_total', 'y'])
    r, dt = np.asarray(A['r_sim'], float), float(A['dt'])
    assert np.array_equal(r, np.asarray(B['r_sim'], float)), 'twins differ in r'
    w_n = np.asarray(A['u_total'], float) - np.asarray(B['u_total'], float)        # noise-induced force [N]
    N = r.shape[0]
    fk = np.fft.rfftfreq(N, dt)
    mk = (fk > 0) & (fk < FB + CELL)
    fkm = fk[mk]
    if not esd:
        Kk = C.ctrl_frf(0.0, fkm)
        zk = np.exp(1j * 2 * np.pi * fkm * dt)
        esd = {k: np.zeros((len(fkm), 3)) for k in ('ref', 'wn', 'ms', 'yr', 'yn')}
    spec = lambda x: dt * np.fft.rfft(x, axis=0)[mk]                              # noqa: E731  [unit s]
    # THEORY: r starts and ends at rest, so diff(r) has finite support and X = dt D / (e^{jw} - 1) exactly (j9_refcov.py)
    Xr = dt * np.fft.rfft(np.diff(r, axis=0), n=N, axis=0)[mk] / (zk - 1.0)[:, None]
    parts = {'ref': np.einsum('kij,kj->ki', Kk, Xr) @ C.P.T,                       # logical force of K1 r [N s]
             'wn': spec(w_n) @ C.P.T,                                              # logical noise force
             'ms': spec(np.asarray(A['f_sim'], float)) @ C.P.T,                    # logical multisine force
             'yr': spec((np.asarray(B['y'], float) - r) @ PinvT.T),                # noise-free response y - r
             'yn': spec((np.asarray(A['y'], float) - np.asarray(B['y'], float)) @ PinvT.T)}   # output noise
    for k, X in parts.items():
        esd[k] += 2.0 * np.abs(X) ** 2                                             # THEORY: one-sided ESD
    print('read', fn, flush=True)

lo_edges = np.arange(0.5, FB, CELL)
fcell = lo_edges + CELL / 2


def cells(a, b):
    out = []
    for lo in lo_edges:
        m = (fkm >= lo) & (fkm < lo + CELL)
        with np.errstate(divide='ignore'):
            out.append(10 * np.log10(esd[a][m].mean(0) / esd[b][m].mean(0)))
    return np.array(out)


def report(title, rat, bands, ch, crit):
    print('\n' + title)
    for lo, hi in bands:
        b = (fcell >= lo) & (fcell < hi)
        med, p10, mn = np.median(rat[b], 0), np.percentile(rat[b], 10, axis=0), rat[b].min(0)
        print('%5.1f to %5.1f Hz: median %s, 10th pct %s, min %s  -> %s' % (
            lo, hi, np.round(med, 1), np.round(p10, 1), np.round(mn, 1),
            ', '.join('%s %s' % (c, 'holds' if o else 'DOES NOT HOLD') for c, o in zip(ch, crit(med, mn)))))
    low = np.argwhere(rat < 10.0)
    print('  cells below 10 dB (f [Hz], channel, dB):',
          [(round(fcell[i], 2), ch[j], round(rat[i, j], 1)) for i, j in low if bands[0][0] <= fcell[i] < bands[-1][1]][:40])


report('(2) ESD K1 r / ESD w_n per 1.5 Hz cell [dB], training set (18 records, 216 s); holds: median >= 30 dB',
       cells('ref', 'wn'), BANDS_REF, CH_F, lambda med, mn: med >= 30.0)
report('(1) force side: ESD f_sim / ESD w_n in B_tr [dB]; holds: min >= 20 dB',
       cells('ms', 'wn'), [(FC, FB)], CH_F, lambda med, mn: mn >= 20.0)
report('(1) output side: ESD (y - r) / ESD (y_noisy - y_free) in B_tr [dB], logical positions; holds: min >= 20 dB',
       cells('yr', 'yn'), [(FC, FB)], CH_Q, lambda med, mn: mn >= 20.0)
df = fkm[1] - fkm[0]
print('\nnoise-induced force rms below %.0f Hz per logical channel [N]:' % FC,
      np.round(np.sqrt(esd['wn'][fkm < FC].sum(0) * df / (18 * 12.0)), 4))
print('noise-induced force rms in B_tr [N]:', np.round(np.sqrt(esd['wn'][fkm >= FC].sum(0) * df / (18 * 12.0)), 4),
      '  multisine force rms in B_tr [N]:', np.round(np.sqrt(esd['ms'][fkm >= FC].sum(0) * df / (18 * 12.0)), 2))
