"""E5 check: input spectral matrix of the injected logical multisines (read-only).

Per record: at every excited line k the single-experiment input spectrum X(k) X(k)^H is rank one
by construction (one periodic realisation, three channels on the same lines). Reported: the
per-line rank (numerical), the condition number of the band-summed matrix sum_k X X^H, and the
largest normalised cross-correlation between channels over the band. Also the same for the
union of T1..T5 (five independent realisations at five Y).
"""
import os
import numpy as np
from scipy.io import loadmat
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
DATA = os.path.join(REPO, 'data', 'gantry', 'matlab', 'trajectory', 'augmentation_ma50_z03_b140-230_a6_telica')
Lb = 0.725
P = np.array([[1, 1, 0], [Lb / 2, -Lb / 2, 0], [0, 0, 1]])
NAMES = ['T1_standstill_Ym30', 'T2_standstill_Ym15', 'T3_standstill_Y000', 'T4_standstill_Yp15',
         'T5_standstill_Yp30', 'T7_ysweep_fast', 'T11_aprbs_100']
tot = np.zeros((3, 3), complex)
for n in NAMES:
    d = loadmat(os.path.join(DATA, n + '.mat'), squeeze_me=True)
    f = np.asarray(d['f_sim'], np.float64) @ P.T
    f = f / f.std(0)                                   # unit RMS per channel: shape, not scale
    X = np.fft.rfft(f, axis=0)
    fr = np.fft.rfftfreq(len(f), 1 / float(d['fs']))
    on = (fr >= 140) & (fr <= 230)
    Xb = X[on]
    ranks = [np.linalg.matrix_rank(np.outer(x, x.conj()), tol=1e-9 * np.abs(x).max() ** 2) for x in Xb[:50]]
    S = Xb.T @ Xb.conj()
    C = np.abs(S) / np.sqrt(np.outer(np.diag(S).real, np.diag(S).real))
    ev = np.linalg.eigvalsh(S)
    print('%-20s lines %d  per-line rank max %d  cond(sum_k XX^H) %.3f  max|coh| off-diag %.4f'
          % (n, on.sum(), max(ranks), ev[-1] / ev[0], (C - np.eye(3)).max()))
    if n.startswith('T') and 'standstill' in n:
        tot += S
ev = np.linalg.eigvalsh(tot)
print('union T1..T5          cond(sum) %.3f' % (ev[-1] / ev[0]))
