"""J8: confirmation of the candidate training band (EXCITATION-VALIDATION.md s16, criterion T2 of s15b).

B11 measures the Coulomb + MSD truth with the candidate spectrum itself (106 to 297 Hz, 0.5 Hz grid,
A_prod, K1, Y = 0, M = 10); the BLA depends on the input spectrum (lectures 5SMB0 L13). Compared with
B10 (same point, broadband 1 Hz to 1 kHz) on the shared lines, stage coordinates.
Pass: the dip (Y <- F_Y), the cross-entry notch (X1 <- F_Y) and the closed-loop peak (Y <- F_Y) of B11
lie inside the band. Reported beside it: the change B11 - B10 against 2.45 sigma (L8 slide 47) and
the median relative change per entry.
"""
__project_origin__ = "added"

import os

import numpy as np
from scipy.io import loadmat

import common as C

Z95 = 2.45                                   # THEORY: 95 % bound for a complex-valued FRF (L8 slide 47)
BAND = (106.0, 297.0)
CH = ['X1', 'X2', 'Y']
j2 = np.load(os.path.join(C.OUT, 'j2_bla.npz'))
fa, fb = j2['fl_K1_Y+0.00'][:, 1], j2['fl_K1_B11'][:, 1]
ia = np.searchsorted(fa, fb)
assert np.allclose(fa[ia], fb), 'line grids differ'
for c in range(3):
    assert np.allclose(j2['fl_K1_Y+0.00'][ia, c], j2['fl_K1_B11'][:, c]), c
A = j2['PS_K1_Y+0.00'][ia] @ C.P             # B10, stage inputs
B = j2['PS_K1_B11'] @ C.P                    # B11, stage inputs
sA = np.real(j2['vt_K1_Y+0.00'][ia]) @ (C.P ** 2)
sB = np.real(j2['vt_K1_B11']) @ (C.P ** 2)
f = fb


def features(PS):
    a = np.abs(PS[:, 2, 2])
    kd = np.argmin(a)
    kp = kd + np.argmax(a[kd:])
    return dict(dip=float(f[kd]), peak=float(f[kp]), notch=float(f[np.argmin(np.abs(PS[:, 0, 2]))]))


fA, fB = features(A), features(B)
print('lines compared: %d, %.1f to %.1f Hz' % (len(f), f[0], f[-1]))
for k in ('dip', 'notch', 'peak'):
    print('  %-5s B10 %.1f Hz, B11 %.1f Hz' % (k, fA[k], fB[k]))
inside = all(BAND[0] <= v <= BAND[1] for v in fB.values())
edge = min(min(v - BAND[0], BAND[1] - v) for v in fB.values())
print('[T2 confirmation] features of B11 inside %s Hz: %s (closest to an edge: %.1f Hz)' % (BAND, 'PASS' if inside else 'FAIL', edge))
D = B - A
res = np.abs(D) > Z95 * np.sqrt(sA + sB)
rel = np.abs(D) / np.abs(A)
print('change B11 - B10 per entry (out <- in): median |change| / |B10|, share of lines resolved (> 2.45 sigma)')
for o in range(3):
    print('  ' + '   '.join('%s<-F_%s %.3f %3.0f %%' % (CH[o], CH[i], np.median(rel[:, o, i]), 100 * res[:, o, i].mean())
                           for i in range(3)))
for tag, op in (('B10', 'B10'), ('B11', 'B11')):
    d = loadmat(os.path.join(C.OUT, 'j2_bla', op + '.mat'), simplify_cells=True)
    print('  %s stick fraction %s, rms velocity %s m/s' % (tag, np.array2string(np.asarray(d['stick']).mean(0), precision=3),
                                                          np.array2string(np.asarray(d['vrms']).mean(0), precision=4)))
