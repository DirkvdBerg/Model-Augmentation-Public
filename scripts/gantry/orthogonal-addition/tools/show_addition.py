"""Print the nonzero terms of addition .mat files (read only). Usage: show_addition.py <mat> ..."""
__project_origin__ = "added"

import sys

import numpy as np
from scipy.io import loadmat

for path in sys.argv[1:]:
    L = loadmat(path)
    print(path)
    for k in ('oa_Ma0', 'oa_Ma1', 'oa_Ca0', 'oa_Ka0', 'oa_Ka1', 'oa_Ka2'):
        M = L[k]
        for i in range(3):
            for j in range(i, 3):
                if M[i, j] != 0:
                    print('  %s[%d,%d] %+.4e' % (k, i + 1, j + 1, M[i, j]))
    m = L['oa_m'].ravel()
    for s in np.nonzero(m)[0]:
        print('  absorber %d: m %.4f kg, f %.2f Hz, b0 %s, b1 %s' % (
            s, m[s], L['oa_w'].ravel()[s] / (2 * np.pi), L['oa_b0'][:, s], L['oa_b1'][:, s]))
