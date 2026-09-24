"""Diagnostic after G2 attempt 3 (no gate role): agreement and size ratio per S element.

    python sensitivity/diag_elements.py <g2 run folder name>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cn_env                                                           # noqa: E402

import numpy as np                                                      # noqa: E402
cn_env.check_no_leak()

RUN = sys.argv[1]
AX = cn_env.AX
z = np.load(os.path.join(cn_env.OUTPUTS, RUN, 'g2.npz'), allow_pickle=True)
fe, S, Sm, tol, coh, sd = z['fe'], z['S_emp'], z['S_mod_e'], z['tol'], z['coh'], z['sd']
for i in range(3):
    m = coh[:, i] >= 0.9
    for j in range(3):
        ok = np.abs(S[m, i, j] - Sm[m, i, j]) <= tol[m, i, j]
        r = np.abs(S[m, i, j]) / np.maximum(np.abs(Sm[m, i, j]), 1e-15)
        print(f'[diag] S_{AX[i]}{AX[j]}: agree {ok.mean() * 100:5.1f} % of {m.sum()} coherent bins; '
              f'|emp|/|model| median {np.median(r):8.3f} (10-90 % {np.percentile(r, 10):.3f}-'
              f'{np.percentile(r, 90):.3f}); median |model| {np.median(np.abs(Sm[m, i, j])):.2e}, '
              f'|emp| {np.median(np.abs(S[m, i, j])):.2e}, tol {np.median(tol[m, i, j]):.2e}')
