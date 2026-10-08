"""D3 (post-run diagnosis only, after the final checkpoint is fixed): added-state self-map eigenvalues at the 24
encoder points stored in a ps2 history file. Reports, per eval: median dominant |z|, and the least-damped complex pair
per point (frequency, zeta) over ALL frequencies (no band is used); the absorber value (212 Hz, zeta 0.043) is quoted
only in the report for comparison. Usage: python d3_poles.py TAG [TAG ...]
"""
import os
import sys
import numpy as np
FS = 4000.0
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
for tag in sys.argv[1:]:
    h = np.load(os.path.join(OUT, 'ps2_%s_hist.npz' % tag), allow_pickle=True)
    print('######## %s' % tag)
    for it, ph, Ea in zip(h['it'], h['phase'], h['eig_a']):
        if it % 50 and ph != 'S2end':
            continue
        dom = np.abs(Ea).max(1)
        fz = []
        for e in Ea:
            e = e[e.imag > 1e-9]
            if len(e):
                s = np.log(e) * FS
                f = np.abs(s.imag) / (2 * np.pi); z = -s.real / np.abs(s); j = np.argmin(z); fz.append((f[j], z[j]))
        share = len(fz) / len(Ea)
        fz = np.array(fz) if fz else np.full((1, 2), np.nan)
        print('  it %4d %-5s | dom |z| med %.3f | complex pair at %3.0f %% of points, f med %.0f Hz, zeta med %.3f' % (
            it, ph, np.median(dom), 100 * share, np.nanmedian(fz[:, 0]), np.nanmedian(fz[:, 1])))
