"""Tabulate measure_delta JSONs (J at theta*, stencil): rho*, ||Delta*||, bias per parameter.
Usage: scan_summary.py <label>=<json> ...   (the first may be the null control, as reference)"""
__project_origin__ = "added"

import json
import sys

import numpy as np

rows = []
for arg in sys.argv[1:]:
    lab, path = arg.split('=', 1)
    js = json.load(open(path))
    p = next(p for p in js['predictions'] if p['tag'] == 'J @ theta*, Delta* (stencil)')
    rows.append((lab, p, js))
names = rows[0][2]['combo_names']
print('%-8s %8s %9s %8s %9s  %s' % ('set', 'rho*', '||D*||', 'RMS %', 'mdiff own', '  '.join('%8s' % n for n in names)))
for lab, p, js in rows:
    pct = np.asarray(p['pct'])
    print('%-8s %8.4f %9.4f %8.3f %9.3f  %s' % (lab, p['rho'], js['delta_norm']['stencil'],
          float(np.sqrt((pct ** 2).mean())), p['m_diff_own_pct'],
          '  '.join('%+8.3f' % v for v in pct)))
