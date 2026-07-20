"""Check M0 values during standstill to determine correct motion threshold."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from telica_loader import _clean_header

PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', '..',
    'kamtin-data', 'Data Telica', '06 40 mm XL 80 mm YL',
    'train', 'xpos_-60_ypos-40', 'iterETEL.log'
)

with open(PATH, 'r') as fh:
    header_line = fh.readline()
cols = _clean_header(header_line)
raw = pd.read_csv(PATH, sep='\t', header=None, names=cols, skiprows=1, engine='python')
raw = raw.dropna(axis=1, how='all')
m0 = raw['BHL_GTRX1_M0'].to_numpy(dtype=np.float64)

print(f'Total samples N = {len(m0)}')
print(f'First 10 M0 values: {m0[:10]}')
print(f'Diffs first 10: {np.diff(m0[:11])}')
print(f'N samples where M0 == m0[0]: {np.sum(m0 == m0[0])}')
print(f'N unique M0 values: {len(np.unique(m0))}')

for thresh in [0, 1e-12, 1e-9, 1e-6, 1e-3, 0.01, 0.1, 0.5, 1.0]:
    deviations = np.abs(m0 - m0[0]) > thresh
    idx = int(np.argmax(deviations))
    found = bool(deviations[idx])
    print(f'  threshold {thresh:.0e}: motion at sample {idx if found else "NOT FOUND"} '
          f'(M0={m0[idx]:.9f})')
