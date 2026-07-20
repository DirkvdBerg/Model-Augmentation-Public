"""Inspect M0 and M2 raw values to determine which is actual position."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import pandas as pd

PATH = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..',
    'kamtin-data', 'Data Telica', '06 40 mm XL 80 mm YL',
    'train', 'xpos_-60_ypos-40', 'iterETEL.log'
))

# Read raw header
with open(PATH, 'r') as fh:
    header_line = fh.readline()

# Clean header (strip :channel suffix, replace . with _)
fields = header_line.strip().split('\t')
cols = []
for f in fields:
    f = f.strip()
    if ':' in f:
        f = f.split(':')[0]
    f = f.replace('.', '_')
    cols.append(f)
while cols and cols[-1] == '':
    cols.pop()

raw = pd.read_csv(PATH, sep='\t', header=None, names=cols, skiprows=1, engine='python')
raw = raw.dropna(axis=1, how='all')

print('Columns:', list(raw.columns))
print()
print('M0 X1 raw range:', raw['BHL_GTRX1_M0'].min(), 'to', raw['BHL_GTRX1_M0'].max())
print('M2 X1 raw range:', raw['BHL_GTRX1_M2'].min(), 'to', raw['BHL_GTRX1_M2'].max())
print('MF30 X1 raw range:', raw['BHL_GTRX1_MF30'].min(), 'to', raw['BHL_GTRX1_MF30'].max())
print()
print('M0 X1 first 5:', raw['BHL_GTRX1_M0'].values[:5].tolist())
print('M2 X1 first 5:', raw['BHL_GTRX1_M2'].values[:5].tolist())
print()
print('M0 * 1e-6 range [m]:', raw['BHL_GTRX1_M0'].min()*1e-6, 'to', raw['BHL_GTRX1_M0'].max()*1e-6)
print('M2 * 1e-6 range [m]:', raw['BHL_GTRX1_M2'].min()*1e-6, 'to', raw['BHL_GTRX1_M2'].max()*1e-6)
