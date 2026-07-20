import os, sys
import pandas as pd
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..', '..'))
sys.path.insert(0, _HERE)

from telica_loader import _clean_header

path = os.path.join(_ROOT, 'kamtin-data', 'Data Telica',
    '06 40 mm XL 80 mm YL', 'train', 'xpos_-60_ypos-40', 'iterETEL.log')

# -- show raw text columns ---------------------------------------------------
with open(path, 'r') as f:
    header = f.readline()
    row1 = f.readline()

h  = header.strip().split('\t')
d1 = row1.strip().split('\t')
print(f'Header cols: {len(h)},  Data row1 cols: {len(d1)}')
print()
for i in range(min(len(h), len(d1))):
    print(f'  col {i:2d}: {h[i]:<38s}  {d1[i]}')

# -- check pandas column selection -------------------------------------------
print()
cols = _clean_header(header)
print(f'_clean_header produced {len(cols)} names')
print('First 13:', cols[:13])

raw = pd.read_csv(path, sep='\t', header=None, names=cols, skiprows=1, engine='python')
raw = raw.dropna(axis=1, how='all')
print(f'\nPandas columns ({len(raw.columns)}):')
print(list(raw.columns))

print('\nFirst row of key columns:')
for c in ['BHL_GTRX1_M0', 'BHL_GTRX1_M2', 'BHL_GTRX1_MF230', 'BHL_GTRX1_MF30',
          'BHL_GTRY_M0', 'BHL_GTRY_MF30']:
    if c in raw.columns:
        print(f'  {c}: {raw[c].iloc[0]:.6f}')
    else:
        print(f'  {c}: NOT FOUND in columns')
