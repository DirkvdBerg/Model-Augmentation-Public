"""Check M0 time trace to understand scan profile and units."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import pandas as pd

PATH = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..',
    'kamtin-data', 'Data Telica', '06 40 mm XL 80 mm YL',
    'train', 'xpos_-60_ypos-40', 'iterETEL.log'
))

with open(PATH, 'r') as fh:
    header_line = fh.readline()

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

N = len(raw)
ts = raw['TimeStamp'].to_numpy(dtype=np.float64)
M0 = raw['BHL_GTRX1_M0'].to_numpy(dtype=np.float64)
M2 = raw['BHL_GTRX1_M2'].to_numpy(dtype=np.float64)
MF30 = raw['BHL_GTRX1_MF30'].to_numpy(dtype=np.float64)

t_span_ms = ts[-1] - ts[0]
print(f'N raw samples: {N}')
print(f'TimeStamp span: {t_span_ms:.0f} ms = {t_span_ms/1000:.1f} s')
print(f'Inferred fs_native: {(N-1)/t_span_ms*1000:.0f} Hz')
print()

# Sample at 100 evenly-spaced indices
idx = np.linspace(0, N-1, 100, dtype=int)
print('Time [ms] | M0 X1 | M2 X1 | M0-M2 | MF30/481.882 [A]')
print('-'*70)
for i in idx[::10]:  # every 10th of 100 = every 10% of data
    print(f'{ts[i]:8.0f}ms | {M0[i]:+7.4f} | {M2[i]:+7.4f} | {M0[i]-M2[i]:+7.4f} | {MF30[i]/481.882:+8.3f} A')

print()
print(f'M0  mean: {np.mean(M0):.4f}, std: {np.std(M0):.4f}')
print(f'M2  mean: {np.mean(M2):.4f}, std: {np.std(M2):.4f}')
print(f'MF30/481 mean: {np.mean(MF30)/481.882:.3f} A, std: {np.std(MF30)/481.882:.3f} A')
print()

# Find where M0 is moving (deviation from M0[0] > 10% of its range)
M0_range = np.max(M0) - np.min(M0)
threshold = 0.1 * M0_range if M0_range > 0 else 0.001
moving = np.abs(M0 - M0[0]) > threshold
first_move = np.argmax(moving) if np.any(moving) else N
print(f'M0 range: {M0_range:.4f}')
print(f'First motion sample (M0 deviation > {threshold:.4f}): {first_move} (t={ts[first_move]:.0f} ms)')
print(f'  M0[{first_move}] = {M0[first_move]:.4f}')
