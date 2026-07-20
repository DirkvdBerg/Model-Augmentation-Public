"""
diag_loader.py
--------------
Quick diagnostic: load one Telica log file with the updated telica_loader and
report shapes, units, value ranges, and basic sanity checks.

Run as:
    conda run -n GraduationProject python scripts/gantry/real-data-verification/diag_loader.py
"""

__project_origin__ = "added"

import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, '..', '..', '..'))
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd
from telica_loader import _clean_header, _find_motion_start, _A_TO_N, _DPI_TO_M, _NATIVE_FS

LOG_PATH = os.path.join(
    _ROOT, 'kamtin-data', 'Data Telica',
    '06 40 mm XL 80 mm YL', 'train', 'xpos_-60_ypos-40', 'iterETEL.log'
)

_BEAM_HEAD = 'BHL'
_AXES      = ('GTRX1', 'GTRX2', 'GTRY')

print('=' * 60)
print('Telica raw-value diagnostic')
print('=' * 60)

# -- 1. Parse header -----------------------------------------------------------
with open(LOG_PATH, 'r') as fh:
    header_line = fh.readline()
cols = _clean_header(header_line)
print(f'\nParsed columns ({len(cols)}):')
for c in cols:
    print(f'  {c}')

# -- 2. Load raw data ----------------------------------------------------------
raw = pd.read_csv(LOG_PATH, sep='\t', header=None, names=cols,
                  skiprows=1, engine='python', index_col=False)
raw = raw.dropna(axis=1, how='all')
N_total = len(raw)
print(f'\nTotal rows: {N_total}')

bh = _BEAM_HEAD
M0   = raw[[f'{bh}_{ax}_M0'   for ax in _AXES]].to_numpy(dtype=np.float64)
M2   = raw[[f'{bh}_{ax}_M2'   for ax in _AXES]].to_numpy(dtype=np.float64)
MF30 = raw[[f'{bh}_{ax}_MF30' for ax in _AXES]].to_numpy(dtype=np.float64)

# -- 3. Raw value ranges (before any conversion) -------------------------------
axes = ['X1 (GTRX1)', 'X2 (GTRX2)', 'Y (GTRY)']
print('\nRaw M0 values (dpi):')
for i, ax in enumerate(axes):
    print(f'  {ax}: min={M0[:,i].min():.2f}  max={M0[:,i].max():.2f}  '
          f'range={M0[:,i].max()-M0[:,i].min():.2f}')

print('\nRaw M2 values (dpi):')
for i, ax in enumerate(axes):
    print(f'  {ax}: min={M2[:,i].min():.6f}  max={M2[:,i].max():.6f}')

print('\nRaw MF30 values (A):')
for i, ax in enumerate(axes):
    print(f'  {ax}: min={MF30[:,i].min():.2f}  max={MF30[:,i].max():.2f}  '
          f'mean={MF30[:,i].mean():.2f}  std={MF30[:,i].std():.2f}')

# -- 4. Motion detection -------------------------------------------------------
motion_idx  = _find_motion_start(M0)
pre_samples = int(round(50.0 * _NATIVE_FS / 1000.0))
trim_start  = max(0, motion_idx - pre_samples)
print(f'\nMotion detection:')
print(f'  motion_idx  = {motion_idx}  ({motion_idx / _NATIVE_FS * 1000:.1f} ms)')
print(f'  pre_samples = {pre_samples}  (50 ms at {_NATIVE_FS:.0f} Hz)')
print(f'  trim_start  = {trim_start}')
print(f'  kept samples: {trim_start} to {N_total}  ({N_total - trim_start} samples)')

# -- 5. M0 first-10 values to see if data is zero-based or absolute ------------
print(f'\nFirst 5 raw M0 rows (dpi):')
for r in range(min(5, N_total)):
    print(f'  [{M0[r,0]:.1f}, {M0[r,1]:.1f}, {M0[r,2]:.1f}]')
print(f'Last 5 raw M0 rows (dpi):')
for r in range(max(0, N_total-5), N_total):
    print(f'  [{M0[r,0]:.1f}, {M0[r,1]:.1f}, {M0[r,2]:.1f}]')

# -- 6. After trim: position and force ranges ----------------------------------
M0_t   = M0[trim_start:]
M2_t   = M2[trim_start:]
MF30_t = MF30[trim_start:]

q1_raw = (M0_t - M2_t) * _DPI_TO_M
u_raw  = MF30_t * _A_TO_N

print(f'\nAfter trim - q1 ranges [mm]:')
for i, ax in enumerate(axes):
    print(f'  {ax}: [{q1_raw[:,i].min()*1e3:.4f}, {q1_raw[:,i].max()*1e3:.4f}] mm'
          f'  range={( q1_raw[:,i].max()-q1_raw[:,i].min())*1e3:.4f} mm')

print(f'\nAfter trim - u (force) ranges [N]:')
for i, ax in enumerate(axes):
    print(f'  {ax}: [{u_raw[:,i].min():.2f}, {u_raw[:,i].max():.2f}] N'
          f'  RMS={np.sqrt(np.mean(u_raw[:,i]**2)):.2f} N')
