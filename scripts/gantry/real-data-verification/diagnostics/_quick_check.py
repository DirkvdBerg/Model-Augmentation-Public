"""Quick Stage 1+2 only check — no simulation."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))

import torch
from telica_loader import load_telica_log

PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', '..',
    'kamtin-data', 'Data Telica', '06 40 mm XL 80 mm YL',
    'train', 'xpos_-60_ypos-40', 'iterETEL.log'
)

print('Loading...')
u, q1, fs = load_telica_log(os.path.normpath(PATH), dtype=torch.float64)
T = q1.shape[0]
print(f'u   shape : {tuple(u.shape)}')
print(f'q1  shape : {tuple(q1.shape)}')
print(f'fs        : {fs} Hz')
print(f'T         : {T} samples = {T/fs:.2f} s at 20 kHz')
print(f'q1  range : {q1.min().item()*1e3:.2f} to {q1.max().item()*1e3:.2f} mm')
print(f'u   range : {u.min().item():.4f} to {u.max().item():.4f} A')
