"""Debug q1 values with full precision."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.dirname(__file__))

import torch
from telica_loader import load_telica_log

PATH = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..',
    'kamtin-data', 'Data Telica', '06 40 mm XL 80 mm YL',
    'train', 'xpos_-60_ypos-40', 'iterETEL.log'
))

u, q1, fs = load_telica_log(PATH, dtype=torch.float64)

print(f'q1 X1 [m]: {q1[:,0].min().item():.6f} to {q1[:,0].max().item():.6f}')
print(f'q1 X2 [m]: {q1[:,1].min().item():.6f} to {q1[:,1].max().item():.6f}')
print(f'q1 Y  [m]: {q1[:,2].min().item():.6f} to {q1[:,2].max().item():.6f}')
print(f'u  X1 [A]: {u[0,:,0].min().item():.4f} to {u[0,:,0].max().item():.4f}')
print(f'q1 first 5 X1 [m]: {q1[:5,0].tolist()}')
