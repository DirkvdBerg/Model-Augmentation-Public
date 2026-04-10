"""
plot_lpv_vs_frozen.py
---------------------
Figure 1: X1, X2, Y trajectories — ref (MATLAB), Python LPV, Python frozen.
Figure 2: X1, X2, Y absolute errors — |LPV - ref| and |frozen - ref|, log scale.

Run as:
    conda run -n GraduationProject python -m lpv_lfr_baseline.scripts.plot_lpv_vs_frozen
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.io import loadmat

from lpv_lfr_baseline.core.physics import M0, M1, M2, K, C, P, ts
from lpv_lfr_baseline.core.lfr_simulate import simulate, simulate_frozen

_MAT_PATH  = os.path.join(os.path.dirname(__file__), '..', '..', 'Matlab-output', 'lpv_sim_varying_y.mat')
_Y_FREEZE  = 0.3
_CH_NAMES  = ['X1', 'X2', 'Y']
_DTYPE     = torch.float64

# --- Load data ---
mat         = loadmat(_MAT_PATH)
q1_ref      = torch.tensor(mat['q1'],   dtype=_DTYPE)   # (N, 3) stage coords
u_seq_stage = torch.tensor(mat['u_q1'], dtype=_DTYPE)   # (N, 3) stage forces
t_sim       = mat['t_sim'].squeeze()                     # (N,)
N           = q1_ref.shape[0]

# --- Simulate ---
x0      = torch.zeros(1, 6, dtype=_DTYPE)
x0[0, 2] = 0.3
u_batch = u_seq_stage.unsqueeze(0)   # (1, N, 3)

with torch.no_grad():
    res_lpv    = simulate(      x0, u_batch, M0, M1, M2, K, C, P, ts)
    res_frozen = simulate_frozen(x0, u_batch, M0, M1, M2, K, C, P, ts, Y_freeze=_Y_FREEZE)

y_ref    = q1_ref.numpy()
y_lpv    = res_lpv.Y[0].numpy()
y_frozen = res_frozen.Y[0].numpy()

err_lpv    = np.abs(y_lpv    - y_ref)
err_frozen = np.abs(y_frozen - y_ref)

# --- Figure 1: trajectories ---
fig1, axes1 = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
fig1.suptitle('Trajectories: MATLAB ref vs Python LPV vs Python frozen', fontsize=12)

for i, (ax, ch) in enumerate(zip(axes1, _CH_NAMES)):
    ax.plot(t_sim[:N], y_ref[:, i],    color='tab:orange', linestyle='--', linewidth=1.5, label='MATLAB ref')
    ax.plot(t_sim[:N], y_lpv[:, i],    color='tab:blue',   linestyle='-',  linewidth=1.0, label='Python LPV')
    ax.plot(t_sim[:N], y_frozen[:, i], color='tab:red',    linestyle='-',  linewidth=1.0, label='Python frozen', alpha=0.7)
    ax.set_ylabel(f'{ch} [m]')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper right')

axes1[-1].set_xlabel('Time [s]')
plt.tight_layout()

# --- Figure 2: errors ---
fig2, axes2 = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
fig2.suptitle('Absolute errors: |Python LPV - ref|  vs  |Python frozen - ref|', fontsize=12)

for i, (ax, ch) in enumerate(zip(axes2, _CH_NAMES)):
    ax.semilogy(t_sim[:N], err_lpv[:, i]    + 1e-20, color='tab:blue', linewidth=1.0, label='|LPV - ref|')
    ax.semilogy(t_sim[:N], err_frozen[:, i] + 1e-20, color='tab:red',  linewidth=1.0, label='|frozen - ref|', alpha=0.8)
    ax.set_ylabel(f'{ch} |error| [m]')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8, loc='upper right')

axes2[-1].set_xlabel('Time [s]')
plt.tight_layout()

plt.show()
