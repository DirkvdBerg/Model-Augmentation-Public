"""
gantry_lpv_compare.py
---------------------
Compare the Python DT-LPV simulator against the MATLAB CT quasi-LPV reference
(q1 from export_lpv_sim.m) to validate the ZOH discretization.

What this shows:
    DT-LPV vs q1:  Both have identical physics (M(Y), C, K, no Coriolis).
                   Any residual is purely ZOH discretization error.
                   Expected: small (16 kHz, Delta-Y <= 0.125 mm/sample).

External scheduling is used (Y_schedule = q1(:,3) from MATLAB). This isolates
ZOH error only; self-scheduling would add a second approximation.

Run from repo root:
    conda run -n GraduationProject python scripts/gantry/gantry_lpv_compare.py
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat

sys.path.insert(0, os.path.dirname(__file__))
from gantry_lpv_sim_torch import GantryLPVSimulator

# ---------------------------------------------------------------------------
# 1. Load MATLAB reference data
# ---------------------------------------------------------------------------
mat_path = os.path.join(
    os.path.dirname(__file__), '..', '..',
    'Matlab-output', 'lpv_sim_varying_y.mat'
)

if not os.path.exists(mat_path):
    raise FileNotFoundError(
        f'MATLAB export not found:\n  {mat_path}\n'
        'Run Matlab-scripts/export_lpv_sim.m first.'
    )

mat = loadmat(mat_path)
q1          = mat['q1'].astype(np.float64)          # (N, 3)  stage: [X1, X2, Y]
u_q1        = mat['u_q1'].astype(np.float64)        # (N, 3)  stage: [F_X1, F_X2, F_Y]
Y_traj      = mat['Y_trajectory'].astype(np.float64).squeeze()  # (N,)
t_sim       = mat['t_sim'].astype(np.float64).squeeze()         # (N,)
fs          = float(mat['fs'].squeeze())

N = q1.shape[0]
print(f'Loaded: N={N} samples, fs={fs:.0f} Hz, duration={t_sim[-1]:.3f} s')
print(f'Y range: {Y_traj.min()*1e3:.1f} mm to {Y_traj.max()*1e3:.1f} mm')

# ---------------------------------------------------------------------------
# 2. Build initial state x0 in logical coordinates
#
# The simulator state is [X, Theta, Y, dX, dTheta, dY] (logical).
# Outputs are [X1, X2, Y] in stage coordinates.
# Stage-to-logical position inversion (from C_stage = P.T @ C_logical):
#   X1 = X + (Lb/2) * Theta,  X2 = X - (Lb/2) * Theta,  Y_stage = Y_log
#   => X = (X1 + X2) / 2,  Theta = (X1 - X2) / Lb,  Y_log = Y_stage
# Velocities: zero at simulation start (hold period, system at rest).
# ---------------------------------------------------------------------------
Lb = 0.725  # must match gantry_lpv_torch.py

X1_0, X2_0, Y_0 = q1[0, 0], q1[0, 1], q1[0, 2]
x0 = torch.zeros(6, dtype=torch.float64)
x0[0] = (X1_0 + X2_0) / 2        # X  (logical position)
x0[1] = (X1_0 - X2_0) / Lb       # Theta
x0[2] = Y_0                       # Y  (scheduling variable starts here)
# x0[3:] = 0  (velocities: at rest)

print(f'\nInitial state (logical): X={x0[0].item()*1e3:.3f} mm, '
      f'Theta={x0[1].item()*1e6:.3f} urad, Y={x0[2].item()*1e3:.3f} mm')

# ---------------------------------------------------------------------------
# 3. Run DT-LPV simulator with external scheduling
# ---------------------------------------------------------------------------
sim = GantryLPVSimulator(fs=torch.tensor(fs, dtype=torch.float64))

u_t = torch.tensor(u_q1, dtype=torch.float64)            # (N, 3)
Y_t = torch.tensor(Y_traj, dtype=torch.float64)          # (N,)

y_lpv = sim.simulate(x0, u_t, Y_schedule=Y_t).numpy()    # (N, 3)

# ---------------------------------------------------------------------------
# 4. Residual: DT-LPV vs q1 (CT quasi-LPV)
# ---------------------------------------------------------------------------
residual = y_lpv - q1  # (N, 3)

channel_names = ['X1', 'X2', 'Y']

print('\n' + '=' * 55)
print('  DT-LPV vs q1 (CT quasi-LPV) -- ZOH discretization error')
print('=' * 55)
print(f'  {"Channel":<8}  {"RMS [m]":>12}  {"Max |err| [m]":>14}')
print('  ' + '-' * 42)
for i, ch in enumerate(channel_names):
    rms = np.sqrt(np.mean(residual[:, i]**2))
    mx  = np.max(np.abs(residual[:, i]))
    print(f'  {ch:<8}  {rms:>12.3e}  {mx:>14.3e}')

# BFR per channel: 1 - ||e|| / ||q1 - mean(q1)||
print()
print(f'  {"Channel":<8}  {"BFR [%]":>10}')
print('  ' + '-' * 22)
for i, ch in enumerate(channel_names):
    denom = np.linalg.norm(q1[:, i] - np.mean(q1[:, i]))
    bfr = (1.0 - np.linalg.norm(residual[:, i]) / denom) * 100.0 if denom > 0 else float('nan')
    print(f'  {ch:<8}  {bfr:>10.4f}')
print('=' * 55)

# ---------------------------------------------------------------------------
# 5. Plot
# ---------------------------------------------------------------------------
out_dir = os.path.join(
    os.path.dirname(__file__), '..', '..',
    'simulations', 'lpv_zoh'
)
os.makedirs(out_dir, exist_ok=True)

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

for i, (ax, ch) in enumerate(zip(axes, channel_names)):
    ax.plot(t_sim, q1[:, i] * 1e3, 'k-',  lw=1.2, label='q1 (CT MATLAB)')
    ax.plot(t_sim, y_lpv[:, i] * 1e3, 'r--', lw=1.0, label='DT-LPV (Python)')
    ax.set_ylabel(f'{ch} [mm]')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.4)

axes[-1].set_xlabel('Time [s]')
axes[0].set_title('DT-LPV vs CT quasi-LPV (q1) -- ZOH discretization validation')

fig.tight_layout()
traj_path = os.path.join(out_dir, 'trajectories.png')
fig.savefig(traj_path, dpi=150)
plt.close(fig)

# Residual plot
fig2, axes2 = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
for i, (ax, ch) in enumerate(zip(axes2, channel_names)):
    ax.plot(t_sim, residual[:, i], lw=0.8)
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set_ylabel(f'{ch} error [m]')
    ax.grid(True, alpha=0.4)

axes2[-1].set_xlabel('Time [s]')
axes2[0].set_title('ZOH discretization error: DT-LPV minus q1 (CT quasi-LPV)')

fig2.tight_layout()
err_path = os.path.join(out_dir, 'zoh_error.png')
fig2.savefig(err_path, dpi=150)
plt.close(fig2)

print(f'\nPlots saved to: simulations/lpv_zoh/')
print(f'  {os.path.basename(traj_path)}')
print(f'  {os.path.basename(err_path)}')
