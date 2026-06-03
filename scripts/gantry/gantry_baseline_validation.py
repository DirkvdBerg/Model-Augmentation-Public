"""
gantry_baseline_validation.py
------------------------------
Validates the Python Gantry_State_Block physics against MATLAB ground truth.

Section 1 — Normalization computed from gantry_comb_baseline (Y moves, meaningful std).
Section 2 — Normalization checks on gantry_comb_baseline:
              a) Algebraic: Cd @ x_logical vs y_ref
              b) One-step teacher forcing: per-step prediction error
Section 3 — Open-loop rollout on gantry_lti_train  (LTI data, expect near-zero error)
Section 4 — Open-loop rollout on gantry_comb_baseline (Y moves, expect LPV error)

One shared normalization and one shared Gantry_State_Block (Y_op=None, LPV).
One rollout function and one plot function, called for each dataset.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.io import loadmat

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from model_augmentation.systems.gantry_ss import Cd
from model_augmentation.fit_systems.blocks import Gantry_State_Block

# ── Constants ──────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'gantry', 'matlab')
Cd_np = Cd.numpy()  # (3, 6)  Dd = 0, no feedthrough

CH_LABELS    = ['X1 [m]', 'X2 [m]', 'Y [m]']
U_LABELS     = ['F_X1 [N]', 'F_X2 [N]', 'F_Y [N]']
STATE_LABELS = ['q1 (mean-X) [m]', 'q2 (arm-rot) [rad]', 'q3 (Y) [m]',
                'dq1 [m/s]',       'dq2 [rad/s]',         'dq3 [m/s]']


# ── Data loading ───────────────────────────────────────────────────────────────
def load_mat(name):
    d = loadmat(os.path.join(DATA_DIR, f'{name}.mat'), squeeze_me=True)
    return {
        'u':         d['u'].astype(np.float32),          # (T, 3) stage forces
        'y':         d['y'].astype(np.float32),          # (T, 3) stage positions
        'x_logical': d['x_logical'].astype(np.float32), # (T, 6) [q; dq] logical coords
        'dt':        float(d['dt']),
    }


# ── Section 1: Normalization from gantry_comb_baseline ────────────────────────
comb = load_mat('gantry_comb_baseline')

x_mean = comb['x_logical'].mean(axis=0)         # (6,)  captures Y_op from data
std_x  = comb['x_logical'].std(axis=0) + 1e-8  # (6,)  std around mean
std_u  = comb['u'].std(axis=0)         + 1e-8  # (3,)
ystd   = comb['y'].std(axis=0)         + 1e-8  # (3,)  for NRMS only

print('=== Section 1: Normalization parameters (source: gantry_comb_baseline) ===')
print(f'  x_mean : {np.array2string(x_mean, precision=4)}')
print(f'  std_x  : {np.array2string(std_x,  precision=4)}')
print(f'  std_u  : {np.array2string(std_u,  precision=4)}')
print(f'  ystd   : {np.array2string(ystd,   precision=4)}')

x_norm_comb = (comb['x_logical'] - x_mean) / std_x  # (T, 6)
u_norm_comb =  comb['u'] / std_u                     # (T, 3)

print('\n  Normalized state statistics (target: mean≈0, std≈1, range≈[-3, 3])')
print(f'  {"Channel":<22} {"mean":>7} {"std":>7} {"min":>7} {"max":>7}')
for i, lbl in enumerate(STATE_LABELS):
    c = x_norm_comb[:, i]
    print(f'  {lbl:<22} {c.mean():>7.3f} {c.std():>7.3f} {c.min():>7.3f} {c.max():>7.3f}')

print(f'\n  {"Input":<22} {"mean":>7} {"std":>7} {"min":>7} {"max":>7}')
for i, lbl in enumerate(U_LABELS):
    c = u_norm_comb[:, i]
    print(f'  {lbl:<22} {c.mean():>7.3f} {c.std():>7.3f} {c.min():>7.3f} {c.max():>7.3f}')


# ── Build shared block ─────────────────────────────────────────────────────────
# Y_op=None: LPV mode — Y read from state each RK4 substep.
# For lti_train where Y≈const≈0.3 m, LPV gives the same result as frozen Y_op=0.3.
block = Gantry_State_Block(
    Y_op   = None,
    std_x  = std_x.reshape(6, 1),
    std_u  = std_u.reshape(3, 1),
    x_mean = x_mean.reshape(6, 1),
)
block.eval()


# ── Section 2a: Algebraic output check ────────────────────────────────────────
# x_logical positions are derived in MATLAB as (P^T)^{-1} @ y_stage,
# so Cd @ x_logical = P^T @ q = y_stage exactly by construction.
# Non-zero RMS here means a bug in Cd or x_logical construction.
y_alg    = (Cd_np @ comb['x_logical'].T).T   # (T, 3)
err_alg  = y_alg - comb['y']

print('\n=== Section 2a: Algebraic check  Cd @ x_logical vs y  (expect ~0) ===')
for i, lbl in enumerate(CH_LABELS):
    rms = np.sqrt((err_alg[:, i] ** 2).mean())
    mx  = np.abs(err_alg[:, i]).max()
    print(f'  {lbl}: RMS={rms:.3e}  max={mx:.3e}')


# ── Section 2b: One-step teacher forcing ──────────────────────────────────────
# Feed ground-truth x[t] and u[t], predict x[t+1].
# Batched over all T-1 steps — each step is independent so batch is valid.
# Velocity channels have MATLAB numerical-diff noise; small errors expected there.
T      = len(comb['u'])
x_in   = torch.tensor(x_norm_comb[:-1].reshape(T - 1, 6, 1), dtype=torch.float32)
u_in   = torch.tensor(u_norm_comb[:-1].reshape(T - 1, 3, 1), dtype=torch.float32)
z_in   = torch.cat([x_in, u_in], dim=1)   # (T-1, 9, 1)

with torch.no_grad():
    x_pred_norm = block.nonlinear_function(z_in).squeeze(-1).numpy()  # (T-1, 6)

x_pred = x_pred_norm * std_x + x_mean     # denormalise
x_true = comb['x_logical'][1:]

err_step = x_pred - x_true
print('\n=== Section 2b: One-step teacher forcing RMS ===')
print('  (velocity channels: small errors expected from MATLAB numerical diff)')
for i, lbl in enumerate(STATE_LABELS):
    print(f'  {lbl}: {np.sqrt((err_step[:, i] ** 2).mean()):.3e}')


# ── Rollout ────────────────────────────────────────────────────────────────────
def rollout(data):
    x_logical, u = data['x_logical'], data['u']
    T = len(u)

    x_norm   = torch.tensor(
        ((x_logical[0] - x_mean) / std_x).reshape(1, 6, 1), dtype=torch.float32
    )
    u_tensor = torch.tensor(
        (u / std_u).reshape(T, 3, 1), dtype=torch.float32
    )  # precomputed: avoids per-step tensor allocation in loop
    y_hat = np.zeros((T, 3), dtype=np.float32)

    with torch.no_grad():
        for t in range(T):
            x_phys    = x_norm.squeeze().numpy() * std_x + x_mean   # (6,)
            y_hat[t]  = Cd_np @ x_phys                               # (3,)
            z         = torch.cat([x_norm, u_tensor[t:t+1]], dim=1)  # (1, 9, 1)
            x_norm    = block.nonlinear_function(z)                   # (1, 6, 1)

    return y_hat


# ── Plot ───────────────────────────────────────────────────────────────────────
def plot_validation(data, y_hat, label):
    t    = np.arange(len(data['u'])) * data['dt']
    u    = data['u']
    y    = data['y']
    nrms = np.sqrt(((y_hat - y) ** 2).mean(axis=0)) / ystd

    # Inputs
    fig, axes = plt.subplots(3, 1, figsize=(12, 5), sharex=True)
    for i, (ax, lbl) in enumerate(zip(axes, U_LABELS)):
        ax.plot(t, u[:, i], 'k', lw=0.7)
        ax.set_ylabel(lbl); ax.grid(True)
    axes[0].set_title(f'Inputs — {label}')
    axes[-1].set_xlabel('Time [s]')
    fig.tight_layout()

    # Trajectories (left) and differences (right)
    fig, axes = plt.subplots(3, 2, figsize=(14, 7), sharex=True)
    for i, lbl in enumerate(CH_LABELS):
        axes[i, 0].plot(t, y[:, i],     'k',  lw=0.8, label='MATLAB')
        axes[i, 0].plot(t, y_hat[:, i], 'C0', lw=0.9, linestyle='--',
                        label=f'Python RK4  NRMS={nrms[i]:.4f}')
        axes[i, 0].set_ylabel(lbl); axes[i, 0].legend(fontsize=7); axes[i, 0].grid(True)

        diff = y_hat[:, i] - y[:, i]
        axes[i, 1].plot(t, diff, 'C1', lw=0.7)
        axes[i, 1].axhline(0, color='k', lw=0.5, linestyle='--')
        axes[i, 1].set_ylabel(f'Δ{lbl}'); axes[i, 1].grid(True)

    axes[0, 0].set_title(f'Trajectories — {label}')
    axes[0, 1].set_title('Difference  Python − MATLAB')
    axes[-1, 0].set_xlabel('Time [s]')
    axes[-1, 1].set_xlabel('Time [s]')
    fig.tight_layout()

    print(f'\n=== NRMS — {label} ===')
    for i, lbl in enumerate(CH_LABELS):
        print(f'  {lbl}: {nrms[i]:.4f}')


# ── Sections 3 & 4: Validate both datasets ────────────────────────────────────
datasets = [
    ('gantry_lti_train',    'Section 3 — LTI train      (expect near-zero error)'),
    ('gantry_comb_baseline','Section 4 — Comb baseline   (Y moves, LPV error visible)'),
]

for name, label in datasets:
    data  = load_mat(name)
    y_hat = rollout(data)
    plot_validation(data, y_hat, label)

plt.show()
