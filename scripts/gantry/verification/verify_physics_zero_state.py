"""
verify_physics_zero_state.py
----------------------------
Verify that the physics interconnect (zero initial state) reproduces the
measured gantry data. No encoder, no training — just the ODE rollout from x=0.

If the physics is correct and the data starts near x=0, RMSE should be small
for X1 and X2. Y will drift because x=0 in normalised space corresponds to
Y=0, not Y_OP=0.3 m.

Run from project root:
    conda run -n GraduationProject python scripts/gantry/verification/verify_physics_zero_state.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import deepSI
from scipy.io import loadmat

from model_augmentation.fit_systems.interconnect import Interconnect
from model_augmentation.fit_systems.blocks import Gantry_State_Block, Linear_Output_Block
from model_augmentation.systems.gantry_ss import Cd, Dd

# ── Configuration ─────────────────────────────────────────────────────────────
N_HOLD = 10000
Y_OP   = 0.3
NX, NU, NY = 6, 3, 3

_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'gantry', 'matlab')
plot_dir  = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'simulations', 'gantry_subnet', 'physics_verify')
os.makedirs(plot_dir, exist_ok=True)

# ── Load data (stripped, same as verification script) ─────────────────────────
def load_mat(split):
    d = loadmat(os.path.join(_DATA_DIR, f'gantry_lti_{split}.mat'), squeeze_me=True)
    return deepSI.System_data(
        u  = d['u'][N_HOLD:-N_HOLD].astype(np.float32),
        y  = d['y'][N_HOLD:-N_HOLD].astype(np.float32),
        x  = d['x_logical'][N_HOLD:-N_HOLD].astype(np.float32),
        dt = float(d['dt']),
    )

train_data = load_mat('train')
val_data   = load_mat('val')

dt = train_data.dt
print(f'Train: T={len(train_data.u)}  Val: T={len(val_data.u)}  dt={dt}')

# ── Normalisation ─────────────────────────────────────────────────────────────
std_u = train_data.u.std(axis=0).reshape(NU, 1).astype(np.float32) + 1e-8
std_x = train_data.x.std(axis=0).reshape(NX, 1).astype(np.float32) + 1e-8
std_x[2] = Y_OP
ystd  = train_data.y.std(axis=0).astype(np.float32) + 1e-8
ystd[2] = Y_OP
Cd_norm = Cd.numpy() * std_x.flatten()[None, :] / ystd[:, None]

# ── Build interconnect (no encoder, no training) ──────────────────────────────
interconnect = Interconnect(nx=NX, nu=NU, ny=NY)
gantry_block = Gantry_State_Block(Y_op=Y_OP, std_x=std_x, std_u=std_u)
output_block  = Linear_Output_Block(C=Cd_norm, D=Dd.numpy())
interconnect.add_block(gantry_block)
interconnect.add_block(output_block)
interconnect.connect_signals("x", gantry_block)
interconnect.connect_block_signals(gantry_block, ["u"], [])
interconnect.connect_signals(gantry_block, "xp")
interconnect.connect_signals("x", output_block)
interconnect.connect_block_signals(output_block, ["u"], ["y"])

# ── Zero-state rollout ────────────────────────────────────────────────────────
def zero_state_rollout(data, print_every=2000):
    """Roll out the interconnect from x=0 using measured inputs. Returns y_hat in physical units."""
    u_norm = data.u / std_u.flatten()   # (T, NU) normalised
    T = len(u_norm)
    x = torch.zeros(1, NX)
    y_out = np.zeros((T, NY), dtype=np.float32)
    with torch.no_grad():
        for t in range(T):
            if t % print_every == 0:
                print(f'  step {t}/{T}', flush=True)
            u_t = torch.tensor(u_norm[t], dtype=torch.float32).unsqueeze(0)
            y_t, x = interconnect(x, u_t)
            y_out[t] = y_t.squeeze().numpy()
    return y_out * ystd   # denormalise → physical [m]

print('Running zero-state rollout on train data...')
y_zero_train = zero_state_rollout(train_data)
print('Running zero-state rollout on val data...')
y_zero_val   = zero_state_rollout(val_data)

# ── RMSE per channel ──────────────────────────────────────────────────────────
ch_labels = ['X1', 'X2', 'Y']

def print_rmse(y_hat, y_ref, label):
    rmse = np.sqrt(((y_hat - y_ref) ** 2).mean(axis=0))
    nrms = rmse / ystd
    print(f'\n  {label}:')
    print(f'    {"Ch":4s}  {"RMSE [m]":>12s}  {"NRMS":>8s}')
    for ch, name in enumerate(ch_labels):
        print(f'    {name:4s}  {rmse[ch]:12.6f}  {nrms[ch]:8.4f}')
    return rmse, nrms

print('\n=== Zero-state RMSE ===')
rmse_tr, nrms_tr = print_rmse(y_zero_train, train_data.y, 'Train')
rmse_vl, nrms_vl = print_rmse(y_zero_val,   val_data.y,   'Val')

# ── Plots ─────────────────────────────────────────────────────────────────────
t_tr = np.arange(len(train_data.u)) * dt
t_vl = np.arange(len(val_data.u))   * dt

for split, t_ax, y_hat, y_ref, rmse in [
    ('train', t_tr, y_zero_train, train_data.y, rmse_tr),
    ('val',   t_vl, y_zero_val,   val_data.y,   rmse_vl),
]:
    fig, axes = plt.subplots(NY * 2, 1, figsize=(12, 10), sharex=True)
    for ch, name in enumerate(ch_labels):
        # Reference vs predicted
        ax_top = axes[ch * 2]
        ax_top.plot(t_ax, y_ref[:, ch],  color='C0', lw=0.6, label='Measured')
        ax_top.plot(t_ax, y_hat[:, ch],  color='C1', lw=0.6, linestyle='--', label='Zero-state')
        ax_top.set_ylabel(f'{name} [m]')
        ax_top.legend(fontsize=7, loc='upper right')
        ax_top.grid(True)

        # Absolute error
        ax_bot = axes[ch * 2 + 1]
        ax_bot.plot(t_ax, np.abs(y_hat[:, ch] - y_ref[:, ch]), color='C2', lw=0.6)
        ax_bot.set_ylabel(f'|err| {name} [m]')
        ax_bot.set_yscale('log')
        ax_bot.set_title(f'RMSE={rmse[ch]:.4e} m', fontsize=8)
        ax_bot.grid(True, which='both')

    axes[-1].set_xlabel('Time [s]')
    fig.suptitle(f'Zero-state physics rollout vs measured — {split}\n'
                 f'x₀=0 in normalised space (Y starts at 0, not Y_OP=0.3 m)')
    fig.tight_layout()
    path = os.path.join(plot_dir, f'physics_zero_state_{split}.png')
    fig.savefig(path, dpi=150)
    print(f'Saved: {path}')

print('\nDone.')
