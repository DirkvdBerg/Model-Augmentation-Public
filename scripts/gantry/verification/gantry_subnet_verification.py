"""
gantry_subnet_verification.py
------------------------------
Verify that the full SSE_Interconnect pipeline (encoder + Gantry_State_Block +
Linear_Output_Block) trains correctly on matched MATLAB data (Phase 1).

Two checks after training:
  1. Encoder-initialised sim-NRMS per channel (X1, X2, Y) — primary success metric
  2. Zero-state sim-NRMS per channel — encoder must beat this

Check 3 (x̂₀ vs x_logical) is not implemented: the encoder output is calibrated to
satisfy Cd @ x̂₀ ≈ (y - y0)/ystd, not x̂₀ ≈ x_logical/std_x. A direct state
comparison is not valid without knowing the model's internal coordinate convention.

Run from project root:
    conda run -n GraduationProject python scripts/gantry/verification/gantry_subnet_verification.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import torch
import matplotlib.pyplot as plt
import deepSI
from scipy.io import loadmat

from model_augmentation.fit_systems.interconnect import Interconnect, SSE_Interconnect
from model_augmentation.fit_systems.blocks import Gantry_State_Block, Linear_Output_Block
from model_augmentation.systems.gantry_ss import Cd, Dd

# ── Hyperparameters ───────────────────────────────────────────────────────────
NA     = 100   # encoder history [samples] — 5 ms at 20 kHz
NB     = 100
NF     = 200   # BPTT horizon [samples]   — 32.5 ms (validated at this rate in train_param_recovery.py)
EPOCHS = 5
BATCH  = 256
SAVE   = True

NX, NU, NY = 6, 3, 3

# ── Load data ─────────────────────────────────────────────────────────────────
_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'gantry', 'matlab')

def load_mat(split):
    d = loadmat(os.path.join(_DATA_DIR, f'gantry_lti_{split}.mat'), squeeze_me=True)
    return deepSI.System_data(
        u  = d['u'].astype(np.float32),            # (T, 3)  stage forces [N]
        y  = d['y'].astype(np.float32),            # (T, 3)  stage positions [m]
        x  = d['x_logical'].astype(np.float32),    # (T, 6)  logical state — encoder verify only
        dt = float(d['dt']),
    )

train_data = load_mat('train')
val_data   = load_mat('val')
print(f'Train: T={len(train_data.u)}  Val: T={len(val_data.u)}')

# ── Normalisation stats for Gantry_State_Block ────────────────────────────────
# std_x / std_u normalise the physical ODE inputs inside the block.
# Separate from fit_sys.norm, which normalises the encoder's (u, y) observations.
std_u = train_data.u.std(axis=0).reshape(NU, 1).astype(np.float32) + 1e-8  # (3, 1) [N]
std_x = train_data.x.std(axis=0).reshape(NX, 1).astype(np.float32) + 1e-8  # (6, 1) [m, m/s]
# 1e-8 guard: Y position is frozen at 0.3 m → std_x[2] ≈ 0 without it.

# ── Build Interconnect ────────────────────────────────────────────────────────
interconnect = Interconnect(nx=NX, nu=NU, ny=NY)

gantry_block = Gantry_State_Block(Y_op=0.3, std_x=std_x, std_u=std_u)
output_block  = Linear_Output_Block(C=Cd, D=Dd)

interconnect.add_block(gantry_block)
interconnect.add_block(output_block)

# Wiring — identical to verify_interconnect.py (already verified bit-exact)
interconnect.connect_signals("x", gantry_block)
interconnect.connect_block_signals(gantry_block, ["u"], [])
interconnect.connect_signals(gantry_block, "xp")

interconnect.connect_signals("x", output_block)
interconnect.connect_block_signals(output_block, ["u"], ["y"])

# ── Train ─────────────────────────────────────────────────────────────────────
fit_sys = SSE_Interconnect(
    na=NA, nb=NB,
    interconnect=interconnect,
    e_net_kwargs={'n_nodes_per_layer': 64, 'n_hidden_layers': 2},
)

fit_sys.fit(
    train_sys_data=train_data,
    val_sys_data=val_data,
    epochs=EPOCHS,
    batch_size=BATCH,
    auto_fit_norm=True,
    loss_kwargs={'nf': NF},
    validation_measure='sim-NRMS',
)

fit_sys.eval()

# ── Check 1: Encoder-initialised sim-NRMS ────────────────────────────────────
# apply_experiment: encoder(u[:NB], y[:NA]) → x̂₀, then rolls out the interconnect
# for the full trajectory. Returns fixed_System_data with .normed=False (physical y)
# and .cheat_n = max(NA, NB): the first cheat_n rows are copied verbatim from
# val_data.y (encoder warmup). NRMS must exclude these rows — error is zero there
# by construction, not because the model predicted correctly.
sim_result = fit_sys.apply_experiment(val_data)
cheat_n    = sim_result.cheat_n          # = max(NA, NB) = 100
y_hat_enc  = sim_result.y               # (T, 3) physical [m], normed=False
y_ref      = val_data.y                 # (T, 3) physical [m]

nrms_enc = (
    np.sqrt(((y_hat_enc[cheat_n:] - y_ref[cheat_n:]) ** 2).mean(axis=0))
    / y_ref[cheat_n:].std(axis=0)
)
print('\n=== Check 1: Encoder-initialised sim-NRMS ===')
for ch, label in enumerate(['X1', 'X2', 'Y ']):
    print(f'  {label}: {nrms_enc[ch]:.4f}')

# ── Check 2: Zero-state sim-NRMS ─────────────────────────────────────────────
# Simulate the same trajectory from x₀ = 0, with no encoder — baseline to beat.
# The interconnect operates on normalised (u, y); output is renormalised after.
val_norm = fit_sys.norm.transform(val_data)  # normalised u and y; x is stripped (becomes None)
T_val    = len(val_norm.u)

x_zero       = torch.zeros(1, NX)
y_zero_norm  = np.zeros((T_val, NY), dtype=np.float32)

with torch.no_grad():
    for t in range(T_val):
        u_t = torch.tensor(val_norm.u[t], dtype=torch.float32).unsqueeze(0)  # (1, 3)
        y_t, x_zero = interconnect(x_zero, u_t)
        y_zero_norm[t] = y_t.squeeze().numpy()

y_zero = y_zero_norm * fit_sys.norm.ystd + fit_sys.norm.y0  # denormalise → physical [m]

nrms_zero = (
    np.sqrt(((y_zero[cheat_n:] - y_ref[cheat_n:]) ** 2).mean(axis=0))
    / y_ref[cheat_n:].std(axis=0)
)
print('\n=== Check 2: Zero-state vs encoder sim-NRMS ===')
for ch, label in enumerate(['X1', 'X2', 'Y ']):
    status = 'PASS' if nrms_enc[ch] < nrms_zero[ch] else 'FAIL'
    print(f'  {label}: enc={nrms_enc[ch]:.4f}  zero={nrms_zero[ch]:.4f}  {status}')

# ── Plots ─────────────────────────────────────────────────────────────────────
plot_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'simulations', 'gantry_subnet')
os.makedirs(plot_dir, exist_ok=True)

# Plot 1: Validation loss convergence (mirrors Fig. 4 in Hoekstra 2025)
fig1, ax1 = plt.subplots(figsize=(7, 3.5))
ax1.semilogy(fit_sys.epoch_id, fit_sys.Loss_val)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Validation RMSE')
ax1.set_title('SSE_Interconnect — validation loss convergence (Phase 1 baseline)')
ax1.grid(True, which='both')
fig1.tight_layout()
fig1.savefig(os.path.join(plot_dir, 'phase1_val_loss.png'), dpi=150)

# Plot 2: Simulated vs reference per channel (mirrors Fig. 5 in Hoekstra 2025)
# Shows encoder-initialised sim, zero-state sim, and reference over the full
# validation trajectory. Vertical line marks warmup boundary (cheat_n samples).
t_val   = np.arange(len(y_ref)) * val_data.dt   # time axis [s]
cheat_t = cheat_n * val_data.dt                  # warmup end [s]
ch_labels = ['X1 [m]', 'X2 [m]', 'Y [m]']

fig2, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
for ch, (ax, lab) in enumerate(zip(axes, ch_labels)):
    ax.plot(t_val, y_ref[:, ch],      color='black',  lw=0.8,  label='Reference')
    ax.plot(t_val, y_hat_enc[:, ch],  color='C0',     lw=0.9,  label=f'Encoder-init (NRMS={nrms_enc[ch]:.3f})')
    ax.plot(t_val, y_zero[:, ch],     color='C1',     lw=0.9,  linestyle='--', label=f'Zero-state   (NRMS={nrms_zero[ch]:.3f})')
    ax.axvline(cheat_t, color='grey', linestyle=':', lw=0.8,  label='Warmup end')
    ax.set_ylabel(lab)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True)
axes[-1].set_xlabel('Time [s]')
fig2.suptitle('Validation simulation — encoder-init vs zero-state (Phase 1 baseline)')
fig2.tight_layout()
fig2.savefig(os.path.join(plot_dir, 'phase1_simulation.png'), dpi=150)

plt.show()

# ── Save ──────────────────────────────────────────────────────────────────────
if SAVE:
    # Model checkpoint — can be reloaded with SSE_Interconnect.load_system()
    save_path = os.path.join(plot_dir, 'phase1')
    fit_sys.save_system(save_path)

    # All results in one file — full trajectories, loss history, metrics.
    # Load with: d = np.load('phase1_results.npz'); d['y_hat_enc'] etc.
    np.savez(
        os.path.join(plot_dir, 'phase1_results.npz'),
        # Trajectories — full length (T,3), physical units [m]
        y_ref     = y_ref,
        y_hat_enc = y_hat_enc,
        y_zero    = y_zero,
        t_val     = t_val,        # time axis [s], shape (T,)
        # Convergence
        epoch_id  = np.array(fit_sys.epoch_id),
        loss_val  = np.array(fit_sys.Loss_val),
        # Per-channel NRMS (post-warmup window)
        nrms_enc  = nrms_enc,     # shape (3,): [X1, X2, Y]
        nrms_zero = nrms_zero,    # shape (3,): [X1, X2, Y]
        # Metadata needed to interpret the arrays
        cheat_n   = np.array(cheat_n),
        dt        = np.array(val_data.dt),
        NA        = np.array(NA),
        NB        = np.array(NB),
        NF        = np.array(NF),
    )
    print(f'\nSaved model : {save_path}')
    print(f'Saved results: {os.path.join(plot_dir, "phase1_results.npz")}')
