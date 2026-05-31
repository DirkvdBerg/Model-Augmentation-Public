"""
gantry_subnet_verification.py
------------------------------
Verify that the full SSE_Interconnect pipeline (encoder + Gantry_State_Block +
Linear_Output_Block) trains correctly on matched MATLAB data (Phase 1).

Three checks after training:
  1. Encoder-initialised sim-NRMS per channel (X1, X2, Y) — primary success metric
  2. Zero-state sim-NRMS per channel — encoder must beat this
  3. x̂₀ vs x_logical[NA] per channel — direct encoder quality check

Run from project root:
    conda run -n GraduationProject python scripts/gantry/verification/gantry_subnet_verification.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import torch
import deepSI
from scipy.io import loadmat

from model_augmentation.fit_systems.interconnect import Interconnect, SSE_Interconnect
from model_augmentation.fit_systems.blocks import Gantry_State_Block, Linear_Output_Block
from model_augmentation.systems.gantry_ss import Cd, Dd

# ── Hyperparameters ───────────────────────────────────────────────────────────
NA     = 100   # encoder history [samples] — 5 ms at 20 kHz
NB     = 100
NF     = 650   # BPTT horizon [samples]   — 32.5 ms (validated at this rate in train_param_recovery.py)
EPOCHS = 30
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
std_u = train_data.u.std(axis=0).reshape(NU, 1).astype(np.float32)  # (3, 1) [N]
std_x = train_data.x.std(axis=0).reshape(NX, 1).astype(np.float32)  # (6, 1) [m, m/s]

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
# fit_sys.simulate: runs encoder on first NA/NB samples → x̂₀, then steps
# the interconnect forward for the full trajectory. Returns physical-unit ŷ.
sim_result = fit_sys.simulate(val_data)
y_hat_enc  = sim_result.y if hasattr(sim_result, 'y') else np.array(sim_result)
y_ref      = val_data.y   # (T, 3) physical [m]

nrms_enc = np.sqrt(((y_hat_enc - y_ref) ** 2).mean(axis=0)) / y_ref.std(axis=0)
print('\n=== Check 1: Encoder-initialised sim-NRMS ===')
for ch, label in enumerate(['X1', 'X2', 'Y ']):
    print(f'  {label}: {nrms_enc[ch]:.4f}')

# ── Check 2: Zero-state sim-NRMS ─────────────────────────────────────────────
# Simulate the same trajectory from x₀ = 0, with no encoder — baseline to beat.
# The interconnect operates on normalised (u, y); output is renormalised after.
val_norm = fit_sys.norm.transform(val_data)  # normalised u and y; x passed through
T_val    = len(val_norm.u)

x_zero       = torch.zeros(1, NX)
y_zero_norm  = np.zeros((T_val, NY), dtype=np.float32)

with torch.no_grad():
    for t in range(T_val):
        u_t = torch.tensor(val_norm.u[t], dtype=torch.float32).unsqueeze(0)  # (1, 3)
        y_t, x_zero = interconnect(x_zero, u_t)
        y_zero_norm[t] = y_t.squeeze().numpy()

y_zero = y_zero_norm * fit_sys.norm.ystd + fit_sys.norm.y0  # denormalise → physical [m]

nrms_zero = np.sqrt(((y_zero - y_ref) ** 2).mean(axis=0)) / y_ref.std(axis=0)
print('\n=== Check 2: Zero-state vs encoder sim-NRMS ===')
for ch, label in enumerate(['X1', 'X2', 'Y ']):
    status = 'PASS' if nrms_enc[ch] < nrms_zero[ch] else 'FAIL'
    print(f'  {label}: enc={nrms_enc[ch]:.4f}  zero={nrms_zero[ch]:.4f}  {status}')

# ── Check 3: x̂₀ vs x_logical[NA] ────────────────────────────────────────────
# Encoder maps first NB u samples and NA y samples → x̂₀ in normalised state units.
# Block convention: x_phys = x_norm * std_x, so x̂₀_phys = x̂₀_norm * std_x.
# Compare with x_logical at sample NA (true state after the encoder warmup window).
u_past = torch.tensor(val_norm.u[:NB], dtype=torch.float32).unsqueeze(0)  # (1, NB, 3)
y_past = torch.tensor(val_norm.y[:NA], dtype=torch.float32).unsqueeze(0)  # (1, NA, 3)

with torch.no_grad():
    x_hat_norm = fit_sys.encoder(u_past, y_past)              # (1, 6) normalised

x_hat_phys = x_hat_norm.squeeze().numpy() * std_x.flatten()  # (6,) physical [m, m/s]
x_true      = val_data.x[NA]                                   # (6,) physical [m, m/s]

print('\n=== Check 3: x̂₀ vs x_logical[NA] ===')
labels_x = ['q1   [m]   ', 'q2   [rad] ', 'q3   [m]   ',
            'q1dot[m/s] ', 'q2dot[r/s] ', 'q3dot[m/s] ']
for i, lab in enumerate(labels_x):
    print(f'  {lab}  true={x_true[i]:+.4e}  hat={x_hat_phys[i]:+.4e}  |err|={abs(x_hat_phys[i]-x_true[i]):.2e}')

# ── Save ──────────────────────────────────────────────────────────────────────
if SAVE:
    save_dir  = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'simulations', 'gantry_subnet')
    save_path = os.path.join(save_dir, 'phase1')
    os.makedirs(save_dir, exist_ok=True)
    fit_sys.save_system(save_path)
    print(f'\nSaved: {save_path}')
