"""
verify_baseline_rms.py
----------------------
Physics-only baseline RMS diagnostic for the gantry SUBNET.

Uses the same data loading, normalization, decimation, and windowing as
diagnose_encoder.py, but with NO training. Evaluates pure physics model
accuracy using analytical x0 (logical coords, finite-diff velocities).

Two configurations:
  A. Physics only (nxd=6): Gantry_State_Block + Linear_Output_Block
  B. Physics + inert ANN (nxd=8): same as diagnose_encoder.py, zero-init ANN

Config A and B should give identical RMS (sanity check).
Per-channel RMS should be in ~1e-4 m range (matching MATLAB MSD comparison).

Run: conda run -n GraduationProject python scripts/gantry/verification/verify_baseline_rms.py
"""

import os
import sys
import numpy as np
import torch
from scipy.io import loadmat

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from model_augmentation.utils.utils import selection_matrix, expansion_matrix
from model_augmentation.utils.torch_nets import zero_init_feed_forward_nn
from model_augmentation.fit_systems.interconnect import Interconnect
from model_augmentation.fit_systems.blocks import (
    Gantry_State_Block, Linear_Output_Block, Static_ANN_Block,
)
from model_augmentation.systems.gantry_ss import Cd, Dd, P

# =========================================================================
# Configuration (from diagnose_encoder.py)
# =========================================================================

NX_PHYS = 6
NX_ANN  = 2
nu = 3
ny = 3
Y_OP = None

FS_ORIG = 20000
FS_NEW  = 1000
D       = FS_ORIG // FS_NEW
TS_NEW  = 1.0 / FS_NEW

DTYPE_NP = np.float32
DTYPE_PT = torch.float32
PHY_IX   = np.arange(NX_PHYS)

nf = 350  # rollout horizon (same as diagnose_encoder.py)

# =========================================================================
# Data loading (from diagnose_encoder.py)
# =========================================================================

TRAJ_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..',
                        'data', 'gantry', 'matlab', 'trajectories')

TRAIN_FILES = [
    'T1_Y_sweep_conservative.mat', 'T2_X_sym_Y030.mat',
    'T3_X_sym_Y000.mat', 'T4_X_antisym_Y020.mat',
    'T5_X_sym_Y_sweep.mat', 'T6_Y_sweep_aggressive.mat',
    'T7_X_antisym_Y_sweep.mat', 'T8_X_sym_anti_Y_sweep.mat',
]
VAL_FILE  = 'V1_X_sym_Y_mid_sweep.mat'
TEST_FILE = 'E1_X_sym_anti_Y_low_offset_sweep.mat'

def load_traj(filename):
    d = loadmat(os.path.join(TRAJ_DIR, filename), squeeze_me=True)
    return {
        'u': d['u'][::D].astype(DTYPE_NP),
        'y': d['y'][::D].astype(DTYPE_NP),
    }

train_list = [load_traj(f) for f in TRAIN_FILES]
val_traj   = load_traj(VAL_FILE)
test_traj  = load_traj(TEST_FILE)

all_trajs  = train_list + [val_traj, test_traj]
all_labels = TRAIN_FILES + [VAL_FILE, TEST_FILE]

print(f'Loaded {len(train_list)} training + 1 val + 1 test trajectories')

# =========================================================================
# Normalisation (from diagnose_encoder.py, with coordinate fix)
# =========================================================================

u_all = np.concatenate([t['u'] for t in train_list])
y_all = np.concatenate([t['y'] for t in train_list])

fs = FS_NEW
P_inv_T = np.linalg.inv(P.numpy().T).astype(DTYPE_NP)  # stage -> logical
x_logical_list = []
for t in train_list:
    pos_logical = (P_inv_T @ t['y'].T).T
    vel_logical = np.diff(pos_logical, axis=0) * fs
    vel_logical = np.vstack([vel_logical[:1], vel_logical])
    x_logical_list.append(np.hstack([pos_logical, vel_logical]))
x_all = np.concatenate(x_logical_list)

x_mean = x_all.mean(axis=0).reshape(NX_PHYS, 1).astype(DTYPE_NP)
std_x  = x_all.std(axis=0).reshape(NX_PHYS, 1).astype(DTYPE_NP) + 1e-8
std_u  = u_all.std(axis=0).reshape(nu, 1).astype(DTYPE_NP) + 1e-8
u_mean = u_all.mean(axis=0).reshape(nu, 1).astype(DTYPE_NP)
ystd   = y_all.std(axis=0).astype(DTYPE_NP) + 1e-8
y0     = (Cd.numpy() @ x_mean.flatten()).astype(DTYPE_NP)

Cd_np   = Cd.numpy()
Cd_norm = Cd_np * std_x.flatten()[None, :] / ystd[:, None]
Dd_np   = Dd.numpy()

# =========================================================================
# Helpers (from diagnose_encoder.py, with coordinate fix)
# =========================================================================

def analytical_x0_at(y_seq, t_idx, dt):
    """Analytical state in logical coordinates."""
    pos_stage = y_seq[t_idx]
    pos = P_inv_T @ pos_stage
    if 0 < t_idx < len(y_seq) - 1:
        vel_stage = (y_seq[t_idx + 1] - y_seq[t_idx - 1]) / (2 * dt)
    elif t_idx == 0:
        vel_stage = (y_seq[t_idx + 1] - y_seq[t_idx]) / dt
    else:
        vel_stage = (y_seq[t_idx] - y_seq[t_idx - 1]) / dt
    vel = P_inv_T @ vel_stage
    return np.concatenate([pos, vel])


def normalize_x(x_phys):
    return ((x_phys.reshape(NX_PHYS, 1) - x_mean) / std_x).flatten()


def rollout(interconnect, x0_full, u_norm_seq, nxd_cfg):
    """Roll out nf steps, return denormalized y_hat (n_steps, 3)."""
    x = torch.tensor(x0_full.reshape(1, nxd_cfg), dtype=DTYPE_PT)
    y_list = []
    with torch.no_grad():
        for t in range(len(u_norm_seq)):
            y_t, x = interconnect.forward(x, u_norm_seq[t:t + 1])
            y_list.append(y_t.squeeze().numpy())
    y_hat_norm = np.array(y_list)
    return y_hat_norm * ystd + y0


# =========================================================================
# Build Config A: Physics only (nxd=6)
# =========================================================================

print('\nBuilding Config A: physics only (nxd=6)')
ic_a = Interconnect(NX_PHYS, nu, ny, debugging=False)

phy_a = Gantry_State_Block(
    Y_op=Y_OP, std_x=std_x, std_u=std_u,
    x_mean=x_mean, u_mean=u_mean, Ts=TS_NEW,
).to(DTYPE_PT)
out_a = Linear_Output_Block(C=Cd_norm, D=Dd_np)

ic_a.add_block(phy_a)
ic_a.add_block(out_a)
ic_a.connect_signals("x", phy_a)
ic_a.connect_block_signals(phy_a, ["u"], [])
ic_a.connect_signals(phy_a, "xp")
ic_a.connect_signals("x", out_a)
ic_a.connect_block_signals(out_a, ["u"], ["y"])

# =========================================================================
# Build Config B: Physics + inert ANN (nxd=8)
# =========================================================================

nxd_b = NX_PHYS + NX_ANN
print(f'Building Config B: physics + inert ANN (nxd={nxd_b})')
ic_b = Interconnect(nxd_b, nu, ny, debugging=False)

phy_b = Gantry_State_Block(
    Y_op=Y_OP, std_x=std_x, std_u=std_u,
    x_mean=x_mean, u_mean=u_mean, Ts=TS_NEW,
).to(DTYPE_PT)
out_b = Linear_Output_Block(C=Cd_norm, D=Dd_np)
ann_b = Static_ANN_Block(
    nz=nxd_b + nu, nw=nxd_b,
    n_nodes_per_layer=128, n_hidden_layers=3,
    net=zero_init_feed_forward_nn, activation=torch.nn.Tanh,
)

ic_b.add_block(phy_b)
ic_b.add_block(out_b)
ic_b.add_block(ann_b)

ic_b.connect_block_signals(ann_b, ["x", "u"], ["xp"])
ic_b.connect_signals("x", phy_b, "concat", selection_matrix(PHY_IX, nxd_b))
ic_b.connect_block_signals(phy_b, ["u"], [])
ic_b.connect_signals(phy_b, "xp", "additive", expansion_matrix(PHY_IX, nxd_b))
ic_b.connect_signals("x", out_b, "concat", selection_matrix(PHY_IX, nxd_b))
ic_b.connect_block_signals(out_b, ["u"], ["y"])

# =========================================================================
# Windowed rollout evaluation
# =========================================================================

cheat_n_a = NX_PHYS  # minimal history needed (no encoder)
cheat_n_b = nxd_b

CH_LABELS = ['X1', 'X2', 'Y']

print(f'\n{"="*70}')
print(f'Windowed rollout evaluation (nf={nf} steps = {nf*TS_NEW:.3f} s)')
print(f'{"="*70}')

header = (f'  {"Trajectory":<40s}  {"Win":>4s}  '
          f'{"A:X1":>9s} {"A:X2":>9s} {"A:Y":>9s}  '
          f'{"B:X1":>9s} {"B:X2":>9s} {"B:Y":>9s}  '
          f'{"maxdiff":>9s}')
print(header)
print('  ' + '-' * (len(header) - 2))

all_rms_a = []
all_rms_b = []
max_ab_diff = 0.0

for traj, label in zip(all_trajs, all_labels):
    T = len(traj['u'])
    usable_start = max(cheat_n_a, cheat_n_b, 1)  # need at least 1 for finite diff
    usable_end = T - nf

    if usable_end <= usable_start:
        print(f'  {label:<40s}  SKIPPED (too short)')
        continue

    n_windows = min(10, (usable_end - usable_start) // nf)
    if n_windows < 1:
        n_windows = 1
    starts = np.linspace(usable_start, usable_end, n_windows, dtype=int)

    # Normalize u for this trajectory
    u_norm = (traj['u'] - u_mean.flatten()) / std_u.flatten()
    u_norm_t = torch.tensor(u_norm, dtype=DTYPE_PT)

    traj_rms_a_ch = []
    traj_rms_b_ch = []

    for t_s in starts:
        y_ref = traj['y'][t_s:t_s + nf]
        n_cmp = min(nf, len(y_ref))
        u_seg = u_norm_t[t_s:t_s + n_cmp]

        # Analytical x0
        x0_phys = analytical_x0_at(traj['y'], t_s, TS_NEW)
        x0_norm = normalize_x(x0_phys)

        # Config A: nxd=6
        x0_a = x0_norm.copy()
        y_a = rollout(ic_a, x0_a, u_seg, NX_PHYS)

        # Config B: nxd=8, pad ANN=0
        x0_b = np.zeros(nxd_b, dtype=DTYPE_NP)
        x0_b[:NX_PHYS] = x0_norm
        y_b = rollout(ic_b, x0_b, u_seg, nxd_b)

        rms_a_ch = np.sqrt(np.mean((y_a[:n_cmp] - y_ref[:n_cmp]) ** 2, axis=0))
        rms_b_ch = np.sqrt(np.mean((y_b[:n_cmp] - y_ref[:n_cmp]) ** 2, axis=0))

        traj_rms_a_ch.append(rms_a_ch)
        traj_rms_b_ch.append(rms_b_ch)

    mean_a = np.mean(traj_rms_a_ch, axis=0)
    mean_b = np.mean(traj_rms_b_ch, axis=0)
    ab_diff = np.abs(mean_a - mean_b).max()
    max_ab_diff = max(max_ab_diff, ab_diff)

    all_rms_a.append(mean_a)
    all_rms_b.append(mean_b)

    print(f'  {label:<40s}  {len(starts):>4d}  '
          f'{mean_a[0]:9.6f} {mean_a[1]:9.6f} {mean_a[2]:9.6f}  '
          f'{mean_b[0]:9.6f} {mean_b[1]:9.6f} {mean_b[2]:9.6f}  '
          f'{ab_diff:9.2e}')

# =========================================================================
# Summary
# =========================================================================

overall_a = np.mean(all_rms_a, axis=0)
overall_b = np.mean(all_rms_b, axis=0)

print(f'\n{"="*70}')
print('SUMMARY')
print(f'{"="*70}')
print(f'  Overall mean RMS per channel (Config A, physics only):')
print(f'    X1: {overall_a[0]:.6f} m    X2: {overall_a[1]:.6f} m    Y: {overall_a[2]:.6f} m')
print(f'  Overall mean RMS per channel (Config B, physics + inert ANN):')
print(f'    X1: {overall_b[0]:.6f} m    X2: {overall_b[1]:.6f} m    Y: {overall_b[2]:.6f} m')
print(f'\n  Max A-vs-B difference: {max_ab_diff:.2e} m')
if max_ab_diff < 1e-6:
    print('  PASS: Config A and B are identical (ANN is truly inert)')
else:
    print('  WARNING: Config A and B differ, inert ANN is not truly inert')

target = 1e-4
print(f'\n  Expected range: ~{target:.0e} m per channel (from MATLAB MSD comparison)')
for ch, lbl in enumerate(CH_LABELS):
    ratio = overall_a[ch] / target
    print(f'    {lbl}: {overall_a[ch]:.2e} m = {ratio:.1f}x target')
