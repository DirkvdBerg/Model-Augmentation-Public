"""
verify_encoder_normalization.py
-------------------------------
Normalization diagnostic for diagnose_encoder.py.

Uses the EXACT same data loading, normalization, and model construction as
diagnose_encoder.py (trajectory files, decimation, finite-diff velocities).
No training. Zero-init ANN (inert).

Tests:
  1. Output at rest (pure zero physical state) through hfn
  2. Output at a known physical state vs Cd @ x_phys
  3. analytical_x0_at + normalize_x round-trip through output block
  4. Cd_norm algebraic consistency with trajectory-derived stats
  5. rollout_from_x0 first-step output vs y_ref
  6. Print normalization constants for sanity checking

Run: conda run -n GraduationProject python scripts/gantry/verification/verify_encoder_normalization.py
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
# Configuration (copied from diagnose_encoder.py)
# =========================================================================

NX_PHYS = 6
NX_ANN  = 2
nxd     = NX_PHYS + NX_ANN
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

# =========================================================================
# Data loading (copied from diagnose_encoder.py)
# =========================================================================

TRAJ_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..',
                        'data', 'gantry', 'matlab', 'trajectories')

TRAIN_FILES = [
    'T1_Y_sweep_conservative.mat', 'T2_X_sym_Y030.mat',
    'T3_X_sym_Y000.mat', 'T4_X_antisym_Y020.mat',
    'T5_X_sym_Y_sweep.mat', 'T6_Y_sweep_aggressive.mat',
    'T7_X_antisym_Y_sweep.mat', 'T8_X_sym_anti_Y_sweep.mat',
]

train_list = []
for f in TRAIN_FILES:
    d = loadmat(os.path.join(TRAJ_DIR, f), squeeze_me=True)
    train_list.append({
        'u': d['u'][::D].astype(DTYPE_NP),
        'y': d['y'][::D].astype(DTYPE_NP),
    })

print(f'Loaded {len(train_list)} training trajectories')

# =========================================================================
# Normalisation (copied from diagnose_encoder.py)
# =========================================================================

u_all = np.concatenate([t['u'] for t in train_list])
y_all = np.concatenate([t['y'] for t in train_list])

fs = FS_NEW
P_inv_T = np.linalg.inv(P.numpy().T).astype(DTYPE_NP)  # stage -> logical
x_logical_list = []
for t in train_list:
    pos_logical = (P_inv_T @ t['y'].T).T        # (N, 3) stage -> logical
    vel_logical = np.diff(pos_logical, axis=0) * fs  # (N-1, 3)
    vel_logical = np.vstack([vel_logical[:1], vel_logical])  # (N, 3)
    x_logical_list.append(np.hstack([pos_logical, vel_logical]))  # (N, 6)
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
# Functions from diagnose_encoder.py
# =========================================================================

def analytical_x0_at(y_seq, t_idx, dt):
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

# =========================================================================
# Build model (copied from diagnose_encoder.py, no training)
# =========================================================================

ic = Interconnect(nxd, nu, ny, debugging=False)

phy_block = Gantry_State_Block(
    Y_op=Y_OP, std_x=std_x, std_u=std_u,
    x_mean=x_mean, u_mean=u_mean, Ts=TS_NEW,
).to(DTYPE_PT)
out_block = Linear_Output_Block(C=Cd_norm, D=Dd_np)
ic.add_block(phy_block)
ic.add_block(out_block)

ann_block = Static_ANN_Block(
    nz=nxd + nu, nw=nxd,
    n_nodes_per_layer=128,
    n_hidden_layers=3,
    net=zero_init_feed_forward_nn,
    activation=torch.nn.Tanh,
)
ic.add_block(ann_block)

ic.connect_block_signals(ann_block, ["x", "u"], ["xp"])
ic.connect_signals("x", phy_block, "concat", selection_matrix(PHY_IX, nxd))
ic.connect_block_signals(phy_block, ["u"], [])
ic.connect_signals(phy_block, "xp", "additive", expansion_matrix(PHY_IX, nxd))
ic.connect_signals("x", out_block, "concat", selection_matrix(PHY_IX, nxd))
ic.connect_block_signals(out_block, ["u"], ["y"])

n_pass = 0
n_fail = 0

def report(name, err, tol, detail=''):
    global n_pass, n_fail
    if err < tol:
        print(f'  PASS (err={err:.3e}, tol={tol:.3e})')
        n_pass += 1
    else:
        print(f'  ** FAIL ** (err={err:.3e}, tol={tol:.3e})')
        n_fail += 1
    if detail:
        print(f'  {detail}')


# =========================================================================
# Test 6 (first): Print normalization constants
# =========================================================================

print(f'\n{"="*70}')
print('Normalization constants (from trajectory files + finite-diff)')
print(f'{"="*70}')
print(f'  x_mean (pos):  {x_mean.flatten()[:3]}')
print(f'  x_mean (vel):  {x_mean.flatten()[3:]}')
print(f'  std_x  (pos):  {std_x.flatten()[:3]}')
print(f'  std_x  (vel):  {std_x.flatten()[3:]}')
print(f'  std_u:         {std_u.flatten()}')
print(f'  u_mean:        {u_mean.flatten()}')
print(f'  ystd:          {ystd}')
print(f'  y0 = Cd@x_mean: {y0}')
print(f'  y_all.mean():  {y_all.mean(axis=0)}')
print(f'  Dd:            {Dd_np}')
print(f'  Cd_norm:\n{Cd_norm}')

# Check: y0 should equal mean of measured y if Cd picks positions
y_mean_measured = y_all.mean(axis=0)
y0_vs_ymean = np.abs(y0 - y_mean_measured).max()
print(f'\n  |y0 - mean(y_measured)| = {y0_vs_ymean:.6f} m')
if y0_vs_ymean > 0.01:
    print(f'  WARNING: y0 and mean(y) differ by {y0_vs_ymean:.4f} m.')
    print(f'  This means Cd@x_mean != mean(Cd@x), i.e. the "x" used for')
    print(f'  normalization (stage pos + finite-diff vel) may not be consistent')
    print(f'  with the output equation y = Cd @ x.')

# =========================================================================
# Test 1: Output at rest (pure zero physical state)
# =========================================================================

print(f'\n{"="*70}')
print('Test 1: Output at rest (x_phys = 0, u = 0)')
print(f'{"="*70}')

x_phys_zero = np.zeros(NX_PHYS, dtype=DTYPE_NP)
x_norm_zero = normalize_x(x_phys_zero)
x_full_zero = np.zeros(nxd, dtype=DTYPE_NP)
x_full_zero[:NX_PHYS] = x_norm_zero

# u=0 in physical space, but norm.transform does (u - u_mean) / std_u
# So to get u_phys=0, we need u_norm = (0 - u_mean) / std_u
u_phys_zero = np.zeros(nu, dtype=DTYPE_NP)
u_norm_zero = (u_phys_zero - u_mean.flatten()) / std_u.flatten()

x_t = torch.tensor(x_full_zero.reshape(1, nxd), dtype=DTYPE_PT)
u_t = torch.tensor(u_norm_zero.reshape(1, nu), dtype=DTYPE_PT)

with torch.no_grad():
    y_norm_out, xp = ic.forward(x_t, u_t)

y_norm_out_np = y_norm_out.squeeze().numpy()
y_phys_out = y_norm_out_np * ystd + y0

print(f'  x_norm_zero:   {x_norm_zero}')
print(f'  u_norm_zero:   {u_norm_zero}')
print(f'  y_norm (hfn):  {y_norm_out_np}')
print(f'  y_phys (denorm): {y_phys_out}')
print(f'  expected:      [0, 0, 0]')

err_1 = np.abs(y_phys_out).max()
report('Test 1', err_1, 1e-4)

# =========================================================================
# Test 2: Output at a known physical state
# =========================================================================

print(f'\n{"="*70}')
print('Test 2: Output at known physical state (first data point of T3)')
print(f'{"="*70}')

# T3 = T3_X_sym_Y000.mat, Y=0 trajectory, pick a mid-point
traj = train_list[2]  # T3
t_idx = 500  # some point with known positions
y_ref = traj['y'][t_idx]

x_phys = analytical_x0_at(traj['y'], t_idx, TS_NEW)
x_norm = normalize_x(x_phys)
x_full = np.zeros(nxd, dtype=DTYPE_NP)
x_full[:NX_PHYS] = x_norm

# For the output, we only need x (Dd=0), but provide u anyway
u_phys = traj['u'][t_idx]
u_norm = (u_phys - u_mean.flatten()) / std_u.flatten()

x_t = torch.tensor(x_full.reshape(1, nxd), dtype=DTYPE_PT)
u_t = torch.tensor(u_norm.reshape(1, nu), dtype=DTYPE_PT)

with torch.no_grad():
    y_norm_out, _ = ic.forward(x_t, u_t)

y_phys_out = y_norm_out.squeeze().numpy() * ystd + y0

# Reference: Cd @ x_phys (only positions matter, Cd has zeros for velocities)
y_ref_Cd = Cd_np @ x_phys

print(f'  y_ref (measured):   {y_ref}')
print(f'  y_phys (hfn+denorm): {y_phys_out}')
print(f'  Cd @ x_phys:        {y_ref_Cd}')
print(f'  |hfn - measured|:   {np.abs(y_phys_out - y_ref).max():.6f} m')
print(f'  |hfn - Cd@x_phys|: {np.abs(y_phys_out - y_ref_Cd).max():.6f} m')

# hfn output should match Cd @ x_phys (output block is purely C@x + D@u)
err_2a = np.abs(y_phys_out - y_ref_Cd).max()
report('Test 2a (hfn vs Cd@x_phys)', err_2a, 1e-4)

# Cd @ x_phys should match measured y (positions from y_seq)
# Since x_phys[:3] = y_ref and Cd picks positions via P^T:
# Cd @ x_phys = P^T @ [pos_stage] ... but pos are in stage coordinates
# so Cd @ [stage_pos; vel] = P^T @ stage_pos
err_2b = np.abs(y_ref_Cd - y_ref).max()
print(f'\n  Check: |Cd @ x_phys - y_measured| = {err_2b:.6f} m')
if err_2b > 0.001:
    print(f'  WARNING: Cd @ [stage_pos; vel] != stage_pos.')
    print(f'  This means analytical_x0_at puts positions in a space')
    print(f'  that is not consistent with the output equation y = Cd @ x.')
    print(f'  Cd expects logical coordinates, but positions are stage coordinates.')
    print(f'  Cd[:,:3] = P^T, so Cd @ [stage_pos; vel] = P^T @ stage_pos != stage_pos')
    print(f'  (unless P^T = I, which it is NOT for the gantry)')
report('Test 2b (Cd@x_phys vs measured)', err_2b, 1e-3)

# =========================================================================
# Test 3: analytical_x0_at + normalize_x round-trip through output block
# =========================================================================

print(f'\n{"="*70}')
print('Test 3: Output block only (bypass dynamics)')
print(f'{"="*70}')

# Directly test: Cd_norm @ x_norm * ystd + y0 == Cd @ x_phys
x_norm_3 = normalize_x(x_phys)
y_via_norm_chain = Cd_norm @ x_norm_3 * ystd + y0
y_via_raw = Cd_np @ x_phys

print(f'  y (norm chain):  {y_via_norm_chain}')
print(f'  y (Cd @ x_phys): {y_via_raw}')

err_3 = np.abs(y_via_norm_chain - y_via_raw).max()
report('Test 3 (norm chain round-trip)', err_3, 1e-5)

# =========================================================================
# Test 4: Cd_norm algebraic consistency
# =========================================================================

print(f'\n{"="*70}')
print('Test 4: Cd_norm algebraic consistency')
print(f'{"="*70}')

# Cd_norm @ x_norm should equal (Cd @ x_phys - y0) / ystd
y_via_Cd_norm = Cd_norm @ x_norm_3
y_via_raw_norm = (Cd_np @ x_phys - y0) / ystd

print(f'  Cd_norm @ x_norm:           {y_via_Cd_norm}')
print(f'  (Cd @ x_phys - y0) / ystd:  {y_via_raw_norm}')

err_4 = np.abs(y_via_Cd_norm - y_via_raw_norm).max()
report('Test 4', err_4, 1e-5)

# =========================================================================
# Test 5: rollout_from_x0 first-step output (full chain as diagnose_encoder uses it)
# =========================================================================

print(f'\n{"="*70}')
print('Test 5: Full rollout_from_x0 chain (1 step)')
print(f'{"="*70}')

# Replicate exactly what diagnose_encoder does:
# 1. norm.transform the data
import deepSI

# Build a System_data for T3
sys_data_t3 = deepSI.System_data(
    u=traj['u'], y=traj['y'], dt=TS_NEW,
)

# Create norm object matching diagnose_encoder.py
from model_augmentation.utils.deepSI_corrections import fixed_System_data_norm
norm = fixed_System_data_norm()
norm.u0   = u_mean.flatten()
norm.ustd = std_u.flatten()
norm.y0   = y0
norm.ystd = ystd

data_norm = norm.transform(sys_data_t3)

# 2. Build x0 analytically (same as diagnose_encoder.py multi_window_rollouts)
x0_ana_phys = analytical_x0_at(traj['y'], t_idx, TS_NEW)
x0_ana_norm = normalize_x(x0_ana_phys)
x0_ana_full = np.zeros(nxd, dtype=DTYPE_NP)
x0_ana_full[:NX_PHYS] = x0_ana_norm

# 3. Rollout 1 step (replicating rollout_from_x0)
x = torch.tensor(x0_ana_full.reshape(1, -1), dtype=DTYPE_PT)
u_norm_step = torch.tensor(
    np.ascontiguousarray(data_norm.u[t_idx:t_idx + 1]),
    dtype=DTYPE_PT,
)

with torch.no_grad():
    y_t, x_next = ic.forward(x, u_norm_step)

y_hat_norm = y_t.squeeze().numpy()
y_hat = y_hat_norm * ystd + y0  # diagnose_encoder.py line 275

y_ref_step = traj['y'][t_idx]

print(f'  y_hat (rollout):  {y_hat}')
print(f'  y_ref (measured): {y_ref_step}')
print(f'  |y_hat - y_ref|:  {np.abs(y_hat - y_ref_step)}')

err_5 = np.abs(y_hat - y_ref_step).max()
report('Test 5', err_5, 1e-3, f'Offset = {err_5:.6f} m')

# =========================================================================
# Extra: Check if the issue is Cd @ stage_pos != stage_pos
# =========================================================================

print(f'\n{"="*70}')
print('Coordinate analysis: is analytical_x0_at in the right space for Cd?')
print(f'{"="*70}')

print(f'\n  Cd[:, :3] (should map positions to output):')
print(f'  {Cd_np[:, :3]}')
print(f'\n  Cd[:, 3:] (velocity part, should be zero):')
print(f'  {Cd_np[:, 3:]}')

# If Cd[:,:3] = P^T and positions are in stage coordinates,
# then Cd @ [stage_pos; vel] = P^T @ stage_pos, which is NOT stage_pos
# unless P^T = I.
from model_augmentation.systems.gantry_ss import P
P_np = P.numpy() if hasattr(P, 'numpy') else np.array(P)
print(f'\n  P^T (coordinate transform logical->stage):')
print(f'  {P_np.T}')
print(f'\n  Is P^T == I? {np.allclose(P_np.T, np.eye(3))}')

if not np.allclose(P_np.T, np.eye(3)):
    # Show what happens: Cd @ [stage_pos; 0] vs stage_pos
    test_pos = np.array([0.01, -0.01, 0.02], dtype=DTYPE_NP)
    x_test = np.zeros(6, dtype=DTYPE_NP)
    x_test[:3] = test_pos
    y_Cd = Cd_np @ x_test
    print(f'\n  Example: stage_pos = {test_pos}')
    print(f'  Cd @ [stage_pos; 0] = P^T @ stage_pos = {y_Cd}')
    print(f'  These are NOT equal because P^T != I')
    print(f'\n  This means analytical_x0_at (which puts stage positions')
    print(f'  directly into x_phys[:3]) is INCOMPATIBLE with Cd,')
    print(f'  which expects logical positions in x[:3].')
    print(f'\n  The gantry state vector is x = [q_logical; qdot_logical]')
    print(f'  but analytical_x0_at fills it with [y_stage; dy_stage/dt]')


# =========================================================================
# Summary
# =========================================================================

print(f'\n{"="*70}')
print(f'SUMMARY: {n_pass} passed, {n_fail} failed')
print(f'{"="*70}')
if n_fail > 0:
    print('  Check the FAIL and WARNING messages above for the root cause.')
