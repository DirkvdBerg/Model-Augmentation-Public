"""
verify_interconnect_normalization.py
------------------------------------
End-to-end smoke test: does the full Interconnect + normalization pipeline
produce correct outputs?

This tests what gantry_baseline_validation.py does NOT test:
  1. Linear_Output_Block with Cd_norm gives correct normalized output
  2. Interconnect.forward() signal routing (selection/expansion matrices)
  3. fit_sys.norm (u0, ustd, y0, ystd) applied to data
  4. The full chain: raw data → norm.transform() → interconnect → denorm → physical y

Test strategy:
  - Build the exact same interconnect as gantry_interconnect_dynamic.py
  - ANN is zero-init → contributes nothing → only physics + output block active
  - Feed known x_logical[0] and u[0:N] through the pipeline
  - Compare against raw Cd @ x_phys (standalone block rollout)
  - Both paths must agree to float precision

Run: conda run -n GraduationProject python scripts/gantry/verification/verify_interconnect_normalization.py
"""

import os
import sys
import numpy as np
import torch
from scipy.io import loadmat

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from model_augmentation.utils.utils import selection_matrix, expansion_matrix
from model_augmentation.utils.torch_nets import zero_init_feed_forward_nn
from model_augmentation.fit_systems.interconnect import Interconnect, SSE_Interconnect
from model_augmentation.fit_systems.blocks import (
    Gantry_State_Block, Linear_Output_Block, Static_ANN_Block,
)
from model_augmentation.systems.gantry_ss import Cd, Dd

# ── Config ────────────────────────────────────────────────────────────────────
NX_PHYS = 6
NX_ANN  = 2
nxd     = NX_PHYS + NX_ANN
nu, ny  = 3, 3
Y_OP    = None      # LPV
N_STEPS = 100       # number of rollout steps to test
PHY_IX  = np.arange(NX_PHYS)

# ── Load data ─────────────────────────────────────────────────────────────────
data_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'gantry', 'matlab')
d = loadmat(os.path.join(data_dir, 'gantry_comb_baseline.mat'), squeeze_me=True)
u_raw = d['u'][:N_STEPS].astype(np.float32)       # (N, 3)
y_raw = d['y'][:N_STEPS].astype(np.float32)        # (N, 3)
x_log = d['x_logical'][:N_STEPS].astype(np.float32)  # (N, 6)

# Use full dataset for normalization (same as gantry_interconnect_dynamic.py)
d_full = loadmat(os.path.join(data_dir, 'gantry_comb_baseline.mat'), squeeze_me=True)
x_all = d_full['x_logical'].astype(np.float32)
u_all = d_full['u'].astype(np.float32)
y_all = d_full['y'].astype(np.float32)

x_mean = x_all.mean(axis=0).reshape(NX_PHYS, 1).astype(np.float32)
std_x  = x_all.std(axis=0).reshape(NX_PHYS, 1).astype(np.float32) + 1e-8
std_u  = u_all.std(axis=0).reshape(nu, 1).astype(np.float32) + 1e-8
ystd   = y_all.std(axis=0).astype(np.float32) + 1e-8
y0     = (Cd.numpy() @ x_mean.flatten()).astype(np.float32)
Cd_np  = Cd.numpy()
Cd_norm = Cd_np * std_x.flatten()[None, :] / ystd[:, None]
Dd_np   = Dd.numpy()

print('=== Normalization parameters ===')
print(f'  x_mean: {x_mean.flatten()}')
print(f'  std_x:  {std_x.flatten()}')
print(f'  std_u:  {std_u.flatten()}')
print(f'  ystd:   {ystd}')
print(f'  y0:     {y0}')

# ── Test 1: Cd_norm algebra ───────────────────────────────────────────────────
# Verify: Cd_norm @ x_norm == (Cd @ x_phys - y0) / ystd
print('\n=== Test 1: Cd_norm algebraic consistency ===')
x_phys_0  = x_log[0]                                    # (6,) physical
x_norm_0  = (x_phys_0 - x_mean.flatten()) / std_x.flatten()   # (6,) normalized

y_via_Cd_norm  = Cd_norm @ x_norm_0                        # normalized output via Cd_norm
y_via_raw      = (Cd_np @ x_phys_0 - y0) / ystd           # normalized output via raw Cd

err_1 = np.abs(y_via_Cd_norm - y_via_raw).max()
print(f'  max|Cd_norm @ x_norm - (Cd @ x_phys - y0)/ystd| = {err_1:.3e}')
assert err_1 < 1e-5, f'FAIL: Cd_norm inconsistency = {err_1}'
print('  PASS')

# ── Test 2: Linear_Output_Block gives correct normalized y ────────────────────
print('\n=== Test 2: Linear_Output_Block forward ===')
output_block = Linear_Output_Block(C=Cd_norm, D=Dd_np)

# Input to output block: [x_norm[0:6], u_norm]  (as interconnect would route it)
x_norm_t = torch.tensor(x_norm_0.reshape(1, NX_PHYS, 1), dtype=torch.float32)
u_norm_0 = u_raw[0] / std_u.flatten()
u_norm_t = torch.tensor(u_norm_0.reshape(1, nu, 1), dtype=torch.float32)
z_out    = torch.cat([x_norm_t, u_norm_t], dim=1)

with torch.no_grad():
    y_norm_block = output_block.forward(z_out).squeeze().numpy()

err_2 = np.abs(y_norm_block - y_via_Cd_norm).max()
print(f'  max|output_block(x_norm, u_norm) - Cd_norm @ x_norm| = {err_2:.3e}')
assert err_2 < 1e-6, f'FAIL: Output block mismatch = {err_2}'
print('  PASS')

# ── Test 3: Full Interconnect forward — single step ──────────────────────────
print('\n=== Test 3: Interconnect.forward() single step ===')

interconnect = Interconnect(nxd, nu, ny, debugging=False)

state_block  = Gantry_State_Block(Y_op=Y_OP, std_x=std_x, std_u=std_u, x_mean=x_mean)
out_block    = Linear_Output_Block(C=Cd_norm, D=Dd_np)
ann_block    = Static_ANN_Block(nz=nxd+nu, nw=nxd, n_nodes_per_layer=64,
                                 net=zero_init_feed_forward_nn, activation=torch.nn.Tanh)

interconnect.add_block(state_block)
interconnect.add_block(out_block)
interconnect.add_block(ann_block)

interconnect.connect_block_signals(ann_block, ["x", "u"], ["xp"])
interconnect.connect_signals("x", state_block, "concat", selection_matrix(PHY_IX, nxd))
interconnect.connect_block_signals(state_block, ["u"], [])
interconnect.connect_signals(state_block, "xp", "additive", expansion_matrix(PHY_IX, nxd))
interconnect.connect_signals("x", out_block, "concat", selection_matrix(PHY_IX, nxd))
interconnect.connect_block_signals(out_block, ["u"], ["y"])

# State: x_norm padded with zeros for ANN states
x_full = torch.zeros(1, nxd, dtype=torch.float32)
x_full[0, :NX_PHYS] = torch.tensor(x_norm_0, dtype=torch.float32)
u_step = torch.tensor(u_norm_0.reshape(1, nu), dtype=torch.float32)

with torch.no_grad():
    y_ic, xp_ic = interconnect.forward(x_full, u_step)

y_norm_ic = y_ic.squeeze().numpy()

# Compare: denormalize interconnect output and check against Cd @ x_phys
y_phys_ic = y_norm_ic * ystd + y0
y_phys_ref = Cd_np @ x_phys_0   # raw physical output at t=0

err_3 = np.abs(y_phys_ic - y_phys_ref).max()
print(f'  max|interconnect_y_phys - Cd @ x_phys| = {err_3:.3e}')
assert err_3 < 1e-5, f'FAIL: Interconnect output mismatch = {err_3}'
print(f'  xp shape: {xp_ic.shape} (expected [1, {nxd}])')
assert xp_ic.shape == (1, nxd), f'FAIL: xp shape {xp_ic.shape}'
print('  PASS')

# ── Test 4: Multi-step rollout — Interconnect vs standalone block ─────────────
print(f'\n=== Test 4: {N_STEPS}-step rollout — Interconnect vs standalone block ===')

# Path A: standalone block rollout (same as gantry_baseline_validation.py)
x_norm_sa = torch.tensor(x_norm_0.reshape(1, NX_PHYS, 1), dtype=torch.float32)
u_norm_all = torch.tensor(
    (u_raw / std_u.flatten()).reshape(N_STEPS, nu, 1), dtype=torch.float32
)
y_standalone = np.zeros((N_STEPS, ny), dtype=np.float32)

with torch.no_grad():
    for t in range(N_STEPS):
        x_phys_sa = x_norm_sa.squeeze().numpy() * std_x.flatten() + x_mean.flatten()
        y_standalone[t] = Cd_np @ x_phys_sa
        z_sa = torch.cat([x_norm_sa, u_norm_all[t:t+1]], dim=1)
        x_norm_sa = state_block.nonlinear_function(z_sa)

# Path B: Interconnect rollout (as SSE_Interconnect would do it)
x_ic = torch.zeros(1, nxd, dtype=torch.float32)
x_ic[0, :NX_PHYS] = torch.tensor(x_norm_0, dtype=torch.float32)
u_ic_all = torch.tensor(
    (u_raw / std_u.flatten()).reshape(N_STEPS, nu), dtype=torch.float32
)
y_interconnect = np.zeros((N_STEPS, ny), dtype=np.float32)

with torch.no_grad():
    for t in range(N_STEPS):
        y_t, x_ic = interconnect.forward(x_ic, u_ic_all[t:t+1])
        y_phys_t = y_t.squeeze().numpy() * ystd + y0
        y_interconnect[t] = y_phys_t

# Compare
err_4 = np.abs(y_interconnect - y_standalone).max(axis=0)
err_4_max = err_4.max()
print(f'  max|y_interconnect - y_standalone| per channel:')
for ch, lbl in enumerate(['X1', 'X2', 'Y ']):
    print(f'    {lbl}: {err_4[ch]:.3e}')
print(f'  overall max: {err_4_max:.3e}')
assert err_4_max < 1e-4, f'FAIL: Rollout mismatch = {err_4_max}'
print('  PASS')

# ── Test 5: Interconnect rollout vs MATLAB reference ──────────────────────────
print(f'\n=== Test 5: {N_STEPS}-step rollout — Interconnect vs MATLAB ===')
err_5 = np.abs(y_interconnect - y_raw).max(axis=0)
nrms_5 = np.sqrt(((y_interconnect - y_raw) ** 2).mean(axis=0)) / ystd
print(f'  max|y_interconnect - y_matlab| per channel:')
for ch, lbl in enumerate(['X1', 'X2', 'Y ']):
    print(f'    {lbl}: max={err_5[ch]:.3e}  NRMS={nrms_5[ch]:.3e}')
print('  (expect same order as gantry_baseline_validation.py LPV results)')

# ── Test 6: fit_sys.norm.transform consistency ────────────────────────────────
print('\n=== Test 6: fit_sys.norm.transform consistency ===')
import deepSI

sys_data = deepSI.System_data(u=u_raw, y=y_raw, dt=1/20000)

# Build a temporary SSE_Interconnect just to test norm
interconnect2 = Interconnect(nxd, nu, ny, debugging=False)
state_block2 = Gantry_State_Block(Y_op=Y_OP, std_x=std_x, std_u=std_u, x_mean=x_mean)
out_block2   = Linear_Output_Block(C=Cd_norm, D=Dd_np)
ann_block2   = Static_ANN_Block(nz=nxd+nu, nw=nxd, n_nodes_per_layer=64,
                                 net=zero_init_feed_forward_nn, activation=torch.nn.Tanh)
interconnect2.add_block(state_block2)
interconnect2.add_block(out_block2)
interconnect2.add_block(ann_block2)
interconnect2.connect_block_signals(ann_block2, ["x", "u"], ["xp"])
interconnect2.connect_signals("x", state_block2, "concat", selection_matrix(PHY_IX, nxd))
interconnect2.connect_block_signals(state_block2, ["u"], [])
interconnect2.connect_signals(state_block2, "xp", "additive", expansion_matrix(PHY_IX, nxd))
interconnect2.connect_signals("x", out_block2, "concat", selection_matrix(PHY_IX, nxd))
interconnect2.connect_block_signals(out_block2, ["u"], ["y"])

fit_sys = SSE_Interconnect(interconnect=interconnect2, na=20, nb=20,
                            e_net_kwargs={"n_nodes_per_layer": 16, "n_hidden_layers": 1})
fit_sys.norm.u0   = np.zeros(nu, dtype=np.float32)
fit_sys.norm.ustd = std_u.flatten()
fit_sys.norm.y0   = y0
fit_sys.norm.ystd = ystd

# Transform data through norm
normed = fit_sys.norm.transform(sys_data)

# Check u normalization: u_norm = (u - u0) / ustd = u / std_u (since u0=0)
u_norm_expected = u_raw / std_u.flatten()
err_6u = np.abs(normed.u - u_norm_expected).max()
print(f'  max|norm.transform(u) - u/std_u| = {err_6u:.3e}')
assert err_6u < 1e-6, f'FAIL: u norm mismatch = {err_6u}'

# Check y normalization: y_norm = (y - y0) / ystd
y_norm_expected = (y_raw - y0) / ystd
err_6y = np.abs(normed.y - y_norm_expected).max()
print(f'  max|norm.transform(y) - (y-y0)/ystd| = {err_6y:.3e}')
assert err_6y < 1e-6, f'FAIL: y norm mismatch = {err_6y}'
print('  PASS')

# ══════════════════════════════════════════════════════════════════════════════
# Option B: u_mean in the block, u0 = mean(u) in norm
# ══════════════════════════════════════════════════════════════════════════════
# This tests the generalised normalisation where fit_sys.norm.u0 = mean(u)
# and the block receives u_mean so it can recover physical forces:
#   u_norm = (u_raw - u0) / ustd          (done by fit_sys.norm)
#   u_phys = u_norm * std_u + u_mean      (done by block.deriv)
# When u0 == u_mean, u_phys == u_raw — works for any dataset, not just zero-mean.

u_mean_val = u_all.mean(axis=0).reshape(nu, 1).astype(np.float32)
print(f'\n{"="*70}')
print(f'Option B: u_mean = {u_mean_val.flatten()}')
print(f'{"="*70}')

# ── Test 7: Option B — single-step block matches Option A ───────────────────
print('\n=== Test 7: Option B — single-step block vs Option A ===')
state_block_B = Gantry_State_Block(Y_op=Y_OP, std_x=std_x, std_u=std_u,
                                    x_mean=x_mean, u_mean=u_mean_val)

# Normalise u with u0 = mean(u) (as fit_sys.norm would)
u_norm_B_0 = (u_raw[0] - u_mean_val.flatten()) / std_u.flatten()
u_norm_B_t = torch.tensor(u_norm_B_0.reshape(1, nu, 1), dtype=torch.float32)
z_B = torch.cat([x_norm_t, u_norm_B_t], dim=1)   # x_norm_t same as before

with torch.no_grad():
    xp_B = state_block_B.nonlinear_function(z_B)

# Option A block (u_mean=0, u_norm = u_raw / std_u)
z_A = torch.cat([x_norm_t, torch.tensor(u_norm_0.reshape(1, nu, 1), dtype=torch.float32)], dim=1)
with torch.no_grad():
    xp_A = state_block.nonlinear_function(z_A)

err_7 = (xp_B - xp_A).abs().max().item()
print(f'  max|xp_B - xp_A| = {err_7:.3e}')
assert err_7 < 1e-6, f'FAIL: Option B single-step mismatch = {err_7}'
print('  PASS — both options recover the same physical forces')

# ── Test 8: Option B — Interconnect rollout matches Option A rollout ─────────
print(f'\n=== Test 8: Option B — {N_STEPS}-step Interconnect rollout ===')

interconnect_B = Interconnect(nxd, nu, ny, debugging=False)
state_block_B2 = Gantry_State_Block(Y_op=Y_OP, std_x=std_x, std_u=std_u,
                                     x_mean=x_mean, u_mean=u_mean_val)
out_block_B    = Linear_Output_Block(C=Cd_norm, D=Dd_np)
ann_block_B    = Static_ANN_Block(nz=nxd+nu, nw=nxd, n_nodes_per_layer=64,
                                   net=zero_init_feed_forward_nn, activation=torch.nn.Tanh)
interconnect_B.add_block(state_block_B2)
interconnect_B.add_block(out_block_B)
interconnect_B.add_block(ann_block_B)
interconnect_B.connect_block_signals(ann_block_B, ["x", "u"], ["xp"])
interconnect_B.connect_signals("x", state_block_B2, "concat", selection_matrix(PHY_IX, nxd))
interconnect_B.connect_block_signals(state_block_B2, ["u"], [])
interconnect_B.connect_signals(state_block_B2, "xp", "additive", expansion_matrix(PHY_IX, nxd))
interconnect_B.connect_signals("x", out_block_B, "concat", selection_matrix(PHY_IX, nxd))
interconnect_B.connect_block_signals(out_block_B, ["u"], ["y"])

# Rollout with u_norm = (u_raw - u_mean) / std_u
x_ic_B = torch.zeros(1, nxd, dtype=torch.float32)
x_ic_B[0, :NX_PHYS] = torch.tensor(x_norm_0, dtype=torch.float32)
u_ic_B = torch.tensor(
    ((u_raw - u_mean_val.flatten()) / std_u.flatten()).reshape(N_STEPS, nu),
    dtype=torch.float32,
)
y_interconnect_B = np.zeros((N_STEPS, ny), dtype=np.float32)

with torch.no_grad():
    for t in range(N_STEPS):
        y_t_B, x_ic_B = interconnect_B.forward(x_ic_B, u_ic_B[t:t+1])
        y_interconnect_B[t] = y_t_B.squeeze().numpy() * ystd + y0

err_8 = np.abs(y_interconnect_B - y_interconnect).max(axis=0)
err_8_max = err_8.max()
print(f'  max|y_B - y_A| per channel:')
for ch, lbl in enumerate(['X1', 'X2', 'Y ']):
    print(f'    {lbl}: {err_8[ch]:.3e}')
print(f'  overall max: {err_8_max:.3e}')
assert err_8_max < 1e-4, f'FAIL: Option B rollout mismatch = {err_8_max}'
print('  PASS — Option B produces identical physical outputs')

# ── Test 9: Option B — fit_sys.norm.transform + block roundtrip ──────────────
print('\n=== Test 9: Option B — norm.transform consistency ===')

interconnect_B2 = Interconnect(nxd, nu, ny, debugging=False)
state_block_B3  = Gantry_State_Block(Y_op=Y_OP, std_x=std_x, std_u=std_u,
                                      x_mean=x_mean, u_mean=u_mean_val)
out_block_B2    = Linear_Output_Block(C=Cd_norm, D=Dd_np)
ann_block_B2    = Static_ANN_Block(nz=nxd+nu, nw=nxd, n_nodes_per_layer=64,
                                    net=zero_init_feed_forward_nn, activation=torch.nn.Tanh)
interconnect_B2.add_block(state_block_B3)
interconnect_B2.add_block(out_block_B2)
interconnect_B2.add_block(ann_block_B2)
interconnect_B2.connect_block_signals(ann_block_B2, ["x", "u"], ["xp"])
interconnect_B2.connect_signals("x", state_block_B3, "concat", selection_matrix(PHY_IX, nxd))
interconnect_B2.connect_block_signals(state_block_B3, ["u"], [])
interconnect_B2.connect_signals(state_block_B3, "xp", "additive", expansion_matrix(PHY_IX, nxd))
interconnect_B2.connect_signals("x", out_block_B2, "concat", selection_matrix(PHY_IX, nxd))
interconnect_B2.connect_block_signals(out_block_B2, ["u"], ["y"])

fit_sys_B = SSE_Interconnect(interconnect=interconnect_B2, na=20, nb=20,
                              e_net_kwargs={"n_nodes_per_layer": 16, "n_hidden_layers": 1})
fit_sys_B.norm.u0   = u_mean_val.flatten()
fit_sys_B.norm.ustd = std_u.flatten()
fit_sys_B.norm.y0   = y0
fit_sys_B.norm.ystd = ystd

normed_B = fit_sys_B.norm.transform(sys_data)

# u_norm = (u - u_mean) / std_u
u_norm_B_expected = (u_raw - u_mean_val.flatten()) / std_u.flatten()
err_9u = np.abs(normed_B.u - u_norm_B_expected).max()
print(f'  max|norm_B.transform(u) - (u-u_mean)/std_u| = {err_9u:.3e}')
assert err_9u < 1e-6, f'FAIL: Option B u norm mismatch = {err_9u}'

# y normalization unchanged
err_9y = np.abs(normed_B.y - y_norm_expected).max()
print(f'  max|norm_B.transform(y) - (y-y0)/ystd| = {err_9y:.3e}')
assert err_9y < 1e-6, f'FAIL: Option B y norm mismatch = {err_9y}'
print('  PASS — Option B norm.transform is consistent')

# ── Summary ───────────────────────────────────────────────────────────────────
print('\n=== ALL TESTS PASSED ===')
print('Option A (u0=0, no u_mean) and Option B (u0=mean(u), u_mean in block)')
print('both produce identical physical outputs through the full pipeline.')
print('Option B generalises to datasets where forces are NOT zero-mean.')
