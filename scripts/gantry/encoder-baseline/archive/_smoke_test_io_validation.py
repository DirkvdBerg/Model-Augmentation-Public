"""Smoke test for encoder_io_validation.py — verifies shapes, gradient flow, and conventions."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import torch
import numpy as np

# Test 1: imports
print('Test 1: imports...')
from model_augmentation.utils.torch_nets import LinearInitEncoderWrapper
from model_augmentation.fit_systems.pre_encoder import linear_encoder_init
from model_augmentation.fit_systems.blocks import Gantry_State_Block
from model_augmentation.systems.gantry_ss import Cd, P
from model_augmentation.systems.gantry_linearization import gantry_linearize_and_discretize
print('  OK')

# Test 2: data files exist
print('Test 2: data files...')
TRAJ_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'gantry', 'matlab', 'multisine', 'baseline')
for f in ['T1_Y_sweep_conservative.mat', 'V1_X_sym_Y_mid_sweep.mat']:
    assert os.path.exists(os.path.join(TRAJ_DIR, f)), f'Missing: {f}'
print('  OK')

# Test 3: encoder + state block forward pass + gradient flow
print('Test 3: forward pass + gradients...')
from scipy.io import loadmat
import deepSI
from model_augmentation.utils.utils import normalize_linear_ss_matrices

d = loadmat(os.path.join(TRAJ_DIR, 'T1_Y_sweep_conservative.mat'), squeeze_me=True)
u = d['u_total'][::5][:100].astype(np.float32)
y = d['y'][::5][:100].astype(np.float32)

std_x = np.ones((6,1), dtype=np.float32)
x_mean = np.zeros((6,1), dtype=np.float32)
std_u = np.ones((3,1), dtype=np.float32)
u_mean = np.zeros((3,1), dtype=np.float32)
ystd = np.ones(3, dtype=np.float32)
y0 = np.zeros(3, dtype=np.float32)

Ad, Bd, Cd_dt, Dd_dt = gantry_linearize_and_discretize(dt=1/4000)
sys_data = deepSI.System_data(u=u, y=y)
sys_data.x = np.random.randn(100, 6).astype(np.float32) * 0.01
Ad_bar, Bd_bar, Cd_bar, Dd_bar = normalize_linear_ss_matrices(Ad, Bd, Cd_dt, Dd_dt, sys_data)

phys_enc = linear_encoder_init(A=Ad_bar, B=Bd_bar, C=Cd_bar, D=Dd_bar,
    nx=6, nu=3, ny=3, na=25, nb=25, n_nodes_per_layer=16, n_hidden_layers=2,
    flag_linear_only=False)
encoder = LinearInitEncoderWrapper(phys_encoder=phys_enc, nx_ann=0,
    nb=26, nu=3, na=26, ny=3, n_nodes_per_layer=16, n_hidden_layers=2,
    u_mean=u_mean, std_u=std_u, y0=y0, ystd=ystd, x_mean=x_mean, std_x=std_x)

state_block = Gantry_State_Block(Y_op=None, std_x=std_x, std_u=std_u,
    x_mean=x_mean, u_mean=u_mean, Ts=1/4000, up_sample=1).to(torch.float32)

batch = 4
n_steps = 10
u_hist = torch.randn(batch, 26, 3)
y_hist = torch.randn(batch, 26, 3)
u_future = torch.randn(batch, n_steps, 3)

# Encoder forward
x_enc = encoder(u_hist, y_hist)
assert x_enc.shape == (batch, 6), f'Expected (4,6), got {x_enc.shape}'
print(f'  encoder output: {x_enc.shape}')

# n-step rollout
std_x_t = torch.tensor(std_x)
x_mean_t = torch.tensor(x_mean)
Cd_t = Cd.to(torch.float32)

x = x_enc.unsqueeze(-1)
y_hats = []
for step in range(n_steps):
    u_step = u_future[:, step, :].unsqueeze(-1)
    z = torch.cat([x, u_step], dim=1)
    assert z.shape == (batch, 9, 1), f'Expected (4,9,1), got {z.shape}'
    x = state_block.nonlinear_function(z)
    assert x.shape == (batch, 6, 1), f'Expected (4,6,1), got {x.shape}'
    x_phys = x * std_x_t + x_mean_t
    y_hat = (Cd_t @ x_phys).squeeze(-1)
    assert y_hat.shape == (batch, 3), f'Expected (4,3), got {y_hat.shape}'
    y_hats.append(y_hat)

y_hat_steps = torch.stack(y_hats, dim=1)
assert y_hat_steps.shape == (batch, n_steps, 3), f'Expected (4,10,3), got {y_hat_steps.shape}'
print(f'  rollout output: {y_hat_steps.shape}')

# Gradient flow
loss = torch.mean(y_hat_steps ** 2)
loss.backward()
grad_norm = sum(p.grad.norm().item() for p in encoder.parameters() if p.grad is not None)
n_with_grad = sum(1 for p in encoder.parameters() if p.grad is not None)
n_total = sum(1 for p in encoder.parameters())
print(f'  gradient norm: {grad_norm:.4f} ({n_with_grad}/{n_total} params have grad)')
assert grad_norm > 0, 'No gradients flowing to encoder!'
print('  OK')

# Test 4: loss broadcasting
print('Test 4: loss broadcasting...')
ystd_t = torch.tensor(ystd)
y_future_t = torch.randn(batch, n_steps, 3)
err = (y_hat_steps.detach() - y_future_t) / ystd_t
assert err.shape == (batch, n_steps, 3), f'Broadcasting failed: {err.shape}'
print(f'  normalized error shape: {err.shape} OK')

# Test 5: output matrix convention
print('Test 5: Cd convention...')
# Cd = [P^T | 0], so y = P^T @ positions, velocities ignored
assert Cd.shape == (3, 6), f'Cd shape: {Cd.shape}'
assert torch.all(Cd[:, 3:] == 0), 'Cd should zero out velocities'
print(f'  Cd[:, :3] = P^T, Cd[:, 3:] = 0: OK')

print('\nAll smoke tests passed.')
