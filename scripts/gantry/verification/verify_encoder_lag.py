"""
verify_encoder_lag.py
---------------------
Self-contained diagnostic: does the hybrid encoder's physical branch align
to x(k) or x(k-1)?

The hybrid encoder computes positions and velocities analytically from the
output history window. deepSI's default convention (na_right=0) means ypast
ends at y[k-1], so the encoder's "last sample" is one step behind the rollout
start at k. This script proves/disproves the one-sample lag without any
training -- the physical branch is deterministic and detached.

Method:
  1. Build the hybrid encoder (no training needed).
  2. Construct history windows exactly as deepSI does (na_right=0).
  3. Run the encoder to get x_hat_phys (6 physical states, normalized).
  4. Reconstruct x_true from data at time k and k-1 (same FD formula).
  5. Compare x_hat against x_true(k) vs x_true(k-1).

Expected result:
  - x_hat matches x_true(k-1) to float32 precision (< 1e-5)
  - x_hat vs x_true(k) shows systematic O(v*Ts) position error
    and O(a*1.5*Ts) velocity error

Usage:
  python verify_encoder_lag.py
"""

import os
import sys
import numpy as np
import torch
from scipy.io import loadmat

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from model_augmentation.utils.torch_nets import HybridGantryEncoder
from model_augmentation.systems.gantry_ss import Cd, P

# =========================================================================
# Configuration (matches gantry_interconnect_dynamic.py)
# =========================================================================

NX_PHYS = 6
nu, ny = 3, 3
SEED = 42

FS_ORIG = 20000
FS_NEW  = 4000
D       = FS_ORIG // FS_NEW
TS_NEW  = 1.0 / FS_NEW

USE_F64  = False
DTYPE_NP = np.float64 if USE_F64 else np.float32
DTYPE_PT = torch.float64 if USE_F64 else torch.float32

NX_ANN = 2
nxd = NX_PHYS + NX_ANN
na = 120   # encoder history length (samples); exact value doesn't matter, just needs to be >= 2
nb = 120

# =========================================================================
# Data loading (identical to gantry_interconnect_dynamic.py)
# =========================================================================

np.random.seed(SEED)
torch.manual_seed(SEED)

MODE = 'multisine'
DATA_SUBDIR = 'multisine' if MODE == 'multisine' else 'trajectories'
TRAJ_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..',
                        'data', 'gantry', 'matlab', DATA_SUBDIR)

TRAIN_FILES = [
    'T1_Y_sweep_conservative.mat', 'T2_X_sym_Y030.mat',
    'T3_X_sym_Y000.mat', 'T4_X_antisym_Y020.mat',
    'T5_X_sym_Y_sweep.mat', 'T6_Y_sweep_aggressive.mat',
    'T7_X_antisym_Y_sweep.mat', 'T8_X_sym_anti_Y_sweep.mat',
]
VAL_FILE = 'V1_X_sym_Y_mid_sweep.mat'


def _load_u(d):
    if 'u_total' in d:
        return d['u_total']
    return d['u']


def load_traj(filename):
    d = loadmat(os.path.join(TRAJ_DIR, filename), squeeze_me=True)
    u = _load_u(d)[::D].astype(DTYPE_NP)
    y = d['y'][::D].astype(DTYPE_NP)
    return u, y


print(f'Data dir ({MODE}): {TRAJ_DIR}')
train_u, train_y = zip(*[load_traj(f) for f in TRAIN_FILES])
val_u, val_y = load_traj(VAL_FILE)
print(f'Loaded {len(TRAIN_FILES)} train + 1 val.  Val shape: u={val_u.shape}, y={val_y.shape}')

# =========================================================================
# Normalisation (identical to gantry_interconnect_dynamic.py)
# =========================================================================

u_all = np.concatenate(train_u)
y_all = np.concatenate(train_y)

fs = FS_NEW
P_inv_T = np.linalg.inv(P.numpy().T).astype(DTYPE_NP)

x_logical_list = []
for yi in train_y:
    pos_logical = (P_inv_T @ yi.T).T
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

# =========================================================================
# Reconstruct "true" physical states for the validation trajectory
# =========================================================================

pos_val = (P_inv_T @ val_y.T).T                             # (N,3) THEORY: q = inv(P^T) y
vel_val = np.diff(pos_val, axis=0) * fs                     # HEURISTIC: backward FD at fs=4000
vel_val = np.vstack([vel_val[:1], vel_val])                  # (N,3)
x_true = np.hstack([pos_val, vel_val]).astype(DTYPE_NP)      # (N,6)
x_true_norm = (x_true - x_mean.flatten()) / std_x.flatten() # (N,6)

# =========================================================================
# Build hybrid encoder (no training -- physical branch is deterministic)
# =========================================================================

encoder = HybridGantryEncoder(
    nb=nb, nu=nu, na=na, ny=ny, nx=nxd,
    P_inv_T=P_inv_T, y0=y0, ystd=ystd,
    x_mean=x_mean.flatten(), std_x=std_x.flatten(),
    fs=fs, NX_PHYS=NX_PHYS,
    n_nodes_per_layer=8, n_hidden_layers=1,
).to(DTYPE_PT)
encoder.eval()

# =========================================================================
# Construct windows and run encoder (deepSI convention: na_right=0)
# =========================================================================

# Normalize val data the same way deepSI does
yn = ((val_y - y0) / ystd).astype(DTYPE_NP)
un = ((val_u - u_mean.flatten()) / std_u.flatten()).astype(DTYPE_NP)

N = len(yn)
k0 = max(na, nb) + 1   # +1 so k-1 reference exists
max_windows = 2000
stride = max(1, (N - k0) // max_windows)   # HEURISTIC: cap windows to bound memory
k_ix = np.arange(k0, N, stride)
n_windows = len(k_ix)

# deepSI convention (na_right=0): ypast = y[k-na : k], last sample is y[k-1]
ypast = np.ascontiguousarray(np.stack([yn[k - na:k] for k in k_ix]))   # (Nk, na, ny)
upast = np.ascontiguousarray(np.stack([un[k - nb:k] for k in k_ix]))   # (Nk, nb, nu)

with torch.no_grad():
    x_hat = encoder(
        torch.tensor(upast, dtype=DTYPE_PT),
        torch.tensor(ypast, dtype=DTYPE_PT),
    ).numpy()   # (Nk, nxd)

x_hat_phys = x_hat[:, :NX_PHYS]   # (Nk, 6)

# =========================================================================
# Compare against x_true(k) and x_true(k-1)
# =========================================================================

xt_k   = x_true_norm[k_ix]       # true state at k   (what the rollout expects)
xt_km1 = x_true_norm[k_ix - 1]   # true state at k-1 (what the window actually gives)

def rmse_per_ch(ref, est):
    return np.sqrt(((ref - est) ** 2).mean(axis=0))

def r2_per_ch(ref, est):
    ss_res = ((ref - est) ** 2).sum(axis=0)
    ss_tot = ((ref - ref.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / ss_tot

rmse_k   = rmse_per_ch(xt_k,   x_hat_phys)
rmse_km1 = rmse_per_ch(xt_km1, x_hat_phys)
r2_k     = r2_per_ch(xt_k,   x_hat_phys)
r2_km1   = r2_per_ch(xt_km1, x_hat_phys)

labels = ['q1 ', 'q2 ', 'q3 ', 'dq1', 'dq2', 'dq3']

print(f'\n{"="*70}')
print('Hybrid encoder lag diagnostic')
print(f'{"="*70}')
print(f'  Windows: {n_windows} (stride {stride}), na={na}, nb={nb}, fs={fs} Hz')
print(f'  deepSI na_right=0: ypast ends at y[k-1], encoder initializes x(k)')
print(f'\n  If R2(k-1) >> R2(k): encoder is aligned to k-1, confirming the lag.')
print(f'  If R2(k) ~ R2(k-1): no lag (both very close, e.g. slow dynamics).')
print()
print(f'  {"chan":4s}  {"RMSE vs x(k)":>14s}  {"RMSE vs x(k-1)":>14s}  '
      f'{"R2 vs x(k)":>12s}  {"R2 vs x(k-1)":>14s}  {"verdict":8s}')
print(f'  {"----":4s}  {"----------":>14s}  {"--------------":>14s}  '
      f'{"----------":>12s}  {"--------------":>14s}  {"-------":8s}')

for ch in range(NX_PHYS):
    if rmse_k[ch] < rmse_km1[ch]:
        verdict = 'ok'       # closer to x(k) than x(k-1)
    elif rmse_km1[ch] < 1e-4 and rmse_k[ch] / max(rmse_km1[ch], 1e-30) > 10:
        verdict = 'LAG'      # clearly aligned to k-1
    else:
        verdict = '??'
    print(f'  {labels[ch]}   {rmse_k[ch]:14.6e}  {rmse_km1[ch]:14.6e}  '
          f'{r2_k[ch]:+12.6f}  {r2_km1[ch]:+14.6f}  {verdict}')

print()
ratio_pos = rmse_k[:3].mean() / max(rmse_km1[:3].mean(), 1e-30)
ratio_vel = rmse_k[3:].mean() / max(rmse_km1[3:].mean(), 1e-30)
print(f'  Mean RMSE ratio (k/k-1): positions = {ratio_pos:.1f}x, velocities = {ratio_vel:.1f}x')

if all(rmse_km1[ch] < 1e-4 for ch in range(NX_PHYS)):
    print('\n  CONFIRMED: encoder physical states match x(k-1), not x(k).')
    print('  Fix: pass na_right=1 to SSE_Interconnect and size encoder with na+1.')
else:
    print('\n  Lag not clearly confirmed from RMSE alone. Inspect R2 columns.')

print(f'{"="*70}\n')
