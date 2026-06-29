"""diag22_oracle_aug.py

Oracle augmented simulation: what NRMS floor is achievable with perfect MSD state knowledge?

No model, no training. Only val data + baseline FP simulation.

Approach:
  1. Run baseline FP simulation (y_fp): GT x_logical init, pure FP model, no MSD.
     NRMS(y_fp, y_gt) = baseline FP NRMS (already known: ~0.003-0.004).

  2. Compute residual: r[t] = y_gt[t] - y_fp[t]  (what the FP model misses -- the MSD effect).

  3. Fit optimal C_aug via OLS:  r ~ x_aug_GT @ C_aug.T
     Finds the best *static* linear coupling from GT (delta_a, vdelta_a) to the residual.

  4. Oracle NRMS = RMS(y_fp + x_aug_GT @ C_aug_opt.T - y_gt) / ystd
     = residual after explaining as much as possible with GT MSD states.
     This is the floor: below this, the augmented model cannot go with the current
     FP model structure and a linear output coupling.

  5. R2 per output channel: fraction of FP residual variance explained by GT x_aug.
     High R2 -> strong learnable signal.  Low R2 -> MSD barely visible in that channel.

Outputs (console only, no files saved):
  - Baseline FP NRMS (reference)
  - Residual power per channel (how large is the MSD contribution?)
  - R2 of residual vs GT x_aug (how learnable is it?)
  - Optimal C_aug matrix (what coupling does OLS find?)
  - Oracle NRMS (the floor for 100-epoch training to approach)

Runtime: < 30 seconds.
"""

import os
import sys
import numpy as np
import torch
import deepSI
from scipy.io import loadmat

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from model_augmentation.utils.utils import *
from model_augmentation.systems.gantry_ss import Cd, Dd, P
from model_augmentation.systems.gantry_linearization import gantry_linearize_and_discretize
from model_augmentation.fit_systems.blocks import Gantry_State_Block

## ============================================================
## Configuration -- verbatim from gantry_interconnect_dynamic.py
## ============================================================

MODE    = 'multisine'
NX_PHYS = 6
nu      = 3
ny      = 3
Y_OP    = None
SEED    = 42

FS_ORIG = 20000
FS_NEW  = 4000
D       = FS_ORIG // FS_NEW
TS_NEW  = 1.0 / FS_NEW

UP_SAMPLE = 2   # same as DEFAULT_HP['up_sample']

USE_F64  = False
DTYPE_NP = np.float64    if USE_F64 else np.float32
DTYPE_PT = torch.float64 if USE_F64 else torch.float32

np.random.seed(SEED)
torch.manual_seed(SEED)

## ============================================================
## Data loading -- verbatim
## ============================================================

DATA_SUBDIR = 'multisine'
TRAJ_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'gantry', 'matlab', DATA_SUBDIR)

TRAIN_FILES = [
    'T1_Y_sweep_conservative.mat',
    'T2_X_sym_Y030.mat',
    'T3_X_sym_Y000.mat',
    'T4_X_antisym_Y020.mat',
    'T5_X_sym_Y_sweep.mat',
    'T6_Y_sweep_aggressive.mat',
    'T7_X_antisym_Y_sweep.mat',
    'T8_X_sym_anti_Y_sweep.mat',
]
VAL_FILE = 'V1_X_sym_Y_mid_sweep.mat'

def _load_u(d):
    return d['u_total'] if 'u_total' in d else d['u']

def load_traj(filename):
    d = loadmat(os.path.join(TRAJ_DIR, filename), squeeze_me=True)
    return deepSI.System_data(
        u=_load_u(d)[::D].astype(DTYPE_NP),
        y=d['y'][::D].astype(DTYPE_NP),
        dt=TS_NEW,
    )

def load_mat_aug(filename):
    d = loadmat(os.path.join(TRAJ_DIR, filename), squeeze_me=True)
    u = _load_u(d)[::D].astype(DTYPE_NP)
    y = d['y'][::D].astype(DTYPE_NP)
    x_logical = d['x_logical'][::D].astype(DTYPE_NP)
    delta_a   = d['delta_a'][::D].astype(DTYPE_NP)
    # HEURISTIC: backward FD for velocity
    vdelta_a      = np.zeros_like(delta_a)
    vdelta_a[1:]  = (delta_a[1:] - delta_a[:-1]) * FS_NEW
    vdelta_a[0]   = vdelta_a[1]
    x_aug = np.stack([delta_a, vdelta_a], axis=1)   # (N, 2) physical units [m, m/s]
    return u, y, x_logical, x_aug

train_list = [load_traj(f) for f in TRAIN_FILES]
val_data   = load_traj(VAL_FILE)
_, _, val_x_logical, val_x_aug = load_mat_aug(VAL_FILE)

## ============================================================
## Normalisation -- verbatim
## ============================================================

u_all = np.concatenate([t.u for t in train_list])
y_all = np.concatenate([t.y for t in train_list])

fs = 1.0 / train_list[0].dt
P_inv_T = np.linalg.inv(P.numpy().T).astype(DTYPE_NP)
x_logical_list = []
for t in train_list:
    pos_logical = (P_inv_T @ t.y.T).T
    vel_logical = np.diff(pos_logical, axis=0) * fs
    vel_logical = np.vstack([vel_logical[:1], vel_logical])
    x_logical_list.append(np.hstack([pos_logical, vel_logical]))
x_all = np.concatenate(x_logical_list)

x_mean = x_all.mean(axis=0).reshape(NX_PHYS, 1).astype(DTYPE_NP)
std_x  = x_all.std(axis=0).reshape(NX_PHYS, 1).astype(DTYPE_NP) + 1e-8
std_u  = u_all.std(axis=0).reshape(nu, 1).astype(DTYPE_NP) + 1e-8
u_mean = u_all.mean(axis=0).reshape(nu, 1).astype(DTYPE_NP)
ystd   = y_all.std(axis=0).astype(DTYPE_NP) + 1e-8
y0     = y_all.mean(axis=0).astype(DTYPE_NP)

Cd_norm = Cd.numpy() * std_x.flatten()[None, :] / ystd[:, None]   # (3, 6)
Dd_np   = Dd.numpy()                                                # (3, 3)

## ============================================================
## Baseline FP simulation (oracle x_phys init, no augmentation)
## ============================================================

def run_baseline_fp():
    """Simulate FP model from GT x_logical[0]. Returns y_fp (N, 3) in physical units."""
    phy_block = Gantry_State_Block(
        Y_op=Y_OP, std_x=std_x, std_u=std_u,
        x_mean=x_mean, u_mean=u_mean, Ts=TS_NEW,
        up_sample=UP_SAMPLE,
    ).to(DTYPE_PT)
    phy_block.eval()

    x_norm_np = ((val_x_logical[0] - x_mean.flatten()) / std_x.flatten()).astype(DTYPE_NP)
    u_val_norm = ((val_data.u - u_mean.flatten()) / std_u.flatten()).astype(DTYPE_NP)

    y_hat_list = []
    with torch.no_grad():
        for t in range(len(val_data.u)):
            u_norm_np = u_val_norm[t]
            y_norm = Cd_norm @ x_norm_np + Dd_np @ u_norm_np
            y_hat_list.append(y_norm * ystd + y0)
            x_t = torch.tensor(x_norm_np, dtype=DTYPE_PT).view(1, NX_PHYS, 1)
            u_t = torch.tensor(u_norm_np, dtype=DTYPE_PT).view(1, nu, 1)
            x_norm_np = phy_block(
                torch.cat([x_t, u_t], dim=1)).view(NX_PHYS).cpu().numpy()

    return np.array(y_hat_list, dtype=DTYPE_NP)   # (N, 3)


## ============================================================
## Main
## ============================================================

print('\n' + '='*60)
print('diag22 -- oracle augmented NRMS floor')
print('='*60)

print('\nRunning baseline FP simulation...')
y_fp  = run_baseline_fp()          # (N, 3) [m]
y_gt  = val_data.y                 # (N, 3) [m]
N     = len(y_gt)

nrms_fp = np.sqrt(((y_fp - y_gt) ** 2).mean(axis=0)) / ystd
rms_fp  = nrms_fp * ystd * 1e6
print('\n=== Baseline FP NRMS (reference) ===')
for ch, lbl in enumerate(['X1', 'X2', 'Y ']):
    print(f'  {lbl}: {nrms_fp[ch]:.4f}  ({rms_fp[ch]:.1f} um)')

# Residual: what the FP model misses (= MSD contribution to y)
residual = y_gt - y_fp             # (N, 3) [m]
rms_resid = np.sqrt((residual ** 2).mean(axis=0))
print('\n=== Residual RMS (MSD contribution to each output) ===')
for ch, lbl in enumerate(['X1', 'X2', 'Y ']):
    print(f'  {lbl}: {rms_resid[ch]*1e6:.1f} um  '
          f'({100*rms_resid[ch]/ystd[ch]:.2f}% of ystd)')

# GT x_aug stats
print('\n=== GT x_aug stats (val trajectory) ===')
x_aug_labels = ['delta_a [m]', 'vdelta_a [m/s]']
for i, lbl in enumerate(x_aug_labels):
    print(f'  {lbl:20s}: mean={val_x_aug[:,i].mean():.3e}  '
          f'std={val_x_aug[:,i].std():.3e}  '
          f'max|x|={np.abs(val_x_aug[:,i]).max():.3e}')

# R2: how much of the FP residual is explained by GT x_aug?
# residual_norm[t, ch] = residual[t, ch] / ystd[ch]  (dimensionless)
residual_norm = residual / ystd[None, :]               # (N, 3) dimensionless

# Normalize x_aug to unit std for numerical stability
x_aug_std  = val_x_aug.std(axis=0) + 1e-12            # (2,)
x_aug_norm = val_x_aug / x_aug_std[None, :]           # (N, 2) unit-std

A_ols = np.hstack([x_aug_norm, np.ones((N, 1), dtype=DTYPE_NP)])   # (N, 3)

# Fit: residual_norm ~ A_ols @ W  ->  W shape (3, ny)
W, *_ = np.linalg.lstsq(A_ols, residual_norm, rcond=None)   # (3, ny)

residual_pred_norm = A_ols @ W    # (N, ny) -- OLS fit of residual

def r2_per_channel(ref, est):
    ss_res = ((ref - est) ** 2).sum(axis=0)
    ss_tot = ((ref - ref.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / (ss_tot + 1e-12)

r2_resid = r2_per_channel(residual_norm, residual_pred_norm)

print('\n=== R2: fraction of FP residual explained by GT x_aug (OLS) ===')
print('  (R2 ~ 1 -> strong learnable MSD signal;  R2 ~ 0 -> MSD barely visible)')
for ch, lbl in enumerate(['X1', 'X2', 'Y ']):
    print(f'  {lbl}: R2={r2_resid[ch]:+.4f}')

# Optimal C_aug in physical units (maps x_aug [m, m/s] -> y residual [m])
# W[:2, :] maps x_aug_norm -> residual_norm
# C_aug_phys[ch, i] = W[i, ch] / x_aug_std[i] * ystd[ch]  (residual in [m])
C_aug_phys = (W[:2, :] / x_aug_std[:, None]).T * ystd[:, None]  # (ny, 2) [m/m, m/(m/s)]
print('\n=== Optimal C_aug (OLS, physical units: y[m] per x_aug[m or m/s]) ===')
x_aug_units = ['m/m ', 'm/(m/s)']
for ch, lbl in enumerate(['X1', 'X2', 'Y ']):
    row = '  '.join([f'{C_aug_phys[ch, i]:.3e} {x_aug_units[i]}'
                     for i in range(2)])
    print(f'  {lbl}: {row}')

# Oracle NRMS: FP sim + OLS-fitted MSD contribution
oracle_resid_norm = residual_norm - residual_pred_norm   # (N, ny)
nrms_oracle = np.sqrt((oracle_resid_norm ** 2).mean(axis=0))
rms_oracle  = nrms_oracle * ystd * 1e6

print('\n=== Oracle NRMS (FP + optimal linear C_aug @ GT x_aug) ===')
print('  (This is the floor for gantry_interconnect_dynamic.py with 100 epochs)')
for ch, lbl in enumerate(['X1', 'X2', 'Y ']):
    improv = 100.0 * (nrms_fp[ch] - nrms_oracle[ch]) / (nrms_fp[ch] + 1e-12)
    print(f'  {lbl}: {nrms_oracle[ch]:.4f}  ({rms_oracle[ch]:.1f} um)  '
          f'vs baseline_FP={nrms_fp[ch]:.4f} ({rms_fp[ch]:.1f} um)  '
          f'improvement={improv:+.1f}%')

print('\n=== Summary ===')
print(f'  channel   baseline_FP   oracle_floor   improvement')
for ch, lbl in enumerate(['X1 ', 'X2 ', 'Y  ']):
    improv = 100.0 * (nrms_fp[ch] - nrms_oracle[ch]) / (nrms_fp[ch] + 1e-12)
    print(f'  {lbl}       {nrms_fp[ch]:.4f}         {nrms_oracle[ch]:.4f}        {improv:+.1f}%')

print('\n' + '='*60)
print('diag22 complete')
print('='*60)
