"""
verify_normalization.py
-----------------------
Static (no training, no simulation) verification that the normalization pipeline
used in gantry_subnet_verification.py is self-consistent.

Four checks are performed on the MATLAB training and validation data:

  1. Cd_norm correctness
       Cd_norm @ x_norm  ==  y_phys / ystd   (output reconstruction in norm space)
       Equivalently: P.T @ q_logical / ystd matches the stage positions from MATLAB.

  2. Force normalization round-trip
       (u / ustd) * ustd  ==  u   (no mean subtraction, u0 = 0)

  3. Output normalization round-trip
       (y / ystd) * ystd  ==  y   (no mean subtraction, y0 = 0)

  4. P.T identity
       P.T @ q_logical (logical→stage) matches y_train directly from MATLAB,
       confirming that the MATLAB stage positions are P.T @ q_logical and that
       our coordinate conventions are consistent.

All tests print MAX absolute error and PASS / FAIL.  Tolerance = 1e-5 m.

Run from project root:
    conda run -n GraduationProject python scripts/gantry/verification/verify_normalization.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
from scipy.io import loadmat
import deepSI

from model_augmentation.systems.gantry_ss import Cd, P

# ── Config ────────────────────────────────────────────────────────────────────
_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'gantry', 'matlab')
N_HOLD    = 10000   # hold-period samples stripped from each end (0.5 s at 20 kHz)
TOL       = 1e-5    # absolute error tolerance [m] (or [N] for forces)
NX, NU, NY = 6, 3, 3

# ── Load data ─────────────────────────────────────────────────────────────────
def load_mat(split):
    d = loadmat(os.path.join(_DATA_DIR, f'gantry_lti_{split}.mat'), squeeze_me=True)
    return deepSI.System_data(
        u  = d['u'][N_HOLD:-N_HOLD].astype(np.float32),
        y  = d['y'][N_HOLD:-N_HOLD].astype(np.float32),
        x  = d['x_logical'][N_HOLD:-N_HOLD].astype(np.float32),
        dt = float(d['dt']),
    )

print('Loading data …')
train_data = load_mat('train')
val_data   = load_mat('val')
print(f'  Train T={len(train_data.u)},  Val T={len(val_data.u)}')

# ── Normalization stats — identical to gantry_subnet_verification.py ──────────
std_u = train_data.u.std(axis=0).reshape(NU, 1).astype(np.float32) + 1e-8  # (3,1) [N]
std_x = train_data.x.std(axis=0).reshape(NX, 1).astype(np.float32) + 1e-8  # (6,1) [m or rad, m/s]
ystd  = train_data.y.std(axis=0).astype(np.float32) + 1e-8                  # (3,)  [m]

# Cd_norm: Cd_norm[i,j] = Cd[i,j] * std_x[j] / ystd[i]
# Derivation: y_norm = y_phys/ystd = (Cd@x_phys)/ystd = (Cd @ (x_norm*std_x))/ystd
#   → y_norm[i] = sum_j Cd[i,j]*std_x[j]/ystd[i] * x_norm[j]  = Cd_norm[i,:] @ x_norm
Cd_np     = Cd.numpy()                                              # (3, 6)
Cd_norm   = Cd_np * std_x.flatten()[None, :] / ystd[:, None]      # (3, 6)

# ── Print normalization constants for inspection ──────────────────────────────
print('\n── Normalization constants ──────────────────────────────────────────────')
print(f'  ustd  = {std_u.flatten()}  [N]')
print(f'  std_x = {std_x.flatten()}')
print(f'  ystd  = {ystd}  [m]')
print(f'\n  Cd (physical):\n{Cd_np}')
print(f'\n  Cd_norm:\n{Cd_norm}')

# ── Helper ────────────────────────────────────────────────────────────────────
def check(name, lhs, rhs, tol=TOL):
    err = np.abs(lhs - rhs).max()
    status = 'PASS' if err < tol else 'FAIL'
    print(f'  [{status}]  {name}  |  max |err| = {err:.3e}  (tol={tol:.0e})')
    return status == 'PASS'

all_pass = True
print('\n── Verification checks ──────────────────────────────────────────────────')

# ── Check 1: Cd_norm @ x_norm == y_phys / ystd ───────────────────────────────
# Use training data. x_logical from MATLAB, y = stage positions from MATLAB.
# x_norm = x_logical / std_x  (shape: (T, 6))
# lhs = (Cd_norm @ x_norm.T).T  →  (T, 3) normalized stage positions
# rhs = y_train / ystd           →  (T, 3) normalized stage positions
print('\nCheck 1: Cd_norm @ x_norm  ==  y / ystd  (train data)')
x_norm_train = train_data.x / std_x.flatten()[None, :]              # (T, 6)
lhs1_train   = (Cd_norm @ x_norm_train.T).T                          # (T, 3)
rhs1_train   = train_data.y / ystd[None, :]                          # (T, 3)
all_pass &= check('train set (X1)', lhs1_train[:, 0], rhs1_train[:, 0])
all_pass &= check('train set (X2)', lhs1_train[:, 1], rhs1_train[:, 1])
all_pass &= check('train set (Y) ', lhs1_train[:, 2], rhs1_train[:, 2])

print('\nCheck 1: Cd_norm @ x_norm  ==  y / ystd  (val data)')
x_norm_val = val_data.x / std_x.flatten()[None, :]                  # (T, 6)
lhs1_val   = (Cd_norm @ x_norm_val.T).T                              # (T, 3)
rhs1_val   = val_data.y / ystd[None, :]                              # (T, 3)
all_pass &= check('val   set (X1)', lhs1_val[:, 0], rhs1_val[:, 0])
all_pass &= check('val   set (X2)', lhs1_val[:, 1], rhs1_val[:, 1])
all_pass &= check('val   set (Y) ', lhs1_val[:, 2], rhs1_val[:, 2])

# ── Check 2: Force normalization round-trip (u0 = 0) ─────────────────────────
# u_norm = u / ustd  →  u_phys = u_norm * ustd  ==  u
print('\nCheck 2: (u / ustd) * ustd  ==  u  (u0 = 0)')
u_norm_train = train_data.u / std_u.flatten()[None, :]
u_recovered  = u_norm_train * std_u.flatten()[None, :]
all_pass &= check('FX1', u_recovered[:, 0], train_data.u[:, 0])
all_pass &= check('FX2', u_recovered[:, 1], train_data.u[:, 1])
all_pass &= check('FY ', u_recovered[:, 2], train_data.u[:, 2])

# ── Check 3: Output normalization round-trip (y0 = 0) ─────────────────────────
# y_norm = y / ystd  →  y_phys = y_norm * ystd  ==  y
print('\nCheck 3: (y / ystd) * ystd  ==  y  (y0 = 0)')
y_norm_train = train_data.y / ystd[None, :]
y_recovered  = y_norm_train * ystd[None, :]
all_pass &= check('X1', y_recovered[:, 0], train_data.y[:, 0])
all_pass &= check('X2', y_recovered[:, 1], train_data.y[:, 1])
all_pass &= check('Y ', y_recovered[:, 2], train_data.y[:, 2])

# ── Check 4: P.T @ q_logical == y_stage directly (MATLAB data consistency) ───
# Confirms that the MATLAB stage positions y = [X1, X2, Y] are exactly
# P.T @ q_logical, i.e., our coordinate convention matches the MATLAB ground truth.
# Also confirms Cd (physical) is consistent with P before any normalization.
print('\nCheck 4: P.T @ q_logical  ==  y_stage  (raw MATLAB data, no normalization)')
P_np     = P.numpy()                                                 # (3, 3)
q_log    = train_data.x[:, :3]                                       # (T, 3) logical positions [m, rad, m]
y_stage  = (P_np.T @ q_log.T).T                                      # (T, 3) stage positions [m]
all_pass &= check('X1', y_stage[:, 0], train_data.y[:, 0])
all_pass &= check('X2', y_stage[:, 1], train_data.y[:, 1])
all_pass &= check('Y ', y_stage[:, 2], train_data.y[:, 2])

# ── Summary ───────────────────────────────────────────────────────────────────
print('\n────────────────────────────────────────────────────────────────────────')
if all_pass:
    print('ALL CHECKS PASSED — normalization pipeline is self-consistent.')
else:
    print('ONE OR MORE CHECKS FAILED — review output above.')
print('────────────────────────────────────────────────────────────────────────')
