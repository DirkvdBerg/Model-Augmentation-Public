"""
diag5_normalization_convention.py
----------------------------------
Tests whether mean subtraction in I/O normalization breaks the W^b
reconstructability guarantee (Hoekstra 2026 Eq. 16-17).

Background:
  normalize_linear_ss_matrices produces W^b for pure-scaled data:
    u/std_u,  y/std_y  (no mean subtraction)
  The gantry pipeline normalizes with mean subtraction:
    (u-u_mean)/std_u,  (y-y0)/std_y
  W^b is linear, so feeding mean-subtracted data introduces a constant
  bias in state space:
    x_b_B = x_b_A - bias_x
    bias_x = W^b_y @ [y0/std_y repeated (na+1) times]
           + W^b_u @ [u_mean/std_u repeated (nb+1) times]
  This follows directly from Hoekstra 2026 Eq. 16-17.

Three conditions, same W^b matrices throughout:
  A: pure-scaled      y/std_y,       u/std_u          (paper intention)
  B: mean-subtracted  (y-y0)/std_y,  (u-u_mean)/std_u (pipeline without fix)
  C: mean-sub + fix   add back offset before W^b  (must equal A exactly)

Part 1 -- Severity: y0/std_y and u_mean/std_u per channel.
Part 2 -- Noiseless LTI simulation (zero model mismatch, zero noise):
  N1: Condition A NRMS < 1e-3  (formula valid for pure-scaled)
  N2: Condition B NRMS >> A    (mean subtraction breaks formula)
  N3: (x_b_A - x_b_B) = bias_x at every timestep  (max_dev < 1e-4)
  N4: Condition C == A  (fix restores formula exactly)
Part 3 -- Real multisine data (V1_osc_Y025.mat):
  R1/R2: NRMS table A vs B per channel
  R3: (x_b_A - x_b_B) constant across windows, equal to bias_x
  R4: Condition C == A on real data

Usage:
    conda run -n GraduationProject python \\
        scripts/gantry/encoder-augmentation/diag5_normalization_convention.py
"""

import os
import sys
import json
import numpy as np
import torch
import deepSI
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from model_augmentation.utils.utils import normalize_linear_ss_matrices
from model_augmentation.fit_systems.pre_encoder import linear_encoder_init_aug
from model_augmentation.systems.gantry_ss import P
from model_augmentation.systems.gantry_linearization import gantry_linearize_and_discretize

# =============================================================================
# Configuration (identical to diag2/3)
# =============================================================================

NX_PHYS = 6
nu, ny  = 3, 3

FS_ORIG = 20000
FS_NEW  = 4000
D       = FS_ORIG // FS_NEW
TS_NEW  = 1.0 / FS_NEW

DTYPE_NP = np.float32
DTYPE_PT = torch.float32

na = 4 * NX_PHYS + 1   # = 25  HEURISTIC: Jan's rule of thumb
nb = na

_NB_WIN    = nb + 1   # encoder sees nb+1 u samples
_NA_WIN    = na + 1   # encoder sees na+1 y samples
_WIN_START = max(_NB_WIN, _NA_WIN) - 1   # = 25

T_SIM = 3000   # noiseless simulation length (samples)

TRAJ_DIR = os.path.join(PROJECT_ROOT, 'data', 'gantry', 'matlab',
                        'multisine', 'm50', 'narrowband')
TRAIN_FILES = [
    'T1_Y_osc.mat', 'T2_X_sym_Y_sweep.mat', 'T3_X_sym_Y000.mat',
    'T4_X_sym_Y030.mat', 'T5_theta_Y_coupling.mat', 'T6_lissajous_XY.mat',
    'T7_full_MIMO.mat', 'T8_multi_amp.mat', 'T9_Y_sweep_repeated.mat',
    'T10_multi_axis_repeated.mat',
]
VAL_FILE = 'V1_osc_Y025.mat'

OUT_DIR = os.path.join(PROJECT_ROOT, 'simulations', 'gantry_subnet', 'encoder')
os.makedirs(OUT_DIR, exist_ok=True)

STATE_NAMES = ['q1', 'q2', 'q3', 'dq1', 'dq2', 'dq3']

# =============================================================================
# Check helper
# =============================================================================

results = {}

def check(name, condition, detail=''):
    status = 'PASS' if condition else 'FAIL'
    results[name] = status
    marker = '  ' if condition else '!!'
    print(f'  [{status}] {marker} {name}' + (f'  ({detail})' if detail else ''))
    return condition

# =============================================================================
# Data loading (same as diag2/3)
# =============================================================================

def load_mat(filename):
    d     = loadmat(os.path.join(TRAJ_DIR, filename), squeeze_me=True)
    u     = d['u_total'][::D].astype(DTYPE_NP)
    y     = d['y'][::D].astype(DTYPE_NP)
    x_log = d['x_logical'][::D].astype(DTYPE_NP)
    return u, y, x_log

print('Loading data...')
train_data = [load_mat(f) for f in TRAIN_FILES]
val_u, val_y, val_x = load_mat(VAL_FILE)

# Statistics from training data
u_all = np.concatenate([u for u, _, _ in train_data])
y_all = np.concatenate([y for _, y, _ in train_data])
x_all = np.concatenate([x for _, _, x in train_data])

std_u  = u_all.std(axis=0).astype(DTYPE_NP)  + 1e-8   # (3,) pure-scaled
std_y  = y_all.std(axis=0).astype(DTYPE_NP)  + 1e-8   # (3,)
std_x  = x_all.std(axis=0).astype(DTYPE_NP)  + 1e-8   # (6,)
u_mean = u_all.mean(axis=0).astype(DTYPE_NP)           # (3,) operating point
y0     = y_all.mean(axis=0).astype(DTYPE_NP)           # (3,)

# =============================================================================
# System matrices and encoder
# =============================================================================

Ad, Bd, Cd_dt, Dd_dt = gantry_linearize_and_discretize(dt=TS_NEW)
sys_data_tmp = deepSI.System_data(u=u_all, y=y_all)
sys_data_tmp.x = x_all
Ad_bar, Bd_bar, Cd_bar, Dd_bar = normalize_linear_ss_matrices(
    Ad, Bd, Cd_dt, Dd_dt, sys_data_tmp)

torch.manual_seed(0)
enc = linear_encoder_init_aug(
    A=Ad_bar, B=Bd_bar, C=Cd_bar, D=Dd_bar,
    nx=NX_PHYS, nu=nu, ny=ny, na=na, nb=nb,
    nx_aug=0, flag_linear_only=True,
)
enc.eval()

Wb_y = enc.Wb_psi_y.detach().numpy()   # (nx, (na+1)*ny)
Wb_u = enc.Wb_psi_u.detach().numpy()   # (nx, (nb+1)*nu)

# =============================================================================
# Analytical bias
# THEORY: direct consequence of linearity of W^b (Hoekstra 2026 Eq. 16-17).
# When (y - y0)/std_y is fed instead of y/std_y, the formula produces:
#   x_b_B = x_b_A - bias_x
# =============================================================================

y0_ratio     = (y0    / std_y).astype(DTYPE_NP)    # (ny,)  operating pt / excitation
u_mean_ratio = (u_mean / std_u).astype(DTYPE_NP)   # (nu,)

y_off_vec = np.tile(y0_ratio,     na + 1)   # ((na+1)*ny,)  tiled offset for y window
u_off_vec = np.tile(u_mean_ratio, nb + 1)   # ((nb+1)*nu,)  tiled offset for u window

bias_x = Wb_y @ y_off_vec + Wb_u @ u_off_vec   # (nx,)  constant bias in state space

# =============================================================================
# Window helper
# =============================================================================

def to_windows_np(u_n, y_n):
    """(T, nu), (T, ny) pure-scaled arrays -> flat window arrays for encoder."""
    u_wins = np.lib.stride_tricks.sliding_window_view(
        u_n, (_NB_WIN, nu)).reshape(-1, _NB_WIN * nu)
    y_wins = np.lib.stride_tricks.sliding_window_view(
        y_n, (_NA_WIN, ny)).reshape(-1, _NA_WIN * ny)
    return u_wins.copy(), y_wins.copy()


def run_enc(uh_np, yh_np):
    with torch.no_grad():
        return enc(
            torch.tensor(uh_np, dtype=DTYPE_PT),
            torch.tensor(yh_np, dtype=DTYPE_PT),
        ).numpy()   # (N_windows, NX_PHYS)


def nrms_per_ch(x_enc_norm, x_gt_phys):
    """x_enc_norm in normalized coords, x_gt_phys in physical units."""
    x_phys = x_enc_norm * std_x
    T = min(len(x_phys), len(x_gt_phys))
    err = np.sqrt(np.mean((x_phys[:T] - x_gt_phys[:T]) ** 2, axis=0))
    ref = np.sqrt(np.mean(x_gt_phys[:T] ** 2, axis=0))
    return err / (ref + 1e-12)


# =============================================================================
# PART 1: Severity ratios
# =============================================================================

print('\n' + '=' * 70)
print('PART 1: Severity ratios  (operating point / excitation amplitude)')
print('=' * 70)
print('\n  If |y0/std_y| >> 0, mean subtraction introduces a large bias.')
print('\n  Output channels:')
for i, ch in enumerate(['X1 (stage)', 'X2 (stage)', 'Y  (stage)']):
    r = abs(y0[i]) / std_y[i]
    print(f'    {ch}: y0={y0[i]:+.4f} m   std_y={std_y[i]:.4f} m   |y0/std_y|={r:.3f}')

print('\n  Input channels:')
for i, ch in enumerate(['F1', 'F2', 'F3']):
    r = abs(u_mean[i]) / std_u[i]
    print(f'    {ch}: u_mean={u_mean[i]:+.4f}   std_u={std_u[i]:.4f}   |u_mean/std_u|={r:.3f}')

print(f'\n  Analytical bias_x (normalized state space):')
for i, name in enumerate(STATE_NAMES):
    print(f'    {name}: {bias_x[i]:+.4f}')
print(f'  ||bias_x|| = {np.linalg.norm(bias_x):.4f}')

# =============================================================================
# PART 2: Noiseless LTI simulation
# =============================================================================

print('\n' + '=' * 70)
print('PART 2: Noiseless LTI simulation')
print('=' * 70)
print(f'  Simulating normalized LTI model for T={T_SIM} steps...')

# Simulate x(k+1) = Ad_bar @ x(k) + Bd_bar @ u(k),  y(k) = Cd_bar @ x(k)
# Input is zero-mean random (pure-scaled, already in normalized scale)
rng   = np.random.default_rng(42)
u_sim = rng.standard_normal((T_SIM, nu)).astype(DTYPE_NP)   # zero-mean
x_sim = np.zeros((T_SIM, NX_PHYS), dtype=DTYPE_NP)
y_sim = np.zeros((T_SIM, ny),      dtype=DTYPE_NP)

A_np = np.array(Ad_bar, dtype=DTYPE_NP)
B_np = np.array(Bd_bar, dtype=DTYPE_NP)
C_np = np.array(Cd_bar, dtype=DTYPE_NP)

for k in range(T_SIM - 1):
    y_sim[k]     = C_np @ x_sim[k]
    x_sim[k + 1] = A_np @ x_sim[k] + B_np @ u_sim[k]
y_sim[-1] = C_np @ x_sim[-1]

# Ground truth for encoder windows: x_sim[_WIN_START:]
x_gt_sim = x_sim[_WIN_START:]   # (T_SIM - na, NX_PHYS), normalized coords

# --- Condition A: pure-scaled (formula as derived) ---
uh_A, yh_A = to_windows_np(u_sim, y_sim)
x_b_A = run_enc(uh_A, yh_A)   # (T_SIM-na, NX_PHYS)

# --- Condition B: mean-subtracted (what pipeline feeds without fix) ---
# Artificially impose the operating point offset from real data statistics
u_sim_B = u_sim - u_mean_ratio[None, :]   # (T_SIM, nu): subtract u_mean/std_u
y_sim_B = y_sim - y0_ratio[None, :]       # (T_SIM, ny): subtract y0/std_y
uh_B, yh_B = to_windows_np(u_sim_B, y_sim_B)
x_b_B = run_enc(uh_B, yh_B)

# --- Condition C: mean-subtracted + fix (add back offset before W^b) ---
uh_C = uh_B + u_off_vec[None, :]   # restores u_sim windows
yh_C = yh_B + y_off_vec[None, :]   # restores y_sim windows
x_b_C = run_enc(uh_C, yh_C)

# --- NRMS in normalized coords (sim data IS in normalized coords) ---
T_cmp = min(len(x_b_A), len(x_gt_sim))

def nrms_norm(x_enc, x_gt):
    T = min(len(x_enc), len(x_gt))
    err = np.sqrt(np.mean((x_enc[:T] - x_gt[:T]) ** 2, axis=0))
    ref = np.sqrt(np.mean(x_gt[:T] ** 2, axis=0))
    return err / (ref + 1e-12)

nrms_sim_A = nrms_norm(x_b_A, x_gt_sim)
nrms_sim_B = nrms_norm(x_b_B, x_gt_sim)
nrms_sim_C = nrms_norm(x_b_C, x_gt_sim)

print('\n  NRMS per channel (normalized coords, lower = closer to true state):')
print(f'  {"channel":<6} | {"Cond A (pure-scaled)":>20} | '
      f'{"Cond B (mean-sub)":>17} | {"ratio B/A":>9}')
print('  ' + '-' * 62)
for i, name in enumerate(STATE_NAMES):
    ratio = nrms_sim_B[i] / (nrms_sim_A[i] + 1e-12)
    print(f'  {name:<6} | {nrms_sim_A[i]:>20.2e} | {nrms_sim_B[i]:>17.2e} | {ratio:>9.1f}x')

print('\n--- Noiseless simulation checks ---')

# N1: pure-scaled gives near-exact state recovery
# Mean NRMS across channels -- on noiseless LTI data this should be tiny
check('N1: Cond A mean NRMS < 1e-3  (formula exact for pure-scaled)',
      float(nrms_sim_A.mean()) < 1e-3,
      f'mean_NRMS_A={nrms_sim_A.mean():.2e}')

# N2: mean-subtracted is significantly worse -- at least 10x on mean
check('N2: Cond B mean NRMS > 10x Cond A  (mean-sub breaks formula)',
      float(nrms_sim_B.mean()) > 10 * float(nrms_sim_A.mean() + 1e-10),
      f'mean_NRMS_B={nrms_sim_B.mean():.2e}  mean_NRMS_A={nrms_sim_A.mean():.2e}')

# N3: (x_b_A - x_b_B) = bias_x analytically at every timestep
diff_AB_sim = x_b_A[:T_cmp] - x_b_B[:T_cmp]   # (N, nx): should be constant
deviation   = diff_AB_sim - bias_x[None, :]     # should be all zeros
n3_max      = float(np.abs(deviation).max())
check('N3: (x_b_A - x_b_B) = bias_x analytically  (max_dev < 1e-4)',
      n3_max < 1e-4,
      f'max_dev={n3_max:.2e}  (proves bias is purely linear, not noise)')

# N4: condition C == A
# Tolerance 1e-3 (not 1e-5): float32 (a-b)+b != a due to rounding; the
# meaningful check is that C is orders-of-magnitude closer to A than B is.
n4_max = float(np.abs(x_b_C[:T_cmp] - x_b_A[:T_cmp]).max())
check('N4: Cond C == Cond A  (fix restores formula, max_diff < 1e-3)',
      n4_max < 1e-3,
      f'max_diff={n4_max:.2e}  (float32 (a-b)+b rounding; B is off by ~0.1-2.0 NRMS)')

print(f'\n  Std of (x_b_A - x_b_B) over time per channel (should be ~0):')
for i, name in enumerate(STATE_NAMES):
    print(f'    {name}: std={diff_AB_sim[:, i].std():.2e}  '
          f'mean={diff_AB_sim[:, i].mean():.4f}  bias_x={bias_x[i]:.4f}')

# =============================================================================
# PART 3: Real multisine data
# =============================================================================

print('\n' + '=' * 70)
print('PART 3: Real multisine data  (V1_osc_Y025.mat)')
print('=' * 70)

# --- Condition A: pure-scaled ---
u_val_A = val_u / std_u
y_val_A = val_y / std_y
uh_val_A, yh_val_A = to_windows_np(u_val_A, y_val_A)
x_val_A = run_enc(uh_val_A, yh_val_A)

# --- Condition B: mean-subtracted ---
u_val_B = (val_u - u_mean) / std_u
y_val_B = (val_y - y0) / std_y
uh_val_B, yh_val_B = to_windows_np(u_val_B, y_val_B)
x_val_B = run_enc(uh_val_B, yh_val_B)

# --- Condition C: mean-subtracted + fix ---
uh_val_C = uh_val_B + u_off_vec[None, :]
yh_val_C = yh_val_B + y_off_vec[None, :]
x_val_C  = run_enc(uh_val_C, yh_val_C)

# Ground truth: x_logical from val file
x_gt_val = val_x[_WIN_START:]

nrms_val_A = nrms_per_ch(x_val_A, x_gt_val)
nrms_val_B = nrms_per_ch(x_val_B, x_gt_val)
nrms_val_C = nrms_per_ch(x_val_C, x_gt_val)

print('\n  NRMS vs x_logical per channel (physical units):')
print(f'  {"channel":<6} | {"Cond A (pure-scaled)":>20} | '
      f'{"Cond B (mean-sub)":>17} | {"Cond C (fix)":>12} | {"ratio B/A":>9}')
print('  ' + '-' * 75)
for i, name in enumerate(STATE_NAMES):
    ratio = nrms_val_B[i] / (nrms_val_A[i] + 1e-12)
    print(f'  {name:<6} | {nrms_val_A[i]:>20.4f} | {nrms_val_B[i]:>17.4f} | '
          f'{nrms_val_C[i]:>12.4f} | {ratio:>9.1f}x')

print('\n--- Real data checks ---')

# R3: (x_val_A - x_val_B) is constant across all windows, equal to bias_x
diff_val_AB = x_val_A - x_val_B   # (N_windows, nx)
diff_std     = diff_val_AB.std(axis=0)    # should be ~0 (constant offset)
diff_mean    = diff_val_AB.mean(axis=0)   # should equal bias_x

r3_std_max = float(diff_std.max())
check('R3: (x_b_A - x_b_B) constant across windows  (max std < 1e-4)',
      r3_std_max < 1e-4,
      f'max_std={r3_std_max:.2e}')

r3b_bias_max = float(np.abs(diff_mean - bias_x).max())
check('R3b: mean(x_b_A - x_b_B) = bias_x  (max_dev < 1e-3)',
      r3b_bias_max < 1e-3,
      f'max_dev={r3b_bias_max:.2e}')

# R4: condition C == A
r4_max = float(np.abs(x_val_C - x_val_A).max())
check('R4: Cond C == Cond A on real data  (max_diff < 1e-3)',
      r4_max < 1e-3,
      f'max_diff={r4_max:.2e}  (float32 rounding; B is off by 0.1-2.0 NRMS)')

# =============================================================================
# Plots
# =============================================================================

T_plot = min(500, len(x_b_A))
t_sim  = np.arange(T_plot) / FS_NEW

# Plot 1: NRMS bar chart A vs B, real data
fig1, axes1 = plt.subplots(1, 2, figsize=(12, 4))
x_pos = np.arange(NX_PHYS)
w = 0.35
axes1[0].bar(x_pos - w/2, nrms_val_A, w, label='A: pure-scaled', color='C0')
axes1[0].bar(x_pos + w/2, nrms_val_B, w, label='B: mean-subtracted', color='C3')
axes1[0].set_xticks(x_pos); axes1[0].set_xticklabels(STATE_NAMES)
axes1[0].set_ylabel('NRMS vs x_logical')
axes1[0].set_title('Real data: NRMS per channel -- does mean-sub degrade init?')
axes1[0].legend(); axes1[0].grid(True, axis='y', alpha=0.3)
axes1[0].set_yscale('log')

# Plot 2: bias_x vs empirical mean difference (should match)
axes1[1].bar(x_pos - w/2, bias_x,    w, label='Analytical bias_x', color='C0')
axes1[1].bar(x_pos + w/2, diff_mean, w, label='Empirical mean(A-B)', color='C1', alpha=0.7)
axes1[1].set_xticks(x_pos); axes1[1].set_xticklabels(STATE_NAMES)
axes1[1].set_ylabel('Bias (normalized state units)')
axes1[1].set_title('Analytical vs empirical bias -- should be identical')
axes1[1].legend(); axes1[1].grid(True, axis='y', alpha=0.3)

fig1.tight_layout()
p1 = os.path.join(OUT_DIR, 'diag5_nrms_and_bias.png')
fig1.savefig(p1, dpi=150, bbox_inches='tight')
plt.close(fig1)
print(f'\n  Saved: {p1}')

# Plot 2: Time traces (noiseless sim), A vs B for first 3 channels
fig2, axes2 = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
for i, (ax, name) in enumerate(zip(axes2, STATE_NAMES[:3])):
    ax.plot(t_sim, x_gt_sim[:T_plot, i],  'k',  lw=0.8, label='True (noiseless sim)')
    ax.plot(t_sim, x_b_A[:T_plot, i],     'C0', lw=1.0, label='A: pure-scaled')
    ax.plot(t_sim, x_b_B[:T_plot, i],     'C3--', lw=1.0, label='B: mean-subtracted')
    ax.set_ylabel(f'{name} (norm)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
axes2[-1].set_xlabel('Time [s]')
fig2.suptitle('Noiseless sim: A tracks true state, B is shifted by bias_x')
fig2.tight_layout()
p2 = os.path.join(OUT_DIR, 'diag5_sim_traces.png')
fig2.savefig(p2, dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f'  Saved: {p2}')

# Plot 3: (x_b_A - x_b_B) over time vs analytical bias (flat line)
T_diff = min(500, len(diff_AB_sim))
t_diff = np.arange(T_diff) / FS_NEW
fig3, axes3 = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
for i, (ax, name) in enumerate(zip(axes3, STATE_NAMES[:3])):
    ax.plot(t_diff, diff_AB_sim[:T_diff, i], 'C0', lw=0.8,
            label='(x_b_A - x_b_B)[sim]')
    ax.axhline(bias_x[i], color='C3', lw=1.5, ls='--',
               label=f'bias_x = {bias_x[i]:.4f}')
    ax.set_ylabel(f'{name}')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
axes3[-1].set_xlabel('Time [s]')
fig3.suptitle('(x_b_A - x_b_B) should be flat = bias_x at every timestep (N3)')
fig3.tight_layout()
p3 = os.path.join(OUT_DIR, 'diag5_bias_constancy.png')
fig3.savefig(p3, dpi=150, bbox_inches='tight')
plt.close(fig3)
print(f'  Saved: {p3}')

# =============================================================================
# Summary
# =============================================================================

n_pass = sum(v == 'PASS' for v in results.values())
n_fail = sum(v == 'FAIL' for v in results.values())

print('\n' + '=' * 70)
print(f'  {n_pass}/{len(results)} checks passed')

if n_fail > 0:
    print(f'  FAILED: {[k for k, v in results.items() if v == "FAIL"]}')

# Print severity verdict
max_y_ratio = float(np.abs(y0_ratio).max())
max_u_ratio = float(np.abs(u_mean_ratio).max())
print(f'\n  Severity verdict:')
print(f'    max |y0/std_y|     = {max_y_ratio:.3f}')
print(f'    max |u_mean/std_u| = {max_u_ratio:.3f}')
print(f'    ||bias_x||         = {np.linalg.norm(bias_x):.4f}  (normalized state units)')
if max_y_ratio > 1.0 or max_u_ratio > 1.0:
    print('    VERDICT: operating point is large relative to excitation.')
    print('    Mean subtraction SIGNIFICANTLY degrades W^b initialization.')
    print('    Convention fix is REQUIRED to preserve warm-start quality.')
elif max_y_ratio > 0.2 or max_u_ratio > 0.2:
    print('    VERDICT: operating point is moderate. Fix is recommended.')
else:
    print('    VERDICT: operating point is small. Mean subtraction has minor effect.')

# Save JSON
json_path = os.path.join(OUT_DIR, 'diag5_normalization_convention.json')
with open(json_path, 'w') as f:
    json.dump(dict(
        checks=results,
        severity=dict(
            y0_ratio=y0_ratio.tolist(),
            u_mean_ratio=u_mean_ratio.tolist(),
            bias_x=bias_x.tolist(),
            bias_x_norm=float(np.linalg.norm(bias_x)),
        ),
        nrms_sim=dict(
            A=nrms_sim_A.tolist(),
            B=nrms_sim_B.tolist(),
            C=nrms_sim_C.tolist(),
        ),
        nrms_val=dict(
            A=nrms_val_A.tolist(),
            B=nrms_val_B.tolist(),
            C=nrms_val_C.tolist(),
        ),
        state_names=STATE_NAMES,
    ), f, indent=2)
print(f'\n  Saved: {json_path}')

if n_fail > 0:
    sys.exit(1)
