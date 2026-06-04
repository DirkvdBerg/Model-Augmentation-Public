"""
gantry_baseline_validation.py

Validates the Python Gantry_State_Block physics against MATLAB ground truth.

Normalization  : computed from gantry_comb_baseline (Y moves, meaningful std).
Norm checks    : algebraic Cd @ x_logical vs y_ref, one-step teacher forcing (LTI block).
Datasets       : gantry_lti_train (Y fixed), gantry_comb_baseline (Y varying, baseline system),
                 gantry_comb_augmented (Y varying, augmented system with hidden MSD).

Two blocks (LTI: Y_op frozen, LPV: Y_op=None) run on each dataset.
Comparison plot: baseline vs augmented error side-by-side (same y-scale per channel).
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')   # headless - no display required (cluster safe)
import matplotlib.pyplot as plt
from scipy.io import loadmat
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from model_augmentation.systems.gantry_ss import Cd
from model_augmentation.fit_systems.blocks import Gantry_State_Block

# ── Constants ──────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'gantry', 'matlab')
OUT_DIR     = os.path.join(os.path.dirname(__file__), '..', '..', 'simulations', 'baseline_validation')
Cd_np       = Cd.numpy()   # (3, 6)  Dd = 0, no feedthrough
MAX_SAMPLES = None         # cap rollout to this many steps (None = full dataset)
run_id      = os.environ.get('SLURM_JOB_ID') or datetime.now().strftime('%Y%m%d_%H%M%S')

os.makedirs(OUT_DIR, exist_ok=True)

CH_LABELS    = ['X1 [m]', 'X2 [m]', 'Y [m]']
U_LABELS     = ['F_X1 [N]', 'F_X2 [N]', 'F_Y [N]']
STATE_LABELS = ['q1 (mean-X) [m]', 'q2 (arm-rot) [rad]', 'q3 (Y) [m]',
                'dq1 [m/s]',       'dq2 [rad/s]',         'dq3 [m/s]']


# ── Data loading ───────────────────────────────────────────────────────────────
def load_mat(name, max_samples=MAX_SAMPLES):
    d = loadmat(os.path.join(DATA_DIR, f'{name}.mat'), squeeze_me=True)
    s = slice(None, max_samples)   # slice(None, None) = full dataset
    return {
        'u':         d['u'][s].astype(np.float32),
        'y':         d['y'][s].astype(np.float32),
        'x_logical': d['x_logical'][s].astype(np.float32),
        'dt':        float(d['dt']),
        'name':      name,
    }


# ── Section 1: Normalization from gantry_comb_baseline ────────────────────────
# Always use the FULL dataset for normalization - Y movement may occur late in
# the trajectory, and capping to MAX_SAMPLES could miss it entirely.
comb_full = load_mat('gantry_comb_baseline', max_samples=None)
comb      = load_mat('gantry_comb_baseline')   # capped - used for checks and rollout

x_mean = comb_full['x_logical'].mean(axis=0)         # (6,)  captures Y_op from data
std_x  = comb_full['x_logical'].std(axis=0) + 1e-8  # (6,)  std around mean
std_u  = comb_full['u'].std(axis=0)         + 1e-8  # (3,)
ystd   = comb_full['y'].std(axis=0)         + 1e-8  # (3,)  for NRMS only

print('=== Section 1: Normalization parameters (source: gantry_comb_baseline, full) ===')
print(f'  x_mean : {np.array2string(x_mean, precision=4)}')
print(f'  std_x  : {np.array2string(std_x,  precision=4)}')
print(f'  std_u  : {np.array2string(std_u,  precision=4)}')
print(f'  ystd   : {np.array2string(ystd,   precision=4)}')

x_norm_full = (comb_full['x_logical'] - x_mean) / std_x
u_norm_full =  comb_full['u'] / std_u

print('\n  Normalized state statistics - full dataset (target: mean≈0, std≈1, range≈[-3, 3])')
print(f'  {"Channel":<22} {"mean":>7} {"std":>7} {"min":>7} {"max":>7}')
for i, lbl in enumerate(STATE_LABELS):
    c = x_norm_full[:, i]
    print(f'  {lbl:<22} {c.mean():>7.3f} {c.std():>7.3f} {c.min():>7.3f} {c.max():>7.3f}')

print(f'\n  {"Input":<22} {"mean":>7} {"std":>7} {"min":>7} {"max":>7}')
for i, lbl in enumerate(U_LABELS):
    c = u_norm_full[:, i]
    print(f'  {lbl:<22} {c.mean():>7.3f} {c.std():>7.3f} {c.min():>7.3f} {c.max():>7.3f}')


# ── Build blocks ───────────────────────────────────────────────────────────────
Y_op = float(x_mean[2])
print(f'\n  Y_op (from data mean): {Y_op:.4f} m')

_block_kwargs = dict(std_x=std_x.reshape(6, 1), std_u=std_u.reshape(3, 1),
                     x_mean=x_mean.reshape(6, 1))

block_lti = Gantry_State_Block(Y_op=Y_op,  **_block_kwargs)
block_lpv = Gantry_State_Block(Y_op=None,  **_block_kwargs)
block_lti.eval()
block_lpv.eval()


# ── Section 2a: Algebraic output check ────────────────────────────────────────
# x_logical positions are derived in MATLAB as (P^T)^{-1} @ y_stage,
# so Cd @ x_logical = P^T @ q = y_stage exactly by construction.
# Non-zero RMS here means a bug in Cd or x_logical construction.
# Use full dataset - includes Y movement, exercises all entries of Cd.
y_alg   = (Cd_np @ comb_full['x_logical'].T).T   # (T, 3)
err_alg = y_alg - comb_full['y']

print('\n=== Section 2a: Algebraic check  Cd @ x_logical vs y  (expect ~0) ===')
for i, lbl in enumerate(CH_LABELS):
    rms = np.sqrt((err_alg[:, i] ** 2).mean())
    mx  = np.abs(err_alg[:, i]).max()
    print(f'  {lbl}: RMS={rms:.3e}  max={mx:.3e}')


# ── Section 2b: One-step teacher forcing (LTI block) ──────────────────────────
# Feed ground-truth x[t] and u[t], predict x[t+1].
# Batched over all T-1 steps - each step is independent so batch is valid.
# Velocity channels have MATLAB numerical-diff noise; small errors expected there.
x_norm_comb = (comb['x_logical'] - x_mean) / std_x  # (T, 6) capped
u_norm_comb =  comb['u'] / std_u                     # (T, 3) capped
T_cap = len(comb['u'])

x_in = torch.tensor(x_norm_comb[:-1].reshape(T_cap - 1, 6, 1), dtype=torch.float32)
u_in = torch.tensor(u_norm_comb[:-1].reshape(T_cap - 1, 3, 1), dtype=torch.float32)
z_in = torch.cat([x_in, u_in], dim=1)   # (T-1, 9, 1)

with torch.no_grad():
    x_pred_norm = block_lti.nonlinear_function(z_in).squeeze(-1).numpy()  # (T-1, 6)

x_pred   = x_pred_norm * std_x + x_mean
err_step = x_pred - comb['x_logical'][1:]

print('\n=== Section 2b: One-step teacher forcing RMS (LTI block) ===')
print('  (velocity channels: small errors expected from MATLAB numerical diff)')
for i, lbl in enumerate(STATE_LABELS):
    print(f'  {lbl}: {np.sqrt((err_step[:, i] ** 2).mean()):.3e}')


# ── Rollout ────────────────────────────────────────────────────────────────────
def rollout(data, block, label):
    x_logical, u = data['x_logical'], data['u']
    T = len(u)

    x_norm   = torch.tensor(
        ((x_logical[0] - x_mean) / std_x).reshape(1, 6, 1), dtype=torch.float32
    )
    u_tensor = torch.tensor(
        (u / std_u).reshape(T, 3, 1), dtype=torch.float32
    )
    y_hat = np.zeros((T, 3), dtype=np.float32)

    print(f'  [{label}] Rolling out {T} steps ({T * data["dt"]:.2f} s)...')
    log_every = max(1, T // 10)

    with torch.no_grad():
        for t in range(T):
            x_phys   = x_norm.squeeze().numpy() * std_x + x_mean   # (6,)
            y_hat[t] = Cd_np @ x_phys                               # (3,)
            z        = torch.cat([x_norm, u_tensor[t:t+1]], dim=1)  # (1, 9, 1)
            x_norm   = block.nonlinear_function(z)                   # (1, 6, 1)
            if (t + 1) % log_every == 0:
                print(f'    {t + 1}/{T}')

    return y_hat


# ── Plot ───────────────────────────────────────────────────────────────────────
def plot_validation(data, y_lti, y_lpv, label):
    t    = np.arange(len(data['u'])) * data['dt']
    u    = data['u']
    y    = data['y']
    slug = data['name']

    nrms_lti = np.sqrt(((y_lti - y) ** 2).mean(axis=0)) / ystd
    nrms_lpv = np.sqrt(((y_lpv - y) ** 2).mean(axis=0)) / ystd

    # Inputs
    fig, axes = plt.subplots(3, 1, figsize=(12, 5), sharex=True)
    for i, (ax, lbl) in enumerate(zip(axes, U_LABELS)):
        ax.plot(t, u[:, i], 'k', lw=0.7)
        ax.set_ylabel(lbl); ax.grid(True)
    axes[0].set_title(f'Inputs: {label}')
    axes[-1].set_xlabel('Time [s]')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'{slug}_inputs_{run_id}.png'), dpi=150)
    plt.close(fig)

    # Trajectories (left) and differences (right)
    fig, axes = plt.subplots(3, 2, figsize=(14, 7), sharex=True)
    for i, lbl in enumerate(CH_LABELS):
        axes[i, 0].plot(t, y[:, i],     'k',  lw=0.9, label='MATLAB')
        axes[i, 0].plot(t, y_lti[:, i], 'C0', lw=0.9, linestyle='--',
                        label=f'LTI  NRMS={nrms_lti[i]:.3e}')
        axes[i, 0].plot(t, y_lpv[:, i], 'C1', lw=0.9, linestyle='-.',
                        label=f'LPV  NRMS={nrms_lpv[i]:.3e}')
        axes[i, 0].set_ylabel(lbl); axes[i, 0].legend(fontsize=7); axes[i, 0].grid(True)

        axes[i, 1].plot(t, y_lti[:, i] - y[:, i], 'C0', lw=0.7, label='LTI')
        axes[i, 1].plot(t, y_lpv[:, i] - y[:, i], 'C1', lw=0.7, label='LPV')
        axes[i, 1].axhline(0, color='k', lw=0.5, linestyle='--')
        axes[i, 1].set_ylabel(f'Δ{lbl}'); axes[i, 1].legend(fontsize=7); axes[i, 1].grid(True)

    axes[0, 0].set_title(f'Trajectories: {label}')
    axes[0, 1].set_title(f'Difference (Python - MATLAB): {label}')
    axes[-1, 0].set_xlabel('Time [s]')
    axes[-1, 1].set_xlabel('Time [s]')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'{slug}_trajectories_{run_id}.png'), dpi=150)
    plt.close(fig)

    # Differences per channel - LTI-MATLAB and LPV-MATLAB combined
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    for i, (ax, lbl) in enumerate(zip(axes, CH_LABELS)):
        ax.plot(t, y_lti[:, i] - y[:, i], 'C0', lw=0.8, label=f'LTI - MATLAB  NRMS={nrms_lti[i]:.3e}')
        ax.plot(t, y_lpv[:, i] - y[:, i], 'C1', lw=0.8, label=f'LPV - MATLAB  NRMS={nrms_lpv[i]:.3e}')
        ax.axhline(0, color='k', lw=0.5, linestyle='--')
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.set_ylabel(f'Δ{lbl}'); ax.legend(fontsize=7); ax.grid(True)
    axes[0].set_title(f'Model error vs MATLAB: {label}')
    axes[-1].set_xlabel('Time [s]')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'{slug}_differences_{run_id}.png'), dpi=150)
    plt.close(fig)

    return nrms_lti, nrms_lpv


# ── Rollout and per-dataset plots ─────────────────────────────────────────────
datasets = [
    ('gantry_lti_train',       'Baseline system, Y fixed'),
    ('gantry_comb_baseline',   'Baseline system, Y varying'),
    ('gantry_comb_augmented',  'Augmented system, Y varying'),
]

nrms_table   = {}   # {name: (nrms_lti, nrms_lpv)}
results_store = {}  # {name: dict} — y arrays kept for comparison plot

for name, label in datasets:
    print(f'\n=== Rollout: {label} ===')
    data  = load_mat(name)
    y_lti = rollout(data, block_lti, 'LTI')
    y_lpv = rollout(data, block_lpv, 'LPV')
    nrms_lti, nrms_lpv = plot_validation(data, y_lti, y_lpv, label)
    nrms_table[name]    = (nrms_lti, nrms_lpv)
    results_store[name] = {
        'y': data['y'], 'y_lti': y_lti, 'y_lpv': y_lpv,
        'nrms_lti': nrms_lti, 'nrms_lpv': nrms_lpv,
        'dt': data['dt'],
    }

# ── Comparison plot: baseline vs augmented (same y-scale per channel) ─────────
def plot_comparison(res_base, res_aug):
    fig, axes = plt.subplots(3, 2, figsize=(14, 7), sharex='col', sharey='row')
    sys_labels = ['Baseline system, Y varying', 'Augmented system, Y varying']
    for col, res in enumerate([res_base, res_aug]):
        t = np.arange(len(res['y'])) * res['dt']
        for i, ch_lbl in enumerate(CH_LABELS):
            ax = axes[i, col]
            ax.plot(t, res['y_lti'][:, i] - res['y'][:, i], 'C0', lw=0.8,
                    label=f'LTI - MATLAB  NRMS={res["nrms_lti"][i]:.3e}')
            ax.plot(t, res['y_lpv'][:, i] - res['y'][:, i], 'C1', lw=0.8,
                    label=f'LPV - MATLAB  NRMS={res["nrms_lpv"][i]:.3e}')
            ax.axhline(0, color='k', lw=0.5, linestyle='--')
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            ax.set_ylabel(f'D{ch_lbl}')
            ax.legend(fontsize=7)
            ax.grid(True)
        axes[0, col].set_title(sys_labels[col])
        axes[-1, col].set_xlabel('Time [s]')
    fig.suptitle('Model error: baseline vs augmented system')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f'comparison_baseline_vs_augmented_{run_id}.png'), dpi=150)
    plt.close(fig)

plot_comparison(results_store['gantry_comb_baseline'], results_store['gantry_comb_augmented'])

# ── Summary NRMS table ────────────────────────────────────────────────────────
print('\n=== NRMS Summary ===')
col_w = 18
print(f'  {"":22}', end='')
for name, _ in datasets:
    short = name.replace('gantry_', '')
    print(f'  {"LTI":>{col_w}}  {"LPV":>{col_w}}', end='')
print()
print(f'  {"Channel":<22}', end='')
for name, _ in datasets:
    short = name.replace('gantry_', '')
    print(f'  {short+" LTI":>{col_w}}  {short+" LPV":>{col_w}}', end='')
print()
for i, lbl in enumerate(CH_LABELS):
    print(f'  {lbl:<22}', end='')
    for name, _ in datasets:
        lti, lpv = nrms_table[name]
        print(f'  {lti[i]:>{col_w}.3e}  {lpv[i]:>{col_w}.3e}', end='')
    print()

print(f'\nPlots saved to: {OUT_DIR}')
