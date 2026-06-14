"""
encoder_baseline_standalone.py
------------------------------
Encoder standalone validation on BASELINE data (no MSD).

Train the encoder via direct regression ||encoder(u_hist, y_hist) - x_target||^2.
Target: positions from P_inv @ y (exact), velocities from finite-diff of positions.

Additional checks:
1. Velocity verification: Python finite-diff vs MATLAB x_logical velocities.
2. I/O check: feed encoder states through one RK4 step, compare y_hat vs y_measured.

Expected: encoder should match x_logical near-perfectly (no model mismatch).

THEORY: Hoekstra 2026 Eq. 35 — pre-train encoder via state reconstruction loss.

Usage:
    conda run -n GraduationProject python scripts/gantry/encoder/encoder_baseline_standalone.py
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

import deepSI
from model_augmentation.utils.utils import normalize_linear_ss_matrices
from model_augmentation.utils.torch_nets import LinearInitEncoderWrapper
from model_augmentation.fit_systems.pre_encoder import linear_encoder_init
from model_augmentation.fit_systems.blocks import Gantry_State_Block
from model_augmentation.systems.gantry_ss import Cd, P
from model_augmentation.systems.gantry_linearization import gantry_linearize_and_discretize

# =============================================================================
# Configuration
# =============================================================================

NX_PHYS = 6
nu = 3
ny = 3

FS_ORIG = 20000
FS_NEW = 4000
D = FS_ORIG // FS_NEW
TS_NEW = 1.0 / FS_NEW

DTYPE_NP = np.float32
DTYPE_PT = torch.float32

# HEURISTIC: Jan's rule of thumb for encoder history length
na = 4 * NX_PHYS + 1  # = 25
nb = na
na_right = 1
nb_right = 1

HP = dict(
    n_nodes_per_layer=16,
    n_hidden_layers=2,
    lr=1e-3,           # HEURISTIC: higher than pipeline (no rollout instability)
    epochs=200,
    batch_size=256,
)

TRAJ_DIR = os.path.join(PROJECT_ROOT, 'data', 'gantry', 'matlab', 'multisine', 'baseline')

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

OUT_DIR = os.path.join(PROJECT_ROOT, 'simulations', 'gantry_subnet', 'encoder')
os.makedirs(OUT_DIR, exist_ok=True)

# Logical coordinate names
STATE_NAMES = ['X', 'theta', 'Y', 'dX', 'dtheta', 'dY']
PHYS_UNITS = ['m', 'rad', 'm', 'm/s', 'rad/s', 'm/s']

# Stage coordinate names
STAGE_NAMES = ['x1', 'x2', 'Y', 'dx1', 'dx2', 'dY']
STAGE_UNITS = ['m', 'm', 'm', 'm/s', 'm/s', 'm/s']

# P^T maps logical positions to stage positions: y_stage = P^T @ q_logical
P_np = P.numpy().astype(DTYPE_NP)
PT = P_np.T  # (3, 3)


# =============================================================================
# Coordinate transforms
# =============================================================================

def logical_to_stage(x_logical):
    """Convert (N, 6) logical states to (N, 6) stage states via P^T."""
    pos_stage = (PT @ x_logical[:, :3].T).T
    vel_stage = (PT @ x_logical[:, 3:].T).T
    return np.hstack([pos_stage, vel_stage])


# =============================================================================
# Data loading
# =============================================================================

def load_mat(filename):
    """Load u, y, x_logical from .mat file, downsample to FS_NEW."""
    d = loadmat(os.path.join(TRAJ_DIR, filename), squeeze_me=True)
    u = d['u_total'][::D].astype(DTYPE_NP) if 'u_total' in d else d['u'][::D].astype(DTYPE_NP)
    y = d['y'][::D].astype(DTYPE_NP)
    x_logical = d['x_logical'][::D].astype(DTYPE_NP)
    return u, y, x_logical


# =============================================================================
# Velocity computation from positions (reproduces what we'd do on real gantry)
# =============================================================================

def compute_velocities_from_positions(y, P_inv_T):
    """Compute states from measurements only: P_inv @ y for positions,
    central finite-diff for velocities. This is what we can do on the real system."""
    pos = (P_inv_T @ y.T).T  # (N, 3)
    vel = np.zeros_like(pos)
    # THEORY: central difference (matches MATLAB gradient())
    vel[1:-1] = (pos[2:] - pos[:-2]) * (FS_NEW / 2.0)
    vel[0] = (pos[1] - pos[0]) * FS_NEW      # forward diff at start
    vel[-1] = (pos[-1] - pos[-2]) * FS_NEW    # backward diff at end
    return np.hstack([pos, vel])


# =============================================================================
# Normalization (same as gantry_interconnect_dynamic.py)
# =============================================================================

def compute_normalization(train_data):
    """Compute normalization constants from training data.
    Uses Python-computed states (positions from P_inv@y, velocities from finite-diff)
    as the target, NOT MATLAB x_logical directly."""
    P_inv_T = np.linalg.inv(P.numpy().T).astype(DTYPE_NP)

    u_all = np.concatenate([u for u, _, _ in train_data])
    y_all = np.concatenate([y for _, y, _ in train_data])

    # Compute states from measurements (what we'd do on real gantry)
    x_computed_list = []
    for _, y_traj, _ in train_data:
        x_computed_list.append(compute_velocities_from_positions(y_traj, P_inv_T))
    x_all = np.concatenate(x_computed_list)

    x_mean = x_all.mean(axis=0).reshape(NX_PHYS, 1).astype(DTYPE_NP)
    std_x = x_all.std(axis=0).reshape(NX_PHYS, 1).astype(DTYPE_NP) + 1e-8
    std_u = u_all.std(axis=0).reshape(nu, 1).astype(DTYPE_NP) + 1e-8
    u_mean = u_all.mean(axis=0).reshape(nu, 1).astype(DTYPE_NP)
    ystd = y_all.std(axis=0).astype(DTYPE_NP) + 1e-8
    y0 = (Cd.numpy() @ x_mean.flatten()).astype(DTYPE_NP)

    return dict(
        x_mean=x_mean, std_x=std_x, std_u=std_u, u_mean=u_mean,
        ystd=ystd, y0=y0, P_inv_T=P_inv_T,
        u_all=u_all, y_all=y_all, x_all=x_all,
    )


# =============================================================================
# Create windowed data
# =============================================================================

def create_encoder_windows(u, y, x_target, norm):
    """Create (u_hist, y_hist, x_target_norm) windows for encoder training/eval.
    x_target is the Python-computed state (positions + finite-diff velocities)."""
    na_total = na + na_right  # 26
    nb_total = nb + nb_right  # 26
    N = u.shape[0]
    history = max(na_total, nb_total)
    M = N - history

    u_norm = (u - norm['u_mean'].flatten()) / norm['std_u'].flatten()
    y_norm = (y - norm['y0']) / norm['ystd']
    x_norm = (x_target - norm['x_mean'].flatten()) / norm['std_x'].flatten()

    u_hist = np.zeros((M, nb_total, nu), dtype=DTYPE_NP)
    y_hist = np.zeros((M, na_total, ny), dtype=DTYPE_NP)
    x_tgt = np.zeros((M, NX_PHYS), dtype=DTYPE_NP)

    for i in range(M):
        k = history + i
        u_hist[i] = u_norm[k - nb_total + 1: k + 1]
        y_hist[i] = y_norm[k - na_total + 1: k + 1]
        x_tgt[i] = x_norm[k]

    return u_hist, y_hist, x_tgt


# =============================================================================
# Analytical baseline (P_inv + backward finite-diff)
# =============================================================================

def compute_analytical_baseline(y, norm):
    """Analytical state estimates: P_inv for positions, backward finite-diff for velocities."""
    na_total = na + na_right
    nb_total = nb + nb_right
    history = max(na_total, nb_total)
    N = y.shape[0]
    M = N - history

    pos = (norm['P_inv_T'] @ y.T).T
    vel = np.zeros_like(pos)
    vel[1:] = (pos[1:] - pos[:-1]) * FS_NEW
    vel[0] = vel[1]
    x_analytical = np.hstack([pos, vel])

    x_analytical_norm = (x_analytical - norm['x_mean'].flatten()) / norm['std_x'].flatten()
    return x_analytical_norm[history: history + M]


# =============================================================================
# Build encoder (model-based init as warm start)
# =============================================================================

def build_encoder(norm):
    """Build LinearInitEncoderWrapper with linear_encoder_init."""
    Ad, Bd, Cd_dt, Dd_dt = gantry_linearize_and_discretize(dt=TS_NEW)

    sys_data_with_x = deepSI.System_data(u=norm['u_all'], y=norm['y_all'])
    sys_data_with_x.x = norm['x_all']

    Ad_bar, Bd_bar, Cd_bar, Dd_bar = normalize_linear_ss_matrices(
        Ad, Bd, Cd_dt, Dd_dt, sys_data_with_x)

    phys_encoder = linear_encoder_init(
        A=Ad_bar, B=Bd_bar, C=Cd_bar, D=Dd_bar,
        nx=NX_PHYS, nu=nu, ny=ny, na=na, nb=nb,
        n_nodes_per_layer=HP['n_nodes_per_layer'],
        n_hidden_layers=HP['n_hidden_layers'],
        flag_linear_only=False,
    )

    encoder = LinearInitEncoderWrapper(
        phys_encoder=phys_encoder,
        nx_ann=0,
        nb=nb + nb_right, nu=nu, na=na + na_right, ny=ny,
        n_nodes_per_layer=HP['n_nodes_per_layer'],
        n_hidden_layers=HP['n_hidden_layers'],
        u_mean=norm['u_mean'], std_u=norm['std_u'],
        y0=norm['y0'], ystd=norm['ystd'],
        x_mean=norm['x_mean'], std_x=norm['std_x'],
    ).to(DTYPE_PT)

    return encoder


# =============================================================================
# I/O check: one-step output prediction using encoder states
# =============================================================================

def io_check(encoder, u, y, norm):
    """Feed encoder states through one RK4 step, compare predicted output vs measured y."""
    na_total = na + na_right
    nb_total = nb + nb_right
    history = max(na_total, nb_total)
    N = u.shape[0]
    M = N - history - 1

    u_norm = (u - norm['u_mean'].flatten()) / norm['std_u'].flatten()
    y_norm = (y - norm['y0']) / norm['ystd']

    u_hist = np.zeros((M, nb_total, nu), dtype=DTYPE_NP)
    y_hist = np.zeros((M, na_total, ny), dtype=DTYPE_NP)
    u_current = np.zeros((M, nu), dtype=DTYPE_NP)

    for i in range(M):
        k = history + i
        u_hist[i] = u_norm[k - nb_total + 1: k + 1]
        y_hist[i] = y_norm[k - na_total + 1: k + 1]
        u_current[i] = u_norm[k]

    encoder.eval()
    with torch.no_grad():
        x_enc = encoder(
            torch.tensor(u_hist, dtype=DTYPE_PT),
            torch.tensor(y_hist, dtype=DTYPE_PT),
        )

    state_block = Gantry_State_Block(
        Y_op=None,
        std_x=norm['std_x'],
        std_u=norm['std_u'],
        x_mean=norm['x_mean'],
        u_mean=norm['u_mean'],
        Ts=TS_NEW,
        up_sample=1,
    ).to(DTYPE_PT)
    state_block.eval()

    u_cur_t = torch.tensor(u_current, dtype=DTYPE_PT).unsqueeze(-1)
    x_enc_3d = x_enc.unsqueeze(-1)
    z_input = torch.cat([x_enc_3d, u_cur_t], dim=1)

    with torch.no_grad():
        x_next = state_block.nonlinear_function(z_input)

    x_next_phys = x_next.squeeze(-1) * torch.tensor(norm['std_x'].flatten(), dtype=DTYPE_PT) \
                  + torch.tensor(norm['x_mean'].flatten(), dtype=DTYPE_PT)
    Cd_t = torch.tensor(Cd.numpy(), dtype=DTYPE_PT)
    y_hat = (Cd_t @ x_next_phys.unsqueeze(-1)).squeeze(-1).numpy()

    y_next = y[history + 1: history + 1 + M]

    err = y_hat - y_next
    rms_err = np.sqrt(np.mean(err**2, axis=0))
    rms_y = np.sqrt(np.mean(y_next**2, axis=0))
    nrms_io = rms_err / (rms_y + 1e-12)

    return nrms_io, y_hat, y_next


# =============================================================================
# Evaluation
# =============================================================================

def compute_nrms(x_hat, x_target):
    """Per-channel NRMS."""
    err = x_hat - x_target
    rms_err = np.sqrt(np.mean(err**2, axis=0))
    rms_gt = np.sqrt(np.mean(x_target**2, axis=0))
    return rms_err / (rms_gt + 1e-12)


def compute_rms_error(x_hat, x_target):
    """Per-channel RMS error (same units as input)."""
    return np.sqrt(np.mean((x_hat - x_target)**2, axis=0))


def evaluate_encoder(encoder, u_hist, y_hist, x_target):
    """Forward pass and compute NRMS."""
    encoder.eval()
    with torch.no_grad():
        x_hat = encoder(
            torch.tensor(u_hist, dtype=DTYPE_PT),
            torch.tensor(y_hist, dtype=DTYPE_PT),
        ).numpy()
    return x_hat, compute_nrms(x_hat, x_target)


def denormalize_states(x_norm, norm):
    """Convert normalized states back to physical units."""
    return x_norm * norm['std_x'].flatten() + norm['x_mean'].flatten()


# =============================================================================
# Plotting helpers
# =============================================================================

def format_rms_unit(rms_val, unit):
    """Format RMS with appropriate SI prefix."""
    if unit in ('m', 'rad'):
        if rms_val < 1e-3:
            return f'{rms_val*1e6:.1f} \u03bcm' if unit == 'm' else f'{rms_val*1e6:.1f} \u03bcrad'
        elif rms_val < 1.0:
            return f'{rms_val*1e3:.2f} mm' if unit == 'm' else f'{rms_val*1e3:.2f} mrad'
        else:
            return f'{rms_val:.3f} {unit}'
    elif unit in ('m/s', 'rad/s'):
        if rms_val < 1e-3:
            return f'{rms_val*1e6:.1f} \u03bc{unit}'
        elif rms_val < 1.0:
            return f'{rms_val*1e3:.2f} m{unit}'
        else:
            return f'{rms_val:.3f} {unit}'
    return f'{rms_val:.2e} {unit}'


# =============================================================================
# Plotting
# =============================================================================

def plot_loss_curve(train_losses, val_losses, out_path):
    """Training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(train_losses, label='train MSE', linewidth=0.8)
    ax.semilogy(val_losses, label='val MSE', linewidth=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Encoder baseline standalone: training loss')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def plot_comparison(x_enc, x_ana, x_target, nrms_enc, nrms_ana,
                    rms_enc, rms_ana, names, units, title, out_path):
    """Time-domain overlay with NRMS and RMS in legend."""
    n_states = len(names)
    T = min(2000, len(x_enc))
    t = np.arange(T) / FS_NEW

    fig, axes = plt.subplots(n_states, 1, figsize=(14, 2.5 * n_states), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, x_target[:T, i], 'k-', linewidth=0.8, label='target')
        rms_enc_str = format_rms_unit(rms_enc[i], units[i])
        rms_ana_str = format_rms_unit(rms_ana[i], units[i])
        ax.plot(t, x_enc[:T, i], 'r--', linewidth=0.8,
                label=f'encoder (NRMS={nrms_enc[i]:.2e}, RMS={rms_enc_str})')
        ax.plot(t, x_ana[:T, i], 'b:', linewidth=0.8,
                label=f'analytical (NRMS={nrms_ana[i]:.2e}, RMS={rms_ana_str})')
        ax.set_ylabel(f'{names[i]} [{units[i]}]')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time [s]')
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def plot_difference(x_enc, x_ana, x_target, names, units, title, out_path):
    """Difference plot: encoder error vs analytical error (shows improvement margin)."""
    n_states = len(names)
    T = min(2000, len(x_enc))
    t = np.arange(T) / FS_NEW

    fig, axes = plt.subplots(n_states, 1, figsize=(14, 2.5 * n_states), sharex=True)
    for i, ax in enumerate(axes):
        enc_err = x_enc[:T, i] - x_target[:T, i]
        ana_err = x_ana[:T, i] - x_target[:T, i]
        ax.plot(t, enc_err, 'r-', linewidth=0.6, label='encoder error')
        ax.plot(t, ana_err, 'b-', linewidth=0.6, alpha=0.7, label='analytical error')
        ax.set_ylabel(f'{names[i]} error [{units[i]}]')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time [s]')
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def plot_nrms_bar(nrms_before, nrms_after, nrms_ana, names, out_path):
    """Bar chart: NRMS per channel — before, after, analytical."""
    n = len(names)
    x = np.arange(n)
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, nrms_before, w, label='model-based init', color='tab:orange', alpha=0.8)
    ax.bar(x, nrms_after, w, label='after pre-training', color='tab:red', alpha=0.8)
    ax.bar(x + w, nrms_ana, w, label='analytical baseline', color='tab:blue', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel('NRMS')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('Encoder baseline standalone: NRMS comparison')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# =============================================================================
# Main
# =============================================================================

def main():
    print('=' * 70)
    print('Encoder baseline standalone: train + validate (no MSD)')
    print('=' * 70)

    # --- Load data ---
    print(f'\nLoading data from: {TRAJ_DIR}')
    train_data = [load_mat(f) for f in TRAIN_FILES]
    val_u, val_y, val_x_logical = load_mat(VAL_FILE)

    for i, (fname, (u, y, x)) in enumerate(zip(TRAIN_FILES, train_data)):
        print(f'  T{i+1} ({fname}): u={u.shape}, y={y.shape}, x_logical={x.shape}')
    print(f'  Val ({VAL_FILE}): u={val_u.shape}, y={val_y.shape}')

    # --- Normalization ---
    norm = compute_normalization(train_data)
    print(f'\nNormalization:')
    print(f'  std_x = {norm["std_x"].flatten()}')
    print(f'  std_u = {norm["std_u"].flatten()}')
    print(f'  ystd  = {norm["ystd"]}')

    # =================================================================
    # CHECK 1: Velocity verification — Python finite-diff vs MATLAB
    # =================================================================
    print('\n--- CHECK 1: Velocity verification (Python vs MATLAB) ---')
    x_python = compute_velocities_from_positions(val_y, norm['P_inv_T'])
    vel_python = x_python[:, 3:]
    vel_matlab = val_x_logical[:, 3:]
    vel_diff_rms = np.sqrt(np.mean((vel_python - vel_matlab)**2, axis=0))
    vel_matlab_rms = np.sqrt(np.mean(vel_matlab**2, axis=0))
    vel_nrms = vel_diff_rms / (vel_matlab_rms + 1e-12)
    print(f'  {"Channel":<8s}  {"RMS diff":>14s}  {"NRMS":>10s}')
    for i, name in enumerate(STATE_NAMES[3:]):
        print(f'  {name:<8s}  {vel_diff_rms[i]:>14.4e}  {vel_nrms[i]:>10.4e}')
    print(f'  Max velocity NRMS (Python vs MATLAB): {np.max(vel_nrms):.4e}')
    if np.max(vel_nrms) < 0.01:
        print('  OK: Python velocities match MATLAB within 1%.')
    else:
        print('  WARNING: Python velocities differ from MATLAB significantly.')

    # Use Python-computed states as target (reproducible from measurements)
    x_target_all = []
    for _, y_traj, _ in train_data:
        x_target_all.append(compute_velocities_from_positions(y_traj, norm['P_inv_T']))
    val_x_target = compute_velocities_from_positions(val_y, norm['P_inv_T'])

    # --- Create windowed datasets ---
    print('\nCreating windows...')
    train_windows = []
    for (u, y, _), x_tgt in zip(train_data, x_target_all):
        uh, yh, xt = create_encoder_windows(u, y, x_tgt, norm)
        train_windows.append((uh, yh, xt))

    u_train = np.concatenate([w[0] for w in train_windows])
    y_train = np.concatenate([w[1] for w in train_windows])
    x_train = np.concatenate([w[2] for w in train_windows])
    print(f'  Training windows: {len(u_train)}')

    u_val, y_val, x_val = create_encoder_windows(val_u, val_y, val_x_target, norm)
    print(f'  Validation windows: {len(u_val)}')

    # --- Build encoder (model-based init as warm start) ---
    print('\nBuilding encoder (model-based init)...')
    encoder = build_encoder(norm)
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f'  Parameters: {n_params}')

    # --- Evaluate BEFORE pre-training ---
    print('\n--- BEFORE pre-training ---')
    x_hat_before, nrms_before = evaluate_encoder(encoder, u_val, y_val, x_val)
    x_hat_ana = compute_analytical_baseline(val_y, norm)
    nrms_ana = compute_nrms(x_hat_ana, x_val)

    print(f'  {"State":<8s}  {"Model-based init":>18s}  {"Analytical":>14s}')
    print(f'  {"-"*8}  {"-"*18}  {"-"*14}')
    for i, name in enumerate(STATE_NAMES):
        print(f'  {name:<8s}  {nrms_before[i]:>18.4e}  {nrms_ana[i]:>14.4e}')
    print(f'  Max NRMS: {np.max(nrms_before):.4e}')

    # --- Training ---
    print(f'\n--- Training ({HP["epochs"]} epochs, lr={HP["lr"]}, batch={HP["batch_size"]}) ---')
    encoder.train()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=HP['lr'])
    criterion = nn.MSELoss()

    u_train_t = torch.tensor(u_train, dtype=DTYPE_PT)
    y_train_t = torch.tensor(y_train, dtype=DTYPE_PT)
    x_train_t = torch.tensor(x_train, dtype=DTYPE_PT)
    u_val_t = torch.tensor(u_val, dtype=DTYPE_PT)
    y_val_t = torch.tensor(y_val, dtype=DTYPE_PT)
    x_val_t = torch.tensor(x_val, dtype=DTYPE_PT)

    N_train = len(u_train_t)
    batch_size = HP['batch_size']
    train_losses = []
    val_losses = []

    t_start = time.time()
    for epoch in range(HP['epochs']):
        perm = torch.randperm(N_train)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, N_train, batch_size):
            idx = perm[start: start + batch_size]
            ub = u_train_t[idx]
            yb = y_train_t[idx]
            xb = x_train_t[idx]

            x_hat = encoder(ub, yb)
            loss = criterion(x_hat, xb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches
        train_losses.append(avg_train_loss)

        encoder.eval()
        with torch.no_grad():
            x_hat_val = encoder(u_val_t, y_val_t)
            val_loss = criterion(x_hat_val, x_val_t).item()
        val_losses.append(val_loss)
        encoder.train()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            elapsed = time.time() - t_start
            print(f'  Epoch {epoch+1:4d}/{HP["epochs"]}  '
                  f'train={avg_train_loss:.4e}  val={val_loss:.4e}  '
                  f'[{elapsed:.0f}s]')

    elapsed_total = time.time() - t_start
    print(f'\nTraining complete in {elapsed_total:.0f}s')

    # --- Evaluate AFTER pre-training ---
    print('\n--- AFTER pre-training ---')
    x_hat_after, nrms_after = evaluate_encoder(encoder, u_val, y_val, x_val)

    print(f'  {"State":<8s}  {"Before":>14s}  {"After":>14s}  {"Analytical":>14s}')
    print(f'  {"-"*8}  {"-"*14}  {"-"*14}  {"-"*14}')
    for i, name in enumerate(STATE_NAMES):
        print(f'  {name:<8s}  {nrms_before[i]:>14.4e}  {nrms_after[i]:>14.4e}  {nrms_ana[i]:>14.4e}')

    print(f'\n  Max NRMS before:     {np.max(nrms_before):.4e}')
    print(f'  Max NRMS after:      {np.max(nrms_after):.4e}')
    print(f'  Max NRMS analytical: {np.max(nrms_ana):.4e}')

    # Physical-unit RMS error (logical coordinates)
    x_after_phys = denormalize_states(x_hat_after, norm)
    x_ana_phys = denormalize_states(x_hat_ana, norm)
    x_gt_phys = denormalize_states(x_val, norm)
    rms_enc = compute_rms_error(x_after_phys, x_gt_phys)
    rms_ana = compute_rms_error(x_ana_phys, x_gt_phys)

    print(f'\n  Per-channel RMS error (logical coordinates):')
    print(f'  {"State":<8s}  {"Encoder":>14s}  {"Analytical":>14s}  {"Unit":>6s}')
    for i, name in enumerate(STATE_NAMES):
        print(f'  {name:<8s}  {rms_enc[i]:>14.4e}  {rms_ana[i]:>14.4e}  {PHYS_UNITS[i]:>6s}')

    # Physical-unit RMS error (stage coordinates)
    x_after_stage = logical_to_stage(x_after_phys)
    x_ana_stage = logical_to_stage(x_ana_phys)
    x_gt_stage = logical_to_stage(x_gt_phys)
    rms_enc_stage = compute_rms_error(x_after_stage, x_gt_stage)
    rms_ana_stage = compute_rms_error(x_ana_stage, x_gt_stage)
    nrms_after_stage = compute_nrms(x_after_stage, x_gt_stage)
    nrms_ana_stage = compute_nrms(x_ana_stage, x_gt_stage)

    print(f'\n  Per-channel RMS error (stage coordinates):')
    print(f'  {"State":<8s}  {"Encoder":>14s}  {"Analytical":>14s}  {"Unit":>6s}')
    for i, name in enumerate(STAGE_NAMES):
        print(f'  {name:<8s}  {rms_enc_stage[i]:>14.4e}  {rms_ana_stage[i]:>14.4e}  {STAGE_UNITS[i]:>6s}')

    # =================================================================
    # CHECK 2: I/O check — one-step output prediction
    # =================================================================
    print('\n--- CHECK 2: I/O check (one-step output prediction) ---')
    nrms_io, y_hat_io, y_next_io = io_check(encoder, val_u, val_y, norm)
    print(f'  {"Output":<6s}  {"1-step NRMS":>14s}')
    print(f'  {"-"*6}  {"-"*14}')
    for i, name in enumerate(STAGE_NAMES[:3]):
        print(f'  {name:<6s}  {nrms_io[i]:>14.4e}')
    print(f'  Max I/O NRMS: {np.max(nrms_io):.4e}')
    if np.max(nrms_io) < 0.01:
        print('  OK: Encoder states are pipeline-compatible (1-step output NRMS < 1%).')
    else:
        print('  WARNING: Encoder states produce poor 1-step output predictions.')

    # --- Verdict ---
    beats_analytical = np.all(nrms_after < nrms_ana)
    beats_init = np.max(nrms_after) < np.max(nrms_before)

    print('\n--- VERDICT ---')
    if beats_analytical:
        print('  PASS: Pre-trained encoder beats analytical baseline on ALL channels.')
    elif beats_init:
        print('  IMPROVED: Pre-training improved over model-based init.')
        worse = [STATE_NAMES[i] for i in range(NX_PHYS) if nrms_after[i] >= nrms_ana[i]]
        print(f'  Still worse than analytical on: {", ".join(worse)}')
        # Note about positions
        pos_worse = [STATE_NAMES[i] for i in range(3) if nrms_after[i] >= nrms_ana[i]]
        vel_better = [STATE_NAMES[i] for i in range(3, 6) if nrms_after[i] < nrms_ana[i]]
        if pos_worse and vel_better:
            print(f'  NOTE: Positions have NRMS>0 because encoder uses a window (not exact inversion).')
            print(f'        Analytical positions are exact (P_inv @ y = 0 error).')
            print(f'        Velocities (the hard part) are better: {", ".join(vel_better)}')
    else:
        print('  NO IMPROVEMENT: Pre-training did not improve over model-based init.')

    # --- Save encoder weights ---
    weights_path = os.path.join(OUT_DIR, 'encoder_baseline_weights.pt')
    torch.save(encoder.state_dict(), weights_path)
    print(f'\nSaved encoder weights: {weights_path}')

    # --- Save results ---
    results = dict(
        nrms_before={name: float(nrms_before[i]) for i, name in enumerate(STATE_NAMES)},
        nrms_after={name: float(nrms_after[i]) for i, name in enumerate(STATE_NAMES)},
        nrms_analytical={name: float(nrms_ana[i]) for i, name in enumerate(STATE_NAMES)},
        rms_enc_logical={name: float(rms_enc[i]) for i, name in enumerate(STATE_NAMES)},
        rms_ana_logical={name: float(rms_ana[i]) for i, name in enumerate(STATE_NAMES)},
        rms_enc_stage={name: float(rms_enc_stage[i]) for i, name in enumerate(STAGE_NAMES)},
        rms_ana_stage={name: float(rms_ana_stage[i]) for i, name in enumerate(STAGE_NAMES)},
        nrms_io={name: float(nrms_io[i]) for i, name in enumerate(STAGE_NAMES[:3])},
        vel_verification_nrms={name: float(vel_nrms[i]) for i, name in enumerate(STATE_NAMES[3:])},
        n_params=n_params,
        hp=HP,
        train_time_s=elapsed_total,
        final_train_loss=train_losses[-1],
        final_val_loss=val_losses[-1],
    )
    json_path = os.path.join(OUT_DIR, 'encoder_baseline_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved: {json_path}')

    # Save trajectories
    npz_path = os.path.join(OUT_DIR, 'encoder_baseline_data.npz')
    np.savez_compressed(npz_path,
        x_before_norm=x_hat_before, x_after_norm=x_hat_after,
        x_analytical_norm=x_hat_ana, x_target_norm=x_val,
        nrms_before=nrms_before, nrms_after=nrms_after, nrms_analytical=nrms_ana,
        train_losses=train_losses, val_losses=val_losses,
        std_x=norm['std_x'], x_mean=norm['x_mean'],
        state_names=STATE_NAMES, stage_names=STAGE_NAMES, fs=FS_NEW,
    )
    print(f'Saved: {npz_path}')

    # --- Plots ---
    # Compute before-training physical states for bar chart
    x_before_phys = denormalize_states(x_hat_before, norm)
    rms_before = compute_rms_error(x_before_phys, x_gt_phys)

    plot_loss_curve(train_losses, val_losses,
                    os.path.join(OUT_DIR, 'encoder_baseline_loss.png'))

    # Logical coordinate plots
    plot_comparison(x_after_phys, x_ana_phys, x_gt_phys, nrms_after, nrms_ana,
                    rms_enc, rms_ana, STATE_NAMES, PHYS_UNITS,
                    'Encoder baseline (logical coords): encoder vs analytical',
                    os.path.join(OUT_DIR, 'encoder_baseline_comparison_logical.png'))

    plot_difference(x_after_phys, x_ana_phys, x_gt_phys, STATE_NAMES, PHYS_UNITS,
                    'Encoder baseline (logical coords): error comparison',
                    os.path.join(OUT_DIR, 'encoder_baseline_difference_logical.png'))

    # Stage coordinate plots
    plot_comparison(x_after_stage, x_ana_stage, x_gt_stage,
                    nrms_after_stage, nrms_ana_stage,
                    rms_enc_stage, rms_ana_stage, STAGE_NAMES, STAGE_UNITS,
                    'Encoder baseline (stage coords): encoder vs analytical',
                    os.path.join(OUT_DIR, 'encoder_baseline_comparison_stage.png'))

    plot_difference(x_after_stage, x_ana_stage, x_gt_stage, STAGE_NAMES, STAGE_UNITS,
                    'Encoder baseline (stage coords): error comparison',
                    os.path.join(OUT_DIR, 'encoder_baseline_difference_stage.png'))

    # NRMS bar chart (logical)
    plot_nrms_bar(nrms_before, nrms_after, nrms_ana, STATE_NAMES,
                  os.path.join(OUT_DIR, 'encoder_baseline_nrms_bar.png'))

    print('\n' + '=' * 70)
    print('Encoder baseline standalone complete.')


if __name__ == '__main__':
    main()
