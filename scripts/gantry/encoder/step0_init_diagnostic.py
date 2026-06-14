"""
step0_init_diagnostic.py
------------------------
Encoder verification Step 0: check initialization quality (no training).

Tests whether linear_encoder_init produces states close to ground truth
at initialization, BEFORE any training. Jan: "encoder initialize close
to perfect."

Also computes the analytical baseline (P_inv + finite-diff) as reference.
Both are compared against x_logical from the .mat files.

If init NRMS >> 1e-3, debug normalization/matrices/windowing before proceeding.

Usage:
    conda run -n GraduationProject python scripts/gantry/encoder/step0_init_diagnostic.py
"""

import os
import sys
import json
import numpy as np
import torch
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
)

MODE = 'multisine'
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

STATE_NAMES = ['q1', 'q2', 'q3', 'dq1', 'dq2', 'dq3']

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
# Normalization (same as gantry_interconnect_dynamic.py)
# =============================================================================

def compute_normalization(train_data):
    """Compute normalization constants from training data.

    Uses x_logical from .mat files (not finite-diff) as state reference.
    """
    P_inv_T = np.linalg.inv(P.numpy().T).astype(DTYPE_NP)

    u_all = np.concatenate([u for u, _, _ in train_data])
    y_all = np.concatenate([y for _, y, _ in train_data])
    x_all = np.concatenate([x for _, _, x in train_data])

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

def create_encoder_windows(u, y, x_logical, norm):
    """Create (u_hist, y_hist, x_target_norm) windows for encoder evaluation.

    The encoder sees normalized (u, y) and outputs normalized states.
    x_target is normalized x_logical for comparison.
    """
    na_total = na + na_right  # 26
    nb_total = nb + nb_right  # 26
    N = u.shape[0]
    history = max(na_total, nb_total)
    M = N - history

    # Normalize
    u_norm = (u - norm['u_mean'].flatten()) / norm['std_u'].flatten()
    y_norm = (y - norm['y0']) / norm['ystd']
    x_norm = (x_logical - norm['x_mean'].flatten()) / norm['std_x'].flatten()

    u_hist = np.zeros((M, nb_total, nu), dtype=DTYPE_NP)
    y_hist = np.zeros((M, na_total, ny), dtype=DTYPE_NP)
    x_target = np.zeros((M, NX_PHYS), dtype=DTYPE_NP)

    for i in range(M):
        k = history + i
        u_hist[i] = u_norm[k - nb_total + 1: k + 1]
        y_hist[i] = y_norm[k - na_total + 1: k + 1]
        x_target[i] = x_norm[k]

    return u_hist, y_hist, x_target


# =============================================================================
# Analytical baseline (P_inv + finite-diff)
# =============================================================================

def compute_analytical_baseline(y, norm):
    """Compute analytical state estimates: P_inv for positions, backward FD for velocities.

    This is what HybridGantryEncoder does (without the extrapolation).
    Returns normalized states for the same time indices as the encoder windows.
    """
    na_total = na + na_right
    nb_total = nb + nb_right
    history = max(na_total, nb_total)
    N = y.shape[0]
    M = N - history

    P_inv_T = norm['P_inv_T']

    # THEORY: q_logical = P_inv @ y_stage (measurement equation)
    pos = (P_inv_T @ y.T).T  # (N, 3)

    # HEURISTIC: backward finite difference for velocity, O(Ts) accurate
    vel = np.zeros_like(pos)
    vel[1:] = (pos[1:] - pos[:-1]) * FS_NEW
    vel[0] = vel[1]

    x_analytical = np.hstack([pos, vel])  # (N, 6)

    # Normalize
    x_analytical_norm = (x_analytical - norm['x_mean'].flatten()) / norm['std_x'].flatten()

    # Select same time indices as encoder windows
    x_analytical_windows = x_analytical_norm[history: history + M]

    return x_analytical_windows


# =============================================================================
# Build encoder (same as build_model in gantry_interconnect_dynamic.py)
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
        nx_ann=0,  # no augmented states for diagnostic
        nb=nb + nb_right, nu=nu, na=na + na_right, ny=ny,
        n_nodes_per_layer=HP['n_nodes_per_layer'],
        n_hidden_layers=HP['n_hidden_layers'],
        u_mean=norm['u_mean'], std_u=norm['std_u'],
        y0=norm['y0'], ystd=norm['ystd'],
        x_mean=norm['x_mean'], std_x=norm['std_x'],
    ).to(DTYPE_PT)

    return encoder


# =============================================================================
# Evaluation
# =============================================================================

PHYS_UNITS = ['m', 'm', 'm', 'm/s', 'm/s', 'm/s']


def compute_nrms(x_hat, x_target):
    """Per-channel NRMS."""
    err = x_hat - x_target
    rms_err = np.sqrt(np.mean(err**2, axis=0))
    rms_gt = np.sqrt(np.mean(x_target**2, axis=0))
    return rms_err / (rms_gt + 1e-12)


def compute_rms_error(x_hat, x_target):
    """Per-channel RMS error (same units as input)."""
    return np.sqrt(np.mean((x_hat - x_target)**2, axis=0))


# =============================================================================
# Plotting
# =============================================================================

def plot_comparison(x_encoder, x_analytical, x_target, nrms_enc, nrms_ana, out_path):
    """Time-domain: encoder init vs analytical baseline vs ground truth."""
    T = min(2000, len(x_encoder))
    t = np.arange(T) / FS_NEW

    fig, axes = plt.subplots(NX_PHYS, 1, figsize=(14, 2.5 * NX_PHYS), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, x_target[:T, i], 'k-', linewidth=0.8, label='x_logical (ground truth)')
        ax.plot(t, x_encoder[:T, i], 'r--', linewidth=0.8,
                label=f'encoder init (NRMS={nrms_enc[i]:.2e})')
        ax.plot(t, x_analytical[:T, i], 'b:', linewidth=0.8,
                label=f'analytical (NRMS={nrms_ana[i]:.2e})')
        ax.set_ylabel(STATE_NAMES[i])
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time [s]')
    fig.suptitle('Step 0: Init diagnostic (no training)', y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def plot_error(x_encoder, x_analytical, x_target, out_path):
    """Error time series: encoder init error vs analytical error."""
    T = min(2000, len(x_encoder))
    t = np.arange(T) / FS_NEW

    fig, axes = plt.subplots(NX_PHYS, 1, figsize=(14, 2.5 * NX_PHYS), sharex=True)
    for i, ax in enumerate(axes):
        err_enc = x_encoder[:T, i] - x_target[:T, i]
        err_ana = x_analytical[:T, i] - x_target[:T, i]
        ax.plot(t, err_enc, 'r-', linewidth=0.6, label='encoder init error')
        ax.plot(t, err_ana, 'b-', linewidth=0.6, label='analytical error')
        ax.set_ylabel(f'{STATE_NAMES[i]} error')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time [s]')
    fig.suptitle('Step 0: Error comparison (no training)', y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def plot_nrms_bar(nrms_enc, nrms_ana, title, out_path):
    """Bar chart: NRMS per channel, encoder vs analytical side by side."""
    x = np.arange(NX_PHYS)
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, nrms_enc, w, label='encoder init', color='tab:red', alpha=0.8)
    ax.bar(x + w/2, nrms_ana, w, label='analytical', color='tab:blue', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(STATE_NAMES)
    ax.set_ylabel('NRMS')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# =============================================================================
# Main
# =============================================================================

def main():
    print('=' * 70)
    print('Step 0: Encoder init diagnostic (no training)')
    print('=' * 70)

    # --- Load data ---
    print(f'\nLoading data from: {TRAJ_DIR}')
    train_data = [load_mat(f) for f in TRAIN_FILES]
    val_u, val_y, val_x_logical = load_mat(VAL_FILE)

    for i, (fname, (u, y, x)) in enumerate(zip(TRAIN_FILES, train_data)):
        print(f'  T{i+1} ({fname}): u={u.shape}, y={y.shape}, x_logical={x.shape}')
    print(f'  Val ({VAL_FILE}): u={val_u.shape}, y={val_y.shape}, x_logical={val_x_logical.shape}')

    # --- Normalization ---
    norm = compute_normalization(train_data)
    print(f'\nNormalization:')
    print(f'  std_x = {norm["std_x"].flatten()}')
    print(f'  std_u = {norm["std_u"].flatten()}')
    print(f'  ystd  = {norm["ystd"]}')

    # --- Build encoder ---
    print('\nBuilding encoder...')
    encoder = build_encoder(norm)
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f'Encoder parameters: {n_params}')

    # --- Create windowed data for validation trajectory ---
    u_hist, y_hist, x_target = create_encoder_windows(val_u, val_y, val_x_logical, norm)
    print(f'\nValidation windows: {len(u_hist)}')

    # --- Encoder init: forward pass ---
    encoder.eval()
    with torch.no_grad():
        x_hat_enc = encoder(
            torch.tensor(u_hist, dtype=DTYPE_PT),
            torch.tensor(y_hist, dtype=DTYPE_PT),
        ).numpy()

    # --- Analytical baseline ---
    x_hat_ana = compute_analytical_baseline(val_y, norm)

    # --- Compute NRMS ---
    nrms_enc = compute_nrms(x_hat_enc, x_target)
    nrms_ana = compute_nrms(x_hat_ana, x_target)

    # De-normalize for physical-unit comparison
    x_enc_phys = x_hat_enc * norm['std_x'].flatten() + norm['x_mean'].flatten()
    x_ana_phys = x_hat_ana * norm['std_x'].flatten() + norm['x_mean'].flatten()
    x_gt_phys = x_target * norm['std_x'].flatten() + norm['x_mean'].flatten()
    rms_enc_phys = compute_rms_error(x_enc_phys, x_gt_phys)
    rms_ana_phys = compute_rms_error(x_ana_phys, x_gt_phys)

    print('\n--- Per-channel NRMS (normalized states) ---')
    print(f'  {"State":<6s}  {"Encoder init":>14s}  {"Analytical":>14s}')
    print(f'  {"-"*6}  {"-"*14}  {"-"*14}')
    for i, name in enumerate(STATE_NAMES):
        print(f'  {name:<6s}  {nrms_enc[i]:>14.4e}  {nrms_ana[i]:>14.4e}')

    print(f'\n  Max NRMS  encoder: {np.max(nrms_enc):.4e}')
    print(f'  Max NRMS  analytical: {np.max(nrms_ana):.4e}')

    print('\n--- Per-channel RMS error (physical units) ---')
    print(f'  {"State":<6s}  {"Unit":<5s}  {"Encoder init":>14s}  {"Analytical":>14s}')
    print(f'  {"-"*6}  {"-"*5}  {"-"*14}  {"-"*14}')
    for i, name in enumerate(STATE_NAMES):
        print(f'  {name:<6s}  {PHYS_UNITS[i]:<5s}  {rms_enc_phys[i]:>14.4e}  {rms_ana_phys[i]:>14.4e}')

    # --- Diagnostic verdict ---
    if np.max(nrms_enc) < 1e-2:
        print('\nINIT OK: encoder init NRMS < 1e-2, proceed to training.')
    elif np.max(nrms_enc) < 1e-1:
        print('\nINIT MARGINAL: encoder init NRMS 1e-2 to 1e-1.')
        print('May work after training, but investigate if it can be improved.')
    else:
        print('\nINIT POOR: encoder init NRMS > 1e-1.')
        print('Debug normalization, matrices, or windowing before training.')

    # --- Also evaluate on each training trajectory ---
    print('\n--- Per-trajectory encoder init NRMS (max across channels) ---')
    for i, (fname, (u, y, x)) in enumerate(zip(TRAIN_FILES, train_data)):
        uh, yh, xt = create_encoder_windows(u, y, x, norm)
        with torch.no_grad():
            xh = encoder(
                torch.tensor(uh, dtype=DTYPE_PT),
                torch.tensor(yh, dtype=DTYPE_PT),
            ).numpy()
        nrms_traj = compute_nrms(xh, xt)
        print(f'  T{i+1} ({fname}): max NRMS = {np.max(nrms_traj):.4e}')

    # --- Save results ---
    results = dict(
        nrms_encoder_init={name: float(nrms_enc[i]) for i, name in enumerate(STATE_NAMES)},
        nrms_analytical={name: float(nrms_ana[i]) for i, name in enumerate(STATE_NAMES)},
        rms_phys_encoder={name: float(rms_enc_phys[i]) for i, name in enumerate(STATE_NAMES)},
        rms_phys_analytical={name: float(rms_ana_phys[i]) for i, name in enumerate(STATE_NAMES)},
        n_params=n_params,
        na=na, nb=nb, na_right=na_right, nb_right=nb_right,
    )
    json_path = os.path.join(OUT_DIR, 'step0_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {json_path}')

    # Save trajectories for plot reconstruction
    npz_path = os.path.join(OUT_DIR, 'step0_data.npz')
    np.savez_compressed(npz_path,
        x_encoder_norm=x_hat_enc, x_analytical_norm=x_hat_ana, x_target_norm=x_target,
        x_encoder_phys=x_enc_phys, x_analytical_phys=x_ana_phys, x_target_phys=x_gt_phys,
        nrms_encoder=nrms_enc, nrms_analytical=nrms_ana,
        std_x=norm['std_x'], x_mean=norm['x_mean'],
        state_names=STATE_NAMES, fs=FS_NEW,
    )
    print(f'Saved: {npz_path}')

    # --- Plots ---
    plot_comparison(
        x_hat_enc, x_hat_ana, x_target, nrms_enc, nrms_ana,
        os.path.join(OUT_DIR, 'step0_comparison.png'))
    plot_error(
        x_hat_enc, x_hat_ana, x_target,
        os.path.join(OUT_DIR, 'step0_error.png'))
    plot_nrms_bar(
        nrms_enc, nrms_ana,
        'Step 0: NRMS per channel (encoder init vs analytical)',
        os.path.join(OUT_DIR, 'step0_nrms_bar.png'))


if __name__ == '__main__':
    main()
