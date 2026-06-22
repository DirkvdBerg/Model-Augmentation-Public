"""
encoder_diagnostic.py
---------------------
Fast diagnostic for encoder I/O validation on baseline data.

Two diagnostics:
  1. Horizon sweep: NRMS vs prediction horizon (h=1..500) for init vs analytical.
     Answers: at what horizon do velocity errors become visible in the output?
  2. LR x nf grid: 10-epoch training runs at different (lr, nf) combinations.
     Answers: which (lr, nf) regime has useful gradient signal?

No full training. Runs locally in ~5-10 min.

Usage:
    conda run -n GraduationProject python scripts/gantry/encoder/encoder_diagnostic.py
"""

import os
import sys
import copy
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
# Configuration (same as encoder_io_validation.py)
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

na = 4 * NX_PHYS + 1  # = 25
nb = na
na_right = 1
nb_right = 1

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

OUT_DIR = os.path.join(PROJECT_ROOT, 'simulations', 'gantry_subnet', 'encoder', 'diagnostics')
os.makedirs(OUT_DIR, exist_ok=True)

STAGE_NAMES = ['x1', 'x2', 'Y']
STATE_NAMES = ['X', 'theta', 'Y', 'dX', 'dtheta', 'dY']

P_np = P.numpy().astype(DTYPE_NP)


# =============================================================================
# Data loading (same as encoder_io_validation.py)
# =============================================================================

def load_mat(filename):
    d = loadmat(os.path.join(TRAJ_DIR, filename), squeeze_me=True)
    u = d['u_total'][::D].astype(DTYPE_NP) if 'u_total' in d else d['u'][::D].astype(DTYPE_NP)
    y = d['y'][::D].astype(DTYPE_NP)
    x_logical = d['x_logical'][::D].astype(DTYPE_NP)
    return u, y, x_logical


def compute_velocities_from_positions(y, P_inv_T):
    pos = (P_inv_T @ y.T).T
    vel = np.zeros_like(pos)
    # THEORY: central difference (matches MATLAB gradient())
    vel[1:-1] = (pos[2:] - pos[:-2]) * (FS_NEW / 2.0)
    vel[0] = (pos[1] - pos[0]) * FS_NEW
    vel[-1] = (pos[-1] - pos[-2]) * FS_NEW
    return np.hstack([pos, vel])


def compute_normalization(train_data):
    P_inv_T = np.linalg.inv(P.numpy().T).astype(DTYPE_NP)
    u_all = np.concatenate([u for u, _, _ in train_data])
    y_all = np.concatenate([y for _, y, _ in train_data])
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
# Build encoder and state block (same as encoder_io_validation.py)
# =============================================================================

def build_encoder(norm):
    Ad, Bd, Cd_dt, Dd_dt = gantry_linearize_and_discretize(dt=TS_NEW)
    sys_data_with_x = deepSI.System_data(u=norm['u_all'], y=norm['y_all'])
    sys_data_with_x.x = norm['x_all']
    Ad_bar, Bd_bar, Cd_bar, Dd_bar = normalize_linear_ss_matrices(
        Ad, Bd, Cd_dt, Dd_dt, sys_data_with_x)

    phys_encoder = linear_encoder_init(
        A=Ad_bar, B=Bd_bar, C=Cd_bar, D=Dd_bar,
        nx=NX_PHYS, nu=nu, ny=ny, na=na, nb=nb,
        n_nodes_per_layer=16, n_hidden_layers=2, flag_linear_only=False)

    encoder = LinearInitEncoderWrapper(
        phys_encoder=phys_encoder, nx_ann=0,
        nb=nb + nb_right, nu=nu, na=na + na_right, ny=ny,
        n_nodes_per_layer=16, n_hidden_layers=2,
        u_mean=norm['u_mean'], std_u=norm['std_u'],
        y0=norm['y0'], ystd=norm['ystd'],
        x_mean=norm['x_mean'], std_x=norm['std_x'],
    ).to(DTYPE_PT)
    return encoder


def build_state_block(norm):
    return Gantry_State_Block(
        Y_op=None, std_x=norm['std_x'], std_u=norm['std_u'],
        x_mean=norm['x_mean'], u_mean=norm['u_mean'],
        Ts=TS_NEW, up_sample=1,
    ).to(DTYPE_PT)


# =============================================================================
# Analytical baseline states
# =============================================================================

def compute_analytical_states(y, norm, history):
    """Analytical state estimates at IO window time indices."""
    pos = (norm['P_inv_T'] @ y.T).T
    vel = np.zeros_like(pos)
    vel[1:] = (pos[1:] - pos[:-1]) * FS_NEW
    vel[0] = vel[1]
    x_analytical = np.hstack([pos, vel])
    x_norm = (x_analytical - norm['x_mean'].flatten()) / norm['std_x'].flatten()
    return x_norm[history:]


# =============================================================================
# Diagnostic 1: Horizon sweep
# =============================================================================

def horizon_sweep(encoder, state_block, val_u, val_y, norm, max_horizon=500):
    """Compute output NRMS at each horizon h=1..max_horizon for init and analytical."""
    na_total = na + na_right
    nb_total = nb + nb_right
    history = max(na_total, nb_total)
    N = val_u.shape[0]
    M = N - history - max_horizon

    if M < 100:
        print(f'  WARNING: only {M} samples available for horizon {max_horizon}.')
        print(f'  Reducing max_horizon...')
        max_horizon = N - history - 100
        M = 100
        print(f'  New max_horizon={max_horizon}, M={M}')

    # Build encoder windows
    u_norm = (val_u - norm['u_mean'].flatten()) / norm['std_u'].flatten()
    y_norm = (val_y - norm['y0']) / norm['ystd']

    u_hist = np.zeros((M, nb_total, nu), dtype=DTYPE_NP)
    y_hist = np.zeros((M, na_total, ny), dtype=DTYPE_NP)
    u_future = np.zeros((M, max_horizon, nu), dtype=DTYPE_NP)
    y_future = np.zeros((M, max_horizon, ny), dtype=DTYPE_NP)

    for i in range(M):
        k = history + i
        u_hist[i] = u_norm[k - nb_total + 1: k + 1]
        y_hist[i] = y_norm[k - na_total + 1: k + 1]
        u_future[i] = u_norm[k: k + max_horizon]
        y_future[i] = val_y[k + 1: k + 1 + max_horizon]

    # Analytical states at same time indices
    x_ana_full = compute_analytical_states(val_y, norm, history)
    x_ana = x_ana_full[:M]

    # Precompute tensors
    std_x_t = torch.tensor(norm['std_x'].astype(DTYPE_NP), dtype=DTYPE_PT)
    x_mean_t = torch.tensor(norm['x_mean'].astype(DTYPE_NP), dtype=DTYPE_PT)
    Cd_t = Cd.to(DTYPE_PT)
    uf_t = torch.tensor(u_future, dtype=DTYPE_PT)

    # Get encoder states (init, no training)
    encoder.eval()
    state_block.eval()
    with torch.no_grad():
        x_enc = encoder(
            torch.tensor(u_hist, dtype=DTYPE_PT),
            torch.tensor(y_hist, dtype=DTYPE_PT),
        )  # (M, 6)

    # Roll both forward, collect NRMS at each horizon
    nrms_enc = np.zeros((max_horizon, 3))
    nrms_ana = np.zeros((max_horizon, 3))

    print(f'  Rolling forward {max_horizon} steps for {M} samples...')
    with torch.no_grad():
        x_roll_enc = x_enc.unsqueeze(-1)  # (M, 6, 1)
        x_roll_ana = torch.tensor(x_ana, dtype=DTYPE_PT).unsqueeze(-1)

        for h in range(max_horizon):
            u_step = uf_t[:, h, :].unsqueeze(-1)

            # Encoder rollout
            z_enc = torch.cat([x_roll_enc, u_step], dim=1)
            x_roll_enc = state_block.nonlinear_function(z_enc)
            x_phys_enc = x_roll_enc * std_x_t + x_mean_t
            y_hat_enc = (Cd_t @ x_phys_enc).squeeze(-1).numpy()

            # Analytical rollout
            z_ana = torch.cat([x_roll_ana, u_step], dim=1)
            x_roll_ana = state_block.nonlinear_function(z_ana)
            x_phys_ana = x_roll_ana * std_x_t + x_mean_t
            y_hat_ana = (Cd_t @ x_phys_ana).squeeze(-1).numpy()

            # Target
            y_tgt = y_future[:, h, :]

            # Per-channel NRMS
            for ch in range(3):
                rms_gt = np.sqrt(np.mean(y_tgt[:, ch]**2))
                nrms_enc[h, ch] = np.sqrt(np.mean((y_hat_enc[:, ch] - y_tgt[:, ch])**2)) / (rms_gt + 1e-12)
                nrms_ana[h, ch] = np.sqrt(np.mean((y_hat_ana[:, ch] - y_tgt[:, ch])**2)) / (rms_gt + 1e-12)

            if (h + 1) % 100 == 0:
                print(f'    h={h+1}: enc max NRMS={np.max(nrms_enc[h]):.4e}, '
                      f'ana max NRMS={np.max(nrms_ana[h]):.4e}')

    return nrms_enc, nrms_ana, max_horizon


# =============================================================================
# Diagnostic 2: LR x nf grid
# =============================================================================

def lr_nf_grid(encoder_init, state_block, val_u, val_y, norm,
               train_data, n_epochs=10):
    """Short training runs at different (lr, nf) combinations."""
    lr_values = [1e-3, 1e-4, 1e-5, 1e-6]
    nf_values = [10, 50, 100]
    batch_size = 256

    na_total = na + na_right
    nb_total = nb + nb_right
    history = max(na_total, nb_total)

    std_x_t = torch.tensor(norm['std_x'].astype(DTYPE_NP), dtype=DTYPE_PT)
    x_mean_t = torch.tensor(norm['x_mean'].astype(DTYPE_NP), dtype=DTYPE_PT)
    Cd_t = Cd.to(DTYPE_PT)
    ystd_t = torch.tensor(norm['ystd'], dtype=DTYPE_PT)

    # Precompute init baseline NRMS at nf=10 for reference
    # (we'll compute per-nf init NRMS inside the loop)

    results = {}

    for nf in nf_values:
        # Build windows for this nf
        train_windows = []
        for u, y, _ in train_data:
            N = u.shape[0]
            M = N - history - nf
            if M < 1:
                continue
            u_norm = (u - norm['u_mean'].flatten()) / norm['std_u'].flatten()
            y_norm = (y - norm['y0']) / norm['ystd']
            uh = np.zeros((M, nb_total, nu), dtype=DTYPE_NP)
            yh = np.zeros((M, na_total, ny), dtype=DTYPE_NP)
            uf = np.zeros((M, nf, nu), dtype=DTYPE_NP)
            yf = np.zeros((M, nf, ny), dtype=DTYPE_NP)
            for i in range(M):
                k = history + i
                uh[i] = u_norm[k - nb_total + 1: k + 1]
                yh[i] = y_norm[k - na_total + 1: k + 1]
                uf[i] = u_norm[k: k + nf]
                yf[i] = y[k + 1: k + 1 + nf]
            train_windows.append((uh, yh, uf, yf))

        u_tr = np.concatenate([w[0] for w in train_windows])
        y_tr = np.concatenate([w[1] for w in train_windows])
        uf_tr = np.concatenate([w[2] for w in train_windows])
        yf_tr = np.concatenate([w[3] for w in train_windows])

        # Val windows
        M_val = val_u.shape[0] - history - nf
        u_norm_v = (val_u - norm['u_mean'].flatten()) / norm['std_u'].flatten()
        y_norm_v = (val_y - norm['y0']) / norm['ystd']
        uh_v = np.zeros((M_val, nb_total, nu), dtype=DTYPE_NP)
        yh_v = np.zeros((M_val, na_total, ny), dtype=DTYPE_NP)
        uf_v = np.zeros((M_val, nf, nu), dtype=DTYPE_NP)
        yf_v = np.zeros((M_val, nf, ny), dtype=DTYPE_NP)
        for i in range(M_val):
            k = history + i
            uh_v[i] = u_norm_v[k - nb_total + 1: k + 1]
            yh_v[i] = y_norm_v[k - na_total + 1: k + 1]
            uf_v[i] = u_norm_v[k: k + nf]
            yf_v[i] = val_y[k + 1: k + 1 + nf]

        uh_v_t = torch.tensor(uh_v, dtype=DTYPE_PT)
        yh_v_t = torch.tensor(yh_v, dtype=DTYPE_PT)
        uf_v_t = torch.tensor(uf_v, dtype=DTYPE_PT)
        yf_v_t = torch.tensor(yf_v, dtype=DTYPE_PT)

        # Init NRMS at this nf
        encoder_init.eval()
        state_block.eval()
        with torch.no_grad():
            init_nrms = _eval_nstep_nrms(
                encoder_init, state_block, uh_v_t, yh_v_t, uf_v_t, yf_v,
                std_x_t, x_mean_t, Cd_t, nf)

        for lr in lr_values:
            # Fresh copy of encoder for each run
            encoder = copy.deepcopy(encoder_init)
            optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)

            u_tr_t = torch.tensor(u_tr, dtype=DTYPE_PT)
            y_tr_t = torch.tensor(y_tr, dtype=DTYPE_PT)
            uf_tr_t = torch.tensor(uf_tr, dtype=DTYPE_PT)
            yf_tr_t = torch.tensor(yf_tr, dtype=DTYPE_PT)

            N_train = len(u_tr_t)
            t0 = time.time()

            for epoch in range(n_epochs):
                encoder.train()
                perm = torch.randperm(N_train)
                for start in range(0, N_train, batch_size):
                    idx = perm[start: start + batch_size]

                    x = encoder(u_tr_t[idx], y_tr_t[idx]).unsqueeze(-1)
                    y_hats = []
                    for step in range(nf):
                        u_step = uf_tr_t[idx, step, :].unsqueeze(-1)
                        z = torch.cat([x, u_step], dim=1)
                        x = state_block.nonlinear_function(z)
                        x_phys = x * std_x_t + x_mean_t
                        y_hat = (Cd_t @ x_phys).squeeze(-1)
                        y_hats.append(y_hat)
                    y_hat_steps = torch.stack(y_hats, dim=1)
                    err_norm = (y_hat_steps - yf_tr_t[idx]) / ystd_t
                    loss = torch.mean(err_norm ** 2)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            elapsed = time.time() - t0

            # Evaluate after training
            encoder.eval()
            with torch.no_grad():
                after_nrms = _eval_nstep_nrms(
                    encoder, state_block, uh_v_t, yh_v_t, uf_v_t, yf_v,
                    std_x_t, x_mean_t, Cd_t, nf)

            # Improvement ratio: <1 means training helped, >1 means it hurt
            ratio = np.max(after_nrms) / (np.max(init_nrms) + 1e-15)

            results[(lr, nf)] = dict(
                init_nrms=init_nrms.copy(),
                after_nrms=after_nrms.copy(),
                ratio=ratio,
                elapsed=elapsed,
            )

            status = 'BETTER' if ratio < 0.95 else ('WORSE' if ratio > 1.05 else 'SAME')
            print(f'  nf={nf:3d}  lr={lr:.0e}  '
                  f'init={np.max(init_nrms):.2e}  after={np.max(after_nrms):.2e}  '
                  f'ratio={ratio:.3f}  [{elapsed:.0f}s]  {status}')

    return results, lr_values, nf_values


def _eval_nstep_nrms(encoder, state_block, uh_v_t, yh_v_t, uf_v_t, yf_v_np,
                     std_x_t, x_mean_t, Cd_t, nf):
    """Evaluate nf-step output NRMS (last step). Returns (3,) array."""
    x = encoder(uh_v_t, yh_v_t).unsqueeze(-1)
    for step in range(nf):
        u_step = uf_v_t[:, step, :].unsqueeze(-1)
        z = torch.cat([x, u_step], dim=1)
        x = state_block.nonlinear_function(z)
    x_phys = x * std_x_t + x_mean_t
    y_hat = (Cd_t @ x_phys).squeeze(-1).numpy()
    y_tgt = yf_v_np[:, -1, :]

    nrms = np.zeros(3)
    for ch in range(3):
        rms_gt = np.sqrt(np.mean(y_tgt[:, ch]**2))
        nrms[ch] = np.sqrt(np.mean((y_hat[:, ch] - y_tgt[:, ch])**2)) / (rms_gt + 1e-12)
    return nrms


# =============================================================================
# Plotting
# =============================================================================

def plot_horizon_sweep(nrms_enc, nrms_ana, max_horizon, out_path):
    horizons = np.arange(1, max_horizon + 1)
    time_ms = horizons / FS_NEW * 1000  # convert to ms

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for i, ax in enumerate(axes):
        ax.semilogy(time_ms, nrms_enc[:, i], 'r-', linewidth=0.8,
                    label='model-based init')
        ax.semilogy(time_ms, nrms_ana[:, i], 'b-', linewidth=0.8,
                    label='analytical (P_inv + bwd diff)')
        ax.set_ylabel(f'{STAGE_NAMES[i]} NRMS')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

        # Mark key horizons
        for nf_mark in [10, 20, 50, 100, 200]:
            t_mark = nf_mark / FS_NEW * 1000
            if t_mark <= time_ms[-1]:
                ax.axvline(t_mark, color='gray', alpha=0.3, linestyle='--', linewidth=0.5)
                if i == 0:
                    ax.text(t_mark, ax.get_ylim()[1], f'nf={nf_mark}',
                            fontsize=6, ha='center', va='bottom', color='gray')

    axes[-1].set_xlabel('Prediction horizon [ms]')
    fig.suptitle('Horizon sweep: at what horizon do velocity errors appear?', y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def plot_lr_nf_grid(results, lr_values, nf_values, out_path):
    # Build ratio matrix for heatmap
    n_lr = len(lr_values)
    n_nf = len(nf_values)
    ratio_matrix = np.zeros((n_lr, n_nf))
    for i, lr in enumerate(lr_values):
        for j, nf in enumerate(nf_values):
            ratio_matrix[i, j] = results[(lr, nf)]['ratio']

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(ratio_matrix, cmap='RdYlGn_r', aspect='auto',
                   vmin=0.5, vmax=2.0)
    ax.set_xticks(range(n_nf))
    ax.set_xticklabels([str(nf) for nf in nf_values])
    ax.set_yticks(range(n_lr))
    ax.set_yticklabels([f'{lr:.0e}' for lr in lr_values])
    ax.set_xlabel('nf (prediction steps)')
    ax.set_ylabel('Learning rate')

    # Annotate cells
    for i in range(n_lr):
        for j in range(n_nf):
            r = ratio_matrix[i, j]
            color = 'white' if abs(r - 1.0) > 0.3 else 'black'
            ax.text(j, i, f'{r:.2f}', ha='center', va='center',
                    fontsize=10, fontweight='bold', color=color)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('NRMS ratio (after/init). <1 = improved, >1 = degraded')
    ax.set_title('LR x nf grid: 10-epoch training probe (ratio after/init)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# =============================================================================
# Main
# =============================================================================

def main():
    print('=' * 70)
    print('Encoder diagnostic: horizon sweep + LR x nf grid')
    print('=' * 70)

    # --- Load data ---
    print('\nLoading data...')
    train_data = [load_mat(f) for f in TRAIN_FILES]
    val_u, val_y, _ = load_mat(VAL_FILE)
    print(f'  {len(TRAIN_FILES)} train trajectories, 1 val trajectory')
    print(f'  Val: {val_u.shape[0]} samples at {FS_NEW} Hz = {val_u.shape[0]/FS_NEW:.1f}s')

    norm = compute_normalization(train_data)

    # --- Build encoder and state block ---
    print('\nBuilding encoder (model-based init)...')
    encoder = build_encoder(norm)
    state_block = build_state_block(norm)
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f'  Parameters: {n_params}')

    # =================================================================
    # Diagnostic 1: Horizon sweep
    # =================================================================
    print('\n' + '=' * 70)
    print('DIAGNOSTIC 1: Horizon sweep (init vs analytical)')
    print('=' * 70)
    max_h = 500
    available = val_u.shape[0] - max(na + na_right, nb + nb_right) - 100
    if max_h > available:
        max_h = available
        print(f'  Capped max_horizon to {max_h} (limited by val trajectory length)')

    t0 = time.time()
    nrms_enc, nrms_ana, max_h = horizon_sweep(
        encoder, state_block, val_u, val_y, norm, max_horizon=max_h)
    print(f'  Completed in {time.time()-t0:.0f}s')

    # Print key horizons
    print(f'\n  Summary (max channel NRMS):')
    print(f'  {"Horizon":>8s}  {"Time [ms]":>10s}  {"Init":>12s}  {"Analytical":>12s}  {"Ratio":>8s}')
    for h in [1, 10, 20, 50, 100, 200, 500]:
        if h <= max_h:
            idx = h - 1
            enc_max = np.max(nrms_enc[idx])
            ana_max = np.max(nrms_ana[idx])
            ratio = enc_max / (ana_max + 1e-15)
            t_ms = h / FS_NEW * 1000
            print(f'  {h:>8d}  {t_ms:>10.2f}  {enc_max:>12.4e}  {ana_max:>12.4e}  {ratio:>8.2f}x')

    plot_horizon_sweep(nrms_enc, nrms_ana, max_h,
                       os.path.join(OUT_DIR, 'diagnostic_horizon_sweep.png'))

    # =================================================================
    # Diagnostic 2: LR x nf grid
    # =================================================================
    print('\n' + '=' * 70)
    print('DIAGNOSTIC 2: LR x nf grid (10 epochs each)')
    print('=' * 70)

    t0 = time.time()
    grid_results, lr_values, nf_values = lr_nf_grid(
        encoder, state_block, val_u, val_y, norm, train_data, n_epochs=10)
    print(f'\n  Grid completed in {time.time()-t0:.0f}s')

    plot_lr_nf_grid(grid_results, lr_values, nf_values,
                    os.path.join(OUT_DIR, 'diagnostic_lr_nf_grid.png'))

    # --- Save results ---
    import json
    save_results = {
        'horizon_sweep': {
            'max_horizon': int(max_h),
            'fs': int(FS_NEW),
        },
        'lr_nf_grid': {
            f'lr={lr:.0e}_nf={nf}': {
                'ratio': float(grid_results[(lr, nf)]['ratio']),
                'init_max_nrms': float(np.max(grid_results[(lr, nf)]['init_nrms'])),
                'after_max_nrms': float(np.max(grid_results[(lr, nf)]['after_nrms'])),
                'elapsed_s': float(grid_results[(lr, nf)]['elapsed']),
            }
            for lr in lr_values for nf in nf_values
        },
    }
    json_path = os.path.join(OUT_DIR, 'diagnostic_results.json')
    with open(json_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f'Saved: {json_path}')

    # Save horizon sweep data
    np.savez_compressed(os.path.join(OUT_DIR, 'diagnostic_horizon_data.npz'),
        nrms_enc=nrms_enc, nrms_ana=nrms_ana,
        max_horizon=max_h, fs=FS_NEW)
    print(f'Saved: {os.path.join(OUT_DIR, "diagnostic_horizon_data.npz")}')

    print('\n' + '=' * 70)
    print('Diagnostic complete.')


if __name__ == '__main__':
    main()
