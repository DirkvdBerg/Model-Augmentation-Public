"""
step1b_direct_regression.py
---------------------------
Encoder verification Step 1b: direct regression test (baseline = system).

Isolated encoder test: train the encoder directly to minimize
    ||encoder(u_hist, y_hist) - x_target||^2
without any state equation, rollout, or output block. This is a pure test
of whether the encoder architecture + initialization can reconstruct states
from I/O history.

Complements step1_baseline_equals_system.py (full pipeline test).

Usage:
    conda run -n GraduationProject python scripts/gantry/encoder-baseline/verification/step1b_direct_regression.py
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Project root ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

import deepSI
from model_augmentation.utils.utils import normalize_linear_ss_matrices
from model_augmentation.utils.torch_nets import LinearInitEncoderWrapper
from model_augmentation.fit_systems.pre_encoder import linear_encoder_init
from model_augmentation.systems.gantry_ss import Cd, P
from model_augmentation.systems.gantry_linearization import gantry_linearize_and_discretize
from model_augmentation.fit_systems.blocks import Gantry_State_Block

# =============================================================================
# Configuration
# =============================================================================

NX_PHYS = 6
nu = 3
ny = 3
Y_OP = None

FS_ORIG = 20000
FS_NEW = 4000
D = FS_ORIG // FS_NEW
TS_NEW = 1.0 / FS_NEW

DTYPE_NP = np.float32
DTYPE_PT = torch.float32

SEED = 42

# HEURISTIC: Jan's rule of thumb for encoder history length
na = 4 * NX_PHYS + 1  # = 25
nb = na
na_right = 1
nb_right = 1

# Training hyperparameters
HP = dict(
    NX_ANN=0,
    n_nodes_per_layer=16,
    n_hidden_layers=2,
    up_sample=2,
    lr=1e-3,
    epochs=200,
    batch_size=512,
)

# Data
MODE = 'multisine'
TRAJ_DIR = os.path.join(PROJECT_ROOT, 'data', 'gantry', 'matlab', 'multisine')

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

OUT_DIR = os.path.join(PROJECT_ROOT, 'simulations', 'gantry_subnet', 'diagnostics',
                       'encoder_validation')
os.makedirs(OUT_DIR, exist_ok=True)

STATE_NAMES = ['q1', 'q2', 'q3', 'dq1', 'dq2', 'dq3']

# =============================================================================
# Data loading
# =============================================================================

np.random.seed(SEED)
torch.manual_seed(SEED)


def _load_u(d):
    if 'u_total' in d:
        return d['u_total']
    return d['u']


def load_mat(filename):
    d = loadmat(os.path.join(TRAJ_DIR, filename), squeeze_me=True)
    u = _load_u(d)[::D].astype(DTYPE_NP)
    y = d['y'][::D].astype(DTYPE_NP)
    return u, y


# =============================================================================
# FP model simulation (same as step1)
# =============================================================================

@torch.no_grad()
def simulate_fp_model(u, norm, block):
    """Simulate Gantry_State_Block, return (x_phys, y_sim)."""
    N = u.shape[0]
    x_mean = norm['x_mean']
    std_x = norm['std_x']
    std_u = norm['std_u']
    u_mean_val = norm['u_mean']

    x0_phys = np.zeros((6, 1), dtype=DTYPE_NP)
    x0_norm = ((x0_phys - x_mean) / std_x).astype(DTYPE_NP)
    u_norm = ((u.T - u_mean_val) / std_u).T

    x_states_phys = np.zeros((N, NX_PHYS), dtype=DTYPE_NP)
    x_cur = torch.tensor(x0_norm, dtype=DTYPE_PT).unsqueeze(0)
    x_states_phys[0] = x0_phys.flatten()

    for k in range(N - 1):
        u_k = torch.tensor(u_norm[k].reshape(1, nu, 1), dtype=DTYPE_PT)
        z = torch.cat([x_cur, u_k], dim=1)
        x_next = block.nonlinear_function(z)
        x_cur = x_next
        x_norm_np = x_next.squeeze().numpy()
        x_states_phys[k + 1] = x_norm_np * std_x.flatten() + x_mean.flatten()

    # THEORY: y = Cd @ x (measurement equation)
    Cd_np = Cd.numpy()
    y_sim = (Cd_np @ x_states_phys.T).T
    return x_states_phys, y_sim


# =============================================================================
# Create windowed training data for direct encoder regression
# =============================================================================

def create_encoder_windows(u, y, x_phys, norm, na_total, nb_total):
    """Create (u_hist, y_hist, x_target) windows for encoder regression.

    Parameters
    ----------
    u, y : (N, 3) raw input/output
    x_phys : (N, 6) physical states (ground truth)
    norm : dict with normalization constants
    na_total : na + na_right (total y history length)
    nb_total : nb + nb_right (total u history length)

    Returns
    -------
    u_hist : (M, nb_total, nu) normalized u windows
    y_hist : (M, na_total, ny) normalized y windows
    x_target : (M, NX_PHYS) normalized state targets
    """
    N = u.shape[0]

    # Normalize inputs/outputs (same as deepSI norm.transform)
    u_norm = (u - norm['u_mean'].flatten()) / norm['std_u'].flatten()
    y_norm = (y - norm['y0']) / norm['ystd']

    # Normalize states
    x_norm = (x_phys - norm['x_mean'].flatten()) / norm['std_x'].flatten()

    # Window size: encoder sees [k-na, ..., k-1, k] for y (with na_right=1)
    # and [k-nb, ..., k-1, k] for u (with nb_right=1)
    # The target is x(k) at the rightmost position
    history = max(na_total, nb_total)
    M = N - history  # number of valid windows

    u_hist = np.zeros((M, nb_total, nu), dtype=DTYPE_NP)
    y_hist = np.zeros((M, na_total, ny), dtype=DTYPE_NP)
    x_target = np.zeros((M, NX_PHYS), dtype=DTYPE_NP)

    for i in range(M):
        # Window ending at time step (history + i)
        k = history + i
        u_hist[i] = u_norm[k - nb_total + 1: k + 1]  # nb_total samples ending at k
        y_hist[i] = y_norm[k - na_total + 1: k + 1]   # na_total samples ending at k
        x_target[i] = x_norm[k]

    return u_hist, y_hist, x_target


# =============================================================================
# Build encoder
# =============================================================================

def build_encoder(hp, norm):
    """Build LinearInitEncoderWrapper with linear_encoder_init."""
    Ad, Bd, Cd_dt, Dd_dt = gantry_linearize_and_discretize(dt=TS_NEW)

    sys_data_with_x = deepSI.System_data(u=norm['u_all'], y=norm['y_all'])
    sys_data_with_x.x = norm['x_all']

    Ad_bar, Bd_bar, Cd_bar, Dd_bar = normalize_linear_ss_matrices(
        Ad, Bd, Cd_dt, Dd_dt, sys_data_with_x)

    phys_encoder = linear_encoder_init(
        A=Ad_bar, B=Bd_bar, C=Cd_bar, D=Dd_bar,
        nx=NX_PHYS, nu=nu, ny=ny, na=na, nb=nb,
        n_nodes_per_layer=hp['n_nodes_per_layer'],
        n_hidden_layers=hp['n_hidden_layers'],
        flag_linear_only=False,
    )

    encoder = LinearInitEncoderWrapper(
        phys_encoder=phys_encoder,
        nx_ann=hp['NX_ANN'],
        nb=nb + nb_right, nu=nu, na=na + na_right, ny=ny,
        n_nodes_per_layer=hp['n_nodes_per_layer'],
        n_hidden_layers=hp['n_hidden_layers'],
        u_mean=norm['u_mean'], std_u=norm['std_u'],
        y0=norm['y0'], ystd=norm['ystd'],
        x_mean=norm['x_mean'], std_x=norm['std_x'],
    ).to(DTYPE_PT)

    return encoder


# =============================================================================
# Training loop
# =============================================================================

def train_encoder(encoder, u_hist_train, y_hist_train, x_target_train,
                  u_hist_val, y_hist_val, x_target_val, hp):
    """Direct regression: minimize ||encoder(u_hist, y_hist) - x_target||^2."""
    optimizer = torch.optim.Adam(encoder.parameters(), lr=hp['lr'])
    batch_size = hp['batch_size']
    N_train = len(u_hist_train)

    best_val_loss = float('inf')
    best_state = None
    train_losses = []
    val_losses = []

    # Convert validation data to tensors
    u_val_t = torch.tensor(u_hist_val, dtype=DTYPE_PT)
    y_val_t = torch.tensor(y_hist_val, dtype=DTYPE_PT)
    x_val_t = torch.tensor(x_target_val, dtype=DTYPE_PT)

    for epoch in range(hp['epochs']):
        encoder.train()
        # Shuffle training data
        perm = np.random.permutation(N_train)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, N_train, batch_size):
            idx = perm[start:start + batch_size]
            u_batch = torch.tensor(u_hist_train[idx], dtype=DTYPE_PT)
            y_batch = torch.tensor(y_hist_train[idx], dtype=DTYPE_PT)
            x_batch = torch.tensor(x_target_train[idx], dtype=DTYPE_PT)

            optimizer.zero_grad()
            x_hat = encoder(u_batch, y_batch)
            loss = nn.functional.mse_loss(x_hat, x_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / n_batches
        train_losses.append(train_loss)

        # Validation
        encoder.eval()
        with torch.no_grad():
            x_hat_val = encoder(u_val_t, y_val_t)
            val_loss = nn.functional.mse_loss(x_hat_val, x_val_t).item()
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in encoder.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'  Epoch {epoch+1:4d}/{hp["epochs"]}  '
                  f'train_loss={train_loss:.6e}  val_loss={val_loss:.6e}')

    # Restore best
    if best_state is not None:
        encoder.load_state_dict(best_state)
    print(f'  Best val loss: {best_val_loss:.6e}')

    return train_losses, val_losses


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_nrms(encoder, u_hist, y_hist, x_target):
    """Compute per-channel NRMS of encoder output vs target (both normalized)."""
    encoder.eval()
    with torch.no_grad():
        x_hat = encoder(
            torch.tensor(u_hist, dtype=DTYPE_PT),
            torch.tensor(y_hist, dtype=DTYPE_PT),
        ).numpy()

    err = x_hat - x_target
    rms_err = np.sqrt(np.mean(err**2, axis=0))
    rms_gt = np.sqrt(np.mean(x_target**2, axis=0))
    nrms = rms_err / (rms_gt + 1e-12)
    return nrms, x_hat


# =============================================================================
# Plotting
# =============================================================================

def plot_state_comparison(x_hat, x_target, nrms, title, out_path):
    """First 2000 samples: encoder output vs ground truth (normalized)."""
    T = min(2000, len(x_hat))
    t = np.arange(T) / FS_NEW

    fig, axes = plt.subplots(NX_PHYS, 1, figsize=(12, 2.5 * NX_PHYS), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, x_target[:T, i], 'k-', linewidth=0.8, label='ground truth')
        ax.plot(t, x_hat[:T, i], 'r--', linewidth=0.8, label='encoder')
        ax.set_ylabel(STATE_NAMES[i])
        ax.set_title(f'{STATE_NAMES[i]}  NRMS = {nrms[i]:.2e}')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time [s]')
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def plot_loss_curves(train_losses, val_losses, out_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(train_losses, label='train')
    ax.semilogy(val_losses, label='val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Step 1b: Encoder regression loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# =============================================================================
# Main
# =============================================================================

def main():
    print('=' * 70)
    print('Step 1b: Encoder regression test (baseline = system)')
    print('=' * 70)

    # --- Load data ---
    print(f'\nLoading data from: {TRAJ_DIR}')
    train_raw = [load_mat(f) for f in TRAIN_FILES]
    val_u_raw, _ = load_mat(VAL_FILE)

    # --- Normalization (same as step1 and training script) ---
    P_inv_T = np.linalg.inv(P.numpy().T).astype(DTYPE_NP)
    u_all_raw = np.concatenate([u for u, _ in train_raw])
    y_all_raw = np.concatenate([y for _, y in train_raw])
    x_fd_list = []
    for _, y in train_raw:
        pos = (P_inv_T @ y.T).T
        vel = np.diff(pos, axis=0) * FS_NEW
        vel = np.vstack([vel[:1], vel])
        x_fd_list.append(np.hstack([pos, vel]))
    x_fd_all = np.concatenate(x_fd_list)

    norm = dict(
        x_mean=x_fd_all.mean(axis=0).reshape(NX_PHYS, 1).astype(DTYPE_NP),
        std_x=x_fd_all.std(axis=0).reshape(NX_PHYS, 1).astype(DTYPE_NP) + 1e-8,
        std_u=u_all_raw.std(axis=0).reshape(nu, 1).astype(DTYPE_NP) + 1e-8,
        u_mean=u_all_raw.mean(axis=0).reshape(nu, 1).astype(DTYPE_NP),
        ystd=y_all_raw.std(axis=0).astype(DTYPE_NP) + 1e-8,
        y0=(Cd.numpy() @ x_fd_all.mean(axis=0)).astype(DTYPE_NP),
        P_inv_T=P_inv_T,
        u_all=u_all_raw,
        y_all=y_all_raw,
        x_all=x_fd_all,
    )

    # --- Simulate FP model ---
    block = Gantry_State_Block(
        Y_op=Y_OP, std_x=norm['std_x'], std_u=norm['std_u'],
        x_mean=norm['x_mean'], u_mean=norm['u_mean'],
        Ts=TS_NEW, up_sample=HP['up_sample'],
    ).to(DTYPE_PT)
    block.eval()

    print('\nSimulating FP model (baseline = system)...')
    train_x_list, train_y_list, train_u_list = [], [], []
    for i, (fname, (u, _)) in enumerate(zip(TRAIN_FILES, train_raw)):
        x_phys, y_sim = simulate_fp_model(u, norm, block)
        train_x_list.append(x_phys)
        train_y_list.append(y_sim)
        train_u_list.append(u)
        print(f'  T{i+1} ({fname}): {x_phys.shape[0]} samples')

    val_x_phys, val_y_sim = simulate_fp_model(val_u_raw, norm, block)
    print(f'  Val: {val_x_phys.shape[0]} samples')

    # Update norm with simulated data for encoder matrix normalization
    norm['x_all'] = np.concatenate(train_x_list)
    norm['y_all'] = np.concatenate(train_y_list)

    # --- Create windowed data ---
    na_total = na + na_right  # 26
    nb_total = nb + nb_right  # 26

    print(f'\nCreating encoder windows (na_total={na_total}, nb_total={nb_total})...')
    u_hist_trains, y_hist_trains, x_target_trains = [], [], []
    for u, y, x in zip(train_u_list, train_y_list, train_x_list):
        uh, yh, xt = create_encoder_windows(u, y, x, norm, na_total, nb_total)
        u_hist_trains.append(uh)
        y_hist_trains.append(yh)
        x_target_trains.append(xt)

    u_hist_train = np.concatenate(u_hist_trains)
    y_hist_train = np.concatenate(y_hist_trains)
    x_target_train = np.concatenate(x_target_trains)

    u_hist_val, y_hist_val, x_target_val = create_encoder_windows(
        val_u_raw, val_y_sim, val_x_phys, norm, na_total, nb_total)

    print(f'Training windows: {len(u_hist_train)}')
    print(f'Validation windows: {len(u_hist_val)}')

    # --- Build encoder ---
    print('\nBuilding encoder...')
    encoder = build_encoder(HP, norm)
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f'Encoder parameters: {n_params}')

    # --- Evaluate BEFORE training ---
    print('\n--- Encoder NRMS BEFORE training (init quality) ---')
    nrms_init, x_hat_init = evaluate_nrms(
        encoder, u_hist_val, y_hist_val, x_target_val)
    for i, name in enumerate(STATE_NAMES):
        print(f'  {name}: NRMS = {nrms_init[i]:.4e}')

    # --- Train ---
    print(f'\nTraining: {HP["epochs"]} epochs, lr={HP["lr"]}, batch_size={HP["batch_size"]}')
    train_losses, val_losses = train_encoder(
        encoder, u_hist_train, y_hist_train, x_target_train,
        u_hist_val, y_hist_val, x_target_val, HP)

    # --- Evaluate AFTER training ---
    print('\n--- Encoder NRMS AFTER training ---')
    nrms_post, x_hat_post = evaluate_nrms(
        encoder, u_hist_val, y_hist_val, x_target_val)
    for i, name in enumerate(STATE_NAMES):
        print(f'  {name}: NRMS = {nrms_post[i]:.4e}')

    # --- Pass/fail ---
    max_nrms = np.max(nrms_post)
    print(f'\nMax NRMS across channels: {max_nrms:.4e}')
    if max_nrms < 1e-2:
        print('PASS: encoder reconstructs states with NRMS < 1e-2')
    else:
        print('FAIL: encoder NRMS too high, investigate')

    # --- Save ---
    results = dict(
        nrms_init={name: float(nrms_init[i]) for i, name in enumerate(STATE_NAMES)},
        nrms_post={name: float(nrms_post[i]) for i, name in enumerate(STATE_NAMES)},
        hp=HP,
        best_val_loss=float(min(val_losses)),
    )
    json_path = os.path.join(OUT_DIR, 'step1b_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved: {json_path}')

    # --- Plot ---
    plot_state_comparison(
        x_hat_init, x_target_val, nrms_init,
        'Step 1b: Before training (init quality)',
        os.path.join(OUT_DIR, 'step1b_before_training.png'))
    plot_state_comparison(
        x_hat_post, x_target_val, nrms_post,
        'Step 1b: After training (best checkpoint)',
        os.path.join(OUT_DIR, 'step1b_after_training.png'))
    plot_loss_curves(
        train_losses, val_losses,
        os.path.join(OUT_DIR, 'step1b_loss_curves.png'))


if __name__ == '__main__':
    main()
