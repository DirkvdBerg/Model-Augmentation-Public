"""
encoder_io_comparison.py
------------------------
Compare 6-state (NX_ANN=0) vs 6+2-state (NX_ANN=2) encoder using
multi-step I/O prediction loss on MSD data.

Both models use the same architecture:
  - Encoder: LinearInitEncoderWrapper (model-based Wb init)
  - State model: Gantry_State_Block + Static_ANN_Block (via Interconnect)
  - Loss: nf-step I/O prediction ||y_hat - y_measured||^2

No deepSI pipeline, no SSE_Interconnect, no truncation/segmentation.
Just a simple PyTorch training loop with the Interconnect for state stepping.

THEORY: Hoekstra 2026 Eq. 8 (encoder ResNet), Eq. 16-17 (Wb init from
reconstructability map), Eq. 35 (encoder pre-training concept).

Usage:
    conda run -n GraduationProject python scripts/gantry/encoder/encoder_io_comparison.py
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
from model_augmentation.utils.utils import (
    normalize_linear_ss_matrices, selection_matrix, expansion_matrix,
)
from model_augmentation.utils.torch_nets import (
    LinearInitEncoderWrapper, zero_init_feed_forward_nn,
)
from model_augmentation.fit_systems.pre_encoder import linear_encoder_init
from model_augmentation.fit_systems.blocks import (
    Gantry_State_Block, Static_ANN_Block, Linear_Output_Block,
)
from model_augmentation.fit_systems.interconnect import Interconnect
from model_augmentation.systems.gantry_ss import Cd, Dd, P
from model_augmentation.systems.gantry_linearization import (
    gantry_linearize_and_discretize,
)

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
    lr=1e-4,            # HEURISTIC: lower than standalone (gradients through RK4 are larger)
    epochs=200,
    batch_size=256,
    nf=20,              # HEURISTIC: 20 steps = 5 ms at 4 kHz, ~1 MSD period
    up_sample=2,        # HEURISTIC: RK4 sub-steps per sample
)

# MSD data
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

OUT_DIR = os.path.join(PROJECT_ROOT, 'simulations', 'gantry_subnet', 'encoder')
os.makedirs(OUT_DIR, exist_ok=True)

STATE_NAMES = ['X', 'theta', 'Y', 'dX', 'dtheta', 'dY']
PHYS_UNITS = ['m', 'rad', 'm', 'm/s', 'rad/s', 'm/s']
STAGE_NAMES = ['x1', 'x2', 'Y', 'dx1', 'dx2', 'dY']
STAGE_UNITS = ['m', 'm', 'm', 'm/s', 'm/s', 'm/s']
AUG_NAMES = ['aug1', 'aug2']

P_np = P.numpy().astype(DTYPE_NP)
PT = P_np.T


# =============================================================================
# Coordinate transforms (reuse from encoder_baseline_standalone.py)
# =============================================================================

def logical_to_stage(x_logical):
    """Convert (N, 6) logical states to (N, 6) stage states via P^T."""
    pos_stage = (PT @ x_logical[:, :3].T).T
    vel_stage = (PT @ x_logical[:, 3:].T).T
    return np.hstack([pos_stage, vel_stage])


# =============================================================================
# Data loading (reuse)
# =============================================================================

def load_mat(filename):
    """Load u, y, x_logical, delta_a from .mat file, downsample to FS_NEW."""
    d = loadmat(os.path.join(TRAJ_DIR, filename), squeeze_me=True)
    u = d['u_total'][::D].astype(DTYPE_NP) if 'u_total' in d else d['u'][::D].astype(DTYPE_NP)
    y = d['y'][::D].astype(DTYPE_NP)
    x_logical = d['x_logical'][::D].astype(DTYPE_NP)
    delta_a = d['delta_a'][::D].astype(DTYPE_NP) if 'delta_a' in d else None
    return u, y, x_logical, delta_a


# =============================================================================
# Velocity computation (reuse)
# =============================================================================

def compute_velocities_from_positions(y, P_inv_T):
    """Compute states from measurements: P_inv @ y for positions,
    central finite-diff for velocities."""
    pos = (P_inv_T @ y.T).T
    vel = np.zeros_like(pos)
    # THEORY: central difference (matches MATLAB gradient())
    vel[1:-1] = (pos[2:] - pos[:-2]) * (FS_NEW / 2.0)
    vel[0] = (pos[1] - pos[0]) * FS_NEW
    vel[-1] = (pos[-1] - pos[-2]) * FS_NEW
    return np.hstack([pos, vel])


# =============================================================================
# Normalization (reuse)
# =============================================================================

def compute_normalization(train_data):
    """Compute normalization constants from training data."""
    P_inv_T = np.linalg.inv(P.numpy().T).astype(DTYPE_NP)

    u_all = np.concatenate([u for u, _, _, _ in train_data])
    y_all = np.concatenate([y for _, y, _, _ in train_data])

    x_computed_list = []
    for _, y_traj, _, _ in train_data:
        x_computed_list.append(compute_velocities_from_positions(y_traj, P_inv_T))
    x_all = np.concatenate(x_computed_list)

    x_mean = x_all.mean(axis=0).reshape(NX_PHYS, 1).astype(DTYPE_NP)
    std_x = x_all.std(axis=0).reshape(NX_PHYS, 1).astype(DTYPE_NP) + 1e-8
    std_u = u_all.std(axis=0).reshape(nu, 1).astype(DTYPE_NP) + 1e-8
    u_mean = u_all.mean(axis=0).reshape(nu, 1).astype(DTYPE_NP)
    ystd = y_all.std(axis=0).astype(DTYPE_NP) + 1e-8
    y0 = (Cd.numpy() @ x_mean.flatten()).astype(DTYPE_NP)

    # Normalized output matrix: maps normalized x to normalized y
    Cd_norm = Cd.numpy() * std_x.flatten()[None, :] / ystd[:, None]
    Dd_np = Dd.numpy()

    return dict(
        x_mean=x_mean, std_x=std_x, std_u=std_u, u_mean=u_mean,
        ystd=ystd, y0=y0, P_inv_T=P_inv_T,
        u_all=u_all, y_all=y_all, x_all=x_all,
        Cd_norm=Cd_norm, Dd_np=Dd_np,
    )


# =============================================================================
# Analytical baseline (reuse)
# =============================================================================

def compute_analytical_baseline(y, norm):
    """Analytical state estimates (6 physical states only)."""
    na_total = na + na_right
    nb_total = nb + nb_right
    history = max(na_total, nb_total)
    N = y.shape[0]
    nf = HP['nf']
    M = N - history - nf

    pos = (norm['P_inv_T'] @ y.T).T
    vel = np.zeros_like(pos)
    vel[1:] = (pos[1:] - pos[:-1]) * FS_NEW
    vel[0] = vel[1]
    x_analytical = np.hstack([pos, vel])

    x_analytical_norm = (x_analytical - norm['x_mean'].flatten()) / norm['std_x'].flatten()
    return x_analytical_norm[history: history + M]


# =============================================================================
# Build encoder
# =============================================================================

def build_encoder(norm, nx_ann):
    """Build LinearInitEncoderWrapper with given nx_ann."""
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
        nx_ann=nx_ann,
        nb=nb + nb_right, nu=nu, na=na + na_right, ny=ny,
        n_nodes_per_layer=HP['n_nodes_per_layer'],
        n_hidden_layers=HP['n_hidden_layers'],
        u_mean=norm['u_mean'], std_u=norm['std_u'],
        y0=norm['y0'], ystd=norm['ystd'],
        x_mean=norm['x_mean'], std_x=norm['std_x'],
    ).to(DTYPE_PT)

    return encoder


# =============================================================================
# Build interconnect (state model)
# =============================================================================

def build_interconnect(norm, nx_ann):
    """Build Interconnect with Gantry_State_Block + Static_ANN_Block.
    Mirrors gantry_interconnect_dynamic.py build_model()."""
    nxd = NX_PHYS + nx_ann
    PHY_IX = np.arange(NX_PHYS)

    ic = Interconnect(nxd, nu, ny, debugging=False)

    phy_block = Gantry_State_Block(
        Y_op=None,
        std_x=norm['std_x'], std_u=norm['std_u'],
        x_mean=norm['x_mean'], u_mean=norm['u_mean'],
        Ts=TS_NEW, up_sample=HP['up_sample'],
    ).to(DTYPE_PT)

    out_block = Linear_Output_Block(C=norm['Cd_norm'], D=norm['Dd_np'])

    ann_block = Static_ANN_Block(
        nz=nxd + nu, nw=nxd,
        n_nodes_per_layer=HP['n_nodes_per_layer'],
        n_hidden_layers=HP['n_hidden_layers'],
        net=zero_init_feed_forward_nn,
        activation=nn.Tanh,
    )

    ic.add_block(phy_block)
    ic.add_block(out_block)
    ic.add_block(ann_block)

    ic.connect_block_signals(ann_block, ["x", "u"], ["xp"])
    ic.connect_signals("x", phy_block, "concat", selection_matrix(PHY_IX, nxd))
    ic.connect_block_signals(phy_block, ["u"], [])
    ic.connect_signals(phy_block, "xp", "additive", expansion_matrix(PHY_IX, nxd))
    ic.connect_signals("x", out_block, "concat", selection_matrix(PHY_IX, nxd))
    ic.connect_block_signals(out_block, ["u"], ["y"])

    return ic


# =============================================================================
# Create I/O windows
# =============================================================================

def create_io_windows(u, y, norm, nf):
    """Create (u_hist, y_hist, u_future, y_future) windows.
    u_hist/y_hist: encoder input (past).
    u_future: inputs for nf rollout steps, shape (M, nf, nu).
    y_future: target outputs for nf steps, shape (M, nf, ny)."""
    na_total = na + na_right
    nb_total = nb + nb_right
    N = u.shape[0]
    history = max(na_total, nb_total)
    M = N - history - nf  # valid windows

    u_norm = (u - norm['u_mean'].flatten()) / norm['std_u'].flatten()
    y_norm = (y - norm['y0']) / norm['ystd']

    u_hist = np.zeros((M, nb_total, nu), dtype=DTYPE_NP)
    y_hist = np.zeros((M, na_total, ny), dtype=DTYPE_NP)
    u_future = np.zeros((M, nf, nu), dtype=DTYPE_NP)
    y_future = np.zeros((M, nf, ny), dtype=DTYPE_NP)

    for i in range(M):
        k = history + i
        u_hist[i] = u_norm[k - nb_total + 1: k + 1]
        y_hist[i] = y_norm[k - na_total + 1: k + 1]
        u_future[i] = u_norm[k: k + nf]
        y_future[i] = y_norm[k: k + nf]

    return u_hist, y_hist, u_future, y_future


# =============================================================================
# Rollout: encoder -> nf steps -> y_hat
# =============================================================================

def rollout(encoder, interconnect, u_hist_b, y_hist_b, u_future_b):
    """Forward pass: encoder -> nf RK4 steps via interconnect -> y_hat.

    Args:
        u_hist_b: (batch, nb_total, nu) past inputs
        y_hist_b: (batch, na_total, ny) past outputs
        u_future_b: (batch, nf, nu) future inputs

    Returns:
        y_hat: (batch, nf, ny) predicted outputs
        x0: (batch, nxd) encoder initial state (detached, for diagnostics)
    """
    x = encoder(u_hist_b, y_hist_b)  # (batch, nxd)
    x0 = x.detach().clone()

    nf = u_future_b.shape[1]
    y_preds = []
    for k in range(nf):
        u_k = u_future_b[:, k, :]           # (batch, nu)
        y_k, x = interconnect(x, u_k)       # y_k: (batch, ny), x: (batch, nxd)
        y_preds.append(y_k)

    y_hat = torch.stack(y_preds, dim=1)      # (batch, nf, ny)
    return y_hat, x0


# =============================================================================
# Training loop
# =============================================================================

def train_model(encoder, interconnect, train_windows, val_windows, hp):
    """Train encoder + interconnect with nf-step I/O prediction loss."""
    u_train, y_train, uf_train, yf_train = [
        torch.tensor(w, dtype=DTYPE_PT) for w in train_windows]
    u_val, y_val, uf_val, yf_val = [
        torch.tensor(w, dtype=DTYPE_PT) for w in val_windows]

    all_params = list(encoder.parameters()) + list(interconnect.parameters())
    optimizer = torch.optim.Adam(all_params, lr=hp['lr'])
    criterion = nn.MSELoss()

    N_train = len(u_train)
    batch_size = hp['batch_size']
    train_losses = []
    val_losses = []

    t_start = time.time()
    for epoch in range(hp['epochs']):
        encoder.train()
        interconnect.train()
        perm = torch.randperm(N_train)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, N_train, batch_size):
            idx = perm[start: start + batch_size]
            y_hat, _ = rollout(
                encoder, interconnect,
                u_train[idx], y_train[idx], uf_train[idx])
            loss = criterion(y_hat, yf_train[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / n_batches
        train_losses.append(avg_train)

        # Validation
        encoder.eval()
        interconnect.eval()
        with torch.no_grad():
            y_hat_val, _ = rollout(encoder, interconnect, u_val, y_val, uf_val)
            val_loss = criterion(y_hat_val, yf_val).item()
        val_losses.append(val_loss)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            elapsed = time.time() - t_start
            print(f'  Epoch {epoch+1:4d}/{hp["epochs"]}  '
                  f'train={avg_train:.4e}  val={val_loss:.4e}  '
                  f'[{elapsed:.0f}s]')

    elapsed_total = time.time() - t_start
    print(f'  Training complete in {elapsed_total:.0f}s')
    return train_losses, val_losses, elapsed_total


# =============================================================================
# Evaluation helpers (reuse)
# =============================================================================

def compute_nrms(x_hat, x_target):
    err = x_hat - x_target
    rms_err = np.sqrt(np.mean(err**2, axis=0))
    rms_gt = np.sqrt(np.mean(x_target**2, axis=0))
    return rms_err / (rms_gt + 1e-12)


def compute_rms_error(x_hat, x_target):
    return np.sqrt(np.mean((x_hat - x_target)**2, axis=0))


def denormalize_states(x_norm, norm):
    return x_norm * norm['std_x'].flatten() + norm['x_mean'].flatten()


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
# Evaluate one model
# =============================================================================

def evaluate_model(encoder, interconnect, val_windows, norm, label):
    """Evaluate encoder + interconnect on validation data.
    Returns dict with physical states, augmented states, I/O NRMS, etc."""
    u_hist, y_hist, u_future, y_future = val_windows

    encoder.eval()
    interconnect.eval()
    with torch.no_grad():
        y_hat, x0 = rollout(
            encoder, interconnect,
            torch.tensor(u_hist, dtype=DTYPE_PT),
            torch.tensor(y_hist, dtype=DTYPE_PT),
            torch.tensor(u_future, dtype=DTYPE_PT),
        )
    y_hat_np = y_hat.numpy()           # (M, nf, ny) normalized
    x0_np = x0.numpy()                 # (M, nxd) normalized

    # I/O NRMS: per output channel, averaged over nf steps
    io_err = y_hat_np - y_future
    io_rms = np.sqrt(np.mean(io_err**2, axis=(0, 1)))     # (ny,)
    io_rms_ref = np.sqrt(np.mean(y_future**2, axis=(0, 1)))
    io_nrms = io_rms / (io_rms_ref + 1e-12)

    # Physical state NRMS (encoder output vs ground truth)
    x_phys_norm = x0_np[:, :NX_PHYS]

    # Augmented states
    nx_ann = x0_np.shape[1] - NX_PHYS
    x_aug = x0_np[:, NX_PHYS:] if nx_ann > 0 else None

    print(f'\n  [{label}] I/O prediction NRMS ({HP["nf"]}-step):')
    for ch, name in enumerate(STAGE_NAMES[:ny]):
        print(f'    {name}: {io_nrms[ch]:.4e}')

    return dict(
        x_phys_norm=x_phys_norm,
        x_aug=x_aug,
        y_hat=y_hat_np,
        io_nrms=io_nrms,
        io_rms=io_rms,
        nx_ann=nx_ann,
    )


# =============================================================================
# Plotting
# =============================================================================

def plot_loss_comparison(losses_0, losses_2, out_path):
    """Overlay train/val loss for both models."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(losses_0['train'], 'C0-', linewidth=0.8, alpha=0.5, label='NX_ANN=0 train')
    ax.semilogy(losses_0['val'], 'C0-', linewidth=1.2, label='NX_ANN=0 val')
    ax.semilogy(losses_2['train'], 'C1-', linewidth=0.8, alpha=0.5, label='NX_ANN=2 train')
    ax.semilogy(losses_2['val'], 'C1-', linewidth=1.2, label='NX_ANN=2 val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('I/O MSE loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f'I/O prediction loss comparison ({HP["nf"]}-step rollout)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def plot_io_prediction(y_hat_0, y_hat_2, y_target, out_path):
    """Compare I/O predictions from both models at a single window."""
    # Pick a window near the middle of the validation set
    mid = len(y_hat_0) // 2
    nf = y_hat_0.shape[1]
    t = np.arange(nf) / FS_NEW * 1000  # ms

    fig, axes = plt.subplots(ny, 1, figsize=(12, 2.5 * ny), sharex=True)
    for ch, (ax, name) in enumerate(zip(axes, STAGE_NAMES[:ny])):
        ax.plot(t, y_target[mid, :, ch], 'k-', linewidth=1.0, label='measured')
        ax.plot(t, y_hat_0[mid, :, ch], 'C0--', linewidth=0.9, label='NX_ANN=0')
        ax.plot(t, y_hat_2[mid, :, ch], 'C1--', linewidth=0.9, label='NX_ANN=2')
        ax.set_ylabel(f'{name} (norm)')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time [ms]')
    fig.suptitle(f'{nf}-step I/O prediction (single window)', y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def plot_state_comparison(x0, x2, x_ana, x_target, nrms_0, nrms_2, nrms_ana,
                          names, units, title, out_path):
    """Time-domain state comparison: NX_ANN=0 vs NX_ANN=2 vs analytical."""
    n_states = len(names)
    T = min(2000, len(x0))
    t = np.arange(T) / FS_NEW

    fig, axes = plt.subplots(n_states, 1, figsize=(14, 2.5 * n_states), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, x_target[:T, i], 'k-', linewidth=0.8, label='target')
        ax.plot(t, x0[:T, i], 'C0--', linewidth=0.8,
                label=f'NX_ANN=0 (NRMS={nrms_0[i]:.2e})')
        ax.plot(t, x2[:T, i], 'C1--', linewidth=0.8,
                label=f'NX_ANN=2 (NRMS={nrms_2[i]:.2e})')
        ax.plot(t, x_ana[:T, i], 'C2:', linewidth=0.8,
                label=f'analytical (NRMS={nrms_ana[i]:.2e})')
        ax.set_ylabel(f'{names[i]} [{units[i]}]')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time [s]')
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def plot_nrms_bar(nrms_0, nrms_2, nrms_ana, names, out_path):
    """Bar chart: NRMS per channel, 3 models."""
    n = len(names)
    x = np.arange(n)
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w, nrms_0, w, label='NX_ANN=0', color='C0', alpha=0.8)
    ax.bar(x, nrms_2, w, label='NX_ANN=2', color='C1', alpha=0.8)
    ax.bar(x + w, nrms_ana, w, label='analytical', color='C2', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel('NRMS')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title('Encoder I/O comparison: physical state NRMS')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def plot_augmented_vs_delta_a(x_aug, delta_a, ddelta_a, out_path):
    """Augmented states vs delta_a and ddelta_a (NX_ANN=2 only)."""
    T = min(2000, len(x_aug))
    t = np.arange(T) / FS_NEW

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax = axes[0]
    ax.plot(t, delta_a[:T], 'k-', linewidth=0.8, label='delta_a (MATLAB)')
    ax2 = ax.twinx()
    ax2.plot(t, x_aug[:T, 0], 'r--', linewidth=0.8, label='aug1 (encoder)')
    ax.set_ylabel('delta_a [m]')
    ax2.set_ylabel('aug1 (normalized)')
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title('Augmented state 1 vs MSD displacement')

    ax = axes[1]
    ax.plot(t, ddelta_a[:T], 'k-', linewidth=0.8, label='ddelta_a (finite-diff)')
    ax2 = ax.twinx()
    ax2.plot(t, x_aug[:T, 1], 'r--', linewidth=0.8, label='aug2 (encoder)')
    ax.set_ylabel('ddelta_a [m/s]')
    ax2.set_ylabel('aug2 (normalized)')
    ax.legend(loc='upper left', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title('Augmented state 2 vs MSD velocity')

    axes[-1].set_xlabel('Time [s]')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


# =============================================================================
# Main
# =============================================================================

def main():
    print('=' * 70)
    print('Encoder I/O comparison: NX_ANN=0 vs NX_ANN=2 on MSD data')
    print('=' * 70)
    print(f'nf={HP["nf"]} steps ({HP["nf"]/FS_NEW*1000:.1f} ms), '
          f'lr={HP["lr"]}, epochs={HP["epochs"]}, batch={HP["batch_size"]}')

    # --- Load data ---
    print(f'\nLoading MSD data from: {TRAJ_DIR}')
    train_data = [load_mat(f) for f in TRAIN_FILES]
    val_u, val_y, val_x_logical, val_delta_a = load_mat(VAL_FILE)

    for i, (fname, (u, y, x, da)) in enumerate(zip(TRAIN_FILES, train_data)):
        msd_tag = '(has delta_a)' if da is not None else '(NO delta_a)'
        print(f'  T{i+1} ({fname}): u={u.shape}, y={y.shape} {msd_tag}')
    val_msd_tag = '(has delta_a)' if val_delta_a is not None else '(NO delta_a)'
    print(f'  Val ({VAL_FILE}): u={val_u.shape}, y={val_y.shape} {val_msd_tag}')

    # --- Normalization ---
    norm = compute_normalization(train_data)
    print(f'\nNormalization:')
    print(f'  std_x = {norm["std_x"].flatten()}')
    print(f'  std_u = {norm["std_u"].flatten()}')
    print(f'  ystd  = {norm["ystd"]}')

    # =================================================================
    # CHECK 1: Velocity verification (Python vs MATLAB)
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
    if np.max(vel_nrms) < 0.01:
        print('  OK: Python velocities match MATLAB within 1%.')
    else:
        print('  WARNING: Python velocities differ from MATLAB significantly.')

    # --- Compute targets ---
    x_target_all = []
    for _, y_traj, _, _ in train_data:
        x_target_all.append(compute_velocities_from_positions(y_traj, norm['P_inv_T']))
    val_x_target = compute_velocities_from_positions(val_y, norm['P_inv_T'])

    # --- Create I/O windows ---
    print('\nCreating I/O windows...')
    nf = HP['nf']
    train_windows_list = []
    for (u, y, _, _), x_tgt in zip(train_data, x_target_all):
        train_windows_list.append(create_io_windows(u, y, norm, nf))

    train_windows = tuple(
        np.concatenate([w[i] for w in train_windows_list])
        for i in range(4)
    )
    val_windows = create_io_windows(val_u, val_y, norm, nf)
    print(f'  Training windows: {len(train_windows[0])}')
    print(f'  Validation windows: {len(val_windows[0])}')

    # --- Analytical baseline ---
    x_hat_ana = compute_analytical_baseline(val_y, norm)
    M = len(val_windows[0])
    x_hat_ana = x_hat_ana[:M]  # align length

    # --- Physical state target (normalized, aligned to windows) ---
    na_total = na + na_right
    nb_total = nb + nb_right
    history = max(na_total, nb_total)
    val_x_target_norm = (val_x_target - norm['x_mean'].flatten()) / norm['std_x'].flatten()
    x_target_aligned = val_x_target_norm[history: history + M]

    nrms_ana = compute_nrms(x_hat_ana, x_target_aligned)

    # =================================================================
    # Train NX_ANN=0 and NX_ANN=2
    # =================================================================
    results = {}
    losses = {}

    for nx_ann in [0, 2]:
        label = f'NX_ANN={nx_ann}'
        nxd = NX_PHYS + nx_ann
        print(f'\n{"="*70}')
        print(f'Training {label} ({nxd} states, nf={nf})')
        print(f'{"="*70}')

        encoder = build_encoder(norm, nx_ann)
        interconnect = build_interconnect(norm, nx_ann)

        n_enc = sum(p.numel() for p in encoder.parameters())
        n_ic = sum(p.numel() for p in interconnect.parameters())
        print(f'  Encoder params: {n_enc}, Interconnect params: {n_ic}, Total: {n_enc + n_ic}')

        train_losses, val_losses, elapsed = train_model(
            encoder, interconnect, train_windows, val_windows, HP)

        # Evaluate
        res = evaluate_model(encoder, interconnect, val_windows, norm, label)
        res['n_params_encoder'] = n_enc
        res['n_params_interconnect'] = n_ic
        res['train_time_s'] = elapsed
        res['nrms_phys'] = compute_nrms(res['x_phys_norm'], x_target_aligned)

        results[nx_ann] = res
        losses[nx_ann] = dict(train=train_losses, val=val_losses)

        # Save weights
        wt_path = os.path.join(OUT_DIR, f'encoder_io_nx{nx_ann}_weights.pt')
        torch.save(encoder.state_dict(), wt_path)
        ic_path = os.path.join(OUT_DIR, f'interconnect_io_nx{nx_ann}_weights.pt')
        torch.save(interconnect.state_dict(), ic_path)
        print(f'  Saved: {wt_path}')
        print(f'  Saved: {ic_path}')

    # =================================================================
    # Comparison tables
    # =================================================================
    r0 = results[0]
    r2 = results[2]

    print(f'\n{"="*70}')
    print('COMPARISON: Physical state NRMS')
    print(f'{"="*70}')
    print(f'  {"State":<8s}  {"NX_ANN=0":>14s}  {"NX_ANN=2":>14s}  {"Analytical":>14s}  {"Winner":>10s}')
    print(f'  {"-"*8}  {"-"*14}  {"-"*14}  {"-"*14}  {"-"*10}')
    for i, name in enumerate(STATE_NAMES):
        vals = [r0['nrms_phys'][i], r2['nrms_phys'][i], nrms_ana[i]]
        best = np.argmin(vals)
        winner = ['NX_ANN=0', 'NX_ANN=2', 'analytical'][best]
        print(f'  {name:<8s}  {vals[0]:>14.4e}  {vals[1]:>14.4e}  {vals[2]:>14.4e}  {winner:>10s}')

    # Physical-unit RMS error
    x0_phys = denormalize_states(r0['x_phys_norm'], norm)
    x2_phys = denormalize_states(r2['x_phys_norm'], norm)
    x_ana_phys = denormalize_states(x_hat_ana, norm)
    x_gt_phys = denormalize_states(x_target_aligned, norm)
    rms_0 = compute_rms_error(x0_phys, x_gt_phys)
    rms_2 = compute_rms_error(x2_phys, x_gt_phys)
    rms_ana = compute_rms_error(x_ana_phys, x_gt_phys)

    print(f'\n  Per-channel RMS error (physical units):')
    print(f'  {"State":<8s}  {"NX_ANN=0":>14s}  {"NX_ANN=2":>14s}  {"Analytical":>14s}  {"Unit":>6s}')
    for i, name in enumerate(STATE_NAMES):
        print(f'  {name:<8s}  {format_rms_unit(rms_0[i], PHYS_UNITS[i]):>14s}  '
              f'{format_rms_unit(rms_2[i], PHYS_UNITS[i]):>14s}  '
              f'{format_rms_unit(rms_ana[i], PHYS_UNITS[i]):>14s}  {PHYS_UNITS[i]:>6s}')

    print(f'\n{"="*70}')
    print('COMPARISON: I/O prediction NRMS')
    print(f'{"="*70}')
    print(f'  {"Output":<6s}  {"NX_ANN=0":>14s}  {"NX_ANN=2":>14s}  {"Winner":>10s}')
    print(f'  {"-"*6}  {"-"*14}  {"-"*14}  {"-"*10}')
    for ch, name in enumerate(STAGE_NAMES[:ny]):
        winner = 'NX_ANN=2' if r2['io_nrms'][ch] < r0['io_nrms'][ch] else 'NX_ANN=0'
        print(f'  {name:<6s}  {r0["io_nrms"][ch]:>14.4e}  {r2["io_nrms"][ch]:>14.4e}  {winner:>10s}')

    # =================================================================
    # CHECK 2: Augmented state correlation with delta_a (NX_ANN=2 only)
    # =================================================================
    print(f'\n--- CHECK 2: Augmented states vs delta_a (NX_ANN=2) ---')
    if val_delta_a is not None and r2['x_aug'] is not None:
        delta_a_aligned = val_delta_a[history: history + M]
        ddelta_a = np.zeros_like(delta_a_aligned)
        ddelta_a[1:-1] = (delta_a_aligned[2:] - delta_a_aligned[:-2]) * (FS_NEW / 2.0)
        ddelta_a[0] = (delta_a_aligned[1] - delta_a_aligned[0]) * FS_NEW
        ddelta_a[-1] = (delta_a_aligned[-1] - delta_a_aligned[-2]) * FS_NEW

        for j in range(r2['nx_ann']):
            corr = np.corrcoef(r2['x_aug'][:, j], delta_a_aligned)[0, 1]
            corr_vel = np.corrcoef(r2['x_aug'][:, j], ddelta_a)[0, 1]
            print(f'  {AUG_NAMES[j]}:  corr(delta_a)={corr:+.4f}  corr(ddelta_a)={corr_vel:+.4f}')
    else:
        print('  delta_a not available.')
        delta_a_aligned = None
        ddelta_a = None

    # =================================================================
    # Verdict
    # =================================================================
    io_wins = sum(1 for ch in range(ny) if r2['io_nrms'][ch] < r0['io_nrms'][ch])
    phys_wins = sum(1 for i in range(NX_PHYS) if r2['nrms_phys'][i] < r0['nrms_phys'][i])

    print(f'\n--- VERDICT ---')
    print(f'  I/O prediction: NX_ANN=2 wins on {io_wins}/{ny} output channels')
    print(f'  Physical states: NX_ANN=2 wins on {phys_wins}/{NX_PHYS} state channels')
    print(f'  Final val loss:  NX_ANN=0={losses[0]["val"][-1]:.4e}  '
          f'NX_ANN=2={losses[2]["val"][-1]:.4e}')
    if losses[2]['val'][-1] < losses[0]['val'][-1]:
        print('  PASS: NX_ANN=2 achieves lower I/O loss than NX_ANN=0.')
    else:
        print('  FAIL: NX_ANN=2 does NOT beat NX_ANN=0 on I/O loss.')

    # =================================================================
    # Save results
    # =================================================================
    json_results = dict(
        hp=HP,
        nx_ann_values=[0, 2],
        io_nrms_0={name: float(r0['io_nrms'][ch]) for ch, name in enumerate(STAGE_NAMES[:ny])},
        io_nrms_2={name: float(r2['io_nrms'][ch]) for ch, name in enumerate(STAGE_NAMES[:ny])},
        phys_nrms_0={name: float(r0['nrms_phys'][i]) for i, name in enumerate(STATE_NAMES)},
        phys_nrms_2={name: float(r2['nrms_phys'][i]) for i, name in enumerate(STATE_NAMES)},
        phys_nrms_ana={name: float(nrms_ana[i]) for i, name in enumerate(STATE_NAMES)},
        rms_0={name: float(rms_0[i]) for i, name in enumerate(STATE_NAMES)},
        rms_2={name: float(rms_2[i]) for i, name in enumerate(STATE_NAMES)},
        rms_ana={name: float(rms_ana[i]) for i, name in enumerate(STATE_NAMES)},
        final_val_loss_0=losses[0]['val'][-1],
        final_val_loss_2=losses[2]['val'][-1],
        n_params_0=r0['n_params_encoder'] + r0['n_params_interconnect'],
        n_params_2=r2['n_params_encoder'] + r2['n_params_interconnect'],
        io_wins=io_wins,
        phys_wins=phys_wins,
        vel_verification_nrms={name: float(vel_nrms[i]) for i, name in enumerate(STATE_NAMES[3:])},
    )
    json_path = os.path.join(OUT_DIR, 'encoder_io_comparison_results.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f'\nSaved: {json_path}')

    # Save trajectories
    save_dict = dict(
        x_phys_0_norm=r0['x_phys_norm'],
        x_phys_2_norm=r2['x_phys_norm'],
        x_analytical_norm=x_hat_ana,
        x_target_norm=x_target_aligned,
        y_hat_0=r0['y_hat'], y_hat_2=r2['y_hat'],
        y_target=val_windows[3],  # y_future
        train_losses_0=losses[0]['train'], val_losses_0=losses[0]['val'],
        train_losses_2=losses[2]['train'], val_losses_2=losses[2]['val'],
        nrms_phys_0=r0['nrms_phys'], nrms_phys_2=r2['nrms_phys'],
        nrms_analytical=nrms_ana,
        io_nrms_0=r0['io_nrms'], io_nrms_2=r2['io_nrms'],
        std_x=norm['std_x'], x_mean=norm['x_mean'],
        ystd=norm['ystd'], y0=norm['y0'],
        state_names=STATE_NAMES, fs=FS_NEW,
    )
    if r2['x_aug'] is not None:
        save_dict['x_aug_2'] = r2['x_aug']
    npz_path = os.path.join(OUT_DIR, 'encoder_io_comparison_data.npz')
    np.savez_compressed(npz_path, **save_dict)
    print(f'Saved: {npz_path}')

    # =================================================================
    # Plots
    # =================================================================

    # Loss comparison
    plot_loss_comparison(losses[0], losses[2],
                         os.path.join(OUT_DIR, 'encoder_io_loss_comparison.png'))

    # I/O prediction comparison
    plot_io_prediction(r0['y_hat'], r2['y_hat'], val_windows[3],
                       os.path.join(OUT_DIR, 'encoder_io_prediction.png'))

    # Physical state comparison (logical coordinates)
    plot_state_comparison(
        x0_phys, x2_phys, x_ana_phys, x_gt_phys,
        r0['nrms_phys'], r2['nrms_phys'], nrms_ana,
        STATE_NAMES, PHYS_UNITS,
        'Encoder I/O comparison (logical coords)',
        os.path.join(OUT_DIR, 'encoder_io_states_logical.png'))

    # Physical state comparison (stage coordinates)
    x0_stage = logical_to_stage(x0_phys)
    x2_stage = logical_to_stage(x2_phys)
    x_ana_stage = logical_to_stage(x_ana_phys)
    x_gt_stage = logical_to_stage(x_gt_phys)
    nrms_0_stage = compute_nrms(x0_stage, x_gt_stage)
    nrms_2_stage = compute_nrms(x2_stage, x_gt_stage)
    nrms_ana_stage = compute_nrms(x_ana_stage, x_gt_stage)
    plot_state_comparison(
        x0_stage, x2_stage, x_ana_stage, x_gt_stage,
        nrms_0_stage, nrms_2_stage, nrms_ana_stage,
        STAGE_NAMES, STAGE_UNITS,
        'Encoder I/O comparison (stage coords)',
        os.path.join(OUT_DIR, 'encoder_io_states_stage.png'))

    # NRMS bar chart
    plot_nrms_bar(r0['nrms_phys'], r2['nrms_phys'], nrms_ana, STATE_NAMES,
                  os.path.join(OUT_DIR, 'encoder_io_nrms_bar.png'))

    # Augmented states vs delta_a
    if val_delta_a is not None and r2['x_aug'] is not None and delta_a_aligned is not None:
        plot_augmented_vs_delta_a(r2['x_aug'], delta_a_aligned, ddelta_a,
                                  os.path.join(OUT_DIR, 'encoder_io_augmented.png'))

    print(f'\n{"="*70}')
    print('Encoder I/O comparison complete.')


if __name__ == '__main__':
    main()
